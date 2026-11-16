"""Audio-to-transcription cross-check (CLAUDE.md sections 13, 47, 49).

Section 49 specifies both findings by example. The critical property tested
here is the negative one: an *unconfirmed* detection must never be reported as
missing, or the panel would pressure the annotator into writing a tag for
something they have not agreed is there.
"""

from __future__ import annotations

import pytest

from app.config import ValidationConfig
from app.ermis.crosscheck import apply_cross_check, cross_check
from app.ermis.findings import Code, Severity
from app.ermis.validator import validate
from app.models.event import AnalysisEvent, EventType


def _confirmed(event_type: EventType, start: float = 1.0, tag: str | None = None):
    event = AnalysisEvent(event_type, start, start + 0.5, 0.7)
    event.confirm(tag)
    return event


class TestMissingTagWarning:
    def test_confirmed_event_without_its_tag_warns(self):
        findings = cross_check("play music", [_confirmed(EventType.POSSIBLE_LAUGHTER)])
        assert len(findings) == 1
        assert findings[0].code is Code.EVENT_NOT_TRANSCRIBED
        assert findings[0].severity is Severity.WARNING

    def test_message_names_the_tag_and_time(self):
        findings = cross_check(
            "play music", [_confirmed(EventType.POSSIBLE_LAUGHTER, start=12.4)]
        )
        message = findings[0].message
        assert "[laughter]" in message
        assert "00:12.40" in message

    def test_no_warning_when_the_tag_is_present(self):
        assert (
            cross_check(
                "play [laughter] music", [_confirmed(EventType.POSSIBLE_LAUGHTER)]
            )
            == []
        )

    @pytest.mark.parametrize(
        "event_type,tag",
        [
            (EventType.POSSIBLE_LAUGHTER, "[laughter]"),
            (EventType.POSSIBLE_HN, "[hn]"),
            (EventType.POSSIBLE_ARTIFACT, "[artifact]"),
            (EventType.POSSIBLE_BACKGROUND, "[bg]"),
        ],
    )
    def test_every_taggable_type(self, event_type, tag):
        assert cross_check("play music", [_confirmed(event_type)])
        assert cross_check(f"play {tag} music", [_confirmed(event_type)]) == []

    def test_each_event_consumes_one_tag(self):
        # Three confirmed coughs are not satisfied by a single [hn].
        events = [
            _confirmed(EventType.POSSIBLE_HN, 1.0),
            _confirmed(EventType.POSSIBLE_HN, 4.0),
            _confirmed(EventType.POSSIBLE_HN, 7.0),
        ]
        assert len(cross_check("a [hn] b", events)) == 2
        assert len(cross_check("a [hn] b [hn] c [hn] d", events)) == 0

    def test_override_is_checked_against_the_chosen_tag(self):
        # The human changed a laughter detection to [hn]; [hn] is what should
        # appear, not [laughter].
        event = _confirmed(EventType.POSSIBLE_LAUGHTER, tag="[hn]")
        assert cross_check("play [hn] music", [event]) == []
        assert cross_check("play [laughter] music", [event])

    def test_suggestion_is_the_tag(self):
        findings = cross_check("play music", [_confirmed(EventType.POSSIBLE_LAUGHTER)])
        assert findings[0].suggestion == "[laughter]"

    def test_findings_are_ordered_by_time(self):
        events = [
            _confirmed(EventType.POSSIBLE_HN, 9.0),
            _confirmed(EventType.POSSIBLE_LAUGHTER, 2.0),
        ]
        findings = cross_check("play music", events)
        assert "00:02.00" in findings[0].message
        assert "00:09.00" in findings[1].message


class TestUnconfirmedIsNeverReported:
    """Sections 47 and 56.22 -- the load-bearing negative property."""

    def test_pending_event_produces_no_warning(self):
        pending = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.95)
        assert pending.requires_human_confirmation
        assert cross_check("play music", [pending]) == []

    def test_high_confidence_pending_event_still_silent(self):
        pending = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.99)
        assert cross_check("play music", [pending]) == []

    def test_rejected_event_produces_no_warning(self):
        rejected = AnalysisEvent(EventType.POSSIBLE_ARTIFACT, 1.0, 1.1, 0.8)
        rejected.reject("just a door")
        assert cross_check("play music", [rejected]) == []

    def test_confirmation_is_what_enables_the_check(self):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.7)
        assert cross_check("play music", [event]) == []
        event.confirm()
        assert cross_check("play music", [event])


class TestPauseInfo:
    def test_pause_reported_as_info(self):
        findings = cross_check("play music", [AnalysisEvent(EventType.PAUSE, 3.42, 5.22)])
        assert len(findings) == 1
        assert findings[0].code is Code.PAUSE_DETECTED
        assert findings[0].severity is Severity.INFO

    def test_message_matches_section_49(self):
        findings = cross_check("play music", [AnalysisEvent(EventType.PAUSE, 3.42, 5.22)])
        message = findings[0].message
        assert "1.8 second pause" in message
        assert "internal metadata" in message
        assert "should not be inserted" in message

    def test_pause_is_never_reported_as_missing(self):
        # A pause cannot be "missing": V2 has no pause tag (section 13).
        pause = AnalysisEvent(EventType.PAUSE, 1.0, 2.0)
        pause.confirm()
        findings = cross_check("play music", [pause])
        assert all(f.code is not Code.EVENT_NOT_TRANSCRIBED for f in findings)

    def test_silence_is_not_reported(self):
        assert cross_check("play music", [AnalysisEvent(EventType.SILENCE, 0.0, 0.5)]) == []

    def test_speech_is_not_reported(self):
        assert cross_check("play music", [AnalysisEvent(EventType.SPEECH, 0.0, 1.0)]) == []


class TestConfiguration:
    def test_warnings_can_be_disabled(self):
        config = ValidationConfig()
        config.report_audio_dependent_warnings = False
        findings = cross_check(
            "play music", [_confirmed(EventType.POSSIBLE_LAUGHTER)], config
        )
        assert findings == []

    def test_pause_info_can_be_disabled(self):
        config = ValidationConfig()
        config.report_pause_info = False
        findings = cross_check(
            "play music", [AnalysisEvent(EventType.PAUSE, 1.0, 2.0)], config
        )
        assert findings == []


class TestReportIntegration:
    def test_apply_adds_to_an_existing_report(self):
        report = validate("play music")
        assert not report.findings
        apply_cross_check(
            report,
            [
                _confirmed(EventType.POSSIBLE_LAUGHTER),
                AnalysisEvent(EventType.PAUSE, 3.0, 4.0),
            ],
        )
        assert report.has(Code.EVENT_NOT_TRANSCRIBED)
        assert report.has(Code.PAUSE_DETECTED)

    def test_cross_check_never_invalidates(self):
        # The annotator's ears outrank the detector.
        report = apply_cross_check(
            validate("play music"), [_confirmed(EventType.POSSIBLE_LAUGHTER)]
        )
        assert report.is_valid
        assert report.warnings

    def test_returns_the_same_report(self):
        report = validate("play music")
        assert apply_cross_check(report, []) is report

    def test_combines_with_ordinary_findings(self):
        report = apply_cross_check(
            validate("Play music?"), [_confirmed(EventType.POSSIBLE_LAUGHTER)]
        )
        assert report.errors  # uppercase and punctuation
        assert report.warnings  # the cross-check
        assert not report.is_valid

    def test_serializes(self):
        report = apply_cross_check(
            validate("play music"),
            [
                _confirmed(EventType.POSSIBLE_HN),
                AnalysisEvent(EventType.PAUSE, 1.0, 2.0),
            ],
        )
        payload = report.to_dict()
        codes = {f["code"] for f in payload["findings"]}
        assert "event-not-transcribed" in codes
        assert "pause-detected" in codes


class TestPipelineIntegration:
    def test_real_analysis_produces_pause_info(self, wav_two_speakers):
        from app.audio.loader import load_audio
        from app.detection.pipeline import AnalysisPipeline

        result = AnalysisPipeline().run(load_audio(wav_two_speakers))
        report = apply_cross_check(validate("<s1> a <s2> b"), result.events)
        assert report.has(Code.PAUSE_DETECTED)

    def test_real_analysis_warns_nothing_before_confirmation(self, wav_laughter):
        from app.audio.loader import load_audio
        from app.detection.pipeline import AnalysisPipeline

        result = AnalysisPipeline().run(load_audio(wav_laughter))
        report = apply_cross_check(validate("play music"), result.events)
        # Candidates are all still pending, so nothing is "missing" yet.
        assert not report.has(Code.EVENT_NOT_TRANSCRIBED)
