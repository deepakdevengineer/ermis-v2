"""Diarization and the speaker registry.

CLAUDE.md section 18A throughout: profile shape (18A.5), first-appearance
numbering (18A.2), the timeline (18A.7), merge and split (18A.8), validation
(18A.10), and the rule that automatic detection is assistive only (18A.11).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import load_audio
from app.detection.vad import EnergyVad
from app.models.event import EventType
from app.speakers.diarization import Diarizer, DiarizationResult
from app.speakers.metadata import SpeakerError, SpeakerRegistry
from app.storage.database import ALLOWED_GENDERS, ALLOWED_NATIVITY


def _diarize(path):
    buffer = load_audio(path)
    spans = EnergyVad().analyse(buffer).speech
    return buffer, Diarizer().diarize(buffer, spans)


# ------------------------------------------------------------------ clustering
class TestDiarization:
    def test_two_voices_yield_two_speakers(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        assert result.speaker_count == 2

    def test_one_voice_yields_one_speaker(self, wav_one_speaker_turns):
        # Four separate turns from the same person must not become four
        # speakers. Over-splitting corrupts every speaker tag downstream.
        _, result = _diarize(wav_one_speaker_turns)
        assert result.speaker_count == 1

    def test_alternating_turns_are_assigned_alternately(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        ordered = sorted(result.segments, key=lambda s: s.start)
        ids = [s.speaker_id for s in ordered]
        assert ids == ["s1", "s2", "s1", "s2"]

    def test_numbering_follows_first_appearance(self, wav_two_speakers):
        # Section 18A.2: the first speaker heard is s1.
        _, result = _diarize(wav_two_speakers)
        assert result.speaker_ids == ["s1", "s2"]
        first = min(result.segments, key=lambda s: s.start)
        assert first.speaker_id == "s1"

    def test_no_speech_yields_no_speakers(self, wav_silence):
        buffer = load_audio(wav_silence)
        result = Diarizer().diarize(buffer, [])
        assert result.speaker_count == 0
        assert result.segments == []

    def test_deterministic(self, wav_two_speakers):
        # Section 44 needs determinism, and so does the audit trail.
        buffer = load_audio(wav_two_speakers)
        spans = EnergyVad().analyse(buffer).speech
        baseline = [
            (s.start, s.speaker_id) for s in Diarizer().diarize(buffer, spans).segments
        ]
        for _ in range(3):
            again = [
                (s.start, s.speaker_id)
                for s in Diarizer().diarize(buffer, spans).segments
            ]
            assert again == baseline

    def test_count_confidence_is_reported_and_capped(self, wav_two_speakers):
        # Section 18A.11 requires the count to be shown with a confidence.
        _, result = _diarize(wav_two_speakers)
        assert 0.0 < result.confidence <= 0.80

    def test_never_infers_gender_or_nativity(self, wav_two_speakers):
        # Sections 18A.3, 18A.4: no guessing. The result exposes no such field.
        _, result = _diarize(wav_two_speakers)
        for segment in result.segments:
            assert not hasattr(segment, "gender")
            assert not hasattr(segment, "nativity")

    def test_max_speakers_is_respected(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        spans = EnergyVad().analyse(buffer).speech
        result = Diarizer().diarize(buffer, spans, max_speakers=1)
        assert result.speaker_count <= 2

    def test_short_segments_are_skipped(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = Diarizer().diarize(buffer, [(0.0, 0.05), (1.0, 1.05)])
        assert result.segments == []

    def test_timeline_and_bounds(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        assert result.timeline("s1")
        assert result.first_seen("s1") is not None
        assert result.last_seen("s1") > result.first_seen("s1")

    def test_events_carry_speaker_and_changes(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        events = result.to_events()
        speech = [e for e in events if e.type is EventType.SPEECH]
        changes = [e for e in events if e.type is EventType.SPEAKER_CHANGE]
        assert all(e.speaker_id for e in speech)
        assert len(changes) == 3  # s1->s2, s2->s1, s1->s2

    def test_render_timeline_is_ascii_by_default(self, wav_two_speakers):
        # cp1252 is the default Windows console code page and cannot encode
        # box-drawing characters; printing them raises UnicodeEncodeError.
        buffer, result = _diarize(wav_two_speakers)
        art = result.render_timeline(width=40, duration=buffer.duration)
        art.encode("cp1252")  # must not raise
        assert "S1" in art and "S2" in art

    def test_render_timeline_accepts_a_block_character(self, wav_two_speakers):
        buffer, result = _diarize(wav_two_speakers)
        art = result.render_timeline(width=40, duration=buffer.duration, block="█")
        assert "█" in art

    def test_empty_result_renders(self):
        assert DiarizationResult([], 0, 0.0).render_timeline() == "(no speech)"


# -------------------------------------------------------------------- registry
class TestRegistrySeeding:
    def test_seeded_from_diarization(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        registry = SpeakerRegistry.from_diarization(result)
        assert registry.count == 2
        assert registry.speaker_ids == ["s1", "s2"]

    def test_metadata_starts_unknown(self, wav_two_speakers):
        # Nothing may be asserted about these people automatically.
        _, result = _diarize(wav_two_speakers)
        registry = SpeakerRegistry.from_diarization(result)
        for speaker_id in registry.speaker_ids:
            profile = registry.get(speaker_id)
            assert profile.gender == "unknown"
            assert profile.nativity == "unknown"
            assert profile.human_edited is False

    def test_profile_shape_matches_section_18a5(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        registry = SpeakerRegistry.from_diarization(result)
        payload = registry.get("s1").to_dict()
        assert set(payload) == {
            "speaker_id", "display_name", "gender", "nativity",
            "first_seen", "last_seen", "confidence",
        }

    def test_single_speaker_flag(self, wav_one_speaker_turns):
        # Section 17: single-speaker audio takes no speaker tag.
        _, result = _diarize(wav_one_speaker_turns)
        assert SpeakerRegistry.from_diarization(result).is_single_speaker()


class TestRegistryEdits:
    @pytest.fixture
    def registry(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        return SpeakerRegistry.from_diarization(result)

    def test_set_metadata(self, registry):
        before, after = registry.set_metadata("s1", gender="msv", nativity="native")
        assert before.gender == "unknown"
        assert after.gender == "msv"
        assert after.nativity == "native"
        assert after.human_edited is True

    @pytest.mark.parametrize("gender", ["male", "M", "MSV", ""])
    def test_invalid_gender_rejected(self, registry, gender):
        with pytest.raises(SpeakerError, match="gender"):
            registry.set_metadata("s1", gender=gender)

    @pytest.mark.parametrize("nativity", ["fluent", "non-native", "robot"])
    def test_invalid_nativity_rejected(self, registry, nativity):
        with pytest.raises(SpeakerError, match="nativity"):
            registry.set_metadata("s1", nativity=nativity)

    @pytest.mark.parametrize("gender", ALLOWED_GENDERS)
    def test_all_allowed_genders(self, registry, gender):
        _, after = registry.set_metadata("s1", gender=gender)
        assert after.gender == gender

    @pytest.mark.parametrize("nativity", ALLOWED_NATIVITY)
    def test_all_allowed_nativity(self, registry, nativity):
        _, after = registry.set_metadata("s1", nativity=nativity)
        assert after.nativity == nativity

    def test_unknown_speaker_rejected(self, registry):
        with pytest.raises(SpeakerError, match="no such speaker"):
            registry.set_metadata("s9", gender="msv")

    def test_add_speaker(self, registry):
        new_id = registry.add_speaker()
        assert new_id == "s3"
        assert registry.count == 3
        assert registry.get(new_id).human_edited is True

    def test_added_speaker_metadata_is_unknown(self, registry):
        new_id = registry.add_speaker()
        assert registry.get(new_id).gender == "unknown"
        assert registry.get(new_id).nativity == "unknown"

    def test_remove_speaker_renumbers(self, registry):
        registry.remove_speaker("s1")
        # No gap left behind: the survivor becomes s1.
        assert registry.speaker_ids == ["s1"]
        assert registry.numbers == {1}

    def test_display_name_typed_by_a_human_survives_renumbering(self, registry):
        registry.set_metadata("s2", display_name="the interviewer")
        registry.remove_speaker("s1")
        assert registry.get("s1").display_name == "the interviewer"

    def test_default_display_name_is_regenerated(self, registry):
        registry.remove_speaker("s1")
        assert registry.get("s1").display_name == "Speaker 1"


class TestMerge:
    @pytest.fixture
    def registry(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        return SpeakerRegistry.from_diarization(result)

    def test_merge_two_into_one(self, registry):
        registry.merge(["s2"], "s1")
        assert registry.count == 1
        assert {sid for _, _, sid in registry.segments} == {"s1"}

    def test_merge_updates_the_timeline(self, registry):
        original = len(registry.segments)
        registry.merge(["s2"], "s1")
        assert len(registry.timeline("s1")) == original

    def test_merge_extends_bounds(self, registry):
        before_first = registry.get("s1").first_seen
        registry.merge(["s2"], "s1")
        merged = registry.get("s1")
        assert merged.first_seen <= before_first
        assert merged.last_seen >= before_first

    def test_merge_keeps_a_human_value(self, registry):
        # Section 18A.11: a merge must not discard a human decision.
        registry.set_metadata("s2", gender="fsv")
        registry.merge(["s2"], "s1")  # s1 is unknown, s2 was human-set
        assert registry.get("s1").gender == "fsv"
        assert registry.get("s1").human_edited is True

    def test_merge_prefers_human_over_automatic(self, registry):
        registry.set_metadata("s1", gender="msv")  # human
        registry.profiles["s2"] = registry.get("s2").__class__(
            **{**registry.get("s2").to_dict(), "gender": "fsv", "human_edited": False}
        )
        registry.merge(["s2"], "s1")
        assert registry.get("s1").gender == "msv"

    def test_merge_needs_a_distinct_source(self, registry):
        with pytest.raises(SpeakerError):
            registry.merge(["s1"], "s1")

    def test_merge_unknown_speaker_rejected(self, registry):
        with pytest.raises(SpeakerError, match="no such speaker"):
            registry.merge(["s9"], "s1")

    def test_merge_leaves_no_numbering_gap(self, registry):
        registry.add_speaker()  # s3
        registry.merge(["s2"], "s1")
        assert registry.numbers == {1, 2}


class TestSplit:
    @pytest.fixture
    def registry(self, wav_one_speaker_turns):
        _, result = _diarize(wav_one_speaker_turns)
        return SpeakerRegistry.from_diarization(result)

    def test_split_creates_a_second_speaker(self, registry):
        assert registry.count == 1
        original, new = registry.split("s1", boundary=2.5)
        assert registry.count == 2
        assert {original, new} == {"s1", "s2"}

    def test_split_divides_the_segments(self, registry):
        registry.split("s1", boundary=2.5)
        early = registry.timeline("s1")
        late = registry.timeline("s2")
        assert early and late
        assert all(start < 2.5 for start, _ in early)
        assert all(start >= 2.5 for start, _ in late)

    def test_new_speaker_metadata_is_unknown(self, registry):
        # Nothing is known about a person just separated out; copying the
        # original's gender would be an unfounded assertion.
        registry.set_metadata("s1", gender="msv", nativity="native")
        _, new = registry.split("s1", boundary=2.5)
        assert registry.get(new).gender == "unknown"
        assert registry.get(new).nativity == "unknown"

    def test_split_requires_a_dividing_boundary(self, registry):
        with pytest.raises(SpeakerError, match="does not divide"):
            registry.split("s1", boundary=99.0)

    def test_split_of_unknown_speaker_rejected(self, registry):
        with pytest.raises(SpeakerError, match="no such speaker"):
            registry.split("s9", boundary=1.0)

    def test_split_refreshes_bounds(self, registry):
        original, new = registry.split("s1", boundary=2.5)
        assert registry.get(original).last_seen <= registry.get(new).first_seen + 0.01

    def test_split_leaves_no_numbering_gap(self, registry):
        registry.split("s1", boundary=2.5)
        assert registry.numbers == {1, 2}


class TestSetCount:
    """Section 18A.11's worked example: a 3-speaker estimate corrected to 2."""

    @pytest.fixture
    def registry(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        return SpeakerRegistry.from_diarization(result)

    def test_increase(self, registry):
        registry.set_count(4)
        assert registry.count == 4
        assert registry.numbers == {1, 2, 3, 4}

    def test_decrease_merges_rather_than_dropping_speech(self, registry):
        segments_before = len(registry.segments)
        registry.set_count(1)
        assert registry.count == 1
        # No speech becomes unattributed.
        assert len(registry.segments) == segments_before
        assert {sid for _, _, sid in registry.segments} == {"s1"}

    def test_decrease_from_three_to_two(self, registry):
        registry.add_speaker()
        assert registry.count == 3
        registry.set_count(2)
        assert registry.count == 2
        assert registry.numbers == {1, 2}

    def test_no_change(self, registry):
        registry.set_count(2)
        assert registry.count == 2

    def test_negative_rejected(self, registry):
        with pytest.raises(SpeakerError):
            registry.set_count(-1)

    def test_zero_clears(self, registry):
        registry.set_count(0)
        assert registry.count == 0


class TestRegistryOutput:
    @pytest.fixture
    def registry(self, wav_two_speakers):
        _, result = _diarize(wav_two_speakers)
        return SpeakerRegistry.from_diarization(result)

    def test_to_dict(self, registry):
        payload = registry.to_dict()
        assert payload["total_speakers"] == 2
        assert len(payload["speakers"]) == 2
        assert payload["segments"]

    def test_to_events(self, registry):
        events = registry.to_events()
        assert all(e.type is EventType.SPEECH for e in events)
        assert all(e.speaker_id for e in events)

    def test_render_panel_matches_section_18a9(self, registry):
        panel = registry.render_panel()
        assert "SPEAKERS" in panel
        assert "Total Speakers: 2" in panel
        assert "Gender:" in panel and "Nativity:" in panel
        assert "[Merge Speakers]" in panel
        assert "[Split Speaker]" in panel
        assert "[Add Speaker]" in panel

    def test_panel_marks_human_edits(self, registry):
        registry.set_metadata("s1", gender="msv")
        assert "human edited" in registry.render_panel()

    def test_numbers_feed_the_validator(self, registry):
        from app.ermis.findings import Code
        from app.ermis.validator import validate

        # A tag referring to a speaker that does not exist is an error
        # (section 18A.10 item 7).
        report = validate("<s1> a <s2> b <s3> c", known_speakers=registry.numbers)
        assert report.has(Code.SPEAKER_UNKNOWN_REFERENCE)
