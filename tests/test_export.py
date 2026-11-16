"""Export (CLAUDE.md sections 17, 42, 47, 50, 53).

The export boundary is the last place the workflow rules can be enforced, so it
re-checks them rather than trusting upstream: no unconfirmed detection becomes
a tag, no pause carries a tag, and no ASR-sourced text can be written.
"""

from __future__ import annotations

import json

import pytest

from app.ermis.validator import validate
from app.models.event import AnalysisEvent, EventType
from app.speakers.metadata import SpeakerRegistry
from app.storage.database import SpeakerProfile
from app.storage.export import Annotation, Exporter, ExportError, export_annotation


def _registry(count: int = 2) -> SpeakerRegistry:
    registry = SpeakerRegistry()
    for index in range(1, count + 1):
        registry.profiles[f"s{index}"] = SpeakerProfile(
            speaker_id=f"s{index}",
            display_name=f"Speaker {index}",
            gender="msv" if index == 1 else "fsv",
            nativity="native",
            first_seen=float(index),
            last_seen=float(index) + 1.0,
            confidence=0.8,
            human_edited=True,
        )
        registry.segments.append((float(index), float(index) + 1.0, f"s{index}"))
    return registry


def _annotation(text: str = "<s1> play music <s2> ok", **kwargs) -> Annotation:
    defaults = dict(
        audio_path="clip.wav",
        duration=5.0,
        transcription=text,
        speakers=_registry(),
        events=[],
        validation=validate(text, speaker_count=2),
    )
    defaults.update(kwargs)
    return Annotation(**defaults)


class TestJsonExport:
    def test_writes_a_file(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        assert path.is_file()

    def test_structure(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == "ermis-annotation"
        assert payload["ermis_version"] == "V2"
        assert payload["transcription"]["text"] == "<s1> play music <s2> ok"

    def test_records_provenance_of_human_text(self, tmp_path):
        # Provenance is stated in the artefact so a downstream reader never has
        # to guess whether a machine wrote the text.
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["transcription"]["source"] == "human"
        assert payload["transcription"]["asr_used"] is False
        assert payload["transcription"]["human_reviewed"] is True

    def test_records_provenance_of_a_reviewed_draft(self, tmp_path):
        annotation = _annotation()
        annotation.source = "human_edited"
        annotation.asr_model = "openai/whisper-small.en"
        path = Exporter().export_json(annotation, tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["transcription"]["asr_used"] is True
        assert payload["transcription"]["asr_model"] == "openai/whisper-small.en"
        assert payload["transcription"]["human_reviewed"] is True

    def test_unreviewed_draft_is_refused_by_default(self, tmp_path):
        annotation = _annotation()
        annotation.source = "asr_draft"
        with pytest.raises(ExportError, match="unreviewed machine draft"):
            Exporter().export_json(annotation, tmp_path / "out.json")

    def test_unreviewed_draft_can_be_exported_when_asked_explicitly(self, tmp_path):
        annotation = _annotation()
        annotation.source = "asr_draft"
        path = Exporter().export_json(
            annotation, tmp_path / "out.json", allow_unreviewed=True
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["transcription"]["source"] == "asr_draft"
        assert payload["transcription"]["human_reviewed"] is False

    def test_includes_speaker_metadata(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["speakers"]["total_speakers"] == 2
        assert payload["speakers"]["speakers"][0]["gender"] == "msv"

    def test_includes_validation(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["validation"]["is_valid"] is True

    def test_analysis_section_explains_itself(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        note = payload["analysis"]["note"]
        assert "internal QA metadata" in note
        assert "section 13" in note

    def test_creates_missing_directories(self, tmp_path):
        path = Exporter().export_json(
            _annotation(), tmp_path / "nested" / "deep" / "out.json"
        )
        assert path.is_file()


class TestTextExport:
    def test_writes_the_transcription_alone(self, tmp_path):
        path = Exporter().export_text(_annotation(), tmp_path / "out.txt")
        assert path.read_text(encoding="utf-8") == "<s1> play music <s2> ok\n"

    def test_strips_surrounding_whitespace(self, tmp_path):
        annotation = _annotation("  play music  ")
        annotation.validation = None
        path = Exporter().export_text(annotation, tmp_path / "out.txt")
        # Section 41 forbids leading and trailing space.
        assert path.read_text(encoding="utf-8") == "play music\n"

    def test_whole_file_tag_exports_alone(self, tmp_path):
        path = Exporter().export_text(_annotation("<na>"), tmp_path / "out.txt")
        assert path.read_text(encoding="utf-8") == "<na>\n"


class TestCsvExport:
    def test_writes_the_event_timeline(self, tmp_path):
        events = [
            AnalysisEvent(EventType.SPEECH, 0.0, 1.0, speaker_id="s1"),
            AnalysisEvent(EventType.PAUSE, 1.0, 1.8),
            AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 2.0, 2.5, 0.6),
        ]
        path = Exporter().export_events_csv(
            _annotation(events=events), tmp_path / "events.csv"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("start,end,duration,type")
        assert len(lines) == 4

    def test_pending_state_is_visible(self, tmp_path):
        events = [AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.5)]
        path = Exporter().export_events_csv(
            _annotation(events=events), tmp_path / "events.csv"
        )
        assert "pending" in path.read_text(encoding="utf-8")

    def test_empty_event_list(self, tmp_path):
        path = Exporter().export_events_csv(_annotation(), tmp_path / "events.csv")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1


class TestConfirmationBoundary:
    """Only confirmed events contribute tags (sections 46, 47)."""

    def test_pending_event_contributes_no_tag(self):
        pending = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.9)
        annotation = _annotation(events=[pending])
        assert annotation.confirmed_tags == []
        assert annotation.pending_events == [pending]

    def test_confirmed_event_contributes_its_tag(self):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.9)
        event.confirm()
        assert _annotation(events=[event]).confirmed_tags == ["[laughter]"]

    def test_rejected_event_contributes_nothing(self):
        event = AnalysisEvent(EventType.POSSIBLE_ARTIFACT, 1.0, 1.1, 0.7)
        event.reject()
        assert _annotation(events=[event]).confirmed_tags == []

    def test_override_is_exported(self):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.95)
        event.confirm("[hn]")  # the human disagreed with the detector
        assert _annotation(events=[event]).confirmed_tags == ["[hn]"]

    def test_confirmed_pause_contributes_no_tag(self):
        # Section 13: V2 has no pause tag.
        event = AnalysisEvent(EventType.PAUSE, 1.0, 2.0)
        event.confirm()
        assert _annotation(events=[event]).confirmed_tags == []

    def test_tags_are_in_timeline_order(self):
        late = AnalysisEvent(EventType.POSSIBLE_HN, 5.0, 5.2, 0.6)
        early = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 1.5, 0.6)
        late.confirm()
        early.confirm()
        assert _annotation(events=[late, early]).confirmed_tags == [
            "[laughter]",
            "[hn]",
        ]


class TestExportGuards:
    def test_empty_transcription_refused(self, tmp_path):
        annotation = _annotation("")
        annotation.validation = None
        with pytest.raises(ExportError, match="empty"):
            Exporter().export_json(annotation, tmp_path / "out.json")

    def test_invalid_transcription_refused_by_default(self, tmp_path):
        annotation = _annotation("Play music?")
        with pytest.raises(ExportError, match="validation error"):
            Exporter().export_json(annotation, tmp_path / "out.json")

    def test_invalid_can_be_exported_as_work_in_progress(self, tmp_path):
        annotation = _annotation("Play music?")
        path = Exporter().export_json(
            annotation, tmp_path / "out.json", require_valid=False
        )
        assert path.is_file()

    def test_error_message_names_the_problems(self, tmp_path):
        annotation = _annotation("Play music?")
        with pytest.raises(ExportError) as exc:
            Exporter().export_json(annotation, tmp_path / "out.json")
        assert "uppercase" in str(exc.value).lower() or "lowercase" in str(exc.value)

    def test_warnings_do_not_block_export(self, tmp_path):
        # A warning marks something only the audio can settle, not an error.
        annotation = _annotation("dont stop")
        assert annotation.validation.warnings
        assert annotation.validation.is_valid
        assert Exporter().export_json(annotation, tmp_path / "out.json").is_file()


class TestExportAll:
    def test_writes_three_files(self, tmp_path):
        written = export_annotation(_annotation(), tmp_path)
        assert set(written) == {"json", "txt", "csv"}
        for path in written.values():
            assert path.is_file()

    def test_stem_comes_from_the_audio_name(self, tmp_path):
        annotation = _annotation()
        annotation.audio_path = "interview_042.wav"
        written = export_annotation(annotation, tmp_path)
        assert written["json"].name == "interview_042.json"

    def test_explicit_stem(self, tmp_path):
        written = export_annotation(_annotation(), tmp_path, stem="custom")
        assert written["txt"].name == "custom.txt"

    def test_research_flag_is_carried(self, tmp_path):
        # Section 43: a manual "Research Required" flag rather than silently
        # changing an uncertain word.
        annotation = _annotation(research_required=True, notes="verify the band name")
        path = Exporter().export_json(annotation, tmp_path / "out.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["transcription"]["research_required"] is True
        assert payload["notes"] == "verify the band name"


class TestPrivacy:
    def test_export_writes_only_where_told(self, tmp_path, monkeypatch):
        # Section 53: local, explicit, no upload.
        import socket

        def fail(*args, **kwargs):
            raise AssertionError("export must not open a socket")

        monkeypatch.setattr(socket, "socket", fail)
        monkeypatch.setattr(socket, "create_connection", fail)
        written = export_annotation(_annotation(), tmp_path)
        for path in written.values():
            assert tmp_path in path.parents

    def test_export_contains_no_credentials(self, tmp_path):
        path = Exporter().export_json(_annotation(), tmp_path / "out.json")
        content = path.read_text(encoding="utf-8").lower()
        for word in ("password", "secret", "api_key", "token", "credential"):
            assert word not in content
