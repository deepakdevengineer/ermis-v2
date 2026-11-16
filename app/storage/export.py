"""Export the finished annotation (CLAUDE.md sections 17, 50, 53).

Writes the human-reviewed annotation to a local file. Three formats:

  json  full record: transcription, speaker metadata, confirmed events,
        validation findings, and the analysis events kept as QA context
  txt   the transcription alone, which is what the workflow submits
  csv   the event timeline, for reviewing detections in a spreadsheet

Two rules are enforced at the boundary rather than trusted upstream:

  - A `possible_*` event that no human confirmed is never exported as a tag.
    Unconfirmed events appear only in the analysis section, labelled pending
    (sections 46, 47).
  - Pause events carry no tag, ever. They are exported as analysis metadata
    with an explicit note, because section 13 gives V2 no pause tag.

Export is local and explicit: nothing is uploaded, and the user chooses the
path (section 53).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import APP_NAME, APP_VERSION, ERMIS_VERSION
from app.ermis.findings import ValidationReport
from app.models.event import AnalysisEvent, ConfirmationState, EventType
from app.speakers.metadata import SpeakerRegistry


class ExportError(RuntimeError):
    """Raised when an export would produce an invalid annotation."""


@dataclass
class Annotation:
    """One file's complete, human-reviewed annotation."""

    audio_path: str
    duration: float
    #: The transcription text. `source` says who produced it.
    transcription: str
    speakers: SpeakerRegistry
    events: list[AnalysisEvent]
    #: Provenance of `transcription`. 'asr_draft' means a local recogniser
    #: produced it and nobody has reviewed it yet; the export says so rather
    #: than presenting machine text as human work.
    source: str = "human"
    #: Recogniser that produced or seeded the text, when one was used.
    asr_model: str = ""
    validation: ValidationReport | None = None
    #: Free-text note from the annotator, e.g. a "Research Required" flag
    #: (section 43).
    notes: str = ""
    research_required: bool = False

    # ------------------------------------------------------------------ views
    @property
    def asr_used(self) -> bool:
        """True when a recogniser produced or seeded this text."""
        return self.source == "asr_draft" or bool(self.asr_model)

    @property
    def human_reviewed(self) -> bool:
        """False for a draft nobody has touched yet."""
        return self.source != "asr_draft"

    @property
    def confirmed_events(self) -> list[AnalysisEvent]:
        return [e for e in self.events if e.state is ConfirmationState.CONFIRMED]

    @property
    def pending_events(self) -> list[AnalysisEvent]:
        return [e for e in self.events if e.requires_human_confirmation]

    @property
    def confirmed_tags(self) -> list[str]:
        """Tags from confirmed events, in timeline order.

        Only confirmed events contribute, and a confirmed event whose chosen
        tag is None (a pause, for instance) contributes nothing.
        """
        tags = []
        for event in sorted(self.confirmed_events, key=lambda e: e.start):
            tag = event.tag
            if tag:
                tags.append(tag)
        return tags

    def to_dict(self) -> dict:
        return {
            "format": "ermis-annotation",
            "format_version": 1,
            "ermis_version": ERMIS_VERSION,
            "generator": f"{APP_NAME} {APP_VERSION}",
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "audio": {"path": self.audio_path, "duration": round(self.duration, 3)},
            "transcription": {
                # Provenance is stated explicitly so a downstream reader never
                # has to guess whether a machine wrote this text.
                "source": self.source,
                "asr_used": self.asr_used,
                "asr_model": self.asr_model or None,
                "human_reviewed": self.human_reviewed,
                "text": self.transcription,
                "research_required": self.research_required,
            },
            "speakers": self.speakers.to_dict(),
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "analysis": {
                "note": (
                    "Analysis events are internal QA metadata (CLAUDE.md section 3, "
                    "Layer A). Only events marked confirmed were accepted by a human. "
                    "Pause events never correspond to a transcription tag (section 13)."
                ),
                "confirmed": [e.to_dict() for e in self.confirmed_events],
                "pending": [e.to_dict() for e in self.pending_events],
                "all_events": [e.to_dict() for e in sorted(self.events, key=lambda e: e.start)],
            },
            "notes": self.notes,
        }

    # ------------------------------------------------------------- validation
    def check_exportable(
        self, require_valid: bool = True, allow_unreviewed: bool = False
    ) -> None:
        """Refuse to export an annotation that is not fit to submit."""
        if not allow_unreviewed and self.source == "asr_draft":
            raise ExportError(
                "this transcription is an unreviewed machine draft. Listen to "
                "the audio and correct it first, or export with "
                "allow_unreviewed=True to keep a labelled work-in-progress copy."
            )
        if not self.transcription.strip():
            raise ExportError(
                "transcription is empty. A clip needs transcription text or a "
                "whole-file tag such as <na> or <ns> (section 42)."
            )
        if require_valid and self.validation is not None and not self.validation.is_valid:
            errors = "; ".join(f.message for f in self.validation.errors[:3])
            raise ExportError(
                f"transcription has {len(self.validation.errors)} validation "
                f"error(s): {errors}. Fix them, or export with require_valid=False "
                "to keep a work-in-progress copy."
            )
        for event in self.events:
            if event.state is ConfirmationState.CONFIRMED and event.confirmed_tag:
                if event.type is EventType.PAUSE:
                    raise ExportError(
                        "a pause event carries a tag. V2 has no pause tag "
                        "(section 13)."
                    )


class Exporter:
    """Writes annotations to local files."""

    def export_json(
        self,
        annotation: Annotation,
        path: str | Path,
        require_valid: bool = True,
        allow_unreviewed: bool = False,
    ) -> Path:
        annotation.check_exportable(require_valid, allow_unreviewed)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(annotation.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def export_text(
        self,
        annotation: Annotation,
        path: str | Path,
        require_valid: bool = True,
        allow_unreviewed: bool = False,
    ) -> Path:
        """Write the transcription alone -- what the workflow submits."""
        annotation.check_exportable(require_valid, allow_unreviewed)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # No trailing newline games: the transcription is a single line, and
        # section 41 forbids stray leading or trailing space.
        target.write_text(annotation.transcription.strip() + "\n", encoding="utf-8")
        return target

    def export_events_csv(self, annotation: Annotation, path: str | Path) -> Path:
        """Write the event timeline for spreadsheet review."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "start", "end", "duration", "type", "confidence",
                    "speaker", "state", "confirmed_tag", "note",
                ]
            )
            for event in sorted(annotation.events, key=lambda e: e.start):
                writer.writerow(
                    [
                        f"{event.start:.3f}",
                        f"{event.end:.3f}",
                        f"{event.duration:.3f}",
                        event.type.value,
                        f"{event.confidence:.3f}",
                        event.speaker_id or "",
                        event.state.value,
                        event.confirmed_tag or "",
                        event.note,
                    ]
                )
        return target

    def export_all(
        self,
        annotation: Annotation,
        directory: str | Path,
        stem: str | None = None,
        require_valid: bool = True,
        allow_unreviewed: bool = False,
    ) -> dict[str, Path]:
        """Write all three formats into `directory`."""
        base = Path(directory)
        name = stem or Path(annotation.audio_path).stem or "annotation"
        return {
            "json": self.export_json(
                annotation, base / f"{name}.json", require_valid, allow_unreviewed
            ),
            "txt": self.export_text(
                annotation, base / f"{name}.txt", require_valid, allow_unreviewed
            ),
            "csv": self.export_events_csv(annotation, base / f"{name}.events.csv"),
        }


def export_annotation(
    annotation: Annotation,
    directory: str | Path,
    stem: str | None = None,
    require_valid: bool = True,
) -> dict[str, Path]:
    return Exporter().export_all(annotation, directory, stem, require_valid)
