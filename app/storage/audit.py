"""Audit trail (CLAUDE.md section 50).

Records what was detected, what a human confirmed or rejected, and how the
transcription changed. The table is append-only at the database level, so the
trail cannot be rewritten from application code.

The audit log holds annotation decisions only. It never records credentials,
and it stores audio *offsets*, not audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ermis.findings import ValidationReport
from app.models.event import AnalysisEvent
from app.storage.database import Database, SpeakerProfile


class Action:
    """Audit action vocabulary."""

    AUDIO_LOADED = "audio_loaded"
    RECORDING_CAPTURED = "recording_captured"
    #: A local recogniser produced draft text (the section 1 / 56.1 deviation).
    DRAFT_CREATED = "asr_draft_created"
    #: A tag reached the draft from a detection without a human clicking
    #: Confirm. Recorded separately from EVENT_CONFIRMED so the audit trail
    #: never claims a person made that call (section 47).
    EVENT_AUTO_CONFIRMED = "auto_confirmed_event"
    ANALYSIS_RUN = "analysis_run"
    EVENT_DETECTED = "event_detected"
    EVENT_CONFIRMED = "confirmed_event"
    EVENT_REJECTED = "rejected_event"
    EVENT_CHANGED = "changed_event"
    TRANSCRIPTION_SAVED = "transcription_saved"
    VALIDATION_RUN = "validation_run"
    SPEAKER_METADATA_CHANGED = "speaker_metadata_changed"
    SPEAKERS_MERGED = "speakers_merged"
    SPEAKER_SPLIT = "speaker_split"
    EXPORTED = "exported"


@dataclass
class AuditLog:
    """Convenience writer over the audit_logs table."""

    db: Database

    def audio_loaded(self, audio_file_id: int, path: str, duration: float) -> int:
        return self.db.append_audit(
            Action.AUDIO_LOADED,
            audio_file_id=audio_file_id,
            detail={"path": path, "duration": round(duration, 3)},
        )

    def recording_captured(self, audio_file_id: int, result) -> int:
        """Record that this clip came from the microphone, not from a file.

        Provenance matters when the annotation is reviewed later: it says the
        audio was captured on this machine, at what rate, and what the capture
        measured (clipping, silence) before anyone listened to it.
        """
        return self.db.append_audit(
            Action.RECORDING_CAPTURED,
            audio_file_id=audio_file_id,
            audio_start=0.0,
            audio_end=result.duration,
            detail=result.to_dict() | {"notes": result.notes()},
        )

    def event_detected(self, audio_file_id: int, event: AnalysisEvent) -> int:
        return self.db.append_audit(
            Action.EVENT_DETECTED,
            audio_file_id=audio_file_id,
            event_type=event.type.value,
            audio_start=event.start,
            audio_end=event.end,
            detail={"confidence": event.confidence, "speaker": event.speaker_id},
        )

    def event_confirmed(self, audio_file_id: int, event: AnalysisEvent) -> int:
        """Record a human confirmation, including the tag the human chose."""
        return self.db.append_audit(
            Action.EVENT_CONFIRMED,
            audio_file_id=audio_file_id,
            event_type=event.type.value,
            audio_start=event.start,
            audio_end=event.end,
            detail={
                "confirmed_tag": event.confirmed_tag,
                "proposed_tag": event.proposed_tag,
                "detector_confidence": event.confidence,
                # Recorded so an override of the suggestion is visible later.
                "overrode_suggestion": event.confirmed_tag != event.proposed_tag,
            },
        )

    def event_rejected(self, audio_file_id: int, event: AnalysisEvent) -> int:
        return self.db.append_audit(
            Action.EVENT_REJECTED,
            audio_file_id=audio_file_id,
            event_type=event.type.value,
            audio_start=event.start,
            audio_end=event.end,
            detail={"detector_confidence": event.confidence, "note": event.note},
        )

    def transcription_saved(
        self, audio_file_id: int, revision: int, text: str, source: str
    ) -> int:
        return self.db.append_audit(
            Action.TRANSCRIPTION_SAVED,
            audio_file_id=audio_file_id,
            detail={
                "revision": revision,
                "source": source,
                "length": len(text),
                # The text itself is in the transcriptions table; the audit row
                # records that a save happened and how large it was.
            },
        )

    def validation_run(self, audio_file_id: int, report: ValidationReport) -> int:
        return self.db.append_audit(
            Action.VALIDATION_RUN,
            audio_file_id=audio_file_id,
            detail={
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "infos": len(report.infos),
                "codes": sorted(c.value for c in report.codes()),
            },
        )

    def speaker_metadata_changed(
        self, audio_file_id: int, before: SpeakerProfile | None, after: SpeakerProfile
    ) -> int:
        return self.db.append_audit(
            Action.SPEAKER_METADATA_CHANGED,
            audio_file_id=audio_file_id,
            detail={
                "speaker_id": after.speaker_id,
                "before": before.to_dict() if before else None,
                "after": after.to_dict(),
            },
        )

    def speakers_merged(self, audio_file_id: int, sources: list[str], target: str) -> int:
        return self.db.append_audit(
            Action.SPEAKERS_MERGED,
            audio_file_id=audio_file_id,
            detail={"merged": sources, "into": target},
        )

    def speaker_split(self, audio_file_id: int, source: str, results: list[str]) -> int:
        return self.db.append_audit(
            Action.SPEAKER_SPLIT,
            audio_file_id=audio_file_id,
            detail={"split": source, "into": results},
        )

    def exported(self, audio_file_id: int, path: str, kind: str) -> int:
        return self.db.append_audit(
            Action.EXPORTED,
            audio_file_id=audio_file_id,
            detail={"path": path, "kind": kind},
        )
