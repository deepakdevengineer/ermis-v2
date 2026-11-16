"""Internal analysis event model (CLAUDE.md sections 3, 13, 46).

An AnalysisEvent belongs to Layer A: audio analysis. It is never a
transcription tag. The bridge from a `possible_*` event to a V2 tag is a human
decision, recorded through `confirm()` / `reject()` and the audit log
(section 47).

The invariant enforced here: `suggested_tag` is available only after a human
has confirmed the event. Reading it before confirmation raises, so no code
path can silently promote a detector guess into the transcription
(section 56.22).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.ermis import tags as T


class EventType(str, enum.Enum):
    """Analysis event types (CLAUDE.md section 46)."""

    SPEECH = "speech"
    PAUSE = "pause"
    SILENCE = "silence"
    POSSIBLE_LAUGHTER = "possible_laughter"
    POSSIBLE_HN = "possible_hn"
    POSSIBLE_ARTIFACT = "possible_artifact"
    POSSIBLE_BACKGROUND = "possible_background"
    POSSIBLE_OVERLAP = "possible_overlap"
    SPEAKER_CHANGE = "speaker_change"
    GARBLED_AUDIO = "garbled_audio"
    NO_AUDIO = "no_audio"
    NO_SPEECH = "no_speech"

    @property
    def is_uncertain(self) -> bool:
        """True for detector guesses, which always need human confirmation."""
        return self.value.startswith("possible_")


class ConfirmationState(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


#: The V2 tag a confirmed event *may* correspond to. Advisory: the annotator
#: can pick a different tag, which is why confirm() takes an override.
#:
#: PAUSE and SILENCE map to nothing on purpose. V2 has no pause tag and one is
#: never inserted (sections 13 and 56.10).
SUGGESTED_TAG: dict[EventType, str | None] = {
    EventType.SPEECH: None,
    EventType.PAUSE: None,
    EventType.SILENCE: None,
    EventType.POSSIBLE_LAUGHTER: T.LAUGHTER,
    EventType.POSSIBLE_HN: T.NON_LEXICAL,
    EventType.POSSIBLE_ARTIFACT: T.ARTIFACT,
    EventType.POSSIBLE_BACKGROUND: T.BACKGROUND,
    EventType.POSSIBLE_OVERLAP: T.OVERLAP_OPEN,
    EventType.SPEAKER_CHANGE: None,
    EventType.GARBLED_AUDIO: T.GARBLED,
    EventType.NO_AUDIO: T.NO_AUDIO,
    EventType.NO_SPEECH: T.NO_SPEECH,
}


class UnconfirmedEventError(RuntimeError):
    """Raised when code tries to read a tag from an unconfirmed event."""


@dataclass
class AnalysisEvent:
    """One detected audio event.

    Times are seconds from the start of the file.
    """

    type: EventType
    start: float
    end: float
    confidence: float = 1.0
    speaker_id: str | None = None
    event_id: str | None = None
    audio_file_id: int | None = None
    state: ConfirmationState = ConfirmationState.PENDING
    #: The tag the human chose at confirmation time. None means "no tag".
    confirmed_tag: str | None = None
    note: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"event end {self.end} precedes start {self.start}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")

    # ------------------------------------------------------------- properties
    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def requires_human_confirmation(self) -> bool:
        """Every uncertain event needs confirmation (section 46)."""
        return self.type.is_uncertain and self.state is ConfirmationState.PENDING

    @property
    def is_confirmed(self) -> bool:
        return self.state is ConfirmationState.CONFIRMED

    @property
    def proposed_tag(self) -> str | None:
        """The tag this event *would* map to, for display in the review UI.

        This is a suggestion shown next to a confidence figure. It is not a
        transcription edit; only `tag` (post-confirmation) may reach the
        transcription.
        """
        return SUGGESTED_TAG.get(self.type)

    @property
    def tag(self) -> str | None:
        """The V2 tag this event contributes to the transcription.

        Raises UnconfirmedEventError while the event is still PENDING, which is
        what keeps a detector guess out of the final transcription
        (sections 46, 47, 56.22).

        A REJECTED event returns None: that is a settled human decision of
        "no tag", so callers building a transcription can treat it as such
        without handling an exception.
        """
        if self.state is ConfirmationState.PENDING:
            raise UnconfirmedEventError(
                f"{self.type.value} at {self.start:.2f}-{self.end:.2f} is "
                f"{self.state.value}; a tag is only available after human "
                "confirmation (CLAUDE.md section 47)"
            )
        return self.confirmed_tag

    # ---------------------------------------------------------------- actions
    def confirm(self, tag: str | None = None) -> None:
        """Record a human confirmation.

        Args:
            tag: the tag the annotator chose. Defaults to `proposed_tag`. Pass
                None explicitly via `confirm(tag=None)` on an event whose
                proposal is None to confirm "no tag".
        """
        chosen = self.proposed_tag if tag is None else tag
        # The pause check comes first so its specific explanation wins over the
        # generic "not a valid tag" message.
        if chosen in T.PAUSE_TAGS:
            raise ValueError(
                f"{chosen!r}: V2 defines no pause tag; pause stays internal "
                "metadata (CLAUDE.md section 13)"
            )
        if chosen is not None and chosen not in T.ALL_FIXED_TAGS and not T.is_speaker_tag(chosen):
            raise ValueError(f"{chosen!r} is not a valid Ermis V2 tag")
        self.confirmed_tag = chosen
        self.state = ConfirmationState.CONFIRMED

    def reject(self, note: str = "") -> None:
        """Record a human rejection. The event is kept for the audit trail."""
        self.state = ConfirmationState.REJECTED
        self.confirmed_tag = None
        if note:
            self.note = note

    # ------------------------------------------------------------ conversion
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 4),
            "speaker": self.speaker_id,
            "state": self.state.value,
            "confirmed_tag": self.confirmed_tag,
            "requires_human_confirmation": self.requires_human_confirmation,
            "note": self.note,
        }

    def __str__(self) -> str:
        return (
            f"{_ts(self.start)} - {_ts(self.end)}  {self.type.value.upper()}"
            + (f"  conf={self.confidence:.2f}" if self.type.is_uncertain else "")
            + (f"  {self.speaker_id}" if self.speaker_id else "")
        )


def _ts(seconds: float) -> str:
    """Format seconds as MM:SS.CC, the timeline format used in CLAUDE.md."""
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def format_timeline(events: list[AnalysisEvent]) -> str:
    """Render events as the analysis timeline shown in CLAUDE.md section 3."""
    return "\n".join(str(e) for e in sorted(events, key=lambda e: (e.start, e.end)))
