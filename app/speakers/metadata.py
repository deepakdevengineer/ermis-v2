"""Speaker registry: metadata, merge, split, renumbering (CLAUDE.md section 18A).

Holds the annotator's view of who is speaking. Automatic diarization only ever
*seeds* this registry; from then on it is the human's, and section 18A.11's rule
is enforced structurally: "Never silently overwrite human speaker assignments."

Key invariants:
  - Speaker ids are always s1..sN with no gaps, renumbered by FIRST APPEARANCE
    after any edit (section 18A.2). A merge or split therefore cannot leave a
    hole such as s1, s3.
  - gender is one of msv / fsv / unknown; nativity is one of
    native / nonnative / unknown / tts. Nothing else, ever (section 18A.10).
  - Both default to `unknown`, and no automatic process may set them. Section
    18A.3 and 18A.4 forbid guessing, and section 37 requires `unknown` when a
    speaker produces only laughter or coughing with no intelligible speech.
  - Every edit is reported so the caller can write it to the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.models.event import AnalysisEvent, EventType
from app.speakers.diarization import DiarizationResult
from app.storage.database import ALLOWED_GENDERS, ALLOWED_NATIVITY, SpeakerProfile


class SpeakerError(ValueError):
    """Raised on an invalid speaker operation."""


@dataclass
class SpeakerRegistry:
    """The annotator's speaker list for one audio file."""

    #: Profiles keyed by speaker id ('s1', 's2', ...).
    profiles: dict[str, SpeakerProfile] = field(default_factory=dict)
    #: Speech segments as (start, end, speaker_id).
    segments: list[tuple[float, float, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    @property
    def count(self) -> int:
        """Total speakers, as displayed by section 18A.1."""
        return len(self.profiles)

    @property
    def speaker_ids(self) -> list[str]:
        """Ids in numeric order, which is also first-appearance order."""
        return sorted(self.profiles, key=_speaker_number)

    @property
    def numbers(self) -> set[int]:
        """Speaker numbers, for the validator's reference check."""
        return {_speaker_number(sid) for sid in self.profiles}

    def get(self, speaker_id: str) -> SpeakerProfile:
        if speaker_id not in self.profiles:
            raise SpeakerError(f"no such speaker: {speaker_id}")
        return self.profiles[speaker_id]

    def timeline(self, speaker_id: str) -> list[tuple[float, float]]:
        return [(a, b) for a, b, sid in self.segments if sid == speaker_id]

    def is_single_speaker(self) -> bool:
        """Single-speaker audio takes no speaker tag at all (section 17)."""
        return self.count <= 1

    # ------------------------------------------------------------------- seed
    @classmethod
    def from_diarization(cls, result: DiarizationResult) -> SpeakerRegistry:
        """Seed a registry from a diarization suggestion.

        gender and nativity are left `unknown` and `human_edited` is False, so
        the annotator can see nothing has been asserted about these people.
        """
        registry = cls()
        for index, speaker_id in enumerate(result.speaker_ids, start=1):
            registry.profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                display_name=f"Speaker {index}",
                gender="unknown",
                nativity="unknown",
                first_seen=result.first_seen(speaker_id),
                last_seen=result.last_seen(speaker_id),
                confidence=result.speaker_confidence(speaker_id),
                human_edited=False,
            )
        registry.segments = [
            (s.start, s.end, s.speaker_id)
            for s in sorted(result.segments, key=lambda s: s.start)
        ]
        return registry

    # -------------------------------------------------------------- mutations
    def add_speaker(self, display_name: str | None = None) -> str:
        """Add a speaker manually (section 18A.1)."""
        number = max(self.numbers, default=0) + 1
        speaker_id = f"s{number}"
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name or f"Speaker {number}",
            human_edited=True,
        )
        return speaker_id

    def remove_speaker(self, speaker_id: str) -> None:
        """Remove a speaker and detach their segments."""
        self.get(speaker_id)
        del self.profiles[speaker_id]
        self.segments = [seg for seg in self.segments if seg[2] != speaker_id]
        self._renumber()

    def set_metadata(
        self,
        speaker_id: str,
        gender: str | None = None,
        nativity: str | None = None,
        display_name: str | None = None,
    ) -> tuple[SpeakerProfile, SpeakerProfile]:
        """Set metadata as a human edit.

        Returns (before, after) so the caller can record the change in the
        audit log. Marks the profile `human_edited`, which stops any later
        automatic pass from overwriting it (section 18A.11).
        """
        before = self.get(speaker_id)
        if gender is not None and gender not in ALLOWED_GENDERS:
            raise SpeakerError(
                f"gender {gender!r} invalid; allowed: {ALLOWED_GENDERS} (section 18A.3)"
            )
        if nativity is not None and nativity not in ALLOWED_NATIVITY:
            raise SpeakerError(
                f"nativity {nativity!r} invalid; allowed: {ALLOWED_NATIVITY} "
                "(section 18A.4)"
            )
        after = replace(
            before,
            gender=before.gender if gender is None else gender,
            nativity=before.nativity if nativity is None else nativity,
            display_name=before.display_name if display_name is None else display_name,
            human_edited=True,
        )
        self.profiles[speaker_id] = after
        return before, after

    def suggest_metadata(
        self,
        speaker_id: str,
        gender: str | None = None,
        nativity: str | None = None,
        confidence: float | None = None,
    ) -> tuple[SpeakerProfile, SpeakerProfile] | None:
        """Apply an AUTOMATIC suggestion, never a human edit (section 18A.11).

        Returns (before, after), or None when nothing was applied. The profile
        is not marked `human_edited`, so the panel can still show the value as
        a suggestion, and a profile the annotator has already edited is left
        untouched: the application must never silently overwrite a human
        speaker decision.
        """
        before = self.get(speaker_id)
        if before.human_edited:
            return None
        if gender is not None and gender not in ALLOWED_GENDERS:
            raise SpeakerError(
                f"gender {gender!r} invalid; allowed: {ALLOWED_GENDERS} (section 18A.3)"
            )
        if nativity is not None and nativity not in ALLOWED_NATIVITY:
            raise SpeakerError(
                f"nativity {nativity!r} invalid; allowed: {ALLOWED_NATIVITY} "
                "(section 18A.4)"
            )
        after = replace(
            before,
            gender=before.gender if gender is None else gender,
            nativity=before.nativity if nativity is None else nativity,
            confidence=before.confidence if confidence is None else confidence,
            human_edited=False,
        )
        if after == before:
            return None
        self.profiles[speaker_id] = after
        return before, after

    def merge(self, sources: list[str], target: str) -> None:
        """Merge speakers that are actually the same person (section 18A.8).

        All segments and metadata move to `target`, then ids are renumbered so
        no gap is left behind.
        """
        if target in sources:
            sources = [s for s in sources if s != target]
        if not sources:
            raise SpeakerError("merge needs at least one source distinct from the target")
        self.get(target)
        for source in sources:
            self.get(source)

        merged = self.profiles[target]
        for source in sources:
            other = self.profiles.pop(source)
            # Keep whichever metadata a human set; a merge must not discard a
            # human decision (section 18A.11).
            merged = replace(
                merged,
                gender=_prefer(merged.gender, other.gender, merged, other),
                nativity=_prefer(merged.nativity, other.nativity, merged, other),
                first_seen=_min_opt(merged.first_seen, other.first_seen),
                last_seen=_max_opt(merged.last_seen, other.last_seen),
                human_edited=merged.human_edited or other.human_edited,
            )
        self.profiles[target] = merged
        self.segments = [
            (a, b, target if sid in sources else sid) for a, b, sid in self.segments
        ]
        self._renumber()

    def split(self, speaker_id: str, boundary: float) -> tuple[str, str]:
        """Split one speaker into two at a timeline boundary (section 18A.8).

        Segments starting before `boundary` stay with the original speaker;
        the rest move to a new one. Returns the two ids after renumbering.

        The new speaker's metadata starts at `unknown`: nothing is known about
        a person who has just been separated out, and copying the original's
        gender or nativity would be an unfounded assertion.
        """
        self.get(speaker_id)
        own = self.timeline(speaker_id)
        if not own:
            raise SpeakerError(f"{speaker_id} has no segments to split")
        if not any(a < boundary for a, _ in own) or not any(
            a >= boundary for a, _ in own
        ):
            raise SpeakerError(
                f"boundary {boundary} does not divide {speaker_id}'s segments"
            )

        new_id = self.add_speaker()
        self.profiles[new_id] = replace(self.profiles[new_id], human_edited=True)
        self.segments = [
            (a, b, new_id if sid == speaker_id and a >= boundary else sid)
            for a, b, sid in self.segments
        ]
        self._refresh_bounds()
        mapping = self._renumber()
        return mapping.get(speaker_id, speaker_id), mapping.get(new_id, new_id)

    def reassign_segment(self, index: int, speaker_id: str) -> None:
        """Move one segment to a different speaker."""
        self.get(speaker_id)
        if not 0 <= index < len(self.segments):
            raise SpeakerError(f"segment index {index} out of range")
        a, b, _ = self.segments[index]
        self.segments[index] = (a, b, speaker_id)
        self._refresh_bounds()
        self._renumber()

    def set_count(self, count: int) -> None:
        """Force the speaker count to the human's figure (section 18A.11).

        The example in section 18A.11 is exactly this: a 76%-confident estimate
        of 3 speakers, corrected by the annotator to 2. Reducing the count
        merges the surplus speakers into the last kept one rather than dropping
        their segments, so no speech becomes unattributed.
        """
        if count < 0:
            raise SpeakerError("speaker count cannot be negative")
        while self.count < count:
            self.add_speaker()
        if self.count > count:
            if count == 0:
                self.profiles.clear()
                self.segments = [(a, b, "") for a, b, _ in self.segments]
                return
            keep = self.speaker_ids[:count]
            surplus = self.speaker_ids[count:]
            self.merge(surplus, keep[-1])

    # ------------------------------------------------------------- numbering
    def _renumber(self) -> dict[str, str]:
        """Renumber to s1..sN by first appearance (section 18A.2).

        Returns the old-id to new-id mapping. Speakers with no segments are
        ordered after those that have them, keeping their relative order.
        """
        first: dict[str, float] = {}
        for start, _, sid in sorted(self.segments, key=lambda seg: seg[0]):
            first.setdefault(sid, start)

        with_audio = sorted(
            (sid for sid in self.profiles if sid in first), key=lambda s: first[s]
        )
        without_audio = [
            sid for sid in self.speaker_ids if sid not in first
        ]
        order = with_audio + without_audio

        mapping = {old: f"s{i + 1}" for i, old in enumerate(order)}
        if all(old == new for old, new in mapping.items()):
            return mapping

        renamed: dict[str, SpeakerProfile] = {}
        for old, new in mapping.items():
            profile = self.profiles[old]
            number = _speaker_number(new)
            renamed[new] = replace(
                profile,
                speaker_id=new,
                # Only regenerate a default display name; a name the annotator
                # typed is theirs and must survive renumbering.
                display_name=(
                    f"Speaker {number}"
                    if _is_default_name(profile.display_name)
                    else profile.display_name
                ),
            )
        self.profiles = renamed
        self.segments = [
            (a, b, mapping.get(sid, sid)) for a, b, sid in self.segments
        ]
        return mapping

    def _refresh_bounds(self) -> None:
        """Recompute first_seen / last_seen from the current segments."""
        for speaker_id, profile in list(self.profiles.items()):
            own = self.timeline(speaker_id)
            self.profiles[speaker_id] = replace(
                profile,
                first_seen=min((a for a, _ in own), default=None),
                last_seen=max((b for _, b in own), default=None),
            )

    # ---------------------------------------------------------------- output
    def to_events(self) -> list[AnalysisEvent]:
        """Speech events with the registry's current attribution."""
        return [
            AnalysisEvent(
                type=EventType.SPEECH, start=a, end=b, speaker_id=sid, confidence=1.0
            )
            for a, b, sid in sorted(self.segments)
            if sid
        ]

    def to_dict(self) -> dict:
        return {
            "total_speakers": self.count,
            "speakers": [self.profiles[sid].to_dict() for sid in self.speaker_ids],
            "segments": [
                {"start": round(a, 3), "end": round(b, 3), "speaker": sid}
                for a, b, sid in sorted(self.segments)
            ],
        }

    def render_panel(self) -> str:
        """The speaker panel of section 18A.9, as text."""
        lines = [
            "-----------------------------------------",
            "SPEAKERS",
            "-----------------------------------------",
            "",
            f"Total Speakers: {self.count}",
            "",
        ]
        for index, speaker_id in enumerate(self.speaker_ids, start=1):
            profile = self.profiles[speaker_id]
            confidence = (
                f"{profile.confidence * 100:.0f}%"
                if profile.confidence is not None
                else "n/a"
            )
            lines += [
                f"Speaker {index}",
                f"ID: {profile.speaker_id}",
                f"Gender: [{profile.gender}]",
                f"Nativity: [{profile.nativity}]",
                f"Confidence: {confidence}"
                + ("  (human edited)" if profile.human_edited else ""),
                "",
            ]
        lines += [
            "[Merge Speakers]",
            "[Split Speaker]",
            "[Add Speaker]",
            "-----------------------------------------",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _speaker_number(speaker_id: str) -> int:
    digits = speaker_id.lstrip("s")
    return int(digits) if digits.isdigit() else 0


def _is_default_name(name: str) -> bool:
    parts = name.split()
    return len(parts) == 2 and parts[0] == "Speaker" and parts[1].isdigit()


def _prefer(
    a: str, b: str, profile_a: SpeakerProfile, profile_b: SpeakerProfile
) -> str:
    """Choose between two metadata values during a merge.

    A human-set value always beats an automatic one; a concrete value beats
    `unknown`; otherwise the target's value stands.
    """
    if a == b:
        return a
    if profile_a.human_edited and not profile_b.human_edited:
        return a
    if profile_b.human_edited and not profile_a.human_edited:
        return b
    if a == "unknown":
        return b
    return a


def _min_opt(a: float | None, b: float | None) -> float | None:
    values = [v for v in (a, b) if v is not None]
    return min(values) if values else None


def _max_opt(a: float | None, b: float | None) -> float | None:
    values = [v for v in (a, b) if v is not None]
    return max(values) if values else None
