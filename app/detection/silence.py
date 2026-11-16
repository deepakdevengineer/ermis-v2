"""Silence and pause detection (CLAUDE.md sections 13, 45, 46).

CRITICAL RULE, section 13: Ermis V2 defines no transcription tag for pauses.
Pause detection exists for navigation and QA only. Nothing in this module may
produce a tag, and PAUSE events map to `None` in the event model's tag table.
The validator reports `[pause]` in a transcription as an error, and the
database CHECK constraint rejects it too.

Distinction used here:
  - SILENCE: a quiet region, regardless of context.
  - PAUSE: a silence *between two speech regions*, i.e. a gap in speech rather
    than lead-in or trailing quiet.

Leading and trailing quiet is reported as SILENCE, never PAUSE, because a
speaker had not started or had finished; calling that a pause would mislead the
annotator reviewing turn structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import FrameGrid, compute_grid
from app.config import DetectionConfig
from app.detection.vad import EnergyVad, VadResult
from app.models.event import AnalysisEvent, EventType


@dataclass(frozen=True)
class SilenceResult:
    silence: list[tuple[float, float]]
    pauses: list[tuple[float, float]]
    speech: list[tuple[float, float]]

    @property
    def longest_pause(self) -> float:
        return max((b - a for a, b in self.pauses), default=0.0)

    @property
    def total_silence(self) -> float:
        return float(sum(b - a for a, b in self.silence))


class SilenceDetector:
    """Finds quiet regions and classifies the ones between speech as pauses."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def analyse(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        vad_result: VadResult | None = None,
    ) -> SilenceResult:
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        vad_result = vad_result or EnergyVad(self.config).analyse(buffer, grid)

        speech = vad_result.speech
        duration = buffer.duration

        # Silence is the complement of speech, subject to a minimum length so
        # the inter-word gaps inside an utterance are not each reported.
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in speech:
            if start - cursor >= self.config.min_silence_duration:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if duration - cursor >= self.config.min_silence_duration:
            gaps.append((cursor, duration))

        # A pause is a gap with speech on both sides.
        pauses = [
            (start, end)
            for start, end in gaps
            if end - start >= self.config.pause_min_duration
            and any(s_end <= start + 1e-9 for _, s_end in speech)
            and any(s_start >= end - 1e-9 for s_start, _ in speech)
        ]
        return SilenceResult(silence=gaps, pauses=pauses, speech=speech)

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        vad_result: VadResult | None = None,
    ) -> list[AnalysisEvent]:
        """Silence and pause regions as analysis events.

        Both types are certain measurements rather than guesses, so neither
        carries a `possible_` type. Neither maps to a transcription tag.
        """
        result = self.analyse(buffer, grid, vad_result)
        pause_set = set(result.pauses)
        events: list[AnalysisEvent] = []
        for start, end in result.silence:
            is_pause = (start, end) in pause_set
            events.append(
                AnalysisEvent(
                    type=EventType.PAUSE if is_pause else EventType.SILENCE,
                    start=start,
                    end=end,
                    confidence=1.0,
                    note=(
                        "internal QA metadata; V2 has no pause tag (section 13)"
                        if is_pause
                        else ""
                    ),
                )
            )
        return sorted(events, key=lambda e: e.start)


def detect_silence(
    buffer: AudioBuffer, config: DetectionConfig | None = None
) -> list[AnalysisEvent]:
    return SilenceDetector(config).detect(buffer)


def format_pause_info(events: list[AnalysisEvent]) -> list[str]:
    """Human-readable INFO lines for the validation panel (section 49).

    The wording states the rule every time, so an annotator reading the panel
    is never nudged toward inserting a tag.
    """
    lines = []
    for event in events:
        if event.type is not EventType.PAUSE:
            continue
        lines.append(
            f"{event.duration:.1f} second pause detected at "
            f"{event.start:.2f}s. Pause is internal metadata and should not be "
            "inserted as a transcription tag."
        )
    return lines


def silence_covers_whole_file(
    buffer: AudioBuffer, result: SilenceResult, tolerance: float = 0.05
) -> bool:
    """True when no speech was found anywhere.

    A candidate for the <ns> or <na> whole-file conditions -- but only a
    candidate. The loader distinguishes "no signal" (<na>) from "signal but no
    speech" (<ns>), and section 12 requires that no other tag applies, which
    only a human can settle by listening.
    """
    if not np.isfinite(buffer.duration) or buffer.duration <= 0:
        return True
    return not result.speech or result.total_silence >= buffer.duration - tolerance
