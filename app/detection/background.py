"""Background sound detection (CLAUDE.md sections 21, 46).

Finds candidates for [bg], used when a sound cannot be attributed to an
identified speaker: background speech, laughter, coughing or other non-lexical
sounds from unidentified people.

Section 21 sets two traps this detector is written to avoid:

  1. "Do NOT use [bg] just because speech is quiet or distant."
     Low level alone is never sufficient here. A candidate also has to be
     *acoustically separate* from the identified speakers -- markedly different
     spectral character, not merely quieter.

  2. "If an already identified speaker becomes temporarily unintelligible, do
     NOT convert them to [bg]." Use their speaker tag with (()).
     So candidates overlapping a known speaker's region are suppressed, and the
     event note states the rule.

Honest limit: attribution is fundamentally a human judgement. This detector can
say "there is a second, quieter, spectrally distinct voice-like sound here"; it
cannot say whether the annotator has identified that person. Everything is
`possible_background` and requires confirmation.
"""

from __future__ import annotations

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    band_energy_ratio,
    boolean_runs,
    compute_grid,
    spectral_centroid,
    spectral_flatness,
)
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Minimum duration for a background candidate. Shorter events are better
#: described as artifacts or non-lexical sounds.
MIN_BG_SECONDS = 0.25

#: How far below the foreground speech level a background sound must sit.
#: Level alone is not sufficient (section 21) but it is necessary.
MIN_LEVEL_GAP_DB = 10.0

#: Spectral-centroid difference, in Hz, that counts as "acoustically distinct"
#: from the foreground. This is the part that stops mere quietness qualifying.
MIN_CENTROID_DELTA_HZ = 250.0


class BackgroundDetector:
    """Finds quiet, spectrally distinct voice-like sounds."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> list[AnalysisEvent]:
        """Find [bg] candidates.

        Args:
            speech_spans: VAD speech regions, treated as the identified
                speakers' audio. Candidates inside them are suppressed, per
                section 21's rule about not converting a known speaker to [bg].
        """
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        if grid.frame_count < 8:
            return []

        rms_db = grid.rms_dbfs
        finite = rms_db[np.isfinite(rms_db)]
        if finite.size == 0:
            return []

        # Foreground reference: the level and spectral character of the loud
        # part of the file, which is the identified speaker(s).
        foreground_db = float(np.percentile(finite, 90))
        floor_db = float(np.percentile(finite, 10))

        centroid = spectral_centroid(grid)
        flatness = spectral_flatness(grid)
        voice_band = band_energy_ratio(grid, 300.0, 3400.0)

        loud_frames = rms_db >= foreground_db - 6.0
        foreground_centroid = (
            float(np.median(centroid[loud_frames])) if np.any(loud_frames) else 0.0
        )

        # Necessary condition: clearly below the foreground, but above the
        # noise floor -- otherwise it is silence, not a background sound.
        quiet = (rms_db <= foreground_db - MIN_LEVEL_GAP_DB) & (rms_db >= floor_db + 6.0)

        # Voice-like: energy in the speech band and not noise-like. This is
        # what separates background *speech* from background hiss, which is
        # not [bg] at all.
        voice_like = (voice_band >= 0.25) & (flatness < 0.60)

        # The section 21 guard: acoustically distinct from the foreground, not
        # merely quieter. Without this clause, every quiet tail of the main
        # speaker's own sentence would be proposed as [bg].
        distinct = np.abs(centroid - foreground_centroid) >= MIN_CENTROID_DELTA_HZ

        candidate = quiet & voice_like & distinct

        if speech_spans is not None:
            candidate &= ~self._span_mask(speech_spans, grid)

        frame_seconds = grid.frame_length / grid.sample_rate
        events: list[AnalysisEvent] = []
        for start_frame, end_frame in boolean_runs(candidate):
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds
            if end - start < MIN_BG_SECONDS:
                continue
            window = slice(start_frame, max(start_frame + 1, end_frame))
            events.append(
                AnalysisEvent(
                    type=EventType.POSSIBLE_BACKGROUND,
                    start=start,
                    end=min(end, buffer.duration),
                    confidence=self._confidence(
                        foreground_db - float(np.median(rms_db[window])),
                        float(np.median(np.abs(centroid[window] - foreground_centroid))),
                    ),
                    note=(
                        "quiet, spectrally distinct voice-like sound. Use [bg] only if "
                        "the source cannot be attributed to an identified speaker. "
                        "Quiet or distant speech from a known speaker is NOT [bg], and "
                        "a known speaker who becomes unintelligible keeps their speaker "
                        "tag with (()) (section 21)."
                    ),
                )
            )
        return events

    @staticmethod
    def _span_mask(spans: list[tuple[float, float]], grid: FrameGrid) -> np.ndarray:
        mask = np.zeros(grid.frame_count, dtype=bool)
        for start, end in spans:
            mask[grid.time_to_frame(start) : grid.time_to_frame(end) + 1] = True
        return mask

    @staticmethod
    def _confidence(level_gap_db: float, centroid_delta: float) -> float:
        """Confidence capped at 0.60.

        Attribution is the whole question and the signal cannot answer it, so
        this stays low by design.
        """
        score = 0.0
        score += min(0.15, max(0.0, (level_gap_db - MIN_LEVEL_GAP_DB) / 40.0))
        score += min(0.15, max(0.0, (centroid_delta - MIN_CENTROID_DELTA_HZ) / 2000.0))
        return float(np.clip(0.30 + score, 0.30, 0.60))


def detect_background(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[AnalysisEvent]:
    return BackgroundDetector(config).detect(buffer, speech_spans=speech_spans)
