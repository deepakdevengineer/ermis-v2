"""Brief audio artifact detection (CLAUDE.md sections 9, 40, 46).

Finds candidates for [artifact]: a pop, click, static burst, feedback squeal or
momentary glitch, where surrounding speech remains usable.

Section 9's decision rule cannot be automated:

    Can meaningful speech still be transcribed around the noise?
      YES -> [artifact]
      NO  -> <ga>

Only a listener can answer that. This detector finds *brief transients*, which
is the observable half of the rule, and every event it emits is
`possible_artifact` requiring confirmation. The review UI shows the rule
alongside the candidate so the annotator makes the [artifact]-versus-<ga> call.

Section 9 also excludes persistent background hiss and hum from [artifact].
This detector is duration-bounded, so a steady hiss never qualifies -- it is
not a transient.
"""

from __future__ import annotations

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    boolean_runs,
    compute_grid,
    spectral_flatness,
    spectral_flux,
)
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Longest a transient can be and still be a brief artifact. Beyond this it is
#: sustained noise, which section 9 excludes from [artifact].
MAX_ARTIFACT_SECONDS = 0.40

#: Shortest event worth surfacing. Below this it is a single sample of
#: quantisation noise, not something an annotator can hear or act on.
MIN_ARTIFACT_SECONDS = 0.004

#: Spectral flatness above which a transient is broadband enough to be
#: technical noise rather than a voiced syllable.
#:
#: Measured on synthetic connected speech, syllable-onset flatness peaks around
#: 0.36 (99th percentile) with a maximum near 0.42, while a broadband click
#: sits near 0.49. 0.45 is above the speech distribution and below the click.
BROADBAND_FLATNESS = 0.45

#: Spectral flux z-score above which a frame is a sharp transient.
FLUX_Z_THRESHOLD = 4.0

#: Energy jump in dB over the preceding frames that marks an onset.
ONSET_JUMP_DB = 12.0


class ArtifactDetector:
    """Finds brief transients that may warrant [artifact]."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        vocal_events: list[AnalysisEvent] | None = None,
    ) -> list[AnalysisEvent]:
        """Find [artifact] candidates.

        Args:
            vocal_events: laughter and [hn] candidates. Overlapping artifact
                candidates are dropped. Section 9 defines [artifact] as
                *technical* noise -- a pop, click or static burst -- so a cough
                or a laugh, which are broadband transients too, belong to [hn]
                and [laughter] instead.
        """
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        if grid.frame_count < 4:
            return []

        flux = spectral_flux(grid)
        flatness = spectral_flatness(grid)
        rms_db = grid.rms_dbfs

        # A transient is a sudden spectral change. Standardising flux makes the
        # threshold independent of overall level, so a quiet recording's clicks
        # are found as readily as a loud one's.
        flux_z = _zscore(flux)

        # A click is also an abrupt energy jump relative to what came before.
        jump = np.zeros_like(rms_db)
        jump[3:] = rms_db[3:] - np.maximum.reduce(
            [rms_db[:-3], rms_db[1:-2], rms_db[2:-1]]
        )

        transient = (flux_z >= FLUX_Z_THRESHOLD) | (jump >= ONSET_JUMP_DB)

        # A click, pop or static burst is broadband. Requiring noise-like
        # spectra separates it from a sung note or a plosive consonant, which
        # are also abrupt but harmonic.
        broadband = flatness > BROADBAND_FLATNESS

        # Spike, not step. The start of an utterance is also an abrupt
        # broadband energy jump, so without this every speech onset would be
        # reported as an artifact. A click rises and falls back to roughly the
        # surrounding level; a speech onset rises and stays up.
        spike = self._is_spike(rms_db, grid)

        candidate = transient & broadband & spike

        frame_seconds = grid.frame_length / grid.sample_rate
        raw: list[tuple[int, int]] = []
        for start_frame, end_frame in boolean_runs(candidate):
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds

            # Duration bound is what keeps section 9's exclusion of persistent
            # hiss and hum honest: those are not transients.
            if not (MIN_ARTIFACT_SECONDS <= end - start <= MAX_ARTIFACT_SECONDS):
                continue
            raw.append((start_frame, end_frame))

        # One noise, one tag. A single click can straddle two analysis frames
        # and produce two adjacent runs; reporting it twice would invite the
        # annotator to write "[artifact] [artifact]", which section 41 rejects
        # as a duplicate.
        merged = self._merge_adjacent(raw, grid)

        events: list[AnalysisEvent] = []
        vocal = vocal_events or []
        for start_frame, end_frame in merged:
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds
            if end - start > MAX_ARTIFACT_SECONDS:
                continue

            # A vocal sound already claimed this moment; section 9 reserves
            # [artifact] for technical noise.
            if any(not (end <= ev.start or start >= ev.end) for ev in vocal):
                continue

            window = slice(start_frame, max(start_frame + 1, end_frame))
            events.append(
                AnalysisEvent(
                    type=EventType.POSSIBLE_ARTIFACT,
                    start=start,
                    end=min(end, buffer.duration),
                    confidence=self._confidence(
                        float(np.max(flux_z[window])),
                        float(np.max(jump[window])),
                        float(np.mean(flatness[window])),
                    ),
                    note=(
                        "brief transient. Section 9: use [artifact] if speech can "
                        "still be transcribed around it, <ga> if it compromises "
                        "evaluation."
                    ),
                )
            )
        return events

    @staticmethod
    def _is_spike(
        rms_db: np.ndarray,
        grid: FrameGrid,
        context: float = 0.15,
        guard: float = 0.04,
        prominence_db: float = 4.0,
    ) -> np.ndarray:
        """True where a frame stands above the level on *both* sides.

        This is the spike-versus-step discriminator:

            click:         low  --[HIGH]--  low      (above both sides)
            speech onset:  low  --[ mid ]-- high     (above one side only)
            speech offset: high --[ mid ]-- low      (above one side only)

        Comparing the two sides to each other instead would be confounded by
        ordinary speech amplitude modulation, which moves the level by more
        than any sensible tolerance over a 150 ms window.

        `guard` skips the frames immediately around the transient, whose energy
        is smeared into them by the overlapping analysis window.
        """
        n = rms_db.size
        if n < 3:
            return np.zeros(n, dtype=bool)
        span = max(1, int(round(context / grid.hop_seconds)))
        gap = max(1, int(round(guard / grid.hop_seconds)))
        finite = np.where(np.isfinite(rms_db), rms_db, -120.0)

        before = np.empty(n)
        after = np.empty(n)
        for i in range(n):
            lo, hi = max(0, i - span - gap), max(1, i - gap)
            before[i] = np.median(finite[lo:hi]) if hi > lo else finite[i]
            lo2, hi2 = min(n - 1, i + gap), min(n, i + span + gap)
            after[i] = np.median(finite[lo2:hi2]) if hi2 > lo2 else finite[i]

        return (finite - before >= prominence_db) & (finite - after >= prominence_db)

    @staticmethod
    def _merge_adjacent(
        runs: list[tuple[int, int]], grid: FrameGrid, gap_seconds: float = 0.05
    ) -> list[tuple[int, int]]:
        """Merge runs separated by less than `gap_seconds`."""
        if not runs:
            return []
        gap_frames = max(1, int(round(gap_seconds / grid.hop_seconds)))
        merged = [list(runs[0])]
        for start, end in runs[1:]:
            if start - merged[-1][1] <= gap_frames:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _confidence(flux_z: float, jump_db: float, flatness: float) -> float:
        """Confidence that this is a technical artifact.

        Capped at 0.75. A transient detector cannot distinguish a microphone
        pop from a door slam or a tapped desk, and section 9's usability
        question is unanswerable from the signal, so certainty is never
        warranted (section 56.22).
        """
        score = 0.0
        score += min(0.25, max(0.0, (flux_z - FLUX_Z_THRESHOLD) / 12.0))
        score += min(0.20, max(0.0, (jump_db - ONSET_JUMP_DB) / 30.0))
        score += min(0.15, max(0.0, (flatness - BROADBAND_FLATNESS) / 2.0))
        return float(np.clip(0.35 + score, 0.35, 0.75))


def _zscore(values: np.ndarray) -> np.ndarray:
    """Robust z-score using the median and MAD, so one huge spike does not
    inflate the scale and hide the others."""
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = mad * 1.4826 if mad > 1e-12 else float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - median) / scale


def detect_artifacts(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    vocal_events: list[AnalysisEvent] | None = None,
) -> list[AnalysisEvent]:
    return ArtifactDetector(config).detect(buffer, vocal_events=vocal_events)
