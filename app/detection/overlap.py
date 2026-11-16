"""Overlapping speech detection (CLAUDE.md sections 18, 19, 46).

Finds candidates for <ol> ... </ol>, where two or more people speak at once.

Method: simultaneous speech from two voices produces two concurrent harmonic
series. That shows up as an unusually rich, non-single-pitch spectrum -- more
strong spectral peaks than one voice produces, and a harmonic structure that no
single fundamental explains. This detector measures that.

The <ol> versus <ct> decision is NOT made here. Section 19:

    <ol>  intelligible simultaneous speech
    <ct>  overlapping speech that cannot be understood

Intelligibility is exactly what a detector cannot judge, so every event is
`possible_overlap` and the note states both options. Section 56.16 is also
respected: uncertainty about *speaker order* is not cross talk, and this
detector never proposes <ct> on its own.

Honest limits, which are severe:
  - Reverberant single-speaker audio, music behind speech, and two harmonically
    related voices all produce similar richness.
  - It cannot count the speakers or say who they are.
  - Confidence is capped at 0.55, the lowest of any detector here, because this
    is the least reliable inference in the set.
"""

from __future__ import annotations

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    boolean_runs,
    compute_grid,
    spectral_flatness,
)
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Minimum overlap duration worth surfacing. Brief coincidences of a word
#: ending as another begins are normal turn-taking, not an overlap block.
MIN_OVERLAP_SECONDS = 0.30

#: Number of strong spectral peaks above which a frame looks like more than
#: one voice. A weak secondary signal only -- see the note in detect().
PEAK_COUNT_THRESHOLD = 7

#: Absolute floor on the harmonic residual. Below this, the spectrum is well
#: explained by one fundamental whatever the rest of the file looks like.
RESIDUAL_FLOOR = 0.12

#: How much a frame's harmonic residual must exceed the file's own baseline
#: residual to count as a second voice.
#:
#: Relative rather than absolute because the residual a *single* voice leaves
#: varies widely with the speaker, the microphone and the room: a breathy voice
#: or a reverberant recording leaves a much larger residual than a clean close
#: mic, and no fixed cut-off separates "one voice" from "two" across both. The
#: baseline is the file's own median, so the comparison is always against the
#: same speaker in the same room.
RESIDUAL_RATIO = 1.6

#: Search range for a voice fundamental, in Hz.
F0_MIN = 70.0
F0_MAX = 320.0

#: Analysis frame for this detector, in seconds.
#:
#: Longer than the shared 25 ms grid on purpose. Separating two fundamentals
#: needs frequency resolution: 25 ms at 16 kHz gives 40 Hz bins, so a harmonic
#: tolerance wide enough to catch a peak spans more than the gap between two
#: speakers' harmonics, and every candidate f0 appears to explain the whole
#: spectrum. 64 ms gives ~16 Hz bins, which resolves them.
OVERLAP_FRAME_SECONDS = 0.064
OVERLAP_HOP_SECONDS = 0.016


class OverlapDetector:
    """Finds regions whose spectrum suggests more than one simultaneous voice."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> list[AnalysisEvent]:
        # Always analyse on the higher-resolution grid; a caller's 25 ms grid
        # cannot resolve two fundamentals (see OVERLAP_FRAME_SECONDS).
        grid = compute_grid(
            buffer.samples,
            buffer.sample_rate,
            frame_seconds=OVERLAP_FRAME_SECONDS,
            hop_seconds=OVERLAP_HOP_SECONDS,
        )
        if grid.frame_count < 8:
            return []

        peaks = self._peak_counts(grid)
        residual = self._harmonic_residual(grid)
        flatness = spectral_flatness(grid)

        # Still has to be voiced, not broadband noise: noise also produces many
        # peaks and a large residual.
        voiced = flatness < 0.50

        # Baseline residual for this recording, measured over its voiced frames.
        # Using the file's own single-voice residual as the reference is what
        # makes the threshold portable across speakers and rooms.
        # A low percentile, not the median: when overlapping speech occupies a
        # large share of the clip it drags the median up, hiding the very thing
        # being looked for. The quietest-residual voiced frames are the ones
        # most likely to be a single voice, so they define the baseline.
        reference = residual[voiced & (residual > 0)]
        baseline = float(np.percentile(reference, 25)) if reference.size else 0.0
        threshold = max(RESIDUAL_FLOOR, baseline * RESIDUAL_RATIO)

        unexplained = residual >= threshold

        # Peak count is a supporting signal, not a requirement. Whether a
        # second voice *adds* peaks depends on where its harmonics fall: they
        # can land between the first voice's harmonics (more peaks) or on top
        # of them (no more peaks). Requiring it would miss the second case, so
        # it only contributes to confidence.
        candidate = unexplained & voiced

        if speech_spans is not None:
            # Overlap can only occur where there is speech.
            candidate &= self._span_mask(speech_spans, grid)

        frame_seconds = grid.frame_length / grid.sample_rate
        events: list[AnalysisEvent] = []
        for start_frame, end_frame in boolean_runs(candidate):
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds
            if end - start < MIN_OVERLAP_SECONDS:
                continue
            window = slice(start_frame, max(start_frame + 1, end_frame))
            events.append(
                AnalysisEvent(
                    type=EventType.POSSIBLE_OVERLAP,
                    start=start,
                    end=min(end, buffer.duration),
                    confidence=self._confidence(
                        float(np.mean(peaks[window])),
                        float(np.mean(residual[window])),
                    ),
                    note=(
                        "spectrum suggests more than one simultaneous voice. If the "
                        "speech is intelligible use <ol> with every speaker re-tagged "
                        "inside the block; if it cannot be understood use <ct> "
                        "(sections 18, 19). Uncertainty about speaker order alone is "
                        "not <ct> (section 56.16)."
                    ),
                )
            )
        return events

    # -------------------------------------------------------------- internals
    @staticmethod
    def _peak_counts(grid: FrameGrid) -> np.ndarray:
        """Count strong spectral peaks per frame."""
        spectrum = grid.spectrum
        if spectrum.shape[1] < 3:
            return np.zeros(grid.frame_count)
        interior = spectrum[:, 1:-1]
        is_peak = (interior > spectrum[:, :-2]) & (interior >= spectrum[:, 2:])
        # Only peaks that are a meaningful fraction of the frame maximum count,
        # so spectral ripple in a quiet frame is ignored.
        frame_max = spectrum.max(axis=1, keepdims=True)
        strong = is_peak & (interior >= 0.20 * np.where(frame_max > 0, frame_max, 1.0))
        return strong.sum(axis=1).astype(np.float64)

    @staticmethod
    def _harmonic_residual(grid: FrameGrid) -> np.ndarray:
        """Fraction of spectral energy unexplained by the best single f0.

        For each frame, the fundamental in [F0_MIN, F0_MAX] whose harmonic
        series captures the most energy is chosen; the residual is what that
        series leaves behind. One voice leaves little; two voices leave a lot.
        """
        frequencies = grid.frequencies
        spectrum = grid.spectrum
        if spectrum.size == 0:
            return np.zeros(grid.frame_count)

        energy = np.square(spectrum)
        total = energy.sum(axis=1)
        total_safe = np.where(total > 0, total, 1.0)

        # Tolerance is one bin: wide enough to catch a harmonic that falls
        # between bins, tight enough that a wrong f0 does not appear to
        # explain the spectrum.
        bin_width = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 1.0
        tolerance = bin_width

        # The sweep step is finer than the bin width so a true f0 is never
        # missed between steps.
        best = np.zeros(spectrum.shape[0])
        for f0 in np.arange(F0_MIN, F0_MAX, 2.0):
            harmonics = np.arange(f0, frequencies[-1], f0)
            if harmonics.size == 0:
                continue
            mask = np.zeros(frequencies.size, dtype=bool)
            for harmonic in harmonics:
                mask |= np.abs(frequencies - harmonic) <= tolerance
            captured = energy[:, mask].sum(axis=1) / total_safe
            best = np.maximum(best, captured)

        return np.clip(1.0 - best, 0.0, 1.0)

    @staticmethod
    def _span_mask(spans: list[tuple[float, float]], grid: FrameGrid) -> np.ndarray:
        mask = np.zeros(grid.frame_count, dtype=bool)
        for start, end in spans:
            mask[grid.time_to_frame(start) : grid.time_to_frame(end) + 1] = True
        return mask

    @staticmethod
    def _confidence(peak_count: float, residual: float) -> float:
        """Confidence capped at 0.55 -- the lowest of any detector here."""
        score = 0.0
        score += min(0.12, max(0.0, (peak_count - PEAK_COUNT_THRESHOLD) / 40.0))
        score += min(0.13, max(0.0, (residual - RESIDUAL_FLOOR) / 3.0))
        return float(np.clip(0.30 + score, 0.30, 0.55))


def detect_overlap(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[AnalysisEvent]:
    return OverlapDetector(config).detect(buffer, speech_spans=speech_spans)
