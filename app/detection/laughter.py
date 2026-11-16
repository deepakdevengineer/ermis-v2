"""Laughter candidate detection (CLAUDE.md sections 8, 20, 46, 47).

Finds candidates for [laughter], which V2 keeps as its own tag -- laughter is
never converted to [hn] (sections 8 and 56.7).

Method: laughter has a distinctive rhythm. It is a train of short voiced
bursts ("ha-ha-ha") at roughly 4-7 Hz, with vowel-like spectra and unusually
high energy variance for a voiced region. This detector looks for that
periodic burst pattern inside speech-level audio.

Honest limits -- and they are significant. Read these before trusting output:

  - Laughter and rhythmic speech genuinely overlap in every feature available
    without a trained model. Speech syllables run at 3-8 Hz, which is exactly
    the laughter burst rate. Published laughter detectors use learned acoustic
    models for this reason; this is a hand-built heuristic and is materially
    weaker than one.
  - It is therefore tuned for PRECISION over RECALL: thresholds are narrow, so
    it misses quiet, breathy or short laughter rather than firing on ordinary
    speech. A missed laugh costs nothing -- the annotator is listening anyway
    and will hear it. A false positive costs attention on every clip and
    teaches the annotator to ignore the panel, which is far worse.
  - Even so it still produces false positives on unusually regular speech.
  - Confidence is capped at 0.70 and is not a probability.

Validated only against synthetic fixtures, not a real laughter corpus, so the
real-world hit rate is unmeasured.

Because of all this every event is `possible_laughter` and requires human
confirmation before any tag exists (section 47). The review UI offers
"Confirm laughter", "Change to hn" and "Reject" precisely because this detector
is expected to be wrong a fair share of the time.
"""

from __future__ import annotations

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    band_energy_ratio,
    boolean_runs,
    compute_grid,
    spectral_flatness,
)
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Laughter burst rate, in bursts per second. Deliberately narrower than the
#: 3-8 Hz range laughter can span, because ordinary speech syllables occupy the
#: same range and the overlap is where the false positives live.
MIN_BURST_HZ = 4.0
MAX_BURST_HZ = 7.5

#: Shortest laughter candidate. Below this there is no rhythm to measure.
MIN_LAUGHTER_SECONDS = 0.45

#: Longest. A "laughter" region spanning most of a clip is almost always
#: rhythmic speech that the rhythm test latched onto.
MAX_LAUGHTER_SECONDS = 6.0

#: Fraction of the clip a candidate may cover. Beyond this the detector is
#: describing the speech itself, not an event inside it.
MAX_CLIP_FRACTION = 0.5

#: Window over which burst rhythm is measured.
ANALYSIS_WINDOW_SECONDS = 0.60

#: Minimum bursts inside a window for the pattern to count as a train.
MIN_BURSTS = 3

#: Maximum coefficient of variation of inter-burst intervals. Laughter is
#: metronomic; speech syllable timing is not. Tight, because this is the single
#: most discriminating feature available without a trained model.
MAX_INTERVAL_VARIATION = 0.22


class LaughterDetector:
    """Finds rhythmic voiced burst trains that may be laughter."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> list[AnalysisEvent]:
        """Find laughter candidates.

        Args:
            speech_spans: VAD speech regions. When given, only these are
                searched, which cuts false positives from noise considerably.
        """
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        if grid.frame_count < 10:
            return []

        rms_db = grid.rms_dbfs
        voice_band = band_energy_ratio(grid, 300.0, 3000.0)
        flatness = spectral_flatness(grid)

        # Bursts are local energy maxima standing clear of their surroundings.
        bursts = self._burst_frames(rms_db)

        window_frames = max(3, int(round(ANALYSIS_WINDOW_SECONDS / grid.hop_seconds)))
        burst_rate = self._burst_rate(bursts, window_frames, grid.hop_seconds)

        rhythmic = (burst_rate >= MIN_BURST_HZ) & (burst_rate <= MAX_BURST_HZ)

        # Laughter is voiced and vowel-like: energy in the vowel band, not
        # noise-like. This is what separates it from panting or rustling.
        #
        # Tested per *window*, over the energetic frames only, not per frame.
        # Laughter is bursts separated by gaps, and a near-silent gap is
        # spectrally flat by definition -- a per-frame test would fail on every
        # gap and chop the burst train into fragments too short to qualify.
        vowel_like = self._windowed_vowel_ratio(
            voice_band, flatness, rms_db, window_frames
        ) >= 0.60

        # High short-term energy variance is characteristic: laughter pulses,
        # whereas continuous speech is comparatively steady.
        pulsing = self._energy_variance(rms_db, window_frames) > 6.0

        candidate = rhythmic & vowel_like & pulsing
        if speech_spans is not None:
            candidate &= self._span_mask(speech_spans, grid)

        frame_seconds = grid.frame_length / grid.sample_rate
        events: list[AnalysisEvent] = []
        for start_frame, end_frame in boolean_runs(candidate):
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds
            duration = end - start
            if not (MIN_LAUGHTER_SECONDS <= duration <= MAX_LAUGHTER_SECONDS):
                continue
            if buffer.duration > 0 and duration / buffer.duration > MAX_CLIP_FRACTION:
                continue

            window = slice(start_frame, max(start_frame + 1, end_frame))
            onsets = np.flatnonzero(bursts[window])
            if onsets.size < MIN_BURSTS:
                continue

            # Laughter is metronomic: its "ha" syllables land at near-constant
            # intervals. Speech syllables fall in the same 3-8 Hz range but
            # their spacing varies with stress and word length, so interval
            # regularity is what separates a laugh from a fast phrase or a
            # repeated backchannel. It is not decisive -- a very even phrase
            # will still pass -- which is why confirmation is required.
            if True:
                intervals = np.diff(onsets).astype(np.float64)
                mean_interval = float(np.mean(intervals))
                if mean_interval <= 0:
                    continue
                variation = float(np.std(intervals)) / mean_interval
                if variation > MAX_INTERVAL_VARIATION:
                    continue

            events.append(
                AnalysisEvent(
                    type=EventType.POSSIBLE_LAUGHTER,
                    start=start,
                    end=min(end, buffer.duration),
                    confidence=self._confidence(
                        float(np.mean(burst_rate[window])),
                        float(np.mean(voice_band[window])),
                        end - start,
                    ),
                    note=(
                        "rhythmic voiced bursts. Laughter is [laughter] in V2 and is "
                        "never written as [hn] (section 8). A stutter, panting or "
                        "repeated backchannels can look identical here -- listen "
                        "before confirming."
                    ),
                )
            )
        return events

    # -------------------------------------------------------------- internals
    @staticmethod
    def _burst_frames(rms_db: np.ndarray, onset_db: float = 3.0) -> np.ndarray:
        """Frames where a burst of energy begins.

        Onsets rather than strict local maxima. A laughter syllable is an
        attack followed by a short plateau and decay, and a plateau has no
        single local maximum -- peak-picking finds nothing on it, or finds a
        ripple at an arbitrary point. A rising edge is present in either
        shape, so counting onsets measures the burst rhythm reliably.
        """
        if rms_db.size < 3:
            return np.zeros(rms_db.size, dtype=bool)
        finite = np.where(np.isfinite(rms_db), rms_db, -120.0)
        rise = np.zeros(finite.size, dtype=bool)
        rise[1:] = np.diff(finite) >= onset_db

        # Keep only the first frame of each rising run, so one attack counts
        # once however many frames it climbs over.
        first_of_run = rise.copy()
        first_of_run[1:] &= ~rise[:-1]

        # The onset must land somewhere meaningfully above the local floor, or
        # ripple in near-silence would register as a burst.
        valid = finite[np.isfinite(finite)]
        floor = float(np.percentile(valid, 25)) if valid.size else -120.0
        return first_of_run & (finite > floor + 4.0)

    @staticmethod
    def _burst_rate(
        bursts: np.ndarray, window_frames: int, hop_seconds: float
    ) -> np.ndarray:
        """Sliding-window burst rate in Hz."""
        if bursts.size == 0:
            return np.zeros(0)
        counts = _sliding_sum(bursts.astype(np.float64), window_frames)
        return counts / (window_frames * hop_seconds)

    @staticmethod
    def _windowed_vowel_ratio(
        voice_band: np.ndarray,
        flatness: np.ndarray,
        rms_db: np.ndarray,
        window_frames: int,
    ) -> np.ndarray:
        """Fraction of a window's *energetic* frames that look vowel-like.

        Restricting the denominator to energetic frames is the point: it asks
        "when this speaker was making sound, did it sound like a vowel?", which
        is answerable for a pulsed signal, instead of "was every frame a
        vowel?", which is not.
        """
        finite = np.where(np.isfinite(rms_db), rms_db, -120.0)
        valid = finite[finite > -120.0]
        floor = float(np.percentile(valid, 25)) if valid.size else -120.0
        loud = finite > floor + 6.0

        vowel_frame = (voice_band >= 0.30) & (flatness < 0.50) & loud
        numerator = _sliding_sum(vowel_frame.astype(np.float64), window_frames)
        denominator = _sliding_sum(loud.astype(np.float64), window_frames)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

    @staticmethod
    def _energy_variance(rms_db: np.ndarray, window_frames: int) -> np.ndarray:
        """Sliding standard deviation of frame energy, in dB."""
        finite = np.where(np.isfinite(rms_db), rms_db, -120.0)
        mean = _sliding_sum(finite, window_frames) / window_frames
        mean_sq = _sliding_sum(np.square(finite), window_frames) / window_frames
        return np.sqrt(np.maximum(0.0, mean_sq - np.square(mean)))

    @staticmethod
    def _span_mask(spans: list[tuple[float, float]], grid: FrameGrid) -> np.ndarray:
        mask = np.zeros(grid.frame_count, dtype=bool)
        for start, end in spans:
            mask[grid.time_to_frame(start) : grid.time_to_frame(end) + 1] = True
        return mask

    @staticmethod
    def _confidence(burst_rate: float, voice_band: float, duration: float) -> float:
        """Confidence capped at 0.70.

        A rhythm heuristic simply cannot be certain about laughter, and treating
        a high number as truth is exactly what section 56.22 forbids.
        """
        score = 0.0
        # Closest to the middle of the laughter burst range scores best.
        centre = (MIN_BURST_HZ + MAX_BURST_HZ) / 2
        score += 0.20 * max(0.0, 1.0 - abs(burst_rate - centre) / centre)
        score += 0.15 * min(1.0, max(0.0, (voice_band - 0.30) / 0.5))
        score += 0.10 * min(1.0, duration / 1.5)
        return float(np.clip(0.30 + score, 0.30, 0.70))


def _sliding_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Centred sliding-window sum, edge-padded to preserve length."""
    if values.size == 0:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    cumulative = np.concatenate([[0.0], np.cumsum(padded)])
    return cumulative[window:] - cumulative[:-window]


def detect_laughter(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[AnalysisEvent]:
    return LaughterDetector(config).detect(buffer, speech_spans=speech_spans)
