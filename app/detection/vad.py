"""Voice activity detection (CLAUDE.md sections 45, 46).

Produces SPEECH regions for navigation and QA. Speech regions drive the
timeline, the speaker lanes and the "jump to next utterance" action; they never
produce transcription text.

Approach: an adaptive noise floor plus a two-threshold (hysteresis) decision on
three cheap, complementary features. Deterministic, no model weights, no
network.

A fixed dBFS threshold fails badly across the range of recordings this workflow
sees -- a quiet far-field clip and a hot close-mic clip need different cut-offs
-- so the floor is estimated per file from a low percentile of frame energy.

Honest limits, which is why every consumer treats this as assistive:
  - Harmonic non-speech at speech level -- music, singing, engine hum, a
    babbling crowd -- can be marked as speech. The tonality gate rejects
    broadband noise, not pitched sound.
  - Very quiet speech under a loud noise floor can be missed.
  - It detects *voice activity*, not *speech content*: it cannot tell who is
    speaking or what was said, and never tries.

If `silero_vad` or `torch` happens to be installed locally, `SileroVad` can be
used instead; it is never required and is never downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    band_energy_ratio,
    boolean_runs,
    compute_grid,
    merge_spans,
    smooth_flags,
    spectral_flatness,
)
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Spectral flatness at or above which a frame is broadband noise rather than
#: harmonic (voiced) speech. Shared with app.audio.quality.
NOISE_FLATNESS = 0.55

#: Absolute floor below which audio is treated as carrying no speech at all.
#:
#: Deliberately far lower than DetectionConfig.silence_dbfs_threshold, which
#: governs *silence* detection. Reusing that -40 dBFS value here would discard
#: speech in any genuinely quiet recording -- a far-field or low-gain clip can
#: sit entirely below it -- which is exactly what the adaptive noise floor
#: exists to handle. This floor only rules out audio that is effectively
#: digital silence.
ABSOLUTE_FLOOR_DBFS = -70.0


@dataclass(frozen=True)
class VadResult:
    """Per-frame speech decision plus the derived regions."""

    grid: FrameGrid
    #: Per-frame speech flag after smoothing.
    flags: np.ndarray
    #: (start, end) second spans of speech.
    speech: list[tuple[float, float]]
    #: Estimated noise floor in dBFS.
    noise_floor_dbfs: float
    #: Threshold actually used, in dBFS.
    threshold_dbfs: float
    #: Fraction of the file marked as speech.
    speech_ratio: float

    @property
    def has_speech(self) -> bool:
        return bool(self.speech)


class EnergyVad:
    """Adaptive-threshold voice activity detector."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------ entry
    def analyse(self, buffer: AudioBuffer, grid: FrameGrid | None = None) -> VadResult:
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        if grid.frame_count == 0:
            return VadResult(grid, np.zeros(0, bool), [], -np.inf, -np.inf, 0.0)

        rms_db = grid.rms_dbfs
        noise_floor = self._noise_floor(rms_db)

        # Speech must clear this file's own noise floor by a margin, and also
        # clear an absolute floor so that essentially-silent audio never
        # produces speech regions however its internal levels scale.
        margin_threshold = noise_floor + self._margin_db(rms_db)
        threshold = max(margin_threshold, ABSOLUTE_FLOOR_DBFS)

        loud = rms_db >= threshold

        # Speech energy concentrates in 200-4000 Hz. This rejects low-frequency
        # rumble and high-frequency hiss that clear the energy threshold.
        voice_band = band_energy_ratio(grid, 200.0, 4000.0)
        band_ok = voice_band >= 0.25

        # Very high zero-crossing rate with little voice-band energy indicates
        # noise or a fricative burst rather than voiced speech. Keeping the
        # band condition here avoids discarding genuine unvoiced consonants.
        noisy = (grid.zcr > 0.45) & (voice_band < 0.15)

        # Voiced speech is harmonic, so its spectrum is peaky (low flatness).
        # Broadband noise -- hiss, static, air conditioning -- is flat and
        # would otherwise clear the energy and voice-band tests, because a
        # flat spectrum puts plenty of energy in 200-4000 Hz.
        #
        # This gate only has to catch the *onset* of a region: unvoiced
        # consonants are flat too, but they are short and the hysteresis below
        # carries an already-open region straight through them.
        tonal = spectral_flatness(grid) < NOISE_FLATNESS

        candidate = loud & band_ok & ~noisy & tonal
        flags = self._hysteresis(rms_db, candidate, threshold)

        window = max(1, int(round(0.05 / grid.hop_seconds)))
        flags = smooth_flags(flags, window)

        spans = merge_spans(
            self._spans(flags, grid), gap=self.config.min_silence_duration
        )
        spans = [s for s in spans if s[1] - s[0] >= self.config.min_speech_duration]

        ratio = float(np.count_nonzero(flags) / flags.size) if flags.size else 0.0
        return VadResult(
            grid=grid,
            flags=flags,
            speech=spans,
            noise_floor_dbfs=float(noise_floor),
            threshold_dbfs=float(threshold),
            speech_ratio=ratio,
        )

    def detect(
        self, buffer: AudioBuffer, grid: FrameGrid | None = None
    ) -> list[AnalysisEvent]:
        """Speech regions as analysis events."""
        result = self.analyse(buffer, grid)
        return [
            AnalysisEvent(
                type=EventType.SPEECH,
                start=start,
                end=end,
                confidence=self._region_confidence(result, start, end),
            )
            for start, end in result.speech
        ]

    # -------------------------------------------------------------- internals
    @staticmethod
    def _noise_floor(rms_db: np.ndarray) -> float:
        """Estimate the noise floor from a low percentile of frame energy.

        The 10th percentile is robust for the usual case of speech with gaps.
        For a file that is almost entirely speech there are few quiet frames,
        so the percentile drifts upward; clamping to a floor relative to the
        median keeps the threshold from climbing into the speech itself.
        """
        finite = rms_db[np.isfinite(rms_db)]
        if finite.size == 0:
            return -np.inf
        floor = float(np.percentile(finite, 10))
        median = float(np.median(finite))
        return min(floor, median - 6.0)

    @staticmethod
    def _margin_db(rms_db: np.ndarray) -> float:
        """How far above the noise floor speech must sit.

        Scaled by the file's dynamic range: a clip with a wide range gets a
        wider margin, a compressed clip a narrower one, so neither over- nor
        under-detects.
        """
        finite = rms_db[np.isfinite(rms_db)]
        if finite.size == 0:
            return 10.0
        spread = float(np.percentile(finite, 95) - np.percentile(finite, 10))
        return float(np.clip(spread * 0.35, 6.0, 18.0))

    @staticmethod
    def _hysteresis(
        rms_db: np.ndarray, candidate: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Two-threshold tracking so speech is not chopped at every dip.

        A region starts only when the strict condition is met, then continues
        while energy stays within 6 dB of the threshold. This keeps a word's
        quiet tail attached to the word.
        """
        release = threshold - 6.0
        flags = np.zeros(candidate.size, dtype=bool)
        active = False
        for i in range(candidate.size):
            if active:
                active = rms_db[i] >= release
            else:
                active = bool(candidate[i])
            flags[i] = active
        return flags

    @staticmethod
    def _spans(flags: np.ndarray, grid: FrameGrid) -> list[tuple[float, float]]:
        frame_seconds = grid.frame_length / grid.sample_rate
        spans = []
        for start, end in boolean_runs(flags):
            spans.append(
                (grid.frame_to_time(start), grid.frame_to_time(end - 1) + frame_seconds)
            )
        return spans

    def _region_confidence(
        self, result: VadResult, start: float, end: float
    ) -> float:
        """Confidence from how far the region sits above the threshold.

        Deliberately conservative. Confidence is a display value for the
        annotator, never a licence to skip confirmation (section 56.22).
        """
        grid = result.grid
        a, b = grid.time_to_frame(start), grid.time_to_frame(end)
        window = grid.rms_dbfs[a : max(a + 1, b)]
        if window.size == 0 or not np.isfinite(result.threshold_dbfs):
            return 0.5
        headroom = float(np.median(window) - result.threshold_dbfs)
        return float(np.clip(0.5 + headroom / 24.0, 0.3, 0.95))


class SileroVad:
    """Optional wrapper for a locally installed Silero VAD.

    Never required and never downloaded. `available` is False unless the
    package and its weights are already present on this machine, which keeps
    the privacy guarantee in section 53 intact.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        try:
            from silero_vad import load_silero_vad  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(
        self, buffer: AudioBuffer, grid: FrameGrid | None = None
    ) -> list[AnalysisEvent]:
        if not self.available:
            raise RuntimeError(
                "Silero VAD is not installed locally. EnergyVad is the default "
                "and needs no model."
            )
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore

        if self._model is None:
            self._model = load_silero_vad()
        stamps = get_speech_timestamps(
            buffer.samples, self._model, sampling_rate=buffer.sample_rate
        )
        return [
            AnalysisEvent(
                type=EventType.SPEECH,
                start=s["start"] / buffer.sample_rate,
                end=s["end"] / buffer.sample_rate,
                confidence=0.9,
            )
            for s in stamps
        ]


def detect_speech(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    grid: FrameGrid | None = None,
) -> list[AnalysisEvent]:
    """Detect speech regions with the default detector."""
    return EnergyVad(config).detect(buffer, grid)
