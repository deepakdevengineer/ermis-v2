"""Audio quality analysis (CLAUDE.md sections 9, 10, 11, 12, 45).

Measures the conditions behind three whole-file tags:

  <na>  no audio signal at all              (section 11)
  <ga>  recording too damaged to evaluate   (section 10)
  <ns>  audio present but no human speech   (section 12)

These are *measurements and candidates*, never decisions. Section 10 defines
<ga> as "distorted, damaged, or skipping enough to compromise evaluation", and
whether evaluation is compromised is a human judgement about intelligibility --
no meter can settle it. So this module reports what it measured, proposes a
candidate, and leaves the call to the annotator.

Two traps it deliberately avoids:
  - Section 10: intentionally pitch-shifted audio (used for privacy) is NOT
    <ga>. Pitch shifting leaves harmonic structure intact, so the clip does not
    trip the distortion measures here.
  - Section 12: <ns> does not apply when another tag does. A clip of pure
    laughter is [laughter], not <ns>, so the candidate is offered only when no
    non-lexical sound was detected either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import (
    FrameGrid,
    compute_grid,
    spectral_flatness,
)
from app.config import DetectionConfig
from app.ermis import tags as T
from app.models.event import AnalysisEvent, EventType

#: Peak dBFS at or below which a file carries no signal at all.
NO_SIGNAL_DBFS = -90.0

#: Fraction of samples at full scale above which clipping is severe enough to
#: be a candidate for <ga> rather than a passing artifact.
SEVERE_CLIP_RATIO = 0.01

#: Fraction of interior frames that must be digitally silent for the recording
#: to look like it is skipping. This counts *exact* digital silence, which a
#: normal speech pause never contains -- a real pause carries room noise -- so
#: the threshold can be low without flagging ordinary turn-taking gaps.
DROPOUT_RATIO = 0.15

#: Spectral flatness above which a frame is noise-like rather than tonal.
NOISE_FLATNESS = 0.55

#: How far below the loud level a frame must sit to count as a noise sample,
#: and what fraction of frames must qualify before SNR is reported at all.
MIN_NOISE_GAP_DB = 15.0
MIN_NOISE_FRAME_FRACTION = 0.05


@dataclass
class QualityReport:
    """What was measured, and what it might mean."""

    duration: float
    peak_dbfs: float
    rms_dbfs: float
    #: Fraction of samples at or beyond full scale.
    clipping_ratio: float
    #: Fraction of frames that are digitally silent mid-file.
    dropout_ratio: float
    #: Fraction of frames that are noise-like.
    noise_ratio: float
    #: Estimated signal-to-noise ratio in dB, or None when undeterminable.
    snr_db: float | None
    has_signal: bool
    #: Candidate whole-file tags, most likely first. Advisory only.
    candidates: list[str] = field(default_factory=list)
    #: Plain-language reasons, shown next to each candidate in the UI.
    notes: list[str] = field(default_factory=list)

    @property
    def is_silent(self) -> bool:
        return not self.has_signal

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "peak_dbfs": None if not np.isfinite(self.peak_dbfs) else round(self.peak_dbfs, 2),
            "rms_dbfs": None if not np.isfinite(self.rms_dbfs) else round(self.rms_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 5),
            "dropout_ratio": round(self.dropout_ratio, 4),
            "noise_ratio": round(self.noise_ratio, 4),
            "snr_db": None if self.snr_db is None else round(self.snr_db, 2),
            "has_signal": self.has_signal,
            "candidates": list(self.candidates),
            "notes": list(self.notes),
        }


class QualityAnalyzer:
    """Measures recording health and proposes whole-file candidates."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------ entry
    def analyse(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        has_speech: bool | None = None,
        has_non_lexical: bool = False,
    ) -> QualityReport:
        """Measure quality.

        Args:
            has_speech: the VAD's verdict, when available. Needed for the <ns>
                candidate, which is about the absence of *speech*, not of sound.
            has_non_lexical: whether a laughter or [hn] candidate was found.
                When True, <ns> is not offered -- section 12 forbids <ns> when
                another tag applies.
        """
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        samples = buffer.samples

        peak_dbfs = buffer.peak_dbfs
        rms_dbfs = buffer.rms_dbfs
        has_signal = peak_dbfs > NO_SIGNAL_DBFS

        clipping = self._clipping_ratio(samples)
        dropout = self._dropout_ratio(grid)
        noise = self._noise_ratio(grid)
        snr = self._snr_db(grid)

        report = QualityReport(
            duration=buffer.duration,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            clipping_ratio=clipping,
            dropout_ratio=dropout,
            noise_ratio=noise,
            snr_db=snr,
            has_signal=has_signal,
        )
        self._propose(report, has_speech=has_speech, has_non_lexical=has_non_lexical)
        return report

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        has_speech: bool | None = None,
        has_non_lexical: bool = False,
    ) -> list[AnalysisEvent]:
        """Whole-file quality conditions as analysis events.

        NO_AUDIO is emitted as a measured fact -- digital silence is not a
        judgement call. GARBLED_AUDIO and NO_SPEECH are emitted as events the
        annotator must confirm, because both hinge on intelligibility.
        """
        report = self.analyse(buffer, grid, has_speech, has_non_lexical)
        events: list[AnalysisEvent] = []
        duration = buffer.duration

        if not report.has_signal:
            events.append(
                AnalysisEvent(
                    type=EventType.NO_AUDIO,
                    start=0.0,
                    end=duration,
                    confidence=1.0,
                    note="peak is at or below the digital-silence floor",
                )
            )
            return events

        if T.GARBLED in report.candidates:
            events.append(
                AnalysisEvent(
                    type=EventType.GARBLED_AUDIO,
                    start=0.0,
                    end=duration,
                    confidence=self._garbled_confidence(report),
                    note="; ".join(report.notes),
                )
            )

        if T.NO_SPEECH in report.candidates:
            events.append(
                AnalysisEvent(
                    type=EventType.NO_SPEECH,
                    start=0.0,
                    end=duration,
                    confidence=0.6,
                    note="signal present but no speech detected",
                )
            )
        return events

    # -------------------------------------------------------------- measures
    @staticmethod
    def _clipping_ratio(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        # 0.999 rather than 1.0: a clipped 16-bit sample decodes just under
        # full scale, so an exact comparison would miss real clipping.
        return float(np.count_nonzero(np.abs(samples) >= 0.999) / samples.size)

    @staticmethod
    def _dropout_ratio(grid: FrameGrid) -> float:
        """Fraction of interior frames that are digitally silent.

        Leading and trailing silence is excluded: quiet before speech starts is
        normal, whereas silence *inside* the recording suggests a dropout or a
        skipping file.
        """
        if grid.frame_count == 0:
            return 0.0
        silent = grid.peak <= 1e-6
        voiced = np.flatnonzero(~silent)
        if voiced.size == 0:
            return 0.0
        interior = silent[voiced[0] : voiced[-1] + 1]
        return float(np.count_nonzero(interior) / interior.size) if interior.size else 0.0

    @staticmethod
    def _noise_ratio(grid: FrameGrid) -> float:
        if grid.frame_count == 0:
            return 0.0
        active = grid.rms > 1e-5
        if not np.any(active):
            return 0.0
        return float(np.mean(spectral_flatness(grid)[active] > NOISE_FLATNESS))

    @staticmethod
    def _snr_db(grid: FrameGrid) -> float | None:
        """SNR from the gap between loud frames and the noise floor.

        Returns None when the file has no measurably quiet frames. Without a
        noise sample there is nothing to compare the signal against, and the
        loud-minus-quiet gap degenerates into the *dynamic range of the speech
        itself*: wall-to-wall connected speech shows a 5 dB spread and would be
        misreported as 5 dB SNR, proposing <ga> for a perfectly clean
        recording.
        """
        finite = grid.rms_dbfs[np.isfinite(grid.rms_dbfs)]
        if finite.size < 10:
            return None
        loud = float(np.percentile(finite, 90))
        quiet = float(np.percentile(finite, 10))
        # At least 5% of frames must sit well below the loud level for the
        # low percentile to represent noise rather than more speech.
        noise_frames = float(np.mean(finite <= loud - MIN_NOISE_GAP_DB))
        if noise_frames < MIN_NOISE_FRAME_FRACTION:
            return None
        return loud - quiet

    # ------------------------------------------------------------- proposals
    def _propose(
        self,
        report: QualityReport,
        has_speech: bool | None,
        has_non_lexical: bool,
    ) -> None:
        """Fill in candidates and notes. Order is most-likely first."""
        if not report.has_signal:
            report.candidates.append(T.NO_AUDIO)
            report.notes.append(
                f"peak {report.peak_dbfs:.0f} dBFS is at or below the digital-silence "
                f"floor of {NO_SIGNAL_DBFS:.0f} dBFS"
            )
            return

        # Objective damage: clipping and mid-file digital silence indicate a
        # broken recording regardless of whether speech was found.
        garbled_reasons: list[str] = []
        if report.clipping_ratio >= SEVERE_CLIP_RATIO:
            garbled_reasons.append(
                f"{report.clipping_ratio * 100:.1f}% of samples are clipped at full scale"
            )
        if report.dropout_ratio >= DROPOUT_RATIO:
            garbled_reasons.append(
                f"{report.dropout_ratio * 100:.0f}% of the interior is digitally "
                "silent, which looks like dropouts or a skipping file"
            )

        # Poor SNR and noise-like spectra only point to <ga> when there is
        # speech to be compromised. Steady broadband noise with no speech is
        # ambient room noise, which section 12 gives as an <ns> example
        # ("faint white noise") and section 9 explicitly excludes from
        # [artifact]. Calling it <ga> would be wrong.
        if has_speech:
            if report.snr_db is not None and report.snr_db < 6.0:
                garbled_reasons.append(
                    f"speech was detected but the signal-to-noise ratio is only "
                    f"{report.snr_db:.1f} dB"
                )
            if report.noise_ratio > 0.85:
                garbled_reasons.append(
                    f"speech was detected but {report.noise_ratio * 100:.0f}% of "
                    "active frames are noise-like rather than tonal"
                )

        if garbled_reasons:
            report.candidates.append(T.GARBLED)
            report.notes.extend(garbled_reasons)
            report.notes.append(
                "<ga> applies only if this compromises evaluation. If speech can "
                "still be transcribed around the noise, use [artifact] instead "
                "(section 9). Pitch-shifted privacy audio is never <ga> (section 10)."
            )

        # Section 12: <ns> only when there is no speech AND no other tag
        # applies. A clip of only laughter or only a cough is [laughter] or
        # [hn], not <ns>.
        if has_speech is False and not has_non_lexical:
            report.candidates.append(T.NO_SPEECH)
            report.notes.append(
                "signal is present but no speech was detected. <ns> applies only if "
                "no other tag fits: check for laughter or a non-lexical sound first "
                "(section 12)."
            )
            if report.noise_ratio > 0.85:
                report.notes.append(
                    "the signal is broadband noise. Steady room noise or faint white "
                    "noise is <ns> (section 12). Only consider <ga> if this is "
                    "microphone or recording distortion rather than ambient noise."
                )
        elif has_speech is False and has_non_lexical:
            report.notes.append(
                "no speech detected, but a non-lexical sound was found. Section 12 "
                "forbids <ns> when another tag applies; consider [laughter] or [hn]."
            )

    @staticmethod
    def _garbled_confidence(report: QualityReport) -> float:
        """Confidence that the recording is compromised.

        Capped well below certainty: intelligibility is a human judgement, and
        a high number here must not read as a verdict (section 56.22).
        """
        score = 0.0
        if report.clipping_ratio >= SEVERE_CLIP_RATIO:
            score += min(0.3, report.clipping_ratio * 6)
        if report.dropout_ratio >= DROPOUT_RATIO:
            score += min(0.3, report.dropout_ratio * 0.8)
        if report.snr_db is not None and report.snr_db < 6.0:
            score += (6.0 - max(0.0, report.snr_db)) / 20.0
        if report.noise_ratio > 0.85:
            score += 0.15
        return float(np.clip(0.35 + score, 0.35, 0.85))


def analyse_quality(
    buffer: AudioBuffer, config: DetectionConfig | None = None
) -> QualityReport:
    return QualityAnalyzer(config).analyse(buffer)
