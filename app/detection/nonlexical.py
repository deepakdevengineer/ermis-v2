"""Non-lexical vocal sound detection (CLAUDE.md sections 8, 39, 46).

Finds candidates for [hn], which in V2 covers cough, sigh, throat-clear,
sneeze, lip-smack, yawn, cry, groan, humming, and a noticeable sharp
inhalation or exhalation.

Two rules from section 8 are enforced structurally rather than left to the
annotator to remember:

  - Laughter is NOT [hn]. Laughter has its own tag, so this detector never
    proposes [hn] for something the laughter detector claimed; overlapping
    candidates are dropped in favour of laughter.
  - The same sound repeated in immediate succession takes ONE tag. Adjacent
    candidates within `MERGE_GAP_SECONDS` are merged into a single event, so
    two coughs produce one [hn] rather than "[hn] [hn]" (case B, section 55).

Section 8 also excludes ordinary breathing from [hn] -- only a *noticeable*
sharp inhalation counts. That is why the energy threshold is relative to the
speaker's own level rather than absolute: quiet breath between words does not
qualify.

Honest limits: this distinguishes "a non-speech vocal-ish sound happened" from
"speech", not one sound from another. It cannot tell a cough from a sneeze from
a desk thump, and it does not try -- [hn] covers them all, and a desk thump is
not [hn] at all, so confirmation matters.
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

#: Non-lexical sounds are short events, not sustained regions.
MIN_HN_SECONDS = 0.06
MAX_HN_SECONDS = 2.50

#: Adjacent candidates closer than this merge into one event, implementing
#: section 8's "same sound in immediate succession uses one tag".
MERGE_GAP_SECONDS = 0.45

#: How far above the local floor a sound must sit to be "noticeable", which
#: keeps ordinary breathing out (section 8).
NOTICEABLE_DB = 8.0


class NonLexicalDetector:
    """Finds candidates for [hn]."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(
        self,
        buffer: AudioBuffer,
        grid: FrameGrid | None = None,
        speech_spans: list[tuple[float, float]] | None = None,
        laughter_events: list[AnalysisEvent] | None = None,
    ) -> list[AnalysisEvent]:
        """Find [hn] candidates.

        Args:
            speech_spans: VAD speech regions. Candidates inside sustained
                speech are suppressed, since a non-lexical sound interrupts
                speech rather than being part of it.
            laughter_events: laughter candidates. Overlapping [hn] candidates
                are dropped, enforcing section 8's rule that laughter is never
                written as [hn].
        """
        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        if grid.frame_count < 4:
            return []

        rms_db = grid.rms_dbfs
        finite = rms_db[np.isfinite(rms_db)]
        floor = float(np.percentile(finite, 25)) if finite.size else -120.0

        flatness = spectral_flatness(grid)
        centroid = spectral_centroid(grid)
        voice_band = band_energy_ratio(grid, 300.0, 3000.0)
        low_band = band_energy_ratio(grid, 0.0, 300.0)

        noticeable = rms_db > floor + NOTICEABLE_DB

        # Three acoustic shapes cover most of what section 8 lists.
        #
        # Plosive/breathy: cough, sneeze, throat-clear, sharp breath. Broadband
        # and noise-like, with little harmonic structure.
        plosive = (flatness > 0.40) & (voice_band < 0.45)
        # Low, dull energy: sigh, groan, yawn. Concentrated below the vowel
        # band with a low spectral centroid.
        dull = (low_band > 0.35) & (centroid < 700.0)
        # Lip-smack and click-like mouth noise: bright and very noise-like.
        smack = (flatness > 0.55) & (centroid > 2500.0)

        candidate = noticeable & (plosive | dull | smack)

        if speech_spans is not None:
            # A non-lexical sound interrupts speech; it is not the speech
            # itself. Suppress candidates well inside a long speech region.
            candidate &= ~self._interior_speech_mask(speech_spans, grid)

        frame_seconds = grid.frame_length / grid.sample_rate
        raw: list[tuple[float, float]] = []
        for start_frame, end_frame in boolean_runs(candidate):
            start = grid.frame_to_time(start_frame)
            end = grid.frame_to_time(end_frame - 1) + frame_seconds
            if MIN_HN_SECONDS <= end - start <= MAX_HN_SECONDS:
                raw.append((start, min(end, buffer.duration)))

        merged = self._merge_successive(raw)
        # The duration bound has to be re-applied after merging. Two coughs
        # merge into one event, but if the merge spans longer than a
        # non-lexical sound plausibly lasts, the run was never a discrete
        # sound in the first place -- it is sustained audio, which [hn] does
        # not describe.
        merged = [(a, b) for a, b in merged if b - a <= MAX_HN_SECONDS]
        merged = self._drop_overlapping_laughter(merged, laughter_events or [])

        events: list[AnalysisEvent] = []
        for start, end in merged:
            a, b = grid.time_to_frame(start), grid.time_to_frame(end)
            window = slice(a, max(a + 1, b))
            events.append(
                AnalysisEvent(
                    type=EventType.POSSIBLE_HN,
                    start=start,
                    end=end,
                    confidence=self._confidence(
                        float(np.max(rms_db[window]) - floor),
                        float(np.mean(flatness[window])),
                    ),
                    note=(
                        "possible non-lexical vocal sound. [hn] covers cough, sigh, "
                        "throat-clear, sneeze, lip-smack, yawn, cry, groan and "
                        "humming, but NOT laughter (that is [laughter]) and not "
                        "ordinary breathing (section 8)."
                    ),
                )
            )
        return events

    # -------------------------------------------------------------- internals
    @staticmethod
    def _merge_successive(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Merge near-adjacent candidates into one event.

        This is section 8's rule in code: two coughs in immediate succession
        are one [hn], not "[hn] [hn]". The validator independently reports
        "[hn] [hn]" in a transcription as an error, so the rule holds whether
        or not the annotator used this suggestion.
        """
        if not spans:
            return []
        merged = [list(spans[0])]
        for start, end in spans[1:]:
            if start - merged[-1][1] <= MERGE_GAP_SECONDS:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _drop_overlapping_laughter(
        spans: list[tuple[float, float]], laughter: list[AnalysisEvent]
    ) -> list[tuple[float, float]]:
        """Remove candidates overlapping a laughter candidate (section 8)."""
        if not laughter:
            return spans
        kept = []
        for start, end in spans:
            if any(not (end <= ev.start or start >= ev.end) for ev in laughter):
                continue
            kept.append((start, end))
        return kept

    @staticmethod
    def _interior_speech_mask(
        spans: list[tuple[float, float]], grid: FrameGrid, edge: float = 0.15
    ) -> np.ndarray:
        """Frames well inside a speech region, excluding its edges.

        Edges are left available because a cough or breath most often lands at
        the boundary of an utterance.
        """
        mask = np.zeros(grid.frame_count, dtype=bool)
        for start, end in spans:
            if end - start <= 2 * edge:
                continue
            a = grid.time_to_frame(start + edge)
            b = grid.time_to_frame(end - edge)
            if b > a:
                mask[a:b] = True
        return mask

    @staticmethod
    def _confidence(headroom_db: float, flatness: float) -> float:
        """Confidence capped at 0.65.

        The detector knows a non-speech sound occurred; it does not know it was
        vocal. A desk thump and a cough look much alike from the spectrum, and
        only one of them is [hn].
        """
        score = 0.0
        score += min(0.20, max(0.0, (headroom_db - NOTICEABLE_DB) / 40.0))
        score += min(0.15, max(0.0, (flatness - 0.35) / 2.0))
        return float(np.clip(0.30 + score, 0.30, 0.65))


def detect_non_lexical(
    buffer: AudioBuffer,
    config: DetectionConfig | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
    laughter_events: list[AnalysisEvent] | None = None,
) -> list[AnalysisEvent]:
    return NonLexicalDetector(config).detect(
        buffer, speech_spans=speech_spans, laughter_events=laughter_events
    )
