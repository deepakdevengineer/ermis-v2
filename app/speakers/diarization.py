"""Speaker diarization assistance (CLAUDE.md sections 17, 18A, 46).

Groups speech segments by voice similarity to suggest how many speakers there
are and where each one talks.

Method: per-segment MFCC-like cepstral statistics, then deterministic
agglomerative clustering with average linkage. No random initialisation, no
model weights, no network -- the same audio always produces the same result,
which section 44's determinism requirement and the audit trail both need.

Speaker numbering follows FIRST APPEARANCE in the audio (section 18A.2), never
loudness, gender, confidence or cluster size (section 18A.2's explicit
prohibition).

Honest limits, and they are the reason section 18A.11 exists:
  - Cluster count is an estimate. Similar-sounding voices merge; one person
    whose voice changes (shouting, then whispering) can split.
  - Short segments carry too little information to place reliably.
  - It cannot identify anyone, and it never guesses gender or nativity --
    those stay `unknown` until a human sets them (sections 18A.3, 18A.4).

The output is a suggestion with a confidence figure. The annotator can merge,
split, add or remove speakers, and `human_edited` protects their decisions from
any later automatic pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import FrameGrid, compute_grid
from app.config import DetectionConfig
from app.models.event import AnalysisEvent, EventType

#: Segments shorter than this carry too little voice information to cluster.
MIN_SEGMENT_SECONDS = 0.35

#: Number of mel-frequency cepstral coefficients used per segment.
N_CEPSTRA = 13

#: Mel filterbank size.
N_MEL_FILTERS = 26

#: Cosine-distance threshold below which two clusters are the same speaker.
#:
#: CALIBRATION CAVEAT: chosen against synthetic fixtures, where a single
#: speaker's segments sit below 0.001 and two speakers sit above 0.19 -- an
#: unrealistically wide gap. Real recordings have far more within-speaker
#: variance (shouting then whispering, moving off-mic, different phonetic
#: content), so this value is a starting point, not a validated constant.
#:
#: Erring toward splitting is deliberate. An extra cluster is visible in the
#: speaker panel and the annotator can merge it in one action; two people
#: silently collapsed into one speaker is far harder to notice, and it corrupts
#: every speaker tag in the transcription.
MERGE_DISTANCE = 0.08

#: Hard ceiling on suggested speakers. Section 22 says a large group is treated
#: as one speaker, so a suggestion of dozens is always wrong.
MAX_SUGGESTED_SPEAKERS = 8


@dataclass
class SpeakerSegment:
    """One speech segment with its assigned speaker."""

    start: float
    end: float
    speaker_id: str
    confidence: float
    features: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Suggested speaker structure. Assistive only (section 18A.11)."""

    segments: list[SpeakerSegment]
    speaker_count: int
    #: Overall confidence in the speaker *count*, shown next to it in the UI.
    confidence: float
    #: Mean pairwise distance between clusters; low values mean the voices were
    #: hard to separate and the count is doubtful.
    separation: float = 0.0

    @property
    def speaker_ids(self) -> list[str]:
        """Speaker ids in first-appearance order."""
        seen: list[str] = []
        for segment in sorted(self.segments, key=lambda s: s.start):
            if segment.speaker_id not in seen:
                seen.append(segment.speaker_id)
        return seen

    def timeline(self, speaker_id: str) -> list[tuple[float, float]]:
        return [
            (s.start, s.end) for s in self.segments if s.speaker_id == speaker_id
        ]

    def first_seen(self, speaker_id: str) -> float | None:
        times = [s.start for s in self.segments if s.speaker_id == speaker_id]
        return min(times) if times else None

    def last_seen(self, speaker_id: str) -> float | None:
        times = [s.end for s in self.segments if s.speaker_id == speaker_id]
        return max(times) if times else None

    def speaker_confidence(self, speaker_id: str) -> float:
        values = [s.confidence for s in self.segments if s.speaker_id == speaker_id]
        return float(np.mean(values)) if values else 0.0

    def to_events(self) -> list[AnalysisEvent]:
        """Speech events carrying speaker attribution, plus change markers."""
        events: list[AnalysisEvent] = []
        ordered = sorted(self.segments, key=lambda s: s.start)
        for segment in ordered:
            events.append(
                AnalysisEvent(
                    type=EventType.SPEECH,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id,
                )
            )
        for previous, current in zip(ordered, ordered[1:]):
            if previous.speaker_id != current.speaker_id:
                events.append(
                    AnalysisEvent(
                        type=EventType.SPEAKER_CHANGE,
                        start=previous.end,
                        end=current.start,
                        confidence=min(previous.confidence, current.confidence),
                        speaker_id=current.speaker_id,
                    )
                )
        return sorted(events, key=lambda e: (e.start, e.end))

    def render_timeline(
        self, width: int = 60, duration: float | None = None, block: str = "#"
    ) -> str:
        """Speaker timeline in the style of CLAUDE.md section 18A.7.

        Defaults to plain ASCII. The section 18A.7 example uses box-drawing
        characters, but this is a Windows application and the default console
        code page (cp1252) cannot encode them -- printing them raises
        UnicodeEncodeError. Pass `block="\\u2588"` where the output is known to
        be UTF-8, such as a log file or the Qt UI.
        """
        total = duration or max((s.end for s in self.segments), default=0.0)
        if total <= 0:
            return "(no speech)"
        rule = "-" if block == "#" else "─"
        lines = [f"00:00 {rule * max(1, width - 12)} {_ts(total)}", ""]
        for speaker_id in self.speaker_ids:
            row = [" "] * width
            for start, end in self.timeline(speaker_id):
                a = int(start / total * (width - 1))
                b = max(a, int(end / total * (width - 1)))
                for i in range(a, min(width, b + 1)):
                    row[i] = block
            lines.append(f"{speaker_id.upper():4s}{''.join(row)}")
        return "\n".join(lines)


class Diarizer:
    """Deterministic agglomerative speaker clustering."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------ entry
    def diarize(
        self,
        buffer: AudioBuffer,
        speech_spans: list[tuple[float, float]],
        grid: FrameGrid | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        usable = [
            (start, end)
            for start, end in speech_spans
            if end - start >= MIN_SEGMENT_SECONDS
        ]
        if not usable:
            return DiarizationResult([], 0, 0.0)

        grid = grid or compute_grid(buffer.samples, buffer.sample_rate)
        features = np.vstack(
            [self._segment_features(buffer, start, end) for start, end in usable]
        )

        if len(usable) == 1:
            return DiarizationResult(
                segments=[
                    SpeakerSegment(usable[0][0], usable[0][1], "s1", 0.5, features[0])
                ],
                speaker_count=1,
                confidence=0.5,
            )

        labels, separation = self._cluster(
            features, max_speakers or MAX_SUGGESTED_SPEAKERS
        )
        speaker_map = self._number_by_first_appearance(usable, labels)

        segments = [
            SpeakerSegment(
                start=start,
                end=end,
                speaker_id=speaker_map[label],
                confidence=self._segment_confidence(
                    features, labels, index, separation
                ),
                features=features[index],
            )
            for index, ((start, end), label) in enumerate(zip(usable, labels))
        ]
        count = len(set(labels))
        return DiarizationResult(
            segments=segments,
            speaker_count=count,
            confidence=self._count_confidence(count, separation, len(usable)),
            separation=separation,
        )

    # ---------------------------------------------------------------- features
    def _segment_features(
        self, buffer: AudioBuffer, start: float, end: float
    ) -> np.ndarray:
        """Cepstral statistics summarising one segment's voice character.

        Mean and standard deviation of MFCCs, which capture vocal-tract shape
        and are the standard cheap speaker descriptor.
        """
        samples = buffer.slice_seconds(start, end)
        mfcc = self._mfcc(samples, buffer.sample_rate)
        if mfcc.size == 0:
            return np.zeros((N_CEPSTRA - 1) * 2)

        # Drop C0. It is the frame's overall log-energy, not a property of the
        # voice, and it is an order of magnitude larger than the coefficients
        # that do describe vocal-tract shape. Keeping it makes every feature
        # vector point in almost the same direction, collapsing the cosine
        # distances between different speakers to near zero.
        timbre = mfcc[:, 1:]
        vector = np.concatenate([timbre.mean(axis=0), timbre.std(axis=0)])
        return np.nan_to_num(vector)

    def _mfcc(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute MFCCs. Self-contained so librosa is not required."""
        grid = compute_grid(samples, sample_rate)
        if grid.frame_count == 0:
            return np.zeros((0, N_CEPSTRA))

        power = np.square(grid.spectrum)
        filterbank = _mel_filterbank(
            grid.frequencies, N_MEL_FILTERS, sample_rate
        )
        mel = power @ filterbank.T
        log_mel = np.log(mel + 1e-10)
        # DCT-II of the log-mel spectrum gives the cepstral coefficients.
        return _dct(log_mel)[:, :N_CEPSTRA]

    # --------------------------------------------------------------- clustering
    def _cluster(
        self, features: np.ndarray, max_speakers: int
    ) -> tuple[list[int], float]:
        """Average-linkage agglomerative clustering on cosine distance.

        Implemented directly rather than via scipy so the tie-breaking is
        explicit and fully deterministic: merges are chosen by the smallest
        distance, and ties resolve to the lowest cluster indices.
        """
        normalized = _l2_normalize(features)
        n = normalized.shape[0]
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        distances = _cosine_distance_matrix(normalized)

        merged_at: list[float] = []
        while len(clusters) > 1:
            keys = sorted(clusters)
            best: tuple[float, int, int] | None = None
            for i_pos, i in enumerate(keys):
                for j in keys[i_pos + 1 :]:
                    d = float(
                        np.mean(
                            distances[np.ix_(clusters[i], clusters[j])]
                        )
                    )
                    if best is None or d < best[0] - 1e-12:
                        best = (d, i, j)
            assert best is not None
            distance, i, j = best
            if distance > MERGE_DISTANCE and len(clusters) <= max_speakers:
                break
            clusters[i].extend(clusters.pop(j))
            merged_at.append(distance)

        labels = [0] * n
        for label, (_, members) in enumerate(sorted(clusters.items())):
            for member in members:
                labels[member] = label

        separation = self._separation(distances, labels)
        return labels, separation

    @staticmethod
    def _separation(distances: np.ndarray, labels: list[int]) -> float:
        """Mean between-cluster distance minus mean within-cluster distance.

        A small or negative value means the clusters are not really distinct,
        which lowers the reported confidence rather than being hidden.
        """
        labels_array = np.asarray(labels)
        if labels_array.size < 2 or len(set(labels)) < 2:
            return 0.0
        same = labels_array[:, None] == labels_array[None, :]
        upper = np.triu(np.ones_like(same, dtype=bool), k=1)
        within = distances[same & upper]
        between = distances[~same & upper]
        if within.size == 0 or between.size == 0:
            return 0.0
        return float(np.mean(between) - np.mean(within))

    # ------------------------------------------------------------- numbering
    @staticmethod
    def _number_by_first_appearance(
        spans: list[tuple[float, float]], labels: list[int]
    ) -> dict[int, str]:
        """Map cluster labels to s1, s2, ... by first appearance in the audio.

        Section 18A.2: numbering follows first appearance, never loudness,
        gender, nativity, confidence or cluster size.
        """
        order: list[int] = []
        for _, label in sorted(zip(spans, labels), key=lambda pair: pair[0][0]):
            if label not in order:
                order.append(label)
        return {label: f"s{i + 1}" for i, label in enumerate(order)}

    # ------------------------------------------------------------ confidence
    @staticmethod
    def _segment_confidence(
        features: np.ndarray, labels: list[int], index: int, separation: float
    ) -> float:
        """How well this segment fits its assigned cluster."""
        normalized = _l2_normalize(features)
        own = [i for i, label in enumerate(labels) if label == labels[index]]
        if len(own) < 2:
            return float(np.clip(0.35 + separation, 0.30, 0.60))
        centroid = normalized[own].mean(axis=0)
        distance = float(1.0 - _cosine(normalized[index], centroid))
        return float(np.clip(0.85 - distance * 2.0, 0.30, 0.85))

    @staticmethod
    def _count_confidence(count: int, separation: float, n_segments: int) -> float:
        """Confidence in the suggested speaker count.

        Capped at 0.80. Section 18A.11 requires the count to be shown with a
        confidence and to remain editable, and the example there shows a
        76%-confident 3-speaker estimate being corrected to 2 by a human.
        """
        score = 0.30
        score += min(0.30, max(0.0, separation * 1.5))
        score += min(0.20, n_segments / 40.0)
        if count > MAX_SUGGESTED_SPEAKERS:
            score -= 0.20
        return float(np.clip(score, 0.25, 0.80))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _mel_filterbank(
    frequencies: np.ndarray, n_filters: int, sample_rate: int
) -> np.ndarray:
    """Triangular mel filterbank over the given FFT bin frequencies."""

    def to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def from_mel(mel: np.ndarray | float) -> np.ndarray | float:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    low, high = 60.0, min(sample_rate / 2, 7600.0)
    points = from_mel(np.linspace(to_mel(low), to_mel(high), n_filters + 2))

    bank = np.zeros((n_filters, frequencies.size))
    for i in range(n_filters):
        left, centre, right = points[i], points[i + 1], points[i + 2]
        rising = (frequencies >= left) & (frequencies <= centre)
        falling = (frequencies > centre) & (frequencies <= right)
        if centre > left:
            bank[i, rising] = (frequencies[rising] - left) / (centre - left)
        if right > centre:
            bank[i, falling] = (right - frequencies[falling]) / (right - centre)
    return bank


def _dct(values: np.ndarray) -> np.ndarray:
    """DCT-II along the last axis, orthonormal-ish scaling."""
    n = values.shape[-1]
    if n == 0:
        return values
    k = np.arange(n)
    basis = np.cos(np.pi * (k[:, None] + 0.5) * k[None, :] / n)
    return values @ basis * (2.0 / n)


# Note: per-dimension standardisation (cepstral mean-variance normalisation)
# was tried here and removed. It rescales distances by the spread *within this
# file*, which destroys the absolute scale the merge threshold depends on, and
# with only two segments it z-scores them to +/-1 and forces their distance to
# the maximum regardless of whether they are the same speaker. Dropping C0 in
# _segment_features is what actually fixed the distance collapse.


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def _cosine_distance_matrix(normalized: np.ndarray) -> np.ndarray:
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return 1.0 - similarity


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{int(rest):02d}"


def diarize(
    buffer: AudioBuffer,
    speech_spans: list[tuple[float, float]],
    config: DetectionConfig | None = None,
) -> DiarizationResult:
    return Diarizer(config).diarize(buffer, speech_spans)
