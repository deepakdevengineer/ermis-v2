"""Voice descriptors suggested from the audio (CLAUDE.md sections 18A.3, 37).

Suggests the `gender` field -- `msv` / `fsv` / `unknown` -- from each speaker's
fundamental frequency, measured by autocorrelation over their own segments.

Three rules the spec is explicit about, all enforced here:

  - It is a voice-SOUND descriptor, not a claim about a person (section 18A.3).
  - The system must not force a classification. A pitch that lands in the
    overlap band between typical ranges, or a speaker with too little voiced
    audio to measure, comes back `unknown` rather than as a coin flip.
  - `nativity` is never guessed. Accent, microphone quality and volume are
    explicitly ruled out as evidence (sections 18A.4, 37), and nothing local
    can hear nativity, so it stays `unknown` for a human to set.

Every suggestion carries a confidence and is overridable, which is the point of
section 18A.11: automatic detection is assistive only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.audio.loader import AudioBuffer
from app.speakers.diarization import DiarizationResult

log = logging.getLogger("ermis.voice")

#: Human speech fundamentals live between these bounds; anything outside is a
#: measurement artefact rather than a voice.
_F0_MIN = 60.0
_F0_MAX = 400.0

#: Below this median f0 the voice reads as masculine, above the upper bound as
#: feminine. The gap between them is the overlap band where adult ranges
#: genuinely coincide, and everything inside it stays `unknown`.
_MASCULINE_MAX = 155.0
_FEMININE_MIN = 190.0

#: Minimum voiced audio needed before a suggestion is offered at all.
_MIN_VOICED_SECONDS = 0.35

#: Frame length and hop for the pitch tracker, in seconds.
_FRAME = 0.040
_HOP = 0.020


@dataclass(frozen=True)
class VoiceSuggestion:
    """A suggested voice descriptor for one speaker."""

    speaker_id: str
    gender: str = "unknown"
    #: Confidence in `gender`, in [0, 1]. 0 when nothing could be measured.
    confidence: float = 0.0
    #: Median fundamental frequency in Hz, or None when unmeasurable.
    median_f0: float | None = None
    voiced_seconds: float = 0.0
    #: Always "unknown": nativity is never inferred from audio (section 18A.4).
    nativity: str = "unknown"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "gender": self.gender,
            "nativity": self.nativity,
            "confidence": round(self.confidence, 3),
            "median_f0": round(self.median_f0, 1) if self.median_f0 else None,
            "voiced_seconds": round(self.voiced_seconds, 2),
            "reason": self.reason,
        }


def suggest_voice_profiles(
    buffer: AudioBuffer, diarization: DiarizationResult
) -> dict[str, VoiceSuggestion]:
    """Suggest a voice descriptor per speaker. Assistive only."""
    out: dict[str, VoiceSuggestion] = {}
    for speaker_id in diarization.speaker_ids:
        spans = diarization.timeline(speaker_id)
        out[speaker_id] = _suggest_one(buffer, speaker_id, spans)
    return out


def _suggest_one(
    buffer: AudioBuffer, speaker_id: str, spans: list[tuple[float, float]]
) -> VoiceSuggestion:
    values: list[float] = []
    for start, end in spans:
        values.extend(estimate_f0(buffer.slice_seconds(start, end), buffer.sample_rate))

    voiced_seconds = len(values) * _HOP
    if voiced_seconds < _MIN_VOICED_SECONDS:
        return VoiceSuggestion(
            speaker_id,
            reason=(
                f"only {voiced_seconds:.2f}s of voiced audio; too little to "
                "measure a pitch (section 18A.3 requires unknown when "
                "confidence is insufficient)"
            ),
        )

    median = float(np.median(values))
    if median <= _MASCULINE_MAX:
        gender = "msv"
        margin = (_MASCULINE_MAX - median) / _MASCULINE_MAX
    elif median >= _FEMININE_MIN:
        gender = "fsv"
        margin = (median - _FEMININE_MIN) / _FEMININE_MIN
    else:
        return VoiceSuggestion(
            speaker_id,
            median_f0=median,
            voiced_seconds=voiced_seconds,
            reason=(
                f"median pitch {median:.0f} Hz falls in the overlap band "
                f"({_MASCULINE_MAX:.0f}-{_FEMININE_MIN:.0f} Hz) where masculine "
                "and feminine ranges coincide, so no descriptor is claimed"
            ),
        )

    # Spread widens the uncertainty: a pitch track jumping around is usually a
    # tracking failure or two voices caught in one segment.
    #
    # Octave errors are measured out first. Autocorrelation regularly locks onto
    # half or double the fundamental on a few frames, and counting those as
    # instability would push an otherwise clear voice below the threshold for
    # being suggested at all -- the opposite of the intended caution.
    array = np.asarray(values, dtype=np.float64)
    in_octave = array[(array > median * 0.6) & (array < median * 1.7)]
    if in_octave.size >= 8:
        spread = float(np.percentile(in_octave, 75) - np.percentile(in_octave, 25))
    else:
        spread = float(np.percentile(array, 75) - np.percentile(array, 25))
    stability = max(0.0, 1.0 - spread / max(1.0, median))
    confidence = max(0.0, min(1.0, 0.45 + 0.9 * margin)) * (0.55 + 0.45 * stability)

    return VoiceSuggestion(
        speaker_id=speaker_id,
        gender=gender,
        confidence=round(confidence, 3),
        median_f0=median,
        voiced_seconds=voiced_seconds,
        reason=(
            f"median pitch {median:.0f} Hz over {voiced_seconds:.1f}s of voiced "
            f"audio reads as a {'masculine' if gender == 'msv' else 'feminine'}"
            "-sounding voice; editable"
        ),
    )


def estimate_f0(
    samples: np.ndarray, sample_rate: int, threshold: float = 0.35
) -> list[float]:
    """Per-frame fundamental frequency of the voiced frames, in Hz.

    Autocorrelation with a normalised peak test. Unvoiced or too-quiet frames
    are dropped rather than reported as zero, so a median over the result is a
    median over actual voicing.
    """
    if samples.size == 0 or sample_rate <= 0:
        return []

    frame = int(_FRAME * sample_rate)
    hop = int(_HOP * sample_rate)
    if frame <= 1 or samples.size < frame:
        return []

    min_lag = max(2, int(sample_rate / _F0_MAX))
    max_lag = min(frame - 1, int(sample_rate / _F0_MIN))
    if max_lag <= min_lag:
        return []

    window = np.hanning(frame).astype(np.float32)
    out: list[float] = []
    for start in range(0, samples.size - frame + 1, hop):
        block = samples[start : start + frame].astype(np.float64)
        rms = float(np.sqrt(np.mean(block * block)))
        if rms < 1e-4:  # silence: nothing to track
            continue
        block = (block - block.mean()) * window

        correlation = np.correlate(block, block, mode="full")[frame - 1 :]
        if correlation[0] <= 0:
            continue
        normalized = correlation / correlation[0]
        segment = normalized[min_lag : max_lag + 1]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment)) + min_lag
        if normalized[lag] < threshold:  # not periodic enough to be voiced
            continue
        out.append(sample_rate / lag)
    return out


#: A suggestion below this confidence is reported but not written into the
#: registry. Section 18A.3 says the system must not force a classification, and
#: a barely-supported guess written into a metadata field is exactly that -- the
#: annotator would see a filled-in value with no idea how thin the evidence was.
MIN_APPLY_CONFIDENCE = 0.55


def apply_suggestions(
    registry,
    suggestions: dict[str, VoiceSuggestion],
    only_unset: bool = True,
    min_confidence: float = MIN_APPLY_CONFIDENCE,
) -> list[str]:
    """Write confident suggestions into a SpeakerRegistry. Returns what changed.

    `only_unset` keeps a human decision: a profile the annotator already edited
    is left alone, because the application must never silently overwrite a
    human speaker assignment (section 18A.11). A suggestion below
    `min_confidence` is skipped and stays visible as a suggestion only.
    """
    changed: list[str] = []
    for speaker_id, suggestion in suggestions.items():
        if speaker_id not in registry.profiles:
            continue
        profile = registry.profiles[speaker_id]
        if only_unset and (profile.human_edited or profile.gender != "unknown"):
            continue
        if suggestion.gender == "unknown":
            continue
        if suggestion.confidence < min_confidence:
            continue
        applied = registry.suggest_metadata(
            speaker_id,
            gender=suggestion.gender,
            confidence=suggestion.confidence,
        )
        if applied is not None:
            changed.append(
                f"{speaker_id}: gender suggested as {suggestion.gender} "
                f"({suggestion.confidence * 100:.0f}% confidence, editable)"
            )
    return changed
