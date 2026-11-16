"""Nativity suggestion per speaker: native / nonnative / tts / unknown.

DEVIATION NOTICE. CLAUDE.md section 18A.4 says nativity must NOT be guessed,
and lists accent, audio quality, volume and background noise as evidence that
does not count. This module exists because the project owner asked for nativity
to be produced from the audio anyway (deviation log, section 60.4). Read the
limits below before trusting a value it produces.

What it actually measures, per speaker:

  synthetic speech (tts)
      A text-to-speech voice is genuinely detectable: pitch that barely moves,
      almost no cycle-to-cycle jitter, digital silence between phrases instead
      of room noise, and no breaths or lip noise anywhere. Two or more of those
      together is strong evidence.

  fluency (native / nonnative)
      Speaking rate, how much of the turn is long pauses, filler density, and
      how confidently an English recogniser scored the words. These correlate
      with nativity; none of them measures it.

The honest caveats, stated because the spec is right about this:

  - Recogniser confidence is partly a function of accent, so this feature is a
    proxy for exactly the evidence section 18A.4 rules out. It is weighted
    lightly and never sufficient on its own.
  - A slow, careful native speaker and a fluent non-native speaker both defeat
    these features.
  - Confidence is capped below certainty on purpose, short or sparse speech
    returns `unknown`, and every value stays editable in the speaker panel
    (section 18A.11).

Use it as a first guess to correct, not as an answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.audio.loader import AudioBuffer
from app.config import SpeakerProfilingConfig
from app.speakers.diarization import DiarizationResult
from app.speakers.voice_profile import estimate_f0

log = logging.getLogger("ermis.nativity")

#: Fillers and hesitation markers, in their canonical V2 spellings (section 7).
_FILLERS = frozenset({"uh", "um", "hmm", "mm-hm", "uh-huh", "mm-mm", "nuh-huh", "[fp]"})

#: A gap at or above this length inside a turn counts as a long pause.
_LONG_PAUSE = 0.6

#: Speaking rates, in words per second of speech (not wall clock). Typical
#: conversational English runs 2.5-4 w/s; L2 speech is usually slower.
_FAST_RATE = 3.2
_SLOW_RATE = 1.9

#: Recogniser confidence bands used as a light-weight signal.
_HIGH_CONFIDENCE = 0.85
_LOW_CONFIDENCE = 0.60

#: Nativity is never claimed above this confidence: the features do not support
#: certainty and a filled-in field should not look like one.
_CONFIDENCE_CEILING = 0.75

#: Synthetic-speech thresholds. Both the pitch and the jitter test must pass:
#: a short human phrase can hold a steady pitch, and a clip trimmed out of a
#: longer recording carries digital silence, so neither of those alone is
#: evidence. Jitter -- how much the period wobbles cycle to cycle -- is the one
#: measurement a vocoder cannot fake by accident.
_TTS_MIN_STABILITY = 0.94
_TTS_MAX_JITTER = 0.012
_TTS_MIN_SPEAKING = 2.0


@dataclass(frozen=True)
class NativitySuggestion:
    """A suggested nativity value with the evidence behind it."""

    speaker_id: str
    nativity: str = "unknown"
    confidence: float = 0.0
    words: int = 0
    speaking_seconds: float = 0.0
    words_per_second: float = 0.0
    filler_ratio: float = 0.0
    long_pause_ratio: float = 0.0
    mean_word_confidence: float = 0.0
    #: 0 = pitch jumps around, 1 = perfectly steady (a TTS signature).
    pitch_stability: float = 0.0
    jitter: float = 0.0
    #: True when the audio between phrases is digital silence rather than a room.
    silent_background: bool = False
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "nativity": self.nativity,
            "confidence": round(self.confidence, 3),
            "words": self.words,
            "speaking_seconds": round(self.speaking_seconds, 2),
            "words_per_second": round(self.words_per_second, 2),
            "filler_ratio": round(self.filler_ratio, 3),
            "long_pause_ratio": round(self.long_pause_ratio, 3),
            "mean_word_confidence": round(self.mean_word_confidence, 3),
            "pitch_stability": round(self.pitch_stability, 3),
            "jitter": round(self.jitter, 4),
            "silent_background": self.silent_background,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


def suggest_nativity(
    buffer: AudioBuffer,
    diarization: DiarizationResult,
    words=(),
    config: SpeakerProfilingConfig | None = None,
) -> dict[str, NativitySuggestion]:
    """Suggest nativity per speaker. `words` are DraftWord-like objects.

    Without words only the synthetic-speech check can run; fluency needs to know
    what was said and when.
    """
    config = config or SpeakerProfilingConfig()
    out: dict[str, NativitySuggestion] = {}
    for speaker_id in diarization.speaker_ids:
        spans = diarization.timeline(speaker_id)
        speaker_words = [
            word
            for word in words
            if getattr(word, "speaker_id", None) == speaker_id
        ]
        out[speaker_id] = _suggest_one(
            buffer, speaker_id, spans, speaker_words, config
        )
    return out


def _suggest_one(
    buffer: AudioBuffer,
    speaker_id: str,
    spans: list[tuple[float, float]],
    words: list,
    config: SpeakerProfilingConfig,
) -> NativitySuggestion:
    speaking = sum(max(0.0, end - start) for start, end in spans)
    pitch_stability, jitter = _pitch_shape(buffer, spans)
    silent_background = _background_is_digital_silence(buffer, spans)

    rate, filler_ratio, long_pause_ratio, mean_confidence = _fluency(words)
    evidence = {
        "speaking_seconds": round(speaking, 2),
        "words": len(words),
        "words_per_second": round(rate, 2),
        "filler_ratio": round(filler_ratio, 3),
        "long_pause_ratio": round(long_pause_ratio, 3),
        "mean_word_confidence": round(mean_confidence, 3),
        "pitch_stability": round(pitch_stability, 3),
        "jitter": round(jitter, 4),
        "silent_background": silent_background,
    }

    base = dict(
        speaker_id=speaker_id,
        words=len(words),
        speaking_seconds=speaking,
        words_per_second=rate,
        filler_ratio=filler_ratio,
        long_pause_ratio=long_pause_ratio,
        mean_word_confidence=mean_confidence,
        pitch_stability=pitch_stability,
        jitter=jitter,
        silent_background=silent_background,
        evidence=evidence,
    )

    # --- synthetic speech first: it is the one class with a real signature ---
    #
    # Both voice-intrinsic signals are required. Digital silence around the
    # speech only corroborates: a clip trimmed out of a longer recording is
    # padded with exact zeros too, and on this project's own sample set that
    # alone was enough to call a human "tts".
    # pitch_stability is 0 when there was nothing to measure, which is also
    # what an unmeasurable jitter reads as -- so the measurement guard is on
    # stability, and a jitter of exactly zero is a real (very synthetic) value.
    measured = pitch_stability > 0.0
    steady_pitch = pitch_stability >= _TTS_MIN_STABILITY
    low_jitter = jitter <= _TTS_MAX_JITTER
    if measured and steady_pitch and low_jitter and speaking >= _TTS_MIN_SPEAKING:
        signals = [
            f"pitch barely varies (stability {pitch_stability:.2f})",
            f"almost no cycle-to-cycle jitter ({jitter:.3f})",
        ]
        confidence = 0.6
        if silent_background:
            signals.append("digital silence around the speech instead of a room")
            confidence = _CONFIDENCE_CEILING
        return NativitySuggestion(
            nativity="tts",
            confidence=confidence,
            reason=(
                "reads as computer-generated speech: "
                + "; ".join(signals)
                + " -- confirm by listening, and clear it if a person is speaking"
            ),
            **base,
        )

    # --- fluency: only with enough speech to measure ------------------------
    if len(words) < config.min_words_for_nativity or speaking < config.min_speaking_seconds:
        return NativitySuggestion(
            reason=(
                f"{len(words)} word(s) over {speaking:.1f}s is too little to judge "
                f"fluency (needs {config.min_words_for_nativity} words and "
                f"{config.min_speaking_seconds:.1f}s); section 18A.4 wants unknown "
                "rather than a guess"
            ),
            **base,
        )

    score = 0.0
    reasons: list[str] = []
    if rate >= _FAST_RATE:
        score += 0.35
        reasons.append(f"fast delivery ({rate:.1f} words/s)")
    elif rate <= _SLOW_RATE:
        score -= 0.35
        reasons.append(f"slow delivery ({rate:.1f} words/s)")

    if long_pause_ratio >= 0.30:
        score -= 0.25
        reasons.append(f"frequent long pauses ({long_pause_ratio:.2f} per word)")
    elif long_pause_ratio <= 0.08:
        score += 0.15
        reasons.append("few long pauses")

    if filler_ratio >= 0.18:
        score -= 0.10
        reasons.append(f"heavy filler use ({filler_ratio:.2f} per word)")

    # Weighted lightly on purpose: recogniser confidence is partly accent, the
    # very evidence section 18A.4 excludes.
    if mean_confidence >= _HIGH_CONFIDENCE:
        score += 0.15
        reasons.append("the recogniser scored the words confidently")
    elif 0.0 < mean_confidence <= _LOW_CONFIDENCE:
        score -= 0.15
        reasons.append("the recogniser scored the words poorly")

    if abs(score) < config.nativity_decision_margin:
        return NativitySuggestion(
            reason=(
                "fluency evidence is mixed ("
                + ("; ".join(reasons) if reasons else "nothing decisive")
                + "), so no nativity is claimed"
            ),
            **base,
        )

    nativity = "native" if score > 0 else "nonnative"
    confidence = min(_CONFIDENCE_CEILING, 0.35 + abs(score))
    return NativitySuggestion(
        nativity=nativity,
        confidence=confidence,
        reason=(
            f"fluency features lean {nativity}: "
            + "; ".join(reasons)
            + ". This is a proxy, not a measurement -- accent alone is not "
            "evidence (section 18A.4), so verify before submitting"
        ),
        **base,
    )


# ----------------------------------------------------------------- features
def _fluency(words: list) -> tuple[float, float, float, float]:
    """Speaking rate, filler ratio, long-pause ratio, mean word confidence."""
    if not words:
        return 0.0, 0.0, 0.0, 0.0

    ordered = sorted(words, key=lambda w: w.start)
    spoken = sum(max(0.0, w.end - w.start) for w in ordered)
    gaps = [
        max(0.0, later.start - earlier.end)
        for earlier, later in zip(ordered, ordered[1:])
    ]
    long_pauses = sum(1 for gap in gaps if gap >= _LONG_PAUSE)
    # Rate over speech plus short gaps: pausing to think should not count as
    # speaking slowly twice over.
    active = spoken + sum(gap for gap in gaps if gap < _LONG_PAUSE)
    rate = len(ordered) / active if active > 0 else 0.0
    fillers = sum(1 for word in ordered if word.text in _FILLERS)
    confidences = [
        word.confidence for word in ordered if getattr(word, "confidence", None)
    ]
    return (
        rate,
        fillers / len(ordered),
        long_pauses / len(ordered),
        float(np.mean(confidences)) if confidences else 0.0,
    )


def _pitch_shape(
    buffer: AudioBuffer, spans: list[tuple[float, float]]
) -> tuple[float, float]:
    """Pitch stability in [0, 1] and mean relative jitter.

    A TTS voice holds its fundamental almost perfectly steady; a human one
    wobbles from breath to breath and cycle to cycle.
    """
    values: list[float] = []
    for start, end in spans:
        values.extend(estimate_f0(buffer.slice_seconds(start, end), buffer.sample_rate))
    if len(values) < 8:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    if median <= 0:
        return 0.0, 0.0
    in_octave = array[(array > median * 0.6) & (array < median * 1.7)]
    if in_octave.size < 8:
        in_octave = array
    spread = float(np.percentile(in_octave, 75) - np.percentile(in_octave, 25))
    stability = max(0.0, 1.0 - spread / median)
    jitter = float(np.mean(np.abs(np.diff(in_octave))) / median)
    return stability, jitter


def _background_is_digital_silence(
    buffer: AudioBuffer, spans: list[tuple[float, float]]
) -> bool:
    """True when the audio outside this speaker's turns has no noise floor.

    A microphone always records a room; a synthesised file usually has exact
    zeros between phrases.
    """
    if not spans or buffer.samples.size == 0:
        return False
    mask = np.ones(buffer.samples.size, dtype=bool)
    rate = buffer.sample_rate
    for start, end in spans:
        a = max(0, int(start * rate))
        b = min(buffer.samples.size, int(end * rate))
        mask[a:b] = False
    outside = buffer.samples[mask]
    if outside.size < rate // 4:  # under 0.25s of context: no opinion
        return False
    rms = float(np.sqrt(np.mean(np.square(outside, dtype=np.float64))))
    if rms <= 0.0:
        return True  # exact zeros: nothing was recorded, it was rendered
    # -75 dBFS is quieter than any real room recorded at a usable level.
    # bool() rather than the numpy scalar: this value is serialised to JSON.
    return bool(20 * np.log10(rms) < -75.0)


def apply_nativity(
    registry,
    suggestions: dict[str, NativitySuggestion],
    config: SpeakerProfilingConfig | None = None,
) -> list[str]:
    """Write confident nativity suggestions into a SpeakerRegistry.

    Skips anything below the threshold, anything `unknown`, and any profile the
    annotator has already edited (section 18A.11).
    """
    config = config or SpeakerProfilingConfig()
    changed: list[str] = []
    if not config.suggest_nativity:
        return changed
    for speaker_id, suggestion in suggestions.items():
        if speaker_id not in registry.profiles:
            continue
        if suggestion.nativity == "unknown":
            continue
        if suggestion.confidence < config.min_nativity_confidence:
            continue
        profile = registry.profiles[speaker_id]
        if profile.human_edited or profile.nativity != "unknown":
            continue
        applied = registry.suggest_metadata(
            speaker_id, nativity=suggestion.nativity
        )
        if applied is not None:
            changed.append(
                f"{speaker_id}: nativity suggested as {suggestion.nativity} "
                f"({suggestion.confidence * 100:.0f}% confidence, editable)"
            )
    return changed
