"""Nativity suggestion (CLAUDE.md sections 18A.4, 18A.11, and 60.4).

Section 18A.4 forbids guessing nativity; the project owner asked for it anyway,
so these tests hold the estimator to the only standards that still make sense:

  - it returns `unknown` unless there is real material to measure
  - it never claims certainty (confidence is capped)
  - synthetic speech, which does have a signature, is the one class it is
    allowed to be fairly confident about
  - it never overwrites a human decision, and only writes above threshold
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import load_audio
from app.config import SpeakerProfilingConfig
from app.speakers.diarization import DiarizationResult, SpeakerSegment
from app.speakers.metadata import SpeakerRegistry
from app.speakers.nativity import (
    NativitySuggestion,
    apply_nativity,
    suggest_nativity,
)
from tests.conftest import SAMPLE_RATE, VOICE_A, room_noise, speech, write_wav


class Word:
    """Stand-in for DraftWord: the estimator only reads these four fields."""

    def __init__(self, text, start, end, confidence=0.9, speaker_id="s1"):
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
        self.speaker_id = speaker_id


def one_speaker(duration: float) -> DiarizationResult:
    return DiarizationResult(
        [SpeakerSegment(0.0, duration, "s1", 0.9)], speaker_count=1, confidence=0.9
    )


def fluent_words(count=14, rate=3.6, confidence=0.92):
    """Fast, evenly spaced, confidently recognised words."""
    step = 1.0 / rate
    return [
        Word(f"word{i}", i * step, i * step + step * 0.8, confidence)
        for i in range(count)
    ]


def halting_words(count=14, confidence=0.5):
    """Slow words with a long pause after every second one.

    Deliberately well past the thresholds: the estimator is built to return
    `unknown` on anything borderline, so a fixture that only just leans one way
    would be testing the threshold rather than the direction.
    """
    words = []
    clock = 0.0
    for index in range(count):
        words.append(Word(f"word{index}", clock, clock + 0.7, confidence))
        clock += 0.75 + (1.1 if index % 2 == 1 else 0.05)
    return words


def jittered_voice(
    duration: float, f0: float = 150.0, jitter: float = 0.04, seed: int = 1
) -> np.ndarray:
    """A voice with a steady median pitch but human cycle-to-cycle jitter.

    conftest's `speech()` drives a perfectly periodic pulse train, which is
    acoustically closer to a vocoder than to a person; this randomises the
    period so the jitter test has something real to measure.
    """
    from tests.conftest import _resonator

    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    pulses = np.zeros(n)
    position = 0.0
    while position < n - 1:
        pulses[int(position)] = 1.0
        period = SAMPLE_RATE / f0
        position += period * (1.0 + rng.normal(0.0, jitter))
    voiced = sum(
        gain * _resonator(pulses, formant)
        for formant, gain in zip((600.0, 1100.0, 2400.0), (1.0, 0.6, 0.35))
    )
    peak = float(np.max(np.abs(voiced))) or 1.0
    return (0.3 * voiced / peak).astype(np.float32)


@pytest.fixture
def human_clip(tmp_path):
    """A human-shaped recording: jittered voice over a room-noise floor.

    conftest's `speech()` is a perfectly periodic pulse train -- zero jitter,
    which is a synthetic-speech signature, so it is the wrong model for "a
    person was recorded". `jittered_voice` randomises the glottal period the way
    a real voice does.
    """
    signal = jittered_voice(6.0, f0=115.0, jitter=0.04, seed=4)
    signal = signal + room_noise(6.0, level=0.002, seed=3)[: signal.size]
    path = write_wav(tmp_path / "human.wav", signal.astype(np.float32))
    return load_audio(path)


class TestUnknownWhenThereIsNothingToMeasure:
    def test_no_words_means_unknown(self, human_clip):
        result = suggest_nativity(human_clip, one_speaker(human_clip.duration))
        assert result["s1"].nativity == "unknown"
        assert result["s1"].confidence == 0.0

    def test_too_few_words_means_unknown(self, human_clip):
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words(count=4)
        )
        assert result["s1"].nativity == "unknown"
        assert "too little to judge fluency" in result["s1"].reason

    def test_short_turn_means_unknown(self, tmp_path):
        path = write_wav(tmp_path / "short.wav", speech(1.0, seed=2, **VOICE_A))
        buffer = load_audio(path)
        result = suggest_nativity(buffer, one_speaker(1.0), fluent_words(count=20))
        assert result["s1"].nativity == "unknown"

    def test_thresholds_are_configurable(self, human_clip):
        config = SpeakerProfilingConfig(
            min_words_for_nativity=2, min_speaking_seconds=0.5
        )
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words(count=3), config
        )
        # Now measurable, though the verdict may still be unknown on the merits.
        assert "too little to judge fluency" not in result["s1"].reason


class TestFluencyLean:
    def test_fast_confident_speech_leans_native(self, human_clip):
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words()
        )
        assert result["s1"].nativity == "native"
        assert result["s1"].words_per_second > 3.0

    def test_slow_halting_low_confidence_speech_leans_nonnative(self, human_clip):
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), halting_words()
        )
        assert result["s1"].nativity == "nonnative"
        assert result["s1"].long_pause_ratio > 0.2

    def test_mixed_evidence_stays_unknown(self, human_clip):
        # Fast delivery but poorly recognised: the features disagree, so no
        # claim is made.
        result = suggest_nativity(
            human_clip,
            one_speaker(human_clip.duration),
            fluent_words(rate=2.4, confidence=0.55),
        )
        assert result["s1"].nativity == "unknown"
        assert "mixed" in result["s1"].reason

    def test_confidence_never_reaches_certainty(self, human_clip):
        for words in (fluent_words(count=40), halting_words(count=40)):
            result = suggest_nativity(
                human_clip, one_speaker(human_clip.duration), words
            )
            assert result["s1"].confidence <= 0.75

    def test_reason_states_that_accent_is_not_evidence(self, human_clip):
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words()
        )
        assert "accent alone is not" in result["s1"].reason

    def test_fillers_are_counted(self, human_clip):
        words = fluent_words(count=12)
        words[2].text = "um"
        words[5].text = "uh"
        words[8].text = "hmm"
        result = suggest_nativity(human_clip, one_speaker(human_clip.duration), words)
        assert result["s1"].filler_ratio == pytest.approx(3 / 12, abs=0.01)


class TestSyntheticSpeech:
    def test_flat_pitch_over_digital_silence_reads_as_tts(self, tmp_path):
        # A synthesised voice: perfectly steady fundamental, exact zeros around
        # the phrase instead of a room.
        rate = SAMPLE_RATE
        t = np.arange(int(2.0 * rate)) / rate
        tone = 0.3 * (
            np.sin(2 * np.pi * 150 * t)
            + 0.4 * np.sin(2 * np.pi * 300 * t)
            + 0.2 * np.sin(2 * np.pi * 450 * t)
        )
        signal = np.concatenate(
            [np.zeros(rate), tone.astype(np.float32), np.zeros(rate)]
        ).astype(np.float32)
        path = write_wav(tmp_path / "tts.wav", signal)
        buffer = load_audio(path)
        diarization = DiarizationResult(
            [SpeakerSegment(1.0, 3.0, "s1", 0.9)], speaker_count=1, confidence=0.9
        )
        result = suggest_nativity(buffer, diarization)
        assert result["s1"].nativity == "tts"
        assert result["s1"].pitch_stability > 0.9
        assert "computer-generated" in result["s1"].reason

    def test_steady_pitch_over_digital_silence_alone_is_not_enough(self, tmp_path):
        """Regression from the project's own sample set.

        A 1.9s human clip held a steady pitch and, being trimmed out of a longer
        recording, was padded with exact zeros -- two "signals" that together
        called a person synthetic. Cycle-to-cycle jitter, which a real voice
        always has, is now required as well.
        """
        signal = jittered_voice(3.0, f0=150.0, jitter=0.04, seed=8)
        padded = np.concatenate(
            [np.zeros(SAMPLE_RATE // 2), signal, np.zeros(SAMPLE_RATE // 2)]
        ).astype(np.float32)
        buffer = load_audio(write_wav(tmp_path / "trimmed.wav", padded))
        diarization = DiarizationResult(
            [SpeakerSegment(0.5, 3.5, "s1", 0.9)], speaker_count=1, confidence=0.9
        )
        suggestion = suggest_nativity(buffer, diarization)["s1"]
        assert suggestion.silent_background is True
        assert suggestion.jitter > 0.012
        assert suggestion.nativity != "tts"

    def test_too_short_to_call_synthetic(self, tmp_path):
        rate = SAMPLE_RATE
        t = np.arange(int(1.0 * rate)) / rate
        tone = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        buffer = load_audio(write_wav(tmp_path / "short_tts.wav", tone))
        diarization = DiarizationResult(
            [SpeakerSegment(0.0, 1.0, "s1", 0.9)], speaker_count=1, confidence=0.9
        )
        assert suggest_nativity(buffer, diarization)["s1"].nativity != "tts"

    def test_a_human_recording_is_not_called_tts(self, human_clip):
        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words()
        )
        assert result["s1"].nativity != "tts"


class TestApplying:
    def registry(self) -> SpeakerRegistry:
        return SpeakerRegistry.from_diarization(one_speaker(6.0))

    def test_confident_suggestion_is_written(self):
        registry = self.registry()
        applied = apply_nativity(
            registry, {"s1": NativitySuggestion("s1", "native", 0.7)}
        )
        assert applied
        assert registry.profiles["s1"].nativity == "native"
        # Still a suggestion, not a human decision.
        assert registry.profiles["s1"].human_edited is False

    def test_weak_suggestion_is_not_written(self):
        registry = self.registry()
        applied = apply_nativity(
            registry, {"s1": NativitySuggestion("s1", "native", 0.2)}
        )
        assert applied == []
        assert registry.profiles["s1"].nativity == "unknown"

    def test_unknown_is_never_written(self):
        registry = self.registry()
        assert apply_nativity(registry, {"s1": NativitySuggestion("s1")}) == []

    def test_human_edit_wins(self):
        registry = self.registry()
        registry.set_metadata("s1", nativity="nonnative")
        apply_nativity(registry, {"s1": NativitySuggestion("s1", "native", 0.75)})
        assert registry.profiles["s1"].nativity == "nonnative"

    def test_disabled_config_writes_nothing(self):
        registry = self.registry()
        config = SpeakerProfilingConfig(suggest_nativity=False)
        applied = apply_nativity(
            registry, {"s1": NativitySuggestion("s1", "native", 0.75)}, config
        )
        assert applied == []
        assert registry.profiles["s1"].nativity == "unknown"

    def test_only_allowed_values_can_be_written(self):
        from app.speakers.metadata import SpeakerError

        registry = self.registry()
        with pytest.raises(SpeakerError):
            registry.suggest_metadata("s1", nativity="probably native")


class TestEvidenceIsReported:
    def test_dict_is_json_shaped(self, human_clip):
        import json

        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words()
        )
        payload = json.loads(json.dumps(result["s1"].to_dict()))
        assert payload["nativity"] == "native"
        assert payload["evidence"]["words"] == 14
        assert payload["reason"]

    def test_every_field_is_json_serialisable(self, human_clip):
        """Regression: silent_background was a numpy bool and broke json.dumps.

        The batch run over ten clips produced every report correctly and then
        died writing the JSON, which is the worst place to find out.
        """
        import json

        result = suggest_nativity(
            human_clip, one_speaker(human_clip.duration), fluent_words()
        )
        payload = result["s1"].to_dict()
        json.dumps(payload)  # must not raise
        assert isinstance(payload["silent_background"], bool)
        assert isinstance(payload["evidence"]["silent_background"], bool)

    def test_digital_silence_is_detected_as_such(self, tmp_path):
        # Exact zeros outside the turn: nothing was recorded, it was rendered.
        rate = SAMPLE_RATE
        signal = np.concatenate(
            [np.zeros(rate), speech(2.0, seed=1, **VOICE_A), np.zeros(rate)]
        ).astype(np.float32)
        buffer = load_audio(write_wav(tmp_path / "rendered.wav", signal))
        diarization = DiarizationResult(
            [SpeakerSegment(1.0, 3.0, "s1", 0.9)], speaker_count=1, confidence=0.9
        )
        assert suggest_nativity(buffer, diarization)["s1"].silent_background is True

    def test_a_room_floor_is_not_digital_silence(self, human_clip):
        diarization = DiarizationResult(
            [SpeakerSegment(1.0, 3.0, "s1", 0.9)], speaker_count=1, confidence=0.9
        )
        assert (
            suggest_nativity(human_clip, diarization)["s1"].silent_background is False
        )

    def test_per_speaker_words_are_separated(self, human_clip):
        diarization = DiarizationResult(
            [
                SpeakerSegment(0.0, 3.0, "s1", 0.9),
                SpeakerSegment(3.0, 6.0, "s2", 0.9),
            ],
            speaker_count=2,
            confidence=0.8,
        )
        words = fluent_words(count=10) + [
            Word(f"b{i}", 3.0 + i * 0.3, 3.0 + i * 0.3 + 0.2, 0.9, "s2")
            for i in range(10)
        ]
        result = suggest_nativity(human_clip, diarization, words)
        assert result["s1"].words == 10
        assert result["s2"].words == 10
