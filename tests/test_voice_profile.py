"""Voice descriptors suggested from audio (CLAUDE.md sections 18A.3, 18A.11, 37).

The suggestions are assistive, and the tests are mostly about restraint:

  - a pitch in the overlap band, or too little voiced audio, must come back
    `unknown` rather than as a guess
  - `nativity` is never inferred at all
  - a profile the annotator edited is never overwritten
  - a weakly supported suggestion is reported but not written into the registry
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import load_audio
from app.detection.pipeline import AnalysisPipeline
from app.speakers.diarization import DiarizationResult, SpeakerSegment
from app.speakers.metadata import SpeakerRegistry
from app.speakers.voice_profile import (
    MIN_APPLY_CONFIDENCE,
    VoiceSuggestion,
    apply_suggestions,
    estimate_f0,
    suggest_voice_profiles,
)
from tests.conftest import SAMPLE_RATE, VOICE_A, VOICE_B, speech, write_wav


def one_speaker(duration: float) -> DiarizationResult:
    segment = SpeakerSegment(0.0, duration, "s1", 0.9)
    return DiarizationResult([segment], speaker_count=1, confidence=0.9)


class TestPitchTracking:
    @pytest.mark.parametrize("f0", [110.0, 150.0, 225.0])
    def test_recovers_a_known_fundamental(self, f0):
        signal = speech(1.0, f0=f0, seed=3)
        values = estimate_f0(signal, SAMPLE_RATE)
        assert values, "voiced speech should produce pitch estimates"
        assert np.median(values) == pytest.approx(f0, rel=0.12)

    def test_silence_produces_nothing(self):
        assert estimate_f0(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) == []

    def test_empty_input_is_safe(self):
        assert estimate_f0(np.zeros(0, dtype=np.float32), SAMPLE_RATE) == []

    def test_audio_shorter_than_a_frame_is_safe(self):
        assert estimate_f0(np.zeros(100, dtype=np.float32), SAMPLE_RATE) == []


class TestGenderSuggestion:
    def _suggest(self, tmp_path, name, **voice):
        signal = speech(2.0, seed=5, **voice)
        path = write_wav(tmp_path / name, signal)
        buffer = load_audio(path)
        return suggest_voice_profiles(buffer, one_speaker(buffer.duration))["s1"]

    def test_low_voice_reads_as_masculine_sounding(self, tmp_path):
        suggestion = self._suggest(tmp_path, "low.wav", **VOICE_A)
        assert suggestion.gender == "msv"
        assert suggestion.confidence > 0

    def test_high_voice_reads_as_feminine_sounding(self, tmp_path):
        suggestion = self._suggest(tmp_path, "high.wav", **VOICE_B)
        assert suggestion.gender == "fsv"

    def test_overlap_band_stays_unknown(self, tmp_path):
        # Section 18A.3: the system must not force a classification.
        suggestion = self._suggest(
            tmp_path, "mid.wav", f0=170.0, formants=(700.0, 1300.0, 2600.0)
        )
        assert suggestion.gender == "unknown"
        assert "overlap band" in suggestion.reason

    def test_too_little_voiced_audio_stays_unknown(self, tmp_path):
        path = write_wav(tmp_path / "short.wav", speech(0.12, seed=1, **VOICE_A))
        buffer = load_audio(path)
        suggestion = suggest_voice_profiles(buffer, one_speaker(buffer.duration))["s1"]
        assert suggestion.gender == "unknown"
        assert suggestion.confidence == 0.0

    def test_silence_stays_unknown(self, wav_silence):
        buffer = load_audio(wav_silence)
        suggestion = suggest_voice_profiles(buffer, one_speaker(buffer.duration))["s1"]
        assert suggestion.gender == "unknown"

    def test_nativity_is_never_inferred(self, tmp_path):
        # Sections 18A.4 and 37 rule out accent, level and microphone as
        # evidence, and nothing local can hear nativity.
        for name, voice in (("a.wav", VOICE_A), ("b.wav", VOICE_B)):
            assert self._suggest(tmp_path, name, **voice).nativity == "unknown"

    def test_every_speaker_gets_a_suggestion(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        analysis = AnalysisPipeline().run(buffer)
        suggestions = suggest_voice_profiles(buffer, analysis.diarization)
        assert set(suggestions) == set(analysis.diarization.speaker_ids)

    def test_suggestion_dict_is_json_shaped(self, tmp_path):
        import json

        suggestion = self._suggest(tmp_path, "j.wav", **VOICE_A)
        payload = json.loads(json.dumps(suggestion.to_dict()))
        assert payload["gender"] == "msv"
        assert payload["nativity"] == "unknown"
        assert payload["reason"]


class TestApplyingSuggestions:
    def registry(self) -> SpeakerRegistry:
        return SpeakerRegistry.from_diarization(one_speaker(5.0))

    def test_confident_suggestion_is_written(self):
        registry = self.registry()
        applied = apply_suggestions(
            registry, {"s1": VoiceSuggestion("s1", "msv", 0.8)}
        )
        assert applied
        assert registry.profiles["s1"].gender == "msv"

    def test_written_value_is_not_marked_as_a_human_edit(self):
        # Section 18A.11: it is a suggestion, and the panel must still show it
        # as one so the annotator knows nobody has checked it.
        registry = self.registry()
        apply_suggestions(registry, {"s1": VoiceSuggestion("s1", "msv", 0.8)})
        assert registry.profiles["s1"].human_edited is False

    def test_weak_suggestion_is_not_written(self):
        registry = self.registry()
        weak = MIN_APPLY_CONFIDENCE - 0.2
        applied = apply_suggestions(
            registry, {"s1": VoiceSuggestion("s1", "fsv", weak)}
        )
        assert applied == []
        assert registry.profiles["s1"].gender == "unknown"

    def test_unknown_suggestion_writes_nothing(self):
        registry = self.registry()
        assert apply_suggestions(registry, {"s1": VoiceSuggestion("s1")}) == []

    def test_human_edit_is_never_overwritten(self):
        registry = self.registry()
        registry.set_metadata("s1", gender="fsv")
        apply_suggestions(registry, {"s1": VoiceSuggestion("s1", "msv", 0.95)})
        assert registry.profiles["s1"].gender == "fsv"
        assert registry.profiles["s1"].human_edited is True

    def test_existing_value_is_kept_by_default(self):
        registry = self.registry()
        registry.suggest_metadata("s1", gender="fsv", confidence=0.7)
        apply_suggestions(registry, {"s1": VoiceSuggestion("s1", "msv", 0.95)})
        assert registry.profiles["s1"].gender == "fsv"

    def test_unknown_speaker_id_is_ignored(self):
        registry = self.registry()
        assert apply_suggestions(registry, {"s9": VoiceSuggestion("s9", "msv", 0.9)}) == []


class TestRegistrySuggestMetadata:
    def test_invalid_gender_is_refused(self):
        from app.speakers.metadata import SpeakerError

        registry = SpeakerRegistry.from_diarization(one_speaker(5.0))
        with pytest.raises(SpeakerError):
            registry.suggest_metadata("s1", gender="male")

    def test_invalid_nativity_is_refused(self):
        from app.speakers.metadata import SpeakerError

        registry = SpeakerRegistry.from_diarization(one_speaker(5.0))
        with pytest.raises(SpeakerError):
            registry.suggest_metadata("s1", nativity="probably-native")

    def test_no_change_returns_none(self):
        registry = SpeakerRegistry.from_diarization(one_speaker(5.0))
        assert registry.suggest_metadata("s1") is None
