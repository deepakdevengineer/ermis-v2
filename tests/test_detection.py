"""Analysis engine: framing, VAD, silence/pause, quality, candidate detectors.

CLAUDE.md sections 3 (two layers), 13 (pause is metadata), 45 (analysis
engine), 46 (event model), 47 (human confirmation).

The governing property, asserted repeatedly below: no detector may produce a
transcription tag. Detection yields `possible_*` events; only a human
confirmation turns one into a tag.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import load_audio
from app.audio.preprocessing import (
    band_energy_ratio,
    boolean_runs,
    compute_grid,
    merge_spans,
    smooth_flags,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    to_dbfs,
)
from app.audio.quality import QualityAnalyzer
from app.detection.artifact import ArtifactDetector
from app.detection.background import BackgroundDetector
from app.detection.laughter import LaughterDetector
from app.detection.nonlexical import NonLexicalDetector
from app.detection.overlap import OverlapDetector
from app.detection.pipeline import AnalysisPipeline
from app.detection.silence import SilenceDetector, format_pause_info
from app.detection.vad import EnergyVad
from app.ermis import tags as T
from app.models.event import AnalysisEvent, EventType, UnconfirmedEventError
from tests.conftest import SAMPLE_RATE, room_noise, speech, VOICE_A


# ---------------------------------------------------------------- primitives
class TestFrameGrid:
    def test_frame_count_and_hop(self, wav_speech):
        buffer = load_audio(wav_speech)
        grid = compute_grid(buffer.samples, buffer.sample_rate)
        assert grid.hop_seconds == pytest.approx(0.010)
        # ~100 frames per second of audio.
        assert grid.frame_count == pytest.approx(buffer.duration * 100, rel=0.05)

    def test_feature_shapes_agree(self, wav_speech):
        buffer = load_audio(wav_speech)
        grid = compute_grid(buffer.samples, buffer.sample_rate)
        n = grid.frame_count
        assert grid.rms.shape == (n,)
        assert grid.peak.shape == (n,)
        assert grid.zcr.shape == (n,)
        assert grid.spectrum.shape[0] == n
        assert grid.spectrum.shape[1] == grid.frequencies.size

    def test_empty_signal(self):
        grid = compute_grid(np.zeros(0, dtype=np.float32), 16000)
        assert grid.frame_count == 0

    def test_time_frame_round_trip(self, wav_speech):
        buffer = load_audio(wav_speech)
        grid = compute_grid(buffer.samples, buffer.sample_rate)
        for seconds in (0.0, 0.5, 1.0, 1.9):
            index = grid.time_to_frame(seconds)
            assert abs(grid.frame_to_time(index) - seconds) <= grid.hop_seconds

    def test_tail_is_padded_not_dropped(self):
        # An event at the very end of a file must still be analysed.
        signal = np.zeros(16000 + 37, dtype=np.float32)
        signal[-10:] = 0.9
        grid = compute_grid(signal, 16000)
        assert float(grid.peak[-1]) > 0.5

    def test_dbfs_floors_silence(self):
        values = to_dbfs(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(values))
        assert values[1] == pytest.approx(0.0)


class TestSpectralFeatures:
    def test_flatness_separates_tone_from_noise(self):
        t = np.linspace(0, 1, 16000, endpoint=False)
        tone_grid = compute_grid((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
        noise_grid = compute_grid(room_noise(1.0, 0.2, seed=2)[:16000], 16000)
        assert float(np.median(spectral_flatness(tone_grid))) < 0.2
        assert float(np.median(spectral_flatness(noise_grid))) > 0.5

    def test_centroid_rises_with_frequency(self):
        t = np.linspace(0, 1, 16000, endpoint=False)
        low = compute_grid((0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 16000)
        high = compute_grid((0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32), 16000)
        assert float(np.median(spectral_centroid(low))) < float(
            np.median(spectral_centroid(high))
        )

    def test_flux_spikes_at_a_transient(self, wav_click):
        buffer = load_audio(wav_click)
        grid = compute_grid(buffer.samples, buffer.sample_rate)
        flux = spectral_flux(grid)
        index = grid.time_to_frame(1.0)
        local = flux[max(0, index - 2) : index + 3]
        assert float(local.max()) > float(np.median(flux)) * 3

    def test_band_energy_ratio_bounds(self, wav_speech):
        buffer = load_audio(wav_speech)
        grid = compute_grid(buffer.samples, buffer.sample_rate)
        ratio = band_energy_ratio(grid, 300.0, 3000.0)
        assert np.all((ratio >= 0.0) & (ratio <= 1.0))

    def test_band_energy_ratio_handles_silent_frames(self):
        # Must not divide by zero or emit a warning on a silent frame.
        grid = compute_grid(np.zeros(16000, dtype=np.float32), 16000)
        with np.errstate(all="raise"):
            ratio = band_energy_ratio(grid, 300.0, 3000.0)
        assert np.all(ratio == 0.0)


class TestRunHelpers:
    def test_boolean_runs(self):
        flags = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        assert boolean_runs(flags) == [(1, 3), (5, 6)]

    def test_boolean_runs_empty(self):
        assert boolean_runs(np.zeros(0, dtype=bool)) == []

    def test_smooth_flags_removes_flicker(self):
        flags = np.array([1, 0, 1, 1, 1, 0, 1], dtype=bool)
        assert smooth_flags(flags, 3).tolist() == [True] * 7

    def test_merge_spans(self):
        assert merge_spans([(0, 1), (1.05, 2), (3, 4)], gap=0.1) == [(0, 2), (3, 4)]

    def test_merge_spans_empty(self):
        assert merge_spans([], gap=1.0) == []


# ----------------------------------------------------------------------- VAD
class TestVad:
    def test_finds_speech_regions(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = EnergyVad().analyse(buffer)
        # Four turns: 0.3-1.3, 2.0-2.9, 3.6-4.4, 5.1-6.1 (0.7s gaps).
        assert len(result.speech) == 4
        starts = [round(s, 1) for s, _ in result.speech]
        assert starts == [0.3, 2.0, 3.6, 5.1]

    def test_boundaries_are_accurate(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = EnergyVad().analyse(buffer)
        first_start, first_end = result.speech[0]
        assert first_start == pytest.approx(0.3, abs=0.05)
        assert first_end == pytest.approx(1.3, abs=0.05)

    def test_rejects_broadband_noise(self, wav_hiss):
        # Steady hiss is not speech. Without the tonality gate an energy-only
        # detector calls it speech, because flat spectra put plenty of energy
        # in the 200-4000 Hz band.
        buffer = load_audio(wav_hiss)
        assert not EnergyVad().analyse(buffer).has_speech

    def test_rejects_silence(self, wav_silence):
        buffer = load_audio(wav_silence)
        assert not EnergyVad().analyse(buffer).has_speech

    def test_adapts_the_threshold_to_the_recording(self, tmp_path):
        from tests.conftest import write_wav

        loud = write_wav(tmp_path / "loud.wav", speech(1.5, amplitude=0.5, **VOICE_A))
        quiet = write_wav(tmp_path / "quiet.wav", speech(1.5, amplitude=0.02, **VOICE_A))
        # A fixed dBFS threshold would miss the quiet clip entirely.
        assert EnergyVad().analyse(load_audio(loud)).has_speech
        assert EnergyVad().analyse(load_audio(quiet)).has_speech

    def test_speech_events_are_not_uncertain(self, wav_speech):
        buffer = load_audio(wav_speech)
        for event in EnergyVad().detect(buffer):
            assert event.type is EventType.SPEECH
            assert not event.type.is_uncertain

    def test_confidence_is_bounded(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        for event in EnergyVad().detect(buffer):
            assert 0.0 <= event.confidence <= 0.95

    def test_deterministic(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        first = EnergyVad().analyse(buffer).speech
        for _ in range(3):
            assert EnergyVad().analyse(buffer).speech == first


# ------------------------------------------------------- silence and pauses
class TestSilenceAndPause:
    def test_gap_between_speech_is_a_pause(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = SilenceDetector().analyse(buffer)
        assert result.pauses
        assert all(end - start >= 0.5 for start, end in result.pauses)

    def test_leading_silence_is_not_a_pause(self, wav_two_speakers):
        # Quiet before the speaker starts is not a gap *in* speech.
        buffer = load_audio(wav_two_speakers)
        result = SilenceDetector().analyse(buffer)
        assert all(start > 0.2 for start, _ in result.pauses)

    def test_trailing_silence_is_not_a_pause(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = SilenceDetector().analyse(buffer)
        last_speech_end = max(end for _, end in result.speech)
        assert all(end <= last_speech_end + 0.01 for _, end in result.pauses)

    def test_pause_events_carry_no_tag(self, wav_two_speakers):
        # Section 13: V2 defines no pause tag.
        buffer = load_audio(wav_two_speakers)
        for event in SilenceDetector().detect(buffer):
            if event.type is EventType.PAUSE:
                assert event.proposed_tag is None
                event.confirm()
                assert event.tag is None

    def test_pause_tag_cannot_be_confirmed(self):
        event = AnalysisEvent(EventType.PAUSE, 1.0, 2.0)
        with pytest.raises(ValueError, match="no pause tag"):
            event.confirm("[pause]")

    def test_pause_info_states_the_rule(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        lines = format_pause_info(SilenceDetector().detect(buffer))
        assert lines
        for line in lines:
            assert "should not be inserted" in line

    def test_silence_covers_whole_file_for_silence(self, wav_silence):
        from app.detection.silence import silence_covers_whole_file

        buffer = load_audio(wav_silence)
        result = SilenceDetector().analyse(buffer)
        assert silence_covers_whole_file(buffer, result)


# ------------------------------------------------------------------- quality
class TestQuality:
    def test_digital_silence_proposes_na(self, wav_silence):
        buffer = load_audio(wav_silence)
        report = QualityAnalyzer().analyse(buffer, has_speech=False)
        assert T.NO_AUDIO in report.candidates
        assert report.is_silent

    def test_clipping_proposes_ga(self, wav_clipped):
        buffer = load_audio(wav_clipped)
        report = QualityAnalyzer().analyse(buffer, has_speech=True)
        assert T.GARBLED in report.candidates

    def test_steady_hiss_proposes_ns_not_ga(self, wav_hiss):
        # Section 12 gives "faint white noise" as an <ns> example, and section 9
        # excludes persistent hiss from [artifact]. Steady noise is not garbled.
        buffer = load_audio(wav_hiss)
        report = QualityAnalyzer().analyse(buffer, has_speech=False)
        assert T.NO_SPEECH in report.candidates
        assert T.GARBLED not in report.candidates

    def test_clean_speech_proposes_nothing(self, wav_speech):
        buffer = load_audio(wav_speech)
        report = QualityAnalyzer().analyse(buffer, has_speech=True)
        assert report.candidates == []

    def test_ns_suppressed_when_a_non_lexical_sound_was_found(self, wav_hiss):
        # Section 12: do not use <ns> when another tag applies.
        buffer = load_audio(wav_hiss)
        report = QualityAnalyzer().analyse(
            buffer, has_speech=False, has_non_lexical=True
        )
        assert T.NO_SPEECH not in report.candidates
        assert any("forbids <ns>" in note for note in report.notes)

    def test_ga_note_states_the_artifact_distinction(self, wav_clipped):
        buffer = load_audio(wav_clipped)
        report = QualityAnalyzer().analyse(buffer, has_speech=True)
        joined = " ".join(report.notes)
        assert "[artifact]" in joined
        assert "pitch-shifted" in joined.lower()

    def test_garbled_confidence_never_certain(self, wav_clipped):
        buffer = load_audio(wav_clipped)
        for event in QualityAnalyzer().detect(buffer, has_speech=True):
            if event.type is EventType.GARBLED_AUDIO:
                assert event.confidence <= 0.85

    def test_no_audio_is_measured_not_guessed(self, wav_silence):
        buffer = load_audio(wav_silence)
        events = QualityAnalyzer().detect(buffer, has_speech=False)
        assert [e.type for e in events] == [EventType.NO_AUDIO]
        assert events[0].confidence == 1.0

    def test_report_serializes_with_infinite_peak(self, wav_silence):
        buffer = load_audio(wav_silence)
        payload = QualityAnalyzer().analyse(buffer, has_speech=False).to_dict()
        assert payload["peak_dbfs"] is None  # -inf is not JSON-representable


# ---------------------------------------------------------------- candidates
class TestLaughterDetector:
    def test_finds_the_laughter_bout(self, wav_laughter):
        buffer = load_audio(wav_laughter)
        spans = EnergyVad().analyse(buffer).speech
        events = LaughterDetector().detect(buffer, speech_spans=spans)
        assert events, "should find the metronomic burst train"
        # The bout runs 1.2-2.8s.
        assert any(1.0 <= e.start <= 2.0 for e in events)

    def test_events_are_uncertain_and_pending(self, wav_laughter):
        buffer = load_audio(wav_laughter)
        for event in LaughterDetector().detect(buffer):
            assert event.type is EventType.POSSIBLE_LAUGHTER
            assert event.requires_human_confirmation
            with pytest.raises(UnconfirmedEventError):
                _ = event.tag

    def test_proposes_laughter_never_hn(self, wav_laughter):
        # Section 8 and 56.7: laughter is never [hn].
        buffer = load_audio(wav_laughter)
        for event in LaughterDetector().detect(buffer):
            assert event.proposed_tag == T.LAUGHTER

    def test_confidence_capped(self, wav_laughter):
        buffer = load_audio(wav_laughter)
        for event in LaughterDetector().detect(buffer):
            assert event.confidence <= 0.70

    def test_note_warns_about_confusable_sounds(self, wav_laughter):
        buffer = load_audio(wav_laughter)
        events = LaughterDetector().detect(buffer)
        assert events
        assert "listen before confirming" in events[0].note

    def test_silence_yields_nothing(self, wav_silence):
        buffer = load_audio(wav_silence)
        assert LaughterDetector().detect(buffer) == []

    def test_does_not_span_most_of_the_clip(self, wav_speech):
        # A "laughter" region covering the whole file is the rhythm test
        # latching onto ordinary speech.
        buffer = load_audio(wav_speech)
        for event in LaughterDetector().detect(buffer):
            assert event.duration / buffer.duration <= 0.5


class TestNonLexicalDetector:
    def test_finds_a_cough(self, wav_two_coughs):
        buffer = load_audio(wav_two_coughs)
        spans = EnergyVad().analyse(buffer).speech
        events = NonLexicalDetector().detect(buffer, speech_spans=spans)
        assert events

    def test_two_coughs_in_succession_produce_one_event(self, wav_two_coughs):
        # Case B, section 55: two coughs -> [hn], not "[hn] [hn]".
        buffer = load_audio(wav_two_coughs)
        spans = EnergyVad().analyse(buffer).speech
        events = NonLexicalDetector().detect(buffer, speech_spans=spans)
        # Both coughs fall between 1.0s and 1.7s.
        in_window = [e for e in events if 0.9 <= e.start <= 1.8]
        assert len(in_window) == 1, f"expected one merged event, got {in_window}"

    def test_proposes_hn(self, wav_two_coughs):
        buffer = load_audio(wav_two_coughs)
        for event in NonLexicalDetector().detect(buffer):
            assert event.type is EventType.POSSIBLE_HN
            assert event.proposed_tag == T.NON_LEXICAL

    def test_defers_to_laughter(self, wav_laughter):
        # Section 8: laughter is never [hn]. An overlapping candidate is
        # dropped in favour of the laughter event.
        buffer = load_audio(wav_laughter)
        spans = EnergyVad().analyse(buffer).speech
        laughs = LaughterDetector().detect(buffer, speech_spans=spans)
        hn = NonLexicalDetector().detect(
            buffer, speech_spans=spans, laughter_events=laughs
        )
        for event in hn:
            for laugh in laughs:
                assert event.end <= laugh.start or event.start >= laugh.end

    def test_note_excludes_breathing_and_laughter(self, wav_two_coughs):
        buffer = load_audio(wav_two_coughs)
        events = NonLexicalDetector().detect(buffer)
        assert events
        note = events[0].note
        assert "NOT laughter" in note
        assert "breathing" in note

    def test_confidence_capped(self, wav_two_coughs):
        buffer = load_audio(wav_two_coughs)
        for event in NonLexicalDetector().detect(buffer):
            assert event.confidence <= 0.65


class TestArtifactDetector:
    def test_finds_the_click(self, wav_click):
        buffer = load_audio(wav_click)
        events = ArtifactDetector().detect(buffer)
        assert events, "should find the 4 ms broadband click"
        assert any(abs(e.start - 1.0) < 0.1 for e in events)

    def test_one_click_one_event(self, wav_click):
        # A click straddling two frames must not be reported twice, or the
        # annotator is invited to write "[artifact] [artifact]".
        buffer = load_audio(wav_click)
        near = [e for e in ArtifactDetector().detect(buffer) if abs(e.start - 1.0) < 0.15]
        assert len(near) == 1

    def test_ignores_speech_onsets(self, wav_speech):
        # A speech onset is an abrupt broadband energy jump too. Without the
        # spike-versus-step test every utterance start becomes an artifact.
        buffer = load_audio(wav_speech)
        assert ArtifactDetector().detect(buffer) == []

    def test_ignores_steady_hiss(self, wav_hiss):
        # Section 9: [artifact] is not for persistent background hiss.
        buffer = load_audio(wav_hiss)
        assert ArtifactDetector().detect(buffer) == []

    def test_duration_is_bounded(self, wav_click):
        from app.detection.artifact import MAX_ARTIFACT_SECONDS

        buffer = load_audio(wav_click)
        for event in ArtifactDetector().detect(buffer):
            assert event.duration <= MAX_ARTIFACT_SECONDS

    def test_defers_to_vocal_events(self, wav_two_coughs):
        # Section 9 reserves [artifact] for technical noise. A cough is a
        # broadband transient but belongs to [hn].
        buffer = load_audio(wav_two_coughs)
        spans = EnergyVad().analyse(buffer).speech
        hn = NonLexicalDetector().detect(buffer, speech_spans=spans)
        events = ArtifactDetector().detect(buffer, vocal_events=hn)
        for event in events:
            for vocal in hn:
                assert event.end <= vocal.start or event.start >= vocal.end

    def test_note_states_the_ga_decision_rule(self, wav_click):
        buffer = load_audio(wav_click)
        events = ArtifactDetector().detect(buffer)
        assert events
        assert "<ga>" in events[0].note


class TestOverlapDetector:
    def test_events_are_uncertain(self, wav_overlap):
        buffer = load_audio(wav_overlap)
        for event in OverlapDetector().detect(buffer):
            assert event.type is EventType.POSSIBLE_OVERLAP
            assert event.requires_human_confirmation

    def test_no_overlap_in_sequential_speech(self, wav_two_speakers):
        # Turn-taking is not overlap.
        buffer = load_audio(wav_two_speakers)
        spans = EnergyVad().analyse(buffer).speech
        assert OverlapDetector().detect(buffer, speech_spans=spans) == []

    def test_no_overlap_in_single_speaker_audio(self, wav_speech):
        buffer = load_audio(wav_speech)
        spans = EnergyVad().analyse(buffer).speech
        assert OverlapDetector().detect(buffer, speech_spans=spans) == []

    def test_never_proposes_cross_talk(self, wav_overlap):
        # Section 19 and 56.16: intelligibility decides <ol> versus <ct>, and a
        # detector cannot judge it.
        buffer = load_audio(wav_overlap)
        for event in OverlapDetector().detect(buffer):
            assert event.proposed_tag == T.OVERLAP_OPEN
            assert event.proposed_tag != T.CROSS_TALK

    def test_note_offers_both_options(self, wav_overlap):
        buffer = load_audio(wav_overlap)
        events = OverlapDetector().detect(buffer)
        if events:
            assert "<ol>" in events[0].note and "<ct>" in events[0].note

    def test_confidence_is_lowest_of_all_detectors(self, wav_overlap):
        buffer = load_audio(wav_overlap)
        for event in OverlapDetector().detect(buffer):
            assert event.confidence <= 0.55


class TestBackgroundDetector:
    def test_events_are_uncertain(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        for event in BackgroundDetector().detect(buffer):
            assert event.type is EventType.POSSIBLE_BACKGROUND
            assert event.proposed_tag == T.BACKGROUND

    def test_quiet_speech_alone_is_not_background(self, tmp_path):
        # Section 21: do NOT use [bg] just because speech is quiet or distant.
        from tests.conftest import write_wav

        path = write_wav(
            tmp_path / "quiet.wav", speech(2.0, amplitude=0.03, seed=2, **VOICE_A)
        )
        buffer = load_audio(path)
        spans = EnergyVad().analyse(buffer).speech
        assert BackgroundDetector().detect(buffer, speech_spans=spans) == []

    def test_note_states_the_attribution_rule(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        events = BackgroundDetector().detect(buffer)
        if events:
            note = events[0].note
            assert "cannot be attributed" in note
            assert "(())" in note

    def test_confidence_capped(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        for event in BackgroundDetector().detect(buffer):
            assert event.confidence <= 0.60


# ------------------------------------------------------------------ pipeline
class TestPipeline:
    def test_produces_a_timeline(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = AnalysisPipeline().run(buffer)
        assert result.events
        assert result.speech_events
        assert "SPEECH" in result.timeline()

    def test_speaker_attribution_replaces_bare_speech(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        result = AnalysisPipeline().run(buffer)
        # Every speech event should carry a speaker once diarization ran.
        assert all(e.speaker_id for e in result.speech_events)

    def test_detects_two_speakers(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        assert AnalysisPipeline().run(buffer).speaker_count == 2

    def test_detects_one_speaker(self, wav_one_speaker_turns):
        buffer = load_audio(wav_one_speaker_turns)
        assert AnalysisPipeline().run(buffer).speaker_count == 1

    def test_pending_events_are_all_uncertain(self, wav_laughter):
        buffer = load_audio(wav_laughter)
        for event in AnalysisPipeline().run(buffer).pending:
            assert event.type.is_uncertain

    def test_no_event_produces_transcription_text(self, wav_two_speakers):
        # Layer A never yields words (section 3).
        buffer = load_audio(wav_two_speakers)
        for event in AnalysisPipeline().run(buffer).events:
            tag = event.proposed_tag
            assert tag is None or tag in T.ALL_FIXED_TAGS

    def test_pause_events_never_propose_a_tag(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        for event in AnalysisPipeline().run(buffer).events:
            if event.type in (EventType.PAUSE, EventType.SILENCE):
                assert event.proposed_tag is None

    def test_silence_yields_no_audio_candidate(self, wav_silence):
        buffer = load_audio(wav_silence)
        result = AnalysisPipeline().run(buffer)
        assert T.NO_AUDIO in result.whole_file_candidates

    def test_hiss_yields_no_speech_candidate(self, wav_hiss):
        buffer = load_audio(wav_hiss)
        result = AnalysisPipeline().run(buffer)
        assert T.NO_SPEECH in result.whole_file_candidates

    def test_summary_serializes(self, wav_two_speakers):
        import json

        buffer = load_audio(wav_two_speakers)
        payload = AnalysisPipeline().run(buffer).summary()
        json.dumps(payload)  # must not raise
        assert payload["speaker_count"] == 2

    def test_deterministic(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        first = AnalysisPipeline().run(buffer).summary()
        for _ in range(3):
            assert AnalysisPipeline().run(buffer).summary() == first

    def test_events_sorted_by_time(self, wav_two_speakers):
        buffer = load_audio(wav_two_speakers)
        events = AnalysisPipeline().run(buffer).events
        assert events == sorted(events, key=lambda e: (e.start, e.end, e.type.value))

    def test_empty_audio_does_not_crash(self, tmp_path):
        from tests.conftest import write_wav

        path = write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))
        result = AnalysisPipeline().run(load_audio(path))
        assert result.speaker_count == 0

    def test_no_network_access(self, wav_two_speakers, monkeypatch):
        # Section 53: analysis is entirely local.
        import socket

        def fail(*args, **kwargs):
            raise AssertionError("analysis must not open a socket")

        monkeypatch.setattr(socket, "socket", fail)
        monkeypatch.setattr(socket, "create_connection", fail)
        AnalysisPipeline().run(load_audio(wav_two_speakers))
