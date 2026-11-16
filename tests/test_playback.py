"""Playback transport (CLAUDE.md section 48).

The engine is tested without an audio device by pulling output blocks directly
through `render_block`, which is why that method is separate from the audio
callback.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.loader import load_audio
from app.audio.playback import PlaybackEngine, PlaybackState
from app.models.event import AnalysisEvent, EventType


@pytest.fixture
def engine(wav_speech):
    playback = PlaybackEngine()
    playback.set_audio(load_audio(wav_speech))
    yield playback
    playback.stop()


class TestTransport:
    def test_starts_stopped(self, engine):
        assert engine.state is PlaybackState.STOPPED
        assert not engine.is_playing

    def test_play_and_pause(self, engine):
        engine.play()
        assert engine.is_playing
        engine.pause()
        assert engine.state is PlaybackState.PAUSED

    def test_toggle(self, engine):
        engine.toggle()
        assert engine.is_playing
        engine.toggle()
        assert not engine.is_playing

    def test_stop_resets_state(self, engine):
        engine.play()
        engine.stop()
        assert engine.state is PlaybackState.STOPPED

    def test_seek(self, engine):
        engine.seek(1.0)
        assert engine.position == pytest.approx(1.0, abs=0.01)

    def test_seek_clamps(self, engine):
        engine.seek(-5.0)
        assert engine.position == 0.0
        engine.seek(999.0)
        assert engine.position == pytest.approx(engine.duration, abs=0.01)

    def test_duration(self, engine):
        assert engine.duration == pytest.approx(2.0, abs=0.05)

    def test_no_audio_is_a_no_op(self):
        playback = PlaybackEngine()
        playback.play()
        assert not playback.is_playing


class TestRendering:
    def test_advances_the_playhead(self, engine):
        engine.play()
        engine.render_block(320)
        assert engine.position > 0

    def test_silent_when_stopped(self, engine):
        block = engine.render_block(320)
        assert np.all(block == 0.0)

    def test_block_length_is_exact(self, engine):
        engine.play()
        for frames in (64, 320, 1024):
            assert engine.render_block(frames).shape == (frames,)

    def test_output_matches_the_source_at_normal_speed(self, engine):
        engine.play(start=0.5)
        block = engine.render_block(256)
        expected = engine._buffer.slice_seconds(0.5, 0.5 + 256 / 16000)[:256]
        assert np.allclose(block[: expected.size], expected, atol=1e-6)

    def test_reaching_the_end_stops(self, engine):
        engine.play(start=engine.duration - 0.01)
        for _ in range(20):
            engine.render_block(320)
        assert not engine.is_playing

    def test_finished_callback_fires(self, engine):
        calls = []
        engine.set_finished_callback(lambda: calls.append(1))
        engine.play(start=engine.duration - 0.005)
        for _ in range(20):
            engine.render_block(320)
        assert calls


class TestSpeed:
    def test_default_is_normal(self, engine):
        assert engine.speed == 1.0

    @pytest.mark.parametrize("speed", [0.5, 0.75, 1.0, 1.5, 2.0])
    def test_configured_speeds(self, engine, speed):
        engine.set_speed(speed)
        assert engine.speed == speed

    def test_zero_and_negative_rejected(self, engine):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                engine.set_speed(bad)

    def test_slow_speed_consumes_less_source_audio(self, engine):
        # Section 14 expects the annotator to be able to slow audio down.
        engine.set_speed(0.5)
        engine.play(start=0.0)
        engine.render_block(1600)
        slow_position = engine.position

        engine.stop()
        engine.set_speed(1.0)
        engine.play(start=0.0)
        engine.render_block(1600)
        assert slow_position < engine.position

    def test_fast_speed_consumes_more(self, engine):
        engine.set_speed(2.0)
        engine.play(start=0.0)
        engine.render_block(800)
        fast = engine.position

        engine.stop()
        engine.set_speed(1.0)
        engine.play(start=0.0)
        engine.render_block(800)
        assert fast > engine.position

    def test_out_of_band_speed_is_clamped(self, engine):
        engine.set_speed(99.0)
        assert engine.speed <= 4.0


class TestSegments:
    def test_play_segment_bounds_playback(self, engine):
        engine.play_segment(0.5, 0.7)
        for _ in range(200):
            engine.render_block(320)
            if not engine.is_playing:
                break
        assert engine.position == pytest.approx(0.7, abs=0.02)

    def test_reversed_segment_rejected(self, engine):
        with pytest.raises(ValueError):
            engine.play_segment(1.0, 0.5)

    def test_loop_restarts_at_the_segment_start(self, engine):
        engine.play_segment(0.5, 0.6, loop=True)
        for _ in range(100):
            engine.render_block(320)
        # Still looping rather than stopped at the end.
        assert engine.is_playing
        assert 0.5 <= engine.position <= 0.61

    def test_replay_event_pads_for_context(self, engine):
        # A 0.05s candidate played bare is very hard to judge (section 47).
        event = AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.05)
        engine.replay_event(event, pad=0.25)
        assert engine.position == pytest.approx(0.75, abs=0.02)

    def test_replay_event_clamps_at_the_file_start(self, engine):
        event = AnalysisEvent(EventType.POSSIBLE_HN, 0.05, 0.1)
        engine.replay_event(event, pad=0.5)
        assert engine.position == 0.0


class TestDeviceIndependence:
    def test_available_is_a_boolean(self, engine):
        assert isinstance(engine.available, bool)

    def test_transport_works_without_a_device(self, engine, monkeypatch):
        # A machine with no output device must still track position, so the UI
        # and the test suite keep working.
        monkeypatch.setattr("app.audio.playback._sounddevice", lambda: None)
        engine.play()
        engine.render_block(320)
        assert engine.position > 0

    def test_context_manager_stops(self, wav_speech):
        with PlaybackEngine() as playback:
            playback.set_audio(load_audio(wav_speech))
            playback.play()
        assert not playback.is_playing

    def test_set_audio_resets(self, engine, wav_tone):
        engine.play()
        engine.set_audio(load_audio(wav_tone))
        assert engine.position == 0.0
        assert not engine.is_playing

    def test_set_audio_none(self, engine):
        engine.set_audio(None)
        assert engine.duration == 0.0
