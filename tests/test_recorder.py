"""Microphone capture (CLAUDE.md sections 45, 53, 59).

The recorder is tested without an audio device by starting it with
`open_stream=False` and pushing blocks through `write_block`, the same way the
playback tests pull blocks through `render_block`.

Two properties matter most here and both are asserted end to end:

  1. What lands on disk is exactly what was pushed in, because the file is the
     source of truth for every later replay and analysis (section 59).
  2. A recorded clip loads through the ordinary AudioLoader and runs through
     the ordinary analysis pipeline, so capture adds an input path and changes
     nothing downstream (section 3).
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from app.audio.loader import AudioLoader, load_audio
from app.audio.recorder import (
    Recorder,
    RecorderState,
    RecordingError,
    RecordingResult,
    _float_to_pcm,
)
from app.config import AppConfig, AudioConfig, RecordingConfig
from tests.conftest import VOICE_A, speech

RATE = 16_000


def make_recorder(tmp_path, **overrides) -> Recorder:
    """A recorder writing into tmp_path at a test-friendly rate."""
    config = RecordingConfig(sample_rate=RATE, **overrides)
    return Recorder(AudioConfig(), config, output_dir=tmp_path)


def blocks(data: np.ndarray, size: int = 1024):
    """Split a signal into capture-sized blocks."""
    for start in range(0, len(data), size):
        yield data[start : start + size]


def push(recorder: Recorder, data: np.ndarray, size: int = 1024) -> int:
    return sum(recorder.write_block(block) for block in blocks(data, size))


class TestLifecycle:
    def test_starts_idle(self, tmp_path):
        recorder = make_recorder(tmp_path)
        assert recorder.state is RecorderState.IDLE
        assert not recorder.is_active
        assert recorder.elapsed == 0.0

    def test_start_creates_the_file_and_reports_recording(self, tmp_path):
        recorder = make_recorder(tmp_path)
        path = recorder.start(open_stream=False)
        try:
            assert path.exists()
            assert recorder.state is RecorderState.RECORDING
            assert recorder.is_recording
        finally:
            recorder.stop()

    def test_auto_named_file_uses_the_prefix_and_lands_in_output_dir(self, tmp_path):
        recorder = make_recorder(tmp_path, filename_prefix="clip")
        path = recorder.start(open_stream=False)
        try:
            assert path.parent == tmp_path
            assert path.name.startswith("clip-")
            assert path.suffix == ".wav"
        finally:
            recorder.stop()

    def test_non_wav_extension_is_corrected(self, tmp_path):
        # Capture is PCM WAV, so the name has to match what is in the file --
        # the loader dispatches on the extension.
        recorder = make_recorder(tmp_path)
        path = recorder.start(tmp_path / "take.mp3", open_stream=False)
        try:
            assert path.suffix == ".wav"
        finally:
            recorder.stop()

    def test_starting_twice_is_refused(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        try:
            with pytest.raises(RecordingError, match="already recording"):
                recorder.start(open_stream=False)
        finally:
            recorder.stop()

    def test_stop_without_start_returns_none(self, tmp_path):
        assert make_recorder(tmp_path).stop() is None

    def test_context_manager_stops_capture(self, tmp_path):
        with make_recorder(tmp_path) as recorder:
            recorder.start(open_stream=False)
            push(recorder, np.zeros(RATE, dtype=np.float32))
        assert recorder.state is RecorderState.IDLE

    def test_write_block_after_stop_is_ignored(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, speech(0.2, **VOICE_A))
        recorder.stop()
        assert recorder.write_block(np.ones(128, dtype=np.float32)) == 0


class TestCapturedFile:
    def test_duration_matches_what_was_pushed(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(tmp_path / "take.wav", open_stream=False)
        push(recorder, speech(1.0, sample_rate=RATE, **VOICE_A))
        result = recorder.stop()

        assert result is not None
        assert result.frames == RATE
        assert result.duration == pytest.approx(1.0, abs=0.01)
        assert result.sample_rate == RATE
        assert result.channels == 1

    def test_file_is_readable_pcm_wav_with_the_right_header(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(tmp_path / "take.wav", open_stream=False)
        push(recorder, speech(0.5, sample_rate=RATE, **VOICE_A))
        result = recorder.stop()

        assert result is not None
        with wave.open(str(result.path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == RATE
            assert wf.getnframes() == result.frames

    def test_samples_survive_the_round_trip(self, tmp_path):
        signal = speech(0.5, sample_rate=RATE, **VOICE_A)
        recorder = make_recorder(tmp_path)
        recorder.start(tmp_path / "take.wav", open_stream=False)
        push(recorder, signal)
        result = recorder.stop()

        assert result is not None
        buffer = AudioLoader(AudioConfig(target_sample_rate=RATE)).load(result.path)
        # 16-bit quantisation is the only loss; the shape must be identical.
        assert buffer.samples.size == signal.size
        assert np.max(np.abs(buffer.samples - signal)) < 1e-3

    def test_recorded_clip_loads_and_analyses_like_any_other_file(self, tmp_path):
        """Capture adds an input path and changes nothing downstream."""
        from app.detection.pipeline import AnalysisPipeline

        recorder = make_recorder(tmp_path)
        recorder.start(tmp_path / "take.wav", open_stream=False)
        push(recorder, speech(1.5, sample_rate=RATE, **VOICE_A))
        result = recorder.stop()

        assert result is not None
        buffer = load_audio(result.path)
        analysis = AnalysisPipeline().run(buffer)
        assert analysis.speech_events, "captured speech should produce speech regions"

    @pytest.mark.parametrize("bits,width", [(16, 2), (24, 3), (32, 4)])
    def test_every_supported_width_is_written_and_readable(self, tmp_path, bits, width):
        signal = speech(0.3, sample_rate=RATE, **VOICE_A)
        recorder = make_recorder(tmp_path, sample_width_bits=bits)
        recorder.start(tmp_path / f"take{bits}.wav", open_stream=False)
        push(recorder, signal)
        result = recorder.stop()

        assert result is not None
        with wave.open(str(result.path), "rb") as wf:
            assert wf.getsampwidth() == width
        buffer = AudioLoader(AudioConfig(target_sample_rate=RATE)).load(result.path)
        assert np.max(np.abs(buffer.samples - signal)) < 1e-3

    def test_stereo_capture_is_written_interleaved(self, tmp_path):
        left = speech(0.4, sample_rate=RATE, **VOICE_A)
        stereo = np.stack([left, left * 0.5], axis=1).astype(np.float32)

        recorder = make_recorder(tmp_path, channels=2)
        recorder.start(tmp_path / "stereo.wav", open_stream=False)
        for start in range(0, len(stereo), 512):
            recorder.write_block(stereo[start : start + 512])
        result = recorder.stop()

        assert result is not None
        assert result.channels == 2
        assert result.frames == len(left)
        with wave.open(str(result.path), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getnframes() == len(left)
        # The loader downmixes to mono, so the average of the two channels is
        # what analysis sees.
        buffer = AudioLoader(AudioConfig(target_sample_rate=RATE)).load(result.path)
        assert np.max(np.abs(buffer.samples - left * 0.75)) < 1e-3


class TestMeasurements:
    def test_peak_and_rms_reflect_the_capture(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, np.full(RATE, 0.5, dtype=np.float32))
        result = recorder.stop()

        assert result is not None
        assert result.peak == pytest.approx(0.5, abs=1e-4)
        assert result.rms == pytest.approx(0.5, abs=1e-3)
        assert result.peak_dbfs == pytest.approx(-6.02, abs=0.05)

    def test_silence_is_reported_as_no_signal(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, np.zeros(RATE, dtype=np.float32))
        result = recorder.stop()

        assert result is not None
        assert result.is_silent
        assert any("<na>" in note for note in result.notes())

    def test_quiet_capture_is_flagged_without_claiming_a_tag(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, np.full(RATE, 0.002, dtype=np.float32))
        result = recorder.stop()

        assert result is not None
        assert not result.is_silent
        notes = " ".join(result.notes())
        assert "very quiet" in notes

    def test_clipping_is_counted_and_left_to_the_human(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, np.ones(2_000, dtype=np.float32))
        result = recorder.stop()

        assert result is not None
        assert result.clipped_frames == 2_000
        notes = " ".join(result.notes())
        # The recorder measures; it must not decide between [artifact] and <ga>.
        assert "[artifact]" in notes and "<ga>" in notes

    def test_level_meter_reads_and_resets(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        try:
            recorder.write_block(np.full(512, 0.25, dtype=np.float32))
            assert recorder.read_level() == pytest.approx(0.25, abs=1e-4)
            # Nothing pushed since, so the window is empty again.
            assert recorder.read_level() == 0.0
        finally:
            recorder.stop()

    def test_empty_capture_writes_no_file(self, tmp_path):
        recorder = make_recorder(tmp_path)
        path = recorder.start(tmp_path / "empty.wav", open_stream=False)
        assert recorder.stop() is None
        assert not path.exists()

    def test_result_dict_is_json_shaped(self, tmp_path):
        import json

        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, speech(0.2, sample_rate=RATE, **VOICE_A))
        result = recorder.stop()

        assert result is not None
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["sample_rate"] == RATE
        assert payload["duration"] > 0


class TestPauseAndLimits:
    def test_paused_capture_discards_frames(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        push(recorder, np.full(RATE // 2, 0.4, dtype=np.float32))
        recorder.pause()
        assert recorder.state is RecorderState.PAUSED
        assert push(recorder, np.full(RATE, 0.4, dtype=np.float32)) == 0
        recorder.resume()
        push(recorder, np.full(RATE // 2, 0.4, dtype=np.float32))
        result = recorder.stop()

        assert result is not None
        assert result.frames == RATE
        assert result.duration == pytest.approx(1.0, abs=0.01)

    def test_toggle_pause_round_trips(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        try:
            recorder.toggle_pause()
            assert recorder.state is RecorderState.PAUSED
            recorder.toggle_pause()
            assert recorder.state is RecorderState.RECORDING
        finally:
            recorder.stop()

    def test_capture_stops_at_the_duration_ceiling(self, tmp_path):
        recorder = make_recorder(tmp_path, max_duration_seconds=0.5)
        recorder.start(open_stream=False)
        kept = push(recorder, np.full(RATE * 2, 0.3, dtype=np.float32))

        assert kept == RATE // 2
        assert recorder.limit_reached
        result = recorder.stop()
        assert result is not None
        assert result.truncated
        assert result.duration == pytest.approx(0.5, abs=0.01)
        # The audio captured before the ceiling is kept, not discarded.
        assert result.path.exists()

    def test_elapsed_tracks_captured_audio(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(open_stream=False)
        try:
            push(recorder, np.zeros(RATE // 4, dtype=np.float32))
            assert recorder.elapsed == pytest.approx(0.25, abs=0.01)
        finally:
            recorder.stop()


class TestCancel:
    def test_cancel_deletes_the_partial_file(self, tmp_path):
        recorder = make_recorder(tmp_path)
        path = recorder.start(tmp_path / "scrap.wav", open_stream=False)
        push(recorder, speech(0.3, sample_rate=RATE, **VOICE_A))
        recorder.cancel()

        assert recorder.state is RecorderState.IDLE
        assert not path.exists()

    def test_cancel_when_idle_is_a_no_op(self, tmp_path):
        make_recorder(tmp_path).cancel()

    def test_a_new_capture_can_start_after_cancel(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.start(tmp_path / "first.wav", open_stream=False)
        recorder.cancel()
        second = recorder.start(tmp_path / "second.wav", open_stream=False)
        push(recorder, speech(0.2, sample_rate=RATE, **VOICE_A))
        result = recorder.stop()
        assert result is not None
        assert result.path == second


class TestDevices:
    def test_device_listing_degrades_without_a_backend(self, tmp_path):
        """No audio backend must be a clear no, not a crash."""
        import app.audio.recorder as module

        recorder = make_recorder(tmp_path)
        saved = (module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE)
        module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE = True, None
        try:
            assert not recorder.available
            assert recorder.list_devices() == []
            assert recorder.default_device() is None
            with pytest.raises(RecordingError, match="sounddevice"):
                recorder.start(tmp_path / "x.wav")
        finally:
            module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE = saved

    def test_failed_start_leaves_no_file_behind(self, tmp_path):
        import app.audio.recorder as module

        recorder = make_recorder(tmp_path)
        saved = (module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE)
        module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE = True, None
        try:
            with pytest.raises(RecordingError):
                recorder.start(tmp_path / "never.wav")
        finally:
            module._SOUNDDEVICE_CHECKED, module._SOUNDDEVICE = saved
        assert not (tmp_path / "never.wav").exists()
        assert recorder.state is RecorderState.IDLE


class TestPcmConversion:
    @pytest.mark.parametrize("bits,per_frame", [(16, 2), (24, 3), (32, 4)])
    def test_byte_width_per_sample(self, bits, per_frame):
        data = np.zeros(10, dtype=np.float32)
        assert len(_float_to_pcm(data, bits)) == 10 * per_frame

    def test_out_of_range_samples_are_clipped_not_wrapped(self):
        raw = _float_to_pcm(np.array([2.0, -2.0], dtype=np.float32), 16)
        values = np.frombuffer(raw, dtype="<i2")
        assert values[0] == 32767
        assert values[1] == -32767

    def test_unsupported_width_is_refused(self):
        with pytest.raises(RecordingError, match="unsupported capture width"):
            _float_to_pcm(np.zeros(4, dtype=np.float32), 8)


class TestConfig:
    def test_recording_defaults_are_local_and_mono(self):
        recording = AppConfig().recording
        assert recording.channels == 1
        assert recording.sample_width_bits in (16, 24, 32)
        assert recording.max_duration_seconds > 0
        assert recording.device is None

    def test_recordings_dir_is_under_data(self):
        paths = AppConfig().paths
        assert paths.recordings_dir.parent == paths.data_dir

    def test_ensure_creates_the_recordings_dir(self, tmp_path):
        config = AppConfig()
        for name in (
            "data_dir",
            "projects_dir",
            "models_dir",
            "log_dir",
            "exports_dir",
            "recordings_dir",
        ):
            setattr(config.paths, name, tmp_path / name)
        config.paths.database_path = tmp_path / "db.sqlite3"
        config.paths.ensure()
        assert (tmp_path / "recordings_dir").is_dir()

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("channels", 3, "1 or 2"),
            ("sample_width_bits", 8, "16, 24 or 32"),
            ("sample_rate", 0, "positive"),
            ("max_duration_seconds", 0, "positive"),
        ],
    )
    def test_invalid_recording_settings_are_rejected(self, field, value, match):
        config = AppConfig()
        setattr(config.recording, field, value)
        with pytest.raises(ValueError, match=match):
            config.validate()

    def test_recording_section_round_trips_through_json(self, tmp_path):
        path = tmp_path / "config.json"
        config = AppConfig()
        config.recording.sample_rate = 44_100
        config.recording.filename_prefix = "session"
        config.save(path)

        loaded = AppConfig.load(path)
        assert loaded.recording.sample_rate == 44_100
        assert loaded.recording.filename_prefix == "session"


class TestAuditProvenance:
    def test_a_recording_is_recorded_as_captured_not_merely_loaded(self, db, audio_row):
        """The audit trail says where the audio came from (section 50)."""
        from app.storage.audit import Action, AuditLog

        result = RecordingResult(
            path=tmp_wav_name(),
            sample_rate=RATE,
            channels=1,
            frames=RATE,
            peak=0.5,
            rms=0.2,
            clipped_frames=0,
        )
        AuditLog(db).recording_captured(audio_row, result)

        rows = db.list_audit(audio_row)
        assert [r["action"] for r in rows] == [Action.RECORDING_CAPTURED]
        assert rows[0]["audio_end"] == pytest.approx(1.0)


def tmp_wav_name():
    from pathlib import Path

    return Path("recording-20260101-000000.wav")
