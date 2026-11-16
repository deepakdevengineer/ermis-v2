"""Audio loading and waveform extraction.

CLAUDE.md sections 45 (analysis engine), 48 (editor), 53 (privacy).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from app.audio.loader import AudioLoader, AudioLoadError, load_audio
from app.audio.waveform import DBFS_FLOOR, compute_waveform, render_ascii
from app.config import AudioConfig
from tests.conftest import tone, write_wav

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class TestWavLoading:
    def test_loads_mono_wav(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert buffer.duration == pytest.approx(1.0, abs=0.01)
        assert buffer.source_sample_rate == 44100
        assert buffer.source_channels == 1
        assert buffer.codec == "pcm_s16le"

    def test_resamples_to_the_analysis_rate(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert buffer.sample_rate == 16000
        assert buffer.frame_count == pytest.approx(16000, abs=100)

    def test_samples_are_mono_float32_in_range(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert buffer.samples.ndim == 1
        assert buffer.samples.dtype == np.float32
        assert np.all(np.abs(buffer.samples) <= 1.0)

    def test_stereo_is_downmixed(self, wav_stereo):
        buffer = load_audio(wav_stereo)
        assert buffer.source_channels == 2
        assert buffer.samples.ndim == 1
        assert buffer.duration == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_pcm_bit_depths(self, tmp_path, width):
        path = write_wav(tmp_path / f"w{width}.wav", tone(0.5), sample_width=width)
        buffer = load_audio(path)
        assert buffer.duration == pytest.approx(0.5, abs=0.02)
        assert buffer.peak > 0.1

    def test_original_file_is_not_modified(self, wav_tone):
        before = wav_tone.read_bytes()
        load_audio(wav_tone)
        assert wav_tone.read_bytes() == before


class TestLevels:
    def test_silence_is_detected(self, wav_silence):
        buffer = load_audio(wav_silence)
        assert buffer.is_silent
        assert buffer.peak == 0.0
        assert buffer.peak_dbfs == -np.inf

    def test_tone_is_not_silent(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert not buffer.is_silent
        assert -20 < buffer.peak_dbfs < 0

    def test_rms_below_peak(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert buffer.rms < buffer.peak

    def test_very_quiet_audio_is_not_reported_as_no_signal(self, tmp_path):
        # A faint room-noise clip is <ns>, not <na>. The loader must not
        # confuse "quiet" with "no signal" (sections 11 and 12).
        path = write_wav(tmp_path / "quiet.wav", tone(1.0, amp=0.002))
        buffer = load_audio(path)
        assert not buffer.is_silent


class TestSlicing:
    def test_slice_seconds(self, wav_speech_gap):
        buffer = load_audio(wav_speech_gap)
        segment = buffer.slice_seconds(0.0, 1.0)
        assert len(segment) == pytest.approx(16000, abs=50)

    def test_slice_of_the_silent_gap_is_quiet(self, wav_speech_gap):
        buffer = load_audio(wav_speech_gap)
        gap = buffer.slice_seconds(1.1, 1.5)
        speech = buffer.slice_seconds(0.1, 0.5)
        assert float(np.max(np.abs(gap))) < float(np.max(np.abs(speech)))

    def test_slice_clamps_to_the_buffer(self, wav_tone):
        buffer = load_audio(wav_tone)
        assert len(buffer.slice_seconds(-5.0, 99.0)) == buffer.frame_count

    def test_reversed_slice_rejected(self, wav_tone):
        with pytest.raises(ValueError):
            load_audio(wav_tone).slice_seconds(1.0, 0.5)


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(AudioLoadError, match="not a file"):
            load_audio(tmp_path / "nope.wav")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(AudioLoadError, match="unsupported extension"):
            load_audio(path)

    def test_duration_limit(self, wav_tone):
        config = AudioConfig()
        config.max_duration_seconds = 0.1
        with pytest.raises(AudioLoadError, match="limit"):
            AudioLoader(config).load(wav_tone)

    def test_corrupt_wav(self, tmp_path):
        path = tmp_path / "corrupt.wav"
        path.write_bytes(b"RIFF____WAVEjunkjunkjunk")
        with pytest.raises(AudioLoadError):
            load_audio(path)


class TestProbe:
    def test_probe_duration_of_wav_uses_stdlib(self, wav_tone):
        assert AudioLoader().probe_duration(wav_tone) == pytest.approx(1.0, abs=0.01)

    def test_probe_duration_of_missing_file_is_none(self, tmp_path):
        assert AudioLoader().probe_duration(tmp_path / "nope.wav") is None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
class TestFfmpegBackend:
    def test_loads_mp3(self, tmp_path, wav_tone):
        import subprocess

        mp3 = tmp_path / "tone.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(wav_tone), str(mp3)],
            check=True,
        )
        buffer = load_audio(mp3)
        assert buffer.sample_rate == 16000
        assert buffer.duration == pytest.approx(1.0, abs=0.1)
        assert not buffer.is_silent


class TestWaveform:
    def test_bucket_count(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets=100)
        assert waveform.bucket_count == 100

    def test_buckets_per_second(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets_per_second=50)
        assert waveform.bucket_count == pytest.approx(50, abs=1)

    def test_envelope_brackets_the_signal(self, wav_tone):
        buffer = load_audio(wav_tone)
        waveform = compute_waveform(buffer, buckets=50)
        assert np.all(waveform.minima <= waveform.maxima)
        assert float(waveform.maxima.max()) <= buffer.peak + 1e-6
        assert float(waveform.minima.min()) >= -buffer.peak - 1e-6

    def test_rms_within_the_peak_envelope(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets=50)
        peak = np.maximum(np.abs(waveform.minima), np.abs(waveform.maxima))
        assert np.all(waveform.rms <= peak + 1e-6)

    def test_silent_gap_shows_as_a_dip(self, wav_speech_gap):
        buffer = load_audio(wav_speech_gap)
        waveform = compute_waveform(buffer, buckets=300)
        gap = waveform.rms[waveform.time_to_bucket(1.1) : waveform.time_to_bucket(1.5)]
        speech = waveform.rms[waveform.time_to_bucket(0.1) : waveform.time_to_bucket(0.5)]
        assert float(gap.max()) < float(speech.mean())

    def test_brief_click_survives_bucketing(self, wav_speech_gap):
        # A 2 ms click must still stand above the surrounding tone at
        # full-file zoom, where one bucket spans ~15 ms.
        buffer = load_audio(wav_speech_gap)
        waveform = compute_waveform(buffer, buckets=200)
        click = float(waveform.maxima[waveform.time_to_bucket(2.0)])
        tone_level = float(np.median(waveform.maxima[:30]))
        assert click > tone_level * 1.5

    def test_envelope_preserves_extremes_rather_than_averaging(self, wav_speech_gap):
        # This is the reason the envelope stores per-bucket min/max: a mean
        # reduction over the same buckets loses the click entirely.
        buffer = load_audio(wav_speech_gap)
        waveform = compute_waveform(buffer, buckets=200)
        index = waveform.time_to_bucket(2.0)

        per_bucket = np.array_split(buffer.samples, 200)
        mean_reduced = float(np.abs(per_bucket[index].mean()))
        assert float(waveform.maxima[index]) > mean_reduced * 3

    def test_zoom_range(self, wav_speech_gap):
        buffer = load_audio(wav_speech_gap)
        waveform = compute_waveform(buffer, buckets=50, start=1.0, end=1.5)
        assert waveform.start == pytest.approx(1.0)
        assert waveform.end == pytest.approx(1.5)
        assert waveform.duration == pytest.approx(0.5)

    def test_range_clamped_to_the_buffer(self, wav_tone):
        buffer = load_audio(wav_tone)
        waveform = compute_waveform(buffer, buckets=10, start=0.0, end=99.0)
        assert waveform.end == pytest.approx(buffer.duration)

    def test_silence_yields_a_floored_dbfs(self, wav_silence):
        waveform = compute_waveform(load_audio(wav_silence), buckets=10)
        assert np.all(waveform.rms_dbfs() == DBFS_FLOOR)
        assert np.all(np.isfinite(waveform.rms_dbfs()))

    def test_more_buckets_than_samples(self, tmp_path):
        path = write_wav(tmp_path / "tiny.wav", tone(0.001))
        waveform = compute_waveform(load_audio(path), buckets=500)
        assert waveform.bucket_count == 500
        assert np.all(np.isfinite(waveform.maxima))

    def test_empty_audio(self, tmp_path):
        path = write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))
        waveform = compute_waveform(load_audio(path), buckets=20)
        assert waveform.bucket_count == 20
        assert float(np.abs(waveform.maxima).max()) == 0.0

    def test_time_and_bucket_mapping_round_trip(self, wav_speech_gap):
        waveform = compute_waveform(load_audio(wav_speech_gap), buckets=300)
        for seconds in (0.0, 0.5, 1.5, 2.9):
            index = waveform.time_to_bucket(seconds)
            assert abs(waveform.bucket_time(index) - seconds) <= waveform.seconds_per_bucket

    def test_time_to_bucket_clamps(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets=10)
        assert waveform.time_to_bucket(-99) == 0
        assert waveform.time_to_bucket(99) == 9


class TestAsciiRender:
    def test_renders_expected_shape(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets=40)
        art = render_ascii(waveform, width=40, height=9)
        lines = art.splitlines()
        assert len(lines) == 10  # 9 rows plus the ruler
        assert all(len(line) <= 41 for line in lines)

    def test_silence_renders_without_error(self, wav_silence):
        waveform = compute_waveform(load_audio(wav_silence), buckets=20)
        assert render_ascii(waveform, width=20, height=5)

    def test_ruler_shows_the_time_range(self, wav_speech_gap):
        waveform = compute_waveform(load_audio(wav_speech_gap), buckets=60)
        art = render_ascii(waveform, width=60)
        assert "00:00.00" in art
        assert "00:03.00" in art

    def test_even_height_gets_a_centre_row(self, wav_tone):
        waveform = compute_waveform(load_audio(wav_tone), buckets=20)
        assert len(render_ascii(waveform, width=20, height=8).splitlines()) == 10


class TestPrivacy:
    def test_loader_makes_no_network_calls(self, wav_tone, monkeypatch):
        # Section 53: local processing only. Any socket use is a failure.
        import socket

        def fail(*args, **kwargs):
            raise AssertionError("audio loading must not open a socket")

        monkeypatch.setattr(socket, "socket", fail)
        monkeypatch.setattr(socket, "create_connection", fail)
        buffer = load_audio(wav_tone)
        compute_waveform(buffer, buckets=50)
