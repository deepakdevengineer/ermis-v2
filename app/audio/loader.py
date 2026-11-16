"""Local audio loading (CLAUDE.md sections 45, 53).

Decoding is entirely local. Two backends:

  - stdlib `wave` for PCM WAV, so the common case needs no external tool.
  - the `ffmpeg` executable for every other container.

`soundfile` is used when installed, purely as a faster path for formats it
handles. Nothing here reaches the network.

Audio is decoded to mono float32 in [-1, 1] at the analysis sample rate.
Downmixing and resampling apply to the *analysis* copy only; the original file
is never modified, and the original sample rate and channel count are preserved
in the returned metadata.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import AudioConfig

#: Amplitude below which a buffer is treated as containing no audio signal.
#: -90 dBFS is below the noise floor of any real recording, so a buffer under
#: it is digital silence rather than a quiet room.
_NO_SIGNAL_DBFS = -90.0


class AudioLoadError(RuntimeError):
    """Raised when a file cannot be decoded."""


@dataclass(frozen=True)
class AudioBuffer:
    """Decoded mono analysis audio plus the source file's real properties."""

    #: Mono float32 samples in [-1, 1], at `sample_rate`.
    samples: np.ndarray
    #: Sample rate of `samples` (the analysis rate, not necessarily the file's).
    sample_rate: int
    path: Path
    #: Properties of the file as it exists on disk.
    source_sample_rate: int
    source_channels: int
    codec: str = ""

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def frame_count(self) -> int:
        return int(len(self.samples))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0

    @property
    def peak_dbfs(self) -> float:
        return _to_dbfs(self.peak)

    @property
    def rms(self) -> float:
        if not self.samples.size:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    @property
    def rms_dbfs(self) -> float:
        return _to_dbfs(self.rms)

    @property
    def is_silent(self) -> bool:
        """True when the file carries no audio signal at all.

        This is the measurement behind the <na> whole-file condition
        (CLAUDE.md section 11). It is a measurement, not a decision: the
        annotator confirms <na>.
        """
        return self.peak_dbfs <= _NO_SIGNAL_DBFS

    # --------------------------------------------------------------- slicing
    def slice_seconds(self, start: float, end: float) -> np.ndarray:
        """Samples between two timestamps, clamped to the buffer.

        Backs "replay selected segment" in the editor (CLAUDE.md section 48).
        """
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        a = max(0, int(round(start * self.sample_rate)))
        b = min(len(self.samples), int(round(end * self.sample_rate)))
        return self.samples[a:b]

    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.samples), dtype=np.float64) / self.sample_rate

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "duration": round(self.duration, 4),
            "sample_rate": self.sample_rate,
            "source_sample_rate": self.source_sample_rate,
            "source_channels": self.source_channels,
            "codec": self.codec,
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "is_silent": self.is_silent,
        }


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return -math.inf
    return float(20.0 * math.log10(amplitude))


class AudioLoader:
    """Decodes audio files to mono analysis buffers."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

    # ------------------------------------------------------------------ entry
    def load(self, path: str | Path) -> AudioBuffer:
        p = Path(path)
        if not p.is_file():
            raise AudioLoadError(f"not a file: {p}")
        if p.suffix.lower() not in self.config.supported_extensions:
            raise AudioLoadError(
                f"unsupported extension {p.suffix!r}; supported: "
                f"{', '.join(self.config.supported_extensions)}"
            )

        duration = self.probe_duration(p)
        if duration is not None and duration > self.config.max_duration_seconds:
            raise AudioLoadError(
                f"{p.name} is {duration / 60:.1f} minutes, above the "
                f"{self.config.max_duration_seconds / 60:.0f} minute limit"
            )

        if p.suffix.lower() == ".wav":
            try:
                return self._load_wav_stdlib(p)
            except AudioLoadError:
                # Non-PCM WAV (for example a-law, or a WAV holding MP3 data).
                # ffmpeg handles those.
                pass

        if _soundfile() is not None:
            try:
                return self._load_soundfile(p)
            except Exception:
                pass

        return self._load_ffmpeg(p)

    # ------------------------------------------------------------- backends
    def _load_wav_stdlib(self, p: Path) -> AudioBuffer:
        try:
            with wave.open(str(p), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        except (wave.Error, EOFError, RuntimeError, OSError, ValueError) as exc:
            # A truncated or malformed RIFF header surfaces from stdlib `chunk`
            # as EOFError or a bare RuntimeError, not always as wave.Error.
            # All of them mean the same thing here: this is not readable PCM.
            raise AudioLoadError(
                f"{p.name}: not readable as PCM WAV ({type(exc).__name__}: {exc})"
            ) from exc

        data = _pcm_bytes_to_float32(frames, sample_width)
        if data is None:
            raise AudioLoadError(f"{p.name}: unsupported PCM width {sample_width * 8} bit")

        mono = _downmix(data, channels)
        resampled = self._resample(mono, rate)
        return AudioBuffer(
            samples=resampled,
            sample_rate=self.config.target_sample_rate,
            path=p,
            source_sample_rate=rate,
            source_channels=channels,
            codec=f"pcm_s{sample_width * 8}le",
        )

    def _load_soundfile(self, p: Path) -> AudioBuffer:
        sf = _soundfile()
        assert sf is not None
        data, rate = sf.read(str(p), dtype="float32", always_2d=True)
        channels = int(data.shape[1])
        mono = data.mean(axis=1).astype(np.float32) if channels > 1 else data[:, 0]
        return AudioBuffer(
            samples=self._resample(mono, rate),
            sample_rate=self.config.target_sample_rate,
            path=p,
            source_sample_rate=int(rate),
            source_channels=channels,
            codec="soundfile",
        )

    def _load_ffmpeg(self, p: Path) -> AudioBuffer:
        exe = shutil.which(self.config.ffmpeg_path)
        if exe is None:
            raise AudioLoadError(
                f"cannot decode {p.name}: {self.config.ffmpeg_path} is not on PATH. "
                "Install FFmpeg, or convert the file to PCM WAV."
            )
        info = self.probe(p)
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-i", str(p),
            "-map", "a:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", str(self.config.target_channels),
            "-ar", str(self.config.target_sample_rate),
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=False)
        except OSError as exc:
            raise AudioLoadError(f"failed to run ffmpeg: {exc}") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioLoadError(
                f"ffmpeg could not decode {p.name}: {detail[-1] if detail else 'unknown error'}"
            )
        samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
        return AudioBuffer(
            samples=samples,
            sample_rate=self.config.target_sample_rate,
            path=p,
            source_sample_rate=int(info.get("sample_rate") or self.config.target_sample_rate),
            source_channels=int(info.get("channels") or 1),
            codec=str(info.get("codec_name") or "ffmpeg"),
        )

    # -------------------------------------------------------------- resample
    def _resample(self, mono: np.ndarray, rate: int) -> np.ndarray:
        """Resample to the analysis rate.

        Uses scipy's polyphase filter when available and falls back to linear
        interpolation. The analysis copy only ever feeds VAD, silence detection
        and the waveform display, all of which tolerate either.
        """
        target = self.config.target_sample_rate
        if rate == target or mono.size == 0:
            return np.ascontiguousarray(mono, dtype=np.float32)

        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(rate), int(target))
            out = resample_poly(mono.astype(np.float64), target // g, rate // g)
            return np.ascontiguousarray(out, dtype=np.float32)
        except ImportError:
            n_out = int(round(mono.size * target / rate))
            if n_out <= 0:
                return np.zeros(0, dtype=np.float32)
            src = np.linspace(0.0, mono.size - 1, num=mono.size, dtype=np.float64)
            dst = np.linspace(0.0, mono.size - 1, num=n_out, dtype=np.float64)
            return np.interp(dst, src, mono).astype(np.float32)

    # ----------------------------------------------------------------- probe
    def probe(self, path: str | Path) -> dict:
        """Container metadata via ffprobe. Returns {} when unavailable."""
        exe = shutil.which(self.config.ffprobe_path)
        if exe is None:
            return {}
        cmd = [
            exe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
            if proc.returncode != 0:
                return {}
            payload = json.loads(proc.stdout or b"{}")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return {}

        streams = payload.get("streams") or [{}]
        info = dict(streams[0])
        fmt_duration = (payload.get("format") or {}).get("duration")
        if info.get("duration") is None and fmt_duration is not None:
            info["duration"] = fmt_duration
        return info

    def probe_duration(self, path: str | Path) -> float | None:
        """Duration in seconds without fully decoding, or None if unknown."""
        p = Path(path)
        if p.suffix.lower() == ".wav":
            try:
                with wave.open(str(p), "rb") as wf:
                    rate = wf.getframerate()
                    return wf.getnframes() / rate if rate else None
            except (wave.Error, EOFError, RuntimeError, OSError, ValueError):
                pass
        raw = self.probe(p).get("duration")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------- helpers
def _pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray | None:
    """Convert interleaved PCM bytes to float32 in [-1, 1]."""
    if sample_width == 1:
        # 8-bit WAV is unsigned.
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        return packed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        return (data / 2147483648.0).astype(np.float32)
    return None


def _downmix(data: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels into mono."""
    if channels <= 1:
        return data
    usable = (data.size // channels) * channels
    return data[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


_SOUNDFILE_CHECKED = False
_SOUNDFILE = None


def _soundfile():
    """Import soundfile once, returning None when it is not installed."""
    global _SOUNDFILE_CHECKED, _SOUNDFILE
    if not _SOUNDFILE_CHECKED:
        _SOUNDFILE_CHECKED = True
        try:
            import soundfile

            _SOUNDFILE = soundfile
        except ImportError:
            _SOUNDFILE = None
    return _SOUNDFILE


def load_audio(path: str | Path, config: AudioConfig | None = None) -> AudioBuffer:
    """Load `path` as a mono analysis buffer."""
    return AudioLoader(config).load(path)
