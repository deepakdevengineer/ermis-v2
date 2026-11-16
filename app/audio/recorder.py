"""Local microphone capture (CLAUDE.md sections 45, 53, 59).

Lets the annotator produce a clip with the machine's own input device instead
of only opening a file that arrived as an attachment. Capture is the one new
I/O path; everything after it is unchanged:

    microphone -> WAV under paths.recordings_dir -> AudioLoader -> analysis

Writing the capture to disk first and loading it back through the ordinary
loader is deliberate. Section 59 makes the audio the source of truth, so the
file on disk is what gets analysed, replayed and archived: a recorded clip and
an opened clip follow exactly the same path, and the annotator can replay the
recording later and hear precisely what the detectors heard.

Recording captures audio. It does not transcribe it: there is no ASR here and
a recorded clip enters the same two-layer workflow as any other file
(section 3). The final transcription is still typed by a person.

`sounddevice` is imported lazily, so importing this module never requires an
audio backend. Without one, `Recorder.available` is False and `start()` says so
instead of failing obscurely. Capture never leaves the machine: samples go from
the input device to a local file and nowhere else (section 53).
"""

from __future__ import annotations

import logging
import math
import queue
import re
import threading
import wave
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from app.config import AudioConfig, PathsConfig, RecordingConfig

log = logging.getLogger("ermis.recorder")

#: Frames requested per input callback. ~20 ms at 48 kHz: short enough that the
#: level meter tracks the voice, long enough not to starve the writer thread.
_BLOCK_FRAMES = 1024

#: Same threshold the loader uses for "this carries no audio signal at all".
_NO_SIGNAL_DBFS = -90.0

#: A capture peaking below this is suspiciously quiet -- usually a muted input
#: or the wrong device, which is worth saying before the annotator starts
#: transcribing near-silence.
_QUIET_PEAK_DBFS = -45.0

#: Names that suggest a Bluetooth capture endpoint. Windows exposes a headset
#: microphone through the Hands-Free profile, which it labels "Headset (... )"
#: or "... Hands-Free AG Audio"; the stereo A2DP side is output only and never
#: carries a microphone.
_BLUETOOTH_HINT = re.compile(
    r"hands[\s-]*free|headset|bluetooth|bt|airpods|buds|AG Audio",
    re.IGNORECASE,
)

#: Rates to try when the configured one is refused, after the device's own
#: default. A Bluetooth headset microphone runs the Hands-Free profile, which is
#: 8 kHz narrowband or 16 kHz wideband -- it will refuse 48 kHz outright, so
#: without these a perfectly good headset mic looks broken.
_FALLBACK_RATES = (48_000, 44_100, 32_000, 16_000, 8_000)


class RecordingError(RuntimeError):
    """Raised when capture cannot start, or is asked for out of order."""


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


@dataclass(frozen=True)
class InputDevice:
    """One capture device offered by the host audio backend.

    `host_api` matters on Windows, where the same physical microphone is
    reported once per host API (MME, DirectSound, WASAPI, WDM-KS). A Bluetooth
    headset in particular behaves differently between them, so the name alone
    does not identify a device -- the annotator needs to see which backend a
    duplicate belongs to in order to pick one that opens.
    """

    index: int
    name: str
    channels: int
    default_sample_rate: float
    host_api: str = ""
    #: True when the backend reported no usable name for the endpoint. Windows
    #: does this for a paired-but-disconnected Bluetooth device, which is worth
    #: showing rather than hiding: it is often the device the user is looking
    #: for, and it explains why selecting it fails.
    unnamed: bool = False

    @property
    def is_bluetooth(self) -> bool:
        """True when the name looks like a Bluetooth capture endpoint.

        A guess from the name, used only to word a hint -- never to filter a
        device out.
        """
        return bool(_BLUETOOTH_HINT.search(self.name))

    def __str__(self) -> str:
        host = f", {self.host_api}" if self.host_api else ""
        return (
            f"[{self.index}] {self.name} "
            f"({self.channels} ch, {self.default_sample_rate:.0f} Hz{host})"
        )


@dataclass(frozen=True)
class RecordingResult:
    """What a finished capture produced. Measurements only, never decisions."""

    path: Path
    sample_rate: int
    channels: int
    frames: int
    peak: float
    rms: float
    clipped_frames: int
    #: True when capture stopped because it hit `max_duration_seconds`.
    truncated: bool = False
    #: Input blocks the backend reported as overflowed, i.e. dropped samples.
    overflows: int = 0

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def peak_dbfs(self) -> float:
        return _to_dbfs(self.peak)

    @property
    def rms_dbfs(self) -> float:
        return _to_dbfs(self.rms)

    @property
    def is_silent(self) -> bool:
        """True when nothing audible was captured (the <na> measurement)."""
        return self.peak_dbfs <= _NO_SIGNAL_DBFS

    def notes(self) -> list[str]:
        """Human-readable observations about the capture.

        These describe the recording, they do not choose a tag. Whether a noisy
        capture is `[artifact]` or `<ga>`, or a blank one is `<na>`, stays a
        human decision (sections 9, 10, 11, 47).
        """
        out: list[str] = []
        if self.frames == 0:
            return ["nothing was captured"]
        if self.is_silent:
            out.append(
                "no audio signal captured -- check the input device; <na> may "
                "apply if the clip really is blank (section 11)"
            )
        elif self.peak_dbfs < _QUIET_PEAK_DBFS:
            out.append(
                f"very quiet capture (peak {self.peak_dbfs:.1f} dBFS) -- check "
                "the input level before transcribing"
            )
        if self.clipped_frames:
            share = 100.0 * self.clipped_frames / max(1, self.frames)
            out.append(
                f"{self.clipped_frames} clipped sample(s) ({share:.2f}%) -- "
                "listen before deciding between [artifact] and <ga> "
                "(sections 9 and 10)"
            )
        if self.overflows:
            out.append(
                f"{self.overflows} input overflow(s) -- the audio backend "
                "dropped samples, so the recording has gaps"
            )
        if self.truncated:
            out.append("capture stopped at the configured maximum duration")
        return out

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "duration": round(self.duration, 4),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "peak_dbfs": round(self.peak_dbfs, 2) if self.peak > 0 else None,
            "rms_dbfs": round(self.rms_dbfs, 2) if self.rms > 0 else None,
            "clipped_frames": self.clipped_frames,
            "overflows": self.overflows,
            "truncated": self.truncated,
            "is_silent": self.is_silent,
        }


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return -math.inf
    return float(20.0 * math.log10(amplitude))


@dataclass
class _Accumulator:
    """Running measurements over the capture, updated per block."""

    frames: int = 0
    peak: float = 0.0
    sum_squares: float = 0.0
    clipped: int = 0
    #: Peak since the level meter last read it.
    window_peak: float = 0.0
    overflows: int = 0
    truncated: bool = False

    def rms(self) -> float:
        if self.frames == 0:
            return 0.0
        return float(math.sqrt(self.sum_squares / self.frames))


class Recorder:
    """Captures the input device to a local WAV file.

    The device callback only measures and enqueues; a writer thread does the
    file I/O. That keeps the audio thread short -- an overrun there costs
    samples -- and keeps memory flat, so a long session streams to disk instead
    of accumulating in RAM.

    Capture can also be driven with no audio device at all by starting with
    `open_stream=False` and pushing blocks through `write_block`. That is how
    the test suite exercises the recorder, mirroring
    `PlaybackEngine.render_block` on the output side.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        recording: RecordingConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.recording = recording or RecordingConfig()
        self.output_dir = Path(output_dir or PathsConfig().recordings_dir)

        self._state = RecorderState.IDLE
        self._lock = threading.RLock()
        self._path: Path | None = None
        self._sample_rate = int(self.recording.sample_rate)
        self._channels = int(self.recording.channels)
        self._acc = _Accumulator()

        self._stream = None
        self._writer: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._wave = None
        self._writer_error: BaseException | None = None

    # ------------------------------------------------------------ properties
    @property
    def available(self) -> bool:
        """True when an audio input backend is importable."""
        return _sounddevice() is not None

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state is RecorderState.RECORDING

    @property
    def is_active(self) -> bool:
        """True while a capture is open, whether running or paused."""
        return self._state is not RecorderState.IDLE

    @property
    def path(self) -> Path | None:
        """Where the in-progress (or last finished) capture is written."""
        return self._path

    @property
    def sample_rate(self) -> int:
        """Rate capture actually opened at, which may not be the requested one."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def elapsed(self) -> float:
        """Seconds captured so far. Counts audio, not wall-clock.

        A paused capture stops advancing and discarded blocks are not counted,
        so this always matches the length of the file being written.
        """
        with self._lock:
            return self._acc.frames / self._sample_rate if self._sample_rate else 0.0

    @property
    def limit_reached(self) -> bool:
        """True once `max_duration_seconds` has been hit.

        Capture stops accepting frames at that point but the stream stays open,
        because closing a device from inside its own callback can deadlock. The
        owner polls this and calls `stop()`.
        """
        with self._lock:
            return self._acc.truncated

    # --------------------------------------------------------------- devices
    def list_devices(self, *, rescan: bool = False) -> list[InputDevice]:
        """Capture devices the backend reports. Empty when there is no backend.

        PortAudio enumerates devices once, when it initialises, and then serves
        that snapshot for the life of the process. A Bluetooth headset switched
        on after the window opened is therefore invisible until the library is
        re-initialised, which is what `rescan=True` does -- without it, "re-scan
        devices" would return the same stale list every time.
        """
        sd = _sounddevice()
        if sd is None:
            return []
        if rescan:
            rescan_devices()
        try:
            raw = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:  # pragma: no cover - backend specific
            log.warning("could not query input devices: %s", exc)
            return []
        devices: list[InputDevice] = []
        for index, info in enumerate(raw):
            channels = int(info.get("max_input_channels") or 0)
            if channels <= 0:
                continue
            raw_name = str(info.get("name") or "").strip()
            unnamed = _is_unnamed(raw_name)
            devices.append(
                InputDevice(
                    index=index,
                    name=raw_name or f"device {index}",
                    channels=channels,
                    default_sample_rate=float(info.get("default_samplerate") or 0.0),
                    host_api=_host_api_name(host_apis, info.get("hostapi")),
                    unnamed=unnamed,
                )
            )
        return devices

    def bluetooth_devices(self, *, rescan: bool = False) -> list[InputDevice]:
        """Capture devices whose name looks like a Bluetooth headset mic."""
        return [d for d in self.list_devices(rescan=rescan) if d.is_bluetooth]

    def default_device(self) -> InputDevice | None:
        """The system's default capture device, or None."""
        sd = _sounddevice()
        if sd is None:
            return None
        try:
            index = sd.default.device[0]
        except Exception:  # pragma: no cover - backend specific
            return None
        if index is None or index < 0:
            return None
        for device in self.list_devices():
            if device.index == index:
                return device
        return None

    # ----------------------------------------------------------------- start
    def start(
        self,
        path: str | Path | None = None,
        *,
        device: int | None = None,
        open_stream: bool = True,
    ) -> Path:
        """Begin capturing to `path` (auto-named under `output_dir` if omitted).

        With `open_stream=False` no device is touched and frames must be pushed
        through `write_block`; everything else behaves identically.
        """
        with self._lock:
            if self.is_active:
                raise RecordingError(
                    f"already {self._state.value}; stop the current capture first"
                )
            target = self._resolve_path(path)
            self._path = target
            self._acc = _Accumulator()
            self._writer_error = None
            self._queue = queue.Queue()
            self._sample_rate = int(self.recording.sample_rate)
            self._channels = int(self.recording.channels)

        if open_stream:
            self._open_stream(device)

        # Opened after the stream, so a device failure leaves no empty file.
        self._open_wave(target)
        self._start_writer()

        with self._lock:
            self._state = RecorderState.RECORDING
        if self._stream is not None:
            self._stream.start()
        log.info(
            "recording to %s at %d Hz, %d channel(s)",
            target,
            self._sample_rate,
            self._channels,
        )
        return target

    def _resolve_path(self, path: str | Path | None) -> Path:
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.output_dir / f"{self.recording.filename_prefix}-{stamp}.wav"
        target = Path(path)
        if target.suffix.lower() != ".wav":
            # Capture is written as PCM WAV, so the extension has to match what
            # is really in the file -- the loader dispatches on it.
            target = target.with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _open_stream(self, device: int | None) -> None:
        sd = _sounddevice()
        if sd is None:
            raise RecordingError(
                "microphone capture needs the sounddevice package and a working "
                "PortAudio library. Install it with `pip install sounddevice`, "
                "or open an existing audio file instead."
            )
        index = self.recording.device if device is None else device
        info = self._device_info(index)

        # A Bluetooth headset microphone is mono, so asking for the configured
        # channel count fails on a device that only has one. Clamp instead: the
        # analysis copy is mono anyway, and a refused open here reads to the
        # annotator as "my headset does not work" (section 59 -- capture the
        # audio that exists rather than the audio that was expected).
        available = int((info or {}).get("max_input_channels") or 0)
        if available and available < self._channels:
            log.info(
                "device offers %d input channel(s); capturing %d instead of %d",
                available,
                available,
                self._channels,
            )
            self._channels = available

        attempts: list[int] = [int(self.recording.sample_rate)]
        default_rate = float((info or {}).get("default_samplerate") or 0.0)
        if default_rate and int(default_rate) not in attempts:
            attempts.append(int(default_rate))
        # Then the common capture rates, ending at the Hands-Free profile rates
        # a Bluetooth headset microphone actually runs at.
        attempts.extend(rate for rate in _FALLBACK_RATES if rate not in attempts)

        last: Exception | None = None
        for rate in attempts:
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=self._channels,
                    dtype="float32",
                    blocksize=_BLOCK_FRAMES,
                    device=index,
                    callback=self._callback,
                )
            except Exception as exc:
                last = exc
                continue
            self._stream = stream
            self._sample_rate = int(rate)
            if rate != int(self.recording.sample_rate):
                log.info(
                    "device refused %d Hz; capturing at %d Hz instead",
                    self.recording.sample_rate,
                    rate,
                )
            return

        self._state = RecorderState.IDLE
        raise RecordingError(_open_failure_message(info, last))

    def _device_info(self, index: int | None) -> dict | None:
        """Backend description of one input device, or None."""
        sd = _sounddevice()
        if sd is None:
            return None
        try:
            if index is None:
                return dict(sd.query_devices(kind="input"))
            return dict(sd.query_devices(index, "input"))
        except Exception:
            return None

    def _device_default_rate(self, index: int | None) -> float | None:
        info = self._device_info(index)
        if not info:
            return None
        return float(info.get("default_samplerate") or 0.0) or None

    # ---------------------------------------------------------------- writer
    def _open_wave(self, target: Path) -> None:
        width = self.recording.sample_width_bits // 8
        try:
            handle = wave.open(str(target), "wb")
            handle.setnchannels(self._channels)
            handle.setsampwidth(width)
            handle.setframerate(self._sample_rate)
        except (OSError, wave.Error, ValueError) as exc:
            self._close_stream()
            self._state = RecorderState.IDLE
            raise RecordingError(f"cannot write {target}: {exc}") from exc
        self._wave = handle

    def _start_writer(self) -> None:
        self._writer = threading.Thread(
            target=self._writer_loop, name="ermis-recorder-writer", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        """Drain the queue into the WAV file until the sentinel arrives."""
        while True:
            block = self._queue.get()
            if block is None:
                return
            try:
                if self._wave is not None:
                    self._wave.writeframes(
                        _float_to_pcm(block, self.recording.sample_width_bits)
                    )
            except BaseException as exc:  # pragma: no cover - disk failure
                # Keep draining so stop() is never blocked behind a full queue;
                # the error surfaces from stop().
                self._writer_error = exc
                log.error("recorder writer failed: %s", exc)

    # ----------------------------------------------------------------- input
    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover
        """Input callback. Runs on the audio thread: measure and enqueue only."""
        if status and getattr(status, "input_overflow", False):
            with self._lock:
                self._acc.overflows += 1
        self.write_block(indata)

    def write_block(self, block: np.ndarray) -> int:
        """Accept one block of captured float samples. Returns frames kept.

        Accepts mono `(n,)` or interleaved `(n, channels)`. Blocks arriving
        while paused or stopped are discarded, and a block is trimmed when it
        would cross `max_duration_seconds`.
        """
        data = np.asarray(block, dtype=np.float32)
        if data.size == 0:
            return 0
        frames = int(data.shape[0])

        with self._lock:
            if self._state is not RecorderState.RECORDING:
                return 0
            limit = int(self.recording.max_duration_seconds * self._sample_rate)
            room = max(0, limit - self._acc.frames)
            if room == 0:
                self._acc.truncated = True
                return 0
            if frames > room:
                frames = room
                data = data[:room]
                self._acc.truncated = True

            flat = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
            peak = float(np.max(np.abs(flat))) if flat.size else 0.0
            self._acc.frames += frames
            self._acc.peak = max(self._acc.peak, peak)
            self._acc.window_peak = max(self._acc.window_peak, peak)
            # Divided by the channel count so RMS is per sample position, and
            # a stereo capture is not reported as louder than a mono one.
            self._acc.sum_squares += float(
                np.sum(np.square(flat, dtype=np.float64))
            ) / max(1, self._channels)
            self._acc.clipped += int(
                np.count_nonzero(np.abs(flat) >= self.recording.clip_threshold)
            )
            # Copied because the backend reuses its callback buffer.
            self._queue.put(flat.copy())
        return frames

    # -------------------------------------------------------------- controls
    def pause(self) -> None:
        """Stop accepting frames without closing the file."""
        with self._lock:
            if self._state is RecorderState.RECORDING:
                self._state = RecorderState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state is RecorderState.PAUSED:
                self._state = RecorderState.RECORDING

    def toggle_pause(self) -> None:
        if self._state is RecorderState.RECORDING:
            self.pause()
        elif self._state is RecorderState.PAUSED:
            self.resume()

    def read_level(self) -> float:
        """Peak amplitude since the previous call, in [0, 1].

        Read-and-reset, which is what a level meter wants: each UI tick shows
        the loudest sample in the interval it is drawing.
        """
        with self._lock:
            peak = self._acc.window_peak
            self._acc.window_peak = 0.0
            return peak

    def read_level_dbfs(self) -> float:
        return _to_dbfs(self.read_level())

    def stop(self) -> RecordingResult | None:
        """Finish the capture and close the file.

        Returns the result, or None when nothing was captured -- in which case
        the empty file is removed rather than left for the annotator to open.
        """
        with self._lock:
            if self._state is RecorderState.IDLE:
                return None
            self._state = RecorderState.IDLE
            target = self._path
            acc = self._acc
            sample_rate = self._sample_rate
            channels = self._channels

        self._close_stream()
        self._finish_writer()

        if self._writer_error is not None:
            error, self._writer_error = self._writer_error, None
            raise RecordingError(f"recording was not written correctly: {error}")

        if target is None:
            return None
        if acc.frames == 0:
            _remove_quietly(target)
            log.info("capture produced no frames; %s removed", target)
            return None

        result = RecordingResult(
            path=target,
            sample_rate=sample_rate,
            channels=channels,
            frames=acc.frames,
            peak=acc.peak,
            rms=acc.rms(),
            clipped_frames=acc.clipped,
            truncated=acc.truncated,
            overflows=acc.overflows,
        )
        log.info(
            "captured %.2fs to %s (peak %.1f dBFS)",
            result.duration,
            target,
            result.peak_dbfs,
        )
        return result

    def cancel(self) -> None:
        """Abandon the capture and delete the partial file."""
        with self._lock:
            if self._state is RecorderState.IDLE:
                return
            self._state = RecorderState.IDLE
            target = self._path
        self._close_stream()
        self._finish_writer()
        self._writer_error = None
        if target is not None:
            _remove_quietly(target)
            log.info("capture cancelled; %s removed", target)

    # -------------------------------------------------------------- teardown
    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            # A device unplugged mid-capture must not lose what was recorded.
            pass

    def _finish_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            self._queue.put(None)
            writer.join(timeout=10.0)
        handle, self._wave = self._wave, None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - disk failure
                log.error("could not close the recording file: %s", exc)

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.is_active:
            self.stop()


# ---------------------------------------------------------------- helpers
def _float_to_pcm(data: np.ndarray, bits: int) -> bytes:
    """Convert float samples in [-1, 1] to little-endian PCM bytes."""
    clipped = np.clip(data, -1.0, 1.0)
    if bits == 16:
        return (clipped * 32767.0).astype("<i2").tobytes()
    if bits == 32:
        return (clipped.astype(np.float64) * 2147483647.0).astype("<i4").tobytes()
    if bits == 24:
        scaled = (clipped.astype(np.float64) * 8388607.0).astype("<i4")
        # Little-endian int32 holds the 24-bit value in its low three bytes.
        return np.ascontiguousarray(scaled).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    raise RecordingError(f"unsupported capture width {bits} bit; use 16, 24 or 32")


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - permissions
        log.warning("could not remove %s: %s", path, exc)


_SOUNDDEVICE_CHECKED = False
_SOUNDDEVICE = None


def rescan_devices() -> bool:
    """Re-enumerate the machine's audio devices. True when the rescan ran.

    PortAudio builds its device table at initialisation and never refreshes it,
    so a device connected afterwards -- the ordinary case for a Bluetooth
    headset, which is switched on once the application is already open -- is
    missing from every later query. Terminating and re-initialising the library
    is the only way to see it without restarting the process.

    Safe to call while nothing is capturing. It is not safe during a capture:
    terminating PortAudio invalidates open streams, so `Recorder` only calls
    this from the idle path.
    """
    sd = _sounddevice()
    if sd is None:
        return False
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:  # pragma: no cover - backend specific
        log.warning("could not re-scan audio devices: %s", exc)
        return False
    return True


#: Blank line separator used when a message carries the help text below it.
_GAP = chr(10) * 2


#: Told to the annotator when a Bluetooth headset is expected but no capture
#: endpoint for it exists. Windows only exposes a headset microphone once the
#: Hands-Free Telephony service is enabled for that device; with only the
#: stereo A2DP profile connected there is a "Headphones (...)" output and no
#: microphone at all, and no application can enumerate what Windows does not
#: expose.
BLUETOOTH_MIC_HELP = """A Bluetooth headset only offers a microphone through the Hands-Free
profile. If the headset plays audio but never appears as a mic:
  1. Settings > Bluetooth & devices > Devices > More devices and printer
     settings, right-click the headset > Properties > Services, tick
     "Handsfree Telephony", then Apply.
  2. Reconnect the headset, then check Settings > Sound > Input lists a
     "Headset (<name> Hands-Free AG Audio)" entry.
  3. Settings > Privacy & security > Microphone must allow desktop apps.
  4. Re-scan devices here -- no restart is needed.
Hands-Free capture is 8-16 kHz mono and the headset leaves stereo while the
mic is open. That is the Bluetooth profile, not this application."""


def bluetooth_mic_note(devices: list[InputDevice]) -> str | None:
    """Advice to show when no Bluetooth capture endpoint is present.

    Returns None when one is already listed -- there is nothing to explain.
    """
    if any(device.is_bluetooth for device in devices):
        return None
    return BLUETOOTH_MIC_HELP


def _open_failure_message(info: dict | None, error: Exception | None) -> str:
    """Why the stream would not open, worded for the person at the keyboard."""
    name = str((info or {}).get("name") or "").strip()
    base = f"could not open input device {name!r}" if name else "could not open an input device"
    detail = f" ({error})" if error is not None else ""
    if name and _BLUETOOTH_HINT.search(name):
        return (
            f"{base}{detail}. The headset is paired but its microphone is not "
            "available right now -- reconnect it, then re-scan devices."
            + _GAP
            + BLUETOOTH_MIC_HELP
        )
    if name and _is_unnamed(name):
        return (
            f"{base}{detail}. This entry is a placeholder for a device that is "
            "not connected; connect it and re-scan devices."
        )
    return (
        f"{base}{detail}. Check that a microphone is connected and that this "
        "application is allowed to use it."
    )


def _host_api_name(host_apis, index) -> str:
    """Name of host API `index`, or "" when it cannot be resolved."""
    try:
        return str(host_apis[int(index)]["name"])
    except (TypeError, ValueError, IndexError, KeyError):
        return ""


def _is_unnamed(name: str) -> bool:
    """True for a placeholder endpoint name carrying no device identity.

    Windows reports a paired-but-disconnected Bluetooth endpoint through
    WDM-KS as "Input ()" -- the parenthesised device name is empty because the
    device is not there. Such an entry is kept in the list (it is often exactly
    what the annotator is hunting for) but is labelled, because selecting it
    fails until the headset actually connects.
    """
    stripped = name.strip()
    if not stripped:
        return True
    return bool(re.fullmatch(r"[^()]*\(\s*\)", stripped))


def _sounddevice():
    """Import sounddevice once, returning None when unavailable."""
    global _SOUNDDEVICE_CHECKED, _SOUNDDEVICE
    if not _SOUNDDEVICE_CHECKED:
        _SOUNDDEVICE_CHECKED = True
        try:
            import sounddevice

            _SOUNDDEVICE = sounddevice
        except (ImportError, OSError):
            # OSError covers a present package with no PortAudio library.
            _SOUNDDEVICE = None
    return _SOUNDDEVICE


def record_to_file(
    seconds: float,
    path: str | Path | None = None,
    config: AudioConfig | None = None,
    recording: RecordingConfig | None = None,
    device: int | None = None,
    on_tick=None,
) -> RecordingResult | None:
    """Capture `seconds` of audio to a local WAV file (0 means until Ctrl+C).

    Blocking convenience wrapper for the CLI. `on_tick(elapsed, peak)` is
    called about ten times a second so a caller can draw a level meter.
    """
    import time

    recorder = Recorder(config, recording)
    recorder.start(path, device=device)
    deadline = time.monotonic() + seconds if seconds > 0 else math.inf
    try:
        while time.monotonic() < deadline and not recorder.limit_reached:
            time.sleep(0.1)
            if on_tick is not None:
                on_tick(recorder.elapsed, recorder.read_level())
    except KeyboardInterrupt:
        pass
    return recorder.stop()
