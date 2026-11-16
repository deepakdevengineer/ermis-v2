"""Local audio playback (CLAUDE.md sections 45, 47, 48).

Backs the editor's play/pause/seek, playback-speed control and
"replay selected segment" actions. Playback is what lets the annotator answer
"what is actually audible?" -- section 59 -- so segment replay and slow-down
are first-class operations, not extras.

`sounddevice` is imported lazily, so importing this module never requires an
audio backend. Without one, `PlaybackEngine.available` is False and the
controller degrades to a silent transport that still tracks position, which
keeps the UI and the test suite working on a machine with no audio device.

Nothing here writes to the audio file, and nothing leaves the machine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.audio.loader import AudioBuffer
from app.config import AudioConfig

#: Frames per callback block. ~20 ms at 16 kHz: small enough that pause and
#: seek feel immediate, large enough to avoid underruns.
_BLOCK_FRAMES = 320


class PlaybackState(str, Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass(frozen=True)
class PlaybackRange:
    """The region being played. `None` end means "to the end of the file"."""

    start: float
    end: float | None = None
    #: When True, playback restarts at `start` on reaching `end`. Used by the
    #: review workflow, where an annotator listens to one candidate event
    #: repeatedly before deciding (section 47).
    loop: bool = False


class PlaybackEngine:
    """Streams an AudioBuffer to the default output device.

    Position is reported in *source* seconds, so a playback speed of 0.5 still
    maps onto the same waveform timestamps and the playhead stays aligned with
    the timeline.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        self._buffer: AudioBuffer | None = None
        self._state = PlaybackState.STOPPED
        self._speed = 1.0
        self._range = PlaybackRange(0.0, None)
        #: Playhead in source frames.
        self._frame = 0
        self._stream = None
        self._lock = threading.RLock()
        self._on_position: Callable[[float], None] | None = None
        self._on_finished: Callable[[], None] | None = None

    # ------------------------------------------------------------ properties
    @property
    def available(self) -> bool:
        """True when an audio output backend is importable."""
        return _sounddevice() is not None

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state is PlaybackState.PLAYING

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def position(self) -> float:
        """Playhead in source seconds."""
        with self._lock:
            if self._buffer is None or not self._buffer.sample_rate:
                return 0.0
            return self._frame / self._buffer.sample_rate

    @property
    def duration(self) -> float:
        return self._buffer.duration if self._buffer else 0.0

    # ----------------------------------------------------------------- setup
    def set_audio(self, buffer: AudioBuffer | None) -> None:
        self.stop()
        with self._lock:
            self._buffer = buffer
            self._frame = 0
            self._range = PlaybackRange(0.0, None)

    def set_position_callback(self, callback: Callable[[float], None] | None) -> None:
        """Called with the source-seconds position as playback advances."""
        self._on_position = callback

    def set_finished_callback(self, callback: Callable[[], None] | None) -> None:
        self._on_finished = callback

    def set_speed(self, speed: float) -> None:
        """Set the playback rate.

        Resampling is done by linear interpolation, which shifts pitch. That is
        acceptable and often useful here: the annotator is listening to decide
        what was said, and section 14 explicitly suggests slowing the audio
        down to resolve unclear speech.
        """
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        allowed = self.config.playback_speeds
        if allowed and speed not in allowed:
            # Not fatal -- the UI offers presets but a caller may ask for any
            # rate. Clamp to a sane band so a typo cannot produce noise.
            speed = max(0.25, min(4.0, speed))
        self._speed = float(speed)

    # -------------------------------------------------------------- controls
    def play(
        self,
        start: float | None = None,
        end: float | None = None,
        loop: bool = False,
    ) -> None:
        """Start or resume playback.

        With no arguments, resumes from the current position to the end.
        """
        if self._buffer is None:
            return
        with self._lock:
            if start is not None or end is not None:
                self._range = PlaybackRange(start or 0.0, end, loop)
                self._frame = self._frames(self._range.start)
            elif self._state is PlaybackState.STOPPED:
                self._range = PlaybackRange(self.position, None, loop)
            self._state = PlaybackState.PLAYING
        self._ensure_stream()

    def play_segment(self, start: float, end: float, loop: bool = False) -> None:
        """Replay one segment -- the editor's "replay selected segment"."""
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        self.play(start=start, end=end, loop=loop)

    def replay_event(self, event, loop: bool = False, pad: float = 0.25) -> None:
        """Replay an analysis event with a little context on each side.

        The padding matters for the confirmation workflow: a 0.3 s candidate
        played in isolation is very hard to judge, so the annotator hears what
        surrounds it (section 47).
        """
        self.play_segment(
            max(0.0, event.start - pad), min(self.duration, event.end + pad), loop=loop
        )

    def pause(self) -> None:
        with self._lock:
            if self._state is PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED

    def toggle(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        with self._lock:
            self._state = PlaybackState.STOPPED
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                # A device removed mid-playback must not propagate; the
                # transport is already stopped from the caller's perspective.
                pass

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._frame = self._frames(seconds)

    # ------------------------------------------------------------- streaming
    def _frames(self, seconds: float) -> int:
        if self._buffer is None:
            return 0
        total = len(self._buffer.samples)
        return max(0, min(total, int(round(seconds * self._buffer.sample_rate))))

    def _range_end_frame(self) -> int:
        assert self._buffer is not None
        if self._range.end is None:
            return len(self._buffer.samples)
        return self._frames(self._range.end)

    def _ensure_stream(self) -> None:
        sd = _sounddevice()
        if sd is None or self._stream is not None:
            return
        assert self._buffer is not None
        try:
            self._stream = sd.OutputStream(
                samplerate=self._buffer.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=_BLOCK_FRAMES,
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            # No usable output device. Playback silently no-ops; the caller can
            # check `available` to tell the user.
            self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:  # pragma: no cover
        """Fill one output block. Runs on the audio thread."""
        block = self.render_block(frames)
        outdata[:, 0] = block
        if self._on_position is not None:
            self._on_position(self.position)

    # ------------------------------------------------------------ rendering
    def render_block(self, frames: int) -> np.ndarray:
        """Produce the next `frames` output samples and advance the playhead.

        Separated from the audio callback so playback can be exercised without
        a sound device: the test suite pulls blocks directly and asserts the
        transport behaviour.
        """
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            if self._buffer is None or self._state is not PlaybackState.PLAYING:
                return out

            samples = self._buffer.samples
            end_frame = self._range_end_frame()
            needed = frames * self._speed
            start = self._frame
            stop = min(end_frame, int(round(start + needed)))

            if stop <= start:
                self._finish()
                return out

            segment = samples[start:stop]
            if self._speed == 1.0:
                take = min(frames, segment.size)
                out[:take] = segment[:take]
            elif segment.size:
                # Linear interpolation onto the output grid.
                src = np.arange(segment.size, dtype=np.float64)
                dst = np.linspace(0.0, segment.size - 1, num=frames, dtype=np.float64)
                out[:] = np.interp(dst, src, segment).astype(np.float32)

            self._frame = stop
            if self._frame >= end_frame:
                self._finish()
        return out

    def _finish(self) -> None:
        """Handle reaching the end of the playback range. Caller holds the lock."""
        if self._range.loop:
            self._frame = self._frames(self._range.start)
            return
        self._state = PlaybackState.STOPPED
        if self._on_finished is not None:
            self._on_finished()

    # ---------------------------------------------------------------- helper
    def wait(self, timeout: float = 30.0) -> bool:
        """Block until playback stops. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while self.is_playing:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def __enter__(self) -> PlaybackEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


_SOUNDDEVICE_CHECKED = False
_SOUNDDEVICE = None


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
