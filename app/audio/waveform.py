"""Waveform envelope extraction (CLAUDE.md sections 2.2, 45, 48).

Pure numpy. Contains no GUI code so it can be tested headless; the PySide6
widget in app/ui/waveform_view.py renders what this module produces.

The display is a peak envelope: each horizontal pixel column shows the minimum
and maximum sample in its time bucket, which is what makes a brief click or
pop visible at full-file zoom. An RMS envelope is computed alongside it for the
quieter inner band, and because RMS is what a level threshold should be
compared against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.audio.loader import AudioBuffer


@dataclass(frozen=True)
class Waveform:
    """A bucketed envelope ready to draw."""

    #: Most negative sample in each bucket, shape (n_buckets,).
    minima: np.ndarray
    #: Most positive sample in each bucket, shape (n_buckets,).
    maxima: np.ndarray
    #: RMS level in each bucket, shape (n_buckets,).
    rms: np.ndarray
    #: Seconds of audio per bucket.
    seconds_per_bucket: float
    #: Time span this envelope covers, in seconds from the file start.
    start: float
    end: float
    sample_rate: int

    @property
    def bucket_count(self) -> int:
        return int(self.minima.size)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def bucket_time(self, index: int) -> float:
        """Start time of a bucket, for mapping a click to a timestamp."""
        return self.start + index * self.seconds_per_bucket

    def time_to_bucket(self, seconds: float) -> int:
        """Bucket index containing `seconds`, clamped to the envelope."""
        if self.seconds_per_bucket <= 0:
            return 0
        idx = int((seconds - self.start) / self.seconds_per_bucket)
        return max(0, min(self.bucket_count - 1, idx))

    def peak_dbfs(self) -> np.ndarray:
        """Per-bucket peak in dBFS, with silence floored rather than -inf."""
        peak = np.maximum(np.abs(self.minima), np.abs(self.maxima))
        return _safe_dbfs(peak)

    def rms_dbfs(self) -> np.ndarray:
        """Per-bucket RMS in dBFS. This is what a level threshold compares to."""
        return _safe_dbfs(self.rms)


#: dBFS value used in place of -inf for a fully silent bucket, so arrays stay
#: finite and plottable.
DBFS_FLOOR = -120.0


def _safe_dbfs(amplitude: np.ndarray) -> np.ndarray:
    out = np.full(amplitude.shape, DBFS_FLOOR, dtype=np.float64)
    nonzero = amplitude > 0
    out[nonzero] = 20.0 * np.log10(amplitude[nonzero])
    return np.maximum(out, DBFS_FLOOR)


def compute_waveform(
    buffer: AudioBuffer,
    buckets: int | None = None,
    buckets_per_second: int | None = None,
    start: float = 0.0,
    end: float | None = None,
) -> Waveform:
    """Compute a peak/RMS envelope for a time range.

    Args:
        buffer: decoded audio.
        buckets: exact bucket count, normally the widget's pixel width. Takes
            precedence over `buckets_per_second`.
        buckets_per_second: resolution, when a fixed count is not wanted.
            Defaults to 200.
        start: range start in seconds.
        end: range end in seconds; defaults to the end of the buffer.

    Zooming is just a narrower `start`/`end` with the same bucket count, so the
    widget re-renders by calling this again rather than caching pyramids.
    """
    sr = buffer.sample_rate
    total = buffer.duration
    end = total if end is None else min(end, total)
    start = max(0.0, min(start, end))

    if buckets is None:
        per_second = buckets_per_second or 200
        buckets = max(1, int(math.ceil((end - start) * per_second)))
    buckets = max(1, int(buckets))

    a = int(round(start * sr))
    b = int(round(end * sr))
    segment = buffer.samples[a:b]

    minima, maxima, rms = _bucket_envelope(segment, buckets)
    span = (end - start) if end > start else 0.0
    return Waveform(
        minima=minima,
        maxima=maxima,
        rms=rms,
        seconds_per_bucket=(span / buckets) if buckets else 0.0,
        start=start,
        end=end,
        sample_rate=sr,
    )


def _bucket_envelope(
    segment: np.ndarray, buckets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce `segment` to per-bucket min, max and RMS.

    Buckets are split with `array_split`, which spreads the remainder rather
    than dropping the tail -- a truncated tail would hide a click at the very
    end of a file.
    """
    if segment.size == 0:
        zeros = np.zeros(buckets, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()

    if segment.size < buckets:
        # More columns than samples: pad so every bucket has a defined value
        # rather than producing a short array the widget has to special-case.
        padded = np.zeros(buckets, dtype=np.float32)
        padded[: segment.size] = segment
        return padded.copy(), padded.copy(), np.abs(padded)

    # Even split is the common case and is far faster than array_split.
    if segment.size % buckets == 0:
        grid = segment.reshape(buckets, -1)
        minima = grid.min(axis=1)
        maxima = grid.max(axis=1)
        rms = np.sqrt(np.mean(np.square(grid, dtype=np.float64), axis=1))
        return (
            minima.astype(np.float32),
            maxima.astype(np.float32),
            rms.astype(np.float32),
        )

    chunks = np.array_split(segment, buckets)
    minima = np.fromiter((c.min() for c in chunks), dtype=np.float32, count=buckets)
    maxima = np.fromiter((c.max() for c in chunks), dtype=np.float32, count=buckets)
    rms = np.fromiter(
        (math.sqrt(float(np.mean(np.square(c, dtype=np.float64)))) for c in chunks),
        dtype=np.float32,
        count=buckets,
    )
    return minima, maxima, rms


def render_ascii(waveform: Waveform, width: int = 72, height: int = 9) -> str:
    """Render an envelope as ASCII art.

    Used by the CLI and by tests, and it makes the waveform inspectable without
    a GUI toolkit installed.
    """
    if height % 2 == 0:
        height += 1  # keep a centre row for the zero line
    mid = height // 2

    step = max(1, waveform.bucket_count // width)
    idx = np.arange(0, waveform.bucket_count, step)[:width]
    lo = waveform.minima[idx]
    hi = waveform.maxima[idx]

    scale = float(max(np.max(np.abs(lo)) if lo.size else 0.0,
                      np.max(np.abs(hi)) if hi.size else 0.0))
    if scale <= 0:
        scale = 1.0

    grid = [[" "] * len(idx) for _ in range(height)]
    for col, (low, high) in enumerate(zip(lo, hi)):
        top = mid - int(round(high / scale * mid))
        bottom = mid - int(round(low / scale * mid))
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height - 1, bottom))
        for row in range(min(top, bottom), max(top, bottom) + 1):
            grid[row][col] = "#"
        grid[mid][col] = "#" if grid[mid][col] == "#" else "-"

    body = "\n".join("".join(row) for row in grid)
    return f"{body}\n{_ruler(waveform, len(idx))}"


def _ruler(waveform: Waveform, width: int) -> str:
    left = _timestamp(waveform.start)
    right = _timestamp(waveform.end)
    gap = max(1, width - len(left) - len(right))
    return f"{left}{' ' * gap}{right}"


def _timestamp(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"
