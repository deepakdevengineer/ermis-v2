"""Shared framing and spectral primitives for the analysis engine.

Every detector in app/detection/ works from the same frame grid, so their
events line up on the timeline and can be compared frame by frame. Keeping the
framing in one place is what makes the analysis reproducible.

Pure numpy/scipy. Deterministic: no randomness, no model weights, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Default analysis frame length and hop, in seconds. 25 ms / 10 ms is the
#: standard speech-analysis grid and gives 100 frames per second.
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010

#: dBFS value standing in for -inf so arrays stay finite.
DBFS_FLOOR = -120.0


@dataclass(frozen=True)
class FrameGrid:
    """Framed signal plus per-frame features shared by all detectors."""

    #: Frame start times in seconds, shape (n_frames,).
    times: np.ndarray
    #: Per-frame RMS amplitude, shape (n_frames,).
    rms: np.ndarray
    #: Per-frame peak absolute amplitude.
    peak: np.ndarray
    #: Per-frame zero-crossing rate in [0, 1].
    zcr: np.ndarray
    #: Magnitude spectrogram, shape (n_frames, n_bins).
    spectrum: np.ndarray
    #: Frequency of each spectrum bin in Hz, shape (n_bins,).
    frequencies: np.ndarray
    sample_rate: int
    frame_length: int
    hop_length: int

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def hop_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def rms_dbfs(self) -> np.ndarray:
        return to_dbfs(self.rms)

    def time_to_frame(self, seconds: float) -> int:
        if self.frame_count == 0:
            return 0
        idx = int(round(seconds / self.hop_seconds))
        return max(0, min(self.frame_count - 1, idx))

    def frame_to_time(self, index: int) -> float:
        return float(index * self.hop_seconds)


def to_dbfs(amplitude: np.ndarray) -> np.ndarray:
    """Convert amplitude to dBFS, flooring silence instead of returning -inf."""
    out = np.full(np.shape(amplitude), DBFS_FLOOR, dtype=np.float64)
    arr = np.asarray(amplitude, dtype=np.float64)
    nonzero = arr > 0
    out[nonzero] = 20.0 * np.log10(arr[nonzero])
    return np.maximum(out, DBFS_FLOOR)


def frame_signal(
    samples: np.ndarray, frame_length: int, hop_length: int
) -> np.ndarray:
    """Split a signal into overlapping frames, shape (n_frames, frame_length).

    The tail is zero-padded rather than dropped, so an event at the very end of
    a file is still analysed.
    """
    if samples.size == 0 or frame_length <= 0:
        return np.zeros((0, max(1, frame_length)), dtype=np.float32)

    n_frames = max(1, 1 + int(math.ceil(max(0, samples.size - frame_length) / hop_length)))
    needed = (n_frames - 1) * hop_length + frame_length
    if needed > samples.size:
        padded = np.zeros(needed, dtype=np.float32)
        padded[: samples.size] = samples
    else:
        padded = np.ascontiguousarray(samples, dtype=np.float32)

    # as_strided avoids copying the overlapping frames.
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    return np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, frame_length), strides=strides, writeable=False
    )


def compute_grid(
    samples: np.ndarray,
    sample_rate: int,
    frame_seconds: float = FRAME_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> FrameGrid:
    """Compute the shared analysis grid for a signal."""
    frame_length = max(1, int(round(frame_seconds * sample_rate)))
    hop_length = max(1, int(round(hop_seconds * sample_rate)))

    frames = frame_signal(samples, frame_length, hop_length)
    n_frames = frames.shape[0]

    if n_frames == 0 or samples.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return FrameGrid(
            times=empty,
            rms=empty,
            peak=empty,
            zcr=empty,
            spectrum=np.zeros((0, frame_length // 2 + 1)),
            frequencies=np.fft.rfftfreq(frame_length, 1.0 / sample_rate),
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = np.max(np.abs(frames), axis=1).astype(np.float64)

    signs = np.signbit(frames)
    zcr = np.count_nonzero(np.diff(signs, axis=1), axis=1) / max(1, frame_length - 1)

    window = np.hanning(frame_length).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)

    times = np.arange(n_frames, dtype=np.float64) * (hop_length / sample_rate)

    return FrameGrid(
        times=times,
        rms=rms,
        peak=peak,
        zcr=zcr.astype(np.float64),
        spectrum=spectrum,
        frequencies=frequencies,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )


# ---------------------------------------------------------------------------
# Spectral descriptors
# ---------------------------------------------------------------------------
def spectral_centroid(grid: FrameGrid) -> np.ndarray:
    """Per-frame spectral centroid in Hz -- the "brightness" of each frame."""
    total = grid.spectrum.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    centroid = (grid.spectrum * grid.frequencies).sum(axis=1) / safe
    return np.where(total > 0, centroid, 0.0)


def spectral_flatness(grid: FrameGrid) -> np.ndarray:
    """Per-frame spectral flatness in [0, 1].

    Near 1 means noise-like (hiss, static, a click); near 0 means tonal, which
    voiced speech is. Used to separate an artifact from speech.
    """
    magnitude = grid.spectrum + 1e-12
    log_mean = np.exp(np.mean(np.log(magnitude), axis=1))
    arith_mean = np.mean(magnitude, axis=1)
    return np.where(arith_mean > 0, log_mean / arith_mean, 0.0)


def spectral_rolloff(grid: FrameGrid, fraction: float = 0.85) -> np.ndarray:
    """Frequency below which `fraction` of the spectral energy lies."""
    cumulative = np.cumsum(grid.spectrum, axis=1)
    total = cumulative[:, -1:]
    safe = np.where(total > 0, total, 1.0)
    normalized = cumulative / safe
    indices = np.argmax(normalized >= fraction, axis=1)
    return grid.frequencies[indices]


def spectral_flux(grid: FrameGrid) -> np.ndarray:
    """Frame-to-frame spectral change. A sharp transient produces a spike."""
    if grid.frame_count < 2:
        return np.zeros(grid.frame_count, dtype=np.float64)
    total = grid.spectrum.sum(axis=1, keepdims=True)
    normalized = grid.spectrum / np.where(total > 0, total, 1.0)
    diff = np.diff(normalized, axis=0)
    flux = np.sqrt(np.sum(np.square(diff), axis=1))
    return np.concatenate([[0.0], flux])


def band_energy_ratio(grid: FrameGrid, low_hz: float, high_hz: float) -> np.ndarray:
    """Fraction of per-frame energy inside a frequency band."""
    mask = (grid.frequencies >= low_hz) & (grid.frequencies < high_hz)
    band = np.square(grid.spectrum[:, mask]).sum(axis=1)
    total = np.square(grid.spectrum).sum(axis=1)
    # Divide only where the denominator is non-zero. np.where would still
    # evaluate the division everywhere and warn on silent frames.
    return np.divide(band, total, out=np.zeros_like(band), where=total > 0)


# ---------------------------------------------------------------------------
# Run-length helpers -- turning per-frame booleans into time regions
# ---------------------------------------------------------------------------
def boolean_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as inclusive-exclusive frame index pairs."""
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def runs_to_spans(
    runs: list[tuple[int, int]], grid: FrameGrid, min_duration: float = 0.0
) -> list[tuple[float, float]]:
    """Convert frame runs to (start, end) second spans, dropping short ones.

    A run's end time includes the final frame's length, so a span covers the
    audio the frames actually saw rather than only their start times.
    """
    spans: list[tuple[float, float]] = []
    frame_seconds = grid.frame_length / grid.sample_rate
    for start, end in runs:
        t0 = grid.frame_to_time(start)
        t1 = grid.frame_to_time(end - 1) + frame_seconds
        if t1 - t0 >= min_duration:
            spans.append((t0, t1))
    return spans


def smooth_flags(flags: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth a boolean sequence to remove single-frame flicker."""
    if window <= 1 or flags.size == 0:
        return flags.astype(bool)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(flags.astype(np.float64), half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1) >= 0.5


def merge_spans(
    spans: list[tuple[float, float]], gap: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping or near-adjacent spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]
