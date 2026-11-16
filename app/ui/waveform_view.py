"""Waveform / timeline widget (CLAUDE.md sections 2.2, 48, 18A.7).

Requires PySide6. The import is isolated in this module so the rest of the
application -- and the whole test suite -- runs without a GUI toolkit
installed.

What the widget shows:
  - the peak envelope, with an inner RMS band
  - speech and silence/pause regions from the analysis engine
  - event markers for detected events awaiting human confirmation
  - a speaker activity lane per speaker (section 18A.7)
  - the playhead and the current selection

Pause regions are drawn because they matter for navigation and QA. Drawing a
pause never implies a transcription tag: V2 has no pause tag (section 13).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from PySide6.QtCore import QPoint, QRect, Qt, Signal
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import QSizePolicy, QWidget
except ImportError as exc:  # pragma: no cover - exercised only without PySide6
    raise ImportError(
        "app.ui.waveform_view requires PySide6. Install it with "
        "`pip install PySide6`, or use app.audio.waveform.render_ascii for a "
        "headless waveform."
    ) from exc

from app.audio.loader import AudioBuffer
from app.audio.waveform import Waveform, compute_waveform
from app.models.event import AnalysisEvent, EventType


@dataclass(frozen=True)
class Palette:
    """Dark-theme colours for the waveform view."""

    background: QColor = QColor("#14161a")
    envelope: QColor = QColor("#5aa9e6")
    envelope_rms: QColor = QColor("#a5d8ff")
    centre_line: QColor = QColor("#2b3038")
    speech: QColor = QColor(90, 169, 230, 28)
    silence: QColor = QColor(120, 120, 130, 22)
    pause: QColor = QColor(230, 190, 90, 34)
    playhead: QColor = QColor("#ff6b6b")
    selection: QColor = QColor(255, 255, 255, 30)
    ruler_text: QColor = QColor("#7a8290")
    #: Event markers, keyed by event type.
    event_pending: QColor = QColor("#ffd43b")
    event_confirmed: QColor = QColor("#51cf66")
    event_rejected: QColor = QColor("#868e96")
    speaker_lane: QColor = QColor("#4dabf7")


#: Region-shading colour per analysis event type.
_REGION_TYPES = {
    EventType.SPEECH: "speech",
    EventType.SILENCE: "silence",
    EventType.PAUSE: "pause",
}

_RULER_HEIGHT = 16
_LANE_HEIGHT = 10
_MARKER_HEIGHT = 6


class WaveformView(QWidget):
    """Scrollable, zoomable waveform with event and speaker overlays."""

    #: Emitted when the user clicks a position, in seconds.
    positionClicked = Signal(float)
    #: Emitted when the user drags a selection, as (start, end) in seconds.
    selectionChanged = Signal(float, float)
    #: Emitted when the visible window changes, as (start, end) in seconds.
    viewRangeChanged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self.palette_colors = Palette()
        self._buffer: AudioBuffer | None = None
        self._waveform: Waveform | None = None
        self._events: list[AnalysisEvent] = []
        self._speaker_order: list[str] = []

        self._view_start = 0.0
        self._view_end = 0.0
        self._playhead = 0.0
        self._selection: tuple[float, float] | None = None
        self._drag_origin: float | None = None

    # ------------------------------------------------------------------ data
    def set_audio(self, buffer: AudioBuffer | None) -> None:
        self._buffer = buffer
        self._events = []
        self._speaker_order = []
        self._playhead = 0.0
        self._selection = None
        self._view_start = 0.0
        self._view_end = buffer.duration if buffer else 0.0
        self._rebuild()

    def set_events(self, events: list[AnalysisEvent]) -> None:
        """Attach analysis events. Speaker lanes follow first appearance."""
        self._events = sorted(events, key=lambda e: (e.start, e.end))
        order: list[str] = []
        for e in self._events:
            if e.speaker_id and e.speaker_id not in order:
                order.append(e.speaker_id)
        self._speaker_order = order
        self.update()

    @property
    def duration(self) -> float:
        return self._buffer.duration if self._buffer else 0.0

    # ------------------------------------------------------------- view state
    def set_view_range(self, start: float, end: float) -> None:
        total = self.duration
        if total <= 0:
            return
        # Keep at least 10 ms visible so the envelope never collapses.
        span = max(0.01, min(total, end - start))
        start = max(0.0, min(start, total - span))
        self._view_start, self._view_end = start, start + span
        self._rebuild()
        self.viewRangeChanged.emit(self._view_start, self._view_end)

    def zoom(self, factor: float, anchor: float | None = None) -> None:
        """Zoom by `factor` (>1 zooms in) keeping `anchor` seconds fixed."""
        if self.duration <= 0:
            return
        anchor = self._playhead if anchor is None else anchor
        span = (self._view_end - self._view_start) / factor
        span = max(0.01, min(self.duration, span))
        ratio = 0.5
        if self._view_end > self._view_start:
            ratio = (anchor - self._view_start) / (self._view_end - self._view_start)
            ratio = max(0.0, min(1.0, ratio))
        self.set_view_range(anchor - span * ratio, anchor - span * ratio + span)

    def zoom_to_fit(self) -> None:
        self.set_view_range(0.0, self.duration)

    def zoom_to_selection(self) -> None:
        if self._selection:
            self.set_view_range(*self._selection)

    def set_playhead(self, seconds: float) -> None:
        self._playhead = max(0.0, min(seconds, self.duration))
        self.update()

    def set_selection(self, start: float, end: float) -> None:
        lo, hi = (start, end) if start <= end else (end, start)
        self._selection = (max(0.0, lo), min(hi, self.duration))
        self.update()

    def clear_selection(self) -> None:
        self._selection = None
        self.update()

    @property
    def selection(self) -> tuple[float, float] | None:
        return self._selection

    # ------------------------------------------------------------- geometry
    def _plot_rect(self) -> QRect:
        lanes = len(self._speaker_order) * _LANE_HEIGHT
        top = _RULER_HEIGHT
        height = max(20, self.height() - top - lanes - _MARKER_HEIGHT)
        return QRect(0, top, self.width(), height)

    def _x_for_time(self, seconds: float) -> int:
        span = self._view_end - self._view_start
        if span <= 0:
            return 0
        return int((seconds - self._view_start) / span * self.width())

    def _time_for_x(self, x: int) -> float:
        span = self._view_end - self._view_start
        if self.width() <= 0:
            return self._view_start
        return self._view_start + (x / self.width()) * span

    def _rebuild(self) -> None:
        """Recompute the envelope for the visible window at pixel resolution."""
        if self._buffer is None or self.width() <= 0:
            self._waveform = None
        else:
            self._waveform = compute_waveform(
                self._buffer,
                buckets=max(1, self.width()),
                start=self._view_start,
                end=self._view_end,
            )
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rebuild()

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        c = self.palette_colors
        painter.fillRect(self.rect(), c.background)

        if self._buffer is None or self._waveform is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        plot = self._plot_rect()
        self._draw_regions(painter, plot)
        self._draw_envelope(painter, plot)
        self._draw_ruler(painter)
        self._draw_event_markers(painter, plot)
        self._draw_speaker_lanes(painter, plot)
        self._draw_selection(painter, plot)
        self._draw_playhead(painter, plot)
        painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(self.palette_colors.ruler_text))
        painter.drawText(
            self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio loaded"
        )

    def _draw_envelope(self, painter: QPainter, plot: QRect) -> None:
        wf = self._waveform
        assert wf is not None
        c = self.palette_colors
        mid = plot.center().y()
        half = plot.height() / 2.0

        painter.setPen(QPen(c.centre_line, 1))
        painter.drawLine(plot.left(), mid, plot.right(), mid)

        # Peak envelope: one vertical line per pixel column.
        painter.setPen(QPen(c.envelope, 1))
        count = min(wf.bucket_count, plot.width())
        for i in range(count):
            top = mid - int(float(wf.maxima[i]) * half)
            bottom = mid - int(float(wf.minima[i]) * half)
            if top == bottom:
                bottom = top + 1
            painter.drawLine(plot.left() + i, top, plot.left() + i, bottom)

        # Inner RMS band, so overall loudness reads at a glance.
        painter.setPen(QPen(c.envelope_rms, 1))
        for i in range(count):
            r = int(float(wf.rms[i]) * half)
            if r > 0:
                painter.drawLine(plot.left() + i, mid - r, plot.left() + i, mid + r)

    def _draw_regions(self, painter: QPainter, plot: QRect) -> None:
        """Shade speech / silence / pause regions."""
        c = self.palette_colors
        painter.setPen(Qt.PenStyle.NoPen)
        for e in self._events:
            attr = _REGION_TYPES.get(e.type)
            if attr is None or e.end < self._view_start or e.start > self._view_end:
                continue
            x1 = self._x_for_time(max(e.start, self._view_start))
            x2 = self._x_for_time(min(e.end, self._view_end))
            painter.setBrush(QBrush(getattr(c, attr)))
            painter.drawRect(QRect(x1, plot.top(), max(1, x2 - x1), plot.height()))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_event_markers(self, painter: QPainter, plot: QRect) -> None:
        """Draw a marker strip for point-like events needing confirmation."""
        c = self.palette_colors
        y = plot.bottom() + 1
        painter.setPen(Qt.PenStyle.NoPen)
        for e in self._events:
            if e.type in _REGION_TYPES:
                continue
            if e.end < self._view_start or e.start > self._view_end:
                continue
            if e.is_confirmed:
                colour = c.event_confirmed
            elif e.requires_human_confirmation:
                colour = c.event_pending
            else:
                colour = c.event_rejected
            x1 = self._x_for_time(max(e.start, self._view_start))
            x2 = self._x_for_time(min(e.end, self._view_end))
            painter.setBrush(QBrush(colour))
            painter.drawRect(QRect(x1, y, max(2, x2 - x1), _MARKER_HEIGHT - 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_speaker_lanes(self, painter: QPainter, plot: QRect) -> None:
        """One lane per speaker (CLAUDE.md section 18A.7)."""
        if not self._speaker_order:
            return
        c = self.palette_colors
        font = QFont(painter.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2))
        painter.setFont(font)

        base = plot.bottom() + _MARKER_HEIGHT
        for lane, speaker in enumerate(self._speaker_order):
            y = base + lane * _LANE_HEIGHT
            painter.setPen(QPen(c.ruler_text))
            painter.drawText(QPoint(2, y + _LANE_HEIGHT - 2), speaker)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(c.speaker_lane))
            for e in self._events:
                if e.speaker_id != speaker or e.type is not EventType.SPEECH:
                    continue
                if e.end < self._view_start or e.start > self._view_end:
                    continue
                x1 = self._x_for_time(max(e.start, self._view_start))
                x2 = self._x_for_time(min(e.end, self._view_end))
                painter.drawRect(QRect(x1, y + 2, max(1, x2 - x1), _LANE_HEIGHT - 5))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_ruler(self, painter: QPainter) -> None:
        c = self.palette_colors
        painter.setPen(QPen(c.ruler_text))
        font = QFont(painter.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2))
        painter.setFont(font)

        span = self._view_end - self._view_start
        if span <= 0:
            return
        step = _tick_step(span)
        tick = step * int(self._view_start / step)
        while tick <= self._view_end:
            if tick >= self._view_start:
                x = self._x_for_time(tick)
                painter.drawLine(x, _RULER_HEIGHT - 4, x, _RULER_HEIGHT)
                painter.drawText(QPoint(x + 2, _RULER_HEIGHT - 5), _timestamp(tick))
            tick += step

    def _draw_selection(self, painter: QPainter, plot: QRect) -> None:
        if not self._selection:
            return
        start, end = self._selection
        x1 = self._x_for_time(start)
        x2 = self._x_for_time(end)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.palette_colors.selection))
        painter.drawRect(QRect(x1, plot.top(), max(1, x2 - x1), plot.height()))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_playhead(self, painter: QPainter, plot: QRect) -> None:
        if not (self._view_start <= self._playhead <= self._view_end):
            return
        x = self._x_for_time(self._playhead)
        painter.setPen(QPen(self.palette_colors.playhead, 1))
        painter.drawLine(x, 0, x, plot.bottom())

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._buffer is None or event.button() is not Qt.MouseButton.LeftButton:
            return
        t = self._time_for_x(int(event.position().x()))
        self._drag_origin = t
        self.set_playhead(t)
        self.positionClicked.emit(t)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_origin is None:
            return
        t = self._time_for_x(int(event.position().x()))
        self.set_selection(self._drag_origin, t)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_origin is None:
            return
        t = self._time_for_x(int(event.position().x()))
        # A click without movement is a seek, not a zero-length selection.
        if abs(t - self._drag_origin) < 1e-3:
            self.clear_selection()
        else:
            self.set_selection(self._drag_origin, t)
            assert self._selection is not None
            self.selectionChanged.emit(*self._selection)
        self._drag_origin = None

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._buffer is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            anchor = self._time_for_x(int(event.position().x()))
            self.zoom(1.25**steps, anchor=anchor)


def _tick_step(span: float) -> float:
    """Choose a ruler tick interval that yields roughly 4-10 labels."""
    for candidate in (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if span / candidate <= 10:
            return float(candidate)
    return 900.0


def _timestamp(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"
