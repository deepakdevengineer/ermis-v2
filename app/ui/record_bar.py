"""Microphone capture bar (CLAUDE.md sections 45, 48, 53).

A transport for the input side of the application: pick a device, record,
pause, stop, or discard. It gives the annotator a way to produce a clip when
there is no attachment to open -- a live conversation, a re-recorded prompt, a
quick capture to check a microphone.

The bar owns a `Recorder` and polls it on a timer. It never touches the
transcription: on stop it emits the path of the finished WAV, and the main
window loads that file through the ordinary loader, so a recorded clip enters
the same analysis and validation path as any other file (section 3).

The level meter and the elapsed clock exist because capture is the one place
where a mistake cannot be undone later. A muted input or a clipped level is
worth seeing while recording, not after the annotator has typed a transcription
of nothing.

Requires PySide6.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.record_bar requires PySide6. Install it with `pip install PySide6`."
    ) from exc

from app.audio.recorder import (
    Recorder,
    RecordingError,
    RecordingResult,
    bluetooth_mic_note,
)
from app.config import AppConfig

#: Blank line separator inside a multi-paragraph tooltip.
_GAP = chr(10) * 2

#: Meter floor. Speech at a sane input level sits well above -60 dBFS, so
#: anchoring the bar here makes a quiet capture visibly quiet rather than just
#: slightly shorter.
_METER_FLOOR_DBFS = -60.0

#: Above this the capture is close to full scale and about to clip.
_METER_HOT_DBFS = -3.0


class RecordBar(QWidget):
    """Record / pause / stop transport for the local input device."""

    #: Emitted with the path of a finished capture, once it is on disk.
    recordingFinished = Signal(str)
    #: Emitted when capture starts, so the owner can stop playback first.
    recordingStarted = Signal()
    #: Status text for the window's status bar.
    statusMessage = Signal(str)

    def __init__(self, config: AppConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config or AppConfig()
        self.recorder = Recorder(
            self.config.audio,
            self.config.recording,
            output_dir=self.config.paths.recordings_dir,
        )
        self._last_result: RecordingResult | None = None
        self._devices: list = []

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(max(20, int(self.config.recording.meter_interval_ms)))
        self._timer.timeout.connect(self._tick)

        self.refresh_devices(rescan=False)
        self._update_controls()

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.record_button = QPushButton("● Record")
        self.record_button.setToolTip(
            "Capture from the microphone to a local WAV file, then analyse it "
            "like any other clip"
        )
        self.record_button.clicked.connect(self.toggle_recording)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setToolTip("Stop capturing without closing the file")
        self.pause_button.clicked.connect(self.toggle_pause)

        self.discard_button = QPushButton("Discard")
        self.discard_button.setToolTip("Abandon this capture and delete the file")
        self.discard_button.clicked.connect(self.discard_recording)

        self.device_box = QComboBox()
        self.device_box.setMinimumWidth(220)
        self.device_box.setToolTip("Input device used for capture")

        self.refresh_button = QPushButton("↻")
        self.refresh_button.setFixedWidth(28)
        self.refresh_button.setToolTip(
            "Re-scan input devices. Use this after switching on a Bluetooth "
            "headset -- the audio backend does not notice new devices by itself."
        )
        self.refresh_button.clicked.connect(lambda: self.refresh_devices(rescan=True))

        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(True)
        self.meter.setFormat("input level")
        self.meter.setFixedWidth(180)

        self.elapsed_label = QLabel("00:00.0")

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        row.addWidget(QLabel("Mic:"))
        row.addWidget(self.device_box)
        row.addWidget(self.refresh_button)
        row.addWidget(separator)
        row.addWidget(self.record_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.discard_button)
        row.addWidget(self.meter)
        row.addWidget(self.elapsed_label)
        row.addStretch()

    def refresh_devices(self, *, rescan: bool = True) -> None:
        """Re-scan input devices, keeping the current selection when possible.

        `rescan` re-initialises PortAudio, which is what makes a headset
        switched on after the window opened appear at all. It is skipped while
        a capture is running, because tearing the library down would invalidate
        the open stream.
        """
        if self.recorder.is_active:
            rescan = False
        previous = self.device_box.currentData()
        self.device_box.clear()
        self.device_box.addItem("System default", None)
        devices = self.recorder.list_devices(rescan=rescan)
        for device in devices:
            # The host API is part of the identity on Windows, where one
            # microphone is listed once per backend and a Bluetooth headset
            # does not behave the same through each of them.
            host = f", {device.host_api}" if device.host_api else ""
            label = f"{device.name} ({device.channels} ch{host})"
            if device.unnamed:
                label += "  -- not connected"
            self.device_box.addItem(label, device.index)
            row = self.device_box.count() - 1
            if device.unnamed:
                self.device_box.setItemData(
                    row,
                    "The backend reports no name for this endpoint, which "
                    "usually means the device is paired but not connected.",
                    Qt.ItemDataRole.ToolTipRole,
                )
        self._devices = devices
        if previous is not None:
            index = self.device_box.findData(previous)
            if index >= 0:
                self.device_box.setCurrentIndex(index)
        self._update_device_hint(devices)

        if not self.recorder.available:
            self.device_box.setEnabled(False)
            self.record_button.setEnabled(False)
            self.record_button.setToolTip(
                "Microphone capture needs the sounddevice package "
                "(pip install sounddevice) and a working audio input device."
            )
            self.meter.setFormat("no input backend")

    def _update_device_hint(self, devices) -> None:
        """Explain a missing Bluetooth microphone on the device picker itself.

        A headset that plays audio but offers no capture endpoint is a Windows
        profile setting, not a fault here, and the annotator has no way to know
        that from an empty list.
        """
        note = bluetooth_mic_note(devices)
        if note is None:
            self.device_box.setToolTip("Input device used for capture")
            return
        self.device_box.setToolTip(
            "Input device used for capture."
            + _GAP
            + "No Bluetooth microphone is listed."
            + _GAP
            + note
        )

    # ------------------------------------------------------------- recording
    def toggle_recording(self) -> None:
        if self.recorder.is_active:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.recorder.is_active:
            return
        device = self.device_box.currentData()
        try:
            path = self.recorder.start(device=device)
        except RecordingError as exc:
            QMessageBox.warning(self, "Cannot record", str(exc))
            self.statusMessage.emit(f"Recording failed: {exc}")
            return

        self.recordingStarted.emit()
        self._timer.start()
        self._update_controls()
        limit = self.config.recording.max_duration_seconds
        self.statusMessage.emit(
            f"Recording to {path.name} at {self.recorder.sample_rate} Hz "
            f"({self.recorder.channels} ch). Stops automatically after "
            f"{limit / 60:.0f} minute(s). Nothing leaves this machine."
        )

    def stop_recording(self) -> None:
        """Finish the capture and hand the file to the owner."""
        if not self.recorder.is_active:
            return
        self._timer.stop()
        try:
            result = self.recorder.stop()
        except RecordingError as exc:
            self._update_controls()
            QMessageBox.critical(self, "Recording problem", str(exc))
            self.statusMessage.emit(f"Recording problem: {exc}")
            return
        self._reset_meter()
        self._update_controls()

        if result is None:
            self.statusMessage.emit("Nothing was captured, so no file was written.")
            return

        self._last_result = result
        message = (
            f"Recorded {result.duration:.2f}s to {result.path.name} "
            f"({result.sample_rate} Hz, {result.channels} ch, "
            f"peak {result.peak_dbfs:.1f} dBFS)"
        )
        notes = result.notes()
        if notes:
            message += "  ·  " + "  ·  ".join(notes)
        self.statusMessage.emit(message)
        self.recordingFinished.emit(str(result.path))

    def discard_recording(self) -> None:
        if not self.recorder.is_active:
            return
        answer = QMessageBox.question(
            self,
            "Discard recording?",
            "Delete this capture without keeping the file? This cannot be undone.",
        )
        if answer is not QMessageBox.StandardButton.Yes:
            return
        self._timer.stop()
        self.recorder.cancel()
        self._reset_meter()
        self._update_controls()
        self.statusMessage.emit("Recording discarded; the file was deleted.")

    def toggle_pause(self) -> None:
        if not self.recorder.is_active:
            return
        self.recorder.toggle_pause()
        self._update_controls()
        self.statusMessage.emit(
            "Recording paused." if not self.recorder.is_recording else "Recording resumed."
        )

    @property
    def last_result(self) -> RecordingResult | None:
        """The most recent finished capture, for the owner's audit record."""
        return self._last_result

    # ------------------------------------------------------------------ timer
    def _tick(self) -> None:
        """Update the clock and the meter, and honour the duration ceiling."""
        self.elapsed_label.setText(_clock(self.recorder.elapsed))

        if self.recorder.is_recording:
            dbfs = self.recorder.read_level_dbfs()
            self.meter.setValue(_meter_value(dbfs))
            if dbfs == float("-inf"):
                self.meter.setFormat("silent")
            elif dbfs >= _METER_HOT_DBFS:
                self.meter.setFormat(f"{dbfs:.0f} dBFS — hot")
            else:
                self.meter.setFormat(f"{dbfs:.0f} dBFS")

        if self.recorder.limit_reached:
            # The recorder stops taking frames at the ceiling but leaves the
            # device open, because closing it from the audio callback can
            # deadlock. Closing it here is that deferred stop.
            self.stop_recording()
            self.statusMessage.emit(
                "Recording stopped at the maximum duration; the file was kept."
            )

    def _reset_meter(self) -> None:
        self.meter.setValue(0)
        self.meter.setFormat("input level")
        self.elapsed_label.setText("00:00.0")

    def _update_controls(self) -> None:
        active = self.recorder.is_active
        if self.recorder.available:
            self.record_button.setEnabled(True)
        self.record_button.setText("■ Stop" if active else "● Record")
        self.pause_button.setEnabled(active)
        self.pause_button.setText(
            "Resume" if active and not self.recorder.is_recording else "Pause"
        )
        self.discard_button.setEnabled(active)
        self.device_box.setEnabled(not active and self.recorder.available)
        self.refresh_button.setEnabled(not active)

    # --------------------------------------------------------------- teardown
    def shutdown(self) -> Path | None:
        """Stop any capture in progress. Returns the kept file, if any.

        Called when the window closes: an in-progress recording is kept rather
        than discarded, because the audio is the one thing that cannot be
        recreated.
        """
        if not self.recorder.is_active:
            return None
        self._timer.stop()
        try:
            result = self.recorder.stop()
        except RecordingError:
            return None
        return result.path if result else None


# ---------------------------------------------------------------- helpers
def _meter_value(dbfs: float) -> int:
    """Map dBFS onto the 0-100 meter, floored at _METER_FLOOR_DBFS."""
    if dbfs == float("-inf") or dbfs <= _METER_FLOOR_DBFS:
        return 0
    return int(round(100.0 * (dbfs - _METER_FLOOR_DBFS) / -_METER_FLOOR_DBFS))


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"
