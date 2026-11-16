"""Speaker panel (CLAUDE.md section 18A.9).

Shows the total speaker count and one editable row per speaker: id, gender,
nativity, confidence. Provides Merge, Split, Add and Remove.

Section 18A.11 is the governing rule -- automatic detection is assistive only,
the confidence must be visible, and the human's decision is final. So:

  - Both dropdowns are populated only with the allowed vocabularies, making an
    invalid value unselectable rather than merely rejected later.
  - Both default to `unknown`, and nothing here ever guesses. Section 18A.3
    and 18A.4 forbid inferring gender or nativity from insufficient evidence.
  - Any edit marks the profile human-edited, which blocks later automatic
    passes from overwriting it.
  - The detected count is shown alongside the editable one, so the annotator
    can see what was suggested and what it now is.

Requires PySide6.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.speaker_panel requires PySide6. Install it with `pip install PySide6`."
    ) from exc

from app.speakers.metadata import SpeakerError, SpeakerRegistry
from app.storage.database import ALLOWED_GENDERS, ALLOWED_NATIVITY, SpeakerProfile

_GENDER_LABELS = {
    "msv": "msv (masculine-sounding voice)",
    "fsv": "fsv (feminine-sounding voice)",
    "unknown": "unknown (cannot reliably determine)",
}

_NATIVITY_LABELS = {
    "native": "native (native English speaker)",
    "nonnative": "nonnative (non-native English speaker)",
    "unknown": "unknown (cannot reliably determine)",
    "tts": "tts (computer-generated / synthetic)",
}


class SpeakerRow(QWidget):
    """One speaker's editable metadata."""

    changed = Signal(str, str, str)  # speaker_id, gender, nativity

    def __init__(self, profile: SpeakerProfile, index: int, parent=None) -> None:
        super().__init__(parent)
        self.speaker_id = profile.speaker_id

        title = QLabel(f"<b>Speaker {index}</b>  <code>{profile.speaker_id}</code>")

        self.gender = QComboBox()
        for value in ALLOWED_GENDERS:
            self.gender.addItem(_GENDER_LABELS[value], value)
        self.gender.setCurrentIndex(list(ALLOWED_GENDERS).index(profile.gender))

        self.nativity = QComboBox()
        for value in ALLOWED_NATIVITY:
            self.nativity.addItem(_NATIVITY_LABELS[value], value)
        self.nativity.setCurrentIndex(list(ALLOWED_NATIVITY).index(profile.nativity))

        confidence = (
            f"{profile.confidence * 100:.0f}%"
            if profile.confidence is not None
            else "n/a"
        )
        seen = ""
        if profile.first_seen is not None and profile.last_seen is not None:
            seen = f"  ·  {profile.first_seen:.2f}s – {profile.last_seen:.2f}s"
        meta = QLabel(
            f"Detection confidence: {confidence}{seen}"
            + ("  ·  <b>human edited</b>" if profile.human_edited else "")
        )
        small = QFont()
        small.setPointSize(8)
        meta.setFont(small)

        self.gender.currentIndexChanged.connect(self._emit)
        self.nativity.currentIndexChanged.connect(self._emit)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gender:"))
        row.addWidget(self.gender)
        row.addWidget(QLabel("Nativity:"))
        row.addWidget(self.nativity)
        row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(title)
        layout.addLayout(row)
        layout.addWidget(meta)

    def _emit(self) -> None:
        self.changed.emit(
            self.speaker_id,
            self.gender.currentData(),
            self.nativity.currentData(),
        )


class SpeakerPanel(QWidget):
    """The full speaker panel of section 18A.9."""

    #: Emitted whenever the registry changes, with a description for the audit
    #: log.
    registryChanged = Signal(str)
    #: Emitted with (speaker_id, before, after) after a metadata edit.
    metadataEdited = Signal(str, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = SpeakerRegistry()
        self._detected_count: int | None = None
        self._detected_confidence: float | None = None

        self.total = QLabel("Total Speakers: 0")
        bold = QFont()
        bold.setBold(True)
        bold.setPointSize(11)
        self.total.setFont(bold)

        self.detected = QLabel("")
        small = QFont()
        small.setPointSize(8)
        self.detected.setFont(small)
        self.detected.setWordWrap(True)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 20)
        self.count_spin.setToolTip(
            "The human-confirmed number of speakers. Automatic detection is a "
            "suggestion; your figure wins (section 18A.11)."
        )
        self.count_spin.valueChanged.connect(self._on_count_changed)
        self._suppress_count_signal = False

        count_row = QHBoxLayout()
        count_row.addWidget(self.total)
        count_row.addStretch()
        count_row.addWidget(QLabel("Set count:"))
        count_row.addWidget(self.count_spin)

        self.speaker_list = QListWidget()
        self.speaker_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )

        self.merge_button = QPushButton("Merge Speakers")
        self.merge_button.setToolTip(
            "Select two or more speakers who are actually the same person"
        )
        self.split_button = QPushButton("Split Speaker")
        self.add_button = QPushButton("Add Speaker")
        self.remove_button = QPushButton("Remove Speaker")

        self.merge_button.clicked.connect(self._merge)
        self.split_button.clicked.connect(self._split)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)

        buttons = QHBoxLayout()
        for button in (
            self.merge_button,
            self.split_button,
            self.add_button,
            self.remove_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()

        box = QGroupBox("SPEAKERS")
        box_layout = QVBoxLayout(box)
        box_layout.addLayout(count_row)
        box_layout.addWidget(self.detected)
        box_layout.addWidget(self.speaker_list)
        box_layout.addLayout(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(box)

        #: Boundary used by Split, set from the waveform selection.
        self.split_boundary: float | None = None

    # ------------------------------------------------------------------ input
    def set_registry(
        self,
        registry: SpeakerRegistry,
        detected_count: int | None = None,
        detected_confidence: float | None = None,
    ) -> None:
        self.registry = registry
        if detected_count is not None:
            self._detected_count = detected_count
            self._detected_confidence = detected_confidence
        self.refresh()

    def set_split_boundary(self, seconds: float | None) -> None:
        """Set where Split will divide, normally the waveform selection start."""
        self.split_boundary = seconds
        self.split_button.setToolTip(
            f"Split the selected speaker at {seconds:.2f}s"
            if seconds is not None
            else "Select a point in the waveform first, then split"
        )

    # ---------------------------------------------------------------- display
    def refresh(self) -> None:
        self.total.setText(f"Total Speakers: {self.registry.count}")

        if self._detected_count is not None:
            confidence = (
                f", confidence {self._detected_confidence * 100:.0f}%"
                if self._detected_confidence is not None
                else ""
            )
            note = (
                f"Detected: speaker count = {self._detected_count}{confidence}. "
                "Automatic detection is assistive only -- correct it if it is wrong "
                "(section 18A.11)."
            )
            if self._detected_count != self.registry.count:
                note += f" You changed it to {self.registry.count}."
            self.detected.setText(note)

        self._suppress_count_signal = True
        self.count_spin.setValue(self.registry.count)
        self._suppress_count_signal = False

        self.speaker_list.clear()
        for index, speaker_id in enumerate(self.registry.speaker_ids, start=1):
            profile = self.registry.profiles[speaker_id]
            row = SpeakerRow(profile, index)
            row.changed.connect(self._on_metadata_changed)
            item = QListWidgetItem(self.speaker_list)
            item.setData(Qt.ItemDataRole.UserRole, speaker_id)
            item.setSizeHint(row.sizeHint())
            self.speaker_list.addItem(item)
            self.speaker_list.setItemWidget(item, row)

        single = self.registry.is_single_speaker()
        self.merge_button.setEnabled(not single)
        self.split_button.setEnabled(self.registry.count >= 1)
        self.remove_button.setEnabled(self.registry.count > 0)

    def _selected_ids(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.speaker_list.selectedItems()
        ]

    # ---------------------------------------------------------------- actions
    def _on_metadata_changed(self, speaker_id: str, gender: str, nativity: str) -> None:
        try:
            before, after = self.registry.set_metadata(
                speaker_id, gender=gender, nativity=nativity
            )
        except SpeakerError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        self.metadataEdited.emit(speaker_id, before, after)
        self.registryChanged.emit(f"metadata edited for {speaker_id}")

    def _on_count_changed(self, value: int) -> None:
        if self._suppress_count_signal or value == self.registry.count:
            return
        try:
            self.registry.set_count(value)
        except SpeakerError as exc:
            QMessageBox.warning(self, "Cannot change count", str(exc))
            return
        self.refresh()
        self.registryChanged.emit(f"speaker count set to {value} by the annotator")

    def _merge(self) -> None:
        selected = self._selected_ids()
        if len(selected) < 2:
            QMessageBox.information(
                self,
                "Merge speakers",
                "Select two or more speakers to merge. They will become the "
                "lowest-numbered one of the selection.",
            )
            return
        target = min(selected, key=lambda sid: int(sid.lstrip("s")))
        sources = [sid for sid in selected if sid != target]
        try:
            self.registry.merge(sources, target)
        except SpeakerError as exc:
            QMessageBox.warning(self, "Cannot merge", str(exc))
            return
        self.refresh()
        self.registryChanged.emit(f"merged {', '.join(sources)} into {target}")

    def _split(self) -> None:
        selected = self._selected_ids()
        if len(selected) != 1:
            QMessageBox.information(
                self, "Split speaker", "Select exactly one speaker to split."
            )
            return
        if self.split_boundary is None:
            QMessageBox.information(
                self,
                "Split speaker",
                "Click a point in the waveform first. Segments starting before it "
                "stay with this speaker; the rest become a new one.",
            )
            return
        try:
            original, new = self.registry.split(selected[0], self.split_boundary)
        except SpeakerError as exc:
            QMessageBox.warning(self, "Cannot split", str(exc))
            return
        self.refresh()
        self.registryChanged.emit(
            f"split {selected[0]} at {self.split_boundary:.2f}s into {original} and {new}"
        )

    def _add(self) -> None:
        speaker_id = self.registry.add_speaker()
        self.refresh()
        self.registryChanged.emit(f"added {speaker_id}")

    def _remove(self) -> None:
        selected = self._selected_ids()
        if len(selected) != 1:
            QMessageBox.information(
                self, "Remove speaker", "Select exactly one speaker to remove."
            )
            return
        confirm = QMessageBox.question(
            self,
            "Remove speaker",
            f"Remove {selected[0]}? Their segments become unattributed and the "
            "remaining speakers are renumbered.",
        )
        if confirm is not QMessageBox.StandardButton.Yes:
            return
        self.registry.remove_speaker(selected[0])
        self.refresh()
        self.registryChanged.emit(f"removed {selected[0]}")
