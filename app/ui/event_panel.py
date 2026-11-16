"""Event confirmation panel (CLAUDE.md section 47).

Implements the human-confirmation workflow verbatim from section 47:

    1. Highlight the timeline
    2. Play the relevant audio
    3. Show the detected event
    4. Show confidence
    5. Let the user confirm / change / reject
    6. Insert the corresponding tag only after confirmation
    7. Keep an audit record

The panel is the only route by which a detected event can become a tag, and it
cannot do so without a click. Nothing here confirms anything on the user's
behalf, however confident the detector was (section 56.22).

Pause events appear in the list for navigation but offer no tag action at all,
because V2 has no pause tag (section 13).

Requires PySide6.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.event_panel requires PySide6. Install it with `pip install PySide6`."
    ) from exc

from app.ermis import tags as T
from app.models.event import AnalysisEvent, ConfirmationState, EventType

#: Tags the annotator may pick when overriding a suggestion. Section 47's
#: example offers "Change to hn" on a laughter detection, so any inline tag has
#: to be reachable.
_OVERRIDE_TAGS: tuple[str, ...] = (
    T.LAUGHTER,
    T.NON_LEXICAL,
    T.ARTIFACT,
    T.BACKGROUND,
    T.FILLED_PAUSE,
    T.GARBLED,
    T.NO_SPEECH,
    T.NO_AUDIO,
)

_STATE_COLOR = {
    ConfirmationState.PENDING: QColor("#f59f00"),
    ConfirmationState.CONFIRMED: QColor("#2f9e44"),
    ConfirmationState.REJECTED: QColor("#868e96"),
}


class EventPanel(QWidget):
    """Lists analysis events and drives their confirmation."""

    #: Emitted with (start, end) to select the event on the timeline.
    eventSelected = Signal(float, float)
    #: Emitted with (start, end) to replay just that segment.
    playRequested = Signal(float, float)
    #: Emitted after a human decision, for the audit log and the editor.
    eventConfirmed = Signal(object)  # AnalysisEvent
    eventRejected = Signal(object)  # AnalysisEvent
    #: Emitted with the tag to insert once an event is confirmed.
    tagRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: list[AnalysisEvent] = []

        self.summary = QLabel("No analysis run yet")
        self.summary.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Time", "Detected", "Confidence", "State"])
        # Wide enough for a full "MM:SS.CC - MM:SS.CC" range; the previous 130
        # clipped the end time to "00:00.28 - ...".
        self.tree.setColumnWidth(0, 170)
        self.tree.setColumnWidth(1, 200)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        small = QFont()
        small.setPointSize(8)
        self.detail.setFont(small)

        self.play_button = QPushButton("Play Segment")
        self.confirm_button = QPushButton("Confirm")
        self.change_combo = QComboBox()
        self.change_combo.addItem("Change to...", None)
        for tag in _OVERRIDE_TAGS:
            self.change_combo.addItem(tag, tag)
        self.reject_button = QPushButton("Reject")

        self.play_button.clicked.connect(self._play)
        self.confirm_button.clicked.connect(self._confirm)
        self.change_combo.activated.connect(self._change)
        self.reject_button.clicked.connect(self._reject)

        actions = QHBoxLayout()
        for widget in (
            self.play_button,
            self.confirm_button,
            self.change_combo,
            self.reject_button,
        ):
            actions.addWidget(widget)
        actions.addStretch()

        box = QGroupBox("DETECTED EVENTS -- every uncertain event needs your confirmation")
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(self.summary)
        box_layout.addWidget(self.tree)
        box_layout.addWidget(self.detail)
        box_layout.addLayout(actions)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        self._update_buttons(None)

    # ------------------------------------------------------------------ input
    def set_events(self, events: list[AnalysisEvent]) -> None:
        self._events = sorted(events, key=lambda e: (e.start, e.end))
        self.tree.clear()

        pending = [e for e in self._events if e.requires_human_confirmation]
        self.summary.setText(
            f"{len(self._events)} event(s), "
            f"<b style='color:#f59f00'>{len(pending)} awaiting confirmation</b>. "
            "A detection never becomes a tag until you confirm it."
        )

        groups: dict[str, QTreeWidgetItem] = {}
        for event in self._events:
            key = _group_name(event.type)
            if key not in groups:
                group = QTreeWidgetItem(self.tree, [key, "", "", ""])
                bold = QFont()
                bold.setBold(True)
                group.setFont(0, bold)
                group.setExpanded(True)
                # Let the group label span every column. Without this it is
                # clipped to the Time column, which silently hid the pause
                # group's "(metadata only -- never a tag)" reminder -- exactly
                # the section 13 safeguard the label exists to carry.
                group.setFirstColumnSpanned(True)
                groups[key] = group
            self._add_event(groups[key], event)

    def _add_event(self, parent: QTreeWidgetItem, event: AnalysisEvent) -> None:
        confidence = (
            f"{event.confidence * 100:.0f}%" if event.type.is_uncertain else "measured"
        )
        state = event.state.value
        if event.state is ConfirmationState.CONFIRMED and event.confirmed_tag:
            state = f"confirmed -> {event.confirmed_tag}"
        elif event.type is EventType.PAUSE:
            state = "metadata only"
        elif not event.type.is_uncertain and event.state is ConfirmationState.PENDING:
            # A measured region has no confirmation semantics. Showing the raw
            # "pending" here implied action was needed on every speech region
            # and contradicted the summary's "N awaiting confirmation" count,
            # which only counts uncertain events.
            state = "n/a"

        item = QTreeWidgetItem(
            parent,
            [
                f"{_ts(event.start)} - {_ts(event.end)}",
                event.type.value.replace("_", " "),
                confidence,
                state,
            ],
        )
        # Measured rows are grey, not pending-orange: nothing is waiting on
        # the annotator there.
        if not event.type.is_uncertain and event.state is ConfirmationState.PENDING:
            item.setForeground(3, _STATE_COLOR[ConfirmationState.REJECTED])
        else:
            item.setForeground(3, _STATE_COLOR[event.state])
        item.setData(0, Qt.ItemDataRole.UserRole, event)
        if event.speaker_id:
            item.setToolTip(0, f"speaker {event.speaker_id}")

    # --------------------------------------------------------------- reaction
    def _selected(self) -> AnalysisEvent | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _on_selection(self) -> None:
        event = self._selected()
        self._update_buttons(event)
        if event is None:
            self.detail.setText("")
            return
        self.eventSelected.emit(event.start, event.end)

        parts = [f"<b>{event.type.value.replace('_', ' ')}</b>"]
        if event.type.is_uncertain:
            parts.append(
                f"confidence {event.confidence * 100:.0f}% -- a detector estimate, "
                "not a verdict"
            )
        if event.note:
            parts.append(event.note)
        self.detail.setText("<br>".join(parts))

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        event = item.data(0, Qt.ItemDataRole.UserRole)
        if event is not None:
            self.playRequested.emit(event.start, event.end)

    def _update_buttons(self, event: AnalysisEvent | None) -> None:
        # A pause offers no tag action at all: V2 has no pause tag, so there is
        # nothing to confirm into the transcription (section 13).
        taggable = (
            event is not None
            and event.proposed_tag is not None
            and event.type is not EventType.PAUSE
        )
        self.play_button.setEnabled(event is not None)
        self.confirm_button.setEnabled(bool(taggable))
        self.change_combo.setEnabled(bool(taggable))
        self.reject_button.setEnabled(bool(event is not None and event.type.is_uncertain))
        if event is not None and event.proposed_tag:
            self.confirm_button.setText(f"Confirm {event.proposed_tag}")
        else:
            self.confirm_button.setText("Confirm")

    # ---------------------------------------------------------------- actions
    def _play(self) -> None:
        event = self._selected()
        if event is not None:
            self.playRequested.emit(event.start, event.end)

    def _confirm(self, tag: str | None = None) -> None:
        event = self._selected()
        if event is None:
            return
        event.confirm(tag)
        self.eventConfirmed.emit(event)
        if event.confirmed_tag:
            self.tagRequested.emit(event.confirmed_tag)
        self.set_events(self._events)

    def _change(self, index: int) -> None:
        tag = self.change_combo.itemData(index)
        self.change_combo.setCurrentIndex(0)
        if tag:
            self._confirm(tag)

    def _reject(self) -> None:
        event = self._selected()
        if event is None:
            return
        event.reject("rejected by annotator after listening")
        self.eventRejected.emit(event)
        self.set_events(self._events)


def _group_name(event_type: EventType) -> str:
    if event_type is EventType.SPEECH:
        return "Speech regions"
    if event_type in (EventType.PAUSE, EventType.SILENCE):
        return "Pauses and silence (metadata only -- never a tag)"
    if event_type is EventType.SPEAKER_CHANGE:
        return "Speaker changes"
    if event_type in (EventType.NO_AUDIO, EventType.NO_SPEECH, EventType.GARBLED_AUDIO):
        return "Whole-file conditions"
    return "Candidates awaiting confirmation"


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"
