"""Validation panel (CLAUDE.md section 49).

Shows findings grouped by severity, each citing its rule. Clicking a finding
selects the offending span in the editor.

Section 49's requirement drives the design: validation is non-destructive and
must never automatically rewrite the transcription. So a suggestion is shown as
text, and the "Apply suggestion" button is deliberately a separate, explicit
action on a selected finding -- never automatic, never bulk.

Requires PySide6.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.validation_panel requires PySide6. Install it with "
        "`pip install PySide6`."
    ) from exc

from app.ermis.findings import Finding, Severity, ValidationReport

_SEVERITY_COLOR = {
    Severity.ERROR: QColor("#e03131"),
    Severity.WARNING: QColor("#f59f00"),
    Severity.INFO: QColor("#4dabf7"),
}

_SEVERITY_HINT = {
    Severity.ERROR: "definite rule violations",
    Severity.WARNING: "probable, or need the audio to settle",
    Severity.INFO: "informational, no action needed",
}


class ValidationPanel(QWidget):
    """Tree of findings, grouped by severity."""

    #: Emitted with (start, end) when a finding is selected.
    findingSelected = Signal(int, int)
    #: Emitted with (start, end, replacement) when the user applies a suggestion.
    suggestionApplied = Signal(int, int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: ValidationReport | None = None

        self.summary = QLabel("No transcription yet")
        self.summary.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Finding", "Rule", "Suggestion"])
        # The suggestion is the actionable part of a finding, so it must never
        # be clipped. Fixed widths did clip it ("mm-..." instead of "mm-hm"),
        # so Rule and Suggestion size to their contents and the Finding message
        # absorbs the remaining width.
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)

        self.apply_button = QPushButton("Apply suggestion")
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip(
            "Replace the highlighted span with the suggested text. Validation "
            "never changes your transcription on its own (section 49)."
        )
        self.apply_button.clicked.connect(self._apply_selected)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.tree)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ input
    def set_report(self, report: ValidationReport) -> None:
        self._report = report
        self.tree.clear()
        self.apply_button.setEnabled(False)

        if not report.text.strip():
            self.summary.setText("No transcription yet")
            return

        if not report.findings:
            self.summary.setText(
                "<b style='color:#2f9e44'>No findings.</b> The transcription "
                "satisfies every rule the validator can check from text alone."
            )
            return

        self.summary.setText(
            f"<b>{len(report.errors)}</b> error(s), "
            f"<b>{len(report.warnings)}</b> warning(s), "
            f"<b>{len(report.infos)}</b> info"
        )

        for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
            findings = [f for f in report.sorted_findings() if f.severity is severity]
            if not findings:
                continue
            group = QTreeWidgetItem(
                self.tree,
                [
                    f"{severity.value} ({len(findings)}) -- {_SEVERITY_HINT[severity]}",
                    "",
                    "",
                ],
            )
            bold = QFont()
            bold.setBold(True)
            group.setFont(0, bold)
            group.setForeground(0, _SEVERITY_COLOR[severity])
            group.setExpanded(True)

            for finding in findings:
                self._add_finding(group, finding)

    def _add_finding(self, parent: QTreeWidgetItem, finding: Finding) -> None:
        label = finding.message
        if finding.excerpt and finding.end > finding.start:
            label = f"{finding.excerpt!r}: {label}"
        item = QTreeWidgetItem(
            parent,
            [label, finding.rule_ref, finding.suggestion or ""],
        )
        item.setForeground(0, _SEVERITY_COLOR[finding.severity])
        item.setToolTip(0, f"{finding.code.value} -- {finding.message}")
        item.setData(0, Qt.ItemDataRole.UserRole, finding)

    # --------------------------------------------------------------- reaction
    def _selected_finding(self) -> Finding | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _on_selection(self) -> None:
        finding = self._selected_finding()
        has_span = finding is not None and finding.end > finding.start
        # A suggestion is only applicable when it targets a real span. A
        # whole-transcription finding has no span to replace.
        self.apply_button.setEnabled(
            bool(finding and finding.suggestion is not None and has_span)
        )
        if finding and has_span:
            self.findingSelected.emit(finding.start, finding.end)

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        finding = item.data(0, Qt.ItemDataRole.UserRole)
        if finding and finding.end > finding.start:
            self.findingSelected.emit(finding.start, finding.end)

    def _apply_selected(self) -> None:
        finding = self._selected_finding()
        if finding is None or finding.suggestion is None:
            return
        if finding.end <= finding.start:
            return
        self.suggestionApplied.emit(finding.start, finding.end, finding.suggestion)


class PauseInfoPanel(QWidget):
    """Standing reminder that pause is metadata, not a tag (section 13).

    Detected pauses are listed here, never in the validation tree, so a pause
    can never read as something to fix by adding a tag.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.set_pauses([])

    def set_pauses(self, pauses: list[tuple[float, float]]) -> None:
        if not pauses:
            self.label.setText("<i>No pauses detected.</i>")
            return
        lines = [
            "<b>Pauses detected (internal metadata only).</b> Ermis V2 defines no "
            "transcription tag for pauses -- do not insert one.<br>"
        ]
        for start, end in pauses[:12]:
            lines.append(f"&nbsp;&nbsp;{end - start:.1f}s pause at {start:.2f}s<br>")
        if len(pauses) > 12:
            lines.append(f"&nbsp;&nbsp;... and {len(pauses) - 12} more")
        self.label.setText("".join(lines))
