"""Transcription editor (CLAUDE.md sections 48, 49).

A plain-text editor with tag-insertion buttons and live validation.

Two behaviours are load-bearing rather than cosmetic:

  - Nothing autocorrects. The editor never rewrites what the annotator typed,
    even when the validator has a suggestion (sections 49, 56.21). Applying a
    suggestion is always an explicit action on a selected finding.
  - Tag buttons insert at the caret and never reorder or reformat the
    surrounding text. Tag *placement* carries meaning (section 40), so the
    editor puts the tag exactly where the caret is and leaves the rest alone.

Requires PySide6.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QFont, QTextCursor
    from PySide6.QtWidgets import (
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.transcription_editor requires PySide6. Install it with "
        "`pip install PySide6`."
    ) from exc

from app.ermis import tags as T
from app.ermis.findings import ValidationReport
from app.ermis.validator import ErmisValidator

#: Inline tag buttons, in the order listed in section 48.
INLINE_TAGS: tuple[str, ...] = (
    T.FILLED_PAUSE,
    T.NON_LEXICAL,
    T.LAUGHTER,
    T.BACKGROUND,
    T.ARTIFACT,
    T.UNINTELLIGIBLE,
    "{{}}",
    T.OVERLAP_OPEN,
    T.OVERLAP_CLOSE,
    T.CROSS_TALK,
)

#: Whole-file buttons (section 48). These replace the whole transcription,
#: because a whole-file tag must stand alone (section 42).
WHOLE_FILE_TAGS: tuple[str, ...] = (
    T.GARBLED,
    T.NO_AUDIO,
    T.NO_SPEECH,
    T.NON_TARGET,
    T.SENSITIVE,
)

#: Backchannel buttons with their intent, so the annotator picks by meaning
#: rather than trying to recall the exact spelling (section 7).
BACKCHANNEL_BUTTONS: tuple[tuple[str, str], ...] = (
    ("mm-hm", "closed-mouth agreement"),
    ("uh-huh", "open-mouth agreement"),
    ("mm-mm", "closed-mouth disagreement"),
    ("nuh-huh", "open-mouth disagreement"),
    ("hmm", "thinking"),
    ("uh", "open-vowel hesitation"),
    ("um", "nasal hesitation"),
    ("oh", "surprise"),
    ("ooh", "impressed"),
)

#: Delay before revalidating, so validation does not run on every keystroke.
_VALIDATE_DELAY_MS = 250


class TranscriptionEditor(QWidget):
    """Text editor plus tag palette, with debounced live validation."""

    textEdited = Signal(str)
    validated = Signal(object)  # ValidationReport

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._validator = ErmisValidator()
        self._speaker_count: int | None = None
        self._known_speakers: set[int] | None = None

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "Type the transcription you hear. Lowercase, no sentence punctuation, "
            "numbers as words. Nothing is auto-corrected."
        )
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(12)
        self.editor.setFont(font)
        # Single-line transcription: keep it visually compact but scrollable.
        self.editor.setMinimumHeight(90)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_VALIDATE_DELAY_MS)
        self._timer.timeout.connect(self._run_validation)
        self.editor.textChanged.connect(self._on_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(self._build_palette())
        layout.addWidget(self.status)

    # ------------------------------------------------------------------ state
    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, text: str) -> None:
        """Replace the text programmatically, e.g. when loading a saved draft."""
        self.editor.blockSignals(True)
        self.editor.setPlainText(text)
        self.editor.blockSignals(False)
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)
        self._run_validation()

    def set_speaker_context(
        self, speaker_count: int | None, known_speakers: set[int] | None
    ) -> None:
        """Give the validator the speaker panel's current state.

        Lets it check `<s1>` in single-speaker audio and speaker tags that
        refer to nobody (sections 17, 18A.10).
        """
        self._speaker_count = speaker_count
        self._known_speakers = known_speakers
        self._validator = ErmisValidator(
            known_speakers=known_speakers, speaker_count=speaker_count
        )
        self._run_validation()

    # ------------------------------------------------------------- validation
    def _on_changed(self) -> None:
        self.textEdited.emit(self.text())
        self._timer.start()

    def _run_validation(self) -> None:
        report = self._validator.validate(self.text())
        self._show_summary(report)
        self.validated.emit(report)

    def _show_summary(self, report: ValidationReport) -> None:
        if not self.text().strip():
            self.status.setText("")
            return
        errors, warnings = len(report.errors), len(report.warnings)
        if errors:
            self.status.setText(
                f"<b style='color:#e03131'>{errors} error(s)</b>, {warnings} warning(s)"
            )
        elif warnings:
            self.status.setText(
                f"<b style='color:#f59f00'>{warnings} warning(s)</b> -- these need the "
                "audio to settle"
            )
        else:
            self.status.setText("<b style='color:#2f9e44'>No findings</b>")

    # ---------------------------------------------------------------- palette
    def _build_palette(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        # --- inline tags ---
        tags_box = QGroupBox("Tags")
        tags_layout = QHBoxLayout(tags_box)
        for tag in INLINE_TAGS:
            button = QPushButton(tag)
            button.setToolTip(f"Insert {tag} at the caret")
            button.clicked.connect(lambda _=False, value=tag: self.insert_tag(value))
            tags_layout.addWidget(button)
        tags_layout.addStretch()
        outer.addWidget(tags_box)

        # --- speaker tags ---
        speaker_box = QGroupBox("Speakers")
        speaker_layout = QHBoxLayout(speaker_box)
        self._speaker_buttons: list[QPushButton] = []
        for number in range(1, 7):
            tag = f"<s{number}>"
            button = QPushButton(tag)
            button.clicked.connect(lambda _=False, value=tag: self.insert_tag(value))
            speaker_layout.addWidget(button)
            self._speaker_buttons.append(button)
        speaker_layout.addStretch()
        outer.addWidget(speaker_box)

        # --- backchannels ---
        back_box = QGroupBox("Backchannels and filled pauses")
        back_layout = QGridLayout(back_box)
        for index, (spelling, intent) in enumerate(BACKCHANNEL_BUTTONS):
            button = QPushButton(spelling)
            button.setToolTip(intent)
            button.clicked.connect(
                lambda _=False, value=spelling: self.insert_word(value)
            )
            back_layout.addWidget(button, index // 5, index % 5)
        fp = QPushButton(T.FILLED_PAUSE)
        fp.setToolTip("unidentifiable filled pause")
        fp.clicked.connect(lambda: self.insert_tag(T.FILLED_PAUSE))
        back_layout.addWidget(fp, 1, 4)
        outer.addWidget(back_box)

        # --- whole-file ---
        whole_box = QGroupBox("Whole-file conditions (replace the entire transcription)")
        whole_layout = QHBoxLayout(whole_box)
        for tag in WHOLE_FILE_TAGS:
            button = QPushButton(tag)
            button.setToolTip(
                f"Replace the transcription with {tag}. A whole-file tag must "
                "stand alone (section 42)."
            )
            button.clicked.connect(
                lambda _=False, value=tag: self.set_whole_file_tag(value)
            )
            whole_layout.addWidget(button)
        whole_layout.addStretch()
        outer.addWidget(whole_box)

        return container

    def update_speaker_buttons(self, count: int) -> None:
        """Enable only the speaker tags that exist.

        Single-speaker audio takes no speaker tag at all, so with one speaker
        every button is disabled (section 17).
        """
        for index, button in enumerate(self._speaker_buttons, start=1):
            button.setEnabled(count >= 2 and index <= count)

    # ---------------------------------------------------------------- editing
    def insert_tag(self, tag: str) -> None:
        """Insert a tag at the caret, space-separated from its neighbours.

        Both sides are checked. Inserting before a word has to add a trailing
        space too, or the tag runs into it -- '[hn]play' instead of
        '[hn] play' -- which the tokenizer would read as one malformed token.

        `(())` and `{{}}` place the caret inside the delimiters so a guess can
        be typed straight away.
        """
        cursor = self.editor.textCursor()
        text = self.editor.toPlainText()
        position = cursor.position()
        before, after = text[:position], text[position:]

        prefix = " " if before and not before.endswith(" ") else ""
        suffix = " " if after and not after.startswith(" ") else ""

        cursor.insertText(prefix + tag + suffix)
        if tag in (T.UNINTELLIGIBLE, "{{}}"):
            # Land between the delimiters: '((|))'.
            cursor.setPosition(cursor.position() - len(suffix) - 2)
            self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    def insert_word(self, word: str) -> None:
        """Insert a word, spaced from its neighbours on both sides."""
        cursor = self.editor.textCursor()
        text = self.editor.toPlainText()
        position = cursor.position()
        before, after = text[:position], text[position:]
        prefix = " " if before and not before.endswith(" ") else ""
        # No trailing space at the end of the text: section 41 forbids a
        # trailing space, and the validator would warn about it immediately.
        suffix = " " if after and not after.startswith(" ") else ""
        cursor.insertText(prefix + word + suffix)
        self.editor.setFocus()

    def set_whole_file_tag(self, tag: str) -> None:
        """Replace the entire transcription with a whole-file tag.

        Replacement rather than insertion, because section 42 forbids combining
        a whole-file tag with anything else. The previous text stays in the
        undo stack, so this is recoverable with Ctrl+Z.
        """
        cursor = self.editor.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(tag)
        self.editor.setFocus()

    def apply_suggestion(self, start: int, end: int, replacement: str) -> None:
        """Replace a span with a validator suggestion.

        Only ever called from an explicit user action on a finding. The
        validator itself never triggers this (sections 49, 56.21).
        """
        cursor = self.editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement)
        self.editor.setFocus()

    def highlight_range(self, start: int, end: int) -> None:
        """Select a span, used when a finding is clicked in the panel."""
        cursor = self.editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    # ------------------------------------------------------------ undo / redo
    def undo(self) -> None:
        self.editor.undo()

    def redo(self) -> None:
        self.editor.redo()
