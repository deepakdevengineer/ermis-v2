"""Main window (CLAUDE.md sections 47, 48, 49, 50).

Wires the pieces together: waveform, playback transport, transcription editor,
validation panel, speaker panel, event confirmation panel, and the audit log.

The window enforces the two-layer separation of section 3 at the seams:

  - Analysis output populates the timeline and the event panel. It never
    touches the transcription text.
  - The only path from a detected event to the transcription is the event
    panel's Confirm action, which the annotator has to click.
  - Every confirmation, rejection, metadata edit, save and export is written
    to the audit log (section 50).

Requires PySide6.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.main_window requires PySide6. Install it with `pip install PySide6`."
    ) from exc

from app.asr.draft import DraftResult
from app.audio.loader import AudioLoader, AudioLoadError
from app.audio.playback import PlaybackEngine
from app.config import APP_NAME, APP_VERSION, ERMIS_VERSION, AppConfig
from app.detection.pipeline import AnalysisPipeline, AnalysisResult
from app.ermis.crosscheck import apply_cross_check
from app.ermis.findings import ValidationReport
from app.models.event import EventType
from app.speakers.metadata import SpeakerRegistry
from app.speakers.nativity import apply_nativity, suggest_nativity
from app.speakers.voice_profile import apply_suggestions, suggest_voice_profiles
from app.storage.audit import Action, AuditLog
from app.storage.database import Database
from app.storage.export import Annotation, Exporter, ExportError
from app.ui.draft_worker import DraftWorker, ModelPreloadWorker
from app.ui.event_panel import EventPanel
from app.ui.record_bar import RecordBar
from app.ui.speaker_panel import SpeakerPanel
from app.ui.transcription_editor import TranscriptionEditor
from app.ui.validation_panel import PauseInfoPanel, ValidationPanel
from app.ui.waveform_view import WaveformView

_POSITION_POLL_MS = 40


class MainWindow(QMainWindow):
    """The application shell."""

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or AppConfig()
        self.config.paths.ensure()

        self.loader = AudioLoader(self.config.audio)
        self.playback = PlaybackEngine(self.config.audio)
        self.pipeline = AnalysisPipeline(self.config.detection)

        self.db = Database(self.config.paths.database_path)
        self.audit = AuditLog(self.db)
        self.project_id = self._ensure_project()

        self.buffer = None
        self.analysis: AnalysisResult | None = None
        self.audio_file_id: int | None = None
        self._last_report: ValidationReport | None = None
        #: One-line description of the latest analysis, so a caller with its
        #: own status message can carry it instead of overwriting it.
        self._analysis_summary: str = ""

        #: The most recent machine draft, and whether the editor still holds it
        #: unchanged. Provenance follows from these two: unedited draft text is
        #: saved as 'asr_draft', text a person touched as 'human_edited'.
        self._draft: DraftResult | None = None
        self._text_is_draft = False
        #: Latest automatic speaker-descriptor suggestions, kept for the panel
        #: and for the audit record.
        self._voice_suggestions: dict = {}
        self._nativity_suggestions: dict = {}
        self._draft_worker: DraftWorker | None = None
        self._preload_worker: ModelPreloadWorker | None = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} — Ermis {ERMIS_VERSION}")
        self.resize(1400, 900)
        self._build_ui()
        self._build_menu()

        # Playback position is polled rather than pushed, because the audio
        # callback runs on a different thread and must not touch widgets.
        self._poll = QTimer(self)
        self._poll.setInterval(_POSITION_POLL_MS)
        self._poll.timeout.connect(self._sync_playhead)
        self._poll.start()

        self._preload_model()

    def _preload_model(self) -> None:
        """Warm the speech model while the annotator gets started."""
        if not (self.config.asr.enabled and self.config.asr.preload_on_start):
            return
        worker = ModelPreloadWorker(self.config.asr, self)
        worker.ready.connect(
            lambda name: self._set_status(f"Speech model ready ({name}).")
        )
        worker.failed.connect(
            lambda reason: self._set_status(
                f"Drafting unavailable: {reason} Transcription is typed by hand."
            )
        )
        worker.finished.connect(self._on_preload_finished)
        self._preload_worker = worker
        worker.start()
        self._set_status(
            "Loading the speech model in the background; recording and analysis "
            "work meanwhile."
        )

    def _on_preload_finished(self) -> None:
        self._preload_worker = None

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.waveform = WaveformView()
        self.waveform.positionClicked.connect(self._on_waveform_click)
        self.waveform.selectionChanged.connect(self._on_selection)

        self.editor = TranscriptionEditor()
        self.editor.validated.connect(self._on_validated)
        # A keystroke in the editor means the text is no longer purely machine
        # output, so it stops being labelled as a draft.
        self.editor.textEdited.connect(self._on_text_edited)

        self.validation = ValidationPanel()
        self.validation.findingSelected.connect(self.editor.highlight_range)
        self.validation.suggestionApplied.connect(self.editor.apply_suggestion)

        self.pauses = PauseInfoPanel()

        self.speakers = SpeakerPanel()
        self.speakers.registryChanged.connect(self._on_registry_changed)
        self.speakers.metadataEdited.connect(self._on_metadata_edited)

        self.events = EventPanel()
        self.events.eventSelected.connect(self._on_event_selected)
        self.events.playRequested.connect(self._play_segment)
        self.events.eventConfirmed.connect(self._on_event_confirmed)
        self.events.eventRejected.connect(self._on_event_rejected)
        self.events.tagRequested.connect(self.editor.insert_tag)

        right = QTabWidget()
        right.addTab(self.validation, "Validation")
        right.addTab(self.events, "Events")
        right.addTab(self.speakers, "Speakers")
        right.addTab(self.pauses, "Pauses")

        # Capture bar: the input side of the transport. A finished recording
        # is handed back as a file path and then loaded like any other clip,
        # so nothing downstream needs to know where the audio came from.
        self.record_bar = RecordBar(self.config)
        self.record_bar.recordingStarted.connect(self.playback.stop)
        self.record_bar.recordingFinished.connect(self._on_recording_finished)
        self.record_bar.statusMessage.connect(self._set_status)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.waveform, stretch=3)
        left_layout.addWidget(self._build_transport())
        left_layout.addWidget(self.record_bar)
        left_layout.addWidget(self.editor, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([840, 560])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._set_status("Open an audio file, or record from the microphone, to begin.")

    def _build_transport(self) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.playback.stop)
        self.replay_button = QPushButton("Replay selection")
        self.replay_button.setToolTip("Replay the selected segment (section 48)")
        self.replay_button.clicked.connect(self._replay_selection)
        self.loop_button = QPushButton("Loop selection")
        self.loop_button.clicked.connect(self._loop_selection)

        self.speed = QComboBox()
        for value in self.config.audio.playback_speeds:
            self.speed.addItem(f"{value}x", value)
        self.speed.setCurrentIndex(list(self.config.audio.playback_speeds).index(1.0))
        self.speed.currentIndexChanged.connect(
            lambda: self.playback.set_speed(self.speed.currentData())
        )

        self.position_label = QLabel("00:00.00 / 00:00.00")

        for widget in (
            self.play_button,
            self.stop_button,
            self.replay_button,
            self.loop_button,
        ):
            row.addWidget(widget)
        row.addWidget(QLabel("Speed:"))
        row.addWidget(self.speed)
        row.addStretch()
        row.addWidget(self.position_label)
        return container

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open audio...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_audio)
        file_menu.addAction(open_action)

        record_action = QAction("&Record from microphone", self)
        record_action.setShortcut("Ctrl+Shift+R")
        record_action.setToolTip(
            "Capture a clip locally instead of opening an attachment"
        )
        record_action.triggered.connect(self.record_bar.toggle_recording)
        file_menu.addAction(record_action)

        discard_action = QAction("Discard recording in progress", self)
        discard_action.triggered.connect(self.record_bar.discard_recording)
        file_menu.addAction(discard_action)

        save_action = QAction("&Save transcription", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_transcription)
        file_menu.addAction(save_action)

        export_action = QAction("&Export annotation...", self)
        export_action.triggered.connect(self.export_annotation)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo = QAction("&Undo", self)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.triggered.connect(self.editor.undo)
        edit_menu.addAction(undo)
        redo = QAction("&Redo", self)
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        redo.triggered.connect(self.editor.redo)
        edit_menu.addAction(redo)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        run = QAction("&Run analysis", self)
        run.setShortcut("Ctrl+R")
        run.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run)

        analysis_menu.addSeparator()
        draft_action = QAction("&Draft transcription from audio", self)
        draft_action.setShortcut("Ctrl+D")
        draft_action.setToolTip(
            "Run the local recogniser and fill the editor with a draft to correct"
        )
        draft_action.triggered.connect(self.start_draft)
        analysis_menu.addAction(draft_action)

        self.auto_draft_action = QAction("Draft automatically after loading", self)
        self.auto_draft_action.setCheckable(True)
        self.auto_draft_action.setChecked(self.config.asr.auto_draft_on_load)
        self.auto_draft_action.toggled.connect(self._set_auto_draft)
        analysis_menu.addAction(self.auto_draft_action)

        play_menu = self.menuBar().addMenu("&Playback")
        toggle = QAction("Play / Pause", self)
        toggle.setShortcut("Space")
        toggle.triggered.connect(self._toggle_play)
        play_menu.addAction(toggle)
        replay = QAction("Replay selection", self)
        replay.setShortcut("Ctrl+Space")
        replay.triggered.connect(self._replay_selection)
        play_menu.addAction(replay)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    # ------------------------------------------------------------------ files
    def _ensure_project(self) -> int:
        for row in self.db.list_projects():
            if row["name"] == "default":
                return int(row["id"])
        return self.db.create_project("default", "Created automatically")

    def open_audio(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in self.config.audio.supported_extensions)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio", str(self.config.paths.data_dir), f"Audio ({patterns})"
        )
        if path:
            self.load_audio(path)

    def load_audio(self, path: str | Path) -> None:
        try:
            buffer = self.loader.load(path)
        except AudioLoadError as exc:
            QMessageBox.critical(self, "Cannot open audio", str(exc))
            return

        self.buffer = buffer
        self.playback.set_audio(buffer)
        self.waveform.set_audio(buffer)
        self.analysis = None
        self._analysis_summary = ""
        self._draft = None
        self._text_is_draft = False
        self._update_title()
        self.events.set_events([])
        self.pauses.set_pauses([])
        self.speakers.set_registry(SpeakerRegistry(), None, None)

        self.audio_file_id = self._register_audio(buffer)
        self.audit.audio_loaded(self.audio_file_id, str(buffer.path), buffer.duration)

        existing = self.db.latest_transcription(self.audio_file_id)
        self.editor.set_text(existing["text"] if existing else "")

        self._set_status(
            f"{buffer.path.name} — {buffer.duration:.2f}s, "
            f"{buffer.source_sample_rate} Hz, {buffer.source_channels} ch "
            f"(analysed at {buffer.sample_rate} Hz mono)"
            + ("  ·  NO AUDIO SIGNAL: <na> may apply" if buffer.is_silent else "")
        )
        self.run_analysis()

    def _register_audio(self, buffer) -> int:
        digest = Database.file_hash(buffer.path)
        for row in self.db.list_audio_files(self.project_id):
            if row["path"] == str(buffer.path):
                return int(row["id"])
        return self.db.add_audio_file(
            self.project_id,
            buffer.path,
            buffer.duration,
            buffer.sample_rate,
            buffer.source_channels,
            content_hash=digest,
            peak_dbfs=None if buffer.is_silent else buffer.peak_dbfs,
            rms_dbfs=None if buffer.is_silent else buffer.rms_dbfs,
            is_silent=buffer.is_silent,
        )

    def _on_recording_finished(self, path: str) -> None:
        """Load a just-captured file and record where it came from.

        Recording produces audio, nothing else: the clip goes through the same
        loader, analysis and validation path as an attachment, and the
        transcription box stays empty until a person types in it (section 3).
        """
        self.load_audio(path)
        if self.buffer is None or self.buffer.path != Path(path):
            return  # the load failed and already reported why

        result = self.record_bar.last_result
        if result is not None and self.audio_file_id is not None:
            self.audit.recording_captured(self.audio_file_id, result)

        # `load_audio` has already run the analysis and put its description in
        # the status bar. Carry that description into this message rather than
        # replacing it: what the analysis found is the reason to look at the
        # timeline, and a recording is the case where the annotator has not
        # seen the audio in any other form.
        parts = [f"Recorded {self.buffer.duration:.2f}s into {Path(path).name}"]
        parts.extend(result.notes() if result is not None else [])
        if self._analysis_summary:
            parts.append(self._analysis_summary)
        parts.append(self._provenance_note())
        self._set_status("  ·  ".join(parts))

    def _provenance_note(self) -> str:
        """Who produced the words, said in the configuration's own terms.

        With drafting off this is section 1 as written: the tool analyses, the
        annotator types. With drafting on the honest sentence is a different
        one -- the words came from a machine and are not a submission until a
        person has corrected them (section 60.2).
        """
        if self.config.asr.enabled:
            return (
                "Any drafted words are machine output — replay the audio and "
                "correct them before saving."
            )
        return "Type the transcription yourself — this tool does not transcribe audio."

    # --------------------------------------------------------------- analysis
    def run_analysis(self) -> None:
        if self.buffer is None:
            return
        self._set_status("Running analysis...")
        result = self.pipeline.run(self.buffer)
        self.analysis = result

        self.waveform.set_events(result.events)
        self.events.set_events(result.events)
        self.pauses.set_pauses([(e.start, e.end) for e in result.pause_events])

        registry = SpeakerRegistry.from_diarization(result.diarization)
        # Suggest the voice descriptor from the audio before the panel is
        # populated. Suggestions are assistive: only confident ones are written,
        # a profile the annotator edited is never touched, and nativity is never
        # guessed (sections 18A.3, 18A.4, 18A.11).
        self._voice_suggestions = suggest_voice_profiles(self.buffer, result.diarization)
        suggested = (
            apply_suggestions(
                registry,
                self._voice_suggestions,
                min_confidence=self.config.speakers.min_gender_confidence,
            )
            if self.config.speakers.suggest_gender
            else []
        )
        self.speakers.set_registry(
            registry, result.speaker_count, result.diarization.confidence
        )
        self._sync_speaker_context()
        if suggested and self.audio_file_id is not None:
            self.db.append_audit(
                "speaker_metadata_suggested",
                audio_file_id=self.audio_file_id,
                detail={
                    "applied": suggested,
                    "measurements": [
                        s.to_dict() for s in self._voice_suggestions.values()
                    ],
                },
            )

        if self.audio_file_id is not None:
            self.db.add_events(self.audio_file_id, result.events)
            self.db.append_audit(
                "analysis_run",
                audio_file_id=self.audio_file_id,
                detail=result.summary()
                | {"events": len(result.events)},  # keep the row compact
            )

        if self.config.asr.enabled and self.config.asr.auto_draft_on_load:
            self.start_draft(auto=True)

        self._analysis_summary = self._describe_analysis(result)
        self._set_status(self._analysis_summary)

    @staticmethod
    def _describe_analysis(result: AnalysisResult) -> str:
        """What the analysis found, as one line for the status bar.

        Kept separate from `run_analysis` so a caller that reports something of
        its own -- a finished recording -- can include this rather than
        overwrite it.
        """
        message = (
            f"Analysis complete — {len(result.speech_events)} speech region(s), "
            f"{len(result.pause_events)} pause(s), "
            f"{result.speaker_count} speaker(s) suggested, "
            f"{len(result.pending)} candidate(s) awaiting confirmation"
        )
        candidates = result.whole_file_candidates
        if candidates:
            message += f"  ·  whole-file candidate(s): {' '.join(candidates)}"
        return message

    # ---------------------------------------------------------------- drafting
    def _set_auto_draft(self, enabled: bool) -> None:
        self.config.asr.auto_draft_on_load = bool(enabled)
        self._set_status(
            "Automatic drafting is on: each clip is recognised after analysis."
            if enabled
            else "Automatic drafting is off. Use Analysis > Draft transcription."
        )

    def start_draft(self, auto: bool = False) -> None:
        """Recognise the audio and offer the result as a draft.

        Runs on a worker thread: the first clip pays for loading the model, and
        blocking the GUI for that would look like a hang.
        """
        if self.buffer is None or self.analysis is None:
            self._set_status("Open or record audio first.")
            return
        if not self.config.asr.enabled:
            self._set_status("ASR is disabled in configuration (asr.enabled).")
            return
        if self._draft_worker is not None and self._draft_worker.isRunning():
            self._set_status("A draft is already being produced.")
            return

        worker = DraftWorker(
            self.buffer, self.analysis, self.config.asr, self.config.validation, self
        )
        worker.progress.connect(self._set_status)
        worker.drafted.connect(lambda result: self._on_drafted(result, auto))
        worker.failed.connect(self._on_draft_failed)
        worker.finished.connect(self._on_draft_worker_finished)
        self._draft_worker = worker
        worker.start()

    def _on_draft_worker_finished(self) -> None:
        self._draft_worker = None

    def _on_drafted(self, result: DraftResult, auto: bool = False) -> None:
        """Apply a finished draft, or keep it aside if the editor has text."""
        self._draft = result
        if self.audio_file_id is not None:
            self.db.append_audit(
                Action.DRAFT_CREATED,
                audio_file_id=self.audio_file_id,
                detail={
                    "backend": result.asr.backend if result.asr else None,
                    "model": result.asr.model if result.asr else None,
                    "words": len(result.words),
                    "unclear_words": result.unclear_count,
                    "inserted_tags": [tag for _at, tag in result.inserted_tags],
                    "whole_file": result.whole_file,
                    "valid": bool(result.report and result.report.is_valid),
                    "notes": result.notes,
                },
            )

        existing = self.editor.text().strip()
        if existing and not self._text_is_draft:
            # Never overwrite text a person wrote (section 56.21). The draft is
            # kept and can be applied deliberately.
            answer = QMessageBox.question(
                self,
                "Replace your text with the draft?",
                "A machine draft is ready, but the editor already contains your "
                "text.\n\nReplace it? Your version stays in the saved revision "
                "history either way.",
            )
            if answer is not QMessageBox.StandardButton.Yes:
                self._set_status(
                    "Draft kept aside; your text was not touched. "
                    "Analysis > Draft transcription applies it."
                )
                return

        self.apply_draft(result)

    def apply_draft(self, result: DraftResult) -> None:
        """Put draft text in the editor and record what it is."""
        self._draft = result
        self.editor.set_text(result.text)
        self._text_is_draft = True
        self._update_title()

        # The tags in the draft came from detections, so those detections are
        # marked confirmed-by-draft rather than left pending: the text and the
        # event panel must not disagree. The note records that a machine, not a
        # person, made the call -- section 47 wants a human decision, and this
        # is the honest record of a deviation from it.
        for event in result.used_events:
            if not event.is_confirmed:
                event.confirm(event.proposed_tag)
                event.note = "auto-confirmed by ASR draft; verify against the audio"
                if self.audio_file_id is not None:
                    self.db.append_audit(
                        Action.EVENT_AUTO_CONFIRMED,
                        audio_file_id=self.audio_file_id,
                        event_type=event.type.value,
                        audio_start=event.start,
                        audio_end=event.end,
                        detail={
                            "tag": event.confirmed_tag,
                            "detector_confidence": event.confidence,
                        },
                    )
        if result.used_events:
            self.events.set_events(self.analysis.events if self.analysis else [])

        self.editor._run_validation()
        self._suggest_nativity(result)

        message = (
            f"Draft ready — {len(result.words)} word(s)"
            + (f", {result.unclear_count} unclear" if result.unclear_count else "")
            + (
                f", tags: {' '.join(tag for _at, tag in result.inserted_tags)}"
                if result.inserted_tags
                else ""
            )
            + ". This is machine text: replay the audio and correct it before saving."
        )
        if result.notes:
            message += "  ·  " + "  ·  ".join(result.notes[:2])
        self._set_status(message)

    def _suggest_nativity(self, result: DraftResult) -> None:
        """Suggest nativity now that words exist to measure fluency on.

        Deviation from section 18A.4, which forbids guessing nativity; see
        CLAUDE.md section 60.4. Only confident suggestions are written, a
        profile the annotator edited is left alone, and every value stays
        editable in the panel (section 18A.11).
        """
        if self.buffer is None or self.analysis is None:
            return
        if not self.config.speakers.suggest_nativity:
            return
        self._nativity_suggestions = suggest_nativity(
            self.buffer,
            self.analysis.diarization,
            result.words,
            self.config.speakers,
        )
        applied = apply_nativity(
            self.speakers.registry, self._nativity_suggestions, self.config.speakers
        )
        if not applied:
            return
        self.speakers.refresh()
        if self.audio_file_id is not None:
            self.db.append_audit(
                "speaker_nativity_suggested",
                audio_file_id=self.audio_file_id,
                detail={
                    "applied": applied,
                    "evidence": [
                        s.to_dict() for s in self._nativity_suggestions.values()
                    ],
                },
            )

    def _on_draft_failed(self, reason: str) -> None:
        """Report a failed draft without losing what the analysis found.

        The draft arrives asynchronously, so this lands after the load or
        recording message and would otherwise be the last thing written to the
        status bar -- leaving the annotator with the reason there is no draft
        and no sign of the speech regions, pauses and speakers that were
        found regardless.
        """
        from app.asr.engine import available_backends

        parts = [f"No draft: {reason}"]
        # Typing it is always available, so that half is said either way. The
        # install hint is only true when nothing is installed -- with a backend
        # present it sends the annotator after a package they already have.
        parts.append(
            "Install a recogniser (see README), or type the transcription yourself."
            if not available_backends()
            else "Type the transcription yourself, or retry from Analysis > Draft."
        )
        if self._analysis_summary:
            parts.append(self._analysis_summary)
        self._set_status("  ·  ".join(parts))

    def _on_text_edited(self, _text: str) -> None:
        if self._text_is_draft:
            self._text_is_draft = False
            self._update_title()

    def _update_title(self) -> None:
        base = f"{APP_NAME} {APP_VERSION} — Ermis {ERMIS_VERSION}"
        if self.buffer is not None:
            base += f" — {Path(self.buffer.path).name}"
        if self._text_is_draft:
            base += "  [UNREVIEWED MACHINE DRAFT]"
        self.setWindowTitle(base)

    @property
    def transcription_source(self) -> str:
        """Provenance label for whatever is in the editor right now."""
        if self._text_is_draft:
            return "asr_draft"
        if self._draft is not None:
            return "human_edited"
        return "human"

    # -------------------------------------------------------------- playback
    def _toggle_play(self) -> None:
        if self.buffer is None:
            return
        if not self.playback.available:
            self._set_status(
                "No audio output device available. Install sounddevice and check "
                "your output device."
            )
            return
        self.playback.toggle()
        self.play_button.setText("Pause" if self.playback.is_playing else "Play")

    def _play_segment(self, start: float, end: float) -> None:
        if self.buffer is None:
            return
        self.waveform.set_selection(start, end)
        self.playback.play_segment(start, end)
        self.play_button.setText("Pause")

    def _replay_selection(self) -> None:
        selection = self.waveform.selection
        if selection is None:
            self._set_status("Select a region in the waveform first.")
            return
        self._play_segment(*selection)

    def _loop_selection(self) -> None:
        selection = self.waveform.selection
        if selection is None:
            self._set_status("Select a region in the waveform first.")
            return
        self.playback.play_segment(*selection, loop=True)
        self.play_button.setText("Pause")

    def _sync_playhead(self) -> None:
        if self.buffer is None:
            return
        position = self.playback.position
        self.waveform.set_playhead(position)
        self.position_label.setText(
            f"{_ts(position)} / {_ts(self.buffer.duration)}"
        )
        if not self.playback.is_playing and self.play_button.text() == "Pause":
            self.play_button.setText("Play")

    # ----------------------------------------------------------------- events
    def _on_waveform_click(self, seconds: float) -> None:
        self.playback.seek(seconds)
        self.speakers.set_split_boundary(seconds)

    def _on_selection(self, start: float, _end: float) -> None:
        self.speakers.set_split_boundary(start)

    def _on_event_selected(self, start: float, end: float) -> None:
        self.waveform.set_selection(start, end)
        # Keep the event visible when the view is zoomed in.
        if not (self.waveform._view_start <= start <= self.waveform._view_end):
            span = self.waveform._view_end - self.waveform._view_start
            self.waveform.set_view_range(start - span / 4, start - span / 4 + span)

    def _on_event_confirmed(self, event) -> None:
        if self.audio_file_id is not None:
            self.audit.event_confirmed(self.audio_file_id, event)
        # Re-run validation so the cross-check reflects the new confirmation.
        self.editor._run_validation()
        self._set_status(
            f"Confirmed {event.type.value} at {event.start:.2f}s"
            + (f" as {event.confirmed_tag}" if event.confirmed_tag else "")
        )

    def _on_event_rejected(self, event) -> None:
        if self.audio_file_id is not None:
            self.audit.event_rejected(self.audio_file_id, event)
        self._set_status(f"Rejected {event.type.value} at {event.start:.2f}s")

    # --------------------------------------------------------------- speakers
    def _on_registry_changed(self, description: str) -> None:
        self._sync_speaker_context()
        if self.audio_file_id is not None:
            self.db.append_audit(
                "speaker_structure_changed",
                audio_file_id=self.audio_file_id,
                detail={"change": description, "count": self.speakers.registry.count},
            )
        self.waveform.set_events(
            self.speakers.registry.to_events()
            + [
                e
                for e in (self.analysis.events if self.analysis else [])
                if e.type is not EventType.SPEECH
            ]
        )
        self._set_status(f"Speakers: {description}")

    def _on_metadata_edited(self, _speaker_id: str, before, after) -> None:
        if self.audio_file_id is None:
            return
        self.db.upsert_speaker(self.audio_file_id, after, human_edited=True)
        self.audit.speaker_metadata_changed(self.audio_file_id, before, after)

    def _sync_speaker_context(self) -> None:
        """Give the editor and validator the current speaker structure."""
        registry = self.speakers.registry
        self.editor.update_speaker_buttons(registry.count)
        self.editor.set_speaker_context(
            speaker_count=registry.count or None,
            known_speakers=registry.numbers or None,
        )

    # ------------------------------------------------------------- validation
    def _on_validated(self, report: ValidationReport) -> None:
        """Add the audio cross-check, then show the report (section 49).

        Only confirmed events participate, so an unconfirmed detection never
        pressures the annotator into adding a tag they have not agreed to.
        """
        if self.analysis is not None:
            apply_cross_check(report, self.analysis.events, self.config.validation)
        self._last_report = report
        self.validation.set_report(report)

    # ------------------------------------------------------------------- save
    def save_transcription(self) -> None:
        if self.audio_file_id is None:
            self._set_status("Open an audio file first.")
            return
        text = self.editor.text().strip()
        if not text:
            self._set_status("Nothing to save.")
            return

        # Provenance follows the editor's state: untouched machine text is
        # saved as a draft, text a person changed as a human edit. Nothing is
        # ever recorded as human work on a person's behalf.
        source = self.transcription_source
        transcription_id, revision = self.db.save_transcription(
            self.audio_file_id,
            text,
            source=source,
            speaker_count=self.speakers.registry.count or None,
        )
        self.audit.transcription_saved(self.audio_file_id, revision, text, source)

        if self._last_report is not None:
            self.db.save_validation(transcription_id, self._last_report)
            self.audit.validation_run(self.audio_file_id, self._last_report)

        for speaker_id in self.speakers.registry.speaker_ids:
            profile = self.speakers.registry.profiles[speaker_id]
            self.db.upsert_speaker(
                self.audio_file_id, profile, human_edited=profile.human_edited
            )

        note = (
            "  ·  saved as an UNREVIEWED machine draft: correct it against the "
            "audio before submitting"
            if source == "asr_draft"
            else ""
        )
        self._set_status(
            f"Saved revision {revision} as {source}. Previous revisions are kept."
            + note
        )

    def export_annotation(self) -> None:
        if self.buffer is None or self.audio_file_id is None:
            self._set_status("Open an audio file first.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Export to folder", str(self.config.paths.exports_dir)
        )
        if not directory:
            return

        annotation = Annotation(
            audio_path=str(self.buffer.path),
            duration=self.buffer.duration,
            transcription=self.editor.text().strip(),
            speakers=self.speakers.registry,
            events=self.analysis.events if self.analysis else [],
            # Provenance travels with the export: an unreviewed machine draft is
            # labelled as one, and the exporter refuses to write it until the
            # annotator says so explicitly.
            source=self.transcription_source,
            asr_model=(
                self._draft.asr.model if self._draft and self._draft.asr else ""
            ),
            validation=self._last_report,
        )
        try:
            written = Exporter().export_all(annotation, directory)
        except ExportError as exc:
            answer = QMessageBox.question(
                self,
                "Export anyway?",
                f"{exc}\n\nExport a labelled work-in-progress copy regardless?",
            )
            if answer is not QMessageBox.StandardButton.Yes:
                return
            written = Exporter().export_all(
                annotation, directory, require_valid=False, allow_unreviewed=True
            )

        for kind, path in written.items():
            self.audit.exported(self.audio_file_id, str(path), kind)
        self._set_status(f"Exported to {directory}")

    # ------------------------------------------------------------------ misc
    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            f"<b>{APP_NAME} {APP_VERSION}</b><br>"
            f"Local Ermis {ERMIS_VERSION} annotation and QA assistant.<br><br>"
            "This tool assists a human annotator. It does not transcribe audio: "
            "there is no ASR, and the final transcription is always typed and "
            "reviewed by a person.<br><br>"
            "All processing is local. No network access, no telemetry, no "
            "credentials stored.",
        )

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for attribute in ("_draft_worker", "_preload_worker"):
            worker = getattr(self, attribute, None)
            setattr(self, attribute, None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)
        # An in-progress capture is finished and kept rather than dropped: the
        # audio is the one thing that cannot be recreated afterwards.
        kept = self.record_bar.shutdown()
        if kept is not None:
            log_path = str(kept)
            if self.audio_file_id is not None:
                self.db.append_audit(
                    "recording_kept_on_exit",
                    audio_file_id=self.audio_file_id,
                    detail={"path": log_path},
                )
        self.playback.stop()
        self.db.close()
        super().closeEvent(event)


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"
