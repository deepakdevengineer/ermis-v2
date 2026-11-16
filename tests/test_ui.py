"""UI widgets (CLAUDE.md sections 47, 48, 49).

Runs headless via Qt's offscreen platform, so the suite needs no display.
Skipped entirely when PySide6 is not installed.

The tests concentrate on the behaviours that carry workflow meaning rather than
on appearance: tag insertion must produce valid text, validation must never
rewrite the editor by itself, a detection must not become a tag without a
click, and a pause must offer no tag action at all.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="UI tests need PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.audio.loader import load_audio  # noqa: E402
from app.ermis.validator import validate  # noqa: E402
from app.models.event import AnalysisEvent, ConfirmationState, EventType  # noqa: E402
from app.speakers.metadata import SpeakerRegistry  # noqa: E402
from app.storage.database import SpeakerProfile  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """Make QMessageBox non-blocking for the whole module.

    A modal dialog waits for a click even under the offscreen platform, so any
    test touching a code path that shows one would hang the suite indefinitely.
    The panels use dialogs for guidance ("select two speakers to merge") and for
    confirmations, which is correct UX; here they are stubbed out.

    `question` answers No, so a destructive confirmation is never silently
    accepted by a test that did not mean to trigger one.
    """
    from PySide6.QtWidgets import QMessageBox

    for name in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(
            QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
        )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )


@pytest.fixture
def editor(qt_app):
    from app.ui.transcription_editor import TranscriptionEditor

    return TranscriptionEditor()


def _set_caret(editor, position: int) -> None:
    cursor = editor.editor.textCursor()
    cursor.setPosition(position)
    editor.editor.setTextCursor(cursor)


# ---------------------------------------------------------------- the editor
class TestTagInsertion:
    def test_insert_at_start_spaces_both_sides(self, editor):
        editor.set_text("play music")
        _set_caret(editor, 0)
        editor.insert_tag("[hn]")
        assert editor.text() == "[hn] play music"

    def test_insert_at_end(self, editor):
        editor.set_text("play music")
        editor.insert_tag("[hn]")
        assert editor.text() == "play music [hn]"

    def test_insert_mid_text(self, editor):
        editor.set_text("play music")
        _set_caret(editor, 5)
        editor.insert_tag("[hn]")
        assert editor.text() == "play [hn] music"

    def test_insert_into_empty(self, editor):
        editor.set_text("")
        editor.insert_tag("[laughter]")
        assert editor.text() == "[laughter]"

    @pytest.mark.parametrize(
        "tag", ["[fp]", "[hn]", "[laughter]", "[bg]", "[artifact]", "<ct>"]
    )
    def test_every_inline_tag_yields_valid_text(self, editor, tag):
        editor.set_text("play music")
        editor.insert_tag(tag)
        assert validate(editor.text()).is_valid, editor.text()

    def test_unintelligible_places_the_caret_inside(self, editor):
        editor.set_text("play")
        editor.insert_tag("(())")
        editor.editor.textCursor().insertText("songs")
        assert editor.text() == "play ((songs))"

    def test_misspoken_places_the_caret_inside(self, editor):
        editor.set_text("i need")
        editor.insert_tag("{{}}")
        editor.editor.textCursor().insertText("authentication")
        assert editor.text() == "i need {{authentication}}"

    def test_word_insertion_spaces_both_sides(self, editor):
        editor.set_text("i think")
        _set_caret(editor, 0)
        editor.insert_word("um")
        assert editor.text() == "um i think"

    def test_backchannel_insertion_is_valid(self, editor):
        for spelling in ("mm-hm", "uh-huh", "hmm", "nuh-huh"):
            editor.set_text("")
            editor.insert_word(spelling)
            assert validate(editor.text()).is_valid


class TestWholeFileButtons:
    @pytest.mark.parametrize("tag", ["<ga>", "<na>", "<ns>", "<nt>", "<sc>"])
    def test_replaces_the_entire_transcription(self, editor, tag):
        # Section 42: a whole-file tag must stand alone, so the button
        # replaces rather than inserts.
        editor.set_text("play some music")
        editor.set_whole_file_tag(tag)
        assert editor.text() == tag
        assert validate(editor.text()).is_valid

    def test_replacement_is_undoable(self, editor):
        editor.set_text("play music")
        editor.set_whole_file_tag("<na>")
        editor.undo()
        assert "play music" in editor.text()


class TestEditorValidation:
    def test_validated_signal_fires(self, editor):
        received = []
        editor.validated.connect(received.append)
        editor.set_text("Play music?")
        assert received
        assert not received[-1].is_valid

    def test_editor_never_rewrites_the_text_itself(self, editor):
        # Sections 49 and 56.21: validation is non-destructive.
        original = "Play 23 songs? okay mhm"
        editor.set_text(original)
        editor._run_validation()
        assert editor.text() == original

    def test_apply_suggestion_is_explicit(self, editor):
        editor.set_text("play music?")
        report = validate(editor.text())
        finding = report.errors[0]
        # Nothing changed until the caller asks.
        assert editor.text() == "play music?"
        editor.apply_suggestion(finding.start, finding.end, finding.suggestion or "")
        assert editor.text() == "play music"

    def test_speaker_context_reaches_the_validator(self, editor):
        editor.set_speaker_context(speaker_count=1, known_speakers={1})
        received = []
        editor.validated.connect(received.append)
        editor.set_text("<s1> play music")
        from app.ermis.findings import Code

        assert received[-1].has(Code.SPEAKER_SINGLE_TAGGED)

    def test_speaker_buttons_disabled_for_one_speaker(self, editor):
        # Section 17: single-speaker audio takes no speaker tag.
        editor.update_speaker_buttons(1)
        assert not any(b.isEnabled() for b in editor._speaker_buttons)

    def test_speaker_buttons_enabled_for_two(self, editor):
        editor.update_speaker_buttons(2)
        enabled = [b.text() for b in editor._speaker_buttons if b.isEnabled()]
        assert enabled == ["<s1>", "<s2>"]

    def test_undo_redo(self, editor):
        editor.set_text("play")
        editor.insert_tag("[hn]")
        editor.undo()
        assert editor.text() == "play"
        editor.redo()
        assert "[hn]" in editor.text()


# ------------------------------------------------------- the validation panel
class TestValidationPanel:
    @pytest.fixture
    def panel(self, qt_app):
        from app.ui.validation_panel import ValidationPanel

        return ValidationPanel()

    def test_clean_report(self, panel):
        panel.set_report(validate("play music"))
        assert "No findings" in panel.summary.text()

    def test_groups_by_severity(self, panel):
        panel.set_report(validate("Play music?  dont stop"))
        labels = [
            panel.tree.topLevelItem(i).text(0)
            for i in range(panel.tree.topLevelItemCount())
        ]
        assert any(label.startswith("ERROR") for label in labels)
        assert any(label.startswith("WARNING") for label in labels)

    def test_finding_rows_cite_their_rule(self, panel):
        panel.set_report(validate("Play music?"))
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(1).startswith("section")

    def test_apply_button_starts_disabled(self, panel):
        panel.set_report(validate("Play music?"))
        assert not panel.apply_button.isEnabled()

    def test_apply_button_enables_for_a_finding_with_a_span(self, panel):
        panel.set_report(validate("Play music"))
        group = panel.tree.topLevelItem(0)
        panel.tree.setCurrentItem(group.child(0))
        assert panel.apply_button.isEnabled()

    def test_suggestion_signal_carries_the_replacement(self, panel):
        received = []
        panel.suggestionApplied.connect(lambda *args: received.append(args))
        panel.set_report(validate("Play music"))
        group = panel.tree.topLevelItem(0)
        panel.tree.setCurrentItem(group.child(0))
        panel._apply_selected()
        assert received
        start, end, replacement = received[0]
        assert replacement == "p"

    def test_empty_text(self, panel):
        panel.set_report(validate(""))
        assert panel.tree.topLevelItemCount() >= 0

    def test_suggestion_column_is_not_clipped(self, panel):
        # The suggestion is the actionable part of a finding. Fixed column
        # widths clipped it to "mm-..." instead of "mm-hm".
        from PySide6.QtWidgets import QHeaderView

        header = panel.tree.header()
        assert header.sectionResizeMode(2) is QHeaderView.ResizeMode.ResizeToContents
        panel.set_report(validate("mhm"))
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(2) == "mm-hm"


class TestPauseInfoPanel:
    @pytest.fixture
    def panel(self, qt_app):
        from app.ui.validation_panel import PauseInfoPanel

        return PauseInfoPanel()

    def test_no_pauses(self, panel):
        panel.set_pauses([])
        assert "No pauses" in panel.label.text()

    def test_states_the_rule_every_time(self, panel):
        # Section 13: a pause must never read as something to fix with a tag.
        panel.set_pauses([(1.0, 2.5)])
        text = panel.label.text()
        assert "no transcription tag" in text
        assert "do not insert" in text.lower()

    def test_truncates_a_long_list(self, panel):
        panel.set_pauses([(float(i), float(i) + 0.6) for i in range(30)])
        assert "and 18 more" in panel.label.text()


# ------------------------------------------------------------ the event panel
class TestEventPanel:
    @pytest.fixture
    def panel(self, qt_app):
        from app.ui.event_panel import EventPanel

        return EventPanel()

    def _select_first_candidate(self, panel):
        from PySide6.QtCore import Qt

        for i in range(panel.tree.topLevelItemCount()):
            group = panel.tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                event = item.data(0, Qt.ItemDataRole.UserRole)
                if event is not None and event.type.is_uncertain:
                    panel.tree.setCurrentItem(item)
                    return event
        raise AssertionError("no uncertain candidate row found to select")

    def test_summary_counts_pending(self, panel):
        panel.set_events(
            [
                AnalysisEvent(EventType.SPEECH, 0.0, 1.0),
                AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.6),
            ]
        )
        assert "1 awaiting confirmation" in panel.summary.text()

    def test_detection_becomes_a_tag_only_on_confirm(self, panel):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.95)
        panel.set_events([event])
        assert event.state is ConfirmationState.PENDING

        emitted = []
        panel.tagRequested.connect(emitted.append)
        self._select_first_candidate(panel)
        panel._confirm()
        assert event.state is ConfirmationState.CONFIRMED
        assert emitted == ["[laughter]"]

    def test_change_to_overrides_the_suggestion(self, panel):
        # Section 47's "Change to hn" action.
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.95)
        panel.set_events([event])
        self._select_first_candidate(panel)
        index = panel.change_combo.findData("[hn]")
        panel._change(index)
        assert event.confirmed_tag == "[hn]"

    def test_reject_yields_no_tag(self, panel):
        event = AnalysisEvent(EventType.POSSIBLE_ARTIFACT, 1.0, 1.1, 0.7)
        panel.set_events([event])
        self._select_first_candidate(panel)
        panel._reject()
        assert event.state is ConfirmationState.REJECTED
        assert event.tag is None

    def test_pause_offers_no_tag_action(self, panel):
        # Section 13: nothing to confirm into the transcription.
        pause = AnalysisEvent(EventType.PAUSE, 1.0, 2.0)
        panel.set_events([pause])
        for i in range(panel.tree.topLevelItemCount()):
            group = panel.tree.topLevelItem(i)
            for j in range(group.childCount()):
                panel.tree.setCurrentItem(group.child(j))
        assert not panel.confirm_button.isEnabled()
        assert not panel.change_combo.isEnabled()

    def test_pause_row_is_labelled_metadata(self, panel):
        panel.set_events([AnalysisEvent(EventType.PAUSE, 1.0, 2.0)])
        group = panel.tree.topLevelItem(0)
        assert "metadata only" in group.child(0).text(3)
        assert "never a tag" in group.text(0)

    def test_measured_events_show_no_pending_state(self, panel):
        # A measured region has nothing to confirm. Showing "pending" implied
        # action was needed and contradicted the "N awaiting confirmation"
        # summary, which counts only uncertain events.
        panel.set_events([AnalysisEvent(EventType.SPEECH, 0.0, 1.0)])
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(3) == "n/a"

    def test_pending_state_only_on_uncertain_events(self, panel):
        panel.set_events(
            [
                AnalysisEvent(EventType.SPEECH, 0.0, 1.0),
                AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.6),
            ]
        )
        states = []
        for i in range(panel.tree.topLevelItemCount()):
            group = panel.tree.topLevelItem(i)
            for j in range(group.childCount()):
                states.append(group.child(j).text(3))
        assert states.count("pending") == 1

    def test_time_range_is_not_truncated(self, panel):
        # The Time column must fit a full "MM:SS.CC - MM:SS.CC" range.
        panel.set_events([AnalysisEvent(EventType.SPEECH, 0.28, 1.31)])
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(0) == "00:00.28 - 00:01.31"

    def test_group_labels_span_all_columns(self, panel):
        # Otherwise the pause group's "(metadata only -- never a tag)" reminder
        # is clipped to the narrow Time column and the section 13 safeguard
        # becomes invisible.
        panel.set_events([AnalysisEvent(EventType.PAUSE, 1.0, 2.0)])
        group = panel.tree.topLevelItem(0)
        assert group.isFirstColumnSpanned()

    def test_confidence_shown_for_candidates(self, panel):
        panel.set_events([AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.63)])
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(2) == "63%"

    def test_measured_events_are_not_given_a_percentage(self, panel):
        panel.set_events([AnalysisEvent(EventType.SPEECH, 0.0, 1.0)])
        group = panel.tree.topLevelItem(0)
        assert group.child(0).text(2) == "measured"

    def test_detail_warns_confidence_is_not_a_verdict(self, panel):
        panel.set_events([AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.6)])
        self._select_first_candidate(panel)
        assert "not a verdict" in panel.detail.text()

    def test_play_request_signal(self, panel):
        received = []
        panel.playRequested.connect(lambda a, b: received.append((a, b)))
        panel.set_events([AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.6)])
        self._select_first_candidate(panel)
        panel._play()
        assert received == [(1.0, 1.2)]


# ---------------------------------------------------------- the speaker panel
class TestSpeakerPanel:
    @pytest.fixture
    def panel(self, qt_app):
        from app.ui.speaker_panel import SpeakerPanel

        registry = SpeakerRegistry()
        for index in (1, 2):
            registry.profiles[f"s{index}"] = SpeakerProfile(
                speaker_id=f"s{index}",
                display_name=f"Speaker {index}",
                first_seen=float(index),
                last_seen=float(index) + 1,
                confidence=0.76,
            )
            registry.segments.append((float(index), float(index) + 1, f"s{index}"))
        widget = SpeakerPanel()
        widget.set_registry(registry, detected_count=2, detected_confidence=0.76)
        return widget

    def test_shows_the_total(self, panel):
        assert panel.total.text() == "Total Speakers: 2"

    def test_shows_the_detected_count_and_confidence(self, panel):
        # Section 18A.11 requires both to be visible.
        text = panel.detected.text()
        assert "speaker count = 2" in text
        assert "76%" in text
        assert "assistive only" in text

    def test_one_row_per_speaker(self, panel):
        assert panel.speaker_list.count() == 2

    def test_dropdowns_offer_only_allowed_values(self, panel):
        from app.storage.database import ALLOWED_GENDERS, ALLOWED_NATIVITY

        row = panel.speaker_list.itemWidget(panel.speaker_list.item(0))
        genders = [row.gender.itemData(i) for i in range(row.gender.count())]
        nativities = [row.nativity.itemData(i) for i in range(row.nativity.count())]
        assert genders == list(ALLOWED_GENDERS)
        assert nativities == list(ALLOWED_NATIVITY)

    def test_defaults_are_unknown(self, panel):
        row = panel.speaker_list.itemWidget(panel.speaker_list.item(0))
        assert row.gender.currentData() == "unknown"
        assert row.nativity.currentData() == "unknown"

    def test_editing_marks_human_edited(self, panel):
        row = panel.speaker_list.itemWidget(panel.speaker_list.item(0))
        row.gender.setCurrentIndex(0)  # msv
        assert panel.registry.get("s1").gender == "msv"
        assert panel.registry.get("s1").human_edited is True

    def test_metadata_signal_carries_before_and_after(self, panel):
        received = []
        panel.metadataEdited.connect(lambda *args: received.append(args))
        row = panel.speaker_list.itemWidget(panel.speaker_list.item(0))
        row.gender.setCurrentIndex(1)  # fsv
        assert received
        _, before, after = received[0]
        assert before.gender == "unknown"
        assert after.gender == "fsv"

    def test_add_speaker(self, panel):
        panel._add()
        assert panel.registry.count == 3
        assert panel.speaker_list.count() == 3

    def test_count_spinbox_forces_the_human_figure(self, panel):
        # Section 18A.11's worked example.
        panel.count_spin.setValue(1)
        assert panel.registry.count == 1

    def test_merge_requires_two_selected(self, panel):
        panel.speaker_list.item(0).setSelected(True)
        panel._merge()  # shows guidance, changes nothing
        assert panel.registry.count == 2

    def test_merge_two(self, panel):
        panel.speaker_list.item(0).setSelected(True)
        panel.speaker_list.item(1).setSelected(True)
        panel._merge()
        assert panel.registry.count == 1

    def test_split_needs_a_boundary(self, panel):
        panel.set_split_boundary(None)
        panel.speaker_list.item(0).setSelected(True)
        panel._split()
        assert panel.registry.count == 2  # unchanged

    def test_registry_changed_signal_describes_the_edit(self, panel):
        received = []
        panel.registryChanged.connect(received.append)
        panel._add()
        assert received and "added" in received[0]

    def test_merge_disabled_for_a_single_speaker(self, panel):
        panel.count_spin.setValue(1)
        assert not panel.merge_button.isEnabled()


# --------------------------------------------------------------- the record bar
class TestRecordBar:
    """Capture transport (section 45).

    Driven with `open_stream=False`, so no test opens a microphone. That is the
    same headless path tests/test_recorder.py uses.
    """

    @pytest.fixture
    def bar(self, qt_app, tmp_path):
        from app.config import AppConfig
        from app.ui.record_bar import RecordBar

        config = AppConfig()
        config.paths.recordings_dir = tmp_path / "recordings"
        config.recording.sample_rate = 16_000
        widget = RecordBar(config)
        yield widget
        if widget.recorder.is_active:
            widget.recorder.cancel()

    @staticmethod
    def _capture(bar, seconds: float = 0.5, amplitude: float = 0.4):
        """Start a headless capture and push `seconds` of constant tone."""
        import numpy as np

        path = bar.recorder.start(open_stream=False)
        bar._timer.start()
        bar._update_controls()
        frames = int(seconds * bar.recorder.sample_rate)
        bar.recorder.write_block(np.full(frames, amplitude, dtype=np.float32))
        return path

    def test_constructs_with_a_default_device_entry(self, bar):
        assert bar.device_box.count() >= 1
        assert bar.device_box.itemData(0) is None
        assert bar.record_button.text().endswith("Record")

    def test_idle_controls_are_disabled(self, bar):
        assert not bar.pause_button.isEnabled()
        assert not bar.discard_button.isEnabled()

    def test_controls_flip_while_capturing(self, bar):
        self._capture(bar)
        assert bar.record_button.text().endswith("Stop")
        assert bar.pause_button.isEnabled()
        assert bar.discard_button.isEnabled()
        # The device cannot be swapped mid-capture.
        assert not bar.device_box.isEnabled()

    def test_meter_and_clock_track_the_capture(self, bar):
        self._capture(bar, seconds=1.0, amplitude=0.5)
        bar._tick()
        assert bar.elapsed_label.text() == "00:01.0"
        assert bar.meter.value() > 0
        assert "dBFS" in bar.meter.format()

    def test_stop_emits_the_finished_path(self, bar):
        seen: list[str] = []
        bar.recordingFinished.connect(seen.append)
        path = self._capture(bar)
        bar.stop_recording()

        assert seen == [str(path)]
        assert path.is_file()
        assert bar.last_result is not None
        assert bar.last_result.duration == pytest.approx(0.5, abs=0.01)

    def test_stop_reports_the_capture_in_the_status_message(self, bar):
        messages: list[str] = []
        bar.statusMessage.connect(messages.append)
        self._capture(bar)
        bar.stop_recording()
        assert any("Recorded" in message for message in messages)

    def test_empty_capture_emits_nothing_and_writes_no_file(self, bar):
        seen: list[str] = []
        messages: list[str] = []
        bar.recordingFinished.connect(seen.append)
        bar.statusMessage.connect(messages.append)

        path = bar.recorder.start(open_stream=False)
        bar._timer.start()
        bar.stop_recording()

        assert seen == []
        assert not path.exists()
        assert any("Nothing was captured" in message for message in messages)

    def test_pause_button_toggles_and_relabels(self, bar):
        self._capture(bar)
        bar.toggle_pause()
        assert not bar.recorder.is_recording
        assert bar.pause_button.text() == "Resume"
        bar.toggle_pause()
        assert bar.recorder.is_recording
        assert bar.pause_button.text() == "Pause"

    def test_discard_needs_confirmation(self, bar):
        # The autouse stub answers No to every question dialog, so a capture
        # must survive a discard the annotator did not confirm.
        path = self._capture(bar)
        bar.discard_recording()
        assert bar.recorder.is_active
        assert path.exists()

    def test_confirmed_discard_deletes_the_file(self, bar, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox,
            "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
        )
        path = self._capture(bar)
        bar.discard_recording()
        assert not bar.recorder.is_active
        assert not path.exists()

    def test_duration_ceiling_stops_capture_and_keeps_the_audio(self, bar):
        import numpy as np

        bar.config.recording.max_duration_seconds = 0.25
        seen: list[str] = []
        bar.recordingFinished.connect(seen.append)

        path = bar.recorder.start(open_stream=False)
        bar._timer.start()
        bar.recorder.write_block(np.full(16_000, 0.3, dtype=np.float32))
        bar._tick()  # the timer tick is where the deferred stop happens

        assert not bar.recorder.is_active
        assert seen == [str(path)]
        assert path.is_file()

    def test_shutdown_keeps_an_in_progress_capture(self, bar):
        path = self._capture(bar)
        kept = bar.shutdown()
        assert kept == path
        assert path.is_file()

    def test_shutdown_when_idle_returns_nothing(self, bar):
        assert bar.shutdown() is None

    def test_meter_maps_dbfs_onto_the_bar(self):
        from app.ui.record_bar import _meter_value

        assert _meter_value(float("-inf")) == 0
        assert _meter_value(-90.0) == 0
        assert _meter_value(-30.0) == 50
        assert _meter_value(0.0) == 100


# ------------------------------------------------------------- the main window
class TestMainWindow:
    @pytest.fixture
    def window(self, qt_app, tmp_path):
        from app.config import AppConfig
        from app.ui.main_window import MainWindow

        config = AppConfig()
        for field in (
            "data_dir",
            "projects_dir",
            "models_dir",
            "exports_dir",
            "log_dir",
            "recordings_dir",
        ):
            setattr(config.paths, field, tmp_path / field)
        config.paths.database_path = tmp_path / "test.sqlite3"
        # Automatic drafting is off in the fixture: loading a speech model costs
        # tens of seconds, and the drafting behaviour is tested with a stub in
        # TestDrafting instead of by running a real recogniser in every test.
        config.asr.enabled = False
        config.asr.auto_draft_on_load = False
        widget = MainWindow(config)
        yield widget
        widget.close()

    def test_constructs(self, window):
        assert window.windowTitle().startswith("ermis-local-assistant")

    def test_loads_audio_and_runs_analysis(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        assert window.buffer is not None
        assert window.analysis is not None
        assert window.analysis.events

    def test_speaker_panel_is_populated(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        assert window.speakers.registry.count == 2

    def test_waveform_receives_the_audio(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        assert window.waveform.duration > 0

    def test_pause_panel_is_populated(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        assert "pause" in window.pauses.label.text().lower()

    def test_analysis_does_not_touch_the_transcription(self, window, wav_two_speakers):
        # Section 3: Layer A never writes Layer B.
        window.load_audio(wav_two_speakers)
        assert window.editor.text() == ""

    def test_save_creates_a_revision_and_audit_rows(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.editor.set_text("<s1> play music <s2> ok")
        window.save_transcription()
        latest = window.db.latest_transcription(window.audio_file_id)
        assert latest["text"] == "<s1> play music <s2> ok"
        assert latest["source"] == "human"
        actions = [r["action"] for r in window.db.list_audit(window.audio_file_id)]
        assert "transcription_saved" in actions
        assert "validation_run" in actions

    def test_reopening_restores_the_draft(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.editor.set_text("play music")
        window.save_transcription()
        window.load_audio(wav_two_speakers)
        assert window.editor.text() == "play music"

    def test_confirming_an_event_inserts_its_tag(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.editor.set_text("play music")
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.8)
        window.events.set_events([event])
        for i in range(window.events.tree.topLevelItemCount()):
            group = window.events.tree.topLevelItem(i)
            if group.childCount():
                window.events.tree.setCurrentItem(group.child(0))
        window.events._confirm()
        assert "[laughter]" in window.editor.text()

    # ------------------------------------------- the analysis description
    def test_analysis_description_reaches_the_status_bar(
        self, window, wav_two_speakers
    ):
        window.load_audio(wav_two_speakers)
        status = window.statusBar().currentMessage()
        assert "Analysis complete" in status
        assert "speech region(s)" in status
        assert "speaker(s) suggested" in status

    def test_recording_keeps_the_analysis_description(self, window, wav_two_speakers):
        """Regression: recording used to overwrite what the analysis found.

        `load_audio` runs the analysis and describes it, then the recording
        handler set a status message of its own -- so an annotator who recorded
        a clip never saw the description an annotator who opened one did.
        """
        window._on_recording_finished(str(wav_two_speakers))
        status = window.statusBar().currentMessage()
        assert "Recorded" in status, status
        assert "Analysis complete" in status, status
        assert "speech region(s)" in status, status

    def test_recording_status_still_names_the_file_and_duration(
        self, window, wav_two_speakers
    ):
        window._on_recording_finished(str(wav_two_speakers))
        status = window.statusBar().currentMessage()
        assert wav_two_speakers.name in status
        assert status.startswith("Recorded ")
        # The two halves are joined, not one in place of the other.
        assert status.index("Recorded") < status.index("Analysis complete")

    def test_a_new_clip_does_not_inherit_the_previous_description(
        self, window, wav_two_speakers, wav_silence
    ):
        window.load_audio(wav_two_speakers)
        two_speaker_summary = window._analysis_summary
        window.load_audio(wav_silence)
        assert window._analysis_summary != two_speaker_summary
        assert window._analysis_summary  # the new clip has one of its own

    def test_describe_analysis_reports_whole_file_candidates(
        self, window, wav_silence
    ):
        window.load_audio(wav_silence)
        # A blank clip is the <na> case (section 11); the description is where
        # the annotator is told so.
        assert "whole-file candidate(s)" in window._analysis_summary

    def test_export_writes_files(self, window, wav_two_speakers, tmp_path):
        from app.storage.export import Annotation, Exporter

        window.load_audio(wav_two_speakers)
        window.editor.set_text("<s1> play music <s2> ok")
        annotation = Annotation(
            audio_path=str(window.buffer.path),
            duration=window.buffer.duration,
            transcription=window.editor.text(),
            speakers=window.speakers.registry,
            events=window.analysis.events,
            validation=window._last_report,
        )
        written = Exporter().export_all(annotation, tmp_path / "out", require_valid=False)
        assert all(path.is_file() for path in written.values())

    def test_silent_audio_is_flagged_in_the_status_bar(self, window, wav_silence):
        window.load_audio(wav_silence)
        assert "<na>" in window.statusBar().currentMessage()

    # ------------------------------------------------------------- recording
    @staticmethod
    def _record(window, seconds: float = 1.0):
        """Capture headlessly through the window's own record bar."""
        import numpy as np

        from tests.conftest import VOICE_A, speech

        bar = window.record_bar
        bar.config.recording.sample_rate = 16_000
        bar.recorder.recording.sample_rate = 16_000
        bar.recorder.start(open_stream=False)
        signal = speech(seconds, sample_rate=16_000, **VOICE_A)
        for start in range(0, signal.size, 1024):
            bar.recorder.write_block(np.ascontiguousarray(signal[start : start + 1024]))
        bar.stop_recording()

    def test_recording_is_loaded_and_analysed_like_any_other_clip(self, window):
        self._record(window)
        assert window.buffer is not None
        assert window.buffer.path.parent == window.config.paths.recordings_dir
        assert window.analysis is not None

    def test_recording_does_not_write_the_transcription(self, window):
        # Section 3 again, on the capture path: audio in, no text.
        self._record(window)
        assert window.editor.text() == ""

    def test_recording_provenance_is_audited(self, window):
        self._record(window)
        actions = [r["action"] for r in window.db.list_audit(window.audio_file_id)]
        assert "recording_captured" in actions
        assert "audio_loaded" in actions

    def test_status_bar_says_the_annotator_types_the_transcription(self, window):
        self._record(window)
        message = window.statusBar().currentMessage()
        assert "does not transcribe" in message

    def test_status_bar_calls_a_draft_machine_output_when_drafting_is_on(
        self, window
    ):
        # The other sentence would be a lie in this configuration: with
        # drafting enabled the tool does put words in the editor, and section
        # 60.2 requires them to be labelled as machine output.
        window.config.asr.enabled = True
        self._record(window)
        message = window.statusBar().currentMessage()
        assert "machine output" in message
        assert "does not transcribe" not in message

    def test_recording_status_keeps_the_analysis_description_either_way(
        self, window
    ):
        for enabled in (False, True):
            window.config.asr.enabled = enabled
            self._record(window)
            assert "Analysis complete" in window.statusBar().currentMessage()

    # -------------------------------------------------------------- drafting
    @staticmethod
    def _stub_draft(text="hello how are you", events=None, unclear=0):
        """A DraftResult as the worker would deliver it, with no model loaded."""
        from app.asr.draft import DraftResult, DraftWord
        from app.asr.engine import AsrResult, AsrWord
        from app.ermis.validator import validate

        pieces = text.split()
        words = [
            DraftWord(word, index * 0.4, index * 0.4 + 0.3, 0.9)
            for index, word in enumerate(pieces)
        ]
        asr = AsrResult(
            [AsrWord(w.text, w.start, w.end, w.confidence) for w in words],
            backend="stub",
            model="openai/whisper-small.en",
        )
        return DraftResult(
            text=text,
            words=words,
            speaker_count=1,
            used_events=list(events or []),
            asr=asr,
            report=validate(text),
        )

    def test_applying_a_draft_fills_the_editor(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.apply_draft(self._stub_draft())
        assert window.editor.text() == "hello how are you"

    def test_a_draft_is_labelled_in_the_window_title(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.apply_draft(self._stub_draft())
        assert "UNREVIEWED MACHINE DRAFT" in window.windowTitle()

    def test_draft_text_saves_under_the_draft_source(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.apply_draft(self._stub_draft())
        window.save_transcription()
        assert window.db.latest_transcription(window.audio_file_id)["source"] == (
            "asr_draft"
        )

    def test_editing_a_draft_makes_it_human_edited(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.apply_draft(self._stub_draft())
        # A keystroke is what turns machine text into reviewed text.
        window.editor.textEdited.emit("hello how are ya")
        assert window.transcription_source == "human_edited"
        assert "UNREVIEWED" not in window.windowTitle()

    def test_human_text_is_never_overwritten_without_consent(
        self, window, wav_two_speakers
    ):
        # The autouse dialog stub answers No, standing in for an annotator who
        # declines (section 56.21).
        window.load_audio(wav_two_speakers)
        window.editor.set_text("play music")
        window._on_drafted(self._stub_draft())
        assert window.editor.text() == "play music"
        assert window._draft is not None

    def test_draft_is_recorded_in_the_audit_trail(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window._on_drafted(self._stub_draft())
        actions = [r["action"] for r in window.db.list_audit(window.audio_file_id)]
        assert "asr_draft_created" in actions

    def test_tags_from_a_draft_are_audited_as_auto_confirmed(
        self, window, wav_two_speakers
    ):
        # Section 47 wants a human decision; when a draft supplies the tag the
        # audit trail must say so rather than claiming a person confirmed it.
        window.load_audio(wav_two_speakers)
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 1.4, 0.9)
        window.analysis.events.append(event)
        window.apply_draft(
            self._stub_draft(text="[laughter] hello there", events=[event])
        )
        actions = [r["action"] for r in window.db.list_audit(window.audio_file_id)]
        assert "auto_confirmed_event" in actions
        assert "confirmed_event" not in actions
        assert event.is_confirmed
        assert "verify against the audio" in event.note

    def test_draft_failure_tells_the_annotator_to_type(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window._on_draft_failed("no local ASR backend is installed")
        message = window.statusBar().currentMessage()
        assert "no local ASR backend" in message
        assert "yourself" in message

    def test_drafting_without_audio_is_a_no_op(self, window):
        window.start_draft()
        assert "Open or record audio first" in window.statusBar().currentMessage()

    def test_disabled_asr_says_so(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.config.asr.enabled = False
        window.start_draft()
        assert "disabled" in window.statusBar().currentMessage()

    def test_preload_is_skipped_when_asr_is_off(self, window):
        # The fixture disables ASR, so no model thread may be started: a test
        # suite must never load a gigabyte of weights.
        assert window._preload_worker is None

    def test_preload_starts_a_worker_when_asr_is_on(self, qt_app, tmp_path, monkeypatch):
        from app.config import AppConfig
        from app.ui import draft_worker
        from app.ui.main_window import MainWindow

        started: list[str] = []
        monkeypatch.setattr(
            draft_worker.ModelPreloadWorker, "start", lambda self: started.append("go")
        )

        config = AppConfig()
        for field in (
            "data_dir", "projects_dir", "models_dir", "exports_dir", "log_dir",
            "recordings_dir",
        ):
            setattr(config.paths, field, tmp_path / field)
        config.paths.database_path = tmp_path / "preload.sqlite3"
        window = MainWindow(config)
        try:
            assert started == ["go"]
            assert "background" in window.statusBar().currentMessage()
        finally:
            window._preload_worker = None
            window.close()

    def test_auto_draft_toggle_updates_the_config(self, window):
        window.auto_draft_action.setChecked(True)
        assert window.config.asr.auto_draft_on_load is True
        window.auto_draft_action.setChecked(False)
        assert window.config.asr.auto_draft_on_load is False

    def test_export_labels_an_unreviewed_draft(self, window, wav_two_speakers, tmp_path):
        from app.storage.export import Annotation, ExportError, Exporter

        window.load_audio(wav_two_speakers)
        window.apply_draft(self._stub_draft())
        annotation = Annotation(
            audio_path=str(window.buffer.path),
            duration=window.buffer.duration,
            transcription=window.editor.text(),
            speakers=window.speakers.registry,
            events=window.analysis.events,
            source=window.transcription_source,
            asr_model="openai/whisper-small.en",
        )
        with pytest.raises(ExportError, match="unreviewed machine draft"):
            Exporter().export_json(annotation, tmp_path / "out.json")
        path = Exporter().export_json(
            annotation, tmp_path / "out.json", allow_unreviewed=True
        )
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["transcription"]["asr_used"] is True
        assert payload["transcription"]["human_reviewed"] is False

    def test_speaker_gender_is_suggested_from_the_audio(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        # Suggestions are attached to the analysis run, with or without ASR.
        assert window._voice_suggestions
        for suggestion in window._voice_suggestions.values():
            assert suggestion.gender in ("msv", "fsv", "unknown")
            assert suggestion.nativity == "unknown"

    def test_starting_a_recording_stops_playback(self, window, wav_two_speakers):
        window.load_audio(wav_two_speakers)
        window.playback.play()
        window.record_bar.recordingStarted.emit()
        assert not window.playback.is_playing
