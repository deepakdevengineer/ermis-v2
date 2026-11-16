"""SQLite schema, storage invariants, and the audit trail.

CLAUDE.md sections 50 (audit log), 52 (database), 53 (privacy).
"""

from __future__ import annotations

import sqlite3

import pytest

from app.ermis.validator import validate
from app.models.event import AnalysisEvent, ConfirmationState, EventType
from app.storage.audit import Action, AuditLog
from app.storage.database import Database, SpeakerProfile


class TestSchema:
    def test_expected_tables_exist(self, db):
        names = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "projects", "audio_files", "speakers", "events",
            "transcriptions", "validation_results", "audit_logs",
        } <= names

    def test_schema_version_recorded(self, db):
        # Version 2 widened transcriptions.source to carry 'asr_draft'.
        assert db.schema_version == 2

    def test_ermis_version_recorded(self, db):
        row = db.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'ermis_version'"
        ).fetchone()
        assert row["value"] == "V2"

    def test_foreign_keys_enforced(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO audio_files (project_id, path, filename, duration, "
                "sample_rate, channels) VALUES (999, 'x', 'x', 1.0, 16000, 1)"
            )

    def test_cascade_delete(self, db):
        project_id = db.create_project("p")
        audio_id = db.add_audio_file(project_id, "a.wav", 1.0, 16000, 1)
        db.save_transcription(audio_id, "text")
        db.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        db.conn.commit()
        assert db.latest_transcription(audio_id) is None

    def test_latest_transcriptions_view(self, db, audio_row):
        db.save_transcription(audio_row, "first")
        db.save_transcription(audio_row, "second")
        rows = db.conn.execute("SELECT * FROM latest_transcriptions").fetchall()
        assert len(rows) == 1
        assert rows[0]["text"] == "second"


class TestSchemaConstraints:
    def test_invalid_gender_rejected_at_the_database_level(self, db, audio_row):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO speakers (audio_file_id, speaker_id, display_name, gender) "
                "VALUES (?, 's1', 'Speaker 1', 'male')",
                (audio_row,),
            )

    def test_invalid_nativity_rejected_at_the_database_level(self, db, audio_row):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO speakers (audio_file_id, speaker_id, display_name, nativity) "
                "VALUES (?, 's1', 'Speaker 1', 'fluent')",
                (audio_row,),
            )

    def test_invalid_event_type_rejected(self, db, audio_row):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO events (audio_file_id, type, start_time, end_time) "
                "VALUES (?, 'asr_transcript', 0, 1)",
                (audio_row,),
            )

    def test_end_before_start_rejected(self, db, audio_row):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO events (audio_file_id, type, start_time, end_time) "
                "VALUES (?, 'speech', 5, 1)",
                (audio_row,),
            )

    def test_confidence_range_enforced(self, db, audio_row):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO events (audio_file_id, type, start_time, end_time, confidence) "
                "VALUES (?, 'speech', 0, 1, 1.5)",
                (audio_row,),
            )

    def test_pause_tag_rejected_at_the_database_level(self, db, audio_row):
        # Section 13 enforced in the schema, not only in Python.
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO events (audio_file_id, type, start_time, end_time, "
                "state, confirmed_tag) VALUES (?, 'pause', 0, 1, 'confirmed', '[pause]')",
                (audio_row,),
            )

    def test_tag_on_an_unconfirmed_event_rejected(self, db, audio_row):
        # Section 47 enforced in the schema.
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO events (audio_file_id, type, start_time, end_time, "
                "state, confirmed_tag) VALUES (?, 'possible_laughter', 0, 1, "
                "'pending', '[laughter]')",
                (audio_row,),
            )

    def test_unlabelled_machine_source_rejected_at_the_database_level(
        self, db, audio_row
    ):
        """Machine text may only be stored under the 'asr_draft' label.

        The CHECK constraint is what stops a caller inventing its own
        provenance word and blurring the line between drafted and reviewed
        text.
        """
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO transcriptions (audio_file_id, revision, text, source) "
                "VALUES (?, 1, 'text', 'asr')",
                (audio_row,),
            )

    def test_asr_draft_source_accepted(self, db, audio_row):
        db.conn.execute(
            "INSERT INTO transcriptions (audio_file_id, revision, text, source) "
            "VALUES (?, 1, 'hello how are you', 'asr_draft')",
            (audio_row,),
        )
        row = db.latest_transcription(audio_row)
        assert row["source"] == "asr_draft"

    def test_no_credential_columns_anywhere(self, db):
        """The application stores no secrets, so no column may look like one."""
        forbidden = ("password", "passwd", "secret", "token", "api_key", "credential")
        tables = [
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        for table in tables:
            for column in db.conn.execute(f"PRAGMA table_info({table})"):
                assert not any(bad in column["name"].lower() for bad in forbidden), (
                    f"{table}.{column['name']} looks like a credential column"
                )


class TestTranscriptionRevisions:
    def test_revisions_increment(self, db, audio_row):
        _, first = db.save_transcription(audio_row, "one")
        _, second = db.save_transcription(audio_row, "two")
        assert (first, second) == (1, 2)

    def test_history_is_append_only(self, db, audio_row):
        db.save_transcription(audio_row, "human original")
        db.save_transcription(audio_row, "human revised")
        texts = [row["text"] for row in db.transcription_history(audio_row)]
        assert texts == ["human original", "human revised"]

    def test_latest_returns_the_newest(self, db, audio_row):
        db.save_transcription(audio_row, "old")
        db.save_transcription(audio_row, "new")
        assert db.latest_transcription(audio_row)["text"] == "new"

    @pytest.mark.parametrize(
        "source", ["human", "human_edited", "imported", "asr_draft"]
    )
    def test_allowed_sources(self, db, audio_row, source):
        db.save_transcription(audio_row, "text", source=source)

    @pytest.mark.parametrize("source", ["asr", "whisper", "auto", "model"])
    def test_disallowed_sources(self, db, audio_row, source):
        # Machine text has exactly one label: 'asr_draft'. Any other spelling
        # would let provenance drift.
        with pytest.raises(ValueError):
            db.save_transcription(audio_row, "text", source=source)

    def test_a_draft_and_its_review_are_separate_revisions(self, db, audio_row):
        db.save_transcription(audio_row, "hello how are you", source="asr_draft")
        db.save_transcription(audio_row, "hello how are ya", source="human_edited")
        history = db.transcription_history(audio_row)
        assert [row["source"] for row in history] == ["asr_draft", "human_edited"]


class TestEventStorage:
    def test_round_trip(self, db, audio_row):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 7.42, 8.15, 0.91, "s1")
        db.add_event(audio_row, event)
        stored = db.list_events(audio_row)
        assert len(stored) == 1
        assert stored[0].type is EventType.POSSIBLE_LAUGHTER
        assert stored[0].start == pytest.approx(7.42)
        assert stored[0].confidence == pytest.approx(0.91)
        assert stored[0].speaker_id == "s1"
        assert stored[0].state is ConfirmationState.PENDING

    def test_events_ordered_by_time(self, db, audio_row):
        for start in (5.0, 1.0, 3.0):
            db.add_event(audio_row, AnalysisEvent(EventType.SPEECH, start, start + 0.5))
        assert [e.start for e in db.list_events(audio_row)] == [1.0, 3.0, 5.0]

    def test_filter_by_state(self, db, audio_row):
        db.add_event(audio_row, AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2, 0.6))
        confirmed = AnalysisEvent(EventType.POSSIBLE_HN, 2.0, 2.2, 0.9)
        confirmed.confirm()
        db.add_event(audio_row, confirmed)
        pending = db.list_events(audio_row, ConfirmationState.PENDING)
        assert len(pending) == 1
        assert pending[0].start == pytest.approx(1.0)

    def test_set_state_rejects_a_tag_on_a_rejection(self, db, audio_row):
        row_id = db.add_event(audio_row, AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.2))
        with pytest.raises(ValueError, match="confirmed"):
            db.set_event_state(row_id, ConfirmationState.REJECTED, "[hn]")

    def test_pause_event_is_stored_but_carries_no_tag(self, db, audio_row):
        db.add_event(audio_row, AnalysisEvent(EventType.PAUSE, 3.42, 5.18))
        stored = db.list_events(audio_row)[0]
        assert stored.type is EventType.PAUSE
        assert stored.confirmed_tag is None


class TestValidationStorage:
    def test_findings_persisted(self, db, audio_row):
        transcription_id, _ = db.save_transcription(audio_row, "Play 2 songs?")
        report = validate("Play 2 songs?")
        count = db.save_validation(transcription_id, report)
        assert count == len(report.findings)
        rows = db.list_validation(transcription_id)
        assert {row["severity"] for row in rows} <= {"ERROR", "WARNING", "INFO"}

    def test_clean_report_persists_nothing(self, db, audio_row):
        transcription_id, _ = db.save_transcription(audio_row, "play music")
        assert db.save_validation(transcription_id, validate("play music")) == 0

    def test_rule_reference_is_kept(self, db, audio_row):
        transcription_id, _ = db.save_transcription(audio_row, "play music?")
        db.save_validation(transcription_id, validate("play music?"))
        rows = db.list_validation(transcription_id)
        assert all(row["rule_ref"].startswith("section") for row in rows)


class TestAuditLog:
    def test_append_only_blocks_update(self, db, audio_row):
        db.append_audit("test", audio_file_id=audio_row)
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute("UPDATE audit_logs SET action = 'tampered'")

    def test_append_only_blocks_delete(self, db, audio_row):
        db.append_audit("test", audio_file_id=audio_row)
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute("DELETE FROM audit_logs")

    def test_confirmation_is_recorded(self, db, audio_row):
        log = AuditLog(db)
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 7.42, 8.15, 0.91)
        event.confirm()
        log.event_confirmed(audio_row, event)
        row = db.list_audit(audio_row)[0]
        assert row["action"] == Action.EVENT_CONFIRMED
        assert row["event_type"] == "possible_laughter"
        assert row["audio_start"] == pytest.approx(7.42)
        assert row["audio_end"] == pytest.approx(8.15)

    def test_override_of_a_suggestion_is_visible(self, db, audio_row):
        import json

        log = AuditLog(db)
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.95)
        event.confirm("[hn]")  # human disagreed with the detector
        log.event_confirmed(audio_row, event)
        detail = json.loads(db.list_audit(audio_row)[0]["detail"])
        assert detail["confirmed_tag"] == "[hn]"
        assert detail["proposed_tag"] == "[laughter]"
        assert detail["overrode_suggestion"] is True

    def test_rejection_is_recorded(self, db, audio_row):
        log = AuditLog(db)
        event = AnalysisEvent(EventType.POSSIBLE_ARTIFACT, 2.0, 2.1, 0.7)
        event.reject("door closing")
        log.event_rejected(audio_row, event)
        assert db.list_audit(audio_row)[0]["action"] == Action.EVENT_REJECTED

    def test_speaker_metadata_change_is_recorded(self, db, audio_row):
        import json

        log = AuditLog(db)
        before = SpeakerProfile("s1", "Speaker 1")
        after = SpeakerProfile("s1", "Speaker 1", gender="msv", nativity="native")
        log.speaker_metadata_changed(audio_row, before, after)
        detail = json.loads(db.list_audit(audio_row)[0]["detail"])
        assert detail["before"]["gender"] == "unknown"
        assert detail["after"]["gender"] == "msv"

    def test_validation_run_is_recorded(self, db, audio_row):
        import json

        log = AuditLog(db)
        log.validation_run(audio_row, validate("Play music?"))
        detail = json.loads(db.list_audit(audio_row)[0]["detail"])
        assert detail["errors"] >= 1
        assert "uppercase" in detail["codes"]

    def test_merge_and_split_are_recorded(self, db, audio_row):
        log = AuditLog(db)
        log.speakers_merged(audio_row, ["s2"], "s1")
        log.speaker_split(audio_row, "s1", ["s1", "s2"])
        actions = [row["action"] for row in db.list_audit(audio_row)]
        assert actions == [Action.SPEAKERS_MERGED, Action.SPEAKER_SPLIT]

    def test_ordering_is_chronological(self, db, audio_row):
        log = AuditLog(db)
        log.audio_loaded(audio_row, "a.wav", 1.0)
        log.validation_run(audio_row, validate("play music"))
        log.exported(audio_row, "out.json", "annotation")
        actions = [row["action"] for row in db.list_audit(audio_row)]
        assert actions == [Action.AUDIO_LOADED, Action.VALIDATION_RUN, Action.EXPORTED]


class TestFileHash:
    def test_hash_is_stable(self, wav_tone):
        assert Database.file_hash(wav_tone) == Database.file_hash(wav_tone)

    def test_hash_differs_for_different_content(self, wav_tone, wav_silence):
        assert Database.file_hash(wav_tone) != Database.file_hash(wav_silence)

    def test_hash_stored_with_the_audio_file(self, db, wav_tone):
        project_id = db.create_project("p")
        digest = Database.file_hash(wav_tone)
        audio_id = db.add_audio_file(
            project_id, wav_tone, 1.0, 16000, 1, content_hash=digest
        )
        assert db.get_audio_file(audio_id)["content_hash"] == digest


class TestPersistence:
    def test_database_survives_reopen(self, tmp_path):
        path = tmp_path / "ermis.sqlite3"
        with Database(path) as db:
            project_id = db.create_project("persisted")
            audio_id = db.add_audio_file(project_id, "a.wav", 2.0, 16000, 1)
            db.save_transcription(audio_id, "play music")
        with Database(path) as db:
            assert [row["name"] for row in db.list_projects()] == ["persisted"]
            assert db.latest_transcription(1)["text"] == "play music"

    def test_duplicate_project_name_rejected(self, db):
        db.create_project("dup")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_project("dup")

    def test_duplicate_audio_path_rejected(self, db):
        project_id = db.create_project("p")
        db.add_audio_file(project_id, "a.wav", 1.0, 16000, 1)
        with pytest.raises(sqlite3.IntegrityError):
            db.add_audio_file(project_id, "a.wav", 1.0, 16000, 1)
