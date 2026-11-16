-- Ermis V2 Local Audio Annotation & QA Assistant -- SQLite schema
-- CLAUDE.md section 52.
--
-- Local file only. No credentials, no tokens, no user accounts: the
-- application authenticates against nothing, so there is nothing to store.
-- Do not add a column that holds a secret.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- projects
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    notes       TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- audio_files
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audio_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path            TEXT    NOT NULL,
    filename        TEXT    NOT NULL,
    -- sha256 of the file contents, so a re-import is recognized and analysis
    -- results stay attached to the right audio.
    content_hash    TEXT,
    duration        REAL    NOT NULL,
    sample_rate     INTEGER NOT NULL,
    channels        INTEGER NOT NULL,
    -- Analysis-time facts recorded by the loader.
    peak_dbfs       REAL,
    rms_dbfs        REAL,
    is_silent       INTEGER NOT NULL DEFAULT 0 CHECK (is_silent IN (0, 1)),
    imported_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (project_id, path)
);

CREATE INDEX IF NOT EXISTS idx_audio_files_project ON audio_files(project_id);
CREATE INDEX IF NOT EXISTS idx_audio_files_hash    ON audio_files(content_hash);

-- ---------------------------------------------------------------------------
-- speakers  (CLAUDE.md section 18A.5)
--
-- gender and nativity are constrained to the allowed vocabularies from
-- sections 18A.3, 18A.4 and 18A.10. No other value is accepted, at any layer.
--
-- `human_edited` records that a person changed the automatic suggestion, so
-- nothing may silently overwrite it (section 18A.11).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS speakers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id   INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    speaker_id      TEXT    NOT NULL,   -- 's1', 's2', ...
    display_name    TEXT    NOT NULL,   -- 'Speaker 1'
    gender          TEXT    NOT NULL DEFAULT 'unknown'
                        CHECK (gender IN ('msv', 'fsv', 'unknown')),
    nativity        TEXT    NOT NULL DEFAULT 'unknown'
                        CHECK (nativity IN ('native', 'nonnative', 'unknown', 'tts')),
    first_seen      REAL,
    last_seen       REAL,
    confidence      REAL    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    human_edited    INTEGER NOT NULL DEFAULT 0 CHECK (human_edited IN (0, 1)),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (audio_file_id, speaker_id)
);

CREATE INDEX IF NOT EXISTS idx_speakers_audio ON speakers(audio_file_id);

-- ---------------------------------------------------------------------------
-- events  (CLAUDE.md sections 46, 52)
--
-- `state` starts at 'pending'. A 'possible_*' event may only contribute a tag
-- to the transcription once state = 'confirmed' (section 47).
--
-- The CHECK on confirmed_tag is the database-level half of the rule that a
-- pause never becomes a transcription tag (section 13).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id   INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    type            TEXT    NOT NULL CHECK (type IN (
                        'speech', 'pause', 'silence',
                        'possible_laughter', 'possible_hn', 'possible_artifact',
                        'possible_background', 'possible_overlap',
                        'speaker_change', 'garbled_audio', 'no_audio', 'no_speech')),
    start_time      REAL    NOT NULL CHECK (start_time >= 0),
    end_time        REAL    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0
                        CHECK (confidence >= 0 AND confidence <= 1),
    speaker_id      TEXT,
    state           TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (state IN ('pending', 'confirmed', 'rejected')),
    confirmed_tag   TEXT    CHECK (confirmed_tag IS NULL OR confirmed_tag NOT IN
                        ('[pause]', '<pause>', '[silence]', '<silence>')),
    note            TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (end_time >= start_time),
    -- A tag may only exist on a confirmed event.
    CHECK (confirmed_tag IS NULL OR state = 'confirmed')
);

CREATE INDEX IF NOT EXISTS idx_events_audio ON events(audio_file_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_state ON events(state);

-- ---------------------------------------------------------------------------
-- transcriptions
--
-- Revision history is append-only: each save inserts a new revision, so an
-- automated process can never destroy a human's text (section 56.21).
--
-- `source` records the provenance of the text and is constrained to a closed
-- vocabulary:
--
--   human        typed by the annotator
--   human_edited a draft the annotator changed
--   imported     text brought in from elsewhere
--   asr_draft    produced by the local recogniser and NOT yet reviewed
--
-- 'asr_draft' exists because the project owner asked for the words to be
-- drafted from the audio, a documented deviation from sections 1 and 56.1. It
-- is a label, not a loophole: machine text is stored as machine text so a
-- reviewer can always tell the two apart, and a saved draft becomes
-- 'human_edited' only once a person has actually changed it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transcriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id   INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    revision        INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'human'
                        CHECK (source IN ('human', 'human_edited', 'imported', 'asr_draft')),
    speaker_count   INTEGER,
    is_final        INTEGER NOT NULL DEFAULT 0 CHECK (is_final IN (0, 1)),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (audio_file_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_transcriptions_audio
    ON transcriptions(audio_file_id, revision DESC);

-- ---------------------------------------------------------------------------
-- validation_results  (CLAUDE.md section 44)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS validation_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id    INTEGER NOT NULL REFERENCES transcriptions(id) ON DELETE CASCADE,
    code                TEXT    NOT NULL,
    severity            TEXT    NOT NULL CHECK (severity IN ('ERROR', 'WARNING', 'INFO')),
    message             TEXT    NOT NULL,
    rule_ref            TEXT    NOT NULL DEFAULT '',
    start_offset        INTEGER NOT NULL DEFAULT 0,
    end_offset          INTEGER NOT NULL DEFAULT 0,
    excerpt             TEXT    NOT NULL DEFAULT '',
    suggestion          TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_validation_transcription
    ON validation_results(transcription_id, severity);

-- ---------------------------------------------------------------------------
-- audit_logs  (CLAUDE.md section 50)
--
-- Append-only. Triggers below block UPDATE and DELETE so the trail cannot be
-- rewritten from application code.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id   INTEGER REFERENCES audio_files(id) ON DELETE SET NULL,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    action          TEXT    NOT NULL,
    event_type      TEXT,
    audio_start     REAL,
    audio_end       REAL,
    -- JSON blob of action-specific detail. Never contains credentials.
    detail          TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_audio ON audit_logs(audio_file_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);

CREATE TRIGGER IF NOT EXISTS audit_logs_no_update
BEFORE UPDATE ON audit_logs
BEGIN
    SELECT RAISE(ABORT, 'audit_logs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete
BEFORE DELETE ON audit_logs
BEGIN
    SELECT RAISE(ABORT, 'audit_logs is append-only');
END;

-- ---------------------------------------------------------------------------
-- schema_meta
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '2');
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('ermis_version', 'V2');

-- ---------------------------------------------------------------------------
-- Convenience view: the newest transcription revision per audio file.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS latest_transcriptions AS
SELECT t.*
FROM transcriptions t
JOIN (
    SELECT audio_file_id, MAX(revision) AS revision
    FROM transcriptions
    GROUP BY audio_file_id
) m ON m.audio_file_id = t.audio_file_id AND m.revision = t.revision;
