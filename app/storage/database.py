"""Local SQLite storage (CLAUDE.md sections 52, 53).

Everything lives in one local file. No network, no credentials.

Two invariants are enforced here as well as in the schema, because they are
workflow rules rather than data-shape rules:

  - Transcription saves are append-only revisions, so nothing can destroy a
    human's text (section 56.21).
  - An event's tag only reaches the transcription after human confirmation
    (section 47).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.ermis.findings import ValidationReport
from app.models.event import AnalysisEvent, ConfirmationState, EventType

log = logging.getLogger("ermis.database")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Current schema version. Bump it together with a migration in `_migrate`.
_SCHEMA_VERSION = 2

#: Provenance values a transcription revision may carry.
#:
#: 'asr_draft' is machine text that nobody has reviewed yet. It is stored under
#: its own label rather than as human work, so the difference survives into the
#: database, the export and the audit trail.
ALLOWED_TRANSCRIPTION_SOURCES: tuple[str, ...] = (
    "human",
    "human_edited",
    "imported",
    "asr_draft",
)

#: Sources that represent a human decision about the text.
HUMAN_TRANSCRIPTION_SOURCES: tuple[str, ...] = ("human", "human_edited", "imported")

ALLOWED_GENDERS = ("msv", "fsv", "unknown")
ALLOWED_NATIVITY = ("native", "nonnative", "unknown", "tts")


@dataclass(frozen=True)
class SpeakerProfile:
    """Speaker metadata (CLAUDE.md section 18A.5)."""

    speaker_id: str
    display_name: str
    gender: str = "unknown"
    nativity: str = "unknown"
    first_seen: float | None = None
    last_seen: float | None = None
    confidence: float | None = None
    human_edited: bool = False

    def __post_init__(self) -> None:
        if self.gender not in ALLOWED_GENDERS:
            raise ValueError(
                f"gender {self.gender!r} invalid; allowed: {ALLOWED_GENDERS} "
                "(CLAUDE.md section 18A.3)"
            )
        if self.nativity not in ALLOWED_NATIVITY:
            raise ValueError(
                f"nativity {self.nativity!r} invalid; allowed: {ALLOWED_NATIVITY} "
                "(CLAUDE.md section 18A.4)"
            )

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "gender": self.gender,
            "nativity": self.nativity,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
        }


class Database:
    """Thin, explicit data-access layer over the local SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema version.

        `CREATE TABLE IF NOT EXISTS` leaves an existing table untouched, so a
        constraint change needs an explicit migration or an old file would
        silently keep the old CHECK.
        """
        version = self.schema_version
        if version >= _SCHEMA_VERSION:
            return
        if version < 2:
            self._migrate_to_2()
        self.conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (str(_SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate_to_2(self) -> None:
        """Widen transcriptions.source to include 'asr_draft'."""
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='transcriptions'"
        ).fetchone()
        if row is None or "asr_draft" in (row["sql"] or ""):
            return
        log.info("migrating transcriptions.source to schema version 2")
        with self.transaction() as conn:
            conn.execute("ALTER TABLE transcriptions RENAME TO transcriptions_v1")
            conn.executescript(
                """
                CREATE TABLE transcriptions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_file_id   INTEGER NOT NULL
                                        REFERENCES audio_files(id) ON DELETE CASCADE,
                    revision        INTEGER NOT NULL,
                    text            TEXT    NOT NULL,
                    source          TEXT    NOT NULL DEFAULT 'human'
                                        CHECK (source IN ('human', 'human_edited',
                                                          'imported', 'asr_draft')),
                    speaker_count   INTEGER,
                    is_final        INTEGER NOT NULL DEFAULT 0
                                        CHECK (is_final IN (0, 1)),
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    UNIQUE (audio_file_id, revision)
                );
                """
            )
            conn.execute(
                """INSERT INTO transcriptions
                       (id, audio_file_id, revision, text, source, speaker_count,
                        is_final, created_at)
                   SELECT id, audio_file_id, revision, text, source, speaker_count,
                          is_final, created_at
                   FROM transcriptions_v1"""
            )
            conn.execute("DROP TABLE transcriptions_v1")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcriptions_audio "
                "ON transcriptions(audio_file_id, revision DESC)"
            )

    # ------------------------------------------------------------ life cycle
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    # -------------------------------------------------------------- projects
    def create_project(self, name: str, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO projects (name, notes) VALUES (?, ?)", (name, notes)
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_project(self, project_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    def list_projects(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM projects ORDER BY name").fetchall()

    # ----------------------------------------------------------- audio files
    def add_audio_file(
        self,
        project_id: int,
        path: str | Path,
        duration: float,
        sample_rate: int,
        channels: int,
        content_hash: str | None = None,
        peak_dbfs: float | None = None,
        rms_dbfs: float | None = None,
        is_silent: bool = False,
    ) -> int:
        p = Path(path)
        cur = self.conn.execute(
            """INSERT INTO audio_files
                   (project_id, path, filename, content_hash, duration,
                    sample_rate, channels, peak_dbfs, rms_dbfs, is_silent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                str(p),
                p.name,
                content_hash,
                duration,
                sample_rate,
                channels,
                peak_dbfs,
                rms_dbfs,
                int(is_silent),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_audio_file(self, audio_file_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM audio_files WHERE id = ?", (audio_file_id,)
        ).fetchone()

    def list_audio_files(self, project_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM audio_files WHERE project_id = ? ORDER BY filename",
            (project_id,),
        ).fetchall()

    @staticmethod
    def file_hash(path: str | Path) -> str:
        """sha256 of a file's contents, read in chunks."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # -------------------------------------------------------------- speakers
    def upsert_speaker(
        self, audio_file_id: int, profile: SpeakerProfile, human_edited: bool | None = None
    ) -> int:
        """Insert or update a speaker profile.

        An automatic update never clears the `human_edited` flag: once a person
        has set a value, a later detector run must not silently overwrite it
        (CLAUDE.md sections 18A.11 and 56.15). Pass `human_edited=True` to
        record a human edit.
        """
        edited = profile.human_edited if human_edited is None else human_edited
        existing = self.conn.execute(
            "SELECT id, human_edited FROM speakers WHERE audio_file_id = ? AND speaker_id = ?",
            (audio_file_id, profile.speaker_id),
        ).fetchone()

        if existing is None:
            cur = self.conn.execute(
                """INSERT INTO speakers
                       (audio_file_id, speaker_id, display_name, gender, nativity,
                        first_seen, last_seen, confidence, human_edited)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audio_file_id,
                    profile.speaker_id,
                    profile.display_name,
                    profile.gender,
                    profile.nativity,
                    profile.first_seen,
                    profile.last_seen,
                    profile.confidence,
                    int(edited),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

        if existing["human_edited"] and not edited:
            # Refuse to let an automatic pass overwrite a human decision.
            # Timeline bounds are still safe to refresh.
            self.conn.execute(
                "UPDATE speakers SET first_seen = ?, last_seen = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (profile.first_seen, profile.last_seen, existing["id"]),
            )
            self.conn.commit()
            return int(existing["id"])

        self.conn.execute(
            """UPDATE speakers
                  SET display_name = ?, gender = ?, nativity = ?, first_seen = ?,
                      last_seen = ?, confidence = ?, human_edited = ?,
                      updated_at = datetime('now')
                WHERE id = ?""",
            (
                profile.display_name,
                profile.gender,
                profile.nativity,
                profile.first_seen,
                profile.last_seen,
                profile.confidence,
                int(edited or existing["human_edited"]),
                existing["id"],
            ),
        )
        self.conn.commit()
        return int(existing["id"])

    def list_speakers(self, audio_file_id: int) -> list[SpeakerProfile]:
        rows = self.conn.execute(
            "SELECT * FROM speakers WHERE audio_file_id = ? ORDER BY speaker_id",
            (audio_file_id,),
        ).fetchall()
        return [
            SpeakerProfile(
                speaker_id=r["speaker_id"],
                display_name=r["display_name"],
                gender=r["gender"],
                nativity=r["nativity"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                confidence=r["confidence"],
                human_edited=bool(r["human_edited"]),
            )
            for r in rows
        ]

    def speaker_numbers(self, audio_file_id: int) -> set[int]:
        """Speaker numbers present, for the validator's reference check."""
        return {
            int(p.speaker_id.lstrip("s"))
            for p in self.list_speakers(audio_file_id)
            if p.speaker_id.lstrip("s").isdigit()
        }

    # ---------------------------------------------------------------- events
    def add_event(self, audio_file_id: int, event: AnalysisEvent) -> int:
        cur = self.conn.execute(
            """INSERT INTO events
                   (audio_file_id, type, start_time, end_time, confidence,
                    speaker_id, state, confirmed_tag, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                audio_file_id,
                event.type.value,
                event.start,
                event.end,
                event.confidence,
                event.speaker_id,
                event.state.value,
                event.confirmed_tag,
                event.note,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_events(self, audio_file_id: int, events: list[AnalysisEvent]) -> list[int]:
        return [self.add_event(audio_file_id, e) for e in events]

    def list_events(
        self, audio_file_id: int, state: ConfirmationState | None = None
    ) -> list[AnalysisEvent]:
        sql = "SELECT * FROM events WHERE audio_file_id = ?"
        params: list[object] = [audio_file_id]
        if state is not None:
            sql += " AND state = ?"
            params.append(state.value)
        sql += " ORDER BY start_time"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            AnalysisEvent(
                type=EventType(r["type"]),
                start=r["start_time"],
                end=r["end_time"],
                confidence=r["confidence"],
                speaker_id=r["speaker_id"],
                event_id=str(r["id"]),
                audio_file_id=r["audio_file_id"],
                state=ConfirmationState(r["state"]),
                confirmed_tag=r["confirmed_tag"],
                note=r["note"],
            )
            for r in rows
        ]

    def set_event_state(
        self, event_row_id: int, state: ConfirmationState, confirmed_tag: str | None = None
    ) -> None:
        if state is not ConfirmationState.CONFIRMED and confirmed_tag is not None:
            raise ValueError("only a confirmed event may carry a tag (section 47)")
        self.conn.execute(
            "UPDATE events SET state = ?, confirmed_tag = ? WHERE id = ?",
            (state.value, confirmed_tag, event_row_id),
        )
        self.conn.commit()

    # -------------------------------------------------------- transcriptions
    def save_transcription(
        self,
        audio_file_id: int,
        text: str,
        source: str = "human",
        speaker_count: int | None = None,
        is_final: bool = False,
    ) -> tuple[int, int]:
        """Append a new transcription revision.

        Returns (row_id, revision). Existing revisions are never modified, so a
        previous human version is always recoverable.
        """
        if source not in ALLOWED_TRANSCRIPTION_SOURCES:
            raise ValueError(
                f"source {source!r} not allowed; use one of "
                f"{sorted(ALLOWED_TRANSCRIPTION_SOURCES)}. Machine text must be "
                "stored as 'asr_draft' so it is never mistaken for human work."
            )
        row = self.conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS r FROM transcriptions WHERE audio_file_id = ?",
            (audio_file_id,),
        ).fetchone()
        revision = int(row["r"]) + 1
        cur = self.conn.execute(
            """INSERT INTO transcriptions
                   (audio_file_id, revision, text, source, speaker_count, is_final)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (audio_file_id, revision, text, source, speaker_count, int(is_final)),
        )
        self.conn.commit()
        return int(cur.lastrowid), revision

    def latest_transcription(self, audio_file_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM transcriptions WHERE audio_file_id = ? "
            "ORDER BY revision DESC LIMIT 1",
            (audio_file_id,),
        ).fetchone()

    def transcription_history(self, audio_file_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM transcriptions WHERE audio_file_id = ? ORDER BY revision",
            (audio_file_id,),
        ).fetchall()

    # ---------------------------------------------------- validation results
    def save_validation(self, transcription_id: int, report: ValidationReport) -> int:
        rows = [
            (
                transcription_id,
                f.code.value,
                f.severity.value,
                f.message,
                f.rule_ref,
                f.start,
                f.end,
                f.excerpt,
                f.suggestion,
            )
            for f in report.sorted_findings()
        ]
        self.conn.executemany(
            """INSERT INTO validation_results
                   (transcription_id, code, severity, message, rule_ref,
                    start_offset, end_offset, excerpt, suggestion)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def list_validation(self, transcription_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM validation_results WHERE transcription_id = ? "
            "ORDER BY severity, start_offset",
            (transcription_id,),
        ).fetchall()

    # ------------------------------------------------------------- audit log
    def append_audit(
        self,
        action: str,
        audio_file_id: int | None = None,
        event_type: str | None = None,
        audio_start: float | None = None,
        audio_end: float | None = None,
        detail: dict | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO audit_logs
                   (audio_file_id, action, event_type, audio_start, audio_end, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                audio_file_id,
                action,
                event_type,
                audio_start,
                audio_end,
                json.dumps(detail or {}, default=str),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_audit(self, audio_file_id: int | None = None) -> list[sqlite3.Row]:
        if audio_file_id is None:
            return self.conn.execute(
                "SELECT * FROM audit_logs ORDER BY id"
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM audit_logs WHERE audio_file_id = ? ORDER BY id",
            (audio_file_id,),
        ).fetchall()
