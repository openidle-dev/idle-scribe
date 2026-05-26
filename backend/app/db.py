from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import Job, JobStatus, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    original_path     TEXT NOT NULL,
    engine            TEXT,
    language          TEXT,
    normalized_path   TEXT,
    transcript_path   TEXT,
    edited_transcript_path TEXT,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL,
    segment_start  REAL NOT NULL,
    segment_end    REAL NOT NULL,
    original_text  TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    speaker        TEXT,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
"""

# Columns added after the initial schema shipped; applied to pre-existing dbs.
_MIGRATIONS = {
    "transcript_path": "ALTER TABLE jobs ADD COLUMN transcript_path TEXT",
    "engine": "ALTER TABLE jobs ADD COLUMN engine TEXT",
    "language": "ALTER TABLE jobs ADD COLUMN language TEXT",
    "edited_transcript_path": "ALTER TABLE jobs ADD COLUMN edited_transcript_path TEXT",
}


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in existing:
                conn.execute(ddl)


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        original_filename=row["original_filename"],
        status=JobStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        original_path=row["original_path"],
        engine=row["engine"],
        language=row["language"],
        normalized_path=row["normalized_path"],
        transcript_path=row["transcript_path"],
        edited_transcript_path=row["edited_transcript_path"],
        error=row["error"],
    )


def create_job(db_path: Path, job: Job) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, original_filename, status, created_at,
                              updated_at, original_path, engine, language,
                              normalized_path, transcript_path, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.original_filename,
                job.status.value,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
                job.original_path,
                job.engine,
                job.language,
                job.normalized_path,
                job.transcript_path,
                job.error,
            ),
        )


def get_job(db_path: Path, job_id: str) -> Job | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(db_path: Path) -> list[Job]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def update_job(
    db_path: Path,
    job_id: str,
    *,
    status: JobStatus | None = None,
    normalized_path: str | None = None,
    transcript_path: str | None = None,
    edited_transcript_path: str | None = None,
    error: str | None = None,
) -> None:
    sets = ["updated_at = ?"]
    values: list[object] = [utcnow().isoformat()]
    if status is not None:
        sets.append("status = ?")
        values.append(status.value)
    if normalized_path is not None:
        sets.append("normalized_path = ?")
        values.append(normalized_path)
    if transcript_path is not None:
        sets.append("transcript_path = ?")
        values.append(transcript_path)
    if edited_transcript_path is not None:
        sets.append("edited_transcript_path = ?")
        values.append(edited_transcript_path)
    if error is not None:
        sets.append("error = ?")
        values.append(error)
    values.append(job_id)
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", values)


def replace_corrections(db_path: Path, job_id: str, corrections: list[dict]) -> int:
    """Replace the job's training pairs (audio time range <-> corrected text) with
    the current set. Idempotent — re-saving an edit doesn't accumulate duplicates.
    Returns the number of rows written."""
    now = utcnow().isoformat()
    rows = [
        (
            job_id,
            c["segment_start"],
            c["segment_end"],
            c["original_text"],
            c["corrected_text"],
            c.get("speaker"),
            now,
        )
        for c in corrections
    ]
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM corrections WHERE job_id = ?", (job_id,))
        if rows:
            conn.executemany(
                """
                INSERT INTO corrections (job_id, segment_start, segment_end,
                    original_text, corrected_text, speaker, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return len(rows)
