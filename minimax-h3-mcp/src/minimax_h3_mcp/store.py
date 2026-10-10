from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import JobRecord, JobStatus, utc_now


_COLUMNS = (
    "id",
    "status",
    "progress",
    "stage",
    "message",
    "prompt",
    "negative_prompt",
    "soundscape",
    "prompt_mode",
    "model_id",
    "requested_duration_seconds",
    "num_frames",
    "width",
    "height",
    "fps",
    "seed",
    "audio_enabled",
    "aspect_ratio",
    "resolution",
    "first_frame_asset_id",
    "last_frame_asset_id",
    "reference_image_asset_id",
    "first_frame_path",
    "last_frame_path",
    "reference_image_path",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "output_path",
    "error",
)


class JobStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT,
                    soundscape TEXT,
                    prompt_mode TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    requested_duration_seconds REAL NOT NULL,
                    num_frames INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    fps INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    audio_enabled INTEGER NOT NULL,
                    aspect_ratio TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    first_frame_asset_id TEXT,
                    last_frame_asset_id TEXT,
                    reference_image_asset_id TEXT,
                    first_frame_path TEXT,
                    last_frame_path TEXT,
                    reference_image_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    output_path TEXT,
                    error TEXT
                )
                """
            )
            existing = {
                str(row[1]) for row in self._connection.execute("PRAGMA table_info(jobs)")
            }
            for column in ("reference_image_asset_id", "reference_image_path"):
                if column not in existing:
                    self._connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at DESC)"
            )

    def create(self, job: JobRecord) -> JobRecord:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        values = tuple(getattr(job, column) for column in _COLUMNS)
        with self._lock, self._connection:
            self._connection.execute(
                f"INSERT INTO jobs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def require(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job_id: {job_id}")
        return job

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        allowed = set(_COLUMNS) - {"id", "created_at"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown job fields: {', '.join(sorted(unknown))}")
        if not changes:
            return self.require(job_id)
        changes["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in changes)
        values = tuple(changes.values()) + (job_id,)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?", values
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown job_id: {job_id}")
        return self.require(job_id)

    def list(self, limit: int = 20, status: str | None = None) -> list[JobRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        query = "SELECT * FROM jobs"
        parameters: list[Any] = []
        if status:
            query += " WHERE status = ?"
            parameters.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._from_row(row) for row in rows]

    def count_active(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM jobs
                WHERE status IN (?, ?, ?)
                """,
                (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED),
            ).fetchone()
        return int(row["count"])

    def recover_interrupted_jobs(self) -> int:
        now = utc_now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET status = ?, progress = 100, stage = ?, message = ?, error = ?,
                    updated_at = ?, finished_at = ?
                WHERE status IN (?, ?, ?)
                """,
                (
                    JobStatus.FAILED,
                    "interrupted",
                    "服务重启，原任务已中断",
                    "service_restarted",
                    now,
                    now,
                    JobStatus.QUEUED,
                    JobStatus.RUNNING,
                    JobStatus.CANCEL_REQUESTED,
                ),
            )
        return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> JobRecord:
        values = {column: row[column] for column in _COLUMNS}
        values["audio_enabled"] = bool(values["audio_enabled"])
        return JobRecord(**values)
