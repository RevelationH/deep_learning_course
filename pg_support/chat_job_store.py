from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from .connection import open_connection


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


class ChatJobStore:
    def fail_running_jobs_for_restart(self, error_message: str = "Job was interrupted by a service restart.") -> int:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_chat_jobs
                SET status = 'failed',
                    error_message = %s,
                    finished_at = %s,
                    updated_at = %s
                WHERE status = 'running'
                """,
                (" ".join(str(error_message or "").split())[:500], now, now),
            )
            return int(cur.rowcount or 0)

    def touch_running_job(self, job_id: str, worker_id: str) -> bool:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_chat_jobs
                SET updated_at = %s
                WHERE job_id = %s
                  AND status = 'running'
                  AND worker_id = %s
                """,
                (now, str(job_id or "").strip(), str(worker_id or "").strip()),
            )
            return bool(cur.rowcount or 0)

    def enqueue(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        request_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job_id = uuid4().hex
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_chat_jobs
                (job_id, user_id, session_id, message, status, result, error_message, created_at, updated_at, worker_id, request_meta)
                VALUES (%s, %s, %s, %s, 'queued', '{}'::jsonb, '', %s, %s, '', %s::jsonb)
                RETURNING job_id, user_id, session_id, message, status, result, error_message,
                          created_at, started_at, finished_at, updated_at, worker_id, request_meta
                """,
                (
                    job_id,
                    str(user_id or "").strip(),
                    str(session_id or "").strip(),
                    str(message or "").strip(),
                    now,
                    now,
                    json.dumps(dict(request_meta or {}), ensure_ascii=False),
                ),
            )
            row = cur.fetchone()
        return self._serialize(row or {})

    def get_job(self, user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, user_id, session_id, message, status, result, error_message,
                       created_at, started_at, finished_at, updated_at, worker_id, request_meta
                FROM deep_learning_chat_jobs
                WHERE user_id = %s AND job_id = %s
                """,
                (str(user_id or "").strip(), str(job_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def get_active_job_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, user_id, session_id, message, status, result, error_message,
                       created_at, started_at, finished_at, updated_at, worker_id, request_meta
                FROM deep_learning_chat_jobs
                WHERE user_id = %s AND status IN ('queued', 'running')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (str(user_id or "").strip(),),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def count_active_jobs(self) -> Dict[str, Any]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'queued') AS queued,
                    COUNT(*) FILTER (WHERE status = 'running') AS running,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                    COUNT(DISTINCT user_id) FILTER (WHERE status IN ('queued', 'running')) AS active_users,
                    COALESCE(EXTRACT(EPOCH FROM (NOW() - MIN(created_at) FILTER (WHERE status = 'queued'))), 0) AS oldest_queued_age_seconds,
                    COALESCE(
                        AVG(EXTRACT(EPOCH FROM (finished_at - started_at))) FILTER (
                            WHERE status = 'completed'
                              AND started_at IS NOT NULL
                              AND finished_at IS NOT NULL
                              AND finished_at >= NOW() - INTERVAL '2 hours'
                        ),
                        0
                    ) AS avg_completion_seconds,
                    COALESCE(
                        MAX(EXTRACT(EPOCH FROM (NOW() - started_at))) FILTER (
                            WHERE status = 'running'
                              AND started_at IS NOT NULL
                        ),
                        0
                    ) AS longest_running_age_seconds
                FROM deep_learning_chat_jobs
                """
            )
            row = cur.fetchone() or {}
        return {
            "queued": int(row.get("queued") or 0),
            "running": int(row.get("running") or 0),
            "completed": int(row.get("completed") or 0),
            "failed": int(row.get("failed") or 0),
            "active_users": int(row.get("active_users") or 0),
            "oldest_queued_age_seconds": float(row.get("oldest_queued_age_seconds") or 0.0),
            "avg_completion_seconds": float(row.get("avg_completion_seconds") or 0.0),
            "longest_running_age_seconds": float(row.get("longest_running_age_seconds") or 0.0),
        }

    def queue_position(self, job_id: str) -> int:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH current_job AS (
                    SELECT created_at, status
                    FROM deep_learning_chat_jobs
                    WHERE job_id = %s
                )
                SELECT COUNT(*) AS ahead
                FROM deep_learning_chat_jobs, current_job
                WHERE current_job.status = 'queued'
                  AND deep_learning_chat_jobs.status = 'queued'
                  AND deep_learning_chat_jobs.created_at < current_job.created_at
                """,
                (str(job_id or "").strip(),),
            )
            row = cur.fetchone() or {}
        ahead = int(row.get("ahead") or 0)
        return ahead + 1 if ahead >= 0 else 0

    def claim_next_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH next_job AS (
                    SELECT job_id
                    FROM deep_learning_chat_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE deep_learning_chat_jobs AS jobs
                SET status = 'running',
                    started_at = %s,
                    updated_at = %s,
                    worker_id = %s,
                    error_message = ''
                FROM next_job
                WHERE jobs.job_id = next_job.job_id
                RETURNING jobs.job_id, jobs.user_id, jobs.session_id, jobs.message, jobs.status, jobs.result,
                          jobs.error_message, jobs.created_at, jobs.started_at, jobs.finished_at, jobs.updated_at,
                          jobs.worker_id, jobs.request_meta
                """,
                (now, now, str(worker_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def mark_completed(self, job_id: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_chat_jobs
                SET status = 'completed',
                    result = %s::jsonb,
                    error_message = '',
                    finished_at = %s,
                    updated_at = %s
                WHERE job_id = %s
                RETURNING job_id, user_id, session_id, message, status, result, error_message,
                          created_at, started_at, finished_at, updated_at, worker_id, request_meta
                """,
                (json.dumps(dict(result or {}), ensure_ascii=False), now, now, str(job_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def mark_failed(self, job_id: str, error_message: str) -> Optional[Dict[str, Any]]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_chat_jobs
                SET status = 'failed',
                    error_message = %s,
                    finished_at = %s,
                    updated_at = %s
                WHERE job_id = %s
                RETURNING job_id, user_id, session_id, message, status, result, error_message,
                          created_at, started_at, finished_at, updated_at, worker_id, request_meta
                """,
                (" ".join(str(error_message or "").split())[:500], now, now, str(job_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def fail_stale_running_jobs(self, stale_seconds: int) -> int:
        seconds = max(int(stale_seconds or 0), 1)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_chat_jobs
                SET status = 'failed',
                    error_message = 'Job was interrupted by a service restart.',
                    finished_at = NOW(),
                    updated_at = NOW()
                WHERE status = 'running'
                  AND updated_at < NOW() - make_interval(secs => %s)
                """,
                (seconds,),
            )
            return int(cur.rowcount or 0)

    def prune_finished_jobs(self, retention_seconds: int) -> int:
        seconds = max(int(retention_seconds or 0), 60)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM deep_learning_chat_jobs
                WHERE status IN ('completed', 'failed')
                  AND updated_at < NOW() - make_interval(secs => %s)
                """,
                (seconds,),
            )
            return int(cur.rowcount or 0)

    def _serialize(self, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(row or {})
        return {
            "job_id": str(payload.get("job_id") or ""),
            "user_id": str(payload.get("user_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "message": str(payload.get("message") or ""),
            "status": str(payload.get("status") or "queued"),
            "result": dict(payload.get("result") or {}),
            "error_message": str(payload.get("error_message") or ""),
            "created_at": _iso(payload.get("created_at")),
            "started_at": _iso(payload.get("started_at")),
            "finished_at": _iso(payload.get("finished_at")),
            "updated_at": _iso(payload.get("updated_at")),
            "worker_id": str(payload.get("worker_id") or ""),
            "request_meta": dict(payload.get("request_meta") or {}),
        }
