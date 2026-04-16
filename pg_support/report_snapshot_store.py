from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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


class LearningReportSnapshotStore:
    def get_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, signature_key, target_signature_key, status, payload, generated_at,
                       updated_at, error_message, worker_id
                FROM deep_learning_learning_report_snapshots
                WHERE user_id = %s
                """,
                (str(user_id or "").strip(),),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def schedule_refresh(self, user_id: str, signature_key: str) -> Dict[str, Any]:
        clean_user_id = str(user_id or "").strip()
        clean_signature = str(signature_key or "").strip()
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_learning_report_snapshots
                (user_id, signature_key, target_signature_key, status, payload, generated_at, updated_at, error_message, worker_id)
                VALUES (%s, '', %s, 'queued', '{}'::jsonb, NULL, %s, '', '')
                ON CONFLICT (user_id) DO UPDATE
                SET target_signature_key = EXCLUDED.target_signature_key,
                    status = CASE
                        WHEN deep_learning_learning_report_snapshots.signature_key = EXCLUDED.target_signature_key
                             AND deep_learning_learning_report_snapshots.status = 'ready'
                        THEN 'ready'
                        WHEN deep_learning_learning_report_snapshots.target_signature_key = EXCLUDED.target_signature_key
                             AND deep_learning_learning_report_snapshots.status IN ('queued', 'running')
                        THEN deep_learning_learning_report_snapshots.status
                        ELSE 'queued'
                    END,
                    updated_at = EXCLUDED.updated_at,
                    error_message = CASE
                        WHEN deep_learning_learning_report_snapshots.signature_key = EXCLUDED.target_signature_key
                             AND deep_learning_learning_report_snapshots.status = 'ready'
                        THEN deep_learning_learning_report_snapshots.error_message
                        ELSE ''
                    END
                RETURNING user_id, signature_key, target_signature_key, status, payload, generated_at,
                          updated_at, error_message, worker_id
                """,
                (clean_user_id, clean_signature, now),
            )
            row = cur.fetchone()
        return self._serialize(row or {})

    def claim_next_refresh(self, worker_id: str) -> Optional[Dict[str, Any]]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH next_snapshot AS (
                    SELECT user_id
                    FROM deep_learning_learning_report_snapshots
                    WHERE status = 'queued'
                    ORDER BY updated_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE deep_learning_learning_report_snapshots AS snapshots
                SET status = 'running',
                    updated_at = %s,
                    error_message = '',
                    worker_id = %s
                FROM next_snapshot
                WHERE snapshots.user_id = next_snapshot.user_id
                RETURNING snapshots.user_id, snapshots.signature_key, snapshots.target_signature_key,
                          snapshots.status, snapshots.payload, snapshots.generated_at, snapshots.updated_at,
                          snapshots.error_message, snapshots.worker_id
                """,
                (now, str(worker_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row) if row else None

    def mark_ready(self, user_id: str, signature_key: str, payload: Dict[str, Any], worker_id: str = "") -> Dict[str, Any]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_learning_report_snapshots
                SET signature_key = %s,
                    target_signature_key = %s,
                    status = 'ready',
                    payload = %s::jsonb,
                    generated_at = %s,
                    updated_at = %s,
                    error_message = '',
                    worker_id = %s
                WHERE user_id = %s
                RETURNING user_id, signature_key, target_signature_key, status, payload, generated_at,
                          updated_at, error_message, worker_id
                """,
                (
                    str(signature_key or "").strip(),
                    str(signature_key or "").strip(),
                    json.dumps(dict(payload or {}), ensure_ascii=False),
                    now,
                    now,
                    str(worker_id or "").strip(),
                    str(user_id or "").strip(),
                ),
            )
            row = cur.fetchone()
        return self._serialize(row or {})

    def mark_failed(self, user_id: str, error_message: str, worker_id: str = "") -> Dict[str, Any]:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_learning_report_snapshots
                SET status = 'failed',
                    updated_at = %s,
                    error_message = %s,
                    worker_id = %s
                WHERE user_id = %s
                RETURNING user_id, signature_key, target_signature_key, status, payload, generated_at,
                          updated_at, error_message, worker_id
                """,
                (now, " ".join(str(error_message or "").split())[:500], str(worker_id or "").strip(), str(user_id or "").strip()),
            )
            row = cur.fetchone()
        return self._serialize(row or {})

    def requeue_stale_running(self, stale_seconds: int) -> int:
        seconds = max(int(stale_seconds or 0), 1)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deep_learning_learning_report_snapshots
                SET status = 'queued',
                    updated_at = NOW(),
                    error_message = ''
                WHERE status = 'running'
                  AND updated_at < NOW() - make_interval(secs => %s)
                """,
                (seconds,),
            )
            return int(cur.rowcount or 0)

    def stats(self) -> Dict[str, int]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'queued') AS queued,
                    COUNT(*) FILTER (WHERE status = 'running') AS running,
                    COUNT(*) FILTER (WHERE status = 'ready') AS ready,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed
                FROM deep_learning_learning_report_snapshots
                """
            )
            row = cur.fetchone() or {}
        return {
            "queued": int(row.get("queued") or 0),
            "running": int(row.get("running") or 0),
            "ready": int(row.get("ready") or 0),
            "failed": int(row.get("failed") or 0),
        }

    def _serialize(self, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(row or {})
        return {
            "user_id": str(payload.get("user_id") or ""),
            "signature_key": str(payload.get("signature_key") or ""),
            "target_signature_key": str(payload.get("target_signature_key") or ""),
            "status": str(payload.get("status") or "empty"),
            "payload": dict(payload.get("payload") or {}),
            "generated_at": _iso(payload.get("generated_at")),
            "updated_at": _iso(payload.get("updated_at")),
            "error_message": str(payload.get("error_message") or ""),
            "worker_id": str(payload.get("worker_id") or ""),
        }
