from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connection import open_connection


DEFAULT_STUDENT_PREFIX = "\u5b66\u751f-"


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


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class ProgressStore:
    def __init__(self, data_path: Optional[Path] = None):
        self.legacy_data_path = Path(data_path) if data_path else None

    def ensure_user(self, user_id: str, display_name: Optional[str] = None, account_name: Optional[str] = None) -> Dict[str, Any]:
        user_id = self._clean_user_id(user_id)
        if not user_id:
            raise ValueError("Missing user_id.")
        default_name = f"学生-{user_id[:8]}"
        clean_display = self._clean_name(display_name)
        clean_account = self._clean_name(account_name)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at
                FROM deep_learning_progress_users
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                payload = {
                    "display_name": clean_display or clean_account or default_name,
                    "account_name": clean_account or clean_display or default_name,
                    "attempt_count": 0,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "last_attempt_at": None,
                }
                cur.execute(
                    """
                    INSERT INTO deep_learning_progress_users
                    (user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        payload["display_name"],
                        payload["account_name"],
                        payload["attempt_count"],
                        payload["created_at"],
                        payload["updated_at"],
                        payload["last_attempt_at"],
                    ),
                )
                return self._serialize_user(payload, user_id)

            payload = dict(row)
            changed = False
            if not self._clean_name(payload.get("account_name")):
                payload["account_name"] = self._clean_name(payload.get("display_name")) or default_name
                changed = True
            if not self._clean_name(payload.get("display_name")):
                payload["display_name"] = self._clean_name(payload.get("account_name")) or default_name
                changed = True
            if payload.get("attempt_count") is None:
                payload["attempt_count"] = 0
                changed = True
            if not payload.get("created_at"):
                payload["created_at"] = _now()
                changed = True
            if clean_account:
                previous_account = self._clean_name(payload.get("account_name"))
                payload["account_name"] = clean_account
                current_display = self._clean_name(payload.get("display_name"))
                if (not current_display) or current_display.startswith(("Learner-", "Student-", "学生-")) or current_display == previous_account:
                    payload["display_name"] = clean_account
                changed = True
            elif clean_display:
                payload["display_name"] = clean_display
                changed = True
            if changed:
                payload["updated_at"] = _now()
                cur.execute(
                    """
                    UPDATE deep_learning_progress_users
                    SET display_name = %s,
                        account_name = %s,
                        attempt_count = %s,
                        created_at = %s,
                        updated_at = %s,
                        last_attempt_at = %s
                    WHERE user_id = %s
                    """,
                    (
                        payload["display_name"],
                        payload["account_name"],
                        int(payload.get("attempt_count") or 0),
                        payload.get("created_at"),
                        payload.get("updated_at"),
                        payload.get("last_attempt_at"),
                        user_id,
                    ),
                )
        return self._serialize_user(payload, user_id)

    def set_display_name(self, user_id: str, display_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(display_name) or f"学生-{self._clean_user_id(user_id)[:8]}"
        return self.ensure_user(user_id, clean_name)

    def set_account_name(self, user_id: str, account_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(account_name) or f"学生-{self._clean_user_id(user_id)[:8]}"
        return self.ensure_user(user_id, account_name=clean_name)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self.ensure_user(user_id)

    def all_attempts(self, user_id: str) -> List[Dict[str, Any]]:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_attempt_id, timestamp, kp_id, kp_name, question_id, question_type, question,
                       submitted_answer, reference_answer, is_correct
                FROM deep_learning_attempts
                WHERE user_id = %s
                ORDER BY timestamp ASC, attempt_pk ASC
                """,
                (clean_user_id,),
            )
            rows = cur.fetchall() or []
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = self._serialize_attempt(row)
            item["attempt_id"] = str(row.get("source_attempt_id") or "")
            output.append(item)
        return output

    def latest_attempts(self, user_id: str) -> List[Dict[str, Any]]:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, question_id, timestamp, kp_id, kp_name, question_type, question,
                       submitted_answer, reference_answer, is_correct
                FROM deep_learning_latest_attempts
                WHERE user_id = %s
                ORDER BY timestamp ASC, question_id ASC
                """,
                (clean_user_id,),
            )
            rows = cur.fetchall() or []
        if rows:
            return [self._serialize_attempt(row) for row in rows]
        return self._backfill_latest_attempts(clean_user_id)

    def attempt_count(self, user_id: str) -> int:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT attempt_count FROM deep_learning_progress_users WHERE user_id = %s", (clean_user_id,))
            row = cur.fetchone()
            count = row.get("attempt_count") if row else None
            if isinstance(count, int) and count >= 0:
                return count
            cur.execute("SELECT COUNT(*) AS count_value FROM deep_learning_attempts WHERE user_id = %s", (clean_user_id,))
            computed = int((cur.fetchone() or {}).get("count_value") or 0)
            cur.execute(
                """
                INSERT INTO deep_learning_progress_users (user_id, display_name, account_name, attempt_count, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET attempt_count = EXCLUDED.attempt_count, updated_at = EXCLUDED.updated_at
                """,
                (clean_user_id, f"学生-{clean_user_id[:8]}", f"学生-{clean_user_id[:8]}", computed, _now(), _now()),
            )
            return computed

    def record_attempts(self, user_id: str, kp_id: str, kp_name: str, results: List[Dict[str, Any]]) -> None:
        clean_user_id = self._clean_user_id(user_id)
        if not results:
            return
        profile = self.ensure_user(clean_user_id)
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            for index, result in enumerate(results, start=1):
                payload = {
                    "timestamp": now,
                    "kp_id": str(kp_id or ""),
                    "kp_name": str(kp_name or ""),
                    "question_id": str(result.get("question_id") or ""),
                    "question_type": str(result.get("question_type") or ""),
                    "question": str(result.get("question") or ""),
                    "submitted_answer": str(result.get("submitted_answer") or ""),
                    "reference_answer": str(result.get("reference_answer") or ""),
                    "is_correct": bool(result.get("is_correct")),
                }
                source_attempt_id = self._generated_attempt_source_id(clean_user_id, index, payload)
                cur.execute(
                    """
                    INSERT INTO deep_learning_attempts
                    (user_id, source_attempt_id, timestamp, kp_id, kp_name, question_id, question_type,
                     question, submitted_answer, reference_answer, is_correct, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (user_id, source_attempt_id) DO NOTHING
                    """,
                    (
                        clean_user_id,
                        source_attempt_id,
                        payload["timestamp"],
                        payload["kp_id"],
                        payload["kp_name"],
                        payload["question_id"],
                        payload["question_type"],
                        payload["question"],
                        payload["submitted_answer"],
                        payload["reference_answer"],
                        payload["is_correct"],
                        json.dumps({}),
                    ),
                )
                if payload["question_id"]:
                    cur.execute(
                        """
                        INSERT INTO deep_learning_latest_attempts
                        (user_id, question_id, timestamp, kp_id, kp_name, question_type, question,
                         submitted_answer, reference_answer, is_correct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, question_id) DO UPDATE
                        SET timestamp = EXCLUDED.timestamp,
                            kp_id = EXCLUDED.kp_id,
                            kp_name = EXCLUDED.kp_name,
                            question_type = EXCLUDED.question_type,
                            question = EXCLUDED.question,
                            submitted_answer = EXCLUDED.submitted_answer,
                            reference_answer = EXCLUDED.reference_answer,
                            is_correct = EXCLUDED.is_correct
                        """,
                        (
                            clean_user_id,
                            payload["question_id"],
                            payload["timestamp"],
                            payload["kp_id"],
                            payload["kp_name"],
                            payload["question_type"],
                            payload["question"],
                            payload["submitted_answer"],
                            payload["reference_answer"],
                            payload["is_correct"],
                        ),
                    )
            cur.execute(
                """
                INSERT INTO deep_learning_progress_users
                (user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    account_name = EXCLUDED.account_name,
                    attempt_count = deep_learning_progress_users.attempt_count + %s,
                    updated_at = EXCLUDED.updated_at,
                    last_attempt_at = EXCLUDED.last_attempt_at
                """,
                (
                    clean_user_id,
                    profile.get("display_name") or f"学生-{clean_user_id[:8]}",
                    profile.get("account_name") or profile.get("display_name") or f"学生-{clean_user_id[:8]}",
                    0,
                    _parse_ts(profile.get("created_at")) or now,
                    now,
                    now,
                    len(results),
                ),
            )
            cur.execute(
                """
                INSERT INTO deep_learning_followup_cache (user_id, cache_key, created_at, items)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (user_id) DO UPDATE
                SET cache_key = EXCLUDED.cache_key,
                    created_at = EXCLUDED.created_at,
                    items = EXCLUDED.items
                """,
                (clean_user_id, "", now, json.dumps([])),
            )

    def get_generated_followups(self, user_id: str, cache_key: str) -> List[Dict[str, Any]]:
        clean_user_id = self._clean_user_id(user_id)
        if not str(cache_key or "").strip():
            return []
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT cache_key, items FROM deep_learning_followup_cache WHERE user_id = %s", (clean_user_id,))
            row = cur.fetchone()
        if not row or str(row.get("cache_key") or "") != str(cache_key or ""):
            return []
        return [self._sanitize_nested_item(item) for item in (row.get("items") or []) if isinstance(item, dict)]

    def set_generated_followups(self, user_id: str, cache_key: str, items: List[Dict[str, Any]]) -> None:
        clean_user_id = self._clean_user_id(user_id)
        profile = self.ensure_user(clean_user_id)
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_followup_cache (user_id, cache_key, created_at, items)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (user_id) DO UPDATE
                SET cache_key = EXCLUDED.cache_key,
                    created_at = EXCLUDED.created_at,
                    items = EXCLUDED.items
                """,
                (
                    clean_user_id,
                    str(cache_key or ""),
                    now,
                    json.dumps([self._sanitize_nested_item(item) for item in items if isinstance(item, dict)], ensure_ascii=False),
                ),
            )
            cur.execute(
                """
                INSERT INTO deep_learning_progress_users
                (user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    account_name = EXCLUDED.account_name,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    clean_user_id,
                    profile.get("display_name") or f"学生-{clean_user_id[:8]}",
                    profile.get("account_name") or profile.get("display_name") or f"学生-{clean_user_id[:8]}",
                    int(profile.get("attempt_count") or 0),
                    _parse_ts(profile.get("created_at")) or now,
                    now,
                        _parse_ts(profile.get("last_attempt_at")),
                ),
            )

    def learning_report_signature(self, user_id: str) -> Dict[str, Any]:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT attempt_count, last_attempt_at, updated_at
                FROM deep_learning_progress_users
                WHERE user_id = %s
                """,
                (clean_user_id,),
            )
            row = cur.fetchone()
        if not row:
            profile = self.ensure_user(clean_user_id)
            return {
                "attempt_count": 0,
                "last_attempt_at": "",
                "updated_at": str(profile.get("updated_at") or ""),
            }
        return {
            "attempt_count": int(row.get("attempt_count") or 0),
            "last_attempt_at": _iso(row.get("last_attempt_at")),
            "updated_at": _iso(row.get("updated_at")),
        }

    def summary(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        attempts = self.latest_attempts(user_id)
        answered = len(attempts)
        correct = sum(1 for item in attempts if item.get("is_correct"))
        wrong = answered - correct
        return {"answered": answered, "correct": correct, "wrong": wrong, "accuracy": round((correct / answered) * 100, 1) if answered else 0.0, "display_name": user.get("display_name")}

    def kp_stats(self, user_id: str) -> List[Dict[str, Any]]:
        attempts = self.latest_attempts(user_id)
        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"kp_id": "", "kp_name": "", "answered": 0, "correct": 0, "wrong": 0, "last_at": ""})
        for item in attempts:
            bucket = grouped[item["kp_id"]]
            bucket["kp_id"] = item["kp_id"]
            bucket["kp_name"] = item["kp_name"]
            bucket["answered"] += 1
            if item.get("is_correct"):
                bucket["correct"] += 1
            else:
                bucket["wrong"] += 1
            bucket["last_at"] = max(bucket["last_at"], item.get("timestamp") or "")
        rows = list(grouped.values())
        for row in rows:
            row["accuracy"] = round((row["correct"] / row["answered"]) * 100, 1) if row["answered"] else 0.0
        rows.sort(key=lambda row: (-row["wrong"], row["accuracy"], row["kp_name"]))
        return rows

    def weak_points(self, user_id: str, minimum_attempts: int = 1) -> List[Dict[str, Any]]:
        rows = [row for row in self.kp_stats(user_id) if row["answered"] >= minimum_attempts and (row["wrong"] > 0 or row["accuracy"] < 70)]
        rows.sort(key=lambda row: (-row["wrong"], row["accuracy"], row["kp_name"]))
        return rows[:6]

    def recent_attempts(self, user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_attempt_id, timestamp, kp_id, kp_name, question_id, question_type, question,
                       submitted_answer, reference_answer, is_correct
                FROM deep_learning_attempts
                WHERE user_id = %s
                ORDER BY timestamp DESC, attempt_pk DESC
                LIMIT %s
                """,
                (clean_user_id, max(int(limit), 1)),
            )
            rows = cur.fetchall() or []
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = self._serialize_attempt(row)
            item["attempt_id"] = str(row.get("source_attempt_id") or "")
            output.append(item)
        return output

    def upsert_profile_snapshot(self, user_id: str, payload: Dict[str, Any]) -> None:
        clean_user_id = self._clean_user_id(user_id)
        display_name = self._clean_name(payload.get("display_name")) or f"学生-{clean_user_id[:8]}"
        account_name = self._clean_name(payload.get("account_name")) or display_name
        created_at = _parse_ts(payload.get("created_at")) or _now()
        updated_at = _parse_ts(payload.get("updated_at")) or created_at
        last_attempt_at = _parse_ts(payload.get("last_attempt_at"))
        attempt_count = int(payload.get("attempt_count") or 0)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_progress_users
                (user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    account_name = EXCLUDED.account_name,
                    attempt_count = EXCLUDED.attempt_count,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    last_attempt_at = EXCLUDED.last_attempt_at
                """,
                (clean_user_id, display_name, account_name, attempt_count, created_at, updated_at, last_attempt_at),
            )

    def insert_attempt_snapshot(self, user_id: str, attempt_doc: Dict[str, Any], source_attempt_id: Optional[str] = None) -> None:
        clean_user_id = self._clean_user_id(user_id)
        payload = self._serialize_attempt(attempt_doc)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_attempts
                (user_id, source_attempt_id, timestamp, kp_id, kp_name, question_id, question_type,
                 question, submitted_answer, reference_answer, is_correct, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (user_id, source_attempt_id) DO UPDATE
                SET timestamp = EXCLUDED.timestamp,
                    kp_id = EXCLUDED.kp_id,
                    kp_name = EXCLUDED.kp_name,
                    question_id = EXCLUDED.question_id,
                    question_type = EXCLUDED.question_type,
                    question = EXCLUDED.question,
                    submitted_answer = EXCLUDED.submitted_answer,
                    reference_answer = EXCLUDED.reference_answer,
                    is_correct = EXCLUDED.is_correct
                """,
                (
                    clean_user_id,
                    str(source_attempt_id or "") or None,
                    _parse_ts(payload.get("timestamp")) or _now(),
                    payload.get("kp_id") or "",
                    payload.get("kp_name") or "",
                    payload.get("question_id") or "",
                    payload.get("question_type") or "",
                    payload.get("question") or "",
                    payload.get("submitted_answer") or "",
                    payload.get("reference_answer") or "",
                    bool(payload.get("is_correct")),
                    json.dumps({}),
                ),
            )

    def upsert_latest_attempt_snapshot(self, user_id: str, attempt_doc: Dict[str, Any]) -> None:
        clean_user_id = self._clean_user_id(user_id)
        payload = self._serialize_attempt(attempt_doc)
        question_id = payload.get("question_id") or ""
        if not question_id:
            return
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_latest_attempts
                (user_id, question_id, timestamp, kp_id, kp_name, question_type, question,
                 submitted_answer, reference_answer, is_correct)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, question_id) DO UPDATE
                SET timestamp = EXCLUDED.timestamp,
                    kp_id = EXCLUDED.kp_id,
                    kp_name = EXCLUDED.kp_name,
                    question_type = EXCLUDED.question_type,
                    question = EXCLUDED.question,
                    submitted_answer = EXCLUDED.submitted_answer,
                    reference_answer = EXCLUDED.reference_answer,
                    is_correct = EXCLUDED.is_correct
                """,
                (
                    clean_user_id,
                    question_id,
                    _parse_ts(payload.get("timestamp")) or _now(),
                    payload.get("kp_id") or "",
                    payload.get("kp_name") or "",
                    payload.get("question_type") or "",
                    payload.get("question") or "",
                    payload.get("submitted_answer") or "",
                    payload.get("reference_answer") or "",
                    bool(payload.get("is_correct")),
                ),
            )

    def set_followup_snapshot(self, user_id: str, payload: Dict[str, Any]) -> None:
        clean_user_id = self._clean_user_id(user_id)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_followup_cache (user_id, cache_key, created_at, items)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (user_id) DO UPDATE
                SET cache_key = EXCLUDED.cache_key,
                    created_at = EXCLUDED.created_at,
                    items = EXCLUDED.items
                """,
                (
                    clean_user_id,
                    str(payload.get("cache_key") or ""),
                    _parse_ts(payload.get("created_at")) or _now(),
                    json.dumps([self._sanitize_nested_item(item) for item in payload.get("items", []) if isinstance(item, dict)], ensure_ascii=False),
                ),
            )

    def _serialize_user(self, payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        default_name = f"学生-{user_id[:8]}"
        display_name = self._clean_name(payload.get("display_name")) or self._clean_name(payload.get("account_name")) or default_name
        account_name = self._clean_name(payload.get("account_name")) or display_name
        return {"display_name": display_name, "account_name": account_name, "attempt_count": int(payload.get("attempt_count") or 0), "created_at": _iso(payload.get("created_at")), "updated_at": _iso(payload.get("updated_at")), "last_attempt_at": _iso(payload.get("last_attempt_at"))}

    def _serialize_attempt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"timestamp": _iso(payload.get("timestamp")), "kp_id": str(payload.get("kp_id") or ""), "kp_name": str(payload.get("kp_name") or ""), "question_id": str(payload.get("question_id") or ""), "question_type": str(payload.get("question_type") or ""), "question": str(payload.get("question") or ""), "submitted_answer": str(payload.get("submitted_answer") or ""), "reference_answer": str(payload.get("reference_answer") or ""), "is_correct": bool(payload.get("is_correct"))}

    def _sanitize_nested_item(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._sanitize_nested_item(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_nested_item(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_nested_item(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, datetime):
            return _iso(value)
        return value

    def _clean_name(self, value: Optional[str]) -> str:
        return (value or "").strip()[:40]

    def _clean_user_id(self, value: Optional[str]) -> str:
        return (value or "").strip()

    def _generated_attempt_source_id(self, user_id: str, index: int, attempt: Dict[str, Any]) -> str:
        raw = "|".join([user_id, str(index), str(attempt.get("timestamp") or ""), str(attempt.get("kp_id") or ""), str(attempt.get("question_id") or ""), str(attempt.get("submitted_answer") or ""), str(attempt.get("reference_answer") or "")])
        return f"generated-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"

    def _backfill_latest_attempts(self, user_id: str) -> List[Dict[str, Any]]:
        attempts = self.all_attempts(user_id)
        if not attempts:
            return []
        latest: Dict[str, Dict[str, Any]] = {}
        for item in attempts:
            question_id = str(item.get("question_id") or "")
            if not question_id:
                continue
            previous = latest.get(question_id)
            if previous is None or (item.get("timestamp") or "") >= (previous.get("timestamp") or ""):
                latest[question_id] = item
        rows = list(latest.values())
        rows.sort(key=lambda item: (item.get("timestamp") or "", item.get("question_id") or ""))
        return rows
