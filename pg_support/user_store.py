from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .connection import open_connection


def _now() -> datetime:
    return datetime.now(timezone.utc)


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


class User:
    def __init__(self, username: str, password: str, is_admin: bool = False):
        self.username = str(username or "").strip()
        self.password = str(password or "")
        self.is_admin = bool(is_admin)

    def save(self) -> None:
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_users (
                    username,
                    password_hash,
                    is_admin,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE
                SET password_hash = EXCLUDED.password_hash,
                    is_admin = EXCLUDED.is_admin,
                    updated_at = EXCLUDED.updated_at
                """,
                (self.username, self.password, self.is_admin, now, now),
            )

    @classmethod
    def get_by_username(cls, username: str) -> Optional["User"]:
        clean_username = str(username or "").strip()
        if not clean_username:
            return None
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT username, password_hash, is_admin
                FROM deep_learning_users
                WHERE username = %s
                """,
                (clean_username,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return cls(
            username=str(row.get("username") or ""),
            password=str(row.get("password_hash") or ""),
            is_admin=bool(row.get("is_admin")),
        )

    def add_wrong_answer(
        self,
        question: str,
        std_answer: str,
        user_answer: str,
        timestamp: Any,
        keypoint: str,
        source_doc_id: Optional[str] = None,
    ) -> None:
        asked_at = _parse_ts(timestamp)
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deep_learning_user_wrong_questions (
                    username,
                    keypoint,
                    question,
                    std_answer,
                    user_answer,
                    source_doc_id,
                    asked_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username, keypoint, source_doc_id) DO UPDATE
                SET question = EXCLUDED.question,
                    std_answer = EXCLUDED.std_answer,
                    user_answer = EXCLUDED.user_answer,
                    asked_at = EXCLUDED.asked_at
                """,
                (
                    self.username,
                    str(keypoint or ""),
                    str(question or ""),
                    str(std_answer or ""),
                    str(user_answer or ""),
                    str(source_doc_id or "") or None,
                    asked_at,
                ),
            )
