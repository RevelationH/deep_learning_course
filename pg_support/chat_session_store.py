from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .connection import open_connection


NEW_CHAT_TITLE = "\u65b0\u5bf9\u8bdd"


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


TITLE_SANITIZE_RE = re.compile(r"[\r\n\t]+")
COURSE_SOURCE_TOKEN_RE = re.compile(r"@@COURSE_SOURCE_\d+@@")
COURSE_SOURCE_LINE_RE = re.compile(r"(?is)(?:\n|\r|\s)*(?:Course source:|\u8bb2\u4e49\u6765\u6e90\uff1a).*$")
USED_SOURCES_RE = re.compile(r"(?is)(?:^|\n)\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\].*$")


def _clean_title(text: str, fallback: str = NEW_CHAT_TITLE) -> str:
    compact = TITLE_SANITIZE_RE.sub(" ", str(text or "").strip())
    compact = " ".join(compact.split()).strip(" -|:;,.\"'")
    if not compact:
        return fallback
    compact = compact[:56].strip()
    return compact or fallback


def _preview_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _clean_session_summary(text: str, limit: int = 560) -> str:
    compact = " ".join(str(text or "").strip().split())
    if not compact:
        return ""
    return compact[:limit].strip()


def _clean_active_topic(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").strip().split())
    if not compact:
        return ""
    return compact[:limit].strip(" -|:;,.")


def _sanitize_assistant_content(text: str, citations: Optional[List[Dict[str, Any]]] = None) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    candidate = COURSE_SOURCE_TOKEN_RE.sub("", candidate)
    candidate = USED_SOURCES_RE.sub("", candidate)
    candidate = COURSE_SOURCE_LINE_RE.sub("", candidate)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip().strip(",").strip()
    return candidate


class ChatSessionStore:
    def list_sessions(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at FROM deep_learning_chat_sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                (str(user_id or "").strip(), max(int(limit), 1)),
            )
            rows = cur.fetchall() or []
        return [self._serialize_session(row["session_id"], row) for row in rows]

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at FROM deep_learning_chat_sessions WHERE user_id = %s AND session_id = %s",
                (str(user_id or "").strip(), str(session_id or "").strip()),
            )
            row = cur.fetchone()
        if not row:
            return None
        return self._serialize_session(row["session_id"], row)

    def create_session(self, user_id: str, title: str = NEW_CHAT_TITLE) -> Dict[str, Any]:
        now = _now()
        session_id = uuid4().hex
        payload = {"title": _clean_title(title), "title_generated": False, "created_at": now, "updated_at": now, "last_message_preview": "", "message_count": 0, "course": "deep_learning", "session_summary": "", "active_topic": "", "summary_updated_at": now}
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO deep_learning_chat_sessions (session_id, user_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (session_id, str(user_id or "").strip(), payload["title"], payload["title_generated"], payload["created_at"], payload["updated_at"], payload["last_message_preview"], payload["message_count"], payload["course"], payload["session_summary"], payload["active_topic"], payload["summary_updated_at"]))
        return self._serialize_session(session_id, payload)

    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id:
            existing = self.get_session(user_id, session_id)
            if existing:
                return existing
        return self.create_session(user_id)

    def get_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT source_message_id, role, content, citations, created_at, order_index FROM deep_learning_chat_messages WHERE user_id = %s AND session_id = %s ORDER BY order_index ASC", (str(user_id or "").strip(), str(session_id or "").strip()))
            rows = cur.fetchall() or []
        return [{"message_id": str(row.get("source_message_id") or ""), "role": str(row.get("role") or "assistant"), "content": _sanitize_assistant_content(row.get("content"), row.get("citations")) if str(row.get("role") or "assistant") == "assistant" else str(row.get("content") or ""), "citations": list(row.get("citations") or []), "created_at": _iso(row.get("created_at")), "order_index": int(row.get("order_index") or 0)} for row in rows]

    def recent_history_for_model(self, user_id: str, session_id: str, limit: int = 8) -> List[Dict[str, str]]:
        rows = self.get_messages(user_id, session_id)
        trimmed = rows[-limit:]
        return [{"role": row["role"], "content": row["content"]} for row in trimmed if row.get("content")]

    def append_exchange(self, user_id: str, session_id: str, user_message: str, assistant_message: str, citations: List[Dict[str, Any]], mode: str, session_title: Optional[str] = None, session_summary: Optional[str] = None, active_topic: Optional[str] = None) -> Dict[str, Any]:
        clean_user_id = str(user_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT session_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at FROM deep_learning_chat_sessions WHERE user_id = %s AND session_id = %s FOR UPDATE", (clean_user_id, clean_session_id))
            row = cur.fetchone()
            if not row:
                row = {"session_id": clean_session_id, "title": NEW_CHAT_TITLE, "title_generated": False, "created_at": now, "updated_at": now, "last_message_preview": "", "message_count": 0, "course": "deep_learning", "session_summary": "", "active_topic": "", "summary_updated_at": now}
                cur.execute("INSERT INTO deep_learning_chat_sessions (session_id, user_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (clean_session_id, clean_user_id, row["title"], row["title_generated"], row["created_at"], row["updated_at"], row["last_message_preview"], row["message_count"], row["course"], row["session_summary"], row["active_topic"], row["summary_updated_at"]))
            message_count = int(row.get("message_count") or 0)
            base_title = _clean_title(str(row.get("title") or ""), fallback=NEW_CHAT_TITLE)
            title_generated = bool(row.get("title_generated"))
            if session_title and (message_count == 0 or base_title == NEW_CHAT_TITLE or not title_generated):
                base_title = _clean_title(session_title, fallback=NEW_CHAT_TITLE)
                title_generated = True
            sanitized_assistant = _sanitize_assistant_content(assistant_message, citations)
            user_order = message_count + 1
            assistant_order = message_count + 2
            cur.execute("INSERT INTO deep_learning_chat_messages (session_id, user_id, source_message_id, role, content, citations, created_at, order_index, mode) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s) ON CONFLICT (session_id, order_index) DO UPDATE SET role = EXCLUDED.role, content = EXCLUDED.content, citations = EXCLUDED.citations, created_at = EXCLUDED.created_at, mode = EXCLUDED.mode", (clean_session_id, clean_user_id, f"m{user_order:06d}", "user", str(user_message or ""), json.dumps([]), now, user_order, ""))
            cur.execute("INSERT INTO deep_learning_chat_messages (session_id, user_id, source_message_id, role, content, citations, created_at, order_index, mode) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s) ON CONFLICT (session_id, order_index) DO UPDATE SET role = EXCLUDED.role, content = EXCLUDED.content, citations = EXCLUDED.citations, created_at = EXCLUDED.created_at, mode = EXCLUDED.mode", (clean_session_id, clean_user_id, f"m{assistant_order:06d}", "assistant", sanitized_assistant, json.dumps(list(citations or []), ensure_ascii=False), now, assistant_order, str(mode or "")))
            next_summary = _clean_session_summary(session_summary) if session_summary is not None else _clean_session_summary(row.get("session_summary") or "")
            next_topic = _clean_active_topic(active_topic) if active_topic is not None else _clean_active_topic(row.get("active_topic") or "")
            cur.execute("UPDATE deep_learning_chat_sessions SET title = %s, title_generated = %s, updated_at = %s, last_message_preview = %s, message_count = %s, session_summary = %s, active_topic = %s, summary_updated_at = %s WHERE session_id = %s AND user_id = %s", (base_title, title_generated, now, _preview_text(sanitized_assistant or user_message), assistant_order, next_summary, next_topic, now if (session_summary is not None or active_topic is not None) else row.get("summary_updated_at") or now, clean_session_id, clean_user_id))
            cur.execute("SELECT session_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at FROM deep_learning_chat_sessions WHERE user_id = %s AND session_id = %s", (clean_user_id, clean_session_id))
            updated = cur.fetchone()
        return self._serialize_session(clean_session_id, updated or row)

    def set_session_title(self, user_id: str, session_id: str, title: str, *, generated: bool = True) -> Optional[Dict[str, Any]]:
        clean_user_id = str(user_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        now = _now()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE deep_learning_chat_sessions SET title = %s, title_generated = %s, updated_at = %s WHERE user_id = %s AND session_id = %s RETURNING session_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at", (_clean_title(title, fallback=NEW_CHAT_TITLE), bool(generated), now, clean_user_id, clean_session_id))
            row = cur.fetchone()
        if not row:
            return None
        return self._serialize_session(clean_session_id, row)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        clean_user_id = str(user_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM deep_learning_chat_sessions WHERE user_id = %s AND session_id = %s", (clean_user_id, clean_session_id))
            return cur.rowcount > 0

    def upsert_session_snapshot(self, user_id: str, session_id: str, payload: Dict[str, Any]) -> None:
        clean_user_id = str(user_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO deep_learning_chat_sessions (session_id, user_id, title, title_generated, created_at, updated_at, last_message_preview, message_count, course, session_summary, active_topic, summary_updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (session_id) DO UPDATE SET user_id = EXCLUDED.user_id, title = EXCLUDED.title, title_generated = EXCLUDED.title_generated, created_at = EXCLUDED.created_at, updated_at = EXCLUDED.updated_at, last_message_preview = EXCLUDED.last_message_preview, message_count = EXCLUDED.message_count, course = EXCLUDED.course, session_summary = EXCLUDED.session_summary, active_topic = EXCLUDED.active_topic, summary_updated_at = EXCLUDED.summary_updated_at",
                (
                    clean_session_id,
                    clean_user_id,
                    _clean_title(str(payload.get("title") or NEW_CHAT_TITLE)),
                    bool(payload.get("title_generated")),
                    _parse_ts(payload.get("created_at")) or _now(),
                    _parse_ts(payload.get("updated_at")) or _now(),
                    _preview_text(str(payload.get("last_message_preview") or "")),
                    int(payload.get("message_count") or 0),
                    str(payload.get("course") or "deep_learning"),
                    _clean_session_summary(str(payload.get("session_summary") or "")),
                    _clean_active_topic(str(payload.get("active_topic") or "")),
                    _parse_ts(payload.get("summary_updated_at")) or _now(),
                ),
            )

    def upsert_message_snapshot(self, user_id: str, session_id: str, payload: Dict[str, Any], source_message_id: Optional[str] = None) -> None:
        clean_user_id = str(user_id or "").strip()
        clean_session_id = str(session_id or "").strip()
        role = str(payload.get("role") or "assistant")
        content = _sanitize_assistant_content(payload.get("content"), payload.get("citations")) if role == "assistant" else str(payload.get("content") or "")
        with open_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO deep_learning_chat_messages (session_id, user_id, source_message_id, role, content, citations, created_at, order_index, mode) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s) ON CONFLICT (session_id, order_index) DO UPDATE SET user_id = EXCLUDED.user_id, source_message_id = EXCLUDED.source_message_id, role = EXCLUDED.role, content = EXCLUDED.content, citations = EXCLUDED.citations, created_at = EXCLUDED.created_at, mode = EXCLUDED.mode",
                (
                    clean_session_id,
                    clean_user_id,
                    str(source_message_id or "") or None,
                    role,
                    content,
                    json.dumps(list(payload.get("citations") or []), ensure_ascii=False),
                    _parse_ts(payload.get("created_at")) or _now(),
                    int(payload.get("order_index") or 0),
                    str(payload.get("mode") or ""),
                ),
            )

    def _serialize_session(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"session_id": session_id, "title": str(payload.get("title") or NEW_CHAT_TITLE), "title_generated": bool(payload.get("title_generated")), "created_at": _iso(payload.get("created_at")), "updated_at": _iso(payload.get("updated_at")), "last_message_preview": str(payload.get("last_message_preview") or ""), "message_count": int(payload.get("message_count") or 0), "course": str(payload.get("course") or "deep_learning"), "session_summary": _clean_session_summary(str(payload.get("session_summary") or "")), "active_topic": _clean_active_topic(str(payload.get("active_topic") or "")), "summary_updated_at": _iso(payload.get("summary_updated_at"))}
