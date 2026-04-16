from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_loader import load_project_env  # type: ignore
from db import fire_db  # type: ignore
from deep_learning_portal.progress_store import (  # type: ignore
    PROGRESS_USER_COLLECTION,
    ATTEMPT_COLLECTION,
    LATEST_ATTEMPT_COLLECTION,
    ARTIFACT_COLLECTION,
    FOLLOW_UP_DOC_ID,
)
from deep_learning_portal.chat_session_store import (  # type: ignore
    SESSION_COLLECTION,
    MESSAGE_COLLECTION,
)

from pg_support.connection import open_connection
from pg_support.progress_store import ProgressStore
from pg_support.user_store import User
from pg_support.chat_session_store import ChatSessionStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate_users(fdb: Any) -> Dict[str, int]:
    migrated_users = 0
    migrated_wrong_questions = 0
    for user_doc in fdb.collection("users").stream():
        payload = user_doc.to_dict() or {}
        username = str(payload.get("username") or user_doc.id or "").strip()
        if not username:
            continue
        User(
            username=username,
            password=str(payload.get("password") or ""),
            is_admin=bool(payload.get("is_admin")),
        ).save()
        migrated_users += 1

        user = User.get_by_username(username)
        if not user:
            continue
        for keypoint_doc in user_doc.reference.collection("wrong_questions").stream():
            keypoint = str(keypoint_doc.id or "")
            for wrong_doc in keypoint_doc.reference.collection("questions").stream():
                item = wrong_doc.to_dict() or {}
                user.add_wrong_answer(
                    question=str(item.get("question") or ""),
                    std_answer=str(item.get("std_answer") or ""),
                    user_answer=str(item.get("user_answer") or ""),
                    timestamp=item.get("timestamp"),
                    keypoint=keypoint,
                    source_doc_id=wrong_doc.id,
                )
                migrated_wrong_questions += 1

    return {"users": migrated_users, "wrong_questions": migrated_wrong_questions}


def migrate_progress(fdb: Any) -> Dict[str, int]:
    store = ProgressStore()
    migrated_profiles = 0
    migrated_attempts = 0
    migrated_latest_attempts = 0
    migrated_followups = 0

    for profile_doc in fdb.collection(PROGRESS_USER_COLLECTION).stream():
        user_id = str(profile_doc.id or "").strip()
        if not user_id:
            continue
        payload = profile_doc.to_dict() or {}
        store.upsert_profile_snapshot(user_id, payload)
        migrated_profiles += 1

        for attempt_doc in profile_doc.reference.collection(ATTEMPT_COLLECTION).stream():
            store.insert_attempt_snapshot(user_id, attempt_doc.to_dict() or {}, source_attempt_id=attempt_doc.id)
            migrated_attempts += 1

        for latest_doc in profile_doc.reference.collection(LATEST_ATTEMPT_COLLECTION).stream():
            store.upsert_latest_attempt_snapshot(user_id, latest_doc.to_dict() or {})
            migrated_latest_attempts += 1

        followup_doc = profile_doc.reference.collection(ARTIFACT_COLLECTION).document(FOLLOW_UP_DOC_ID).get()
        if followup_doc.exists:
            store.set_followup_snapshot(user_id, followup_doc.to_dict() or {})
            migrated_followups += len((followup_doc.to_dict() or {}).get("items") or [])

    return {
        "profiles": migrated_profiles,
        "attempts": migrated_attempts,
        "latest_attempts": migrated_latest_attempts,
        "followups": migrated_followups,
    }


def migrate_chat(fdb: Any) -> Dict[str, int]:
    store = ChatSessionStore()
    migrated_sessions = 0
    migrated_messages = 0

    for user_doc in fdb.collection("users").stream():
        user_id = str(user_doc.id or "").strip()
        if not user_id:
            continue
        for session_doc in user_doc.reference.collection(SESSION_COLLECTION).stream():
            session_payload = session_doc.to_dict() or {}
            session_id = str(session_doc.id or "").strip()
            if not session_id:
                continue
            store.upsert_session_snapshot(user_id, session_id, session_payload)
            migrated_sessions += 1

            for message_doc in session_doc.reference.collection(MESSAGE_COLLECTION).stream():
                message_payload = message_doc.to_dict() or {}
                store.upsert_message_snapshot(user_id, session_id, message_payload, source_message_id=message_doc.id)
                migrated_messages += 1

    return {"sessions": migrated_sessions, "messages": migrated_messages}


def write_migration_meta(summary: Dict[str, Any]) -> None:
    with open_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO deep_learning_migration_meta (migration_key, payload, updated_at)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (migration_key) DO UPDATE
            SET payload = EXCLUDED.payload,
                updated_at = EXCLUDED.updated_at
            """,
            ("firebase_to_postgres_v1", json.dumps(summary, ensure_ascii=False), datetime.now(timezone.utc)),
        )


def main() -> None:
    load_project_env()
    fdb = fire_db()
    summary = {
        "ran_at": _now_iso(),
        "users": migrate_users(fdb),
        "progress": migrate_progress(fdb),
        "chat": migrate_chat(fdb),
    }
    write_migration_meta(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
