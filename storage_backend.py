from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple


ROOT_DIR = Path(__file__).resolve().parent
POSTGRES_CONFIG_KEYS = (
    "DEEP_LEARNING_POSTGRES_DSN",
    "POSTGRES_DSN",
    "DATABASE_URL",
    "DEEP_LEARNING_PGHOST",
    "PGHOST",
    "DEEP_LEARNING_PGDATABASE",
    "PGDATABASE",
    "DEEP_LEARNING_PGUSER",
    "PGUSER",
)


def _normalize_backend_name(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"postgres", "postgresql", "pg"}:
        return "postgresql"
    if text in {"firebase", "firestore"}:
        return "firebase"
    return ""


def _env_file_has_postgres_config(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key.strip() in POSTGRES_CONFIG_KEYS and value.strip().strip("'").strip('"'):
                return True
    except Exception:
        return False
    return False


def _postgres_config_present() -> bool:
    for key in POSTGRES_CONFIG_KEYS:
        if os.getenv(key, "").strip():
            return True
    for candidate in (
        ROOT_DIR / ".env.local",
        ROOT_DIR / ".env",
        ROOT_DIR / "deploy.env",
    ):
        if _env_file_has_postgres_config(candidate):
            return True
    return False


def get_storage_backend_name() -> str:
    explicit = _normalize_backend_name(os.getenv("DEEP_LEARNING_STORAGE_BACKEND", ""))
    if explicit:
        return explicit
    if _postgres_config_present():
        return "postgresql"
    return "firebase"


def load_storage_classes() -> Tuple[Any, Any, Any]:
    backend = get_storage_backend_name()
    if backend == "firebase":
        from user import User
        from deep_learning_portal.progress_store import ProgressStore
        from deep_learning_portal.chat_session_store import ChatSessionStore

        return User, ProgressStore, ChatSessionStore

    from pg_support.user_store import User
    from pg_support.progress_store import ProgressStore
    from pg_support.chat_session_store import ChatSessionStore

    return User, ProgressStore, ChatSessionStore


def storage_backend_ready() -> bool:
    try:
        backend = get_storage_backend_name()
        if backend == "firebase":
            from db import firebase_credentials_configured

            return firebase_credentials_configured()

        from pg_support.connection import require_driver
        from pg_support.env import load_settings

        require_driver()
        load_settings()
        return True
    except Exception:
        return False


def storage_backend_notice() -> str:
    backend = get_storage_backend_name()
    if backend == "firebase":
        from db import firebase_credentials_configured

        if firebase_credentials_configured():
            return ""
        return (
            "当前仍在使用 Firebase 存储，但尚未配置 Firebase 服务账号。"
            "请填写 FIREBASE_CREDENTIALS 或 GOOGLE_APPLICATION_CREDENTIALS。"
        )

    try:
        from pg_support.connection import require_driver
        from pg_support.env import load_settings

        require_driver()
        load_settings()
        return ""
    except Exception as exc:
        message = " ".join(str(exc).strip().split())
        if not message:
            message = "Missing PostgreSQL configuration."
        return f"当前已切换为 PostgreSQL 存储，但 PostgreSQL 尚未完成配置：{message}"
