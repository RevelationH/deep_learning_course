from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
_LOADED = False


def load_local_env() -> None:
    global _LOADED
    if _LOADED:
        return

    for candidate in (ROOT_DIR / ".env.local", ROOT_DIR / ".env.postgres", ROOT_DIR / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)

    _LOADED = True


@dataclass(frozen=True)
class PostgresSettings:
    dsn: str
    connect_timeout: int = 10
    application_name: str = "deep_learning_portal"


def build_dsn() -> str:
    load_local_env()

    direct = (
        os.getenv("DEEP_LEARNING_POSTGRES_DSN", "").strip()
        or os.getenv("POSTGRES_DSN", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if direct:
        return direct

    host = os.getenv("DEEP_LEARNING_PGHOST", "").strip() or os.getenv("PGHOST", "").strip()
    port = os.getenv("DEEP_LEARNING_PGPORT", "").strip() or os.getenv("PGPORT", "").strip() or "5432"
    dbname = os.getenv("DEEP_LEARNING_PGDATABASE", "").strip() or os.getenv("PGDATABASE", "").strip()
    user = os.getenv("DEEP_LEARNING_PGUSER", "").strip() or os.getenv("PGUSER", "").strip()
    password = os.getenv("DEEP_LEARNING_PGPASSWORD", "").strip() or os.getenv("PGPASSWORD", "").strip()
    sslmode = os.getenv("DEEP_LEARNING_PGSSLMODE", "").strip() or os.getenv("PGSSLMODE", "").strip() or "prefer"

    if not all([host, port, dbname, user]):
        raise RuntimeError(
            "PostgreSQL is not configured. Set DEEP_LEARNING_POSTGRES_DSN or "
            "DEEP_LEARNING_PGHOST / DEEP_LEARNING_PGPORT / DEEP_LEARNING_PGDATABASE / "
            "DEEP_LEARNING_PGUSER / DEEP_LEARNING_PGPASSWORD."
        )

    password_part = f":{password}" if password else ""
    return f"postgresql://{user}{password_part}@{host}:{port}/{dbname}?sslmode={sslmode}"


def load_settings() -> PostgresSettings:
    load_local_env()
    timeout_raw = os.getenv("DEEP_LEARNING_PGCONNECT_TIMEOUT", "").strip() or "10"
    app_name = os.getenv("DEEP_LEARNING_PGAPPLICATION_NAME", "").strip() or "deep_learning_portal"
    try:
        timeout = max(int(timeout_raw), 1)
    except ValueError:
        timeout = 10
    return PostgresSettings(dsn=build_dsn(), connect_timeout=timeout, application_name=app_name)
