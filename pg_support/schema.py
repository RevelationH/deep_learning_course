from __future__ import annotations

from pathlib import Path
from threading import Lock

from .connection import open_connection


ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT_DIR / "pg_schema.sql"

_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with open_connection(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(schema_sql)
        _SCHEMA_READY = True
