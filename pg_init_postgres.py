from __future__ import annotations

from pathlib import Path

from pg_support.connection import open_connection
from pg_support.env import load_settings


ROOT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT_DIR / "pg_schema.sql"


def main() -> None:
    settings = load_settings()
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with open_connection(settings=settings, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(schema_sql)
    print(f"PostgreSQL schema initialized from: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
