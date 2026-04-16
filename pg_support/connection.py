from __future__ import annotations

import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None
    dict_row = None

try:
    from psycopg_pool import ConnectionPool
except Exception:
    ConnectionPool = None

from .env import PostgresSettings, load_settings


class MissingPostgresDriverError(RuntimeError):
    pass


_POOL_LOCK = Lock()
_POOL: Optional["ConnectionPool[Any]"] = None
_POOL_KEY: Optional[tuple[Any, ...]] = None


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(value, minimum)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    return max(value, minimum)


def require_driver() -> None:
    if psycopg is None or dict_row is None:
        raise MissingPostgresDriverError(
            "psycopg is not installed. Run: pip install -r "
            "D:\\digital_human\\deep_learning\\requirements.txt"
        )


def pool_enabled() -> bool:
    return os.getenv("DEEP_LEARNING_DISABLE_PGPOOL", "").strip() != "1"


def _pool_config(settings: PostgresSettings) -> dict[str, Any]:
    max_size = _env_int("DEEP_LEARNING_PGPOOL_MAX_SIZE", 24, minimum=1)
    min_size = _env_int("DEEP_LEARNING_PGPOOL_MIN_SIZE", min(4, max_size), minimum=1)
    min_size = min(min_size, max_size)
    return {
        "min_size": min_size,
        "max_size": max_size,
        "timeout": _env_float("DEEP_LEARNING_PGPOOL_WAIT_TIMEOUT", max(settings.connect_timeout, 15), minimum=1.0),
        "max_waiting": _env_int("DEEP_LEARNING_PGPOOL_MAX_WAITING", max_size * 4, minimum=0),
        "max_lifetime": _env_float("DEEP_LEARNING_PGPOOL_MAX_LIFETIME", 900.0, minimum=60.0),
        "max_idle": _env_float("DEEP_LEARNING_PGPOOL_MAX_IDLE", 180.0, minimum=30.0),
        "reconnect_timeout": _env_float("DEEP_LEARNING_PGPOOL_RECONNECT_TIMEOUT", 120.0, minimum=5.0),
        "num_workers": _env_int("DEEP_LEARNING_PGPOOL_NUM_WORKERS", 3, minimum=1),
    }


def _pool_key(settings: PostgresSettings) -> tuple[Any, ...]:
    config = _pool_config(settings)
    return (
        settings.dsn,
        settings.connect_timeout,
        settings.application_name,
        config["min_size"],
        config["max_size"],
        config["timeout"],
        config["max_waiting"],
        config["max_lifetime"],
        config["max_idle"],
        config["reconnect_timeout"],
        config["num_workers"],
    )


def _build_pool(settings: PostgresSettings) -> "ConnectionPool[Any]":
    require_driver()
    if ConnectionPool is None:
        raise MissingPostgresDriverError(
            "psycopg_pool is not installed. Run: pip install -r "
            "D:\\digital_human\\deep_learning\\requirements.txt"
        )
    config = _pool_config(settings)
    pool = ConnectionPool(
        conninfo=settings.dsn,
        kwargs={
            "connect_timeout": settings.connect_timeout,
            "application_name": settings.application_name,
            "row_factory": dict_row,
        },
        min_size=config["min_size"],
        max_size=config["max_size"],
        timeout=config["timeout"],
        max_waiting=config["max_waiting"],
        max_lifetime=config["max_lifetime"],
        max_idle=config["max_idle"],
        reconnect_timeout=config["reconnect_timeout"],
        num_workers=config["num_workers"],
        open=False,
        name="deep_learning_pg_pool",
    )
    pool.open(wait=True, timeout=config["timeout"])
    return pool


def get_pool(*, settings: Optional[PostgresSettings] = None) -> "ConnectionPool[Any]":
    if not pool_enabled():
        raise RuntimeError("PostgreSQL connection pool is disabled.")
    cfg = settings or load_settings()
    key = _pool_key(cfg)
    global _POOL, _POOL_KEY
    with _POOL_LOCK:
        if _POOL is None or _POOL_KEY != key:
            if _POOL is not None:
                try:
                    _POOL.close()
                except Exception:
                    pass
            _POOL = _build_pool(cfg)
            _POOL_KEY = key
        return _POOL


def pool_status(*, settings: Optional[PostgresSettings] = None) -> dict[str, Any]:
    if not pool_enabled():
        return {"enabled": False, "available": False}
    try:
        pool = get_pool(settings=settings)
        return {"enabled": True, "available": True, **pool.get_stats()}
    except Exception as exc:
        return {"enabled": True, "available": False, "error": " ".join(str(exc).split())[:220]}


def close_pool() -> None:
    global _POOL, _POOL_KEY
    with _POOL_LOCK:
        if _POOL is not None:
            try:
                _POOL.close()
            finally:
                _POOL = None
                _POOL_KEY = None


@contextmanager
def open_connection(
    *,
    settings: Optional[PostgresSettings] = None,
    autocommit: bool = False,
) -> Iterator["psycopg.Connection[Any]"]:
    require_driver()
    cfg = settings or load_settings()
    conn: Optional["psycopg.Connection[Any]"] = None
    borrowed_from_pool = False
    pool: Optional["ConnectionPool[Any]"] = None

    if pool_enabled():
        pool = get_pool(settings=cfg)
        conn = pool.getconn(timeout=_pool_config(cfg)["timeout"])
        borrowed_from_pool = True
    else:
        conn = psycopg.connect(
            cfg.dsn,
            row_factory=dict_row,
            autocommit=False,
            connect_timeout=cfg.connect_timeout,
            application_name=cfg.application_name,
        )

    try:
        conn.row_factory = dict_row
        conn.autocommit = autocommit
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if conn is not None and not autocommit:
            conn.rollback()
        raise
    finally:
        if conn is None:
            return
        if borrowed_from_pool and pool is not None:
            try:
                if autocommit:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                conn.autocommit = False
                conn.row_factory = dict_row
                pool.putconn(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            conn.close()
