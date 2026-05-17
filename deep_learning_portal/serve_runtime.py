from __future__ import annotations

import os
from typing import Any

from waitress import serve
from runtime_tuning import default_waitress_connection_limit, default_waitress_threads


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(value, minimum)


def serve_app(app: Any, *, host: str, port: int) -> None:
    threads = _env_int("DEEP_LEARNING_WAITRESS_THREADS", default_waitress_threads(), minimum=4)
    connection_limit = _env_int("DEEP_LEARNING_WAITRESS_CONNECTION_LIMIT", default_waitress_connection_limit(), minimum=64)
    channel_timeout = _env_int("DEEP_LEARNING_WAITRESS_CHANNEL_TIMEOUT", 180, minimum=30)
    cleanup_interval = _env_int("DEEP_LEARNING_WAITRESS_CLEANUP_INTERVAL", 30, minimum=5)
    asyncore_loop_timeout = max(float(os.getenv("DEEP_LEARNING_WAITRESS_LOOP_TIMEOUT", "1.0") or 1.0), 0.1)

    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        connection_limit=connection_limit,
        channel_timeout=channel_timeout,
        cleanup_interval=cleanup_interval,
        asyncore_loop_timeout=asyncore_loop_timeout,
        ident="deep-learning-portal",
        expose_tracebacks=False,
    )
