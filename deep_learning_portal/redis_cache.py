from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any, Optional

try:
    import redis
except Exception:
    redis = None


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(value, minimum)


class RedisJsonCache:
    def __init__(self) -> None:
        self.url = (
            os.getenv("DEEP_LEARNING_REDIS_URL", "").strip()
            or os.getenv("REDIS_URL", "").strip()
        )
        self.prefix = os.getenv("DEEP_LEARNING_REDIS_PREFIX", "deep-learning:").strip() or "deep-learning:"
        self.default_ttl = _env_int("DEEP_LEARNING_REDIS_DEFAULT_TTL_SEC", 900, minimum=30)
        self._client = None
        self._lock = Lock()
        self._last_error = ""

    def available(self) -> bool:
        if not self.url or redis is None:
            return False
        return self.client() is not None

    def client(self) -> Optional[Any]:
        if not self.url or redis is None:
            return None
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                client = redis.Redis.from_url(self.url, decode_responses=True)
                client.ping()
                self._client = client
                self._last_error = ""
            except Exception as exc:
                self._last_error = " ".join(str(exc).split())[:220]
                self._client = None
            return self._client

    def make_key(self, *parts: Any) -> str:
        normalized = ":".join(str(part or "").strip() for part in parts if str(part or "").strip())
        return f"{self.prefix}{normalized}" if normalized else self.prefix.rstrip(":")

    def get_json(self, key: str) -> Optional[Any]:
        client = self.client()
        if client is None:
            return None
        try:
            raw = client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as exc:
            self._last_error = " ".join(str(exc).split())[:220]
            return None

    def set_json(self, key: str, value: Any, *, ttl: Optional[int] = None) -> bool:
        client = self.client()
        if client is None:
            return False
        try:
            client.setex(key, int(ttl or self.default_ttl), json.dumps(value, ensure_ascii=False))
            return True
        except Exception as exc:
            self._last_error = " ".join(str(exc).split())[:220]
            return False

    def delete(self, key: str) -> None:
        client = self.client()
        if client is None:
            return
        try:
            client.delete(key)
        except Exception as exc:
            self._last_error = " ".join(str(exc).split())[:220]

    def stats(self) -> dict[str, Any]:
        return {
            "configured": bool(self.url),
            "available": self.available(),
            "prefix": self.prefix,
            "default_ttl": self.default_ttl,
            "error": self._last_error,
        }
