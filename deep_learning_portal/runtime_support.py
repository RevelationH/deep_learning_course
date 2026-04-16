from __future__ import annotations

import copy
import json
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from typing import Any, Callable, Dict, Iterator, Optional


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


class ChatGateBusyError(RuntimeError):
    def __init__(self, waited_seconds: float) -> None:
        super().__init__("Chat concurrency gate timed out.")
        self.waited_seconds = waited_seconds


class ChatConcurrencyGate:
    def __init__(self) -> None:
        self.max_concurrent = _env_int("DEEP_LEARNING_CHAT_MAX_CONCURRENCY", 6, minimum=1)
        self.wait_timeout = _env_float("DEEP_LEARNING_CHAT_WAIT_TIMEOUT_SEC", 45.0, minimum=1.0)
        self._semaphore = BoundedSemaphore(self.max_concurrent)
        self._lock = Lock()
        self._waiting = 0
        self._inflight = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        started = time.monotonic()
        with self._lock:
            self._waiting += 1
        acquired = self._semaphore.acquire(timeout=self.wait_timeout)
        with self._lock:
            self._waiting -= 1
            if acquired:
                self._inflight += 1
        if not acquired:
            raise ChatGateBusyError(time.monotonic() - started)
        try:
            yield
        finally:
            with self._lock:
                self._inflight = max(self._inflight - 1, 0)
            self._semaphore.release()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "wait_timeout": self.wait_timeout,
                "inflight": self._inflight,
                "waiting": self._waiting,
            }


class LearningReportCache:
    def __init__(self) -> None:
        self.ttl_seconds = _env_float("DEEP_LEARNING_REPORT_CACHE_TTL_SEC", 300.0, minimum=30.0)
        self.max_entries = _env_int("DEEP_LEARNING_REPORT_CACHE_MAX_ENTRIES", 256, minimum=16)
        self._entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = Lock()
        self._build_locks: Dict[str, Lock] = {}

    def _signature_key(self, signature: Dict[str, Any]) -> str:
        return json.dumps(signature or {}, ensure_ascii=False, sort_keys=True)

    def get_or_build(
        self,
        user_id: str,
        signature: Dict[str, Any],
        builder: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        now = time.monotonic()
        signature_key = self._signature_key(signature)
        with self._lock:
            entry = self._entries.get(user_id)
            if entry and entry["signature_key"] == signature_key and entry["expires_at"] > now:
                self._entries.move_to_end(user_id)
                return copy.deepcopy(entry["value"])
            build_lock = self._build_locks.setdefault(user_id, Lock())

        with build_lock:
            now = time.monotonic()
            with self._lock:
                entry = self._entries.get(user_id)
                if entry and entry["signature_key"] == signature_key and entry["expires_at"] > now:
                    self._entries.move_to_end(user_id)
                    return copy.deepcopy(entry["value"])

            value = builder()
            cached_value = copy.deepcopy(value)
            with self._lock:
                self._entries[user_id] = {
                    "signature_key": signature_key,
                    "expires_at": time.monotonic() + self.ttl_seconds,
                    "value": cached_value,
                }
                self._entries.move_to_end(user_id)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            return value

    def warm(
        self,
        user_id: str,
        signature: Dict[str, Any],
        builder: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self.get_or_build(user_id, signature, builder)

    def invalidate(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            if user_id is None:
                self._entries.clear()
                return
            self._entries.pop(user_id, None)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "cached_users": len(self._entries),
            }
