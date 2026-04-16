from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from deep_learning_portal.redis_cache import RedisJsonCache
from deep_learning_portal.student_analytics import build_attempt_state, build_learning_report_context
from pg_support.chat_job_store import ChatJobStore
from pg_support.report_snapshot_store import LearningReportSnapshotStore


logger = logging.getLogger(__name__)


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


def _default_chat_worker_count() -> int:
    cpu_total = max(int(os.cpu_count() or 4), 1)
    if cpu_total <= 4:
        return 4
    return min(max(cpu_total, 4), 8)


def signature_to_key(signature: Dict[str, Any]) -> str:
    payload = json.dumps(signature or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


class ChatJobAlreadyActiveError(RuntimeError):
    def __init__(self, job: Dict[str, Any]) -> None:
        super().__init__("A chat job is already active for this user.")
        self.job = job


class ChatQueueFullError(RuntimeError):
    def __init__(
        self,
        queue_size: int,
        max_queue_size: int,
        *,
        reason: str = "queue_full",
        estimated_wait_seconds: int = 0,
        max_estimated_wait_seconds: int = 0,
        retry_after_seconds: int = 0,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__("Chat queue is full.")
        self.queue_size = queue_size
        self.max_queue_size = max_queue_size
        self.reason = str(reason or "queue_full")
        self.estimated_wait_seconds = max(int(estimated_wait_seconds or 0), 0)
        self.max_estimated_wait_seconds = max(int(max_estimated_wait_seconds or 0), 0)
        self.retry_after_seconds = max(int(retry_after_seconds or 0), 0)
        self.snapshot = dict(snapshot or {})


class AsyncChatService:
    def __init__(
        self,
        *,
        kb: Any,
        chat_pipeline: Any,
        chat_store_factory: Callable[[], Any],
        redis_cache: Optional[RedisJsonCache] = None,
    ) -> None:
        self.kb = kb
        self.chat_pipeline = chat_pipeline
        self.chat_store_factory = chat_store_factory
        self.redis_cache = redis_cache or RedisJsonCache()
        self.job_store = ChatJobStore()
        self.worker_count = _env_int("DEEP_LEARNING_CHAT_WORKERS", _default_chat_worker_count(), minimum=1)
        self.max_queue_size = _env_int(
            "DEEP_LEARNING_CHAT_QUEUE_MAX_SIZE",
            max(self.worker_count * 24, 96),
            minimum=max(self.worker_count * 4, 8),
        )
        self.poll_interval = _env_float("DEEP_LEARNING_CHAT_WORKER_POLL_SEC", 0.8, minimum=0.1)
        self.job_retention_sec = _env_int("DEEP_LEARNING_CHAT_JOB_RETENTION_SEC", 21600, minimum=600)
        self.stale_running_sec = _env_int("DEEP_LEARNING_CHAT_RUNNING_STALE_SEC", 900, minimum=60)
        self.cache_ttl = _env_int("DEEP_LEARNING_CHAT_JOB_CACHE_TTL_SEC", 3600, minimum=60)
        self.default_estimated_duration_sec = _env_float("DEEP_LEARNING_CHAT_DEFAULT_ESTIMATED_DURATION_SEC", 18.0, minimum=1.0)
        self.max_estimated_wait_sec = _env_int("DEEP_LEARNING_CHAT_MAX_ESTIMATED_WAIT_SEC", 150, minimum=15)
        self.min_retry_after_sec = _env_int("DEEP_LEARNING_CHAT_RETRY_AFTER_SEC", 12, minimum=1)
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            self.job_store.fail_stale_running_jobs(self.stale_running_sec)
            self.job_store.prune_finished_jobs(self.job_retention_sec)
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"deep-learning-chat-worker-{index + 1}",
                    args=(index + 1,),
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def enqueue(self, *, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        active_job = self.job_store.get_active_job_for_user(user_id)
        if active_job:
            raise ChatJobAlreadyActiveError(self._decorate_job(active_job))
        metrics = self._build_queue_snapshot(self.job_store.count_active_jobs())
        queue_size = int(metrics.get("active_total") or 0)
        estimated_wait_seconds = self._estimate_wait_seconds_for_position(
            queue_position=int(metrics.get("queued") or 0) + 1,
            metrics=metrics,
        )
        retry_after_seconds = self._estimate_retry_after_seconds(estimated_wait_seconds, metrics=metrics)
        if queue_size >= self.max_queue_size:
            raise ChatQueueFullError(
                queue_size=queue_size,
                max_queue_size=self.max_queue_size,
                reason="queue_full",
                estimated_wait_seconds=estimated_wait_seconds,
                max_estimated_wait_seconds=self.max_estimated_wait_sec,
                retry_after_seconds=retry_after_seconds,
                snapshot=metrics,
            )
        if estimated_wait_seconds > self.max_estimated_wait_sec:
            raise ChatQueueFullError(
                queue_size=queue_size,
                max_queue_size=self.max_queue_size,
                reason="wait_too_long",
                estimated_wait_seconds=estimated_wait_seconds,
                max_estimated_wait_seconds=self.max_estimated_wait_sec,
                retry_after_seconds=retry_after_seconds,
                snapshot=metrics,
            )
        job = self.job_store.enqueue(
            user_id=user_id,
            session_id=session_id,
            message=message,
            request_meta={"course": "deep_learning"},
        )
        decorated = self._decorate_job(job)
        self._cache_job(decorated)
        return decorated

    def get_job(self, user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        cached = self._read_cached_job(job_id)
        if cached and str(cached.get("user_id") or "") == str(user_id or "").strip():
            if str(cached.get("status") or "") in {"completed", "failed"}:
                return cached
        job = self.job_store.get_job(user_id, job_id)
        if job:
            decorated = self._decorate_job(job)
            self._cache_job(decorated)
            return decorated
        if cached and str(cached.get("user_id") or "") == str(user_id or "").strip():
            return cached
        return None

    def snapshot(self) -> Dict[str, Any]:
        metrics = self._build_queue_snapshot(self.job_store.count_active_jobs())
        with self._inflight_lock:
            inflight = self._inflight
        metrics["inflight"] = inflight
        return metrics

    def _worker_loop(self, worker_number: int) -> None:
        worker_id = f"chat-worker-{worker_number}"
        idle_cycles = 0
        while not self._stop_event.is_set():
            job = self.job_store.claim_next_job(worker_id)
            if not job:
                idle_cycles += 1
                if idle_cycles % 120 == 0:
                    try:
                        self.job_store.prune_finished_jobs(self.job_retention_sec)
                    except Exception:
                        logger.exception("Failed to prune finished chat jobs.")
                time.sleep(self.poll_interval)
                continue
            idle_cycles = 0
            with self._inflight_lock:
                self._inflight += 1
            try:
                completed = self._process_job(job, worker_id=worker_id)
                self._cache_job(completed)
            except Exception as exc:
                logger.exception("Chat worker failed for job %s.", job.get("job_id"))
                failed = self.job_store.mark_failed(job.get("job_id", ""), str(exc)) or self._decorate_job(
                    {"job_id": job.get("job_id"), "user_id": job.get("user_id"), "status": "failed", "error_message": str(exc)}
                )
                self._cache_job(self._decorate_job(failed))
            finally:
                with self._inflight_lock:
                    self._inflight = max(self._inflight - 1, 0)

    def _process_job(self, job: Dict[str, Any], *, worker_id: str) -> Dict[str, Any]:
        user_id = str(job.get("user_id") or "").strip()
        session_id = str(job.get("session_id") or "").strip()
        message = str(job.get("message") or "").strip()
        chat_store = self.chat_store_factory()
        existing_session = chat_store.get_session(user_id, session_id) if session_id else None
        history = chat_store.recent_history_for_model(user_id, session_id, limit=8) if session_id else []
        answer = self.chat_pipeline.answer(message, history=history, top_k=5, session_memory=existing_session or {})
        generated_title = self.kb.suggest_session_title(message, answer["answer"])
        citations = list(answer.get("citations") or [])
        session_data = chat_store.get_or_create_session(user_id, session_id=session_id)
        session_data = chat_store.append_exchange(
            user_id=user_id,
            session_id=session_data["session_id"],
            user_message=message,
            assistant_message=answer["answer"],
            citations=citations,
            mode=str(answer.get("mode") or "assistant"),
            session_title=generated_title,
            session_summary=str((answer.get("session_memory") or {}).get("session_summary") or ""),
            active_topic=str((answer.get("session_memory") or {}).get("active_topic") or ""),
        )
        result = {
            "answer": answer["answer"],
            "citations": citations,
            "related_kps": list(answer.get("related_kps") or []),
            "mode": str(answer.get("mode") or "assistant"),
            "actions": {"quiz_url": "/quiz_deep_learning"},
            "session": session_data,
        }
        completed = self.job_store.mark_completed(job["job_id"], result) or {
            **job,
            "status": "completed",
            "result": result,
            "worker_id": worker_id,
        }
        return self._decorate_job(completed)

    def _cache_job(self, job: Dict[str, Any]) -> None:
        key = self.redis_cache.make_key("chat-job", job.get("job_id"))
        self.redis_cache.set_json(key, job, ttl=self.cache_ttl)

    def _read_cached_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = self.redis_cache.make_key("chat-job", job_id)
        payload = self.redis_cache.get_json(key)
        return dict(payload or {}) if isinstance(payload, dict) else None

    def _decorate_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        decorated = dict(job or {})
        status = str(decorated.get("status") or "queued")
        metrics = self._build_queue_snapshot(self.job_store.count_active_jobs())
        decorated["queue_position"] = self.job_store.queue_position(decorated.get("job_id", "")) if status == "queued" else 0
        decorated["estimated_duration_seconds"] = int(metrics.get("estimated_duration_seconds") or 0)
        decorated["estimated_wait_seconds"] = (
            self._estimate_wait_seconds_for_position(int(decorated.get("queue_position") or 0), metrics=metrics)
            if status == "queued"
            else 0
        )
        decorated["retry_after_seconds"] = self._estimate_retry_after_seconds(
            int(decorated.get("estimated_wait_seconds") or 0),
            metrics=metrics,
        )
        decorated["queue_state"] = {
            "queued": int(metrics.get("queued") or 0),
            "running": int(metrics.get("running") or 0),
            "active_total": int(metrics.get("active_total") or 0),
            "active_users": int(metrics.get("active_users") or 0),
            "queue_pressure": str(metrics.get("queue_pressure") or "low"),
            "max_queue_size": int(metrics.get("max_queue_size") or self.max_queue_size),
        }
        return decorated

    def _build_queue_snapshot(self, counts: Dict[str, Any]) -> Dict[str, Any]:
        queued = int(counts.get("queued") or 0)
        running = int(counts.get("running") or 0)
        completed = int(counts.get("completed") or 0)
        failed = int(counts.get("failed") or 0)
        active_users = int(counts.get("active_users") or 0)
        oldest_queued_age_seconds = int(round(float(counts.get("oldest_queued_age_seconds") or 0.0)))
        avg_completion_seconds = float(counts.get("avg_completion_seconds") or 0.0)
        longest_running_age_seconds = int(round(float(counts.get("longest_running_age_seconds") or 0.0)))
        estimated_duration_seconds = int(round(avg_completion_seconds)) if avg_completion_seconds >= 1.0 else int(
            round(self.default_estimated_duration_sec)
        )
        active_total = queued + running
        estimated_wait_for_new_job_seconds = self._estimate_wait_seconds_for_position(
            queue_position=queued + 1,
            metrics={
                "queued": queued,
                "running": running,
                "estimated_duration_seconds": estimated_duration_seconds,
            },
        )
        queue_pressure = "low"
        if active_total >= max(int(self.max_queue_size * 0.85), self.worker_count * 6):
            queue_pressure = "high"
        elif active_total >= max(int(self.max_queue_size * 0.55), self.worker_count * 3):
            queue_pressure = "medium"
        return {
            "available": True,
            "worker_count": self.worker_count,
            "max_queue_size": self.max_queue_size,
            "queued": queued,
            "running": running,
            "completed": completed,
            "failed": failed,
            "waiting": queued,
            "active_total": active_total,
            "active_users": active_users,
            "max_concurrent": self.worker_count,
            "estimated_duration_seconds": estimated_duration_seconds,
            "estimated_wait_for_new_job_seconds": estimated_wait_for_new_job_seconds,
            "max_estimated_wait_seconds": self.max_estimated_wait_sec,
            "oldest_queued_age_seconds": oldest_queued_age_seconds,
            "avg_completion_seconds": round(avg_completion_seconds, 2),
            "longest_running_age_seconds": longest_running_age_seconds,
            "queue_pressure": queue_pressure,
        }

    def _estimate_wait_seconds_for_position(self, queue_position: int, *, metrics: Dict[str, Any]) -> int:
        position = max(int(queue_position or 0), 0)
        if position <= 0:
            return 0
        running = max(int(metrics.get("running") or 0), 0)
        worker_count = max(int(metrics.get("worker_count") or self.worker_count), 1)
        estimated_duration_seconds = max(
            int(metrics.get("estimated_duration_seconds") or round(self.default_estimated_duration_sec)),
            1,
        )
        available_slots = max(worker_count - running, 0)
        if position <= available_slots:
            return 0
        remaining_slots_needed = position - available_slots
        waves_ahead = max(math.ceil(remaining_slots_needed / worker_count), 0)
        return max(waves_ahead * estimated_duration_seconds, 0)

    def _estimate_retry_after_seconds(self, estimated_wait_seconds: int, *, metrics: Dict[str, Any]) -> int:
        wait_seconds = max(int(estimated_wait_seconds or 0), 0)
        estimated_duration_seconds = max(int(metrics.get("estimated_duration_seconds") or round(self.default_estimated_duration_sec)), 1)
        if wait_seconds <= 0:
            return max(self.min_retry_after_sec, min(estimated_duration_seconds, 30))
        return min(max(self.min_retry_after_sec, int(math.ceil(min(wait_seconds, 60) / 1.0))), 60)


class LearningReportSnapshotService:
    def __init__(
        self,
        *,
        kb: Any,
        progress_store_factory: Callable[[], Any],
        redis_cache: Optional[RedisJsonCache] = None,
    ) -> None:
        self.kb = kb
        self.progress_store_factory = progress_store_factory
        self.redis_cache = redis_cache or RedisJsonCache()
        self.snapshot_store = LearningReportSnapshotStore()
        self.worker_count = _env_int("DEEP_LEARNING_REPORT_WORKERS", 1, minimum=1)
        self.poll_interval = _env_float("DEEP_LEARNING_REPORT_WORKER_POLL_SEC", 1.0, minimum=0.2)
        self.stale_running_sec = _env_int("DEEP_LEARNING_REPORT_RUNNING_STALE_SEC", 600, minimum=30)
        self.cache_ttl = _env_int("DEEP_LEARNING_REPORT_REDIS_TTL_SEC", 900, minimum=60)
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            self.snapshot_store.requeue_stale_running(self.stale_running_sec)
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"deep-learning-report-worker-{index + 1}",
                    args=(index + 1,),
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def schedule_refresh(self, user_id: str, signature: Dict[str, Any]) -> Dict[str, Any]:
        signature_key = signature_to_key(signature)
        snapshot = self.snapshot_store.schedule_refresh(user_id, signature_key)
        self.redis_cache.delete(self._report_cache_key(user_id, signature_key))
        return snapshot

    def load_for_request(self, user_id: str, signature: Dict[str, Any]) -> Dict[str, Any]:
        signature_key = signature_to_key(signature)
        cached = self.redis_cache.get_json(self._report_cache_key(user_id, signature_key))
        if isinstance(cached, dict) and cached:
            return {
                "payload": cached,
                "fresh": True,
                "status": "ready",
                "signature_key": signature_key,
                "pending": False,
                "error_message": "",
            }

        snapshot = self.snapshot_store.get_snapshot(user_id)
        if not snapshot:
            self.schedule_refresh(user_id, signature)
            return {
                "payload": {},
                "fresh": False,
                "status": "queued",
                "signature_key": signature_key,
                "pending": True,
                "error_message": "",
            }

        payload = dict(snapshot.get("payload") or {})
        fresh = str(snapshot.get("status") or "") == "ready" and str(snapshot.get("signature_key") or "") == signature_key
        pending = str(snapshot.get("status") or "") in {"queued", "running"} and str(snapshot.get("target_signature_key") or "") == signature_key
        if fresh and payload:
            self.redis_cache.set_json(self._report_cache_key(user_id, signature_key), payload, ttl=self.cache_ttl)
            return {
                "payload": payload,
                "fresh": True,
                "status": "ready",
                "signature_key": signature_key,
                "pending": False,
                "error_message": str(snapshot.get("error_message") or ""),
            }

        if not pending:
            self.schedule_refresh(user_id, signature)
        return {
            "payload": payload,
            "fresh": False,
            "status": str(snapshot.get("status") or "queued"),
            "signature_key": signature_key,
            "pending": True,
            "error_message": str(snapshot.get("error_message") or ""),
        }

    def snapshot(self) -> Dict[str, Any]:
        counts = self.snapshot_store.stats()
        return {
            "worker_count": self.worker_count,
            "queued": int(counts.get("queued") or 0),
            "running": int(counts.get("running") or 0),
            "ready": int(counts.get("ready") or 0),
            "failed": int(counts.get("failed") or 0),
        }

    def _worker_loop(self, worker_number: int) -> None:
        worker_id = f"report-worker-{worker_number}"
        while not self._stop_event.is_set():
            snapshot = self.snapshot_store.claim_next_refresh(worker_id)
            if not snapshot:
                time.sleep(self.poll_interval)
                continue
            user_id = str(snapshot.get("user_id") or "").strip()
            signature_key = str(snapshot.get("target_signature_key") or "").strip()
            try:
                progress_store = self.progress_store_factory()
                state = build_attempt_state(self.kb, progress_store, user_id)
                payload = build_learning_report_context(self.kb, progress_store, user_id, state=state)
                self.snapshot_store.mark_ready(user_id, signature_key, payload, worker_id=worker_id)
                self.redis_cache.set_json(self._report_cache_key(user_id, signature_key), payload, ttl=self.cache_ttl)
            except Exception as exc:
                logger.exception("Learning report refresh failed for user %s.", user_id)
                self.snapshot_store.mark_failed(user_id, str(exc), worker_id=worker_id)

    def _report_cache_key(self, user_id: str, signature_key: str) -> str:
        return self.redis_cache.make_key("learning-report", user_id, signature_key)
