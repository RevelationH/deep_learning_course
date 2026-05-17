from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from deep_learning_portal.redis_cache import RedisJsonCache
from deep_learning_portal.student_analytics import build_attempt_state, build_learning_report_context
from pg_support.chat_job_store import ChatJobStore
from pg_support.report_snapshot_store import LearningReportSnapshotStore
from runtime_tuning import default_chat_queue_max_size, default_chat_worker_count, default_report_worker_count


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


class ChatWorkersUnavailableError(RuntimeError):
    def __init__(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("No live chat workers are available.")
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
        self.worker_count = _env_int("DEEP_LEARNING_CHAT_WORKERS", default_chat_worker_count(), minimum=1)
        self.max_queue_size = _env_int(
            "DEEP_LEARNING_CHAT_QUEUE_MAX_SIZE",
            default_chat_queue_max_size(self.worker_count),
            minimum=max(self.worker_count * 4, 8),
        )
        self.poll_interval = _env_float("DEEP_LEARNING_CHAT_WORKER_POLL_SEC", 0.8, minimum=0.1)
        self.job_retention_sec = _env_int("DEEP_LEARNING_CHAT_JOB_RETENTION_SEC", 21600, minimum=600)
        self.stale_running_sec = _env_int("DEEP_LEARNING_CHAT_RUNNING_STALE_SEC", 180, minimum=60)
        self.cache_ttl = _env_int("DEEP_LEARNING_CHAT_JOB_CACHE_TTL_SEC", 3600, minimum=60)
        self.default_estimated_duration_sec = _env_float("DEEP_LEARNING_CHAT_DEFAULT_ESTIMATED_DURATION_SEC", 18.0, minimum=1.0)
        self.max_estimated_wait_sec = _env_int("DEEP_LEARNING_CHAT_MAX_ESTIMATED_WAIT_SEC", 150, minimum=15)
        self.min_retry_after_sec = _env_int("DEEP_LEARNING_CHAT_RETRY_AFTER_SEC", 12, minimum=1)
        self.job_touch_interval_sec = _env_float("DEEP_LEARNING_CHAT_JOB_TOUCH_SEC", 10.0, minimum=2.0)
        self.worker_heartbeat_ttl_sec = _env_int("DEEP_LEARNING_CHAT_WORKER_HEARTBEAT_TTL_SEC", 45, minimum=10)
        self.worker_heartbeat_interval_sec = _env_float("DEEP_LEARNING_CHAT_WORKER_HEARTBEAT_SEC", 15.0, minimum=2.0)
        self.queued_poll_floor_ms = _env_int("DEEP_LEARNING_CHAT_QUEUED_POLL_FLOOR_MS", 2200, minimum=800)
        self.running_poll_floor_ms = _env_int("DEEP_LEARNING_CHAT_RUNNING_POLL_FLOOR_MS", 1400, minimum=700)
        self.max_poll_after_ms = _env_int("DEEP_LEARNING_CHAT_MAX_POLL_AFTER_MS", 6500, minimum=1500)
        self.instance_id = os.getenv("DEEP_LEARNING_CHAT_INSTANCE_ID", "").strip() or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        )
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            self.job_store.fail_stale_running_jobs(self.stale_running_sec)
            self.job_store.prune_finished_jobs(self.job_retention_sec)
            self._refresh_worker_heartbeat()
            if self._heartbeat_thread is None:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="deep-learning-chat-heartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
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
        live_snapshot = self.snapshot()
        if int(live_snapshot.get("live_worker_slots") or 0) <= 0:
            raise ChatWorkersUnavailableError(snapshot=live_snapshot)
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
        metrics["instance_id"] = self.instance_id
        metrics["local_workers_started"] = len(self._threads)
        metrics["live_worker_instances"] = len(self.list_live_worker_instances())
        return metrics

    def _worker_loop(self, worker_number: int) -> None:
        worker_id = f"{self.instance_id}/chat-worker-{worker_number}"
        idle_cycles = 0
        while not self._stop_event.is_set():
            job = self.job_store.claim_next_job(worker_id)
            if not job:
                idle_cycles += 1
                if idle_cycles % 120 == 0:
                    try:
                        self.job_store.fail_stale_running_jobs(self.stale_running_sec)
                    except Exception:
                        logger.exception("Failed to fail stale chat jobs.")
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
        job_id = str(job.get("job_id") or "").strip()
        touch_stop = threading.Event()
        touch_thread = threading.Thread(
            target=self._touch_running_job_loop,
            name=f"chat-job-touch-{job_id[:8]}",
            args=(job_id, worker_id, touch_stop),
            daemon=True,
        )
        touch_thread.start()
        chat_store = self.chat_store_factory()
        try:
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
                "response_kind": str(answer.get("response_kind") or ""),
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
        finally:
            touch_stop.set()
            touch_thread.join(timeout=1.0)

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
        decorated["poll_after_ms"] = self._recommended_poll_after_ms(decorated, metrics=metrics)
        return decorated

    def _recommended_poll_after_ms(self, job: Dict[str, Any], *, metrics: Dict[str, Any]) -> int:
        status = str(job.get("status") or "queued")
        if status in {"completed", "failed"}:
            return 0

        queue_pressure = str(metrics.get("queue_pressure") or "low")
        estimated_wait_seconds = max(int(job.get("estimated_wait_seconds") or 0), 0)
        retry_after_ms = max(int(job.get("retry_after_seconds") or 0) * 1000, 0)
        queue_position = max(int(job.get("queue_position") or 0), 0)
        worker_count = max(int(metrics.get("worker_count") or self.worker_count), 1)

        if status == "queued":
            base_ms = max(retry_after_ms, self.queued_poll_floor_ms)
            if estimated_wait_seconds >= 45:
                base_ms = max(base_ms, 6000)
            elif estimated_wait_seconds >= 20:
                base_ms = max(base_ms, 4500)
            elif estimated_wait_seconds >= 8:
                base_ms = max(base_ms, 3200)
            if queue_position > worker_count * 3:
                base_ms = max(base_ms, 4200)
            if queue_pressure == "high":
                base_ms = max(base_ms, 5000)
            elif queue_pressure == "medium":
                base_ms = max(base_ms, 3000)
            return min(base_ms, self.max_poll_after_ms)

        base_ms = self.running_poll_floor_ms
        if queue_pressure == "high":
            base_ms = max(base_ms, 1800)
        elif queue_pressure == "medium":
            base_ms = max(base_ms, 1500)
        return min(base_ms, self.max_poll_after_ms)

    def _build_queue_snapshot(self, counts: Dict[str, Any]) -> Dict[str, Any]:
        queued = int(counts.get("queued") or 0)
        running = int(counts.get("running") or 0)
        completed = int(counts.get("completed") or 0)
        failed = int(counts.get("failed") or 0)
        active_users = int(counts.get("active_users") or 0)
        oldest_queued_age_seconds = int(round(float(counts.get("oldest_queued_age_seconds") or 0.0)))
        avg_completion_seconds = float(counts.get("avg_completion_seconds") or 0.0)
        longest_running_age_seconds = int(round(float(counts.get("longest_running_age_seconds") or 0.0)))
        live_worker_slots = self.cluster_worker_count()
        estimated_duration_seconds = int(round(avg_completion_seconds)) if avg_completion_seconds >= 1.0 else int(
            round(self.default_estimated_duration_sec)
        )
        active_total = queued + running
        estimated_wait_for_new_job_seconds = self._estimate_wait_seconds_for_position(
            queue_position=queued + 1,
            metrics={
                "queued": queued,
                "running": running,
                "worker_count": live_worker_slots,
                "estimated_duration_seconds": estimated_duration_seconds,
            },
        )
        queue_pressure = "low"
        if active_total >= max(int(self.max_queue_size * 0.85), live_worker_slots * 6):
            queue_pressure = "high"
        elif active_total >= max(int(self.max_queue_size * 0.55), live_worker_slots * 3):
            queue_pressure = "medium"
        return {
            "available": True,
            "worker_count": live_worker_slots,
            "local_worker_count": self.worker_count,
            "max_queue_size": self.max_queue_size,
            "queued": queued,
            "running": running,
            "completed": completed,
            "failed": failed,
            "waiting": queued,
            "active_total": active_total,
            "active_users": active_users,
            "max_concurrent": live_worker_slots,
            "live_worker_slots": live_worker_slots,
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
        worker_count_raw = metrics.get("worker_count")
        if worker_count_raw is None:
            worker_count = self.worker_count
        else:
            worker_count = int(worker_count_raw or 0)
        if worker_count <= 0:
            estimated_duration_seconds = max(
                int(metrics.get("estimated_duration_seconds") or round(self.default_estimated_duration_sec)),
                1,
            )
            return max(self.max_estimated_wait_sec + estimated_duration_seconds, estimated_duration_seconds)
        worker_count = max(worker_count, 1)
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

    def list_live_worker_instances(self) -> list[Dict[str, Any]]:
        if not self.redis_cache.available():
            if self._threads:
                return [self._worker_heartbeat_payload()]
            return []
        pattern = self.redis_cache.make_key("chat-workers", "*")
        rows = self.redis_cache.list_json(pattern, limit=128)
        now_ts = int(time.time())
        max_age_seconds = max(int(self.worker_heartbeat_interval_sec * 2), 30)
        return [
            row
            for row in rows
            if isinstance(row, dict)
            and int(row.get("worker_count") or 0) > 0
            and max(now_ts - int(row.get("updated_at") or 0), 0) <= max_age_seconds
        ]

    def cluster_worker_count(self) -> int:
        live_instances = self.list_live_worker_instances()
        if live_instances:
            return max(sum(max(int(item.get("worker_count") or 0), 0) for item in live_instances), 0)
        return self.worker_count if self._threads else 0

    def _worker_heartbeat_key(self) -> str:
        return self.redis_cache.make_key("chat-workers", self.instance_id)

    def _worker_heartbeat_payload(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "worker_count": self.worker_count,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "updated_at": int(time.time()),
        }

    def _refresh_worker_heartbeat(self) -> None:
        if not self.redis_cache.available():
            return
        self.redis_cache.set_json(
            self._worker_heartbeat_key(),
            self._worker_heartbeat_payload(),
            ttl=self.worker_heartbeat_ttl_sec,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            self._refresh_worker_heartbeat()
            self._stop_event.wait(self.worker_heartbeat_interval_sec)

    def _touch_running_job_loop(self, job_id: str, worker_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.job_touch_interval_sec):
            try:
                self.job_store.touch_running_job(job_id, worker_id)
            except Exception:
                logger.exception("Failed to touch running chat job %s.", job_id)


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
        self.worker_count = _env_int("DEEP_LEARNING_REPORT_WORKERS", default_report_worker_count(), minimum=1)
        self.poll_interval = _env_float("DEEP_LEARNING_REPORT_WORKER_POLL_SEC", 1.0, minimum=0.2)
        self.stale_running_sec = _env_int("DEEP_LEARNING_REPORT_RUNNING_STALE_SEC", 600, minimum=30)
        self.cache_ttl = _env_int("DEEP_LEARNING_REPORT_REDIS_TTL_SEC", 900, minimum=60)
        self.refresh_debounce_sec = _env_int("DEEP_LEARNING_REPORT_REFRESH_DEBOUNCE_SEC", 20, minimum=5)
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
        if not (
            str(snapshot.get("status") or "") == "ready"
            and str(snapshot.get("signature_key") or "") == signature_key
        ):
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

        active_snapshot = dict(snapshot)
        payload = dict(active_snapshot.get("payload") or {})
        fresh = (
            str(active_snapshot.get("status") or "") == "ready"
            and str(active_snapshot.get("signature_key") or "") == signature_key
            and bool(payload)
        )
        pending = (
            str(active_snapshot.get("status") or "") in {"queued", "running"}
            and str(active_snapshot.get("target_signature_key") or "") == signature_key
        )
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

        if not pending and self._should_schedule_refresh(active_snapshot, signature_key):
            active_snapshot = self.schedule_refresh(user_id, signature)
            payload = dict(active_snapshot.get("payload") or payload)
            pending = (
                str(active_snapshot.get("status") or "") in {"queued", "running"}
                and str(active_snapshot.get("target_signature_key") or "") == signature_key
            )
        status = str(active_snapshot.get("status") or "queued")
        if payload and not fresh and not pending and status == "ready":
            status = "stale"
        return {
            "payload": payload,
            "fresh": False,
            "status": status,
            "signature_key": signature_key,
            "pending": pending,
            "error_message": str(active_snapshot.get("error_message") or ""),
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

    def _should_schedule_refresh(self, snapshot: Dict[str, Any], signature_key: str) -> bool:
        status = str(snapshot.get("status") or "")
        current_signature = str(snapshot.get("signature_key") or "")
        target_signature = str(snapshot.get("target_signature_key") or "")
        if status in {"queued", "running"} and target_signature == signature_key:
            return False
        if status == "ready" and current_signature == signature_key:
            return False

        updated_at = str(snapshot.get("updated_at") or "").strip()
        if updated_at:
            try:
                updated_ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if updated_ts.tzinfo is None:
                    updated_ts = updated_ts.replace(tzinfo=timezone.utc)
                age_seconds = max((datetime.now(timezone.utc) - updated_ts.astimezone(timezone.utc)).total_seconds(), 0.0)
            except Exception:
                age_seconds = float(self.refresh_debounce_sec)
            if age_seconds < self.refresh_debounce_sec:
                return False
        return True
