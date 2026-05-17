from __future__ import annotations

import json
import random
from itertools import count
from typing import ClassVar

from gevent import sleep
from gevent.lock import Semaphore
from locust import HttpUser, between, task


ACCOUNT_PREFIX = "loadtestdl"
ACCOUNT_COUNT = 240
_ACCOUNT_COUNTER = count(1)
_ACCOUNT_LOCK = Semaphore()

QUESTION_BANK = [
    "什么是反向传播？",
    "解释一下卷积和 padding 的作用。",
    "为什么深度学习里会出现过拟合？",
    "CNN 和 RNN 的主要区别是什么？",
    "Transformer 里的 attention 是怎么工作的？",
    "Batch normalization 解决了什么问题？",
    "为什么需要激活函数？",
    "请解释 diffusion model 的直觉。",
]


def next_poll_delay_seconds(job: dict[str, object], attempt: int) -> float:
    status = str(job.get("status") or "").strip()
    suggested = max(float(job.get("poll_after_ms") or 0.0), 0.0) / 1000.0
    retry_after = max(float(job.get("retry_after_seconds") or 0.0), 0.0)
    estimated_wait = max(float(job.get("estimated_wait_seconds") or 0.0), 0.0)

    if status == "queued":
        base = max(suggested, retry_after, 1.8)
        if attempt >= 4:
            base = max(base, min(estimated_wait * 0.25, 5.5))
        return min(base, 7.0)
    if status == "running":
        return min(max(suggested, 1.2 if attempt < 6 else 1.8), 3.2)
    return 0.0


class DeepLearningChatOnlyUser(HttpUser):
    wait_time = between(2, 5)
    account_number: int
    username: str
    password: str
    session_id: str | None
    _login_ok: bool
    _assigned_accounts: ClassVar[set[int]] = set()

    def on_start(self) -> None:
        with _ACCOUNT_LOCK:
            next_id = next(_ACCOUNT_COUNTER)
        account_id = ((next_id - 1) % ACCOUNT_COUNT) + 1
        self.account_number = account_id
        self.username = f"{ACCOUNT_PREFIX}{account_id:03d}"
        self.password = f"ld{account_id:03d}"
        self.session_id = None
        self._login_ok = False
        with self.client.post(
            "/login_deep_learning",
            data={"username": self.username, "password": self.password},
            name="POST /login_deep_learning",
            catch_response=True,
            allow_redirects=False,
        ) as response:
            if response.status_code not in (302, 303):
                response.failure(f"login status={response.status_code}")
                return
            location = response.headers.get("Location", "")
            if "/chatapi_deep_learning.html" not in location:
                response.failure(f"unexpected redirect={location}")
                return
            response.success()
            self._login_ok = True

        if self._login_ok:
            self.client.get("/chatapi_deep_learning.html", name="GET /chatapi_deep_learning.html")

    @task
    def chat_only(self) -> None:
        if not self._login_ok:
            return
        payload = {
            "message": random.choice(QUESTION_BANK),
            "history": [],
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        with self.client.post(
            "/api/deep-learning/chat",
            json=payload,
            name="POST /api/deep-learning/chat",
            catch_response=True,
            timeout=120,
        ) as response:
            try:
                body = response.json()
            except Exception as exc:
                response.failure(f"invalid-json={exc}")
                return

            job = {}
            if response.status_code in (200, 202) and body.get("ok"):
                job = body.get("job") or {}
            elif response.status_code == 429 and isinstance(body.get("active_job"), dict):
                job = body.get("active_job") or {}
            else:
                response.failure(f"status={response.status_code} body={body!r}")
                return

            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                response.failure(f"missing-job={body!r}")
                return

            session = body.get("session") or {}
            session_id = str(session.get("session_id") or "").strip()
            if session_id:
                self.session_id = session_id
            response.success()

        completed = False
        for attempt in range(180):
            with self.client.get(
                f"/api/deep-learning/chat/jobs/{job_id}",
                name="GET /api/deep-learning/chat/jobs",
                catch_response=True,
                timeout=120,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"status={response.status_code}")
                    return
                try:
                    body = response.json()
                except Exception as exc:
                    response.failure(f"invalid-json={exc}")
                    return
                if not body.get("ok"):
                    response.failure(f"invalid-body={body!r}")
                    return
                job = body.get("job") or {}
                status = str(job.get("status") or "").strip()
                if status == "completed":
                    response.success()
                    completed = True
                    break
                if status == "failed":
                    response.failure(f"job-failed={job!r}")
                    return
                if status not in {"queued", "running"}:
                    response.failure(f"unexpected-job-status={job!r}")
                    return
                response.success()
            delay_seconds = next_poll_delay_seconds(job, attempt)
            if delay_seconds > 0:
                sleep(delay_seconds)

        if not completed:
            raise RuntimeError(f"chat job timeout for {job_id}")
