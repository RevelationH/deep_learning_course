from __future__ import annotations

import json
import random
from itertools import count
from pathlib import Path
from typing import ClassVar

from gevent import sleep
from gevent.lock import Semaphore
from locust import HttpUser, between, task


BASE_DIR = Path(__file__).resolve().parent
KP_PATH = BASE_DIR / "deep_learning_rag" / "artifacts_full_course" / "knowledge_points.json"

ACCOUNT_PREFIX = "loadtestdl"
ACCOUNT_COUNT = 240
_ACCOUNT_COUNTER = count(1)
_ACCOUNT_LOCK = Semaphore()

QUESTION_BANK = [
    "What is backpropagation?",
    "Explain the role of convolution and padding.",
    "Why does overfitting happen in deep learning?",
    "What is the difference between CNN and RNN?",
    "How does attention work in a transformer?",
    "What problem does batch normalization solve?",
    "Why do we need activation functions?",
    "Explain the intuition behind diffusion models.",
]


def load_kp_ids() -> list[str]:
    try:
        payload = json.loads(KP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    kp_ids: list[str] = []
    for item in payload:
        kp_id = str(item.get("kp_id") or "").strip()
        if kp_id:
            kp_ids.append(kp_id)
    return kp_ids


KP_IDS = load_kp_ids()


class DeepLearningPortalUser(HttpUser):
    wait_time = between(20, 35)
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

    @task(4)
    def chat(self) -> None:
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
        for _ in range(150):
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
            sleep(1.1 if status == "queued" else 0.8)
        if not completed:
            raise RuntimeError(f"chat job timeout for {job_id}")

    @task(1)
    def quiz_dashboard(self) -> None:
        if not self._login_ok:
            return
        self.client.get("/quiz_deep_learning", name="GET /quiz_deep_learning")

    @task(1)
    def quiz_practice(self) -> None:
        if not self._login_ok or not KP_IDS:
            return
        kp_id = random.choice(KP_IDS)
        self.client.get(f"/quiz_deep_learning/practice/{kp_id}", name="GET /quiz_deep_learning/practice")

    @task(1)
    def learning_report(self) -> None:
        if not self._login_ok:
            return
        self.client.get("/learning_report_deep_learning", name="GET /learning_report_deep_learning")
