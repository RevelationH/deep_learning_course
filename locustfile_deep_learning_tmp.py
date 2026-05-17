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
QUESTIONS_PATH = BASE_DIR / "deep_learning_rag" / "artifacts_full_course" / "questions.json"

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


def load_questions_by_kp() -> dict[str, list[dict[str, str]]]:
    try:
        payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in payload:
        kp_id = str(item.get("kp_id") or "").strip()
        question_id = str(item.get("question_id") or "").strip()
        if not kp_id or not question_id:
            continue
        options = item.get("options") or []
        labels: list[str] = []
        for option in options:
            option_text = str(option).strip()
            if len(option_text) >= 2 and option_text[1:2] == ".":
                labels.append(option_text[0].upper())
        if not labels:
            labels = ["A", "B", "C", "D"]
        grouped.setdefault(kp_id, []).append({"question_id": question_id, "options": labels})
    return grouped


QUESTIONS_BY_KP = load_questions_by_kp()


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


class DeepLearningPortalUser(HttpUser):
    wait_time = between(2, 6)
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
        for attempt in range(150):
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
        with self.client.get(
            f"/quiz_deep_learning/practice/{kp_id}",
            name="GET /quiz_deep_learning/practice",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return
            if "practice-form" not in response.text and "answer_" not in response.text:
                response.failure("missing-practice-form")
                return
            response.success()

    @task(1)
    def quiz_submit(self) -> None:
        if not self._login_ok or not KP_IDS:
            return
        kp_id = random.choice(KP_IDS)
        questions = QUESTIONS_BY_KP.get(kp_id) or []
        if not questions:
            return
        form_data: dict[str, str] = {}
        for item in questions:
            form_data[f"answer_{item['question_id']}"] = random.choice(item["options"])
        with self.client.post(
            f"/quiz_deep_learning/practice/{kp_id}",
            data=form_data,
            name="POST /quiz_deep_learning/practice",
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return
            if "正确答案" not in response.text and "已作答" not in response.text:
                response.failure("missing-graded-content")
                return
            response.success()

    @task(1)
    def learning_report(self) -> None:
        if not self._login_ok:
            return
        with self.client.get(
            "/learning_report_deep_learning",
            name="GET /learning_report_deep_learning",
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return
            if (
                "学习报告" not in response.text
                and "优先回顾" not in response.text
                and "report-status-banner" not in response.text
            ):
                response.failure("missing-report-content")
                return
            response.success()
