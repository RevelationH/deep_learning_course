from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests


ROOT_DIR = Path(__file__).resolve().parent
CHECKLIST_PATH = ROOT_DIR / "deep_learning_chat_test_question_checklist.md"

INLINE_SOURCE_PATTERNS = (
    re.compile(r"\.pdf", re.IGNORECASE),
    re.compile(r"page\s*\d+", re.IGNORECASE),
    re.compile(r"lecture\s*\d+", re.IGNORECASE),
)
OUT_OF_SCOPE_SCOPE_HINTS = tuple(
    marker.lower()
    for marker in (
        "深度学习课程",
        "当前课程",
        "本课程",
        "这门课",
        "课程范围",
        "课程主题",
        "course",
        "deep learning",
    )
)
OUT_OF_SCOPE_BOUNDARY_HINTS = tuple(
    marker.lower()
    for marker in (
        "无关",
        "不属于",
        "不在",
        "超出",
        "先不直接回答",
        "先不展开",
        "不直接作答",
        "不直接回答",
        "outside",
        "out of scope",
        "won't answer",
        "not answer directly",
    )
)
OUT_OF_SCOPE_REDIRECT_HINTS = tuple(
    marker.lower()
    for marker in (
        "可以继续问",
        "切回课程主题",
        "神经网络",
        "cnn",
        "transformer",
        "quiz",
        "learning report",
    )
)
CODE_REQUEST_RE = re.compile(r"(代码|code|pytorch|python|tensorflow)", re.IGNORECASE)
EXPLAIN_RE = re.compile(r"(解释|说明|explain)", re.IGNORECASE)
CODE_ONLY_RE = re.compile(r"(只给代码|只要代码|only code|just code|仅代码|不要解释|无需解释)", re.IGNORECASE)
EXISTING_CODE_REFERENCE_RE = re.compile(
    r"(上面(?:的)?代码|上面的例子|这段代码|代码里|代码中的|这份代码|上一段代码|刚才那段代码|上面代码里|上面代码中的)",
    re.IGNORECASE,
)
SUMMARY_RE = re.compile(r"(总结|概括|一句话|三句话|summary|summari[sz]e)", re.IGNORECASE)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_code_blocks(text: str) -> List[str]:
    return re.findall(r"```[\s\S]*?```", text or "")


def strip_code_blocks(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", " ", text or "")


def parse_checklist(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    sections: List[tuple[str, List[List[str]]]] = []
    current_title: str | None = None
    current_groups: List[List[str]] = []
    current_group: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("## "):
            if current_title is not None:
                if current_group:
                    current_groups.append(current_group)
                    current_group = []
                sections.append((current_title, current_groups))
            current_title = line[3:].strip()
            current_groups = []
            current_group = []
            continue

        stripped = line.strip()
        match = re.match(r"-\s+`(.+?)`\s*$", stripped)
        if match:
            current_group.append(match.group(1))
            continue

        if not stripped and current_group:
            current_groups.append(current_group)
            current_group = []

    if current_title is not None:
        if current_group:
            current_groups.append(current_group)
        sections.append((current_title, current_groups))

    cases: List[Dict[str, Any]] = []
    case_index = 1

    def add_single(section_num: int, section_title: str, question: str) -> None:
        nonlocal case_index
        cases.append(
            {
                "id": f"C{case_index:03d}",
                "type": "single",
                "section_num": section_num,
                "section_title": section_title,
                "title": question,
                "questions": [question],
            }
        )
        case_index += 1

    def add_flow(section_num: int, section_title: str, questions: Sequence[str], title: str) -> None:
        nonlocal case_index
        cases.append(
            {
                "id": f"C{case_index:03d}",
                "type": "flow",
                "section_num": section_num,
                "section_title": section_title,
                "title": title,
                "questions": list(questions),
            }
        )
        case_index += 1

    for title, groups in sections:
        match = re.match(r"(\d+)\.", title)
        if not match:
            continue
        section_num = int(match.group(1))
        flat = [item for group in groups for item in group]
        if section_num in {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15}:
            for question in flat:
                add_single(section_num, title, question)
        elif section_num == 10:
            add_flow(section_num, title, flat, "追问记忆流程")
        elif section_num == 14:
            add_flow(section_num, title, flat, "重复提问与改写流程")
        elif section_num == 16:
            for question in flat[:9]:
                add_single(section_num, title, question)
            add_flow(section_num, title, ["卷积", "再解释一下", "举个例子", "代码呢", "那这个为什么", "继续"], "短问题追问流程 A")
            add_flow(section_num, title, ["BatchNorm", "LayerNorm", "和上一个有什么区别", "继续"], "短问题比较流程 B")
        elif section_num == 17:
            for idx, group in enumerate(groups, start=1):
                if group:
                    add_flow(section_num, title, group, f"端到端学生流程 {idx}")

    return cases


def references_existing_code(question: str) -> bool:
    return bool(EXISTING_CODE_REFERENCE_RE.search(normalize_text(question)))


def needs_code(question: str, section_num: int) -> bool:
    if references_existing_code(question):
        return False
    if CODE_ONLY_RE.search(question):
        return True
    if section_num in (8, 9):
        return True
    return bool(CODE_REQUEST_RE.search(question))


def needs_explanation(question: str, section_num: int) -> bool:
    return section_num == 9 or bool(EXPLAIN_RE.search(question)) or references_existing_code(question) or bool(SUMMARY_RE.search(question))


def code_only_request(question: str) -> bool:
    return bool(CODE_ONLY_RE.search(question))


def needs_citation(question: str, section_num: int) -> bool:
    return section_num == 11 or "reference" in normalize_text(question).lower()


def is_out_of_scope_case(section_num: int) -> bool:
    return section_num == 15


def looks_like_out_of_scope_answer(answer: str) -> bool:
    lowered = normalize_text(answer).lower()
    has_scope = any(marker in lowered for marker in OUT_OF_SCOPE_SCOPE_HINTS)
    has_boundary = any(marker in lowered for marker in OUT_OF_SCOPE_BOUNDARY_HINTS)
    has_redirect = any(marker in lowered for marker in OUT_OF_SCOPE_REDIRECT_HINTS)
    return has_scope and (has_boundary or has_redirect)


def normalize_response_kind(value: Any) -> str:
    return normalize_text(value).lower().replace(" ", "_")


def mentions_source_inline(answer: str) -> bool:
    return any(pattern.search(answer or "") for pattern in INLINE_SOURCE_PATTERNS)


class ChecklistRunner:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "deep-learning-checklist-runner/1.0"})

    def login(self) -> None:
        response = self.session.post(
            f"{self.base_url}/login_deep_learning",
            data={"username": self.username, "password": self.password},
            allow_redirects=True,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"login failed: {response.status_code} {response.url}")

    def request_json(self, method: str, path: str, timeout: int = 60, **kwargs: Any) -> Dict[str, Any]:
        for _ in range(2):
            response = self.session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
            content_type = response.headers.get("content-type", "")
            looks_like_login = "login_deep_learning" in (response.url or "") or ("text/html" in content_type and "form" in response.text[:400].lower())
            if response.status_code in {401, 403} or looks_like_login:
                self.login()
                continue
            if "application/json" not in content_type:
                raise RuntimeError(f"non-json response from {path}: {response.status_code} {content_type} {response.text[:180]}")
            return response.json()
        raise RuntimeError(f"auth failed for {path}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        payload = self.request_json("GET", "/api/deep-learning/chat/sessions", timeout=30)
        return list(payload.get("sessions") or [])

    def delete_session(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            self.request_json("DELETE", f"/api/deep-learning/chat/sessions/{session_id}", timeout=30)
        except Exception:
            pass

    def send_and_wait(self, message: str, session_id: str | None = None) -> tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        accepted = self.request_json("POST", "/api/deep-learning/chat", json=payload, timeout=60)
        current_session_id = str((accepted.get("session") or {}).get("session_id") or session_id or "")
        job_id = str((accepted.get("job") or {}).get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"missing job id: {accepted!r}")
        for _ in range(120):
            job_payload = self.request_json("GET", f"/api/deep-learning/chat/jobs/{job_id}", timeout=30)
            job = job_payload.get("job") or {}
            status = str(job.get("status") or "")
            if status in {"completed", "failed"}:
                return current_session_id, job
            time.sleep(2)
        raise RuntimeError(f"job timeout: {job_id}")


def evaluate_turn(section_num: int, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(result.get("answer") or "")
    citations = list(result.get("citations") or [])
    response_kind = normalize_response_kind(result.get("response_kind") or "")
    plain = normalize_text(strip_code_blocks(answer))
    code_blocks = extract_code_blocks(answer)
    flags: Dict[str, Any] = {
        "empty_answer": not bool(plain or code_blocks),
        "json_leak": "@@COURSE_SOURCE_" in answer or answer.lstrip().startswith("{"),
        "missing_code": False,
        "missing_explanation": False,
        "missing_citations": False,
        "unexpected_citations": False,
        "empty_refusal": False,
        "wrong_response_kind": False,
        "inline_source_leak": mentions_source_inline(answer),
    }
    if needs_code(question, section_num):
        flags["missing_code"] = len(code_blocks) == 0
    if needs_explanation(question, section_num) and needs_code(question, section_num):
        flags["missing_explanation"] = len(plain) < 30
    if code_only_request(question):
        flags["too_much_non_code"] = len(plain) > 220
    if needs_citation(question, section_num):
        flags["missing_citations"] = len(citations or []) == 0
    if is_out_of_scope_case(section_num):
        flags["unexpected_citations"] = len(citations or []) > 0
        if response_kind:
            flags["wrong_response_kind"] = response_kind != "out_of_scope"
            valid_refusal = response_kind == "out_of_scope" and len(plain) >= 20
        else:
            valid_refusal = len(plain) >= 20 and looks_like_out_of_scope_answer(answer)
        flags["empty_refusal"] = not valid_refusal
    return flags


def run_checklist(base_url: str, username: str, password: str) -> Dict[str, Any]:
    cases = parse_checklist(CHECKLIST_PATH)
    runner = ChecklistRunner(base_url, username, password)
    runner.login()
    baseline_sessions = {row.get("session_id") for row in runner.list_sessions()}
    start_time = time.time()
    results: List[Dict[str, Any]] = []
    section_counts: Dict[str, int] = {}

    try:
        for case in cases:
            case_start = time.time()
            record: Dict[str, Any] = {
                "id": case["id"],
                "type": case["type"],
                "section_num": case["section_num"],
                "section_title": case["section_title"],
                "title": case["title"],
                "turns": [],
                "issues": [],
                "duration_seconds": 0,
            }
            current_session_id = ""
            answer_variants: List[str] = []
            for index, question in enumerate(case["questions"], start=1):
                turn: Dict[str, Any] = {"index": index, "question": question}
                try:
                    current_session_id, job = runner.send_and_wait(question, session_id=current_session_id or None)
                    result = job.get("result") or {}
                    answer = str(result.get("answer") or "")
                    citations = list(result.get("citations") or [])
                    flags = evaluate_turn(case["section_num"], question, result)
                    turn.update(
                        {
                            "job_status": job.get("status"),
                            "response_kind": normalize_response_kind(result.get("response_kind") or ""),
                            "mode": normalize_text(result.get("mode") or ""),
                            "answer_excerpt": normalize_text(answer)[:280],
                            "answer_len": len(answer),
                            "citation_count": len(citations),
                            "flags": flags,
                        }
                    )
                    answer_variants.append(normalize_text(answer))
                    for name, value in flags.items():
                        if value:
                            record["issues"].append(f"turn{index}:{name}")
                except Exception as exc:
                    turn["exception"] = str(exc)
                    record["issues"].append(f"turn{index}:exception")
                record["turns"].append(turn)
                time.sleep(0.4)

            if case["section_num"] == 14:
                unique_answers = len({text for text in answer_variants if text})
                if unique_answers <= 2:
                    record["issues"].append("repetition_flow_low_variation")

            record["duration_seconds"] = round(time.time() - case_start, 2)
            if current_session_id:
                runner.delete_session(current_session_id)
            results.append(record)
            section_key = str(case["section_num"])
            section_counts[section_key] = section_counts.get(section_key, 0) + 1
    finally:
        for row in runner.list_sessions():
            session_id = row.get("session_id")
            if session_id and session_id not in baseline_sessions:
                runner.delete_session(session_id)

    failed_cases = [row for row in results if row.get("issues")]
    summary = {
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_cases": len(results),
        "failed_cases": len(failed_cases),
        "passed_cases": len(results) - len(failed_cases),
        "section_case_counts": section_counts,
        "baseline_session_count": len(baseline_sessions),
        "remaining_session_count": len(runner.list_sessions()),
    }
    return {
        "summary": summary,
        "flagged": failed_cases[:60],
        "all_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对已部署的深度学习课程聊天系统执行测试清单回归。")
    parser.add_argument("--base-url", default="https://lesson.znbverynb.xin")
    parser.add_argument("--username", default="lx")
    parser.add_argument("--password", default="lx")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "manual_checks" / time.strftime("%Y%m%d_chat_checklist_run")))
    args = parser.parse_args()

    payload = run_checklist(args.base_url, args.username, args.password)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(str(output_path))


if __name__ == "__main__":
    main()
