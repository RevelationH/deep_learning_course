from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Iterable, List, Sequence

from deep_learning_portal.kb_service import STOPWORDS, parse_options, tokenize


def build_attempt_state(kb: Any, store: Any, user_id: str) -> Dict[str, Any]:
    return _build_attempt_state(kb, store, user_id)


def build_dashboard_context(kb: Any, store: Any, user_id: str, *, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    state = state or _build_attempt_state(kb, store, user_id)
    stats_map = {row["kp_id"]: row for row in state["kp_stats"]}
    kps = []
    for kp in kb.list_knowledge_points():
        stat = stats_map.get(kp["kp_id"], {})
        row = dict(kp)
        row["answered"] = stat.get("answered", 0)
        row["correct"] = stat.get("correct", 0)
        row["wrong"] = stat.get("wrong", 0)
        row["accuracy"] = stat.get("accuracy", 0.0)
        kps.append(row)
    return {
        "summary": state["summary"],
        "kp_stats": state["kp_stats"],
        "kp_stats_map": stats_map,
        "kps": kps,
        "recent_attempts": state["recent_attempts"],
        "latest_attempts": state["latest_attempts"],
    }


def build_learning_report_context(
    kb: Any,
    store: Any,
    user_id: str,
    follow_up_limit: int = 4,
    *,
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    state = state or _build_attempt_state(kb, store, user_id)
    weak_points = _build_weak_points(state)
    strength_points = _build_strength_points(state)
    review_queue = _build_review_queue(state)
    recommendations = _build_recommendations(state, weak_points, strength_points)
    follow_up_questions = _build_follow_up_questions(kb, store, user_id, weak_points, limit=follow_up_limit)
    return {
        "summary": state["summary"],
        "kp_stats": state["kp_stats"],
        "weak_points": weak_points,
        "strength_points": strength_points,
        "review_queue": review_queue,
        "recommendations": recommendations,
        "recent_attempts": state["recent_attempts"],
        "follow_up_questions": follow_up_questions,
        "activity": {
            "total_attempts": state["attempt_count"],
            "recent_accuracy": state["summary"]["recent_accuracy"],
        },
    }


def _build_attempt_state(kb: Any, store: Any, user_id: str) -> Dict[str, Any]:
    question_lookup = _question_lookup(kb)
    attempt_count = store.attempt_count(user_id)
    latest_attempts = _filter_current_attempts(store.latest_attempts(user_id), question_lookup, kb)
    latest_attempts.sort(key=lambda item: (item.get("timestamp") or "", item.get("question_id") or ""))
    kp_stats = _build_kp_stats(kb, latest_attempts)
    recent_attempts = _filter_current_attempts(store.recent_attempts(user_id, limit=8), question_lookup, kb)
    recent_attempts.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    summary = _build_summary(latest_attempts, kp_stats, recent_attempts, attempt_count)
    return {
        "latest_attempts": latest_attempts,
        "kp_stats": kp_stats,
        "summary": summary,
        "recent_attempts": _build_recent_attempts(recent_attempts, limit=8),
        "attempt_count": attempt_count,
        "question_lookup": question_lookup,
    }


def _question_lookup(kb: Any) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for questions in kb.questions_by_kp.values():
        for question in questions:
            lookup[str(question.get("question_id") or "")] = dict(question)
    return lookup


def _filter_current_attempts(
    attempts: Sequence[Dict[str, Any]],
    question_lookup: Dict[str, Dict[str, Any]],
    kb: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt in attempts:
        question_id = str(attempt.get("question_id") or "")
        question = question_lookup.get(question_id)
        if not question:
            continue
        row = dict(attempt)
        row["kp_id"] = question.get("kp_id")
        row["kp_name"] = question.get("kp_name")
        row["question"] = question.get("question")
        row["question_type"] = question.get("question_type") or "multiple_choice"
        row["reference_answer"] = question.get("correct_option") or attempt.get("reference_answer") or ""
        row["parsed_options"] = list(question.get("parsed_options") or parse_options(question.get("options") or []))
        row["submitted_answer"] = str(attempt.get("submitted_answer") or "").strip()
        row["is_correct"] = bool(attempt.get("is_correct"))
        row["explanation"] = question.get("explanation") or ""
        row["review_refs"] = list(question.get("review_refs") or kb.kp_review_refs(question.get("kp_id"), limit=2))
        row["image_path"] = question.get("image_path")
        row["image_caption"] = question.get("image_caption")
        row["timestamp_display"] = _format_timestamp(row.get("timestamp"))
        rows.append(row)
    rows.sort(key=lambda item: item.get("timestamp") or "")
    return rows


def _latest_attempts(attempts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for attempt in attempts:
        key = str(attempt.get("question_id") or "")
        previous = latest.get(key)
        if previous is None or str(attempt.get("timestamp") or "") >= str(previous.get("timestamp") or ""):
            latest[key] = attempt
    rows = list(latest.values())
    rows.sort(key=lambda item: (item.get("timestamp") or "", item.get("question_id") or ""))
    return rows


def _build_kp_stats(kb: Any, latest_attempts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for kp in kb.list_knowledge_points():
        rows[kp["kp_id"]] = {
            "kp_id": kp["kp_id"],
            "kp_name": kp["name"],
            "description": kp["description"],
            "weeks": list(kp.get("weeks", [])),
            "answered": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "last_at": "",
        }
    for attempt in latest_attempts:
        kp_id = str(attempt.get("kp_id") or "")
        if kp_id not in rows:
            continue
        row = rows[kp_id]
        row["answered"] += 1
        if attempt.get("is_correct"):
            row["correct"] += 1
        else:
            row["wrong"] += 1
        row["last_at"] = max(row["last_at"], str(attempt.get("timestamp") or ""))
    for row in rows.values():
        row["accuracy"] = round((row["correct"] / row["answered"]) * 100, 1) if row["answered"] else 0.0
    return sorted(
        rows.values(),
        key=lambda item: (
            item["answered"] == 0,
            item["accuracy"] if item["answered"] else 101.0,
            item["kp_name"],
        ),
    )


def _build_summary(
    latest_attempts: Sequence[Dict[str, Any]],
    kp_stats: Sequence[Dict[str, Any]],
    recent_attempts: Sequence[Dict[str, Any]],
    attempt_count: int,
) -> Dict[str, Any]:
    answered = len(latest_attempts)
    correct = sum(1 for item in latest_attempts if item.get("is_correct"))
    wrong = answered - correct
    explored_points = sum(1 for row in kp_stats if row["answered"] > 0)
    mastered_points = sum(1 for row in kp_stats if row["answered"] >= 3 and row["accuracy"] >= 80)
    weak_points = sum(1 for row in kp_stats if row["answered"] > 0 and row["wrong"] > 0)
    recent_accuracy = _accuracy(recent_attempts[:8])
    overall_accuracy = round((correct / answered) * 100, 1) if answered else 0.0
    if attempt_count < 3:
        trend_label = "起步阶段"
        trend_note = "当前作答数据还比较少，建议先完成更多题目后再看趋势。"
    elif recent_accuracy >= overall_accuracy + 10:
        trend_label = "近期上升"
        trend_note = "你最近几次作答的正确率高于整体平均，状态在提升。"
    elif recent_accuracy <= overall_accuracy - 10:
        trend_label = "需要回顾"
        trend_note = "你最近几次作答的表现低于整体平均，建议尽快回顾薄弱知识点。"
    else:
        trend_label = "整体稳定"
        trend_note = "你最近的作答表现与整体水平接近，建议继续稳步练习。"
    return {
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "accuracy": overall_accuracy,
        "explored_points": explored_points,
        "mastered_points": mastered_points,
        "weak_points": weak_points,
        "recent_accuracy": recent_accuracy,
        "trend_label": trend_label,
        "trend_note": trend_note,
    }


def _build_recent_attempts(attempts: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    rows = list(attempts)[:limit]
    rows.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return rows


def _build_strength_points(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        row for row in state["kp_stats"]
        if row["answered"] >= 2 and row["accuracy"] >= 80
    ]
    rows.sort(key=lambda item: (-item["accuracy"], -item["answered"], item["kp_name"]))
    return rows[:4]


def _build_weak_points(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest_attempts = list(state["latest_attempts"])
    rows = [
        row for row in state["kp_stats"]
        if row["answered"] > 0 and (row["wrong"] > 0 or row["accuracy"] < 70)
    ]
    rows.sort(key=lambda item: (-item["wrong"], item["accuracy"], item["kp_name"]))
    enriched: List[Dict[str, Any]] = []
    for row in rows[:4]:
        latest_wrong = next(
            (
                item for item in reversed(latest_attempts)
                if item.get("kp_id") == row["kp_id"] and not item.get("is_correct")
            ),
            None,
        )
        enriched_row = dict(row)
        if latest_wrong:
            enriched_row["latest_mistake_question"] = latest_wrong.get("question") or ""
            enriched_row["latest_mistake_explanation"] = latest_wrong.get("explanation") or ""
            enriched_row["review_refs"] = list(latest_wrong.get("review_refs") or [])
        else:
            enriched_row["latest_mistake_question"] = ""
            enriched_row["latest_mistake_explanation"] = ""
            enriched_row["review_refs"] = []
        enriched.append(enriched_row)
    return enriched


def _build_review_queue(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [item for item in reversed(state["latest_attempts"]) if not item.get("is_correct")]
    return rows[:6]


def _build_recommendations(
    state: Dict[str, Any],
    weak_points: Sequence[Dict[str, Any]],
    strength_points: Sequence[Dict[str, Any]],
) -> List[str]:
    recommendations: List[str] = []
    if weak_points:
        top = weak_points[0]
        recommendations.append(f"优先回顾“{top['kp_name']}”。你在这一部分的当前正确率为 {top['accuracy']}%，而且最近的错误记录说明这一知识点还不稳定。")
    if len(weak_points) >= 2:
        second = weak_points[1]
        recommendations.append(f"把“{second['kp_name']}”安排在下一轮练习中。建议先看讲义来源，再重新做同主题选择题。")
    if strength_points:
        top_strength = strength_points[0]
        recommendations.append(f"“{top_strength['kp_name']}”目前是你的相对优势项。可以用少量题目维持熟练度，把主要时间分配给薄弱项。")
    if state["summary"]["recent_accuracy"] < 70 and state["summary"]["answered"] >= 4:
        recommendations.append("最近几次作答波动较大。建议先做小范围复习，再回到新题练习，而不是连续跨主题做题。")
    if not recommendations:
        recommendations.append("当前数据还比较少。建议先完成 2 到 3 个知识点的练习，学习报告会更准确。")
    return recommendations[:4]


def _follow_up_cache_key(weak_points: Sequence[Dict[str, Any]], limit: int) -> str:
    base = "|".join(f"{row['kp_id']}:{row['wrong']}:{row['accuracy']}" for row in weak_points[:3]) + f"|{limit}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def _build_follow_up_questions(
    kb: Any,
    store: Any,
    user_id: str,
    weak_points: Sequence[Dict[str, Any]],
    limit: int = 4,
) -> List[Dict[str, Any]]:
    if not weak_points:
        return []
    cache_key = _follow_up_cache_key(weak_points, limit)
    cached = store.get_generated_followups(user_id, cache_key)
    if cached:
        return [_format_follow_up_item(item, reason=item.get("reason") or "根据你最近的作答表现，这道题用于补强当前薄弱点。") for item in cached[:limit]]

    rows: List[Dict[str, Any]] = []
    target_kps = list(weak_points[:2])
    per_kp_limit = max(1, limit // max(len(target_kps), 1))
    for weak in target_kps:
        selected = _select_follow_up_candidates(
            kb,
            weak["kp_id"],
            weak.get("latest_mistake_question") or "",
            limit=per_kp_limit + 1,
        )
        for item in selected:
            row = _format_follow_up_item(
                item,
                reason=f"推荐继续巩固“{weak['kp_name']}”，因为你在这一知识点上仍有明显失分。",
            )
            rows.append(row)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break

    if rows:
        store.set_generated_followups(user_id, cache_key, rows)
    return rows[:limit]


def _select_follow_up_candidates(
    kb: Any,
    kp_id: str,
    latest_mistake_question: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    questions = list(kb.related_questions(kp_id))
    if not questions:
        return []

    latest_text = str(latest_mistake_question or "").strip()
    latest_tokens = set(tokenize(latest_text))

    scored: List[tuple[int, int, str, Dict[str, Any]]] = []
    for item in questions:
        question_text = str(item.get("question") or "").strip()
        question_tokens = set(tokenize(question_text))
        overlap = len(latest_tokens & question_tokens) if latest_tokens else 0
        image_bonus = 1 if item.get("image_path") else 0
        exact_penalty = 1 if latest_text and question_text == latest_text else 0
        score = overlap * 3 + image_bonus - exact_penalty * 4
        scored.append((score, image_bonus, question_text, item))

    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected: List[Dict[str, Any]] = []
    seen_questions: set[str] = set()
    for _score, _image_bonus, question_text, item in scored:
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        if latest_text and question_text == latest_text and len(scored) > limit:
            continue
        selected.append(dict(item))
        if len(selected) >= limit:
            break

    return selected[:limit]


def _format_follow_up_item(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    parsed_options = item.get("parsed_options") or parse_options(item.get("options") or [])
    reference_answer = item.get("reference_answer") or _format_reference_answer(item)
    return {
        "question_id": item.get("question_id"),
        "kp_id": item.get("kp_id"),
        "kp_name": item.get("kp_name"),
        "question": item.get("question") or "",
        "parsed_options": parsed_options,
        "reference_answer": reference_answer,
        "explanation": item.get("explanation") or "",
        "review_refs": list(item.get("review_refs") or []),
        "image_path": item.get("image_path"),
        "image_caption": item.get("image_caption"),
        "reason": reason,
    }


def _format_reference_answer(question: Dict[str, Any]) -> str:
    correct_label = str(question.get("correct_option") or "").upper().strip()
    parsed_options = question.get("parsed_options") or parse_options(question.get("options") or [])
    option_lookup = {item["label"]: item["text"] for item in parsed_options if item.get("label")}
    if correct_label and correct_label in option_lookup:
        return f"{correct_label}. {option_lookup[correct_label]}"
    answer = str(question.get("answer") or "").strip()
    return answer


def _accuracy(attempts: Sequence[Dict[str, Any]]) -> float:
    rows = [item for item in attempts if item.get("submitted_answer") or item.get("is_correct") is not None]
    if not rows:
        return 0.0
    correct = sum(1 for item in rows if item.get("is_correct"))
    return round((correct / len(rows)) * 100, 1)


def _format_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "无时间记录"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return text


def _token_signature(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1 and token not in STOPWORDS}
