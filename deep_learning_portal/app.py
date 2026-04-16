from __future__ import annotations

import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_loader import load_project_env

load_project_env()

from deep_learning_portal.async_services import (
    AsyncChatService,
    ChatJobAlreadyActiveError,
    ChatQueueFullError,
    LearningReportSnapshotService,
    signature_to_key,
)
from deep_learning_portal.chat_pipeline import DeepLearningChatPipeline
from deep_learning_portal.kb_service import DeepLearningKnowledgeBase, clean_display_text
from deep_learning_portal.redis_cache import RedisJsonCache
from deep_learning_portal.serve_runtime import serve_app
from deep_learning_portal.student_analytics import build_attempt_state, build_dashboard_context
from pg_support.schema import ensure_schema
from storage_backend import get_storage_backend_name, load_storage_classes, storage_backend_notice, storage_backend_ready

try:
    User, ProgressStore, ChatSessionStore = load_storage_classes()
    from werkzeug.security import check_password_hash, generate_password_hash
    AUTH_BACKEND_ERROR: Optional[Exception] = None
except Exception as exc:
    User = None
    ProgressStore = None
    ChatSessionStore = None
    check_password_hash = None
    generate_password_hash = None
    AUTH_BACKEND_ERROR = exc


ANSWER_LABEL_RE = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,40}$")
PUBLIC_ENDPOINTS = {
    "healthz",
    "login_deep_learning",
    "register_deep_learning",
    "logout_deep_learning",
    "static",
    "deep_learning_material_asset",
}


def clean_account_name(value: Optional[str]) -> str:
    return (value or "").strip()[:40]


def clean_login_name(value: Optional[str]) -> str:
    return (value or "").strip()


def is_portal_authenticated() -> bool:
    return bool(session.get("deep_learning_logged_in") and clean_login_name(session.get("deep_learning_username")))


def safe_next_path(candidate: Optional[str]) -> str:
    value = (candidate or "").strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return ""
    if not value.startswith("/"):
        return ""
    return value


def login_error_message(exc: Exception) -> str:
    message = clean_display_text(str(exc))
    return f"登录服务暂时不可用：{message}" if message else "登录服务暂时不可用，请稍后重试。"


def registration_error_message(exc: Exception) -> str:
    message = clean_display_text(str(exc))
    return f"注册服务暂时不可用：{message}" if message else "注册服务暂时不可用，请稍后重试。"


def auth_backend_ready() -> bool:
    return AUTH_BACKEND_ERROR is None and storage_backend_ready()


def auth_backend_notice() -> str:
    notice = storage_backend_notice()
    if notice:
        return notice
    if AUTH_BACKEND_ERROR is not None:
        return login_error_message(AUTH_BACKEND_ERROR)
    if False:  # Legacy Firebase-only notice path.
        return "当前尚未配置 Firebase 服务账号。请先在项目根目录的 .env 或 .env.local 中填写 FIREBASE_CREDENTIALS。"
    return ""


def validate_registration_form(username: str, password: str, confirm_password: str) -> Optional[str]:
    if not username or not password or not confirm_password:
        return "请完整填写注册信息。"
    if not USERNAME_RE.fullmatch(username):
        return "账号名需为 2 到 40 个字符，只允许字母、数字、点、下划线和连字符。"
    if password != confirm_password:
        return "两次输入的密码不一致。"
    if len(password) < 2:
        return "密码长度至少为 2 个字符。"
    return None


def should_refresh_chat_title(
    title: Optional[str],
    *,
    title_generated: bool = False,
    message_count: int = 0,
) -> bool:
    text = " ".join(str(title or "").strip().split())
    if not text or text == "新对话":
        return True
    if not title_generated and int(message_count or 0) >= 2:
        return True
    if len(text) > 28:
        return True
    if "?" in text or "？" in text:
        return True
    return False


def normalize_material_relative_path(filename: str) -> Path:
    cleaned = str(filename or "").replace("\\", "/").strip()
    parts = [part for part in Path(cleaned).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("缺少讲义路径。")
    return Path(*parts)


def material_root_candidates() -> List[Path]:
    candidates: List[Path] = []
    env_root = os.getenv("DEEP_LEARNING_MATERIAL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            ROOT_DIR / "deep_learning_materials",
            APP_DIR / "materials",
        ]
    )
    rows: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    return rows


@lru_cache(maxsize=512)
def locate_material(filename: str) -> Optional[Path]:
    try:
        relative_path = normalize_material_relative_path(filename)
    except ValueError:
        return None
    for root in material_root_candidates():
        candidate = root / relative_path
        if candidate.exists() and candidate.is_file():
            return candidate
    target_name = relative_path.name.lower()
    for root in material_root_candidates():
        if not root.exists():
            continue
        for candidate in root.rglob(relative_path.name):
            if candidate.is_file() and candidate.name.lower() == target_name:
                return candidate
    return None


def _material_params(item: Dict[str, Any]) -> Dict[str, Any]:
    params = {"source": str(item.get("source") or "").replace("\\", "/")}
    if item.get("unit_type"):
        params["unit_type"] = item["unit_type"]
    if item.get("unit_index") not in (None, ""):
        params["unit_index"] = item["unit_index"]
    if item.get("chunk_index") not in (None, ""):
        params["chunk_index"] = item["chunk_index"]
    return params


def build_material_reference_url(item: Dict[str, Any]) -> str:
    return url_for("deep_learning_material_reference", **_material_params(item))


def build_material_open_url(item: Dict[str, Any]) -> str:
    return url_for("deep_learning_material_open", **_material_params(item))


def resolve_material_open_target(source: str, *, unit_type: str = "", unit_index: int = 0) -> Optional[Dict[str, str]]:
    path = locate_material(source)
    if not path:
        return None
    page_index = int(unit_index or 0)
    base_url = url_for("deep_learning_material_asset", filename=source.replace("\\", "/"))
    if page_index > 0:
        return {"url": f"{base_url}#page={page_index}", "label": f"打开 PDF 第 {page_index} 页"}
    return {"url": base_url, "label": "打开 PDF"}


def enrich_material_reference(item: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(item)
    source = str(enriched.get("source") or "").replace("\\", "/").strip()
    if not source:
        return enriched
    enriched["display_source"] = Path(source).name
    enriched["material_url"] = build_material_open_url(enriched)
    enriched["reference_url"] = build_material_reference_url(enriched)
    return enriched


def enrich_review_references_payload(value: Any) -> Any:
    if isinstance(value, dict):
        enriched: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "review_refs" and isinstance(item, list):
                enriched[key] = [
                    enrich_material_reference(ref) if isinstance(ref, dict) else ref
                    for ref in item
                ]
                continue
            enriched[key] = enrich_review_references_payload(item)
        return enriched
    if isinstance(value, list):
        return [enrich_review_references_payload(item) for item in value]
    return value


def enrich_citation_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            rows.append(enrich_material_reference(item))
    return rows


def enrich_message_payloads(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in messages or []:
        payload = dict(item or {})
        payload["citations"] = enrich_citation_list(list(payload.get("citations") or []))
        rows.append(payload)
    return rows


def default_learning_report_context(summary: Dict[str, Any]) -> Dict[str, Any]:
    base_summary = {
        "answered": int(summary.get("answered") or 0),
        "correct": int(summary.get("correct") or 0),
        "wrong": int(summary.get("wrong") or 0),
        "accuracy": float(summary.get("accuracy") or 0.0),
        "explored_points": int(summary.get("explored_points") or 0),
        "mastered_points": int(summary.get("mastered_points") or 0),
        "weak_points": int(summary.get("weak_points") or 0),
        "recent_accuracy": float(summary.get("recent_accuracy") or 0.0),
        "trend_label": str(summary.get("trend_label") or "准备中"),
        "trend_note": str(summary.get("trend_note") or "系统正在整理你的最新练习表现。"),
    }
    return {
        "summary": base_summary,
        "kp_stats": [],
        "weak_points": [],
        "strength_points": [],
        "review_queue": [],
        "recommendations": [],
        "recent_attempts": [],
        "follow_up_questions": [],
        "activity": {
            "total_attempts": base_summary["answered"],
            "recent_accuracy": base_summary["recent_accuracy"],
        },
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    app.config["SECRET_KEY"] = os.getenv("DEEP_LEARNING_PORTAL_SECRET", "").strip() or "deep-learning-portal-secret"
    app.config["SESSION_COOKIE_NAME"] = "deep_learning_portal_session"
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    if get_storage_backend_name() == "postgresql" and storage_backend_ready():
        ensure_schema()

    kb = DeepLearningKnowledgeBase()
    chat_pipeline = DeepLearningChatPipeline(kb)
    redis_cache = RedisJsonCache()
    runtime_services_ready = get_storage_backend_name() == "postgresql" and storage_backend_ready()
    store: Optional[Any] = None
    chat_store: Optional[Any] = None

    def get_store() -> ProgressStore:
        nonlocal store
        if ProgressStore is None:
            raise AUTH_BACKEND_ERROR or RuntimeError(auth_backend_notice() or "Storage backend is unavailable.")
        if store is None:
            store = ProgressStore(data_path=APP_DIR / "data" / "progress.json")
        return store

    def get_chat_store() -> ChatSessionStore:
        nonlocal chat_store
        if ChatSessionStore is None:
            raise AUTH_BACKEND_ERROR or RuntimeError(auth_backend_notice() or "Storage backend is unavailable.")
        if chat_store is None:
            chat_store = ChatSessionStore()
        return chat_store

    chat_service: Optional[AsyncChatService] = None
    report_service: Optional[LearningReportSnapshotService] = None
    if runtime_services_ready:
        chat_service = AsyncChatService(
            kb=kb,
            chat_pipeline=chat_pipeline,
            chat_store_factory=get_chat_store,
            redis_cache=redis_cache,
        )
        report_service = LearningReportSnapshotService(
            kb=kb,
            progress_store_factory=get_store,
            redis_cache=redis_cache,
        )
        chat_service.start()
        report_service.start()

    def analytics_cache() -> Dict[str, Any]:
        cache = getattr(g, "deep_learning_analytics_cache", None)
        if cache is None:
            cache = {}
            g.deep_learning_analytics_cache = cache
        return cache

    def get_attempt_state_cached(user_id: str) -> Dict[str, Any]:
        cache_key = f"attempt_state::{user_id}"
        cached = analytics_cache().get(cache_key)
        if cached is not None:
            return cached
        state = build_attempt_state(kb, get_store(), user_id)
        analytics_cache()[cache_key] = state
        return state

    def get_dashboard_context_cached(user_id: str) -> Dict[str, Any]:
        cache_key = f"dashboard_context::{user_id}"
        cached = analytics_cache().get(cache_key)
        if cached is not None:
            return cached
        context = build_dashboard_context(kb, get_store(), user_id, state=get_attempt_state_cached(user_id))
        analytics_cache()[cache_key] = context
        return context

    def get_learning_report_view_cached(user_id: str) -> Dict[str, Any]:
        cache_key = f"report_view::{user_id}"
        cached = analytics_cache().get(cache_key)
        if cached is not None:
            return cached
        progress_store = get_store()
        signature = (
            progress_store.learning_report_signature(user_id)
            if hasattr(progress_store, "learning_report_signature")
            else {"attempt_count": progress_store.attempt_count(user_id)}
        )
        if report_service is None:
            context = default_learning_report_context(progress_store.summary(user_id))
            context["report_status"] = {
                "fresh": False,
                "pending": False,
                "state": "unavailable",
                "signature_key": signature_to_key(signature),
                "error_message": auth_backend_notice() or "学习报告服务尚未就绪。",
            }
            analytics_cache()[cache_key] = context
            return context

        load_state = report_service.load_for_request(user_id, signature)
        context = enrich_review_references_payload(
            load_state.get("payload") or default_learning_report_context(progress_store.summary(user_id))
        )
        context["report_status"] = {
            "fresh": bool(load_state.get("fresh")),
            "pending": bool(load_state.get("pending")),
            "state": str(load_state.get("status") or "queued"),
            "signature_key": signature_to_key(signature),
            "error_message": str(load_state.get("error_message") or ""),
        }
        analytics_cache()[cache_key] = context
        return context

    def render_login_page(
        *,
        error_message: str = "",
        success_message: str = "",
        username_value: str = "",
        next_path: str = "",
        form_mode: str = "login",
        status_code: int = 200,
    ) -> Any:
        resolved_next = safe_next_path(next_path) or url_for("chatapi_deep_learning")
        return (
            render_template(
                "login_4186.html",
                error_message=error_message,
                success_message=success_message,
                username_value=username_value,
                next_path=resolved_next,
                auth_backend_ready=auth_backend_ready(),
                auth_backend_notice=auth_backend_notice(),
                form_mode=form_mode if form_mode in {"login", "register"} else "login",
                login_url=url_for("login_deep_learning"),
                register_url=url_for("register_deep_learning"),
            ),
            status_code,
        )

    @app.before_request
    def ensure_course_user() -> Optional[Any]:
        endpoint = request.endpoint or ""
        if endpoint in PUBLIC_ENDPOINTS:
            return None
        if not is_portal_authenticated():
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "请先登录。", "login_url": url_for("login_deep_learning")}), 401
            next_path = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            return redirect(url_for("login_deep_learning", next=next_path))

        username = clean_login_name(session.get("deep_learning_username"))
        if not username:
            session.clear()
            return redirect(url_for("login_deep_learning"))
        session["deep_learning_user_id"] = username
        session["deep_learning_account_name"] = clean_account_name(username) or username
        get_store().ensure_user(username, display_name=username, account_name=username)
        return None

    @app.context_processor
    def inject_course_nav() -> Dict[str, Any]:
        if not is_portal_authenticated():
            return {
                "course4186_profile": {"display_name": "学生", "account_name": ""},
                "course4186_summary": {"answered": 0, "accuracy": 0.0},
            }
        user_id = clean_login_name(session.get("deep_learning_user_id"))
        if not user_id:
            return {
                "course4186_profile": {"display_name": "学生", "account_name": ""},
                "course4186_summary": {"answered": 0, "accuracy": 0.0},
            }
        try:
            progress_store = get_store()
            dashboard_context = get_dashboard_context_cached(user_id)
            profile = progress_store.get_user(user_id)
        except Exception:
            dashboard_context = {"summary": {"answered": 0, "accuracy": 0.0}}
            profile = {"display_name": user_id, "account_name": user_id}
        return {"course4186_profile": profile, "course4186_summary": dashboard_context["summary"]}

    @app.get("/")
    def index() -> Any:
        return redirect(url_for("chatapi_deep_learning"))

    @app.route("/login_deep_learning", methods=["GET", "POST"])
    def login_deep_learning() -> Any:
        if is_portal_authenticated():
            return redirect(safe_next_path(request.values.get("next")) or url_for("chatapi_deep_learning"))
        if request.method == "GET":
            success_message = "注册成功，请使用新账号登录。" if request.args.get("registered") == "1" else ""
            return render_login_page(
                success_message=success_message,
                username_value=clean_login_name(request.args.get("username", "")),
                next_path=request.args.get("next", ""),
                form_mode="login",
            )

        username = clean_login_name(request.form.get("username"))
        password = request.form.get("password") or ""
        next_path = request.form.get("next") or ""

        if not auth_backend_ready() or User is None or check_password_hash is None:
            return render_login_page(
                error_message=auth_backend_notice() or login_error_message(AUTH_BACKEND_ERROR or RuntimeError("Unknown backend error")),
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=503,
            )
        if not username or not password:
            return render_login_page(
                error_message="请输入账号和密码。",
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=400,
            )

        try:
            user = User.get_by_username(username)
        except Exception as exc:
            return render_login_page(
                error_message=login_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=503,
            )

        if not user or not check_password_hash(user.password, password):
            return render_login_page(
                error_message="账号或密码错误。",
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=401,
            )

        session.clear()
        session["deep_learning_logged_in"] = True
        session["deep_learning_user_id"] = user.username
        session["deep_learning_username"] = user.username
        session["deep_learning_is_admin"] = bool(getattr(user, "is_admin", False))
        session["deep_learning_account_name"] = clean_account_name(user.username) or user.username
        try:
            get_store().ensure_user(user.username, display_name=user.username, account_name=user.username)
        except Exception as exc:
            session.clear()
            return render_login_page(
                error_message=login_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=503,
            )
        return redirect(safe_next_path(next_path) or url_for("chatapi_deep_learning"))

    @app.route("/register_deep_learning", methods=["GET", "POST"])
    def register_deep_learning() -> Any:
        if is_portal_authenticated():
            return redirect(safe_next_path(request.values.get("next")) or url_for("chatapi_deep_learning"))
        if request.method == "GET":
            return render_login_page(
                username_value=clean_login_name(request.args.get("username", "")),
                next_path=request.args.get("next", ""),
                form_mode="register",
            )

        username = clean_login_name(request.form.get("username"))
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        next_path = request.form.get("next") or ""

        if not auth_backend_ready() or User is None or generate_password_hash is None:
            return render_login_page(
                error_message=auth_backend_notice() or registration_error_message(AUTH_BACKEND_ERROR or RuntimeError("Unknown backend error")),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        validation_error = validate_registration_form(username, password, confirm_password)
        if validation_error:
            return render_login_page(
                error_message=validation_error,
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=400,
            )

        try:
            existing_user = User.get_by_username(username)
        except Exception as exc:
            return render_login_page(
                error_message=registration_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        if existing_user:
            return render_login_page(
                error_message="该账号名已被使用，请更换后再试。",
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=409,
            )

        try:
            user = User(username, generate_password_hash(password), False)
            user.save()
        except Exception as exc:
            return render_login_page(
                error_message=registration_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        return redirect(
            url_for(
                "login_deep_learning",
                registered="1",
                username=username,
                next=safe_next_path(next_path) or url_for("chatapi_deep_learning"),
            )
        )

    @app.get("/logout_deep_learning")
    def logout_deep_learning() -> Any:
        session.clear()
        return redirect(url_for("login_deep_learning"))

    @app.get("/healthz")
    def healthz() -> Any:
        pool_info: Optional[Dict[str, Any]] = None
        chat_snapshot = chat_service.snapshot() if chat_service is not None else {"available": False}
        report_snapshot = report_service.snapshot() if report_service is not None else {"available": False}
        if get_storage_backend_name() == "postgresql":
            try:
                from pg_support.connection import pool_status

                pool_info = pool_status()
            except Exception as exc:
                pool_info = {"enabled": True, "available": False, "error": clean_display_text(str(exc))[:220]}
        return jsonify(
            {
                "ok": True,
                "knowledge_points": len(kb.list_knowledge_points()),
                "questions": len(kb.questions),
                "storage_backend": get_storage_backend_name(),
                "storage_ready": auth_backend_ready(),
                "chat_gate": chat_snapshot,
                "chat_queue": chat_snapshot,
                "learning_report_cache": report_snapshot,
                "learning_report_jobs": report_snapshot,
                "redis": redis_cache.stats(),
                "pg_pool": pool_info,
            }
        )

    @app.get("/chatapi_deep_learning")
    @app.get("/chatapi_deep_learning.html")
    def chatapi_deep_learning() -> Any:
        profile = get_store().get_user(session["deep_learning_user_id"])
        return render_template(
            "chatapi_4186.html",
            display_name=profile["display_name"],
            dense_enabled=kb.dense_enabled,
        )

    @app.get("/deep-learning/materials/<path:filename>")
    def deep_learning_material_asset(filename: str) -> Any:
        resolved = locate_material(filename)
        if not resolved or resolved.suffix.lower() != ".pdf":
            abort(404)
        return send_file(resolved, conditional=True)

    @app.get("/deep-learning/materials/open")
    def deep_learning_material_open() -> Any:
        source = (request.args.get("source") or "").strip()
        unit_type = (request.args.get("unit_type") or "").strip()
        try:
            unit_index = int(request.args.get("unit_index", "0") or 0)
        except ValueError:
            unit_index = 0
        target = resolve_material_open_target(source, unit_type=unit_type, unit_index=unit_index)
        if target:
            return redirect(target["url"])
        return redirect(url_for("deep_learning_material_reference", **_material_params(request.args.to_dict())))

    @app.get("/deep-learning/reference")
    def deep_learning_material_reference() -> Any:
        source = (request.args.get("source") or "").strip()
        unit_type = (request.args.get("unit_type") or "").strip()
        try:
            unit_index = int(request.args.get("unit_index", "0") or 0)
        except ValueError:
            unit_index = 0
        try:
            chunk_index = int(request.args.get("chunk_index", "0") or 0)
        except ValueError:
            chunk_index = 0
        reference = kb.reference_context(source, unit_type=unit_type, unit_index=unit_index, chunk_index=chunk_index)
        if not reference:
            abort(404)
        reference = enrich_material_reference(reference)
        target = resolve_material_open_target(reference["source"], unit_type=reference.get("unit_type"), unit_index=reference.get("unit_index"))
        return render_template(
            "source_reference_4186.html",
            reference=reference,
            open_material_url=(target or {}).get("url", ""),
            open_material_label=(target or {}).get("label", ""),
        )

    @app.get("/deep-learning/artifacts/<path:filename>")
    def deep_learning_artifact_asset(filename: str) -> Any:
        return send_from_directory(kb.artifact_dir, filename)

    @app.post("/api/deep-learning/profile")
    def update_profile() -> Any:
        payload = request.get_json(silent=True) or {}
        progress_store = get_store()
        profile = progress_store.set_display_name(session["deep_learning_user_id"], payload.get("display_name", ""))
        return jsonify({"ok": True, "profile": profile, "summary": progress_store.summary(session["deep_learning_user_id"])})

    @app.post("/api/deep-learning/chat")
    def api_chat_deep_learning() -> Any:
        try:
            payload = request.get_json(silent=True) or {}
            message = (payload.get("message") or "").strip()
            session_id = str(payload.get("session_id") or "").strip()
            if not message:
                return jsonify({"ok": False, "error": "请输入问题后再发送。"}), 400
            if chat_service is None:
                return jsonify({"ok": False, "error": auth_backend_notice() or "问答服务尚未就绪。"}), 503

            user_id = session["deep_learning_user_id"]
            chat_store = get_chat_store()
            session_data = chat_store.get_or_create_session(user_id, session_id=session_id)
            job = chat_service.enqueue(user_id=user_id, session_id=session_data["session_id"], message=message)
            return jsonify(
                {
                    "ok": True,
                    "accepted": True,
                    "job": job,
                    "session": session_data,
                }
            ), 202
        except ChatJobAlreadyActiveError as exc:
            return jsonify(
                {
                    "ok": False,
                    "error": "你上一条提问仍在处理中，请等待当前回答完成后再继续发送。",
                    "active_job": exc.job,
                    "retry_after_seconds": int((exc.job or {}).get("retry_after_seconds") or 0),
                }
            ), 429
        except ChatQueueFullError as exc:
            if exc.reason == "wait_too_long":
                error_message = "当前提问人数较多，预计等待时间过长，请稍后再试。"
                details = f"系统估计等待约 {exc.estimated_wait_seconds} 秒，建议 {exc.retry_after_seconds} 秒后重试。"
            else:
                error_message = "当前提问人数较多，系统正在排队处理中，请稍后再试。"
                details = f"当前排队任务约 {exc.queue_size} 个，系统上限为 {exc.max_queue_size} 个。"
            return jsonify(
                {
                    "ok": False,
                    "error": error_message,
                    "details": details,
                    "reason": exc.reason,
                    "queue": exc.snapshot,
                    "estimated_wait_seconds": exc.estimated_wait_seconds,
                    "max_estimated_wait_seconds": exc.max_estimated_wait_seconds,
                    "retry_after_seconds": exc.retry_after_seconds,
                }
            ), 503
        except Exception as exc:
            app.logger.exception("Deep learning chat API failed.")
            return jsonify({"ok": False, "error": "服务器暂时无法生成回答，请稍后再试。", "details": clean_display_text(str(exc))[:220]}), 500

    @app.get("/api/deep-learning/chat/jobs/<string:job_id>")
    def api_chat_job_detail_deep_learning(job_id: str) -> Any:
        if chat_service is None:
            return jsonify({"ok": False, "error": auth_backend_notice() or "问答服务尚未就绪。"}), 503
        job = chat_service.get_job(session["deep_learning_user_id"], job_id)
        if not job:
            return jsonify({"ok": False, "error": "未找到该聊天任务。"}), 404
        payload = dict(job)
        result = dict(payload.get("result") or {})
        if result:
            result["citations"] = enrich_citation_list(list(result.get("citations") or []))
            payload["result"] = result
        return jsonify({"ok": True, "job": payload})

    @app.get("/api/deep-learning/chat/sessions")
    def api_chat_sessions_deep_learning() -> Any:
        user_id = session["deep_learning_user_id"]
        chat_store = get_chat_store()
        sessions = chat_store.list_sessions(user_id)
        refreshed = False
        for item in sessions:
            if not should_refresh_chat_title(item.get("title"), title_generated=bool(item.get("title_generated")), message_count=int(item.get("message_count") or 0)):
                continue
            messages = chat_store.get_messages(user_id, item["session_id"])
            user_message = next((row.get("content", "") for row in reversed(messages) if row.get("role") == "user" and row.get("content")), "")
            assistant_message = next((row.get("content", "") for row in reversed(messages) if row.get("role") == "assistant" and row.get("content")), "")
            if not user_message:
                continue
            suggested = kb.suggest_session_title(user_message, assistant_message)
            updated = chat_store.set_session_title(user_id, item["session_id"], suggested, generated=True)
            if updated:
                refreshed = True
        if refreshed:
            sessions = chat_store.list_sessions(user_id)
        return jsonify({"ok": True, "sessions": sessions})

    @app.post("/api/deep-learning/chat/sessions")
    def api_create_chat_session_deep_learning() -> Any:
        payload = request.get_json(silent=True) or {}
        session_data = get_chat_store().create_session(session["deep_learning_user_id"], title=str(payload.get("title") or "新对话"))
        return jsonify({"ok": True, "session": session_data}), 201

    @app.get("/api/deep-learning/chat/sessions/<string:session_id>")
    def api_chat_session_detail_deep_learning(session_id: str) -> Any:
        chat_store = get_chat_store()
        session_data = chat_store.get_session(session["deep_learning_user_id"], session_id)
        if not session_data:
            return jsonify({"ok": False, "error": "未找到该对话。"}), 404
        messages = enrich_message_payloads(chat_store.get_messages(session["deep_learning_user_id"], session_id))
        return jsonify({"ok": True, "session": session_data, "messages": messages})

    @app.delete("/api/deep-learning/chat/sessions/<string:session_id>")
    def api_delete_chat_session_deep_learning(session_id: str) -> Any:
        deleted = get_chat_store().delete_session(session["deep_learning_user_id"], session_id)
        if not deleted:
            return jsonify({"ok": False, "error": "未找到该对话。"}), 404
        return jsonify({"ok": True})

    @app.get("/quiz_deep_learning")
    def quiz_dashboard_deep_learning() -> Any:
        progress_store = get_store()
        dashboard_context = get_dashboard_context_cached(session["deep_learning_user_id"])
        return render_template("quiz_dashboard_4186.html", kps=dashboard_context["kps"], profile=progress_store.get_user(session["deep_learning_user_id"]))

    @app.route("/quiz_deep_learning/practice/<string:kp_id>", methods=["GET", "POST"])
    def quiz_practice_deep_learning(kp_id: str) -> Any:
        kp = kb.get_kp(kp_id)
        if not kp:
            return "未找到该知识点。", 404
        questions = kp.get("questions", [])
        results_map: Dict[str, Dict[str, Any]] = {}
        summary = None
        practice_note = "请选择答案并提交。提交后，系统会展示正确答案、简要解释，以及建议回看的讲义页。"

        if request.method == "POST":
            submission_rows: List[Dict[str, Any]] = []
            answered = 0
            correct = 0
            for question in questions:
                field_name = f"answer_{question['question_id']}"
                user_answer = (request.form.get(field_name) or "").strip()
                result = grade_question(question, kp, user_answer)
                results_map[question["question_id"]] = result
                if user_answer:
                    answered += 1
                    if result["is_correct"]:
                        correct += 1
                    submission_rows.append(
                        {
                            "question_id": question["question_id"],
                            "question_type": question.get("question_type"),
                            "question": question.get("question"),
                            "submitted_answer": user_answer,
                            "reference_answer": question.get("correct_option") or question.get("answer"),
                            "is_correct": result["is_correct"],
                        }
                    )
            summary = {
                "answered": answered,
                "correct": correct,
                "wrong": max(answered - correct, 0),
                "accuracy": round((correct / answered) * 100, 1) if answered else 0.0,
            }
            if submission_rows:
                user_id = session["deep_learning_user_id"]
                progress_store = get_store()
                progress_store.record_attempts(user_id, kp_id=kp["kp_id"], kp_name=kp["name"], results=submission_rows)
                if report_service is not None:
                    signature = (
                        progress_store.learning_report_signature(user_id)
                        if hasattr(progress_store, "learning_report_signature")
                        else {"attempt_count": progress_store.attempt_count(user_id)}
                    )
                    report_service.schedule_refresh(user_id, signature)

        return render_template(
            "quiz_practice_4186.html",
            kp=kp,
            questions=questions,
            results_map=results_map,
            summary=summary,
            practice_note=practice_note,
        )

    @app.get("/learning_report_deep_learning")
    @app.get("/learning_report_deep_learning.html")
    @app.get("/quiz_deep_learning/analysis")
    def quiz_analysis_deep_learning() -> Any:
        progress_store = get_store()
        report_context = get_learning_report_view_cached(session["deep_learning_user_id"])
        return render_template("quiz_analysis_4186.html", profile=progress_store.get_user(session["deep_learning_user_id"]), **report_context)

    return app


def pick_choice_label(text: str) -> Optional[str]:
    match = ANSWER_LABEL_RE.search(text or "")
    return match.group(1).upper() if match else None


def display_answer_text(raw_value: str, option_lookup: Dict[str, str]) -> str:
    label = pick_choice_label(raw_value)
    if label and option_lookup.get(label):
        return f"{label}. {option_lookup[label]}"
    return raw_value.strip()


def grade_question(question: Dict[str, Any], kp: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    parsed_options = question.get("parsed_options") or []
    option_lookup = {option["label"]: option["text"] for option in parsed_options if option.get("label")}
    review_refs = [enrich_material_reference(ref) for ref in question.get("review_refs") or []]
    correct_label = str(question.get("correct_option") or "").upper().strip()
    if not user_answer:
        reference = f"{correct_label}. {option_lookup.get(correct_label, question.get('answer', ''))}" if correct_label else question.get("answer", "")
        return {
            "status": "not_answered",
            "is_correct": False,
            "submitted_answer": "",
            "submitted_answer_display": "",
            "reference_answer": reference,
            "explanation": question.get("explanation") or "",
            "feedback_note": "这道题尚未选择答案。",
            "review_refs": review_refs,
        }
    submitted_label = pick_choice_label(user_answer) or user_answer.strip().upper()
    is_correct = submitted_label == correct_label
    return {
        "status": "correct" if is_correct else "incorrect",
        "is_correct": is_correct,
        "submitted_answer": submitted_label,
        "submitted_answer_display": display_answer_text(submitted_label, option_lookup),
        "reference_answer": display_answer_text(correct_label, option_lookup),
        "explanation": question.get("explanation") or "",
        "feedback_note": "这道题掌握得比较稳定。" if is_correct else "建议根据下方讲义来源回看后再做一次同主题练习。",
        "review_refs": review_refs,
    }


def main() -> None:
    default_port_raw = os.getenv("DEEP_LEARNING_PORTAL_PORT", "50225") or "50225"
    try:
        default_port = max(int(default_port_raw), 1)
    except ValueError:
        default_port = 50225
    parser = argparse.ArgumentParser(description="运行中文深度学习课程学习平台。")
    parser.add_argument("--host", default=os.getenv("DEEP_LEARNING_PORTAL_HOST", "").strip() or "0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    app = create_app()
    serve_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
