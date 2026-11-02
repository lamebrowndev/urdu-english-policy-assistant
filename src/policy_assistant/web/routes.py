"""Server-rendered web routes."""

from __future__ import annotations

import hmac
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from policy_assistant.answering.prompts import error_text
from policy_assistant.api.shared import (
    SESSION_COOKIE,
    AskBusy,
    AskFailed,
    authenticate,
    clear_session_cookie,
    run_ask,
    session_user,
    set_session_cookie,
    validate_question,
)
from policy_assistant.auth.sessions import SessionUser, delete_session
from policy_assistant.reviews.store import (
    ReviewConflict,
    add_review,
    history,
)
from policy_assistant.reviews.store import (
    item as review_item,
)
from policy_assistant.reviews.store import (
    queue as review_queue,
)
from policy_assistant.web.i18n import t as translate
from policy_assistant.web.render import (
    cited_flags,
    page_label,
    rerank_display,
    segment_views,
    text_dir,
    text_lang,
    verdicts_for,
)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
)
_env.filters["text_lang"] = text_lang
_env.filters["text_dir"] = text_dir
_env.filters["page_label"] = page_label
_env.filters["rerank_display"] = rerank_display
templates = Jinja2Templates(env=_env)

router = APIRouter()
OUTCOMES = [
    "answered",
    "abstained_retrieval",
    "abstained_model",
    "abstained_uncited",
    "refused",
    "error",
]


def _ui_lang(request: Request) -> str:
    return "ur" if request.cookies.get("upa_lang") == "ur" else "en"


def _context(request: Request, user: SessionUser | None, **extra: Any) -> dict[str, Any]:
    lang = _ui_lang(request)
    current_path = request.url.path
    if request.url.query:
        current_path = f"{current_path}?{request.url.query}"
    context: dict[str, Any] = {
        "request": request,
        "ui": lang,
        "t": lambda key: translate(lang, key),
        "user": user,
        "current_path": current_path,
        "csrf_token": user.csrf_token if user is not None else "",
    }
    context.update(extra)
    return context


def _page(
    request: Request,
    user: SessionUser | None,
    template: str,
    *,
    status_code: int = 200,
    **extra: Any,
) -> Any:
    return templates.TemplateResponse(
        request=request,
        name=template,
        context=_context(request, user, **extra),
        status_code=status_code,
    )


def _error_page(
    request: Request, user: SessionUser | None, status_code: int, message_key: str
) -> Any:
    lang = _ui_lang(request)
    return _page(
        request,
        user,
        "error.html",
        status_code=status_code,
        message=translate(lang, message_key),
        page_title=translate(lang, "app_title"),
    )


def _origin_ok(request: Request) -> bool:
    expected = str(request.base_url).rstrip("/")
    origin = request.headers.get("origin")
    if origin is None:
        referer = request.headers.get("referer")
        if referer:
            parts = urlsplit(referer)
            if parts.scheme and parts.netloc:
                origin = f"{parts.scheme}://{parts.netloc}"
    return origin is not None and origin.rstrip("/") == expected


def _post_guard(request: Request, user: SessionUser, csrf_token: str) -> Any | None:
    if not _origin_ok(request):
        return _error_page(request, user, 403, "origin_error")
    if not csrf_token or not hmac.compare_digest(csrf_token, user.csrf_token):
        return _error_page(request, user, 403, "csrf_error")
    return None


def _encode_cursor(value: tuple[datetime, uuid.UUID]) -> str:
    return f"{value[0].isoformat()}|{value[1]}"


def _decode_cursor(value: str | None) -> tuple[datetime, uuid.UUID] | None:
    if not value:
        return None
    try:
        created, audit_id = value.split("|", 1)
        return datetime.fromisoformat(created), uuid.UUID(audit_id)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid cursor") from exc


def _older_url(path: str, cursor: tuple[datetime, uuid.UUID] | None) -> str | None:
    if cursor is None:
        return None
    return f"{path}?cursor={quote(_encode_cursor(cursor), safe='')}"


def _review_older_url(
    status: str,
    outcome: str | None,
    cursor: tuple[datetime, uuid.UUID] | None,
) -> str | None:
    if cursor is None:
        return None
    params = [f"status={quote(status, safe='')}"]
    if outcome:
        params.append(f"outcome={quote(outcome, safe='')}")
    params.append(f"cursor={quote(_encode_cursor(cursor), safe='')}")
    return "/review?" + "&".join(params)


def _chat_page(
    request: Request,
    user: SessionUser,
    *,
    status_code: int = 200,
    message: str | None = None,
    cursor: tuple[datetime, uuid.UUID] | None = None,
) -> Any:
    with request.app.state.pool.connection() as conn:
        items, next_cursor = history(conn, user, limit=20, cursor=cursor)
    return _page(
        request,
        user,
        "chat.html",
        status_code=status_code,
        items=items,
        older_url=_older_url("/", next_cursor),
        message=message,
        csrf_token=user.csrf_token,
        question_max_chars=request.app.state.settings.question_max_chars,
        page_title=translate(_ui_lang(request), "page_chat_title"),
    )


def _review_item_context(
    request: Request,
    user: SessionUser,
    payload: dict[str, Any],
    *,
    message: str | None = None,
) -> dict[str, Any]:
    answer = payload.get("answer") or {}
    sources = answer.get("sources") or []
    retrieval = payload.get("retrieval") or {}
    chunks: list[dict[str, Any]] = []
    for chunk in retrieval.get("chunks") or []:
        flags = cited_flags(chunk, sources)
        sentences = [
            {"text": sentence, "cited": cited}
            for sentence, cited in zip(chunk.get("sentences") or [], flags, strict=False)
        ]
        chunks.append({"chunk": chunk, "sentences": sentences})
    outcome = str(payload.get("outcome", "error"))
    return _context(
        request,
        user,
        item=payload,
        answer=answer,
        sources=sources,
        chunks=chunks,
        segments=segment_views(answer),
        llm=payload.get("llm"),
        outcome=outcome,
        flags=payload.get("flags") or [],
        verdicts=verdicts_for(outcome),
        show_form=outcome != "error",
        show_citation=outcome == "answered",
        csrf_token=user.csrf_token,
        message=message,
        page_title=translate(_ui_lang(request), "page_review_item_title"),
    )


def _review_item_page(
    request: Request,
    user: SessionUser,
    audit_id: uuid.UUID,
    *,
    status_code: int = 200,
    message: str | None = None,
) -> Any:
    with request.app.state.pool.connection() as conn:
        payload = review_item(conn, user, audit_id)
    if payload is None:
        return _error_page(request, user, 404, "not_found")
    return templates.TemplateResponse(
        request=request,
        name="review_item.html",
        context=_review_item_context(request, user, payload, message=message),
        status_code=status_code,
    )


@router.get("/login")
def login_page(request: Request) -> Any:
    user = session_user(request)
    if user is not None:
        return RedirectResponse("/", status_code=303)
    lang = _ui_lang(request)
    return _page(
        request,
        None,
        "login.html",
        page_title=translate(lang, "page_login_title"),
    )


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
) -> Any:
    if not _origin_ok(request):
        return _error_page(request, None, 403, "origin_error")
    authenticated = authenticate(request, username, password)
    if authenticated is None:
        lang = _ui_lang(request)
        return _page(
            request,
            None,
            "login.html",
            status_code=401,
            error=translate(lang, "login_error"),
            page_title=translate(lang, "page_login_title"),
        )
    _user, token, _csrf = authenticated
    response = RedirectResponse("/", status_code=303)
    set_session_cookie(response, request.app.state.settings, token)
    return response


@router.post("/logout")
def logout_submit(
    request: Request,
    csrf_token: str = Form(""),
) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    denied = _post_guard(request, user, csrf_token)
    if denied is not None:
        return denied
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with request.app.state.pool.connection() as conn:
            delete_session(conn, token)
    response = RedirectResponse("/login", status_code=303)
    clear_session_cookie(response, request.app.state.settings)
    return response


@router.get("/")
def chat_page(request: Request, cursor: str | None = None) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    try:
        decoded = _decode_cursor(cursor)
    except ValueError:
        return _chat_page(
            request,
            user,
            status_code=422,
            message=translate(_ui_lang(request), "invalid_request"),
        )
    return _chat_page(request, user, cursor=decoded)


@router.post("/ask")
def ask_submit(
    request: Request,
    question: str = Form(""),
    csrf_token: str = Form(""),
) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    denied = _post_guard(request, user, csrf_token)
    if denied is not None:
        return denied
    try:
        cleaned = validate_question(request.app.state.settings, question)
    except ValueError:
        return _chat_page(
            request,
            user,
            status_code=422,
            message=translate(_ui_lang(request), "invalid_request"),
        )
    try:
        result = run_ask(request, user, cleaned)
    except AskBusy:
        return _chat_page(
            request,
            user,
            status_code=503,
            message=translate(_ui_lang(request), "busy"),
        )
    except AskFailed as exc:
        return _chat_page(
            request,
            user,
            status_code=500,
            message=error_text(exc.language),
        )
    return RedirectResponse(f"/#a-{result.audit_id}", status_code=303)


@router.get("/review")
def review_queue_page(
    request: Request,
    status: str = "unreviewed",
    outcome: str | None = None,
    cursor: str | None = None,
) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if user.role not in {"reviewer", "admin"}:
        return _error_page(request, user, 403, "forbidden")
    try:
        decoded = _decode_cursor(cursor)
        with request.app.state.pool.connection() as conn:
            items, next_cursor = review_queue(
                conn,
                user,
                status=status,
                outcome=outcome,
                limit=20,
                cursor=decoded,
            )
    except ValueError:
        return _error_page(request, user, 422, "invalid_request")
    lang = _ui_lang(request)
    return _page(
        request,
        user,
        "review_queue.html",
        status_code=200,
        items=items,
        older_url=_review_older_url(status, outcome, next_cursor),
        selected_status=status,
        selected_outcome=outcome,
        outcomes=OUTCOMES,
        page_title=translate(lang, "page_review_title"),
    )


@router.get("/review/{audit_id}")
def review_item_page(request: Request, audit_id: uuid.UUID) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if user.role not in {"reviewer", "admin"}:
        return _error_page(request, user, 403, "forbidden")
    return _review_item_page(request, user, audit_id)


@router.post("/review/{audit_id}")
def review_submit(
    request: Request,
    audit_id: uuid.UUID,
    verdict: str = Form(""),
    citation_ok: str = Form(""),
    notes: str = Form(""),
    csrf_token: str = Form(""),
) -> Any:
    user = session_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if user.role not in {"reviewer", "admin"}:
        return _error_page(request, user, 403, "forbidden")
    denied = _post_guard(request, user, csrf_token)
    if denied is not None:
        return denied
    value = citation_ok.strip().lower()
    if value == "":
        citation_value: bool | None = None
    elif value == "true":
        citation_value = True
    elif value == "false":
        citation_value = False
    else:
        return _review_item_page(
            request,
            user,
            audit_id,
            status_code=422,
            message=translate(_ui_lang(request), "invalid_request"),
        )
    try:
        with request.app.state.pool.connection() as conn:
            review = add_review(conn, user, audit_id, verdict, citation_value, notes)
    except ReviewConflict as exc:
        return _review_item_page(
            request,
            user,
            audit_id,
            status_code=409,
            message=str(exc),
        )
    except ValueError as exc:
        return _review_item_page(
            request,
            user,
            audit_id,
            status_code=422,
            message=str(exc),
        )
    if review is None:
        return _error_page(request, user, 404, "not_found")
    return RedirectResponse(f"/review/{audit_id}", status_code=303)


@router.get("/set-language")
def set_language(request: Request, lang: str = "en", next: str = "/") -> Any:
    target = next
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        target = "/"
    language = "ur" if lang == "ur" else "en"
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        "upa_lang",
        language,
        max_age=31536000,
        httponly=True,
        secure=request.app.state.settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )
    return response


def static_files() -> StaticFiles:
    return StaticFiles(directory=str(STATIC_DIR))


__all__ = ["router", "static_files", "templates"]
