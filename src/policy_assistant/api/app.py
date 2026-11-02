"""FastAPI application for accounts, asking, history, and reviews."""

from __future__ import annotations

import hmac
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, cast

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from policy_assistant.answering.model import build_answer_model
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
from policy_assistant.auth.users import User
from policy_assistant.config import Settings, get_settings
from policy_assistant.db.migrate import pending_migrations
from policy_assistant.retrieval.models import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from policy_assistant.retrieval.readiness import index_problem
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
from policy_assistant.web.routes import router as web_router
from policy_assistant.web.routes import static_files

LOGGER = logging.getLogger(__name__)

CSP = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"


class _LockedEmbedder:
    def __init__(self, inner: Embedder, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, texts: list[str]) -> Any:
        with self._lock:
            return self._inner.embed(texts)


class _LockedReranker:
    def __init__(self, inner: Reranker, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            return self._inner.score(query, passages)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginBody(_StrictModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=1024)


class AskBody(_StrictModel):
    question: str = Field(max_length=10000)


class ReviewBody(_StrictModel):
    verdict: str = Field(max_length=32)
    citation_ok: bool | None = None
    notes: str = Field(default="", max_length=4000)


def _pool(request: Request) -> ConnectionPool[Any]:
    return cast(ConnectionPool[Any], request.app.state.pool)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _user_payload(user: SessionUser | User) -> dict[str, Any]:
    return {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "groups": sorted(user.groups),
    }


def _status(outcome: str) -> str:
    if outcome == "answered":
        return "answered"
    if outcome.startswith("abstained_") or outcome == "refused":
        return "abstained"
    return "error"


def _encode_cursor(value: tuple[datetime, uuid.UUID]) -> str:
    return f"{value[0].isoformat()}|{value[1]}"


def _decode_cursor(value: str | None) -> tuple[datetime, uuid.UUID] | None:
    if value is None:
        return None
    try:
        created, audit_id = value.split("|", 1)
        return datetime.fromisoformat(created), uuid.UUID(audit_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="invalid cursor") from exc


def current_user(request: Request) -> SessionUser:
    user = session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and request.url.path != "/api/auth/login"
    ):
        provided = request.headers.get("X-CSRF-Token")
        if not provided or not hmac.compare_digest(provided, user.csrf_token):
            raise HTTPException(status_code=403, detail="csrf token missing or invalid")
    return user


def require_reviewer(
    user: Annotated[SessionUser, Depends(current_user)],
) -> SessionUser:
    if user.role not in {"reviewer", "admin"}:
        raise HTTPException(status_code=403, detail="reviewer access required")
    return user


def create_app(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    answer_model: Any | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool: ConnectionPool[Any] = ConnectionPool(
            settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            kwargs={"application_name": "upa-api"},
            open=False,
        )
        try:
            try:
                pool.open(wait=True, timeout=10)
            except Exception as exc:
                raise RuntimeError(f"database pool failed to open: {exc}") from exc
            with pool.connection() as conn:
                pending = pending_migrations(conn)
                if pending:
                    names = ", ".join(migration.version for migration in pending)
                    raise RuntimeError(f"pending migrations: {names}")
                problem = index_problem(conn, settings)
                if problem is not None:
                    raise RuntimeError(problem)
            actual_embedder = embedder or SentenceTransformerEmbedder(settings)
            actual_reranker = reranker or CrossEncoderReranker(settings)
            actual_model = answer_model or build_answer_model(settings)
            lock = threading.Lock()
            app.state.pool = pool
            app.state.embedder = _LockedEmbedder(actual_embedder, lock)
            app.state.reranker = _LockedReranker(actual_reranker, lock)
            app.state.answer_model = actual_model
            app.state.ask_slots = threading.BoundedSemaphore(settings.max_concurrent_asks)
            app.state.settings = settings
            yield
        finally:
            pool.close()

    api = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)

    @api.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @api.post("/api/auth/login")
    def login(body: LoginBody, request: Request, response: Response) -> dict[str, Any]:
        authenticated = authenticate(request, body.username, body.password)
        if authenticated is None:
            raise HTTPException(status_code=401, detail="invalid username or password")
        user, token, csrf_token = authenticated
        set_session_cookie(response, _settings(request), token)
        return {"user": _user_payload(user), "csrf_token": csrf_token}

    @api.post("/api/auth/logout")
    def logout(
        request: Request,
        response: Response,
        user: Annotated[SessionUser, Depends(current_user)],
    ) -> Response:
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            with _pool(request).connection() as conn:
                delete_session(conn, token)
        clear_session_cookie(response, _settings(request))
        response.status_code = 204
        return response

    @api.get("/api/me")
    def me(
        user: Annotated[SessionUser, Depends(current_user)],
    ) -> dict[str, Any]:
        return {"user": _user_payload(user), "csrf_token": user.csrf_token}

    @api.post("/api/ask")
    def ask(
        body: AskBody,
        request: Request,
        user: Annotated[SessionUser, Depends(current_user)],
    ) -> Any:
        settings = _settings(request)
        try:
            question = validate_question(settings, body.question)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = run_ask(request, user, question)
        except AskBusy as exc:
            raise HTTPException(status_code=503, detail="busy") from exc
        except AskFailed as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "language": exc.language,
                    "text": error_text(exc.language),
                    "audit_id": None,
                    "sources": [],
                },
            )
        return {
            "audit_id": result.audit_id,
            "status": _status(result.outcome),
            "language": result.language,
            "text": result.text,
            "sources": list(result.sources),
        }

    @api.get("/api/history")
    def history_endpoint(
        request: Request,
        user: Annotated[SessionUser, Depends(current_user)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        decoded = _decode_cursor(cursor)
        with _pool(request).connection() as conn:
            items, next_cursor = history(conn, user, limit=limit, cursor=decoded)
        return {
            "items": items,
            "next_cursor": _encode_cursor(next_cursor) if next_cursor else None,
        }

    @api.get("/api/review/queue")
    def review_queue_endpoint(
        request: Request,
        user: Annotated[SessionUser, Depends(require_reviewer)],
        status: str = "unreviewed",
        outcome: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        decoded = _decode_cursor(cursor)
        try:
            with _pool(request).connection() as conn:
                items, next_cursor = review_queue(
                    conn,
                    user,
                    status=status,
                    outcome=outcome,
                    limit=limit,
                    cursor=decoded,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "items": items,
            "next_cursor": _encode_cursor(next_cursor) if next_cursor else None,
        }

    @api.get("/api/review/items/{audit_id}")
    def review_item_endpoint(
        request: Request,
        audit_id: uuid.UUID,
        user: Annotated[SessionUser, Depends(require_reviewer)],
    ) -> dict[str, Any]:
        with _pool(request).connection() as conn:
            payload = review_item(conn, user, audit_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="not found")
        return payload

    @api.post("/api/review/items/{audit_id}/reviews", status_code=201)
    def review_create_endpoint(
        request: Request,
        audit_id: uuid.UUID,
        body: ReviewBody,
        user: Annotated[SessionUser, Depends(require_reviewer)],
    ) -> dict[str, Any]:
        try:
            with _pool(request).connection() as conn:
                review = add_review(
                    conn,
                    user,
                    audit_id,
                    body.verdict,
                    body.citation_ok,
                    body.notes,
                )
        except ReviewConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if review is None:
            raise HTTPException(status_code=404, detail="not found")
        return review

    api.mount("/static", static_files(), name="static")
    api.include_router(web_router)

    @api.get("/healthz")
    def healthz(request: Request) -> Any:
        try:
            with _pool(request).connection(timeout=2) as conn:
                conn.execute("SELECT 1").fetchone()
        except psycopg.Error:
            return JSONResponse(status_code=503, content={"status": "error"})
        return {"status": "ok"}

    return api


def create_app_from_env() -> FastAPI:
    return create_app(get_settings())
