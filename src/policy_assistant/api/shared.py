"""Helpers shared by JSON and server-rendered routes."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import Request, Response

from policy_assistant.answering.language import answer_language
from policy_assistant.answering.service import AnswerResult, answer
from policy_assistant.auth.passwords import (
    burn_verification,
    hash_password,
    needs_rehash,
    verify_password,
)
from policy_assistant.auth.sessions import SessionUser, create_session, load_session
from policy_assistant.auth.users import User, get_user, normalize_username
from policy_assistant.config import Settings
from policy_assistant.retrieval.access import Principal

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "upa_session"


class AskBusy(Exception):
    """The ask semaphore did not free in time."""


class AskFailed(Exception):
    """The answer pipeline raised an unexpected exception."""

    def __init__(self, language: Literal["ur", "en"]) -> None:
        super().__init__("answer failed")
        self.language = language


def session_user(request: Request) -> SessionUser | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with request.app.state.pool.connection() as conn:
        return load_session(conn, token)


def authenticate(request: Request, username: str, password: str) -> tuple[User, str, str] | None:
    settings: Settings = request.app.state.settings
    normalized = normalize_username(username)
    with request.app.state.pool.connection() as conn:
        row = conn.execute(
            "SELECT id, password_hash, is_active FROM users WHERE username = %s",
            (normalized,),
        ).fetchone()
        if row is None or not bool(row[2]):
            burn_verification(password)
            return None
        user_id = int(row[0])
        password_hash = str(row[1])
        if not verify_password(password_hash, password):
            return None
        if needs_rehash(password_hash):
            with conn.transaction():
                conn.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (hash_password(password), user_id),
                )
        token, csrf_token = create_session(conn, user_id, settings.session_ttl_hours)
        user = get_user(conn, normalized)
    if user is None:
        return None
    return user, token, csrf_token


def set_session_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="strict",
    )


def validate_question(settings: Settings, raw: str) -> str:
    question = raw.strip()
    if not 1 <= len(question) <= settings.question_max_chars:
        raise ValueError("invalid question length")
    return question


def run_ask(request: Request, user: SessionUser, question: str) -> AnswerResult:
    settings: Settings = request.app.state.settings
    slots = request.app.state.ask_slots
    if not slots.acquire(timeout=settings.ask_queue_timeout_seconds):
        raise AskBusy
    try:
        with request.app.state.pool.connection() as conn:
            try:
                return answer(
                    conn,
                    question,
                    Principal(user.groups),
                    actor=f"user:{user.username}",
                    embedder=request.app.state.embedder,
                    reranker=request.app.state.reranker,
                    model=request.app.state.answer_model,
                    settings=settings,
                )
            except Exception as exc:
                LOGGER.exception("ask failed")
                raise AskFailed(answer_language(question)) from exc
    finally:
        slots.release()
