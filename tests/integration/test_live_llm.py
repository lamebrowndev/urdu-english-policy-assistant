from __future__ import annotations

import os
from pathlib import Path

import pytest

from policy_assistant.answering.model import ClaudeAnswerModel
from policy_assistant.answering.service import answer
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from policy_assistant.ingest.tokens import load_bge_m3_counter
from policy_assistant.retrieval.access import Principal
from policy_assistant.retrieval.models import CrossEncoderReranker, SentenceTransformerEmbedder
from policy_assistant.text.language import script_letter_counts

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("UPA_LLM_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
        reason="live LLM tests require UPA_LLM_TESTS=1 and ANTHROPIC_API_KEY",
    ),
]


def test_live_llm_english_urdu_and_restricted(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
) -> None:
    settings = settings_for
    counter = load_bge_m3_counter(settings.tokenizer_repo, settings.tokenizer_revision)
    embedder = SentenceTransformerEmbedder(settings)
    reranker = CrossEncoderReranker(settings)
    ingest(
        sample_manifest,
        repo_root=sample_repo,
        only_ids=None,
        force=False,
        prune=False,
        count_tokens=counter,
        embedder=embedder,
        settings=settings,
    )
    model = ClaudeAnswerModel(settings)
    with connect(db_dsn) as conn:
        english = answer(
            conn,
            "How many days of annual leave do employees get?",
            Principal(frozenset({"all_staff"})),
            actor="test:live",
            embedder=embedder,
            reranker=reranker,
            model=model,
            settings=settings,
        )
        urdu = answer(
            conn,
            "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟",
            Principal(frozenset({"all_staff"})),
            actor="test:live",
            embedder=embedder,
            reranker=reranker,
            model=model,
            settings=settings,
        )
        restricted = answer(
            conn,
            "What is the Finance Director's approval limit?",
            Principal(frozenset({"all_staff"})),
            actor="test:live",
            embedder=embedder,
            reranker=reranker,
            model=model,
            settings=settings,
        )

    assert english.outcome == "answered"
    assert any(
        source["document_id"] == "leave-policy-en" and "20" in source["cited_text"]
        for source in english.sources
    )
    assert urdu.outcome == "answered"
    arabic, latin = script_letter_counts(urdu.text)
    assert arabic > latin
    assert any(source["document_id"] == "leave-policy-ur" for source in urdu.sources)
    assert restricted.outcome.startswith("abstained_")
