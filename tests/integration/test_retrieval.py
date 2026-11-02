from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pgvector.psycopg import register_vector
from typer.testing import CliRunner

from policy_assistant.cli import app
from policy_assistant.config import Settings
from policy_assistant.db.connection import connect
from policy_assistant.ingest.manifest import Manifest
from policy_assistant.ingest.pipeline import ingest
from policy_assistant.retrieval.access import Principal, allowed_document_ids
from policy_assistant.retrieval.bm25 import search_bm25
from policy_assistant.retrieval.context import retrieval_text
from policy_assistant.retrieval.dense import search_dense
from policy_assistant.retrieval.retriever import RetrievalMode, retrieve
from policy_assistant.text.terms import bm25_terms
from tests.fakes import HashingEmbedder, OverlapReranker

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES: list[RetrievalMode] = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _ingest_sample(
    manifest: Manifest,
    root: Path,
    settings: Settings,
    counter: Callable[[str], int],
    embedder: HashingEmbedder,
) -> None:
    ingest(
        manifest,
        repo_root=root,
        only_ids=None,
        force=False,
        prune=False,
        count_tokens=counter,
        embedder=embedder,
        settings=settings,
    )


def test_index_data_is_complete_and_consistent(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT c.id, c.document_id, d.title, c.section_heading, c.text,
                   c.embedding, c.bm25_length
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
        assert rows
        for row in rows:
            chunk_id = int(row[0])
            embedding = _vector(row[5])
            assert embedding.shape == (1024,)
            assert abs(float(np.linalg.norm(embedding)) - 1.0) <= 1e-3
            terms = bm25_terms(retrieval_text(str(row[2]), row[3], str(row[4])))
            expected = Counter(terms)
            stored = dict(
                conn.execute(
                    "SELECT term, tf FROM chunk_terms WHERE chunk_id = %s",
                    (chunk_id,),
                ).fetchall()
            )
            assert stored == expected
            assert int(row[6]) == sum(expected.values())


def _reference_scores(
    records: list[tuple[int, str, Counter[str], int]],
    query_terms: list[str],
    k1: float,
    b: float,
    allowed: set[str] | None,
) -> list[tuple[int, float]]:
    n = len(records)
    avgdl = sum(record[3] for record in records) / n
    document_frequency: Counter[str] = Counter()
    for _, _, terms, _ in records:
        document_frequency.update(terms.keys())
    unique_terms = list(dict.fromkeys(query_terms))
    scores: list[tuple[int, float]] = []
    for chunk_id, document_id, terms, length in records:
        if allowed is not None and document_id not in allowed:
            continue
        score = 0.0
        for term in unique_terms:
            tf = terms.get(term, 0)
            if tf == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf / (tf + k1 * (1 - b + b * length / avgdl))
        if score > 0:
            scores.append((chunk_id, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def test_bm25_matches_python_reference(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.document_id, d.title, c.section_heading, c.text, c.bm25_length
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
        records = [
            (
                int(row[0]),
                str(row[1]),
                Counter(bm25_terms(retrieval_text(str(row[2]), row[3], str(row[4])))),
                int(row[5]),
            )
            for row in rows
        ]
        allowed_all = sorted({record[1] for record in records})
        queries = [
            "How many working days of annual leave do employees get?",
            "When is a medical certificate required for sick leave?",
            "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟",
            "زچگی کی چھٹی کتنی ہے؟",
            "annual leave 20 دن",
        ]
        for query in queries:
            terms = bm25_terms(query)
            hits = search_bm25(
                conn, terms, allowed_all, 1000, settings_for.bm25_k1, settings_for.bm25_b
            )
            expected = _reference_scores(
                records,
                terms,
                settings_for.bm25_k1,
                settings_for.bm25_b,
                allowed=None,
            )
            assert [hit.chunk_id for hit in hits] == [item[0] for item in expected]
            for hit, (_, score) in zip(hits, expected, strict=True):
                assert abs(hit.score - score) <= 1e-6

        allowed_staff = set(allowed_document_ids(conn, Principal(frozenset({"all_staff"}))))
        for query in queries[:2]:
            terms = bm25_terms(query)
            hits = search_bm25(
                conn,
                terms,
                sorted(allowed_staff),
                1000,
                settings_for.bm25_k1,
                settings_for.bm25_b,
            )
            expected = _reference_scores(
                records,
                terms,
                settings_for.bm25_k1,
                settings_for.bm25_b,
                allowed=allowed_staff,
            )
            assert [hit.chunk_id for hit in hits] == [item[0] for item in expected]
            for hit, (_, score) in zip(hits, expected, strict=True):
                assert abs(hit.score - score) <= 1e-6


def test_empty_principal_and_finance_acl(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    query = "Finance Director approval limit Rs. 2,000,000"
    with connect(db_dsn) as conn:
        for mode in MODES:
            empty = retrieve(
                conn,
                query,
                Principal(frozenset()),
                mode=mode,
                embedder=fake_embedder,
                reranker=fake_reranker,
                settings=settings_for,
            )
            assert empty.chunks == ()
            assert empty.allowed_document_count == 0

            staff = retrieve(
                conn,
                query,
                Principal(frozenset({"all_staff"})),
                mode=mode,
                embedder=fake_embedder,
                reranker=fake_reranker,
                settings=settings_for,
            )
            assert all(chunk.document_id != "finance-approvals-en" for chunk in staff.chunks)

        finance = retrieve(
            conn,
            query,
            Principal(frozenset({"finance"})),
            mode="bm25",
            embedder=fake_embedder,
            reranker=fake_reranker,
            settings=settings_for,
        )
        assert finance.chunks[0].document_id == "finance-approvals-en"


def test_randomized_acl_never_leaks(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    rng = random.Random(1234)
    groups = ["all_staff", "hr", "finance"]
    queries = [
        "How many days of annual leave do employees get?",
        "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟",
        "Finance Director approval limit",
    ]
    with connect(db_dsn) as conn:
        for _ in range(30):
            documents = [
                document.model_copy(
                    update={"acl": sorted(rng.sample(groups, rng.randint(1, len(groups))))}
                )
                for document in sample_manifest.documents
            ]
            variant = sample_manifest.model_copy(update={"documents": documents})
            _ingest_sample(variant, sample_repo, settings_for, fake_counter, fake_embedder)
            principal = Principal(frozenset(rng.sample(groups, rng.randint(0, len(groups)))))
            allowed = set(allowed_document_ids(conn, principal))
            for query in queries:
                for mode in MODES:
                    result = retrieve(
                        conn,
                        query,
                        principal,
                        mode=mode,
                        embedder=fake_embedder,
                        reranker=fake_reranker,
                        settings=settings_for,
                    )
                    assert all(chunk.document_id in allowed for chunk in result.chunks)


def test_filtered_dense_returns_k_allowed(db_dsn: str, fake_embedder: HashingEmbedder) -> None:
    with connect(db_dsn) as conn:
        register_vector(conn)
        conn.execute("INSERT INTO groups (id, description) VALUES ('all_staff', 'Everyone')")
        documents = []
        chunks = []
        for document_index in range(20):
            document_id = f"synth-{document_index:02d}"
            documents.append(
                (
                    document_id,
                    f"Synthetic {document_index}",
                    "en",
                    "txt",
                    "a" * 64,
                    "Fictional.",
                    1,
                    "test",
                )
            )
            for ordinal in range(10):
                text = f"synthetic chunk {document_index}-{ordinal} about leave"
                embedding = fake_embedder.embed([text])[0]
                chunks.append(
                    (
                        document_id,
                        ordinal,
                        [text],
                        text,
                        "en",
                        1,
                        1,
                        1,
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        embedding,
                        1,
                    )
                )
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO documents (
                    id, title, language, file_type, source_sha256, license_note,
                    page_count, pipeline_fingerprint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                documents,
            )
            cursor.executemany(
                """
                INSERT INTO chunks (
                    document_id, ordinal, sentences, text, language, page_start,
                    page_end, token_count, content_sha256, embedding, bm25_length
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                chunks,
            )
            cursor.executemany(
                "INSERT INTO document_acl (document_id, group_id) VALUES (%s, 'all_staff')",
                [("synth-00",), ("synth-01",)],
            )
        allowed = allowed_document_ids(conn, Principal(frozenset({"all_staff"})))
        assert sorted(allowed) == ["synth-00", "synth-01"]
        query_vector = fake_embedder.embed(["leave policy"])[0]
        hits = search_dense(conn, query_vector, allowed, 8, 100)
        assert len(hits) == 8
        allowed_chunk_ids = {
            int(row[0])
            for row in conn.execute(
                "SELECT id FROM chunks WHERE document_id = ANY(%s)", (allowed,)
            ).fetchall()
        }
        assert all(hit.chunk_id in allowed_chunk_ids for hit in hits)


def test_cli_retrieve_refuses_incomplete_index(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    with connect(db_dsn) as conn:
        conn.execute("UPDATE chunks SET embedding = NULL WHERE id = (SELECT min(id) FROM chunks)")
    monkeypatch.setattr(
        "policy_assistant.cli.SentenceTransformerEmbedder",
        lambda settings: fake_embedder,
    )
    monkeypatch.setattr(
        "policy_assistant.cli.CrossEncoderReranker",
        lambda settings: fake_reranker,
    )
    result = CliRunner().invoke(
        app,
        ["retrieve", "annual leave", "--group", "all_staff"],
        env={"UPA_DATABASE_URL": db_dsn},
    )
    assert result.exit_code == 1, result.output
    assert "index incomplete: run upa ingest" in result.output


def test_cli_eval_writes_report_with_zero_leaks(
    db_dsn: str,
    sample_repo: Path,
    sample_manifest: Manifest,
    settings_for: Settings,
    fake_counter: Callable[[str], int],
    fake_embedder: HashingEmbedder,
    fake_reranker: OverlapReranker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ingest_sample(sample_manifest, sample_repo, settings_for, fake_counter, fake_embedder)
    dataset = REPO_ROOT / "evals" / "retrieval_seed.sample.jsonl"
    data_dir = tmp_path / "reports-data"
    monkeypatch.setattr(
        "policy_assistant.cli.SentenceTransformerEmbedder",
        lambda settings: fake_embedder,
    )
    monkeypatch.setattr(
        "policy_assistant.cli.CrossEncoderReranker",
        lambda settings: fake_reranker,
    )
    result = CliRunner().invoke(
        app,
        ["eval", "retrieval", "--dataset", str(dataset), "--k", "10"],
        env={
            "UPA_DATABASE_URL": db_dsn,
            "UPA_DATA_DIR": str(data_dir),
        },
    )
    assert result.exit_code == 0, result.output
    report_line = next(line for line in result.output.splitlines() if line.startswith("report: "))
    report_path = Path(report_line.removeprefix("report: "))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert report["pipeline_fingerprint"]
    assert report["models"]["embedding_revision"]
    for mode_report in report["modes"].values():
        assert mode_report["overall"]["leaks"] == 0
