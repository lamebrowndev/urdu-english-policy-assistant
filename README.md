# Urdu–English Policy Assistant

A bilingual Urdu/English policy assistant whose answers must cite exact source
sentences, use only documents the asking user may see, and log every answer for
human review. Phase 1 built the ingestion foundation: a PostgreSQL schema,
Urdu-aware text normalization, PDF/TXT extraction with a fail-closed quality
gate, heading-aware chunking, and idempotent ingestion. Phase 2 adds hybrid
retrieval: BM25 in SQL, bge-m3 vectors in pgvector, reciprocal rank fusion, and
bge-reranker-v2-m3. Later phases add cited answers, access control and UI,
evaluation, and a showcase.

## Status

Phase 2 of 6 complete: hybrid retrieval. No retrieval or answer-accuracy numbers
appear here yet, because none have been measured on the real corpus. The seed
dataset is a pipeline check, not a performance result.

The sample corpus is fictional. The project does not redistribute third-party
documents: real manifests store a source URL and a pinned SHA-256, and fetched
files stay under `data/` (git-ignored).

## Quickstart

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make setup
make db-up
uv run upa db migrate
uv run upa ingest --manifest corpus/manifest.sample.yaml
uv run upa stats
uv run upa retrieve "How many days of annual leave do employees get?" --group all_staff
```

The first `ingest` or `retrieve` downloads bge-m3 and bge-reranker-v2-m3 from
Hugging Face: about 4.6 GB in total. Model files stay in the Hugging Face cache,
not in this repository.

`upa corpus inspect --manifest corpus/manifest.sample.yaml` extracts and scores
the corpus without touching the database. `upa eval retrieval --dataset
evals/retrieval_seed.sample.jsonl` runs the four retrieval modes over the seed
questions.

## What Phase 1 does

1. Validates `corpus/manifest.yaml` (or a `--manifest` override): unique ids,
   at least one ACL group per document, pinned SHA-256, translation links.
2. Extracts pages with pypdfium2 (PDF) or strict UTF-8 (TXT, split on form feed).
3. Cleans display text, measures per-page quality, and quarantines documents
   whose text layer is empty, private-use-heavy, wrong-script, low-lexicon, or
   reversed.
4. Chunks included pages on headings and sentence boundaries with the bge-m3
   tokenizer, capped at 350 tokens with 50-token overlap.
5. Stores documents, ACLs, all pages (included and excluded, with quality
   JSON), and chunks in one transaction per document. Unchanged documents are
   metadata/ACL-only updates, so chunk ids stay stable.

## What Phase 2 adds

1. Migration `0002_retrieval_index.sql`: a 1024-dimensional embedding column, a
   BM25 length column, the `chunk_terms` inverted index, and an HNSW cosine
   index.
2. A shared `bm25_terms()` function (Urdu folding, stopword removal, English
   Snowball stemming) used for both indexing and queries.
3. Ingest computes bge-m3 embeddings, BM25 term counts, and `bm25_length` before
   the write transaction.
4. Retrieval offers `bm25`, `dense`, `hybrid`, and `hybrid_rerank` modes. ACL
   filtering runs inside the BM25 and dense SQL, never as a post-filter.
5. `upa eval retrieval` reports hit@k, recall@k, MRR@k, leaks, and per-stage
   latency over a JSONL dataset.

## Layout

```
src/policy_assistant/
  config.py               settings (UPA_ prefix, .env)
  cli.py                  upa db | corpus | ingest | stats | retrieve | eval
  db/                     connection, migrations 0001/0002
  text/                   normalize, language, sentences, terms, stopwords
  ingest/                 manifest, fetch, extract, quality, tokens, chunker, pipeline
  retrieval/              access, bm25, dense, fusion, retriever, models, evaluation
corpus/                   manifest.yaml, sample manifest, fictional samples
evals/                    retrieval seed dataset
tests/                    text, ingest, retrieval, integration
docs/                     ROADMAP.md, ADRs
docker/initdb/            test-database bootstrap
```

Read [docs/ROADMAP.md](docs/ROADMAP.md) for all six phases. Architecture
decisions live in [docs/adr](docs/adr): single Postgres + pgvector, Urdu
normalization, the extraction quality gate, default-deny ACLs, BM25 in SQL, and
hybrid retrieval.

## Development

```bash
make check            # format check, lint, mypy --strict, unit tests
make test-integration # requires PostgreSQL; uses UPA_TEST_DATABASE_URL
make fmt              # apply formatting
```

Unit and integration tests use deterministic fakes and need no network. The
real-model test is opt-in: `UPA_MODEL_TESTS=1 uv run pytest -k real_model`.
Integration tests refuse a database whose name does not end in `_test`, and fail
instead of skipping when `UPA_REQUIRE_DB=1`.

Configuration is documented in [.env.example](.env.example). Pinned revisions:
tokenizer and embeddings `5617a9f61b028005a4858fdac845db406aefb181`, reranker
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

## License

MIT. See [LICENSE](LICENSE).
