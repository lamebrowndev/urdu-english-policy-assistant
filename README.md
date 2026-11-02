# Urdu–English Policy Assistant

A bilingual Urdu/English policy assistant whose answers must cite exact source
sentences, use only documents the asking user may see, and log every answer for
human review. Phases 1–3 built ingestion, hybrid retrieval, cited answers, and
an append-only audit log. Phase 4A adds accounts, server-side sessions, an
authenticated HTTP API, and a reviewer workflow. Phase 4B adds the web UI.

## Status

Phases 1–3 and 4A complete; next: 4B web UI. No answer-accuracy numbers appear
here yet, because none have been measured on the real corpus. The seed
retrieval dataset is a pipeline check, not a performance result.

The sample corpus is fictional. The project does not redistribute third-party
documents: real manifests store a source URL and a pinned SHA-256, and fetched
files stay under `data/` (git-ignored).

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/), and Claude API credentials.

```bash
make setup
make db-up
uv run upa db migrate
uv run upa ingest --manifest corpus/manifest.sample.yaml
uv run upa stats

export ANTHROPIC_API_KEY=...        # or use: ant auth login
uv run upa user add alice --display-name Alice --role user --group all_staff --password-stdin
uv run upa serve --host 127.0.0.1 --port 8000
```

The first `ingest` or API ask downloads bge-m3 and bge-reranker-v2-m3: about
4.6 GB in total. Model files stay in the Hugging Face cache. Once cached,
`export HF_HUB_OFFLINE=1` keeps loads fully local. The Claude API key is read by
the Anthropic SDK from `ANTHROPIC_API_KEY` or an `ant auth login` profile; it is
never written to this repository, logs, or audit rows.

`upa corpus inspect --manifest corpus/manifest.sample.yaml` extracts and scores
the corpus without touching the database. `upa eval retrieval --dataset
evals/retrieval_seed.sample.jsonl` runs the four retrieval modes over the seed
questions.

## HTTP API session

With the server running, a local plain-http session looks like this. The cookie
is Secure by default, so disable that only for local curl:

```bash
UPA_SESSION_COOKIE_SECURE=false uv run upa serve --port 8000

curl -c "${TMPDIR:-/tmp}/upa-cookies.txt" -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"your-password"}'

# Copy csrf_token from the login JSON, then:
curl -b "${TMPDIR:-/tmp}/upa-cookies.txt" -s -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -H "X-CSRF-Token: $CSRF" \
  -d '{"question":"How many days of annual leave do employees get?"}'
```

Unauthenticated responses are 401. Unsafe methods without the CSRF header are
403. The client never supplies groups, roles, or user ids; the server reads them
from the session on every request.

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
2. Shared `bm25_terms()` (Urdu folding, stopword removal, English Snowball
   stemming) for both indexing and queries.
3. bge-m3 embeddings and BM25 term counts written at ingest.
4. `bm25`, `dense`, `hybrid`, and `hybrid_rerank` modes; ACL filtering inside
   the BM25 and dense SQL.
5. `upa eval retrieval` for hit@k, recall@k, MRR@k, leaks, and latency.

## What Phase 3 adds

1. Migration `0003_answer_audit.sql`: an append-only audit table with
   database-enforced `UPDATE`/`DELETE`/`TRUNCATE` rejection.
2. Claude answers over one custom-content document per chunk, with each chunk
   sentence as a citation block.
3. A retrieval gate plus a model gate: missing evidence, `NO_ANSWER`, an
   uncited answer, a refusal, or an API error all become a recorded non-answer.
4. Identical user-facing abstention text for missing and restricted topics.
5. `upa ask`, `upa audit list`, and `upa audit show`.

## What Phase 4A adds

1. Migration `0004_accounts_reviews.sql`: users, user groups, server-side
   sessions, and append-only answer reviews.
2. Argon2id passwords, SHA-256-hashed session tokens, HttpOnly/Secure/
   SameSite=Strict cookies, and CSRF tokens for unsafe methods.
3. An authenticated FastAPI with login, ask, history, reviewer queue, item, and
   review endpoints; group and role changes apply to live sessions immediately.
4. `upa user` for bootstrap and admin operations, and `upa serve` for uvicorn.
5. ACL-filtered review queues where hidden and missing rows are both 404, and
   error rows are available only with `outcome=error`.

## Layout

```
src/policy_assistant/
  config.py               settings (UPA_ prefix, .env)
  cli.py                  upa db | corpus | ingest | stats | retrieve | eval | ask | audit | user | serve
  db/                     connection, migrations 0001–0004
  text/                   normalize, language, sentences, terms, stopwords
  ingest/                 manifest, fetch, extract, quality, tokens, chunker, pipeline
  retrieval/              access, bm25, dense, fusion, retriever, models, readiness, evaluation
  answering/              model client, prompts, documents, citations, service, audit store
  auth/                   passwords, users, sessions
  reviews/                review queue, review writes, history
  api/                    FastAPI app and dependencies
corpus/                   manifest.yaml, sample manifest, fictional samples
evals/                    retrieval seed dataset
tests/                    text, ingest, retrieval, answering, integration
docs/                     ROADMAP.md, ADRs
docker/initdb/            test-database bootstrap
```

Read [docs/ROADMAP.md](docs/ROADMAP.md) for all phases. Architecture decisions
live in [docs/adr](docs/adr): single Postgres + pgvector, Urdu normalization,
the extraction quality gate, default-deny ACLs, BM25 in SQL, hybrid retrieval,
cited answers, the append-only audit log, accounts and sessions, and the review
workflow.

## Development

```bash
make check            # format check, lint, mypy --strict, unit tests
make test-integration # requires PostgreSQL; uses UPA_TEST_DATABASE_URL
make fmt              # apply formatting
```

Unit and integration tests use deterministic fakes and need no network or
Claude API. Opt-in tests:

```bash
UPA_MODEL_TESTS=1 uv run pytest -k real_model
UPA_LLM_TESTS=1 uv run pytest -k live_llm   # needs real Claude credentials
```

Integration tests refuse a database whose name does not end in `_test`, and fail
instead of skipping when `UPA_REQUIRE_DB=1`.

Configuration is documented in [.env.example](.env.example). Pinned revisions:
tokenizer and embeddings `9a0624b896d81da7492a910ffa53731274b6cf3d`, reranker
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. The default model is
`claude-opus-5-5`.

## License

MIT. See [LICENSE](LICENSE).
