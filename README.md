# Urdu–English Policy Assistant

A bilingual Urdu/English policy assistant whose answers must cite exact source
sentences, use only documents the asking user may see, and log every answer for
human review. Phase 1 builds the ingestion foundation: a PostgreSQL schema,
Urdu-aware text normalization, PDF/TXT extraction with a fail-closed quality
gate, heading-aware chunking, idempotent ingestion, a CLI, tests, and CI. Phase 2
adds hybrid retrieval; later phases add cited answers, access control and UI,
evaluation, and a showcase.

## Status

Phase 1 of 6 complete: ingestion foundation. No retrieval or answer-accuracy
numbers appear here yet, because none have been measured.

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
```

`upa corpus inspect --manifest corpus/manifest.sample.yaml` extracts and scores
the corpus without touching the database.

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

## Layout

```
src/policy_assistant/
  config.py               settings (UPA_ prefix, .env)
  cli.py                  upa db | corpus | ingest | stats
  db/                     connection, migrations, 0001_init.sql
  text/                   normalize (clean_text/fold), language, sentences
  ingest/                 manifest, fetch, extract, quality, tokens, chunker, pipeline
corpus/                   manifest.yaml, sample manifest, fictional samples
tests/                    text, ingest, integration
docs/                     ROADMAP.md, ADRs
docker/initdb/            test-database bootstrap
```

Read [docs/ROADMAP.md](docs/ROADMAP.md) for all six phases. Architecture
decisions live in [docs/adr](docs/adr): single Postgres + pgvector, Urdu
normalization, the extraction quality gate, and default-deny ACLs.

## Development

```bash
make check            # format check, lint, mypy --strict, unit tests
make test-integration # requires PostgreSQL; uses UPA_TEST_DATABASE_URL
make fmt              # apply formatting
```

Unit tests need no network and no database. Integration tests refuse a
database whose name does not end in `_test`, and fail instead of skipping when
`UPA_REQUIRE_DB=1`.

Configuration is documented in [.env.example](.env.example). The pinned bge-m3
tokenizer revision is `5617a9f61b028005a4858fdac845db406aefb181`.

## License

MIT. See [LICENSE](LICENSE).
