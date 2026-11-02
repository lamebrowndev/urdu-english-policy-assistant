# 0008 — Append-only audit log

## Status

Accepted for Phase 3.

## Context

Every answer must be reviewable after the fact: what the user asked, which
documents the caller was allowed to see, exactly which chunk texts and sentences
the model received, which model answered, and what it cost. Re-ingestion
changes chunk ids and document metadata, so audit rows must not depend on the
current corpus.

## Decision

Once `retrieve()` has returned, `answer()` writes exactly one `answer_audit`
row before it returns or raises. Failures before that point (database
unavailable, model loading, retrieval errors) raise with no row. Record the
actor, principal groups, question and its language, outcome, the exact
user-facing text, the model reply JSON (`segments`, `sources`, `raw_text`), the
full `RetrievalResult.as_dict()` snapshot, model/request/usage metadata, flags,
prompt version, pipeline fingerprint, and stage latencies.

The table has no foreign keys to `documents` or `chunks`. A PostgreSQL trigger
raises `answer_audit is append-only` on `UPDATE`, `DELETE`, and `TRUNCATE`.
`access_violation` and `internal_error` are recorded, then the original
exception is re-raised. `api_error` is recorded and returned as an error
outcome. If the audit insert fails, `answer()` raises and no answer is
returned: no audit, no answer.

No database transaction is held open while the model is called.

Human reviews are separate append-only rows in Phase 4, not updates to this
table.

## Consequences

- Re-ingesting or pruning the corpus never changes or removes audit history.
- The stored retrieval snapshot can contain document text; the table must be
  protected like the corpus itself.
- Reviewers can reconstruct the exact model context without replaying
  retrieval.
- Append-only storage will grow; retention/archival is a later operational
  concern.
