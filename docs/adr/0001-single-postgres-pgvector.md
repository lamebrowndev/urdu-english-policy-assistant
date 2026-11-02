# 0001 — Single Postgres + pgvector store

## Status

Accepted for Phase 1.

## Context

The product needs documents, per-document ACLs, chunks, and — from Phase 2 —
dense vectors, a BM25 index, and append-only audit rows. Access filtering must
apply to every retrieval path. Splitting these across stores would force
cross-store joins and a distributed consistency story before the corpus is
large enough to need one.

## Decision

Use one PostgreSQL database with the pgvector extension for all of the above.
ACL rows live next to documents and chunks, so retrieval SQL can join them in
the same query and the same transaction. Phase 2 will implement BM25 in SQL
over an Urdu-folded inverted index; its own ADR will record the ranking and
index details.

## Consequences

- Authorization is a SQL predicate, not a post-filter in application code.
- A single backup, migration stream, and transaction boundary covers metadata,
  text, vectors, and audit rows.
- The design is comfortable up to roughly 10^5–10^6 chunks.
- Beyond that ceiling, the scale-up path is ParadeDB or OpenSearch; moving
  would require revisiting the ACL-in-query guarantee.
