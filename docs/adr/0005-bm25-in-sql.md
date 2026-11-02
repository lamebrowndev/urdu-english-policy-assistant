# 0005 — BM25 in SQL

## Status

Accepted for Phase 2.

## Context

Retrieval must apply the caller's ACL before ranking, not after. Keeping the
BM25 index next to `chunks` and `document_acl` lets one SQL statement filter
and score in the same transaction. An external search service would introduce
a second store, a synchronization path, and a place for unauthorized text to
leak into a candidate set.

## Decision

Implement Okapi BM25 over `chunk_terms` in PostgreSQL. The IDF is the Lucene
form `ln(1 + (N - df + 0.5) / (df + 0.5))`. Corpus statistics (`N`, `avgdl`,
`df`) are global over all chunks, while the ACL predicate filters candidate
chunks inside the same query. The SQL scorer is tested against a plain-Python
reference to 1e-6 and identical order.

## Consequences

- ACL filtering and ranking share one query and one transactional boundary.
- The implementation is small enough to audit and compare against a reference.
- Global statistics create a side channel: scores depend on documents the
  caller cannot see. Raw BM25 scores are therefore shown only to admins;
  ordinary answers expose citations, not scores.
- ParadeDB or OpenSearch remains the scale-up path if SQL BM25 or the single
  Postgres instance becomes the bottleneck.
