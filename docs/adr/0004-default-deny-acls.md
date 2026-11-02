# 0004 — Default-deny ACLs

## Status

Accepted for Phase 1.

## Context

The assistant must use only documents the asking user may see. If permissions
live in a second system, retrieval can accidentally include text before the
filter runs, and the model may then receive unauthorized content.

## Decision

Every document lists at least one group in the corpus manifest, which is the
single source of truth for ACLs. Ingest refuses documents without an ACL
entry and syncs ACL rows on every run, including metadata-only runs. A
document with no ACL rows is visible to no one. From Phase 2 the ACL filter
runs inside both retrieval queries, never as a post-filter, so the LLM only
ever receives authorized text.

## Consequences

- Removing a document from every group is impossible by construction; the
  manifest must name an accessible group or the document is not ingested.
- ACL edits apply without re-chunking, and stale ACL rows are replaced rather
  than merged.
- Phase 2 must prove with a property test that no result falls outside the
  caller's ACL.
