# 0006 — Hybrid retrieval

## Status

Accepted for Phase 2.

## Context

Urdu and English policy questions mix lexical and semantic cues. BM25 is
strong on exact terms, numbers, and section headings; dense bge-m3 embeddings
handle paraphrase and cross-lingual questions. Neither alone was sufficient
for the planned evaluation.

## Decision

Use bge-m3 embeddings in pgvector with an HNSW cosine index and iterative
scans (`hnsw.iterative_scan = strict_order`, `hnsw.ef_search` configurable).
Fuse BM25 and dense rankings with reciprocal rank fusion, `k = 60`. Rerank the
top 30 fused passages with bge-reranker-v2-m3, capped at 512 tokens. Expose
four modes: `bm25`, `dense`, `hybrid`, and `hybrid_rerank` (default).

Chunks stay capped at 350 bge-m3 tokens, leaving room for the question and
the title/heading inside the reranker's 512-token window. bge-m3 uses no
query prefix.

## Consequences

- Both lexical and semantic recall are available without a second database.
- Every stage keeps the ACL predicate in its SQL; the reranker only receives
  passages that were already authorized.
- BM25-only mode remains available for debugging and ablations.
- Retrieval quality on the real corpus is unmeasured until Phase 5; the seed
  dataset is a pipeline check, not a performance claim.
