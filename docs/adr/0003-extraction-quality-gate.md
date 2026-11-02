# 0003 — Extraction quality gate, fail closed

## Status

Accepted for Phase 1. Thresholds are provisional.

## Context

Urdu PDFs frequently have broken text layers: private-use glyphs instead of
letters, Latin-1 mojibake, reversed text, and image-only pages. Indexing that
text would produce confident but meaningless retrieval and citations. OCR is
out of scope for this phase.

## Decision

Assess every page after `clean_text()` and record its metrics and flags:
`empty`, `pua`, `wrong_script`, `low_lexicon`, `reversed_text`. A page is
included only when it has no flags. A document is quarantined for `no_text`,
`likely_scanned`, or `bad_text_layer`; otherwise it is accepted and flagged
pages are excluded individually. Quarantined documents are never written to
the database, and an ingest with any quarantine exits 2.

## Consequences

- Fail closed: questionable text cannot become searchable or citable.
- The inspect CLI and JSON report expose per-page metrics so a human can
  calibrate thresholds on real PDFs.
- Thresholds will change after calibration; they are part of the pipeline
  fingerprint, so a change triggers re-ingestion.
- Scanned documents need OCR before they can be ingested.
