# 0002 — Urdu text normalization

## Status

Accepted for Phase 1.

## Context

Urdu corpus text mixes Arabic and Urdu code points, uses optional diacritics,
contains three digit systems, and carries zero-width and bidi controls. Exact
matching on raw text would miss obvious variants, while aggressive
normalization would corrupt what readers and auditors see.

## Decision

Keep two text forms and never mix them:

- `clean_text()` produces stored and displayed text: NFKC, normalized line
  endings, control removal (except newline), targeted zero-width/bidi removal
  that preserves ZWNJ, whitespace collapse, and diacritic preservation.
- `fold()` produces a lossy lookup key: casefold, Arabic-to-Urdu letter
  mapping, hamza handling, digit conversion, diacritic/mark and tatweel
  removal, and whitespace collapse.

Only `clean_text()` output is written to `pages.text`, `chunks.text`, and
`chunks.sentences`. `fold()` is used for tokens, lexicon checks, and later
retrieval indexes.

## Consequences

- Display text keeps the source's diacritics, ZWNJ, and deliberate Arabic code
  points; the quality tests and spot checks rely on this.
- `fold()` is lossy and must never be stored as display text.
- Changing the normalization changes the pipeline fingerprint and forces
  re-ingestion, so the behavior is versioned with the pipeline.
