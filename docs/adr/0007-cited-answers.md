# 0007 — Cited answers

## Status

Accepted for Phase 3.

## Context

Answers must be auditable sentence by sentence, and a confident but uncited
answer is worse than no answer. The model must also never see text outside the
caller's ACL. Urdu and English sentence boundaries differ, so citation anchors
cannot rely on character offsets from one language model's tokenizer.

## Decision

Send one custom-content document per retrieved chunk, with each cleaned chunk
sentence as a separate content block. Document index `i` is chunk `i`; block
index `j` is `chunk.sentences[j]`. This gives sentence-exact
`content_block_location` citations independent of language.

Apply two abstention layers before showing anything:

1. Retrieval gate: no chunks, or a top rerank score below the configured
   threshold, produces `abstained_retrieval` without calling the model.
2. Model gate: `NO_ANSWER`, no valid citation, refusal, or an API error
   produces an abstained/refused/error outcome.

Every abstention and refusal uses the same localized message, so a missing
topic and a restricted topic are indistinguishable. An answered response is
shown only when at least one citation validates. Refusals fall back to the
abstention message.

Structured outputs are not used because `output_config.format` cannot be
combined with citations. Sampling parameters (`temperature`, `top_p`, `top_k`)
are never sent: they return HTTP 400 for this model. `output_config.effort` is
the cost/quality lever, and thinking stays at its model default.

## Consequences

- Users see only cited answers; reviewers see the raw model text in the audit
  row when it was not citable.
- Citation validity is checked against the exact chunk and sentence range we
  sent, not against free text.
- A prompt change requires bumping `PROMPT_VERSION` and re-reviewing the
  evaluation set.
