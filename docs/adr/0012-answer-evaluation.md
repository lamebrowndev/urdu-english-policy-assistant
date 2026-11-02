# 0012 — Answer evaluation

## Status

Accepted for Phase 5.

## Context

The project's claim is human-scored answer accuracy on a balanced Urdu/English
test set. That number is only meaningful if the questions cannot drift after
the system is tuned, if the system is exercised exactly as users exercise it,
if permission behaviour is scored (abstaining on restricted topics is a correct
answer), and if the uncertainty of a 200-question sample is reported.

## Decision

- Datasets are JSONL with three categories: `answerable`, `unanswerable`, and
  `restricted`. Evidence quotes are verified against the index, and categories
  are verified against the current ACLs, before any paid call.
- The test split is frozen by a SHA-256 record written once next to the file.
  A test run refuses an unfrozen or changed file and cannot be limited.
- Runs go through the production `answer()` path as each item's principal, under
  the actor `eval:<run_id>`. Every answer is therefore in the append-only audit
  log with its full context. The run row records the dataset hash and every
  setting that can change an answer.
- Scoring uses the existing review workflow: reviewers see the evaluation
  reference on the review page, and the latest append-only review per answer is
  the verdict. `correct` and `correct_abstention` count as right;
  `partially_correct` and system errors count as wrong.
- Accuracy is reported with a Wilson 95% interval, overall and by language,
  category, and tag; reports stay `PRELIMINARY` until every item is scored.
- `eval_runs` and `eval_results` are append-only, like the audit log.
- No LLM judge scores the test set. A judge could speed up dev iterations later,
  but only after calibration against human labels.

## Consequences

- The same review screen serves production review and evaluation scoring.
- Results are reproducible from the database: the run's settings, the audited
  context of every answer, and every review are retained.
- A full test run costs 200 Claude calls plus human review time; the dev split
  is where iteration happens.
