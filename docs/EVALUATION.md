# Evaluation protocol

The headline metric is **human-scored answer accuracy on a frozen test set of
200 Urdu and English policy questions**. This document is the protocol. A
number appears in the README only after a complete run on the frozen test set,
with its sample size and 95% confidence interval.

## Datasets

| File | Size | Use |
|---|---|---|
| `evals/answers.sample.jsonl` | 12 | Pipeline check over the fictional sample corpus. Not a result. |
| `evals/answers.dev.jsonl` | ~100 | Iteration: prompts, thresholds, retrieval settings. Run as often as needed. |
| `evals/answers.test.jsonl` | 200 (100 Urdu, 100 English) | Frozen before its first run, never edited afterwards. |

Each line is one question:

```json
{"id": "ur-annual-days", "question": "ملازمین کو سال میں کتنے دن کی سالانہ چھٹی ملتی ہے؟",
 "language": "ur", "groups": ["all_staff"], "category": "answerable", "tags": ["lookup"],
 "reference_answer": "…", "evidence": [{"document_id": "leave-policy-ur", "quote": "…"}]}
```

| Field | Meaning |
|---|---|
| `id` | Stable id, `[A-Za-z0-9._-]{1,80}`. |
| `question` | Asked exactly as written. Its script must match `language`. |
| `groups` | The asker's groups. The question is asked as this principal. |
| `category` | `answerable` (a document the asker may read answers it), `unanswerable` (no document answers it), or `restricted` (only documents the asker may not read answer it). |
| `tags` | Optional question types for slicing: `lookup`, `synthesis`, `cross-lingual` (question and source in different languages), or your own. |
| `reference_answer` | What a correct answer says, or why the system should abstain. Shown to reviewers. |
| `evidence` | Quotes that answer the question. Required for `answerable` and `restricted`, forbidden for `unanswerable`. |

Before any paid call, `upa eval answers` checks every quote against the index
and every category against the current ACLs: answerable evidence must be
readable by `groups`, restricted evidence must not be.

Suggested composition per language: about 70% answerable, 15% unanswerable,
15% restricted, with lookup, synthesis, and cross-lingual questions in the
answerable share. Questions should be written by people who did not tune the
system, against the real corpus.

## Running

```bash
uv run upa eval freeze --dataset evals/answers.test.jsonl   # once, before the first test run
uv run upa eval answers --dataset evals/answers.test.jsonl --split test
```

The run asks every question through the production `answer()` path as the
item's principal and audits each answer under the actor `eval:<run_id>`. It
records the dataset hash and every setting that can change an answer: model,
effort, retrieval mode, thresholds, prompt version, and pipeline fingerprint.
`--split test` refuses an unfrozen or changed file and does not allow
`--limit`. Every question is one paid Claude call, so the command asks for
confirmation unless `--yes` is given.

## Human review

Open `/review?status=all&actor=eval:<run_id>` as a reviewer whose groups can
read the documents involved, or as an admin. Each item shows its evaluation
reference next to the question, the answer the user saw, the cited sentences,
and the retrieved context.

| Shown to the user | Verdict | Use when |
|---|---|---|
| An answer | `correct` | Every claim is supported by the cited sentences and matches the reference, with no missing limit, condition, or exception that changes the meaning. |
| An answer | `partially_correct` | Right direction but incomplete or imprecise. Counts as wrong. |
| An answer | `incorrect` | Any wrong or unsupported claim, or any answer where the system should have abstained. |
| An abstention | `correct_abstention` | The asker's documents do not answer the question (unanswerable or restricted). |
| An abstention | `missed_answer` | The asker's documents answer the question. |

For every answer, also record whether the citations are correct: do the cited
sentences support the claims they are attached to?

System errors (API failures, timeouts) cannot be reviewed and count as wrong.
Reviews are append-only: to change a verdict, add a new review; the latest one
counts. Two independent reviewers for the test set are recommended, with
disagreements settled by discussion and recorded as a new review.

## Scoring

```bash
uv run upa eval score <run_id>
uv run upa eval score <run_id> --json > evals/results/<run_id>.json
```

Accuracy is `(correct + correct_abstention) / scored`, with a Wilson 95%
interval, overall and by language, category, and tag. The report says
`PRELIMINARY` until every item has a verdict. Commit the JSON of the run you
report.

## Reporting rules

- Report only a complete run on the frozen test set: accuracy, n, the 95%
  interval, and the Urdu and English numbers separately.
- Never present a target as a result, and never pick the best of several test
  runs. Label dev-set numbers as dev.
- With n = 200, a measured 90% has an interval of about 85–93%. Say so.
