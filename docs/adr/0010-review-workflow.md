# 0010 — Review workflow

## Status

Accepted for Phase 4A.

## Context

Phase 5 needs human labels for answers, and reviewers must not learn about text
they are not allowed to read. Reviews also need to remain a durable, auditable
record rather than editable notes.

## Decision

Store reviews in append-only `answer_reviews` rows. A database trigger rejects
UPDATE, DELETE, and TRUNCATE. Verdicts are `correct`, `partially_correct`,
`incorrect`, `correct_abstention`, and `missed_answer`. Answered rows require a
correctness verdict and an explicit `citation_ok` boolean; non-answered rows
require an abstention verdict and a NULL `citation_ok`. Error rows cannot be
reviewed and return 409.

The reviewer queue is ACL-filtered against the current corpus: a reviewer sees
a row only if every document id in its retrieval snapshot is in the reviewer's
current allowed set. Admins see every row. A row that is missing and a row that
is hidden from the reviewer both return 404. Error rows are hidden from the
normal queue and appear only when the caller filters `outcome=error`.

Users see their own history with statuses `answered`, `abstained`, or `error`;
`sources` are returned only for answered rows. Reviews are the Phase 5 human
scoring tool.

## Consequences

- Reviews cannot be corrected in place; corrections are additional review rows.
- A reviewer losing access to a document also loses visibility of old rows that
  cite it.
- The queue uses current ACLs, not the ACL snapshot at answer time; that is
  deliberate for present-day confidentiality.
