# Security model

## Assets

- Policy documents and their document-level ACLs; some documents are restricted
  to specific groups.
- Users' questions, which can be sensitive.
- The audit log, which stores each question with the exact policy text that was
  retrieved for it, plus reviews and evaluation runs.
- Session tokens, password hashes, and the Claude API key.

## Trust boundaries

- **Browser to server.** Authenticated with a server-side session cookie. Every
  unsafe request needs the session's CSRF token; HTML forms also pass a
  same-origin check.
- **Server to PostgreSQL.** Trusted. The `upa` CLI holds database credentials and
  is a trusted operator tool: `upa ask --group` deliberately acts as any group.
- **Server to the Claude API.** Questions and authorized policy text leave the
  machine over TLS. Model output is treated as untrusted input.
- **Documents.** Extracted text is untrusted: it may contain instructions aimed
  at the model, and it is never rendered as HTML.

## Threats and mitigations

| Threat | Mitigation | Tested in |
|---|---|---|
| A user reads a restricted policy through an answer | Default-deny ACL filter inside both the BM25 and the vector SQL; a second authorization check before the model call; the model only receives authorized chunks | `test_retrieval.py` (randomized ACL cases), `test_answering.py` |
| A refusal reveals that a restricted document exists | Restricted and missing topics produce byte-identical replies; users only see `answered`, `abstained`, or `error` | `test_answering.py`, `test_api.py` |
| A reviewer reads restricted text through the audit log | Review queue, review items, and history are filtered by the viewer's current ACL; the filter fails closed on malformed snapshots; hidden and missing rows are both 404 | `test_api.py` |
| A client sends its own groups or role | Request models reject unknown fields; the principal is loaded from the database on every request | `test_api.py` |
| Revoked access keeps working | Groups, role, and active status are re-read on every request; disabling a user deletes their sessions | `test_api.py` |
| A database leak exposes live sessions | Only SHA-256 hashes of session tokens are stored | `auth/sessions.py` |
| Password guessing and user enumeration | Argon2id hashes; identical failures with equal hashing cost for unknown users | `test_api.py` |
| Cross-site request forgery | SameSite=Strict cookies, per-session CSRF tokens, same-origin check on forms, `Referrer-Policy: same-origin` so browsers keep sending the real origin | `test_api.py`, `test_web.py` |
| Cross-site scripting through questions, answers, or documents | Jinja autoescaping everywhere with no `\|safe`; strict CSP with no inline script, inline style, or event handlers | `test_web.py` |
| Open redirect | The language switch only redirects to local paths | `test_web.py` |
| Prompt injection inside a document | The system prompt treats documents as data; every claim must be cited; uncited model text is never shown; the model has no tools and no write access | `answering/prompts.py`, `test_answering.py` |
| Rewriting history | Audit, review, and evaluation tables reject UPDATE, DELETE, and TRUNCATE with database triggers; re-ingestion cannot touch them | `test_migrations.py`, `test_answering.py` |
| Silent failures | Every failure after retrieval is written to the audit log before it propagates | `test_answering.py` |
| Supply-chain drift | `uv.lock` pins every package; model revisions and the bundled font are pinned by commit and SHA-256 | `pyproject.toml`, `config.py` |

## Known gaps (prototype scope)

- No login rate limiting or lockout, no password reset, no SSO or MFA.
- No TLS termination: run behind an HTTPS reverse proxy and keep Secure cookies
  on. `UPA_SESSION_COOKIE_SECURE=false` exists only for plain-http local use.
- One process; paid model calls have a global concurrency cap but no per-user
  quota.
- No request-body size limit in the application; set one at the proxy.
- The audit log holds policy text and questions; protect the database like the
  corpus itself.
- The database passwords in `docker-compose.yml` and the CI workflow are local
  defaults, not secrets.
