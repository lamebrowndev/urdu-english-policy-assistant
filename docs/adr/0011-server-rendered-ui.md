# 0011 — Server-rendered UI

## Status

Accepted for Phase 4B.

## Context

The UI must be usable without JavaScript, render mixed Urdu/English text
correctly, and stay testable by an automated agent that cannot open a browser.
A client-side SPA would add a build step, an inline-script surface, and a
second authorization path that could drift from the JSON API.

## Decision

Render pages with Jinja2 on the server. The environment uses `autoescape=True`
and never applies `|safe` or `Markup` to stored, user or model text. The CSP is
strict (`default-src 'self'`), with no inline script, inline style, event-handler
attributes, CDNs, or external requests. `app.js` is a progressive enhancement:
it only disables the ask button and shows a "Thinking…" label.

Every form is a normal HTML form. POSTs require a same-origin `Origin` (or the
origin of `Referer`) and, for authenticated forms, a session CSRF token. Every
successful POST redirects with 303 (POST/redirect/GET).

Each stored text block gets `lang` and `dir` from script detection independent
of the English/Urdu UI cookie. The UI toggle sets `<html lang>` and `dir`; a
mixed thread can contain blocks in either direction. The Noto Nastaliq Urdu
font is self-hosted (OFL-1.1, pinned SHA-256), with `local()` first so an
installed copy is preferred. Citations are ordinary anchors; CSS `:target`
highlights the cited source without JavaScript.

Reviewers see the rerank score in the retrieved-context view, but never raw
BM25, dense-distance, or RRF scores.

### Referrer policy and the Origin check

Pages send `Referrer-Policy: same-origin`, never `no-referrer`. Under
`no-referrer`, browsers send `Origin: null` (and no `Referer`) on HTML form
submissions, per the Fetch spec rules for navigation requests. The same-origin
check must keep rejecting a null origin, so `no-referrer` would block every form
in a real browser. This was found in Chrome, not by `TestClient`, which sets
`Origin` by hand; a test now pins the header. `same-origin` still sends nothing
to other sites.

### Direction of answers

An answer is rendered as one paragraph whose `lang` and `dir` come from the
whole answer text. A citation segment gets its own `lang`/`dir` isolate only
when it contains letters in the other language; punctuation-only segments
(such as the Urdu full stop) inherit. Without this, an Urdu answer inside the
English UI laid its segments out left to right, so the sentence read backwards.
Document titles and section headings in metadata lines are wrapped in `<bdi>`.

## Consequences

- Every page is inspectable as HTML and testable with `TestClient`.
- Disabling JavaScript leaves every workflow intact.
- Fonts and assets are packaged in the wheel; there is no asset build.
- The `build_documents()` prompt contract is unaffected by presentation.
