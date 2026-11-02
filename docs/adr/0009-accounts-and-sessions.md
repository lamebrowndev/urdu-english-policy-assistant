# 0009 — Accounts and sessions

## Status

Accepted for Phase 4A.

## Context

The HTTP API must know who is asking and which document groups they may see.
The browser will hold a session, and every unsafe request needs CSRF
protection. The operator also needs a bootstrap path before any web UI exists.

## Decision

Store users, group membership, and sessions in PostgreSQL. Passwords use
Argon2id via `argon2.PasswordHasher`, with 12–1024 character passwords.
Sessions are opaque random tokens; only their SHA-256 hashes are stored.
Cookies are HttpOnly, Secure by default, SameSite=Strict, Path=/, with an
8-hour default TTL. Every unsafe request must also present the session's CSRF
token in `X-CSRF-Token`; the token comes from the login response or `/api/me`.

The API loads the user and their groups from the database on every request, so
group, role, and active-state changes apply to live sessions immediately.
Login failures for unknown, inactive, and wrong-password users return the same
401 body. Unknown and inactive users still pay an Argon2 verification against
a lazy dummy hash. `upa user` is the trusted operator tool for bootstrapping
accounts; the API never accepts groups, roles, or user ids from clients.

## Consequences

- Deleting a session row, expiring it, or disabling the user invalidates a
  cookie on the next request.
- Group changes are immediate, at the cost of one session/group query per
  request.
- Known limitations: no rate limiting, account lockout, password reset, or
  self-registration. A production deployment adds these at the proxy or
  identity provider.
