.PHONY: setup db-up db-down migrate fmt fmt-check lint typecheck test test-integration check demo

setup:
	uv sync --all-groups

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

migrate:
	uv run upa db migrate

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

check: fmt-check lint typecheck test

# One-command local demo on the fictional sample corpus (plain http on 127.0.0.1).
demo: db-up
	uv run upa db migrate
	uv run upa ingest --manifest corpus/manifest.sample.yaml
	@if uv run upa user list | grep -q '^demo '; then \
		echo "User 'demo' already exists; log in with the password printed when it was created."; \
	else \
		PW=$$(uv run python -c 'import secrets; print(secrets.token_urlsafe(12))'); \
		printf '%s\n' "$$PW" | uv run upa user add demo --display-name 'Demo Reviewer' \
			--role reviewer --group all_staff --group finance --password-stdin; \
		echo "Log in as demo with password: $$PW"; \
	fi
	@echo 'Open http://127.0.0.1:8000 (set ANTHROPIC_API_KEY first to get answers).'
	UPA_SESSION_COOKIE_SECURE=false uv run upa serve
