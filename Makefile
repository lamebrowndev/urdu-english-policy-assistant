.PHONY: setup db-up db-down migrate fmt fmt-check lint typecheck test test-integration check

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
