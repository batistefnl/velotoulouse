.PHONY: data test lint

data:
	uv run python -m velo.data

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app
