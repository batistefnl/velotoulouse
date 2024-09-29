.PHONY: all data figures test lint

all: data figures

data:
	uv run python -m velo.data

figures:
	uv run python -m velo.figures

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app
