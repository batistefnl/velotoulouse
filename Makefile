.PHONY: all data model figures test lint

all: data model figures

data:
	uv run python -m velo.data

model:  # ~4 min
	uv run python -m velo.model

figures:
	uv run python -m velo.figures

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app
