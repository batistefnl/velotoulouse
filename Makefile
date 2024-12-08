.PHONY: all data model figures rain app test lint

all: data model figures

data:
	uv run python -m velo.data

model:  # ~4 min
	uv run python -m velo.model

figures:
	uv run python -m velo.figures

rain:
	uv run python -m velo.causal

app:
	uv run streamlit run app/app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app
