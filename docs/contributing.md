# Contributing

Development uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev --group notebooks
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python scripts/execute_notebooks.py   # when the library or a notebook changed
```

To build this documentation:

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going docs docs/_build/html
```

The conventions contributors (human or AI) are expected to follow, from evaluation counting to float32 tolerances and notebook outputs, are in [AGENTS.md](https://github.com/cai4cai/torch-piazzi/blob/main/AGENTS.md). Planned work is tracked in the [GitHub issues](https://github.com/cai4cai/torch-piazzi/issues).
