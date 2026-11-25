# Contributing to `llm-tape`

Thank you for your interest in contributing to `llm-tape`!

## Development Setup

`llm-tape` uses [`uv`](https://docs.astral.sh/uv/) for dependency management and targets Python `>=3.10`.

```bash
git clone https://github.com/Hoaiduc195/llm-tape.git
cd llm-tape
uv sync --extra dev
```

## Running Checks Locally

All pull requests must pass formatting, linting, strict typing, and tests with zero warnings:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest -v -W error
```

## Design Principles

- **Minimal core runtime dependencies**: `httpx>=0.27.0` is the sole hard runtime dependency. Optional integrations (`pytest`, `requests`, `aiohttp`) stay in optional extras.
- **Deterministic & offline-first**: All unit and integration tests must run completely offline without real API keys.
- **Zero secret leaks**: Never commit real API keys, tokens, or PII in cassettes. Run `uv run llm-tape verify . --check-secrets --check-pii` before opening a PR.
- **Strict typing**: Public APIs must be fully annotated and pass `mypy --strict` (`py.typed` is shipped).
