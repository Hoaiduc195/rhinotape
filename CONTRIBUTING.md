# Contributing to `rhinotape`

Thank you for your interest in contributing to `rhinotape`! We welcome bug fixes, new provider parsers, framework integration guides, CLI improvements, and tests.

Looking for a place to start? Check out our open [**good first issues**](https://github.com/Hoaiduc195/llm-tape/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [**help wanted**](https://github.com/Hoaiduc195/llm-tape/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) issues.

---

## Architecture Overview

| Module | Responsibility |
| :--- | :--- |
| `src/rhinotape/models.py` | Core immutable data models (`Interaction`, `RecordedRequest`, `RecordedResponse`, `Usage`, `ToolCall`, `LLMCallSummary`, `TapeMode`, exceptions) |
| `src/rhinotape/cassette.py` | Schema v2 `.tape.json(.gz)` serialization, atomic writes, bounded decompression, validation, merge/filter/slice |
| `src/rhinotape/scrubbing.py` | Secret/header/URL/PII redaction, field stripping, request fingerprinting, semantic JSON diffs |
| `src/rhinotape/parsers.py` | Provider-aware JSON, SSE, and NDJSON parsers + stream chunk redaction |
| `src/rhinotape/pricing.py` | Built-in USD per-1M-token pricing table, cache read/write billing, `RHINOTAPE_PRICING_FILE` loader |
| `src/rhinotape/session.py` | `TapeSession` lifecycle, `contextvars` dispatcher, streaming wrappers, guardrails, and assertions |
| `src/rhinotape/transport.py` | Explicit `httpx` `TapeTransport` / `AsyncTapeTransport` wrappers |
| `src/rhinotape/*_adapter.py` | Optional `requests`, `aiohttp`, and stdlib `urllib.request` adapters |
| `src/rhinotape/integrations.py` | Zero-dependency ASGI/WSGI middleware, `TapeConfig`, `unittest` mixin, SDK factories, generic HTTP bridge |
| `src/rhinotape/server.py` | Local reverse-proxy record/replay server & `/__tape__/*` polyglot HTTP control plane |
| `src/rhinotape/matchers.py` | Composable request matchers (`match_all`, `match_json_subset`, etc.) |
| `src/rhinotape/chaos.py` | Composable fault-injection `on_replay` hooks (`fail_nth`, `truncate_stream`, etc.) |
| `src/rhinotape/stores.py` | `CassetteStore` protocol, `MemoryCassetteStore`, `FileCassetteStore` |
| `src/rhinotape/html_report.py` | Self-contained, XSS-safe, zero-CDN HTML trace viewer |
| `src/rhinotape/pytest_plugin.py` | Auto-discovered `pytest11` plugin, fixtures, budgets, and xdist aggregation |
| `src/rhinotape/cli.py` | `rhinotape` CLI subcommands (`inspect`, `verify`, `diff`, `serve`, `run`, `export`, etc.) |

---

## Development Setup

`rhinotape` uses [`uv`](https://docs.astral.sh/uv/) for dependency management and targets Python `>=3.10`.

```bash
git clone https://github.com/Hoaiduc195/llm-tape.git
cd llm-tape
uv sync --extra dev
```

## Running Checks Locally

Every pull request must pass linting, formatting, strict typing, zero-warning tests, coverage (`>=93%`), and secret/PII verification:

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv run coverage run --source=rhinotape -m pytest -v -W error
uv run coverage report -m --fail-under=93
uv run rhinotape verify . --check-secrets --check-pii
```

---

## Design Principles

- **Minimal runtime dependencies:** `httpx>=0.27.0` is the sole hard runtime dependency. Optional integrations (`pytest`, `requests`, `aiohttp`) must remain optional extras and never break `import rhinotape` when absent.
- **Deterministic & offline-first:** All tests must run completely offline (`httpx.MockTransport` or local loopback sockets) with zero real API keys or external network calls.
- **Zero secret leaks:** Never commit real API keys, tokens, or PII in cassettes or tests.
- **Strict typing:** All public APIs must be fully annotated and pass `mypy --strict` (`py.typed` is shipped).
- **Lean & backwards-compatible:** Prefer small, focused changes that preserve existing public APIs and cassette schema v2 compatibility.

## Adding a New LLM Provider

1. Add provider detection and JSON/SSE parsing in `src/rhinotape/parsers.py`.
2. Add reference model rates (if public) in `src/rhinotape/pricing.py`.
3. Add offline E2E record/replay/streaming tests in `tests/`.
4. Update the Supported Providers table in `README.md`.
