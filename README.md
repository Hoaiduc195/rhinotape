# llm-tape

Lightweight **record, replay, and assertion** library for testing LLM-powered Python apps in `pytest` — zero API cost in CI, automatic secret scrubbing, **true sync & async SSE streaming**, and built-in assertions for **token budgets, USD cost, and tool-calling sequences**.

Works transparently with any SDK built on `httpx` (`openai`, `anthropic`, `google-genai`, or raw `httpx.Client` / `httpx.AsyncClient`) without modifying your application code.

## Features

- **Record & Replay (`once`, `replay`, `record`, `none`):** Record real LLM HTTP & SSE streaming calls once into human-readable `.tape.json` files (schema v2 with atomic writes), then replay offline deterministically in milliseconds.
- **True SSE Streaming (Sync & Async):** Incrementally decodes UTF-8 across TCP chunk boundaries, only commits completed streams (aborted streams are ignored), and replays chunk-by-chunk via `iter_lines()` / `aiter_text()`.
- **Concurrency & Nesting Safe:** Uses a `contextvars.ContextVar` + reference-counted `httpx` dispatcher — safe for nested `with tape(...)` blocks and concurrent `asyncio.gather(...)` tasks.
- **Automatic Secret Scrubbing:** Redacts `Authorization`, `x-api-key`, `x-goog-api-key`, `ocp-apim-subscription-key`, `x-amz-security-token`, cookies, URL basic-auth userinfo, and `?key=` query parameters before saving.
- **Flexible Filtering & Actionable Diffs:** Ignore non-LLM traffic (`ignore_localhost`, `ignore_hosts`, `ignore_request`), ignore volatile JSON fields (`ignore_fields=["metadata.trace_id", "messages.*.id"]`) and query params (`ignore_query_params`), and get a clean unified diff on `UnmatchedRequestError`.
- **LLM-Aware Parsing:** Automatically extracts provider, model, token usage (including Anthropic prompt cache tokens and Gemini thinking tokens), estimated USD cost, response text, and tool calls for:
  - **OpenAI** (Chat Completions & Responses API, JSON & SSE)
  - **Anthropic** (Messages API, JSON & SSE)
  - **Google Gemini** (`:generateContent` & `:streamGenerateContent`, JSON & SSE)
- **Built-in Assertions:**
  - `session.assert_cost_under(0.01)`
  - `session.assert_tokens_under(2000)`
  - `session.assert_tool_called("search_docs", min_times=1)`
  - `session.assert_tool_sequence(["search_docs", "summarize"])`
  - `session.assert_call_count(2)`
- **Pytest Plugin + CI-ready CLI:** Auto-discovered `llm_tape` fixture (`--tape-mode`, `--tape-dir`, `@pytest.mark.tape(...)`) and `llm-tape inspect | summary | verify | diff` CLI with `--json` output.

## Installation

```bash
pip install llm-tape
# or with uv:
uv add --dev llm-tape
```

`llm-tape` has only one runtime dependency (`httpx`) and works even without `pytest` installed.

## Quickstart

### 1. Context Manager & Decorator (Sync or Async)

```python
import httpx
from llm_tape import tape, use_tape

with tape(
    "tests/tapes/weather.tape.json",
    mode="once",
    ignore_localhost=True,
    ignore_fields=["metadata.trace_id"],
) as session:
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-secret"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    session.assert_call_count(1)
    session.assert_tokens_under(500)
    session.assert_cost_under(0.005)


@use_tape("tests/tapes/async_helper.tape.json", mode="once")
async def fetch_summary(text: str) -> str:
    ...
```

### 2. Pytest Fixture

```python
import pytest
from llm_tape import TapeSession

@pytest.mark.tape(
    name="agent_search",
    ignore_localhost=True,
    ignore_fields=["metadata.trace_id"],
)
def test_agent_workflow(llm_tape: TapeSession) -> None:
    result = run_my_agent("Find Python 3.13 release notes")

    llm_tape.assert_tool_sequence(["web_search", "summarize"])
    llm_tape.assert_cost_under(0.01)
    assert "3.13" in result
```

Run in CI with strict offline replay (fails with a unified diff if a prompt changed or a cassette is missing):

```bash
pytest --tape-mode=replay
```

Re-record cassettes after intentionally changing prompts:

```bash
pytest --tape-mode=record
```

### 3. CLI Commands

```bash
# Inspect a single cassette (human-readable or JSON)
llm-tape inspect tests/tapes/test_agent/agent_search.tape.json
llm-tape inspect tests/tapes/test_agent/agent_search.tape.json --json

# Summarize all cassettes in a directory
llm-tape summary tests/tapes --json

# Validate all cassettes in CI (fails on corrupt JSON or unsupported schema)
llm-tape verify tests/tapes

# Compare calls, tokens, cost, and tool sequences between two cassettes
llm-tape diff baseline.tape.json candidate.tape.json
```

## Development

```bash
uv sync --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run coverage run --source=llm_tape -m pytest -v
uv run coverage report -m
```

## License

MIT
