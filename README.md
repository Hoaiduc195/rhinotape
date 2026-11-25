# llm-tape

Lightweight **record, replay, and assertion** library for testing LLM-powered Python apps in `pytest` — zero API cost in CI, automatic secret scrubbing, **true sync & async SSE streaming**, and built-in assertions for **token budgets, USD cost, and tool-calling sequences**.

Works transparently with any SDK built on `httpx` (`openai`, `anthropic`, `google-genai`, or raw `httpx.Client` / `httpx.AsyncClient`), plus optional zero-dependency adapters for `requests` (`pip install "llm-tape[requests]"`) and `aiohttp` (`pip install "llm-tape[aiohttp]"`) without modifying your application code.

## Features

- **Record & Replay (`once`, `replay`, `record`, `none`):** Record real LLM HTTP & SSE streaming calls once into human-readable `.tape.json` or deterministic gzip-compressed `.tape.json.gz` files (schema v2 with atomic writes and bounded decompression safety), then replay offline deterministically in milliseconds.
- **True SSE Streaming (Sync & Async):** Incrementally decodes UTF-8 across TCP chunk boundaries, supports `gzip`/`deflate`/`br`/`zstd` streams, only commits completed streams (aborted streams are ignored), and replays chunk-by-chunk via `iter_lines()` / `aiter_text()`.
- **Concurrency & Nesting Safe:** Uses a `contextvars.ContextVar` + reference-counted `httpx`, optional `requests`, and optional `aiohttp` dispatchers — safe for nested `with tape(...)` blocks, multi-threaded workloads, concurrent `asyncio.gather(...)` tasks, and `pytest-xdist`.
- **Automatic Secret & PII Scrubbing:** Redacts `Authorization`, `x-api-key`, `x-goog-api-key`, `ocp-apim-subscription-key`, `x-amz-security-token`, cookies, URL basic-auth userinfo, `?key=` query parameters, plus body `redact_fields`, pre-compiled `redact_patterns` (regex on JSON & cross-chunk SSE streams), and custom `scrub_body` hooks.
- **Flexible Filtering, Matching & Latency Replay:** Ignore non-LLM traffic (`ignore_localhost`, `ignore_hosts`, `ignore_request`), ignore volatile JSON fields (`ignore_fields=["metadata.trace_id", "messages.*.id"]`) and query params (`ignore_query_params`), customize `match_on` or `custom_matcher`, simulate timing with `replay_latency_scale`, and get a clean unified diff on `UnmatchedRequestError`.
- **LLM-Aware Parsing, Thinking/Reasoning Tokens & Cached Pricing:** Automatically extracts provider, model, token usage (including OpenAI/Anthropic/Gemini/Bedrock cached input tokens and OpenAI/Gemini `reasoning_tokens`), extended `thinking` blocks (Anthropic, DeepSeek-R1, Gemini), estimated USD cost (including `o3`, `o4-mini`, `gpt-4.1`, `claude-4`, `gemini-2.5-*`, `deepseek-*`, `mistral-*`, overrideable via `custom_pricing` or `LLM_TAPE_PRICING_FILE`), response text, and tool calls for:
  - **OpenAI / Azure OpenAI / OpenRouter** (Chat Completions & Responses API, JSON & SSE)
  - **Anthropic** (Messages API, JSON & SSE)
  - **Google Gemini / Vertex AI** (`:generateContent` & `:streamGenerateContent`, JSON & SSE)
  - **Ollama** (`/api/chat`, `/api/generate`, JSON & NDJSON streaming)
  - **AWS Bedrock** (Converse API)
- **Built-in Assertions:**
  - `session.assert_cost_under(0.01)`
  - `session.assert_tokens_under(2000)`
  - `session.assert_cached_tokens_at_least(500)`
  - `session.assert_model("gpt-4o-mini")` / `session.assert_provider("openai")`
  - `session.assert_tool_called("search_docs", min_times=1)`
  - `session.assert_tool_args("search_docs", {"query": "pytest"})`
  - `session.assert_tool_sequence(["search_docs", "summarize"])`
  - `session.assert_call_count(2)` / `session.assert_no_errors()`
  - `session.assert_no_unused_interactions()`
  - `session.assert_max_latency_ms(2000.0)`
- **Pytest Plugin + CI-ready CLI:** Auto-discovered `llm_tape` fixture (`--tape-mode`, `--tape-dir`, `--tape-compress`, `--tape-report`, `--tape-json-report`, `--tape-max-cost`, `--tape-max-tokens`, `@pytest.mark.tape(...)`, `pyproject.toml` ini defaults, `pytest-xdist`-safe) and `llm-tape inspect | summary | verify | diff | redact | prune` CLI with `--json` and CI regression/secret gates.

## Installation

```bash
pip install llm-tape
# Optional HTTP adapters:
pip install "llm-tape[requests]"
pip install "llm-tape[aiohttp]"
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
# Inspect a single cassette (human-readable or JSON; supports .tape.json & .tape.json.gz)
llm-tape inspect tests/tapes/test_agent/agent_search.tape.json --json

# Summarize all cassettes with optional CI budget gates
llm-tape summary tests/tapes --max-cost 0.50 --max-tokens 200000

# Validate all cassettes & scan for leaked API keys/tokens in CI
llm-tape verify tests/tapes --check-secrets

# Compare two cassettes or two cassette directories with CI regression gates
llm-tape diff baseline_tapes/ candidate_tapes/ --max-cost-increase-pct 15 --fail-on-tool-change

# Re-scrub secrets/PII in existing cassettes (preview with --dry-run)
llm-tape redact tests/tapes --field user.email --pattern "sk-[A-Za-z0-9]+" --dry-run

# Find (dry-run) or remove (--apply) orphaned or stale cassettes
llm-tape prune tests/tapes --tests-dir tests/ --older-than-days 30

# Export cassettes to Markdown PR table, CSV cost report, or HAR 1.2 archive
llm-tape export tests/tapes --format markdown
llm-tape export tests/tapes --format har -o trace.har

# Bulk convert cassettes between plain .tape.json and gzip .tape.json.gz
llm-tape convert tests/tapes --to gzip --delete-source
```

Run pytest with terminal cost summary & JSON report for CI:

```bash
pytest --tape-mode=replay --tape-report --tape-json-report=tape-report.json --tape-max-cost=0.50
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
