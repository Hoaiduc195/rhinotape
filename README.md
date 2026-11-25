# llm-tape

Lightweight **record, replay, and assertion** library for testing LLM-powered Python apps in `pytest` — zero API cost in CI, automatic secret & PII scrubbing, **true sync & async SSE streaming**, local HTTP replay server, interactive HTML trace viewer, and built-in assertions for **token budgets, USD cost, cache hit rates, JSON output, and tool-calling sequences**.

Works transparently with any SDK built on `httpx` (`openai`, `anthropic`, `google-genai`, or raw `httpx.Client` / `httpx.AsyncClient`), optional zero-bloat adapters for `requests` (`pip install "llm-tape[requests]"`), `aiohttp` (`pip install "llm-tape[aiohttp]"`), opt-in stdlib `urllib.request` (`patch_urllib=True`), and **any language or workflow engine** (Node, Go, Rust, n8n, Dify, curl) via `llm-tape serve`.

## Features

- **Record & Replay (`once`, `replay`, `record`, `new_episodes`, `none`):** Record real LLM HTTP & SSE streaming calls into `.tape.json` or deterministic gzip-compressed `.tape.json.gz` cassettes (schema v2, atomic writes, `MAX_CASSETTE_BYTES` safety cap, `content_hash()`), or store them in-memory with `MemoryCassetteStore` / custom `CassetteStore` backends.
- **Local HTTP Replay Server (`TapeReplayServer` / `llm-tape serve`):** Serve recorded cassettes (including chunked SSE & latency pacing) over localhost with `/__tape__/health` and `/__tape__/stats` so any SDK or non-Python service can replay deterministically.
- **Interactive HTML Trace Viewer (`--tape-html-report` / `llm-tape export --format html`):** Self-contained, XSS-safe, zero-CDN HTML report with dark/light mode, KPI cards, latency waterfall bars, thinking/tool-call panels, and live search/provider/status filters.
- **Chaos & Fault-Injection Toolkit (`chaos.py`):** Composable `on_replay` hooks — `fail_nth()`, `fail_first()`, `inject_latency()`, `corrupt_json_body()`, `truncate_stream()`, `chain_replay_hooks()`, and `replace_response()`.
- **Automatic Secret & PII Scrubbing:** Redacts API keys, cookies, URL userinfo, `?key=` query params, wildcard headers (`extra_sensitive_headers=["x-secret-*"]`), URL path segments (`redact_url_patterns`), dot-path `redact_fields`, regex `redact_patterns`, and built-in PII presets (`redact_pii=True` or `["email", "phone", "ssn", "credit_card", "ipv4", "jwt"]`).
- **Composable Request Matchers & Semantic Diffs:** Use presets (`MATCH_STRICT`, `MATCH_RELAXED`, `MATCH_PATH_AND_BODY`), `ignore_fields`, `ignore_query_params`, `ignore_json_array_order`, `match_headers`, or build custom matchers with `match_all`, `match_any`, `match_method`, `match_path`, `match_json_subset`, `match_json_regex`, `match_header_equals`. On mismatch, `UnmatchedRequestError` provides a semantic JSON key-path summary, unified diff, and structured `.incoming`/`.closest`/`.diff_text` attributes.
- **Recording Hooks & Live Circuit Breakers:** Filter/transform interactions with `on_record`, attach custom `metadata`, flush via `session.save()`, and cap live API spend/calls in real time with `max_interactions` and `max_cost_usd`.
- **LLM-Aware Parsing, Cache-Write Billing & Cost Analytics:** Extracts provider, model, token usage (`input`, `output`, `cached_input_tokens`, `cache_creation_tokens`, `reasoning_tokens`), extended `thinking` blocks, tool calls, estimated USD cost (including Anthropic 1.25x cache-write pricing), and `estimate_cache_savings_usd()` across:
  - **OpenAI / Azure OpenAI / OpenRouter / DeepSeek / Groq / Together / Fireworks / Mistral (incl. FIM) / xAI Grok / Perplexity / Hugging Face** (Chat Completions, Responses API, Embeddings, Audio, Images — JSON & SSE)
  - **Anthropic** (Messages API + Extended Thinking + Prompt Caching — JSON & SSE)
  - **Google Gemini / Vertex AI** (`:generateContent` & `:streamGenerateContent` — JSON & SSE)
  - **Cohere** (`/v2/chat` & `/v1/chat` — JSON & SSE), **Ollama** (JSON & NDJSON), **AWS Bedrock** (Converse API)
- **Rich Assertions:** `assert_cost_under`, `assert_max_cost_per_call`, `assert_tokens_under`, `assert_max_tokens_per_call`, `assert_reasoning_tokens_under`, `assert_cached_tokens_at_least`, `assert_cache_hit_rate_at_least`, `assert_json_output`, `assert_text_contains`, `assert_text_matches`, `assert_thinking_present`, `assert_streamed`, `assert_not_streamed`, `assert_status_codes`, `assert_model`, `assert_provider`, `assert_tool_called`, `assert_no_tool_called`, `assert_tool_call_count`, `assert_tool_args`, `assert_tool_sequence`, `assert_call_count`, `assert_no_errors`, `assert_no_unused_interactions`, `assert_max_latency_ms`.
- **Pytest Plugin & CI/CD Integration:** Auto-discovered `llm_tape` & `llm_tape_dir` fixtures, env-var fallbacks (`LLM_TAPE_MODE`, `LLM_TAPE_DIR`, `LLM_TAPE_MAX_COST`, `LLM_TAPE_MAX_TOKENS`, `LLM_TAPE_REDACT_PII`), `--tape-update-stale <days>`, `--tape-report`, `--tape-json-report`, `--tape-html-report`, `pytest-xdist`-safe aggregation, and `.pre-commit-hooks.yaml`.

## Installation

```bash
pip install llm-tape
# Optional HTTP adapters:
pip install "llm-tape[requests]"
pip install "llm-tape[aiohttp]"
# or with uv:
uv add --dev llm-tape
```

`llm-tape` has only **one** hard runtime dependency (`httpx>=0.27.0`) and works even without `pytest` installed.

## Quickstart

### 1. Context Manager, Decorator & In-Memory Store

```python
import httpx
from llm_tape import MemoryCassetteStore, tape, use_tape

store = MemoryCassetteStore(compress=True)

with tape(
    "tests/tapes/weather.tape.json",
    mode="once",
    redact_pii=True,
    max_cost_usd=0.05,
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


@use_tape("tests/tapes/async_helper.tape.json", mode="once", store=store)
async def fetch_summary(text: str) -> str:
    ...
```

### 2. Pytest Fixture & CI Budgets

```python
import pytest
from llm_tape import TapeSession

@pytest.mark.tape(
    name="agent_search",
    redact_pii=["email", "ssn"],
    ignore_localhost=True,
    ignore_fields=["metadata.trace_id"],
)
def test_agent_workflow(llm_tape: TapeSession) -> None:
    result = run_my_agent("Find Python 3.13 release notes")

    llm_tape.assert_tool_sequence(["web_search", "summarize"])
    llm_tape.assert_cost_under(0.01)
    assert "3.13" in result
```

```bash
# Strict offline CI replay with terminal, JSON, and interactive HTML reports:
pytest --tape-mode=replay --tape-report --tape-json-report=tape.json \
       --tape-html-report=trace.html --tape-max-cost=0.50

# Auto-refresh cassettes older than 30 days:
pytest --tape-update-stale=30
```

### 3. CLI Power Tools

```bash
# Inspect, summarize (flat or grouped by dir), list, and search cassettes
llm-tape inspect tests/tapes/agent.tape.json --show-bodies --show-thinking
llm-tape summary tests/tapes --group-by dir --max-cost 0.50
llm-tape list tests/tapes --sort cost --limit 10
llm-tape search tests/tapes --tool web_search --status 200 --min-cost 0.001
llm-tape stats tests/tapes
llm-tape pricing --filter claude

# CI verification, secret & PII scanning, fingerprint integrity, and regression diffs
llm-tape verify tests/tapes --check-secrets --check-pii
llm-tape replay-check tests/tapes
llm-tape diff baseline/ candidate/ --max-cost-increase-pct 15 --fail-on-tool-change --show-text-diff

# Maintenance: redact PII, reparse pricing, merge/split, convert gzip, prune stale
llm-tape redact tests/tapes --pii email,ssn --dry-run
llm-tape reparse tests/tapes
llm-tape merge a.tape.json b.tape.json -o combined.tape.json --deduplicate
llm-tape split combined.tape.json -o split_dir/
llm-tape convert tests/tapes --to gzip --delete-source
llm-tape prune tests/tapes --tests-dir tests/ --older-than-days 30

# Export reports (Markdown, CSV, HAR 1.2, interactive HTML, JUnit XML) & serve locally
llm-tape export tests/tapes --format html -o trace.html
llm-tape export tests/tapes --format junit -o junit.xml
llm-tape serve tests/tapes --port 8787
```

### 4. Pre-Commit Hook

```yaml
repos:
  - repo: https://github.com/Hoaiduc195/llm-tape
    rev: v0.3.0
    hooks:
      - id: llm-tape-verify-pii
```

## Development

```bash
uv sync --extra dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv run pytest -v -W error
```

## License

MIT
