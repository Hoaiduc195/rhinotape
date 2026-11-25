# llm-tape

Lightweight **record, replay, and assertion** library for testing LLM-powered Python apps in `pytest` — zero API cost in CI, automatic secret scrubbing, and built-in assertions for **token budgets, USD cost, and tool-calling sequences**.

Works transparently with any SDK built on `httpx` (`openai`, `anthropic`, `google-genai`, or raw `httpx.Client` / `httpx.AsyncClient`).

## Features

- **Record & Replay (`once`, `replay`, `record`, `none`):** Record real LLM HTTP calls once into clean `.tape.json` files, then replay offline deterministically in milliseconds.
- **Sync & Async:** Supports both `httpx.Client` and `httpx.AsyncClient`, either via automatic context manager patching or explicit `TapeTransport` / `AsyncTapeTransport`.
- **Automatic Secret Scrubbing:** Redacts `Authorization`, `x-api-key`, `x-goog-api-key`, cookies, and `?key=` query parameters before writing cassettes.
- **LLM-Aware Parsing:** Automatically extracts provider, model, token usage, estimated USD cost, response text, and tool calls for **OpenAI** (Chat Completions & Responses API), **Anthropic** (Messages API), and **Google Gemini** (`generateContent`).
- **Built-in Assertions:**
  - `session.assert_cost_under(0.01)`
  - `session.assert_tokens_under(2000)`
  - `session.assert_tool_called("search_docs", min_times=1)`
  - `session.assert_tool_sequence(["search_docs", "summarize"])`
  - `session.assert_call_count(2)`
- **Pytest Plugin + CLI:** Auto-discovered `llm_tape` fixture (`--tape-mode`, `--tape-dir`) and `llm-tape inspect` / `llm-tape summary` CLI commands.

## Installation

```bash
pip install llm-tape
# or with uv:
uv add --dev llm-tape
```

## Quickstart

### 1. Context Manager

```python
import httpx
from llm_tape import tape

with tape("tests/tapes/weather.tape.json", mode="once") as session:
    # Call OpenAI / Anthropic / Gemini SDK or httpx directly
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-secret"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    session.assert_call_count(1)
    session.assert_tokens_under(500)
    session.assert_cost_under(0.005)
```

### 2. Pytest Fixture

```python
import pytest

@pytest.mark.tape(name="agent_search")
def test_agent_workflow(llm_tape):
    result = run_my_agent("Find Python 3.13 release notes")

    llm_tape.assert_tool_sequence(["web_search", "summarize"])
    llm_tape.assert_cost_under(0.01)
    assert "3.13" in result
```

Run in CI with strict offline replay (fails if a cassette is missing or a prompt changed):

```bash
pytest --tape-mode=replay
```

Re-record cassettes after intentionally changing prompts:

```bash
pytest --tape-mode=record
```

### 3. CLI Inspection

```bash
llm-tape inspect tests/tapes/test_agent/agent_search.tape.json
llm-tape summary tests/tapes
```

## License

MIT
