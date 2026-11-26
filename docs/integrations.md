# Framework & Polyglot Integrations

`rhinotape` integrates with any Python web framework, LLM SDK, test runner, or polyglot language stack with **zero extra runtime dependencies** (`httpx` remains the sole core dependency).

All Python integration helpers live in `rhinotape.integrations`:

- `current_session()` / `require_session()` — contextvars accessors for the active `TapeSession`
- `TapeConfig` / `TapeConfig.from_env()` / `tape_from_env()` — reusable configuration & env-var bootstrap
- `make_httpx_client()` / `make_async_httpx_client()` — explicit SDK client factories (`patch_httpx=False`)
- `TapeASGIMiddleware` — pure ASGI 3.0 middleware (FastAPI, Starlette, Litestar, Quart, Django ASGI)
- `TapeWSGIMiddleware` — pure PEP 3333 WSGI middleware (Flask, Django WSGI, Bottle, Pyramid, Falcon)
- `TapeTestCaseMixin` — stdlib `unittest.TestCase` & Django `TestCase` mixin
- `activate_tape()` / `activate_from_env()` — process/worker-level activation handle
- `GenericHTTPRequest` / `GenericHTTPResponse` / `handle_generic_request()` / `handle_generic_request_async()` — bridge for custom HTTP clients

---

## 1. Environment-Driven Config (`TapeConfig`)

Configure recording, scrubbing, and budgets once via environment variables:

```bash
export RHINOTAPE_MODE=once                  # once | replay | record | new_episodes | none
export RHINOTAPE_DIR=tests/tapes
export RHINOTAPE_COMPRESS=true
export RHINOTAPE_IGNORE_LOCALHOST=true
export RHINOTAPE_IGNORE_FIELDS=request_id,timestamp
export RHINOTAPE_REDACT_PII=email,phone,ssn
export RHINOTAPE_MAX_COST=0.25
export RHINOTAPE_MAX_INTERACTIONS=10
```

```python
from rhinotape.integrations import TapeConfig, current_session, require_session

config = TapeConfig.from_env()

@config.decorator("checkout_agent")
async def run_checkout_agent(prompt: str) -> str:
    # Any helper or callback can access the active session without threading it through args:
    session = require_session()
    ...
    session.assert_cost_under(0.05)
    return "done"
```

---

## 2. Official LLM SDKs (OpenAI, Anthropic, Gemini, Groq, Mistral)

When you prefer explicit dependency injection over global patching (`patch_httpx=False`), use `make_httpx_client` or `make_async_httpx_client`:

### OpenAI / Groq / Mistral / xAI / OpenRouter

```python
from openai import AsyncOpenAI, OpenAI
from rhinotape import TapeMode, TapeSession
from rhinotape.integrations import make_async_httpx_client, make_httpx_client

with TapeSession("tapes/openai_chat.tape.json", mode=TapeMode.ONCE, patch_httpx=False) as session:
    client = OpenAI(http_client=make_httpx_client(session))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize rhinotape in one sentence."}],
    )
    session.assert_tokens_under(200)
```

### Anthropic

```python
from anthropic import Anthropic
from rhinotape import TapeMode, TapeSession
from rhinotape.integrations import make_httpx_client

with TapeSession("tapes/claude.tape.json", mode=TapeMode.ONCE, patch_httpx=False) as session:
    client = Anthropic(http_client=make_httpx_client(session))
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    session.assert_provider("anthropic")
```

### Google Gemini (`google-genai`)

`google-genai` uses `httpx` under the hood, so default `TapeSession` patching intercepts it automatically, or you can pass `http_options={"httpx_client": make_httpx_client(session)}`.

---

## 3. Agent Frameworks (LangChain, LlamaIndex, PydanticAI)

### LangChain

```python
from langchain_openai import ChatOpenAI
from rhinotape.integrations import TapeConfig, make_httpx_client

cfg = TapeConfig.from_env()
with cfg.session("langchain_chain", patch_httpx=False) as session:
    llm = ChatOpenAI(model="gpt-4o-mini", http_client=make_httpx_client(session))
    result = llm.invoke("Give me 3 bullet points on deterministic tests.")
    session.assert_call_count(1)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from rhinotape.integrations import TapeConfig, make_httpx_client, make_async_httpx_client

cfg = TapeConfig.from_env()
with cfg.session("llamaindex_query", patch_httpx=False) as session:
    llm = LlamaOpenAI(
        model="gpt-4o-mini",
        http_client=make_httpx_client(session),
        async_http_client=make_async_httpx_client(session),
    )
    answer = llm.complete("Explain prompt caching.")
    session.assert_no_errors()
```

### PydanticAI

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from rhinotape.integrations import TapeConfig, make_async_httpx_client

cfg = TapeConfig.from_env()
async with cfg.session("pydantic_ai_agent", patch_httpx=False) as session:
    provider = OpenAIProvider(http_client=make_async_httpx_client(session))
    agent = Agent(OpenAIModel("gpt-4o-mini", provider=provider))
    res = await agent.run("Extract the city from: Flight to Tokyo.")
    session.assert_tokens_under(500)
```

---

## 4. FastAPI / Starlette / Litestar / Django ASGI (`TapeASGIMiddleware`)

`TapeASGIMiddleware` is pure ASGI 3.0 — no framework imports needed. It skips `lifespan` and `websocket` scopes, isolates each HTTP request with `contextvars`, exposes `scope["state"]["rhinotape"]`, and optionally emits `X-Tape-Calls`, `X-Tape-Tokens`, and `X-Tape-Cost-Usd` response headers.

```python
from fastapi import FastAPI, Request
from rhinotape.integrations import TapeASGIMiddleware, TapeConfig, require_session

app = FastAPI()
app.add_middleware(
    TapeASGIMiddleware,
    config=TapeConfig.from_env(),
    expose_headers=True,
    skip_paths=("/healthz", "/metrics"),
)

@app.post("/v1/chat")
async def chat(request: Request):
    # Access either via request.state.rhinotape or require_session()
    session = require_session()
    ...
    return {"calls": session.call_count, "tokens": session.total_tokens}
```

Clients (or E2E tests) can select a cassette per request using the `X-Tape-Name` header (path traversal characters are rejected safely), or you can supply a custom `cassette_resolver=lambda scope: ...` (returning `None` skips recording/replay for that request).

---

## 5. Flask / Django WSGI (`TapeWSGIMiddleware`)

`TapeWSGIMiddleware` is pure PEP 3333 WSGI and keeps the `TapeSession` open until streaming response iterators finish or `close()` is called.

### Flask

```python
from flask import Flask, request
from rhinotape.integrations import TapeConfig, TapeWSGIMiddleware, require_session

app = Flask(__name__)
app.wsgi_app = TapeWSGIMiddleware(
    app.wsgi_app,
    config=TapeConfig.from_env(),
    expose_headers=True,
    skip_paths=("/health",),
)

@app.post("/ask")
def ask():
    session = request.environ["rhinotape.session"]  # or require_session()
    ...
    return {"tokens": session.total_tokens}
```

### Django WSGI (`wsgi.py`)

```python
from django.core.wsgi import get_wsgi_application
from rhinotape.integrations import TapeConfig, TapeWSGIMiddleware

application = get_wsgi_application()
application = TapeWSGIMiddleware(application, config=TapeConfig.from_env())
```

---

## 6. stdlib `unittest` & Django `TestCase` (`TapeTestCaseMixin`)

```python
import unittest
from pathlib import Path
from rhinotape import TapeMode
from rhinotape.integrations import TapeTestCaseMixin

class ChatServiceTests(TapeTestCaseMixin, unittest.TestCase):
    tape_dir = Path("tests/tapes")
    tape_mode = TapeMode.ONCE
    tape_strict_unused = True
    tape_kwargs = {"redact_pii": "email"}

    def test_chat_completion(self) -> None:
        # Cassette saved to tests/tapes/ChatServiceTests__test_chat_completion.tape.json
        ...
        self.rhinotape.assert_call_count(1)
        self.rhinotape.assert_cost_under(0.01)
```

Failed tests automatically pass `exc_occurred=True` so broken recordings are not saved unless `save_on_error=True`.

---

## 7. Workers, Eval Scripts & App Entrypoints (`activate_from_env`)

Safe to call unconditionally in Celery workers, Gradio/Streamlit apps, or CLI scripts — returns `None` when neither `RHINOTAPE_ENABLED` nor `RHINOTAPE_CASSETTE` is set:

```python
import atexit
from rhinotape.integrations import activate_from_env

handle = activate_from_env()
if handle is not None:
    atexit.register(handle.close)
```

---

## 8. Custom HTTP Libraries (`GenericHTTPRequest` Bridge)

Integrate `curl_cffi`, `niquests`, `pycurl`, `boto3`, or any custom HTTP transport in ~10 lines while keeping 100% of `TapeSession` features (scrubbing, PII redaction, matching, chaos hooks, circuit breakers, assertions):

```python
from rhinotape import TapeMode, TapeSession
from rhinotape.integrations import (
    GenericHTTPRequest,
    GenericHTTPResponse,
    handle_generic_request,
)

def send_with_custom_lib(req: GenericHTTPRequest) -> GenericHTTPResponse:
    # Call curl_cffi / niquests / pycurl / boto3 here
    raw = my_http_client.request(req.method, req.url, headers=req.headers, data=req.body)
    return GenericHTTPResponse(
        status_code=raw.status_code,
        headers=dict(raw.headers),
        body=raw.content,
    )

with TapeSession("tapes/custom_http.tape.json", mode=TapeMode.ONCE) as session:
    resp = handle_generic_request(
        session,
        GenericHTTPRequest("POST", "https://api.openai.com/v1/chat/completions", body=b"{}"),
        send_with_custom_lib,
    )
```

For async clients, use `await handle_generic_request_async(session, req, async_sender)`.

---

## 9. Polyglot Frameworks via `rhinotape serve` (Reverse-Proxy Record/Replay + Control API)

For **Node.js / TypeScript, Go, Rust, Java, Ruby, Playwright, Vitest, Jest, n8n, Dify**, or multi-service **Docker Compose** setups, run `rhinotape serve` as a deterministic local HTTP reverse proxy and point your SDK's base URL at it:

```bash
# Offline replay mode:
rhinotape serve tests/tapes/ --host 127.0.0.1 --port 8787 \
  --ignore-fields request_id,timestamp \
  --match-on method,path,body

# Reverse-proxy record-once mode with CORS + automatic PII redaction:
rhinotape serve tests/tapes/checkout.tape.json --port 8787 \
  --mode once \
  --upstream https://api.openai.com \
  --cors \
  --redact-pii

# Zero-code wrapper for ANY language/framework (starts ephemeral proxy, injects *_BASE_URL env vars, checks budgets):
rhinotape run tests/tapes/checkout.tape.json --mode replay --max-cost 0.05 --strict-unused -- npm test
rhinotape run tests/tapes/go_agent.tape.json --mode once --upstream https://api.openai.com -- go test ./...
```

### Built-in HTTP Control Plane (`/__tape__/*`)

External test runners (Vitest, Jest, Playwright, Go `testing`, RSpec, JUnit) can control cassettes and run full LLM assertions over HTTP:

- `GET /__tape__/health` — server status, mode, active cassette, and loaded interaction count
- `GET /__tape__/stats` — `loaded_interactions`, `served_requests`, `recorded_interactions`, `unmatched_requests`, `total_tokens`, `estimated_cost_usd`, `tool_calls`
- `GET /__tape__/interactions` — structured JSON summaries of all served/recorded interactions
- `POST /__tape__/reset` — resets replay queues and counters between tests
- `POST /__tape__/load` — switches active cassette safely inside the cassette root (`{"cassette": "login.tape.json", "mode": "replay"}`)
- `POST /__tape__/save` — flushes recorded interactions to disk atomically
- `POST /__tape__/assert` — declarative assertions returning `200 {"ok": true, "failures": []}` or `400 {"ok": false, "failures": [...]}`:

```bash
curl -sS -X POST http://127.0.0.1:8787/__tape__/assert \
  -H 'Content-Type: application/json' \
  -d '{
    "call_count": 2,
    "max_tokens": 1500,
    "max_cost_usd": 0.02,
    "no_errors": true,
    "tool_sequence": ["web_search", "summarize"],
    "no_unused_interactions": true
  }'
```

### Node.js / TypeScript (OpenAI SDK / Vitest / Playwright)

```ts
import OpenAI from "openai";

const TAPE_BASE = process.env.RHINOTAPE_URL ?? "http://127.0.0.1:8787";

// Before each test, switch cassette cleanly over HTTP:
await fetch(`${TAPE_BASE}/__tape__/load`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ cassette: "chat_flow.tape.json", mode: "replay" }),
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY ?? "test-key",
  baseURL: `${TAPE_BASE}/v1`,
});

const completion = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello" }],
});

// Assert token/cost/tool budgets from TypeScript:
const check = await fetch(`${TAPE_BASE}/__tape__/assert`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ call_count: 1, max_cost_usd: 0.01, no_errors: true }),
}).then((r) => r.json());
if (!check.ok) throw new Error(check.failures.join("; "));
```

### Go

```go
cfg := openai.DefaultConfig("test-key")
cfg.BaseURL = "http://127.0.0.1:8787/v1"
client := openai.NewClientWithConfig(cfg)
```

### Rust

```rust
let config = async_openai::config::OpenAIConfig::new()
    .with_api_key("test-key")
    .with_api_base("http://127.0.0.1:8787/v1");
let client = async_openai::Client::with_config(config);
```

### Docker Compose (CI / E2E Testing)

```yaml
services:
  rhinotape:
    image: python:3.12-slim
    working_dir: /app
    volumes:
      - ./tests/tapes:/tapes:ro
    command: >
      sh -c "pip install -q rhinotape &&
      rhinotape serve /tapes --host 0.0.0.0 --port 8787 --match-on method,path,body"
    ports:
      - "8787:8787"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/__tape__/health')"]
      interval: 2s
      timeout: 2s
      retries: 10

  app:
    build: .
    depends_on:
      rhinotape:
        condition: service_healthy
    environment:
      OPENAI_BASE_URL: http://rhinotape:8787/v1
      OPENAI_API_KEY: dummy-replay-key
```

`rhinotape serve` handles `SIGTERM` cleanly so Docker Compose and CI jobs shut down gracefully.
