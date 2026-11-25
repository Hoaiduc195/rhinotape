# Changelog

## 0.3.0 - 2026-11-25

### Added
- **Single-source dynamic versioning** via `src/llm_tape/_version.py` (`hatch` dynamic version).
- **Cached input & reasoning token tracking + external pricing files:** `Usage.cached_input_tokens`, `Usage.reasoning_tokens` (OpenAI Chat Completions/Responses & Gemini `thoughtsTokenCount`), `o3`/`o4-mini` pricing, `ModelPrice.cached_input_per_million`, `load_pricing_file()`, and `LLM_TAPE_PRICING_FILE` env var override.
- **Deterministic gzip-compressed cassettes (`.tape.json.gz`):** Transparent load/save with `mtime=0` deterministic headers, bounded decompression safety (`MAX_CASSETTE_BYTES`), and `--tape-compress` / `tape_compress` / `@pytest.mark.tape(compress=...)`.
- **Chaos & fault-injection replay (`on_replay` + `replace_response`):** Inject 429/503 errors, mutated LLM responses, or custom latency on replay across all HTTP adapters without modifying cassettes.
- **`TapeMode.NEW_EPISODES`:** VCR-classic mode that replays existing matches offline while recording and appending only newly added requests (preserving existing `cassette.metadata`).
- **Optional `requests`, `aiohttp`, and stdlib `urllib.request` adapters:** Thread-safe, `contextvars`-isolated interception for `requests` (`llm-tape[requests]`), async `aiohttp` (`llm-tape[aiohttp]`), and opt-in stdlib `urllib.request` (`patch_urllib=True`) with zero new hard runtime dependencies.
- **Extended thinking, OpenAI Embeddings, Cohere v1/v2 & WHATWG SSE hardening:** `LLMCallSummary.thinking` extraction (Anthropic, DeepSeek-R1, Gemini), OpenAI embeddings & Cohere JSON/SSE parsers + pricing, and WHATWG SSE spec compliance (UTF-8 BOM, CRLF, single-space strip, bare fields).
- **Hot-path performance optimizations:** Pre-compiled `redact_patterns`, single-pass pattern-only `redact_body`, mtime/size-cached `LLM_TAPE_PRICING_FILE`, and zero-copy `DEFAULT_PRICING` lookup.
- **Selective header matching & strict unused checks:** `match_headers` whitelist in `fingerprint_request`/`TapeSession` and `--tape-strict-unused` / `tape_strict_unused` pytest enforcement.
- **Expanded assertions & session telemetry:** `assert_text_contains()`, `assert_text_matches()`, `assert_no_tool_called()`, `assert_thinking_present()`, `assert_reasoning_tokens_under()`, `assert_cached_tokens_at_least()`, `assert_model()`, `assert_provider()`, `assert_no_errors()`, `export_summary()`, `last_text`, `last_thinking`, and `all_texts`.
- **Full CI/CD CLI suite & pre-commit hooks:** `llm-tape stats` (per-provider/model cache hit % & avg/p95 latency), `llm-tape inspect --show-bodies --show-thinking --interaction` + `Cassette.content_hash()`, `llm-tape schema` (Draft 2020-12 JSON Schema), `llm-tape export` (Markdown/CSV/HAR 1.2), `llm-tape convert` (gzip ↔ json with `--force`), multi-target `llm-tape verify --check-secrets`, `.pre-commit-hooks.yaml`, `llm-tape diff` directory comparison & CI gates, `llm-tape prune`, `llm-tape redact`, and `llm-tape summary --max-cost` / `--max-tokens`.
- **Trusted Publishing workflow:** `.github/workflows/publish.yml` for automated TestPyPI & PyPI releases.

## 0.2.0 - 2026-11-25

### Added
- **True SSE Streaming support** (sync & async `httpx` streaming record/replay + token/tool-call parsing) for OpenAI Chat Completions, OpenAI Responses API, Anthropic Messages, and Google Gemini (`:streamGenerateContent`).
- **`contextvars`-based `httpx` dispatcher** with reference-counted thread-safe patching for nested sessions and concurrent `asyncio.gather` tasks.
- **Request filtering & flexible matching:** `ignore_hosts`, `ignore_localhost`, `ignore_request`, `ignore_fields` (dot-path + `*` wildcards), `ignore_query_params`, and `allow_playback_repeats`.
- **Actionable `UnmatchedRequestError` diagnostics:** distinguishes exhausted replay queues from unmatched requests and prints a unified diff against the closest recorded request.
- **Atomic cassette writes** (`NamedTemporaryFile` + `os.fsync` + `os.replace`) and safe default `save_on_error=False` so failing tests never overwrite cassettes with partial data.
- **PEP 561 `py.typed` marker**, strict `mypy` verification, `pytester` end-to-end plugin tests, and GitHub Actions CI matrix (Python 3.10–3.13).

### Fixed
- Binary payloads (`base64`) and JSON primitives (`null`, strings) now round-trip faithfully via explicit `body_encoding`.
- Anthropic cache tokens (`cache_creation_input_tokens`, `cache_read_input_tokens`) and Gemini `thoughtsTokenCount` are now included in usage and cost calculations.
- Boundary-safe model pricing prefix matching prevents short prefixes like `gpt-4` from matching `gpt-4o`.
- Parametrized pytest tests with `@pytest.mark.tape(name=...)` no longer collide on cassette filenames.
- `Cassette.load()` wraps malformed JSON and unsupported schema versions in `CorruptCassetteError`.

## 0.1.0 - 2026-11-25

- Initial release with sync/async `httpx` record & replay, secret scrubbing, OpenAI/Anthropic/Gemini JSON parsers, assertions, pytest plugin, and CLI.
