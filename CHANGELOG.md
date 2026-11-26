# Changelog

## 0.4.0 - 2026-11-26

### Changed
- **Renamed package to `rhinotape`** (`pip install rhinotape`, `import rhinotape`, `rhinotape` CLI, `rhinotape` / `rhinotape_dir` pytest fixtures, `RHINOTAPE_*` env vars) with new rhinoceros brand logo and redesigned docs.

### Added
- **Zero-code CLI wrapper (`rhinotape run`) & `python -m rhinotape`:** Wrap any test runner or app (`npm test`, `vitest`, `go test`, `cargo test`, `pytest`) against an ephemeral local reverse proxy with automatic provider base-URL injection and `--max-cost` / `--max-tokens` / `--strict-unused` CI gates.
- **Universal Python framework integration layer (`rhinotape.integrations`):** Pure ASGI 3.0 `TapeASGIMiddleware` (FastAPI, Starlette, Litestar, Django ASGI), pure PEP 3333 `TapeWSGIMiddleware` (Flask, Django WSGI, Bottle, Falcon), `TapeConfig.from_env()`, `current_session()` / `require_session()`, `make_httpx_client()` / `make_async_httpx_client()`, `TapeTestCaseMixin`, `activate_tape()` / `activate_from_env()`, and `handle_generic_request()` bridge.
- **Polyglot reverse-proxy & HTTP control plane (`TapeReplayServer`):** Record/replay proxy modes (`once`, `record`, `new_episodes`), CORS support, PII redaction, and `/__tape__/{health,stats,interactions,reset,load,save,assert}` endpoints.

### Fixed
- **Production & security hardening:** Bounded request/response sizes (`max_request_bytes` → HTTP 413, upstream `MAX_CASSETTE_BYTES` cap), bounded chunked decoding, request timeouts, thread-safe concurrent saves & `max_interactions` accounting, strict `NaN`/`Infinity`/`bool` validation across cassettes/models/pricing/assertions, expanded secret & base64 verification in `rhinotape verify`, and Python 3.10–3.13 + Windows cross-platform CLI/signal fixes.

## 0.3.0 - 2026-11-25

### Added
- **Single-source dynamic versioning** via `src/rhinotape/_version.py` (`hatch` dynamic version).
- **Cached input & reasoning token tracking + external pricing files:** `Usage.cached_input_tokens`, `Usage.reasoning_tokens` (OpenAI Chat Completions/Responses & Gemini `thoughtsTokenCount`), `o3`/`o4-mini` pricing, `ModelPrice.cached_input_per_million`, `load_pricing_file()`, and `RHINOTAPE_PRICING_FILE` env var override.
- **Deterministic gzip-compressed cassettes (`.tape.json.gz`):** Transparent load/save with `mtime=0` deterministic headers, bounded decompression safety (`MAX_CASSETTE_BYTES`), and `--tape-compress` / `tape_compress` / `@pytest.mark.tape(compress=...)`.
- **Chaos & fault-injection replay (`on_replay` + `replace_response`):** Inject 429/503 errors, mutated LLM responses, or custom latency on replay across all HTTP adapters without modifying cassettes.
- **`TapeMode.NEW_EPISODES`:** VCR-classic mode that replays existing matches offline while recording and appending only newly added requests (preserving existing `cassette.metadata`).
- **Optional `requests`, `aiohttp`, and stdlib `urllib.request` adapters:** Thread-safe, `contextvars`-isolated interception for `requests` (`rhinotape[requests]`), async `aiohttp` (`rhinotape[aiohttp]`), and opt-in stdlib `urllib.request` (`patch_urllib=True`) with zero new hard runtime dependencies.
- **Extended thinking, OpenAI Embeddings, Cohere v1/v2 & WHATWG SSE hardening:** `LLMCallSummary.thinking` extraction (Anthropic, DeepSeek-R1, Gemini), OpenAI embeddings & Cohere JSON/SSE parsers + pricing, and WHATWG SSE spec compliance (UTF-8 BOM, CRLF, single-space strip, bare fields).
- **Hot-path performance optimizations:** Pre-compiled `redact_patterns`, single-pass pattern-only `redact_body`, mtime/size-cached `RHINOTAPE_PRICING_FILE`, and zero-copy `DEFAULT_PRICING` lookup.
- **Selective header matching & strict unused checks:** `match_headers` whitelist in `fingerprint_request`/`TapeSession` and `--tape-strict-unused` / `tape_strict_unused` pytest enforcement.
- **Expanded assertions & session telemetry:** `assert_text_contains()`, `assert_text_matches()`, `assert_no_tool_called()`, `assert_thinking_present()`, `assert_reasoning_tokens_under()`, `assert_cached_tokens_at_least()`, `assert_model()`, `assert_provider()`, `assert_no_errors()`, `export_summary()`, `last_text`, `last_thinking`, and `all_texts`.
- **Full CI/CD CLI suite & pre-commit hooks:** `rhinotape stats` (per-provider/model cache hit %, cache savings USD, avg/p95 latency), `rhinotape inspect --show-bodies --show-thinking --interaction` + `Cassette.content_hash()`, `rhinotape schema` (Draft 2020-12 JSON Schema), `rhinotape export` (Markdown/CSV/HAR 1.2/HTML/JUnit XML), `rhinotape convert` (gzip ↔ json with `--force`), `rhinotape verify --check-secrets --check-pii`, `.pre-commit-hooks.yaml`, `rhinotape diff` directory comparison, `--show-text-diff` & CI gates, `rhinotape prune`, `rhinotape redact --pii`, `rhinotape summary --group-by dir --max-cost --max-tokens`, `rhinotape search`, `rhinotape list`, `rhinotape pricing`, `rhinotape replay-check`, `rhinotape reparse`, `rhinotape merge`, `rhinotape split`, and `rhinotape serve`.
- **Cassette operations & in-memory / pluggable persistence:** `Cassette.merge()`, `filter()`, `slice()`, `deduplicate()`, `clone()`, `stats()`, `sort_interactions()`, `validate()`, `add_interaction()`, `to_json()`/`from_json()`, `to_bytes()`/`from_bytes()`, `CassetteStore` Protocol, thread-safe `MemoryCassetteStore`, path-traversal-guarded `FileCassetteStore`, and `TapeSession(..., store=...)`.
- **Local HTTP Replay Server & Interactive HTML Trace Viewer:** Stdlib `TapeReplayServer` (`/__tape__/health`, `/__tape__/stats`, SSE streaming, latency pacing, query-param scrubbing on 404s) and zero-CDN, XSS-safe `render_html_report()` wired into `rhinotape export --format html` and `pytest --tape-html-report`.
- **PII Presets, Wildcard Header Scrubbing & URL Path Redaction:** Built-in `PII_EMAIL`, `PII_PHONE`, `PII_SSN`, `PII_CREDIT_CARD`, `PII_IPV4`, `PII_JWT`, `PII_PRESETS`, `resolve_pii_patterns()`, `redact_pii`, `fnmatch` wildcards in `scrub_headers()`, and `redact_url_patterns`.
- **More Providers & Cache-Write Billing:** Mistral FIM (`codestral`/`ministral-*`), xAI Grok (`grok-2`/`grok-3`/`grok-3-mini`), Perplexity (`sonar*`), OpenAI Audio/Image endpoints, Hugging Face router, `ModelPrice.cache_write_per_million` (Anthropic 1.25x), `Usage.cache_creation_tokens`, and `estimate_cache_savings_usd()`.
- **Composable Request Matchers & Chaos Toolkit:** `MATCH_STRICT`, `MATCH_RELAXED`, `MATCH_PATH_AND_BODY`, `ignore_json_array_order`, `match_all`/`match_any`/`match_method`/`match_path`/`match_json_subset`/`match_json_regex`/`match_header_equals`, plus `fail_nth`, `fail_first`, `inject_latency`, `corrupt_json_body`, `truncate_stream`, and `chain_replay_hooks`.
- **Recording Hooks, Live Circuit Breakers & Advanced Assertions:** `on_record`, custom cassette `metadata`, `TapeSession.save()`, `max_interactions`, `max_cost_usd`, `assert_json_output`, `assert_max_cost_per_call`, `assert_max_tokens_per_call`, `assert_streamed`/`assert_not_streamed`, `assert_status_codes`, `assert_tool_call_count`, and `assert_cache_hit_rate_at_least`.
- **Pytest Plugin & Diagnostics UX:** `tape_max_cost`, `tape_max_tokens`, `tape_redact_pii` ini options, `RHINOTAPE_*` env-var fallbacks, `rhinotape_dir` fixture, `--tape-update-stale <days>`, semantic JSON key-path diff summaries in `format_request_diff()`, and structured `UnmatchedRequestError` (`.incoming`, `.closest`, `.diff_text`).
- **Performance & Security Hardening:** Direct dict literals in `to_dict()`, pre-computed `_detect_provider()` lookup tables, cached `compile_sensitive_headers()`, lazy UTF-8 SSE replay encoding, and pre-read `MAX_CASSETTE_BYTES` stat guard in `Cassette.load()`.
- **Trusted Publishing workflow & Community Docs:** `.github/workflows/publish.yml`, `CONTRIBUTING.md`, `SECURITY.md`, and runnable examples in `examples/`.

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
