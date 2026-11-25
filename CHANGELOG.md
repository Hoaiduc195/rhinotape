# Changelog

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
