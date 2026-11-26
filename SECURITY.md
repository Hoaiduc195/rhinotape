# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release on `master`.

## Reporting a Vulnerability

If you discover a security vulnerability (for example, a secret-scrubbing bypass, XSS in HTML trace reports, path-traversal in `FileCassetteStore`, or unbounded decompression), please do **not** open a public issue.

Instead, report it privately via GitHub Security Advisories on `Hoaiduc195/rhinotape` or contact the maintainer directly. Please include:

- A description of the vulnerability and its impact.
- Minimal steps or a proof-of-concept cassette/request to reproduce it.
- Any suggested remediation if known.

## Security Guarantees & Best Practices

- `rhinotape` automatically scrubs common API key headers (`Authorization`, `x-api-key`, `x-goog-api-key`, etc.), sensitive query parameters (`key`, `api_key`, `token`), and URL userinfo before writing cassettes.
- Use `redact_pii=True` (or specific PII presets) when recording prompts containing user data.
- Run `rhinotape verify <dir> --check-secrets --check-pii` in CI or via `.pre-commit-hooks.yaml` to block accidental leaks.
- Cassette loading enforces `MAX_CASSETTE_BYTES` (100 MiB) on both plain JSON and gzip-compressed files to guard against zip bombs.
