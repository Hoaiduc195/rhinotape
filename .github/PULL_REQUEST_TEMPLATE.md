## Summary

<!-- Briefly describe what this PR changes and why. Link any related issue (`Closes #123`). -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / provider / integration
- [ ] Documentation / examples
- [ ] Refactor / performance / tests

## Checklist

- [ ] Added or updated offline tests (no live API calls or real secrets)
- [ ] `uv run ruff check src tests examples` & `uv run ruff format --check src tests examples` pass
- [ ] `uv run mypy` passes (`strict = true`)
- [ ] `uv run pytest -v -W error` passes with coverage `>= 93%`
- [ ] `uv run llm-tape verify . --check-secrets --check-pii` passes
- [ ] No new hard runtime dependencies added (`httpx` remains the sole core dependency)
