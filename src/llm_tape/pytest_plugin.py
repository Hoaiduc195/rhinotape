"""Pytest plugin for llm-tape (auto-discovered via pytest11 entry point)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Generator
from pathlib import Path

import pytest

from llm_tape.models import TapeMode
from llm_tape.session import TapeSession

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_STEM_LEN = 120


def _sanitize_segment(name: str, *, unique_seed: str | None = None) -> str:
    sanitized = _SAFE_NAME_RE.sub("_", name).strip("_") or "test"
    needs_hash = sanitized != name or len(sanitized) > _MAX_STEM_LEN
    if len(sanitized) > _MAX_STEM_LEN:
        sanitized = sanitized[:_MAX_STEM_LEN].rstrip("_")
    if needs_hash and unique_seed:
        short_hash = hashlib.sha256(unique_seed.encode("utf-8")).hexdigest()[:8]
        return f"{sanitized}_{short_hash}"
    return sanitized


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm-tape", "Record and replay LLM HTTP interactions")
    group.addoption(
        "--tape-mode",
        action="store",
        default=TapeMode.ONCE.value,
        choices=[m.value for m in TapeMode],
        help="Tape recording mode: once (default), replay, record, or none.",
    )
    group.addoption(
        "--tape-dir",
        action="store",
        default=None,
        help="Base directory for .tape.json cassettes (defaults to <test_dir>/tapes).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "tape(name=None, mode=None, ignore_localhost=False, ignore_hosts=(), ignore_fields=()): "
        "configure cassette options for the llm_tape fixture.",
    )


@pytest.fixture
def llm_tape(request: pytest.FixtureRequest) -> Generator[TapeSession, None, None]:
    """Provide an active TapeSession scoped to the current pytest test."""
    cli_mode = str(request.config.getoption("--tape-mode"))
    cli_dir = request.config.getoption("--tape-dir")

    marker = request.node.get_closest_marker("tape")
    marker_kwargs = marker.kwargs if marker else {}
    marker_name = marker_kwargs.get("name")
    marker_mode = marker_kwargs.get("mode")

    mode = TapeMode.coerce(marker_mode or cli_mode)

    test_path = Path(str(request.path))
    if cli_dir:
        base_dir = Path(str(cli_dir))
        try:
            rel_path = test_path.relative_to(Path(str(request.config.rootpath)))
            file_slug = _sanitize_segment(
                rel_path.with_suffix("").as_posix().replace("/", "__"),
                unique_seed=rel_path.as_posix(),
            )
        except ValueError:
            file_slug = _sanitize_segment(test_path.stem, unique_seed=str(test_path))
    else:
        base_dir = test_path.parent / "tapes"
        file_slug = _sanitize_segment(test_path.stem, unique_seed=str(test_path))

    # Avoid parametrized tests colliding when a fixed marker name is given
    raw_test_name = str(marker_name) if marker_name else request.node.name
    if marker_name and "[" in request.node.name:
        param_suffix = request.node.name[request.node.name.index("[") :]
        raw_test_name = f"{marker_name}{param_suffix}"

    test_slug = _sanitize_segment(raw_test_name, unique_seed=request.node.nodeid)
    cassette_path = base_dir / file_slug / f"{test_slug}.tape.json"

    with TapeSession(
        cassette_path,
        mode=mode,
        ignore_localhost=bool(marker_kwargs.get("ignore_localhost", False)),
        ignore_hosts=marker_kwargs.get("ignore_hosts", ()),
        ignore_fields=marker_kwargs.get("ignore_fields", ()),
    ) as session:
        yield session
