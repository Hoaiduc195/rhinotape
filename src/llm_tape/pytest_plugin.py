"""Pytest plugin for llm-tape (auto-discovered via pytest11 entry point)."""

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

import pytest

from llm_tape.models import TapeMode
from llm_tape.session import TapeSession

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_node_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "test"


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
        "tape(name=None, mode=None): configure cassette name or mode for the llm_tape fixture.",
    )


@pytest.fixture
def llm_tape(request: pytest.FixtureRequest) -> Generator[TapeSession, None, None]:
    """Provide an active TapeSession scoped to the current pytest test."""
    cli_mode = request.config.getoption("--tape-mode")
    cli_dir = request.config.getoption("--tape-dir")

    marker = request.node.get_closest_marker("tape")
    marker_name = marker.kwargs.get("name") if marker else None
    marker_mode = marker.kwargs.get("mode") if marker else None

    mode = TapeMode.coerce(marker_mode or cli_mode)

    test_path = Path(str(request.path))
    base_dir = Path(cli_dir) if cli_dir else test_path.parent / "tapes"
    file_slug = _sanitize_node_name(test_path.stem)
    test_slug = _sanitize_node_name(marker_name or request.node.name)
    cassette_path = base_dir / file_slug / f"{test_slug}.tape.json"

    with TapeSession(cassette_path, mode=mode) as session:
        yield session
