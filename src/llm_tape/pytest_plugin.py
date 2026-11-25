"""Pytest plugin for llm-tape (auto-discovered via pytest11 entry point)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from llm_tape.models import TapeMode
from llm_tape.scrubbing import DEFAULT_MATCH_ON
from llm_tape.session import TapeSession

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_STEM_LEN = 120
_FAILED_ATTR = "_llm_tape_call_failed"


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
        default=None,
        choices=[m.value for m in TapeMode],
        help="Tape recording mode: once (default), replay, record, or none.",
    )
    group.addoption(
        "--tape-dir",
        action="store",
        default=None,
        help="Base directory for .tape.json cassettes (defaults to <test_dir>/tapes).",
    )

    parser.addini(
        "tape_mode",
        help="Default llm-tape mode (once, replay, record, none).",
        default=TapeMode.ONCE.value,
    )
    parser.addini(
        "tape_dir",
        help="Default base directory for .tape.json cassettes.",
        default="",
    )
    parser.addini(
        "tape_ignore_localhost",
        help="Ignore localhost HTTP requests by default.",
        type="bool",
        default=False,
    )
    parser.addini(
        "tape_ignore_hosts",
        help="List of host patterns to ignore by default.",
        type="linelist",
        default=[],
    )
    parser.addini(
        "tape_ignore_fields",
        help="List of JSON dot-paths to ignore during request matching by default.",
        type="linelist",
        default=[],
    )
    parser.addini(
        "tape_ignore_query_params",
        help="List of URL query parameters to ignore during request matching by default.",
        type="linelist",
        default=[],
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "tape(name=None, mode=None, ignore_localhost=None, ignore_hosts=(), "
        "ignore_fields=(), ignore_query_params=(), redact_fields=(), redact_patterns=(), "
        "scrub_body=None, match_on=None, custom_matcher=None, replay_latency_scale=0.0, "
        "allow_playback_repeats=False, save_on_error=False): "
        "configure cassette options for the llm_tape fixture.",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        setattr(item, _FAILED_ATTR, bool(rep.failed))


@pytest.fixture
def llm_tape(request: pytest.FixtureRequest) -> Generator[TapeSession, None, None]:
    """Provide an active TapeSession scoped to the current pytest test."""
    cli_mode = request.config.getoption("--tape-mode")
    cli_dir = request.config.getoption("--tape-dir")

    ini_mode = str(request.config.getini("tape_mode") or TapeMode.ONCE.value)
    ini_dir = str(request.config.getini("tape_dir") or "")
    ini_ignore_localhost = bool(request.config.getini("tape_ignore_localhost"))
    ini_ignore_hosts = list(request.config.getini("tape_ignore_hosts") or [])
    ini_ignore_fields = list(request.config.getini("tape_ignore_fields") or [])
    ini_ignore_query_params = list(request.config.getini("tape_ignore_query_params") or [])

    marker = request.node.get_closest_marker("tape")
    marker_kwargs = marker.kwargs if marker else {}
    marker_name = marker_kwargs.get("name")
    marker_mode = marker_kwargs.get("mode")

    mode = TapeMode.coerce(marker_mode or cli_mode or ini_mode)

    effective_dir = cli_dir or ini_dir or None
    test_path = Path(str(request.path))
    if effective_dir:
        base_dir = Path(str(effective_dir))
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

    # Include class name when no explicit marker name is set to avoid TestA.test_x vs TestB.test_x
    if marker_name:
        raw_test_name = str(marker_name)
        if "[" in request.node.name:
            param_suffix = request.node.name[request.node.name.index("[") :]
            raw_test_name = f"{marker_name}{param_suffix}"
    elif request.cls is not None:
        raw_test_name = f"{request.cls.__name__}__{request.node.name}"
    else:
        raw_test_name = request.node.name

    test_slug = _sanitize_segment(raw_test_name, unique_seed=request.node.nodeid)
    cassette_path = base_dir / file_slug / f"{test_slug}.tape.json"

    ignore_localhost = (
        bool(marker_kwargs["ignore_localhost"])
        if "ignore_localhost" in marker_kwargs
        else ini_ignore_localhost
    )
    ignore_hosts = (*ini_ignore_hosts, *marker_kwargs.get("ignore_hosts", ()))
    ignore_fields = (*ini_ignore_fields, *marker_kwargs.get("ignore_fields", ()))
    ignore_query_params = (*ini_ignore_query_params, *marker_kwargs.get("ignore_query_params", ()))

    session = TapeSession(
        cassette_path,
        mode=mode,
        ignore_localhost=ignore_localhost,
        ignore_hosts=ignore_hosts,
        ignore_fields=ignore_fields,
        ignore_query_params=ignore_query_params,
        redact_fields=marker_kwargs.get("redact_fields", ()),
        redact_patterns=marker_kwargs.get("redact_patterns", ()),
        scrub_body=marker_kwargs.get("scrub_body"),
        match_on=marker_kwargs.get("match_on") or DEFAULT_MATCH_ON,
        custom_matcher=marker_kwargs.get("custom_matcher"),
        replay_latency_scale=float(marker_kwargs.get("replay_latency_scale", 0.0)),
        allow_playback_repeats=bool(marker_kwargs.get("allow_playback_repeats", False)),
        save_on_error=bool(marker_kwargs.get("save_on_error", False)),
    )
    session.__enter__()
    try:
        yield session
    finally:
        test_failed = bool(getattr(request.node, _FAILED_ATTR, False))
        session._finalize(exc_occurred=test_failed)
