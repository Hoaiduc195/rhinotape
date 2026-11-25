"""Pytest plugin for llm-tape (auto-discovered via pytest11 entry point)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from llm_tape.models import TapeMode
from llm_tape.pricing import estimate_cache_savings_usd
from llm_tape.scrubbing import DEFAULT_MATCH_ON
from llm_tape.session import TapeSession

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_STEM_LEN = 120
_FAILED_ATTR = "_llm_tape_call_failed"
_REPORT_KEY: pytest.StashKey[list[dict[str, Any]]] = pytest.StashKey()
_VIOLATIONS_KEY: pytest.StashKey[list[str]] = pytest.StashKey()


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
    group.addoption(
        "--tape-compress",
        action="store_true",
        default=False,
        help="Write gzip-compressed .tape.json.gz cassettes.",
    )
    group.addoption(
        "--tape-report",
        action="store_true",
        default=False,
        help="Print a per-test LLM token and cost summary table at the end of pytest.",
    )
    group.addoption(
        "--tape-json-report",
        action="store",
        default=None,
        help="Write a machine-readable JSON token/cost report to the given file path.",
    )
    group.addoption(
        "--tape-html-report",
        action="store",
        default=None,
        help="Write a self-contained HTML trace report of executed cassettes to the given path.",
    )
    group.addoption(
        "--tape-max-cost",
        action="store",
        type=float,
        default=None,
        help="Fail the pytest session if total estimated LLM USD cost exceeds this limit.",
    )
    group.addoption(
        "--tape-max-tokens",
        action="store",
        type=int,
        default=None,
        help="Fail the pytest session if total LLM tokens exceed this limit.",
    )
    group.addoption(
        "--tape-strict-unused",
        action="store_true",
        default=False,
        help="Automatically fail passing replay tests that leave recorded interactions unused.",
    )
    group.addoption(
        "--tape-update-stale",
        action="store",
        type=float,
        default=None,
        help="Re-record ONCE-mode cassettes older than this many days.",
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
        "tape_compress",
        help="Write gzip-compressed .tape.json.gz cassettes by default.",
        type="bool",
        default=False,
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
    parser.addini(
        "tape_strict_unused",
        help="Automatically assert no unused interactions on replay tests.",
        type="bool",
        default=False,
    )
    parser.addini(
        "tape_max_cost",
        help="Fail the pytest session if total estimated LLM USD cost exceeds this limit.",
        default="",
    )
    parser.addini(
        "tape_max_tokens",
        help="Fail the pytest session if total LLM tokens exceed this limit.",
        default="",
    )
    parser.addini(
        "tape_redact_pii",
        help=(
            "Default PII redaction ('true'/'all', or comma-separated presets: "
            "email,phone,ssn,credit_card,ipv4,jwt)."
        ),
        default="",
    )


def _resolve_max_cost(config: pytest.Config) -> float | None:
    cli_val = config.getoption("--tape-max-cost", default=None)
    if cli_val is not None:
        val = float(cli_val)
    else:
        getini = getattr(config, "getini", None)
        ini_val = str(getini("tape_max_cost") or "") if callable(getini) else ""
        raw = os.environ.get("LLM_TAPE_MAX_COST", "").strip() or ini_val.strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError as exc:
            raise pytest.UsageError(f"Invalid tape_max_cost value: {raw!r}") from exc
    if not math.isfinite(val) or val < 0.0:
        raise pytest.UsageError("tape_max_cost / --tape-max-cost must be a finite number >= 0.")
    return val


def _resolve_max_tokens(config: pytest.Config) -> int | None:
    cli_val = config.getoption("--tape-max-tokens", default=None)
    if cli_val is not None:
        val = int(cli_val)
    else:
        getini = getattr(config, "getini", None)
        ini_val = str(getini("tape_max_tokens") or "") if callable(getini) else ""
        raw = os.environ.get("LLM_TAPE_MAX_TOKENS", "").strip() or ini_val.strip()
        if not raw:
            return None
        try:
            val = int(raw)
        except ValueError as exc:
            raise pytest.UsageError(f"Invalid tape_max_tokens value: {raw!r}") from exc
    if val < 0:
        raise pytest.UsageError("tape_max_tokens / --tape-max-tokens must be >= 0.")
    return val


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_REPORT_KEY] = []
    config.stash[_VIOLATIONS_KEY] = []

    _resolve_max_cost(config)
    _resolve_max_tokens(config)
    stale_days = config.getoption("--tape-update-stale", default=None)
    if stale_days is not None and (not math.isfinite(stale_days) or stale_days < 0.0):
        raise pytest.UsageError("--tape-update-stale must be a finite number >= 0.")

    config.addinivalue_line(
        "markers",
        "tape(name=None, mode=None, compress=None, ignore_localhost=None, ignore_hosts=(), "
        "ignore_fields=(), ignore_query_params=(), redact_fields=(), redact_patterns=(), "
        "scrub_body=None, match_on=None, custom_matcher=None, replay_latency_scale=0.0, "
        "allow_playback_repeats=False, save_on_error=False, patch_requests=True): "
        "configure cassette options for the llm_tape fixture.",
    )


def _compute_report_totals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tests": len(entries),
        "calls": sum(int(e["calls"]) for e in entries),
        "input_tokens": sum(int(e["input_tokens"]) for e in entries),
        "cached_input_tokens": sum(int(e["cached_input_tokens"]) for e in entries),
        "output_tokens": sum(int(e["output_tokens"]) for e in entries),
        "total_tokens": sum(int(e["total_tokens"]) for e in entries),
        "estimated_cost_usd": round(sum(float(e["estimated_cost_usd"]) for e in entries), 8),
        "cache_savings_usd": round(sum(float(e.get("cache_savings_usd", 0.0)) for e in entries), 8),
        "latency_ms": round(sum(float(e["latency_ms"]) for e in entries), 2),
    }


def _write_json_report_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """Aggregate llm-tape report entries from pytest-xdist workers on the controller."""
    del error
    workeroutput = getattr(node, "workeroutput", None)
    if not isinstance(workeroutput, dict):
        return
    worker_entries = workeroutput.get("llm_tape_entries")
    if isinstance(worker_entries, list):
        config = getattr(node, "config", None)
        if config is not None:
            entries = config.stash.get(_REPORT_KEY, None)
            if entries is not None:
                entries.extend(worker_entries)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    entries = session.config.stash.get(_REPORT_KEY, [])
    if hasattr(session.config, "workerinput"):
        # Running inside a pytest-xdist worker: forward entries to controller
        workeroutput = getattr(session.config, "workeroutput", None)
        if isinstance(workeroutput, dict):
            workeroutput["llm_tape_entries"] = entries
        return

    totals = _compute_report_totals(entries)

    violations: list[str] = []
    max_tokens = _resolve_max_tokens(session.config)
    max_cost = _resolve_max_cost(session.config)

    if max_tokens is not None and int(totals["total_tokens"]) > max_tokens:
        violations.append(
            f"LLM token budget exceeded: {totals['total_tokens']} > --tape-max-tokens {max_tokens}"
        )
    if max_cost is not None and round(float(totals["estimated_cost_usd"]), 8) > max_cost:
        violations.append(
            f"LLM cost budget exceeded: ${totals['estimated_cost_usd']:.6f} "
            f"> --tape-max-cost ${max_cost:.6f}"
        )

    session.config.stash[_VIOLATIONS_KEY] = violations
    if violations and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

    json_report_path = session.config.getoption("--tape-json-report", default=None)
    if json_report_path:
        _write_json_report_atomic(
            Path(str(json_report_path)),
            {
                "tests": entries,
                "totals": totals,
                "budget_violations": violations,
            },
        )

    html_report_path = session.config.getoption("--tape-html-report", default=None)
    if html_report_path:
        from llm_tape.cassette import Cassette
        from llm_tape.html_report import render_html_report

        cassettes: list[Cassette] = []
        seen_paths: set[str] = set()
        for entry in entries:
            cpath_str = str(entry.get("cassette", ""))
            if not cpath_str or cpath_str in seen_paths:
                continue
            seen_paths.add(cpath_str)
            cpath = Path(cpath_str)
            if cpath.is_file():
                try:
                    cassettes.append(Cassette.load(cpath))
                except Exception:
                    continue
        html_text = render_html_report(cassettes, title="pytest llm-tape Report")
        out_path = Path(str(html_report_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=out_path.parent,
                prefix=f".{out_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(html_text.encode("utf-8"))
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, out_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    del exitstatus
    show_report = bool(config.getoption("--tape-report", default=False))
    entries = config.stash.get(_REPORT_KEY, [])
    violations = config.stash.get(_VIOLATIONS_KEY, [])

    if not show_report and not violations:
        return

    totals = _compute_report_totals(entries)
    terminalreporter.write_sep("=", "llm-tape summary")
    if show_report:
        if entries:
            terminalreporter.write_line(
                f"{'Test':<52} {'Mode':<9} {'Calls':>5} {'Tokens':>8} {'Cached':>7} {'Cost':>11}"
            )
            terminalreporter.write_line("-" * 97)
            for row in entries:
                nodeid = str(row["nodeid"])
                if len(nodeid) > 52:
                    nodeid = "..." + nodeid[-49:]
                terminalreporter.write_line(
                    f"{nodeid:<52} {row['action']:<9} {row['calls']:>5} "
                    f"{row['total_tokens']:>8} {row['cached_input_tokens']:>7} "
                    f"${row['estimated_cost_usd']:>10.6f}"
                )
            terminalreporter.write_line("-" * 97)
            total_label = f"TOTAL ({totals['tests']} tests)"
            terminalreporter.write_line(
                f"{total_label:<62} {totals['calls']:>5} "
                f"{totals['total_tokens']:>8} {totals['cached_input_tokens']:>7} "
                f"${totals['estimated_cost_usd']:>10.6f}"
            )
            if totals.get("cache_savings_usd"):
                terminalreporter.write_line(
                    f"Estimated cache savings: ${totals['cache_savings_usd']:.6f}"
                )
        else:
            terminalreporter.write_line("No llm_tape sessions executed.")

    for msg in violations:
        terminalreporter.write_line(f"FAILED: {msg}", red=True, bold=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        setattr(item, _FAILED_ATTR, bool(rep.failed))


def _resolve_cassette_dir(request: pytest.FixtureRequest) -> Path:
    cli_dir = request.config.getoption("--tape-dir")
    env_dir = os.environ.get("LLM_TAPE_DIR", "").strip() or None
    ini_dir = str(request.config.getini("tape_dir") or "")
    effective_dir = cli_dir or env_dir or ini_dir or None
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
    return base_dir / file_slug


def _parse_redact_pii_setting(raw: str) -> bool | list[str]:
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"false", "0", "no", "off"}:
        return False
    if cleaned in {"true", "1", "yes", "on", "all"}:
        return True
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def _is_cassette_stale(path: Path, max_age_days: float) -> bool:
    if not path.is_file():
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    recorded_dt: datetime | None = None
    try:
        from llm_tape.cassette import Cassette

        cass = Cassette.load(path)
        if cass.recorded_at:
            raw = cass.recorded_at.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            recorded_dt = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        recorded_dt = None
    if recorded_dt is None:
        try:
            recorded_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return False
    return recorded_dt < cutoff


@pytest.fixture
def llm_tape_dir(request: pytest.FixtureRequest) -> Path:
    """Return the resolved cassette directory Path for the current test module."""
    return _resolve_cassette_dir(request)


@pytest.fixture
def llm_tape(request: pytest.FixtureRequest) -> Generator[TapeSession, None, None]:
    """Provide an active TapeSession scoped to the current pytest test."""
    cli_mode = request.config.getoption("--tape-mode")
    cli_compress = bool(request.config.getoption("--tape-compress"))

    ini_mode = str(request.config.getini("tape_mode") or TapeMode.ONCE.value)
    ini_compress = bool(request.config.getini("tape_compress"))
    ini_ignore_localhost = bool(request.config.getini("tape_ignore_localhost"))
    ini_ignore_hosts = list(request.config.getini("tape_ignore_hosts") or [])
    ini_ignore_fields = list(request.config.getini("tape_ignore_fields") or [])
    ini_ignore_query_params = list(request.config.getini("tape_ignore_query_params") or [])

    marker = request.node.get_closest_marker("tape")
    marker_kwargs = marker.kwargs if marker else {}
    marker_name = marker_kwargs.get("name")
    marker_mode = marker_kwargs.get("mode")

    env_mode = os.environ.get("LLM_TAPE_MODE", "").strip() or None
    mode = TapeMode.coerce(marker_mode or cli_mode or env_mode or ini_mode)

    cassette_dir = _resolve_cassette_dir(request)

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
    compress = (
        bool(marker_kwargs["compress"])
        if marker_kwargs.get("compress") is not None
        else (cli_compress or ini_compress)
    )
    ext = ".tape.json.gz" if compress else ".tape.json"
    cassette_path = cassette_dir / f"{test_slug}{ext}"

    update_stale = request.config.getoption("--tape-update-stale", default=None)
    if (
        update_stale is not None
        and mode is TapeMode.ONCE
        and _is_cassette_stale(cassette_path, float(update_stale))
    ):
        mode = TapeMode.RECORD

    ignore_localhost = (
        bool(marker_kwargs["ignore_localhost"])
        if marker_kwargs.get("ignore_localhost") is not None
        else ini_ignore_localhost
    )
    ignore_hosts = (*ini_ignore_hosts, *marker_kwargs.get("ignore_hosts", ()))
    ignore_fields = (*ini_ignore_fields, *marker_kwargs.get("ignore_fields", ()))
    ignore_query_params = (*ini_ignore_query_params, *marker_kwargs.get("ignore_query_params", ()))

    if "redact_pii" in marker_kwargs:
        redact_pii: Any = marker_kwargs["redact_pii"]
    else:
        raw_pii = os.environ.get("LLM_TAPE_REDACT_PII", "").strip() or str(
            request.config.getini("tape_redact_pii") or ""
        )
        redact_pii = _parse_redact_pii_setting(raw_pii)

    session = TapeSession(
        cassette_path,
        mode=mode,
        ignore_localhost=ignore_localhost,
        ignore_hosts=ignore_hosts,
        ignore_fields=ignore_fields,
        ignore_query_params=ignore_query_params,
        ignore_json_array_order=marker_kwargs.get("ignore_json_array_order", ()),
        redact_fields=marker_kwargs.get("redact_fields", ()),
        redact_patterns=marker_kwargs.get("redact_patterns", ()),
        redact_pii=redact_pii,
        redact_url_patterns=marker_kwargs.get("redact_url_patterns", ()),
        scrub_body=marker_kwargs.get("scrub_body"),
        match_on=marker_kwargs.get("match_on") or DEFAULT_MATCH_ON,
        match_headers=marker_kwargs.get("match_headers", ()),
        custom_matcher=marker_kwargs.get("custom_matcher"),
        on_replay=marker_kwargs.get("on_replay"),
        on_record=marker_kwargs.get("on_record"),
        replay_latency_scale=float(marker_kwargs.get("replay_latency_scale", 0.0)),
        allow_playback_repeats=bool(marker_kwargs.get("allow_playback_repeats", False)),
        save_on_error=bool(marker_kwargs.get("save_on_error", False)),
        patch_requests=bool(marker_kwargs.get("patch_requests", True)),
        patch_aiohttp=bool(marker_kwargs.get("patch_aiohttp", True)),
        patch_urllib=bool(marker_kwargs.get("patch_urllib", False)),
    )
    session.__enter__()
    try:
        yield session
    finally:
        test_failed = bool(getattr(request.node, _FAILED_ATTR, False))
        strict_unused = (
            bool(marker_kwargs["strict_unused"])
            if marker_kwargs.get("strict_unused") is not None
            else (
                bool(request.config.getoption("--tape-strict-unused"))
                or bool(request.config.getini("tape_strict_unused"))
            )
        )
        unused_err: Exception | None = None
        if (
            strict_unused
            and not test_failed
            and session._is_replaying
            and not session._is_recording
        ):
            try:
                session.assert_no_unused_interactions()
            except Exception as exc:
                unused_err = exc
        if session._is_recording and session._is_replaying:
            action = "updated" if session._recorded else "replayed"
        elif session._is_recording:
            action = "recorded"
        elif session._is_replaying:
            action = "replayed"
        else:
            action = "none"
        session._finalize(exc_occurred=test_failed or unused_err is not None)
        usage = session.total_usage
        interactions = session.interactions
        entries = request.config.stash.get(_REPORT_KEY, None)
        if entries is not None:
            cache_savings = sum(
                estimate_cache_savings_usd(
                    i.llm.model,
                    i.llm.usage.input_tokens,
                    i.llm.usage.output_tokens,
                    cached_input_tokens=i.llm.usage.cached_input_tokens,
                    cache_creation_tokens=i.llm.usage.cache_creation_tokens,
                )
                for i in interactions
            )
            entries.append(
                {
                    "nodeid": request.node.nodeid,
                    "cassette": str(cassette_path),
                    "action": action,
                    "calls": len(interactions),
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost_usd": round(session.total_cost_usd, 8),
                    "cache_savings_usd": round(cache_savings, 8),
                    "latency_ms": round(sum(i.latency_ms for i in interactions), 2),
                }
            )
        if unused_err is not None:
            raise unused_err
