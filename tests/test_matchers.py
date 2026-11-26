"""Tests for match presets, ignore_json_array_order, and composable RequestMatchers."""

from __future__ import annotations

from pathlib import Path

import httpx

from rhinotape import (
    MATCH_PATH_AND_BODY,
    MATCH_RELAXED,
    MATCH_STRICT,
    RecordedRequest,
    TapeMode,
    match_all,
    match_any,
    match_header_equals,
    match_json_regex,
    match_json_subset,
    match_method,
    match_path,
    normalize_json_array_order,
    tape,
)


def _req(
    *,
    method: str = "POST",
    url: str = "https://api.openai.com/v1/chat/completions",
    headers: dict[str, str] | None = None,
    body: object = None,
) -> RecordedRequest:
    return RecordedRequest(
        method=method,
        url=url,
        headers=headers or {},
        body=body,
        fingerprint="fp",
    )


def test_match_presets_and_normalize_json_array_order(tmp_path: Path) -> None:
    assert MATCH_STRICT == ("method", "url", "body", "headers")
    assert MATCH_RELAXED == ("method", "path")
    assert MATCH_PATH_AND_BODY == ("method", "path", "body")

    body_a = {"model": "gpt-4o", "tools": [{"name": "b"}, {"name": "a"}]}
    body_b = {"model": "gpt-4o", "tools": [{"name": "a"}, {"name": "b"}]}
    assert normalize_json_array_order(body_a, ["tools"]) == normalize_json_array_order(
        body_b, ["tools"]
    )
    # Original input is not mutated
    assert body_a["tools"] == [{"name": "b"}, {"name": "a"}]

    cassette_path = tmp_path / "tools_order.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD, ignore_json_array_order=["tools"]):
        with httpx.Client(transport=transport) as client:
            client.post("https://api.openai.com/v1/chat/completions", json=body_a)

    with tape(cassette_path, mode=TapeMode.REPLAY, ignore_json_array_order=["tools"]) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", json=body_b)
            assert resp.status_code == 200
            assert session.call_count == 1


def test_composable_matchers_unit() -> None:
    inc = _req(
        headers={"X-Tenant": "Acme", "Content-Type": "application/json"},
        body={
            "model": "gpt-4o",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Order #12345 status?"}],
        },
    )
    rec = _req(
        url="https://proxy.local/v1/chat/completions",
        headers={"x-tenant": "acme"},
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Order #99999 status?"}],
        },
    )

    assert match_method()(rec, inc)
    assert match_path()(rec, inc)
    assert match_json_subset(["model"])(rec, inc)
    assert not match_json_subset(["model", "temperature"])(rec, inc)
    assert match_json_regex("messages.0.content", r"Order #\d+")(rec, inc)
    assert not match_json_regex("messages.0.content", r"Invoice #\d+")(rec, inc)
    assert match_header_equals("x-tenant")(rec, inc)
    assert match_header_equals("x-tenant", "ACME")(rec, inc)
    assert not match_header_equals("x-tenant", "ACME", case_sensitive=True)(rec, inc)

    combined = match_all(
        match_method(),
        match_path(),
        match_json_subset(["model"]),
        match_json_regex("messages.0.content", r"Order #\d+"),
        match_header_equals("x-tenant"),
    )
    assert combined(rec, inc)
    assert match_any(match_json_subset(["temperature"]), match_path())(rec, inc)
    assert not match_any(match_json_subset(["temperature"]))(rec, inc)


def test_match_json_subset_end_to_end_in_tape_session(tmp_path: Path) -> None:
    cassette_path = tmp_path / "subset_e2e.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "subset ok"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini"},
            )

    # Incoming request adds extra fields (trace_id, temperature) not present in recorded subset
    with tape(cassette_path, mode=TapeMode.REPLAY, custom_matcher=match_json_subset()) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "trace_id": "abc-123", "temperature": 0.2},
            )
            assert resp.status_code == 200
            assert session.call_count == 1


def test_composable_matchers_with_tape_session(tmp_path: Path) -> None:
    cassette_path = tmp_path / "matcher.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "matched"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "ticket-1"}],
                },
            )

    matcher = match_all(
        match_method(),
        match_path(),
        match_json_subset(["model"]),
        match_json_regex("messages.0.content", r"^ticket-\d+$"),
    )
    with tape(cassette_path, mode=TapeMode.REPLAY, custom_matcher=matcher) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "ticket-42"}],
                },
            )
            assert resp.status_code == 200
            assert session.call_count == 1
