"""Targeted coverage tests for stores.py, matchers.py, and chaos.py (Wave 109)."""

from __future__ import annotations

from pathlib import Path

from rhinotape import (
    FileCassetteStore,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    chain_replay_hooks,
    corrupt_json_body,
    fail_nth,
    inject_latency,
    match_header_equals,
    match_json_regex,
    match_json_subset,
    truncate_stream,
)


def _sample_interaction() -> Interaction:
    return Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"X-Foo": "bar"},
            body={"items": ["a"]},
            fingerprint="fp1",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"Content-Type": "application/json", "content-type": "text/plain"},
            body={"ok": True},
        ),
        latency_ms=10.0,
    )


def test_file_store_missing_root_and_matcher_edges(tmp_path: Path) -> None:
    store = FileCassetteStore(tmp_path / "nonexistent_dir")
    assert store.list_keys() == []

    req_list = RecordedRequest("POST", "https://x/a", {}, {"items": ["a"]}, "f1")
    req_str = RecordedRequest("POST", "https://x/a", {}, "plain", "f2")
    # Out-of-bounds list index & missing regex path
    assert not match_json_regex("items.5", r"a")(req_list, req_list)
    assert not match_json_subset()(req_list, req_str)
    # Missing header returns None == None when value is None, False when value expected
    assert match_header_equals("x-missing")(req_list, req_str)
    assert not match_header_equals("x-missing", "val")(req_list, req_str)


def test_chaos_edge_branches() -> None:
    inter = _sample_interaction()

    # Non-matching nth returns None
    assert inject_latency(50.0, nth=2)(inter, 0) is None
    assert corrupt_json_body(nth=2)(inter, 0) is None
    assert truncate_stream(1, nth=2)(inter, 0) is None
    # truncate_stream on non-stream interaction returns None
    assert truncate_stream(1)(inter, 0) is None

    # chain_replay_hooks returns None when no hook modifies interaction
    noop_chain = chain_replay_hooks(fail_nth(5), inject_latency(10.0, nth=5))
    assert noop_chain(inter, 0) is None

    # Explicit content-type override + case-insensitive duplicate merging in _merge_fault_headers
    res = fail_nth(
        0,
        status_code=500,
        body={"err": 1},
        headers={"Content-Type": "application/problem+json", "X-Custom": "1"},
    )(inter, 0)
    assert res is not None
    assert res.response.headers["Content-Type"] == "application/problem+json"
    assert res.response.headers["X-Custom"] == "1"
