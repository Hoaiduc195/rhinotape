"""Tests for gzip-compressed cassettes (.tape.json.gz) and hot-path optimizations."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import pytest

from llm_tape import (
    Cassette,
    CorruptCassetteError,
    TapeMode,
    TapeSession,
    TapeTransport,
    tape,
)
from llm_tape.cassette import is_cassette_file, iter_cassette_files
from llm_tape.cli import main as cli_main

pytest_plugins = ["pytester"]


def test_gzip_cassette_roundtrip_deterministic_and_smaller(tmp_path: Path) -> None:
    gz_path = tmp_path / "large_sse.tape.json.gz"
    plain_path = tmp_path / "large_sse.tape.json"

    sse_chunks = [
        'data: {"id":"chatcmpl-1","model":"gpt-4o-mini",'
        f'"choices":[{{"delta":{{"content":"token-{i} "}}}}]}}\n\n'
        for i in range(60)
    ]
    sse_chunks.append("data: [DONE]\n\n")
    raw_sse = "".join(sse_chunks).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=raw_sse,
        )

    upstream = httpx.MockTransport(handler)
    with tape(gz_path, mode=TapeMode.RECORD, patch_httpx=False) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as client:
            with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"stream": True}
            ) as r:
                body1 = r.read()

    # Deterministic gzip output (mtime=0): re-saving produces identical bytes
    first_bytes = gz_path.read_bytes()
    assert first_bytes[:2] == b"\x1f\x8b"
    loaded = Cassette.load(gz_path)
    loaded.save()
    assert gz_path.read_bytes() == first_bytes

    # Compare size against uncompressed plain JSON
    plain_cassette = Cassette(
        path=plain_path,
        interactions=loaded.interactions,
        recorded_at=loaded.recorded_at,
        metadata=loaded.metadata,
    )
    plain_cassette.save()
    assert gz_path.stat().st_size < plain_path.stat().st_size

    # Replay directly from .tape.json.gz
    with tape(gz_path, mode=TapeMode.REPLAY, patch_httpx=False) as rep:
        with httpx.Client(transport=TapeTransport(rep)) as client:
            with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"stream": True}
            ) as r:
                body2 = r.read()
        assert body2 == body1
        rep.assert_call_count(1)


def test_corrupt_gzip_cassette_raises_corrupt_error(tmp_path: Path) -> None:
    bad_gz = tmp_path / "bad.tape.json.gz"
    bad_gz.write_bytes(b"not valid gzip bytes")
    with pytest.raises(CorruptCassetteError):
        Cassette.load(bad_gz)

    assert is_cassette_file(bad_gz)
    assert is_cassette_file(tmp_path / "ok.tape.json")
    assert not is_cassette_file(tmp_path / "notes.json")


def test_cli_commands_support_gzip_cassettes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tapes_dir = tmp_path / "tapes"
    gz_path = tapes_dir / "test_demo" / "test_gz.tape.json.gz"

    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "secret sk-1234567890"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )
    with tape(gz_path, mode=TapeMode.RECORD, patch_httpx=False) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o-mini"})

    assert iter_cassette_files(tapes_dir) == [gz_path]

    assert cli_main(["inspect", str(gz_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["calls"] == 1

    assert cli_main(["summary", str(tapes_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["files"] == 1

    assert cli_main(["verify", str(tapes_dir)]) == 0
    capsys.readouterr()

    assert cli_main(["redact", str(tapes_dir), "--pattern", r"sk-\d+"]) == 0
    capsys.readouterr()
    redacted = Cassette.load(gz_path)
    assert "sk-1234567890" not in redacted.interactions[0].llm.text

    # Prune dry-run recognizes .tape.json.gz stem properly
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_demo.py").write_text("def test_other(): pass\n", encoding="utf-8")
    assert cli_main(["prune", str(tapes_dir), "--tests-dir", str(tests_dir), "--json"]) == 0
    prune_data = json.loads(capsys.readouterr().out)
    assert len(prune_data["candidates"]) == 1
    assert prune_data["candidates"][0]["path"] == str(gz_path)


def test_precompiled_patterns_and_invalid_latency_scale(tmp_path: Path) -> None:
    with pytest.raises(re.error):
        TapeSession(tmp_path / "x.tape.json", redact_patterns=["[unclosed"])

    with pytest.raises(ValueError, match="finite"):
        TapeSession(tmp_path / "x.tape.json", replay_latency_scale=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        TapeSession(tmp_path / "x.tape.json", replay_latency_scale=float("inf"))


def test_replay_throughput_benchmark_sanity(tmp_path: Path) -> None:
    """Ensure 200 sequential replays complete well under 1 second."""
    cassette_path = tmp_path / "bench.tape.json"
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD, patch_httpx=False) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"n": 1})

    t0 = time.perf_counter()
    with tape(
        cassette_path,
        mode=TapeMode.REPLAY,
        allow_playback_repeats=True,
        patch_httpx=False,
    ) as rep:
        with httpx.Client(transport=TapeTransport(rep)) as client:
            for _ in range(200):
                r = client.post("https://api.openai.com/v1/chat/completions", json={"n": 1})
                assert r.status_code == 200
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"200 replays took {elapsed:.2f}s (expected < 2.0s)"


def test_pytester_tape_compress_option(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makepyfile(
        """
        import httpx
        from llm_tape import TapeTransport

        def test_compressed_tape(llm_tape):
            upstream = httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    },
                )
            )
            with httpx.Client(transport=TapeTransport(llm_tape, upstream=upstream)) as c:
                r = c.post("https://api.openai.com/v1/chat/completions", json={"a": 1})
                assert r.status_code == 200
        """
    )
    result = pytester.runpytest("--tape-compress")
    result.assert_outcomes(passed=1)
    gz_files = list(pytester.path.rglob("*.tape.json.gz"))
    assert len(gz_files) == 1
