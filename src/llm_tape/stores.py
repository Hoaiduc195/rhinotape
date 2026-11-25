"""Pluggable CassetteStore protocol and built-in Memory/File store implementations."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from llm_tape.cassette import Cassette
from llm_tape.models import TapeNotFoundError


@runtime_checkable
class CassetteStore(Protocol):
    """Protocol for pluggable cassette persistence backends (memory, filesystem, S3, Redis)."""

    def exists(self, key: str) -> bool: ...

    def load(self, key: str) -> Cassette: ...

    def save(self, key: str, cassette: Cassette) -> None: ...

    def delete(self, key: str) -> bool: ...

    def list_keys(self) -> list[str]: ...


class MemoryCassetteStore:
    """Thread-safe in-memory CassetteStore ideal for ephemeral tests and benchmarks."""

    def __init__(self, *, compress: bool = False) -> None:
        self._compress = compress
        self._lock = threading.Lock()
        self._data: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def load(self, key: str) -> Cassette:
        with self._lock:
            raw = self._data.get(key)
        if raw is None:
            raise TapeNotFoundError(f"Cassette key not found in MemoryCassetteStore: {key!r}")
        return Cassette.from_bytes(raw, path=Path(key))

    def save(self, key: str, cassette: Cassette) -> None:
        raw = cassette.to_bytes(compress=self._compress)
        with self._lock:
            self._data[key] = raw

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
