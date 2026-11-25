"""Public API surface (__all__) and PEP 561 py.typed audit tests (Wave 111)."""

from __future__ import annotations

from pathlib import Path

import llm_tape


def test_public_all_exports_sorted_and_present() -> None:
    exported = llm_tape.__all__
    assert len(exported) == len(set(exported)), "Duplicate entries in llm_tape.__all__"
    assert exported == sorted(exported), "llm_tape.__all__ should remain sorted alphabetically"
    for symbol in exported:
        assert hasattr(llm_tape, symbol), f"Missing exported symbol: {symbol}"


def test_py_typed_marker_present() -> None:
    pkg_dir = Path(llm_tape.__file__).parent
    assert (pkg_dir / "py.typed").is_file()
