from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TRACES = ["traceA", "coder", "thinking", "traceB"]


def fixture_path(name: str) -> Path:
    return FIXTURES / f"qwen_{name}_blksz_16.jsonl"


def cap_blocks(n_keys: int, frac: float) -> int:
    """Same as run::cap_blocks in Rust: f64::round (half away from zero), at least one block."""
    x = n_keys * frac
    whole = int(x)
    return max(1, whole + (1 if x - whole >= 0.5 else 0))


@pytest.fixture(scope="session")
def pacer_bin() -> Path:
    p = Path(os.environ.get("PACER_BIN", "target/release/pacer"))
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        pytest.skip(f"pacer binary not found at {p} (build it or set PACER_BIN)")
    return p
