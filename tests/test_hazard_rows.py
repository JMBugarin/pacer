"""`pacer hazard` rows must never depend on events after the training cutoff."""

from __future__ import annotations

import subprocess

import numpy as np
from conftest import fixture_path
from pacer.hazard import FEATURES, N_COLS


def test_rows_end_before_cutoff(pacer_bin, tmp_path):
    out = tmp_path / "rows.npy"
    cutoff = 60.0
    subprocess.run(
        [pacer_bin, "hazard", "--trace", fixture_path("traceA"), "--cutoff", str(cutoff),
         "--rate", "1", "--per-touch", "8", "--out", out],
        check=True,
        capture_output=True,
    )  # fmt: skip
    rows = np.load(out)
    assert rows.shape[1] == N_COLS and len(rows) > 1000
    t = rows[:, 0]
    a = rows[:, 1 + FEATURES.index("a")]
    h = rows[:, 1 + FEATURES.index("h")]
    assert np.all(t + a + h <= cutoff + 1e-3)
    assert np.all((a >= 0) & (h >= 10 - 1e-3) & (h <= 2000 + 1e-3))
    assert set(np.unique(rows[:, -1])) <= {0.0, 1.0}
    assert 0.05 < rows[:, -1].mean() < 0.95
