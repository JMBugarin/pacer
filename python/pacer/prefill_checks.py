"""Prefill diagnostics independent of the optional MLX runtime."""

from collections.abc import Sequence

import numpy as np


def check_summary(checks: Sequence[dict]) -> dict:
    if not checks:
        return {"n": 0}
    d = np.array([c["max_abs"] for c in checks])
    eq = np.array([c["argmax_equal"] for c in checks])
    margin = np.array([c["ref_margin"] for c in checks])
    ctl = [c["control_max_abs"] for c in checks if "control_max_abs" in c]
    q = np.percentile(d, [50, 90, 99, 100]).tolist()
    return {
        "n": len(checks),
        "protocol_version": 2,
        "clear_margin_n": int((margin > 2 * d).sum()),
        "max_abs_diff": dict(
            zip(["p50", "p90", "p99", "max"], q, strict=True), mean=float(d.mean())
        ),
        "argmax_equal_frac": float(eq.mean()),
        "argmax_mismatch_ref_margins": margin[~eq].tolist(),
        "argmax_equal_frac_clear_margin": float(eq[margin > 2 * d].mean())
        if (margin > 2 * d).any()
        else None,
        "ref_absmax_p50": float(np.median([c["ref_absmax"] for c in checks])),
        "control_max_abs_p50": float(np.median(ctl)) if ctl else None,
        "control_n": len(ctl),
        "hit_frac_mean": float(np.mean([c["hit_blocks"] / c["n_blocks"] for c in checks])),
    }
