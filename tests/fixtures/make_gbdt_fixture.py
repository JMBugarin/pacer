"""Regenerates the GBDT golden files read by crates/pacer/tests/gbdt.rs and tests/test_hazard.py:
gbdt_model.json (Booster.dump_model of a small regression model with numerical splits) and
gbdt_cases.json (float32 input rows and LightGBM's raw predictions for them).

    uv run python tests/fixtures/make_gbdt_fixture.py

Rows are float32 because the Rust evaluator takes f32 features; LightGBM sees the same values
widened to float64, so the two must agree to rounding of the tree sum. The rows are also stored
as float32 bit patterns, since serde_json's default float parser can be one ulp off. A fifth of
the rows sit exactly on, or one float32 step either side of, a split threshold, and one feature
is integer-valued so many rows land on LightGBM's bin boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
N_FEATURES = 6
N_ROWS = 200


def features(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    x = rng.normal(scale=scale, size=(n, N_FEATURES))
    x[:, 3] = rng.integers(-1, 6, size=n)
    x[:, 4] = np.where(rng.random(n) < 0.3, 0.0, x[:, 4])
    x[:, 5] = np.exp(rng.normal(size=n)) * 100.0
    return x.astype(np.float32)


def splits(node: dict) -> list[tuple[int, float]]:
    if "leaf_value" in node:
        return []
    assert node["decision_type"] == "<=", node["decision_type"]
    here = [(node["split_feature"], node["threshold"])]
    return here + splits(node["left_child"]) + splits(node["right_child"])


def main() -> None:
    assert lgb.__version__ == "4.7.0", lgb.__version__
    rng = np.random.default_rng(20261101)
    x = features(rng, 3000, 1.0)
    y = (
        np.sin(x[:, 0])
        + x[:, 1] * x[:, 2]
        + 0.3 * x[:, 3]
        - 0.5 * (x[:, 4] == 0)
        + np.log1p(x[:, 5])
        + 0.1 * rng.normal(size=len(x))
    )
    params = {
        "objective": "regression",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "seed": 7,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    names = [f"f{i}" for i in range(N_FEATURES)]
    booster = lgb.train(params, lgb.Dataset(x, y, feature_name=names), num_boost_round=25)
    model = booster.dump_model()
    cuts = [s for t in model["tree_info"] for s in splits(t["tree_structure"])]

    rows = features(rng, N_ROWS, 1.5)
    for i in range(N_ROWS // 5):
        f, thr = cuts[rng.integers(len(cuts))]
        v = np.float32(thr)
        rows[i, f] = [v, np.nextafter(v, np.float32(-np.inf)), np.nextafter(v, np.float32(np.inf))][
            i % 3
        ]
    pred = booster.predict(rows.astype(np.float64), raw_score=True)

    (HERE / "gbdt_model.json").write_text(json.dumps(model) + "\n")
    cases = {
        "lightgbm": lgb.__version__,
        "rows": rows.tolist(),
        "row_bits": rows.view(np.uint32).tolist(),
        "pred": pred.tolist(),
    }
    (HERE / "gbdt_cases.json").write_text(json.dumps(cases) + "\n")
    print(f"{booster.num_trees()} trees, {len(cuts)} splits, {N_ROWS} rows")


if __name__ == "__main__":
    main()
