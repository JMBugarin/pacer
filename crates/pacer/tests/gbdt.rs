//! Golden test against LightGBM 4.7.0: tests/fixtures/make_gbdt_fixture.py trains a small
//! regression model, dumps it with `Booster.dump_model()` and records LightGBM's predictions
//! for 200 float32 rows, a fifth of them on or one float32 step from a split threshold.

use pacer::gbdt::Gbdt;
use pacer::policy::{self, Params, Resources};
use pacer::trace::{self, Trace};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn cases() -> (Vec<Vec<f32>>, Vec<f64>) {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("gbdt_cases.json")).unwrap())
            .unwrap();
    assert_eq!(v["lightgbm"], "4.7.0");
    let floats = |key: &str| -> Vec<Vec<f64>> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect()
    };
    let rows: Vec<Vec<f32>> = floats("row_bits")
        .iter()
        .map(|r| r.iter().map(|&b| f32::from_bits(b as u32)).collect())
        .collect();
    for (decimal, exact) in floats("rows").iter().zip(&rows) {
        let cast: Vec<f32> = decimal.iter().map(|&x| x as f32).collect();
        assert_eq!(&cast, exact);
    }
    let pred = v["pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_f64().unwrap())
        .collect();
    (rows, pred)
}

#[test]
fn matches_lightgbm_predictions() {
    let g = Gbdt::load(&fixture("gbdt_model.json")).unwrap();
    assert_eq!((g.n_features, g.n_trees()), (6, 25));
    let (rows, pred) = cases();
    assert_eq!(rows.len(), 200);
    let worst = rows
        .iter()
        .zip(&pred)
        .map(|(x, &want)| (g.predict(x) - want).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-9, "max |rust - lightgbm| = {worst:e}");
}

#[test]
fn batch_prediction_is_bit_identical() {
    let g = Gbdt::load(&fixture("gbdt_model.json")).unwrap();
    let (rows, _) = cases();
    let flat: Vec<f32> = rows.concat();
    let mut out = vec![f64::NAN; rows.len() + 3];
    g.predict_batch(&flat, rows.len(), &mut out);
    for (x, y) in rows.iter().zip(&out) {
        assert_eq!(g.predict(x).to_bits(), y.to_bits());
    }
    assert!(out[rows.len()..].iter().all(|y| y.is_nan()));
}

#[test]
fn split_goes_left_on_equal_and_single_leaf_trees_add() {
    let s = r#"{"feature_names": ["a", "b"], "tree_info": [
        {"tree_structure": {"split_feature": 1, "threshold": 0.5, "decision_type": "<=",
            "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}},
        {"tree_structure": {"leaf_value": 0.25}}]}"#;
    let g = Gbdt::from_json(s).unwrap();
    assert_eq!(g.n_features, 2);
    assert_eq!(g.predict(&[9.0, 0.5]), 1.25);
    assert_eq!(g.predict(&[9.0, 0.500_000_06]), 2.25);
}

#[test]
fn rejects_categorical_splits() {
    let s = r#"{"feature_names": ["a"], "tree_info": [{"tree_structure": {"split_feature": 0,
        "threshold": "1||2", "decision_type": "==",
        "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]}"#;
    assert!(Gbdt::from_json(s).is_err());
}

#[test]
fn rejects_split_feature_out_of_range() {
    let s = r#"{"feature_names": ["a", "b"], "tree_info": [{"tree_structure": {"split_feature": 2,
        "threshold": 0.5, "decision_type": "<=",
        "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]}"#;
    let e = Gbdt::from_json(s)
        .err()
        .expect("split on a missing feature");
    assert!(e.to_string().contains("split_feature 2"), "{e}");
}

fn repo_model(name: &str) -> Arc<Gbdt> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name);
    Arc::new(Gbdt::load(&p).unwrap())
}

fn make_with(policy: &str, model: Arc<Gbdt>) -> anyhow::Result<()> {
    let tr: Trace = trace::load(
        &fixture("qwen_traceA_blksz_16.jsonl"),
        Some(20),
        trace::DEFAULT_TPOT,
    )
    .unwrap();
    let res = Resources {
        models: BTreeMap::from([("0".to_string(), model)]),
        wa_fits: BTreeMap::new(),
    };
    let params = Params::from([("model".to_string(), 0.0)]);
    policy::make(policy, &params, &tr, 64, &res).map(|_| ())
}

#[test]
fn pacer_rejects_model_with_other_features() {
    let g = Arc::new(Gbdt::load(&fixture("gbdt_model.json")).unwrap());
    let e = make_with("pacer", g).expect_err("6-feature model accepted as hazard model");
    let msg = format!("{e:#}");
    assert!(msg.contains("f0") && msg.contains("a_over_gap"), "{msg}");
}

#[test]
fn pacer_accepts_hazard_model() {
    make_with("pacer", repo_model("fast_traceA.json")).unwrap();
}

#[test]
fn pacer_lrb_rejects_hazard_model() {
    assert!(make_with("pacer-lrb", repo_model("fast_traceA.json")).is_err());
    make_with("pacer-lrb", repo_model("lrb_traceA.json")).unwrap();
}

#[test]
fn pacer_rejects_lrb_model() {
    assert!(make_with("pacer", repo_model("lrb_traceA.json")).is_err());
}

#[test]
fn schema_checks_feature_order() {
    let g = Gbdt::from_json(r#"{"feature_names":["b","a"],"tree_info":[]}"#).unwrap();
    assert!(g.check_schema(&["a", "b"]).is_err());
}
