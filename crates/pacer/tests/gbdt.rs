//! Golden test against LightGBM 4.7.0: tests/fixtures/make_gbdt_fixture.py trains a small
//! regression model, dumps it with `Booster.dump_model()` and records LightGBM's predictions
//! for 200 float32 rows, a fifth of them on or one float32 step from a split threshold.

use pacer::gbdt::Gbdt;
use serde_json::Value;
use std::path::PathBuf;

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
