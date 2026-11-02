//! Evaluator for LightGBM models exported with `Booster.dump_model()`. Numerical `<=` splits
//! only; trees are flattened into structure-of-arrays form so prediction is a few hundred
//! predictable loads per sample.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

pub struct Gbdt {
    pub n_features: usize,
    pub feature_names: Vec<String>,
    roots: Vec<i32>,
    feature: Vec<u16>,
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf_value: Vec<f64>,
}

impl Gbdt {
    pub fn load(path: &Path) -> Result<Self> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&s)
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s)?;
        let names: Vec<String> = v["feature_names"]
            .as_array()
            .context("feature_names")?
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut g = Gbdt {
            n_features: names.len(),
            feature_names: names,
            roots: Vec::new(),
            feature: Vec::new(),
            threshold: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            leaf_value: Vec::new(),
        };
        for t in v["tree_info"].as_array().context("tree_info")? {
            let root = g.add(&t["tree_structure"])?;
            g.roots.push(root);
        }
        Ok(g)
    }

    /// Returns a node id (>= 0) or an encoded leaf (-1 - leaf_id).
    fn add(&mut self, n: &Value) -> Result<i32> {
        if let Some(lv) = n.get("leaf_value") {
            self.leaf_value.push(lv.as_f64().context("leaf_value")?);
            return Ok(-(self.leaf_value.len() as i32));
        }
        if n["decision_type"].as_str() != Some("<=") {
            bail!("only numerical <= splits are supported");
        }
        let id = self.feature.len();
        self.feature
            .push(n["split_feature"].as_u64().context("split_feature")? as u16);
        self.threshold
            .push(n["threshold"].as_f64().context("threshold")?);
        self.left.push(0);
        self.right.push(0);
        let l = self.add(&n["left_child"])?;
        let r = self.add(&n["right_child"])?;
        self.left[id] = l;
        self.right[id] = r;
        Ok(id as i32)
    }

    #[inline]
    pub fn predict(&self, x: &[f32]) -> f64 {
        let mut s = 0.0;
        for &root in &self.roots {
            let mut n = root;
            while n >= 0 {
                let i = n as usize;
                n = if (x[self.feature[i] as usize] as f64) <= self.threshold[i] {
                    self.left[i]
                } else {
                    self.right[i]
                };
            }
            s += self.leaf_value[(-n - 1) as usize];
        }
        s
    }

    /// Predicts `n` rows stored back to back with stride `n_features`. Trees are the outer
    /// loop so each tree's nodes stay in cache across rows; per-row sums are accumulated in the
    /// same tree order as `predict`, so results are bit-identical.
    pub fn predict_batch(&self, xs: &[f32], n: usize, out: &mut [f64]) {
        let d = self.n_features;
        out[..n].fill(0.0);
        for &root in &self.roots {
            for (j, o) in out[..n].iter_mut().enumerate() {
                let x = &xs[j * d..(j + 1) * d];
                let mut node = root;
                while node >= 0 {
                    let i = node as usize;
                    node = if (x[self.feature[i] as usize] as f64) <= self.threshold[i] {
                        self.left[i]
                    } else {
                        self.right[i]
                    };
                }
                *o += self.leaf_value[(-node - 1) as usize];
            }
        }
    }

    pub fn n_trees(&self) -> usize {
        self.roots.len()
    }
}
