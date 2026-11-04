mod arc;
mod heap;
mod qd;
mod s3fifo;
pub mod sampled;
mod sieve;
mod wa;

use crate::gbdt::Gbdt;
use crate::sim::{Core, Pos};
use crate::trace::Trace;
use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::sync::Arc;

pub use arc::Arc as ArcPolicy;
pub use heap::{Belady, Fifo, Gdsf, Lfu, Lru};
pub use qd::LruQd;
pub use s3fifo::S3Fifo;
pub use sampled::{Sampled, Scorer};
pub use sieve::Sieve;
pub use wa::{Fit as WaFit, WorkloadAware};

/// Eviction policy over a prefix-closed block cache. The simulator only accepts victims that
/// are evictable (cached, unpinned, no cached children).
pub trait Policy: Send {
    fn on_insert(&mut self, core: &mut Core, tr: &Trace, k: u32, pos: Pos);
    fn on_access(&mut self, core: &mut Core, tr: &Trace, k: u32, pos: Pos);
    fn on_release(&mut self, _core: &mut Core, _tr: &Trace, _k: u32, _pos: Pos) {}
    fn on_evictable(&mut self, _core: &Core, _tr: &Trace, _k: u32) {}
    fn victim(&mut self, core: &mut Core, tr: &Trace, pos: Pos) -> Option<u32>;
    fn on_evict(&mut self, _core: &Core, _tr: &Trace, _k: u32) {}
}

pub type Params = BTreeMap<String, f64>;

pub fn param(p: &Params, k: &str, default: f64) -> f64 {
    p.get(k).copied().unwrap_or(default)
}

pub struct Resources {
    pub models: BTreeMap<String, Arc<Gbdt>>,
    /// ATC'25 workload parameters per trace name, fitted in Python with censoring handled;
    /// without one (or with `naive=1`), the policy uses a simple in-Rust fit up to `fit_end`.
    pub wa_fits: BTreeMap<String, Arc<wa::Fit>>,
}

pub fn make(
    name: &str,
    params: &Params,
    tr: &Trace,
    cap: u64,
    res: &Resources,
) -> Result<Box<dyn Policy>> {
    let n = tr.n_keys;
    let seed = param(params, "seed", 1.0) as u64;
    Ok(match name {
        "lru" => Box::new(Lru::new(n)),
        "fifo" => Box::new(Fifo::new(n)),
        "lfu" => Box::new(Lfu::new(n)),
        "belady" => Box::new(Belady::new(n)),
        "gdsf" => Box::new(Gdsf::new(
            n,
            param(params, "c0", 1.0),
            param(params, "c1", 1.0 / 64.0),
        )),
        "sieve" => Box::new(Sieve::new(n)),
        "s3fifo" => Box::new(S3Fifo::new(
            n,
            cap,
            param(params, "small", 0.1),
            param(params, "threshold", 2.0) as u8,
        )),
        "arc" => Box::new(ArcPolicy::new(n, cap, param(params, "release", 0.0) > 0.5)),
        "lruqd" => Box::new(LruQd::new(n, cap, param(params, "small", 0.1))),
        "wa" => {
            let fit = match res.wa_fits.get(&tr.name) {
                Some(f) if param(params, "naive", 0.0) < 0.5 => f.clone(),
                _ => Arc::new(wa::Fit::from_trace(tr, param(params, "fit_end", 3600.0))),
            };
            Box::new(WorkloadAware::new(n, fit))
        }
        "sampled-lru" | "sampled-belady" | "oracle-dead" | "oracle-within" | "pacer"
        | "pacer-lrb" => {
            let scorer = match name {
                "sampled-lru" => Scorer::Age,
                "sampled-belady" => Scorer::Oracle,
                "oracle-dead" => Scorer::Dead,
                "oracle-within" => Scorer::Within(param(params, "horizon", 120.0)),
                _ => {
                    let m = params
                        .get("model")
                        .map(|v| format!("{}", *v as i64))
                        .unwrap_or_else(|| "0".into());
                    let g = match res.models.get(&m) {
                        Some(g) => g.clone(),
                        None => bail!("{name} needs a model (model={m}); load one with --model"),
                    };
                    if name == "pacer" {
                        Scorer::Hazard(g, param(params, "mult", 1.0))
                    } else {
                        Scorer::Learned(g)
                    }
                }
            };
            Box::new(
                Sampled::new(
                    scorer,
                    param(params, "k", 64.0) as usize,
                    param(params, "follow", 1.0) > 0.5,
                    seed,
                )
                .with_reuse(param(params, "reuse", 1.0) as usize)
                .with_cost(param(params, "cost_alpha", 0.0)),
            )
        }
        _ => bail!("unknown policy {name}"),
    })
}
