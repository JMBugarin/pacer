//! Workload-aware eviction from "KVCache Cache in the Wild" (ATC'25): each workload
//! (request type, turn) gets an exponential reuse-time model fitted on a training window;
//! the victim is the least likely to be reused among the LRU block of every workload, with
//! deeper blocks going first on ties. The paper's unspecified "lifespan regulator" is omitted.

use super::heap::LazyHeap;
use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::Trace;

pub const MAX_TURN: u32 = 5;
pub const N_WORKLOADS: usize = 8 * MAX_TURN as usize;

pub fn workload(tr: &Trace, req: u32) -> usize {
    let r = &tr.reqs[req as usize];
    r.rtype as usize * MAX_TURN as usize + (r.turn.clamp(1, MAX_TURN) - 1) as usize
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub p_reuse: Vec<f64>,
    pub tau: Vec<f64>,
}

impl Fit {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        #[derive(serde::Deserialize)]
        struct F {
            p_reuse: Vec<f64>,
            tau: Vec<f64>,
        }
        let f: F = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        anyhow::ensure!(
            f.p_reuse.len() == N_WORKLOADS && f.tau.len() == N_WORKLOADS,
            "expected {N_WORKLOADS} workloads"
        );
        Ok(Fit {
            p_reuse: f.p_reuse,
            tau: f.tau,
        })
    }

    pub fn from_trace(tr: &Trace, fit_end: f64) -> Self {
        let mut reused = vec![0f64; N_WORKLOADS];
        let mut total = vec![0f64; N_WORKLOADS];
        let mut dt = vec![0f64; N_WORKLOADS];
        for (i, r) in tr.reqs.iter().enumerate() {
            if r.t >= fit_end {
                break;
            }
            let w = workload(tr, i as u32);
            let from = tr.first_after_completion(i as u32);
            let mut account = |k: u32, t: f64| {
                total[w] += 1.0;
                let j = tr.next_use(k, from) as usize;
                if j < tr.reqs.len() && tr.reqs[j].t < fit_end {
                    reused[w] += 1.0;
                    dt[w] += tr.reqs[j].t - t;
                }
            };
            for &k in r.input(&tr.keys) {
                account(k, r.done);
            }
            for &k in r.output(&tr.keys) {
                account(k, r.done);
            }
        }
        let gr: f64 = reused.iter().sum();
        let gt: f64 = total.iter().sum();
        let gd: f64 = dt.iter().sum();
        let g_tau = if gr > 0.0 { gd / gr } else { 300.0 };
        let mut p_reuse = vec![0.0; N_WORKLOADS];
        let mut tau = vec![0.0; N_WORKLOADS];
        for w in 0..N_WORKLOADS {
            p_reuse[w] = (reused[w] + gr / gt.max(1.0)) / (total[w] + 1.0);
            tau[w] = if reused[w] > 0.0 {
                (dt[w] + g_tau) / (reused[w] + 1.0)
            } else {
                g_tau
            };
        }
        Fit { p_reuse, tau }
    }

    /// P(reused later | not reused for `age` seconds) under the workload's exponential model.
    pub fn prob(&self, w: usize, age: f64) -> f64 {
        let x = self.p_reuse[w] * (-age.max(0.0) / self.tau[w]).exp();
        x / (x + 1.0 - self.p_reuse[w]).max(1e-12)
    }
}

pub struct WorkloadAware {
    fit: std::sync::Arc<Fit>,
    heaps: Vec<LazyHeap>,
    wl: Vec<u8>,
    seq: u64,
}

impl WorkloadAware {
    pub fn new(n: u32, fit: std::sync::Arc<Fit>) -> Self {
        WorkloadAware {
            fit,
            heaps: (0..N_WORKLOADS).map(|_| LazyHeap::new(n)).collect(),
            wl: vec![0; n as usize],
            seq: 0,
        }
    }

    fn touch(&mut self, c: &Core, tr: &Trace, k: u32) {
        let w = workload(tr, c.owner[k as usize]);
        let old = self.wl[k as usize] as usize;
        if old != w {
            self.heaps[old].invalidate(k);
        }
        self.wl[k as usize] = w as u8;
        self.seq += 1;
        self.heaps[w].set(k, (self.seq, 0, 0));
    }
}

impl Policy for WorkloadAware {
    fn on_insert(&mut self, c: &mut Core, tr: &Trace, k: u32, _p: Pos) {
        self.touch(c, tr, k)
    }
    fn on_access(&mut self, c: &mut Core, tr: &Trace, k: u32, _p: Pos) {
        self.touch(c, tr, k)
    }
    fn on_release(&mut self, c: &mut Core, tr: &Trace, k: u32, _p: Pos) {
        self.touch(c, tr, k)
    }
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.heaps[self.wl[k as usize] as usize].repush(k)
    }
    fn victim(&mut self, c: &mut Core, tr: &Trace, pos: Pos) -> Option<u32> {
        let mut best: Option<(f64, u32, u32, usize)> = None;
        for w in 0..N_WORKLOADS {
            if let Some(k) = self.heaps[w].peek_valid(c) {
                let pr = self.fit.prob(w, pos.t - c.last_touch[k as usize]);
                let d = tr.key_depth[k as usize];
                let better = match best {
                    None => true,
                    Some((bp, bd, bk, _)) => (pr, u32::MAX - d, k) < (bp, u32::MAX - bd, bk),
                };
                if better {
                    best = Some((pr, d, k, w));
                }
            }
        }
        let (_, _, k, w) = best?;
        self.heaps[w].pop();
        Some(k)
    }
}
