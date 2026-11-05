//! Sampled eviction (as in LRB): draw K evictable blocks, evict the one whose score is largest.
//! The scorer decides what "largest" means: recency, the true next use (an oracle), or a GBDT
//! prediction of time to next use. With `follow`, after evicting a block the policy evicts its
//! parent next, without a new sample, when the two have identical history (same owner and
//! access count), which amortizes model calls along a session's private suffix.

use super::Policy;
use crate::features::{self, Global, N_FEAT, N_HAZ};
use crate::gbdt::Gbdt;
use crate::sim::{Core, Pos};
use crate::trace::{Trace, NONE};
use std::sync::Arc;

#[derive(Clone)]
pub enum Scorer {
    Age,
    Oracle,
    Dead,
    Within(f64),
    Learned(Arc<Gbdt>),
    /// Session-return hazard model: evict the candidate least likely to be used within
    /// `mult` times the cache's recent eviction age.
    Hazard(Arc<Gbdt>, f64),
}

pub struct Logger {
    pub rate: f64,
    pub window: (f64, f64),
    pub rows: Vec<f32>,
    pub n_rows: usize,
    decision: u64,
}

impl Logger {
    pub fn new(rate: f64, window: (f64, f64)) -> Self {
        Logger {
            rate,
            window,
            rows: Vec::new(),
            n_rows: 0,
            decision: 0,
        }
    }
    /// Columns: decision id, time, features..., label ln(1 + ttnu).
    pub const N_COLS: usize = N_FEAT + 3;
}

pub struct Sampled {
    scorer: Scorer,
    k: usize,
    follow: bool,
    rng: u64,
    pending: u32,
    pending_seen: (u32, u32, f64),
    glob: Global,
    feat: Vec<[f32; N_FEAT]>,
    cand: Vec<u32>,
    hbuf: Vec<f32>,
    hout: Vec<f64>,
    cost_alpha: f64,
    reuse: usize,
    spare: Vec<u32>,
    spare_at: (f64, u8, u32),
    order: Vec<usize>,
    pub logger: Option<Logger>,
    pub model_calls: u64,
}

impl Sampled {
    pub fn new(scorer: Scorer, k: usize, follow: bool, seed: u64) -> Self {
        Sampled {
            scorer,
            k: k.max(1),
            follow,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            pending: NONE,
            pending_seen: (NONE, 0, 0.0),
            glob: Global {
                evict_age_ewma: 60.0,
            },
            feat: vec![[0.0; N_FEAT]; k.max(1)],
            cand: Vec::with_capacity(k.max(1)),
            hbuf: vec![0.0; k.max(1) * N_HAZ],
            hout: vec![0.0; k.max(1)],
            cost_alpha: 0.0,
            reuse: 1,
            spare: Vec::new(),
            spare_at: (f64::NAN, 0, 0),
            order: Vec::with_capacity(k.max(1)),
            logger: None,
            model_calls: 0,
        }
    }

    /// Lets one scored sample supply up to `reuse` victims within the same eviction burst
    /// (the evictions triggered by one arrival or completion), best score first.
    /// Scores candidates by expected saved prefill instead of reuse probability: P(reuse) times
    /// the block's relative recompute cost 1 + alpha * depth. alpha = 0 keeps plain reuse.
    pub fn with_cost(mut self, alpha: f64) -> Self {
        self.cost_alpha = alpha.max(0.0);
        self
    }

    pub fn with_reuse(mut self, reuse: usize) -> Self {
        self.reuse = reuse.max(1);
        self
    }

    #[inline]
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn score(&mut self, c: &Core, tr: &Trace, j: usize, pos: Pos) -> f64 {
        let k = self.cand[j];
        match &self.scorer {
            Scorer::Age => pos.t - c.last_touch[k as usize],
            Scorer::Oracle => features::ttnu(tr, k, pos.t, pos.next),
            Scorer::Dead => {
                let age = pos.t - c.last_touch[k as usize];
                if tr.next_use(k, pos.next) as usize >= tr.reqs.len() {
                    1e12 + age
                } else {
                    age
                }
            }
            Scorer::Within(h) => {
                let age = pos.t - c.last_touch[k as usize];
                if features::ttnu(tr, k, pos.t, pos.next) > *h {
                    1e12 + age
                } else {
                    age
                }
            }
            Scorer::Learned(m) => {
                self.model_calls += 1;
                m.predict(&self.feat[j])
            }
            Scorer::Hazard(m, mult) => {
                self.model_calls += 1;
                let a = pos.t - c.last_touch[k as usize];
                let h = (mult * self.glob.evict_age_ewma).max(1.0);
                let mut f = [0f32; N_HAZ];
                features::hazard(c, tr, k, a, h, &mut f);
                -m.predict(&f)
            }
        }
    }

    fn needs_features(&self) -> bool {
        matches!(self.scorer, Scorer::Learned(_)) || self.logger.is_some()
    }
}

impl Policy for Sampled {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, _k: u32, _p: Pos) {}
    fn on_access(&mut self, _c: &mut Core, _t: &Trace, _k: u32, _p: Pos) {}

    fn victim(&mut self, c: &mut Core, tr: &Trace, pos: Pos) -> Option<u32> {
        if self.follow && self.pending != NONE && c.evictable(self.pending) && self.unchanged(c) {
            let k = self.pending;
            self.set_pending(c, tr, k);
            self.observe_eviction(pos.t - c.last_touch[k as usize]);
            return Some(k);
        }
        self.pending = NONE;
        let burst = (pos.t, pos.kind, pos.idx);
        if self.spare_at == burst {
            while let Some(k) = self.spare.pop() {
                if c.evictable(k) {
                    if self.follow {
                        self.set_pending(c, tr, k);
                    }
                    self.observe_eviction(pos.t - c.last_touch[k as usize]);
                    return Some(k);
                }
            }
        }
        let n = c.evictable_set.len();
        if n == 0 {
            return None;
        }
        self.cand.clear();
        for _ in 0..self.k {
            let r = (self.next_rand() % n as u64) as usize;
            self.cand.push(c.evictable_set[r]);
        }
        let want_feat = self.needs_features();
        if want_feat {
            for j in 0..self.cand.len() {
                let mut f = [0f32; N_FEAT];
                features::compute(c, tr, self.cand[j], pos.t, pos.next, &self.glob, &mut f);
                self.feat[j] = f;
            }
        }
        if let Scorer::Hazard(m, mult) = &self.scorer {
            let h = (mult * self.glob.evict_age_ewma).max(1.0);
            for (j, &k) in self.cand.iter().enumerate() {
                let a = pos.t - c.last_touch[k as usize];
                let mut f = [0f32; N_HAZ];
                features::hazard(c, tr, k, a, h, &mut f);
                self.hbuf[j * N_HAZ..(j + 1) * N_HAZ].copy_from_slice(&f);
            }
            m.predict_batch(&self.hbuf, self.cand.len(), &mut self.hout);
            if self.cost_alpha > 0.0 {
                for (j, &k) in self.cand.iter().enumerate() {
                    let p = 1.0 / (1.0 + (-self.hout[j]).exp());
                    let cost = 1.0 + self.cost_alpha * tr.key_depth[k as usize] as f64;
                    self.hout[j] = (p * cost).ln();
                }
            }
            self.model_calls += self.cand.len() as u64;
        }
        let mut best = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for j in 0..self.cand.len() {
            let s = if matches!(self.scorer, Scorer::Hazard(..)) {
                -self.hout[j]
            } else {
                self.score(c, tr, j, pos)
            };
            if s > best_s {
                best_s = s;
                best = j;
            }
        }
        if let Some(lg) = self.logger.as_mut() {
            lg.decision += 1;
            let in_window = pos.t >= lg.window.0 && pos.t < lg.window.1;
            let x = self.rng;
            let u = ((x >> 11) as f64) / ((1u64 << 53) as f64);
            if in_window && u < lg.rate {
                for j in 0..self.cand.len() {
                    let k = self.cand[j];
                    lg.rows.push(lg.decision as f32);
                    lg.rows.push(pos.t as f32);
                    lg.rows.extend_from_slice(&self.feat[j]);
                    let y = features::ttnu(tr, k, pos.t, pos.next);
                    lg.rows.push((1.0 + y).ln() as f32);
                    lg.n_rows += 1;
                }
            }
        }
        let k = self.cand[best];
        if self.reuse > 1 {
            self.order.clear();
            self.order.extend(0..self.cand.len());
            let sc: Vec<f64> = (0..self.cand.len())
                .map(|j| {
                    if matches!(self.scorer, Scorer::Hazard(..)) {
                        -self.hout[j]
                    } else {
                        self.score(c, tr, j, pos)
                    }
                })
                .collect();
            self.order
                .sort_by(|&x, &y| sc[y].total_cmp(&sc[x]).then(x.cmp(&y)));
            self.spare.clear();
            for &j in self.order.iter().skip(1).take(self.reuse - 1).rev() {
                let kk = self.cand[j];
                if kk != k && !self.spare.contains(&kk) {
                    self.spare.push(kk);
                }
            }
            self.spare_at = burst;
        }
        if self.follow {
            self.set_pending(c, tr, k);
        }
        self.observe_eviction(pos.t - c.last_touch[k as usize]);
        Some(k)
    }
}

impl Sampled {
    /// Remembers the parent to evict next, with the history it had when chosen; it is used
    /// only if that history is unchanged (no hit, release or re-insert in between).
    fn set_pending(&mut self, c: &Core, tr: &Trace, k: u32) {
        self.pending = self.chain_next(c, tr, k);
        if self.pending != NONE {
            let p = self.pending as usize;
            self.pending_seen = (c.owner[p], c.n_access[p], c.last_touch[p]);
        }
    }

    fn unchanged(&self, c: &Core) -> bool {
        let p = self.pending as usize;
        (c.owner[p], c.n_access[p], c.last_touch[p]) == self.pending_seen
    }

    fn chain_next(&self, c: &Core, tr: &Trace, k: u32) -> u32 {
        let p = tr.key_parent[k as usize];
        if p != NONE
            && c.owner[p as usize] == c.owner[k as usize]
            && c.n_access[p as usize] == c.n_access[k as usize]
        {
            p
        } else {
            NONE
        }
    }

    pub fn observe_eviction(&mut self, age: f64) {
        self.glob.evict_age_ewma = 0.999 * self.glob.evict_age_ewma + 0.001 * age;
    }
}
