use crate::gbdt::Gbdt;
use crate::policy::{self, sampled::Logger, Params, Policy, Resources, Sampled, Scorer};
use crate::sim::{Sim, SimConfig, Stats};
use crate::trace::Trace;
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct RunSpec {
    pub policy: String,
    pub label: String,
    pub params: Params,
    pub cap_frac: f64,
    pub window: (f64, f64),
    pub time_victims: bool,
}

#[derive(Serialize, Debug)]
pub struct RunResult {
    pub trace: String,
    pub policy: String,
    pub label: String,
    pub params: Params,
    pub cap_frac: f64,
    pub cap_blocks: u64,
    pub window: (f64, f64),
    pub hit_ratio: f64,
    pub wall_s: f64,
    pub ns_per_decision: f64,
    #[serde(flatten)]
    pub stats: Stats,
}

pub fn cap_blocks(tr: &Trace, frac: f64) -> u64 {
    ((tr.n_keys as f64) * frac).round().max(1.0) as u64
}

pub fn run(tr: &Arc<Trace>, spec: &RunSpec, res: &Resources) -> Result<RunResult> {
    let cap = cap_blocks(tr, spec.cap_frac);
    let mut pol = policy::make(&spec.policy, &spec.params, tr, cap, res)?;
    let t0 = Instant::now();
    let stats = simulate(tr, pol.as_mut(), cap, spec.window, spec.time_victims);
    let wall = t0.elapsed().as_secs_f64();
    Ok(RunResult {
        trace: tr.name.clone(),
        policy: spec.policy.clone(),
        label: spec.label.clone(),
        params: spec.params.clone(),
        cap_frac: spec.cap_frac,
        cap_blocks: cap,
        window: spec.window,
        hit_ratio: stats.hit_ratio(),
        wall_s: wall,
        ns_per_decision: if stats.decisions > 0 {
            stats.victim_ns as f64 / stats.decisions as f64
        } else {
            0.0
        },
        stats,
    })
}

pub fn simulate(
    tr: &Arc<Trace>,
    pol: &mut dyn Policy,
    cap: u64,
    window: (f64, f64),
    time_victims: bool,
) -> Stats {
    let cfg = SimConfig {
        cap,
        window,
        time_victims,
    };
    let mut sim = Sim::new(tr.clone(), &cfg);
    sim.run_until(pol, window.1);
    sim.stats
}

/// Runs a sampled policy with candidate logging and returns (rows, n_rows).
pub fn label(
    tr: &Arc<Trace>,
    scorer: Scorer,
    k: usize,
    cap_frac: f64,
    log_window: (f64, f64),
    rate: f64,
    seed: u64,
) -> (Vec<f32>, usize, Stats) {
    let cap = cap_blocks(tr, cap_frac);
    let mut pol = Sampled::new(scorer, k, true, seed);
    pol.logger = Some(Logger::new(rate, log_window));
    let stats = simulate(tr, &mut pol, cap, (0.0, f64::INFINITY), false);
    let lg = pol.logger.take().unwrap();
    (lg.rows, lg.n_rows, stats)
}

/// Cache-independent training rows for the hazard model. Replays the trace with an unbounded
/// cache; at a sampled fraction of block releases, draws `per_touch` (elapsed a, horizon h)
/// pairs and labels whether the block is next used in (a, a + h], using only uses before
/// `cutoff`. Rows whose window ends after `cutoff` are dropped whatever their label (keeping
/// the positives would over-represent reuse at long horizons), as are rows already used by
/// `a`. Columns: touch time, hazard features..., label.
pub fn hazard_rows(
    tr: &Arc<Trace>,
    cutoff: f64,
    rate: f64,
    per_touch: usize,
    seed: u64,
) -> (Vec<f32>, usize) {
    struct Rec {
        lru: crate::policy::Lru,
        rate: f64,
        per_touch: usize,
        cutoff: f64,
        rng: u64,
        rows: Vec<f32>,
        n: usize,
    }
    impl Rec {
        fn rand(&mut self) -> f64 {
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng = x;
            ((x >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }
    impl Policy for Rec {
        fn on_insert(&mut self, c: &mut crate::sim::Core, t: &Trace, k: u32, p: crate::sim::Pos) {
            self.lru.on_insert(c, t, k, p)
        }
        fn on_access(&mut self, c: &mut crate::sim::Core, t: &Trace, k: u32, p: crate::sim::Pos) {
            self.lru.on_access(c, t, k, p)
        }
        fn on_release(&mut self, c: &mut crate::sim::Core, t: &Trace, k: u32, p: crate::sim::Pos) {
            if p.t >= self.cutoff || self.rand() >= self.rate {
                return;
            }
            let nu = crate::features::next_use_before(t, k, p.next, self.cutoff);
            for _ in 0..self.per_touch {
                let a = if self.rand() < 0.2 {
                    0.0
                } else {
                    (5f64.ln() + self.rand() * (3600f64 / 5.0).ln()).exp()
                };
                let h = (10f64.ln() + self.rand() * (2000f64 / 10.0).ln()).exp();
                if p.t + a + h > self.cutoff {
                    continue;
                }
                let label = match nu {
                    Some(u) if u <= p.t + a => continue,
                    Some(u) => (u <= p.t + a + h) as u8,
                    None => 0,
                };
                let mut f = [0f32; crate::features::N_HAZ];
                crate::features::hazard(c, t, k, a, h, &mut f);
                self.rows.push(p.t as f32);
                self.rows.extend_from_slice(&f);
                self.rows.push(label as f32);
                self.n += 1;
            }
        }
        fn victim(
            &mut self,
            c: &mut crate::sim::Core,
            t: &Trace,
            p: crate::sim::Pos,
        ) -> Option<u32> {
            self.lru.victim(c, t, p)
        }
    }
    let mut rec = Rec {
        lru: crate::policy::Lru::new(tr.n_keys),
        rate,
        per_touch,
        cutoff,
        rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        rows: Vec::new(),
        n: 0,
    };
    let cfg = SimConfig {
        cap: tr.n_keys as u64,
        window: (0.0, cutoff),
        time_victims: false,
    };
    let mut sim = Sim::new(tr.clone(), &cfg);
    while !sim.finished() {
        if let Some(r) = tr.reqs.get(sim.next_arrival_index()) {
            if r.t >= cutoff {
                break;
            }
        }
        sim.step(&mut rec, None);
    }
    (rec.rows, rec.n)
}

pub fn load_wa_fits(
    paths: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, Arc<crate::policy::WaFit>>> {
    let mut m = BTreeMap::new();
    for (trace, p) in paths {
        m.insert(
            trace.clone(),
            Arc::new(crate::policy::WaFit::load(std::path::Path::new(p))?),
        );
    }
    Ok(m)
}

pub fn load_models(paths: &BTreeMap<String, String>) -> Result<BTreeMap<String, Arc<Gbdt>>> {
    let mut m = BTreeMap::new();
    for (id, p) in paths {
        m.insert(id.clone(), Arc::new(Gbdt::load(std::path::Path::new(p))?));
    }
    Ok(m)
}
