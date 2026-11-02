//! Victim-selection cost on the traceA fixture slice (400 requests, cache at 10% of its blocks).
//!
//! `victim/*` reports only the time spent inside `Policy::victim`, as timed by the simulator
//! (`time_victims`, which adds two `Instant::now` calls per decision), with one decision per
//! element; `replay/*` is the wall time of a whole replay for context.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use pacer::policy::{self, Params, Resources};
use pacer::run::cap_blocks;
use pacer::sim::{Sim, SimConfig, Stats};
use pacer::trace::{self, Trace};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const POLICIES: [&str; 3] = ["lru", "wa", "sampled-lru"];

fn replay(tr: &Arc<Trace>, name: &str, cap: u64, time_victims: bool) -> Stats {
    let res = Resources {
        models: BTreeMap::new(),
        wa_fits: BTreeMap::new(),
    };
    let mut pol = policy::make(name, &Params::new(), tr, cap, &res).unwrap();
    let cfg = SimConfig {
        cap,
        window: (0.0, f64::INFINITY),
        time_victims,
    };
    let mut sim = Sim::new(tr.clone(), &cfg);
    sim.run(pol.as_mut());
    sim.stats
}

fn decision(c: &mut Criterion) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/qwen_traceA_blksz_16.jsonl");
    let tr = Arc::new(trace::load(&path, None, trace::DEFAULT_TPOT).unwrap());
    let cap = cap_blocks(&tr, 0.1);

    let mut g = c.benchmark_group("victim");
    g.sample_size(10).warm_up_time(Duration::from_millis(300));
    g.measurement_time(Duration::from_secs(1));
    for name in POLICIES {
        g.throughput(Throughput::Elements(
            replay(&tr, name, cap, false).decisions,
        ));
        g.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let ns: u64 = (0..iters)
                    .map(|_| replay(&tr, name, cap, true).victim_ns)
                    .sum();
                Duration::from_nanos(ns)
            })
        });
    }
    g.finish();

    let mut g = c.benchmark_group("replay");
    g.sample_size(10).warm_up_time(Duration::from_millis(300));
    g.measurement_time(Duration::from_secs(1));
    for name in POLICIES {
        g.bench_function(name, |b| b.iter(|| replay(&tr, name, cap, false)));
    }
    g.finish();
}

criterion_group!(benches, decision);
criterion_main!(benches);
