//! Simulator invariants checked after every step, for every policy that needs no model, on
//! random synthetic traces (proptest) and on a prefix of the traceA fixture.

use pacer::policy::{self, Params, Resources};
use pacer::run::cap_blocks;
use pacer::sim::{Sim, SimConfig, StepLog};
use pacer::trace::{self, RawReq, Trace, NONE};
use proptest::prelude::*;
use proptest::sample::Index;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const POLICIES: [&str; 14] = [
    "lru",
    "fifo",
    "lfu",
    "belady",
    "gdsf",
    "sieve",
    "s3fifo",
    "arc",
    "lruqd",
    "wa",
    "sampled-lru",
    "sampled-belady",
    "oracle-dead",
    "oracle-within",
];
const SYSTEM: [&[u64]; 4] = [&[], &[900], &[900, 901], &[902]];
const GAPS: [f64; 6] = [0.0, 0.0, 0.01, 0.1, 0.3, 1.0];
const OUT_LENS: [u32; 7] = [0, 0, 8, 16, 24, 40, 70];
const MAX_BLOCKS: usize = 8;

#[derive(Clone, Debug)]
struct Spec {
    parent: Option<Index>,
    system: usize,
    extra: usize,
    diverge: bool,
    partial: bool,
    out: usize,
    gap: usize,
    rtype: u8,
}

fn spec() -> impl Strategy<Value = Spec> {
    (
        proptest::option::weighted(0.6, any::<Index>()),
        0..SYSTEM.len(),
        0..4usize,
        proptest::bool::weighted(0.15),
        proptest::bool::weighted(0.2),
        0..OUT_LENS.len(),
        0..GAPS.len(),
        0..8u8,
    )
        .prop_map(
            |(parent, system, extra, diverge, partial, out, gap, rtype)| Spec {
                parent,
                system,
                extra,
                diverge,
                partial,
                out,
                gap,
                rtype,
            },
        )
}

/// Sessions extend (or with `diverge`, fork from) a parent's full input blocks. With
/// `positive`, every request ends strictly after it starts: no empty outputs and no child
/// within 2 ms of its parent.
fn rows(specs: &[Spec], positive: bool) -> Vec<RawReq> {
    let mut rows: Vec<RawReq> = Vec::new();
    let mut fresh = 1u64;
    let mut t = 0.0;
    for (i, s) in specs.iter().enumerate() {
        t += GAPS[s.gap];
        let eligible: Vec<usize> = (0..rows.len())
            .filter(|&j| !positive || rows[j].timestamp <= t - 0.002)
            .collect();
        let (mut hashes, parent, turn) = match s.parent {
            Some(ix) if !eligible.is_empty() => {
                let p = &rows[eligible[ix.index(eligible.len())]];
                let mut base: Vec<u64> = p
                    .hash_ids
                    .iter()
                    .take((p.input_length / 16) as usize)
                    .copied()
                    .collect();
                if s.diverge {
                    if let Some(last) = base.last_mut() {
                        *last = fresh;
                        fresh += 1;
                    }
                }
                (base, p.chat_id, p.turn + 1)
            }
            _ => (SYSTEM[s.system].to_vec(), -1, 1),
        };
        for _ in 0..s.extra {
            hashes.push(fresh);
            fresh += 1;
        }
        hashes.truncate(MAX_BLOCKS);
        let cut = if s.partial { 7 } else { 0 };
        let input_length = if hashes.is_empty() {
            2 * cut
        } else {
            16 * hashes.len() as u32 - cut
        };
        let out = OUT_LENS[s.out];
        rows.push(RawReq {
            chat_id: i as i64,
            parent_chat_id: parent,
            timestamp: t,
            input_length,
            output_length: if positive { out.max(1) } else { out },
            rtype: s.rtype,
            turn,
            hash_ids: hashes,
        });
    }
    rows
}

struct Outcome {
    stats: String,
    hits: u64,
    evicted: Vec<Vec<u32>>,
}

fn check_state(sim: &Sim, tr: &Trace, live: &[bool], name: &str) -> Result<(), TestCaseError> {
    let c = &sim.core;
    let n = tr.n_keys as usize;
    let mut pins = vec![0u32; n];
    for (r, _) in tr.reqs.iter().zip(live).filter(|(_, l)| **l) {
        for &k in r.input(&tr.keys) {
            pins[k as usize] += 1;
        }
    }
    prop_assert_eq!(&c.refc, &pins, "{}: pin counts", name);
    let pinned_uncached = pins
        .iter()
        .zip(&c.cached)
        .position(|(&p, &cached)| p > 0 && !cached);
    prop_assert!(
        pinned_uncached.is_none(),
        "{name}: pinned block {pinned_uncached:?} not cached"
    );
    let mut children = vec![0u32; n];
    for k in (0..n).filter(|&k| c.cached[k]) {
        let p = tr.key_parent[k];
        if p != NONE {
            prop_assert!(
                c.cached[p as usize],
                "{name}: block {k} cached without parent {p}"
            );
            children[p as usize] += 1;
        }
    }
    prop_assert_eq!(&c.children, &children, "{}: child counts", name);
    prop_assert_eq!(
        c.size,
        c.cached.iter().filter(|&&x| x).count() as u64,
        "{}: size",
        name
    );
    let mut set = c.evictable_set.clone();
    set.sort_unstable();
    let want: Vec<u32> = (0..n)
        .filter(|&k| c.cached[k] && c.refc[k] == 0 && children[k] == 0)
        .map(|k| k as u32)
        .collect();
    prop_assert_eq!(&set, &want, "{}: evictable set", name);
    if c.size > c.cap {
        prop_assert!(
            sim.stats.max_overflow >= c.size - c.cap,
            "{name}: unrecorded overflow"
        );
        prop_assert!(
            want.is_empty(),
            "{name}: over capacity with {} evictable",
            want.len()
        );
    }
    Ok(())
}

fn run(tr: &Arc<Trace>, name: &str, cap: u64) -> Result<Outcome, TestCaseError> {
    let res = Resources {
        models: BTreeMap::new(),
        wa_fits: BTreeMap::new(),
    };
    let mut pol = policy::make(name, &Params::new(), tr, cap, &res).unwrap();
    let cfg = SimConfig {
        cap,
        window: (0.0, f64::INFINITY),
        time_victims: false,
    };
    let mut sim = Sim::new(tr.clone(), &cfg);
    let mut live = vec![false; tr.reqs.len()];
    let mut hit_sum = 0u64;
    let mut evicted = Vec::new();
    while !sim.finished() {
        let mut log = StepLog::default();
        sim.step(pol.as_mut(), Some(&mut log));
        let r = &tr.reqs[log.req as usize];
        prop_assert_ne!(live[log.req as usize], log.arrival, "{}: event order", name);
        live[log.req as usize] = log.arrival;
        if log.arrival {
            prop_assert!(log.hit_blocks <= r.n_in);
            hit_sum += log.hit_blocks as u64;
        }
        for &k in &log.evicted {
            prop_assert!(
                !sim.core.cached[k as usize],
                "{name}: evicted block {k} still cached"
            );
        }
        for &k in &log.inserted {
            prop_assert!(
                sim.core.cached[k as usize] || log.evicted.contains(&k),
                "{name}: inserted block {k} vanished"
            );
        }
        check_state(&sim, tr, &live, name)?;
        evicted.push(log.evicted);
    }
    prop_assert!(live.iter().all(|&l| !l));
    let s = &sim.stats;
    prop_assert_eq!(s.hits, hit_sum);
    prop_assert_eq!(s.accesses, tr.input_accesses());
    prop_assert!(s.hits <= s.accesses);
    Ok(Outcome {
        stats: serde_json::to_string(s).unwrap(),
        hits: s.hits,
        evicted,
    })
}

fn check_all(tr: &Arc<Trace>, cap: u64) -> Result<(), TestCaseError> {
    // Belady can lose to other policies when a request completes at its own arrival time (see
    // belady_zero_duration_* below), so dominance is only asserted without such requests.
    let positive = tr.reqs.iter().all(|r| r.done > r.t);
    let mut hits = BTreeMap::new();
    for name in POLICIES {
        let a = run(tr, name, cap)?;
        let b = run(tr, name, cap)?;
        prop_assert_eq!(
            &a.stats,
            &b.stats,
            "{}: stats differ between identical runs",
            name
        );
        prop_assert!(
            a.evicted == b.evicted,
            "{name}: eviction sequence differs between runs"
        );
        hits.insert(name, a.hits);
    }
    if positive {
        let best = hits["belady"];
        for (name, &h) in &hits {
            prop_assert!(h <= best, "{name} beats belady: {h} > {best} at cap {cap}");
        }
    }
    Ok(())
}

fn build(rows: Vec<RawReq>) -> Arc<Trace> {
    Arc::new(trace::build("prop".into(), rows, trace::DEFAULT_TPOT).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    #[test]
    fn invariants_hold_on_random_traces(
        specs in prop::collection::vec(spec(), 1..40),
        cap in 1u64..32,
        positive in any::<bool>(),
    ) {
        check_all(&build(rows(&specs, positive)), cap)?;
    }
}

#[test]
fn invariants_hold_on_fixture_prefix() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/qwen_traceA_blksz_16.jsonl");
    let tr = Arc::new(trace::load(&path, Some(120), trace::DEFAULT_TPOT).unwrap());
    assert!(tr.reqs.iter().all(|r| r.done > r.t));
    for frac in [0.05, 0.3] {
        check_all(&tr, cap_blocks(&tr, frac)).unwrap();
    }
}

fn raw(chat: i64, parent: i64, t: f64, input: u32, output: u32, hashes: &[u64]) -> RawReq {
    RawReq {
        chat_id: chat,
        parent_chat_id: parent,
        timestamp: t,
        input_length: input,
        output_length: output,
        rtype: 0,
        turn: if parent == -1 { 1 } else { 2 },
        hash_ids: hashes.to_vec(),
    }
}

fn hits(tr: &Arc<Trace>, name: &str, cap: u64) -> u64 {
    run(tr, name, cap).unwrap().hits
}

/// Regression: request 0 completes at t=0, right after its own arrival. Belady once took request
/// 0 itself as the next use of block A, kept A (never used again) and evicted D, which request 2
/// needs. Next use is now measured from the first unprocessed arrival. Optimal: 3 hits.
#[test]
fn belady_zero_duration_keeps_dead_block() {
    let tr = build(vec![
        raw(0, -1, 0.0, 25, 0, &[900, 1]),
        raw(1, -1, 0.0, 41, 8, &[902, 2, 3]),
        raw(2, 1, 0.01, 49, 8, &[902, 2, 4]),
    ]);
    assert_eq!(hits(&tr, "belady", 3), 3);
}

/// Regression: request 1 has no full input block, completes at t=0.3 after request 0's arrival
/// at t=0.3, and inserts request 2's first block C. Belady once counted request 0's (already
/// processed) arrival as a future use of block B and evicted C instead of B.
#[test]
fn belady_zero_duration_loses_to_lru() {
    let tr = build(vec![
        raw(0, -1, 0.3, 16, 0, &[1]),
        raw(1, -1, 0.3, 9, 8, &[7]),
        raw(2, 1, 0.3, 17, 0, &[2]),
        raw(3, -1, 1.3, 16, 0, &[1]),
    ]);
    assert!(hits(&tr, "belady", 1) >= hits(&tr, "lru", 1));
}
