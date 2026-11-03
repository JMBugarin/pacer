//! Features of an evictable block at decision time. Everything here is known online: block
//! history, the request that last touched the block, its session so far, and global load.

use crate::sim::Core;
use crate::trace::{Trace, NONE};

pub const N_FEAT: usize = 18;
pub const FEATURE_NAMES: [&str; N_FEAT] = [
    "age",
    "age_insert",
    "n_access",
    "depth",
    "is_output",
    "fanout",
    "turn",
    "rtype",
    "input_len",
    "output_len",
    "since_done",
    "prev_gap",
    "mean_gap",
    "session_age",
    "child_arrived",
    "arrivals_60s",
    "evict_age_ewma",
    "rel_depth",
];

/// Features of the session-return hazard model: what was known when the block was last
/// touched, plus the elapsed time `a` since then and the horizon `h` being asked about.
pub const N_HAZ: usize = 16;
pub const HAZARD_NAMES: [&str; N_HAZ] = [
    "a",
    "h",
    "n_access",
    "depth",
    "is_output",
    "fanout",
    "turn",
    "rtype",
    "input_len",
    "output_len",
    "prev_gap",
    "mean_gap",
    "session_age",
    "arrivals_60s",
    "rel_depth",
    "a_over_gap",
];

pub fn hazard(core: &Core, tr: &Trace, k: u32, a: f64, h: f64, out: &mut [f32; N_HAZ]) {
    let i = k as usize;
    let o = core.owner[i];
    let r = &tr.reqs[o as usize];
    let root = &tr.reqs[r.root as usize];
    let touch = core.last_touch[i];
    let prev_gap = if r.parent != NONE {
        r.t - tr.reqs[r.parent as usize].t
    } else {
        -1.0
    };
    let mean_gap = if r.turn > 1 && r.root != o {
        (r.t - root.t) / (r.turn - 1) as f64
    } else {
        -1.0
    };
    let chain_len = (r.n_in + r.n_out).max(1) as f64;
    let vals = [
        a,
        h,
        core.n_access[i] as f64,
        tr.key_depth[i] as f64,
        tr.key_output[i] as u8 as f64,
        core.fanout[i] as f64,
        r.turn as f64,
        r.rtype as f64,
        r.input_len as f64,
        r.output_len as f64,
        prev_gap,
        mean_gap,
        touch - root.t,
        if touch == r.done {
            tr.arrivals_at_done[o as usize] as f64
        } else {
            arrivals_since(tr, touch - 60.0, touch) as f64
        },
        tr.key_depth[i] as f64 / chain_len,
        if prev_gap > 0.0 { a / prev_gap } else { -1.0 },
    ];
    for (d, v) in out.iter_mut().zip(vals) {
        *d = v as f32;
    }
}

/// Label horizon: time-to-next-use is capped here; "never used again" maps to the cap.
pub const TTNU_CAP: f64 = 20000.0;

#[derive(Clone, Debug, Default)]
pub struct Global {
    pub evict_age_ewma: f64,
}

/// Arrivals in [from, now).
pub fn arrivals_since(tr: &Trace, from: f64, now: f64) -> usize {
    let a = tr.reqs.partition_point(|r| r.t < from);
    let b = tr.reqs.partition_point(|r| r.t < now);
    b.saturating_sub(a)
}

/// `next` is the first arrival not processed yet; a child counts as arrived only before it.
pub fn compute(
    core: &Core,
    tr: &Trace,
    k: u32,
    now: f64,
    next: u32,
    g: &Global,
    out: &mut [f32; N_FEAT],
) {
    let i = k as usize;
    let o = core.owner[i];
    let r = &tr.reqs[o as usize];
    let root = &tr.reqs[r.root as usize];
    let prev_gap = if r.parent != NONE {
        r.t - tr.reqs[r.parent as usize].t
    } else {
        -1.0
    };
    let mean_gap = if r.turn > 1 && r.root != o {
        (r.t - root.t) / (r.turn - 1) as f64
    } else {
        -1.0
    };
    let child_arrived = r.first_child != NONE && r.first_child < next;
    let chain_len = (r.n_in + r.n_out).max(1) as f64;
    let vals = [
        now - core.last_touch[i],
        now - core.insert_time[i],
        core.n_access[i] as f64,
        tr.key_depth[i] as f64,
        tr.key_output[i] as u8 as f64,
        core.fanout[i] as f64,
        r.turn as f64,
        r.rtype as f64,
        r.input_len as f64,
        r.output_len as f64,
        now - r.done,
        prev_gap,
        mean_gap,
        now - root.t,
        child_arrived as u8 as f64,
        arrivals_since(tr, now - 60.0, now) as f64,
        g.evict_age_ewma,
        tr.key_depth[i] as f64 / chain_len,
    ];
    for (d, v) in out.iter_mut().zip(vals) {
        *d = v as f32;
    }
}

/// Time of the block's first use from arrival `from` on that happens before `cutoff`.
pub fn next_use_before(tr: &Trace, k: u32, from: u32, cutoff: f64) -> Option<f64> {
    let j = tr.next_use(k, from) as usize;
    if j < tr.reqs.len() && tr.reqs[j].t < cutoff {
        Some(tr.reqs[j].t)
    } else {
        None
    }
}

/// Seconds from `now` until the block's next use by an arrival from `from` on, capped at
/// TTNU_CAP.
pub fn ttnu(tr: &Trace, k: u32, now: f64, from: u32) -> f64 {
    let j = tr.next_use(k, from) as usize;
    if j < tr.reqs.len() {
        (tr.reqs[j].t - now).min(TTNU_CAP)
    } else {
        TTNU_CAP
    }
}
