use crate::policy::Policy;
use crate::trace::{Trace, NONE};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

pub const BUCKET_SECS: f64 = 300.0;

/// Event position: time, kind (0 completion, 1 arrival), request index, and `next`, the index
/// of the first arrival not processed yet. "Future" always means arrivals from `next` on;
/// completions are processed before arrivals with the same time.
#[derive(Clone, Copy, Debug)]
pub struct Pos {
    pub t: f64,
    pub kind: u8,
    pub idx: u32,
    pub next: u32,
}

pub struct Core {
    pub cap: u64,
    pub size: u64,
    pub cached: Vec<bool>,
    pub children: Vec<u32>,
    pub refc: Vec<u32>,
    pub last_touch: Vec<f64>,
    pub insert_time: Vec<f64>,
    pub n_access: Vec<u32>,
    pub owner: Vec<u32>,
    pub fanout: Vec<u32>,
    pub seen: Vec<bool>,
    pub seq: u64,
    evict_pos: Vec<u32>,
    pub evictable_set: Vec<u32>,
}

impl Core {
    pub fn new(n_keys: u32, cap: u64) -> Self {
        let n = n_keys as usize;
        Core {
            cap,
            size: 0,
            cached: vec![false; n],
            children: vec![0; n],
            refc: vec![0; n],
            last_touch: vec![0.0; n],
            insert_time: vec![0.0; n],
            n_access: vec![0; n],
            owner: vec![NONE; n],
            fanout: vec![0; n],
            seen: vec![false; n],
            seq: 0,
            evict_pos: vec![NONE; n],
            evictable_set: Vec::new(),
        }
    }

    #[inline]
    pub fn evictable(&self, k: u32) -> bool {
        let i = k as usize;
        self.cached[i] && self.refc[i] == 0 && self.children[i] == 0
    }

    #[inline]
    pub fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn set_add(&mut self, k: u32) {
        if self.evict_pos[k as usize] == NONE {
            self.evict_pos[k as usize] = self.evictable_set.len() as u32;
            self.evictable_set.push(k);
        }
    }

    fn set_remove(&mut self, k: u32) {
        let p = self.evict_pos[k as usize];
        if p != NONE {
            let last = *self.evictable_set.last().unwrap();
            self.evictable_set.swap_remove(p as usize);
            if last != k {
                self.evict_pos[last as usize] = p;
            }
            self.evict_pos[k as usize] = NONE;
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Bucket {
    pub hits: u64,
    pub accesses: u64,
    pub hit_depth_sum: u64,
    pub access_depth_sum: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub hits: u64,
    pub accesses: u64,
    pub hit_depth_sum: u64,
    pub access_depth_sum: u64,
    pub evictions: u64,
    pub decisions: u64,
    pub max_overflow: u64,
    pub victim_ns: u64,
    pub buckets: Vec<Bucket>,
}

impl Stats {
    pub fn hit_ratio(&self) -> f64 {
        if self.accesses == 0 {
            0.0
        } else {
            self.hits as f64 / self.accesses as f64
        }
    }
}

pub struct SimConfig {
    pub cap: u64,
    pub window: (f64, f64),
    pub time_victims: bool,
}

pub struct Sim {
    pub trace: Arc<Trace>,
    pub core: Core,
    pub stats: Stats,
    window: (f64, f64),
    time_victims: bool,
    next_arrival: usize,
    done: BinaryHeap<Reverse<(OrdF64, u32)>>,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct OrdF64(pub f64);
impl Eq for OrdF64 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// What one step of the simulation did; the live MLX replay mirrors these on its block pool.
#[derive(Debug, Default, Clone)]
pub struct StepLog {
    pub arrival: bool,
    pub req: u32,
    pub hit_blocks: u32,
    pub inserted: Vec<u32>,
    pub evicted: Vec<u32>,
}

impl Sim {
    pub fn new(trace: Arc<Trace>, cfg: &SimConfig) -> Self {
        Sim {
            core: Core::new(trace.n_keys, cfg.cap),
            trace,
            stats: Stats::default(),
            window: cfg.window,
            time_victims: cfg.time_victims,
            next_arrival: 0,
            done: BinaryHeap::new(),
        }
    }

    pub fn next_arrival_index(&self) -> usize {
        self.next_arrival
    }

    pub fn finished(&self) -> bool {
        self.next_arrival >= self.trace.reqs.len() && self.done.is_empty()
    }

    pub fn run(&mut self, pol: &mut dyn Policy) {
        self.run_until(pol, f64::INFINITY);
    }

    /// Runs every event before `stop` (arrivals at or after `stop` are not processed).
    pub fn run_until(&mut self, pol: &mut dyn Policy, stop: f64) {
        while !self.finished() {
            if let Some(r) = self.trace.reqs.get(self.next_arrival) {
                if r.t >= stop && self.done.peek().is_none_or(|d| d.0 .0 .0 >= stop) {
                    break;
                }
            } else if self.done.peek().is_some_and(|d| d.0 .0 .0 >= stop) {
                break;
            }
            self.step(pol, None);
        }
    }

    pub fn step(&mut self, pol: &mut dyn Policy, mut log: Option<&mut StepLog>) {
        let arc = self.trace.clone();
        let tr: &Trace = &arc;
        let take_done = match self.done.peek() {
            Some(Reverse((d, _))) => {
                self.next_arrival >= tr.reqs.len() || d.0 <= tr.reqs[self.next_arrival].t
            }
            None => false,
        };
        if take_done {
            let Reverse((d, idx)) = self.done.pop().unwrap();
            let pos = Pos {
                t: d.0,
                kind: 0,
                idx,
                next: self.next_arrival as u32,
            };
            self.complete(tr, pol, idx, pos, log.as_deref_mut());
        } else {
            let idx = self.next_arrival as u32;
            self.next_arrival += 1;
            let pos = Pos {
                t: tr.reqs[idx as usize].t,
                kind: 1,
                idx,
                next: self.next_arrival as u32,
            };
            self.arrive(tr, pol, idx, pos, log);
        }
    }

    fn add(&mut self, tr: &Trace, k: u32, owner: u32, now: f64) {
        let c = &mut self.core;
        let i = k as usize;
        c.cached[i] = true;
        c.children[i] = 0;
        c.refc[i] = 0;
        c.insert_time[i] = now;
        c.last_touch[i] = now;
        c.owner[i] = owner;
        c.size += 1;
        c.set_add(k);
        let first = !c.seen[i];
        c.seen[i] = true;
        let p = tr.key_parent[i];
        if p != NONE {
            let pi = p as usize;
            c.children[pi] += 1;
            if first {
                c.fanout[pi] += 1;
            }
            if c.children[pi] == 1 {
                c.set_remove(p);
            }
        }
    }

    fn remove(&mut self, tr: &Trace, pol: &mut dyn Policy, k: u32) {
        let c = &mut self.core;
        c.set_remove(k);
        c.cached[k as usize] = false;
        c.size -= 1;
        let p = tr.key_parent[k as usize];
        if p != NONE {
            let pi = p as usize;
            c.children[pi] -= 1;
            if c.children[pi] == 0 && c.refc[pi] == 0 {
                c.set_add(p);
                pol.on_evictable(&self.core, tr, p);
            }
        }
    }

    fn evict_to_capacity(
        &mut self,
        tr: &Trace,
        pol: &mut dyn Policy,
        pos: Pos,
        log: &mut Option<&mut StepLog>,
    ) {
        while self.core.size > self.core.cap {
            let t0 = if self.time_victims {
                Some(Instant::now())
            } else {
                None
            };
            let v = pol.victim(&mut self.core, tr, pos);
            if let Some(t0) = t0 {
                self.stats.victim_ns += t0.elapsed().as_nanos() as u64;
            }
            self.stats.decisions += 1;
            match v {
                Some(v) if self.core.evictable(v) => {
                    pol.on_evict(&self.core, tr, v);
                    self.remove(tr, pol, v);
                    self.stats.evictions += 1;
                    if let Some(l) = log.as_deref_mut() {
                        l.evicted.push(v);
                    }
                }
                _ => {
                    let over = self.core.size - self.core.cap;
                    self.stats.max_overflow = self.stats.max_overflow.max(over);
                    return;
                }
            }
        }
    }

    fn arrive(
        &mut self,
        tr: &Trace,
        pol: &mut dyn Policy,
        idx: u32,
        pos: Pos,
        mut log: Option<&mut StepLog>,
    ) {
        let r = &tr.reqs[idx as usize];
        let keys = r.input(&tr.keys);
        let mut p = 0usize;
        while p < keys.len() && self.core.cached[keys[p] as usize] {
            p += 1;
        }
        let n = keys.len() as u64;
        let pu = p as u64;
        if r.t >= self.window.0 && r.t < self.window.1 {
            let s = &mut self.stats;
            s.hits += pu;
            s.accesses += n;
            s.hit_depth_sum += pu * pu.saturating_sub(1) / 2;
            s.access_depth_sum += n * n.saturating_sub(1) / 2;
            let b = ((r.t - self.window.0) / BUCKET_SECS) as usize;
            if s.buckets.len() <= b {
                s.buckets.resize(b + 1, Bucket::default());
            }
            let bk = &mut s.buckets[b];
            bk.hits += pu;
            bk.accesses += n;
            bk.hit_depth_sum += pu * pu.saturating_sub(1) / 2;
            bk.access_depth_sum += n * n.saturating_sub(1) / 2;
        }
        if let Some(l) = log.as_deref_mut() {
            l.arrival = true;
            l.req = idx;
            l.hit_blocks = p as u32;
        }
        for &k in &keys[p..] {
            self.add(tr, k, idx, r.t);
            if let Some(l) = log.as_deref_mut() {
                l.inserted.push(k);
            }
        }
        for j in (0..keys.len()).rev() {
            let k = keys[j];
            let c = &mut self.core;
            c.last_touch[k as usize] = r.t;
            c.n_access[k as usize] += 1;
            c.owner[k as usize] = idx;
            if j >= p {
                pol.on_insert(&mut self.core, tr, k, pos);
            } else {
                pol.on_access(&mut self.core, tr, k, pos);
            }
        }
        for &k in keys {
            let c = &mut self.core;
            c.refc[k as usize] += 1;
            if c.refc[k as usize] == 1 && c.children[k as usize] == 0 {
                c.set_remove(k);
            }
        }
        self.evict_to_capacity(tr, pol, pos, &mut log);
        self.done.push(Reverse((OrdF64(r.done), idx)));
    }

    fn complete(
        &mut self,
        tr: &Trace,
        pol: &mut dyn Policy,
        idx: u32,
        pos: Pos,
        mut log: Option<&mut StepLog>,
    ) {
        let r = &tr.reqs[idx as usize];
        if let Some(l) = log.as_deref_mut() {
            l.arrival = false;
            l.req = idx;
        }
        for &k in r.output(&tr.keys) {
            if !self.core.cached[k as usize] {
                self.add(tr, k, idx, pos.t);
                if let Some(l) = log.as_deref_mut() {
                    l.inserted.push(k);
                }
                pol.on_insert(&mut self.core, tr, k, pos);
            }
        }
        for &k in r.input(&tr.keys) {
            let c = &mut self.core;
            c.refc[k as usize] -= 1;
            if c.refc[k as usize] == 0 && c.children[k as usize] == 0 {
                c.set_add(k);
                pol.on_evictable(&self.core, tr, k);
            }
        }
        for &k in r.chain(&tr.keys).iter().rev() {
            let c = &mut self.core;
            if c.cached[k as usize] {
                c.last_touch[k as usize] = pos.t;
                c.owner[k as usize] = idx;
                pol.on_release(&mut self.core, tr, k, pos);
            }
        }
        self.evict_to_capacity(tr, pol, pos, &mut log);
    }
}
