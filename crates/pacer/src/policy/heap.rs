//! Priority policies over the evictable set, kept in a lazily invalidated min-heap.

use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::Trace;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub type Prio = (u64, u64, u32);

pub struct LazyHeap {
    heap: BinaryHeap<Reverse<(Prio, u32, u32)>>,
    ver: Vec<u32>,
    prio: Vec<Prio>,
    live: Vec<bool>,
}

impl LazyHeap {
    pub fn new(n: u32) -> Self {
        LazyHeap {
            heap: BinaryHeap::new(),
            ver: vec![0; n as usize],
            prio: vec![(0, 0, 0); n as usize],
            live: vec![false; n as usize],
        }
    }

    pub fn set(&mut self, k: u32, p: Prio) {
        let i = k as usize;
        self.ver[i] = self.ver[i].wrapping_add(1);
        self.prio[i] = p;
        self.live[i] = true;
        self.heap.push(Reverse((p, k, self.ver[i])));
    }

    pub fn invalidate(&mut self, k: u32) {
        let i = k as usize;
        self.ver[i] = self.ver[i].wrapping_add(1);
        self.live[i] = false;
    }

    pub fn repush(&mut self, k: u32) {
        let i = k as usize;
        if self.live[i] {
            self.heap.push(Reverse((self.prio[i], k, self.ver[i])));
        }
    }

    pub fn pop_valid(&mut self, core: &Core) -> Option<u32> {
        if self.heap.len() > 4 * core.evictable_set.len() + (1 << 20) {
            self.rebuild(core);
        }
        while let Some(Reverse((_, k, v))) = self.heap.pop() {
            if v == self.ver[k as usize] && core.evictable(k) {
                return Some(k);
            }
        }
        None
    }

    pub fn peek_valid(&mut self, core: &Core) -> Option<u32> {
        if self.heap.len() > 4 * core.evictable_set.len() + (1 << 20) {
            self.rebuild(core);
        }
        while let Some(Reverse((_, k, v))) = self.heap.peek().copied() {
            if v == self.ver[k as usize] && core.evictable(k) {
                return Some(k);
            }
            self.heap.pop();
        }
        None
    }

    pub fn pop(&mut self) {
        self.heap.pop();
    }

    fn rebuild(&mut self, core: &Core) {
        let items: Vec<_> = core
            .evictable_set
            .iter()
            .filter(|&&k| self.live[k as usize])
            .map(|&k| Reverse((self.prio[k as usize], k, self.ver[k as usize])))
            .collect();
        self.heap = BinaryHeap::from(items);
    }
}

pub struct Lru {
    h: LazyHeap,
    seq: u64,
}

impl Lru {
    pub fn new(n: u32) -> Self {
        Lru {
            h: LazyHeap::new(n),
            seq: 0,
        }
    }
    fn touch(&mut self, k: u32) {
        self.seq += 1;
        self.h.set(k, (self.seq, 0, 0));
    }
}

impl Policy for Lru {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.touch(k)
    }
    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.touch(k)
    }
    fn on_release(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.touch(k)
    }
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.h.repush(k)
    }
    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        self.h.pop_valid(c)
    }
}

pub struct Fifo {
    h: LazyHeap,
    seq: u64,
}

impl Fifo {
    pub fn new(n: u32) -> Self {
        Fifo {
            h: LazyHeap::new(n),
            seq: 0,
        }
    }
}

impl Policy for Fifo {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.seq += 1;
        self.h.set(k, (self.seq, 0, 0));
    }
    fn on_access(&mut self, _c: &mut Core, _t: &Trace, _k: u32, _p: Pos) {}
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.h.repush(k)
    }
    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        self.h.pop_valid(c)
    }
}

pub struct Lfu {
    h: LazyHeap,
    count: Vec<u64>,
    seq: u64,
}

impl Lfu {
    pub fn new(n: u32) -> Self {
        Lfu {
            h: LazyHeap::new(n),
            count: vec![0; n as usize],
            seq: 0,
        }
    }
}

impl Policy for Lfu {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.seq += 1;
        self.count[k as usize] = 1;
        self.h.set(k, (1, self.seq, 0));
    }
    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.seq += 1;
        self.count[k as usize] += 1;
        self.h.set(k, (self.count[k as usize], self.seq, 0));
    }
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.h.repush(k)
    }
    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        self.h.pop_valid(c)
    }
}

/// Offline MIN: evict the block whose next use is farthest away; ties go to the deeper block,
/// then to the larger key. Under prefix closure a parent is never used later than its child,
/// so the farthest block is always a leaf (or tied with one).
pub struct Belady {
    h: LazyHeap,
}

impl Belady {
    pub fn new(n: u32) -> Self {
        Belady {
            h: LazyHeap::new(n),
        }
    }
    fn touch(&mut self, tr: &Trace, k: u32, p: Pos) {
        let nu = tr.next_use(k, p.next) as u64;
        let n = tr.reqs.len() as u64;
        let d = tr.key_depth[k as usize] as u64;
        self.h.set(k, (n - nu, u32::MAX as u64 - d, u32::MAX - k));
    }
}

impl Policy for Belady {
    fn on_insert(&mut self, _c: &mut Core, t: &Trace, k: u32, p: Pos) {
        self.touch(t, k, p)
    }
    fn on_access(&mut self, _c: &mut Core, t: &Trace, k: u32, p: Pos) {
        self.touch(t, k, p)
    }
    fn on_release(&mut self, _c: &mut Core, t: &Trace, k: u32, p: Pos) {
        self.touch(t, k, p)
    }
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.h.repush(k)
    }
    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        self.h.pop_valid(c)
    }
}

/// GreedyDual-Size-Frequency with unit size and a per-block recompute cost that grows with
/// prefix depth (attention over a longer context): H = L + freq * (c0 + c1 * depth).
pub struct Gdsf {
    h: LazyHeap,
    freq: Vec<u32>,
    hval: Vec<f64>,
    l: f64,
    c0: f64,
    c1: f64,
    seq: u64,
}

impl Gdsf {
    pub fn new(n: u32, c0: f64, c1: f64) -> Self {
        Gdsf {
            h: LazyHeap::new(n),
            freq: vec![0; n as usize],
            hval: vec![0.0; n as usize],
            l: 0.0,
            c0,
            c1,
            seq: 0,
        }
    }
    fn touch(&mut self, tr: &Trace, k: u32, inc: bool, reset: bool) {
        let i = k as usize;
        if reset {
            self.freq[i] = 0;
        }
        if inc {
            self.freq[i] += 1;
        }
        let cost = self.c0 + self.c1 * tr.key_depth[i] as f64;
        let hv = self.l + self.freq[i] as f64 * cost;
        self.hval[i] = hv;
        self.seq += 1;
        self.h.set(k, (hv.to_bits(), self.seq, 0));
    }
}

impl Policy for Gdsf {
    fn on_insert(&mut self, _c: &mut Core, t: &Trace, k: u32, _p: Pos) {
        self.touch(t, k, true, true)
    }
    fn on_access(&mut self, _c: &mut Core, t: &Trace, k: u32, _p: Pos) {
        self.touch(t, k, true, false)
    }
    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.h.repush(k)
    }
    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        self.h.pop_valid(c)
    }
    fn on_evict(&mut self, _c: &Core, _t: &Trace, k: u32) {
        self.l = self.l.max(self.hval[k as usize]);
    }
}
