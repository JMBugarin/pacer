//! LRU with quick demotion of one-hit prefixes, the design recommended by "When Fancy Eviction
//! Fails" (Liu, Yu, Yang 2026): blocks that have not been re-accessed sit in a probation FIFO
//! and are evicted first while probation holds more than `small` of the cache; a second access
//! promotes a block to the LRU main queue.

use super::heap::LazyHeap;
use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::Trace;

pub struct LruQd {
    small_cap: u64,
    prob: LazyHeap,
    main: LazyHeap,
    in_main: Vec<bool>,
    cached_prob: u64,
    seq: u64,
}

impl LruQd {
    pub fn new(n: u32, cap: u64, small: f64) -> Self {
        LruQd {
            small_cap: ((cap as f64 * small) as u64).max(1),
            prob: LazyHeap::new(n),
            main: LazyHeap::new(n),
            in_main: vec![false; n as usize],
            cached_prob: 0,
            seq: 0,
        }
    }
}

impl Policy for LruQd {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.seq += 1;
        self.in_main[k as usize] = false;
        self.cached_prob += 1;
        self.main.invalidate(k);
        self.prob.set(k, (self.seq, 0, 0));
    }

    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.seq += 1;
        if !self.in_main[k as usize] {
            self.in_main[k as usize] = true;
            self.cached_prob -= 1;
            self.prob.invalidate(k);
        }
        self.main.set(k, (self.seq, 0, 0));
    }

    fn on_release(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        if self.in_main[k as usize] {
            self.seq += 1;
            self.main.set(k, (self.seq, 0, 0));
        }
    }

    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        if self.in_main[k as usize] {
            self.main.repush(k)
        } else {
            self.prob.repush(k)
        }
    }

    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        if self.cached_prob > self.small_cap {
            self.prob.pop_valid(c).or_else(|| self.main.pop_valid(c))
        } else {
            self.main.pop_valid(c).or_else(|| self.prob.pop_valid(c))
        }
    }

    fn on_evict(&mut self, _c: &Core, _t: &Trace, k: u32) {
        if !self.in_main[k as usize] {
            self.cached_prob -= 1;
        }
    }
}
