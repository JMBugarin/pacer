//! S3-FIFO (SOSP'23, Algorithm 1) adapted to prefix closure: a tail block that is not
//! evictable is moved back to the head of its queue without changing its frequency.

use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::Trace;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

pub struct S3Fifo {
    small_cap: usize,
    main_cap: usize,
    threshold: u8,
    s: VecDeque<u32>,
    m: VecDeque<u32>,
    freq: Vec<u8>,
    ghost: FxHashMap<u32, u64>,
    ghost_q: VecDeque<(u32, u64)>,
    ghost_ins: u64,
}

impl S3Fifo {
    pub fn new(n: u32, cap: u64, small_ratio: f64, threshold: u8) -> Self {
        let small_cap = ((cap as f64 * small_ratio) as usize).max(1);
        S3Fifo {
            small_cap,
            main_cap: (cap as usize).saturating_sub(small_cap),
            threshold,
            s: VecDeque::new(),
            m: VecDeque::new(),
            freq: vec![0; n as usize],
            ghost: FxHashMap::default(),
            ghost_q: VecDeque::new(),
            ghost_ins: 0,
        }
    }

    fn ghost_add(&mut self, k: u32) {
        self.ghost_ins += 1;
        self.ghost.insert(k, self.ghost_ins);
        self.ghost_q.push_back((k, self.ghost_ins));
        while let Some(&(gk, gt)) = self.ghost_q.front() {
            if gt + self.main_cap as u64 > self.ghost_ins {
                break;
            }
            self.ghost_q.pop_front();
            if self.ghost.get(&gk) == Some(&gt) {
                self.ghost.remove(&gk);
            }
        }
    }

    fn evict_s(&mut self, c: &Core) -> Option<u32> {
        let mut skips = 0usize;
        while !self.s.is_empty() && skips <= self.s.len() {
            let t = self.s.pop_back().unwrap();
            if self.freq[t as usize] >= self.threshold {
                self.m.push_front(t);
                if self.m.len() > self.main_cap {
                    if let Some(v) = self.evict_m(c) {
                        return Some(v);
                    }
                }
            } else if !c.evictable(t) {
                self.s.push_front(t);
                skips += 1;
            } else {
                self.ghost_add(t);
                return Some(t);
            }
        }
        None
    }

    fn evict_m(&mut self, c: &Core) -> Option<u32> {
        let limit = 4 * self.m.len() + 4;
        let mut steps = 0usize;
        while !self.m.is_empty() && steps <= limit {
            let t = self.m.pop_back().unwrap();
            let i = t as usize;
            if self.freq[i] > 0 {
                self.freq[i] -= 1;
                self.m.push_front(t);
            } else if !c.evictable(t) {
                self.m.push_front(t);
            } else {
                return Some(t);
            }
            steps += 1;
        }
        None
    }
}

impl Policy for S3Fifo {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.freq[k as usize] = 0;
        if self.ghost.remove(&k).is_some() {
            self.m.push_front(k);
        } else {
            self.s.push_front(k);
        }
    }

    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        let f = &mut self.freq[k as usize];
        *f = (*f + 1).min(3);
    }

    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        if self.s.len() >= self.small_cap || self.m.is_empty() {
            self.evict_s(c).or_else(|| self.evict_m(c))
        } else {
            self.evict_m(c).or_else(|| self.evict_s(c))
        }
    }
}
