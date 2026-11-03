//! ARC (FAST'03) adapted to prefix closure: REPLACE takes the least recent *evictable* block
//! of the chosen list, falling back to the other list. Ghost lists are bounded as in ARC
//! (|T1| + |B1| <= c, total <= 2c). T1 and T2 keep their evictable blocks in lazy heaps
//! ordered by recency, so pinned and interior blocks are never scanned.

use super::heap::LazyHeap;
use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::Trace;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq)]
enum Loc {
    None,
    T1,
    T2,
    B1,
    B2,
}

pub struct Arc {
    refresh_on_release: bool,
    c: f64,
    p: f64,
    seq: u64,
    stamp: Vec<u64>,
    loc: Vec<Loc>,
    t1: LazyHeap,
    t2: LazyHeap,
    n_t1: usize,
    n_t2: usize,
    b1: BTreeMap<u64, u32>,
    b2: BTreeMap<u64, u32>,
}

impl Arc {
    pub fn new(n: u32, cap: u64, refresh_on_release: bool) -> Self {
        Arc {
            refresh_on_release,
            c: cap as f64,
            p: 0.0,
            seq: 0,
            stamp: vec![0; n as usize],
            loc: vec![Loc::None; n as usize],
            t1: LazyHeap::new(n),
            t2: LazyHeap::new(n),
            n_t1: 0,
            n_t2: 0,
            b1: BTreeMap::new(),
            b2: BTreeMap::new(),
        }
    }

    fn detach(&mut self, k: u32) {
        let i = k as usize;
        match self.loc[i] {
            Loc::T1 => {
                self.t1.invalidate(k);
                self.n_t1 -= 1;
            }
            Loc::T2 => {
                self.t2.invalidate(k);
                self.n_t2 -= 1;
            }
            Loc::B1 => {
                self.b1.remove(&self.stamp[i]);
            }
            Loc::B2 => {
                self.b2.remove(&self.stamp[i]);
            }
            Loc::None => {}
        }
        self.loc[i] = Loc::None;
    }

    fn attach(&mut self, k: u32, l: Loc) {
        self.seq += 1;
        let i = k as usize;
        self.stamp[i] = self.seq;
        self.loc[i] = l;
        match l {
            Loc::T1 => {
                self.t1.set(k, (self.seq, 0, 0));
                self.n_t1 += 1;
            }
            Loc::T2 => {
                self.t2.set(k, (self.seq, 0, 0));
                self.n_t2 += 1;
            }
            Loc::B1 => {
                self.b1.insert(self.seq, k);
            }
            Loc::B2 => {
                self.b2.insert(self.seq, k);
            }
            Loc::None => {}
        }
    }

    fn trim(&mut self) {
        let c = self.c as usize;
        while self.n_t1 + self.b1.len() > c {
            match self.b1.pop_first() {
                Some((_, k)) => self.loc[k as usize] = Loc::None,
                None => break,
            }
        }
        while self.n_t1 + self.n_t2 + self.b1.len() + self.b2.len() > 2 * c {
            match self.b2.pop_first() {
                Some((_, k)) => self.loc[k as usize] = Loc::None,
                None => break,
            }
        }
    }
}

impl Policy for Arc {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        match self.loc[k as usize] {
            Loc::B1 => {
                let d = (self.b2.len() as f64 / self.b1.len() as f64).max(1.0);
                self.p = (self.p + d).min(self.c);
                self.detach(k);
                self.attach(k, Loc::T2);
            }
            Loc::B2 => {
                let d = (self.b1.len() as f64 / self.b2.len() as f64).max(1.0);
                self.p = (self.p - d).max(0.0);
                self.detach(k);
                self.attach(k, Loc::T2);
            }
            _ => {
                self.detach(k);
                self.attach(k, Loc::T1);
            }
        }
        self.trim();
    }

    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        if matches!(self.loc[k as usize], Loc::T1 | Loc::T2) {
            self.detach(k);
            self.attach(k, Loc::T2);
        }
    }

    /// Optional: a release (the request finished) refreshes recency within the block's list,
    /// as vLLM's free-queue order does, without promoting T1 to T2.
    fn on_release(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        if self.refresh_on_release {
            let l = self.loc[k as usize];
            if matches!(l, Loc::T1 | Loc::T2) {
                self.detach(k);
                self.attach(k, l);
            }
        }
    }

    fn on_evictable(&mut self, _c: &Core, _t: &Trace, k: u32) {
        match self.loc[k as usize] {
            Loc::T1 => self.t1.repush(k),
            Loc::T2 => self.t2.repush(k),
            _ => {}
        }
    }

    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        let prefer_t1 = self.n_t1 > 0 && self.n_t1 as f64 > self.p;
        if prefer_t1 {
            self.t1.peek_valid(c).or_else(|| self.t2.peek_valid(c))
        } else {
            self.t2.peek_valid(c).or_else(|| self.t1.peek_valid(c))
        }
    }

    fn on_evict(&mut self, _c: &Core, _t: &Trace, k: u32) {
        let l = self.loc[k as usize];
        self.detach(k);
        match l {
            Loc::T1 => self.attach(k, Loc::B1),
            Loc::T2 => self.attach(k, Loc::B2),
            _ => {}
        }
        self.trim();
    }
}
