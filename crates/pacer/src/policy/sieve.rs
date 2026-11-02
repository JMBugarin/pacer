//! SIEVE (NSDI'24) adapted to prefix closure: the hand skips blocks that are not evictable.

use super::Policy;
use crate::sim::{Core, Pos};
use crate::trace::{Trace, NONE};

pub struct Sieve {
    prev: Vec<u32>,
    next: Vec<u32>,
    visited: Vec<bool>,
    head: u32,
    tail: u32,
    hand: u32,
    len: usize,
}

impl Sieve {
    pub fn new(n: u32) -> Self {
        let n = n as usize;
        Sieve {
            prev: vec![NONE; n],
            next: vec![NONE; n],
            visited: vec![false; n],
            head: NONE,
            tail: NONE,
            hand: NONE,
            len: 0,
        }
    }
}

impl Policy for Sieve {
    fn on_insert(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        let i = k as usize;
        self.next[i] = self.head;
        self.prev[i] = NONE;
        if self.head != NONE {
            self.prev[self.head as usize] = k;
        }
        self.head = k;
        if self.tail == NONE {
            self.tail = k;
        }
        self.visited[i] = false;
        self.len += 1;
    }

    fn on_access(&mut self, _c: &mut Core, _t: &Trace, k: u32, _p: Pos) {
        self.visited[k as usize] = true;
    }

    fn victim(&mut self, c: &mut Core, _t: &Trace, _p: Pos) -> Option<u32> {
        let mut o = if self.hand != NONE {
            self.hand
        } else {
            self.tail
        };
        let limit = 2 * self.len + 2;
        let mut steps = 0;
        while o != NONE && steps <= limit {
            let i = o as usize;
            if self.visited[i] {
                self.visited[i] = false;
            } else if c.evictable(o) {
                self.hand = self.prev[i];
                return Some(o);
            }
            o = if self.prev[i] != NONE {
                self.prev[i]
            } else {
                self.tail
            };
            steps += 1;
        }
        None
    }

    fn on_evict(&mut self, _c: &Core, _t: &Trace, k: u32) {
        let i = k as usize;
        let (p, n) = (self.prev[i], self.next[i]);
        if p != NONE {
            self.next[p as usize] = n;
        } else {
            self.head = n;
        }
        if n != NONE {
            self.prev[n as usize] = p;
        } else {
            self.tail = p;
        }
        self.prev[i] = NONE;
        self.next[i] = NONE;
        self.len -= 1;
    }
}
