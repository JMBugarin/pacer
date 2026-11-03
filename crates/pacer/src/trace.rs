use anyhow::{bail, Context, Result};
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const BLOCK: u32 = 16;
pub const NONE: u32 = u32::MAX;
pub const DEFAULT_TPOT: f64 = 0.02;
pub const TYPES: [&str; 7] = [
    "text", "image", "search", "file", "api", "coder", "thinking",
];

#[derive(Deserialize)]
struct Row {
    chat_id: i64,
    parent_chat_id: i64,
    timestamp: f64,
    input_length: u32,
    output_length: u32,
    #[serde(rename = "type")]
    rtype: String,
    turn: u32,
    hash_ids: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Req {
    pub t: f64,
    pub done: f64,
    pub chat_id: i64,
    pub parent: u32,
    pub first_child: u32,
    pub root: u32,
    pub input_len: u32,
    pub output_len: u32,
    pub rtype: u8,
    pub turn: u32,
    pub start: u32,
    pub n_in: u32,
    pub n_out: u32,
}

impl Req {
    pub fn input<'a>(&self, keys: &'a [u32]) -> &'a [u32] {
        &keys[self.start as usize..(self.start + self.n_in) as usize]
    }
    pub fn output<'a>(&self, keys: &'a [u32]) -> &'a [u32] {
        let s = (self.start + self.n_in) as usize;
        &keys[s..s + self.n_out as usize]
    }
    pub fn chain<'a>(&self, keys: &'a [u32]) -> &'a [u32] {
        &keys[self.start as usize..(self.start + self.n_in + self.n_out) as usize]
    }
}

pub struct Trace {
    pub name: String,
    pub reqs: Vec<Req>,
    pub keys: Vec<u32>,
    pub n_keys: u32,
    pub key_parent: Vec<u32>,
    pub key_depth: Vec<u32>,
    pub key_output: Vec<bool>,
    pub use_off: Vec<u32>,
    pub use_req: Vec<u32>,
    pub duration: f64,
    /// Arrivals in [done - 60 s, done) for each request: the load feature at release time. Arrivals
    /// at exactly `done` are processed after the completion, so they are not counted.
    pub arrivals_at_done: Vec<u32>,
}

pub struct RawReq {
    pub chat_id: i64,
    pub parent_chat_id: i64,
    pub timestamp: f64,
    pub input_length: u32,
    pub output_length: u32,
    pub rtype: u8,
    pub turn: u32,
    pub hash_ids: Vec<u64>,
}

pub fn type_code(s: &str) -> u8 {
    TYPES.iter().position(|t| *t == s).unwrap_or(TYPES.len()) as u8
}

pub fn read_jsonl(path: &Path, limit: Option<usize>) -> Result<Vec<RawReq>> {
    read_jsonl_from(path, 0.0, limit)
}

/// Reads rows with timestamp >= `start`, at most `limit` of them.
pub fn read_jsonl_from(path: &Path, start: f64, limit: Option<usize>) -> Result<Vec<RawReq>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::with_capacity(1 << 20, f).lines().enumerate() {
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Row = serde_json::from_str(&line).with_context(|| format!("line {}", i + 1))?;
        if r.timestamp < start {
            continue;
        }
        out.push(RawReq {
            chat_id: r.chat_id,
            parent_chat_id: r.parent_chat_id,
            timestamp: r.timestamp,
            input_length: r.input_length,
            output_length: r.output_length,
            rtype: type_code(&r.rtype),
            turn: r.turn,
            hash_ids: r.hash_ids,
        });
    }
    Ok(out)
}

pub fn load(path: &Path, limit: Option<usize>, tpot: f64) -> Result<Trace> {
    load_from(path, 0.0, limit, tpot)
}

/// Loads the requests that arrive at or after `start`; parents before `start` are unknown, so
/// those turns start new sessions.
pub fn load_from(path: &Path, start: f64, limit: Option<usize>, tpot: f64) -> Result<Trace> {
    let rows = read_jsonl_from(path, start, limit)?;
    let name = path
        .file_stem()
        .map(|s| {
            s.to_string_lossy()
                .replace("qwen_", "")
                .replace("_blksz_16", "")
        })
        .unwrap_or_default();
    build(name, rows, tpot)
}

/// Same preprocessing as python/pacer/trace.py: prefix-chain re-keying of full input blocks,
/// decode-completed output blocks taken from the first child's chain when the child's prompt is
/// long enough to hold the whole output verbatim (otherwise fresh keys: a thinking model's
/// follow-up drops the reasoning, so its blocks are not the ones the server decoded), and a
/// completion time of `tpot * output_len`, capped so it lands before the first child arrives.
pub fn build(name: String, rows: Vec<RawReq>, tpot: f64) -> Result<Trace> {
    for w in rows.windows(2) {
        if w[1].timestamp < w[0].timestamp {
            bail!("trace is not sorted by timestamp");
        }
    }
    let n = rows.len();
    let mut intern: FxHashMap<u64, u32> = FxHashMap::default();
    let mut key_parent: Vec<u32> = Vec::new();
    let mut key_depth: Vec<u32> = Vec::new();
    let mut chains: Vec<Vec<u32>> = Vec::with_capacity(n);
    for r in &rows {
        let f_in = (r.input_length / BLOCK) as usize;
        let mut prev = NONE;
        let mut chain = Vec::with_capacity(f_in.min(r.hash_ids.len()));
        for (j, &h) in r.hash_ids.iter().take(f_in).enumerate() {
            if h >= 1 << 32 {
                bail!("hash id {h} does not fit in 32 bits");
            }
            let packed = ((prev.wrapping_add(1) as u64) << 32) | h;
            let next = intern.len() as u32;
            let k = *intern.entry(packed).or_insert_with(|| {
                key_parent.push(prev);
                key_depth.push(j as u32);
                next
            });
            chain.push(k);
            prev = k;
        }
        chains.push(chain);
    }

    let mut by_chat: FxHashMap<i64, u32> = FxHashMap::default();
    let mut reqs: Vec<Req> = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
        reqs.push(Req {
            t: r.timestamp,
            done: r.timestamp,
            chat_id: r.chat_id,
            parent: NONE,
            first_child: NONE,
            root: i as u32,
            input_len: r.input_length,
            output_len: r.output_length,
            rtype: r.rtype,
            turn: r.turn,
            start: 0,
            n_in: chains[i].len() as u32,
            n_out: 0,
        });
        by_chat.insert(r.chat_id, i as u32);
    }
    for i in 0..n {
        let pc = rows[i].parent_chat_id;
        if pc == -1 {
            continue;
        }
        if let Some(&p) = by_chat.get(&pc) {
            if (p as usize) < i {
                reqs[i].parent = p;
                reqs[i].root = reqs[p as usize].root;
                let fc = reqs[p as usize].first_child;
                if fc == NONE || reqs[i].t < reqs[fc as usize].t {
                    reqs[p as usize].first_child = i as u32;
                }
            }
        }
    }

    let mut next_key = intern.len() as u32;
    let mut key_output = vec![false; next_key as usize];
    let mut keys: Vec<u32> = Vec::new();
    for i in 0..n {
        let f_in = chains[i].len();
        let f_tot = ((reqs[i].input_len + reqs[i].output_len) / BLOCK) as usize;
        reqs[i].start = keys.len() as u32;
        keys.extend_from_slice(&chains[i]);
        let mut out: Vec<u32> = Vec::new();
        let c = reqs[i].first_child;
        let verbatim =
            c != NONE && reqs[c as usize].input_len >= reqs[i].input_len + reqs[i].output_len;
        if verbatim && f_tot > f_in {
            let child = &chains[c as usize];
            let cf = child.len();
            let matches = f_in == 0 || (cf >= f_in && child[f_in - 1] == chains[i][f_in - 1]);
            if matches && cf > f_in {
                out.extend_from_slice(&child[f_in..f_tot.min(cf)]);
            }
        }
        for &k in &out {
            key_output[k as usize] = true;
        }
        while f_in + out.len() < f_tot {
            let prev = if f_in + out.len() == 0 {
                NONE
            } else if out.is_empty() {
                chains[i][f_in - 1]
            } else {
                *out.last().unwrap()
            };
            key_parent.push(prev);
            key_depth.push((f_in + out.len()) as u32);
            key_output.push(true);
            out.push(next_key);
            next_key += 1;
        }
        reqs[i].n_out = out.len() as u32;
        keys.extend_from_slice(&out);

        let mut dur = tpot * reqs[i].output_len as f64;
        if c != NONE {
            dur = dur.min((reqs[c as usize].t - reqs[i].t - 1e-3).max(0.0));
        }
        reqs[i].done = reqs[i].t + dur;
    }

    let nk = next_key as usize;
    let mut use_off = vec![0u32; nk + 1];
    for r in &reqs {
        for &k in r.input(&keys) {
            use_off[k as usize + 1] += 1;
        }
    }
    for k in 0..nk {
        use_off[k + 1] += use_off[k];
    }
    let mut fill = use_off.clone();
    let mut use_req = vec![0u32; use_off[nk] as usize];
    for (i, r) in reqs.iter().enumerate() {
        for &k in r.input(&keys) {
            use_req[fill[k as usize] as usize] = i as u32;
            fill[k as usize] += 1;
        }
    }
    let duration = reqs.last().map(|r| r.t).unwrap_or(0.0);
    let arrivals_at_done = reqs
        .iter()
        .map(|r| {
            let a = reqs.partition_point(|q| q.t < r.done - 60.0);
            let b = reqs.partition_point(|q| q.t < r.done);
            b.saturating_sub(a) as u32
        })
        .collect();
    Ok(Trace {
        name,
        reqs,
        keys,
        n_keys: next_key,
        key_parent,
        key_depth,
        key_output,
        use_off,
        use_req,
        duration,
        arrivals_at_done,
    })
}

impl Trace {
    pub fn uses(&self, k: u32) -> &[u32] {
        &self.use_req[self.use_off[k as usize] as usize..self.use_off[k as usize + 1] as usize]
    }

    /// Index of the first request at or after `from` whose input contains k, or reqs.len().
    pub fn next_use(&self, k: u32, from: u32) -> u32 {
        let u = self.uses(k);
        let p = u.partition_point(|&j| j < from);
        if p < u.len() {
            u[p]
        } else {
            self.reqs.len() as u32
        }
    }

    /// First arrival not yet processed when request `i` completes: arrivals before the
    /// completion time, and request `i` itself, are past.
    pub fn first_after_completion(&self, i: u32) -> u32 {
        let d = self.reqs[i as usize].done;
        (self.reqs.partition_point(|r| r.t < d) as u32).max(i + 1)
    }

    pub fn input_accesses(&self) -> u64 {
        self.reqs.iter().map(|r| r.n_in as u64).sum()
    }
}
