use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use pacer::policy::{Params, Resources, Scorer};
use pacer::run::{self, RunSpec};
use pacer::trace::{self, Trace};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "pacer", about = "LLM prefix KV-cache eviction simulator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print trace statistics.
    Info {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, default_value_t = trace::DEFAULT_TPOT)]
        tpot: f64,
    },
    /// Run one policy and print a JSON result.
    Sim {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        policy: String,
        #[arg(long, default_value_t = 0.02)]
        cap: f64,
        #[arg(long, value_parser = parse_window, default_value = "0,inf")]
        window: (f64, f64),
        #[arg(long = "param", value_parser = parse_kv)]
        params: Vec<(String, f64)>,
        #[arg(long = "model", value_parser = parse_model)]
        models: Vec<(String, String)>,
        /// ATC'25 workload fit (JSON) for this trace.
        #[arg(long)]
        wa_fit: Option<String>,
        #[arg(long, default_value_t = trace::DEFAULT_TPOT)]
        tpot: f64,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        time: bool,
        #[arg(long)]
        buckets: bool,
    },
    /// Run a grid of (trace, capacity, policy) in parallel; writes JSON lines.
    Sweep {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Cache-independent training rows for the session-return hazard model (.npy, f32).
    Hazard {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, default_value_t = 3600.0)]
        cutoff: f64,
        #[arg(long, default_value_t = 0.02)]
        rate: f64,
        #[arg(long, default_value_t = 6)]
        per_touch: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = trace::DEFAULT_TPOT)]
        tpot: f64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Log sampled eviction candidates with features and next-use labels (.npy, f32).
    Label {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, default_value = "age")]
        driver: String,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 0.02)]
        cap: f64,
        #[arg(long, value_parser = parse_window, default_value = "0,3600")]
        window: (f64, f64),
        #[arg(long, default_value_t = 0.05)]
        rate: f64,
        #[arg(long, default_value_t = 64)]
        k: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = trace::DEFAULT_TPOT)]
        tpot: f64,
        #[arg(long)]
        out: PathBuf,
    },
}

fn parse_window(s: &str) -> Result<(f64, f64), String> {
    let (a, b) = s.split_once(',').ok_or("expected a,b")?;
    let f = |x: &str| -> Result<f64, String> {
        if x == "inf" {
            Ok(f64::INFINITY)
        } else {
            x.parse().map_err(|e| format!("{e}"))
        }
    };
    Ok((f(a)?, f(b)?))
}

fn parse_kv(s: &str) -> Result<(String, f64), String> {
    let (a, b) = s.split_once('=').ok_or("expected k=v")?;
    Ok((a.to_string(), b.parse().map_err(|e| format!("{e}"))?))
}

fn parse_model(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((a, b)) => Ok((a.to_string(), b.to_string())),
        None => Ok(("0".to_string(), s.to_string())),
    }
}

#[derive(Deserialize)]
struct SweepTrace {
    path: String,
    window: (f64, f64),
}

#[derive(Deserialize, Clone)]
struct SweepPolicy {
    name: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    params: Params,
    #[serde(default)]
    caps: Option<Vec<f64>>,
    #[serde(default)]
    traces: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct SweepSpec {
    traces: Vec<SweepTrace>,
    caps: Vec<f64>,
    policies: Vec<SweepPolicy>,
    #[serde(default)]
    models: BTreeMap<String, String>,
    /// ATC'25 workload fits per trace name (JSON from `python -m pacer.hazard --wa-out`).
    #[serde(default)]
    wa_fits: BTreeMap<String, String>,
    #[serde(default = "default_tpot")]
    tpot: f64,
    #[serde(default)]
    time: bool,
}

fn default_tpot() -> f64 {
    trace::DEFAULT_TPOT
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { trace: p, tpot } => {
            let tr = trace::load(&p, None, tpot)?;
            let n_out: u64 = tr.reqs.iter().map(|r| r.n_out as u64).sum();
            println!(
                "{}: {} requests, {} unique blocks ({} decode-completed), {} input block accesses, {:.0} s",
                tr.name,
                tr.reqs.len(),
                tr.n_keys,
                n_out,
                tr.input_accesses(),
                tr.duration
            );
        }
        Cmd::Sim {
            trace: p,
            policy,
            cap,
            window,
            params,
            models,
            wa_fit,
            tpot,
            limit,
            time,
            buckets,
        } => {
            let tr = Arc::new(trace::load(&p, limit, tpot)?);
            let fits = wa_fit
                .map(|f| BTreeMap::from([(tr.name.clone(), f)]))
                .unwrap_or_default();
            let res = Resources {
                models: run::load_models(&models.into_iter().collect())?,
                wa_fits: run::load_wa_fits(&fits)?,
            };
            let spec = RunSpec {
                label: policy.clone(),
                policy,
                params: params.into_iter().collect(),
                cap_frac: cap,
                window,
                time_victims: time,
            };
            let mut r = run::run(&tr, &spec, &res)?;
            if !buckets {
                r.stats.buckets.clear();
            }
            println!("{}", serde_json::to_string(&r)?);
        }
        Cmd::Sweep { spec, out, threads } => {
            let s: SweepSpec =
                serde_json::from_str(&std::fs::read_to_string(&spec).context("read spec")?)?;
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build_global()
                    .ok();
            }
            let traces: Vec<(Arc<Trace>, (f64, f64))> = s
                .traces
                .par_iter()
                .map(|t| {
                    Ok((
                        Arc::new(trace::load(std::path::Path::new(&t.path), None, s.tpot)?),
                        t.window,
                    ))
                })
                .collect::<Result<_>>()?;
            let res = Resources {
                models: run::load_models(&s.models)?,
                wa_fits: run::load_wa_fits(&s.wa_fits)?,
            };
            let mut jobs = Vec::new();
            for (ti, (tr, w)) in traces.iter().enumerate() {
                for p in &s.policies {
                    if let Some(ts) = &p.traces {
                        if !ts.iter().any(|n| *n == tr.name) {
                            continue;
                        }
                    }
                    for &c in p.caps.as_ref().unwrap_or(&s.caps) {
                        jobs.push((ti, *w, p.clone(), c));
                    }
                }
            }
            eprintln!("{} runs", jobs.len());
            let file = std::sync::Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out)?));
            let done = std::sync::atomic::AtomicUsize::new(0);
            let total = jobs.len();
            jobs.par_iter()
                .try_for_each(|(ti, w, p, c)| -> Result<()> {
                    let tr = &traces[*ti].0;
                    let spec = RunSpec {
                        policy: p.name.clone(),
                        label: p.label.clone().unwrap_or_else(|| p.name.clone()),
                        params: p.params.clone(),
                        cap_frac: *c,
                        window: *w,
                        time_victims: s.time,
                    };
                    let r = run::run(tr, &spec, &res)?;
                    let line = serde_json::to_string(&r)?;
                    let mut f = file.lock().unwrap();
                    writeln!(f, "{line}")?;
                    f.flush()?;
                    let n = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    eprintln!(
                        "[{n}/{total}] {} {} cap={} hit={:.4} ({:.1}s)",
                        r.trace, r.label, c, r.hit_ratio, r.wall_s
                    );
                    Ok(())
                })?;
        }
        Cmd::Hazard {
            trace: p,
            cutoff,
            rate,
            per_touch,
            seed,
            tpot,
            out,
        } => {
            let tr = Arc::new(trace::load(&p, None, tpot)?);
            let (rows, n) = run::hazard_rows(&tr, cutoff, rate, per_touch, seed);
            pacer::npy::write_f32(&out, &rows, n, pacer::features::N_HAZ + 2)?;
            eprintln!("{}: {} hazard rows before t={}", tr.name, n, cutoff);
        }
        Cmd::Label {
            trace: p,
            driver,
            model,
            cap,
            window,
            rate,
            k,
            seed,
            tpot,
            out,
        } => {
            let tr = Arc::new(trace::load(&p, None, tpot)?);
            let scorer = match driver.as_str() {
                "age" => Scorer::Age,
                "oracle" => Scorer::Oracle,
                "model" => {
                    let m = model.context("--driver model needs --model")?;
                    Scorer::Learned(Arc::new(pacer::gbdt::Gbdt::load(&m)?))
                }
                d => bail!("unknown driver {d}"),
            };
            let (rows, n, stats) = run::label(&tr, scorer, k, cap, window, rate, seed);
            pacer::npy::write_f32(&out, &rows, n, pacer::policy::sampled::Logger::N_COLS)?;
            eprintln!(
                "{}: {} rows from {} decisions, driver hit ratio {:.4}",
                tr.name,
                n,
                stats.decisions,
                stats.hit_ratio()
            );
        }
    }
    Ok(())
}
