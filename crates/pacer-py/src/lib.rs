//! Python bindings: the live MLX replay drives the same simulator and policies as the sweeps,
//! step by step, and mirrors each step's inserts and evictions on its KV block pool.

use pacer::policy::{self, Params, Policy, Resources};
use pacer::run;
use pacer::sim::{Sim, SimConfig, StepLog};
use pacer::trace::{self, Trace};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::sync::Arc;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyclass(name = "Trace", frozen)]
struct PyTrace {
    inner: Arc<Trace>,
}

#[pymethods]
impl PyTrace {
    #[getter]
    fn n_keys(&self) -> u32 {
        self.inner.n_keys
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.reqs.len()
    }

    fn request<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let tr = &self.inner;
        let r = tr
            .reqs
            .get(i)
            .ok_or_else(|| err("request index out of range"))?;
        let d = PyDict::new(py);
        d.set_item("t", r.t)?;
        d.set_item("done", r.done)?;
        d.set_item("input_len", r.input_len)?;
        d.set_item("output_len", r.output_len)?;
        d.set_item("turn", r.turn)?;
        d.set_item(
            "type",
            trace::TYPES
                .get(r.rtype as usize)
                .copied()
                .unwrap_or("other"),
        )?;
        d.set_item("input_keys", r.input(&tr.keys).to_vec())?;
        d.set_item("output_keys", r.output(&tr.keys).to_vec())?;
        Ok(d)
    }

    fn key_parent(&self, k: u32) -> i64 {
        let p = self.inner.key_parent[k as usize];
        if p == trace::NONE {
            -1
        } else {
            p as i64
        }
    }
}

#[pyfunction]
#[pyo3(signature = (path, tpot = trace::DEFAULT_TPOT, limit = None, start = 0.0))]
fn load_trace(path: &str, tpot: f64, limit: Option<usize>, start: f64) -> PyResult<PyTrace> {
    let tr = trace::load_from(std::path::Path::new(path), start, limit, tpot).map_err(err)?;
    Ok(PyTrace {
        inner: Arc::new(tr),
    })
}

fn resources(model: Option<&str>) -> PyResult<Resources> {
    let mut paths = BTreeMap::new();
    if let Some(m) = model {
        paths.insert("0".to_string(), m.to_string());
    }
    Ok(Resources {
        models: run::load_models(&paths).map_err(err)?,
        wa_fits: BTreeMap::new(),
    })
}

#[pyclass(unsendable)]
struct Stepper {
    sim: Sim,
    pol: Box<dyn Policy>,
}

#[pymethods]
impl Stepper {
    #[new]
    #[pyo3(signature = (trace, policy, cap_blocks, params = None, model = None))]
    fn new(
        trace: &PyTrace,
        policy: &str,
        cap_blocks: u64,
        params: Option<BTreeMap<String, f64>>,
        model: Option<&str>,
    ) -> PyResult<Self> {
        let params: Params = params.unwrap_or_default();
        let res = resources(model)?;
        let pol = policy::make(policy, &params, &trace.inner, cap_blocks, &res).map_err(err)?;
        let cfg = SimConfig {
            cap: cap_blocks,
            window: (0.0, f64::INFINITY),
            time_victims: false,
        };
        Ok(Stepper {
            sim: Sim::new(trace.inner.clone(), &cfg),
            pol,
        })
    }

    /// Advances one event. Returns None when the trace is exhausted, otherwise a dict with
    /// arrival (bool), req, hit_blocks, inserted and evicted key lists.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if self.sim.finished() {
            return Ok(None);
        }
        let mut log = StepLog::default();
        self.sim.step(self.pol.as_mut(), Some(&mut log));
        let d = PyDict::new(py);
        d.set_item("arrival", log.arrival)?;
        d.set_item("req", log.req)?;
        d.set_item("hit_blocks", log.hit_blocks)?;
        d.set_item("inserted", log.inserted)?;
        d.set_item("evicted", log.evicted)?;
        Ok(Some(d))
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.sim.stats;
        let d = PyDict::new(py);
        d.set_item("hits", s.hits)?;
        d.set_item("accesses", s.accesses)?;
        d.set_item("evictions", s.evictions)?;
        d.set_item("max_overflow", s.max_overflow)?;
        d.set_item("hit_ratio", s.hit_ratio())?;
        Ok(d)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrace>()?;
    m.add_class::<Stepper>()?;
    m.add_function(wrap_pyfunction!(load_trace, m)?)?;
    Ok(())
}
