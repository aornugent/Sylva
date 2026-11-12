// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings 3dtrees' fork adds (CHANGELOG-3dtrees.md): the scan index, the
//! joint model and `support_tree`, the evidence reduction, the trees' context
//! and contest, and the ray-cloud tiles. Kept apart from upstream's binding
//! files, so a merge from upstream meets them in one line of `lib.rs`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::io;
use sylva_rs::qsm::{self, support_tree};
use sylva_rs::qsm::evidence::{self, Beam, EvidenceParams};
use sylva_rs::qsm::scan_index::{self, ScanIndex};
use sylva_rs::qsm::joint::{self, Echoes, JointParams, Owner, Pulse, Value};
use sylva_rs::qsm::joint_fit::{PriorParams, Rows};

use crate::{err, xyz_from_py, xyz_to_py};

/// One scan's pulses indexed by direction, read from a single-position Shots
/// file and held on the Rust side.
#[pyclass(name = "ScanIndex", frozen)]
pub(crate) struct PyScanIndex {
    inner: ScanIndex,
}

#[pymethods]
impl PyScanIndex {
    #[new]
    fn new(py: Python<'_>, path: PathBuf, threads: usize) -> PyResult<Self> {
        let inner = py.detach(|| ScanIndex::from_file(&path, threads)).map_err(err)?;
        Ok(PyScanIndex { inner })
    }

    #[getter]
    fn misses(&self) -> bool {
        self.inner.misses()
    }

    #[getter]
    fn n_pulses(&self) -> usize {
        self.inner.n_pulses()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    #[getter]
    fn origin(&self) -> (f64, f64, f64) {
        let o = self.inner.origin();
        (o[0], o[1], o[2])
    }
}

/// The fields of `EvidenceParams`, flat, with the beam's prefixed `beam_`
/// (except `range_noise`), and the semantics `version`.
fn evidence_params_to_py<'py>(py: Python<'py>, p: &EvidenceParams) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", EvidenceParams::VERSION)?;
    for (k, v) in [
        ("beam_divergence", p.beam.divergence),
        ("beam_exit", p.beam.exit),
        ("range_noise", p.beam.range_noise),
        ("floor_log", p.floor_log),
        ("footprint_sd", p.footprint_sd),
        ("spread_sd", p.spread_sd),
        ("echoes_only_min_radius", p.echoes_only_min_radius),
        ("min_footprints", p.min_footprints),
        ("window_cap", p.window_cap),
        ("window_min", p.window_min),
        ("window_hi_factor", p.window_hi_factor),
        ("window_hi_margin", p.window_hi_margin),
        ("window_r_factor", p.window_r_factor),
        ("window_r_margin", p.window_r_margin),
        ("min_length", p.min_length),
    ] {
        d.set_item(k, v)?;
    }
    d.set_item("rounds", p.rounds)?;
    Ok(d)
}

fn evidence_params_from_py(d: &Bound<'_, PyDict>) -> PyResult<EvidenceParams> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("evidence params: no {k:?}")));
    let f = |k: &str| get(k)?.extract::<f64>();
    if get("version")?.extract::<u32>()? != EvidenceParams::VERSION {
        return Err(PyValueError::new_err(format!("evidence params: version {} is not this Sylva's ({})", get("version")?, EvidenceParams::VERSION)));
    }
    let known = evidence_params_to_py(d.py(), &EvidenceParams::default())?;
    if d.len() != known.len() || d.keys().iter().any(|k| !known.contains(&k).unwrap_or(false)) {
        return Err(PyValueError::new_err("evidence params: unexpected fields"));
    }
    Ok(EvidenceParams {
        beam: Beam { divergence: f("beam_divergence")?, exit: f("beam_exit")?, range_noise: f("range_noise")? },
        floor_log: f("floor_log")?,
        footprint_sd: f("footprint_sd")?,
        spread_sd: f("spread_sd")?,
        echoes_only_min_radius: f("echoes_only_min_radius")?,
        min_footprints: f("min_footprints")?,
        window_cap: f("window_cap")?,
        window_min: f("window_min")?,
        window_hi_factor: f("window_hi_factor")?,
        window_hi_margin: f("window_hi_margin")?,
        window_r_factor: f("window_r_factor")?,
        window_r_margin: f("window_r_margin")?,
        rounds: get("rounds")?.extract()?,
        min_length: f("min_length")?,
    })
}

/// The default evidence settings, as `evidence_params_to_py` lays them out.
#[pyfunction]
fn qsm_evidence_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    evidence_params_to_py(py, &EvidenceParams::default())
}

/// The names of the `reason` codes.
#[pyfunction]
fn qsm_evidence_reasons() -> Vec<(&'static str, u8)> {
    use evidence::reason::*;
    vec![("MEASURED", MEASURED), ("NO_HITS", NO_HITS), ("CLUTTER", CLUTTER), ("FOOTPRINT", FOOTPRINT), ("ECHOES_ONLY", ECHOES_ONLY)]
}

#[pyfunction]
fn qsm_evidence<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, r_claim: Option<PyReadonlyArray1<f64>>, params: Bound<'py, PyDict>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let r_claim: Option<Vec<f64>> = r_claim.map(|r| r.as_array().to_vec());
    let p = evidence_params_from_py(&params)?;
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let e = py.detach(|| evidence::evidence(&refs, &a, &b, r_claim.as_deref(), &p, threads)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("mu", e.mu.into_pyarray(py))?;
    d.set_item("tau", e.tau.into_pyarray(py))?;
    d.set_item("lo", e.lo.into_pyarray(py))?;
    d.set_item("hi", e.hi.into_pyarray(py))?;
    d.set_item("n_hit", e.n_hit.into_pyarray(py))?;
    d.set_item("n_cross", e.n_cross.into_pyarray(py))?;
    d.set_item("n_contra", e.n_contra.into_pyarray(py))?;
    d.set_item("reason", e.reason.into_pyarray(py))?;
    d.set_item("n_core", e.n_core.into_pyarray(py))?;
    d.set_item("n_pen", e.n_pen.into_pyarray(py))?;
    d.set_item("reflectance", e.reflectance.into_pyarray(py))?;
    d.set_item("deviation", e.deviation.into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn qsm_pairs<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, window: PyReadonlyArray1<f64>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let window = window.as_array().to_vec();
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let p = py.detach(|| scan_index::pairs(&refs, &a, &b, &window, threads)).map_err(err)?;
    let d = PyDict::new(py);
    let dir: Vec<f32> = p.dir.iter().flatten().copied().collect();
    d.set_item("cyl_start", p.cyl_start.into_pyarray(py))?;
    d.set_item("scan", p.scan.into_pyarray(py))?;
    d.set_item("pulse", p.pulse.into_pyarray(py))?;
    d.set_item("dir", dir.into_pyarray(py).reshape([p.t.len(), 3])?)?;
    d.set_item("t", p.t.into_pyarray(py))?;
    d.set_item("s", p.s.into_pyarray(py))?;
    d.set_item("d", p.d.into_pyarray(py))?;
    d.set_item("lost", p.lost.into_pyarray(py))?;
    d.set_item("returns", p.returns.into_pyarray(py))?;
    d.set_item("return_number", p.return_number.into_pyarray(py))?;
    d.set_item("echo_start", p.echo_start.into_pyarray(py))?;
    d.set_item("range", p.range.into_pyarray(py))?;
    d.set_item("reflectance", p.reflectance.into_pyarray(py))?;
    d.set_item("deviation", p.deviation.into_pyarray(py))?;
    Ok(d)
}

/// A joint params value as Python holds it.
fn value_to_py<'py>(py: Python<'py>, v: Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Real(x) => x.into_pyobject(py)?.into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
    })
}

/// The default settings of `qsm_support_tree`, flat: the likelihood's fields, then the prior's.
#[pyfunction]
fn qsm_joint_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let (like, prior) = (JointParams::default(), PriorParams::default());
    let d = PyDict::new(py);
    for &k in JointParams::FIELDS {
        d.set_item(k, value_to_py(py, like.get(k).unwrap())?)?;
    }
    for &k in PriorParams::FIELDS {
        d.set_item(k, value_to_py(py, prior.get(k).unwrap())?)?;
    }
    Ok(d)
}

/// The settings `qsm_joint_defaults` lays out, each field given once (`dataclasses.asdict(JointParams)`).
fn joint_params(d: &Bound<'_, PyDict>) -> PyResult<(JointParams, PriorParams)> {
    let (mut like, mut prior) = (JointParams::default(), PriorParams::default());
    for (k, v) in d.iter() {
        let k: String = k.extract()?;
        let v = match v.extract::<u64>() {
            Ok(i) => Value::Int(i),
            Err(_) => Value::Real(v.extract()?),
        };
        if !(like.set(&k, v) || prior.set(&k, v)) {
            return Err(PyValueError::new_err(format!("joint params: {k:?} is not a field, or not of its type")));
        }
    }
    if d.len() != JointParams::FIELDS.len() + PriorParams::FIELDS.len() {
        return Err(PyValueError::new_err("joint params: every field once"));
    }
    Ok((like, prior))
}

fn rows<const K: usize>(a: PyReadonlyArray2<f64>, what: &str) -> PyResult<Vec<[f64; K]>> {
    let a = a.as_array();
    if a.ncols() != K {
        return Err(PyValueError::new_err(format!("joint: {what} must be (n, {K})")));
    }
    Ok(a.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect())
}

fn dict_item<'py>(d: &Bound<'py, PyDict>, k: &str, what: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("{what}: no {k:?}")))
}

/// The pairs of a joint table's columns: each `Pulse` field an `(m,)` array (`jb`, `jt`: `(m, 2)`; `stop`, `early`:
/// bool).
fn pulses_from(table: &Bound<'_, PyDict>) -> PyResult<Vec<Pulse>> {
    let what = "joint table";
    let col = |k: &str| -> PyResult<Vec<f64>> { Ok(dict_item(table, k, what)?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec()) };
    let flag = |k: &str| -> PyResult<Vec<bool>> { Ok(dict_item(table, k, what)?.extract::<PyReadonlyArray1<bool>>()?.as_array().to_vec()) };
    let (b, t, sin, c0, c1, x, q) = (col("b")?, col("t")?, col("sin")?, col("c0")?, col("c1")?, col("x")?, col("q")?);
    let (jb, jt) = (rows::<2>(dict_item(table, "jb", what)?.extract()?, "jb")?, rows::<2>(dict_item(table, "jt", what)?.extract()?, "jt")?);
    let (stop, early, deleted) = (flag("stop")?, flag("early")?, flag("deleted")?);
    let n = b.len();
    if [t.len(), sin.len(), jb.len(), jt.len(), c0.len(), c1.len(), x.len(), q.len(), deleted.len(), stop.len(), early.len()].iter().any(|&l| l != n) {
        return Err(PyValueError::new_err("joint table: every column must have one entry per pair"));
    }
    Ok((0..n).map(|i| Pulse { b: b[i], t: t[i], sin: sin[i], jb: jb[i], jt: jt[i], c0: c0[i], c1: c1[i], x: x[i], deleted: deleted[i], stop: stop[i], early: early[i], q: q[i] }).collect())
}

/// Each pair's log likelihood at its own state `state` (m, 4: two offsets, log R, log lambda), the pairs as
/// `qsm_joint_table` gives them: the likelihood without the fit, for checking it.
#[pyfunction]
#[pyo3(signature = (table, state, params, threads=0))]
fn qsm_joint_loglik<'py>(py: Python<'py>, table: Bound<'py, PyDict>, state: PyReadonlyArray2<f64>, params: Bound<'py, PyDict>, threads: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (pulses, state, p) = (pulses_from(&table)?, rows::<4>(state, "state")?, joint_params(&params)?.0);
    if state.len() != pulses.len() {
        return Err(PyValueError::new_err("joint: one state per pair"));
    }
    let owners: Vec<Owner> = (0..pulses.len() as u32).map(|row| Owner { row, nu: u32::MAX }).collect();
    let ll = py.detach(|| joint::joint_loglik(&pulses, &owners, &state, &[], &p, threads)).map_err(err)?;
    Ok(ll.into_pyarray(py))
}

/// The joint model's pair table of cylinders `a -> b`, each with its tube `window`: named columns (the `Pulse`
/// fields, plus `row`, `scan`, `pulse`, `n_echo`).
#[pyfunction]
fn qsm_joint_table<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, window: PyReadonlyArray1<f64>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let window = window.as_array().to_vec();
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let t = py.detach(|| joint::joint_table(&refs, &a, &b, &window, threads)).map_err(err)?;
    let d = PyDict::new(py);
    let f = |g: fn(&Pulse) -> f64| t.pulses.iter().map(g).collect::<Vec<f64>>();
    for (k, g) in [("b", (|p: &Pulse| p.b) as fn(&Pulse) -> f64), ("t", |p| p.t), ("sin", |p| p.sin), ("c0", |p| p.c0), ("c1", |p| p.c1), ("x", |p| p.x), ("q", |p| p.q)] {
        d.set_item(k, f(g).into_pyarray(py))?;
    }
    let m = t.pulses.len();
    let jb: Vec<f64> = t.pulses.iter().flat_map(|p| p.jb).collect();
    let jt: Vec<f64> = t.pulses.iter().flat_map(|p| p.jt).collect();
    d.set_item("jb", jb.into_pyarray(py).reshape([m, 2])?)?;
    d.set_item("jt", jt.into_pyarray(py).reshape([m, 2])?)?;
    d.set_item("stop", t.pulses.iter().map(|p| p.stop).collect::<Vec<bool>>().into_pyarray(py))?;
    d.set_item("early", t.pulses.iter().map(|p| p.early).collect::<Vec<bool>>().into_pyarray(py))?;
    d.set_item("deleted", t.pulses.iter().map(|p| p.deleted).collect::<Vec<bool>>().into_pyarray(py))?;
    d.set_item("row", t.row.into_pyarray(py))?;
    d.set_item("scan", t.scan.into_pyarray(py))?;
    d.set_item("pulse", t.pulse.into_pyarray(py))?;
    d.set_item("n_echo", t.n_echo.into_pyarray(py))?;
    Ok(d)
}

/// A tree's classified echoes as a dict of `(m,)` arrays.
fn echoes_to_py<'py>(py: Python<'py>, e: Echoes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("scan", e.scan.into_pyarray(py))?;
    d.set_item("pulse", e.pulse.into_pyarray(py))?;
    d.set_item("echo", e.echo.into_pyarray(py))?;
    d.set_item("p_wood", e.p_wood.into_pyarray(py))?;
    Ok(d)
}

/// The names of the `flags` bits of `qsm_support_tree`.
#[pyfunction]
fn qsm_support_flags() -> Vec<(&'static str, u8)> {
    use support_tree::flag::*;
    vec![("RUNS", RUNS), ("HULL", HULL), ("BOUNDS", BOUNDS), ("POROUS", POROUS)]
}

/// The names of the `diag` bits of `qsm_support_tree`.
#[pyfunction]
fn qsm_support_diags() -> Vec<(&'static str, u16)> {
    use support_tree::diag::*;
    vec![("ONE_SIDED", ONE_SIDED), ("SHADOWED", SHADOWED), ("SUB_FOOTPRINT", SUB_FOOTPRINT), ("TUBE_EDGE", TUBE_EDGE), ("CONTESTED", CONTESTED), ("REFUTED", REFUTED), ("PINNED", PINNED), ("CAPPED", CAPPED), ("MOVED", MOVED)]
}

/// The names of the `basis` codes of `qsm_support_tree`.
#[pyfunction]
fn qsm_support_bases() -> Vec<(&'static str, u8)> {
    use support_tree::basis::*;
    vec![("MEASURED", MEASURED), ("POINTS", POINTS), ("BOUNDED", BOUNDED), ("PRIOR", PRIOR), ("CAPPED", CAPPED)]
}

/// A support fit's rows or pieces as a dict of arrays.
fn support_fit<'py>(py: Python<'py>, f: support_tree::Fit) -> PyResult<Bound<'py, PyDict>> {
    let n = f.r.len();
    let d = PyDict::new(py);
    let xyz = |v: &[[f64; 3]]| v.concat().into_pyarray(py).reshape([n, 3]);
    let two = |v: &[[f64; 2]]| v.concat().into_pyarray(py).reshape([n, 2]);
    d.set_item("a", xyz(&f.a)?)?;
    d.set_item("b", xyz(&f.b)?)?;
    d.set_item("e1", xyz(&f.e1)?)?;
    d.set_item("e2", xyz(&f.e2)?)?;
    d.set_item("centre", two(&f.centre)?)?;
    d.set_item("lean", two(&f.lean)?)?;
    for (k, v) in [("r", f.r), ("sd_log_r", f.sd_log_r), ("r_lo", f.r_lo), ("r_hi", f.r_hi), ("p_wood", f.p_wood), ("data_share", f.data_share), ("contested", f.contested), ("deleted", f.deleted), ("r_cap", f.r_cap), ("pass_share", f.pass_share)] {
        d.set_item(k, v.into_pyarray(py))?;
    }
    d.set_item("diag", f.diag.into_pyarray(py))?;
    d.set_item("flags", f.flags.into_pyarray(py))?;
    d.set_item("n_scans", f.n_scans.into_pyarray(py))?;
    d.set_item("k_stems", f.k_stems.into_pyarray(py))?;
    d.set_item("n_pass", f.n_pass.into_pyarray(py))?;
    d.set_item("basis", f.basis.into_pyarray(py))?;
    Ok(d)
}

/// Each of `trees`' context for `support_tree` (see `sylva.qsm.tree_context`).
#[pyfunction]
fn qsm_tree_context<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, cyl_offsets: Vec<usize>, trees: Vec<usize>,
                         reach: f64) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    if b.len() != a.len() || cyl_offsets.last() != Some(&a.len()) || cyl_offsets.windows(2).any(|w| w[0] > w[1])
        || trees.iter().any(|&g| g + 1 >= cyl_offsets.len()) {
        return Err(PyValueError::new_err("a and b one row per cylinder; cyl_offsets rising to the cylinder count; trees indices into them"));
    }
    let (off, idx) = py.detach(|| support_tree::tree_context(&a, &b, &cyl_offsets, &trees, reach));
    Ok((off.into_pyarray(py), idx.into_pyarray(py)))
}

/// Fits one tree from its rows' support summaries (see `sylva.qsm.support_tree`); `params` as `qsm_joint_defaults` lays them out.
#[pyfunction]
#[pyo3(signature = (scans, a, b, r_prior, parent, ctx_a, ctx_b, ctx_r, params, threads=0))]
#[allow(clippy::too_many_arguments)]
fn qsm_support_tree<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, r_prior: PyReadonlyArray1<f64>, parent: PyReadonlyArray1<i64>, ctx_a: PyReadonlyArray2<f64>, ctx_b: PyReadonlyArray2<f64>, ctx_r: PyReadonlyArray1<f64>, params: Bound<'py, PyDict>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let (ca, cb, cr) = (xyz_from_py(ctx_a)?, xyz_from_py(ctx_b)?, ctx_r.as_array().to_vec());
    if ca.len() != cb.len() || ca.len() != cr.len() {
        return Err(PyValueError::new_err("support tree: the context's a, b and r must have one entry per cylinder"));
    }
    let context: Vec<_> = ca.into_iter().zip(cb).zip(cr).map(|((a, b), r)| (a, b, r)).collect();
    let (r_prior, parent) = (r_prior.as_array().to_vec(), parent.as_array().to_vec());
    let p = joint_params(&params)?;
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let f = py.detach(|| {
        let rows = Rows { a: &a, b: &b, r_prior: &r_prior, r_max: &[], parent: &parent };
        support_tree::support_tree(&refs, &rows, &context, &p.0, &p.1, threads)
    }).map_err(err)?;
    let d = support_fit(py, f.rows)?;
    let pieces = support_fit(py, f.pieces)?;
    pieces.set_item("row", f.piece_row.into_pyarray(py))?;
    pieces.set_item("src", f.piece_src.into_pyarray(py))?;
    pieces.set_item("stem", f.piece_stem.into_pyarray(py))?;
    d.set_item("pieces", pieces)?;
    d.set_item("converged", f.converged)?;
    d.set_item("echoes", echoes_to_py(py, f.echoes)?)?;
    let v = f.volume;
    let volume = PyDict::new(py);
    for (k, x) in [("point", v.point), ("mean", v.mean), ("q05", v.q05), ("q95", v.q95)] {
        volume.set_item(k, x)?;
    }
    // the mean by basis, each under its basis's name
    for (name, b) in qsm_support_bases() {
        volume.set_item(name.to_lowercase(), v.by_basis[usize::from(b) - 1])?;
    }
    d.set_item("volume", volume)?;
    Ok(d)
}

/// A base's sections and columns as a dict of arrays (see `sylva.qsm.rebase`).
fn sections_to_py<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, secs: &[support_tree::base::Section], cols: &[support_tree::base::Column]) -> PyResult<()> {
    use support_tree::base::Section;
    let f = |g: fn(&Section) -> f64| secs.iter().map(g).collect::<Vec<f64>>();
    for (k, v) in [("z", f(|s| s.z)), ("x", f(|s| s.c[0])), ("y", f(|s| s.c[1])), ("r", f(|s| s.r)), ("hits", f(|s| s.hits)), ("evidence", f(|s| s.evidence))] {
        d.set_item(k, v.into_pyarray(py))?;
    }
    d.set_item("slab", secs.iter().map(|s| s.slab).collect::<Vec<u32>>().into_pyarray(py))?;
    d.set_item("fitted", secs.iter().map(|s| s.fitted).collect::<Vec<bool>>().into_pyarray(py))?;
    d.set_item("column", secs.iter().map(|s| s.column).collect::<Vec<i32>>().into_pyarray(py))?;
    d.set_item("col_parent", cols.iter().map(|c| c.parent).collect::<Vec<i32>>().into_pyarray(py))?;
    d.set_item("col_at", cols.iter().map(|c| c.at).collect::<Vec<i32>>().into_pyarray(py))?;
    Ok(())
}

/// A plant's base re-sectioned from the rays (see `sylva.qsm.rebase`).
#[pyfunction]
#[pyo3(signature = (scans, a, b, r_prior, parent, member, centre, ctx_a, ctx_b, ctx_r, params, z_lo=0.1, z_hi=1.5, dz=0.1, reach=0.6, link=0.02, keep=0.8, threads=0))]
#[allow(clippy::too_many_arguments)]
fn qsm_rebase<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, r_prior: PyReadonlyArray1<f64>, parent: PyReadonlyArray1<i64>, member: PyReadonlyArray1<i64>, centre: [f64; 3], ctx_a: PyReadonlyArray2<f64>, ctx_b: PyReadonlyArray2<f64>, ctx_r: PyReadonlyArray1<f64>, params: Bound<'py, PyDict>, z_lo: f64, z_hi: f64, dz: f64, reach: f64, link: f64, keep: f64, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let (ca, cb, cr) = (xyz_from_py(ctx_a)?, xyz_from_py(ctx_b)?, ctx_r.as_array().to_vec());
    if ca.len() != cb.len() || ca.len() != cr.len() {
        return Err(PyValueError::new_err("rebase: the context's a, b and r must have one entry per cylinder"));
    }
    let context: Vec<_> = ca.into_iter().zip(cb).zip(cr).map(|((a, b), r)| (a, b, r)).collect();
    let (r_prior, parent, member) = (r_prior.as_array().to_vec(), parent.as_array().to_vec(), member.as_array().to_vec());
    let p = joint_params(&params)?;
    let bp = support_tree::base::BaseParams { z_lo, z_hi, dz, reach, link, keep };
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let f = py.detach(|| {
        let rows = Rows { a: &a, b: &b, r_prior: &r_prior, r_max: &[], parent: &parent };
        support_tree::base::rebase(&refs, &rows, &member, &centre, &bp, &context, &p.0, &p.1, threads)
    }).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("a", xyz_to_py(py, &f.a))?;
    d.set_item("b", xyz_to_py(py, &f.b))?;
    d.set_item("r_prior", f.r_prior.into_pyarray(py))?;
    d.set_item("parent", f.parent.into_pyarray(py))?;
    d.set_item("member", f.member.into_pyarray(py))?;
    d.set_item("column", f.column.into_pyarray(py))?;
    let s = PyDict::new(py);
    sections_to_py(py, &s, &f.sections, &f.columns)?;
    d.set_item("sections", s)?;
    let c = PyDict::new(py);
    let g = |h: fn(&support_tree::base::ColumnOut) -> f64| f.out.iter().map(h).collect::<Vec<f64>>();
    for (k, v) in [("e_col", g(|c| c.e_col)), ("e_rct", g(|c| c.e_rct)), ("measured", g(|c| c.measured)), ("r", g(|c| c.r))] {
        c.set_item(k, v.into_pyarray(py))?;
    }
    c.set_item("chosen", f.out.iter().map(|c| c.chosen).collect::<Vec<bool>>().into_pyarray(py))?;
    c.set_item("believable", f.out.iter().map(|c| c.believable).collect::<Vec<bool>>().into_pyarray(py))?;
    c.set_item("owner", f.out.iter().map(|c| c.owner).collect::<Vec<i64>>().into_pyarray(py))?;
    c.set_item("joined", f.out.iter().map(|c| c.joined).collect::<Vec<u32>>().into_pyarray(py))?;
    c.set_item("grafted", f.out.iter().map(|c| c.grafted).collect::<Vec<u32>>().into_pyarray(py))?;
    d.set_item("columns", c)?;
    let counts = PyDict::new(py);
    for (k, v) in ["entries", "joined", "grafted", "kept", "dropped", "hung", "reverted"].iter().zip(f.counts) {
        counts.set_item(*k, v)?;
    }
    d.set_item("counts", counts)?;
    Ok(d)
}


/// Each tree's contested points (see `sylva.qsm.contest`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn qsm_contest<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i32>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>,
                    radius: PyReadonlyArray1<f64>, cyl_offsets: Vec<usize>, tol: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let (points, labels, a, b, r) = (xyz_from_py(points)?, labels.as_slice()?, xyz_from_py(a)?, xyz_from_py(b)?, radius.as_slice()?);
    if labels.len() != points.len() || r.len() != a.len() || b.len() != a.len() || cyl_offsets.last().is_some_and(|&n| n != a.len()) || cyl_offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(PyValueError::new_err("one label per point, one radius per cylinder, and cyl_offsets rising to the cylinder count"));
    }
    let (share, owner) = py.detach(|| qsm::support::contest(&sylva_rs::util::spatial::KdTree::new(&points), &points, labels, &a, &b, r, &cyl_offsets, tol));
    Ok((share.into_pyarray(py), owner.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (files, out, width, buffer, voxel, terrain_voxel, extent=None, min_rays=100_000, threads=4))]
#[allow(clippy::too_many_arguments)]
fn tile_ray_clouds<'py>(py: Python<'py>, files: Vec<PathBuf>, out: PathBuf, width: f64, buffer: f64, voxel: f64, terrain_voxel: f64,
                        extent: Option<(f64, f64, f64, f64)>, min_rays: usize, threads: usize) -> PyResult<(Vec<Bound<'py, PyDict>>, usize)> {
    let opts = io::ray_tiles::TileOptions { width, buffer, voxel, terrain_voxel, extent: extent.map(|e| [e.0, e.1, e.2, e.3]), min_rays };
    let (tiles, n_terrain) = py.detach(|| io::ray_tiles::tile_ray_clouds(&files, &out, &opts, threads)).map_err(err)?;
    let tiles = tiles
        .into_iter()
        .map(|t| {
            let d = PyDict::new(py);
            d.set_item("tile", format!("{}_{}", t.kx, t.ky))?;
            d.set_item("raw", t.raw)?;
            d.set_item("rays", t.rays)?;
            d.set_item("written", t.rays >= min_rays.max(1))?;
            Ok(d)
        })
        .collect::<PyResult<_>>()?;
    Ok((tiles, n_terrain))
}
/// Adds this file's functions to the `_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScanIndex>()?;
    for f in [
        wrap_pyfunction!(qsm_rebase, m)?,
        wrap_pyfunction!(qsm_support_flags, m)?,
        wrap_pyfunction!(qsm_support_diags, m)?,
        wrap_pyfunction!(qsm_support_bases, m)?,
        wrap_pyfunction!(qsm_support_tree, m)?,
        wrap_pyfunction!(qsm_tree_context, m)?,
        wrap_pyfunction!(qsm_joint_defaults, m)?,
        wrap_pyfunction!(qsm_joint_loglik, m)?,
        wrap_pyfunction!(qsm_joint_table, m)?,
        wrap_pyfunction!(qsm_evidence, m)?,
        wrap_pyfunction!(qsm_pairs, m)?,
        wrap_pyfunction!(qsm_evidence_defaults, m)?,
        wrap_pyfunction!(qsm_evidence_reasons, m)?,
        wrap_pyfunction!(qsm_contest, m)?,
        wrap_pyfunction!(tile_ray_clouds, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
