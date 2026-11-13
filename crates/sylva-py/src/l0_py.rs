// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the L0 view, the moments' candidates and the ray likelihood (sylva_rs::shots::view, ::moments,
//! ::rays).
//!
//! - **Error model and scene:** flat dicts whose keys are the structs' fields (`beam_exit` and `beam_divergence` for
//!   the beam; arrays as lists). `l0_error_model()` returns the defaults.
//! - **Elements:** a dict of arrays, one row per element: `a`, `b` (n, 3) and `r` (n,) are required; `opacity` (n,)
//!   defaults to 1, `misfit` (n,) to the error model's prior `misfit[0] + misfit[1] r`, and `patch` (n,) bool to
//!   false. A patch is a ball about `a` of radius `r` (its extent); its `b` is ignored.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::shots::moments::{Candidate, Moments};
use sylva_rs::shots::rays::{energy, predict, Element, Form, Prediction, Scene};
use sylva_rs::shots::view::{ErrorModel, View, ViewParams};

use crate::xyz_from_py;

fn bad(k: &str) -> PyErr {
    PyValueError::new_err(format!("unknown or malformed key {k:?}"))
}

fn error_model(d: Option<&Bound<'_, PyDict>>) -> PyResult<ErrorModel> {
    let mut m = ErrorModel::default();
    for (k, v) in d.into_iter().flat_map(|d| d.iter()) {
        let k: String = k.extract()?;
        let x = |v: &Bound<'_, PyAny>| v.extract::<f64>();
        match k.as_str() {
            "beam_exit" => m.beam.exit = x(&v)?,
            "beam_divergence" => m.beam.divergence = x(&v)?,
            "step" => m.step = x(&v)?,
            "class_edges" => m.class_edges = v.extract()?,
            "sigma_e" => m.sigma_e = v.extract()?,
            "nu_e" => m.nu_e = x(&v)?,
            "bias" => m.bias = v.extract()?,
            "bias_ref" => m.bias_ref = x(&v)?,
            "field" => m.field = v.extract()?,
            "field_spacing" => m.field_spacing = x(&v)?,
            "field_along" => m.field_along = v.extract()?,
            "field_epoch" => m.field_epoch = v.extract()?,
            "field_across" => m.field_across = x(&v)?,
            "rho" => m.rho = x(&v)?,
            "cut_db" => m.cut_db = x(&v)?,
            "refl_noise_db" => m.refl_noise_db = x(&v)?,
            "p_drop" => m.p_drop = x(&v)?,
            "p_outlier" => m.p_outlier = x(&v)?,
            "misfit" => m.misfit = v.extract()?,
            "edge_fill" => m.edge_fill = x(&v)?,
            _ => return Err(bad(&k)),
        }
    }
    Ok(m)
}

/// The error model's defaults (AM's calibration), as the dict the other calls take.
#[pyfunction]
fn l0_error_model<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let m = ErrorModel::default();
    let d = PyDict::new(py);
    for (k, v) in [("beam_exit", m.beam.exit), ("beam_divergence", m.beam.divergence), ("step", m.step), ("nu_e", m.nu_e), ("bias_ref", m.bias_ref),
                   ("field_spacing", m.field_spacing), ("field_across", m.field_across), ("rho", m.rho), ("cut_db", m.cut_db), ("refl_noise_db", m.refl_noise_db), ("p_drop", m.p_drop),
                   ("p_outlier", m.p_outlier), ("edge_fill", m.edge_fill)] {
        d.set_item(k, v)?;
    }
    d.set_item("class_edges", m.class_edges.to_vec())?;
    d.set_item("sigma_e", m.sigma_e.to_vec())?;
    d.set_item("bias", m.bias.to_vec())?;
    d.set_item("field", m.field)?;
    d.set_item("field_along", m.field_along.to_vec())?;
    d.set_item("field_epoch", m.field_epoch.to_vec())?;
    d.set_item("misfit", m.misfit.to_vec())?;
    Ok(d)
}

fn scene(d: Option<&Bound<'_, PyDict>>) -> PyResult<Scene> {
    let mut s = Scene::default();
    for (k, v) in d.into_iter().flat_map(|d| d.iter()) {
        let k: String = k.extract()?;
        let x: f64 = v.extract()?;
        match k.as_str() {
            "kappa" => s.kappa = x,
            "ground_z" => s.ground_z = x,
            "wood_db" => s.wood_db = x,
            "ground_db" => s.ground_db = x,
            "t_max" => s.t_max = x,
            _ => return Err(bad(&k)),
        }
    }
    Ok(s)
}

fn elements_from_py(d: &Bound<'_, PyDict>, em: &ErrorModel) -> PyResult<Vec<Element>> {
    let get = |k: &str| d.get_item(k);
    let req = |k: &str| get(k)?.ok_or_else(|| PyValueError::new_err(format!("elements need {k:?}")));
    let a = xyz_from_py(req("a")?.extract()?)?;
    let b = xyz_from_py(req("b")?.extract()?)?;
    let r: Vec<f64> = req("r")?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec();
    let col = |k: &str, dflt: &dyn Fn(usize) -> f64| -> PyResult<Vec<f64>> {
        Ok(match get(k)? {
            Some(v) => v.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec(),
            None => (0..r.len()).map(dflt).collect(),
        })
    };
    let opacity = col("opacity", &|_| 1.0)?;
    let misfit = col("misfit", &|i| em.misfit[0] + em.misfit[1] * r[i])?;
    let patch: Vec<bool> = match get("patch")? {
        Some(v) => v.extract::<PyReadonlyArray1<bool>>()?.as_array().to_vec(),
        None => vec![false; r.len()],
    };
    let n = r.len();
    if [a.len(), b.len(), opacity.len(), misfit.len(), patch.len()].iter().any(|&m| m != n) {
        return Err(PyValueError::new_err("elements: every array needs one row per element"));
    }
    Ok((0..n)
        .map(|i| Element {
            form: if patch[i] { Form::Patch { centre: a[i] } } else { Form::Stick { a: a[i], b: b[i] } },
            r: r[i], opacity: opacity[i], misfit: misfit[i],
        })
        .collect())
}

/// One scan's L0 view: its pulses in acquisition order (`time` s, `direction` world unit vectors, first-echo
/// `range` with NaN for a pulse without one), its pose `rot` (3 × 3, world = rot · scanner + origin) and `epoch`.
#[pyclass(name = "L0View", frozen)]
struct PyView {
    inner: View,
}

#[pymethods]
impl PyView {
    #[new]
    #[pyo3(signature = (time, direction, range, rot, origin, epoch=0))]
    fn new(py: Python<'_>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>,
           origin: (f64, f64, f64), epoch: u32) -> PyResult<Self> {
        let dir = xyz_from_py(direction)?;
        let (t, r) = (time.as_array().to_vec(), range.as_array().to_vec());
        if t.len() != dir.len() || r.len() != dir.len() {
            return Err(PyValueError::new_err("time, direction and range need one row per pulse"));
        }
        let ra = rot.as_array();
        let rot = [0, 1, 2].map(|i| [0, 1, 2].map(|j| ra[[i, j]]));
        let p = ViewParams::default();
        let inner = py.detach(|| View::new(&t, &dir, &r, rot, [origin.0, origin.1, origin.2], epoch, &p));
        if inner.n_lines == 0 {
            return Err(PyValueError::new_err("no lattice: too few pulses"));
        }
        Ok(PyView { inner })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_lines, self.inner.n_cols)
    }

    /// The export's in-surface deletion measured on this scan (View::deletion).
    #[getter]
    fn deletion(&self) -> f64 {
        self.inner.deletion
    }

    /// The cell arrays, each (n_lines, n_cols[, k]): state, range, pulse (-1 none), normal, cos_inc, excess, run
    /// (along, across), surface_id (-1 for none); and from the error model `err`, surface (class) and across
    /// (footprints across the shorter run).
    #[pyo3(signature = (err=None))]
    fn arrays<'py>(&self, py: Python<'py>, err: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        let em = error_model(err)?;
        let v = &self.inner;
        let sh = [v.n_lines, v.n_cols];
        let d = PyDict::new(py);
        d.set_item("state", v.state.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("range", v.range.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("pulse", v.pulse.iter().map(|&p| if p == u32::MAX { -1 } else { p as i64 }).collect::<Vec<_>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("normal", v.normal.iter().flatten().copied().collect::<Vec<f32>>().into_pyarray(py).reshape([sh[0], sh[1], 3])?)?;
        d.set_item("cos_inc", v.cos_inc.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("excess", v.excess.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("run", v.run.iter().flatten().copied().collect::<Vec<u16>>().into_pyarray(py).reshape([sh[0], sh[1], 2])?)?;
        d.set_item("surface_id", v.surface_id.iter().map(|&s| if s == u32::MAX { -1 } else { s as i64 }).collect::<Vec<_>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("surface", v.excess.iter().map(|&e| em.surface(e as f64)).collect::<Vec<u8>>().into_pyarray(py).reshape(sh)?)?;
        let across: Vec<f32> = v.run.iter().zip(&v.range).map(|(r, &g)| em.across(r[0].min(r[1]) as f64, g as f64) as f32).collect();
        d.set_item("across", across.into_pyarray(py).reshape(sh)?)?;
        Ok(d)
    }

    /// The fractional (line, col) and range of each world point (NaN outside the lattice), flat (n × 3).
    fn cells<'py>(&self, py: Python<'py>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = xyz_from_py(points)?;
        let out: Vec<f64> = p.iter().flat_map(|x| self.inner.cell_of(x).map_or([f64::NAN; 3], |(l, c, r)| [l, c, r])).collect();
        Ok(out.into_pyarray(py))
    }
}

/// What the elements predict in one view, per touched cell with a layer of fill `min_fill` or more (rays::predict):
/// cell (flat index), elem, surface, code (0 on, 1 empty, 2 front, 3 behind), and fill, kept, depth, sd, obs, weight
/// (1/m), footprint (m), x and s (where the ray passes the axis, across and along).
#[pyfunction]
#[pyo3(signature = (view, elements, err=None, scene_=None, min_fill=0.01))]
fn l0_predict<'py>(py: Python<'py>, view: PyRef<'py, PyView>, elements: &Bound<'_, PyDict>, err: Option<&Bound<'_, PyDict>>,
                   scene_: Option<&Bound<'_, PyDict>>, min_fill: f64) -> PyResult<Bound<'py, PyDict>> {
    let (em, sc) = (error_model(err)?, scene(scene_)?);
    let el = elements_from_py(elements, &em)?;
    let v = &view.inner;
    let p = py.detach(|| predict(v, &el, &em, &sc, min_fill));
    let d = PyDict::new(py);
    d.set_item("cell", p.iter().map(|x| x.cell as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("elem", p.iter().map(|x| x.elem).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", p.iter().map(|x| x.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("code", p.iter().map(|x| x.code).collect::<Vec<_>>().into_pyarray(py))?;
    type F = fn(&Prediction) -> f64;
    for (k, f) in [("fill", (|x| x.fill) as F), ("kept", |x| x.kept), ("depth", |x| x.depth), ("sd", |x| x.sd), ("obs", |x| x.obs),
                   ("weight", |x| x.weight), ("footprint", |x| x.footprint), ("x", |x| x.x), ("s", |x| x.s)] {
        d.set_item(k, p.iter().map(f).collect::<Vec<_>>().into_pyarray(py))?;
    }
    Ok(d)
}

/// The energy of `elements` against `views` (rays::energy): −log likelihood ratio against the empty scene. Returns
/// (energy, rows): per (element, scan) its cells, support, footprints across, class, energy share and the offset
/// field's posterior at it (shift_along, sd_along, shift_across). With
/// `only=i`, the energy change of adding element i to the rest (rows empty).
#[pyfunction]
#[pyo3(signature = (views, elements, err=None, scene_=None, only=None))]
fn l0_energy<'py>(py: Python<'py>, views: Vec<PyRef<'py, PyView>>, elements: &Bound<'_, PyDict>, err: Option<&Bound<'_, PyDict>>,
                  scene_: Option<&Bound<'_, PyDict>>, only: Option<usize>) -> PyResult<(f64, Bound<'py, PyDict>)> {
    let (em, sc) = (error_model(err)?, scene(scene_)?);
    let el = elements_from_py(elements, &em)?;
    let vs: Vec<&View> = views.iter().map(|v| &v.inner).collect();
    let (e, rows) = py.detach(|| energy(&vs, &el, &em, &sc, only));
    let d = PyDict::new(py);
    d.set_item("elem", rows.iter().map(|r| r.elem).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", rows.iter().map(|r| r.scan).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cells", rows.iter().map(|r| r.cells).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("support", rows.iter().map(|r| r.support).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("across", rows.iter().map(|r| r.across).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", rows.iter().map(|r| r.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("energy", rows.iter().map(|r| r.energy).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("shift_along", rows.iter().map(|r| r.shift_along).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("sd_along", rows.iter().map(|r| r.sd_along).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("shift_across", rows.iter().map(|r| r.shift_across).collect::<Vec<_>>().into_pyarray(py))?;
    Ok((e, d))
}

/// The moments' candidates (moments::Moments::candidates) from the views' surfels inside the box lo-hi, on `voxel`
/// cells: key (k, 3), point, u, lam (k, 3), r, ratio, n, extent (k,), and per scan (k, S) scan_n and arc.
#[pyfunction]
#[pyo3(signature = (views, lo, hi, voxel=0.03, min_n=30.0, r_max=0.3))]
fn l0_elements<'py>(py: Python<'py>, views: Vec<PyRef<'py, PyView>>, lo: [f64; 3], hi: [f64; 3], voxel: f64, min_n: f64, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    use nalgebra::Vector3;
    use rayon::prelude::*;
    let vs: Vec<&View> = views.iter().map(|v| &v.inner).collect();
    let el = py.detach(|| {
        let v3 = |p: &[f64; 3]| Vector3::new(p[0], p[1], p[2]);
        let scans: Vec<_> = vs.par_iter().map(|v| {
            let (p, n) = v.surfels(&lo, &hi);
            (p.iter().map(v3).collect(), n.iter().map(v3).collect())
        }).collect();
        Moments::new(&scans, voxel).candidates(min_n, r_max)
    });
    let (k, s) = (el.len(), vs.len());
    let d = PyDict::new(py);
    let flat3 = |f: &dyn Fn(&Candidate) -> [f64; 3]| el.iter().flat_map(f).collect::<Vec<f64>>();
    d.set_item("key", el.iter().flat_map(|e| e.key).collect::<Vec<i32>>().into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("point", flat3(&|e| [e.point[0], e.point[1], e.point[2]]).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("u", flat3(&|e| [e.u[0], e.u[1], e.u[2]]).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("lam", flat3(&|e| e.lam).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("r", el.iter().map(|e| e.r).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("ratio", el.iter().map(|e| e.ratio).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n", el.iter().map(|e| e.n).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("extent", el.iter().map(|e| e.extent).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan_n", el.iter().flat_map(|e| e.scan_n.clone()).collect::<Vec<_>>().into_pyarray(py).reshape([k, s])?)?;
    d.set_item("arc", el.iter().flat_map(|e| e.arc.clone()).collect::<Vec<_>>().into_pyarray(py).reshape([k, s])?)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyView>()?;
    m.add_function(wrap_pyfunction!(l0_energy, m)?)?;
    m.add_function(wrap_pyfunction!(l0_predict, m)?)?;
    m.add_function(wrap_pyfunction!(l0_elements, m)?)?;
    m.add_function(wrap_pyfunction!(l0_error_model, m)?)?;
    Ok(())
}
