// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the L0 view and ray likelihood (sylva_rs::shots::view, ::rays). The error model and the scene
//! travel as flat dicts whose keys are the structs' fields (`beam_exit`, `beam_divergence` for the beam; arrays as
//! lists); `l0_error_model()` returns the defaults.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::shots::moments::{key, Moments};
use sylva_rs::shots::rays::{energy, predict, Scene, Stick};
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
            "sigma_l" => m.sigma_l = v.extract()?,
            "nu_l" => m.nu_l = x(&v)?,
            "block" => m.block = x(&v)?,
            "cut_db" => m.cut_db = x(&v)?,
            "refl_noise_db" => m.refl_noise_db = x(&v)?,
            "p_drop" => m.p_drop = x(&v)?,
            "p_outlier" => m.p_outlier = x(&v)?,
            "rho" => m.rho = x(&v)?,
            "ghost" => m.ghost = x(&v)?,
            "ghost_depth" => m.ghost_depth = x(&v)?,
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
    for (k, v) in [("beam_exit", m.beam.exit), ("beam_divergence", m.beam.divergence), ("step", m.step), ("nu_e", m.nu_e), ("bias_ref", m.bias_ref), ("nu_l", m.nu_l), ("block", m.block), ("cut_db", m.cut_db), ("refl_noise_db", m.refl_noise_db), ("p_drop", m.p_drop), ("p_outlier", m.p_outlier), ("rho", m.rho), ("ghost", m.ghost), ("ghost_depth", m.ghost_depth)] {
        d.set_item(k, v)?;
    }
    d.set_item("class_edges", m.class_edges.to_vec())?;
    d.set_item("sigma_e", m.sigma_e.to_vec())?;
    d.set_item("bias", m.bias.to_vec())?;
    d.set_item("sigma_l", m.sigma_l.to_vec())?;
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

    /// The cell arrays, each (n_lines, n_cols[, k]): state, range, pulse (-1 none), normal, cos_inc, excess, run
    /// (along, across), ends, surface_id (-1 for none); and from the error model `err`, surface (class) and across (footprints across the
    /// shorter run).
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
        d.set_item("ends", v.ends.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("surface_id", v.surface_id.iter().map(|&s| if s == u32::MAX { -1 } else { s as i64 }).collect::<Vec<_>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("surface", v.excess.iter().map(|&e| em.surface(e as f64)).collect::<Vec<u8>>().into_pyarray(py).reshape(sh)?)?;
        let across: Vec<f32> = v.run.iter().zip(&v.range).map(|(r, &g)| em.across(r[0].min(r[1]) as f64, g as f64) as f32).collect();
        d.set_item("across", across.into_pyarray(py).reshape(sh)?)?;
        Ok(d)
    }

    /// The fractional (line, col) and range of each world point (NaN outside the lattice).
    fn cells<'py>(&self, py: Python<'py>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = xyz_from_py(points)?;
        let out: Vec<f64> = p.iter().flat_map(|x| self.inner.cell_of(x).map_or([f64::NAN; 3], |(l, c, r)| [l, c, r])).collect();
        Ok(out.into_pyarray(py))
    }
}

fn sticks_from_py(sticks: PyReadonlyArray2<f64>, surface: Option<PyReadonlyArray1<i8>>) -> PyResult<Vec<Stick>> {
    let a = sticks.as_array();
    if a.ncols() != 7 {
        return Err(PyValueError::new_err("sticks must be (n, 7): a, b, radius"));
    }
    let surf: Vec<i8> = surface.map_or(vec![-1; a.nrows()], |s| s.as_array().to_vec());
    Ok(a.rows().into_iter().zip(&surf).map(|(r, &s)| Stick { a: [r[0], r[1], r[2]], b: [r[3], r[4], r[5]], r: r[6], surface: s }).collect())
}

/// What sticks predict in one view, per touched cell (rays::Prediction): cell (flat), stick, fill, kept, depth,
/// sd, surface, obs.
#[pyfunction]
#[pyo3(signature = (view, sticks, surface=None, err=None, scene_=None))]
fn l0_predict<'py>(py: Python<'py>, view: PyRef<'py, PyView>, sticks: PyReadonlyArray2<f64>, surface: Option<PyReadonlyArray1<i8>>,
                   err: Option<&Bound<'_, PyDict>>, scene_: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
    let st = sticks_from_py(sticks, surface)?;
    let (em, sc) = (error_model(err)?, scene(scene_)?);
    let v = &view.inner;
    let p = py.detach(|| predict(v, &st, &em, &sc));
    let d = PyDict::new(py);
    d.set_item("cell", p.iter().map(|x| x.cell as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("stick", p.iter().map(|x| x.stick).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", p.iter().map(|x| x.surface).collect::<Vec<_>>().into_pyarray(py))?;
    for (k, f) in [("fill", (|x: &sylva_rs::shots::rays::Prediction| x.fill) as fn(&_) -> f64), ("kept", |x| x.kept), ("depth", |x| x.depth), ("sd", |x| x.sd), ("obs", |x| x.obs)] {
        d.set_item(k, p.iter().map(f).collect::<Vec<_>>().into_pyarray(py))?;
    }
    Ok(d)
}

/// The energy of sticks against views: `sticks` (n, 7) = a, b, radius; `surface` (n,) class 0-2 or -1 (from the
/// view). Returns (energy, rows): −log likelihood ratio against the empty scene, and per (stick, scan) its cells,
/// support, footprints across, class and energy share. With `only=i`, the energy change of adding stick i to the
/// rest (rows empty).
#[pyfunction]
#[pyo3(signature = (views, sticks, surface=None, err=None, scene_=None, only=None))]
fn l0_energy<'py>(py: Python<'py>, views: Vec<PyRef<'py, PyView>>, sticks: PyReadonlyArray2<f64>, surface: Option<PyReadonlyArray1<i8>>,
                  err: Option<&Bound<'_, PyDict>>, scene_: Option<&Bound<'_, PyDict>>, only: Option<usize>) -> PyResult<(f64, Bound<'py, PyDict>)> {
    let st = sticks_from_py(sticks, surface)?;
    let (em, sc) = (error_model(err)?, scene(scene_)?);
    let vs: Vec<&View> = views.iter().map(|v| &v.inner).collect();
    let (e, rows) = py.detach(|| energy(&vs, &st, &em, &sc, only));
    let d = PyDict::new(py);
    d.set_item("stick", rows.iter().map(|r| r.stick).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", rows.iter().map(|r| r.scan).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cells", rows.iter().map(|r| r.cells).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("support", rows.iter().map(|r| r.support).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("across", rows.iter().map(|r| r.across).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", rows.iter().map(|r| r.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("energy", rows.iter().map(|r| r.energy).collect::<Vec<_>>().into_pyarray(py))?;
    Ok((e, d))
}

fn moments_of(views: &[&View], lo: [f64; 3], hi: [f64; 3], voxel: f64) -> Moments {
    use nalgebra::Vector3;
    use rayon::prelude::*;
    let v3 = |p: &[f64; 3]| Vector3::new(p[0], p[1], p[2]);
    let scans: Vec<_> = views.par_iter().map(|v| {
        let (p, n) = v.surfels(&lo, &hi);
        (p.iter().map(v3).collect(), n.iter().map(v3).collect())
    }).collect();
    Moments::new(&scans, views.iter().map(|v| v3(&v.origin)).collect(), voxel)
}

/// Candidate elements from the views' surfels inside the box lo-hi on `voxel` cells (moments::Moments::elements):
/// key, point, u, r, ratio, lam (k, 3), n, and per scan (k, S) scan_n, arc, excess.
#[pyfunction]
#[pyo3(signature = (views, lo, hi, voxel=0.03, min_n=30.0, r_max=0.3))]
fn l0_elements<'py>(py: Python<'py>, views: Vec<PyRef<'py, PyView>>, lo: [f64; 3], hi: [f64; 3], voxel: f64, min_n: f64, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    let vs: Vec<&View> = views.iter().map(|v| &v.inner).collect();
    let step = ErrorModel::default().step;
    let el = py.detach(|| moments_of(&vs, lo, hi, voxel).elements(min_n, r_max, step));
    let (k, s) = (el.len(), vs.len());
    let d = PyDict::new(py);
    let flat3 = |f: &dyn Fn(&sylva_rs::shots::moments::Element) -> [f64; 3]| el.iter().flat_map(f).collect::<Vec<f64>>();
    d.set_item("key", el.iter().flat_map(|e| e.key).collect::<Vec<i32>>().into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("point", flat3(&|e| [e.point[0], e.point[1], e.point[2]]).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("u", flat3(&|e| [e.u[0], e.u[1], e.u[2]]).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("lam", flat3(&|e| e.lam).into_pyarray(py).reshape([k, 3])?)?;
    d.set_item("r", el.iter().map(|e| e.r).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("ratio", el.iter().map(|e| e.ratio).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n", el.iter().map(|e| e.n).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan_n", el.iter().flat_map(|e| e.scan_n.clone()).collect::<Vec<_>>().into_pyarray(py).reshape([k, s])?)?;
    d.set_item("arc", el.iter().flat_map(|e| e.arc.clone()).collect::<Vec<_>>().into_pyarray(py).reshape([k, s])?)?;
    d.set_item("excess", el.iter().flat_map(|e| e.excess.clone()).collect::<Vec<_>>().into_pyarray(py).reshape([k, s])?)?;
    Ok(d)
}

/// One view's patch excess (moments::Mom::excess) at each point: its surfels' moments in the point's 27-cell
/// neighbourhood on `voxel` cells.
#[pyfunction]
#[pyo3(signature = (view, points, voxel=0.03))]
fn l0_excess<'py>(py: Python<'py>, view: PyRef<'py, PyView>, points: PyReadonlyArray2<f64>, voxel: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = xyz_from_py(points)?;
    let v = &view.inner;
    let out = py.detach(|| {
        let lo = [0, 1, 2].map(|a| p.iter().map(|x| x[a]).fold(f64::INFINITY, f64::min) - 3.0 * voxel);
        let hi = [0, 1, 2].map(|a| p.iter().map(|x| x[a]).fold(f64::NEG_INFINITY, f64::max) + 3.0 * voxel);
        let m = moments_of(&[v], lo, hi, voxel);
        let step = ErrorModel::default().step;
        p.iter().map(|x| m.neighbourhood(&key(&nalgebra::Vector3::new(x[0], x[1], x[2]), voxel))[0].excess(&m.origins[0], step)).collect::<Vec<f64>>()
    });
    Ok(out.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyView>()?;
    m.add_function(wrap_pyfunction!(l0_energy, m)?)?;
    m.add_function(wrap_pyfunction!(l0_predict, m)?)?;
    m.add_function(wrap_pyfunction!(l0_elements, m)?)?;
    m.add_function(wrap_pyfunction!(l0_excess, m)?)?;
    m.add_function(wrap_pyfunction!(l0_error_model, m)?)?;
    Ok(())
}
