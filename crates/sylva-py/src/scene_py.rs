// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for range images, candidates and the scene (sylva_rs::scene).
//!
//! - **Error model and background:** flat dicts whose keys are the structs' fields (arrays as lists).
//!   `error_model_defaults()` returns the error model's.
//! - **Elements:** a dict of arrays, one row per element: `a`, `b` (n, 3) and `r` (n,) are required; `opacity` (n,)
//!   defaults to 1, `misfit` (n,) to the error model's prior `misfit[0] + misfit[1] r`, and `patch` (n,) bool to
//!   false. A patch is a ball about `a` of radius `r` (its extent); its `b` is ignored.
//! - **Scenes:** `Scene(images, elements, err, background)` holds elements against range images: `cost()`, `rows()`,
//!   the cost change of each move (`add`, `remove`, `change`, `cost_to_add`), and an element's counts (`counts`)
//!   and cell predictions (`predict`).

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use sylva_rs::scene::model::{Background, ErrorModel};
use sylva_rs::scene::moments::{Candidate, Moments};
use sylva_rs::scene::range_image::{RangeImage, RangeImageParams};
use sylva_rs::scene::scene::{CellPrediction, Element, ElementScan, Form, Scene};

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
            "exit" => m.exit = x(&v)?,
            "divergence" => m.divergence = x(&v)?,
            "step" => m.step = x(&v)?,
            "class_edges" => m.class_edges = v.extract()?,
            "sigma_e" => m.sigma_e = v.extract()?,
            "nu_e" => m.nu_e = x(&v)?,
            "bias" => m.bias = v.extract()?,
            "bias_ref" => m.bias_ref = x(&v)?,
            "field_spacing" => m.field_spacing = x(&v)?,
            "field_along" => m.field_along = v.extract()?,
            "field_epoch" => m.field_epoch = v.extract()?,
            "cut_db" => m.cut_db = x(&v)?,
            "refl_noise_db" => m.refl_noise_db = x(&v)?,
            "p_drop" => m.p_drop = x(&v)?,
            "p_outlier" => m.p_outlier = x(&v)?,
            "misfit" => m.misfit = v.extract()?,
            "dispersion" => m.dispersion = x(&v)?,
            _ => return Err(bad(&k)),
        }
    }
    Ok(m)
}

/// The error model's defaults (AM's calibration), as the dict the other calls take.
#[pyfunction]
fn error_model_defaults<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {    let m = ErrorModel::default();
    let d = PyDict::new(py);
    for (k, v) in [("exit", m.exit), ("divergence", m.divergence), ("step", m.step), ("nu_e", m.nu_e), ("bias_ref", m.bias_ref),
                   ("field_spacing", m.field_spacing), ("cut_db", m.cut_db), ("refl_noise_db", m.refl_noise_db), ("p_drop", m.p_drop),
                   ("p_outlier", m.p_outlier), ("dispersion", m.dispersion)] {
        d.set_item(k, v)?;
    }
    d.set_item("class_edges", m.class_edges.to_vec())?;
    d.set_item("sigma_e", m.sigma_e.to_vec())?;
    d.set_item("bias", m.bias.to_vec())?;
    d.set_item("field_along", m.field_along.to_vec())?;
    d.set_item("field_epoch", m.field_epoch.to_vec())?;
    d.set_item("misfit", m.misfit.to_vec())?;
    Ok(d)
}

fn background(d: Option<&Bound<'_, PyDict>>) -> PyResult<Background> {
    let mut s = Background::default();    for (k, v) in d.into_iter().flat_map(|d| d.iter()) {
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

/// One scan as a range image: its pulses in acquisition order (`time` s, `direction` world unit vectors,
/// first-echo `range` with NaN for a pulse without one), its pose `rot` (3 × 3, world = rot · scanner + origin) and
/// `epoch`.
#[pyclass(name = "RangeImage", frozen)]
struct PyRangeImage {
    inner: Arc<RangeImage>,
}

#[pymethods]
impl PyRangeImage {
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
        let p = RangeImageParams::default();
        let inner = py.detach(|| RangeImage::new(&t, &dir, &r, rot, [origin.0, origin.1, origin.2], epoch, &p));
        if inner.n_lines == 0 {
            return Err(PyValueError::new_err("no lattice: too few pulses"));
        }
        Ok(PyRangeImage { inner: Arc::new(inner) })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_lines, self.inner.n_cols)
    }

    /// The export's in-surface deletion measured on this scan (RangeImage::deletion).
    #[getter]
    fn deletion(&self) -> f64 {
        self.inner.deletion
    }

    /// The cell arrays, each (n_lines, n_cols[, k]): state, range, pulse (-1 none), normal, cos_inc, excess; and
    /// from the error model `err`, surface (class).
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
        d.set_item("surface", v.excess.iter().map(|&e| em.surface(e as f64)).collect::<Vec<u8>>().into_pyarray(py).reshape(sh)?)?;
        Ok(d)
    }

    /// The fractional (line, col) and range of each world point (NaN outside the lattice), flat (n × 3).
    fn cells<'py>(&self, py: Python<'py>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = xyz_from_py(points)?;
        let out: Vec<f64> = p.iter().flat_map(|x| self.inner.cell_of(x).map_or([f64::NAN; 3], |(l, c, r)| [l, c, r])).collect();
        Ok(out.into_pyarray(py))
    }
}

fn predictions_dict<'py>(py: Python<'py>, p: &[CellPrediction]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("cell", p.iter().map(|x| x.cell as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", p.iter().map(|x| x.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("outcome", p.iter().map(|x| x.outcome as u8).collect::<Vec<_>>().into_pyarray(py))?;
    type F = fn(&CellPrediction) -> f64;
    for (k, f) in [("fill", (|x| x.fill) as F), ("kept", |x| x.kept), ("depth", |x| x.depth), ("sd", |x| x.sd), ("obs", |x| x.obs),
                   ("weight", |x| x.weight), ("footprint", |x| x.footprint), ("x", |x| x.x), ("s", |x| x.s)] {
        d.set_item(k, p.iter().map(f).collect::<Vec<_>>().into_pyarray(py))?;
    }
    Ok(d)
}

fn rows_dict<'py>(py: Python<'py>, rows: &[ElementScan]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("elem", rows.iter().map(|r| r.elem).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", rows.iter().map(|r| r.scan).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cells", rows.iter().map(|r| r.cells).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("support", rows.iter().map(|r| r.support).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("across", rows.iter().map(|r| r.across).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", rows.iter().map(|r| r.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cost", rows.iter().map(|r| r.cost).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("shift_along", rows.iter().map(|r| r.shift_along).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("sd_along", rows.iter().map(|r| r.sd_along).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

/// The candidate elements read off the images' surfel moments (moments::Moments::candidates) inside the box lo-hi,
/// on `voxel` cells: key (k, 3), point, u, lam (k, 3), r, ratio, n, extent (k,), and per scan (k, S) scan_n and arc.
#[pyfunction]
#[pyo3(signature = (images, lo, hi, voxel=0.03, min_n=30.0, r_max=0.3))]
fn candidates<'py>(py: Python<'py>, images: Vec<PyRef<'py, PyRangeImage>>, lo: [f64; 3], hi: [f64; 3], voxel: f64, min_n: f64, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    use nalgebra::Vector3;
    use rayon::prelude::*;
    let vs: Vec<&RangeImage> = images.iter().map(|v| &*v.inner).collect();
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
    m.add_class::<PyRangeImage>()?;
    m.add_class::<PyScene>()?;
    m.add_function(wrap_pyfunction!(candidates, m)?)?;
    m.add_function(wrap_pyfunction!(error_model_defaults, m)?)?;
    Ok(())
}

/// A scene of elements held against range images (scene::Scene). Each element is rendered once per image, and a
/// move recomputes only the offset-field components it touches.
#[pyclass(name = "Scene")]
struct PyScene {
    inner: Scene,
    em: ErrorModel,
}

#[pymethods]
impl PyScene {
    #[new]
    #[pyo3(signature = (images, elements, err=None, background=None))]
    fn new(py: Python<'_>, images: Vec<PyRef<'_, PyRangeImage>>, elements: &Bound<'_, PyDict>, err: Option<&Bound<'_, PyDict>>,
           background: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let (em, bg) = (error_model(err)?, self::background(background)?);
        let el = elements_from_py(elements, &em)?;
        let vs: Vec<Arc<RangeImage>> = images.iter().map(|v| v.inner.clone()).collect();
        Ok(PyScene { inner: py.detach(|| Scene::new(vs, &el, em, bg)), em })
    }

    /// The scene's cost: −log of its likelihood ratio against the empty scene.
    fn cost(&self) -> f64 {
        self.inner.cost()
    }

    /// Per (element, scan): elem, scan, cells, support, footprints across, surface class, its share of the cost,
    /// and the offset field's posterior at it (shift_along, sd_along).
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        rows_dict(py, &self.inner.rows())
    }

    /// Add elements: their ids and the cost change.
    fn add(&mut self, py: Python<'_>, elements: &Bound<'_, PyDict>) -> PyResult<(Vec<u32>, f64)> {
        let el = elements_from_py(elements, &self.em)?;
        let c = &mut self.inner;
        Ok(py.detach(|| el.into_iter().fold((Vec::new(), 0.0), |(mut ids, d), e| {
            let (id, de) = c.add(e);
            ids.push(id);
            (ids, d + de)
        })))
    }

    /// Remove elements by id: the cost change.
    fn remove(&mut self, py: Python<'_>, ids: Vec<u32>) -> f64 {
        let c = &mut self.inner;
        py.detach(|| ids.into_iter().map(|id| c.remove(id)).sum())
    }

    /// Replace an element: the cost change.
    fn change(&mut self, py: Python<'_>, id: u32, element: &Bound<'_, PyDict>) -> PyResult<f64> {
        let e = one(elements_from_py(element, &self.em)?)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.change(id, e)))
    }

    /// The cost change of adding one element, the scene left as it was.
    fn cost_to_add(&mut self, py: Python<'_>, element: &Bound<'_, PyDict>) -> PyResult<f64> {
        let e = one(elements_from_py(element, &self.em)?)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.cost_to_add(e)))
    }

    /// One element's counts against the scene, per scan (scene::ElementCounts): scan, misfit_ss, misfit_w,
    /// misfit_cells, on, empty, behind; `shift` (m, per scan) comes off the depth residuals.
    #[pyo3(signature = (element, shift=None))]
    fn counts<'py>(&self, py: Python<'py>, element: &Bound<'_, PyDict>, shift: Option<Vec<f64>>) -> PyResult<Bound<'py, PyDict>> {
        let e = one(elements_from_py(element, &self.em)?)?;
        let c = &self.inner;
        let st = py.detach(|| c.counts(e, &shift.unwrap_or_default()));
        let d = PyDict::new(py);
        d.set_item("scan", st.iter().map(|x| x.scan).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("misfit_ss", st.iter().map(|x| x.misfit_ss).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("misfit_w", st.iter().map(|x| x.misfit_w).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("misfit_cells", st.iter().map(|x| x.misfit_cells).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("on", st.iter().map(|x| x.on).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("empty", st.iter().map(|x| x.empty).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("behind", st.iter().map(|x| x.behind).collect::<Vec<_>>().into_pyarray(py))?;
        Ok(d)
    }

    /// What one element predicts in scan `scan` against the scene, per cell it fronts: cell (flat index), surface,
    /// outcome (0 on, 1 empty, 2 in front, 3 behind), fill, kept, depth, sd, obs, weight (1/m), footprint (m), x and
    /// s (where the ray passes the axis, across and along).
    #[pyo3(signature = (element, scan=0))]
    fn predict<'py>(&self, py: Python<'py>, element: &Bound<'_, PyDict>, scan: usize) -> PyResult<Bound<'py, PyDict>> {
        let e = one(elements_from_py(element, &self.em)?)?;
        let c = &self.inner;
        let p = py.detach(|| c.predict(e, scan));
        predictions_dict(py, &p)
    }
}

fn one(el: Vec<Element>) -> PyResult<Element> {
    match el.as_slice() {
        [e] => Ok(*e),
        _ => Err(PyValueError::new_err("expected one element")),
    }
}
