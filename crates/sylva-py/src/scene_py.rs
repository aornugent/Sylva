// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for range images, candidates and the scene (sylva_rs::scene; its module documentation holds the
//! glossary these names follow).
//!
//! - **Settings and background:** flat dicts of the records' fields (`Settings`, `Background`), each optional;
//!   `settings_defaults()` returns the settings' defaults (AM's). An unknown key raises. A few of the scene's rules for
//!   what consumers read beside the cost: `footprint`, `floor`, `p_detect` (the detection's chance), `mismatch_sd`
//!   (the shape mismatch's prior), `beam_fill` (a fill and its slope in ln r), `field_ln_nb` (the foliage field's
//!   marginal per voxel) and `alignment_hats` (the scan alignment error's nodes about points).
//! - **Elements:** a dict of arrays, one row per element: `a`, `b` (n, 3) and `r` (n,) are required; `opacity` (n,)
//!   defaults to 1, `mismatch` (n,) to the settings' prior `mismatch_base + mismatch_slope r`, `patch` (n,) bool
//!   to false, `leaf` (n,) bool, its class (wood or leaf), to `patch`, and `group` (n,) int, its chain's key
//!   (elements of one key are one term per scan: one set of nuisances), to -1, alone. A patch is a ball about `a` of radius `r`
//!   (its extent); its `b` is ignored. An element that can't be scored (a radius that isn't positive, an opacity
//!   outside (0, 1], a stick of no length) raises.
//! - **Scenes:** `Scene(images, elements, settings, background, foliage, ref_epoch)` holds elements against range
//!   images, the alignment error's spread stated for the scans of epoch `ref_epoch` (another epoch's add `align_epoch_*`): `cost()`, `rows()`, each scan's `alignment_parts()` and the foliage field's `field_part()`, the cost change of each move (`add`,
//!   `remove`, `change`, `costs_to_add`, `costs_to_remove`) with the alignment held, `refit_alignment()` and
//!   `exact_cost()`, which hypotheses interact (`interactions`), an element's counts (`counts`), cell predictions (`predict`), what one
//!   hypothesis is expected to cost against another (`divergence`) and how well the scans pin elements down, one or
//!   several jointly (`information`). A bad element id or scan index raises.
//! - **The foliage field:** `foliage` is a dict `lo`, `hi` (the box, 3 each) and `voxel` (m; its prior is the
//!   settings' `field_prior_*`); the field is then what the elements leave of each echo there (`FoliageBox`; soft
//!   attribution, 3dtrees research 130 slice-soft), its marginal in
//!   the cost and what a pulse meets of it held through moves. `refit_foliage()`,
//!   `promote(element)` and `demote(id)` move mass between the field and the elements; `foliage_voxels()` gives it
//!   per voxel and `transmission(image, cells)` its optical depth across the box along other images' rays.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use sylva_rs::scene::chain::{Coupling, Grow};
use sylva_rs::scene::foliage::FoliageBox;
use sylva_rs::scene::moments::{Candidate, Moments};
use sylva_rs::scene::pyramid::Pyramid;
use sylva_rs::scene::range_image::{Cell, RangeImage, RangeImageParams};
use sylva_rs::scene::query::{CellPrediction, Information};
use sylva_rs::scene::render::{Class, Element, Form};
use sylva_rs::scene::scene::{ElementScan, Scene, SceneError};
use sylva_rs::scene::settings::{Background, Settings};

use crate::xyz_from_py;

fn bad(k: &str) -> PyErr {
    PyValueError::new_err(format!("unknown or malformed key {k:?}"))
}

fn scene_err(e: SceneError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// A settings record from a flat dict of its fields, each optional (the rest at their defaults).
fn record<T: Default>(d: Option<&Bound<'_, PyDict>>, set: fn(&mut T, &str, f64) -> bool) -> PyResult<T> {
    let mut m = T::default();
    for (k, v) in d.into_iter().flat_map(|d| d.iter()) {
        let k: String = k.extract()?;
        if !set(&mut m, &k, v.extract().map_err(|_| bad(&k))?) {
            return Err(bad(&k));
        }
    }
    Ok(m)
}

fn settings(d: Option<&Bound<'_, PyDict>>) -> PyResult<Settings> {
    record(d, Settings::set)
}

fn background(d: Option<&Bound<'_, PyDict>>) -> PyResult<Background> {
    record(d, Background::set)
}

/// The settings at their defaults (AM's calibration), as the flat dict the other calls take.
#[pyfunction]
fn settings_defaults<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let (m, d) = (Settings::default(), PyDict::new(py));
    for &k in Settings::FIELDS {
        d.set_item(k, m.get(k))?;
    }
    Ok(d)
}

/// The beam's diameter (m) at each range (`Settings::footprint`), under `settings` (the defaults without).
#[pyfunction]
#[pyo3(signature = (range, settings=None))]
fn footprint<'py>(py: Python<'py>, range: PyReadonlyArray1<f64>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let em = self::settings(settings)?;
    Ok(range.as_slice()?.iter().map(|&t| em.footprint(t)).collect::<Vec<_>>().into_pyarray(py))
}

/// The detection floor (dB) at each range (`Settings::floor`), under `settings` (the defaults without).
#[pyfunction]
#[pyo3(signature = (range, settings=None))]
fn floor<'py>(py: Python<'py>, range: PyReadonlyArray1<f64>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let em = self::settings(settings)?;
    Ok(range.as_slice()?.iter().map(|&t| em.floor(t)).collect::<Vec<_>>().into_pyarray(py))
}

/// The chance an echo is recorded (`Settings::p_detect`) off a surface of reflectance `db` (dB, at full fill and face-on)
/// taking energy share `share` (its fill × cos^k ι) at range `range` (arrays alike), under `settings`.
#[pyfunction]
#[pyo3(signature = (db, share, range, settings=None))]
fn p_detect<'py>(py: Python<'py>, db: PyReadonlyArray1<f64>, share: PyReadonlyArray1<f64>, range: PyReadonlyArray1<f64>,
                 settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let em = self::settings(settings)?;
    let (d, e, t) = (db.as_slice()?, share.as_slice()?, range.as_slice()?);
    if d.len() != e.len() || t.len() != e.len() {
        return Err(PyValueError::new_err("p_detect: db, share and range need one value each"));
    }
    Ok(d.iter().zip(e).zip(t).map(|((&d, &e), &t)| em.p_detect::<f64, f64>(d, e, t)).collect::<Vec<_>>().into_pyarray(py))
}

/// An element's shape mismatch prior sd (m) at each radius (`Settings::mismatch_sd`), under `settings`.
#[pyfunction]
#[pyo3(signature = (r, settings=None))]
fn mismatch_sd<'py>(py: Python<'py>, r: PyReadonlyArray1<f64>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let em = self::settings(settings)?;
    Ok(r.as_slice()?.iter().map(|&r| em.mismatch_sd(r)).collect::<Vec<_>>().into_pyarray(py))
}

/// The scan alignment error's nodes about each point (scene::hats: normalised trilinear hats on a grid of `spacing` m):
/// keys (n, 8, 3) and weights (n, 8), a point's unused slots weight 0.
#[pyfunction]
#[pyo3(signature = (points, spacing=sylva_rs::scene::settings::ALIGN_SPACING))]
fn alignment_hats<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, spacing: f64) -> PyResult<(Bound<'py, PyArray3<i64>>, Bound<'py, PyArray2<f64>>)> {
    let p = xyz_from_py(points)?;
    let n = p.len();
    let (mut keys, mut ws) = (vec![0i64; n * 24], vec![0.0; n * 8]);
    for (i, x) in p.iter().enumerate() {
        for (j, (k, w)) in sylva_rs::scene::alignment::hats(x, spacing).into_iter().enumerate() {
            keys[(i * 8 + j) * 3..(i * 8 + j) * 3 + 3].copy_from_slice(&k);
            ws[i * 8 + j] = w;
        }
    }
    Ok((keys.into_pyarray(py).reshape([n, 8, 3])?, ws.into_pyarray(py).reshape([n, 8])?))
}

/// A layer's fill at one section radius `r` and its slope in ln r (`scene::beam_fill`), its axis `d` from the centre of
/// a beam of radius `a` (arrays alike).
#[pyfunction]
fn beam_fill<'py>(py: Python<'py>, d: PyReadonlyArray1<f64>, r: f64, a: PyReadonlyArray1<f64>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (f, g): (Vec<f64>, Vec<f64>) = d.as_slice()?.iter().zip(a.as_slice()?).map(|(&d, &a)| sylva_rs::scene::render::beam_fill(d, r, a)).unzip();
    Ok((f.into_pyarray(py), g.into_pyarray(py)))
}

/// Each voxel's `ln NB(hits, path)` under the foliage field's Gamma prior of mean `mean` worth `prior_path` m of
/// weighted ray (`foliage::ln_nb`; 0 with no counts).
#[pyfunction]
fn field_ln_nb<'py>(py: Python<'py>, hits: PyReadonlyArray1<f64>, path: PyReadonlyArray1<f64>, mean: f64, prior_path: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(hits.as_slice()?.iter().zip(path.as_slice()?).map(|(&h, &l)| sylva_rs::scene::foliage::ln_nb(h, l, mean * prior_path, prior_path))
        .collect::<Vec<_>>().into_pyarray(py))
}

/// The coupling row (scene::chain::Coupling::row) along sticks (n, 7: a, b, r) joined end to end in order: length and
/// gap (m), and the declared coupling's cost of their radii and bends (nats), beside the cost, never in it.
#[pyfunction]
fn coupling<'py>(py: Python<'py>, sticks: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let st = sticks.as_array();
    if st.ncols() != 7 || st.nrows() == 0 {
        return Err(PyValueError::new_err("coupling needs sticks (n, 7), n > 0"));
    }
    let rows: Vec<([f64; 3], [f64; 3], f64)> = st.rows().into_iter().map(|x| ([x[0], x[1], x[2]], [x[3], x[4], x[5]], x[6])).collect();
    let c = Coupling::default().row(&rows);
    let d = PyDict::new(py);
    for (k, v) in [("length", c.length), ("gap", c.gap), ("radius", c.radius), ("bend", c.bend)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Sticks as rows (n, 7: a, b, r).
fn sticks_of<'py>(py: Python<'py>, es: &[Element]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = es.iter().flat_map(|e| match e.form {
        Form::Stick { a, b } => [a[0], a[1], a[2], b[0], b[1], b[2], e.r],
        Form::Patch { centre: c } => [c[0], c[1], c[2], c[0], c[1], c[2], e.r],
    }).collect();
    Ok(flat.into_pyarray(py).reshape([es.len(), 7])?)
}

fn elements_from_py(d: &Bound<'_, PyDict>, em: &Settings) -> PyResult<Vec<Element>> {
    let get = |k: &str| d.get_item(k);
    let req = |k: &str| get(k)?.ok_or_else(|| PyValueError::new_err(format!("elements need {k:?}")));
    for k in d.keys() {
        let k: String = k.extract()?;
        if !["a", "b", "r", "opacity", "mismatch", "patch", "leaf", "group"].contains(&k.as_str()) {
            return Err(bad(&k));
        }
    }
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
    let mismatch = col("mismatch", &|i| em.mismatch_sd(r[i]))?;
    let flag = |k: &str, dflt: Vec<bool>| -> PyResult<Vec<bool>> {
        Ok(match get(k)? {
            Some(v) => v.extract::<PyReadonlyArray1<bool>>()?.as_array().to_vec(),
            None => dflt,
        })
    };
    let patch = flag("patch", vec![false; r.len()])?;
    let leaf = flag("leaf", patch.clone())?;
    let group: Vec<i64> = match get("group")? {
        Some(v) => v.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec(),
        None => vec![-1; r.len()],
    };
    let n = r.len();
    if [a.len(), b.len(), opacity.len(), mismatch.len(), patch.len(), leaf.len(), group.len()].iter().any(|&m| m != n) {
        return Err(PyValueError::new_err("elements: every array needs one row per element"));
    }
    (0..n)
        .map(|i| {
            let e = Element {
                form: if patch[i] { Form::Patch { centre: a[i] } } else { Form::Stick { a: a[i], b: b[i] } },
                class: if leaf[i] { Class::Leaf } else { Class::Wood },
                r: r[i], opacity: opacity[i], mismatch: mismatch[i], group: u32::try_from(group[i]).ok(),
            };
            match e.invalid() {
                Some(why) => Err(PyValueError::new_err(format!("element {i}: {why}"))),
                None => Ok(e),
            }
        })
        .collect()
}

/// One scan as a range image: `RangeImage.from_file(path, epoch)` reads a shots file (every echo of every pulse
/// that returned) and its sidecar's pose; `RangeImage(time, direction, range, rot, origin, epoch)` takes pulses with at
/// most one echo each (`range` NaN for none) in acquisition order, `direction` world unit vectors and the pose `rot`
/// (3 × 3, world = rot · scanner + origin).
#[pyclass(name = "RangeImage", frozen)]
struct PyRangeImage {
    inner: Arc<RangeImage>,
    /// Its depth pyramid, built on first use under the settings it was asked with (rebuilt for others).
    pyramid: std::sync::Mutex<Option<(String, Arc<Pyramid>)>>,
}

impl PyRangeImage {
    fn of(inner: RangeImage) -> Self {
        PyRangeImage { inner: Arc::new(inner), pyramid: std::sync::Mutex::new(None) }
    }

    /// The image's depth pyramid under `em`, built once.
    fn pyramid_of(&self, py: Python<'_>, em: &Settings) -> Arc<Pyramid> {
        let key = format!("{:?}", Settings::FIELDS.iter().map(|k| em.get(k)).collect::<Vec<Option<f64>>>());
        let mut slot = self.pyramid.lock().unwrap();
        if let Some((_, p)) = slot.as_ref().filter(|(k, _)| *k == key) {
            return p.clone();
        }
        let v = &self.inner;
        let p = Arc::new(py.detach(|| Pyramid::new(v, em)));
        *slot = Some((key, p.clone()));
        p
    }
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
        let inner = py.detach(|| RangeImage::from_first_echoes(&t, &dir, &r, rot, [origin.0, origin.1, origin.2], epoch, &p));
        if inner.n_lines == 0 {
            return Err(PyValueError::new_err("no lattice: too few pulses"));
        }
        Ok(PyRangeImage::of(inner))
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_lines, self.inner.n_cols)
    }

    /// The lattice step (rad), as this scan's timing measures it.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step()
    }

    /// One scan's range image from its shots file and the pose in its sidecar (`.json` beside it).
    #[staticmethod]
    #[pyo3(signature = (path, epoch=0))]
    fn from_file(py: Python<'_>, path: std::path::PathBuf, epoch: u32) -> PyResult<Self> {
        let p = RangeImageParams::default();
        let inner = py.detach(|| RangeImage::from_file(&path, epoch, &p)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRangeImage::of(inner))
    }

    /// Writes the image to `path`, a cache for `RangeImage.load` on this machine.
    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.save(&path)).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The image `save` wrote; raises for a file from a build with another `build_version`.
    #[staticmethod]
    fn load(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| RangeImage::load(&path)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRangeImage::of(inner))
    }

    /// A hash of the code that builds images: a saved image loads only into a build with the same one.
    #[staticmethod]
    fn build_version() -> u64 {
        sylva_rs::scene::range_image::build_version()
    }

    /// The cell arrays, each (n_lines, n_cols[, k]): cell (0 echoes, 1 miss, 2 no pulse), echoes (their count),
    /// range and refl (the first echo's; refl in dB, NaN unknown), pulse (-1 none), normal, cos_inc, excess; and,
    /// under the `settings`, surface (its class by the class edges) and free_to (the range up to which its pulse shows
    /// clear space: scene::settings::Settings::free_to; NaN where no pulse).
    #[pyo3(signature = (settings=None))]
    fn arrays<'py>(&self, py: Python<'py>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        let em = self::settings(settings)?;
        let v = &self.inner;
        let sh = [v.n_lines, v.n_cols];
        let d = PyDict::new(py);
        d.set_item("cell", v.cell.iter().map(|&c| c as u8).collect::<Vec<u8>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("echoes", (0..v.cell.len()).map(|k| v.echoes(k).len() as u32).collect::<Vec<u32>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("range", v.range.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("refl", (0..v.cell.len()).map(|k| v.echo_refls(k).first().copied().unwrap_or(f32::NAN)).collect::<Vec<f32>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("pulse", v.pulse.iter().map(|&p| if p == u32::MAX { -1 } else { p as i64 }).collect::<Vec<_>>().into_pyarray(py).reshape(sh)?)?;
        d.set_item("normal", v.normal.iter().flatten().copied().collect::<Vec<f32>>().into_pyarray(py).reshape([sh[0], sh[1], 3])?)?;
        d.set_item("cos_inc", v.cos_inc.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("excess", v.excess.clone().into_pyarray(py).reshape(sh)?)?;
        d.set_item("surface", v.excess.iter().map(|&e| em.surface(e as f64)).collect::<Vec<u8>>().into_pyarray(py).reshape(sh)?)?;
        let free = |k: usize| if v.cell[k] == Cell::NoPulse { f64::NAN } else { em.free_to(v.range[k] as f64) };
        d.set_item("free_to", (0..v.cell.len()).map(free).collect::<Vec<f64>>().into_pyarray(py).reshape(sh)?)?;
        Ok(d)
    }

    /// The depth pyramid (scene::pyramid), built once per image under `settings`: its kept levels' shapes (lines,
    /// columns) and, over the whole image, pulses, misses, echoes and energy.
    #[pyo3(signature = (settings=None))]
    fn pyramid<'py>(&self, py: Python<'py>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        let em = self::settings(settings)?;
        let p = self.pyramid_of(py, &em);
        let (shapes, t) = (p.shapes(), p.total());
        let d = PyDict::new(py);
        d.set_item("levels", shapes)?;
        d.set_item("pulses", t.pulses)?;
        d.set_item("misses", t.misses)?;
        d.set_item("echoes", t.echoes)?;
        d.set_item("energy", t.energy)?;
        Ok(d)
    }

    /// Per voxel of the box `lo`-`hi` (m, `voxel` m cubes; (nx, ny, nz) counts) under `settings`, read off the image's
    /// depth pyramid (scene::pyramid::carve): `pulses` in its cover (the cells its corners project to, a cell wider),
    /// those `clear` past its far side by the receiver's resolution (or with no echo), and those `ending` in it or
    /// within the resolution behind it. A voxel is carved where enough pulses cover it and all are clear.
    #[pyo3(signature = (lo, hi, voxel, settings=None))]
    fn carve<'py>(&self, py: Python<'py>, lo: [f64; 3], hi: [f64; 3], voxel: f64, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        if !(voxel > 0.0) || (0..3).any(|a| !(hi[a] > lo[a])) {
            return Err(PyValueError::new_err("carve needs voxel > 0 and hi > lo"));
        }
        let em = self::settings(settings)?;
        let n = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / voxel).ceil() as usize);
        let p = self.pyramid_of(py, &em);
        let v = &self.inner;
        let counts = py.detach(|| sylva_rs::scene::pyramid::carve(v, &p, lo, hi, voxel, &em));
        let d = PyDict::new(py);
        for (name, c) in ["pulses", "clear", "ending"].into_iter().zip(counts) {
            d.set_item(name, c.into_pyarray(py).reshape(n)?)?;
        }
        Ok(d)
    }

    /// Per stick (n, 7: a, b, r), the pulses in its cover and those that reach its middle's range (a first echo beyond
    /// it, or none), off the depth pyramid (scene::pyramid::arrival): (n, 2).
    #[pyo3(signature = (sticks, settings=None))]
    fn arrival<'py>(&self, py: Python<'py>, sticks: PyReadonlyArray2<f64>, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let em = self::settings(settings)?;
        let st = sticks.as_array();
        if st.ncols() != 7 {
            return Err(PyValueError::new_err("arrival needs sticks (n, 7)"));
        }
        let rows: Vec<([f64; 3], [f64; 3], f64)> = st.rows().into_iter().map(|x| ([x[0], x[1], x[2]], [x[3], x[4], x[5]], x[6])).collect();
        let p = self.pyramid_of(py, &em);
        let v = &self.inner;
        let out = py.detach(|| sylva_rs::scene::pyramid::arrival(v, &p, &rows));
        Ok(out.iter().flat_map(|x| [x.0, x.1]).collect::<Vec<u64>>().into_pyarray(py).reshape([out.len(), 2])?)
    }

    /// The fractional (line, col) and range of each world point (NaN outside the lattice), flat (n × 3).
    fn cells<'py>(&self, py: Python<'py>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = xyz_from_py(points)?;
        let out: Vec<f64> = p.iter().flat_map(|x| self.inner.cell_of(x).map_or([f64::NAN; 3], |(l, c, r)| [l, c, r])).collect();
        Ok(out.into_pyarray(py))
    }

    /// Each cell's beam direction (world unit vector) for flat cell indices, flat (n × 3).
    fn directions<'py>(&self, py: Python<'py>, cells: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = &self.inner;
        let n = (v.n_lines * v.n_cols) as i64;
        let cs = cells.as_array();
        if let Some(k) = cs.iter().find(|&&k| k < 0 || k >= n) {
            return Err(PyValueError::new_err(format!("no cell {k}")));
        }
        let out: Vec<f64> = cs.iter().flat_map(|&k| {
            let k = k as usize;
            v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64)
        }).collect();
        Ok(out.into_pyarray(py))
    }
}

fn predictions_dict<'py>(py: Python<'py>, p: &[CellPrediction]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("cell", p.iter().map(|x| x.cell as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("elem", p.iter().map(|x| x.elem).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", p.iter().map(|x| x.surface).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("outcome", p.iter().map(|x| x.outcome as u8).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("read", p.iter().map(|x| x.read).collect::<Vec<_>>().into_pyarray(py))?;
    type F = fn(&CellPrediction) -> f64;
    for (k, f) in [("fill", (|x| x.fill) as F), ("recorded", |x| x.recorded), ("arrives", |x| x.arrives), ("depth", |x| x.depth), ("sd", |x| x.sd),
                   ("refl", |x| x.refl), ("cos", |x| x.cos), ("merge", |x| x.merge), ("obs", |x| x.obs), ("obs_refl", |x| x.obs_refl), ("weight", |x| x.weight), ("footprint", |x| x.footprint), ("across", |x| x.across), ("along", |x| x.along),
                   ("returns", |x| x.returns), ("beam", |x| x.beam), ("owned", |x| x.owned)] {
        d.set_item(k, p.iter().map(f).collect::<Vec<_>>().into_pyarray(py))?;
    }
    Ok(d)
}

fn rows_dict<'py>(py: Python<'py>, rows: &[ElementScan]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("elem", rows.iter().map(|r| r.elem).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("group", rows.iter().map(|r| r.group.map_or(-1, i64::from)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", rows.iter().map(|r| r.scan).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cells", rows.iter().map(|r| r.cells).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("cells_on", rows.iter().map(|r| r.cells_on).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("surface", rows.iter().map(|r| r.surface).collect::<Vec<_>>().into_pyarray(py))?;
    type F = fn(&ElementScan) -> f64;
    for (k, f) in [("arc", (|r| r.arc) as F), ("footprints_across", |r| r.footprints_across), ("cost", |r| r.cost), ("ray_on", |r| r.ray_on), ("ray_empty", |r| r.ray_empty),
                   ("ray_in_front", |r| r.ray_in_front), ("ray_behind", |r| r.ray_behind), ("alignment", |r| r.alignment),
                   ("field", |r| r.field), ("discount", |r| r.discount), ("align_shift", |r| r.align_shift), ("align_sd", |r| r.align_sd), ("front_shift", |r| r.front_shift),
                   ("front_shift_sd", |r| r.front_shift_sd), ("refl_level", |r| r.refl_level), ("refl_level_sd", |r| r.refl_level_sd)] {
        d.set_item(k, rows.iter().map(f).collect::<Vec<_>>().into_pyarray(py))?;
    }
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
    m.add_function(wrap_pyfunction!(settings_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(footprint, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(beam_fill, m)?)?;
    m.add_function(wrap_pyfunction!(p_detect, m)?)?;
    m.add_function(wrap_pyfunction!(mismatch_sd, m)?)?;
    m.add_function(wrap_pyfunction!(alignment_hats, m)?)?;
    m.add_function(wrap_pyfunction!(field_ln_nb, m)?)?;
    m.add_function(wrap_pyfunction!(coupling, m)?)?;
    Ok(())
}

/// A scene of elements held against range images (scene::Scene). Each element is rendered once per image; moves and
/// queries hold the scan alignment error (its mode and precision) and recompute only the elements whose cells they
/// touch, and `refit_alignment` re-solves it.
#[pyclass(name = "Scene")]
struct PyScene {
    inner: Scene,
    em: Settings,
}

impl PyScene {
    fn one(&self, element: &Bound<'_, PyDict>) -> PyResult<Element> {
        match elements_from_py(element, &self.em)?.as_slice() {
            [e] => Ok(*e),
            _ => Err(PyValueError::new_err("expected one element")),
        }
    }
}

#[pymethods]
impl PyScene {
    #[new]
    #[pyo3(signature = (images, elements, settings=None, background=None, foliage=None, ref_epoch=0))]
    fn new(py: Python<'_>, images: Vec<PyRef<'_, PyRangeImage>>, elements: &Bound<'_, PyDict>, settings: Option<&Bound<'_, PyDict>>,
           background: Option<&Bound<'_, PyDict>>, foliage: Option<&Bound<'_, PyDict>>, ref_epoch: u32) -> PyResult<Self> {
        let (em, bg) = (self::settings(settings)?, self::background(background)?);
        let el = elements_from_py(elements, &em)?;
        let spec = foliage.map(|d| -> PyResult<FoliageBox> {
            let req = |k: &str| d.get_item(k)?.ok_or_else(|| bad(k));
            Ok(FoliageBox { lo: req("lo")?.extract()?, hi: req("hi")?.extract()?, voxel: req("voxel")?.extract()? })
        }).transpose()?;
        let vs: Vec<Arc<RangeImage>> = images.iter().map(|v| v.inner.clone()).collect();
        Ok(PyScene { inner: py.detach(|| Scene::new(vs, &el, em, bg, spec, ref_epoch)), em })
    }

    /// Re-solve the foliage field from what the elements leave of the rays (one step of EM): the cost change.
    fn refit_foliage(&mut self, py: Python<'_>) -> f64 {
        let c = &mut self.inner;
        py.detach(|| c.refit_foliage())
    }

    /// Hold `other`'s foliage field fixed (estimated from its scans, e.g. the training scans): the cost change.
    fn hold_foliage(&mut self, py: Python<'_>, other: PyRef<'_, PyScene>) -> f64 {
        let ext = other.inner.foliage_field().clone();
        let c = &mut self.inner;
        py.detach(|| c.hold_foliage(&ext))
    }

    /// Add one element and take the mass it explains out of the foliage field: its id and the cost change.
    fn promote(&mut self, py: Python<'_>, element: &Bound<'_, PyDict>) -> PyResult<(u32, f64)> {
        let e = self.one(element)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.promote(e)))
    }

    /// Remove an element and give its mass back to the foliage field: the cost change.
    fn demote(&mut self, py: Python<'_>, id: u32) -> PyResult<f64> {
        self.inner.check_id(id).map_err(scene_err)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.demote(id)))
    }

    /// The foliage field per voxel with counts: voxel (n, 3, its index in the box), centre (n, 3), sigma (1/m, as
    /// held), the weighted counts it is fitted from as the scene now is, hits (less what the elements take of them) and
    /// path (m). Empty when the field is held from elsewhere.
    fn foliage_voxels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (spec, rows) = self.inner.foliage();
        let d = PyDict::new(py);
        let n = rows.len();
        d.set_item("voxel", rows.iter().flat_map(|r| r.0).collect::<Vec<i32>>().into_pyarray(py).reshape([n, 3])?)?;
        if let Some(sp) = spec {
            d.set_item("centre", rows.iter().flat_map(|r| [0, 1, 2].map(|a| sp.lo[a] + (r.0[a] as f64 + 0.5) * sp.voxel)).collect::<Vec<f64>>()
                .into_pyarray(py).reshape([n, 3])?)?;
        }
        d.set_item("sigma", rows.iter().map(|r| r.1).collect::<Vec<f64>>().into_pyarray(py))?;
        d.set_item("hits", rows.iter().map(|r| r.2 .0).collect::<Vec<f64>>().into_pyarray(py))?;
        d.set_item("path", rows.iter().map(|r| r.2 .1).collect::<Vec<f64>>().into_pyarray(py))?;
        Ok(d)
    }

    /// For flat `cells` of another image: the foliage field's optical depth across its box along each cell's ray
    /// (tau), and the ray's entry and exit ranges (t_in, t_out); NaN where the ray misses the box.
    fn transmission<'py>(&self, py: Python<'py>, image: PyRef<'py, PyRangeImage>, cells: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyDict>> {
        let v = image.inner.clone();
        let n = (v.n_lines * v.n_cols) as i64;
        if let Some(k) = cells.as_array().iter().find(|&&k| k < 0 || k >= n) {
            return Err(PyValueError::new_err(format!("no cell {k}")));
        }
        let cs: Vec<usize> = cells.as_array().iter().map(|&k| k as usize).collect();
        let c = &self.inner;
        let t = py.detach(|| c.transmission(&v, &cs));
        let d = PyDict::new(py);
        d.set_item("tau", t.iter().map(|x| x.0).collect::<Vec<f64>>().into_pyarray(py))?;
        d.set_item("t_in", t.iter().map(|x| x.1).collect::<Vec<f64>>().into_pyarray(py))?;
        d.set_item("t_out", t.iter().map(|x| x.2).collect::<Vec<f64>>().into_pyarray(py))?;
        Ok(d)
    }

    /// The scene's cost: −log of its likelihood ratio against the empty scene, the alignment held.
    fn cost(&self) -> f64 {
        self.inner.cost()
    }

    /// The scene's exact cost: the alignment re-solved for the elements as they are (the referee for `cost`; the scene
    /// unchanged).
    fn exact_cost(&self, py: Python<'_>) -> f64 {
        let c = &self.inner;
        py.detach(|| c.exact_cost())
    }

    /// Re-solve the scan alignment error for the elements as they are: the cost change (after it, `cost` is
    /// `exact_cost`).
    fn refit_alignment(&mut self, py: Python<'_>) -> f64 {
        let c = &mut self.inner;
        py.detach(|| c.refit_alignment())
    }

    /// Per (term, scan) (scene::ElementScan): elem (a group's first member), group (-1 alone), scan, cells, cells_on, arc (rad), footprints_across, surface; its cost
    /// and the parts it sums (ray_on, ray_empty, ray_in_front, ray_behind, and alignment: its own nodes' prior and
    /// log-determinant; each scan's alignment part is `alignment_parts()`, the field's `field_part()`); beside it, not
    /// in it, field: the foliage field's part of its cost to remove in this scan; the evidence discount (discount); the
    /// posterior along the beam at it, the alignment held: align_shift, align_sd, front_shift, front_shift_sd (m); and
    /// its reflectance level's: refl_level, refl_level_sd (dB).
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        rows_dict(py, &self.inner.rows())
    }

    /// Each scan's alignment part (its cost less its rows'), so the rows, these and `field_part()` sum to `cost()`.
    fn alignment_parts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.alignment_parts().into_pyarray(py)
    }

    /// The foliage field's part of the cost: its marginal against the empty scene's, less the plug-in the cells'
    /// ratios already read (0 without a field, or with one held from elsewhere).
    fn field_part(&self) -> f64 {
        self.inner.field_part()
    }

    /// Add elements together (a chain's cells read once, scene::Scene::add_many): their ids and the cost change.
    fn add(&mut self, py: Python<'_>, elements: &Bound<'_, PyDict>) -> PyResult<(Vec<u32>, f64)> {
        let el = elements_from_py(elements, &self.em)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.add_many(&el)))
    }

    /// Remove elements by id: the cost change.
    fn remove(&mut self, py: Python<'_>, ids: Vec<u32>) -> PyResult<f64> {
        let mut seen = std::collections::HashSet::new();
        for &id in &ids {
            self.inner.check_id(id).map_err(scene_err)?;
            if !seen.insert(id) {
                return Err(PyValueError::new_err(format!("element {id} twice")));
            }
        }
        let c = &mut self.inner;
        Ok(py.detach(|| ids.into_iter().map(|id| c.remove(id)).sum()))
    }

    /// Replace an element: the cost change.
    fn change(&mut self, py: Python<'_>, id: u32, element: &Bound<'_, PyDict>) -> PyResult<f64> {
        self.inner.check_id(id).map_err(scene_err)?;
        let e = self.one(element)?;
        let c = &mut self.inner;
        Ok(py.detach(|| c.change(id, e)))
    }

    /// Many hypotheses' costs to add, each alone, per scan and then, in a scene with a foliage field, the field's part:
    /// `hypotheses` a list of element dicts (one or more elements each); returns (hypotheses, scans), or (hypotheses,
    /// scans + 1) with a field, the scene unchanged and its alignment held. A row sums to the move.
    fn costs_to_add<'py>(&self, py: Python<'py>, hypotheses: Vec<Bound<'py, PyDict>>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let hs: Vec<Vec<Element>> = hypotheses.iter().map(|d| elements_from_py(d, &self.em)).collect::<PyResult<_>>()?;
        let c = &self.inner;
        let out = py.detach(|| c.costs_to_add(&hs));
        let (n, m) = (out.len(), out.first().map_or(self.inner.n_scans(), Vec::len));
        Ok(out.into_iter().flatten().collect::<Vec<f64>>().into_pyarray(py).reshape([n, m])?)
    }

    /// Each seed stick (an element dict, one row a seed) grown into a chain by the likelihood (scene::chain: a beam of
    /// `beam` continuations, pieces 4 r long within `step` m, up to `max_length` m, inside the region: `inside` takes
    /// points (n, 3) and returns a bool per point), the scene unchanged. Per seed a dict: sticks (k, 7) in order along
    /// the chain; cost and scan (per scan, then with a field the field's part), its cost to add read exactly; kalman
    /// (the chain state's own cost); seed_cost; steps; coupling (the departures' ½ Σ z²); length (m).
    #[pyo3(signature = (seeds, inside, beam=3, max_length=1.5, step=(0.03, 0.10)))]
    fn grow<'py>(&self, py: Python<'py>, seeds: &Bound<'py, PyDict>, inside: Py<PyAny>, beam: usize, max_length: f64, step: (f64, f64))
        -> PyResult<Vec<Bound<'py, PyDict>>> {
        let es = elements_from_py(seeds, &self.em)?;
        let test = |ps: &[[f64; 3]]| -> Vec<bool> {
            Python::attach(|py| -> PyResult<Vec<bool>> {
                let a = ps.iter().flatten().copied().collect::<Vec<f64>>().into_pyarray(py).reshape([ps.len(), 3])?;
                let ok: PyReadonlyArray1<bool> = inside.call1(py, (a,))?.extract(py)?;
                Ok(ok.as_array().to_vec())
            }).unwrap_or_else(|e| panic!("grow's region test failed: {e}"))
        };
        let opts = Grow { beam, max_length, step, ..Grow::default() };
        let c = &self.inner;
        let out = py.detach(|| c.grow(&es, &test, &opts));
        out.iter().map(|g| {
            let d = PyDict::new(py);
            d.set_item("sticks", sticks_of(py, &g.sticks)?)?;
            d.set_item("scan", g.scan.clone().into_pyarray(py))?;
            for (k, v) in [("cost", g.cost), ("kalman", g.kalman), ("seed_cost", g.seed_cost), ("coupling", g.coupling), ("length", g.length)] {
                d.set_item(k, v)?;
            }
            d.set_item("steps", g.steps)?;
            Ok(d)
        }).collect()
    }

    /// Each element's cost to remove, per scan and then, with a field, the field's part (as `costs_to_add`): `ids`
    /// live element ids; returns (ids, scans) or (ids, scans + 1), the scene unchanged.
    fn costs_to_remove<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        for &id in &ids {
            self.inner.check_id(id).map_err(scene_err)?;
        }
        let c = &self.inner;
        let out = py.detach(|| c.costs_to_remove(&ids));
        let (n, m) = (out.len(), out.first().map_or(self.inner.n_scans(), Vec::len));
        Ok(out.into_iter().flatten().collect::<Vec<f64>>().into_pyarray(py).reshape([n, m])?)
    }

    /// Which hypotheses interact per scan (scene::Scene::interactions), each pair once: i, j (i < j), scan, cells
    /// (they front a cell, or cells one scene term owns, in common, or join one group) and nodes (they share an
    /// alignment node).
    fn interactions<'py>(&self, py: Python<'py>, hypotheses: Vec<Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
        let hs: Vec<Vec<Element>> = hypotheses.iter().map(|d| elements_from_py(d, &self.em)).collect::<PyResult<_>>()?;
        let c = &self.inner;
        let out = py.detach(|| c.interactions(&hs));
        let d = PyDict::new(py);
        d.set_item("i", out.iter().map(|x| x.0).collect::<Vec<u32>>().into_pyarray(py))?;
        d.set_item("j", out.iter().map(|x| x.1).collect::<Vec<u32>>().into_pyarray(py))?;
        d.set_item("scan", out.iter().map(|x| x.2).collect::<Vec<u32>>().into_pyarray(py))?;
        d.set_item("cells", out.iter().map(|x| x.3).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("nodes", out.iter().map(|x| x.4).collect::<Vec<bool>>().into_pyarray(py))?;
        Ok(d)
    }

    /// How well the scans pin elements `ids` down, one or several jointly (scene::Scene::information), from one forward
    /// pass per cell: per scan, the cost's `slope` (scans, n) and its expected (Fisher) information `expected` (scans,
    /// n, n) in their parameters, each element's in turn: a stick's ln r, its axis moved across itself (two
    /// directions, m) and tilted (rad); a patch's ln r and its centre moved (x, y, z, m). `expected` is positive
    /// semi-definite, the latents the cells share marginalised once, so summed over scans with `slope` it gives a
    /// Fisher-scoring step; `gradient_variance` (scans, n, n) is the slope's variance were the cells independent, for
    /// the sandwich `E⁻¹ J E⁻¹`.
    fn information<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyDict>> {
        for (k, &id) in ids.iter().enumerate() {
            self.inner.check_id(id).map_err(scene_err)?;
            if ids[..k].contains(&id) {
                return Err(PyValueError::new_err(format!("element {id} twice")));
            }
        }
        let c = &self.inner;
        let out = py.detach(|| c.information(&ids));
        let (s, n) = (out.len(), out.first().map_or(0, |m| m.slope.len()));
        let d = PyDict::new(py);
        d.set_item("slope", out.iter().flat_map(|x| x.slope.clone()).collect::<Vec<f64>>().into_pyarray(py).reshape([s, n])?)?;
        let square = |f: fn(&Information) -> &Vec<f64>| out.iter().flat_map(|x| f(x).clone()).collect::<Vec<f64>>().into_pyarray(py).reshape([s, n, n]);
        d.set_item("expected", square(|x| &x.expected)?)?;
        d.set_item("gradient_variance", square(|x| &x.gradient_variance)?)?;
        Ok(d)
    }

    /// What hypothesis `q` is expected to cost against hypothesis `p` were `p` true, per scan (scene::Scene::divergence;
    /// nats): `p` and `q` element dicts; returns (scans,).
    fn divergence<'py>(&self, py: Python<'py>, p: &Bound<'py, PyDict>, q: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (p, q) = (elements_from_py(p, &self.em)?, elements_from_py(q, &self.em)?);
        let c = &self.inner;
        Ok(py.detach(|| c.divergence(&p, &q)).into_pyarray(py))
    }

    /// One element's counts against the scene, per scan, in footprints (scene::ElementCounts): scan, mismatch_ss,
    /// mismatch_w, mismatch_cells, on, empty, behind; `shift` (m, per scan) comes off the observed echoes.
    #[pyo3(signature = (element, shift=None))]
    fn counts<'py>(&self, py: Python<'py>, element: &Bound<'_, PyDict>, shift: Option<Vec<f64>>) -> PyResult<Bound<'py, PyDict>> {
        let e = self.one(element)?;
        let c = &self.inner;
        let st = py.detach(|| c.counts(e, &shift.unwrap_or_default()));
        let d = PyDict::new(py);
        d.set_item("scan", st.iter().map(|x| x.scan).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("mismatch_ss", st.iter().map(|x| x.mismatch_ss).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("mismatch_w", st.iter().map(|x| x.mismatch_w).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("mismatch_cells", st.iter().map(|x| x.mismatch_cells).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("on", st.iter().map(|x| x.on).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("empty", st.iter().map(|x| x.empty).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("behind", st.iter().map(|x| x.behind).collect::<Vec<_>>().into_pyarray(py))?;
        Ok(d)
    }

    /// What elements predict in scan `scan` against the scene, per cell one of them fronts (scene::CellPrediction): cell
    /// (flat index), elem (which of them, its row), surface, outcome (0 on, 1 empty, 2 in front, 3 behind), fill,
    /// recorded, arrives, depth, sd, refl (the echo's predicted reflectance, dB), cos (the mean cos ι where the element
    /// meets the footprint), obs, obs_refl (the first echo's reflectance, dB), read (any echo the likelihood reads is
    /// on it), merge (the foliage field's own merge share with the medium behind, before the site's map), weight (the
    /// cell weight), footprint (m), across and along (where the ray passes the axis); from the same pass over each
    /// pulse, returns (the expected number of echoes the receiver records from the cell's layers and ground), beam (the
    /// share of the beam the element leaves) and owned (how many of the pulse's echoes are the element's in
    /// expectation).
    #[pyo3(signature = (elements, scan=0))]
    fn predict<'py>(&self, py: Python<'py>, elements: &Bound<'_, PyDict>, scan: usize) -> PyResult<Bound<'py, PyDict>> {
        self.inner.check_scan(scan).map_err(scene_err)?;
        let es = elements_from_py(elements, &self.em)?;
        let c = &self.inner;
        let p = py.detach(|| c.predict(&es, scan));
        predictions_dict(py, &p)
    }
}
