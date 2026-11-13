// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings 3dtrees' fork adds (CHANGELOG-3dtrees.md): the scan index, the
//! joint table and `support_tree`, the trees' context and contest, the ray-cloud
//! tiles, and the beam scan with solids. Kept apart from upstream's binding
//! files, so a merge from upstream meets them in one line of `lib.rs`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::io;
use sylva_rs::qsm::{self, support_tree};
use sylva_rs::qsm::scan_index::{self, ScanIndex};
use sylva_rs::qsm::joint::{self, Echoes, JointParams, Pulse};
use sylva_rs::qsm::joint_fit::{PriorParams, Rows};
use sylva_rs::synthetic::scan::{self as synth, BeamScan, Reflectance, Solid};

use crate::{cloud_from_py, err, xyz_from_py};

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

/// The default settings of `qsm_support_tree`, flat: the likelihood's fields, then the prior's.
#[pyfunction]
fn qsm_joint_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let (like, prior) = (JointParams::default(), PriorParams::default());
    let d = PyDict::new(py);
    for &k in JointParams::FIELDS {
        d.set_item(k, like.get(k))?;
    }
    for &k in PriorParams::FIELDS {
        d.set_item(k, prior.get(k))?;
    }
    Ok(d)
}

/// The settings `qsm_joint_defaults` lays out, each field given once (`dataclasses.asdict(JointParams)`).
fn joint_params(d: &Bound<'_, PyDict>) -> PyResult<(JointParams, PriorParams)> {
    let (mut like, mut prior) = (JointParams::default(), PriorParams::default());
    for (k, v) in d.iter() {
        let (k, v): (String, f64) = (k.extract()?, v.extract()?);
        if !(like.set(&k, v) || prior.set(&k, v)) {
            return Err(PyValueError::new_err(format!("joint params: {k:?} is not a field")));
        }
    }
    if d.len() != JointParams::FIELDS.len() + PriorParams::FIELDS.len() {
        return Err(PyValueError::new_err("joint params: every field once"));
    }
    Ok((like, prior))
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
/// Upstream's `synthetic_scan_beam`, its arguments in its order, with solid capped cylinders `a -> b` of radius `r`
/// in the scene, each echoing as the cloud's point `point` (see `sylva.synthetic.scan`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn synthetic_scan_beam_solids<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, origin: (f64, f64, f64), resolution_deg: f64, min_zenith_deg: f64, max_zenith_deg: f64, max_echoes: usize, echo_separation: f64, range_noise: f64, range_noise_slope: f64, beam_divergence: f64, exit_diameter: f64, footprint_samples: usize, mixed_pixels: bool, target_radius: Option<f64>, oriented: bool, detection_threshold: f64, reflectance_map: Vec<(i64, f64)>, reflectance_default: f64, reflectance_attr: Option<String>, reflectance_constant: Option<f64>, max_range: f64, seed: u64, solid_a: PyReadonlyArray2<f64>, solid_b: PyReadonlyArray2<f64>, solid_r: PyReadonlyArray1<f64>, solid_point: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyDict>> {
    let cloud = cloud_from_py(xyz, attrs)?;
    let (sa, sb, sr, sp) = (solid_a.as_array(), solid_b.as_array(), solid_r.as_array(), solid_point.as_array());
    if sa.ncols() != 3 || sb.ncols() != 3 || sb.nrows() != sa.nrows() || sr.len() != sa.nrows() || sp.len() != sa.nrows() || sp.iter().any(|&i| i < 0) {
        return Err(PyValueError::new_err("solids: a and b (n, 3), r and point (n,), point indices of the cloud"));
    }
    let p = BeamScan {
        solids: (0..sa.nrows()).map(|i| Solid { a: [sa[[i, 0]], sa[[i, 1]], sa[[i, 2]]], b: [sb[[i, 0]], sb[[i, 1]], sb[[i, 2]]], r: sr[i], point: sp[i] as usize }).collect(),
        resolution_deg,
        min_zenith_deg,
        max_zenith_deg,
        max_echoes,
        echo_separation,
        range_noise,
        range_noise_slope,
        divergence_mrad: beam_divergence,
        exit_diameter,
        footprint_samples,
        mixed_pixels,
        target_radius,
        oriented,
        detection_threshold,
        reflectance: match (reflectance_attr, reflectance_constant) {
            (Some(a), _) => Reflectance::Attribute(a),
            (None, Some(c)) => Reflectance::Constant(c),
            (None, None) => Reflectance::ByClass(reflectance_map, reflectance_default),
        },
        max_range,
        seed,
        ..BeamScan::new([origin.0, origin.1, origin.2])
    };
    let s = py.detach(|| synth::scan_beam(&cloud, &p)).map_err(err)?;
    crate::shots_into_py(py, s)
}

/// Adds this file's functions to the `_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScanIndex>()?;
    for f in [
        wrap_pyfunction!(qsm_support_flags, m)?,
        wrap_pyfunction!(qsm_support_diags, m)?,
        wrap_pyfunction!(qsm_support_bases, m)?,
        wrap_pyfunction!(qsm_support_tree, m)?,
        wrap_pyfunction!(qsm_tree_context, m)?,
        wrap_pyfunction!(qsm_joint_defaults, m)?,
        wrap_pyfunction!(qsm_joint_table, m)?,
        wrap_pyfunction!(qsm_pairs, m)?,
        wrap_pyfunction!(qsm_contest, m)?,
        wrap_pyfunction!(tile_ray_clouds, m)?,
        wrap_pyfunction!(synthetic_scan_beam_solids, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
