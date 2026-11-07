// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva.qsm: the cylinder table's totals and cuts, the CSV
//! reader, the mesh writers, buttress meshing and fusion, and the plot loop
//! with its table and files. Models cross as `(n, 12)` float arrays; a plot
//! crosses as a list of `(tree_id, cylinders, points, height, buttress)`
//! tuples, the buttress `(vertices, faces, volume, top, top_z)` or None.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::qsm::buttress::{Buttress, ButtressParams};
use sylva_rs::qsm::evidence::{self, Beam, EvidenceParams, ScanIndex};
use sylva_rs::qsm::joint::{self, JointParams, Owner, Pulse};
use sylva_rs::qsm::joint_fit::{self, FitOptions, PriorParams, Rows};
use sylva_rs::qsm::joint_tree::{self, TreeParams};
use sylva_rs::qsm::QsmParams;
use sylva_rs::qsm::ops::{self as ops, Row, Segment};
use sylva_rs::qsm::plot::{self as plot, ButtressView, PlotEntry, PlotParams};
use sylva_rs::io::mesh as mesh_io;
use sylva_rs::Point;

use crate::{err, xyz_from_py, xyz_to_py};

type Matrix<'py> = Bound<'py, PyArray2<f64>>;

fn rows_from_py(cylinders: PyReadonlyArray2<f64>) -> PyResult<Vec<Row>> {
    let a = cylinders.as_array();
    if a.ncols() != 12 {
        return Err(PyValueError::new_err("cylinder array must have 12 columns"));
    }
    Ok(a.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect())
}

fn rows_to_py<'py>(py: Python<'py>, rows: &[Row]) -> PyResult<Matrix<'py>> {
    PyArray1::from_vec(py, rows.iter().flatten().copied().collect::<Vec<f64>>()).reshape([rows.len(), 12])
}

fn faces_from_py(f: PyReadonlyArray2<u32>) -> PyResult<Vec<[u32; 3]>> {
    let a = f.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err("faces must have three columns"));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn faces_to_py<'py>(py: Python<'py>, f: &[[u32; 3]]) -> PyResult<Bound<'py, PyArray2<u32>>> {
    PyArray1::from_vec(py, f.iter().flatten().copied().collect::<Vec<u32>>()).reshape([f.len(), 3])
}

fn segments_from_py(s: PyReadonlyArray3<f64>) -> PyResult<Vec<Segment>> {
    let a = s.as_array();
    if a.shape()[1] != 2 || a.shape()[2] != 2 {
        return Err(PyValueError::new_err("a section must have shape (n, 2, 2)"));
    }
    Ok((0..a.shape()[0]).map(|i| [[a[[i, 0, 0]], a[[i, 0, 1]]], [a[[i, 1, 0]], a[[i, 1, 1]]]]).collect())
}

fn segments_to_py<'py>(py: Python<'py>, s: &[Segment]) -> PyResult<Bound<'py, PyArray3<f64>>> {
    PyArray1::from_vec(py, s.iter().flatten().flatten().copied().collect::<Vec<f64>>()).reshape([s.len(), 2, 2])
}

fn io_err(path: &std::path::Path, e: sylva_rs::Error) -> PyErr {
    match e {
        sylva_rs::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(format!("{} not found.", path.display())),
        e => err(e),
    }
}

// ------------------------------------------------------------ the table

/// Volumes, lengths and the highest branch order of a model.
#[pyfunction]
fn qsm_totals<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let t = ops::totals(&rows_from_py(cylinders)?);
    let d = PyDict::new(py);
    d.set_item("total_volume", t.total_volume)?;
    d.set_item("stem_volume", t.stem_volume)?;
    d.set_item("branch_volume", t.branch_volume)?;
    d.set_item("total_length", t.total_length)?;
    d.set_item("max_branch_order", t.max_branch_order)?;
    Ok(d)
}

#[pyfunction]
fn qsm_ends<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Matrix<'py>> {
    Ok(xyz_to_py(py, &ops::ends(&rows_from_py(cylinders)?)))
}

#[pyfunction]
fn qsm_volumes<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(ops::volumes(&rows_from_py(cylinders)?).into_pyarray(py))
}

#[pyfunction]
fn qsm_volume_above(cylinders: PyReadonlyArray2<f64>, z: f64) -> PyResult<f64> {
    Ok(ops::volume_above(&rows_from_py(cylinders)?, z))
}

#[pyfunction]
fn qsm_above<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>, z: f64) -> PyResult<Matrix<'py>> {
    rows_to_py(py, &ops::above(&rows_from_py(cylinders)?, z))
}

#[pyfunction]
fn qsm_read_csv<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Matrix<'py>> {
    let rows = ops::read_csv(&path).map_err(|e| io_err(&path, e))?;
    rows_to_py(py, &rows)
}

// ------------------------------------------------------------ meshes

#[pyfunction]
#[pyo3(signature = (path, cylinders, sides=12, contiguous=false))]
fn qsm_write_obj(path: PathBuf, cylinders: PyReadonlyArray2<f64>, sides: usize, contiguous: bool) -> PyResult<()> {
    ops::write_model_obj(path, &rows_from_py(cylinders)?, sides, contiguous).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, cylinders, sides=12, color=None, contiguous=false))]
fn qsm_write_ply(path: PathBuf, cylinders: PyReadonlyArray2<f64>, sides: usize, color: Option<[u8; 3]>, contiguous: bool) -> PyResult<()> {
    ops::write_model_ply(path, &rows_from_py(cylinders)?, sides, color, contiguous).map_err(err)
}

/// A binary PLY triangle mesh; `faces` are int32 and `face_rgb` one colour per face.
#[pyfunction]
#[pyo3(signature = (path, vertices, faces, face_rgb=None))]
fn write_ply_mesh(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<i32>, face_rgb: Option<PyReadonlyArray2<u8>>) -> PyResult<()> {
    let v = xyz_from_py(vertices)?;
    let f = faces.as_array();
    if f.ncols() != 3 {
        return Err(PyValueError::new_err("faces must have three columns"));
    }
    let f: Vec<[i32; 3]> = f.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let rgb = match face_rgb {
        Some(c) => {
            let c = c.as_array();
            if c.ncols() != 3 {
                return Err(PyValueError::new_err("face_rgb must have three columns"));
            }
            Some(c.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect::<Vec<[u8; 3]>>())
        }
        None => None,
    };
    mesh_io::write_ply(path, &v, &f, rgb.as_deref()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, cx, cy, ground_z=None, resolution=0.02, slice=0.05, close_radius=0.08, max_radius=4.0, max_height=6.0, top=None, solidity=0.9, round_run=4, min_top=0.5, min_points=30, max_flare=1.0, smooth=10))]
#[allow(clippy::too_many_arguments)]
fn qsm_buttress_mesh<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, cx: f64, cy: f64, ground_z: Option<f64>, resolution: f64, slice: f64, close_radius: f64, max_radius: f64, max_height: f64, top: Option<f64>, solidity: f64, round_run: usize, min_top: f64, min_points: usize, max_flare: f64, smooth: usize) -> PyResult<Bound<'py, PyDict>> {
    let pts = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let p = ButtressParams { resolution, slice, close_radius, max_radius, max_height, top, solidity, round_run, min_top, min_points, max_flare, smooth };
    let b = py.detach(|| plot::buttress_mesh(&pts, &h, cx, cy, ground_z, &p)).map_err(err)?;
    buttress_to_py(py, &b)
}

fn buttress_to_py<'py>(py: Python<'py>, b: &Buttress) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("vertices", xyz_to_py(py, &b.vertices))?;
    let f: Vec<i64> = b.faces.iter().flatten().map(|&x| x as i64).collect();
    d.set_item("faces", PyArray1::from_vec(py, f).reshape([b.faces.len(), 3])?)?;
    d.set_item("volume", b.volume)?;
    d.set_item("top", b.top)?;
    d.set_item("top_z", b.top_z)?;
    d.set_item("heights", b.heights.clone().into_pyarray(py))?;
    d.set_item("areas", b.areas.clone().into_pyarray(py))?;
    d.set_item("solidities", b.solidities.clone().into_pyarray(py))?;
    d.set_item("open", b.open.clone().into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn qsm_section<'py>(py: Python<'py>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, z: f64) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let s = ops::section(&xyz_from_py(vertices)?, &faces_from_py(faces)?, z).map_err(err)?;
    segments_to_py(py, &s)
}

#[pyfunction]
fn qsm_inside<'py>(py: Python<'py>, section: PyReadonlyArray3<f64>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let s = segments_from_py(section)?;
    let p = points.as_array();
    if p.ncols() < 2 {
        return Err(PyValueError::new_err("points must be (n, 2)"));
    }
    let p: Vec<[f64; 2]> = p.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    Ok(ops::inside(&s, &p).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (base, wood, cell=0.02))]
fn qsm_join_fit(base: PyReadonlyArray3<f64>, wood: PyReadonlyArray3<f64>, cell: f64) -> PyResult<(f64, f64)> {
    Ok(ops::join_fit(&segments_from_py(base)?, &segments_from_py(wood)?, cell))
}

#[pyfunction]
#[pyo3(signature = (vertices, faces, volume, top_z, cylinders, sides=12, contiguous=true, overlap=0.1))]
#[allow(clippy::too_many_arguments)]
fn qsm_fuse<'py>(py: Python<'py>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, volume: f64, top_z: f64, cylinders: PyReadonlyArray2<f64>, sides: usize, contiguous: bool, overlap: f64) -> PyResult<Bound<'py, PyDict>> {
    let (v, f, rows) = (xyz_from_py(vertices)?, faces_from_py(faces)?, rows_from_py(cylinders)?);
    let t = py.detach(|| ops::fuse(&v, &f, volume, top_z, &rows, sides, contiguous, overlap)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("vertices", xyz_to_py(py, &t.vertices))?;
    d.set_item("faces", faces_to_py(py, &t.faces)?)?;
    d.set_item("part", t.part.into_pyarray(py))?;
    d.set_item("buttress_volume", t.buttress_volume)?;
    d.set_item("wood_volume", t.wood_volume)?;
    d.set_item("top_z", t.top_z)?;
    d.set_item("offset", t.offset)?;
    d.set_item("overhang", t.overhang)?;
    Ok(d)
}

#[pyfunction]
fn tree_mesh_write_obj(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, part: PyReadonlyArray1<u8>) -> PyResult<()> {
    ops::write_tree_mesh_obj(path, &xyz_from_py(vertices)?, &faces_from_py(faces)?, &part.as_array().to_vec()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, vertices, faces, part, color=None))]
fn tree_mesh_write_ply(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, part: PyReadonlyArray1<u8>, color: Option<[u8; 3]>) -> PyResult<()> {
    ops::write_tree_mesh_ply(path, &xyz_from_py(vertices)?, &faces_from_py(faces)?, &part.as_array().to_vec(), color).map_err(err)
}

// ------------------------------------------------------------ plots

/// QSM settings from a dict holding every field of `QsmParams`.
fn qsm_params_from_py(d: &Bound<'_, PyDict>) -> PyResult<QsmParams> {
    let mut p = QsmParams::default();
    macro_rules! take {
        ($($f:ident),*) => {
            $(p.$f = d.get_item(stringify!($f))?.ok_or_else(|| PyValueError::new_err(concat!("missing QSM setting ", stringify!($f))))?.extract()?;)*
        };
    }
    take!(k, max_edge, bin_length, min_points, ransac_threshold, max_radius, taper_limit, max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction, prune_points, fit_min_points, crop_length, butt_height, relative_tolerance, base_radius, allometry_tolerance, stem_radius_cap, buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack, branch_min_inlier_fraction, spacing_scale, radius_power, power_above_spacing, sensor_noise, cluster_eps, centre_fit_points, radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius, min_weight, min_mean_weight);
    Ok(p)
}

#[pyfunction]
#[pyo3(signature = (xyz, labels, heights, stem_ids, stem_xy, stem_dbh, voxel_size, wood, buttress, min_points, params, wood_labels=None, wood_weights=None))]
#[allow(clippy::too_many_arguments)]
fn qsm_build_plot<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>, heights: Option<PyReadonlyArray1<f64>>, stem_ids: Vec<i64>, stem_xy: Vec<(f64, f64)>, stem_dbh: Vec<f64>, voxel_size: f64, wood: bool, buttress: bool, min_points: f64, params: &Bound<'_, PyDict>, wood_labels: Option<PyReadonlyArray1<bool>>, wood_weights: Option<PyReadonlyArray1<f64>>) -> PyResult<Bound<'py, PyDict>> {
    let pts = xyz_from_py(xyz)?;
    let labels = labels.as_array().to_vec();
    let h = heights.map(|h| h.as_array().to_vec());
    if stem_ids.len() != stem_xy.len() || stem_ids.len() != stem_dbh.len() {
        return Err(PyValueError::new_err("one stem position and DBH per stem id"));
    }
    let stems: Vec<(i64, [f64; 2], f64)> = stem_ids.into_iter().zip(stem_xy).zip(stem_dbh).map(|((t, (x, y)), d)| (t, [x, y], d)).collect();
    let p = PlotParams { voxel_size, wood, buttress, min_points, qsm: qsm_params_from_py(params)? };
    let wood_labels = wood_labels.map(|w| w.as_array().to_vec());
    let wood_weights = wood_weights.map(|w| w.as_array().to_vec());
    let given = match (&wood_labels, &wood_weights) {
        (Some(w), _) => Some(plot::PointWood::Labels(w)),
        (None, Some(w)) => Some(plot::PointWood::Weights(w)),
        (None, None) => None,
    };
    let r = py.detach(|| plot::build_plot_with(&pts, &labels, h.as_deref(), &stems, &p, given)).map_err(err)?;
    let share = py.detach(|| plot::median_measured_length(r.models.iter().map(|m| m.1.as_slice())));
    let d = PyDict::new(py);
    let models = PyList::empty(py);
    for (t, rows) in &r.models {
        models.append((*t, rows_to_py(py, rows)?))?;
    }
    let bases = PyList::empty(py);
    for (t, b) in &r.buttresses {
        bases.append((*t, buttress_to_py(py, b)?))?;
    }
    d.set_item("models", models)?;
    d.set_item("buttresses", bases)?;
    d.set_item("skipped", r.skipped)?;
    d.set_item("points", r.points)?;
    d.set_item("heights", r.heights)?;
    d.set_item("median_measured_length", share)?;
    Ok(d)
}

type ButtressArgs<'py> = (PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, u32>, f64, f64, f64);
type EntryArgs<'py> = (i64, PyReadonlyArray2<'py, f64>, Option<i64>, Option<f64>, Option<ButtressArgs<'py>>);

type OwnedButtress = (Vec<Point>, Vec<[u32; 3]>, f64, f64, f64);

struct Owned {
    tree_id: i64,
    rows: Vec<Row>,
    points: Option<i64>,
    height: Option<f64>,
    buttress: Option<OwnedButtress>,
}

fn entries_from_py(entries: Vec<EntryArgs<'_>>) -> PyResult<Vec<Owned>> {
    entries
        .into_iter()
        .map(|(tree_id, cyl, points, height, b)| {
            let buttress = match b {
                Some((v, f, volume, top, top_z)) => Some((xyz_from_py(v)?, faces_from_py(f)?, volume, top, top_z)),
                None => None,
            };
            Ok(Owned { tree_id, rows: rows_from_py(cyl)?, points, height, buttress })
        })
        .collect()
}

fn views(owned: &[Owned]) -> Vec<PlotEntry<'_>> {
    owned
        .iter()
        .map(|o| PlotEntry {
            tree_id: o.tree_id,
            rows: &o.rows,
            points: o.points,
            height: o.height,
            buttress: o.buttress.as_ref().map(|(v, f, volume, top, top_z)| ButtressView { vertices: v, faces: f, volume: *volume, top: *top, top_z: *top_z }),
        })
        .collect()
}

#[pyfunction]
fn plot_total_volume(entries: Vec<EntryArgs<'_>>) -> PyResult<f64> {
    Ok(plot::total_volume(&views(&entries_from_py(entries)?)))
}

/// The plot table as `(tree_id, points, volume_m3, dbh_m, height_m,
/// n_cylinders, measured_volume, measured_length, buttress_m3,
/// buttress_top_m)` tuples, None where a value is missing.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn plot_table(py: Python<'_>, entries: Vec<EntryArgs<'_>>) -> PyResult<Vec<(i64, Option<i64>, f64, f64, f64, usize, f64, f64, Option<f64>, Option<f64>)>> {
    let owned = entries_from_py(entries)?;
    let rows = py.detach(|| plot::table(&views(&owned)));
    Ok(rows.into_iter().map(|r| (r.tree_id, r.points, r.volume_m3, r.dbh_m, r.height_m, r.n_cylinders, r.measured_volume, r.measured_length, r.buttress_m3, r.buttress_top_m)).collect())
}

#[pyfunction]
fn plot_write_csv(py: Python<'_>, path: PathBuf, entries: Vec<EntryArgs<'_>>) -> PyResult<()> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_table_csv(&path, &views(&owned))).map_err(err)
}

#[pyfunction]
fn plot_write_meshes(py: Python<'_>, directory: PathBuf, entries: Vec<EntryArgs<'_>>, fmt: &str, sides: usize, contiguous: bool, prefix: &str) -> PyResult<Vec<PathBuf>> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_meshes(&directory, &views(&owned), fmt, sides, contiguous, prefix)).map_err(err)
}

#[pyfunction]
fn plot_write_cylinders(py: Python<'_>, directory: PathBuf, entries: Vec<EntryArgs<'_>>, prefix: &str) -> PyResult<()> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_cylinders(&directory, &views(&owned), prefix)).map_err(err)
}

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
    if d.len() != 17 {
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
    let p = py.detach(|| evidence::pairs(&refs, &a, &b, &window, threads)).map_err(err)?;
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

fn joint_params(params: &Bound<'_, PyDict>) -> PyResult<JointParams> {
    let get = |k: &str| -> PyResult<f64> { params.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("joint params: no {k:?}")))?.extract() };
    Ok(JointParams { exit: get("exit")?, divergence: get("divergence")?, sigma_r: get("sigma_r")?, rough: get("rough")?, tau: get("tau")?, edge: get("edge")?, eps: get("eps")?, eps_min: get("eps_min")?, mp_len: get("mp_len")?, hide: get("hide")? })
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
    let (pulses, state, p) = (pulses_from(&table)?, rows::<4>(state, "state")?, joint_params(&params)?);
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

/// Fits the joint model (see `sylva.qsm.joint_fit`).
#[pyfunction]
#[pyo3(signature = (table, a, b, r_prior, r_max, parent, n_scans, params, schedule, iters, tol, theta0=None, threads=0))]
#[allow(clippy::too_many_arguments)]
fn qsm_joint_fit<'py>(py: Python<'py>, table: Bound<'py, PyDict>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, r_prior: PyReadonlyArray1<f64>, r_max: PyReadonlyArray1<f64>, parent: PyReadonlyArray1<i64>, n_scans: usize, params: Bound<'py, PyDict>, schedule: Vec<f64>, iters: usize, tol: f64, theta0: Option<PyReadonlyArray2<f64>>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let pulses = pulses_from(&table)?;
    let row = dict_item(&table, "row", "joint table")?.extract::<PyReadonlyArray1<u32>>()?.as_array().to_vec();
    let scan = dict_item(&table, "scan", "joint table")?.extract::<PyReadonlyArray1<u32>>()?.as_array().to_vec();
    let (a, b) = (xyz_from_py(a)?, xyz_from_py(b)?);
    let (r_prior, r_max, parent) = (r_prior.as_array().to_vec(), r_max.as_array().to_vec(), parent.as_array().to_vec());
    let like = joint_params(&params)?;
    let get = |k: &str| -> PyResult<f64> { dict_item(&params, k, "joint params")?.extract() };
    let prior = PriorParams { lam_prior: get("lam_prior")?, lam_sd: get("lam_sd")?, sig_ell: get("sig_ell")?, sig_eta: get("sig_eta")?, mu_fork: get("mu_fork")?, sig_fork: get("sig_fork")?, sig_root: get("sig_root")?, r_tip: get("r_tip")?, sig_tip: get("sig_tip")?, sig_delta: get("sig_delta")?, centre_rel: get("centre_rel")?, centre_abs: get("centre_abs")?, scan_abs: get("scan_abs")?, scan_rel: get("scan_rel")? };
    let theta0 = theta0.map(|t| rows::<4>(t, "theta0")).transpose()?;
    let opts = FitOptions { schedule, iters, tol, threads };
    let fit = py.detach(|| {
        let rows = Rows { a: &a, b: &b, r_prior: &r_prior, r_max: &r_max, parent: &parent };
        joint_fit::joint_fit(&pulses, &row, &scan, &rows, n_scans, &like, &prior, &opts, theta0.as_deref())
    }).map_err(err)?;
    let n = fit.theta.len();
    let d = PyDict::new(py);
    d.set_item("theta", fit.theta.concat().into_pyarray(py).reshape([n, 4])?)?;
    d.set_item("nu", fit.nu.concat().into_pyarray(py).reshape([fit.nu.len(), 2])?)?;
    d.set_item("sd", fit.sd.concat().into_pyarray(py).reshape([n, 4])?)?;
    d.set_item("rho_local", fit.rho_local.into_pyarray(py))?;
    d.set_item("rho_pooled", fit.rho_pooled.into_pyarray(py))?;
    d.set_item("grad", fit.grad.concat().into_pyarray(py).reshape([n, 4])?)?;
    d.set_item("hess", fit.hess.iter().flatten().flatten().copied().collect::<Vec<f64>>().into_pyarray(py).reshape([n, 4, 4])?)?;
    d.set_item("profile", fit.profile.concat().into_pyarray(py).reshape([n, 21])?)?;
    d.set_item("profile_grid", joint_fit::profile_grid().to_vec().into_pyarray(py))?;
    d.set_item("objective", fit.objective)?;
    d.set_item("iterations", fit.iterations)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("exact", fit.exact)?;
    Ok(d)
}

const JOINT_LIKE: [&str; 10] = ["exit", "divergence", "sigma_r", "rough", "tau", "edge", "eps", "eps_min", "mp_len", "hide"];
const JOINT_PRIOR: [&str; 14] = ["lam_prior", "lam_sd", "sig_ell", "sig_eta", "mu_fork", "sig_fork", "sig_root", "r_tip", "sig_tip", "sig_delta", "centre_rel", "centre_abs", "scan_abs", "scan_rel"];
const JOINT_TREE: [&str; 9] = ["version", "iters", "tol", "sweeps", "split_share", "split_min_r", "split_through", "split_min_votes", "drop"];

/// The default settings of `qsm_joint_tree`, flat.
#[pyfunction]
fn qsm_joint_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let t = TreeParams::default();
    let (l, p) = (t.like, t.prior);
    let d = PyDict::new(py);
    for (k, v) in JOINT_LIKE.iter().zip([l.exit, l.divergence, l.sigma_r, l.rough, l.tau, l.edge, l.eps, l.eps_min, l.mp_len, l.hide]) {
        d.set_item(k, v)?;
    }
    for (k, v) in JOINT_PRIOR.iter().zip([p.lam_prior, p.lam_sd, p.sig_ell, p.sig_eta, p.mu_fork, p.sig_fork, p.sig_root, p.r_tip, p.sig_tip, p.sig_delta, p.centre_rel, p.centre_abs, p.scan_abs, p.scan_rel]) {
        d.set_item(k, v)?;
    }
    d.set_item("version", TreeParams::VERSION)?;
    d.set_item("iters", t.iters)?;
    d.set_item("tol", t.tol)?;
    d.set_item("sweeps", t.sweeps)?;
    d.set_item("split_share", t.split_share)?;
    d.set_item("split_min_r", t.split_min_r)?;
    d.set_item("split_through", t.split_through)?;
    d.set_item("split_min_votes", t.split_min_votes)?;
    d.set_item("drop", t.drop)?;
    Ok(d)
}

/// The names of the `reason` codes of `qsm_joint_tree`.
#[pyfunction]
fn qsm_joint_reasons() -> Vec<(&'static str, u8)> {
    use joint_tree::reason::*;
    vec![("MEASURED", MEASURED), ("FLAT", FLAT), ("NO_WOOD", NO_WOOD)]
}

fn tree_params(d: &Bound<'_, PyDict>) -> PyResult<TreeParams> {
    let what = "joint params";
    let f = |k: &str| -> PyResult<f64> { dict_item(d, k, what)?.extract() };
    let u = |k: &str| -> PyResult<usize> { dict_item(d, k, what)?.extract() };
    if u("version")? != TreeParams::VERSION as usize {
        return Err(PyValueError::new_err(format!("joint params: version {} is not this Sylva's ({})", u("version")?, TreeParams::VERSION)));
    }
    if d.len() != JOINT_LIKE.len() + JOINT_PRIOR.len() + JOINT_TREE.len() {
        return Err(PyValueError::new_err("joint params: unexpected fields"));
    }
    let prior = PriorParams { lam_prior: f("lam_prior")?, lam_sd: f("lam_sd")?, sig_ell: f("sig_ell")?, sig_eta: f("sig_eta")?, mu_fork: f("mu_fork")?, sig_fork: f("sig_fork")?, sig_root: f("sig_root")?, r_tip: f("r_tip")?, sig_tip: f("sig_tip")?, sig_delta: f("sig_delta")?, centre_rel: f("centre_rel")?, centre_abs: f("centre_abs")?, scan_abs: f("scan_abs")?, scan_rel: f("scan_rel")? };
    Ok(TreeParams { like: joint_params(d)?, prior, iters: u("iters")?, tol: f("tol")?, sweeps: u("sweeps")?, split_share: f("split_share")?, split_min_r: f("split_min_r")?, split_through: f("split_through")?, split_min_votes: u("split_min_votes")?, drop: f("drop")? })
}

/// Fits the joint model to one tree (see `sylva.qsm.joint_tree`).
#[pyfunction]
#[pyo3(signature = (scans, a, b, r_prior, parent, ctx_a, ctx_b, ctx_r, params, split=None, threads=0))]
#[allow(clippy::too_many_arguments)]
fn qsm_joint_tree<'py>(py: Python<'py>, scans: Vec<Bound<'py, PyScanIndex>>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, r_prior: PyReadonlyArray1<f64>, parent: PyReadonlyArray1<i64>, ctx_a: PyReadonlyArray2<f64>, ctx_b: PyReadonlyArray2<f64>, ctx_r: PyReadonlyArray1<f64>, params: Bound<'py, PyDict>, split: Option<PyReadonlyArray1<bool>>, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b, ctx_a, ctx_b) = (xyz_from_py(a)?, xyz_from_py(b)?, xyz_from_py(ctx_a)?, xyz_from_py(ctx_b)?);
    let (r_prior, parent, ctx_r) = (r_prior.as_array().to_vec(), parent.as_array().to_vec(), ctx_r.as_array().to_vec());
    let split = split.map(|s| s.as_array().to_vec());
    let p = tree_params(&params)?;
    let refs: Vec<&ScanIndex> = scans.iter().map(|s| &s.get().inner).collect();
    let f = py.detach(|| {
        let rows = Rows { a: &a, b: &b, r_prior: &r_prior, r_max: &[], parent: &parent };
        joint_tree::joint_tree(&refs, &rows, &ctx_a, &ctx_b, &ctx_r, &p, split.as_deref(), threads)
    }).map_err(err)?;
    let n = f.r.len();
    let d = PyDict::new(py);
    let xyz = |v: &[[f64; 3]]| v.concat().into_pyarray(py).reshape([n, 3]);
    d.set_item("src", f.src.into_pyarray(py))?;
    d.set_item("parent", f.parent.into_pyarray(py))?;
    d.set_item("a", xyz(&f.a)?)?;
    d.set_item("b", xyz(&f.b)?)?;
    d.set_item("e1", xyz(&f.e1)?)?;
    d.set_item("e2", xyz(&f.e2)?)?;
    for (k, v) in [("r", f.r), ("sd_log_r", f.sd_log_r), ("lo", f.lo), ("hi", f.hi), ("lam", f.lam), ("log_odds", f.log_odds), ("p_wood", f.p_wood), ("rho_local", f.rho_local), ("rho_pooled", f.rho_pooled)] {
        d.set_item(k, v.into_pyarray(py))?;
    }
    d.set_item("flat", f.flat.into_pyarray(py))?;
    d.set_item("reason", f.reason.into_pyarray(py))?;
    d.set_item("n_pairs", f.n_pairs.into_pyarray(py))?;
    d.set_item("grad", f.grad.concat().into_pyarray(py).reshape([n, 4])?)?;
    d.set_item("hess", f.hess.iter().flatten().flatten().copied().collect::<Vec<f64>>().into_pyarray(py).reshape([n, 4, 4])?)?;
    d.set_item("u", f.u)?;
    d.set_item("objective", f.objective)?;
    d.set_item("converged", f.converged)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qsm_joint_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_joint_reasons, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_joint_tree, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_joint_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_joint_table, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_joint_fit, m)?)?;
    m.add_class::<PyScanIndex>()?;
    m.add_function(wrap_pyfunction!(qsm_evidence, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_evidence_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(qsm_evidence_reasons, m)?)?;
    for f in [
        wrap_pyfunction!(qsm_totals, m)?,
        wrap_pyfunction!(qsm_ends, m)?,
        wrap_pyfunction!(qsm_volumes, m)?,
        wrap_pyfunction!(qsm_volume_above, m)?,
        wrap_pyfunction!(qsm_above, m)?,
        wrap_pyfunction!(qsm_read_csv, m)?,
        wrap_pyfunction!(qsm_write_obj, m)?,
        wrap_pyfunction!(qsm_write_ply, m)?,
        wrap_pyfunction!(write_ply_mesh, m)?,
        wrap_pyfunction!(qsm_buttress_mesh, m)?,
        wrap_pyfunction!(qsm_section, m)?,
        wrap_pyfunction!(qsm_inside, m)?,
        wrap_pyfunction!(qsm_join_fit, m)?,
        wrap_pyfunction!(qsm_fuse, m)?,
        wrap_pyfunction!(tree_mesh_write_obj, m)?,
        wrap_pyfunction!(tree_mesh_write_ply, m)?,
        wrap_pyfunction!(qsm_build_plot, m)?,
        wrap_pyfunction!(plot_total_volume, m)?,
        wrap_pyfunction!(plot_table, m)?,
        wrap_pyfunction!(plot_write_csv, m)?,
        wrap_pyfunction!(plot_write_meshes, m)?,
        wrap_pyfunction!(plot_write_cylinders, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
