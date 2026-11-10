// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the pulse operations of sylva_rs::shots and
//! sylva_rs::shots::ops: echo bookkeeping, subsets, stacking, beam angles and
//! the misses added from a scan pattern.
#![allow(clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::shots::ops::azimuth_deg;
use sylva_rs::canopy::profile as canopy_profile;
use sylva_rs::Shots;

use crate::canopy_py::pattern_from_py;
use crate::{err, shots_from_py, shots_to_py, xyz_from_py, xyz_to_py};

fn int64<'py>(py: Python<'py>, v: impl IntoIterator<Item = usize>) -> Bound<'py, PyArray1<i64>> {
    v.into_iter().map(|x| x as i64).collect::<Vec<_>>().into_pyarray(py)
}

/// Index of the owning shot of each echo.
#[pyfunction]
fn shots_shot_of_echo<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(int64(py, shots_from_py(shots)?.shot_of_echo()))
}

/// Rank of each echo within its shot (0 = first).
#[pyfunction]
fn shots_echo_rank<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(int64(py, shots_from_py(shots)?.echo_rank().into_iter().map(|r| r as usize)))
}

/// Echo coordinates, `origin + direction * range`.
#[pyfunction]
fn shots_echo_xyz<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(xyz_to_py(py, &shots_from_py(shots)?.echo_xyz()))
}

/// The shots where `keep` is true, with all their echoes.
#[pyfunction]
fn shots_subset<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, keep: PyReadonlyArray1<bool>) -> PyResult<Bound<'py, PyDict>> {
    let s = shots_from_py(shots)?;
    let keep = keep.as_array().to_vec();
    if keep.len() != s.n_shots() {
        return Err(PyValueError::new_err(format!("mask has {} values for {} shots", keep.len(), s.n_shots())));
    }
    shots_to_py(py, &s.subset(&keep))
}

/// Stack pulse sets; echo attributes common to all are kept.
#[pyfunction]
fn shots_concatenate<'py>(py: Python<'py>, parts: &Bound<'_, PyList>) -> PyResult<Bound<'py, PyDict>> {
    let parts: Vec<Shots> = parts.iter().map(|p| shots_from_py(p.cast::<PyDict>()?)).collect::<PyResult<_>>()?;
    shots_to_py(py, &Shots::concatenate(&parts.iter().collect::<Vec<_>>()).map_err(err)?)
}

/// Zenith and azimuth (degrees) of each direction.
#[pyfunction]
fn shots_zenith_azimuth<'py>(py: Python<'py>, direction: PyReadonlyArray2<f64>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let d = xyz_from_py(direction)?;
    Ok((canopy_profile::zenith_deg(&d).into_pyarray(py), azimuth_deg(&d).into_pyarray(py)))
}

/// The shots followed by the misses the scan pattern implies, or None if
/// nothing is missing.
#[pyfunction]
#[pyo3(signature = (shots, pattern, pulses_per_line=None, seed=0, shot_stride=1))]
fn shots_fill_missing<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, pattern: &Bound<'_, PyDict>, pulses_per_line: Option<i64>, seed: u64, shot_stride: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
    let s = shots_from_py(shots)?;
    let pattern = pattern_from_py(pattern)?;
    s.fill_missing(&pattern, pulses_per_line, seed, shot_stride.max(1)).map(|f| shots_to_py(py, &f)).transpose()
}

/// The scan's native lattice (sylva_rs::shots::lattice): `time` (s) and
/// `direction` (n, 3, scanner frame) per pulse in acquisition order. Returns
/// per-pulse `line`, `col` and residuals, the per-cell `count` (n_lines,
/// n_cols) and the steps.
#[pyfunction]
fn shots_scan_lattice<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    if t.len() != dir.len() {
        return Err(PyValueError::new_err(format!("time has {} values for {} pulses", t.len(), dir.len())));
    }
    let l = py.detach(|| sylva_rs::shots::lattice::scan_lattice(&t, &dir));
    let d = PyDict::new(py);
    d.set_item("line", l.line.into_pyarray(py))?;
    d.set_item("col", l.col.into_pyarray(py))?;
    d.set_item("slot_residual", l.slot_residual.into_pyarray(py))?;
    d.set_item("zenith_residual", l.zenith_residual.into_pyarray(py))?;
    d.set_item("azimuth_residual", l.azimuth_residual.into_pyarray(py))?;
    d.set_item("count", l.count.into_pyarray(py).reshape([l.n_lines, l.n_cols])?)?;
    d.set_item("line_phase", l.line_phase.into_pyarray(py))?;
    d.set_item("line_jitter", l.line_jitter.into_pyarray(py))?;
    d.set_item("line_azimuth", l.line_azimuth.into_pyarray(py))?;
    d.set_item("zenith_profile", l.zenith_profile.into_pyarray(py))?;
    d.set_item("azimuth_profile", l.azimuth_profile.into_pyarray(py))?;
    for (k, v) in [("pulse_interval", l.pulse_interval), ("line_period", l.line_period), ("zenith_step", l.zenith_step), ("azimuth_step", l.azimuth_step), ("zenith0", l.zenith0)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// One scan's opaque wood as silhouette stripes in its native range image
/// (sylva_rs::shots::stripes): `time` (s), `direction` (n, 3, unit, scanner
/// frame) and `range` (first echo, m) per pulse in acquisition order. Builds
/// the lattice, then returns per slice its stripe, pass, image centre
/// (`line`, `col`), run ends (`lo`, `hi`), `width_cells`, `lean`, front
/// `range`, `half_app`, `r`, `r_depth`, edges (`edge_lo`, `edge_hi`: 0 sharp,
/// 1 unknown, 2 occluded, 3 border), `forked`, `n_runs` and the `axis` point
/// (n, 3, scanner frame); plus each pulse's lattice `pulse_line`, `pulse_col`,
/// `n_lines`, `n_cols`, and the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, jump=0.05, gap=2, slice=4, min_cells=3, min_lines=8, tau=None))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_stripes<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, jump: f64, gap: usize, slice: usize, min_cells: usize, min_lines: usize, tau: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::stripes::{stripes, StripeParams};
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    if t.len() != dir.len() || r.len() != dir.len() {
        return Err(PyValueError::new_err("time, direction and range differ in length"));
    }
    let mut p = StripeParams { jump, gap, slice: slice.max(1), min_cells, min_lines, ..StripeParams::default() };
    if let Some(tau) = tau {
        p.beam.tau = tau;
    }
    let (lat, s, t_lat, t_str) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let c1 = std::time::Instant::now();
        let s = stripes(&lat, &r, &dir, &p);
        (lat, s, (c1 - c0).as_secs_f64(), c1.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, s.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!("stripe", |x| x.stripe);
    col!("pass", |x| x.pass);
    col!("line", |x| x.line);
    col!("col", |x| x.col);
    col!("lo", |x| x.lo);
    col!("hi", |x| x.hi);
    col!("n_runs", |x| x.n_runs);
    col!("forked", |x| x.forked);
    col!("width_cells", |x| x.width_cells);
    col!("lean", |x| x.lean);
    col!("range", |x| x.range);
    col!("half_app", |x| x.half_app);
    col!("r", |x| x.r);
    col!("r_depth", |x| x.r_depth);
    col!("edge_lo", |x| x.edge_lo);
    col!("edge_hi", |x| x.edge_hi);
    col!("open_lo", |x| x.open_lo);
    col!("open_hi", |x| x.open_hi);
    col!("step_m", |x| x.step_m);
    col!("footprint", |x| x.footprint);
    d.set_item("axis", xyz_to_py(py, &s.iter().map(|x| x.axis).collect::<Vec<_>>()))?;
    d.set_item("pulse_line", lat.line.into_pyarray(py))?;
    d.set_item("pulse_col", lat.col.into_pyarray(py))?;
    for (k, v) in [("n_lines", lat.n_lines as f64), ("n_cols", lat.n_cols as f64), ("lattice_s", t_lat), ("stripes_s", t_str), ("zenith_step", lat.zenith_step), ("azimuth_step", lat.azimuth_step)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(shots_shot_of_echo, m)?,
        wrap_pyfunction!(shots_echo_rank, m)?,
        wrap_pyfunction!(shots_echo_xyz, m)?,
        wrap_pyfunction!(shots_subset, m)?,
        wrap_pyfunction!(shots_concatenate, m)?,
        wrap_pyfunction!(shots_zenith_azimuth, m)?,
        wrap_pyfunction!(shots_fill_missing, m)?,
        wrap_pyfunction!(shots_scan_lattice, m)?,
        wrap_pyfunction!(shots_scan_stripes, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
