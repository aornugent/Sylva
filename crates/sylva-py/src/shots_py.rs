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

fn point3(v: Vec<f64>, name: &str) -> PyResult<[f64; 3]> {
    v.try_into().map_err(|_| PyValueError::new_err(format!("{name} needs 3 values")))
}

/// One scan's stripes (as `shots_scan_stripes`) as short axis pieces in the
/// world frame (sylva_rs::shots::tracts::segments): `rot` (3, 3, world =
/// rot . scanner) and `origin` place the scan; pieces whose point lies in the
/// box `lo`..`hi` are kept. Returns `p`, `u` (n, 3), `half`, `w`, `r`, the
/// slice count and the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, rot, origin, lo, hi, window=5, max_step=6.0))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_segments<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, window: usize, max_step: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::stripes::{stripes, StripeParams};
    use sylva_rs::shots::tracts::segments;
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    if t.len() != dir.len() || r.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("time, direction and range differ in length, or rot isn't 3 x 3"));
    }
    let (origin, lo, hi) = (point3(origin, "origin")?, point3(lo, "lo")?, point3(hi, "hi")?);
    let (n_sl, segs, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let c1 = std::time::Instant::now();
        let s = stripes(&lat, &r, &dir, &StripeParams::default());
        let c2 = std::time::Instant::now();
        let g = segments(&s, [rot[0], rot[1], rot[2]], origin, window, max_step, lo, hi);
        (s.len(), g, [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), c2.elapsed().as_secs_f64()])
    });
    let d = PyDict::new(py);
    d.set_item("p", xyz_to_py(py, &segs.iter().map(|x| x.p).collect::<Vec<_>>()))?;
    d.set_item("u", xyz_to_py(py, &segs.iter().map(|x| x.u).collect::<Vec<_>>()))?;
    d.set_item("half", segs.iter().map(|x| x.half).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("w", segs.iter().map(|x| x.w).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("r", segs.iter().map(|x| x.r).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n_slices", n_sl)?;
    for (k, v) in ["lattice_s", "stripes_s", "segments_s"].iter().zip(secs) {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

/// Streamlines through the orientation field of axis pieces `p`, `u`, `half`,
/// `w` (sylva_rs::shots::tracts): from each seed (n, 3), heading up, within
/// the box `lo`..`hi`. Returns per point `point` (n, 3), `line`, `seed`; per
/// line `line_seed`, `line_parent` (-1 from the seed) and `line_stop` (0 band,
/// 1 no data, 2 merged, 3 step cap, 4 line cap); the voxels' `voxel_key`
/// (n, 3), `voxel_w` and `voxel_l1` (the tensor's l1 / trace); and the seconds
/// to build the field and to trace.
#[pyfunction]
#[pyo3(signature = (p, u, half, w, seeds, lo, hi, voxel=0.025, step=0.01, cone=45.0, min_w=0.05, gap=0.10, seed_gap=0.5, fork_share=0.25, fork_angle=20.0, fork_every=0.10, centre=0.5, merge_after=0.10, max_lines=400))]
#[allow(clippy::too_many_arguments)]
fn shots_tracts<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>, half: PyReadonlyArray1<f64>, w: PyReadonlyArray1<f64>, seeds: PyReadonlyArray2<f64>, lo: Vec<f64>, hi: Vec<f64>, voxel: f64, step: f64, cone: f64, min_w: f64, gap: f64, seed_gap: f64, fork_share: f64, fork_angle: f64, fork_every: f64, centre: f64, merge_after: f64, max_lines: usize) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::tracts::{trace, Field, Segment, TractParams};
    let (pp, uu, seeds) = (xyz_from_py(p)?, xyz_from_py(u)?, xyz_from_py(seeds)?);
    let (half, w) = (half.as_array().to_vec(), w.as_array().to_vec());
    if uu.len() != pp.len() || half.len() != pp.len() || w.len() != pp.len() {
        return Err(PyValueError::new_err("p, u, half and w differ in length"));
    }
    let (lo, hi) = (point3(lo, "lo")?, point3(hi, "hi")?);
    let segs: Vec<Segment> = (0..pp.len()).map(|i| Segment { p: pp[i], u: uu[i], half: half[i], w: w[i], r: f64::NAN, slice: 0 }).collect();
    let prm = TractParams { voxel, step, cone, min_w, gap, seed_gap, fork_share, fork_angle, fork_every, centre, merge_after, max_lines, ..TractParams::default() };
    let (t, summary, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let f = Field::build(&segs, prm.voxel);
        let c1 = std::time::Instant::now();
        let t = trace(&f, &seeds, lo, hi, &prm);
        let c2 = std::time::Instant::now();
        (t, f.summary(), [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64()])
    });
    let d = PyDict::new(py);
    d.set_item("point", xyz_to_py(py, &t.point))?;
    d.set_item("line", t.line.into_pyarray(py))?;
    d.set_item("seed", t.seed.into_pyarray(py))?;
    d.set_item("line_seed", t.line_seed.into_pyarray(py))?;
    d.set_item("line_parent", t.line_parent.into_pyarray(py))?;
    d.set_item("line_stop", t.line_stop.into_pyarray(py))?;
    d.set_item("voxel_key", summary.iter().flat_map(|x| x.0).collect::<Vec<i32>>().into_pyarray(py).reshape([summary.len(), 3])?)?;
    d.set_item("voxel_w", summary.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("voxel_l1", summary.iter().map(|x| x.2).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("build_s", secs[0])?;
    d.set_item("trace_s", secs[1])?;
    Ok(d)
}

/// One scan as a shadow map (sylva_rs::shots::opacity), for 3dtrees research
/// 128's opacity spike. Takes the pulses as `shots_scan_segments` does, plus
/// each pulse's echo count `n_echo` and first-echo reflectance `refl`, and
/// query cylinders `cyl_p`, `cyl_u` (n, 3, world), `cyl_half`, `cyl_r` (m).
/// Returns the scan's own segments in the box `lo`..`hi` (`p`, `u`, `half`,
/// `w`, `r`) with their slices' `pass`, `line`, `col`, `lo_cell`, `hi_cell`,
/// `range`, `half_app`, `edge_lo`, `edge_hi`, `open_lo`, `open_hi`,
/// `width_cells`, `stripe`, and the single-scan features (`jitter`, `multi`,
/// `refl`, `empty`, `arc`, `fa`, `cells`); the query cylinders' samples
/// against this scan (`s_cyl`, `s_outcome` 0 support 1 refute 2 occluded 3
/// unknown, `s_chord`, `s_miss`, `s_t_in`, `s_depth`); the lattice round trip
/// on every 997th pulse (`roundtrip_line`, `roundtrip_col`: cell_of minus the
/// cell); optionally (`image`) every pulse's `pulse_line`, `pulse_col`; and
/// the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, n_echo, refl, rot, origin, lo, hi, cyl_p, cyl_u, cyl_half, cyl_r, tol=0.015, voxel=0.06, pad=3, window=5, max_step=6.0, max_cells=60.0, image=false))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_opacity<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, n_echo: PyReadonlyArray1<u8>, refl: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, cyl_p: PyReadonlyArray2<f64>, cyl_u: PyReadonlyArray2<f64>, cyl_half: PyReadonlyArray1<f64>, cyl_r: PyReadonlyArray1<f64>, tol: f64, voxel: f64, pad: i64, window: usize, max_step: f64, max_cells: f64, image: bool) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::opacity::{features, Cyl, ShadowMap};
    use sylva_rs::shots::stripes::{stripes, StripeParams};
    use sylva_rs::shots::tracts::segments;
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let ne = n_echo.as_array().to_vec();
    let rf = refl.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    if t.len() != dir.len() || r.len() != dir.len() || ne.len() != dir.len() || rf.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("per-pulse arrays differ in length, or rot isn't 3 x 3"));
    }
    let (origin, lo, hi) = (point3(origin, "origin")?, point3(lo, "lo")?, point3(hi, "hi")?);
    let (cp, cu) = (xyz_from_py(cyl_p)?, xyz_from_py(cyl_u)?);
    let (ch, cr) = (cyl_half.as_array().to_vec(), cyl_r.as_array().to_vec());
    let cyls: Vec<Cyl> = (0..cp.len()).map(|i| Cyl { p: cp[i], u: cu[i], half: ch[i], r: cr[i] }).collect();
    let rot = [rot[0], rot[1], rot[2]];
    let (sl, segs, feats, samples, rt, secs, img) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let c1 = std::time::Instant::now();
        let prm = StripeParams::default();
        let sl = stripes(&lat, &r, &dir, &prm);
        let segs = segments(&sl, rot, origin, window, max_step, lo, hi);
        let c2 = std::time::Instant::now();
        let map = ShadowMap::new(&lat, &r, rot, origin);
        let feats = features(&map, &sl, &segs, &ne, &rf, voxel, pad);
        let c3 = std::time::Instant::now();
        let beam = prm.beam.clone();
        let samples = map.test(&cyls, |x| beam.footprint(x), tol, max_cells);
        let c4 = std::time::Instant::now();
        let mut rt = (Vec::new(), Vec::new());
        for i in (0..dir.len()).step_by(997) {
            if lat.line[i] == u32::MAX {
                continue;
            }
            if let Some((l, c)) = lat.cell_of(&dir[i]) {
                rt.0.push(l - lat.line[i] as f64);
                rt.1.push(c - lat.col[i] as f64);
            }
        }
        let secs = [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), (c3 - c2).as_secs_f64(), (c4 - c3).as_secs_f64()];
        let img = image.then(|| (lat.line.clone(), lat.col.clone(), lat.n_lines, lat.n_cols));
        (sl, segs, feats, samples, rt, secs, img)
    });
    let d = PyDict::new(py);
    d.set_item("p", xyz_to_py(py, &segs.iter().map(|x| x.p).collect::<Vec<_>>()))?;
    d.set_item("u", xyz_to_py(py, &segs.iter().map(|x| x.u).collect::<Vec<_>>()))?;
    macro_rules! col {
        ($v:expr, $name:literal, $f:expr) => {
            d.set_item($name, $v.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!(segs, "half", |x| x.half);
    col!(segs, "w", |x| x.w);
    col!(segs, "r", |x| x.r);
    col!(segs, "pass", |x| sl[x.slice].pass);
    col!(segs, "line", |x| sl[x.slice].line);
    col!(segs, "col", |x| sl[x.slice].col);
    col!(segs, "lo_cell", |x| sl[x.slice].lo);
    col!(segs, "hi_cell", |x| sl[x.slice].hi);
    col!(segs, "range", |x| sl[x.slice].range);
    col!(segs, "half_app", |x| sl[x.slice].half_app);
    col!(segs, "edge_lo", |x| sl[x.slice].edge_lo);
    col!(segs, "edge_hi", |x| sl[x.slice].edge_hi);
    col!(segs, "open_lo", |x| sl[x.slice].open_lo);
    col!(segs, "open_hi", |x| sl[x.slice].open_hi);
    col!(segs, "width_cells", |x| sl[x.slice].width_cells);
    col!(segs, "stripe", |x| sl[x.slice].stripe);
    col!(feats, "jitter", |x| x.jitter);
    col!(feats, "multi", |x| x.multi);
    col!(feats, "refl", |x| x.refl);
    col!(feats, "empty", |x| x.empty);
    col!(feats, "arc", |x| x.arc);
    col!(feats, "fa", |x| x.fa);
    col!(feats, "cells", |x| x.cells);
    col!(samples, "s_cyl", |x| x.cyl);
    col!(samples, "s_outcome", |x| x.outcome);
    col!(samples, "s_chord", |x| x.chord);
    col!(samples, "s_miss", |x| x.miss);
    col!(samples, "s_t_in", |x| x.t_in);
    col!(samples, "s_depth", |x| x.depth);
    d.set_item("roundtrip_line", rt.0.into_pyarray(py))?;
    d.set_item("roundtrip_col", rt.1.into_pyarray(py))?;
    d.set_item("n_slices", sl.len())?;
    if let Some((pl, pc, nl, nc)) = img {
        d.set_item("pulse_line", pl.into_pyarray(py))?;
        d.set_item("pulse_col", pc.into_pyarray(py))?;
        d.set_item("n_lines", nl)?;
        d.set_item("n_cols", nc)?;
    }
    for (k, v) in ["lattice_s", "stripes_s", "features_s", "test_s"].iter().zip(secs) {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

/// One scan's surfels (sylva_rs::shots::surfels), for 3dtrees research 128's
/// surfel spike. Takes the pulses as `shots_scan_opacity` does (directions in
/// the scanner's frame) and keeps first echoes in the box `lo`..`hi`, or, given
/// `window` (line0, line1, col0, col1), those in that block of cells instead.
/// Returns per surfel `pulse`, `line`, `col`, world `p` and `n` (toward the
/// scanner), `sigma_n` (rad), `sigma_lat` (m), `rms` (m), `cells`, `cos_inc`;
/// `failed` (kept pulses with no surfel) and the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, rot, origin, lo, hi, window=None, half=3, jump=0.05, range_noise=0.003, min_aspect=0.2))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_surfels<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, window: Option<(u32, u32, u32, u32)>, half: i64, jump: f64, range_noise: f64, min_aspect: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::opacity::ShadowMap;
    use sylva_rs::shots::surfels::{surfels, SurfelParams};
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    if t.len() != dir.len() || r.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("per-pulse arrays differ in length, or rot isn't 3 x 3"));
    }
    let (origin, lo, hi) = (point3(origin, "origin")?, point3(lo, "lo")?, point3(hi, "hi")?);
    let rot = [rot[0], rot[1], rot[2]];
    let (s, failed, line, col, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let c1 = std::time::Instant::now();
        let map = ShadowMap::new(&lat, &r, rot, origin);
        let beam = sylva_rs::qsm::joint::JointParams::default();
        let prm = SurfelParams { half, jump, range_noise, min_aspect, ..SurfelParams::default() };
        let keep = |i: usize, p: &[f64; 3]| match window {
            Some((l0, l1, k0, k1)) => (l0..l1).contains(&lat.line[i]) && (k0..k1).contains(&lat.col[i]),
            None => (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]),
        };
        let (s, failed) = surfels(&map, &dir, keep, |x| beam.footprint(x), &prm);
        let line: Vec<u32> = s.iter().map(|x| lat.line[x.pulse as usize]).collect();
        let col: Vec<u32> = s.iter().map(|x| lat.col[x.pulse as usize]).collect();
        let c2 = std::time::Instant::now();
        (s, failed, line, col, [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64()])
    });
    let d = PyDict::new(py);
    d.set_item("p", xyz_to_py(py, &s.iter().map(|x| x.p).collect::<Vec<_>>()))?;
    d.set_item("n", xyz_to_py(py, &s.iter().map(|x| x.n).collect::<Vec<_>>()))?;
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, s.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!("pulse", |x| x.pulse);
    col!("sigma_n", |x| x.sigma_n);
    col!("sigma_lat", |x| x.sigma_lat);
    col!("rms", |x| x.rms);
    col!("cells", |x| x.cells);
    col!("cos_inc", |x| x.cos_inc);
    d.set_item("line", line.into_pyarray(py))?;
    d.set_item("col", col.into_pyarray(py))?;
    d.set_item("failed", failed)?;
    d.set_item("lattice_s", secs[0])?;
    d.set_item("surfels_s", secs[1])?;
    Ok(d)
}

/// The surfel statistics of query regions (sylva_rs::shots::surfels::region_stats): surfels `p`, `n` (m, 3) with
/// their `scan`; regions `reg_p`, `reg_u` (k, 3), `reg_half`, `reg_r` and `reg_scan` (-1: every scan's surfels).
/// Returns per region `n`, `n_scans`, `thick_plane`, `thick_cyl`, `lam` (k, 3), `axis` (k, 3), `medial_r`,
/// `spread`, `spread0`, `centre` (k, 3), and `stats_s`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, reg_p, reg_u, reg_half, reg_r, reg_scan, margin=0.02, min_n=10, r_max=0.3))]
#[allow(clippy::too_many_arguments)]
fn shots_surfel_stats<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, reg_p: PyReadonlyArray2<f64>, reg_u: PyReadonlyArray2<f64>, reg_half: PyReadonlyArray1<f64>, reg_r: PyReadonlyArray1<f64>, reg_scan: PyReadonlyArray1<i32>, margin: f64, min_n: usize, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::surfels::{region_stats, Region};
    let (p, n, scan) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec());
    let (rp, ru) = (xyz_from_py(reg_p)?, xyz_from_py(reg_u)?);
    let (rh, rr, rs) = (reg_half.as_array().to_vec(), reg_r.as_array().to_vec(), reg_scan.as_array().to_vec());
    if n.len() != p.len() || scan.len() != p.len() || ru.len() != rp.len() || rh.len() != rp.len() || rr.len() != rp.len() || rs.len() != rp.len() {
        return Err(PyValueError::new_err("surfel or region arrays differ in length"));
    }
    let regs: Vec<Region> = (0..rp.len()).map(|i| Region { p: rp[i], u: ru[i], half: rh[i], r: rr[i], scan: rs[i] }).collect();
    let (st, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let st = region_stats(&p, &n, &scan, &regs, margin, min_n, r_max);
        (st, c0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, st.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!("n", |x| x.n);
    col!("n_scans", |x| x.n_scans);
    col!("thick_plane", |x| x.thick_plane);
    col!("thick_cyl", |x| x.thick_cyl);
    col!("medial_r", |x| x.medial_r);
    col!("spread", |x| x.spread);
    col!("spread0", |x| x.spread0);
    d.set_item("lam", xyz_to_py(py, &st.iter().map(|x| x.lam).collect::<Vec<_>>()))?;
    d.set_item("axis", xyz_to_py(py, &st.iter().map(|x| x.axis).collect::<Vec<_>>()))?;
    d.set_item("centre", xyz_to_py(py, &st.iter().map(|x| x.centre).collect::<Vec<_>>()))?;
    d.set_item("stats_s", secs)?;
    Ok(d)
}

/// One scan's identity signals (sylva_rs::shots::identities), for 3dtrees research 128's identities spike: per kept
/// pulse (first echo in the box `lo`..`hi`) its world `p`, `surfel` (whether it has a plane-fit surfel; its normal in
/// `n_plane`, NaN without), the gradient-of-log-range normals `n_grad1` / `n_grad3` (3 x 3 and 7 x 7 windows) with
/// `cos_grad1` / `cos_grad3` and `full1` / `full3`, and `ridge` (its ridge, -1 none); per ridge `r_cells`,
/// `r_len_cells`, `r_width_cells`, `r_centre`, `r_u`, `r_len`, `r_rms_off`, `r_end0`, `r_end1`, `r_plane`, `r_refl`,
/// `r_multi`, `r_range`; and the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, refl, echoes, rot, origin, lo, hi, jump=0.05))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_identities<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, refl: PyReadonlyArray1<f32>, echoes: PyReadonlyArray1<u8>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, jump: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::identities::{grad_normals, ridges};
    use sylva_rs::shots::opacity::ShadowMap;
    use sylva_rs::shots::surfels::{surfels, SurfelParams};
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let (rf, ne) = (refl.as_array().to_vec(), echoes.as_array().to_vec());
    let rot = xyz_from_py(rot)?;
    if t.len() != dir.len() || r.len() != dir.len() || rf.len() != dir.len() || ne.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("per-pulse arrays differ in length, or rot isn't 3 x 3"));
    }
    let (origin, lo, hi) = (point3(origin, "origin")?, point3(lo, "lo")?, point3(hi, "hi")?);
    let rot = [rot[0], rot[1], rot[2]];
    let nan = [f64::NAN; 3];
    let out = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let map = ShadowMap::new(&lat, &r, rot, origin);
        let beam = sylva_rs::qsm::joint::JointParams::default();
        let keep = |_: usize, p: &[f64; 3]| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
        let c1 = std::time::Instant::now();
        let (s, _) = surfels(&map, &dir, keep, |x| beam.footprint(x), &SurfelParams::default());
        let c2 = std::time::Instant::now();
        let g1 = grad_normals(&map, &dir, keep, 1, 0.05, 0.2);
        let c3 = std::time::Instant::now();
        let g3 = grad_normals(&map, &dir, keep, 3, 0.05, 0.2);
        let c4 = std::time::Instant::now();
        let mut has = vec![false; dir.len()];
        for x in &s {
            has[x.pulse as usize] = true;
        }
        let (label, rd) = ridges(&map, &dir, &rf, &ne, keep, &has, jump);
        let c5 = std::time::Instant::now();
        // the kept pulses are those with a gradient normal attempt: every kept first echo
        let pulses: Vec<usize> = g3.iter().map(|x| x.pulse as usize).collect();
        let secs = [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), (c3 - c2).as_secs_f64(), (c4 - c3).as_secs_f64(), (c5 - c4).as_secs_f64()];
        (s, g1, g3, label, rd, pulses, secs)
    });
    let (s, g1, g3, label, rd, pulses, secs) = out;
    // the kept pulses: those with a 7 x 7 gradient normal (every kept cell with a neighbour), plus surfel pulses
    let mut all: Vec<usize> = pulses.clone();
    all.extend(s.iter().map(|x| x.pulse as usize));
    all.extend(label.iter().enumerate().filter(|(_, &l)| l != u32::MAX).map(|(i, _)| i));
    all.sort_unstable();
    all.dedup();
    let pos: std::collections::HashMap<usize, usize> = all.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let n = all.len();
    let mut n_plane = vec![nan; n];
    let mut surf = vec![false; n];
    for x in &s {
        let k = pos[&(x.pulse as usize)];
        n_plane[k] = x.n;
        surf[k] = true;
    }
    let fill = |g: &[sylva_rs::shots::identities::GradNormal]| {
        let (mut nn, mut cs, mut fl) = (vec![nan; n], vec![f64::NAN; n], vec![false; n]);
        for x in g {
            let k = pos[&(x.pulse as usize)];
            nn[k] = x.n;
            cs[k] = x.cos_inc;
            fl[k] = x.full;
        }
        (nn, cs, fl)
    };
    let (n1, c1, f1) = fill(&g1);
    let (n3, c3, f3) = fill(&g3);
    let rot_m = rot;
    let p: Vec<[f64; 3]> = all
        .iter()
        .map(|&i| {
            let x = [dir[i][0] * r[i] as f64, dir[i][1] * r[i] as f64, dir[i][2] * r[i] as f64];
            [0, 1, 2].map(|a| rot_m[a][0] * x[0] + rot_m[a][1] * x[1] + rot_m[a][2] * x[2] + origin[a])
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("pulse", all.iter().map(|&i| i as u32).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("p", xyz_to_py(py, &p))?;
    d.set_item("surfel", surf.into_pyarray(py))?;
    d.set_item("n_plane", xyz_to_py(py, &n_plane))?;
    d.set_item("n_grad1", xyz_to_py(py, &n1))?;
    d.set_item("n_grad3", xyz_to_py(py, &n3))?;
    d.set_item("cos_grad1", c1.into_pyarray(py))?;
    d.set_item("cos_grad3", c3.into_pyarray(py))?;
    d.set_item("full1", f1.into_pyarray(py))?;
    d.set_item("full3", f3.into_pyarray(py))?;
    d.set_item("ridge", all.iter().map(|&i| if label[i] == u32::MAX { -1 } else { label[i] as i64 }).collect::<Vec<_>>().into_pyarray(py))?;
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, rd.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    macro_rules! xyz {
        ($name:literal, $f:expr) => {
            d.set_item($name, xyz_to_py(py, &rd.iter().map($f).collect::<Vec<_>>()))?;
        };
    }
    col!("r_cells", |x| x.cells);
    col!("r_len_cells", |x| x.len_cells);
    col!("r_width_cells", |x| x.width_cells);
    col!("r_len", |x| x.len);
    col!("r_rms_off", |x| x.rms_off);
    col!("r_refl", |x| x.refl);
    col!("r_multi", |x| x.multi);
    col!("r_range", |x| x.range);
    xyz!("r_centre", |x| x.centre);
    xyz!("r_u", |x| x.u);
    xyz!("r_end0", |x| x.end0);
    xyz!("r_end1", |x| x.end1);
    xyz!("r_plane", |x| x.plane);
    for (k, name) in ["lattice_s", "surfels_s", "grad1_s", "grad3_s", "ridges_s"].iter().enumerate() {
        d.set_item(*name, secs[k])?;
    }
    Ok(d)
}

/// Line-complex axes, medial collapse and support harmonics of query regions (sylva_rs::shots::identities::
/// region_lines): surfels `p`, `n` (m, 3), `scan`, `delta` (m; each surfel moved along its normal by it before the
/// fits, after the gather; empty for none), `attr` (k, m: k attributes averaged over each region); regions as
/// `shots_surfel_stats`; `what` as region_lines. Returns per region `n`, `n_scans`; for the cylinder (`cyl_`) and
/// cone (`cone_`) fits `u`, `c` (k, 3), `lam`, `lam2`, `r`, `nu`, `used`; medial `med_u`, `med_c`, `med_r`,
/// `med_spread`, `med_spread0`; the same about the line-complex axis `mlc_c`, `mlc_r`, `mlc_spread`, `mlc_spread0`;
/// `harm` (k, 5), `harm_gap` (rad), `harm_rms` (m), `harm_n`; `attr` (k_regions, k);
/// `secs`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, delta, attr, reg_p, reg_u, reg_half, reg_r, reg_scan, margin=0.02, min_n=10, what=31))]
#[allow(clippy::too_many_arguments)]
fn shots_region_lines<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, delta: PyReadonlyArray1<f64>, attr: PyReadonlyArray2<f64>, reg_p: PyReadonlyArray2<f64>, reg_u: PyReadonlyArray2<f64>, reg_half: PyReadonlyArray1<f64>, reg_r: PyReadonlyArray1<f64>, reg_scan: PyReadonlyArray1<i32>, margin: f64, min_n: usize, what: u32) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray2;
    use sylva_rs::shots::identities::region_lines;
    use sylva_rs::shots::surfels::Region;
    let (p, n, scan, delta) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec(), delta.as_array().to_vec());
    let attr: Vec<Vec<f64>> = attr.as_array().rows().into_iter().map(|r| r.to_vec()).collect();
    let (rp, ru) = (xyz_from_py(reg_p)?, xyz_from_py(reg_u)?);
    let (rh, rr, rs) = (reg_half.as_array().to_vec(), reg_r.as_array().to_vec(), reg_scan.as_array().to_vec());
    if n.len() != p.len() || scan.len() != p.len() || (!delta.is_empty() && delta.len() != p.len()) || attr.iter().any(|a| a.len() != p.len())
        || ru.len() != rp.len() || rh.len() != rp.len() || rr.len() != rp.len() || rs.len() != rp.len()
    {
        return Err(PyValueError::new_err("surfel or region arrays differ in length"));
    }
    let regs: Vec<Region> = (0..rp.len()).map(|i| Region { p: rp[i], u: ru[i], half: rh[i], r: rr[i], scan: rs[i] }).collect();
    let (st, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let st = region_lines(&p, &n, &scan, &delta, &attr, &regs, margin, min_n, what);
        (st, c0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    let nan = [f64::NAN; 3];
    let p3 = |x: &nalgebra::Vector3<f64>| [x[0], x[1], x[2]];
    d.set_item("n", st.iter().map(|x| x.n).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n_scans", st.iter().map(|x| x.n_scans).collect::<Vec<_>>().into_pyarray(py))?;
    for (pre, get) in [("cyl_", 0usize), ("cone_", 1)] {
        let f = |x: &sylva_rs::shots::identities::RegionLines| if get == 0 { x.cyl } else { x.cone };
        d.set_item(format!("{pre}u"), xyz_to_py(py, &st.iter().map(|x| f(x).map(|l| p3(&l.u)).unwrap_or(nan)).collect::<Vec<_>>()))?;
        d.set_item(format!("{pre}c"), xyz_to_py(py, &st.iter().map(|x| f(x).map(|l| p3(&l.c)).unwrap_or(nan)).collect::<Vec<_>>()))?;
        d.set_item(format!("{pre}lam"), st.iter().map(|x| f(x).map(|l| l.lam).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item(format!("{pre}lam2"), st.iter().map(|x| f(x).map(|l| l.lam2).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item(format!("{pre}r"), st.iter().map(|x| f(x).map(|l| l.r).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item(format!("{pre}nu"), st.iter().map(|x| f(x).map(|l| l.nu).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item(format!("{pre}used"), st.iter().map(|x| f(x).map(|l| l.used as i64).unwrap_or(0)).collect::<Vec<_>>().into_pyarray(py))?;
    }
    d.set_item("med_u", xyz_to_py(py, &st.iter().map(|x| x.medial.map(|m| p3(&m.0)).unwrap_or(nan)).collect::<Vec<_>>()))?;
    d.set_item("med_c", xyz_to_py(py, &st.iter().map(|x| x.medial.map(|m| p3(&m.4)).unwrap_or(nan)).collect::<Vec<_>>()))?;
    d.set_item("med_r", st.iter().map(|x| x.medial.map(|m| m.1).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("med_spread", st.iter().map(|x| x.medial.map(|m| m.2).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("med_spread0", st.iter().map(|x| x.medial.map(|m| m.3).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("mlc_c", xyz_to_py(py, &st.iter().map(|x| x.medial_lc.map(|m| p3(&m.4)).unwrap_or(nan)).collect::<Vec<_>>()))?;
    d.set_item("mlc_r", st.iter().map(|x| x.medial_lc.map(|m| m.1).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("mlc_spread", st.iter().map(|x| x.medial_lc.map(|m| m.2).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("mlc_spread0", st.iter().map(|x| x.medial_lc.map(|m| m.3).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    let harm: Vec<Vec<f64>> = st.iter().map(|x| x.support.map(|s| s.0.to_vec()).unwrap_or(vec![f64::NAN; 5])).collect();
    d.set_item("harm", PyArray2::from_vec2(py, &harm)?)?;
    d.set_item("harm_gap", st.iter().map(|x| x.support.map(|s| s.1).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("harm_rms", st.iter().map(|x| x.support.map(|s| s.2).unwrap_or(f64::NAN)).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("harm_n", st.iter().map(|x| x.support.map(|s| s.3 as i64).unwrap_or(0)).collect::<Vec<_>>().into_pyarray(py))?;
    let k = attr.len();
    let am: Vec<Vec<f64>> = st.iter().map(|x| if k == 0 { vec![] } else { x.attr.clone() }).collect();
    if k > 0 {
        d.set_item("attr", PyArray2::from_vec2(py, &am)?)?;
    }
    d.set_item("secs", secs)?;
    Ok(d)
}

/// Per region and scan, where that scan's medial points sit across the given axis (sylva_rs::shots::strands::
/// scan_offsets): surfels `p`, `n`, `scan`; regions `reg_p`, `reg_u`, `reg_half`, `reg_r` (the gather), each one's
/// line point `c` and radius `r`. Returns `region`, `scan`, `offset` (k, 3, world, across the axis), `count`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, reg_p, reg_u, reg_half, reg_r, c, r, margin=0.02, min_n=8))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_offsets<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, reg_p: PyReadonlyArray2<f64>, reg_u: PyReadonlyArray2<f64>, reg_half: PyReadonlyArray1<f64>, reg_r: PyReadonlyArray1<f64>, c: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, margin: f64, min_n: usize) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::strands::scan_offsets;
    use sylva_rs::shots::surfels::Region;
    let (p, n, scan) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec());
    let (rp, ru, cc) = (xyz_from_py(reg_p)?, xyz_from_py(reg_u)?, xyz_from_py(c)?);
    let (rh, rr, r) = (reg_half.as_array().to_vec(), reg_r.as_array().to_vec(), r.as_array().to_vec());
    if n.len() != p.len() || scan.len() != p.len() || ru.len() != rp.len() || rh.len() != rp.len() || rr.len() != rp.len() || cc.len() != rp.len() || r.len() != rp.len() {
        return Err(PyValueError::new_err("surfel or region arrays differ in length"));
    }
    let regs: Vec<Region> = (0..rp.len()).map(|i| Region { p: rp[i], u: ru[i], half: rh[i], r: rr[i], scan: -1 }).collect();
    let o = py.detach(|| scan_offsets(&p, &n, &scan, &regs, &cc, &r, margin, min_n));
    let d = PyDict::new(py);
    d.set_item("region", o.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", o.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("offset", xyz_to_py(py, &o.iter().map(|x| x.2).collect::<Vec<_>>()))?;
    d.set_item("count", o.iter().map(|x| x.3).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

/// Candidate links between fragment ends (sylva_rs::shots::strands::link_ends): ends `p`, outward `d` (k, 3), `r`,
/// `frag`. Returns for every candidate (`all_`) and the mutual-best subset (`best_`) `a`, `b`, `gap`, `ang_a`,
/// `ang_b`, `dlogr`, `cost`.
#[pyfunction]
#[pyo3(signature = (p, d, r, frag, min_gap=0.02, max_gap=1.0, max_ang=30.0, max_dlogr=0.4, sd_ang=15.0, sd_logr=0.2))]
#[allow(clippy::too_many_arguments)]
fn shots_link_ends<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, d: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, frag: PyReadonlyArray1<u32>, min_gap: f64, max_gap: f64, max_ang: f64, max_dlogr: f64, sd_ang: f64, sd_logr: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::strands::{link_ends, End};
    let (p, dd, r, f) = (xyz_from_py(p)?, xyz_from_py(d)?, r.as_array().to_vec(), frag.as_array().to_vec());
    if dd.len() != p.len() || r.len() != p.len() || f.len() != p.len() {
        return Err(PyValueError::new_err("end arrays differ in length"));
    }
    let ends: Vec<End> = (0..p.len()).map(|i| End { p: p[i], d: dd[i], r: r[i], frag: f[i] }).collect();
    let (all, best) = py.detach(|| link_ends(&ends, min_gap, max_gap, max_ang, max_dlogr, sd_ang, sd_logr));
    let out = PyDict::new(py);
    for (pre, l) in [("all_", &all), ("best_", &best)] {
        out.set_item(format!("{pre}a"), l.iter().map(|x| x.a).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}b"), l.iter().map(|x| x.b).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}gap"), l.iter().map(|x| x.gap).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}ang_a"), l.iter().map(|x| x.ang_a).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}ang_b"), l.iter().map(|x| x.ang_b).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}dlogr"), l.iter().map(|x| x.dlogr).collect::<Vec<_>>().into_pyarray(py))?;
        out.set_item(format!("{pre}cost"), l.iter().map(|x| x.cost).collect::<Vec<_>>().into_pyarray(py))?;
    }
    Ok(out)
}

/// Bundles of lifted per-scan pieces (sylva_rs::shots::strands::bundles): `p`, `u` (k, 3), `half`, `r`, `scan`,
/// `stripe`, image `line`, `col`, front `range`, `width` (cells). Returns `bundle` (a root piece per piece) and the
/// counts `edges`, `merged`, `blocked`, `cannot`.
#[pyfunction]
#[pyo3(signature = (p, u, half, r, scan, stripe, line, col, range, width, voxel=0.03, max_ang=25.0, max_dlogr=0.5, near_cells=3.0, depth_step=0.1))]
#[allow(clippy::too_many_arguments)]
fn shots_bundles<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>, half: PyReadonlyArray1<f64>, r: PyReadonlyArray1<f64>, scan: PyReadonlyArray1<i32>, stripe: PyReadonlyArray1<u32>, line: PyReadonlyArray1<f64>, col: PyReadonlyArray1<f64>, range: PyReadonlyArray1<f64>, width: PyReadonlyArray1<f64>, voxel: f64, max_ang: f64, max_dlogr: f64, near_cells: f64, depth_step: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::strands::{bundles, Piece};
    let (p, u) = (xyz_from_py(p)?, xyz_from_py(u)?);
    let (h, r, s, st) = (half.as_array().to_vec(), r.as_array().to_vec(), scan.as_array().to_vec(), stripe.as_array().to_vec());
    let (l, c, rg, w) = (line.as_array().to_vec(), col.as_array().to_vec(), range.as_array().to_vec(), width.as_array().to_vec());
    let n = p.len();
    if [u.len(), h.len(), r.len(), s.len(), st.len(), l.len(), c.len(), rg.len(), w.len()].iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("piece arrays differ in length"));
    }
    let pc: Vec<Piece> = (0..n).map(|i| Piece { p: p[i], u: u[i], half: h[i], r: r[i], scan: s[i], stripe: st[i], line: l[i], col: c[i], range: rg[i] }).collect();
    let (b, ms) = py.detach(|| bundles(&pc, &w, voxel, max_ang, max_dlogr, near_cells, depth_step));
    let d = PyDict::new(py);
    d.set_item("bundle", b.into_pyarray(py))?;
    d.set_item("edges", ms.edges)?;
    d.set_item("merged", ms.merged)?;
    d.set_item("blocked", ms.blocked)?;
    d.set_item("cannot", ms.cannot)?;
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
        wrap_pyfunction!(shots_scan_segments, m)?,
        wrap_pyfunction!(shots_tracts, m)?,
        wrap_pyfunction!(shots_scan_opacity, m)?,
        wrap_pyfunction!(shots_scan_surfels, m)?,
        wrap_pyfunction!(shots_surfel_stats, m)?,
        wrap_pyfunction!(shots_scan_identities, m)?,
        wrap_pyfunction!(shots_region_lines, m)?,
        wrap_pyfunction!(shots_scan_offsets, m)?,
        wrap_pyfunction!(shots_link_ends, m)?,
        wrap_pyfunction!(shots_bundles, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
