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

/// Amodal completion in one scan's range image (sylva_rs::shots::amodal), for 3dtrees research 128's amodal spike.
/// Takes the pulses as `shots_scan_surfels` does; stripe ends `end_p`, `end_d` (outward), `end_r`; candidate pairs
/// `pair_a`, `pair_b` (indices into the ends); 3-D Hermite paths `h_pa`, `h_ta`, `h_pb`, `h_tb` (m, 3), `h_ra`,
/// `h_rb`; cylinders `cyl_*` for the wood-only confirmation, with this scan's surfels `wood_p`, `wood_n`.
/// Returns per end `e_class` (0 occluded 1 real 2 ambiguous 3 none), `e_near`, `e_far`, `e_empty`, `e_same`,
/// `e_l`, `e_c`, `e_rho`, `e_sl`, `e_sc` (rad per line, per column), `e_ul`, `e_uc`, `e_fore`; per pair `p_ok`, `p_gap_cells`, `p_mis_a`, `p_mis_b` (deg),
/// `p_straight`, `p_hermite` (n, 4: support refute occluder unknown); per 3-D path `h_count` (m, 4); per
/// cylinder `c_support`, `c_wood`, `c_inf`; the wood node count and the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, rot, origin, end_p, end_d, end_r, pair_a, pair_b, h_pa, h_ta, h_pb, h_tb, h_ra, h_rb, cyl_p, cyl_u, cyl_half, cyl_r, wood_p, wood_n, margin=0.05, window=4, tol=0.03, cyl_tol=0.015, max_cells=60.0))]
#[allow(clippy::too_many_arguments)]
fn shots_amodal_scan<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, end_p: PyReadonlyArray2<f64>, end_d: PyReadonlyArray2<f64>, end_r: PyReadonlyArray1<f64>, pair_a: PyReadonlyArray1<u32>, pair_b: PyReadonlyArray1<u32>, h_pa: PyReadonlyArray2<f64>, h_ta: PyReadonlyArray2<f64>, h_pb: PyReadonlyArray2<f64>, h_tb: PyReadonlyArray2<f64>, h_ra: PyReadonlyArray1<f64>, h_rb: PyReadonlyArray1<f64>, cyl_p: PyReadonlyArray2<f64>, cyl_u: PyReadonlyArray2<f64>, cyl_half: PyReadonlyArray1<f64>, cyl_r: PyReadonlyArray1<f64>, wood_p: PyReadonlyArray2<f64>, wood_n: PyReadonlyArray2<f64>, margin: f64, window: usize, tol: f64, cyl_tol: f64, max_cells: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::amodal::{image_ends, image_pairs, score_path, wood_keys, wood_support, world_path, AmodalParams, PathCount};
    use sylva_rs::shots::opacity::{Cyl, ShadowMap};
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    if t.len() != dir.len() || r.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("per-pulse arrays differ in length, or rot isn't 3 x 3"));
    }
    let origin = point3(origin, "origin")?;
    let rot = [rot[0], rot[1], rot[2]];
    let (ep, ed, er) = (xyz_from_py(end_p)?, xyz_from_py(end_d)?, end_r.as_array().to_vec());
    let ends: Vec<(sylva_rs::Point, sylva_rs::Point, f64)> = (0..ep.len()).map(|i| (ep[i], ed[i], er[i])).collect();
    let pairs: Vec<(usize, usize)> = pair_a.as_array().iter().zip(pair_b.as_array().iter()).map(|(&a, &b)| (a as usize, b as usize)).collect();
    if pairs.iter().any(|&(a, b)| a >= ends.len() || b >= ends.len()) {
        return Err(PyValueError::new_err("a pair names an end that doesn't exist"));
    }
    let (hpa, hta, hpb, htb) = (xyz_from_py(h_pa)?, xyz_from_py(h_ta)?, xyz_from_py(h_pb)?, xyz_from_py(h_tb)?);
    let (hra, hrb) = (h_ra.as_array().to_vec(), h_rb.as_array().to_vec());
    let (cp, cu) = (xyz_from_py(cyl_p)?, xyz_from_py(cyl_u)?);
    let (ch, cr) = (cyl_half.as_array().to_vec(), cyl_r.as_array().to_vec());
    let cyls: Vec<Cyl> = (0..cp.len()).map(|i| Cyl { p: cp[i], u: cu[i], half: ch[i], r: cr[i] }).collect();
    let (wp, wn) = (xyz_from_py(wood_p)?, xyz_from_py(wood_n)?);
    let prm = AmodalParams { margin, window, tol, ..AmodalParams::default() };
    let (ie, ip, hc, cs, n_wood, secs) = py.detach(|| {
        use rayon::prelude::*;
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let map = ShadowMap::new(&lat, &r, rot, origin);
        let c1 = std::time::Instant::now();
        let ie = image_ends(&map, &ends, &prm);
        let ip = image_pairs(&map, &ie, &pairs, &prm);
        let hc: Vec<PathCount> = (0..hpa.len()).into_par_iter().map(|i| score_path(&map, &world_path(&map, &hpa[i], &hta[i], &hpb[i], &htb[i], hra[i], hrb[i]), tol)).collect();
        let c2 = std::time::Instant::now();
        let wood = wood_keys(&wp, &wn, 10.0, 0.5, 0.3);
        let beam = sylva_rs::shots::stripes::StripeParams::default().beam;
        let cs = wood_support(&map, &dir, &cyls, &wood, |x| beam.footprint(x), cyl_tol, max_cells);
        let c3 = std::time::Instant::now();
        (ie, ip, hc, cs, wood.len(), [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), (c3 - c2).as_secs_f64()])
    });
    let d = PyDict::new(py);
    let nan = f64::NAN;
    macro_rules! col {
        ($v:expr, $name:literal, $f:expr) => {
            d.set_item($name, $v.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!(ie, "e_class", |x| x.map_or(3u8, |e| e.label.class));
    col!(ie, "e_near", |x| x.map_or(0u32, |e| e.label.near));
    col!(ie, "e_far", |x| x.map_or(0u32, |e| e.label.far));
    col!(ie, "e_empty", |x| x.map_or(0u32, |e| e.label.empty));
    col!(ie, "e_same", |x| x.map_or(0u32, |e| e.label.same));
    col!(ie, "e_l", |x| x.map_or(nan, |e| e.at.l));
    col!(ie, "e_c", |x| x.map_or(nan, |e| e.at.c));
    col!(ie, "e_rho", |x| x.map_or(nan, |e| e.at.rho));
    col!(ie, "e_sl", |x| x.map_or(nan, |e| e.at.sl));
    col!(ie, "e_sc", |x| x.map_or(nan, |e| e.at.sc));
    col!(ie, "e_ul", |x| x.map_or(nan, |e| e.u[0]));
    col!(ie, "e_uc", |x| x.map_or(nan, |e| e.u[1]));
    col!(ie, "e_fore", |x| x.map_or(nan, |e| e.fore));
    col!(ip, "p_ok", |x| x.is_some());
    col!(ip, "p_gap_cells", |x| x.map_or(nan, |p| p.0));
    col!(ip, "p_mis_a", |x| x.map_or(nan, |p| p.1));
    col!(ip, "p_mis_b", |x| x.map_or(nan, |p| p.2));
    let counts = |v: Vec<PathCount>| -> PyResult<Bound<'py, PyArray2<u32>>> {
        let n = v.len();
        Ok(v.into_iter().flat_map(|c| [c.support, c.refute, c.occluder, c.unknown]).collect::<Vec<u32>>().into_pyarray(py).reshape([n, 4])?)
    };
    d.set_item("p_straight", counts(ip.iter().map(|x| x.map_or(PathCount::default(), |p| p.3)).collect())?)?;
    d.set_item("p_hermite", counts(ip.iter().map(|x| x.map_or(PathCount::default(), |p| p.4)).collect())?)?;
    d.set_item("h_count", counts(hc)?)?;
    col!(cs, "c_support", |x| x[0]);
    col!(cs, "c_wood", |x| x[1]);
    col!(cs, "c_inf", |x| x[2]);
    d.set_item("n_wood_nodes", n_wood)?;
    for (k, v) in ["lattice_s", "image_s", "wood_s"].iter().zip(secs) {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

/// The moment octree of every scan but `exclude` (sylva_rs::shots::amodal::lift), from surfels `p`, `n` of scans
/// `scan`, queried at `q` with a 9 cm neighbourhood. Returns per point `n`, `ratio`, `r`, `lam3`, `u` (NaN where
/// fewer than `min_n` surfels) and `secs`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, exclude, q, min_n=10.0, r_max=0.3))]
#[allow(clippy::too_many_arguments)]
fn shots_amodal_lift<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, exclude: i32, q: PyReadonlyArray2<f64>, min_n: f64, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    let (p, nn, scan, q) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec(), xyz_from_py(q)?);
    if nn.len() != p.len() || scan.len() != p.len() {
        return Err(PyValueError::new_err("surfel arrays differ in length"));
    }
    let (o, secs) = py.detach(|| {
        let t0 = std::time::Instant::now();
        let o = sylva_rs::shots::amodal::lift(&p, &nn, &scan, exclude, &q, min_n, r_max);
        (o, t0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    d.set_item("n", o.iter().map(|x| x.n).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("ratio", o.iter().map(|x| x.ratio).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("r", o.iter().map(|x| x.r).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("lam3", o.iter().map(|x| x.lam3).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("u", xyz_to_py(py, &o.iter().map(|x| x.u).collect::<Vec<_>>()))?;
    d.set_item("secs", secs)?;
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
/// unknown, `s_chord`, `s_miss`, `s_t_in`, `s_depth`, `s_fp`, `s_sin`, `s_excess`, `s_m` (k, 3, world), `s_pulse` (the cell's pulse, u32 max if empty); with
/// `margin` > 0, the widened gate's possible refutes only, as opacity's docs say); the lattice round trip
/// on every 997th pulse (`roundtrip_line`, `roundtrip_col`: cell_of minus the
/// cell); optionally (`image`) every pulse's `pulse_line`, `pulse_col`; and
/// the seconds each step took.
#[pyfunction]
#[pyo3(signature = (time, direction, range, n_echo, refl, rot, origin, lo, hi, cyl_p, cyl_u, cyl_half, cyl_r, tol=0.015, voxel=0.06, pad=3, window=5, max_step=6.0, max_cells=60.0, image=false, margin=0.0))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_opacity<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, n_echo: PyReadonlyArray1<u8>, refl: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, cyl_p: PyReadonlyArray2<f64>, cyl_u: PyReadonlyArray2<f64>, cyl_half: PyReadonlyArray1<f64>, cyl_r: PyReadonlyArray1<f64>, tol: f64, voxel: f64, pad: i64, window: usize, max_step: f64, max_cells: f64, image: bool, margin: f64) -> PyResult<Bound<'py, PyDict>> {
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
        let samples = map.test(&cyls, |x| beam.footprint(x), tol, max_cells, margin);
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
    col!(samples, "s_fp", |x| x.fp as f32);
    col!(samples, "s_sin", |x| x.sin as f32);
    col!(samples, "s_excess", |x| x.excess as f32);
    col!(samples, "s_pulse", |x| x.pulse);
    d.set_item("s_m", xyz_to_py(py, &samples.iter().map(|x| x.m).collect::<Vec<_>>()))?;
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
/// `harm` (k, 5; with what bit 32 a convex section), `harm_gap` (rad), `harm_rms` (m), `harm_n`; `attr` (k_regions, k);
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

/// The completion field's walk fit (sylva_rs::shots::completion): slice axis points `p`, directions `u` (k, 3) and
/// fragment ids `frag`. Returns the pairs' `ds`, `theta`, and the Brownian (`b_`) and kinked (`k_`) fits' `d`,
/// `sigma`, `lambda`, `loglik`.
#[pyfunction]
#[pyo3(signature = (p, u, frag, min_sep=0.02, max_sep=0.5, kappa=std::f64::consts::FRAC_PI_2))]
fn shots_completion_fit<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>, frag: PyReadonlyArray1<i64>, min_sep: f64, max_sep: f64, kappa: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::completion::{fit_walk, tangent_pairs};
    let (p, u, f) = (xyz_from_py(p)?, xyz_from_py(u)?, frag.as_array().to_vec());
    if u.len() != p.len() || f.len() != p.len() {
        return Err(PyValueError::new_err("slice arrays differ in length"));
    }
    let (pairs, fb, fk) = py.detach(|| {
        let pairs = tangent_pairs(&p, &u, &f, min_sep, max_sep);
        let fb = fit_walk(&pairs, false, kappa);
        let fk = fit_walk(&pairs, true, kappa);
        (pairs, fb, fk)
    });
    let d = PyDict::new(py);
    d.set_item("ds", pairs.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("theta", pairs.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    for (pre, w) in [("b_", fb), ("k_", fk)] {
        d.set_item(format!("{pre}d"), w.d)?;
        d.set_item(format!("{pre}sigma"), w.sigma)?;
        d.set_item(format!("{pre}lambda"), w.lambda)?;
        d.set_item(format!("{pre}loglik"), w.loglik)?;
    }
    Ok(d)
}

fn completion_ends(p: PyReadonlyArray2<f64>, d: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, sig_r: Vec<f64>) -> PyResult<Vec<sylva_rs::shots::completion::End>> {
    let (p, dd, r) = (xyz_from_py(p)?, xyz_from_py(d)?, r.as_array().to_vec());
    if dd.len() != p.len() || r.len() != p.len() || sig_r.len() != p.len() {
        return Err(PyValueError::new_err("end arrays differ in length"));
    }
    Ok((0..p.len()).map(|i| sylva_rs::shots::completion::End { p: p[i], d: dd[i], r: r[i], sig_r: sig_r[i] }).collect())
}

/// Each candidate link's Hermite path as short cylinders: ends `p`, `d` (k, 3), `r`, pairs `a`, `b`; about `seg`
/// metres a cylinder, at most `max_seg`. Returns `p`, `u`, `half`, `r` and each cylinder's `pair`.
#[pyfunction]
#[pyo3(signature = (p, d, r, a, b, seg=0.05, max_seg=20))]
#[allow(clippy::too_many_arguments)]
fn shots_completion_paths<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, d: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, a: PyReadonlyArray1<u32>, b: PyReadonlyArray1<u32>, seg: f64, max_seg: usize) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::completion::hermite_cyls;
    let n_ends = r.as_array().len();
    let ends = completion_ends(p, d, r, vec![0.0; n_ends])?;
    let (a, b) = (a.as_array().to_vec(), b.as_array().to_vec());
    let (cyl, pair) = py.detach(|| {
        let mut cyl = Vec::new();
        let mut pair = Vec::new();
        for k in 0..a.len() {
            let (ea, eb) = (&ends[a[k] as usize], &ends[b[k] as usize]);
            let l = ((0..3).map(|i| (eb.p[i] - ea.p[i]).powi(2)).sum::<f64>()).sqrt();
            let n = ((l / seg).ceil() as usize).clamp(1, max_seg);
            for c in hermite_cyls(ea, eb, n) {
                cyl.push(c);
                pair.push(k as u32);
            }
        }
        (cyl, pair)
    });
    let out = PyDict::new(py);
    out.set_item("p", xyz_to_py(py, &cyl.iter().map(|c| c.p).collect::<Vec<_>>()))?;
    out.set_item("u", xyz_to_py(py, &cyl.iter().map(|c| c.u).collect::<Vec<_>>()))?;
    out.set_item("half", cyl.iter().map(|c| c.half).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("r", cyl.iter().map(|c| c.r).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("pair", pair.into_pyarray(py))?;
    Ok(out)
}

/// Each candidate link's prior log weight under the completion field: ends `p`, `d`, `r`, `sig_r`, pairs `a`, `b`.
/// Returns `log_dir` (the walk) and `log_rad` (0 with `radius=False`).
#[pyfunction]
#[pyo3(signature = (p, d, r, sig_r, a, b, diff, lam, kappa, sig_dir, sig_pos, taper=0.15, q=0.04, radius=true))]
#[allow(clippy::too_many_arguments)]
fn shots_completion_prior<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, d: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, sig_r: PyReadonlyArray1<f64>, a: PyReadonlyArray1<u32>, b: PyReadonlyArray1<u32>, diff: f64, lam: f64, kappa: f64, sig_dir: f64, sig_pos: f64, taper: f64, q: f64, radius: bool) -> PyResult<Bound<'py, PyDict>> {
    use rayon::prelude::*;
    use sylva_rs::shots::completion::{log_prior, Field};
    let ends = completion_ends(p, d, r, sig_r.as_array().to_vec())?;
    let (a, b) = (a.as_array().to_vec(), b.as_array().to_vec());
    let f = Field { d: diff, lambda: lam, kappa, sig_dir, sig_pos, taper, q, radius };
    let lp: Vec<(f64, f64)> = py.detach(|| (0..a.len()).into_par_iter().map(|k| log_prior(&ends[a[k] as usize], &ends[b[k] as usize], &f)).collect());
    let out = PyDict::new(py);
    out.set_item("log_dir", lp.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("log_rad", lp.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

/// Per-end normalisation of link log weights `logw` over pairs `a`, `b` of `n_ends` ends, with a null option of
/// weight `log_null`. Returns per pair `pa` (a's share on b), `pb`, and per end `null` and `entropy` (nats).
#[pyfunction]
fn shots_completion_normalise<'py>(py: Python<'py>, n_ends: usize, a: PyReadonlyArray1<u32>, b: PyReadonlyArray1<u32>, logw: PyReadonlyArray1<f64>, log_null: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::completion::normalise;
    let (a, b, w) = (a.as_array().to_vec(), b.as_array().to_vec(), logw.as_array().to_vec());
    if b.len() != a.len() || w.len() != a.len() || a.iter().chain(&b).any(|&e| e as usize >= n_ends) {
        return Err(PyValueError::new_err("pair arrays differ in length or name a missing end"));
    }
    let (pairs, ends) = py.detach(|| normalise(n_ends, &a, &b, &w, log_null));
    let out = PyDict::new(py);
    out.set_item("pa", pairs.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("pb", pairs.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("null", ends.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("entropy", ends.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
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

/// Bundles by line consistency (sylva_rs::shots::strands::bundles_lines): pieces as `shots_bundles`, plus `origin`
/// (k, 3) each piece's scanner position. Returns `bundle` per piece; per stripe line `line_u`, `line_c`, `line_t0`,
/// `line_t1`, `line_r`, `line_beam`, `line_piece`; and `edges`, `merged`, `blocked`, `cannot`.
#[pyfunction]
#[pyo3(signature = (p, u, half, r, scan, stripe, line, col, range, width, origin, k_beam=0.5, sd_beam_min=0.005, sd_across=0.003, sd_floor=0.004, max_ang=20.0, max_dlogr=0.7, max_gap=0.05, chi2=9.21, reach=0.06, near_cells=3.0, depth_step=0.1))]
#[allow(clippy::too_many_arguments)]
fn shots_bundles_lines<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>, half: PyReadonlyArray1<f64>, r: PyReadonlyArray1<f64>, scan: PyReadonlyArray1<i32>, stripe: PyReadonlyArray1<u32>, line: PyReadonlyArray1<f64>, col: PyReadonlyArray1<f64>, range: PyReadonlyArray1<f64>, width: PyReadonlyArray1<f64>, origin: PyReadonlyArray2<f64>, k_beam: f64, sd_beam_min: f64, sd_across: f64, sd_floor: f64, max_ang: f64, max_dlogr: f64, max_gap: f64, chi2: f64, reach: f64, near_cells: f64, depth_step: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::strands::{bundles_lines, LineParams, Piece};
    let (p, u, o) = (xyz_from_py(p)?, xyz_from_py(u)?, xyz_from_py(origin)?);
    let (h, r, s, st) = (half.as_array().to_vec(), r.as_array().to_vec(), scan.as_array().to_vec(), stripe.as_array().to_vec());
    let (l, c, rg, w) = (line.as_array().to_vec(), col.as_array().to_vec(), range.as_array().to_vec(), width.as_array().to_vec());
    let n = p.len();
    if [u.len(), h.len(), r.len(), s.len(), st.len(), l.len(), c.len(), rg.len(), w.len(), o.len()].iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("piece arrays differ in length"));
    }
    let pc: Vec<Piece> = (0..n).map(|i| Piece { p: p[i], u: u[i], half: h[i], r: r[i], scan: s[i], stripe: st[i], line: l[i], col: c[i], range: rg[i] }).collect();
    let prm = LineParams { k_beam, sd_beam_min, sd_across, sd_floor, max_ang, max_dlogr, max_gap, chi2 };
    let (b, ms, lines) = py.detach(|| bundles_lines(&pc, &w, &o, &prm, reach, near_cells, depth_step));
    let d = PyDict::new(py);
    let p3 = |x: &nalgebra::Vector3<f64>| [x[0], x[1], x[2]];
    d.set_item("bundle", b.into_pyarray(py))?;
    d.set_item("line_u", xyz_to_py(py, &lines.iter().map(|x| p3(&x.u)).collect::<Vec<_>>()))?;
    d.set_item("line_c", xyz_to_py(py, &lines.iter().map(|x| p3(&x.c)).collect::<Vec<_>>()))?;
    d.set_item("line_beam", xyz_to_py(py, &lines.iter().map(|x| p3(&x.beam)).collect::<Vec<_>>()))?;
    d.set_item("line_t0", lines.iter().map(|x| x.t0).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("line_t1", lines.iter().map(|x| x.t1).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("line_r", lines.iter().map(|x| x.r).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("line_piece", lines.iter().map(|x| x.piece as u64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("edges", ms.edges)?;
    d.set_item("merged", ms.merged)?;
    d.set_item("blocked", ms.blocked)?;
    d.set_item("cannot", ms.cannot)?;
    Ok(d)
}

/// Per region and scan, offsets and radial distances against given lines (sylva_rs::shots::strands::view_offsets):
/// surfels `p`, `n`, `scan`; regions `reg_p`, `reg_u`, `reg_half`, `reg_r`; line points `c`, radii `r`; `shift`
/// (k, n_scans, 3) per region and scan (in `scans`' order), subtracted from that scan's surfels. Returns `region`,
/// `scan`, `offset` (m, 3), `count`, `radial`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, reg_p, reg_u, reg_half, reg_r, c, r, shift, scans, margin=0.02, min_n=8))]
#[allow(clippy::too_many_arguments)]
fn shots_view_offsets<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, reg_p: PyReadonlyArray2<f64>, reg_u: PyReadonlyArray2<f64>, reg_half: PyReadonlyArray1<f64>, reg_r: PyReadonlyArray1<f64>, c: PyReadonlyArray2<f64>, r: PyReadonlyArray1<f64>, shift: numpy::PyReadonlyArray3<f64>, scans: Vec<i32>, margin: f64, min_n: usize) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::strands::view_offsets;
    use sylva_rs::shots::surfels::Region;
    let (p, n, scan) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec());
    let (rp, ru, cc) = (xyz_from_py(reg_p)?, xyz_from_py(reg_u)?, xyz_from_py(c)?);
    let (rh, rr, r) = (reg_half.as_array().to_vec(), reg_r.as_array().to_vec(), r.as_array().to_vec());
    let sh = shift.as_array();
    if sh.shape() != [rp.len(), scans.len(), 3] || n.len() != p.len() || scan.len() != p.len() || ru.len() != rp.len() || cc.len() != rp.len() || r.len() != rp.len() {
        return Err(PyValueError::new_err("array shapes differ"));
    }
    let shift: Vec<Vec<[f64; 3]>> = (0..rp.len()).map(|k| (0..scans.len()).map(|j| [sh[[k, j, 0]], sh[[k, j, 1]], sh[[k, j, 2]]]).collect()).collect();
    let regs: Vec<Region> = (0..rp.len()).map(|i| Region { p: rp[i], u: ru[i], half: rh[i], r: rr[i], scan: -1 }).collect();
    let o = py.detach(|| view_offsets(&p, &n, &scan, &regs, &cc, &r, &shift, &scans, margin, min_n));
    let d = PyDict::new(py);
    d.set_item("region", o.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("scan", o.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("offset", xyz_to_py(py, &o.iter().map(|x| x.2).collect::<Vec<_>>()))?;
    d.set_item("count", o.iter().map(|x| x.3).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("radial", o.iter().map(|x| x.4).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

/// Surface elements about query points (sylva_rs::shots::elements::elements): surfels `p`, `n`, `scan`, `attr` (k, m);
/// points `c`. Returns `n`, `n_scans`, `lam` (q, 3), `normal` (q, 3), `extent` (q, 2), `thick`, `attr` (q, k), `secs`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, attr, c, rad=0.04, min_n=5))]
#[allow(clippy::too_many_arguments)]
fn shots_elements<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, attr: PyReadonlyArray2<f64>, c: PyReadonlyArray2<f64>, rad: f64, min_n: usize) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray2;
    use sylva_rs::shots::elements::elements;
    let (p, n, scan, c) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec(), xyz_from_py(c)?);
    let attr: Vec<Vec<f64>> = attr.as_array().rows().into_iter().map(|r| r.to_vec()).collect();
    if n.len() != p.len() || scan.len() != p.len() || attr.iter().any(|a| a.len() != p.len()) {
        return Err(PyValueError::new_err("surfel arrays differ in length"));
    }
    let (e, secs) = py.detach(|| {
        let t = std::time::Instant::now();
        let e = elements(&p, &n, &scan, &attr, &c, rad, min_n);
        (e, t.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    d.set_item("n", e.iter().map(|x| x.n).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n_scans", e.iter().map(|x| x.n_scans).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("lam", xyz_to_py(py, &e.iter().map(|x| x.lam).collect::<Vec<_>>()))?;
    d.set_item("normal", xyz_to_py(py, &e.iter().map(|x| x.normal).collect::<Vec<_>>()))?;
    d.set_item("extent", PyArray2::from_vec2(py, &e.iter().map(|x| x.extent.to_vec()).collect::<Vec<_>>())?)?;
    d.set_item("thick", e.iter().map(|x| x.thick).collect::<Vec<_>>().into_pyarray(py))?;
    if !attr.is_empty() {
        d.set_item("attr", PyArray2::from_vec2(py, &e.iter().map(|x| x.attr.clone()).collect::<Vec<_>>())?)?;
    }
    d.set_item("secs", secs)?;
    Ok(d)
}

/// One scan's surfels for the moment octree (3dtrees research 128 `octree.md`): the lattice, then a normal per kept
/// first echo, `method` "grad" (n ∝ ω − ∇ ln r over `half` cells either way, sylva_rs::shots::identities::
/// grad_normals; only those spanning both directions) or "plane" (sylva_rs::shots::surfels). A pulse is kept when
/// `keep` (per pulse, optional) is true and its point is inside `lo`..`hi` (optional). Returns `pulse`, world `p`,
/// `n`, `cos_inc`; `tested` (kept cells); `lattice_s`, `map_s`, `stencil_s` (wall seconds).
#[pyfunction]
#[pyo3(signature = (time, direction, range, rot, origin, keep=None, lo=None, hi=None, method="grad", half=3, jump=0.05))]
#[allow(clippy::too_many_arguments)]
fn shots_scan_normals<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, keep: Option<PyReadonlyArray1<bool>>, lo: Option<Vec<f64>>, hi: Option<Vec<f64>>, method: &str, half: i64, jump: f64) -> PyResult<Bound<'py, PyDict>> {
    use sylva_rs::shots::identities::grad_normals;
    use sylva_rs::shots::opacity::ShadowMap;
    use sylva_rs::shots::surfels::{surfels, SurfelParams};
    let dir = xyz_from_py(direction)?;
    let t = time.as_array().to_vec();
    let r = range.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    let mask: Option<Vec<bool>> = keep.map(|k| k.as_array().to_vec());
    if t.len() != dir.len() || r.len() != dir.len() || rot.len() != 3 || mask.as_ref().is_some_and(|m| m.len() != dir.len()) {
        return Err(PyValueError::new_err("per-pulse arrays differ in length, or rot isn't 3 x 3"));
    }
    let origin = point3(origin, "origin")?;
    let lo = lo.map(|x| point3(x, "lo")).transpose()?.unwrap_or([f64::NEG_INFINITY; 3]);
    let hi = hi.map(|x| point3(x, "hi")).transpose()?.unwrap_or([f64::INFINITY; 3]);
    let rot = [rot[0], rot[1], rot[2]];
    let grad = match method {
        "grad" => true,
        "plane" => false,
        _ => return Err(PyValueError::new_err("method is 'grad' or 'plane'")),
    };
    let (pulse, p, n, cos, tested, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let lat = sylva_rs::shots::lattice::scan_lattice(&t, &dir);
        let c1 = std::time::Instant::now();
        let map = ShadowMap::new(&lat, &r, rot, origin);
        let c2 = std::time::Instant::now();
        let tested = std::sync::atomic::AtomicUsize::new(0);
        let keep = |i: usize, x: &[f64; 3]| {
            let ok = mask.as_ref().is_none_or(|m| m[i]) && (0..3).all(|k| x[k] >= lo[k] && x[k] <= hi[k]);
            if ok {
                tested.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ok
        };
        let world = |i: usize| {
            let x = [dir[i][0] * r[i] as f64, dir[i][1] * r[i] as f64, dir[i][2] * r[i] as f64];
            [0, 1, 2].map(|a| rot[a][0] * x[0] + rot[a][1] * x[1] + rot[a][2] * x[2] + origin[a])
        };
        let out: Vec<(u32, [f64; 3], [f64; 3], f64)> = if grad {
            grad_normals(&map, &dir, keep, half, jump, 0.2).into_iter().filter(|g| g.full).map(|g| (g.pulse, world(g.pulse as usize), g.n, g.cos_inc)).collect()
        } else {
            let beam = sylva_rs::qsm::joint::JointParams::default();
            let prm = SurfelParams { half, jump, ..SurfelParams::default() };
            surfels(&map, &dir, keep, |x| beam.footprint(x), &prm).0.into_iter().map(|s| (s.pulse, s.p, s.n, s.cos_inc)).collect()
        };
        let c3 = std::time::Instant::now();
        let secs = [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), (c3 - c2).as_secs_f64()];
        let pulse: Vec<u32> = out.iter().map(|x| x.0).collect();
        let p: Vec<[f64; 3]> = out.iter().map(|x| x.1).collect();
        let n: Vec<[f64; 3]> = out.iter().map(|x| x.2).collect();
        let cos: Vec<f64> = out.iter().map(|x| x.3).collect();
        (pulse, p, n, cos, tested.into_inner(), secs)
    });
    let d = PyDict::new(py);
    d.set_item("pulse", pulse.into_pyarray(py))?;
    d.set_item("p", xyz_to_py(py, &p))?;
    d.set_item("n", xyz_to_py(py, &n))?;
    d.set_item("cos_inc", cos.into_pyarray(py))?;
    d.set_item("tested", tested)?;
    for (k, name) in ["lattice_s", "map_s", "stencil_s"].iter().enumerate() {
        d.set_item(*name, secs[k])?;
    }
    Ok(d)
}

/// The moment octree of surfels `p`, `n` (m, 3) from scans `scan` (sylva_rs::shots::moments): per-(voxel, scan)
/// moments on `voxel` cells, `levels` levels, then per query point `q` (k, 3) and level the node holding it (with
/// `nbhd`, it and its 26 neighbours summed). Returns, each (levels, k) or flat (levels * k * 3): `count`, `n_scans`;
/// the normal tensor's `lam` and null direction `t_u`, with medial collapse about it `t_r`, `t_spread`, `spread0`;
/// the line-complex axis (cylinder rows scaled by `s`) `lc_u`, `lc_c`, `lc_lam`, `lc_lam2`, and collapse about it
/// `lc_r`, `lc_spread`, `lc_spread0`; with `q_u` (k, 3), collapse about it `g_r`, `g_spread`, `g_spread0`. Also
/// `nodes` (per level), `voxel_scans` (per-(voxel, scan) entries), `bytes_per_node`, `reduce_s`, `octree_s`, `query_s`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, q, q_u=None, voxel=0.03, levels=4, s=0.01, nbhd=false, r_max=0.3))]
#[allow(clippy::too_many_arguments)]
fn shots_moment_octree<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, q: PyReadonlyArray2<f64>, q_u: Option<PyReadonlyArray2<f64>>, voxel: f64, levels: usize, s: f64, nbhd: bool, r_max: f64) -> PyResult<Bound<'py, PyDict>> {
    use nalgebra::Vector3;
    use rayon::prelude::*;
    use sylva_rs::shots::moments::{per_scan, Mom, Octree};
    type V3 = Vector3<f64>;
    let vv = |x: Vec<[f64; 3]>| x.into_iter().map(|a| V3::new(a[0], a[1], a[2])).collect::<Vec<_>>();
    let (p, n, scan, q) = (vv(xyz_from_py(p)?), vv(xyz_from_py(n)?), scan.as_array().to_vec(), vv(xyz_from_py(q)?));
    let qu = q_u.map(|x| xyz_from_py(x).map(vv)).transpose()?;
    if n.len() != p.len() || scan.len() != p.len() || qu.as_ref().is_some_and(|u| u.len() != q.len()) || levels == 0 {
        return Err(PyValueError::new_err("surfel or query arrays differ in length"));
    }
    const NAN: f64 = f64::NAN;
    let a3 = |x: &V3| [x[0], x[1], x[2]];
    // per (level, query): count, n_scans, t_r, t_spread, spread0, lc_lam, lc_lam2, lc_r, lc_spread, g_r, g_spread,
    // lc_spread0, g_spread0;
    // and lam, t_u, lc_u, lc_c
    type Row = ([f64; 13], [[f64; 3]; 4]);
    let (rows, nodes, nvs, secs) = py.detach(|| {
        let c0 = std::time::Instant::now();
        let ps = per_scan(&p, &n, &scan, voxel);
        let c1 = std::time::Instant::now();
        let o = Octree::new(&ps, voxel, levels);
        let c2 = std::time::Instant::now();
        let rows: Vec<Row> = (0..levels * q.len())
            .into_par_iter()
            .map(|j| {
                let (l, k) = (j / q.len(), j % q.len());
                let (m, ns): (Mom, u32) = o.query(l, &q[k], nbhd);
                let mut f = [NAN; 13];
                let mut x = [[NAN; 3]; 4];
                f[0] = m.n;
                f[1] = ns as f64;
                if m.n < 6.0 {
                    return (f, x);
                }
                let (lam, u) = m.tensor();
                x[0] = lam;
                x[1] = a3(&u);
                (f[2], f[3], f[4]) = m.collapse(&u, r_max);
                // the line-complex axis, coordinates about the node's centroid
                if let Some((lu, lc, l1, l2)) = m.axis(&(m.sp / m.n), s) {
                    let (r, sp, s0) = m.collapse(&lu, r_max);
                    (f[5], f[6], f[7], f[8], f[11]) = (l1, l2, r, sp, s0);
                    (x[2], x[3]) = (a3(&lu), a3(&lc));
                }
                if let Some(qu) = &qu {
                    let (r, sp, s0) = m.collapse(&qu[k].normalize(), r_max);
                    (f[9], f[10], f[12]) = (r, sp, s0);
                }
                (f, x)
            })
            .collect();
        let c3 = std::time::Instant::now();
        let nodes: Vec<usize> = o.levels.iter().map(|l| l.len()).collect();
        (rows, nodes, ps.len(), [(c1 - c0).as_secs_f64(), (c2 - c1).as_secs_f64(), (c3 - c2).as_secs_f64()])
    });
    let k = q.len();
    let d = PyDict::new(py);
    let names = ["count", "n_scans", "t_r", "t_spread", "spread0", "lc_lam", "lc_lam2", "lc_r", "lc_spread", "g_r", "g_spread", "lc_spread0", "g_spread0"];
    for (j, name) in names.iter().enumerate() {
        let g: Vec<Vec<f64>> = (0..levels).map(|l| rows[l * k..(l + 1) * k].iter().map(|r| r.0[j]).collect()).collect();
        d.set_item(*name, PyArray2::from_vec2(py, &g)?)?;
    }
    for (j, name) in ["lam", "t_u", "lc_u", "lc_c"].iter().enumerate() {
        d.set_item(*name, rows.iter().flat_map(|r| r.1[j]).collect::<Vec<_>>().into_pyarray(py))?;
    }
    d.set_item("nodes", nodes)?;
    d.set_item("voxel_scans", nvs)?;
    d.set_item("bytes_per_node", std::mem::size_of::<Mom>())?;
    for (j, name) in ["reduce_s", "octree_s", "query_s"].iter().enumerate() {
        d.set_item(*name, secs[j])?;
    }
    Ok(d)
}

fn field_grid(lo: Vec<f64>, h: f64, n: Vec<usize>) -> PyResult<sylva_rs::shots::fields::Grid> {
    let n: [usize; 3] = n.try_into().map_err(|_| PyValueError::new_err("n needs 3 values"))?;
    if !(h > 0.0) {
        return Err(PyValueError::new_err("h must be positive"));
    }
    Ok(sylva_rs::shots::fields::Grid { lo: point3(lo, "lo")?, h, n })
}

/// One scan's ray counts on the voxel box `lo`, `h`, `dims` (sylva_rs::shots::fields::ray_counts): pulse directions
/// `direction` (m, 3, scanner frame; world = `rot` . d), first-echo `range` (NaN or 0: no return). Returns `counts`
/// (voxels, 4: hit, pass, occluded, no return; x fastest), `pass_t` (voxels, 6: the pass rays' Σ ωωᵀ, xx yy zz xy
/// xz yz) and `secs`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn shots_field_counts<'py>(py: Python<'py>, direction: PyReadonlyArray2<f64>, range: PyReadonlyArray1<f32>, rot: PyReadonlyArray2<f64>, origin: Vec<f64>, lo: Vec<f64>, h: f64, dims: Vec<usize>) -> PyResult<Bound<'py, PyDict>> {
    let g = field_grid(lo, h, dims)?;
    let dir = xyz_from_py(direction)?;
    let r = range.as_array().to_vec();
    let rot = xyz_from_py(rot)?;
    if r.len() != dir.len() || rot.len() != 3 {
        return Err(PyValueError::new_err("range and direction differ in length, or rot isn't 3 x 3"));
    }
    let (origin, rot) = (point3(origin, "origin")?, [rot[0], rot[1], rot[2]]);
    let (c, secs) = py.detach(|| {
        let t0 = std::time::Instant::now();
        let c = sylva_rs::shots::fields::ray_counts(&g, &origin, &rot, &dir, &r);
        (c, t0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    let flat: Vec<u32> = c.iter().flat_map(|x| x.0).collect();
    let tens: Vec<f64> = c.iter().flat_map(|x| x.1).collect();
    d.set_item("counts", flat.into_pyarray(py).reshape([g.len(), 4])?)?;
    d.set_item("pass_t", tens.into_pyarray(py).reshape([g.len(), 6])?)?;
    d.set_item("secs", secs)?;
    Ok(d)
}

/// Leaf area and visible wood per voxel from surfels `p`, `n` (m, 3) of scans `scan` on a 3 cm moment octree
/// (sylva_rs::shots::fields::node_inputs). Returns `leaf` (m²), `wood_r` (m, 0 without wood), `wood_n`, the soft
/// wood `soft_sum`, `soft_max`, `soft_t` (voxels, 6) and `secs`.
#[pyfunction]
#[pyo3(signature = (p, n, scan, lo, h, dims, min_n=10.0, wood_ratio=0.5, r_max=0.3, soft_c=0.85, soft_s=0.05))]
#[allow(clippy::too_many_arguments)]
fn shots_field_nodes<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, lo: Vec<f64>, h: f64, dims: Vec<usize>, min_n: f64, wood_ratio: f64, r_max: f64, soft_c: f64, soft_s: f64) -> PyResult<Bound<'py, PyDict>> {
    let g = field_grid(lo, h, dims)?;
    let (p, nn, scan) = (xyz_from_py(p)?, xyz_from_py(n)?, scan.as_array().to_vec());
    if nn.len() != p.len() || scan.len() != p.len() {
        return Err(PyValueError::new_err("surfel arrays differ in length"));
    }
    let (o, secs) = py.detach(|| {
        let t0 = std::time::Instant::now();
        let o = sylva_rs::shots::fields::node_inputs(&g, &p, &nn, &scan, min_n, wood_ratio, r_max, soft_c, soft_s);
        (o, t0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    let nv = g.len();
    d.set_item("leaf", o.leaf.into_pyarray(py))?;
    d.set_item("wood_r", o.wood_r.into_pyarray(py))?;
    d.set_item("wood_n", o.wood_n.into_pyarray(py))?;
    d.set_item("soft_sum", o.soft_sum.into_pyarray(py))?;
    d.set_item("soft_max", o.soft_max.into_pyarray(py))?;
    d.set_item("soft_t", o.soft_t.into_iter().flatten().collect::<Vec<f64>>().into_pyarray(py).reshape([nv, 6])?)?;
    d.set_item("secs", secs)?;
    Ok(d)
}

/// `−∇·(κ∇φ) + σφ = s` on the voxel box (sylva_rs::shots::fields::solve): per voxel `kappa` (m; or 3 per voxel,
/// κ_xx κ_yy κ_zz, voxel-major), `sigma` (1/m), `s`, and `fixed` (NaN free, else Dirichlet). Returns `phi`, `iters`,
/// `residual`, `secs`.
#[pyfunction]
#[pyo3(signature = (kappa, sigma, s, fixed, lo, h, dims, tol=1e-8, max_iter=5000))]
#[allow(clippy::too_many_arguments)]
fn shots_field_solve<'py>(py: Python<'py>, kappa: PyReadonlyArray1<f64>, sigma: PyReadonlyArray1<f64>, s: PyReadonlyArray1<f64>, fixed: PyReadonlyArray1<f64>, lo: Vec<f64>, h: f64, dims: Vec<usize>, tol: f64, max_iter: usize) -> PyResult<Bound<'py, PyDict>> {
    let g = field_grid(lo, h, dims)?;
    let (k, sg, s, f) = (kappa.as_array().to_vec(), sigma.as_array().to_vec(), s.as_array().to_vec(), fixed.as_array().to_vec());
    if [sg.len(), s.len(), f.len()].iter().any(|&x| x != g.len()) || (k.len() != g.len() && k.len() != 3 * g.len()) {
        return Err(PyValueError::new_err("per-voxel arrays must have one value per voxel"));
    }
    let ((phi, it, res), secs) = py.detach(|| {
        let t0 = std::time::Instant::now();
        let o = sylva_rs::shots::fields::solve(&g, &k, &sg, &s, &f, tol, max_iter);
        (o, t0.elapsed().as_secs_f64())
    });
    let d = PyDict::new(py);
    d.set_item("phi", phi.into_pyarray(py))?;
    d.set_item("iters", it)?;
    d.set_item("residual", res)?;
    d.set_item("secs", secs)?;
    Ok(d)
}

/// Per-scan offsets by alternation on moments (sylva_rs::shots::moments::offsets): for each query point `q` (k, 3)
/// the per-scan moments of the level-`level` node holding it (with `nbhd`, it and its 26 neighbours), scans with at
/// least `min_n` surfels there. Returns flat per (query, scan) `query`, `scan`, `t` (m, 3; the shift that brings the
/// scan onto the axis), `medial` (the scan's mean `P(p − R n)` about the axis, count-weighted mean zero: the
/// per-surfel medial offset on the same surfels), `count`; per query the axis `u`, `c` (k, 3; NaN without two scans).
#[pyfunction]
#[pyo3(signature = (p, n, scan, q, level, voxel=0.03, nbhd=true, s=0.01, iters=10, min_n=8))]
#[allow(clippy::too_many_arguments)]
fn shots_moment_offsets<'py>(py: Python<'py>, p: PyReadonlyArray2<f64>, n: PyReadonlyArray2<f64>, scan: PyReadonlyArray1<i32>, q: PyReadonlyArray2<f64>, level: u32, voxel: f64, nbhd: bool, s: f64, iters: usize, min_n: usize) -> PyResult<Bound<'py, PyDict>> {
    use nalgebra::Vector3;
    use rayon::prelude::*;
    use std::collections::{BTreeMap, HashMap};
    use sylva_rs::shots::moments::{key, offsets, per_scan, Key, Mom};
    let vv = |x: Vec<[f64; 3]>| x.into_iter().map(|a| Vector3::new(a[0], a[1], a[2])).collect::<Vec<_>>();
    let (p, n, scan, q) = (vv(xyz_from_py(p)?), vv(xyz_from_py(n)?), scan.as_array().to_vec(), vv(xyz_from_py(q)?));
    let cell = voxel * (1u32 << level) as f64;
    type Fit = (Vec<(i32, [f64; 3], [f64; 3], f64)>, [f64; 3], [f64; 3]);
    let out: Vec<Fit> = py.detach(|| {
        // per-(node, scan) moments at the level, summed from the per-(voxel, scan) ones
        let mut node: HashMap<Key, BTreeMap<i32, Mom>> = HashMap::new();
        for ((k, sc), m) in per_scan(&p, &n, &scan, voxel) {
            node.entry(k.map(|x| x >> level)).or_default().entry(sc).or_insert_with(Mom::zero).merge(&m);
        }
        let r = if nbhd { 1 } else { 0 };
        q.par_iter()
            .map(|x| {
                let k = key(x, cell);
                let mut by: BTreeMap<i32, Mom> = BTreeMap::new();
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            for (sc, m) in node.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).into_iter().flatten() {
                                by.entry(*sc).or_insert_with(Mom::zero).merge(m);
                            }
                        }
                    }
                }
                let sc: Vec<(i32, Mom)> = by.into_iter().filter(|(_, m)| m.n as usize >= min_n).collect();
                let none = (vec![], [f64::NAN; 3], [f64::NAN; 3]);
                if sc.len() < 2 {
                    return none;
                }
                let ms: Vec<Mom> = sc.iter().map(|x| x.1).collect();
                let (sp, cnt) = ms.iter().fold((Vector3::zeros(), 0.0), |a, m| (a.0 + m.sp, a.1 + m.n));
                let Some((u, c, t)) = offsets(&ms, &(sp / cnt), s, iters) else { return none };
                // the medial offsets on the same moments: each scan's mean P(p − R n) about the axis (R from all
                // scans' collapse about it), count-weighted mean zero
                let all = ms.iter().fold(Mom::zero(), |mut a, m| {
                    a.merge(m);
                    a
                });
                let r = all.collapse(&u, 0.3).0;
                let pr = nalgebra::Matrix3::identity() - u * u.transpose();
                let d: Vec<Vector3<f64>> = ms.iter().map(|m| pr * ((m.sp - m.sn * r) / m.n - c)).collect();
                let dm = d.iter().zip(&ms).fold(Vector3::zeros(), |a, (x, m)| a + x * m.n) / cnt;
                let per = sc.iter().zip(t).zip(d).map(|(((s, m), t), d)| (*s, [t[0], t[1], t[2]], [d[0] - dm[0], d[1] - dm[1], d[2] - dm[2]], m.n)).collect();
                (per, [u[0], u[1], u[2]], [c[0], c[1], c[2]])
            })
            .collect()
    });
    let (mut qi, mut si, mut ti, mut di, mut ni) = (vec![], vec![], vec![], vec![], vec![]);
    for (j, (per, _, _)) in out.iter().enumerate() {
        for (s, t, dd, c) in per {
            qi.push(j as i64);
            si.push(*s);
            ti.push(*t);
            di.push(*dd);
            ni.push(*c);
        }
    }
    let d = PyDict::new(py);
    d.set_item("query", qi.into_pyarray(py))?;
    d.set_item("scan", si.into_pyarray(py))?;
    d.set_item("t", xyz_to_py(py, &ti))?;
    d.set_item("medial", xyz_to_py(py, &di))?;
    d.set_item("count", ni.into_pyarray(py))?;
    d.set_item("u", xyz_to_py(py, &out.iter().map(|x| x.1).collect::<Vec<_>>()))?;
    d.set_item("c", xyz_to_py(py, &out.iter().map(|x| x.2).collect::<Vec<_>>()))?;
    Ok(d)
}

/// One evidence term per (hypothesis, scan) (sylva_rs::shots::evidence): the weighted ML beta-binomial of `k`
/// refuting cells out of `n` informative ones (`binomial` holds rho at 0, per-cell counting). Returns `mu`, `rho`,
/// `loglik`.
#[pyfunction]
#[pyo3(signature = (k, n, w, binomial=false))]
fn shots_evidence_fit<'py>(py: Python<'py>, k: PyReadonlyArray1<u32>, n: PyReadonlyArray1<u32>, w: PyReadonlyArray1<f64>, binomial: bool) -> PyResult<Bound<'py, PyDict>> {
    let (k, n, w) = (k.as_array().to_vec(), n.as_array().to_vec(), w.as_array().to_vec());
    if n.len() != k.len() || w.len() != k.len() {
        return Err(PyValueError::new_err("k, n and w must have one value per row"));
    }
    let f = py.detach(|| sylva_rs::shots::evidence::fit(&k, &n, &w, binomial));
    let d = PyDict::new(py);
    d.set_item("mu", f.mu)?;
    d.set_item("rho", f.rho)?;
    d.set_item("loglik", f.loglik)?;
    Ok(d)
}

/// Per row, `ln P(k | n)` (without the binomial coefficient) under a beta-binomial with per-row `mu` and `rho`
/// (sylva_rs::shots::evidence::ln_pmf). A scan's evidence term is the difference of two of these.
#[pyfunction]
fn shots_evidence_lnpmf<'py>(py: Python<'py>, k: PyReadonlyArray1<u32>, n: PyReadonlyArray1<u32>, mu: PyReadonlyArray1<f64>, rho: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (k, n, mu, rho) = (k.as_array(), n.as_array(), mu.as_array(), rho.as_array());
    if [n.len(), mu.len(), rho.len()].iter().any(|&x| x != k.len()) {
        return Err(PyValueError::new_err("k, n, mu and rho must have one value per row"));
    }
    let v: Vec<f64> = (0..k.len()).map(|i| sylva_rs::shots::evidence::ln_pmf(k[i], n[i], mu[i], rho[i])).collect();
    Ok(v.into_pyarray(py))
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
        wrap_pyfunction!(shots_completion_fit, m)?,
        wrap_pyfunction!(shots_completion_paths, m)?,
        wrap_pyfunction!(shots_completion_prior, m)?,
        wrap_pyfunction!(shots_completion_normalise, m)?,
        wrap_pyfunction!(shots_bundles, m)?,
        wrap_pyfunction!(shots_bundles_lines, m)?,
        wrap_pyfunction!(shots_view_offsets, m)?,
        wrap_pyfunction!(shots_elements, m)?,
        wrap_pyfunction!(shots_scan_normals, m)?,
        wrap_pyfunction!(shots_moment_octree, m)?,
        wrap_pyfunction!(shots_moment_offsets, m)?,
        wrap_pyfunction!(shots_field_counts, m)?,
        wrap_pyfunction!(shots_field_nodes, m)?,
        wrap_pyfunction!(shots_amodal_scan, m)?,
        wrap_pyfunction!(shots_amodal_lift, m)?,
        wrap_pyfunction!(shots_field_solve, m)?,
        wrap_pyfunction!(shots_evidence_fit, m)?,
        wrap_pyfunction!(shots_evidence_lnpmf, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
