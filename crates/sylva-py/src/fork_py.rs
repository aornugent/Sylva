// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings 3dtrees' fork adds (CHANGELOG-3dtrees.md): the ray-cloud tiles and
//! a scan read straight into a Shots file.
//! Kept apart from upstream's binding files, so a merge from upstream meets
//! them in one line of `lib.rs`.

use std::path::PathBuf;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sylva_rs::io;

use crate::err;

/// `read_rxp_shots`, transformed by `matrix` and written as `write_shots` writes them, all without leaving
/// Rust: the pulses are held once, never copied into Python and back. Returns (pulses, echoes, report).
#[pyfunction]
#[pyo3(signature = (path, out, matrix, library=None, drop_pseudo_echoes=true, min_range=0.5, max_range=f64::INFINITY, stride=1, max_points=None, echoes="all".to_string(), shot_stride=1, mta="auto", max_zone=6, keep_uncertain=false, shim=None, max_lines=None, mta_model=None, fill_missing=None, double=false, row_group_size=1048576, zstd_level=3, origin_tolerance=1e-3))]
#[allow(clippy::too_many_arguments)]
fn save_rxp_shots<'py>(py: Python<'py>, path: PathBuf, out: PathBuf, matrix: PyReadonlyArray2<f64>, library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize,
                       max_points: Option<usize>, echoes: String, shot_stride: usize, mta: &str, max_zone: usize, keep_uncertain: bool, shim: Option<PathBuf>, max_lines: Option<usize>,
                       mta_model: Option<String>, fill_missing: Option<&Bound<'py, PyAny>>, double: bool, row_group_size: usize, zstd_level: i32, origin_tolerance: f64)
                       -> PyResult<(usize, usize, Option<Bound<'py, PyDict>>)> {
    let mut opts = crate::rxp_opts(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride, mta, max_zone, keep_uncertain, shim, max_lines, mta_model)?;
    (opts.fill_missing, opts.zenith_range) = crate::fill_option(fill_missing)?;
    let t = crate::matrix_from_py(Some(matrix))?.unwrap();
    let write = io::shots::ShotsWriteOptions { double, row_group_size, zstd_level, origin_tolerance };
    let (n_shots, n_echoes, report) = py
        .detach(|| -> sylva_rs::Result<_> {
            let (mut s, report) = io::riegl::read_rxp_shots_with_report(&path, &opts)?;
            s.transform(&t);
            io::shots::write_shots(&s, &out, &write)?;
            Ok((s.n_shots(), s.n_echoes(), report))
        })
        .map_err(err)?;
    let report = report.map(|r| crate::miss_report_to_py(py, &r)).transpose()?;
    Ok((n_shots, n_echoes, report))
}

#[pyfunction]
#[pyo3(signature = (files, out, width, buffer, voxel, terrain_voxel, extent=None, min_rays=100_000, threads=4, memory=u64::MAX))]
#[allow(clippy::too_many_arguments)]
fn tile_ray_clouds<'py>(py: Python<'py>, files: Vec<PathBuf>, out: PathBuf, width: f64, buffer: f64, voxel: f64, terrain_voxel: f64,
                        extent: Option<(f64, f64, f64, f64)>, min_rays: usize, threads: usize, memory: u64) -> PyResult<(Vec<Bound<'py, PyDict>>, usize)> {
    let opts = io::ray_tiles::TileOptions { width, buffer, voxel, terrain_voxel, extent: extent.map(|e| [e.0, e.1, e.2, e.3]), min_rays };
    let (tiles, n_terrain) = py.detach(|| io::ray_tiles::tile_ray_clouds(&files, &out, &opts, threads, memory)).map_err(err)?;
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
    m.add_function(wrap_pyfunction!(tile_ray_clouds, m)?)?;
    m.add_function(wrap_pyfunction!(save_rxp_shots, m)?)
}
