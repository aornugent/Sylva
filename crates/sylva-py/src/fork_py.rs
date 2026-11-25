// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings 3dtrees' fork adds (CHANGELOG-3dtrees.md): the ray-cloud tiles.
//! Kept apart from upstream's binding files, so a merge from upstream meets
//! them in one line of `lib.rs`.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::io;

use crate::err;

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

/// Adds this file's functions to the `_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tile_ray_clouds, m)?)
}
