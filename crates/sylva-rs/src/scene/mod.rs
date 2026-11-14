// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of proposed elements (sticks and leaf patches) scored against scans' range images by the ray likelihood
//! (3dtrees research 130). Its inputs are pulses (`crate::io::shots`, `crate::shots::reconstruct`).
//!
//! - [`lattice`]: each pulse's native (line, col) cell from its time.
//! - [`range_image`]: one scan's observations on its lattice, and the signals read off them.
//! - [`moments`]: surfel moments per (voxel, scan), and the candidate elements read off them.
//! - [`model`]: the error model and the background.
//! - [`extinction`]: the diffuse field, a residual extinction on voxels.
//! - [`scene`]: elements, their rendering, the per-scan offset field and the [`scene::Scene`] that holds them.

pub mod extinction;
pub mod lattice;
pub mod model;
pub mod moments;
pub mod range_image;
pub mod scene;
