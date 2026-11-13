//! sylva-rs: LiDAR processing for forest ecology and remote sensing research.
//!
//! Copyright (C) 2026 Tim Devereux, The University of Queensland. Free
//! software under the GNU General Public License v3.0 or later; see the
//! LICENSE file. There is no warranty, to the extent permitted by law.
//!
//! The crate is organised around two data models:
//!
//! * [`PointCloud`] — an `(N, 3)` set of coordinates plus named per-point
//!   attributes, the unit of exchange with LAS/LAZ/PLY/text files.
//! * [`Shots`] — pulse-centric data (per-pulse origin and direction, with a
//!   CSR list of echoes), which is what ray-based canopy metrics need.
//!
//! Algorithms are grouped by topic: [`filters`], [`cluster`], [`ground`],
//! [`canopy`], [`voxel`], [`trees`], [`registration`] and [`qsm`].

/// The settings record written once: the struct, each field a real with its
/// default (`name: f64 = default`), its [`Default`], and its fields by name
/// (`FIELDS`, `get`, `set`), which is how they cross to Python.
macro_rules! params {
    ($(#[$m:meta])* pub struct $T:ident { $($(#[$fm:meta])* pub $f:ident: f64 = $v:expr,)* }) => {
        $(#[$m])*
        pub struct $T { $($(#[$fm])* pub $f: f64,)* }

        impl Default for $T {
            fn default() -> Self {
                $T { $($f: $v,)* }
            }
        }

        impl $T {
            /// The fields' names, in order.
            pub const FIELDS: &'static [&'static str] = &[$(stringify!($f)),*];

            /// The field `name`'s value; `None` if there is no such field.
            pub fn get(&self, name: &str) -> Option<f64> {
                match name {
                    $(stringify!($f) => Some(self.$f),)*
                    _ => None,
                }
            }

            /// Sets the field `name`; false if there is no such field.
            pub fn set(&mut self, name: &str, v: f64) -> bool {
                match name {
                    $(stringify!($f) => self.$f = v,)*
                    _ => return false,
                }
                true
            }
        }
    };
}

pub mod als;
pub mod canopy;
pub mod change;
pub mod cluster;
pub mod coreg;
pub mod error;
pub mod filters;
pub mod fusion;
pub mod geo;
pub mod ground;
pub mod io;
pub mod leaves;
pub mod pointcloud;
pub mod qsm;
pub mod quality;
pub mod raster;
pub mod registration;
pub mod riscan;
pub mod scene;
pub mod shots;
pub mod synthetic;
pub mod transform;
pub mod trees;
pub mod util;
pub mod voxel;
pub mod waveform;

pub use error::{Error, Result};
pub use pointcloud::{Attr, PointCloud};
pub use raster::Raster;
pub use shots::Shots;
pub use transform::Transform;

/// A 3-D point.
pub type Point = [f64; 3];
