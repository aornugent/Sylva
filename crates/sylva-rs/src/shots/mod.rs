// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Pulse-centric data: per-shot origin and direction with a CSR echo list.
//!
//! This is the representation ray-based canopy metrics need: a shot with
//! `echo_count == 0` is a genuine miss and still carries information about
//! free space. Modelled on canopygrid's `Shots` and raycloudtools' ray clouds.

pub mod ops;
pub mod reconstruct;

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::transform::{add, normalize, scale, sub, Transform};
use crate::{Point, PointCloud};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Shots {
    /// Beam origin per shot.
    pub origin: Vec<Point>,
    /// Unit beam direction per shot.
    pub direction: Vec<Point>,
    /// Offset of each shot's first echo in the echo arrays.
    pub echo_start: Vec<usize>,
    /// Echoes per shot (0 = no return).
    pub echo_count: Vec<u32>,
    /// Range from origin to each echo, ascending within a shot.
    pub echo_range: Vec<f64>,
    /// Extra per-echo attributes (amplitude, reflectance, deviation, ...).
    pub echo_attrs: BTreeMap<String, Attr>,
}

impl Shots {
    pub fn n_shots(&self) -> usize {
        self.origin.len()
    }

    pub fn n_echoes(&self) -> usize {
        self.echo_range.len()
    }

    /// Echo coordinates.
    pub fn echo_xyz(&self) -> Vec<Point> {
        let mut out = Vec::with_capacity(self.n_echoes());
        for s in 0..self.n_shots() {
            let (o, d) = (self.origin[s], self.direction[s]);
            let a = self.echo_start[s];
            for e in a..a + self.echo_count[s] as usize {
                out.push(add(&o, &scale(&d, self.echo_range[e])));
            }
        }
        out
    }

    /// Shot index for each echo.
    pub fn shot_of_echo(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.n_echoes());
        for s in 0..self.n_shots() {
            out.extend(std::iter::repeat_n(s, self.echo_count[s] as usize));
        }
        out
    }

    /// Echo rank within its shot (0 = first return).
    pub fn echo_rank(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.n_echoes());
        for &c in &self.echo_count {
            out.extend(0..c);
        }
        out
    }

    /// Convert echoes to a point cloud with `return_number`, `number_of_returns`,
    /// `range` and the echo attributes.
    pub fn to_pointcloud(&self) -> PointCloud {
        let xyz = self.echo_xyz();
        let mut cloud = PointCloud::new(xyz);
        let ranks: Vec<u8> = self.echo_rank().iter().map(|r| (r + 1).min(255) as u8).collect();
        let nret: Vec<u8> =
            self.shot_of_echo().iter().map(|&s| self.echo_count[s].min(255) as u8).collect();
        cloud.attrs.insert("return_number".into(), Attr::U8(ranks));
        cloud.attrs.insert("number_of_returns".into(), Attr::U8(nret));
        cloud.attrs.insert("range".into(), Attr::F32(self.echo_range.iter().map(|&r| r as f32).collect()));
        for (k, v) in &self.echo_attrs {
            cloud.attrs.insert(k.clone(), v.clone());
        }
        cloud
    }

    /// Apply a rigid transform to origins and directions.
    pub fn transformed(&self, t: &Transform) -> Shots {
        let mut out = self.clone();
        for o in &mut out.origin {
            *o = t.apply(o);
        }
        for d in &mut out.direction {
            *d = normalize(&t.apply_dir(d));
        }
        out
    }

    /// Keep a subset of shots (echo arrays are re-packed).
    pub fn subset(&self, keep: &[bool]) -> Shots {
        let mut out = Shots::default();
        let mut echo_idx = Vec::new();
        for s in 0..self.n_shots() {
            if !keep[s] {
                continue;
            }
            out.origin.push(self.origin[s]);
            out.direction.push(self.direction[s]);
            out.echo_start.push(out.echo_range.len());
            out.echo_count.push(self.echo_count[s]);
            let a = self.echo_start[s];
            for e in a..a + self.echo_count[s] as usize {
                out.echo_range.push(self.echo_range[e]);
                echo_idx.push(e);
            }
        }
        out.echo_attrs = self.echo_attrs.iter().map(|(k, v)| (k.clone(), v.take(&echo_idx))).collect();
        out
    }

    /// Build shots from a point cloud where each point is one single-return
    /// pulse from `origin` (e.g. a scan in SOCS). Adjacent points with equal
    /// `gps_time` (if present) are grouped into one multi-echo shot.
    pub fn from_pointcloud(cloud: &PointCloud, origin: Point) -> Shots {
        let times = cloud.attr_f64("gps_time");
        let mut shots = Shots::default();
        let mut i = 0;
        let n = cloud.len();
        let mut echo_idx = Vec::with_capacity(n);
        while i < n {
            let mut j = i + 1;
            if let Some(t) = &times {
                while j < n && t[j] == t[i] {
                    j += 1;
                }
            }
            // Sort this pulse's echoes by range.
            let mut members: Vec<(f64, usize)> = (i..j)
                .map(|k| (crate::transform::norm(&sub(&cloud.xyz[k], &origin)), k))
                .collect();
            members.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let far = members.last().unwrap().1;
            shots.origin.push(origin);
            shots.direction.push(normalize(&sub(&cloud.xyz[far], &origin)));
            shots.echo_start.push(shots.echo_range.len());
            shots.echo_count.push(members.len() as u32);
            for (r, k) in members {
                shots.echo_range.push(r);
                echo_idx.push(k);
            }
            i = j;
        }
        shots.echo_attrs = cloud
            .attrs
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "return_number" | "number_of_returns" | "range"))
            .map(|(k, v)| (k.clone(), v.take(&echo_idx)))
            .collect();
        shots
    }

    /// Pulses from one scan's echoes in the scanner's frame (SOCS), in any
    /// order: a LAS/LAZ/COPC export, which stores points spatially rather
    /// than as fired. Echoes are sorted by `gps_time` (required) and grouped
    /// into pulses as in [`Shots::from_pointcloud`]. The LAS
    /// `return_number` and `number_of_returns` are kept as echo attributes
    /// `las_return_number` and `las_number_of_returns`, so a pulse that lost
    /// echoes in export can still be recognised.
    ///
    /// With `fill_missing`, the pulses that returned nothing are recovered
    /// from the pulse timing ([`reconstruct::reconstruct_misses`], as for an
    /// `.rxp`) and appended as echo-less shots. A pulse whose every echo was
    /// dropped by the export is indistinguishable from a miss and is
    /// recovered as one.
    pub fn from_scan_points(cloud: &PointCloud, fill_missing: bool, zenith_range: Option<(f64, f64)>) -> Result<(Shots, Option<reconstruct::MissReport>)> {
        let t = cloud.attr_f64("gps_time").ok_or_else(|| Error::invalid("from_scan_points: the points need a gps_time attribute"))?;
        let mut order: Vec<usize> = (0..cloud.len()).collect();
        order.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
        let mut sorted = cloud.take(&order);
        for (from, to) in [("return_number", "las_return_number"), ("number_of_returns", "las_number_of_returns")] {
            if let Some(v) = sorted.attrs.remove(from) {
                sorted.attrs.insert(to.into(), v);
            }
        }
        let shots = Shots::from_pointcloud(&sorted, [0.0, 0.0, 0.0]);
        if !fill_missing || shots.n_shots() < 2 {
            return Ok((shots, None));
        }
        // One time per pulse: the groups from_pointcloud formed, in order.
        let ts: Vec<f64> = order.iter().map(|&i| t[i]).collect();
        let t0 = ts[0];
        let mut pulse_time = Vec::with_capacity(shots.n_shots());
        for (i, &x) in ts.iter().enumerate() {
            if i == 0 || x != ts[i - 1] {
                pulse_time.push(((x - t0) * 1e9).round() as u64);
            }
        }
        let rec = reconstruct::reconstruct_misses(&pulse_time, &shots.direction, zenith_range, 1);
        let n = rec.misses.len();
        if n == 0 {
            return Ok((shots, Some(rec.report)));
        }
        let empty = Shots {
            origin: vec![[0.0, 0.0, 0.0]; n],
            direction: rec.misses,
            echo_start: vec![shots.echo_range.len(); n],
            echo_count: vec![0; n],
            echo_range: Vec::new(),
            echo_attrs: shots.echo_attrs.iter().map(|(k, v)| (k.clone(), v.take(&[]))).collect(),
        };
        Ok((Shots::concatenate(&[&shots, &empty])?, Some(rec.report)))
    }

    /// Build from a raycloudtools-style ray cloud: per point `xyz` and the
    /// vector from the end point back to the sensor (`nx, ny, nz` or
    /// `rayx, rayy, rayz` in PLY ray clouds, `sx, sy, sz` in LAS/LAZ ones,
    /// which store `start - end` as float32). Points with
    /// `bound == 0` (or, without that attribute, `alpha == 0`) are unbounded
    /// rays: they become shots without an echo, so their far end is dropped
    /// and traversals run them to the edge of the grid.
    ///
    /// When `number_of_returns` and a pulse key (`beam_id`, else `gps_time`)
    /// are present, the returns of a pulse are joined into one multi-echo
    /// shot wherever they sit in the file, as rayvoxel does; otherwise every
    /// point is its own shot.
    pub fn from_ray_cloud(cloud: &PointCloud) -> Result<Shots> {
        let triple = |a: &str, b: &str, c: &str| Some((cloud.attr_f64(a)?, cloud.attr_f64(b)?, cloud.attr_f64(c)?));
        let n = cloud.len();
        let offsets = triple("sx", "sy", "sz").or_else(|| triple("nx", "ny", "nz")).or_else(|| triple("rayx", "rayy", "rayz"));
        let starts: Vec<Point> = if let Some((x, y, z)) = offsets {
            (0..n).map(|i| add(&cloud.xyz[i], &[x[i], y[i], z[i]])).collect()
        } else {
            return Err(Error::invalid("ray cloud needs sx,sy,sz or nx,ny,nz (or rayx,rayy,rayz) attributes"));
        };
        let bound = cloud.attr_f64("bound").or_else(|| cloud.attr_f64("alpha"));
        let is_bound = |i: usize| bound.as_ref().map(|a| a[i] > 0.0).unwrap_or(true);

        // Pulses in first-seen order, each a list of point indices.
        let nor = cloud.attr_f64("number_of_returns");
        let key = cloud.attr_f64("beam_id").filter(|b| b.iter().any(|&v| v >= 0.0)).or_else(|| cloud.attr_f64("gps_time"));
        let mut pulses: Vec<Vec<usize>> = Vec::with_capacity(n);
        match (nor, key) {
            (Some(nor), Some(key)) => {
                let mut open: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
                for i in 0..n {
                    if nor[i] <= 1.0 {
                        pulses.push(vec![i]);
                        continue;
                    }
                    let k = key[i].to_bits();
                    let slot = *open.entry(k).or_insert_with(|| {
                        pulses.push(Vec::new());
                        pulses.len() - 1
                    });
                    pulses[slot].push(i);
                    if pulses[slot].len() as f64 >= nor[i] {
                        open.remove(&k);
                    }
                }
            }
            _ => pulses.extend((0..n).map(|i| vec![i])),
        }

        let mut shots = Shots::default();
        let mut echo_idx = Vec::with_capacity(n);
        for members in &pulses {
            let origin = starts[members[0]];
            let mut by_range: Vec<(f64, usize)> = members.iter().map(|&i| (crate::transform::norm(&sub(&cloud.xyz[i], &origin)), i)).collect();
            by_range.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let far = by_range.last().unwrap().1;
            shots.origin.push(origin);
            shots.direction.push(normalize(&sub(&cloud.xyz[far], &origin)));
            shots.echo_start.push(shots.echo_range.len());
            let before = shots.echo_range.len();
            for (r, i) in by_range {
                if is_bound(i) {
                    shots.echo_range.push(r);
                    echo_idx.push(i);
                }
            }
            shots.echo_count.push((shots.echo_range.len() - before) as u32);
        }
        shots.echo_attrs = cloud
            .attrs
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "nx" | "ny" | "nz" | "rayx" | "rayy" | "rayz" | "sx" | "sy" | "sz"))
            .map(|(k, v)| (k.clone(), v.take(&echo_idx)))
            .collect();
        Ok(shots)
    }
}
