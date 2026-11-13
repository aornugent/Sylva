// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One scan as a range image: its observations on the native lattice and the
//! signals read off them, with no judgement (3dtrees research 130).
//!
//! - **cells:** the lattice is cropped to the scan pattern's zenith range, sky
//!   included. Each cell is [`Cell::Echoes`] (its pulse's echoes, every one
//!   kept; [`RangeImage::range`] holds the first), [`Cell::Miss`] (a pulse
//!   fired with no echo: a raw scan records only pulses that returned, so
//!   every in-pattern cell without one is a miss), or [`Cell::NoPulse`] (a
//!   line the scan recorded nothing on, whose cells the lattice can't place).
//! - **normals:** `n ∝ ω − ∇ ln r` by least squares over the 7 × 7 window,
//!   neighbours beyond a `jump` left out, one refit without 3-MAD outliers;
//!   `cos ι = 1/√(1 + |∇ ln r|²)`. Where the window's cells lie in a row,
//!   only the gradient along it is known.
//! - **excess:** the surface's continuity at cm scale, alignment-rca's
//!   covariate on the lattice: `√(25 / n)` for the `n` surfel cells (a normal
//!   facing within 60° of the beam) of the 5 × 5 window, itself included,
//!   within `plane_tol` of the cell's tangent plane. 1 on a solid face, 2
//!   where three quarters of it is holes, texture or other surfaces.
//!   [`ErrorModel::surface`](super::model::ErrorModel::surface) turns it into the surface class (solid, mid,
//!   fragmented) that sets the per-echo noise, the depth bias and the
//!   offset's spread.

use super::lattice::{scan_lattice, Lattice};
use crate::error::{Error, Result};
use crate::io::shots::ShotsFile;
use crate::transform::{cross, dot, normalize};
use crate::Point;
use rayon::prelude::*;
use std::path::Path;

/// What a cell holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cell {
    /// A pulse that returned, with its echoes.
    Echoes = 0,
    /// A pulse fired with no echo.
    Miss = 1,
    /// No pulse the scan can place here.
    NoPulse = 2,
}

/// Settings of [`RangeImage::new`].
#[derive(Debug, Clone, Copy)]
pub struct RangeImageParams {
    /// Neighbours farther apart than this (m) are different surfaces.
    pub jump: f64,
    /// Half-width (cells) of the normal's window.
    pub half: i64,
    /// A cell lies on the plane within this (m), and is a surfel with cos ι above `min_cos`, for `excess`.
    pub plane_tol: f64,
    pub min_cos: f64,
}

impl Default for RangeImageParams {
    fn default() -> Self {
        RangeImageParams { jump: 0.05, half: 3, plane_tol: 0.003, min_cos: 0.5 }
    }
}

/// One scan's range image. Cell arrays are row-major `(line, col)` over `n_lines × n_cols`; column `c` is the
/// lattice's column `c + c0`.
pub struct RangeImage {
    pub lat: Lattice,
    pub c0: usize,
    pub n_lines: usize,
    pub n_cols: usize,
    /// world = rot · scanner + origin.
    pub rot: [[f64; 3]; 3],
    pub origin: Point,
    /// The scan's epoch (its day, or any unit within which the scene holds still).
    pub epoch: u32,
    pub cell: Vec<Cell>,
    /// First-echo range (m), NaN unless the cell has echoes.
    pub range: Vec<f32>,
    /// The cell's pulse (`u32::MAX` if none), and each pulse's echo ranges (m) in a list, pulse i's at
    /// `echo_range[echo_start[i]..echo_start[i + 1]]`.
    pub pulse: Vec<u32>,
    pub echo_start: Vec<u32>,
    pub echo_range: Vec<f32>,
    /// Unit normal toward the scanner (world), `cos ι` and excess; NaN where no fit.
    pub normal: Vec<[f32; 3]>,
    pub cos_inc: Vec<f32>,
    pub excess: Vec<f32>,
}

fn rot_t(r: &[[f64; 3]; 3], d: &Point) -> Point {
    [0, 1, 2].map(|k| r[0][k] * d[0] + r[1][k] * d[1] + r[2][k] * d[2])
}

fn rot_mul(r: &[[f64; 3]; 3], d: &Point) -> Point {
    [0, 1, 2].map(|k| r[k][0] * d[0] + r[k][1] * d[1] + r[k][2] * d[2])
}

impl RangeImage {
    /// One scan's range image from its shots file (Sylva's Shots, raw: every echo of every pulse that returned)
    /// and the pose in its sidecar (`ScanPosNNN.json`'s `origin` and `sop`, world = sop · scanner).
    pub fn from_file(path: impl AsRef<Path>, epoch: u32, p: &RangeImageParams) -> Result<RangeImage> {
        let path = path.as_ref();
        let side = std::fs::read_to_string(path.with_extension("json")).map_err(|e| Error::file(path.with_extension("json"), e.to_string()))?;
        let meta: serde_json::Value = serde_json::from_str(&side).map_err(|e| Error::invalid(format!("scan sidecar: {e}")))?;
        let num = |v: &serde_json::Value| v.as_f64().ok_or_else(|| Error::invalid("scan sidecar: not a number"));
        let o = meta["origin"].as_array().ok_or_else(|| Error::invalid("scan sidecar: no origin"))?;
        let origin = [num(&o[0])?, num(&o[1])?, num(&o[2])?];
        let mut rot = [[0.0; 3]; 3];
        for (i, row) in rot.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = num(&meta["sop"][i][j])?;
            }
        }
        let f = ShotsFile::open(path)?;
        let groups: Vec<_> = (0..f.n_groups()).into_par_iter().map(|g| -> Result<_> {
            let f = f.reopen()?;
            Ok((f.read_rays(g, &[])?, f.read_echo_f64(g, "gps_time")?))
        }).collect::<Result<_>>()?;
        // pulses that returned, each with its first echo's time; the pattern's zenith range from every row
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut pulses: Vec<(f64, Point, usize, usize)> = Vec::new();
        let mut echoes: Vec<f32> = Vec::new();
        for (rays, time) in &groups {
            let mut e = 0usize;
            for (i, &n) in rays.echo_count.iter().enumerate() {
                let (z, a) = (rays.zenith[i] as f64, rays.azimuth[i] as f64);
                let n = n as usize;
                if n > 0 {
                    let d = [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()];
                    pulses.push((time[e], d, echoes.len(), n));
                    echoes.extend_from_slice(&rays.range[e..e + n]);
                }
                e += n;
            }
        }
        for (rays, _) in &groups {
            for i in 0..rays.zenith.len() {
                let d = rot_t(&rot, &{
                    let (z, a) = (rays.zenith[i] as f64, rays.azimuth[i] as f64);
                    [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
                });
                let z = d[2].clamp(-1.0, 1.0).acos().to_degrees();
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
        }
        drop(groups);
        pulses.par_sort_by(|a, b| a.0.total_cmp(&b.0));
        let time: Vec<f64> = pulses.iter().map(|q| q.0).collect();
        let dir: Vec<Point> = pulses.iter().map(|q| q.1).collect();
        let mut echo_start = Vec::with_capacity(pulses.len() + 1);
        let mut echo_range = Vec::with_capacity(echoes.len());
        echo_start.push(0u32);
        for q in &pulses {
            echo_range.extend_from_slice(&echoes[q.2..q.2 + q.3]);
            echo_start.push(echo_range.len() as u32);
        }
        Ok(RangeImage::new(&time, &dir, echo_start, echo_range, (zmin, zmax), rot, origin, epoch, p))
    }

    /// One scan's range image from pulses with at most one echo each: `range` NaN for a pulse fired with no echo;
    /// the pattern's zenith range is the pulses' own.
    pub fn from_first_echoes(time: &[f64], direction: &[Point], range: &[f32], rot: [[f64; 3]; 3], origin: Point, epoch: u32, p: &RangeImageParams) -> RangeImage {
        let mut echo_start = vec![0u32];
        let mut echo_range = Vec::new();
        for &r in range {
            if r.is_finite() && r > 0.0 {
                echo_range.push(r);
            }
            echo_start.push(echo_range.len() as u32);
        }
        let z = direction.iter().map(|d| rot_t(&rot, d)[2].clamp(-1.0, 1.0).acos().to_degrees());
        let zr = z.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), z| (a.min(z), b.max(z)));
        RangeImage::new(time, direction, echo_start, echo_range, zr, rot, origin, epoch, p)
    }

    /// The range image of one scan from its pulses in acquisition order: `time` (s), `direction` (world, unit),
    /// their echoes (`echo_start`, `echo_range`, as the fields) and the scan pattern's zenith range (degrees, scanner
    /// frame), to which the lattice is cropped.
    #[allow(clippy::too_many_arguments)]
    pub fn new(time: &[f64], direction: &[Point], echo_start: Vec<u32>, echo_range: Vec<f32>, zenith: (f64, f64), rot: [[f64; 3]; 3], origin: Point,
               epoch: u32, p: &RangeImageParams) -> RangeImage {
        let scanner: Vec<Point> = direction.iter().map(|d| rot_t(&rot, d)).collect();
        let lat = scan_lattice(time, &scanner);
        drop(scanner);
        let col = |z: f64| (z - lat.zenith0) / lat.zenith_step;
        let (ca, cb) = (col(zenith.0), col(zenith.1));
        let last = lat.n_cols.saturating_sub(1) as f64;
        let c0 = ca.min(cb).floor().clamp(0.0, last) as usize;
        let c1 = ca.max(cb).ceil().clamp(0.0, last) as usize;
        let (nl, nc) = (lat.n_lines, (c1 + 1).saturating_sub(c0));
        let n = nl * nc;
        // in the pattern, a cell without a pulse is a miss, except on a line the scan recorded nothing on
        let mut cell: Vec<Cell> = (0..n).map(|k| if lat.line_phase[k / nc.max(1)].is_finite() { Cell::Miss } else { Cell::NoPulse }).collect();
        let mut rng = vec![f32::NAN; n];
        let mut pulse = vec![u32::MAX; n];
        for (i, (&l, &c)) in lat.line.iter().zip(&lat.col).enumerate() {
            let c = c as usize;
            if l == u32::MAX || c < c0 || c > c1 {
                continue;
            }
            let k = l as usize * nc + c - c0;
            pulse[k] = i as u32;
            let (a, b) = (echo_start[i] as usize, echo_start[i + 1] as usize);
            if b > a {
                cell[k] = Cell::Echoes;
                rng[k] = echo_range[a];
            } else {
                cell[k] = Cell::Miss;
            }
        }
        let mut v = RangeImage {
            lat, c0, n_lines: nl, n_cols: nc, rot, origin, epoch, cell, range: rng, pulse, echo_start, echo_range,
            normal: Vec::new(), cos_inc: Vec::new(), excess: Vec::new(),
        };
        let fits: Vec<([f32; 3], f32)> = (0..n).into_par_iter().map(|k| v.fit(k, direction, p)).collect();
        (v.normal, v.cos_inc) = fits.into_iter().unzip();
        v.excess = (0..n).into_par_iter().map(|k| v.excess_at(k, direction, p)).collect();
        v
    }

    /// A cell's echo ranges (m), nearest first; empty unless it holds echoes.
    pub fn echoes(&self, k: usize) -> &[f32] {
        match self.pulse[k] {
            u32::MAX => &[],
            i => &self.echo_range[self.echo_start[i as usize] as usize..self.echo_start[i as usize + 1] as usize],
        }
    }

    /// The world direction of a cell's centre.
    pub fn direction(&self, l: f64, c: f64) -> Point {
        rot_mul(&self.rot, &self.lat.cell_direction(l, c + self.c0 as f64))
    }

    /// The fractional cell a world point falls in, and its range.
    pub fn cell_of(&self, x: &Point) -> Option<(f64, f64, f64)> {
        let d = rot_t(&self.rot, &[x[0] - self.origin[0], x[1] - self.origin[1], x[2] - self.origin[2]]);
        let (l, c) = self.lat.cell_of(&d)?;
        let c = c - self.c0 as f64;
        (c > -0.5 && c < self.n_cols as f64 - 0.5).then(|| (l, c, dot(&d, &d).sqrt()))
    }

    /// One cell's normal and cos ι.
    fn fit(&self, k: usize, dir: &[Point], p: &RangeImageParams) -> ([f32; 3], f32) {
        let nan = ([f32::NAN; 3], f32::NAN);
        if self.cell[k] != Cell::Echoes {
            return nan;
        }
        let (nl, nc) = (self.n_lines as i64, self.n_cols as i64);
        let (l, c) = ((k / self.n_cols) as i64, (k % self.n_cols) as i64);
        let w = dir[self.pulse[k] as usize];
        let ri = self.range[k] as f64;
        let e1 = normalize(&cross(&w, &if w[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
        let e2 = cross(&w, &e1);
        let mut obs: Vec<(f64, f64, f64)> = Vec::new();
        for dl in -p.half..=p.half {
            for dc in -p.half..=p.half {
                let (ll, cc) = (l + dl, c + dc);
                if (dl == 0 && dc == 0) || ll < 0 || cc < 0 || ll >= nl || cc >= nc {
                    continue;
                }
                let j = (ll * nc + cc) as usize;
                let rj = self.range[j] as f64;
                if !(rj > 0.0) || (rj - ri).abs() > p.jump {
                    continue;
                }
                let wj = dir[self.pulse[j] as usize];
                obs.push((dot(&wj, &e1), dot(&wj, &e2), (rj / ri).ln()));
            }
        }
        let fit = |o: &[(f64, f64, f64)]| -> Option<(f64, f64)> {
            if o.len() < 2 {
                return None;
            }
            let (mut a, mut b, mut d, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &(x1, x2, y) in o {
                a += x1 * x1;
                b += x1 * x2;
                d += x2 * x2;
                y1 += x1 * y;
                y2 += x2 * y;
            }
            let disc = ((a - d).powi(2) + 4.0 * b * b).sqrt();
            let (lmax, lmin) = (0.5 * (a + d + disc), 0.5 * (a + d - disc));
            if lmax <= 0.0 {
                return None;
            }
            let det = a * d - b * b;
            if (lmin / lmax).max(0.0).sqrt() >= 0.3 && det > 0.0 {
                Some(((d * y1 - b * y2) / det, (a * y2 - b * y1) / det))
            } else {
                // a row of cells: the gradient along it only
                let (vx, vy) = if b.abs() > 1e-30 { (lmax - d, b) } else if a >= d { (1.0, 0.0) } else { (0.0, 1.0) };
                let nv = (vx * vx + vy * vy).sqrt();
                let g = (vx * y1 + vy * y2) / (lmax * nv);
                Some((g * vx / nv, g * vy / nv))
            }
        };
        let res = |g: (f64, f64)| -> Vec<f64> { obs.iter().map(|&(x1, x2, y)| (y - g.0 * x1 - g.1 * x2).abs()).collect() };
        let mut g = fit(&obs);
        if let Some(g0) = g {
            let r = res(g0);
            let mut s = r.clone();
            s.sort_by(f64::total_cmp);
            let cut = (3.0 * s[s.len() / 2] / 0.6745).max(0.5 * 0.003 / ri);
            let inl: Vec<_> = obs.iter().zip(&r).filter(|(_, &e)| e <= cut).map(|(o, _)| *o).collect();
            if inl.len() < obs.len() {
                g = fit(&inl).or(g);
            }
        }
        let Some((g1, g2)) = g else { return nan };
        let grad = [0, 1, 2].map(|a| e1[a] * g1 + e2[a] * g2);
        let n = normalize(&[grad[0] - w[0], grad[1] - w[1], grad[2] - w[2]]);
        (n.map(|x| x as f32), (1.0 / (1.0 + dot(&grad, &grad)).sqrt()) as f32)
    }

    /// One cell's excess: `√(25 / n)` for the n surfel cells (a normal with cos ι > `min_cos`) of its 5 × 5
    /// window, itself included, within `plane_tol` of its tangent plane. NaN without a normal.
    fn excess_at(&self, k: usize, dir: &[Point], p: &RangeImageParams) -> f32 {
        if !self.cos_inc[k].is_finite() {
            return f32::NAN;
        }
        let (nl, nc) = (self.n_lines as i64, self.n_cols as i64);
        let (l, c) = ((k / self.n_cols) as i64, (k % self.n_cols) as i64);
        let ni = self.normal[k].map(|x| x as f64);
        let xi = dir[self.pulse[k] as usize].map(|x| x * self.range[k] as f64);
        let mut n = 0usize;
        for ll in (l - 2).max(0)..=(l + 2).min(nl - 1) {
            for cc in (c - 2).max(0)..=(c + 2).min(nc - 1) {
                let j = (ll * nc + cc) as usize;
                if self.cos_inc[j] > p.min_cos as f32 {
                    let xj = dir[self.pulse[j] as usize].map(|x| x * self.range[j] as f64);
                    n += (dot(&ni, &[xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]]).abs() < p.plane_tol || j == k) as usize;
                }
            }
        }
        (25.0 / n.max(1) as f64).sqrt() as f32
    }

    /// The surfels of this scan's echo cells with a normal whose echo lies inside the box `lo`-`hi` (world positions
    /// along the cell centre's ray, and normals), for the moments.
    pub fn surfels(&self, lo: &Point, hi: &Point) -> (Vec<Point>, Vec<Point>) {
        let inside = |p: &Point| (0..3).all(|a| p[a] >= lo[a] && p[a] < hi[a]);
        (0..self.cell.len())
            .filter(|&k| self.cos_inc[k].is_finite())
            .filter_map(|k| {
                let d = self.direction((k / self.n_cols) as f64, (k % self.n_cols) as f64);
                let r = self.range[k] as f64;
                let p = [0, 1, 2].map(|a| self.origin[a] + d[a] * r);
                inside(&p).then(|| (p, self.normal[k].map(|x| x as f64)))
            })
            .unzip()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;

    /// A scan of a vertical wall 5 m away (the y = 5 plane), lines 0.03° apart in azimuth, a hole of unrecorded
    /// pulses in it, and a pole 1 m in front of part of it: normals face the scanner, the hole reads as misses, and
    /// the wall beside the hole is less continuous.
    #[test]
    fn a_wall_with_a_hole_and_a_pole() {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let (mut t, mut d, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..200usize {
            for k in 0..400usize {
                let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
                if (94..97).contains(&l) && (204..207).contains(&k) {
                    continue; // unrecorded: no echo came back
                }
                let w = dir(zen, az);
                let mut range = 5.0 / w[1];
                if (40..60).contains(&l) {
                    range = 4.0 / w[1]; // a pole in front
                }
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = RangeImage::from_first_echoes(&t, &d, &r, id, [0.0; 3], 0, &RangeImageParams::default());
        let nc = v.n_cols;
        let cell = |l: usize, k: usize| l * nc + k;
        assert_eq!(v.cell[cell(95, 205)], Cell::Miss);
        assert_eq!(v.cell[cell(120, 300)], Cell::Echoes);
        assert_eq!(v.echoes(cell(120, 300)).len(), 1);
        let k = cell(150, 100);
        let n = v.normal[k];
        assert!(n[1] < -0.999, "{n:?}");
        assert!((v.excess[k] - 1.0).abs() < 1e-6);
        // beside the hole, the wall is less continuous
        assert!(v.excess[cell(95, 203)] > 1.1);
    }
}
