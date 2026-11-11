// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scan's range image as a shadow map, and the single-scan features of its
//! stripes (3dtrees research 128, `opacity.md`; a spike, not a fit).
//!
//! - **shadow map:** each lattice cell's first-echo range. A short cylinder
//!   (a stripe slice's axis piece from another scan, with its R) is tested
//!   against it cell by cell: every cell whose centre ray passes through the
//!   cylinder's *core* (within `R - footprint/2`, at least `R/2`, of the
//!   axis, inside its length) is one sample, so the footprint's cells share
//!   the outcome. With a `margin` > 0 the gate widens to `R + footprint/2 +
//!   margin` and only cells whose first echo lies beyond the closest
//!   approach (possible refutes) are kept, each with its miss vector, so a
//!   caller can soften or shift the refute test (3dtrees research 128,
//!   `fields-aniso.md`). A sample is *support* (the first echo lies in the
//!   cylinder's depth span, widened by `tol`), *refute* (it lies beyond: the
//!   pulse went through), *occluded* (nearer) or *unknown* (an empty cell:
//!   no echo, or one deleted in export). Each sample keeps its chord through
//!   the cylinder, for a Beer-Lambert foliage term downstream.
//! - **features:** per slice, from its own scan's image alone: the range
//!   jitter along its runs, the share of its pulses with later echoes, its
//!   first echoes' median reflectance, the share of empty cells around it,
//!   how much better a circle of its width fits its front's depth profile
//!   than a flat line, and the fractional anisotropy of its scan's
//!   orientation tensor around it.

use super::lattice::Lattice;
use super::stripes::Slice;
use super::tracts::{Field, Segment};
use crate::transform::{dot, norm, sub};
use crate::Point;

/// What a cell says about a cylinder.
pub mod outcome {
    pub const SUPPORT: u8 = 0;
    pub const REFUTE: u8 = 1;
    pub const OCCLUDED: u8 = 2;
    pub const UNKNOWN: u8 = 3;
}

/// A short cylinder in the world frame.
#[derive(Debug, Clone, Copy)]
pub struct Cyl {
    pub p: Point,
    pub u: Point,
    pub half: f64,
    pub r: f64,
}

/// One cell's test of one cylinder.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub cyl: u32,
    pub outcome: u8,
    /// Chord (m) of the cell's ray through the cylinder, its miss distance from the axis (m), the entry range (m)
    /// and the cell's first-echo range (NaN for an empty cell).
    pub chord: f64,
    pub miss: f64,
    pub t_in: f64,
    pub depth: f64,
    /// The footprint (m) at the closest approach, the sine of the ray's angle to the axis, the first echo's range
    /// beyond the closest approach (m), and the miss vector (world, m: the ray's closest point minus the axis's).
    pub fp: f64,
    pub sin: f64,
    pub excess: f64,
    pub m: Point,
    /// The cell's pulse (`u32::MAX` for an empty cell), so a caller can place the echo and time it.
    pub pulse: u32,
}

/// A scan's range image: the cell -> pulse index (`u32::MAX` empty), with the lattice and the scan's pose
/// (world = `rot` . scanner + `origin`).
pub struct ShadowMap<'a> {
    pub lat: &'a Lattice,
    pub cell: Vec<u32>,
    pub range: &'a [f32],
    pub rot: [[f64; 3]; 3],
    pub origin: Point,
}

impl<'a> ShadowMap<'a> {
    pub fn new(lat: &'a Lattice, range: &'a [f32], rot: [[f64; 3]; 3], origin: Point) -> Self {
        let nc = lat.n_cols;
        let mut cell = vec![u32::MAX; lat.n_lines * nc];
        for (i, (&l, &c)) in lat.line.iter().zip(&lat.col).enumerate() {
            if l != u32::MAX && range[i].is_finite() && range[i] > 0.0 {
                cell[l as usize * nc + c as usize] = i as u32;
            }
        }
        ShadowMap { lat, cell, range, rot, origin }
    }

    pub(crate) fn to_scanner(&self, p: &Point) -> Point {
        let d = sub(p, &self.origin);
        // rot^T d
        [0, 1, 2].map(|k| self.rot[0][k] * d[0] + self.rot[1][k] * d[1] + self.rot[2][k] * d[2])
    }

    pub(crate) fn rot_t(&self, u: &Point) -> Point {
        [0, 1, 2].map(|k| self.rot[0][k] * u[0] + self.rot[1][k] * u[1] + self.rot[2][k] * u[2])
    }

    /// Every core cell of each cylinder against this map. `footprint(t)` is the beam's width at range t; `tol`
    /// widens the depth span for range noise and registration; at most `max_cells` cells across a cylinder's
    /// image either way are visited. `margin` > 0 widens the gate as the module docs say. Cylinders run in
    /// parallel; samples come back in cylinder order.
    pub fn test(&self, cyls: &[Cyl], footprint: impl Fn(f64) -> f64 + Sync, tol: f64, max_cells: f64, margin: f64) -> Vec<Sample> {
        use rayon::prelude::*;
        let lat = self.lat;
        let (zs, az) = (lat.zenith_step.abs().to_radians(), lat.azimuth_step.abs().to_radians());
        let wide = margin > 0.0;
        let per: Vec<Vec<Sample>> = cyls.par_iter().enumerate().map(|(ci, cy)| {
            let mut out = Vec::new();
            let c = self.to_scanner(&cy.p);
            let u = self.rot_t(&cy.u);
            let dist = norm(&c);
            if !(dist > 0.1) || !(cy.r > 0.0) {
                return out;
            }
            let Some((l0, c0)) = lat.cell_of(&c) else { return out };
            let rg = if wide { cy.r + 0.5 * footprint(dist) + margin } else { cy.r };
            let alpha = (cy.half * cy.half + rg * rg).sqrt() / dist;
            let sin_z = (lat.cell_angles(l0, c0).0.to_radians().sin()).abs().max(0.05);
            let dl = (alpha / (az * sin_z)).min(max_cells).ceil() as i64 + 1;
            let dc = (alpha / zs).min(max_cells).ceil() as i64 + 1;
            let cu = dot(&c, &u);
            // b: the perpendicular of -c off the axis
            let b = [-(c[0] - cu * u[0]), -(c[1] - cu * u[1]), -(c[2] - cu * u[2])];
            let (li, cj) = (l0.round() as i64, c0.round() as i64);
            for l in (li - dl).max(0)..=(li + dl).min(lat.n_lines as i64 - 1) {
                for k in (cj - dc).max(0)..=(cj + dc).min(lat.n_cols as i64 - 1) {
                    let v = lat.cell_direction(l as f64, k as f64);
                    let vu = dot(&v, &u);
                    let a = [v[0] - vu * u[0], v[1] - vu * u[1], v[2] - vu * u[2]];
                    let aa = dot(&a, &a);
                    if aa < 1e-12 {
                        continue;
                    }
                    let ab = dot(&a, &b);
                    let bb = dot(&b, &b);
                    let miss2 = (bb - ab * ab / aa).max(0.0);
                    let tm = -ab / aa;
                    let fp = footprint(tm.max(0.0));
                    let core = if wide { cy.r + 0.5 * fp + margin } else { (cy.r - 0.5 * fp).max(0.5 * cy.r) };
                    if miss2 > core * core {
                        continue;
                    }
                    // the infinite cylinder of radius R (a ray outside it spans its closest approach only), then the
                    // slab |s| <= half along the axis
                    let hw = ((cy.r * cy.r - miss2).max(0.0) / aa).sqrt();
                    let (mut t0, mut t1) = (tm - hw, tm + hw);
                    if vu.abs() > 1e-9 {
                        let (s0, s1) = ((cu - cy.half) / vu, (cu + cy.half) / vu);
                        t0 = t0.max(s0.min(s1));
                        t1 = t1.min(s0.max(s1));
                    } else if (cu).abs() > cy.half {
                        continue;
                    }
                    // the ray's point at its closest approach must lie inside the length (a core ray, not a cap)
                    let s_m = tm * vu - cu;
                    if t1 < t0 || t0 <= 0.0 || s_m.abs() > cy.half {
                        continue;
                    }
                    let j = self.cell[l as usize * lat.n_cols + k as usize];
                    let depth = if j == u32::MAX { f64::NAN } else { self.range[j as usize] as f64 };
                    if wide && !(depth > tm + tol) {
                        continue;
                    }
                    let (o, d) = if j == u32::MAX {
                        (outcome::UNKNOWN, f64::NAN)
                    } else {
                        let d = self.range[j as usize] as f64;
                        let o = if d < t0 - tol {
                            outcome::OCCLUDED
                        } else if d > t1 + tol {
                            outcome::REFUTE
                        } else {
                            outcome::SUPPORT
                        };
                        (o, d)
                    };
                    // the miss vector in the scanner frame, then rotated to the world
                    let q = [tm * v[0] - c[0], tm * v[1] - c[1], tm * v[2] - c[2]];
                    let qu = dot(&q, &u);
                    let ms = [q[0] - qu * u[0], q[1] - qu * u[1], q[2] - qu * u[2]];
                    let m = [0, 1, 2].map(|a| self.rot[a][0] * ms[0] + self.rot[a][1] * ms[1] + self.rot[a][2] * ms[2]);
                    out.push(Sample {
                        cyl: ci as u32, outcome: o, chord: t1 - t0, miss: miss2.sqrt(), t_in: t0, depth: d,
                        fp, sin: aa.sqrt(), excess: depth - tm, m, pulse: j,
                    });
                }
            }
            out
        }).collect();
        per.into_iter().flatten().collect()
    }
}

/// Single-scan features of one slice (NaN where there are too few cells).
#[derive(Debug, Clone, Copy, Default)]
pub struct Features {
    /// Median |second difference| of range along the slice's runs (m), over sqrt(6): the range jitter.
    pub jitter: f64,
    /// Share of the slice's pulses with more than one echo.
    pub multi: f64,
    /// Median first-echo reflectance (dB) over the slice's cells.
    pub refl: f64,
    /// Share of empty cells in the slice's box widened by `pad` cells across and along.
    pub empty: f64,
    /// ln(rms flat / rms circle) of the front-depth profile across the slice (positive: circle fits better).
    pub arc: f64,
    /// Fractional anisotropy of the orientation tensor of this scan's segments in the 27 voxels around it.
    pub fa: f64,
    /// Cells counted as the slice's.
    pub cells: u32,
}

/// The features of the slices behind `segs` (each [`Segment::slice`] indexes `slices`), from one scan's image:
/// `n_echo` and `refl` per pulse (first echo), `voxel` the field's edge for the anisotropy, `pad` cells around.
pub fn features(map: &ShadowMap, slices: &[Slice], segs: &[Segment], n_echo: &[u8], refl: &[f32], voxel: f64, pad: i64) -> Vec<Features> {
    let lat = map.lat;
    let (nl, nc) = (lat.n_lines as i64, lat.n_cols as i64);
    let field = Field::build(segs, voxel);
    let at = |l: i64, c: i64| -> Option<u32> {
        if l < 0 || c < 0 || l >= nl || c >= nc {
            return None;
        }
        let j = map.cell[(l * nc + c) as usize];
        (j != u32::MAX).then_some(j)
    };
    let med = |v: &mut Vec<f64>| if v.is_empty() { f64::NAN } else { super::reconstruct::median(v) };
    segs.iter()
        .map(|g| {
            let s = &slices[g.slice];
            let mut f = Features::default();
            // the slice in the image: pass 0 runs along lines (k = line) at each col u; pass 1 the other way
            let (uc, kc) = if s.pass == 0 { (s.col, s.line) } else { (s.line, s.col) };
            let ext = |u: i64, k: i64| if s.pass == 0 { at(k, u) } else { at(u, k) };
            let half_w = 0.5 * (s.hi - s.lo + 1.0);
            let (u0, u1) = ((uc - 2.0).round() as i64, (uc + 1.0).round() as i64);
            let (mut d2, mut ne, mut rf) = (Vec::new(), Vec::new(), Vec::new());
            let (mut flat, mut circ, mut nprof) = (0.0, 0.0, 0usize);
            let (mut empty, mut total) = (0usize, 0usize);
            let rad_cells = s.half_app / s.step_m.max(1e-6);
            for u in (u0 - pad)..=(u1 + pad) {
                for k in ((kc - half_w).floor() as i64 - pad)..=((kc + half_w).ceil() as i64 + pad) {
                    total += 1;
                    if ext(u, k).is_none() {
                        empty += 1;
                    }
                }
            }
            for u in u0..=u1 {
                // the run at u: present cells within the slice's width, at its front's depth
                let mut run: Vec<(f64, f64)> = Vec::new();
                for k in ((kc - half_w).floor() as i64)..=((kc + half_w).ceil() as i64) {
                    if let Some(j) = ext(u, k) {
                        let r = map.range[j as usize] as f64;
                        if r >= s.range - 0.02 && r <= s.range + 2.0 * s.half_app.max(0.0) + 0.05 {
                            run.push((k as f64, r));
                            ne.push((n_echo[j as usize] > 1) as u8 as f64);
                            if refl[j as usize].is_finite() {
                                rf.push(refl[j as usize] as f64);
                            }
                        }
                    }
                }
                for w in run.windows(3) {
                    if w[2].0 - w[0].0 == 2.0 {
                        d2.push((w[0].1 - 2.0 * w[1].1 + w[2].1).abs() / 6f64.sqrt());
                    }
                }
                if run.len() >= 4 && rad_cells > 0.5 {
                    // flat: depth = a; circle: depth = a + R - sqrt(R^2 - x^2) around the run's middle
                    let mid = 0.5 * (run[0].0 + run[run.len() - 1].0);
                    let shape: Vec<f64> = run
                        .iter()
                        .map(|q| {
                            let x = ((q.0 - mid) / rad_cells).clamp(-1.0, 1.0);
                            s.half_app * (1.0 - (1.0 - x * x).sqrt())
                        })
                        .collect();
                    let n = run.len() as f64;
                    let mr = run.iter().map(|q| q.1).sum::<f64>() / n;
                    let ms = shape.iter().sum::<f64>() / n;
                    for (q, sh) in run.iter().zip(&shape) {
                        flat += (q.1 - mr).powi(2);
                        circ += (q.1 - mr - (sh - ms)).powi(2);
                    }
                    nprof += run.len();
                }
            }
            f.cells = ne.len() as u32;
            f.jitter = med(&mut d2);
            f.multi = if ne.is_empty() { f64::NAN } else { ne.iter().sum::<f64>() / ne.len() as f64 };
            f.refl = med(&mut rf);
            f.empty = empty as f64 / total.max(1) as f64;
            f.arc = if nprof >= 8 { 0.5 * ((flat + 1e-10) / (circ + 1e-10)).ln() } else { f64::NAN };
            // the anisotropy around the segment's point
            let k0 = field.key(&g.p);
            let mut m = [[0.0; 3]; 3];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = field.cells.get(&[k0[0] + dx, k0[1] + dy, k0[2] + dz]) {
                            for (u, w, _) in v {
                                for x in 0..3 {
                                    for y in 0..3 {
                                        m[x][y] += w * u[x] * u[y];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let e = nalgebra::SymmetricEigen::new(nalgebra::Matrix3::from_fn(|i, j| m[i][j])).eigenvalues;
            let (a, b, c) = (e[0], e[1], e[2]);
            let ss = a * a + b * b + c * c;
            f.fa = if ss > 0.0 { (0.5 * ((a - b).powi(2) + (b - c).powi(2) + (c - a).powi(2)) / ss).sqrt() } else { f64::NAN };
            f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::lattice::scan_lattice;
    use crate::shots::reconstruct::dir;

    /// A regular little scan (lines 0.03 deg apart in azimuth, cols 0.03 deg in zenith) of a vertical opaque
    /// cylinder `r` at (0, d, 0) in front of a wall at 20 m; returns the lattice and ranges.
    fn scene(r: f64, d: f64) -> (Lattice, Vec<f32>) {
        let (period, interval, step) = (6.6e-3, 0.825e-6, 0.03);
        let omega = step / interval;
        let (mut t, mut dirs, mut rng) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..400usize {
            for k in 0..600usize {
                let zen = 81.0 + step * k as f64;
                let ti = l as f64 * period + (zen - 81.0) / omega;
                let azi = -6.0 + 0.03 * ti / period;
                let v = dir(zen, azi);
                // ray against the vertical cylinder x^2 + (y - d)^2 = r^2
                let (a, b, c) = (v[0] * v[0] + v[1] * v[1], -2.0 * d * v[1], d * d - r * r);
                let disc = b * b - 4.0 * a * c;
                let hit = if disc >= 0.0 { (-b - disc.sqrt()) / (2.0 * a) } else { 20.0 };
                t.push(ti);
                dirs.push(v);
                rng.push(hit as f32);
            }
        }
        (scan_lattice(&t, &dirs), rng)
    }

    const ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn opaque_wood_supports_and_a_cylinder_beside_it_is_refuted() {
        let (lat, rng) = scene(0.03, 8.0);
        let map = ShadowMap::new(&lat, &rng, ID, [0.0; 3]);
        let fp = |t: f64| 0.007 + 0.35e-3 * t;
        let on = Cyl { p: [0.0, 8.0, 0.0], u: [0.0, 0.0, 1.0], half: 0.05, r: 0.03 };
        let off = Cyl { p: [0.15, 8.0, 0.0], u: [0.0, 0.0, 1.0], half: 0.05, r: 0.03 };
        let behind = Cyl { p: [0.0, 12.0, 0.0], u: [0.0, 0.0, 1.0], half: 0.05, r: 0.03 };
        let s = map.test(&[on, off, behind], fp, 0.015, 60.0, 0.0);
        let count = |c: u32, o: u8| s.iter().filter(|x| x.cyl == c && x.outcome == o).count();
        assert!(count(0, outcome::SUPPORT) > 20 && count(0, outcome::REFUTE) == 0, "{} {}", count(0, 0), count(0, 1));
        assert!(count(1, outcome::REFUTE) > 20 && count(1, outcome::SUPPORT) == 0);
        assert!(count(2, outcome::OCCLUDED) > 20 && count(2, outcome::SUPPORT) == 0);
        // the chord through the middle is about the diameter
        let cmax = s.iter().filter(|x| x.cyl == 0).map(|x| x.chord).fold(0.0, f64::max);
        assert!((cmax - 0.06).abs() < 0.01, "{cmax}");
        // a support's pulse is the cell's: its range is the sample's depth
        for x in s.iter().filter(|x| x.outcome == outcome::SUPPORT) {
            assert!(x.pulse != u32::MAX && (rng[x.pulse as usize] as f64 - x.depth).abs() < 1e-9);
        }
    }

    #[test]
    fn a_wide_gate_keeps_the_hard_refutes_and_their_miss_vectors() {
        let (lat, rng) = scene(0.03, 8.0);
        let map = ShadowMap::new(&lat, &rng, ID, [0.0; 3]);
        let fp = |t: f64| 0.007 + 0.35e-3 * t;
        let off = Cyl { p: [0.15, 8.0, 0.0], u: [0.0, 0.0, 1.0], half: 0.05, r: 0.02 };
        let hard = map.test(&[off], fp, 0.015, 60.0, 0.0);
        let wide = map.test(&[off], fp, 0.015, 60.0, 0.02);
        let n_hard = hard.iter().filter(|x| x.outcome == outcome::REFUTE).count();
        let core = |x: &Sample| (0.02 - 0.5 * x.fp).max(0.01);
        let n_core = wide.iter().filter(|x| x.outcome == outcome::REFUTE && x.miss <= core(x)).count();
        assert!(n_hard > 5 && n_core == n_hard, "{n_hard} {n_core}");
        assert!(wide.len() > 2 * n_hard);
        for x in &wide {
            // the miss vector is across the (vertical) axis, as long as the miss, and within the gate
            let mm = (x.m[0] * x.m[0] + x.m[1] * x.m[1] + x.m[2] * x.m[2]).sqrt();
            assert!(x.m[2].abs() < 1e-9 && (mm - x.miss).abs() < 1e-9 && x.miss <= 0.02 + 0.5 * x.fp + 0.02 + 1e-12);
            assert!(x.excess > 0.015);
        }
    }
}
