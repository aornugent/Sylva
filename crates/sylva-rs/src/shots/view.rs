// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The L0 view of one scan: its observations on the native lattice and the
//! signals read off them, with no judgement (3dtrees research 130, layers 1
//! and 2), and the error model those observations carry.
//!
//! - **cells:** the lattice is cropped to the swept band (columns at least
//!   1 % of lines record; stray columns can stretch it a hundredfold). Each
//!   cell is an [`state::ECHO`] (its first echo's range), a [`state::MISS`]
//!   (a pulse recorded with no echo, in exports that keep them), or empty:
//!   [`state::DELETED`] when at least half its 7 × 7 window holds echoes
//!   (the export deleted it, or it never returned), else [`state::UNKNOWN`].
//!   Deleted is a label for readers; the likelihood sees every empty cell as
//!   "no exported echo".
//! - **normals:** `n ∝ ω − ∇ ln r` by least squares over the 7 × 7 window,
//!   neighbours beyond a `jump` left out, one refit without 3-MAD outliers;
//!   `cos ι = 1/√(1 + |∇ ln r|²)`. Where the window's cells lie in a row,
//!   only the gradient along it is known.
//! - **runs and ends:** along each lattice axis, cells within `jump` of their
//!   neighbour are one run; each cell keeps its run's length and how the run
//!   ends at either side ([`end`]). Footprints across a run are
//!   [`ErrorModel::across`].
//! - **excess:** the surface's continuity at cm scale, alignment-rca's
//!   covariate on the lattice: `√(25 / n)` for the `n` surfel cells (a normal
//!   facing within 60° of the beam) of the 5 × 5 window, itself included,
//!   within `plane_tol` of the cell's tangent plane. 1 on a solid face, 2
//!   where three quarters of it is holes, texture or other surfaces.
//!   [`ErrorModel::surface`] turns it into the surface class (solid, mid,
//!   fragmented) that sets the per-echo noise, the depth bias and the
//!   offset's spread.

use super::lattice::{scan_lattice, Lattice};
use crate::qsm::evidence::Beam;
use crate::transform::{cross, dot, normalize};
use crate::Point;
use rayon::prelude::*;

/// A cell's observation.
pub mod state {
    pub const ECHO: u8 = 0;
    pub const DELETED: u8 = 1;
    pub const MISS: u8 = 2;
    pub const UNKNOWN: u8 = 3;
}

/// How a run ends on one side (2 bits each in [`View::ends`]: along-low, along-high, across-low, across-high,
/// from the low bits up; "along" runs along a mirror line, across the columns).
pub mod end {
    /// The next cell is farther: a silhouette edge.
    pub const SHARP: u8 = 0;
    /// The next cell is empty.
    pub const UNKNOWN: u8 = 1;
    /// The next cell is nearer: this surface passes behind something.
    pub const OCCLUDED: u8 = 2;
    pub const BORDER: u8 = 3;
}

/// Settings of [`View::new`].
#[derive(Debug, Clone, Copy)]
pub struct ViewParams {
    /// Neighbours farther apart than this (m) are different surfaces.
    pub jump: f64,
    /// Half-width (cells) of the normal's window.
    pub half: i64,
    /// A cell lies on the plane within this (m), and is a surfel with cos ι above `min_cos`, for `excess`.
    pub plane_tol: f64,
    pub min_cos: f64,
}

impl Default for ViewParams {
    fn default() -> Self {
        ViewParams { jump: 0.05, half: 3, plane_tol: 0.003, min_cos: 0.5 }
    }
}

/// The error model, as data. Defaults are AM's calibration (alignment-rca, research 129): per surface class
/// (solid, mid, fragmented) the per-echo noise, the depth bias and the per-scan offset's spread; the export's
/// reflectance cut (sim-clumps' calibration); the VZ-400i beam.
#[derive(Debug, Clone, Copy)]
pub struct ErrorModel {
    pub beam: Beam,
    /// The lattice step (rad).
    pub step: f64,
    /// Excess at the solid/mid and mid/fragmented boundaries.
    pub class_edges: [f64; 2],
    /// Per-echo noise (m) per class, a t with `nu_e` degrees of freedom.
    pub sigma_e: [f64; 3],
    pub nu_e: f64,
    /// Depth bias per class: an echo reads `-(b + c (range - bias_ref))` behind its surface (b in m, c in m/m).
    /// Measured between views, so a bias common to every view is not in it.
    pub bias: [[f64; 2]; 3],
    pub bias_ref: f64,
    /// The offset field's spread (m) per class, within one epoch and across epochs, between two views; one scan's
    /// spread is this over √2. A t with `nu_l` degrees of freedom.
    pub sigma_l: [[f64; 2]; 3],
    pub nu_l: f64,
    /// The offset is one value per (scan, block of this size, m): correlated over 1-5 m.
    pub block: f64,
    /// The export: echoes under `cut_db` after a per-echo reflectance error of `refl_noise_db` are deleted, and
    /// a kept echo is lost anyway with `p_drop`. Echo densities carry a uniform outlier share `p_outlier`.
    pub cut_db: f64,
    pub refl_noise_db: f64,
    pub p_drop: f64,
    pub p_outlier: f64,
    /// Overdispersion: a group's `n` footprints count as `n / (1 + (n − 1) ρ)` independent ones (the
    /// beta-binomial design effect of research 128's evidence term, ρ 0.1-0.6 there).
    pub rho: f64,
}

impl Default for ErrorModel {
    fn default() -> Self {
        ErrorModel {
            beam: Beam::VZ400I,
            step: 0.5227e-3,
            class_edges: [1.2, 1.8],
            sigma_e: [0.0024, 0.0037, 0.0051],
            nu_e: 4.0,
            bias: [[-0.0007, -0.00003], [-0.00175, -0.000055], [-0.00235, -0.000135]],
            bias_ref: 4.0,
            sigma_l: [[0.0019, 0.00275], [0.0029, 0.0044], [0.00365, 0.0061]],
            nu_l: 4.5,
            block: 2.0,
            cut_db: -25.0,
            refl_noise_db: 2.0,
            p_drop: 0.02,
            p_outlier: 0.01,
            rho: 0.3,
        }
    }
}

impl ErrorModel {
    /// The beam's diameter (m) at a range.
    pub fn footprint(&self, range: f64) -> f64 {
        self.beam.footprint(range)
    }

    /// Footprints per cell, `(footprint / cell)²`, at least 1: how many cells see the same spot.
    pub fn multiplicity(&self, range: f64) -> f64 {
        (self.footprint(range) / (self.step * range)).powi(2).max(1.0)
    }

    /// Footprints across a run of `cells` at a range.
    pub fn across(&self, cells: f64, range: f64) -> f64 {
        cells * self.step * range / self.footprint(range)
    }

    /// The surface class (0 solid, 1 mid, 2 fragmented) of an excess; NaN reads fragmented.
    pub fn surface(&self, excess: f64) -> u8 {
        if excess < self.class_edges[0] {
            0
        } else if excess < self.class_edges[1] {
            1
        } else {
            2
        }
    }

    /// How far behind its surface an echo of class `k` at `range` reads (m).
    pub fn depth_bias(&self, k: u8, range: f64) -> f64 {
        let [b, c] = self.bias[k as usize];
        -(b + c * (range - self.bias_ref))
    }
}

/// One scan's view. Cell arrays are row-major `(line, col)` over `n_lines × n_cols`; column `c` is the
/// lattice's column `c + c0`.
pub struct View {
    pub lat: Lattice,
    pub c0: usize,
    pub n_lines: usize,
    pub n_cols: usize,
    /// world = rot · scanner + origin.
    pub rot: [[f64; 3]; 3],
    pub origin: Point,
    /// The scan's epoch (its day, or any unit within which the scene holds still).
    pub epoch: u32,
    pub state: Vec<u8>,
    /// First-echo range (m), NaN unless an echo.
    pub range: Vec<f32>,
    /// The cell's pulse (`u32::MAX` if none).
    pub pulse: Vec<u32>,
    /// Unit normal toward the scanner (world), `cos ι` and excess; NaN where no fit.
    pub normal: Vec<[f32; 3]>,
    pub cos_inc: Vec<f32>,
    pub excess: Vec<f32>,
    /// Run length (cells) along and across.
    pub run: Vec<[u16; 2]>,
    pub ends: Vec<u8>,
}

fn rot_t(r: &[[f64; 3]; 3], d: &Point) -> Point {
    [0, 1, 2].map(|k| r[0][k] * d[0] + r[1][k] * d[1] + r[2][k] * d[2])
}

fn rot_mul(r: &[[f64; 3]; 3], d: &Point) -> Point {
    [0, 1, 2].map(|k| r[k][0] * d[0] + r[k][1] * d[1] + r[k][2] * d[2])
}

impl View {
    /// The view of one scan from its pulses in acquisition order: `time` (s), `direction` (world, unit) and the
    /// first echo's `range` (NaN for a pulse with no echo).
    pub fn new(time: &[f64], direction: &[Point], range: &[f32], rot: [[f64; 3]; 3], origin: Point, epoch: u32, p: &ViewParams) -> View {
        let scanner: Vec<Point> = direction.iter().map(|d| rot_t(&rot, d)).collect();
        let lat = scan_lattice(time, &scanner);
        drop(scanner);
        let mut per_col = vec![0usize; lat.n_cols];
        for (&l, &c) in lat.line.iter().zip(&lat.col) {
            if l != u32::MAX {
                per_col[c as usize] += 1;
            }
        }
        let min = lat.n_lines / 100;
        let c0 = per_col.iter().position(|&k| k > min).unwrap_or(0);
        let c1 = per_col.iter().rposition(|&k| k > min).unwrap_or(0);
        let (nl, nc) = (lat.n_lines, (c1 + 1).saturating_sub(c0));
        let n = nl * nc;
        let mut state_ = vec![state::UNKNOWN; n];
        let mut rng = vec![f32::NAN; n];
        let mut pulse = vec![u32::MAX; n];
        for (i, (&l, &c)) in lat.line.iter().zip(&lat.col).enumerate() {
            let c = c as usize;
            if l == u32::MAX || c < c0 || c > c1 {
                continue;
            }
            let k = l as usize * nc + c - c0;
            pulse[k] = i as u32;
            if range[i].is_finite() && range[i] > 0.0 {
                state_[k] = state::ECHO;
                rng[k] = range[i];
            } else {
                state_[k] = state::MISS;
            }
        }
        // deleted: an empty cell whose 7 x 7 window is at least half echoes (a summed-area table of echoes)
        let mut sat = vec![0u32; (nl + 1) * (nc + 1)];
        for l in 0..nl {
            for c in 0..nc {
                sat[(l + 1) * (nc + 1) + c + 1] = (state_[l * nc + c] == state::ECHO) as u32 + sat[l * (nc + 1) + c + 1]
                    + sat[(l + 1) * (nc + 1) + c]
                    - sat[l * (nc + 1) + c];
            }
        }
        for l in 0..nl {
            let (a, b) = (l.saturating_sub(3), (l + 4).min(nl));
            for c in 0..nc {
                let k = l * nc + c;
                if pulse[k] != u32::MAX {
                    continue;
                }
                let (e, f) = (c.saturating_sub(3), (c + 4).min(nc));
                let s = sat[b * (nc + 1) + f] + sat[a * (nc + 1) + e] - sat[a * (nc + 1) + f] - sat[b * (nc + 1) + e];
                if 2 * s >= ((b - a) * (f - e)) as u32 {
                    state_[k] = state::DELETED;
                }
            }
        }
        drop(sat);
        let mut v = View {
            lat, c0, n_lines: nl, n_cols: nc, rot, origin, epoch, state: state_, range: rng, pulse,
            normal: Vec::new(), cos_inc: Vec::new(), excess: Vec::new(), run: Vec::new(), ends: Vec::new(),
        };
        let fits: Vec<([f32; 3], f32)> = (0..n).into_par_iter().map(|k| v.fit(k, direction, p)).collect();
        (v.normal, v.cos_inc) = fits.into_iter().unzip();
        v.excess = (0..n).into_par_iter().map(|k| v.excess_at(k, direction, p)).collect();
        v.runs(p.jump);
        v
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
    fn fit(&self, k: usize, dir: &[Point], p: &ViewParams) -> ([f32; 3], f32) {
        let nan = ([f32::NAN; 3], f32::NAN);
        if self.state[k] != state::ECHO {
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
    fn excess_at(&self, k: usize, dir: &[Point], p: &ViewParams) -> f32 {
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

    /// Runs and their ends along both axes.
    fn runs(&mut self, jump: f64) {
        let (nl, nc) = (self.n_lines, self.n_cols);
        self.run = vec![[0u16; 2]; nl * nc];
        self.ends = vec![0u8; nl * nc];
        // axis 0: along a mirror line (stride 1, nc cells per line); axis 1: across lines (stride nc)
        for (axis, (lines, len, stride, step)) in [(nl, nc, nc, 1), (nc, nl, 1, nc)].into_iter().enumerate() {
            for s in 0..lines {
                let at = |i: usize| s * stride + i * step;
                let mut i = 0;
                while i < len {
                    let k = at(i);
                    if self.state[k] != state::ECHO {
                        i += 1;
                        continue;
                    }
                    let mut j = i + 1;
                    while j < len && self.state[at(j)] == state::ECHO && (self.range[at(j)] - self.range[at(j - 1)]).abs() as f64 <= jump {
                        j += 1;
                    }
                    let code = |nb: Option<usize>, here: usize| -> u8 {
                        match nb {
                            None => end::BORDER,
                            Some(m) if self.state[m] != state::ECHO => end::UNKNOWN,
                            Some(m) if self.range[m] > self.range[here] => end::SHARP,
                            _ => end::OCCLUDED,
                        }
                    };
                    let lo = code(i.checked_sub(1).map(at), at(i));
                    let hi = code((j < len).then(|| at(j)), at(j - 1));
                    let bits = (lo | hi << 2) << (4 * axis);
                    for m in i..j {
                        self.run[at(m)][axis] = (j - i).min(u16::MAX as usize) as u16;
                        self.ends[at(m)] |= bits;
                    }
                    i = j;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;

    /// A scan of a vertical wall 5 m away (the y = 5 plane), lines 0.03° apart in azimuth, a hole of deleted
    /// pulses in it, and a pole 1 m in front of part of it: normals face the scanner, the hole reads deleted, and
    /// the pole's runs end sharp against the wall.
    #[test]
    fn a_wall_with_a_hole_and_a_pole() {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let (mut t, mut d, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..200usize {
            for k in 0..400usize {
                let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
                if (94..97).contains(&l) && (204..207).contains(&k) {
                    continue; // deleted
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
        let v = View::new(&t, &d, &r, id, [0.0; 3], 0, &ViewParams::default());
        let nc = v.n_cols;
        let cell = |l: usize, k: usize| l * nc + k;
        assert_eq!(v.state[cell(95, 205)], state::DELETED);
        assert_eq!(v.state[cell(120, 300)], state::ECHO);
        let k = cell(150, 100);
        let n = v.normal[k];
        assert!(n[1] < -0.999, "{n:?}");
        assert!((v.excess[k] - 1.0).abs() < 1e-6);
        // the pole: 20 lines across, sharp at both sides across lines, and the wall beside it occluded
        let k = cell(50, 100);
        assert_eq!(v.run[k][1], 20);
        assert_eq!((v.ends[k] >> 4) & 3, end::SHARP);
        assert_eq!((v.ends[k] >> 6) & 3, end::SHARP);
        assert_eq!((v.ends[cell(39, 100)] >> 6) & 3, end::OCCLUDED);
        // beside the hole, the wall is less continuous
        assert!(v.excess[cell(95, 203)] > 1.1);
    }
}
