// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Surfels from a scan's range image, and three shape statistics of the
//! surfels of every scan in a small region (3dtrees research 128,
//! `surfels.md`; a spike, not a fit).
//!
//! - **surfel:** a lattice cell's first echo with a normal from a plane fit
//!   to the echoes of the cells around it (`half` cells either way), skipping
//!   empty cells and any whose range differs from the centre's by more than
//!   `jump` (a depth discontinuity), refitted twice without the outliers.
//!   The normal is signed toward the scanner. A neighbourhood too narrow
//!   across (a row of cells: a twig a cell or two wide) gives no surfel. Its
//!   uncertainty: the range noise along the beam, a quarter footprint across
//!   it (projected through the incidence angle), over the fit's lever arm.
//! - **region statistics** of the surfels inside a query cylinder (widened by
//!   a margin), from all scans or one:
//!   - **thickness:** the robust spread (1.4826 MAD) of the positions about a
//!     fitted plane and about the medial cylinder; wood is range noise thick
//!     about one of them, a clump of phyllodes is not;
//!   - **normal tensor** `Σ n nᵀ / N`: its eigenvalues (sum 1, descending)
//!     read as a rank pattern (a plane 1, a cylinder 2 with its axis the null
//!     direction, a clump 3);
//!   - **medial collapse:** with the axis `a` the tensor's null direction,
//!     the spread about a line of `p - R n` taken across `a` is quadratic in
//!     R, so its minimum and R are solved directly (once more without the
//!     outliers); a cylinder collapses onto its axis, a clump doesn't.

use super::lattice::Lattice;
use super::opacity::ShadowMap;
use crate::Point;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use std::collections::HashMap;

pub(super) type V = Vector3<f64>;

pub(super) fn v(p: &Point) -> V {
    V::new(p[0], p[1], p[2])
}

pub(super) fn pt(x: &V) -> Point {
    [x[0], x[1], x[2]]
}

/// Eigenvalues descending, with their unit eigenvectors.
pub(super) fn eig(m: &Matrix3<f64>) -> ([f64; 3], [V; 3]) {
    let e = SymmetricEigen::new(*m);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e.eigenvalues[b].partial_cmp(&e.eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));
    let col = |k: usize| e.eigenvectors.column(k).into_owned();
    ([e.eigenvalues[idx[0]], e.eigenvalues[idx[1]], e.eigenvalues[idx[2]]], [col(idx[0]), col(idx[1]), col(idx[2])])
}

pub(super) fn mad(mut x: Vec<f64>) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    1.4826 * super::reconstruct::median(&mut x)
}

#[derive(Debug, Clone, Copy)]
pub struct SurfelParams {
    /// Cells either way of the centre in the fit's window.
    pub half: i64,
    /// A neighbour whose range differs from the centre's by more is across a depth discontinuity (m).
    pub jump: f64,
    /// The range noise (m, 1 sigma).
    pub range_noise: f64,
    /// Fewest cells in the fit.
    pub min_cells: usize,
    /// Fewest of the neighbourhood's minor in-plane rms over its major (a row of cells has no normal).
    pub min_aspect: f64,
}

impl Default for SurfelParams {
    fn default() -> Self {
        SurfelParams { half: 3, jump: 0.05, range_noise: 0.003, min_cells: 6, min_aspect: 0.2 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Surfel {
    pub pulse: u32,
    /// World position (the pulse's first echo) and unit normal toward the scanner.
    pub p: Point,
    pub n: Point,
    /// Normal's 1-sigma angle (rad); the 1-sigma position along the beam and across it (m).
    pub sigma_n: f64,
    pub sigma_r: f64,
    pub sigma_lat: f64,
    /// Rms of the fit's residuals (m), cells in it, and |cos| of the angle between the normal and the beam.
    pub rms: f64,
    pub cells: u32,
    pub cos_inc: f64,
}

/// The surfels of one scan's first echoes: `dir` each pulse's unit direction in the scanner frame (its echo is at
/// `range` along it), the map's pose to world, `keep(pulse, world point)` the pulses wanted, `footprint(t)` the beam's
/// width at range t. Also returns how many kept pulses had no surfel (too few cells, or a row of them).
pub fn surfels(
    map: &ShadowMap,
    dir: &[Point],
    keep: impl Fn(usize, &Point) -> bool + Sync,
    footprint: impl Fn(f64) -> f64 + Sync,
    prm: &SurfelParams,
) -> (Vec<Surfel>, usize) {
    use rayon::prelude::*;
    let lat: &Lattice = map.lat;
    let (nl, nc) = (lat.n_lines as i64, lat.n_cols as i64);
    let rot = Matrix3::from_fn(|i, j| map.rot[i][j]);
    let org = v(&map.origin);
    let x_of = |j: usize| v(&dir[j]) * map.range[j] as f64;
    let out: Vec<Option<Surfel>> = (0..lat.line.len())
        .into_par_iter()
        .filter_map(|i| {
            let (l, c) = (lat.line[i], lat.col[i]);
            if l == u32::MAX || map.cell[l as usize * lat.n_cols + c as usize] != i as u32 {
                return None;
            }
            let xi = x_of(i);
            let pw = rot * xi + org;
            if !keep(i, &pt(&pw)) {
                return None;
            }
            let ri = map.range[i] as f64;
            let mut nb: Vec<V> = Vec::with_capacity(((2 * prm.half + 1) * (2 * prm.half + 1)) as usize);
            for dl in -prm.half..=prm.half {
                for dc in -prm.half..=prm.half {
                    let (ll, cc) = (l as i64 + dl, c as i64 + dc);
                    if ll < 0 || cc < 0 || ll >= nl || cc >= nc {
                        continue;
                    }
                    let j = map.cell[(ll * nc + cc) as usize];
                    if j != u32::MAX && ((map.range[j as usize] as f64) - ri).abs() <= prm.jump {
                        nb.push(x_of(j as usize));
                    }
                }
            }
            Some(fit(&nb, &xi, ri, footprint(ri), prm).map(|(n, sig_n, sig_lat, rms, cells, cos_inc)| Surfel {
                pulse: i as u32,
                p: pt(&pw),
                n: pt(&(rot * n)),
                sigma_n: sig_n,
                sigma_r: prm.range_noise,
                sigma_lat: sig_lat,
                rms,
                cells,
                cos_inc,
            }))
        })
        .collect();
    let failed = out.iter().filter(|s| s.is_none()).count();
    (out.into_iter().flatten().collect(), failed)
}

/// A robust plane through the neighbourhood (scanner frame), its normal toward the scanner: (n, sigma_n, sigma_lat,
/// rms, cells, |cos incidence|).
fn fit(nb: &[V], xi: &V, ri: f64, fp: f64, prm: &SurfelParams) -> Option<(V, f64, f64, f64, u32, f64)> {
    let mut pts: Vec<V> = nb.to_vec();
    let mut res = None;
    for _ in 0..3 {
        if pts.len() < prm.min_cells {
            return None;
        }
        let m = pts.iter().sum::<V>() / pts.len() as f64;
        let cov = pts.iter().map(|q| (q - m) * (q - m).transpose()).sum::<Matrix3<f64>>() / pts.len() as f64;
        let (e, vec) = eig(&cov);
        if !(e[1] > 0.0) || (e[1] / e[0].max(1e-30)).sqrt() < prm.min_aspect {
            return None;
        }
        let n = vec[2];
        let r: Vec<f64> = pts.iter().map(|q| (q - m).dot(&n)).collect();
        let s = mad(r.iter().map(|x| x.abs()).collect());
        let gate = (2.5 * s).max(2.0 * prm.range_noise);
        res = Some((n, m, e, r.iter().map(|x| x * x).sum::<f64>() / r.len() as f64, pts.len()));
        let before = pts.len();
        pts = pts.into_iter().zip(&r).filter(|(_, x)| x.abs() <= gate).map(|(q, _)| q).collect();
        if pts.len() == before {
            break;
        }
    }
    let (mut n, _m, e, ms, cells) = res?;
    if n.dot(xi) > 0.0 {
        n = -n;
    }
    let cos_inc = (n.dot(xi) / ri).abs();
    let tan = ((1.0 - cos_inc * cos_inc).max(0.0).sqrt() / cos_inc.max(0.2)).min(5.0);
    let sig_lat = 0.25 * fp;
    let sig_eff = (prm.range_noise.powi(2) + (sig_lat * tan).powi(2)).sqrt();
    let sig_n = (sig_eff / (e[1].sqrt() * (cells as f64).sqrt())).atan();
    Some((n, sig_n, sig_lat, ms.sqrt(), cells as u32, cos_inc))
}

/// A query region: a short cylinder in the world frame, and whose surfels count (`scan` -1: every scan's).
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub p: Point,
    pub u: Point,
    pub half: f64,
    pub r: f64,
    pub scan: i32,
}

/// The three statistics of one region (NaN where it has fewer than `min_n` surfels).
#[derive(Debug, Clone, Copy)]
pub struct RegionStats {
    pub n: u32,
    pub n_scans: u32,
    /// Robust thickness (m) about the positions' plane and about the medial cylinder.
    pub thick_plane: f64,
    pub thick_cyl: f64,
    /// The normal tensor's eigenvalues (descending, sum 1) and its null direction.
    pub lam: [f64; 3],
    pub axis: Point,
    /// Medial collapse: R (m), the spread of p - R n about the line (m), the spread at R = 0, and the line's point
    /// nearest the region's centre.
    pub medial_r: f64,
    pub spread: f64,
    pub spread0: f64,
    pub centre: Point,
}

const NAN3: Point = [f64::NAN; 3];

impl Default for RegionStats {
    fn default() -> Self {
        RegionStats {
            n: 0,
            n_scans: 0,
            thick_plane: f64::NAN,
            thick_cyl: f64::NAN,
            lam: NAN3,
            axis: NAN3,
            medial_r: f64::NAN,
            spread: f64::NAN,
            spread0: f64::NAN,
            centre: NAN3,
        }
    }
}

/// Medial collapse about axis `a`: (R, spread, spread at 0, the line's mean point), over the surfels in `idx`.
pub(super) fn collapse(p: &[V], n: &[V], idx: &[usize], a: &V, r_max: f64) -> (f64, f64, f64, V) {
    let proj = Matrix3::identity() - a * a.transpose();
    let k = idx.len() as f64;
    let pp: Vec<V> = idx.iter().map(|&i| proj * p[i]).collect();
    let nn: Vec<V> = idx.iter().map(|&i| proj * n[i]).collect();
    let mp = pp.iter().sum::<V>() / k;
    let mn = nn.iter().sum::<V>() / k;
    let (mut aa, mut bb, mut cc) = (0.0, 0.0, 0.0);
    for (x, y) in pp.iter().zip(&nn) {
        let (dx, dy) = (x - mp, y - mn);
        aa += dx.dot(&dx);
        bb += dx.dot(&dy);
        cc += dy.dot(&dy);
    }
    let (aa, bb, cc) = (aa / k, bb / k, cc / k);
    let r = if cc > 1e-6 { (bb / cc).clamp(0.0, r_max) } else { 0.0 };
    let s2 = (aa - 2.0 * r * bb + r * r * cc).max(0.0);
    let q = idx.iter().map(|&i| p[i] - n[i] * r).sum::<V>() / k;
    (r, s2.sqrt(), aa.sqrt(), q)
}

/// The surfels in a hash grid, to gather a region's.
pub struct SurfelGrid {
    cell: f64,
    grid: HashMap<[i32; 3], Vec<u32>>,
}

impl SurfelGrid {
    pub fn new(p: &[Point]) -> Self {
        let cell = 0.05;
        let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, x) in p.iter().enumerate() {
            grid.entry(Self::key(cell, x)).or_default().push(i as u32);
        }
        SurfelGrid { cell, grid }
    }

    fn key(cell: f64, x: &Point) -> [i32; 3] {
        [(x[0] / cell).floor() as i32, (x[1] / cell).floor() as i32, (x[2] / cell).floor() as i32]
    }

    /// The surfels of `g` (of its scan, or every scan's): within `1.5 r + margin` of its axis and `max(half, r) +
    /// margin` of its centre along it.
    pub fn gather(&self, pv: &[V], scan: &[i32], g: &Region, margin: f64) -> Vec<usize> {
        let (c, u) = (v(&g.p), v(&g.u).normalize());
        let (rho, h) = (1.5 * g.r + margin, g.half.max(g.r) + margin);
        let reach = (rho * rho + h * h).sqrt();
        let (k0, k1) = (Self::key(self.cell, &pt(&c.add_scalar(-reach))), Self::key(self.cell, &pt(&c.add_scalar(reach))));
        let mut idx = Vec::new();
        for x in k0[0]..=k1[0] {
            for y in k0[1]..=k1[1] {
                for z in k0[2]..=k1[2] {
                    for &i in self.grid.get(&[x, y, z]).map(|v| v.as_slice()).unwrap_or(&[]) {
                        let i = i as usize;
                        if g.scan >= 0 && scan[i] != g.scan {
                            continue;
                        }
                        let d = pv[i] - c;
                        let s = d.dot(&u);
                        if s.abs() <= h && (d - u * s).norm() <= rho {
                            idx.push(i);
                        }
                    }
                }
            }
        }
        idx
    }
}

/// Each region's statistics over the surfels `p`, `n` (world) of scans `scan`, gathered as `SurfelGrid::gather` does.
pub fn region_stats(p: &[Point], n: &[Point], scan: &[i32], regions: &[Region], margin: f64, min_n: usize, r_max: f64) -> Vec<RegionStats> {
    use rayon::prelude::*;
    let grid = SurfelGrid::new(p);
    let pv: Vec<V> = p.iter().map(v).collect();
    let nv: Vec<V> = n.iter().map(v).collect();
    regions
        .par_iter()
        .map(|g| {
            let mut st = RegionStats::default();
            let c = v(&g.p);
            let idx = grid.gather(&pv, scan, g, margin);
            st.n = idx.len() as u32;
            let mut sc: Vec<i32> = idx.iter().map(|&i| scan[i]).collect();
            sc.sort_unstable();
            sc.dedup();
            st.n_scans = sc.len() as u32;
            if idx.len() < min_n {
                return st;
            }
            let k = idx.len() as f64;
            let t = idx.iter().map(|&i| nv[i] * nv[i].transpose()).sum::<Matrix3<f64>>() / k;
            let (lam, ev) = eig(&t);
            let tot = lam.iter().sum::<f64>();
            st.lam = lam.map(|x| x / tot);
            let a = ev[2];
            st.axis = pt(&a);
            // thickness about the positions' plane
            let m = idx.iter().map(|&i| pv[i]).sum::<V>() / k;
            let cov = idx.iter().map(|&i| (pv[i] - m) * (pv[i] - m).transpose()).sum::<Matrix3<f64>>() / k;
            let (_, pe) = eig(&cov);
            st.thick_plane = mad(idx.iter().map(|&i| (pv[i] - m).dot(&pe[2]).abs()).collect());
            // medial collapse, then again without the surfels far off the line
            let (mut r, mut s, s0, mut q) = collapse(&pv, &nv, &idx, &a, r_max);
            let off = |r: f64, q: &V, i: usize| {
                let d = pv[i] - nv[i] * r - q;
                (d - a * d.dot(&a)).norm()
            };
            let res: Vec<f64> = idx.iter().map(|&i| off(r, &q, i)).collect();
            let cut = 3.0 * mad(res.clone()).max(1e-4);
            let inl: Vec<usize> = idx.iter().zip(&res).filter(|(_, &x)| x <= cut).map(|(&i, _)| i).collect();
            if inl.len() >= min_n && inl.len() < idx.len() {
                (r, s, _, q) = collapse(&pv, &nv, &inl, &a, r_max);
            }
            st.medial_r = r;
            st.spread = s;
            st.spread0 = s0;
            // thickness about the medial cylinder (axis a through q, radius r)
            st.thick_cyl = mad(idx
                .iter()
                .map(|&i| {
                    let d = pv[i] - q;
                    ((d - a * d.dot(&a)).norm() - r).abs()
                })
                .collect());
            st.centre = pt(&(q + a * (c - q).dot(&a)));
            st
        })
        .collect()
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::shots::lattice::scan_lattice;
    use crate::shots::reconstruct::dir;

    const ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// A regular little scan (0.03 deg steps) of a vertical cylinder `r` at (0, d, 0) before a wall at 20 m, with
    /// `noise` (m) of uniform-ish range noise: the lattice, directions and ranges.
    pub(in crate::shots) fn scene(r: f64, d: f64, noise: f64) -> (Lattice, Vec<Point>, Vec<f32>) {
        let (period, interval, step) = (6.6e-3, 0.825e-6, 0.03);
        let omega = step / interval;
        let (mut t, mut dirs, mut rng) = (Vec::new(), Vec::new(), Vec::new());
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for l in 0..400usize {
            for k in 0..600usize {
                let zen = 81.0 + step * k as f64;
                let ti = l as f64 * period + (zen - 81.0) / omega;
                let azi = -6.0 + 0.03 * ti / period;
                let v = dir(zen, azi);
                let (a, b, c) = (v[0] * v[0] + v[1] * v[1], -2.0 * d * v[1], d * d - r * r);
                let disc = b * b - 4.0 * a * c;
                let hit = if disc >= 0.0 { (-b - disc.sqrt()) / (2.0 * a) } else { 20.0 };
                // a sum of three uniforms: about Gaussian, sd `noise`
                let e = (rnd() + rnd() + rnd() - 1.5) * 2.0 * noise;
                t.push(ti);
                dirs.push(v);
                rng.push((hit + e) as f32);
            }
        }
        (scan_lattice(&t, &dirs), dirs, rng)
    }

    #[test]
    fn a_cylinders_surfels_point_away_from_its_axis() {
        let (r, d) = (0.05, 8.0);
        let (lat, dirs, rng) = scene(r, d, 0.003);
        let map = ShadowMap::new(&lat, &rng, ID, [0.0; 3]);
        let on = |_: usize, p: &Point| (p[0].powi(2) + (p[1] - d).powi(2)).sqrt() < r + 0.01;
        let (s, _) = surfels(&map, &dirs, on, |t| 0.007 + 0.35e-3 * t, &SurfelParams::default());
        assert!(s.len() > 200, "{}", s.len());
        let mut err: Vec<f64> = s
            .iter()
            .map(|x| {
                let radial = V::new(x.p[0], x.p[1] - d, 0.0).normalize();
                v(&x.n).dot(&radial).clamp(-1.0, 1.0).acos().to_degrees()
            })
            .collect();
        err.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (med, p90) = (err[err.len() / 2], err[err.len() * 9 / 10]);
        let mut sig: Vec<f64> = s.iter().map(|x| x.sigma_n.to_degrees()).collect();
        let sig_med = super::super::reconstruct::median(&mut sig);
        eprintln!("synthetic cylinder r 5 cm at 8 m, 3 mm noise: {} surfels, normal error median {med:.2} deg, p90 {p90:.2} deg, sigma_n median {sig_med:.2} deg", s.len());
        assert!(med < 6.0 && p90 < 12.0 && med < 1.6 * sig_med, "{med} {p90} {sig_med}");
        // one view's arc: medial collapse finds R and the axis
        let p: Vec<Point> = s.iter().map(|x| x.p).collect();
        let n: Vec<Point> = s.iter().map(|x| x.n).collect();
        let sc = vec![0; p.len()];
        let g = Region { p: [0.0, d, 0.0], u: [0.0, 0.0, 1.0], half: 0.05, r, scan: -1 };
        let st = region_stats(&p, &n, &sc, &[g], 0.02, 10, 0.5)[0];
        eprintln!("one-view arc: R {:.4}, spread {:.4}, spread0 {:.4}, lam {:?}, axis {:?}, thick cyl {:.4} plane {:.4}", st.medial_r, st.spread, st.spread0, st.lam, st.axis, st.thick_cyl, st.thick_plane);
        assert!((st.medial_r - r).abs() < 0.01, "{}", st.medial_r);
        assert!(st.axis[2].abs() > 0.95);
        assert!(st.lam[2] < 0.02);
        assert!(st.thick_cyl < 0.006);
    }

    #[test]
    fn a_clump_neither_collapses_nor_is_thin() {
        // random surfels in a 5 cm ball, normals random over the sphere: rank 3, no collapse
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let (mut p, mut n) = (Vec::new(), Vec::new());
        while p.len() < 400 {
            let x = [rnd(), rnd(), rnd()];
            let y = V::new(rnd(), rnd(), rnd());
            if x.iter().map(|a| a * a).sum::<f64>() <= 1.0 && y.norm() <= 1.0 && y.norm() > 0.1 {
                p.push(x.map(|a| 0.05 * a));
                n.push(pt(&y.normalize()));
            }
        }
        let sc = vec![0; p.len()];
        let g = Region { p: [0.0; 3], u: [0.0, 0.0, 1.0], half: 0.03, r: 0.02, scan: -1 };
        let st = region_stats(&p, &n, &sc, &[g], 0.02, 10, 0.5)[0];
        eprintln!("clump: R {:.4}, spread {:.4}, spread0 {:.4}, lam {:?}, thick cyl {:.4} plane {:.4}", st.medial_r, st.spread, st.spread0, st.lam, st.thick_cyl, st.thick_plane);
        assert!(st.lam[2] > 0.2);
        assert!(st.spread > 0.7 * st.spread0);
        assert!(st.thick_cyl > 0.01 && st.thick_plane > 0.01);
    }
}
