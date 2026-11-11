// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The far-field completion field on a dense voxel box: the walk's first-order
//! (P1) form, a screened Poisson equation (3dtrees research 128, `fields.md`;
//! a spike, not a fit).
//!
//! - **ray counts:** each pulse is marched through the box (Amanatides-Woo).
//!   Per voxel and scan it counts rays whose first echo lies in the voxel
//!   (*hit*), beyond it (*pass*: the voxel was seen through), before it
//!   (*occluded*), and rays with no echo (*no return*, uninformative: sky or
//!   deleted in export).
//! - **node inputs from the moment octree** (`moments`), per voxel:
//!   - *visible wood:* a 3 cm node whose 9 cm neighbourhood (it and its 26
//!     neighbours) collapses (ratio < `wood_ratio` about the normal tensor's
//!     null direction, at least `min_n` surfels); the voxel keeps the largest
//!     R* and the count of such nodes;
//!   - *leaf area:* every 6 cm node with at least 3 surfels that does not
//!     collapse on its own adds `3 λ₃ · (6 cm)²`, λ₃ the normal tensor's
//!     smallest eigenvalue (sum 1): isotropic (rank-3) nodes count fully,
//!     planes and lines not at all. Occupancy, not surfel count, so scan
//!     density doesn't weigh it.
//! - **the solve:** `−∇·(κ∇φ) + σφ = s` by Jacobi-preconditioned conjugate
//!   gradients, matrix-free on the 7-point stencil, κ at a face the harmonic
//!   mean of its voxels'. Voxels with a finite `fixed` value are Dirichlet;
//!   the box's faces are closed (zero flux).
//! - **ownership** (research 128 `fields-plot.md`): the harmonic measure of
//!   each stem's sinks, one local solve per stem with every other stem's
//!   sinks, the ground and the box sides absorbing (random-walker
//!   segmentation).

use super::moments::{per_scan, Octree};
use crate::Point;
use nalgebra::Vector3;
use rayon::prelude::*;

/// A dense box of cubic voxels, x fastest.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub lo: Point,
    pub h: f64,
    pub n: [usize; 3],
}

impl Grid {
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n[1] + j) * self.n[0] + i
    }

    pub fn locate(&self, p: &Point) -> Option<usize> {
        let mut c = [0usize; 3];
        for a in 0..3 {
            let x = ((p[a] - self.lo[a]) / self.h).floor();
            if !(x >= 0.0 && (x as usize) < self.n[a]) {
                return None;
            }
            c[a] = x as usize;
        }
        Some(self.idx(c[0], c[1], c[2]))
    }
}

pub const HIT: usize = 0;
pub const PASS: usize = 1;
pub const OCCLUDED: usize = 2;
pub const NO_RETURN: usize = 3;

/// One scan's ray counts per voxel ([hit, pass, occluded, no return]). Pulse i's world ray is
/// `origin + t rot·dir[i]`, its first echo at `range[i]` (not finite or ≤ 0: no return). Counted into one shared
/// array of atomics, so memory stays one grid whatever the thread count (plot-scale boxes).
pub fn ray_counts(g: &Grid, origin: &Point, rot: &[[f64; 3]; 3], dir: &[Point], range: &[f32]) -> Vec<[u32; 4]> {
    use std::sync::atomic::{AtomicU32, Ordering};
    let hi = [0, 1, 2].map(|a| g.lo[a] + g.h * g.n[a] as f64);
    let acc: Vec<AtomicU32> = (0..4 * g.len()).into_par_iter().map(|_| AtomicU32::new(0)).collect();
    dir.par_iter().zip(range.par_iter()).with_min_len(1 << 12).for_each(|(d0, &r)| {
        let d = [0, 1, 2].map(|a| rot[a][0] * d0[0] + rot[a][1] * d0[1] + rot[a][2] * d0[2]);
        let nd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(nd > 0.0) {
            return;
        }
        let d = d.map(|x| x / nd);
        let r = if r.is_finite() && r > 0.0 { r as f64 } else { f64::INFINITY };
        march(g, origin, &d, &hi, r, &mut |i, o| {
            acc[4 * i + o].fetch_add(1, Ordering::Relaxed);
        });
    });
    acc.par_chunks(4).map(|c| [0, 1, 2, 3].map(|k| c[k].load(Ordering::Relaxed))).collect()
}

fn march(g: &Grid, o: &Point, d: &Point, hi: &[f64; 3], r: f64, add: &mut impl FnMut(usize, usize)) {
    // the slab test: the ray's span inside the box
    let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
    for a in 0..3 {
        if d[a].abs() < 1e-12 {
            if o[a] < g.lo[a] || o[a] > hi[a] {
                return;
            }
        } else {
            let (u, v) = ((g.lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
    }
    if t1 <= t0 {
        return;
    }
    let p = [0, 1, 2].map(|a| o[a] + (t0 + 1e-9) * d[a]);
    let mut c = [0i64; 3];
    let (mut step, mut tmax, mut tdel) = ([0i64; 3], [f64::INFINITY; 3], [f64::INFINITY; 3]);
    for a in 0..3 {
        c[a] = (((p[a] - g.lo[a]) / g.h).floor() as i64).clamp(0, g.n[a] as i64 - 1);
        if d[a] > 1e-12 {
            step[a] = 1;
            tmax[a] = (g.lo[a] + (c[a] + 1) as f64 * g.h - o[a]) / d[a];
            tdel[a] = g.h / d[a];
        } else if d[a] < -1e-12 {
            step[a] = -1;
            tmax[a] = (g.lo[a] + c[a] as f64 * g.h - o[a]) / d[a];
            tdel[a] = -g.h / d[a];
        }
    }
    let mut ta = t0;
    loop {
        let a = if tmax[0] < tmax[1] { if tmax[0] < tmax[2] { 0 } else { 2 } } else if tmax[1] < tmax[2] { 1 } else { 2 };
        let tb = tmax[a].min(t1);
        let i = g.idx(c[0] as usize, c[1] as usize, c[2] as usize);
        let o = if r.is_infinite() {
            NO_RETURN
        } else if r < ta {
            OCCLUDED
        } else if r < tb {
            HIT
        } else {
            PASS
        };
        add(i, o);
        if tb >= t1 {
            return;
        }
        ta = tb;
        c[a] += step[a];
        if c[a] < 0 || c[a] >= g.n[a] as i64 {
            return;
        }
        tmax[a] += tdel[a];
    }
}

/// Per voxel: (leaf area m², largest visible-wood R* m, visible-wood 3 cm nodes) from surfels on a 3 cm moment
/// octree, as the module docs say.
pub fn node_inputs(g: &Grid, p: &[Point], n: &[Point], scan: &[i32], min_n: f64, wood_ratio: f64, r_max: f64) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let vv = |x: &[Point]| x.iter().map(|a| Vector3::new(a[0], a[1], a[2])).collect::<Vec<_>>();
    let (pv, nv) = (vv(p), vv(n));
    let o = Octree::new(&per_scan(&pv, &nv, scan, 0.03), 0.03, 2);
    let (mut leaf, mut wood_r, mut wood_n) = (vec![0.0; g.len()], vec![0.0; g.len()], vec![0u32; g.len()]);
    let centre = |k: &[i32; 3], cell: f64| [0, 1, 2].map(|a| (k[a] as f64 + 0.5) * cell);
    let wood: Vec<([i32; 3], f64)> = o.levels[0]
        .par_iter()
        .filter_map(|(k, _)| {
            let c = centre(k, 0.03);
            let (m, _) = o.query(0, &Vector3::new(c[0], c[1], c[2]), true);
            if m.n < min_n {
                return None;
            }
            let (_, u) = m.tensor();
            let (r, sp, s0) = m.collapse(&u, r_max);
            (s0 > 0.0 && sp / s0 < wood_ratio).then_some((*k, r))
        })
        .collect();
    for (k, r) in wood {
        if let Some(i) = g.locate(&centre(&k, 0.03)) {
            wood_r[i] = f64::max(wood_r[i], r);
            wood_n[i] += 1;
        }
    }
    for (k, (m, _)) in &o.levels[1] {
        if m.n < 3.0 {
            continue;
        }
        let (lam, u) = m.tensor();
        let (_, sp, s0) = m.collapse(&u, r_max);
        if s0 > 0.0 && sp / s0 < wood_ratio {
            continue;
        }
        if let Some(i) = g.locate(&centre(k, 0.06)) {
            leaf[i] += 3.0 * lam[2] * 0.06 * 0.06;
        }
    }
    (leaf, wood_r, wood_n)
}

/// `−∇·(κ∇φ) + σφ = s` on the free voxels (`fixed` NaN); returns (φ with the fixed values in place, iterations,
/// relative residual).
pub fn solve(g: &Grid, kappa: &[f64], sigma: &[f64], s: &[f64], fixed: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize, f64) {
    let nv = g.len();
    let h2 = g.h * g.h;
    let free: Vec<bool> = fixed.iter().map(|x| !x.is_finite()).collect();
    let face = |i: usize, j: usize| 2.0 * kappa[i] * kappa[j] / (kappa[i] + kappa[j]).max(1e-300) / h2;
    let nbrs = |i: usize| {
        let (x, y, z) = (i % g.n[0], (i / g.n[0]) % g.n[1], i / (g.n[0] * g.n[1]));
        let mut out = [usize::MAX; 6];
        if x > 0 { out[0] = i - 1 }
        if x + 1 < g.n[0] { out[1] = i + 1 }
        if y > 0 { out[2] = i - g.n[0] }
        if y + 1 < g.n[1] { out[3] = i + g.n[0] }
        if z > 0 { out[4] = i - g.n[0] * g.n[1] }
        if z + 1 < g.n[2] { out[5] = i + g.n[0] * g.n[1] }
        out
    };
    // the diagonal, and the fixed neighbours moved to the right-hand side
    let (diag, b): (Vec<f64>, Vec<f64>) = (0..nv)
        .into_par_iter()
        .map(|i| {
            if !free[i] {
                return (1.0, 0.0);
            }
            let (mut dg, mut bi) = (sigma[i], s[i]);
            for j in nbrs(i) {
                if j != usize::MAX {
                    let f = face(i, j);
                    dg += f;
                    if !free[j] {
                        bi += f * fixed[j];
                    }
                }
            }
            (dg.max(1e-300), bi)
        })
        .unzip();
    let apply = |x: &[f64]| -> Vec<f64> {
        (0..nv)
            .into_par_iter()
            .map(|i| {
                if !free[i] {
                    return 0.0;
                }
                let mut y = diag[i] * x[i];
                for j in nbrs(i) {
                    if j != usize::MAX && free[j] {
                        y -= face(i, j) * x[j];
                    }
                }
                y
            })
            .collect()
    };
    let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut x = vec![0.0; nv];
    let mut r = b.clone();
    let bn = dot(&b, &b).sqrt().max(1e-300);
    let mut z: Vec<f64> = r.iter().zip(&diag).map(|(a, d)| a / d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut it = 0;
    let mut res = dot(&r, &r).sqrt() / bn;
    while it < max_iter && res > tol {
        let ap = apply(&p);
        let alpha = rz / dot(&p, &ap).max(1e-300);
        x.par_iter_mut().zip(&p).for_each(|(a, b)| *a += alpha * b);
        r.par_iter_mut().zip(&ap).for_each(|(a, b)| *a -= alpha * b);
        z = r.iter().zip(&diag).map(|(a, d)| a / d).collect();
        let rz1 = dot(&r, &z);
        let beta = rz1 / rz;
        rz = rz1;
        p.par_iter_mut().zip(&z).for_each(|(a, b)| *a = b + beta * *a);
        it += 1;
        res = dot(&r, &r).sqrt() / bn;
    }
    for i in 0..nv {
        if !free[i] {
            x[i] = fixed[i];
        }
    }
    (x, it, res)
}

/// Soft ownership by harmonic measure (random-walker segmentation), one local solve per stem: on a box of
/// half-width `half` voxels in x and y (full height) about the stem's first sink voxel, `u` solves
/// `−∇·(κ∇u) + σu = 0` with u = 1 on the stem's own `sinks`, 0 on every other stem's sinks, on `ground` and on
/// the box's side faces (absorbing: walks that leave are lost). Returns, for each `targets` voxel inside a stem's
/// box with u > 0, (target position, stem, u); the caller normalises over stems.
#[allow(clippy::too_many_arguments)]
pub fn ownership(g: &Grid, kappa: &[f64], sigma: &[f64], ground: &[bool], sinks: &[Vec<usize>], half: usize, targets: &[usize], tol: f64) -> Vec<(u32, u32, f64)> {
    let mut sink_of = vec![u32::MAX; g.len()];
    for (s, vs) in sinks.iter().enumerate() {
        for &v in vs {
            sink_of[v] = s as u32;
        }
    }
    let mut target_of = vec![u32::MAX; g.len()];
    for (t, &v) in targets.iter().enumerate() {
        target_of[v] = t as u32;
    }
    sinks
        .par_iter()
        .enumerate()
        .filter(|(_, vs)| !vs.is_empty())
        .flat_map_iter(|(s, vs)| {
            let c = (vs[0] % g.n[0], (vs[0] / g.n[0]) % g.n[1]);
            let (i0, i1) = (c.0.saturating_sub(half), (c.0 + half + 1).min(g.n[0]));
            let (j0, j1) = (c.1.saturating_sub(half), (c.1 + half + 1).min(g.n[1]));
            let sub = Grid { lo: [g.lo[0] + g.h * i0 as f64, g.lo[1] + g.h * j0 as f64, g.lo[2]], h: g.h, n: [i1 - i0, j1 - j0, g.n[2]] };
            let mut map = Vec::with_capacity(sub.len());
            for k in 0..g.n[2] {
                for j in j0..j1 {
                    for i in i0..i1 {
                        map.push(g.idx(i, j, k));
                    }
                }
            }
            let fixed: Vec<f64> = map
                .iter()
                .enumerate()
                .map(|(q, &v)| {
                    let (i, j) = (q % sub.n[0], (q / sub.n[0]) % sub.n[1]);
                    if sink_of[v] != u32::MAX && (sink_of[v] == s as u32 || vs.contains(&v)) {
                        1.0
                    } else if ground[v] || sink_of[v] != u32::MAX || i == 0 || j == 0 || i + 1 == sub.n[0] || j + 1 == sub.n[1] {
                        0.0
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let k: Vec<f64> = map.iter().map(|&v| kappa[v]).collect();
            let sg: Vec<f64> = map.iter().map(|&v| sigma[v]).collect();
            let (u, _, _) = solve(&sub, &k, &sg, &vec![0.0; sub.len()], &fixed, tol, 5000);
            map.into_iter()
                .zip(u)
                .filter_map(|(v, x)| (target_of[v] != u32::MAX && x > 0.0).then_some((target_of[v], s as u32, x)))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ray_counts_pass_hit_occluded() {
        let g = Grid { lo: [0.0, 0.0, 0.0], h: 1.0, n: [5, 1, 1] };
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // from x = -1 along +x, echo at 3.5 (x = 2.5): voxels 0, 1 seen through, 2 hit, 3, 4 occluded
        let c = ray_counts(&g, &[-1.0, 0.5, 0.5], &id, &[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &[3.5, f32::NAN]);
        let want = [[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]];
        assert_eq!(c, want);
    }

    #[test]
    fn screened_poisson_matches_the_1d_solution() {
        // φ(0) = 1 (fixed), closed at x = L: φ = cosh((L - x)/ℓ) / cosh(L/ℓ), ℓ = sqrt(κ/σ)
        let n = 200;
        let h = 0.01;
        let g = Grid { lo: [0.0; 3], h, n: [n, 1, 1] };
        let (kappa, sigma) = (0.5, 50.0);
        let mut fixed = vec![f64::NAN; n];
        fixed[0] = 1.0;
        let (phi, _, res) = solve(&g, &vec![kappa; n], &vec![sigma; n], &vec![0.0; n], &fixed, 1e-12, 10_000);
        assert!(res < 1e-10);
        let l = (kappa / sigma).sqrt();
        let len = (n - 1) as f64 * h + 0.5 * h;
        for i in [10, 50, 100] {
            let x = i as f64 * h;
            let want = ((len - x) / l).cosh() / (len / l).cosh();
            assert!((phi[i] - want).abs() < 2e-3, "{i}: {} vs {want}", phi[i]);
        }
    }

    #[test]
    fn ownership_is_symmetric_and_favours_the_nearer_stem() {
        // two stems at x = 5 and x = 15 on a 21 x 9 x 5 box, no ground: the midpoint is owned equally, x = 7 by stem 0
        let g = Grid { lo: [0.0; 3], h: 1.0, n: [21, 9, 5] };
        let nv = g.len();
        let sinks = vec![vec![g.idx(5, 4, 2)], vec![g.idx(15, 4, 2)]];
        let targets = [g.idx(10, 4, 2), g.idx(7, 4, 2)];
        let o = ownership(&g, &vec![1.0; nv], &vec![0.1; nv], &vec![false; nv], &sinks, 8, &targets, 1e-12);
        let u = |t: u32, s: u32| o.iter().find(|x| x.0 == t && x.1 == s).map_or(0.0, |x| x.2);
        assert!((u(0, 0) - u(0, 1)).abs() < 1e-9 * u(0, 0).max(1e-300), "{} vs {}", u(0, 0), u(0, 1));
        assert!(u(1, 0) > 5.0 * u(1, 1));
    }
}
