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
//!   deleted in export), and the pass rays' direction tensor `Σ ωωᵀ`, additive
//!   like the counts (`fields-aniso.md`: the killing of a hidden pipe along u
//!   is `∝ Σ_pass |ω × u| = Σ √(1 − (ω·u)²)`).
//! - **node inputs from the moment octree** (`moments`), per voxel:
//!   - *visible wood:* a 3 cm node whose 9 cm neighbourhood (it and its 26
//!     neighbours) collapses (ratio < `wood_ratio` about the normal tensor's
//!     null direction, at least `min_n` surfels); the voxel keeps the largest
//!     R* and the count of such nodes;
//!   - *leaf area:* every 6 cm node with at least 3 surfels that does not
//!     collapse on its own adds `3 λ₃ · (6 cm)²`, λ₃ the normal tensor's
//!     smallest eigenvalue (sum 1): isotropic (rank-3) nodes count fully,
//!     planes and lines not at all. Occupancy, not surfel count, so scan
//!     density doesn't weigh it;
//!   - *soft wood orientation:* every 3 cm node with at least `min_n`
//!     surfels in its 9 cm neighbourhood gets `p_w = 1/(1 + exp((ratio −
//!     soft_c)/soft_s))` from its collapse ratio; the voxel sums `p_w` and
//!     `p_w uuᵀ` (u the null direction) and keeps the largest `p_w`.
//! - **the solve:** `−∇·(κ∇φ) + σφ = s` by Jacobi-preconditioned conjugate
//!   gradients, matrix-free on the 7-point stencil, κ at a face the harmonic
//!   mean of its voxels'. κ is a scalar per voxel or a diagonal tensor (the
//!   per-axis conductances κ_xx, κ_yy, κ_zz): a face normal to axis a uses
//!   κ_aa, and the off-diagonal terms are dropped. Voxels with a finite `fixed` value are Dirichlet;
//!   the box's faces are closed (zero flux).

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

/// A voxel's ray counts ([hit, pass, occluded, no return]) and its pass rays' Σ ωωᵀ (xx, yy, zz, xy, xz, yz).
pub type Rays = ([u32; 4], [f64; 6]);

/// One scan's [`Rays`] per voxel. Pulse i's world ray is `origin + t rot·dir[i]`, its first echo at `range[i]` (not
/// finite or ≤ 0: no return).
pub fn ray_counts(g: &Grid, origin: &Point, rot: &[[f64; 3]; 3], dir: &[Point], range: &[f32]) -> Vec<Rays> {
    let hi = [0, 1, 2].map(|a| g.lo[a] + g.h * g.n[a] as f64);
    let zero: Rays = ([0; 4], [0.0; 6]);
    dir.par_chunks(1 << 16)
        .zip(range.par_chunks(1 << 16))
        .fold(
            || vec![zero; g.len()],
            |mut acc, (ds, rs)| {
                for (d0, &r) in ds.iter().zip(rs) {
                    let d = [0, 1, 2].map(|a| rot[a][0] * d0[0] + rot[a][1] * d0[1] + rot[a][2] * d0[2]);
                    let nd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if !(nd > 0.0) {
                        continue;
                    }
                    let d = d.map(|x| x / nd);
                    let r = if r.is_finite() && r > 0.0 { r as f64 } else { f64::INFINITY };
                    march(g, origin, &d, &hi, r, &mut acc);
                }
                acc
            },
        )
        .reduce(
            || vec![zero; g.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    for k in 0..4 {
                        x.0[k] += y.0[k];
                    }
                    for k in 0..6 {
                        x.1[k] += y.1[k];
                    }
                }
                a
            },
        )
}

fn march(g: &Grid, o: &Point, d: &Point, hi: &[f64; 3], r: f64, acc: &mut [Rays]) {
    let dd = [d[0] * d[0], d[1] * d[1], d[2] * d[2], d[0] * d[1], d[0] * d[2], d[1] * d[2]];
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
        acc[i].0[o] += 1;
        if o == PASS {
            for k in 0..6 {
                acc[i].1[k] += dd[k];
            }
        }
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

/// Per voxel inputs from the moment octree, as the module docs say.
pub struct Nodes {
    /// Leaf area (m²), the largest visible-wood R* (m), visible-wood 3 cm nodes.
    pub leaf: Vec<f64>,
    pub wood_r: Vec<f64>,
    pub wood_n: Vec<u32>,
    /// Soft wood: Σ p_w, the largest p_w, Σ p_w uuᵀ (xx, yy, zz, xy, xz, yz).
    pub soft_sum: Vec<f64>,
    pub soft_max: Vec<f64>,
    pub soft_t: Vec<[f64; 6]>,
}

/// [`Nodes`] from surfels on a 3 cm moment octree.
#[allow(clippy::too_many_arguments)]
pub fn node_inputs(g: &Grid, p: &[Point], n: &[Point], scan: &[i32], min_n: f64, wood_ratio: f64, r_max: f64, soft_c: f64, soft_s: f64) -> Nodes {
    let vv = |x: &[Point]| x.iter().map(|a| Vector3::new(a[0], a[1], a[2])).collect::<Vec<_>>();
    let (pv, nv) = (vv(p), vv(n));
    let o = Octree::new(&per_scan(&pv, &nv, scan, 0.03), 0.03, 2);
    let (mut leaf, mut wood_r, mut wood_n) = (vec![0.0; g.len()], vec![0.0; g.len()], vec![0u32; g.len()]);
    let centre = |k: &[i32; 3], cell: f64| [0, 1, 2].map(|a| (k[a] as f64 + 0.5) * cell);
    // per 3 cm node with enough surfels around it: (key, R*, ratio, u)
    let nodes: Vec<([i32; 3], f64, f64, Vector3<f64>)> = o.levels[0]
        .par_iter()
        .filter_map(|(k, _)| {
            let c = centre(k, 0.03);
            let (m, _) = o.query(0, &Vector3::new(c[0], c[1], c[2]), true);
            if m.n < min_n {
                return None;
            }
            let (_, u) = m.tensor();
            let (r, sp, s0) = m.collapse(&u, r_max);
            (s0 > 0.0).then_some((*k, r, sp / s0, u))
        })
        .collect();
    let (mut soft_sum, mut soft_max, mut soft_t) = (vec![0.0; g.len()], vec![0.0; g.len()], vec![[0.0; 6]; g.len()]);
    for (k, r, ratio, u) in nodes {
        if let Some(i) = g.locate(&centre(&k, 0.03)) {
            if ratio < wood_ratio {
                wood_r[i] = f64::max(wood_r[i], r);
                wood_n[i] += 1;
            }
            let pw = 1.0 / (1.0 + ((ratio - soft_c) / soft_s).exp());
            soft_sum[i] += pw;
            soft_max[i] = f64::max(soft_max[i], pw);
            let uu = [u[0] * u[0], u[1] * u[1], u[2] * u[2], u[0] * u[1], u[0] * u[2], u[1] * u[2]];
            for a in 0..6 {
                soft_t[i][a] += pw * uu[a];
            }
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
    Nodes { leaf, wood_r, wood_n, soft_sum, soft_max, soft_t }
}

/// `−∇·(κ∇φ) + σφ = s` on the free voxels (`fixed` NaN); κ has one value per voxel, or three (κ_xx, κ_yy, κ_zz,
/// voxel-major). Returns (φ with the fixed values in place, iterations, relative residual).
pub fn solve(g: &Grid, kappa: &[f64], sigma: &[f64], s: &[f64], fixed: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize, f64) {
    let nv = g.len();
    let h2 = g.h * g.h;
    let free: Vec<bool> = fixed.iter().map(|x| !x.is_finite()).collect();
    let tensor = kappa.len() == 3 * nv;
    // the face between i and its neighbour j across axis a (nbrs' slot / 2)
    let face = |i: usize, j: usize, a: usize| {
        let (ki, kj) = if tensor { (kappa[3 * i + a], kappa[3 * j + a]) } else { (kappa[i], kappa[j]) };
        2.0 * ki * kj / (ki + kj).max(1e-300) / h2
    };
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
            for (slot, j) in nbrs(i).into_iter().enumerate() {
                if j != usize::MAX {
                    let f = face(i, j, slot / 2);
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
                for (slot, j) in nbrs(i).into_iter().enumerate() {
                    if j != usize::MAX && free[j] {
                        y -= face(i, j, slot / 2) * x[j];
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
        assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), want);
        // the pass tensor holds one ωωᵀ = e_x e_xᵀ per pass ray
        for (x, w) in c.iter().zip(&want) {
            assert_eq!(x.1, [w[1] as f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        // an oblique ray: Σ ωωᵀ has trace equal to the pass count, and xy = n/2
        let s2 = 0.5f64.sqrt();
        let g2 = Grid { lo: [0.0, 0.0, 0.0], h: 1.0, n: [3, 3, 1] };
        let c2 = ray_counts(&g2, &[0.01, 0.0, 0.5], &id, &[[s2, s2, 0.0]], &[100.0]);
        assert!(c2.iter().map(|x| x.0[PASS]).sum::<u32>() >= 3);
        for x in &c2 {
            let n = x.0[PASS] as f64;
            assert!((x.1[0] + x.1[1] + x.1[2] - n).abs() < 1e-12 && (x.1[3] - 0.5 * n).abs() < 1e-12);
        }
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
    fn a_diagonal_tensor_conductance_uses_each_axis_own_value() {
        // the 1-D problem along x with κ_xx; large κ_yy, κ_zz change nothing on a line of voxels, and the same
        // problem laid along y uses κ_yy
        let n = 200;
        let h = 0.01;
        let (kx, sigma) = (0.5, 50.0);
        let mut fixed = vec![f64::NAN; n];
        fixed[0] = 1.0;
        let gx = Grid { lo: [0.0; 3], h, n: [n, 1, 1] };
        let gy = Grid { lo: [0.0; 3], h, n: [1, n, 1] };
        let kt_x: Vec<f64> = (0..n).flat_map(|_| [kx, 9.0, 9.0]).collect();
        let kt_y: Vec<f64> = (0..n).flat_map(|_| [9.0, kx, 9.0]).collect();
        let (a, _, _) = solve(&gx, &vec![kx; n], &vec![sigma; n], &vec![0.0; n], &fixed, 1e-12, 10_000);
        let (b, _, _) = solve(&gx, &kt_x, &vec![sigma; n], &vec![0.0; n], &fixed, 1e-12, 10_000);
        let (c, _, _) = solve(&gy, &kt_y, &vec![sigma; n], &vec![0.0; n], &fixed, 1e-12, 10_000);
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-9 && (a[i] - c[i]).abs() < 1e-9);
        }
    }
}
