// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Additive surfel moments per (voxel, scan), and the elements read off them (3dtrees research 128 `octree.md`,
//! research 130 slice 2).
//!
//! - **Moments** of a set of surfels (p, n): count, `Σp`, `Σppᵀ`, `Σn`, `Σnnᵀ`, `Σpnᵀ` and the line matrix `Σaaᵀ`
//!   with `a = (p × n, n)`. They are sums, so sets merge by adding them.
//! - **Rigid motions act on them.** A translation t sends `a` to `Ad(t) a` with `Ad(t) = [[I, [t]×], [0, I]]` (and
//!   `Σp`, `Σppᵀ`, `Σpnᵀ` move as p + t does). A rotation R acts as `diag(R, R)`.
//!   - **Invariant parts** (`Σnnᵀ`, so the normal tensor and the line complex's direction block) fuse across scans
//!     whatever their offsets.
//!   - **Covariant parts** (positions, moments of lines) stay per scan in [`Moments`], so a scan's offset field moves
//!     them by `Ad(t)` before they fuse.
//! - **Estimators in O(1)** from a sum of moments:
//!   - **medial collapse** about u: `R* = B/C` (the spread of `P(p − Rn)` is `A − 2RB + R²C`);
//!   - **the line-complex axis:** `s² Σnnᵀ` added for the cylinder's `n·u = 0`, m eliminated by a Schur complement,
//!     u the 3 × 3's smallest eigenvector;
//!   - **the normal tensor's** eigenvalues.
//! - **Candidates** ([`Moments::candidates`]): each occupied voxel with its 26 neighbours (a 9 cm cube at 3 cm
//!   voxels, octree.md's best) read out as a candidate element. It gets an axis, collapse R* and ratio,
//!   normal-tensor eigenvalues, an extent, and per scan the arc its normals cover around the axis (the support
//!   function's coverage). These are marginals of one neighbourhood, labelled as such: cached potentials and the
//!   proposal support, not estimates.

use nalgebra::{Matrix3, Matrix6, SymmetricEigen, Vector3, Vector6};
use std::collections::HashMap;

type V = Vector3<f64>;
type M3 = Matrix3<f64>;
pub type Key = [i32; 3];

/// A symmetric 3 × 3's eigenvalues, descending, and their vectors.
fn eig(m: &M3) -> ([f64; 3], [V; 3]) {
    let e = SymmetricEigen::new(*m);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e.eigenvalues[b].total_cmp(&e.eigenvalues[a]));
    (idx.map(|i| e.eigenvalues[i]), idx.map(|i| e.eigenvectors.column(i).into_owned()))
}

fn skew(t: &V) -> M3 {
    M3::new(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mom {
    pub n: f64,
    pub sp: V,
    pub spp: M3,
    pub sn: V,
    pub snn: M3,
    /// `Σ p nᵀ`.
    pub spn: M3,
    /// `Σ a aᵀ`, `a = (p × n, n)`.
    pub saa: Matrix6<f64>,
}

impl Mom {
    pub fn zero() -> Self {
        Mom { n: 0.0, sp: V::zeros(), spp: M3::zeros(), sn: V::zeros(), snn: M3::zeros(), spn: M3::zeros(), saa: Matrix6::zeros() }
    }

    pub fn add(&mut self, p: &V, n: &V) {
        let q = p.cross(n);
        let a = Vector6::new(q[0], q[1], q[2], n[0], n[1], n[2]);
        self.n += 1.0;
        self.sp += p;
        self.spp += p * p.transpose();
        self.sn += n;
        self.snn += n * n.transpose();
        self.spn += p * n.transpose();
        self.saa += a * a.transpose();
    }

    pub fn of(p: &[V], n: &[V]) -> Self {
        let mut m = Mom::zero();
        for (x, y) in p.iter().zip(n) {
            m.add(x, y);
        }
        m
    }

    pub fn merge(&mut self, o: &Mom) {
        self.n += o.n;
        self.sp += o.sp;
        self.spp += o.spp;
        self.sn += o.sn;
        self.snn += o.snn;
        self.spn += o.spn;
        self.saa += o.saa;
    }

    /// The moments of the surfels moved by t (p → p + t).
    pub fn moved(&self, t: &V) -> Mom {
        let mut ad = Matrix6::identity();
        ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(t));
        Mom {
            n: self.n,
            sp: self.sp + t * self.n,
            spp: self.spp + self.sp * t.transpose() + t * self.sp.transpose() + t * t.transpose() * self.n,
            sn: self.sn,
            snn: self.snn,
            spn: self.spn + t * self.sn.transpose(),
            saa: ad * self.saa * ad.transpose(),
        }
    }

    /// Medial collapse about unit u: (R* = B/C clamped to 0..r_max, spread at R*, spread at 0), spreads as rms (m).
    pub fn collapse(&self, u: &V, r_max: f64) -> (f64, f64, f64) {
        let p = M3::identity() - u * u.transpose();
        let k = self.n;
        let (pp, pn) = (p * self.sp, p * self.sn);
        let a = ((p * self.spp).trace() - pp.dot(&pp) / k) / k;
        let b = ((p * self.spn).trace() - pp.dot(&pn) / k) / k;
        let c = ((p * self.snn).trace() - pn.dot(&pn) / k) / k;
        let r = if c > 1e-6 { (b / c).clamp(0.0, r_max) } else { 0.0 };
        (r, (a - 2.0 * r * b + r * r * c).max(0.0).sqrt(), a.max(0.0).sqrt())
    }

    /// The normal tensor's eigenvalues (descending, sum 1).
    pub fn tensor(&self) -> [f64; 3] {
        let (lam, _) = eig(&(self.snn / self.n));
        let t = lam.iter().sum::<f64>();
        lam.map(|x| x / t)
    }

    /// The line-complex axis, coordinates about `c0`, with the cylinder rows `s (n, 0)`: (u, the axis point nearest
    /// c0, mean squared residual).
    pub fn axis(&self, c0: &V, s: f64) -> Option<(V, V, f64)> {
        if self.n < 6.0 {
            return None;
        }
        let m = self.moved(&-c0).saa;
        let a = m.fixed_view::<3, 3>(0, 0) + m.fixed_view::<3, 3>(3, 3) * (s * s);
        let b = m.fixed_view::<3, 3>(0, 3).into_owned();
        let c = m.fixed_view::<3, 3>(3, 3).into_owned();
        let ci = (c + M3::identity() * (1e-9 * c.trace())).try_inverse()?;
        let sc = a - b * ci * b.transpose();
        let (lam, ev) = eig(&((sc + sc.transpose()) * 0.5));
        let mut u = ev[2];
        if u[2] < 0.0 {
            u = -u;
        }
        let mut mm = -(ci * b.transpose() * u);
        mm -= u * u.dot(&mm);
        Some((u, c0 + u.cross(&mm), lam[2].max(0.0) / self.n))
    }
}

pub fn key(p: &V, cell: f64) -> Key {
    [(p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32]
}

/// A candidate element read off one neighbourhood: its centre and axis (u, through `point`), collapse R* and the
/// ratio spread(R*)/spread(0), the normal tensor's eigenvalues, its surfel count; and per scan (in the order given)
/// its surfels and the arc (rad) its normals cover around the axis.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub key: Key,
    pub point: V,
    pub u: V,
    pub r: f64,
    pub ratio: f64,
    pub lam: [f64; 3],
    pub n: f64,
    /// The neighbourhood's extent: the sd of its surfels' positions along their widest direction (m).
    pub extent: f64,
    pub scan_n: Vec<f64>,
    pub arc: Vec<f64>,
}

/// Per (voxel, scan) moments: covariant parts kept per scan, fused on demand.
pub struct Moments {
    pub voxel: f64,
    pub cells: HashMap<Key, Vec<Mom>>,
    pub n_scans: usize,
}

impl Moments {
    /// From each scan's surfels (p, n), on `voxel` cells.
    pub fn new(scans: &[(Vec<V>, Vec<V>)], voxel: f64) -> Self {
        let k = scans.len();
        let mut cells: HashMap<Key, Vec<Mom>> = HashMap::new();
        for (s, (p, n)) in scans.iter().enumerate() {
            for (x, y) in p.iter().zip(n) {
                cells.entry(key(x, voxel)).or_insert_with(|| vec![Mom::zero(); k])[s].add(x, y);
            }
        }
        Moments { voxel, cells, n_scans: k }
    }

    /// The 27-cell neighbourhood of `k`, per scan.
    pub fn neighbourhood(&self, k: &Key) -> Vec<Mom> {
        let mut out = vec![Mom::zero(); self.n_scans];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ms) = self.cells.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for (o, m) in out.iter_mut().zip(ms) {
                            o.merge(m);
                        }
                    }
                }
            }
        }
        out
    }

    /// Candidates at every voxel whose neighbourhood holds at least `min_n` surfels, in key order.
    pub fn candidates(&self, min_n: f64, r_max: f64) -> Vec<Candidate> {
        use rayon::prelude::*;
        let mut keys: Vec<Key> = self.cells.keys().copied().collect();
        keys.sort();
        keys.par_iter()
            .filter_map(|k| {
                let per = self.neighbourhood(k);
                let mut all = Mom::zero();
                for m in &per {
                    all.merge(m);
                }
                if all.n < min_n {
                    return None;
                }
                let c = all.sp / all.n;
                let (u, point, _) = all.axis(&c, self.voxel)?;
                let (r, s1, s0) = all.collapse(&u, r_max);
                let e1 = u.cross(&if u[2].abs() < 0.9 { V::z() } else { V::x() }).normalize();
                let e2 = u.cross(&e1);
                // a scan's arc: the circular spread of its normals' angle about the axis, as the width of a uniform
                // arc with the same mean resultant length (R = sin(w/2) / (w/2)), from Σn alone
                let arc = per
                    .iter()
                    .map(|m| {
                        if m.n < 3.0 {
                            return 0.0;
                        }
                        let rl = ((m.sn.dot(&e1)).powi(2) + (m.sn.dot(&e2)).powi(2)).sqrt() / m.n;
                        let (mut lo, mut hi) = (1e-6, std::f64::consts::TAU);
                        for _ in 0..40 {
                            let w = 0.5 * (lo + hi);
                            if (w / 2.0).sin() / (w / 2.0) > rl { lo = w } else { hi = w }
                        }
                        0.5 * (lo + hi)
                    })
                    .collect();
                Some(Candidate {
                    key: *k, point, u, r, ratio: if s0 > 0.0 { s1 / s0 } else { f64::NAN }, lam: all.tensor(), n: all.n,
                    extent: eig(&(all.spp / all.n - c * c.transpose())).0[0].max(0.0).sqrt(),
                    scan_n: per.iter().map(|m| m.n).collect(),
                    arc,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Surfels on a cylinder (axis through c along u, radius r) with `noise` on the normals, from `views` scans each
    /// seeing a 120° arc.
    fn cylinder(c: V, u: V, r: f64, noise: f64, views: usize, per: usize, seed: u64) -> Vec<(Vec<V>, Vec<V>)> {
        let mut rnd = rng(seed);
        let e1 = u.cross(&V::new(0.3, 0.2, 0.9)).normalize();
        let e2 = u.cross(&e1);
        (0..views)
            .map(|v| {
                let phi0 = v as f64 * std::f64::consts::TAU / views as f64;
                (0..per)
                    .map(|_| {
                        let phi = phi0 + rnd() * std::f64::consts::FRAC_PI_3;
                        let nr = e1 * phi.cos() + e2 * phi.sin();
                        (c + u * (rnd() * 0.05) + nr * r, (nr + V::new(rnd(), rnd(), rnd()) * noise).normalize())
                    })
                    .unzip()
            })
            .collect()
    }

    /// The moments of surfels rotated by r about the origin: diag(R, R) on the line matrix, conjugation on the
    /// tensors (synthesis §13's rotation; the invariant the test checks).
    fn rotated(m: &Mom, r: &M3) -> Mom {
        let mut d = Matrix6::zeros();
        d.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
        d.fixed_view_mut::<3, 3>(3, 3).copy_from(r);
        Mom {
            n: m.n,
            sp: r * m.sp,
            spp: r * m.spp * r.transpose(),
            sn: r * m.sn,
            snn: r * m.snn * r.transpose(),
            spn: r * m.spn * r.transpose(),
            saa: d * m.saa * d.transpose(),
        }
    }

    fn close(a: &Mom, b: &Mom, tol: f64) -> bool {
        let d = |x: &f64, y: &f64| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs()));
        d(&a.n, &b.n)
            && a.sp.iter().zip(b.sp.iter()).all(|(x, y)| d(x, y))
            && a.spp.iter().zip(b.spp.iter()).all(|(x, y)| d(x, y))
            && a.snn.iter().zip(b.snn.iter()).all(|(x, y)| d(x, y))
            && a.spn.iter().zip(b.spn.iter()).all(|(x, y)| d(x, y))
            && a.saa.iter().zip(b.saa.iter()).all(|(x, y)| d(x, y))
    }

    #[test]
    fn merging_does_not_depend_on_order_and_moments_move_with_the_surfels() {
        let scans = cylinder(V::new(3.0, -2.0, 1.5), V::new(0.1, 0.2, 1.0).normalize(), 0.03, 0.2, 4, 300, 1);
        let per: Vec<Mom> = scans.iter().map(|(p, n)| Mom::of(p, n)).collect();
        let (mut a, mut b) = (Mom::zero(), Mom::zero());
        for i in [0, 1, 2, 3] {
            a.merge(&per[i]);
        }
        for i in [2, 0, 3, 1] {
            b.merge(&per[i]);
        }
        assert!(close(&a, &b, 1e-12));
        let (p, n) = &scans[0];
        let t = V::new(0.013, -0.007, 0.021);
        let pt: Vec<V> = p.iter().map(|x| x + t).collect();
        assert!(close(&per[0].moved(&t), &Mom::of(&pt, n), 1e-11));
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.5, 1.1).into_inner();
        let (pr, nr): (Vec<V>, Vec<V>) = p.iter().zip(n).map(|(x, y)| (r * x, r * y)).unzip();
        assert!(close(&rotated(&per[0], &r), &Mom::of(&pr, &nr), 1e-11));
    }

    #[test]
    fn collapse_from_moments_is_the_per_surfel_collapse() {
        let scans = cylinder(V::new(-4.0, 7.0, 2.0), V::new(0.0, 0.0, 1.0), 0.05, 0.3, 2, 200, 3);
        let (mut p, mut n) = (Vec::new(), Vec::new());
        for (x, y) in scans {
            p.extend(x);
            n.extend(y);
        }
        let u = V::new(0.05, 0.02, 1.0).normalize();
        let (r1, s1, _) = Mom::of(&p, &n).collapse(&u, 0.3);
        // per surfel: R minimising the spread of P(p − R n) about its mean, in closed form
        let pr = M3::identity() - u * u.transpose();
        let (x, y): (Vec<V>, Vec<V>) = p.iter().zip(&n).map(|(a, b)| (pr * a, pr * b)).unzip();
        let (mx, my) = (x.iter().sum::<V>() / x.len() as f64, y.iter().sum::<V>() / y.len() as f64);
        let b: f64 = x.iter().zip(&y).map(|(a, c)| (a - mx).dot(&(c - my))).sum();
        let c: f64 = y.iter().map(|c| (c - my).norm_squared()).sum();
        let r0 = b / c;
        let s0 = (x.iter().zip(&y).map(|(a, c)| ((a - mx) - (c - my) * r0).norm_squared()).sum::<f64>() / x.len() as f64).sqrt();
        assert!((r0 - r1).abs() < 1e-9 && (s0 - s1).abs() < 1e-9, "{r0} {r1} {s0} {s1}");
        assert!((r1 - 0.05).abs() < 0.01);
    }

    #[test]
    fn the_axis_ignores_offsets_along_the_normals() {
        let (c, u) = (V::new(2.0, -1.0, 1.2), V::new(0.2, 0.1, 1.0).normalize());
        let scans = cylinder(c, u, 0.04, 0.15, 3, 300, 6);
        let mut rnd = rng(9);
        let (mut m0, mut m1) = (Mom::zero(), Mom::zero());
        for (p, n) in &scans {
            for (x, y) in p.iter().zip(n) {
                m0.add(x, y);
                m1.add(&(x + y * (0.005 * rnd())), y);
            }
        }
        let (u0, x0, _) = m0.axis(&c, 0.04).unwrap();
        let (u1, x1, _) = m1.axis(&c, 0.04).unwrap();
        assert!(u0.dot(&u1).abs() > 1.0 - 1e-12 && (x0 - x1).norm() < 1e-8);
        assert!(u0.dot(&u).abs() > 0.99);
    }

    #[test]
    fn a_stick_reads_out_as_an_element() {
        let scans = cylinder(V::new(1.0, 1.0, 1.0), V::new(0.0, 0.0, 1.0), 0.02, 0.05, 2, 1500, 4);
        let m = Moments::new(&scans, 0.03);
        let el = m.candidates(30.0, 0.3);
        let best = el.iter().find(|e| e.key == [33, 33, 33]).unwrap();
        assert!(best.u[2] > 0.99 && (best.r - 0.02).abs() < 0.004, "{:?} {}", best.u, best.r);
        // each scan's normals cover its 120° arc
        for a in &best.arc {
            assert!((a.to_degrees() - 120.0).abs() < 25.0, "{}", a.to_degrees());
        }
    }
}
