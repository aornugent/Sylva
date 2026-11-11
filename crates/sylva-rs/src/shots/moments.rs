// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Additive surfel moments on a sparse voxel hash, and an octree of them
//! (3dtrees research 128, `octree.md`; a spike, not a fit).
//!
//! - **moments** of a set of surfels (p, n): count, `Σp`, `Σppᵀ`, `Σn`,
//!   `Σnnᵀ`, `Σpnᵀ` and the line matrix `Σaaᵀ` with `a = (p × n, n)`. They
//!   are sums, so sets merge by adding them (scans, tiles, octree levels).
//! - **rigid motions act on them:** a translation t sends `a` to `Ad(t) a`
//!   with `Ad(t) = [[I, [t]×], [0, I]]` (and `Σp`, `Σppᵀ`, `Σpnᵀ` as p + t
//!   does); a rotation R acts as `diag(R, R)` on `a` and by conjugation on
//!   the 3×3 tensors.
//! - **estimators in O(1) from one node's moments:**
//!   - **medial collapse** about a direction u: the spread of `P(p − Rn)` is
//!     `A − 2RB + R²C`, so `R* = B/C` (non-robust; `surfels::collapse`
//!     without its refit);
//!   - **the line-complex axis** (`identities::line_complex` without its
//!     refit): moments moved to the node's centre, `s² Σnnᵀ` added for the
//!     cylinder's `n·u = 0`, `m` eliminated by a Schur complement, u the 3×3's
//!     smallest eigenvector, m put on `u·m = 0`;
//!   - **the normal tensor** `Σnnᵀ / N`: eigenvalues (sum 1) and its null
//!     direction.
//! - **per-scan offsets** by alternation: the axis from all scans' moments
//!   with each scan's moved by its `Ad(t_s)`, then each `t_s` as the least
//!   squares shift that makes that scan's normal lines meet the axis
//!   (quadratic in t for a fixed axis), the count-weighted mean shift held at
//!   zero.
//! - **the octree** is a hash map per level, a node's key its children's
//!   halved (floor); every internal node is the exact sum of its children.

use super::surfels::eig;
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
use std::collections::HashMap;

type V = Vector3<f64>;
type M3 = Matrix3<f64>;
pub type Key = [i32; 3];

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

fn skew(t: &V) -> M3 {
    M3::new(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
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

    /// The moments of the surfels rotated by r about the origin.
    pub fn rotated(&self, r: &M3) -> Mom {
        let mut d = Matrix6::zeros();
        d.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
        d.fixed_view_mut::<3, 3>(3, 3).copy_from(r);
        Mom {
            n: self.n,
            sp: r * self.sp,
            spp: r * self.spp * r.transpose(),
            sn: r * self.sn,
            snn: r * self.snn * r.transpose(),
            spn: r * self.spn * r.transpose(),
            saa: d * self.saa * d.transpose(),
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

    /// The normal tensor's eigenvalues (descending, sum 1) and its null direction.
    pub fn tensor(&self) -> ([f64; 3], V) {
        let (lam, ev) = eig(&(self.snn / self.n));
        let t = lam.iter().sum::<f64>();
        (lam.map(|x| x / t), ev[2])
    }

    /// The line-complex axis, coordinates about `c0`, with the cylinder rows `s (n, 0)`: (u, the axis point nearest
    /// c0, mean squared residual, the next eigenvalue's).
    pub fn axis(&self, c0: &V, s: f64) -> Option<(V, V, f64, f64)> {
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
        Some((u, c0 + u.cross(&mm), lam[2].max(0.0) / self.n, lam[1].max(0.0) / self.n))
    }
}

/// Per-scan shifts `t_s` (count-weighted mean zero) that make every scan's normal lines meet one axis, by
/// alternation from zero: (u, axis point, shifts). `c0` the coordinates' origin, `s` the cylinder rows' scale.
pub fn offsets(per_scan: &[Mom], c0: &V, s: f64, iters: usize) -> Option<(V, V, Vec<V>)> {
    let local: Vec<Mom> = per_scan.iter().map(|m| m.moved(&-c0)).collect();
    let mut t = vec![V::zeros(); local.len()];
    let mut fit = None;
    for _ in 0..iters {
        let mut all = Mom::zero();
        for (m, ts) in local.iter().zip(&t) {
            all.merge(&m.moved(ts));
        }
        let (u, x, _, _) = all.axis(&V::zeros(), s)?;
        fit = Some((u, x));
        // the axis as (u, m) about the origin; a surfel's residual is aᵀ(u, m), and a shift t adds t·(n × u)
        let l = {
            let mut l = Vector6::zeros();
            l.fixed_rows_mut::<3>(0).copy_from(&u);
            l.fixed_rows_mut::<3>(3).copy_from(&x.cross(&u));
            l
        };
        let k = skew(&u);
        let mut wsum = 0.0;
        let mut mean = V::zeros();
        for (m, ts) in local.iter().zip(t.iter_mut()) {
            // Σ e0 w with w = n × u = −[u]× n, and Σ w wᵀ = [u]× Σnnᵀ [u]×ᵀ
            let na = m.saa.fixed_view::<3, 6>(3, 0) * l;
            let g = -(k * na);
            let h = k * m.snn * k.transpose();
            let hr = h + M3::identity() * (1e-9 * h.trace() + 1e-12);
            *ts = -(hr.try_inverse()? * g);
            mean += *ts * m.n;
            wsum += m.n;
        }
        let mean = mean / wsum.max(1.0);
        for ts in t.iter_mut() {
            *ts -= mean;
        }
    }
    let (u, x) = fit?;
    Some((u, x + c0, t))
}

pub fn key(p: &V, cell: f64) -> Key {
    [(p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32]
}

/// Per (voxel, scan) moments of surfels `p`, `n` from scans `scan`, on `voxel` cells.
pub fn per_scan(p: &[V], n: &[V], scan: &[i32], voxel: f64) -> HashMap<(Key, i32), Mom> {
    let mut h: HashMap<(Key, i32), Mom> = HashMap::new();
    for i in 0..p.len() {
        h.entry((key(&p[i], voxel), scan[i])).or_insert_with(Mom::zero).add(&p[i], &n[i]);
    }
    h
}

/// The octree: level 0 the voxels (all scans summed, in ascending scan order so the sums don't depend on the order
/// the scans arrived in), each level above with cells twice as wide, every node the sum of its children.
pub struct Octree {
    pub voxel: f64,
    pub levels: Vec<HashMap<Key, (Mom, u32)>>,
}

impl Octree {
    /// From per-(voxel, scan) moments, with `n_levels` levels; a node's u32 is how many scans reached it.
    pub fn new(ps: &HashMap<(Key, i32), Mom>, voxel: f64, n_levels: usize) -> Self {
        let mut items: Vec<(&(Key, i32), &Mom)> = ps.iter().collect();
        items.sort_by_key(|(k, _)| (k.0, k.1));
        let mut level: HashMap<Key, (Mom, u32)> = HashMap::new();
        let mut sets: HashMap<Key, Vec<i32>> = HashMap::new();
        for ((k, s), m) in items {
            level.entry(*k).or_insert((Mom::zero(), 0)).0.merge(m);
            sets.entry(*k).or_default().push(*s);
        }
        let mut levels = Vec::with_capacity(n_levels);
        loop {
            // a node's scans: the union of its children's
            for (k, s) in sets.iter_mut() {
                s.sort_unstable();
                s.dedup();
                level.get_mut(k).unwrap().1 = s.len() as u32;
            }
            if levels.len() + 1 == n_levels {
                levels.push(level);
                break;
            }
            let mut keys: Vec<Key> = level.keys().copied().collect();
            keys.sort();
            let (mut up, mut up_sets): (HashMap<Key, (Mom, u32)>, HashMap<Key, Vec<i32>>) = (HashMap::new(), HashMap::new());
            for k in keys {
                let pk = k.map(|x| x >> 1);
                up.entry(pk).or_insert((Mom::zero(), 0)).0.merge(&level[&k].0);
                up_sets.entry(pk).or_default().extend(&sets[&k]);
            }
            levels.push(level);
            (level, sets) = (up, up_sets);
        }
        Octree { voxel, levels }
    }

    pub fn cell(&self, level: usize) -> f64 {
        self.voxel * (1 << level) as f64
    }

    /// The node holding x at `level` (with `nbhd`, it and its 26 neighbours summed): (moments, scans of the centre
    /// node).
    pub fn query(&self, level: usize, x: &V, nbhd: bool) -> (Mom, u32) {
        let k = key(x, self.cell(level));
        let lv = &self.levels[level];
        let mut m = Mom::zero();
        let r = if nbhd { 1 } else { 0 };
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some((o, _)) = lv.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        m.merge(o);
                    }
                }
            }
        }
        (m, lv.get(&k).map(|x| x.1).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::surfels::collapse;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Surfels on a cylinder (axis through c along u, radius r) with `noise` (rad) on the normals, from `views`
    /// scans each seeing a 120° arc, tagged by scan.
    fn cylinder(c: V, u: V, r: f64, noise: f64, views: usize, per: usize, seed: u64) -> (Vec<V>, Vec<V>, Vec<i32>) {
        let mut rnd = rng(seed);
        let e1 = u.cross(&V::new(0.3, 0.2, 0.9)).normalize();
        let e2 = u.cross(&e1);
        let (mut p, mut n, mut s) = (vec![], vec![], vec![]);
        for v in 0..views {
            let phi0 = v as f64 * std::f64::consts::TAU / views as f64;
            for _ in 0..per {
                let phi = phi0 + rnd() * std::f64::consts::FRAC_PI_3;
                let h = rnd() * 0.05;
                let nr = e1 * phi.cos() + e2 * phi.sin();
                p.push(c + u * h + nr * r);
                n.push((nr + V::new(rnd(), rnd(), rnd()) * noise).normalize());
                s.push(v as i32);
            }
        }
        (p, n, s)
    }

    fn close(a: &Mom, b: &Mom, tol: f64) -> bool {
        let d = |x: f64, y: f64| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs()));
        d(a.n, b.n)
            && a.sp.iter().zip(b.sp.iter()).all(|(x, y)| d(*x, *y))
            && a.spp.iter().zip(b.spp.iter()).all(|(x, y)| d(*x, *y))
            && a.sn.iter().zip(b.sn.iter()).all(|(x, y)| d(*x, *y))
            && a.snn.iter().zip(b.snn.iter()).all(|(x, y)| d(*x, *y))
            && a.spn.iter().zip(b.spn.iter()).all(|(x, y)| d(*x, *y))
            && a.saa.iter().zip(b.saa.iter()).all(|(x, y)| d(*x, *y))
    }

    #[test]
    fn merging_scans_does_not_depend_on_their_order() {
        let (p, n, s) = cylinder(V::new(3.0, -2.0, 1.5), V::new(0.1, 0.2, 1.0).normalize(), 0.03, 0.2, 5, 400, 1);
        let ps = per_scan(&p, &n, &s, 0.03);
        // scans reduced in two orders
        let fold = |order: &[i32]| {
            let mut h: HashMap<Key, Mom> = HashMap::new();
            for o in order {
                for ((k, sc), m) in &ps {
                    if sc == o {
                        h.entry(*k).or_insert_with(Mom::zero).merge(m);
                    }
                }
            }
            h
        };
        let (a, b) = (fold(&[0, 1, 2, 3, 4]), fold(&[3, 0, 4, 2, 1]));
        assert_eq!(a.len(), b.len());
        let mut worst = 0.0f64;
        for (k, m) in &a {
            assert!(close(m, &b[k], 1e-12));
            worst = worst.max((m.saa - b[k].saa).abs().max() / m.saa.abs().max());
        }
        eprintln!("order-independence: worst relative difference in Σaaᵀ between two scan orders {worst:.1e}");
        // the octree sorts by (voxel, scan), so it is bitwise the same whatever order the map iterates in
        let o1 = Octree::new(&ps, 0.03, 3);
        let mut ps2: HashMap<(Key, i32), Mom> = HashMap::new();
        let mut items: Vec<_> = ps.iter().collect();
        items.reverse();
        for (k, m) in items {
            ps2.insert(*k, *m);
        }
        let o2 = Octree::new(&ps2, 0.03, 3);
        for (l1, l2) in o1.levels.iter().zip(&o2.levels) {
            for (k, m) in l1 {
                assert_eq!(m, &l2[k]);
            }
        }
    }

    #[test]
    fn moments_move_with_the_surfels() {
        let (p, n, _) = cylinder(V::new(1.0, 2.0, 0.5), V::new(0.3, -0.1, 1.0).normalize(), 0.04, 0.3, 3, 300, 2);
        let m = Mom::of(&p, &n);
        let t = V::new(0.013, -0.007, 0.021);
        let pt: Vec<V> = p.iter().map(|x| x + t).collect();
        assert!(close(&m.moved(&t), &Mom::of(&pt, &n), 1e-11));
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.5, 1.1).into_inner();
        let (pr, nr): (Vec<V>, Vec<V>) = p.iter().zip(&n).map(|(x, y)| (r * x, r * y)).unzip();
        assert!(close(&m.rotated(&r), &Mom::of(&pr, &nr), 1e-11));
    }

    #[test]
    fn collapse_from_moments_is_the_per_surfel_collapse() {
        for (seed, noise) in [(3u64, 0.05), (4, 0.3), (5, 1.0)] {
            let (p, n, _) = cylinder(V::new(-4.0, 7.0, 2.0), V::new(0.0, 0.0, 1.0), 0.05, noise, 2, 200, seed);
            let u = V::new(0.05, 0.02, 1.0).normalize();
            let idx: Vec<usize> = (0..p.len()).collect();
            let (r0, s0, z0, _) = collapse(&p, &n, &idx, &u, 0.3);
            let (r1, s1, z1) = Mom::of(&p, &n).collapse(&u, 0.3);
            eprintln!("collapse noise {noise}: per surfel R {r0:.6} spread {s0:.6}; moments R {r1:.6} spread {s1:.6}");
            assert!((r0 - r1).abs() < 1e-9 && (s0 - s1).abs() < 1e-9 && (z0 - z1).abs() < 1e-9);
        }
    }

    #[test]
    fn the_axis_ignores_offsets_along_the_normals() {
        let (c, u) = (V::new(2.0, -1.0, 1.2), V::new(0.2, 0.1, 1.0).normalize());
        let (p, n, _) = cylinder(c, u, 0.04, 0.15, 3, 300, 6);
        let mut rnd = rng(9);
        let q: Vec<V> = p.iter().zip(&n).map(|(x, y)| x + y * (0.005 * rnd())).collect();
        let (u0, x0, _, _) = Mom::of(&p, &n).axis(&c, 0.04).unwrap();
        let (u1, x1, _, _) = Mom::of(&q, &n).axis(&c, 0.04).unwrap();
        let ang = u0.dot(&u1).abs().min(1.0).acos().to_degrees();
        eprintln!("axis under ±5 mm along the normals: {ang:.2e} deg, {:.2e} m; off the true axis {:.2} deg", (x0 - x1).norm(), u0.dot(&u).abs().min(1.0).acos().to_degrees());
        assert!(ang < 1e-6 && (x0 - x1).norm() < 1e-8);
        // and agrees with the per-surfel solve (identities::line_complex, whose refit drops nothing here)
        let idx: Vec<usize> = (0..p.len()).collect();
        let f = crate::shots::identities::line_complex(&p, &n, &idx, &c, Some(0.04)).unwrap();
        assert!(f.u.dot(&u0).abs() > 1.0 - 1e-6 || f.used < p.len(), "{} {}", f.u.dot(&u0), f.used);
    }

    #[test]
    fn an_octree_node_is_the_sum_of_its_children() {
        let (p, n, s) = cylinder(V::new(0.1, -0.2, 1.0), V::new(0.5, 0.2, 1.0).normalize(), 0.06, 0.3, 4, 2000, 7);
        let o = Octree::new(&per_scan(&p, &n, &s, 0.03), 0.03, 4);
        for l in 1..4 {
            let mut sum: HashMap<Key, Mom> = HashMap::new();
            for (k, (m, _)) in &o.levels[l - 1] {
                sum.entry(k.map(|x| x >> 1)).or_insert_with(Mom::zero).merge(m);
            }
            assert_eq!(sum.len(), o.levels[l].len());
            for (k, (m, _)) in &o.levels[l] {
                assert!(close(m, &sum[k], 1e-12));
            }
        }
        // the root-most level holds every surfel
        let tot: f64 = o.levels[3].values().map(|(m, _)| m.n).sum();
        assert_eq!(tot as usize, p.len());
    }

    #[test]
    fn offsets_by_alternation_recover_per_scan_shifts() {
        let (c, u) = (V::new(5.0, 5.0, 1.0), V::new(0.0, 0.0, 1.0));
        let (p, n, s) = cylinder(c, u, 0.05, 0.1, 3, 400, 8);
        let shift = [V::new(0.006, -0.004, 0.0), V::new(-0.003, 0.005, 0.01), V::new(0.0, 0.0, 0.0)];
        let mut ms = vec![Mom::zero(); 3];
        for i in 0..p.len() {
            ms[s[i] as usize].add(&(p[i] + shift[s[i] as usize]), &n[i]);
        }
        let (u1, _, t) = offsets(&ms, &c, 0.05, 20).unwrap();
        // the truth up to the common shift and anything along the axis
        let mean = (shift[0] + shift[1] + shift[2]) / 3.0;
        let perp = |x: V| x - u1 * x.dot(&u1);
        let err = (0..3).map(|k| (perp(t[k]) - perp(-(shift[k] - mean))).norm()).fold(0.0, f64::max);
        eprintln!("offsets by alternation: worst error across the axis {:.2} mm", err * 1e3);
        assert!(err < 1e-3, "{err}");
    }
}
