// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! How well a set of cylinders is supported by the points around them.
//!
//! Two point-based checks for QSM cylinders (e.g. from raycloudtools'
//! `_trees.txt`), batched over every cylinder of a plot tile:
//!
//! - [`surface_support`]: sample each cylinder's lateral surface and ask
//!   whether a point of the cloud lies within `tol` of each sample. A fat or
//!   misplaced cylinder leaves most of its samples unsupported.
//! - [`surface_distance`]: the signed distance from each point to the
//!   nearest cylinder surface of its own tree (< 0 inside).
//!
//! Both run in parallel and are deterministic: each cylinder draws its
//! samples from its own seeded generator.

use rayon::prelude::*;

use crate::transform::{cross, dot, norm, normalize, scale, sub};
use crate::util::spatial::KdTree;
use crate::Point;

/// SplitMix64: a tiny, seedable generator, one per cylinder.
struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn frame(a: &Point, b: &Point) -> Option<(Point, Point, Point, f64)> {
    let d = sub(b, a);
    let len = norm(&d);
    if len == 0.0 {
        return None;
    }
    let u = scale(&d, 1.0 / len);
    let helper = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = normalize(&cross(&u, &helper));
    Some((u, e1, cross(&u, &e1), len))
}

/// Samples on the cylinders' surfaces and whether each has a cloud point within `tol`.
#[derive(Debug, Clone, Default)]
pub struct SurfaceSupport {
    pub samples: Vec<Point>,
    /// Index of the cylinder each sample lies on.
    pub cylinder: Vec<u32>,
    pub supported: Vec<bool>,
}

/// Sample every cylinder `a[i] -> b[i]` of radius `r[i]` at `per_m2`
/// samples per square metre of lateral surface (at least `min_samples`;
/// none for a zero-length cylinder) and test each against `cloud`.
#[allow(clippy::too_many_arguments)]
pub fn surface_support(cloud: &KdTree, a: &[Point], b: &[Point], r: &[f64], tol: f64, per_m2: f64, min_samples: usize, seed: u64) -> SurfaceSupport {
    let per_cyl: Vec<(Vec<Point>, Vec<bool>)> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let Some((u, e1, e2, len)) = frame(&a[i], &b[i]) else { return (Vec::new(), Vec::new()) };
            let n = ((2.0 * std::f64::consts::PI * r[i] * len * per_m2) as usize).max(min_samples);
            let mut rng = SplitMix(seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut pts = Vec::with_capacity(n);
            let mut ok = Vec::with_capacity(n);
            for _ in 0..n {
                let (h, th) = (rng.unit() * len, rng.unit() * 2.0 * std::f64::consts::PI);
                let (c, s) = (th.cos() * r[i], th.sin() * r[i]);
                let p = [
                    a[i][0] + h * u[0] + c * e1[0] + s * e2[0],
                    a[i][1] + h * u[1] + c * e1[1] + s * e2[1],
                    a[i][2] + h * u[2] + c * e1[2] + s * e2[2],
                ];
                ok.push(cloud.nearest(&p).is_some_and(|(_, d)| d <= tol));
                pts.push(p);
            }
            (pts, ok)
        })
        .collect();
    let mut out = SurfaceSupport::default();
    for (i, (pts, ok)) in per_cyl.into_iter().enumerate() {
        out.cylinder.extend(std::iter::repeat_n(i as u32, pts.len()));
        out.samples.extend(pts);
        out.supported.extend(ok);
    }
    out
}

/// Signed distance from `p` to the surface of cylinder `a -> b` (radius `r`),
/// measured to the axis segment (end caps rounded).
fn to_surface(p: &Point, a: &Point, b: &Point, r: f64) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(&ab, &ab);
    let t = if l2 > 0.0 { (dot(&sub(p, a), &ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let foot = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    norm(&sub(p, &foot)) - r
}

/// For each group `g` (a tree), the points `point_offsets[g]..point_offsets[g + 1]`
/// and cylinders `cyl_offsets[g]..cyl_offsets[g + 1]`: each point's signed
/// distance to the nearest surface among the `k` cylinders of its group whose
/// midpoints are nearest. Infinite for a group without cylinders.
pub fn surface_distance(points: &[Point], point_offsets: &[usize], a: &[Point], b: &[Point], r: &[f64], cyl_offsets: &[usize], k: usize) -> Vec<f64> {
    let groups = point_offsets.len().saturating_sub(1);
    let per_group: Vec<Vec<f64>> = (0..groups)
        .into_par_iter()
        .map(|g| {
            let (p0, p1, c0, c1) = (point_offsets[g], point_offsets[g + 1], cyl_offsets[g], cyl_offsets[g + 1]);
            if c0 == c1 {
                return vec![f64::INFINITY; p1 - p0];
            }
            let mids: Vec<Point> = (c0..c1).map(|c| scale(&[a[c][0] + b[c][0], a[c][1] + b[c][1], a[c][2] + b[c][2]], 0.5)).collect();
            let kd = KdTree::new(&mids);
            points[p0..p1]
                .iter()
                .map(|p| {
                    kd.knn(p, k.min(c1 - c0))
                        .into_iter()
                        .map(|(j, _)| to_surface(p, &a[c0 + j], &b[c0 + j], r[c0 + j]))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect()
        })
        .collect();
    per_group.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_cloud(radius: f64, keep: impl Fn(f64) -> bool) -> Vec<Point> {
        let mut out = Vec::new();
        for i in 0..200 {
            for j in 0..100 {
                let th = i as f64 / 200.0 * 2.0 * std::f64::consts::PI;
                if keep(th) {
                    out.push([radius * th.cos(), radius * th.sin(), j as f64 * 0.02]);
                }
            }
        }
        out
    }

    fn support_of(cloud: &[Point], r: f64) -> f64 {
        let kd = KdTree::new(cloud);
        let s = surface_support(&kd, &[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 2.0]], &[r], 0.03, 400.0, 16, 0);
        s.supported.iter().filter(|&&x| x).count() as f64 / s.supported.len() as f64
    }

    #[test]
    fn covered_cylinder_is_supported_and_fat_one_is_not() {
        let cloud = ring_cloud(0.1, |_| true);
        assert!(support_of(&cloud, 0.1) > 0.98);
        assert!(support_of(&cloud, 0.3) < 0.05);
        let half = ring_cloud(0.1, |th| th < std::f64::consts::PI);
        let s = support_of(&half, 0.1);
        assert!((0.5..0.65).contains(&s), "{s}");
    }

    #[test]
    fn support_is_deterministic() {
        let cloud = ring_cloud(0.1, |th| th < 4.0);
        assert_eq!(support_of(&cloud, 0.1), support_of(&cloud, 0.1));
    }

    #[test]
    fn distance_is_to_the_own_group_only() {
        // group 0: a cylinder at x = 0; group 1: a cylinder at x = 1.
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]];
        let r = [0.1, 0.1];
        let pts = [[0.9, 0.0, 0.5], [0.95, 0.0, 0.5]]; // both nearer group 1's cylinder
        let d = surface_distance(&pts, &[0, 1, 2], &a, &b, &r, &[0, 1, 2], 8);
        assert!((d[0] - 0.8).abs() < 1e-12); // point 0 belongs to group 0
        assert!((d[1] + 0.05).abs() < 1e-12); // point 1 is inside group 1's cylinder
    }
}
