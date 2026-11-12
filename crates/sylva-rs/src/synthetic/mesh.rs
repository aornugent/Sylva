// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Triangle meshes for [`crate::synthetic::scan`]: opaque, two-sided
//! triangles found by a bounding volume hierarchy, so a scene of millions of
//! triangles (bark tubes, phyllodes, terrain) casts in about log time per ray.
//!
//! The hierarchy splits the triangles at the median centroid along the box's
//! longest axis until at most [`LEAF`] remain; a ray visits the nearer child
//! first and skips any box beyond its nearest hit so far. A hit is found by
//! the Möller–Trumbore test. Its normal is the vertex normals interpolated
//! at the hit (smooth shading of a faceted tube) when they are given, else
//! the face's own.

use crate::error::{Error, Result};
use crate::Point;

/// A triangle mesh: vertices, optional per-vertex normals (empty for face
/// normals), faces as vertex indices, and for each face the scene point
/// whose attributes and reflectance its echoes take.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub v: Vec<Point>,
    pub n: Vec<Point>,
    pub f: Vec<[u32; 3]>,
    pub point: Vec<u32>,
}

/// Most triangles in a leaf of the hierarchy.
const LEAF: usize = 4;

#[derive(Debug, Clone, Copy)]
struct Node {
    lo: [f32; 3],
    hi: [f32; 3],
    /// First child (inner node, the second is `start + 1`) or first triangle in `order` (leaf).
    start: u32,
    /// Triangles in a leaf; 0 for an inner node.
    count: u32,
}

/// A mesh indexed for ray casting.
pub struct Bvh<'a> {
    mesh: &'a Mesh,
    nodes: Vec<Node>,
    order: Vec<u32>,
}

fn sub(a: &Point, b: &Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Mesh {
    pub fn check(&self, n_points: usize) -> Result<()> {
        if self.point.len() != self.f.len() {
            return Err(Error::invalid("mesh: one scene point per face"));
        }
        if !self.n.is_empty() && self.n.len() != self.v.len() {
            return Err(Error::invalid("mesh: vertex normals, if given, one per vertex"));
        }
        if self.f.iter().flatten().any(|&i| i as usize >= self.v.len()) || self.point.iter().any(|&i| i as usize >= n_points) {
            return Err(Error::invalid("mesh: face vertex or scene point out of range"));
        }
        if self.v.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::invalid("mesh: vertices must be finite"));
        }
        Ok(())
    }

    fn corners(&self, t: usize) -> [Point; 3] {
        let f = self.f[t];
        [self.v[f[0] as usize], self.v[f[1] as usize], self.v[f[2] as usize]]
    }

    /// The unit normal at barycentric `(u, v)` of face `t`, not oriented to the ray.
    pub fn normal(&self, t: usize, u: f64, v: f64) -> Point {
        let f = self.f[t];
        let m = if self.n.is_empty() {
            let [a, b, c] = self.corners(t);
            cross(&sub(&b, &a), &sub(&c, &a))
        } else {
            let (na, nb, nc) = (self.n[f[0] as usize], self.n[f[1] as usize], self.n[f[2] as usize]);
            std::array::from_fn(|k| (1.0 - u - v) * na[k] + u * nb[k] + v * nc[k])
        };
        let l = dot(&m, &m).sqrt();
        if l > 0.0 { [m[0] / l, m[1] / l, m[2] / l] } else { [0.0, 0.0, 1.0] }
    }
}

/// Möller–Trumbore: `(t, u, v)` where the ray `o + t d` meets the triangle, either side.
fn tri_hit(c: &[Point; 3], o: &Point, d: &Point) -> Option<(f64, f64, f64)> {
    let (e1, e2) = (sub(&c[1], &c[0]), sub(&c[2], &c[0]));
    let p = cross(d, &e2);
    let det = dot(&e1, &p);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, &c[0]);
    let u = dot(&s, &p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(&s, &e1);
    let v = dot(d, &q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some((dot(&e2, &q) * inv, u, v))
}

/// Entry and exit of the ray (inverse direction `inv`) through a box, or None.
fn box_hit(lo: &[f32; 3], hi: &[f32; 3], o: &Point, inv: &Point, t_max: f64) -> Option<f64> {
    let (mut ta, mut tb) = (0.0f64, t_max);
    for k in 0..3 {
        let (t0, t1) = ((lo[k] as f64 - o[k]) * inv[k], (hi[k] as f64 - o[k]) * inv[k]);
        let (t0, t1) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        ta = ta.max(t0);
        tb = tb.min(t1);
        if ta > tb {
            return None;
        }
    }
    Some(ta)
}

impl<'a> Bvh<'a> {
    pub fn new(mesh: &'a Mesh) -> Bvh<'a> {
        let nt = mesh.f.len();
        let mut lo = vec![[0f32; 3]; nt];
        let mut hi = vec![[0f32; 3]; nt];
        let mut ctr = vec![[0f64; 3]; nt];
        for t in 0..nt {
            let c = mesh.corners(t);
            for k in 0..3 {
                let (a, b) = (c[0][k].min(c[1][k]).min(c[2][k]), c[0][k].max(c[1][k]).max(c[2][k]));
                // f32 boxes, widened by a hair so rounding never shrinks them
                lo[t][k] = (a - 1e-6 - a.abs() * 1e-6) as f32;
                hi[t][k] = (b + 1e-6 + b.abs() * 1e-6) as f32;
                ctr[t][k] = (a + b) / 2.0;
            }
        }
        let mut order: Vec<u32> = (0..nt as u32).collect();
        let mut nodes = vec![Node { lo: [0.0; 3], hi: [0.0; 3], start: 0, count: 0 }];
        // (node, range of `order`)
        let mut stack = vec![(0usize, 0usize, nt)];
        while let Some((ni, a, b)) = stack.pop() {
            let (mut bl, mut bh) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for &t in &order[a..b] {
                for k in 0..3 {
                    bl[k] = bl[k].min(lo[t as usize][k]);
                    bh[k] = bh[k].max(hi[t as usize][k]);
                }
            }
            nodes[ni].lo = bl;
            nodes[ni].hi = bh;
            if b - a <= LEAF {
                nodes[ni].start = a as u32;
                nodes[ni].count = (b - a) as u32;
                continue;
            }
            let ax = (0..3).max_by(|&i, &j| (bh[i] - bl[i]).total_cmp(&(bh[j] - bl[j]))).unwrap();
            let mid = (a + b) / 2;
            order[a..b].select_nth_unstable_by(mid - a, |&i, &j| ctr[i as usize][ax].total_cmp(&ctr[j as usize][ax]));
            let left = nodes.len();
            nodes.push(Node { lo: [0.0; 3], hi: [0.0; 3], start: 0, count: 0 });
            nodes.push(Node { lo: [0.0; 3], hi: [0.0; 3], start: 0, count: 0 });
            nodes[ni].start = left as u32;
            stack.push((left, a, mid));
            stack.push((left + 1, mid, b));
        }
        Bvh { mesh, nodes, order }
    }

    /// The nearest triangle the ray `o + t d` meets at `t` in `(t_min, t_max)`: `(t, face, u, v)`.
    pub fn cast(&self, o: &Point, d: &Point, t_min: f64, t_max: f64) -> Option<(f64, usize, f64, f64)> {
        if self.order.is_empty() {
            return None;
        }
        let inv: Point = std::array::from_fn(|k| 1.0 / d[k]);
        let mut best: Option<(f64, usize, f64, f64)> = None;
        let mut limit = t_max;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        if box_hit(&self.nodes[0].lo, &self.nodes[0].hi, o, &inv, limit).is_some() {
            stack.push(0);
        }
        while let Some(ni) = stack.pop() {
            let n = self.nodes[ni as usize];
            if n.count > 0 {
                for &t in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let t = t as usize;
                    if let Some((h, u, v)) = tri_hit(&self.mesh.corners(t), o, d) {
                        if h > t_min && h < limit {
                            limit = h;
                            best = Some((h, t, u, v));
                        }
                    }
                }
                continue;
            }
            let (l, r) = (n.start, n.start + 1);
            let hl = box_hit(&self.nodes[l as usize].lo, &self.nodes[l as usize].hi, o, &inv, limit);
            let hr = box_hit(&self.nodes[r as usize].lo, &self.nodes[r as usize].hi, o, &inv, limit);
            match (hl, hr) {
                (Some(a), Some(b)) => {
                    // the nearer child is popped first
                    if a <= b {
                        stack.push(r);
                        stack.push(l);
                    } else {
                        stack.push(l);
                        stack.push(r);
                    }
                }
                (Some(_), None) => stack.push(l),
                (None, Some(_)) => stack.push(r),
                (None, None) => {}
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::nprandom::Generator;

    #[test]
    fn the_hierarchy_finds_the_nearest_triangle_as_brute_force_does() {
        let mut rng = Generator::new(7);
        let mut m = Mesh::default();
        for t in 0..2000u32 {
            let c = rng.normal_n(0.0, 1.0, 3);
            for _ in 0..3 {
                let e = rng.normal_n(0.0, 0.1, 3);
                m.v.push([c[0] + e[0], c[1] + e[1], c[2] + e[2]]);
            }
            m.f.push([3 * t, 3 * t + 1, 3 * t + 2]);
            m.point.push(0);
        }
        let bvh = Bvh::new(&m);
        let mut hits = 0;
        for _ in 0..500 {
            let o = rng.normal_n(0.0, 3.0, 3);
            let g = rng.normal_n(0.0, 0.5, 3);
            let d = sub(&[g[0], g[1], g[2]], &[o[0], o[1], o[2]]);
            let l = dot(&d, &d).sqrt();
            let (o, d) = ([o[0], o[1], o[2]], [d[0] / l, d[1] / l, d[2] / l]);
            let brute = (0..m.f.len()).filter_map(|t| tri_hit(&m.corners(t), &o, &d).filter(|h| h.0 > 0.0).map(|h| (h.0, t))).min_by(|a, b| a.0.total_cmp(&b.0));
            let got = bvh.cast(&o, &d, 0.0, f64::INFINITY).map(|h| (h.0, h.1));
            assert_eq!(got.map(|g| g.1), brute.map(|b| b.1));
            hits += usize::from(got.is_some());
        }
        assert!(hits > 100, "{hits}");
    }
}
