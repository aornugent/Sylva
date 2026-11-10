// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Wood as a fibre-orientation field, traced as streamlines from stem bases
//! (3dtrees research 128, `tracts.md`; a spike, not a fit).
//!
//! - **segments:** each [`Slice`] of a stripe, with its neighbours along the
//!   stripe (`window` either side, no gap), gives a short 3D piece of axis:
//!   the principal direction of their fronts (each slice's front range along
//!   its axis point's direction: R, behind it, is far noisier than the range),
//!   the slice's axis point moved onto the window's axis line with that
//!   direction, half the local slice spacing as its half-length, and a
//!   weight from the window's linearity and the slice's sharp edges.
//! - **field:** the pieces of every scan, deposited in a sparse voxel grid.
//!   A voxel keeps its contributions (direction, weight, point); its
//!   orientation tensor `sum w u u^T` summarises them.
//! - **tracing:** deterministic streamlines. At a point with a heading, the
//!   contributions of the 27 voxels around it, Gaussian-weighted by distance
//!   and within `cone` of the heading, are split into at most two axial
//!   clusters. The line steps along the cluster nearest its heading (pulled
//!   `centre` of the way towards that cluster's centroid, across the step);
//!   a second cluster distinct enough spawns a line (a fork). Too little
//!   weight coasts on the heading for at most `gap`; leaving the band stops
//!   it, and so does entering a voxel another line of the same seed visited.
//!   Every line keeps its seed's id.

use std::collections::HashMap;

use super::stripes::{edge, Slice};
use crate::transform::{add, dot, norm, normalize, scale, sub};
use crate::Point;

/// A short piece of wood axis from one scan, in the world frame.
#[derive(Debug, Clone, Default)]
pub struct Segment {
    pub p: Point,
    pub u: Point,
    pub half: f64,
    pub w: f64,
    pub r: f64,
    /// The index of the slice it came from.
    pub slice: usize,
}

fn eig3(m: [[f64; 3]; 3]) -> ([f64; 3], [Point; 3]) {
    let e = nalgebra::SymmetricEigen::new(nalgebra::Matrix3::from_fn(|i, j| m[i][j]));
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e.eigenvalues[b].partial_cmp(&e.eigenvalues[a]).unwrap());
    let v = |k: usize| [e.eigenvectors[(0, k)], e.eigenvectors[(1, k)], e.eigenvectors[(2, k)]];
    ([e.eigenvalues[idx[0]], e.eigenvalues[idx[1]], e.eigenvalues[idx[2]]], [v(idx[0]), v(idx[1]), v(idx[2])])
}

/// The axis pieces of one scan's slices (in the order [`super::stripes::stripes`] gives them: a stripe's slices
/// contiguous, along it), taken into the world frame by `rot` (world = rot . scanner) and `origin`. Neighbours along
/// a stripe within `window` slices either side, with no step over `max_step` cells between them, form a window; a
/// window needs at least 3 slices. Only pieces whose point lies in the box `lo`..`hi` are kept.
pub fn segments(slices: &[Slice], rot: [[f64; 3]; 3], origin: Point, window: usize, max_step: f64, lo: Point, hi: Point) -> Vec<Segment> {
    let world = |p: &Point| add(&[dot(&rot[0], p), dot(&rot[1], p), dot(&rot[2], p)], &origin);
    let along = |s: &Slice| if s.pass == 0 { s.col } else { s.line };
    let inside = |p: &Point| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
    let mut out = Vec::new();
    let mut a = 0;
    while a < slices.len() {
        let b = slices[a..].iter().position(|s| s.stripe != slices[a].stripe).map_or(slices.len(), |x| a + x);
        // runs of slices with no gap along the stripe
        let mut c0 = a;
        while c0 < b {
            let mut c1 = c0 + 1;
            while c1 < b && (along(&slices[c1]) - along(&slices[c1 - 1])).abs() <= max_step {
                c1 += 1;
            }
            let pts: Vec<Point> = slices[c0..c1].iter().map(|s| world(&s.axis)).collect();
            // the fronts: the axis lies R behind them, and R is noisy where the front's range is not
            let fronts: Vec<Point> = slices[c0..c1]
                .iter()
                .map(|s| {
                    let n = norm(&s.axis);
                    world(&if n > 0.0 { scale(&s.axis, s.range / n) } else { s.axis })
                })
                .collect();
            if pts.iter().any(inside) && pts.len() >= 3 {
                for i in 0..pts.len() {
                    if !inside(&pts[i]) {
                        continue;
                    }
                    let (w0, w1) = (i.saturating_sub(window), (i + window + 1).min(pts.len()));
                    let win = &fronts[w0..w1];
                    if win.len() < 3 {
                        continue;
                    }
                    let n = win.len() as f64;
                    let c = scale(&win.iter().fold([0.0; 3], |s, p| add(&s, p)), 1.0 / n);
                    let mut m = [[0.0; 3]; 3];
                    for p in win {
                        let d = sub(p, &c);
                        for x in 0..3 {
                            for y in 0..3 {
                                m[x][y] += d[x] * d[y];
                            }
                        }
                    }
                    let (l, v) = eig3(m);
                    if l[0] <= 0.0 {
                        continue;
                    }
                    let u = normalize(&v[0]);
                    let lin = 1.0 - (l[1].max(0.0) / l[0]).sqrt();
                    let spacing = (norm(&sub(&win[win.len() - 1], &win[0]))) / (n - 1.0);
                    let s = &slices[c0 + i];
                    let sharp = (s.edge_lo == edge::SHARP) as u8 + (s.edge_hi == edge::SHARP) as u8;
                    let ca = scale(&pts[w0..w1].iter().fold([0.0; 3], |s, p| add(&s, p)), 1.0 / n);
                    let p = add(&ca, &scale(&u, dot(&sub(&pts[i], &ca), &u)));
                    out.push(Segment { p, u, half: 0.5 * spacing, w: lin * (1.0 + sharp as f64) / 3.0, r: s.r, slice: c0 + i });
                }
            }
            c0 = c1;
        }
        a = b;
    }
    out
}

#[derive(Debug, Clone)]
pub struct TractParams {
    /// Voxel edge (m), step (m), the cone (deg) a direction must lie in of the heading, the least kernel weight
    /// that is data, the most length (m) a line coasts without it (and from a seed, before it first finds wood).
    pub voxel: f64,
    pub step: f64,
    pub cone: f64,
    pub min_w: f64,
    pub gap: f64,
    pub seed_gap: f64,
    /// A second cluster forks when it holds `fork_share` of the weight in the cone and lies `fork_angle` (deg) or
    /// more from the first; a line forks again only `fork_every` (m) after its last fork.
    pub fork_share: f64,
    pub fork_angle: f64,
    pub fork_every: f64,
    /// The share of the offset to the cluster's centroid, across the step, a step moves (0: pure field lines).
    pub centre: f64,
    /// A line stops in a voxel another line of its seed visited, once it is `merge_after` (m) from its start.
    pub merge_after: f64,
    pub max_lines: usize,
    pub max_steps: usize,
}

impl Default for TractParams {
    fn default() -> Self {
        TractParams { voxel: 0.025, step: 0.01, cone: 45.0, min_w: 0.05, gap: 0.10, seed_gap: 0.5, fork_share: 0.25, fork_angle: 20.0, fork_every: 0.10, centre: 0.5, merge_after: 0.10, max_lines: 400, max_steps: 3000 }
    }
}

type Key = [i32; 3];

/// The field: each voxel's contributions (direction, weight, point).
pub struct Field {
    pub voxel: f64,
    pub cells: HashMap<Key, Vec<(Point, f64, Point)>>,
}

impl Field {
    pub fn key(&self, p: &Point) -> Key {
        [(p[0] / self.voxel).floor() as i32, (p[1] / self.voxel).floor() as i32, (p[2] / self.voxel).floor() as i32]
    }

    /// Deposits each segment along its length, a sample every half voxel.
    pub fn build(segs: &[Segment], voxel: f64) -> Field {
        let mut f = Field { voxel, cells: HashMap::new() };
        for s in segs {
            let h = s.half.max(0.5 * voxel);
            let n = (2.0 * h / (0.5 * voxel)).ceil().max(1.0) as usize;
            let ds = 2.0 * h / n as f64;
            for k in 0..n {
                let q = add(&s.p, &scale(&s.u, -h + (k as f64 + 0.5) * ds));
                let key = f.key(&q);
                f.cells.entry(key).or_default().push((s.u, s.w * ds / voxel, q));
            }
        }
        f
    }

    /// Per voxel: total weight and the tensor's principal share `l1 / trace` (1: one direction; 1/3: isotropic).
    pub fn summary(&self) -> Vec<(Key, f64, f64)> {
        let mut out: Vec<(Key, f64, f64)> = self
            .cells
            .iter()
            .map(|(k, v)| {
                let mut m = [[0.0; 3]; 3];
                let mut wt = 0.0;
                for (u, w, _) in v {
                    wt += w;
                    for x in 0..3 {
                        for y in 0..3 {
                            m[x][y] += w * u[x] * u[y];
                        }
                    }
                }
                let (l, _) = eig3(m);
                (*k, wt, l[0] / (l[0] + l[1] + l[2]).max(1e-12))
            })
            .collect();
        out.sort_by_key(|x| x.0);
        out
    }

    /// The clusters (direction aligned to `h`, weight, centroid) of the field at `x` within `cone` of `h`: one, or
    /// two when the second is distinct (`p.fork_share`, `p.fork_angle`), the nearer `h` first.
    pub fn clusters(&self, x: &Point, h: &Point, p: &TractParams) -> Vec<(Point, f64, Point)> {
        let k0 = self.key(x);
        let cosc = p.cone.to_radians().cos();
        let s2 = 2.0 * self.voxel * self.voxel;
        let mut c: Vec<(Point, f64, Point)> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.cells.get(&[k0[0] + dx, k0[1] + dy, k0[2] + dz]) {
                        for (u, w, q) in v {
                            let a = dot(u, h);
                            if a.abs() < cosc {
                                continue;
                            }
                            let d = sub(q, x);
                            let kw = w * (-dot(&d, &d) / s2).exp();
                            c.push((if a < 0.0 { scale(u, -1.0) } else { *u }, kw, *q));
                        }
                    }
                }
            }
        }
        let total: f64 = c.iter().map(|e| e.1).sum();
        if total < p.min_w {
            return Vec::new();
        }
        let mean = |set: &dyn Fn(usize) -> bool| {
            let (mut u, mut w, mut q) = ([0.0; 3], 0.0, [0.0; 3]);
            for (_, e) in c.iter().enumerate().filter(|(i, _)| set(*i)) {
                u = add(&u, &scale(&e.0, e.1));
                q = add(&q, &scale(&e.2, e.1));
                w += e.1;
            }
            (if norm(&u) > 0.0 { normalize(&u) } else { *h }, w, if w > 0.0 { scale(&q, 1.0 / w) } else { *x })
        };
        let one = mean(&|_| true);
        // axial 2-means, started from the mean and the contribution farthest from it
        let far = c.iter().min_by(|a, b| dot(&a.0, &one.0).partial_cmp(&dot(&b.0, &one.0)).unwrap()).unwrap().0;
        if dot(&far, &one.0) > (0.5 * p.fork_angle).to_radians().cos() {
            return vec![one];
        }
        let (mut c1, mut c2) = (one.0, far);
        let mut assign = vec![false; c.len()];
        for _ in 0..6 {
            for (i, e) in c.iter().enumerate() {
                assign[i] = dot(&e.0, &c2) > dot(&e.0, &c1);
            }
            let (m1, m2) = (mean(&|i| !assign[i]), mean(&|i| assign[i]));
            c1 = m1.0;
            c2 = m2.0;
        }
        let (m1, m2) = (mean(&|i| !assign[i]), mean(&|i| assign[i]));
        if m1.1.min(m2.1) < p.fork_share * total || dot(&m1.0, &m2.0) > p.fork_angle.to_radians().cos() {
            return vec![one];
        }
        if dot(&m1.0, h) >= dot(&m2.0, h) { vec![m1, m2] } else { vec![m2, m1] }
    }
}

/// The traced lines: per point, its position, line and seed; per line, its seed, parent line (-1: from the seed)
/// and why it stopped (0 band, 1 no data, 2 merged, 3 step cap, 4 line cap).
#[derive(Debug, Default)]
pub struct Tracts {
    pub point: Vec<Point>,
    pub line: Vec<u32>,
    pub seed: Vec<u32>,
    pub line_seed: Vec<u32>,
    pub line_parent: Vec<i32>,
    pub line_stop: Vec<u8>,
}

/// Lines traced from each seed (heading +z) through `field`, within the box `lo`..`hi`.
pub fn trace(field: &Field, seeds: &[Point], lo: Point, hi: Point, p: &TractParams) -> Tracts {
    use rayon::prelude::*;
    let per: Vec<Tracts> = seeds.par_iter().enumerate().map(|(si, s)| trace_seed(field, si as u32, s, lo, hi, p)).collect();
    let mut out = Tracts::default();
    for t in per {
        let off = out.line_seed.len() as u32;
        out.point.extend(t.point);
        out.line.extend(t.line.iter().map(|l| l + off));
        out.seed.extend(t.seed);
        out.line_seed.extend(t.line_seed);
        out.line_parent.extend(t.line_parent.iter().map(|&l| if l < 0 { l } else { l + off as i32 }));
        out.line_stop.extend(t.line_stop);
    }
    out
}

fn trace_seed(field: &Field, si: u32, seed: &Point, lo: Point, hi: Point, p: &TractParams) -> Tracts {
    let mut t = Tracts::default();
    let mut visited: HashMap<Key, u32> = HashMap::new();
    // (start, heading, parent, gap allowed before the first wood)
    let mut queue = vec![(*seed, [0.0, 0.0, 1.0], -1i32, p.seed_gap)];
    let inside = |x: &Point| (0..3).all(|k| x[k] >= lo[k] && x[k] <= hi[k]);
    while let Some((x0, h0, parent, gap0)) = queue.pop() {
        let li = t.line_seed.len() as u32;
        if li as usize >= p.max_lines {
            break;
        }
        t.line_seed.push(si);
        t.line_parent.push(parent);
        let (mut x, mut h) = (x0, h0);
        let (mut travelled, mut since_fork, mut coast, mut found) = (0.0, 0.0, 0.0, parent >= 0);
        let mut pts: Vec<Point> = vec![x];
        let mut stop = 3u8;
        for _ in 0..p.max_steps {
            if !inside(&x) {
                stop = 0;
                break;
            }
            let key = field.key(&x);
            match visited.get(&key) {
                Some(&l) if l != li && travelled > p.merge_after => {
                    stop = 2;
                    break;
                }
                None => {
                    visited.insert(key, li);
                }
                _ => {}
            }
            let cl = field.clusters(&x, &h, p);
            if cl.is_empty() {
                coast += p.step;
                if coast > if found { p.gap } else { gap0 } {
                    stop = 1;
                    break;
                }
                x = add(&x, &scale(&h, p.step));
            } else {
                found = true;
                coast = 0.0;
                if cl.len() == 2 && since_fork >= p.fork_every {
                    queue.push((x, cl[1].0, li as i32, 0.0));
                    since_fork = 0.0;
                }
                let (d, _, c) = cl[0];
                let off = sub(&c, &x);
                let perp = sub(&off, &scale(&d, dot(&off, &d)));
                x = add(&add(&x, &scale(&d, p.step)), &scale(&perp, p.centre));
                h = d;
            }
            travelled += p.step;
            since_fork += p.step;
            pts.push(x);
        }
        // a line that ends coasting loses its coasted tail
        let keep = pts.len() - ((coast / p.step).round() as usize).min(pts.len() - 1);
        for q in &pts[..keep] {
            t.point.push(*q);
            t.line.push(li);
            t.seed.push(si);
        }
        t.line_stop.push(stop);
    }
    if t.line_seed.len() >= p.max_lines {
        if let Some(s) = t.line_stop.last_mut() {
            *s = 4;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(p: Point, u: Point) -> Segment {
        Segment { p, u: normalize(&u), half: 0.01, w: 0.5, r: 0.02, slice: 0 }
    }

    /// A Y: a vertical stem to 1 m, then two branches leaning 30 deg either way, sampled every 2 cm.
    fn y_field() -> Field {
        let mut s = Vec::new();
        for k in 0..50 {
            s.push(seg([0.0, 0.0, 0.02 * k as f64], [0.0, 0.0, 1.0]));
        }
        let (sx, cz) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
        for side in [-1.0, 1.0] {
            for k in 1..40 {
                let t = 0.02 * k as f64;
                s.push(seg([side * sx * t, 0.0, 1.0 + cz * t], [side * sx, 0.0, cz]));
            }
        }
        Field::build(&s, 0.025)
    }

    #[test]
    fn a_stem_and_its_fork_are_traced_from_the_base() {
        let f = y_field();
        let t = trace(&f, &[[0.0, 0.0, 0.0]], [-2.0, -2.0, -0.5], [2.0, 2.0, 3.0], &TractParams::default());
        assert!(t.line_seed.len() >= 2, "{} lines", t.line_seed.len());
        let top = |side: f64| t.point.iter().any(|q| q[0] * side > 0.3 && q[2] > 1.5);
        assert!(top(1.0) && top(-1.0), "both branches reached");
        // every point lies near the Y
        for q in &t.point {
            let on_stem = q[2] <= 1.0 && (q[0] * q[0] + q[1] * q[1]).sqrt() < 0.03;
            let on_branch = q[2] >= 0.97 && ((q[0].abs() - (q[2] - 1.0) * 30f64.to_radians().tan()).abs() < 0.04);
            assert!(on_stem || on_branch, "{q:?} off the wood");
        }
    }

    #[test]
    fn a_line_stops_at_a_gap_and_a_sharp_turn() {
        // a stem to 0.5 m then a horizontal branch: the line must not turn 90 deg
        let mut s: Vec<Segment> = (0..25).map(|k| seg([0.0, 0.0, 0.02 * k as f64], [0.0, 0.0, 1.0])).collect();
        s.extend((1..30).map(|k| seg([0.02 * k as f64, 0.0, 0.5], [1.0, 0.0, 0.0])));
        let f = Field::build(&s, 0.025);
        let t = trace(&f, &[[0.0, 0.0, 0.0]], [-2.0, -2.0, -0.5], [2.0, 2.0, 3.0], &TractParams::default());
        assert!(t.point.iter().all(|q| q[0] < 0.1 && q[2] < 0.6), "turned onto the branch");
        assert!(t.line_stop.iter().all(|&s| s == 1));
    }

    #[test]
    fn segments_follow_a_stripe() {
        // slices of one stripe along a line leaning in x; the window gives its direction
        let sl: Vec<Slice> = (0..10)
            .map(|k| {
                let axis = [0.1 * k as f64 * 0.02, 5.0, 0.02 * k as f64];
                // a noisy R: the axis jitters in depth by 2 cm, the front doesn't
                let r = if k % 2 == 0 { 0.03 } else { 0.01 };
                let range = norm(&axis);
                Slice { stripe: 2, pass: 0, col: 4.0 * k as f64, axis: scale(&axis, (range + r) / range), range, edge_lo: edge::SHARP, ..Slice::default() }
            })
            .collect();
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let s = segments(&sl, id, [0.0; 3], 2, 6.0, [-1.0; 3], [1.0, 10.0, 1.0]);
        assert_eq!(s.len(), 10);
        let want = normalize(&[0.1, 0.0, 1.0]);
        assert!(s.iter().all(|x| dot(&x.u, &want).abs() > 0.999));
    }
}
