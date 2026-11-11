// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Amodal completion of branches within one scan's range image (3dtrees
//! research 128, `amodal.md`; a spike, not a fit).
//!
//! The image is a [`ShadowMap`]: each lattice cell's first-echo range. Image
//! coordinates are measured as angles: a line step is `|azimuth_step| sin(zenith)`
//! radians, a column step `|zenith_step|`, both taken at the point in question.
//!
//! - **end labels (T-junctions):** a stripe end (its axis point `p`, outward
//!   direction `d`, radius `r`) projects to a cell at axis range ρ, front ρ − r.
//!   Its image direction is that of `p + delta d`. The window is `window`
//!   cells beyond the end along that direction, each with up to 2 cells either
//!   side (half the silhouette's width). A window cell is *near* (first echo
//!   nearer than the front by more than `margin`), *far* (beyond the back,
//!   ρ + r + `margin`), *empty* (no return) or *same*. The end is
//!   **occluded** if at least `occ_share` of the cells are near; **real** (a
//!   tip, a silhouette or a deletion) if at least `real_share` are far or empty
//!   and under `real_near_max` near; **ambiguous** otherwise.
//! - **the hidden path as a depth inequality:** a candidate link between two
//!   ends in the same scan is a curve of samples (image position, axis range,
//!   radius): an image cubic Hermite (tangents the ends' image directions, or
//!   the chord for a straight path) with range and radius linear between the
//!   ends; or a 3-D cubic Hermite projected sample by sample. Each cell within
//!   half the radius of a sample is counted once: *occluder* (nearer than the
//!   front by more than `tol`), *refute* (beyond the back by more than `tol`:
//!   the scan saw through), *unknown* (empty) or *support*. A path is
//!   admissible when nothing refutes; the refutes are its soft score.
//! - **wood-only confirmation:** a cylinder's support cells in a scan whose
//!   first echoes fall in a 3 cm node that scan's own surfels call wood (the
//!   node and its 26 neighbours, at least `min_n` surfels, collapse ratio under
//!   `ratio`).
//! - **lifted moments:** the moment octree of every scan but one, queried at
//!   points (a 9 cm neighbourhood): surfel count, collapse ratio, R*, the normal
//!   tensor's two smaller eigenvalues λ₂, λ₃ (sum 1) and its null direction.
//! - **occluders:** the world points of the first echoes behind an occluded
//!   label (an end's near window cells) and behind a path's occluder cells, so
//!   the octree can say what stands in front.

use super::moments::{key, per_scan, Key, Octree};
use super::opacity::{outcome, Cyl, ShadowMap};
use crate::transform::{norm, sub};
use crate::Point;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::collections::HashSet;

/// An end's class.
pub mod class {
    pub const OCCLUDED: u8 = 0;
    pub const REAL: u8 = 1;
    pub const AMBIGUOUS: u8 = 2;
    /// The end doesn't project into the image, or has no image direction.
    pub const NONE: u8 = 3;
}

#[derive(Debug, Clone)]
pub struct AmodalParams {
    pub margin: f64,
    pub window: usize,
    pub occ_share: f64,
    pub real_share: f64,
    pub real_near_max: f64,
    pub tol: f64,
    pub delta: f64,
}

impl Default for AmodalParams {
    fn default() -> Self {
        AmodalParams { margin: 0.05, window: 4, occ_share: 0.6, real_share: 0.6, real_near_max: 0.2, tol: 0.03, delta: 0.05 }
    }
}

/// A range image: a cell's first-echo range, `None` for an empty cell; `None` from `inside` off the image.
pub trait Depth: Sync {
    fn inside(&self, l: i64, c: i64) -> bool;
    fn depth(&self, l: i64, c: i64) -> Option<f64>;
}

impl Depth for ShadowMap<'_> {
    fn inside(&self, l: i64, c: i64) -> bool {
        l >= 0 && c >= 0 && (l as usize) < self.lat.n_lines && (c as usize) < self.lat.n_cols
    }
    fn depth(&self, l: i64, c: i64) -> Option<f64> {
        let j = self.cell[l as usize * self.lat.n_cols + c as usize];
        (j != u32::MAX).then(|| self.range[j as usize] as f64)
    }
}

/// A point on a path in the image: fractional cell, axis range, radius, and the angular steps (rad per line, per
/// column) there.
#[derive(Debug, Clone, Copy)]
pub struct Pt {
    pub l: f64,
    pub c: f64,
    pub rho: f64,
    pub r: f64,
    pub sl: f64,
    pub sc: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EndLabel {
    pub class: u8,
    pub near: u32,
    pub far: u32,
    pub empty: u32,
    pub same: u32,
}

/// An end in the image: where it falls, its unit image direction (angle space, (line, column)), how foreshortened
/// that direction is (image angle per radian of `delta` across the beam: 1 square on, 0 along the beam) and its
/// label.
#[derive(Debug, Clone, Copy)]
pub struct ImgEnd {
    pub at: Pt,
    pub u: [f64; 2],
    pub fore: f64,
    pub label: EndLabel,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PathCount {
    pub support: u32,
    pub refute: u32,
    pub occluder: u32,
    pub unknown: u32,
}

/// The fractional cell, axis range and angular steps of a world point.
pub fn project(map: &ShadowMap, p: &Point, r: f64) -> Option<Pt> {
    let s = map.to_scanner(p);
    let rho = norm(&s);
    let (l, c) = map.lat.cell_of(&s)?;
    let lat = map.lat;
    let sin_z = lat.cell_angles(l, c).0.to_radians().sin().abs().max(0.05);
    Some(Pt { l, c, rho, r, sl: lat.azimuth_step.abs().to_radians() * sin_z, sc: lat.zenith_step.abs().to_radians() })
}

fn classify(d: Option<f64>, rho: f64, r: f64, tol: f64) -> u8 {
    match d {
        None => outcome::UNKNOWN,
        Some(d) if d < rho - r - tol => outcome::OCCLUDED,
        Some(d) if d > rho + r + tol => outcome::REFUTE,
        _ => outcome::SUPPORT,
    }
}

/// The T-junction window of an end at `e` with unit image direction `u`: its cells on the image, each once.
fn window_cells(img: &impl Depth, e: &Pt, u: [f64; 2], prm: &AmodalParams) -> Vec<(i64, i64)> {
    let w = (0.5 * e.r / e.rho / e.sc).round().clamp(0.0, 2.0) as i64;
    let perp = [-u[1], u[0]];
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for k in 1..=prm.window as i64 {
        for j in -w..=w {
            let x = [(k as f64 * u[0] + j as f64 * perp[0]) * e.sc, (k as f64 * u[1] + j as f64 * perp[1]) * e.sc];
            let (l, c) = ((e.l + x[0] / e.sl).round() as i64, (e.c + x[1] / e.sc).round() as i64);
            if img.inside(l, c) && seen.insert((l, c)) {
                out.push((l, c));
            }
        }
    }
    out
}

/// The T-junction label of an end at `e` with unit image direction `u`, as the module docs say.
pub fn end_label(img: &impl Depth, e: &Pt, u: [f64; 2], prm: &AmodalParams) -> EndLabel {
    let mut o = EndLabel::default();
    for (l, c) in window_cells(img, e, u, prm) {
        match classify(img.depth(l, c), e.rho, e.r, prm.margin) {
            outcome::OCCLUDED => o.near += 1,
            outcome::REFUTE => o.far += 1,
            outcome::UNKNOWN => o.empty += 1,
            _ => o.same += 1,
        }
    }
    let n = (o.near + o.far + o.empty + o.same).max(1) as f64;
    let (near, beyond) = (o.near as f64 / n, (o.far + o.empty) as f64 / n);
    o.class = if o.near + o.far + o.empty + o.same == 0 {
        class::NONE
    } else if near >= prm.occ_share {
        class::OCCLUDED
    } else if beyond >= prm.real_share && near < prm.real_near_max {
        class::REAL
    } else {
        class::AMBIGUOUS
    };
    o
}

/// Every cell within half the radius of a path's samples, once, with the first sample that reached it.
fn path_cells<'a>(img: &impl Depth, path: &'a [Pt]) -> Vec<(i64, i64, &'a Pt)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for s in path {
        let (wl, wc) = ((0.5 * s.r / s.rho / s.sl).floor() as i64, (0.5 * s.r / s.rho / s.sc).floor() as i64);
        let (l0, c0) = (s.l.round() as i64, s.c.round() as i64);
        for l in l0 - wl..=l0 + wl {
            for c in c0 - wc..=c0 + wc {
                if img.inside(l, c) && seen.insert((l, c)) {
                    out.push((l, c, s));
                }
            }
        }
    }
    out
}

/// Every cell within half the radius of a path's samples, counted once, as the module docs say.
pub fn score_path(img: &impl Depth, path: &[Pt], tol: f64) -> PathCount {
    let mut o = PathCount::default();
    for (l, c, s) in path_cells(img, path) {
        match classify(img.depth(l, c), s.rho, s.r, tol) {
            outcome::OCCLUDED => o.occluder += 1,
            outcome::REFUTE => o.refute += 1,
            outcome::UNKNOWN => o.unknown += 1,
            _ => o.support += 1,
        }
    }
    o
}

fn hermite(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2]
}

/// An image path from end a to end b: a cubic Hermite in angle space leaving a along `ua` and arriving along −`ub`
/// (b's outward direction), or straight; range and radius linear; samples every quarter column step or so.
pub fn image_path(a: &Pt, ua: [f64; 2], b: &Pt, ub: [f64; 2], straight: bool) -> Vec<Pt> {
    let (sl, sc) = (0.5 * (a.sl + b.sl), 0.5 * (a.sc + b.sc));
    let (pa, pb) = ([a.l * sl, a.c * sc], [b.l * sl, b.c * sc]);
    let ch = [pb[0] - pa[0], pb[1] - pa[1]];
    let len = (ch[0] * ch[0] + ch[1] * ch[1]).sqrt();
    let ok = |u: [f64; 2]| u[0].is_finite() && u[1].is_finite();
    let (ta, tb) = if straight || !ok(ua) || !ok(ub) { (ch, ch) } else { ([ua[0] * len, ua[1] * len], [-ub[0] * len, -ub[1] * len]) };
    let n = ((4.0 * 1.5 * len / sc).ceil() as usize).clamp(2, 8000);
    (0..=n).map(|i| {
        let t = i as f64 / n as f64;
        let h = hermite(t);
        let x = [0, 1].map(|k| h[0] * pa[k] + h[1] * ta[k] + h[2] * pb[k] + h[3] * tb[k]);
        Pt { l: x[0] / sl, c: x[1] / sc, rho: (1.0 - t) * a.rho + t * b.rho, r: (1.0 - t) * a.r + t * b.r, sl, sc }
    }).collect()
}

/// A 3-D cubic Hermite from `pa` (tangent `ta`) to `pb` (tangent `tb`, both with the chord's length), radius linear,
/// projected into the image sample by sample; samples off the image are dropped.
#[allow(clippy::too_many_arguments)]
pub fn world_path(map: &ShadowMap, pa: &Point, ta: &Point, pb: &Point, tb: &Point, ra: f64, rb: f64) -> Vec<Pt> {
    let len = norm(&sub(pb, pa));
    let rho = norm(&map.to_scanner(pa)).max(0.5);
    let sc = map.lat.zenith_step.abs().to_radians();
    let n = ((4.0 * 1.5 * len / rho / sc).ceil() as usize).clamp(2, 8000);
    (0..=n).filter_map(|i| {
        let t = i as f64 / n as f64;
        let h = hermite(t);
        let x = [0, 1, 2].map(|k| h[0] * pa[k] + h[1] * ta[k] + h[2] * pb[k] + h[3] * tb[k]);
        project(map, &x, (1.0 - t) * ra + t * rb)
    }).collect()
}

/// Each end in the image, labelled; `None` off the image or without an image direction.
pub fn image_ends(map: &ShadowMap, ends: &[(Point, Point, f64)], prm: &AmodalParams) -> Vec<Option<ImgEnd>> {
    ends.par_iter().map(|(p, d, r)| {
        let a = project(map, p, *r)?;
        let q = [p[0] + prm.delta * d[0], p[1] + prm.delta * d[1], p[2] + prm.delta * d[2]];
        let b = project(map, &q, *r)?;
        let mut dl = b.l - a.l;
        let nl = map.lat.n_lines as f64;
        if dl.abs() > 0.5 * nl {
            dl -= nl * dl.signum();
        }
        let du = [dl * a.sl, (b.c - a.c) * a.sc];
        let len = (du[0] * du[0] + du[1] * du[1]).sqrt();
        if !(len > 1e-12) {
            return None;
        }
        let u = [du[0] / len, du[1] / len];
        Some(ImgEnd { at: a, u, fore: len / (prm.delta / a.rho), label: end_label(map, &a, u, prm) })
    }).collect()
}

/// Per pair of image ends: the gap in columns, each end's image direction against the chord (deg; b's against b→a),
/// and the straight and Hermite image paths' counts. `None` if either end is missing or the pair straddles the
/// image's seam.
pub fn image_pairs(map: &ShadowMap, ends: &[Option<ImgEnd>], pairs: &[(usize, usize)], prm: &AmodalParams) -> Vec<Option<(f64, f64, f64, PathCount, PathCount)>> {
    pairs.par_iter().map(|&(i, j)| {
        let (a, b) = (ends[i]?, ends[j]?);
        if (a.at.l - b.at.l).abs() > 0.5 * map.lat.n_lines as f64 {
            return None;
        }
        let (sl, sc) = (0.5 * (a.at.sl + b.at.sl), 0.5 * (a.at.sc + b.at.sc));
        let ch = [(b.at.l - a.at.l) * sl, (b.at.c - a.at.c) * sc];
        let len = (ch[0] * ch[0] + ch[1] * ch[1]).sqrt().max(1e-12);
        let ang = |u: [f64; 2], s: f64| ((u[0] * ch[0] + u[1] * ch[1]) * s / len).clamp(-1.0, 1.0).acos().to_degrees();
        let st = score_path(map, &image_path(&a.at, a.u, &b.at, b.u, true), prm.tol);
        let he = score_path(map, &image_path(&a.at, a.u, &b.at, b.u, false), prm.tol);
        Some((len / sc, ang(a.u, 1.0), ang(b.u, -1.0), st, he))
    }).collect()
}

/// The world point of a cell's first echo: the cell centre's direction at its range.
fn echo_point(map: &ShadowMap, l: i64, c: i64) -> Option<Point> {
    let d = map.depth(l, c)?;
    let v = map.lat.cell_direction(l as f64, c as f64);
    Some([0, 1, 2].map(|a| (map.rot[a][0] * v[0] + map.rot[a][1] * v[1] + map.rot[a][2] * v[2]) * d + map.origin[a]))
}

/// What stands in front, as world points of first echoes (3dtrees research 128, `occluder.md`): per end, its
/// window's near cells; per pair, the occluder cells of its Hermite image path. Each list is (owner, point), owner
/// the end's or the pair's index; pairs `image_pairs` drops have none.
pub fn occluders(map: &ShadowMap, ends: &[Option<ImgEnd>], pairs: &[(usize, usize)], prm: &AmodalParams) -> (Vec<(u32, Point)>, Vec<(u32, Point)>) {
    let at_end = ends.par_iter().enumerate().flat_map_iter(|(i, e)| {
        let pts: Vec<Point> = e.map_or_else(Vec::new, |e| window_cells(map, &e.at, e.u, prm).into_iter()
            .filter(|&(l, c)| classify(map.depth(l, c), e.at.rho, e.at.r, prm.margin) == outcome::OCCLUDED)
            .filter_map(|(l, c)| echo_point(map, l, c)).collect());
        pts.into_iter().map(move |p| (i as u32, p))
    }).collect();
    let on_path = pairs.par_iter().enumerate().flat_map_iter(|(k, &(i, j))| {
        let pts: Vec<Point> = match (ends[i], ends[j]) {
            (Some(a), Some(b)) if (a.at.l - b.at.l).abs() <= 0.5 * map.lat.n_lines as f64 => {
                let path = image_path(&a.at, a.u, &b.at, b.u, false);
                path_cells(map, &path).into_iter()
                    .filter(|&(l, c, s)| classify(map.depth(l, c), s.rho, s.r, prm.tol) == outcome::OCCLUDED)
                    .filter_map(|(l, c, _)| echo_point(map, l, c)).collect()
            }
            _ => Vec::new(),
        };
        pts.into_iter().map(move |p| (k as u32, p))
    }).collect();
    (at_end, on_path)
}

/// The 3 cm nodes one scan's surfels call wood, as the module docs say.
pub fn wood_keys(p: &[Point], n: &[Point], min_n: f64, ratio: f64, r_max: f64) -> HashSet<Key> {
    let vv = |x: &[Point]| x.iter().map(|a| Vector3::new(a[0], a[1], a[2])).collect::<Vec<_>>();
    let o = Octree::new(&per_scan(&vv(p), &vv(n), &vec![0; p.len()], 0.03), 0.03, 1);
    o.levels[0].par_iter().filter_map(|(k, _)| {
        let c = Vector3::new((k[0] as f64 + 0.5) * 0.03, (k[1] as f64 + 0.5) * 0.03, (k[2] as f64 + 0.5) * 0.03);
        let (m, _) = o.query(0, &c, true);
        if m.n < min_n {
            return None;
        }
        let (_, u) = m.tensor();
        let (_, sp, s0) = m.collapse(&u, r_max);
        (s0 > 0.0 && sp / s0 < ratio).then_some(*k)
    }).collect()
}

/// Per cylinder against this scan: [support cells, of them in a wood node, informative cells], with the shadow-map
/// test as `ShadowMap::test` runs it. `dir` and `range` are the scan's pulses.
pub fn wood_support(map: &ShadowMap, dir: &[Point], cyls: &[Cyl], wood: &HashSet<Key>, footprint: impl Fn(f64) -> f64 + Sync, tol: f64, max_cells: f64) -> Vec<[u32; 3]> {
    let mut out = vec![[0u32; 3]; cyls.len()];
    for s in map.test(cyls, footprint, tol, max_cells, 0.0) {
        let o = &mut out[s.cyl as usize];
        if s.outcome == outcome::SUPPORT {
            o[0] += 1;
            let j = s.pulse as usize;
            let x = [0, 1, 2].map(|k| dir[j][k] * map.range[j] as f64);
            let w = [0, 1, 2].map(|a| map.rot[a][0] * x[0] + map.rot[a][1] * x[1] + map.rot[a][2] * x[2] + map.origin[a]);
            if wood.contains(&key(&Vector3::new(w[0], w[1], w[2]), 0.03)) {
                o[1] += 1;
            }
        }
        if s.outcome == outcome::SUPPORT || s.outcome == outcome::REFUTE {
            o[2] += 1;
        }
    }
    out
}

/// The fused moments at one point: surfels in the 9 cm neighbourhood, collapse ratio, R*, the normal tensor's λ₂ and
/// λ₃ and its null direction.
#[derive(Debug, Clone, Copy)]
pub struct Lift {
    pub n: f64,
    pub ratio: f64,
    pub r: f64,
    pub lam2: f64,
    pub lam3: f64,
    pub u: Point,
}

/// The moment octree of every scan but `exclude`, queried at `q`, as the module docs say (NaN where fewer than
/// `min_n` surfels).
pub fn lift(p: &[Point], n: &[Point], scan: &[i32], exclude: i32, q: &[Point], min_n: f64, r_max: f64) -> Vec<Lift> {
    let keep: Vec<usize> = (0..p.len()).filter(|&i| scan[i] != exclude).collect();
    let vv = |x: &[Point]| keep.iter().map(|&i| Vector3::new(x[i][0], x[i][1], x[i][2])).collect::<Vec<_>>();
    let sc: Vec<i32> = keep.iter().map(|&i| scan[i]).collect();
    let o = Octree::new(&per_scan(&vv(p), &vv(n), &sc, 0.03), 0.03, 1);
    q.par_iter().map(|x| {
        let (m, _) = o.query(0, &Vector3::new(x[0], x[1], x[2]), true);
        let nan = Lift { n: m.n, ratio: f64::NAN, r: f64::NAN, lam2: f64::NAN, lam3: f64::NAN, u: [f64::NAN; 3] };
        if m.n < min_n {
            return nan;
        }
        let (lam, u) = m.tensor();
        let (r, sp, s0) = m.collapse(&u, r_max);
        if !(s0 > 0.0) {
            return nan;
        }
        Lift { n: m.n, ratio: sp / s0, r, lam2: lam[1], lam3: lam[2], u: [u[0], u[1], u[2]] }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 100 x 100 image at 10 m (a wall), with a horizontal 4-cell-wide branch on rows (columns) 48-51 at 8 m from
    /// line 10 to 39 and 60 to 89, and a nearer 3 m post over lines 40-59 covering it; `see_through` empties the
    /// post's cells over the branch and puts the wall there instead.
    struct Img(Vec<Option<f64>>);
    impl Depth for Img {
        fn inside(&self, l: i64, c: i64) -> bool {
            (0..100).contains(&l) && (0..100).contains(&c)
        }
        fn depth(&self, l: i64, c: i64) -> Option<f64> {
            self.0[(l * 100 + c) as usize]
        }
    }
    fn img(post: bool) -> Img {
        let mut v = vec![Some(10.0); 10000];
        for l in 0..100 {
            for c in 48..52 {
                if (10..40).contains(&l) || (60..90).contains(&l) {
                    v[l * 100 + c] = Some(8.0);
                }
            }
            if post && (40..60).contains(&l) {
                for c in 0..100 {
                    v[l * 100 + c] = Some(3.0);
                }
            }
        }
        Img(v)
    }
    const S: f64 = 0.001;
    fn pt(l: f64) -> Pt {
        // 4 cells at 8 m and 1 mrad: r = 1.6 cm, axis 1.6 cm behind the front
        Pt { l, c: 49.5, rho: 8.016, r: 0.016, sl: S, sc: S }
    }

    #[test]
    fn t_junctions_label_ends() {
        let p = AmodalParams::default();
        // the end at line 39.5 facing +l meets the post: occluded; the far end at 10 faces the wall: real
        assert_eq!(end_label(&img(true), &pt(39.5), [1.0, 0.0], &p).class, class::OCCLUDED);
        assert_eq!(end_label(&img(true), &pt(9.5), [-1.0, 0.0], &p).class, class::REAL);
        // without the post the gap is wall: both inner ends are real
        assert_eq!(end_label(&img(false), &pt(39.5), [1.0, 0.0], &p).class, class::REAL);
    }

    #[test]
    fn hidden_path_is_admissible_behind_an_occluder_only() {
        let p = AmodalParams::default();
        for straight in [true, false] {
            let path = image_path(&pt(39.5), [1.0, 0.0], &pt(59.5), [-1.0, 0.0], straight);
            let behind = score_path(&img(true), &path, p.tol);
            assert_eq!(behind.refute, 0);
            assert!(behind.occluder >= 18, "{behind:?}");
            let through = score_path(&img(false), &path, p.tol);
            assert!(through.refute >= 18, "{through:?}");
        }
    }
}
