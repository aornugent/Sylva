// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Opaque wood as silhouette stripes in one scan's native range image
//! (3dtrees research 128, `stripes.md`; a spike, not a fit).
//!
//! The image is the scan's [`Lattice`], each cell holding its pulse's first
//! echo range (later echoes are ignored). An empty cell among returns is
//! *unknown* (no echo, or one deleted in export), never a pass.
//!
//! - **runs:** along one lattice axis, maximal sequences of present cells
//!   whose ranges step by at most `jump`. Each end is *sharp* (the next cell
//!   returned from farther: a silhouette edge against the background),
//!   *unknown* (the next cell is empty: the edge is censored), *occluded*
//!   (the next cell returned from nearer: a nearer surface hides the edge) or
//!   at the image border.
//! - **stripes:** runs on adjacent lines (the other axis) that overlap and
//!   agree in front range, joined. Two passes, one per axis: across mirror
//!   lines (near-vertical wood) and along them (near-horizontal wood); a
//!   stripe leaning more than 45° in its pass's image belongs to the other.
//! - **slices:** `slice` consecutive lines of a stripe. The silhouette's
//!   apparent half-width is `R_app = R + w (1/2 - tau)` ([`JointParams::app`],
//!   w the footprint at its range); the width across the stripe is the run's
//!   cells times the angular step, foreshortened by the stripe's lean in the
//!   image. The front's depth is `v.c - R`, so the axis lies `R` behind it.
//!   The front-depth bump's curvature gives a second, independent radius.
//!
//! No association across scans, no structure, no leaf model: resolved wood
//! only (at least `min_cells` cells wide).

use super::lattice::Lattice;
use crate::qsm::joint::JointParams;
use crate::Point;

/// How a run ends on one side.
pub mod edge {
    /// The next cell returned from farther away: a silhouette edge.
    pub const SHARP: u8 = 0;
    /// The next cell is empty: no echo or one deleted in export, so the edge is censored.
    pub const UNKNOWN: u8 = 1;
    /// The next cell returned from nearer: the edge is hidden behind a nearer surface.
    pub const OCCLUDED: u8 = 2;
    /// The image ends.
    pub const BORDER: u8 = 3;
}

#[derive(Debug, Clone)]
pub struct StripeParams {
    /// Largest range step (m) between neighbouring cells of one surface, and the most empty cells a run spans
    /// between two of its returns.
    pub jump: f64,
    pub gap: usize,
    /// Lines per slice.
    pub slice: usize,
    /// Fewest cells across a run, and fewest lines in a stripe.
    pub min_cells: usize,
    pub min_lines: usize,
    /// Widest run kept (m): wider is ground or a wall, not wood.
    pub max_width: f64,
    /// The beam (footprint, detection share) as the joint fit has it.
    pub beam: JointParams,
}

impl Default for StripeParams {
    fn default() -> Self {
        StripeParams { jump: 0.05, gap: 2, slice: 4, min_cells: 3, min_lines: 8, max_width: 1.0, beam: JointParams::default() }
    }
}

/// One slice of a stripe: its silhouette, front and axis.
#[derive(Debug, Clone, Default)]
pub struct Slice {
    pub stripe: u32,
    /// 0: runs across mirror lines (near-vertical wood); 1: along them.
    pub pass: u8,
    /// Centre in the image (line, col), and the median run's two ends along the run axis (cells, inclusive).
    pub line: f64,
    pub col: f64,
    pub lo: f64,
    pub hi: f64,
    /// Runs measured, and whether a line of the slice held two runs of the stripe (a fork in the image).
    pub n_runs: u32,
    pub forked: bool,
    /// Median cells across, and the stripe's lean in the image (metres across per metre along).
    pub width_cells: f64,
    pub lean: f64,
    /// Front range (m, the median run's nearest echo), the apparent half-width (m) and R after the footprint
    /// correction; R from the front's curvature (NaN where it is flat or too few cells).
    pub range: f64,
    pub half_app: f64,
    pub r: f64,
    pub r_depth: f64,
    /// Each side's [`edge`] (the commonest over the slice's runs).
    pub edge_lo: u8,
    pub edge_hi: u8,
    /// Beyond an unknown side, the median count of empty cells before the next return (capped at 16; 0 for a side
    /// that isn't unknown): the edge lies within them.
    pub open_lo: f64,
    pub open_hi: f64,
    /// One cell across the stripe (m, foreshortened by its lean) and the footprint's width (m) at its range.
    pub step_m: f64,
    pub footprint: f64,
    /// The axis point in the scanner's frame (m).
    pub axis: Point,
}

struct Run {
    u: u32,
    lo: u32,
    hi: u32,
    front: f32,
    edge_lo: u8,
    edge_hi: u8,
    open_lo: u8,
    open_hi: u8,
}

/// Empty cells counted beyond an unknown edge before giving up.
const OPEN_MAX: u8 = 16;

fn find(p: &mut [u32], mut i: u32) -> u32 {
    while p[i as usize] != i {
        p[i as usize] = p[p[i as usize] as usize];
        i = p[i as usize];
    }
    i
}

fn zenith(d: &Point) -> f64 {
    (d[2] / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()).clamp(-1.0, 1.0).acos()
}

/// The stripes of one scan's range image: `range` is each pulse's first echo (m) and `direction` its unit direction
/// in the scanner's frame, both in the lattice's pulse order.
pub fn stripes(lat: &Lattice, range: &[f32], direction: &[Point], p: &StripeParams) -> Vec<Slice> {
    let (nl, nc) = (lat.n_lines, lat.n_cols);
    let mut cell = vec![u32::MAX; nl * nc];
    for (i, (&l, &c)) in lat.line.iter().zip(&lat.col).enumerate() {
        if l != u32::MAX && range[i].is_finite() && range[i] > 0.0 {
            cell[l as usize * nc + c as usize] = i as u32;
        }
    }
    let (zstep, astep) = (lat.zenith_step.abs().to_radians(), lat.azimuth_step.abs().to_radians());
    let mut out = Vec::new();
    for pass in 0..2u8 {
        // pass 0: u = col (lines run along the wood), k = line; pass 1: u = line, k = col
        let (nu, nk) = if pass == 0 { (nc, nl) } else { (nl, nc) };
        let at = |u: usize, k: usize| if pass == 0 { cell[k * nc + u] } else { cell[u * nc + k] };
        // angular steps (rad) across and along the wood at a pulse
        let steps = |i: u32| {
            let s = zenith(&direction[i as usize]).sin().max(0.05);
            if pass == 0 { (astep * s, zstep) } else { (zstep, astep * s) }
        };
        let rng = |i: u32| range[i as usize] as f64;
        let mut runs: Vec<Run> = Vec::new();
        for u in 0..nu {
            let mut k = 0;
            while k < nk {
                let i = at(u, k);
                if i == u32::MAX {
                    k += 1;
                    continue;
                }
                let lo = k;
                let mut front = rng(i);
                // extend over present cells, and over gaps of up to `gap` empty cells the same surface spans
                loop {
                    let mut n = k + 1;
                    while n < nk && n <= k + 1 + p.gap && at(u, n) == u32::MAX {
                        n += 1;
                    }
                    if n >= nk || at(u, n) == u32::MAX || (rng(at(u, n)) - rng(at(u, k))).abs() > p.jump {
                        break;
                    }
                    k = n;
                    front = front.min(rng(at(u, n)));
                }
                let hi = k;
                k += 1;
                if hi + 1 - lo < 2 {
                    continue;
                }
                // a side: what lies beyond its end, and the empty cells before the next return (capped at OPEN_MAX)
                let side = |end: usize, dir: i64| {
                    let mut n = end as i64 + dir;
                    let mut open = 0u8;
                    while n >= 0 && (n as usize) < nk && at(u, n as usize) == u32::MAX && open < OPEN_MAX {
                        open += 1;
                        n += dir;
                    }
                    let e = if open > 0 {
                        edge::UNKNOWN
                    } else if n < 0 || n as usize >= nk {
                        edge::BORDER
                    } else if rng(at(u, n as usize)) > rng(at(u, end)) {
                        edge::SHARP
                    } else {
                        edge::OCCLUDED
                    };
                    (e, open)
                };
                if (hi + 1 - lo) as f64 * steps(at(u, lo)).0 * front > p.max_width {
                    continue;
                }
                let ((el, ol), (eh, oh)) = (side(lo, -1), side(hi, 1));
                runs.push(Run { u: u as u32, lo: lo as u32, hi: hi as u32, front: front as f32, edge_lo: el, edge_hi: eh, open_lo: ol, open_hi: oh });
            }
        }
        // link runs on adjacent lines that overlap and agree in front range
        let mut parent: Vec<u32> = (0..runs.len() as u32).collect();
        let mut a0 = 0;
        while a0 < runs.len() {
            let u = runs[a0].u;
            let a1 = runs[a0..].iter().position(|r| r.u != u).map_or(runs.len(), |x| a0 + x);
            let b1 = runs[a1..].iter().position(|r| r.u != u + 1).map_or(runs.len(), |x| a1 + x);
            if a1 < runs.len() && runs[a1].u == u + 1 {
                let mut b = a1;
                for a in a0..a1 {
                    while b < b1 && runs[b].hi < runs[a].lo {
                        b += 1;
                    }
                    let mut j = b;
                    while j < b1 && runs[j].lo <= runs[a].hi {
                        if (runs[j].front - runs[a].front).abs() as f64 <= 2.0 * p.jump {
                            let (x, y) = (find(&mut parent, a as u32), find(&mut parent, j as u32));
                            parent[x.max(y) as usize] = x.min(y);
                        }
                        j += 1;
                    }
                }
            }
            a0 = a1;
        }
        let mut members: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for i in 0..runs.len() as u32 {
            let r = find(&mut parent, i);
            members.entry(r).or_default().push(i);
        }
        let mut roots: Vec<u32> = members.keys().copied().collect();
        roots.sort_unstable();
        for root in roots {
            let m = &members[&root];
            let (u_first, u_last) = (runs[m[0] as usize].u, runs[*m.last().unwrap() as usize].u);
            if ((u_last - u_first + 1) as usize) < p.min_lines {
                continue;
            }
            // runs per line (a line with two is a fork in the image)
            let mut by_u: Vec<Vec<u32>> = vec![Vec::new(); (u_last - u_first + 1) as usize];
            for &i in m {
                by_u[(runs[i as usize].u - u_first) as usize].push(i);
            }
            let mid = |i: u32| 0.5 * (runs[i as usize].lo + runs[i as usize].hi) as f64;
            for s0 in (0..by_u.len()).step_by(p.slice) {
                let lines = &by_u[s0..(s0 + p.slice).min(by_u.len())];
                let one: Vec<u32> = lines.iter().filter(|v| v.len() == 1).map(|v| v[0]).collect();
                if one.len() * 2 < p.slice {
                    continue;
                }
                let forked = lines.iter().any(|v| v.len() > 1);
                let mut widths: Vec<f64> = one.iter().map(|&i| (runs[i as usize].hi + 1 - runs[i as usize].lo) as f64).collect();
                let wc = super::reconstruct::median(&mut widths);
                if wc < p.min_cells as f64 {
                    continue;
                }
                // the stripe's lean in the image: the Theil-Sen slope of midpoints against u over the slice and two
                // slices either side (ragged, censored edges move single midpoints by cells)
                let w0 = s0.saturating_sub(2 * p.slice);
                let w1 = (s0 + 3 * p.slice).min(by_u.len());
                let pts: Vec<(f64, f64)> = by_u[w0..w1].iter().filter(|v| v.len() == 1).map(|v| (runs[v[0] as usize].u as f64, mid(v[0]))).collect();
                let mut slopes: Vec<f64> = Vec::new();
                for (x, a) in pts.iter().enumerate() {
                    for b in &pts[x + 1..] {
                        if b.0 - a.0 >= (p.slice / 2).max(1) as f64 {
                            slopes.push((b.1 - a.1) / (b.0 - a.0));
                        }
                    }
                }
                let slope = if slopes.is_empty() { 0.0 } else { super::reconstruct::median(&mut slopes) };
                // the median run of the slice
                let mut order = one.clone();
                order.sort_by(|&a, &b| {
                    let (wa, wb) = (runs[a as usize].hi - runs[a as usize].lo, runs[b as usize].hi - runs[b as usize].lo);
                    wa.cmp(&wb)
                });
                let med = order[order.len() / 2];
                let mr = &runs[med as usize];
                let mut fronts: Vec<f64> = one.iter().map(|&i| runs[i as usize].front as f64).collect();
                let front = super::reconstruct::median(&mut fronts);
                let cell_at = |u: u32, k: u32| at(u as usize, k as usize);
                let (across, along) = steps(cell_at(mr.u, mr.lo));
                let lean = slope * across / along;
                let cos_t = 1.0 / (1.0 + lean * lean).sqrt();
                if lean.abs() > 1.0 {
                    continue;
                }
                let step_m = across * front * cos_t;
                let half_app = 0.5 * wc * step_m;
                let w = p.beam.footprint(front);
                let r = half_app - p.beam.app(w);
                // the front's curvature: range = a + b^2 / (2 R) over the central cells, one intercept per run
                let (mut sxx, mut sxy) = (0.0, 0.0);
                let mut cells = 0;
                for &i in &one {
                    let ru = &runs[i as usize];
                    let c = mid(i);
                    let pts: Vec<(f64, f64)> = (ru.lo..=ru.hi)
                        .filter_map(|k| {
                            let b = (k as f64 - c) * step_m;
                            let j = cell_at(ru.u, k);
                            (j != u32::MAX && b.abs() <= 0.8 * half_app).then(|| (b * b, rng(j)))
                        })
                        .collect();
                    if pts.len() < 3 {
                        continue;
                    }
                    let nn = pts.len() as f64;
                    let (mx, my) = (pts.iter().map(|q| q.0).sum::<f64>() / nn, pts.iter().map(|q| q.1).sum::<f64>() / nn);
                    for q in &pts {
                        sxx += (q.0 - mx).powi(2);
                        sxy += (q.0 - mx) * (q.1 - my);
                    }
                    cells += pts.len();
                }
                let curv = if sxx > 0.0 { sxy / sxx } else { f64::NAN };
                let r_depth = if cells >= 9 && curv > 0.0 { 0.5 / curv } else { f64::NAN };
                // the axis: the runs' middle directions, R behind the front
                let mut d = [0.0; 3];
                for &i in &one {
                    let ru = &runs[i as usize];
                    for k in [(ru.lo + ru.hi) / 2, (ru.lo + ru.hi).div_ceil(2)] {
                        // the present cell nearest k (gaps inside a run are empty)
                        let k = (0..=ru.hi - ru.lo).flat_map(|o| [k.saturating_sub(o), (k + o).min(ru.hi)]).find(|&k| k >= ru.lo && cell_at(ru.u, k) != u32::MAX).unwrap();
                        let v = direction[cell_at(ru.u, k) as usize];
                        for x in 0..3 {
                            d[x] += v[x];
                        }
                    }
                }
                let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let depth = front + r.max(0.0);
                let axis = [d[0] / dn * depth, d[1] / dn * depth, d[2] / dn * depth];
                let mode = |f: &dyn Fn(&Run) -> u8| {
                    let mut c = [0u32; 4];
                    for &i in &one {
                        c[f(&runs[i as usize]) as usize] += 1;
                    }
                    (0..4u8).max_by_key(|&e| (c[e as usize], 3 - e)).unwrap()
                };
                let open = |f: &dyn Fn(&Run) -> (u8, u8)| {
                    let mut v: Vec<f64> = one.iter().map(|&i| f(&runs[i as usize])).filter(|q| q.0 == edge::UNKNOWN).map(|q| q.1 as f64).collect();
                    if v.is_empty() { 0.0 } else { super::reconstruct::median(&mut v) }
                };
                let uc = one.iter().map(|&i| runs[i as usize].u as f64).sum::<f64>() / one.len() as f64;
                let kc = one.iter().map(|&i| mid(i)).sum::<f64>() / one.len() as f64;
                let (line, col) = if pass == 0 { (kc, uc) } else { (uc, kc) };
                out.push(Slice {
                    stripe: root,
                    pass,
                    line,
                    col,
                    lo: mr.lo as f64,
                    hi: mr.hi as f64,
                    n_runs: one.len() as u32,
                    forked,
                    width_cells: wc,
                    lean,
                    range: front,
                    half_app,
                    r,
                    r_depth,
                    open_lo: open(&|r: &Run| (r.edge_lo, r.open_lo)),
                    open_hi: open(&|r: &Run| (r.edge_hi, r.open_hi)),
                    step_m,
                    footprint: w,
                    edge_lo: mode(&|r: &Run| r.edge_lo),
                    edge_hi: mode(&|r: &Run| r.edge_hi),
                    axis,
                });
            }
        }
        // stripe ids unique across passes
        if pass == 0 {
            for s in out.iter_mut() {
                s.stripe *= 2;
            }
        } else {
            for s in out.iter_mut().filter(|s| s.pass == 1) {
                s.stripe = 2 * s.stripe + 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;

    /// A lattice of `nl` lines by `nc` columns (0.03 deg both ways, zenith from 80 deg, azimuth from 0), one pulse a
    /// cell, rendered against an opaque cylinder of radius `r` through `(x0, y0)` leaning `lean` (m per m, in x)
    /// before a wall 4 m behind it. A pulse whose beam axis passes within the apparent half-width
    /// `r + w (1/2 - tau)` returns from the cylinder's front, else from the wall; `drop(line, col)` deletes it.
    fn render(nl: usize, nc: usize, r: f64, (x0, y0): (f64, f64), lean: f64, drop: impl Fn(usize, usize, bool) -> bool) -> (Lattice, Vec<f32>, Vec<Point>) {
        let p = JointParams::default();
        let (mut line, mut col, mut range, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for l in 0..nl {
            for c in 0..nc {
                let v = dir(80.0 + 0.03 * c as f64, 0.03 * l as f64);
                // the axis through (x0 + lean z, y0, z): solve the closest approach in the xy plane at the ray's height
                let t_wall = (y0 + 4.0) / v[1];
                // the beam meets the cylinder where its perpendicular distance from the axis is below r
                let a = [lean, 0.0, 1.0];
                let an = (1.0 + lean * lean).sqrt();
                let a = [a[0] / an, a[1] / an, a[2] / an];
                let o = [-x0, -y0, 0.0]; // ray origin relative to the axis point (x0, y0, 0)
                let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
                let vp = [v[0] - dot(v, a) * a[0], v[1] - dot(v, a) * a[1], v[2] - dot(v, a) * a[2]];
                let op = [o[0] - dot(o, a) * a[0], o[1] - dot(o, a) * a[1], o[2] - dot(o, a) * a[2]];
                let (qa, qb) = (dot(vp, vp), dot(vp, op));
                let tc = -qb / qa;
                let dist = (dot(op, op) - qb * qb / qa).max(0.0).sqrt();
                let w = p.footprint(tc);
                let hit = dist <= r + p.app(w);
                let t = if hit {
                    let rr = dist.min(r);
                    tc - ((r * r - rr * rr) / qa).sqrt()
                } else {
                    t_wall
                };
                if drop(l, c, hit) {
                    continue;
                }
                line.push(l as u32);
                col.push(c as u32);
                range.push(t as f32);
                d.push(v);
            }
        }
        let lat = Lattice { line, col, n_lines: nl, n_cols: nc, zenith_step: 0.03, azimuth_step: 0.03, ..Lattice::default() };
        (lat, range, d)
    }

    fn check(r: f64, at: (f64, f64), lean: f64, drop: impl Fn(usize, usize, bool) -> bool) -> Vec<Slice> {
        let (lat, range, d) = render(400, 200, r, at, lean, drop);
        // the wall's own patches beside the stem are stripes too: keep the stem's
        let s: Vec<Slice> = stripes(&lat, &range, &d, &StripeParams::default()).into_iter().filter(|s| s.pass == 0 && s.range < at.1 + 1.0).collect();
        assert!(s.len() >= 20, "{} slices", s.len());
        assert!(s.iter().all(|x| x.stripe == s[0].stripe), "one stripe");
        let mut errs: Vec<f64> = s.iter().map(|x| (x.r / r).ln().abs()).collect();
        let med = crate::shots::reconstruct::median(&mut errs);
        assert!(med < 0.12, "|log R/r| median {med}, r {r} {:?}", s.iter().map(|x| x.r).collect::<Vec<_>>());
        for x in &s {
            // the axis point lies on the cylinder's axis to within a cell
            let ax = at.0 + lean * x.axis[2];
            let e = ((x.axis[0] - ax).powi(2) + (x.axis[1] - at.1).powi(2)).sqrt();
            assert!(e < 0.01 + 0.15 * r, "axis off by {e}");
        }
        s
    }

    #[test]
    fn a_resolved_cylinder_gives_its_radius_and_axis() {
        // vertical stems at 8 m, 5 and 2 cm, against the wall: sharp edges
        for r in [0.05, 0.02] {
            let s = check(r, (0.8, 8.0), 0.0, |_, _, _| false);
            assert!(s.iter().all(|x| x.edge_lo == edge::SHARP && x.edge_hi == edge::SHARP));
            assert!(s.iter().all(|x| x.lean.abs() < 0.05));
        }
    }

    #[test]
    fn a_leaning_cylinder_is_foreshortened() {
        let s = check(0.04, (0.6, 6.0), 0.3, |_, _, _| false);
        let mut leans: Vec<f64> = s.iter().map(|x| x.lean.abs()).collect();
        let m = crate::shots::reconstruct::median(&mut leans);
        assert!((m - 0.3).abs() < 0.05, "lean {m}");
    }

    #[test]
    fn deleted_edge_pulses_censor_the_edge() {
        // the export deletes the background pulse beside the wood on every line: edges become unknown
        let (lat, range, d) = render(400, 200, 0.05, (0.8, 8.0), 0.0, |_, _, _| false);
        let hit = |l: usize, c: usize| range[l * 200 + c] < 10.0;
        let keep: Vec<bool> = (0..lat.line.len())
            .map(|i| {
                let (l, c) = (lat.line[i] as usize, lat.col[i] as usize);
                hit(l, c) || !((l > 0 && hit(l - 1, c)) || (l + 1 < 400 && hit(l + 1, c)))
            })
            .collect();
        let pick = |v: &Vec<f32>| v.iter().zip(&keep).filter(|q| *q.1).map(|q| *q.0).collect::<Vec<f32>>();
        let lat2 = Lattice {
            line: lat.line.iter().zip(&keep).filter(|q| *q.1).map(|q| *q.0).collect(),
            col: lat.col.iter().zip(&keep).filter(|q| *q.1).map(|q| *q.0).collect(),
            ..lat.clone()
        };
        let d2: Vec<Point> = d.iter().zip(&keep).filter(|q| *q.1).map(|q| *q.0).collect();
        let s: Vec<Slice> = stripes(&lat2, &pick(&range), &d2, &StripeParams::default()).into_iter().filter(|s| s.pass == 0).collect();
        assert!(s.len() >= 20);
        assert!(s.iter().all(|x| x.edge_lo == edge::UNKNOWN && x.edge_hi == edge::UNKNOWN));
        // the hit run itself is unchanged, so R is too
        let mut errs: Vec<f64> = s.iter().map(|x| (x.r / 0.05).ln().abs()).collect();
        assert!(crate::shots::reconstruct::median(&mut errs) < 0.12);
    }

    #[test]
    fn a_nearer_stem_occludes_the_edge() {
        // two stems side by side in the image, one 1 m nearer: two stripes, the far one's edge occluded
        let (lat, mut range, d) = render(400, 200, 0.05, (0.8, 8.0), 0.0, |_, _, _| false);
        let (_, near, _) = render(400, 200, 0.04, (0.76, 7.0), 0.0, |_, _, _| false);
        for (a, b) in range.iter_mut().zip(&near) {
            *a = a.min(*b);
        }
        let s: Vec<Slice> = stripes(&lat, &range, &d, &StripeParams::default()).into_iter().filter(|s| s.pass == 0).collect();
        let mut ids: Vec<u32> = s.iter().map(|x| x.stripe).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 2, "two stripes");
        assert!(s.iter().filter(|x| x.range > 7.5).all(|x| x.edge_lo == edge::OCCLUDED || x.edge_hi == edge::OCCLUDED));
    }
    #[test]
    fn holes_inside_the_wood_do_not_split_it() {
        // the export deletes one pulse in every few inside the stem: still one stripe, its edges sharp, R unchanged
        let s = check(0.05, (0.8, 8.0), 0.0, |l, c, hit| hit && (l * 7 + c * 3) % 11 == 0);
        assert!(s.iter().filter(|x| x.edge_lo == edge::SHARP && x.edge_hi == edge::SHARP).count() * 4 >= s.len() * 3);
    }
}
