// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Strand tests (3dtrees research 128, `strands.md`; a spike, not a fit).
//!
//! - **scan offsets:** per region and scan, the mean of `p − R n` across a
//!   given axis, from the axis: where one scan's arc sits against the others'
//!   (registration, if the section is round).
//! - **end links:** fragment ends of one scan paired by a cost from direction
//!   (each end's outward direction against the gap) and conserved area (log
//!   radius ratio), mutual best within gates.
//! - **bundles:** per-scan pieces (stripe slices lifted to the world) merged
//!   by union-find: a stripe's own pieces always, pieces of different stripes
//!   when they share a voxel and agree in axis and radius, strongest first, and
//!   never across a cannot-link (two stripes a scan saw side by side in its
//!   image at different depths).

use super::surfels::{pt, v, Region, SurfelGrid, V};
use crate::Point;
use std::collections::{HashMap, HashSet};

/// Per region and scan: `(region, scan, offset (world, across the axis), surfels)` of the scan's mean `p − R n`
/// about the line through `c` along `u`, over the surfels the region gathers (at least `min_n` of that scan).
pub fn scan_offsets(p: &[Point], n: &[Point], scan: &[i32], regions: &[Region], c: &[Point], r: &[f64], margin: f64, min_n: usize) -> Vec<(u32, i32, Point, u32)> {
    use rayon::prelude::*;
    let grid = SurfelGrid::new(p);
    let pv: Vec<V> = p.iter().map(v).collect();
    let nv: Vec<V> = n.iter().map(v).collect();
    regions
        .par_iter()
        .enumerate()
        .flat_map_iter(|(k, g)| {
            let idx = grid.gather(&pv, scan, g, margin);
            let u = v(&g.u).normalize();
            let cc = v(&c[k]);
            let mut acc: HashMap<i32, (V, u32)> = HashMap::new();
            for &i in &idx {
                let d = pv[i] - nv[i] * r[k] - cc;
                let e = acc.entry(scan[i]).or_insert((V::zeros(), 0));
                e.0 += d - u * d.dot(&u);
                e.1 += 1;
            }
            let mut out: Vec<(u32, i32, Point, u32)> =
                acc.into_iter().filter(|(_, (_, m))| *m as usize >= min_n).map(|(s, (x, m))| (k as u32, s, pt(&(x / m as f64)), m)).collect();
            out.sort_by_key(|x| x.1);
            out.into_iter()
        })
        .collect()
}

/// A fragment's end: where it stops (world), its outward unit direction, its radius, and its fragment.
#[derive(Debug, Clone, Copy)]
pub struct End {
    pub p: Point,
    pub d: Point,
    pub r: f64,
    pub frag: u32,
}

/// A candidate link: the two ends, the gap (m), each end's angle to the gap (deg), the log radius ratio and the cost.
#[derive(Debug, Clone, Copy)]
pub struct Link {
    pub a: u32,
    pub b: u32,
    pub gap: f64,
    pub ang_a: f64,
    pub ang_b: f64,
    pub dlogr: f64,
    pub cost: f64,
}

/// Every pair of ends of different fragments within `[min_gap, max_gap]` whose outward directions both face the
/// gap within `max_ang` (deg) and whose radii agree within `max_dlogr`, costed `(ang_a² + ang_b²) / sd_ang² +
/// dlogr² / sd_logr²`; and the mutual-best subset (each end in at most one link, lowest cost first).
pub fn link_ends(ends: &[End], min_gap: f64, max_gap: f64, max_ang: f64, max_dlogr: f64, sd_ang: f64, sd_logr: f64) -> (Vec<Link>, Vec<Link>) {
    let cell = max_gap.max(1e-3);
    let key = |x: &Point| [(x[0] / cell).floor() as i64, (x[1] / cell).floor() as i64, (x[2] / cell).floor() as i64];
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, e) in ends.iter().enumerate() {
        grid.entry(key(&e.p)).or_default().push(i as u32);
    }
    let mut all = Vec::new();
    for (i, e) in ends.iter().enumerate() {
        let k = key(&e.p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).map(|x| x.as_slice()).unwrap_or(&[]) {
                        let j = j as usize;
                        if j <= i || ends[j].frag == e.frag {
                            continue;
                        }
                        let g = v(&ends[j].p) - v(&e.p);
                        let gap = g.norm();
                        if gap < min_gap || gap > max_gap {
                            continue;
                        }
                        let gu = g / gap;
                        let ang_a = v(&e.d).dot(&gu).clamp(-1.0, 1.0).acos().to_degrees();
                        let ang_b = (-v(&ends[j].d)).dot(&gu).clamp(-1.0, 1.0).acos().to_degrees();
                        let dlogr = (ends[j].r / e.r).ln();
                        if ang_a > max_ang || ang_b > max_ang || dlogr.abs() > max_dlogr {
                            continue;
                        }
                        let cost = (ang_a * ang_a + ang_b * ang_b) / (sd_ang * sd_ang) + dlogr * dlogr / (sd_logr * sd_logr);
                        all.push(Link { a: i as u32, b: j as u32, gap, ang_a, ang_b, dlogr, cost });
                    }
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by(|&x, &y| all[x].cost.partial_cmp(&all[y].cost).unwrap());
    let mut used = vec![false; ends.len()];
    let mut best = Vec::new();
    for k in order {
        let l = all[k];
        if !used[l.a as usize] && !used[l.b as usize] {
            used[l.a as usize] = true;
            used[l.b as usize] = true;
            best.push(l);
        }
    }
    (all, best)
}

/// One scan's piece: a lifted stripe slice (world axis point, unit direction, half-length, radius), its scan and
/// stripe, and its place in that scan's image (line, col, front range).
#[derive(Debug, Clone, Copy)]
pub struct Piece {
    pub p: Point,
    pub u: Point,
    pub half: f64,
    pub r: f64,
    pub scan: i32,
    pub stripe: u32,
    pub line: f64,
    pub col: f64,
    pub range: f64,
}

/// What `bundles` did: candidate cross-stripe edges, merges made, merges refused by a cannot-link, cannot-links.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeStats {
    pub edges: usize,
    pub merged: usize,
    pub blocked: usize,
    pub cannot: usize,
}

fn find(par: &mut [usize], mut x: usize) -> usize {
    while par[x] != x {
        par[x] = par[par[x]];
        x = par[x];
    }
    x
}

fn skey(p: &Piece) -> u64 {
    ((p.scan as u64) << 32) | p.stripe as u64
}

/// Union-find over pieces: a stripe's own pieces first, then `edges` (score, piece, piece) lowest score first, each
/// refused when it would join two stripes under a cannot-link (two stripes of one scan whose pieces lie within
/// `near_cells` (+ half their widths in cells, `width`) in its image and differ in front range by more than
/// `depth_step`).
fn merge_with(pc: &[Piece], width: &[f64], near_cells: f64, depth_step: f64, mut edges: Vec<(f64, usize, usize)>) -> (Vec<u32>, MergeStats) {
    let n = pc.len();
    let mut par: Vec<usize> = (0..n).collect();
    let mut members: Vec<HashSet<u64>> = pc.iter().map(|p| HashSet::from([skey(p)])).collect();
    let mut cannot: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut st = MergeStats::default();
    let mut img: HashMap<(i32, i64, i64), Vec<usize>> = HashMap::new();
    let ic = 8.0;
    for (i, p) in pc.iter().enumerate() {
        img.entry((p.scan, (p.line / ic).floor() as i64, (p.col / ic).floor() as i64)).or_default().push(i);
    }
    for (i, p) in pc.iter().enumerate() {
        let (kl, kc) = ((p.line / ic).floor() as i64, (p.col / ic).floor() as i64);
        for dl in -2..=2 {
            for dc in -2..=2 {
                for &j in img.get(&(p.scan, kl + dl, kc + dc)).map(|x| x.as_slice()).unwrap_or(&[]) {
                    let q = &pc[j];
                    if j <= i || q.stripe == p.stripe {
                        continue;
                    }
                    let reach = near_cells + 0.5 * (width[i] + width[j]);
                    if (p.line - q.line).hypot(p.col - q.col) <= reach && (p.range - q.range).abs() > depth_step {
                        let (a, b) = (skey(p), skey(q));
                        if cannot.entry(a).or_default().insert(b) {
                            st.cannot += 1;
                        }
                        cannot.entry(b).or_default().insert(a);
                    }
                }
            }
        }
    }
    let union = |par: &mut Vec<usize>, members: &mut Vec<HashSet<u64>>, a: usize, b: usize, check: bool| -> Option<bool> {
        let (ra, rb) = (find(par, a), find(par, b));
        if ra == rb {
            return None;
        }
        let (big, small) = if members[ra].len() >= members[rb].len() { (ra, rb) } else { (rb, ra) };
        if check {
            for m in &members[small] {
                if let Some(cs) = cannot.get(m) {
                    if cs.iter().any(|x| members[big].contains(x)) {
                        return Some(false);
                    }
                }
            }
        }
        let moved = std::mem::take(&mut members[small]);
        members[big].extend(moved);
        par[small] = big;
        Some(true)
    };
    let mut first: HashMap<u64, usize> = HashMap::new();
    for (i, p) in pc.iter().enumerate() {
        match first.get(&skey(p)) {
            Some(&j) => {
                union(&mut par, &mut members, i, j, false);
            }
            None => {
                first.insert(skey(p), i);
            }
        }
    }
    edges.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    st.edges = edges.len();
    for (_, a, b) in edges {
        match union(&mut par, &mut members, a, b, true) {
            Some(true) => st.merged += 1,
            Some(false) => st.blocked += 1,
            None => {}
        }
    }
    ((0..n).map(|i| find(&mut par, i) as u32).collect(), st)
}

/// The bundle (a root piece index) of every piece; see the module notes. Pieces share a voxel (`voxel` m) when
/// their axes, sampled every half voxel within their half-lengths, touch a common one; they agree when their axes
/// are within `max_ang` (deg) and radii within `max_dlogr`. Cannot-links as `merge_with`.
pub fn bundles(pc: &[Piece], width: &[f64], voxel: f64, max_ang: f64, max_dlogr: f64, near_cells: f64, depth_step: f64) -> (Vec<u32>, MergeStats) {
    let key = |x: &V| [(x[0] / voxel).floor() as i64, (x[1] / voxel).floor() as i64, (x[2] / voxel).floor() as i64];
    let mut vox: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, p) in pc.iter().enumerate() {
        let (c, u) = (v(&p.p), v(&p.u).normalize());
        let m = (p.half / (0.5 * voxel)).ceil() as i64;
        let mut ks: Vec<[i64; 3]> = (-m..=m).map(|t| key(&(c + u * (t as f64 * 0.5 * voxel).clamp(-p.half, p.half)))).collect();
        ks.sort_unstable();
        ks.dedup();
        for k in ks {
            vox.entry(k).or_default().push(i);
        }
    }
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for list in vox.values() {
        for (x, &i) in list.iter().enumerate() {
            for &j in &list[x + 1..] {
                let (a, b) = (i.min(j), i.max(j));
                if skey(&pc[a]) == skey(&pc[b]) || !seen.insert((a, b)) {
                    continue;
                }
                let ang = v(&pc[a].u).normalize().dot(&v(&pc[b].u).normalize()).abs().clamp(0.0, 1.0).acos().to_degrees();
                if ang <= max_ang && (pc[a].r / pc[b].r).ln().abs() <= max_dlogr {
                    edges.push((ang, a, b));
                }
            }
        }
    }
    merge_with(pc, width, near_cells, depth_step, edges)
}

/// A stripe as one line: unit direction (up), centre, its extent along the direction about the centre, median
/// radius, the unit beam from its scan to it, and a piece of it.
#[derive(Debug, Clone, Copy)]
pub struct StripeLine {
    pub u: V,
    pub c: V,
    pub t0: f64,
    pub t1: f64,
    pub r: f64,
    pub beam: V,
    pub piece: usize,
}

/// Each stripe's line (pieces weighted by half-length; `origin` per piece is its scanner's position).
pub fn stripe_lines(pc: &[Piece], origin: &[Point]) -> Vec<StripeLine> {
    let mut by: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, p) in pc.iter().enumerate() {
        by.entry(skey(p)).or_default().push(i);
    }
    let mut keys: Vec<u64> = by.keys().cloned().collect();
    keys.sort_unstable();
    keys.iter()
        .map(|k| {
            let ix = &by[k];
            let w: Vec<f64> = ix.iter().map(|&i| pc[i].half.max(1e-3)).collect();
            let sw: f64 = w.iter().sum();
            let t = ix.iter().zip(&w).map(|(&i, &wi)| {
                let u = v(&pc[i].u).normalize();
                u * u.transpose() * wi
            }).sum::<nalgebra::Matrix3<f64>>();
            let mut u = super::surfels::eig(&t).1[0];
            if u[2] < 0.0 {
                u = -u;
            }
            let c = ix.iter().zip(&w).map(|(&i, &wi)| v(&pc[i].p) * wi).sum::<V>() / sw;
            let (mut t0, mut t1) = (f64::MAX, f64::MIN);
            for &i in ix {
                let s = (v(&pc[i].p) - c).dot(&u);
                t0 = t0.min(s - pc[i].half);
                t1 = t1.max(s + pc[i].half);
            }
            let mut rr: Vec<f64> = ix.iter().map(|&i| pc[i].r).collect();
            let r = super::reconstruct::median(&mut rr);
            let beam = (c - v(&origin[ix[0]])).normalize();
            StripeLine { u, c, t0, t1, r, beam, piece: ix[0] }
        })
        .collect()
}

/// Line-consistency parameters: along a view's beam its line's position is uncertain by `max(k_beam R, sd_beam_min)`
/// (the one-sided depth–radius confound), across it by `sd_across`; `sd_floor` is added both ways (local
/// misalignment between scans).
#[derive(Debug, Clone, Copy)]
pub struct LineParams {
    pub k_beam: f64,
    pub sd_beam_min: f64,
    pub sd_across: f64,
    pub sd_floor: f64,
    pub max_ang: f64,
    pub max_dlogr: f64,
    pub max_gap: f64,
    pub chi2: f64,
}

/// The offset (across the shared direction) between two stripes' lines where they overlap, and its chi-square
/// under their covariances; None when they don't overlap within `max_gap` or fail the angle or radius gate.
pub fn line_consistency(a: &StripeLine, b: &StripeLine, prm: &LineParams) -> Option<(f64, V)> {
    let cosab = a.u.dot(&b.u);
    if cosab.abs().clamp(0.0, 1.0).acos().to_degrees() > prm.max_ang || (a.r / b.r).ln().abs() > prm.max_dlogr {
        return None;
    }
    let ub = if cosab >= 0.0 { b.u } else { -b.u };
    let u = (a.u + ub).normalize();
    // the two lines in Plücker form (direction, moment = point x direction); their points at a station s along u
    let (ma, mb) = (a.c.cross(&a.u), b.c.cross(&ub));
    let at = |dir: &V, m: &V, s: f64| {
        let x0 = dir.cross(m); // the point nearest the origin
        x0 + dir * ((s - x0.dot(&u)) / dir.dot(&u))
    };
    let span = |l: &StripeLine| {
        let (e0, e1) = ((l.c + l.u * l.t0).dot(&u), (l.c + l.u * l.t1).dot(&u));
        (e0.min(e1), e0.max(e1))
    };
    let (a0, a1) = span(a);
    let (b0, b1) = span(b);
    let (lo, hi) = (a0.max(b0), a1.min(b1));
    if hi - lo < -prm.max_gap {
        return None;
    }
    let s = 0.5 * (lo + hi);
    let d = at(&ub, &mb, s) - at(&a.u, &ma, s);
    let e1 = u.cross(&if u[2].abs() < 0.9 { V::z() } else { V::x() }).normalize();
    let e2 = u.cross(&e1);
    let cov = |l: &StripeLine| {
        let w = l.beam - u * l.beam.dot(&u);
        let sb = (prm.k_beam * l.r).max(prm.sd_beam_min);
        let mut c = nalgebra::Matrix2::<f64>::identity() * (prm.sd_across * prm.sd_across + prm.sd_floor * prm.sd_floor);
        if w.norm() > 1e-6 {
            let w = w.normalize();
            let x = nalgebra::Vector2::new(w.dot(&e1), w.dot(&e2));
            c += x * x.transpose() * (sb * sb - prm.sd_across * prm.sd_across);
        }
        c
    };
    let dd = nalgebra::Vector2::new(d.dot(&e1), d.dot(&e2));
    let cinv = (cov(a) + cov(b)).try_inverse()?;
    Some(((dd.transpose() * cinv * dd)[0], e1 * dd[0] + e2 * dd[1]))
}

/// Bundles by line consistency: stripes whose lines pass within `reach` (sampled every 5 cm) are tested by
/// `line_consistency` and merged lowest chi-square first, within `prm.chi2`, never across a cannot-link.
pub fn bundles_lines(pc: &[Piece], width: &[f64], origin: &[Point], prm: &LineParams, reach: f64, near_cells: f64, depth_step: f64) -> (Vec<u32>, MergeStats, Vec<StripeLine>) {
    let lines = stripe_lines(pc, origin);
    let cell = reach.max(0.02);
    let key = |x: &V| [(x[0] / cell).floor() as i64, (x[1] / cell).floor() as i64, (x[2] / cell).floor() as i64];
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (k, l) in lines.iter().enumerate() {
        let m = ((l.t1 - l.t0) / 0.05).ceil().max(1.0) as usize;
        let mut ks: Vec<[i64; 3]> = (0..=m).map(|j| key(&(l.c + l.u * (l.t0 + (l.t1 - l.t0) * j as f64 / m as f64)))).collect();
        ks.sort_unstable();
        ks.dedup();
        for kk in ks {
            grid.entry(kk).or_default().push(k);
        }
    }
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();
    for (kk, list) in &grid {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(other) = grid.get(&[kk[0] + dx, kk[1] + dy, kk[2] + dz]) {
                        for &a in list {
                            for &b in other {
                                if a < b {
                                    pairs.insert((a, b));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut edges = Vec::new();
    for (a, b) in pairs {
        if let Some((x2, _)) = line_consistency(&lines[a], &lines[b], prm) {
            if x2 <= prm.chi2 {
                edges.push((x2, lines[a].piece, lines[b].piece));
            }
        }
    }
    let (b, st) = merge_with(pc, width, near_cells, depth_step, edges);
    (b, st, lines)
}

/// Per region and scan, against the line through `c` along the region's `u` with each scan's surfels moved by
/// `-shift[region][scan index]` (`scans` the ids, in order): `(region, scan, offset, surfels, radial)`, with offset
/// the mean of `p − R n − c` across the axis and radial the mean distance of `p` from the axis.
pub fn view_offsets(p: &[Point], n: &[Point], scan: &[i32], regions: &[Region], c: &[Point], r: &[f64], shift: &[Vec<Point>], scans: &[i32], margin: f64, min_n: usize) -> Vec<(u32, i32, Point, u32, f64)> {
    use rayon::prelude::*;
    let grid = SurfelGrid::new(p);
    let pv: Vec<V> = p.iter().map(v).collect();
    let nv: Vec<V> = n.iter().map(v).collect();
    let si: HashMap<i32, usize> = scans.iter().enumerate().map(|(k, &s)| (s, k)).collect();
    regions
        .par_iter()
        .enumerate()
        .flat_map_iter(|(k, g)| {
            let idx = grid.gather(&pv, scan, g, margin);
            let u = v(&g.u).normalize();
            let cc = v(&c[k]);
            let mut acc: HashMap<i32, (V, u32, f64)> = HashMap::new();
            for &i in &idx {
                let t = si.get(&scan[i]).and_then(|&j| shift.get(k).and_then(|s| s.get(j))).map(v).unwrap_or_else(V::zeros);
                let q = pv[i] - t - cc;
                let d = q - nv[i] * r[k];
                let e = acc.entry(scan[i]).or_insert((V::zeros(), 0, 0.0));
                e.0 += d - u * d.dot(&u);
                e.1 += 1;
                e.2 += (q - u * q.dot(&u)).norm();
            }
            let mut out: Vec<(u32, i32, Point, u32, f64)> = acc
                .into_iter()
                .filter(|(_, (_, m, _))| *m as usize >= min_n)
                .map(|(s, (x, m, rad))| (k as u32, s, pt(&(x / m as f64)), m, rad / m as f64))
                .collect();
            out.sort_by_key(|x| x.1);
            out.into_iter()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(p: Point, scan: i32, stripe: u32, line: f64, range: f64) -> Piece {
        Piece { p, u: [0.0, 0.0, 1.0], half: 0.03, r: 0.02, scan, stripe, line, col: 0.0, range }
    }

    #[test]
    fn views_merge_a_stem_but_not_across_a_cannot_link() {
        // scan 0 sees one stem as stripes 0 and 1 (a gap between), scan 1 sees it whole as stripe 0
        let mut pc = vec![piece([0.0, 0.0, 0.0], 0, 0, 0.0, 5.0), piece([0.0, 0.0, 0.3], 0, 1, 50.0, 5.0)];
        for k in 0..6 {
            pc.push(piece([0.001, 0.0, 0.06 * k as f64], 1, 0, k as f64 * 10.0, 7.0));
        }
        // a second stem 1 cm away (as scan 0 saw it: beside stripe 0 in its image, 40 cm deeper) touching scan 1's stem
        let mut q = piece([0.01, 0.0, 0.0], 0, 2, 1.0, 5.4);
        q.u = [0.17, 0.0, 0.985];
        pc.push(q);
        let w = vec![4.0; pc.len()];
        let (b, st) = bundles(&pc, &w, 0.03, 20.0, 0.5, 3.0, 0.1);
        assert_eq!(b[0], b[1]);
        assert_eq!(b[0], b[2]);
        assert_ne!(b[8], b[0], "the cannot-link must keep the deeper stem apart");
        assert!(st.blocked >= 1 && st.cannot == 1, "{st:?}");
    }

    #[test]
    fn lines_offset_along_their_beams_are_consistent_but_not_across() {
        let prm = LineParams { k_beam: 0.5, sd_beam_min: 0.005, sd_across: 0.003, sd_floor: 0.004, max_ang: 20.0, max_dlogr: 0.7, max_gap: 0.05, chi2: 9.21 };
        let l = |c: V, beam: V| StripeLine { u: V::z(), c, t0: -0.1, t1: 0.1, r: 0.03, beam, piece: 0 };
        // scan a looks along +x, scan b along +y; each reads the axis 1.5 cm along its own beam
        let a = l(V::new(0.015, 0.0, 0.0), V::x());
        let b = l(V::new(0.0, 0.015, 0.0), V::y());
        let (x2, _) = line_consistency(&a, &b, &prm).unwrap();
        assert!(x2 < 9.21, "{x2}");
        // 3 cm apart across both beams (a sideways offset): refused
        let c = l(V::new(0.0, 0.0, 0.0), V::x());
        let d = l(V::new(0.0, 0.03, 0.0), V::x());
        assert!(line_consistency(&c, &d, &prm).unwrap().0 > 9.21);
    }

    #[test]
    fn ends_link_by_direction_and_radius() {
        let e = |p: Point, d: Point, r: f64, frag: u32| End { p, d, r, frag };
        let ends = vec![
            e([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.02, 0),
            e([0.0, 0.0, 0.1], [0.0, 0.0, -1.0], 0.021, 1),
            e([0.0, 0.05, 0.1], [0.0, 0.0, -1.0], 0.05, 2),
        ];
        let (all, best) = link_ends(&ends, 0.01, 0.5, 30.0, 0.4, 20.0, 0.2);
        assert_eq!(all.len(), 1);
        assert_eq!((best[0].a, best[0].b), (0, 1));
    }
}
