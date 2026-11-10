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

/// The bundle (a root piece index) of every piece; see the module notes. Pieces share a voxel (`voxel` m) when
/// their axes, sampled every half voxel within their half-lengths, touch a common one; they agree when their axes
/// are within `max_ang` (deg) and radii within `max_dlogr`. A cannot-link joins two stripes of one scan whose pieces
/// lie within `near_cells` (+ half their widths in cells, `width`) in its image and differ in front range by more
/// than `depth_step`.
pub fn bundles(pc: &[Piece], width: &[f64], voxel: f64, max_ang: f64, max_dlogr: f64, near_cells: f64, depth_step: f64) -> (Vec<u32>, MergeStats) {
    let n = pc.len();
    let mut par: Vec<usize> = (0..n).collect();
    let skey = |p: &Piece| ((p.scan as u64) << 32) | p.stripe as u64;
    // members (stripe keys) of each root, and the stripe keys each root may not join
    let mut members: Vec<HashSet<u64>> = pc.iter().map(|p| HashSet::from([skey(p)])).collect();
    let mut cannot: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut st = MergeStats::default();
    // cannot-links, per scan, from a grid over the image
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
    // a stripe's own pieces
    let mut first: HashMap<u64, usize> = HashMap::new();
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
    // voxels touched by each piece's axis
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
    edges.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    st.edges = edges.len();
    for (_, a, b) in edges {
        match union(&mut par, &mut members, a, b, true) {
            Some(true) => st.merged += 1,
            Some(false) => st.blocked += 1,
            None => {}
        }
    }
    let out = (0..n).map(|i| find(&mut par, i) as u32).collect();
    (out, st)
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
