// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What a scanner's rays say about each QSM cylinder, independent of the
//! cylinder's claimed radius.
//!
//! For a fitted cylinder `a -> b` (e.g. from raycloudtools' `_trees.txt`,
//! whose radii are a taper prior, not a measurement), the rays of the scans
//! give, per cylinder:
//!
//! - `mu`, `tau`: the log of the median distance of its echoes from the axis,
//!   and its precision. Each echo is worth `(r / sigma)^2`, `sigma` half the
//!   beam footprint at its range combined with the ranging noise; each scan's
//!   sum is capped by a floor that more echoes don't remove (registration,
//!   wind), and scans add. The footprint (echoes sit outside the surface by a
//!   fraction of it) and the spread of the echo distances (clutter) add
//!   systematic variance. `tau = 0` where the rays don't determine the
//!   radius ([`prior_only`]).
//! - `lo`, `hi`: the beam calliper. A pulse that hit the cylinder came within
//!   `d` of the axis, so `r > d - w/2`; one that went past and echoed beyond
//!   came within `d` without hitting, so `r < d + w/2` (`w`: the footprint).
//! - counts: echoes, pulses that reached it, and passes inside a claimed
//!   radius (the audit of a fat claim).
//!
//! Every echo goes to its nearest axis within `q_max`, so pass every cylinder
//! near the ones of interest (a whole tile) and a neighbour's echoes go to the
//! neighbour. Speed comes from looking only where wood is: each scan is
//! indexed once by pulse direction ([`ScanIndex`]), each cylinder visits only
//! the direction cells its axis covers seen from the scanner, and cylinders
//! run in parallel.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::shots::Shots;
use crate::util::spatial::KdTree;
use crate::Point;

/// A scanner's beam: footprint `exit + divergence * range` and ranging noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beam {
    /// Full beam divergence (rad).
    pub divergence: f64,
    /// Beam diameter at the aperture (m).
    pub exit: f64,
    /// Ranging precision (m).
    pub range_noise: f64,
}

impl Beam {
    /// RIEGL VZ-400i (and VZ-2000i).
    pub const VZ400I: Beam = Beam { divergence: 0.35e-3, exit: 0.007, range_noise: 0.003 };

    pub fn footprint(&self, range: f64) -> f64 {
        self.exit + self.divergence * range
    }
}

/// Settings of [`evidence`]; the defaults are the calibrated ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceParams {
    pub beam: Beam,
    /// Log-radius error per scan that more echoes don't remove.
    pub floor_log: f64,
    /// Systematic log sd per unit of `(w/2) / r` (echoes outside the surface).
    pub footprint_sd: f64,
    /// Systematic log sd per unit of `spread / r` (clutter around the axis).
    pub spread_sd: f64,
    /// Below this radius an echoes-only store can't tell a twig from the
    /// leaves around it: prior only.
    pub echoes_only_min_radius: f64,
    /// An echo farther than this from every axis belongs to no cylinder (m).
    pub q_max: f64,
}

impl Default for EvidenceParams {
    fn default() -> Self {
        EvidenceParams { beam: Beam::VZ400I, floor_log: 0.05, footprint_sd: 0.9, spread_sd: 1.0, echoes_only_min_radius: 0.02, q_max: 0.5 }
    }
}

const CELL: f64 = 0.2 * std::f64::consts::PI / 180.0; // direction grid of the scan index
const NZ: usize = 900; // ceil(pi / CELL)
const NA: usize = 1800; // ceil(2 pi / CELL)
const SAMPLE: f64 = 0.02; // axis sampling for echo assignment (m)

fn cell_of(zen: f64, az: f64) -> usize {
    let iz = ((zen / CELL) as usize).min(NZ - 1);
    let ia = ((az.rem_euclid(2.0 * std::f64::consts::PI) / CELL) as usize).min(NA - 1);
    iz * NA + ia
}

/// One scan's pulses, indexed by direction: what [`evidence`] reads.
#[derive(Debug, Clone)]
pub struct ScanIndex {
    pub origin: Point,
    /// Whether the store recorded pulses without echoes (free space).
    pub misses: bool,
    dir: Vec<[f32; 3]>,
    /// Range of each pulse's last echo; infinite for a miss.
    last: Vec<f32>,
    /// Pulse `p`'s echoes are `range[echo_start[p]..echo_start[p + 1]]`.
    echo_start: Vec<u64>,
    range: Vec<f32>,
    /// Pulses sorted by direction cell, and where each cell starts in it.
    order: Vec<u32>,
    cell_start: Vec<u32>,
}

impl ScanIndex {
    /// Index the shots of one scan position. `misses`: whether the store kept
    /// pulses without echoes (an echoes-only export has none, but may have
    /// lost echoes too, which this can't see).
    pub fn new(shots: &Shots, misses: bool) -> Result<Self> {
        let n = shots.n_shots();
        let origin = shots.origin.first().copied().unwrap_or([0.0; 3]);
        if shots.origin.iter().any(|o| (0..3).any(|k| (o[k] - origin[k]).abs() > 1e-6)) {
            return Err(Error::invalid("ScanIndex: the shots come from more than one scan position"));
        }
        if n >= u32::MAX as usize {
            return Err(Error::invalid("ScanIndex: too many pulses for one scan"));
        }
        let dir: Vec<[f32; 3]> = shots.direction.iter().map(|d| [d[0] as f32, d[1] as f32, d[2] as f32]).collect();
        let mut echo_start = Vec::with_capacity(n + 1);
        let mut range = Vec::with_capacity(shots.n_echoes());
        let mut last = Vec::with_capacity(n);
        for p in 0..n {
            echo_start.push(range.len() as u64);
            let (s, c) = (shots.echo_start[p], shots.echo_count[p] as usize);
            range.extend(shots.echo_range[s..s + c].iter().map(|&r| r as f32));
            last.push(if c > 0 { shots.echo_range[s + c - 1] as f32 } else { f32::INFINITY });
        }
        echo_start.push(range.len() as u64);
        let cell: Vec<u32> = shots.direction.iter().map(|d| cell_of(d[2].clamp(-1.0, 1.0).acos(), d[0].atan2(d[1])) as u32).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.par_sort_by_key(|&p| (cell[p as usize], p));
        let mut cell_start = vec![0u32; NZ * NA + 1];
        for &c in &cell {
            cell_start[c as usize + 1] += 1;
        }
        for i in 0..NZ * NA {
            cell_start[i + 1] += cell_start[i];
        }
        Ok(ScanIndex { origin, misses, dir, last, echo_start, range, order, cell_start })
    }

    pub fn n_pulses(&self) -> usize {
        self.dir.len()
    }

    fn dir(&self, p: usize) -> Point {
        let d = self.dir[p];
        [d[0] as f64, d[1] as f64, d[2] as f64]
    }

    /// Direction cells of the rectangle a segment covers, padded by `window`.
    fn for_cells_near(&self, a: &Point, b: &Point, window: f64, mut f: impl FnMut(usize)) {
        let (z0, nz, a0, na) = rectangle(&sub(a, &self.origin), &sub(b, &self.origin), window);
        for iz in z0..z0 + nz {
            for j in 0..na {
                f(iz * NA + (a0 + j as i64).rem_euclid(NA as i64) as usize);
            }
        }
    }

    fn pulses_in(&self, cell: usize) -> &[u32] {
        &self.order[self.cell_start[cell] as usize..self.cell_start[cell + 1] as usize]
    }

    /// Pulses whose direction falls in the cell rectangle a segment covers, padded by `window`.
    fn for_pulses_near(&self, a: &Point, b: &Point, window: f64, mut f: impl FnMut(usize)) {
        self.for_cells_near(a, b, window, |c| self.pulses_in(c).iter().for_each(|&p| f(p as usize)));
    }
}

fn sub(a: &Point, b: &Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &Point) -> f64 {
    dot(a, a).sqrt()
}

/// Direction cells (zenith row start and count, azimuth column start and
/// count) that a segment `pa -> pb` (relative to the scanner) covers, padded
/// by the angle `window` subtends at its nearest point, and by one cell. A
/// pulse whose line passes within `window` of the segment points into it.
fn rectangle(pa: &Point, pb: &Point, window: f64) -> (usize, usize, i64, usize) {
    use std::f64::consts::PI;
    let ab = sub(pb, pa);
    let t = (-dot(pa, &ab) / dot(&ab, &ab).max(1e-18)).clamp(0.0, 1.0);
    let r_min = norm(&[pa[0] + t * ab[0], pa[1] + t * ab[1], pa[2] + t * ab[2]]);
    let alpha = (window / r_min.max(1e-9)).min(1.0).asin() + CELL;
    let zen_az = |p: &Point| ((p[2] / norm(p).max(1e-9)).clamp(-1.0, 1.0).acos(), p[0].atan2(p[1]));
    let ((za, aa), (zb, ab_)) = (zen_az(pa), zen_az(pb));
    let (z_lo, z_hi) = (za.min(zb) - alpha, za.max(zb) + alpha);
    let turn = (ab_ - aa + PI).rem_euclid(2.0 * PI) - PI; // the short way round from a to b
    let half = alpha / z_lo.max(0.0).sin().min(z_hi.min(PI).sin()).max(1e-9);
    // a segment passing over or under the scanner (within its window in plan) sweeps every azimuth
    let dxy = [ab[0], ab[1]];
    let s = (-(pa[0] * dxy[0] + pa[1] * dxy[1]) / (dxy[0] * dxy[0] + dxy[1] * dxy[1]).max(1e-18)).clamp(0.0, 1.0);
    let plan = ((pa[0] + s * dxy[0]).powi(2) + (pa[1] + s * dxy[1]).powi(2)).sqrt();
    let z0 = (z_lo.max(0.0) / CELL) as usize;
    let nz = ((z_hi / CELL) as usize).min(NZ - 1) + 1 - z0.min(NZ - 1);
    if half >= PI / 2.0 || plan <= window || window >= r_min {
        return (z0.min(NZ - 1), nz, 0, NA);
    }
    let a0 = ((aa + turn.min(0.0) - half) / CELL).floor() as i64;
    let a1 = ((aa + turn.max(0.0) + half) / CELL).floor() as i64;
    (z0.min(NZ - 1), nz, a0, ((a1 - a0 + 1) as usize).min(NA))
}

/// Per-cylinder evidence; see the module documentation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Evidence {
    pub mu: Vec<f64>,
    pub tau: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub n_hit: Vec<u32>,
    pub n_cross: Vec<u32>,
    pub n_contra: Vec<u32>,
}

/// Where the rays don't determine a cylinder's radius, which then rests on
/// the prior: pulses went past closer to the axis than its echoes sit (the
/// echoes are leaves around a twig, not its surface), or none of the scans
/// that saw it recorded misses and it is thinner than echoes alone can tell
/// from clutter.
pub fn prior_only(radius: f64, hi: f64, seen_with_misses: bool, p: &EvidenceParams) -> bool {
    hi < radius || (!seen_with_misses && radius < p.echoes_only_min_radius)
}

/// Median as NumPy takes it (the mean of the two middle values).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    (v[(n - 1) / 2] + v[n / 2]) / 2.0
}

struct Echo {
    cyl: u32,
    scan: u32,
    pulse: u32,
    q: f64,
    range: f64,
}

/// The evidence the `scans` give about cylinders `a[i] -> b[i]`; `r_claim`,
/// if given, the claimed radii to audit.
pub fn evidence(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_claim: Option<&[f64]>, p: &EvidenceParams) -> Result<Evidence> {
    let n = a.len();
    if b.len() != n || r_claim.is_some_and(|r| r.len() != n) {
        return Err(Error::invalid("evidence: a, b and r_claim must have one row per cylinder"));
    }
    let len: Vec<f64> = (0..n).map(|i| norm(&sub(&b[i], &a[i])).max(1e-9)).collect();
    let u: Vec<Point> = (0..n).map(|i| {
        let d = sub(&b[i], &a[i]);
        [d[0] / len[i], d[1] / len[i], d[2] / len[i]]
    }).collect();
    // segment distance and position along it
    let seg = |c: usize, q: &Point| -> (f64, f64) {
        let w = sub(q, &a[c]);
        let t = dot(&w, &u[c]);
        let tc = t.clamp(0.0, len[c]);
        (norm(&[w[0] - tc * u[c][0], w[1] - tc * u[c][1], w[2] - tc * u[c][2]]), t)
    };

    // 1. echoes near any axis -> their nearest cylinder (axis samples every 2 cm, then the exact segment)
    let mut owner = Vec::new();
    let mut samples = Vec::new();
    for c in 0..n {
        let k = ((len[c] / SAMPLE) as usize).max(1) + 1;
        for j in 0..k {
            let f = j as f64 / (k - 1) as f64 * len[c];
            samples.push([a[c][0] + f * u[c][0], a[c][1] + f * u[c][1], a[c][2] + f * u[c][2]]);
            owner.push(c);
        }
    }
    let kd = KdTree::new(&samples);
    let (kd, owner, seg, len_ref) = (&kd, &owner, &seg, &len); // shared by the parallel closures
    let mut echoes: Vec<Echo> = Vec::new();
    for (si, s) in scans.iter().enumerate() {
        let mut marked = vec![false; NZ * NA];
        for c in 0..n {
            s.for_cells_near(&a[c], &b[c], p.q_max, |cell| marked[cell] = true);
        }
        let cells: Vec<usize> = (0..NZ * NA).filter(|&c| marked[c]).collect();
        let found: Vec<Echo> = cells.par_iter().flat_map_iter(|&cell| s.pulses_in(cell).iter().map(|&pu| pu as usize)).flat_map_iter(|pu| {
            let d = s.dir(pu);
            let (e0, e1) = (s.echo_start[pu] as usize, s.echo_start[pu + 1] as usize);
            (e0..e1).filter_map(move |e| {
                let rng = s.range[e] as f64;
                let q = [s.origin[0] + rng * d[0], s.origin[1] + rng * d[1], s.origin[2] + rng * d[2]];
                // every cylinder with a point within q_max has a sample within q_max + SAMPLE / 2: the
                // nearest of those the echo projects onto (ties to the lower index)
                let mut best: Option<(f64, usize)> = None;
                for (i, _) in kd.within(&q, p.q_max + SAMPLE / 2.0) {
                    let c = owner[i];
                    let (dist, t) = seg(c, &q);
                    if t >= 0.0 && t <= len_ref[c] && dist < p.q_max && best.is_none_or(|(bd, bc)| dist < bd || (dist == bd && c < bc)) {
                        best = Some((dist, c));
                    }
                }
                best.map(|(q, c)| Echo { cyl: c as u32, scan: si as u32, pulse: pu as u32, q, range: rng })
            }).collect::<Vec<_>>()
        }).collect();
        echoes.extend(found);
    }
    echoes.sort_by_key(|e| (e.cyl, e.scan, e.pulse));
    let mut at = vec![0usize; n + 1];
    for e in &echoes {
        at[e.cyl as usize + 1] += 1;
    }
    for i in 0..n {
        at[i + 1] += at[i];
    }

    let ns = scans.len();
    let mut out = Evidence {
        mu: vec![f64::NAN; n], tau: vec![0.0; n], lo: vec![0.0; n], hi: vec![f64::INFINITY; n],
        n_hit: (0..n).map(|c| (at[c + 1] - at[c]) as u32).collect(), n_cross: vec![0; n], n_contra: vec![0; n],
    };
    let mut med = vec![f64::NAN; n];
    let mut seen_with_misses = vec![false; n];
    for c in 0..n {
        let mine = &echoes[at[c]..at[c + 1]];
        if mine.is_empty() {
            continue;
        }
        let m = median(&mut mine.iter().map(|e| e.q).collect::<Vec<_>>());
        let w = median(&mut mine.iter().map(|e| p.beam.footprint(e.range)).collect::<Vec<_>>());
        let spread = 1.4826 * median(&mut mine.iter().map(|e| (e.q - m).abs()).collect::<Vec<_>>());
        let mut per_scan = vec![0.0; ns];
        for e in mine {
            let sigma = (p.beam.footprint(e.range) / 2.0).hypot(p.beam.range_noise);
            per_scan[e.scan as usize] += (m / sigma).powi(2);
            seen_with_misses[c] |= scans[e.scan as usize].misses;
        }
        let tau: f64 = per_scan.iter().filter(|&&t| t > 0.0).map(|t| 1.0 / (1.0 / t + p.floor_log * p.floor_log)).sum();
        out.tau[c] = 1.0 / (1.0 / tau + (p.footprint_sd * w / 2.0 / m).powi(2) + (p.spread_sd * spread / m).powi(2));
        out.mu[c] = m.max(1e-4).ln();
        med[c] = m;
    }

    // 2. pulses passing near each axis, within a window set by its echoes: the calliper and the audit
    let hit_keys: Vec<Vec<u64>> = (0..ns).map(|si| {
        let mut k: Vec<u64> = echoes.iter().filter(|e| e.scan as usize == si).map(|e| ((e.pulse as u64) << 32) | e.cyl as u64).collect();
        k.sort_unstable();
        k.dedup();
        k
    }).collect();
    let per_cyl: Vec<(u32, f64, f64, u32)> = (0..n).into_par_iter().map(|c| {
        let window = if med[c].is_nan() { 0.05 } else { (3.0 * med[c]).max(0.05) };
        let (mut hits, mut passes, mut contra) = (Vec::new(), Vec::new(), 0u32);
        for (si, s) in scans.iter().enumerate() {
            s.for_pulses_near(&a[c], &b[c], window, |pu| {
                let d = s.dir(pu);
                let w0 = sub(&s.origin, &a[c]);
                let (bb, dd, ee) = (dot(&d, &u[c]), dot(&w0, &u[c]), dot(&d, &w0));
                let den = (1.0 - bb * bb).max(1e-12);
                let (t_ray, s_ax) = ((bb * dd - ee) / den, (dd - bb * ee) / den);
                let sc = s_ax.clamp(0.0, len[c]);
                let gap = [
                    s.origin[0] + t_ray * d[0] - (a[c][0] + sc * u[c][0]),
                    s.origin[1] + t_ray * d[1] - (a[c][1] + sc * u[c][1]),
                    s.origin[2] + t_ray * d[2] - (a[c][2] + sc * u[c][2]),
                ];
                let dist = norm(&gap);
                let last = s.last[pu] as f64;
                if !(dist < window && s_ax > 0.0 && s_ax < len[c] && t_ray > 0.0 && last >= t_ray - window) {
                    return;
                }
                let w = p.beam.footprint(t_ray);
                if hit_keys[si].binary_search(&(((pu as u64) << 32) | c as u64)).is_ok() {
                    hits.push(dist - w / 2.0);
                } else if last > t_ray {
                    passes.push(dist + w / 2.0);
                    if r_claim.is_some_and(|r| dist < r[c] - w / 2.0) {
                        contra += 1;
                    }
                }
            });
        }
        let n_cross = (hits.len() + passes.len()) as u32;
        let (lo, hi) = if n_cross > 0 { let (_, lo, hi) = calliper(&hits, &passes, 2); (lo, hi) } else { (0.0, f64::INFINITY) };
        (n_cross, lo, hi, contra)
    }).collect();
    for (c, (n_cross, lo, hi, contra)) in per_cyl.into_iter().enumerate() {
        out.n_cross[c] = n_cross;
        out.lo[c] = lo;
        out.hi[c] = hi;
        out.n_contra[c] = contra;
        if out.n_hit[c] > 0 && prior_only(out.mu[c].exp(), hi, seen_with_misses[c], p) {
            out.tau[c] = 0.0;
        }
    }
    Ok(out)
}

/// The radius as the distance threshold that best separates pulses that hit
/// (`hit_d`, closer to the axis) from pulses that passed (`pass_d`, farther):
/// `(best threshold, and the range of thresholds within `slack`
/// misclassified pulses of the best)`. A few clutter hits or leaks only widen
/// the range.
pub fn calliper(hit_d: &[f64], pass_d: &[f64], slack: usize) -> (f64, f64, f64) {
    if hit_d.is_empty() {
        let hi = pass_d.iter().copied().fold(f64::INFINITY, f64::min);
        return (f64::NAN, 0.0, hi);
    }
    if pass_d.is_empty() {
        return (f64::NAN, median(&mut hit_d.to_vec()), f64::INFINITY);
    }
    let mut x: Vec<(f64, bool)> = hit_d.iter().map(|&d| (d, true)).chain(pass_d.iter().map(|&d| (d, false))).collect();
    x.sort_by(|p, q| p.0.total_cmp(&q.0)); // stable, like NumPy's stable argsort
    let m = x.len();
    let n_hits = hit_d.len();
    // errors with the threshold just after position k: passes at or below it, hits above it
    let mut err = Vec::with_capacity(m + 1);
    let (mut passes_below, mut hits_below) = (0usize, 0usize);
    err.push(n_hits);
    for &(_, h) in &x {
        if h { hits_below += 1 } else { passes_below += 1 }
        err.push(passes_below + n_hits - hits_below);
    }
    let thr: Vec<f64> = std::iter::once(x[0].0 - 1e-4).chain(x.windows(2).map(|w| (w[0].0 + w[1].0) / 2.0)).chain(std::iter::once(x[m - 1].0 + 1e-4)).collect();
    let best = *err.iter().min().unwrap();
    let mut at_best: Vec<f64> = (0..=m).filter(|&k| err[k] == best).map(|k| thr[k]).collect();
    let ok: Vec<f64> = (0..=m).filter(|&k| err[k] <= best + slack).map(|k| thr[k]).collect();
    let lo = ok.iter().copied().fold(f64::INFINITY, f64::min).max(0.0);
    let hi = ok.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (median(&mut at_best), lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scan of pulses on a fine grid from the origin towards a vertical
    /// cylinder of radius `r` at distance 5 m, with a wall behind it.
    fn scene(r: f64, misses: bool) -> (ScanIndex, Point, Point) {
        let (cx, cy) = (0.0, 5.0);
        let mut shots = Shots::default();
        let step = 0.0005f64; // rad
        let mut az = -0.03f64;
        while az <= 0.03 {
            let mut el = -0.05f64;
            while el <= 0.05 {
                let zen = std::f64::consts::FRAC_PI_2 - el;
                let d = [zen.sin() * az.sin(), zen.sin() * az.cos(), zen.cos()];
                // ray-circle intersection in plan, then the wall at y = 8
                let (dx, dy) = (d[0], d[1]);
                let h = (dx * dx + dy * dy).sqrt();
                let (ux, uy) = (dx / h, dy / h);
                let tca = ux * cx + uy * cy;
                let d2 = cx * cx + cy * cy - tca * tca;
                let hit = d2 <= r * r && (-1.0..=1.0).contains(&(d[2] * (tca / h)));
                shots.echo_start.push(shots.echo_range.len());
                if hit {
                    shots.echo_range.push((tca - (r * r - d2).sqrt()) / h);
                    shots.echo_count.push(1);
                } else if misses || el.abs() < 0.04 {
                    shots.echo_range.push(8.0 / dy);
                    shots.echo_count.push(1);
                } else {
                    shots.echo_count.push(0);
                }
                shots.origin.push([0.0; 3]);
                shots.direction.push(d);
                el += step;
            }
            az += step;
        }
        (ScanIndex::new(&shots, misses).unwrap(), [cx, cy, -1.0], [cx, cy, 1.0])
    }

    #[test]
    fn a_measured_cylinder_gets_its_radius_and_a_calliper_around_it() {
        let r = 0.05;
        let (s, a, b) = scene(r, true);
        let e = evidence(&[&s], &[a], &[b], Some(&[1.5 * r]), &EvidenceParams::default()).unwrap();
        // echoes lie on the near surface, so their median axis distance is about r
        assert!((e.mu[0].exp() / r - 1.0).abs() < 0.05, "mu {}", e.mu[0].exp());
        assert!(e.tau[0] > 0.0);
        assert!(e.lo[0] < r + 0.005 && e.hi[0] > r - 0.005, "calliper [{}, {}]", e.lo[0], e.hi[0]);
        assert!(e.n_hit[0] > 100 && e.n_cross[0] > e.n_hit[0]);
        assert!(e.n_contra[0] > 0, "a 50 % fat claim is contradicted by passes");
    }

    #[test]
    fn an_honest_claim_is_not_contradicted_and_thin_wood_without_misses_is_prior_only() {
        let (s, a, b) = scene(0.05, true);
        let e = evidence(&[&s], &[a], &[b], Some(&[0.05]), &EvidenceParams::default()).unwrap();
        assert_eq!(e.n_contra[0], 0);
        let (s, a, b) = scene(0.01, false);
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default()).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0, "1 cm in an echoes-only store rests on the prior");
    }

    #[test]
    fn calliper_matches_a_clean_separation() {
        let (best, lo, hi) = calliper(&[0.01, 0.02, 0.03], &[0.05, 0.06], 0);
        assert!((best - 0.04).abs() < 1e-12 && (lo - 0.04).abs() < 1e-12 && (hi - 0.04).abs() < 1e-12);
    }
}
