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
//! - `mu`, `tau`: the log of the median distance of its hits from the axis,
//!   and its precision. Each hit is worth `(r / sigma)^2`, `sigma` half the
//!   beam footprint at its range combined with the ranging noise; each scan's
//!   sum is capped by a floor that more echoes don't remove (registration,
//!   wind), and scans add. The footprint (echoes sit outside the surface by a
//!   fraction of it) and the spread of the echo distances (clutter) add
//!   systematic variance. `tau = 0` where the rays don't determine the
//!   radius, which then rests on the prior: pulses went past closer to the
//!   axis than its echoes sit (the echoes are leaves around a twig, not its
//!   surface); none of the scans that saw it recorded misses and it is
//!   thinner than echoes alone can tell from clutter; or its echoes sit
//!   closer to the axis than a fraction of the beam footprint (wood the beam
//!   is too wide to measure). So `tau == 0 && n_hit > 0` is "prior only".
//! - `lo`, `hi`: the beam calliper. A pulse that hit the cylinder came within
//!   `d` of the axis, so `r > d - w/2`; one that passed came within `d`
//!   without hitting, so `r < d + w/2` (`w`: the footprint).
//! - counts: `n_hit` (hits), `n_cross` (hits and passes in the window) and
//!   `n_contra` (passes inside a claimed radius). A tolerant audit of a fat
//!   claim is `n_contra >= max(2, 0.05 * n_cross)`: off-centre fitted axes
//!   leave honest claims a few passes inside; the verdict is the caller's.
//!
//! Ray first: one enumeration of the (pulse, cylinder) pairs whose line
//! passes within `window_cap` of the axis, with the echoes of those pulses
//! near the axis, then a few rounds over these tables:
//!
//! 1. each echo goes to the one cylinder whose surface it lies nearest,
//!    `|q - r|`, among those whose window holds it (the first round: the
//!    nearest axis); a pulse hits a cylinder at most once;
//! 2. `r` is the median distance of a cylinder's hits;
//! 3. a pair in the window is a hit, a pass (no echo of the pulse near the
//!    axis, and the pulse went on), or neither (its echo near this axis went
//!    to a neighbour or a fork: no evidence about this cylinder);
//! 4. the next window is `max(1.5 hi, hi + 2 cm)` where passes bound the
//!    calliper, else `2 r + 2 cm`, within `[window_min, window_cap]`.
//!
//! The window comes from the rays, so a claim can't move it and leaves beyond
//! the calliper can't inflate it. Pass every cylinder near the ones of
//! interest (a whole tile), so a neighbour's echoes go to the neighbour.
//! Speed comes from looking only where wood is: each scan is indexed once by
//! pulse direction ([`ScanIndex`]), each cylinder visits only the direction
//! cells its axis covers seen from the scanner, and cylinders run in
//! parallel.

use std::path::Path;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::io::shots::{angles, direction, Rays, ShotsFile};
use crate::shots::Shots;
use crate::transform::{add, dot, norm, scale, sub};
use crate::util::numeric::median;
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
    /// Where the hits' median distance from the axis is below this many beam
    /// footprints, the beam is too wide to measure the wood: prior only. At
    /// 0.5 the radius the echoes imply (they sit about `w/2` outside the
    /// surface) is not positive. 1 (wood thinner than the beam) also drops
    /// well-measured 1-2 cm wood and leaves the clutter-fattened twigs.
    pub min_footprints: f64,
    /// The widest window (m): no echo farther from an axis is a candidate
    /// hit, so it must exceed the largest radius to be measured.
    pub window_cap: f64,
    /// The narrowest window (m).
    pub window_min: f64,
    /// Rounds of assignment and windowing.
    pub rounds: usize,
}

impl Default for EvidenceParams {
    fn default() -> Self {
        EvidenceParams {
            beam: Beam::VZ400I,
            floor_log: 0.05,
            footprint_sd: 0.9,
            spread_sd: 1.0,
            echoes_only_min_radius: 0.02,
            min_footprints: 0.5,
            window_cap: 0.25,
            window_min: 0.02,
            rounds: 3,
        }
    }
}

const CELL: f64 = 0.2 * std::f64::consts::PI / 180.0; // direction grid of the scan index
const NZ: usize = 900; // ceil(pi / CELL)
const NA: usize = 1800; // ceil(2 pi / CELL)

fn cell_of(zen: f64, az: f64) -> usize {
    let iz = ((zen / CELL) as usize).min(NZ - 1);
    let ia = ((az.rem_euclid(2.0 * std::f64::consts::PI) / CELL) as usize).min(NA - 1);
    iz * NA + ia
}

/// A pool of `threads` threads (0: one per core).
fn pool(threads: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| Error::invalid(format!("thread pool: {e}")))
}

/// One scan's pulses, indexed by direction: what [`evidence`] reads. About
/// 16 bytes per pulse and 4 per echo, plus a 6.5 MB cell table.
#[derive(Debug, Clone)]
pub struct ScanIndex {
    pub origin: Point,
    /// Whether the store recorded pulses without echoes (free space).
    pub misses: bool,
    /// Pulses in direction-cell order: cell `c` holds `cell_start[c]..cell_start[c + 1]`.
    dir: Vec<[f32; 3]>,
    /// Pulse `p`'s echoes are `range[echo_start[p]..echo_start[p + 1]]`, nearest first.
    echo_start: Vec<u32>,
    range: Vec<f32>,
    cell_start: Vec<u32>,
}

impl ScanIndex {
    /// Index a single-position Shots file, reading only angles and ranges,
    /// row groups in parallel on `threads` threads (0: one per core).
    /// `misses`: whether the store kept pulses without echoes (an echoes-only
    /// export has none, but may have lost echoes too, which this can't see).
    pub fn from_file(path: impl AsRef<Path>, misses: bool, threads: usize) -> Result<Self> {
        let file = ShotsFile::open(path)?;
        let [origin] = *file.scans() else {
            return Err(Error::invalid("ScanIndex: the shots file must hold exactly one scan position"));
        };
        pool(threads)?.install(|| {
            // a handle per row group: handles share no file position
            let groups: Vec<Rays> = (0..file.n_groups()).into_par_iter().map(|g| file.reopen()?.read_rays(g)).collect::<Result<_>>()?;
            let mut rays = Rays::default();
            for g in groups {
                rays.zenith.extend(g.zenith);
                rays.azimuth.extend(g.azimuth);
                rays.echo_count.extend(g.echo_count);
                rays.range.extend(g.range);
            }
            Self::build(origin, misses, rays)
        })
    }

    /// Index the shots of one scan position held in memory.
    pub fn new(shots: &Shots, misses: bool) -> Result<Self> {
        let origin = shots.origin.first().copied().unwrap_or([0.0; 3]);
        if shots.origin.iter().any(|o| (0..3).any(|k| (o[k] - origin[k]).abs() > 1e-6)) {
            return Err(Error::invalid("ScanIndex: the shots come from more than one scan position"));
        }
        let (zenith, azimuth) = shots.direction.iter().map(|d| {
            let (z, a) = angles(d);
            (z as f32, a as f32)
        }).unzip();
        let range = (0..shots.n_shots()).flat_map(|p| &shots.echo_range[shots.echo_start[p]..shots.echo_start[p] + shots.echo_count[p] as usize]).map(|&r| r as f32).collect();
        Self::build(origin, misses, Rays { zenith, azimuth, echo_count: shots.echo_count.clone(), range })
    }

    /// Put the pulses in direction-cell order (a stable counting sort).
    fn build(origin: Point, misses: bool, rays: Rays) -> Result<Self> {
        let n = rays.zenith.len();
        if n >= u32::MAX as usize || rays.range.len() >= u32::MAX as usize {
            return Err(Error::invalid("ScanIndex: too many pulses or echoes for one scan"));
        }
        if misses && n > 0 && !rays.echo_count.contains(&0) {
            return Err(Error::invalid("ScanIndex: misses=true, but every pulse has an echo, as in an echoes-only store"));
        }
        let cell: Vec<u32> = rays.zenith.par_iter().zip(&rays.azimuth).map(|(&z, &a)| cell_of(z as f64, a as f64) as u32).collect();
        let mut cell_start = vec![0u32; NZ * NA + 1];
        for &c in &cell {
            cell_start[c as usize + 1] += 1;
        }
        for i in 0..NZ * NA {
            cell_start[i + 1] += cell_start[i];
        }
        let mut next = cell_start.clone();
        let mut src = vec![0u32; n]; // the pulse of `rays` at each place
        for (p, &c) in cell.iter().enumerate() {
            src[next[c as usize] as usize] = p as u32;
            next[c as usize] += 1;
        }
        drop((cell, next));
        let mut from = Vec::with_capacity(n + 1); // where each pulse's echoes start in `rays.range`
        from.push(0u32);
        for &k in &rays.echo_count {
            from.push(from[from.len() - 1] + k);
        }
        let mut echo_start = Vec::with_capacity(n + 1);
        echo_start.push(0u32);
        for &p in &src {
            echo_start.push(echo_start[echo_start.len() - 1] + rays.echo_count[p as usize]);
        }
        let dir = src.par_iter().map(|&p| direction(rays.zenith[p as usize] as f64, rays.azimuth[p as usize] as f64).map(|x| x as f32)).collect();
        let range = src.par_iter().flat_map_iter(|&p| &rays.range[from[p as usize] as usize..from[p as usize + 1] as usize]).copied().collect();
        Ok(ScanIndex { origin, misses, dir, echo_start, range, cell_start })
    }

    pub fn n_pulses(&self) -> usize {
        self.dir.len()
    }

    fn dir(&self, p: usize) -> Point {
        self.dir[p].map(|x| x as f64)
    }

    fn echoes(&self, p: usize) -> &[f32] {
        &self.range[self.echo_start[p] as usize..self.echo_start[p + 1] as usize]
    }

    /// Pulses whose direction falls in the cell rectangle a segment covers, padded by `window`.
    fn for_pulses_near(&self, a: &Point, b: &Point, window: f64, mut f: impl FnMut(usize)) {
        let (z0, nz, a0, na) = rectangle(&sub(a, &self.origin), &sub(b, &self.origin), window);
        for iz in z0..z0 + nz {
            for j in 0..na {
                let cell = iz * NA + (a0 + j as i64).rem_euclid(NA as i64) as usize;
                (self.cell_start[cell] as usize..self.cell_start[cell + 1] as usize).for_each(&mut f);
            }
        }
    }
}

/// Direction cells (zenith row start and count, azimuth column start and
/// count) that a segment `pa -> pb` (relative to the scanner) covers, padded
/// by the angle `window` subtends at its nearest point, and by one cell. A
/// pulse whose line passes within `window` of the segment points into it.
fn rectangle(pa: &Point, pb: &Point, window: f64) -> (usize, usize, i64, usize) {
    use std::f64::consts::PI;
    let ab = sub(pb, pa);
    let t = (-dot(pa, &ab) / dot(&ab, &ab).max(1e-18)).clamp(0.0, 1.0);
    let r_min = norm(&add(pa, &scale(&ab, t)));
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

/// A pulse whose line passes within the window cap of a cylinder's axis,
/// closest at `d` (`t` along the ray) inside the segment and ahead of the
/// scanner. Its echoes near the axis are the cylinder's candidates
/// `cand..cand + n_cand`.
struct Pair {
    cand: u32,
    n_cand: u32,
    d: f32,
    t: f32,
    /// The pulse went on past `t`: its last echo is beyond, or it missed.
    beyond: bool,
}

/// An echo `q` from a cylinder's axis, projecting onto the segment: a
/// candidate hit. `echo` indexes the scan's ranges.
struct Cand {
    scan: u32,
    pulse: u32,
    echo: u32,
    q: f32,
    range: f32,
}

/// One cylinder's result of a round.
struct Round {
    med: f64,
    mu: f64,
    tau: f64,
    lo: f64,
    hi: f64,
    n_hit: u32,
    n_cross: u32,
    n_contra: u32,
}

/// The evidence the `scans` give about cylinders `a[i] -> b[i]`; `r_claim`,
/// if given, the claimed radii to audit. Runs on `threads` threads (0: one
/// per core).
pub fn evidence(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_claim: Option<&[f64]>, p: &EvidenceParams, threads: usize) -> Result<Evidence> {
    let n = a.len();
    if b.len() != n || r_claim.is_some_and(|r| r.len() != n) {
        return Err(Error::invalid("evidence: a, b and r_claim must have one row per cylinder"));
    }
    pool(threads)?.install(|| Ok(evidence_on(scans, a, b, r_claim, p)))
}

fn evidence_on(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_claim: Option<&[f64]>, p: &EvidenceParams) -> Evidence {
    let n = a.len();
    let cap = p.window_cap;
    let len: Vec<f64> = (0..n).map(|i| norm(&sub(&b[i], &a[i])).max(1e-9)).collect();
    let u: Vec<Point> = (0..n).map(|i| sub(&b[i], &a[i]).map(|x| x / len[i])).collect();

    // 1. once: the (pulse, cylinder) pairs within the cap, and the echoes of their pulses near the axis
    let found: Vec<(Vec<Pair>, Vec<Cand>)> = (0..n).into_par_iter().map(|c| {
        let (mut pairs, mut cands) = (Vec::new(), Vec::new());
        for (si, s) in scans.iter().enumerate() {
            let w0 = sub(&s.origin, &a[c]);
            let dd = dot(&w0, &u[c]);
            s.for_pulses_near(&a[c], &b[c], cap, |pu| {
                let d = s.dir(pu);
                let (bb, ee) = (dot(&d, &u[c]), dot(&d, &w0));
                let den = (1.0 - bb * bb).max(1e-12);
                let (t_ray, s_ax) = ((bb * dd - ee) / den, (dd - bb * ee) / den);
                let dist = norm(&sub(&add(&w0, &scale(&d, t_ray)), &scale(&u[c], s_ax.clamp(0.0, len[c]))));
                if !(dist < cap && t_ray > 0.0) {
                    return;
                }
                let first = cands.len();
                for e in s.echo_start[pu] as usize..s.echo_start[pu + 1] as usize {
                    let w = add(&w0, &scale(&d, s.range[e] as f64)); // the echo, from a
                    let t = dot(&w, &u[c]);
                    let q = norm(&sub(&w, &scale(&u[c], t)));
                    if t >= 0.0 && t <= len[c] && q < cap {
                        cands.push(Cand { scan: si as u32, pulse: pu as u32, echo: e as u32, q: q as f32, range: s.range[e] });
                    }
                }
                let last = s.echoes(pu).last().map_or(f64::INFINITY, |&r| r as f64); // infinite for a miss
                // a pulse that stopped short without an echo near the axis is neither a hit nor a pass
                if s_ax > 0.0 && s_ax < len[c] && (last > t_ray || cands.len() > first) {
                    pairs.push(Pair { cand: first as u32, n_cand: (cands.len() - first) as u32, d: dist as f32, t: t_ray as f32, beyond: last > t_ray });
                }
            });
        }
        (pairs, cands)
    }).collect();
    let mut cand_at = vec![0usize; n + 1];
    for (c, (_, cands)) in found.iter().enumerate() {
        cand_at[c + 1] = cand_at[c] + cands.len();
    }
    let (pairs, cands): (Vec<Vec<Pair>>, Vec<Vec<Cand>>) = found.into_iter().unzip();
    let cands: Vec<Cand> = cands.into_iter().flatten().collect();
    // each echo's candidates together, cylinders ascending: (scan, echo, cylinder, candidate)
    let mut by_echo: Vec<(u32, u32, u32, u32)> = (0..n).into_par_iter().flat_map_iter(|c| {
        let cands = &cands;
        (cand_at[c]..cand_at[c + 1]).map(move |i| (cands[i].scan, cands[i].echo, c as u32, i as u32))
    }).collect();
    by_echo.par_sort_unstable();

    // 2. rounds over the tables: assign the echoes, classify the pairs, and set each window from its calliper
    let (mut r_hat, mut win) = (vec![0.0; n], vec![cap; n]);
    let mut out = Vec::new();
    for _ in 0..p.rounds.max(1) {
        // an echo goes to the cylinder whose surface it lies nearest (ties: the lower index), among those
        // whose window holds it
        let winners: Vec<u32> = by_echo.par_chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)).filter_map(|group| {
            let mut best: Option<(f64, u32)> = None;
            for &(_, _, c, i) in group {
                let q = cands[i as usize].q as f64;
                let resid = (q - r_hat[c as usize]).abs();
                if q < win[c as usize] && best.is_none_or(|(br, _)| resid < br) {
                    best = Some((resid, i));
                }
            }
            best.map(|(_, i)| i)
        }).collect();
        let mut won = vec![false; cands.len()];
        for i in winners {
            won[i as usize] = true;
        }
        out = (0..n).into_par_iter().map(|c| {
            let at = cand_at[c]..cand_at[c + 1];
            cylinder(&pairs[c], &cands[at.clone()], &won[at], r_hat[c], win[c], r_claim.map(|r| r[c]), scans, p)
        }).collect::<Vec<Round>>();
        for c in 0..n {
            let (r, hi) = (out[c].med, out[c].hi);
            let w = if hi.is_finite() && hi > 0.0 { (1.5 * hi).max(hi + 0.02) } else { 2.0 * r + 0.02 };
            (r_hat[c], win[c]) = (r, w.clamp(p.window_min, cap));
        }
    }
    Evidence {
        mu: out.iter().map(|o| o.mu).collect(),
        tau: out.iter().map(|o| o.tau).collect(),
        lo: out.iter().map(|o| o.lo).collect(),
        hi: out.iter().map(|o| o.hi).collect(),
        n_hit: out.iter().map(|o| o.n_hit).collect(),
        n_cross: out.iter().map(|o| o.n_cross).collect(),
        n_contra: out.iter().map(|o| o.n_contra).collect(),
    }
}

/// One cylinder in a round: its hits (per pulse, the echo it won nearest its
/// surface `r_prev`), its pairs within the window `win` as hits, passes or
/// neither, and what they say.
#[allow(clippy::too_many_arguments)]
fn cylinder(pairs: &[Pair], cands: &[Cand], won: &[bool], r_prev: f64, win: f64, r_claim: Option<f64>, scans: &[&ScanIndex], p: &EvidenceParams) -> Round {
    let resid = |k: usize| (cands[k].q as f64 - r_prev).abs();
    let mut hits: Vec<&Cand> = Vec::new();
    for run in (0..cands.len()).collect::<Vec<_>>().chunk_by(|&x, &y| (cands[x].scan, cands[x].pulse) == (cands[y].scan, cands[y].pulse)) {
        // a pulse's candidates are consecutive; the first of equals wins
        let best = run.iter().copied().filter(|&k| won[k]).fold(None, |b: Option<usize>, k| if b.is_none_or(|b| resid(k) < resid(b)) { Some(k) } else { b });
        hits.extend(best.map(|k| &cands[k]));
    }
    let (mut hit_d, mut pass_d, mut n_contra) = (Vec::new(), Vec::new(), 0u32);
    for pr in pairs.iter().filter(|pr| (pr.d as f64) < win) {
        let mine = pr.cand as usize..(pr.cand + pr.n_cand) as usize;
        let (d, w) = (pr.d as f64, p.beam.footprint(pr.t as f64));
        if mine.clone().any(|k| won[k]) {
            hit_d.push(d - w / 2.0);
        } else if pr.beyond && !mine.into_iter().any(|k| (cands[k].q as f64) < win) {
            // (a pulse with an echo near the axis that went to a neighbour or a fork is neither)
            pass_d.push(d + w / 2.0);
            n_contra += r_claim.is_some_and(|r| d < r - w / 2.0) as u32;
        }
    }
    let n_cross = (hit_d.len() + pass_d.len()) as u32;
    let (lo, hi) = if n_cross > 0 { calliper(&hit_d, &pass_d, 2) } else { (0.0, f64::INFINITY) };
    let mut out = Round { med: 0.0, mu: f64::NAN, tau: 0.0, lo, hi, n_hit: hits.len() as u32, n_cross, n_contra };
    if hits.is_empty() {
        return out;
    }
    let m = median(&hits.iter().map(|e| e.q as f64).collect::<Vec<_>>());
    let w = median(&hits.iter().map(|e| p.beam.footprint(e.range as f64)).collect::<Vec<_>>());
    let spread = 1.4826 * median(&hits.iter().map(|e| (e.q as f64 - m).abs()).collect::<Vec<_>>());
    let mut per_scan = vec![0.0; scans.len()];
    let mut seen_with_misses = false;
    for e in &hits {
        let sigma = (p.beam.footprint(e.range as f64) / 2.0).hypot(p.beam.range_noise);
        per_scan[e.scan as usize] += (m / sigma).powi(2);
        seen_with_misses |= scans[e.scan as usize].misses;
    }
    out.med = m;
    out.mu = m.max(1e-4).ln();
    let r = out.mu.exp();
    // prior only (module docs): passes inside the echoes, thin wood seen only by echoes-only scans, or wood
    // the beam is too wide to measure
    if !(hi < r || (!seen_with_misses && r < p.echoes_only_min_radius) || m < p.min_footprints * w) {
        let tau: f64 = per_scan.iter().filter(|&&t| t > 0.0).map(|t| 1.0 / (1.0 / t + p.floor_log * p.floor_log)).sum();
        out.tau = 1.0 / (1.0 / tau + (p.footprint_sd * w / 2.0 / m).powi(2) + (p.spread_sd * spread / m).powi(2));
    }
    out
}

/// The radius as a distance threshold separating pulses that hit (`hit_d`,
/// closer to the axis) from pulses that passed (`pass_d`, farther): the range
/// `(lo, hi)` of thresholds within `slack` misclassified pulses of the best.
/// A few clutter hits or leaks only widen the range.
pub fn calliper(hit_d: &[f64], pass_d: &[f64], slack: usize) -> (f64, f64) {
    if hit_d.is_empty() {
        return (0.0, pass_d.iter().copied().fold(f64::INFINITY, f64::min));
    }
    if pass_d.is_empty() {
        return (median(hit_d), f64::INFINITY);
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
    let ok = (0..=m).filter(|&k| err[k] <= best + slack).map(|k| thr[k]);
    let (lo, hi) = ok.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| (lo.min(t), hi.max(t)));
    (lo.max(0.0), hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::shots::{write_shots, ShotsWriteOptions};

    /// Pulses on a fine grid from the origin towards a vertical cylinder of
    /// radius `r` at distance 5 m, with a wall behind it. With `misses` the
    /// pulses over and under the wall are kept, without echoes; otherwise
    /// the wall's top and bottom are lost too.
    fn shots(r: f64, misses: bool) -> Shots {
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
                } else if el.abs() < 0.04 || (misses && el.abs() < 0.045) {
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
        shots
    }

    fn scene(r: f64, misses: bool) -> (ScanIndex, Point, Point) {
        (ScanIndex::new(&shots(r, misses), misses).unwrap(), [0.0, 5.0, -1.0], [0.0, 5.0, 1.0])
    }

    #[test]
    fn a_measured_cylinder_gets_its_radius_and_a_calliper_around_it() {
        let r = 0.05;
        let (s, a, b) = scene(r, true);
        let e = evidence(&[&s], &[a], &[b], Some(&[1.5 * r]), &EvidenceParams::default(), 2).unwrap();
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
        let e = evidence(&[&s], &[a], &[b], Some(&[0.05]), &EvidenceParams::default(), 0).unwrap();
        assert_eq!(e.n_contra[0], 0);
        let (s, a, b) = scene(0.01, false);
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0, "1 cm in an echoes-only store rests on the prior");
    }

    #[test]
    fn wood_thinner_than_the_beam_is_prior_only_even_with_misses() {
        let (s, a, b) = scene(0.002, true);
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0, "2 mm under a 9 mm footprint rests on the prior");
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams { min_footprints: 0.0, ..Default::default() }, 0).unwrap();
        assert!(e.tau[0] > 0.0, "only the footprint rule makes it so");
    }

    #[test]
    fn a_neighbours_echoes_are_not_counted() {
        let (s, a, b) = scene(0.05, true);
        let alone = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        // an empty axis 12 cm beside it, nearer the scanner's line of sight to some of its echoes
        let (a2, b2) = ([0.12, 5.0, -1.0], [0.12, 5.0, 1.0]);
        let e = evidence(&[&s], &[a, a2], &[b, b2], None, &EvidenceParams::default(), 0).unwrap();
        assert_eq!(e.n_hit[1], 0, "the empty axis takes none of its neighbour's echoes");
        assert_eq!((e.n_hit[0], e.mu[0]), (alone.n_hit[0], alone.mu[0]));
        assert!(e.hi[1] < 0.02, "pulses pass through the empty axis's place: {}", e.hi[1]);
    }

    #[test]
    fn a_file_indexes_as_its_shots_and_misses_need_a_pulse_without_echoes() {
        let path = std::env::temp_dir().join(format!("sylva_evidence_{}.parquet", std::process::id()));
        let shots = shots(0.05, true);
        write_shots(&shots, &path, &ShotsWriteOptions { row_group_size: 1000, ..Default::default() }).unwrap();
        let (from_file, from_shots) = (ScanIndex::from_file(&path, true, 3).unwrap(), ScanIndex::new(&shots, true).unwrap());
        assert_eq!((from_file.dir, from_file.echo_start, from_file.range), (from_shots.dir, from_shots.echo_start, from_shots.range));
        let echoes_only = shots.subset(&(0..shots.n_shots()).map(|p| shots.echo_count[p] > 0).collect::<Vec<_>>());
        write_shots(&echoes_only, &path, &ShotsWriteOptions::default()).unwrap();
        assert!(ScanIndex::from_file(&path, true, 0).is_err());
        assert!(ScanIndex::from_file(&path, false, 0).is_ok());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn calliper_matches_a_clean_separation() {
        let (lo, hi) = calliper(&[0.01, 0.02, 0.03], &[0.05, 0.06], 0);
        assert!((lo - 0.04).abs() < 1e-12 && (hi - 0.04).abs() < 1e-12);
    }
}
