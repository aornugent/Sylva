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
//!   is too wide to measure). So `tau == 0 && n_hit > 0` is "prior only",
//!   and `reason` (see [`reason`]) says which.
//! - `lo`, `hi`: the beam calliper. A pulse that hit the cylinder came within
//!   `d` of the axis, so `r > d - w/2`; one that passed came within `d`
//!   without hitting, so `r < d + w/2` (`w`: the footprint).
//! - counts: `n_hit` (the hits `mu` is the median of), `n_cross` (pulses in
//!   the window that hit or passed) and `n_contra` (passes inside a claimed
//!   radius). A hit is an echo the cylinder won beside the segment, whether
//!   or not its pulse's line comes closest to the axis within the segment;
//!   `n_cross` counts only pulses that do, so `n_hit` can exceed `n_cross`
//!   (pulses near the ends, or along the axis). A tolerant audit of a fat
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
//!    calliper, else `2 r + 2 cm`, within `[window_min, window_cap]` (the
//!    constants are [`EvidenceParams`]').
//!
//! The window comes from the rays, so a claim can't move it and leaves beyond
//! the calliper can't inflate it. Pass every cylinder near the ones of
//! interest (a whole tile), so a neighbour's echoes go to the neighbour.
//! Speed comes from looking only where wood is: each scan is indexed once by
//! pulse direction ([`ScanIndex`]), each cylinder visits only the pulses near
//! its axis, and cylinders run in parallel.
//!
//! [`pairs`] returns the same enumeration before any reduction: every pulse
//! near each cylinder with all its echoes, for a per-pulse likelihood fitted
//! outside.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::scan_index::{closest, pool, ScanIndex};
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

    /// The beam's diameter (m) at `range` (m).
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
    /// hit, so it must exceed the largest radius to be measured. It is also
    /// how far a cylinder's neighbours reach into its evidence.
    pub window_cap: f64,
    /// The narrowest window (m).
    pub window_min: f64,
    /// Where passes bound the calliper, the next window is
    /// `max(window_hi_factor * hi, hi + window_hi_margin)`...
    pub window_hi_factor: f64,
    pub window_hi_margin: f64,
    /// ...else `window_r_factor * r + window_r_margin`.
    pub window_r_factor: f64,
    pub window_r_margin: f64,
    /// Rounds of assignment and windowing.
    pub rounds: usize,
    /// Cylinders shorter than this (m) have no axis to measure from: they
    /// take no echoes and get no evidence.
    pub min_length: f64,
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
            window_hi_factor: 1.5,
            window_hi_margin: 0.02,
            window_r_factor: 2.0,
            window_r_margin: 0.02,
            rounds: 3,
            min_length: 0.001,
        }
    }
}

impl EvidenceParams {
    /// Bumped whenever the evidence changes for the same settings, so a
    /// cache keyed on the settings and this can trust what it holds.
    pub const VERSION: u32 = 2;

    /// An error unless the settings make sense together.
    pub fn check(&self) -> Result<()> {
        let b = &self.beam;
        let non_negative = [
            b.divergence, b.exit, b.range_noise, self.floor_log, self.footprint_sd, self.spread_sd, self.echoes_only_min_radius, self.min_footprints,
            self.window_hi_factor, self.window_hi_margin, self.window_r_factor, self.window_r_margin, self.min_length,
        ];
        if !non_negative.iter().all(|&x| x.is_finite() && x >= 0.0) {
            return Err(Error::invalid("evidence params: the beam, floor, sds, radii, window rule and lengths must be finite and not negative"));
        }
        if !(self.window_min > 0.0 && self.window_min <= self.window_cap && self.window_cap.is_finite()) {
            return Err(Error::invalid("evidence params: need 0 < window_min <= window_cap"));
        }
        if self.rounds == 0 {
            return Err(Error::invalid("evidence params: need at least one round"));
        }
        Ok(())
    }

    /// The next window of a cylinder whose hits sit at `r` and whose calliper
    /// tops out at `hi`.
    fn window(&self, r: f64, hi: f64) -> f64 {
        let w = if hi.is_finite() && hi > 0.0 { (self.window_hi_factor * hi).max(hi + self.window_hi_margin) } else { self.window_r_factor * r + self.window_r_margin };
        w.clamp(self.window_min, self.window_cap)
    }
}

/// Why a cylinder's `tau` is what it is: [`evidence`]'s `reason`.
pub mod reason {
    /// The rays measure the radius: `tau > 0`.
    pub const MEASURED: u8 = 0;
    /// No hits (or no length): no evidence.
    pub const NO_HITS: u8 = 1;
    /// Prior only: pulses passed closer to the axis than its echoes sit
    /// (`hi < r`: the echoes are clutter around something thinner).
    pub const CLUTTER: u8 = 2;
    /// Prior only: its echoes sit within `min_footprints` beam footprints of
    /// the axis (the beam is too wide to measure it).
    pub const FOOTPRINT: u8 = 3;
    /// Prior only: thinner than `echoes_only_min_radius` and seen only by
    /// scans that kept no misses.
    pub const ECHOES_ONLY: u8 = 4;
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
    /// Why `tau` is what it is: one of [`reason`].
    pub reason: Vec<u8>,
    /// Penetration at the measured radius: pulses aimed at its core that
    /// reached it (`n_core`), and those of them that went through or stopped
    /// deep inside it (`n_pen`). Wood stops nearly all; foliage lets many by.
    pub n_core: Vec<u32>,
    pub n_pen: Vec<u32>,
    /// The median reflectance (dB) and deviation of its hits, NaN without
    /// hits or where the store has no such attribute.
    pub reflectance: Vec<f64>,
    pub deviation: Vec<f64>,
}

/// A pulse whose line passes within the window cap of a cylinder's axis,
/// closest at `d` (`t` along the ray) inside the segment and ahead of the
/// scanner. Its echoes near the axis are the cylinder's candidates
/// `cand..cand + n_cand`.
struct Pair {
    cand: usize,
    n_cand: u32,
    d: f32,
    t: f32,
    /// The pulse went on past `t`: its last echo is beyond, or it missed.
    beyond: bool,
    /// Its last echo is the last of its candidates: where it stopped, beside the segment.
    last_cand: bool,
}

/// An echo, the `k`-th of a scan's pulse, at `q` from a cylinder's axis and
/// projecting onto the segment: a candidate hit.
struct Cand {
    scan: u32,
    pulse: u32,
    k: u32,
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
    reason: u8,
    n_core: u32,
    n_pen: u32,
    reflectance: f64,
    deviation: f64,
}

/// The evidence the `scans` give about cylinders `a[i] -> b[i]`; `r_claim`,
/// if given, the claimed radii to audit. Runs on `threads` threads (0: one
/// per core).
pub fn evidence(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_claim: Option<&[f64]>, p: &EvidenceParams, threads: usize) -> Result<Evidence> {
    let n = a.len();
    if b.len() != n || r_claim.is_some_and(|r| r.len() != n) {
        return Err(Error::invalid("evidence: a, b and r_claim must have one row per cylinder"));
    }
    if n >= u32::MAX as usize || scans.len() >= u32::MAX as usize {
        return Err(Error::invalid("evidence: too many cylinders or scans"));
    }
    p.check()?;
    pool(threads)?.install(|| Ok(evidence_on(scans, a, b, r_claim, p)))
}

fn evidence_on(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_claim: Option<&[f64]>, p: &EvidenceParams) -> Evidence {
    let n = a.len();
    let cap = p.window_cap;
    let len: Vec<f64> = (0..n).map(|i| norm(&sub(&b[i], &a[i]))).collect();
    let u: Vec<Point> = (0..n).map(|i| sub(&b[i], &a[i]).map(|x| x / len[i].max(1e-9))).collect();
    let misses: Vec<bool> = scans.iter().map(|s| s.misses()).collect();

    // 1. once: the (pulse, cylinder) pairs within the cap, and the echoes of their pulses near the axis
    let found: Vec<(Vec<Pair>, Vec<Cand>)> = (0..n).into_par_iter().map(|c| {
        let (mut pairs, mut cands) = (Vec::new(), Vec::new());
        if !(len[c] >= p.min_length) {
            return (pairs, cands); // no axis: every echo near its ends would be a candidate
        }
        for (si, s) in scans.iter().enumerate() {
            let w0 = sub(&s.origin(), &a[c]);
            let dd = dot(&w0, &u[c]);
            s.pulses_near(&a[c], &b[c], cap, |pulse, d, echoes| {
                let (t_ray, s_ax, _) = closest(&d, &u[c], &w0, dd);
                let dist = norm(&sub(&add(&w0, &scale(&d, t_ray)), &scale(&u[c], s_ax.clamp(0.0, len[c]))));
                if !(dist < cap && t_ray > 0.0) {
                    return;
                }
                let first = cands.len();
                for (k, &range) in echoes.iter().enumerate() {
                    let w = add(&w0, &scale(&d, range as f64)); // the echo, from a
                    let t = dot(&w, &u[c]);
                    let q = norm(&sub(&w, &scale(&u[c], t)));
                    if t >= 0.0 && t <= len[c] && q < cap {
                        cands.push(Cand { scan: si as u32, pulse, k: k as u32, q: q as f32, range });
                    }
                }
                let last = echoes.last().map_or(f64::INFINITY, |&r| r as f64); // infinite for a miss
                // a pulse that stopped short without an echo near the axis is neither a hit nor a pass
                if s_ax > 0.0 && s_ax < len[c] && (last > t_ray || cands.len() > first) {
                    let last_cand = cands.len() > first && cands[cands.len() - 1].k as usize + 1 == echoes.len();
                    pairs.push(Pair { cand: first, n_cand: (cands.len() - first) as u32, d: dist as f32, t: t_ray as f32, beyond: last > t_ray, last_cand });
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
    // each echo's candidates together, cylinders ascending: (scan, pulse, echo, cylinder, candidate)
    let mut by_echo: Vec<(u32, u32, u32, u32, usize)> = (0..n).into_par_iter().flat_map_iter(|c| {
        let cands = &cands;
        (cand_at[c]..cand_at[c + 1]).map(move |i| (cands[i].scan, cands[i].pulse, cands[i].k, c as u32, i))
    }).collect();
    by_echo.par_sort_unstable();

    // 2. rounds over the tables: assign the echoes, classify the pairs, and set each window from its calliper
    let (mut r_hat, mut win) = (vec![0.0; n], vec![cap; n]);
    let mut out = Vec::new();
    for round in 0..p.rounds {
        // an echo goes to the cylinder whose surface it lies nearest (ties: the lower index), among those
        // whose window holds it
        let winners: Vec<usize> = by_echo.par_chunk_by(|x, y| (x.0, x.1, x.2) == (y.0, y.1, y.2)).filter_map(|group| {
            let mut best: Option<(f64, usize)> = None;
            for &(_, _, _, c, i) in group {
                let q = cands[i].q as f64;
                let resid = (q - r_hat[c as usize]).abs();
                if q < win[c as usize] && best.is_none_or(|(br, _)| resid < br) {
                    best = Some((resid, i));
                }
            }
            best.map(|(_, i)| i)
        }).collect();
        let mut won = vec![false; cands.len()];
        for i in winners {
            won[i] = true;
        }
        out = (0..n).into_par_iter().map(|c| {
            let at = cand_at[c]..cand_at[c + 1];
            cylinder(&pairs[c], &cands[at.clone()], &won[at], r_hat[c], win[c], r_claim.map(|r| r[c]), scans, &misses, p)
        }).collect::<Vec<Round>>();
        if round + 1 == p.rounds {
            break;
        }
        for c in 0..n {
            (r_hat[c], win[c]) = (out[c].med, p.window(out[c].med, out[c].hi));
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
        reason: out.iter().map(|o| o.reason).collect(),
        n_core: out.iter().map(|o| o.n_core).collect(),
        n_pen: out.iter().map(|o| o.n_pen).collect(),
        reflectance: out.iter().map(|o| o.reflectance).collect(),
        deviation: out.iter().map(|o| o.deviation).collect(),
    }
}

/// One cylinder in a round: its hits (per pulse, the echo it won nearest its
/// surface `r_prev`), its pairs within the window `win` as hits, passes or
/// neither, and what they say. `misses[s]`: whether scan `s` kept misses.
#[allow(clippy::too_many_arguments)]
fn cylinder(pairs: &[Pair], cands: &[Cand], won: &[bool], r_prev: f64, win: f64, r_claim: Option<f64>, scans: &[&ScanIndex], misses: &[bool], p: &EvidenceParams) -> Round {
    let resid = |k: usize| (cands[k].q as f64 - r_prev).abs();
    let mut hits: Vec<&Cand> = Vec::new();
    for run in (0..cands.len()).collect::<Vec<_>>().chunk_by(|&x, &y| (cands[x].scan, cands[x].pulse) == (cands[y].scan, cands[y].pulse)) {
        // a pulse's candidates are consecutive; the first of equals wins
        let best = run.iter().copied().filter(|&k| won[k]).fold(None, |b: Option<usize>, k| if b.is_none_or(|b| resid(k) < resid(b)) { Some(k) } else { b });
        hits.extend(best.map(|k| &cands[k]));
    }
    let (mut hit_d, mut pass_d, mut n_contra) = (Vec::new(), Vec::new(), 0u32);
    for pr in pairs.iter().filter(|pr| (pr.d as f64) < win) {
        let mine = pr.cand..pr.cand + pr.n_cand as usize;
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
    let mut out = Round {
        med: 0.0, mu: f64::NAN, tau: 0.0, lo, hi, n_hit: hits.len() as u32, n_cross, n_contra, reason: reason::NO_HITS,
        n_core: 0, n_pen: 0, reflectance: f64::NAN, deviation: f64::NAN,
    };
    if hits.is_empty() {
        return out;
    }
    let m = median(&hits.iter().map(|e| e.q as f64).collect::<Vec<_>>());
    (out.n_core, out.n_pen) = penetration(pairs, cands, won, m.max(1e-4), win, p);
    let character: Vec<(f64, f64)> = hits.iter().map(|e| scans[e.scan as usize].character(e.pulse, e.k)).collect();
    let finite_median = |v: Vec<f64>| if v.is_empty() { f64::NAN } else { median(&v) };
    out.reflectance = finite_median(character.iter().map(|c| c.0).filter(|x| x.is_finite()).collect());
    out.deviation = finite_median(character.iter().map(|c| c.1).filter(|x| x.is_finite()).collect());
    let w = median(&hits.iter().map(|e| p.beam.footprint(e.range as f64)).collect::<Vec<_>>());
    let spread = 1.4826 * median(&hits.iter().map(|e| (e.q as f64 - m).abs()).collect::<Vec<_>>());
    let mut per_scan = vec![0.0; misses.len()];
    let mut seen_with_misses = false;
    for e in &hits {
        let sigma = (p.beam.footprint(e.range as f64) / 2.0).hypot(p.beam.range_noise);
        per_scan[e.scan as usize] += (m / sigma).powi(2);
        seen_with_misses |= misses[e.scan as usize];
    }
    out.med = m;
    out.mu = m.max(1e-4).ln();
    let r = out.mu.exp();
    // prior only (module docs): passes inside the echoes, wood the beam is too wide to measure, or thin wood
    // seen only by echoes-only scans
    out.reason = if hi < r {
        reason::CLUTTER
    } else if m < p.min_footprints * w || m <= 0.0 {
        reason::FOOTPRINT
    } else if !seen_with_misses && r < p.echoes_only_min_radius {
        reason::ECHOES_ONLY
    } else {
        let tau: f64 = per_scan.iter().filter(|&&t| t > 0.0).map(|t| 1.0 / (1.0 / t + p.floor_log * p.floor_log)).sum();
        out.tau = 1.0 / (1.0 / tau + (p.footprint_sd * w / 2.0 / m).powi(2) + (p.spread_sd * spread / m).powi(2));
        reason::MEASURED
    };
    out
}

/// Penetration at radius `r`: `(n_core, n_pen)`. A pulse aimed at the core
/// (`d < r - band`, `band = w/2 + max(1 cm, 0.3 r)` for axis offsets) that
/// reached `r` either went through (it went on past the axis and its last
/// echo isn't within `r + band` of it) or stopped at its last echo `q` from
/// the axis: by the surface (`q` within `band` of `r`), or deep inside
/// (`q < r - band`). Through or deep is penetration, which a solid of radius
/// `r` forbids and foliage allows. A pulse that stopped short of `r + band`
/// (or before the segment), or whose last echo went to a neighbour or a fork
/// (not won here, but within the window), says nothing.
fn penetration(pairs: &[Pair], cands: &[Cand], won: &[bool], r: f64, win: f64, p: &EvidenceParams) -> (u32, u32) {
    let (mut n_core, mut n_pen) = (0, 0);
    for pr in pairs {
        let band = p.beam.footprint(pr.t as f64) / 2.0 + (0.3 * r).max(0.01);
        let last = pr.last_cand.then(|| pr.cand + pr.n_cand as usize - 1);
        if pr.d as f64 >= r - band || last.is_some_and(|k| !won[k] && (cands[k].q as f64) < win) {
            continue;
        }
        let q = last.map_or(f64::INFINITY, |k| cands[k].q as f64);
        if (pr.beyond && q > r + band) || q < r - band {
            (n_core, n_pen) = (n_core + 1, n_pen + 1);
        } else if q <= r + band {
            n_core += 1;
        }
    }
    (n_core, n_pen)
}

/// The radius as a distance threshold separating pulses that hit (`hit_d`,
/// closer to the axis) from pulses that passed (`pass_d`, farther): the range
/// `(lo, hi)` of thresholds within `slack` misclassified pulses of the best.
/// A few clutter hits or leaks only widen the range.
fn calliper(hit_d: &[f64], pass_d: &[f64], slack: usize) -> (f64, f64) {
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
    use crate::qsm::scan_index::tests::{index, scene};
    use crate::shots::Shots;


    #[test]
    fn a_measured_cylinder_gets_its_radius_and_a_calliper_around_it() {
        let r = 0.05;
        let (s, a, b) = scene(r, true);
        let e = evidence(&[&s], &[a], &[b], Some(&[1.5 * r]), &EvidenceParams::default(), 2).unwrap();
        // echoes lie on the near surface, so their median axis distance is about r
        assert!((e.mu[0].exp() / r - 1.0).abs() < 0.05, "mu {}", e.mu[0].exp());
        assert!(e.tau[0] > 0.0 && e.reason[0] == reason::MEASURED);
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
        assert!(!s.misses());
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0, "1 cm in an echoes-only store rests on the prior");
        assert_eq!(e.reason[0], reason::ECHOES_ONLY);
    }

    #[test]
    fn wood_thinner_than_the_beam_is_prior_only_even_with_misses() {
        let (s, a, b) = scene(0.002, true);
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0, "2 mm under a 9 mm footprint rests on the prior");
        assert_eq!(e.reason[0], reason::FOOTPRINT);
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
        assert_eq!(e.reason[1], reason::NO_HITS);
    }

    #[test]
    fn echoes_around_an_axis_that_pulses_pass_through_are_clutter() {
        // a few leaves 4.5-5 cm from a vertical axis 5 m away; the pulses between them go through to a wall
        let mut shots = Shots::default();
        for i in 0..80 {
            for j in 0..200 {
                let (az, zen) = (-0.02 + i as f64 * 0.0005, std::f64::consts::FRAC_PI_2 - 0.05 + j as f64 * 0.0005);
                let d = [zen.sin() * az.sin(), zen.sin() * az.cos(), zen.cos()];
                let off = 5.0 * az.sin(); // the closest approach to the axis, in plan
                shots.echo_start.push(shots.echo_range.len());
                shots.echo_range.push(if (0.045..0.05).contains(&off.abs()) { 5.0 * az.cos() / zen.sin() } else { 8.0 / d[1] });
                shots.echo_count.push(1);
                shots.origin.push([0.0; 3]);
                shots.direction.push(d);
            }
        }
        let s = index(&shots, 10_000);
        let (a, b) = ([0.0, 5.0, -0.2], [0.0, 5.0, 0.2]);
        // in the first round's window the leaves are its hits, but the calliper is below them
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams { rounds: 1, ..Default::default() }, 0).unwrap();
        assert!(e.n_hit[0] > 0 && e.tau[0] == 0.0 && e.reason[0] == reason::CLUTTER, "{e:?}");
        // and the window then closes on the calliper, leaving them out
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_hit[0] == 0 && e.reason[0] == reason::NO_HITS, "{e:?}");
    }

    /// A porous ball of leaves (radius 12 cm, reflectance -20 dB, deviation
    /// 15) around a point 5 m from the scanner, before a wall (-10 dB,
    /// deviation 3): a pulse into the ball stops on a leaf at a random depth
    /// half the time, else goes on to the wall. Echoes only, as an export.
    fn leaf_ball() -> Shots {
        let (c, rad) = ([0.0, 5.0, 0.0], 0.12);
        let mut seed = 7u64;
        let mut uniform = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut shots = Shots::default();
        let (mut refl, mut dev) = (Vec::new(), Vec::new());
        for i in 0..121 {
            for j in 0..201 {
                let (az, zen) = (-0.03 + i as f64 * 0.0005, std::f64::consts::FRAC_PI_2 - 0.05 + j as f64 * 0.0005);
                let d = [zen.sin() * az.sin(), zen.sin() * az.cos(), zen.cos()];
                let tc = dot(&d, &c);
                let gap2 = dot(&c, &c) - tc * tc;
                shots.echo_start.push(shots.echo_range.len());
                if gap2 < rad * rad && uniform() < 0.5 {
                    let half = (rad * rad - gap2).sqrt();
                    shots.echo_range.push(tc - half + 2.0 * half * uniform());
                    (refl.push(-20.0f32), dev.push(15u16));
                } else {
                    shots.echo_range.push(8.0 / d[1]);
                    (refl.push(-10.0), dev.push(3));
                }
                shots.echo_count.push(1);
                shots.origin.push([0.0; 3]);
                shots.direction.push(d);
            }
        }
        shots.echo_attrs.insert("reflectance".into(), crate::pointcloud::Attr::F32(refl));
        shots.echo_attrs.insert("deviation".into(), crate::pointcloud::Attr::U16(dev));
        shots
    }

    #[test]
    fn foliage_lets_pulses_through_its_core_and_wood_does_not() {
        let (s, a, b) = scene(0.05, true);
        let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        assert!(e.n_core[0] > 100 && e.n_pen[0] * 50 <= e.n_core[0], "solid wood: {} of {} core pulses through", e.n_pen[0], e.n_core[0]);
        assert!(e.reflectance[0].is_nan() && e.deviation[0].is_nan(), "a store without echo character");
        let s = index(&leaf_ball(), 10_000);
        let (a, b) = ([0.0, 5.0, -0.1], [0.0, 5.0, 0.1]);
        for rounds in [1, 3] {
            let e = evidence(&[&s], &[a], &[b], None, &EvidenceParams { rounds, ..Default::default() }, 0).unwrap();
            assert!(rounds > 1 || e.n_hit[0] > 0, "the first round's window holds the leaves");
            if e.n_hit[0] > 0 {
                assert!(e.n_core[0] > 20 && e.n_pen[0] * 2 > e.n_core[0], "leaves: {} of {} core pulses through ({e:?})", e.n_pen[0], e.n_core[0]);
                assert_eq!((e.reflectance[0], e.deviation[0]), (-20.0, 15.0));
            }
        }
    }

    #[test]
    fn a_segment_overhead_finds_the_pulses_towards_its_middle() {
        // seen from the scanner, a horizontal segment overhead is nearest the zenith in its middle (11 degrees
        // from it), not at its ends (46 degrees)
        let mut shots = Shots::default();
        for k in 0..21 {
            let target = [(k as f64 - 10.0) * 0.01, 1.0, 5.0];
            shots.direction.push(target.map(|x| x / norm(&target)));
            shots.origin.push([0.0; 3]);
            shots.echo_start.push(k);
            shots.echo_count.push(1);
            shots.echo_range.push(norm(&target) - 0.05);
        }
        let s = index(&shots, 100);
        let mut seen = 0;
        s.pulses_near(&[-5.0, 1.0, 5.0], &[5.0, 1.0, 5.0], 0.1, |_, _, _| seen += 1);
        assert_eq!(seen, 21);
        let e = evidence(&[&s], &[[-5.0, 1.0, 5.0]], &[[5.0, 1.0, 5.0]], None, &EvidenceParams::default(), 1).unwrap();
        assert_eq!(e.n_hit[0], 21);
    }

    #[test]
    fn a_cylinder_without_length_takes_no_echoes() {
        let (s, a, b) = scene(0.05, true);
        let alone = evidence(&[&s], &[a], &[b], None, &EvidenceParams::default(), 0).unwrap();
        // a point on the near surface, where the echoes are
        let dot = [0.0, 4.95, 0.0];
        let e = evidence(&[&s], &[dot, a], &[dot, b], Some(&[0.05, 0.05]), &EvidenceParams::default(), 0).unwrap();
        assert!(e.mu[0].is_nan() && e.tau[0] == 0.0 && (e.n_hit[0], e.n_cross[0], e.n_contra[0]) == (0, 0, 0) && e.reason[0] == reason::NO_HITS);
        assert_eq!((e.n_hit[1], e.mu[1], e.hi[1]), (alone.n_hit[0], alone.mu[0], alone.hi[0]));
    }

    #[test]
    fn settings_that_dont_make_sense_are_an_error() {
        let (s, a, b) = scene(0.05, true);
        let defaults = EvidenceParams::default();
        for p in [EvidenceParams { window_min: 0.3, ..defaults }, EvidenceParams { rounds: 0, ..defaults }, EvidenceParams { floor_log: -1.0, ..defaults }] {
            assert!(evidence(&[&s], &[a], &[b], None, &p, 0).is_err());
        }
    }

    #[test]
    fn calliper_matches_a_clean_separation() {
        let (lo, hi) = calliper(&[0.01, 0.02, 0.03], &[0.05, 0.06], 0);
        assert!((lo - 0.04).abs() < 1e-12 && (hi - 0.04).abs() < 1e-12);
    }
}
