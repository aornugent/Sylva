// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A routed piece (RUNS or HULL) fitted by the exact likelihood (3dtrees
//! research 93 §18, after review): seeds from its runs, each stem on its own
//! leaning axis, and its stems' likelihood messages.
//!
//! - **Seeds.** Each scan's profile is cut into slices along the piece
//!   ([`SLICE`] long): within a slice a leaning stem barely moves, so its runs
//!   separate. Each run is then measured as the support path measures a
//!   piece's own ([`super::measure`] about the run, which takes its lean from
//!   the run's own hits and its front's depth against the axial position,
//!   then [`super::solve_row`]): a circle with its centre, lean and radius
//!   from one scan. The circles are scored by the woody stops of every scan on
//!   their front arcs, each stop moved back along the circle's own lean
//!   (within [`SEED_TOL`]); a circle's radius may grow by [`SEED_GROW`] with
//!   its front point held (a run is cut short where a stem beside it hides an
//!   edge); hulls (a quarter of the pulses aimed at the core go through) are
//!   dropped; and at most [`EXACT_K`] that don't overlap are taken greedily,
//!   strongest first. Never the piece's circle: on a hull its lean is the line
//!   between two stems (a fork's stems side by side in one piece).
//! - **The fit.** The piece's pulses are taken once, in a tube about the
//!   claim's axis that holds every seed, and each seed's view of them is the
//!   pair measured again from its own leaning axis (the chord through the tube
//!   kept: the same foliage for every stem). A stop that a context cylinder
//!   (the tree's measured and bounded pieces) explains better than every seed
//!   is the context's. Every pulse is fitted where the piece has at most
//!   [`FIT_PAIRS`], and one in `ceil(n / FIT_PAIRS)` where it has more (a
//!   hash of the pulse's direction): a scan's
//!   pulses are about five times more correlated than the likelihood assumes
//!   (research 93 §17's [`EXIST_SCALE`]), and fitting half of every piece made k
//!   flip with the half (§18-§19). Then `k = 1, 2, ...` stems, each from
//!   its seed on its own axis ([`fit_stems`]: each pulse seen from the stem
//!   it meets first, each stem its own foliage, Newton to its stopping rule
//!   within [`EXACT_ITERS`]), while the same pulses' BIC rises and the fits
//!   converge (an unconverged `k > 1` isn't weighed; an unconverged one stem
//!   is kept, to be refused).
//! - **The message.** Each stem's likelihood curvature at its fit (directions
//!   of positive curvature carry nothing; foliage integrated out), its
//!   information scaled to the whole set's and discounted by the
//!   pulses' correlation ([`JointParams::exist_scale`]), with
//!   [`JointParams::model_sd`] added to its variances as a measured piece's
//!   are; the lean is its seed's, with the seed's variance.
//! - **Refusal,** decided here alone ([`exact_piece`]): no
//!   seed (no fallback: no seed, no fit), no pulse left, Newton unconverged, a
//!   stem not finite, one stem narrower than the piece's detection bound, or
//!   one stem saying nothing of its radius ([`UNINFORMED_SD`]). Of several
//!   stems, one that isn't finite or says nothing of its radius refutes its
//!   rung of the ladder: the fit of one stem fewer stands.
//! - TODO: passes through a context cylinder aren't removed (context is the
//!   stops it explains); nor are other trees' cylinders context.

use nalgebra::{Matrix2, Matrix3, Matrix4, Matrix5, SymmetricEigen, Vector5};

use super::{fate, marginalise, measure, median, p_wood, profile_runs, solve_row, spacing, unseen, weights, Bound, Circle, Fate, Group, Side, Sight, Stats, CAP_SD, DEPTH_TOL, EXIST_SCALE, HULL_BEHIND, HULL_SHARE, MODEL_SD, RUN_HITS, SEEN_MIN};
use crate::error::Result;
use crate::qsm::scan_index::ScanIndex;
use crate::qsm::joint::{frames, joint_loglik, joint_table, joint_wood_share, window, Echoes, JointPairs, JointParams, Owner, Pulse, EDGE, TAU};
use crate::qsm::joint_fit::{joint_fit, Fit, FitOptions, Frame, Rows};
use crate::transform::{cross, dot, norm, sub};
use crate::Point;

/// The slices a scan's profile is cut into for its runs (m along the piece).
const SLICE: f64 = 0.1;
/// How far a stop may lie from a seed's circle and vote for it (m), and how much wider than its run a seed may grow
/// (m: a run is cut short where a stem beside it hides an edge, but a much wider circle spans several stems' fronts).
const SEED_TOL: f64 = 0.006;
const SEED_GROW: f64 = 0.015;
/// How far a run's depth may move along the piece with a stem's lean, for the run's measurement (m).
const LEAN_DEPTH: f64 = 0.05;
/// The most runs of one scan measured as seeds, strongest first.
const AIMS: usize = 6;
/// The most stems a routed piece is fitted with.
pub(super) const EXACT_K: usize = 4;
/// The exact fit's Newton iterations at most per number of stems, and its stopping tolerance (see
/// [`FitOptions::tol`]: on a piece of a few thousand pairs, a decrement of about a log unit, a fraction of the
/// tempered message's sd).
const EXACT_ITERS: usize = 20;
const EXACT_TOL: f64 = 3e-4;
/// Every pulse is fitted on a piece of at most [`FIT_PAIRS`] pairs, and one in `ceil(n / FIT_PAIRS)` (by [`kept`]) on
/// a larger one, so that at most about that many are (a trunk near a scanner is the most oversampled).
const THIN: u32 = 1;
const FIT_PAIRS: usize = 8000;
/// A stem whose message's sd of log R exceeds this says nothing of its radius (a factor of e either way): the fit is
/// refused, as no measurement of the stem (a stem on a piece of twigs had 4e4).
const UNINFORMED_SD: f64 = 1.0;

/// How far from a shadowed stem's fitted surface its stops may lie and bound it from below (m, plus `0.3 R`): the range
/// noise's scale (a reach of `DEPTH_TOL` takes in a face beside a thin stem).
const ARC_TOL: f64 = 0.005;

/// A seed: a stem's centre at the piece's middle and lean (along the piece's `e1, e2`, the lean per m), its radius,
/// and its lean's variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Seed {
    pub c: [f64; 2],
    pub t: [f64; 2],
    pub r: f64,
    pub t_var: [f64; 2],
}

/// A point of the cross-section from a pulse's impact `b` and depth along the group's `n` and `v`; None where the
/// beam lies nearly along its own normal.
fn section(n: [f64; 2], v: [f64; 2], b: f64, depth: f64) -> Option<[f64; 2]> {
    let det = n[0] * v[1] - n[1] * v[0];
    (det.abs() >= 0.2).then(|| [(b * v[1] - depth * n[1]) / det, (n[0] * depth - v[0] * b) / det])
}

/// Each scan's runs measured as circles with their lean (see the module documentation), with the beam that saw each.
fn proposals(gs: &[Group], r_claim: f64, p: &JointParams) -> Vec<(Seed, [f64; 2])> {
    let mut out = Vec::new();
    for g in gs {
        let n = g.recs.len();
        if n == 0 {
            continue;
        }
        let fates: Vec<Fate> = g.recs.iter().map(|r| fate(r, g, None)).collect();
        let wt = weights(g, &fates, r_claim, p);
        let w_med = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
        let (s0, s1) = g.recs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| (a.min(r.s), b.max(r.s)));
        let ns = ((s1 - s0) / SLICE).ceil().max(1.0) as usize;
        let slice = |s: f64| (((s - s0) / (s1 - s0).max(1e-9) * ns as f64) as usize).min(ns - 1);
        // per slice, its runs: (middle, half width, depths, hits)
        let mut aims: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
        for j in 0..ns {
            let idx: Vec<usize> = (0..n).filter(|&i| slice(g.recs[i].s) == j).collect();
            let bp: Vec<f64> = idx.iter().map(|&i| g.recs[i].b).collect();
            let f: Vec<Fate> = idx.iter().map(|&i| fates[i]).collect();
            let w: Vec<f64> = idx.iter().map(|&i| wt[i]).collect();
            let d: Vec<f64> = idx.iter().map(|&i| g.recs[i].depth).collect();
            let (_, bw) = spacing(&bp, &f);
            for q in profile_runs(&bp, &f, &w, &d, bw, w_med) {
                if q.hits >= RUN_HITS / ns as f64 && q.bins >= 2 {
                    aims.push((0.5 * (q.lo + q.hi), 0.5 * (q.hi - q.lo), q.dlo, q.dhi, q.hits));
                }
            }
        }
        // the strongest first, at most AIMS; one nearly on a run already taken is that run
        aims.sort_by(|a, b| b.4.total_cmp(&a.4));
        let mut taken: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
        for a in aims {
            if taken.len() < AIMS && !taken.iter().any(|t| (t.0 - a.0).abs() < 0.5 * t.1.min(a.1)) {
                taken.push(a);
            }
        }
        for (mid, hw, dlo, dhi, _) in taken {
            let ra = hw + 0.01;
            let depths = if dlo.is_finite() && dhi.is_finite() { (dlo - DEPTH_TOL - LEAN_DEPTH, dhi + DEPTH_TOL + LEAN_DEPTH) } else { (f64::NEG_INFINITY, f64::INFINITY) };
            let o = measure(g, &fates, &wt, mid, ra, depths, p);
            let Some(c) = solve_row(&Stats::of(&o.meas), ra) else { continue };
            if c.x[4] >= 0.005 && c.x[4] <= 0.3 {
                out.push((Seed { c: [c.x[0], c.x[1]], t: [c.x[2], c.x[3]], r: c.x[4], t_var: [c.cov[(2, 2)], c.cov[(3, 3)]] }, g.v));
            }
        }
    }
    out
}

/// A routed piece's seeds (see the module documentation), strongest first; none where no run makes one (the piece
/// is then refused).
pub(super) fn seeds(gs: &[Group], r_claim: f64, p: &JointParams) -> Vec<Seed> {
    let props = proposals(gs, r_claim, p);
    if props.is_empty() {
        return Vec::new();
    }
    // the woody stops in the cross-section (at their own axial place), with their axial place and beam
    let mut pts: Vec<([f64; 2], f64, [f64; 2])> = Vec::new();
    for g in gs {
        for r in g.recs.iter().filter(|r| r.is_stop() && p_wood(r, r_claim, p) >= 0.5) {
            if let Some(q) = section(g.n, g.v, r.b, r.depth) {
                pts.push((q, r.s, g.v));
            }
        }
    }
    // on a circle `(c, r)`: a stop moved back along the lean `t`, within SEED_TOL of it, the centre behind it
    let on = |c: [f64; 2], r: f64, q: &([f64; 2], [f64; 2])| {
        let (dx, dy) = (c[0] - q.0[0], c[1] - q.0[1]);
        (dx.hypot(dy) - r).abs() < SEED_TOL && dx * q.1[0] + dy * q.1[1] > 0.0
    };
    // a seed is solid where at most HULL_SHARE of the pulses aimed at its core that reached its front went on behind
    // it (a hull around several stems is hit on its arcs, but its core's pulses go through to the stems behind)
    let solid = |s: &Seed| {
        let (mut core, mut through) = (0usize, 0usize);
        for g in gs {
            let (n, v) = (g.n, g.v);
            let (nt, vt) = (n[0] * s.t[0] + n[1] * s.t[1], v[0] * s.t[0] + v[1] * s.t[1]);
            let (cn, cv) = (n[0] * s.c[0] + n[1] * s.c[1], v[0] * s.c[0] + v[1] * s.c[1]);
            for r in g.recs.iter().filter(|r| !r.blocked) {
                let miss = r.b - r.s * nt - cn;
                if miss.abs() >= 0.5 * s.r {
                    continue;
                }
                let front = cv - (s.r * s.r - miss * miss).sqrt();
                let d = r.depth - r.s * vt;
                if d < front - DEPTH_TOL {
                    continue; // stopped before it got there
                }
                core += 1;
                through += usize::from(!d.is_finite() || d > front + HULL_BEHIND + 0.1 * s.r);
            }
        }
        core == 0 || (through as f64) <= HULL_SHARE * core as f64
    };
    // each proposal's radius, its front point held, as the stops on its arc say (of the solid circles; its own
    // where none does better); with the stops near it, moved back along its lean (an index into pts each)
    let cands: Vec<(Seed, Vec<(usize, ([f64; 2], [f64; 2]))>)> = props.iter().filter_map(|&(s, v)| {
        let front = [s.c[0] - s.r * v[0], s.c[1] - s.r * v[1]];
        let reach = 2.0 * (s.r + SEED_GROW) + SEED_TOL;
        let local: Vec<(usize, ([f64; 2], [f64; 2]))> = pts.iter().enumerate().filter_map(|(i, q)| {
            let m = [q.0[0] - q.1 * s.t[0], q.0[1] - q.1 * s.t[1]];
            ((m[0] - front[0]).hypot(m[1] - front[1]) <= reach).then_some((i, (m, q.2)))
        }).collect();
        let votes = |c: [f64; 2], r: f64| local.iter().filter(|q| on(c, r, &q.1)).count();
        let own = votes(s.c, s.r);
        let best = (0..59).map(|i| 0.005 + 0.0025 * i as f64).filter(|&r| r >= s.r - SEED_GROW && r <= s.r + SEED_GROW).map(|r| ([front[0] + r * v[0], front[1] + r * v[1]], r))
            .map(|(c, r)| (votes(c, r), c, r)).filter(|q| q.0 > own).max_by_key(|q| q.0);
        let grown = best.map(|(_, c, r)| Seed { c, r, ..s }).filter(|g| solid(g));
        grown.or_else(|| solid(&s).then_some(s)).map(|g| (g, local))
    }).collect();
    let mut left = vec![true; pts.len()];
    let mut out: Vec<Seed> = Vec::new();
    let mut first = 0usize;
    while out.len() < EXACT_K {
        let best = cands.iter().filter(|(c, _)| out.iter().all(|o| (o.c[0] - c.c[0]).hypot(o.c[1] - c.c[1]) >= o.r + c.r - SEED_TOL))
            .map(|(c, local)| (local.iter().filter(|q| left[q.0] && on(c.c, c.r, &q.1)).count(), c, local)).max_by_key(|q| q.0);
        let Some((votes, c, local)) = best else { break };
        if (votes as f64) < RUN_HITS || 4 * votes < first {
            break;
        }
        first = first.max(votes);
        for q in local {
            if (c.c[0] - q.1 .0[0]).hypot(c.c[1] - q.1 .0[1]) < c.r + SEED_TOL {
                left[q.0] = false; // its stops, and any inside it, are explained
            }
        }
        out.push(*c);
    }
    out
}

/// A pair measured again from another axis (from `a`, frame `(u, e1, e2)`): its impact, range of closest approach,
/// angle and their rates, from its ray (`o`, `d`); its chord through the tube and its fate there are kept.
fn remeasure(pu: &Pulse, (o, d): (Point, Point), a: &Point, (u, e1, e2): (Point, Point, Point)) -> Pulse {
    let cr = cross(&u, &d);
    let sin = dot(&cr, &cr).sqrt().max(1e-9);
    let n = cr.map(|x| x / sin);
    let w0 = sub(&o, a);
    let bb = dot(&d, &u);
    let ph: Point = std::array::from_fn(|k| (d[k] - bb * u[k]) / sin);
    Pulse {
        b: dot(&w0, &n),
        t: (bb * dot(&w0, &u) - dot(&d, &w0)) / (sin * sin),
        sin,
        jb: [dot(&n, &e1), dot(&n, &e2)],
        jt: [dot(&ph, &e1) / sin, dot(&ph, &e2) / sin],
        ..*pu
    }
}

/// A routed piece's accepted exact fit (see [`exact_piece`]): its stems' summaries as measurements of the piece's
/// `[c1, c2, t1, t2, R]` (in the order fitted), and its stops classified (`node`: the stem).
pub(super) struct Exact {
    pub stems: Vec<Circle>,
    pub echoes: Echoes,
    /// Each stem's own bounds where the sides its scans see leave its radius undetermined ([`look`]; its message
    /// then says nothing along what they leave free), None where they don't.
    pub shadow: Vec<Option<Vec<Bound>>>,
    /// The tube its pulses were taken in (m, about the claim's axis), and each stem's stops the context owned (of
    /// those in the seeds' windows, by the stem they meet first).
    pub tube: f64,
    pub n_ctx: Vec<u32>,
}

/// The least stops on a stem's front arc, from one scan, to bound its radius from below by their span (a span of
/// fewer is the noise's).
const ARC_MIN: usize = 10;
/// The share of those stops left out at each end of their span (5 %: a stray echo off the arc doesn't widen it).
const ARC_TRIM: usize = 20;
/// The width of an edge's neighbourhood, a share of R (a half), within these (m: wider than the footprint's edge,
/// narrower than a neighbour's gap).
const BAND_SHARE: f64 = 0.5;
const BAND: (f64, f64) = (0.01, 0.03);

/// How each scan sees a fitted stem (`r`, on the seed's `axis` (its ends) with frame `fr`, moved by `d`), from every
/// pulse of the table `t` and its echoes' depths about the stem (research 93 §20): a side (the neighbourhood of an
/// edge, `R` to `R + W` beyond the axis across the beam, `W` [`BAND_SHARE`] of the radius within [`BAND`], or the
/// front's middle) is seen where at least [`SEEN_MIN`] pulses reached it (an echo at or behind it, or none recorded), and weighs the pulses that did per m across the beam; one whose pulses all echoed
/// further in front (a crossing branch's shadow over a fork's stem) is not seen. The directions are the
/// piece's (`claim`: its axis' start, frame, and the table's tube about it). And the stem's bounds: below, half the
/// span of the stops on its front arc (less [`ARC_TRIM`] at each end, from at least [`ARC_MIN`]) less the
/// footprint's widening, the tightest of the scans'; the capacity floor at the widest footprint.
///
/// None where an edge it doesn't see lies past the table's tube: whether it is unseen isn't judged (the fit's
/// TUBE_EDGE says why).
#[allow(clippy::too_many_arguments)]
fn look(t: &JointPairs, scans: &[&ScanIndex], axis: &(Point, Point), fr: (Point, Point, Point), d: [f64; 2], r: f64, claim: (&Point, Frame, f64), like: &JointParams) -> Option<(Vec<Sight>, Vec<Bound>)> {
    let (a0, (_, e1, e2, _), tube) = claim;
    let us = fr.0;
    let band = (BAND_SHARE * r).clamp(BAND.0, BAND.1);
    let tol = DEPTH_TOL + 0.3 * r;
    // the stem's centre across the claim's axis at its two ends
    let off: Point = std::array::from_fn(|i| d[0] * fr.1[i] + d[1] * fr.2[i]);
    let centres = [axis.0, axis.1].map(|p| {
        let q: Point = std::array::from_fn(|i| p[i] + off[i] - a0[i]);
        [dot(&q, &e1), dot(&q, &e2)]
    });
    // each side's weight is its pulses per m across the beam (the front's band is R wide, an edge's `band`)
    let width = [band, band, r];
    let ns = scans.len();
    let (mut nsum, mut vsum) = (vec![[0.0f64; 2]; ns], vec![[0.0f64; 2]; ns]);
    let mut reached = vec![[0usize; 3]; ns];
    let mut arc: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ns];
    for (i, pu) in t.pulses.iter().enumerate() {
        let s = t.scan[i] as usize;
        let sc = scans[s];
        let ray = (sc.origin(), sc.direction(t.pulse[i]));
        let v = shifted(&remeasure(pu, ray, &axis.0, fr), d);
        let (bb, at) = (v.b, |range: f64| (range - v.t) * v.sin);
        if pu.stop {
            let dx = at(pu.x);
            // on its surface within the range noise's scale (the reach `tol` would take in a face beside a thin stem)
            if dx <= 0.0 && (bb.hypot(dx) - r).abs() < ARC_TOL + 0.3 * r {
                arc[s].push((bb, like.footprint(v.t)));
            }
        }
        let side = if bb.abs() < 0.5 * r { 2 } else if bb > r && bb <= r + band { 0 } else if bb < -r && bb >= -r - band { 1 } else { continue };
        let front = if side == 2 { -(r * r - bb * bb).max(0.0).sqrt() } else { 0.0 };
        let echoes = sc.ranges(t.pulse[i]);
        if echoes.is_empty() || echoes.iter().any(|&e| at(e as f64) >= front - tol) {
            reached[s][side] += 1;
        }
        let dir = ray.1;
        let du = dot(&dir, &us);
        let sin = (1.0 - du * du).max(1e-12).sqrt();
        let n = cross(&us, &dir).map(|x| x / sin);
        let vv: Point = std::array::from_fn(|k| (dir[k] - du * us[k]) / sin);
        nsum[s] = [nsum[s][0] + dot(&n, &e1), nsum[s][1] + dot(&n, &e2)];
        vsum[s] = [vsum[s][0] + dot(&vv, &e1), vsum[s][1] + dot(&vv, &e2)];
    }
    let unit = |q: [f64; 2]| {
        let l = q[0].hypot(q[1]).max(1e-12);
        [q[0] / l, q[1] / l]
    };
    let sights: Vec<Sight> = (0..ns).filter(|&s| nsum[s] != [0.0; 2]).map(|s| Sight { n: unit(nsum[s]), v: unit(vsum[s]), w: std::array::from_fn(|k| if reached[s][k] >= SEEN_MIN { reached[s][k] as f64 / width[k] } else { 0.0 }) }).collect();
    // an unseen edge whose neighbourhood reaches past the tube at either end: the table holds no pulse to say
    for s in &sights {
        for (k, sign) in [(0, 1.0), (1, -1.0)] {
            if s.w[k] == 0.0 && centres.iter().any(|c| (c[0] * s.n[0] + c[1] * s.n[1] + sign * (r + band)).abs() > tube) {
                return None;
            }
        }
    }
    let mut bounds = Vec::new();
    let mut lo = f64::NEG_INFINITY;
    let mut ws: Vec<f64> = Vec::new();
    for mut a in arc.into_iter().filter(|a| a.len() >= ARC_MIN) {
        a.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        let m = a.len();
        let w = median(&mut a.iter().map(|q| q.1).collect::<Vec<_>>());
        ws.push(w);
        let half = 0.5 * (a[((ARC_TRIM - 1) * m) / ARC_TRIM - 1].0 - a[m / ARC_TRIM].0) - like.app(w);
        if half > 0.5 * TAU * w && half.ln() > lo {
            lo = half.ln();
            bounds.retain(|b: &Bound| b.side != Side::Lower);
            bounds.push(Bound { eta: lo, sd: EDGE * w / std::f64::consts::SQRT_2 / half, side: Side::Lower });
        }
    }
    if let Some(w) = ws.into_iter().reduce(f64::max) {
        bounds.push(Bound { eta: (0.5 * TAU * w).ln(), sd: CAP_SD, side: Side::Floor });
    }
    Some((sights, bounds))
}

/// What became of a routed piece's exact fit: accepted, or refused.
pub(super) enum Fitted {
    Accepted(Exact),
    Refused,
}

/// Whether every stem's message says something of its radius (sd of log R at most [`UNINFORMED_SD`]).
fn informed(stems: &[Circle]) -> bool {
    stems.iter().all(|c| c.cov[(4, 4)].sqrt() / c.x[4] <= UNINFORMED_SD)
}

/// The pulses fitted on a piece of `n_all` pairs: one in `thin`, at least `min` (see [`THIN`]), so that at most
/// about [`FIT_PAIRS`] are.
fn thinning(n_all: usize, min: u32) -> u32 {
    (n_all.div_ceil(FIT_PAIRS) as u32).max(min)
}

/// Whether the pulse of direction `d` is one of the `1 / thin` fitted: by a hash of its direction (splitmix64), so
/// the choice is the pulse's own, not its place in a file (a scan cut to a tree's pulses fits the same ones).
fn kept(d: Point, thin: u32) -> bool {
    let mut z = d.iter().fold(0u64, |h, x| h.rotate_left(21) ^ x.to_bits()).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) % thin as u64 == 0
}

/// The seeded exact fit of a routed piece (claimed from `a0`, frame `frame`; see the module documentation) from
/// its `seeds`, with `ctx` (each cylinder's ends and radius) owning the stops it explains better; refused where there is no seed or no pulse, where Newton didn't converge, where a stem isn't finite or says
/// nothing of its radius, or where one stem is narrower than the piece's detection bound `lower` (log R).
#[allow(clippy::too_many_arguments)]
pub(super) fn exact_piece(scans: &[&ScanIndex], a0: &Point, frame: Frame, seeds: &[Seed], ctx: &[(Point, Point, f64)], lower: f64, like: &JointParams) -> Result<Fitted> {
    exact_thinned(scans, a0, frame, seeds, ctx, lower, like, THIN)
}

/// [`exact_piece`], fitting one pulse in at least `min_thin`.
#[allow(clippy::too_many_arguments)]
fn exact_thinned(scans: &[&ScanIndex], a0: &Point, frame: Frame, seeds: &[Seed], ctx: &[(Point, Point, f64)], lower: f64, like: &JointParams, min_thin: u32) -> Result<Fitted> {
    if seeds.is_empty() {
        return Ok(Fitted::Refused);
    }
    let (u, e1, e2, len) = frame;
    let mid: Point = std::array::from_fn(|i| a0[i] + 0.5 * len * u[i]);
    let fb: Point = std::array::from_fn(|i| a0[i] + len * u[i]);
    // each seed's axis: through its centre at the middle, along its lean, across the piece's length
    let axes: Vec<(Point, Point)> = seeds.iter().map(|s| {
        let up: Point = std::array::from_fn(|i| u[i] + s.t[0] * e1[i] + s.t[1] * e2[i]);
        let cm: Point = std::array::from_fn(|i| mid[i] + s.c[0] * e1[i] + s.c[1] * e2[i]);
        (std::array::from_fn(|i| cm[i] - 0.5 * len * up[i]), std::array::from_fn(|i| cm[i] + 0.5 * len * up[i]))
    }).collect();
    let sfr: Vec<(Point, Point, Point)> = axes.iter().map(|(a, b)| {
        let f = frames(&[*a], &[*b])[0];
        (f.0, f.1, f.2)
    }).collect();
    let tube = seeds.iter().map(|s| s.c[0].hypot(s.c[1]) + 0.5 * len * s.t[0].hypot(s.t[1]) + window(s.r)).fold(0.0, f64::max);
    let t = joint_table(scans, &[*a0], &[fb], &[tube], 1)?;
    let ray = |i: usize| -> (Point, Point) {
        let sc = scans[t.scan[i] as usize];
        (sc.origin(), sc.direction(t.pulse[i]))
    };
    // how far a point lies from a segment's axis
    let seg = |p: &Point, a: &Point, b: &Point| -> f64 {
        let (ab, ap) = (sub(b, a), sub(p, a));
        let s = (dot(&ap, &ab) / dot(&ab, &ab).max(1e-18)).clamp(0.0, 1.0);
        norm(&sub(&ap, &ab.map(|v| v * s)))
    };
    // context: of the cylinders near the piece, the stops each explains better than every seed does (each cylinder
    // tested only where the stop lies within its reach)
    let keep: Vec<bool> = {
        let near: Vec<(&(Point, Point, f64), Point, f64)> = ctx.iter().filter_map(|c| {
            let (a, b, r) = c;
            let m: Point = std::array::from_fn(|i| 0.5 * (a[i] + b[i]));
            let half = 0.5 * norm(&sub(b, a));
            let reach = half + r + DEPTH_TOL + 0.3 * r;
            (norm(&sub(&m, &mid)) < 0.5 * len + tube + reach).then_some((c, m, reach))
        }).collect();
        (0..t.pulses.len()).map(|i| {
            let pu = &t.pulses[i];
            if !pu.stop || near.is_empty() {
                return true;
            }
            let (o, d) = ray(i);
            let pt: Point = std::array::from_fn(|j| o[j] + pu.x * d[j]);
            let mut mine = None;
            !near.iter().any(|((a, b, r), m, reach)| {
                if norm(&sub(&pt, m)) > *reach {
                    return false;
                }
                let dc = (seg(&pt, a, b) - r).abs();
                if dc >= DEPTH_TOL + 0.3 * r {
                    return false;
                }
                let own = *mine.get_or_insert_with(|| seeds.iter().zip(&axes).map(|(s, (sa, sb))| (seg(&pt, sa, sb) - s.r).abs()).fold(f64::INFINITY, f64::min));
                dc < own
            })
        }).collect()
    };
    // the pairs within some seed's own tube (the table's holds them all, and is wider where the seeds are apart)
    let in_window = |i: usize| seeds.iter().enumerate().any(|(j, s)| remeasure(&t.pulses[i], ray(i), &axes[j].0, sfr[j]).b.abs() < window(s.r));
    let idx: Vec<usize> = (0..t.pulses.len()).filter(|&i| keep[i] && in_window(i)).collect();
    let n_all = idx.len();
    let thin = thinning(n_all, min_thin);
    let fit_idx: Vec<usize> = idx.iter().copied().filter(|&i| kept(ray(i).1, thin)).collect();
    let n = fit_idx.len();
    if n == 0 {
        return Ok(Fitted::Refused);
    }
    // each seed's view of the fitted pairs, from its own axis
    let views: Vec<Vec<Pulse>> = seeds.iter().enumerate().map(|(j, _)| fit_idx.iter().map(|&i| remeasure(&t.pulses[i], ray(i), &axes[j].0, sfr[j])).collect()).collect();
    let range: Vec<f64> = fit_idx.iter().map(|&i| t.pulses[i].t).collect();
    let scan: Vec<u32> = fit_idx.iter().map(|&i| t.scan[i]).collect();
    let opts = FitOptions { iters: EXACT_ITERS, tol: EXACT_TOL, threads: 1 };
    // k stems while the same pulses' BIC rises: the k - 1 fitted (each its seed's axis moved by its offset, with its
    // radius) and the next seed
    type State = Vec<(usize, [f64; 2], f64)>;
    // the ladder: each k's fit, while BIC rose
    let mut ladder: Vec<(f64, State, StemFit)> = Vec::new();
    for k in 1..=seeds.len() {
        let st: State = ladder.last().map_or(Vec::new(), |b| b.1.clone()).into_iter().chain([(k - 1, [0.0, 0.0], seeds[k - 1].r)]).collect();
        let moved: Vec<(Point, Point)> = st.iter().map(|&(j, d, _)| {
            let (_, f1, f2) = sfr[j];
            let off: Point = std::array::from_fn(|i| d[0] * f1[i] + d[1] * f2[i]);
            (std::array::from_fn(|i| axes[j].0[i] + off[i]), std::array::from_fn(|i| axes[j].1[i] + off[i]))
        }).collect();
        let radii: Vec<f64> = st.iter().map(|s| s.2).collect();
        let see = |q: usize, i: usize, d: [f64; 2]| shifted(&views[st[q].0][i], [st[q].1[0] + d[0], st[q].1[1] + d[1]]);
        let sf = fit_stems(&range, &see, &moved, &radii, &scan, scans.len(), like, &opts)?;
        // an unconverged fit of k > 1 stems isn't weighed; one stem's is kept, to be refused as a message, and
        // no more are tried
        if k > 1 && !sf.fit.converged {
            break;
        }
        let more = sf.fit.converged;
        let bic = sf.ll - 2.0 * (k as f64 - 1.0) * (n as f64).ln();
        if ladder.last().is_some_and(|b| bic <= b.0) {
            break;
        }
        let fitted: State = st.iter().zip(&sf.stems).map(|(&(j, d, _), &(o, r))| (j, [d[0] + o[0], d[1] + o[1]], r)).collect();
        ladder.push((bic, fitted, sf));
        if !more {
            break;
        }
    }
    // each stem's message, as a measurement of the piece's [c1, c2, t1, t2, R]
    let messages = |st: &State, sf: &StemFit| -> Vec<Circle> { st.iter().enumerate().map(|(q, &(j, d, r))| {
        let s = &seeds[j];
        let (_, f1, f2) = sfr[j];
        let up: Point = std::array::from_fn(|i| u[i] + s.t[0] * e1[i] + s.t[1] * e2[i]);
        // the stem's centre where it crosses the piece's middle, in the piece's frame, and its rate in the offsets
        let cm: Point = std::array::from_fn(|i| mid[i] + s.c[0] * e1[i] + s.c[1] * e2[i] + d[0] * f1[i] + d[1] * f2[i]);
        let back = |w: Point| -> Point {
            let sa = dot(&w, &u);
            std::array::from_fn(|i| w[i] - sa * up[i])
        };
        let p0 = back(sub(&cm, &mid));
        let c = [dot(&p0, &e1), dot(&p0, &e2)];
        let (g1, g2) = (back(f1), back(f2));
        let m2 = Matrix2::new(dot(&g1, &e1), dot(&g2, &e1), dot(&g1, &e2), dot(&g2, &e2));
        // the likelihood's information: its curvature where negative, nothing where it isn't; foliage integrated out
        let e = SymmetricEigen::new(-Matrix4::from(sf.fit.hess[q]));
        let h = e.eigenvectors * Matrix4::from_diagonal(&e.eigenvalues.map(|l| if l.is_finite() { l.max(0.0) } else { 0.0 })) * e.eigenvectors.transpose();
        let mut i3 = Matrix3::from_fn(|a, b| h[(a, b)]);
        if h[(3, 3)] > 1e-12 {
            i3 -= nalgebra::Vector3::from_fn(|a, _| h[(a, 3)]) * nalgebra::Vector3::from_fn(|a, _| h[(a, 3)]).transpose() / h[(3, 3)];
        }
        // [c1, c2, R] = A [d1, d2, eta] near the fit; the thinned pulses' information made the whole set's, then
        // discounted by the pulses' correlation, and the model's floor added to its variances
        let mut am = Matrix3::zeros();
        am.fixed_view_mut::<2, 2>(0, 0).copy_from(&m2);
        am[(2, 2)] = r;
        let ai = am.try_inverse().unwrap_or_else(Matrix3::identity);
        let im = ai.transpose() * i3 * ai * (n_all as f64 / n as f64 * EXIST_SCALE);
        // (in its eigenbasis, so a direction it doesn't inform stays uninformed: variance 1 / l + model_sd^2)
        let s2 = MODEL_SD * MODEL_SD;
        let e3 = SymmetricEigen::new(0.5 * (im + im.transpose()));
        let l3 = e3.eigenvalues.map(|l| if l.is_finite() { l.max(0.0) } else { 0.0 });
        let im = e3.eigenvectors * Matrix3::from_diagonal(&l3.map(|l| l / (1.0 + l * s2))) * e3.eigenvectors.transpose();
        let cm3 = e3.eigenvectors * Matrix3::from_diagonal(&l3.map(|l| if l > 1e-6 { 1.0 / l + s2 } else { 1e6 })) * e3.eigenvectors.transpose();
        let (mut info, mut cov) = (Matrix5::zeros(), Matrix5::zeros());
        for (a, &pa) in [0usize, 1, 4].iter().enumerate() {
            for (b, &pb) in [0usize, 1, 4].iter().enumerate() {
                info[(pa, pb)] = im[(a, b)];
                cov[(pa, pb)] = cm3[(a, b)];
            }
        }
        // the lean: its seed's
        for i in 0..2 {
            info[(2 + i, 2 + i)] = 1.0 / s.t_var[i].max(1e-12);
            cov[(2 + i, 2 + i)] = s.t_var[i];
        }
        let y = Vector5::new(c[0], c[1], s.t[0], s.t[1], r);
        let h5 = info * y;
        Circle { st: Stats { info, h: h5, yy: y.dot(&h5), n: 5 }, x: y, cov, chi2: f64::NAN }
    }).collect() };
    // the most stems whose messages are all finite and all say something of their radius: a stem that doesn't
    // refutes its rung, and the fit of one stem fewer stands (only one stem's fit can be unconverged: it is refused;
    // where no rung stands, the piece is refused for the last one's fault)
    let mut chosen = None;
    while let Some((_, st, sf)) = ladder.pop() {
        if !sf.fit.converged {
            return Ok(Fitted::Refused);
        }
        let stems = messages(&st, &sf);
        if !stems.iter().all(|c| c.x.iter().chain(c.st.info.iter()).chain(c.cov.iter()).all(|v| v.is_finite())) {
            continue;
        }
        // one stem as narrow as the run it was seen in can't be (several may span it together)
        if st.len() == 1 && stems[0].x[4].ln() < lower {
            return Ok(Fitted::Refused);
        }
        if informed(&stems) {
            chosen = Some((st, sf, stems));
            break;
        }
    }
    let Some((st, sf, mut stems)) = chosen else { return Ok(Fitted::Refused) };
    let k = st.len();
    // each stem as its scans see it: where the sides they see leave its radius undetermined, its message says nothing
    // along what they leave free, and its own bounds go with it (research 93 §20)
    let mut shadow = vec![None; k];
    for (q, &(j, d, r)) in st.iter().enumerate() {
        let Some((sights, b)) = look(&t, scans, &axes[j], sfr[j], d, r, (a0, frame, tube), like) else { continue };
        let free = unseen(&sights);
        if !free.is_empty() {
            stems[q] = marginalise(&stems[q], &free);
            shadow[q] = Some(b);
        }
    }
    // each stop in the seeds' windows as the stems see it, and the stem it meets first
    let radii: Vec<f64> = st.iter().map(|s| s.2).collect();
    let meet = |i: usize| -> (Pulse, usize) {
        let v: Vec<Pulse> = st.iter().map(|&(j, d, _)| shifted(&remeasure(&t.pulses[i], ray(i), &axes[j].0, sfr[j]), d)).collect();
        let q = first_met(&v, &radii, like.footprint(t.pulses[i].t));
        (v[q], q)
    };
    // the context's stops there, each to the stem it meets first
    let mut n_ctx = vec![0u32; k];
    for i in (0..t.pulses.len()).filter(|&i| !keep[i] && in_window(i)) {
        n_ctx[meet(i).1] += 1;
    }
    // the classification: every kept stop (fitted or not) by the softmax of its wood and leaf terms at the fit of
    // the stem it meets first
    let echoes = {
        let stops: Vec<usize> = idx.iter().copied().filter(|&i| t.pulses[i].stop).collect();
        let (seen, owners): (Vec<Pulse>, Vec<Owner>) = stops.iter().map(|&i| {
            let (v, q) = meet(i);
            (v, Owner { row: q as u32, nu: u32::MAX })
        }).unzip();
        let share = joint_wood_share(&seen, &owners, &sf.theta, &[], like, 1)?;
        let mut out = Echoes::default();
        for (q, &i) in stops.iter().enumerate() {
            let s = t.scan[i];
            let j = owners[q].row;
            out.push(s, scans[s as usize].file_row(t.pulse[i]), t.echo[i], j, share[q], surface_dist(&seen[q], sf.theta[j as usize]));
        }
        out
    };
    Ok(Fitted::Accepted(Exact { stems, echoes, shadow, tube, n_ctx }))
}

/// How far a stop lies from the circle of state `s` (offsets and log R in its row's frame) in the cross-section (m).
pub(crate) fn surface_dist(pu: &Pulse, s: [f64; 4]) -> f64 {
    let (y, _) = cross_section(pu);
    ((y[0] - s[0]).hypot(y[1] - s[1]) - s[2].exp()).abs()
}

/// A pair measured from its row's axis moved by `d` across it: the same pulse
/// and chord (so the same foliage exposure), seen from a parallel axis.
pub(crate) fn shifted(pu: &Pulse, d: [f64; 2]) -> Pulse {
    Pulse { b: pu.b - (d[0] * pu.jb[0] + d[1] * pu.jb[1]), t: pu.t + (d[0] * pu.jt[0] + d[1] * pu.jt[1]), ..*pu }
}

/// Which of several circles a pair meets first, from its view from each
/// circle's centre (`views`, each measured from that circle's axis), their
/// radii and its footprint `w`: of those its footprint touches, the one whose
/// front it reaches first; if none, the one whose silhouette it passes
/// nearest. Solid stems don't overlap, so a pulse's fate is the first one's.
pub(crate) fn first_met(views: &[Pulse], radii: &[f64], w: f64) -> usize {
    let mut best = (false, f64::INFINITY, 0);
    for (i, (s, &r)) in views.iter().zip(radii).enumerate() {
        let touch = s.b.abs() < r + 0.5 * w;
        let key = if touch { s.t - (r * r - s.b * s.b).max(0.0).sqrt() / s.sin.max(1e-9) } else { s.b.abs() - r };
        if (touch && !best.0) || (touch == best.0 && key < best.1) {
            best = (touch, key, i);
        }
    }
    best.2
}

/// Stems fitted to one set of pairs (see [`fit_stems`]).
pub(crate) struct StemFit {
    /// The pairs' log likelihood at the fit, each seen from the fitted stem it meets first.
    pub ll: f64,
    pub fit: Fit,
    /// Each fitted stem's offset from its own axis (along that axis's frame) and radius, and the state its pairs
    /// were scored at (seen from it).
    pub stems: Vec<([f64; 2], f64)>,
    pub theta: Vec<[f64; 4]>,
}

/// Stems fitted to one set of pairs, the pair `i` at range `range[i]` (for its footprint) and from `scan[i]`:
/// `see(j, i, d)` is the pair measured from stem `j`'s axis (`axes[j]`, prior radius `radii[j]`) moved by `d`
/// across it, along that axis's frame. Each pair is seen from the stem it meets first ([`first_met`]), each stem a
/// row of its own with its own foliage, fitted by [`joint_fit`] with `opts` from the axes; then each pair is seen
/// again from the fitted stem it meets first, and scored there: the seeded exact fit of a routed piece, each stem
/// on its own leaning axis (3dtrees research 93 §18).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_stems(range: &[f64], see: &(dyn Fn(usize, usize, [f64; 2]) -> Pulse + Sync), axes: &[(Point, Point)], radii: &[f64], scan: &[u32], n_scans: usize, p: &JointParams, opts: &FitOptions) -> Result<StemFit> {
    let m = axes.len();
    let (sa, sb): (Vec<Point>, Vec<Point>) = axes.iter().copied().unzip();
    let sw: Vec<f64> = radii.iter().map(|&r| window(r)).collect();
    let parent = vec![-1i64; m];
    // each pair seen from the stem (moved by its offset, at its radius) it meets first
    let assign = |st: &[([f64; 2], f64)]| -> (Vec<Pulse>, Vec<Owner>) {
        let rs: Vec<f64> = st.iter().map(|s| s.1).collect();
        range.iter().enumerate().map(|(i, &t)| {
            let views: Vec<Pulse> = (0..m).map(|j| see(j, i, st[j].0)).collect();
            let k = first_met(&views, &rs, p.footprint(t));
            (views[k], Owner { row: k as u32, nu: u32::MAX })
        }).unzip()
    };
    let (sp, own) = assign(&radii.iter().map(|&r| ([0.0, 0.0], r)).collect::<Vec<_>>());
    let row: Vec<u32> = own.iter().map(|o| o.row).collect();
    let rows = Rows { a: &sa, b: &sb, r_prior: radii, r_max: &sw, parent: &parent };
    let f = joint_fit(&sp, &row, scan, &rows, n_scans, p, opts)?;
    let fitted: Vec<([f64; 2], f64)> = f.theta.iter().map(|t| ([t[0], t[1]], t[2].exp())).collect();
    let (sp, own) = assign(&fitted);
    let theta: Vec<[f64; 4]> = f.theta.iter().map(|t| [0.0, 0.0, t[2], t[3]]).collect();
    let ll = joint_loglik(&sp, &own, &theta, &[], p, opts.threads)?.iter().sum();
    Ok(StemFit { ll, fit: f, stems: fitted, theta })
}


/// A stop's place in its row's cross-section (frame `e1, e2`, from the axis)
/// and the unit direction its pulse travelled across the axis: its closest
/// approach lies `b` along the normal (`jb`), the stop `(x - t) sin` further
/// along that direction (`jt sin`).
fn cross_section(pu: &Pulse) -> ([f64; 2], [f64; 2]) {
    let v = [pu.jt[0] * pu.sin, pu.jt[1] * pu.sin];
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-12);
    let v = [v[0] / l, v[1] / l];
    let s = (pu.x - pu.t) * pu.sin;
    ([pu.b * pu.jb[0] + s * v[0], pu.b * pu.jb[1] + s * v[1]], v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qsm::joint::frames;

    #[test]
    fn a_pair_measured_from_a_moved_axis_is_the_pair_shifted() {
        // a pulse across a vertical axis; the same axis moved 2 cm along e1: remeasuring is the shift, exactly
        let a: Point = [0.0, 0.0, 0.0];
        let b: Point = [0.0, 0.0, 1.0];
        let f = frames(&[a], &[b])[0];
        let (o, d) = ([5.0, -0.03, 0.4], [-1.0, 0.0, 0.05]);
        let dn = norm(&d);
        let d: Point = d.map(|x| x / dn);
        let base = Pulse { b: 0.0, t: 0.0, sin: 1.0, jb: [0.0; 2], jt: [0.0; 2], c0: 4.8, c1: 5.2, x: 4.97, stop: true, early: false, q: 1.0 };
        let pu = remeasure(&base, (o, d), &a, (f.0, f.1, f.2));
        let moved: Point = std::array::from_fn(|i| a[i] + 0.02 * f.1[i]);
        let pm = remeasure(&base, (o, d), &moved, (f.0, f.1, f.2));
        let sh = shifted(&pu, [0.02, 0.0]);
        assert!((pm.b - sh.b).abs() < 1e-12 && (pm.t - sh.t).abs() < 1e-12, "{pm:?} {sh:?}");
        // the chord and the fate are the table's
        assert!(pm.c0 == base.c0 && pm.x == base.x && pm.stop);
    }

    #[test]
    fn a_side_behind_a_screen_leaves_the_radius_undetermined() {
        // the fan's 5 cm stem seen whole, and with a screen 1 m in front of its left edge (stopping the pulses that
        // would cross the stem's plane 2-15 cm left of its axis): seen whole, both edges and the front decide R; behind
        // the screen, one edge and the front leave it trading against the centre
        let like = JointParams::default();
        let (whole, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let (screened, _, _) = crate::qsm::scan_index::tests::scene_screened(0.05, (-0.15, -0.02));
        let (a0, b0) = ([a[0], a[1], -0.15], [b[0], b[1], 0.15]);
        let frame = frames(&[a0], &[b0])[0];
        let fr = (frame.0, frame.1, frame.2);
        let look_in = |scan: &ScanIndex, tube: f64| {
            let t = joint_table(&[scan], &[a0], &[b0], &[tube], 1).unwrap();
            look(&t, &[scan], &(a0, b0), fr, [0.0, 0.0], 0.05, (&a0, frame, tube), &like)
        };
        for (scan, shadowed) in [(&whole, false), (&screened, true)] {
            let (sights, bounds) = look_in(scan, 0.2).unwrap();
            assert_eq!(sights.len(), 1);
            let seen = sights[0].w.iter().filter(|&&w| w > 0.0).count();
            assert_eq!((unseen(&sights).is_empty(), seen), (!shadowed, if shadowed { 2 } else { 3 }), "{sights:?}");
            if shadowed {
                // the stops on its seen arc bound it from below, no tighter than it is; and the capacity floor
                assert!(bounds.iter().filter(|q| q.side == Side::Lower).all(|q| q.eta.exp() <= 0.053) && bounds.iter().any(|q| q.side == Side::Floor), "{bounds:?}");
            }
        }
        // a table whose tube ends inside the stem's edges' neighbourhoods: the screened edge, unseen there, isn't
        // judged; seen whole, both edges still are
        assert!(look_in(&screened, 0.06).is_none());
        assert!(look_in(&whole, 0.06).is_some_and(|(s, _)| unseen(&s).is_empty()));
    }

    #[test]
    fn pieces_over_the_budget_are_thinned_evenly() {
        assert_eq!((thinning(10, THIN), thinning(FIT_PAIRS, THIN), thinning(FIT_PAIRS + 1, THIN), thinning(10 * FIT_PAIRS, THIN)), (1, 1, 2, 10));
        assert_eq!(thinning(100, 2), 2);
        // a fan of directions: each thinning keeps its share, whichever pulses the file holds
        let dirs: Vec<Point> = (0..200_000).map(|i| {
            let (z, a) = (1.0 + 1e-4 * (i / 500) as f64, 1e-4 * (i % 500) as f64);
            [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
        }).collect();
        for thin in [2u32, 3, 7] {
            let share = dirs.iter().filter(|&&d| kept(d, thin)).count() as f64 / dirs.len() as f64;
            assert!((share * thin as f64 - 1.0).abs() < 0.03, "thin {thin}: {share}");
        }
    }

    #[test]
    fn a_stem_that_says_nothing_of_its_radius_is_uninformed() {
        let stem = |r: f64, sd_r: f64| Circle { st: Stats { info: Matrix5::zeros(), h: Vector5::zeros(), yy: 0.0, n: 5 }, x: Vector5::new(0.0, 0.0, 0.0, 0.0, r), cov: Matrix5::from_diagonal_element(sd_r * sd_r), chi2: f64::NAN };
        assert!(informed(&[stem(0.05, 0.005), stem(0.03, 0.006)]));
        // one stem of sd 2 R among informed ones: the rung is refuted
        assert!(!informed(&[stem(0.05, 0.005), stem(0.02, 0.04), stem(0.03, 0.006)]));
    }

    #[test]
    fn the_message_is_scaled_to_all_the_pairs() {
        // the fan's 5 cm stem, fitted from every pulse and from every other one: the thinned fit's message, scaled
        // by n_all / n, carries about the information of the whole set's
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let (a0, b0) = ([a[0], a[1], -0.15], [b[0], b[1], 0.15]);
        let fr = frames(&[a0], &[b0])[0];
        let seed = [Seed { c: [0.0, 0.0], t: [0.0, 0.0], r: 0.05, t_var: [0.01, 0.01] }];
        let fit = |thin: u32| match exact_thinned(&[&scan], &a0, fr, &seed, &[], f64::NEG_INFINITY, &JointParams::default(), thin).unwrap() {
            Fitted::Accepted(e) => e,
            Fitted::Refused => panic!("refused"),
        };
        let (all, half) = (fit(1), fit(2));
        let (ia, ih) = (all.stems[0].st.info, half.stems[0].st.info);
        for i in [0usize, 1, 4] {
            let ratio = ih[(i, i)] / ia[(i, i)];
            assert!(ratio > 0.7 && ratio < 1.4, "info[{i}]: thinned {} against all {}", ih[(i, i)], ia[(i, i)]);
        }
    }

    #[test]
    fn a_shifted_pair_is_the_pair_seen_from_the_moved_axis() {
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let t = joint_table(&[&scan], &[a], &[b], &[window(0.08)], 2).unwrap();
        let p = JointParams::default();
        let d = [0.012, -0.007];
        let moved: Vec<Pulse> = t.pulses.iter().map(|pu| shifted(pu, d)).collect();
        let own: Vec<Owner> = t.pulses.iter().map(|_| Owner { row: 0, nu: u32::MAX }).collect();
        let at = joint_loglik(&t.pulses, &own, &[[d[0], d[1], 0.05f64.ln(), 0.0]], &[], &p, 2).unwrap();
        let from = joint_loglik(&moved, &own, &[[0.0, 0.0, 0.05f64.ln(), 0.0]], &[], &p, 2).unwrap();
        // equal but for the footprint, taken at the moved range of closest approach (divergence x the shift)
        let worst = at.iter().zip(&from).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(worst < 0.01, "worst {worst}");
        // a pulse aimed at the stem meets it before a circle behind it; a pulse aimed between two meets the nearer
        let first_circle = |pu: &Pulse, circles: &[(f64, f64, f64)]| {
            let views: Vec<Pulse> = circles.iter().map(|&(cx, cy, _)| shifted(pu, [cx, cy])).collect();
            first_met(&views, &circles.iter().map(|c| c.2).collect::<Vec<_>>(), p.footprint(pu.t))
        };
        let hit = t.pulses.iter().find(|pu| pu.stop && pu.b.abs() < 0.01).unwrap();
        let v = [hit.jt[0] * hit.sin, hit.jt[1] * hit.sin];
        let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let behind = (0.12 * v[0] / l, 0.12 * v[1] / l, 0.03);
        assert_eq!(first_circle(hit, &[behind, (0.0, 0.0, 0.05)]), 1);
        assert_eq!(first_circle(hit, &[(0.0, 0.0, 0.05), behind]), 0);
    }
}
