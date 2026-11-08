// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One tree's wood from the support function of each row's cross-section, as
//! each scan measures it, and one linear solve over the tree (3dtrees
//! research 93 §10, §16): the fast path of the joint model, with flags that
//! send the rows it can't summarise to [`crate::qsm::joint_tree`].
//!
//! Within one (row, scan) group the pulses are nearly parallel, so they
//! reduce to a profile across the row: impact `b` (along the group's `n`,
//! across the beam) and, where a pulse stopped, depth (along `v`, the beam's
//! direction in the cross-section). For opaque wood that profile measures the
//! cross-section's support function, linear in a circle's centre `c` and
//! radius `R`: the silhouette's edges `n.c ± R_app`, `R_app = R + w (1/2 -
//! tau)` (w the footprint's width), and the front's depth `v.c - R`. The row
//! may lean: `c(s) = c0 + s t`, `s` along the axis from the row's middle.
//! A row longer than [`PIECE_LEN`] is measured in pieces no longer than that,
//! each with its own lean, and their summaries are combined exactly into the
//! row's line (a bent row then shows as misfit).
//!
//! - **Pairs.** Each pulse whose line comes closest to a piece's axis within
//!   the piece, inside a tube wide enough for the lean (its tube in
//!   [`crate::qsm::joint::window`] plus a quarter of its length), with its
//!   fate there: a stop (its first echo in the tube, with its depth), a pass,
//!   or blocked (its last echo is before the tube: it says nothing about it).
//! - **Echo character.** A stop counts as a hit on wood with the weight
//!   `P(wood | reflectance, return position)` from the densities of
//!   [`JointParams`] (research 93 §11.2), wood's mean darkened by `10
//!   log10(2R/w)` where the wood is narrower than the footprint; the rest of
//!   the stop is a pulse blocked by leaf.
//! - **Per group, twice.** The lean across the beam (a line through the hits'
//!   centre against `s`) is removed; the profile is binned no finer than the
//!   impacts' spacing; runs of mostly-hit bins are cut where the hits' depth
//!   jumps, so stems side by side in a projection separate. A run of the
//!   claim overlaps it across the beam and in depth; the one nearest the axis
//!   is the row's, and the lean is taken again from its own hits and the
//!   profile cut again. Each of its edges comes by maximum likelihood on its
//!   hit/pass transition (the joint likelihood's probit of sd `edge w`, under
//!   the run's own hit share), with its variance and the impacts'
//!   quantisation. An edge with stops in front of where it would end (or
//!   another run) is censored and left out; one with nothing recorded beyond
//!   it was seen against the sky (an export of echoes only keeps no misses)
//!   and ends half a spacing past its outermost hit. The front's depth near
//!   the run's middle, regressed on `s`, gives `v.c0 - R` and `v.t`. The
//!   second pass gates each stop by the first pass's circle: on its front arc
//!   it is a hit, well behind it a pass, in front of it blocked.
//! - **Per row,** the measurements' information about `(c0, t, R)`, which may
//!   be rank-deficient (a one-sided row's lean unseen): the tree's prior
//!   fills what it lacks. Its estimate, with a weak anchor of the centre and
//!   lean, is inflated by the misfit (reduced chi-square on the rank).
//! - **Flags:** [`flag::RUNS`] (a group's profile, its stops gated by the
//!   fitted circle's depth band, shows two or more stems within the claim),
//!   [`flag::HULL`] (the support is wider than one circle explains: more than
//!   a share of the stops aimed at the fitted circle's core lie behind its
//!   front, the measurements or the front's stops misfit it, or the row's run
//!   is far wider than it), [`flag::FEW`] (R isn't determined), [`flag::POROUS`]
//!   (pulses but no opaque run). A flagged row's summary doesn't enter the
//!   tree: its radius there is the prior's, and the exact likelihood is for it.
//! - **The tree:** the unflagged rows' summaries as Gaussian measurements of
//!   `(c0, t, R)`, and a Gaussian Markov prior over the rows after the joint
//!   fit's [`PriorParams`] (the base's level, the taper's steps, at a fork the
//!   child's share of the subtree wood length rather than of the prior area,
//!   the centre's and lean's continuity along a branch, the centre's anchor on
//!   the claim and a weak anchor of the lean; no tip anchor). Every term links
//!   a row only to its parent, so the system is solved exactly by eliminating
//!   rows into their parents, leaves first (as [`crate::qsm::joint_fit`]'s
//!   branch blocks, which without the per-scan offsets are single rows), with
//!   each row's marginal covariance. The radius is measured in R and modelled
//!   in log R: three Gauss-Newton steps.

use nalgebra::{Matrix5, SymmetricEigen, Vector5};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::evidence::ScanIndex;
use crate::qsm::joint::{pool, window, JointParams};
use crate::qsm::joint_fit::{ForkShare, Frame, PriorParams, Rows};
use crate::transform::{cross, dot, sub};
use crate::Point;

/// Why a row's own summary is not used (bits of [`SupportTree::flags`]).
pub mod flag {
    /// A group's profile holds two or more stems within the claim.
    pub const RUNS: u8 = 1;
    /// The support is wider than one circle explains.
    pub const HULL: u8 = 2;
    /// Too few edges or depths to determine the circle.
    pub const FEW: u8 = 4;
    /// Pulses, but no opaque run.
    pub const POROUS: u8 = 8;
}

/// A group needs this much (wood-weighted) pulses that say something (hits and passes).
const MIN_PULSES: f64 = 30.0;
/// A run's least wood-weighted hits, to be the row's or to count as a stem beside it.
const RUN_HITS: f64 = 10.0;
/// The tube's widening for the lean, a share of the piece's length (a lean of up to about 27°).
const LEAN_TUBE: f64 = 0.25;
/// Rows longer than this (m) are measured in pieces of at most this length.
pub const PIECE_LEN: f64 = 0.3;
/// The lean's anchor (m per m), and the floor on each measurement's sd (m): non-round sections, a ~4 mm offset
/// between scans that no sway explains (research 93 §15.6; at 2 mm Alice Mulga's clean rows misfit by a reduced
/// chi-square of 4-9).
const LEAN_SD: f64 = 0.5;
const MODEL_SD: f64 = 0.004;
/// The narrowest bin (m), the depth jump that separates two stems in a run (m), and how far a stop may sit from
/// a run's depths or a circle's front and still be on it (`DEPTH_TOL + 0.3 R`).
const BIN_MIN: f64 = 0.002;
const DEPTH_JUMP: f64 = 0.02;
const DEPTH_TOL: f64 = 0.02;
/// A stop aimed at a circle's core is "behind" it this far behind its front (`HULL_BEHIND + 0.1 R`); a row is a
/// hull when more than `HULL_SHARE` of them are, or a misfit (reduced chi-square) exceeds `HULL_CHI2`.
const HULL_BEHIND: f64 = 0.01;
const HULL_SHARE: f64 = 0.25;
const HULL_CHI2: f64 = 9.0;
/// The bound on a stop's log likelihood ratio of wood to leaf (research 93 §11.2).
const LLR_CLIP: f64 = 3.0;

/// The fit of one tree: per row (in the input's order).
#[derive(Debug, Clone, PartialEq)]
pub struct SupportTree {
    /// The fitted axis (world coordinates: the claim moved by the centre and turned by the lean) and radius, with
    /// the posterior sd of log R. Where the row is flagged these are the prior's, given the other rows.
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r: Vec<f64>,
    pub sd_log_r: Vec<f64>,
    /// The centre at the row's middle and the lean (per m along the axis), along `e1, e2`, with their sds.
    pub centre: Vec<[f64; 2]>,
    pub sd_centre: Vec<[f64; 2]>,
    pub lean: Vec<[f64; 2]>,
    pub sd_lean: Vec<[f64; 2]>,
    pub e1: Vec<Point>,
    pub e2: Vec<Point>,
    /// The row's own summary, NaN where it is flagged or has none: its radius and sd.
    pub r_row: Vec<f64>,
    pub sd_r_row: Vec<f64>,
    /// The row's misfits, NaN where it has none: of its measurements to its circle (reduced chi-square on their
    /// rank), and of its front's stops about that circle against the range noise, roughness and the floor.
    pub chi2: Vec<f64>,
    pub chi2_front: Vec<f64>,
    /// [`flag`] bits: the row's summary entered the tree only where 0.
    pub flags: Vec<u8>,
    /// The most runs (stems within the claim) any group shows, the groups that gave measurements, and the
    /// row's pairs.
    pub n_runs: Vec<u32>,
    pub n_scans: Vec<u32>,
    pub n_pairs: Vec<u32>,
}

/// A pulse in a piece's cross-section: impact, axial position from the piece's middle, depth of its stop (NaN:
/// none in the tube), footprint width, its stop's reflectance (dB) and whether it was its pulse's only return, and
/// whether it was blocked before the tube.
#[derive(Debug, Clone, Copy)]
struct Rec {
    b: f64,
    s: f64,
    depth: f64,
    w: f64,
    refl: f64,
    single: bool,
    blocked: bool,
}

/// One (piece, scan) group: the median impact and depth directions (in the row's `e1, e2`) and its pulses.
struct Group {
    n: [f64; 2],
    v: [f64; 2],
    recs: Vec<Rec>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fate {
    Hit,
    Pass,
    Blocked,
}

/// One linear measurement of a row's `[c1, c2, t1, t2, R]`: coefficients, value and variance.
type Meas = ([f64; 5], f64, f64);

/// Sufficient statistics of a set of measurements: `A^T W A`, `A^T W y`, `y^T W y` and their count.
#[derive(Debug, Clone, Copy)]
struct Stats {
    info: Matrix5<f64>,
    h: Vector5<f64>,
    yy: f64,
    n: usize,
}

impl Stats {
    fn of(meas: &[Meas]) -> Self {
        let mut st = Stats { info: Matrix5::zeros(), h: Vector5::zeros(), yy: 0.0, n: meas.len() };
        for (a, y, var) in meas {
            let a = Vector5::from(*a);
            st.info += a * a.transpose() / *var;
            st.h += a * (*y / *var);
            st.yy += y * y / var;
        }
        st
    }

    /// The same measurements of a piece whose middle is `s` along the row, as measurements of the row's state
    /// (the piece's centre is the row's `c0 + s t`).
    fn moved(&self, s: f64) -> Self {
        let mut a = Matrix5::<f64>::identity();
        a[(0, 2)] = s;
        a[(1, 3)] = s;
        Stats { info: a.transpose() * self.info * a, h: a.transpose() * self.h, ..*self }
    }

    fn add(&mut self, o: &Stats) {
        self.info += o.info;
        self.h += o.h;
        self.yy += o.yy;
        self.n += o.n;
    }
}

/// A row's circle `[c1, c2, t1, t2, R]` (centre at the middle, lean, radius): its measurements' statistics, the
/// estimate and covariance (with a weak anchor of the centre and lean on the claim, inflated by the misfit) and
/// the misfit.
#[derive(Debug, Clone, Copy)]
struct Circle {
    st: Stats,
    x: Vector5<f64>,
    cov: Matrix5<f64>,
    chi2: f64,
}

impl Circle {
    fn inflate(&self) -> f64 {
        if self.chi2 > 1.0 { self.chi2 } else { 1.0 }
    }
}

/// What a row's own pulses say.
struct RowSummary {
    circle: Option<Circle>,
    chi2_front: f64,
    flags: u8,
    n_runs: u32,
    n_scans: u32,
    n_pairs: u32,
}

impl RowSummary {
    /// The circle, where the row's summary enters the tree.
    fn used(&self) -> Option<&Circle> {
        if self.flags == 0 { self.circle.as_ref() } else { None }
    }
}

fn ncdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
}

fn npdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn median(v: &mut [f64]) -> f64 {
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f64::total_cmp).1
}

/// The weight of a stop as a hit on wood of radius `r`: the chance its echo came from wood rather than leaf by its
/// reflectance and return position (research 93 §11.2, even odds, the log ratio clipped to ±[`LLR_CLIP`]), over
/// that chance at the clip, so an echo as surely wood as the clip allows counts fully (a uniform discount of every
/// hit would pull each edge inwards). Wood narrower than the footprint `w` fills only `2r/w` of it and reads that
/// much darker. An echo of unknown reflectance: 1.
fn p_wood(rec: &Rec, r: f64, p: &JointParams) -> f64 {
    if !rec.refl.is_finite() {
        return 1.0;
    }
    let (shift, var) = if rec.single { (0.0, 0.0) } else { (p.refl_multi, p.refl_multi_sd * p.refl_multi_sd) };
    let thin = 10.0 * (2.0 * r / rec.w).clamp(1e-3, 1.0).log10();
    let ll = |m: f64, sd: f64| {
        let s = (sd * sd + var).sqrt();
        -0.5 * ((rec.refl - m) / s).powi(2) - s.ln()
    };
    let llr = (ll(p.refl_wood + shift + thin, p.refl_wood_sd) - ll(p.refl_leaf + shift, p.refl_leaf_sd)).clamp(-LLR_CLIP, LLR_CLIP);
    (1.0 + (-LLR_CLIP).exp()) / (1.0 + (-llr).exp())
}

/// Each pulse's weight for its fate: a hit counts as wood by [`p_wood`] (for wood of radius `r`), a pass fully,
/// a blocked pulse not at all.
fn weights(g: &Group, fates: &[Fate], r: f64, p: &JointParams) -> Vec<f64> {
    g.recs.iter().zip(fates).map(|(rec, f)| match f {
        Fate::Hit => p_wood(rec, r, p),
        Fate::Pass => 1.0,
        Fate::Blocked => 0.0,
    }).collect()
}

/// A piece's groups, one per scan with pulses in its tube.
fn groups(scans: &[&ScanIndex], a: &Point, b: &Point, frame: Frame, tube: f64, p: &JointParams) -> Vec<Group> {
    let (u, e1, e2, len) = frame;
    let mut out = Vec::new();
    for sc in scans {
        let o = sc.origin();
        let w0 = sub(&o, a);
        let dd = dot(&w0, &u);
        let mut recs = Vec::new();
        let (mut ns, mut vs) = (Vec::new(), Vec::new());
        sc.pulses_near(a, b, tube, |pulse, d, echoes| {
            let bb = dot(&d, &u);
            let den = 1.0 - bb * bb;
            if den < 1e-12 {
                return;
            }
            let ee = dot(&d, &w0);
            let (t, s_ax) = ((bb * dd - ee) / den, (dd - bb * ee) / den);
            if !(t > 0.0 && (0.0..=len).contains(&s_ax)) {
                return;
            }
            let sin = den.sqrt();
            let n = cross(&u, &d).map(|x| x / sin);
            let bimp = dot(&w0, &n);
            if bimp.abs() >= tube {
                return;
            }
            let v: Point = std::array::from_fn(|i| (d[i] - bb * u[i]) / sin);
            let half = (tube * tube - bimp * bimp).sqrt() / sin;
            let (c0, c1) = (t - half, t + half);
            let first = echoes.iter().position(|&e| (e as f64) >= c0 && (e as f64) <= c1);
            let blocked = echoes.last().is_some_and(|&e| (e as f64) < c0);
            let (depth, refl) = match first {
                Some(i) => (((echoes[i] as f64) - t) * sin, sc.character(pulse, i as u32).0),
                None => (f64::NAN, f64::NAN),
            };
            ns.push([dot(&n, &e1), dot(&n, &e2)]);
            vs.push([dot(&v, &e1), dot(&v, &e2)]);
            recs.push(Rec { b: bimp, s: s_ax - 0.5 * len, depth, w: p.footprint(t), refl, single: echoes.len() == 1, blocked });
        });
        if recs.is_empty() {
            continue;
        }
        let unit = |m: Vec<[f64; 2]>| {
            let mut x: Vec<f64> = m.iter().map(|q| q[0]).collect();
            let mut y: Vec<f64> = m.iter().map(|q| q[1]).collect();
            let (x, y) = (median(&mut x), median(&mut y));
            let l = x.hypot(y).max(1e-12);
            [x / l, y / l]
        };
        out.push(Group { n: unit(ns), v: unit(vs), recs });
    }
    out
}

/// A pulse's fate: on the first pass from its echo alone; on the second, against the circle `fit`.
fn fate(r: &Rec, n: [f64; 2], v: [f64; 2], fit: Option<&Vector5<f64>>) -> Fate {
    if r.blocked {
        return Fate::Blocked;
    }
    if !r.depth.is_finite() {
        return Fate::Pass;
    }
    let Some(x) = fit else { return Fate::Hit };
    let (cx, cy, rr) = (x[0] + r.s * x[2], x[1] + r.s * x[3], x[4]);
    let bc = r.b - (n[0] * cx + n[1] * cy);
    let front = v[0] * cx + v[1] * cy - (rr * rr - bc * bc).max(0.0).sqrt();
    let tol = DEPTH_TOL + 0.3 * rr;
    if bc.abs() < rr + DEPTH_TOL && (r.depth - front).abs() < tol {
        Fate::Hit
    } else if r.depth > front + tol {
        Fate::Pass
    } else {
        Fate::Blocked
    }
}

/// The slope of `b` against `s` (a line through the medians of six slices of `s`) and its variance; None with
/// fewer than 10 points or three slices, or no spread in `s`.
fn lean_line(pts: &mut [(f64, f64)]) -> Option<(f64, f64)> {
    if pts.len() < 10 {
        return None;
    }
    pts.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    let m = pts.len();
    let mut xy = Vec::new();
    for j in 0..6 {
        let sl = &pts[j * m / 6..(j + 1) * m / 6];
        if sl.len() >= 3 {
            let mut s: Vec<f64> = sl.iter().map(|q| q.0).collect();
            let mut b: Vec<f64> = sl.iter().map(|q| q.1).collect();
            xy.push((median(&mut s), median(&mut b)));
        }
    }
    if xy.len() < 3 {
        return None;
    }
    let k = xy.len() as f64;
    let (mx, my) = (xy.iter().map(|q| q.0).sum::<f64>() / k, xy.iter().map(|q| q.1).sum::<f64>() / k);
    let sxx: f64 = xy.iter().map(|q| (q.0 - mx).powi(2)).sum();
    if sxx < 1e-8 {
        return None;
    }
    let slope = xy.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum::<f64>() / sxx;
    let rss: f64 = xy.iter().map(|q| (q.1 - my - slope * (q.0 - mx)).powi(2)).sum();
    Some((slope, (rss / (k - 2.0)).max(MODEL_SD * MODEL_SD) / sxx))
}

/// A run of mostly-hit bins, cut where its depth jumps: its impact bounds, wood-weighted hits, bins, and the range
/// of its bins' median depths.
#[derive(Debug, Clone, Copy)]
struct Piece {
    lo: f64,
    hi: f64,
    hits: f64,
    bins: usize,
    dlo: f64,
    dhi: f64,
}

/// The impacts' spacing (`b * 1e4` rounded, the informative pulses') and the bin width no finer than it.
fn spacing(bp: &[f64], fates: &[Fate]) -> (f64, f64) {
    let mut ub: Vec<i64> = bp.iter().zip(fates).filter(|(_, &f)| f != Fate::Blocked).map(|(&b, _)| (b * 1e4).round() as i64).collect();
    ub.sort_unstable();
    ub.dedup();
    let step = if ub.len() > 2 { median(&mut ub.windows(2).map(|w| (w[1] - w[0]) as f64 * 1e-4).collect::<Vec<_>>()) } else { 0.0 };
    (step, (1.5 * step).max(BIN_MIN))
}

/// The pieces of a profile: `bp` impacts (lean removed), fates, weights and depths, bins `bw` wide; a run bridges
/// a stretch of bins with fewer than two informative pulses up to `bridge` wide (it says little: a scan's columns
/// can be sparser than the bins, and a lean smears a column across a bin's edge).
fn pieces(bp: &[f64], fates: &[Fate], wt: &[f64], depth: &[f64], bw: f64, bridge: f64) -> Vec<Piece> {
    let inf: Vec<usize> = (0..bp.len()).filter(|&i| fates[i] != Fate::Blocked).collect();
    if inf.is_empty() {
        return Vec::new();
    }
    let lo0 = inf.iter().map(|&i| bp[i]).fold(f64::INFINITY, f64::min);
    let hi0 = inf.iter().map(|&i| bp[i]).fold(f64::NEG_INFINITY, f64::max);
    let nb = ((hi0 - lo0) / bw) as usize + 1;
    let (mut cnt, mut n, mut h) = (vec![0usize; nb], vec![0.0; nb], vec![0.0; nb]);
    let mut d: Vec<Vec<f64>> = vec![Vec::new(); nb];
    for &i in &inf {
        let j = (((bp[i] - lo0) / bw) as usize).min(nb - 1);
        cnt[j] += 1;
        n[j] += wt[i];
        if fates[i] == Fate::Hit {
            h[j] += wt[i];
            if wt[i] >= 0.5 {
                d[j].push(depth[i]);
            }
        }
    }
    let mut isw: Vec<bool> = (0..nb).map(|j| n[j] >= 1.5 && 2.0 * h[j] >= n[j]).collect();
    for j in 1..nb.saturating_sub(1) {
        if !isw[j] && isw[j - 1] && isw[j + 1] && n[j] < 4.0 {
            isw[j] = true; // a one-bin gap of few pulses
        }
    }
    let mut j = 1;
    while j < nb {
        let mut e = j;
        while !isw[e] && cnt[e] < 2 && e + 1 < nb {
            e += 1;
        }
        if e > j && isw[j - 1] && isw[e] && (e - j) as f64 * bw <= bridge {
            isw[j..e].fill(true);
        }
        j = e + 1;
    }
    let med: Vec<f64> = d.iter_mut().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
    let mut out = Vec::new();
    let mut j = 0;
    while j < nb {
        if !isw[j] {
            j += 1;
            continue;
        }
        let mut e = j;
        while e + 1 < nb && isw[e + 1] {
            e += 1;
        }
        // cut where consecutive bins' depths jump, leaving at least three bins either side
        let mut cuts = vec![j];
        let mut last = f64::NAN;
        for q in j..=e {
            if med[q].is_finite() {
                if last.is_finite() && (med[q] - last).abs() > DEPTH_JUMP && q - cuts[cuts.len() - 1] >= 3 && e + 1 - q >= 3 {
                    cuts.push(q);
                }
                last = med[q];
            }
        }
        cuts.push(e + 1);
        for c in cuts.windows(2) {
            let (a, z) = (c[0], c[1]);
            let ds: Vec<f64> = med[a..z].iter().copied().filter(|x| x.is_finite()).collect();
            out.push(Piece {
                lo: lo0 + a as f64 * bw,
                hi: lo0 + z as f64 * bw,
                hits: h[a..z].iter().sum(),
                bins: z - a,
                dlo: ds.iter().copied().fold(f64::INFINITY, f64::min),
                dhi: ds.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            });
        }
        j = e + 1;
    }
    out
}

/// A run's distance across the beam from `c` (0 inside it).
fn dist(q: &Piece, c: f64) -> f64 {
    if q.lo <= c && c <= q.hi { 0.0 } else { (q.lo - c).abs().min((q.hi - c).abs()) }
}

/// The edge `e` of a run's right side by Fisher scoring from `e0`, and its variance: a pulse at `b` is a hit with
/// probability `p_in Phi((e + w (1/2 - tau) - b) / (edge w))`, the joint likelihood's soft detection edge under
/// the run's own hit share `p_in` (thin or partly screened wood isn't hit by every pulse even at its middle).
/// `pts` are `(b, w, hit, weight)`.
fn edge(pts: &[(f64, f64, bool, f64)], e0: f64, p_in: f64, p: &JointParams) -> Option<(f64, f64)> {
    let score = |e: f64| {
        let (mut g, mut info) = (0.0, 0.0);
        for &(b, w, hit, wt) in pts {
            let sd = p.edge * w;
            let z = (e + p.app(w) - b) / sd;
            let ph = (p_in * ncdf(z)).clamp(1e-12, 1.0 - 1e-12);
            let f = p_in * npdf(z) / sd;
            g += wt * if hit { f / ph } else { -f / (1.0 - ph) };
            info += wt * f * f / (ph * (1.0 - ph));
        }
        (g, info)
    };
    let cap = pts.iter().map(|q| p.edge * q.1).fold(0.0, f64::max);
    let mut e = e0;
    for _ in 0..50 {
        let (g, info) = score(e);
        if !(info > 0.0) {
            return None;
        }
        let step = (g / info).clamp(-cap, cap);
        e += step;
        if step.abs() < 1e-7 {
            break;
        }
    }
    let (_, info) = score(e);
    (info > 0.0).then(|| (e, 1.0 / info))
}

/// What one group says on one pass: its measurements, whether it had enough pulses and an opaque run.
struct GroupOut {
    meas: Vec<Meas>,
    enough: bool,
    run: bool,
}

/// One group's measurements, its pulses' fates and weights given; `centre` is where the row's axis is expected
/// along `n` (`r_claim` its claimed radius).
fn measure(g: &Group, fates: &[Fate], wt: &[f64], centre: f64, r_claim: f64, depths: (f64, f64), p: &JointParams) -> GroupOut {
    let mut out = GroupOut { meas: Vec::new(), enough: false, run: false };
    if wt.iter().sum::<f64>() < MIN_PULSES {
        return out;
    }
    out.enough = true;
    let w_med = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
    let app = p.app(w_med);
    let depth: Vec<f64> = g.recs.iter().map(|r| r.depth).collect();
    let woody = |i: usize| fates[i] == Fate::Hit && wt[i] >= 0.5;
    // a run of the claim overlaps it across the beam and in depth (`depths`: where its front can lie)
    let reach = (centre - r_claim - 0.01, centre + r_claim + 0.01);
    let eligible = |q: &Piece| q.hits >= RUN_HITS && q.bins >= 2 && q.hi > reach.0 && q.lo < reach.1 && q.dhi > depths.0 && q.dlo < depths.1;
    // the lean across the beam: a line through the hits' centre against s, first of the hits near the claim, then
    // of the chosen run's own; the profile is cut again with the second
    let mut near: Vec<(f64, f64)> = (0..depth.len()).filter(|&i| woody(i) && (g.recs[i].b - centre).abs() < 1.5 * r_claim + 0.01 && depth[i] > depths.0 && depth[i] < depths.1).map(|i| (g.recs[i].s, g.recs[i].b)).collect();
    let mut lean = lean_line(&mut near);
    let (mut bp, mut step, mut bw, mut chosen, mut aim) = (Vec::new(), 0.0, BIN_MIN, None, centre);
    for round in 0..2 {
        let k = lean.map_or(0.0, |l| l.0);
        bp = g.recs.iter().map(|r| r.b - k * r.s).collect();
        // bins no finer than the impacts' spacing (a scan's columns can align with the row: then every pulse along
        // it shares a column's impact)
        (step, bw) = spacing(&bp, fates);
        let ps: Vec<Piece> = pieces(&bp, fates, wt, &depth, bw, w_med).into_iter().filter(eligible).collect();
        chosen = ps.iter().min_by(|x, y| dist(x, aim).total_cmp(&dist(y, aim))).copied();
        let Some(pc) = chosen else { break };
        if round == 0 {
            let mut own: Vec<(f64, f64)> = (0..depth.len()).filter(|&i| woody(i) && bp[i] >= pc.lo && bp[i] <= pc.hi && depth[i] >= pc.dlo - DEPTH_TOL && depth[i] <= pc.dhi + DEPTH_TOL).map(|i| (g.recs[i].s, g.recs[i].b)).collect();
            match lean_line(&mut own) {
                Some(l) => (lean, aim) = (Some(l), 0.5 * (pc.lo + pc.hi)),
                None => break,
            }
        }
    }
    let Some(pc) = chosen else { return out };
    out.run = true;
    // fates relative to this piece: a stop well behind its depths passed it, one well in front never reached it
    let local: Vec<Fate> = fates.iter().zip(&depth).map(|(&f, &d)| match f {
        Fate::Hit if d > pc.dhi + DEPTH_TOL => Fate::Pass,
        Fate::Hit if d < pc.dlo - DEPTH_TOL => Fate::Blocked,
        f => f,
    }).collect();
    let lw: Vec<f64> = local.iter().zip(fates).zip(wt).map(|((&l, &f), &w)| if l == f { w } else if l == Fate::Pass { 1.0 } else { 0.0 }).collect();
    let sd_max = p.edge * g.recs.iter().map(|r| r.w).fold(0.0, f64::max);
    let reach_out = 2.0 * sd_max + 2.0 * bw;
    // the run's own hit share, in its middle half
    let (q0, q1) = (0.75 * pc.lo + 0.25 * pc.hi, 0.25 * pc.lo + 0.75 * pc.hi);
    let (mut nh, mut np) = (0.0, 0.0);
    for i in (0..bp.len()).filter(|&i| bp[i] >= q0 && bp[i] <= q1) {
        match local[i] {
            Fate::Hit => nh += lw[i],
            Fate::Pass => np += 1.0,
            Fate::Blocked => {}
        }
    }
    let p_in = if nh + np > 0.0 { (nh / (nh + np)).max(0.2) } else { 1.0 };
    let mid = 0.5 * (pc.lo + pc.hi);
    let mut edges = [None, None];
    for (side, sign) in [(0usize, 1.0), (1, -1.0)] {
        // the right edge, or the left one mirrored; its window reaches into the run no further than its middle
        let at = if sign > 0.0 { pc.hi } else { -pc.lo };
        let inner = (at - 4.0 * sd_max - bw).max(sign * mid);
        let beyond = |f: Fate| (0..bp.len()).filter(|&i| local[i] == f && sign * bp[i] > at && sign * bp[i] <= at + reach_out).count();
        let pts: Vec<(f64, f64, bool, f64)> = (0..bp.len()).filter(|&i| local[i] != Fate::Blocked && sign * bp[i] >= inner && sign * bp[i] <= at + 4.0 * sd_max + bw).map(|i| (sign * bp[i], g.recs[i].w, local[i] == Fate::Hit, lw[i])).collect();
        if beyond(Fate::Pass) < 3 {
            if beyond(Fate::Blocked) + beyond(Fate::Hit) > 0 {
                continue; // censored: stopped in front of where it ends, or another run's
            }
            // nothing recorded beyond it: the scan's pulses there missed (an export of echoes only keeps no
            // misses), so it ends half a spacing past its outermost hit
            let out_hit = (0..bp.len()).filter(|&i| local[i] == Fate::Hit && lw[i] >= 0.5 && sign * bp[i] <= at).map(|i| sign * bp[i]).fold(f64::NEG_INFINITY, f64::max);
            if out_hit.is_finite() {
                edges[side] = Some((sign * (out_hit + 0.5 * step - app), step * step / 12.0 + MODEL_SD * MODEL_SD));
            }
            continue;
        }
        if pts.iter().filter(|q| q.2).map(|q| q.3).sum::<f64>() < 1.5 {
            continue;
        }
        // an edge that ran out of its window is no edge
        if let Some((e, var)) = edge(&pts, at - app, p_in, p).filter(|&(e, _)| e + app >= inner && e + app <= at + 4.0 * sd_max + bw) {
            // (plus the impacts' quantisation: aligned columns place an edge only between them)
            edges[side] = Some((sign * e, var + step * step / 12.0 + MODEL_SD * MODEL_SD));
        }
    }
    if let [Some((r, _)), Some((l, _))] = edges {
        if r <= l {
            edges = [None, None];
        }
    }
    let [nx, ny] = g.n;
    if let Some((e, var)) = edges[0] {
        out.meas.push(([nx, ny, 0.0, 0.0, 1.0], e, var));
    }
    if let Some((e, var)) = edges[1] {
        out.meas.push(([nx, ny, 0.0, 0.0, -1.0], e, var));
    }
    if let Some((k, var)) = lean {
        out.meas.push(([0.0, 0.0, nx, ny, 0.0], k, var));
    }
    // the front's depth near the middle: each hit's depth moved to the apex by the arc, regressed on s (weighted)
    let (mid, r0) = match edges {
        [Some((r, _)), Some((l, _))] => (0.5 * (r + l), (0.5 * (r - l)).max(0.001)),
        _ => (0.5 * (pc.lo + pc.hi), (0.5 * (pc.hi - pc.lo) - app).max(0.001)),
    };
    let mut pts: Vec<(f64, f64, f64)> = (0..bp.len()).filter(|&i| local[i] == Fate::Hit && lw[i] > 0.0 && (bp[i] - mid).abs() <= 0.6 * r0).map(|i| {
        let bc = bp[i] - mid;
        (g.recs[i].s, depth[i] - r0 + (r0 * r0 - bc * bc).sqrt(), lw[i])
    }).collect();
    let floor = (p.sigma_r * p.sigma_r + (p.rough * r0).powi(2)).max(MODEL_SD * MODEL_SD);
    for round in 0..2 {
        let m: f64 = pts.iter().map(|q| q.2).sum();
        if pts.len() < 5 || m < 3.0 {
            break;
        }
        let (ms, my) = (pts.iter().map(|q| q.2 * q.0).sum::<f64>() / m, pts.iter().map(|q| q.2 * q.1).sum::<f64>() / m);
        let sxx: f64 = pts.iter().map(|q| q.2 * (q.0 - ms).powi(2)).sum();
        let beta = if sxx > 1e-6 { pts.iter().map(|q| q.2 * (q.0 - ms) * (q.1 - my)).sum::<f64>() / sxx } else { 0.0 };
        let res: Vec<f64> = pts.iter().map(|q| q.1 - my - beta * (q.0 - ms)).collect();
        if round == 0 {
            // drop the mixed pixels and strays once
            let mad = 1.4826 * median(&mut res.iter().map(|x| x.abs()).collect::<Vec<_>>());
            let cut = 3.0 * mad.max(floor.sqrt());
            let kept: Vec<(f64, f64, f64)> = pts.iter().zip(&res).filter(|(_, r)| r.abs() <= cut).map(|(q, _)| *q).collect();
            pts = kept;
            continue;
        }
        let s2 = (pts.iter().zip(&res).map(|(q, r)| q.2 * r * r).sum::<f64>() / (m - 2.0)).max(floor);
        let alpha = my - beta * ms;
        let at_mid = if sxx > 1e-6 { ms * ms / sxx } else { 0.0 };
        out.meas.push(([g.v[0], g.v[1], 0.0, 0.0, -1.0], alpha, s2 * (1.0 / m + at_mid) + MODEL_SD * MODEL_SD));
        if sxx > 1e-6 {
            out.meas.push(([0.0, 0.0, g.v[0], g.v[1], 0.0], beta, s2 / sxx));
        }
    }
    out
}

/// The stems a group's profile shows within the claim (`r_claim` about the circle `x`'s centre), its stops gated by
/// depth alone, the circle's band `v.c(s) ± (R + tol)` (a stop behind it passed, one in front never reached it),
/// so stems side by side separate and wood behind or in front doesn't count, and cut where their depth from its
/// front arc jumps; and whether the run nearest the circle is far wider than it.
fn runs_of(g: &Group, x: &Vector5<f64>, r_claim: f64, p: &JointParams) -> (usize, bool) {
    let rr = x[4];
    let tol = DEPTH_TOL + 0.3 * rr;
    let fates: Vec<Fate> = g.recs.iter().map(|r| {
        if r.blocked {
            return Fate::Blocked;
        }
        if !r.depth.is_finite() {
            return Fate::Pass;
        }
        let mid = g.v[0] * (x[0] + r.s * x[2]) + g.v[1] * (x[1] + r.s * x[3]);
        if r.depth > mid + rr + tol { Fate::Pass } else if r.depth < mid - rr - tol { Fate::Blocked } else { Fate::Hit }
    }).collect();
    let wt = weights(g, &fates, rr, p);
    let (k, centre) = (g.n[0] * x[2] + g.n[1] * x[3], g.n[0] * x[0] + g.n[1] * x[1]);
    let bp: Vec<f64> = g.recs.iter().map(|r| r.b - k * r.s).collect();
    // depth from the circle's front arc, so its own face is flat and a jump is another body
    let depth: Vec<f64> = g.recs.iter().map(|r| {
        let (cx, cy) = (x[0] + r.s * x[2], x[1] + r.s * x[3]);
        let bc = r.b - (g.n[0] * cx + g.n[1] * cy);
        r.depth - (g.v[0] * cx + g.v[1] * cy - (rr * rr - bc * bc).max(0.0).sqrt())
    }).collect();
    let (_, bw) = spacing(&bp, &fates);
    let reach = (centre - r_claim - 0.01, centre + r_claim + 0.01);
    let w_med = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
    let ps: Vec<Piece> = pieces(&bp, &fates, &wt, &depth, bw, w_med).into_iter().filter(|q| q.hits >= RUN_HITS && q.bins >= 2 && q.hi > reach.0 && q.lo < reach.1).collect();
    let Some(pc) = ps.iter().min_by(|x, y| dist(x, centre).total_cmp(&dist(y, centre))).copied() else { return (0, false) };
    // the stems: runs at least a fifth of the circle's own
    (ps.iter().filter(|q| 5.0 * q.hits >= pc.hits).count(), pc.hi - pc.lo > 4.0 * rr + 0.04)
}

/// Weighted least squares for `[c1, c2, t1, t2, R]` from the measurements' statistics, the centre and lean weakly
/// anchored on the claim (`r_claim`, [`LEAN_SD`]; R free), with the covariance inflated by the misfit (reduced
/// chi-square on the measurements' rank, where they over-determine it); None where R isn't determined (its sd
/// above half its value and the floor).
fn solve_row(st: &Stats, r_claim: f64) -> Option<Circle> {
    let sc = r_claim + 0.003;
    let anchor = Matrix5::from_diagonal(&Vector5::new(1.0 / (sc * sc), 1.0 / (sc * sc), 1.0 / (LEAN_SD * LEAN_SD), 1.0 / (LEAN_SD * LEAN_SD), 0.0));
    let m = st.info + anchor;
    let eig = SymmetricEigen::new(m);
    if !(eig.eigenvalues.min() > 1e-12 * eig.eigenvalues.max()) {
        return None;
    }
    let mut cov = m.try_inverse()?;
    let x = cov * st.h;
    if !(x[4] > 0.0 && cov[(4, 4)].sqrt() <= 0.5 * x[4] + MODEL_SD) {
        return None;
    }
    let ev = SymmetricEigen::new(st.info).eigenvalues;
    let rank = ev.iter().filter(|&&e| e > 1e-9 * ev.max()).count();
    let dof = st.n as f64 - rank as f64;
    let ssr = (st.yy - 2.0 * x.dot(&st.h) + x.dot(&(st.info * x))).max(0.0);
    let chi2 = if dof > 0.0 { ssr / dof } else { f64::NAN };
    let c = Circle { st: *st, x, cov, chi2 };
    cov *= c.inflate();
    Some(Circle { cov, ..c })
}

/// What one piece's pulses say: its second pass's measurements (None where it never got there), its flags (RUNS,
/// HULL by the stops behind its circle, POROUS), runs, scans and pairs, and its front's stops' misfit about its
/// circle (sum of squares in units of their variance, and degrees of freedom).
struct PieceSummary {
    st: Option<Stats>,
    flags: u8,
    n_runs: u32,
    n_scans: u32,
    n_pairs: u32,
    front: (f64, f64),
}

fn summarise_piece(scans: &[&ScanIndex], a: &Point, b: &Point, frame: Frame, r_claim: f64, p: &JointParams) -> PieceSummary {
    let tube = window(r_claim) + LEAN_TUBE * frame.3;
    let gs = groups(scans, a, b, frame, tube, p);
    let n_pairs = gs.iter().map(|g| g.recs.len()).sum::<usize>() as u32;
    let mut out = PieceSummary { st: None, flags: 0, n_runs: 0, n_scans: 0, n_pairs, front: (0.0, 0.0) };
    // the first pass: each stop by its echo
    let first: Vec<GroupOut> = gs.iter().map(|g| {
        let f: Vec<Fate> = g.recs.iter().map(|r| fate(r, g.n, g.v, None)).collect();
        measure(g, &f, &weights(g, &f, r_claim, p), 0.0, r_claim, (-2.0 * r_claim - DEPTH_TOL, r_claim + DEPTH_TOL), p)
    }).collect();
    if first.iter().any(|o| o.enough) && !first.iter().any(|o| o.run) {
        out.flags |= flag::POROUS;
        return out;
    }
    let Some(c1) = solve_row(&Stats::of(&first.iter().flat_map(|o| o.meas.clone()).collect::<Vec<_>>()), r_claim) else { return out };
    // the second: each stop gated by the first's circle
    let x = c1.x;
    let mut meas = Vec::new();
    let mut fates2 = Vec::with_capacity(gs.len());
    for g in &gs {
        let f: Vec<Fate> = g.recs.iter().map(|r| fate(r, g.n, g.v, Some(&x))).collect();
        let w = weights(g, &f, x[4], p);
        let o = measure(g, &f, &w, g.n[0] * x[0] + g.n[1] * x[1], x[4].max(0.005), (f64::NEG_INFINITY, f64::INFINITY), p);
        out.n_scans += u32::from(!o.meas.is_empty());
        meas.extend(o.meas);
        fates2.push((f, w));
    }
    let st = Stats::of(&meas);
    out.st = Some(st);
    let Some(c2) = solve_row(&st, r_claim) else { return out };
    let y = c2.x;
    let rr = y[4];
    // the stems beside it and the run's width, its stops gated by its depth band
    out.n_runs = gs.iter().map(|g| runs_of(g, &y, r_claim, p).0 as u32).max().unwrap_or(0);
    if out.n_runs >= 2 {
        out.flags |= flag::RUNS;
    }
    if gs.iter().any(|g| runs_of(g, &y, r_claim, p).1) {
        out.flags |= flag::HULL;
    }
    // the hull: of the stops (any echo, before any gate) aimed at the circle's core that reached its front, the
    // share behind it (research 93 §13.1); and the front's hits' misfit about it
    let (mut on, mut behind) = (0usize, 0usize);
    let var = (p.sigma_r * p.sigma_r + (p.rough * rr).powi(2)).max(MODEL_SD * MODEL_SD);
    let (mut ss, mut nw, mut ng) = (0.0, 0.0, 0usize);
    for (g, (f, w)) in gs.iter().zip(&fates2) {
        let mut used = false;
        for (i, r) in g.recs.iter().enumerate() {
            if r.blocked || !r.depth.is_finite() {
                continue;
            }
            let (cx, cy) = (y[0] + r.s * y[2], y[1] + r.s * y[3]);
            let bc = r.b - (g.n[0] * cx + g.n[1] * cy);
            if bc.abs() >= 0.8 * rr {
                continue;
            }
            let front = g.v[0] * cx + g.v[1] * cy - (rr * rr - bc * bc).sqrt();
            if r.depth >= front - DEPTH_TOL - 0.3 * rr {
                on += 1;
                behind += usize::from(r.depth > front + HULL_BEHIND + 0.1 * rr);
            }
            if f[i] == Fate::Hit && bc.abs() < 0.6 * rr {
                ss += w[i] * (r.depth - front).powi(2) / var;
                nw += w[i];
                used = true;
            }
        }
        ng += usize::from(used);
    }
    if on > 0 && behind as f64 > HULL_SHARE * on as f64 {
        out.flags |= flag::HULL;
    }
    out.front = (ss, nw - ng as f64);
    out
}

/// One row's summary: its pieces' (one, or more where it is longer than [`PIECE_LEN`]), their measurements moved
/// to the row's middle and solved together, then the flags.
fn summarise_row(scans: &[&ScanIndex], a: &Point, b: &Point, frame: Frame, r_claim: f64, p: &JointParams) -> RowSummary {
    let (u, e1, e2, len) = frame;
    let m = (len / PIECE_LEN).ceil().max(1.0) as usize;
    let pl = len / m as f64;
    let mut out = RowSummary { circle: None, chi2_front: f64::NAN, flags: 0, n_runs: 0, n_scans: 0, n_pairs: 0 };
    let mut st: Option<Stats> = None;
    let mut front = (0.0, 0.0);
    for j in 0..m {
        let at = |x: f64| -> Point { std::array::from_fn(|i| a[i] + x * u[i]) };
        let (pa, pb) = if m == 1 { (*a, *b) } else { (at(j as f64 * pl), at((j + 1) as f64 * pl)) };
        let ps = summarise_piece(scans, &pa, &pb, (u, e1, e2, pl), r_claim, p);
        out.flags |= ps.flags;
        out.n_runs = out.n_runs.max(ps.n_runs);
        out.n_scans = out.n_scans.max(ps.n_scans);
        out.n_pairs += ps.n_pairs;
        front = (front.0 + ps.front.0, front.1 + ps.front.1);
        if let Some(s) = ps.st {
            let s = s.moved((j as f64 + 0.5) * pl - 0.5 * len);
            match st.as_mut() {
                Some(t) => t.add(&s),
                None => st = Some(s),
            }
        }
    }
    if front.1 > 0.0 {
        out.chi2_front = front.0 / front.1;
    }
    if out.flags & flag::POROUS != 0 {
        return out;
    }
    out.circle = st.and_then(|s| solve_row(&s, r_claim));
    match &out.circle {
        None => out.flags |= flag::FEW,
        Some(c) if c.chi2 > HULL_CHI2 || out.chi2_front > HULL_CHI2 => out.flags |= flag::HULL,
        _ => {}
    }
    out
}

/// Fits one tree from its rows' support summaries: `rows` (with `r_max` ignored) against `scans`, the prior's
/// widths from `prior` and the footprint, detection threshold, range noise and echo character from `like`. Runs on
/// `threads` threads (0: one per core).
pub fn support_tree(scans: &[&ScanIndex], rows: &Rows, like: &JointParams, prior: &PriorParams, threads: usize) -> Result<SupportTree> {
    let fr = rows.check("support tree")?;
    prior.check()?;
    let n = rows.a.len();
    let sums: Vec<RowSummary> = pool(threads)?.install(|| {
        (0..n).into_par_iter().map(|k| summarise_row(scans, &rows.a[k], &rows.b[k], fr[k], rows.r_prior[k], like)).collect()
    });
    let (x, cov) = tree_solve(rows, &fr, &sums, prior)?;

    let mut out = SupportTree {
        a: Vec::new(), b: Vec::new(), r: Vec::new(), sd_log_r: Vec::new(), centre: Vec::new(), sd_centre: Vec::new(), lean: Vec::new(), sd_lean: Vec::new(),
        e1: Vec::new(), e2: Vec::new(), r_row: Vec::new(), sd_r_row: Vec::new(), chi2: Vec::new(), chi2_front: Vec::new(), flags: Vec::new(),
        n_runs: Vec::new(), n_scans: Vec::new(), n_pairs: Vec::new(),
    };
    for k in 0..n {
        let (_, e1, e2, len) = fr[k];
        let xk = x[k];
        let at = |s: f64| -> Point { std::array::from_fn(|i| (xk[0] + s * xk[2]) * e1[i] + (xk[1] + s * xk[3]) * e2[i]) };
        let (oa, ob) = (at(-0.5 * len), at(0.5 * len));
        out.a.push(std::array::from_fn(|i| rows.a[k][i] + oa[i]));
        out.b.push(std::array::from_fn(|i| rows.b[k][i] + ob[i]));
        out.r.push(xk[4].exp());
        let sd = |i: usize| cov[k][(i, i)].max(0.0).sqrt();
        out.sd_log_r.push(sd(4));
        out.centre.push([xk[0], xk[1]]);
        out.sd_centre.push([sd(0), sd(1)]);
        out.lean.push([xk[2], xk[3]]);
        out.sd_lean.push([sd(2), sd(3)]);
        out.e1.push(e1);
        out.e2.push(e2);
        let s = &sums[k];
        out.r_row.push(s.used().map_or(f64::NAN, |c| c.x[4]));
        out.sd_r_row.push(s.used().map_or(f64::NAN, |c| c.cov[(4, 4)].max(0.0).sqrt()));
        out.chi2.push(s.circle.map_or(f64::NAN, |c| c.chi2));
        out.chi2_front.push(s.chi2_front);
        out.flags.push(s.flags);
        out.n_runs.push(s.n_runs);
        out.n_scans.push(s.n_scans);
        out.n_pairs.push(s.n_pairs);
    }
    Ok(out)
}

/// The tree's MAP state `[c1, c2, t1, t2, log R]` per row and each row's marginal covariance (see the module
/// documentation): factors on a row and its parent, eliminated leaves first.
#[allow(clippy::type_complexity)]
fn tree_solve(rows: &Rows, fr: &[Frame], sums: &[RowSummary], p: &PriorParams) -> Result<(Vec<Vector5<f64>>, Vec<Matrix5<f64>>)> {
    let n = rows.a.len();
    let (parent, h0, c0, g0) = prior_system(rows, fr, p);
    // the measurements, linearised in log R, three times
    let mut lin: Vec<f64> = (0..n).map(|k| sums[k].used().map_or(rows.r_prior[k].ln(), |c| c.x[4].ln())).collect();
    let mut sol = (vec![Vector5::zeros(); n], vec![Matrix5::zeros(); n]);
    for _ in 0..3 {
        let (mut h, mut g) = (h0.clone(), g0.clone());
        for k in 0..n {
            let Some(c) = sums[k].used() else { continue };
            // R = rl (1 + eta - log rl) near the linearisation point: x = J x' + off
            let rl = lin[k].exp();
            let mut jac = Matrix5::<f64>::identity();
            jac[(4, 4)] = rl;
            let mut off = Vector5::<f64>::zeros();
            off[4] = rl * (1.0 - lin[k]);
            let (info, hh) = (c.st.info / c.inflate(), c.st.h / c.inflate());
            h[k] += jac.transpose() * info * jac;
            g[k] += jac.transpose() * (hh - info * off);
        }
        sol = eliminate(&parent, h, &c0, g)?;
        for k in 0..n {
            lin[k] = sol.0[k][4];
        }
    }
    Ok(sol)
}

/// The prior's normal equations on the rows: each row's block, its coupling to its parent's row, and the
/// right-hand side (the prior is Gaussian in `[c1, c2, t1, t2, log R]`).
#[allow(clippy::type_complexity)]
fn prior_system(rows: &Rows, fr: &[Frame], p: &PriorParams) -> (Vec<Option<usize>>, Vec<Matrix5<f64>>, Vec<Matrix5<f64>>, Vec<Vector5<f64>>) {
    let n = rows.a.len();
    let parent: Vec<Option<usize>> = rows.parent.iter().map(|&q| (q >= 0).then_some(q as usize)).collect();
    let share = rows.fork_share(&fr.iter().map(|f| f.3).collect::<Vec<_>>(), ForkShare::WoodLength);
    let eta0: Vec<f64> = rows.r_prior.iter().map(|r| r.ln()).collect();
    // a factor: sum of coefficients times (row, unknown), its mean and sd
    type Factor = (Vec<(usize, usize, f64)>, f64, f64);
    let mut factors: Vec<Factor> = Vec::new();
    for k in 0..n {
        let sc = p.centre_rel * rows.r_prior[k] + p.centre_abs;
        factors.push((vec![(k, 0, 1.0)], 0.0, sc));
        factors.push((vec![(k, 1, 1.0)], 0.0, sc));
        factors.push((vec![(k, 2, 1.0)], 0.0, LEAN_SD));
        factors.push((vec![(k, 3, 1.0)], 0.0, LEAN_SD));
        let Some(q) = parent[k] else {
            factors.push((vec![(k, 4, 1.0)], eta0[k], p.sig_root));
            continue;
        };
        let len = fr[k].3;
        let ls = len.max(0.01).sqrt();
        if share[k].is_nan() {
            factors.push((vec![(k, 4, 1.0), (q, 4, -1.0)], eta0[k] - eta0[q], p.sig_eta * ls));
            // the child's start (its centre less half its length's lean) at the parent's end, in the child's frame
            let (ek, eq) = ([fr[k].1, fr[k].2], [fr[q].1, fr[q].2]);
            let lq = fr[q].3;
            for j in 0..2 {
                let tr = [dot(&ek[j], &eq[0]), dot(&ek[j], &eq[1])];
                factors.push((vec![(k, j, 1.0), (k, 2 + j, -0.5 * len), (q, 0, -tr[0]), (q, 1, -tr[1]), (q, 2, -0.5 * lq * tr[0]), (q, 3, -0.5 * lq * tr[1])], 0.0, p.sig_delta * ls));
            }
        } else {
            factors.push((vec![(k, 4, 1.0), (q, 4, -1.0)], 0.5 * (p.mu_fork + share[k].ln()), p.sig_fork));
        }
    }
    // the prior's precision (fixed), as row blocks and child-parent couplings
    let (mut h0, mut c0) = (vec![Matrix5::<f64>::zeros(); n], vec![Matrix5::<f64>::zeros(); n]);
    let mut g0 = vec![Vector5::<f64>::zeros(); n];
    for (vars, mean, sd) in &factors {
        let w = 1.0 / (sd * sd);
        let k = vars[0].0; // the factor's own row: a child's, where it links to its parent
        for &(ri, i, ci) in vars {
            g0[ri][i] += w * mean * ci;
            for &(rj, j, cj) in vars {
                if ri == rj {
                    h0[ri][(i, j)] += w * ci * cj;
                } else if ri == k {
                    c0[k][(i, j)] += w * ci * cj; // (child, parent); the other orientation is its transpose
                }
            }
        }
    }
    (parent, h0, c0, g0)
}

/// Solves `H x = g` on a tree of 5x5 row blocks (`c[k]`: row k's coupling to its parent's row), leaves first,
/// with each row's marginal covariance top-down (`Sigma_k = H_k^-1 + Y Sigma_q Y^T`). The prior's anchors make
/// every block positive definite: one that isn't is an error.
#[allow(clippy::type_complexity)]
fn eliminate(parent: &[Option<usize>], mut h: Vec<Matrix5<f64>>, c: &[Matrix5<f64>], mut g: Vec<Vector5<f64>>) -> Result<(Vec<Vector5<f64>>, Vec<Matrix5<f64>>)> {
    let n = h.len();
    let mut inv = vec![Matrix5::<f64>::zeros(); n];
    let mut y = vec![Matrix5::<f64>::zeros(); n];
    let mut yv = vec![Vector5::<f64>::zeros(); n];
    for k in (0..n).rev() {
        inv[k] = h[k].cholesky().ok_or_else(|| Error::invalid(format!("support tree: row {k}'s block is not positive definite")))?.inverse();
        yv[k] = inv[k] * g[k];
        if let Some(q) = parent[k] {
            y[k] = inv[k] * c[k];
            h[q] -= c[k].transpose() * y[k];
            g[q] -= c[k].transpose() * yv[k];
        }
    }
    let (mut x, mut sig) = (vec![Vector5::<f64>::zeros(); n], vec![Matrix5::<f64>::zeros(); n]);
    for k in 0..n {
        match parent[k] {
            Some(q) => {
                x[k] = yv[k] - y[k] * x[q];
                sig[k] = inv[k] + y[k] * sig[q] * y[k].transpose();
            }
            None => {
                x[k] = yv[k];
                sig[k] = inv[k];
            }
        }
    }
    Ok((x, sig))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qsm::joint::frames;
    use crate::qsm::joint_fit::PriorParams;

    fn like() -> JointParams {
        JointParams { tau: 0.17, edge: 0.25, hide: 0.0, ..JointParams::default() }
    }

    #[test]
    fn a_scanned_stem_is_measured_from_one_scan() {
        // the evidence tests' fan: a 5 cm stem at 5 m, claimed at 8 cm 1 cm off its axis
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let (a1, b1) = ([a[0] + 0.01, a[1], a[2]], [b[0] + 0.01, b[1], b[2]]);
        let rows = Rows { a: &[a1], b: &[b1], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let f = support_tree(&[&scan], &rows, &like(), &PriorParams::default(), 2).unwrap();
        assert_eq!(f.flags[0], 0, "flags {}", f.flags[0]);
        assert!(f.n_scans[0] == 1 && f.n_runs[0] == 1, "scans {} runs {}", f.n_scans[0], f.n_runs[0]);
        assert!((f.r[0] - 0.05).abs() < 0.0025, "R {}", f.r[0]);
        // (measured in seven pieces of the 2 m row: the middle is the claim moved back by 1 cm, the lean within its sd)
        assert!((0.5 * (f.a[0][0] + f.b[0][0])).abs() < 0.002 && f.lean[0][1].abs() < 2.0 * f.sd_lean[0][1], "x {} {} lean {:?}", f.a[0][0], f.b[0][0], f.lean[0]);
        assert!(f.sd_log_r[0] < 0.1 && (f.r_row[0] - f.r[0]).abs() < 0.002);
        // one scan: the front's stops still give a misfit
        assert!(f.chi2_front[0].is_finite() && f.chi2_front[0] < HULL_CHI2, "chi2_front {}", f.chi2_front[0]);
    }

    fn circle(x: Vector5<f64>, sd: f64) -> Circle {
        let info = Matrix5::<f64>::identity() / (sd * sd);
        Circle { st: Stats { info, h: info * x, yy: x.dot(&(info * x)), n: 5 }, x, cov: Matrix5::identity() * sd * sd, chi2: f64::NAN }
    }

    #[test]
    fn the_tree_solve_is_exact_and_the_prior_carries_unmeasured_rows() {
        // three rows in a chain and a fork; the middle row measured: the elimination equals the dense solve
        let rows_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let rows_b = [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.3, 0.0, 1.5], [-0.3, 0.0, 1.5]];
        let rp = [0.1, 0.08, 0.05, 0.04];
        let rows = Rows { a: &rows_a, b: &rows_b, r_prior: &rp, r_max: &[], parent: &[-1, 0, 1, 1] };
        let fr = frames(&rows_a, &rows_b);
        let c = circle(Vector5::new(0.01, -0.005, 0.02, 0.0, 0.06), 1e-3);
        let none = || RowSummary { circle: None, chi2_front: f64::NAN, flags: flag::FEW, n_runs: 0, n_scans: 0, n_pairs: 0 };
        // a flagged row's circle stays out of the tree
        let flagged = RowSummary { circle: Some(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.2), 1e-3)), chi2_front: f64::NAN, flags: flag::RUNS, n_runs: 2, n_scans: 1, n_pairs: 100 };
        let sums = vec![none(), RowSummary { circle: Some(c), chi2_front: f64::NAN, flags: 0, n_runs: 1, n_scans: 1, n_pairs: 100 }, flagged, none()];
        let (x, sig) = tree_solve(&rows, &fr, &sums, &PriorParams::default()).unwrap();
        // the measured row follows its measurement; its neighbours follow the taper from it
        assert!((x[1][4].exp() - 0.06).abs() < 0.001 && (x[1][0] - 0.01).abs() < 0.001, "{:?}", x[1]);
        // (the base between the taper's step from it, 7.5 cm, and its own anchor on the claim, 10 cm)
        assert!(x[0][4].exp() > 0.075 && x[0][4].exp() < 0.09, "{}", x[0][4].exp());
        assert!(sig[1][(4, 4)] < sig[0][(4, 4)] && sig[1][(4, 4)] < sig[2][(4, 4)]);
        // the prior alone leaves a flagged row's radius loose: the fork's spread, not a measurement's
        let pp = PriorParams::default();
        assert!(sig[2][(4, 4)].sqrt() > pp.sig_fork * 0.9, "sd {}", sig[2][(4, 4)].sqrt());
        // the forks' children split the parent's area by wood length (equal here), with mu_fork
        assert!((x[2][4] - x[3][4]).abs() < 1e-9);
        // the elimination is the dense solve of the same system, mean and marginal covariances
        let (parent, h, c, g) = prior_system(&rows, &fr, &PriorParams::default());
        let mut hd = nalgebra::DMatrix::<f64>::zeros(20, 20);
        let mut gd = nalgebra::DVector::<f64>::zeros(20);
        for k in 0..4 {
            hd.view_mut((5 * k, 5 * k), (5, 5)).copy_from(&h[k]);
            gd.rows_mut(5 * k, 5).copy_from(&g[k]);
            if let Some(q) = parent[k] {
                hd.view_mut((5 * k, 5 * q), (5, 5)).copy_from(&c[k]);
                hd.view_mut((5 * q, 5 * k), (5, 5)).copy_from(&c[k].transpose());
            }
        }
        let inv = hd.try_inverse().unwrap();
        let xd = &inv * &gd;
        let (xe, se) = eliminate(&parent, h, &c, g).unwrap();
        for k in 0..4 {
            for i in 0..5 {
                assert!((xe[k][i] - xd[5 * k + i]).abs() < 1e-9, "x[{k}][{i}]");
                for j in 0..5 {
                    assert!((se[k][(i, j)] - inv[(5 * k + i, 5 * k + j)]).abs() < 1e-9 * (1.0 + inv[(5 * k + i, 5 * k + j)].abs()), "cov[{k}]");
                }
            }
        }
        // a block that isn't positive definite is an error, not a silent zero
        let bad = vec![Matrix5::<f64>::zeros()];
        assert!(eliminate(&[None], bad, &[Matrix5::zeros()], vec![Vector5::zeros()]).is_err());
    }

    #[test]
    fn pieces_combine_exactly_and_misfit_counts_on_the_rank() {
        // two pieces of a 0.6 m row, each measuring a circle at its own middle: together the row's centre and lean
        let (c0, t, r) = ([0.01, -0.02], [0.05, 0.0], 0.03);
        let mut st: Option<Stats> = None;
        for s in [-0.15, 0.15] {
            let x = Vector5::new(c0[0] + s * t[0], c0[1] + s * t[1], t[0], t[1], r);
            let m = circle(x, 1e-3).st.moved(s);
            match st.as_mut() {
                Some(a) => a.add(&m),
                None => st = Some(m),
            }
        }
        let c = solve_row(&st.unwrap(), 0.05).unwrap();
        let want = Vector5::new(c0[0], c0[1], t[0], t[1], r);
        assert!((c.x - want).norm() < 1e-4, "{:?}", c.x);
        assert!(c.chi2 < 1e-3, "chi2 {}", c.chi2);
        // a measurement set of rank 2 (two edges): no redundancy, no misfit
        let n = [1.0, 0.0];
        let edges: Vec<Meas> = vec![([n[0], n[1], 0.0, 0.0, 1.0], 0.05, 1e-6), ([n[0], n[1], 0.0, 0.0, -1.0], -0.03, 1e-6), ([n[0], n[1], 0.0, 0.0, 1.0], 0.052, 1e-6)];
        let st = Stats::of(&edges);
        let ev = SymmetricEigen::new(st.info).eigenvalues;
        assert_eq!(ev.iter().filter(|&&e| e > 1e-9 * ev.max()).count(), 2);
        let c = solve_row(&st, 0.05).unwrap();
        assert!(c.chi2.is_finite() && c.chi2 > 0.5, "one redundant edge: chi2 {}", c.chi2);
    }

    #[test]
    fn echo_character_weighs_stops() {
        let p = JointParams::default();
        let rec = |refl: f64, single: bool| Rec { b: 0.0, s: 0.0, depth: 0.0, w: 0.01, refl, single, blocked: false };
        // bark-bright is wood, leaf-dark is leaf, unknown is wood; a multi-return echo is judged darker
        assert!(p_wood(&rec(-15.5, true), 0.05, &p) > 0.9);
        assert!(p_wood(&rec(-21.0, true), 0.05, &p) < 0.1);
        assert_eq!(p_wood(&rec(f64::NAN, true), 0.05, &p), 1.0);
        assert!(p_wood(&rec(-18.5, false), 0.05, &p) > 0.5 && p_wood(&rec(-18.5, true), 0.05, &p) < 0.5);
        // wood narrower than the footprint reads darker: a dark stop is likelier wood on a 1 mm twig
        assert!(p_wood(&rec(-19.0, true), 0.003, &p) > p_wood(&rec(-19.0, true), 0.05, &p));
    }

    #[test]
    fn inputs_are_checked() {
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let bad = Rows { a: &[a], b: &[b], r_prior: &[0.08], r_max: &[], parent: &[0] };
        assert!(support_tree(&[&scan], &bad, &like(), &PriorParams::default(), 1).is_err());
        let flat = Rows { a: &[a], b: &[a], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        assert!(support_tree(&[&scan], &flat, &like(), &PriorParams::default(), 1).is_err());
    }
}
