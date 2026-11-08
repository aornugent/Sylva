// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One tree's wood from the support function of each row's cross-section, as
//! each scan measures it, and one solve over the tree (3dtrees research 93
//! §10, §16, §17): the fast path of the joint model, with flags that send the
//! pieces it can't summarise (several stems, a hull) to
//! [`crate::qsm::joint_tree`].
//!
//! Within one (piece, scan) group the pulses are nearly parallel, so they
//! reduce to a profile across the row: impact `b` (along the group's `n`,
//! across the beam) and, where a pulse stopped, depth (along `v`, the beam's
//! direction in the cross-section). For opaque wood that profile measures the
//! cross-section's support function, linear in a circle's centre `c` and
//! radius `R`: the silhouette's edges `n.c ± R_app`, `R_app = R + w (1/2 -
//! tau)` (w the footprint's width), and the front's depth `v.c - R`. The
//! piece may lean: `c(s) = c0 + s t`, `s` along the axis from its middle.
//!
//! - **Pieces.** A row longer than [`PIECE_LEN`] is cut into equal pieces no
//!   longer than that; each piece is a node of the tree, its parent the piece
//!   before it (a row's first piece's, the last of its parent row's), so a
//!   bent row is a chain of straight pieces held together by the prior.
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
//!   is the piece's, and the lean is taken again from its own hits and the
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
//! - **Per piece,** the measurements' information about `(c0, t, R)`, which
//!   may be rank-deficient (a one-sided piece's lean unseen): the tree's prior
//!   fills what it lacks. Its estimate, with a weak anchor of the centre and
//!   lean, is inflated by the misfit (reduced chi-square on the rank).
//! - **Where the circle isn't determined** (thin wood, one side seen, no
//!   opaque run), each group still bounds R (research 93 §17): its woody stops
//!   at the claim's depth, about the one nearest the axis, span no more than
//!   the silhouette (a lower bound, `R >= span / 2 - w (1/2 - tau)`), the
//!   nearest passes either side of them along their stretch of the axis lie
//!   outside it (an upper bound, `R <= gap / 2 - w (1/2 - tau)`; nothing
//!   recorded beyond an edge is sky, a stop or blocked pulse there with no
//!   pass censors it), and wood that was hit fills at least `tau` of the
//!   footprint (the capacity limit `R >= tau w / 2`). Each is a probit term on
//!   log R of the edge's sd.
//! - **Existence.** Each group's stops and passes aimed at the core (its
//!   woody cluster's span, or the claim's middle half where none) give a log
//!   odds of wood against foliage: each stop's echo character and, per pulse,
//!   opacity (wood stops [`STOP_WOOD`] of them, foliage [`STOP_LEAF`]),
//!   divided by the beam's oversampling `(w / spacing)^2` (research 93 §15.6).
//!   A chain along the branches (wood doesn't stop and restart) turns the
//!   pieces' log odds into `p_wood`, exactly by sum-product.
//! - **Flags:** [`flag::RUNS`] (a group's profile, its stops gated by the
//!   fitted circle's depth band, shows two or more stems within the claim),
//!   [`flag::HULL`] (the support is wider than one circle explains within the
//!   piece: more than a share of the stops aimed at the fitted circle's core
//!   lie behind its front, the measurements or the front's stops misfit it,
//!   or the run is far wider than it), [`flag::BOUNDS`] (R isn't determined:
//!   bounds), [`flag::POROUS`] (pulses but no opaque run: existence and
//!   bounds). Only RUNS and HULL are for the exact likelihood; their radius
//!   is the prior's.
//! - **The tree:** the unflagged pieces' summaries as Gaussian measurements
//!   of `(c0, t, R)`, the bounds of BOUNDS and POROUS pieces as censored
//!   measurements of log R, and a Gaussian Markov prior over the pieces after
//!   the joint fit's [`PriorParams`] (the base's level, the taper's steps, at
//!   a fork the child's share of the subtree wood length rather than of the
//!   prior area, the centre's and lean's continuity along a branch, the
//!   centre's anchor on the claim and a weak anchor of the lean; no tip
//!   anchor). Every term links a piece only to its parent, so each Gaussian
//!   system is solved exactly by eliminating pieces into their parents, leaves
//!   first (as [`crate::qsm::joint_fit`]'s branch blocks, which without the
//!   per-scan offsets are single rows), with each piece's marginal covariance.
//!   The radius is measured in R and modelled in log R (Gauss-Newton), and the
//!   censored terms enter by expectation propagation: each a Gaussian site on
//!   its piece's log R matched to the moments of its probit times the cavity
//!   (exact per term), all sites updated from the marginals of one solve and
//!   damped, until nothing moves. The posterior is then the prior cut by the
//!   bounds, mean and sd. A row's output is its pieces': the line from the
//!   first's start to the last's end and the mean of their log R, with the
//!   covariance of the chain.

use nalgebra::{Matrix5, SymmetricEigen, Vector5};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::evidence::ScanIndex;
use crate::qsm::joint::{pool, window, JointParams};
use crate::qsm::joint_fit::{ForkShare, Frame, PriorParams, Rows};
use crate::transform::{cross, dot, sub};
use crate::Point;

/// Why a piece's own summary is not used (bits of [`Fit::flags`]). Only [`RUNS`] and [`HULL`] are for the exact
/// likelihood; the others are resolved here.
pub mod flag {
    /// A group's profile holds two or more stems within the claim.
    pub const RUNS: u8 = 1;
    /// The support is wider than one circle explains within a piece: several stems or a non-round section.
    pub const HULL: u8 = 2;
    /// The circle isn't determined: the radius is the prior cut by the bounds.
    pub const BOUNDS: u8 = 4;
    /// Pulses, but no opaque run: existence (`p_wood`), and the radius the prior cut by any bounds.
    pub const POROUS: u8 = 8;
}

/// A group needs this much (wood-weighted) pulses that say something (hits and passes).
const MIN_PULSES: f64 = 30.0;
/// A run's least wood-weighted hits, to be the row's or to count as a stem beside it.
const RUN_HITS: f64 = 10.0;
/// The tube's widening for the lean, a share of the piece's length (a lean of up to about 27°).
const LEAN_TUBE: f64 = 0.25;
/// Rows longer than this (m) are cut into pieces of at most this length.
pub const PIECE_LEN: f64 = 0.3;
/// The lean's anchor (m per m), and the floor on each measurement's sd (m): non-round sections, a ~4 mm offset
/// between scans that no sway explains (research 93 §15.6; at 2 mm Alice Mulga's clean rows misfit by a reduced
/// chi-square of 4-9).
const LEAN_SD: f64 = 0.5;
const MODEL_SD: f64 = 0.004;
/// The lean's bend between a row's pieces, per square root of m (research 93 §13.1: rows of 67-98 cm whose centres
/// drift 8-24 cm along them).
const LEAN_BEND: f64 = 0.3;
/// The narrowest bin (m), the depth jump that separates two stems in a run (m), and how far a stop may sit from
/// a run's depths or a circle's front and still be on it (`DEPTH_TOL + 0.3 R`).
const BIN_MIN: f64 = 0.002;
const DEPTH_JUMP: f64 = 0.02;
const DEPTH_TOL: f64 = 0.02;
/// A stop aimed at a circle's core is "behind" it this far behind its front (`HULL_BEHIND + 0.1 R`); a piece is a
/// hull when more than `HULL_SHARE` of them are, or a misfit (reduced chi-square) exceeds `HULL_CHI2`.
const HULL_BEHIND: f64 = 0.01;
const HULL_SHARE: f64 = 0.25;
const HULL_CHI2: f64 = 9.0;
/// The bound on a stop's log likelihood ratio of wood to leaf (research 93 §11.2).
const LLR_CLIP: f64 = 3.0;
/// The share of the pulses aimed at a core that stop there: on wood (opaque but for the soft edge) and on foliage
/// (research 93 §11.1: synthetic foliage lets 65 % through).
const STOP_WOOD: f64 = 0.9;
const STOP_LEAF: f64 = 0.35;
/// Existence along a branch: each piece is wood a priori with `EXIST` (about `joint_tree`'s share of RCT's rows on
/// Alice Mulga tree 849), one after none with `EXIST_OFF` (wood doesn't stop and restart), and one after wood with
/// `EXIST_ON`, which keeps every piece's prior at `EXIST` however far out it is.
const EXIST: f64 = 0.8;
const EXIST_OFF: f64 = 0.02;
const EXIST_ON: f64 = 1.0 - (1.0 - EXIST) * EXIST_OFF / EXIST;
/// The capacity limit's sd in log R (`tau` itself is a calibration).
const CAP_SD: f64 = 0.3;
/// The censored solve: at most `EP_ROUNDS` rounds, each site moved `EP_DAMP` of the way, until no site or
/// linearisation moves by more than `EP_TOL` (in units of its piece's sd).
const EP_ROUNDS: usize = 100;
const EP_DAMP: f64 = 0.5;
const EP_TOL: f64 = 1e-4;
const EP_NODES: usize = 64;

/// One tree's fit, per row or per piece (the tree's nodes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fit {
    /// The fitted axis (world coordinates: the claim moved by the centre and turned by the lean) and radius, with
    /// the posterior sd of log R. Where a piece is RUNS or HULL these are the prior's, given the others.
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r: Vec<f64>,
    pub sd_log_r: Vec<f64>,
    /// The centre at the middle and the lean (per m along the axis), along `e1, e2`, with their sds.
    pub centre: Vec<[f64; 2]>,
    pub sd_centre: Vec<[f64; 2]>,
    pub lean: Vec<[f64; 2]>,
    pub sd_lean: Vec<[f64; 2]>,
    pub e1: Vec<Point>,
    pub e2: Vec<Point>,
    /// The own summary's radius and sd, NaN where it is flagged or has none (a row's: the geometric mean of its
    /// measured pieces').
    pub r_row: Vec<f64>,
    pub sd_r_row: Vec<f64>,
    /// The misfits, NaN where there are none (a row's: its pieces' largest): of the measurements to the circle
    /// (reduced chi-square on their rank), and of the front's stops about it against the range noise, roughness
    /// and the floor.
    pub chi2: Vec<f64>,
    pub chi2_front: Vec<f64>,
    /// The bounds on R, the tightest lower and upper, NaN where there are none (a row's: its pieces' widest).
    pub r_lo: Vec<f64>,
    pub r_hi: Vec<f64>,
    /// The existence evidence (log odds of wood against foliage; a row's: its pieces' sum) and the chance of wood
    /// under the branches' chain (a row's: its pieces' mean, the share of it that is wood).
    pub log_odds: Vec<f64>,
    pub p_wood: Vec<f64>,
    /// [`flag`] bits (a row's: its pieces'): the summary entered the tree only where 0.
    pub flags: Vec<u8>,
    /// The most runs (stems within the claim) any group shows, the groups that gave measurements, and the pairs
    /// (a row's: the most, the most, the sum).
    pub n_runs: Vec<u32>,
    pub n_scans: Vec<u32>,
    pub n_pairs: Vec<u32>,
}

/// The fit of one tree: per row (in the input's order) and per piece, with each piece's row.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportTree {
    pub rows: Fit,
    pub pieces: Fit,
    pub piece_row: Vec<u32>,
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
}

/// A piece's circle `[c1, c2, t1, t2, R]` (centre at the middle, lean, radius): its measurements' statistics, the
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

/// A censored measurement of a piece's log R, `log R <= eta` (upper) or `>= eta`, softened by a Gaussian of sd
/// `sd`: the probit `Phi(±(eta - log R) / sd)`, log-concave.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bound {
    eta: f64,
    sd: f64,
    upper: bool,
}

/// What a piece's own pulses say: its circle and front misfit, flags, runs, scans and pairs, and its bounds and
/// existence evidence (log odds).
struct Summary {
    circle: Option<Circle>,
    chi2_front: f64,
    flags: u8,
    n_runs: u32,
    n_scans: u32,
    n_pairs: u32,
    bounds: Vec<Bound>,
    log_odds: f64,
}

impl Summary {
    /// The circle, where the piece's summary enters the tree.
    fn used(&self) -> Option<&Circle> {
        if self.flags == 0 { self.circle.as_ref() } else { None }
    }

    /// The bounds that enter the tree: a piece with no circle, not routed.
    fn censored(&self) -> &[Bound] {
        if self.flags & (flag::BOUNDS | flag::POROUS) != 0 && self.flags & (flag::RUNS | flag::HULL) == 0 { &self.bounds } else { &[] }
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

/// A stop's log likelihood ratio of wood of radius `r` to leaf by its reflectance and return position (research 93
/// §11.2), clipped to ±[`LLR_CLIP`]: wood narrower than the footprint `w` fills only `2r/w` of it and reads that
/// much darker. An echo of unknown reflectance: 0.
fn llr(rec: &Rec, r: f64, p: &JointParams) -> f64 {
    if !rec.refl.is_finite() {
        return 0.0;
    }
    let (shift, var) = if rec.single { (0.0, 0.0) } else { (p.refl_multi, p.refl_multi_sd * p.refl_multi_sd) };
    let thin = 10.0 * (2.0 * r / rec.w).clamp(1e-3, 1.0).log10();
    let ll = |m: f64, sd: f64| {
        let s = (sd * sd + var).sqrt();
        -0.5 * ((rec.refl - m) / s).powi(2) - s.ln()
    };
    (ll(p.refl_wood + shift + thin, p.refl_wood_sd) - ll(p.refl_leaf + shift, p.refl_leaf_sd)).clamp(-LLR_CLIP, LLR_CLIP)
}

/// The weight of a stop as a hit on wood of radius `r`: the chance its echo came from wood rather than leaf at even
/// odds ([`llr`]), over that chance at the clip, so an echo as surely wood as the clip allows counts fully (a
/// uniform discount of every hit would pull each edge inwards). An echo of unknown reflectance: 1.
fn p_wood(rec: &Rec, r: f64, p: &JointParams) -> f64 {
    if !rec.refl.is_finite() {
        return 1.0;
    }
    (1.0 + (-LLR_CLIP).exp()) / (1.0 + (-llr(rec, r, p)).exp())
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

/// What a group says beyond its circle (research 93 §17): its bounds on R, its existence log odds, and the
/// footprint of its woody stops (0: none) for the capacity limit. See the module documentation.
///
/// The bounds are taken per slice across the piece, one pulse spacing along it: there the woody stops contiguous in
/// impact about the group's woody centre are one run, at least as wide as the silhouette's detection region
/// (`2 R_app`), and the passes just outside it, where both are passes, at most as far apart. Each slice's numbers
/// are valid bounds unless clutter joins the run or an echo was lost inside it, so the group's are their medians
/// (looser than the tightest slice, robust to a minority of corrupted ones).
fn censored(g: &Group, r_claim: f64, p: &JointParams) -> (Vec<Bound>, f64, f64) {
    let (d0, d1) = (-2.0 * r_claim - DEPTH_TOL, r_claim + DEPTH_TOL);
    // at the claim's depth a hit; behind it (or no stop in the tube) a pass; in front of it blocked
    let fates: Vec<Fate> = g.recs.iter().map(|r| {
        if r.blocked || r.depth < d0 {
            Fate::Blocked
        } else if !r.depth.is_finite() || r.depth > d1 {
            Fate::Pass
        } else {
            Fate::Hit
        }
    }).collect();
    let n = g.recs.len();
    let woody: Vec<bool> = (0..n).map(|i| fates[i] == Fate::Hit && p_wood(&g.recs[i], r_claim, p) >= 0.5).collect();
    let w = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
    let app = p.app(w);
    let near: Vec<usize> = (0..n).filter(|&i| woody[i] && g.recs[i].b.abs() < 1.5 * r_claim + 0.01).collect();
    let k = lean_line(&mut near.iter().map(|&i| (g.recs[i].s, g.recs[i].b)).collect::<Vec<_>>()).map_or(0.0, |l| l.0);
    let bp: Vec<f64> = g.recs.iter().map(|r| r.b - k * r.s).collect();
    // the pulses' spacing, from their density over the patch they cover (impact by axial position)
    let inf: Vec<usize> = (0..n).filter(|&i| fates[i] != Fate::Blocked).collect();
    let span = |v: &mut dyn Iterator<Item = f64>| v.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
    let (b0, b1) = span(&mut inf.iter().map(|&i| bp[i]));
    let (t0, t1) = span(&mut inf.iter().map(|&i| g.recs[i].s));
    let step = if inf.len() > 1 { ((b1 - b0).max(BIN_MIN) * (t1 - t0).max(BIN_MIN) / inf.len() as f64).sqrt() } else { w };
    // each slice's run of `member` pulses (contiguous in impact) about the one nearest `c`: its half-width, and
    // where both its sides are passes (a side with no pulse beyond it is sky: half a spacing out) half their gap
    let mut order: Vec<usize> = (0..n).collect();
    let slice = |i: usize| ((g.recs[i].s - t0) / step).floor() as i64;
    order.sort_unstable_by(|&i, &j| slice(i).cmp(&slice(j)).then(bp[i].total_cmp(&bp[j])));
    let runs = |member: &[bool], c: f64| {
        let (mut lo_s, mut hi_s) = (Vec::new(), Vec::new());
        for sl in order.chunk_by(|&i, &j| slice(i) == slice(j)) {
            let Some(j0) = (0..sl.len()).filter(|&j| member[sl[j]] && (bp[sl[j]] - c).abs() < r_claim + w).min_by(|&x, &y| (bp[sl[x]] - c).abs().total_cmp(&(bp[sl[y]] - c).abs())) else { continue };
            let (mut a, mut z) = (j0, j0);
            while a > 0 && member[sl[a - 1]] {
                a -= 1;
            }
            while z + 1 < sl.len() && member[sl[z + 1]] {
                z += 1;
            }
            lo_s.push(0.5 * (bp[sl[z]] - bp[sl[a]]));
            let left = if a == 0 { Some(bp[sl[a]] - 0.5 * step) } else { (fates[sl[a - 1]] == Fate::Pass).then(|| bp[sl[a - 1]]) };
            let right = if z + 1 == sl.len() { Some(bp[sl[z]] + 0.5 * step) } else { (fates[sl[z + 1]] == Fate::Pass).then(|| bp[sl[z + 1]]) };
            if let (Some(l), Some(r)) = (left, right) {
                hi_s.push(0.5 * (r - l));
            }
        }
        (lo_s, hi_s)
    };
    let centre = |member: &[bool]| {
        let mut v: Vec<f64> = (0..n).filter(|&i| member[i] && g.recs[i].b.abs() < 1.5 * r_claim + 0.01).map(|i| bp[i]).collect();
        if v.is_empty() { 0.0 } else { median(&mut v) }
    };
    let (mut lo_s, mut hi_s) = runs(&woody, centre(&woody));
    let mut bounds = Vec::new();
    let cap = 0.5 * p.tau * w;
    let sd = p.edge * w / std::f64::consts::SQRT_2;
    if lo_s.len() >= 3 {
        let lo_r = median(&mut lo_s) - app;
        if lo_r > cap {
            bounds.push(Bound { eta: lo_r.ln(), sd: sd / lo_r, upper: false });
        }
    }
    if hi_s.len() >= 3 {
        let hi_r = (median(&mut hi_s) - app).max(cap);
        bounds.push(Bound { eta: hi_r.ln(), sd: (sd * sd + step * step / 12.0).sqrt() / hi_r, upper: true });
    }
    // existence, over the oversampling: the core's stops by their echo character, and its pulses by opacity, which
    // only counts for wood (research 93 §11.1: opaque from one scan is reliable, leaky is no proof of foliage: the
    // core may be misplaced). The core is found from the runs of any stop, blind to echo character (a core of
    // woody runs would hold woody echoes by construction), else it is the claim's middle half.
    let hit: Vec<bool> = fates.iter().map(|&f| f == Fate::Hit).collect();
    let ch = centre(&hit);
    let (mut lo_e, _) = runs(&hit, ch);
    let (core, half) = if lo_e.is_empty() { (0.0, 0.5 * r_claim) } else { (ch, median(&mut lo_e).max(0.25 * w)) };
    let (on, off) = ((STOP_WOOD / STOP_LEAF).ln(), ((1.0 - STOP_WOOD) / (1.0 - STOP_LEAF)).ln());
    let (mut echo, mut opacity) = (0.0, 0.0);
    for i in (0..n).filter(|&i| (bp[i] - core).abs() <= half) {
        match fates[i] {
            Fate::Hit => (echo, opacity) = (echo + llr(&g.recs[i], r_claim, p), opacity + on),
            Fate::Pass => opacity += off,
            Fate::Blocked => {}
        }
    }
    // a scan's verdict on a piece counts at most as one sure echo: beyond the beam's oversampling, foliage is
    // dispersed at 10-40 cm (research 93 §15.6), the piece's scale, so its pulses there are one observation
    let m = (w / step).powi(2).max(1.0);
    let lam = ((echo + opacity.max(0.0)) / m).clamp(-LLR_CLIP, LLR_CLIP);
    (bounds, lam, if near.is_empty() { 0.0 } else { w })
}

/// What one piece's pulses say: its circle (from its second pass), flags, runs, scans and pairs, its front's
/// stops' misfit about the circle, and its bounds and existence evidence.
fn summarise_piece(scans: &[&ScanIndex], a: &Point, b: &Point, frame: Frame, r_claim: f64, p: &JointParams) -> Summary {
    let tube = window(r_claim) + LEAN_TUBE * frame.3;
    let gs = groups(scans, a, b, frame, tube, p);
    let n_pairs = gs.iter().map(|g| g.recs.len()).sum::<usize>() as u32;
    let mut out = Summary { circle: None, chi2_front: f64::NAN, flags: 0, n_runs: 0, n_scans: 0, n_pairs, bounds: Vec::new(), log_odds: 0.0 };
    let mut cap = 0.0f64;
    for g in &gs {
        let (bs, lam, w) = censored(g, r_claim, p);
        out.bounds.extend(bs);
        out.log_odds += lam;
        cap = cap.max(w);
    }
    if cap > 0.0 {
        out.bounds.push(Bound { eta: (0.5 * p.tau * cap).ln(), sd: CAP_SD, upper: false });
    }
    // the first pass: each stop by its echo
    let first: Vec<GroupOut> = gs.iter().map(|g| {
        let f: Vec<Fate> = g.recs.iter().map(|r| fate(r, g.n, g.v, None)).collect();
        measure(g, &f, &weights(g, &f, r_claim, p), 0.0, r_claim, (-2.0 * r_claim - DEPTH_TOL, r_claim + DEPTH_TOL), p)
    }).collect();
    if first.iter().any(|o| o.enough) && !first.iter().any(|o| o.run) {
        out.flags |= flag::POROUS;
        return out;
    }
    let Some(c1) = solve_row(&Stats::of(&first.iter().flat_map(|o| o.meas.clone()).collect::<Vec<_>>()), r_claim) else {
        out.flags |= flag::BOUNDS;
        return out;
    };
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
    let Some(c2) = solve_row(&Stats::of(&meas), r_claim) else {
        out.flags |= flag::BOUNDS;
        return out;
    };
    let y = c2.x;
    let rr = y[4];
    // the stems beside it and the run's width, its stops gated by its depth band
    let runs: Vec<(usize, bool)> = gs.iter().map(|g| runs_of(g, &y, r_claim, p)).collect();
    out.n_runs = runs.iter().map(|q| q.0 as u32).max().unwrap_or(0);
    if out.n_runs >= 2 {
        out.flags |= flag::RUNS;
    }
    if runs.iter().any(|q| q.1) {
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
    if nw - ng as f64 > 0.0 {
        out.chi2_front = ss / (nw - ng as f64);
    }
    if c2.chi2 > HULL_CHI2 || out.chi2_front > HULL_CHI2 {
        out.flags |= flag::HULL;
    }
    out.circle = Some(c2);
    out
}

/// The tree's nodes, the rows' pieces: each row cut into `ceil(len / PIECE_LEN)` equal pieces, in order, a row's
/// first piece's parent the last of its parent row's and each later piece's the one before it.
struct Nodes {
    row: Vec<usize>,
    /// Each row's first piece, and one past the last row's last.
    first: Vec<usize>,
    parent: Vec<Option<usize>>,
    frame: Vec<Frame>,
    a: Vec<Point>,
}

impl Nodes {
    fn of(rows: &Rows, fr: &[Frame]) -> Self {
        let mut nd = Nodes { row: Vec::new(), first: Vec::new(), parent: Vec::new(), frame: Vec::new(), a: Vec::new() };
        for (k, &(u, e1, e2, len)) in fr.iter().enumerate() {
            let m = (len / PIECE_LEN - 1e-9).ceil().max(1.0) as usize;
            let pl = len / m as f64;
            nd.first.push(nd.row.len());
            for j in 0..m {
                nd.parent.push(match (j, rows.parent[k]) {
                    (0, q) if q >= 0 => Some(nd.first[q as usize + 1] - 1),
                    (0, _) => None,
                    _ => Some(nd.row.len() - 1),
                });
                nd.row.push(k);
                nd.frame.push((u, e1, e2, pl));
                nd.a.push(std::array::from_fn(|i| rows.a[k][i] + j as f64 * pl * u[i]));
            }
        }
        nd.first.push(nd.row.len());
        nd
    }
}

/// Fits one tree from its rows' support summaries: `rows` (with `r_max` ignored) against `scans`, the prior's
/// widths from `prior` and the footprint, detection threshold, range noise and echo character from `like`. Runs on
/// `threads` threads (0: one per core).
pub fn support_tree(scans: &[&ScanIndex], rows: &Rows, like: &JointParams, prior: &PriorParams, threads: usize) -> Result<SupportTree> {
    let fr = rows.check("support tree")?;
    prior.check()?;
    let nd = Nodes::of(rows, &fr);
    let sums: Vec<Summary> = pool(threads)?.install(|| {
        (0..nd.row.len()).into_par_iter().map(|k| {
            let (u, _, _, pl) = nd.frame[k];
            let b: Point = std::array::from_fn(|i| nd.a[k][i] + pl * u[i]);
            summarise_piece(scans, &nd.a[k], &b, nd.frame[k], rows.r_prior[nd.row[k]], like)
        }).collect()
    });
    let sol = tree_solve(rows, &fr, &nd, &sums, prior)?;
    let p_wood = existence(&nd.parent, &sums.iter().map(|s| s.log_odds).collect::<Vec<_>>());

    let mut out = SupportTree { rows: Fit::default(), pieces: Fit::default(), piece_row: nd.row.iter().map(|&k| k as u32).collect() };
    let bound = |s: &Summary, upper: bool| {
        let it = s.censored().iter().filter(|b| b.upper == upper).map(|b| b.eta);
        let e = if upper { it.fold(f64::INFINITY, f64::min) } else { it.fold(f64::NEG_INFINITY, f64::max) };
        if e.is_finite() { e.exp() } else { f64::NAN }
    };
    for (k, s) in sums.iter().enumerate() {
        let own = Own {
            r_row: s.used().map_or(f64::NAN, |c| c.x[4]),
            sd_r_row: s.used().map_or(f64::NAN, |c| c.cov[(4, 4)].max(0.0).sqrt()),
            chi2: s.circle.map_or(f64::NAN, |c| c.chi2),
            chi2_front: s.chi2_front,
            r_lo: bound(s, false),
            r_hi: bound(s, true),
            log_odds: s.log_odds,
            p_wood: p_wood[k],
            flags: s.flags,
            n_runs: s.n_runs,
            n_scans: s.n_scans,
            n_pairs: s.n_pairs,
        };
        out.pieces.push(&sol.x[k], &sol.cov[k], nd.frame[k], nd.a[k], own);
    }
    for k in 0..rows.a.len() {
        let (k0, k1) = (nd.first[k], nd.first[k + 1]);
        let (x, cov) = row_line(&sol, k0, k1, fr[k].3, nd.frame[k0].3);
        let p = &out.pieces;
        let fold = |v: &[f64], f: fn(f64, f64) -> f64| v[k0..k1].iter().copied().filter(|x| x.is_finite()).reduce(f).unwrap_or(f64::NAN);
        // the measured pieces' own radii, independent: the geometric mean and its sd
        let used: Vec<(f64, f64)> = (k0..k1).filter(|&j| p.r_row[j].is_finite()).map(|j| (p.r_row[j].ln(), p.sd_r_row[j] / p.r_row[j])).collect();
        let nu = used.len() as f64;
        let eta = used.iter().map(|q| q.0).sum::<f64>() / nu;
        let sd = used.iter().map(|q| q.1 * q.1).sum::<f64>().sqrt() / nu;
        let own = Own {
            r_row: eta.exp(),
            sd_r_row: eta.exp() * sd,
            chi2: fold(&p.chi2, f64::max),
            chi2_front: fold(&p.chi2_front, f64::max),
            r_lo: fold(&p.r_lo, f64::min),
            r_hi: fold(&p.r_hi, f64::max),
            log_odds: p.log_odds[k0..k1].iter().sum(),
            p_wood: p.p_wood[k0..k1].iter().sum::<f64>() / (k1 - k0) as f64,
            flags: p.flags[k0..k1].iter().fold(0, |a, &f| a | f),
            n_runs: p.n_runs[k0..k1].iter().copied().max().unwrap_or(0),
            n_scans: p.n_scans[k0..k1].iter().copied().max().unwrap_or(0),
            n_pairs: p.n_pairs[k0..k1].iter().sum(),
        };
        out.rows.push(&x, &cov, fr[k], rows.a[k], own);
    }
    Ok(out)
}

/// A piece's or row's own outputs (see [`Fit`]).
struct Own {
    r_row: f64,
    sd_r_row: f64,
    chi2: f64,
    chi2_front: f64,
    r_lo: f64,
    r_hi: f64,
    log_odds: f64,
    p_wood: f64,
    flags: u8,
    n_runs: u32,
    n_scans: u32,
    n_pairs: u32,
}

impl Fit {
    /// Appends a piece or row: its state `[c1, c2, t1, t2, log R]` and covariance, frame and claimed start `a0`.
    fn push(&mut self, x: &Vector5<f64>, cov: &Matrix5<f64>, (u, e1, e2, len): Frame, a0: Point, own: Own) {
        let at = |s: f64| -> Point { std::array::from_fn(|i| a0[i] + (s + 0.5 * len) * u[i] + (x[0] + s * x[2]) * e1[i] + (x[1] + s * x[3]) * e2[i]) };
        self.a.push(at(-0.5 * len));
        self.b.push(at(0.5 * len));
        self.r.push(x[4].exp());
        let sd = |i: usize| cov[(i, i)].max(0.0).sqrt();
        self.sd_log_r.push(sd(4));
        self.centre.push([x[0], x[1]]);
        self.sd_centre.push([sd(0), sd(1)]);
        self.lean.push([x[2], x[3]]);
        self.sd_lean.push([sd(2), sd(3)]);
        self.e1.push(e1);
        self.e2.push(e2);
        self.r_row.push(own.r_row);
        self.sd_r_row.push(own.sd_r_row);
        self.chi2.push(own.chi2);
        self.chi2_front.push(own.chi2_front);
        self.r_lo.push(own.r_lo);
        self.r_hi.push(own.r_hi);
        self.log_odds.push(own.log_odds);
        self.p_wood.push(own.p_wood);
        self.flags.push(own.flags);
        self.n_runs.push(own.n_runs);
        self.n_scans.push(own.n_scans);
        self.n_pairs.push(own.n_pairs);
    }
}

/// A row's state from its pieces `k0..k1` (a chain, each `pl` long, the row `len`): the line from the first's start
/// to the last's end as the centre at the row's middle and the lean, and the mean of their log R; with its
/// covariance from the chain's joint covariance (`Cov(x_j, x_i) = -Y_j Cov(x_{j-1}, x_i)` below the diagonal).
fn row_line(sol: &Solved, k0: usize, k1: usize, len: f64, pl: f64) -> (Vector5<f64>, Matrix5<f64>) {
    let m = k1 - k0;
    let mut a = vec![Matrix5::<f64>::zeros(); m];
    for i in 0..2 {
        // start = c_0 - pl/2 t_0, end = c_last + pl/2 t_last
        a[0][(i, i)] += 0.5;
        a[0][(i, 2 + i)] -= 0.25 * pl;
        a[m - 1][(i, i)] += 0.5;
        a[m - 1][(i, 2 + i)] += 0.25 * pl;
        a[0][(2 + i, i)] -= 1.0 / len;
        a[0][(2 + i, 2 + i)] += 0.5 * pl / len;
        a[m - 1][(2 + i, i)] += 1.0 / len;
        a[m - 1][(2 + i, 2 + i)] += 0.5 * pl / len;
    }
    for aj in a.iter_mut() {
        aj[(4, 4)] = 1.0 / m as f64;
    }
    let x = (0..m).map(|j| a[j] * sol.x[k0 + j]).sum();
    let mut c: Vec<Vec<Matrix5<f64>>> = vec![Vec::with_capacity(m); m];
    let mut cov = Matrix5::zeros();
    for j in 0..m {
        for i in 0..j {
            let cji = -(sol.y[k0 + j] * c[j - 1][i]);
            let t = a[j] * cji * a[i].transpose();
            cov += t + t.transpose();
            c[j].push(cji);
        }
        c[j].push(sol.cov[k0 + j]);
        cov += a[j] * sol.cov[k0 + j] * a[j].transpose();
    }
    (x, cov)
}

/// The tree's solution: each piece's state `[c1, c2, t1, t2, log R]`, marginal covariance, and its elimination
/// into its parent (`x_k = ... - Y_k x_parent`).
struct Solved {
    x: Vec<Vector5<f64>>,
    cov: Vec<Matrix5<f64>>,
    y: Vec<Matrix5<f64>>,
}

/// The tree's posterior (see the module documentation): the prior, the used pieces' summaries linearised in log R,
/// and the censored pieces' bounds as expectation-propagation sites on log R, all re-solved until nothing moves.
fn tree_solve(rows: &Rows, fr: &[Frame], nd: &Nodes, sums: &[Summary], p: &PriorParams) -> Result<Solved> {
    let n = nd.row.len();
    let (h0, c0, g0) = prior_system(rows, fr, nd, p);
    let mut lin: Vec<f64> = (0..n).map(|k| sums[k].used().map_or(rows.r_prior[nd.row[k]].ln(), |c| c.x[4].ln())).collect();
    let terms: Vec<(usize, &[Bound])> = (0..n).map(|k| (k, sums[k].censored())).filter(|t| !t.1.is_empty()).collect();
    // each piece's site: precision and precision times mean, on its log R
    let mut site = vec![(0.0, 0.0); terms.len()];
    let mut round = 0;
    loop {
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
        for (&(k, _), &(tau, nu)) in terms.iter().zip(&site) {
            h[k][(4, 4)] += tau;
            g[k][4] += nu;
        }
        let sol = eliminate(&nd.parent, h, &c0, g)?;
        let mut moved = 0.0f64;
        for k in (0..n).filter(|&k| sums[k].used().is_some()) {
            moved = moved.max((sol.x[k][4] - lin[k]).abs() / sol.cov[k][(4, 4)].sqrt());
            lin[k] = sol.x[k][4];
        }
        for (t, &(k, bs)) in terms.iter().enumerate() {
            let (v, mu) = (sol.cov[k][(4, 4)], sol.x[k][4]);
            let (tau, nu) = site[t];
            // the cavity (the marginal without this site), the tilted moments, the site that matches them
            let pc = 1.0 / v - tau;
            if !(pc > 0.0) {
                continue;
            }
            let mc = (mu / v - nu) / pc;
            let Some((mh, vh)) = tilted(mc, 1.0 / pc, (mu, v.sqrt()), bs) else { continue };
            let (tau_new, nu_new) = ((1.0 / vh - pc).max(0.0), mh / vh - mc * pc);
            let next = (tau + EP_DAMP * (tau_new - tau), nu + EP_DAMP * (nu_new - nu));
            moved = moved.max((next.0 - tau).abs() * v).max((next.1 - nu).abs() * v.sqrt());
            site[t] = next;
        }
        round += 1;
        if (round >= 3 && moved < EP_TOL) || round >= EP_ROUNDS {
            return Ok(sol);
        }
    }
}

/// The mean and variance of `N(mu, v)` times a piece's probit terms `bs`, by the trapezoid rule on [`EP_NODES`]
/// points across `at.0 ± 6 at.1` (the piece's marginal, which the tilted distribution approaches); None where the
/// grid holds no mass. One site per piece makes the expectation propagation exact for a piece alone (a site per
/// term would not: two bounds on one piece leave its sd ~10 % wide).
fn tilted(mu: f64, v: f64, at: (f64, f64), bs: &[Bound]) -> Option<(f64, f64)> {
    let h = 12.0 * at.1 / (EP_NODES - 1) as f64;
    let e: Vec<f64> = (0..EP_NODES).map(|i| at.0 - 6.0 * at.1 + i as f64 * h).collect();
    let lf: Vec<f64> = e.iter().map(|&x| {
        -0.5 * (x - mu).powi(2) / v + bs.iter().map(|b| ln_ncdf(if b.upper { b.eta - x } else { x - b.eta } / b.sd)).sum::<f64>()
    }).collect();
    let top = lf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return None;
    }
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for (i, (&x, &l)) in e.iter().zip(&lf).enumerate() {
        let w = (l - top).exp() * if i == 0 || i + 1 == EP_NODES { 0.5 } else { 1.0 };
        (z, m1, m2) = (z + w, m1 + w * x, m2 + w * x * x);
    }
    let m = m1 / z;
    Some((m, (m2 / z - m * m).max(1e-6 * at.1 * at.1)))
}

/// `log Phi(z)`, by its asymptote where `Phi` underflows.
fn ln_ncdf(z: f64) -> f64 {
    if z < -30.0 { -0.5 * z * z - (-z).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() } else { ncdf(z).ln() }
}

/// Each piece's chance of being wood: its evidence (`log_odds`) under a chain along the branches (a base piece is
/// wood with [`EXIST`], a piece after wood with [`EXIST_ON`], one after none with [`EXIST_OFF`]), exactly by
/// sum-product, leaves up then root down.
fn existence(parent: &[Option<usize>], log_odds: &[f64]) -> Vec<f64> {
    let n = parent.len();
    let trans = |ep: usize, ek: usize| {
        let on = if ep == 1 { EXIST_ON } else { EXIST_OFF };
        if ek == 1 { on } else { 1.0 - on }
    };
    let norm = |v: [f64; 2]| {
        let s = v[0] + v[1];
        [v[0] / s, v[1] / s]
    };
    // up: the evidence of a piece's subtree given its state; msg: what it tells its parent
    let mut up: Vec<[f64; 2]> = log_odds.iter().map(|&l| {
        let pw = 1.0 / (1.0 + (-l).exp());
        [1.0 - pw, pw]
    }).collect();
    let mut msg = vec![[1.0; 2]; n];
    for k in (0..n).rev() {
        up[k] = norm(up[k]);
        if let Some(q) = parent[k] {
            msg[k] = norm(std::array::from_fn(|ep| trans(ep, 0) * up[k][0] + trans(ep, 1) * up[k][1]));
            up[q] = [up[q][0] * msg[k][0], up[q][1] * msg[k][1]];
        }
    }
    let mut bel = vec![[0.0; 2]; n];
    for k in 0..n {
        let down = match parent[k] {
            None => [1.0 - EXIST, EXIST],
            Some(q) => {
                let pq = [bel[q][0] / msg[k][0], bel[q][1] / msg[k][1]];
                std::array::from_fn(|ek| trans(0, ek) * pq[0] + trans(1, ek) * pq[1])
            }
        };
        bel[k] = norm([up[k][0] * down[0], up[k][1] * down[1]]);
    }
    bel.iter().map(|b| b[1]).collect()
}

/// The prior's normal equations on the pieces: each piece's block, its coupling to its parent's, and the
/// right-hand side (the prior is Gaussian in `[c1, c2, t1, t2, log R]`). A row's first piece takes the row's taper
/// step or fork share; its later pieces continue it, with no step.
#[allow(clippy::type_complexity)]
fn prior_system(rows: &Rows, fr: &[Frame], nd: &Nodes, p: &PriorParams) -> (Vec<Matrix5<f64>>, Vec<Matrix5<f64>>, Vec<Vector5<f64>>) {
    let n = nd.row.len();
    let share = rows.fork_share(&fr.iter().map(|f| f.3).collect::<Vec<_>>(), ForkShare::WoodLength);
    let eta0: Vec<f64> = nd.row.iter().map(|&k| rows.r_prior[k].ln()).collect();
    // a factor: sum of coefficients times (piece, unknown), its mean and sd
    type Factor = (Vec<(usize, usize, f64)>, f64, f64);
    let mut factors: Vec<Factor> = Vec::new();
    for k in 0..n {
        let row = nd.row[k];
        let sc = p.centre_rel * rows.r_prior[row] + p.centre_abs;
        factors.push((vec![(k, 0, 1.0)], 0.0, sc));
        factors.push((vec![(k, 1, 1.0)], 0.0, sc));
        factors.push((vec![(k, 2, 1.0)], 0.0, LEAN_SD));
        factors.push((vec![(k, 3, 1.0)], 0.0, LEAN_SD));
        let Some(q) = nd.parent[k] else {
            factors.push((vec![(k, 4, 1.0)], eta0[k], p.sig_root));
            continue;
        };
        let len = nd.frame[k].3;
        let ls = len.max(0.01).sqrt();
        if nd.first[row] == k && share[row].is_finite() {
            factors.push((vec![(k, 4, 1.0), (q, 4, -1.0)], 0.5 * (p.mu_fork + share[row].ln()), p.sig_fork));
            continue;
        }
        factors.push((vec![(k, 4, 1.0), (q, 4, -1.0)], eta0[k] - eta0[q], p.sig_eta * ls));
        // the child's start (its centre less half its length's lean) at the parent's end, in the child's frame
        let (ek, eq) = ([nd.frame[k].1, nd.frame[k].2], [nd.frame[q].1, nd.frame[q].2]);
        let lq = nd.frame[q].3;
        for j in 0..2 {
            let tr = [dot(&ek[j], &eq[0]), dot(&ek[j], &eq[1])];
            factors.push((vec![(k, j, 1.0), (k, 2 + j, -0.5 * len), (q, 0, -tr[0]), (q, 1, -tr[1]), (q, 2, -0.5 * lq * tr[0]), (q, 3, -0.5 * lq * tr[1])], 0.0, p.sig_delta * ls));
            // within a row (one frame), the lean bends by a random walk
            if nd.row[q] == row {
                factors.push((vec![(k, 2 + j, 1.0), (q, 2 + j, -1.0)], 0.0, LEAN_BEND * ls));
            }
        }
    }
    // the prior's precision (fixed), as piece blocks and child-parent couplings
    let (mut h0, mut c0) = (vec![Matrix5::<f64>::zeros(); n], vec![Matrix5::<f64>::zeros(); n]);
    let mut g0 = vec![Vector5::<f64>::zeros(); n];
    for (vars, mean, sd) in &factors {
        let w = 1.0 / (sd * sd);
        let k = vars[0].0; // the factor's own piece: a child's, where it links to its parent
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
    (h0, c0, g0)
}

/// Solves `H x = g` on a tree of 5x5 blocks (`c[k]`: block k's coupling to its parent's), leaves first, with each
/// block's marginal covariance top-down (`Sigma_k = H_k^-1 + Y Sigma_q Y^T`). The prior's anchors make every block
/// positive definite: one that isn't is an error.
fn eliminate(parent: &[Option<usize>], mut h: Vec<Matrix5<f64>>, c: &[Matrix5<f64>], mut g: Vec<Vector5<f64>>) -> Result<Solved> {
    let n = h.len();
    let mut inv = vec![Matrix5::<f64>::zeros(); n];
    let mut y = vec![Matrix5::<f64>::zeros(); n];
    let mut yv = vec![Vector5::<f64>::zeros(); n];
    for k in (0..n).rev() {
        inv[k] = h[k].cholesky().ok_or_else(|| Error::invalid(format!("support tree: piece {k}'s block is not positive definite")))?.inverse();
        yv[k] = inv[k] * g[k];
        if let Some(q) = parent[k] {
            y[k] = inv[k] * c[k];
            h[q] -= c[k].transpose() * y[k];
            g[q] -= c[k].transpose() * yv[k];
        }
    }
    let (mut x, mut cov) = (vec![Vector5::<f64>::zeros(); n], vec![Matrix5::<f64>::zeros(); n]);
    for k in 0..n {
        match parent[k] {
            Some(q) => {
                x[k] = yv[k] - y[k] * x[q];
                cov[k] = inv[k] + y[k] * cov[q] * y[k].transpose();
            }
            None => {
                x[k] = yv[k];
                cov[k] = inv[k];
            }
        }
    }
    Ok(Solved { x, cov, y })
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
        let t = support_tree(&[&scan], &rows, &like(), &PriorParams::default(), 2).unwrap();
        let (f, q) = (&t.rows, &t.pieces);
        // the 2 m row is seven pieces of the tree; the fan sees the middle three, and they are measured
        assert_eq!(t.piece_row, vec![0; 7]);
        let seen: Vec<usize> = (0..7).filter(|&j| q.n_pairs[j] > 0).collect();
        assert_eq!(seen, vec![2, 3, 4], "pairs {:?}", q.n_pairs);
        assert!(seen.iter().all(|&j| q.flags[j] == 0 && (q.r_row[j] - 0.05).abs() < 0.0025 && q.n_runs[j] == 1 && q.n_scans[j] == 1), "flags {:?} r {:?}", q.flags, q.r_row);
        // the others are the prior's along the chain, with nothing to bound them; the row says so
        assert!((0..7).filter(|j| !seen.contains(j)).all(|j| q.flags[j] == flag::BOUNDS && q.r_lo[j].is_nan() && q.r_hi[j].is_nan()));
        assert_eq!(f.flags[0], flag::BOUNDS);
        assert!((f.r_row[0] - 0.05).abs() < 0.0025 && (f.r[0] - 0.05).abs() < 0.0025, "R {} {}", f.r_row[0], f.r[0]);
        // the row runs from its first piece's start to its last's end
        assert!((0..3).all(|i| (f.a[0][i] - q.a[0][i]).abs() < 1e-9 && (f.b[0][i] - q.b[6][i]).abs() < 1e-9));
        // an opaque bark-free stem is wood, and so are the unseen pieces it carries on from
        assert!(seen.iter().all(|&j| q.log_odds[j] > LLR_CLIP - 1e-9) && q.p_wood.iter().all(|&p| p > 0.9), "log odds {:?} p_wood {:?}", q.log_odds, q.p_wood);

        // the part the fan sees, 48 cm in two pieces: measured, the claim moved back by 1 cm, upright
        let (a2, b2) = ([a1[0], a1[1], -0.24], [b1[0], b1[1], 0.24]);
        let rows = Rows { a: &[a2], b: &[b2], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let t = support_tree(&[&scan], &rows, &like(), &PriorParams::default(), 2).unwrap();
        let (f, q) = (&t.rows, &t.pieces);
        assert!(t.piece_row == vec![0, 0] && f.flags[0] == 0 && q.flags == vec![0, 0], "flags {:?}", q.flags);
        assert!((f.r[0] - 0.05).abs() < 0.0025 && (f.r_row[0] - f.r[0]).abs() < 0.002 && f.sd_log_r[0] < 0.1, "R {} {}", f.r[0], f.r_row[0]);
        assert!((0.5 * (f.a[0][0] + f.b[0][0])).abs() < 0.002 && f.lean[0][1].abs() < 2.0 * f.sd_lean[0][1], "x {} {} lean {:?}", f.a[0][0], f.b[0][0], f.lean[0]);
        // one scan: the front's stops still give a misfit
        assert!(f.chi2_front[0].is_finite() && f.chi2_front[0] < HULL_CHI2, "chi2_front {}", f.chi2_front[0]);
    }

    fn circle(x: Vector5<f64>, sd: f64) -> Circle {
        let info = Matrix5::<f64>::identity() / (sd * sd);
        Circle { st: Stats { info, h: info * x, yy: x.dot(&(info * x)), n: 5 }, x, cov: Matrix5::identity() * sd * sd, chi2: f64::NAN }
    }

    fn summary(circle: Option<Circle>, flags: u8, bounds: Vec<Bound>) -> Summary {
        Summary { circle, chi2_front: f64::NAN, flags, n_runs: 1, n_scans: 1, n_pairs: 100, bounds, log_odds: 0.0 }
    }

    #[test]
    fn the_tree_solve_is_exact_and_the_prior_carries_unmeasured_pieces() {
        // a 0.6 m base (two pieces), a 0.3 m row, and a fork of two; the 0.3 m row measured
        let rows_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.9], [0.0, 0.0, 0.9]];
        let rows_b = [[0.0, 0.0, 0.6], [0.0, 0.0, 0.9], [0.2, 0.0, 1.1], [-0.2, 0.0, 1.1]];
        let rp = [0.1, 0.08, 0.05, 0.04];
        let rows = Rows { a: &rows_a, b: &rows_b, r_prior: &rp, r_max: &[], parent: &[-1, 0, 1, 1] };
        let fr = frames(&rows_a, &rows_b);
        let nd = Nodes::of(&rows, &fr);
        assert_eq!((nd.row.clone(), nd.first.clone()), (vec![0, 0, 1, 2, 3], vec![0, 2, 3, 4, 5]));
        assert_eq!(nd.parent, vec![None, Some(0), Some(1), Some(2), Some(2)]);
        let c = circle(Vector5::new(0.01, -0.005, 0.02, 0.0, 0.06), 1e-3);
        // a routed piece's circle stays out of the tree
        let routed = summary(Some(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.2), 1e-3)), flag::RUNS, Vec::new());
        let sums = vec![summary(None, flag::BOUNDS, Vec::new()), summary(None, flag::BOUNDS, Vec::new()), summary(Some(c), 0, Vec::new()), routed, summary(None, flag::BOUNDS, Vec::new())];
        let pp = PriorParams::default();
        let sol = tree_solve(&rows, &fr, &nd, &sums, &pp).unwrap();
        let (x, sig) = (&sol.x, &sol.cov);
        // the measured piece follows its measurement; its neighbours follow the taper from it (the base's upper
        // piece between the taper's step from it, 7.5 cm, and the base's anchor on the claim, 10 cm)
        assert!((x[2][4].exp() - 0.06).abs() < 0.001 && (x[2][0] - 0.01).abs() < 0.001, "{:?}", x[2]);
        assert!(x[1][4].exp() > 0.075 && x[1][4].exp() < 0.1, "{}", x[1][4].exp());
        assert!(sig[2][(4, 4)] < sig[1][(4, 4)] && sig[2][(4, 4)] < sig[3][(4, 4)]);
        // the prior alone leaves an unmeasured piece's radius loose: the fork's spread, not a measurement's; the
        // fork's children split the parent's area by wood length (equal here), with mu_fork
        assert!(sig[3][(4, 4)].sqrt() > pp.sig_fork * 0.9, "sd {}", sig[3][(4, 4)].sqrt());
        assert!((x[3][4] - x[4][4]).abs() < 1e-9);
        // the elimination is the dense solve of the same system, mean and marginal covariances
        let (h, c, g) = prior_system(&rows, &fr, &nd, &pp);
        let n = nd.row.len();
        let mut hd = nalgebra::DMatrix::<f64>::zeros(5 * n, 5 * n);
        let mut gd = nalgebra::DVector::<f64>::zeros(5 * n);
        for k in 0..n {
            hd.view_mut((5 * k, 5 * k), (5, 5)).copy_from(&h[k]);
            gd.rows_mut(5 * k, 5).copy_from(&g[k]);
            if let Some(q) = nd.parent[k] {
                hd.view_mut((5 * k, 5 * q), (5, 5)).copy_from(&c[k]);
                hd.view_mut((5 * q, 5 * k), (5, 5)).copy_from(&c[k].transpose());
            }
        }
        let inv = hd.try_inverse().unwrap();
        let xd = &inv * &gd;
        let se = eliminate(&nd.parent, h, &c, g).unwrap();
        for k in 0..n {
            for i in 0..5 {
                assert!((se.x[k][i] - xd[5 * k + i]).abs() < 1e-9, "x[{k}][{i}]");
                for j in 0..5 {
                    assert!((se.cov[k][(i, j)] - inv[(5 * k + i, 5 * k + j)]).abs() < 1e-9 * (1.0 + inv[(5 * k + i, 5 * k + j)].abs()), "cov[{k}]");
                }
            }
        }
        // the base row from its two pieces: its mean log R and lean along e1, against the dense covariance
        let (xr, cr) = row_line(&se, 0, 2, 0.6, 0.3);
        let mut a = nalgebra::DVector::<f64>::zeros(5 * n);
        (a[4], a[9]) = (0.5, 0.5);
        assert!((xr[4] - (0.5 * (xd[4] + xd[9]))).abs() < 1e-12 && (cr[(4, 4)] - (a.transpose() * &inv * &a)[0]).abs() < 1e-12);
        let mut l = nalgebra::DVector::<f64>::zeros(5 * n);
        (l[0], l[2], l[5], l[7]) = (-1.0 / 0.6, 0.25, 1.0 / 0.6, 0.25);
        assert!((cr[(2, 2)] - (l.transpose() * &inv * &l)[0]).abs() < 1e-12, "{} {}", cr[(2, 2)], (l.transpose() * &inv * &l)[0]);
        // a block that isn't positive definite is an error, not a silent zero
        let bad = vec![Matrix5::<f64>::zeros()];
        assert!(eliminate(&[None], bad, &[Matrix5::zeros()], vec![Vector5::zeros()]).is_err());
    }

    #[test]
    fn bounds_cut_the_prior_to_the_moments_of_its_truncation() {
        // one base piece, its log R a priori N(log 1 cm, sig_root): an upper bound at 6 mm, then also a lower at 3 mm
        let rows = Rows { a: &[[0.0, 0.0, 0.0]], b: &[[0.0, 0.0, 0.2]], r_prior: &[0.01], r_max: &[], parent: &[-1] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let pp = PriorParams::default();
        let up = Bound { eta: 0.006f64.ln(), sd: 0.1, upper: true };
        let lo = Bound { eta: 0.003f64.ln(), sd: 0.1, upper: false };
        // the posterior's mean and sd by quadrature
        let truth = |bs: &[Bound]| {
            let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
            for i in 0..200_000 {
                let e = 0.01f64.ln() - 4.0 + 8.0 * i as f64 / 200_000.0;
                let mut w = (-0.5 * ((e - 0.01f64.ln()) / pp.sig_root).powi(2)).exp();
                for b in bs {
                    w *= ncdf(if b.upper { b.eta - e } else { e - b.eta } / b.sd);
                }
                (z, m1, m2) = (z + w, m1 + w * e, m2 + w * e * e);
            }
            (m1 / z, (m2 / z - (m1 / z).powi(2)).sqrt())
        };
        for bs in [vec![up], vec![up, lo]] {
            let sol = tree_solve(&rows, &fr, &nd, &[summary(None, flag::BOUNDS, bs.clone())], &pp).unwrap();
            let (m, s) = truth(&bs);
            let (me, se) = (sol.x[0][4], sol.cov[0][(4, 4)].sqrt());
            // one piece: its site is exact (up to the quadrature)
            assert!((me - m).abs() < 0.01 * s && (se - s).abs() < 0.01 * s, "{} bounds: {me} {se} against {m} {s}", bs.len());
            assert!(se < pp.sig_root && me < 0.006f64.ln() + 0.1);
        }
        // a routed piece's bounds stay out
        let sol = tree_solve(&rows, &fr, &nd, &[summary(None, flag::BOUNDS | flag::HULL, vec![up])], &pp).unwrap();
        assert!((sol.x[0][4] - 0.01f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn existence_is_exact_on_the_branches() {
        // a base, its child, and a fork of two, by enumeration of the 16 states
        let parent = [None, Some(0), Some(1), Some(1)];
        let lo = [2.0, -1.0, 0.5, -3.0];
        let p = existence(&parent, &lo);
        let mut marg = [0.0; 4];
        let mut z = 0.0;
        for st in 0..16usize {
            let e = |k: usize| (st >> k) & 1;
            let mut w = 1.0;
            for k in 0..4 {
                w *= if e(k) == 1 { lo[k].exp() } else { 1.0 };
                let pk = match parent[k] {
                    None => EXIST,
                    Some(q) => if e(q) == 1 { EXIST_ON } else { EXIST_OFF },
                };
                w *= if e(k) == 1 { pk } else { 1.0 - pk };
            }
            z += w;
            for k in 0..4 {
                marg[k] += w * e(k) as f64;
            }
        }
        for k in 0..4 {
            assert!((p[k] - marg[k] / z).abs() < 1e-12, "{k}: {} {}", p[k], marg[k] / z);
        }
        // wood doesn't restart: a child's wood is evidence of its parent's
        let p = existence(&[None, Some(0)], &[-6.0, 4.0]);
        let alone = existence(&[None], &[-6.0])[0];
        assert!(p[0] > 10.0 * alone && p[1] > p[0], "{p:?} {alone}");
        // and with no evidence, every piece keeps the prior however far out
        let p = existence(&(0..50usize).map(|k| k.checked_sub(1)).collect::<Vec<_>>(), &[0.0; 50]);
        assert!(p.iter().all(|&x| (x - EXIST).abs() < 1e-9), "{p:?}");
    }

    #[test]
    fn misfit_counts_on_the_rank() {
        // a measurement set of rank 2 (two edges): no redundancy, no misfit; a third edge is redundant
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
        // bark-bright is wood, leaf-dark is leaf, unknown is wood (and no evidence); a multi-return echo is judged darker
        assert!(p_wood(&rec(-15.5, true), 0.05, &p) > 0.9);
        assert!(p_wood(&rec(-21.0, true), 0.05, &p) < 0.1);
        assert_eq!(p_wood(&rec(f64::NAN, true), 0.05, &p), 1.0);
        assert_eq!(llr(&rec(f64::NAN, true), 0.05, &p), 0.0);
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
