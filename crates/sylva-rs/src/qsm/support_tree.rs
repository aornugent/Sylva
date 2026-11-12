// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One tree's wood from the support function of each row's cross-section, as
//! each scan measures it, and one solve over the tree (3dtrees research 93
//! §10, §16, §17): the fast path of the joint model, with flags that send the
//! pieces it can't summarise (several stems, a hull) to
//! the exact likelihood (`routed.rs`).
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
//!   log10(2R/w)` where the wood is narrower than the footprint and by
//!   [`JointParams::refl_thin`] on thin wood (research 93 §16); the rest of
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
//!   opaque run), each group still bounds R (research 93 §17), slice by slice
//!   along the piece: its woody stops at the claim's depth, about the one
//!   nearest the axis, span no more than the silhouette (a lower bound, `R >=
//!   span / 2 - w (1/2 - tau)`); the nearest passes either side of them, or
//!   either side of the woody centre in a slice where no woody stop is, lie
//!   outside it (an upper bound, `R <= gap / 2 - w (1/2 - tau)`, a pass
//!   there a pulse that met nothing in the tube; a side with no recorded
//!   pulse, or a stop anywhere in the tube or a blocked pulse next, gives
//!   none); and wood
//!   that was hit fills at least `tau` of the footprint (the capacity floor `R
//!   >= tau w / 2`). Each is a probit term on log
//!   R: the bounds of the edge's sd (and the impacts' quantisation), the floor
//!   of [`CAP_SD`].
//! - **Existence.** Each group's stops and passes aimed at the core (the
//!   span of its runs of stops of any echo character about their centre, or
//!   the claim's middle half where none) give a log odds of wood against
//!   foliage: each stop's echo character and, per pulse, opacity (wood stops
//!   [`JointParams::stop_wood`] of them, foliage [`JointParams::stop_leaf`]:
//!   a stop counts for wood, a pass against it, the passes' net at most one
//!   sure echo), divided by the beam's oversampling `(w / spacing)^2` (research
//!   93 §15.6) and weighted by [`JointParams::exist_scale`] (fitted on
//!   `joint_tree`'s wood and NO_WOOD rows, research 93 §17 review: a scan's
//!   pulses are more correlated than the oversampling says). A chain along the branches (wood doesn't stop and restart,
//!   [`PriorParams::exist`] and [`PriorParams::exist_off`]) turns the pieces'
//!   log odds into `p_wood`, exactly by sum-product.
//! - **Flags:** [`flag::RUNS`] (a group's profile, its stops gated by the
//!   fitted circle's depth band, shows two or more stems within the claim),
//!   [`flag::HULL`] (the support is wider than one circle explains within the
//!   piece: more than a share of the stops aimed at the fitted circle's core
//!   lie behind its front, the measurements or the front's stops misfit it,
//!   or the run is far wider than it), [`flag::BOUNDS`] (R isn't determined:
//!   bounds), [`flag::POROUS`] (pulses but no opaque run: existence and
//!   bounds). RUNS and HULL pieces are fitted by the exact likelihood here.
//! - **Routed pieces, exactly** (research 93 §18; [`routed`]): seeds from the
//!   piece's runs, each a circle with its own lean, measured slice by slice;
//!   its pulses taken once about the claim's axis, each stem seeing them from
//!   its own leaning axis, the stops the tree's measured and bounded pieces (a
//!   first solve without the routed ones) explain better left to them; then
//!   1, 2, ... stems by the exact likelihood (Newton from the seeds to its
//!   stopping rule) while BIC on the same pulses rises. Each stem's
//!   likelihood message, tempered by the pulses' correlation, measures the
//!   piece's centre and R, its lean its seed's: of the stems starting near
//!   its parent's end, the one whose message says most of its radius
//!   continues the piece ([`enter`]); another continues a stem of a routed
//!   parent piece whose end it starts at, or else is a node beside it. A fit
//!   refused (no seed or pulse, unconverged, not finite, a stem that says
//!   nothing of its radius, or one stem narrower than the piece's detection
//!   bound) isn't used: the
//!   piece is the prior cut by its bounds. An accepted stem whose sides, as its
//!   scans see them, leave its radius trading against its centre ([`unseen`]:
//!   one edge and the front from one scan, the other edge behind a crossing
//!   branch) says nothing along that ridge ([`marginalise`]), and its own
//!   bounds enter the tree beside it (research 93 §20).
//! - **Passes alone** bound a piece no woody stop near the claim gives a
//!   centre ([`veto`], 3dtrees research 106): its R is at most the widest
//!   circle about the claim's axis that the ray veto doesn't refute (at least
//!   [`VETO_PASSES`] passes through it, a share [`VETO_SHARE`] of its stops and
//!   passes). Every node's fitted circle is audited the same way (`n_pass`,
//!   `pass_share`, [`diag::REFUTED`]).
//! - **Point evidence** (3dtrees ticket 109, research 111): a row none of
//!   whose pieces has a circle of its own gets one from the stops in its
//!   pieces' tubes, placed across its axis, as the refit fitted RCT's points
//!   ([`point_circle`]: RANSAC, least squares on its inliers, and the refit's
//!   tests of inliers, arc, misfit, shift and width, at most [`PTS_BELIEVE`]
//!   times the site's cap, and short of its pieces' tubes' edge, where a piece's
//!   own circle would be refused too). Unless the ray veto refutes it (on a
//!   tangle a circle can be another tree's), it measures each of the row's n
//!   pieces' R and both coordinates of its centre with sd `PTS_SD R sqrt(n)`
//!   ([`PTS_SD`] in log R; one circle shared by n pieces, not n circles), and the
//!   pieces' upper bounds (passes) enter beside it ([`basis::POINTS`]); the
//!   bounds its own stops gave (a woody run's span, the capacity floor) it
//!   replaces, so they aren't counted twice (3dtrees ticket 115).
//! - **The site's cap** (`PriorParams::dh`, 3dtrees research 62, 106
//!   `uncertain.md`): after the solve, every node into which no circle
//!   enters (its own or its row's point circle) has R at most `dh` times its
//!   reach to the tips over 2, applied to that node alone (its point and its volume's draws), never in the solve:
//!   it is a stem-scale rule, no true bound on a short-reach twig, so it must
//!   not carry down the tree's chain.
//! - **Held inside its bounds:** a posterior that the soft terms leave outside
//!   its own bounds is moved onto them (the upper where they cross), and every
//!   node says how its radius is known ([`basis`]; [`diag::PINNED`],
//!   [`diag::CAPPED`]). Then the site's cap, then the veto: a node with no
//!   circle of its own whose posterior circle the veto refutes is narrowed
//!   about its centre until it isn't (the cap first, as a narrower circle can be
//!   one the rays refute), or where that would cut it below its lower bound,
//!   moved onto the nearest unrefuted circle that wide within three of its radii
//!   whose centre is its own claim's, not a neighbour's ([`recentre`],
//!   [`diag::MOVED`]: its wood beside the claim's axis, 3dtrees research 127).
//!   A point circle's piece has no lower bound and never moves; a refused
//!   routed piece's lower bound is its capacity floor, so it may move within
//!   three floor radii.
//! - **The volume** ([`Volume`]): the tree's pieces' at their posterior R, by
//!   basis, and an interval from draws of the joint posterior, each node's
//!   sd [`WIDEN`] times the solve's for its basis (as the reported
//!   `sd_log_r`; a row's from its pieces', each widened by its own).
//! - **Diagnostics** ([`diag`]): a flag word and three numbers per node and row
//!   for a reviewer or the screen.
//! - **The classification:** every stop of the tree, keyed by scan, pulse row
//!   in its Shots file and echo ([`Echoes`]): on the front arc of its piece's
//!   circle (its own, or the tree's) or not, times its echo's weight; on a
//!   routed piece with a fit, the exact likelihood's wood and leaf terms at its
//!   stem's fit.
//! - **The tree:** the unflagged pieces' summaries as Gaussian measurements
//!   of `(c0, t, R)`, the bounds of BOUNDS and POROUS pieces as censored
//!   measurements of log R, and a Gaussian Markov prior over the pieces after
//!   the joint fit's [`PriorParams`] (the base's level, the taper's steps, at
//!   a fork the child's share of the subtree wood length rather than of the
//!   prior area, heavy-tailed (Student-t of [`FORK_DF`] degrees of freedom, by
//!   expectation-maximisation's reweighting) so a small shoot measured fat
//!   can't inflate its parent, the centre's continuity along a branch, the
//!   lean's random walk between a row's pieces, the centre's anchor on the
//!   claim and a weak anchor of the lean; no tip anchor). Every term links a
//!   piece only to its parent, so each Gaussian
//!   system is solved exactly by eliminating pieces into their parents, leaves
//!   first (as [`crate::qsm::joint_fit`]'s branch blocks, which without the
//!   per-scan offsets are single rows), with each piece's marginal covariance.
//!   The radius is measured in R and modelled in log R (Gauss-Newton), and the
//!   censored terms enter by expectation propagation: each a Gaussian site on
//!   its piece's log R matched to the moments of its probit times the cavity
//!   (exact per term), all sites updated from the marginals of one solve and
//!   damped, until nothing moves (or [`EP_ROUNDS`] pass: the tree's
//!   `converged` says which). The posterior is then the prior cut by the
//!   bounds, mean and sd. A row's output is its pieces': the line from the
//!   first's start to the last's end and their root mean square R (the row's
//!   volume is theirs), with the covariance of the chain.

use nalgebra::{Matrix3, Matrix4, Matrix5, SymmetricEigen, Vector3, Vector4, Vector5};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::scan_index::{pool, ScanIndex};
use crate::qsm::joint::{log_ndtr_value, window, Echoes, JointParams};
use crate::qsm::joint_fit::{ForkShare, Frame, PriorParams, Rows};
use crate::transform::{cross, dot, sub};
use crate::trees::{fit_circle_ransac, RansacCircleParams};
use crate::Point;

pub mod base;
mod routed;

/// Why a piece's own summary is not used (bits of [`Fit::flags`]): [`RUNS`] and [`HULL`] pieces are fitted by the
/// exact likelihood ([`routed`]), [`BOUNDS`] and [`POROUS`] ones are bounded.
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

/// What a reviewer or the screen should know of a node's fit (bits of [`Fit::diag`], 3dtrees research 93 §20; a
/// row's: its pieces'), beside its three numbers (`data_share`, `contested`, `deleted`): only what no other output
/// says. Each threshold is a constant below with its reason.
pub mod diag {
    /// One scan supplies almost all its pulses ([`super::ONE_SCAN`]), or every pair of the views that carry a share of
    /// them ([`super::VIEW_SHARE`]) lies within [`super::VIEW_MIN`] of each other: no other view checks it (research 93
    /// §10, §11.1).
    pub const ONE_SIDED: u16 = 1;
    /// An exact stem a side of whose silhouette is unseen (no pulse reaches the circle's neighbourhood there from a
    /// scan that would decide R, as behind a crossing branch): its radius isn't determined by its own data
    /// ([`super::unseen`]), so its message says nothing along what its sides leave free and its own bounds enter the
    /// tree. A side the fit's table doesn't reach isn't judged ([`TUBE_EDGE`] says so).
    pub const SHADOWED: u16 = 2;
    /// R below about half the footprint at its range ([`super::SUB_W`]): the edges are the beam's, not the wood's.
    pub const SUB_FOOTPRINT: u16 = 4;
    /// Its own fitted circle reaches within [`super::TUBE_MARGIN`] of the tube its pulses were taken in: evidence
    /// beyond it is cut off. A piece's own circle that does is not used (3dtrees ticket 109: a stem captured by a
    /// neighbour's or the ground's mass, or drifted onto a leaf clump at the tube's edge): its bounds are, but for the
    /// lower bound of a scan whose woody run the circle spans (the captor's); an exact stem's is only reported.
    pub const TUBE_EDGE: u16 = 8;
    /// At least [`super::CONTESTED_SHARE`] of its stops are claimed by context or a neighbouring node (`contested`).
    pub const CONTESTED: u16 = 16;
    /// The rays refute its fitted circle: at least [`super::VETO_PASSES`] pulses pass through it, and passes are at
    /// least [`super::VETO_SHARE`] of its stops and passes (3dtrees research 106 `geometry.md`; a row's: of its
    /// pieces' counts summed, not their flags). An unmeasured piece is bounded below this ([`super::veto`]), so it is
    /// rare there.
    pub const REFUTED: u16 = 32;
    /// Its radius sits on its own bounds: the tree without them would put it past one (its cavity's mean lies beyond
    /// a bound), or the posterior did and was held inside them. What the rays say there is a bound, not a value.
    pub const PINNED: u16 = 64;
    /// Its posterior median reached the site's D/H cap (`PriorParams::dh`), applied after the solve: it is served
    /// at the cap.
    pub const CAPPED: u16 = 128;
    /// The veto moved its circle onto the wood its lower bound saw beside it ([`super::recentre`]), refuted or not.
    pub const MOVED: u16 = 256;
}

/// How a node's radius is known ([`Fit::basis`]; a row's: most of its pieces', the less known on a tie), the classes
/// its tree's volume is split into ([`Volume`]).
pub mod basis {
    /// Its own circle, or an exact stem's message, that isn't [`super::diag::SHADOWED`].
    pub const MEASURED: u8 = 1;
    /// Its row's point circle ([`super::point_circle`]): the rays gave it no circle of its own.
    pub const POINTS: u8 = 2;
    /// Its own bounds give at least [`super::BOUNDED_SHARE`] of its precision (`data_share`), and it isn't pinned.
    pub const BOUNDED: u8 = 3;
    /// The prior's, given the rest of the tree: too little of its own data, or pinned to a bound of its own.
    pub const PRIOR: u8 = 4;
    /// The site's cap ([`super::diag::CAPPED`]).
    pub const CAPPED: u8 = 5;
}

/// [`diag::ONE_SIDED`]: one scan's share of a piece's informative pulses (90 %: the rest can't check it), or the
/// spread of the directions its scans look from, those with [`VIEW_SHARE`] of its pulses (20 degrees: the support
/// function's orders beyond the circle need views far apart, research 93 §10).
const ONE_SCAN: f64 = 0.9;
const VIEW_MIN: f64 = 20.0;
/// [`diag::ONE_SIDED`]: the share of a piece's pulses a scan needs for its view to count (a tenth: a scan with fewer
/// checks too little of the circle to be a second view).
const VIEW_SHARE: f64 = 0.1;
/// [`diag::SUB_FOOTPRINT`]: R below this share of the footprint (the brief's "about half": below it the hit
/// share across the silhouette is the beam's profile).
const SUB_W: f64 = 0.5;
/// [`diag::TUBE_EDGE`]: how near the tube's edge a fitted stem reaches (m: a few edge widths of a footprint at
/// 10-20 m, past which an edge's passes are cut off).
const TUBE_MARGIN: f64 = 0.01;
/// [`diag::CONTESTED`]: the share of a node's stops claimed elsewhere (a half: most of what it saw is another's;
/// held out over halvings of the bench's trees, research 93 §20).
const CONTESTED_SHARE: f64 = 0.5;
/// A circle's radius is undetermined by the sides its scans see where `var(R) W`, R's variance from the sides
/// (each a measurement of the support function, of weight `w`) times their total weight `W`, exceeds this squared:
/// `sqrt(var(R) W)` is R's sd against one measurement of R with all the weight. One scan's two edges and front give
/// 1.22, two edges alone 1; one edge and the front, unbounded; a second edge of 6 % of the first's weight, 3
/// (research 93 §20).
const SHADOW_SD: f64 = 3.0;
/// A side (an edge's neighbourhood, or the front) is seen by a scan where at least this many pulses reached it.
const SEEN_MIN: usize = 3;
/// The variance (m^2) a radius left undetermined is reported with, along the directions its sides leave free.
const UNSEEN_VAR: f64 = 1.0;
/// The ray veto (3dtrees research 106 `geometry.md`): a circle is refuted where at least `VETO_PASSES` pulses pass
/// through it and passes are at least `VETO_SHARE` of its stops and passes, a stop within `VETO_TOL` (m) of its chord
/// being on it (right circles read 0-6 %, every phantom over 40 %). An unmeasured piece's bound from it is the
/// widest circle about the claim's axis the veto leaves, on a grid of `VETO_STEPS` radii from `VETO_MIN` (m) to its
/// tube, entered with an sd of `VETO_SD` in log R.
const VETO_PASSES: u32 = 10;
const VETO_SHARE: f64 = 0.25;
const VETO_TOL: f64 = 0.02;
const VETO_STEPS: usize = 48;
const VETO_MIN: f64 = 0.001;
const VETO_SD: f64 = 0.1;
/// The step (log R) by which an unmeasured node's posterior circle the veto refutes is narrowed until it doesn't.
const VETO_STEP: f64 = 0.05;
/// [`recentre`]'s grid of centres: rings and spacing of `RECENTRE_STEP` of the radius kept (at least `RECENTRE_MIN`
/// m), out to `RECENTRE_REACH` radii (a ring beside the claim's axis, research 122 F14, not another stem).
const RECENTRE_STEP: f64 = 0.5;
const RECENTRE_MIN: f64 = 0.002;
const RECENTRE_REACH: f64 = 3.0;
/// [`basis::BOUNDED`]: the least `data_share` of a node whose bounds say something (3dtrees research 106
/// `uncertain.md`: the posterior's interval covers the refit's circles as well from there as on measured rows).
const BOUNDED_SHARE: f64 = 0.1;
/// The factor each basis's posterior sds of log R are widened by, in the reported `sd_log_r` and the volume's draws
/// (by [`basis`], from MEASURED): the raw posterior is overconfident. Learned per basis as 3dtrees research 106
/// `uncertain.md` §1 learned one (3dtrees ticket 109): the factor that puts 90 % of Alice Mulga's refit circles (their
/// own noise, 0.136 in log R, added) inside the 90 % interval, on random halves of the trees, 90.0-90.4 % on the other
/// halves; the circles on PINNED nodes left out (6.5 %: a circle past the rays' own bound, 86 % of them fatter, by a
/// median 4.4x). MEASURED 2.1 [2.0-2.2], BOUNDED 3.6 [3.3-3.8], PRIOR 3.1 [2.9-3.2], CAPPED 2.3 [2.1-2.4]. POINTS
/// learns 1.2 [1.1-1.3], but that is a floor, not a calibration: its reference (the refit's circle) was fitted to the
/// same stops, so their errors are shared and cancel; it takes MEASURED's 2.1, the factor of a circle checked against
/// an independent one. One factor for all (2.0) left bounded and prior rows short.
pub const WIDEN: [f64; 5] = [2.1, 2.1, 3.6, 3.1, 2.3];

/// The factor a node of basis `b` is widened by ([`WIDEN`]).
fn widen(b: u8) -> f64 {
    WIDEN[usize::from(b) - 1]
}

/// Point evidence ([`point_circle`], 3dtrees research 111), with the refit's settings (3dtrees research 52, 62): a row
/// claimed at least [`PTS_R_MIN`] (m) wide with at least [`PTS_MIN`] stops; RANSAC's inlier band `PTS_BAND_SHARE` of
/// the claim, between [`PTS_BAND_MIN`] and [`PTS_BAND`] (m: a band growing with a fat claim lets a wide circle through
/// a few scattered stems confirm it); at least [`PTS_INLIERS`] of the stops on it, filling [`PTS_ARC`] of the 36
/// ten-degree bins about it (120 degrees: three thin stems about a vase aren't one circle), their rms distance from
/// it at most `PTS_RMSE.0 R + PTS_RMSE.1`, its centre within the wider of R and the claim plus [`PTS_SHIFT`] of the
/// axis, and R at most [`PTS_GROW`] times the claim and [`PTS_BELIEVE`] times the site's cap (a wide circle through
/// a bush of thin stems passes every other test). Its noise, [`PTS_SD`] in log R, is the refit circles' (3dtrees
/// research 93 §19).
const PTS_R_MIN: f64 = 0.01;
const PTS_MIN: usize = 12;
const PTS_BAND_SHARE: f64 = 0.25;
const PTS_BAND_MIN: f64 = 0.003;
const PTS_BAND: f64 = 0.02;
const PTS_INLIERS: f64 = 0.5;
const PTS_ARC: usize = 12;
const PTS_RMSE: (f64, f64) = (0.35, 0.002);
const PTS_SHIFT: f64 = 0.02;
const PTS_GROW: f64 = 2.0;
const PTS_BELIEVE: f64 = 2.0;
const PTS_SD: f64 = 0.136;
/// The draws of a tree's volume from its joint posterior ([`Volume`]): its 5th and 95th percentiles to about a tenth
/// of the interval's log width (one standard error), at ~25 ms for a tree of 300 pieces.
const VOLUME_DRAWS: usize = 1000;

/// How one scan sees a circle: its impact and depth directions (in the piece's `e1, e2`) and what its edge on `+n`,
/// on `-n` and its front weigh: the information of its measurement of each, or the pulses that reached it (0:
/// unseen).
#[derive(Debug, Clone, Copy)]
struct Sight {
    n: [f64; 2],
    v: [f64; 2],
    w: [f64; 3],
}

/// The directions of a circle's `(c1, c2, R)` that the sides its scans see leave free, where they leave its radius
/// undetermined (none where R is determined): each side is a measurement of the support function, `n.c + R`, `-n.c
/// + R` or the front's `-v.c + R`, of its weight (research 93 §10: the rank and conditioning of the design matrix).
/// One scan's edge and front leave R trading against the centre (a fork's stem with its other edge behind a crossing
/// branch, research 93 §20).
fn unseen(sights: &[Sight]) -> Vec<Vector3<f64>> {
    let mut m = Matrix3::zeros();
    let mut total = 0.0;
    for s in sights {
        for (k, a) in [[s.n[0], s.n[1], 1.0], [-s.n[0], -s.n[1], 1.0], [-s.v[0], -s.v[1], 1.0]].into_iter().enumerate() {
            if s.w[k] > 0.0 {
                let a = Vector3::from(a);
                m += a * a.transpose() * s.w[k];
                total += s.w[k];
            }
        }
    }
    if total <= 0.0 {
        return vec![Vector3::x(), Vector3::y(), Vector3::z()];
    }
    // R's variance with a negligible anchor of the centre (a centre direction unseen alone leaves R determined),
    // against the evidence's total
    let anchor = Matrix3::from_diagonal(&Vector3::new(1e-6 * total, 1e-6 * total, 0.0));
    let e_r = (m + anchor).try_inverse().map_or(f64::INFINITY, |i| i[(2, 2)].max(0.0) * total);
    if e_r <= SHADOW_SD * SHADOW_SD {
        return Vec::new();
    }
    let e = SymmetricEigen::new(m);
    (0..3).filter(|&i| e.eigenvalues[i] * SHADOW_SD * SHADOW_SD < total).map(|i| e.eigenvectors.column(i).into_owned()).collect()
}

/// A circle's measurement with nothing said along `free` (directions of `(c1, c2, R)`): its information
/// marginalised over them (`I - I r r^T I / r^T I r`, each in turn), and its covariance [`UNSEEN_VAR`] wider
/// along each.
fn marginalise(c: &Circle, free: &[Vector3<f64>]) -> Circle {
    let (mut info, mut h, mut yy, mut cov) = (c.st.info, c.st.h, c.st.yy, c.cov);
    for d in free {
        let r = Vector5::new(d[0], d[1], 0.0, 0.0, d[2]);
        let ir = info * r;
        let q = r.dot(&ir);
        if q > 1e-12 * info.norm() {
            let rh = r.dot(&h);
            info -= ir * ir.transpose() / q;
            h -= ir * (rh / q);
            yy -= rh * rh / q;
        }
        cov += r * r.transpose() * UNSEEN_VAR;
    }
    Circle { st: Stats { info: 0.5 * (info + info.transpose()), h, yy, n: c.st.n }, cov, ..*c }
}

/// What a measurement `info` (of `[c1, c2, t1, t2, R]`) says of R with the rest unknown: the Schur complement
/// `I_RR - I_Ro I_oo^+ I_oR` (a pseudo-inverse: directions it doesn't inform say nothing).
fn info_r(info: &Matrix5<f64>) -> f64 {
    let o = Matrix4::from_fn(|i, j| info[(i, j)]);
    let b = Vector4::from_fn(|i, _| info[(i, 4)]);
    let e = SymmetricEigen::new(o);
    let top = e.eigenvalues.iter().fold(0.0f64, |a, &x| a.max(x));
    let mut s = info[(4, 4)];
    for i in 0..4 {
        let l = e.eigenvalues[i];
        if l > 1e-10 * top && l > 0.0 {
            let p = e.eigenvectors.column(i).dot(&b);
            s -= p * p / l;
        }
    }
    s.max(0.0)
}

/// A group needs this much (wood-weighted) pulses that say something (hits and passes).
const MIN_PULSES: f64 = 30.0;
/// A run's least wood-weighted hits, to be the row's or to count as a stem beside it.
const RUN_HITS: f64 = 10.0;
/// The tube's widening for the lean, a share of the piece's length (a lean of up to about 27°).
const LEAN_TUBE: f64 = 0.25;
/// Rows longer than this (m) are cut into pieces of at most this length.
pub const PIECE_LEN: f64 = 0.3;
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
/// The capacity floor's sd in log R (`tau` itself is a calibration).
const CAP_SD: f64 = 0.3;
/// The fork-share term's degrees of freedom: Student-t, so a child far off its share (a basal shoot measured fat, a
/// clump's envelope) is discounted rather than moving its parent (research 93 §17 review: a trunk with basal
/// shoots, 26 cm at 4 degrees of freedom, 19.5 at 2 and 18.9 at 1, against RCT's 15.8 and a lower bound's truncation
/// at ~19).
const FORK_DF: f64 = 2.0;
/// Where thin wood's reflectance shift is full (R at most this, m) and where it is none (from this).
const THIN_FULL: f64 = 0.01;
const THIN_NONE: f64 = 0.03;
/// The censored solve: at most `EP_ROUNDS` rounds, each site moved `EP_DAMP` of the way (halved, down to
/// `EP_DAMP_MIN`, each time its step turns back, and grown back by `EP_REGROW` while it doesn't; a fork weight likewise
/// from a full step), until no linearisation moves, and no site or fork weight would move undamped, by more than
/// `EP_TOL` (in units of its piece's sd; a fork weight by more than that itself) (the bench's
/// largest tree takes ~200 rounds at 1e-4 and fewer at 1e-3, with the same radii to the bench's precision; a round is one
/// elimination, ~3 us per piece; 3dtrees research 93 §19).
const EP_ROUNDS: usize = 300;
const EP_DAMP: f64 = 0.5;
/// The least damping of a site whose steps keep turning back, and how fast it grows back (up to `EP_DAMP`, a fork
/// weight's to a full step) while its steps keep their sign.
const EP_DAMP_MIN: f64 = 0.02;
const EP_REGROW: f64 = 1.5;
const EP_TOL: f64 = 1e-3;
const EP_NODES: usize = 64;
/// The rounds of the first solve, which only places the routed pieces' context and parents' ends: its bounds'
/// sites a few steps in, not settled (each round is one elimination).
const FIRST_ROUNDS: usize = 5;

/// One tree's fit, per row or per piece (the tree's nodes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fit {
    /// The fitted axis (world coordinates: the claim moved by the centre and turned by the lean) and radius, with
    /// the posterior sd of log R widened by its basis's factor ([`WIDEN`]). A routed (RUNS or HULL) piece's follow its
    /// exact fit's continuing stem, or where it has none accepted, the prior's cut by its bounds. A row's: the line
    /// from its first piece's start to its last's end, its pieces' root mean square R, and the sd of that R's log,
    /// linearised from the chain's joint covariance with each piece's log R widened by its own basis's factor (a row of
    /// mixed bases piece by piece, not by its own `basis`).
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r: Vec<f64>,
    pub sd_log_r: Vec<f64>,
    /// The centre at the middle and the lean (per m along the axis), along `e1, e2`.
    pub centre: Vec<[f64; 2]>,
    pub lean: Vec<[f64; 2]>,
    pub e1: Vec<Point>,
    pub e2: Vec<Point>,
    /// The bounds on R that entered the tree, NaN where none did: a piece's where no circle of its own does (a
    /// routed piece's without its detection bound, which can span several stems; a point circle's piece's upper
    /// alone), so its radius is held in them but where they cross or the veto narrows it ([`diag::PINNED`]); a row's,
    /// its pieces' root mean square as its R is (a piece without a lower bound at 0, without an upper unbounded): the
    /// tightest lower bound from detection (a woody run's span) and upper (passes beside or through).
    pub r_lo: Vec<f64>,
    pub r_hi: Vec<f64>,
    /// The chance of wood under the branches' chain, from each piece's existence evidence (a row's: its pieces' mean,
    /// the share of it that is wood).
    pub p_wood: Vec<f64>,
    /// [`flag`] bits (a row's: its pieces'): 0, its own circle entered the tree; RUNS or HULL, its exact fit did,
    /// where `k_stems` > 0 (else its bounds); BOUNDS or POROUS, its bounds.
    pub flags: Vec<u8>,
    /// The groups (scans) that gave measurements (a row's: the most).
    pub n_scans: Vec<u32>,
    /// The stems a routed piece's exact fit chose that entered the tree (0: none, or its fit refused; 3dtrees research
    /// 93 §18; a row's: the most).
    pub k_stems: Vec<u32>,
    /// The fit's diagnostics ([`diag`] bits; a row's: its pieces'), and three numbers: `data_share`, the share of log
    /// R's posterior precision its own data give (its measurement's, R's information with the centre and lean
    /// unknown, and its bounds' EP site; 0-1, NaN where that site's precision came out negative); `contested`, the
    /// share of its stops claimed by context or another node (NaN with none); `deleted`, the share of its pulses
    /// with echoes the export deleted (an exact stem's: its piece's). A row's: its pieces' means. The solve's
    /// `converged` is the tree's.
    pub diag: Vec<u16>,
    pub data_share: Vec<f64>,
    pub contested: Vec<f64>,
    pub deleted: Vec<f64>,
    /// The site's D/H cap on R (`PriorParams::dh` times the reach from its start to the tips over 2; NaN without a
    /// cap, or where a circle, its own or its row's point circle, measures it; a row's: its first piece's, the widest).
    pub r_cap: Vec<f64>,
    /// The ray veto's audit of its fitted circle: the pulses that pass through it, and their share of its stops and
    /// passes (NaN with none; a row's: its pieces' summed). [`diag::REFUTED`] says where they refute it.
    pub n_pass: Vec<u32>,
    pub pass_share: Vec<f64>,
    /// How its radius is known ([`basis`]).
    pub basis: Vec<u8>,
}

/// A tree's volume (m^3), over its pieces (the served wood: a routed piece's further stems aside): the point, from
/// each piece's posterior R (the drawn geometry's: per-piece medians); the mean and the 5th and 95th percentiles of
/// [`VOLUME_DRAWS`] draws from the tree's joint posterior (the solve's exact covariance, each node's sd widened by
/// [`WIDEN`] for its basis, each draw held inside its piece's bounds): the mean is the volume of record (3dtrees
/// ticket 109: a sum of medians can lie under its own q05); and the mean split by each piece's [`basis`], `by_basis`
/// indexed as [`WIDEN`] is, from MEASURED (the draws' mean of its pieces' volume, so the parts add to the mean).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Volume {
    pub point: f64,
    pub mean: f64,
    pub q05: f64,
    pub q95: f64,
    pub by_basis: [f64; 5],
}

/// The fit of one tree: per row (in the input's order) and per node, with each node's row; the nodes are the pieces,
/// then the routed pieces' further stems, each with the piece it came from (`piece_src`) and its stem there
/// (`piece_stem`, 0 for a piece's own). And its echoes classified.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportTree {
    pub rows: Fit,
    pub pieces: Fit,
    pub piece_row: Vec<u32>,
    pub piece_src: Vec<u32>,
    pub piece_stem: Vec<u32>,
    /// Whether the censored solve settled within [`EP_ROUNDS`].
    pub converged: bool,
    pub echoes: Echoes,
    pub volume: Volume,
}

/// A pulse in a piece's cross-section: impact, axial position from the piece's middle, depth of its stop (NaN:
/// none in the tube), footprint width, its stop's reflectance (dB) and whether it was its pulse's only return, and
/// whether it was blocked before the tube; and which pulse it is (in its scan's index) and which of its echoes
/// stopped it.
#[derive(Debug, Clone, Copy)]
struct Rec {
    b: f64,
    s: f64,
    depth: f64,
    /// Its stop's place across the axis (along `e1, e2`; NaN: no stop in the tube).
    xy: [f64; 2],
    w: f64,
    refl: f64,
    single: bool,
    blocked: bool,
    lost: bool,
    pulse: u32,
    echo: u8,
}

impl Rec {
    /// Whether it stopped in the tube.
    fn is_stop(&self) -> bool {
        !self.blocked && self.depth.is_finite()
    }
}

/// One (piece, scan) group: its scan, the median impact and depth directions (in the row's `e1, e2`) and its
/// pulses.
struct Group {
    scan: u32,
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

/// Which side a [`Bound`] holds R from, and on what evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Detection: a woody run's span.
    Lower,
    /// Passes beside or through.
    Upper,
    /// The capacity floor: wood that was hit fills at least `tau` of the footprint.
    Floor,
}

/// A censored measurement of a piece's log R, `log R <= eta` (upper) or `>= eta`, softened by a Gaussian of sd
/// `sd`: the probit `Phi(±(eta - log R) / sd)`, log-concave.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bound {
    eta: f64,
    sd: f64,
    side: Side,
}

impl Bound {
    /// Its log probit at log R `x`.
    fn ln_p(&self, x: f64) -> f64 {
        log_ndtr_value(if self.side == Side::Upper { self.eta - x } else { x - self.eta } / self.sd)
    }
}

/// What became of a piece's own summary: the one decision behind its flags.
#[derive(Debug, Clone, Copy)]
enum Outcome {
    /// Its circle enters the tree.
    Measured(Circle),
    /// Several stems within the claim (`runs`) or a hull (`hull`): fitted by the exact likelihood ([`routed`]); its
    /// circle isn't used. With no fit accepted, its radius is the prior's cut by its bounds.
    Routed { runs: bool, hull: bool },
    /// No circle is determined: the radius is the prior cut by the bounds.
    Bounded,
    /// Pulses, but no opaque run: existence, and the radius the prior cut by any bounds.
    Porous,
}

impl Outcome {
    /// Its [`flag`] bits.
    fn flags(&self) -> u8 {
        match *self {
            Outcome::Measured(_) => 0,
            Outcome::Routed { runs, hull, .. } => u8::from(runs) * flag::RUNS | u8::from(hull) * flag::HULL,
            Outcome::Bounded => flag::BOUNDS,
            Outcome::Porous => flag::POROUS,
        }
    }
}

/// What a piece's own pulses say: what became of its summary, its front misfit, runs and scans, its bounds
/// (where it isn't measured; they enter the tree where no circle does) and existence evidence (log odds); where it is
/// routed, its seeds (strongest first) and, once fitted, its stem's exact summary; and its groups' pulses that reached
/// its tube.
struct Summary {
    outcome: Outcome,
    chi2_front: f64,
    n_runs: u32,
    n_scans: u32,
    bounds: Vec<Bound>,
    log_odds: f64,
    seeds: Vec<routed::Seed>,
    exact: Option<Circle>,
    /// Its groups' pulses that reached its tube (not blocked before it): its stops, for the classification, and with
    /// its passes, for the ray veto's audit of the fitted circle.
    seen: Vec<Group>,
    /// An exact stem whose radius the sides its scans see leave undetermined (research 93 §20): its message says
    /// nothing along what they leave free, and its own bounds (in `bounds`) enter the tree with it.
    shadowed: bool,
    /// Its row's point circle, where the rays gave none of its pieces a circle of their own and the veto doesn't
    /// refute it ([`point_circle`]): it enters the tree with the piece's upper bounds beside it (its lower bound and
    /// floor, from the same stops, dropped).
    points: Option<Circle>,
    diag: DiagIn,
}

/// What a node's diagnostics need beyond its fit, filled where each is known: the tube its pulses were taken in (m;
/// an exact stem's, its fit's), their median footprint (NaN: none), whether one scan or one view supplies them, the
/// share with deleted echoes, and the stops its exact fit's context owned (in the seeds' windows, by the stem they
/// meet first).
#[derive(Debug, Clone, Copy)]
struct DiagIn {
    tube: f64,
    w: f64,
    one_sided: bool,
    deleted: f64,
    n_ctx: u32,
    /// Its own circle reached its tube's edge, and wasn't used ([`diag::TUBE_EDGE`]).
    edge: bool,
}

impl Summary {
    /// The circle of its own pulses, where one enters the tree: its summary's, or the exact fit's of a routed piece.
    fn own(&self) -> Option<&Circle> {
        match &self.outcome {
            Outcome::Measured(c) => Some(c),
            _ => self.exact.as_ref(),
        }
    }

    /// The circle that enters the tree: its own, or its row's point circle.
    fn used(&self) -> Option<&Circle> {
        self.own().or(self.points.as_ref())
    }

    /// Whether its bounds enter the tree: no circle of its own, or one that leaves its radius undetermined.
    fn bounded(&self) -> bool {
        self.own().is_none() || self.shadowed
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
/// much darker, and thin wood reads [`JointParams::refl_thin`] darker still (in full from [`THIN_FULL`] down, none
/// from [`THIN_NONE`] up, linear in log R between). An echo of unknown reflectance: 0.
fn llr(rec: &Rec, r: f64, p: &JointParams) -> f64 {
    if !rec.refl.is_finite() {
        return 0.0;
    }
    let (shift, var) = if rec.single { (0.0, 0.0) } else { (p.refl_multi, p.refl_multi_sd * p.refl_multi_sd) };
    let ramp = ((THIN_NONE / r).ln() / (THIN_NONE / THIN_FULL).ln()).clamp(0.0, 1.0);
    let thin = 10.0 * (2.0 * r / rec.w).clamp(1e-3, 1.0).log10() + p.refl_thin * ramp;
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
    for (si, sc) in scans.iter().enumerate() {
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
            let (nn, vv) = ([dot(&n, &e1), dot(&n, &e2)], [dot(&v, &e1), dot(&v, &e2)]);
            ns.push(nn);
            vs.push(vv);
            let echo = first.map_or(u8::MAX, |i| i.min(254) as u8);
            let xy = [bimp * nn[0] + depth * vv[0], bimp * nn[1] + depth * vv[1]];
            recs.push(Rec { b: bimp, s: s_ax - 0.5 * len, depth, xy, w: p.footprint(t), refl, single: echoes.len() == 1, blocked, lost: sc.lost_echoes(pulse), pulse, echo });
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
        out.push(Group { scan: si as u32, n: unit(ns), v: unit(vs), recs });
    }
    out
}

/// A pulse's fate against a band of depth `[lo, hi]`: a stop in it is a hit, behind it a pass (it went through
/// where the band is), in front of it blocked (it never reached it); a pulse with no stop in the tube passed, and
/// one blocked before the tube is blocked. Every gate of the profile is this one with its own band.
fn gate(r: &Rec, lo: f64, hi: f64) -> Fate {
    if r.blocked || r.depth < lo {
        Fate::Blocked
    } else if !r.depth.is_finite() || r.depth > hi {
        Fate::Pass
    } else {
        Fate::Hit
    }
}

/// Where a pulse meets the circle `x` (`[c1, c2, t1, t2, R]`) in its group's profile: its impact from the circle's
/// centre at its `s`, and the depth of the circle's front arc there (its apex beyond the silhouette).
fn arc(x: &Vector5<f64>, g: &Group, r: &Rec) -> (f64, f64) {
    let (cx, cy) = (x[0] + r.s * x[2], x[1] + r.s * x[3]);
    let bc = r.b - (g.n[0] * cx + g.n[1] * cy);
    (bc, g.v[0] * cx + g.v[1] * cy - (x[4] * x[4] - bc * bc).max(0.0).sqrt())
}

/// A pulse's fate: on the first pass from its echo alone; on the second, against the circle `fit`'s front arc
/// (a stop beside the circle never reached it).
fn fate(r: &Rec, g: &Group, fit: Option<&Vector5<f64>>) -> Fate {
    let Some(x) = fit else { return gate(r, f64::NEG_INFINITY, f64::INFINITY) };
    let (bc, front) = arc(x, g, r);
    let tol = DEPTH_TOL + 0.3 * x[4];
    match gate(r, front - tol, front + tol) {
        Fate::Hit if bc.abs() >= x[4] + DEPTH_TOL => Fate::Blocked,
        f => f,
    }
}

/// The slope of `b` against `s` (a line through the medians of six slices of `s`) and its variance; None with
/// fewer than 10 points or three slices, or no spread in `s`.
fn lean_line(pts: &mut [(f64, f64)], floor: f64) -> Option<(f64, f64)> {
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
    Some((slope, (rss / (k - 2.0)).max(floor * floor) / sxx))
}

/// A run of mostly-hit bins, cut where its depth jumps: its impact bounds, wood-weighted hits, bins, and the range
/// of its bins' median depths.
#[derive(Debug, Clone, Copy)]
struct Run {
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
fn profile_runs(bp: &[f64], fates: &[Fate], wt: &[f64], depth: &[f64], bw: f64, bridge: f64) -> Vec<Run> {
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
            out.push(Run {
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
fn dist(q: &Run, c: f64) -> f64 {
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
    let eligible = |q: &Run| q.hits >= RUN_HITS && q.bins >= 2 && q.hi > reach.0 && q.lo < reach.1 && q.dhi > depths.0 && q.dlo < depths.1;
    // the lean across the beam: a line through the hits' centre against s, first of the hits near the claim, then
    // of the chosen run's own; the profile is cut again with the second
    let mut near: Vec<(f64, f64)> = (0..depth.len()).filter(|&i| woody(i) && (g.recs[i].b - centre).abs() < 1.5 * r_claim + 0.01 && depth[i] > depths.0 && depth[i] < depths.1).map(|i| (g.recs[i].s, g.recs[i].b)).collect();
    let mut lean = lean_line(&mut near, p.model_sd);
    let (mut bp, mut step, mut bw, mut chosen, mut aim) = (Vec::new(), 0.0, BIN_MIN, None, centre);
    for round in 0..2 {
        let k = lean.map_or(0.0, |l| l.0);
        bp = g.recs.iter().map(|r| r.b - k * r.s).collect();
        // bins no finer than the impacts' spacing (a scan's columns can align with the row: then every pulse along
        // it shares a column's impact)
        (step, bw) = spacing(&bp, fates);
        let ps: Vec<Run> = profile_runs(&bp, fates, wt, &depth, bw, w_med).into_iter().filter(eligible).collect();
        chosen = ps.iter().min_by(|x, y| dist(x, aim).total_cmp(&dist(y, aim))).copied();
        let Some(pc) = chosen else { break };
        if round == 0 {
            let mut own: Vec<(f64, f64)> = (0..depth.len()).filter(|&i| woody(i) && bp[i] >= pc.lo && bp[i] <= pc.hi && depth[i] >= pc.dlo - DEPTH_TOL && depth[i] <= pc.dhi + DEPTH_TOL).map(|i| (g.recs[i].s, g.recs[i].b)).collect();
            match lean_line(&mut own, p.model_sd) {
                Some(l) => (lean, aim) = (Some(l), 0.5 * (pc.lo + pc.hi)),
                None => break,
            }
        }
    }
    let Some(pc) = chosen else { return out };
    out.run = true;
    // fates relative to this run: a stop well behind its depths passed it, one well in front never reached it
    let local: Vec<Fate> = fates.iter().zip(&g.recs).map(|(&f, r)| if f == Fate::Hit { gate(r, pc.dlo - DEPTH_TOL, pc.dhi + DEPTH_TOL) } else { f }).collect();
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
                edges[side] = Some((sign * (out_hit + 0.5 * step - app), step * step / 12.0 + p.model_sd * p.model_sd));
            }
            continue;
        }
        if pts.iter().filter(|q| q.2).map(|q| q.3).sum::<f64>() < 1.5 {
            continue;
        }
        // an edge that ran out of its window is no edge
        if let Some((e, var)) = edge(&pts, at - app, p_in, p).filter(|&(e, _)| e + app >= inner && e + app <= at + 4.0 * sd_max + bw) {
            // (plus the impacts' quantisation: aligned columns place an edge only between them)
            edges[side] = Some((sign * e, var + step * step / 12.0 + p.model_sd * p.model_sd));
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
    let floor = (p.sigma_r * p.sigma_r + (p.rough * r0).powi(2)).max(p.model_sd * p.model_sd);
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
        out.meas.push(([g.v[0], g.v[1], 0.0, 0.0, -1.0], alpha, s2 * (1.0 / m + at_mid) + p.model_sd * p.model_sd));
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
        let mid = g.v[0] * (x[0] + r.s * x[2]) + g.v[1] * (x[1] + r.s * x[3]);
        gate(r, mid - rr - tol, mid + rr + tol)
    }).collect();
    let wt = weights(g, &fates, rr, p);
    let (k, centre) = (g.n[0] * x[2] + g.n[1] * x[3], g.n[0] * x[0] + g.n[1] * x[1]);
    let bp: Vec<f64> = g.recs.iter().map(|r| r.b - k * r.s).collect();
    // depth from the circle's front arc, so its own face is flat and a jump is another body
    let depth: Vec<f64> = g.recs.iter().map(|r| r.depth - arc(x, g, r).1).collect();
    let (_, bw) = spacing(&bp, &fates);
    let reach = (centre - r_claim - 0.01, centre + r_claim + 0.01);
    let w_med = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
    let ps: Vec<Run> = profile_runs(&bp, &fates, &wt, &depth, bw, w_med).into_iter().filter(|q| q.hits >= RUN_HITS && q.bins >= 2 && q.hi > reach.0 && q.lo < reach.1).collect();
    let Some(pc) = ps.iter().min_by(|x, y| dist(x, centre).total_cmp(&dist(y, centre))).copied() else { return (0, false) };
    // the stems: runs at least a fifth of the circle's own
    (ps.iter().filter(|q| 5.0 * q.hits >= pc.hits).count(), pc.hi - pc.lo > 4.0 * x[4] + 0.04)
}

/// Weighted least squares for `[c1, c2, t1, t2, R]` from the measurements' statistics, the centre and lean weakly
/// anchored on the claim (`r_claim`, `lean_sd`; R free), with the covariance inflated by the misfit (reduced
/// chi-square on the measurements' rank, where they over-determine it); None where R isn't determined (its sd
/// above half its value and the floor `model_sd`).
fn solve_row(st: &Stats, r_claim: f64, lean_sd: f64, model_sd: f64) -> Option<Circle> {
    let sc = r_claim + 0.003;
    let al = 1.0 / (lean_sd * lean_sd);
    let anchor = Matrix5::from_diagonal(&Vector5::new(1.0 / (sc * sc), 1.0 / (sc * sc), al, al, 0.0));
    let m = st.info + anchor;
    let eig = SymmetricEigen::new(m);
    if !(eig.eigenvalues.min() > 1e-12 * eig.eigenvalues.max()) {
        return None;
    }
    let mut cov = m.try_inverse()?;
    let x = cov * st.h;
    if !(x[4] > 0.0 && cov[(4, 4)].sqrt() <= 0.5 * x[4] + model_sd) {
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

/// How far across the beam from the claim's axis a stop counts as near it (m).
fn near_claim(r_claim: f64) -> f64 {
    1.5 * r_claim + 0.01
}

/// A group's profile at the claim's depth, for what it says beyond a circle (research 93 §17): each pulse's fate
/// there (a stop at the claim's depth a hit, behind it or none in the tube a pass, in front of it blocked), which
/// hits are woody, the impacts with the woody stops' lean removed, the median footprint, the pulses' spacing (from
/// their density over the patch they cover, impact by axial position) and the pulses in slices one spacing long
/// along the piece, each slice sorted by impact.
struct Profile<'a> {
    g: &'a Group,
    fates: Vec<Fate>,
    woody: Vec<bool>,
    bp: Vec<f64>,
    w: f64,
    step: f64,
    slices: Vec<Vec<usize>>,
}

impl<'a> Profile<'a> {
    fn of(g: &'a Group, r_claim: f64, p: &JointParams) -> Self {
        let n = g.recs.len();
        let fates: Vec<Fate> = g.recs.iter().map(|r| gate(r, -2.0 * r_claim - DEPTH_TOL, r_claim + DEPTH_TOL)).collect();
        let woody: Vec<bool> = (0..n).map(|i| fates[i] == Fate::Hit && p_wood(&g.recs[i], r_claim, p) >= 0.5).collect();
        let w = median(&mut g.recs.iter().map(|r| r.w).collect::<Vec<_>>());
        let mut near: Vec<(f64, f64)> = (0..n).filter(|&i| woody[i] && g.recs[i].b.abs() < near_claim(r_claim)).map(|i| (g.recs[i].s, g.recs[i].b)).collect();
        let k = lean_line(&mut near, p.model_sd).map_or(0.0, |l| l.0);
        let bp: Vec<f64> = g.recs.iter().map(|r| r.b - k * r.s).collect();
        let inf: Vec<usize> = (0..n).filter(|&i| fates[i] != Fate::Blocked).collect();
        let span = |v: &mut dyn Iterator<Item = f64>| v.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
        let (b0, b1) = span(&mut inf.iter().map(|&i| bp[i]));
        let (t0, t1) = span(&mut inf.iter().map(|&i| g.recs[i].s));
        let step = if inf.len() > 1 { ((b1 - b0).max(BIN_MIN) * (t1 - t0).max(BIN_MIN) / inf.len() as f64).sqrt() } else { w };
        let t0 = if t0.is_finite() { t0 } else { 0.0 };
        let slice = |i: usize| ((g.recs[i].s - t0) / step).floor() as i64;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&i, &j| slice(i).cmp(&slice(j)).then(bp[i].total_cmp(&bp[j])));
        let slices = order.chunk_by(|&i, &j| slice(i) == slice(j)).map(|s| s.to_vec()).collect();
        Profile { g, fates, woody, bp, w, step, slices }
    }

    /// The median impact of the `member` pulses near the claim, None where there are none.
    fn centre(&self, member: &[bool], r_claim: f64) -> Option<f64> {
        let mut v: Vec<f64> = (0..self.bp.len()).filter(|&i| member[i] && self.g.recs[i].b.abs() < near_claim(r_claim)).map(|i| self.bp[i]).collect();
        (!v.is_empty()).then(|| median(&mut v))
    }

    /// Per slice, the run of `member` pulses (contiguous in impact) about the one nearest `c` within the claim: its
    /// half-width, and half the gap between the pulses either side of it where both are passes. With `through`, a
    /// slice with no member there gives half the gap between the pulses either side of `c` where both are passes
    /// (pulses through the core). A side with no recorded pulse gives no gap: a slice can be sparse and the tube
    /// ends, so nothing recorded there is no evidence of sky.
    fn runs(&self, member: &[bool], c: f64, r_claim: f64, through: bool) -> (Vec<f64>, Vec<f64>) {
        let bp = &self.bp;
        // a pass beside the run is a pulse that met nothing in the tube: one that stopped behind the claim's depth
        // window may have stopped on this wood, off the claim's axis in depth (a thin claim in a fat stem, a point
        // circle behind it), and read as a pass it bounded that wood at millimetres (3dtrees research 127)
        let pass = |i: usize| self.fates[i] == Fate::Pass && !self.g.recs[i].depth.is_finite();
        let (mut lo_s, mut hi_s) = (Vec::new(), Vec::new());
        for sl in &self.slices {
            let j0 = (0..sl.len()).filter(|&j| member[sl[j]] && (bp[sl[j]] - c).abs() < r_claim + self.w).min_by(|&x, &y| (bp[sl[x]] - c).abs().total_cmp(&(bp[sl[y]] - c).abs()));
            let Some(j0) = j0 else {
                let k = sl.partition_point(|&i| bp[i] < c);
                if through && k > 0 && k < sl.len() && pass(sl[k - 1]) && pass(sl[k]) {
                    hi_s.push(0.5 * (bp[sl[k]] - bp[sl[k - 1]]));
                }
                continue;
            };
            let (mut a, mut z) = (j0, j0);
            while a > 0 && member[sl[a - 1]] {
                a -= 1;
            }
            while z + 1 < sl.len() && member[sl[z + 1]] {
                z += 1;
            }
            lo_s.push(0.5 * (bp[sl[z]] - bp[sl[a]]));
            if a > 0 && z + 1 < sl.len() && pass(sl[a - 1]) && pass(sl[z + 1]) {
                hi_s.push(0.5 * (bp[sl[z + 1]] - bp[sl[a - 1]]));
            }
        }
        (lo_s, hi_s)
    }

    /// Its bounds on R from detection and passes (see the module documentation): a slice's numbers are valid bounds
    /// unless clutter joins the run or an echo was lost inside it, so the group's are their medians over at least
    /// three slices (looser than the tightest, robust to a minority of corrupted ones). None without woody stops near
    /// the claim: then no centre is known for the passes to straddle. With a `captor` (the piece's own circle, refused
    /// at its tube's edge: [`diag::TUBE_EDGE`]) whose silhouette, as this scan sees it, holds the woody centre, no
    /// lower bound: the run about it is the captor's, and its span says nothing of the piece.
    fn bounds(&self, r_claim: f64, p: &JointParams, captor: Option<&Vector5<f64>>) -> Vec<Bound> {
        let Some(c) = self.centre(&self.woody, r_claim) else { return Vec::new() };
        let (mut lo_s, mut hi_s) = self.runs(&self.woody, c, r_claim, true);
        let (app, floor) = (p.app(self.w), 0.5 * p.tau * self.w);
        let sd = p.edge * self.w / std::f64::consts::SQRT_2;
        let captured = captor.is_some_and(|x| (c - (self.g.n[0] * x[0] + self.g.n[1] * x[1])).abs() < x[4] + app);
        let mut out = Vec::new();
        if lo_s.len() >= 3 && !captured {
            let lo_r = median(&mut lo_s) - app;
            if lo_r > floor {
                out.push(Bound { eta: lo_r.ln(), sd: sd / lo_r, side: Side::Lower });
            }
        }
        if hi_s.len() >= 3 {
            let hi_r = (median(&mut hi_s) - app).max(floor);
            out.push(Bound { eta: hi_r.ln(), sd: (sd * sd + self.step * self.step / 12.0).sqrt() / hi_r, side: Side::Upper });
        }
        out
    }

    /// The footprint of its woody stops near the claim, for the capacity floor (None: there are none).
    fn woody_w(&self, r_claim: f64) -> Option<f64> {
        self.centre(&self.woody, r_claim).map(|_| self.w)
    }

    /// Its existence log odds of wood against foliage (see the module documentation), over the beam's oversampling:
    /// the core's stops by their echo character, and its pulses by opacity, a stop for wood and a pass against it,
    /// the passes' net at most one sure echo (research 93 §11.1: leaky is weak proof of foliage, the core may be
    /// misplaced). The core is found from the runs of any stop, blind to echo character (a core of woody runs would
    /// hold woody echoes by construction), else it is the claim's middle half.
    fn log_odds(&self, r_claim: f64, p: &JointParams) -> f64 {
        let hit: Vec<bool> = self.fates.iter().map(|&f| f == Fate::Hit).collect();
        let ch = self.centre(&hit, r_claim).unwrap_or(0.0);
        let (mut lo_e, _) = self.runs(&hit, ch, r_claim, false);
        let (core, half) = if lo_e.is_empty() { (0.0, 0.5 * r_claim) } else { (ch, median(&mut lo_e).max(0.25 * self.w)) };
        let (on, off) = ((p.stop_wood / p.stop_leaf).ln(), ((1.0 - p.stop_wood) / (1.0 - p.stop_leaf)).ln());
        let (mut echo, mut opacity) = (0.0, 0.0);
        for i in (0..self.bp.len()).filter(|&i| (self.bp[i] - core).abs() <= half) {
            match self.fates[i] {
                Fate::Hit => (echo, opacity) = (echo + llr(&self.g.recs[i], r_claim, p), opacity + on),
                Fate::Pass => opacity += off,
                Fate::Blocked => {}
            }
        }
        // a scan's verdict on a piece counts at most as one sure echo: beyond the beam's oversampling, foliage is
        // dispersed at 10-40 cm (research 93 §15.6), the piece's scale, so its pulses there are one observation
        let m = (self.w / self.step).powi(2).max(1.0);
        p.exist_scale * ((echo + opacity.max(-LLR_CLIP)) / m).clamp(-LLR_CLIP, LLR_CLIP)
    }
}

/// Whether one scan supplies at least [`ONE_SCAN`] of a piece's informative pulses (not blocked before its tube), or
/// every pair of the scans with [`VIEW_SHARE`] of them looks along directions within [`VIEW_MIN`] degrees of each
/// other ([`diag::ONE_SIDED`]; false with none).
fn one_sided(gs: &[Group]) -> bool {
    let n: Vec<f64> = gs.iter().map(|g| g.recs.iter().filter(|r| !r.blocked).count() as f64).collect();
    let all: f64 = n.iter().sum();
    if all == 0.0 {
        return false;
    }
    if n.iter().any(|&k| k >= ONE_SCAN * all) {
        return true;
    }
    let views: Vec<[f64; 2]> = gs.iter().zip(&n).filter(|(_, &k)| k >= VIEW_SHARE * all).map(|(g, _)| g.v).collect();
    let cos = VIEW_MIN.to_radians().cos();
    views.iter().all(|a| views.iter().all(|b| a[0] * b[0] + a[1] * b[1] >= cos))
}

/// What one piece's pulses say: what becomes of its summary (from its circle's second pass), its runs, scans and
/// pairs, its front's stops' misfit about the circle, its existence evidence, and where it isn't measured its bounds
/// and capacity floor (and where it is routed, its seeds).
#[allow(clippy::too_many_arguments)]
fn summarise_piece(scans: &[&ScanIndex], a: &Point, b: &Point, frame: Frame, r_claim: f64, p: &JointParams, lean_sd: f64) -> Summary {
    let tube = window(r_claim) + LEAN_TUBE * frame.3;
    let gs = groups(scans, a, b, frame, tube, p);
    let n_pairs = gs.iter().map(|g| g.recs.len()).sum::<usize>() as u32;
    let profiles: Vec<Profile> = gs.iter().map(|g| Profile::of(g, r_claim, p)).collect();
    let log_odds = profiles.iter().map(|q| q.log_odds(r_claim, p)).sum();
    let w = if n_pairs > 0 { median(&mut gs.iter().flat_map(|g| g.recs.iter().map(|r| r.w)).collect::<Vec<_>>()) } else { f64::NAN };
    let deleted = if n_pairs > 0 { gs.iter().map(|g| g.recs.iter().filter(|r| r.lost).count()).sum::<usize>() as f64 / f64::from(n_pairs) } else { f64::NAN };
    let diag = DiagIn { tube, w, one_sided: one_sided(&gs), deleted, n_ctx: 0, edge: false };
    let mut out = Summary { outcome: Outcome::Bounded, chi2_front: f64::NAN, n_runs: 0, n_scans: 0, bounds: Vec::new(), log_odds, seeds: Vec::new(), exact: None, seen: Vec::new(),
        shadowed: false, points: None, diag };
    out.outcome = outcome(&gs, r_claim, p, lean_sd, &mut out);
    // a circle reaching its tube's edge, past which the evidence is cut off, is a capture (by a neighbour's or the
    // ground's mass) or a drift (onto a leaf clump), not a measurement: the piece is bounded, but not by the woody run
    // the captor spans ([`diag::TUBE_EDGE`])
    let captor = match out.outcome {
        Outcome::Measured(c) if reaches_edge(&c.x, 0.5 * frame.3, tube) => Some(c.x),
        _ => None,
    };
    if captor.is_some() {
        (out.outcome, out.diag.edge) = (Outcome::Bounded, true);
    }
    if !matches!(out.outcome, Outcome::Measured(_)) {
        out.bounds = profiles.iter().flat_map(|q| q.bounds(r_claim, p, captor.as_ref())).collect();
        match profiles.iter().filter_map(|q| q.woody_w(r_claim)).reduce(f64::max) {
            Some(w) => out.bounds.push(Bound { eta: (0.5 * p.tau * w).ln(), sd: CAP_SD, side: Side::Floor }),
            // no woody stop near the claim gives a centre for the passes to straddle: passes through the claim's
            // axis alone bound it (a veto about an axis off the wood would cut real wood: research 106, 103 §20)
            None => out.bounds.extend(veto(&gs, tube)),
        }
    }
    if let Outcome::Routed { .. } = out.outcome {
        out.seeds = routed::seeds(&gs, r_claim, p, lean_sd);
    }
    // the pulses that reached the tube: the stops, for the classification once the tree is solved (a routed piece's
    // come from its exact fit, where it has one), and with the passes, for the veto's audit of the fitted circle
    out.seen = gs.into_iter().map(|g| Group { recs: g.recs.into_iter().filter(|r| !r.blocked).collect(), ..g }).collect();
    out
}

/// A row's point circle (3dtrees research 111: the refit's measurement, research 52 and 62), `[c1, c2, 0, 0, R]`: RANSAC
/// on its stops' places across its axis `xy`, then least squares on the inliers, accepted on the tests of
/// [`PTS_R_MIN`] and after for a row claimed `r0` wide whose circle may be at most `believe` wide (m; NaN: no cap).
fn point_circle(xy: &[[f64; 2]], r0: f64, believe: f64) -> Option<Vector5<f64>> {
    if xy.len() < PTS_MIN {
        return None;
    }
    let band = (PTS_BAND_SHARE * r0).min(PTS_BAND).max(PTS_BAND_MIN);
    let ransac = RansacCircleParams { threshold: band, iterations: 200, min_radius: 0.002, max_radius: 1.0, seed: 0 };
    let (cx, cy, r, inl) = fit_circle_ransac(xy, &ransac).ok()?;
    let (on, rmse, arc) = ring(xy, &inl, cx, cy, r);
    if on < PTS_INLIERS * xy.len() as f64 || !(r.is_finite() && r > 0.0) || r > PTS_GROW * r0 || r > believe {
        return None;
    }
    (cx.hypot(cy) <= r.max(r0) + PTS_SHIFT && rmse <= PTS_RMSE.0 * r + PTS_RMSE.1 && arc >= PTS_ARC).then(|| Vector5::new(cx, cy, 0.0, 0.0, r))
}

/// A circle `(cx, cy, r)`'s fit to the stops `xy` whose `inliers` it is: how many there are, their rms misfit about it
/// and the ten-degree bins they fill (a point circle's tests, [`point_circle`], and a base section's).
fn ring(xy: &[[f64; 2]], inliers: &[bool], cx: f64, cy: f64, r: f64) -> (f64, f64, usize) {
    let on: Vec<[f64; 2]> = xy.iter().zip(inliers).filter(|q| *q.1).map(|(q, _)| [q[0] - cx, q[1] - cy]).collect();
    let rmse = (on.iter().map(|q| (q[0].hypot(q[1]) - r).powi(2)).sum::<f64>() / on.len().max(1) as f64).sqrt();
    let mut bins = [false; 36];
    for q in &on {
        bins[(((q[1].atan2(q[0]).to_degrees() + 180.0) / 10.0) as usize).min(35)] = true;
    }
    (on.len() as f64, rmse, bins.iter().filter(|&&b| b).count())
}

/// The stops in row `k`'s pieces' tubes (their places across its axis) that lie nearer its claimed axis than any
/// other the tree or its context claims, as the refit took each of RCT's points to its nearest cylinder: a neighbour's
/// stem or the tree's own next row doesn't lend the row its stops.
fn own_stops(k: usize, rows: &Rows, context: &[(Point, Point, f64)], nd: &Nodes, sums: &[Summary]) -> Vec<[f64; 2]> {
    let (k0, k1) = (nd.first[k], nd.first[k + 1]);
    let owns = owner(k, rows, context, sums[k0..k1].iter().map(|s| s.diag.tube).fold(0.0, f64::max));
    let mut out = Vec::new();
    for j in k0..k1 {
        let (u, e1, e2, pl) = nd.frame[j];
        for r in sums[j].seen.iter().flat_map(|g| g.recs.iter()).filter(|r| r.is_stop()) {
            let x: Point = std::array::from_fn(|i| nd.a[j][i] + (r.s + 0.5 * pl) * u[i] + r.xy[0] * e1[i] + r.xy[1] * e2[i]);
            if owns(&x) {
                out.push(r.xy);
            }
        }
    }
    out
}

/// Whether a place (world) within `reach` of row `k`'s claimed axis lies nearer it than any other axis the tree or
/// its `context` claims: what row `k` may take as its own (a point circle's stops, a moved circle's centre).
fn owner(k: usize, rows: &Rows, context: &[(Point, Point, f64)], reach: f64) -> impl Fn(&Point) -> bool {
    let (a, b) = (rows.a[k], rows.b[k]);
    let lo: Point = std::array::from_fn(|i| a[i].min(b[i]) - reach);
    let hi: Point = std::array::from_fn(|i| a[i].max(b[i]) + reach);
    let others: Vec<(Point, Point)> = rows.a.iter().zip(rows.b).enumerate().filter(|&(j, _)| j != k).map(|(_, (p, q))| (*p, *q))
        .chain(context.iter().map(|c| (c.0, c.1)))
        .filter(|(p, q)| (0..3).all(|i| p[i].min(q[i]) <= hi[i] && p[i].max(q[i]) >= lo[i])).collect();
    let to_axis = |x: &Point, (p, q): &(Point, Point)| {
        let (pq, px) = (sub(q, p), sub(x, p));
        let t = (dot(&px, &pq) / dot(&pq, &pq).max(1e-12)).clamp(0.0, 1.0);
        crate::transform::norm(&sub(&px, &pq.map(|v| v * t)))
    };
    move |x: &Point| {
        let own = to_axis(x, &(a, b));
        others.iter().all(|o| to_axis(x, o) >= own)
    }
}

/// A point circle `x` as the measurement of one of a row's `pieces` pieces: its centre and R, each of sd [`PTS_SD`] R
/// times the root of the pieces (one circle, shared by them); the lean unmeasured.
fn point_measure(x: &Vector5<f64>, pieces: f64) -> Circle {
    let var = (PTS_SD * x[4]).powi(2) * pieces;
    let meas: [Meas; 3] = [([1.0, 0.0, 0.0, 0.0, 0.0], x[0], var), ([0.0, 1.0, 0.0, 0.0, 0.0], x[1], var), ([0.0, 0.0, 0.0, 0.0, 1.0], x[4], var)];
    Circle { st: Stats::of(&meas), x: *x, cov: Matrix5::from_diagonal(&Vector5::new(var, var, UNSEEN_VAR, UNSEEN_VAR, var)), chi2: f64::NAN }
}

/// The ray veto's tally of a circle `x` (`[c1, c2, t1, t2, R]`, in the piece's frame) against a piece's groups: the
/// pulses whose line crosses it that stop on it (within [`VETO_TOL`] of its chord) and that pass through it (every
/// echo beyond the chord, or none in the tube). A pulse that stopped in front of it says nothing.
fn tally(gs: &[Group], x: &Vector5<f64>) -> (u32, u32) {
    let (mut stop, mut pass) = (0u32, 0u32);
    for g in gs {
        for r in g.recs.iter().filter(|r| !r.blocked) {
            let (cx, cy) = (x[0] + r.s * x[2], x[1] + r.s * x[3]);
            let bc = r.b - (g.n[0] * cx + g.n[1] * cy);
            if bc.abs() >= x[4] {
                continue;
            }
            let half = (x[4] * x[4] - bc * bc).sqrt();
            let d = r.depth - (g.v[0] * cx + g.v[1] * cy);
            if !d.is_finite() || d > half + VETO_TOL {
                pass += 1;
            } else if d >= -half - VETO_TOL {
                stop += 1;
            }
        }
    }
    (stop, pass)
}

/// Whether the ray veto refutes a circle with these stops and passes through it ([`diag::REFUTED`]).
fn refuted((stop, pass): (u32, u32)) -> bool {
    pass >= VETO_PASSES && f64::from(pass) >= VETO_SHARE * f64::from(stop + pass)
}

/// Where the veto refutes the circle `x` (`[c1, c2, t1, t2, log R]`) against a piece's groups: the log of the widest
/// radius below it, in steps of [`VETO_STEP`] down to [`VETO_MIN`], about the same axis, that it doesn't refute; None
/// where it doesn't refute `x`.
fn veto_at(gs: &[Group], x: &Vector5<f64>) -> Option<f64> {
    let mut y = *x;
    y[4] = x[4].exp();
    if !refuted(tally(gs, &y)) {
        return None;
    }
    let mut eta = x[4];
    while eta > VETO_MIN.ln() {
        eta -= VETO_STEP;
        y[4] = eta.exp();
        if !refuted(tally(gs, &y)) {
            break;
        }
    }
    Some(eta.max(VETO_MIN.ln()))
}

/// Where a circle `x` (`[c1, c2, t1, t2, log R]`) the veto refutes can move to keep the radius `exp(eta)` (its lower
/// bound): on rings about its centre, [`RECENTRE_STEP`] times that radius apart (at least [`RECENTRE_MIN`] m), out to
/// [`RECENTRE_REACH`] radii, with the circle inside the `tube` its pulses were taken in and its centre one its node
/// `owns` (nearer its own claimed axis than any other claim's: a neighbour's stem isn't taken), the nearest ring holding
/// a circle with a stop on it that the veto doesn't refute, and on it the one with the most stops; None where there is
/// none.
fn recentre(gs: &[Group], x: &Vector5<f64>, eta: f64, tube: f64, owns: impl Fn(f64, f64) -> bool) -> Option<(f64, f64)> {
    let r = eta.exp();
    let step = (RECENTRE_STEP * r).max(RECENTRE_MIN);
    let mut d = step;
    while d <= RECENTRE_REACH * r {
        let n = (std::f64::consts::TAU * d / step).ceil();
        let mut best: Option<(u32, (f64, f64))> = None;
        for i in 0..n as usize {
            let a = std::f64::consts::TAU * i as f64 / n;
            let c = (x[0] + d * a.cos(), x[1] + d * a.sin());
            if c.0.hypot(c.1) + r > tube || !owns(c.0, c.1) {
                continue;
            }
            let (stop, pass) = tally(gs, &Vector5::new(c.0, c.1, x[2], x[3], r));
            if stop > 0 && !refuted((stop, pass)) && best.map_or(true, |b| stop > b.0) {
                best = Some((stop, c));
            }
        }
        if let Some(b) = best {
            return Some(b.1);
        }
        d += step;
    }
    None
}

/// An unmeasured piece's upper bound from the ray veto (3dtrees research 106 `geometry.md`): the widest circle about
/// the claim's axis, of [`VETO_STEPS`] radii from [`VETO_MIN`] to the `tube`, below the narrowest the veto refutes;
/// None where it refutes none (passes alone bound R where no woody stop gives a centre: research 106 found trunks with
/// 47-84 passes through RCT's circle and no bound).
fn veto(gs: &[Group], tube: f64) -> Option<Bound> {
    let step = (tube.max(2.0 * VETO_MIN) / VETO_MIN).ln() / (VETO_STEPS - 1) as f64;
    let first = (0..VETO_STEPS).find(|&i| refuted(tally(gs, &Vector5::new(0.0, 0.0, 0.0, 0.0, VETO_MIN * (step * i as f64).exp()))))?;
    Some(Bound { eta: VETO_MIN.ln() + step * first.saturating_sub(1) as f64, sd: VETO_SD, side: Side::Upper })
}

/// What becomes of a piece's own summary: porous (pulses, no opaque run), bounded (no circle determined), routed
/// (several stems or a hull) or measured (its circle, which [`summarise_piece`] still refuses at its tube's edge);
/// recording its runs, scans and front misfit in `out`.
fn outcome(gs: &[Group], r_claim: f64, p: &JointParams, lean_sd: f64, out: &mut Summary) -> Outcome {
    // the first pass: each stop by its echo
    let first: Vec<GroupOut> = gs.iter().map(|g| {
        let f: Vec<Fate> = g.recs.iter().map(|r| fate(r, g, None)).collect();
        measure(g, &f, &weights(g, &f, r_claim, p), 0.0, r_claim, (-2.0 * r_claim - DEPTH_TOL, r_claim + DEPTH_TOL), p)
    }).collect();
    if first.iter().any(|o| o.enough) && !first.iter().any(|o| o.run) {
        return Outcome::Porous;
    }
    let Some(c1) = solve_row(&Stats::of(&first.iter().flat_map(|o| o.meas.clone()).collect::<Vec<_>>()), r_claim, lean_sd, p.model_sd) else {
        return Outcome::Bounded;
    };
    // the second: each stop gated by the first's circle
    let x = c1.x;
    let mut meas = Vec::new();
    let mut fates2 = Vec::with_capacity(gs.len());
    for g in gs {
        let f: Vec<Fate> = g.recs.iter().map(|r| fate(r, g, Some(&x))).collect();
        let w = weights(g, &f, x[4], p);
        let o = measure(g, &f, &w, g.n[0] * x[0] + g.n[1] * x[1], x[4].max(0.005), (f64::NEG_INFINITY, f64::INFINITY), p);
        out.n_scans += u32::from(!o.meas.is_empty());
        meas.extend(o.meas);
        fates2.push((f, w));
    }
    let Some(c2) = solve_row(&Stats::of(&meas), r_claim, lean_sd, p.model_sd) else {
        return Outcome::Bounded;
    };
    let y = c2.x;
    let rr = y[4];
    // the stems beside it and the run's width, its stops gated by its depth band
    let runs: Vec<(usize, bool)> = gs.iter().map(|g| runs_of(g, &y, r_claim, p)).collect();
    out.n_runs = runs.iter().map(|q| q.0 as u32).max().unwrap_or(0);
    let mut hull = runs.iter().any(|q| q.1);
    // the hull: of the stops (any echo, before any gate) aimed at the circle's core that reached its front, the
    // share behind it (research 93 §13.1); and the front's hits' misfit about it
    let (mut on, mut behind) = (0usize, 0usize);
    let var = (p.sigma_r * p.sigma_r + (p.rough * rr).powi(2)).max(p.model_sd * p.model_sd);
    let (mut ss, mut nw, mut ng) = (0.0, 0.0, 0usize);
    for (g, (f, w)) in gs.iter().zip(&fates2) {
        let mut used = false;
        for (i, r) in g.recs.iter().enumerate() {
            if r.blocked || !r.depth.is_finite() {
                continue;
            }
            let (bc, front) = arc(&y, g, r);
            if bc.abs() >= 0.8 * rr {
                continue;
            }
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
    if nw - ng as f64 > 0.0 {
        out.chi2_front = ss / (nw - ng as f64);
    }
    hull |= (on > 0 && behind as f64 > HULL_SHARE * on as f64) || c2.chi2 > HULL_CHI2 || out.chi2_front > HULL_CHI2;
    let runs = out.n_runs >= 2;
    // (a measured piece's measurements already leave an unseen edge out, so in the tree the prior's centre fills the
    // ridge it leaves; bounding it instead cost its circles, research 93 §20: 0.064 to 0.150 on the bench's 15)
    if runs || hull {
        Outcome::Routed { runs, hull }
    } else {
        Outcome::Measured(c2)
    }
}

/// Whether the circle `x` (`[c1, c2, t1, t2, R]`) reaches within [`TUBE_MARGIN`] of the `tube` its pulses were taken
/// in, anywhere along its half-length `half` ([`diag::TUBE_EDGE`]).
fn reaches_edge(x: &Vector5<f64>, half: f64, tube: f64) -> bool {
    let reach = [-half, half].iter().map(|&q| (x[0] + q * x[2]).hypot(x[1] + q * x[3])).fold(0.0, f64::max) + x[4];
    reach > tube - TUBE_MARGIN
}

/// A routed piece's further stem (3dtrees research 93 §18): the piece it came from and its stem there; and its share
/// of the stems' area where it is a node beside its piece (a fork term), none where it continues a stem of the
/// piece's routed parent.
#[derive(Debug, Clone, Copy)]
struct Further {
    piece: usize,
    stem: u32,
    share: Option<f64>,
}

/// The tree's nodes, the rows' pieces: each row cut into `ceil(len / PIECE_LEN)` equal pieces, in order, a row's
/// first piece's parent the last of its parent row's and each later piece's the one before it. After them, a routed
/// piece's further stems (3dtrees research 93 §18), in its piece's frame: each continues a further stem of its
/// piece's parent where it starts at that stem's end, and is otherwise a node beside its piece (its piece's parent's
/// child, by a fork term on its share of the stems' area).
struct Nodes {
    row: Vec<usize>,
    /// Each row's first piece, and one past the last row's last.
    first: Vec<usize>,
    parent: Vec<Option<usize>>,
    frame: Vec<Frame>,
    a: Vec<Point>,
    further: Vec<Option<Further>>,
}

impl Nodes {
    /// Appends a further stem of piece `k`: continuing node `from`, or else beside the piece with its `share`.
    fn push_stem(&mut self, k: usize, stem: u32, from: Option<usize>, share: f64) {
        self.row.push(self.row[k]);
        self.parent.push(from.or(self.parent[k]));
        self.frame.push(self.frame[k]);
        self.a.push(self.a[k]);
        self.further.push(Some(Further { piece: k, stem, share: from.is_none().then_some(share) }));
    }

    /// The piece a node comes from (itself, for a piece) and its stem there (0: the piece's own).
    fn src(&self, k: usize) -> (usize, u32) {
        self.further[k].map_or((k, 0), |f| (f.piece, f.stem))
    }

    fn of(rows: &Rows, fr: &[Frame]) -> Self {
        let mut nd = Nodes { row: Vec::new(), first: Vec::new(), parent: Vec::new(), frame: Vec::new(), a: Vec::new(), further: Vec::new() };
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
                nd.further.push(None);
                nd.row.push(k);
                nd.frame.push((u, e1, e2, pl));
                nd.a.push(std::array::from_fn(|i| rows.a[k][i] + j as f64 * pl * u[i]));
            }
        }
        nd.first.push(nd.row.len());
        nd
    }
}

/// The point at axial place `s` (from the middle) of the axis `x` (`[c1, c2, t1, t2, ...]`) of a node claimed from
/// `a0` with `frame`.
fn axis_point(a0: &Point, (u, e1, e2, len): Frame, x: &Vector5<f64>, s: f64) -> Point {
    std::array::from_fn(|i| a0[i] + (s + 0.5 * len) * u[i] + (x[0] + s * x[2]) * e1[i] + (x[1] + s * x[3]) * e2[i])
}

/// How far a stem may start from where a node's stem ends, beyond the larger of their radii, and still continue it
/// (m).
const CHAIN_GAP: f64 = 0.02;

/// Enters routed piece `k`'s accepted `stems` into the tree (pieces come parents first) and returns each stem's node.
/// The stem that continues the piece is, of those starting within [`CHAIN_GAP`] (beyond the larger radius) of where
/// its parent's stem ends, the one whose message says most of its radius (the nearest where none is that near; the
/// most informed at a base). Each other continues, once, the further stem of the parent piece that ends nearest its
/// start within the same reach, or else is a node beside the piece (its parent's child, by a fork term on its share
/// of the stems' area). `ends` holds each node's end and radius there, `further` each piece's further stems' nodes. A
/// shadowed stem's own bounds go with it.
fn enter(k: usize, e: &routed::Exact, nd: &mut Nodes, sums: &mut Vec<Summary>, ends: &mut Vec<(Point, f64)>, further: &mut [Vec<usize>]) -> Vec<u32> {
    let stems = &e.stems[..];
    let (a_k, f_k) = (nd.a[k], nd.frame[k]);
    let start = |c: &Circle| axis_point(&a_k, f_k, &c.x, -0.5 * f_k.3);
    let end = |c: &Circle| axis_point(&a_k, f_k, &c.x, 0.5 * f_k.3);
    let gap = |p: &Point, q: &Point| crate::transform::norm(&sub(p, q));
    // what a stem's message says of its radius: 1 / var(log R)
    let info = |i: &usize| stems[*i].x[4] * stems[*i].x[4] / stems[*i].cov[(4, 4)];
    let n = stems.len();
    let lead = match nd.parent[k] {
        Some(q) => {
            let (e, r) = ends[q];
            let d = |i: &usize| gap(&start(&stems[*i]), &e);
            (0..n).filter(|i| d(i) < stems[*i].x[4].max(r) + CHAIN_GAP).max_by(|i, j| info(i).total_cmp(&info(j)))
                .or_else(|| (0..n).min_by(|i, j| d(i).total_cmp(&d(j))))
        }
        None => (0..n).max_by(|i, j| info(i).total_cmp(&info(j))),
    }.unwrap_or(0);
    ends[k] = (end(&stems[lead]), stems[lead].x[4]);
    sums[k].exact = Some(stems[lead]);
    (sums[k].diag.tube, sums[k].diag.n_ctx) = (e.tube, e.n_ctx[lead]);
    if let Some(b) = &e.shadow[lead] {
        sums[k].shadowed = true;
        sums[k].bounds = b.clone();
    }
    let area: f64 = stems.iter().map(|c| c.x[4] * c.x[4]).sum();
    let mut node = vec![k as u32; n];
    let mut taken: Vec<usize> = Vec::new();
    for (s, (j, c)) in stems.iter().enumerate().filter(|&(j, _)| j != lead).enumerate() {
        let from = nd.parent[k].and_then(|q| further[q].iter().copied().filter(|m| !taken.contains(m))
            .map(|m| (m, gap(&ends[m].0, &start(c))))
            .filter(|&(m, d)| d < c.x[4].max(ends[m].1) + CHAIN_GAP)
            .min_by(|x, y| x.1.total_cmp(&y.1)).map(|(m, _)| m));
        taken.extend(from);
        node[j] = nd.row.len() as u32;
        further[k].push(nd.row.len());
        nd.push_stem(k, s as u32 + 1, from, c.x[4] * c.x[4] / area);
        let p = &sums[k];
        sums.push(Summary { outcome: p.outcome, chi2_front: f64::NAN, n_runs: p.n_runs, n_scans: p.n_scans, bounds: e.shadow[j].clone().unwrap_or_default(),
            log_odds: p.log_odds, seeds: Vec::new(), exact: Some(*c), seen: Vec::new(), shadowed: e.shadow[j].is_some(), points: None, diag: DiagIn { n_ctx: e.n_ctx[j], ..p.diag } });
        ends.push((end(c), c.x[4]));
    }
    node
}

/// Each of `trees`' [`support_tree`] context: of a plot's cylinders `a -> b`,
/// tree `g` owning `cyl_offsets[g]..cyl_offsets[g + 1]`, the other trees'
/// whose segments come within `reach` of one of `g`'s, as sorted indices
/// (CSR: tree `trees[i]`'s are `idx[off[i]..off[i + 1]]`). A cylinder of
/// zero length is never close. The distance is written as 3dtrees' numpy
/// wrote it, so the two agree to the bit (the context keys its store).
pub fn tree_context(a: &[Point], b: &[Point], cyl_offsets: &[usize], trees: &[usize], reach: f64) -> (Vec<usize>, Vec<usize>) {
    // Indexed by pieces no longer than `reach` (a plot's cylinders run from millimetres to metres): segments within
    // `reach` have pieces whose midpoints are within half of each plus `reach`, so a piece's query reaches no further
    // than its own half, `reach / 2` and `reach`, however long the longest cylinder.
    let len: Vec<f64> = (0..a.len()).map(|c| crate::transform::norm(&sub(&b[c], &a[c]))).collect();
    let (mut mids, mut halves, mut of) = (Vec::new(), Vec::new(), Vec::new());
    for c in (0..a.len()).filter(|&c| len[c] > 0.0) {
        let k = if reach > 0.0 { (len[c] / reach).ceil().max(1.0) as usize } else { 1 };
        for p in 0..k {
            let f = (p as f64 + 0.5) / k as f64;
            mids.push([a[c][0] + f * (b[c][0] - a[c][0]), a[c][1] + f * (b[c][1] - a[c][1]), a[c][2] + f * (b[c][2] - a[c][2])]);
            halves.push(len[c] / k as f64 / 2.0);
            of.push(c);
        }
    }
    let kd = crate::util::spatial::KdTree::new(&mids);
    let start: Vec<usize> = (0..=a.len()).map(|c| of.partition_point(|&x| x < c)).collect(); // cylinder c's pieces
    let per: Vec<Vec<usize>> = trees
        .par_iter()
        .map(|&g| {
            let (own, of) = (cyl_offsets[g]..cyl_offsets[g + 1], &of);
            let mut cand: Vec<(usize, usize)> = (start[own.start]..start[own.end])
                .flat_map(|p| {
                    kd.within(&mids[p], (halves[p] + reach / 2.0 + reach) * (1.0 + 1e-9) + 1e-9)
                        .into_iter()
                        .map(move |(q, _)| (of[p], of[q]))
                })
                .filter(|&(_, j)| !own.contains(&j))
                .collect();
            cand.sort_unstable();
            cand.dedup();
            let mut near: Vec<usize> = cand.into_iter().filter(|&(c, j)| segment_distance(&a[c], &b[c], &a[j], &b[j]) <= reach).map(|(_, j)| j).collect();
            near.sort_unstable();
            near.dedup();
            near
        })
        .collect();
    let off = std::iter::once(0).chain(per.iter().scan(0, |n, p| { *n += p.len(); Some(*n) })).collect();
    (off, per.concat())
}

/// Closest distance between segments p1-q1 and p2-q2 (3dtrees' `joint.segment_distance`, operation for operation).
fn segment_distance(p1: &Point, q1: &Point, p2: &Point, q2: &Point) -> f64 {
    let (d1, d2, w) = (sub(q1, p1), sub(q2, p2), sub(p1, p2));
    let dot3 = |u: &Point, v: &Point| (u[0] * v[0] + u[1] * v[1]) + u[2] * v[2];
    let (a, e, b, c, f) = (dot3(&d1, &d1).max(1e-12), dot3(&d2, &d2).max(1e-12), dot3(&d1, &d2), dot3(&d1, &w), dot3(&d2, &w));
    let den = a * e - b * b;
    let s = (if den > 1e-12 { (b * f - c * e) / den } else { -c / a }).clamp(0.0, 1.0);
    let t = (b * s + f) / e;
    let s = if t < 0.0 { (-c / a).clamp(0.0, 1.0) } else if t > 1.0 { ((b - c) / a).clamp(0.0, 1.0) } else { s };
    let t = t.clamp(0.0, 1.0);
    let g = [p1[0] + d1[0] * s - p2[0] - d2[0] * t, p1[1] + d1[1] * s - p2[1] - d2[1] * t, p1[2] + d1[2] * s - p2[2] - d2[2] * t];
    ((g[0] * g[0] + g[1] * g[1]) + g[2] * g[2]).sqrt()
}

/// Fits one tree from its rows' support summaries: `rows` (with `r_max` ignored) against `scans`, the prior's
/// widths from `prior` and the footprint, detection threshold, range noise and echo character from `like`. Runs on
/// `threads` threads (0: one per core).
pub fn support_tree(scans: &[&ScanIndex], rows: &Rows, context: &[(Point, Point, f64)], like: &JointParams, prior: &PriorParams, threads: usize) -> Result<SupportTree> {
    let fr = rows.check("support tree")?;
    prior.check()?;
    let mut nd = Nodes::of(rows, &fr);
    let n0 = nd.row.len();
    let mut sums: Vec<Summary> = pool(threads)?.install(|| {
        (0..n0).into_par_iter().map(|k| {
            let (u, _, _, pl) = nd.frame[k];
            let b: Point = std::array::from_fn(|i| nd.a[k][i] + pl * u[i]);
            summarise_piece(scans, &nd.a[k], &b, nd.frame[k], rows.r_prior[nd.row[k]], like, prior.lean_sd)
        }).collect()
    });

    // the routed pieces' exact fits (3dtrees research 93 §18), with the tree as it stands without them: their
    // neighbours' cylinders, measured or bounded, as context, and where each piece's stem continues from
    let (sol0, _) = tree_solve(rows, &fr, &nd, &sums, prior, FIRST_ROUNDS)?;
    let ends_of = |nd: &Nodes, k: usize, x: &Vector5<f64>| {
        let h = 0.5 * nd.frame[k].3;
        (axis_point(&nd.a[k], nd.frame[k], x, -h), axis_point(&nd.a[k], nd.frame[k], x, h))
    };
    let ctx: Vec<(Point, Point, f64)> = (0..n0).filter(|&k| matches!(sums[k].outcome, Outcome::Measured(_) | Outcome::Bounded))
        .map(|k| {
            let (a, b) = ends_of(&nd, k, &sol0.x[k]);
            (a, b, sol0.x[k][4].exp())
        }).chain(context.iter().copied()).collect();
    let routed: Vec<usize> = (0..n0).filter(|&k| matches!(sums[k].outcome, Outcome::Routed { .. })).collect();
    let fit = |&k: &usize| {
        // the piece's detection bound (log R), its tightest
        let lower = sums[k].bounds.iter().filter(|b| b.side == Side::Lower).map(|b| b.eta).fold(f64::NEG_INFINITY, f64::max);
        routed::exact_piece(scans, &nd.a[k], nd.frame[k], &sums[k].seeds, &ctx, lower, like, prior)
    };
    let fits: Vec<Result<routed::Fitted>> = pool(threads)?.install(|| routed.par_iter().map(fit).collect());
    let mut k_stems = vec![0u32; n0];
    let mut echoes = Echoes::default();
    // where each node's stem ends (world) and its radius there: the first solve's, or a routed piece's stem's; and
    // each routed piece's further stems' nodes
    let mut ends: Vec<(Point, f64)> = (0..n0).map(|k| (ends_of(&nd, k, &sol0.x[k]).1, sol0.x[k][4].exp())).collect();
    let mut further: Vec<Vec<usize>> = vec![Vec::new(); n0];
    for (&k, f) in routed.iter().zip(fits) {
        match f? {
            routed::Fitted::Refused => {}
            routed::Fitted::Accepted(e) => {
                k_stems[k] = e.stems.len() as u32;
                let node = enter(k, &e, &mut nd, &mut sums, &mut ends, &mut further);
                k_stems.resize(nd.row.len(), 0);
                echoes.append(e.echoes, |j| node[j as usize]);
            }
        }
    }

    // the site's D/H cap (log R; NaN without one): the widest point circle's ceiling, and after the solve each
    // unmeasured node's
    let caps: Vec<f64> = if prior.dh > 0.0 {
        reach(rows, &fr, &nd).iter().map(|l| (0.5 * prior.dh * l).ln()).collect()
    } else {
        vec![f64::NAN; nd.row.len()]
    };
    // point evidence: a row none of whose pieces has a circle of its own gets its stops' circle, unless the veto
    // refutes it (3dtrees research 111: on tangles a circle can be another tree's)
    for k in 0..rows.a.len() {
        let (k0, k1) = (nd.first[k], nd.first[k + 1]);
        if rows.r_prior[k] < PTS_R_MIN || (k0..k1).any(|j| sums[j].own().is_some()) {
            continue;
        }
        let xy = own_stops(k, rows, context, &nd, &sums);
        let Some(x) = point_circle(&xy, rows.r_prior[k], PTS_BELIEVE * caps[k0].exp()) else { continue };
        // a circle at its pieces' tubes' edge is the captor a piece's own circle there was refused for (TUBE_EDGE)
        if (k0..k1).any(|j| reaches_edge(&x, 0.0, sums[j].diag.tube)) {
            continue;
        }
        if !refuted((k0..k1).map(|j| tally(&sums[j].seen, &x)).fold((0, 0), |a, q| (a.0 + q.0, a.1 + q.1))) {
            let c = point_measure(&x, (k1 - k0) as f64);
            for s in &mut sums[k0..k1] {
                s.points = Some(c);
                // the bounds its stops gave (the woody run's span, the capacity floor) are the circle's own data
                s.bounds.retain(|b| b.side == Side::Upper);
            }
        }
    }

    let (mut sol, converged) = tree_solve(rows, &fr, &nd, &sums, prior, EP_ROUNDS)?;
    let entered = terms(&sums);
    let (mut held, mut bits) = hold(&entered, nd.row.len(), &mut sol);
    // the nodes with no circle of their own (a point circle's among them), and those with none at all, which the cap
    // holds
    let unmeasured: Vec<bool> = sums.iter().map(Summary::bounded).collect();
    let free: Vec<bool> = sums.iter().map(|s| s.bounded() && s.points.is_none()).collect();
    // the cap, after the solve, on each node with no circle at all, alone; before the veto, since a narrower circle
    // can be one the rays refute (its stops on the front arc fall away before its passes do: 3dtrees research 127)
    cap(&caps, &free, &mut sol, &mut held, &mut bits);
    // what the rays don't measure is held under the veto about its own posterior circle too: a woody run's bounds
    // straddle its run, not the circle served (research 106: 17 % of such rows' volume was refuted). Where that would
    // narrow it below its own lower bound, the circle is beside the wood its bound saw: it moves onto it, where the
    // rays allow a circle that wide (3dtrees research 127), else it narrows about its own centre
    // (a point circle's piece has no lower bound in the tree, so it is never moved)
    for k in (0..nd.row.len()).filter(|&k| unmeasured[k]) {
        let gs = &sums[nd.src(k).0].seen;
        let Some(eta) = veto_at(gs, &sol.x[k]) else { continue };
        let moved = (eta < held[k].0).then(|| {
            let owns = owner(nd.row[k], rows, context, sums[k].diag.tube);
            let mid = |c1: f64, c2: f64| axis_point(&nd.a[k], nd.frame[k], &Vector5::new(c1, c2, 0.0, 0.0, 0.0), 0.0);
            recentre(gs, &sol.x[k], held[k].0, sums[k].diag.tube, |c1, c2| owns(&mid(c1, c2)))
        }).flatten();
        let eta = match moved {
            Some(c) => {
                (sol.x[k][0], sol.x[k][1]) = c;
                bits[k] |= diag::MOVED;
                // the moved circle at its lower bound isn't refuted, so the veto narrows it no further
                match veto_at(gs, &sol.x[k]) {
                    Some(e) => e.max(held[k].0),
                    None => continue,
                }
            }
            None => eta,
        };
        sol.x[k][4] = eta;
        held[k] = (held[k].0.min(eta), held[k].1.min(eta));
        bits[k] |= diag::PINNED;
    }
    let p_wood = existence(&nd.parent, &sums.iter().map(|s| s.log_odds).collect::<Vec<_>>(), prior);
    // the classification on the support path: each stop by its piece's circle (its own, or the tree's) and its echo
    for (k, s) in sums.iter().enumerate().filter(|(_, s)| s.exact.is_none()) {
        classify(s, &sol.x[k], k as u32, scans, like, &mut echoes);
    }
    // each node's stops before the context and its neighbours took theirs, and after
    let nn = nd.row.len();
    let mut seen_stops: Vec<u32> = sums.iter().map(|s| s.diag.n_ctx).collect();
    for &q in &echoes.node {
        seen_stops[q as usize] += 1;
    }
    let echoes = echoes.dedup();
    let mut kept_stops = vec![0u32; nn];
    for &q in &echoes.node {
        kept_stops[q as usize] += 1;
    }
    // the veto's audit of each node's fitted circle, from its piece's pulses
    let audit: Vec<(u32, u32)> = (0..nn).map(|k| {
        let mut x = sol.x[k];
        x[4] = x[4].exp();
        tally(&sums[nd.src(k).0].seen, &x)
    }).collect();

    let mut out = SupportTree {
        rows: Fit::default(),
        pieces: Fit::default(),
        piece_row: nd.row.iter().map(|&k| k as u32).collect(),
        piece_src: (0..nd.row.len()).map(|k| nd.src(k).0 as u32).collect(),
        piece_stem: (0..nd.row.len()).map(|k| nd.src(k).1).collect(),
        converged,
        echoes,
        volume: Volume::default(),
    };
    // the tightest bound of a side, of those that entered the tree (so a node's radius is held in what it reports)
    let mut bounds_of: Vec<&[Bound]> = vec![&[]; nd.row.len()];
    for (k, bs) in &entered {
        bounds_of[*k] = bs;
    }
    let bound = |k: usize, side: Side| {
        let it = bounds_of[k].iter().filter(|b| b.side == side).map(|b| b.eta);
        let e = if side == Side::Upper { it.fold(f64::INFINITY, f64::min) } else { it.fold(f64::NEG_INFINITY, f64::max) };
        if e.is_finite() { e.exp() } else { f64::NAN }
    };
    for (k, s) in sums.iter().enumerate() {
        let (diag, data_share, contested) = diagnose(s, sol.x[k][4].exp(), sol.cov[k][(4, 4)], sol.site_tau[k], 0.5 * nd.frame[k].3, (seen_stops[k], kept_stops[k]));
        let diag = diag | bits[k] | if refuted(audit[k]) { diag::REFUTED } else { 0 };
        let basis = if !s.bounded() {
            basis::MEASURED
        } else if s.points.is_some() {
            basis::POINTS
        } else if diag & diag::CAPPED != 0 {
            basis::CAPPED
        } else if diag & diag::PINNED != 0 || !(data_share >= BOUNDED_SHARE) {
            basis::PRIOR
        } else {
            basis::BOUNDED
        };
        let own = Own {
            diag,
            data_share,
            contested,
            deleted: s.diag.deleted,
            r_lo: bound(k, Side::Lower),
            r_hi: bound(k, Side::Upper),
            r_cap: if free[k] { caps[k].exp() } else { f64::NAN },
            p_wood: p_wood[k],
            flags: s.outcome.flags(),
            n_scans: s.n_scans,
            k_stems: k_stems[k],
            audit: audit[k],
            basis,
        };
        out.pieces.push(&sol.x[k], &widened(&sol.cov[k], widen(basis)), nd.frame[k], nd.a[k], own);
    }
    for k in 0..rows.a.len() {
        let (k0, k1) = (nd.first[k], nd.first[k + 1]);
        let p = &out.pieces;
        // each piece's log R widened by its own basis's factor: a row of mixed bases piece by piece
        let k_of: Vec<f64> = p.basis[k0..k1].iter().map(|&b| widen(b)).collect();
        let (x, cov) = row_line(&sol, k0, k1, fr[k].3, nd.frame[k0].3, &k_of);
        let fold = |v: &[f64], f: fn(f64, f64) -> f64| v[k0..k1].iter().copied().filter(|x| x.is_finite()).reduce(f).unwrap_or(f64::NAN);
        let mean = |v: &[f64]| {
            let v: Vec<f64> = v[k0..k1].iter().copied().filter(|x| x.is_finite()).collect();
            if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
        };
        // a row's bounds as its radius: its pieces' root mean square, a piece without a lower bound at 0 and one
        // without an upper unbounded, so a row whose pieces lie inside their bounds lies inside its own
        let rms = |v: &[f64], none: f64| {
            let v = &v[k0..k1];
            let ms = v.iter().map(|x| if x.is_finite() { x * x } else { none }).sum::<f64>() / v.len() as f64;
            if v.iter().all(|x| x.is_nan()) || ms.is_infinite() { f64::NAN } else { ms.sqrt() }
        };
        let audit = audit[k0..k1].iter().fold((0, 0), |a, q| (a.0 + q.0, a.1 + q.1));
        // most of its pieces' basis, the less known on a tie (max_by_key keeps the last of equal maxima)
        let basis = (1..=basis::CAPPED).max_by_key(|&b| p.basis[k0..k1].iter().filter(|&&x| x == b).count()).unwrap_or(basis::PRIOR);
        let own = Own {
            diag: (p.diag[k0..k1].iter().fold(0, |a, &f| a | f) & !diag::REFUTED) | if refuted(audit) { diag::REFUTED } else { 0 },
            data_share: mean(&p.data_share),
            contested: mean(&p.contested),
            deleted: mean(&p.deleted),
            r_lo: rms(&p.r_lo, 0.0),
            r_hi: rms(&p.r_hi, f64::INFINITY),
            r_cap: fold(&p.r_cap, f64::max),
            p_wood: p.p_wood[k0..k1].iter().sum::<f64>() / (k1 - k0) as f64,
            flags: p.flags[k0..k1].iter().fold(0, |a, &f| a | f),
            n_scans: p.n_scans[k0..k1].iter().copied().max().unwrap_or(0),
            k_stems: p.k_stems[k0..k1].iter().copied().max().unwrap_or(0),
            audit,
            basis,
        };
        out.rows.push(&x, &cov, fr[k], rows.a[k], own);
    }
    out.volume = volume(&nd, &sol, &held, &out.pieces.basis);
    Ok(out)
}

/// Each node's reach: the longest path (m) from its start, along its row and the rows after it, to a tip; a further
/// stem's, its piece's (what the site's D/H cap is a share of, as the refit's was: research 62).
fn reach(rows: &Rows, fr: &[Frame], nd: &Nodes) -> Vec<f64> {
    let mut tip: Vec<f64> = fr.iter().map(|f| f.3).collect();
    for k in (0..fr.len()).rev() {
        if rows.parent[k] >= 0 {
            let q = rows.parent[k] as usize;
            tip[q] = tip[q].max(fr[q].3 + tip[k]);
        }
    }
    (0..nd.row.len()).map(|k| {
        let j = nd.src(k).0;
        let row = nd.row[j];
        tip[row] - (j - nd.first[row]) as f64 * nd.frame[j].3
    }).collect()
}

/// Holds each bounded node's posterior log R inside its own bounds (3dtrees research 106: 5.6 % of rows lay outside
/// them): between the tightest lower (detection, the capacity floor) and the tightest upper (passes), the upper where
/// they cross. Returns each node's interval (log R; unbounded where it has none) and its [`diag::PINNED`] bit: where
/// its cavity (the tree without its bounds) or its posterior lay beyond one.
fn hold(terms: &[(usize, Vec<Bound>)], n: usize, sol: &mut Solved) -> (Vec<(f64, f64)>, Vec<u16>) {
    let (mut held, mut bits) = (vec![(f64::NEG_INFINITY, f64::INFINITY); n], vec![0u16; n]);
    for (k, bs) in terms {
        let k = *k;
        let eta = |side: Side| {
            let it = bs.iter().filter(|b| b.side == side).map(|b| b.eta);
            if side == Side::Upper { it.fold(f64::INFINITY, f64::min) } else { it.fold(f64::NEG_INFINITY, f64::max) }
        };
        let (lo, upper) = (eta(Side::Lower).max(eta(Side::Floor)), eta(Side::Upper));
        let lower = lo.min(upper);
        let (x, mc) = (sol.x[k][4], sol.cavity[k]);
        if x > upper || mc > upper || x < lower || mc < lo {
            bits[k] = diag::PINNED;
        }
        sol.x[k][4] = x.max(lower).min(upper);
        held[k] = (lower, upper);
    }
    (held, bits)
}

/// The site's D/H cap (3dtrees research 106 `uncertain.md`), after the solve and on each `free` node alone (one into
/// which no circle enters, its own or its row's point circle), never below what its rays hold it above (its
/// interval's lower end, `held`): its log R
/// (`caps`, NaN for none) clipped there, its interval's upper end too (so are its volume's draws), and
/// [`diag::CAPPED`] where its posterior median reached a cap above that lower end. A site rule never overrides the
/// rays.
fn cap(caps: &[f64], free: &[bool], sol: &mut Solved, held: &mut [(f64, f64)], bits: &mut [u16]) {
    for k in (0..caps.len()).filter(|&k| free[k] && caps[k].is_finite()) {
        let c = caps[k].max(held[k].0);
        if sol.x[k][4] >= c {
            bits[k] |= if caps[k] >= held[k].0 { diag::CAPPED } else { 0 };
            sol.x[k][4] = c;
        }
        held[k].1 = held[k].1.min(c);
    }
}

/// A tree's volume ([`Volume`]): its pieces' (a routed piece's further stems aside), at their posterior R and split by
/// their `basis`, and by [`VOLUME_DRAWS`] draws from the joint posterior: root first, each node's deviation its
/// parent's carried by the elimination (`-Y_k d_parent`) plus its own given its parent's, each node's then `widen`
/// times for its basis (a diagonal scaling of the joint covariance: the nodes' correlations kept, each sd widened as
/// its basis's), each piece's log R then held in its interval (`held`). A fixed seed, so a tree's volume is a function
/// of its fit.
fn volume(nd: &Nodes, sol: &Solved, held: &[(f64, f64)], basis: &[u8]) -> Volume {
    let n = nd.row.len();
    let pieces: Vec<usize> = (0..n).filter(|&k| nd.further[k].is_none()).collect();
    let vol = |k: usize, eta: f64| std::f64::consts::PI * (2.0 * eta).exp() * nd.frame[k].3;
    let mut v = Volume { point: pieces.iter().map(|&k| vol(k, sol.x[k][4])).sum(), ..Volume::default() };
    let chol: Vec<Matrix5<f64>> = sol.inv.iter().map(|m| m.cholesky().map_or_else(|| Matrix5::from_diagonal(&m.diagonal().map(|d| d.max(0.0).sqrt())), |c| c.l())).collect();
    let mut rng = Normal::new(0x5EED_0F_7EE5);
    let k_of: Vec<f64> = basis.iter().map(|&b| widen(b)).collect();
    let mut d = vec![Vector5::<f64>::zeros(); n];
    let mut draws: Vec<f64> = (0..VOLUME_DRAWS).map(|_| {
        for k in 0..n {
            let z = Vector5::from_fn(|_, _| rng.next());
            d[k] = chol[k] * z - nd.parent[k].map_or(Vector5::zeros(), |q| sol.y[k] * d[q]);
        }
        pieces.iter().map(|&k| {
            let x = vol(k, (sol.x[k][4] + k_of[k] * d[k][4]).max(held[k].0).min(held[k].1));
            v.by_basis[usize::from(basis[k]) - 1] += x;
            x
        }).sum()
    }).collect();
    draws.sort_unstable_by(f64::total_cmp);
    let q = |p: f64| draws[(p * (VOLUME_DRAWS - 1) as f64).round() as usize];
    v.mean = draws.iter().sum::<f64>() / VOLUME_DRAWS as f64;
    (v.q05, v.q95) = (q(0.05), q(0.95));
    v.by_basis = v.by_basis.map(|x| x / VOLUME_DRAWS as f64);
    v
}

/// Standard normal draws from a fixed seed (splitmix64, Box-Muller): the volume's draws are reproducible.
struct Normal {
    state: u64,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Normal { state: seed, spare: None }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (u, v) = (self.uniform(), self.uniform());
        let (r, t) = ((-2.0 * u.ln()).sqrt(), 2.0 * std::f64::consts::PI * v);
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

/// A node's diagnostics ([`diag`] bits, `data_share` and `contested`; see [`Fit::diag`]) from its summary, its
/// posterior R and log R's variance, its bounds' EP site precision, its half-length and its stops before and after
/// the context and its neighbours took theirs.
fn diagnose(s: &Summary, r: f64, var_eta: f64, site_tau: f64, half: f64, (seen, kept): (u32, u32)) -> (u16, f64, f64) {
    let d = &s.diag;
    let contested = if seen > 0 { 1.0 - f64::from(kept) / f64::from(seen) } else { f64::NAN };
    // log R's precision from the node's own data: its measurement's (R's information, the centre and lean unknown,
    // at the posterior R) and its bounds' site; a negative site precision (an EP site that isn't a likelihood's)
    // leaves it undetermined
    let own = s.used().map_or(0.0, |c| info_r(&(c.st.info / c.inflate())) * r * r);
    let data_share = if site_tau < 0.0 { f64::NAN } else { ((own + site_tau) * var_eta).clamp(0.0, 1.0) };
    // how far its own circle reaches from the claim's axis (over its length), against the tube its pulses came from
    // (a piece's own circle that reached it wasn't used)
    let tube_edge = d.edge || s.used().is_some_and(|c| reaches_edge(&c.x, half, d.tube));
    let bits = [
        (d.one_sided, diag::ONE_SIDED),
        (s.shadowed, diag::SHADOWED),
        (r < SUB_W * d.w, diag::SUB_FOOTPRINT),
        (tube_edge, diag::TUBE_EDGE),
        (contested >= CONTESTED_SHARE, diag::CONTESTED),
    ];
    (bits.iter().filter(|b| b.0).fold(0, |a, b| a | b.1), data_share, contested)
}

/// A piece's or row's own outputs (see [`Fit`]).
struct Own {
    diag: u16,
    data_share: f64,
    contested: f64,
    deleted: f64,
    r_lo: f64,
    r_hi: f64,
    r_cap: f64,
    p_wood: f64,
    flags: u8,
    n_scans: u32,
    k_stems: u32,
    /// The veto's stops and passes through its fitted circle.
    audit: (u32, u32),
    basis: u8,
}

/// Classifies a piece's stops with no exact fit (see [`Echoes`]): on its circle's front arc (the tree's posterior
/// `x`, log R last) or not, times the weight of its echo's character.
fn classify(s: &Summary, x: &Vector5<f64>, node: u32, scans: &[&ScanIndex], p: &JointParams, out: &mut Echoes) {
    let mut y = *x;
    y[4] = x[4].exp();
    for g in &s.seen {
        for r in g.recs.iter().filter(|r| r.is_stop()) {
            let w = p_wood(r, y[4], p);
            let pw = if fate(r, g, Some(&y)) == Fate::Hit { w } else { 0.0 };
            let (cx, cy) = (y[0] + r.s * y[2], y[1] + r.s * y[3]);
            let dist = ((r.b - g.n[0] * cx - g.n[1] * cy).hypot(r.depth - g.v[0] * cx - g.v[1] * cy) - y[4]).abs();
            out.push(g.scan, scans[g.scan as usize].file_row(r.pulse), r.echo, node, pw, dist);
        }
    }
}

/// A covariance of `[c1, c2, t1, t2, log R]` with log R's sd `f` times as wide.
fn widened(cov: &Matrix5<f64>, f: f64) -> Matrix5<f64> {
    let mut c = *cov;
    c.row_mut(4).scale_mut(f);
    c.column_mut(4).scale_mut(f);
    c
}

impl Fit {
    /// Appends a piece or row: its state `[c1, c2, t1, t2, log R]` and covariance (log R's widened: [`WIDEN`]), frame
    /// and claimed start `a0`.
    fn push(&mut self, x: &Vector5<f64>, cov: &Matrix5<f64>, (u, e1, e2, len): Frame, a0: Point, own: Own) {
        let at = |s: f64| -> Point { std::array::from_fn(|i| a0[i] + (s + 0.5 * len) * u[i] + (x[0] + s * x[2]) * e1[i] + (x[1] + s * x[3]) * e2[i]) };
        self.a.push(at(-0.5 * len));
        self.b.push(at(0.5 * len));
        self.r.push(x[4].exp());
        let sd = |i: usize| cov[(i, i)].max(0.0).sqrt();
        self.sd_log_r.push(sd(4));
        self.centre.push([x[0], x[1]]);
        self.lean.push([x[2], x[3]]);
        self.e1.push(e1);
        self.e2.push(e2);
        self.r_lo.push(own.r_lo);
        self.r_hi.push(own.r_hi);
        self.p_wood.push(own.p_wood);
        self.flags.push(own.flags);
        self.n_scans.push(own.n_scans);
        self.k_stems.push(own.k_stems);
        self.diag.push(own.diag);
        self.data_share.push(own.data_share);
        self.contested.push(own.contested);
        self.deleted.push(own.deleted);
        self.r_cap.push(own.r_cap);
        let (stop, pass) = own.audit;
        self.n_pass.push(pass);
        self.pass_share.push(if stop + pass > 0 { f64::from(pass) / f64::from(stop + pass) } else { f64::NAN });
        self.basis.push(own.basis);
    }
}

/// A row's state from its pieces `k0..k1` (a chain, each `pl` long, the row `len`): the line from the first's start
/// to the last's end as the centre at the row's middle and the lean, and the log of their root mean square R, so the
/// row's volume is its pieces' (3dtrees ticket 109: one volume per tree); with its covariance from the chain's joint
/// covariance (`Cov(x_j, x_i) = -Y_j Cov(x_{j-1}, x_i)` below the diagonal), log R's linearised (each piece's weight
/// its share of the row's area) with each piece's log R widened by its factor `k_of` (its basis's: [`WIDEN`]).
fn row_line(sol: &Solved, k0: usize, k1: usize, len: f64, pl: f64, k_of: &[f64]) -> (Vector5<f64>, Matrix5<f64>) {
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
    let area: Vec<f64> = (k0..k1).map(|k| (2.0 * sol.x[k][4]).exp()).collect();
    let total: f64 = area.iter().sum();
    let mut x: Vector5<f64> = (0..m).map(|j| a[j] * sol.x[k0 + j]).sum();
    x[4] = 0.5 * (total / m as f64).ln();
    for ((aj, s), k) in a.iter_mut().zip(&area).zip(k_of) {
        aj[(4, 4)] = k * s / total;
    }
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
    /// The fork terms' weights the solve ended with (1 where a piece has none).
    fork_w: Vec<f64>,
    /// Each piece's bounds' site precision on log R (0 where it has none), and its cavity's mean (NaN where it has no
    /// bounds).
    site_tau: Vec<f64>,
    cavity: Vec<f64>,
    /// Each piece's covariance given its parent (`H_k^-1` after its children's elimination): with `y`, the tree's
    /// joint posterior as a chain of conditionals, root first.
    inv: Vec<Matrix5<f64>>,
}

/// The tree's posterior (see the module documentation): the prior (its fork terms reweighted as a Student-t's), the
/// used pieces' summaries linearised in log R, and the bounded pieces' bounds as expectation-propagation sites on
/// log R, all re-solved until nothing moves; and whether that happened within `rounds` (at least 3).
fn tree_solve(rows: &Rows, fr: &[Frame], nd: &Nodes, sums: &[Summary], p: &PriorParams, rounds: usize) -> Result<(Solved, bool)> {
    let n = nd.row.len();
    let forks = fork_means(rows, fr, nd, p);
    let mut fork_w = vec![1.0; n];
    // the prior without its fork terms, once; each round adds those at their weights
    let (h_prior, c_prior, g_prior) = prior_system(rows, nd, p, &forks, &vec![0.0; n]);
    let mut lin: Vec<f64> = (0..n).map(|k| sums[k].used().map_or(rows.r_prior[nd.row[k]].ln(), |c| c.x[4].ln())).collect();
    let terms = terms(sums);
    // each piece's site: precision and precision times mean, on its log R
    let mut site = vec![(0.0, 0.0); terms.len()];
    // each site's damping and its last step: a site whose step turns back is damped harder (an oscillating site,
    // a base piece's bounds against its linked measurements, held one tree's solve for its 300 rounds)
    let mut damp = vec![EP_DAMP; terms.len()];
    let mut last = vec![(0.0f64, 0.0f64); terms.len()];
    // and each fork weight's (the Student-t's reweighting turns with its sites at a fork), at first in full
    let (mut fork_damp, mut fork_last) = (vec![1.0f64; n], vec![0.0f64; n]);
    let mut round = 0;
    loop {
        let (mut h, mut c0, mut g) = (h_prior.clone(), c_prior.clone(), g_prior.clone());
        for k in 0..n {
            let (Some(m), Some(q)) = (forks[k], nd.parent[k]) else { continue };
            let w = fork_w[k] / (p.sig_fork * p.sig_fork);
            h[k][(4, 4)] += w;
            h[q][(4, 4)] += w;
            c0[k][(4, 4)] -= w;
            g[k][4] += w * m;
            g[q][4] -= w * m;
        }
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
        // each fork term's weight, (nu + 1) / (nu + E[r^2] / sig^2) at the posterior (expectation-maximisation for
        // the Student-t's scale mixture); the child's and parent's covariance, -Y_k Sigma_q
        for k in 0..n {
            let (Some(m), Some(q)) = (forks[k], nd.parent[k]) else { continue };
            let cross = -(sol.y[k] * sol.cov[q])[(4, 4)];
            let e2 = (sol.x[k][4] - sol.x[q][4] - m).powi(2) + sol.cov[k][(4, 4)] + sol.cov[q][(4, 4)] - 2.0 * cross;
            let step = (FORK_DF + 1.0) / (FORK_DF + e2 / (p.sig_fork * p.sig_fork)) - fork_w[k];
            fork_damp[k] = if step * fork_last[k] < 0.0 { (0.5 * fork_damp[k]).max(EP_DAMP_MIN) } else { (EP_REGROW * fork_damp[k]).min(1.0) };
            fork_last[k] = step;
            // converged on the step it would take undamped: a damped step is small however far it has to go
            moved = moved.max(step.abs());
            fork_w[k] += fork_damp[k] * step;
        }
        for (t, (k, bs)) in terms.iter().enumerate() {
            let k = *k;
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
            let step = (tau_new - tau, nu_new - nu);
            damp[t] = if step.0 * last[t].0 < 0.0 || step.1 * last[t].1 < 0.0 { (0.5 * damp[t]).max(EP_DAMP_MIN) } else { (EP_REGROW * damp[t]).min(1.0) };
            last[t] = step;
            moved = moved.max(step.0.abs() * v).max(step.1.abs() * v.sqrt());
            site[t] = (tau + damp[t] * step.0, nu + damp[t] * step.1);
        }
        round += 1;
        if (round >= 3 && moved < EP_TOL) || round >= rounds.max(3) {
            // each bounded piece's site precision, and its cavity's mean (its marginal without its own bounds)
            let (mut site_tau, mut cavity) = (vec![0.0; n], vec![f64::NAN; n]);
            for (&(k, _), &(tau, nu)) in terms.iter().zip(&site) {
                let (v, mu) = (sol.cov[k][(4, 4)], sol.x[k][4]);
                site_tau[k] = tau;
                if 1.0 / v - tau > 0.0 {
                    cavity[k] = (mu / v - nu) / (1.0 / v - tau);
                }
            }
            return Ok((Solved { fork_w, site_tau, cavity, ..sol }, moved < EP_TOL));
        }
    }
}

/// The censored terms of each piece that has any (see the module documentation): its bounds where no circle of its
/// own enters the tree, a routed piece's without its detection bound (its woody run can span several stems), and a
/// shadowed exact stem's own beside its message.
fn terms(sums: &[Summary]) -> Vec<(usize, Vec<Bound>)> {
    (0..sums.len()).filter(|&k| sums[k].bounded()).map(|k| {
        let routed = matches!(sums[k].outcome, Outcome::Routed { .. }) && !sums[k].shadowed;
        (k, sums[k].bounds.iter().copied().filter(|b| !(routed && b.side == Side::Lower)).collect::<Vec<_>>())
    }).filter(|t| !t.1.is_empty()).collect()
}

/// The mean and variance of `N(mu, v)` times a piece's probit terms `bs`, by the trapezoid rule on [`EP_NODES`]
/// points across `at.0 ± 6 at.1` (the piece's marginal, which the tilted distribution approaches), or where its mass
/// peaks at that grid's edge (the cavity far from the bounds, as at a tree's base: research 93 §19), across the cavity's
/// ±8 sd and the bounds and again about the peak; None where the grid holds no mass. One site per piece makes the
/// expectation propagation exact for a piece alone (a site per
/// term would not: two bounds on one piece leave its sd ~10 % wide); across linked pieces it is within ~0.1 sd of
/// the mean and ~12 % of the sd (research 93 §17 review, against brute-force quadrature).
fn tilted(mu: f64, v: f64, at: (f64, f64), bs: &[Bound]) -> Option<(f64, f64)> {
    let lf = |x: f64| -0.5 * (x - mu).powi(2) / v + bs.iter().map(|b| b.ln_p(x)).sum::<f64>();
    // the trapezoid rule on n points across [lo, hi]: the mean, the variance, the spacing and where the mass peaks
    let grid = |lo: f64, hi: f64, n: usize| -> Option<(f64, f64, f64, usize)> {
        let h = (hi - lo) / (n - 1) as f64;
        let l: Vec<f64> = (0..n).map(|i| lf(lo + i as f64 * h)).collect();
        let (top_i, top) = l.iter().copied().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, x)| if x > b.1 { (i, x) } else { b });
        if !top.is_finite() {
            return None;
        }
        let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
        for (i, &li) in l.iter().enumerate() {
            let x = lo + i as f64 * h;
            let w = (li - top).exp() * if i == 0 || i + 1 == n { 0.5 } else { 1.0 };
            (z, m1, m2) = (z + w, m1 + w * x, m2 + w * x * x);
        }
        let m = m1 / z;
        Some((m, (m2 / z - m * m).max(0.0), h, top_i))
    };
    let (m, var, _, top) = grid(at.0 - 6.0 * at.1, at.0 + 6.0 * at.1, EP_NODES)?;
    if top != 0 && top != EP_NODES - 1 {
        return Some((m, var.max(1e-6 * at.1 * at.1)));
    }
    // the mass lies beyond the marginal's grid (a cavity far from its bounds, as in the first rounds): across the
    // cavity and the bounds, then again about the peak where that grid is coarse for it
    let reach = |b: &Bound| 4.0 * b.sd.min(1.0);
    let lo = bs.iter().map(|b| b.eta - reach(b)).fold(mu - 8.0 * v.sqrt(), f64::min);
    let hi = bs.iter().map(|b| b.eta + reach(b)).fold(mu + 8.0 * v.sqrt(), f64::max);
    let (m, var, h, top) = grid(lo, hi, 4 * EP_NODES)?;
    if var.sqrt() > 2.0 * h {
        return Some((m, var));
    }
    let x = lo + top as f64 * h;
    let (m, var, h, _) = grid(x - 12.0 * h, x + 12.0 * h, EP_NODES)?;
    Some((m, var.max(1e-4 * h * h)))
}

/// Each piece's chance of being wood: its evidence (`log_odds`) under a chain along the branches (a base piece is
/// wood with `p.exist`, a piece after one that isn't with `p.exist_off`, one after wood with the chance that keeps
/// every piece's prior at `p.exist` however far out it is), exactly by sum-product, leaves up then root down.
fn existence(parent: &[Option<usize>], log_odds: &[f64], p: &PriorParams) -> Vec<f64> {
    let n = parent.len();
    let exist_on = 1.0 - (1.0 - p.exist) * p.exist_off / p.exist;
    let trans = |ep: usize, ek: usize| {
        let on = if ep == 1 { exist_on } else { p.exist_off };
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
            None => [1.0 - p.exist, p.exist],
            Some(q) => {
                let pq = [bel[q][0] / msg[k][0], bel[q][1] / msg[k][1]];
                std::array::from_fn(|ek| trans(0, ek) * pq[0] + trans(1, ek) * pq[1])
            }
        };
        bel[k] = norm([up[k][0] * down[0], up[k][1] * down[1]]);
    }
    bel.iter().map(|b| b[1]).collect()
}

/// Each piece's fork term's mean, where it is a fork's child (a row's first piece whose parent row forks): the log
/// R step `(mu_fork + log share) / 2`, the share by subtree wood length.
fn fork_means(rows: &Rows, fr: &[Frame], nd: &Nodes, p: &PriorParams) -> Vec<Option<f64>> {
    let share = rows.fork_share(&fr.iter().map(|f| f.3).collect::<Vec<_>>(), ForkShare::WoodLength);
    (0..nd.row.len()).map(|k| {
        let row = nd.row[k];
        // a further stem: beside its piece, by its share of the stems' area; continuing a stem, none
        if let Some(f) = nd.further[k] {
            return f.share.filter(|_| nd.parent[k].is_some()).map(|s| 0.5 * (p.mu_fork + s.ln()));
        }
        (nd.parent[k].is_some() && nd.first[row] == k && share[row].is_finite()).then(|| 0.5 * (p.mu_fork + share[row].ln()))
    }).collect()
}

/// The prior's normal equations on the pieces: each piece's block, its coupling to its parent's, and the
/// right-hand side (the prior is Gaussian in `[c1, c2, t1, t2, log R]` given the fork terms' weights `fork_w`). A
/// row's first piece takes the row's taper step or fork share (`forks`); its later pieces continue it, with no
/// step.
#[allow(clippy::type_complexity)]
fn prior_system(rows: &Rows, nd: &Nodes, p: &PriorParams, forks: &[Option<f64>], fork_w: &[f64]) -> (Vec<Matrix5<f64>>, Vec<Matrix5<f64>>, Vec<Vector5<f64>>) {
    let n = nd.row.len();
    let eta0: Vec<f64> = nd.row.iter().map(|&k| rows.r_prior[k].ln()).collect();
    // a factor: sum of coefficients times (piece, unknown), its mean and sd
    type Factor = (Vec<(usize, usize, f64)>, f64, f64);
    let mut factors: Vec<Factor> = Vec::new();
    for k in 0..n {
        let row = nd.row[k];
        let sc = p.centre_rel * rows.r_prior[row] + p.centre_abs;
        factors.push((vec![(k, 0, 1.0)], 0.0, sc));
        factors.push((vec![(k, 1, 1.0)], 0.0, sc));
        factors.push((vec![(k, 2, 1.0)], 0.0, p.lean_sd));
        factors.push((vec![(k, 3, 1.0)], 0.0, p.lean_sd));
        let Some(q) = nd.parent[k] else {
            factors.push((vec![(k, 4, 1.0)], eta0[k], p.sig_root));
            continue;
        };
        let len = nd.frame[k].3;
        let ls = len.max(0.01).sqrt();
        if let Some(m) = forks[k] {
            factors.push((vec![(k, 4, 1.0), (q, 4, -1.0)], m, p.sig_fork / fork_w[k].sqrt()));
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
                factors.push((vec![(k, 2 + j, 1.0), (q, 2 + j, -1.0)], 0.0, p.lean_bend * ls));
            }
        }
    }
    // the prior's precision, as piece blocks and child-parent couplings
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
    Ok(Solved { x, cov, y, fork_w: Vec::new(), site_tau: Vec::new(), cavity: Vec::new(), inv })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qsm::joint::frames;
    use crate::qsm::joint_fit::PriorParams;

    #[test]
    fn context_is_the_other_trees_cylinders_within_reach() {
        // tree 0: a 2 m stem at the origin; tree 1: a stem 0.4 m away and one of zero length beside it; tree 2: 3 m away
        let a = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.5], [3.0, 0.0, 0.0]];
        let b = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.4, 0.0, 2.0], [0.1, 0.0, 0.5], [3.0, 0.0, 2.0]];
        let (off, idx) = tree_context(&a, &b, &[0, 2, 4, 5], &[0, 1, 2], 0.5);
        assert_eq!(off, vec![0, 1, 3, 3]);
        assert_eq!(idx, vec![2, 0, 1]); // tree 0's: 1's stem, not the zero-length one; tree 1's: 0's two; tree 2's: none
        assert_eq!(segment_distance(&a[0], &b[0], &a[2], &b[2]), 0.4);
        assert_eq!(tree_context(&a, &b, &[0, 2, 4, 5], &[0], 0.0), (vec![0, 0], vec![])); // no reach: nothing close
        assert_eq!(segment_distance(&a[0], &b[0], &[1.0, 0.0, 3.0], &[1.0, 0.0, 4.0]), 5f64.sqrt()); // end to end
    }

    fn like() -> JointParams {
        JointParams { tau: 0.17, edge: 0.25, hide: 0.0, ..JointParams::default() }
    }

    #[test]
    fn a_scanned_stem_is_measured_from_one_scan() {
        // the evidence tests' fan: a 5 cm stem at 5 m, claimed at 8 cm 1 cm off its axis
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let (a1, b1) = ([a[0] + 0.01, a[1], a[2]], [b[0] + 0.01, b[1], b[2]]);
        let rows = Rows { a: &[a1], b: &[b1], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let t = support_tree(&[&scan], &rows, &[], &like(), &PriorParams::default(), 2).unwrap();
        let (f, q) = (&t.rows, &t.pieces);
        // the 2 m row is seven pieces of the tree; the fan sees the middle three, and they are measured
        assert_eq!(t.piece_row, vec![0; 7]);
        let seen: Vec<usize> = (0..7).filter(|&j| q.n_scans[j] > 0).collect();
        assert_eq!(seen, vec![2, 3, 4], "scans {:?}", q.n_scans);
        assert!(seen.iter().all(|&j| q.flags[j] == 0 && (q.r[j] - 0.05).abs() < 0.0025 && q.n_scans[j] == 1), "flags {:?} r {:?}", q.flags, q.r);
        // the others are the prior's along the chain, with nothing to bound them; the row says so
        assert!((0..7).filter(|j| !seen.contains(j)).all(|j| q.flags[j] == flag::BOUNDS && q.r_lo[j].is_nan() && q.r_hi[j].is_nan()));
        assert_eq!(f.flags[0], flag::BOUNDS);
        assert!((f.r[0] - 0.05).abs() < 0.0025, "R {}", f.r[0]);
        // the row runs from its first piece's start to its last's end
        assert!((0..3).all(|i| (f.a[0][i] - q.a[0][i]).abs() < 1e-9 && (f.b[0][i] - q.b[6][i]).abs() < 1e-9));
        // and its volume is its pieces'
        let area: f64 = q.r.iter().map(|r| r * r).sum::<f64>() / 7.0;
        assert!((f.r[0] * f.r[0] - area).abs() < 1e-12 * area, "{} against {}", f.r[0], area.sqrt());
        // an opaque bark-free stem is wood, at the most one scan can say, and lifts the unseen pieces it carries on from
        let pp = PriorParams::default();
        assert!(seen.iter().all(|&j| q.p_wood[j] > 0.75) && q.p_wood.iter().all(|&p| p > pp.exist), "p_wood {:?}", q.p_wood);
        // its diagnostics: one scan, both edges seen; its own data give most of what the tree says of its measured
        // pieces' R, none of the others'
        assert!(seen.iter().all(|&j| q.diag[j] == diag::ONE_SIDED && q.data_share[j] > 0.5 && q.data_share[j] <= 1.0 && q.deleted[j] == 0.0 && (0.0..=1.0).contains(&q.contested[j])), "diag {:?} share {:?}", q.diag, q.data_share);
        assert!((0..7).filter(|j| !seen.contains(j)).all(|j| q.diag[j] == 0 && q.data_share[j] == 0.0 && q.contested[j].is_nan() && q.deleted[j].is_nan()), "diag {:?} share {:?}", q.diag, q.data_share);
        assert_eq!(f.diag[0], diag::ONE_SIDED);

        // the part the fan sees, 48 cm in two pieces: measured, the claim moved back by 1 cm, upright
        let (a2, b2) = ([a1[0], a1[1], -0.24], [b1[0], b1[1], 0.24]);
        let rows = Rows { a: &[a2], b: &[b2], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let t = support_tree(&[&scan], &rows, &[], &like(), &PriorParams::default(), 2).unwrap();
        let (f, q) = (&t.rows, &t.pieces);
        assert!(t.piece_row == vec![0, 0] && f.flags[0] == 0 && q.flags == vec![0, 0], "flags {:?}", q.flags);
        assert!((f.r[0] - 0.05).abs() < 0.0025 && f.sd_log_r[0] / widen(f.basis[0]) < 0.1, "R {}", f.r[0]);
        assert!((0.5 * (f.a[0][0] + f.b[0][0])).abs() < 0.002 && f.lean[0][1].abs() < 0.02, "x {} {} lean {:?}", f.a[0][0], f.b[0][0], f.lean[0]);
    }

    /// An accepted exact fit of `stems`, none shadowed.
    fn exact(stems: &[Circle]) -> routed::Exact {
        let n = stems.len();
        routed::Exact { stems: stems.to_vec(), echoes: Echoes::default(), shadow: vec![None; n], tube: 0.2, n_ctx: vec![0; n] }
    }

    fn circle(x: Vector5<f64>, sd: f64) -> Circle {
        let info = Matrix5::<f64>::identity() / (sd * sd);
        Circle { st: Stats { info, h: info * x, yy: x.dot(&(info * x)), n: 5 }, x, cov: Matrix5::identity() * sd * sd, chi2: f64::NAN }
    }

    fn summary(outcome: Outcome, bounds: Vec<Bound>) -> Summary {
        Summary { outcome, chi2_front: f64::NAN, n_runs: 1, n_scans: 1, bounds, log_odds: 0.0, seeds: Vec::new(), exact: None, seen: Vec::new(), shadowed: false, points: None,
            diag: DiagIn { tube: 0.1, w: 0.005, one_sided: false, deleted: 0.0, n_ctx: 0, edge: false } }
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
        let routed = summary(Outcome::Routed { runs: true, hull: false }, Vec::new());
        let sums = vec![summary(Outcome::Bounded, Vec::new()), summary(Outcome::Bounded, Vec::new()), summary(Outcome::Measured(c), Vec::new()), routed, summary(Outcome::Bounded, Vec::new())];
        let pp = PriorParams::default();
        let (sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        assert!(converged);
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
        let (h, c, g) = prior_system(&rows, &nd, &pp, &fork_means(&rows, &fr, &nd, &pp), &[1.0; 5]);
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
        let (xr, cr) = row_line(&se, 0, 2, 0.6, 0.3, &[1.0, 1.0]);
        let mut a = nalgebra::DVector::<f64>::zeros(5 * n);
        (a[4], a[9]) = (0.5, 0.5);
        assert!((xr[4] - (0.5 * (xd[4] + xd[9]))).abs() < 1e-12 && (cr[(4, 4)] - (a.transpose() * &inv * &a)[0]).abs() < 1e-12);
        // a row of mixed bases is widened piece by piece: its log R's variance is the pieces' widened, its line not
        let (xw, cw) = row_line(&se, 0, 2, 0.6, 0.3, &[2.0, 3.0]);
        (a[4], a[9]) = (1.0, 1.5);
        assert!(xw == xr && (cw[(4, 4)] - (a.transpose() * &inv * &a)[0]).abs() < 1e-12 && cw[(2, 2)] == cr[(2, 2)]);
        let mut l = nalgebra::DVector::<f64>::zeros(5 * n);
        (l[0], l[2], l[5], l[7]) = (-1.0 / 0.6, 0.25, 1.0 / 0.6, 0.25);
        assert!((cr[(2, 2)] - (l.transpose() * &inv * &l)[0]).abs() < 1e-12, "{} {}", cr[(2, 2)], (l.transpose() * &inv * &l)[0]);
        // a block that isn't positive definite is an error, not a silent zero
        let bad = vec![Matrix5::<f64>::zeros()];
        assert!(eliminate(&[None], bad, &[Matrix5::zeros()], vec![Vector5::zeros()]).is_err());
    }

    #[test]
    fn tilted_moments_reach_bounds_far_from_the_marginal() {
        // a cavity at log R -4.7 (sd 0.16) and two-sided bounds 15 sd away at -2.25 (a tree's base piece): the
        // moments lie between the bounds, as brute force on a fine grid says, not at the marginal's grid's edge
        let bs = [Bound { eta: -2.28, sd: 0.04, side: Side::Lower }, Bound { eta: -2.22, sd: 0.05, side: Side::Upper }];
        let (mu, v) = (-4.7, 0.16f64.powi(2));
        let (m, var) = tilted(mu, v, (mu, v.sqrt()), &bs).unwrap();
        let xs: Vec<f64> = (0..200_001).map(|i| -8.0 + 7.0 * i as f64 / 200_000.0).collect();
        let lw: Vec<f64> = xs.iter().map(|&x| -0.5 * (x - mu).powi(2) / v + bs.iter().map(|b| b.ln_p(x)).sum::<f64>()).collect();
        let top = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (z, m1, m2) = xs.iter().zip(&lw).fold((0.0, 0.0, 0.0), |(z, a, b), (&x, &l)| {
            let w = (l - top).exp();
            (z + w, a + w * x, b + w * x * x)
        });
        let (bm, bv) = (m1 / z, m2 / z - (m1 / z).powi(2));
        assert!((m - bm).abs() < 0.05 * bv.sqrt() && (var.sqrt() / bv.sqrt() - 1.0).abs() < 0.05, "{m} {} against {bm} {}", var.sqrt(), bv.sqrt());
        // a marginal that holds the mass: the moments of its own grid, unchanged
        let (m0, v0) = tilted(-2.25, 0.03f64.powi(2), (-2.25, 0.03), &bs).unwrap();
        assert!(m0 > -2.3 && m0 < -2.2 && v0 > 0.0);
    }

    #[test]
    fn bounds_cut_the_prior_to_the_moments_of_its_truncation() {
        // one base piece, its log R a priori N(log 1 cm, sig_root): an upper bound at 6 mm, then also a lower at 3 mm
        let rows = Rows { a: &[[0.0, 0.0, 0.0]], b: &[[0.0, 0.0, 0.2]], r_prior: &[0.01], r_max: &[], parent: &[-1] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let pp = PriorParams::default();
        let up = Bound { eta: 0.006f64.ln(), sd: 0.1, side: Side::Upper };
        let lo = Bound { eta: 0.003f64.ln(), sd: 0.1, side: Side::Lower };
        // the posterior's mean and sd by quadrature
        let truth = |bs: &[Bound]| {
            let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
            for i in 0..200_000 {
                let e = 0.01f64.ln() - 4.0 + 8.0 * i as f64 / 200_000.0;
                let mut w = (-0.5 * ((e - 0.01f64.ln()) / pp.sig_root).powi(2)).exp();
                for b in bs {
                    w *= b.ln_p(e).exp();
                }
                (z, m1, m2) = (z + w, m1 + w * e, m2 + w * e * e);
            }
            (m1 / z, (m2 / z - (m1 / z).powi(2)).sqrt())
        };
        for bs in [vec![up], vec![up, lo]] {
            let sol = tree_solve(&rows, &fr, &nd, &[summary(Outcome::Bounded, bs.clone())], &pp, EP_ROUNDS).unwrap().0;
            let (m, s) = truth(&bs);
            let (me, se) = (sol.x[0][4], sol.cov[0][(4, 4)].sqrt());
            // one piece: its site is exact (up to the quadrature)
            assert!((me - m).abs() < 0.01 * s && (se - s).abs() < 0.01 * s, "{} bounds: {me} {se} against {m} {s}", bs.len());
            assert!(se < pp.sig_root && me < 0.006f64.ln() + 0.1);
        }
    }

    /// A chain's log-R posterior by brute force on a grid of `m` points per piece across `±8` prior sds: `ln_prior`
    /// the log prior density of the pieces' log R, `bounds` each piece's probits. Its means and sds.
    fn brute(n: usize, centre: &[f64], half: &[f64], m: usize, ln_prior: impl Fn(&[f64]) -> f64, bounds: &[Vec<Bound>]) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n];
        let (mut z, mut m1, mut m2) = (0.0, vec![0.0; n], vec![0.0; n]);
        let mut lp = Vec::with_capacity(m.pow(n as u32));
        for idx in 0..m.pow(n as u32) {
            let mut r = idx;
            for k in 0..n {
                x[k] = centre[k] - half[k] + 2.0 * half[k] * (r % m) as f64 / (m - 1) as f64;
                r /= m;
            }
            lp.push(ln_prior(&x) + (0..n).map(|k| bounds[k].iter().map(|b| b.ln_p(x[k])).sum::<f64>()).sum::<f64>());
        }
        let top = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (idx, l) in lp.iter().enumerate() {
            let w = (l - top).exp();
            let mut r = idx;
            z += w;
            for k in 0..n {
                let xk = centre[k] - half[k] + 2.0 * half[k] * (r % m) as f64 / (m - 1) as f64;
                r /= m;
                m1[k] += w * xk;
                m2[k] += w * xk * xk;
            }
        }
        let mean: Vec<f64> = m1.iter().map(|v| v / z).collect();
        let sd = (0..n).map(|k| (m2[k] / z - mean[k] * mean[k]).sqrt()).collect();
        (mean, sd)
    }

    #[test]
    fn expectation_propagation_holds_on_linked_pieces() {
        let pp = PriorParams::default();
        let ln_n = |x: f64, m: f64, s: f64| -0.5 * ((x - m) / s).powi(2);
        let b = |r: f64, sd: f64, side: Side| Bound { eta: r.ln(), sd, side };
        let check = |name: &str, (me, se): (Vec<f64>, Vec<f64>), (mt, st): (Vec<f64>, Vec<f64>)| {
            for k in 0..me.len() {
                assert!((me[k] - mt[k]).abs() < 0.1 * st[k] && (se[k] / st[k] - 1.0).abs() < 0.15, "{name} piece {k}: EP {} {} against {} {}", me[k], se[k], mt[k], st[k]);
            }
        };
        // a 0.9 m row of three pieces, 15 cm a priori, two of them bounded well below that (research 93 §17 review)
        let rows = Rows { a: &[[0.0, 0.0, 0.0]], b: &[[0.0, 0.0, 0.9]], r_prior: &[0.15], r_max: &[], parent: &[-1] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let bs = vec![Vec::new(), vec![b(0.05, 0.05, Side::Upper), b(0.03, 0.3, Side::Lower)], vec![b(0.05, 0.05, Side::Upper)]];
        let sums: Vec<Summary> = bs.iter().map(|v| summary(Outcome::Bounded, v.clone())).collect();
        let (sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        assert!(converged);
        let step = pp.sig_eta * 0.3f64.sqrt();
        let e0 = 0.15f64.ln();
        let truth = brute(3, &[e0 - 1.5; 3], &[2.5; 3], 121, |x| ln_n(x[0], e0, pp.sig_root) + ln_n(x[1], x[0], step) + ln_n(x[2], x[1], step), &bs);
        check("chain", ((0..3).map(|k| sol.x[k][4]).collect(), (0..3).map(|k| sol.cov[k][(4, 4)].sqrt()).collect()), truth);

        // a fork: a base and two children by wood length 0.3 : 0.095, the larger child bounded below its share and the
        // smaller one above it; the fork terms Gaussian at the weights the solve settled on (expectation propagation
        // is checked here, not the Student-t's reweighting)
        let (ra, rb) = ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.3]], [[0.0, 0.0, 0.3], [0.0, 0.1, 0.58], [0.0, -0.03, 0.39]]);
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.1, 0.08, 0.04], r_max: &[], parent: &[-1, 0, 0] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let fm = fork_means(&rows, &fr, &nd, &pp);
        let bs = vec![Vec::new(), vec![b(0.03, 0.1, Side::Upper)], vec![b(0.04, 0.1, Side::Lower)]];
        let sums: Vec<Summary> = bs.iter().map(|v| summary(Outcome::Bounded, v.clone())).collect();
        let (sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        assert!(converged);
        let sf = |k: usize| pp.sig_fork / sol.fork_w[k].sqrt();
        assert!(sol.fork_w[1] < 1.0 && sol.fork_w[2] < 1.0, "{:?}", sol.fork_w);
        let e0 = 0.1f64.ln();
        let truth = brute(3, &[e0 - 0.5, e0 - 1.0, e0 - 1.0], &[2.0, 2.0, 2.0], 121, |x| {
            ln_n(x[0], e0, pp.sig_root) + ln_n(x[1], x[0] + fm[1].unwrap(), sf(1)) + ln_n(x[2], x[0] + fm[2].unwrap(), sf(2))
        }, &bs);
        check("fork", ((0..3).map(|k| sol.x[k][4]).collect(), (0..3).map(|k| sol.cov[k][(4, 4)].sqrt()).collect()), truth);
    }

    #[test]
    fn a_small_child_measured_fat_does_not_inflate_its_parent() {
        // an unmeasured stem forks into a long leader (2 m of wood) measured at 14 cm, and a 3 cm shoot measured at
        // 6 cm: by wood length the shoot's share is 1.5 %, so it says the stem is ~6 / sqrt(0.015) ~ 46 cm and the
        // leader says ~13.6 cm. Gaussian fork terms would split the difference (~25 cm); heavy tails keep the leader's.
        let (ra, rb) = ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.3]], [[0.0, 0.0, 0.3], [0.0, 0.0, 2.3], [0.03, 0.0, 0.3]]);
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.15, 0.14, 0.02], r_max: &[], parent: &[-1, 0, 0] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let n = nd.row.len();
        let mut sums: Vec<Summary> = (0..n).map(|_| summary(Outcome::Bounded, Vec::new())).collect();
        sums[1] = summary(Outcome::Measured(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.14), 1e-3)), Vec::new());
        sums[n - 1] = summary(Outcome::Measured(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.06), 1e-3)), Vec::new());
        let (sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &PriorParams::default(), EP_ROUNDS).unwrap();
        assert!(converged);
        let (stem, shoot) = (sol.x[0][4].exp(), sol.x[n - 1][4].exp());
        assert!((shoot - 0.06).abs() < 0.002 && stem > 0.12 && stem < 0.17, "stem {stem} shoot {shoot}");
        assert!(sol.fork_w[n - 1] < 0.1 && sol.fork_w[1] > 0.5, "{:?}", sol.fork_w);
    }

    #[test]
    fn a_profile_bounds_what_it_sees_and_no_more() {
        let p = JointParams { tau: 0.17, ..JointParams::default() };
        let w = 0.005;
        let rec = |b: f64, s: f64, depth: f64| Rec { b, s, depth, xy: [f64::NAN; 2], w, refl: f64::NAN, single: true, blocked: false, lost: false, pulse: 0, echo: 0 };
        let group = |recs: Vec<Rec>| Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs };
        // a 2 cm stem across a grid of pulses 5 mm apart both ways: hit within its silhouette (R + w (1/2 - tau)),
        // passed beside it
        let (r, app) = (0.02, p.app(w));
        let grid = |keep: &dyn Fn(f64) -> bool| {
            let mut v = Vec::new();
            for j in 0..20 {
                for i in -12..=12 {
                    let b = i as f64 * 0.005;
                    if b.abs() < r + app {
                        v.push(rec(b, j as f64 * 0.005, -(r * r - b.min(r).powi(2)).max(0.0).sqrt()));
                    } else if keep(b) {
                        v.push(rec(b, j as f64 * 0.005, f64::NAN));
                    }
                }
            }
            group(v)
        };
        let g = grid(&|_| true);
        let q = Profile::of(&g, 0.03, &p);
        let bs = q.bounds(0.03, &p, None);
        let side = |s: Side| bs.iter().find(|b| b.side == s).map(|b| b.eta.exp());
        let (lo, hi) = (side(Side::Lower).unwrap(), side(Side::Upper).unwrap());
        assert!(lo <= r && r <= hi && hi - lo < 0.006, "lo {lo} hi {hi}");
        assert_eq!(q.woody_w(0.03), Some(w));
        // a captor refused at the tube's edge (TUBE_EDGE) takes only the lower bound of a scan whose woody run it
        // spans: a 4 cm circle 1 cm off spans the stem's run, a 3 cm one 9 cm off (its silhouette from 6 cm) doesn't
        let captor = |c: f64, r: f64| q.bounds(0.03, &p, Some(&Vector5::new(c, 0.0, 0.0, 0.0, r)));
        let spanned = captor(0.01, 0.04);
        assert!(spanned.iter().all(|b| b.side != Side::Lower) && spanned.iter().any(|b| b.side == Side::Upper), "{spanned:?}");
        assert_eq!(captor(0.09, 0.03), bs);
        // an export of echoes only keeps no misses: with nothing recorded beside the stem, no upper bound
        let g = grid(&|_| false);
        let bs = Profile::of(&g, 0.03, &p).bounds(0.03, &p, None);
        assert!(bs.iter().all(|b| b.side == Side::Lower), "{bs:?}");
        // a 1 mm wire the grid hits in three rows and passes through, either side of it, in the others: the passes
        // through bound it
        let mut v = Vec::new();
        for j in 0..20 {
            let s = j as f64 * 0.005;
            if j % 7 == 0 {
                v.extend([rec(-0.005, s, f64::NAN), rec(0.0, s, 0.0), rec(0.005, s, f64::NAN)]);
            } else {
                v.extend([rec(-0.0075, s, f64::NAN), rec(-0.0025, s, f64::NAN), rec(0.0025, s, f64::NAN), rec(0.0075, s, f64::NAN)]);
            }
        }
        let g = group(v);
        let bs = Profile::of(&g, 0.01, &p).bounds(0.01, &p, None);
        let hi = bs.iter().find(|b| b.side == Side::Upper).map(|b| b.eta.exp()).unwrap();
        assert!(hi >= 0.0005 && hi <= (0.0025 - app).max(0.5 * p.tau * w) + 1e-9, "hi {hi} {bs:?}");
        // and with no woody stop, no centre to bound about
        let g = group((0..20).map(|j| rec(0.0, j as f64 * 0.005, f64::NAN)).collect());
        assert!(Profile::of(&g, 0.01, &p).bounds(0.01, &p, None).is_empty());
    }

    #[test]
    fn existence_is_exact_on_the_branches() {
        // a chain that holds (2a's): wood after wood is near sure, after none rare
        let pp = PriorParams { exist: 0.8, exist_off: 0.02, ..PriorParams::default() };
        let exist_on = 1.0 - (1.0 - pp.exist) * pp.exist_off / pp.exist;
        // a base, its child, and a fork of two, by enumeration of the 16 states
        let parent = [None, Some(0), Some(1), Some(1)];
        let lo = [2.0, -1.0, 0.5, -3.0];
        let p = existence(&parent, &lo, &pp);
        let mut marg = [0.0; 4];
        let mut z = 0.0;
        for st in 0..16usize {
            let e = |k: usize| (st >> k) & 1;
            let mut w = 1.0;
            for k in 0..4 {
                w *= if e(k) == 1 { lo[k].exp() } else { 1.0 };
                let pk = match parent[k] {
                    None => pp.exist,
                    Some(q) => if e(q) == 1 { exist_on } else { pp.exist_off },
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
        let p = existence(&[None, Some(0)], &[-6.0, 4.0], &pp);
        let alone = existence(&[None], &[-6.0], &pp)[0];
        assert!(p[0] > 10.0 * alone && p[1] > p[0], "{p:?} {alone}");
        // and with no evidence, every piece keeps the prior however far out
        let p = existence(&(0..50usize).map(|k| k.checked_sub(1)).collect::<Vec<_>>(), &[0.0; 50], &pp);
        assert!(p.iter().all(|&x| (x - pp.exist).abs() < 1e-9), "{p:?}");
    }

    #[test]
    fn diagnostics_judge_the_nodes_own_circle() {
        // a 9 cm circle 2 cm off the claim's axis in a 10 cm tube: at its edge, though the tree shrank it to 5 cm
        let mut s = summary(Outcome::Bounded, Vec::new());
        s.exact = Some(circle(Vector5::new(0.02, 0.0, 0.0, 0.0, 0.09), 0.005));
        let (bits, share, contested) = diagnose(&s, 0.05, 0.01, 0.0, 0.25, (10, 4));
        assert_eq!(bits, diag::TUBE_EDGE | diag::CONTESTED);
        assert!((contested - 0.6).abs() < 1e-12 && share > 0.0 && share <= 1.0, "{share} {contested}");
        // narrower than half the footprint; no stops; a negative EP site leaves the share undetermined
        s.exact = Some(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.002), 0.001));
        let (bits, share, contested) = diagnose(&s, 0.002, 0.01, -1.0, 0.25, (0, 0));
        assert_eq!(bits, diag::SUB_FOOTPRINT);
        assert!(share.is_nan() && contested.is_nan());
    }

    #[test]
    fn one_edge_and_the_front_leave_the_radius_free() {
        let s = |n: [f64; 2], w: [f64; 3]| Sight { n, v: [-n[1], n[0]], w };
        // one scan's edges and front decide R, and its two edges alone; one edge and the front don't
        assert!(unseen(&[s([1.0, 0.0], [1.0, 1.0, 1.0])]).is_empty());
        assert!(unseen(&[s([1.0, 0.0], [1.0, 1.0, 0.0])]).is_empty());
        let free = unseen(&[s([1.0, 0.0], [1.0, 0.0, 1.0])]);
        assert_eq!(free.len(), 1);
        // the ridge: R grows as the centre moves away from the seen edge and from the scanner
        let r = free[0] / free[0][2];
        assert!((r - Vector3::new(-1.0, 1.0, 1.0)).norm() < 1e-9, "{r:?}");
        // the other edge seen by a tenth of the pulses decides it; by a fiftieth, not
        assert!(unseen(&[s([1.0, 0.0], [1.0, 0.1, 1.0])]).is_empty());
        assert!(!unseen(&[s([1.0, 0.0], [1.0, 0.02, 1.0])]).is_empty());
        // a second scan at right angles sees what the first can't; nothing seen leaves everything free
        assert!(unseen(&[s([1.0, 0.0], [1.0, 0.0, 1.0]), s([0.0, 1.0], [1.0, 0.0, 1.0])]).is_empty());
        assert_eq!(unseen(&[s([1.0, 0.0], [0.0; 3])]).len(), 3);
        // a measurement marginalised along the ridge says nothing of R with the centre unknown, nothing along the
        // ridge, and the rest (here the lean) as before
        let x = Vector5::new(0.01, 0.0, 0.0, 0.0, 0.05);
        let c0 = circle(x, 0.002);
        assert!((info_r(&c0.st.info) - 1.0 / 4e-6).abs() < 1e-3);
        let c = marginalise(&c0, &free);
        let rr = Vector5::new(free[0][0], free[0][1], 0.0, 0.0, free[0][2]);
        assert!((c.st.info * rr).norm() < 1e-9 * c.st.info.norm() && info_r(&c.st.info) < 1e-6 * info_r(&c0.st.info), "{:?}", c.st.info);
        assert!(c.st.info[(2, 2)] == c0.st.info[(2, 2)] && c.cov[(4, 4)] > 0.1);
        // and its estimate still solves its normal equations
        assert!((c.st.info * x - c.st.h).norm() < 1e-6 * c.st.h.norm().max(1.0));
    }

    #[test]
    fn misfit_counts_on_the_rank() {
        // a measurement set of rank 2 (two edges): no redundancy, no misfit; a third edge is redundant
        let n = [1.0, 0.0];
        let edges: Vec<Meas> = vec![([n[0], n[1], 0.0, 0.0, 1.0], 0.05, 1e-6), ([n[0], n[1], 0.0, 0.0, -1.0], -0.03, 1e-6), ([n[0], n[1], 0.0, 0.0, 1.0], 0.052, 1e-6)];
        let st = Stats::of(&edges);
        let ev = SymmetricEigen::new(st.info).eigenvalues;
        assert_eq!(ev.iter().filter(|&&e| e > 1e-9 * ev.max()).count(), 2);
        let c = solve_row(&st, 0.05, 0.5, 0.004).unwrap();
        assert!(c.chi2.is_finite() && c.chi2 > 0.5, "one redundant edge: chi2 {}", c.chi2);
    }

    #[test]
    fn echo_character_weighs_stops() {
        let p = JointParams::default();
        let rec = |refl: f64, single: bool| Rec { b: 0.0, s: 0.0, depth: 0.0, xy: [0.0; 2], w: 0.01, refl, single, blocked: false, lost: false, pulse: 0, echo: 0 };
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
    fn a_routed_piece_is_fitted_exactly_from_its_seeds() {
        // the fan's 5 cm stem; a 30 cm piece claimed 1 cm off it, its circle a hull of 8 cm on the claim
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let (a0, b0) = ([a[0] + 0.01, a[1], -0.15], [b[0] + 0.01, b[1], 0.15]);
        let fr = frames(&[a0], &[b0])[0];
        let (_, e1, e2, _) = fr;
        let claim = |w: Point| (dot(&w, &e1), dot(&w, &e2));
        let truth = claim([-0.01, 0.0, 0.0]);
        // seeds: the stem at 6 cm, and a phantom 15 cm beside it that nothing stops on
        let phantom = claim([0.14, 0.0, 0.0]);
        let seed = |c: (f64, f64), r: f64| routed::Seed { c: [c.0, c.1], t: [0.0, 0.0], r, t_var: [0.01, 0.01] };
        let seeds = [seed(truth, 0.06), seed(phantom, 0.03)];
        let e = accepted(routed::exact_piece(&[&scan], &a0, fr, &seeds, &[], f64::NEG_INFINITY, &like(), &PriorParams::default()).unwrap());
        assert_eq!(e.stems.len(), 1, "one stem by the same pulses' evidence");
        let c = &e.stems[0];
        assert!((c.x[4] - 0.05).abs() < 0.003 && (c.x[0] - truth.0).hypot(c.x[1] - truth.1) < 0.003, "{:?} against {truth:?}", c.x);
        // its message measures the centre and R, tempered: no sharper than the model's floor; the lean is its seed's
        let sd = |i: usize| c.cov[(i, i)].sqrt();
        assert!(sd(4) >= like().model_sd && sd(0) >= like().model_sd && c.st.info[(4, 4)] > 0.0);
        assert!(c.x[2] == 0.0 && (c.cov[(2, 2)] - 0.01).abs() < 1e-12);
        // its stops (every one, though every other pulse is fitted) are the stem's echoes
        assert!(e.echoes.len() > 50);
        assert!(e.echoes.p_wood.iter().filter(|&&q| q > 0.9).count() as f64 > 0.9 * e.echoes.len() as f64);
        // a context cylinder on the stem owns its stops: none is left for the piece, whose passes alone are fitted
        // (or the fit refused)
        let ctx = [(a, b, 0.05)];
        match routed::exact_piece(&[&scan], &a0, fr, &seeds[..1], &ctx, f64::NEG_INFINITY, &like(), &PriorParams::default()).unwrap() {
            routed::Fitted::Accepted(f) => assert!(f.echoes.is_empty()),
            routed::Fitted::Refused => {}
        }
    }

    fn accepted(f: routed::Fitted) -> routed::Exact {
        match f {
            routed::Fitted::Accepted(e) => e,
            routed::Fitted::Refused => panic!("refused"),
        }
    }

    fn refused(f: routed::Fitted) -> bool {
        match f {
            routed::Fitted::Accepted(e) => panic!("accepted with {} stems", e.stems.len()),
            routed::Fitted::Refused => true,
        }
    }

    #[test]
    fn a_fit_is_refused_without_seed_pulse_or_width() {
        // the fan's 5 cm stem, a piece claimed on it and seeded there
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let (a0, b0) = ([a[0], a[1], -0.15], [b[0], b[1], 0.15]);
        let fr = frames(&[a0], &[b0])[0];
        let seed = [routed::Seed { c: [0.0, 0.0], t: [0.0, 0.0], r: 0.05, t_var: [0.01, 0.01] }];
        let (p, pp) = (like(), PriorParams::default());
        let fit = |a0: &Point, fr: Frame, seeds: &[routed::Seed], lower: f64| routed::exact_piece(&[&scan], a0, fr, seeds, &[], lower, &p, &pp).unwrap();
        // no seed, no fit: never a fallback
        assert!(refused(fit(&a0, fr, &[], f64::NEG_INFINITY)));
        // one stem narrower than the piece's detection bound (8 cm): refused; under it (4 cm), accepted
        assert!(refused(fit(&a0, fr, &seed, 0.08f64.ln())));
        assert_eq!(accepted(fit(&a0, fr, &seed, 0.04f64.ln())).stems.len(), 1);
        // a piece above what the fan sees: no pulse
        let (a1, b1) = ([a[0], a[1], 0.7], [b[0], b[1], 1.0]);
        assert!(refused(fit(&a1, frames(&[a1], &[b1])[0], &seed, f64::NEG_INFINITY)));
        // where no run is, the seeds are none (and the piece is refused, above), not the strongest candidate
        let gs = groups(&[&scan], &a1, &b1, frames(&[a1], &[b1])[0], window(0.08) + LEAN_TUBE * 0.3, &p);
        assert!(routed::seeds(&gs, 0.08, &p, pp.lean_sd).is_empty(), "no run, no seed");
    }

    #[test]
    fn the_stem_that_continues_a_piece_is_near_and_informed_and_further_stems_chain() {
        // two upright rows of one piece each, the second continuing the first; both routed
        let rows_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]];
        let rows_b = [[0.0, 0.0, 0.3], [0.0, 0.0, 0.6]];
        let rows = Rows { a: &rows_a, b: &rows_b, r_prior: &[0.05, 0.05], r_max: &[], parent: &[-1, 0] };
        let fr = rows.check("test").unwrap();
        let mut nd = Nodes::of(&rows, &fr);
        let routed = Outcome::Routed { runs: true, hull: false };
        let mut sums = vec![summary(routed, Vec::new()), summary(routed, Vec::new())];
        let mut ends: Vec<(Point, f64)> = vec![(rows_b[0], 0.05), (rows_b[1], 0.05)];
        let mut further = vec![Vec::new(); 2];
        let stem = |c1: f64, r: f64, sd: f64| circle(Vector5::new(c1, 0.0, 0.0, 0.0, r), sd);
        // the base piece: a stem on the axis and one 15 cm beside it; at a base the better informed continues
        let node0 = enter(0, &exact(&[stem(0.15, 0.03, 0.01), stem(0.0, 0.05, 0.002)]), &mut nd, &mut sums, &mut ends, &mut further);
        assert_eq!(node0, vec![2, 0]);
        assert_eq!((nd.parent[2], nd.src(2)), (None, (0, 1)));
        // the next piece: a poorly informed stem nearest the parent's end, a well informed one 2 cm off it (both
        // within reach: the informed one continues), the side stem again (it continues the side stem below), and a
        // stem 8 cm off that ends near nothing (a node beside the piece)
        let node1 = enter(1, &exact(&[stem(0.0, 0.04, 0.03), stem(0.02, 0.045, 0.002), stem(0.15, 0.028, 0.01), stem(-0.08, 0.02, 0.01)]), &mut nd, &mut sums, &mut ends, &mut further);
        assert_eq!(node1[1], 1, "the informed stem continues the piece: {node1:?}");
        let (poor, side, apart) = (node1[0] as usize, node1[2] as usize, node1[3] as usize);
        assert_eq!(nd.parent[side], Some(2), "the side stem continues the base's side stem");
        assert!(nd.further[side].unwrap().share.is_none());
        // the poor stem and the one apart start nowhere near a further stem's end: beside the piece, under a fork term
        for k in [poor, apart] {
            assert_eq!(nd.parent[k], Some(0));
            assert!(nd.further[k].unwrap().share.is_some_and(|s| s > 0.0 && s < 1.0));
        }
        // each node is one (piece, stem)
        let mut keys: Vec<(usize, u32)> = (0..nd.row.len()).map(|k| nd.src(k)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), nd.row.len());
        assert_eq!(sums.len(), nd.row.len());
        assert!(ends.len() == nd.row.len() && (ends[1].1 - 0.045).abs() < 1e-12);
        // a far parent end: no stem within reach, so the nearest continues
        let mut ends2: Vec<(Point, f64)> = vec![([0.0, 0.0, 0.3], 0.05), ([0.0, 0.0, 0.6], 0.05)];
        ends2[0].0[0] = -0.5;
        let mut nd2 = Nodes::of(&rows, &fr);
        let mut sums2 = vec![summary(routed, Vec::new()), summary(routed, Vec::new())];
        let node = enter(1, &exact(&[stem(0.15, 0.03, 0.002), stem(-0.1, 0.03, 0.02)]), &mut nd2, &mut sums2, &mut ends2, &mut vec![Vec::new(); 2]);
        assert_eq!(node, vec![2, 1]);
    }

    #[test]
    fn a_leaning_stem_is_seeded_and_fitted_on_its_own_axis() {
        // the fan's 5 cm stem, upright; the piece claimed leaning 0.3 m per m across it, 4 cm off: a seed takes the
        // stem's lean from its runs, not the claim's, and the fit lies on the stem at both ends of the piece
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let lean = 0.3;
        let (a0, b0) = ([a[0] + 0.04 - 0.15 * lean, a[1], -0.15], [b[0] + 0.04 + 0.15 * lean, b[1], 0.15]);
        let fr = frames(&[a0], &[b0])[0];
        let (u, e1, e2, len) = fr;
        let p = like();
        let gs = groups(&[&scan], &a0, &b0, fr, window(0.08) + LEAN_TUBE * len, &p);
        let seeds = routed::seeds(&gs, 0.08, &p, PriorParams::default().lean_sd);
        assert!(!seeds.is_empty());
        let s = seeds[0];
        // the stem in the piece's frame, at its ends: where the true axis crosses the planes s = -len/2 and len/2
        let truth = |sa: f64| -> [f64; 2] {
            // the true axis is vertical through (a[0], a[1]): where it crosses the plane across the piece at sa
            let m: Point = std::array::from_fn(|i| a0[i] + (0.5 * len + sa) * u[i]);
            let z = m[2] - ((a[0] - m[0]) * u[0] + (a[1] - m[1]) * u[1]) / u[2];
            let w = sub(&[a[0], a[1], z], &m);
            [dot(&w, &e1), dot(&w, &e2)]
        };
        let (lo, hi) = (truth(-0.5 * len), truth(0.5 * len));
        let at = |c: [f64; 2], t: [f64; 2], sa: f64| [c[0] + sa * t[0], c[1] + sa * t[1]];
        for (sa, want) in [(-0.5 * len, lo), (0.5 * len, hi)] {
            let got = at(s.c, s.t, sa);
            assert!((got[0] - want[0]).hypot(got[1] - want[1]) < 0.01, "seed at {sa}: {got:?} against {want:?} ({s:?})");
        }
        let e = accepted(routed::exact_piece(&[&scan], &a0, fr, &seeds, &[], f64::NEG_INFINITY, &p, &PriorParams::default()).unwrap());
        let c = &e.stems[0];
        assert!((c.x[4] - 0.05).abs() < 0.004, "{:?}", c.x);
        for (sa, want) in [(-0.5 * len, lo), (0.5 * len, hi)] {
            let got = at([c.x[0], c.x[1]], [c.x[2], c.x[3]], sa);
            assert!((got[0] - want[0]).hypot(got[1] - want[1]) < 0.006, "fit at {sa}: {got:?} against {want:?}");
        }
    }

    #[test]
    fn the_echoes_are_keyed_by_their_file_row() {
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let n = scan.n_pulses();
        let mut rows: Vec<u32> = (0..n as u32).map(|k| scan.file_row(k)).collect();
        rows.sort_unstable();
        assert!(rows.iter().enumerate().all(|(i, &r)| r as usize == i), "the file rows are the pulses, once each");
        let rows = Rows { a: &[[a[0], a[1], -0.24]], b: &[[b[0], b[1], 0.24]], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let t = support_tree(&[&scan], &rows, &[], &like(), &PriorParams::default(), 2).unwrap();
        let e = &t.echoes;
        assert!(e.len() > 100);
        // one record per echo, sorted; on the stem's front arc, bark-free echoes are wood
        assert!((1..e.len()).all(|i| (e.scan[i - 1], e.pulse[i - 1], e.echo[i - 1]) < (e.scan[i], e.pulse[i], e.echo[i])));
        let on = (0..e.len()).filter(|&i| e.dist[i] < 0.01).count();
        assert!(on > 100 && (0..e.len()).filter(|&i| e.dist[i] < 0.01).all(|i| e.p_wood[i] == 1.0), "{on}");
    }

    #[test]
    fn stops_off_the_fitted_exact_path_are_classified_by_their_circles_arc() {
        // a 5 cm circle at the piece's centre: a stop on its front arc is wood by its echo (unknown reflectance:
        // fully), one well behind it went through and isn't, whatever became of the piece's summary, the porous
        // piece's too (its p_wood no longer scales it)
        let (scan, _, _) = crate::qsm::scan_index::tests::scene(0.05, true);
        let p = like();
        let rec = |b: f64, depth: f64, pulse: u32| Rec { b, s: 0.0, depth, xy: [f64::NAN; 2], w: 0.005, refl: f64::NAN, single: true, blocked: false, lost: false, pulse, echo: 0 };
        let stops = vec![Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs: vec![rec(0.0, -0.05, 0), rec(0.0, 0.10, 1)] }];
        let x = Vector5::new(0.0, 0.0, 0.0, 0.0, 0.05f64.ln());
        let circle = circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.05), 1e-3);
        for outcome in [Outcome::Bounded, Outcome::Porous, Outcome::Measured(circle), Outcome::Routed { runs: true, hull: false }] {
            let s = Summary { seen: stops.iter().map(|g| Group { recs: g.recs.clone(), ..*g }).collect(), log_odds: -3.0, ..summary(outcome, Vec::new()) };
            let mut e = Echoes::default();
            classify(&s, &x, 7, &[&scan], &p, &mut e);
            assert!(e.len() == 2 && e.node.iter().all(|&k| k == 7), "{e:?}");
            assert!(e.p_wood[0] == 1.0 && e.p_wood[1] == 0.0 && e.dist[0] < 1e-9 && (e.dist[1] - 0.05).abs() < 1e-9, "{e:?}");
            assert_eq!(e.pulse, vec![scan.file_row(0), scan.file_row(1)]);
        }
    }

    #[test]
    fn inputs_are_checked() {
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let bad = Rows { a: &[a], b: &[b], r_prior: &[0.08], r_max: &[], parent: &[0] };
        assert!(support_tree(&[&scan], &bad, &[], &like(), &PriorParams::default(), 1).is_err());
        let flat = Rows { a: &[a], b: &[a], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        assert!(support_tree(&[&scan], &flat, &[], &like(), &PriorParams::default(), 1).is_err());
    }

    #[test]
    fn the_sites_cap_holds_each_unmeasured_node_alone_after_the_solve() {
        // a 1.2 m stem (four pieces of 30 cm) forking into two 0.6 m rows (two pieces each), RCT's taper claiming 10, 6
        // and 5 cm; one child's first piece measured at 3 cm, nothing else seen
        let ra = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.0, 1.2]];
        let rb = [[0.0, 0.0, 1.2], [0.0, 0.36, 1.68], [0.0, -0.36, 1.68]];
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.1, 0.06, 0.05], r_max: &[], parent: &[-1, 0, 0] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let n = nd.row.len();
        // the reach to the tips: the stem's first piece 1.2 + 0.6 m, each later one 30 cm less; a child's first 0.6 m
        let l = reach(&rows, &fr, &nd);
        assert!([1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0.6, 0.3].iter().zip(&l).all(|(a, b)| (a - b).abs() < 1e-9), "{l:?}");
        let mut sums: Vec<Summary> = (0..n).map(|_| summary(Outcome::Bounded, Vec::new())).collect();
        sums[4] = summary(Outcome::Measured(circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.03), 1e-3)), Vec::new());
        let (mut sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &PriorParams::default(), EP_ROUNDS).unwrap();
        assert!(converged);
        let before: Vec<f64> = sol.x.iter().map(|x| x[4]).collect();
        let caps: Vec<f64> = l.iter().map(|l| (0.5 * 0.035 * l).ln()).collect();
        let free: Vec<bool> = (0..n).map(|k| k != 4).collect();
        let (mut held, mut bits) = (vec![(f64::NEG_INFINITY, f64::INFINITY); n], vec![0u16; n]);
        cap(&caps, &free, &mut sol, &mut held, &mut bits);
        for k in (0..n).filter(|&k| free[k]) {
            // each node alone: clipped at its own cap where it reached it, else as the solve left it (the tips' caps
            // don't carry down the stem), its draws held under it, and CAPPED where it was clipped
            assert!((sol.x[k][4] - before[k].min(caps[k])).abs() < 1e-12 && held[k].1 == caps[k], "piece {k}");
            assert_eq!(bits[k], if before[k] >= caps[k] { diag::CAPPED } else { 0 }, "piece {k}");
        }
        assert!(bits[5] == diag::CAPPED && bits[7] == diag::CAPPED);
        // the measured piece keeps its circle: the cap holds only what the rays don't measure
        assert!(sol.x[4][4] == before[4] && bits[4] == 0 && held[4] == (f64::NEG_INFINITY, f64::INFINITY));
        // a cap below what the rays hold a node above gives way to them: it is served at its lower bound, not capped
        let lo = caps[5] + 0.5;
        let mut held = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        held[5] = (lo, f64::INFINITY);
        sol.x[5][4] = lo + 1.0;
        let mut bits = vec![0u16; n];
        cap(&caps, &free, &mut sol, &mut held, &mut bits);
        assert!(sol.x[5][4] == lo && held[5] == (lo, lo) && bits[5] == 0, "{} {:?} {}", sol.x[5][4], held[5], bits[5]);
    }

    /// Stops on an arc of a circle of radius `r` about `c`, from `a0` to `a1` degrees, `n` of them, `jitter` (m) off it
    /// by a fixed pattern.
    fn arc_stops(c: [f64; 2], r: f64, (a0, a1): (f64, f64), n: usize, jitter: f64) -> Vec<[f64; 2]> {
        (0..n).map(|i| {
            let t = (a0 + (a1 - a0) * i as f64 / (n - 1) as f64).to_radians();
            let rr = r + jitter * ((i * 7 % 5) as f64 - 2.0) / 2.0;
            [c[0] + rr * t.cos(), c[1] + rr * t.sin()]
        }).collect()
    }

    #[test]
    fn a_rows_stops_give_its_point_circle_on_the_refits_tests() {
        // a 5 cm stem 1 cm off the axis, seen over 200 degrees with 2 mm of roughness, and a few leaves about it
        let mut xy = arc_stops([0.01, 0.0], 0.05, (-100.0, 100.0), 60, 0.002);
        xy.extend([[0.12, 0.05], [-0.1, 0.11], [0.0, -0.13], [0.09, -0.1]]);
        let x = point_circle(&xy, 0.06, f64::NAN).unwrap();
        assert!((x[4] - 0.05).abs() < 0.002 && (x[0] - 0.01).abs() < 0.002 && x[1].abs() < 0.002 && x[2] == 0.0 && x[3] == 0.0, "{x:?}");
        // the site's cap, through `believe`: a circle wider than it is none
        assert!(point_circle(&xy, 0.06, 0.04).is_none());
        // a claim under half its width: a circle over twice the claim is a bush, not a stem
        assert!(point_circle(&xy, 0.024, f64::NAN).is_none());
        // too few stops, or an arc of 90 degrees
        assert!(point_circle(&xy[..11], 0.06, f64::NAN).is_none());
        assert!(point_circle(&arc_stops([0.0, 0.0], 0.05, (0.0, 90.0), 40, 0.001), 0.06, f64::NAN).is_none());
        // three thin stems about a vase: no stem through them all
        let vase: Vec<[f64; 2]> = (0..3).flat_map(|k| arc_stops(std::array::from_fn(|i| 0.06 * (k as f64 * 2.1 + i as f64 * 1.57).cos()), 0.008, (0.0, 360.0), 15, 0.0)).collect();
        assert!(point_circle(&vase, 0.08, f64::NAN).is_none_or(|x| x[4] < 0.02), "{:?}", point_circle(&vase, 0.08, f64::NAN));
        // most stops off any circle: leaves
        let mut leafy = arc_stops([0.0, 0.0], 0.03, (-90.0, 90.0), 20, 0.0);
        leafy.extend((0..60).map(|i| [0.3 * ((i * 37 % 61) as f64 / 61.0 - 0.5), 0.3 * ((i * 53 % 59) as f64 / 59.0 - 0.5)]));
        assert!(point_circle(&leafy, 0.05, f64::NAN).is_none());
    }

    #[test]
    fn a_circle_at_the_tubes_edge_is_not_a_measurement() {
        // the fan's 5 cm stem, claimed 1 cm off it as itself (the circle well inside its 15.5 cm tube) and as a 5 mm
        // twig 10 cm long (the stem reaches past its 6.5 cm tube: a twig captured by a neighbouring mass, research 106)
        let (scan, _, _) = crate::qsm::scan_index::tests::scene(0.05, true);
        let piece = |r: f64, len: f64| {
            let rows = Rows { a: &[[0.01, 5.0, -0.5 * len]], b: &[[0.01, 5.0, 0.5 * len]], r_prior: &[r], r_max: &[], parent: &[-1] };
            support_tree(&[&scan], &rows, &[], &like(), &PriorParams::default(), 2).unwrap()
        };
        let near = piece(0.05, 0.3);
        assert!(near.pieces.flags[0] == 0 && near.pieces.diag[0] & diag::TUBE_EDGE == 0 && (near.pieces.r[0] - 0.05).abs() < 0.003, "{:?}", near.pieces);
        let far = piece(0.005, 0.1);
        let q = &far.pieces;
        assert!(q.flags[0] == flag::BOUNDS && q.diag[0] & diag::TUBE_EDGE != 0 && q.basis[0] != basis::MEASURED,
            "flags {} diag {} basis {}", q.flags[0], q.diag[0], q.basis[0]);
        // nor is the captor's span a lower bound on it
        assert!(q.r_lo[0].is_nan() && q.r[0] < 0.02, "r_lo {} r {}", q.r_lo[0], q.r[0]);
        // nor does the row's point circle re-fit the captor: a 2.5 cm claim 4.8 cm off the stem, whose own circle and
        // point circle (inside the refit's shift and width tests) both reach its 10.5 cm tube's edge
        let rows = Rows { a: &[[0.048, 5.0, -0.05]], b: &[[0.048, 5.0, 0.05]], r_prior: &[0.025], r_max: &[], parent: &[-1] };
        let t = support_tree(&[&scan], &rows, &[], &like(), &PriorParams::default(), 2).unwrap();
        let q = &t.pieces;
        assert!(q.flags[0] == flag::BOUNDS && q.diag[0] & diag::TUBE_EDGE != 0 && q.basis[0] != basis::POINTS,
            "flags {} diag {} basis {}", q.flags[0], q.diag[0], q.basis[0]);
        assert!(reaches_edge(&Vector5::new(0.07, 0.0, 0.0, 0.0, 0.05), 0.15, 0.115) && !reaches_edge(&Vector5::new(0.01, 0.0, 0.0, 0.0, 0.05), 0.15, 0.115));
        // a lean carries a circle to the edge at its ends
        assert!(reaches_edge(&Vector5::new(0.0, 0.0, 0.3, 0.0, 0.05), 0.15, 0.1));
    }

    #[test]
    fn a_rows_point_circle_takes_only_the_stops_nearest_its_own_axis() {
        // two upright rows 8 cm apart; stops ring a 3 cm stem about the second's axis, inside the first's tube too:
        // the first row's point circle takes none of them, nor those nearer a context cylinder
        let ra = [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]];
        let rb = [[0.0, 0.0, 0.3], [0.08, 0.0, 0.3]];
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.03, 0.03], r_max: &[], parent: &[-1, -1] };
        let fr = frames(&ra, &rb);
        let nd = Nodes::of(&rows, &fr);
        let (_, e1, e2, _) = fr[0];
        let stop = |x: f64, y: f64| {
            // a stop at world (x, y) and mid-height, as the first row's piece sees it
            let xy = [x * e1[0] + y * e1[1], x * e2[0] + y * e2[1]];
            Rec { b: 0.0, s: 0.0, depth: 0.0, xy, w: 0.005, refl: f64::NAN, single: true, blocked: false, lost: false, pulse: 0, echo: 0 }
        };
        let ring = |cx: f64, n: usize| (0..n).map(move |i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            (cx + 0.03 * t.cos(), 0.03 * t.sin())
        });
        let recs: Vec<Rec> = ring(0.08, 24).chain(ring(0.0, 12)).map(|(x, y)| stop(x, y)).collect();
        let mut sums = vec![summary(Outcome::Bounded, Vec::new()), summary(Outcome::Bounded, Vec::new())];
        sums[0].seen = vec![Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs }];
        assert_eq!(own_stops(0, &rows, &[], &nd, &sums).len(), 12);
        // a context stem 3.1 cm beside the first row's axis takes the quarter of its ring nearer its own
        let ctx = [([-0.031, 0.0, 0.0], [-0.031, 0.0, 0.3], 0.03)];
        assert_eq!(own_stops(0, &rows, &ctx, &nd, &sums).len(), 9);
    }

    #[test]
    fn a_point_circle_measures_its_pieces_with_their_bounds_beside_it() {
        // a row of two pieces under a measured base: the point circle (5 cm) enters each piece at PTS_SD times the
        // root of two in log R, and the pieces' upper bounds still enter beside it
        let rows_a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]];
        let rows_b = [[0.0, 0.0, 0.3], [0.0, 0.0, 0.9]];
        let rows = Rows { a: &rows_a, b: &rows_b, r_prior: &[0.1, 0.09], r_max: &[], parent: &[-1, 0] };
        let fr = frames(&rows_a, &rows_b);
        let nd = Nodes::of(&rows, &fr);
        let x = Vector5::new(0.003, 0.0, 0.0, 0.0, 0.05);
        let c = point_measure(&x, 2.0);
        assert!((c.cov[(4, 4)].sqrt() / 0.05 - PTS_SD * 2f64.sqrt()).abs() < 1e-12);
        let base = circle(Vector5::new(0.0, 0.0, 0.0, 0.0, 0.09), 1e-3);
        let upper = Bound { eta: 0.08f64.ln(), sd: 0.1, side: Side::Upper };
        let mut sums = vec![summary(Outcome::Measured(base), Vec::new()), summary(Outcome::Bounded, vec![upper]), summary(Outcome::Bounded, vec![upper])];
        let pp = PriorParams::default();
        let (prior_only, _) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        for s in &mut sums[1..] {
            s.points = Some(c);
        }
        assert!(sums[1].bounded() && sums[1].used().is_some() && sums[1].own().is_none() && !sums[0].bounded());
        assert_eq!(terms(&sums).iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 2]);
        let (sol, converged) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        assert!(converged);
        // the pieces move from the taper's guess towards the circle, and are surer of it
        for k in 1..3 {
            assert!((sol.x[k][4] - 0.05f64.ln()).abs() < (prior_only.x[k][4] - 0.05f64.ln()).abs() && sol.cov[k][(4, 4)] < prior_only.cov[k][(4, 4)],
                "{k}: {} against {}", sol.x[k][4].exp(), prior_only.x[k][4].exp());
        }
    }

    #[test]
    fn passes_through_a_circle_bound_it_and_audit_it() {
        let w = 0.005;
        let rec = |b: f64, s: f64, depth: f64| Rec { b, s, depth, xy: [f64::NAN; 2], w, refl: f64::NAN, single: true, blocked: false, lost: false, pulse: 0, echo: 0 };
        let group = |recs: Vec<Rec>| Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs };
        let circle = |r: f64| Vector5::new(0.0, 0.0, 0.0, 0.0, r);
        // a phantom: RCT claims 5 cm where a grid of pulses 5 mm apart passes, but for a 2 mm twig on the axis that
        // stops the middle column (research 106: trunks with 47-84 passes through RCT's circle, and no bound)
        let mut v = Vec::new();
        for j in 0..20 {
            for i in -12..=12 {
                let b = i as f64 * 0.005;
                v.push(rec(b, j as f64 * 0.005, if i == 0 { -0.002 } else { f64::NAN }));
            }
        }
        let phantom = [group(v)];
        let (stop, pass) = tally(&phantom, &circle(0.05));
        assert!(refuted((stop, pass)) && stop == 20 && pass == 360, "{stop} {pass}");
        assert!(!refuted(tally(&phantom, &circle(0.004))));
        let b = veto(&phantom, 0.1).unwrap();
        assert!(b.side == Side::Upper && b.eta.exp() >= 0.002 && b.eta.exp() < 0.006, "{}", b.eta.exp());
        // a posterior circle the veto refutes is narrowed, about its own axis, until it isn't
        let eta = veto_at(&phantom, &Vector5::new(0.0, 0.0, 0.0, 0.0, 0.05f64.ln())).unwrap();
        assert!(eta.exp() < 0.006 && !refuted(tally(&phantom, &circle(eta.exp()))), "{}", eta.exp());
        // a solid 2 cm stem, hit across its silhouette and passed beside it: the veto leaves it its radius and a
        // pulse spacing, and its own circle reads no passes
        let mut v = Vec::new();
        for j in 0..20 {
            for i in -12..=12 {
                let b = i as f64 * 0.005;
                v.push(rec(b, j as f64 * 0.005, if b.abs() < 0.02 { -(0.0004 - b * b).sqrt() } else { f64::NAN }));
            }
        }
        let stem = [group(v)];
        assert!(tally(&stem, &circle(0.02)).1 == 0 && veto_at(&stem, &Vector5::new(0.0, 0.0, 0.0, 0.0, 0.02f64.ln())).is_none());
        // the same stem 3 cm beside the claim's axis (research 122 F14): a 2 cm circle about the axis is refuted and
        // narrowing it about the axis leaves a thread, but moved onto the stem it keeps its 2 cm, and nothing moves
        // where no circle that wide is left unrefuted (the phantom)
        let beside = [group(stem[0].recs.iter().map(|r| Rec { b: r.b + 0.03, ..*r }).collect())];
        let lo = 0.02f64.ln();
        let x = Vector5::new(0.0, 0.0, 0.0, 0.0, lo);
        assert!(veto_at(&beside, &x).unwrap() < 0.005f64.ln());
        let (c1, c2) = recentre(&beside, &x, lo, 0.1, |_, _| true).unwrap();
        assert!((c1 - 0.03).abs() < 0.006 && c2.abs() < 0.011, "{c1} {c2}");
        assert!(veto_at(&beside, &Vector5::new(c1, c2, 0.0, 0.0, lo)).is_none());
        assert!(recentre(&phantom, &Vector5::new(0.0, 0.0, 0.0, 0.0, lo), lo, 0.1, |_, _| true).is_none());
        // where that stem is a neighbour's (its axis at 3 cm, nearer it than the claim's past 1.5 cm), it isn't taken
        assert!(recentre(&beside, &x, lo, 0.1, |c1, _| c1 < 0.015).is_none());
        let b = veto(&stem, 0.1).unwrap();
        assert!(b.eta.exp() >= 0.02 && b.eta.exp() < 0.03, "{}", b.eta.exp());
        // a pulse blocked in front, or stopping before the circle, says nothing of it
        let front = [group((0..20).map(|j| rec(0.0, j as f64 * 0.005, -0.05)).collect())];
        assert_eq!(tally(&front, &circle(0.02)), (0, 0));
        assert!(veto(&front, 0.1).is_none());
    }

    #[test]
    fn a_stem_off_the_claims_axis_in_depth_is_not_bounded_by_its_own_flanks() {
        // a 2.25 cm stem 4.5 cm behind a 5 mm claim's axis: its flanks fall behind the claim's depth window, and read
        // as passes either side of a 2 cm run they bounded it at 1.5 cm, pinning a point circle on it or a thin
        // claim in a fat base (3dtrees research 127); a pass is a pulse that met nothing in the tube
        let rec = |b: f64, s: f64, depth: f64| Rec { b, s, depth, xy: [f64::NAN; 2], w: 0.005, refl: f64::NAN, single: true, blocked: false, lost: false, pulse: 0, echo: 0 };
        let mut v = Vec::new();
        for j in 0..20 {
            for i in -12..=12 {
                let b = i as f64 * 0.005;
                v.push(rec(b, j as f64 * 0.005, if b.abs() < 0.0225 { 0.045 - (0.0225f64.powi(2) - b * b).sqrt() } else { f64::NAN }));
            }
        }
        let gs = [Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs: v }];
        let p = like();
        let upper = |bs: Vec<Bound>| bs.iter().filter(|b| b.side == Side::Upper).map(|b| b.eta.exp()).fold(f64::INFINITY, f64::min);
        assert!(upper(Profile::of(&gs[0], 0.005, &p).bounds(0.005, &p, None)) >= 0.0225);
        // the same stem on the claim's axis is bounded by the passes beside it
        let on: Vec<Rec> = gs[0].recs.iter().map(|r| Rec { depth: r.depth - 0.045, ..*r }).collect();
        let hi = upper(Profile::of(&Group { scan: 0, n: [1.0, 0.0], v: [0.0, 1.0], recs: on }, 0.02, &p).bounds(0.02, &p, None));
        assert!((0.0225..0.03).contains(&hi), "{hi}");
    }

    #[test]
    fn a_posterior_is_held_inside_its_own_bounds() {
        // a base piece claimed at 2 cm whose passes say at most 3 mm, softly (a footprint wide against a thin piece:
        // research 106 found bounds of 0.1-0.5 cm under posteriors of 1.3-2.3 cm)
        let rows = Rows { a: &[[0.0, 0.0, 0.0]], b: &[[0.0, 0.0, 0.2]], r_prior: &[0.02], r_max: &[], parent: &[-1] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let up = Bound { eta: 0.003f64.ln(), sd: 1.0, side: Side::Upper };
        let sums = [summary(Outcome::Bounded, vec![up])];
        let (mut sol, _) = tree_solve(&rows, &fr, &nd, &sums, &PriorParams::default(), EP_ROUNDS).unwrap();
        assert!(sol.x[0][4] > up.eta, "the soft bound alone leaves it outside: {}", sol.x[0][4].exp());
        let (held, bits) = hold(&terms(&sums), 1, &mut sol);
        assert!(sol.x[0][4] <= up.eta && held[0].1 == up.eta && bits[0] == diag::PINNED, "{} {:?} {}", sol.x[0][4].exp(), held[0], bits[0]);
        // bounds that cross: the upper wins (wood the rays pass through isn't served)
        let lo = Bound { eta: 0.005f64.ln(), sd: 0.1, side: Side::Lower };
        let sums = [summary(Outcome::Bounded, vec![up, lo])];
        let (mut sol, _) = tree_solve(&rows, &fr, &nd, &sums, &PriorParams::default(), EP_ROUNDS).unwrap();
        let (held, _) = hold(&terms(&sums), 1, &mut sol);
        assert!(held[0] == (up.eta, up.eta) && sol.x[0][4] == up.eta);
        // a posterior inside its bounds, with the prior there, isn't moved or pinned
        let wide = [summary(Outcome::Bounded, vec![Bound { eta: 0.05f64.ln(), sd: 0.1, side: Side::Upper }, Bound { eta: 0.01f64.ln(), sd: 0.1, side: Side::Lower }])];
        let (mut sol, _) = tree_solve(&rows, &fr, &nd, &wide, &PriorParams::default(), EP_ROUNDS).unwrap();
        let x = sol.x[0][4];
        let (_, bits) = hold(&terms(&wide), 1, &mut sol);
        assert!(sol.x[0][4] == x && bits[0] == 0);
    }

    #[test]
    fn a_trees_volume_is_drawn_from_its_joint_posterior() {
        // a 0.9 m row of three pieces and a 0.3 m child, none measured: their log R a chain, strongly correlated
        let ra = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]];
        let rb = [[0.0, 0.0, 0.9], [0.0, 0.1, 1.18]];
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.05, 0.03], r_max: &[], parent: &[-1, 0] };
        let fr = rows.check("test").unwrap();
        let nd = Nodes::of(&rows, &fr);
        let n = nd.row.len();
        let sums: Vec<Summary> = (0..n).map(|_| summary(Outcome::Bounded, Vec::new())).collect();
        let pp = PriorParams::default();
        let (sol, _) = tree_solve(&rows, &fr, &nd, &sums, &pp, EP_ROUNDS).unwrap();
        let free = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        // the first row's pieces prior, the child bounded: each widened by its own basis's factor
        let basis: Vec<u8> = (0..n).map(|k| if nd.row[k] == 0 { basis::PRIOR } else { basis::BOUNDED }).collect();
        // the same distribution by brute force: draws from the dense covariance of the same system
        let (h, c, g) = prior_system(&rows, &nd, &pp, &fork_means(&rows, &fr, &nd, &pp), &sol.fork_w);
        let mut hd = nalgebra::DMatrix::<f64>::zeros(5 * n, 5 * n);
        for k in 0..n {
            hd.view_mut((5 * k, 5 * k), (5, 5)).copy_from(&h[k]);
            if let Some(q) = nd.parent[k] {
                hd.view_mut((5 * k, 5 * q), (5, 5)).copy_from(&c[k]);
                hd.view_mut((5 * q, 5 * k), (5, 5)).copy_from(&c[k].transpose());
            }
        }
        let _ = g;
        let chol = hd.try_inverse().unwrap().cholesky().unwrap().l();
        // each piece widened by its basis's factor
        {
            let v = volume(&nd, &sol, &free, &basis);
            let mut rng = Normal::new(7);
            let m = 40_000;
            let mut draws: Vec<f64> = (0..m).map(|_| {
                let z = nalgebra::DVector::<f64>::from_fn(5 * n, |_, _| rng.next());
                let d = &chol * z;
                (0..n).map(|k| std::f64::consts::PI * (2.0 * (sol.x[k][4] + widen(basis[k]) * d[5 * k + 4])).exp() * nd.frame[k].3).sum::<f64>()
            }).collect();
            draws.sort_unstable_by(f64::total_cmp);
            let (q05, q95) = (draws[m / 20], draws[19 * m / 20]);
            let mean = draws.iter().sum::<f64>() / m as f64;
            // within three standard errors of a percentile of VOLUME_DRAWS draws (in log V, its sd from its spread)
            let se = (0.05f64 * 0.95 / VOLUME_DRAWS as f64).sqrt() / npdf(1.645) * (q95 / q05).ln() / 3.29;
            assert!((v.q05 / q05).ln().abs() < 3.0 * se && (v.q95 / q95).ln().abs() < 3.0 * se && (v.mean / mean).ln().abs() < 3.0 * se,
                "{v:?} against q05 {q05} q95 {q95} mean {mean} (se {se})");
            // the point is the posterior's; the mean is split by basis (the first row prior, the child bounded)
            let point: f64 = (0..n).map(|k| std::f64::consts::PI * (2.0 * sol.x[k][4]).exp() * nd.frame[k].3).sum();
            let (prior, bounded) = (v.by_basis[usize::from(basis::PRIOR) - 1], v.by_basis[usize::from(basis::BOUNDED) - 1]);
            assert!((v.point - point).abs() < 1e-12 && (prior + bounded - v.mean).abs() < 1e-9 * v.mean && prior > 0.0 && bounded > 0.0 && v.q05 < v.point && v.point < v.q95);
        }
        // a piece held in its interval holds every draw: none wider than its upper bound
        let mut held = free.clone();
        held[0] = (f64::NEG_INFINITY, sol.x[0][4]);
        let v = volume(&nd, &sol, &held, &basis);
        let w = volume(&nd, &sol, &free, &basis);
        assert!(v.q95 < w.q95 && v.mean < w.mean);
    }
}
