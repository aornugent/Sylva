// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of elements scored against scans' range images by the ray likelihood (3dtrees research 130): its cost,
//! `−log` of its likelihood ratio against the empty scene, the cost change of a move, what an element predicts in an
//! image, and how well the scans pin an element's geometry down.
//!
//! - **Elements** ([`Element`]): a form, a stick (an axis, its section's mean radius a₀) or a patch (a ball with its
//!   extent, a disc in every view), and a class, wood or leaf, each with an opacity and a shape mismatch. The form is
//!   geometry only; the class sets the element's reflectance ([`Settings`]'s `refl_wood`, `refl_leaf`), so the same
//!   stick marked wood and marked leaf shares its cells and nuisances (3dtrees research 130 slice 5a). Class evidence
//!   from photometry is the consumer's, read off [`Scene::predict`]'s cells beside the cost, never in it (slice 9).
//!   Opacity is the chance a pulse the section fills meets solid there, its gaps wider than the footprint (synthesis
//!   §1's "extent, opacity"). The mismatch is the sd of the section's and axis's
//!   departure from the form, which blurs the fill and spreads the depth.
//! - **The k-buffer:** each element renders into each view's cells. A cell keeps its layers, nearest first, up to the
//!   [`K`]th wholly in it ([`keep`]), each with a depth, a depth spread, a fill and a cosine. Fill is the beam (a uniform disc of the
//!   footprint's diameter, as AM's raw edges measure it) convolved with the section, at three section radii over
//!   the mismatch; an echo's chance is the detection's mean over them. Along a stick's axis it is cut square, so a
//!   chain of sticks joins without seams.
//! - **The ray likelihood, one cell:** its pulse whole (3dtrees research 130 slice 2c), against occluders (the
//!   layers in depth order, then a ground plane) and a medium (the foliage field; κ, the background, outside its
//!   box): after any return, the next surface is the medium's ([`Pass::lik`]). A pulse at the target cap says
//!   nothing past its last echo.
//!   - **One pass per pulse** ([`Pulse::read`]; 3dtrees research 130 formulation.md, slice-pulse.md): the cell's
//!     owners (its layers, the ground, the medium) and the receiver ([`Receiver`]: merges, detection, the cap) are read
//!     once, and that pass gives everything the scene reads of the pulse: its log-likelihood with its slope and
//!     curvature in the own nodes ([`Dual2`]; with the geometry, its slope in [`Scene::information`]: [`Dual`]), the element's echo ([`front_echo`]), each owner's beam left and
//!     expected echoes, and each echo's responsibilities over the owners. The cost keeps the hard attribution (a
//!     cell's evidence is its front element's term's, the field's cut); the responsibilities are reported, for the
//!     soft attribution to come.
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is recorded with
//!     `op_i det_i`, where `det_i` is the chance it clears the detection floor ([`Settings::p_detect`]):
//!     `Φ((ρ_dB + 10 log10(e_i cos ι_i) − floor(t_i)) / σ)`.
//!   - Adjacent returns (layers and the ground) resolve into two echoes with the instrument's
//!     [`Settings::p_resolve`] of their separation and energy ratio; else they merge into one, recorded if any
//!     member is, pulled back from the nearer's depth by the measured pull ([`Settings::pull`]; energy: op · e · cos
//!     · reflectance). A cell's likelihood is the mixture over these merge patterns.
//!   - The first echo at r has density `Σ_i P_i op_i det_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i)` over the
//!     returns, plus `σ(r) e^{−τ(r)} Π_{t_j<r} (1 − op_j det_j)`, where P_i is the chance that nothing nearer was kept,
//!     plus a uniform outlier share. Each later echo is read the same way behind the last, the medium's path
//!     restarting at it plus the resolution, in the beam the echoes so far left. A pulse with no echo has probability
//!     `e^{−τ_end} Π (1 − op_j det_j)`. Each `1 − op_j det_j` is computed as `op_j miss_j + 1 − op_j`, `miss_j` the
//!     detection's own tail ([`Settings::p_miss`]): `1 − det_j` cancels to 0 for a bright near-full hit, and a cliff
//!     of hundreds of nats follows (3dtrees research 130 rca-level-modes.md).
//!   - **The merge with the medium** ([`merges`]): a layer's return merges with a medium surface within the resolution
//!     behind its far side with the site's calibrated merge share, at unshifted ranges, built once per cell; inside
//!     the return's density it is a mixture, its merged echo pulled back and brighter ([`Return::density`]).
//!   - **The empty scene** every cell is a ratio against is the medium and the ground alone, the same for every
//!     hypothesis.
//!   - **Located edges** (3dtrees research 130 step 2): the echo's reflectance x (dB) is read beside its range. A
//!     return predicts `10 log10 Σ e_i cos^k ι_i 10^(ρ_i / 10)` over its members (one layer, or a merge with a
//!     surface behind), at the mismatch's three section radii, and x reads about it with the per-echo noise, wider at
//!     a partial fill, truncated by the detection floor ([`Settings::refl_density`], [`Return::refl_density`]); the hazard's and the outliers' echoes read the background's
//!     spread ([`Settings::refl_background`]). So a silhouette's partial cells place its edge within a cell. Each
//!     element's reflectance level (its offset from its surface's reflectance, per scan) is a latent beside its
//!     shift, so an unknown bark brightness is read as a level, not a width.
//!   - Every cell an element's layer enters counts, partial hits included, so every hypothesis is scored on the same
//!     echoes (slice 7c: a gate at fill 0.5, with the band below it scored by located edges, let a wider stick move
//!     echoes from the edges into the gate, where each is worth far more).
//!   - **The fill-weighted entry** (3dtrees research 130 slice-trim): a layer is in its cell with its entry weight
//!     ([`Layer::entry`]), a smooth step in its widest fill from 0 at [`MIN_FILL`] to 1 at [`FULL_FILL`]; a cell's
//!     likelihood is the mixture over which of its partial layers are in it ([`mixture`]), and its evidence is shared
//!     among its layers' terms by the chance each is the nearest ([`front_shares`]). So no fill, and no layer passing
//!     in front of another as it enters, steps the cost (a hard entry at 1 % did, by up to tenths of a nat on the
//!     stand); where the receiver cut (a merge with the medium behind under 1e-6 of the beam, a merge pattern under
//!     1e-4) it fades ([`smooth`]).
//! - **The element's echo** ([`front_echo`]): the first return of the most likely way a cell comes back, its sd
//!   widened by the alignment error and the stem front shift. [`Outcome::of`] against it is the one rule for "the
//!   cell's first echo is the element's", for reporting only: the cells_on, arcs and rows' parts, `predict` and
//!   `counts` (and, for any echo of the pulse, `predict`'s `read`); the cost never reads it. A cell mostly filled
//!   ([`MOSTLY`]) is where an echo reads the element's surface.
//! - **Depth over a footprint:** a layer's depth is the mean of the section's depth over the part of the footprint
//!   it meets, weighted by each part's energy (the cosine of incidence), as a scanner's mixed echo is
//!   ([`footprint_depth`]). The mismatch spreads the depth but doesn't move it (slice 7b).
//! - **The scan alignment error** (research 130 offset-field-design.md, slices 7 and 7b): per scan, an along-beam
//!   shift on nodes ℓ apart (trilinear hats, normalised) with a unit Gaussian prior, scaled at each element by its
//!   surface class's spread (and the other-epoch part for a scan off the reference epoch, which the caller
//!   declares). A cell's depths move by the shift at its front element. Neighbours sharing nodes share one shift,
//!   and one scan's joint evidence on a place is capped by the prior. **Surface classes** (3dtrees research 130
//!   slice-structure): every echo reads the depth bias and noise of its own cell's class ([`cell_class`]: the excess
//!   there, data); a term's class, which sets its along spread, is the median over the cells it mostly fills
//!   ([`term_classes`]). So no term's evidence depends on another's class. With it, the **stem
//!   front shift** (slice 14): a node of the element's own per scan, its section's departure from round as that view
//!   sees it. A scan's cost is the Laplace marginal of its data over both and the reflectance levels
//!   ([`solve_alignment`]); between refits the alignment is held ([`Held`], research 130 step 3a).
//! - **Weights:** each cell counts [`Settings::cell_weight`], `1 / (φ m(ρ))`, with m the footprints per cell
//!   (overlapping footprints, on the image's own lattice step) and φ the evidence discount per cell: one constant
//!   for every cell and every hypothesis, so it widens posteriors without favouring any hypothesis (a
//!   quasi-likelihood). It replaced slice 7's ρ_res, `1 + (n − 1) ρ` with n an element's own footprints, which
//!   differed between hypotheses and, once every fronted cell counted, favoured the thinner (slice 7c).
//!
//! - **Chains** (3dtrees research 130 slice 6): elements of one group ([`Element`]'s `group`) are one term per scan
//!   ([`Owner`]), so a chain of short sticks shares its nuisances. A declared term over
//!   its geometry alone (its coupling: taper, bends) is the consumer's, beside the cost, never in it.
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A move (add, remove,
//!   change) returns its cost change with the alignment held, recomputing only the terms of the elements whose cells
//!   it touches; `refit_alignment` re-solves the alignment. Its rows ([`ElementScan`]) split each (element, scan)'s cost into its parts, and
//!   [`Scene::information`] gives the slope and Fisher information of an element's cost in its geometry.
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use super::dual::{Dual, Dual2, Over, Real};
use nalgebra::{Matrix2, Vector2};
use super::settings::{Background, Settings, ALIGN_SPACING, P_OUTLIER, T_MAX};
use super::foliage::{self, FieldCounts, FixedCounts, Foliage, FoliageBox, Medium};
use super::range_image::{Cell, RangeImage};
use std::sync::Arc;
use crate::transform::{cross, norm, normalize, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Layers a cell's pulse reads, nearest first. A cell that runs out (rare among a few elements) drops its farthest.
pub const K: usize = 3;

/// The widest fill a layer's entry weight rises from 0 at ([`Layer::entry`]): a layer fainter than this is no part of
/// its cell.
pub const MIN_FILL: f64 = 0.002;

/// The widest fill from which a layer is wholly part of its cell (entry weight 1).
pub const FULL_FILL: f64 = 0.01;

/// The fill over which a cell is mostly the element's: its echo there reads the element's surface (its term's surface
/// class and its shape mismatch).
pub const MOSTLY: f64 = 0.5;

/// An element's form: its geometry.
#[derive(Debug, Clone, Copy)]
pub enum Form {
    /// An axis from `a` to `b` (world, m).
    Stick { a: Point, b: Point },
    /// A ball about `centre`: a leaf patch or clump, its section a disc in every view.
    Patch { centre: Point },
}

/// What an element is made of: it sets the element's reflectance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Wood,
    Leaf,
}

/// An element hypothesis: its form, its class, its section's radius (a stick's a₀; a patch's extent), its opacity
/// (1 solid), its shape mismatch sd (m) and its group. The caller supplies opacity and mismatch (each with a declared
/// default). Elements of one `group` (a chain: research 130 slice 6) are one term per scan ([`Owner`]); none, the
/// element is alone.
#[derive(Debug, Clone, Copy)]
pub struct Element {
    pub form: Form,
    pub class: Class,
    pub r: f64,
    pub opacity: f64,
    pub mismatch: f64,
    pub group: Option<u32>,
}

/// Whose term a cell's evidence enters, per scan: an element alone (by id), or a group (by its key). A group's members
/// share one set of nuisances (one alignment shift, at their middles' mean; one stem front shift, its prior sd their
/// mean; one reflectance level) and one surface class (the median over their cells: [`term_classes`]), so a chain of short sticks is
/// charged its nuisances once. Its members' cells are scored as one element's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Owner {
    Alone(u32),
    Group(u32),
}

/// Element `id`'s owner.
fn owner(id: u32, e: &Element) -> Owner {
    e.group.map_or(Owner::Alone(id), Owner::Group)
}

/// Each element's term among its owners: the place its owner first appears.
fn slots(owners: &[Owner]) -> Vec<usize> {
    let mut at: HashMap<Owner, usize> = HashMap::new();
    owners.iter().map(|o| {
        let n = at.len();
        *at.entry(*o).or_insert(n)
    }).collect()
}

/// The mean over some elements of a quantity (one element's is exactly its own).
fn mean_of(es: &[&Element], f: impl Fn(&Element) -> f64) -> f64 {
    es.iter().map(|e| f(e)).sum::<f64>() / es.len() as f64
}

/// A term's middle: its elements' middles' mean.
fn middle(es: &[&Element]) -> Point {
    [0, 1, 2].map(|k| mean_of(es, |e| e.mid()[k]))
}

/// A term's latent nodes ([`Owner`]: an element's, or a group's members' together): its alignment hats at its middle
/// ([`hats`]) and its stem front shift's prior sd (its members' mean). The cost ([`elem_data`]) and the information
/// ([`Shared`]) read them here, so a group's front shift has one prior in both.
fn term_nodes(es: &[&Element], em: &Settings) -> (Vec<([i64; 3], f64)>, f64) {
    (hats(&middle(es), ALIGN_SPACING), mean_of(es, |e| e.front_shift_sd(em)))
}

impl Element {
    fn mid(&self) -> Point {
        match self.form {
            Form::Stick { a, b } => [0, 1, 2].map(|k| 0.5 * (a[k] + b[k])),
            Form::Patch { centre } => centre,
        }
    }

    /// Its surface's reflectance (dB at full fill, face-on), by its class.
    fn refl(&self, em: &Settings) -> f64 {
        match self.class {
            Class::Wood => em.refl_wood,
            Class::Leaf => em.refl_leaf,
        }
    }

    /// Its stem front shift's prior sd (m): a stick's section departs from round as each view sees it; a patch's disc
    /// has no front to shift.
    fn front_shift_sd(&self, em: &Settings) -> f64 {
        match self.form {
            Form::Stick { .. } => em.front_shift_sd(self.r),
            Form::Patch { .. } => 0.0,
        }
    }

    /// Why the element can't be scored, if it can't: a radius that isn't positive, an opacity outside (0, 1], a
    /// negative mismatch, a point that isn't finite, or a stick of no length.
    pub fn invalid(&self) -> Option<&'static str> {
        let finite = |p: &Point| p.iter().all(|x| x.is_finite());
        if !(self.r > 0.0 && self.r.is_finite()) {
            return Some("its radius must be positive");
        }
        if !(self.opacity > 0.0 && self.opacity <= 1.0) {
            return Some("its opacity must be in (0, 1]");
        }
        if !(self.mismatch >= 0.0 && self.mismatch.is_finite()) {
            return Some("its mismatch must be finite and not negative");
        }
        match self.form {
            Form::Stick { a, b } if !(finite(&a) && finite(&b)) => Some("its ends must be finite"),
            Form::Stick { a, b } if !(norm(&sub(&b, &a)) > 0.0) => Some("a stick needs length"),
            Form::Patch { centre } if !finite(&centre) => Some("its centre must be finite"),
            _ => None,
        }
    }
}

/// What a cell's first echo says about an element: on it, none, in front of it, or behind it (the pulse passed
/// through).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Outcome {
    On = 0,
    Empty = 1,
    InFront = 2,
    Behind = 3,
}

impl Outcome {
    /// The outcome of a first echo `obs` (NaN for none) against the element's echo predicted at `depth` with sd `sd`
    /// ([`front_echo`]): on within `3 sd + 1 cm`, in front beyond `3 sd + 3 cm`, else behind.
    pub fn of(obs: f64, depth: f64, sd: f64) -> Outcome {
        if !obs.is_finite() {
            Outcome::Empty
        } else if obs < Outcome::in_front_of(depth, sd) {
            Outcome::InFront
        } else if (obs - depth).abs() < 3.0 * sd + 0.01 {
            Outcome::On
        } else {
            Outcome::Behind
        }
    }

    /// The range an echo is in front of the element's echo (at `depth`, sd `sd`) before: `3 sd + 3 cm` short of it.
    pub fn in_front_of(depth: f64, sd: f64) -> f64 {
        depth - 3.0 * sd - 0.03
    }
}

/// One element's layer in one cell: what the ray likelihood reads of it, its geometry's quantities in `P` (`f64`, or a
/// [`Dual`] carrying their slope in the element's parameters: [`Scene::information`]).
#[derive(Debug, Clone, Copy)]
struct Layer<P = f64> {
    t: P,
    /// The range of the section's far side where the beam meets it: where the medium behind it starts.
    far: P,
    spread: P,
    /// The fill: its mean over the mismatch (`f`), and at section radii r − √3 m, r and r + √3 m (`fq`, the
    /// mismatch's three-point Gauss–Hermite rule, weights 1/6, 2/3, 1/6).
    f: P,
    fq: [P; 3],
    cos: P,
    /// The surface's reflectance (dB at full fill, face-on), by the element's class.
    db: f64,
    elem: u32,
    op: f64,
    /// Where the cell's ray passes the axis: across it (signed, m, positive on `u × (c − o)`'s side) and along it;
    /// for a patch, the distance from the centre and 0.
    x: f64,
    s: f64,
}

impl<P: Real> Layer<P> {
    /// Its entry weight (3dtrees research 130 slice-trim): the chance it is part of its cell, a smooth step (C¹) in its
    /// widest fill from 0 at [`MIN_FILL`] to 1 at [`FULL_FILL`]. A cell's likelihood is the mixture over which of its
    /// partial layers are in it ([`mixture`]), so a layer's part in the cost goes to 0 with its fill, slope and all.
    fn entry(&self) -> P {
        smooth((self.fq[2] - MIN_FILL) / (FULL_FILL - MIN_FILL))
    }

    /// Whether it enters the k-buffer: an entry weight above 0.
    fn enters(&self) -> bool {
        self.fq[2].v() > MIN_FILL
    }

    /// Whether it is wholly part of its cell: an entry weight of 1.
    fn full(&self) -> bool {
        self.fq[2].v() >= FULL_FILL
    }
}

impl Layer {
    /// The layer in another number type, its quantities constants there.
    fn lift<P: Real>(&self) -> Layer<P> {
        let c = P::c;
        Layer { t: c(self.t), far: c(self.far), spread: c(self.spread), f: c(self.f), fq: self.fq.map(c), cos: c(self.cos), db: self.db, elem: self.elem,
                op: self.op, x: self.x, s: self.s }
    }

    /// Where the foliage field's fit stops reading a ray this layer fronts ([`foliage`]'s module docs): an echo before
    /// it is in front of the layer ([`Outcome::in_front_of`] its depth, at the noisiest surface class's sd), one at or
    /// after it is the layer's or behind it. A partial layer ([`Layer::full`] false) cuts nothing (∞): the field's cut
    /// is hard, so it stays where it was, at [`FULL_FILL`] (3dtrees research 130 slice-trim).
    fn cut(&self, em: &Settings) -> f64 {
        if self.full() { Outcome::in_front_of(self.t, em.depth_sd(2, self.spread)) } else { f64::INFINITY }
    }
}

/// What one scan says about one element, or one group ([`Owner`]: `group`, and `elem` its first member), its cost
/// split into parts that sum to `cost`. A group's row reads its members' cells together.
/// - `cells`: cells it has a front share in ([`front_shares`]); `cells_on`: those it is the nearest layer of whose first echo is
///   its ([`Outcome::On`]); `across`: footprints across its diameter; `surface`: its term's surface class in this
///   scan ([`term_classes`]).
/// - `arc`: the arc of a stick's outline its echoes span (rad, 0 to π; the support function's coverage, as the
///   candidates' arcs): `asin(x_max / r) − asin(x_min / r)` over its `cells_on` rays' places across its axis, so π
///   where the scan sees its whole near side, less where it is hidden or seen past the edge only.
/// - The ray likelihood's cost at the alignment's mode, by what each cell's first echo says about the element:
///   `ray_on` (its echoes), `ray_empty` (pulses with no echo), `ray_in_front` (hidden by something nearer) and
///   `ray_behind` (pulses that passed through it: evidence against).
/// - `alignment`: its own nodes' (stem front shift, reflectance level) prior and log-determinant terms. The scan's alignment part is the scan's, not shared out among its
///   elements ([`Scene::alignment_parts`]), so the rows, those parts and the field's part ([`Scene::field_part`]) sum
///   to the scene's cost.
/// - `field`: beside `cost`, not in it: the foliage field's part of its cost to remove, were its cuts in this scan
///   lifted (the field's term is non-linear in the counts, so these don't sum over elements or scans).
/// - `discount`: the evidence discount φ every cell's evidence was divided by.
/// - The posterior at the element in this scan, along the beam (m): the scan alignment error's shift and sd, and the
///   stem front shift's (0 when it is off); and its reflectance level's (dB; 0 while reflectance isn't read). Shifts
///   and sds are the held alignment's (exact after a refit).
#[derive(Debug, Clone, Copy)]
pub struct ElementScan {
    pub elem: u32,
    pub group: Option<u32>,
    pub scan: u32,
    pub cells: u32,
    pub cells_on: u32,
    pub arc: f64,
    pub footprints_across: f64,
    pub surface: u8,
    pub cost: f64,
    pub ray_on: f64,
    pub ray_empty: f64,
    pub ray_in_front: f64,
    pub ray_behind: f64,
    pub alignment: f64,
    pub field: f64,
    pub discount: f64,
    pub align_shift: f64,
    pub align_sd: f64,
    pub front_shift: f64,
    pub front_shift_sd: f64,
    pub refl_level: f64,
    pub refl_level_sd: f64,
}

/// A smooth step: 0 at `x` ≤ 0, 1 at `x` ≥ 1, `3x² − 2x³` between (its slope continuous). Where the cost would cut, it
/// fades (3dtrees research 130 slice-trim).
fn smooth<P: Real>(x: P) -> P {
    let x = x.clamp(0.0, 1.0);
    x * x * (P::c(3.0) - x * 2.0)
}

/// The share of a uniform disc's energy (radius 1) nearer than a line `z` radii from its centre: the CDF of its 1-D
/// marginal, a semicircle.
fn disc_cdf<P: Real>(z: P) -> P {
    let z = z.clamp(-1.0, 1.0);
    (z * (P::c(1.0) - z * z).sqrt() + z.asin()) / std::f64::consts::PI + 0.5
}

fn t_pdf<T: Real>(x: T, nu: f64) -> T {
    let c = super::settings::t_log_norm(nu);
    (-((x * x / nu + 1.0).ln() * (0.5 * (nu + 1.0))) + c).exp()
}

fn dot3<P: Real>(a: &[P; 3], b: &[P; 3]) -> P {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// An element placed in one view ([`placed`]): its section's radius, its middle from the scanner (`c`) and, a stick's,
/// its unit axis, its half-length and the unit normal to the axis and `c` (the side `across` is positive on).
struct Placed<P> {
    r: P,
    c: [P; 3],
    axis: Option<([P; 3], P, [P; 3])>,
}

/// Element `e` placed in a view from `o`, moved by `th` in its parameters if given ([`Scene::information`]'s): a
/// stick's ln r, its axis moved across itself (along p₁ and p₂, perpendicular to it) and its ends by the tilts about
/// its middle (the same directions, rad at its half-length); a patch's ln r and its centre moved (x, y, z).
fn placed<P: Real>(e: &Element, o: &Point, th: Option<&[P]>) -> Placed<P> {
    let lift = |p: &Point| p.map(P::c);
    let (a, b) = match (e.form, th) {
        (Form::Stick { a, b }, None) => (lift(&a), Some(lift(&b))),
        (Form::Stick { a, b }, Some(th)) => {
            let ax = sub(&b, &a);
            let half = 0.5 * norm(&ax);
            let u = ax.map(|x| x / (2.0 * half));
            let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
            let p2 = cross(&u, &p1);
            let mv = |p: &Point, end: f64| [0, 1, 2].map(|k| (th[1] + th[3] * (end * half)) * p1[k] + (th[2] + th[4] * (end * half)) * p2[k] + p[k]);
            (mv(&a, -1.0), Some(mv(&b, 1.0)))
        }
        (Form::Patch { centre }, None) => (lift(&centre), None),
        (Form::Patch { centre }, Some(th)) => ([0, 1, 2].map(|k| th[k + 1] + centre[k]), None),
    };
    let r = th.map_or(P::c(e.r), |th| th[0].exp() * e.r);
    let Some(b) = b else { return Placed { r, c: [0, 1, 2].map(|k| a[k] - o[k]), axis: None } };
    let ax = [0, 1, 2].map(|k| b[k] - a[k]);
    let len = dot3(&ax, &ax).sqrt();
    let u = ax.map(|x| x / len);
    let c = [0, 1, 2].map(|k| (a[k] + b[k]) * 0.5 - o[k]);
    let n = [u[1] * c[2] - u[2] * c[1], u[2] * c[0] - u[0] * c[2], u[0] * c[1] - u[1] * c[0]];
    let nn = dot3(&n, &n).sqrt();
    let n = if nn.v() > 0.0 { let k = P::c(1.0) / nn; n.map(|x| x * k) } else { [0.0, 0.0, 1.0].map(P::c) };
    Placed { r, c, axis: Some((u, len * 0.5, n)) }
}

/// One element's layers in one view, out to the beam's edge and two cells past its section.
fn render(v: &RangeImage, e: &Element, id: u32, em: &Settings) -> Vec<(usize, Layer)> {
    let mut out = Vec::new();
    if e.invalid().is_some() {
        return out;
    }
    let half = match e.form {
        Form::Stick { a, b } => 0.5 * norm(&sub(&b, &a)),
        Form::Patch { .. } => 0.0,
    };
    let mid = e.mid();
    let c = sub(&mid, &v.origin);
    let dist = norm(&c);
    let Some((l0, k0, _)) = v.cell_of(&mid) else { return out };
    let alpha = (half * half + (e.r + em.footprint(dist)).powi(2)).sqrt() / dist;
    let sin_z = ((c[0] * c[0] + c[1] * c[1]).sqrt() / dist).max(0.05);
    let (zs, az) = (v.lat.zenith_step.abs().to_radians(), v.lat.azimuth_step.abs().to_radians());
    let dl = (alpha / (az * sin_z)).min(4000.0).ceil() as i64 + 1;
    let dc = (alpha / zs).min(4000.0).ceil() as i64 + 1;
    let (li, ki) = (l0.round() as i64, k0.round() as i64);
    let p = placed(e, &v.origin, None);
    for l in (li - dl).max(0)..=(li + dl).min(v.n_lines as i64 - 1) {
        for k in (ki - dc).max(0)..=(ki + dc).min(v.n_cols as i64 - 1) {
            let w = v.direction(l as f64, k as f64);
            if let Some(ly) = layer(&p, &w, e, id, em, v.step()) {
                out.push((l as usize * v.n_cols + k as usize, ly));
            }
        }
    }
    out
}

/// Element `e`'s layer (as `id`) in the cell whose ray from the scanner is `w`, placed as `p` ([`placed`]), on a lattice
/// of step `step`: none past the beam's edge and two cells past its section, or past a stick's ends (cut square).
fn layer<P: Real>(p: &Placed<P>, w: &Point, e: &Element, id: u32, em: &Settings, step: f64) -> Option<Layer<P>> {
    let (c, w) = (&p.c, w.map(P::c));
    // the ray's closest approach to the axis (to the centre for a patch): range, distance, signed place across, place
    // along, and the sine of the ray's angle to the axis
    let (tm, d, x, along, sin) = match &p.axis {
        None => {
            let tm = dot3(&w, c);
            let q = [0, 1, 2].map(|j| tm * w[j] - c[j]);
            let d = dot3(&q, &q).sqrt();
            (tm, d, d, P::c(0.0), P::c(1.0))
        }
        Some((u, half, n)) => {
            let wu = dot3(&w, u);
            let a = [0, 1, 2].map(|j| w[j] - wu * u[j]);
            let aa = dot3(&a, &a);
            if aa.v() < 1e-12 {
                return None;
            }
            let cu = dot3(c, u);
            let bv = [0, 1, 2].map(|j| cu * u[j] - c[j]);
            let ab = dot3(&a, &bv);
            let tm = -ab / aa;
            let along = tm * wu - cu;
            if along.v().abs() > half.v() {
                return None;
            }
            let q = [0, 1, 2].map(|j| tm * w[j] - c[j] - along * u[j]);
            (tm, dot3(&q, &q).sqrt(), dot3(&q, n), along, aa.sqrt())
        }
    };
    let r = p.r;
    let a = em.footprint(tm) * 0.5;
    let wide = r + 3f64.sqrt() * e.mismatch;
    if tm.v() <= 0.0 || d.v() > (wide + a + tm * (2.0 * step)).v() {
        return None;
    }
    let fq = [-1.0, 0.0, 1.0].map(|z: f64| {
        let r = (r + z * 3f64.sqrt() * e.mismatch).max(0.0);
        disc_cdf((r - d) / a) - disc_cdf((-r - d) / a)
    });
    let f = (fq[0] + fq[1] * 4.0 + fq[2]) / 6.0;
    let (t, spread, cos) = footprint_depth(tm, d, r, sin, a);
    let inner = ((d - a).max(0.0) / r).min(1.0);
    Some(Layer {
        t,
        far: tm + r * (P::c(1.0) - inner * inner).sqrt() / sin,
        spread: (spread * spread + e.mismatch * e.mismatch).sqrt(),
        f,
        fq,
        cos: (cos * sin).max(0.05),
        db: e.refl(em),
        elem: id,
        op: e.opacity,
        x: x.v(),
        s: along.v(),
    })
}

/// The echo's depth over a footprint (a disc of radius `a`, across the section its semicircle marginal) whose centre
/// passes `d` from the axis at range `tm`: the mean of the section's depth over the part it meets, each part weighted
/// by its energy (the cosine of incidence, Lambertian); the spread of those depths; and the mean cosine over the part
/// it meets. The mismatch stays out: it spreads the depth about the form but doesn't move it. Quadrature: 24 midpoints
/// over the part the footprint meets, so a sliver at the edge is resolved as well as the middle (its cosine sets a
/// located edge's reflectance; 9 nodes over the whole footprint read a 4 % sliver's as the grazing 0.05).
fn footprint_depth<P: Real>(tm: P, d: P, r: P, sin: P, a: P) -> (P, P, P) {
    let (lo, hi) = ((d - a).larger(-r), (d + a).smaller(r));
    // past the edge: the grazing surface
    let grazing = || {
        let c = (P::c(1.0) - (d.smaller(r) / r).sq()).max(0.0).sqrt();
        (tm - r * c / sin, P::c(0.0), c)
    };
    if hi.v() <= lo.v() {
        return grazing();
    }
    let z0 = P::c(0.0);
    let (mut sw, mut swc, mut sz, mut szz) = (z0, z0, z0, z0);
    const N: usize = 24;
    for j in 0..N {
        let x = lo + (hi - lo) * (j as f64 + 0.5) / N as f64;
        let w = (P::c(1.0) - ((x - d) / a).sq()).max(0.0).sqrt();
        let c = (P::c(1.0) - (x / r).sq()).max(0.0).sqrt();
        let z = -r * c / sin;
        sw += w;
        swc += w * c;
        sz += w * c * z;
        szz += w * c * z * z;
    }
    if swc.v() <= 0.0 {
        return grazing();
    }
    let z = sz / swc;
    (tm + z, (szz / swc - z * z).max(0.0).sqrt(), swc / sw)
}

/// Every element rendered into one view.
fn render_all(v: &RangeImage, elems: &[Element], em: &Settings) -> Vec<Vec<(usize, Layer)>> {
    elems.par_iter().enumerate().map(|(i, e)| render(v, e, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, the layers that enter it ([`Layer::enters`]), nearest first ([`keep`]).
fn kbuffer(per: &[Vec<(usize, Layer)>]) -> HashMap<usize, Vec<Layer>> {
    let mut buf: HashMap<usize, Vec<Layer>> = HashMap::new();
    for &(cell, ly) in per.iter().flatten() {
        if ly.enters() {
            buf.entry(cell).or_default().push(ly);
        }
    }
    for ls in buf.values_mut() {
        keep(ls, |l| l.t, |l| l.full());
    }
    buf
}

/// A cell's layers `xs` (range `t`; wholly in it or not, `full`) sorted nearest first, ties in their given order, up
/// to the [`K`]th wholly in it: every subset of its mixture ([`mixture`]) reads its K nearest from these.
fn keep<X>(xs: &mut Vec<X>, t: impl Fn(&X) -> f64, full: impl Fn(&X) -> bool) {
    xs.sort_by(|a, b| t(a).total_cmp(&t(b)));
    let mut n = 0;
    if let Some(i) = xs.iter().position(|x| {
        n += full(x) as usize;
        n == K
    }) {
        xs.truncate(i + 1);
    }
}

/// A cell's likelihood as a mixture (3dtrees research 130 slice-trim): per subset of its partial layers
/// ([`Layer::full`] false), its chance (each partial layer in with its entry weight, [`Layer::entry`], else out) and
/// its layers, the [`K`] nearest. The subset with every layer in comes first; a cell with no partial layer is one
/// subset of chance 1.
fn mixture<P: Real>(layers: &[Layer<P>]) -> Vec<(P, Vec<Layer<P>>)> {
    let part: Vec<usize> = (0..layers.len()).filter(|&i| !layers[i].full()).collect();
    (0..1u32 << part.len()).rev().map(|mask| {
        let (mut p, mut ls) = (P::c(1.0), Vec::with_capacity(layers.len()));
        for (i, l) in layers.iter().enumerate() {
            match part.iter().position(|&j| j == i) {
                None => ls.push(*l),
                Some(b) if mask >> b & 1 == 1 => {
                    p *= l.entry();
                    ls.push(*l);
                }
                Some(_) => p *= P::c(1.0) - l.entry(),
            }
        }
        ls.truncate(K);
        (p, ls)
    }).collect()
}

/// The front share of each of a cell's layers (nearest first): the chance it is the nearest layer in the cell's
/// mixture ([`mixture`]), `w_i Π_{j<i} (1 − w_j)` over the entry weights, normalised over the layers (a cell whose
/// layers are all partial is scored only where one is in). The first layer wholly in it takes the rest, and those
/// behind it none. A cell's evidence is shared among its layers' terms by these, each reading it at its own nodes, so
/// the cost doesn't jump where a partial layer enters in front of another.
fn front_shares<P: Real>(layers: &[Layer<P>]) -> Vec<P> {
    let mut left = P::c(1.0);
    let mut out: Vec<P> = layers.iter().map(|l| {
        let w = l.entry();
        let share = left * w;
        left *= P::c(1.0) - w;
        share
    }).collect();
    let total = P::c(1.0) - left;
    out.iter_mut().for_each(|s| *s = *s / total);
    out
}

/// A cell's log-likelihood at shift `shift` and reflectance level `level`: the log of its mixture's chances times its
/// subsets' likelihoods ([`Pulse::read`]); a cell with one subset is that subset's.
fn mixture_ll<P: Real, T: Over<P>>(pulses: &[(P, Pulse<P>)], shift: T, level: T, em: &Settings) -> T {
    if let [(_, p)] = pulses {
        return p.read(shift, level, em).ll;
    }
    let lls: Vec<T> = pulses.iter().map(|(_, p)| p.read(shift, level, em).ll).collect();
    let m = lls.iter().map(|l| l.v()).fold(f64::NEG_INFINITY, f64::max);
    pulses.iter().zip(&lls).map(|((w, _), &l)| (l - m).exp() * *w).sum::<T>().ln() + m
}

/// A cell's first echo (NaN for none) and the ground along its ray (range, cos), if the background has one.
fn obs_ground(v: &RangeImage, cell: usize, sc: &Background) -> (f64, Option<(f64, f64)>, Point) {
    let obs = if v.cell[cell] == Cell::Echoes { v.range[cell] as f64 } else { f64::NAN };
    let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
    let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
    (obs, ground, w)
}

/// A cell as the likelihood reads it: its pulse per subset of its mixture with that subset's chance ([`mixture`]; the
/// one with every layer in first), its first echo (NaN for none) and its weight ([`Settings::cell_weight`], at its
/// first echo's range or else its layers' ranges weighted by their front shares, [`front_shares`]), in the layers'
/// numbers `P`.
struct CellView<P = f64> {
    obs: f64,
    wt: P,
    pulses: Vec<(P, Pulse<P>)>,
}

/// Cell `cell` of image `v` with `layers` in front of it (nearest first, [`keep`]), read at the cell's surface class
/// ([`cell_class`]), against the foliage field `ext` and the background's ground.
fn cell_view<P: Real>(v: &RangeImage, cell: usize, layers: Vec<Layer<P>>, em: &Settings, sc: &Background, ext: &Foliage) -> CellView<P> {
    let (obs, ground, w) = obs_ground(v, cell, sc);
    let at = if obs.is_finite() { P::c(obs) } else { front_shares(&layers).iter().zip(&layers).map(|(&s, l)| s * l.t).sum() };
    let wt = em.cell_weight(at, v.step());
    let (m, class) = (ext.medium(&v.origin, &w), cell_class(v, cell, em));
    let pulses = mixture(&layers).into_iter()
        .map(|(p, ls)| (p, Pulse::on(occluders(ls, ground), class, &m, v.echoes(cell), v.echo_refls(cell), v.max_targets, em))).collect();
    CellView { obs, wt, pulses }
}

/// Cell `cell`'s surface class: its echo's surface continuity (the excess, data), mid where it has no surface fit.
/// Every return of the cell's pulse reads that class's depth bias and noise (3dtrees research 130 optimisation.md).
fn cell_class(v: &RangeImage, cell: usize, em: &Settings) -> u8 {
    let x = v.excess[cell];
    if v.cell[cell] == Cell::Echoes && x.is_finite() { em.surface(x as f64) } else { 1 }
}

/// The front element's echo in a cell as the likelihood reads it: the first return of the most likely way the cell's
/// occluders come back ([`Receiver::front`]; none without an occluder), its sd widened by the alignment error's spread
/// at the front element's term's surface class `k` ([`term_classes`]) and the stem front shift's prior spread (the front element's, `front_sd`;
/// `other` for a scan off the reference epoch). [`Outcome::of`] against it is the one rule for "the cell's first echo
/// is the element's".
fn front_echo(rx: &Receiver<f64>, k: u8, front_sd: f64, other: bool, em: &Settings) -> Option<Return> {
    rx.front().map(|f| Return { sd: f.sd.hypot(em.align_sd(k, other)).hypot(front_sd), ..f })
}

/// The depth and sd of an element's echo (NaN for none), as [`Outcome::of`] reads it.
fn depth_sd(fe: Option<&Return>) -> (f64, f64) {
    fe.map_or((f64::NAN, f64::NAN), |f| (f.mu, f.sd))
}

/// Each element's term's surface class in one view (`slot`, per element: [`slots`]), which sets the term's alignment
/// spread: the median class of the cells its members mostly fill ([`MOSTLY`]) that hold an echo with a surface fit (mid
/// where none). The echoes themselves read their own cells' class ([`cell_class`]).
fn term_classes(v: &RangeImage, buf: &HashMap<usize, Vec<Layer>>, slot: &[usize], em: &Settings) -> Vec<u8> {
    let mut ks: Vec<Vec<u8>> = vec![Vec::new(); slot.iter().max().map_or(0, |m| m + 1)];
    for (&cell, ls) in buf {
        let Some(front) = ls.iter().find(|l| l.full()) else { continue };
        if front.f > MOSTLY && v.cell[cell] == Cell::Echoes && v.excess[cell].is_finite() {
            ks[slot[front.elem as usize]].push(em.surface(v.excess[cell] as f64));
        }
    }
    let per: Vec<u8> = ks.iter_mut().map(|k| {
        k.sort_unstable();
        k.get(k.len() / 2).copied().unwrap_or(1)
    }).collect();
    slot.iter().map(|&t| per[t]).collect()
}

/// What an element predicts in one cell it fronts ([`Scene::predict`]), on the path the likelihood scores: its fill,
/// the chance its echo is recorded, the chance the pulse arrives (meets nothing of the foliage field before the
/// element's return), the echo's predicted depth and sd ([`front_echo`]) and reflectance (dB, at the element's
/// declared reflectance and its own section radius, without the shape mismatch's blur; NaN where that section
/// misses the beam), the mean cos ι over the part of the footprint the element meets, its term's surface class,
/// the observed first echo (NaN for none) and its reflectance (NaN unknown), the [`Outcome`], whether any echo the
/// likelihood reads of the cell's pulse is on the element (`read`: [`Outcome::of`] for each), its merge share with the
/// medium behind it (`merge`: [`merges`]), the cell's weight ([`Settings::cell_weight`]), the footprint (m) and where
/// its ray passes the axis (across, along). From the same pass over the pulse ([`Pulse::read`]): the expected number
/// of returns the receiver records from the cell's occluders (`returns`: its layers and the ground, merged, detected
/// and capped; the medium's echoes are the foliage field's), the share of the beam the element leaves behind it
/// (`beam`), and how many of the pulse's echoes are the element's in expectation (`owned`: its responsibilities summed
/// over the echoes, the soft `read`). `beam` is NaN and `owned` 0 where the element lies past the beam's end.
pub struct CellPrediction {
    pub cell: usize,
    pub elem: u32,
    pub fill: f64,
    pub recorded: f64,
    pub arrives: f64,
    pub depth: f64,
    pub sd: f64,
    pub refl: f64,
    pub cos: f64,
    pub surface: u8,
    pub obs: f64,
    pub obs_refl: f64,
    pub outcome: Outcome,
    pub read: bool,
    pub merge: f64,
    pub weight: f64,
    pub footprint: f64,
    pub across: f64,
    pub along: f64,
    pub returns: f64,
    pub beam: f64,
    pub owned: f64,
}


/// What lies behind a layer in the medium (the foliage field; κ outside its box), at unshifted ranges ([`merges`]):
/// the share of pulses whose return from it merges with a medium surface behind it, that surface's energy share
/// when it does (the beam the layer leaves, at the medium's mean reflectance), and how far the merge pulls the echo
/// back ([`Settings::pull`], at the merged surface's mean separation).
#[derive(Debug, Clone, Copy)]
struct Merge<P = f64> {
    share: P,
    energy: P,
    pull: P,
}

impl<P: Real> Merge<P> {
    /// No merge.
    fn none() -> Self {
        Merge { share: P::c(0.0), energy: P::c(0.0), pull: P::c(0.0) }
    }
}

/// What a pulse meets as a surface: an element's layer, or the ground plane, at the cosine of incidence along the ray.
#[derive(Debug, Clone, Copy)]
enum Surface<P = f64> {
    Layer(Layer<P>),
    Ground(f64),
}

/// One of a pulse's occluders, nearest first (3dtrees research 130 formulation.md: the owners that deposit energy
/// along the ray): its surface, its range, the foliage field's optical depth to it and its merge with the medium behind
/// it ([`merges`]; none for the ground). The ground, where the background has one, is the last: the beam ends there.
#[derive(Debug, Clone, Copy)]
struct Occluder<P = f64> {
    surface: Surface<P>,
    t: P,
    tau: P,
    merge: Merge<P>,
}

/// A cell's occluders: its `layers` (nearest first) in front of the ground at `ground` = (range, cos), then the
/// ground; no optical depths or merges yet ([`Pulse::new`] reads them).
fn occluders<P: Real>(layers: Vec<Layer<P>>, ground: Option<(f64, f64)>) -> Vec<Occluder<P>> {
    let t_end = ground.map_or(T_MAX, |g| g.0);
    let bare = |surface, t| Occluder { surface, t, tau: P::c(0.0), merge: Merge::none() };
    layers.into_iter().take_while(|l| l.t.v() < t_end).map(|l| bare(Surface::Layer(l), l.t))
        .chain(ground.map(|(t, cos)| bare(Surface::Ground(cos), P::c(t)))).collect()
}

/// One return a pulse can record: a run of occluders within the receiver's resolution of its first, merged, from
/// member `first` (the one that makes its echo) to `last` (its merge with the medium behind is the return's). `s` is the
/// chance it is recorded and `q` the chance it isn't (its members' `q` multiplied, never `1 − s`, which cancels for a
/// bright near-full hit: [`Settings::p_miss`]), `mu` and `sd` its echo's depth and spread; `s_q` the chance it is
/// recorded, `db` its echo's predicted reflectance (dB) and `fill` the share of the beam it meets, at the shape
/// mismatch's three section radii (as [`Layer`]'s `fq`).
/// Its geometry's quantities are in `P` and what the own nodes move in `T` ([`Over`]).
#[derive(Clone, Copy)]
struct Return<P = f64, T = P> {
    first: usize,
    last: usize,
    mu: T,
    sd: T,
    s: T,
    q: T,
    s_q: [T; 3],
    db: [T; 3],
    fill: [P; 3],
}

impl<P: Real, T: Over<P>> Return<P, T> {
    /// Its echo's reflectance (dB) at its mean energy share over the shape mismatch's three section radii (their rule
    /// weights; the energy share is linear in the fill, so this is the echo at the mean fill). [`Scene::information`]
    /// reads it, not the middle radius's: a radius's share grows from 0 as the chord where its section enters the beam,
    /// smoothly, but its reflectance as `10 log10` of it, whose slope is unbounded there.
    fn reflectance(&self) -> T {
        ((self.db[0] / 10.0).exp10() + (self.db[1] / 10.0).exp10() * 4.0 + (self.db[2] / 10.0).exp10()).log10() * 10.0 - 10.0 * 6f64.log10()
    }

    /// The density of its echo's reflectance `x` (dB) as recorded ([`Settings::refl_density`], at range `t`), with
    /// `add` more energy share merged into it (a medium surface's, read at the medium's spread for its part): a mixture
    /// over the shape mismatch's three section radii, each weighted by its rule weight and its chance of being recorded
    /// (a radius whose section misses the beam records nothing, so it says nothing of the reflectance), each widened by
    /// the reflectance's spread between adjacent radii. Three sharp points would leave gaps between them that a
    /// reflectance noise of tenths of a dB reads as misfits.
    fn refl_density(&self, x: f64, em: &Settings, add: P, t: P) -> T {
        if !x.is_finite() || em.refl_noise <= 0.0 {
            return T::c(1.0);
        }
        let w: [T; 3] = [0, 1, 2].map(|q| self.s_q[q] * [1.0, 4.0, 1.0][q]);
        let total: T = w.iter().copied().sum();
        if total.v() <= 0.0 {
            return T::c(1.0);
        }
        let db = self.db.map(|d| if add.v() > 0.0 { ((d / 10.0).exp10() + add).log10() * 10.0 } else { d });
        let gap = |a: T, b: T| if a.v() > -200.0 && b.v() > -200.0 { (if a.v() < b.v() { b - a } else { a - b }) / 2.0 } else { T::c(0.0) };
        let (g1, g2) = (gap(db[1], db[0]), gap(db[2], db[1]));
        let spread = if g2.v() > g1.v() { g2 } else { g1 };
        let medium = |q: usize| if add.v() > 0.0 { T::from(add * em.refl_background_sd) / (db[q] / 10.0).exp10() } else { T::c(0.0) };
        (0..3).filter(|&q| w[q].v() > 0.0).map(|q| w[q] * em.refl_density(x, db[q], spread.hypot(medium(q)), self.fill[q], t)).sum::<T>() / total
    }

    /// The density of its echo at range `o` with reflectance `x` (dB; NaN unknown), its members among `occ`: its own
    /// echo, or, in its merge share, merged with the medium surface behind it, pulled back by the measured pull
    /// ([`Settings::pull`], its sd as large) and brighter by that surface's energy (3dtrees research 130 slice 2c: a
    /// mixture inside the return, never another return).
    fn density(&self, o: f64, x: f64, em: &Settings, occ: &[Occluder<P>]) -> T {
        let (t, merge) = (occ[self.first].t, occ[self.last].merge);
        let alone = t_pdf((T::c(o) - self.mu) / self.sd, em.noise_dof) / self.sd * self.refl_density(x, em, P::c(0.0), t);
        let m = merge.share;
        if m.v() <= 0.0 {
            return alone;
        }
        let sd = self.sd.hypot(T::from(merge.pull));
        alone * (P::c(1.0) - m) + t_pdf((T::c(o) - self.mu - merge.pull) / sd, em.noise_dof) * m / sd * self.refl_density(x, em, merge.energy, t)
    }
}

/// What the receiver reads of one occluder: its echo's depth and sd, the chance it is recorded and (`q`, without
/// cancellation) not, its energy (opacity × the beam it meets × cos ι × reflectance), its energy share when it is hit
/// and the share of the beam it meets, at the shape mismatch's three section radii (the share without the opacity; its
/// echo's reflectance is `10 log10` of it), and the beam left behind it.
struct Member<P, T> {
    depth: T,
    sd: P,
    s: T,
    q: T,
    s_q: [T; 3],
    energy: T,
    share: [T; 3],
    fill: [P; 3],
    beam: P,
}

/// What the receiver makes of a pulse's occluders (3dtrees research 130 formulation.md: the owners deposit energy
/// along the ray, the receiver turns it into echoes): its `members`, one per occluder, and the ways they come back
/// (`patterns`, each a set of returns, nearest first, and its chance) ([`receiver`]).
struct Receiver<P = f64, T = P> {
    members: Vec<Member<P, T>>,
    patterns: Vec<(T, Vec<Return<P, T>>)>,
}

impl<P: Real, T: Over<P>> Receiver<P, T> {
    /// The first return of the most likely pattern (none without an occluder).
    fn front(&self) -> Option<Return<P, T>> {
        self.patterns.iter().max_by(|a, b| a.0.v().total_cmp(&b.0.v())).and_then(|p| p.1.first().copied())
    }

    /// Per member, the expected number of echoes the receiver records of the returns it heads (the nearest member of a
    /// merged return makes its echo), the first `cap` of a pulse's returns kept (the target cap; 0 for none).
    fn expected(&self, cap: usize) -> Vec<f64> {
        let cap = if cap == 0 { usize::MAX } else { cap };
        let mut out = vec![0.0; self.members.len()];
        for (p, rets) in &self.patterns {
            // the chance that n of the returns before this one were recorded
            let mut before = vec![1.0];
            for r in rets {
                let s = r.s.v();
                out[r.first] += p.v() * s * before.iter().take(cap).sum::<f64>();
                let mut next = vec![0.0; before.len() + 1];
                for (n, q) in before.iter().enumerate() {
                    next[n] += q * (1.0 - s);
                    next[n + 1] += q * s;
                }
                before = next;
            }
        }
        out
    }
}

/// The ways a cell's occluders `occ` come back ([`Receiver`]), its layers read at the cell's surface class `k`, with
/// every layer shifted by `shift` along the beam and its reflectance by `level` (dB). Adjacent members resolve into two echoes with [`Settings::p_resolve`] of their
/// separation and energy ratio, else merge, summing their energy shares, into one echo pulled back from the nearest
/// member's by the measured pull ([`Settings::pull`], at the others' energy and energy-weighted separation; 3dtrees
/// research 130 slice 2c, before which it sat at their energy-weighted depth); a pattern's chance fades smoothly to
/// none from 1e-4 to 1e-5 ([`smooth`]), where it is dropped (it was dropped at 1e-4, a jump: slice-trim).
fn receiver<P: Real, T: Over<P>>(occ: &[Occluder<P>], k: u8, shift: T, level: T, em: &Settings) -> Receiver<P, T> {
    let mut beam = P::c(1.0);
    let m: Vec<Member<P, T>> = occ.iter().map(|o| match o.surface {
        Surface::Layer(ly) => {
            let db = level + ly.db;
            let cos = ly.cos.powf(em.refl_incidence);
            let s_q = ly.fq.map(|f| em.p_detect(db, f * beam * cos, ly.t) * ly.op);
            let s = (s_q[0] + s_q[1] * 4.0 + s_q[2]) / 6.0;
            let q_q = ly.fq.map(|f| em.p_miss(db, f * beam * cos, ly.t) * ly.op + (1.0 - ly.op));
            let q = (q_q[0] + q_q[1] * 4.0 + q_q[2]) / 6.0;
            let (b0, unit) = (beam, (db / 10.0).exp10() * (beam * cos));
            let share = ly.fq.map(|f| unit * f);
            beam *= P::c(1.0) - ly.f * ly.op;
            Member { depth: shift + (ly.t + em.depth_bias(k, ly.t)), sd: em.depth_sd(k, ly.spread), s, q, s_q, energy: unit * (ly.f * ly.op), share,
                     fill: ly.fq.map(|f| f * b0), beam }
        }
        Surface::Ground(cos) => {
            let cg = cos.powf(em.refl_incidence);
            let share = beam * cg * 10f64.powf(em.refl_ground / 10.0);
            let s = em.p_detect::<P, P>(P::c(em.refl_ground), beam * cg, o.t) * em.ground_echo;
            let q = em.p_miss::<P, P>(P::c(em.refl_ground), beam * cg, o.t) * em.ground_echo + (1.0 - em.ground_echo);
            Member { depth: T::from(o.t), sd: P::c(em.noise(2).hypot(em.ground_sd / cos)), s: T::from(s), q: T::from(q), s_q: [T::from(s); 3],
                     energy: T::from(share * em.ground_echo), share: [T::from(share); 3], fill: [beam; 3], beam: P::c(0.0) }
        }
    }).collect();
    let n = m.len();
    // each adjacent pair's chance of resolving into two echoes
    let pr: Vec<T> = (1..n).map(|j| {
        let e = (m[j - 1].energy.max(1e-300) / m[j].energy.max(1e-300)).log10() * 10.0;
        em.p_resolve(occ[j].t - occ[j - 1].t, e.clamp(-60.0, 60.0))
    }).collect();
    let group = |first: usize, last: usize| {
        let g = &m[first..=last];
        let db = [0, 1, 2].map(|q| g.iter().map(|x| x.share[q]).sum::<T>().max(1e-30).log10() * 10.0);
        let fill = [0, 1, 2].map(|q| g.iter().map(|x| x.fill[q]).sum::<P>().min(1.0));
        if g.len() == 1 {
            return Return { first, last, mu: g[0].depth, sd: T::from(g[0].sd), s: g[0].s, q: g[0].q, s_q: g[0].s_q, db, fill };
        }
        // the merged echo follows the nearest member, pulled back by the rest
        let rest: T = g[1..].iter().map(|x| x.energy).sum();
        let d = if rest.v() > 0.0 { g[1..].iter().map(|x| x.energy * (x.depth - g[0].depth)).sum::<T>() / rest } else { g[g.len() - 1].depth - g[0].depth };
        let pull = em.pull(d, (g[0].energy.max(1e-300) / rest.max(1e-300)).log10() * 10.0);
        let none: T = g.iter().map(|x| x.q).product();
        let s_q = [0, 1, 2].map(|q| T::c(1.0) - g.iter().map(|x| T::c(1.0) - x.s_q[q]).product::<T>());
        Return { first, last, mu: g[0].depth + pull, sd: T::from(g[0].sd).hypot(pull), s: T::c(1.0) - none, q: none, s_q, db, fill }
    };
    let mut out = Vec::new();
    for mask in 0u32..(1 << n.saturating_sub(1)) {
        // bit j set: members j and j + 1 resolve
        let prob: T = (0..n.saturating_sub(1)).map(|j| if mask >> j & 1 == 1 { pr[j] } else { T::c(1.0) - pr[j] }).product();
        if prob.v() <= 1e-5 {
            continue;
        }
        let prob = if prob.v() < 1e-4 { prob * smooth((prob - 1e-5) / 9e-5) } else { prob };
        let mut rets = Vec::with_capacity(n);
        let mut i = 0;
        for j in 0..n {
            if j + 1 == n || mask >> j & 1 == 1 {
                rets.push(group(i, j));
                i = j + 1;
            }
        }
        out.push((prob, rets));
    }
    Receiver { members: m, patterns: out }
}

/// Each layer's merge with the medium behind it, at unshifted ranges (3dtrees research 130 slice 2c): the share, the
/// site's calibrated map ([`Settings::merge_share`]) of the field's `1 − exp(−b ∫ σ (1 − p))` over the medium from the
/// layer's far side to the next occluder (or the beam's end), b the beam the layer leaves (an opaque layer leaves none,
/// and has no share), σ the foliage field's (κ outside its box), p the receiver's chance of resolving a surface at that
/// separation from the layer's echo ([`Settings::p_resolve`]) at the energy ratio of the layer's echo (at its mean
/// fill) to a medium echo in beam b (at the medium's mean reflectance, `refl_background_mean`); and the pull at the
/// merged surface's mean separation (over eight stretches of the integral, within 2 m behind).
fn merges<P: Real>(occ: &mut [Occluder<P>], m: &Medium, em: &Settings) {
    let mut beam = P::c(1.0);
    for i in 0..occ.len() {
        let Surface::Layer(ly) = occ[i].surface else { continue };
        let echo = (ly.f * beam * ly.cos.powf(em.refl_incidence)).max(1e-12).log10() * 10.0 + ly.db;
        beam *= P::c(1.0) - ly.f * ly.op;
        if beam.v() <= 0.0 {
            continue;
        }
        let medium = beam.log10() * 10.0 + em.refl_background_mean;
        let next = occ.get(i + 1).map_or(P::c(T_MAX), |l| l.t);
        let e = (echo - medium).clamp(-60.0, 60.0);
        let z0 = e * em.res_c + em.res_a;
        let (lo, hi) = (ly.far, next.smaller(ly.t + 2.0));
        let parts: Vec<(P, P)> = (0..8).map(|k| {
            let (a, b) = (lo + (hi - lo) * k as f64 / 8.0, lo + (hi - lo) * (k + 1) as f64 / 8.0);
            ((a + b) * 0.5 - ly.t, m.merge_integral(ly.t, a, b, z0, em.res_b))
        }).collect();
        let whole = m.merge_integral(ly.t, lo, next, z0, em.res_b);
        let near: P = parts.iter().map(|p| p.1).sum();
        let d = if near.v() > 0.0 { parts.iter().map(|p| p.0 * p.1).sum::<P>() / near } else { P::c(0.0) };
        // the share fades to none as the beam left does, under 1e-6 of it (it was cut there, a jump of up to the map's
        // intercept, about 0.4: 3dtrees research 130 slice-trim)
        let share = em.merge_share(P::c(1.0) - (-beam * whole).exp(), ly.t, ly.f) * smooth(beam / 1e-6);
        occ[i].merge = Merge { share, energy: beam * 10f64.powf(em.refl_background_mean / 10.0), pull: em.pull(d, e) };
    }
}

/// One cell's pulse and its owners, as the likelihood reads it (3dtrees research 130 formulation.md: one pass per
/// pulse): its echoes and their reflectances, the most echoes its scan records (`cap`, the target cap; a pulse at it is
/// capped, [`RangeImage::capped`], and says nothing past its last echo), its occluders ([`Occluder`]), the surface class
/// they are read at (its cell's: [`cell_class`]) and what its ray
/// meets of the foliage field besides: the optical depth to the beam's end (`tau_end`) and, per echo (`medium`), the
/// optical depth to it, σ at it and the optical depth to it plus the receiver's resolution, where the medium's path
/// restarts after it. [`Pulse::read`] is the one pass over it. Its occluders' geometry is in `P` (`f64`; a [`Dual`] in
/// [`Scene::information`]).
#[derive(Clone)]
struct Pulse<P = f64> {
    echoes: Vec<f32>,
    refls: Vec<f32>,
    cap: usize,
    occluders: Vec<Occluder<P>>,
    class: u8,
    tau_end: f64,
    medium: Vec<[f64; 3]>,
}

/// What one pass over a pulse gives ([`Pulse::read`]): its log-likelihood (with its derivatives as a [`Dual2`] or a
/// [`Dual`]), the receiver's reading of its occluders ([`Receiver`]: the most likely pattern's first return, each
/// member's beam left, chance recorded and expected echoes), and, per echo, its responsibilities over the pulse's
/// owners (the posterior over who made it): each occluder, by the return it heads, then the medium, then an outlier, a
/// row of `occluders + 2` summing to one.
struct Reading<P, T> {
    ll: T,
    rx: Receiver<P, T>,
    resp: Vec<f64>,
}

impl<P, T> Reading<P, T> {
    /// Occluder `o`'s responsibilities summed over the echoes: how many of the pulse's echoes are its in expectation.
    fn owned(&self, o: usize) -> f64 {
        let n = self.rx.members.len() + 2;
        self.resp.chunks(n).map(|r| r[o]).sum()
    }
}

impl<P: Real> Pulse<P> {
    /// The pulse of a ray from `o` along `w` with its `echoes` (their reflectances `refls`; the scan's target cap
    /// `cap`) against occluders `occ` ([`occluders`]) read at surface class `class` and the foliage field `ext`, its ray
    /// walked once through the field's voxels ([`Medium`]): each occluder's optical depth and merge, and the medium
    /// along the ray.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new(occ: Vec<Occluder<P>>, class: u8, o: &Point, w: &Point, echoes: &[f32], refls: &[f32], cap: usize, em: &Settings, ext: &Foliage) -> Self {
        Self::on(occ, class, &ext.medium(o, w), echoes, refls, cap, em)
    }

    /// The pulse against occluders `occ` along a ray already walked through the field (`m`, [`Medium`]).
    fn on(mut occ: Vec<Occluder<P>>, class: u8, m: &Medium, echoes: &[f32], refls: &[f32], cap: usize, em: &Settings) -> Self {
        for x in occ.iter_mut() {
            x.tau = m.tau(x.t);
        }
        merges(&mut occ, m, em);
        let medium = echoes.iter().map(|&r| [m.tau(r as f64), m.hazard(r as f64), m.tau(r as f64 + em.resolution())]).collect();
        Pulse { echoes: echoes.to_vec(), refls: refls.to_vec(), cap, tau_end: m.tau(Self::end(&occ)), occluders: occ, class, medium }
    }

    /// The range the beam of a pulse with occluders `occ` ends at: the ground's, else [`T_MAX`].
    fn end(occ: &[Occluder<P>]) -> f64 {
        occ.last().filter(|o| matches!(o.surface, Surface::Ground(_))).map_or(T_MAX, |o| o.t.v())
    }

    fn capped(&self) -> bool {
        !self.echoes.is_empty() && self.echoes.len() >= self.cap
    }

    /// The receiver's reading of its occluders, the layers shifted by `shift` along the beam and `level` in
    /// reflectance ([`receiver`]).
    fn receiver<T: Over<P>>(&self, shift: T, level: T, em: &Settings) -> Receiver<P, T> {
        receiver(&self.occluders, self.class, shift, level, em)
    }

    /// The one pass (3dtrees research 130 slice 2c, formulation.md): the pulse whole against the returns its occluders
    /// give ([`receiver`], each with its merge with the medium behind it) and the medium along its ray, the layers
    /// shifted by `shift` along the beam and `level` in reflectance. The likelihood is the mixture over the merge
    /// patterns of [`Pass::lik`]'s recursion over the echoes; the responsibilities are each branch's share of it.
    fn read<T: Over<P>>(&self, shift: T, level: T, em: &Settings) -> Reading<P, T> {
        let rx = self.receiver(shift, level, em);
        let owners = rx.members.len() + 2;
        let mut resp = vec![0.0; self.echoes.len() * owners];
        let pass = Pass { pulse: self, rx: &rx, em, t_end: Self::end(&self.occluders), capped: self.capped(), owners };
        let total = rx.patterns.iter().map(|(p, rets)| *p * pass.lik(rets, 0, 0, P::c(1.0), 1.0, 1.0, 0.0, p.v(), &mut resp)).sum::<T>();
        if total.v() > 0.0 {
            resp.iter_mut().for_each(|r| *r /= total.v());
        }
        Reading { ll: total.max(1e-300).ln(), rx, resp }
    }

    /// The same pulse in the empty scene: the medium and the ground alone, the same for every hypothesis.
    fn empty(&self) -> Self {
        Pulse { occluders: self.occluders.iter().filter(|o| matches!(o.surface, Surface::Ground(_))).copied().collect(), ..self.clone() }
    }
}

/// One pulse's echoes against one way its returns come back ([`Pulse::read`]): the receiver's members, and `owners`
/// columns per echo in the responsibilities.
struct Pass<'a, P, T> {
    pulse: &'a Pulse<P>,
    rx: &'a Receiver<P, T>,
    em: &'a Settings,
    t_end: f64,
    capped: bool,
    owners: usize,
}

impl<P: Real, T: Over<P>> Pass<'_, P, T> {
    /// The chance of echoes `i..` given returns `rets[j..]` not yet accounted for: the pulse meets occluders (the
    /// returns) and a medium (the foliage field), and after any echo the next surface is the medium's (3dtrees
    /// research 130 slice 2c; step 3c's reading, kept where sound). Each echo is one of those returns (the ones before
    /// it not recorded; [`Return::density`], its merge with the medium behind inside it), an echo of the medium (the
    /// returns before it not recorded) or an outlier.
    /// - **The medium's echoes** are a Poisson process along the ray in the beam the echoes so far left, `seen`: each
    ///   echo leaves the share of the beam its reflectance says it didn't take ([`Settings::beam_left`]), as the field
    ///   reads them ([`super::foliage`]).
    ///   That beam is read from the echoes alone, the same for every hypothesis, so how often a pulse echoes again
    ///   doesn't lean on the scene's beam (on the stand's checks the scene's beam there favoured a wider stick
    ///   wherever a pulse returned nothing more: research 130 step 3c). Its path restarts after each echo at the
    ///   receiver's resolution (`tau0`, the optical depth there), as the field's fit reads it.
    /// - **The rest of the beam:** the medium's echo reads `10 log10` of the beam the scene leaves, `scene` (the share
    ///   the returns matched so far leave: the beam behind a return's last member) × `taken` (what the medium's own
    ///   echoes left), dimmer: behind an edge of fill f, `1 − f`, as a located edge reads f.
    /// - **Behind an occluder:** the returns behind a medium echo are met with the chance `taken` (as an element's
    ///   opacity is a chance): one that read as a full hit leaves nothing for them to return, and its pulse is no
    ///   evidence against them.
    ///
    /// After the last echo, the returns left and the medium's path to the beam's end record nothing, unless the pulse
    /// is capped: a capped pulse says nothing past its last echo.
    ///
    /// `prefix` is the chance of the path to here (the pattern's and each earlier echo's branch); each branch adds
    /// `prefix` × its share × what follows to echo i's row of `resp`, under the owner it reads the echo as.
    #[allow(clippy::too_many_arguments)]
    fn lik(&self, rets: &[Return<P, T>], i: usize, j: usize, scene: P, taken: f64, seen: f64, tau0: f64, prefix: f64, resp: &mut [f64]) -> T {
        let (em, p, row) = (self.em, self.pulse, i * self.owners);
        let occ = &p.occluders;
        if i == p.echoes.len() {
            if self.capped {
                return T::c(1.0);
            }
            return rets[j..].iter().map(|r| r.q * taken + (1.0 - taken)).product::<T>() * (-seen * (p.tau_end - tau0).max(0.0)).exp();
        }
        let (o, x) = (p.echoes[i] as f64, p.refls.get(i).map_or(f64::NAN, |&x| x as f64));
        let [tau_o, sig_o, tau_after] = p.medium[i];
        let left = em.beam_left(x);
        let mut total = T::c(0.0);
        // one of the returns
        let mut none = T::c(1.0);
        for (k, r) in rets.iter().enumerate().skip(j) {
            let d = none * ((occ[r.first].tau - tau0).max(0.0) * -seen).exp() * r.s * taken * r.density(o, x, em, occ);
            if d.v() > 0.0 {
                let f = d * (1.0 - P_OUTLIER);
                let rest = self.lik(rets, i + 1, k + 1, self.rx.members[r.last].beam, taken, seen * left, tau_after, prefix * f.v(), resp);
                resp[row + r.first] += prefix * f.v() * rest.v();
                total += f * rest;
            }
            none *= r.q * taken + (1.0 - taken);
        }
        // the medium's
        if o < self.t_end && seen > 0.0 {
            let jn = (j..rets.len()).find(|&k| occ[rets[k].first].t.v() >= o).unwrap_or(rets.len());
            let none: T = rets[j..jn].iter().map(|r| r.q * taken + (1.0 - taken)).product();
            let d = none * (seen * sig_o * (-seen * (tau_o - tau0).max(0.0)).exp()) * em.refl_background(P::c(x) - (scene * taken).max(1e-6).log10() * 10.0);
            if d.v() > 0.0 {
                let f = d * (1.0 - P_OUTLIER);
                let rest = self.lik(rets, i + 1, jn, scene, taken * left, seen * left, tau_after, prefix * f.v(), resp);
                resp[row + self.owners - 2] += prefix * f.v() * rest.v();
                total += f * rest;
            }
        }
        // an outlier
        let f = P_OUTLIER / T_MAX * em.refl_background(x);
        let rest = self.lik(rets, i + 1, j, scene, taken, seen, tau0, prefix * f, resp);
        resp[row + self.owners - 1] += prefix * f * rest.v();
        total + rest * f
    }
}

/// One cell a term has a front share in, as its ray likelihood scores it: its pulse per subset of its mixture
/// ([`CellView`]), its weight times the term's front share ([`front_shares`]), its log-likelihood under the empty
/// scene (the medium and the ground alone: the same for every hypothesis), and what its first echo says about its
/// front element.
struct CellData {
    pulses: Vec<(f64, Pulse)>,
    wt: f64,
    base: f64,
    outcome: Outcome,
}

/// One term's data in one scan (an element's, or a group's: [`Owner`]): the cells its members own (its evidence:
/// [`elem_data`]), and its latent nodes: its alignment nodes with their weights (its alignment shift is `Σ w z`, each
/// hat's weight times its surface class's along-beam spread), and the prior sds of its own two nodes, the stem front
/// shift (m; 0 when off) and the reflectance level (dB; 0 while reflectance isn't read).
struct ElemData {
    /// Its members, as indices into the elements it was built from.
    members: Vec<usize>,
    cells: Vec<CellData>,
    surface: u8,
    /// Cells whose first echo is its.
    cells_on: u32,
    /// The least and greatest of those cells' rays' place across its axis, in radii (clamped to ±1; a stick's).
    on_across: [f64; 2],
    across: f64,
    align: Vec<([i64; 3], f64)>,
    own_sd: Vector2<f64>,
}

impl ElemData {
    /// (δ, reflectance level) at alignment shift `a` and own nodes `y`.
    fn at(&self, a: f64, y: Vector2<f64>) -> (f64, f64) {
        (a + self.own_sd[0] * y[0], self.own_sd[1] * y[1])
    }

    /// Its ray likelihood at `x` = (δ, reflectance level), in total and by [`Outcome`] (in its order), and −ll's slope
    /// and curvature in (δ, level) there, in closed form, one pass over its cells ([`Dual2`]; the value is the `f64`
    /// one), the curvature's negative eigenvalues set to 0. Without a level node only δ's are filled.
    fn quad(&self, x: (f64, f64), em: &Settings) -> Local {
        let level = if self.own_sd[1] > 0.0 { Dual2::var(x.1, 1) } else { Dual2::c(x.1) };
        let (mut total, mut parts) = (Dual2::<2>::c(0.0), [0.0; 4]);
        for c in &self.cells {
            let l = (mixture_ll(&c.pulses, Dual2::var(x.0, 0), level, em) - c.base) * c.wt;
            total += l;
            parts[c.outcome as usize] += l.v;
        }
        let hess = -total.h;
        let hess = if hess[(0, 0)] >= 0.0 && hess[(1, 1)] >= 0.0 && hess.determinant() >= 0.0 {
            hess
        } else {
            let mut eig = hess.symmetric_eigen();
            eig.eigenvalues = eig.eigenvalues.map(|l| l.max(0.0));
            eig.recompose()
        };
        Local { ll: total.v, parts, g: -total.g, hess }
    }

    /// Its curvature over (a, y) at own nodes `y` given its likelihood there (`at`), with y eliminated: over (a, y),
    /// M = Jᵀ Q J + diag(0, I) with J = [[1, s0, 0], [0, 0, s1]], so P = diag(s) Q diag(s) + I over y and the a–y block
    /// is `link`; the slope over y; and the cost's slope and curvature in a, y eliminated (the Schur complement).
    fn eliminate(&self, at: &Local, y: Vector2<f64>) -> Eliminated {
        let (s, q, g) = (self.own_sd, at.hess, at.g);
        let off = s[0] * s[1] * q[(0, 1)];
        let p = Matrix2::new(s[0] * s[0] * q[(0, 0)] + 1.0, off, off, s[1] * s[1] * q[(1, 1)] + 1.0);
        let pi = p.try_inverse().expect("P is the unit prior's precision plus a positive semi-definite part");
        let link = Vector2::new(q[(0, 0)] * s[0], q[(0, 1)] * s[1]);
        let (pl, gy) = (pi * link, s.component_mul(&g) + y);
        Eliminated { pi, det: p.determinant(), link, pl, gy, slope: g[0] - pl[0] * gy[0] - pl[1] * gy[1], curv: q[(0, 0)] - link[0] * pl[0] - link[1] * pl[1] }
    }

    /// Its term at alignment shift `a` (m): its own nodes y at their mode given `a` (Newton from `start`, a step at
    /// most 3 prior sds, halved until the cost falls, stopping under a micronat), the Laplace step over them, and its
    /// cost's slope and curvature in `a` with them eliminated (the Schur complement of its curvature over (a, y)).
    fn term(&self, a: f64, start: Vector2<f64>, em: &Settings) -> Term {
        let s = self.own_sd;
        let prior = |y: Vector2<f64>| 0.5 * (y[0] * y[0] + y[1] * y[1]);
        let mut y = if s == Vector2::zeros() { Vector2::zeros() } else { start };
        let mut at = self.quad(self.at(a, y), em);
        if s != Vector2::zeros() {
            let mut f_now = -at.ll + prior(y);
            for _ in 0..MAX_NEWTON {
                let e = self.eliminate(&at, y);
                let step = cap(e.pi * e.gy);
                let (mut t, mut fall) = (1.0, None);
                while t >= 1e-3 {
                    let yn = y - step * t;
                    let next = self.quad(self.at(a, yn), em);
                    let f_new = -next.ll + prior(yn);
                    if f_new <= f_now {
                        fall = Some(f_now - f_new);
                        (y, f_now, at) = (yn, f_new, next);
                        break;
                    }
                    t *= 0.5;
                }
                if fall.is_none_or(|f| f < 1e-6) {
                    break;
                }
            }
        }
        let e = self.eliminate(&at, y);
        Term {
            align: self.align.clone(),
            fit: -at.ll + prior(y),
            logdet: 0.5 * e.det.ln(),
            slope: e.slope,
            curv: e.curv,
            own: y,
            start,
            own_sd: s,
            own_cov: e.pi,
            own_link: e.pl,
            ray: at.parts.map(|x| -x),
            cells: self.cells.len() as u32,
            cells_on: self.cells_on,
            arc: match self.on_across {
                [lo, hi] if lo <= hi => hi.asin() - lo.asin(),
                _ => 0.0,
            },
            across: self.across,
            surface: self.surface,
        }
    }
}

/// A term's curvature with its own nodes eliminated ([`ElemData::eliminate`]): P⁻¹ and |P| over y, the a–y link
/// and P⁻¹ link, the slope over y, and the cost's slope and curvature in the alignment shift a.
struct Eliminated {
    pi: Matrix2<f64>,
    det: f64,
    link: Vector2<f64>,
    pl: Vector2<f64>,
    gy: Vector2<f64>,
    slope: f64,
    curv: f64,
}

/// A step over a term's own nodes scaled to at most 3 prior sds.
fn cap(step: Vector2<f64>) -> Vector2<f64> {
    step / (step.amax().max(3.0) / 3.0)
}

/// One term's ray likelihood at a point (δ, reflectance level), by outcome, and −ll's slope and curvature there
/// ([`ElemData::quad`]).
struct Local {
    ll: f64,
    parts: [f64; 4],
    g: Vector2<f64>,
    hess: Matrix2<f64>,
}

/// One (element, scan)'s part of its scan's cost at an alignment shift, `fit + logdet`: `fit` its ray likelihood's
/// cost plus its own nodes' prior at their mode ŷ, `logdet` half their Laplace log-determinant ln|P|. With them, the
/// cost's `slope` and `curv`ature in its alignment shift (its own nodes
/// eliminated), where its own nodes' Newton began (`start`: a move recomputes a held term from its reference's, so an
/// unchanged term comes out the same), and what its row reports: ŷ, its own nodes' covariance given the shift P⁻¹ and how they follow it
/// (`own_link`: ŷ moves by −`own_link` per unit of shift), the ray likelihood's cost by outcome, and its cells.
#[derive(Debug, Clone)]
struct Term {
    align: Vec<([i64; 3], f64)>,
    fit: f64,
    logdet: f64,
    slope: f64,
    curv: f64,
    own: Vector2<f64>,
    start: Vector2<f64>,
    own_sd: Vector2<f64>,
    own_cov: Matrix2<f64>,
    own_link: Vector2<f64>,
    ray: [f64; 4],
    cells: u32,
    cells_on: u32,
    arc: f64,
    across: f64,
    surface: u8,
}

impl Term {
    fn cost(&self) -> f64 {
        self.fit + self.logdet
    }

    /// Whether it enters the scan alignment error as `other` does (the same nodes, slope and curvature).
    fn same_alignment(&self, other: &Term) -> bool {
        self.align == other.align && self.slope == other.slope && self.curv == other.curv
    }
}

/// The scan alignment error's nodes about a place: trilinear hats on a grid of spacing ℓ, normalised so the alignment
/// error's variance is the same everywhere (`t(x) = Σ w_j z_j / √Σ w_j²`); each node carries the along-beam shift.
fn hats(x: &Point, l: f64) -> Vec<([i64; 3], f64)> {
    let g = x.map(|v| v / l);
    let f = g.map(|v| v.floor());
    let mut out = Vec::with_capacity(8);
    for c in 0..8 {
        let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
        let w: f64 = (0..3).map(|k| if o[k] == 1 { g[k] - f[k] } else { 1.0 - (g[k] - f[k]) }).product();
        if w > 1e-9 {
            out.push(([0, 1, 2].map(|k| f[k] as i64 + o[k] as i64), w));
        }
    }
    let n = out.iter().map(|(_, w)| w * w).sum::<f64>().sqrt();
    out.iter_mut().for_each(|(_, w)| *w /= n);
    out
}

/// Newton steps at most per mode.
const MAX_NEWTON: usize = 40;

/// One scan's data for the elements `score` (indices into `elems`, each with its layers in `rendered`), one per term
/// (`slot`, per element: [`slots`]), against the k-buffer of every element's layers. A cell's evidence belongs to its
/// layers' terms by their front shares ([`front_shares`]: wholly its front element's where that is wholly in it); its echoes read the cell's own surface class ([`cell_class`]) and its shift the term's
/// alignment spread ([`term_classes`]), so a term depends on no other term's class.
#[allow(clippy::too_many_arguments)]
fn elem_data(v: &RangeImage, rendered: &[Vec<(usize, Layer)>], elems: &[Element], slot: &[usize], score: &[usize], ref_epoch: u32,
             em: &Settings, sc: &Background, ext: &Foliage) -> Vec<ElemData> {
    let buf = kbuffer(rendered);
    let cls = term_classes(v, &buf, slot, em);
    let other = v.epoch != ref_epoch;
    let level = if em.refl_noise > 0.0 && !v.echo_refl.is_empty() { em.refl_level_sd } else { 0.0 };
    let (mut members, mut at, mut by_elem): (Vec<Vec<usize>>, HashMap<usize, usize>, HashMap<u32, usize>) = Default::default();
    for &i in score {
        let d = *at.entry(slot[i]).or_insert_with(|| {
            members.push(Vec::new());
            members.len() - 1
        });
        members[d].push(i);
        by_elem.insert(i as u32, d);
    }
    let mut data: Vec<ElemData> = members.into_iter().map(|ms| {
        let es: Vec<&Element> = ms.iter().map(|&i| &elems[i]).collect();
        let (k, (hs, front)) = (cls[ms[0]], term_nodes(&es, em));
        let s_along = em.align_sd(k, other);
        ElemData { cells: Vec::new(), surface: k, cells_on: 0, on_across: [f64::INFINITY, f64::NEG_INFINITY],
                   across: mean_of(&es, |e| 2.0 * e.r / em.footprint(norm(&sub(&e.mid(), &v.origin)))),
                   align: hs.into_iter().map(|(key, w)| (key, w * s_along)).collect(), own_sd: Vector2::new(front, level), members: ms }
    }).collect();
    // cells in index order, so an element's sums run in the same order in every run (the k-buffer is a hash map)
    let mut order: Vec<usize> = buf.keys().copied().collect();
    order.sort_unstable();
    for cell in order {
        let ls = &buf[&cell];
        // the scored terms with a front share in the cell, in the order their layers come
        let mut shares: Vec<(usize, f64)> = Vec::new();
        for (l, s) in ls.iter().zip(front_shares(ls)).filter(|x| x.1 > 0.0) {
            let Some(&d) = by_elem.get(&l.elem) else { continue };
            match shares.iter_mut().find(|x| x.0 == d) {
                Some(x) => x.1 += s,
                None => shares.push((d, s)),
            }
        }
        if shares.is_empty() || v.cell[cell] == Cell::NoPulse {
            continue; // no pulse was placed here: nothing observed, neither echo nor miss
        }
        let front = ls[0];
        let cv = cell_view(v, cell, ls.clone(), em, sc, ext);
        let (mu, sd) = depth_sd(front_echo(&cv.pulses[0].1.receiver(0.0, 0.0, em), cls[front.elem as usize], elems[front.elem as usize].front_shift_sd(em), other, em).as_ref());
        let outcome = Outcome::of(cv.obs, mu, sd);
        if let Some(&d) = by_elem.get(&front.elem) {
            let el = &mut data[d];
            el.cells_on += (outcome == Outcome::On) as u32;
            if let (Outcome::On, Form::Stick { .. }) = (outcome, elems[front.elem as usize].form) {
                let z = (front.x / elems[front.elem as usize].r).clamp(-1.0, 1.0);
                el.on_across = [el.on_across[0].min(z), el.on_across[1].max(z)];
            }
        }
        let base = cv.pulses[0].1.empty().read(0.0, 0.0, em).ll;
        for (d, s) in shares {
            data[d].cells.push(CellData { pulses: cv.pulses.clone(), wt: cv.wt * s, base, outcome });
        }
    }
    data
}

/// The scan alignment error held at its mode (research 130 offset-field-design.md §3, step 3a): the nodes' mode ẑ
/// and covariance Σ₀ = Λ₀⁻¹ at the last refit, `m` = Σ₀ G₀ (the step Newton would still take, G₀ the cost's slope in
/// the nodes there) and `gm` = G₀ᵀ Σ₀ G₀, and the terms the refit was taken with (`reference`, by [`Owner`]).
///
/// With its own nodes eliminated, each term (an element's or a group's) meets the nodes through its alignment shift alone: slope γ and
/// curvature h along its weights r. For the elements now in the scan, `Λ = I + Σ h r rᵀ` and `G = ẑ + Σ γ r`, and the
/// scan's alignment part is the Laplace marginal of that quadratic about ẑ, `½ ẑᵀẑ + ½ ln|Λ| − ½ Gᵀ Λ⁻¹ G`. Only the
/// elements whose terms differ from the reference enter, as a rank-k correction to Λ₀ (Woodbury; a node new since the
/// refit has its prior alone), so a move costs its own neighbourhood, not the scan, and the cost is a function of the
/// scene, not of the moves that led to it.
struct Held {
    index: HashMap<[i64; 3], usize>,
    z: Vec<f64>,
    cov: nalgebra::DMatrix<f64>,
    m: Vec<f64>,
    gm: f64,
    logdet: f64,
    reference: BTreeMap<Owner, Term>,
}

impl Held {
    /// The alignment shift at weights `r`.
    fn shift(&self, r: &[([i64; 3], f64)]) -> f64 {
        r.iter().map(|(k, w)| self.index.get(k).map_or(0.0, |&i| w * self.z[i])).sum()
    }

    fn cov_at(&self, a: &[i64; 3], b: &[i64; 3]) -> f64 {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.cov[(i, j)],
            _ => (a == b) as u8 as f64,
        }
    }

    /// `rᵀ Σ₀ r2`.
    fn between(&self, r: &[([i64; 3], f64)], r2: &[([i64; 3], f64)]) -> f64 {
        r.iter().map(|(a, wa)| r2.iter().map(|(b, wb)| wa * wb * self.cov_at(a, b)).sum::<f64>()).sum()
    }

    /// The scan's alignment part, given the columns (weights, curvature change, slope change) of the terms that differ
    /// from the reference.
    fn part(&self, cols: &[(&[([i64; 3], f64)], f64, f64)]) -> f64 {
        let prior = 0.5 * self.z.iter().map(|x| x * x).sum::<f64>();
        let k = cols.len();
        if k == 0 {
            return prior + 0.5 * self.logdet - 0.5 * self.gm;
        }
        let s = nalgebra::DMatrix::from_fn(k, k, |i, j| self.between(cols[i].0, cols[j].0));
        let vm = nalgebra::DVector::from_iterator(k, cols.iter().map(|c| c.0.iter().map(|(key, w)| self.index.get(key).map_or(0.0, |&i| w * self.m[i])).sum()));
        let d = nalgebra::DVector::from_iterator(k, cols.iter().map(|c| c.1));
        let c = nalgebra::DVector::from_iterator(k, cols.iter().map(|c| c.2));
        // ln|Λ| = ln|Λ₀| + ln|I + D S| and Λ⁻¹ = Σ₀ − Σ₀ V (I + D S)⁻¹ D Vᵀ Σ₀, with S = Vᵀ Σ₀ V and G = G₀ + V c
        let lu = (nalgebra::DMatrix::identity(k, k) + nalgebra::DMatrix::from_diagonal(&d) * &s).lu();
        let sg = &vm + &s * &c;
        let q = self.gm + 2.0 * c.dot(&vm) + c.dot(&(&s * &c)) - lu.solve(&d.component_mul(&sg)).map_or(f64::NAN, |x| sg.dot(&x));
        prior + 0.5 * (self.logdet + lu.determinant().ln()) - 0.5 * q
    }
}

/// The alignment's nodes over some terms (by first appearance), and its Newton system at nodes `z` from each term's
/// weights, slope and curvature in its shift: the gradient `z + Σ γ r` and the precision `I + Σ h r rᵀ`.
fn alignment_nodes(data: &[ElemData]) -> HashMap<[i64; 3], usize> {
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    for el in data {
        for (k, _) in &el.align {
            let n = index.len();
            index.entry(*k).or_insert(n);
        }
    }
    index
}

fn alignment_system<'a>(index: &HashMap<[i64; 3], usize>, z: &[f64], terms: impl Iterator<Item = (&'a [([i64; 3], f64)], f64, f64)>)
    -> (nalgebra::DVector<f64>, nalgebra::DMatrix<f64>) {
    let n = index.len();
    let mut grad = nalgebra::DVector::from_column_slice(z);
    let mut hess = nalgebra::DMatrix::<f64>::identity(n, n);
    for (align, slope, curv) in terms {
        for &(a, wa) in align {
            let i = index[&a];
            grad[i] += slope * wa;
            for &(b, wb) in align {
                hess[(i, index[&b])] += curv * wa * wb;
            }
        }
    }
    (grad, hess)
}

/// The alignment held at nodes `z` with the system (`grad`, `hess`) there ([`Held`]; its reference set by the caller).
fn held(index: HashMap<[i64; 3], usize>, z: Vec<f64>, grad: nalgebra::DVector<f64>, hess: nalgebra::DMatrix<f64>) -> Held {
    let n = index.len();
    // the precision is the unit prior plus positive semi-definite terms, so only a non-finite cost fails
    let (logdet, cov) = match hess.cholesky() {
        Some(c) => (2.0 * c.l().diagonal().iter().map(|x| x.ln()).sum::<f64>(), c.inverse()),
        None => (f64::NAN, nalgebra::DMatrix::zeros(n, n)),
    };
    let m: Vec<f64> = (&cov * &grad).iter().copied().collect();
    let gm = grad.iter().zip(&m).map(|(a, b)| a * b).sum();
    Held { index, z, cov, m, gm, logdet, reference: BTreeMap::new() }
}

/// A scan's exact Laplace marginal over the alignment error's nodes, given each element's data (the referee for the
/// held alignment, and its refit; 3dtrees research 130 optimisation.md): one Newton over the nodes and every term's
/// own nodes together. Terms are independent given the alignment and each meets it through its shift alone, so the
/// joint curvature is an arrowhead: each term's own block is eliminated by its Schur complement
/// ([`ElemData::eliminate`]), the nodes step on the system that leaves, and each term's own nodes follow, `−P⁻¹(g_y +
/// link Δa)`; both steps at most 3 prior sds, halved together until the total cost falls (each term read once a
/// trial), stopping under a millinat. Then each term's own Newton finishes from the joint mode ([`ElemData::term`]),
/// so its [`Term`] is what the nested solver (the test's referee) defines; the precision is the final terms'. Its value
/// is the held alignment's at its own reference ([`Held::part`] with no columns).
fn solve_alignment(data: &[ElemData], em: &Settings) -> (Held, Vec<Term>) {
    let index = alignment_nodes(data);
    let n = index.len();
    let shift = |el: &ElemData, z: &[f64]| -> f64 { el.align.iter().map(|(k, w)| w * z[index[k]]).sum() };
    let read = |z: &[f64], ys: &[Vector2<f64>]| -> Vec<Local> { data.par_iter().zip(ys).map(|(el, y)| el.quad(el.at(shift(el, z), *y), em)).collect() };
    let objective = |z: &[f64], ys: &[Vector2<f64>], ls: &[Local]| {
        0.5 * z.iter().map(|x| x * x).sum::<f64>() + ls.iter().zip(ys).map(|(l, y)| -l.ll + 0.5 * (y[0] * y[0] + y[1] * y[1])).sum::<f64>()
    };
    let (mut z, mut ys) = (vec![0.0; n], vec![Vector2::zeros(); data.len()]);
    let mut ls = read(&z, &ys);
    for _ in 0..MAX_NEWTON {
        let es: Vec<Eliminated> = data.iter().zip(&ls).zip(&ys).map(|((el, l), y)| el.eliminate(l, *y)).collect();
        let (grad, hess) = alignment_system(&index, &z, data.iter().zip(&es).map(|(el, e)| (&el.align[..], e.slope, e.curv)));
        let step = match (n > 0).then(|| hess.cholesky()).flatten() {
            Some(ch) => {
                let step = ch.solve(&grad);
                &step / (step.amax().max(3.0) / 3.0)
            }
            None => nalgebra::DVector::zeros(n),
        };
        // each term's own step given its shift's, Δa = −rᵀ step
        let dys: Vec<Vector2<f64>> = data.iter().zip(&es).map(|(el, e)| {
            if el.own_sd == Vector2::zeros() {
                return Vector2::zeros();
            }
            let da = -el.align.iter().map(|(k, w)| w * step[index[k]]).sum::<f64>();
            -cap(e.pi * (e.gy + e.link * da))
        }).collect();
        let f_now = objective(&z, &ys, &ls);
        let (mut t, mut fall) = (1.0, None);
        while t >= 1e-3 {
            let zn: Vec<f64> = z.iter().zip(step.iter()).map(|(a, b)| a - t * b).collect();
            let yn: Vec<Vector2<f64>> = ys.iter().zip(&dys).map(|(y, d)| y + d * t).collect();
            let ln = read(&zn, &yn);
            let f_new = objective(&zn, &yn, &ln);
            if f_new <= f_now {
                fall = Some(f_now - f_new);
                (z, ys, ls) = (zn, yn, ln);
                break;
            }
            t *= 0.5;
        }
        if fall.is_none_or(|f| f < 1e-3) {
            break;
        }
    }
    let terms: Vec<Term> = data.par_iter().zip(&ys).map(|(el, y)| el.term(shift(el, &z), *y, em)).collect();
    let (grad, hess) = alignment_system(&index, &z, terms.iter().map(|t| (&t.align[..], t.slope, t.curv)));
    (held(index, z, grad, hess), terms)
}

/// One scan's fit: per [`Owner`] its term (none where the scan doesn't see it), the held alignment, and the scan's cost.
struct Fit {
    terms: BTreeMap<Owner, Term>,
    held: Held,
    cost: f64,
}

/// A move's changes to a scan's fit: the terms by owner (none: it left the scan).
type Changes = Vec<(Owner, Option<Term>)>;

impl Fit {
    /// The scan's cost with the terms `over` (none: the term left) in place of its own: every term's cost plus the
    /// held alignment's part ([`Held::part`]), its columns the terms that differ from the reference, in owner order.
    fn total(&self, over: &HashMap<Owner, Option<Term>>) -> f64 {
        let owners: BTreeSet<Owner> = self.terms.keys().chain(self.held.reference.keys()).chain(over.keys()).copied().collect();
        let mut cost = 0.0;
        let mut cols: Vec<(&[([i64; 3], f64)], f64, f64)> = Vec::new();
        for o in owners {
            let now = match over.get(&o) {
                Some(t) => t.as_ref(),
                None => self.terms.get(&o),
            };
            let was = self.held.reference.get(&o);
            if let Some(t) = now {
                cost += t.cost();
            }
            match (was, now) {
                (Some(r), Some(t)) if r.same_alignment(t) => {}
                (Some(r), Some(t)) if r.align == t.align => cols.push((&t.align, t.curv - r.curv, t.slope - r.slope)),
                (r, t) => {
                    cols.extend(r.map(|r| (&r.align[..], -r.curv, -r.slope)));
                    cols.extend(t.map(|t| (&t.align[..], t.curv, t.slope)));
                }
            }
        }
        cost + self.held.part(&cols)
    }
}

/// How well one scan pins some elements' geometry down ([`Scene::information`]): in their parameters (n of them), the
/// cost's slope (n), the expected (Fisher) information (n × n, row-major) and the slope's variance were the cells
/// independent (n × n).
#[derive(Debug, Clone)]
pub struct Information {
    pub slope: Vec<f64>,
    pub expected: Vec<f64>,
    pub gradient_variance: Vec<f64>,
}

/// An element's counts in one view (the sufficient statistics of its shape mismatch and opacity), each cell counted in footprints (its
/// [`Settings::cell_weight`] times φ):
/// - **Shape mismatch:** over the cells it mostly fills ([`MOSTLY`]) whose first echo is its, from its sharp depth
///   (its own mismatch left out of the sd), the weighted sum of `max(r² − sd², 0)`, the weights' sum and the cells.
/// - **Opacity:** over its cells at fill > 0.9, the weighted counts with its echo, none, and one behind.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementCounts {
    pub scan: u32,
    pub mismatch_ss: f64,
    pub mismatch_w: f64,
    pub mismatch_cells: u32,
    pub on: f64,
    pub empty: f64,
    pub behind: f64,
}

/// What a scene call can't do: an element id that isn't live, a scan index out of range, or an element that can't be
/// scored ([`Element::invalid`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SceneError(pub String);

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SceneError {}

/// A scene: elements held against some range images, and the cost of a move (synthesis §16's incremental form).
///
/// - Each element is rendered once per image, and re-rendered only when it is added or changed.
/// - A scan's cost is a term per element it sees, plus the scan alignment error's part. Moves (`add`, `remove`,
///   `change`) and queries (`costs_to_add`, `costs_to_remove`, `information`) hold the alignment ([`Held`]): its mode
///   and precision stay as the last refit left them, and a move's change to them is a rank-k correction on the nodes
///   it touches. So a move recomputes only the terms of the elements whose cells it touches, not the scan, and
///   removing an element restores the cost exactly (synthesis §18 invariant 7). [`Scene::refit_alignment`] re-solves the alignment (a move like `refit_foliage`), after which
///   the cost is the exact Laplace marginal ([`Scene::exact_cost`], the referee); the scene is built refitted.
/// - Element ids are stable and never reused.
/// - The reference epoch (the scanning day the alignment error's spread is stated for) is declared: a scan of another epoch adds the other-epoch part
///   to its alignment error.
pub struct Scene {
    views: Vec<Arc<RangeImage>>,
    em: Settings,
    sc: Background,
    /// The foliage field the scene is scored against (held between refits), and its live counts (none without a
    /// box, or when the field is held from elsewhere).
    ext: Foliage,
    field: Option<FieldCounts>,
    ref_epoch: u32,
    elems: Vec<Option<Element>>,
    /// [image][id]: the element's layers (empty when removed or unseen).
    rendered: Vec<Vec<Vec<(usize, Layer)>>>,
    /// [image]: per cell, the elements whose layers enter it ([`Entered`]).
    cells: Vec<HashMap<usize, Vec<Entered>>>,
    /// [image]: its fit.
    fits: Vec<Fit>,
}

/// An element's layer in a cell's index: its id, range, [`Layer::cut`] and whether it is wholly in the cell
/// ([`Layer::full`]).
#[derive(Debug, Clone, Copy)]
struct Entered {
    id: u32,
    t: f64,
    cut: f64,
    full: bool,
}

impl Entered {
    fn of(id: u32, l: &Layer, em: &Settings) -> Entered {
        Entered { id, t: l.t, cut: l.cut(em), full: l.full() }
    }
}

/// A cell's entered layers' elements, nearest first as its k-buffer keeps them ([`keep`]), and how many of the first
/// have a front share ([`front_shares`]): those up to the first wholly in it.
fn kept(mut e: Vec<Entered>) -> (Vec<u32>, usize) {
    keep(&mut e, |x| x.t, |x| x.full);
    let owners = e.iter().position(|x| x.full).map_or(e.len(), |i| i + 1);
    (e.into_iter().map(|x| x.id).collect(), owners)
}

/// Enter an element's layers in one image's cell index ([`Layer::enters`]), or take them out.
fn index(cells: &mut HashMap<usize, Vec<Entered>>, id: u32, layers: &[(usize, Layer)], enter: bool, em: &Settings) {
    for (c, l) in layers.iter().filter(|(_, l)| l.enters()) {
        let e = cells.entry(*c).or_default();
        if enter {
            e.push(Entered::of(id, l, em));
        } else {
            e.retain(|x| x.id != id);
        }
    }
}

/// Where the foliage field's fit stops reading cell `c`'s ray: the nearest cut of the layers in it (∞ for none).
fn cut_at(cells: &HashMap<usize, Vec<Entered>>, c: usize) -> f64 {
    cells.get(&c).into_iter().flatten().map(|x| x.cut).fold(f64::INFINITY, f64::min)
}

/// An element's segment for [`FieldCounts::claim`]: a stick's ends, or a patch's centre twice.
fn segment(e: &Element) -> (Point, Point) {
    match e.form {
        Form::Stick { a, b } => (a, b),
        Form::Patch { centre } => (centre, centre),
    }
}

/// The numbers of [`Scene::information`]'s pass: the slope in an element's parameters (its first five; a patch's
/// first four) and in the cell's own nodes, its shift along the beam ([`SHIFT`]) and its reflectance level ([`LEVEL`]).
type Geometry = Dual<7>;
const SHIFT: usize = 5;
const LEVEL: usize = 6;

/// What a cell's echo tells per unit of its slope (research 130 step 3a), the one reading of a cell [`Scene::information`]
/// and [`Scene::divergence`] share: in its chance of an echo `p` (a Bernoulli, `p` kept within [1e-3, 1 − 1e-3]) and,
/// given one, in its depth (of variance `var`, the return's own, without the shifts' spread) and in its reflectance (the
/// per-echo noise, wider at fill `fill`), each of these two a t density's share `(ν + 1) / (ν + 3)` of a Gaussian's.
fn cell_information(em: &Settings, p: f64, var: f64, fill: f64) -> [f64; 3] {
    let p = p.clamp(1e-3, 1.0 - 1e-3);
    let (k_depth, k_refl) = ((em.noise_dof + 1.0) / (em.noise_dof + 3.0), (em.refl_dof + 1.0) / (em.refl_dof + 3.0));
    [1.0 / (p * (1.0 - p)), p * k_depth / var.max(1e-12), p * k_refl / em.refl_noise.hypot(em.refl_edge * (1.0 - fill.clamp(0.0, 1.0))).powi(2)]
}

/// One cell as [`Scene::information`] reads it, in the numbers `P` of its pass: its share of the cost (its weighted log
/// likelihood against the empty scene's, with the sign the cost has), its weight's value, and the element's echo there
/// (the front return of its most likely way back, [`Receiver::front`]): the chance the pulse returns it (it arrives,
/// and is recorded), its depth and its reflectance at its mean fill ([`Return::reflectance`]; −200 dB or less where
/// no section radius meets the beam), with its depth's own sd.
struct CellReading<P> {
    cost: P,
    weight: f64,
    /// Its layers' front shares ([`front_shares`]).
    shares: Vec<f64>,
    echo: Option<([P; 3], f64)>,
}

/// One scan's evidence on n quantities from the cells of some elements, with the latents the cells share
/// marginalised out once (research 130 step 3a, slice 4): the scan alignment error's nodes about each term (shared
/// by neighbours), and each term's stem front shift and reflectance level, each a unit node scaled as the cost scales
/// it ([`term_nodes`]). Each observable adds its slope in the quantities and its loading on the nodes, at its
/// information weight and the cell's weight w; the information over (quantities, nodes) is kept at w and at w² (the
/// gradient's variance were the cells independent).
struct Shared {
    n: usize,
    nodes: HashMap<[i64; 3], usize>,
    /// Per element: its term's hats, the index of its front shift node (its reflectance level's is next) and that
    /// node's prior sd ([`term_nodes`]).
    elems: Vec<(Vec<([i64; 3], f64)>, usize, f64)>,
    m: nalgebra::DMatrix<f64>,
    j: nalgebra::DMatrix<f64>,
}

impl Shared {
    /// For `elems`, a group's members sharing one term's nodes ([`Owner`]: hats at their middles' mean, one front
    /// shift and one reflectance level), as the cost has them when the group is these elements.
    fn new(n: usize, elems: &[Element], em: &Settings) -> Shared {
        let mut nodes: HashMap<[i64; 3], usize> = HashMap::new();
        let slot = slots(&elems.iter().enumerate().map(|(k, e)| owner(k as u32, e)).collect::<Vec<_>>());
        let terms = slot.iter().max().map_or(0, |m| m + 1);
        let tn: Vec<_> = (0..terms).map(|t| term_nodes(&elems.iter().zip(&slot).filter(|(_, &u)| u == t).map(|(e, _)| e).collect::<Vec<_>>(), em)).collect();
        for (k, _) in tn.iter().flat_map(|t| &t.0) {
            let i = nodes.len();
            nodes.entry(*k).or_insert(i);
        }
        let first = nodes.len();
        let elems: Vec<_> = slot.iter().map(|&t| (tn[t].0.clone(), first + 2 * t, tn[t].1)).collect();
        let size = n + first + 2 * terms;
        Shared { n, nodes, elems, m: nalgebra::DMatrix::zeros(size, size), j: nalgebra::DMatrix::zeros(size, size) }
    }

    /// An observable's loading on the nodes, in element `k`'s cell, from its slope in the cell's shift along the beam
    /// (`d_shift`: over the term's alignment hats, each at its class's along-beam spread `align_sd`, and its stem front
    /// shift) and in its reflectance level (`d_level`, at that node's prior sd `level_sd`).
    fn load(&self, k: usize, align_sd: f64, d_shift: f64, d_level: f64, level_sd: f64) -> Vec<(usize, f64)> {
        let (h, own, front) = &self.elems[k];
        let mut out = Vec::new();
        if d_shift != 0.0 {
            out.extend(h.iter().map(|(key, w)| (self.nodes[key], w * align_sd * d_shift)));
            out.push((*own, front * d_shift));
        }
        if d_level != 0.0 {
            out.push((own + 1, level_sd * d_level));
        }
        out
    }

    fn add(&mut self, g: &[f64], load: &[(usize, f64)], info: f64, w: f64) {
        let mut x = nalgebra::DVector::<f64>::zeros(self.m.nrows());
        x.rows_mut(0, self.n).copy_from_slice(g);
        for &(i, v) in load {
            x[self.n + i] += v;
        }
        let outer = &x * x.transpose();
        self.m += &outer * (w * info);
        self.j += outer * (w * w * info);
    }

    /// The information on the quantities (n × n, row-major) with the nodes marginalised under their unit priors (a
    /// Schur complement), and the variance of the efficient gradient (the quantities' less β = M_qn (M_nn + I)⁻¹ times
    /// the nodes', the priors' included).
    fn finish(self) -> [Vec<f64>; 2] {
        let (n, l) = (self.n, self.m.nrows() - self.n);
        let block = |x: &nalgebra::DMatrix<f64>, r: usize, c: usize, nr: usize, nc: usize| x.view((r, c), (nr, nc)).into_owned();
        let eye = nalgebra::DMatrix::<f64>::identity(l, l);
        let inv = (block(&self.m, n, n, l, l) + &eye).try_inverse().expect("the nodes' precision is at least their unit prior");
        let (mq, jq) = (block(&self.m, 0, n, n, l), block(&self.j, 0, n, n, l));
        let beta = &mq * inv;
        let m = block(&self.m, 0, 0, n, n) - &beta * mq.transpose();
        let j = block(&self.j, 0, 0, n, n) - &beta * jq.transpose() - &jq * beta.transpose() + &beta * (block(&self.j, n, n, l, l) + eye) * beta.transpose();
        [m, j].map(|x| x.transpose().iter().copied().collect())
    }
}

impl Scene {
    /// A scene of `elems` against `views`, the alignment error's spread stated for the scans of epoch `ref_epoch`; with `spec`, the foliage
    /// field is estimated in its box as the residual of the elements, otherwise it is the background's κ.
    pub fn new(views: Vec<Arc<RangeImage>>, elems: &[Element], em: Settings, sc: Background, spec: Option<FoliageBox>, ref_epoch: u32) -> Self {
        let n = views.len();
        let mut c = Scene { views, em, sc, ext: Foliage::uniform(sc.kappa), field: None, ref_epoch, elems: elems.iter().map(|e| Some(*e)).collect(),
                             rendered: Vec::new(), cells: vec![HashMap::new(); n], fits: Vec::new() };
        c.rendered = c.views.par_iter().map(|v| render_all(v, elems, &c.em)).collect();
        for (cells, rendered) in c.cells.iter_mut().zip(&c.rendered) {
            for (i, ls) in rendered.iter().enumerate() {
                index(cells, i as u32, ls, true, &em);
            }
        }
        if let Some(sp) = spec {
            let cuts: Vec<HashMap<usize, f64>> = c.cells.iter().map(|cs| cs.keys().map(|&k| (k, cut_at(cs, k))).collect()).collect();
            let mut f = FieldCounts::new(sp, &c.views.iter().map(|v| &**v).collect::<Vec<_>>(), &em, &cuts, sc.kappa);
            for e in elems {
                let (a, b) = segment(e);
                f.claim(&a, &b, e.r, true);
            }
            c.ext = f.field(sc.kappa);
            f.reread(&c.ext);
            c.field = Some(f);
        }
        c.fits = (0..n).into_par_iter().map(|s| c.solve(s)).collect();
        c
    }

    /// The scene's cost against the empty scene: its elements' cells, each against the same cell with the foliage
    /// field alone, and the field's part ([`Scene::field_part`]).
    pub fn cost(&self) -> f64 {
        self.fits.iter().map(|f| f.cost).sum::<f64>() + self.field_part()
    }

    /// The foliage field's part of the cost: its marginal's change against the empty scene's, less the plug-in the
    /// cells' ratios already read ([`FieldCounts`]); 0 without a box or with the field held from elsewhere.
    pub fn field_part(&self) -> f64 {
        self.field.as_ref().map_or(0.0, FieldCounts::cost)
    }

    /// The scene's exact cost: each scan's Laplace marginal over the alignment error re-solved for the elements as
    /// they are (the referee for the held alignment; the scene unchanged).
    pub fn exact_cost(&self) -> f64 {
        (0..self.views.len()).into_par_iter().map(|s| self.solve(s).cost).collect::<Vec<f64>>().iter().sum::<f64>() + self.field_part()
    }

    /// Re-solve the scan alignment error for the elements as they are (a move: its cost change); after it the cost
    /// is [`Scene::exact_cost`], and moves hold the new mode and precision.
    pub fn refit_alignment(&mut self) -> f64 {
        let before = self.cost();
        self.fits = (0..self.views.len()).into_par_iter().map(|s| self.solve(s)).collect();
        self.cost() - before
    }

    /// Re-read the foliage field a pulse meets from the counts the moves since the last refit kept, and refit the
    /// alignment (a move: its cost change). Moves hold that field; after this one the scene's cost is a fresh scene's.
    pub fn refit_foliage(&mut self) -> f64 {
        let before = self.cost();
        let Some(f) = &mut self.field else { return 0.0 };
        self.ext = f.field(self.sc.kappa);
        f.reread(&self.ext);
        self.refit_alignment();
        self.cost() - before
    }

    /// Hold a foliage field estimated elsewhere (another scene's, from other scans) fixed: `refit_foliage` then
    /// leaves it. The cost change.
    pub fn hold_foliage(&mut self, ext: &Foliage) -> f64 {
        self.ext = ext.clone();
        self.field = None;
        self.refit_alignment()
    }

    /// Image `s`'s cell index after element `id`'s layers `old` give way to `new`, and the change to the field's
    /// counts added to `delta`: each cell whose cut moves is counted again ([`foliage::recount`]).
    fn reindex(&mut self, s: usize, id: u32, old: &[(usize, Layer)], new: &[(usize, Layer)], delta: &mut FixedCounts) {
        let em = self.em;
        let touched: BTreeSet<usize> = old.iter().chain(new).filter(|(_, l)| l.full()).map(|(c, _)| *c).collect();
        let before: Vec<f64> = touched.iter().map(|&c| cut_at(&self.cells[s], c)).collect();
        index(&mut self.cells[s], id, old, false, &em);
        index(&mut self.cells[s], id, new, true, &em);
        if let Some(f) = &self.field {
            for (&c, from) in touched.iter().zip(before) {
                foliage::recount(&self.views[s], &f.spec, &em, c, from, cut_at(&self.cells[s], c), delta);
            }
        }
    }

    /// Change the field's counts by `delta`: the change in its part of the cost.
    fn recount(&mut self, delta: FixedCounts) -> f64 {
        match &mut self.field {
            Some(f) => f.apply(delta, &self.ext),
            None => 0.0,
        }
    }

    /// Image `s`'s change to the field's counts were the live elements `without` taken out and elements of layers
    /// `with` put in, added to `delta`, the scene unchanged.
    fn field_delta(&self, s: usize, without: &[u32], with: &[Vec<(usize, Layer)>], delta: &mut FixedCounts) {
        let Some(f) = &self.field else { return };
        let fronting = |ls: &[(usize, Layer)]| ls.iter().filter(|(_, l)| l.full()).map(|(c, l)| (*c, l.cut(&self.em))).collect::<Vec<_>>();
        let index = &self.cells[s];
        // per touched cell, the least cut of the layers put in front of it
        let mut put: BTreeMap<usize, f64> = BTreeMap::new();
        let gone = without.iter().flat_map(|&w| fronting(&self.rendered[s][w as usize])).map(|(c, _)| (c, f64::INFINITY));
        for (c, cut) in with.iter().flat_map(|ls| fronting(ls)).chain(gone) {
            let e = put.entry(c).or_insert(f64::INFINITY);
            *e = e.min(cut);
        }
        for (c, cut) in put {
            let kept = index.get(&c).into_iter().flatten().filter(|x| !without.contains(&x.id)).map(|x| x.cut).fold(cut, f64::min);
            foliage::recount(&self.views[s], &f.spec, &self.em, c, cut_at(index, c), kept, delta);
        }
    }

    /// Element `e` claims the field's voxels within its margin (or, `enter` false, releases them).
    fn claim(&mut self, e: &Element, enter: bool) {
        if let Some(f) = &mut self.field {
            let (a, b) = segment(e);
            f.claim(&a, &b, e.r, enter);
        }
    }

    /// How many scans the scene holds.
    pub fn n_scans(&self) -> usize {
        self.views.len()
    }

    /// The foliage field.
    pub fn foliage_field(&self) -> &Foliage {
        &self.ext
    }

    /// An error unless `id` is a live element.
    pub fn check_id(&self, id: u32) -> Result<(), SceneError> {
        match self.elems.get(id as usize) {
            Some(Some(_)) => Ok(()),
            _ => Err(SceneError(format!("no element {id}"))),
        }
    }

    /// An error unless `s` is one of the scene's scans.
    pub fn check_scan(&self, s: usize) -> Result<(), SceneError> {
        if s < self.views.len() { Ok(()) } else { Err(SceneError(format!("no scan {s}: the scene holds {}", self.views.len()))) }
    }

    /// Promote: add an element and take the mass it explains out of the foliage field. Its id and the cost change.
    pub fn promote(&mut self, e: Element) -> (u32, f64) {
        let (id, d) = self.add(e);
        (id, d + self.refit_foliage())
    }

    /// Demote: remove an element and give the mass back to the foliage field. The cost change.
    pub fn demote(&mut self, id: u32) -> f64 {
        let d = self.remove(id);
        d + self.refit_foliage()
    }

    /// The foliage field and its counts: per voxel with counts, (voxel, σ as held, the counts as the scene now is,
    /// whether an element's margin claims it). Empty when the field is held from elsewhere.
    pub fn foliage(&self) -> (Option<FoliageBox>, Vec<([i32; 3], f64, (f64, f64), bool)>) {
        let Some(f) = &self.field else { return (self.ext.spec, Vec::new()) };
        let mut out: Vec<_> = f.counts().into_iter().map(|(k, c)| (k, self.ext.sigma(&k), c, f.claimed.contains_key(&k))).collect();
        out.sort_by_key(|x| x.0);
        (self.ext.spec, out)
    }

    /// For each of image `v`'s `cells`: the optical depth of the foliage field across its box, and the ray's entry and
    /// exit ranges (NaN where it misses the box).
    pub fn transmission(&self, v: &RangeImage, cells: &[usize]) -> Vec<(f64, f64, f64)> {
        let Some(sp) = self.ext.spec else { return vec![(0.0, f64::NAN, f64::NAN); cells.len()] };
        cells.par_iter().map(|&k| {
            let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
            match foliage::clip(&sp, &v.origin, &w) {
                Some((a, b)) => {
                    let m = self.ext.medium(&v.origin, &w);
                    (m.tau(b) - m.tau(a), a, b)
                }
                None => (f64::NAN, f64::NAN, f64::NAN),
            }
        }).collect()
    }

    /// Each scan's alignment part ([`Held::part`]): its cost less its elements' terms.
    pub fn alignment_parts(&self) -> Vec<f64> {
        self.fits.iter().map(|fit| fit.cost - fit.terms.values().map(Term::cost).sum::<f64>()).collect()
    }

    /// A row per (term, scan), by element id: an element's term, or a group's under its first member's id (the scan's
    /// alignment part is [`Scene::alignment_parts`]).
    pub fn rows(&self) -> Vec<ElementScan> {
        let mut first: HashMap<u32, u32> = HashMap::new();
        for (i, e) in self.elems.iter().enumerate().rev() {
            if let Some(g) = e.and_then(|e| e.group) {
                first.insert(g, i as u32);
            }
        }
        // per (scan, term): its live members, and the field's part of its cost to remove there
        let members: Vec<((usize, Owner), Vec<u32>)> = (0..self.views.len()).flat_map(|s| {
            let mut by: BTreeMap<Owner, Vec<u32>> = BTreeMap::new();
            for (i, e) in self.elems.iter().enumerate() {
                if let Some(e) = e.filter(|_| !self.rendered[s][i].is_empty()) {
                    by.entry(owner(i as u32, &e)).or_default().push(i as u32);
                }
            }
            by.into_iter().map(move |(o, ids)| ((s, o), ids))
        }).collect();
        let field: HashMap<(usize, Owner), f64> = match &self.field {
            Some(f) => members.par_iter().map(|(k, ids)| {
                let mut delta = FixedCounts::default();
                self.field_delta(k.0, ids, &[], &mut delta);
                (*k, -f.change(&delta, &self.ext))
            }).collect(),
            None => HashMap::new(),
        };
        let mut rows = Vec::new();
        for (s, fit) in self.fits.iter().enumerate() {
            for (o, t) in &fit.terms {
                let (i, group) = match *o {
                    Owner::Alone(i) => (i, None),
                    Owner::Group(g) => (first[&g], Some(g)),
                };
                // the shift's posterior from the held alignment, and the own nodes' with it marginalised
                let var = fit.held.between(&t.align, &t.align);
                let own_sd = |j: usize| t.own_sd[j] * (t.own_cov[(j, j)] + t.own_link[j] * t.own_link[j] * var).max(0.0).sqrt();
                rows.push(ElementScan {
                    elem: i, group, scan: s as u32, cells: t.cells, cells_on: t.cells_on, arc: t.arc, footprints_across: t.across, surface: t.surface,
                    cost: t.cost(), ray_on: t.ray[0], ray_empty: t.ray[1], ray_in_front: t.ray[2], ray_behind: t.ray[3],
                    alignment: t.fit - t.ray.iter().sum::<f64>() + t.logdet, field: field.get(&(s, *o)).copied().unwrap_or(0.0),
                    discount: self.em.discount,
                    align_shift: fit.held.shift(&t.align), align_sd: var.max(0.0).sqrt(), front_shift: t.own_sd[0] * t.own[0], front_shift_sd: own_sd(0),
                    refl_level: t.own_sd[1] * t.own[1], refl_level_sd: own_sd(1),
                });
            }
        }
        rows.sort_by_key(|r| (r.elem, r.scan));
        rows
    }

    /// Add an element: its id and the cost change.
    pub fn add(&mut self, e: Element) -> (u32, f64) {
        let id = self.elems.len() as u32;
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter()
            .map(|s| if layers[s].is_empty() { Changes::default() } else { self.reterm(s, &[], &[(e, layers[s].clone())], id) }).collect();
        self.elems.push(Some(e));
        self.claim(&e, true);
        let (mut d, mut delta) = (0.0, FixedCounts::default());
        for (s, (l, ch)) in layers.into_iter().zip(changes).enumerate() {
            self.reindex(s, id, &[], &l, &mut delta);
            self.rendered[s].push(l);
            d += self.apply(s, ch);
        }
        (id, d + self.recount(delta))
    }

    /// Remove a live element ([`Scene::check_id`]): the cost change.
    pub fn remove(&mut self, id: u32) -> f64 {
        let first = self.elems.len() as u32;
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter()
            .map(|s| if self.rendered[s][id as usize].is_empty() { Changes::default() } else { self.reterm(s, &[id], &[], first) }).collect();
        if let Some(e) = self.elems[id as usize].take() {
            self.claim(&e, false);
        }
        let (mut d, mut delta) = (0.0, FixedCounts::default());
        for (s, ch) in changes.into_iter().enumerate() {
            let old = std::mem::take(&mut self.rendered[s][id as usize]);
            self.reindex(s, id, &old, &[], &mut delta);
            d += self.apply(s, ch);
        }
        d + self.recount(delta)
    }

    /// Replace a live element ([`Scene::check_id`]): the cost change.
    pub fn change(&mut self, id: u32, e: Element) -> f64 {
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter().map(|s| {
            if layers[s].is_empty() && self.rendered[s][id as usize].is_empty() { Changes::default() } else { self.reterm(s, &[id], &[(e, layers[s].clone())], id) }
        }).collect();
        if let Some(old) = self.elems[id as usize].replace(e) {
            self.claim(&old, false);
        }
        self.claim(&e, true);
        let (mut d, mut delta) = (0.0, FixedCounts::default());
        for (s, (l, ch)) in layers.into_iter().zip(changes).enumerate() {
            let old = std::mem::take(&mut self.rendered[s][id as usize]);
            self.reindex(s, id, &old, &l, &mut delta);
            self.rendered[s][id as usize] = l;
            d += self.apply(s, ch);
        }
        d + self.recount(delta)
    }

    /// Put a move's new terms into scan `s`'s fit: the cost change.
    fn apply(&mut self, s: usize, terms: Changes) -> f64 {
        if terms.is_empty() {
            return 0.0;
        }
        let fit = &mut self.fits[s];
        for (o, t) in terms {
            match t {
                Some(t) => fit.terms.insert(o, t),
                None => fit.terms.remove(&o),
            };
        }
        let before = fit.cost;
        fit.cost = fit.total(&HashMap::new());
        fit.cost - before
    }

    /// Scan `s`'s data for every live element it sees ([`elem_data`]), each term with its owner.
    fn data(&self, s: usize) -> Vec<(Owner, ElemData)> {
        let ids: Vec<u32> = (0..self.elems.len() as u32).filter(|&i| self.elems[i as usize].is_some() && !self.rendered[s][i as usize].is_empty()).collect();
        let elems: Vec<Element> = ids.iter().map(|&i| self.elems[i as usize].unwrap()).collect();
        let rendered: Vec<Vec<(usize, Layer)>> = ids.iter().enumerate()
            .map(|(k, &i)| self.rendered[s][i as usize].iter().map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect()).collect();
        let owners: Vec<Owner> = ids.iter().zip(&elems).map(|(&i, e)| owner(i, e)).collect();
        let data = elem_data(&self.views[s], &rendered, &elems, &slots(&owners), &(0..ids.len()).collect::<Vec<_>>(), self.ref_epoch, &self.em, &self.sc, &self.ext);
        data.into_iter().map(|d| (owners[d.members[0]], d)).collect()
    }

    /// Scan `s` solved afresh for the elements as they are ([`solve_alignment`]), its alignment held there.
    fn solve(&self, s: usize) -> Fit {
        let (owners, data): (Vec<Owner>, Vec<ElemData>) = self.data(s).into_iter().unzip();
        let (mut held, terms) = solve_alignment(&data, &self.em);
        let fit_terms: BTreeMap<Owner, Term> = owners.into_iter().zip(terms).collect();
        held.reference = fit_terms.clone();
        let mut fit = Fit { terms: fit_terms, held, cost: 0.0 };
        fit.cost = fit.total(&HashMap::new());
        fit
    }

    /// Scan `s` with the live elements `without` taken out and the elements `with` (each with its layers in
    /// the scan) put in as ids `first`, `first + 1`, …: the new term of every owner whose term changes (none: it left
    /// the scan), the alignment held. A term reads only the cells its members are the front of, each with its layers at
    /// the cell's own surface class, and its class reads the same cells ([`term_classes`]). So the terms that change are
    /// those of the fronts of the cells the move touches (before it and after); and a group's term is its members'
    /// together, so every present member of a group one of those (or an element taken out) belongs to is rescored with
    /// it, and its pooled evidence never goes stale.
    fn reterm(&self, s: usize, without: &[u32], with: &[(Element, Vec<(usize, Layer)>)], first: u32) -> Changes {
        let (v, em, fit, index) = (&*self.views[s], &self.em, &self.fits[s], &self.cells[s]);
        let added = |i: u32| i >= first && ((i - first) as usize) < with.len();
        let layers = |i: u32| -> &[(usize, Layer)] { if added(i) { &with[(i - first) as usize].1 } else { &self.rendered[s][i as usize] } };
        let element = |i: u32| if added(i) { with[(i - first) as usize].0 } else { self.elems[i as usize].unwrap() };
        let present = |i: u32| !layers(i).is_empty() && (added(i) || !without.contains(&i));
        let fronting = |i: u32| layers(i).iter().filter(|(_, l)| l.enters()).map(|(c, _)| *c);
        let mut fresh: HashMap<usize, Vec<Entered>> = HashMap::new();
        for (k, (_, ls)) in with.iter().enumerate() {
            for (c, l) in ls.iter().filter(|(_, l)| l.enters()) {
                fresh.entry(*c).or_default().push(Entered::of(first + k as u32, l, em));
            }
        }
        // a cell's k-buffer, before the move and after it: its elements, nearest first, and how many have a front share
        let before = |c: usize| kept(index.get(&c).cloned().unwrap_or_default());
        let after = |c: usize| kept(index.get(&c).into_iter().flatten().filter(|x| !without.contains(&x.id) && !added(x.id))
            .chain(fresh.get(&c).into_iter().flatten()).copied().collect());
        let owner_of = |i: u32| owner(i, &element(i));
        // every present member of the groups of `ids` and of the elements taken out
        let whole = |ids: &mut BTreeSet<u32>| {
            let groups: BTreeSet<u32> = ids.iter().map(|&i| element(i)).chain(without.iter().map(|&i| self.elems[i as usize].unwrap())).filter_map(|e| e.group).collect();
            if groups.is_empty() {
                return;
            }
            let member = |e: &Element| e.group.is_some_and(|g| groups.contains(&g));
            ids.extend((0..self.elems.len() as u32).filter(|&i| self.elems[i as usize].as_ref().is_some_and(member)));
            ids.extend((0..with.len() as u32).filter(|&k| member(&with[k as usize].0)).map(|k| first + k));
            ids.retain(|&i| present(i));
        };
        // the terms with a front share in the cells the move touches, and the elements it adds
        let touched: HashSet<usize> = without.iter().flat_map(|&w| self.rendered[s][w as usize].iter()).filter(|(_, l)| l.enters()).map(|(c, _)| *c)
            .chain(fresh.keys().copied()).collect();
        let mut near: BTreeSet<u32> = (0..with.len() as u32).map(|k| first + k).collect();
        for &c in &touched {
            for (ids, n) in [before(c), after(c)] {
                near.extend(&ids[..n]);
            }
        }
        near.retain(|&i| present(i));
        whole(&mut near);
        // those terms' elements and every layer in the cells they front, numbered from 0 (the context after them)
        let mut cells: Vec<usize> = near.iter().flat_map(|&i| fronting(i)).collect();
        cells.sort_unstable();
        cells.dedup();
        let mut ids: Vec<u32> = near.iter().copied().collect();
        let mut seen: HashSet<u32> = near.iter().copied().collect();
        for &c in &cells {
            ids.extend(after(c).0.into_iter().filter(|&i| seen.insert(i)));
        }
        let cells: HashSet<usize> = cells.into_iter().collect();
        let r: Vec<Vec<(usize, Layer)>> = ids.iter().enumerate()
            .map(|(k, &i)| layers(i).iter().filter(|(c, _)| cells.contains(c)).map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect()).collect();
        let el: Vec<Element> = ids.iter().map(|&i| element(i)).collect();
        let owners: Vec<Owner> = ids.iter().map(|&i| owner_of(i)).collect();
        let data = elem_data(v, &r, &el, &slots(&owners), &(0..near.len()).collect::<Vec<_>>(), self.ref_epoch, em, &self.sc, &self.ext);
        // a held term's own nodes start where its reference's did, so an unchanged term is recomputed exactly
        let mut terms: Vec<(Owner, Option<Term>)> = data.par_iter().map(|d| {
            let o = owners[d.members[0]];
            (o, Some(d.term(fit.held.shift(&d.align), fit.held.reference.get(&o).map_or(Vector2::zeros(), |t| t.start), em)))
        }).collect();
        // the owners of the elements taken out or unseen now, unless a member is still scored
        let scored: HashSet<Owner> = terms.iter().map(|t| t.0).collect();
        let left = without.iter().map(|&i| owner(i, &self.elems[i as usize].unwrap())).chain((0..with.len() as u32).map(|k| owner_of(first + k)));
        terms.extend(left.collect::<BTreeSet<Owner>>().into_iter().filter(|o| !scored.contains(o)).map(|o| (o, None)));
        terms
    }

    /// What some elements predict in image `s` ([`Scene::check_scan`]) against the scene, per cell one of them fronts
    /// (its index in `es` is the prediction's `elem`), on the path the likelihood scores ([`front_echo`],
    /// [`cell_view`]): they are rendered for this, not added, and only the scene's layers in their cells enter their
    /// k-buffer, so their front cells and classes are those of the whole set. It reads the layers wholly in their cells
    /// ([`Layer::full`]): a partial layer's evidence is the cost's, its prediction too faint to report.
    pub fn predict(&self, es: &[Element], s: usize) -> Vec<CellPrediction> {
        let (v, em) = (&*self.views[s], &self.em);
        let full = |ls: Vec<(usize, Layer)>| ls.into_iter().filter(|(_, l)| l.full()).collect::<Vec<_>>();
        let mut per: Vec<Vec<(usize, Layer)>> = es.iter().enumerate().map(|(k, e)| full(render(v, e, k as u32, em))).collect();
        let cells: HashSet<usize> = per.iter().flatten().map(|&(c, _)| c).collect();
        let mut els = es.to_vec();
        let n = self.elems.len() as u32;
        let mut owners: Vec<Owner> = es.iter().enumerate().map(|(k, e)| owner(n + k as u32, e)).collect();
        for (i, el) in self.elems.iter().enumerate() {
            let Some(el) = el.as_ref() else { continue };
            let k = per.len() as u32;
            let l: Vec<(usize, Layer)> = self.rendered[s][i].iter().filter(|(c, l)| cells.contains(c) && l.full()).map(|&(c, ly)| (c, Layer { elem: k, ..ly })).collect();
            if !l.is_empty() {
                els.push(*el);
                per.push(l);
                owners.push(owner(i as u32, el));
            }
        }
        let buf = kbuffer(&per);
        let other = v.epoch != self.ref_epoch;
        let cls = term_classes(v, &buf, &slots(&owners), em);
        let mut out: Vec<CellPrediction> = buf.iter().filter(|(&cell, ls)| (ls[0].elem as usize) < es.len() && v.cell[cell] != Cell::NoPulse).map(|(&cell, ls)| {
            let ly = ls[0];
            let k = ly.elem as usize;
            let cv = cell_view(v, cell, ls.clone(), em, &self.sc, &self.ext);
            let pulse = &cv.pulses[0].1;
            let r = pulse.read(0.0, 0.0, em);
            let fe = front_echo(&r.rx, cls[k], es[k].front_shift_sd(em), other, em);
            let (depth, sd) = depth_sd(fe.as_ref());
            // the element's occluder: none where it lies past the beam's end
            let at = pulse.occluders.iter().position(|o| matches!(o.surface, Surface::Layer(l) if l.elem == ly.elem));
            CellPrediction {
                cell, elem: ly.elem, fill: ly.f, recorded: fe.map_or(0.0, |f| f.s), arrives: fe.map_or(0.0, |f| (-pulse.occluders[f.first].tau).exp()), surface: cls[k], depth, sd,
                refl: fe.map_or(f64::NAN, |f| if f.db[1] > -200.0 { f.db[1] } else { f64::NAN }), cos: ly.cos, obs: cv.obs,
                obs_refl: pulse.refls.first().map_or(f64::NAN, |&x| x as f64),
                outcome: Outcome::of(cv.obs, depth, sd), read: pulse.echoes.iter().any(|&r| Outcome::of(r as f64, depth, sd) == Outcome::On),
                merge: at.map_or(0.0, |i| pulse.occluders[i].merge.share), weight: cv.wt, footprint: em.footprint(ly.t), across: ly.x, along: ly.s,
                returns: r.rx.expected(pulse.cap).iter().sum(), beam: at.map_or(f64::NAN, |i| r.rx.members[i].beam), owned: at.map_or(0.0, |i| r.owned(i)),
            }
        }).collect();
        out.sort_by_key(|p| p.cell);
        out
    }

    /// An element's counts against the scene, per image, from its predictions ([`Scene::predict`]). `shift` (m, per
    /// view; empty for none) is subtracted from the observed echoes first.
    pub fn counts(&self, e: Element, shift: &[f64]) -> Vec<ElementCounts> {
        let phi = self.em.discount;
        (0..self.views.len()).into_par_iter().map(|s| {
            let mut out = ElementCounts { scan: s as u32, ..ElementCounts::default() };
            let sh = shift.get(s).copied().unwrap_or(0.0);
            // the mismatch's from the element drawn sharp (its own mismatch left out of the depths it is measured
            // from), the opacity's from the element as given
            for p in self.predict(&[Element { mismatch: 0.0, ..e }], s) {
                let r = p.obs - sh - p.depth;
                if p.fill > MOSTLY && Outcome::of(p.obs - sh, p.depth, p.sd) == Outcome::On {
                    out.mismatch_ss += phi * p.weight * (r * r - p.sd * p.sd).max(0.0);
                    out.mismatch_w += phi * p.weight;
                    out.mismatch_cells += 1;
                }
            }
            for p in self.predict(&[e], s).into_iter().filter(|p| p.fill > 0.9) {
                match p.outcome {
                    Outcome::On => out.on += phi * p.weight,
                    Outcome::Empty => out.empty += phi * p.weight,
                    Outcome::Behind => out.behind += phi * p.weight,
                    _ => {}
                }
            }
            out
        }).collect()
    }

    /// How well the scans pin some live elements' geometry down ([`Scene::check_id`]; one element, or several
    /// jointly): per scan, the cost's slope and its expected (Fisher) information in their parameters, each element's
    /// in turn, the rest of the scene held. A stick's parameters are `ln r`, its axis moved across itself (two
    /// directions perpendicular to it, m) and tilted about its middle (the same two directions, rad at its ends'
    /// half-length); a patch's are `ln r` and its centre moved (x, y, z, m): n is 5 a stick, 4 a patch. The frame is
    /// the element's own, the same in every scan, so scans' information adds. A scan that doesn't see them gives zeros.
    ///
    /// **One pass per cell** (research 130 slice-derivatives): the ray likelihood read with each element's layer in
    /// [`Geometry`] numbers, forward derivatives in its parameters (through its rendering: the fill, the depth over
    /// the footprint, the far side) and in the cell's own nodes (its shift along the beam and its reflectance level),
    /// at the alignment held and each term's own nodes at their mode. From it:
    /// - **`slope`** (n): the cost's slope, over every cell an element's layer is in (its own, and those it lies
    ///   behind another's front in), each at its term's nodes: the slope of the fit with the nuisances at their modes
    ///   (an envelope: their own slopes vanish there), the Laplace steps' log-determinants held.
    /// - **`expected`** (n × n, row-major; research 130 step 3a): what the elements' own cells would tell on average.
    ///   Each cell is read as an echo or none (a Bernoulli in the chance p the element's echo comes back) and, given one,
    ///   its depth μ and reflectance ρ (at its mean fill: [`Return::reflectance`]; [`cell_information`]):
    ///   `Σ_c w_c [∂p ∂pᵀ / (p (1 − p)) + p (k_d ∂μ ∂μᵀ / sd² +
    ///   k_r ∂ρ ∂ρᵀ / s_ρ²)]`, w_c the cell weight and sd the depth's own. Each observable also loads the latents the
    ///   cells share by its slope in the cell's own nodes (the depth on the shift, so on the alignment's hats and the
    ///   stem front shift; the reflectance and the chance of an echo on the level), and they are marginalised out
    ///   under their priors (a Schur complement, [`Shared`]). It is positive semi-definite, reads a direction the cells
    ///   don't see as 0, and leaves out what other elements' cells say.
    ///
    /// The observed curvature is not given: the fill's curvature is unbounded at the beam's edge, and the expected
    /// information is the calibrated one (research 130 step 3b; optimisation.md). Slope and expected information
    /// together are a Fisher-scoring step.
    ///
    /// **Several elements** (research 130 slice 4: a pair's joint information): each element's cells are read with
    /// the others in place, a cell's observables in every element whose layer is in it, and the latents are
    /// marginalised once over all of them ([`Shared`]): each element's own stem front shift and reflectance level, and
    /// the alignment's nodes, which neighbours share. So the alignment, which moves neighbours' depths together,
    /// cancels in their depth difference, where each element's own information charges it in full. Far apart (no node
    /// or cell in common) the information is the elements' own, side by side. Members of one group (a chain) share
    /// one term's nodes ([`Owner`]), as their cost does when `ids` are the whole group; a chain's knots are a Jacobian
    /// away from its sticks' parameters (research 130 slice 6).
    ///
    /// **What it means under the evidence discount** (research 130 step 3b). Each cell's log-likelihood counts
    /// `w_c = 1 / (φ m)`, so the cost is a quasi-likelihood and E is its curvature: an sd read from E⁻¹ is honest only
    /// where the cells' errors are correlated as φ m declares. In general the estimate's variance is the sandwich
    /// `E⁻¹ J E⁻¹`, J the gradient's actual variance (the inverse of Godambe's information `E J⁻¹ E`), and model
    /// comparison by cost differences needs the composite-likelihood penalty `tr(J E⁻¹)`, not a parameter count. φ
    /// enters E once, and J twice: beside E comes the **gradient's variance were the cells independent**
    /// (`gradient_variance`: the same sums at `w_c²`, its latents' share taken out as E's is, their priors' included),
    /// so `E⁻¹ J E⁻¹` is the variance with no correlation between cells, a lower bound, and E⁻¹ the variance at the
    /// declared φ. On the stand's true sticks (step3b.md), fits from the truth fall within 1.96 sd of E⁻¹ along the
    /// sloppiest direction 95 % of the time at 3 footprints across or more; the sandwich's sds are 1.5–2.6 times too
    /// small (the cells are correlated); and below 1.5 footprints across the cost has minima far from the truth that a
    /// local curvature can't see, so E's sds don't hold there.
    ///
    /// **ln r is the space radius is unbiased in:** the stand's fitted ln r sits −0.003 from the truth (median, R ≥
    /// 1 cm); r's own mean is `exp(ln r + ½ sd²)`, so an average of radii should be taken in ln r.
    pub fn information(&self, ids: &[u32]) -> Vec<Information> {
        let es: Vec<Element> = ids.iter().map(|&id| self.elems[id as usize].expect("a live element")).collect();
        let dims: Vec<usize> = es.iter().map(|e| if let Form::Stick { .. } = e.form { 5 } else { 4 }).collect();
        let offset: Vec<usize> = dims.iter().scan(0, |o, d| Some(std::mem::replace(o, *o + d))).collect();
        let n: usize = dims.iter().sum();
        let em = &self.em;
        (0..self.views.len()).map(|s| {
            let v = &*self.views[s];
            let refl = em.refl_noise > 0.0 && !v.echo_refl.is_empty();
            let other = v.epoch != self.ref_epoch;
            let shared = Shared::new(n, &es, em);
            // per cell: its slope's part, and the element's echo's observables with their loadings and weights
            type Rows = Vec<(Vec<f64>, Vec<(usize, f64)>, f64, f64)>;
            let parts: Vec<(Vec<f64>, Rows)> = self.cells_of(s, ids).into_par_iter().map(|(cell, ls, owners)| {
                let nodes: Vec<(usize, [Geometry; 2])> = owners.iter().map(|&(i, term, [shift, level])| {
                    (i, [Geometry::var(shift, SHIFT), if term.own_sd[1] > 0.0 { Geometry::var(level, LEVEL) } else { Geometry::c(level) }])
                }).collect();
                let mut slope = vec![0.0; n];
                let mut x = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                let (mut echo, mut shares) = (None, Vec::new());
                for (j, ly) in ls.iter().enumerate() {
                    let Some(k) = ids.iter().position(|&i| i == ly.elem) else { continue };
                    let th: Vec<Geometry> = (0..dims[k]).map(|i| Geometry::var(0.0, i)).collect();
                    let r = self.read_cell(s, cell, &ls, j, &es[k], &th, &nodes);
                    for i in 0..dims[k] {
                        slope[offset[k] + i] = r.cost.g[i];
                    }
                    if let Some((q, sd)) = r.echo {
                        for (xo, q) in x.iter_mut().zip(q) {
                            xo[offset[k]..offset[k] + dims[k]].copy_from_slice(&q.g.as_slice()[..dims[k]]);
                        }
                        echo = Some((q, sd, r.weight));
                    }
                    shares = r.shares;
                }
                // the echo's observables, loaded on each owner's nodes at its front share of the cell's weight
                let mut rows = Rows::new();
                if let Some(([p, mu, rho], sd, w)) = echo {
                    let info = cell_information(em, p.v(), sd * sd, ls[0].f);
                    for &(i, term, _) in &owners {
                        let Some(k) = ids.iter().position(|&id| id == ls[i].elem) else { continue };
                        let align = em.align_sd(term.surface, other);
                        let load = |d: &Geometry| shared.load(k, align, d.g[SHIFT], d.g[LEVEL], em.refl_level_sd);
                        let w = w * shares[i];
                        rows.push((x[0].clone(), load(&p), info[0], w));
                        rows.push((x[1].clone(), load(&mu), info[1], w));
                        if refl && rho.v() > -200.0 {
                            rows.push((x[2].clone(), load(&rho), info[2], w));
                        }
                    }
                }
                (slope, rows)
            }).collect();
            let (mut shared, mut slope) = (shared, vec![0.0; n]);
            for (g, rows) in parts {
                slope.iter_mut().zip(g).for_each(|(a, b)| *a += b);
                for (x, load, info, w) in rows {
                    shared.add(&x, &load, info, w);
                }
            }
            let [expected, gradient_variance] = shared.finish();
            Information { slope, expected, gradient_variance }
        }).collect()
    }

    /// The cells of scan `s` that any of the live elements `ids` fronts and that hold a pulse, in index order, each with
    /// its k-buffer (nearest first, ties by id, as the cost's), the term of its front element and that term's own
    /// nodes as the fit holds them: the shift along the beam (the held alignment's and the stem front shift's) and the
    /// reflectance level.
    fn cells_of(&self, s: usize, ids: &[u32]) -> Vec<(usize, Vec<Layer>, Vec<(usize, &Term, [f64; 2])>)> {
        let (v, fit, rendered) = (&*self.views[s], &self.fits[s], &self.rendered[s]);
        let mut cells: Vec<usize> = ids.iter().flat_map(|&id| rendered[id as usize].iter().filter(|(_, l)| l.enters()).map(|(c, _)| *c)).collect();
        cells.sort_unstable();
        cells.dedup();
        cells.into_iter().filter(|&c| v.cell[c] != Cell::NoPulse).filter_map(|cell| {
            let mut ls: Vec<Layer> = self.cells[s][&cell].iter().map(|x| x.id).collect::<BTreeSet<u32>>().into_iter()
                .map(|id| rendered[id as usize][rendered[id as usize].binary_search_by_key(&cell, |x| x.0).expect("an indexed layer")].1).collect();
            keep(&mut ls, |l| l.t, |l| l.full());
            let owners: Vec<(usize, &Term, [f64; 2])> = front_shares(&ls).iter().enumerate().filter(|(_, &w)| w > 0.0).filter_map(|(i, _)| {
                let id = ls[i].elem;
                let term = fit.terms.get(&owner(id, &self.elems[id as usize]?))?;
                Some((i, term, [fit.held.shift(&term.align) + term.own_sd[0] * term.own[0], term.own_sd[1] * term.own[1]]))
            }).collect();
            (!owners.is_empty()).then_some((cell, ls, owners))
        }).collect()
    }

    /// Cell `cell` of scan `s` as [`Scene::information`] reads it ([`CellReading`]), its k-buffer `ls` with the `j`th
    /// layer element `e`'s moved by `th` in its parameters ([`placed`]), each layer with a front share read at its term's
    /// own nodes (`owners`: the layer's index and its shift along the beam and reflectance level), all in the numbers
    /// `P`.
    #[allow(clippy::too_many_arguments)]
    fn read_cell<P: Real>(&self, s: usize, cell: usize, ls: &[Layer], j: usize, e: &Element, th: &[P], owners: &[(usize, [P; 2])]) -> CellReading<P> {
        let (v, em) = (&*self.views[s], &self.em);
        let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
        let moved = layer(&placed(e, &v.origin, Some(th)), &w, e, ls[j].elem, em, v.step()).expect("a layer it fronts");
        let layers: Vec<Layer<P>> = ls.iter().enumerate().map(|(i, l)| if i == j { moved } else { l.lift() }).collect();
        let shares = front_shares(&layers);
        let cv = cell_view(v, cell, layers, em, &self.sc, &self.ext);
        let base = cv.pulses[0].1.empty().read(P::c(0.0), P::c(0.0), em).ll;
        let mut cost = P::c(0.0);
        for &(i, [shift, level]) in owners {
            cost += -((mixture_ll(&cv.pulses, shift, level, em) - base) * cv.wt) * shares[i];
        }
        // the element's echo, at the front owner's nodes: its chance over the mixture, its depth and reflectance with
        // every layer in
        let [shift, level] = owners[0].1;
        let front = |p: &Pulse<P>| p.receiver(shift, level, em).front().map(|f| ((-p.occluders[f.first].tau).exp() * f.s, f));
        let chance = cv.pulses.iter().map(|(w, p)| front(p).map_or(P::c(0.0), |(c, _)| c * *w)).sum::<P>();
        let echo = front(&cv.pulses[0].1).map(|(_, f)| ([chance, f.mu, f.reflectance()], f.sd.v()));
        CellReading { cost, weight: cv.wt.v(), shares: shares.iter().map(|s| s.v()).collect(), echo }
    }

    /// What hypothesis `q` is expected to cost against hypothesis `p` were `p` true, per scan (nats): `E_p[cost(q) −
    /// cost(p)]`, the divergence of `q`'s predictions from `p`'s (each against the scene, [`Scene::predict`]), with
    /// [`Scene::information`]'s reading of a cell ([`cell_information`]): its echo or none, a Bernoulli, exact; given
    /// an echo in both, its depth and reflectance at their own sds, the latents they load marginalised as
    /// `information` does (`p`'s elements' nodes). A cell only one of them fronts reads the other's chance of an echo
    /// as 0 (1e-3). Beside an observed cost difference it separates data that are balanced (observed ≈ 0, expected
    /// large: a misfit) from data that are silent (both ≈ 0).
    pub fn divergence(&self, p: &[Element], q: &[Element]) -> Vec<f64> {
        let em = &self.em;
        (0..self.views.len()).into_par_iter().map(|s| {
            let at = |es: &[Element]| -> BTreeMap<usize, CellPrediction> { self.predict(es, s).into_iter().map(|c| (c.cell, c)).collect() };
            let (pp, qq) = (at(p), at(q));
            let refl = em.refl_noise > 0.0 && !self.views[s].echo_refl.is_empty();
            let other = self.views[s].epoch != self.ref_epoch;
            let mut shared = Shared::new(1, p, em);
            let mut bernoulli = 0.0;
            let echo = |c: Option<&CellPrediction>| c.map_or(0.0, |c| c.arrives * c.recorded).clamp(1e-3, 1.0 - 1e-3);
            let cells: BTreeSet<usize> = pp.keys().chain(qq.keys()).copied().collect();
            for cell in cells {
                let (a, b) = (pp.get(&cell), qq.get(&cell));
                let (ea, eb) = (echo(a), echo(b));
                let w = a.or(b).map_or(0.0, |c| c.weight);
                bernoulli += w * (ea * (ea / eb).ln() + (1.0 - ea) * ((1.0 - ea) / (1.0 - eb)).ln());
                let (Some(a), Some(b)) = (a, b) else { continue };
                let k = a.elem as usize;
                let (align, front) = (em.align_sd(a.surface, other), p[k].front_shift_sd(em));
                let [_, depth, refl_info] = cell_information(em, ea, a.sd * a.sd - align * align - front * front, a.fill);
                let load = shared.load(k, align, 1.0, 0.0, em.refl_level_sd);
                shared.add(&[b.depth - a.depth], &load, depth, w);
                if refl && (b.refl - a.refl).is_finite() {
                    let load = shared.load(k, align, 0.0, 1.0, em.refl_level_sd);
                    shared.add(&[b.refl - a.refl], &load, refl_info, w);
                }
            }
            bernoulli + 0.5 * shared.finish()[0][0]
        }).collect()
    }

    /// Each hypothesis's cost to add alone (one or more elements), per scan and then, when the scene holds a foliage
    /// field, the field's part (the change in [`Scene::field_part`]): (hypotheses × scans, or scans + 1), in parallel,
    /// the scene unchanged, its foliage field and alignment held. A row sums to adding it then removing it. A hypothesis's
    /// elements of one group are one term ([`Owner`]: a chain), joined by the scene's members of that group, which are
    /// rescored with it; its other elements are each alone.
    pub fn costs_to_add(&self, hyps: &[Vec<Element>]) -> Vec<Vec<f64>> {
        self.costs_replacing(&hyps.iter().map(|h| (h.clone(), Vec::new())).collect::<Vec<_>>())
    }

    /// Each live element's cost to remove ([`Scene::check_id`]), per scan and then, with a field, the field's part
    /// (as [`Scene::costs_to_add`]), the scene unchanged. A row sums to removing it then adding it back.
    pub fn costs_to_remove(&self, ids: &[u32]) -> Vec<Vec<f64>> {
        self.costs_replacing(&ids.iter().map(|&id| (Vec::new(), vec![id])).collect::<Vec<_>>())
    }

    /// Per (hypothesis, live elements): the cost of the scene with the elements taken out and the hypothesis put
    /// in, against the scene, per scan and then the field's part.
    fn costs_replacing(&self, items: &[(Vec<Element>, Vec<u32>)]) -> Vec<Vec<f64>> {
        let first = self.elems.len() as u32;
        items.par_iter().map(|(h, without)| {
            let layers: Vec<Vec<Vec<(usize, Layer)>>> = self.views.iter().map(|v| h.iter().map(|e| render(v, e, 0, &self.em)).collect()).collect();
            let mut out: Vec<f64> = layers.iter().enumerate().map(|(s, ls)| {
                if ls.iter().all(Vec::is_empty) && without.iter().all(|&w| self.rendered[s][w as usize].is_empty()) {
                    return 0.0;
                }
                let with: Vec<(Element, Vec<(usize, Layer)>)> = h.iter().copied().zip(ls.iter().cloned()).collect();
                let over: HashMap<Owner, Option<Term>> = self.reterm(s, without, &with, first).into_iter().collect();
                self.fits[s].total(&over) - self.fits[s].cost
            }).collect();
            if let Some(f) = &self.field {
                let mut delta = FixedCounts::default();
                for (s, ls) in layers.iter().enumerate() {
                    self.field_delta(s, without, ls, &mut delta);
                }
                out.push(f.change(&delta, &self.ext));
            }
            out
        }).collect()
    }

    /// Which hypotheses interact in each scan, so that their costs scored together differ from the sum of their costs
    /// alone: (i, j, scan, `cells`, `nodes`) for i < j, `cells` when they enter a cell in common or cells in which one
    /// scene term has a front share ([`Owner`]: an element, or any member of a group), or join one group, `nodes` when they share an
    /// alignment node. With the alignment held, a shared node only couples them through its rank-k correction
    /// (small); shared cells are evidence they share. A hypothesis is one move, so its own elements (a chain's joints)
    /// never interact with each other.
    pub fn interactions(&self, hyps: &[Vec<Element>]) -> Vec<(u32, u32, u32, bool, bool)> {
        let mut out: BTreeMap<(u32, u32, u32), (bool, bool)> = BTreeMap::new();
        for (s, v) in self.views.iter().enumerate() {
            // per key (a cell, or a term its hypotheses front or join), the hypotheses that have it
            let mut cells: HashMap<(bool, u64), BTreeSet<u32>> = HashMap::new();
            let mut nodes: HashMap<[i64; 3], BTreeSet<u32>> = HashMap::new();
            let term = |o: Owner| match o {
                Owner::Alone(i) => i as u64,
                Owner::Group(g) => 1 << 32 | g as u64,
            };
            let rendered: Vec<Vec<Vec<(usize, Layer)>>> = hyps.par_iter().map(|h| h.iter().map(|e| render(v, e, 0, &self.em)).collect()).collect();
            for (k, (h, ls)) in hyps.iter().zip(&rendered).enumerate() {
                for (e, l) in h.iter().zip(ls).filter(|(_, l)| !l.is_empty()) {
                    for (key, _) in hats(&e.mid(), ALIGN_SPACING) {
                        nodes.entry(key).or_default().insert(k as u32);
                    }
                    if let Some(g) = e.group {
                        cells.entry((true, term(Owner::Group(g)))).or_default().insert(k as u32);
                    }
                    for (c, _) in l.iter().filter(|(_, l)| l.enters()) {
                        cells.entry((false, *c as u64)).or_default().insert(k as u32);
                        let (ids, n) = kept(self.cells[s].get(c).cloned().unwrap_or_default());
                        for &o in &ids[..n] {
                            cells.entry((true, term(owner(o, &self.elems[o as usize].unwrap())))).or_default().insert(k as u32);
                        }
                    }
                }
            }
            let mut mark = |set: &BTreeSet<u32>, cell: bool| {
                for (n, &i) in set.iter().enumerate() {
                    for &j in set.iter().skip(n + 1) {
                        let f = out.entry((i, j, s as u32)).or_default();
                        if cell { f.0 = true } else { f.1 = true }
                    }
                }
            };
            cells.values().for_each(|set| mark(set, true));
            nodes.values().for_each(|set| mark(set, false));
        }
        out.into_iter().map(|((i, j, s), (c, n))| (i, j, s, c, n)).collect()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    /// The nested solver the joint one replaced (3dtrees research 130 optimisation.md), kept as its referee: Newton on
    /// the alignment's nodes with each term's own nodes re-solved at every step.
    fn solve_alignment_nested(data: &[ElemData], em: &Settings) -> (Held, Vec<Term>) {
        let index = alignment_nodes(data);
        let n = index.len();
        let terms_at = |z: &[f64], from: &[Term]| -> Vec<Term> {
            data.par_iter().enumerate().map(|(i, el)| el.term(el.align.iter().map(|(k, w)| w * z[index[k]]).sum(), from.get(i).map_or(Vector2::zeros(), |t| t.own), em)).collect()
        };
        let objective = |z: &[f64], t: &[Term]| 0.5 * z.iter().map(|x| x * x).sum::<f64>() + t.iter().map(|t| t.fit).sum::<f64>();
        let system = |z: &[f64], t: &[Term]| alignment_system(&index, z, t.iter().map(|t| (&t.align[..], t.slope, t.curv)));
        let mut z = vec![0.0; n];
        let mut terms = terms_at(&z, &[]);
        let (mut grad, mut hess) = system(&z, &terms);
        for _ in 0..MAX_NEWTON {
            let Some(ch) = (n > 0).then(|| hess.clone().cholesky()).flatten() else { break };
            let mut step = ch.solve(&grad);
            step /= step.amax().max(3.0) / 3.0;
            let f_now = objective(&z, &terms);
            let (mut t, mut fall) = (1.0, None);
            while t >= 1e-3 {
                let zn: Vec<f64> = z.iter().zip(step.iter()).map(|(a, b)| a - t * b).collect();
                let tn = terms_at(&zn, &terms);
                let f_new = objective(&zn, &tn);
                if f_new <= f_now {
                    fall = Some(f_now - f_new);
                    (z, terms) = (zn, tn);
                    break;
                }
                t *= 0.5;
            }
            let Some(fall) = fall else { break };
            (grad, hess) = system(&z, &terms);
            if fall < 1e-3 {
                break;
            }
        }
        (held(index, z, grad, hess), terms)
    }

    /// The joint arrowhead Newton finds the nested solver's alignment and terms: on the pole alone and with a stick
    /// beside it (sharing its node), one scan and two.
    #[test]
    fn the_joint_newton_matches_the_nested_one() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let scene = |els: &[Element], n: usize| Scene::new(vec![v.clone(); n], els, em, sc, None, 0);
        for (els, n) in [(vec![stick(0.0, 4.0, 0.03)], 1), (vec![stick(0.0, 4.0, 0.03), stick(0.15, 4.5, 0.02)], 2)] {
            let c = scene(&els, n);
            for s in 0..n {
                let vv = &*c.views[s];
                let ids: Vec<u32> = (0..els.len() as u32).collect();
                let rendered: Vec<Vec<(usize, Layer)>> = ids.iter().map(|&i| c.rendered[s][i as usize].clone()).collect();
                let slot = slots(&ids.iter().map(|&i| owner(i, &els[i as usize])).collect::<Vec<_>>());
                let data = elem_data(vv, &rendered, &els, &slot, &(0..els.len()).collect::<Vec<_>>(), 0, &em, &sc, &c.ext);
                let ((hj, tj), (hn, tn)) = (solve_alignment(&data, &em), solve_alignment_nested(&data, &em));
                let total = |h: &Held, t: &[Term]| t.iter().map(Term::cost).sum::<f64>() + 0.5 * h.z.iter().map(|x| x * x).sum::<f64>() + 0.5 * h.logdet - 0.5 * h.gm;
                assert!((total(&hj, &tj) - total(&hn, &tn)).abs() < 1e-4, "{} vs {}", total(&hj, &tj), total(&hn, &tn));
                for (a, b) in tj.iter().zip(&tn) {
                    assert!((a.cost() - b.cost()).abs() < 1e-4 && (a.own[1] - b.own[1]).abs() < 1e-3, "{:?} vs {:?}", (a.cost(), a.own), (b.cost(), b.own));
                }
                assert!(hj.z.iter().zip(&hn.z).all(|(a, b)| (a - b).abs() < 1e-3), "{:?} vs {:?}", hj.z, hn.z);
            }
        }
    }
    use crate::shots::reconstruct::dir;
    use crate::scene::range_image::RangeImageParams;
    
    /// The test scan's receiver: returns closer than this merge (m).
    const RESOLUTION: f64 = 0.8;

    /// A scan of a vertical pole (R = 3 cm, 4 m away, centred on the y axis) in front of a wall at 8 m (see
    /// [`pole_scan`]).
    fn pole_image() -> RangeImage {
        pole_scan(0.03, 4.0, 4.0)
    }

    fn thin_pole_view(r_pole: f64, y_pole: f64) -> Arc<RangeImage> {
        Arc::new(pole_scan(r_pole, y_pole, y_pole))
    }

    /// A scan of a vertical pole (radius `r`, `y` m away on the y axis) in front of a wall `gap` m behind it, with the
    /// measured beam (research 130 slice 11): a uniform disc of the footprint's diameter, sampled here across the
    /// pole by 401 lines (its 1-D marginal, a semicircle), each part's energy its cosine (Lambertian; to the default
    /// incidence power for its reflectance and detection). The pole's echo sits at its parts' energy-weighted depth and is recorded where it clears the detection floor (wood at the
    /// default reflectance, 2 dB spread); the wall (−20 dB, face-on) takes the rest of the beam. Returns closer than the
    /// receiver's resolution (0.8 m) merge into one echo at the measured pull behind the pole's ([`Settings::pull`]); otherwise the first is the
    /// pole's if recorded, else the wall's. Each echo's reflectance is its energy share's (merged: summed) with the
    /// default per-echo noise.
    fn pole_scan(r_pole: f64, y_pole: f64, gap: f64) -> RangeImage {
        pole_scan_of(r_pole, y_pole, gap, Settings::default().refl_wood)
    }

    /// [`pole_scan`] with the pole's wood reflecting `wood` dB.
    fn pole_scan_of(r_pole: f64, y_pole: f64, gap: f64, wood: f64) -> RangeImage {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let em = Settings::default();
        let (mut t, mut d, mut r, mut refl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let u = |s: u64| ((s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        for l in 0..200usize {
            for k in 0..400usize {
                let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
                let w = dir(zen, az);
                let h = (w[0] * w[0] + w[1] * w[1]).sqrt();
                let miss = y_pole * w[0] / h;
                let tm = y_pole * w[1] / (h * h);
                let a = 0.5 * em.footprint(tm);
                let (mut f, mut ec, mut ek, mut ez) = (0.0, 0.0, 0.0, 0.0);
                for q in 0..401 {
                    let z = -1.0 + 2.0 * (q as f64 + 0.5) / 401.0;
                    let wt = (1.0 - z * z).sqrt() * 2.0 / 401.0 * 2.0 / std::f64::consts::PI;
                    let x = miss + z * a;
                    if x.abs() < r_pole {
                        let c = (1.0 - (x / r_pole).powi(2)).sqrt();
                        f += wt;
                        ec += wt * c;
                        ek += wt * c.powf(em.refl_incidence);
                        ez += wt * c * (tm - r_pole * c / h);
                    }
                }
                let seed = (l * 400 + k) as u64 * 2 + 1;
                let n = 2.0 * (-2.0 * u(seed).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919)).cos();
                let n_refl = em.refl_noise * (-2.0 * u(seed * 31).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919 * 31)).cos();
                let kept = ec > 0.0 && wood + 10.0 * ek.max(1e-12).log10() + n > em.floor(tm);
                let wall = (y_pole + gap) / w[1];
                let (ep, ew) = (ek * 10f64.powf(wood / 10.0), (1.0 - f) * 0.01);
                let (range, db) = if !kept {
                    (wall, 10.0 * ew.max(1e-12).log10())
                } else if wall - ez / ec < RESOLUTION {
                    (ez / ec + em.pull(wall - ez / ec, 10.0 * (ep / ew.max(1e-300)).log10()), 10.0 * (ep + ew).log10())
                } else {
                    (ez / ec, 10.0 * ep.log10())
                };
                refl.push((db + n_refl) as f32);
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = RangeImage::from_first_echoes(&t, &d, &r, id, [0.0; 3], 0, &RangeImageParams::default());
        v.echo_refl = refl;
        v
    }

    fn pole_view() -> Arc<RangeImage> {
        Arc::new(pole_image())
    }

    /// A scene's cost and rows.
    fn cost(v: &Arc<RangeImage>, els: &[Element], em: &Settings, sc: &Background) -> (f64, Vec<ElementScan>) {
        let s = Scene::new(vec![v.clone()], els, *em, *sc, None, 0);
        (s.cost(), s.rows())
    }

    fn stick(x: f64, y: f64, r: f64) -> Element {
        Element { form: Form::Stick { a: [x, y, -0.2], b: [x, y, 0.2] }, class: Class::Wood, r, opacity: 1.0, mismatch: 0.002, group: None }
    }

    /// The true pole has negative cost; a stick of the wrong radius, shifted, or porous has more; a stick where
    /// the pulses went through to the wall costs; a scene's move agrees with the difference of two
    /// energies, and rows sum to the total.
    #[test]
    fn the_true_pole_wins() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let e = |s: Element| cost(&v, &[s], &em, &sc).0;
        let truth = e(stick(0.0, 4.0, 0.03));
        assert!(truth < -3.0, "{truth}");
        for wrong in [stick(0.0, 4.0, 0.021), stick(0.0, 4.0, 0.015), stick(0.0, 4.0, 0.06), stick(0.03, 4.0, 0.03), stick(0.0, 3.98, 0.03),
                      Element { opacity: 0.3, ..stick(0.0, 4.0, 0.03) }] {
            assert!(e(wrong) > truth + 1.0, "{} vs {truth}", e(wrong));
        }
        assert!(e(stick(0.2, 6.0, 0.03)) > 1.0);
        let pair = [stick(0.0, 4.0, 0.03), stick(0.2, 6.0, 0.03)];
        let both = cost(&v, &pair, &em, &sc);
        let delta: f64 = Scene::new(vec![v.clone()], &pair[..1], em, sc, None, 0).costs_to_add(&[vec![pair[1]]])[0].iter().sum();
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let parts: f64 = Scene::new(vec![v.clone()], &pair, em, sc, None, 0).alignment_parts().iter().sum();
        let sum: f64 = both.1.iter().map(|r| r.cost).sum::<f64>() + parts;
        assert!((sum - both.0).abs() < 1e-6);
    }

    /// A stick barely wider than the footprint, under the measured (disc) beam: the true radius beats a thinner and a
    /// thicker one (research 130 slice 11: a Gaussian beam's tails expected edge echoes the disc doesn't return, so
    /// a thinner stick won).
    #[test]
    fn a_thin_stick_under_the_disc_beam_beats_thinner_and_thicker() {
        let (em, sc) = (Settings::default(), Background::default());
        let v = thin_pole_view(0.006, 6.0);
        let e = |r: f64| cost(&v, &[stick(0.0, 6.0, r)], &em, &sc).0;
        let truth = e(0.006);
        for r in [0.0042, 0.0084] {
            assert!(e(r) > truth + 0.5, "r {r}: {} vs {truth}", e(r));
        }
    }

    /// A pole 0.5 m in front of a wall, closer than the pulse's resolution: its edge echoes merge with the wall's
    /// into one at the measured pull behind the pole's. With the wall in the scene (the front of a ball 2 m in radius), the
    /// true radius beats a thinner and a thicker one.
    #[test]
    fn a_stick_just_before_a_wall_merges_its_edges() {
        let (em, sc) = (Settings::default(), Background::default());
        let v = Arc::new(pole_scan(0.02, 4.0, 0.5));
        let wall = Element { form: Form::Patch { centre: [0.0, 6.5, 0.0] }, class: Class::Leaf, r: 2.0, opacity: 1.0, mismatch: 0.002, group: None };
        let e = |r: f64| cost(&v, &[wall, stick(0.0, 4.0, r)], &em, &sc).0;
        let truth = e(0.02);
        for r in [0.014, 0.028] {
            assert!(e(r) > truth + 0.5, "r {r}: {} vs {truth}", e(r));
        }
    }

    /// The foliage field is a residual: the empty scene's field holds the pole's echoes (its voxel is dense), and
    /// promoting the pole takes them out of the field. Promotion and demotion are exact: the cost after a promotion
    /// is a fresh scene's, its change is that cost, the pole's rows hold all of it, and demotion restores the empty
    /// scene.
    #[test]
    fn the_field_is_a_residual_and_its_moves_are_exact() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let spec = Some(FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, margin: 0.0 });
        let pole = stick(0.0, 4.0, 0.03);
        let sigma = |c: &Scene| c.foliage().1.iter().find(|x| x.0 == [5, 10, 3]).map_or(f64::NAN, |x| x.1);
        let mut c = Scene::new(vec![v.clone()], &[], em, sc, spec, 0);
        assert_eq!(c.cost(), 0.0);
        let dense = sigma(&c);
        assert!(dense > 1.0, "{dense}");
        let (id, d) = c.promote(pole);
        let fresh = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
        assert!((c.cost() - fresh.cost()).abs() < 1e-6 * fresh.cost().abs().max(1.0), "{} vs {}", c.cost(), fresh.cost());
        assert!((d - fresh.cost()).abs() < 1e-6 * d.abs().max(1.0));
        assert!(d < 0.0, "{d}");
        assert!(sigma(&c) < 0.5 * dense, "{} vs {dense}", sigma(&c));
        // each ray is scored once: the elements' cells hold the whole cost (the field's own rays cancel against the
        // empty scene's; before step 2 a plug-in Poisson term scored every ray's counts again)
        let rows: f64 = c.rows().iter().map(|r| r.cost).sum::<f64>() + c.alignment_parts().iter().sum::<f64>() + c.field_part();
        assert!((rows - c.cost()).abs() < 1e-9 * c.cost().abs().max(1.0), "{rows} vs {}", c.cost());
        c.demote(id);
        assert!(c.cost().abs() < 1e-9, "{}", c.cost());
    }

    /// What the field's fit reads (research 130 slice 2b): a ray the pole fronts keeps its clear path in front of it
    /// (a voxel there holds the empty scene's path) and loses the rest (a voxel in its shadow holds less); the voxels
    /// within the margin of its surface are left out (σ is κ).
    #[test]
    fn the_field_reads_the_clear_path_in_front_of_wood() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let spec = Some(FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, margin: 0.1 });
        let empty = Scene::new(vec![v.clone()], &[], em, sc, spec, 0);
        let pole = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, spec, 0);
        let at = |c: &Scene, k: [i32; 3]| c.foliage().1.into_iter().find(|x| x.0 == k).unwrap();
        // in front (y 3.5–3.6 m), in the shadow (y 6.0–6.1 m), at the surface (y 3.9–4.0 m)
        let (front, shadow, surface) = ([5, 6, 3], [5, 31, 3], [5, 10, 3]);
        assert!(at(&empty, front).2 .1 > 0.0);
        assert!((at(&pole, front).2 .1 - at(&empty, front).2 .1).abs() < 1e-9 * at(&empty, front).2 .1);
        assert!(at(&pole, shadow).2 .1 < 0.95 * at(&empty, shadow).2 .1, "{:?} vs {:?}", at(&pole, shadow), at(&empty, shadow));
        let s = at(&pole, surface);
        assert!(s.3 && s.1 == sc.kappa && !at(&empty, surface).3, "{s:?}");
    }

    /// The field is held through moves (research 130 slice 2b), as the alignment is: moves keep its counts (after
    /// any of them, `refit_foliage` gives a fresh scene's counts and cost), adding then removing an element and moving
    /// one there and back restore the cost exactly, and after a small move the held cost is near the refitted one.
    /// After a move to a wrong, wider stick it isn't: the move takes the wall's echoes behind the stick's extra edge
    /// cells (seen by those rays alone here) out of the counts, but what a pulse meets is held, so the stick's pulses
    /// read them as the dense medium they were and it looks better held than refitted (by 27 nats on this one scan).
    /// The field's term scores that change at once; what a pulse meets behind an element waits for the refit (the
    /// hold formulation.md keeps). Before the field was collapsed the empty scene read those echoes the same way, and
    /// the held cost erred the other way.
    #[test]
    fn held_moves_keep_the_fields_counts() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let spec = Some(FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, margin: 0.1 });
        let (pole, nudged, wrong, side) = (stick(0.0, 4.0, 0.03), stick(0.002, 4.0, 0.03), stick(0.01, 4.0, 0.033), stick(0.15, 4.5, 0.02));
        let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
        let e0 = c.cost();
        let (id, _) = c.add(side);
        c.remove(id);
        assert_eq!(c.cost(), e0);
        c.change(0, wrong);
        c.change(0, pole);
        assert_eq!(c.cost(), e0);
        let counts = |c: &Scene| c.foliage().1.into_iter().map(|x| (x.0, x.2, x.3)).collect::<Vec<_>>();
        let same = |a: &Scene, b: &Scene| {
            let (x, y) = (counts(a), counts(b));
            x.len() == y.len() && x.iter().zip(&y).all(|(x, y)| x.0 == y.0 && x.2 == y.2 && (x.1 .0 - y.1 .0).abs() < 1e-9 && (x.1 .1 - y.1 .1).abs() < 1e-9)
        };
        assert!(same(&c, &Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0)));
        for (e, close) in [(nudged, true), (wrong, false)] {
            let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
            c.change(0, e);
            let held = c.cost();
            c.refit_foliage();
            let fresh = Scene::new(vec![v.clone()], &[e], em, sc, spec, 0);
            assert!(same(&c, &fresh));
            assert!((c.cost() - fresh.cost()).abs() < 1e-6 * fresh.cost().abs(), "{} vs {}", c.cost(), fresh.cost());
            if close {
                assert!((held - c.cost()).abs() < 0.01 * c.cost().abs(), "held {held} vs refitted {}", c.cost());
            } else {
                assert!(held < c.cost() - 10.0, "held {held} vs refitted {}", c.cost());
            }
        }
    }

    /// A layer enters its cell by its entry weight (3dtrees research 130 slice-trim), so the cost doesn't step where a
    /// fill crosses [`MIN_FILL`] or [`FULL_FILL`]. A thin stick slid 2 mm across the pole's edge, in front of it (cells
    /// there shared between the two terms by their front shares) and past it: wherever a column of its cells' widest
    /// fill crosses either (found by bisection), its cost to add changes by under 1e-6 nats across ±1e-12 m (an entry
    /// weight rising over 1e-10 of fill at 1 % steps by 0.009 nats). Where a partial layer stands in front of a whole one, both terms read the
    /// cell.
    #[test]
    fn the_cost_is_continuous_where_a_layer_enters() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
        let at = |x: f64| stick(x, 3.9, 0.008);
        let cost = |x: f64| c.costs_to_add(&[vec![at(x)]])[0][0];
        let fills = |x: f64| -> HashMap<usize, f64> { render(&v, &at(x), 0, &em).into_iter().map(|(k, l)| (k, l.fq[2])).collect() };
        let xs: Vec<f64> = (0..=40).map(|i| 0.026 + 5e-5 * i as f64).collect();
        let (mut worst, mut n, mut shared) = (0.0f64, 0, false);
        let both = |x: f64| kbuffer(&render_all(&v, &[stick(0.0, 4.0, 0.03), at(x)], &em));
        for w in xs.windows(2) {
            let (f0, f1) = (fills(w[0]), fills(w[1]));
            for threshold in [MIN_FILL, FULL_FILL] {
                let crosses = |k: &usize| f1.get(k).is_some_and(|&b| (f0[k] - threshold) * (b - threshold) < 0.0);
                let Some(&k) = f0.keys().find(|k| crosses(k)) else { continue };
                let above = |x: f64| fills(x).get(&k).copied().unwrap_or(0.0) > threshold;
                let (mut lo, mut hi, start) = (w[0], w[1], above(w[0]));
                while hi - lo > 1e-13 {
                    let m = 0.5 * (lo + hi);
                    if above(m) == start { lo = m } else { hi = m }
                }
                worst = worst.max((cost(hi + 1e-12) - cost(lo - 1e-12)).abs());
                n += 1;
                shared |= [lo, hi].iter().any(|&x| both(x).values().any(|ls| !ls[0].full() && front_shares(ls).iter().filter(|&&s| s > 0.0).count() == 2));
            }
        }
        assert!(n >= 4 && shared, "{n} crossings");
        assert!(worst < 1e-6, "a step of {worst} nats");
    }

    /// Many hypotheses at once: each one's cost per scan, and the foliage field's part, sum to adding then removing
    /// it, the scene unchanged.
    #[test]
    fn costs_to_add_match_add_then_remove() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, Some(field_box(0.1)), 0);
        let hyps = vec![vec![stick(0.0, 4.0, 0.02)], vec![stick(0.2, 6.0, 0.03), stick(0.25, 6.0, 0.01)], vec![stick(5.0, 4.0, 0.03)], vec![stick(0.0365, 3.9, 0.008)]];
        let e0 = c.cost();
        let all = c.costs_to_add(&hyps);
        assert_eq!(c.cost(), e0);
        for (h, per) in hyps.iter().zip(&all) {
            let (ids, d): (Vec<u32>, f64) = h.iter().fold((Vec::new(), 0.0), |(mut ids, d), e| {
                let (id, de) = c.add(*e);
                ids.push(id);
                (ids, d + de)
            });
            for id in ids {
                c.remove(id);
            }
            let total: f64 = per.iter().sum();
            assert!((total - d).abs() < 1e-6 * d.abs().max(1.0), "{total} vs {d}");
            assert_eq!(c.cost(), e0);
        }
        // a thinner stick behind the pole's front and one outside the box change no cut
        assert!(all[0][2] == 0.0 && all[1][2] != 0.0 && all[2][2] == 0.0, "{all:?}");
    }

    /// The field's box about the test scan's pole and wall, its margin `margin`.
    fn field_box(margin: f64) -> FoliageBox {
        FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, margin }
    }

    /// The field collapsed (research 130 formulation.md): its term, `Σ ln NB` over the counts, is exact through moves
    /// (a fresh scene's, bit for bit, and a recompute from the counts); a move's change to the field's part is the
    /// change in that part (the plug-in held); and adding then removing restores it exactly.
    #[test]
    fn the_fields_marginal_is_exact_through_moves() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let (pole, wrong, side) = (stick(0.0, 4.0, 0.03), stick(0.01, 4.0, 0.033), stick(0.15, 4.5, 0.02));
        let field = |c: &Scene| c.field.clone().unwrap();
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[pole], em, sc, Some(field_box(0.1)), 0);
        let (f0, e0) = (c.field_part(), c.cost());
        let (id, _) = c.add(side);
        c.change(0, wrong);
        c.remove(id);
        c.change(0, pole);
        assert_eq!((c.field_part(), c.cost()), (f0, e0));
        c.change(0, wrong);
        let fresh = Scene::new(vec![v.clone(), v.clone()], &[wrong], em, sc, Some(field_box(0.1)), 0);
        assert_eq!(field(&c).marginal(), field(&fresh).marginal());
        let (a0, b0) = field(&c).prior;
        let recomputed: f64 = field(&c).counts().values().map(|&(h, l)| foliage::ln_nb(h, l, a0, b0)).sum();
        assert!((recomputed - field(&c).marginal()).abs() < 1e-9 * recomputed.abs(), "{recomputed} vs {}", field(&c).marginal());
        c.refit_foliage();
        assert_eq!(c.field_part(), fresh.field_part());
        let before = c.field_part();
        let per = c.costs_to_add(&[vec![side]]);
        c.add(side);
        assert!((per[0][2] - (c.field_part() - before)).abs() < 1e-9, "{} vs {}", per[0][2], c.field_part() - before);
    }

    /// With the field collapsed and a margin about the pole, the pole's cost to add hardly depends on whether the
    /// field it is scored against was fitted with it in (2 nats here), where the scans' parts alone, the plug-in
    /// reading before the collapse, differ by 125: the field's term is a function of the scene alone.
    #[test]
    fn a_hypothesiss_cost_does_not_depend_on_the_last_refit() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = stick(0.0, 4.0, 0.03);
        let out = Scene::new(vec![v.clone()], &[], em, sc, Some(field_box(0.1)), 0).costs_to_add(&[vec![pole]])[0].clone();
        let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, Some(field_box(0.1)), 0);
        let inn: Vec<f64> = c.costs_to_remove(&[0])[0].iter().map(|x| -x).collect();
        let (collapsed, today) = ((out[0] + out[1]) - (inn[0] + inn[1]), out[0] - inn[0]);
        assert!(collapsed.abs() < 0.1 * today.abs(), "collapsed {collapsed} vs today {today}");
        c.remove(0);
    }

    /// The own nodes' slope and curvature in closed form ([`ElemData::quad`], research 130 slice-collapsed) match
    /// central differences of the ray likelihood, at the pole's truth and off it; and a term's mode is the same from
    /// a cold start and a warm one, where the cost's slope in its own nodes vanishes.
    #[test]
    fn closed_form_derivatives_match_finite_differences() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        for e in [stick(0.0, 4.0, 0.03), stick(0.004, 4.0, 0.025)] {
            let rendered = render_all(&v, &[e], &em);
            let d = &elem_data(&v, &rendered, &[e], &[0], &[0], 0, &em, &sc, &Foliage::uniform(sc.kappa))[0];
            assert!(d.own_sd[0] > 0.0 && d.own_sd[1] > 0.0);
            for x in [(0.0, 0.0), (0.002, -0.4)] {
                let q = d.quad(x, &em);
                let f = |dx: f64, db: f64| -d.quad((x.0 + dx, x.1 + db), &em).ll;
                let (h, hb) = (1e-5, 2e-3);
                let g = [(f(h, 0.0) - f(-h, 0.0)) / (2.0 * h), (f(0.0, hb) - f(0.0, -hb)) / (2.0 * hb)];
                let hess = [(f(h, 0.0) - 2.0 * f(0.0, 0.0) + f(-h, 0.0)) / (h * h), (f(h, hb) - f(h, -hb) - f(-h, hb) + f(-h, -hb)) / (4.0 * h * hb),
                            (f(0.0, hb) - 2.0 * f(0.0, 0.0) + f(0.0, -hb)) / (hb * hb)];
                assert_eq!(q.ll, -f(0.0, 0.0));
                for i in 0..2 {
                    assert!((q.g[i] - g[i]).abs() < 1e-4 * (g[i].abs() + 1.0), "{x:?} slope {:?} vs {g:?}", q.g);
                }
                let raw = [q.hess[(0, 0)], q.hess[(0, 1)], q.hess[(1, 1)]];
                for i in 0..3 {
                    assert!((raw[i] - hess[i]).abs() < 1e-3 * (hess[0].abs() * hess[2].abs()).sqrt().max(1.0), "{x:?} curvature {raw:?} vs {hess:?}");
                }
            }
            let (cold, warm) = (d.term(0.0, Vector2::zeros(), &em), d.term(0.0, Vector2::new(0.5, -0.5), &em));
            assert!((cold.own[0] - warm.own[0]).abs() < 1e-3 && (cold.own[1] - warm.own[1]).abs() < 1e-3, "{:?} vs {:?}", cold.own, warm.own);
            assert!((cold.cost() - warm.cost()).abs() < 1e-5, "{} vs {}", cold.cost(), warm.cost());
            let at = d.quad(d.at(0.0, cold.own), &em);
            for i in 0..2 {
                assert!((d.own_sd[i] * at.g[i] + cold.own[i]).abs() < 0.01, "slope in y {:?} at {:?}", at.g, cold.own);
            }
        }
    }

    /// A scene's moves hold the alignment (step 3a): each move's cost change stays near the exact marginal's
    /// (re-solved), for an add beside the pole (sharing its alignment node), an alter and a remove; adding then removing
    /// an element restores the cost exactly (synthesis §18 invariant 7); `costs_to_add` and `costs_to_remove` equal
    /// the moves; and after `refit_alignment` the cost is a fresh scene's.
    #[test]
    fn held_moves_match_the_exact_marginal() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let full = |els: &[Element]| cost(&v, els, &em, &sc).0;
        let (pole, side, wide) = (stick(0.0, 4.0, 0.03), stick(0.15, 4.5, 0.02), stick(0.0, 4.0, 0.035));
        let near = |held: f64, exact: f64| (held - exact).abs() < 0.01 * exact.abs().max(5.0);
        let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, None, 0);
        let e0 = c.cost();
        assert_eq!(e0, c.exact_cost());
        assert!((e0 - full(&[pole])).abs() < 1e-9 * e0.abs().max(1.0));
        let t: f64 = c.costs_to_add(&[vec![side]])[0].iter().sum();
        let (id, d_add) = c.add(side);
        assert_eq!(t, d_add);
        assert!(near(d_add, full(&[pole, side]) - e0), "add {d_add} exact {}", full(&[pole, side]) - e0);
        let d_alter = c.change(0, wide);
        assert!(near(d_alter, full(&[wide, side]) - full(&[pole, side])), "alter {d_alter} exact {}", full(&[wide, side]) - full(&[pole, side]));
        assert!(near(c.cost(), c.exact_cost()), "{} vs {}", c.cost(), c.exact_cost());
        let e1 = c.cost();
        let (other, _) = c.add(stick(-0.1, 5.0, 0.01));
        c.remove(other);
        assert_eq!(c.cost(), e1);
        let r: f64 = c.costs_to_remove(&[id])[0].iter().sum();
        let d_remove = c.remove(id);
        assert_eq!(r, d_remove);
        assert!(near(d_remove, full(&[wide]) - full(&[wide, side])), "remove {d_remove} exact {}", full(&[wide]) - full(&[wide, side]));
        c.refit_alignment();
        assert!((c.cost() - full(&[wide])).abs() < 1e-9 * c.cost().abs().max(1.0), "{} vs {}", c.cost(), full(&[wide]));
    }

    /// A move's change is a full recompute's (3dtrees research 130 slice-structure): each cell read at its own surface
    /// class, a term depends on no other term's class, so after adds, changes and removes among neighbours that share
    /// cells (a group among them), every scan's terms equal its terms recomputed from scratch at the held alignment
    /// (each from its reference's start), bit for bit; and the moves undone restore the cost exactly.
    #[test]
    fn a_moves_terms_are_a_full_recompute() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let front = stick(0.02, 3.9, 0.02);
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03), front, piece_of(-0.2, 0.0, Some(5))], em, sc, Some(field_box(0.1)), 0);
        let check = |c: &Scene| {
            for s in 0..c.n_scans() {
                let fit = &c.fits[s];
                let fresh: BTreeMap<Owner, f64> = c.data(s).into_iter()
                    .map(|(o, d)| (o, d.term(fit.held.shift(&d.align), fit.held.reference.get(&o).map_or(Vector2::zeros(), |t| t.start), &em).cost())).collect();
                assert_eq!(fresh, fit.terms.iter().map(|(o, t)| (*o, t.cost())).collect::<BTreeMap<Owner, f64>>());
            }
        };
        let e0 = c.cost();
        let (a, _) = c.add(stick(0.01, 3.95, 0.025));
        check(&c);
        c.change(1, stick(0.05, 3.85, 0.03));
        check(&c);
        // its edge partial in front of the pole's: cells two terms share
        c.change(1, stick(0.0365, 3.9, 0.008));
        check(&c);
        let (b, _) = c.add(piece_of(0.0, 0.2, Some(5)));
        check(&c);
        c.remove(a);
        check(&c);
        c.remove(b);
        c.change(1, front);
        check(&c);
        assert_eq!(c.cost(), e0);
    }

    /// Which hypotheses interact: two sticks on the pole's cells share cells and a node; one beside them in the view
    /// shares no cell, and a node with one of them.
    #[test]
    fn interactions_name_shared_cells_and_nodes() {
        let v = pole_view();
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], Settings::default(), Background::default(), None, 0);
        let pairs = c.interactions(&[vec![stick(0.0, 3.9, 0.03)], vec![stick(0.005, 4.1, 0.03)], vec![stick(-0.15, 7.5, 0.03)]]);
        assert_eq!(pairs[0], (0, 1, 0, true, true));
        assert!(pairs[1..].iter().all(|p| p.1 == 2 && !p.3 && p.4), "{pairs:?}");
    }

    /// The counts: the true pole's full-fill cells are on it and its echoes fit its sharp depth (a small
    /// excess over the noise); a stick where the pulses went through to the wall reads behind; the scene is
    /// left as it was.
    #[test]
    fn stats_read_on_for_the_pole_and_behind_for_a_phantom() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        let pole = c.counts(stick(0.0, 4.0, 0.03), &[])[0];
        assert!(pole.on > 10.0 * (pole.behind + pole.empty).max(0.01), "{pole:?}");
        assert!(pole.mismatch_cells > 50 && pole.mismatch_ss / pole.mismatch_w < 1e-5, "{pole:?}");
        let phantom = c.counts(stick(0.2, 6.0, 0.03), &[])[0];
        assert!(phantom.behind > 10.0 * phantom.on.max(0.01), "{phantom:?}");
        assert_eq!(c.cost(), 0.0);
    }

    /// A stick under a footprint across leaves partial hits (the export keeps a footprint 8 % filled): scored on
    /// the same echoes, a 40 % wider stick doesn't beat the truth, and a 30 % thinner one barely differs from it.
    /// (With the likelihood gated at fill 0.5 and the band below it scored by located edges, the wider stick won by
    /// 130–280 nats: its gate took the partial hits that the truth's edges scored weakly; slice 7c.)
    #[test]
    fn a_wider_thin_stick_gains_nothing() {
        for y in [4.0, 8.0] {
            let v = thin_pole_view(0.004, y);
            let (em, sc) = (Settings::default(), Background::default());
            let e = |r: f64| cost(&v, &[Element { mismatch: 0.0, ..stick(0.0, y, r) }], &em, &sc).0;
            let (thin, truth, wide) = (e(0.0028), e(0.004), e(0.0056));
            assert!(wide > truth, "{y} m: wide {wide} truth {truth}");
            assert!(thin > truth - 0.02 * truth.abs(), "{y} m: thin {thin} truth {truth}");
        }
    }

    /// The rendered depth is the footprint's mean on the section, which the mismatch (a spread about the form, not a
    /// bias) doesn't move: on the pole's front and halfway out it sits within 0.3 mm of the echoes, whatever the
    /// mismatch. (The mismatch used to widen the depth band, which a section's edge clips, so the front was rendered
    /// behind the echoes; slice 7b.)
    #[test]
    fn mismatch_does_not_move_the_rendered_front() {
        let v = pole_view();
        let em = Settings::default();
        for mismatch in [0.0, 0.002, 0.008] {
            for ly in render(&v, &Element { mismatch: mismatch, ..stick(0.0, 4.0, 0.03) }, 0, &em).iter().map(|(c, l)| (v.range[*c] as f64, l)) {
                let (obs, l) = ly;
                if l.x.abs() < 0.015 && obs.is_finite() && obs < 5.0 {
                    assert!((obs - l.t).abs() < 3e-4, "mismatch {mismatch}: x {} obs {obs} t {}", l.x, l.t);
                }
            }
        }
    }

    /// The Laplace marginal over the alignment error agrees with Gauss–Hermite quadrature on a single-element query (the
    /// check), a range shift in the data is recovered by the alignment error and the stem front shift, and the hats are
    /// normalised.
    #[test]
    fn the_alignment_integral_matches_quadrature_and_finds_a_shift() {
        for x in [[0.3, 4.1, -0.7], [1.0, 2.0, 3.0]] {
            let h = hats(&x, 2.0);
            assert!((h.iter().map(|(_, w)| w * w).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let el = stick(0.0, 4.0, 0.03);
        // shift the pole's echoes 3 mm deeper: the alignment error and the stem front shift together should find about that
        let shifted = {
            let mut w = pole_image();
            for k in 0..w.cell.len() {
                if w.cell[k] == Cell::Echoes && w.range[k] < 4.5 {
                    w.range[k] += 0.003;
                    let i = w.echo_start[w.pulse[k] as usize] as usize;
                    w.echo_range[i] += 0.003;
                }
            }
            Arc::new(w)
        };
        let (_, rows) = cost(&shifted, &[el], &em, &sc);
        assert!((rows[0].align_shift + rows[0].front_shift - 0.003).abs() < 0.0015, "{:?}", rows[0]);
        // the check: a single element's data integrated by quadrature over the prior of its along-beam shift (the
        // alignment error's and the stem front shift's, which add) and of its reflectance level
        let rendered = render_all(&v, &[el], &em);
        let (u_lap, out) = cost(&v, &[el], &em, &sc);
        assert_eq!(out.len(), 1);
        // the element sits on one node set: its alignment shift has prior sd s_along (√Σŵ² = 1)
        let (gh_x, gh_w) = ([-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
                            [0.019953242059045913, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116, 0.019953242059045913]);
        let buf = kbuffer(&rendered);
        let s_along = em.align_sd(term_classes(&v, &buf, &[0], &em)[0], false).hypot(em.front_shift_sd(el.r));
        let ed = |shift: f64, level: f64| -> f64 {
            buf.iter().map(|(&cell, ls)| {
                let echoes = v.echoes(cell);
                let wt = em.cell_weight(echoes.first().map_or(ls[0].t, |&r| r as f64), v.step());
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let pulse = |ls: Vec<Layer>| Pulse::new(occluders(ls, None), cell_class(&v, cell, &em), &v.origin, &w, echoes, v.echo_refls(cell), v.max_targets, &em, &Foliage::uniform(sc.kappa));
                // a partial layer is in the cell with its entry weight
                let lik: f64 = mixture(ls).into_iter().map(|(p, ls)| p * pulse(ls).read(shift, level, &em).ll.exp()).sum();
                wt * (lik.ln() - pulse(Vec::new()).read(0.0, 0.0, &em).ll)
            }).sum()
        };
        let sq = std::f64::consts::SQRT_2;
        let grid: Vec<(f64, f64)> = gh_x.iter().zip(&gh_w).flat_map(|(&x, &wx)| gh_x.iter().zip(&gh_w).map(move |(&y, &wy)| {
            (ed(sq * s_along * x, sq * em.refl_level_sd * y), wx * wy)
        })).collect();
        let m = grid.iter().map(|g| g.0).fold(f64::NEG_INFINITY, f64::max);
        let quad = -(m + (grid.iter().map(|&(e, w)| w * (e - m).exp()).sum::<f64>() / std::f64::consts::PI).ln());
        assert!((u_lap - quad).abs() < 0.02 * quad.abs().max(1.0), "laplace {u_lap} quadrature {quad}");
    }

    /// A patch renders as a disc: full fill at its centre, none beyond its rim; a patch of the pole's size where the
    /// pole is beats one off to the side; `predict` codes the pole's front cells on, and a stick beside it behind.
    #[test]
    fn patches_render_as_discs_and_predict_codes_cells() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let patch = |x: f64| Element { form: Form::Patch { centre: [x, 4.0, 0.0] }, class: Class::Leaf, r: 0.03, opacity: 1.0, mismatch: 0.0, group: None };
        let layers = render(&v, &patch(0.0), 0, &em);
        let f_max = layers.iter().map(|(_, l)| l.f).fold(0.0, f64::max);
        assert!(f_max > 0.99, "{f_max}");
        assert!(layers.iter().all(|(_, l)| l.x < 0.03 + 0.012 || l.f < 0.01));
        let e = |s: Element| cost(&v, &[s], &em, &sc).0;
        assert!(e(patch(0.0)) < e(patch(0.1)));
        let p = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(&[stick(0.0, 4.0, 0.03)], 0);
        let on = p.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::On);
        let beside = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(&[stick(0.1, 4.0, 0.03)], 0);
        let behind = beside.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::Behind);
        assert!(on && behind);
        assert!(p.iter().all(|q| q.weight > 0.0 && q.weight <= 1.0 / em.discount));
    }

    /// The class sets the reflectance: the same stick marked leaf is recorded less often than marked wood when leaf
    /// is dimmer (the review after slice 15: patches were rendered as wood).
    #[test]
    fn a_leaf_stick_reflects_as_leaf() {
        let v = pole_view();
        let em = Settings { refl_leaf: -40.0, ..Settings::default() };
        let c = Scene::new(vec![v.clone()], &[], em, Background::default(), None, 0);
        let wood = Element { mismatch: 0.0, ..stick(0.0, 4.0, 0.03) };
        let mean = |p: Vec<CellPrediction>| p.iter().map(|q| q.recorded).sum::<f64>() / p.len() as f64;
        let (leaf, wood) = (mean(c.predict(&[Element { class: Class::Leaf, ..wood }], 0)), mean(c.predict(&[wood], 0)));
        assert!(leaf < wood - 0.1, "leaf {leaf} wood {wood}");
    }

    /// Class is a mark, not a shape (slice 5a): at equal reflectances the same stick marked wood and marked leaf
    /// reads the same cells with the same nuisances (alignment, stem front shift, reflectance level) and costs the
    /// same (class evidence is the consumer's: slice 9). A patch has no stem front shift.
    #[test]
    fn the_same_stick_marked_wood_and_leaf_shares_cells_and_nuisances() {
        let v = thin_pole_view(0.03, 4.0);
        let em = Settings { refl_leaf: Settings::default().refl_wood, ..Settings::default() };
        let wood = stick(0.0, 4.0, 0.03);
        let rows = |e: Element| cost(&v, &[e], &em, &Background::default()).1[0];
        let (w, l) = (rows(wood), rows(Element { class: Class::Leaf, ..wood }));
        assert_eq!((w.cells, w.cells_on), (l.cells, l.cells_on));
        for (a, b) in [(w.ray_on, l.ray_on), (w.ray_empty, l.ray_empty), (w.ray_behind, l.ray_behind), (w.alignment, l.alignment),
                       (w.align_shift, l.align_shift), (w.front_shift, l.front_shift), (w.refl_level, l.refl_level)] {
            assert_eq!(a, b);
        }
        assert_eq!(w.cost, l.cost);
        let patch = Element { form: Form::Patch { centre: [0.0, 4.0, 0.0] }, class: Class::Leaf, r: 0.03, opacity: 1.0, mismatch: 0.002, group: None };
        assert_eq!(rows(patch).front_shift_sd, 0.0);
    }

    /// The reference epoch is declared, not the first scan's: a scene of the same scans in either order costs the
    /// same, and a scan off the reference epoch gets the wider, other-epoch alignment error.
    #[test]
    fn the_reference_epoch_is_declared_and_scan_order_does_not_matter() {
        let (a, mut b) = (pole_image(), pole_image());
        b.epoch = 1;
        let (a, b) = (Arc::new(a), Arc::new(b));
        let (em, sc) = (Settings::default(), Background::default());
        let el = [stick(0.0, 4.0, 0.03)];
        let ab = Scene::new(vec![a.clone(), b.clone()], &el, em, sc, None, 0);
        let ba = Scene::new(vec![b.clone(), a.clone()], &el, em, sc, None, 0);
        assert!((ab.cost() - ba.cost()).abs() < 1e-9, "{} vs {}", ab.cost(), ba.cost());
        let rows = ab.rows();
        assert!(rows[1].align_sd > rows[0].align_sd, "{:?}", rows);
    }

    /// How well a scan pins a pole down: its expected information is positive in every direction of a stick's
    /// geometry; a row's parts sum to its cost, and the rows and the scans' alignment parts to the scene's.
    #[test]
    fn the_pole_is_pinned_down_and_rows_split_their_cost() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
        let x = &c.information(&[0])[0].expected;
        for i in 0..5 {
            assert!(x[i * 5 + i] > 0.0, "{i}: {x:?}");
        }
        let r = c.rows()[0];
        let parts = r.ray_on + r.ray_empty + r.ray_in_front + r.ray_behind + r.alignment;
        assert!((parts - r.cost).abs() < 1e-9, "{parts} vs {}", r.cost);
        assert!((r.cost + c.alignment_parts()[0] - c.cost()).abs() < 1e-9);
        assert_eq!(r.discount, em.discount);
    }

    /// Element `e` moved by `th` in [`Scene::information`]'s parameters, as an element (the referee for [`placed`]).
    fn moved(e: &Element, th: &[f64]) -> Element {
        let form = match e.form {
            Form::Stick { a, b } => {
                let ax = sub(&b, &a);
                let half = 0.5 * norm(&ax);
                let u = ax.map(|x| x / (2.0 * half));
                let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
                let p2 = cross(&u, &p1);
                let mv = |p: &Point, sign: f64| [0, 1, 2].map(|k| p[k] + (th[1] + sign * half * th[3]) * p1[k] + (th[2] + sign * half * th[4]) * p2[k]);
                Form::Stick { a: mv(&a, -1.0), b: mv(&b, 1.0) }
            }
            Form::Patch { centre } => Form::Patch { centre: [centre[0] + th[1], centre[1] + th[2], centre[2] + th[3]] },
        };
        Element { form, r: e.r * th[0].exp(), ..*e }
    }

    /// The forward slope (research 130 slice-derivatives) matches central differences of the cost of the cells the
    /// element's layer is in, each read with that layer moved (the same reading in `f64`) at its term's nodes: a stick
    /// off the pole, behind a thinner stick (cells it lies behind another's front in) and that stick, and a patch;
    /// the pass's value is the `f64` one bit for bit; and a moved layer is the moved element's rendered. Off the
    /// pole: the test scan's echoes sit at the true pole's layers' ranges, where a layer passing an echo switches
    /// the returns the medium's echo may precede (a jump of a few micronats, measure zero on real data).
    #[test]
    fn forward_slopes_match_finite_differences() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let patch = Element { form: Form::Patch { centre: [0.001, 4.002, 0.0] }, class: Class::Leaf, r: 0.029, opacity: 1.0, mismatch: 0.002, group: None };
        let (pole, front) = (stick(0.002, 4.003, 0.028), stick(0.02, 3.9, 0.02));
        for (els, id) in [(vec![pole], 0), (vec![pole, front], 0), (vec![pole, front], 1), (vec![patch], 0)] {
            let c = Scene::new(vec![v.clone()], &els, em, sc, None, 0);
            let e = els[id as usize];
            let n = if let Form::Stick { .. } = e.form { 5 } else { 4 };
            let slope = &c.information(&[id])[0].slope;
            let cells: Vec<(usize, Vec<Layer>, Vec<(usize, [f64; 2])>)> = c.cells_of(0, &[id]).into_iter()
                .map(|(cell, ls, owners)| (cell, ls, owners.iter().map(|&(i, _, x)| (i, x)).collect())).collect();
            let cost = |th: &[f64]| -> f64 {
                cells.iter().map(|(cell, ls, nodes)| c.read_cell(0, *cell, ls, ls.iter().position(|l| l.elem == id).unwrap(), &e, th, nodes).cost).sum()
            };
            let th0: Vec<Geometry> = (0..n).map(|i| Geometry::var(0.0, i)).collect();
            for (cell, ls, nodes) in &cells {
                let j = ls.iter().position(|l| l.elem == id).unwrap();
                let lifted: Vec<(usize, [Geometry; 2])> = nodes.iter().map(|&(i, x)| (i, x.map(Geometry::c))).collect();
                assert_eq!(c.read_cell(0, *cell, ls, j, &e, &th0, &lifted).cost.v, c.read_cell(0, *cell, ls, j, &e, &vec![0.0; n], nodes).cost);
            }
            assert!(cells.iter().any(|(_, ls, _)| ls[0].elem != id) == (els.len() > 1 && id == 0));
            let th = [0.02, 0.001, -0.002, 0.003, 0.001];
            for (cell, ly) in render(&v, &moved(&e, &th[..n]), id, &em).iter().filter(|(_, l)| l.enters()) {
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let at = super::layer(&placed(&e, &v.origin, Some(&th[..n])), &w, &e, id, &em, v.step()).unwrap();
                assert!((at.t - ly.t).abs() < 1e-12 && (at.f - ly.f).abs() < 1e-12, "{at:?} vs {ly:?}");
            }
            for i in 0..n {
                let h = [1e-6, 1e-7, 1e-7, 1e-6, 1e-6][i];
                let mut th = vec![0.0; n];
                th[i] = h;
                let up = cost(&th);
                th[i] = -h;
                let fd = (up - cost(&th)) / (2.0 * h);
                assert!((slope[i] - fd).abs() < 1e-4 * slope[i].abs().max(fd.abs()) + 1e-3, "{id} parameter {i}: forward {} central {fd}", slope[i]);
            }
        }
    }

    /// The expected information from the forward pass is the limit of the finite-difference stencil it replaced
    /// (research 130 step 3a's, kept here as the referee): the element's predictions at ± a step in each parameter
    /// differenced, its own nodes at 0 and its chance of an echo loading no node. On the pole drawn sharp (no shape
    /// mismatch, so the recorded echo's mean reflectance is the middle radius's the stencil read), at a thousandth of
    /// the old steps (0.05 in ln r, 2 mm in place, that over the half-length in tilt) every entry agrees to 1e-5 of
    /// the matrix's norm (the diagonals to 3e-6); at the old steps the stencil read ln r's information 17 % low and
    /// place across the beam 28 % low, its steps reaching past the edge cells' fills.
    #[test]
    fn expected_information_is_the_old_stencils_limit() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = Element { mismatch: 0.0, ..stick(0.0, 4.0, 0.03) };
        let new = Scene::new(vec![v.clone()], &[pole], em, sc, None, 0).information(&[0]).remove(0).expected;
        let empty = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        let pred = |th: &[f64]| -> HashMap<usize, CellPrediction> { empty.predict(&[moved(&pole, th)], 0).into_iter().map(|p| (p.cell, p)).collect() };
        let stencil = |k: f64| -> Vec<f64> {
            let steps = [0.05, 0.002, 0.002, 0.002 / 0.2, 0.002 / 0.2].map(|x| x * k);
            let sides: Vec<[HashMap<usize, CellPrediction>; 2]> = (0..5).map(|i| [1.0, -1.0].map(|sg| {
                let mut t = [0.0; 5];
                t[i] = sg * steps[i];
                pred(&t)
            })).collect();
            let mut shared = Shared::new(5, &[pole], &em);
            let mut centre: Vec<CellPrediction> = pred(&[0.0; 5]).into_values().collect();
            centre.sort_by_key(|p| p.cell);
            for p in centre {
                let slope = |f: &dyn Fn(&CellPrediction) -> f64, absent: Option<f64>| -> Vec<f64> {
                    (0..5).map(|i| match [0, 1].map(|k| sides[i][k].get(&p.cell).map(f).or(absent)) {
                        [Some(a), Some(b)] if (a - b).is_finite() => (a - b) / (2.0 * steps[i]),
                        _ => 0.0,
                    }).collect()
                };
                let echo = |q: &CellPrediction| q.arrives * q.recorded;
                let (align, front) = (em.align_sd(p.surface, false), pole.front_shift_sd(&em));
                let info = cell_information(&em, echo(&p), p.sd * p.sd - align * align - front * front, p.fill);
                shared.add(&slope(&echo, Some(0.0)), &[], info[0], p.weight);
                let load = shared.load(0, align, 1.0, 0.0, em.refl_level_sd);
                shared.add(&slope(&|q| q.depth, None), &load, info[1], p.weight);
                if p.refl.is_finite() {
                    let load = shared.load(0, align, 0.0, 1.0, em.refl_level_sd);
                    shared.add(&slope(&|q| q.refl, None), &load, info[2], p.weight);
                }
            }
            shared.finish()[0].clone()
        };
        let norm = new.iter().map(|x| x * x).sum::<f64>().sqrt();
        let fine = stencil(1e-3);
        assert!(new.iter().zip(&fine).all(|(a, b)| (a - b).abs() < 1e-5 * norm), "{new:?}\n{fine:?}");
        let old = stencil(1.0);
        assert!((old[0] / new[0] - 0.83).abs() < 0.02 && (old[12] / new[12] - 0.72).abs() < 0.02, "{new:?}\n{old:?}");
    }

    /// Several elements' information (research 130 slice 4). Far apart (no alignment node or cell in common), a pair's
    /// joint expected information is each one's own, side by side. Neighbours sharing their alignment node (its spread
    /// 1 cm here, the stem front shift off) have it counted once: each one's own information charges it in full along
    /// the beam, but it cancels in their depth difference, which the pair pins down far better. Predictions over
    /// several elements are each one's own where they share no cell, and a hypothesis is expected to cost nothing
    /// against itself, and about half the information times a small move's square against that move.
    #[test]
    fn a_pairs_information_counts_the_shared_alignment_once() {
        let v = pole_view();
        let sc = Background::default();
        let em = Settings { align_along_solid: 0.01, align_along_mid: 0.01, align_along_fragmented: 0.01, front_shift_base: 0.0, front_shift_slope: 0.0, ..Settings::default() };
        let block = |e: &[f64], n: usize, o: usize, k: usize| (0..k * k).map(|i| e[(o + i / k) * n + o + i % k]).collect::<Vec<f64>>();
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1e-3));
        // far apart: the pole, and a stick 2.5 m behind it and to the side (other nodes, other cells)
        let (pole, far) = (stick(0.0, 4.0, 0.03), stick(0.25, 6.5, 0.03));
        let c = Scene::new(vec![v.clone()], &[pole, far], em, sc, None, 0);
        let joint = c.information(&[0, 1]).remove(0).expected;
        let (a, b) = (c.information(&[0]).remove(0).expected, c.information(&[1]).remove(0).expected);
        assert!(a[0] > 0.0 && b[0] > 0.0);
        assert!(close(&block(&joint, 10, 0, 5), &a) && close(&block(&joint, 10, 5, 5), &b), "{joint:?}\n{a:?}\n{b:?}");
        assert!((0..5).all(|i| (5..10).all(|j| joint[i * 10 + j].abs() <= 1e-9 * joint[i * 11].max(joint[j * 11]))), "{joint:?}");
        // neighbours on one node: the sd of their along-beam places' difference (parameter 1, p1 along the beam)
        let near = Scene::new(vec![v.clone()], &[pole, stick(0.1, 4.0, 0.03)], em, sc, None, 0);
        let var = |e: &[f64], n: usize, d: &[f64]| {
            let m = nalgebra::DMatrix::from_row_slice(n, n, e) + nalgebra::DMatrix::<f64>::identity(n, n) * 1e-2;
            let d = nalgebra::DVector::from_column_slice(d);
            d.dot(&(m.try_inverse().unwrap() * &d))
        };
        let joint = near.information(&[0, 1]).remove(0).expected;
        let mut d = vec![0.0; 10];
        (d[1], d[6]) = (1.0, -1.0);
        let unit = [0.0, 1.0, 0.0, 0.0, 0.0];
        let own: f64 = [0, 1].map(|k| var(&near.information(&[k]).remove(0).expected, 5, &unit)).iter().sum();
        assert!(var(&joint, 10, &d) < 0.1 * own, "joint {} own {own}", var(&joint, 10, &d));
        // predictions over several elements, and the divergence
        let empty = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        let both = empty.predict(&[pole, far], 0);
        let alone = [empty.predict(&[pole], 0), empty.predict(&[far], 0)];
        assert_eq!(both.len(), alone[0].len() + alone[1].len());
        assert!(both.iter().all(|p| alone[p.elem as usize].iter().any(|q| q.cell == p.cell && q.depth == p.depth)));
        assert_eq!(empty.divergence(&[pole], &[pole])[0], 0.0);
        let info = Scene::new(vec![v.clone()], &[pole], em, sc, None, 0).information(&[0]).remove(0).expected;
        for i in [0, 2] {
            let mut th = [0.0; 5];
            th[i] = if i == 0 { 0.05 } else { 0.002 };
            let (kl, quad) = (empty.divergence(&[pole], &[moved(&pole, &th)])[0], 0.5 * th[i] * th[i] * info[i * 6]);
            assert!(kl > 0.5 * quad && kl < 2.0 * quad, "{i}: divergence {kl}, ½ δ² E {quad}");
        }
    }

    /// The evidence discount enters the information once (research 130 step 3b): in the pole's place across the beam
    /// (read from misses and echoes; no shared latent's prior pins it), doubling φ halves the expected information E
    /// and quarters the gradient's variance with independent cells J, so the sandwich `E⁻¹ J E⁻¹` (the variance were
    /// the cells independent) doesn't depend on φ while E⁻¹ (the variance at φ) doubles. And the pole's echoes span
    /// most of its near side's outline.
    #[test]
    fn the_sandwich_does_not_depend_on_the_discount() {
        let v = pole_view();
        let sc = Background::default();
        let at = |phi: f64| {
            let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], Settings { discount: phi, ..Settings::default() }, sc, None, 0);
            let i = c.information(&[0]).remove(0);
            (i.expected[12], i.gradient_variance[12], c.rows()[0].arc)
        };
        let ((e1, j1, arc), (e2, j2, _)) = (at(10.0), at(20.0));
        assert!((e1 / e2 - 2.0).abs() < 0.2, "E {e1} at φ 10, {e2} at 20");
        assert!((j1 / j2 - 4.0).abs() < 0.4, "J {j1} at φ 10, {j2} at 20");
        let sandwich = |e: f64, j: f64| j / (e * e);
        assert!((sandwich(e1, j1) / sandwich(e2, j2) - 1.0).abs() < 0.1, "{} vs {}", sandwich(e1, j1), sandwich(e2, j2));
        assert!(j1 > 0.0 && j1 < e1 / 10.0, "J {j1} E {e1}: each cell's weight is at most 1/φ");
        assert!(arc > 2.5 && arc <= std::f64::consts::PI, "arc {arc}");
    }

    /// Located edges (research 130 step 2): the silhouette's partial cells, read through their echoes' reflectance
    /// against the beam's fill, pin a thin stick's radius down more than their depths and misses alone, from one view.
    #[test]
    fn located_edges_pin_a_thin_sticks_radius() {
        let sc = Background::default();
        for (r, y) in [(0.01, 6.0), (0.006, 6.0)] {
            let v = thin_pole_view(r, y);
            let info = |em: Settings| Scene::new(vec![v.clone()], &[stick(0.0, y, r)], em, sc, None, 0).information(&[0])[0].expected[0];
            let (with, without) = (info(Settings::default()), info(Settings { refl_noise: 0.0, ..Settings::default() }));
            assert!(with > 1.5 * without, "r {r}: with reflectance {with}, without {without}");
        }
    }

    /// A pole whose wood is 3 dB brighter than declared: its reflectance level takes the difference, so the true radius
    /// still beats a thinner and a thicker one, and the level reads about 3 dB.
    #[test]
    fn a_brighter_wood_is_read_as_its_level_not_its_width() {
        let (em, sc) = (Settings::default(), Background::default());
        let v = Arc::new(pole_scan_of(0.01, 6.0, 6.0, em.refl_wood + 3.0));
        let e = |r: f64| cost(&v, &[stick(0.0, 6.0, r)], &em, &sc);
        let truth = e(0.01);
        for r in [0.008, 0.0125] {
            assert!(e(r).0 > truth.0 + 0.5, "r {r}: {} vs {}", e(r).0, truth.0);
        }
        let row = truth.1[0];
        assert!((row.refl_level - 3.0).abs() < 3.0 * row.refl_level_sd.max(0.3), "{row:?}");
    }

    /// A piece of the test pole, from height `lo` to `hi`, in group `g`.
    fn piece_of(lo: f64, hi: f64, g: Option<u32>) -> Element {
        Element { form: Form::Stick { a: [0.0, 4.0, lo], b: [0.0, 4.0, hi] }, group: g, ..stick(0.0, 4.0, 0.03) }
    }

    /// A group is one term per scan (slice 6): a group of one costs what the element alone does; a pole cut in two
    /// pieces and grouped reads their cells together under one set of nuisances (one row, one stem front shift and
    /// reflectance level, its prior and log-determinant paid once), where the pieces alone pay theirs twice.
    #[test]
    fn a_groups_members_share_their_nuisances() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let one = piece_of(-0.2, 0.2, None);
        assert_eq!(cost(&v, &[one], &em, &sc).0, cost(&v, &[Element { group: Some(3), ..one }], &em, &sc).0);
        let (alone, grouped) = ([piece_of(-0.2, 0.0, None), piece_of(0.0, 0.2, None)], [piece_of(-0.2, 0.0, Some(3)), piece_of(0.0, 0.2, Some(3))]);
        let (a, g) = (cost(&v, &alone, &em, &sc).1, cost(&v, &grouped, &em, &sc).1);
        assert_eq!((a.len(), g.len()), (2, 1));
        assert_eq!((g[0].elem, g[0].group, g[0].cells), (0, Some(3), a[0].cells + a[1].cells));
        assert!(g[0].refl_level_sd < a[0].refl_level_sd.min(a[1].refl_level_sd), "{:?} {:?}", g[0], a);
        // the own nodes' prior and log-determinant: once for the group, once a piece alone
        assert!(g[0].alignment < a[0].alignment + a[1].alignment, "{} vs {}", g[0].alignment, a[0].alignment + a[1].alignment);
    }

    /// A move that touches one member rescores its whole group, so its pooled terms never go stale: pieces of the
    /// pole 10 cm apart (no cell in common), added one by one, reach the same cost as the group built at once;
    /// removing a member restores the cost exactly, and `costs_to_add` of a member joining a group equals adding it.
    #[test]
    fn a_touched_member_rescores_its_group() {
        let v = thin_pole_view(0.0015, 6.0);
        let (em, sc) = (Settings::default(), Background::default());
        let piece = |z: f64| Element { form: Form::Stick { a: [0.0, 6.0, z - 0.02], b: [0.0, 6.0, z + 0.02] }, group: Some(1), ..stick(0.0, 6.0, 0.0015) };
        let pieces = [piece(-0.15), piece(-0.05), piece(0.05), piece(0.15)];
        let at_once = Scene::new(vec![v.clone()], &pieces, em, sc, None, 0);
        let mut by_one = Scene::new(vec![v.clone()], &pieces[..1], em, sc, None, 0);
        for p in &pieces[1..] {
            let t: f64 = by_one.costs_to_add(&[vec![*p]])[0].iter().sum();
            let e0 = by_one.cost();
            assert_eq!(t, by_one.add(*p).1);
            let (id, _) = by_one.add(Element { group: Some(1), ..*p });
            by_one.remove(id);
            assert_eq!(by_one.cost(), e0 + t);
        }
        by_one.refit_alignment();
        assert!((by_one.cost() - at_once.cost()).abs() < 1e-9 * at_once.cost().abs().max(1.0), "{} vs {}", by_one.cost(), at_once.cost());
        assert_eq!(by_one.rows().len(), 1);
    }

    /// A chain is one hypothesis, so its own joints (two sticks meeting, sharing cells and a node) never read as
    /// interacting; a hypothesis fronting cells of a scene group's member, or joining that group, interacts with
    /// another that joins it.
    #[test]
    fn a_chains_joints_are_not_interacting_hypotheses() {
        let v = pole_view();
        let c = Scene::new(vec![v.clone()], &[piece_of(-0.2, 0.0, Some(5)), stick(0.4, 6.0, 0.02)], Settings::default(), Background::default(), None, 0);
        let chain = vec![piece_of(-0.1, 0.05, Some(9)), piece_of(0.05, 0.2, Some(9))];
        assert!(c.interactions(&[chain.clone()]).is_empty());
        assert!(c.interactions(&[chain, vec![stick(-0.4, 7.5, 0.02)]]).iter().all(|p| !p.3), "far apart: cells in common only through a node");
        // one fronts the scene group's member; the other, elsewhere in the view, interacts with it only by joining
        let beside = stick(0.15, 4.0, 0.02);
        let pairs = |g: Option<u32>| c.interactions(&[vec![piece_of(-0.15, -0.05, None)], vec![Element { group: g, ..beside }]]);
        assert!(pairs(Some(5)).iter().any(|p| p.3), "{:?}", pairs(Some(5)));
        assert!(pairs(None).iter().all(|p| !p.3), "{:?}", pairs(None));
    }

    /// A bare layer for the pulse's tests: at range `t`, fill `f` (at every section radius), opacity `op`, face-on.
    fn layer(t: f64, f: f64, op: f64) -> Layer {
        Layer { t, far: t + 0.01, spread: 0.0, f, fq: [f; 3], cos: 1.0, db: Settings::default().refl_wood, elem: 0, op, x: 0.0, s: 0.0 }
    }

    /// One cell's pulse on a ray along y, `ground` (range, cos) behind it, against a medium `ext`; `capped` when it
    /// returned all the echoes it could.
    fn pulse_of(echoes: &[f32], refls: &[f32], capped: bool, layers: &[Layer], ground: Option<(f64, f64)>, ext: &Foliage) -> Pulse {
        pulse_in(&Settings::default(), echoes, refls, capped, layers, ground, ext)
    }

    /// [`pulse_of`] under settings `em`.
    fn pulse_in(em: &Settings, echoes: &[f32], refls: &[f32], capped: bool, layers: &[Layer], ground: Option<(f64, f64)>, ext: &Foliage) -> Pulse {
        let cap = if capped { echoes.len() } else { usize::MAX };
        Pulse::new(occluders(layers.to_vec(), ground), 1, &[0.0; 3], &[0.0, 1.0, 0.0], echoes, refls, cap, em, ext)
    }

    /// One cell's log-likelihood on a ray along y against a medium `ext`.
    fn ll_of(echoes: &[f32], refls: &[f32], capped: bool, layers: &[Layer], ext: &Foliage) -> f64 {
        pulse_of(echoes, refls, capped, layers, None, ext).read(0.0, 0.0, &Settings::default()).ll
    }

    /// Slice 2c: a partial edge leaves beam for the medium behind it, so a later echo there, as dim as that beam,
    /// counts for the edge against an opaque element at the same place, and a pulse at its cap says nothing past its
    /// last echo.
    #[test]
    fn a_later_echo_behind_a_partial_edge_reads_the_medium() {
        let em = Settings::default();
        let flat = Foliage::uniform(0.05);
        let edge = 10.0 * (0.3f64).log10();
        let (first, later) = ((em.refl_wood + edge) as f32, (em.refl_background_mean + 10.0 * (0.7f64).log10()) as f32);
        // the later echo's gain: with it against without it (the pulse not capped), each hypothesis's own first echo
        let gain = |f: f64, first: f32| ll_of(&[4.0, 7.0], &[first, later], false, &[layer(4.0, f, 1.0)], &flat) - ll_of(&[4.0], &[first], false, &[layer(4.0, f, 1.0)], &flat);
        // a full hit at the same place leaves no beam for it
        let (edge_gain, full_gain) = (gain(0.3, first), gain(1.0, em.refl_wood as f32));
        assert!(edge_gain > full_gain + 3.0, "edge {edge_gain} full {full_gain}");
        // capped after the first echo: a stick 6 m behind it (past any merge), never seen, costs nothing; uncapped it does
        let behind = [layer(4.0, 0.3, 1.0), layer(10.0, 1.0, 1.0)];
        let one = [layer(4.0, 0.3, 1.0)];
        assert!((ll_of(&[4.0], &[first], true, &behind, &flat) - ll_of(&[4.0], &[first], true, &one, &flat)).abs() < 1e-6);
        assert!(ll_of(&[4.0], &[first], false, &behind, &flat) < ll_of(&[4.0], &[first], false, &one, &flat) - 1.0);
    }

    /// Slice 2c: a dense medium just behind a partial edge merges with its return (the merge share is high, and none
    /// behind an opaque full hit); the merged echo is brighter by the medium's energy and pulled back by the measured
    /// pull, and an echo so read is likelier with the merge than without it. At slice 2c's map and pull (the box's,
    /// which lean on the field): the second box's (slices 2d, 9) lean on range and pull too little to read here.
    #[test]
    fn a_merge_with_the_medium_reads_the_measured_pull() {
        let em = Settings { pull_a: 1.2, pull_b: 0.075, merge_a: -1.7258, merge_field: 14.9334, merge_range: -0.0844, merge_fill: 1.9841, ..Settings::default() };
        let spec = FoliageBox { lo: [-0.5, 4.0, -0.5], hi: [0.5, 5.0, 0.5], voxel: 1.0, margin: 0.0 };
        let dense = Foliage { kappa: 0.05, spec: Some(spec), post: HashMap::from([([0, 0, 0], (5.0, 1.0))]) };
        let merge = |f: f64, ext: &Foliage| pulse_in(&em, &[], &[], false, &[layer(4.0, f, 1.0)], None, ext).occluders[0].merge;
        let p = pulse_in(&em, &[], &[], false, &[layer(4.0, 0.3, 1.0)], None, &dense);
        let m = p.occluders[0].merge;
        assert!(m.share > 0.9 && m.share < 1.0, "{m:?}");
        assert_eq!(merge(1.0, &dense).share, 0.0);
        assert!(merge(0.3, &Foliage::uniform(0.05)).share < 0.5 * m.share);
        let r = p.receiver(0.0, 0.0, &em).patterns[0].1[0];
        let x = 10.0 * (10f64.powf(r.db[1] / 10.0) + m.energy).log10();
        assert!(m.pull > 0.0 && m.pull < 0.5, "{m:?}");
        let at = r.mu + m.pull;
        let plain = [Occluder { merge: Merge::none(), ..p.occluders[0] }];
        let (merged, alone) = (r.density(at, x, &em, &p.occluders), r.density(at, x, &em, &plain));
        assert!(merged > 1.5 * alone, "{merged} vs {alone}");
        // two layers within the resolution merge into one echo the measured pull behind the nearer
        let pair = receiver(&occluders(vec![layer(4.0, 0.5, 1.0), layer(4.3, 1.0, 1.0)], None), 1, 0.0, 0.0, &em).patterns;
        let merged = pair.iter().find(|p| p.1.len() == 1).unwrap().1[0];
        let e = 10.0 * (0.5f64 / 0.5).log10();
        assert!((merged.mu - (4.0 + em.depth_bias(1, 4.0) + em.pull(0.3 + em.depth_bias(1, 4.3) - em.depth_bias(1, 4.0), e))).abs() < 1e-9, "{}", merged.mu);
        assert!(em.pull(0.3, e) < 0.1 * 0.3 && em.pull(0.3, 20.0) < em.pull(0.3, 0.0) && em.pull(0.3, -20.0) > 0.5 * 0.3);
    }

    /// Slice 2c: every cell is a ratio against one empty scene, the medium and the ground alone: a cell's empty-scene
    /// log-likelihood is the same whichever hypothesis fronts it.
    #[test]
    fn the_empty_scene_is_the_same_for_every_hypothesis() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let ext = Foliage::uniform(sc.kappa);
        let bases = |e: Element| -> HashMap<usize, f64> {
            let rendered = render_all(&v, &[e], &em);
            let d = elem_data(&v, &rendered, &[e], &[0], &[0], 0, &em, &sc, &ext);
            let cells: Vec<usize> = { let mut c: Vec<usize> = kbuffer(&rendered).keys().copied().collect(); c.sort_unstable(); c.retain(|&k| v.cell[k] != Cell::NoPulse); c };
            cells.into_iter().zip(d[0].cells.iter().map(|c| c.base)).collect()
        };
        let (a, b) = (bases(stick(0.0, 4.0, 0.03)), bases(Element { opacity: 0.4, ..stick(0.005, 4.05, 0.045) }));
        let common: Vec<&usize> = a.keys().filter(|k| b.contains_key(k)).collect();
        assert!(common.len() > 20);
        assert!(common.iter().all(|k| a[k] == b[k]));
    }

    /// Slice 2d: the ground plane's echo is read at `ground_sd / cos ι` along the ray and recorded with chance
    /// `ground_echo`. On a grazing ray (cos 0.2) the ground term reads an echo 1.5 m before the plane (terrain 0.3 m
    /// over it), which the per-echo noise alone can't; and a pulse that returns nothing pays about `1 − ground_echo`,
    /// not a near-certain echo's absence.
    #[test]
    fn the_ground_reads_its_misfit_along_the_ray() {
        let em = Settings::default();
        let flat = Foliage::uniform(0.05);
        let (o, w) = ([0.0; 3], [0.0, 0.96f64.sqrt(), -0.2]);
        let ll = |echoes: &[f32], refls: &[f32], em: &Settings| {
            Pulse::new(occluders::<f64>(Vec::new(), Some((10.0, 0.2))), 1, &o, &w, echoes, refls, usize::MAX, em, &flat).read(0.0, 0.0, em).ll
        };
        let x = (em.refl_ground + 10.0 * 0.2f64.powf(em.refl_incidence).log10()) as f32;
        let tight = Settings { ground_sd: 0.0, ..em };
        let (spread, plane) = (ll(&[8.5], &[x], &em), ll(&[8.5], &[x], &tight));
        assert!(spread > plane + 3.0, "{spread} vs {plane}");
        let medium = -0.05 * 10.0;
        let (none, certain) = (ll(&[], &[], &em), ll(&[], &[], &Settings { ground_echo: 1.0, ..em }));
        assert!((none - medium - (1.0 - em.ground_echo).ln()).abs() < 1e-3, "{none}");
        assert!(certain < none - 20.0, "{certain}");
    }

    /// Slice 2d: a group's information loads one stem front shift with the prior its cost gives it (its members' mean
    /// sd: [`term_nodes`]), not each member's own.
    #[test]
    fn a_groups_information_and_cost_share_its_front_shift_prior() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [Element { r: 0.03, ..piece_of(-0.2, 0.0, Some(3)) }, Element { r: 0.015, ..piece_of(0.0, 0.2, Some(3)) }];
        let rendered = render_all(&v, &es, &em);
        let d = elem_data(&v, &rendered, &es, &[0, 0], &[0, 1], 0, &em, &sc, &Foliage::uniform(sc.kappa));
        assert_eq!(d.len(), 1);
        let shared = Shared::new(1, &es, &em);
        for (k, e) in es.iter().enumerate() {
            assert!((e.front_shift_sd(&em) - d[0].own_sd[0]).abs() > 1e-4);
            assert_eq!(shared.load(k, 0.0, 1.0, 0.0, 0.0).last().unwrap().1, d[0].own_sd[0]);
        }
    }

    /// Free space (slice 7): in front of the pole every ray clears a voxel; within the receiver's resolution in front of
    /// an echo none does, though the rays reach it; behind the pole its rays are in shadow; a ray crosses the voxels
    /// its line meets whatever its echo.
    #[test]
    fn free_space_carves_only_past_the_resolution() {
        let v = pole_image();
        let em = Settings::default();
        let [crossed, reached, clear] = foliage::free_space(&v, [-0.1, 1.0, -0.1], [0.1, 8.6, 0.1], 0.2, &em);
        assert_eq!(crossed.len(), 38);
        let at = |y: f64| ((y - 1.0) / 0.2).floor() as usize;
        let near = at(2.1);
        assert!(crossed[near] > 100 && clear[near] == crossed[near], "{} {}", crossed[near], clear[near]);
        let front = at(4.0 - 0.03 - 0.5 * em.resolution()); // within the resolution in front of the pole
        assert!(reached[front] == crossed[front] && clear[front] < crossed[front], "{} {} {}", crossed[front], reached[front], clear[front]);
        let behind = at(5.1);
        assert!(reached[behind] < crossed[behind] && clear[behind] > 0, "{} {} {}", crossed[behind], reached[behind], clear[behind]);
        let wall = at(8.0 - 0.5 * em.resolution());
        assert!(clear[wall] == 0 && reached[wall] > 0, "{} {}", reached[wall], clear[wall]);
        assert!((0..38).all(|i| clear[i] <= reached[i] && reached[i] <= crossed[i]));
    }

    /// Pulses for the pass's tests: every cell two sticks front in the pole's scan, and bare ones with a merge, a
    /// dense medium behind a partial edge, the cap, the ground and a miss.
    fn test_pulses() -> Vec<Pulse> {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [stick(0.0, 4.0, 0.03), Element { opacity: 0.4, ..stick(0.03, 3.9, 0.045) }];
        let rendered = render_all(&v, &es, &em);
        let data = elem_data(&v, &rendered, &es, &[0, 1], &[0, 1], 0, &em, &sc, &Foliage::uniform(sc.kappa));
        let mut out: Vec<Pulse> = data.into_iter().flat_map(|d| d.cells.into_iter().flat_map(|c| c.pulses.into_iter().map(|p| p.1))).collect();
        let spec = FoliageBox { lo: [-0.5, 4.0, -0.5], hi: [0.5, 5.0, 0.5], voxel: 1.0, margin: 0.0 };
        let dense = Foliage { kappa: 0.05, spec: Some(spec), post: HashMap::from([([0, 0, 0], (5.0, 1.0))]) };
        let (flat, wood) = (Foliage::uniform(0.05), em.refl_wood as f32);
        out.push(pulse_of(&[4.0, 4.3, 7.0], &[wood - 5.0, wood - 3.0, -20.0], false, &[layer(4.0, 0.5, 1.0), layer(4.3, 1.0, 1.0)], None, &flat));
        out.push(pulse_of(&[4.02, 4.4], &[wood - 5.0, -18.0], true, &[layer(4.0, 0.3, 1.0), layer(10.0, 1.0, 1.0)], None, &dense));
        out.push(pulse_of(&[3.0, 8.5], &[-25.0, -20.0], false, &[layer(4.0, 0.3, 0.6)], Some((10.0, 0.2)), &flat));
        out.push(pulse_of(&[], &[], false, &[layer(4.0, 0.3, 1.0)], Some((10.0, 0.2)), &dense));
        out
    }

    /// A pulse that returned nothing through an opaque full hit of wood at 4 m, far above the detection floor
    /// (z ≈ 12.7): its likelihood is the chance the hit went unrecorded, Φ(−z), smooth in the reflectance level, never
    /// the 1e-300 floor that `1 − p_detect` cancels to (3dtrees research 130 rca-level-modes.md).
    #[test]
    fn a_miss_through_a_bright_full_hit_reads_the_detection_tail() {
        let em = Settings::default();
        let p = pulse_of(&[], &[], false, &[layer(4.0, 1.0, 1.0)], None, &Foliage::uniform(0.0));
        let z = |level: f64| (em.refl_wood + level - em.floor(4.0)) / em.detect_sd;
        for level in [-1.0, -0.5, 0.0, 0.5] {
            let ll = p.read(0.0, level, &em).ll;
            let tail = (0.5 * libm::erfc(z(level) / std::f64::consts::SQRT_2)).ln();
            assert!((ll - tail).abs() < 1e-6 * tail.abs(), "level {level}: ll {ll}, ln Φ(−z) {tail}");
        }
    }

    /// Each echo's responsibilities over the pulse's owners (members, the medium, an outlier) sum to one; an echo on
    /// an opaque full hit is the layer's, and one far behind a sliver, in a medium, the medium's.
    #[test]
    fn responsibilities_sum_to_one_per_echo() {
        let em = Settings::default();
        for p in test_pulses() {
            let r = p.read(0.0, 0.0, &em);
            let n = r.rx.members.len() + 2;
            assert_eq!(r.resp.len(), p.echoes.len() * n);
            for row in r.resp.chunks(n) {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9 && row.iter().all(|&x| x >= 0.0), "{row:?}");
            }
        }
        let flat = Foliage::uniform(0.05);
        let on = (4.0 + em.depth_bias(1, 4.0)) as f32;
        let r = pulse_of(&[on], &[em.refl_wood as f32], false, &[layer(4.0, 1.0, 1.0)], None, &flat).read(0.0, 0.0, &em);
        assert!(r.owned(0) > 0.99, "{:?}", r.resp);
        let r = pulse_of(&[on, 7.0], &[em.refl_wood as f32 - 13.0, -20.0], false, &[layer(4.0, 0.05, 1.0)], None, &flat).read(0.0, 0.0, &em);
        assert!(r.resp[3 + 1] > 0.9, "{:?}", r.resp);
    }

    /// The receiver's expected echoes match a Monte Carlo of the receiver: each adjacent pair of members resolved
    /// with its chance, else merged; each return recorded with its chance (any of its members); the first `cap` kept.
    #[test]
    fn expected_returns_match_a_monte_carlo_of_the_receiver() {
        let em = Settings::default();
        let cases = [
            (vec![layer(4.0, 0.5, 1.0), layer(4.3, 1.0, 1.0)], None, usize::MAX),
            (vec![layer(4.0, 0.3, 0.6), layer(4.05, 0.4, 1.0), layer(6.0, 1.0, 1.0)], None, 2),
            (vec![layer(4.0, 0.2, 1.0)], Some((4.4, 0.9)), usize::MAX),
            (vec![layer(4.0, 0.6, 1.0)], Some((10.0, 0.2)), 1),
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for (layers, ground, cap) in cases {
            let occ = occluders(layers, ground);
            let rx = receiver(&occ, 1, 0.0, 0.0, &em);
            let m = &rx.members;
            let n = 200_000;
            let mut count = 0;
            for _ in 0..n {
                let (mut recorded, mut none) = (0usize, 1.0);
                for j in 0..m.len() {
                    none *= 1.0 - m[j].s;
                    let ends = j + 1 == m.len() || {
                        let e = ((m[j].energy.max(1e-300) / m[j + 1].energy.max(1e-300)).log10() * 10.0).clamp(-60.0, 60.0);
                        uniform() < em.p_resolve(occ[j + 1].t - occ[j].t, e)
                    };
                    if ends {
                        recorded += (uniform() < 1.0 - none) as usize;
                        none = 1.0;
                    }
                }
                count += recorded.min(cap);
            }
            let (mc, expected) = (count as f64 / n as f64, rx.expected(cap).iter().sum::<f64>());
            assert!((mc - expected).abs() < 0.01, "{} members, cap {cap}: Monte Carlo {mc} against {expected}", m.len());
        }
    }
}
