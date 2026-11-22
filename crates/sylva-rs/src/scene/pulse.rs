// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One cell's pulse against its owners (3dtrees research 130): the occluders, the receiver, the medium, and the one
//! pass that reads it.
//!
//! - **The ray likelihood, one cell:** its pulse whole (3dtrees research 130 slice 2c), against occluders (the
//!   layers in depth order, then a ground plane) and a medium (the foliage field; κ, the background, outside its
//!   box): after any return, the next surface is the medium's ([`Pass::lik`]). A pulse at the target cap says
//!   nothing past its last echo.
//!   - **One pass per pulse** ([`Pulse::read`]; 3dtrees research 130 formulation.md, slice-pulse.md): the cell's
//!     owners (its layers, the ground, the medium) and the receiver ([`Receiver`]: merges, detection, the cap) are read
//!     once, and that pass gives everything the scene reads of the pulse: its log-likelihood with its slope and
//!     curvature in the own nodes ([`Dual2`]; with the geometry, its slope in [`Scene::information`]: [`Dual`]), the element's echo ([`front_echo`]), each owner's beam left and
//!     expected echoes, and each echo's responsibilities over the owners. With a foliage field, the responsibilities
//!     at each term's nodes share each echo between the field and the elements (soft attribution: the term's deposit,
//!     [`Term`]'s `deposit`; 3dtrees research 130 slice-soft).
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is recorded with
//!     `op_i det_i`, where `det_i` is the chance it clears the detection floor ([`Settings::p_detect`]):
//!     `Φ((ρ_dB + 10 log10(e_i cos ι_i) − floor(t_i)) / σ)`.
//!   - Adjacent returns (layers and the ground) resolve into two echoes with the instrument's
//!     [`Settings::p_resolve`] of their separation and energy ratio; else they merge into one, recorded if any
//!     member is, pulled back from the nearer's depth by the measured pull ([`Settings::pull`]; energy: e · cos ·
//!     reflectance); they merge only where both are there (chance op_i op_j). A cell's likelihood is the mixture over these merge patterns.
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
//!     among its layers' terms by the chance each is the nearest ([`super::render::front_shares`]). So no fill, and no layer passing
//!     in front of another as it enters, steps the cost (a hard entry at 1 % did, by up to tenths of a nat on the
//!     stand); where the receiver cut (a merge with the medium behind under 1e-6 of the beam, a merge pattern under
//!     1e-4) it fades ([`smooth`]).
//! - **The element's echo** ([`front_echo`]): the first return of the most likely way a cell comes back, its sd
//!   widened by the alignment error and the stem front shift. [`Outcome::of`] against it is the one rule for "the
//!   cell's first echo is the element's", for reporting only: the cells_on, arcs and rows' parts, `predict` and
//!   `counts` (and, for any echo of the pulse, `predict`'s `read`); the cost never reads it. A cell mostly filled
//!   ([`MOSTLY`]) is where an echo reads the element's surface.

use crate::Point;
use super::dual::{Over, Real};
use super::instrument::Instrument;
use super::foliage::{Foliage, Medium};
use super::range_image::{Cell, RangeImage};
use super::reading::{Reading as FillReading, ReadingModel};
use super::render::{Layer, smooth, mixture};
use super::settings::{Background, Settings, P_OUTLIER, T_MAX};

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
        } else if obs < depth - 3.0 * sd - 0.03 {
            Outcome::InFront
        } else if (obs - depth).abs() < 3.0 * sd + 0.01 {
            Outcome::On
        } else {
            Outcome::Behind
        }
    }
}

pub(super) fn t_pdf<T: Real>(x: T, nu: f64) -> T {
    let c = super::settings::t_log_norm(nu);
    (-((x * x / nu + 1.0).ln() * (0.5 * (nu + 1.0))) + c).exp()
}

/// A cell's log-likelihood at shift `shift` and reflectance level `level`: the log of its mixture's chances times its
/// subsets' likelihoods ([`Pulse::read`]); a cell with one subset is that subset's.
pub(super) fn mixture_ll<P: Real, T: Over<P>>(pulses: &[(P, Pulse<P>)], shift: T, level: T, em: &Settings) -> T {
    if let [(_, p)] = pulses {
        return p.read(shift, level, em).ll;
    }
    let lls: Vec<T> = pulses.iter().map(|(_, p)| p.read(shift, level, em).ll).collect();
    let m = lls.iter().map(|l| l.v()).fold(f64::NEG_INFINITY, f64::max);
    pulses.iter().zip(&lls).map(|((w, _), &l)| (l - m).exp() * *w).sum::<T>().ln() + m
}

/// A cell's first echo (NaN for none) and the ground along its ray (range, cos), if the background has one.
pub(super) fn obs_ground(v: &RangeImage, cell: usize, sc: &Background) -> (f64, Option<(f64, f64)>, Point) {
    let obs = if v.cell[cell] == Cell::Echoes { v.range[cell] as f64 } else { f64::NAN };
    let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
    let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
    (obs, ground, w)
}

/// The range a cell's weight is read at ([`Settings::cell_weight`]), from the data alone, the same for every
/// hypothesis: its first echo `obs`; for a pulse with no echo, the nearest first echo among its eight neighbouring
/// cells (where its footprint overlaps theirs on the surface beside it), else the `ground`'s range, else [`T_MAX`].
/// A stand-in until a shared misfit per element replaces per-cell weights (3dtrees research 130 slice-miss-weight:
/// read where the beam ends, a miss beside an element at 3 m counted as about 26 echoes there; read at the layers'
/// ranges before slice-s5, it moved with the hypothesis and jumped as a faint layer entered).
pub(super) fn weight_range(v: &RangeImage, cell: usize, obs: f64, ground: Option<(f64, f64)>) -> f64 {
    if obs.is_finite() {
        return obs;
    }
    let (l, c, nl, nc) = ((cell / v.n_cols) as i64, (cell % v.n_cols) as i64, v.n_lines as i64, v.n_cols as i64);
    let mut near = f64::INFINITY;
    for (dl, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
        let (ll, cc) = (l + dl, c + dc);
        if ll >= 0 && cc >= 0 && ll < nl && cc < nc {
            let j = (ll * nc + cc) as usize;
            if v.cell[j] == Cell::Echoes {
                near = near.min(v.range[j] as f64);
            }
        }
    }
    if near.is_finite() { near } else { ground.map_or(T_MAX, |g| g.0) }
}

/// A cell as the likelihood reads it: its pulse per subset of its mixture with that subset's chance ([`mixture`]; the
/// one with every layer in first), the same pulse in the empty scene (the medium and the ground alone: the same for
/// every hypothesis), its first echo (NaN for none) and its weight ([`Settings::cell_weight`] at [`weight_range`]), in
/// the layers' numbers `P`.
pub(super) struct CellView<P = f64> {
    pub(super) obs: f64,
    pub(super) wt: P,
    pub(super) pulses: Vec<(P, Pulse<P>)>,
    pub(super) empty: Pulse<P>,
}

/// Cell `cell` of image `v` with `layers` in front of it (nearest first, [`kept`]), read at the cell's surface class
/// ([`cell_class`]), against the foliage field `ext` and the background's ground.
pub(super) fn cell_view<P: Real>(v: &RangeImage, cell: usize, layers: Vec<Layer<P>>, em: &Settings, sc: &Background, ext: &Foliage) -> CellView<P> {
    let (obs, ground, w) = obs_ground(v, cell, sc);
    let wt = em.cell_weight(P::c(weight_range(v, cell, obs, ground)), v.steps(cell));
    let (m, class, slots) = (ext.medium(&v.origin, &w), cell_class(v, cell, em), v.slots(cell));
    let pulses = mixture(&layers).into_iter()
        .map(|(p, ls)| (p, Pulse::on(occluders(ls, ground), class, &m, v.echoes(cell), v.echo_refls(cell), slots, &v.instrument, em))).collect();
    let empty = Pulse::on(occluders(Vec::new(), ground), class, &m, v.echoes(cell), v.echo_refls(cell), slots, &v.instrument, em);
    CellView { obs, wt, pulses, empty }
}

/// Cell `cell`'s surface class: its echo's surface continuity (the excess, data), mid where it has no surface fit.
/// Every return of the cell's pulse reads that class's depth bias and noise (3dtrees research 130 optimisation.md).
pub(super) fn cell_class(v: &RangeImage, cell: usize, em: &Settings) -> u8 {
    let x = v.excess[cell];
    if v.cell[cell] == Cell::Echoes && x.is_finite() { em.surface(x as f64) } else { 1 }
}

/// The front element's echo in a cell as the likelihood reads it: the first return of the most likely way the cell's
/// occluders come back ([`Receiver::front`]; none without an occluder), its sd widened by the alignment error's spread
/// at the front element's term (`spread`: [`term_spreads`], the epoch's part in it) and the stem front shift's prior
/// spread (the front element's, `front_sd`). [`Outcome::of`] against it is the one rule for "the cell's first echo
/// is the element's".
pub(super) fn front_echo(rx: &Receiver<f64>, spread: f64, front_sd: f64) -> Option<Return> {
    rx.front().map(|f| Return { sd: f.sd.hypot(spread).hypot(front_sd), ..f })
}

/// The depth and sd of an element's echo (NaN for none), as [`Outcome::of`] reads it.
pub(super) fn depth_sd(fe: Option<&Return>) -> (f64, f64) {
    fe.map_or((f64::NAN, f64::NAN), |f| (f.mu, f.sd))
}

/// What lies behind a layer in the medium (the foliage field; κ outside its box), at unshifted ranges ([`merges`]):
/// the share of pulses whose return from it merges with a medium surface behind it (`share`, the site's map of the
/// field's own share `field`, [`Settings::merge_share`]), that surface's energy share when it does (the beam the
/// layer leaves, at the medium's mean reflectance), and how far the merge pulls the echo back ([`Settings::pull`], at
/// the merged surface's mean separation).
#[derive(Debug, Clone, Copy)]
pub(super) struct Merge<P = f64> {
    pub(super) share: P,
    pub(super) field: f64,
    pub(super) energy: P,
    pub(super) pull: P,
}

impl<P: Real> Merge<P> {
    /// No merge.
    pub(super) fn none() -> Self {
        Merge { share: P::c(0.0), field: 0.0, energy: P::c(0.0), pull: P::c(0.0) }
    }
}

/// What a pulse meets as a surface: an element's layer, or the ground plane, at the cosine of incidence along the ray.
#[derive(Debug, Clone, Copy)]
pub(super) enum Surface<P = f64> {
    Layer(Layer<P>),
    Ground(f64),
}

/// One of a pulse's occluders, nearest first (3dtrees research 130 formulation.md: the owners that deposit energy
/// along the ray): its surface, its range, the foliage field's optical depth to it and its merge with the medium behind
/// it ([`merges`]; none for the ground). The ground, where the background has one, is the last: the beam ends there.
#[derive(Debug, Clone, Copy)]
pub(super) struct Occluder<P = f64> {
    pub(super) surface: Surface<P>,
    pub(super) t: P,
    pub(super) tau: P,
    pub(super) merge: Merge<P>,
    /// The detection floor (dB) at it, the scan's instrument's ([`Pulse::on`] reads it).
    pub(super) floor: P,
}

/// A cell's occluders: its `layers` (nearest first) in front of the ground at `ground` = (range, cos), then the
/// ground; no optical depths or merges yet ([`Pulse::on`] reads them).
pub(super) fn occluders<P: Real>(layers: Vec<Layer<P>>, ground: Option<(f64, f64)>) -> Vec<Occluder<P>> {
    let t_end = ground.map_or(T_MAX, |g| g.0);
    let bare = |surface, t| Occluder { surface, t, tau: P::c(0.0), merge: Merge::none(), floor: P::c(f64::NEG_INFINITY) };
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
pub(super) struct Return<P = f64, T = P> {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) mu: T,
    pub(super) sd: T,
    pub(super) s: T,
    pub(super) q: T,
    pub(super) s_q: [T; 3],
    pub(super) db: [T; 3],
    pub(super) fill: [P; 3],
    /// The detection floor (dB) at its nearest member.
    pub(super) floor: P,
}

impl<P: Real, T: Over<P>> Return<P, T> {
    /// Its echo's reflectance (dB) at its mean energy share over the shape mismatch's three section radii (their rule
    /// weights; the energy share is linear in the fill, so this is the echo at the mean fill). [`Scene::information`]
    /// reads it, not the middle radius's: a radius's share grows from 0 as the chord where its section enters the beam,
    /// smoothly, but its reflectance as `10 log10` of it, whose slope is unbounded there.
    pub(super) fn reflectance(&self) -> T {
        ((self.db[0] / 10.0).exp10() + (self.db[1] / 10.0).exp10() * 4.0 + (self.db[2] / 10.0).exp10()).log10() * 10.0 - 10.0 * 6f64.log10()
    }

    /// The density of its echo's reflectance `x` (dB) as recorded ([`Settings::refl_density`], at its floor), with
    /// `add` more energy share merged into it (a medium surface's, read at the medium's spread for its part): a mixture
    /// over the shape mismatch's three section radii, each weighted by its rule weight and its chance of being recorded
    /// (a radius whose section misses the beam records nothing, so it says nothing of the reflectance), each widened by
    /// the reflectance's spread between adjacent radii. Three sharp points would leave gaps between them that a
    /// reflectance noise of tenths of a dB reads as misfits.
    pub(super) fn refl_density(&self, x: f64, em: &Settings, add: P) -> T {
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
        (0..3).filter(|&q| w[q].v() > 0.0).map(|q| w[q] * em.refl_density(x, db[q], spread.hypot(medium(q)), self.fill[q], self.floor)).sum::<T>() / total
    }

    /// The density of its echo at range `o` with reflectance `x` (dB; NaN unknown), its members among `occ`: its own
    /// echo, or, in its merge share, merged with the medium surface behind it, pulled back by the measured pull
    /// ([`Settings::pull`], its sd as large) and brighter by that surface's energy (3dtrees research 130 slice 2c: a
    /// mixture inside the return, never another return).
    pub(super) fn density(&self, o: f64, x: f64, em: &Settings, occ: &[Occluder<P>]) -> T {
        let (t, merge) = (occ[self.first].t, occ[self.last].merge);
        let alone = t_pdf((T::c(o) - self.mu) / self.sd, em.noise_dof) / self.sd * self.refl_density(x, em, P::c(0.0));
        let m = merge.share;
        if m.v() <= 0.0 {
            return alone;
        }
        let sd = self.sd.hypot(T::from(merge.pull));
        alone * (P::c(1.0) - m) + t_pdf((T::c(o) - self.mu - merge.pull) / sd, em.noise_dof) * m / sd * self.refl_density(x, em, merge.energy)
    }
}

/// What the receiver reads of one occluder: its echo's depth and sd, the chance it is recorded and (`q`, without
/// cancellation) not, its energy (opacity × the beam it meets × cos ι × reflectance), its energy share when it is hit
/// and the share of the beam it meets, at the shape mismatch's three section radii (the share without the opacity; its
/// echo's reflectance is `10 log10` of it), and the beam left behind it.
pub(super) struct Member<P, T> {
    pub(super) depth: T,
    pub(super) sd: P,
    pub(super) s: T,
    pub(super) q: T,
    pub(super) s_q: [T; 3],
    pub(super) energy: T,
    pub(super) op: f64,
    pub(super) share: [T; 3],
    pub(super) fill: [P; 3],
    pub(super) beam: P,
    /// Its reflectance (dB), the beam's share it meets times cos ι at the three section radii, the chance it returns
    /// when met (its opacity; the ground's `ground_echo`) and its detection floor: its detection re-read in a dimmer
    /// beam ([`Pass::dim`]).
    pub(super) refl: T,
    pub(super) lit: [P; 3],
    pub(super) rec: f64,
    pub(super) floor: P,
}

/// What the receiver makes of a pulse's occluders (3dtrees research 130 formulation.md: the owners deposit energy
/// along the ray, the receiver turns it into echoes): its `members`, one per occluder, and the ways they come back
/// (`patterns`, each a set of returns, nearest first, and its chance) ([`receiver`]).
pub(super) struct Receiver<P = f64, T = P> {
    pub(super) members: Vec<Member<P, T>>,
    pub(super) patterns: Vec<(T, Vec<Return<P, T>>)>,
}

impl<P: Real, T: Over<P>> Receiver<P, T> {
    /// The width of a pulse's responsibility row: each member, then the medium and an outlier.
    pub(super) fn owners(&self) -> usize {
        self.members.len() + 2
    }

    /// The first return of the most likely pattern (none without an occluder).
    pub(super) fn front(&self) -> Option<Return<P, T>> {
        self.patterns.iter().max_by(|a, b| a.0.v().total_cmp(&b.0.v())).and_then(|p| p.1.first().copied())
    }

    /// Per member, the expected number of echoes the receiver records of the returns it heads (the nearest member of a
    /// merged return makes its echo), the first `cap` of a pulse's returns kept (the target cap; 0 for none).
    pub(super) fn expected(&self, cap: usize) -> Vec<f64> {
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
pub(super) fn receiver<P: Real, T: Over<P>>(occ: &[Occluder<P>], k: u8, shift: T, level: T, em: &Settings) -> Receiver<P, T> {
    let mut beam = P::c(1.0);
    let m: Vec<Member<P, T>> = occ.iter().map(|o| match o.surface {
        Surface::Layer(ly) => {
            let db = level + ly.db;
            let cos = ly.cos.powf(em.refl_incidence);
            let s_q = ly.fq.map(|f| em.p_detect(db, f * beam * cos, o.floor) * ly.op);
            let s = (s_q[0] + s_q[1] * 4.0 + s_q[2]) / 6.0;
            let q_q = ly.fq.map(|f| em.p_miss(db, f * beam * cos, o.floor) * ly.op + (1.0 - ly.op));
            let q = (q_q[0] + q_q[1] * 4.0 + q_q[2]) / 6.0;
            let (b0, unit) = (beam, (db / 10.0).exp10() * (beam * cos));
            let share = ly.fq.map(|f| unit * f);
            beam *= P::c(1.0) - ly.f * ly.op;
            Member { depth: shift + (ly.t + em.depth_bias(k, ly.t)), sd: em.depth_sd(k, ly.spread), s, q, s_q, energy: unit * ly.f, op: ly.op, share,
                     fill: ly.fq.map(|f| f * b0), beam, refl: db, lit: ly.fq.map(|f| f * b0 * cos), rec: ly.op, floor: o.floor }
        }
        Surface::Ground(cos) => {
            let cg = cos.powf(em.refl_incidence);
            let share = beam * cg * 10f64.powf(em.refl_ground / 10.0);
            let s = em.p_detect::<P, P>(P::c(em.refl_ground), beam * cg, o.floor) * em.ground_echo;
            let q = em.p_miss::<P, P>(P::c(em.refl_ground), beam * cg, o.floor) * em.ground_echo + (1.0 - em.ground_echo);
            Member { depth: T::from(o.t), sd: P::c(em.noise(2).hypot(em.ground_sd / cos)), s: T::from(s), q: T::from(q), s_q: [T::from(s); 3],
                     energy: T::from(share * em.ground_echo), op: 1.0, share: [T::from(share); 3], fill: [beam; 3], beam: P::c(0.0),
                     refl: T::from(P::c(em.refl_ground)), lit: [beam * cg; 3], rec: em.ground_echo, floor: o.floor }
        }
    }).collect();
    let n = m.len();
    // each adjacent pair's chance of resolving into two echoes: they merge only where both are there (3dtrees research
    // 130 slice-s5: an element's opacity scaled its energy instead, so a faint one merged at a chance falling as a power
    // of its presence, about op^0.53, and a negligible one moved its neighbour's cost by as much)
    let pr: Vec<T> = (1..n).map(|j| {
        let e = (m[j - 1].energy.max(1e-300) / m[j].energy.max(1e-300)).log10() * 10.0;
        T::c(1.0) - (T::c(1.0) - em.p_resolve(occ[j].t - occ[j - 1].t, e.clamp(-60.0, 60.0))) * T::c(m[j - 1].op * m[j].op)
    }).collect();
    let group = |first: usize, last: usize| {
        let g = &m[first..=last];
        let db = [0, 1, 2].map(|q| g.iter().map(|x| x.share[q]).sum::<T>().max(1e-30).log10() * 10.0);
        let fill = [0, 1, 2].map(|q| g.iter().map(|x| x.fill[q]).sum::<P>().min(1.0));
        if g.len() == 1 {
            return Return { first, last, mu: g[0].depth, sd: T::from(g[0].sd), s: g[0].s, q: g[0].q, s_q: g[0].s_q, db, fill, floor: occ[first].floor };
        }
        // the merged echo follows the nearest member, pulled back by the rest
        let rest: T = g[1..].iter().map(|x| x.energy).sum();
        let d = if rest.v() > 0.0 { g[1..].iter().map(|x| x.energy * (x.depth - g[0].depth)).sum::<T>() / rest } else { g[g.len() - 1].depth - g[0].depth };
        let pull = em.pull(d, (g[0].energy.max(1e-300) / rest.max(1e-300)).log10() * 10.0);
        let none: T = g.iter().map(|x| x.q).product();
        let s_q = [0, 1, 2].map(|q| T::c(1.0) - g.iter().map(|x| T::c(1.0) - x.s_q[q]).product::<T>());
        Return { first, last, mu: g[0].depth + pull, sd: T::from(g[0].sd).hypot(pull), s: T::c(1.0) - none, q: none, s_q, db, fill, floor: occ[first].floor }
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
pub(super) fn merges<P: Real>(occ: &mut [Occluder<P>], m: &Medium, em: &Settings) {
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
        let field = P::c(1.0) - (-beam * whole).exp();
        let share = em.merge_share(field, ly.t, ly.f) * smooth(beam / 1e-6);
        occ[i].merge = Merge { share, field: field.v(), energy: beam * 10f64.powf(em.refl_background_mean / 10.0), pull: em.pull(d, e) };
    }
}

/// One cell's pulse and its owners, as the likelihood reads it (3dtrees research 130 formulation.md: one pass per
/// pulse): its echoes and their reflectances, the slots it had for them (`cap`: the programme's target cap less the
/// targets the import dropped, [`RangeImage::slots`]; a pulse at it is censored and says nothing past its last echo), its occluders ([`Occluder`]), the surface class
/// they are read at (its cell's: [`cell_class`]) and what its ray
/// meets of the foliage field besides: the optical depth to the beam's end (`tau_end`) and, per echo (`medium`), the
/// optical depth to it, σ at it and the optical depth to it plus the receiver's resolution, where the medium's path
/// restarts after it. [`Pulse::read`] is the one pass over it. Its occluders' geometry is in `P` (`f64`; a [`Dual`] in
/// [`Scene::information`]).
#[derive(Clone)]
pub(super) struct Pulse<P = f64> {
    pub(super) echoes: Vec<f32>,
    pub(super) refls: Vec<f32>,
    pub(super) cap: usize,
    pub(super) occluders: Vec<Occluder<P>>,
    pub(super) class: u8,
    pub(super) tau_end: f64,
    pub(super) medium: Vec<[f64; 3]>,
}

/// What one pass over a pulse gives ([`Pulse::read`]): its log-likelihood (with its derivatives as a [`Dual2`] or a
/// [`Dual`]), the receiver's reading of its occluders ([`Receiver`]: the most likely pattern's first return, each
/// member's beam left, chance recorded and expected echoes), and, per echo, its responsibilities over the pulse's
/// owners (the posterior over who made it): each occluder, by the return it heads, then the medium, then an outlier, a
/// row of `occluders + 2` summing to one.
pub(super) struct Reading<P, T> {
    pub(super) ll: T,
    pub(super) rx: Receiver<P, T>,
    pub(super) resp: Vec<f64>,
}

impl<P: Real, T: Over<P>> Reading<P, T> {
    /// Its responsibilities, a row per echo ([`Receiver::owners`] wide).
    pub(super) fn rows(&self) -> std::slice::Chunks<'_, f64> {
        self.resp.chunks(self.rx.owners())
    }

    /// Occluder `o`'s responsibilities summed over the echoes: how many of the pulse's echoes are its in expectation.
    pub(super) fn owned(&self, o: usize) -> f64 {
        self.rows().map(|r| r[o]).sum()
    }
}

impl<P: Real> Pulse<P> {
    /// The pulse against occluders `occ` along a ray already walked through the field (`m`, [`Medium`]).
    /// The ground's range and cos ι along the ray, if the ray meets it.
    pub(super) fn ground(&self) -> Option<(f64, f64)> {
        self.occluders.last().and_then(|o| match o.surface { Surface::Ground(cos) => Some((o.t.v(), cos.v())), _ => None })
    }

    pub(super) fn on(mut occ: Vec<Occluder<P>>, class: u8, m: &Medium, echoes: &[f32], refls: &[f32], cap: usize, inst: &Instrument, em: &Settings) -> Self {
        for x in occ.iter_mut() {
            x.tau = m.tau(x.t);
            x.floor = inst.floor(x.t);
        }
        merges(&mut occ, m, em);
        let medium = echoes.iter().map(|&r| [m.tau(r as f64), m.hazard(r as f64), m.tau(r as f64 + em.resolution())]).collect();
        Pulse { echoes: echoes.to_vec(), refls: refls.to_vec(), cap, tau_end: m.tau(Self::end(&occ)), occluders: occ, class, medium }
    }

    /// The range the beam of a pulse with occluders `occ` ends at: the ground's, else [`T_MAX`].
    pub(super) fn end(occ: &[Occluder<P>]) -> f64 {
        occ.last().filter(|o| matches!(o.surface, Surface::Ground(_))).map_or(T_MAX, |o| o.t.v())
    }

    pub(super) fn capped(&self) -> bool {
        !self.echoes.is_empty() && self.echoes.len() >= self.cap
    }

    /// The receiver's reading of its occluders, the layers shifted by `shift` along the beam and `level` in
    /// reflectance ([`receiver`]).
    pub(super) fn receiver<T: Over<P>>(&self, shift: T, level: T, em: &Settings) -> Receiver<P, T> {
        receiver(&self.occluders, self.class, shift, level, em)
    }

    /// The one pass (3dtrees research 130 slice 2c, formulation.md): the pulse whole against the returns its occluders
    /// give ([`receiver`], each with its merge with the medium behind it) and the medium along its ray, the layers
    /// shifted by `shift` along the beam and `level` in reflectance. The likelihood is the mixture over the merge
    /// patterns of [`Pass::lik`]'s recursion over the echoes; the responsibilities are each branch's share of it.
    pub(super) fn read<T: Over<P>>(&self, shift: T, level: T, em: &Settings) -> Reading<P, T> {
        let rx = self.receiver(shift, level, em);
        let owners = rx.owners();
        let mut resp = vec![0.0; self.echoes.len() * owners];
        let pass = Pass { pulse: self, rx: &rx, em, t_end: Self::end(&self.occluders), capped: self.capped(), owners, crown: FillReading::of(&ReadingModel::crown(em)),
                           low: FillReading::of(&ReadingModel::low(em)), ground: self.ground() };
        let total = rx.patterns.iter().map(|(p, rets)| *p * pass.lik(rets, 0, 0, P::c(1.0), 1.0, 1.0, 0.0, p.v(), &mut resp)).sum::<T>();
        if total.v() > 0.0 {
            resp.iter_mut().for_each(|r| *r /= total.v());
        }
        Reading { ll: total.max(1e-300).ln(), rx, resp }
    }

    /// Per echo, the share of it its elements' layers take in `r` (their responsibilities summed; the ground, the medium
    /// and an outlier take the rest).
    pub(super) fn taken<T: Over<P>>(&self, r: &Reading<P, T>) -> Vec<f64> {
        let layers: Vec<usize> = (0..self.occluders.len()).filter(|&o| matches!(self.occluders[o].surface, Surface::Layer(_))).collect();
        r.rows().map(|row| layers.iter().map(|&o| row[o]).sum()).collect()
    }

}

/// One pulse's echoes against one way its returns come back ([`Pulse::read`]): the receiver's members, and `owners`
/// columns per echo in the responsibilities.
pub(super) struct Pass<'a, P, T> {
    pub(super) pulse: &'a Pulse<P>,
    pub(super) rx: &'a Receiver<P, T>,
    pub(super) em: &'a Settings,
    pub(super) t_end: f64,
    pub(super) capped: bool,
    pub(super) owners: usize,
    /// The medium's echoes' reading of their objects' fills, crown leaf and low cover ([`FillReading`]), and the
    /// ground's range and cos ι along the ray, which sets an echo's class by its height.
    pub(super) crown: std::sync::Arc<FillReading>,
    pub(super) low: std::sync::Arc<FillReading>,
    pub(super) ground: Option<(f64, f64)>,
}

impl<P: Real, T: Over<P>> Pass<'_, P, T> {
    /// The reading of a medium's echo at range `o` by its class: low cover under [`super::reading::LOW_COVER`] above
    /// the terrain, else crown leaf.
    pub(super) fn reading_at(&self, o: f64) -> &FillReading {
        match self.ground {
            Some((t, cos)) if (t - o) * cos < super::reading::LOW_COVER => &self.low,
            _ => &self.crown,
        }
    }

    /// Return `r` met in `k` of the beam the scene left it (`taken`: the medium's echoes in front took the rest): each
    /// member's detection and miss re-read at that beam, and its reflectance `10 log10 k` dimmer. Itself at k = 1.
    pub(super) fn dim(&self, r: &Return<P, T>, k: f64) -> Return<P, T> {
        if k >= 1.0 {
            return *r;
        }
        let em = self.em;
        let parts: Vec<([T; 3], T)> = self.rx.members[r.first..=r.last].iter().map(|x| {
            let s_q = x.lit.map(|e| em.p_detect(x.refl, e * k, x.floor) * x.rec);
            let q_q = x.lit.map(|e| em.p_miss(x.refl, e * k, x.floor) * x.rec + (1.0 - x.rec));
            (s_q, (q_q[0] + q_q[1] * 4.0 + q_q[2]) / 6.0)
        }).collect();
        let db = r.db.map(|d| d + 10.0 * k.max(1e-30).log10());
        if let [(s_q, q)] = parts[..] {
            return Return { s: (s_q[0] + s_q[1] * 4.0 + s_q[2]) / 6.0, q, s_q, db, ..*r };
        }
        let none: T = parts.iter().map(|x| x.1).product();
        let s_q = [0, 1, 2].map(|i| T::c(1.0) - parts.iter().map(|x| T::c(1.0) - x.0[i]).product::<T>());
        Return { s: T::c(1.0) - none, q: none, s_q, db, ..*r }
    }

    /// The chance of echoes `i..` given returns `rets[j..]` not yet accounted for: the pulse meets occluders (the
    /// returns) and a medium (the foliage field), and after any echo the next surface is the medium's (3dtrees
    /// research 130 slice 2c; step 3c's reading, kept where sound). Each echo is one of those returns (the ones before
    /// it not recorded; [`Return::density`], its merge with the medium behind inside it), an echo of the medium (the
    /// returns before it not recorded) or an outlier.
    /// - **The medium's echoes** are a Poisson process along the ray in the beam the echoes so far left, `seen`: each
    ///   echo leaves the beam its reading says on average ([`FillReading::mean_left`]), as the field reads them
    ///   ([`super::foliage`]).
    ///   That beam is read from the echoes alone, the same for every hypothesis, so how often a pulse echoes again
    ///   doesn't lean on the scene's beam (on the stand's checks the scene's beam there favoured a wider stick
    ///   wherever a pulse returned nothing more: research 130 step 3c). Its path restarts after each echo at the
    ///   receiver's resolution (`tau0`, the optical depth there), as the field's fit reads it.
    /// - **The rest of the beam:** the medium's echo reads `10 log10` of the beam the scene leaves, `scene` (the share
    ///   the returns matched so far leave: the beam behind a return's last member) × `taken` (what the medium's own
    ///   echoes left), dimmer: behind an edge of fill f, `1 − f`, as a located edge reads f.
    /// - **Its object's fill is a latent** ([`FillReading`]; 3dtrees research 130 slice-dgp): read against that beam, the
    ///   echo's object either filled it, with chance π(u) (*stopped*: the beam is 0, and nothing behind records or says
    ///   anything, the returns nor the medium's path), or took part of it (*partial*: the beam goes on at the partial
    ///   fill's mean, [`FillReading::left`]). A later echo behind a stop can only be an outlier, so data with later echoes
    ///   falls to the partial branch by itself.
    /// - **Behind the medium's echoes:** a return is always *met* (no chance of being blocked: a medium echo is not a
    ///   stop), and met *dimmed*, in the beam the echoes in front left, `taken` of what the scene left ([`Pass::dim`]):
    ///   its detection, its miss and its brightness all read there, the ground's alike. So `taken` enters once, as the
    ///   beam; one that read as a full hit leaves none, and the returns behind it record nothing and say nothing.
    ///   (3dtrees research 130 slice-defects D1: `taken` was the chance a return is met, its detection and brightness
    ///   read at the elements' beam alone, so behind a leaf echo an element's edge was predicted brighter and more
    ///   surely recorded than it could be. The stopped/partial branch, t1-branch, will split `taken` into a chance of
    ///   stopping and the share an unstopped echo leaves.)
    ///
    /// After the last echo, the returns left and the medium's path to the beam's end record nothing, unless the pulse
    /// is capped: a capped pulse says nothing past its last echo, and only past it. A return left in front of the last
    /// echo (the outlier branch matches an echo without moving on) would have taken a slot first, so it still wasn't
    /// recorded; those past it, and the medium's path, are free (3dtrees research 130 slice-s5: every return left was
    /// let off, so on a capped pulse an outlier explained a later echo for free and hypotheses in front of it went
    /// unrefuted).
    ///
    /// `prefix` is the chance of the path to here (the pattern's and each earlier echo's branch); each branch adds
    /// `prefix` × its share × what follows to echo i's row of `resp`, under the owner it reads the echo as.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lik(&self, rets: &[Return<P, T>], i: usize, j: usize, scene: P, taken: f64, seen: f64, tau0: f64, prefix: f64, resp: &mut [f64]) -> T {
        let (em, p, row) = (self.em, self.pulse, i * self.owners);
        let occ = &p.occluders;
        if i == p.echoes.len() {
            if self.capped {
                let last = p.echoes.last().map_or(f64::INFINITY, |&r| r as f64);
                return rets[j..].iter().filter(|r| occ[r.first].t.v() < last).map(|r| self.dim(r, taken).q).product::<T>();
            }
            return rets[j..].iter().map(|r| self.dim(r, taken).q).product::<T>() * (-seen * (p.tau_end - tau0).max(0.0)).exp();
        }
        let (o, x) = (p.echoes[i] as f64, p.refls.get(i).map_or(f64::NAN, |&x| x as f64));
        let [tau_o, sig_o, tau_after] = p.medium[i];
        let mut total = T::c(0.0);
        // one of the returns
        let mut none = T::c(1.0);
        for (k, r) in rets.iter().enumerate().skip(j) {
            let r = &self.dim(r, taken);
            let d = none * ((occ[r.first].tau - tau0).max(0.0) * -seen).exp() * r.s * r.density(o, x, em, occ);
            if d.v() > 0.0 {
                let f = d * (1.0 - P_OUTLIER);
                let rest = self.lik(rets, i + 1, k + 1, self.rx.members[r.last].beam, taken, self.reading_at(o).beam_after(em, seen, x), tau_after, prefix * f.v(), resp);
                resp[row + r.first] += prefix * f.v() * rest.v();
                total += f * rest;
            }
            none *= r.q;
        }
        // the medium's
        if o < self.t_end && seen > 0.0 {
            let jn = (j..rets.len()).find(|&k| occ[rets[k].first].t.v() >= o).unwrap_or(rets.len());
            let none: T = rets[j..jn].iter().map(|r| self.dim(r, taken).q).product();
            let d = none * (seen * sig_o * (-seen * (tau_o - tau0).max(0.0)).exp()) * em.refl_background(P::c(x) - (scene * taken).max(1e-6).log10() * 10.0);
            if d.v() > 0.0 {
                let f = d * (1.0 - P_OUTLIER);
                // its object's fill: stopped, or partial (the beam on at the partial mean)
                let (stop, left) = if x.is_finite() && (scene.v() * taken) > 0.0 {
                    let u = FillReading::reading(em, scene.v() * taken, x);
                    let r = self.reading_at(o);
                    (r.stop(u), r.left(u))
                } else {
                    (0.0, 1.0)
                };
                let seen_on = self.reading_at(o).beam_after(em, seen, x);
                for (share, taken_on, seen_on) in [(1.0 - stop, taken * left, seen_on), (stop, 0.0, 0.0)] {
                    if share > 0.0 {
                        let fb = f * share;
                        let rest = self.lik(rets, i + 1, jn, scene, taken_on, seen_on, tau_after, prefix * fb.v(), resp);
                        resp[row + self.owners - 2] += prefix * fb.v() * rest.v();
                        total += fb * rest;
                    }
                }
            }
        }
        // an outlier
        let f = P_OUTLIER / T_MAX * em.refl_background(x);
        let rest = self.lik(rets, i + 1, j, scene, taken, seen, tau0, prefix * f, resp);
        resp[row + self.owners - 1] += prefix * f * rest.v();
        total + rest * f
    }
}

/// A cell's log-likelihood at shift `shift` and reflectance level `level` ([`mixture_ll`]) and, per echo, the share
/// its elements' layers take ([`Pulse::taken`]): each subset's by the subset's posterior, from the same reads.
pub(super) fn mixture_taken<T: Over<f64>>(pulses: &[(f64, Pulse)], shift: T, level: T, em: &Settings) -> (T, Vec<f64>) {
    let reads: Vec<_> = pulses.iter().map(|(w, p)| (*w, p, p.read(shift, level, em))).collect();
    let m = reads.iter().map(|r| r.2.ll.v()).fold(f64::NEG_INFINITY, f64::max);
    let ll = if let [(_, _, r)] = &reads[..] { r.ll } else { reads.iter().map(|(w, _, r)| (r.ll - m).exp() * *w).sum::<T>().ln() + m };
    let post: Vec<f64> = reads.iter().map(|(w, _, r)| w * (r.ll.v() - m).exp()).collect();
    let total: f64 = post.iter().sum();
    let mut out = vec![0.0; pulses[0].1.echoes.len()];
    for ((_, p, r), q) in reads.iter().zip(&post) {
        for (o, t) in out.iter_mut().zip(p.taken(r)) {
            *o += q / total * t;
        }
    }
    (ll, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use crate::scene::foliage::{Foliage, FoliageBox};
    use crate::scene::render::{Form, Class, Element, Layer, KBuffer, render_all};
    use crate::scene::settings::{Background, Settings};
    use crate::scene::term::elem_data;
    use super::super::fixtures::{cost, pole_scan, pole_view, stick};


    #[test]
    fn a_miss_is_weighted_at_the_nearest_echo_beside_it() {
        // the pole over open sky: a miss beside the pole reads the pole's range, one with no echo beside it the
        // longest, and none reads a layer (the weight is the data's alone)
        let v = pole_scan(0.03, 4.0, f64::INFINITY);
        let (mut beside, mut open) = (0, Vec::new());
        for k in 0..v.n_lines * v.n_cols {
            if v.cell[k] != Cell::Miss {
                continue;
            }
            let r = weight_range(&v, k, f64::NAN, None);
            if r == T_MAX { open.push(k) } else { assert!((r - 4.0).abs() < 0.05, "{r}"); beside += 1 }
        }
        assert!(beside > 0 && !open.is_empty(), "{beside} {}", open.len());
        // with no echo beside it, the ground's range where the ray meets it; an echo's own range wins
        assert_eq!(weight_range(&v, open[0], f64::NAN, Some((12.0, 0.5))), 12.0);
        assert_eq!(weight_range(&v, open[0], 3.0, None), 3.0);
    }

    /// A pole 0.5 m in front of a wall, closer than the pulse's resolution: its edge echoes merge with the wall's
    /// into one at the measured pull behind the pole's. With the wall in the scene (the front of a ball 2 m in radius), the
    /// true radius beats a thinner and a thicker one.
    #[test]
    fn a_stick_just_before_a_wall_merges_its_edges() {
        let (em, sc) = (Settings::default(), Background::default());
        let v = Arc::new(pole_scan(0.02, 4.0, 0.5));
        let wall = Element { form: Form::Patch { centre: [0.0, 6.5, 0.0] }, class: Class::Leaf, r: 2.0, opacity: 1.0, mismatch: 0.002, group: None, key: None };
        let e = |r: f64| cost(&v, &[wall, stick(0.0, 4.0, r)], &em, &sc).0;
        let truth = e(0.02);
        for r in [0.014, 0.028] {
            assert!(e(r) > truth + 0.5, "r {r}: {} vs {truth}", e(r));
        }
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
        Pulse::on(occluders(layers.to_vec(), ground), 1, &ext.medium(&[0.0; 3], &[0.0, 1.0, 0.0]), echoes, refls, cap, &Instrument::default(), em)
    }

    /// Slice-defects D1: behind a leaf echo that left half the beam, a return is met in that half: its echo's
    /// reflectance reads 3 dB dimmer than with nothing in front, an element's and the ground's alike, and a faint
    /// return there is recorded less often (a later echo missing is more likely).
    #[test]
    fn a_return_behind_a_leaf_echo_is_met_in_the_beam_it_left() {
        let em = Settings::default();
        let flat = Foliage::uniform(0.05);
        // a leaf echo whose reading leaves half the beam (slice-dgp: read through its fill, on its partial branch)
        let r = FillReading::of(&ReadingModel::crown(&em));
        let u = (0..4000).map(|i| -30.0 + 0.01 * i as f64).find(|&u| r.left(u) <= 0.5).unwrap();
        let leaf = (em.refl_full_hit() + u) as f32;
        assert!(r.stop(u) < 0.05, "{}", r.stop(u));
        let best = |layers: &[Layer], ground: Option<(f64, f64)>, echoes: &[f32], front: &[f32]| -> f64 {
            (0..=1000).map(|i| -35.0 + 0.025 * i as f64).map(|x| {
                let refls: Vec<f32> = front.iter().copied().chain([x as f32]).collect();
                (x, pulse_in(&em, echoes, &refls, false, layers, ground, &flat).read(0.0, 0.0, &em).ll)
            }).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0
        };
        let stick = [layer(8.0, 1.0, 1.0)];
        let (alone, behind) = (best(&stick, None, &[8.0], &[]), best(&stick, None, &[3.0, 8.0], &[leaf]));
        assert!((behind - alone - 10.0 * 0.5f64.log10()).abs() < 0.1, "an element: alone {alone}, behind {behind}");
        let (alone, behind) = (best(&[], Some((10.0, 0.6)), &[10.0], &[]), best(&[], Some((10.0, 0.6)), &[3.0, 10.0], &[leaf]));
        assert!((behind - alone - 10.0 * 0.5f64.log10()).abs() < 0.1, "the ground: alone {alone}, behind {behind}");
        // a faint edge behind the leaf: its pulse with no later echo is likelier than with nothing in front to dim it
        let edge = [layer(8.0, 0.05, 1.0)];
        let gone = |front: &[f32], echoes: &[f32]| pulse_in(&em, echoes, front, false, &edge, None, &flat).read(0.0, 0.0, &em).ll
            - pulse_in(&em, echoes, front, false, &[], None, &flat).read(0.0, 0.0, &em).ll;
        assert!(gone(&[leaf], &[3.0]) > gone(&[], &[]) + 1e-3, "{} {}", gone(&[leaf], &[3.0]), gone(&[], &[]));
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

    /// A dense medium just behind a partial edge merges with its return: the share is the site's map of the field's own
    /// share, which a dense medium raises and an opaque full hit leaves at none; the merged echo is brighter by the
    /// medium's energy and pulled back by the pull, and an echo so read is likelier with the merge than without it.
    /// Under a map and a pull declared here, steep in the field's share and the energy ratio, so the mechanism shows.
    #[test]
    fn a_merge_with_the_medium_reads_the_pull() {
        let em = Settings { pull_a: 1.0, pull_b: 0.1, merge_a: -2.0, merge_field: 10.0, merge_range: 0.0, merge_fill: 0.0, ..Settings::default() };
        let spec = FoliageBox { lo: [-0.5, 4.0, -0.5], hi: [0.5, 5.0, 0.5], voxel: 1.0 };
        let dense = Foliage { kappa: 0.05, spec: Some(spec), post: HashMap::from([([0, 0, 0], (5.0, 1.0))]), grains: [0.0; 3] };
        let merge = |f: f64, ext: &Foliage| pulse_in(&em, &[], &[], false, &[layer(4.0, f, 1.0)], None, ext).occluders[0].merge;
        let p = pulse_in(&em, &[], &[], false, &[layer(4.0, 0.3, 1.0)], None, &dense);
        let m = p.occluders[0].merge;
        assert!(m.share > 0.9 && m.share < 1.0, "{m:?}");
        // the share is the site's map of the field's own share, which `predict` reports (the map is fitted on it)
        assert!(m.field > 0.0 && m.field < m.share && (m.share - em.merge_share(m.field, 4.0, 0.3)).abs() < 1e-12, "{m:?}");
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
            let d = elem_data(&v, &KBuffer::of(&rendered).cells(), &[e], &[0], &[0], 0, &em, &sc, &ext, None);
            d[0].cells.iter().map(|c| (c.cell, c.base)).collect()
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
            Pulse::on(occluders::<f64>(Vec::new(), Some((10.0, 0.2))), 1, &flat.medium(&o, &w), echoes, refls, usize::MAX, &Instrument::default(), em).read(0.0, 0.0, em).ll
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

    /// Pulses for the pass's tests: every cell two sticks front in the pole's scan, and bare ones with a merge, a
    /// dense medium behind a partial edge, the cap, the ground and a miss.
    fn test_pulses() -> Vec<Pulse> {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [stick(0.0, 4.0, 0.03), Element { opacity: 0.4, ..stick(0.03, 3.9, 0.045) }];
        let rendered = render_all(&v, &es, &em);
        let data = elem_data(&v, &KBuffer::of(&rendered).cells(), &es, &[0, 1], &[0, 1], 0, &em, &sc, &Foliage::uniform(sc.kappa), None);
        let mut out: Vec<Pulse> = data.into_iter().flat_map(|d| d.cells.into_iter().flat_map(|c| c.pulses.into_iter().map(|p| p.1))).collect();
        let spec = FoliageBox { lo: [-0.5, 4.0, -0.5], hi: [0.5, 5.0, 0.5], voxel: 1.0 };
        let dense = Foliage { kappa: 0.05, spec: Some(spec), post: HashMap::from([([0, 0, 0], (5.0, 1.0))]), grains: [0.0; 3] };
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
        let z = |level: f64| (em.refl_wood + level - Instrument::default().floor(4.0)) / em.detect_sd;
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
            assert_eq!(r.resp.len(), p.echoes.len() * r.rx.owners());
            for row in r.rows() {
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
                        uniform() < 1.0 - (1.0 - em.p_resolve(occ[j + 1].t - occ[j].t, e)) * m[j].op * m[j + 1].op
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
