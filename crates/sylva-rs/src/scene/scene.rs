// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of elements scored against scans' range images by the ray likelihood (3dtrees research 130): its cost,
//! `−log` of its likelihood ratio against the empty scene, the cost change of a move, what an element predicts in an
//! image, and how well the scans pin an element's geometry down.
//!
//! The scene's parts: [`super::render`] (elements, layers, the k-buffer), [`super::pulse`] (one cell's pulse and
//! the pass over it), [`super::term`] (an element's term per scan), [`super::alignment`] (the scan alignment error)
//! and [`super::query`] (predictions, counts, information, divergence).
//!
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A [`Proposal`] (elements put
//!   in, live elements taken out: add, remove, swap and join in one shape) is scored by [`Scene::score`], nothing
//!   kept, and applied by [`Scene::keep`], both with the alignment held, recomputing only the terms of the elements
//!   whose cells it touches; `refit_alignment` re-solves the alignment. Every read takes a proposal and answers for the
//!   scene as it would leave it ([`View`]): its rows ([`ElementScan`]) split each (element, scan)'s cost into its parts,
//!   and [`Scene::information`] gives the slope and Fisher information of an element's cost in its geometry, so a refit
//!   is a sequence of reads.
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use nalgebra::Vector2;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use super::alignment::{Reference, solve_alignment, Fit, Changes};
use super::foliage::{self, FieldCounts, FixedHits, Foliage, FoliageBox};
use super::range_image::RangeImage;
use super::render::{Element, Layer, KBuffer, render, render_all, kept, keep, fronting, numbered};
use super::settings::{Background, Settings};
use super::term::{Owner, owner, slots, ElemData, Term, elem_data};

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
///   `ray_behind` (pulses that passed through it: evidence against); with a foliage field, each with the plug-in of
///   the hits its cells' echoes take from the field ([`Term`]'s `plug`).
/// - `alignment`: its own nodes' (stem front shift, reflectance level) prior and log-determinant terms. Each scan's
///   alignment part (its cost less its terms') is the scan's, not shared out among its elements, so the rows, those
///   parts and the field's part ([`Scene::field_part`]) sum to the scene's cost.
/// - `field`: beside `cost`, not in it: the change in the foliage field's term (`Σ ln NB`) in its cost to remove,
///   were its deposit in this scan given back ([`Term`]'s `deposit`; the field's term is non-linear in the counts, so
///   these don't sum over elements or scans).
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
/// - Each element is rendered once per image, when it is kept.
/// - A scan's cost is a term per element it sees, plus the scan alignment error's part. `keep` and the reads (`score`,
///   `information`, `rows`) hold the alignment ([`Held`]): its mode
///   and precision stay as the last refit left them, and a move's change to them is a rank-k correction on the nodes
///   it touches. So a move recomputes only the terms of the elements whose cells it touches, not the scan, and
///   removing an element restores the cost exactly (synthesis §18 invariant 7). [`Scene::refit_alignment`] re-solves
///   the alignment (a move), after which the cost is the exact Laplace marginal ([`Scene::exact_cost`], the referee);
///   the scene is built refitted.
/// - With a foliage field, each term's deposit (the hits its elements take from the field, [`Term`]'s `deposit`)
///   changes with the term, so a move changes the field's counts by its terms' old deposits less their new ones, and
///   the field's term with them, exactly. What a pulse meets of the field is fixed at the build: one step of EM from κ
///   (the terms read the background, their deposits make the field, and the terms read it), or a field held from
///   another scene ([`Scene::hold_foliage`]). The first step carries nearly all of EM's change (3dtrees research 130
///   rca-em: the stand's 1,893 of 2,028 nats, AM's all but 0.13), so a scene reached by moves is compared with a fresh
///   build of the same elements.
/// - Two approximations make the field's EM a surrogate, not the bound it would be (rca-em): the E-step reads σ's
///   posterior mean `α / β` where the mean-field bound's own step reads `exp(E ln σ)` (so iterating to a fixed point
///   drains the voxels the elements hold, without each voxel's Occam term), and the plug-in (each term's, [`Term`]'s
///   `plug`) takes the empty scene to read every echo the elements take as the medium's (weighting it by the empty
///   pulse's medium responsibility moved the stand's build by 962 nats and away from EM's fixed point, no check improving:
///   3dtrees research 130 slice-review).
/// - Element ids are stable and never reused.
/// - The reference epoch (the scanning day the alignment error's spread is stated for) is declared: a scan of another epoch adds the other-epoch part
///   to its alignment error.
pub struct Scene {
    pub(super) views: Vec<Arc<RangeImage>>,
    pub(super) em: Settings,
    pub(super) sc: Background,
    /// The foliage field the scene is scored against (held between refits), and its live counts (none without a
    /// box, or when the field is held from elsewhere).
    pub(super) ext: Foliage,
    pub(super) field: Option<FieldCounts>,
    pub(super) ref_epoch: u32,
    pub(super) elems: Vec<Option<Element>>,
    /// Per group key, its live members' ids.
    pub(super) groups: HashMap<u32, BTreeSet<u32>>,
    /// [image]: its k-buffer, every element's layers by cell.
    pub(super) buffers: Vec<KBuffer>,
    /// [image][id]: the cells the element's layers enter ([`entered`]; none where the image renders none of it, or it
    /// was removed).
    pub(super) entered: Vec<Vec<Option<Vec<usize>>>>,
    /// [image]: its fit.
    pub(super) fits: Vec<Fit>,
}

/// Element `id`'s layers `ls` entered in a k-buffer: the cells they enter, none where the image renders none of it (an
/// element rendered there is in its scan's terms even where no layer enters a cell).
pub(super) fn entered(buf: &mut KBuffer, id: u32, ls: &[(usize, Layer)]) -> Option<Vec<usize>> {
    (!ls.is_empty()).then(|| buf.enter(id, ls))
}

/// The change to the foliage field's counts were a term `old` replaced by `new` (none: no term), added to `acc`: its
/// old deposit given back, its new one taken ([`Term`]'s `deposit`).
pub(super) fn redeposit(acc: &mut FixedHits, old: Option<&Term>, new: Option<&Term>) {
    if let Some(t) = old {
        foliage::give(acc, &t.deposit, 1);
    }
    if let Some(t) = new {
        foliage::give(acc, &t.deposit, -1);
    }
}

/// The change to the foliage field's counts from scans' fits `old` (none for a scan not yet fitted) to `new`.
pub(super) fn redeposits(old: &[Fit], new: &[Fit]) -> FixedHits {
    let mut acc = FixedHits::default();
    for (s, fit) in new.iter().enumerate() {
        old.get(s).into_iter().flat_map(|f| f.terms.values()).for_each(|t| redeposit(&mut acc, Some(t), None));
        fit.terms.values().for_each(|t| redeposit(&mut acc, None, Some(t)));
    }
    acc
}

impl Scene {
    /// A scene of `elems` against `views`, the alignment error's spread stated for the scans of epoch `ref_epoch`; with `spec`, the foliage
    /// field is estimated in its box as the residual of the elements, otherwise it is the background's κ.
    pub fn new(views: Vec<Arc<RangeImage>>, elems: &[Element], em: Settings, sc: Background, spec: Option<FoliageBox>, ref_epoch: u32) -> Self {
        let mut c = Scene { views, em, sc, ext: Foliage::uniform(sc.kappa), field: None, ref_epoch, elems: elems.iter().map(|e| Some(*e)).collect(),
                             groups: HashMap::new(), buffers: Vec::new(), entered: Vec::new(), fits: Vec::new() };
        for (i, e) in elems.iter().enumerate() {
            c.join(i as u32, e);
        }
        (c.buffers, c.entered) = c.views.par_iter().map(|v| {
            let mut b = KBuffer::default();
            let cells = render_all(v, elems, &c.em).iter().enumerate().map(|(i, ls)| entered(&mut b, i as u32, ls)).collect();
            (b, cells)
        }).unzip();
        c.field = spec.map(|sp| FieldCounts::new(sp, &c.views.iter().map(|v| &**v).collect::<Vec<_>>(), &em));
        c.refit_alignment();
        if let Some(f) = &c.field {
            c.ext = f.field(c.sc.kappa);
            c.refit_alignment();
        }
        c
    }

    /// The scene's cost against the empty scene: its elements' cells, each against the same cell with the foliage
    /// field alone, and the field's part ([`Scene::field_part`]).
    pub fn cost(&self) -> f64 {
        self.fits.iter().map(|f| f.cost).sum::<f64>() + self.field_part()
    }

    /// The foliage field's part of the cost: its marginal's change against the empty scene's ([`FieldCounts`]); 0
    /// without a box or with the field held from elsewhere.
    pub fn field_part(&self) -> f64 {
        self.field.as_ref().map_or(0.0, FieldCounts::cost)
    }

    /// The scene's exact cost: each scan's Laplace marginal over the alignment error re-solved for the elements as
    /// they are, and the field's part with the re-solved terms' deposits (the referee for the held alignment; the
    /// scene unchanged).
    pub fn exact_cost(&self) -> f64 {
        let fits: Vec<Fit> = (0..self.views.len()).into_par_iter().map(|s| self.solve(s)).collect();
        let field = self.field.as_ref().map_or(0.0, |f| f.change(&redeposits(&self.fits, &fits)));
        fits.iter().map(|f| f.cost).sum::<f64>() + self.field_part() + field
    }

    /// Re-solve the scan alignment error for the elements as they are, and the field's counts with the terms' deposits
    /// (a move: its cost change); after it the cost is [`Scene::exact_cost`], and moves hold the new mode and precision.
    pub fn refit_alignment(&mut self) -> f64 {
        let before = self.cost();
        let fits: Vec<Fit> = (0..self.views.len()).into_par_iter().map(|s| self.solve(s)).collect();
        let delta = redeposits(&self.fits, &fits);
        self.fits = fits;
        self.give(delta);
        self.cost() - before
    }

    /// Hold a foliage field estimated elsewhere (another scene's, from other scans) in place of the scene's own, which
    /// it drops with its counts. The cost change.
    pub fn hold_foliage(&mut self, ext: &Foliage) -> f64 {
        self.ext = ext.clone();
        self.field = None;
        self.refit_alignment()
    }

    /// Enter element `id` in its group's members.
    pub(super) fn join(&mut self, id: u32, e: &Element) {
        if let Some(g) = e.group {
            self.groups.entry(g).or_default().insert(id);
        }
    }

    /// Take element `id` out of its group's members.
    pub(super) fn leave(&mut self, id: u32, e: &Element) {
        if let Some(set) = e.group.and_then(|g| self.groups.get_mut(&g)) {
            set.remove(&id);
            if set.is_empty() {
                self.groups.remove(&e.group.unwrap());
            }
        }
    }

    /// Change the field's counts by `delta` (deposits given back and taken): the change in its part of the cost.
    pub(super) fn give(&mut self, delta: FixedHits) -> f64 {
        match &mut self.field {
            Some(f) => f.apply(delta),
            None => 0.0,
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

    /// The foliage field and its counts as proposal `p` would leave them ([`Scene::check_proposal`]): per voxel with
    /// counts, (voxel, σ as held, the counts). Empty when the field is held from elsewhere.
    pub fn foliage(&self, p: &Proposal) -> (Option<FoliageBox>, Vec<([i32; 3], f64, (f64, f64))>) {
        let Some(f) = &self.field else { return (self.ext.spec, Vec::new()) };
        let (_, delta) = self.views_after(p);
        let mut out: Vec<_> = f.counts_after(&delta).into_iter().map(|(k, c)| (k, self.ext.sigma(&k), c)).collect();
        out.sort_by_key(|x| x.0);
        (self.ext.spec, out)
    }

    /// A row per (term, scan) of the scene as proposal `p` would leave it ([`Scene::check_proposal`]), by element id:
    /// an element's term, or a group's under its first member's id.
    pub fn rows(&self, p: &Proposal) -> Vec<ElementScan> {
        let (views, delta) = self.views_after(p);
        let mut first: HashMap<u32, u32> = HashMap::new();
        for i in (0..self.next_id() + p.put_in.len() as u32).rev() {
            if let Some(g) = self.element_after(p, i).and_then(|e| e.group) {
                first.insert(g, i);
            }
        }
        // the field's part of a term's cost to remove: its deposit given back, from the counts the proposal leaves
        let field = |t: &Term| self.field.as_ref().map_or(0.0, |f| {
            let mut back = FixedHits::default();
            redeposit(&mut back, Some(t), None);
            -f.change_after(&delta, &back)
        });
        let mut rows = Vec::new();
        for (s, (fit, v)) in self.fits.iter().zip(&views).enumerate() {
            let mut terms: BTreeMap<Owner, &Term> = fit.terms.iter().filter(|(o, _)| !v.index.contains_key(o)).map(|(o, t)| (*o, t)).collect();
            terms.extend(v.terms.iter().filter_map(|(o, t)| t.as_ref().map(|t| (*o, t))));
            for (o, t) in terms {
                let (i, group) = match o {
                    Owner::Alone(i) => (i, None),
                    Owner::Group(g) => (first[&g], Some(g)),
                };
                // the shift's posterior from the held alignment, and the own nodes' with it marginalised
                let var = fit.held.between(&t.align, &t.align);
                let own_sd = |j: usize| t.own_sd[j] * (t.own_cov[(j, j)] + t.own_link[j] * t.own_link[j] * var).max(0.0).sqrt();
                rows.push(ElementScan {
                    elem: i, group, scan: s as u32, cells: t.cells, cells_on: t.cells_on, arc: t.arc, footprints_across: t.across, surface: t.surface,
                    cost: t.cost(), ray_on: t.ray[0], ray_empty: t.ray[1], ray_in_front: t.ray[2], ray_behind: t.ray[3],
                    alignment: t.cost() - t.ray.iter().sum::<f64>(), field: field(t),
                    discount: self.em.discount,
                    align_shift: fit.held.shift(&t.align), align_sd: var.max(0.0).sqrt(), front_shift: t.own_sd[0] * t.own[0], front_shift_sd: own_sd(0),
                    refl_level: t.own_sd[1] * t.own[1], refl_level_sd: own_sd(1),
                });
            }
        }
        rows.sort_by_key(|r| (r.elem, r.scan));
        rows
    }

    /// Keep a proposal ([`Scene::check_proposal`]): its elements taken out and put in, each scan re-termed once, so a
    /// chain costs its cells once. The new elements' ids follow the scene's, in order (ids are never reused, so a swap
    /// gives new ones); the cost change. Taking out what was just put in restores the cost exactly.
    pub fn keep(&mut self, p: &Proposal) -> (Vec<u32>, f64) {
        let first = self.next_id();
        let afters: Vec<(Vec<Vec<(usize, Layer)>>, Changes)> = (0..self.views.len()).into_par_iter().map(|s| {
            let v = self.after(s, p);
            (v.layers, v.terms)
        }).collect();
        for &id in &p.take_out {
            let gone = self.elems[id as usize].take().expect("a live element");
            self.leave(id, &gone);
        }
        for (k, e) in p.put_in.iter().enumerate() {
            self.elems.push(Some(*e));
            self.join(first + k as u32, e);
        }
        let (mut d, mut delta) = (0.0, FixedHits::default());
        for (s, (ls, ch)) in afters.into_iter().enumerate() {
            for &id in &p.take_out {
                if let Some(old) = self.entered[s][id as usize].take() {
                    self.buffers[s].leave(id, &old);
                }
            }
            for (k, l) in ls.iter().enumerate() {
                let cells = entered(&mut self.buffers[s], first + k as u32, l);
                self.entered[s].push(cells);
            }
            d += self.apply(s, ch, &mut delta);
        }
        ((first..first + p.put_in.len() as u32).collect(), d + self.give(delta))
    }

    /// An error unless a proposal can be read or kept: the elements it takes out live and each once, the elements it
    /// puts in scoreable ([`Element::invalid`]).
    pub fn check_proposal(&self, p: &Proposal) -> Result<(), SceneError> {
        for (k, &id) in p.take_out.iter().enumerate() {
            self.check_id(id)?;
            if p.take_out[..k].contains(&id) {
                return Err(SceneError(format!("element {id} taken out twice")));
            }
        }
        match p.put_in.iter().enumerate().find_map(|(k, e)| e.invalid().map(|why| (k, why))) {
            Some((k, why)) => Err(SceneError(format!("element {k} put in: {why}"))),
            None => Ok(()),
        }
    }

    /// The id the next element kept would get: a proposal's elements are numbered from it in every read.
    pub fn next_id(&self) -> u32 {
        self.elems.len() as u32
    }

    /// Element `id`, if it is live.
    pub fn element(&self, id: u32) -> Option<Element> {
        self.elems.get(id as usize).copied().flatten()
    }

    /// Element `id`, if it is live after proposal `p` (one it puts in numbered from [`Scene::next_id`]).
    pub fn element_after(&self, p: &Proposal, id: u32) -> Option<Element> {
        match id.checked_sub(self.next_id()) {
            Some(k) => p.put_in.get(k as usize).copied(),
            None if p.take_out.contains(&id) => None,
            None => self.element(id),
        }
    }

    /// Put a move's new terms into scan `s`'s fit, their deposits' change added to `delta`: the fit's cost change.
    pub(super) fn apply(&mut self, s: usize, terms: Changes, delta: &mut FixedHits) -> f64 {
        if terms.is_empty() {
            return 0.0;
        }
        let fit = &mut self.fits[s];
        for (o, t) in terms {
            let old = match t {
                Some(t) => fit.terms.insert(o, t),
                None => fit.terms.remove(&o),
            };
            redeposit(delta, old.as_ref(), fit.terms.get(&o));
            fit.note(o);
        }
        let before = fit.cost;
        fit.cost = fit.total(&HashMap::new());
        fit.cost - before
    }

    /// Scan `s`'s data for every live element whose layers enter its cells ([`elem_data`]), each term with its owner.
    pub(super) fn data(&self, s: usize) -> Vec<(Owner, ElemData)> {
        let mut ids: Vec<u32> = (0..self.elems.len() as u32).filter(|&i| self.entered[s][i as usize].is_some()).collect();
        let cells = numbered(self.buffers[s].cells(), &mut ids);
        let elems: Vec<Element> = ids.iter().map(|&i| self.elems[i as usize].unwrap()).collect();
        let owners: Vec<Owner> = ids.iter().zip(&elems).map(|(&i, e)| owner(i, e)).collect();
        let data = elem_data(&self.views[s], &cells, &elems, &slots(&owners), &(0..ids.len()).collect::<Vec<_>>(), self.ref_epoch, &self.em, &self.sc, &self.ext,
                             self.field.as_ref().map(|f| &f.spec));
        data.into_iter().map(|d| (owners[d.members[0]], d)).collect()
    }

    /// Scan `s` solved afresh for the elements as they are ([`solve_alignment`]), its alignment held there.
    pub(super) fn solve(&self, s: usize) -> Fit {
        let (owners, data): (Vec<Owner>, Vec<ElemData>) = self.data(s).into_iter().unzip();
        let (mut held, terms) = solve_alignment(&data, &self.em);
        let fit_terms: BTreeMap<Owner, Term> = owners.into_iter().zip(terms).collect();
        held.reference = fit_terms.iter().map(|(o, t)| (*o, Reference::of(t))).collect();
        let mut fit = Fit { terms: fit_terms, held, moved: BTreeSet::new(), cost: 0.0 };
        fit.cost = fit.total(&HashMap::new());
        fit
    }

    /// Scan `s` as proposal `p` would leave it, nothing kept ([`View`]): its elements rendered as ids from
    /// [`Scene::next_id`], and the new term of every owner whose term changes (none: it left the scan), the alignment
    /// held. A term reads only the cells its members are the front of, each with its layers at the cell's own surface
    /// class, and its class reads the same cells ([`term_classes`]). So the terms that change are those of the fronts of
    /// the cells the proposal touches (before it and after); and a group's term is its members' together, so every
    /// present member of a group one of those (or an element taken out) belongs to is rescored with it, and its pooled
    /// evidence never goes stale. A scan the proposal doesn't touch has no new terms.
    pub(super) fn after<'a>(&'a self, s: usize, p: &'a Proposal) -> View<'a> {
        let first = self.next_id();
        let layers: Vec<Vec<(usize, Layer)>> = p.put_in.iter().enumerate().map(|(k, e)| render(&self.views[s], e, first + k as u32, &self.em)).collect();
        let mut fresh = KBuffer::default();
        let cells: Vec<Option<Vec<usize>>> = layers.iter().enumerate().map(|(k, ls)| entered(&mut fresh, first + k as u32, ls)).collect();
        let moved = layers.iter().any(|l| !l.is_empty()) || p.take_out.iter().any(|&w| self.entered[s][w as usize].is_some());
        let terms = if moved {
            let touched: HashSet<usize> = p.take_out.iter().flat_map(|&w| self.entered[s][w as usize].iter().flatten()).chain(cells.iter().flatten().flatten()).copied().collect();
            let with: Vec<(Element, Option<&[usize]>)> = p.put_in.iter().zip(&cells).map(|(e, c)| (*e, c.as_deref())).collect();
            let start: Vec<u32> = (0..with.len() as u32).map(|k| first + k).collect();
            self.terms_after(s, &p.take_out, &with, first, &|c| fresh.get(c).to_vec(), &touched, &start, None)
        } else {
            Changes::default()
        };
        let index = terms.iter().enumerate().map(|(k, t)| (t.0, k)).collect();
        View { scene: self, s, p, first, layers, fresh, cells, terms, index }
    }

    /// [`Scene::after`]'s rule, for a move that touches the cells `touched` and adds the elements `start` (its
    /// elements `with` put in from `first`, each with the cells it enters ([`entered`]); their layers in a cell `fresh`), leaving out
    /// the term of `skip` (a chain grown beside the scene: [`super::chain`], which keeps its own term).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn terms_after(&self, s: usize, without: &[u32], with: &[(Element, Option<&[usize]>)], first: u32, fresh: &dyn Fn(usize) -> Vec<Layer>,
                              touched: &HashSet<usize>, start: &[u32], skip: Option<Owner>) -> Changes {
        let (v, em, fit, buf) = (&*self.views[s], &self.em, &self.fits[s], &self.buffers[s]);
        let added = |i: u32| i >= first && ((i - first) as usize) < with.len();
        let entered = |i: u32| -> Option<&[usize]> { if added(i) { with[(i - first) as usize].1 } else { self.entered[s][i as usize].as_deref() } };
        let element = |i: u32| if added(i) { with[(i - first) as usize].0 } else { self.elems[i as usize].unwrap() };
        let present = |i: u32| entered(i).is_some() && (added(i) || !without.contains(&i)) && Some(owner(i, &element(i))) != skip;
        // a cell's k-buffer after the move
        let after = |c: usize| keep(buf.get(c).iter().filter(|l| !without.contains(&l.elem) && !added(l.elem)).copied().chain(fresh(c)).collect());
        let owner_of = |i: u32| owner(i, &element(i));
        // every present member of the groups of `ids` and of the elements taken out
        let whole = |ids: &mut BTreeSet<u32>| {
            let groups: BTreeSet<u32> = ids.iter().map(|&i| element(i)).chain(without.iter().map(|&i| self.elems[i as usize].unwrap())).filter_map(|e| e.group).collect();
            if groups.is_empty() {
                return;
            }
            ids.extend(groups.iter().filter_map(|g| self.groups.get(g)).flatten());
            ids.extend((0..with.len() as u32).filter(|&k| with[k as usize].0.group.is_some_and(|g| groups.contains(&g))).map(|k| first + k));
            ids.retain(|&i| present(i));
        };
        // the terms with a front share in the cells the move touches, before it and after, and the elements it adds
        let mut near: BTreeSet<u32> = start.iter().copied().collect();
        for &c in touched {
            let (before, after) = (kept(buf.get(c)), after(c));
            near.extend(before[..fronting(before)].iter().chain(&after[..fronting(&after)]).map(|l| l.elem));
        }
        near.retain(|&i| present(i));
        whole(&mut near);
        // the cells those terms enter, with their k-buffers after the move, numbered from 0: those terms' elements, then
        // the context as it comes
        let mut cells: Vec<usize> = near.iter().flat_map(|&i| entered(i).into_iter().flatten()).copied().collect();
        cells.sort_unstable();
        cells.dedup();
        let mut ids: Vec<u32> = near.iter().copied().collect();
        let cells = numbered(cells.into_iter().map(|c| (c, after(c))).collect(), &mut ids);
        let el: Vec<Element> = ids.iter().map(|&i| element(i)).collect();
        let owners: Vec<Owner> = ids.iter().map(|&i| owner_of(i)).collect();
        let data = elem_data(v, &cells, &el, &slots(&owners), &(0..near.len()).collect::<Vec<_>>(), self.ref_epoch, em, &self.sc, &self.ext,
                             self.field.as_ref().map(|f| &f.spec));
        // a held term's own nodes start where its reference's did, so an unchanged term is recomputed exactly
        let mut terms: Vec<(Owner, Option<Term>)> = data.par_iter().map(|d| {
            let o = owners[d.members[0]];
            (o, Some(d.term(fit.held.shift(&d.align), fit.held.reference.get(&o).map_or(Vector2::zeros(), |t| t.start), em)))
        }).collect();
        // the owners of the elements taken out or unseen now, unless a member is still scored
        let scored: HashSet<Owner> = terms.iter().map(|t| t.0).collect();
        let left = without.iter().map(|&i| owner(i, &self.elems[i as usize].unwrap())).chain((0..with.len() as u32).map(|k| owner_of(first + k)));
        terms.extend(left.collect::<BTreeSet<Owner>>().into_iter().filter(|o| !scored.contains(o) && Some(*o) != skip).map(|o| (o, None)));
        terms
    }

    /// Each proposal's cost change were it kept ([`Scene::check_proposal`]), per scan and then, when the scene holds a
    /// foliage field, the field's part (the change in [`Scene::field_part`]): (proposals × scans, or scans + 1), in
    /// parallel, the scene unchanged, its foliage field and alignment held. A row sums to keeping it, bit for bit. A
    /// proposal's elements of one group are one term ([`Owner`]: a chain), joined by the scene's members of that group,
    /// which are rescored with it; its other elements are each alone.
    pub fn score(&self, ps: &[Proposal]) -> Vec<Vec<f64>> {
        ps.par_iter().map(|p| {
            let mut delta = FixedHits::default();
            let mut out: Vec<f64> = (0..self.views.len()).map(|s| {
                let v = self.after(s, p);
                if v.terms.is_empty() {
                    return 0.0;
                }
                let over: HashMap<Owner, Option<Term>> = v.terms.into_iter().collect();
                for (o, t) in &over {
                    redeposit(&mut delta, self.fits[s].terms.get(o), t.as_ref());
                }
                self.fits[s].total(&over) - self.fits[s].cost
            }).collect();
            if let Some(f) = &self.field {
                out.push(f.change(&delta));
            }
            out
        }).collect()
    }

    /// Every scan as proposal `p` would leave it ([`Scene::after`]), and the change to the foliage field's counts were
    /// it kept: each changed term's old deposit given back, its new one taken.
    pub(super) fn views_after<'a>(&'a self, p: &'a Proposal) -> (Vec<View<'a>>, FixedHits) {
        let views: Vec<View> = (0..self.views.len()).into_par_iter().map(|s| self.after(s, p)).collect();
        let mut delta = FixedHits::default();
        for v in &views {
            for (o, t) in &v.terms {
                redeposit(&mut delta, self.fits[v.s].terms.get(o), t.as_ref());
            }
        }
        (views, delta)
    }
}

/// What a proposal puts in and takes out of the scene: the elements it adds (numbered from [`Scene::next_id`], in
/// order, as [`Scene::keep`] would number them) and the live elements it removes. One shape covers add, remove, swap
/// and join (a chain's members taken out and put back under another chain's key). Reads take one and answer for the
/// scene as it would leave it; `keep` applies it.
#[derive(Debug, Clone, Default)]
pub struct Proposal {
    pub put_in: Vec<Element>,
    pub take_out: Vec<u32>,
}

impl Proposal {
    /// A proposal that puts elements in and takes nothing out.
    pub fn put(es: &[Element]) -> Proposal {
        Proposal { put_in: es.to_vec(), take_out: Vec::new() }
    }

    /// A proposal that takes live elements out and puts nothing in.
    pub fn take(ids: &[u32]) -> Proposal {
        Proposal { put_in: Vec::new(), take_out: ids.to_vec() }
    }
}

/// One scan as a proposal would leave it ([`Scene::after`]), nothing kept: the proposal's elements' layers and the
/// cells each enters (none where the scan renders none of it), and the changed terms. Reads look elements, cells and
/// terms up through it, so a read for a proposal is the read the scene would give after keeping it.
pub(super) struct View<'a> {
    pub(super) scene: &'a Scene,
    pub(super) s: usize,
    pub(super) p: &'a Proposal,
    pub(super) first: u32,
    pub(super) layers: Vec<Vec<(usize, Layer)>>,
    pub(super) fresh: KBuffer,
    pub(super) cells: Vec<Option<Vec<usize>>>,
    pub(super) terms: Changes,
    pub(super) index: HashMap<Owner, usize>,
}

impl View<'_> {
    /// Element `i`, live after the proposal (one it puts in from `first`).
    pub(super) fn element(&self, i: u32) -> Option<Element> {
        self.scene.element_after(self.p, i)
    }

    /// The cells element `i`'s layers enter after the proposal ([`entered`]).
    pub(super) fn entered(&self, i: u32) -> Option<&[usize]> {
        if i >= self.first {
            self.cells.get((i - self.first) as usize).and_then(|c| c.as_deref())
        } else if self.p.take_out.contains(&i) {
            None
        } else {
            self.scene.entered[self.s][i as usize].as_deref()
        }
    }

    /// Cell `cell`'s layers after the proposal, as its pulse reads them ([`kept`]).
    pub(super) fn layers(&self, cell: usize) -> Vec<Layer> {
        keep(self.scene.buffers[self.s].get(cell).iter().filter(|l| !self.p.take_out.contains(&l.elem)).copied().chain(self.fresh.get(cell).iter().copied()).collect())
    }

    /// Owner `o`'s term after the proposal.
    pub(super) fn term(&self, o: &Owner) -> Option<&Term> {
        match self.index.get(o) {
            Some(&k) => self.terms[k].1.as_ref(),
            None => self.scene.fits[self.s].terms.get(o),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use crate::scene::foliage::{self, Deposit, FieldCounts, Foliage};
    use crate::scene::render::{Form, Element};
    use crate::scene::settings::{Background, Settings};
    use crate::scene::term::Owner;
    use super::super::fixtures::{add, cost, costs, field_box, piece_of, pole_view, remove, stick, swap, thin_pole_view};

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
        let delta: f64 = costs(&Scene::new(vec![v.clone()], &pair[..1], em, sc, None, 0), &[vec![pair[1]]])[0].iter().sum();
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let parts = alignment_parts(&Scene::new(vec![v.clone()], &pair, em, sc, None, 0));
        let sum: f64 = both.1.iter().map(|r| r.cost).sum::<f64>() + parts;
        assert!((sum - both.0).abs() < 1e-6);
    }

    /// The foliage field is what the elements leave: the empty scene's field holds the pole's echoes (its voxel is
    /// dense), and a scene built with the pole takes its share of them out. Added to the empty scene, its field fixed,
    /// the pole costs less than nothing, that change is the cost, the rows, the alignment parts and the field's part sum
    /// to it, and removing it restores the empty scene exactly; built with the pole, the rows sum to the cost too.
    #[test]
    fn the_field_is_what_the_elements_leave() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = stick(0.0, 4.0, 0.03);
        let sigma = |c: &Scene| c.foliage(&Proposal::default()).1.iter().find(|x| x.0 == [5, 10, 3]).map_or(f64::NAN, |x| x.1);
        let mut c = Scene::new(vec![v.clone()], &[], em, sc, Some(field_box()), 0);
        assert_eq!(c.cost(), 0.0);
        let dense = sigma(&c);
        assert!(dense > 1.0, "{dense}");
        let fresh = Scene::new(vec![v.clone()], &[pole], em, sc, Some(field_box()), 0);
        assert!(sigma(&fresh) < 0.5 * dense, "{} vs {dense}", sigma(&fresh));
        // each datum is scored once: the elements' cells and the field's part hold the whole cost
        let rows = |c: &Scene| c.rows(&Proposal::default()).iter().map(|r| r.cost).sum::<f64>() + alignment_parts(c) + c.field_part();
        assert!((rows(&fresh) - fresh.cost()).abs() < 1e-9 * fresh.cost().abs(), "{} vs {}", rows(&fresh), fresh.cost());
        let (id, d) = add(&mut c, pole);
        assert!((d - c.cost()).abs() < 1e-9 * d.abs().max(1.0) && d < 0.0, "{d}");
        assert!((rows(&c) - c.cost()).abs() < 1e-9 * c.cost().abs().max(1.0), "{} vs {}", rows(&c), c.cost());
        remove(&mut c, id);
        assert_eq!(c.cost(), 0.0);
    }

    /// Soft attribution (research 130 slice-soft): with the pole in (and a thin stick partly in front of its edge, two
    /// terms sharing cells), every voxel keeps the empty scene's path, since the likelihood reads the medium's path in
    /// every way it explains a pulse; and each voxel's hits are the empty scene's less what the elements take, between
    /// none and all of them: each echo's mass sums to one. At the pole's surface the elements take nearly all of it, in
    /// front of the pole none.
    #[test]
    fn the_field_keeps_every_path_and_shares_each_echo() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let counts = |els: &[Element]| -> HashMap<[i32; 3], (f64, f64)> {
            Scene::new(vec![v.clone()], els, em, sc, Some(field_box()), 0).foliage(&Proposal::default()).1.into_iter().map(|x| (x.0, x.2)).collect()
        };
        let (empty, pole) = (counts(&[]), counts(&[stick(0.0, 4.0, 0.03), stick(0.0365, 3.9, 0.008)]));
        let mut taken = 0.0;
        for (k, &(h, l)) in &empty {
            let (hp, lp) = pole.get(k).copied().unwrap_or((0.0, 0.0));
            assert_eq!(lp, l, "{k:?}");
            assert!(hp >= 0.0 && hp <= h, "{k:?}: {hp} of {h}");
            taken += h - hp;
        }
        assert!(pole.keys().all(|k| empty.contains_key(k)) && taken > 0.0);
        // in front (y 3.5–3.6 m) and at the surface (y 3.9–4.0 m: the elements take 87 % of its hits; the rest are its
        // edges' and the medium's)
        let (front, surface) = ([5, 6, 3], [5, 10, 3]);
        assert_eq!(pole[&front], empty[&front]);
        assert!(pole[&surface].0 < 0.2 * empty[&surface].0, "{:?} vs {:?}", pole[&surface], empty[&surface]);
    }

    /// The field is fixed through moves (research 130 slice 2b), as the alignment is held: moves keep its counts
    /// exactly (the empty scene's less every term's deposit, recomputed from scratch), adding then removing an element
    /// restores the cost exactly and swapping one there and back to rounding (each swap gives a new id, so the held
    /// alignment carries a column out and one in), and after a small move, or one to a wrong, wider stick, the cost is
    /// near a fresh build's of the moved scene.
    #[test]
    fn held_moves_keep_the_fields_counts() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let spec = Some(field_box());
        let (pole, nudged, wrong, side) = (stick(0.0, 4.0, 0.03), stick(0.002, 4.0, 0.03), stick(0.01, 4.0, 0.033), stick(0.15, 4.5, 0.02));
        let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
        let e0 = c.cost();
        let (id, _) = add(&mut c, side);
        remove(&mut c, id);
        assert_eq!(c.cost(), e0);
        let (id, _) = swap(&mut c, 0, wrong);
        assert_eq!(c.field.as_ref().unwrap().counts(), recounted(&c));
        swap(&mut c, id, pole);
        assert!((c.cost() - e0).abs() < 1e-9 * e0.abs(), "{} vs {e0}", c.cost());
        // moved against a fresh build, nats: 0.001 nudged, 0.8 wrong
        for (e, tol) in [(nudged, 0.01), (wrong, 1.0)] {
            let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
            swap(&mut c, 0, e);
            let fresh = Scene::new(vec![v.clone()], &[e], em, sc, spec, 0).cost();
            assert!((c.cost() - fresh).abs() < tol, "moved {} vs fresh {fresh}", c.cost());
        }
    }

    /// With a field held, the cost is smooth where an element's front crosses a voxel face (3dtrees research 130
    /// slice-review: a rule that read the medium differently in front of an element in its own voxel stepped the cost
    /// by about 0.28 nats there). The pole slid 1 cm along the beam in 25 µm steps, its front crossing the face at
    /// y = 4.0 m against the empty scene's field (dense in the pole's voxels): its cost to add has no second
    /// difference over 1e-3 nats.
    #[test]
    fn the_cost_is_continuous_across_a_voxel_face_with_a_field_held() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[], em, sc, Some(field_box()), 0);
        let hyps: Vec<Vec<Element>> = (0..=400).map(|i| vec![stick(0.0, 4.025 + 2.5e-5 * i as f64, 0.03)]).collect();
        let costs: Vec<f64> = costs(&c, &hyps).iter().map(|r| r.iter().sum()).collect();
        let worst = costs.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-3, "a second difference of {worst} nats");
    }

    /// Many proposals at once: each one's cost per scan, and the foliage field's part, sum to keeping it, bit for bit
    /// (and taking its elements out again restores the cost exactly), the scene unchanged; a proposal that takes an
    /// element out and puts another in scores what keeping it gives.
    #[test]
    fn scores_match_keeping() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, Some(field_box()), 0);
        let hyps = vec![vec![stick(0.0, 4.0, 0.02)], vec![stick(0.2, 6.0, 0.03), stick(0.25, 6.0, 0.01)], vec![stick(5.0, 4.0, 0.03)], vec![stick(0.0365, 3.9, 0.008)]];
        let e0 = c.cost();
        let all = costs(&c, &hyps);
        assert_eq!(c.cost(), e0);
        for (h, per) in hyps.iter().zip(&all) {
            let (ids, d) = c.keep(&Proposal::put(h));
            assert_eq!(per.iter().sum::<f64>(), d);
            c.keep(&Proposal::take(&ids));
            assert_eq!(c.cost(), e0);
        }
        // a stick outside the box takes nothing from the field
        assert!(all[1][2] != 0.0 && all[2][2] == 0.0, "{all:?}");
        let swap = Proposal { put_in: vec![stick(0.0, 4.0, 0.02)], take_out: vec![0] };
        let per = c.score(std::slice::from_ref(&swap)).remove(0);
        assert_eq!(per.iter().sum::<f64>(), c.keep(&swap).1);
    }

    /// The scans' alignment parts: each scan's cost less its terms'.
    fn alignment_parts(c: &Scene) -> f64 {
        c.fits.iter().map(|f| f.cost - f.terms.values().map(Term::cost).sum::<f64>()).sum()
    }

    /// The field's counts recomputed from scratch: the empty scene's, less every term's deposit as it now stands.
    fn recounted(c: &Scene) -> foliage::Counts {
        let f = c.field.as_ref().unwrap();
        let mut fresh = FieldCounts::new(f.spec, &c.views.iter().map(|v| &**v).collect::<Vec<_>>(), &c.em);
        fresh.apply(redeposits(&[], &c.fits));
        fresh.counts()
    }

    /// The field collapsed (research 130 formulation.md): its term, `Σ ln NB` over the counts, is exact through moves
    /// (its counts are the empty scene's less the terms' deposits recomputed from scratch, and the term a recompute
    /// from the counts); a move's change to the field's part is the change in that part (the plug-in held); and adding
    /// then removing restores it exactly.
    #[test]
    fn the_fields_marginal_is_exact_through_moves() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let (pole, wrong, side) = (stick(0.0, 4.0, 0.03), stick(0.01, 4.0, 0.033), stick(0.15, 4.5, 0.02));
        let field = |c: &Scene| c.field.clone().unwrap();
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[pole], em, sc, Some(field_box()), 0);
        let (f0, e0) = (c.field_part(), c.cost());
        let (id, _) = add(&mut c, side);
        let (w, _) = swap(&mut c, 0, wrong);
        remove(&mut c, id);
        let (p, _) = swap(&mut c, w, pole);
        assert_eq!(c.field_part(), f0);
        assert!((c.cost() - e0).abs() < 1e-9 * e0.abs(), "{} vs {e0}", c.cost());
        swap(&mut c, p, wrong);
        assert_eq!(field(&c).counts(), recounted(&c));
        let (a0, b0) = field(&c).prior;
        let recomputed: f64 = field(&c).counts().values().map(|&(h, l)| foliage::ln_nb(h, l, a0, b0)).sum();
        assert!((recomputed - field(&c).marginal()).abs() < 1e-9 * recomputed.abs(), "{recomputed} vs {}", field(&c).marginal());
        c.refit_alignment();
        assert_eq!(field(&c).counts(), recounted(&c));
        let before = c.field_part();
        let per = costs(&c, &[vec![side]]);
        add(&mut c, side);
        assert!((per[0][2] - (c.field_part() - before)).abs() < 1e-9, "{} vs {}", per[0][2], c.field_part() - before);
    }

    /// With the field collapsed, the pole's cost to add hardly depends on whether the field it is scored against was
    /// fitted with it in, where the plug-in reading before the collapse (each field held fixed: the cells' ratios
    /// alone) differs by far more: the field's term is a function of the scene alone.
    #[test]
    fn a_hypothesiss_cost_does_not_depend_on_the_last_refit() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = stick(0.0, 4.0, 0.03);
        let scene = |els: &[Element], spec| Scene::new(vec![v.clone()], els, em, sc, spec, 0);
        let held = |els: &[Element], ext: &Foliage| {
            let mut c = scene(els, None);
            c.hold_foliage(ext);
            c
        };
        // the pole's cost to add, from the scene without it; and to remove, from the scene with it
        let (without, mut with) = (scene(&[], Some(field_box())), scene(&[pole], Some(field_box())));
        let (out_held, mut in_held) = (held(&[], without.foliage_field()), held(&[pole], with.foliage_field()));
        let put = |c: &Scene| -> f64 { costs(c, &[vec![pole]])[0].iter().sum() };
        let (collapsed, today) = (put(&without) + remove(&mut with, 0), put(&out_held) + remove(&mut in_held, 0));
        assert!(collapsed.abs() < 0.1 * today.abs(), "collapsed {collapsed} vs today {today}");
    }

    /// A move's change is a full recompute's (3dtrees research 130 slice-structure, slice-soft): each cell read at its
    /// own surface class, a term depends on no other term's class, so after adds, swaps and removes among neighbours
    /// that share cells (a group among them), every scan's terms, their deposits included, equal its terms recomputed
    /// from scratch at the held alignment (each from its reference's start), bit for bit, and so do the field's counts;
    /// and the moves undone restore the cost to rounding (a swap gives a new id).
    #[test]
    fn a_moves_terms_are_a_full_recompute() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let front = stick(0.02, 3.9, 0.02);
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03), front, piece_of(-0.2, 0.0, Some(5))], em, sc, Some(field_box()), 0);
        let check = |c: &Scene| {
            for s in 0..c.n_scans() {
                let fit = &c.fits[s];
                let fresh: BTreeMap<Owner, (f64, Deposit)> = c.data(s).into_iter().map(|(o, d)| {
                    let t = d.term(fit.held.shift(&d.align), fit.held.reference.get(&o).map_or(Vector2::zeros(), |t| t.start), &em);
                    (o, (t.cost(), t.deposit))
                }).collect();
                assert_eq!(fresh, fit.terms.iter().map(|(o, t)| (*o, (t.cost(), t.deposit.clone()))).collect::<BTreeMap<Owner, _>>());
                // the owners the fit notes as moved are those whose term differs from the alignment's reference
                let moved: BTreeSet<Owner> = fit.terms.keys().chain(fit.held.reference.keys()).copied()
                    .filter(|o| !matches!((fit.held.reference.get(o), fit.terms.get(o)), (Some(r), Some(t)) if r.same_alignment(t))).collect();
                assert_eq!(moved, fit.moved);
            }
            assert!(c.fits.iter().flat_map(|f| f.terms.values()).any(|t| !t.deposit.is_empty()));
            assert_eq!(c.field.as_ref().unwrap().counts(), recounted(c));
        };
        let e0 = c.cost();
        let (a, _) = add(&mut c, stick(0.01, 3.95, 0.025));
        check(&c);
        let (f, _) = swap(&mut c, 1, stick(0.05, 3.85, 0.03));
        check(&c);
        // its edge partial in front of the pole's: cells two terms share
        let (f, _) = swap(&mut c, f, stick(0.0365, 3.9, 0.008));
        check(&c);
        let (b, _) = add(&mut c, piece_of(0.0, 0.2, Some(5)));
        check(&c);
        remove(&mut c, a);
        check(&c);
        remove(&mut c, b);
        swap(&mut c, f, front);
        check(&c);
        assert!((c.cost() - e0).abs() < 1e-9 * e0.abs(), "{} vs {e0}", c.cost());
    }

    /// Keeping elements together is keeping them one by one, bit for bit: a chain of four pieces of the pole (one group)
    /// and a stick beside it, with a foliage field, give the same cost, rows and field counts either way; and removing
    /// them restores the cost exactly.
    #[test]
    fn keeping_together_is_keeping_one_by_one() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [piece_of(-0.2, -0.1, Some(7)), piece_of(-0.1, 0.0, Some(7)), stick(0.15, 4.5, 0.02), piece_of(0.0, 0.1, Some(7)), piece_of(0.1, 0.2, Some(7))];
        let build = || Scene::new(vec![v.clone(), v.clone()], &[stick(0.4, 6.0, 0.02)], em, sc, Some(field_box()), 0);
        let (mut one, mut many) = (build(), build());
        let e0 = one.cost();
        for e in &es {
            add(&mut one, *e);
        }
        let (ids, d) = many.keep(&Proposal::put(&es));
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(one.cost(), many.cost());
        assert!((d - (many.cost() - e0)).abs() < 1e-9 * d.abs().max(1.0));
        let rows = |c: &Scene| c.rows(&Proposal::default()).iter().map(|r| (r.elem, r.scan, r.cost.to_bits(), r.field.to_bits())).collect::<Vec<_>>();
        // the rows read for the proposal, nothing kept, are the rows after keeping it
        assert_eq!(rows(&one), build().rows(&Proposal::put(&es)).iter().map(|r| (r.elem, r.scan, r.cost.to_bits(), r.field.to_bits())).collect::<Vec<_>>());
        assert_eq!(rows(&one), rows(&many));
        assert_eq!(one.field.as_ref().unwrap().counts(), many.field.as_ref().unwrap().counts());
        many.keep(&Proposal::take(&ids));
        assert_eq!(many.cost(), e0);
    }

    /// A move that touches one member rescores its whole group, so its pooled terms never go stale: pieces of the
    /// pole 10 cm apart (no cell in common), added one by one, reach the same cost as the group built at once;
    /// removing a member restores the cost exactly, and the score of a member joining a group equals keeping it.
    #[test]
    fn a_touched_member_rescores_its_group() {
        let v = thin_pole_view(0.0015, 6.0);
        let (em, sc) = (Settings::default(), Background::default());
        let piece = |z: f64| Element { form: Form::Stick { a: [0.0, 6.0, z - 0.02], b: [0.0, 6.0, z + 0.02] }, group: Some(1), ..stick(0.0, 6.0, 0.0015) };
        let pieces = [piece(-0.15), piece(-0.05), piece(0.05), piece(0.15)];
        let at_once = Scene::new(vec![v.clone()], &pieces, em, sc, None, 0);
        let mut by_one = Scene::new(vec![v.clone()], &pieces[..1], em, sc, None, 0);
        for p in &pieces[1..] {
            let t: f64 = costs(&by_one, &[vec![*p]])[0].iter().sum();
            let e0 = by_one.cost();
            assert_eq!(t, add(&mut by_one, *p).1);
            let (id, _) = add(&mut by_one, Element { group: Some(1), ..*p });
            remove(&mut by_one, id);
            assert_eq!(by_one.cost(), e0 + t);
        }
        by_one.refit_alignment();
        assert!((by_one.cost() - at_once.cost()).abs() < 1e-9 * at_once.cost().abs().max(1.0), "{} vs {}", by_one.cost(), at_once.cost());
        assert_eq!(by_one.rows(&Proposal::default()).len(), 1);
    }
}
