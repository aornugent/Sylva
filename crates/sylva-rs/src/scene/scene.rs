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
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A move (add, remove,
//!   change) returns its cost change with the alignment held, recomputing only the terms of the elements whose cells
//!   it touches; `refit_alignment` re-solves the alignment. Its rows ([`ElementScan`]) split each (element, scan)'s cost into its parts, and
//!   [`Scene::information`] gives the slope and Fisher information of an element's cost in its geometry.
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use nalgebra::Vector2;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use super::alignment::{hats, Reference, solve_alignment, Fit, Changes};
use super::foliage::{self, FieldCounts, FixedHits, Foliage, FoliageBox};
use super::range_image::RangeImage;
use super::render::{Element, Layer, KBuffer, render, render_all, kept, keep, fronting, numbered};
use super::settings::{Background, Settings, ALIGN_SPACING};
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
/// - `alignment`: its own nodes' (stem front shift, reflectance level) prior and log-determinant terms. The scan's alignment part is the scan's, not shared out among its
///   elements ([`Scene::alignment_parts`]), so the rows, those parts and the field's part ([`Scene::field_part`]) sum
///   to the scene's cost.
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
/// - Each element is rendered once per image, and re-rendered only when it is added or changed.
/// - A scan's cost is a term per element it sees, plus the scan alignment error's part. Moves (`add`, `remove`,
///   `change`) and queries (`costs_to_add`, `costs_to_remove`, `information`) hold the alignment ([`Held`]): its mode
///   and precision stay as the last refit left them, and a move's change to them is a rank-k correction on the nodes
///   it touches. So a move recomputes only the terms of the elements whose cells it touches, not the scan, and
///   removing an element restores the cost exactly (synthesis §18 invariant 7). [`Scene::refit_alignment`] re-solves the alignment (a move like `refit_foliage`), after which
///   the cost is the exact Laplace marginal ([`Scene::exact_cost`], the referee); the scene is built refitted.
/// - With a foliage field, each term's deposit (the hits its elements take from the field, [`Term`]'s `deposit`)
///   changes with the term, so a move changes the field's counts by its terms' old deposits less their new ones. A scene
///   is built by one step of EM from κ: the terms read the background, their deposits make the field, and the terms read
///   it ([`Scene::refit_foliage`] is another step). The first step carries nearly all of EM's change (3dtrees research
///   130 rca-em: the stand's 1,893 of 2,028 nats, AM's all but 0.13), so **compare fresh builds with fresh builds**: a
///   scene refitted further, or reached by moves and refits, sits a few tens of nats from a fresh build of the same
///   elements on the stand.
/// - Two approximations make the field's EM a surrogate, not the bound it would be (rca-em): the E-step reads σ's
///   posterior mean `α / β` where the mean-field bound's own step reads `exp(E ln σ)` (so iterating to a fixed point
///   drains the voxels the elements hold, without each voxel's Occam term), and the plug-in (each term's, [`Term`]'s
///   `plug`) takes the empty scene to read every echo the elements take as the medium's.
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
        if c.field.is_some() {
            c.refit_foliage();
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

    /// Re-read the foliage field a pulse meets from the counts the moves since the last refit kept, and refit the
    /// alignment, the terms' responsibilities and so their deposits with it (a move: its cost change; one step of EM).
    /// Moves hold that field.
    pub fn refit_foliage(&mut self) -> f64 {
        let before = self.cost();
        let Some(f) = &mut self.field else { return 0.0 };
        self.ext = f.field(self.sc.kappa);
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

    /// The foliage field and its counts: per voxel with counts, (voxel, σ as held, the counts as the scene now is).
    /// Empty when the field is held from elsewhere.
    pub fn foliage(&self) -> (Option<FoliageBox>, Vec<([i32; 3], f64, (f64, f64))>) {
        let Some(f) = &self.field else { return (self.ext.spec, Vec::new()) };
        let mut out: Vec<_> = f.counts().into_iter().map(|(k, c)| (k, self.ext.sigma(&k), c)).collect();
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
        // the field's part of a term's cost to remove: its deposit given back
        let field = |t: &Term| self.field.as_ref().map_or(0.0, |f| {
            let mut back = FixedHits::default();
            redeposit(&mut back, Some(t), None);
            -f.change(&back)
        });
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

    /// Add an element: its id and the cost change.
    pub fn add(&mut self, e: Element) -> (u32, f64) {
        let (ids, d) = self.add_many(&[e]);
        (ids[0], d)
    }

    /// Add elements together: their ids and the cost change. Every element is rendered first and each scan re-termed
    /// once ([`Scene::reterm`]), so a chain costs its cells once: added one by one, each member re-terms its group whole.
    /// The scene is the same, bit for bit, as after adding them one by one.
    pub fn add_many(&mut self, es: &[Element]) -> (Vec<u32>, f64) {
        let first = self.elems.len() as u32;
        let em = self.em;
        let layers: Vec<Vec<Vec<(usize, Layer)>>> = self.views.par_iter()
            .map(|v| es.iter().enumerate().map(|(k, e)| render(v, e, first + k as u32, &em)).collect()).collect();
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter().map(|s| {
            if layers[s].iter().all(Vec::is_empty) {
                return Changes::default();
            }
            let with: Vec<(Element, Vec<(usize, Layer)>)> = es.iter().copied().zip(layers[s].iter().cloned()).collect();
            self.reterm(s, &[], &with, first)
        }).collect();
        for (k, e) in es.iter().enumerate() {
            self.elems.push(Some(*e));
            self.join(first + k as u32, e);
        }
        let (mut d, mut delta) = (0.0, FixedHits::default());
        for (s, (ls, ch)) in layers.into_iter().zip(changes).enumerate() {
            for (k, l) in ls.iter().enumerate() {
                let cells = entered(&mut self.buffers[s], first + k as u32, l);
                self.entered[s].push(cells);
            }
            d += self.apply(s, ch, &mut delta);
        }
        ((first..first + es.len() as u32).collect(), d + self.give(delta))
    }

    /// Remove a live element ([`Scene::check_id`]): the cost change.
    pub fn remove(&mut self, id: u32) -> f64 {
        let first = self.elems.len() as u32;
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter()
            .map(|s| if self.entered[s][id as usize].is_none() { Changes::default() } else { self.reterm(s, &[id], &[], first) }).collect();
        let gone = self.elems[id as usize].take().expect("a live element");
        self.leave(id, &gone);
        let (mut d, mut delta) = (0.0, FixedHits::default());
        for (s, ch) in changes.into_iter().enumerate() {
            if let Some(old) = self.entered[s][id as usize].take() {
                self.buffers[s].leave(id, &old);
            }
            d += self.apply(s, ch, &mut delta);
        }
        d + self.give(delta)
    }

    /// Replace a live element ([`Scene::check_id`]): the cost change.
    pub fn change(&mut self, id: u32, e: Element) -> f64 {
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        let changes: Vec<Changes> = (0..self.views.len()).into_par_iter().map(|s| {
            if layers[s].is_empty() && self.entered[s][id as usize].is_none() { Changes::default() } else { self.reterm(s, &[id], &[(e, layers[s].clone())], id) }
        }).collect();
        let old = self.elems[id as usize].replace(e).expect("a live element");
        self.leave(id, &old);
        self.join(id, &e);
        let (mut d, mut delta) = (0.0, FixedHits::default());
        for (s, (l, ch)) in layers.into_iter().zip(changes).enumerate() {
            if let Some(old) = self.entered[s][id as usize].take() {
                self.buffers[s].leave(id, &old);
            }
            self.entered[s][id as usize] = entered(&mut self.buffers[s], id, &l);
            d += self.apply(s, ch, &mut delta);
        }
        d + self.give(delta)
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
    pub(super) fn reterm(&self, s: usize, without: &[u32], with: &[(Element, Vec<(usize, Layer)>)], first: u32) -> Changes {
        let mut fresh = KBuffer::default();
        let cells: Vec<Option<Vec<usize>>> = with.iter().enumerate().map(|(k, (_, ls))| entered(&mut fresh, first + k as u32, ls)).collect();
        let touched: HashSet<usize> = without.iter().flat_map(|&w| self.entered[s][w as usize].iter().flatten()).chain(cells.iter().flatten().flatten()).copied().collect();
        let with: Vec<(Element, Option<&[usize]>)> = with.iter().zip(&cells).map(|((e, _), c)| (*e, c.as_deref())).collect();
        let start: Vec<u32> = (0..with.len() as u32).map(|k| first + k).collect();
        self.terms_after(s, without, &with, first, &|c| fresh.get(c).to_vec(), &touched, &start, None)
    }

    /// [`Scene::reterm`]'s rule, for a move that touches the cells `touched` and adds the elements `start` (its
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
    pub(super) fn costs_replacing(&self, items: &[(Vec<Element>, Vec<u32>)]) -> Vec<Vec<f64>> {
        let first = self.elems.len() as u32;
        items.par_iter().map(|(h, without)| {
            let layers: Vec<Vec<Vec<(usize, Layer)>>> = self.views.iter().map(|v| h.iter().map(|e| render(v, e, 0, &self.em)).collect()).collect();
            let mut delta = FixedHits::default();
            let mut out: Vec<f64> = layers.iter().enumerate().map(|(s, ls)| {
                if ls.iter().all(Vec::is_empty) && without.iter().all(|&w| self.entered[s][w as usize].is_none()) {
                    return 0.0;
                }
                let with: Vec<(Element, Vec<(usize, Layer)>)> = h.iter().copied().zip(ls.iter().cloned()).collect();
                let over: HashMap<Owner, Option<Term>> = self.reterm(s, without, &with, first).into_iter().collect();
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
                        let ls = kept(self.buffers[s].get(*c));
                        for o in &ls[..fronting(ls)] {
                            cells.entry((true, term(owner(o.elem, &self.elems[o.elem as usize].unwrap())))).or_default().insert(k as u32);
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
    use std::collections::{BTreeMap, HashMap};
    use crate::scene::foliage::{self, Deposit, FieldCounts, Foliage};
    use crate::scene::render::{Form, Element};
    use crate::scene::settings::{Background, Settings};
    use crate::scene::term::Owner;
    use super::super::fixtures::{cost, field_box, piece_of, pole_view, stick, thin_pole_view};

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

    /// The foliage field is what the elements leave: the empty scene's field holds the pole's echoes (its voxel is
    /// dense), and promoting the pole takes its share of them out. The cost after a promotion is near a fresh scene's
    /// (one step of EM each, from different fields: the empty scene's and κ), its change is that cost, the rows, the
    /// alignment parts and the field's part sum to it, and demotion restores the empty scene exactly.
    #[test]
    fn the_field_is_what_the_elements_leave() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = stick(0.0, 4.0, 0.03);
        let sigma = |c: &Scene| c.foliage().1.iter().find(|x| x.0 == [5, 10, 3]).map_or(f64::NAN, |x| x.1);
        let mut c = Scene::new(vec![v.clone()], &[], em, sc, Some(field_box()), 0);
        assert_eq!(c.cost(), 0.0);
        let dense = sigma(&c);
        assert!(dense > 1.0, "{dense}");
        let (id, d) = c.promote(pole);
        let fresh = Scene::new(vec![v.clone()], &[pole], em, sc, Some(field_box()), 0);
        assert!((c.cost() - fresh.cost()).abs() < 1e-3 * fresh.cost().abs(), "{} vs {}", c.cost(), fresh.cost());
        assert!((d - c.cost()).abs() < 1e-9 * d.abs().max(1.0));
        assert!(d < 0.0, "{d}");
        assert!(sigma(&c) < 0.5 * dense, "{} vs {dense}", sigma(&c));
        // each datum is scored once: the elements' cells and the field's part hold the whole cost
        let rows: f64 = c.rows().iter().map(|r| r.cost).sum::<f64>() + c.alignment_parts().iter().sum::<f64>() + c.field_part();
        assert!((rows - c.cost()).abs() < 1e-9 * c.cost().abs().max(1.0), "{rows} vs {}", c.cost());
        c.demote(id);
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
            Scene::new(vec![v.clone()], els, em, sc, Some(field_box()), 0).foliage().1.into_iter().map(|x| (x.0, x.2)).collect()
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

    /// The field is held through moves (research 130 slice 2b), as the alignment is: moves keep its counts exactly (the
    /// empty scene's less every term's deposit, recomputed from scratch), adding then removing an element and moving
    /// one there and back restore the cost exactly, and after a small move, or one to a wrong, wider stick, the held
    /// cost is near the refitted one.
    #[test]
    fn held_moves_keep_the_fields_counts() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let spec = Some(field_box());
        let (pole, nudged, wrong, side) = (stick(0.0, 4.0, 0.03), stick(0.002, 4.0, 0.03), stick(0.01, 4.0, 0.033), stick(0.15, 4.5, 0.02));
        let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
        let e0 = c.cost();
        let (id, _) = c.add(side);
        c.remove(id);
        assert_eq!(c.cost(), e0);
        c.change(0, wrong);
        assert_eq!(c.field.as_ref().unwrap().counts(), recounted(&c));
        c.change(0, pole);
        assert_eq!(c.cost(), e0);
        // held against refitted, nats: 0.0004 nudged, 0.4 wrong
        for (e, tol) in [(nudged, 0.01), (wrong, 1.0)] {
            let mut c = Scene::new(vec![v.clone()], &[pole], em, sc, spec, 0);
            c.change(0, e);
            let held = c.cost();
            c.refit_foliage();
            assert!((held - c.cost()).abs() < tol, "held {held} vs refitted {}", c.cost());
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
        let costs: Vec<f64> = c.costs_to_add(&hyps).iter().map(|r| r.iter().sum()).collect();
        let worst = costs.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-3, "a second difference of {worst} nats");
    }

    /// Many hypotheses at once: each one's cost per scan, and the foliage field's part, sum to adding then removing
    /// it, the scene unchanged.
    #[test]
    fn costs_to_add_match_add_then_remove() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, Some(field_box()), 0);
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
        // a stick outside the box takes nothing from the field
        assert!(all[1][2] != 0.0 && all[2][2] == 0.0, "{all:?}");
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
        let (id, _) = c.add(side);
        c.change(0, wrong);
        c.remove(id);
        c.change(0, pole);
        assert_eq!((c.field_part(), c.cost()), (f0, e0));
        c.change(0, wrong);
        assert_eq!(field(&c).counts(), recounted(&c));
        let (a0, b0) = field(&c).prior;
        let recomputed: f64 = field(&c).counts().values().map(|&(h, l)| foliage::ln_nb(h, l, a0, b0)).sum();
        assert!((recomputed - field(&c).marginal()).abs() < 1e-9 * recomputed.abs(), "{recomputed} vs {}", field(&c).marginal());
        c.refit_foliage();
        assert_eq!(field(&c).counts(), recounted(&c));
        let before = c.field_part();
        let per = c.costs_to_add(&[vec![side]]);
        c.add(side);
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
        let add = |c: &Scene| -> f64 { c.costs_to_add(&[vec![pole]])[0].iter().sum() };
        let (collapsed, today) = (add(&without) + with.remove(0), add(&out_held) + in_held.remove(0));
        assert!(collapsed.abs() < 0.1 * today.abs(), "collapsed {collapsed} vs today {today}");
    }

    /// A move's change is a full recompute's (3dtrees research 130 slice-structure, slice-soft): each cell read at its
    /// own surface class, a term depends on no other term's class, so after adds, changes and removes among neighbours
    /// that share cells (a group among them), every scan's terms, their deposits included, equal its terms recomputed
    /// from scratch at the held alignment (each from its reference's start), bit for bit, and so do the field's counts;
    /// and the moves undone restore the cost exactly.
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
            }
            assert!(c.fits.iter().flat_map(|f| f.terms.values()).any(|t| !t.deposit.is_empty()));
            assert_eq!(c.field.as_ref().unwrap().counts(), recounted(c));
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

    /// Adding elements together is adding them one by one, bit for bit: a chain of four pieces of the pole (one group)
    /// and a stick beside it, with a foliage field, give the same cost, rows and field counts either way; and removing
    /// them restores the cost exactly.
    #[test]
    fn add_many_is_adding_one_by_one() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [piece_of(-0.2, -0.1, Some(7)), piece_of(-0.1, 0.0, Some(7)), stick(0.15, 4.5, 0.02), piece_of(0.0, 0.1, Some(7)), piece_of(0.1, 0.2, Some(7))];
        let build = || Scene::new(vec![v.clone(), v.clone()], &[stick(0.4, 6.0, 0.02)], em, sc, Some(field_box()), 0);
        let (mut one, mut many) = (build(), build());
        let e0 = one.cost();
        for e in &es {
            one.add(*e);
        }
        let (ids, d) = many.add_many(&es);
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(one.cost(), many.cost());
        assert!((d - (many.cost() - e0)).abs() < 1e-9 * d.abs().max(1.0));
        let rows = |c: &Scene| c.rows().iter().map(|r| (r.elem, r.scan, r.cost.to_bits(), r.field.to_bits())).collect::<Vec<_>>();
        assert_eq!(rows(&one), rows(&many));
        assert_eq!(one.field.as_ref().unwrap().counts(), many.field.as_ref().unwrap().counts());
        for id in ids {
            many.remove(id);
        }
        assert_eq!(many.cost(), e0);
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
}
