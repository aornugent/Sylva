// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! An element's term per scan (3dtrees research 130): the cells it owns, its own nodes' Newton and Laplace step, and
//! what it deposits in the foliage field.
//!
//! - **Weights:** each cell counts [`Settings::cell_weight`], `1 / (φ m(ρ))`, with m the footprints per cell
//!   (overlapping footprints, on the cell's own lattice steps) and φ the evidence discount per cell: one constant
//!   for every cell and every hypothesis, so it widens posteriors without favouring any hypothesis (a
//!   quasi-likelihood). It replaced slice 7's ρ_res, `1 + (n − 1) ρ` with n an element's own footprints, which
//!   differed between hypotheses and, once every fronted cell counted, favoured the thinner (slice 7c).
//! - **Chains** (3dtrees research 130 slice 6): elements of one group ([`Element`]'s `group`) are one term per scan
//!   ([`Owner`]), so a chain of short sticks shares its nuisances. A declared term over
//!   its geometry alone (its coupling: taper, bends) is the consumer's, beside the cost, never in it.

use crate::Point;
use crate::transform::{norm, sub};
use nalgebra::{Matrix2, Vector2};
use std::collections::HashMap;
use super::alignment::hats;
use super::dual::{Dual2, Real};
use super::foliage::{self, Deposit, Foliage, FoliageBox};
use super::pulse::{Outcome, mixture_ll, cell_view, front_echo, depth_sd, Pulse, mixture_taken};
use super::range_image::{Cell, RangeImage};
use super::render::{MOSTLY, Form, Element, Layer, front_shares};
use super::settings::{Background, Settings, ALIGN_SPACING};

/// Whose term a cell's evidence enters, per scan: an element alone (by id), or a group (by its key). A group's members
/// share one set of nuisances (one alignment shift, at their middles' mean; one stem front shift, its prior sd their
/// mean; one reflectance level) and one surface class (the median over their cells: [`term_classes`]), so a chain of short sticks is
/// charged its nuisances once. Its members' cells are scored as one element's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Owner {
    Alone(u32),
    Group(u32),
}

/// Element `id`'s owner.
pub(super) fn owner(id: u32, e: &Element) -> Owner {
    e.group.map_or(Owner::Alone(id), Owner::Group)
}

/// Each element's term among its owners: the place its owner first appears.
pub(super) fn slots(owners: &[Owner]) -> Vec<usize> {
    let mut at: HashMap<Owner, usize> = HashMap::new();
    owners.iter().map(|o| {
        let n = at.len();
        *at.entry(*o).or_insert(n)
    }).collect()
}

/// The mean over some elements of a quantity (one element's is exactly its own).
pub(super) fn mean_of(es: &[&Element], f: impl Fn(&Element) -> f64) -> f64 {
    es.iter().map(|e| f(e)).sum::<f64>() / es.len() as f64
}

/// A term's middle: its elements' middles' mean.
pub(super) fn middle(es: &[&Element]) -> Point {
    [0, 1, 2].map(|k| mean_of(es, |e| e.mid()[k]))
}

/// A term's latent nodes ([`Owner`]: an element's, or a group's members' together): its alignment hats at its middle
/// ([`hats`]) and its stem front shift's prior sd (its members' mean). The cost ([`elem_data`]) and the information
/// ([`Shared`]) read them here, so a group's front shift has one prior in both.
pub(super) fn term_nodes(es: &[&Element], em: &Settings) -> (Vec<([i64; 3], f64)>, f64) {
    (hats(&middle(es), ALIGN_SPACING), mean_of(es, |e| e.front_shift_sd(em)))
}

/// Each element's term's surface class in one view (`slot`, per element: [`slots`]), which sets the term's alignment
/// spread: the median class of the cells its members mostly fill ([`MOSTLY`]) that hold an echo with a surface fit (mid
/// where none). The echoes themselves read their own cells' class ([`cell_class`]).
pub(super) fn term_classes(v: &RangeImage, cells: &[(usize, Vec<Layer>)], slot: &[usize], em: &Settings) -> Vec<u8> {
    let mut ks: Vec<[u32; 3]> = vec![[0; 3]; slot.iter().max().map_or(0, |m| m + 1)];
    for (cell, ls) in cells {
        if let Some((e, k)) = vote(v, *cell, ls, em) {
            ks[slot[e as usize]][k as usize] += 1;
        }
    }
    slot.iter().map(|&t| median_class(&ks[t])).collect()
}

/// A cell's vote for a term's surface class ([`term_classes`]): the element of its first layer wholly in it, and the
/// cell's class, where that layer mostly fills it and the cell holds an echo with a surface fit.
pub(super) fn vote(v: &RangeImage, cell: usize, ls: &[Layer], em: &Settings) -> Option<(u32, u8)> {
    let front = ls.iter().find(|l| l.full())?;
    (front.f > MOSTLY && v.cell[cell] == Cell::Echoes && v.excess[cell].is_finite()).then(|| (front.elem, em.surface(v.excess[cell] as f64)))
}

/// The median of a term's votes, counted by class (mid where it has none).
pub(super) fn median_class(n: &[u32; 3]) -> u8 {
    let total: u32 = n.iter().sum();
    if total == 0 {
        return 1;
    }
    let mut seen = 0;
    (0..3u8).find(|&k| {
        seen += n[k as usize];
        seen > total / 2
    }).unwrap_or(1)
}

/// One cell a term has a front share in (`cell`, its index), as its ray likelihood scores it: its pulse per subset of its mixture
/// ([`CellView`]), its weight times the term's front share ([`front_shares`]), its log-likelihood under the empty
/// scene (the medium and the ground alone: the same for every hypothesis), what its first echo says about its
/// front element, and, with a foliage field, the hits the field counts of its echoes ([`foliage::hits`]: the echo, its
/// voxel, its weight times the term's front share, and ln σ̂ there, the held field the empty scene reads it at).
pub(super) struct CellData {
    pub(super) cell: usize,
    pub(super) pulses: Vec<(f64, Pulse)>,
    pub(super) wt: f64,
    pub(super) base: f64,
    pub(super) outcome: Outcome,
    pub(super) hits: Vec<(usize, [i32; 3], f64, f64)>,
}

/// One term's data in one scan (an element's, or a group's: [`Owner`]): the cells its members own (its evidence:
/// [`elem_data`]), and its latent nodes: its alignment nodes with their weights (its alignment shift is `Σ w z`, each
/// hat's weight times its surface class's along-beam spread), and the prior sds of its own two nodes, the stem front
/// shift (m; 0 when off) and the reflectance level (dB; 0 while reflectance isn't read).
pub(super) struct ElemData {
    /// Its members, as indices into the elements it was built from.
    pub(super) members: Vec<usize>,
    pub(super) cells: Vec<CellData>,
    pub(super) surface: u8,
    /// Cells whose first echo is its.
    pub(super) cells_on: u32,
    /// The least and greatest of those cells' rays' place across its axis, in radii (clamped to ±1; a stick's).
    pub(super) on_across: [f64; 2],
    pub(super) across: f64,
    pub(super) align: Vec<([i64; 3], f64)>,
    pub(super) own_sd: Vector2<f64>,
}

impl ElemData {
    /// (δ, reflectance level) at alignment shift `a` and own nodes `y`.
    pub(super) fn at(&self, a: f64, y: Vector2<f64>) -> (f64, f64) {
        (a + self.own_sd[0] * y[0], self.own_sd[1] * y[1])
    }

    /// Its ray likelihood at `x` = (δ, reflectance level), in total and by [`Outcome`] (in its order), and −ll's slope
    /// and curvature in (δ, level) there, in closed form, one pass over its cells ([`Dual2`]; the value is the `f64`
    /// one), the curvature's negative eigenvalues set to 0. Without a level node only δ's are filled. With `deposit`,
    /// from the same pass, what its elements take of the foliage field's hits there and their plug-in ([`Local`]'s
    /// `deposit` and `plug`), the plug-in in its parts by outcome too.
    pub(super) fn quad(&self, x: (f64, f64), em: &Settings, deposit: bool) -> Local {
        let level = if self.own_sd[1] > 0.0 { Dual2::var(x.1, 1) } else { Dual2::c(x.1) };
        let (mut total, mut parts, mut hits, mut plug) = (Dual2::<2>::c(0.0), [0.0; 4], Vec::new(), 0.0);
        for c in &self.cells {
            let (l, p) = if deposit && !c.hits.is_empty() {
                let (ll, taken) = mixture_taken(&c.pulses, Dual2::var(x.0, 0), level, em);
                let (h, p) = c.take(&taken);
                hits.extend(h);
                ((ll - c.base) * c.wt, p)
            } else {
                (c.read(Dual2::var(x.0, 0), level, em), 0.0)
            };
            total += l;
            parts[c.outcome as usize] += l.v + p;
            plug += p;
        }
        Local { ll: total.v, parts, g: -total.g, hess: clipped(-total.h), deposit: foliage::deposit(hits.into_iter()), plug }
    }

    /// Its curvature with its own nodes `y` eliminated ([`eliminate`]).
    pub(super) fn eliminate(&self, at: &Local, y: Vector2<f64>) -> Eliminated {
        eliminate(self.own_sd, at, y)
    }

    /// Its term at alignment shift `a` (m): its own nodes y at their mode given `a` ([`settle`], from `start`), the
    /// Laplace step over them, and its cost's slope and curvature in `a` with them eliminated (the Schur complement of
    /// its curvature over (a, y)).
    pub(super) fn term(&self, a: f64, start: Vector2<f64>, em: &Settings) -> Term {
        let (y, at) = settle(self.own_sd, start, |y| self.quad(self.at(a, y), em, true));
        let e = eliminate(self.own_sd, &at, y);
        Term {
            align: self.align.clone(),
            fit: -at.ll + 0.5 * (y[0] * y[0] + y[1] * y[1]),
            logdet: 0.5 * e.det.ln(),
            slope: e.slope,
            curv: e.curv,
            own: y,
            start,
            own_sd: self.own_sd,
            own_cov: e.pi,
            own_link: e.pl,
            ray: at.parts.map(|x| -x),
            plug: -at.plug,
            cells: self.cells.len() as u32,
            cells_on: self.cells_on,
            arc: match self.on_across {
                [lo, hi] if lo <= hi => hi.asin() - lo.asin(),
                _ => 0.0,
            },
            across: self.across,
            surface: self.surface,
            deposit: at.deposit,
        }
    }
}

impl CellData {
    /// Its part of its term's log-likelihood at shift `shift` and reflectance level `level`: against the empty scene,
    /// weighted.
    pub(super) fn read(&self, shift: Dual2<2>, level: Dual2<2>, em: &Settings) -> Dual2<2> {
        (mixture_ll(&self.pulses, shift, level, em) - self.base) * self.wt
    }

    /// What its term explains of the field's hits at the shares `taken` of its echoes (its layers' responsibilities):
    /// per hit its voxel and weight (the field's count of it: [`foliage::hits`]), and their plug-in `Σ w t ln σ̂`, how
    /// the empty scene's pulse already reads them in the cell's ratio, as a log-likelihood (the field's term reads them
    /// by its marginal instead: [`Term`]'s `plug`). The plug-in weighs each echo as the cell's ratio does (`wt`), not as
    /// the field counts it, so σ̂ cancels from the cost but through the responsibilities (3dtrees research 130
    /// field-closed-form, slice-path).
    pub(super) fn take(&self, taken: &[f64]) -> (Vec<([i32; 3], f64)>, f64) {
        let hits: Vec<([i32; 3], f64)> = self.hits.iter().map(|&(i, k, w, _)| (k, w * taken[i])).collect();
        (hits, self.hits.iter().map(|&(i, _, _, ln_sigma)| self.wt * taken[i] * ln_sigma).sum())
    }
}

/// A curvature with its negative eigenvalues set to 0.
pub(super) fn clipped(hess: Matrix2<f64>) -> Matrix2<f64> {
    if hess[(0, 0)] >= 0.0 && hess[(1, 1)] >= 0.0 && hess.determinant() >= 0.0 {
        return hess;
    }
    let mut eig = hess.symmetric_eigen();
    eig.eigenvalues = eig.eigenvalues.map(|l| l.max(0.0));
    eig.recompose()
}

/// A term's own nodes' mode (Newton from `start`, a step at most 3 prior sds, halved until the cost falls, stopping
/// under a micronat) and its likelihood there; `read` gives its likelihood at own nodes y, `s` their prior sds.
pub(super) fn settle(s: Vector2<f64>, start: Vector2<f64>, read: impl Fn(Vector2<f64>) -> Local) -> (Vector2<f64>, Local) {
    let prior = |y: Vector2<f64>| 0.5 * (y[0] * y[0] + y[1] * y[1]);
    let mut y = if s == Vector2::zeros() { Vector2::zeros() } else { start };
    let mut at = read(y);
    if s != Vector2::zeros() {
        let mut f_now = -at.ll + prior(y);
        for _ in 0..MAX_NEWTON {
            let e = eliminate(s, &at, y);
            let step = cap(e.pi * e.gy);
            let (mut t, mut fall) = (1.0, None);
            while t >= 1e-3 {
                let yn = y - step * t;
                let next = read(yn);
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
    (y, at)
}

/// A term's curvature over (a, y) at own nodes `y` (prior sds `s`) given its likelihood there (`at`), with y
/// eliminated: over (a, y), M = Jᵀ Q J + diag(0, I) with J = [[1, s0, 0], [0, 0, s1]], so P = diag(s) Q diag(s) + I
/// over y and the a–y block is `link`; the slope over y; and the cost's slope and curvature in a, y eliminated (the
/// Schur complement).
pub(super) fn eliminate(s: Vector2<f64>, at: &Local, y: Vector2<f64>) -> Eliminated {
    let (q, g) = (at.hess, at.g);
    let off = s[0] * s[1] * q[(0, 1)];
    let p = Matrix2::new(s[0] * s[0] * q[(0, 0)] + 1.0, off, off, s[1] * s[1] * q[(1, 1)] + 1.0);
    let pi = p.try_inverse().expect("P is the unit prior's precision plus a positive semi-definite part");
    let link = Vector2::new(q[(0, 0)] * s[0], q[(0, 1)] * s[1]);
    let (pl, gy) = (pi * link, s.component_mul(&g) + y);
    Eliminated { pi, det: p.determinant(), link, pl, gy, slope: g[0] - pl[0] * gy[0] - pl[1] * gy[1], curv: q[(0, 0)] - link[0] * pl[0] - link[1] * pl[1] }
}

/// A term's curvature with its own nodes eliminated ([`ElemData::eliminate`]): P⁻¹ and |P| over y, the a–y link
/// and P⁻¹ link, the slope over y, and the cost's slope and curvature in the alignment shift a.
pub(super) struct Eliminated {
    pub(super) pi: Matrix2<f64>,
    pub(super) det: f64,
    pub(super) link: Vector2<f64>,
    pub(super) pl: Vector2<f64>,
    pub(super) gy: Vector2<f64>,
    pub(super) slope: f64,
    pub(super) curv: f64,
}

/// A step over a term's own nodes scaled to at most 3 prior sds.
pub(super) fn cap(step: Vector2<f64>) -> Vector2<f64> {
    step / (step.amax().max(3.0) / 3.0)
}

/// One term's ray likelihood at a point (δ, reflectance level), by outcome, and −ll's slope and curvature there
/// ([`ElemData::quad`]); and, asked for, what its elements take of the foliage field's hits there (soft attribution;
/// [`foliage`]'s module docs): per hit, its weight (front share included) times the share of its echo the cell's
/// layers take by their responsibilities ([`mixture_taken`]), and their plug-in ([`CellData::take`]), in `parts` but
/// not in `ll`.
pub(super) struct Local {
    pub(super) ll: f64,
    pub(super) parts: [f64; 4],
    pub(super) g: Vector2<f64>,
    pub(super) hess: Matrix2<f64>,
    pub(super) deposit: Deposit,
    pub(super) plug: f64,
}

/// One (element, scan)'s part of its scan's cost at an alignment shift, `fit + logdet`: `fit` its ray likelihood's
/// cost plus its own nodes' prior at their mode ŷ, `logdet` half their Laplace log-determinant ln|P|. With them, the
/// cost's `slope` and `curv`ature in its alignment shift (its own nodes
/// eliminated), where its own nodes' Newton began (`start`: a move recomputes a held term from its reference's, so an
/// unchanged term comes out the same), and what its row reports: ŷ, its own nodes' covariance given the shift P⁻¹ and how they follow it
/// (`own_link`: ŷ moves by −`own_link` per unit of shift), the ray likelihood's cost by outcome, and its cells; and,
/// with a foliage field, the hits its elements take from it at its nodes (`deposit`, from its last read: [`ElemData::quad`])
/// and their plug-in: `plug`, `−Σ h ln σ̂` at the held field, which the cells' ratios read as the empty scene's and the
/// field's term reads by its marginal, so each hit is scored once (3dtrees research 130 formulation.md). It is linear
/// in the deposit at the held σ̂, so it is the term's own; it stays out of the own nodes' Newton, and in its cost and
/// its `ray` by outcome.
#[derive(Debug, Clone)]
pub(super) struct Term {
    pub(super) align: Vec<([i64; 3], f64)>,
    pub(super) fit: f64,
    pub(super) logdet: f64,
    pub(super) slope: f64,
    pub(super) curv: f64,
    pub(super) own: Vector2<f64>,
    pub(super) start: Vector2<f64>,
    pub(super) own_sd: Vector2<f64>,
    pub(super) own_cov: Matrix2<f64>,
    pub(super) own_link: Vector2<f64>,
    pub(super) ray: [f64; 4],
    pub(super) cells: u32,
    pub(super) cells_on: u32,
    pub(super) arc: f64,
    pub(super) across: f64,
    pub(super) surface: u8,
    pub(super) deposit: Deposit,
    pub(super) plug: f64,
}

impl Term {
    pub(super) fn cost(&self) -> f64 {
        self.fit + self.logdet + self.plug
    }
}

/// Newton steps at most per mode.
pub(super) const MAX_NEWTON: usize = 40;

/// One scan's data for the elements `score` (indices into `elems`), one per term (`slot`, per element: [`slots`]),
/// from the cells they may front: `cells`, in index order, each with its k-buffer as its pulse reads it
/// ([`super::render::kept`]), its layers' `elem` indices into `elems`. A cell's evidence belongs to its
/// layers' terms by their front shares ([`front_shares`]: wholly its front element's where that is wholly in it); its echoes read the cell's own surface class ([`cell_class`]) and its shift the term's
/// alignment spread ([`term_classes`]), so a term depends on no other term's class. With the foliage field's box
/// `field`, each cell holds the hits the field counts of its echoes, for the terms' deposits.
#[allow(clippy::too_many_arguments)]
pub(super) fn elem_data(v: &RangeImage, cells: &[(usize, Vec<Layer>)], elems: &[Element], slot: &[usize], score: &[usize], ref_epoch: u32,
             em: &Settings, sc: &Background, ext: &Foliage, field: Option<&FoliageBox>) -> Vec<ElemData> {
    let cls = term_classes(v, cells, slot, em);
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
    // cells in index order, so an element's sums run in the same order in every run
    for (cell, ls) in cells {
        let cell = *cell;
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
        let base = cv.empty.read(0.0, 0.0, em).ll;
        let hits = field.map_or(Vec::new(), |sp| foliage::hits(v, sp, em, cell));
        for (d, s) in shares {
            let hits = hits.iter().map(|&(i, k, w)| (i, k, w * s, ext.sigma(&k).ln())).collect();
            data[d].cells.push(CellData { cell, pulses: cv.pulses.clone(), wt: cv.wt * s, base, outcome, hits });
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;
    use std::sync::Arc;
    use crate::scene::foliage::{self, Foliage};
    use crate::scene::pulse::mixture_taken;
    use crate::scene::range_image::Cell;
    use crate::scene::render::{Form, Class, Element, KBuffer, render_all};
    use crate::scene::settings::{Background, Settings};
    use super::super::fixtures::{cost, field_box, piece_of, pole_scan_of, pole_view, stick, thin_pole_view};

    /// Soft attribution in the pole's interior (fill over 0.99), where who made each echo is plain: what its term's
    /// responsibilities take of the field's hits is within 1 % of every echo at or behind its front (`3 sd + 3 cm`
    /// before its depth, at the noisiest surface class's sd) taken whole.
    #[test]
    fn the_elements_take_the_poles_interior() {
        let v = pole_view();
        let (em, sc, sp) = (Settings::default(), Background::default(), field_box());
        let e = stick(0.0, 4.0, 0.03);
        let rendered = render_all(&v, &[e], &em);
        let cells = KBuffer::of(&rendered).cells();
        let d = &elem_data(&v, &cells, &[e], &[0], &[0], 0, &em, &sc, &Foliage::uniform(sc.kappa), Some(&sp))[0];
        // each cell's front layer, in elem_data's order
        let order: Vec<&(usize, Vec<Layer>)> = cells.iter().filter(|x| v.cell[x.0] != Cell::NoPulse).collect();
        assert_eq!(d.cells.len(), order.len());
        let front = |c: &CellData| order[d.cells.iter().position(|x| std::ptr::eq(x, c)).unwrap()].1[0];
        let interior = |c: &CellData| front(c).f > 0.99;
        let behind = |c: &CellData| -> Vec<f64> {
            let l = front(c);
            c.pulses[0].1.echoes.iter().map(|&r| (interior(c) && r as f64 >= l.t - 3.0 * em.depth_sd(2, l.spread) - 0.03) as u8 as f64).collect()
        };
        let soft = |c: &CellData| if interior(c) { mixture_taken(&c.pulses, 0.0, 0.0, &em).1 } else { vec![0.0; c.pulses[0].1.echoes.len()] };
        let total = |taken: &dyn Fn(&CellData) -> Vec<f64>| -> f64 {
            foliage::deposit(d.cells.iter().flat_map(|c| c.take(&taken(c)).0))
                .iter().map(|x| x.1 as f64).sum()
        };
        let (whole, soft) = (total(&behind), total(&soft));
        assert!(whole > 0.0 && (soft / whole - 1.0).abs() < 0.01, "{soft} vs {whole}");
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
            let d = &elem_data(&v, &KBuffer::of(&rendered).cells(), &[e], &[0], &[0], 0, &em, &sc, &Foliage::uniform(sc.kappa), None)[0];
            assert!(d.own_sd[0] > 0.0 && d.own_sd[1] > 0.0);
            for x in [(0.0, 0.0), (0.002, -0.4)] {
                let q = d.quad(x, &em, false);
                let f = |dx: f64, db: f64| -d.quad((x.0 + dx, x.1 + db), &em, false).ll;
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
            let at = d.quad(d.at(0.0, cold.own), &em, false);
            for i in 0..2 {
                assert!((d.own_sd[i] * at.g[i] + cold.own[i]).abs() < 0.01, "slope in y {:?} at {:?}", at.g, cold.own);
            }
        }
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
        let patch = Element { form: Form::Patch { centre: [0.0, 4.0, 0.0] }, class: Class::Leaf, r: 0.03, opacity: 1.0, mismatch: 0.002, group: None, key: None };
        assert_eq!(rows(patch).front_shift_sd, 0.0);
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
}
