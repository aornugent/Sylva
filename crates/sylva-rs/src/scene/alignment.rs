// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The scan alignment error (3dtrees research 130 offset-field-design.md): its nodes, its joint solve with the terms'
//! own nodes, and the alignment held between refits.
//!
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

use crate::Point;
use nalgebra::Vector2;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use super::settings::Settings;
use super::term::{Owner, ElemData, Eliminated, cap, Local, Term, MAX_NEWTON};

/// The scan alignment error's nodes about a place: trilinear hats on a grid of spacing ℓ, normalised so the alignment
/// error's variance is the same everywhere (`t(x) = Σ w_j z_j / √Σ w_j²`); each node carries the along-beam shift.
pub fn hats(x: &Point, l: f64) -> Vec<([i64; 3], f64)> {
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

/// The scan alignment error held at its mode (research 130 offset-field-design.md §3, step 3a): the nodes' mode ẑ
/// and covariance Σ₀ = Λ₀⁻¹ at the last refit, `m` = Σ₀ G₀ (the step Newton would still take, G₀ the cost's slope in
/// the nodes there) and `gm` = G₀ᵀ Σ₀ G₀, and what it reads of the terms the refit was taken with (`reference`, by
/// [`Owner`]: [`Reference`]).
///
/// With its own nodes eliminated, each term (an element's or a group's) meets the nodes through its alignment shift alone: slope γ and
/// curvature h along its weights r. For the elements now in the scan, `Λ = I + Σ h r rᵀ` and `G = ẑ + Σ γ r`, and the
/// scan's alignment part is the Laplace marginal of that quadratic about ẑ, `½ ẑᵀẑ + ½ ln|Λ| − ½ Gᵀ Λ⁻¹ G`. Only the
/// elements whose terms differ from the reference enter, as a rank-k correction to Λ₀ (Woodbury; a node new since the
/// refit has its prior alone), so a move costs its own neighbourhood, not the scan, and the cost is a function of the
/// scene, not of the moves that led to it.
pub(super) struct Held {
    pub(super) index: HashMap<[i64; 3], usize>,
    pub(super) z: Vec<f64>,
    pub(super) cov: nalgebra::DMatrix<f64>,
    pub(super) m: Vec<f64>,
    pub(super) gm: f64,
    pub(super) logdet: f64,
    pub(super) reference: BTreeMap<Owner, Reference>,
}

/// What the held alignment keeps of a term at its refit: how it meets the nodes (its weights, its cost's slope and
/// curvature in its shift) and where its own nodes' Newton began, so a move recomputes an unchanged term exactly.
#[derive(Debug, Clone)]
pub(super) struct Reference {
    pub(super) align: Vec<([i64; 3], f64)>,
    pub(super) slope: f64,
    pub(super) curv: f64,
    pub(super) start: Vector2<f64>,
}

impl Reference {
    pub(super) fn of(t: &Term) -> Reference {
        Reference { align: t.align.clone(), slope: t.slope, curv: t.curv, start: t.start }
    }

    /// Whether term `t` enters the scan alignment error as this did (the same nodes, slope and curvature).
    pub(super) fn same_alignment(&self, t: &Term) -> bool {
        self.align == t.align && self.slope == t.slope && self.curv == t.curv
    }
}

impl Held {
    /// The alignment shift at weights `r`.
    pub(super) fn shift(&self, r: &[([i64; 3], f64)]) -> f64 {
        r.iter().map(|(k, w)| self.index.get(k).map_or(0.0, |&i| w * self.z[i])).sum()
    }

    pub(super) fn cov_at(&self, a: &[i64; 3], b: &[i64; 3]) -> f64 {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.cov[(i, j)],
            _ => (a == b) as u8 as f64,
        }
    }

    /// `rᵀ Σ₀ r2`.
    pub(super) fn between(&self, r: &[([i64; 3], f64)], r2: &[([i64; 3], f64)]) -> f64 {
        r.iter().map(|(a, wa)| r2.iter().map(|(b, wb)| wa * wb * self.cov_at(a, b)).sum::<f64>()).sum()
    }

    /// The scan's alignment part, given the columns (weights, curvature change, slope change) of the terms that differ
    /// from the reference.
    pub(super) fn part(&self, cols: &[(&[([i64; 3], f64)], f64, f64)]) -> f64 {
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
pub(super) fn alignment_nodes(data: &[ElemData]) -> HashMap<[i64; 3], usize> {
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    for el in data {
        for (k, _) in &el.align {
            let n = index.len();
            index.entry(*k).or_insert(n);
        }
    }
    index
}

pub(super) fn alignment_system<'a>(index: &HashMap<[i64; 3], usize>, z: &[f64], terms: impl Iterator<Item = (&'a [([i64; 3], f64)], f64, f64)>)
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
pub(super) fn held(index: HashMap<[i64; 3], usize>, z: Vec<f64>, grad: nalgebra::DVector<f64>, hess: nalgebra::DMatrix<f64>) -> Held {
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

/// A scan's exact Laplace marginal over the alignment error's nodes, given each element's data (the held
/// alignment's refit; 3dtrees research 130 optimisation.md): one Newton over the nodes and every term's
/// own nodes together. Terms are independent given the alignment and each meets it through its shift alone, so the
/// joint curvature is an arrowhead: each term's own block is eliminated by its Schur complement
/// ([`ElemData::eliminate`]), the nodes step on the system that leaves, and each term's own nodes follow, `−P⁻¹(g_y +
/// link Δa)`; both steps at most 3 prior sds, halved together until the total cost falls (each term read once a
/// trial), stopping under a millinat. Then each term's own Newton finishes from the joint mode ([`ElemData::term`]);
/// the precision is the final terms'. Its value is the held alignment's at its own reference ([`Held::part`] with no
/// columns).
pub(super) fn solve_alignment(data: &[ElemData], em: &Settings) -> (Held, Vec<Term>) {
    let index = alignment_nodes(data);
    let n = index.len();
    let shift = |el: &ElemData, z: &[f64]| -> f64 { el.align.iter().map(|(k, w)| w * z[index[k]]).sum() };
    let read = |z: &[f64], ys: &[Vector2<f64>]| -> Vec<Local> { data.par_iter().zip(ys).map(|(el, y)| el.quad(el.at(shift(el, z), *y), em, false)).collect() };
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
pub(super) struct Fit {
    pub(super) terms: BTreeMap<Owner, Term>,
    pub(super) held: Held,
    pub(super) cost: f64,
}

/// A move's changes to a scan's fit: the terms by owner (none: it left the scan).
pub(super) type Changes = Vec<(Owner, Option<Term>)>;

impl Fit {
    /// The scan's cost with the terms `over` (none: the term left) in place of its own: every term's cost plus the
    /// held alignment's part ([`Held::part`]), its columns the terms that differ from the reference, in owner order.
    pub(super) fn total(&self, over: &HashMap<Owner, Option<Term>>) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::scene::foliage::Foliage;
    use crate::scene::pulse::{cell_class, occluders, Pulse};
    use crate::scene::range_image::Cell;
    use crate::scene::render::{Element, Layer, KBuffer, render_all, mixture};
    use crate::scene::scene::Scene;
    use crate::scene::settings::{Background, Settings};
    use crate::scene::term::{term_classes, ElemData};
    use super::super::fixtures::{cost, pole_image, pole_view, stick};

    /// The joint arrowhead Newton ([`solve_alignment`]) ends at the scan's minimum: on the pole alone and with a stick
    /// beside it (sharing its node), moving any alignment node by ±0.2 prior sd, each term's own nodes re-solved
    /// ([`ElemData::term`]), costs no less than the solve's stopping rule allows (a millinat).
    #[test]
    fn the_joint_newton_ends_at_the_minimum() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        for els in [vec![stick(0.0, 4.0, 0.03)], vec![stick(0.0, 4.0, 0.03), stick(0.15, 4.5, 0.02)]] {
            let c = Scene::new(vec![v.clone()], &els, em, sc, None, 0);
            let data: Vec<ElemData> = c.data(0).into_iter().map(|d| d.1).collect();
            let (held, terms) = solve_alignment(&data, &em);
            let fit = |z: &[f64]| 0.5 * z.iter().map(|x| x * x).sum::<f64>()
                + data.iter().zip(&terms).map(|(el, t)| el.term(el.align.iter().map(|(k, w)| w * z[held.index[k]]).sum(), t.own, &em).fit).sum::<f64>();
            let at = fit(&held.z);
            assert!(!held.z.is_empty());
            for i in 0..held.z.len() {
                for d in [-0.2, 0.2] {
                    let mut z = held.z.clone();
                    z[i] += d;
                    assert!(fit(&z) > at - 1e-3, "node {i} moved {d}: {} vs {at}", fit(&z));
                }
            }
        }
    }

    /// A scene's moves hold the alignment (step 3a): each move's cost change stays near the exact marginal's
    /// (re-solved), for an add beside the pole (sharing its alignment node), an alter and a remove; adding then removing
    /// an element restores the cost exactly (synthesis §18 invariant 7); `costs_to_add` equals
    /// the move; and after `refit_alignment` the cost is a fresh scene's.
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
        let d_remove = c.remove(id);
        assert!(near(d_remove, full(&[wide]) - full(&[wide, side])), "remove {d_remove} exact {}", full(&[wide]) - full(&[wide, side]));
        c.refit_alignment();
        assert!((c.cost() - full(&[wide])).abs() < 1e-9 * c.cost().abs().max(1.0), "{} vs {}", c.cost(), full(&[wide]));
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
        let buf = KBuffer::of(&rendered).cells();
        let s_along = em.align_sd(term_classes(&v, &buf, &[0], &em)[0], false).hypot(em.front_shift_sd(el.r));
        let ed = |shift: f64, level: f64| -> f64 {
            buf.iter().map(|&(cell, ref ls)| {
                let echoes = v.echoes(cell);
                let wt = em.cell_weight(echoes.first().map_or(ls[0].t, |&r| r as f64), v.step());
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let pulse = |ls: Vec<Layer>| Pulse::on(occluders(ls, None), cell_class(&v, cell, &em), &Foliage::uniform(sc.kappa).medium(&v.origin, &w), echoes, v.echo_refls(cell), v.max_targets, &em);
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
}
