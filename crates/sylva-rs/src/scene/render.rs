// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Elements and their rendering into a view's cells (3dtrees research 130): an element's layers, the k-buffer of a
//! cell's layers and its mixture over the partial ones.
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
//! - **Depth over a footprint:** a layer's depth is the mean of the section's depth over the part of the footprint
//!   it meets, weighted by each part's energy (the cosine of incidence), as a scanner's mixed echo is
//!   ([`footprint_depth`]). The mismatch spreads the depth but doesn't move it (slice 7b).

use crate::Point;
use crate::transform::{cross, norm, normalize, sub};
use rayon::prelude::*;
use std::collections::HashMap;
use super::dual::{Dual, Real};
use super::range_image::RangeImage;
use super::settings::Settings;

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

impl Element {
    pub(super) fn mid(&self) -> Point {
        match self.form {
            Form::Stick { a, b } => [0, 1, 2].map(|k| 0.5 * (a[k] + b[k])),
            Form::Patch { centre } => centre,
        }
    }

    /// Its surface's reflectance (dB at full fill, face-on), by its class.
    pub(super) fn refl(&self, em: &Settings) -> f64 {
        match self.class {
            Class::Wood => em.refl_wood,
            Class::Leaf => em.refl_leaf,
        }
    }

    /// Its stem front shift's prior sd (m): a stick's section departs from round as each view sees it; a patch's disc
    /// has no front to shift.
    pub(super) fn front_shift_sd(&self, em: &Settings) -> f64 {
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

/// One element's layer in one cell: what the ray likelihood reads of it, its geometry's quantities in `P` (`f64`, or a
/// [`Dual`] carrying their slope in the element's parameters: [`Scene::information`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Layer<P = f64> {
    pub(super) t: P,
    /// The range of the section's far side where the beam meets it: where the medium behind it starts.
    pub(super) far: P,
    pub(super) spread: P,
    /// The fill: its mean over the mismatch (`f`), and at section radii r − √3 m, r and r + √3 m (`fq`, the
    /// mismatch's three-point Gauss–Hermite rule, weights 1/6, 2/3, 1/6).
    pub(super) f: P,
    pub(super) fq: [P; 3],
    pub(super) cos: P,
    /// The surface's reflectance (dB at full fill, face-on), by the element's class.
    pub(super) db: f64,
    pub(super) elem: u32,
    pub(super) op: f64,
    /// Where the cell's ray passes the axis: across it (signed, m, positive on `u × (c − o)`'s side) and along it;
    /// for a patch, the distance from the centre and 0.
    pub(super) x: f64,
    pub(super) s: f64,
}

impl<P: Real> Layer<P> {
    /// Its entry weight (3dtrees research 130 slice-trim): the chance it is part of its cell, a smooth step (C¹) in its
    /// widest fill from 0 at [`MIN_FILL`] to 1 at [`FULL_FILL`]. A cell's likelihood is the mixture over which of its
    /// partial layers are in it ([`mixture`]), so a layer's part in the cost goes to 0 with its fill, slope and all.
    pub(super) fn entry(&self) -> P {
        smooth((self.fq[2] - MIN_FILL) / (FULL_FILL - MIN_FILL))
    }

    /// Whether it enters the k-buffer: an entry weight above 0.
    pub(super) fn enters(&self) -> bool {
        self.fq[2].v() > MIN_FILL
    }

    /// Whether it is wholly part of its cell: an entry weight of 1.
    pub(super) fn full(&self) -> bool {
        self.fq[2].v() >= FULL_FILL
    }
}

impl Layer {
    /// The layer in another number type, its quantities constants there.
    pub(super) fn lift<P: Real>(&self) -> Layer<P> {
        let c = P::c;
        Layer { t: c(self.t), far: c(self.far), spread: c(self.spread), f: c(self.f), fq: self.fq.map(c), cos: c(self.cos), db: self.db, elem: self.elem,
                op: self.op, x: self.x, s: self.s }
    }
}

/// A smooth step: 0 at `x` ≤ 0, 1 at `x` ≥ 1, `3x² − 2x³` between (its slope continuous). Where the cost would cut, it
/// fades (3dtrees research 130 slice-trim).
pub(super) fn smooth<P: Real>(x: P) -> P {
    let x = x.clamp(0.0, 1.0);
    x * x * (P::c(3.0) - x * 2.0)
}

/// The share of the beam (a uniform disc of radius `a`) a section of radius `r` meets whose axis passes `d` from the
/// beam's centre, and its slope in ln r: a layer's fill at one section radius ([`layer`]'s), for what a consumer reads
/// beside the cost.
pub fn beam_fill(d: f64, r: f64, a: f64) -> (f64, f64) {
    let r = Dual::<1>::var(r.ln(), 0).exp();
    let f = disc_cdf((r - d) / a) - disc_cdf((-r - d) / a);
    (f.v(), f.g[0])
}

/// The share of a uniform disc's energy (radius 1) nearer than a line `z` radii from its centre: the CDF of its 1-D
/// marginal, a semicircle.
pub(super) fn disc_cdf<P: Real>(z: P) -> P {
    let z = z.clamp(-1.0, 1.0);
    (z * (P::c(1.0) - z * z).sqrt() + z.asin()) / std::f64::consts::PI + 0.5
}

pub(super) fn dot3<P: Real>(a: &[P; 3], b: &[P; 3]) -> P {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// An element placed in one view ([`placed`]): its section's radius, its middle from the scanner (`c`) and, a stick's,
/// its unit axis, its half-length and the unit normal to the axis and `c` (the side `across` is positive on).
pub(super) struct Placed<P> {
    pub(super) r: P,
    pub(super) c: [P; 3],
    pub(super) axis: Option<([P; 3], P, [P; 3])>,
}

/// Element `e` placed in a view from `o`, moved by `th` in its parameters if given ([`Scene::information`]'s): a
/// stick's ln r, its axis moved across itself (along p₁ and p₂, perpendicular to it) and its ends by the tilts about
/// its middle (the same directions, rad at its half-length); a patch's ln r and its centre moved (x, y, z).
pub(super) fn placed<P: Real>(e: &Element, o: &Point, th: Option<&[P]>) -> Placed<P> {
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
pub(super) fn render(v: &RangeImage, e: &Element, id: u32, em: &Settings) -> Vec<(usize, Layer)> {
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
pub(super) fn layer<P: Real>(p: &Placed<P>, w: &Point, e: &Element, id: u32, em: &Settings, step: f64) -> Option<Layer<P>> {
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
pub(super) fn footprint_depth<P: Real>(tm: P, d: P, r: P, sin: P, a: P) -> (P, P, P) {
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
pub(super) fn render_all(v: &RangeImage, elems: &[Element], em: &Settings) -> Vec<Vec<(usize, Layer)>> {
    elems.par_iter().enumerate().map(|(i, e)| render(v, e, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, the layers that enter it ([`Layer::enters`]), nearest first ([`keep`]).
pub(super) fn kbuffer(per: &[Vec<(usize, Layer)>]) -> HashMap<usize, Vec<Layer>> {
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
pub(super) fn keep<X>(xs: &mut Vec<X>, t: impl Fn(&X) -> f64, full: impl Fn(&X) -> bool) {
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
pub(super) fn mixture<P: Real>(layers: &[Layer<P>]) -> Vec<(P, Vec<Layer<P>>)> {
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
pub(super) fn front_shares<P: Real>(layers: &[Layer<P>]) -> Vec<P> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::scene::Scene;
    use crate::scene::settings::{Background, Settings};
    use super::super::fixtures::{cost, pole_view, stick, thin_pole_view};

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
}
