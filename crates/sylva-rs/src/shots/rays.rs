// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The ray likelihood of element hypotheses (3dtrees research 130, terms A and C): the energy of a set of elements
//! against some scans' [`View`]s, as `-log` of its likelihood ratio against the empty scene, the energy change of
//! adding or removing one, and what the elements predict in a view.
//!
//! - **Elements** ([`Element`]): a stick (an axis, its section's mean radius a₀) or a leaf patch (a ball with its
//!   extent, a disc in every view), each with an opacity and a misfit. Opacity is the chance a pulse the section
//!   fills meets solid there, its gaps wider than the footprint (synthesis §1's "extent, opacity"). Misfit is the
//!   sd of the section's and axis's departure from the form, which blurs the fill and spreads the depth.
//! - **The k-buffer:** each element renders into each view's cells. A cell keeps up to [`K`] layers, nearest
//!   first, each with a depth, a depth spread, a fill and a cosine. Fill is the footprint (a Gaussian, sd =
//!   footprint / 4, and the misfit) convolved with the section. Along a stick's axis it is cut square, so a chain of
//!   sticks joins without seams.
//! - **Term A, one cell:** its first exported echo, or none, against a diffuse exported-echo hazard κ along the ray
//!   (the crude field), the layers in depth order, then a ground plane (the crude ground).
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is kept with
//!     `op_i sel_i`, where `sel_i = (1 − d) Φ((ρ_dB + 10 log10(e_i cos ι_i) − cut) / σ_refl)`: the export's
//!     reflectance cut as a selection on fill, and d the view's measured in-surface deletion
//!     ([`View::deletion`]).
//!   - An echo at r has density `Σ_i P_i op_i sel_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i)` plus
//!     `κ e^{−κr} Π_{t_j<r} (1 − op_j sel_j)`, where P_i is the chance that nothing nearer was kept, plus a uniform
//!     outlier share. An empty cell has probability `e^{−κ t_end} Π (1 − op_j sel_j)`.
//! - **Term C, a stick's located silhouette edges** ([`edge_ll`]): the fill band under 0.5 leaves A's cells for C,
//!   so no echo counts twice.
//! - **Depth over a footprint:** a layer's depth is the mean of the section's depth over the part of the footprint
//!   it meets, weighted by each part's energy (the cosine of incidence), as a scanner's mixed echo is
//!   ([`footprint_depth`]). The misfit spreads the depth but doesn't move it (slice 7b).
//! - **The offset field** (research 130 offset-field-design.md, slices 7 and 7b): per scan, an along-beam offset
//!   on nodes ℓ apart (trilinear hats, normalised) with a unit Gaussian prior, scaled at each element by its surface
//!   class's spread (and the other-epoch part for a scan off the reference epoch). A cell's depths shift by the
//!   field at its front element. Neighbours sharing nodes share one offset, and one scan's joint evidence on a
//!   place is capped by the prior. (An across-beam part didn't help held-out calibration; slice 7b dropped it.) The surface class k (which sets the along
//!   spread, the bias and the noise) is the median class of the view's echoes on the element's front.
//! - **The shape latent** (slice 7b, off by default): a stick's section less its circle as a random function of
//!   the angle about its axis ([`Block`]), seen by each scan at its front (a depth) and its two edges (moving the
//!   fill and term C). It couples the scans that see the stick.
//! - **The energy** is the Laplace marginal of every scan's data over all these latents ([`joint`]).
//! - **Weights:** each cell counts `1/m(ρ)`, with m the footprints per cell (overlapping footprints). An element's
//!   terms in a scan are tempered by the residual design effect `1 + (n − 1) ρ`, n its fill-weighted footprints:
//!   the occupancy correlation the latents don't carry (design §4's ρ_res). Without it, untempered occupancy
//!   rewards covering more cells, with or without the shape latent (slice 7b).
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use super::view::{state, ErrorModel, View};
use crate::transform::{cross, dot, norm, normalize, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::HashMap;

/// Layers kept per cell. A cell that runs out (rare among a few elements) drops its farthest.
pub const K: usize = 3;

/// An element's form.
#[derive(Debug, Clone, Copy)]
pub enum Form {
    /// An axis from `a` to `b` (world, m).
    Stick { a: Point, b: Point },
    /// A ball about `centre`: a leaf patch or clump, its section a disc in every view.
    Patch { centre: Point },
}

/// An element hypothesis: its form, its section's radius (a stick's a₀; a patch's extent), its opacity (1 solid)
/// and its misfit sd (m). Opacity and misfit are latents the caller supplies.
#[derive(Debug, Clone, Copy)]
pub struct Element {
    pub form: Form,
    pub r: f64,
    pub opacity: f64,
    pub misfit: f64,
}

impl Element {
    fn mid(&self) -> Point {
        match self.form {
            Form::Stick { a, b } => [0, 1, 2].map(|k| 0.5 * (a[k] + b[k])),
            Form::Patch { centre } => centre,
        }
    }
}

/// The crude scene behind the elements: the diffuse exported-echo hazard (1/m), a ground plane's height (NaN for
/// none), the elements' and the ground's reflectance (dB at full fill, face-on) and the longest range.
#[derive(Debug, Clone, Copy)]
pub struct Scene {
    pub kappa: f64,
    pub ground_z: f64,
    pub wood_db: f64,
    pub ground_db: f64,
    pub t_max: f64,
}

impl Default for Scene {
    fn default() -> Self {
        Scene { kappa: 0.05, ground_z: f64::NAN, wood_db: -13.8, ground_db: -20.0, t_max: 80.0 }
    }
}

/// What a cell's first echo says about a layer: on it, none, in front of it, or behind it (the pulse went through).
pub mod code {
    pub const ON: u8 = 0;
    pub const EMPTY: u8 = 1;
    pub const FRONT: u8 = 2;
    pub const BEHIND: u8 = 3;
}

/// The cell's code against a layer whose echo is predicted at `depth` with sd `sd`: on within `3 sd + 1 cm`, in
/// front beyond `3 sd + 3 cm`, else behind. The margins every consumer uses.
pub fn cell_code(obs: f64, depth: f64, sd: f64) -> u8 {
    if !obs.is_finite() {
        code::EMPTY
    } else if obs < depth - 3.0 * sd - 0.03 {
        code::FRONT
    } else if (obs - depth).abs() < 3.0 * sd + 0.01 {
        code::ON
    } else {
        code::BEHIND
    }
}

#[derive(Debug, Clone, Copy)]
struct Layer {
    t: f64,
    spread: f64,
    f: f64,
    cos: f64,
    elem: u32,
    op: f64,
    /// Where the cell's ray passes the axis: across it (signed, m, positive on `u × (c − o)`'s side) and along it;
    /// for a patch, the distance from the centre and 0.
    x: f64,
    s: f64,
}

/// What one scan says about one element. `cells`: cells where it is the nearest layer at fill > 0.5; `support`:
/// those with an echo on it; `across`: footprints across its diameter; `energy`: its share of the scan's energy
/// (its A and C terms at the field's mode, plus an equal share of the field's prior and log-determinant terms), so
/// the rows sum to the total.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub elem: u32,
    pub scan: u32,
    pub cells: u32,
    pub support: u32,
    pub across: f64,
    pub surface: u8,
    pub energy: f64,
    /// The offset field's posterior at the element in this scan: the along-beam shift and its sd (m).
    pub shift_along: f64,
    pub sd_along: f64,
    /// The stick's shape latent at the scan's front angle: its section's radius there less a₀ (m; 0 without one).
    pub shape: f64,
}

fn phi(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

fn t_pdf(x: f64, nu: f64) -> f64 {
    let c = libm::lgamma(0.5 * (nu + 1.0)) - libm::lgamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln();
    (c - 0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln()).exp()
}

/// One element's layers in one view, out to 3 fill sds and two cells past its section (their fill may be ~0, so
/// term C sees what lies beyond each edge).
fn render(v: &View, e: &Element, id: u32, em: &ErrorModel) -> Vec<(usize, Layer)> {
    let mut out = Vec::new();
    if !(e.r > 0.0) {
        return out;
    }
    let (u, half) = match e.form {
        Form::Stick { a, b } => {
            let ax = sub(&b, &a);
            let len = norm(&ax);
            if !(len > 0.0) {
                return out;
            }
            (Some(ax.map(|x| x / len)), 0.5 * len)
        }
        Form::Patch { .. } => (None, 0.0),
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
    for l in (li - dl).max(0)..=(li + dl).min(v.n_lines as i64 - 1) {
        for k in (ki - dc).max(0)..=(ki + dc).min(v.n_cols as i64 - 1) {
            let w = v.direction(l as f64, k as f64);
            // the ray's closest approach to the axis (to the centre for a patch): range, distance, signed place
            // across, place along, and the sine of the ray's angle to the axis
            let (tm, d, x, along, sin) = match u {
                None => {
                    let tm = dot(&w, &c);
                    let d = norm(&[0, 1, 2].map(|j| tm * w[j] - c[j]));
                    (tm, d, d, 0.0, 1.0)
                }
                Some(u) => {
                    let wu = dot(&w, &u);
                    let a = [0, 1, 2].map(|j| w[j] - wu * u[j]);
                    let aa = dot(&a, &a);
                    if aa < 1e-12 {
                        continue;
                    }
                    let cu = dot(&c, &u);
                    let bv = [0, 1, 2].map(|j| cu * u[j] - c[j]);
                    let ab = dot(&a, &bv);
                    let tm = -ab / aa;
                    let along = tm * wu - cu;
                    let q = [0, 1, 2].map(|j| tm * w[j] - c[j] - along * u[j]);
                    (tm, norm(&q), dot(&q, &normalize(&cross(&u, &c))), along, aa.sqrt())
                }
            };
            if tm <= 0.0 || along.abs() > half {
                continue;
            }
            let fp = 0.25 * em.footprint(tm);
            let sd = (fp * fp + e.misfit * e.misfit).sqrt();
            let f = phi((e.r - d) / sd) - phi((-e.r - d) / sd);
            if d > e.r + 3.0 * sd + 2.0 * em.step * tm {
                continue;
            }
            let (t, spread, cos) = footprint_depth(tm, d, e.r, sin, fp);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t,
                    spread: (spread * spread + e.misfit * e.misfit).sqrt(),
                    f,
                    cos: (cos * sin).max(0.05),
                    elem: id,
                    op: e.opacity,
                    x,
                    s: along,
                },
            ));
        }
    }
    out
}

/// The echo's depth over a footprint (Gaussian across the section, sd `fp`) whose centre passes `d` from the axis
/// at range `tm`: the mean of the section's depth over the part it meets, each part weighted by its energy (the
/// cosine of incidence, Lambertian); the spread of those depths; and the mean cosine. The misfit stays out: it
/// spreads the depth about the form but doesn't move it.
fn footprint_depth(tm: f64, d: f64, r: f64, sin: f64, fp: f64) -> (f64, f64, f64) {
    const GH: [(f64, f64); 5] = [(0.0, 0.533333), (1.355626, 0.222076), (-1.355626, 0.222076), (2.856970, 0.011257), (-2.856970, 0.011257)];
    let (mut sw, mut swc, mut sz, mut szz) = (0.0, 0.0, 0.0, 0.0);
    for (u, w) in GH {
        let x = d + u * fp;
        if x.abs() < r {
            let c = (1.0 - (x / r).powi(2)).sqrt();
            let z = -r * c / sin;
            sw += w;
            swc += w * c;
            sz += w * c * z;
            szz += w * c * z * z;
        }
    }
    if swc <= 0.0 {
        // past the edge: the grazing surface
        let c = (1.0 - (d.min(r) / r).powi(2)).max(0.0).sqrt();
        return (tm - r * c / sin, 0.0, c);
    }
    let z = sz / swc;
    (tm + z, (szz / swc - z * z).max(0.0).sqrt(), swc / sw)
}

/// Every element rendered into one view.
fn render_all(v: &View, elems: &[Element], em: &ErrorModel) -> Vec<Vec<(usize, Layer)>> {
    elems.par_iter().enumerate().map(|(i, e)| render(v, e, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, its layers of fill `min_f` or more, nearest first.
fn kbuffer(per: &[Vec<(usize, Layer)>], min_f: f64) -> HashMap<usize, Vec<Layer>> {
    let mut buf: HashMap<usize, Vec<Layer>> = HashMap::new();
    for &(cell, ly) in per.iter().flatten() {
        if ly.f >= min_f {
            buf.entry(cell).or_default().push(ly);
        }
    }
    for ls in buf.values_mut() {
        ls.sort_by(|a, b| a.t.total_cmp(&b.t));
        ls.truncate(K);
    }
    buf
}

/// The probability an echo of energy share `e` (fill × cos) off a surface of `db` survives the export, with `drop`
/// the export's loss inside surfaces.
fn sel(drop: f64, em: &ErrorModel, db: f64, e: f64) -> f64 {
    (1.0 - drop) * phi((db + 10.0 * e.max(1e-12).log10() - em.cut_db) / em.refl_noise_db)
}

/// The export's in-surface loss for a view: the error model's, or the view's measured one.
fn drop_of(em: &ErrorModel, v: &View) -> f64 {
    if em.p_drop.is_nan() { v.deletion } else { em.p_drop }
}

/// Each element's surface class in one view: the median class of the view's echoes on its front (mid where none).
fn classes(v: &View, buf: &HashMap<usize, Vec<Layer>>, n: usize, em: &ErrorModel) -> Vec<u8> {
    let mut ex: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (&cell, ls) in buf {
        let ly = ls[0];
        let r = v.range[cell] as f64;
        if ly.f > 0.5 && v.state[cell] == state::ECHO && (r - ly.t).abs() < 0.03 && v.excess[cell].is_finite() {
            ex[ly.elem as usize].push(v.excess[cell] as f64);
        }
    }
    ex.iter_mut()
        .map(|e| {
            if e.is_empty() {
                1
            } else {
                e.sort_by(f64::total_cmp);
                em.surface(e[e.len() / 2])
            }
        })
        .collect()
}

/// What the elements predict in one view, per touched cell with a layer of fill `min_fill` or more: its nearest
/// layer (element, fill, the chance its echo is kept, the echo's predicted depth with the bias, its sd: noise,
/// spread, misfit and the offset field's along spread), the element's surface class, the observed first echo (NaN for
/// none), the cell's code against the layer ([`cell_code`]), its weight `1/m(ρ)`, the footprint (m) at the layer,
/// and where its ray passes the axis (across, along). For held-out checks: the cell arrays term A scores, without the scene.
pub struct Prediction {
    pub cell: usize,
    pub elem: u32,
    pub fill: f64,
    pub kept: f64,
    pub depth: f64,
    pub sd: f64,
    pub surface: u8,
    pub obs: f64,
    pub code: u8,
    pub weight: f64,
    pub footprint: f64,
    pub x: f64,
    pub s: f64,
}

pub fn predict(v: &View, elems: &[Element], em: &ErrorModel, sc: &Scene, min_fill: f64) -> Vec<Prediction> {
    let buf = kbuffer(&render_all(v, elems, em), min_fill);
    let cls = classes(v, &buf, elems.len(), em);
    let mut out: Vec<Prediction> = buf
        .iter()
        .map(|(&cell, ls)| {
            let ly = ls[0];
            let k = cls[ly.elem as usize];
            let depth = ly.t + em.depth_bias(k, ly.t);
            let off = if em.field { em.field_along[k as usize] } else { 0.0 };
            let sd = (em.sigma_e[k as usize].powi(2) + ly.spread.powi(2) + off * off).sqrt();
            let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
            Prediction {
                cell, elem: ly.elem, fill: ly.f, kept: ly.op * sel(drop_of(em, v), em, sc.wood_db, ly.f * ly.cos), surface: k,
                depth, sd, obs, code: cell_code(obs, depth, sd), weight: 1.0 / em.multiplicity(ly.t), footprint: em.footprint(ly.t), x: ly.x, s: ly.s,
            }
        })
        .collect();
    out.sort_by_key(|p| p.cell);
    out
}

/// One cell's log-likelihood: its first echo `obs` (NaN for none) against its layers (each with its element's
/// surface class), the ground at `ground` = (range, cos) and the field, with every layer shifted by `delta`.
fn cell_ll(obs: f64, layers: &[(Layer, u8)], ground: Option<(f64, f64)>, delta: f64, drop: f64, em: &ErrorModel, sc: &Scene) -> f64 {
    let t_end = ground.map_or(sc.t_max, |g| g.0);
    let (mut beam, mut none_kept, mut none_before_obs, mut dens) = (1.0, 1.0, 1.0, 0.0);
    for &(ly, k) in layers {
        if ly.t >= t_end {
            break;
        }
        let e = ly.f * beam;
        beam *= 1.0 - ly.f * ly.op;
        let s = ly.op * sel(drop, em, sc.wood_db, e * ly.cos);
        if obs.is_finite() {
            let sd = (em.sigma_e[k as usize].powi(2) + ly.spread * ly.spread).sqrt();
            let mu = ly.t + em.depth_bias(k, ly.t) + delta;
            dens += (-sc.kappa * ly.t).exp() * none_kept * s * t_pdf((obs - mu) / sd, em.nu_e) / sd;
            if ly.t < obs {
                none_before_obs *= 1.0 - s;
            }
        }
        none_kept *= 1.0 - s;
    }
    if let Some((tg, cg)) = ground {
        let s = sel(drop, em, sc.ground_db, beam * cg);
        if obs.is_finite() {
            let sd = em.sigma_e[2];
            dens += (-sc.kappa * tg).exp() * none_kept * s * t_pdf((obs - tg) / sd, em.nu_e) / sd;
        }
        none_kept *= 1.0 - s;
    }
    if !obs.is_finite() {
        return ((-sc.kappa * t_end).exp() * none_kept).max(1e-300).ln();
    }
    if obs < t_end {
        dens += sc.kappa * (-sc.kappa * obs).exp() * none_before_obs;
    }
    ((1.0 - em.p_outlier) * dens + em.p_outlier / sc.t_max).ln()
}

/// One element's data in one scan, as a function of three shifts there: δ, along the beam, of the depths of the
/// cells it fronts at fill 0.5 or more (its A cells), and its silhouette's edges moved outward, `e₊` on its +x side
/// and `e₋` on its −x side, which move its A cells' fill and its term C. Each shift is a row over the latents.
struct ElemData<'a> {
    elem: usize,
    /// (obs, layers, ground, weight, the cell's log-likelihood against the empty scene)
    cells: Vec<(f64, Vec<(Layer, u8)>, Option<(f64, f64)>, f64, f64)>,
    rendered: &'a [(usize, Layer)],
    class: u8,
    support: u32,
    n_cells: u32,
    across: f64,
    n_fp: f64,
    /// The residual design effect on its terms, `1 / (1 + (n − 1) ρ)`.
    temper: f64,
    /// The section's angles (about the axis, in the element's frame) the scan sees at its front and its +x and −x
    /// edges.
    angles: [f64; 3],
    /// Its row over the scan's field nodes (local): the along shift, scaled by its class's along spread (and the
    /// other-epoch part).
    along: Vec<(usize, f64)>,
}

impl ElemData<'_> {
    /// Its tempered A and C at shifts x = (δ, e₊, e₋); `e` is the element as rendered.
    fn ll(&self, x: [f64; 3], v: &View, e: &Element, drop: f64, zt: f64, em: &ErrorModel, sc: &Scene) -> f64 {
        let stick = matches!(e.form, Form::Stick { .. });
        let a: f64 = self.cells.iter().map(|(obs, layers, ground, wt, base)| {
            let mut ls = [layers[0]; K];
            ls[..layers.len()].copy_from_slice(layers);
            if stick && (x[1] != 0.0 || x[2] != 0.0) {
                let l = &mut ls[0].0;
                let sd = (0.0625 * em.footprint(l.t).powi(2) + e.misfit * e.misfit).sqrt();
                l.f = phi((e.r + x[1] - l.x) / sd) - phi((-e.r - x[2] - l.x) / sd);
            }
            wt * (cell_ll(*obs, &ls[..layers.len()], *ground, x[0], drop, em, sc) - base)
        }).sum();
        self.temper * (a + edge_ll(v, self.rendered, e, self.class, zt, [x[1], x[2]], em))
    }
}

/// A scan's offset-field nodes: trilinear hats on a grid of spacing ℓ, normalised so the field's variance is the
/// same everywhere (`t(x) = Σ w_j z_j / √Σ w_j²`); each node carries the along-beam offset.
struct Field {
    nodes: HashMap<[i64; 3], usize>,
}

impl Field {
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

    fn node(&mut self, k: [i64; 3]) -> usize {
        let n = self.nodes.len();
        *self.nodes.entry(k).or_insert(n)
    }
}

/// Newton steps at most per energy.
const MAX_NEWTON: usize = 40;

/// A frame about a stick's axis: two unit vectors across it.
fn axis_frame(u: &Point) -> (Point, Point) {
    let p1 = normalize(&cross(u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
    (p1, cross(u, &p1))
}

/// One scan's element data (elements other than `skip`) and its field's nodes.
#[allow(clippy::too_many_arguments)]
fn scan_data<'a>(v: &View, rendered: &'a [Vec<(usize, Layer)>], elems: &[Element], skip: Option<u32>, ref_epoch: u32, em: &ErrorModel, sc: &Scene)
                 -> (Vec<ElemData<'a>>, Field, f64) {
    let empty: &'a [(usize, Layer)] = &[];
    let per: Vec<&'a [(usize, Layer)]> = rendered.iter().enumerate().map(|(i, r)| if Some(i as u32) == skip { empty } else { r.as_slice() }).collect();
    let owned: Vec<Vec<(usize, Layer)>> = per.iter().map(|r| r.to_vec()).collect();
    let buf = kbuffer(&owned, 0.5);
    let cls = classes(v, &buf, elems.len(), em);
    let drop = drop_of(em, v);
    let other = v.epoch != ref_epoch;
    let mut field = Field { nodes: HashMap::new() };
    let mut data: Vec<ElemData> = Vec::new();
    let mut by_elem: HashMap<u32, usize> = HashMap::new();
    for (i, e) in elems.iter().enumerate() {
        if per[i].is_empty() {
            continue;
        }
        let k = cls[i];
        let mid = e.mid();
        let c = sub(&mid, &v.origin);
        let angles = match e.form {
            Form::Stick { a, b } => {
                let u = normalize(&sub(&b, &a));
                let (p1, p2) = axis_frame(&u);
                let ang = |w: &Point| dot(w, &p2).atan2(dot(w, &p1));
                let ps = ang(&normalize(&cross(&u, &c)));
                [ang(&c.map(|x| -x)), ps, ps + std::f64::consts::PI]
            }
            Form::Patch { .. } => [0.0; 3],
        };
        let s_along = (em.field_along[k as usize].powi(2) + if other { em.field_epoch[k as usize].powi(2) } else { 0.0 }).sqrt();
        let along = if em.field {
            Field::hats(&mid, em.field_spacing).into_iter().map(|(key, w)| (field.node(key), w * s_along)).collect()
        } else {
            Vec::new()
        };
        by_elem.insert(i as u32, data.len());
        data.push(ElemData { elem: i, cells: Vec::new(), rendered: per[i], class: k, support: 0, n_cells: 0, across: 2.0 * e.r / em.footprint(norm(&c)),
                             n_fp: 0.0, temper: 1.0, angles, along });
    }
    for (&cell, ls) in &buf {
        let front = ls[0];
        let Some(&d) = by_elem.get(&front.elem) else { continue };
        let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[l.elem as usize])).collect();
        let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
        let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
        let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
        let wt = 1.0 / em.multiplicity(if obs.is_finite() { obs } else { front.t });
        let el = &mut data[d];
        el.n_cells += 1;
        el.n_fp += front.f / em.multiplicity(front.t);
        let sd = (em.sigma_e[el.class as usize].powi(2) + front.spread.powi(2)).sqrt();
        el.support += (cell_code(obs, front.t, sd) == code::ON) as u32;
        el.cells.push((obs, layers, ground, wt, cell_ll(obs, &[], ground, 0.0, drop, em, sc)));
    }
    for el in data.iter_mut() {
        el.temper = 1.0 / (1.0 + (el.n_fp - 1.0).max(0.0) * em.rho);
    }
    (data, field, drop)
}

/// One element's shape latent: the section's radius less its circle at the angles its scans see (front, +x edge,
/// −x edge, per scan), a multivariate t (ν `shape_nu`) with scale `misfit² C`. C is the periodic squared-exponential
/// correlation in angle (length `shape_angle`) with its constant and first harmonics taken out: those are the
/// radius and the axis's place, which the element itself carries, so the latent can't stand in for them.
struct Block {
    start: usize,
    n: usize,
    kinv: nalgebra::DMatrix<f64>,
    /// `ln Γ(ν/2) − ln Γ((ν+n)/2) + n/2 ln(νπ) + ½ ln|K|`: its −log density at 0.
    norm: f64,
}

impl Block {
    fn new(start: usize, angles: &[f64], misfit: f64, em: &ErrorModel) -> Block {
        let n = angles.len();
        // exp(−2 sin²(Δ/2)/θ²) = e^{−x} Σ_n ε_n I_n(x) cos nΔ with x = θ⁻², less its n = 0 and 1 terms, at unit variance
        let x = em.shape_angle.powi(-2);
        let bessel = |m: i32| (0..40).map(|k| (0.5 * x).powi(2 * k + m) / (libm::tgamma(k as f64 + 1.0) * libm::tgamma((k + m) as f64 + 1.0))).sum::<f64>();
        let (c0, c1) = ((-x).exp() * bessel(0), 2.0 * (-x).exp() * bessel(1));
        let corr = |d: f64| ((-2.0 * (0.5 * d).sin().powi(2) * x).exp() - c0 - c1 * d.cos()) / (1.0 - c0 - c1);
        let k = nalgebra::DMatrix::from_fn(n, n, |i, j| misfit * misfit * (corr(angles[i] - angles[j]) + if i == j { 1e-3 } else { 0.0 }));
        let ch = k.cholesky().expect("a shape covariance is positive definite");
        let logdet = 2.0 * ch.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let nu = em.shape_nu;
        let norm = libm::lgamma(0.5 * nu) - libm::lgamma(0.5 * (nu + n as f64)) + 0.5 * n as f64 * (nu * std::f64::consts::PI).ln() + 0.5 * logdet;
        let kinv = ch.inverse();
        Block { start, n, kinv: (&kinv + kinv.transpose()) * 0.5, norm }
    }

    /// Its −log prior at z, the gradient, and the prior's (Gauss–Newton) precision.
    fn prior(&self, z: &[f64], nu: f64) -> (f64, nalgebra::DVector<f64>, nalgebra::DMatrix<f64>) {
        let g = nalgebra::DVector::from_column_slice(&z[self.start..self.start + self.n]);
        let kg = &self.kinv * &g;
        let q = g.dot(&kg);
        let s = (nu + self.n as f64) / (nu + q);
        (self.norm + 0.5 * (nu + self.n as f64) * (1.0 + q / nu).ln(), kg * s, &self.kinv * s)
    }
}

/// The energy of one configuration (elements other than `skip`) against every scan: the Laplace marginal of the
/// scans' data over all the latents, `U = −Σ ll(ẑ) − log p(ẑ) + ½ log|Λ| − (n/2) log 2π`, Λ the posterior precision
/// at the mode ẑ. The latents: each scan's offset-field nodes, and each stick's shape latent ([`Block`]), which
/// couples the scans that see it. Newton steps solve the field after eliminating each stick's block (a Schur
/// complement), so the cost grows with the field's nodes, not the sticks.
#[allow(clippy::too_many_arguments)]
fn joint(views: &[&View], rendered: &[Vec<Vec<(usize, Layer)>>], elems: &[Element], skip: Option<u32>, ref_epoch: u32, em: &ErrorModel, sc: &Scene)
         -> (f64, Vec<Row>) {
    use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
    let zt = phi_inv(1.0 - em.edge_fill);
    let scans: Vec<(Vec<ElemData>, Field, f64)> = views.par_iter().zip(rendered).map(|(v, r)| scan_data(v, r, elems, skip, ref_epoch, em, sc)).collect();
    // the field's global layout, then each stick's block after it
    let mut off = vec![0usize; scans.len() + 1];
    for (s, (_, f, _)) in scans.iter().enumerate() {
        off[s + 1] = off[s] + f.nodes.len();
    }
    let nf = off[scans.len()];
    // the field's prior precision: unit (each element's spread is in its row)
    let q = vec![1.0; nf];
    // terms: (scan, data index, block, its three rows (global index, weight))
    let mut slots: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (s, (data, _, _)) in scans.iter().enumerate() {
        for (d, el) in data.iter().enumerate() {
            if em.shape && matches!(elems[el.elem].form, Form::Stick { .. }) && elems[el.elem].misfit > 0.0 {
                slots.entry(el.elem).or_default().push((s, d));
            }
        }
    }
    let mut keys: Vec<usize> = slots.keys().copied().collect();
    keys.sort();
    let mut blocks = Vec::new();
    let mut block_of: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut n = nf;
    for e in keys {
        let angles: Vec<f64> = slots[&e].iter().flat_map(|&(s, d)| scans[s].0[d].angles).collect();
        for (k, &(s, d)) in slots[&e].iter().enumerate() {
            block_of.insert((s, d), (blocks.len(), n + 3 * k));
        }
        blocks.push(Block::new(n, &angles, elems[e].misfit, em));
        n += angles.len();
    }
    type Rows = [Vec<(usize, f64)>; 3];
    let mut terms: Vec<(usize, usize, Option<usize>, Rows)> = Vec::new();
    for (s, (data, _, _)) in scans.iter().enumerate() {
        for (d, el) in data.iter().enumerate() {
            let mut rows: Rows = [el.along.iter().map(|&(j, w)| (off[s] + j, w)).collect(), Vec::new(), Vec::new()];
            let b = block_of.get(&(s, d)).map(|&(b, at)| {
                rows[0].push((at, -1.0));
                rows[1].push((at + 1, 1.0));
                rows[2].push((at + 2, 1.0));
                b
            });
            terms.push((s, d, b, rows));
        }
    }
    let x_of = |rows: &Rows, z: &[f64]| rows.clone().map(|r| r.iter().map(|&(i, w)| w * z[i]).sum::<f64>());
    let ll = |t: &(usize, usize, Option<usize>, Rows), x: [f64; 3]| {
        let (data, _, drop) = &scans[t.0];
        let el = &data[t.1];
        el.ll(x, views[t.0], &elems[el.elem], *drop, zt, em, sc)
    };
    let nu = em.shape_nu;
    let prior = |z: &[f64]| -> f64 {
        0.5 * z[..nf].iter().zip(&q).map(|(x, p)| p * x * x).sum::<f64>() + blocks.iter().map(|b| b.prior(z, nu).0).sum::<f64>()
    };
    let objective = |z: &[f64]| -> f64 { prior(z) - terms.par_iter().map(|t| ll(t, x_of(&t.3, z))).sum::<f64>() };
    // Newton; each term's −ll as a quadratic in its three shifts by central differences, made positive
    // semi-definite; the blocks eliminated into the field's system
    let h = [3e-4, 5e-4, 5e-4];
    let mut z = vec![0.0; n];
    let mut sinv = DMatrix::<f64>::zeros(nf, nf);
    let mut logdet = 0.0;
    let mut last = false;
    for it in 0..=MAX_NEWTON {
        let quads: Vec<(Vector3<f64>, Matrix3<f64>)> = terms.par_iter().map(|t| {
            let x0 = x_of(&t.3, &z);
            let f = |d: [f64; 3]| -ll(t, [x0[0] + d[0], x0[1] + d[1], x0[2] + d[2]]);
            let used: Vec<usize> = (0..3).filter(|&i| !t.3[i].is_empty()).collect();
            let f0 = f([0.0; 3]);
            let mut g = Vector3::zeros();
            let mut hm = Matrix3::zeros();
            let e = |i: usize, s: f64| { let mut d = [0.0; 3]; d[i] = s * h[i]; d };
            let fp: Vec<f64> = (0..3).map(|i| if used.contains(&i) { f(e(i, 1.0)) } else { f0 }).collect();
            let fm: Vec<f64> = (0..3).map(|i| if used.contains(&i) { f(e(i, -1.0)) } else { f0 }).collect();
            for &i in &used {
                g[i] = (fp[i] - fm[i]) / (2.0 * h[i]);
                hm[(i, i)] = (fp[i] - 2.0 * f0 + fm[i]) / (h[i] * h[i]);
                for &j in used.iter().filter(|&&j| j > i) {
                    let mut dp = e(i, 1.0);
                    dp[j] = h[j];
                    let mut dm = e(i, -1.0);
                    dm[j] = -h[j];
                    let hij = (f(dp) - fp[i] - fp[j] + 2.0 * f0 - fm[i] - fm[j] + f(dm)) / (2.0 * h[i] * h[j]);
                    hm[(i, j)] = hij;
                    hm[(j, i)] = hij;
                }
            }
            if !(g.iter().all(|x| x.is_finite()) && hm.iter().all(|x| x.is_finite())) {
                return (Vector3::zeros(), Matrix3::zeros());
            }
            let eig = hm.symmetric_eigen();
            let hm = eig.eigenvectors * Matrix3::from_diagonal(&eig.eigenvalues.map(|x| x.max(0.0))) * eig.eigenvectors.transpose();
            (g, (hm + hm.transpose()) * 0.5)
        }).collect();
        // assemble: the field's block (dense), each stick's block, and their coupling on the field indices it touches
        let mut grad = DVector::<f64>::zeros(n);
        let mut hff = DMatrix::<f64>::from_diagonal(&DVector::from_vec(q.clone()));
        for i in 0..nf {
            grad[i] = q[i] * z[i];
        }
        let mut hbb: Vec<DMatrix<f64>> = Vec::with_capacity(blocks.len());
        let mut fidx: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
        for blk in &blocks {
            let (_, gb, pb) = blk.prior(&z, nu);
            grad.rows_mut(blk.start, blk.n).copy_from(&gb);
            hbb.push(pb);
        }
        for (_, _, b, rows) in &terms {
            if let Some(b) = b {
                for r in rows {
                    for &(i, _) in r.iter().filter(|(i, _)| *i < nf) {
                        if !fidx[*b].contains(&i) {
                            fidx[*b].push(i);
                        }
                    }
                }
            }
        }
        let mut hfb: Vec<DMatrix<f64>> = blocks.iter().zip(&fidx).map(|(blk, fi)| DMatrix::zeros(fi.len(), blk.n)).collect();
        for ((_, _, b, rows), (g, hm)) in terms.iter().zip(&quads) {
            for a in 0..3 {
                for &(i, wi) in &rows[a] {
                    grad[i] += g[a] * wi;
                    for c in 0..3 {
                        let hac = hm[(a, c)];
                        if hac == 0.0 {
                            continue;
                        }
                        for &(j, wj) in &rows[c] {
                            let v = hac * wi * wj;
                            match (i < nf, j < nf) {
                                (true, true) => hff[(i, j)] += v,
                                (false, false) => {
                                    let blk = &blocks[b.unwrap()];
                                    hbb[b.unwrap()][(i - blk.start, j - blk.start)] += v;
                                }
                                (true, false) => {
                                    let bb = b.unwrap();
                                    let li = fidx[bb].iter().position(|&x| x == i).unwrap();
                                    hfb[bb][(li, j - blocks[bb].start)] += v;
                                }
                                (false, true) => {}
                            }
                        }
                    }
                }
            }
        }
        // eliminate the blocks: S = H_ff − Σ H_fb H_bb⁻¹ H_bf, r = g_f − Σ H_fb H_bb⁻¹ g_b
        let mut s_mat = hff;
        let mut r_f = grad.rows(0, nf).into_owned();
        let mut ld_b = 0.0;
        let mut hbb_inv = Vec::with_capacity(blocks.len());
        for (b, blk) in blocks.iter().enumerate() {
            let hb = (&hbb[b] + hbb[b].transpose()) * 0.5;
            let ch = hb.clone().cholesky().unwrap_or_else(|| panic!("a block's precision is positive definite: {hb} at {:?}", &z[blk.start..blk.start + blk.n]));
            ld_b += 2.0 * ch.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
            let inv = ch.inverse();
            let w = &hfb[b] * &inv;
            let sub_s = &w * hfb[b].transpose();
            let sub_r = &w * grad.rows(blk.start, blk.n);
            for (a, &i) in fidx[b].iter().enumerate() {
                r_f[i] -= sub_r[a];
                for (c, &j) in fidx[b].iter().enumerate() {
                    s_mat[(i, j)] -= sub_s[(a, c)];
                }
            }
            hbb_inv.push(inv);
        }
        let ch = s_mat.clone().cholesky();
        let ld_s = ch.as_ref().map_or(0.0, |c| 2.0 * c.l().diagonal().iter().map(|x| x.ln()).sum::<f64>());
        logdet = ld_b + ld_s;
        sinv = ch.as_ref().map_or_else(|| DMatrix::zeros(nf, nf), |c| c.inverse());
        if last || it == MAX_NEWTON || n == 0 {
            break;
        }
        let df = &sinv * &r_f;
        let mut step = DVector::<f64>::zeros(n);
        step.rows_mut(0, nf).copy_from(&df);
        for (b, blk) in blocks.iter().enumerate() {
            let mut rb = grad.rows(blk.start, blk.n).into_owned();
            for (a, &i) in fidx[b].iter().enumerate() {
                rb -= hfb[b].row(a).transpose() * df[i];
            }
            step.rows_mut(blk.start, blk.n).copy_from(&(&hbb_inv[b] * rb));
        }
        // a step at most 3 prior sds in the field and 2 cm in a shape, then halved until the objective falls; none
        // that does ends the search
        let over = (0..n).map(|i| step[i].abs() * if i < nf { q[i].sqrt() / 3.0 } else { 50.0 }).fold(1.0, f64::max);
        step /= over;
        let f_now = objective(&z);
        let mut tstep = 1.0;
        let mut fall = 0.0;
        while tstep >= 1e-3 {
            let zn: Vec<f64> = z.iter().zip(step.iter()).map(|(a, b)| a - tstep * b).collect();
            let f_new = objective(&zn);
            if f_new <= f_now {
                z = zn;
                fall = f_now - f_new;
                break;
            }
            tstep *= 0.5;
        }
        // converged (under a millinat), or stuck: one more pass refreshes Λ at the mode
        last = tstep < 1e-3 || fall < 1e-3;
    }
    let lls: Vec<f64> = terms.par_iter().map(|t| ll(t, x_of(&t.3, &z))).collect();
    let field_norm = -0.5 * q.iter().map(|p| p.ln()).sum::<f64>() + 0.5 * nf as f64 * (2.0 * std::f64::consts::PI).ln();
    let shared = prior(&z) + field_norm + 0.5 * logdet - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    let u = shared - lls.iter().sum::<f64>();
    let rows = terms.iter().zip(&lls).map(|((s, d, b, rows), l)| {
        let el = &scans[*s].0[*d];
        let field_part = |r: &[(usize, f64)]| r.iter().filter(|(i, _)| *i < nf).map(|&(i, w)| w * z[i]).sum::<f64>();
        let ra: Vec<(usize, f64)> = rows[0].iter().copied().filter(|(i, _)| *i < nf).collect();
        let var: f64 = ra.iter().map(|&(i, wi)| ra.iter().map(|&(j, wj)| wi * wj * sinv[(i, j)]).sum::<f64>()).sum();
        Row {
            elem: el.elem as u32, scan: *s as u32, cells: el.n_cells, support: el.support, across: el.across, surface: el.class,
            energy: -l + shared / terms.len() as f64, shift_along: field_part(&rows[0]), sd_along: var.max(0.0).sqrt(),
            shape: b.map_or(0.0, |_| z[block_of[&(*s, *d)].1]),
        }
    }).collect();
    (u, rows)
}

/// The energy of `elems` against `views` (−log likelihood ratio against the empty scene), and a row per (element,
/// scan). It is the Laplace marginal over the scans' offset fields and the sticks' shape latents ([`joint`]). With
/// `only = Some(i)`, the energy change of adding element i to the others, `E(all) − E(all but i)`; rows are then
/// empty. The reference epoch (whose scans take the same-day spread) is the first view's.
pub fn energy(views: &[&View], elems: &[Element], em: &ErrorModel, sc: &Scene, only: Option<usize>) -> (f64, Vec<Row>) {
    let ref_epoch = views.first().map_or(0, |v| v.epoch);
    let rendered: Vec<Vec<Vec<(usize, Layer)>>> = views.par_iter().map(|v| render_all(v, elems, em)).collect();
    let (u, rows) = joint(views, &rendered, elems, None, ref_epoch, em, sc);
    match only {
        Some(i) => (u - joint(views, &rendered, elems, Some(i as u32), ref_epoch, em, sc).0, Vec::new()),
        None => (u, rows),
    }
}

/// Term C for one stick in one view (0 for a patch): the log likelihood ratio of its located silhouette edges
/// against an edge placed evenly across the rendered band.
///
/// Its cells are cut into strips along the axis, a cell wide. In each strip with echoes on the stick, each side's
/// outermost such echo and the next cell out locate the edge:
/// - **sharp** (the next cell's echo lies behind): the edge lies between the two;
/// - **unknown** (the next cell is empty, its echo deleted): censored, the edge lies beyond the last echo (Tobit);
/// - **occluded** (the next cell's echo lies in front): no information.
///
/// The stick's edge lies `R + z_τ sd + out` from the axis, where sd is the fill's (footprint and misfit), z_τ puts
/// the fill at `edge_fill` (τ), and `out` moves the +x and −x edges outward (the shape latent). Its spread is the misfit and a third of the fill sd. Each strip counts `cell / footprint`
/// (neighbouring strips share footprints).
fn edge_ll(v: &View, cells: &[(usize, Layer)], e: &Element, k: u8, zt: f64, out: [f64; 2], em: &ErrorModel) -> f64 {
    if matches!(e.form, Form::Patch { .. }) {
        return 0.0;
    }
    let mut strips: HashMap<i64, Vec<(f64, u8, f64)>> = HashMap::new();
    for &(cell, ly) in cells {
        let sd = (em.sigma_e[k as usize].powi(2) + ly.spread.powi(2)).sqrt();
        let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
        let st = cell_code(obs, ly.t + em.depth_bias(k, ly.t), sd);
        strips.entry((ly.s / (em.step * ly.t)).floor() as i64).or_default().push((ly.x, st, ly.t));
    }
    let mut ll = 0.0;
    for cs in strips.values() {
        let Some(t) = cs.iter().find(|c| c.1 == code::ON).map(|c| c.2) else { continue };
        let fp = em.footprint(t);
        let sdf = (0.0625 * fp * fp + e.misfit * e.misfit).sqrt();
        let se = (e.misfit * e.misfit + (sdf / 3.0).powi(2)).sqrt();
        let w = (em.step * t / fp).min(1.0);
        for side in [1.0, -1.0] {
            let xe = e.r + zt * sdf + if side > 0.0 { out[0] } else { out[1] };
            let x_in = cs.iter().filter(|c| c.1 == code::ON).map(|c| side * c.0).fold(f64::NEG_INFINITY, f64::max);
            let far = cs.iter().map(|c| side * c.0).fold(0.0, f64::max);
            let Some(&(xo, st, _)) = cs.iter().filter(|c| side * c.0 > x_in).min_by(|a, b| (side * a.0).total_cmp(&(side * b.0))) else { continue };
            let x_out = side * xo;
            if far <= 0.0 || x_in < 0.0 {
                continue;
            }
            let (p, bg) = match st {
                code::BEHIND => (phi((xe - x_in) / se) - phi((xe - x_out) / se), (x_out - x_in) / far),
                code::EMPTY => (phi((xe - x_in) / se), (far - x_in) / far),
                _ => continue,
            };
            ll += w * (p.max(1e-12).ln() - bg.max(1e-6).ln());
        }
    }
    ll
}

/// Φ⁻¹(p), by bisection.
fn phi_inv(p: f64) -> f64 {
    let (mut lo, mut hi) = (-8.0, 8.0);
    for _ in 0..60 {
        let m = 0.5 * (lo + hi);
        if phi(m) < p { lo = m } else { hi = m }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;
    use crate::shots::view::ViewParams;

    /// A scan of a vertical pole (R = 3 cm, 4 m away, centred on the y axis) in front of a wall at 8 m. The cells
    /// whose ray passes the axis 2.9-3.2 cm out on the +x side (its edge and the next pixel out) are left alone
    /// (`edge` 0), emptied (1: deleted), or given an echo at 3 m (2: occluded).
    fn pole_view(edge: u8) -> View {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let (r_pole, y_pole) = (0.03, 4.0);
        let (mut t, mut d, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..200usize {
            for k in 0..400usize {
                let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
                let w = dir(zen, az);
                // ray against the infinite cylinder x² + (y − y_pole)² = R²
                let (a, b, c) = (w[0] * w[0] + w[1] * w[1], -2.0 * w[1] * y_pole, y_pole * y_pole - r_pole * r_pole);
                let disc = b * b - 4.0 * a * c;
                let mut range = if disc > 0.0 { (-b - disc.sqrt()) / (2.0 * a) } else { 8.0 / w[1] };
                let miss = y_pole * w[0] / (w[0] * w[0] + w[1] * w[1]).sqrt();
                if (0.029..0.032).contains(&miss) {
                    range = [range, f64::NAN, 3.0][edge as usize];
                }
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        View::new(&t, &d, &r, id, [0.0; 3], 0, &ViewParams::default())
    }

    fn stick(x: f64, y: f64, r: f64) -> Element {
        Element { form: Form::Stick { a: [x, y, -0.2], b: [x, y, 0.2] }, r, opacity: 1.0, misfit: 0.002 }
    }

    /// The true pole has negative energy; a stick of the wrong radius, shifted, or porous has more; a stick where
    /// the pulses went through to the wall costs energy; `only` agrees with the difference of two energies, and
    /// rows sum to the total.
    #[test]
    fn the_true_pole_wins() {
        let v = pole_view(0);
        let (em, sc) = (ErrorModel::default(), Scene::default());
        let e = |s: Element| energy(&[&v], &[s], &em, &sc, None).0;
        let truth = e(stick(0.0, 4.0, 0.03));
        assert!(truth < -3.0, "{truth}");
        for wrong in [stick(0.0, 4.0, 0.021), stick(0.0, 4.0, 0.015), stick(0.0, 4.0, 0.06), stick(0.03, 4.0, 0.03), stick(0.0, 3.98, 0.03),
                      Element { opacity: 0.3, ..stick(0.0, 4.0, 0.03) }] {
            assert!(e(wrong) > truth + 1.0, "{} vs {truth}", e(wrong));
        }
        assert!(e(stick(0.2, 6.0, 0.03)) > 1.0);
        let pair = [stick(0.0, 4.0, 0.03), stick(0.2, 6.0, 0.03)];
        let both = energy(&[&v], &pair, &em, &sc, None);
        let delta = energy(&[&v], &pair, &em, &sc, Some(1)).0;
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let sum: f64 = both.1.iter().map(|r| r.energy).sum();
        assert!((sum - both.0).abs() < 1e-6);
    }

    /// The rendered depth is the footprint's mean on the section, which the misfit (a spread about the form, not a
    /// bias) doesn't move: on the pole's front and halfway out it sits within 0.3 mm of the echoes, whatever the
    /// misfit. (The misfit used to widen the depth band, which a section's edge clips, so the front was rendered
    /// behind the echoes; slice 7b.)
    #[test]
    fn misfit_does_not_move_the_rendered_front() {
        let v = pole_view(0);
        let em = ErrorModel::default();
        for misfit in [0.0, 0.002, 0.008] {
            for ly in render(&v, &Element { misfit, ..stick(0.0, 4.0, 0.03) }, 0, &em).iter().map(|(c, l)| (v.range[*c] as f64, l)) {
                let (obs, l) = ly;
                if l.x.abs() < 0.015 && obs.is_finite() && obs < 5.0 {
                    assert!((obs - l.t).abs() < 3e-4, "misfit {misfit}: x {} obs {obs} t {}", l.x, l.t);
                }
            }
        }
    }

    /// With the shape latent on (and untempered, so the data speak plainly), the true pole still beats a stick 40 %
    /// too thick, a third too thin and one shifted by R, and the energy is smooth in the radius (one mode: the
    /// latent's edges once made it jump by several nats).
    #[test]
    fn the_shape_latent_keeps_the_true_pole_and_a_smooth_energy() {
        let v = pole_view(0);
        let (em, sc) = (ErrorModel { rho: 0.0, shape: true, ..ErrorModel::default() }, Scene::default());
        let e = |s: Element| energy(&[&v], &[s], &em, &sc, None).0;
        let truth = e(stick(0.0, 4.0, 0.03));
        for wrong in [stick(0.0, 4.0, 0.042), stick(0.0, 4.0, 0.021), stick(0.03, 4.0, 0.03)] {
            assert!(e(wrong) > truth + 10.0, "{} vs {truth}", e(wrong));
        }
        for eps in [1e-4, 5e-4, 1e-3] {
            let d = e(stick(0.0, 4.0, 0.03 * (1.0 + eps))) - truth;
            assert!(d.abs() < 0.5, "{eps}: {d}");
        }
    }

    /// Term C censors an edge whose pixels were deleted rather than reading it as absent, and skips one seen
    /// against something nearer: with the +x edge deleted, the true radius still scores better in C than a stick a
    /// third thinner; with it occluded, C keeps only the other side's evidence (about half).
    #[test]
    fn deleted_edges_are_censored_and_occluded_ones_skipped() {
        let em = ErrorModel::default();
        let zt = phi_inv(1.0 - em.edge_fill);
        let c_of = |v: &View, s: &Element| edge_ll(v, &render(v, s, 0, &em), s, 0, zt, [0.0; 2], &em);
        let (full, deleted, occluded) = (pole_view(0), pole_view(1), pole_view(2));
        let truth = stick(0.0, 4.0, 0.03);
        let c_full = c_of(&full, &truth);
        assert!(c_full > 0.0, "{c_full}");
        assert!(c_of(&deleted, &truth) > c_of(&deleted, &stick(0.0, 4.0, 0.02)));
        let ratio = c_of(&occluded, &truth) / c_full;
        assert!((0.3..0.7).contains(&ratio), "{ratio}");
    }

    /// The Laplace marginal over the field agrees with Gauss–Hermite quadrature on a single-element query (the
    /// referee), a range offset in the data is recovered by the field's along part, and the field's hats are
    /// normalised.
    #[test]
    fn the_field_integral_matches_quadrature_and_finds_an_offset() {
        for x in [[0.3, 4.1, -0.7], [1.0, 2.0, 3.0]] {
            let h = Field::hats(&x, 2.0);
            assert!((h.iter().map(|(_, w)| w * w).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let v = pole_view(0);
        // untempered (ρ 0), so one element's data speak plainly and the quadrature needs no design effect; the field
        // alone (no shape latent)
        let (em, sc) = (ErrorModel { rho: 0.0, shape: false, ..ErrorModel::default() }, Scene::default());
        let el = stick(0.0, 4.0, 0.03);
        // shift the pole's echoes 3 mm deeper: the field's along shift should find about that
        let shifted = {
            let mut w = pole_view(0);
            for (k, r) in w.range.iter_mut().enumerate() {
                if r.is_finite() && *r < 4.5 && w.state[k] == state::ECHO {
                    *r += 0.003;
                }
            }
            w
        };
        let (_, rows) = energy(&[&shifted], &[el], &em, &sc, None);
        assert!((rows[0].shift_along - 0.003).abs() < 0.0015, "{}", rows[0].shift_along);
        // the referee: a single element's data integrated by quadrature over the field's prior
        let rendered = render_all(&v, &[el], &em);
        let (u_lap, out) = energy(&[&v], &[el], &em, &sc, None);
        assert_eq!(out.len(), 1);
        let em_off = ErrorModel { field: false, ..em };
        let u_off = energy(&[&v], &[el], &em_off, &sc, None).0;
        // the element sits on one node set: its along shift has prior sd s_along (√Σŵ² = 1)
        let (gh_x, gh_w) = ([-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
                            [0.019953242059045913, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116, 0.019953242059045913]);
        let buf = kbuffer(&rendered, 0.5);
        let cls = classes(&v, &buf, 1, &em);
        let s_along = em.field_along[cls[0] as usize];
        let ed = |delta: f64| -> f64 {
            buf.iter().map(|(&cell, ls)| {
                let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[0])).collect();
                let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
                let wt = 1.0 / em.multiplicity(if obs.is_finite() { obs } else { ls[0].t });
                wt * (cell_ll(obs, &layers, None, delta, drop_of(&em, &v), &em, &sc) - cell_ll(obs, &[], None, 0.0, drop_of(&em, &v), &em, &sc))
            }).sum()
        };
        let c0 = edge_ll(&v, &rendered[0], &el, cls[0], phi_inv(1.0 - em.edge_fill), [0.0; 2], &em);
        let m = gh_x.iter().map(|&x| ed(std::f64::consts::SQRT_2 * s_along * x)).fold(f64::NEG_INFINITY, f64::max);
        let quad = -(m + (gh_x.iter().zip(&gh_w).map(|(&x, &w)| w * (ed(std::f64::consts::SQRT_2 * s_along * x) - m).exp()).sum::<f64>()
            / std::f64::consts::PI.sqrt()).ln()) - c0;
        assert!((u_lap - quad).abs() < 0.1 * quad.abs().max(1.0), "laplace {u_lap} quadrature {quad} off {u_off}");
    }

    /// A patch renders as a disc: full fill at its centre, none beyond its rim; a patch of the pole's size where the
    /// pole is beats one off to the side; `predict` codes the pole's front cells on and the wall beside it behind.
    #[test]
    fn patches_render_as_discs_and_predict_codes_cells() {
        let v = pole_view(0);
        let (em, sc) = (ErrorModel::default(), Scene::default());
        let patch = |x: f64| Element { form: Form::Patch { centre: [x, 4.0, 0.0] }, r: 0.03, opacity: 1.0, misfit: 0.0 };
        let layers = render(&v, &patch(0.0), 0, &em);
        let f_max = layers.iter().map(|(_, l)| l.f).fold(0.0, f64::max);
        assert!(f_max > 0.99, "{f_max}");
        assert!(layers.iter().all(|(_, l)| l.x < 0.03 + 0.012 || l.f < 0.01));
        let e = |s: Element| energy(&[&v], &[s], &em, &sc, None).0;
        assert!(e(patch(0.0)) < e(patch(0.1)));
        let p = predict(&v, &[stick(0.0, 4.0, 0.03)], &em, &sc, 0.01);
        let on = p.iter().filter(|q| q.fill > 0.9).all(|q| q.code == code::ON);
        let behind = p.iter().filter(|q| q.fill < 0.05 && q.x.abs() > 0.035).any(|q| q.code == code::BEHIND);
        assert!(on && behind);
        assert!(p.iter().all(|q| q.weight > 0.0 && q.weight <= 1.0));
    }
}
