// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of elements scored against scans' range images by the ray likelihood (3dtrees research 130): its cost,
//! `−log` of its likelihood ratio against the empty scene, the cost change of a move, and what an element predicts
//! in an image.
//!
//! - **Elements** ([`Element`]): a stick (an axis, its section's mean radius a₀) or a leaf patch (a ball with its
//!   extent, a disc in every view), each with an opacity and a misfit. Opacity is the chance a pulse the section
//!   fills meets solid there, its gaps wider than the footprint (synthesis §1's "extent, opacity"). Misfit is the
//!   sd of the section's and axis's departure from the form, which blurs the fill and spreads the depth.
//! - **The k-buffer:** each element renders into each view's cells. A cell keeps up to [`K`] layers, nearest
//!   first, each with a depth, a depth spread, a fill and a cosine. Fill is the beam (a uniform disc of the
//!   footprint's diameter, as AM's raw edges measure it) convolved with the section, at three section radii over
//!   the misfit; an echo's chance is the detection's mean over them. Along a stick's axis it is cut square, so a chain of
//!   sticks joins without seams.
//! - **The ray likelihood, one cell:** its first echo, or none, against a diffuse echo hazard κ along the ray (the
//!   background), the layers in depth order, then a ground plane.
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is recorded with
//!     `op_i det_i`, where `det_i` is the chance it clears the detection floor ([`ErrorModel::p_detect`]):
//!     `Φ((ρ_dB + 10 log10(e_i cos ι_i) − floor(t_i)) / σ)`.
//!   - Adjacent returns (layers and the ground) resolve into two echoes with the instrument's
//!     [`ErrorModel::p_resolve`] of their separation and energy ratio; else they merge into one, recorded if any
//!     member is, at their energy-weighted depth (energy: op · e · cos · reflectance), its spread their spread about
//!     it. A cell's likelihood is the mixture over these merge patterns.
//!   - An echo at r has density `Σ_i P_i op_i det_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i)` over the returns, plus
//!     `κ e^{−κr} Π_{t_j<r} (1 − op_j det_j)`, where P_i is the chance that nothing nearer was kept, plus a uniform
//!     outlier share. An empty cell has probability `e^{−κ t_end} Π (1 − op_j det_j)`.
//!   - Every cell an element fronts at fill [`MIN_FILL`] or more counts, partial hits included, so every
//!     hypothesis is scored on the same echoes (slice 7c: a gate at fill 0.5, with the band below it scored by
//!     located edges, let a wider stick move echoes from the edges into the gate, where each is worth far more).
//! - **Depth over a footprint:** a layer's depth is the mean of the section's depth over the part of the footprint
//!   it meets, weighted by each part's energy (the cosine of incidence), as a scanner's mixed echo is
//!   ([`footprint_depth`]). The misfit spreads the depth but doesn't move it (slice 7b).
//! - **The offset field** (research 130 offset-field-design.md, slices 7 and 7b): per scan, an along-beam offset
//!   on nodes ℓ apart (trilinear hats, normalised) with a unit Gaussian prior, scaled at each element by its surface
//!   class's spread (and the other-epoch part for a scan off the reference epoch). A cell's depths shift by the
//!   field at its front element. Neighbours sharing nodes share one offset, and one scan's joint evidence on a
//!   place is capped by the prior. (An across-beam part didn't help held-out calibration; slice 7b dropped it.)
//!   The surface class k (which sets the along spread, the bias and the noise) is the median class of the view's
//!   echoes on the element's front. A scan's cost is the Laplace marginal of its data over its field
//!   ([`scan_cost`]).
//! - **Weights:** each cell counts `1 / (φ m(ρ))`, with m the footprints per cell (overlapping footprints) and φ
//!   the error model's `dispersion`: one constant for every cell and every hypothesis, so it widens posteriors
//!   without favouring any hypothesis (a quasi-likelihood). It replaced slice 7's ρ_res, `1 + (n − 1) ρ` with n an
//!   element's own footprints, which differed between hypotheses and, once every fronted cell counted, favoured
//!   the thinner (slice 7c).
//!
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A move (add, remove,
//!   change) returns its cost change, and recomputes only the field components the touched element joins or leaves.
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use super::model::{Background, ErrorModel, FIELD_SPACING, GROUND_DB, NU_E, P_OUTLIER, SIGMA_E, STEP, T_MAX, WOOD_DB};
use super::extinction::{self, Counts, Extinction, ExtinctionSpec, RayBg};
use super::range_image::{Cell, RangeImage};
use std::sync::Arc;
use crate::transform::{cross, dot, norm, normalize, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Layers kept per cell. A cell that runs out (rare among a few elements) drops its farthest.
pub const K: usize = 3;

/// The least fill a layer keeps in the k-buffer: the ray likelihood scores every cell an element fronts at this fill
/// or more.
pub const MIN_FILL: f64 = 0.01;

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

/// What a cell's first echo says about a layer: on it, none, in front of it, or behind it (the pulse went through).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Outcome {
    On = 0,
    Empty = 1,
    InFront = 2,
    Behind = 3,
}

impl Outcome {
    /// The outcome of a first echo `obs` (NaN for none) against a layer whose echo is predicted at `depth` with sd
    /// `sd`: on within `3 sd + 1 cm`, in front beyond `3 sd + 3 cm`, else behind. The margins every consumer uses.
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

#[derive(Debug, Clone, Copy)]
struct Layer {
    t: f64,
    spread: f64,
    /// The fill: its mean over the misfit (`f`), and at section radii r − √3 m, r and r + √3 m (`fq`, the misfit's
    /// three-point Gauss–Hermite rule, weights 1/6, 2/3, 1/6).
    f: f64,
    fq: [f64; 3],
    cos: f64,
    elem: u32,
    op: f64,
    /// Where the cell's ray passes the axis: across it (signed, m, positive on `u × (c − o)`'s side) and along it;
    /// for a patch, the distance from the centre and 0.
    x: f64,
    s: f64,
}

impl Layer {
    /// Whether the layer enters the k-buffer: its widest fill reaches [`MIN_FILL`].
    fn fronts(&self) -> bool {
        self.fq[2] >= MIN_FILL
    }
}

/// What one scan says about one element. `cells`: cells where it is the nearest layer (at [`MIN_FILL`] or more);
/// `support`: those with an echo on it; `across`: footprints across its diameter; `cost`: its share of the scan's
/// cost (its ray likelihood at the field's mode, plus an equal share of the field's prior and log-determinant terms), so
/// the rows sum to the total.
#[derive(Debug, Clone, Copy)]
pub struct ElementScan {
    pub elem: u32,
    pub scan: u32,
    pub cells: u32,
    pub support: u32,
    pub across: f64,
    pub surface: u8,
    pub cost: f64,
    /// The offset field's posterior at the element in this scan: the along-beam shift and its sd (m).
    pub shift_along: f64,
    pub sd_along: f64,
}

/// The share of a uniform disc's energy (radius 1) nearer than a line `z` radii from its centre: the CDF of its 1-D
/// marginal, a semicircle.
fn disc_cdf(z: f64) -> f64 {
    let z = z.clamp(-1.0, 1.0);
    0.5 + (z * (1.0 - z * z).sqrt() + z.asin()) / std::f64::consts::PI
}

/// The beam's radius (m) at range `tm`.
fn beam_radius(em: &ErrorModel, tm: f64) -> f64 {
    0.5 * em.footprint(tm)
}

fn t_pdf(x: f64, nu: f64) -> f64 {
    let c = libm::lgamma(0.5 * (nu + 1.0)) - libm::lgamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln();
    (c - 0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln()).exp()
}

/// What links an element to others in one view's components: its offset-field nodes and the cells it fronts.
fn links(e: &Element, layers: &[(usize, Layer)]) -> Vec<(u8, [i64; 3])> {
    OffsetField::hats(&e.mid(), FIELD_SPACING).into_iter().map(|(n, _)| (0u8, n))
        .chain(layers.iter().filter(|(_, l)| l.fronts()).map(|&(c, _)| (1u8, [c as i64, 0, 0])))
        .collect()
}

/// One element's layers in one view, out to the beam's edge and two cells past its section.
fn render(v: &RangeImage, e: &Element, id: u32, em: &ErrorModel) -> Vec<(usize, Layer)> {
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
            let a = beam_radius(em, tm);
            let wide = e.r + 3f64.sqrt() * e.misfit;
            if d > wide + a + 2.0 * STEP * tm {
                continue;
            }
            let fq = [-1.0, 0.0, 1.0].map(|z: f64| {
                let r = (e.r + z * 3f64.sqrt() * e.misfit).max(0.0);
                disc_cdf((r - d) / a) - disc_cdf((-r - d) / a)
            });
            let f = (fq[0] + 4.0 * fq[1] + fq[2]) / 6.0;
            let (t, spread, cos) = footprint_depth(tm, d, e.r, sin, a);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t,
                    spread: (spread * spread + e.misfit * e.misfit).sqrt(),
                    f,
                    fq,
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

/// The echo's depth over a footprint (a disc of radius `a`, across the section its semicircle marginal) whose centre
/// passes `d` from the axis at range `tm`: the mean of the section's depth over the part it meets, each part weighted
/// by its energy (the cosine of incidence, Lambertian); the spread of those depths; and the mean cosine. The misfit
/// stays out: it spreads the depth about the form but doesn't move it. Quadrature: Gauss–Chebyshev of the second
/// kind, 9 nodes.
fn footprint_depth(tm: f64, d: f64, r: f64, sin: f64, a: f64) -> (f64, f64, f64) {
    let (mut sw, mut swc, mut sz, mut szz) = (0.0, 0.0, 0.0, 0.0);
    for j in 1..=9 {
        let th = j as f64 * std::f64::consts::PI / 10.0;
        let (u, w) = (th.cos(), th.sin().powi(2));
        let x = d + u * a;
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
fn render_all(v: &RangeImage, elems: &[Element], em: &ErrorModel) -> Vec<Vec<(usize, Layer)>> {
    elems.par_iter().enumerate().map(|(i, e)| render(v, e, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, its layers of fill `min_f` or more, nearest first.
fn kbuffer(per: &[Vec<(usize, Layer)>], min_f: f64) -> HashMap<usize, Vec<Layer>> {
    let mut buf: HashMap<usize, Vec<Layer>> = HashMap::new();
    for &(cell, ly) in per.iter().flatten() {
        if ly.fq[2] >= min_f {
            buf.entry(cell).or_default().push(ly);
        }
    }
    for ls in buf.values_mut() {
        ls.sort_by(|a, b| a.t.total_cmp(&b.t));
        ls.truncate(K);
    }
    buf
}

/// Each element's surface class in one view: the median class of the view's echoes on its front (mid where none).
fn classes(v: &RangeImage, buf: &HashMap<usize, Vec<Layer>>, n: usize, em: &ErrorModel) -> Vec<u8> {
    let mut ex: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (&cell, ls) in buf {
        let ly = ls[0];
        let r = v.range[cell] as f64;
        if ly.f > 0.5 && v.cell[cell] == Cell::Echoes && (r - ly.t).abs() < 0.03 && v.excess[cell].is_finite() {
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

/// What an element predicts in one cell it fronts ([`Scene::predict`]): its fill, the chance its echo is kept, the
/// echo's predicted depth with the bias and its sd (noise, spread, misfit and the offset field's along spread), the
/// element's surface class, the observed first echo (NaN for none), the [`Outcome`], the cell's weight `1/m(ρ)`, the
/// footprint (m) and where its ray passes the axis (across, along).
pub struct CellPrediction {
    pub cell: usize,
    pub elem: u32,
    pub fill: f64,
    pub kept: f64,
    /// The chance the pulse meets nothing of the diffuse field before the element's return.
    pub reach: f64,
    pub depth: f64,
    pub sd: f64,
    pub surface: u8,
    pub obs: f64,
    pub outcome: Outcome,
    pub weight: f64,
    pub footprint: f64,
    pub x: f64,
    pub s: f64,
}


/// One return a pulse can record: a run of layers (and the ground) within the receiver's resolution of its first,
/// merged. `s` is the chance it is recorded, `mu` and `sd` its echo's depth and spread, `t` its first member's range
/// and `surv` the chance the pulse meets nothing of the diffuse field before it.
#[derive(Clone, Copy)]
struct Return {
    t: f64,
    mu: f64,
    sd: f64,
    s: f64,
    surv: f64,
}

/// The ways a cell's layers (each with its element's surface class) and the ground at `ground` = (range, cos) come
/// back, with every layer shifted by `delta`: each a set of returns, nearest first, and its chance; and the range the
/// beam ends at. Adjacent members resolve into two echoes with [`ErrorModel::p_resolve`] of their separation and
/// energy ratio, else merge at their energy-weighted depth; patterns under 1e-4 are dropped. `taus` are the diffuse
/// field's optical depths to each layer and then to the ground ([`RayBg`]).
fn returns(layers: &[(Layer, u8)], ground: Option<(f64, f64)>, delta: f64, em: &ErrorModel, taus: &[f64]) -> (Vec<(f64, Vec<Return>)>, f64) {
    let t_end = ground.map_or(T_MAX, |g| g.0);
    let mut beam = 1.0;
    // members: (range, depth, depth sd, chance recorded, energy, optical depth)
    let mut m: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::with_capacity(layers.len() + 1);
    for (li, &(ly, k)) in layers.iter().enumerate() {
        if ly.t >= t_end {
            break;
        }
        let e = ly.f * beam;
        let s = ly.op * ([1.0, 4.0, 1.0].iter().zip(ly.fq).map(|(w, f)| w * em.p_detect(WOOD_DB, f * beam * ly.cos, ly.t)).sum::<f64>() / 6.0);
        beam *= 1.0 - ly.f * ly.op;
        m.push((ly.t, ly.t + em.depth_bias(k, ly.t) + delta, em.depth_sd(k, ly.spread), s, ly.op * e * ly.cos * 10f64.powf(WOOD_DB / 10.0), taus[li]));
    }
    if let Some((tg, cg)) = ground {
        m.push((tg, tg, SIGMA_E[2], em.p_detect(GROUND_DB, beam * cg, tg), beam * cg * 10f64.powf(GROUND_DB / 10.0), taus[layers.len()]));
    }
    let n = m.len();
    // each adjacent pair's chance of resolving into two echoes
    let pr: Vec<f64> = (1..n).map(|j| {
        let e = 10.0 * (m[j - 1].4.max(1e-300) / m[j].4.max(1e-300)).log10();
        em.p_resolve(m[j].0 - m[j - 1].0, e.clamp(-60.0, 60.0))
    }).collect();
    let group = |g: &[(f64, f64, f64, f64, f64, f64)]| {
        if g.len() == 1 {
            return Return { t: g[0].0, mu: g[0].1, sd: g[0].2, s: g[0].3, surv: (-g[0].5).exp() };
        }
        // energy weights; members the beam no longer reaches weigh alike (the return is never recorded then)
        let total: f64 = g.iter().map(|x| x.4).sum();
        let wt = |x: &(f64, f64, f64, f64, f64, f64)| if total > 0.0 { x.4 / total } else { 1.0 / g.len() as f64 };
        let mu = g.iter().map(|x| wt(x) * x.1).sum::<f64>();
        let var = g.iter().map(|x| wt(x) * (x.2 * x.2 + (x.1 - mu).powi(2))).sum::<f64>();
        let none: f64 = g.iter().map(|x| 1.0 - x.3).product();
        Return { t: g[0].0, mu, sd: var.sqrt(), s: 1.0 - none, surv: (-g[0].5).exp() }
    };
    let mut out = Vec::new();
    for mask in 0u32..(1 << n.saturating_sub(1)) {
        // bit j set: members j and j + 1 resolve
        let prob: f64 = (0..n.saturating_sub(1)).map(|j| if mask >> j & 1 == 1 { pr[j] } else { 1.0 - pr[j] }).product();
        if prob < 1e-4 {
            continue;
        }
        let mut rets = Vec::with_capacity(n);
        let mut i = 0;
        for j in 0..n {
            if j + 1 == n || mask >> j & 1 == 1 {
                rets.push(group(&m[i..=j]));
                i = j + 1;
            }
        }
        out.push((prob, rets));
    }
    (out, t_end)
}

/// One cell's log-likelihood: its first echo (or none) against the diffuse field along its ray (`bg`) and the
/// returns its layers and the ground give ([`returns`]).
fn cell_ll(echoes: &[f32], layers: &[(Layer, u8)], ground: Option<(f64, f64)>, delta: f64, em: &ErrorModel, bg: &RayBg) -> f64 {
    // the first echo only, for now
    let obs = echoes.first().map_or(f64::NAN, |&r| r as f64);
    let (patterns, t_end) = returns(layers, ground, delta, em, &bg.taus);
    let lik = |rets: &[Return]| {
        let (mut none_kept, mut none_before_obs, mut dens) = (1.0, 1.0, 0.0);
        for r in rets {
            if obs.is_finite() {
                dens += r.surv * none_kept * r.s * t_pdf((obs - r.mu) / r.sd, NU_E) / r.sd;
                if r.t < obs {
                    none_before_obs *= 1.0 - r.s;
                }
            }
            none_kept *= 1.0 - r.s;
        }
        if !obs.is_finite() {
            return (-bg.tau_end).exp() * none_kept;
        }
        if obs < t_end {
            dens += bg.sig_obs * (-bg.tau_obs).exp() * none_before_obs;
        }
        (1.0 - P_OUTLIER) * dens + P_OUTLIER / T_MAX
    };
    if patterns.len() == 1 {
        return lik(&patterns[0].1).max(1e-300).ln();
    }
    patterns.iter().map(|(p, rets)| p * lik(rets)).sum::<f64>().max(1e-300).ln()
}

/// One element's data in one scan: the cells it fronts (its A cells) and the along-beam shift δ of their depths, a
/// row over the scan's field nodes.
struct ElemData {
    elem: usize,
    /// (its ray's diffuse field, echoes, layers, ground, weight, the cell's log-likelihood against the empty scene)
    cells: Vec<(RayBg, Vec<f32>, Vec<(Layer, u8)>, Option<(f64, f64)>, f64, f64)>,
    class: u8,
    support: u32,
    across: f64,
    /// Its row over the field's nodes: the along shift, scaled by its class's along spread (and the other-epoch part).
    along: Vec<(usize, f64)>,
}

impl ElemData {
    /// Its ray likelihood with its cells' depths shifted by δ.
    fn ll(&self, delta: f64, em: &ErrorModel) -> f64 {
        self.cells.iter().map(|(bg, echoes, layers, ground, wt, base)| wt * (cell_ll(echoes, layers, *ground, delta, em, bg) - base)).sum()
    }
}

/// A scan's offset-field nodes: trilinear hats on a grid of spacing ℓ, normalised so the field's variance is the
/// same everywhere (`t(x) = Σ w_j z_j / √Σ w_j²`); each node carries the along-beam offset.
struct OffsetField {
    nodes: HashMap<[i64; 3], usize>,
}

impl OffsetField {
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

/// Newton steps at most per cost.
const MAX_NEWTON: usize = 40;

/// One scan's cost for a set of elements: the Laplace marginal of its data over
/// the field's nodes, `U = −Σ_e A_e(δ̂_e) + ½ ẑᵀẑ + ½ log|Λ|`, Λ the posterior precision at the mode ẑ (the prior is
/// unit; each element's spread is in its row), and a row per element.
fn scan_cost(v: &RangeImage, si: usize, rendered: &[Vec<(usize, Layer)>], elems: &[Element], ref_epoch: u32, em: &ErrorModel, sc: &Background, ext: &Extinction)
               -> (f64, Vec<ElementScan>) {
    let per = rendered;
    let buf = kbuffer(per, MIN_FILL);
    let cls = classes(v, &buf, elems.len(), em);
    let other = v.epoch != ref_epoch;
    let mut field = OffsetField { nodes: HashMap::new() };
    let mut data: Vec<ElemData> = Vec::new();
    let mut by_elem: HashMap<u32, usize> = HashMap::new();
    for (i, e) in elems.iter().enumerate() {
        if per[i].is_empty() {
            continue;
        }
        let k = cls[i];
        let mid = e.mid();
        let s_along = em.field_sd(k, other);
        let along = OffsetField::hats(&mid, FIELD_SPACING).into_iter().map(|(key, w)| (field.node(key), w * s_along)).collect();
        by_elem.insert(i as u32, data.len());
        data.push(ElemData { elem: i, cells: Vec::new(), class: k, support: 0, across: 2.0 * e.r / em.footprint(norm(&sub(&mid, &v.origin))), along });
    }
    // cells in index order, so an element's sums run in the same order in every run (the k-buffer is a hash map)
    let mut order: Vec<usize> = buf.keys().copied().collect();
    order.sort_unstable();
    for cell in order {
        let ls = &buf[&cell];
        let front = ls[0];
        let Some(&d) = by_elem.get(&front.elem) else { continue };
        let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[l.elem as usize])).collect();
        let obs = if v.cell[cell] == Cell::Echoes { v.range[cell] as f64 } else { f64::NAN };
        let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
        let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
        let wt = 1.0 / (em.dispersion * em.multiplicity(if obs.is_finite() { obs } else { front.t }));
        let el = &mut data[d];
        let sd = em.depth_sd(el.class, front.spread);
        el.support += (Outcome::of(obs, front.t, sd) == Outcome::On) as u32;
        let echoes = v.echoes(cell);
        let ts: Vec<f64> = layers.iter().map(|(l, _)| l.t).collect();
        let bg = ext.ray_bg(&v.origin, &w, &ts, ground.map(|g| g.0), ground.map_or(T_MAX, |g| g.0), obs);
        let base = cell_ll(echoes, &[], ground, 0.0, em, &bg.base());
        el.cells.push((bg, echoes.to_vec(), layers, ground, wt, base));
    }
    let n = field.nodes.len();
    let x_of = |el: &ElemData, z: &[f64]| el.along.iter().map(|&(j, w)| w * z[j]).sum::<f64>();
    // sums in a fixed order, so the same elements always give the same cost to the bit
    let objective = |z: &[f64]| 0.5 * z.iter().map(|x| x * x).sum::<f64>() - data.par_iter().map(|el| el.ll(x_of(el, z), em)).collect::<Vec<f64>>().iter().sum::<f64>();
    // Newton with each element's −A as a quadratic in δ by central differences (negative curvature dropped); a step
    // at most 3 prior sds, halved until the objective falls, stopping under a millinat
    let h = 3e-4;
    let mut z = vec![0.0; n];
    let mut hess = nalgebra::DMatrix::<f64>::identity(n, n);
    for it in 0..=MAX_NEWTON {
        let mut grad = nalgebra::DVector::from_column_slice(&z);
        hess = nalgebra::DMatrix::identity(n, n);
        // each element's slope and curvature in parallel (the costly part), then assembled
        let quads: Vec<(f64, f64)> = data.par_iter().map(|el| {
            if el.along.is_empty() {
                return (0.0, 0.0);
            }
            let x0 = x_of(el, &z);
            let (fm, f0, fp) = (el.ll(x0 - h, em), el.ll(x0, em), el.ll(x0 + h, em));
            (-(fp - fm) / (2.0 * h), (-(fp - 2.0 * f0 + fm) / (h * h)).max(0.0))
        }).collect();
        for (el, &(g, curv)) in data.iter().zip(&quads) {
            for &(i, wi) in &el.along {
                grad[i] += g * wi;
                for &(j, wj) in &el.along {
                    hess[(i, j)] += curv * wi * wj;
                }
            }
        }
        if it == MAX_NEWTON || n == 0 {
            break;
        }
        let Some(ch) = hess.clone().cholesky() else { break };
        let mut step = ch.solve(&grad);
        step /= step.amax().max(3.0) / 3.0;
        let f_now = objective(&z);
        let (mut t, mut fall) = (1.0, 0.0);
        while t >= 1e-3 {
            let zn: Vec<f64> = z.iter().zip(step.iter()).map(|(a, b)| a - t * b).collect();
            let f_new = objective(&zn);
            if f_new <= f_now {
                z = zn;
                fall = f_now - f_new;
                break;
            }
            t *= 0.5;
        }
        if t < 1e-3 || fall < 1e-3 {
            break;
        }
    }
    let (logdet, cov) = match hess.clone().cholesky() {
        Some(c) => (2.0 * c.l().diagonal().iter().map(|x| x.ln()).sum::<f64>(), c.inverse()),
        None => (0.0, nalgebra::DMatrix::zeros(n, n)),
    };
    let shared = 0.5 * z.iter().map(|x| x * x).sum::<f64>() + 0.5 * logdet;
    let lls: Vec<f64> = data.par_iter().map(|el| el.ll(x_of(el, &z), em)).collect();
    let u = shared - lls.iter().sum::<f64>();
    let rows = data.iter().zip(&lls).map(|(el, l)| {
        let var: f64 = el.along.iter().map(|&(i, wi)| el.along.iter().map(|&(j, wj)| wi * wj * cov[(i, j)]).sum::<f64>()).sum();
        ElementScan {
            elem: el.elem as u32, scan: si as u32, cells: el.cells.len() as u32, support: el.support, across: el.across, surface: el.class,
            cost: -l + shared / data.len() as f64, shift_along: x_of(el, &z), sd_along: var.max(0.0).sqrt(),
        }
    }).collect();
    (u, rows)
}

/// An element's counts in one view (the latents' sufficient statistics), each cell weighted `1/m(ρ)`.
/// - **Misfit:** over its front cells at fill > 0.5 with an echo within `3 sd + 3 cm` of its sharp depth (its own
///   misfit left out of sd), the weighted sum of `max(r² − sd², 0)`, the weights' sum and the cells.
/// - **Opacity:** over its front cells at fill > 0.9, the weighted counts coded on, empty and behind.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementCounts {
    pub scan: u32,
    pub misfit_ss: f64,
    pub misfit_w: f64,
    pub misfit_cells: u32,
    pub on: f64,
    pub empty: f64,
    pub behind: f64,
}

/// A scene: elements held against some range images, and the cost of a move (synthesis §16's incremental form).
///
/// - Each element is rendered once per image, and re-rendered only when it is added or changed.
/// - A scan's cost is a sum over its components: the elements linked by a shared field node or a shared cell. Their
///   Laplace marginals are independent (the field's prior is diagonal, and an element's terms touch only its nodes
///   and cells), so the sum is the scan's whole marginal.
/// - Component energies are memoised by membership. A move recomputes only the components the touched element joins
///   or leaves, and removing an element restores the cost exactly (synthesis §18 invariant 7).
/// - Element ids are stable and never reused.
pub struct Scene {
    views: Vec<Arc<RangeImage>>,
    em: ErrorModel,
    sc: Background,
    /// The diffuse field (a residual: estimated from the rays no element fronts), every ray's counts and the field
    /// estimated from them all (the empty scene's).
    ext: Extinction,
    ext_all: Counts,
    ext0: Extinction,
    ref_epoch: u32,
    elems: Vec<Option<Element>>,
    /// [image][id]: the element's layers (empty when removed or unseen).
    rendered: Vec<Vec<Vec<(usize, Layer)>>>,
    /// [image]: component members (sorted ids) → its cost and rows; and the scan's cost.
    memo: Vec<HashMap<Vec<u32>, (f64, Vec<ElementScan>)>>,
    scan_u: Vec<f64>,
}

impl Scene {
    /// A scene of `elems` against `views`; with `spec`, the diffuse field is estimated in its box as the residual
    /// of the elements, otherwise it is the background's κ.
    pub fn new(views: Vec<Arc<RangeImage>>, elems: &[Element], em: ErrorModel, sc: Background, spec: Option<ExtinctionSpec>) -> Self {
        let ref_epoch = views.first().map_or(0, |v| v.epoch);
        let n = views.len();
        let ext_all = spec.map_or_else(Counts::new, |sp| {
            views.iter().map(|v| extinction::counts(v, &sp, &em, None)).fold(Counts::new(), |mut a, b| {
                for (k, (h, l)) in b {
                    let e = a.entry(k).or_insert((0.0, 0.0));
                    e.0 += h;
                    e.1 += l;
                }
                a
            })
        });
        let ext0 = spec.map_or_else(|| Extinction::uniform(sc.kappa), |sp| Extinction::from_counts(sc.kappa, sp, &ext_all));
        let mut c = Scene { views, em, sc, ext: ext0.clone(), ext_all, ext0, ref_epoch, elems: elems.iter().map(|e| Some(*e)).collect(),
                             rendered: Vec::new(), memo: vec![HashMap::new(); n], scan_u: vec![0.0; n] };
        c.rendered = c.views.par_iter().map(|v| render_all(v, elems, &c.em)).collect();
        c.ext = c.residual();
        c.refresh(&(0..n).collect::<Vec<_>>());
        c
    }

    /// The scene's cost against the empty scene: its elements' cells, and the diffuse field over every ray against
    /// the empty scene's field.
    pub fn cost(&self) -> f64 {
        self.scan_u.iter().sum::<f64>() + self.field_cost()
    }

    fn field_cost(&self) -> f64 {
        if self.ext.spec.is_none() {
            return 0.0;
        }
        self.ext0.log_lik(&self.ext_all) - self.ext.log_lik(&self.ext_all)
    }

    /// The diffuse field estimated from the rays no element fronts.
    fn residual(&self) -> Extinction {
        let Some(sp) = self.ext.spec else { return self.ext.clone() };
        let mut fronted = Counts::new();
        for (s, v) in self.views.iter().enumerate() {
            let cells: HashSet<usize> = self.rendered[s].iter().flatten().filter(|(_, l)| l.fronts()).map(|&(c, _)| c).collect();
            for (k, (h, l)) in extinction::counts(v, &sp, &self.em, Some(&cells)) {
                let e = fronted.entry(k).or_insert((0.0, 0.0));
                e.0 += h;
                e.1 += l;
            }
        }
        Extinction::from_counts(self.sc.kappa, sp, &extinction::minus(&self.ext_all, &fronted))
    }

    /// Re-estimate the diffuse field from the rays the elements leave (a move: its cost change). Moves that change
    /// elements hold the field as it is; after this one the scene's cost is a fresh scene's.
    pub fn refit_field(&mut self) -> f64 {
        if self.ext.spec.is_none() || self.ext_all.is_empty() {
            return 0.0;
        }
        let before = self.cost();
        self.ext = self.residual();
        for m in self.memo.iter_mut() {
            m.clear();
        }
        let n = self.views.len();
        self.refresh(&(0..n).collect::<Vec<_>>());
        self.cost() - before
    }

    /// Hold a diffuse field estimated elsewhere (another scene's, from other scans) fixed: the cost then has no field
    /// term, and `refit_field` leaves it. The cost change.
    pub fn hold_field(&mut self, ext: &Extinction) -> f64 {
        let before = self.cost();
        self.ext = ext.clone();
        self.ext0 = ext.clone();
        self.ext_all.clear();
        for m in self.memo.iter_mut() {
            m.clear();
        }
        let n = self.views.len();
        self.refresh(&(0..n).collect::<Vec<_>>());
        self.cost() - before
    }

    /// How many scans the scene holds.
    pub fn n_scans(&self) -> usize {
        self.views.len()
    }

    /// The diffuse field.
    pub fn extinction(&self) -> &Extinction {
        &self.ext
    }

    /// Promote: add an element and take the mass it explains out of the diffuse field. Its id and the cost change.
    pub fn promote(&mut self, e: Element) -> (u32, f64) {
        let (id, d) = self.add(e);
        (id, d + self.refit_field())
    }

    /// Demote: remove an element and give the mass back to the diffuse field. The cost change.
    pub fn demote(&mut self, id: u32) -> f64 {
        let d = self.remove(id);
        d + self.refit_field()
    }

    /// The diffuse field and its counts: per voxel with data, (voxel, σ, residual counts, every ray's counts).
    pub fn field(&self) -> (Option<ExtinctionSpec>, Vec<([i32; 3], f64, (f64, f64))>) {
        let mut out: Vec<([i32; 3], f64, (f64, f64))> = self.ext_all.iter().map(|(&k, &c)| (k, *self.ext.sigma.get(&k).unwrap_or(&self.ext.kappa), c)).collect();
        out.sort_by_key(|x| x.0);
        (self.ext.spec, out)
    }

    /// For each of image `v`'s `cells`: the optical depth of the diffuse field across its box, and the ray's entry and
    /// exit ranges (NaN where it misses the box).
    pub fn transmission(&self, v: &RangeImage, cells: &[usize]) -> Vec<(f64, f64, f64)> {
        let Some(sp) = self.ext.spec else { return vec![(0.0, f64::NAN, f64::NAN); cells.len()] };
        cells.par_iter().map(|&k| {
            let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
            match extinction::clip(&sp, &v.origin, &w) {
                Some((a, b)) => (self.ext.tau(&v.origin, &w, b) - self.ext.tau(&v.origin, &w, a), a, b),
                None => (f64::NAN, f64::NAN, f64::NAN),
            }
        }).collect()
    }

    /// A row per (element, scan), by element id.
    pub fn rows(&self) -> Vec<ElementScan> {
        let mut rows: Vec<ElementScan> = (0..self.views.len())
            .flat_map(|s| self.components(s).into_iter().flat_map(move |m| self.memo[s][&m].1.clone()))
            .collect();
        rows.sort_by_key(|r| (r.elem, r.scan));
        rows
    }

    /// Add an element: its id and the cost change.
    pub fn add(&mut self, e: Element) -> (u32, f64) {
        let id = self.elems.len() as u32;
        self.elems.push(Some(e));
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        let touched: Vec<usize> = layers.iter().enumerate().filter(|(_, l)| !l.is_empty()).map(|(s, _)| s).collect();
        for (s, l) in layers.into_iter().enumerate() {
            self.rendered[s].push(l);
        }
        (id, self.refresh(&touched))
    }

    /// Remove an element: the cost change.
    pub fn remove(&mut self, id: u32) -> f64 {
        let touched = self.seen_in(id);
        self.elems[id as usize] = None;
        for r in self.rendered.iter_mut() {
            r[id as usize].clear();
        }
        let d = self.refresh(&touched);
        self.forget(id);
        d
    }

    /// Replace an element: the cost change.
    pub fn change(&mut self, id: u32, e: Element) -> f64 {
        let mut touched = self.seen_in(id);
        self.forget(id);
        self.elems[id as usize] = Some(e);
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        for (s, l) in layers.into_iter().enumerate() {
            if !l.is_empty() && !touched.contains(&s) {
                touched.push(s);
            }
            self.rendered[s][id as usize] = l;
        }
        self.refresh(&touched)
    }

    /// What an element predicts in image `s` against the scene, per cell it fronts: it is rendered for this, not
    /// added, and only the scene's layers in its
    /// cells enter its k-buffer, so its front cells and class are those of the whole set.
    pub fn predict(&self, e: Element, s: usize) -> Vec<CellPrediction> {
        let (v, em) = (&*self.views[s], &self.em);
        let mine = render(v, &e, 0, em);
        let cells: std::collections::HashSet<usize> = mine.iter().map(|&(c, _)| c).collect();
        let mut per: Vec<Vec<(usize, Layer)>> = (0..self.elems.len())
            .filter(|&i| self.elems[i].is_some())
            .map(|i| self.rendered[s][i].iter().filter(|(c, _)| cells.contains(c)).copied().collect::<Vec<_>>())
            .filter(|l| !l.is_empty())
            .collect();
        for (k, l) in per.iter_mut().enumerate() {
            l.iter_mut().for_each(|(_, ly)| ly.elem = k as u32 + 1);
        }
        per.insert(0, mine);
        let buf = kbuffer(&per, MIN_FILL);
        let cls = classes(v, &buf, per.len(), em);
        let k = cls[0];
        let off = em.field_sd(k, false);
        let mut out: Vec<CellPrediction> = buf.iter().filter(|(_, ls)| ls[0].elem == 0).map(|(&cell, ls)| {
            let ly = ls[0];
            // its return: the element's echo, merged with any layer behind within the receiver's resolution
            let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[l.elem as usize])).collect();
            let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
            // the field in front of it: up to a voxel short of its front (a voxel's mean can't place mass within it,
            // and the voxel at the front holds the element's own edges)
            let short = self.ext.spec.map_or(0.0, |sp| sp.voxel);
            let mut taus: Vec<f64> = layers.iter().map(|(l, _)| self.ext.tau(&v.origin, &w, (l.t - short).max(0.0))).collect();
            taus.push(0.0);
            // the most likely way it comes back
            let pats = returns(&layers, None, 0.0, em, &taus).0;
            let r = pats.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1[0];
            let (depth, sd) = (r.mu, r.sd.hypot(off));
            let obs = if v.cell[cell] == Cell::Echoes { v.range[cell] as f64 } else { f64::NAN };
            CellPrediction {
                cell, elem: 0, fill: ly.f, kept: r.s, reach: r.surv, surface: k,
                depth, sd, obs, outcome: Outcome::of(obs, depth, sd), weight: 1.0 / em.multiplicity(ly.t), footprint: em.footprint(ly.t), x: ly.x, s: ly.s,
            }
        }).collect();
        out.sort_by_key(|p| p.cell);
        out
    }

    /// An element's counts against the scene, per image, from its predictions
    /// ([`Scene::predict`]). `shift` (m, per view; empty for none) is subtracted from the depth residuals.
    pub fn counts(&self, e: Element, shift: &[f64]) -> Vec<ElementCounts> {
        (0..self.views.len()).into_par_iter().map(|s| {
            let mut out = ElementCounts { scan: s as u32, ..ElementCounts::default() };
            // the misfit's from the element drawn sharp (its own misfit left out of the depths it is measured
            // from), the opacity's from the element as given
            for p in self.predict(Element { misfit: 0.0, ..e }, s) {
                let r = p.obs - p.depth - shift.get(s).copied().unwrap_or(0.0);
                if p.fill > 0.5 && r.abs() < 3.0 * p.sd + 0.03 {
                    out.misfit_ss += p.weight * (r * r - p.sd * p.sd).max(0.0);
                    out.misfit_w += p.weight;
                    out.misfit_cells += 1;
                }
            }
            for p in self.predict(e, s).into_iter().filter(|p| p.fill > 0.9) {
                match p.outcome {
                    Outcome::On => out.on += p.weight,
                    Outcome::Empty => out.empty += p.weight,
                    Outcome::Behind => out.behind += p.weight,
                    _ => {}
                }
            }
            out
        }).collect()
    }

    /// The cost change of adding an element, leaving the scene as it was.
    pub fn cost_to_add(&mut self, e: Element) -> f64 {
        let (id, d) = self.add(e);
        self.remove(id);
        d
    }

    fn seen_in(&self, id: u32) -> Vec<usize> {
        (0..self.views.len()).filter(|&s| !self.rendered[s][id as usize].is_empty()).collect()
    }

    /// Drop memoised components that contain an element whose form changed or left.
    fn forget(&mut self, id: u32) {
        for m in self.memo.iter_mut() {
            m.retain(|k, _| k.binary_search(&id).is_err());
        }
    }

    /// One scan's components: live elements it sees, linked by shared field nodes and shared cells, each sorted, in
    /// order of their first id.
    fn components(&self, s: usize) -> Vec<Vec<u32>> {
        let ids: Vec<u32> = (0..self.elems.len() as u32).filter(|&i| self.elems[i as usize].is_some() && !self.rendered[s][i as usize].is_empty()).collect();
        let mut parent: Vec<usize> = (0..ids.len()).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut owner: HashMap<(u8, [i64; 3]), usize> = HashMap::new();
        for (k, &id) in ids.iter().enumerate() {
            let e = self.elems[id as usize].as_ref().unwrap();
            for key in links(e, &self.rendered[s][id as usize]) {
                match owner.get(&key) {
                    Some(&j) => {
                        let (a, b) = (find(&mut parent, k), find(&mut parent, j));
                        parent[a.max(b)] = a.min(b);
                    }
                    None => {
                        owner.insert(key, k);
                    }
                }
            }
        }
        let mut groups: Vec<Vec<u32>> = Vec::new();
        let mut at: HashMap<usize, usize> = HashMap::new();
        for k in 0..ids.len() {
            let r = find(&mut parent, k);
            let g = *at.entry(r).or_insert_with(|| { groups.push(Vec::new()); groups.len() - 1 });
            groups[g].push(ids[k]);
        }
        groups
    }

    /// Each hypothesis's cost to add alone (one or more elements), per scan: (hypotheses × scans), in parallel, the
    /// scene unchanged and its diffuse field held. Equal to `cost_to_add` per scan: a hypothesis joins the scan's
    /// components it shares a field node or a fronted cell with, and its cost is that union's against theirs.
    pub fn costs_to_add(&self, hyps: &[Vec<Element>]) -> Vec<Vec<f64>> {
        let n = self.views.len();
        // per scan: its components and which component owns each link key
        let comps: Vec<(Vec<Vec<u32>>, HashMap<(u8, [i64; 3]), usize>)> = (0..n).into_par_iter().map(|s| {
            let cs = self.components(s);
            let mut owner = HashMap::new();
            for (g, m) in cs.iter().enumerate() {
                for &id in m {
                    for key in links(self.elems[id as usize].as_ref().unwrap(), &self.rendered[s][id as usize]) {
                        owner.insert(key, g);
                    }
                }
            }
            (cs, owner)
        }).collect();
        hyps.par_iter().map(|h| (0..n).map(|s| {
            let v = &self.views[s];
            let extra: Vec<Vec<(usize, Layer)>> = h.iter().map(|e| render(v, e, 0, &self.em)).collect();
            if extra.iter().all(|l| l.is_empty()) {
                return 0.0;
            }
            let (cs, owner) = &comps[s];
            let mut touched: Vec<usize> = h.iter().zip(&extra).flat_map(|(e, l)| links(e, l)).filter_map(|k| owner.get(&k).copied()).collect();
            touched.sort_unstable();
            touched.dedup();
            let members: Vec<u32> = touched.iter().flat_map(|&g| cs[g].iter().copied()).collect();
            let mut elems: Vec<Element> = members.iter().map(|&i| self.elems[i as usize].unwrap()).collect();
            let mut rendered: Vec<Vec<(usize, Layer)>> = members.iter().enumerate()
                .map(|(k, &i)| self.rendered[s][i as usize].iter().map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect())
                .collect();
            for (e, l) in h.iter().zip(extra) {
                let k = elems.len() as u32;
                elems.push(*e);
                rendered.push(l.into_iter().map(|(c, ly)| (c, Layer { elem: k, ..ly })).collect());
            }
            let with = scan_cost(v, s, &rendered, &elems, self.ref_epoch, &self.em, &self.sc, &self.ext).0;
            with - touched.iter().map(|&g| self.memo[s][&cs[g]].0).sum::<f64>()
        }).collect()).collect()
    }

    /// Recompute the scans `touched` (their new components, from the memo where it holds them) and return the cost
    /// change.
    fn refresh(&mut self, touched: &[usize]) -> f64 {
        let comps: Vec<(usize, Vec<Vec<u32>>)> = touched.iter().map(|&s| (s, self.components(s))).collect();
        let todo: Vec<(usize, Vec<u32>)> = comps.iter().flat_map(|(s, cs)| cs.iter().filter(|m| !self.memo[*s].contains_key(*m)).map(move |m| (*s, m.clone()))).collect();
        let done: Vec<(usize, Vec<u32>, (f64, Vec<ElementScan>))> = todo.into_par_iter().map(|(s, m)| {
            let elems: Vec<Element> = m.iter().map(|&i| self.elems[i as usize].unwrap()).collect();
            // the component's layers, their elements numbered within it
            let rendered: Vec<Vec<(usize, Layer)>> = m.iter().enumerate()
                .map(|(k, &i)| self.rendered[s][i as usize].iter().map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect())
                .collect();
            let (u, mut rows) = scan_cost(&self.views[s], s, &rendered, &elems, self.ref_epoch, &self.em, &self.sc, &self.ext);
            rows.iter_mut().for_each(|r| r.elem = m[r.elem as usize]);
            (s, m, (u, rows))
        }).collect();
        for (s, m, v) in done {
            self.memo[s].insert(m, v);
        }
        let mut d = 0.0;
        for (s, cs) in comps {
            let u: f64 = cs.iter().map(|m| self.memo[s][m].0).sum();
            d += u - self.scan_u[s];
            self.scan_u[s] = u;
        }
        d
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;
    use crate::scene::range_image::RangeImageParams;
    use crate::scene::model::floor_db;

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
    /// pole by 401 lines (its 1-D marginal, a semicircle), each part's energy its cosine (Lambertian). The pole's echo
    /// sits at its parts' energy-weighted depth and is recorded where it clears the detection floor (wood at
    /// −13.8 dB, 2 dB spread); the wall (−20 dB, face-on) takes the rest of the beam. Returns closer than the
    /// receiver's resolution (0.8 m) merge into one echo at their energy-weighted range; otherwise the first is the
    /// pole's if recorded, else the wall's.
    fn pole_scan(r_pole: f64, y_pole: f64, gap: f64) -> RangeImage {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let em = ErrorModel::default();
        let (mut t, mut d, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let u = |s: u64| ((s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        for l in 0..200usize {
            for k in 0..400usize {
                let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
                let w = dir(zen, az);
                let h = (w[0] * w[0] + w[1] * w[1]).sqrt();
                let miss = y_pole * w[0] / h;
                let tm = y_pole * w[1] / (h * h);
                let a = 0.5 * em.footprint(tm);
                let (mut f, mut ec, mut ez) = (0.0, 0.0, 0.0);
                for q in 0..401 {
                    let z = -1.0 + 2.0 * (q as f64 + 0.5) / 401.0;
                    let wt = (1.0 - z * z).sqrt() * 2.0 / 401.0 * 2.0 / std::f64::consts::PI;
                    let x = miss + z * a;
                    if x.abs() < r_pole {
                        let c = (1.0 - (x / r_pole).powi(2)).sqrt();
                        f += wt;
                        ec += wt * c;
                        ez += wt * c * (tm - r_pole * c / h);
                    }
                }
                let seed = (l * 400 + k) as u64 * 2 + 1;
                let n = 2.0 * (-2.0 * u(seed).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919)).cos();
                let kept = ec > 0.0 && -13.8 + 10.0 * ec.max(1e-12).log10() + n > floor_db(tm);
                let wall = (y_pole + gap) / w[1];
                let range = if !kept {
                    wall
                } else if wall - ez / ec < RESOLUTION {
                    let (ep, ew) = (ec * 10f64.powf(-1.38), (1.0 - f) * 10f64.powf(-2.0));
                    (ep * ez / ec + ew * wall) / (ep + ew)
                } else {
                    ez / ec
                };
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        RangeImage::from_first_echoes(&t, &d, &r, id, [0.0; 3], 0, &RangeImageParams::default())
    }

    fn pole_view() -> Arc<RangeImage> {
        Arc::new(pole_image())
    }

    /// A scene's cost and rows.
    fn cost(v: &Arc<RangeImage>, els: &[Element], em: &ErrorModel, sc: &Background) -> (f64, Vec<ElementScan>) {
        let s = Scene::new(vec![v.clone()], els, *em, *sc, None);
        (s.cost(), s.rows())
    }

    fn stick(x: f64, y: f64, r: f64) -> Element {
        Element { form: Form::Stick { a: [x, y, -0.2], b: [x, y, 0.2] }, r, opacity: 1.0, misfit: 0.002 }
    }

    /// The true pole has negative cost; a stick of the wrong radius, shifted, or porous has more; a stick where
    /// the pulses went through to the wall costs; a scene's move agrees with the difference of two
    /// energies, and rows sum to the total.
    #[test]
    fn the_true_pole_wins() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
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
        let delta = Scene::new(vec![v.clone()], &pair[..1], em, sc, None).cost_to_add(pair[1]);
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let sum: f64 = both.1.iter().map(|r| r.cost).sum();
        assert!((sum - both.0).abs() < 1e-6);
    }

    /// A stick barely wider than the footprint, under the measured (disc) beam: the true radius beats a thinner and a
    /// thicker one (research 130 slice 11: a Gaussian beam's tails expected edge echoes the disc doesn't return, so
    /// a thinner stick won).
    #[test]
    fn a_thin_stick_under_the_disc_beam_beats_thinner_and_thicker() {
        let (em, sc) = (ErrorModel::default(), Background::default());
        let v = thin_pole_view(0.006, 6.0);
        let e = |r: f64| cost(&v, &[stick(0.0, 6.0, r)], &em, &sc).0;
        let truth = e(0.006);
        for r in [0.0042, 0.0084] {
            assert!(e(r) > truth + 0.5, "r {r}: {} vs {truth}", e(r));
        }
    }

    /// A pole 0.5 m in front of a wall, closer than the pulse's resolution: its edge echoes merge with the wall's
    /// into one at their energy-weighted range. With the wall in the scene (the front of a ball 2 m in radius), the
    /// true radius beats a thinner and a thicker one.
    #[test]
    fn a_stick_just_before_a_wall_merges_its_edges() {
        let (em, sc) = (ErrorModel::default(), Background::default());
        let v = Arc::new(pole_scan(0.02, 4.0, 0.5));
        let wall = Element { form: Form::Patch { centre: [0.0, 6.5, 0.0] }, r: 2.0, opacity: 1.0, misfit: 0.002 };
        let e = |r: f64| cost(&v, &[wall, stick(0.0, 4.0, r)], &em, &sc).0;
        let truth = e(0.02);
        for r in [0.014, 0.028] {
            assert!(e(r) > truth + 0.5, "r {r}: {} vs {truth}", e(r));
        }
    }

    /// The diffuse field is a residual: the empty scene's field holds the pole's echoes (its voxel is dense), and
    /// promoting the pole takes them out of the field. Promotion and demotion are exact: the cost after a promotion
    /// is a fresh scene's, its change is that cost, and demotion restores the empty scene.
    #[test]
    fn the_field_is_a_residual_and_its_moves_are_exact() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let spec = Some(ExtinctionSpec { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, prior_path: 1.0 });
        let pole = stick(0.0, 4.0, 0.03);
        let sigma = |c: &Scene| c.field().1.iter().find(|x| x.0 == [5, 10, 3]).map_or(f64::NAN, |x| x.1);
        let mut c = Scene::new(vec![v.clone()], &[], em, sc, spec);
        assert_eq!(c.cost(), 0.0);
        let dense = sigma(&c);
        assert!(dense > 1.0, "{dense}");
        let (id, d) = c.promote(pole);
        let fresh = Scene::new(vec![v.clone()], &[pole], em, sc, spec);
        assert!((c.cost() - fresh.cost()).abs() < 1e-6 * fresh.cost().abs().max(1.0), "{} vs {}", c.cost(), fresh.cost());
        assert!((d - fresh.cost()).abs() < 1e-6 * d.abs().max(1.0));
        assert!(d < 0.0, "{d}");
        assert!(sigma(&c) < 0.5 * dense, "{} vs {dense}", sigma(&c));
        c.demote(id);
        assert!(c.cost().abs() < 1e-9, "{}", c.cost());
    }

    /// Many hypotheses at once: each one's cost per scan equals adding then removing it, the scene unchanged.
    #[test]
    fn costs_to_add_match_add_then_remove() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None);
        let hyps = vec![vec![stick(0.0, 4.0, 0.02)], vec![stick(0.2, 6.0, 0.03), stick(0.25, 6.0, 0.01)], vec![stick(5.0, 4.0, 0.03)]];
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
        }
    }

    /// A scene's moves: adding then removing an element restores its cost exactly (synthesis §18
    /// invariant 7), and each move's cost change equals the full recomputation's difference to rounding, for an
    /// add, an alter and a remove, with an unrelated element far away left untouched.
    #[test]
    fn a_scenes_moves_match_full_recomputation() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let full = |els: &[Element]| cost(&v, els, &em, &sc).0;
        let (pole, far, wide) = (stick(0.0, 4.0, 0.03), stick(0.6, 7.0, 0.03), stick(0.0, 4.0, 0.04));
        let mut c = Scene::new(vec![v.clone()], &[far], em, sc, None);
        let e0 = c.cost();
        assert!((e0 - full(&[far])).abs() < 1e-6 * e0.abs().max(1.0));
        let (id, d_add) = c.add(pole);
        assert!((d_add - (full(&[far, pole]) - full(&[far]))).abs() < 1e-6 * d_add.abs().max(1.0), "{d_add}");
        let d_alter = c.change(id, wide);
        assert!((d_alter - (full(&[far, wide]) - full(&[far, pole]))).abs() < 1e-6 * d_alter.abs().max(1.0), "{d_alter}");
        c.remove(id);
        assert_eq!(c.cost(), e0);
        let t = c.cost_to_add(pole);
        assert_eq!(c.cost(), e0);
        assert!((t - d_add).abs() < 1e-9 * t.abs().max(1.0));
    }

    /// The latents' statistics: the true pole's full-fill cells are on it and its echoes fit its sharp depth (a small
    /// excess over the noise); a stick where the pulses went through to the wall reads behind; the scene is
    /// left as it was.
    #[test]
    fn stats_read_on_for_the_pole_and_behind_for_a_phantom() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[], em, sc, None);
        let pole = c.counts(stick(0.0, 4.0, 0.03), &[])[0];
        assert!(pole.on > 10.0 * (pole.behind + pole.empty).max(0.01), "{pole:?}");
        assert!(pole.misfit_cells > 50 && pole.misfit_ss / pole.misfit_w < 1e-5, "{pole:?}");
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
            let (em, sc) = (ErrorModel::default(), Background::default());
            let e = |r: f64| cost(&v, &[Element { misfit: 0.0, ..stick(0.0, y, r) }], &em, &sc).0;
            let (thin, truth, wide) = (e(0.0028), e(0.004), e(0.0056));
            assert!(wide > truth, "{y} m: wide {wide} truth {truth}");
            assert!(thin > truth - 0.02 * truth.abs(), "{y} m: thin {thin} truth {truth}");
        }
    }

    /// The rendered depth is the footprint's mean on the section, which the misfit (a spread about the form, not a
    /// bias) doesn't move: on the pole's front and halfway out it sits within 0.3 mm of the echoes, whatever the
    /// misfit. (The misfit used to widen the depth band, which a section's edge clips, so the front was rendered
    /// behind the echoes; slice 7b.)
    #[test]
    fn misfit_does_not_move_the_rendered_front() {
        let v = pole_view();
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

    /// The Laplace marginal over the field agrees with Gauss–Hermite quadrature on a single-element query (the
    /// referee), a range offset in the data is recovered by the field's along part, and the field's hats are
    /// normalised.
    #[test]
    fn the_field_integral_matches_quadrature_and_finds_an_offset() {
        for x in [[0.3, 4.1, -0.7], [1.0, 2.0, 3.0]] {
            let h = OffsetField::hats(&x, 2.0);
            assert!((h.iter().map(|(_, w)| w * w).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let el = stick(0.0, 4.0, 0.03);
        // shift the pole's echoes 3 mm deeper: the field's along shift should find about that
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
        assert!((rows[0].shift_along - 0.003).abs() < 0.0015, "{}", rows[0].shift_along);
        // the referee: a single element's data integrated by quadrature over the field's prior
        let rendered = render_all(&v, &[el], &em);
        let (u_lap, out) = cost(&v, &[el], &em, &sc);
        assert_eq!(out.len(), 1);
        // the element sits on one node set: its along shift has prior sd s_along (√Σŵ² = 1)
        let (gh_x, gh_w) = ([-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
                            [0.019953242059045913, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116, 0.019953242059045913]);
        let buf = kbuffer(&rendered, MIN_FILL);
        let cls = classes(&v, &buf, 1, &em);
        let s_along = em.field_sd(cls[0], false);
        let ed = |delta: f64| -> f64 {
            buf.iter().map(|(&cell, ls)| {
                let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[0])).collect();
                let echoes = v.echoes(cell);
                let wt = 1.0 / (em.dispersion * em.multiplicity(echoes.first().map_or(ls[0].t, |&r| r as f64)));
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let ts: Vec<f64> = layers.iter().map(|(l, _)| l.t).collect();
                let bg = Extinction::uniform(sc.kappa).ray_bg(&v.origin, &w, &ts, None, T_MAX, echoes.first().map_or(f64::NAN, |&r| r as f64));
                wt * (cell_ll(echoes, &layers, None, delta, &em, &bg) - cell_ll(echoes, &[], None, 0.0, &em, &bg.base()))
            }).sum()
        };
        let m = gh_x.iter().map(|&x| ed(std::f64::consts::SQRT_2 * s_along * x)).fold(f64::NEG_INFINITY, f64::max);
        let quad = -(m + (gh_x.iter().zip(&gh_w).map(|(&x, &w)| w * (ed(std::f64::consts::SQRT_2 * s_along * x) - m).exp()).sum::<f64>()
            / std::f64::consts::PI.sqrt()).ln());
        assert!((u_lap - quad).abs() < 0.1 * quad.abs().max(1.0), "laplace {u_lap} quadrature {quad}");
    }

    /// A patch renders as a disc: full fill at its centre, none beyond its rim; a patch of the pole's size where the
    /// pole is beats one off to the side; `predict` codes the pole's front cells on, and a stick beside it behind.
    #[test]
    fn patches_render_as_discs_and_predict_codes_cells() {
        let v = pole_view();
        let (em, sc) = (ErrorModel::default(), Background::default());
        let patch = |x: f64| Element { form: Form::Patch { centre: [x, 4.0, 0.0] }, r: 0.03, opacity: 1.0, misfit: 0.0 };
        let layers = render(&v, &patch(0.0), 0, &em);
        let f_max = layers.iter().map(|(_, l)| l.f).fold(0.0, f64::max);
        assert!(f_max > 0.99, "{f_max}");
        assert!(layers.iter().all(|(_, l)| l.x < 0.03 + 0.012 || l.f < 0.01));
        let e = |s: Element| cost(&v, &[s], &em, &sc).0;
        assert!(e(patch(0.0)) < e(patch(0.1)));
        let p = Scene::new(vec![v.clone()], &[], em, sc, None).predict(stick(0.0, 4.0, 0.03), 0);
        let on = p.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::On);
        let beside = Scene::new(vec![v.clone()], &[], em, sc, None).predict(stick(0.1, 4.0, 0.03), 0);
        let behind = beside.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::Behind);
        assert!(on && behind);
        assert!(p.iter().all(|q| q.weight > 0.0 && q.weight <= 1.0));
    }
}
