// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scene of elements scored against scans' range images by the ray likelihood (3dtrees research 130): its cost,
//! `−log` of its likelihood ratio against the empty scene, the cost change of a move, what an element predicts in an
//! image, and how well the scans pin an element's geometry down.
//!
//! - **Elements** ([`Element`]): a stick (an axis, its section's mean radius a₀) or a leaf patch (a ball with its
//!   extent, a disc in every view), each with an opacity and a shape mismatch. Opacity is the chance a pulse the
//!   section fills meets solid there, its gaps wider than the footprint (synthesis §1's "extent, opacity"). The
//!   mismatch is the sd of the section's and axis's departure from the form, which blurs the fill and spreads the
//!   depth. A stick reflects as wood, a patch as leaf ([`Settings`]'s `refl_wood`, `refl_leaf`).
//! - **The k-buffer:** each element renders into each view's cells. A cell keeps up to [`K`] layers, nearest
//!   first, each with a depth, a depth spread, a fill and a cosine. Fill is the beam (a uniform disc of the
//!   footprint's diameter, as AM's raw edges measure it) convolved with the section, at three section radii over
//!   the mismatch; an echo's chance is the detection's mean over them. Along a stick's axis it is cut square, so a
//!   chain of sticks joins without seams.
//! - **The ray likelihood, one cell:** its pulse's echoes, or none, against a diffuse echo hazard κ along the ray (the
//!   background), the layers in depth order, then a ground plane.
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is recorded with
//!     `op_i det_i`, where `det_i` is the chance it clears the detection floor ([`Settings::p_detect`]):
//!     `Φ((ρ_dB + 10 log10(e_i cos ι_i) − floor(t_i)) / σ)`.
//!   - Adjacent returns (layers and the ground) resolve into two echoes with the instrument's
//!     [`Settings::p_resolve`] of their separation and energy ratio; else they merge into one, recorded if any
//!     member is, at their energy-weighted depth (energy: op · e · cos · reflectance), its spread their spread about
//!     it. A cell's likelihood is the mixture over these merge patterns.
//!   - The first echo at r has density `Σ_i P_i op_i det_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i)` over the
//!     returns, plus `κ e^{−κr} Π_{t_j<r} (1 − op_j det_j)`, where P_i is the chance that nothing nearer was kept,
//!     plus a uniform outlier share. An empty cell has probability `e^{−κ t_end} Π (1 − op_j det_j)`.
//!   - **The rest of the pulse** (3dtrees research 130 step 3c, [`Pulse::lik`]): every later echo is read the same
//!     way, as one of the returns behind the last one read, an echo of the background (in the beam the echoes so far
//!     left, read from their reflectance; behind an edge of fill f, `10 log10 (1 − f)` dimmer) or an outlier; the
//!     background's path restarts the receiver's resolution after each echo. After the last echo, the
//!     returns left and the background's path record nothing, unless the pulse is at the instrument's cap (it says
//!     nothing past its last echo). Where the layers leave some of the beam, a declared share of cells has an
//!     unmodelled surface within the resolution behind them, which merges with their return or resolves
//!     ([`Settings`]' `behind_share`).
//!   - **Located edges** (3dtrees research 130 step 2): the echo's reflectance x (dB) is read beside its range. A
//!     return predicts `10 log10 Σ e_i cos^k ι_i 10^(ρ_i / 10)` over its members (one layer, or a merge with a
//!     surface behind), at the mismatch's three section radii, and x reads about it with the per-echo noise, wider at
//!     a partial fill, truncated by the detection floor ([`Settings::refl_density`], [`Return::refl_density`]); the hazard's and the outliers' echoes read the background's
//!     spread ([`Settings::refl_background`]). So a silhouette's partial cells place its edge within a cell. Each
//!     element's reflectance level (its offset from its surface's reflectance, per scan) is a latent beside its
//!     shift, so an unknown bark brightness is read as a level, not a width.
//!   - Every cell an element fronts at fill [`MIN_FILL`] or more counts, partial hits included, so every hypothesis
//!     is scored on the same echoes (slice 7c: a gate at fill 0.5, with the band below it scored by located edges,
//!     let a wider stick move echoes from the edges into the gate, where each is worth far more).
//! - **The element's echo** ([`front_echo`]): the first return of the most likely way a cell comes back, its sd
//!   widened by the alignment error and the stem front shift. [`Outcome::of`] against it is the one rule for "the
//!   cell's first echo is the element's", used by the surface class, the cells_on, class evidence, `predict` and
//!   `counts`; a cell mostly filled ([`MOSTLY`]) is where an echo reads the element's surface.
//! - **Depth over a footprint:** a layer's depth is the mean of the section's depth over the part of the footprint
//!   it meets, weighted by each part's energy (the cosine of incidence), as a scanner's mixed echo is
//!   ([`footprint_depth`]). The mismatch spreads the depth but doesn't move it (slice 7b).
//! - **The scan alignment error** (research 130 offset-field-design.md, slices 7 and 7b): per scan, an along-beam
//!   shift on nodes ℓ apart (trilinear hats, normalised) with a unit Gaussian prior, scaled at each element by its
//!   surface class's spread (and the other-epoch part for a scan off the reference epoch, which the caller
//!   declares). A cell's depths move by the shift at its front element. Neighbours sharing nodes share one shift,
//!   and one scan's joint evidence on a place is capped by the prior. The surface class k (which sets the along
//!   spread, the bias and the noise) is the median class of the view's echoes on the element. With it, the **stem
//!   front shift** (slice 14): a node of the element's own per scan, its section's departure from round as that view
//!   sees it. A scan's cost is the Laplace marginal of its data over both and the reflectance levels
//!   ([`solve_alignment`]); between refits the alignment is held ([`Held`], research 130 step 3a).
//! - **Weights:** each cell counts [`Settings::cell_weight`], `1 / (φ m(ρ))`, with m the footprints per cell
//!   (overlapping footprints, on the image's own lattice step) and φ the evidence discount per cell: one constant
//!   for every cell and every hypothesis, so it widens posteriors without favouring any hypothesis (a
//!   quasi-likelihood). It replaced slice 7's ρ_res, `1 + (n − 1) ρ` with n an element's own footprints, which
//!   differed between hypotheses and, once every fronted cell counted, favoured the thinner (slice 7c).
//!
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A move (add, remove,
//!   change) returns its cost change with the alignment held, recomputing only the terms of the elements whose cells
//!   it touches; `refit_alignment` re-solves the alignment. Its rows ([`ElementScan`]) split each (element, scan)'s cost into its parts, and
//!   [`Scene::information`] gives the curvature of an element's cost in its geometry.
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use super::settings::{Background, Settings, ALIGN_SPACING, P_OUTLIER, T_MAX};
use super::foliage::{self, Counts, Foliage, FoliageBox, RayFoliage};
use super::range_image::{Cell, RangeImage};
use std::sync::Arc;
use crate::transform::{cross, dot, norm, normalize, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Layers kept per cell. A cell that runs out (rare among a few elements) drops its farthest.
pub const K: usize = 3;

/// The least fill a layer keeps in the k-buffer: the ray likelihood scores every cell an element fronts at this fill
/// or more.
pub const MIN_FILL: f64 = 0.01;

/// The fill over which a cell is mostly the element's: its echo there reads the element's surface (its class, class
/// evidence and shape mismatch).
pub const MOSTLY: f64 = 0.5;

/// An element's form.
#[derive(Debug, Clone, Copy)]
pub enum Form {
    /// An axis from `a` to `b` (world, m).
    Stick { a: Point, b: Point },
    /// A ball about `centre`: a leaf patch or clump, its section a disc in every view.
    Patch { centre: Point },
}

/// An element hypothesis: its form, its section's radius (a stick's a₀; a patch's extent), its opacity (1 solid)
/// and its shape mismatch sd (m). The caller supplies opacity and mismatch (each with a declared default).
#[derive(Debug, Clone, Copy)]
pub struct Element {
    pub form: Form,
    pub r: f64,
    pub opacity: f64,
    pub mismatch: f64,
}

impl Element {
    fn mid(&self) -> Point {
        match self.form {
            Form::Stick { a, b } => [0, 1, 2].map(|k| 0.5 * (a[k] + b[k])),
            Form::Patch { centre } => centre,
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
    /// The fill: its mean over the mismatch (`f`), and at section radii r − √3 m, r and r + √3 m (`fq`, the
    /// mismatch's three-point Gauss–Hermite rule, weights 1/6, 2/3, 1/6).
    f: f64,
    fq: [f64; 3],
    cos: f64,
    /// The surface's reflectance (dB at full fill, face-on), by the element's form.
    db: f64,
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

/// What one scan says about one element, its cost split into parts that sum to `cost`.
/// - `cells`: cells where it is the nearest layer (at [`MIN_FILL`] or more); `cells_on`: those whose first echo is
///   its ([`Outcome::On`]); `across`: footprints across its diameter; `surface`: its surface class in this scan.
/// - `arc`: the arc of a stick's outline its echoes span (rad, 0 to π; the support function's coverage, as the
///   candidates' arcs): `asin(x_max / r) − asin(x_min / r)` over its `cells_on` rays' places across its axis, so π
///   where the scan sees its whole near side, less where it is hidden or seen past the edge only.
/// - The ray likelihood's cost at the alignment's mode, by what each cell's first echo says about the element:
///   `ray_on` (its echoes), `ray_empty` (pulses with no echo), `ray_in_front` (hidden by something nearer) and
///   `ray_behind` (pulses that passed through it: evidence against).
/// - `class`: class evidence's cost (0 while it is off); `alignment`: its own nodes' (stem front shift, reflectance
///   level) prior and log-determinant terms, and an equal share of its scan's alignment part ([`Held`]).
/// - `discount`: the evidence discount φ every cell's evidence was divided by.
/// - The posterior at the element in this scan, along the beam (m): the scan alignment error's shift and sd, and the
///   stem front shift's (0 when it is off); and its reflectance level's (dB; 0 while reflectance isn't read). Shifts
///   and sds are the held alignment's (exact after a refit).
#[derive(Debug, Clone, Copy)]
pub struct ElementScan {
    pub elem: u32,
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
    pub class: f64,
    pub alignment: f64,
    pub discount: f64,
    pub align_shift: f64,
    pub align_sd: f64,
    pub front_shift: f64,
    pub front_shift_sd: f64,
    pub refl_level: f64,
    pub refl_level_sd: f64,
}

/// The share of a uniform disc's energy (radius 1) nearer than a line `z` radii from its centre: the CDF of its 1-D
/// marginal, a semicircle.
fn disc_cdf(z: f64) -> f64 {
    let z = z.clamp(-1.0, 1.0);
    0.5 + (z * (1.0 - z * z).sqrt() + z.asin()) / std::f64::consts::PI
}

fn t_pdf(x: f64, nu: f64) -> f64 {
    let c = libm::lgamma(0.5 * (nu + 1.0)) - libm::lgamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln();
    (c - 0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln()).exp()
}

/// One element's layers in one view, out to the beam's edge and two cells past its section.
fn render(v: &RangeImage, e: &Element, id: u32, em: &Settings) -> Vec<(usize, Layer)> {
    let mut out = Vec::new();
    if e.invalid().is_some() {
        return out;
    }
    let (u, half, db) = match e.form {
        Form::Stick { a, b } => {
            let ax = sub(&b, &a);
            let len = norm(&ax);
            (Some(ax.map(|x| x / len)), 0.5 * len, em.refl_wood)
        }
        Form::Patch { .. } => (None, 0.0, em.refl_leaf),
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
            let a = 0.5 * em.footprint(tm);
            let wide = e.r + 3f64.sqrt() * e.mismatch;
            if d > wide + a + 2.0 * v.step() * tm {
                continue;
            }
            let fq = [-1.0, 0.0, 1.0].map(|z: f64| {
                let r = (e.r + z * 3f64.sqrt() * e.mismatch).max(0.0);
                disc_cdf((r - d) / a) - disc_cdf((-r - d) / a)
            });
            let f = (fq[0] + 4.0 * fq[1] + fq[2]) / 6.0;
            let (t, spread, cos) = footprint_depth(tm, d, e.r, sin, a);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t,
                    spread: (spread * spread + e.mismatch * e.mismatch).sqrt(),
                    f,
                    fq,
                    cos: (cos * sin).max(0.05),
                    db,
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
/// by its energy (the cosine of incidence, Lambertian); the spread of those depths; and the mean cosine over the part
/// it meets. The mismatch stays out: it spreads the depth about the form but doesn't move it. Quadrature: 24 midpoints
/// over the part the footprint meets, so a sliver at the edge is resolved as well as the middle (its cosine sets a
/// located edge's reflectance; 9 nodes over the whole footprint read a 4 % sliver's as the grazing 0.05).
fn footprint_depth(tm: f64, d: f64, r: f64, sin: f64, a: f64) -> (f64, f64, f64) {
    let (lo, hi) = ((d - a).max(-r), (d + a).min(r));
    if hi <= lo {
        // past the edge: the grazing surface
        let c = (1.0 - (d.min(r) / r).powi(2)).max(0.0).sqrt();
        return (tm - r * c / sin, 0.0, c);
    }
    let (mut sw, mut swc, mut sz, mut szz) = (0.0, 0.0, 0.0, 0.0);
    const N: usize = 24;
    for j in 0..N {
        let x = lo + (hi - lo) * (j as f64 + 0.5) / N as f64;
        let w = (1.0 - ((x - d) / a).powi(2)).max(0.0).sqrt();
        let c = (1.0 - (x / r).powi(2)).max(0.0).sqrt();
        let z = -r * c / sin;
        sw += w;
        swc += w * c;
        sz += w * c * z;
        szz += w * c * z * z;
    }
    if swc <= 0.0 {
        let c = (1.0 - (d.min(r) / r).powi(2)).max(0.0).sqrt();
        return (tm - r * c / sin, 0.0, c);
    }
    let z = sz / swc;
    (tm + z, (szz / swc - z * z).max(0.0).sqrt(), swc / sw)
}

/// Every element rendered into one view.
fn render_all(v: &RangeImage, elems: &[Element], em: &Settings) -> Vec<Vec<(usize, Layer)>> {
    elems.par_iter().enumerate().map(|(i, e)| render(v, e, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, its layers that front it ([`MIN_FILL`] or more), nearest first.
fn kbuffer(per: &[Vec<(usize, Layer)>]) -> HashMap<usize, Vec<Layer>> {
    let mut buf: HashMap<usize, Vec<Layer>> = HashMap::new();
    for &(cell, ly) in per.iter().flatten() {
        if ly.fronts() {
            buf.entry(cell).or_default().push(ly);
        }
    }
    for ls in buf.values_mut() {
        ls.sort_by(|a, b| a.t.total_cmp(&b.t));
        ls.truncate(K);
    }
    buf
}

/// A cell's first echo (NaN for none) and the ground along its ray (range, cos), if the background has one.
fn obs_ground(v: &RangeImage, cell: usize, sc: &Background) -> (f64, Option<(f64, f64)>, Point) {
    let obs = if v.cell[cell] == Cell::Echoes { v.range[cell] as f64 } else { f64::NAN };
    let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
    let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
    (obs, ground, w)
}

/// A cell as the likelihood reads it: its first echo (NaN for none), the ground along its ray, its weight
/// ([`Settings::cell_weight`], at its first echo's range or else its front layer's) and what its ray meets of the
/// foliage field to each layer (ranges `ts`, nearest first), the ground, the beam's end and its first echo.
struct CellView {
    obs: f64,
    ground: Option<(f64, f64)>,
    wt: f64,
    bg: RayFoliage,
}

fn cell_view(v: &RangeImage, cell: usize, ts: &[f64], em: &Settings, sc: &Background, ext: &Foliage) -> CellView {
    let (obs, ground, w) = obs_ground(v, cell, sc);
    let wt = em.cell_weight(if obs.is_finite() { obs } else { ts[0] }, v.step());
    let bg = ext.ray_bg(&v.origin, &w, ts, ground.map(|g| g.0), ground.map_or(T_MAX, |g| g.0), v.echoes(cell), em.resolution());
    CellView { obs, ground, wt, bg }
}

/// The front element's echo in a cell as the likelihood reads it: the first return of the most likely way the cell's
/// layers (each with its surface class) and the ground come back ([`returns`]), with its sd widened by the alignment
/// error's and the stem front shift's prior spreads (an element of radius `r`; `other` for a scan off the reference
/// epoch). [`Outcome::of`] against it is the one rule for "the cell's first echo is the element's".
fn front_echo(layers: &[(Layer, u8)], ground: Option<(f64, f64)>, em: &Settings, taus: &[f64], r: f64, other: bool) -> Return {
    let pats = returns(layers, ground, 0.0, 0.0, em, taus, None).0;
    let first = pats.iter().max_by(|a, b| a.0.total_cmp(&b.0)).and_then(|p| p.1.first().copied());
    let mut f = first.unwrap_or(Return { t: f64::NAN, mu: f64::NAN, sd: f64::NAN, s: 0.0, s_q: [0.0; 3], db: [f64::NAN; 3], fill: [0.0; 3], tau: f64::INFINITY, beam: 1.0 });
    f.sd = f.sd.hypot(em.align_sd(layers[0].1, other)).hypot(em.front_shift_sd(r));
    f
}

/// Each element's surface class in one view: the median class of the view's echoes on it, in cells it mostly fills
/// (mid where none). The class isn't known yet when these echoes are chosen, so they are read under the mid class.
fn classes(v: &RangeImage, buf: &HashMap<usize, Vec<Layer>>, elems: &[Element], em: &Settings, sc: &Background, other: bool) -> Vec<u8> {
    let mut ex: Vec<Vec<f64>> = vec![Vec::new(); elems.len()];
    for (&cell, ls) in buf {
        let ly = ls[0];
        if ly.f <= MOSTLY || v.cell[cell] != Cell::Echoes || !v.excess[cell].is_finite() {
            continue;
        }
        let (obs, ground, _) = obs_ground(v, cell, sc);
        let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, 1)).collect();
        let fe = front_echo(&layers, ground, em, &vec![0.0; layers.len() + 1], elems[ly.elem as usize].r, other);
        if Outcome::of(obs, fe.mu, fe.sd) == Outcome::On {
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

/// What an element predicts in one cell it fronts ([`Scene::predict`]), on the path the likelihood scores: its fill,
/// the chance its echo is recorded, the chance the pulse arrives (meets nothing of the foliage field before the
/// element's return), the echo's predicted depth and sd ([`front_echo`]) and reflectance (dB, at the element's
/// declared reflectance and its own section radius, without the shape mismatch's blur; NaN where that section
/// misses the beam), the mean cos ι over the part of the footprint the element meets, the element's surface class,
/// the observed first echo (NaN for none) and its reflectance (NaN unknown), the [`Outcome`], the cell's weight
/// ([`Settings::cell_weight`]), the footprint (m) and where its ray passes the axis (across, along).
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
    pub weight: f64,
    pub footprint: f64,
    pub across: f64,
    pub along: f64,
}


/// One return a pulse can record: a run of layers (and the ground) within the receiver's resolution of its first,
/// merged. `s` is the chance it is recorded, `mu` and `sd` its echo's depth and spread; `s_q` the chance it is
/// recorded, `db` its echo's predicted reflectance (dB) and `fill` the share of the beam it meets, at the shape
/// mismatch's three section radii (as [`Layer`]'s `fq`); `t` its first member's range, `tau` the foliage field's
/// optical depth to it (the pulse meets nothing of the field before it with `e^{−τ}`) and `beam` the share of the
/// beam the scene leaves behind its last member.
#[derive(Clone, Copy)]
struct Return {
    t: f64,
    mu: f64,
    sd: f64,
    s: f64,
    s_q: [f64; 3],
    db: [f64; 3],
    fill: [f64; 3],
    tau: f64,
    beam: f64,
}

impl Return {
    /// The density of its echo's reflectance `x` (dB) as recorded ([`Settings::refl_density`]): a mixture over the
    /// shape mismatch's three section radii, each weighted by its rule weight and its chance of being recorded (a
    /// radius whose section misses the beam records nothing, so it says nothing of the reflectance), each widened by
    /// the reflectance's spread between adjacent radii. Three sharp points would leave gaps between them that a
    /// reflectance noise of tenths of a dB reads as misfits.
    fn refl_density(&self, x: f64, em: &Settings) -> f64 {
        if !x.is_finite() || em.refl_noise <= 0.0 {
            return 1.0;
        }
        let w: [f64; 3] = [0, 1, 2].map(|q| [1.0, 4.0, 1.0][q] * self.s_q[q]);
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return 1.0;
        }
        let gap = |a: f64, b: f64| if a > -200.0 && b > -200.0 { (a - b).abs() / 2.0 } else { 0.0 };
        let spread = gap(self.db[1], self.db[0]).max(gap(self.db[2], self.db[1]));
        (0..3).filter(|&q| w[q] > 0.0).map(|q| w[q] * em.refl_density(x, self.db[q], spread, self.fill[q], self.t)).sum::<f64>() / total
    }
}

/// One member of a cell's returns, a layer or the ground: its range, its echo's depth and sd, the chance it is
/// recorded, its energy (the merge's weight: opacity × the beam it meets × cos ι × reflectance), its energy share
/// when it is hit and the share of the beam it meets, at the shape mismatch's three section radii (the share without
/// the opacity; its echo's reflectance is `10 log10` of it), the foliage field's optical depth to it and the share of
/// the beam left behind it.
struct Member {
    t: f64,
    depth: f64,
    sd: f64,
    s: f64,
    s_q: [f64; 3],
    energy: f64,
    share: [f64; 3],
    fill: [f64; 3],
    tau: f64,
    beam: f64,
}

/// The ways a cell's layers (each with its element's surface class) and the ground at `ground` = (range, cos) come
/// back, with every layer shifted by `shift` along the beam and its reflectance by `level` (dB): each a set of
/// returns, nearest first, and its chance; and the range the beam ends at. Adjacent members resolve into two echoes
/// with [`Settings::p_resolve`] of their separation and energy ratio, else merge at their energy-weighted depth and
/// summed energy share; patterns under 1e-4 are dropped. `taus` are the foliage field's optical depths to each layer
/// and then to the ground ([`RayFoliage`]). With `behind` (its range and the field's optical depth to it), an
/// unmodelled surface lies there, behind the layers, and takes the rest of the beam ([`Settings`]'s `behind_share`,
/// `behind_refl`).
#[allow(clippy::too_many_arguments)]
fn returns(layers: &[(Layer, u8)], ground: Option<(f64, f64)>, shift: f64, level: f64, em: &Settings, taus: &[f64], behind: Option<(f64, f64)>) -> (Vec<(f64, Vec<Return>)>, f64) {
    let t_end = ground.map_or(T_MAX, |g| g.0);
    let mut beam = 1.0;
    let mut m: Vec<Member> = Vec::with_capacity(layers.len() + 1);
    for (li, &(ly, k)) in layers.iter().enumerate() {
        if ly.t >= t_end {
            break;
        }
        let db = ly.db + level;
        let cos = ly.cos.powf(em.refl_incidence);
        let s_q = ly.fq.map(|f| ly.op * em.p_detect(db, f * beam * cos, ly.t));
        let s = (s_q[0] + 4.0 * s_q[1] + s_q[2]) / 6.0;
        let (b0, unit) = (beam, beam * cos * 10f64.powf(db / 10.0));
        let share = ly.fq.map(|f| f * unit);
        beam *= 1.0 - ly.f * ly.op;
        m.push(Member { t: ly.t, depth: ly.t + em.depth_bias(k, ly.t) + shift, sd: em.depth_sd(k, ly.spread), s, s_q, energy: ly.op * ly.f * unit, share,
                        fill: ly.fq.map(|f| f * b0), tau: taus[li], beam });
    }
    if let Some((t, tau)) = behind {
        // an unmodelled surface taking the rest of the beam (unless the ground is nearer)
        if t < t_end {
            let unit = beam * 10f64.powf(em.behind_refl / 10.0);
            let s = em.p_detect(em.behind_refl, beam, t);
            m.push(Member { t, depth: t + shift, sd: em.noise(2), s, s_q: [s; 3], energy: unit, share: [unit; 3], fill: [beam; 3], tau, beam: 0.0 });
            beam = 0.0;
        }
    }
    if let Some((tg, cg)) = ground {
        let cg = cg.powf(em.refl_incidence);
        let share = beam * cg * 10f64.powf(em.refl_ground / 10.0);
        let s = em.p_detect(em.refl_ground, beam * cg, tg);
        m.push(Member { t: tg, depth: tg, sd: em.noise(2), s, s_q: [s; 3], energy: share, share: [share; 3],
                        fill: [beam; 3], tau: taus[layers.len()], beam: 0.0 });
    }
    let n = m.len();
    // each adjacent pair's chance of resolving into two echoes
    let pr: Vec<f64> = (1..n).map(|j| {
        let e = 10.0 * (m[j - 1].energy.max(1e-300) / m[j].energy.max(1e-300)).log10();
        em.p_resolve(m[j].t - m[j - 1].t, e.clamp(-60.0, 60.0))
    }).collect();
    let group = |g: &[Member]| {
        let db = [0, 1, 2].map(|q| 10.0 * g.iter().map(|x| x.share[q]).sum::<f64>().max(1e-30).log10());
        let fill = [0, 1, 2].map(|q| g.iter().map(|x| x.fill[q]).sum::<f64>().min(1.0));
        if g.len() == 1 {
            return Return { t: g[0].t, mu: g[0].depth, sd: g[0].sd, s: g[0].s, s_q: g[0].s_q, db, fill, tau: g[0].tau, beam: g[0].beam };
        }
        // energy weights; members the beam no longer reaches weigh alike (the return is never recorded then)
        let total: f64 = g.iter().map(|x| x.energy).sum();
        let wt = |x: &Member| if total > 0.0 { x.energy / total } else { 1.0 / g.len() as f64 };
        let mu = g.iter().map(|x| wt(x) * x.depth).sum::<f64>();
        let var = g.iter().map(|x| wt(x) * (x.sd * x.sd + (x.depth - mu).powi(2))).sum::<f64>();
        let none: f64 = g.iter().map(|x| 1.0 - x.s).product();
        let s_q = [0, 1, 2].map(|q| 1.0 - g.iter().map(|x| 1.0 - x.s_q[q]).product::<f64>());
        Return { t: g[0].t, mu, sd: var.sqrt(), s: 1.0 - none, s_q, db, fill, tau: g[0].tau, beam: g[g.len() - 1].beam }
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

/// One cell's log-likelihood: its pulse's echoes (none for a miss), at their ranges and reflectances `refls` (dB; empty
/// unknown), the pulse `capped` when it returned as many echoes as the instrument records, against the foliage field
/// along its ray (`bg`) and the returns its layers and the ground give ([`returns`]), the layers shifted by `shift`
/// along the beam and `level` in reflectance ([`Pulse::lik`]). With `anchor` (a range and the field's optical depth
/// to it: [`behind_anchor`]), a mixture with an unmodelled surface within the receiver's resolution behind it, at
/// three distances over it (research 130 step 3c). The surface is there or not whatever the elements are, so the
/// empty scene's cell takes the same mixture at the same anchor: what the branch costs a cell that refutes it then
/// cancels, and only what it explains counts.
#[allow(clippy::too_many_arguments)]
fn cell_ll(echoes: &[f32], refls: &[f32], capped: bool, layers: &[(Layer, u8)], ground: Option<(f64, f64)>, shift: f64, level: f64, em: &Settings,
           bg: &RayFoliage, anchor: Option<(f64, f64)>) -> f64 {
    let lik = |behind: Option<(f64, f64)>| {
        let (patterns, t_end) = returns(layers, ground, shift, level, em, &bg.taus, behind);
        let pulse = Pulse { echoes, refls, capped, t_end, em, bg };
        patterns.iter().map(|(p, rets)| p * pulse.lik(rets, 0, 0, 1.0, 1.0, 1.0, 0.0)).sum::<f64>()
    };
    let mut l = lik(None);
    if let Some((t, tau)) = anchor {
        let with = [1.0, 3.0, 5.0].iter().map(|&u| lik(Some((t + shift + u / 6.0 * em.resolution(), tau)))).sum::<f64>() / 3.0;
        l = (1.0 - em.behind_share) * l + em.behind_share * with;
    }
    l.max(1e-300).ln()
}

/// Where a cell's unmodelled surface would lie behind ([`cell_ll`]): its last layer's range and the field's optical
/// depth to it, where `behind_share` is set and its layers leave some of the beam (an edge).
fn behind_anchor(layers: &[(Layer, u8)], taus: &[f64], em: &Settings) -> Option<(f64, f64)> {
    let left: f64 = layers.iter().map(|(ly, _)| 1.0 - ly.f * ly.op).product();
    let last = layers.len().checked_sub(1)?;
    (em.behind_share > 0.0 && left > 0.02).then(|| (layers[last].0.t, taus[last]))
}

/// One pulse's echoes against one way its returns come back ([`cell_ll`]).
struct Pulse<'a> {
    echoes: &'a [f32],
    refls: &'a [f32],
    capped: bool,
    t_end: f64,
    em: &'a Settings,
    bg: &'a RayFoliage,
}

impl Pulse<'_> {
    /// The chance of echoes `i..` given returns `rets[j..]` not yet accounted for (research 130 step 3c: every echo
    /// is scored, not the first alone). Each echo is one of those returns (the ones before it not recorded), an echo
    /// of the foliage field (the returns before it not recorded) or an outlier.
    /// - **The field's echoes** are a Poisson process along the ray in the beam the echoes so far left, `seen`: each
    ///   echo's reflectance against a full hit is the share it took, as the field reads them ([`super::foliage`]).
    ///   That beam is read from the echoes alone, the same for every hypothesis, so how often a pulse echoes again
    ///   doesn't lean on the declared background's density along the rest of the ray (on the stand's checks, the
    ///   scene's beam there favoured a wider stick wherever a pulse returned nothing more: research 130 step 3c).
    /// - **The rest of the beam:** the field's echo reads `10 log10` of the beam the scene leaves, `scene` (the share
    ///   the returns matched so far leave: [`Return`]'s `beam`) × `taken` (what the field's own echoes left), dimmer:
    ///   behind an edge of fill f, `1 − f`, as a located edge reads f.
    /// - **Behind an occluder:** the returns behind a field's echo are met with the chance `taken` (as an element's
    ///   opacity is a chance): one that read as a full hit leaves nothing for them to return, and its pulse is no
    ///   evidence against them.
    ///
    /// The path after an echo restarts at its range plus the receiver's resolution (`tau0`, the optical depth there).
    /// After the last echo, the returns left and the field's path to the beam's end record nothing, unless the pulse
    /// is capped: a capped pulse says nothing past its last echo. The first echo reads as before step 3c (the whole
    /// beam, from the origin).
    #[allow(clippy::too_many_arguments)]
    fn lik(&self, rets: &[Return], i: usize, j: usize, scene: f64, taken: f64, seen: f64, tau0: f64) -> f64 {
        let em = self.em;
        if i == self.echoes.len() {
            if self.capped {
                return 1.0;
            }
            return rets[j..].iter().map(|r| 1.0 - r.s * taken).product::<f64>() * (-seen * (self.bg.tau_end - tau0).max(0.0)).exp();
        }
        let (o, x) = (self.echoes[i] as f64, self.refls.get(i).map_or(f64::NAN, |&x| x as f64));
        let [tau_o, sig_o, tau_after] = self.bg.echoes[i];
        let left = if x.is_finite() { (1.0 - 10f64.powf((x - em.refl_full_hit()) / 10.0)).clamp(0.0, 1.0) } else { 1.0 };
        let mut total = 0.0;
        // one of the returns
        let mut none = 1.0;
        for (k, r) in rets.iter().enumerate().skip(j) {
            let d = (-seen * (r.tau - tau0).max(0.0)).exp() * none * r.s * taken * t_pdf((o - r.mu) / r.sd, em.noise_dof) / r.sd * r.refl_density(x, em);
            if d > 0.0 {
                total += (1.0 - P_OUTLIER) * d * self.lik(rets, i + 1, k + 1, r.beam, taken, seen * left, tau_after);
            }
            none *= 1.0 - r.s * taken;
        }
        // the foliage field's
        if o < self.t_end && seen > 0.0 {
            let jn = (j..rets.len()).find(|&k| rets[k].t >= o).unwrap_or(rets.len());
            let none: f64 = rets[j..jn].iter().map(|r| 1.0 - r.s * taken).product();
            let d = seen * sig_o * (-seen * (tau_o - tau0).max(0.0)).exp() * none * em.refl_background(x - 10.0 * (scene * taken).max(1e-6).log10());
            if d > 0.0 {
                total += (1.0 - P_OUTLIER) * d * self.lik(rets, i + 1, jn, scene, taken * left, seen * left, tau_after);
            }
        }
        // an outlier
        total + P_OUTLIER / T_MAX * em.refl_background(x) * self.lik(rets, i + 1, j, scene, taken, seen, tau0)
    }
}

/// One cell an element fronts, as its ray likelihood scores it: its ray through the foliage field, its pulse's echoes
/// and their reflectances (dB; empty unknown), whether the pulse is at the instrument's cap, layers, ground and weight,
/// its log-likelihood under the empty scene, and what its first echo says about the element.
struct CellData {
    bg: RayFoliage,
    echoes: Vec<f32>,
    refls: Vec<f32>,
    capped: bool,
    anchor: Option<(f64, f64)>,
    layers: Vec<(Layer, u8)>,
    ground: Option<(f64, f64)>,
    wt: f64,
    base: f64,
    outcome: Outcome,
}

/// One element's data in one scan: the cells it owns (its evidence: [`elem_data`]), and its latent nodes: its
/// alignment nodes with their weights (its alignment shift is `Σ w z`, each hat's weight times its surface class's
/// along-beam spread), and the prior sds of its own two nodes, the stem front shift (m; 0 when off) and the
/// reflectance level (dB; 0 while reflectance isn't read).
struct ElemData {
    elem: usize,
    cells: Vec<CellData>,
    class: u8,
    /// Cells whose first echo is its.
    cells_on: u32,
    /// The least and greatest of those cells' rays' place across its axis, in radii (clamped to ±1; a stick's).
    on_across: [f64; 2],
    across: f64,
    align: Vec<([i64; 3], f64)>,
    own_sd: [f64; 2],
    /// Class evidence's cost in this scan, from its photometry over its echoes in cells it mostly fills.
    class_cost: f64,
}

impl ElemData {
    /// Its ray likelihood at `x` = (δ, reflectance level), in total and by [`Outcome`] (in its order).
    fn ll_parts(&self, x: (f64, f64), em: &Settings) -> (f64, [f64; 4]) {
        let (mut total, mut parts) = (0.0, [0.0; 4]);
        for c in &self.cells {
            let l = c.wt * (cell_ll(&c.echoes, &c.refls, c.capped, &c.layers, c.ground, x.0, x.1, em, &c.bg, c.anchor) - c.base);
            total += l;
            parts[c.outcome as usize] += l;
        }
        (total, parts)
    }

    fn ll(&self, x: (f64, f64), em: &Settings) -> f64 {
        self.ll_parts(x, em).0
    }

    /// (δ, reflectance level) at alignment shift `a` and own nodes `y`.
    fn at(&self, a: f64, y: [f64; 2]) -> (f64, f64) {
        (a + self.own_sd[0] * y[0], self.own_sd[1] * y[1])
    }

    /// −ll's slope and curvature in (δ, level) at `x`, by central differences (3e-4 m, 0.05 dB; the cross term
    /// forward), the curvature's negative directions dropped. Without a level node only δ's are filled.
    fn quad(&self, x: (f64, f64), em: &Settings) -> ([f64; 2], [[f64; 2]; 2]) {
        let (h, hb) = (3e-4, 0.05);
        let f = |d: f64, b: f64| -self.ll((x.0 + d, x.1 + b), em);
        let (f0, fp, fm) = (f(0.0, 0.0), f(h, 0.0), f(-h, 0.0));
        let mut g = [(fp - fm) / (2.0 * h), 0.0];
        let (a, mut b, mut c) = ((fp - 2.0 * f0 + fm) / (h * h), 0.0, 0.0);
        if self.own_sd[1] > 0.0 {
            let (bp, bm, pp) = (f(0.0, hb), f(0.0, -hb), f(h, hb));
            g[1] = (bp - bm) / (2.0 * hb);
            c = (bp - 2.0 * f0 + bm) / (hb * hb);
            b = (pp - fp - bp + f0) / (h * hb);
        }
        // the 2 × 2 curvature's eigen-decomposition, negative eigenvalues set to 0
        let (m, d) = (0.5 * (a + c), (0.25 * (a - c).powi(2) + b * b).sqrt());
        let (l1, l2) = (m + d, m - d);
        let hess = if l2 >= 0.0 {
            [[a, b], [b, c]]
        } else if l1 <= 0.0 {
            [[0.0; 2]; 2]
        } else {
            // the positive eigenvalue's direction
            let v = if b.abs() > 1e-300 { [b, l1 - a] } else if a >= c { [1.0, 0.0] } else { [0.0, 1.0] };
            let n2 = v[0] * v[0] + v[1] * v[1];
            [[l1 * v[0] * v[0] / n2, l1 * v[0] * v[1] / n2], [l1 * v[0] * v[1] / n2, l1 * v[1] * v[1] / n2]]
        };
        (g, hess)
    }

    /// Its term at alignment shift `a` (m): its own nodes y at their mode given `a` (Newton from 0, a step at most 3
    /// prior sds, halved until the cost falls, stopping under a micronat), the Laplace step over them, and its cost's
    /// slope and curvature in `a` with them eliminated (the Schur complement of its curvature over (a, y)).
    fn term(&self, a: f64, em: &Settings) -> Term {
        let s = self.own_sd;
        let cost = |y: [f64; 2]| -self.ll(self.at(a, y), em) + 0.5 * (y[0] * y[0] + y[1] * y[1]);
        // the curvature over y, P = diag(s) Q diag(s) + I, its inverse and determinant; the slope over y
        let own = |q: [[f64; 2]; 2]| {
            let p = [[s[0] * s[0] * q[0][0] + 1.0, s[0] * s[1] * q[0][1]], [s[0] * s[1] * q[0][1], s[1] * s[1] * q[1][1] + 1.0]];
            let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
            ([[p[1][1] / det, -p[0][1] / det], [-p[1][0] / det, p[0][0] / det]], det)
        };
        let slope_y = |g: [f64; 2], y: [f64; 2]| [s[0] * g[0] + y[0], s[1] * g[1] + y[1]];
        let mul = |m: [[f64; 2]; 2], x: [f64; 2]| [m[0][0] * x[0] + m[0][1] * x[1], m[1][0] * x[0] + m[1][1] * x[1]];
        let mut y = [0.0; 2];
        let (mut g, mut q) = self.quad(self.at(a, y), em);
        if s != [0.0; 2] {
            let mut f_now = cost(y);
            for _ in 0..MAX_NEWTON {
                let mut step = mul(own(q).0, slope_y(g, y));
                step = step.map(|x| x / (step[0].abs().max(step[1].abs()).max(3.0) / 3.0));
                let (mut t, mut fall) = (1.0, None);
                while t >= 1e-3 {
                    let yn = [y[0] - t * step[0], y[1] - t * step[1]];
                    let f_new = cost(yn);
                    if f_new <= f_now {
                        fall = Some(f_now - f_new);
                        (y, f_now) = (yn, f_new);
                        break;
                    }
                    t *= 0.5;
                }
                let Some(fall) = fall else { break };
                // the slope and curvature are always the final mode's
                (g, q) = self.quad(self.at(a, y), em);
                if fall < 1e-6 {
                    break;
                }
            }
        }
        // over (a, y): M = Jᵀ Q J + diag(0, I), J = [[1, s0, 0], [0, 0, s1]]; its a–y block is `link`
        let (pi, det) = own(q);
        let link = [q[0][0] * s[0], q[0][1] * s[1]];
        let pl = mul(pi, link);
        let gy = slope_y(g, y);
        let (ll, parts) = self.ll_parts(self.at(a, y), em);
        Term {
            align: self.align.clone(),
            fit: -ll + 0.5 * (y[0] * y[0] + y[1] * y[1]),
            logdet: 0.5 * det.ln(),
            class: self.class_cost,
            slope: g[0] - pl[0] * gy[0] - pl[1] * gy[1],
            curv: q[0][0] - link[0] * pl[0] - link[1] * pl[1],
            own: y,
            own_sd: s,
            own_cov: pi,
            own_link: pl,
            ray: parts.map(|x| -x),
            cells: self.cells.len() as u32,
            cells_on: self.cells_on,
            arc: match self.on_across {
                [lo, hi] if lo <= hi => hi.asin() - lo.asin(),
                _ => 0.0,
            },
            across: self.across,
            surface: self.class,
        }
    }
}

/// One (element, scan)'s part of its scan's cost at an alignment shift, `fit + logdet + class`: `fit` its ray
/// likelihood's cost plus its own nodes' prior at their mode ŷ, `logdet` half their Laplace log-determinant ln|P|,
/// `class` class evidence's cost. With them, the cost's `slope` and `curv`ature in its alignment shift (its own nodes
/// eliminated), and what its row reports: ŷ, its own nodes' covariance given the shift P⁻¹ and how they follow it
/// (`own_link`: ŷ moves by −`own_link` per unit of shift), the ray likelihood's cost by outcome, and its cells.
#[derive(Debug, Clone)]
struct Term {
    align: Vec<([i64; 3], f64)>,
    fit: f64,
    logdet: f64,
    class: f64,
    slope: f64,
    curv: f64,
    own: [f64; 2],
    own_sd: [f64; 2],
    own_cov: [[f64; 2]; 2],
    own_link: [f64; 2],
    ray: [f64; 4],
    cells: u32,
    cells_on: u32,
    arc: f64,
    across: f64,
    surface: u8,
}

impl Term {
    fn cost(&self) -> f64 {
        self.fit + self.logdet + self.class
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

/// One scan's data for the elements `score` (indices into `elems`, each with its layers in `rendered`), against the
/// k-buffer of every element's layers, under the surface classes `cls`. A cell's evidence belongs to its front
/// element.
#[allow(clippy::too_many_arguments)]
fn elem_data(v: &RangeImage, rendered: &[Vec<(usize, Layer)>], elems: &[Element], cls: &[u8], score: &[usize], ref_epoch: u32, em: &Settings,
             sc: &Background, ext: &Foliage) -> Vec<ElemData> {
    let buf = kbuffer(rendered);
    let other = v.epoch != ref_epoch;
    let level = if em.refl_noise > 0.0 && !v.echo_refl.is_empty() { em.refl_level_sd } else { 0.0 };
    let mut data: Vec<ElemData> = Vec::with_capacity(score.len());
    let mut by_elem: HashMap<u32, usize> = HashMap::new();
    for &i in score {
        let (e, k) = (&elems[i], cls[i]);
        let mid = e.mid();
        let s_along = em.align_sd(k, other);
        by_elem.insert(i as u32, data.len());
        data.push(ElemData { elem: i, cells: Vec::new(), class: k, cells_on: 0, on_across: [f64::INFINITY, f64::NEG_INFINITY], across: 2.0 * e.r / em.footprint(norm(&sub(&mid, &v.origin))),
                             align: hats(&mid, ALIGN_SPACING).into_iter().map(|(key, w)| (key, w * s_along)).collect(),
                             own_sd: [em.front_shift_sd(e.r), level], class_cost: 0.0 });
    }
    let mut photo = vec![[0.0; 5]; data.len()];
    // cells in index order, so an element's sums run in the same order in every run (the k-buffer is a hash map)
    let mut order: Vec<usize> = buf.keys().copied().collect();
    order.sort_unstable();
    for cell in order {
        let ls = &buf[&cell];
        let front = ls[0];
        let Some(&d) = by_elem.get(&front.elem) else { continue };
        if v.cell[cell] == Cell::NoPulse {
            continue; // no pulse was placed here: nothing observed, neither echo nor miss
        }
        let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[l.elem as usize])).collect();
        let ts: Vec<f64> = layers.iter().map(|(l, _)| l.t).collect();
        let cv = cell_view(v, cell, &ts, em, sc, ext);
        let fe = front_echo(&layers, cv.ground, em, &cv.bg.taus, elems[front.elem as usize].r, other);
        let outcome = Outcome::of(cv.obs, fe.mu, fe.sd);
        let el = &mut data[d];
        el.cells_on += (outcome == Outcome::On) as u32;
        if let (Outcome::On, Form::Stick { .. }) = (outcome, elems[front.elem as usize].form) {
            let z = (front.x / elems[front.elem as usize].r).clamp(-1.0, 1.0);
            el.on_across = [el.on_across[0].min(z), el.on_across[1].max(z)];
        }
        let echoes = v.echoes(cell);
        if let (true, true, Some(&db)) = (outcome == Outcome::On, front.f > MOSTLY, v.echo_refls(cell).first()) {
            let c = if v.cos_inc[cell].is_finite() { (v.cos_inc[cell] as f64).clamp(0.05, 1.0) } else { 0.5 };
            let db = db as f64;
            for (a, x) in photo[d].iter_mut().zip([1.0, db, db - 10.0 * c.log10(), (echoes.len() > 1) as u8 as f64, c]) {
                *a += x;
            }
        }
        let (refls, capped) = (v.echo_refls(cell), !echoes.is_empty() && echoes.len() >= v.max_targets);
        let anchor = behind_anchor(&layers, &cv.bg.taus, em);
        let base = cell_ll(echoes, refls, capped, &[], cv.ground, 0.0, 0.0, em, &cv.bg.base(), anchor);
        el.cells.push(CellData { bg: cv.bg, echoes: echoes.to_vec(), refls: refls.to_vec(), capped, anchor, layers, ground: cv.ground, wt: cv.wt, base, outcome });
    }
    // class evidence: the element's form is its class hypothesis (a stick wood, a patch leaf)
    for (el, ph) in data.iter_mut().zip(&photo) {
        let llr = em.class_llr(ph);
        el.class_cost = match elems[el.elem].form { Form::Stick { .. } => -llr, Form::Patch { .. } => llr };
    }
    data
}

/// The scan alignment error held at its mode (research 130 offset-field-design.md §3, step 3a): the nodes' mode ẑ
/// and covariance Σ₀ = Λ₀⁻¹ at the last refit, `m` = Σ₀ G₀ (the step Newton would still take, G₀ the cost's slope in
/// the nodes there) and `gm` = G₀ᵀ Σ₀ G₀, and the terms the refit was taken with (`reference`, by element id).
///
/// With its own nodes eliminated, each element meets the nodes through its alignment shift alone: slope γ and
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
    reference: Vec<Option<Term>>,
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

/// A scan's exact Laplace marginal over the alignment error's nodes, given each element's data (the referee for the
/// held alignment, and its refit): Newton on the nodes with each element's own nodes profiled out at every step
/// ([`ElemData::term`]); a step at most 3 prior sds, halved until the cost falls, stopping under a millinat; the
/// precision is the final mode's. Its value is the held alignment's at its own reference ([`Held::part`] with no
/// columns).
fn solve_alignment(data: &[ElemData], em: &Settings) -> (Held, Vec<Term>) {
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    for el in data {
        for (k, _) in &el.align {
            let n = index.len();
            index.entry(*k).or_insert(n);
        }
    }
    let n = index.len();
    let terms_at = |z: &[f64]| -> Vec<Term> { data.par_iter().map(|el| el.term(el.align.iter().map(|(k, w)| w * z[index[k]]).sum(), em)).collect() };
    let objective = |z: &[f64], t: &[Term]| 0.5 * z.iter().map(|x| x * x).sum::<f64>() + t.iter().map(|t| t.fit).sum::<f64>();
    let assemble = |z: &[f64], t: &[Term]| {
        let mut grad = nalgebra::DVector::from_column_slice(z);
        let mut hess = nalgebra::DMatrix::<f64>::identity(n, n);
        for t in t {
            for &(a, wa) in &t.align {
                let i = index[&a];
                grad[i] += t.slope * wa;
                for &(b, wb) in &t.align {
                    hess[(i, index[&b])] += t.curv * wa * wb;
                }
            }
        }
        (grad, hess)
    };
    let mut z = vec![0.0; n];
    let mut terms = terms_at(&z);
    let (mut grad, mut hess) = assemble(&z, &terms);
    for _ in 0..MAX_NEWTON {
        let Some(ch) = (n > 0).then(|| hess.clone().cholesky()).flatten() else { break };
        let mut step = ch.solve(&grad);
        step /= step.amax().max(3.0) / 3.0;
        let f_now = objective(&z, &terms);
        let (mut t, mut fall) = (1.0, None);
        while t >= 1e-3 {
            let zn: Vec<f64> = z.iter().zip(step.iter()).map(|(a, b)| a - t * b).collect();
            let tn = terms_at(&zn);
            let f_new = objective(&zn, &tn);
            if f_new <= f_now {
                fall = Some(f_now - f_new);
                (z, terms) = (zn, tn);
                break;
            }
            t *= 0.5;
        }
        let Some(fall) = fall else { break };
        (grad, hess) = assemble(&z, &terms);
        if fall < 1e-3 {
            break;
        }
    }
    // the precision is the unit prior plus positive semi-definite terms, so only a non-finite cost fails
    let (logdet, cov) = match hess.cholesky() {
        Some(c) => (2.0 * c.l().diagonal().iter().map(|x| x.ln()).sum::<f64>(), c.inverse()),
        None => (f64::NAN, nalgebra::DMatrix::zeros(n, n)),
    };
    let m: Vec<f64> = (&cov * &grad).iter().copied().collect();
    let gm = grad.iter().zip(&m).map(|(a, b)| a * b).sum();
    (Held { index, z, cov, m, gm, logdet, reference: Vec::new() }, terms)
}

/// One scan's fit: per element id its surface class and term (none where the scan doesn't see it), the held
/// alignment, and the scan's cost.
struct Fit {
    cls: Vec<u8>,
    terms: Vec<Option<Term>>,
    held: Held,
    cost: f64,
}

impl Fit {
    /// The scan's cost with the terms `over` (by id; none: the element left) in place of its own: every term's cost
    /// plus the held alignment's part ([`Held::part`]), its columns the terms that differ from the reference.
    fn total(&self, over: &HashMap<u32, Option<Term>>) -> f64 {
        let n = self.terms.len().max(self.held.reference.len()).max(over.keys().max().map_or(0, |&i| i as usize + 1));
        let mut cost = 0.0;
        let mut cols: Vec<(&[([i64; 3], f64)], f64, f64)> = Vec::new();
        for i in 0..n {
            let now = match over.get(&(i as u32)) {
                Some(t) => t.as_ref(),
                None => self.terms.get(i).and_then(|t| t.as_ref()),
            };
            let was = self.held.reference.get(i).and_then(|t| t.as_ref());
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

/// How well one scan pins an element's geometry down ([`Scene::information`]): in its parameters (n of them), the
/// cost's slope (n) and observed curvature (n × n, row-major), the expected information (n × n) and the gradient's
/// variance were the cells independent (n × n).
#[derive(Debug, Clone)]
pub struct Information {
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
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
///   it touches. So a move recomputes only the terms of the elements whose cells it touches (and of those whose cells
///   a changed surface class touches), not the scan, and removing an element restores the cost exactly (synthesis
///   §18 invariant 7). [`Scene::refit_alignment`] re-solves the alignment (a move like `refit_foliage`), after which
///   the cost is the exact Laplace marginal ([`Scene::exact_cost`], the referee); the scene is built refitted.
/// - Element ids are stable and never reused.
/// - The reference epoch (the scanning day the alignment error's spread is stated for) is declared: a scan of another epoch adds the other-epoch part
///   to its alignment error.
pub struct Scene {
    views: Vec<Arc<RangeImage>>,
    em: Settings,
    sc: Background,
    /// The foliage field (a residual: estimated from the rays no element fronts) and every ray's counts.
    ext: Foliage,
    ext_all: Counts,
    ref_epoch: u32,
    elems: Vec<Option<Element>>,
    /// [image][id]: the element's layers (empty when removed or unseen).
    rendered: Vec<Vec<Vec<(usize, Layer)>>>,
    /// [image]: per cell, the elements whose layers front it: (id, range).
    cells: Vec<HashMap<usize, Vec<(u32, f64)>>>,
    /// [image]: its fit.
    fits: Vec<Fit>,
}

/// Enter an element's fronting layers in one image's cell index, or take them out.
fn index(cells: &mut HashMap<usize, Vec<(u32, f64)>>, id: u32, layers: &[(usize, Layer)], enter: bool) {
    for (c, l) in layers.iter().filter(|(_, l)| l.fronts()) {
        let e = cells.entry(*c).or_default();
        if enter {
            e.push((id, l.t));
        } else {
            e.retain(|x| x.0 != id);
        }
    }
}

impl Scene {
    /// A scene of `elems` against `views`, the alignment error's spread stated for the scans of epoch `ref_epoch`; with `spec`, the foliage
    /// field is estimated in its box as the residual of the elements, otherwise it is the background's κ.
    pub fn new(views: Vec<Arc<RangeImage>>, elems: &[Element], em: Settings, sc: Background, spec: Option<FoliageBox>, ref_epoch: u32) -> Self {
        let n = views.len();
        let ext_all = spec.map_or_else(Counts::new, |sp| {
            views.iter().map(|v| foliage::counts(v, &sp, &em, None)).fold(Counts::new(), |mut a, b| {
                for (k, (h, l)) in b {
                    let e = a.entry(k).or_insert((0.0, 0.0));
                    e.0 += h;
                    e.1 += l;
                }
                a
            })
        });
        let ext = spec.map_or_else(|| Foliage::uniform(sc.kappa), |sp| Foliage::from_counts(sc.kappa, sp, &ext_all));
        let mut c = Scene { views, em, sc, ext, ext_all, ref_epoch, elems: elems.iter().map(|e| Some(*e)).collect(),
                             rendered: Vec::new(), cells: vec![HashMap::new(); n], fits: Vec::new() };
        c.rendered = c.views.par_iter().map(|v| render_all(v, elems, &c.em)).collect();
        for (cells, rendered) in c.cells.iter_mut().zip(&c.rendered) {
            for (i, ls) in rendered.iter().enumerate() {
                index(cells, i as u32, ls, true);
            }
        }
        c.ext = c.residual();
        c.fits = (0..n).into_par_iter().map(|s| c.solve(s)).collect();
        c
    }

    /// The scene's cost against the empty scene: its elements' cells, each against the same cell with the foliage
    /// field alone (the rays no element fronts are the field's in both, and cancel).
    pub fn cost(&self) -> f64 {
        self.fits.iter().map(|f| f.cost).sum()
    }

    /// The scene's exact cost: each scan's Laplace marginal over the alignment error re-solved for the elements as
    /// they are (the referee for the held alignment; the scene unchanged).
    pub fn exact_cost(&self) -> f64 {
        (0..self.views.len()).into_par_iter().map(|s| self.solve(s).cost).collect::<Vec<f64>>().iter().sum()
    }

    /// Re-solve the scan alignment error for the elements as they are (a move: its cost change); after it the cost
    /// is [`Scene::exact_cost`], and moves hold the new mode and precision.
    pub fn refit_alignment(&mut self) -> f64 {
        let before = self.cost();
        self.fits = (0..self.views.len()).into_par_iter().map(|s| self.solve(s)).collect();
        self.cost() - before
    }

    /// The foliage field estimated from the rays no element fronts.
    fn residual(&self) -> Foliage {
        let Some(sp) = self.ext.spec else { return self.ext.clone() };
        let mut fronted = Counts::new();
        for (s, v) in self.views.iter().enumerate() {
            let cells: HashSet<usize> = self.rendered[s].iter().flatten().filter(|(_, l)| l.fronts()).map(|&(c, _)| c).collect();
            for (k, (h, l)) in foliage::counts(v, &sp, &self.em, Some(&cells)) {
                let e = fronted.entry(k).or_insert((0.0, 0.0));
                e.0 += h;
                e.1 += l;
            }
        }
        Foliage::from_counts(self.sc.kappa, sp, &foliage::minus(&self.ext_all, &fronted))
    }

    /// Re-estimate the foliage field from the rays the elements leave, and refit the alignment (a move: its cost
    /// change). Moves that change elements hold the field as it is; after this one the scene's cost is a fresh scene's.
    pub fn refit_foliage(&mut self) -> f64 {
        if self.ext.spec.is_none() || self.ext_all.is_empty() {
            return 0.0;
        }
        self.ext = self.residual();
        self.refit_alignment()
    }

    /// Hold a foliage field estimated elsewhere (another scene's, from other scans) fixed: `refit_foliage` then
    /// leaves it. The cost change.
    pub fn hold_foliage(&mut self, ext: &Foliage) -> f64 {
        self.ext = ext.clone();
        self.ext_all.clear();
        self.refit_alignment()
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

    /// The foliage field and its counts: per voxel with data, (voxel, σ, residual counts, every ray's counts).
    pub fn foliage(&self) -> (Option<FoliageBox>, Vec<([i32; 3], f64, (f64, f64))>) {
        let mut out: Vec<([i32; 3], f64, (f64, f64))> = self.ext_all.iter().map(|(&k, &c)| (k, self.ext.sigma(&k), c)).collect();
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
                Some((a, b)) => (self.ext.tau(&v.origin, &w, b) - self.ext.tau(&v.origin, &w, a), a, b),
                None => (f64::NAN, f64::NAN, f64::NAN),
            }
        }).collect()
    }

    /// A row per (element, scan), by element id: its term, and an equal share of its scan's alignment part.
    pub fn rows(&self) -> Vec<ElementScan> {
        let mut rows = Vec::new();
        for (s, fit) in self.fits.iter().enumerate() {
            let terms: Vec<(usize, &Term)> = fit.terms.iter().enumerate().filter_map(|(i, t)| t.as_ref().map(|t| (i, t))).collect();
            let share = (fit.cost - terms.iter().map(|(_, t)| t.cost()).sum::<f64>()) / terms.len().max(1) as f64;
            for (i, t) in terms {
                // the shift's posterior from the held alignment, and the own nodes' with it marginalised
                let var = fit.held.between(&t.align, &t.align);
                let own_sd = |j: usize| t.own_sd[j] * (t.own_cov[j][j] + t.own_link[j] * t.own_link[j] * var).max(0.0).sqrt();
                rows.push(ElementScan {
                    elem: i as u32, scan: s as u32, cells: t.cells, cells_on: t.cells_on, arc: t.arc, footprints_across: t.across, surface: t.surface,
                    cost: t.cost() + share, ray_on: t.ray[0], ray_empty: t.ray[1], ray_in_front: t.ray[2], ray_behind: t.ray[3], class: t.class,
                    alignment: t.fit - t.ray.iter().sum::<f64>() + t.logdet + share, discount: self.em.discount,
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
        let changes: Vec<Vec<(u32, Option<Term>, u8)>> = (0..self.views.len()).into_par_iter()
            .map(|s| if layers[s].is_empty() { Vec::new() } else { self.reterm(s, None, &[(e, layers[s].clone())], id) }).collect();
        self.elems.push(Some(e));
        let mut d = 0.0;
        for (s, (l, ch)) in layers.into_iter().zip(changes).enumerate() {
            index(&mut self.cells[s], id, &l, true);
            self.rendered[s].push(l);
            d += self.apply(s, ch);
        }
        (id, d)
    }

    /// Remove a live element ([`Scene::check_id`]): the cost change.
    pub fn remove(&mut self, id: u32) -> f64 {
        let first = self.elems.len() as u32;
        let changes: Vec<Vec<(u32, Option<Term>, u8)>> = (0..self.views.len()).into_par_iter()
            .map(|s| if self.rendered[s][id as usize].is_empty() { Vec::new() } else { self.reterm(s, Some(id), &[], first) }).collect();
        self.elems[id as usize] = None;
        let mut d = 0.0;
        for (s, ch) in changes.into_iter().enumerate() {
            let old = std::mem::take(&mut self.rendered[s][id as usize]);
            index(&mut self.cells[s], id, &old, false);
            d += self.apply(s, ch);
        }
        d
    }

    /// Replace a live element ([`Scene::check_id`]): the cost change.
    pub fn change(&mut self, id: u32, e: Element) -> f64 {
        let em = self.em;
        let layers: Vec<Vec<(usize, Layer)>> = self.views.par_iter().map(|v| render(v, &e, id, &em)).collect();
        let changes: Vec<Vec<(u32, Option<Term>, u8)>> = (0..self.views.len()).into_par_iter().map(|s| {
            if layers[s].is_empty() && self.rendered[s][id as usize].is_empty() { Vec::new() } else { self.reterm(s, Some(id), &[(e, layers[s].clone())], id) }
        }).collect();
        self.elems[id as usize] = Some(e);
        let mut d = 0.0;
        for (s, (l, ch)) in layers.into_iter().zip(changes).enumerate() {
            let old = std::mem::replace(&mut self.rendered[s][id as usize], l);
            index(&mut self.cells[s], id, &old, false);
            index(&mut self.cells[s], id, &self.rendered[s][id as usize], true);
            d += self.apply(s, ch);
        }
        d
    }

    /// Put a move's new terms and classes into scan `s`'s fit: the cost change.
    fn apply(&mut self, s: usize, changes: Vec<(u32, Option<Term>, u8)>) -> f64 {
        if changes.is_empty() {
            return 0.0;
        }
        let fit = &mut self.fits[s];
        for (id, t, k) in changes {
            let i = id as usize;
            if fit.terms.len() <= i {
                fit.terms.resize(i + 1, None);
                fit.cls.resize(i + 1, 1);
            }
            (fit.terms[i], fit.cls[i]) = (t, k);
        }
        let before = fit.cost;
        fit.cost = fit.total(&HashMap::new());
        fit.cost - before
    }

    /// Scan `s` solved afresh for the elements as they are ([`solve_alignment`]), its alignment held there.
    fn solve(&self, s: usize) -> Fit {
        let v = &*self.views[s];
        let ids: Vec<u32> = (0..self.elems.len() as u32).filter(|&i| self.elems[i as usize].is_some() && !self.rendered[s][i as usize].is_empty()).collect();
        let elems: Vec<Element> = ids.iter().map(|&i| self.elems[i as usize].unwrap()).collect();
        let rendered: Vec<Vec<(usize, Layer)>> = ids.iter().enumerate()
            .map(|(k, &i)| self.rendered[s][i as usize].iter().map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect()).collect();
        let cls = classes(v, &kbuffer(&rendered), &elems, &self.em, &self.sc, v.epoch != self.ref_epoch);
        let data = elem_data(v, &rendered, &elems, &cls, &(0..ids.len()).collect::<Vec<_>>(), self.ref_epoch, &self.em, &self.sc, &self.ext);
        let (mut held, terms) = solve_alignment(&data, &self.em);
        let n = self.elems.len();
        let (mut fit_terms, mut fit_cls) = (vec![None; n], vec![1u8; n]);
        for (el, t) in data.iter().zip(terms) {
            fit_terms[ids[el.elem] as usize] = Some(t);
        }
        for (k, &i) in ids.iter().enumerate() {
            fit_cls[i as usize] = cls[k];
        }
        held.reference = fit_terms.clone();
        let mut fit = Fit { cls: fit_cls, terms: fit_terms, held, cost: 0.0 };
        fit.cost = fit.total(&HashMap::new());
        fit
    }

    /// Scan `s` with the live element `without` (if any) taken out and the elements `with` (each with its layers in
    /// the scan) put in as ids `first`, `first + 1`, …: the new term and surface class of every element whose term
    /// changes, by id (none: it left the scan), the alignment held. An element's term reads only the cells it is the
    /// front of, each with its layers and their elements' classes, and a class reads the cells its element is the
    /// front of. So the terms that change are those of the fronts of the cells the move touches (before it and after),
    /// and of the fronts of the cells of an element whose class changed.
    fn reterm(&self, s: usize, without: Option<u32>, with: &[(Element, Vec<(usize, Layer)>)], first: u32) -> Vec<(u32, Option<Term>, u8)> {
        let (v, em, fit, index) = (&*self.views[s], &self.em, &self.fits[s], &self.cells[s]);
        let added = |i: u32| i >= first && ((i - first) as usize) < with.len();
        let layers = |i: u32| -> &[(usize, Layer)] { if added(i) { &with[(i - first) as usize].1 } else { &self.rendered[s][i as usize] } };
        let element = |i: u32| if added(i) { with[(i - first) as usize].0 } else { self.elems[i as usize].unwrap() };
        let present = |i: u32| !layers(i).is_empty() && (added(i) || Some(i) != without);
        let fronting = |i: u32| layers(i).iter().filter(|(_, l)| l.fronts()).map(|(c, _)| *c);
        let mut fresh: HashMap<usize, Vec<(u32, f64)>> = HashMap::new();
        for (k, (_, ls)) in with.iter().enumerate() {
            for (c, l) in ls.iter().filter(|(_, l)| l.fronts()) {
                fresh.entry(*c).or_default().push((first + k as u32, l.t));
            }
        }
        // a cell's k-buffer, before the move and after it: its elements, nearest first
        let nearest = |mut e: Vec<(u32, f64)>| -> Vec<u32> {
            e.sort_by(|a, b| a.1.total_cmp(&b.1));
            e.truncate(K);
            e.into_iter().map(|x| x.0).collect()
        };
        let before = |c: usize| nearest(index.get(&c).cloned().unwrap_or_default());
        let after = |c: usize| nearest(index.get(&c).into_iter().flatten().filter(|x| Some(x.0) != without && !added(x.0))
            .chain(fresh.get(&c).into_iter().flatten()).copied().collect());
        // the elements `ids` and every layer in the cells they front, numbered from 0 (the context after them)
        let local = |ids: &BTreeSet<u32>| -> (Vec<u32>, Vec<Element>, Vec<Vec<(usize, Layer)>>) {
            let mut cells: Vec<usize> = ids.iter().flat_map(|&i| fronting(i)).collect();
            cells.sort_unstable();
            cells.dedup();
            let mut all: Vec<u32> = ids.iter().copied().collect();
            let mut seen: HashSet<u32> = ids.iter().copied().collect();
            for &c in &cells {
                all.extend(after(c).into_iter().filter(|&i| seen.insert(i)));
            }
            let cells: HashSet<usize> = cells.into_iter().collect();
            let rendered = all.iter().enumerate()
                .map(|(k, &i)| layers(i).iter().filter(|(c, _)| cells.contains(c)).map(|&(c, l)| (c, Layer { elem: k as u32, ..l })).collect()).collect();
            (all.clone(), all.iter().map(|&i| element(i)).collect(), rendered)
        };
        let stored = |i: u32| fit.cls.get(i as usize).copied().unwrap_or(1);
        let other = v.epoch != self.ref_epoch;
        // the fronts of the cells the move touches, and the elements it adds
        let touched: HashSet<usize> = without.into_iter().flat_map(|w| self.rendered[s][w as usize].iter()).filter(|(_, l)| l.fronts()).map(|(c, _)| *c)
            .chain(fresh.keys().copied()).collect();
        let mut near: BTreeSet<u32> = (0..with.len() as u32).map(|k| first + k).collect();
        for &c in &touched {
            near.extend(before(c).first());
            near.extend(after(c).first());
        }
        near.retain(|&i| present(i));
        let (ids, el, r) = local(&near);
        let cls_near = classes(v, &kbuffer(&r), &el, em, &self.sc, other);
        let cls: HashMap<u32, u8> = ids.iter().zip(&cls_near).filter(|(i, _)| near.contains(i)).map(|(&i, &k)| (i, k)).collect();
        // and the fronts of the cells of an element whose class changed
        let mut score = near.clone();
        for &i in &near {
            if added(i) || cls[&i] != stored(i) {
                for c in fronting(i) {
                    score.extend(after(c).first());
                }
            }
        }
        score.retain(|&i| present(i));
        let (ids, el, r) = local(&score);
        let cls: Vec<u8> = ids.iter().map(|&i| cls.get(&i).copied().unwrap_or_else(|| stored(i))).collect();
        let data = elem_data(v, &r, &el, &cls, &(0..score.len()).collect::<Vec<_>>(), self.ref_epoch, em, &self.sc, &self.ext);
        let mut out: Vec<(u32, Option<Term>, u8)> = data.par_iter().map(|d| (ids[d.elem], Some(d.term(fit.held.shift(&d.align), em)), cls[d.elem])).collect();
        out.extend(without.into_iter().chain((0..with.len() as u32).map(|k| first + k)).filter(|&i| !present(i)).map(|i| (i, None, 1)));
        out
    }

    /// What an element predicts in image `s` ([`Scene::check_scan`]) against the scene, per cell it fronts, on the
    /// path the likelihood scores ([`front_echo`], [`cell_view`]): it is rendered for this, not added, and only the
    /// scene's layers in its cells enter its k-buffer, so its front cells and class are those of the whole set.
    pub fn predict(&self, e: Element, s: usize) -> Vec<CellPrediction> {
        self.predict_without(e, s, None)
    }

    /// [`Scene::predict`] against the scene with the live element `without` (if any) left out.
    fn predict_without(&self, e: Element, s: usize, without: Option<u32>) -> Vec<CellPrediction> {
        let (v, em) = (&*self.views[s], &self.em);
        let mine = render(v, &e, 0, em);
        let cells: HashSet<usize> = mine.iter().map(|&(c, _)| c).collect();
        let (mut els, mut per) = (vec![e], vec![mine]);
        for (i, el) in self.elems.iter().enumerate() {
            let Some(el) = el.as_ref().filter(|_| without != Some(i as u32)) else { continue };
            let k = per.len() as u32;
            let l: Vec<(usize, Layer)> = self.rendered[s][i].iter().filter(|(c, _)| cells.contains(c)).map(|&(c, ly)| (c, Layer { elem: k, ..ly })).collect();
            if !l.is_empty() {
                els.push(*el);
                per.push(l);
            }
        }
        let buf = kbuffer(&per);
        let other = v.epoch != self.ref_epoch;
        let cls = classes(v, &buf, &els, em, &self.sc, other);
        let mut out: Vec<CellPrediction> = buf.iter().filter(|(&cell, ls)| ls[0].elem == 0 && v.cell[cell] != Cell::NoPulse).map(|(&cell, ls)| {
            let ly = ls[0];
            let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[l.elem as usize])).collect();
            let ts: Vec<f64> = layers.iter().map(|(l, _)| l.t).collect();
            let cv = cell_view(v, cell, &ts, em, &self.sc, &self.ext);
            let fe = front_echo(&layers, cv.ground, em, &cv.bg.taus, e.r, other);
            CellPrediction {
                cell, elem: 0, fill: ly.f, recorded: fe.s, arrives: (-fe.tau).exp(), surface: cls[0], depth: fe.mu, sd: fe.sd, refl: if fe.db[1] > -200.0 { fe.db[1] } else { f64::NAN }, cos: ly.cos, obs: cv.obs,
                obs_refl: v.echo_refls(cell).first().map_or(f64::NAN, |&x| x as f64),
                outcome: Outcome::of(cv.obs, fe.mu, fe.sd), weight: cv.wt, footprint: em.footprint(ly.t), across: ly.x, along: ly.s,
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
            for p in self.predict(Element { mismatch: 0.0, ..e }, s) {
                let r = p.obs - sh - p.depth;
                if p.fill > MOSTLY && Outcome::of(p.obs - sh, p.depth, p.sd) == Outcome::On {
                    out.mismatch_ss += phi * p.weight * (r * r - p.sd * p.sd).max(0.0);
                    out.mismatch_w += phi * p.weight;
                    out.mismatch_cells += 1;
                }
            }
            for p in self.predict(e, s).into_iter().filter(|p| p.fill > 0.9) {
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

    /// How well the scans pin a live element's geometry down ([`Scene::check_id`]): per scan, the curvature of its
    /// cost (nats; the observed information under the evidence discount) in its parameters, with the rest of the
    /// scene held, by central differences about the element as it is. A stick's parameters are `ln r`, its axis moved
    /// across itself (two directions perpendicular to it, m) and tilted about its middle (the same two directions, rad
    /// at its ends' half-length); a patch's are `ln r` and its centre moved (x, y, z, m). The parameter frame is the
    /// element's own, the same in every scan, so scans' information adds. Steps: 0.05 in `ln r`, 2 mm in place (about
    /// the depth noise: wider steps read a heavy-tailed cost's secant) and that over the half-length in tilt. Per scan,
    /// the cost's slope (n) and curvature (n × n, row-major; n 5 for a stick, 4 for a patch), so `−H⁻¹ g` is how far
    /// the scan's best fit sits from the element; a scan that doesn't see it (or sees it only hidden) gives zeros.
    ///
    /// Beside it, the **expected information** (n × n; research 130 step 3a, synthesis §9 item 5, §16): what the
    /// element's own cells would tell on average, from what it predicts in them ([`Scene::predict`], the rest of the
    /// scene held) under the same steps. Each cell is read as an echo or none (a Bernoulli in the chance p it returns
    /// an echo) and, given one, its depth and reflectance (t densities of the noise's degrees of freedom ν, which carry
    /// `(ν + 1) / (ν + 3)` of a Gaussian's information): `Σ_c w_c [∂p ∂pᵀ / (p (1 − p)) + p (k_d ∂μ ∂μᵀ / sd² +
    /// k_r ∂ρ ∂ρᵀ / s_ρ²)]`, w_c the cell weight and sd the depth's own (without the shifts' spread); then the
    /// latents every cell shares are marginalised out under their priors (the depth's shift, alignment and stem
    /// front, and the reflectance level: a Schur complement). It is positive semi-definite and reads a direction the
    /// cells don't see as 0, where the observed curvature reads the cost's roughness (half the far views' pairs read
    /// a negative direction: step 2). It leaves out what other elements' cells say, and the later echoes (research
    /// 130 step 3c); a partial cell's reflectance counts `1 − behind_share` of itself, the share of edges with no
    /// unmodelled surface merging with them.
    ///
    /// **What it means under the evidence discount** (research 130 step 3b). Each cell's log-likelihood counts
    /// `w_c = 1 / (φ m)`, so the cost is a quasi-likelihood and E is its curvature: an sd read from E⁻¹ is honest only
    /// where the cells' errors are correlated as φ m declares. In general the estimate's variance is the sandwich
    /// `E⁻¹ J E⁻¹`, J the gradient's actual variance (the inverse of Godambe's information `E J⁻¹ E`), and model
    /// comparison by cost differences needs the composite-likelihood penalty `tr(J E⁻¹)`, not a parameter count. φ
    /// enters E once, and J twice: beside E comes the **gradient's variance were the cells independent** (the same
    /// sums at `w_c²`, its latents' share taken out as E's is, their priors' included), so `E⁻¹ J E⁻¹` is the variance
    /// with no correlation between cells, a lower bound, and E⁻¹ the variance at the declared φ. On the stand's true
    /// sticks (step3b.md), fits from the truth fall within 1.96 sd of E⁻¹ along the sloppiest direction 95 % of the
    /// time at 3 footprints across or more, and within 1 sd 92–96 % (the core is narrower than E says, the tail as
    /// wide); the sandwich's sds are 1.5–2.6 times too small (the cells are correlated); and below 1.5 footprints
    /// across the cost has minima far from the truth that a local curvature can't see, so E's sds don't hold there.
    ///
    /// **ln r is the space radius is unbiased in:** the stand's fitted ln r sits −0.003 from the truth (median, R ≥
    /// 1 cm); r's own mean is `exp(ln r + ½ sd²)`, so an average of radii should be taken in ln r.
    ///
    /// Without `observed`, the stencil is skipped (about two thirds of the cost: research 130 step 3b) and the slope
    /// and observed curvature read NaN, for a consumer that reads the expected information alone.
    pub fn information(&self, id: u32, observed: bool) -> Vec<Information> {
        let e = self.elems[id as usize].expect("a live element");
        let hs = 0.002;
        let (n, steps, perp, half) = match e.form {
            Form::Stick { a, b } => {
                let ax = sub(&b, &a);
                let half = 0.5 * norm(&ax);
                let u = ax.map(|x| x / (2.0 * half));
                let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
                (5, vec![0.05, hs, hs, hs / half, hs / half], [p1, cross(&u, &p1)], half)
            }
            Form::Patch { .. } => (4, vec![0.05, hs, hs, hs], [[0.0; 3]; 2], 0.0),
        };
        let at = |th: &[f64]| -> Element {
            let r = e.r * th[0].exp();
            let form = match e.form {
                Form::Stick { a, b } => {
                    let mv = |p: &Point, sign: f64| [0, 1, 2].map(|k| p[k] + (th[1] + sign * half * th[3]) * perp[0][k] + (th[2] + sign * half * th[4]) * perp[1][k]);
                    Form::Stick { a: mv(&a, -1.0), b: mv(&b, 1.0) }
                }
                Form::Patch { centre } => Form::Patch { centre: [centre[0] + th[1], centre[1] + th[2], centre[2] + th[3]] },
            };
            Element { form, r, ..e }
        };
        // the stencil: the centre, ± each step, and the four corners of each pair
        let mut pts: Vec<Vec<f64>> = vec![vec![0.0; n]];
        for i in 0..n {
            for sg in [1.0, -1.0] {
                let mut t = vec![0.0; n];
                t[i] = sg * steps[i];
                pts.push(t);
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    let mut t = vec![0.0; n];
                    t[i] = si * steps[i];
                    t[j] = sj * steps[j];
                    pts.push(t);
                }
            }
        }
        let items: Vec<(Vec<Element>, Option<u32>)> = if observed { pts.iter().map(|t| (vec![at(t)], Some(id))).collect() } else { Vec::new() };
        let c = self.costs_replacing(&items);
        let em = &self.em;
        let (k_depth, k_refl) = ((em.noise_dof + 1.0) / (em.noise_dof + 3.0), (em.refl_dof + 1.0) / (em.refl_dof + 3.0));
        let expected: Vec<[Vec<f64>; 2]> = (0..self.views.len()).into_par_iter().map(|s| {
            let pred = |t: &[f64]| -> HashMap<usize, CellPrediction> { self.predict_without(at(t), s, Some(id)).into_iter().map(|p| (p.cell, p)).collect() };
            let sides: Vec<[HashMap<usize, CellPrediction>; 2]> = (0..n).map(|i| [pred(&pts[1 + 2 * i]), pred(&pts[2 + 2 * i])]).collect();
            let refl = em.refl_noise > 0.0 && !self.views[s].echo_refl.is_empty();
            let other = self.views[s].epoch != self.ref_epoch;
            // the expected information (m) and the gradient's variance with independent cells (j): each cell's
            // information at its weight w, and at w²
            let (mut m, mut j) = (vec![0.0; n * n], vec![0.0; n * n]);
            // the latents every cell shares, the depth's shift (alignment and stem front) and the reflectance level:
            // each one's information (a; a2 at w²), what it shares with the geometry (b; b2 at w²) and its prior sd,
            // to marginalise it out
            let (mut a, mut b, mut prior) = ([0.0; 2], [vec![0.0; n], vec![0.0; n]], [0.0; 2]);
            let (mut a2, mut b2) = ([0.0; 2], [vec![0.0; n], vec![0.0; n]]);
            // cells in index order, so the sums run in the same order in every run
            let mut centre: Vec<CellPrediction> = pred(&pts[0]).into_values().collect();
            centre.sort_unstable_by_key(|p| p.cell);
            for p in centre {
                let cell = p.cell;
                // each observable's slope over the steps (0 where a side doesn't hold it, or it isn't finite)
                let slope = |f: &dyn Fn(&CellPrediction) -> f64, absent: Option<f64>| -> Vec<f64> {
                    (0..n).map(|i| {
                        let [a, b] = [0, 1].map(|k| sides[i][k].get(&cell).map(f).or(absent));
                        match (a, b) {
                            (Some(a), Some(b)) if (a - b).is_finite() => (a - b) / (2.0 * steps[i]),
                            _ => 0.0,
                        }
                    }).collect()
                };
                let echo = |q: &CellPrediction| q.arrives * q.recorded;
                let pc = echo(&p).clamp(1e-3, 1.0 - 1e-3);
                let w2 = p.weight * p.weight;
                let mut add = |g: &[f64], w: f64| {
                    for i in 0..n {
                        for k in 0..n {
                            m[i * n + k] += p.weight * w * g[i] * g[k];
                            j[i * n + k] += w2 * w * g[i] * g[k];
                        }
                    }
                };
                add(&slope(&echo, Some(0.0)), 1.0 / (pc * (1.0 - pc)));
                // the depth's own sd, without the shifts' spread `predict` widens it by
                let shifts = em.align_sd(p.surface, other).hypot(em.front_shift_sd(e.r));
                prior[0] = shifts;
                let mut obs = vec![(0, slope(&|q| q.depth, None), pc * k_depth / (p.sd * p.sd - shifts * shifts).max(1e-12))];
                if refl && p.refl.is_finite() {
                    prior[1] = em.refl_level_sd;
                    // a partial cell's reflectance reads its fill only when no unmodelled surface merges with it
                    let clear = if p.fill < 0.98 { 1.0 - em.behind_share } else { 1.0 };
                    obs.push((1, slope(&|q| q.refl, None), clear * pc * k_refl / em.refl_noise.hypot(em.refl_edge * (1.0 - p.fill.clamp(0.0, 1.0))).powi(2)));
                }
                for (k, g, w) in obs {
                    add(&g, w);
                    a[k] += p.weight * w;
                    a2[k] += w2 * w;
                    b[k].iter_mut().zip(&g).for_each(|(x, gi)| *x += p.weight * w * gi);
                    b2[k].iter_mut().zip(&g).for_each(|(x, gi)| *x += w2 * w * gi);
                }
            }
            // marginalise each latent: the information's Schur complement, and the variance of the efficient gradient
            // (the geometry's less β = b / d times the latent's, its prior's included)
            for k in 0..2 {
                if prior[k] > 0.0 {
                    let d = a[k] + 1.0 / (prior[k] * prior[k]);
                    let beta: Vec<f64> = b[k].iter().map(|x| x / d).collect();
                    for i in 0..n {
                        for l in 0..n {
                            m[i * n + l] -= b[k][i] * beta[l];
                            j[i * n + l] += -beta[i] * b2[k][l] - b2[k][i] * beta[l] + (a2[k] + 1.0 / (prior[k] * prior[k])) * beta[i] * beta[l];
                        }
                    }
                }
            }
            [m, j]
        }).collect();
        (0..self.views.len()).map(|s| {
            let [expected, gradient_variance] = expected[s].clone();
            if !observed {
                return Information { gradient: vec![f64::NAN; n], hessian: vec![f64::NAN; n * n], expected, gradient_variance };
            }
            let f = |k: usize| c[k][s];
            let g: Vec<f64> = (0..n).map(|i| (f(1 + 2 * i) - f(2 + 2 * i)) / (2.0 * steps[i])).collect();
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                m[i * n + i] = (f(1 + 2 * i) + f(2 + 2 * i) - 2.0 * f(0)) / (steps[i] * steps[i]);
            }
            let mut k = 1 + 2 * n;
            for i in 0..n {
                for j in i + 1..n {
                    let x = (f(k) - f(k + 1) - f(k + 2) + f(k + 3)) / (4.0 * steps[i] * steps[j]);
                    m[i * n + j] = x;
                    m[j * n + i] = x;
                    k += 4;
                }
            }
            Information { gradient: g, hessian: m, expected, gradient_variance }
        }).collect()
    }

    /// Each hypothesis's cost to add alone (one or more elements), per scan: (hypotheses × scans), in parallel, the
    /// scene unchanged, its foliage field and alignment held. Equal to adding it then removing it.
    pub fn costs_to_add(&self, hyps: &[Vec<Element>]) -> Vec<Vec<f64>> {
        self.costs_replacing(&hyps.iter().map(|h| (h.clone(), None)).collect::<Vec<_>>())
    }

    /// Each live element's cost to remove ([`Scene::check_id`]), per scan: (elements × scans), the scene unchanged.
    /// Equal to removing it then adding it back.
    pub fn costs_to_remove(&self, ids: &[u32]) -> Vec<Vec<f64>> {
        self.costs_replacing(&ids.iter().map(|&id| (Vec::new(), Some(id))).collect::<Vec<_>>())
    }

    /// Per (hypothesis, live element or none): the cost of the scene with the element taken out and the hypothesis put
    /// in, against the scene, per scan.
    fn costs_replacing(&self, items: &[(Vec<Element>, Option<u32>)]) -> Vec<Vec<f64>> {
        let first = self.elems.len() as u32;
        items.par_iter().map(|(h, without)| (0..self.views.len()).map(|s| {
            let with: Vec<(Element, Vec<(usize, Layer)>)> = h.iter().map(|e| (*e, render(&self.views[s], e, 0, &self.em))).collect();
            if with.iter().all(|w| w.1.is_empty()) && without.is_none_or(|w| self.rendered[s][w as usize].is_empty()) {
                return 0.0;
            }
            let over: HashMap<u32, Option<Term>> = self.reterm(s, *without, &with, first).into_iter().map(|(i, t, _)| (i, t)).collect();
            self.fits[s].total(&over) - self.fits[s].cost
        }).collect()).collect()
    }

    /// Which hypotheses interact in each scan, so that their costs scored together differ from the sum of their costs
    /// alone: (i, j, scan, `cells`, `nodes`) for i < j, `cells` when they front a cell in common or cells whose front is
    /// one scene element, `nodes` when they share an alignment node. With the alignment held, a
    /// shared node only couples them through its rank-k correction (small); shared cells are evidence they share.
    pub fn interactions(&self, hyps: &[Vec<Element>]) -> Vec<(u32, u32, u32, bool, bool)> {
        let mut out: BTreeMap<(u32, u32, u32), (bool, bool)> = BTreeMap::new();
        for (s, v) in self.views.iter().enumerate() {
            // per key (a cell, an owning element or a node), the hypotheses that have it
            let mut cells: HashMap<(bool, u64), BTreeSet<u32>> = HashMap::new();
            let mut nodes: HashMap<[i64; 3], BTreeSet<u32>> = HashMap::new();
            let rendered: Vec<Vec<Vec<(usize, Layer)>>> = hyps.par_iter().map(|h| h.iter().map(|e| render(v, e, 0, &self.em)).collect()).collect();
            for (k, (h, ls)) in hyps.iter().zip(&rendered).enumerate() {
                for (e, l) in h.iter().zip(ls).filter(|(_, l)| !l.is_empty()) {
                    for (key, _) in hats(&e.mid(), ALIGN_SPACING) {
                        nodes.entry(key).or_default().insert(k as u32);
                    }
                    for (c, _) in l.iter().filter(|(_, l)| l.fronts()) {
                        cells.entry((false, *c as u64)).or_default().insert(k as u32);
                        if let Some(&(o, _)) = self.cells[s].get(c).and_then(|e| e.iter().min_by(|a, b| a.1.total_cmp(&b.1))) {
                            cells.entry((true, o as u64)).or_default().insert(k as u32);
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
    /// receiver's resolution (0.8 m) merge into one echo at their energy-weighted range; otherwise the first is the
    /// pole's if recorded, else the wall's. Each echo's reflectance is its energy share's (merged: summed) with the
    /// default per-echo noise.
    fn pole_scan(r_pole: f64, y_pole: f64, gap: f64) -> RangeImage {
        pole_scan_of(r_pole, y_pole, gap, Settings::default().refl_wood)
    }

    /// [`pole_scan`] with the pole's wood reflecting `wood` dB.
    fn pole_scan_of(r_pole: f64, y_pole: f64, gap: f64, wood: f64) -> RangeImage {
        pole_scan_echoes(r_pole, y_pole, gap, wood, false)
    }

    /// [`pole_scan_of`], and with `later` every echo: a pulse whose pole echo is recorded and resolved from the wall
    /// also returns the wall's, with the rest of the beam, where it clears the detection floor; the instrument's cap
    /// is 4 (none is reached).
    fn pole_scan_echoes(r_pole: f64, y_pole: f64, gap: f64, wood: f64, later: bool) -> RangeImage {
        let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
        let em = Settings::default();
        let (mut t, mut d, mut r, mut refl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut echo_start, mut echo_range, mut echo_refl) = (vec![0u32], Vec::new(), Vec::new());
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
                    ((ep * ez / ec + ew * wall) / (ep + ew), 10.0 * (ep + ew).log10())
                } else {
                    (ez / ec, 10.0 * ep.log10())
                };
                refl.push((db + n_refl) as f32);
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
                echo_range.push(range as f32);
                echo_refl.push((db + n_refl) as f32);
                let n_wall = 2.0 * (-2.0 * u(seed * 37).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919 * 37)).cos();
                if kept && wall - ez / ec >= RESOLUTION && 10.0 * ew.max(1e-12).log10() + n_wall > em.floor(wall) {
                    echo_range.push(wall as f32);
                    echo_refl.push((10.0 * ew.max(1e-12).log10() + em.refl_noise * n_wall / 2.0) as f32);
                }
                echo_start.push(echo_range.len() as u32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        if later {
            let zen = d.iter().map(|w: &Point| w[2].clamp(-1.0, 1.0).acos().to_degrees()).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), z| (a.min(z), b.max(z)));
            let mut v = RangeImage::new(&t, &d, echo_start, echo_range, zen, id, [0.0; 3], 0, &RangeImageParams::default());
            v.echo_refl = echo_refl;
            v.max_targets = 4;
            return v;
        }
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
        Element { form: Form::Stick { a: [x, y, -0.2], b: [x, y, 0.2] }, r, opacity: 1.0, mismatch: 0.002 }
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
        let sum: f64 = both.1.iter().map(|r| r.cost).sum();
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
    /// into one at their energy-weighted range. With the wall in the scene (the front of a ball 2 m in radius), the
    /// true radius beats a thinner and a thicker one.
    #[test]
    fn a_stick_just_before_a_wall_merges_its_edges() {
        let (em, sc) = (Settings::default(), Background::default());
        let v = Arc::new(pole_scan(0.02, 4.0, 0.5));
        let wall = Element { form: Form::Patch { centre: [0.0, 6.5, 0.0] }, r: 2.0, opacity: 1.0, mismatch: 0.002 };
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
        let spec = Some(FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1, prior_path: 1.0 });
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
        let rows: f64 = c.rows().iter().map(|r| r.cost).sum();
        assert!((rows - c.cost()).abs() < 1e-9 * c.cost().abs().max(1.0), "{rows} vs {}", c.cost());
        c.demote(id);
        assert!(c.cost().abs() < 1e-9, "{}", c.cost());
    }

    /// Many hypotheses at once: each one's cost per scan equals adding then removing it, the scene unchanged.
    #[test]
    fn costs_to_add_match_add_then_remove() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let mut c = Scene::new(vec![v.clone(), v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
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
        let cls = classes(&v, &buf, &[el], &em, &sc, false);
        let s_along = em.align_sd(cls[0], false).hypot(em.front_shift_sd(el.r));
        let ed = |shift: f64, level: f64| -> f64 {
            buf.iter().map(|(&cell, ls)| {
                let layers: Vec<(Layer, u8)> = ls.iter().map(|l| (*l, cls[0])).collect();
                let echoes = v.echoes(cell);
                let wt = em.cell_weight(echoes.first().map_or(ls[0].t, |&r| r as f64), v.step());
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let ts: Vec<f64> = layers.iter().map(|(l, _)| l.t).collect();
                let bg = Foliage::uniform(sc.kappa).ray_bg(&v.origin, &w, &ts, None, T_MAX, echoes, em.resolution());
                let (refls, capped) = (v.echo_refls(cell), !echoes.is_empty() && echoes.len() >= v.max_targets);
                let anchor = behind_anchor(&layers, &bg.taus, &em);
                wt * (cell_ll(echoes, refls, capped, &layers, None, shift, level, &em, &bg, anchor) - cell_ll(echoes, refls, capped, &[], None, 0.0, 0.0, &em, &bg.base(), anchor))
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
        let patch = |x: f64| Element { form: Form::Patch { centre: [x, 4.0, 0.0] }, r: 0.03, opacity: 1.0, mismatch: 0.0 };
        let layers = render(&v, &patch(0.0), 0, &em);
        let f_max = layers.iter().map(|(_, l)| l.f).fold(0.0, f64::max);
        assert!(f_max > 0.99, "{f_max}");
        assert!(layers.iter().all(|(_, l)| l.x < 0.03 + 0.012 || l.f < 0.01));
        let e = |s: Element| cost(&v, &[s], &em, &sc).0;
        assert!(e(patch(0.0)) < e(patch(0.1)));
        let p = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(stick(0.0, 4.0, 0.03), 0);
        let on = p.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::On);
        let beside = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(stick(0.1, 4.0, 0.03), 0);
        let behind = beside.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::Behind);
        assert!(on && behind);
        assert!(p.iter().all(|q| q.weight > 0.0 && q.weight <= 1.0 / em.discount));
    }

    /// A patch renders with leaf's reflectance and a stick with wood's: the same disc's echo is recorded less often
    /// as leaf (the review after slice 15: patches were rendered as wood).
    #[test]
    fn a_patch_reflects_as_leaf_and_a_stick_as_wood() {
        let v = pole_view();
        let em = Settings { refl_leaf: -40.0, ..Settings::default() };
        let c = Scene::new(vec![v.clone()], &[], em, Background::default(), None, 0);
        let patch = Element { form: Form::Patch { centre: [0.0, 4.0, 0.0] }, r: 0.03, opacity: 1.0, mismatch: 0.0 };
        let mean = |p: Vec<CellPrediction>| p.iter().map(|q| q.recorded).sum::<f64>() / p.len() as f64;
        let (leaf, wood) = (mean(c.predict(patch, 0)), mean(c.predict(Element { mismatch: 0.0, ..stick(0.0, 4.0, 0.03) }, 0)));
        assert!(leaf < wood - 0.1, "leaf {leaf} wood {wood}");
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

    /// How well a scan pins a pole down: its cost curves up in every direction of a stick's geometry about the truth,
    /// most in place across the beam (the silhouette) and least in tilt, and the expected information agrees within a
    /// factor 4 in ln r and place; a row's parts sum to its cost.
    #[test]
    fn the_pole_is_pinned_down_and_rows_split_their_cost() {
        let v = pole_view();
        // nothing lies within the resolution behind the pole, so no unmodelled surface is declared (the expected
        // information averages over the declared share, which the observed curvature here doesn't see)
        let (em, sc) = (Settings { behind_share: 0.0, ..Settings::default() }, Background::default());
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
        let info = &c.information(0, true)[0];
        let (m, x) = (&info.hessian, &info.expected);
        for i in 0..3 {
            assert!(m[i * 5 + i] > 0.0, "{i}: {m:?}");
            // the expected information agrees with the observed where the pole is seen well
            assert!(x[i * 5 + i] > 0.25 * m[i * 5 + i] && x[i * 5 + i] < 4.0 * m[i * 5 + i], "{i}: expected {x:?} observed {m:?}");
        }
        let r = c.rows()[0];
        let parts = r.ray_on + r.ray_empty + r.ray_in_front + r.ray_behind + r.class + r.alignment;
        assert!((parts - r.cost).abs() < 1e-9, "{parts} vs {}", r.cost);
        assert_eq!(r.discount, em.discount);
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
            let i = c.information(0, true).remove(0);
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
        // the pole stands alone (nothing within the resolution behind it), so no unmodelled surface is declared
        let em = Settings { behind_share: 0.0, ..Settings::default() };
        for (r, y) in [(0.01, 6.0), (0.006, 6.0)] {
            let v = thin_pole_view(r, y);
            let info = |em: Settings| Scene::new(vec![v.clone()], &[stick(0.0, y, r)], em, sc, None, 0).information(0, true)[0].hessian[0];
            let (with, without) = (info(em), info(Settings { refl_noise: 0.0, ..em }));
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

    /// One edge cell, by hand (research 130 step 3c): a stick's layer at 4 m filling `f` of the beam, a wall at 8 m
    /// taking the rest, and the pulse's echoes.
    fn edge_cell(f: f64, echoes: &[f32], refls: &[f32], capped: bool, em: &Settings) -> f64 {
        let layer = |t: f64, f: f64, db: f64, elem: u32| Layer { t, spread: 0.0, f, fq: [f; 3], cos: 1.0, db, elem, op: 1.0, x: 0.0, s: 0.0 };
        let layers = [(layer(4.0, f, em.refl_wood, 0), 0u8), (layer(8.0, 1.0, em.refl_ground, 1), 0u8)];
        let bg = Foliage::uniform(0.05).ray_bg(&[0.0; 3], &[0.0, 1.0, 0.0], &[4.0, 8.0], None, T_MAX, echoes, em.resolution());
        cell_ll(echoes, refls, capped, &layers, None, 0.0, 0.0, em, &bg, behind_anchor(&layers, &bg.taus, em))
    }

    /// Later echoes behind an edge (research 130 step 3c): the wall's echo behind a stick's edge reads the rest of the
    /// beam, so with the edge's own reflectance unknown it alone tells a stick filling 0.3 of the beam from one
    /// filling 0.9 (the wall echo is 10 log10 0.7 dimmer than a full hit); without it (the pulse capped at its first
    /// echo) they differ by little. At the stand's edge spread (2 dB per unit of beam missed; AM's 10 dB reads it
    /// too, but weakly).
    #[test]
    fn a_later_echo_reads_the_rest_of_the_beam() {
        let em = Settings { refl_edge: 2.0, ..Settings::default() };
        let wall = (em.refl_ground + 10.0 * 0.7f64.log10()) as f32;
        let two = |f: f64| edge_cell(f, &[4.0, 8.0], &[f32::NAN, wall], false, &em);
        let one = |f: f64| edge_cell(f, &[4.0], &[f32::NAN], true, &em);
        assert!(two(0.3) > two(0.9) + 3.0, "with the wall's echo: {} vs {}", two(0.3), two(0.9));
        assert!((one(0.3) - one(0.9)).abs() < 0.5, "first echo alone: {} vs {}", one(0.3), one(0.9));
    }

    /// The instrument's cap (research 130 step 3c): a pulse that returned as many echoes as the instrument records
    /// says nothing past its last echo, so a wall behind that would have echoed costs nothing there; uncapped, its
    /// missing echo is evidence against it. Free space reaches the first echo less the receiver's resolution.
    #[test]
    fn a_capped_pulse_says_nothing_past_its_last_echo() {
        let em = Settings::default();
        let (capped, open) = (edge_cell(0.3, &[4.0], &[f32::NAN], true, &em), edge_cell(0.3, &[4.0], &[f32::NAN], false, &em));
        assert!(capped > open + 3.0, "capped {capped}, uncapped {open}");
        assert!((em.free_to(4.0) - (4.0 - em.resolution())).abs() < 1e-12 && em.free_to(f64::NAN) == T_MAX);
    }

    /// A merge with an unmodelled surface (research 130 step 3c): a pole 0.5 m before a wall the scene doesn't hold,
    /// closer than the receiver's resolution, so its edges' echoes merge with the wall's, bright and pulled back. With
    /// the edge model's spread small (2 dB per unit of beam missed), the branch for an unmodelled surface behind
    /// explains them: the true pole costs less with it, and beats a thinner and a thicker one.
    #[test]
    fn an_unmodelled_surface_just_behind_merges_with_the_edge() {
        let sc = Background::default();
        let v = Arc::new(pole_scan(0.02, 4.0, 0.5));
        let em = Settings { refl_edge: 2.0, behind_refl: -20.0, ..Settings::default() };
        let e = |em: &Settings, r: f64| cost(&v, &[stick(0.0, 4.0, r)], em, &sc).0;
        let (off, on) = (Settings { behind_share: 0.0, ..em }, Settings { behind_share: 0.5, ..em });
        assert!(e(&on, 0.02) < e(&off, 0.02) - 3.0, "with the branch {}, without {}", e(&on, 0.02), e(&off, 0.02));
        for r in [0.014, 0.028] {
            assert!(e(&on, r) > e(&on, 0.02) + 0.5, "r {r}: {} vs {}", e(&on, r), e(&on, 0.02));
        }
    }

    /// Later echoes in a scene (research 130 step 3c): a thin pole 3 m before a wall the scene doesn't hold, every
    /// echo kept, so the wall's echoes behind its edges read as the background's in the beam the pole leaves. The true
    /// radius beats a thinner and a thicker one, and those echoes add to the evidence against the thicker (which
    /// leaves them too little beam).
    #[test]
    fn later_echoes_behind_a_thin_poles_edges_count() {
        let (em, sc) = (Settings::default(), Background::default());
        let (first, every) = (Arc::new(pole_scan_echoes(0.006, 6.0, 3.0, em.refl_wood, false)), Arc::new(pole_scan_echoes(0.006, 6.0, 3.0, em.refl_wood, true)));
        let e = |v: &Arc<RangeImage>, r: f64| cost(v, &[stick(0.0, 6.0, r)], &em, &sc).0;
        for v in [&first, &every] {
            for r in [0.0042, 0.0084] {
                assert!(e(v, r) > e(v, 0.006) + 0.5, "r {r}: {} vs {}", e(v, r), e(v, 0.006));
            }
        }
        let against = |v: &Arc<RangeImage>| e(v, 0.0084) - e(v, 0.006);
        assert!(against(&every) > against(&first), "every echo {}, first alone {}", against(&every), against(&first));
    }
}
