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
//! - **The ray likelihood, one cell:** its first echo, or none, against a diffuse echo hazard κ along the ray (the
//!   background), the layers in depth order, then a ground plane.
//!   - The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − op_j f_j)`, and its echo is recorded with
//!     `op_i det_i`, where `det_i` is the chance it clears the detection floor ([`Settings::p_detect`]):
//!     `Φ((ρ_dB + 10 log10(e_i cos ι_i) − floor(t_i)) / σ)`.
//!   - Adjacent returns (layers and the ground) resolve into two echoes with the instrument's
//!     [`Settings::p_resolve`] of their separation and energy ratio; else they merge into one, recorded if any
//!     member is, at their energy-weighted depth (energy: op · e · cos · reflectance), its spread their spread about
//!     it. A cell's likelihood is the mixture over these merge patterns.
//!   - An echo at r has density `Σ_i P_i op_i det_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i)` over the returns, plus
//!     `κ e^{−κr} Π_{t_j<r} (1 − op_j det_j)`, where P_i is the chance that nothing nearer was kept, plus a uniform
//!     outlier share. An empty cell has probability `e^{−κ t_end} Π (1 − op_j det_j)`.
//!   - **Located edges** (3dtrees research 130 step 2): the echo's reflectance x (dB) is read beside its range. A
//!     return predicts `10 log10 Σ e_i cos^k ι_i 10^(ρ_i / 10)` over its members (one layer, or a merge with a
//!     surface behind), at the mismatch's three section radii, and x reads about it with the per-echo noise, wider at
//!     a partial fill, truncated by the detection floor ([`Settings::refl_density`], [`Return::refl_density`]); the hazard's and the outliers' echoes read the background's
//!     spread ([`Settings::refl_background`]). So a silhouette's partial cells place its edge within a cell. Each
//!     element's reflectance level (its offset from its surface's reflectance, per scan) is a latent beside its
//!     shift, so an unknown bark brightness is read as a level, not a width.
//!   - Every cell an element fronts at fill [`MIN_FILL`] or more counts, partial hits included, so every
//!     hypothesis is scored on the same echoes (slice 7c: a gate at fill 0.5, with the band below it scored by
//!     located edges, let a wider stick move echoes from the edges into the gate, where each is worth far more).
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
//!   sees it. A scan's cost is the Laplace marginal of its data over both and the reflectance levels ([`scan_cost`]).
//! - **Weights:** each cell counts [`Settings::cell_weight`], `1 / (φ m(ρ))`, with m the footprints per cell
//!   (overlapping footprints, on the image's own lattice step) and φ the evidence discount per cell: one constant
//!   for every cell and every hypothesis, so it widens posteriors without favouring any hypothesis (a
//!   quasi-likelihood). It replaced slice 7's ρ_res, `1 + (n − 1) ρ` with n an element's own footprints, which
//!   differed between hypotheses and, once every fronted cell counted, favoured the thinner (slice 7c).
//!
//! - **The scene** ([`Scene`]) holds elements against range images, each rendered once. A move (add, remove,
//!   change) returns its cost change, and recomputes only the alignment components the touched element joins or
//!   leaves. Its rows ([`ElementScan`]) split each (element, scan)'s cost into its parts, and
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
use std::collections::{HashMap, HashSet};

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
/// - The ray likelihood's cost at the alignment's mode, by what each cell's first echo says about the element:
///   `ray_on` (its echoes), `ray_empty` (pulses with no echo), `ray_in_front` (hidden by something nearer) and
///   `ray_behind` (pulses that passed through it: evidence against).
/// - `class`: class evidence's cost (0 while it is off); `alignment`: an equal share of the alignment's prior and
///   log-determinant terms.
/// - `discount`: the evidence discount φ every cell's evidence was divided by.
/// - The posterior at the element in this scan, along the beam (m): the scan alignment error's shift and sd, and the
///   stem front shift's (0 when it is off); and its reflectance level's (dB; 0 while reflectance isn't read).
#[derive(Debug, Clone, Copy)]
pub struct ElementScan {
    pub elem: u32,
    pub scan: u32,
    pub cells: u32,
    pub cells_on: u32,
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

/// What links an element to others in one view's components: its alignment nodes and the cells it fronts.
fn links(e: &Element, layers: &[(usize, Layer)]) -> Vec<(u8, [i64; 3])> {
    Alignment::hats(&e.mid(), ALIGN_SPACING).into_iter().map(|(n, _)| (0u8, n))
        .chain(layers.iter().filter(|(_, l)| l.fronts()).map(|&(c, _)| (1u8, [c as i64, 0, 0])))
        .collect()
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
    let bg = ext.ray_bg(&v.origin, &w, ts, ground.map(|g| g.0), ground.map_or(T_MAX, |g| g.0), obs);
    CellView { obs, ground, wt, bg }
}

/// The front element's echo in a cell as the likelihood reads it: the first return of the most likely way the cell's
/// layers (each with its surface class) and the ground come back ([`returns`]), with its sd widened by the alignment
/// error's and the stem front shift's prior spreads (an element of radius `r`; `other` for a scan off the reference
/// epoch). [`Outcome::of`] against it is the one rule for "the cell's first echo is the element's".
fn front_echo(layers: &[(Layer, u8)], ground: Option<(f64, f64)>, em: &Settings, taus: &[f64], r: f64, other: bool) -> Return {
    let pats = returns(layers, ground, 0.0, 0.0, em, taus).0;
    let first = pats.iter().max_by(|a, b| a.0.total_cmp(&b.0)).and_then(|p| p.1.first().copied());
    let mut f = first.unwrap_or(Return { t: f64::NAN, mu: f64::NAN, sd: f64::NAN, s: 0.0, s_q: [0.0; 3], db: [f64::NAN; 3], fill: [0.0; 3], arrives: 0.0 });
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
/// mismatch's three section radii (as [`Layer`]'s `fq`); `t` its first member's range and `arrives` the chance the
/// pulse meets nothing of the foliage field before it.
#[derive(Clone, Copy)]
struct Return {
    t: f64,
    mu: f64,
    sd: f64,
    s: f64,
    s_q: [f64; 3],
    db: [f64; 3],
    fill: [f64; 3],
    arrives: f64,
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
/// the opacity; its echo's reflectance is `10 log10` of it) and the foliage field's optical depth to it.
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
}

/// The ways a cell's layers (each with its element's surface class) and the ground at `ground` = (range, cos) come
/// back, with every layer shifted by `shift` along the beam and its reflectance by `level` (dB): each a set of
/// returns, nearest first, and its chance; and the range the beam ends at. Adjacent members resolve into two echoes
/// with [`Settings::p_resolve`] of their separation and energy ratio, else merge at their energy-weighted depth and
/// summed energy share; patterns under 1e-4 are dropped. `taus` are the foliage field's optical depths to each layer
/// and then to the ground ([`RayFoliage`]).
fn returns(layers: &[(Layer, u8)], ground: Option<(f64, f64)>, shift: f64, level: f64, em: &Settings, taus: &[f64]) -> (Vec<(f64, Vec<Return>)>, f64) {
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
                        fill: ly.fq.map(|f| f * b0), tau: taus[li] });
    }
    if let Some((tg, cg)) = ground {
        let cg = cg.powf(em.refl_incidence);
        let share = beam * cg * 10f64.powf(em.refl_ground / 10.0);
        let s = em.p_detect(em.refl_ground, beam * cg, tg);
        m.push(Member { t: tg, depth: tg, sd: em.noise(2), s, s_q: [s; 3], energy: share, share: [share; 3],
                        fill: [beam; 3], tau: taus[layers.len()] });
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
            return Return { t: g[0].t, mu: g[0].depth, sd: g[0].sd, s: g[0].s, s_q: g[0].s_q, db, fill, arrives: (-g[0].tau).exp() };
        }
        // energy weights; members the beam no longer reaches weigh alike (the return is never recorded then)
        let total: f64 = g.iter().map(|x| x.energy).sum();
        let wt = |x: &Member| if total > 0.0 { x.energy / total } else { 1.0 / g.len() as f64 };
        let mu = g.iter().map(|x| wt(x) * x.depth).sum::<f64>();
        let var = g.iter().map(|x| wt(x) * (x.sd * x.sd + (x.depth - mu).powi(2))).sum::<f64>();
        let none: f64 = g.iter().map(|x| 1.0 - x.s).product();
        let s_q = [0, 1, 2].map(|q| 1.0 - g.iter().map(|x| 1.0 - x.s_q[q]).product::<f64>());
        Return { t: g[0].t, mu, sd: var.sqrt(), s: 1.0 - none, s_q, db, fill, arrives: (-g[0].tau).exp() }
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

/// One cell's log-likelihood: its first echo (or none), at its range and reflectance `refl` (dB; NaN unknown), against
/// the foliage field along its ray (`bg`) and the returns its layers and the ground give ([`returns`]), the layers
/// shifted by `shift` along the beam and `level` in reflectance.
#[allow(clippy::too_many_arguments)]
fn cell_ll(echoes: &[f32], refl: f64, layers: &[(Layer, u8)], ground: Option<(f64, f64)>, shift: f64, level: f64, em: &Settings, bg: &RayFoliage) -> f64 {
    // the first echo only, for now
    let obs = echoes.first().map_or(f64::NAN, |&r| r as f64);
    let (patterns, t_end) = returns(layers, ground, shift, level, em, &bg.taus);
    let other = em.refl_background(refl);
    let lik = |rets: &[Return]| {
        let (mut none_kept, mut none_before_obs, mut dens) = (1.0, 1.0, 0.0);
        for r in rets {
            if obs.is_finite() {
                dens += r.arrives * none_kept * r.s * t_pdf((obs - r.mu) / r.sd, em.noise_dof) / r.sd * r.refl_density(refl, em);
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
            dens += bg.sig_obs * (-bg.tau_obs).exp() * none_before_obs * other;
        }
        (1.0 - P_OUTLIER) * dens + P_OUTLIER / T_MAX * other
    };
    if patterns.len() == 1 {
        return lik(&patterns[0].1).max(1e-300).ln();
    }
    patterns.iter().map(|(p, rets)| p * lik(rets)).sum::<f64>().max(1e-300).ln()
}

/// One cell an element fronts, as its ray likelihood scores it: its ray through the foliage field, echoes, its first
/// echo's reflectance (dB; NaN unknown), layers, ground and weight, its log-likelihood under the empty scene, and what
/// its first echo says about the element.
struct CellData {
    bg: RayFoliage,
    echoes: Vec<f32>,
    refl: f64,
    layers: Vec<(Layer, u8)>,
    ground: Option<(f64, f64)>,
    wt: f64,
    base: f64,
    outcome: Outcome,
}

/// One element's data in one scan: the cells it fronts, and its rows over the scan's latent nodes, each entry (node,
/// scale): `along`, the along-beam shift δ of its cells' depths (the alignment error's nodes, then its own stem front
/// shift node when `front`), and `level`, its own reflectance level node (none while reflectance isn't read).
struct ElemData {
    elem: usize,
    cells: Vec<CellData>,
    class: u8,
    cells_on: u32,
    across: f64,
    along: Vec<(usize, f64)>,
    front: bool,
    level: Option<(usize, f64)>,
    /// Its photometry in this scan, for class evidence: [n, Σ reflectance, Σ albedo, Σ multi-return, Σ cos] over its
    /// echoes in cells it mostly fills.
    photo: [f64; 5],
}

impl ElemData {
    /// Its ray likelihood at `x` = (δ, reflectance level), in total and by [`Outcome`] (in its order).
    fn ll_parts(&self, x: (f64, f64), em: &Settings) -> (f64, [f64; 4]) {
        let (mut total, mut parts) = (0.0, [0.0; 4]);
        for c in &self.cells {
            let l = c.wt * (cell_ll(&c.echoes, c.refl, &c.layers, c.ground, x.0, x.1, em, &c.bg) - c.base);
            total += l;
            parts[c.outcome as usize] += l;
        }
        (total, parts)
    }

    fn ll(&self, x: (f64, f64), em: &Settings) -> f64 {
        self.ll_parts(x, em).0
    }

    /// (δ, reflectance level) from the nodes `z`.
    fn at(&self, z: &[f64]) -> (f64, f64) {
        (self.along.iter().map(|&(j, w)| w * z[j]).sum(), self.level.map_or(0.0, |(j, w)| w * z[j]))
    }

    /// The shift its row gives from the nodes `z`: the alignment's part and the stem front shift's.
    fn shifts(&self, z: &[f64]) -> (f64, f64) {
        let n = self.along.len() - self.front as usize;
        let part = |r: &[(usize, f64)]| r.iter().map(|&(j, w)| w * z[j]).sum::<f64>();
        (part(&self.along[..n]), part(&self.along[n..]))
    }

    /// −ll's slope and curvature in (δ, level) at `x`, by central differences (3e-4 m, 0.05 dB; the cross term
    /// forward), the curvature's negative directions dropped. Without a level node only δ's are filled.
    fn quad(&self, x: (f64, f64), em: &Settings) -> ([f64; 2], [[f64; 2]; 2]) {
        let (h, hb) = (3e-4, 0.05);
        let f = |d: f64, b: f64| -self.ll((x.0 + d, x.1 + b), em);
        let (f0, fp, fm) = (f(0.0, 0.0), f(h, 0.0), f(-h, 0.0));
        let mut g = [(fp - fm) / (2.0 * h), 0.0];
        let (a, mut b, mut c) = ((fp - 2.0 * f0 + fm) / (h * h), 0.0, 0.0);
        if self.level.is_some() {
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
}

/// A scan's alignment nodes: trilinear hats on a grid of spacing ℓ, normalised so the alignment error's variance is
/// the same everywhere (`t(x) = Σ w_j z_j / √Σ w_j²`); each node carries the along-beam shift. Elements' own stem
/// front shift nodes are keyed apart.
struct Alignment {
    nodes: HashMap<[i64; 3], usize>,
}

impl Alignment {
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

/// One scan's cost for a set of elements: the Laplace marginal of its data over the alignment nodes, the stem front
/// shifts and the reflectance levels, `U = −Σ_e A_e(δ̂_e, b̂_e) + ½ ẑᵀẑ + ½ log|Λ|`, Λ the posterior precision at the
/// mode ẑ (the prior is unit; each element's spreads are in its rows), plus class evidence; and a row per element.
fn scan_cost(v: &RangeImage, si: usize, rendered: &[Vec<(usize, Layer)>], elems: &[Element], ref_epoch: u32, em: &Settings, sc: &Background, ext: &Foliage)
               -> (f64, Vec<ElementScan>) {
    let per = rendered;
    let buf = kbuffer(per);
    let other = v.epoch != ref_epoch;
    let cls = classes(v, &buf, elems, em, sc, other);
    let mut align = Alignment { nodes: HashMap::new() };
    let mut data: Vec<ElemData> = Vec::new();
    let mut by_elem: HashMap<u32, usize> = HashMap::new();
    for (i, e) in elems.iter().enumerate() {
        if per[i].is_empty() {
            continue;
        }
        let k = cls[i];
        let mid = e.mid();
        let s_along = em.align_sd(k, other);
        let mut along: Vec<(usize, f64)> = Alignment::hats(&mid, ALIGN_SPACING).into_iter().map(|(key, w)| (align.node(key), w * s_along)).collect();
        let front = em.front_shift_sd(e.r) > 0.0;
        if front {
            // the element's own node: its shape, as this view sees it
            along.push((align.node([i64::MIN, i as i64, 0]), em.front_shift_sd(e.r)));
        }
        by_elem.insert(i as u32, data.len());
        // the element's own reflectance level, where the scan has reflectances and they are read
        let level = (em.refl_noise > 0.0 && em.refl_level_sd > 0.0 && !v.echo_refl.is_empty())
            .then(|| (align.node([i64::MIN + 1, i as i64, 0]), em.refl_level_sd));
        data.push(ElemData { elem: i, cells: Vec::new(), class: k, cells_on: 0, across: 2.0 * e.r / em.footprint(norm(&sub(&mid, &v.origin))), along, front,
                             level, photo: [0.0; 5] });
    }
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
        let echoes = v.echoes(cell);
        if let (true, true, Some(&db)) = (outcome == Outcome::On, front.f > MOSTLY, v.echo_refls(cell).first()) {
            let c = if v.cos_inc[cell].is_finite() { (v.cos_inc[cell] as f64).clamp(0.05, 1.0) } else { 0.5 };
            let db = db as f64;
            for (a, x) in el.photo.iter_mut().zip([1.0, db, db - 10.0 * c.log10(), (echoes.len() > 1) as u8 as f64, c]) {
                *a += x;
            }
        }
        let refl = v.echo_refls(cell).first().map_or(f64::NAN, |&x| x as f64);
        let base = cell_ll(echoes, refl, &[], cv.ground, 0.0, 0.0, em, &cv.bg.base());
        el.cells.push(CellData { bg: cv.bg, echoes: echoes.to_vec(), refl, layers, ground: cv.ground, wt: cv.wt, base, outcome });
    }
    let n = align.nodes.len();
    // sums in a fixed order, so the same elements always give the same cost to the bit
    let objective = |z: &[f64]| 0.5 * z.iter().map(|x| x * x).sum::<f64>() - data.par_iter().map(|el| el.ll(el.at(z), em)).collect::<Vec<f64>>().iter().sum::<f64>();
    // Newton with each element's −A as a quadratic in (δ, level) by central differences (negative curvature dropped);
    // a step at most 3 prior sds, halved until the objective falls, stopping under a millinat. The Hessian the
    // log-determinant reads is the one at the final mode.
    let mut z = vec![0.0; n];
    let mut hess = nalgebra::DMatrix::<f64>::identity(n, n);
    let mut done = false;
    for it in 0..=MAX_NEWTON {
        let mut grad = nalgebra::DVector::from_column_slice(&z);
        hess = nalgebra::DMatrix::identity(n, n);
        // each element's slope and curvature in parallel (the costly part), then assembled over its rows
        let quads: Vec<([f64; 2], [[f64; 2]; 2])> = data.par_iter().map(|el| el.quad(el.at(&z), em)).collect();
        for (el, (g, hl)) in data.iter().zip(&quads) {
            let rows = [&el.along[..], el.level.as_slice()];
            for (a, ra) in rows.iter().enumerate() {
                for &(i, wi) in ra.iter() {
                    grad[i] += g[a] * wi;
                    for (c, rc) in rows.iter().enumerate() {
                        for &(j, wj) in rc.iter() {
                            hess[(i, j)] += hl[a][c] * wi * wj;
                        }
                    }
                }
            }
        }
        if done || it == MAX_NEWTON || n == 0 {
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
        if t < 1e-3 {
            break; // no step taken: the Hessian is the mode's
        }
        // a last pass reads the Hessian at the new mode
        done = fall < 1e-3;
    }
    // the posterior precision is the unit prior plus positive semi-definite terms, so only a non-finite cost fails
    let (logdet, cov) = match hess.clone().cholesky() {
        Some(c) => (2.0 * c.l().diagonal().iter().map(|x| x.ln()).sum::<f64>(), c.inverse()),
        None => (f64::NAN, nalgebra::DMatrix::zeros(n, n)),
    };
    let shared = 0.5 * z.iter().map(|x| x * x).sum::<f64>() + 0.5 * logdet;
    let lls: Vec<(f64, [f64; 4])> = data.par_iter().map(|el| el.ll_parts(el.at(&z), em)).collect();
    // class evidence: the element's form is its class hypothesis (a stick wood, a patch leaf)
    let class: Vec<f64> = data.iter().map(|el| {
        let llr = em.class_llr(&el.photo);
        match elems[el.elem].form { Form::Stick { .. } => -llr, Form::Patch { .. } => llr }
    }).collect();
    let u = shared - lls.iter().map(|x| x.0).sum::<f64>() + class.iter().sum::<f64>();
    let share = shared / data.len().max(1) as f64;
    let rows = data.iter().zip(&lls).zip(&class).map(|((el, (l, parts)), d)| {
        let var = |r: &[(usize, f64)]| r.iter().map(|&(i, wi)| r.iter().map(|&(j, wj)| wi * wj * cov[(i, j)]).sum::<f64>()).sum::<f64>().max(0.0).sqrt();
        let n_align = el.along.len() - el.front as usize;
        let (align_shift, front_shift) = el.shifts(&z);
        let lv = el.level.as_slice();
        ElementScan {
            elem: el.elem as u32, scan: si as u32, cells: el.cells.len() as u32, cells_on: el.cells_on, footprints_across: el.across, surface: el.class,
            cost: -l + d + share, ray_on: -parts[0], ray_empty: -parts[1], ray_in_front: -parts[2], ray_behind: -parts[3], class: *d,
            alignment: share, discount: em.discount, align_shift, align_sd: var(&el.along[..n_align]), front_shift, front_shift_sd: var(&el.along[n_align..]),
            refl_level: el.at(&z).1, refl_level_sd: var(lv),
        }
    }).collect();
    (u, rows)
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
/// - A scan's cost is a sum over its components: the elements linked by a shared alignment node or a shared cell.
///   Their Laplace marginals are independent (the alignment's prior is diagonal, and an element's terms touch only
///   its nodes and cells), so the sum is the scan's whole marginal.
/// - Component energies are memoised by membership. A move recomputes only the components the touched element joins
///   or leaves, and removing an element restores the cost exactly (synthesis §18 invariant 7).
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
    /// [image]: component members (sorted ids) → its cost and rows; and the scan's cost.
    memo: Vec<HashMap<Vec<u32>, (f64, Vec<ElementScan>)>>,
    scan_u: Vec<f64>,
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
                             rendered: Vec::new(), memo: vec![HashMap::new(); n], scan_u: vec![0.0; n] };
        c.rendered = c.views.par_iter().map(|v| render_all(v, elems, &c.em)).collect();
        c.ext = c.residual();
        c.refresh(&(0..n).collect::<Vec<_>>());
        c
    }

    /// The scene's cost against the empty scene: its elements' cells, each against the same cell with the foliage
    /// field alone (the rays no element fronts are the field's in both, and cancel).
    pub fn cost(&self) -> f64 {
        self.scan_u.iter().sum::<f64>()
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

    /// Re-estimate the foliage field from the rays the elements leave (a move: its cost change). Moves that change
    /// elements hold the field as it is; after this one the scene's cost is a fresh scene's.
    pub fn refit_foliage(&mut self) -> f64 {
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

    /// Hold a foliage field estimated elsewhere (another scene's, from other scans) fixed: `refit_foliage` then
    /// leaves it. The cost change.
    pub fn hold_foliage(&mut self, ext: &Foliage) -> f64 {
        let before = self.cost();
        self.ext = ext.clone();
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

    /// Remove a live element ([`Scene::check_id`]): the cost change.
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

    /// Replace a live element ([`Scene::check_id`]): the cost change.
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

    /// What an element predicts in image `s` ([`Scene::check_scan`]) against the scene, per cell it fronts, on the
    /// path the likelihood scores ([`front_echo`], [`cell_view`]): it is rendered for this, not added, and only the
    /// scene's layers in its cells enter its k-buffer, so its front cells and class are those of the whole set.
    pub fn predict(&self, e: Element, s: usize) -> Vec<CellPrediction> {
        let (v, em) = (&*self.views[s], &self.em);
        let mine = render(v, &e, 0, em);
        let cells: HashSet<usize> = mine.iter().map(|&(c, _)| c).collect();
        let (mut els, mut per) = (vec![e], vec![mine]);
        for (i, el) in self.elems.iter().enumerate() {
            let Some(el) = el else { continue };
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
                cell, elem: 0, fill: ly.f, recorded: fe.s, arrives: fe.arrives, surface: cls[0], depth: fe.mu, sd: fe.sd, refl: if fe.db[1] > -200.0 { fe.db[1] } else { f64::NAN }, cos: ly.cos, obs: cv.obs,
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
    pub fn information(&self, id: u32) -> Vec<(Vec<f64>, Vec<f64>)> {
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
        let hyps: Vec<Vec<Element>> = pts.iter().map(|t| vec![at(t)]).collect();
        let c = self.costs_replacing(&hyps, Some(id));
        (0..self.views.len()).map(|s| {
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
            (g, m)
        }).collect()
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

    /// One scan's components: live elements it sees, linked by shared alignment nodes and shared cells, each sorted,
    /// in order of their first id.
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
    /// scene unchanged and its foliage field held. Equal to adding it then removing it: a hypothesis joins the scan's
    /// components it shares an alignment node or a fronted cell with, and its cost is that union's against theirs.
    pub fn costs_to_add(&self, hyps: &[Vec<Element>]) -> Vec<Vec<f64>> {
        self.costs_replacing(hyps, None)
    }

    /// [`Scene::costs_to_add`] with the live element `without` (if any) taken out of the scene first: the cost of
    /// replacing it by each hypothesis.
    fn costs_replacing(&self, hyps: &[Vec<Element>], without: Option<u32>) -> Vec<Vec<f64>> {
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
            let (cs, owner) = &comps[s];
            let extra: Vec<Vec<(usize, Layer)>> = h.iter().map(|e| render(v, e, 0, &self.em)).collect();
            let home = without.and_then(|id| cs.iter().position(|m| m.contains(&id)));
            if extra.iter().all(|l| l.is_empty()) && home.is_none() {
                return 0.0;
            }
            let mut touched: Vec<usize> = h.iter().zip(&extra).flat_map(|(e, l)| links(e, l)).filter_map(|k| owner.get(&k).copied()).chain(home).collect();
            touched.sort_unstable();
            touched.dedup();
            let members: Vec<u32> = touched.iter().flat_map(|&g| cs[g].iter().copied()).filter(|&i| Some(i) != without).collect();
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
                    ((ep * ez / ec + ew * wall) / (ep + ew), 10.0 * (ep + ew).log10())
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

    /// A scene's moves: adding then removing an element restores its cost exactly (synthesis §18
    /// invariant 7), and each move's cost change equals the full recomputation's difference to rounding, for an
    /// add, an alter and a remove, with an unrelated element far away left untouched.
    #[test]
    fn a_scenes_moves_match_full_recomputation() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let full = |els: &[Element]| cost(&v, els, &em, &sc).0;
        let (pole, far, wide) = (stick(0.0, 4.0, 0.03), stick(0.6, 7.0, 0.03), stick(0.0, 4.0, 0.04));
        let mut c = Scene::new(vec![v.clone()], &[far], em, sc, None, 0);
        let e0 = c.cost();
        assert!((e0 - full(&[far])).abs() < 1e-6 * e0.abs().max(1.0));
        let (id, d_add) = c.add(pole);
        assert!((d_add - (full(&[far, pole]) - full(&[far]))).abs() < 1e-6 * d_add.abs().max(1.0), "{d_add}");
        let d_alter = c.change(id, wide);
        assert!((d_alter - (full(&[far, wide]) - full(&[far, pole]))).abs() < 1e-6 * d_alter.abs().max(1.0), "{d_alter}");
        c.remove(id);
        assert_eq!(c.cost(), e0);
        let t: f64 = c.costs_to_add(&[vec![pole]])[0].iter().sum();
        assert_eq!(c.cost(), e0);
        assert!((t - d_add).abs() < 1e-9 * t.abs().max(1.0));
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
            let h = Alignment::hats(&x, 2.0);
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
                let bg = Foliage::uniform(sc.kappa).ray_bg(&v.origin, &w, &ts, None, T_MAX, echoes.first().map_or(f64::NAN, |&r| r as f64));
                let refl = v.echo_refls(cell).first().map_or(f64::NAN, |&x| x as f64);
                wt * (cell_ll(echoes, refl, &layers, None, shift, level, &em, &bg) - cell_ll(echoes, refl, &[], None, 0.0, 0.0, &em, &bg.base()))
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
    /// most in place across the beam (the silhouette) and least in tilt; a row's parts sum to its cost.
    #[test]
    fn the_pole_is_pinned_down_and_rows_split_their_cost() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
        let m = &c.information(0)[0].1;
        for i in 0..3 {
            assert!(m[i * 5 + i] > 0.0, "{i}: {m:?}");
        }
        let r = c.rows()[0];
        let parts = r.ray_on + r.ray_empty + r.ray_in_front + r.ray_behind + r.class + r.alignment;
        assert!((parts - r.cost).abs() < 1e-9, "{parts} vs {}", r.cost);
        assert_eq!(r.discount, em.discount);
    }

    /// Located edges (research 130 step 2): the silhouette's partial cells, read through their echoes' reflectance
    /// against the beam's fill, pin a thin stick's radius down more than their depths and misses alone, from one view.
    #[test]
    fn located_edges_pin_a_thin_sticks_radius() {
        let sc = Background::default();
        for (r, y) in [(0.01, 6.0), (0.006, 6.0)] {
            let v = thin_pole_view(r, y);
            let info = |em: Settings| Scene::new(vec![v.clone()], &[stick(0.0, y, r)], em, sc, None, 0).information(0)[0].1[0];
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
}
