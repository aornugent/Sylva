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
//! - **The offset:** a cell belongs to the (scan, block) of its nearest element. Each group shares one offset δ
//!   along the beam, the scan's offset field there. δ is drawn `t_ν_L(0, σ_L[k] / √2)`, with the cross-epoch part
//!   added when the scans span epochs, and integrated once per group by quadrature. The surface class k (which
//!   sets the spread, the bias and the noise) is the median class of the view's echoes on the element's front.
//! - **Weights:** each cell counts `1/m(ρ)`, with m the footprints per cell (overlapping footprints), over the
//!   design effect `1 + (n − 1) ρ` of its nearest element, n that element's fill-weighted footprints in the scan
//!   (correlated misfit; the element is the unit).
//!
//! Cells no element touches cancel. Context (neighbours, competitors) enters as more elements in the set.

use super::view::{state, ErrorModel, View};
use crate::transform::{cross, dot, norm, normalize, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Layers kept per cell. A cell that runs out (rare among a few elements) drops its farthest.
pub const K: usize = 3;
/// Quadrature nodes over the offset, spanning ±4 sd.
const NODES: usize = 21;

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
/// those with an echo on it; `across`: footprints across its diameter; `energy`: its share of its groups' energy
/// (its cells' terms at the group's most likely offset, plus the group's remainder split by cells, plus its term
/// C), so the rows sum to the total.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub elem: u32,
    pub scan: u32,
    pub cells: u32,
    pub support: u32,
    pub across: f64,
    pub surface: u8,
    pub energy: f64,
}

type Key = (u32, [i64; 3]);

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
            let sd = (0.0625 * em.footprint(tm).powi(2) + e.misfit * e.misfit).sqrt();
            let f = phi((e.r - d) / sd) - phi((-e.r - d) / sd);
            if d > e.r + 3.0 * sd + 2.0 * em.step * tm {
                continue;
            }
            let depth = |x: f64| tm - (e.r * e.r - x * x).max(0.0).sqrt() / sin;
            let x2 = (d + sd).min(e.r);
            let x1 = (d - sd).max(0.0).min(x2);
            let xm = 0.5 * (x1 + x2);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t: 0.5 * (depth(x1) + depth(x2)),
                    spread: ((depth(x2) - depth(x1)).powi(2) / 12.0 + e.misfit * e.misfit).sqrt(),
                    f,
                    cos: ((1.0 - (xm / e.r).powi(2)).max(0.0).sqrt() * sin).max(0.05),
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
/// spread, misfit and one scan's offset spread), the element's surface class, the observed first echo (NaN for
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
            let [same, _] = em.sigma_l[k as usize];
            let depth = ly.t + em.depth_bias(k, ly.t);
            let sd = (em.sigma_e[k as usize].powi(2) + ly.spread.powi(2) + 0.5 * same * same).sqrt();
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

/// The energy of `elems` against `views` (−log likelihood ratio against the empty scene), and its rows. With
/// `only = Some(i)`, the energy change of adding element i to the others, `E(all) − E(all but i)`, over the groups
/// it touches; rows are then empty.
pub fn energy(views: &[&View], elems: &[Element], em: &ErrorModel, sc: &Scene, only: Option<usize>) -> (f64, Vec<Row>) {
    let rendered: Vec<Vec<Vec<(usize, Layer)>>> = views.iter().map(|v| render_all(v, elems, em)).collect();
    let bufs: Vec<HashMap<usize, Vec<Layer>>> = rendered.iter().map(|r| kbuffer(r, 0.5)).collect();
    let classes: Vec<Vec<u8>> = views.iter().zip(&bufs).map(|(v, buf)| classes(v, buf, elems.len(), em)).collect();
    let epochs = views.iter().map(|v| v.epoch).collect::<HashSet<u32>>().len() > 1;
    // each element's design effect per view, from its fill-weighted footprints
    let temper: Vec<Vec<f64>> = bufs
        .iter()
        .map(|buf| {
            let mut n = vec![0.0; elems.len()];
            for ls in buf.values() {
                for l in ls {
                    n[l.elem as usize] += l.f / em.multiplicity(l.t);
                }
            }
            n.iter().map(|&n| 1.0 / (1.0 + (n - 1.0f64).max(0.0) * em.rho)).collect()
        })
        .collect();
    let block = |i: u32| elems[i as usize].mid().map(|x| (x / em.block).floor() as i64);
    // per cell, its group in a configuration that leaves out `skip`
    let key_of = |si: usize, ls: &[Layer], skip: Option<u32>| ls.iter().find(|l| Some(l.elem) != skip).map(|l| (si as u32, block(l.elem)));
    let skip_i = only.map(|i| i as u32);
    let touched: Option<HashSet<Key>> = skip_i.map(|i| {
        let mut g = HashSet::new();
        for (si, buf) in bufs.iter().enumerate() {
            for ls in buf.values() {
                if ls.iter().any(|l| l.elem == i) {
                    g.extend(key_of(si, ls, None));
                    g.extend(key_of(si, ls, Some(i)));
                }
            }
        }
        g
    });
    // the configurations to score: all elements (+1), and without element i (−1)
    let configs: Vec<(Option<u32>, f64)> = match skip_i {
        None => vec![(None, 1.0)],
        Some(i) => vec![(None, 1.0), (Some(i), -1.0)],
    };
    let mut total = 0.0;
    let mut rows: HashMap<(u32, u32), Row> = HashMap::new();
    // term C per (view, element), tempered as A; it doesn't see the offset (it lies across the beam)
    let zt = phi_inv(1.0 - em.edge_fill);
    for (si, v) in views.iter().enumerate() {
        let cs: Vec<f64> = (0..elems.len())
            .into_par_iter()
            .map(|h| {
                let k = classes[si][h];
                temper[si][h] * edge_ll(v, &rendered[si][h], &elems[h], k, zt, sd_of(em, k, epochs), em)
            })
            .collect();
        for (h, c) in cs.into_iter().enumerate() {
            if only.is_none() || only == Some(h) {
                total -= c;
            }
            if only.is_none() && c != 0.0 {
                rows.entry((h as u32, si as u32))
                    .or_insert(Row { elem: h as u32, scan: si as u32, cells: 0, support: 0, across: 0.0, surface: classes[si][h], energy: 0.0 })
                    .energy -= c;
            }
        }
    }
    for &(skip, sign) in &configs {
        // per group: the weighted log-likelihood at each offset node, the empty scene's, and its elements' shares
        struct Acc {
            ll: [f64; NODES],
            ll0: f64,
            members: HashMap<u32, (f64, [f64; NODES])>,
        }
        let mut groups: HashMap<Key, Acc> = HashMap::new();
        for (si, (v, buf)) in views.iter().zip(&bufs).enumerate() {
            let cells: Vec<(usize, &Vec<Layer>)> = buf.iter().map(|(&c, ls)| (c, ls)).collect();
            let drop = drop_of(em, v);
            let terms: Vec<Option<(Key, u32, f64, [f64; NODES])>> = cells
                .par_iter()
                .map(|&(cell, ls)| {
                    let key = key_of(si, ls, skip)?;
                    if touched.as_ref().is_some_and(|t| !t.contains(&key)) {
                        return None;
                    }
                    let layers: Vec<(Layer, u8)> = ls.iter().filter(|l| Some(l.elem) != skip).map(|l| (*l, classes[si][l.elem as usize])).collect();
                    let front = layers[0].0;
                    let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
                    let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                    let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
                    let wt = temper[si][front.elem as usize] / em.multiplicity(if obs.is_finite() { obs } else { front.t });
                    let sd = sd_of(em, classes[si][front.elem as usize], epochs);
                    let ll: [f64; NODES] = std::array::from_fn(|g| wt * cell_ll(obs, &layers, ground, node(g) * sd, drop, em, sc));
                    Some((key, front.elem, wt * cell_ll(obs, &[], ground, 0.0, drop, em, sc), ll))
                })
                .collect();
            for ((cell, ls), t) in cells.iter().zip(terms) {
                let Some((key, front, ll0, ll)) = t else { continue };
                let acc = groups.entry(key).or_insert_with(|| Acc { ll: [0.0; NODES], ll0: 0.0, members: HashMap::new() });
                acc.ll0 += ll0;
                let m = acc.members.entry(front).or_insert((0.0, [0.0; NODES]));
                m.0 += 1.0;
                for g in 0..NODES {
                    acc.ll[g] += ll[g];
                    m.1[g] += ll[g] - ll0;
                }
                if skip.is_none() && only.is_none() {
                    let ly = ls[0];
                    let across = 2.0 * elems[front as usize].r / em.footprint(ly.t);
                    let row = rows.entry((front, si as u32)).or_insert(Row {
                        elem: front, scan: si as u32, cells: 0, support: 0, across, surface: classes[si][front as usize], energy: 0.0,
                    });
                    if row.across == 0.0 {
                        row.across = across;
                    }
                    if ly.f > 0.5 {
                        row.cells += 1;
                        let sd = (em.sigma_e[row.surface as usize].powi(2) + ly.spread * ly.spread).sqrt();
                        let obs = if v.state[*cell] == state::ECHO { v.range[*cell] as f64 } else { f64::NAN };
                        row.support += (cell_code(obs, ly.t, sd) == code::ON) as u32;
                    }
                }
            }
        }
        for (key, acc) in &groups {
            // the offset's prior over the nodes, then −log ∫ p(cells | δ) p(δ) dδ against the empty scene
            let lp: Vec<f64> = (0..NODES).map(|g| t_pdf(node(g), em.nu_l).ln()).collect();
            let z = logsumexp(&lp);
            let post: Vec<f64> = (0..NODES).map(|g| lp[g] - z + acc.ll[g]).collect();
            let e = -(logsumexp(&post) - acc.ll0);
            total += sign * e;
            if skip.is_none() && only.is_none() {
                let best = (0..NODES).max_by(|&a, &b| post[a].total_cmp(&post[b])).unwrap();
                let n: f64 = acc.members.values().map(|m| m.0).sum();
                let shares: f64 = acc.members.values().map(|m| -m.1[best]).sum();
                for (&el, m) in &acc.members {
                    if let Some(r) = rows.get_mut(&(el, key.0)) {
                        r.energy += -m.1[best] + (e - shares) * m.0 / n;
                    }
                }
            }
        }
    }
    let mut rows: Vec<Row> = rows.into_values().collect();
    rows.sort_by_key(|r| (r.elem, r.scan));
    (total, rows)
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
/// The stick's edge lies `R + z_τ sd` from the axis, where sd is the fill's (footprint and misfit) and z_τ puts the
/// fill at `edge_fill` (τ). Its spread is the misfit, the offset across the beam and a third of the fill sd. Each
/// strip counts `cell / footprint` (neighbouring strips share footprints).
fn edge_ll(v: &View, cells: &[(usize, Layer)], e: &Element, k: u8, zt: f64, sd_lat: f64, em: &ErrorModel) -> f64 {
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
        let xe = e.r + zt * sdf;
        let se = (e.misfit * e.misfit + sd_lat * sd_lat + (sdf / 3.0).powi(2)).sqrt();
        let w = (em.step * t / fp).min(1.0);
        for side in [1.0, -1.0] {
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

/// Offset node g, in sds: −4 … 4.
fn node(g: usize) -> f64 {
    -4.0 + 8.0 * g as f64 / (NODES - 1) as f64
}

/// One scan's offset spread (m) for surface class k: half the between-view variance within an epoch, and half
/// the cross-epoch excess when the views span epochs.
fn sd_of(em: &ErrorModel, k: u8, epochs: bool) -> f64 {
    let [same, other] = em.sigma_l[k as usize];
    let var = 0.5 * same * same + if epochs { 0.5 * (other * other - same * same) } else { 0.0 };
    var.sqrt()
}

fn logsumexp(x: &[f64]) -> f64 {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
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

    /// Term C censors an edge whose pixels were deleted rather than reading it as absent, and skips one seen
    /// against something nearer: with the +x edge deleted, the true radius still scores better in C than a stick a
    /// third thinner; with it occluded, C keeps only the other side's evidence (about half).
    #[test]
    fn deleted_edges_are_censored_and_occluded_ones_skipped() {
        let em = ErrorModel::default();
        let zt = phi_inv(1.0 - em.edge_fill);
        let c_of = |v: &View, s: &Element| edge_ll(v, &render(v, s, 0, &em), s, 0, zt, 0.001, &em);
        let (full, deleted, occluded) = (pole_view(0), pole_view(1), pole_view(2));
        let truth = stick(0.0, 4.0, 0.03);
        let c_full = c_of(&full, &truth);
        assert!(c_full > 0.0, "{c_full}");
        assert!(c_of(&deleted, &truth) > c_of(&deleted, &stick(0.0, 4.0, 0.02)));
        let ratio = c_of(&occluded, &truth) / c_full;
        assert!((0.3..0.7).contains(&ratio), "{ratio}");
    }

    /// The offset's spread doubles its variance beyond the same-day part when the scans span epochs.
    #[test]
    fn cross_epoch_offsets_are_wider() {
        let em = ErrorModel::default();
        for k in 0..3u8 {
            let [same, other] = em.sigma_l[k as usize];
            assert!((sd_of(&em, k, false) - same / 2f64.sqrt()).abs() < 1e-12);
            assert!((sd_of(&em, k, true) - other / 2f64.sqrt()).abs() < 1e-12);
        }
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
