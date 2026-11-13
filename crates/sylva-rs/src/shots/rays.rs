// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The ray likelihood of stick hypotheses, minimal (3dtrees research 130, term A, slice 1): the energy of a set
//! of sticks against some scans' [`View`]s, as `-log` of its likelihood ratio against the empty scene, and the
//! energy change of adding or removing one stick.
//!
//! - **The k-buffer:** each stick renders into each view's cells. A cell keeps up to [`K`] layers, nearest first.
//!   Each layer has a depth, a depth spread, a fill and a cosine. Fill is the footprint convolved with the section
//!   across the axis. The footprint is a Gaussian with sd = footprint / 4. Along the axis the stick is cut square,
//!   so a chain of sticks joins without seams. The layer's depth and spread are those of the section's front over
//!   the covered part of the footprint, so the edge offset τ is derived, not fitted.
//! - **One cell:** its first exported echo, or none, against three things in depth order:
//!   - a diffuse exported-echo hazard κ along the ray (the crude field);
//!   - the layers;
//!   - a ground plane (the crude ground).
//!
//!   The beam reaches layer i with energy `e_i = f_i Π_{j<i} (1 − f_j)`. Its echo is kept with
//!   `sel_i = (1 − p_drop) Φ((ρ_dB + 10 log10(e_i cos ι_i) − cut) / σ_refl)`: the export's deletion and cut, as a
//!   selection on fill. An echo at r has density
//!   `Σ_i P_i sel_i t_ν(r − t_i − bias_k − δ; σ_k ⊕ spread_i) + κ e^{−κr} Π_{t_j<r} (1 − sel_j)`, where P_i is the
//!   chance that nothing nearer was kept, plus a uniform outlier share. An empty cell has probability
//!   `e^{−κ t_end} Π (1 − sel)`.
//! - **The offset:** a cell belongs to the (scan, block) of its nearest stick. Each group shares one offset δ along
//!   the beam, the scan's offset field there. δ is drawn `t_ν_L(0, σ_L[k] / √2)`, with the cross-epoch part added
//!   when the scans span epochs, and integrated once per group by quadrature. The surface class k, the latent that
//!   sets the spread, the bias and the noise, is each stick's own. The caller gives it, or it defaults to the
//!   median class of the view's echoes on the stick's front.
//! - **Weights:** each cell counts `1/m(ρ)`, with m the footprints per cell (overlapping footprints), over the
//!   design effect `1 + (n − 1) ρ` of its nearest stick, n that stick's fill-weighted footprints in the scan
//!   (correlated misfit; the element is the unit), so one scan is worth at most about `1/ρ` independent
//!   footprints on a stick, whatever its size or range.
//!
//! Cells no stick touches cancel. Context (neighbours, competitors) enters as more sticks in the set.

use super::view::{state, ErrorModel, View};
use crate::transform::{dot, norm, sub};
use crate::Point;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Layers kept per cell. A cell that runs out (rare among a few sticks) drops its farthest.
pub const K: usize = 3;
/// Quadrature nodes over the offset, spanning ±4 sd.
const NODES: usize = 21;

/// A stick: an axis from `a` to `b` (world, m), a radius, its surface class (0-2, or negative for "from the view")
/// and its opacity: the chance a pulse its section fills meets solid there, its gaps wider than the footprint (1
/// solid wood, less for a porous clump; synthesis §1's "extent, opacity"). A full-fill cell keeps an echo from it
/// with `opacity × sel`, and passes `1 − opacity f` of the beam on. An element property, a latent the caller
/// supplies.
#[derive(Debug, Clone, Copy)]
pub struct Stick {
    pub a: Point,
    pub b: Point,
    pub r: f64,
    pub surface: i8,
    pub opacity: f64,
}

/// The crude scene behind the sticks: the diffuse exported-echo hazard (1/m), a ground plane's height (NaN for
/// none), the sticks' and the ground's reflectance (dB at full fill, face-on) and the longest range.
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

#[derive(Debug, Clone, Copy)]
struct Layer {
    t: f64,
    spread: f64,
    f: f64,
    cos: f64,
    stick: u32,
    /// The stick's opacity: the blocked share of the fill is `f op`.
    op: f64,
    /// Where the cell's ray passes the axis: across it (signed, m, positive on `u × (c − o)`'s side) and along it.
    x: f64,
    s: f64,
}

/// What one scan says about one stick. `cells`: cells where it is the nearest layer at fill > 0.5; `support`:
/// those with an echo within 3 sd of it; `across`: footprints across its diameter; `energy`: its share of its
/// groups' energy (its cells' terms at the group's most likely offset, plus the group's remainder split by
/// cells), so the rows sum to the total.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub stick: u32,
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

/// One stick's layers in one view: (cell, layer). With edges on, cells out to 3 fill sds and two cells past the
/// section come too (their fill may be ~0), so term C sees what lies beyond each edge.
fn render(v: &View, s: &Stick, id: u32, em: &ErrorModel) -> Vec<(usize, Layer)> {
    let mut out = Vec::new();
    let ax = sub(&s.b, &s.a);
    let len = norm(&ax);
    if !(len > 0.0) || !(s.r > 0.0) {
        return out;
    }
    let u = ax.map(|x| x / len);
    let half = 0.5 * len;
    let mid = [0, 1, 2].map(|k| 0.5 * (s.a[k] + s.b[k]));
    let c = sub(&mid, &v.origin);
    let dist = norm(&c);
    let Some((l0, k0, _)) = v.cell_of(&mid) else { return out };
    let alpha = (half * half + (s.r + em.footprint(dist)).powi(2)).sqrt() / dist;
    let sin_z = ((c[0] * c[0] + c[1] * c[1]).sqrt() / dist).max(0.05);
    let (zs, az) = (v.lat.zenith_step.abs().to_radians(), v.lat.azimuth_step.abs().to_radians());
    let dl = (alpha / (az * sin_z)).min(4000.0).ceil() as i64 + 1;
    let dc = (alpha / zs).min(4000.0).ceil() as i64 + 1;
    let cu = dot(&c, &u);
    let bv = [0, 1, 2].map(|k| -(c[k] - cu * u[k]));
    let bb = dot(&bv, &bv);
    let side = crate::transform::normalize(&crate::transform::cross(&u, &c));
    let sm = em.misfit_sd(s.r);
    let (li, ki) = (l0.round() as i64, k0.round() as i64);
    for l in (li - dl).max(0)..=(li + dl).min(v.n_lines as i64 - 1) {
        for k in (ki - dc).max(0)..=(ki + dc).min(v.n_cols as i64 - 1) {
            let w = v.direction(l as f64, k as f64);
            let wu = dot(&w, &u);
            let a = [0, 1, 2].map(|j| w[j] - wu * u[j]);
            let aa = dot(&a, &a);
            if aa < 1e-12 {
                continue;
            }
            let ab = dot(&a, &bv);
            let tm = -ab / aa;
            let along = tm * wu - cu;
            if tm <= 0.0 || along.abs() > half {
                continue;
            }
            let d = (bb - ab * ab / aa).max(0.0).sqrt();
            let sd = (0.0625 * em.footprint(tm).powi(2) + sm * sm).sqrt();
            let f = phi((s.r - d) / sd) - phi((-s.r - d) / sd);
            if f < 0.01 && !(em.edges && d < s.r + 3.0 * sd + 2.0 * em.step * tm) {
                continue;
            }
            let x = dot(&[0, 1, 2].map(|j| tm * w[j] - c[j] - along * u[j]), &side);
            let sin = aa.sqrt();
            let depth = |x: f64| tm - (s.r * s.r - x * x).max(0.0).sqrt() / sin;
            let x2 = (d + sd).min(s.r);
            let x1 = (d - sd).max(0.0).min(x2);
            let xm = 0.5 * (x1 + x2);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t: 0.5 * (depth(x1) + depth(x2)),
                    spread: ((depth(x2) - depth(x1)).powi(2) / 12.0 + sm * sm).sqrt(),
                    f,
                    cos: ((1.0 - (xm / s.r).powi(2)).max(0.0).sqrt() * sin).max(0.05),
                    stick: id,
                    op: s.opacity,
                    x,
                    s: along,
                },
            ));
        }
    }
    out
}

/// Every stick rendered into one view.
fn render_all(v: &View, sticks: &[Stick], em: &ErrorModel) -> Vec<Vec<(usize, Layer)>> {
    sticks.par_iter().enumerate().map(|(i, s)| render(v, s, i as u32, em)).collect()
}

/// The k-buffer of one view: per touched cell, its layers nearest first. With edges on, the fill band under 0.5
/// is term C's, not A's.
fn kbuffer(per: &[Vec<(usize, Layer)>], em: &ErrorModel) -> HashMap<usize, Vec<Layer>> {
    let min_f = if em.edges { 0.5 } else { 0.01 };
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

/// The probability an echo of energy share `e` (fill × cos) off a surface of `db` survives the export.
fn sel(drop: f64, em: &ErrorModel, db: f64, e: f64) -> f64 {
    (1.0 - drop) * phi((db + 10.0 * e.max(1e-12).log10() - em.cut_db) / em.refl_noise_db)
}

/// Each stick's surface class in one view: its own, or the median class of the view's echoes on its front.
fn classes(v: &View, buf: &HashMap<usize, Vec<Layer>>, sticks: &[Stick], em: &ErrorModel) -> Vec<u8> {
    let mut ex: Vec<Vec<f64>> = vec![Vec::new(); sticks.len()];
    for (&cell, ls) in buf {
        let ly = ls[0];
        let r = v.range[cell] as f64;
        if ly.f > 0.5 && v.state[cell] == state::ECHO && (r - ly.t).abs() < 0.03 && v.excess[cell].is_finite() {
            ex[ly.stick as usize].push(v.excess[cell] as f64);
        }
    }
    sticks
        .iter()
        .zip(ex.iter_mut())
        .map(|(s, e)| {
            if s.surface >= 0 {
                (s.surface as u8).min(2)
            } else if e.is_empty() {
                1
            } else {
                e.sort_by(f64::total_cmp);
                em.surface(e[e.len() / 2])
            }
        })
        .collect()
}

/// What the sticks predict in one view, per touched cell: its nearest layer (stick, depth, spread, fill), that
/// stick's class, the chance its echo is kept (the selection term on its fill), the echo's predicted depth (bias
/// added) and sd (noise, spread and one scan's offset spread), and the observed first echo (NaN for none). For
/// held-out checks: the cell arrays term A scores, without the scene.
pub struct Prediction {
    pub cell: usize,
    pub stick: u32,
    pub fill: f64,
    pub kept: f64,
    pub depth: f64,
    pub sd: f64,
    pub surface: u8,
    pub obs: f64,
    /// The stick's misfit sd (m), included in `sd`.
    pub misfit: f64,
}

pub fn predict(v: &View, sticks: &[Stick], em: &ErrorModel, sc: &Scene) -> Vec<Prediction> {
    let buf = kbuffer(&render_all(v, sticks, em), &ErrorModel { edges: false, ..*em });
    let cls = classes(v, &buf, sticks, em);
    let mut out: Vec<Prediction> = buf
        .iter()
        .map(|(&cell, ls)| {
            let ly = ls[0];
            let k = cls[ly.stick as usize];
            let [same, _] = em.sigma_l[k as usize];
            Prediction {
                cell, stick: ly.stick, fill: ly.f, kept: ly.op * sel(drop_of(em, v), em, sc.wood_db, ly.f * ly.cos), surface: k,
                depth: ly.t + em.depth_bias(k, ly.t),
                sd: (em.sigma_e[k as usize].powi(2) + ly.spread.powi(2) + 0.5 * same * same).sqrt(),
                misfit: em.misfit_sd(sticks[ly.stick as usize].r),
                obs: if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN },
            }
        })
        .collect();
    out.sort_by_key(|p| p.cell);
    out
}

/// One cell's log-likelihood: its first echo `obs` (NaN for none) against its layers (each with its stick's
/// surface class), the ground at `ground` = (range, cos) and the field, with every layer shifted by `delta`.
/// The export's in-surface loss for a view: the error model's, or the view's measured one.
fn drop_of(em: &ErrorModel, v: &View) -> f64 {
    if em.p_drop.is_nan() { v.deletion } else { em.p_drop }
}

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
            let g = em.ghost * (1.0 - ly.f);
            let ghost = if obs > mu && obs < mu + em.ghost_depth { g / em.ghost_depth } else { 0.0 };
            dens += (-sc.kappa * ly.t).exp() * none_kept * s * ((1.0 - g) * t_pdf((obs - mu) / sd, em.nu_e) / sd + ghost);
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

/// The energy of `sticks` against `views` (−log likelihood ratio against the empty scene), and its rows. With
/// `only = Some(i)`, the energy change of adding stick i to the others, `E(all) − E(all but i)`, over the groups
/// it touches; rows are then empty.
pub fn energy(views: &[&View], sticks: &[Stick], em: &ErrorModel, sc: &Scene, only: Option<usize>) -> (f64, Vec<Row>) {
    let rendered: Vec<Vec<Vec<(usize, Layer)>>> = views.iter().map(|v| render_all(v, sticks, em)).collect();
    let bufs: Vec<HashMap<usize, Vec<Layer>>> = rendered.iter().map(|r| kbuffer(r, em)).collect();
    let classes: Vec<Vec<u8>> = views.iter().zip(&bufs).map(|(v, buf)| classes(v, buf, sticks, em)).collect();
    let epochs: HashSet<u32> = views.iter().map(|v| v.epoch).collect();
    // each stick's design effect per view, from its fill-weighted footprints
    let temper: Vec<Vec<f64>> = bufs
        .iter()
        .map(|buf| {
            let mut n = vec![0.0; sticks.len()];
            for ls in buf.values() {
                for l in ls {
                    n[l.stick as usize] += l.f / em.multiplicity(l.t);
                }
            }
            n.iter().map(|&n| 1.0 / (1.0 + (n - 1.0f64).max(0.0) * em.rho)).collect()
        })
        .collect();
    let block = |i: u32| {
        let s = &sticks[i as usize];
        [0, 1, 2].map(|k| (0.5 * (s.a[k] + s.b[k]) / em.block).floor() as i64)
    };
    // per cell, its group in a configuration that leaves out `skip`
    let key_of = |si: usize, ls: &[Layer], skip: Option<u32>| ls.iter().find(|l| Some(l.stick) != skip).map(|l| (si as u32, block(l.stick)));
    let skip_i = only.map(|i| i as u32);
    let touched: Option<HashSet<Key>> = skip_i.map(|i| {
        let mut g = HashSet::new();
        for (si, buf) in bufs.iter().enumerate() {
            for ls in buf.values() {
                if ls.iter().any(|l| l.stick == i) {
                    g.extend(key_of(si, ls, None));
                    g.extend(key_of(si, ls, Some(i)));
                }
            }
        }
        g
    });
    // the configurations to score: all sticks (+1), and without stick i (−1)
    let configs: Vec<(Option<u32>, f64)> = match skip_i {
        None => vec![(None, 1.0)],
        Some(i) => vec![(None, 1.0), (Some(i), -1.0)],
    };
    let mut total = 0.0;
    let mut rows: HashMap<(u32, u32), Row> = HashMap::new();
    // term C per (view, stick), tempered as A; it doesn't see the offset (it lies across the beam)
    let c_terms: Vec<Vec<f64>> = if em.edges {
        let zt = phi_inv(1.0 - em.edge_fill);
        views
            .iter()
            .enumerate()
            .map(|(si, v)| {
                (0..sticks.len())
                    .into_par_iter()
                    .map(|h| {
                        let k = classes[si][h];
                        temper[si][h] * edge_ll(v, &rendered[si][h], &sticks[h], k, zt, sd_of(em, k, epochs.len() > 1), em)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![vec![0.0; sticks.len()]; views.len()]
    };
    for (si, cs) in c_terms.iter().enumerate() {
        for (h, &c) in cs.iter().enumerate() {
            match only {
                Some(i) if i != h => {}
                _ => total -= c,
            }
            if only.is_none() && c != 0.0 {
                rows.entry((h as u32, si as u32))
                    .or_insert(Row { stick: h as u32, scan: si as u32, cells: 0, support: 0, across: 0.0, surface: classes[si][h], energy: 0.0 })
                    .energy -= c;
            }
        }
    }
    for &(skip, sign) in &configs {
        // per group: the weighted log-likelihood at each offset node, the empty scene's, and its sticks' shares
        struct Acc {
            ll: [f64; NODES],
            ll0: f64,
            members: HashMap<u32, (f64, [f64; NODES])>,
        }
        let mut groups: HashMap<Key, Acc> = HashMap::new();
        let centres: HashMap<Key, (f64, f64)> =
            if em.medial_offset { medial(views, &bufs, sticks, &classes, em, epochs.len() > 1, &|ls: &[Layer]| ls.iter().find(|l| Some(l.stick) != skip).copied(), &|si, ls| key_of(si, ls, skip)) } else { HashMap::new() };
        for (si, (v, buf)) in views.iter().zip(&bufs).enumerate() {
            let cells: Vec<(usize, &Vec<Layer>)> = buf.iter().map(|(&c, ls)| (c, ls)).collect();
            let terms: Vec<Option<(Key, u32, f64, [f64; NODES])>> = cells
                .par_iter()
                .map(|&(cell, ls)| {
                    let key = key_of(si, ls, skip)?;
                    if touched.as_ref().is_some_and(|t| !t.contains(&key)) {
                        return None;
                    }
                    let layers: Vec<(Layer, u8)> = ls.iter().filter(|l| Some(l.stick) != skip).map(|l| (*l, classes[si][l.stick as usize])).collect();
                    let front = layers[0].0;
                    let obs = if v.state[cell] == state::ECHO { v.range[cell] as f64 } else { f64::NAN };
                    let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                    let ground = (sc.ground_z.is_finite() && w[2] < -1e-6).then(|| ((sc.ground_z - v.origin[2]) / w[2], -w[2]));
                    let wt = temper[si][front.stick as usize] / em.multiplicity(if obs.is_finite() { obs } else { front.t });
                    let (c0, sd) = centres.get(&key).copied().unwrap_or((0.0, sd_of(em, classes[si][front.stick as usize], epochs.len() > 1)));
                    let ll: [f64; NODES] = std::array::from_fn(|g| wt * cell_ll(obs, &layers, ground, c0 + node(g) * sd, drop_of(em, v), em, sc));
                    Some((key, front.stick, wt * cell_ll(obs, &[], ground, 0.0, drop_of(em, v), em, sc), ll))
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
                    let row = rows.entry((front, si as u32)).or_insert(Row {
                        stick: front, scan: si as u32, cells: 0, support: 0,
                        across: 2.0 * sticks[front as usize].r / em.footprint(ly.t),
                        surface: classes[si][front as usize], energy: 0.0,
                    });
                    if row.across == 0.0 {
                        row.across = 2.0 * sticks[front as usize].r / em.footprint(ly.t);
                    }
                    if ly.f > 0.5 {
                        row.cells += 1;
                        let sd = (em.sigma_e[row.surface as usize].powi(2) + ly.spread * ly.spread).sqrt();
                        if v.state[*cell] == state::ECHO && (v.range[*cell] as f64 - ly.t).abs() < 3.0 * sd + 0.005 {
                            row.support += 1;
                        }
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
                for (&st, m) in &acc.members {
                    if let Some(r) = rows.get_mut(&(st, key.0)) {
                        r.energy += -m.1[best] + (e - shares) * m.0 / n;
                    }
                }
            }
        }
    }
    let mut rows: Vec<Row> = rows.into_values().collect();
    rows.sort_by_key(|r| (r.stick, r.scan));
    (total, rows)
}

/// Term C for one stick in one view: the log likelihood ratio of its located silhouette edges against an edge
/// placed evenly across the rendered band.
///
/// Its cells are cut into strips along the axis, a cell wide. In each strip with echoes on the stick, each side's
/// outermost such echo and the next cell out locate the edge:
/// - **sharp** (the next cell's echo lies behind): the edge lies between the two;
/// - **unknown** (the next cell is empty, its echo deleted): censored, the edge lies beyond the last echo (Tobit);
/// - **occluded** (the next cell's echo lies in front): no information.
///
/// The stick's edge lies `R + z_τ sd` from the axis, where sd is the fill's (footprint and misfit) and z_τ puts
/// the fill at `edge_fill` (τ). Its spread is the misfit, the offset across the beam and a third of the fill sd.
/// Each strip counts `cell / footprint` (neighbouring strips share footprints).
fn edge_ll(v: &View, cells: &[(usize, Layer)], s: &Stick, k: u8, zt: f64, sd_lat: f64, em: &ErrorModel) -> f64 {
    const ON: u8 = 0;
    const EMPTY: u8 = 1;
    const FRONT: u8 = 2;
    const BEHIND: u8 = 3;
    let sm = em.misfit_sd(s.r);
    let mut strips: HashMap<i64, Vec<(f64, u8, f64)>> = HashMap::new();
    for &(cell, ly) in cells {
        let cw = em.step * ly.t;
        let obs = v.range[cell] as f64;
        let sd = (em.sigma_e[k as usize].powi(2) + ly.spread.powi(2)).sqrt();
        let st = if v.state[cell] != state::ECHO {
            EMPTY
        } else if obs < ly.t - 3.0 * sd - 0.03 {
            FRONT
        } else if (obs - ly.t - em.depth_bias(k, ly.t)).abs() < 3.0 * sd + 0.01 {
            ON
        } else {
            BEHIND
        };
        strips.entry((ly.s / cw).floor() as i64).or_default().push((ly.x, st, ly.t));
    }
    let mut ll = 0.0;
    for cs in strips.values() {
        let Some(t) = cs.iter().find(|c| c.1 == ON).map(|c| c.2) else { continue };
        let fp = em.footprint(t);
        let sdf = (0.0625 * fp * fp + sm * sm).sqrt();
        let xe = s.r + zt * sdf;
        let se = (sm * sm + sd_lat * sd_lat + (sdf / 3.0).powi(2)).sqrt();
        let w = (em.step * t / fp).min(1.0);
        for side in [1.0, -1.0] {
            let x_in = cs.iter().filter(|c| c.1 == ON).map(|c| side * c.0).fold(f64::NEG_INFINITY, f64::max);
            let far = cs.iter().map(|c| side * c.0).fold(0.0, f64::max);
            let Some(&(xo, st, _)) = cs.iter().filter(|c| side * c.0 > x_in).min_by(|a, b| (side * a.0).total_cmp(&(side * b.0))) else { continue };
            let x_out = side * xo;
            if far <= 0.0 || x_in < 0.0 {
                continue;
            }
            let (p, bg) = match st {
                BEHIND => (phi((xe - x_in) / se) - phi((xe - x_out) / se), (x_out - x_in) / far),
                EMPTY => (phi((xe - x_in) / se), (far - x_in) / far),
                _ => continue,
            };
            ll += w * (p.max(1e-12).ln() - bg.max(1e-6).ln());
        }
    }
    ll
}

/// Each group's offset centred on its medial estimate (synthesis §13): for the echoes on a cell's nearest stick,
/// with a normal, the medial point `p − R n` lies on the axis when the scan has no offset, so its mean distance
/// behind the axis along the beam estimates the offset. Returned with the spread left once that estimate (its
/// standard error) is combined with the scan's prior spread.
fn medial(views: &[&View], bufs: &[HashMap<usize, Vec<Layer>>], sticks: &[Stick], classes: &[Vec<u8>], em: &ErrorModel, epochs: bool,
          front_of: &dyn Fn(&[Layer]) -> Option<Layer>, key_of: &dyn Fn(usize, &[Layer]) -> Option<Key>) -> HashMap<Key, (f64, f64)> {
    let mut acc: HashMap<Key, (f64, f64, f64, f64)> = HashMap::new();
    for (si, (v, buf)) in views.iter().zip(bufs).enumerate() {
        for (&cell, ls) in buf {
            let (Some(ly), Some(key)) = (front_of(ls), key_of(si, ls)) else { continue };
            let obs = v.range[cell] as f64;
            let n = v.normal[cell].map(|x| x as f64);
            if v.state[cell] != state::ECHO || !n[0].is_finite() || (obs - ly.t).abs() > 0.02 {
                continue;
            }
            let st = &sticks[ly.stick as usize];
            let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
            let ax = sub(&st.b, &st.a);
            let u = ax.map(|x| x / norm(&ax));
            let m = [0, 1, 2].map(|j| v.origin[j] + w[j] * obs - st.r * n[j] - st.a[j]);
            let mu = dot(&m, &u);
            let wu = dot(&w, &u);
            let wp = [0, 1, 2].map(|j| w[j] - wu * u[j]);
            let d = dot(&[0, 1, 2].map(|j| m[j] - mu * u[j]), &wp) / norm(&wp);
            let e = acc.entry(key).or_insert((0.0, 0.0, 0.0, sd_of(em, classes[si][ly.stick as usize], epochs)));
            let wt = 1.0 / em.multiplicity(obs);
            e.0 += wt * d;
            e.1 += wt * d * d;
            e.2 += wt;
        }
    }
    acc.into_iter()
        .filter(|(_, a)| a.2 >= 1.0)
        .map(|(k, (s1, s2, n, prior))| {
            let mean = s1 / n;
            let se2 = ((s2 / n - mean * mean).max(1e-8) / n).max(2.5e-7);
            let g = prior * prior / (prior * prior + se2);
            (k, (g * mean, (g * se2).sqrt()))
        })
        .collect()
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

/// One scan's offset spread (m) for surface class k.
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

    /// A scan of a vertical pole (R = 3 cm, 4 m away, centred on the y axis) in front of a wall at 8 m. The true
    /// stick has negative energy; a stick of the wrong radius, or shifted sideways, has more; adding a stick where
    /// the pulses went through to the wall costs energy; and `only` agrees with the difference of two energies.
    #[test]
    fn the_true_pole_wins() {
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
                let range = if disc > 0.0 { (-b - disc.sqrt()) / (2.0 * a) } else { 8.0 / w[1] };
                t.push(l as f64 * period + k as f64 * interval);
                d.push(w);
                r.push(range as f32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = View::new(&t, &d, &r, id, [0.0; 3], 0, &ViewParams::default());
        let (em, sc) = (ErrorModel::default(), Scene::default());
        let stick = |x: f64, y: f64, r: f64| Stick { a: [x, y, -0.2], b: [x, y, 0.2], r, surface: 0, opacity: 1.0 };
        let e = |s: Stick| energy(&[&v], &[s], &em, &sc, None).0;
        let truth = e(stick(0.0, y_pole, r_pole));
        assert!(truth < -3.0, "{truth}");
        for wrong in [stick(0.0, y_pole, 0.021), stick(0.0, y_pole, 0.015), stick(0.0, y_pole, 0.06), stick(0.03, y_pole, r_pole), stick(0.0, y_pole - 0.02, r_pole)] {
            assert!(e(wrong) > truth + 1.0, "{} vs {truth}", e(wrong));
        }
        assert!(e(stick(0.2, 6.0, 0.03)) > 1.0);
        // a porous stick explains a solid pole worse than a solid one does
        assert!(e(Stick { opacity: 0.3, ..stick(0.0, y_pole, r_pole) }) > truth + 1.0);
        let pair = [stick(0.0, y_pole, r_pole), stick(0.2, 6.0, 0.03)];
        let both = energy(&[&v], &pair, &em, &sc, None);
        let delta = energy(&[&v], &pair, &em, &sc, Some(1)).0;
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let sum: f64 = both.1.iter().map(|r| r.energy).sum();
        assert!((sum - both.0).abs() < 1e-6);
    }
}
