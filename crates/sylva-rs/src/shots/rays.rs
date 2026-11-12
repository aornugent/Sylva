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

/// A stick: an axis from `a` to `b` (world, m), a radius, and its surface class (0-2, or negative for "from the
/// view").
#[derive(Debug, Clone, Copy)]
pub struct Stick {
    pub a: Point,
    pub b: Point,
    pub r: f64,
    pub surface: i8,
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

/// One stick's layers in one view: (cell, layer).
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
            if tm <= 0.0 || (tm * wu - cu).abs() > half {
                continue;
            }
            let d = (bb - ab * ab / aa).max(0.0).sqrt();
            let sd = 0.25 * em.footprint(tm);
            let f = phi((s.r - d) / sd) - phi((-s.r - d) / sd);
            if f < 0.01 {
                continue;
            }
            let sin = aa.sqrt();
            let depth = |x: f64| tm - (s.r * s.r - x * x).max(0.0).sqrt() / sin;
            let x2 = (d + sd).min(s.r);
            let x1 = (d - sd).max(0.0).min(x2);
            let xm = 0.5 * (x1 + x2);
            out.push((
                l as usize * v.n_cols + k as usize,
                Layer {
                    t: 0.5 * (depth(x1) + depth(x2)),
                    spread: (depth(x2) - depth(x1)) / 12f64.sqrt(),
                    f,
                    cos: ((1.0 - (xm / s.r).powi(2)).max(0.0).sqrt() * sin).max(0.05),
                    stick: id,
                },
            ));
        }
    }
    out
}

/// The k-buffer of one view: per touched cell, its layers nearest first.
fn kbuffer(v: &View, sticks: &[Stick], em: &ErrorModel) -> HashMap<usize, Vec<Layer>> {
    let per: Vec<Vec<(usize, Layer)>> = sticks.par_iter().enumerate().map(|(i, s)| render(v, s, i as u32, em)).collect();
    let mut buf: HashMap<usize, Vec<Layer>> = HashMap::new();
    for (cell, ly) in per.into_iter().flatten() {
        buf.entry(cell).or_default().push(ly);
    }
    for ls in buf.values_mut() {
        ls.sort_by(|a, b| a.t.total_cmp(&b.t));
        ls.truncate(K);
    }
    buf
}

/// One cell's log-likelihood: its first echo `obs` (NaN for none) against its layers (each with its stick's
/// surface class), the ground at `ground` = (range, cos) and the field, with every layer shifted by `delta`.
fn cell_ll(obs: f64, layers: &[(Layer, u8)], ground: Option<(f64, f64)>, delta: f64, em: &ErrorModel, sc: &Scene) -> f64 {
    let sel = |db: f64, e: f64| (1.0 - em.p_drop) * phi((db + 10.0 * e.max(1e-12).log10() - em.cut_db) / em.refl_noise_db);
    let t_end = ground.map_or(sc.t_max, |g| g.0);
    let (mut beam, mut none_kept, mut none_before_obs, mut dens) = (1.0, 1.0, 1.0, 0.0);
    for &(ly, k) in layers {
        if ly.t >= t_end {
            break;
        }
        let e = ly.f * beam;
        beam *= 1.0 - ly.f;
        let s = sel(sc.wood_db, e * ly.cos);
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
        let s = sel(sc.ground_db, beam * cg);
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
    let bufs: Vec<HashMap<usize, Vec<Layer>>> = views.iter().map(|v| kbuffer(v, sticks, em)).collect();
    // each stick's surface class per view
    let classes: Vec<Vec<u8>> = views
        .iter()
        .zip(&bufs)
        .map(|(v, buf)| {
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
        })
        .collect();
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
    for &(skip, sign) in &configs {
        // per group: the weighted log-likelihood at each offset node, the empty scene's, and its sticks' shares
        struct Acc {
            ll: [f64; NODES],
            ll0: f64,
            members: HashMap<u32, (f64, [f64; NODES])>,
        }
        let mut groups: HashMap<Key, Acc> = HashMap::new();
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
                    let sd = sd_of(em, classes[si][front.stick as usize], epochs.len() > 1);
                    let ll: [f64; NODES] = std::array::from_fn(|g| wt * cell_ll(obs, &layers, ground, node(g) * sd, em, sc));
                    Some((key, front.stick, wt * cell_ll(obs, &[], ground, 0.0, em, sc), ll))
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
        let stick = |x: f64, y: f64, r: f64| Stick { a: [x, y, -0.2], b: [x, y, 0.2], r, surface: 0 };
        let e = |s: Stick| energy(&[&v], &[s], &em, &sc, None).0;
        let truth = e(stick(0.0, y_pole, r_pole));
        assert!(truth < -3.0, "{truth}");
        for wrong in [stick(0.0, y_pole, 0.015), stick(0.0, y_pole, 0.06), stick(0.03, y_pole, r_pole), stick(0.0, y_pole - 0.02, r_pole)] {
            assert!(e(wrong) > truth + 1.0, "{} vs {truth}", e(wrong));
        }
        assert!(e(stick(0.2, 6.0, 0.03)) > 1.0);
        let pair = [stick(0.0, y_pole, r_pole), stick(0.2, 6.0, 0.03)];
        let both = energy(&[&v], &pair, &em, &sc, None);
        let delta = energy(&[&v], &pair, &em, &sc, Some(1)).0;
        assert!((delta - (both.0 - truth)).abs() < 1e-6, "{delta} vs {}", both.0 - truth);
        let sum: f64 = both.1.iter().map(|r| r.energy).sum();
        assert!((sum - both.0).abs() < 1e-6);
    }
}
