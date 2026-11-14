// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The diffuse part of the scene (3dtrees research 130 slice 12; synthesis §2, §15): what the explicit elements
//! don't explain, as a field of mean extinction σ (1/m) on voxels in a box.
//!
//! - **Its form.** Inside the box, σ is constant per voxel and the same in every direction: one coefficient a voxel,
//!   since a voxel is seen from a handful of near-horizontal directions (synthesis §15). Leaf area and
//!   sub-resolution wood both fold into it (σ alone can't separate them). Outside the box, and where the box has no
//!   data, σ is the background's κ. A ray's survival to range t is `exp(−∫₀ᵗ σ)` and its first echo's density at t
//!   is `σ(t) exp(−∫₀ᵗ σ)`, so a miss is evidence of transmission.
//! - **Its estimate.** Each voxel's weighted counts, echoes in it (h) and ray length through it (L), give the
//!   posterior mean `(h + κ b) / (L + b)` under a Gamma prior worth `b` metres of ray at mean κ.
//! - **A pulse's echo sequence** (`multi_echo`): every echo is a hit, and the path up to it is transmission, from the
//!   origin for the first and from the previous echo plus the receiver's resolution after that (a surface closer
//!   behind an echo would have merged with it). After the last echo, the path on is transmission only if the pulse
//!   returned fewer echoes than the instrument records ([`RangeImage::max_targets`]): a pulse at its cap says
//!   nothing about what lies beyond. Without `multi_echo`, only the first echo and the path to it count. Each ray counts `1 / (φ m(ρ))`, as a cell does in the ray
//!   likelihood. The dispersion φ is a compound Poisson's clump size: mulga's leaf hits are clumped, and their
//!   dispersion is flat in cell size (synthesis §2).
//! - **A residual.** The field is estimated from the rays no element fronts, so an element and the field never
//!   explain the same echoes. Its log-likelihood over every ray, `Σ_v h ln σ_v − σ_v L` with all rays' counts, is
//!   the part of the scene's cost the elements' cells don't hold.

use super::model::{ErrorModel, RESOLUTION};
use super::range_image::{Cell, RangeImage};
use crate::Point;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Where the field is estimated: a box `lo`-`hi`, its voxel (m), the prior's worth `prior_path` (m of ray), and
/// whether each pulse's whole echo sequence counts (`multi_echo`) or its first echo only.
#[derive(Debug, Clone, Copy)]
pub struct ExtinctionSpec {
    pub lo: Point,
    pub hi: Point,
    pub voxel: f64,
    pub prior_path: f64,
    pub multi_echo: bool,
}

/// Weighted counts per voxel: (first echoes in it, ray length through it).
pub type Counts = HashMap<[i32; 3], (f64, f64)>;

/// The extinction field: σ per voxel of `spec`'s box, κ elsewhere.
#[derive(Debug, Clone)]
pub struct Extinction {
    pub kappa: f64,
    pub spec: Option<ExtinctionSpec>,
    pub sigma: HashMap<[i32; 3], f64>,
}

/// What a cell's ray meets of the field: the optical depth to each of its layers and then the ground (`taus`), to the
/// beam's end, and to its first echo, with σ at that echo.
#[derive(Debug, Clone)]
pub struct RayBg {
    pub taus: Vec<f64>,
    pub tau_end: f64,
    pub tau_obs: f64,
    pub sig_obs: f64,
}

impl RayBg {
    /// The same ray with its layers left out (the ground's optical depth kept).
    pub fn base(&self) -> RayBg {
        RayBg { taus: vec![*self.taus.last().unwrap_or(&0.0)], ..*self }
    }
}

impl Extinction {
    /// The background alone: κ everywhere.
    pub fn uniform(kappa: f64) -> Self {
        Extinction { kappa, spec: None, sigma: HashMap::new() }
    }

    /// The posterior mean field from counts.
    pub fn from_counts(kappa: f64, spec: ExtinctionSpec, counts: &Counts) -> Self {
        let b = spec.prior_path;
        let sigma = counts.iter().map(|(&k, &(h, l))| (k, (h + kappa * b) / (l + b))).collect();
        Extinction { kappa, spec: Some(spec), sigma }
    }

    /// σ at a point.
    pub fn sigma_at(&self, x: &Point) -> f64 {
        match self.spec {
            Some(sp) if inside(&sp, x) => *self.sigma.get(&key(&sp, x)).unwrap_or(&self.kappa),
            _ => self.kappa,
        }
    }

    /// The optical depth `∫₀ᵗ σ` along the ray from `o` in direction `w`.
    pub fn tau(&self, o: &Point, w: &Point, t: f64) -> f64 {
        let mut tau = self.kappa * t;
        if let Some(sp) = self.spec {
            traverse(&sp, o, w, 0.0, t, |k, a, b| tau += (self.sigma.get(&k).unwrap_or(&self.kappa) - self.kappa) * (b - a));
        }
        tau
    }

    /// What a cell's ray from `o` along `w` meets: optical depths to its layers' ranges, its ground's, the beam's end
    /// and its first echo `obs` (NaN for none).
    pub fn ray_bg(&self, o: &Point, w: &Point, layer_ts: &[f64], ground_t: Option<f64>, t_end: f64, obs: f64) -> RayBg {
        let mut taus: Vec<f64> = layer_ts.iter().map(|&t| self.tau(o, w, t)).collect();
        taus.push(ground_t.map_or(0.0, |t| self.tau(o, w, t)));
        let (tau_obs, sig_obs) = if obs.is_finite() {
            (self.tau(o, w, obs), self.sigma_at(&[0, 1, 2].map(|a| o[a] + w[a] * obs)))
        } else {
            (f64::NAN, f64::NAN)
        };
        RayBg { taus, tau_end: self.tau(o, w, t_end), tau_obs, sig_obs }
    }

    /// The field's log-likelihood over `counts`: `Σ_v h ln σ_v − σ_v L`.
    pub fn log_lik(&self, counts: &Counts) -> f64 {
        let mut keys: Vec<&[i32; 3]> = counts.keys().collect();
        keys.sort_unstable();
        keys.into_iter().map(|k| {
            let (h, l) = counts[k];
            let s = *self.sigma.get(k).unwrap_or(&self.kappa);
            h * s.ln() - s * l
        }).sum()
    }
}

fn inside(sp: &ExtinctionSpec, x: &Point) -> bool {
    (0..3).all(|a| x[a] >= sp.lo[a] && x[a] < sp.hi[a])
}

fn key(sp: &ExtinctionSpec, x: &Point) -> [i32; 3] {
    [0, 1, 2].map(|a| ((x[a] - sp.lo[a]) / sp.voxel).floor() as i32)
}

/// The ray's entry and exit ranges through the box (entry ≥ 0), if it crosses it.
pub fn clip(sp: &ExtinctionSpec, o: &Point, w: &Point) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
    for a in 0..3 {
        if w[a].abs() < 1e-12 {
            if o[a] < sp.lo[a] || o[a] >= sp.hi[a] {
                return None;
            }
            continue;
        }
        let (ta, tb) = ((sp.lo[a] - o[a]) / w[a], (sp.hi[a] - o[a]) / w[a]);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    (t1 > t0).then_some((t0, t1))
}

/// Each voxel segment of the ray inside the box between ranges `t_min` and `t_max`: `f(voxel, from, to)`, in order.
fn traverse(sp: &ExtinctionSpec, o: &Point, w: &Point, t_min: f64, t_max: f64, mut f: impl FnMut([i32; 3], f64, f64)) {
    let Some((t0, t1)) = clip(sp, o, w) else { return };
    let (t0, t1) = (t0.max(t_min), t1.min(t_max));
    if t1 <= t0 {
        return;
    }
    let n = [0, 1, 2].map(|a| ((sp.hi[a] - sp.lo[a]) / sp.voxel).ceil() as i32);
    let p = [0, 1, 2].map(|a| o[a] + w[a] * (t0 + 1e-9));
    let mut k = [0, 1, 2].map(|a| (((p[a] - sp.lo[a]) / sp.voxel).floor() as i32).clamp(0, n[a] - 1));
    let step = [0, 1, 2].map(|a| if w[a] > 0.0 { 1 } else { -1 });
    let mut next = [0, 1, 2].map(|a| {
        if w[a].abs() < 1e-12 {
            f64::INFINITY
        } else {
            let edge = sp.lo[a] + (k[a] + (step[a] > 0) as i32) as f64 * sp.voxel;
            (edge - o[a]) / w[a]
        }
    });
    let delta = [0, 1, 2].map(|a| if w[a].abs() < 1e-12 { f64::INFINITY } else { sp.voxel / w[a].abs() });
    let mut t = t0;
    loop {
        let a = (0..3).min_by(|&i, &j| next[i].total_cmp(&next[j])).unwrap();
        let end = next[a].min(t1);
        if end > t {
            f(k, t, end);
        }
        if next[a] >= t1 {
            break;
        }
        t = next[a];
        k[a] += step[a];
        next[a] += delta[a];
        if k[a] < 0 || k[a] >= n[a] {
            break;
        }
    }
}

/// One cell's ray through the box: `add(voxel, hits, path)` for each echo and each stretch of transmission, weighted
/// `1 / (φ m(ρ))` if `weighted` (see the module docs for which stretches count).
fn ray(v: &RangeImage, sp: &ExtinctionSpec, em: &ErrorModel, cap: usize, weighted: bool, k: usize, mut add: impl FnMut([i32; 3], f64, f64)) {
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    let Some((_, t1)) = clip(sp, &v.origin, &w) else { return };
    let echoes: &[f32] = if v.cell[k] == Cell::Echoes { v.echoes(k) } else { &[] };
    let wt = if weighted { 1.0 / (em.dispersion * em.multiplicity(echoes.first().map_or(t1, |&r| r as f64))) } else { 1.0 };
    let used = if sp.multi_echo { echoes.len() } else { echoes.len().min(1) };
    let mut start = 0.0;
    for &r in &echoes[..used] {
        let r = r as f64;
        traverse(sp, &v.origin, &w, start, r, |key, a, b| add(key, 0.0, wt * (b - a)));
        let x = [0, 1, 2].map(|a| v.origin[a] + w[a] * r);
        if inside(sp, &x) {
            add(key(sp, &x), wt, 0.0);
        }
        start = r + RESOLUTION;
    }
    if echoes.is_empty() || (sp.multi_echo && echoes.len() < cap) {
        traverse(sp, &v.origin, &w, start, f64::INFINITY, |key, a, b| add(key, 0.0, wt * (b - a)));
    }
}

/// The weighted counts of image `v`'s cells in `cells` (every cell when `None`).
pub fn counts(v: &RangeImage, sp: &ExtinctionSpec, em: &ErrorModel, cells: Option<&HashSet<usize>>) -> Counts {
    let cap = v.max_targets();
    let one = |k: usize, acc: &mut Counts| ray(v, sp, em, cap, true, k, |key, h, l| {
        let e = acc.entry(key).or_insert((0.0, 0.0));
        e.0 += h;
        e.1 += l;
    });
    let merge = |mut a: Counts, b: Counts| {
        for (k, (h, l)) in b {
            let e = a.entry(k).or_insert((0.0, 0.0));
            e.0 += h;
            e.1 += l;
        }
        a
    };
    match cells {
        Some(cs) => {
            let mut acc = Counts::new();
            let mut list: Vec<usize> = cs.iter().copied().collect();
            list.sort_unstable();
            list.into_iter().for_each(|k| one(k, &mut acc));
            acc
        }
        None => (0..v.cell.len()).into_par_iter().fold(Counts::new, |mut acc, k| { one(k, &mut acc); acc }).reduce(Counts::new, merge),
    }
}

/// Each voxel's dispersion against the field: image `v`'s rays grouped in tiles of `tile` × `tile` cells, Pearson's
/// `Σ_t (h_t − σ L_t)² / (σ L_t)` over the tiles that cross the voxel, per degree of freedom (1 for a Poisson
/// process with mean σ L; above 1, hits come in clumps), with the tiles counted. Unweighted counts.
pub fn dispersion(v: &RangeImage, ext: &Extinction, em: &ErrorModel, tile: usize) -> HashMap<[i32; 3], (f64, usize)> {
    let Some(sp) = ext.spec else { return HashMap::new() };
    let cap = v.max_targets();
    let nt = v.n_cols.div_ceil(tile);
    let tiles: HashMap<([i32; 3], usize), (f64, f64)> = (0..v.cell.len()).into_par_iter().fold(HashMap::new, |mut acc: HashMap<([i32; 3], usize), (f64, f64)>, k| {
        let t = (k / v.n_cols / tile) * nt + (k % v.n_cols) / tile;
        ray(v, &sp, em, cap, false, k, |key, h, l| {
            let e = acc.entry((key, t)).or_insert((0.0, 0.0));
            e.0 += h;
            e.1 += l;
        });
        acc
    }).reduce(HashMap::new, |mut a, b| {
        for (k, (h, l)) in b {
            let e = a.entry(k).or_insert((0.0, 0.0));
            e.0 += h;
            e.1 += l;
        }
        a
    });
    let mut per: HashMap<[i32; 3], (f64, usize)> = HashMap::new();
    for ((key, _), (h, l)) in tiles {
        let mu = ext.sigma.get(&key).unwrap_or(&ext.kappa) * l;
        if mu > 1e-9 {
            let e = per.entry(key).or_insert((0.0, 0));
            e.0 += (h - mu).powi(2) / mu;
            e.1 += 1;
        }
    }
    per.into_iter().filter(|(_, (_, n))| *n >= 3).map(|(k, (x2, n))| (k, (x2 / (n - 1) as f64, n))).collect()
}

/// `a − b`, voxel by voxel.
pub fn minus(a: &Counts, b: &Counts) -> Counts {
    let mut out = a.clone();
    for (k, (h, l)) in b {
        if let Some(e) = out.get_mut(k) {
            e.0 = (e.0 - h).max(0.0);
            e.1 = (e.1 - l).max(0.0);
        }
    }
    out
}

/// The strand bridge's declared settings (synthesis §17's conservation, as a prior a reconstruction may override):
/// the wood flux an open stick tip carries, its area `π R²`, continues as sub-resolution wood into the diffuse field
/// over `length` (m) ahead of the tip, where it expects mean extinction `gain · π R²`, with log-normal spread `sd`.
#[derive(Debug, Clone, Copy)]
pub struct StrandBridge {
    pub gain: f64,
    pub length: f64,
    pub sd: f64,
}

impl Default for StrandBridge {
    fn default() -> Self {
        StrandBridge { gain: 500.0, length: 1.0, sd: 1.0 }
    }
}

impl StrandBridge {
    /// Its energy for open tips `(tip, unit axis outwards, R)` against the field: a separate factor, never in the
    /// scene's cost or its cached components.
    pub fn energy(&self, ext: &Extinction, tips: &[(Point, Point, f64)]) -> f64 {
        let n = 8;
        tips.iter().map(|(x, u, r)| {
            let mean = (1..=n).map(|j| {
                let s = self.length * j as f64 / n as f64;
                ext.sigma_at(&[0, 1, 2].map(|a| x[a] + u[a] * s))
            }).sum::<f64>() / n as f64;
            let want = self.gain * std::f64::consts::PI * r * r;
            0.5 * (((mean + 1e-3) / (want + 1e-3)).ln() / self.sd).powi(2)
        }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray through a box of 1 m voxels: the segments tile the clipped ray, and a uniform field's optical depth is κ t.
    #[test]
    fn the_traversal_tiles_the_ray() {
        let sp = ExtinctionSpec { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 4.0], voxel: 1.0, prior_path: 1.0, multi_echo: true };
        let (o, w) = ([-1.0, 0.5, 0.5], [0.8, 0.6, 0.0]);
        let mut len = 0.0;
        let mut keys = Vec::new();
        traverse(&sp, &o, &w, 0.0, 100.0, |k, a, b| {
            len += b - a;
            keys.push(k);
        });
        let (t0, t1) = clip(&sp, &o, &w).unwrap();
        assert!((len - (t1 - t0)).abs() < 1e-9, "{len} vs {}", t1 - t0);
        assert_eq!(keys.first(), Some(&[0, 1, 0]));
        let e = Extinction::uniform(0.05);
        assert_eq!(e.tau(&o, &w, 7.0), 0.05 * 7.0);
        let mut f = Extinction::from_counts(0.05, sp, &Counts::from([([0, 0, 0], (0.0, 0.0))]));
        f.sigma.insert([0, 1, 0], 1.05);
        let through = f.tau(&o, &w, 100.0) - f.tau(&o, &w, 0.0);
        assert!(through > 0.05 * 100.0, "{through}");
    }

    /// The strand bridge prefers an open tip whose flux continues into field that holds it over one pointing into
    /// empty space.
    #[test]
    fn an_open_tip_prefers_field_ahead_that_holds_its_flux() {
        let sp = ExtinctionSpec { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 4.0], voxel: 0.5, prior_path: 1.0, multi_echo: true };
        let mut f = Extinction::from_counts(0.0, sp, &Counts::new());
        for i in 0..8 {
            f.sigma.insert([i, 2, 2], 0.15);
        }
        let b = StrandBridge::default();
        let tip = |u: Point| vec![([0.1, 1.25, 1.25], u, 0.01)];
        assert!(b.energy(&f, &tip([1.0, 0.0, 0.0])) < b.energy(&f, &tip([0.0, 1.0, 0.0])) - 1.0);
    }
}
