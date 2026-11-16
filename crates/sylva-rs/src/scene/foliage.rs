// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The foliage field (3dtrees research 130 slice 12; synthesis §2, §15): what the explicit elements don't explain,
//! as a mean extinction σ (1/m) per voxel in a box. Leaf area and wood under a footprint both fold into it (σ alone
//! can't separate them); "foliage" names its main part.
//!
//! - **Its form.** Inside the box, σ is constant per voxel and the same in every direction: one coefficient a voxel,
//!   since a voxel is seen from a handful of near-horizontal directions (synthesis §15). Outside the box, and where
//!   the box has no data, σ is the background's κ. A ray's survival to range t is `exp(−∫₀ᵗ σ)` and its first echo's
//!   density at t is `σ(t) exp(−∫₀ᵗ σ)`, so a miss is evidence of transmission.
//! - **Its estimate.** Each voxel's weighted counts, echoes in it (h) and ray length through it (L), give σ a Gamma
//!   posterior, shape `α = h + κ b` and rate `β = L + b`, under a Gamma prior worth `b` metres of ray at mean κ; its
//!   mean is `α / β`.
//! - **What a ray meets of it** is the Gamma–Poisson predictive, σ integrated out per voxel: a stretch ℓ through a
//!   voxel survives with chance `(β / (β + ℓ))^α` (optical depth `α ln(1 + ℓ / β)`), and an echo after ℓ in it has
//!   hazard `α / (β + ℓ)`. Only the scene's elements' cells are scored against it, from the rays they don't front.
//! - **A pulse's echo sequence:** every echo is a hit, and the path up to it is transmission, from the origin for the
//!   first and from the previous echo plus the receiver's resolution after that (a surface closer behind an echo would
//!   have merged with it: [`Settings::resolution`]). The report is the first K returns in time (research 130 slice
//!   12b), so after the last echo the path on is transmission only if the pulse returned fewer echoes than the
//!   instrument records ([`RangeImage::capped`]): a pulse at its cap says nothing about what lies beyond.
//! - **Energy after an echo.** A pulse's echo energies are bounded by its emitted energy (synthesis §4): an echo's
//!   reflectance against a full-footprint hit on the dimmest canopy surface ([`Settings::refl_full_hit`]) is the
//!   share of the beam it took, so a full hit on leaf or wood reads as the whole beam, and only the rest goes on. So
//!   each stretch after an echo counts as transmission in proportion to the beam left: none past an opaque hit, most
//!   past a sliver. Where echoes carry no reflectance, the beam is taken to go on whole. (On held-out transmission
//!   this reading is as good as the first echo's alone or better: research 130 slices 12c, 13b. A clumping factor Ω
//!   on the optical depth wasn't supported: slice 12c.) Each ray counts [`Settings::cell_weight`] at its first echo's
//!   range, or where it leaves the box. φ there is a compound Poisson's clump size: mulga's leaf hits are clumped,
//!   and their dispersion is flat in cell size (synthesis §2).
//! - **A residual.** The field is estimated from the rays no element fronts, so an element and the field never
//!   explain the same echoes. It enters the scene's cost only as the background along the cells the elements front,
//!   for the scene and for the empty scene alike: the rays it is fitted from are the same in both and cancel, so
//!   it never scores the counts it was fitted from, and each ray is scored by one model (3dtrees research 130 step
//!   2; before, every ray's counts were also scored by a plug-in Poisson term, the fronted ones twice).

use super::settings::Settings;
use super::range_image::{Cell, RangeImage};
use crate::Point;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Where the field is estimated: a box `lo`-`hi`, its voxel (m), and the prior's worth `prior_path` (m of weighted
/// ray).
#[derive(Debug, Clone, Copy)]
pub struct FoliageBox {
    pub lo: Point,
    pub hi: Point,
    pub voxel: f64,
    pub prior_path: f64,
}

/// Weighted counts per voxel: (first echoes in it, ray length through it).
pub type Counts = HashMap<[i32; 3], (f64, f64)>;

/// The foliage field: per voxel of `spec`'s box with data, σ's Gamma posterior (shape, rate); κ elsewhere.
#[derive(Debug, Clone)]
pub struct Foliage {
    pub kappa: f64,
    pub spec: Option<FoliageBox>,
    pub post: HashMap<[i32; 3], (f64, f64)>,
}

/// What a cell's ray meets of the foliage field: the optical depth to each of its layers and then the ground (`taus`), to the
/// beam's end, and to its first echo, with σ at that echo.
#[derive(Debug, Clone)]
pub struct RayFoliage {
    pub taus: Vec<f64>,
    pub tau_end: f64,
    pub tau_obs: f64,
    pub sig_obs: f64,
}

impl RayFoliage {
    /// The same ray with its layers left out (the ground's optical depth kept).
    pub fn base(&self) -> RayFoliage {
        RayFoliage { taus: vec![*self.taus.last().unwrap_or(&0.0)], ..*self }
    }
}

impl Foliage {
    /// The background alone: κ everywhere.
    pub fn uniform(kappa: f64) -> Self {
        Foliage { kappa, spec: None, post: HashMap::new() }
    }

    /// The field's posterior from counts.
    pub fn from_counts(kappa: f64, spec: FoliageBox, counts: &Counts) -> Self {
        let b = spec.prior_path;
        let post = counts.iter().map(|(&k, &(h, l))| (k, (h + kappa * b, l + b))).collect();
        Foliage { kappa, spec: Some(spec), post }
    }

    /// σ's posterior mean in voxel `k` (κ where it has no data).
    pub fn sigma(&self, k: &[i32; 3]) -> f64 {
        self.post.get(k).map_or(self.kappa, |&(a, b)| a / b)
    }

    /// The predictive optical depth along the ray from `o` in direction `w` to range `t`, and the hazard of an echo
    /// at `t`.
    pub fn tau_hazard(&self, o: &Point, w: &Point, t: f64) -> (f64, f64) {
        let (mut tau, mut hazard) = (self.kappa * t, self.kappa);
        if let Some(sp) = self.spec {
            traverse(&sp, o, w, 0.0, t, |k, a, b| {
                if let Some(&(al, be)) = self.post.get(&k) {
                    tau += al * (1.0 + (b - a) / be).ln() - self.kappa * (b - a);
                    hazard = if b >= t { al / (be + b - a) } else { self.kappa };
                } else {
                    hazard = self.kappa;
                }
            });
            let x = [0, 1, 2].map(|j| o[j] + w[j] * t);
            if !inside(&sp, &x) {
                hazard = self.kappa;
            }
        }
        (tau, hazard)
    }

    /// The predictive optical depth along the ray from `o` in direction `w` to range `t`.
    pub fn tau(&self, o: &Point, w: &Point, t: f64) -> f64 {
        self.tau_hazard(o, w, t).0
    }

    /// What a cell's ray from `o` along `w` meets: optical depths to its layers' ranges, its ground's, the beam's end
    /// and its first echo `obs` (NaN for none).
    pub fn ray_bg(&self, o: &Point, w: &Point, layer_ts: &[f64], ground_t: Option<f64>, t_end: f64, obs: f64) -> RayFoliage {
        let mut taus: Vec<f64> = layer_ts.iter().map(|&t| self.tau(o, w, t)).collect();
        taus.push(ground_t.map_or(0.0, |t| self.tau(o, w, t)));
        let (tau_obs, sig_obs) = if obs.is_finite() {
            self.tau_hazard(o, w, obs)
        } else {
            (f64::NAN, f64::NAN)
        };
        RayFoliage { taus, tau_end: self.tau(o, w, t_end), tau_obs, sig_obs }
    }
}

fn inside(sp: &FoliageBox, x: &Point) -> bool {
    (0..3).all(|a| x[a] >= sp.lo[a] && x[a] < sp.hi[a])
}

fn key(sp: &FoliageBox, x: &Point) -> [i32; 3] {
    [0, 1, 2].map(|a| ((x[a] - sp.lo[a]) / sp.voxel).floor() as i32)
}

/// The ray's entry and exit ranges through the box (entry ≥ 0), if it crosses it.
pub fn clip(sp: &FoliageBox, o: &Point, w: &Point) -> Option<(f64, f64)> {
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
fn traverse(sp: &FoliageBox, o: &Point, w: &Point, t_min: f64, t_max: f64, mut f: impl FnMut([i32; 3], f64, f64)) {
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
/// by [`Settings::cell_weight`] (see the module docs for which stretches count).
fn ray(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize, mut add: impl FnMut([i32; 3], f64, f64)) {
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    let Some((_, t1)) = clip(sp, &v.origin, &w) else { return };
    if v.cell[k] == Cell::NoPulse {
        return; // nothing observed
    }
    let echoes: &[f32] = if v.cell[k] == Cell::Echoes { v.echoes(k) } else { &[] };
    let refls = if v.cell[k] == Cell::Echoes { v.echo_refls(k) } else { &[] };
    let wt = em.cell_weight(echoes.first().map_or(t1, |&r| r as f64), v.step());
    let (mut start, mut beam) = (0.0, 1.0);
    for (i, &r) in echoes.iter().enumerate() {
        let r = r as f64;
        let b0 = beam;
        traverse(sp, &v.origin, &w, start, r, |key, a, b| add(key, 0.0, wt * b0 * (b - a)));
        let x = [0, 1, 2].map(|a| v.origin[a] + w[a] * r);
        if inside(sp, &x) {
            add(key(sp, &x), wt * b0, 0.0);
        }
        if let Some(&db) = refls.get(i) {
            beam *= (1.0 - 10f64.powf((db as f64 - em.refl_full_hit()) / 10.0)).clamp(0.0, 1.0);
        }
        start = r + em.resolution();
    }
    if !v.capped(k) {
        let b0 = beam;
        traverse(sp, &v.origin, &w, start, f64::INFINITY, |key, a, b| add(key, 0.0, wt * b0 * (b - a)));
    }
}

/// The weighted counts of image `v`'s cells in `cells` (every cell when `None`).
pub fn counts(v: &RangeImage, sp: &FoliageBox, em: &Settings, cells: Option<&HashSet<usize>>) -> Counts {
    let one = |k: usize, acc: &mut Counts| ray(v, sp, em, k, |key, h, l| {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray through a box of 1 m voxels: the segments tile the clipped ray, and a uniform field's optical depth is κ t.
    #[test]
    fn the_traversal_tiles_the_ray() {
        let sp = FoliageBox { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 4.0], voxel: 1.0, prior_path: 1.0 };
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
        let e = Foliage::uniform(0.05);
        assert_eq!(e.tau(&o, &w, 7.0), 0.05 * 7.0);
        let mut f = Foliage::from_counts(0.05, sp, &Counts::from([([0, 0, 0], (0.0, 0.0))]));
        f.post.insert([0, 1, 0], (1.05 * 1000.0, 1000.0));
        let through = f.tau(&o, &w, 100.0) - f.tau(&o, &w, 0.0);
        assert!(through > 0.05 * 100.0, "{through}");
        // with much data the predictive is the plug-in; with little, a stretch survives more than σ's mean says
        let seg = |f: &Foliage| f.tau(&o, &w, 100.0) - 0.05 * 100.0;
        let plug = seg(&f);
        f.post.insert([0, 1, 0], (1.05, 1.0));
        assert!(seg(&f) < plug, "{} vs {plug}", seg(&f));
    }
}
