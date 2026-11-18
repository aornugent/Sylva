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
//! - **Its prior and estimate.** Each voxel's σ has a Gamma prior of mean m₀ worth `b₀` metres of weighted ray
//!   ([`Settings`]'s `field_prior_mean` and `field_prior_path`: shape `a₀ = m₀ b₀`, rate `b₀`), measured per site as the
//!   empty scene's type-II maximum. Each voxel's weighted counts, echoes in it (h) and ray length through it (L), give
//!   σ the Gamma posterior `(a₀ + h, b₀ + L)`.
//! - **The field's term in the cost** (3dtrees research 130 formulation.md: the field collapsed). σ integrated out,
//!   a voxel's counts have the closed-form marginal `ln NB(h, L) = ln Γ(a₀ + h) − ln Γ(a₀) + a₀ ln b₀ − (a₀ + h)
//!   ln(b₀ + L)` (a Gamma-mixed Poisson), and the field's term is its sum over the voxels, every count scored (the
//!   margin's voxels too). It depends on the scene only through the counts, so a move changes it on the voxels its
//!   cuts change, exactly ([`FieldCounts`]); the scene's cost holds its change against the empty scene's counts (every
//!   ray whole). So an element whose echoes a denser medium could explain pays for them: the medium's alternative is
//!   read with its Occam factor, not at a σ̂ the element's own echoes were left out of.
//! - **What a pulse meets behind an element's front** (the medium behind its returns, their merges and τ inside the
//!   ray likelihood) is σ's posterior mean `α / β` per voxel, for the optical depth, the hazard and the merge share
//!   alike (3dtrees research 130 slice 2c: one medium, read the same at every point of a pulse; the Gamma–Poisson
//!   predictive it replaces read each cell as a draw of its own). That part isn't conjugate (the merge share is a
//!   calibrated map, not a likelihood in σ), so it stays this first-order reading, held between refits (below).
//! - **A pulse's echo sequence:** every echo is a hit, and the path up to it is transmission, from the origin for the
//!   first and from the previous echo plus the receiver's resolution after that (a surface closer behind an echo would
//!   have merged with it: [`Settings::resolution`]). The report is the first K returns in time (research 130 slice
//!   12b), so after the last echo the path on is transmission only if the pulse returned fewer echoes than the
//!   instrument records ([`RangeImage::capped`]): a pulse at its cap says nothing about what lies beyond.
//! - **Energy after an echo.** A pulse's echo energies are bounded by its emitted energy (synthesis §4): an echo's
//!   reflectance against a full-footprint hit on the dimmest canopy surface ([`Settings::refl_full_hit`]) is the
//!   share of the beam it took ([`Settings::beam_left`], as the pulse reads it), so a full hit on leaf or wood reads
//!   as the whole beam, and only the rest goes on. So each stretch after an echo counts as transmission in proportion to the beam left: none past an opaque hit, most
//!   past a sliver. Where echoes carry no reflectance, the beam is taken to go on whole. (On held-out transmission
//!   this reading is as good as the first echo's alone or better: research 130 slices 12c, 13b. A clumping factor Ω
//!   on the optical depth wasn't supported: slice 12c.) Each ray counts [`Settings::cell_weight`] at its first echo's
//!   range, or where it leaves the box. φ there is a compound Poisson's clump size: mulga's leaf hits are clumped,
//!   and their dispersion is flat in cell size (synthesis §2).
//! - **A residual: what the fit reads** (3dtrees research 130 slice 2b). The field is fitted from the parts of rays no
//!   element explains. A ray no element fronts counts whole. A ray an element fronts counts up to its front
//!   element's front surface ([`ray`]'s `cut`): its clear path there and its echoes in front of it, so the field
//!   keeps the clear lines of sight to wood; its echoes at and behind the element stay out, as the element's. So each
//!   datum is scored once, by the field's term in front of the cut or by the element's pulse at and behind it. Voxels
//!   within [`FoliageBox::margin`] of an element's surface read the background's κ where a pulse meets them (above),
//!   so an element's radius error doesn't leak into σ beside it; their counts are scored all the same. (Before, a
//!   fronted ray was dropped whole: the field lost the clear path in front of wood and over-blocked there, research
//!   130 slices 12c, 13b.)
//! - **Held during moves** ([`FieldCounts`]), as the scan alignment is: a move updates the counts of the rays whose
//!   cut it changes, the field's term with them, and the voxels its margin claims; what a pulse meets stays the field
//!   of the last refit, so the cost is a function of the scene and that reference, not of the moves that led to it.
//!   `refit_foliage` re-reads it from the counts. The field's own term is never stale; only that reading is held.
//! - **Counts and the term are fixed point** ([`FIXED`]): integer sums are exact in any order, so a move and its undo
//!   restore them exactly and a count over threads is reproducible.
use super::settings::Settings;
use super::range_image::{Cell, RangeImage};
use crate::Point;
use rayon::prelude::*;
use std::collections::HashMap;

/// Where the field is estimated: a box `lo`-`hi`, its voxel (m), and the `margin` (m) about each element's surface
/// where a pulse reads κ: a voxel whose centre is within it (within `r + margin` of a stick's axis or a patch's
/// centre).
#[derive(Debug, Clone, Copy)]
pub struct FoliageBox {
    pub lo: Point,
    pub hi: Point,
    pub voxel: f64,
    pub margin: f64,
}

/// Weighted counts per voxel: (echoes in it, ray length through it).
pub type Counts = HashMap<[i32; 3], (f64, f64)>;

/// The fixed point's unit: 2⁻⁴⁰ of a count or a nat (the module docs).
pub const FIXED: f64 = (1u64 << 40) as f64;

/// Weighted counts per voxel in fixed point ([`FIXED`]): [hits, path].
pub type FixedCounts = HashMap<[i32; 3], [i64; 2], std::hash::BuildHasherDefault<VoxelHasher>>;

/// A fast hash for voxel keys (FxHash's multiply and rotate): a query recounts millions of voxel stretches.
#[derive(Default, Clone, Copy)]
pub struct VoxelHasher(u64);

impl std::hash::Hasher for VoxelHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(4) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            self.write_u32(u32::from_le_bytes(b));
        }
    }

    fn write_u32(&mut self, x: u32) {
        self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, x: usize) {
        self.write_u32(x as u32);
    }
}

/// A voxel's marginal `ln NB(h, L)` under the Gamma prior (shape `a0`, rate `b0`); 0 with no counts (the module docs).
pub fn ln_nb(h: f64, l: f64, a0: f64, b0: f64) -> f64 {
    if h == 0.0 && l == 0.0 {
        return 0.0;
    }
    libm::lgamma(a0 + h) - libm::lgamma(a0) + a0 * b0.ln() - (a0 + h) * (b0 + l).ln()
}

/// The foliage field: per voxel of `spec`'s box with data, σ's Gamma posterior (shape, rate); κ elsewhere.
#[derive(Debug, Clone)]
pub struct Foliage {
    pub kappa: f64,
    pub spec: Option<FoliageBox>,
    pub post: HashMap<[i32; 3], (f64, f64)>,
}

impl Foliage {
    /// The background alone: κ everywhere.
    pub fn uniform(kappa: f64) -> Self {
        Foliage { kappa, spec: None, post: HashMap::new() }
    }

    /// The field's posterior from counts, under the prior (shape `a0`, rate `b0`).
    pub fn from_counts(kappa: f64, spec: FoliageBox, counts: &Counts, (a0, b0): (f64, f64)) -> Self {
        let post = counts.iter().map(|(&k, &(h, l))| (k, (h + a0, l + b0))).collect();
        Foliage { kappa, spec: Some(spec), post }
    }

    /// σ's posterior mean in voxel `k` (κ where it has no data).
    pub fn sigma(&self, k: &[i32; 3]) -> f64 {
        self.post.get(k).map_or(self.kappa, |&(a, b)| a / b)
    }

    /// The optical depth along the ray from `o` in direction `w` to range `t`, and the hazard of an echo at `t`.
    pub fn tau_hazard(&self, o: &Point, w: &Point, t: f64) -> (f64, f64) {
        let (mut tau, mut hazard) = (self.kappa * t, self.kappa);
        if let Some(sp) = self.spec {
            traverse(&sp, o, w, 0.0, t, |k, a, b| {
                let sig = self.sigma(&k);
                tau += (sig - self.kappa) * (b - a);
                hazard = if b >= t { sig } else { self.kappa };
            });
            let x = [0, 1, 2].map(|j| o[j] + w[j] * t);
            if !inside(&sp, &x) {
                hazard = self.kappa;
            }
        }
        (tau, hazard)
    }

    /// The optical depth along the ray from `o` in direction `w` to range `t`.
    pub fn tau(&self, o: &Point, w: &Point, t: f64) -> f64 {
        self.tau_hazard(o, w, t).0
    }

    /// The ray's merge integral behind a return at range `at` whose resolving logit is `z0 + slope (u − at)` at range u
    /// ([`Settings::p_resolve`]): `∫ σ(u) (1 − p(u)) du` from `from` to `to`, exact over σ's voxels (each constant).
    pub fn merge_integral(&self, o: &Point, w: &Point, at: f64, from: f64, to: f64, z0: f64, slope: f64) -> f64 {
        if to <= from {
            return 0.0;
        }
        // ∫ (1 − p) du = u − softplus(z0 + slope (u − at)) / slope
        let softplus = |z: f64| if z > 30.0 { z } else { z.exp().ln_1p() };
        let part = |a: f64, b: f64| (b - a) - (softplus(z0 + slope * (b - at)) - softplus(z0 + slope * (a - at))) / slope;
        let mut total = self.kappa * part(from, to);
        if let Some(sp) = self.spec {
            traverse(&sp, o, w, from, to, |k, a, b| total += (self.sigma(&k) - self.kappa) * part(a, b));
        }
        total
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

/// One cell's ray through the box up to range `cut` (∞ for a ray no element fronts; the module docs say what the fit
/// reads): `add(voxel, hits, path)` for each echo before the cut and each stretch of transmission, weighted by
/// [`Settings::cell_weight`]. The path after the last echo before the cut runs on to the cut, unless the pulse is
/// capped and that echo was its last. Echoes and voxel stretches that end before `after` are left out: they are the
/// same whatever the cut beyond it.
fn ray(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize, cut: f64, after: f64, mut add: impl FnMut([i32; 3], f64, f64)) {
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
        if r >= cut {
            break;
        }
        let b0 = beam;
        traverse(sp, &v.origin, &w, start, r, |key, a, b| if b >= after { add(key, 0.0, wt * b0 * (b - a)) });
        let x = [0, 1, 2].map(|a| v.origin[a] + w[a] * r);
        if r >= after && inside(sp, &x) {
            add(key(sp, &x), wt * b0, 0.0);
        }
        if let Some(&db) = refls.get(i) {
            beam *= em.beam_left(db as f64);
        }
        start = r + em.resolution();
    }
    let known = if v.capped(k) { echoes.last().map_or(f64::INFINITY, |&r| r as f64) } else { f64::INFINITY };
    let b0 = beam;
    traverse(sp, &v.origin, &w, start, cut.min(known), |key, a, b| if b >= after { add(key, 0.0, wt * b0 * (b - a)) });
}

/// Free space in one scan, per voxel of a box (3dtrees research 130 slice 7: the wood map's bounds on hidden wood).
/// Each cell with a pulse traces its ray through the voxels its line crosses, and counts in each: `crossed`, the ray
/// crosses it; `reached`, it enters it before its first echo; `clear`, it leaves it before its clear range ends,
/// [`Settings::free_to`] of the first echo (a miss reaches and clears every voxel), so a surface that could have merged
/// into that echo is never carved. So `crossed − reached` are the voxel's rays in shadow, and `reached − clear` the
/// ones that end in it or within the receiver's resolution behind it. Counts in voxel order, x slowest:
/// `(i · ny + j) · nz + l`.
pub fn free_space(v: &RangeImage, lo: Point, hi: Point, voxel: f64, em: &Settings) -> [Vec<u32>; 3] {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    let sp = FoliageBox { lo, hi, voxel, margin: 0.0 };
    let n = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / voxel).ceil() as usize);
    let counts: [Vec<AtomicU32>; 3] = std::array::from_fn(|_| (0..n[0] * n[1] * n[2]).map(|_| AtomicU32::new(0)).collect());
    (0..v.cell.len()).into_par_iter().filter(|&k| v.cell[k] != Cell::NoPulse).for_each(|k| {
        let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
        let first = if v.cell[k] == Cell::Echoes { v.range[k] as f64 } else { f64::INFINITY };
        let free = em.free_to(first);
        traverse(&sp, &v.origin, &w, 0.0, f64::INFINITY, |key, a, b| {
            let i = (key[0] as usize * n[1] + key[1] as usize) * n[2] + key[2] as usize;
            counts[0][i].fetch_add(1, Relaxed);
            counts[1][i].fetch_add((a < first) as u32, Relaxed);
            counts[2][i].fetch_add((b <= free) as u32, Relaxed);
        });
    });
    counts.map(|c| c.into_iter().map(AtomicU32::into_inner).collect())
}

/// Add (h, l), signed by `sign`, to a voxel's counts in fixed point; a voxel whose counts fall to nothing is dropped.
fn accumulate(acc: &mut FixedCounts, key: [i32; 3], h: f64, l: f64, sign: i64) {
    let e = acc.entry(key).or_insert([0, 0]);
    e[0] += sign * (h * FIXED).round() as i64;
    e[1] += sign * (l * FIXED).round() as i64;
    if *e == [0, 0] {
        acc.remove(&key);
    }
}

/// Image `v`'s cell `k` counted up to `to` in place of `from`: its change to the counts, added to `acc`.
pub fn recount(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize, from: f64, to: f64, acc: &mut FixedCounts) {
    if from != to {
        let after = from.min(to);
        ray(v, sp, em, k, from, after, |key, h, l| accumulate(acc, key, h, l, -1));
        ray(v, sp, em, k, to, after, |key, h, l| accumulate(acc, key, h, l, 1));
    }
}

fn merged(mut a: FixedCounts, b: FixedCounts) -> FixedCounts {
    for (k, [h, l]) in b {
        let e = a.entry(k).or_insert([0, 0]);
        e[0] += h;
        e[1] += l;
        if *e == [0, 0] {
            a.remove(&k);
        }
    }
    a
}

/// The field's counts, its part of the cost and the voxels elements' margins claim, kept as the scene changes (the
/// module docs).
///
/// Its part of the cost is `ln NB(∅) − ln NB(E) − plug-in`, each summed over the voxels: the field's term of the
/// empty scene's counts (every ray whole) less the scene's, less the plug-in `Σ h ln σ̂ − σ̂ L` of the counts the
/// elements' cuts take out (`behind`) at the field a pulse meets. The plug-in is how the empty scene's pulses already
/// read those segments inside each fronted cell's ratio ([`super::scene`]'s ray likelihood), so the field's part
/// swaps that reading for the predictive: the segments are scored once (3dtrees research 130 formulation.md).
#[derive(Debug, Clone)]
pub struct FieldCounts {
    pub spec: FoliageBox,
    /// The prior's shape and rate.
    pub prior: (f64, f64),
    /// Per voxel, the weighted counts of the parts of rays no element explains, and of the parts they do: the empty
    /// scene's counts less the scene's.
    counts: FixedCounts,
    behind: FixedCounts,
    /// Per voxel, how many elements' margins claim it.
    pub claimed: HashMap<[i32; 3], u32>,
    /// In fixed point: the field's term `Σ ln NB`, the empty scene's, and the plug-in of `behind`.
    term: i128,
    empty: i128,
    plug: i128,
}

fn fixed(x: f64) -> i128 {
    (x * FIXED).round() as i128
}

impl FieldCounts {
    /// The counts of images `views`, each cell's ray up to its cut in `cuts` (per image, by cell; ∞ where absent),
    /// under the prior in `em`, nothing claimed yet: the empty scene's counts, then each fronted cell recounted. Its
    /// plug-in is read at κ `kappa` until [`FieldCounts::reread`].
    pub fn new(spec: FoliageBox, views: &[&RangeImage], em: &Settings, cuts: &[HashMap<usize, f64>], kappa: f64) -> Self {
        let prior = (em.field_prior_mean * em.field_prior_path, em.field_prior_path);
        let counts = views.iter().map(|v| {
            (0..v.cell.len()).into_par_iter().fold(FixedCounts::default, |mut acc, k| {
                ray(v, &spec, em, k, f64::INFINITY, 0.0, |key, h, l| accumulate(&mut acc, key, h, l, 1));
                acc
            }).reduce(FixedCounts::default, merged)
        }).fold(FixedCounts::default(), merged);
        let mut f = FieldCounts { spec, prior, counts, behind: FixedCounts::default(), claimed: HashMap::new(), term: 0, empty: 0, plug: 0 };
        f.term = f.counts.values().map(|&c| f.fixed_term(c)).sum();
        f.empty = f.term;
        let fronted = views.iter().zip(cuts).map(|(v, cut)| {
            cut.par_iter().fold(FixedCounts::default, |mut acc, (&k, &c)| {
                recount(v, &spec, em, k, f64::INFINITY, c, &mut acc);
                acc
            }).reduce(FixedCounts::default, merged)
        }).fold(FixedCounts::default(), merged);
        f.apply(fronted, &Foliage::uniform(kappa));
        f
    }

    /// A voxel's `ln NB` in fixed point.
    fn fixed_term(&self, [h, l]: [i64; 2]) -> i128 {
        fixed(ln_nb(h as f64 / FIXED, l as f64 / FIXED, self.prior.0, self.prior.1))
    }

    /// The plug-in of counts `behind` in voxel `k` at the field `ext`, in fixed point.
    fn fixed_plug(k: &[i32; 3], [h, l]: [i64; 2], ext: &Foliage) -> i128 {
        if h == 0 && l == 0 {
            return 0;
        }
        let s = ext.sigma(k);
        fixed((h as f64 * s.ln() - s * l as f64) / FIXED)
    }

    /// The change in the term and in the plug-in were the counts changed by `delta`.
    fn fixed_change(&self, delta: &FixedCounts, ext: &Foliage) -> (i128, i128) {
        delta.iter().fold((0, 0), |(t, p), (k, d)| {
            let c = self.counts.get(k).copied().unwrap_or([0, 0]);
            let b = self.behind.get(k).copied().unwrap_or([0, 0]);
            (t + self.fixed_term([c[0] + d[0], c[1] + d[1]]) - self.fixed_term(c),
             p + Self::fixed_plug(k, [b[0] - d[0], b[1] - d[1]], ext) - Self::fixed_plug(k, b, ext))
        })
    }

    /// The change in the field's part of the cost were the counts changed by `delta` ([`recount`]), its plug-in read
    /// at `ext`; the counts stay as they are.
    pub fn change(&self, delta: &FixedCounts, ext: &Foliage) -> f64 {
        let (t, p) = self.fixed_change(delta, ext);
        -(t + p) as f64 / FIXED
    }

    /// Change the counts by `delta`, the plug-in read at `ext`: the change in the field's part of the cost.
    pub fn apply(&mut self, delta: FixedCounts, ext: &Foliage) -> f64 {
        let (t, p) = self.fixed_change(&delta, ext);
        (self.term, self.plug) = (self.term + t, self.plug + p);
        let neg: FixedCounts = delta.iter().map(|(&k, &[h, l])| (k, [-h, -l])).collect();
        self.behind = merged(std::mem::take(&mut self.behind), neg);
        self.counts = merged(std::mem::take(&mut self.counts), delta);
        -(t + p) as f64 / FIXED
    }

    /// Read the plug-in at the field `ext` (the field a pulse now meets, after a refit).
    pub fn reread(&mut self, ext: &Foliage) {
        self.plug = self.behind.iter().map(|(k, &b)| Self::fixed_plug(k, b, ext)).sum();
    }

    /// The field's part of the scene's cost (above).
    pub fn cost(&self) -> f64 {
        (self.empty - self.term - self.plug) as f64 / FIXED
    }

    /// The field's term `Σ ln NB` over the voxels: the log marginal likelihood of the counts.
    pub fn marginal(&self) -> f64 {
        self.term as f64 / FIXED
    }

    /// The weighted counts per voxel, (hits, path).
    pub fn counts(&self) -> Counts {
        self.counts.iter().map(|(&k, &[h, l])| (k, (h as f64 / FIXED, l as f64 / FIXED))).collect()
    }

    /// Claim (or, `enter` false, release) the voxels within the margin of an element's surface: a stick from `a` to
    /// `b` of radius `r`, or a patch about `a` (`b` = `a`) of extent `r`.
    pub fn claim(&mut self, a: &Point, b: &Point, r: f64, enter: bool) {
        let sp = self.spec;
        let reach = r + sp.margin;
        let ab = [0, 1, 2].map(|j| b[j] - a[j]);
        let len2: f64 = ab.iter().map(|x| x * x).sum();
        let n = [0, 1, 2].map(|j| ((sp.hi[j] - sp.lo[j]) / sp.voxel).ceil() as i32);
        let lo = [0, 1, 2].map(|j| (((a[j].min(b[j]) - reach - sp.lo[j]) / sp.voxel).floor() as i32).max(0));
        let hi = [0, 1, 2].map(|j| (((a[j].max(b[j]) + reach - sp.lo[j]) / sp.voxel).floor() as i32).min(n[j] - 1));
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for l in lo[2]..=hi[2] {
                    let c = [i, j, l];
                    let x = [0, 1, 2].map(|q| sp.lo[q] + (c[q] as f64 + 0.5) * sp.voxel - a[q]);
                    let s = if len2 > 0.0 { ((0..3).map(|q| x[q] * ab[q]).sum::<f64>() / len2).clamp(0.0, 1.0) } else { 0.0 };
                    if (0..3).map(|q| (x[q] - s * ab[q]).powi(2)).sum::<f64>() >= reach * reach {
                        continue;
                    }
                    let e = self.claimed.entry(c).or_insert(0);
                    if enter {
                        *e += 1;
                    } else {
                        *e -= 1;
                        if *e == 0 {
                            self.claimed.remove(&c);
                        }
                    }
                }
            }
        }
    }

    /// The field a pulse meets: the posterior from the counts, the claimed voxels left out (κ there).
    pub fn field(&self, kappa: f64) -> Foliage {
        let free: Counts = self.counts().into_iter().filter(|(k, _)| !self.claimed.contains_key(k)).collect();
        Foliage::from_counts(kappa, self.spec, &free, self.prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray through a box of 1 m voxels: the segments tile the clipped ray, and a uniform field's optical depth is κ t.
    #[test]
    fn the_traversal_tiles_the_ray() {
        let sp = FoliageBox { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 4.0], voxel: 1.0, margin: 0.0 };
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
        let mut f = Foliage::from_counts(0.05, sp, &Counts::from([([0, 0, 0], (0.0, 0.0))]), (0.05, 1.0));
        f.post.insert([0, 1, 0], (1.05 * 1000.0, 1000.0));
        let through = f.tau(&o, &w, 100.0) - f.tau(&o, &w, 0.0);
        assert!(through > 0.05 * 100.0, "{through}");
        // σ's posterior mean, however much data it rests on (slice 2c)
        let seg = |f: &Foliage| f.tau(&o, &w, 100.0) - 0.05 * 100.0;
        let plug = seg(&f);
        f.post.insert([0, 1, 0], (1.05, 1.0));
        assert!((seg(&f) - plug).abs() < 1e-9, "{} vs {plug}", seg(&f));
        // the merge integral: κ's exact over a ray that misses the box, and nothing past a resolved return
        let (z0, slope) = (-2.563, 2.693);
        let flat = Foliage::uniform(0.05);
        let m = flat.merge_integral(&o, &w, 10.0, 10.0, 100.0, z0, slope);
        let numeric: f64 = (0..20000).map(|i| 0.05 * 0.005 / (1.0 + (z0 + slope * (i as f64 + 0.5) * 0.005).exp())).sum();
        assert!((m - numeric).abs() < 1e-6, "{m} vs {numeric}");
        assert!(flat.merge_integral(&o, &w, 10.0, 20.0, 30.0, z0, slope) < 1e-10);
    }
}
