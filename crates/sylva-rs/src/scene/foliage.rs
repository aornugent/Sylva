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
//!   ln(b₀ + L)` (a Gamma-mixed Poisson), and the field's term is its sum over the voxels, every count scored. It
//!   depends on the scene only through the counts, so a move changes it on the voxels whose hits its terms' deposits
//!   change, exactly ([`FieldCounts`]); the scene's cost holds its change against the empty scene's counts. So an
//!   element whose echoes a denser medium could explain pays for them: the medium's alternative is read with its Occam
//!   factor, not at a σ̂ the element's own echoes were left out of.
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
//! - **One walk per ray** (3dtrees research 130 slice-structure): a cell's ray is traversed through the box once
//!   ([`Walk`]). The pulse reads its optical depths and hazards as prefix sums over the walk's segments, and its merge
//!   integrals as sums over them ([`Medium`]); the field's counts lay its path on the same segments.
//! - **Energy after an echo.** A pulse's echo energies are bounded by its emitted energy (synthesis §4): an echo's
//!   reflectance against a full-footprint hit on the dimmest canopy surface ([`Settings::refl_full_hit`]) is the
//!   share of the beam it took ([`Settings::beam_left`], as the pulse reads it), so a full hit on leaf or wood reads
//!   as the whole beam, and only the rest goes on. So each stretch after an echo counts as transmission in proportion to the beam left: none past an opaque hit, most
//!   past a sliver. Where echoes carry no reflectance, the beam is taken to go on whole. (On held-out transmission
//!   this reading is as good as the first echo's alone or better: research 130 slices 12c, 13b. A clumping factor Ω
//!   on the optical depth wasn't supported: slice 12c.) Each ray counts [`Settings::cell_weight`] at its first echo's
//!   range, or where it leaves the box. φ there is a compound Poisson's clump size: mulga's leaf hits are clumped,
//!   and their dispersion is flat in cell size (synthesis §2).
//! - **Soft attribution: what the field counts** (3dtrees research 130 slice-soft; formulation.md "Soft attribution
//!   and the representation under it"). Every ray counts its whole path, in the beam its echoes left: the ray
//!   likelihood reads the medium's path in every way it explains a pulse (an echo is the medium's, an element's or an
//!   outlier's, and the path to it and on from it is the medium's all the same), so the field does too. Each echo is a
//!   hit, less the share the scene's elements take of it: their responsibilities for it, the posterior over who made
//!   it from the pass over its pulse at its term's nodes, shared among a cell's terms by their front shares (a
//!   term's [`Deposit`]). So each echo's mass sums to one between the field and the elements, and nothing about an
//!   element is set aside. The empty scene's counts read every echo as the medium's.
//! - **Held during moves** ([`FieldCounts`]), as the scan alignment is: a move changes the deposits of the terms it
//!   recomputes (each term's old deposit back, its new one out), and the field's term with them, exactly; what a pulse
//!   meets stays the field of the last refit (the responsibilities read it), so the cost is a function of the scene
//!   and that reference, not of the moves that led to it. `refit_foliage` re-reads it from the counts: one step of EM
//!   (the responsibilities given the field, then the field given them). The field's own term is never stale; only
//!   that reading is held.
//! - **Counts and the term are fixed point** ([`FIXED`]): integer sums are exact in any order, so a move and its undo
//!   restore them exactly and a count over threads is reproducible.
use super::dual::Real;
use super::settings::Settings;
use super::range_image::{Cell, RangeImage};
use crate::Point;
use rayon::prelude::*;
use std::collections::HashMap;

/// Where the field is estimated: a box `lo`-`hi` and its voxel (m).
#[derive(Debug, Clone, Copy)]
pub struct FoliageBox {
    pub lo: Point,
    pub hi: Point,
    pub voxel: f64,
}

/// Weighted counts per voxel: (echoes in it, ray length through it).
pub type Counts = HashMap<[i32; 3], (f64, f64)>;

/// The fixed point's unit: 2⁻⁴⁰ of a count or a nat (the module docs).
pub const FIXED: f64 = (1u64 << 40) as f64;

/// Weighted counts per voxel in fixed point ([`FIXED`]): [hits, path].
pub type FixedCounts = HashMap<[i32; 3], [i64; 2], std::hash::BuildHasherDefault<VoxelHasher>>;

/// Weighted hits per voxel in fixed point: what the elements take, and a move's change to it (the path is never
/// taken).
pub type FixedHits = HashMap<[i32; 3], i64, std::hash::BuildHasherDefault<VoxelHasher>>;

/// The hits a term's elements take from the field (the module docs: soft attribution), per voxel in fixed point, in
/// voxel order: a function of the term alone, so a move's change to the counts is its terms' old deposits less
/// their new ones, exactly.
pub type Deposit = Vec<([i32; 3], i64)>;

/// A deposit from (voxel, hits) pairs: each voxel's sum, in the pairs' order, in fixed point.
pub fn deposit(hits: impl Iterator<Item = ([i32; 3], f64)>) -> Deposit {
    let mut by: std::collections::BTreeMap<[i32; 3], f64> = Default::default();
    for (k, h) in hits {
        *by.entry(k).or_default() += h;
    }
    by.into_iter().map(|(k, h)| (k, (h * FIXED).round() as i64)).filter(|x| x.1 != 0).collect()
}

/// Add a deposit's hits to `acc` with `sign` (+1 back to the field, −1 out of it).
pub fn give(acc: &mut FixedHits, d: &Deposit, sign: i64) {
    for &(k, h) in d {
        add_hits(acc, k, sign * h);
    }
}

/// Add `h` to voxel `k`'s hits in `acc`; a voxel that comes to nothing is dropped.
fn add_hits(acc: &mut FixedHits, k: [i32; 3], h: i64) {
    let e = acc.entry(k).or_insert(0);
    *e += h;
    if *e == 0 {
        acc.remove(&k);
    }
}

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

    /// What the ray from `o` in direction `w` meets of the field ([`Medium`]), from one walk through the box.
    pub fn medium(&self, o: &Point, w: &Point) -> Medium {
        let mut before = 0.0;
        let segs = self.spec.map_or(Vec::new(), |sp| Walk::new(&sp, o, w).0.into_iter().map(|(k, a, b)| {
            let sig = self.sigma(&k);
            let s = [a, b, sig, before];
            before += (sig - self.kappa) * (b - a);
            s
        }).collect());
        Medium { kappa: self.kappa, segs }
    }
}

/// A ray's voxel segments through a box, walked once (one traversal; 3dtrees research 130 slice-structure): (voxel,
/// from, to), in order, tiling the ray where it is inside the box.
pub struct Walk(Vec<([i32; 3], f64, f64)>);

impl Walk {
    pub fn new(sp: &FoliageBox, o: &Point, w: &Point) -> Walk {
        let mut segs = Vec::new();
        traverse(sp, o, w, 0.0, f64::INFINITY, |k, a, b| segs.push((k, a, b)));
        Walk(segs)
    }

    /// Its segments, in order.
    pub fn segments(&self) -> &[([i32; 3], f64, f64)] {
        &self.0
    }

    /// Its segments clipped to the ranges `from` to `to`, in order.
    fn within(&self, from: f64, to: f64) -> impl Iterator<Item = ([i32; 3], f64, f64)> + '_ {
        let i = self.0.partition_point(|s| s.2 <= from);
        self.0[i..].iter().take_while(move |s| s.1 < to).map(move |&(k, a, b)| (k, a.max(from), b.min(to))).filter(|s| s.2 > s.1)
    }
}

/// What one ray meets of the foliage field ([`Foliage::medium`]): κ, and per segment of its walk inside the box (from,
/// to, σ there, and the optical depth's departure from κ's before it), so its optical depths and hazards are prefix
/// sums and its merge integrals sums over the segments, never another traversal.
pub struct Medium {
    kappa: f64,
    segs: Vec<[f64; 4]>,
}

impl Medium {
    /// The optical depth from the ray's origin to range `t`.
    pub fn tau<P: Real>(&self, t: P) -> P {
        let i = self.segs.partition_point(|s| s[0] < t.v());
        let past = if i == 0 { P::c(0.0) } else { let s = self.segs[i - 1]; (t.min(s[1]) - s[0]) * (s[2] - self.kappa) + s[3] };
        t * self.kappa + past
    }

    /// The hazard of an echo at range `t`: σ in the voxel the ray is in there (κ outside the box).
    pub fn hazard(&self, t: f64) -> f64 {
        let i = self.segs.partition_point(|s| s[1] < t);
        self.segs.get(i).filter(|s| s[0] < t).map_or(self.kappa, |s| s[2])
    }

    /// The merge integral behind a return at range `at` whose resolving logit is `z0 + slope (u − at)` at range u
    /// ([`Settings::p_resolve`]): `∫ σ(u) (1 − p(u)) du` from `from` to `to`, exact over σ's voxels (each constant).
    pub fn merge_integral<P: Real>(&self, at: P, from: P, to: P, z0: P, slope: f64) -> P {
        if to.v() <= from.v() {
            return P::c(0.0);
        }
        // ∫ (1 − p) du = u − softplus(z0 + slope (u − at)) / slope
        let softplus = |z: P| if z.v() > 30.0 { z } else { z.exp().ln_1p() };
        let part = |a: P, b: P| (b - a) - (softplus(z0 + (b - at) * slope) - softplus(z0 + (a - at) * slope)) / slope;
        let i = self.segs.partition_point(|s| s[1] <= from.v());
        self.segs[i..].iter().take_while(|s| s[0] < to.v())
            .fold(part(from, to) * self.kappa, |total, s| total + part(from.larger(P::c(s[0])), to.smaller(P::c(s[1]))) * (s[2] - self.kappa))
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

/// One pulse as everything that reads its path reads it (3dtrees research 130 slice-pyramid: the restart, beam-left
/// and miss-weight conventions in one place): its weight ([`Settings::cell_weight`] at its first echo's range, or for
/// a miss at `exit`, where its ray leaves the box the reader counts in); the stretches of its ray it shows clear, each
/// with the beam the echoes before it left ([`Settings::beam_left`]): from the origin to its first echo, then from
/// each echo plus the receiver's resolution to the next (a surface closer behind an echo would have merged with it),
/// and after the last on, unless the pulse is capped ([`RangeImage::capped`]: it says nothing past its last echo);
/// each echo's range and the beam the echoes before it left; its clear range ([`Settings::free_to`] of its first echo,
/// infinite for a miss); and the beam it leaves. None for a cell without a pulse.
pub struct Walked {
    pub weight: f64,
    pub path: Vec<(f64, f64, f64)>,
    pub echoes: Vec<(f64, f64)>,
    pub clear: f64,
    pub beam: f64,
}

/// Cell `k`'s pulse in image `v` as [`Walked`] reads it.
pub fn walked(v: &RangeImage, em: &Settings, k: usize, exit: f64) -> Option<Walked> {
    if v.cell[k] == Cell::NoPulse {
        return None;
    }
    let (echoes, refls) = (v.echoes(k), v.echo_refls(k));
    let weight = em.cell_weight(echoes.first().map_or(exit, |&r| r as f64), v.step());
    let known = if v.capped(k) { echoes.last().map_or(f64::INFINITY, |&r| r as f64) } else { f64::INFINITY };
    let (mut start, mut beam) = (0.0, 1.0);
    let (mut path, mut out) = (Vec::with_capacity(echoes.len() + 1), Vec::with_capacity(echoes.len()));
    for (i, &r) in echoes.iter().enumerate() {
        path.push((start, r as f64, beam));
        out.push((r as f64, beam));
        if let Some(&db) = refls.get(i) {
            beam *= em.beam_left(db as f64);
        }
        start = r as f64 + em.resolution();
    }
    path.push((start, known, beam));
    let clear = echoes.first().map_or(f64::INFINITY, |&r| em.free_to(r as f64));
    Some(Walked { weight, path, echoes: out, clear, beam })
}

/// The beam cell `k`'s echoes leave ([`Walked`]'s `beam`, without its stretches).
pub fn beam_left(v: &RangeImage, em: &Settings, k: usize) -> f64 {
    v.echo_refls(k).iter().fold(1.0, |b, &db| b * em.beam_left(db as f64))
}

/// The echoes of image `v`'s cell `k` the field counts as hits, those inside the box: (the echo's index in its
/// pulse, its voxel, its weight), its pulse's weight times the beam the echoes before it left ([`walked`]). The empty
/// scene counts each whole; a scene's elements take their responsibilities' share (the module docs).
pub fn hits(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize) -> Vec<(usize, [i32; 3], f64)> {
    if v.cell[k] != Cell::Echoes {
        return Vec::new();
    }
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    let p = walked(v, em, k, f64::INFINITY).expect("a pulse");
    p.echoes.iter().enumerate().filter_map(|(i, &(r, beam))| {
        let x = [0, 1, 2].map(|a| v.origin[a] + w[a] * r);
        inside(sp, &x).then(|| (i, key(sp, &x), p.weight * beam))
    }).collect()
}

/// One cell's ray through the box, walked once ([`Walk`]), counted whole into `acc` (the empty scene's reading; the
/// module docs): each echo a hit ([`hits`]) and each stretch it shows clear its path ([`walked`]), weighted as its hits
/// are.
fn ray(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize, acc: &mut FixedCounts) {
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    let Some((_, t1)) = clip(sp, &v.origin, &w) else { return };
    let Some(p) = walked(v, em, k, t1) else { return }; // nothing observed
    let walk = Walk::new(sp, &v.origin, &w);
    for &(from, to, beam) in &p.path {
        for (key, a, b) in walk.within(from, to) {
            accumulate(acc, key, 0.0, p.weight * beam * (b - a));
        }
    }
    for (_, key, h) in hits(v, sp, em, k) {
        accumulate(acc, key, h, 0.0);
    }
}

/// Add (h, l) to a voxel's counts in fixed point; a voxel whose counts come to nothing is dropped.
fn accumulate(acc: &mut FixedCounts, key: [i32; 3], h: f64, l: f64) {
    let e = acc.entry(key).or_insert([0, 0]);
    e[0] += (h * FIXED).round() as i64;
    e[1] += (l * FIXED).round() as i64;
    if *e == [0, 0] {
        acc.remove(&key);
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

/// The field's counts and its part of the cost, kept as the scene changes (the module docs).
///
/// Its part of the cost is `ln NB(∅) − ln NB(E) − plug-in`, each summed over the voxels: the field's term of the
/// empty scene's counts (every echo the medium's) less the scene's, less the plug-in `Σ h ln σ̂` of the hits the
/// elements take (`taken`, the terms' deposits summed) at the field a pulse meets. The plug-in is how the empty scene's
/// pulses already read those echoes inside each cell's ratio ([`super::scene`]'s ray likelihood), so the field's part
/// swaps that reading for the predictive: each echo is scored once, its responsibilities sharing it (3dtrees research
/// 130 formulation.md, slice-soft).
#[derive(Debug, Clone)]
pub struct FieldCounts {
    pub spec: FoliageBox,
    /// The prior's shape and rate.
    pub prior: (f64, f64),
    /// Per voxel, the weighted counts the field explains: the empty scene's less the hits the elements take
    /// (`taken`, the scene's deposits).
    counts: FixedCounts,
    taken: FixedHits,
    /// In fixed point: the field's term `Σ ln NB`, the empty scene's, and the plug-in of `taken`.
    term: i128,
    empty: i128,
    plug: i128,
}

fn fixed(x: f64) -> i128 {
    (x * FIXED).round() as i128
}

impl FieldCounts {
    /// The empty scene's counts of images `views` (every ray whole, every echo the medium's), under the prior in `em`;
    /// nothing taken yet.
    pub fn new(spec: FoliageBox, views: &[&RangeImage], em: &Settings) -> Self {
        let prior = (em.field_prior_mean * em.field_prior_path, em.field_prior_path);
        let counts = views.iter().map(|v| {
            (0..v.cell.len()).into_par_iter().fold(FixedCounts::default, |mut acc, k| {
                ray(v, &spec, em, k, &mut acc);
                acc
            }).reduce(FixedCounts::default, merged)
        }).fold(FixedCounts::default(), merged);
        let mut f = FieldCounts { spec, prior, counts, taken: FixedHits::default(), term: 0, empty: 0, plug: 0 };
        f.term = f.counts.values().map(|&c| f.fixed_term(c)).sum();
        f.empty = f.term;
        f
    }

    /// A voxel's `ln NB` in fixed point.
    fn fixed_term(&self, [h, l]: [i64; 2]) -> i128 {
        fixed(ln_nb(h as f64 / FIXED, l as f64 / FIXED, self.prior.0, self.prior.1))
    }

    /// The plug-in of hits `h` taken in voxel `k` at the field `ext`, in fixed point.
    fn fixed_plug(k: &[i32; 3], h: i64, ext: &Foliage) -> i128 {
        fixed(h as f64 * ext.sigma(k).ln() / FIXED)
    }

    /// The change in the term and in the plug-in were the hits changed by `delta`.
    fn fixed_change(&self, delta: &FixedHits, ext: &Foliage) -> (i128, i128) {
        delta.iter().fold((0, 0), |(t, p), (k, &d)| {
            let c = self.counts.get(k).copied().unwrap_or([0, 0]);
            let b = self.taken.get(k).copied().unwrap_or(0);
            (t + self.fixed_term([c[0] + d, c[1]]) - self.fixed_term(c), p + Self::fixed_plug(k, b - d, ext) - Self::fixed_plug(k, b, ext))
        })
    }

    /// The change in the field's part of the cost were the hits changed by `delta` (deposits given back or taken:
    /// [`give`]), its plug-in read at `ext`; the counts stay as they are.
    pub fn change(&self, delta: &FixedHits, ext: &Foliage) -> f64 {
        let (t, p) = self.fixed_change(delta, ext);
        -(t + p) as f64 / FIXED
    }

    /// Change the hits by `delta`, the plug-in read at `ext`: the change in the field's part of the cost.
    pub fn apply(&mut self, delta: FixedHits, ext: &Foliage) -> f64 {
        let (t, p) = self.fixed_change(&delta, ext);
        (self.term, self.plug) = (self.term + t, self.plug + p);
        for (k, d) in delta {
            add_hits(&mut self.taken, k, -d);
            let e = self.counts.entry(k).or_insert([0, 0]);
            e[0] += d;
            if *e == [0, 0] {
                self.counts.remove(&k);
            }
        }
        -(t + p) as f64 / FIXED
    }

    /// Read the plug-in at the field `ext` (the field a pulse now meets, after a refit).
    pub fn reread(&mut self, ext: &Foliage) {
        self.plug = self.taken.iter().map(|(k, &h)| Self::fixed_plug(k, h, ext)).sum();
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

    /// The field a pulse meets: the posterior from the counts.
    pub fn field(&self, kappa: f64) -> Foliage {
        Foliage::from_counts(kappa, self.spec, &self.counts(), self.prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::norm;

    /// A ray through a box of 1 m voxels: the segments tile the clipped ray, and a uniform field's optical depth is κ t.
    #[test]
    fn the_traversal_tiles_the_ray() {
        let sp = FoliageBox { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 4.0], voxel: 1.0 };
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
        assert_eq!(e.medium(&o, &w).tau(7.0), 0.05 * 7.0);
        let mut f = Foliage::from_counts(0.05, sp, &Counts::from([([0, 0, 0], (0.0, 0.0))]), (0.05, 1.0));
        f.post.insert([0, 1, 0], (1.05 * 1000.0, 1000.0));
        let through = f.medium(&o, &w).tau(100.0) - f.medium(&o, &w).tau(0.0);
        assert!(through > 0.05 * 100.0, "{through}");
        // σ's posterior mean, however much data it rests on (slice 2c)
        let seg = |f: &Foliage| f.medium(&o, &w).tau(100.0) - 0.05 * 100.0;
        let plug = seg(&f);
        f.post.insert([0, 1, 0], (1.05, 1.0));
        assert!((seg(&f) - plug).abs() < 1e-9, "{} vs {plug}", seg(&f));
        // the merge integral: κ's exact over a ray that misses the box, and nothing past a resolved return
        let (z0, slope) = (-2.563, 2.693);
        let flat = Foliage::uniform(0.05).medium(&o, &w);
        let m = flat.merge_integral(10.0, 10.0, 100.0, z0, slope);
        let numeric: f64 = (0..20000).map(|i| 0.05 * 0.005 / (1.0 + (z0 + slope * (i as f64 + 0.5) * 0.005).exp())).sum();
        assert!((m - numeric).abs() < 1e-6, "{m} vs {numeric}");
        assert!(flat.merge_integral(10.0, 20.0, 30.0, z0, slope) < 1e-10);
    }

    /// One walk per ray (3dtrees research 130 slice-structure): the medium's optical depths, hazards and merge
    /// integrals, as sums over the walk's segments, equal a traversal of their own for each (36f7313's), to float
    /// order (1e-12), on rays through a box of uneven σ, at ranges in it, on voxel faces, before it and past it.
    #[test]
    fn one_walk_reads_the_medium_as_traversals_did() {
        let sp = FoliageBox { lo: [0.0, 0.0, 0.0], hi: [4.0, 4.0, 2.5], voxel: 0.5 };
        let mut f = Foliage::from_counts(0.05, sp, &Counts::new(), (0.05, 1.0));
        for i in 0..8 {
            for j in 0..8 {
                for l in 0..5 {
                    f.post.insert([i, j, l], (((i * 7 + j * 3 + l) % 5) as f64 * 0.7 + 0.01, 1.0));
                }
            }
        }
        // a traversal per query, as before
        let tau_hazard = |o: &Point, w: &Point, t: f64| {
            let (mut tau, mut hazard) = (f.kappa * t, f.kappa);
            traverse(&sp, o, w, 0.0, t, |k, a, b| {
                tau += (f.sigma(&k) - f.kappa) * (b - a);
                hazard = if b >= t { f.sigma(&k) } else { f.kappa };
            });
            if !inside(&sp, &[0, 1, 2].map(|j| o[j] + w[j] * t)) {
                hazard = f.kappa;
            }
            (tau, hazard)
        };
        let (z0, slope) = (-2.563, 2.693);
        let merge = |o: &Point, w: &Point, at: f64, from: f64, to: f64| {
            let softplus = |z: f64| if z > 30.0 { z } else { z.exp().ln_1p() };
            let part = |a: f64, b: f64| (b - a) - (softplus(z0 + slope * (b - at)) - softplus(z0 + slope * (a - at))) / slope;
            let mut total = f.kappa * part(from, to);
            traverse(&sp, o, w, from, to, |k, a, b| total += (f.sigma(&k) - f.kappa) * part(a, b));
            total
        };
        let rays = [([-1.0, 0.3, 1.2], [0.8, 0.6, 0.0]), ([-0.5, -0.7, 0.1], [0.6, 0.7, 0.3873]), ([5.0, 2.2, 2.0], [-0.9, 0.1, -0.4243])];
        for (o, w) in rays {
            let w = [0, 1, 2].map(|j| w[j] / norm(&w));
            let m = f.medium(&o, &w);
            for t in (0..140).map(|i| i as f64 * 0.05 + 0.013).chain([1.25, 2.5, 3.0]) {
                let (tau, hazard) = tau_hazard(&o, &w, t);
                assert!((m.tau(t) - tau).abs() < 1e-12 * tau.max(1.0), "tau at {t}: {} vs {tau}", m.tau(t));
                assert_eq!(m.hazard(t), hazard, "hazard at {t}");
                let at = t + 0.02;
                let (a, b) = (merge(&o, &w, t, at, at + 1.3), m.merge_integral(t, at, at + 1.3, z0, slope));
                assert!((a - b).abs() < 1e-12 * a.max(1.0), "merge at {t}: {b} vs {a}");
            }
        }
    }
}
