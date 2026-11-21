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
//! - **The field's term in the cost: the exact formula per voxel** (3dtrees research 130 field-closed-form,
//!   field-slice). Each echo a term takes a share of enters its voxel's formula with its densities under the term and
//!   the medium, read once from the pass over its pulse ([`Echo`]); the scene's gain is the expectation, over the
//!   voxel's density under its empty-scene posterior, of the product of those echoes' factors ([`FieldCounts`]). With
//!   an echo's field weight its cell's, that is the exact marginal over the density and over who made each echo, given
//!   the per-echo densities: the field value a pulse reads cancels from the cost, so no round of EM and no plug-in is
//!   needed, and an element whose echoes a denser medium could explain pays for them with the medium's own penalty
//!   for extra freedom. A move changes it on the voxels whose echoes its terms' deposits change, exactly.
//! - **What a pulse reads of the field** (the medium's hazard, optical depths and merges inside the ray likelihood)
//!   is the empty scene's posterior mean per voxel, `(a₀ + H) / (b₀ + L)`: data alone, the same for every scene. In
//!   the cost it only weighs the echoes between the terms and the medium, which the exact formula then undoes; it
//!   still sets the merge share, the few centimetres of path between a return and its echo, and the terms' own
//!   nodes' modes.
//! - **Grains** ([`Settings::grain_radius`]; 3dtrees research 130 grain-field): σ is a point's extinction among opaque
//!   grains of radius s, and a pulse of footprint w reads it times `(1 + w/2s)²` ([`grain_factor`]), so the field's
//!   counts weigh each stretch's path by that factor too, and each echo and stretch counts the cell's share of a
//!   grain's grown shadow ([`field_weight`]) in place of the cell's weight.
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
//!   share of the emitted beam it took ([`Settings::beam_after`]: shares subtract), as the pulse reads it, so a full hit on leaf or wood reads
//!   as the whole beam, and only the rest goes on. So each stretch after an echo counts as transmission in proportion
//!   to the beam left: none past an opaque hit, most past a sliver. Where echoes carry no reflectance, the beam is
//!   taken to go on whole. The beam left is read as the chance the pulse goes on (the pulse reads the medium past an
//!   echo thinned by it: [`super::pulse`]), so a stretch's path is exposure in proportion to it and each echo is one
//!   hit, whatever came before it. Counting a later echo at the beam its predecessors left as well read σ times that
//!   beam on every stretch after an echo: AM's far scans, whose path in a box is mostly such stretches, saw half the
//!   hits the other scans' field predicts (3dtrees research 130 rca-field-range). (On held-out transmission
//!   this reading is as good as the first echo's alone or better: research 130 slices 12c, 13b. A clumping factor Ω
//!   on the optical depth wasn't supported: slice 12c.) Each echo and each voxel's piece of a ray counts
//!   [`field_weight`] at its own range: without grains the cell's weight, how many neighbouring beams overlap there
//!   (slice-path); with them, the cell's share of a grain.
//! - **Soft attribution: what the field counts** (3dtrees research 130 slice-soft; formulation.md "Soft attribution
//!   and the representation under it"). Every ray counts its whole path, in the beam its echoes left: the ray
//!   likelihood reads the medium's path in every way it explains a pulse (an echo is the medium's, an element's or an
//!   outlier's, and the path to it and on from it is the medium's all the same), so the field does too. Each echo is a
//!   hit, less the share the scene's elements take of it: their responsibilities for it, the posterior over who made
//!   it from the pass over its pulse at its term's nodes, shared among a cell's terms by their front shares (a
//!   term's [`Deposit`]). So each echo's mass sums to one between the field and the elements, and nothing about an
//!   element is set aside. The empty scene's counts read every echo as the medium's.
//! - **Through moves** ([`FieldCounts`]): a move changes the deposits of the terms it recomputes (each term's old
//!   deposit back, its new one out), and the field's term with them, exactly; what a pulse reads is the empty scene's
//!   field, so the cost is a function of the scene alone, and a scene reached by moves is a fresh build of its
//!   elements, to the held alignment's error.
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

/// An echo a term takes a share of, as the voxel's exact formula reads it (3dtrees research 130 field-slice,
/// field-closed-form): its voxel; ω, its count in the field (the field's weight times the term's front share); and,
/// from the pass over its pulse at the read field, `a = σ̂ t / f` and `b = 1 − t`, with t the share the term's layers
/// take, σ̂ the hazard the pulse read at the echo and f the grain factor there ([`grain_factor`]). Its factor in the
/// voxel's formula is `(a / n + b)^ω`, n the voxel's density: the echo's exact evidence for the term over the medium
/// at density n, less the read-field evidence the cell's ratio already holds for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Echo {
    pub voxel: [i32; 3],
    pub omega: f64,
    pub a: f64,
    pub b: f64,
}

/// The least share a term takes of an echo that counts: below it the echo is the medium's (its factor is within
/// ω × 10⁻³ of 1).
pub const T_MIN: f64 = 1e-3;

/// The echoes a term takes a share of, in voxel order: a function of the term alone, so a move's change to the field
/// is its terms' old deposits given back and their new ones taken, exactly.
pub type Deposit = Vec<Echo>;

/// A deposit from echoes: those with a share of at least [`T_MIN`], in voxel order.
pub fn deposit(echoes: impl Iterator<Item = Echo>) -> Deposit {
    let mut d: Deposit = echoes.filter(|e| e.omega > 0.0 && e.b <= 1.0 - T_MIN).collect();
    d.sort_by(|x, y| x.voxel.cmp(&y.voxel));
    d
}

/// A change to what the scene's terms take, per voxel: each echo with its sign (+1 given back to the field, −1 taken
/// from it).
pub type FixedHits = HashMap<[i32; 3], Vec<(i64, Echo)>, std::hash::BuildHasherDefault<VoxelHasher>>;

/// Add a deposit's echoes to `acc` with `sign` (+1 back to the field, −1 out of it).
pub fn give(acc: &mut FixedHits, d: &Deposit, sign: i64) {
    for e in d {
        acc.entry(e.voxel).or_default().push((sign, *e));
    }
}

/// The grain factor at range `range` (3dtrees research 130 grain-field): a pulse of footprint w meets `(1 + w/2s)²`
/// times the extinction a point meets among opaque grains of radius s ([`Settings::grain_radius`]); 1 without grains.
pub fn grain_factor(em: &Settings, range: f64) -> f64 {
    let s = em.grain_radius;
    if s > 0.0 { (1.0 + 0.5 * em.footprint(range) / s).powi(2) } else { 1.0 }
}

/// The weight the field counts an echo, and a stretch of path, with at range `range` on a lattice of `steps`: with
/// grains, one grain stops every cell its grown shadow covers, so each echo counts the cell's share of a grain,
/// `min(1, cell / (π (s + w/2)²))` (3dtrees research 130 grain-field: φ's part in the field); without, the cell's own
/// weight ([`Settings::cell_weight`]).
pub fn field_weight(em: &Settings, range: f64, steps: (f64, f64)) -> f64 {
    let s = em.grain_radius;
    if s > 0.0 {
        let grown = std::f64::consts::PI * (s + 0.5 * em.footprint(range)).powi(2);
        (range * range * steps.0 * steps.1 / grown).min(1.0)
    } else {
        em.cell_weight(range, steps)
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
    /// The grains' radius and the beam's exit width and divergence ([`grain_factor`]): σ is a point's extinction, and
    /// a pulse reads it times the grain factor at its range. A radius of 0: no grains.
    pub grains: [f64; 3],
}

impl Foliage {
    /// The background alone: κ everywhere.
    pub fn uniform(kappa: f64) -> Self {
        Foliage { kappa, spec: None, post: HashMap::new(), grains: [0.0; 3] }
    }

    /// The field's posterior from counts, under the prior (shape `a0`, rate `b0`).
    pub fn from_counts(kappa: f64, spec: FoliageBox, counts: &Counts, (a0, b0): (f64, f64)) -> Self {
        let post = counts.iter().map(|(&k, &(h, l))| (k, (h + a0, l + b0))).collect();
        Foliage { kappa, spec: Some(spec), post, grains: [0.0; 3] }
    }

    /// The grain factor at range `t` ([`grain_factor`]).
    pub fn factor(&self, t: f64) -> f64 {
        let [s, exit, div] = self.grains;
        if s > 0.0 { (1.0 + 0.5 * (exit + div * t) / s).powi(2) } else { 1.0 }
    }

    /// The hazard a pulse reads in voxel `k` at range `t`: σ there times the grain factor (κ where it has no data).
    pub fn hazard_at(&self, k: &[i32; 3], t: f64) -> f64 {
        self.post.get(k).map_or(self.kappa, |&(a, b)| a / b * self.factor(t))
    }

    /// σ's posterior mean in voxel `k` (κ where it has no data). The geometric mean `exp ψ(α) / β`, the bound's own
    /// first-order value, was tried in its place and read far worse: with AM's and the stand's prior shapes far below 1
    /// it is near 0 wherever a voxel holds no echo (3dtrees research 130 slice-path).
    pub fn sigma(&self, k: &[i32; 3]) -> f64 {
        self.post.get(k).map_or(self.kappa, |&(a, b)| a / b)
    }

    /// What the ray from `o` in direction `w` meets of the field ([`Medium`]), from one walk through the box.
    pub fn medium(&self, o: &Point, w: &Point) -> Medium {
        let mut before = 0.0;
        let segs = self.spec.map_or(Vec::new(), |sp| Walk::new(&sp, o, w).0.into_iter().map(|(k, a, b)| {
            let sig = self.hazard_at(&k, 0.5 * (a + b));
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

/// One pulse as everything that reads its path reads it (3dtrees research 130 slice-pyramid: the restart and
/// beam-left conventions in one place): the stretches of its ray it shows clear, each with the beam the echoes before it left ([`Settings::beam_after`]): from the origin to its first echo, then from
/// each echo plus the receiver's resolution to the next (a surface closer behind an echo would have merged with it),
/// and after the last on, unless the pulse is capped ([`RangeImage::capped`]: it says nothing past its last echo);
/// each echo's range; its clear range ([`Settings::free_to`] of its first echo,
/// infinite for a miss); and the beam it leaves. None for a cell without a pulse.
pub struct Walked {
    pub path: Vec<(f64, f64, f64)>,
    pub echoes: Vec<f64>,
    pub clear: f64,
    pub beam: f64,
}

/// Cell `k`'s pulse in image `v` as [`Walked`] reads it.
pub fn walked(v: &RangeImage, em: &Settings, k: usize) -> Option<Walked> {
    if v.cell[k] == Cell::NoPulse {
        return None;
    }
    let (echoes, refls) = (v.echoes(k), v.echo_refls(k));
    let known = if v.capped(k) { echoes.last().map_or(f64::INFINITY, |&r| r as f64) } else { f64::INFINITY };
    let (mut start, mut beam) = (0.0, 1.0);
    let (mut path, mut out) = (Vec::with_capacity(echoes.len() + 1), Vec::with_capacity(echoes.len()));
    for (i, &r) in echoes.iter().enumerate() {
        path.push((start, r as f64, beam));
        out.push(r as f64);
        if let Some(&db) = refls.get(i) {
            beam = em.beam_after(beam, db as f64);
        }
        start = r as f64 + em.resolution();
    }
    path.push((start, known, beam));
    let clear = echoes.first().map_or(f64::INFINITY, |&r| em.free_to(r as f64));
    Some(Walked { path, echoes: out, clear, beam })
}

/// The beam cell `k`'s echoes leave ([`Walked`]'s `beam`, without its stretches).
pub fn beam_left(v: &RangeImage, em: &Settings, k: usize) -> f64 {
    v.echo_refls(k).iter().fold(1.0, |b, &db| em.beam_after(b, db as f64))
}

/// The echoes of image `v`'s cell `k` the field counts as hits, those inside the box: (the echo's index in its
/// pulse, its voxel, its weight), [`Settings::cell_weight`] at the echo's range: one hit, whatever the echoes before it
/// left, since the beam left is the chance the pulse got there and its path counts it (the module docs). The empty
/// scene counts each whole; a scene's elements explain their responsibilities' share (the module docs).
pub fn hits(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize) -> Vec<(usize, [i32; 3], f64, f64)> {
    if v.cell[k] != Cell::Echoes {
        return Vec::new();
    }
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    let p = walked(v, em, k).expect("a pulse");
    let steps = v.steps(k);
    p.echoes.iter().enumerate().filter_map(|(i, &r)| {
        let x = [0, 1, 2].map(|a| v.origin[a] + w[a] * r);
        inside(sp, &x).then(|| (i, key(sp, &x), field_weight(em, r, steps), r))
    }).collect()
}

/// One cell's ray through the box, walked once ([`Walk`]), counted whole into `acc` (the empty scene's reading; the
/// module docs): each echo a hit ([`hits`]) and each stretch it shows clear its path ([`walked`]), each voxel's piece of
/// it weighted at its own range, as an echo there would be (3dtrees research 130 slice-path: the overlap of
/// neighbouring beams is the voxel's, for its hits and its path alike).
fn ray(v: &RangeImage, sp: &FoliageBox, em: &Settings, k: usize, acc: &mut FixedCounts) {
    let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
    if clip(sp, &v.origin, &w).is_none() {
        return;
    }
    let Some(p) = walked(v, em, k) else { return }; // nothing observed
    let steps = v.steps(k);
    let walk = Walk::new(sp, &v.origin, &w);
    for &(from, to, beam) in &p.path {
        for (key, a, b) in walk.within(from, to) {
            let mid = 0.5 * (a + b);
            accumulate(acc, key, 0.0, field_weight(em, mid, steps) * grain_factor(em, mid) * beam * (b - a));
        }
    }
    for (_, key, h, _) in hits(v, sp, em, k) {
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
/// The counts are the empty scene's, every ray whole and every echo the medium's: data alone, the same for every
/// scene. What the scene's terms take enters each voxel's exact formula (3dtrees research 130 field-closed-form,
/// field-slice): with the voxel's density n under its posterior from those counts, Gamma(a₀ + H, b₀ + L), the scene
/// gains `ln E[Π_e (a_e / n + b_e)^(ω_e)]` over the echoes its terms take a share of ([`Echo`]), and the field's part of
/// the cost is minus that, summed over the voxels. With every echo's field weight its cell's weight, this is the exact
/// marginal over the density and the attributions, given each echo's densities under the terms and the medium:
/// the field value the cells' ratios read cancels from the cost. It is computed on a grid in ln n, from 10⁻⁸ (below
/// which the integrand is a power of n, integrated in closed form) to well past the empty posterior, each voxel's sums
/// over its echoes kept in fixed point, so a move and its undo restore it exactly.
#[derive(Debug, Clone)]
pub struct FieldCounts {
    pub spec: FoliageBox,
    /// The prior's shape and rate.
    pub prior: (f64, f64),
    /// The grains' radius and the beam's exit width and divergence ([`Foliage`]'s `grains`).
    pub grains: [f64; 3],
    /// Per voxel, the empty scene's weighted counts (hits, path).
    counts: FixedCounts,
    /// Per voxel with echoes taken, their sums ([`Taken`]).
    taken: HashMap<[i32; 3], Taken, std::hash::BuildHasherDefault<VoxelHasher>>,
    /// In fixed point: the scene's gain over the empty scene, `Σ_v ln E[…]`.
    term: i128,
}

/// A voxel's sums over the echoes taken there, in fixed point: how many, Σ ω, Σ ω (1 − b) (the hits taken, as
/// reported), Σ ω ln a (the closed-form tail) and, per node of its grid, Σ ω ln(a / n + b).
#[derive(Debug, Clone, Default)]
struct Taken {
    count: i64,
    omega: i64,
    hits: i64,
    tail: i64,
    nodes: Vec<i64>,
}

/// The lowest density on a voxel's grid (1/m).
const N_LO: f64 = 1e-8;

/// A voxel's grid in u = ln n, from its empty counts: (lowest u, step, nodes), midpoints `u_lo + (k + ½) du`.
fn grid((a0, b0): (f64, f64), h: f64, l: f64) -> (f64, f64, usize) {
    let al = a0 + h;
    let hi = ((al + 6.0 * (al + 1.0).sqrt() + 10.0) / (b0 + l)).ln();
    let lo = N_LO.ln();
    let du = (0.25 / (al + 1.0).sqrt()).min(0.05);
    (lo, du, ((hi - lo) / du).ceil() as usize)
}

fn fixed(x: f64) -> i128 {
    (x * FIXED).round() as i128
}

fn fx(x: f64) -> i64 {
    (x * FIXED).round() as i64
}

fn lse(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = xs.collect();
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
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
        FieldCounts { spec, prior, grains: [em.grain_radius, em.beam_exit, em.beam_divergence], counts, taken: Default::default(), term: 0 }
    }

    /// Voxel `k`'s empty counts (hits, path).
    fn empty(&self, k: &[i32; 3]) -> (f64, f64) {
        self.counts.get(k).map_or((0.0, 0.0), |&[h, l]| (h as f64 / FIXED, l as f64 / FIXED))
    }

    /// One echo's sums on voxel `k`'s grid, times `sign`.
    fn add_echo(&self, t: &mut Taken, k: &[i32; 3], e: &Echo, sign: i64) {
        let (h, l) = self.empty(k);
        let (lo, du, n) = grid(self.prior, h, l);
        if t.nodes.is_empty() {
            t.nodes = vec![0; n];
        }
        t.count += sign;
        t.omega += sign * fx(e.omega);
        t.hits += sign * fx(e.omega * (1.0 - e.b));
        t.tail += sign * fx(e.omega * e.a.ln());
        for (i, x) in t.nodes.iter_mut().enumerate() {
            let nn = (lo + (i as f64 + 0.5) * du).exp();
            *x += sign * fx(e.omega * (e.a / nn + e.b).ln());
        }
    }

    /// Voxel `k`'s gain `ln E[…]` from its sums.
    fn gain(&self, k: &[i32; 3], t: &Taken) -> f64 {
        if t.count == 0 {
            return 0.0;
        }
        let (h, l) = self.empty(k);
        let (a0, b0) = self.prior;
        let (lo, du, _) = grid(self.prior, h, l);
        let (al, beta) = (a0 + h, b0 + l);
        let free = (al - t.omega as f64 / FIXED).max(a0);
        let node = |i: usize| lo + (i as f64 + 0.5) * du;
        let with = lse(t.nodes.iter().enumerate().map(|(i, &x)| {
            let u = node(i);
            al * u - beta * u.exp() + x as f64 / FIXED + du.ln()
        }).chain(std::iter::once(free * lo + t.tail as f64 / FIXED - free.ln())));
        let without = lse((0..t.nodes.len()).map(|i| {
            let u = node(i);
            al * u - beta * u.exp() + du.ln()
        }).chain(std::iter::once(al * lo - al.ln())));
        with - without
    }

    /// The change in the scene's gain were the echoes taken changed by `base` and then by `delta`.
    fn fixed_change(&self, base: &FixedHits, delta: &FixedHits) -> i128 {
        delta.iter().map(|(k, es)| {
            let mut t = self.taken.get(k).cloned().unwrap_or_default();
            for (sign, e) in base.get(k).into_iter().flatten() {
                self.add_echo(&mut t, k, e, -sign);
            }
            let before = fixed(self.gain(k, &t));
            for (sign, e) in es {
                self.add_echo(&mut t, k, e, -sign);
            }
            fixed(self.gain(k, &t)) - before
        }).sum()
    }

    /// The change in the field's part of the cost were the echoes taken changed by `delta` (deposits given back or
    /// taken: [`give`]); nothing changes.
    pub fn change(&self, delta: &FixedHits) -> f64 {
        self.change_after(&FixedHits::default(), delta)
    }

    /// The change in the field's part of the cost were the echoes taken changed by `delta`, from what a change `base`
    /// would leave (a proposal's: [`super::scene::Proposal`]), exactly as if `base` were applied first.
    pub fn change_after(&self, base: &FixedHits, delta: &FixedHits) -> f64 {
        -self.fixed_change(base, delta) as f64 / FIXED
    }

    /// Change the echoes taken by `delta`: the change in the field's part of the cost.
    pub fn apply(&mut self, delta: FixedHits) -> f64 {
        let mut total = 0i128;
        for (k, es) in delta {
            let mut t = self.taken.remove(&k).unwrap_or_default();
            let before = fixed(self.gain(&k, &t));
            for (sign, e) in &es {
                self.add_echo(&mut t, &k, e, -sign);
            }
            let after = fixed(self.gain(&k, &t));
            total += after - before;
            if t.count != 0 {
                self.taken.insert(k, t);
            }
        }
        self.term += total;
        -total as f64 / FIXED
    }

    /// The field's part of the scene's cost (above).
    pub fn cost(&self) -> f64 {
        -self.term as f64 / FIXED
    }

    /// The weighted counts per voxel, (hits, path): the empty scene's less the hits the terms take, at the shares their
    /// pulses read.
    pub fn counts(&self) -> Counts {
        self.counts_after(&FixedHits::default())
    }

    /// The weighted counts per voxel, (hits, path), were the echoes taken changed by `delta`.
    pub fn counts_after(&self, delta: &FixedHits) -> Counts {
        let mut c = self.counts.clone();
        for (k, t) in &self.taken {
            if let Some(e) = c.get_mut(k) {
                e[0] -= t.hits;
            }
        }
        for (k, es) in delta {
            if let Some(x) = c.get_mut(k) {
                for (sign, e) in es {
                    x[0] += sign * fx(e.omega * (1.0 - e.b));
                }
            }
        }
        c.iter().map(|(&k, &[h, l])| (k, (h as f64 / FIXED, l as f64 / FIXED))).collect()
    }

    /// The field a pulse reads: the posterior from the empty scene's counts (data alone, the same for every scene).
    pub fn field(&self, kappa: f64) -> Foliage {
        let counts: Counts = self.counts.iter().map(|(&k, &[h, l])| (k, (h as f64 / FIXED, l as f64 / FIXED))).collect();
        let mut f = Foliage::from_counts(kappa, self.spec, &counts, self.prior);
        f.grains = self.grains;
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::norm;

    /// The voxel's exact formula (research 130 field-slice): an echo a term takes wholly gains `ω ln σ̂ + ln Γ(α − ω)
    /// − ln Γ(α) + ω ln β` (the density's posterior moment), one it takes none of gains nothing, and a half share
    /// matches a fine direct integral; giving the echoes back restores the term exactly.
    #[test]
    fn the_voxel_formula_is_the_posterior_moment() {
        let spec = FoliageBox { lo: [0.0; 3], hi: [1.0; 3], voxel: 1.0 };
        let (h, l, a0, b0) = (3.0, 4.0, 0.05, 0.5);
        let mut f = FieldCounts { spec, prior: (a0, b0), grains: [0.0; 3], counts: Default::default(), taken: Default::default(), term: 0 };
        f.counts.insert([0, 0, 0], [fx(h), fx(l)]);
        let (al, beta, w, sig) = (a0 + h, b0 + l, 0.7, 0.9);
        let mut d = FixedHits::default();
        give(&mut d, &vec![Echo { voxel: [0, 0, 0], omega: w, a: sig, b: 0.0 }], -1);
        let want = w * sig.ln() + libm::lgamma(al - w) - libm::lgamma(al) + w * beta.ln();
        assert!((-f.change(&d) - want).abs() < 1e-6, "{} vs {want}", -f.change(&d));
        let mut none = FixedHits::default();
        give(&mut none, &vec![Echo { voxel: [0, 0, 0], omega: w, a: 0.0, b: 1.0 }], -1);
        assert!(f.change(&none).abs() < 1e-9);
        let mut half = FixedHits::default();
        give(&mut half, &vec![Echo { voxel: [0, 0, 0], omega: w, a: 0.5 * sig, b: 0.5 }], -1);
        let n = 400_000;
        let (lo, hi) = (-40.0f64, 4.0f64);
        let du = (hi - lo) / n as f64;
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..n {
            let u = lo + (i as f64 + 0.5) * du;
            let base = (al * u - beta * u.exp()).exp();
            num += base * (0.5 * sig / u.exp() + 0.5).powf(w);
            den += base;
        }
        let want = (num / den).ln();
        assert!((-f.change(&half) - want).abs() < 1e-5, "{} vs {want}", -f.change(&half));
        let (c0, ch) = (f.cost(), f.change(&half));
        f.apply(half.clone());
        assert!((f.cost() - c0 - ch).abs() < 1e-9);
        let mut back = FixedHits::default();
        give(&mut back, &vec![Echo { voxel: [0, 0, 0], omega: w, a: 0.5 * sig, b: 0.5 }], 1);
        f.apply(back);
        assert_eq!(f.cost(), c0);
        assert!(f.taken.is_empty());
    }

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

    /// A voxel's path counts with the weight of its own range, wherever the ray ends (3dtrees research 130 slice-path):
    /// two neighbouring rays of the test scan, one stopping on the pole at 4 m, one passing on to the wall at 8 m, count
    /// the same weighted path per metre in the voxels before the pole. Read at each ray's first echo, the wall's ray
    /// counted about three times the pole's there, so a near voxel read thin.
    #[test]
    fn a_voxels_path_counts_at_its_own_range() {
        let v = super::super::fixtures::pole_image();
        let em = Settings::default();
        let sp = super::super::fixtures::field_box();
        let cell = |x: &Point| {
            let (l, c, _) = v.cell_of(x).unwrap();
            l.round() as usize * v.n_cols + c.round() as usize
        };
        let (pole, wall) = (cell(&[0.0, 3.97, 0.0]), cell(&[0.28, 8.0, 0.0]));
        assert!(v.echoes(pole)[0] < 4.1 && v.echoes(wall)[0] > 7.9);
        let per_metre = |k: usize| {
            let mut acc = FixedCounts::default();
            ray(&v, &sp, &em, k, &mut acc);
            let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
            // the voxels from the box's face at 2.9 m to 3.8 m (rows 0..9 across the beam), over that length of ray
            let path: i64 = acc.iter().filter(|(key, _)| key[1] < 9).map(|(_, c)| c[1]).sum();
            path as f64 / FIXED / (0.9 / w[1])
        };
        let (a, b) = (per_metre(pole), per_metre(wall));
        assert!((a / b - 1.0).abs() < 0.01, "{a} against {b}");
    }

    /// The field recovers the medium past an echo as the pulse reads it (3dtrees research 130 rca-field-range): every
    /// pulse first meets a sliver at 3 m whose reflectance leaves half the beam ([`Settings::beam_after`]), then a
    /// medium of σ = 1/m, which past the sliver the pulse reads thinned by the beam left (`pulse`'s `seen`): its
    /// second echo where an exponential free path of rate σ/2 from the receiver's resolution ends. The empty scene's
    /// counts over the box behind read σ: each later echo one hit, its stretch's path at the beam left. Counting the
    /// echo at the beam left too read σ/2.
    #[test]
    fn the_field_reads_the_medium_past_an_echo() {
        use crate::shots::reconstruct::dir;
        let em = Settings::default();
        let (sigma, left) = (1.0, 0.5);
        let sliver = em.refl_full_hit() + 10.0 * (1.0 - left).log10();
        assert!((em.beam_after(1.0, sliver) - left).abs() < 1e-9);
        let u = |s: u64| ((s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let (mut t, mut d, mut start, mut range, mut refl) = (Vec::new(), Vec::new(), vec![0u32], Vec::new(), Vec::new());
        for l in 0..200usize {
            for k in 0..400usize {
                let s = (l * 400 + k) as u64 * 2 + 1;
                t.push(l as f64 * 6.6e-3 + k as f64 * 0.825e-6);
                d.push(dir(84.0 + 0.03 * k as f64, -3.0 + 0.03 * l as f64));
                range.push(3.0f32);
                refl.push(sliver as f32);
                range.push((3.0 + em.resolution() - u(s).ln() / (left * sigma)) as f32);
                refl.push(-30.0);
                start.push(range.len() as u32);
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = RangeImage::new(&t, &d, start, range, (84.0, 96.0), id, [0.0; 3], 0, &super::super::range_image::RangeImageParams::default());
        v.echo_refl = refl;
        // a scanner of two targets: past them, censored
        v.instrument = std::sync::Arc::new(super::super::instrument::Instrument { max_targets: 2, ..Default::default() });
        let sp = FoliageBox { lo: [-1.0, 4.0, -1.0], hi: [1.0, 6.0, 1.0], voxel: 0.25 };
        let (h, l) = FieldCounts::new(sp, &[&v], &em).counts().values().fold((0.0, 0.0), |(h, l), &(a, b)| (h + a, l + b));
        assert!((h / l / sigma - 1.0).abs() < 0.03, "the field reads {} against σ = {sigma}", h / l);
    }

    /// An echo's reflectance is its share of the emitted beam, so two echoes taking 40 % each leave 20 %, not 36 %, and
    /// a third taking 40 % leaves nothing (3dtrees research 130 observation).
    #[test]
    fn shares_of_the_emitted_beam_subtract() {
        let em = Settings::default();
        let x = em.refl_full_hit() + 10.0 * 0.4f64.log10();
        let two = em.beam_after(em.beam_after(1.0, x), x);
        assert!((two - 0.2).abs() < 1e-9, "{two}");
        assert_eq!(em.beam_after(two, x), 0.0);
        assert_eq!(em.beam_after(0.7, f64::NAN), 0.7);
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
