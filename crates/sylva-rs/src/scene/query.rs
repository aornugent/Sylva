// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What the scene says about elements without moving them (3dtrees research 130): predictions per cell, counts, the
//! information and the divergence.

use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use super::dual::{Dual, Real};
use super::pulse::{Outcome, mixture_ll, cell_view, front_echo, depth_sd, Surface, Pulse};
use super::range_image::Cell;
use super::render::{MOSTLY, Form, Element, Layer, placed, render, keep, front_shares, layer};
use super::scene::{Proposal, Scene, View};
use super::settings::Settings;
use super::term::{ChainKey, chain_key, slots, term_nodes, term_classes, Term};

/// What an element predicts in one cell it fronts ([`Scene::predict`]), on the path the likelihood scores: its fill,
/// the chance its echo is recorded, the chance the pulse arrives (meets nothing of the foliage field before the
/// element's return), the echo's predicted depth and sd ([`front_echo`]) and reflectance (dB, at the element's
/// declared reflectance and its own section radius, without the shape mismatch's blur; NaN where that section
/// misses the beam), the mean cos ι over the part of the footprint the element meets, its term's surface class,
/// the observed first echo (NaN for none) and its reflectance (NaN unknown), the [`Outcome`], whether any echo the
/// likelihood reads of the cell's pulse is on the element (`read`: [`Outcome::of`] for each), the foliage field's own
/// merge share behind it (`merge`: [`merges`]' `1 − exp(−b ∫ σ (1 − p))`, before the site's map, which is fitted on
/// it: 3dtrees research 130 slice-products), the cell's weight ([`Settings::cell_weight`]), the footprint (m) and where
/// its ray passes the axis (across, along). From the same pass over the pulse ([`Pulse::read`]): the expected number
/// of returns the receiver records from the cell's occluders (`returns`: its layers and the ground, merged, detected
/// and capped; the medium's echoes are the foliage field's), the share of the beam the element leaves behind it
/// (`beam`), and how many of the pulse's echoes are the element's in expectation (`owned`: its responsibilities summed
/// over the echoes, the soft `read`). `beam` is NaN and `owned` 0 where the element lies past the beam's end.
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
    pub read: bool,
    pub merge: f64,
    pub weight: f64,
    pub footprint: f64,
    pub across: f64,
    pub along: f64,
    pub returns: f64,
    pub beam: f64,
    pub owned: f64,
}

/// How well one scan pins some elements' geometry down ([`Scene::information`]): in their parameters (n of them), the
/// cost's slope (n), the expected (Fisher) information (n × n, row-major) and the slope's variance were the cells
/// independent (n × n).
#[derive(Debug, Clone)]
pub struct Information {
    pub slope: Vec<f64>,
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

/// The numbers of [`Scene::information`]'s pass: the slope in an element's parameters (its first five; a patch's
/// first four) and in the cell's own nodes, its shift along the beam ([`SHIFT`]) and its reflectance level ([`LEVEL`]).
pub(super) type Geometry = Dual<7>;

pub(super) const SHIFT: usize = 5;

pub(super) const LEVEL: usize = 6;

/// What a cell's echo tells per unit of its slope (research 130 step 3a), the one reading of a cell [`Scene::information`]
/// and [`Scene::divergence`] share: in its chance of an echo `p` (a Bernoulli, `p` kept within [1e-3, 1 − 1e-3]) and,
/// given one, in its depth (of variance `var`, the return's own, without the shifts' spread) and in its reflectance (the
/// per-echo noise, wider at fill `fill`), each of these two a t density's share `(ν + 1) / (ν + 3)` of a Gaussian's.
pub(super) fn cell_information(em: &Settings, p: f64, var: f64, fill: f64) -> [f64; 3] {
    let p = p.clamp(1e-3, 1.0 - 1e-3);
    let (k_depth, k_refl) = ((em.noise_dof + 1.0) / (em.noise_dof + 3.0), (em.refl_dof + 1.0) / (em.refl_dof + 3.0));
    [1.0 / (p * (1.0 - p)), p * k_depth / var.max(1e-12), p * k_refl / em.refl_noise.hypot(em.refl_edge * (1.0 - fill.clamp(0.0, 1.0))).powi(2)]
}

/// One cell as [`Scene::information`] reads it, in the numbers `P` of its pass: its share of the cost (its weighted log
/// likelihood against the empty scene's, with the sign the cost has), its weight's value, and the element's echo there
/// (the front return of its most likely way back, [`Receiver::front`]): the chance the pulse returns it (it arrives,
/// and is recorded), its depth and its reflectance at its mean fill ([`Return::reflectance`]; −200 dB or less where
/// no section radius meets the beam), with its depth's own sd.
pub(super) struct CellReading<P> {
    pub(super) cost: P,
    pub(super) weight: f64,
    /// Its layers' front shares ([`front_shares`]).
    pub(super) shares: Vec<f64>,
    pub(super) echo: Option<([P; 3], f64)>,
}

/// One scan's evidence on n quantities from the cells of some elements, with the latents the cells share
/// marginalised out once (research 130 step 3a, slice 4): the scan alignment error's nodes about each term (shared
/// by neighbours), and each term's stem front shift and reflectance level, each a unit node scaled as the cost scales
/// it ([`term_nodes`]). Each observable adds its slope in the quantities and its loading on the nodes, at its
/// information weight and the cell's weight w; the information over (quantities, nodes) is kept at w and at w² (the
/// gradient's variance were the cells independent).
pub(super) struct Shared {
    pub(super) n: usize,
    pub(super) nodes: HashMap<[i64; 3], usize>,
    /// Per element: its term's hats, the index of its front shift node (its reflectance level's is next) and that
    /// node's prior sd ([`term_nodes`]).
    pub(super) elems: Vec<(Vec<([i64; 3], f64)>, usize, f64)>,
    pub(super) m: nalgebra::DMatrix<f64>,
    pub(super) j: nalgebra::DMatrix<f64>,
}

impl Shared {
    /// For `elems`, a group's members sharing one term's nodes ([`ChainKey`]: hats at their middles' mean, one front
    /// shift and one reflectance level), as the cost has them when the group is these elements.
    pub(super) fn new(n: usize, elems: &[Element], em: &Settings) -> Shared {
        let mut nodes: HashMap<[i64; 3], usize> = HashMap::new();
        let slot = slots(&elems.iter().enumerate().map(|(k, e)| chain_key(k as u32, e)).collect::<Vec<_>>());
        let terms = slot.iter().max().map_or(0, |m| m + 1);
        let tn: Vec<_> = (0..terms).map(|t| term_nodes(&elems.iter().zip(&slot).filter(|(_, &u)| u == t).map(|(e, _)| e).collect::<Vec<_>>(), em)).collect();
        for (k, _) in tn.iter().flat_map(|t| &t.0) {
            let i = nodes.len();
            nodes.entry(*k).or_insert(i);
        }
        let first = nodes.len();
        let elems: Vec<_> = slot.iter().map(|&t| (tn[t].0.clone(), first + 2 * t, tn[t].1)).collect();
        let size = n + first + 2 * terms;
        Shared { n, nodes, elems, m: nalgebra::DMatrix::zeros(size, size), j: nalgebra::DMatrix::zeros(size, size) }
    }

    /// An observable's loading on the nodes, in element `k`'s cell, from its slope in the cell's shift along the beam
    /// (`d_shift`: over the term's alignment hats, each at its class's along-beam spread `align_sd`, and its stem front
    /// shift) and in its reflectance level (`d_level`, at that node's prior sd `level_sd`).
    pub(super) fn load(&self, k: usize, align_sd: f64, d_shift: f64, d_level: f64, level_sd: f64) -> Vec<(usize, f64)> {
        let (h, own, front) = &self.elems[k];
        let mut out = Vec::new();
        if d_shift != 0.0 {
            out.extend(h.iter().map(|(key, w)| (self.nodes[key], w * align_sd * d_shift)));
            out.push((*own, front * d_shift));
        }
        if d_level != 0.0 {
            out.push((own + 1, level_sd * d_level));
        }
        out
    }

    pub(super) fn add(&mut self, g: &[f64], load: &[(usize, f64)], info: f64, w: f64) {
        let mut x = nalgebra::DVector::<f64>::zeros(self.m.nrows());
        x.rows_mut(0, self.n).copy_from_slice(g);
        for &(i, v) in load {
            x[self.n + i] += v;
        }
        let outer = &x * x.transpose();
        self.m += &outer * (w * info);
        self.j += outer * (w * w * info);
    }

    /// The information on the quantities (n × n, row-major) with the nodes marginalised under their unit priors (a
    /// Schur complement), and the variance of the efficient gradient (the quantities' less β = M_qn (M_nn + I)⁻¹ times
    /// the nodes', the priors' included).
    pub(super) fn finish(self) -> [Vec<f64>; 2] {
        let (n, l) = (self.n, self.m.nrows() - self.n);
        let block = |x: &nalgebra::DMatrix<f64>, r: usize, c: usize, nr: usize, nc: usize| x.view((r, c), (nr, nc)).into_owned();
        let eye = nalgebra::DMatrix::<f64>::identity(l, l);
        let inv = (block(&self.m, n, n, l, l) + &eye).try_inverse().expect("the nodes' precision is at least their unit prior");
        let (mq, jq) = (block(&self.m, 0, n, n, l), block(&self.j, 0, n, n, l));
        let beta = &mq * inv;
        let m = block(&self.m, 0, 0, n, n) - &beta * mq.transpose();
        let j = block(&self.j, 0, 0, n, n) - &beta * jq.transpose() - &jq * beta.transpose() + &beta * (block(&self.j, n, n, l, l) + eye) * beta.transpose();
        [m, j].map(|x| x.transpose().iter().copied().collect())
    }
}

impl Scene {
    /// What a proposal's elements predict in image `s` ([`Scene::check_scan`], [`Scene::check_proposal`]) against
    /// the scene less the elements it takes out, per cell one of them fronts (its index in `put_in` is the prediction's
    /// `elem`), on the path the likelihood scores ([`front_echo`], [`cell_view`]): they are rendered for this, not
    /// kept, and only the scene's layers in their cells enter their k-buffer, so their front cells and classes are those
    /// of the whole set. It reads the layers wholly in their cells ([`Layer::full`]): a partial layer's evidence is the
    /// cost's, its prediction too faint to report.
    pub fn predict(&self, p: &Proposal, s: usize) -> Vec<CellPrediction> {
        let (v, em, es) = (&*self.views[s], &self.em, &p.put_in[..]);
        let theirs = |l: &&Layer| l.full() && !p.take_out.contains(&l.elem);
        // per cell they enter wholly, their layers and the scene's (numbered after them, in id order), kept
        let mut cells: BTreeMap<usize, Vec<Layer>> = BTreeMap::new();
        for (k, e) in es.iter().enumerate() {
            for (c, l) in render(v, e, k as u32, em).into_iter().filter(|(_, l)| l.full()) {
                cells.entry(c).or_default().push(l);
            }
        }
        let buf = &self.buffers[s];
        let ids: BTreeSet<u32> = cells.keys().flat_map(|&c| buf.get(c).iter().filter(theirs).map(|l| l.elem)).collect();
        let at: HashMap<u32, u32> = ids.iter().enumerate().map(|(k, &i)| (i, (es.len() + k) as u32)).collect();
        let cells: Vec<(usize, Vec<Layer>)> = cells.into_iter().map(|(c, mut ls)| {
            ls.extend(buf.get(c).iter().filter(theirs).map(|l| Layer { elem: at[&l.elem], ..*l }));
            (c, keep(ls))
        }).collect();
        let n = self.elems.len() as u32;
        let els: Vec<Element> = es.iter().copied().chain(ids.iter().map(|&i| self.elems[i as usize].unwrap())).collect();
        let owners: Vec<ChainKey> = es.iter().enumerate().map(|(k, e)| chain_key(n + k as u32, e)).chain(ids.iter().zip(&els[es.len()..]).map(|(&i, e)| chain_key(i, e))).collect();
        let other = v.epoch != self.ref_epoch;
        let cls = term_classes(v, &cells, &slots(&owners), em);
        let mut out: Vec<CellPrediction> = cells.iter().filter(|(cell, ls)| (ls[0].elem as usize) < es.len() && v.cell[*cell] != Cell::NoPulse).map(|(cell, ls)| {
            let cell = *cell;
            let ly = ls[0];
            let k = ly.elem as usize;
            let cv = cell_view(v, cell, ls.clone(), em, &self.sc, &self.ext);
            let pulse = &cv.pulses[0].1;
            let r = pulse.read(0.0, 0.0, em);
            let fe = front_echo(&r.rx, cls[k], es[k].front_shift_sd(em), other, em);
            let (depth, sd) = depth_sd(fe.as_ref());
            // the element's occluder: none where it lies past the beam's end
            let at = pulse.occluders.iter().position(|o| matches!(o.surface, Surface::Layer(l) if l.elem == ly.elem));
            CellPrediction {
                cell, elem: ly.elem, fill: ly.f, recorded: fe.map_or(0.0, |f| f.s), arrives: fe.map_or(0.0, |f| (-pulse.occluders[f.first].tau).exp()), surface: cls[k], depth, sd,
                refl: fe.map_or(f64::NAN, |f| if f.db[1] > -200.0 { f.db[1] } else { f64::NAN }), cos: ly.cos, obs: cv.obs,
                obs_refl: pulse.refls.first().map_or(f64::NAN, |&x| x as f64),
                outcome: Outcome::of(cv.obs, depth, sd), read: pulse.echoes.iter().any(|&r| Outcome::of(r as f64, depth, sd) == Outcome::On),
                merge: at.map_or(0.0, |i| pulse.occluders[i].merge.field), weight: cv.wt, footprint: em.footprint(ly.t), across: ly.x, along: ly.s,
                returns: r.rx.expected(pulse.cap).iter().sum(), beam: at.map_or(f64::NAN, |i| r.rx.members[i].beam), owned: at.map_or(0.0, |i| r.owned(i)),
            }
        }).collect();
        out.sort_by_key(|p| p.cell);
        out
    }

    /// An element's counts against the scene less the elements `take_out`, per image, from its predictions
    /// ([`Scene::predict`]). `shift` (m, per view; empty for none) is subtracted from the observed echoes first.
    pub fn counts(&self, e: Element, take_out: &[u32], shift: &[f64]) -> Vec<ElementCounts> {
        let with = |e: Element| Proposal { put_in: vec![e], take_out: take_out.to_vec() };
        let phi = self.em.discount;
        (0..self.views.len()).into_par_iter().map(|s| {
            let mut out = ElementCounts { scan: s as u32, ..ElementCounts::default() };
            let sh = shift.get(s).copied().unwrap_or(0.0);
            // the mismatch's from the element drawn sharp (its own mismatch left out of the depths it is measured
            // from), the opacity's from the element as given
            for p in self.predict(&with(Element { mismatch: 0.0, ..e }), s) {
                let r = p.obs - sh - p.depth;
                if p.fill > MOSTLY && Outcome::of(p.obs - sh, p.depth, p.sd) == Outcome::On {
                    out.mismatch_ss += phi * p.weight * (r * r - p.sd * p.sd).max(0.0);
                    out.mismatch_w += phi * p.weight;
                    out.mismatch_cells += 1;
                }
            }
            for p in self.predict(&with(e), s).into_iter().filter(|p| p.fill > 0.9) {
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

    /// How well the scans pin some elements' geometry down, in the scene as proposal `p` would leave it
    /// ([`Scene::check_proposal`]; `ids` live there, a proposal's own numbered from [`Scene::next_id`]; one element, or
    /// several jointly): per scan, the cost's slope and its expected (Fisher) information in their parameters, each
    /// element's in turn, the rest of the scene held. So a refit is a sequence of reads, nothing kept. A stick's parameters are `ln r`, its axis moved across itself (two
    /// directions perpendicular to it, m) and tilted about its middle (the same two directions, rad at its ends'
    /// half-length); a patch's are `ln r` and its centre moved (x, y, z, m): n is 5 a stick, 4 a patch. The frame is
    /// the element's own, the same in every scan, so scans' information adds. A scan that doesn't see them gives zeros.
    ///
    /// **One pass per cell** (research 130 slice-derivatives): the ray likelihood read with each element's layer in
    /// [`Geometry`] numbers, forward derivatives in its parameters (through its rendering: the fill, the depth over
    /// the footprint, the far side) and in the cell's own nodes (its shift along the beam and its reflectance level),
    /// at the alignment held and each term's own nodes at their mode. From it:
    /// - **`slope`** (n): the cost's slope, over every cell an element's layer is in (its own, and those it lies
    ///   behind another's front in), each at its term's nodes: the slope of the fit with the nuisances at their modes
    ///   (an envelope: their own slopes vanish there), the Laplace steps' log-determinants held.
    /// - **`expected`** (n × n, row-major; research 130 step 3a): what the elements' own cells would tell on average.
    ///   Each cell is read as an echo or none (a Bernoulli in the chance p the element's echo comes back) and, given one,
    ///   its depth μ and reflectance ρ (at its mean fill: [`Return::reflectance`]; [`cell_information`]):
    ///   `Σ_c w_c [∂p ∂pᵀ / (p (1 − p)) + p (k_d ∂μ ∂μᵀ / sd² +
    ///   k_r ∂ρ ∂ρᵀ / s_ρ²)]`, w_c the cell weight and sd the depth's own. Each observable also loads the latents the
    ///   cells share by its slope in the cell's own nodes (the depth on the shift, so on the alignment's hats and the
    ///   stem front shift; the reflectance and the chance of an echo on the level), and they are marginalised out
    ///   under their priors (a Schur complement, [`Shared`]). It is positive semi-definite, reads a direction the cells
    ///   don't see as 0, and leaves out what other elements' cells say.
    ///
    /// The observed curvature is not given: the fill's curvature is unbounded at the beam's edge, and the expected
    /// information is the calibrated one (research 130 step 3b; optimisation.md). Slope and expected information
    /// together are a Fisher-scoring step.
    ///
    /// **Several elements** (research 130 slice 4: a pair's joint information): each element's cells are read with
    /// the others in place, a cell's observables in every element whose layer is in it, and the latents are
    /// marginalised once over all of them ([`Shared`]): each element's own stem front shift and reflectance level, and
    /// the alignment's nodes, which neighbours share. So the alignment, which moves neighbours' depths together,
    /// cancels in their depth difference, where each element's own information charges it in full. Far apart (no node
    /// or cell in common) the information is the elements' own, side by side. Members of one group (a chain) share
    /// one term's nodes ([`ChainKey`]), as their cost does when `ids` are the whole group; a chain's knots are a Jacobian
    /// away from its sticks' parameters (research 130 slice 6).
    ///
    /// **What it means under the evidence discount** (research 130 step 3b). Each cell's log-likelihood counts
    /// `w_c = 1 / (φ m)`, so the cost is a quasi-likelihood and E is its curvature: an sd read from E⁻¹ is honest only
    /// where the cells' errors are correlated as φ m declares. In general the estimate's variance is the sandwich
    /// `E⁻¹ J E⁻¹`, J the gradient's actual variance (the inverse of Godambe's information `E J⁻¹ E`), and model
    /// comparison by cost differences needs the composite-likelihood penalty `tr(J E⁻¹)`, not a parameter count. φ
    /// enters E once, and J twice: beside E comes the **gradient's variance were the cells independent**
    /// (`gradient_variance`: the same sums at `w_c²`, its latents' share taken out as E's is, their priors' included),
    /// so `E⁻¹ J E⁻¹` is the variance with no correlation between cells, a lower bound, and E⁻¹ the variance at the
    /// declared φ. On the stand's true sticks (step3b.md), fits from the truth fall within 1.96 sd of E⁻¹ along the
    /// sloppiest direction 95 % of the time at 3 footprints across or more; the sandwich's sds are 1.5–2.6 times too
    /// small (the cells are correlated); and below 1.5 footprints across the cost has minima far from the truth that a
    /// local curvature can't see, so E's sds don't hold there.
    ///
    /// **ln r is the space radius is unbiased in:** the stand's fitted ln r sits −0.003 from the truth (median, R ≥
    /// 1 cm); r's own mean is `exp(ln r + ½ sd²)`, so an average of radii should be taken in ln r.
    pub fn information(&self, p: &Proposal, ids: &[u32]) -> Vec<Information> {
        let es: Vec<Element> = ids.iter().map(|&id| self.element_after(p, id).expect("a live element")).collect();
        let dims: Vec<usize> = es.iter().map(|e| if let Form::Stick { .. } = e.form { 5 } else { 4 }).collect();
        let offset: Vec<usize> = dims.iter().scan(0, |o, d| Some(std::mem::replace(o, *o + d))).collect();
        let n: usize = dims.iter().sum();
        let em = &self.em;
        let after: Vec<View> = (0..self.views.len()).into_par_iter().map(|s| self.after(s, p)).collect();
        after.iter().map(|a| {
            let (s, v) = (a.s, &*self.views[a.s]);
            let refl = em.refl_noise > 0.0 && !v.echo_refl.is_empty();
            let other = v.epoch != self.ref_epoch;
            let shared = Shared::new(n, &es, em);
            // per cell: its slope's part, and the element's echo's observables with their loadings and weights
            type Rows = Vec<(Vec<f64>, Vec<(usize, f64)>, f64, f64)>;
            let parts: Vec<(Vec<f64>, Rows)> = Self::cells_of(a, ids).into_par_iter().map(|(cell, ls, owners)| {
                let nodes: Vec<(usize, [Geometry; 2])> = owners.iter().map(|&(i, term, [shift, level])| {
                    (i, [Geometry::var(shift, SHIFT), if term.own_sd[1] > 0.0 { Geometry::var(level, LEVEL) } else { Geometry::c(level) }])
                }).collect();
                let mut slope = vec![0.0; n];
                let mut x = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                let (mut echo, mut shares) = (None, Vec::new());
                for (j, ly) in ls.iter().enumerate() {
                    let Some(k) = ids.iter().position(|&i| i == ly.elem) else { continue };
                    let th: Vec<Geometry> = (0..dims[k]).map(|i| Geometry::var(0.0, i)).collect();
                    let r = self.read_cell(s, cell, &ls, j, &es[k], &th, &nodes);
                    for i in 0..dims[k] {
                        slope[offset[k] + i] = r.cost.g[i];
                    }
                    if let Some((q, sd)) = r.echo {
                        for (xo, q) in x.iter_mut().zip(q) {
                            xo[offset[k]..offset[k] + dims[k]].copy_from_slice(&q.g.as_slice()[..dims[k]]);
                        }
                        echo = Some((q, sd, r.weight));
                    }
                    shares = r.shares;
                }
                // the echo's observables, loaded on each owner's nodes at its front share of the cell's weight
                let mut rows = Rows::new();
                if let Some(([p, mu, rho], sd, w)) = echo {
                    let info = cell_information(em, p.v(), sd * sd, ls[0].f);
                    for &(i, term, _) in &owners {
                        let Some(k) = ids.iter().position(|&id| id == ls[i].elem) else { continue };
                        let align = em.align_sd(term.surface, other);
                        let load = |d: &Geometry| shared.load(k, align, d.g[SHIFT], d.g[LEVEL], em.refl_level_sd);
                        let w = w * shares[i];
                        rows.push((x[0].clone(), load(&p), info[0], w));
                        rows.push((x[1].clone(), load(&mu), info[1], w));
                        if refl && rho.v() > -200.0 {
                            rows.push((x[2].clone(), load(&rho), info[2], w));
                        }
                    }
                }
                (slope, rows)
            }).collect();
            let (mut shared, mut slope) = (shared, vec![0.0; n]);
            for (g, rows) in parts {
                slope.iter_mut().zip(g).for_each(|(a, b)| *a += b);
                for (x, load, info, w) in rows {
                    shared.add(&x, &load, info, w);
                }
            }
            let [expected, gradient_variance] = shared.finish();
            Information { slope, expected, gradient_variance }
        }).collect()
    }

    /// The cells of a scan as a proposal leaves it ([`View`]) that any of the elements `ids` fronts and that hold a
    /// pulse, in index order, each with its k-buffer (nearest first, ties by id, as the cost's), the term of its front
    /// element and that term's own nodes as the fit holds them: the shift along the beam (the held alignment's and the
    /// stem front shift's) and the reflectance level.
    pub(super) fn cells_of<'a>(a: &'a View, ids: &[u32]) -> Vec<(usize, Vec<Layer>, Vec<(usize, &'a Term, [f64; 2])>)> {
        let (v, fit) = (&*a.scene.views[a.s], &a.scene.fits[a.s]);
        let mut cells: Vec<usize> = ids.iter().flat_map(|&id| a.entered(id).into_iter().flatten().copied()).collect();
        cells.sort_unstable();
        cells.dedup();
        cells.into_iter().filter(|&c| v.cell[c] != Cell::NoPulse).filter_map(|cell| {
            let ls = a.layers(cell);
            let owners: Vec<(usize, &Term, [f64; 2])> = front_shares(&ls).iter().enumerate().filter(|(_, &w)| w > 0.0).filter_map(|(i, _)| {
                let id = ls[i].elem;
                let term = a.term(&chain_key(id, &a.element(id)?))?;
                Some((i, term, [fit.held.shift(&term.align) + term.own_sd[0] * term.own[0], term.own_sd[1] * term.own[1]]))
            }).collect();
            (!owners.is_empty()).then_some((cell, ls, owners))
        }).collect()
    }

    /// Cell `cell` of scan `s` as [`Scene::information`] reads it ([`CellReading`]), its k-buffer `ls` with the `j`th
    /// layer element `e`'s moved by `th` in its parameters ([`placed`]), each layer with a front share read at its term's
    /// own nodes (`owners`: the layer's index and its shift along the beam and reflectance level), all in the numbers
    /// `P`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_cell<P: Real>(&self, s: usize, cell: usize, ls: &[Layer], j: usize, e: &Element, th: &[P], owners: &[(usize, [P; 2])]) -> CellReading<P> {
        let (v, em) = (&*self.views[s], &self.em);
        let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
        let moved = layer(&placed(e, &v.origin, Some(th)), &w, e, ls[j].elem, em, v.step()).expect("a layer it fronts");
        let layers: Vec<Layer<P>> = ls.iter().enumerate().map(|(i, l)| if i == j { moved } else { l.lift() }).collect();
        let shares = front_shares(&layers);
        let cv = cell_view(v, cell, layers, em, &self.sc, &self.ext);
        let base = cv.empty.read(P::c(0.0), P::c(0.0), em).ll;
        let mut cost = P::c(0.0);
        for &(i, [shift, level]) in owners {
            cost += -((mixture_ll(&cv.pulses, shift, level, em) - base) * cv.wt) * shares[i];
        }
        // the element's echo, at the front owner's nodes: its chance over the mixture, its depth and reflectance with
        // every layer in
        let [shift, level] = owners[0].1;
        let front = |p: &Pulse<P>| p.receiver(shift, level, em).front().map(|f| ((-p.occluders[f.first].tau).exp() * f.s, f));
        let chance = cv.pulses.iter().map(|(w, p)| front(p).map_or(P::c(0.0), |(c, _)| c * *w)).sum::<P>();
        let echo = front(&cv.pulses[0].1).map(|(_, f)| ([chance, f.mu, f.reflectance()], f.sd.v()));
        CellReading { cost, weight: cv.wt.v(), shares: shares.iter().map(|s| s.v()).collect(), echo }
    }

    /// What hypothesis `q` is expected to cost against hypothesis `p` were `p` true, per scan (nats): `E_p[cost(q) −
    /// cost(p)]`, the divergence of `q`'s predictions from `p`'s (each against the scene, [`Scene::predict`]), with
    /// [`Scene::information`]'s reading of a cell ([`cell_information`]): its echo or none, a Bernoulli, exact; given
    /// an echo in both, its depth and reflectance at their own sds, the latents they load marginalised as
    /// `information` does (`p`'s elements' nodes). A cell only one of them fronts reads the other's chance of an echo
    /// as 0 (1e-3). Beside an observed cost difference it separates data that are balanced (observed ≈ 0, expected
    /// large: a misfit) from data that are silent (both ≈ 0).
    pub fn divergence(&self, p: &[Element], q: &[Element]) -> Vec<f64> {
        let em = &self.em;
        (0..self.views.len()).into_par_iter().map(|s| {
            let at = |es: &[Element]| -> BTreeMap<usize, CellPrediction> { self.predict(&Proposal::put(es), s).into_iter().map(|c| (c.cell, c)).collect() };
            let (pp, qq) = (at(p), at(q));
            let refl = em.refl_noise > 0.0 && !self.views[s].echo_refl.is_empty();
            let other = self.views[s].epoch != self.ref_epoch;
            let mut shared = Shared::new(1, p, em);
            let mut bernoulli = 0.0;
            let echo = |c: Option<&CellPrediction>| c.map_or(0.0, |c| c.arrives * c.recorded).clamp(1e-3, 1.0 - 1e-3);
            let cells: BTreeSet<usize> = pp.keys().chain(qq.keys()).copied().collect();
            for cell in cells {
                let (a, b) = (pp.get(&cell), qq.get(&cell));
                let (ea, eb) = (echo(a), echo(b));
                let w = a.or(b).map_or(0.0, |c| c.weight);
                bernoulli += w * (ea * (ea / eb).ln() + (1.0 - ea) * ((1.0 - ea) / (1.0 - eb)).ln());
                let (Some(a), Some(b)) = (a, b) else { continue };
                let k = a.elem as usize;
                let (align, front) = (em.align_sd(a.surface, other), p[k].front_shift_sd(em));
                let [_, depth, refl_info] = cell_information(em, ea, a.sd * a.sd - align * align - front * front, a.fill);
                let load = shared.load(k, align, 1.0, 0.0, em.refl_level_sd);
                shared.add(&[b.depth - a.depth], &load, depth, w);
                if refl && (b.refl - a.refl).is_finite() {
                    let load = shared.load(k, align, 0.0, 1.0, em.refl_level_sd);
                    shared.add(&[b.refl - a.refl], &load, refl_info, w);
                }
            }
            bernoulli + 0.5 * shared.finish()[0][0]
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use crate::transform::{cross, norm, normalize, sub};
    use crate::scene::dual::Real;
    use crate::scene::foliage::Foliage;
    use crate::scene::pulse::Outcome;
    use crate::scene::render::{Form, Class, Element, Layer, placed, render, render_all, layer, KBuffer};
    use crate::scene::settings::{Background, Settings};
    use crate::scene::term::elem_data;
    use super::super::fixtures::{cost, piece_of, pole_view, stick, thin_pole_view};

    /// The counts: the true pole's full-fill cells are on it and its echoes fit its sharp depth (a small
    /// excess over the noise); a stick where the pulses went through to the wall reads behind; the scene is
    /// left as it was.
    #[test]
    fn stats_read_on_for_the_pole_and_behind_for_a_phantom() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        let pole = c.counts(stick(0.0, 4.0, 0.03), &[], &[])[0];
        assert!(pole.on > 10.0 * (pole.behind + pole.empty).max(0.01), "{pole:?}");
        assert!(pole.mismatch_cells > 50 && pole.mismatch_ss / pole.mismatch_w < 1e-5, "{pole:?}");
        let phantom = c.counts(stick(0.2, 6.0, 0.03), &[], &[])[0];
        assert!(phantom.behind > 10.0 * phantom.on.max(0.01), "{phantom:?}");
        assert_eq!(c.cost(), 0.0);
    }

    /// A patch renders as a disc: full fill at its centre, none beyond its rim; a patch of the pole's size where the
    /// pole is beats one off to the side; `predict` codes the pole's front cells on, and a stick beside it behind.
    #[test]
    fn patches_render_as_discs_and_predict_codes_cells() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let patch = |x: f64| Element { form: Form::Patch { centre: [x, 4.0, 0.0] }, class: Class::Leaf, r: 0.03, opacity: 1.0, mismatch: 0.0, group: None, key: None };
        let layers = render(&v, &patch(0.0), 0, &em);
        let f_max = layers.iter().map(|(_, l)| l.f).fold(0.0, f64::max);
        assert!(f_max > 0.99, "{f_max}");
        assert!(layers.iter().all(|(_, l)| l.x < 0.03 + 0.012 || l.f < 0.01));
        let e = |s: Element| cost(&v, &[s], &em, &sc).0;
        assert!(e(patch(0.0)) < e(patch(0.1)));
        let p = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(&Proposal::put(&[stick(0.0, 4.0, 0.03)]), 0);
        let on = p.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::On);
        let beside = Scene::new(vec![v.clone()], &[], em, sc, None, 0).predict(&Proposal::put(&[stick(0.1, 4.0, 0.03)]), 0);
        let behind = beside.iter().filter(|q| q.fill > 0.9).all(|q| q.outcome == Outcome::Behind);
        assert!(on && behind);
        assert!(p.iter().all(|q| q.weight > 0.0 && q.weight <= 1.0 / em.discount));
    }

    /// The class sets the reflectance: the same stick marked leaf is recorded less often than marked wood when leaf
    /// is dimmer (the review after slice 15: patches were rendered as wood).
    #[test]
    fn a_leaf_stick_reflects_as_leaf() {
        let v = pole_view();
        let em = Settings { refl_leaf: -40.0, ..Settings::default() };
        let c = Scene::new(vec![v.clone()], &[], em, Background::default(), None, 0);
        let wood = Element { mismatch: 0.0, ..stick(0.0, 4.0, 0.03) };
        let mean = |p: Vec<CellPrediction>| p.iter().map(|q| q.recorded).sum::<f64>() / p.len() as f64;
        let (leaf, wood) = (mean(c.predict(&Proposal::put(&[Element { class: Class::Leaf, ..wood }]), 0)), mean(c.predict(&Proposal::put(&[wood]), 0)));
        assert!(leaf < wood - 0.1, "leaf {leaf} wood {wood}");
    }

    /// How well a scan pins a pole down: its expected information is positive in every direction of a stick's
    /// geometry; a row's parts sum to its cost, and the rows and the scans' alignment parts to the scene's.
    #[test]
    fn the_pole_is_pinned_down_and_rows_split_their_cost() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let c = Scene::new(vec![v.clone()], &[stick(0.0, 4.0, 0.03)], em, sc, None, 0);
        let x = &c.information(&Proposal::default(), &[0])[0].expected;
        for i in 0..5 {
            assert!(x[i * 5 + i] > 0.0, "{i}: {x:?}");
        }
        let r = c.rows(&Proposal::default())[0];
        let parts = r.ray_on + r.ray_empty + r.ray_in_front + r.ray_behind + r.alignment;
        assert!((parts - r.cost).abs() < 1e-9, "{parts} vs {}", r.cost);
        assert_eq!(r.discount, em.discount);
    }

    /// Element `e` moved by `th` in [`Scene::information`]'s parameters, as an element (the referee for [`placed`]).
    fn moved(e: &Element, th: &[f64]) -> Element {
        let form = match e.form {
            Form::Stick { a, b } => {
                let ax = sub(&b, &a);
                let half = 0.5 * norm(&ax);
                let u = ax.map(|x| x / (2.0 * half));
                let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
                let p2 = cross(&u, &p1);
                let mv = |p: &Point, sign: f64| [0, 1, 2].map(|k| p[k] + (th[1] + sign * half * th[3]) * p1[k] + (th[2] + sign * half * th[4]) * p2[k]);
                Form::Stick { a: mv(&a, -1.0), b: mv(&b, 1.0) }
            }
            Form::Patch { centre } => Form::Patch { centre: [centre[0] + th[1], centre[1] + th[2], centre[2] + th[3]] },
        };
        Element { form, r: e.r * th[0].exp(), ..*e }
    }

    /// The forward slope (research 130 slice-derivatives) matches central differences of the cost of the cells the
    /// element's layer is in, each read with that layer moved (the same reading in `f64`) at its term's nodes: a stick
    /// off the pole, behind a thinner stick (cells it lies behind another's front in) and that stick, and a patch;
    /// the pass's value is the `f64` one bit for bit; and a moved layer is the moved element's rendered. Off the
    /// pole: the test scan's echoes sit at the true pole's layers' ranges, where a layer passing an echo switches
    /// the returns the medium's echo may precede (a jump of a few micronats, measure zero on real data).
    #[test]
    fn forward_slopes_match_finite_differences() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let patch = Element { form: Form::Patch { centre: [0.001, 4.002, 0.0] }, class: Class::Leaf, r: 0.029, opacity: 1.0, mismatch: 0.002, group: None, key: None };
        let (pole, front) = (stick(0.002, 4.003, 0.028), stick(0.02, 3.9, 0.02));
        for (els, id) in [(vec![pole], 0), (vec![pole, front], 0), (vec![pole, front], 1), (vec![patch], 0)] {
            let c = Scene::new(vec![v.clone()], &els, em, sc, None, 0);
            let e = els[id as usize];
            let n = if let Form::Stick { .. } = e.form { 5 } else { 4 };
            let slope = &c.information(&Proposal::default(), &[id])[0].slope;
            let none = Proposal::default();
            let view = c.after(0, &none);
            let cells: Vec<(usize, Vec<Layer>, Vec<(usize, [f64; 2])>)> = Scene::cells_of(&view, &[id]).into_iter()
                .map(|(cell, ls, owners)| (cell, ls, owners.iter().map(|&(i, _, x)| (i, x)).collect())).collect();
            let cost = |th: &[f64]| -> f64 {
                cells.iter().map(|(cell, ls, nodes)| c.read_cell(0, *cell, ls, ls.iter().position(|l| l.elem == id).unwrap(), &e, th, nodes).cost).sum()
            };
            let th0: Vec<Geometry> = (0..n).map(|i| Geometry::var(0.0, i)).collect();
            for (cell, ls, nodes) in &cells {
                let j = ls.iter().position(|l| l.elem == id).unwrap();
                let lifted: Vec<(usize, [Geometry; 2])> = nodes.iter().map(|&(i, x)| (i, x.map(Geometry::c))).collect();
                assert_eq!(c.read_cell(0, *cell, ls, j, &e, &th0, &lifted).cost.v, c.read_cell(0, *cell, ls, j, &e, &vec![0.0; n], nodes).cost);
            }
            assert!(cells.iter().any(|(_, ls, _)| ls[0].elem != id) == (els.len() > 1 && id == 0));
            let th = [0.02, 0.001, -0.002, 0.003, 0.001];
            for (cell, ly) in render(&v, &moved(&e, &th[..n]), id, &em).iter().filter(|(_, l)| l.enters()) {
                let w = v.direction((cell / v.n_cols) as f64, (cell % v.n_cols) as f64);
                let at = layer(&placed(&e, &v.origin, Some(&th[..n])), &w, &e, id, &em, v.step()).unwrap();
                assert!((at.t - ly.t).abs() < 1e-12 && (at.f - ly.f).abs() < 1e-12, "{at:?} vs {ly:?}");
            }
            for i in 0..n {
                let h = [1e-6, 1e-7, 1e-7, 1e-6, 1e-6][i];
                let mut th = vec![0.0; n];
                th[i] = h;
                let up = cost(&th);
                th[i] = -h;
                let fd = (up - cost(&th)) / (2.0 * h);
                assert!((slope[i] - fd).abs() < 1e-4 * slope[i].abs().max(fd.abs()) + 1e-3, "{id} parameter {i}: forward {} central {fd}", slope[i]);
            }
        }
    }

    /// The expected information is the expected cost's curvature: along each of a stick's parameters, the divergence
    /// of the pole from itself moved ± a small step ([`Scene::divergence`], the same reading of a cell), summed, over
    /// the step's square, is E's diagonal. Drawn sharp (no shape mismatch), so the echo's mean reflectance, which E
    /// reads, is the middle radius's, which `predict` reports.
    #[test]
    fn expected_information_is_the_expected_costs_curvature() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let pole = Element { mismatch: 0.0, ..stick(0.0, 4.0, 0.03) };
        let e = Scene::new(vec![v.clone()], &[pole], em, sc, None, 0).information(&Proposal::default(), &[0]).remove(0).expected;
        let empty = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        for (i, h) in [5e-5, 2e-6, 2e-6, 1e-5, 1e-5].into_iter().enumerate() {
            let kl = [h, -h].map(|h| {
                let mut th = [0.0; 5];
                th[i] = h;
                empty.divergence(&[pole], &[moved(&pole, &th)])[0]
            });
            let curv = (kl[0] + kl[1]) / (h * h);
            assert!((curv / e[i * 6] - 1.0).abs() < 0.01, "{i}: {curv} vs {}", e[i * 6]);
        }
    }

    /// Several elements' information (research 130 slice 4). Far apart (no alignment node or cell in common), a pair's
    /// joint expected information is each one's own, side by side. Neighbours sharing their alignment node (its spread
    /// 1 cm here, the stem front shift off) have it counted once: each one's own information charges it in full along
    /// the beam, but it cancels in their depth difference, which the pair pins down far better. Predictions over
    /// several elements are each one's own where they share no cell, and a hypothesis is expected to cost nothing
    /// against itself (against a small move, `expected_information_is_the_expected_costs_curvature`).
    #[test]
    fn a_pairs_information_counts_the_shared_alignment_once() {
        let v = pole_view();
        let sc = Background::default();
        let em = Settings { align_along_solid: 0.01, align_along_mid: 0.01, align_along_fragmented: 0.01, front_shift_base: 0.0, front_shift_slope: 0.0, ..Settings::default() };
        let block = |e: &[f64], n: usize, o: usize, k: usize| (0..k * k).map(|i| e[(o + i / k) * n + o + i % k]).collect::<Vec<f64>>();
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1e-3));
        // far apart: the pole, and a stick 2.5 m behind it and to the side (other nodes, other cells)
        let (pole, far) = (stick(0.0, 4.0, 0.03), stick(0.25, 6.5, 0.03));
        let c = Scene::new(vec![v.clone()], &[pole, far], em, sc, None, 0);
        let joint = c.information(&Proposal::default(), &[0, 1]).remove(0).expected;
        let (a, b) = (c.information(&Proposal::default(), &[0]).remove(0).expected, c.information(&Proposal::default(), &[1]).remove(0).expected);
        assert!(a[0] > 0.0 && b[0] > 0.0);
        assert!(close(&block(&joint, 10, 0, 5), &a) && close(&block(&joint, 10, 5, 5), &b), "{joint:?}\n{a:?}\n{b:?}");
        assert!((0..5).all(|i| (5..10).all(|j| joint[i * 10 + j].abs() <= 1e-9 * joint[i * 11].max(joint[j * 11]))), "{joint:?}");
        // neighbours on one node: the sd of their along-beam places' difference (parameter 1, p1 along the beam)
        let near = Scene::new(vec![v.clone()], &[pole, stick(0.1, 4.0, 0.03)], em, sc, None, 0);
        let var = |e: &[f64], n: usize, d: &[f64]| {
            let m = nalgebra::DMatrix::from_row_slice(n, n, e) + nalgebra::DMatrix::<f64>::identity(n, n) * 1e-2;
            let d = nalgebra::DVector::from_column_slice(d);
            d.dot(&(m.try_inverse().unwrap() * &d))
        };
        let joint = near.information(&Proposal::default(), &[0, 1]).remove(0).expected;
        let mut d = vec![0.0; 10];
        (d[1], d[6]) = (1.0, -1.0);
        let unit = [0.0, 1.0, 0.0, 0.0, 0.0];
        let own: f64 = [0, 1].map(|k| var(&near.information(&Proposal::default(), &[k]).remove(0).expected, 5, &unit)).iter().sum();
        assert!(var(&joint, 10, &d) < 0.1 * own, "joint {} own {own}", var(&joint, 10, &d));
        // predictions over several elements, and the divergence
        let empty = Scene::new(vec![v.clone()], &[], em, sc, None, 0);
        let both = empty.predict(&Proposal::put(&[pole, far]), 0);
        let alone = [empty.predict(&Proposal::put(&[pole]), 0), empty.predict(&Proposal::put(&[far]), 0)];
        assert_eq!(both.len(), alone[0].len() + alone[1].len());
        assert!(both.iter().all(|p| alone[p.elem as usize].iter().any(|q| q.cell == p.cell && q.depth == p.depth)));
        assert_eq!(empty.divergence(&[pole], &[pole])[0], 0.0);
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
            let i = c.information(&Proposal::default(), &[0]).remove(0);
            (i.expected[12], i.gradient_variance[12], c.rows(&Proposal::default())[0].arc)
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
        for (r, y) in [(0.01, 6.0), (0.006, 6.0)] {
            let v = thin_pole_view(r, y);
            let info = |em: Settings| Scene::new(vec![v.clone()], &[stick(0.0, y, r)], em, sc, None, 0).information(&Proposal::default(), &[0])[0].expected[0];
            let (with, without) = (info(Settings::default()), info(Settings { refl_noise: 0.0, ..Settings::default() }));
            assert!(with > 1.5 * without, "r {r}: with reflectance {with}, without {without}");
        }
    }

    /// Slice 2d: a group's information loads one stem front shift with the prior its cost gives it (its members' mean
    /// sd: [`term_nodes`]), not each member's own.
    #[test]
    fn a_groups_information_and_cost_share_its_front_shift_prior() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        let es = [Element { r: 0.03, ..piece_of(-0.2, 0.0, Some(3)) }, Element { r: 0.015, ..piece_of(0.0, 0.2, Some(3)) }];
        let rendered = render_all(&v, &es, &em);
        let d = elem_data(&v, &KBuffer::of(&rendered).cells(), &es, &[0, 0], &[0, 1], 0, &em, &sc, &Foliage::uniform(sc.kappa), None);
        assert_eq!(d.len(), 1);
        let shared = Shared::new(1, &es, &em);
        for (k, e) in es.iter().enumerate() {
            assert!((e.front_shift_sd(&em) - d[0].own_sd[0]).abs() > 1e-4);
            assert_eq!(shared.load(k, 0.0, 1.0, 0.0, 0.0).last().unwrap().1, d[0].own_sd[0]);
        }
    }
}
