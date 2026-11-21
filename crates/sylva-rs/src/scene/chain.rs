// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Chains grown by the likelihood (3dtrees research 130 slice-grower), and the declared prior over a plant's wood
//! (research 130 plant-prior, slice-s8): the axis prior and the fork as an insertion, a chain's state as it grows
//! beside a scene, and the grower.
//!
//! - **The axis prior** ([`axis_row`]; research 130 lens-tree T1–T2): a declared prior over one axis's geometry alone,
//!   never in a cost. Its clock is length in radii, dτ = ds / r (more generally ds / r^β, `Settings::axis_clock`), so
//!   the taper, the form departures and the bend are each one constant per unit τ from twig to stem. ln r drifts by
//!   `axis_taper` per τ (at β = 1 a linear taper, dr/ds constant) with Ornstein–Uhlenbeck departures about a level the
//!   row profiles out; the direction walks at `axis_bend` per √τ. The row samples the sticks' radius and direction
//!   every `axis_step` of τ about the axis's middle and charges only departures (½ z², no normalisers), so the same
//!   wood cut into more sticks gives the same row (the invariant `the_axis_row_is_the_same_however_the_wood_is_cut`).
//!   Which end is the root isn't known for a chain alone: the radius departure is the cheaper polarity's, with the two
//!   polarities' difference beside it.
//! - **The fork** ([`insertion_row`]; lens-tree T3): a lateral inserted on its parent axis, scored by conservation,
//!   r_before^γ = r_after^γ + r_lat^γ (`fork_gamma`, spread `fork_sd` in ln r), the radii carried by the taper to the
//!   insertion point, and by the lateral's angle to its parent. A crossing is two axes and no insertion.
//! - **A chain's state** ([`Chain`]), per scan: its group's term's evidence in information form (each cell's
//!   −log-likelihood a quadratic in (δ, reflectance level) about the point of its last read, and their sum), its own
//!   nodes' mode and precision and its alignment shift's slope and curvature (its [`Term`]), the context terms it
//!   changes, its layers by cell, and its surface class's votes. A piece added reads only the cells it enters: those
//!   exactly, through the own nodes' Newton warm from the chain's mode, the rest from the sum less the quadratics of
//!   the cells it enters. So growing a chain takes time linear in its length, and scoring an extension costs the
//!   piece's cells. The quadratics are the Laplace step the term already takes, each about its own last read; a grown
//!   chain's cost is read again exactly (`score`) when it is handed over.
//! - **The grower** ([`Scene::grow`]): from a seed stick, a beam of a few chains grows one piece at a time, the ends
//!   taking turns. A piece leaves the end along its axis, 4 r long within [`Grow`]'s `step`; its options are the axis
//!   prior's spread over the piece's τ: straight at the radius or e^(±2 sd) of it, or turned one or two bend sds
//!   either way in two planes. Each option's cost to add is conditioned on its chain (one group, the scene
//!   unchanged). A chain extends by the options whose cost to add is negative, inside the region and within the
//!   longest length; an end with none stops. The beam keeps the chains least in cost plus the axis row (a proposer
//!   ranks as it likes), and the least of them is handed over with its cost and its row apart. A seed that doesn't
//!   beat nothing isn't grown.

use crate::Point;
use crate::transform::{add, cross, dot, norm, normalize, scale, sub};
use nalgebra::{Matrix2, Vector2};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use super::dual::{Dual2, Real};
use super::foliage::{self, Deposit, FixedHits};
use super::pulse::mixture_taken;
use super::render::{Element, Form, Layer, KBuffer, keep, numbered, render};
use super::scene::{Proposal, Scene, redeposit};
use super::settings::Settings;
use super::term::{CellData, Owner, Term, eliminate, elem_data, median_class, owner, settle, slots, term_nodes, vote, clipped, mean_of};

/// The clock's reference radius (m): dτ = ds / r · (r / R_CLOCK)^(1 − β), so β = 1 is length in radii and β = 0 is
/// length in units of R_CLOCK, whatever the radius (`Settings::axis_clock`).
pub const R_CLOCK: f64 = 0.01;

/// τ per metre along wood of radius r (the clock, `Settings::axis_clock`).
pub fn tau_rate(em: &Settings, r: f64) -> f64 {
    (r / R_CLOCK).powf(1.0 - em.axis_clock) / r
}

/// An axis's prior row: its length (m), its length in τ, its widest end-to-start gap (m, reported, not charged), and
/// the departures (nats) of its radii and its direction from the declared process. `radius` is the departure in the
/// cheaper of the two polarities (which end is the root); `polarity` is the other's less this one's, positive when the
/// root is at the first stick's start. The marginal over a fair polarity bit is
/// `radius + ln 2 − ln(1 + e^(−|polarity|))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisRow {
    pub length: f64,
    pub tau: f64,
    pub gap: f64,
    pub radius: f64,
    pub bend: f64,
    pub polarity: f64,
}

impl AxisRow {
    /// The row's departures together (nats).
    pub fn total(&self) -> f64 {
        self.radius + self.bend
    }
}

/// The prior row of sticks (a, b, r) joined end to end in order along one axis (the module's docs, "the axis prior").
pub fn axis_row(em: &Settings, sticks: &[(Point, Point, f64)]) -> AxisRow {
    let mut out = AxisRow { length: 0.0, tau: 0.0, gap: 0.0, radius: 0.0, bend: 0.0, polarity: 0.0 };
    if sticks.is_empty() {
        return out;
    }
    // each stick's end in τ
    let mut ends = Vec::with_capacity(sticks.len());
    for (j, &(a, b, r)) in sticks.iter().enumerate() {
        let len = norm(&sub(&b, &a));
        out.length += len;
        out.tau += len * tau_rate(em, r);
        ends.push(out.tau);
        if j > 0 {
            out.gap = out.gap.max(norm(&sub(&a, &sticks[j - 1].1)));
        }
    }
    // the samples: every `axis_step` of τ about the axis's middle, so a cut that changes nothing adds nothing
    let step = em.axis_step;
    let k = (0.5 * out.tau / step).floor() as i64;
    let samples: Vec<(f64, f64, Point)> = (-k..=k).map(|i| {
        let t = 0.5 * out.tau + i as f64 * step;
        let (a, b, r) = sticks[ends.partition_point(|&e| e <= t).min(sticks.len() - 1)];
        (t, r.ln(), normalize(&sub(&b, &a)))
    }).collect();
    // bends: the direction walks with variance axis_bend² per unit τ in each of two directions
    let q = em.axis_bend.powi(2) * step;
    out.bend = samples.windows(2).map(|w| {
        let turn = dot(&w[0].2, &w[1].2).clamp(-1.0, 1.0).acos();
        0.5 * turn * turn / q
    }).sum();
    // radii: ln r less the taper, an OU about a level the row profiles out, in each polarity
    let rho = (-step / em.axis_form_range).exp();
    let (sd0, sd1) = (em.axis_form_sd, em.axis_form_sd * (1.0 - rho * rho).sqrt());
    let ou = |y: &[f64]| -> f64 {
        // residuals (c − a L) / s, the level L by least squares
        let mut terms = vec![(y[0], 1.0, sd0)];
        terms.extend(y.windows(2).map(|w| (w[1] - rho * w[0], 1.0 - rho, sd1)));
        let (num, den) = terms.iter().fold((0.0, 0.0), |(n, d), &(c, a, s)| (n + a * c / (s * s), d + a * a / (s * s)));
        let l = num / den;
        terms.iter().map(|&(c, a, s)| 0.5 * ((c - a * l) / s).powi(2)).sum()
    };
    let forward: Vec<f64> = samples.iter().map(|&(t, lr, _)| lr - em.axis_taper * t).collect();
    let backward: Vec<f64> = samples.iter().map(|&(t, lr, _)| lr - em.axis_taper * (out.tau - t)).collect();
    let (f, b) = (ou(&forward), ou(&backward));
    out.radius = f.min(b);
    out.polarity = b - f;
    out
}

/// A fork as an insertion (the module's docs, "the fork"): the departures (nats) of conservation and of the lateral's
/// angle, and conservation's departure itself in ln r, `(1/γ) ln(r_after^γ + r_lat^γ) − ln r_before`, each radius
/// carried by the taper from its stick's middle to the insertion point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsertionRow {
    pub conservation: f64,
    pub angle: f64,
    pub departure: f64,
}

impl InsertionRow {
    /// The row's departures together (nats).
    pub fn total(&self) -> f64 {
        self.conservation + self.angle
    }
}

/// The insertion row of a lateral stick on its parent: the parent's stick before the insertion (the root on its
/// start's side, its end at the insertion point), the parent's stick after it, and the lateral, each (a, b, r) with
/// the after and lateral sticks starting at the insertion.
pub fn insertion_row(em: &Settings, before: (Point, Point, f64), after: (Point, Point, f64), lateral: (Point, Point, f64)) -> InsertionRow {
    // a stick's ln r carried from its middle by half its length in τ, toward the tips (+) or toward the root (−)
    let carried = |(a, b, r): (Point, Point, f64), out: f64| r.ln() + out * em.axis_taper * 0.5 * norm(&sub(&b, &a)) * tau_rate(em, r);
    let (pb, pa, pl) = (carried(before, 1.0), carried(after, -1.0), carried(lateral, -1.0));
    let g = em.fork_gamma;
    let departure = ((g * pa).exp() + (g * pl).exp()).ln() / g - pb;
    let theta = dot(&normalize(&sub(&before.1, &before.0)), &normalize(&sub(&lateral.1, &lateral.0))).clamp(-1.0, 1.0).acos();
    InsertionRow {
        conservation: 0.5 * (departure / em.fork_sd).powi(2),
        angle: 0.5 * ((theta - em.fork_angle_mean) / em.fork_angle_sd).powi(2),
        departure,
    }
}

/// The grower's settings: the chains a seed's beam keeps, the longest chain (m), and a piece's length bounds (m; 4 r
/// within them). It proposes under the scene's settings' axis prior.
#[derive(Debug, Clone, Copy)]
pub struct Grow {
    pub beam: usize,
    pub max_length: f64,
    pub step: (f64, f64),
}

impl Default for Grow {
    fn default() -> Self {
        Grow { beam: 3, max_length: 1.5, step: (0.03, 0.10) }
    }
}

/// A grown chain as handed over: its sticks in order along it, its cost to add per scan read exactly (then, with a
/// foliage field, the field's part) and in total, the state's own cost (`kalman`, from the quadratics), the seed's
/// cost, the pieces it added, its axis prior row (apart from the cost: the beam ranked by their sum) and its length
/// (m).
#[derive(Debug, Clone)]
pub struct Grown {
    pub sticks: Vec<Element>,
    pub scan: Vec<f64>,
    pub cost: f64,
    pub kalman: f64,
    pub seed_cost: f64,
    pub steps: usize,
    pub prior: AxisRow,
    pub length: f64,
}

/// −log-likelihood ≈ ½ xᵀ a x + bᵀ x + k over x = (δ, reflectance level): one cell's, about its last read, or a sum.
#[derive(Debug, Clone, Copy)]
struct Quadratic {
    a: Matrix2<f64>,
    b: Vector2<f64>,
    k: f64,
}

impl Quadratic {
    const ZERO: Quadratic = Quadratic { a: Matrix2::new(0.0, 0.0, 0.0, 0.0), b: Vector2::new(0.0, 0.0), k: 0.0 };

    /// From a log-likelihood's value, slope and curvature `l` at `x`.
    fn of(l: &Dual2<2>, x: Vector2<f64>) -> Quadratic {
        let (f, g, h) = (-l.v, -l.g, -l.h);
        Quadratic { a: h, b: g - h * x, k: f - g.dot(&x) + 0.5 * x.dot(&(h * x)) }
    }

    fn add(&mut self, q: &Quadratic, sign: f64) {
        (self.a, self.b, self.k) = (self.a + q.a * sign, self.b + q.b * sign, self.k + q.k * sign);
    }

    /// Its log-likelihood at x, with slope and curvature.
    fn ll(&self, x: Vector2<f64>) -> Dual2<2> {
        let ax = self.a * x;
        Dual2 { v: -(0.5 * x.dot(&ax) + self.b.dot(&x) + self.k), g: -(ax + self.b), h: -self.a }
    }
}

/// One cell's last read: its quadratic, its vote for the term's class, and what the term took of the field's hits
/// there ([`Term`]'s `deposit`).
#[derive(Debug, Clone)]
struct Read {
    q: Quadratic,
    vote: Option<u8>,
    deposit: Deposit,
}

/// A chain's state in one scan (the module's docs).
#[derive(Clone, Default)]
struct ChainScan {
    /// Its pieces' layers by cell.
    buffer: KBuffer,
    /// The pieces the scan sees: its term's members.
    members: Vec<usize>,
    /// Per cell its term has a front share in, its last read; their sum; the class votes.
    cells: HashMap<usize, Read>,
    sum: Option<Quadratic>,
    votes: [u32; 3],
    term: Option<Term>,
    /// The scene's terms it changes.
    ctx: HashMap<Owner, Option<Term>>,
}

/// A chain grown beside a scene, the scene unchanged: its pieces (numbered from the scene's next id), their order
/// along it, its state per scan, the foliage field's counts' change were it added, and its cost to add.
#[derive(Clone)]
pub struct Chain {
    pieces: Vec<Element>,
    order: VecDeque<usize>,
    scans: Vec<ChainScan>,
    delta: FixedHits,
    cost: f64,
    prior: f64,
    length: f64,
    steps: usize,
    done: [bool; 2],
    turn: usize,
}

/// A piece tried on a chain in one scan: its layers, the chain's cells it re-reads (`gone`) and those it reads now
/// with their votes and deposits, the term, the scene's terms it changes (beyond the chain's), and the scan's cost.
struct ScanTrial {
    layers: Vec<(usize, Layer)>,
    gone: Vec<usize>,
    fresh: Vec<(CellData, Option<u8>, Deposit)>,
    term: Option<Term>,
    ctx: Vec<(Owner, Option<Term>)>,
    cost: f64,
}

/// A piece tried on a chain: per scan, the counts' change, the cost.
struct Trial {
    piece: Element,
    scans: Vec<ScanTrial>,
    delta: FixedHits,
    cost: f64,
}

impl Chain {
    fn empty(n: usize) -> Chain {
        Chain { pieces: Vec::new(), order: VecDeque::new(), scans: vec![ChainScan::default(); n], delta: FixedHits::default(), cost: 0.0, prior: 0.0,
                length: 0.0, steps: 0, done: [false; 2], turn: 1 }
    }

    /// Its sticks in order along it.
    fn sticks(&self) -> Vec<Element> {
        self.order.iter().map(|&k| self.pieces[k]).collect()
    }

    fn score(&self) -> f64 {
        self.cost + self.prior
    }

    /// The end that grows next: its turn's, unless that one has stopped.
    fn end(&self) -> usize {
        if self.done[self.turn] { 1 - self.turn } else { self.turn }
    }
}

impl Scene {
    /// Grow each seed stick into a chain by the likelihood (the module's docs), inside the region `inside` (a test of
    /// points), in parallel over seeds, the scene unchanged.
    pub fn grow(&self, seeds: &[Element], inside: &(dyn Fn(&[Point]) -> Vec<bool> + Sync), opts: &Grow) -> Vec<Grown> {
        let g = self.groups.keys().max().map_or(0, |m| m + 1);
        seeds.par_iter().map(|e| self.grow_one(Element { group: Some(g), ..*e }, inside, opts)).collect()
    }

    fn grow_one(&self, seed: Element, inside: &(dyn Fn(&[Point]) -> Vec<bool> + Sync), opts: &Grow) -> Grown {
        let empty = Chain::empty(self.views.len());
        let first = self.take(&empty, self.try_piece(&empty, seed), 1, 0.0);
        let seed_cost = first.cost;
        let mut beam = vec![Chain { done: [seed_cost >= 0.0; 2], ..first }];
        for _ in 0..1000 {
            let live: Vec<usize> = (0..beam.len()).filter(|&i| !beam[i].done.iter().all(|&d| d)).collect();
            if live.is_empty() {
                break;
            }
            // each live chain's options at its growing end, with the chain's prior row were it taken, inside the region
            // and within the longest length
            let props: Vec<(usize, usize, Element, f64)> = live.iter().flat_map(|&i| {
                let end = beam[i].end();
                self.options(&beam[i], end, opts).into_iter().map(move |(e, z)| (i, end, e, z))
            }).collect();
            let tips: Vec<Point> = props.iter().map(|(_, end, e, _)| match e.form {
                Form::Stick { a, b } => if *end == 1 { b } else { a },
                Form::Patch { centre } => centre,
            }).collect();
            let ok = inside(&tips);
            let mut trials: Vec<Option<Trial>> = props.par_iter().zip(&ok).map(|((i, _, e, _), &ok)| {
                let m = &beam[*i];
                (ok && m.length + e.length() <= opts.max_length).then(|| self.try_piece(m, *e))
            }).collect();
            // the pool: each live chain's extensions whose cost to add is negative, or the chain with that end stopped
            let mut pool: Vec<(f64, usize, Option<usize>)> = Vec::new();
            for (i, m) in beam.iter().enumerate() {
                let ext: Vec<usize> = (0..props.len()).filter(|&k| props[k].0 == i && trials[k].as_ref().is_some_and(|t| t.cost < m.cost)).collect();
                if ext.is_empty() {
                    pool.push((m.score(), i, None));
                }
                pool.extend(ext.into_iter().map(|k| (trials[k].as_ref().unwrap().cost + props[k].3, i, Some(k))));
            }
            pool.sort_by(|a, b| a.0.total_cmp(&b.0));
            pool.truncate(opts.beam.max(1));
            beam = pool.into_iter().map(|(_, i, k)| match k {
                Some(k) => self.take(&beam[i], trials[k].take().unwrap(), props[k].1, props[k].3),
                None => {
                    let mut m = beam[i].clone();
                    let end = m.end();
                    m.done[end] = true;
                    m
                }
            }).collect();
        }
        let best = beam.into_iter().min_by(|a, b| a.score().total_cmp(&b.score())).unwrap();
        let sticks = best.sticks();
        let scan = self.score(&[Proposal::put(&sticks)]).remove(0);
        let prior = axis_row(&self.em, &triples(&sticks));
        Grown { cost: scan.iter().sum(), scan, sticks, kalman: best.cost, seed_cost, steps: best.steps, prior, length: best.length }
    }

    /// The next piece's options from chain `m`'s end `end` (0 its first stick's start, 1 its last's end), each with
    /// the chain's axis prior row (its total) were the piece taken (the module's docs).
    fn options(&self, m: &Chain, end: usize, opts: &Grow) -> Vec<(Element, f64)> {
        let tip = m.pieces[if end == 1 { *m.order.back().unwrap() } else { *m.order.front().unwrap() }];
        let Form::Stick { a, b } = tip.form else { return Vec::new() };
        let (p, u) = if end == 1 { (b, normalize(&sub(&b, &a))) } else { (a, normalize(&sub(&a, &b))) };
        let ln = (4.0 * tip.r).clamp(opts.step.0, opts.step.1);
        // the axis prior's spread over the piece's length in τ: the form departures' and the bend's
        let dt = ln * tau_rate(&self.em, tip.r);
        let rho = (-dt / self.em.axis_form_range).exp();
        let (sd, th) = (self.em.axis_form_sd * (1.0 - rho * rho).sqrt(), self.em.axis_bend * dt.sqrt());
        let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
        let p2 = cross(&u, &p1);
        let mut dirs: Vec<(Point, f64)> = [0.0, -2.0, 2.0].iter().map(|&k| (u, k)).collect();
        for j in [1.0, 2.0] {
            for q in [p1, p2] {
                for s in [1.0, -1.0] {
                    dirs.push((add(&scale(&u, (j * th).cos()), &scale(&q, s * (j * th).sin())), 0.0));
                }
            }
        }
        let sticks = triples(&m.sticks());
        dirs.into_iter().map(|(d, k)| {
            let q = add(&p, &scale(&d, ln));
            let r = tip.r * (k * sd).exp();
            let (form, piece) = if end == 1 { (Form::Stick { a: p, b: q }, (p, q, r)) } else { (Form::Stick { a: q, b: p }, (q, p, r)) };
            let with: Vec<_> = if end == 1 { sticks.iter().copied().chain([piece]).collect() } else { [piece].into_iter().chain(sticks.iter().copied()).collect() };
            (Element { form, r, mismatch: self.em.mismatch_sd(r), ..tip }, axis_row(&self.em, &with).total())
        }).collect()
    }

    /// Piece `e` tried on chain `c` (the module's docs): per scan its cost, the chain's cost with it.
    fn try_piece(&self, c: &Chain, e: Element) -> Trial {
        let id = self.elems.len() as u32 + c.pieces.len() as u32;
        let scans: Vec<ScanTrial> = (0..self.views.len()).map(|s| self.try_scan(s, c, &c.scans[s], e, id)).collect();
        let mut delta = FixedHits::default();
        let mut cost = scans.iter().map(|t| t.cost).sum::<f64>();
        if let Some(f) = &self.field {
            delta = c.delta.clone();
            for (s, t) in scans.iter().enumerate() {
                let was = &c.scans[s];
                t.gone.iter().for_each(|k| foliage::give(&mut delta, &was.cells[k].deposit, 1));
                t.fresh.iter().for_each(|x| foliage::give(&mut delta, &x.2, -1));
                for (o, new) in &t.ctx {
                    let old = was.ctx.get(o).map_or(self.fits[s].terms.get(o), Option::as_ref);
                    redeposit(&mut delta, old, new.as_ref());
                }
            }
            cost += f.change(&delta);
        }
        Trial { piece: e, scans, delta, cost }
    }

    /// Piece `e` (as `id`) tried on chain `c`'s state `cs` in scan `s`.
    fn try_scan(&self, s: usize, c: &Chain, cs: &ChainScan, e: Element, id: u32) -> ScanTrial {
        let (v, em, fit) = (&*self.views[s], &self.em, &self.fits[s]);
        let first = self.elems.len() as u32;
        let g = Owner::Group(e.group.expect("a chain's piece has its group"));
        let layers = render(v, &e, id, em);
        let mut own = KBuffer::default();
        let touched = own.enter(id, &layers);
        let piece = |i: u32| if i == id { e } else { c.pieces[(i - first) as usize] };
        // the k-buffers of the cells it enters, the scene's layers, the chain's and its own, numbered from 0
        let in_cell = |cell: usize| keep(self.buffers[s].get(cell).iter().chain(cs.buffer.get(cell)).chain(own.get(cell)).copied().collect());
        let mut ids = Vec::new();
        let cells = numbered(touched.iter().map(|&cell| (cell, in_cell(cell))).collect(), &mut ids);
        let elems: Vec<Element> = ids.iter().map(|&i| if i >= first { piece(i) } else { self.elems[i as usize].unwrap() }).collect();
        let owners: Vec<Owner> = ids.iter().zip(&elems).map(|(&i, x)| if i >= first { g } else { owner(i, x) }).collect();
        let mine: Vec<usize> = (0..ids.len()).filter(|&k| ids[k] >= first).collect();
        let fresh_cells = elem_data(v, &cells, &elems, &slots(&owners), &mine, self.ref_epoch, em, &self.sc, &self.ext, self.field.as_ref().map(|f| &f.spec))
            .into_iter().next().map_or(Vec::new(), |d| d.cells);
        let votes: HashMap<usize, u8> = cells.iter()
            .filter_map(|(cell, ls)| vote(v, *cell, ls, em).filter(|x| ids[x.0 as usize] >= first).map(|x| (*cell, x.1))).collect();
        let gone: Vec<usize> = touched.iter().copied().filter(|k| cs.cells.contains_key(k)).collect();
        // the term: the chain's sum less the cells re-read, those read exactly, its own nodes warm from its mode
        let mut sum = cs.sum.unwrap_or(Quadratic::ZERO);
        let mut tally = cs.votes;
        for k in &gone {
            sum.add(&cs.cells[k].q, -1.0);
            if let Some(x) = cs.cells[k].vote {
                tally[x as usize] -= 1;
            }
        }
        for f in &fresh_cells {
            if let Some(&x) = votes.get(&f.cell) {
                tally[x as usize] += 1;
            }
        }
        let mut members: Vec<Element> = cs.members.iter().map(|&k| c.pieces[k]).collect();
        if !layers.is_empty() {
            members.push(e);
        }
        let ctx = if touched.iter().any(|&k| !self.buffers[s].get(k).is_empty()) {
            // the chain's own term is skipped, so its pieces' cells aren't read
            let with: Vec<(Element, Option<&[usize]>)> = c.pieces.iter().chain([&e]).map(|p| (*p, None)).collect();
            let fresh = |cell: usize| -> Vec<Layer> { cs.buffer.get(cell).iter().chain(own.get(cell)).copied().collect() };
            self.terms_after(s, &[], &with, first, &fresh, &touched.iter().copied().collect::<HashSet<usize>>(), &[], Some(g))
        } else {
            Vec::new()
        };
        let term = (!members.is_empty()).then(|| {
            let es: Vec<&Element> = members.iter().collect();
            let class = median_class(&tally);
            let (hs, front) = term_nodes(&es, em);
            let align: Vec<([i64; 3], f64)> = hs.into_iter().map(|(k, w)| (k, w * em.align_sd(class, v.epoch != self.ref_epoch))).collect();
            let level = if em.refl_noise > 0.0 && !v.echo_refl.is_empty() { em.refl_level_sd } else { 0.0 };
            let own_sd = Vector2::new(front, level);
            let a = fit.held.shift(&align);
            let start = cs.term.as_ref().map_or(Vector2::zeros(), |t| t.own);
            let (y, at) = settle(own_sd, start, |y| {
                let x = Vector2::new(a + own_sd[0] * y[0], own_sd[1] * y[1]);
                let lv = if own_sd[1] > 0.0 { Dual2::var(x[1], 1) } else { Dual2::c(x[1]) };
                let total = fresh_cells.iter().fold(sum.ll(x), |t, f| t + f.read(Dual2::var(x[0], 0), lv, em));
                super::term::Local { ll: total.v, parts: [0.0; 4], g: -total.g, hess: clipped(-total.h), deposit: Vec::new() }
            });
            let el = eliminate(own_sd, &at, y);
            Term {
                align, fit: -at.ll + 0.5 * (y[0] * y[0] + y[1] * y[1]), logdet: 0.5 * el.det.ln(), slope: el.slope, curv: el.curv, own: y, start, own_sd,
                own_cov: el.pi, own_link: el.pl, ray: [0.0; 4], cells: (cs.cells.len() - gone.len() + fresh_cells.len()) as u32, cells_on: 0, arc: 0.0,
                across: mean_of(&es, |x| 2.0 * x.r / em.footprint(norm(&sub(&x.mid(), &v.origin)))), surface: class, deposit: Vec::new(),
            }
        });
        // the deposits of the cells it reads, at the term's mode
        // re-read, and those read now
        let x = term.as_ref().map_or(Vector2::zeros(), |t| Vector2::new(fit.held.shift(&t.align) + t.own_sd[0] * t.own[0], t.own_sd[1] * t.own[1]));
        let fresh: Vec<(CellData, Option<u8>, Deposit)> = fresh_cells.into_iter().map(|f| {
            let dep = if f.hits.is_empty() { Vec::new() } else { foliage::deposit(f.take(&mixture_taken(&f.pulses, x[0], x[1], em).1).into_iter()) };
            let vote = votes.get(&f.cell).copied();
            (f, vote, dep)
        }).collect();
        let mut over: HashMap<Owner, Option<Term>> = cs.ctx.clone();
        over.extend(ctx.iter().cloned());
        over.insert(g, term.clone());
        let cost = fit.total(&over) - fit.cost;
        ScanTrial { layers, gone, fresh, term, ctx, cost }
    }

    /// Chain `c` with trial `t` taken at its end `end`, its axis prior row's total then `prior`.
    fn take(&self, c: &Chain, t: Trial, end: usize, prior: f64) -> Chain {
        let first = self.elems.len() as u32;
        let id = first + c.pieces.len() as u32;
        let mut n = c.clone();
        let k = n.pieces.len();
        n.pieces.push(t.piece);
        if end == 1 { n.order.push_back(k) } else { n.order.push_front(k) }
        for (s, (cs, st)) in n.scans.iter_mut().zip(t.scans).enumerate() {
            cs.buffer.enter(id, &st.layers);
            if !st.layers.is_empty() {
                cs.members.push(k);
            }
            let mut sum = cs.sum.unwrap_or(Quadratic::ZERO);
            for cell in st.gone {
                let r = cs.cells.remove(&cell).unwrap();
                sum.add(&r.q, -1.0);
                if let Some(x) = r.vote {
                    cs.votes[x as usize] -= 1;
                }
            }
            // the cells read now, each as a quadratic about the term's mode
            if let Some(term) = &st.term {
                let x = Vector2::new(self.fits[s].held.shift(&term.align) + term.own_sd[0] * term.own[0], term.own_sd[1] * term.own[1]);
                let lv = if term.own_sd[1] > 0.0 { Dual2::var(x[1], 1) } else { Dual2::c(x[1]) };
                for (f, vote, deposit) in st.fresh {
                    let q = Quadratic::of(&f.read(Dual2::var(x[0], 0), lv, &self.em), x);
                    sum.add(&q, 1.0);
                    if let Some(v) = vote {
                        cs.votes[v as usize] += 1;
                    }
                    cs.cells.insert(f.cell, Read { q, vote, deposit });
                }
            }
            cs.sum = Some(sum);
            cs.term = st.term;
            cs.ctx.extend(st.ctx);
        }
        n.delta = t.delta;
        n.cost = t.cost;
        n.prior = prior;
        n.length += t.piece.length();
        n.steps += !c.pieces.is_empty() as usize;
        n.turn = 1 - end;
        n
    }
}

/// Sticks as (a, b, r), patches left out.
pub fn triples(es: &[Element]) -> Vec<(Point, Point, f64)> {
    es.iter().filter_map(|e| match e.form {
        Form::Stick { a, b } => Some((a, b, e.r)),
        Form::Patch { .. } => None,
    }).collect()
}

impl Element {
    /// A stick's length (0 for a patch).
    pub(super) fn length(&self) -> f64 {
        match self.form {
            Form::Stick { a, b } => norm(&sub(&b, &a)),
            Form::Patch { .. } => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::render::render_all;
    use crate::scene::settings::{Background, Settings};
    use super::super::fixtures::{field_box, piece_of, pole_view};

    /// The chain's state grown piece by piece is its full rebuild: after each piece its cells are those the group's
    /// term reads when the chain is scored whole, and its cost to add is `score`'s for the whole chain within
    /// the Laplace step's error (each cell a quadratic about its last read), with a foliage field and without.
    #[test]
    fn the_chains_state_is_its_full_rebuild() {
        let v = pole_view();
        let (em, sc) = (Settings::default(), Background::default());
        for spec in [None, Some(field_box())] {
            let scene = Scene::new(vec![v.clone()], &[], em, sc, spec, 0);
            let mut c = Chain::empty(1);
            for k in 0..10 {
                let z = -0.15 + 0.03 * k as f64;
                c = scene.take(&c, scene.try_piece(&c, piece_of(z, z + 0.03, Some(7))), 1, 0.0);
                let sticks = c.sticks();
                let whole = scene.score(&[Proposal::put(&sticks)]).remove(0).iter().sum::<f64>();
                assert!((c.cost - whole).abs() < 1e-3 * whole.abs().max(1.0), "{k} pieces: {} against {whole}", c.cost);
                let rendered = render_all(&v, &sticks, &em);
                let d = &elem_data(&v, &KBuffer::of(&rendered).cells(), &sticks, &vec![0; sticks.len()], &(0..sticks.len()).collect::<Vec<_>>(), 0, &em, &sc, &scene.ext,
                                   spec.as_ref())[0];
                let mut cells: Vec<usize> = c.scans[0].cells.keys().copied().collect();
                cells.sort_unstable();
                assert_eq!(cells, d.cells.iter().map(|x| x.cell).collect::<Vec<_>>());
            }
        }
    }

    /// The axis row is the same however the wood is cut (research 130 lens-tree check C): a kinked, tapering axis
    /// whose sticks are each cut into 2 and 4 collinear pieces of the same radius gives the same row to rounding.
    #[test]
    fn the_axis_row_is_the_same_however_the_wood_is_cut() {
        let em = Settings::default();
        let (mut at, mut dir) = ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let mut axis = Vec::new();
        for k in 0..7 {
            let (len, r) = (0.031 + 0.017 * ((k * 5 % 7) as f64), 0.006 * (1.0 - 0.09 * k as f64) * (1.0 + 0.05 * ((k % 3) as f64 - 1.0)));
            let b = add(&at, &scale(&dir, len));
            axis.push((at, b, r));
            at = b;
            let turn = 0.06 * (k as f64 - 3.0);
            dir = normalize(&add(&scale(&dir, turn.cos()), &[turn.sin(), 0.3 * turn.sin(), 0.0]));
        }
        let cut = |n: usize| -> Vec<(Point, Point, f64)> {
            axis.iter().flat_map(|&(a, b, r)| (0..n).map(move |i| {
                let (f0, f1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
                (add(&a, &scale(&sub(&b, &a), f0)), add(&a, &scale(&sub(&b, &a), f1)), r)
            })).collect()
        };
        let one = axis_row(&em, &cut(1));
        assert!(one.bend > 1.0 && one.radius > 0.1, "{one:?}");
        for n in [2, 4] {
            let x = axis_row(&em, &cut(n));
            for (p, q) in [(x.length, one.length), (x.tau, one.tau), (x.radius, one.radius), (x.bend, one.bend), (x.polarity, one.polarity)] {
                assert!((p - q).abs() < 1e-9 * q.abs().max(1.0), "×{n}: {x:?} against {one:?}");
            }
        }
    }

    /// The axis row charges departures only: a straight, untapered axis costs nothing; a kink costs bend and a jump in
    /// radius costs radius; the gap is the widest, reported; an arc of the same shape costs a twig less than a
    /// scaffold (in τ a twig is long); and an axis that thins along its order puts the root at its start.
    #[test]
    fn the_axis_row_charges_departures_only() {
        let em = Settings { axis_taper: 0.0, ..Settings::default() };
        let straight: Vec<(Point, Point, f64)> = (0..10).map(|k| ([0.0, 0.0, 0.05 * k as f64], [0.0, 0.0, 0.05 * (k + 1) as f64], 0.004)).collect();
        let base = axis_row(&em, &straight);
        assert!((base.length - 0.5).abs() < 1e-12 && base.gap == 0.0 && base.radius.abs() < 1e-12 && base.bend == 0.0, "{base:?}");
        let mut bent = straight.clone();
        for s in &mut bent[5..] {
            *s = (add(&s.0, &[0.02 * (s.0[2] - 0.25) / 0.05, 0.0, 0.0]), add(&s.1, &[0.02 * (s.1[2] - 0.25) / 0.05, 0.0, 0.0]), s.2);
        }
        assert!(axis_row(&em, &bent).bend > 1.0);
        let mut jump = straight.clone();
        for s in &mut jump[5..] {
            s.2 = 0.006;
        }
        assert!(axis_row(&em, &jump).radius > 1.0);
        let mut gap = straight.clone();
        gap[4].0[2] += 0.02;
        assert!((axis_row(&em, &gap).gap - 0.02).abs() < 1e-12);
        let arc = |r: f64| -> f64 {
            let pts: Vec<Point> = (0..=20).map(|k| { let t = 0.05 * k as f64; [0.5 * (1.0 - t.cos()), 0.0, 0.5 * t.sin()] }).collect();
            axis_row(&em, &pts.windows(2).map(|w| (w[0], w[1], r)).collect::<Vec<_>>()).bend
        };
        assert!(arc(0.002) < arc(0.02), "{} against {}", arc(0.002), arc(0.02));
        let thinning: Vec<_> = straight.iter().enumerate().map(|(k, &(a, b, _))| (a, b, 0.006 - 0.0004 * k as f64)).collect();
        assert!(axis_row(&Settings::default(), &thinning).polarity > 0.0);
    }

    /// The fork as an insertion: radii that conserve r^γ at the declared γ depart by nothing, and by something at
    /// another γ; a lateral at the declared angle costs nothing for its angle.
    #[test]
    fn the_insertion_conserves_r_to_the_gamma() {
        let em = Settings { axis_taper: 0.0, ..Settings::default() };
        let g = em.fork_gamma;
        let (rb, ra) = (0.01, 0.008);
        let rl = (rb.powf(g) - ra.powf(g)).powf(1.0 / g);
        let th = em.fork_angle_mean;
        let before = ([0.0, 0.0, -0.05], [0.0, 0.0, 0.0], rb);
        let after = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.05], ra);
        let lateral = ([0.0, 0.0, 0.0], [0.05 * th.sin(), 0.0, 0.05 * th.cos()], rl);
        let x = insertion_row(&em, before, after, lateral);
        assert!(x.departure.abs() < 1e-12 && x.conservation < 1e-12 && x.angle < 1e-12, "{x:?}");
        let other = insertion_row(&Settings { fork_gamma: g + 1.0, ..em }, before, after, lateral);
        assert!(other.departure.abs() > 1e-3 && other.conservation > 0.0);
    }

    /// From a short piece of the pole, the grower runs along the pole to the region's ends: a chain on its axis,
    /// costing less than the seed, its state's cost the exact one's within the Laplace step's error.
    #[test]
    fn the_grower_follows_the_pole() {
        let v = pole_view();
        let scene = Scene::new(vec![v], &[], Settings::default(), Background::default(), None, 0);
        let inside = |ps: &[Point]| ps.iter().map(|p| p[2].abs() < 0.35).collect::<Vec<bool>>();
        let g = &scene.grow(&[piece_of(-0.02, 0.02, None)], &inside, &Grow::default())[0];
        assert!(g.length > 0.6 && g.cost < g.seed_cost && g.steps >= 5, "{g:?}");
        for e in &g.sticks {
            let Form::Stick { a, b } = e.form else { panic!() };
            for p in [a, b] {
                assert!(p[0].abs() < 0.01 && (p[1] - 4.0).abs() < 0.01, "{p:?}");
            }
        }
        assert!((g.kalman - g.cost).abs() < 1e-3 * g.cost.abs(), "{} against {}", g.kalman, g.cost);
    }
}
