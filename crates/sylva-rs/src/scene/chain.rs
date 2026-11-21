// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Chains grown by the likelihood (3dtrees research 130 slice-grower): the coupling along a chain, a chain's state as
//! it grows beside a scene, and the grower.
//!
//! - **The coupling** ([`Coupling`]; research 130 slices 6 and 7): a declared prior over a chain's geometry alone, a
//!   Gaussian Markov process along its length. Its ln r drifts (the taper) with stationary form departures about the
//!   drift (an Ornstein–Uhlenbeck process), and its axis's direction walks with a bend sd by radius, measured on the
//!   stand's true axes. It is the consumer's: it proposes and ranks the grower's continuations and gives a chain's
//!   coupling row ([`Coupling::row`]), never enters the cost.
//! - **A chain's state** ([`Chain`]), per scan: its group's term's evidence in information form (each cell's
//!   −log-likelihood a quadratic in (δ, reflectance level) about the point of its last read, and their sum), its own
//!   nodes' mode and precision and its alignment shift's slope and curvature (its [`Term`]), the context terms it
//!   changes, its layers by cell, and its surface class's votes. A piece added reads only the cells it enters: those
//!   exactly, through the own nodes' Newton warm from the chain's mode, the rest from the sum less the quadratics of
//!   the cells it enters. So growing a chain takes time linear in its length, and scoring an extension costs the
//!   piece's cells. The quadratics are the Laplace step the term already takes, each about its own last read; a grown
//!   chain's cost is read again exactly (`score`) when it is handed over.
//! - **The grower** ([`Scene::grow`]): from a seed stick, a beam of a few chains grows one piece at a time, the ends
//!   taking turns. A piece leaves the end along its axis, 4 r long within [`Grow`]'s `step`; its options are the
//!   coupling's: straight at the radius or e^(±2 sd) of it, or turned one or two bend sds either way in two planes.
//!   Each option's cost to add is conditioned on its chain (one group, the scene unchanged). A chain extends by the
//!   options whose cost to add is negative, inside the region and within the longest length; an end with none stops.
//!   The beam keeps the chains least in cost plus the coupling's departures (½ Σ z²), and the least of them is handed
//!   over. A seed that doesn't beat nothing isn't grown.

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
use super::term::{CellData, Owner, Term, eliminate, elem_data, median_class, owner, settle, slots, term_nodes, vote, clipped, mean_of};

/// The coupling's declared row (3dtrees research 128: taper and form departures; research 130 slice 7: the bend sd
/// by radius, measured on the stand's true axes at the median: twigs 1.3, branches 0.48, scaffolds 0.19 rad per √m).
#[derive(Debug, Clone, Copy)]
pub struct Coupling {
    /// The taper: ln r's drift per metre along the chain.
    pub drift: f64,
    /// The form departures about the drift: their sd, and the length (m) over which they decorrelate.
    pub form_sd: f64,
    pub form_range: f64,
    /// The bend sd (rad per √m) under [`BEND_BANDS`]' first radius (twigs), between them (branches) and over
    /// (scaffolds and stems).
    pub bend_sd: [f64; 3],
}

impl Default for Coupling {
    fn default() -> Self {
        Coupling { drift: -0.15, form_sd: 0.2, form_range: 1.0, bend_sd: [1.3, 0.48, 0.19] }
    }
}

/// The radii (m) between the bend sd's bands.
pub const BEND_BANDS: [f64; 2] = [0.005, 0.01];

/// A chain's coupling row: its length and widest end-to-start gap (m), and the coupling's cost of its radii and its
/// bends (nats).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouplingRow {
    pub length: f64,
    pub gap: f64,
    pub radius: f64,
    pub bend: f64,
}

impl Coupling {
    /// The bend sd (rad per √m) at radius r.
    pub fn bend_sd(&self, r: f64) -> f64 {
        self.bend_sd[BEND_BANDS.iter().filter(|&&b| r >= b).count()]
    }

    /// The form departures' sd over a step of `s` m along the chain (the OU's transition).
    pub fn form_step(&self, s: f64) -> f64 {
        self.form_sd * (1.0 - (-2.0 * s / self.form_range).exp()).sqrt()
    }

    /// The row along sticks (a, b, r) joined end to end in order: ln r about the drift from the first stick, an OU
    /// anchored on it, and per joint the turn between the axes over s, the distance between the sticks' middles, at
    /// the bend sd of their mean ln r.
    pub fn row(&self, sticks: &[(Point, Point, f64)]) -> CouplingRow {
        let mut out = CouplingRow { length: 0.0, gap: 0.0, radius: 0.0, bend: 0.0 };
        let Some(&(a0, b0, r0)) = sticks.first() else { return out };
        let mut pos = 0.5 * norm(&sub(&b0, &a0));
        let x0 = r0.ln() - self.drift * pos;
        let (mut x, mut prev) = (x0, (a0, b0, r0));
        out.length = 2.0 * pos;
        for &(a, b, r) in &sticks[1..] {
            let (ln, ln_prev) = (norm(&sub(&b, &a)), norm(&sub(&prev.1, &prev.0)));
            let s = 0.5 * (ln + ln_prev);
            pos += s;
            let xn = r.ln() - self.drift * pos;
            let var = self.form_step(s).powi(2);
            let e = (xn - x0) - (-s / self.form_range).exp() * (x - x0);
            out.radius += 0.5 * e * e / var + 0.5 * (2.0 * std::f64::consts::PI * var).ln();
            let turn = dot(&normalize(&sub(&b, &a)), &normalize(&sub(&prev.1, &prev.0))).clamp(-1.0, 1.0).acos();
            let vb = self.bend_sd((r * prev.2).sqrt()).powi(2) * s;
            out.bend += 0.5 * turn * turn / vb + (2.0 * std::f64::consts::PI * vb).ln();
            out.gap = out.gap.max(norm(&sub(&a, &prev.1)));
            out.length += ln;
            (x, prev) = (xn, (a, b, r));
        }
        out
    }
}

/// The grower's settings: the chains a seed's beam keeps, the longest chain (m), a piece's length bounds (m; 4 r
/// within them), and the coupling.
#[derive(Debug, Clone, Copy)]
pub struct Grow {
    pub beam: usize,
    pub max_length: f64,
    pub step: (f64, f64),
    pub coupling: Coupling,
}

impl Default for Grow {
    fn default() -> Self {
        Grow { beam: 3, max_length: 1.5, step: (0.03, 0.10), coupling: Coupling::default() }
    }
}

/// A grown chain as handed over: its sticks in order along it, its cost to add per scan read exactly (then, with a
/// foliage field, the field's part) and in total, the state's own cost (`kalman`, from the quadratics), the seed's
/// cost, the pieces it added, the coupling's departures (½ Σ z²) and its length (m).
#[derive(Debug, Clone)]
pub struct Grown {
    pub sticks: Vec<Element>,
    pub scan: Vec<f64>,
    pub cost: f64,
    pub kalman: f64,
    pub seed_cost: f64,
    pub steps: usize,
    pub coupling: f64,
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
/// there, and their plug-in ([`Term`]'s `plug`).
#[derive(Debug, Clone)]
struct Read {
    q: Quadratic,
    vote: Option<u8>,
    deposit: Deposit,
    plug: f64,
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
    coupling: f64,
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
    fresh: Vec<(CellData, Option<u8>, Deposit, f64)>,
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
        Chain { pieces: Vec::new(), order: VecDeque::new(), scans: vec![ChainScan::default(); n], delta: FixedHits::default(), cost: 0.0, coupling: 0.0,
                length: 0.0, steps: 0, done: [false; 2], turn: 1 }
    }

    /// Its sticks in order along it.
    fn sticks(&self) -> Vec<Element> {
        self.order.iter().map(|&k| self.pieces[k]).collect()
    }

    fn score(&self) -> f64 {
        self.cost + self.coupling
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
            // each live chain's options at its growing end, inside the region and within the longest length
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
                pool.extend(ext.into_iter().map(|k| (trials[k].as_ref().unwrap().cost + m.coupling + props[k].3, i, Some(k))));
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
        Grown { cost: scan.iter().sum(), scan, sticks, kalman: best.cost, seed_cost, steps: best.steps, coupling: best.coupling, length: best.length }
    }

    /// The next piece's options from chain `m`'s end `end` (0 its first stick's start, 1 its last's end), each with
    /// the coupling's departure ½ z² (the module's docs).
    fn options(&self, m: &Chain, end: usize, opts: &Grow) -> Vec<(Element, f64)> {
        let tip = m.pieces[if end == 1 { *m.order.back().unwrap() } else { *m.order.front().unwrap() }];
        let Form::Stick { a, b } = tip.form else { return Vec::new() };
        let (p, u) = if end == 1 { (b, normalize(&sub(&b, &a))) } else { (a, normalize(&sub(&a, &b))) };
        let ln = (4.0 * tip.r).clamp(opts.step.0, opts.step.1);
        let (sd, th) = (opts.coupling.form_step(ln), opts.coupling.bend_sd(tip.r) * ln.sqrt());
        let p1 = normalize(&cross(&u, &if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
        let p2 = cross(&u, &p1);
        let mut dirs: Vec<(Point, f64, f64)> = [0.0, -2.0, 2.0].iter().map(|&k| (u, k, 0.5 * k * k)).collect();
        for j in [1.0, 2.0] {
            for q in [p1, p2] {
                for s in [1.0, -1.0] {
                    dirs.push((add(&scale(&u, (j * th).cos()), &scale(&q, s * (j * th).sin())), 0.0, 0.5 * j * j));
                }
            }
        }
        dirs.into_iter().map(|(d, k, z)| {
            let q = add(&p, &scale(&d, ln));
            let r = tip.r * (k * sd).exp();
            let form = if end == 1 { Form::Stick { a: p, b: q } } else { Form::Stick { a: q, b: p } };
            (Element { form, r, mismatch: self.em.mismatch_sd(r), ..tip }, z)
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
        let mut term = (!members.is_empty()).then(|| {
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
                super::term::Local { ll: total.v, parts: [0.0; 4], g: -total.g, hess: clipped(-total.h), deposit: Vec::new(), plug: 0.0 }
            });
            let el = eliminate(own_sd, &at, y);
            Term {
                align, fit: -at.ll + 0.5 * (y[0] * y[0] + y[1] * y[1]), logdet: 0.5 * el.det.ln(), slope: el.slope, curv: el.curv, own: y, start, own_sd,
                own_cov: el.pi, own_link: el.pl, ray: [0.0; 4], cells: (cs.cells.len() - gone.len() + fresh_cells.len()) as u32, cells_on: 0, arc: 0.0,
                across: mean_of(&es, |x| 2.0 * x.r / em.footprint(norm(&sub(&x.mid(), &v.origin)))), surface: class, deposit: Vec::new(), plug: 0.0,
            }
        });
        // the deposits of the cells it reads, at the term's mode, and the term's plug-in: the chain's less the cells
        // re-read, and those read now
        let x = term.as_ref().map_or(Vector2::zeros(), |t| Vector2::new(fit.held.shift(&t.align) + t.own_sd[0] * t.own[0], t.own_sd[1] * t.own[1]));
        let fresh: Vec<(CellData, Option<u8>, Deposit, f64)> = fresh_cells.into_iter().map(|f| {
            let (dep, plug) = if f.hits.is_empty() { (Vec::new(), 0.0) } else {
                let (hits, p) = f.take(&mixture_taken(&f.pulses, x[0], x[1], em).1);
                (foliage::deposit(hits.into_iter()), -p)
            };
            let vote = votes.get(&f.cell).copied();
            (f, vote, dep, plug)
        }).collect();
        if let Some(t) = &mut term {
            t.plug = cs.term.as_ref().map_or(0.0, |t| t.plug) - gone.iter().map(|k| cs.cells[k].plug).sum::<f64>() + fresh.iter().map(|x| x.3).sum::<f64>();
        }
        let mut over: HashMap<Owner, Option<Term>> = cs.ctx.clone();
        over.extend(ctx.iter().cloned());
        over.insert(g, term.clone());
        let cost = fit.total(&over) - fit.cost;
        ScanTrial { layers, gone, fresh, term, ctx, cost }
    }

    /// Chain `c` with trial `t` taken at its end `end`, the coupling's departure `z` added.
    fn take(&self, c: &Chain, t: Trial, end: usize, z: f64) -> Chain {
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
                for (f, vote, deposit, plug) in st.fresh {
                    let q = Quadratic::of(&f.read(Dual2::var(x[0], 0), lv, &self.em), x);
                    sum.add(&q, 1.0);
                    if let Some(v) = vote {
                        cs.votes[v as usize] += 1;
                    }
                    cs.cells.insert(f.cell, Read { q, vote, deposit, plug });
                }
            }
            cs.sum = Some(sum);
            cs.term = st.term;
            cs.ctx.extend(st.ctx);
        }
        n.delta = t.delta;
        n.cost = t.cost;
        n.coupling += z;
        n.length += t.piece.length();
        n.steps += !c.pieces.is_empty() as usize;
        n.turn = 1 - end;
        n
    }
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

    /// The coupling row is pure geometry: radii scaled together cost what they did (the OU's anchor moves with
    /// them), a bend and a jump in radius cost, the gap is the widest; and a twig bends more freely than a scaffold.
    #[test]
    fn the_coupling_row_is_pure_geometry() {
        let cp = Coupling::default();
        let straight: Vec<(Point, Point, f64)> = (0..5).map(|k| ([0.0, 0.0, 0.05 * k as f64], [0.0, 0.0, 0.05 * (k + 1) as f64], 0.004)).collect();
        let base = cp.row(&straight);
        assert!((base.length - 0.25).abs() < 1e-12 && base.gap == 0.0);
        let scaled: Vec<_> = straight.iter().map(|&(a, b, r)| (a, b, 2.0 * r)).collect();
        assert!((cp.row(&scaled).radius - base.radius).abs() < 1e-9);
        let mut bent = straight.clone();
        bent[3] = ([0.0, 0.0, 0.15], [0.03, 0.0, 0.19], 0.004);
        assert!(cp.row(&bent).bend > base.bend + 1.0);
        let mut jump = straight.clone();
        jump[2].2 = 0.008;
        assert!(cp.row(&jump).radius > base.radius + 1.0);
        let mut gap = straight.clone();
        gap[4].0[2] += 0.02;
        assert!((cp.row(&gap).gap - 0.02).abs() < 1e-12);
        let excess = |r: f64| {
            let with = |s: &[(Point, Point, f64)]| cp.row(&s.iter().map(|&(a, b, _)| (a, b, r)).collect::<Vec<_>>()).bend;
            with(&bent) - with(&straight)
        };
        assert!(excess(0.003) < excess(0.02));
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
