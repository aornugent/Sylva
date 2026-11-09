// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A research prototype (3dtrees research 113): the marginal likelihood of a
//! structural hypothesis over one fixed echo set, so that hypotheses of the
//! same plants (which stems, which axes, which topology) can be ranked by
//! their evidence.
//!
//! - **The domain and its echo set, once per cluster** ([`cluster_table`]).
//!   The domain is the union of tubes ([`crate::qsm::joint::window`]) about
//!   every segment any hypothesis of the cluster draws. Each pulse that
//!   reaches it (its last echo at or past its entry) is one record: its chord
//!   through the domain `[C0, C1]` (the first entry to the last exit, gaps
//!   between tubes included), its fate there (its first echo in the chord: a
//!   stop; none: a pass), and its geometry against each segment whose tube
//!   it crosses (impact, range of closest approach, angle, and how both move
//!   with an offset of the axis). The set doesn't depend on any hypothesis:
//!   every hypothesis explains every record.
//! - **A hypothesis** is a subset of the segments with a parent for each (a
//!   forest over them). Each pulse is seen from the active segment it meets
//!   first; its likelihood is [`crate::qsm::joint`]'s pair term with the
//!   domain's chord, so foliage competes with that wood along the whole chord.
//!   A pulse no active segment's tube holds is foliage alone. Foliage is one
//!   log rate per height layer of the domain (fixed per cluster), so the
//!   foliage a hypothesis doesn't call wood is priced by the same field in
//!   every hypothesis.
//! - **Beam oversampling** (research 103 §15.6): a pulse weighs `1 / m`, `m =
//!   max(1, (w / (step t))^2)` the pulses of its scan inside one footprint at
//!   its range (`step` the scan's angular step), times a global `temper`.
//! - **The prior,** a proper Gaussian over the hypothesis' state, built as a
//!   product of conditionals down each tree (so it is normalised): at a root,
//!   log R about its claim and the centre about its axis; along a branch, log R
//!   a random walk between the segments' middles about the claim's step, and
//!   the centre one too (the parent's offset carried into the child's frame);
//!   at a fork, a child's log R adds the fork's sd and its centre is anchored
//!   afresh; each layer's log foliage rate about the prior's. A walk whose
//!   variance grows with length is the same process however finely a branch
//!   is cut, which is what makes re-sectioning nearly free.
//! - **The evidence:** Levenberg-Marquardt to the mode of the log posterior
//!   (ownership re-decided at every point), then Laplace: `log Z = log p(y |
//!   x) + log p(x) + d/2 log 2 pi - 1/2 log det P`, `P` the prior's precision
//!   plus the likelihood's curvature made positive semi-definite.
//! - **A reference** ([`ClusterTable::importance`]): importance sampling from
//!   a multivariate t about the mode, the exact likelihood (ownership re-decided
//!   per draw).

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::evidence::{pairs, ScanIndex};
use crate::qsm::joint::{deleted, frames, pair_derivs, pair_value, pair_wood_share, pool, window, JointParams, Pulse};
use crate::transform::{cross, dot, sub};
use crate::Point;

/// A pair's geometry against one segment's axis (see [`crate::qsm::joint::Pulse`]).
#[derive(Debug, Clone, Copy, Default)]
struct Geo {
    b: f64,
    t: f64,
    sin: f64,
    jb: [f64; 2],
    jt: [f64; 2],
}

/// One cluster's fixed echo set and every candidate segment's pairs with it (see the module documentation).
#[derive(Debug, Clone, Default)]
pub struct ClusterTable {
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r_prior: Vec<f64>,
    fr: Vec<(Point, Point, Point, f64)>,
    /// Per record (pulse): its scan and pulse, its chord through the domain, its first echo there (`x`, NaN for a
    /// pass) and that echo's place in its pulse, whether it echoed before the chord, whether a deleted echo could lie
    /// in it, its height layer, its beam oversampling `m`, the range of its chord's middle and the height there.
    pub scan: Vec<u32>,
    pub pulse: Vec<u32>,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub x: Vec<f64>,
    pub echo: Vec<u8>,
    pub stop: Vec<bool>,
    pub early: Vec<bool>,
    pub deleted: Vec<bool>,
    pub layer: Vec<u16>,
    pub m: Vec<f64>,
    pub tmid: Vec<f64>,
    pub z: Vec<f64>,
    pub xy: Vec<[f64; 2]>,
    pub n_layers: usize,
    /// The pairs, by record: `start[i]..start[i + 1]` index `seg` and the geometry.
    pub start: Vec<u32>,
    pub seg: Vec<u32>,
    geo: Vec<Geo>,
}

/// The prior's and the fit's settings (see the module documentation); defaults after [`crate::qsm::joint_fit::PriorParams`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceOpts {
    /// Extra weight on every pulse (1: the oversampling weight alone), and whether the oversampling weight is used.
    pub temper: f64,
    pub oversample: bool,
    /// log R at a root: sd about its claim; per sqrt(m) along a branch; the extra sd of a fork's child.
    pub sig_root: f64,
    pub sig_eta: f64,
    pub sig_fork: f64,
    /// The centre: its anchor at a root or a fork's child (`centre_rel r + centre_abs`), its walk per sqrt(m).
    pub centre_rel: f64,
    pub centre_abs: f64,
    pub sig_delta: f64,
    /// Foliage per layer: prior median (1/m) and log sd.
    pub lam_prior: f64,
    pub lam_sd: f64,
    /// Levenberg-Marquardt: iterations at most, and the stopping decrement (log units).
    pub iters: usize,
    pub tol: f64,
    /// One pulse in `thin` is used (by a hash of its scan, pulse and `thin_seed`), each weighed `thin` times.
    pub thin: u32,
    pub thin_seed: u64,
    /// Who sees a pulse: 0 the active segment it meets first (as `joint_tree`), 1 of the segments whose footprint it
    /// touches the one that explains it best (else the nearest).
    pub owner_rule: u8,
    /// Stems may not overlap: the sd (m) of a soft hinge on two non-adjacent segments' circles overlapping (0: off).
    pub exclude: f64,
}

impl Default for EvidenceOpts {
    fn default() -> Self {
        EvidenceOpts { temper: 1.0, oversample: true, sig_root: 0.5, sig_eta: 0.5, sig_fork: 0.17, centre_rel: 1.0, centre_abs: 0.003, sig_delta: 0.02, lam_prior: 1.0, lam_sd: 2.0, iters: 200, tol: 1e-4, thin: 1, thin_seed: 0, owner_rule: 2, exclude: 0.002 }
    }
}

/// A hypothesis' fit and evidence (see [`ClusterTable::evidence`]).
#[derive(Debug, Clone, Default)]
pub struct EvidenceFit {
    pub logz: f64,
    pub loglik: f64,
    pub logprior: f64,
    pub logdet: f64,
    pub dim: usize,
    pub iters: usize,
    pub converged: bool,
    /// The share of the likelihood's curvature (absolute eigenvalues) that was negative at the mode and dropped.
    pub clipped: f64,
    /// The state at the mode and its marginal sds: per active segment `[d1, d2, eta]` in its order, then each layer's
    /// log rate.
    pub x: Vec<f64>,
    pub sd: Vec<f64>,
    /// Each segment's place among the active ones (-1: inactive).
    pub index: Vec<i64>,
    /// Per record at the mode: its owner (a segment, -1: foliage alone), its weighted log likelihood, and a stop's
    /// chance of being wood's echo (NaN for a pass).
    pub owner: Vec<i32>,
    pub ll: Vec<f64>,
    pub wood: Vec<f64>,
    /// Pair evaluations: values, derivatives.
    pub evals: [u64; 2],
    /// The prior's precision and the likelihood's (clipped) curvature at the mode, `P` = their sum, as dense
    /// matrices (row-major, `dim x dim`).
    pub precision: Vec<f64>,
}

/// Importance sampling's estimate of log Z (see [`ClusterTable::importance`]).
#[derive(Debug, Clone, Default)]
pub struct Importance {
    pub logz: f64,
    pub se: f64,
    pub ess: f64,
    pub max_share: f64,
    pub log_w: Vec<f64>,
}

/// The cluster's table of segments `a[i] -> b[i]` (claimed radius `r_prior[i]`): every pulse that reaches the union
/// of their tubes, once (see the module documentation), whose stop (or chord's middle) lies at heights `z0..z1` (the slab
/// the cluster is judged in: hypotheses draw their wood past its ends, so where a segment ends doesn't decide which
/// pulses count). Layers are `layer_h` m tall from `z0` (0: one layer); `step`
/// is the scans' angular step (rad) for the beam's oversampling.
#[allow(clippy::too_many_arguments)]
pub fn cluster_table(scans: &[&ScanIndex], a: &[Point], b: &[Point], r_prior: &[f64], z0: f64, z1: f64, layer_h: f64, step: f64, like: &JointParams, threads: usize) -> Result<ClusterTable> {
    let n = a.len();
    if b.len() != n || r_prior.len() != n || n == 0 {
        return Err(Error::invalid("cluster table: a, b and r_prior need one entry per segment"));
    }
    let win: Vec<f64> = r_prior.iter().map(|&r| window(r)).collect();
    let pp = pairs(scans, a, b, &win, threads)?;
    let fr = frames(a, b);
    // records keyed by (scan, pulse); each pair's geometry and its chord through its own tube
    let mut key: HashMap<u64, usize> = HashMap::new();
    struct Acc {
        scan: u32,
        pulse: u32,
        e0: usize,
        e1: usize,
        c0: f64,
        c1: f64,
        dir: Point,
        returns: u8,
        pairs: Vec<(u32, Geo)>,
    }
    let mut acc: Vec<Acc> = Vec::new();
    for c in 0..n {
        let (u, e1, e2, _) = fr[c];
        for i in pp.cyl_start[c] as usize..pp.cyl_start[c + 1] as usize {
            let d = pp.dir[i].map(|x| x as f64);
            let o = scans[pp.scan[i] as usize].origin();
            let cr = cross(&u, &d);
            let sin = dot(&cr, &cr).sqrt();
            if sin < 1e-9 {
                continue;
            }
            let nrm = cr.map(|x| x / sin);
            let bimp = dot(&sub(&o, &a[c]), &nrm);
            let du = dot(&d, &u);
            let p_hat = [0, 1, 2].map(|k| (d[k] - du * u[k]) / sin);
            let t = pp.t[i] as f64;
            let half = (win[c] * win[c] - bimp * bimp).max(0.0).sqrt() / sin;
            let g = Geo { b: bimp, t, sin, jb: [dot(&nrm, &e1), dot(&nrm, &e2)], jt: [dot(&p_hat, &e1) / sin, dot(&p_hat, &e2) / sin] };
            let k = (pp.scan[i] as u64) << 32 | pp.pulse[i] as u64;
            let j = *key.entry(k).or_insert_with(|| {
                acc.push(Acc { scan: pp.scan[i], pulse: pp.pulse[i], e0: pp.echo_start[i] as usize, e1: pp.echo_start[i + 1] as usize, c0: f64::INFINITY, c1: f64::NEG_INFINITY, dir: d, returns: pp.returns[i], pairs: Vec::new() });
                acc.len() - 1
            });
            let r = &mut acc[j];
            r.c0 = r.c0.min(t - half);
            r.c1 = r.c1.max(t + half);
            r.pairs.push((c as u32, g));
        }
    }
    // in a fixed order: by scan and pulse
    acc.sort_unstable_by_key(|r| (r.scan, r.pulse));
    let mut out = ClusterTable { a: a.to_vec(), b: b.to_vec(), r_prior: r_prior.to_vec(), fr, start: vec![0], ..Default::default() };
    for r in &acc {
        let echoes = &pp.range[r.e0..r.e1];
        let numbers = &pp.return_number[r.e0..r.e1];
        let first_at = echoes.iter().position(|&e| (e as f64) >= r.c0 && (e as f64) <= r.c1);
        let x = first_at.map_or(f64::NAN, |k| echoes[k] as f64);
        let limit = if x.is_finite() { x } else { r.c1 };
        let hidden = deleted(echoes, numbers, r.returns).into_iter().any(|(lo, hi)| lo < limit && hi > r.c0);
        let tm = 0.5 * (r.c0 + r.c1);
        let o = scans[r.scan as usize].origin();
        let z = o[2] + if x.is_finite() { x } else { tm } * r.dir[2];
        if !(z >= z0 && z <= z1) {
            continue;
        }
        out.scan.push(r.scan);
        out.pulse.push(r.pulse);
        out.c0.push(r.c0);
        out.c1.push(r.c1);
        out.x.push(x);
        out.echo.push(first_at.map_or(u8::MAX, |k| k.min(254) as u8));
        out.stop.push(x.is_finite());
        out.early.push(echoes.first().is_some_and(|&e| (e as f64) < r.c0));
        out.deleted.push(hidden);
        out.tmid.push(tm);
        out.z.push(z);
        let at = if x.is_finite() { x } else { tm };
        out.xy.push([o[0] + at * r.dir[0], o[1] + at * r.dir[1]]);
        let w = like.footprint(tm);
        out.m.push(if step > 0.0 { (w / (step * tm)).powi(2).max(1.0) } else { 1.0 });
        for (s, g) in &r.pairs {
            out.seg.push(*s);
            out.geo.push(*g);
        }
        out.start.push(out.seg.len() as u32);
    }
    if layer_h < 0.0 {
        // a foliage rate per voxel of -layer_h m (of the record's stop, or its chord's middle), in order of first use
        let v = -layer_h;
        let mut vox: HashMap<(i64, i64, i64), u16> = HashMap::new();
        out.layer = out.xy.iter().zip(&out.z).map(|(q, &z)| {
            let k = ((q[0] / v).floor() as i64, (q[1] / v).floor() as i64, (z / v).floor() as i64);
            let n = vox.len() as u16;
            *vox.entry(k).or_insert(n)
        }).collect();
        out.n_layers = vox.len().max(1);
        return Ok(out);
    }
    let zmax = out.z.iter().copied().fold(z0, f64::max);
    out.n_layers = if layer_h > 0.0 { (((zmax - z0) / layer_h).floor() as usize + 1).max(1) } else { 1 };
    let nl = out.n_layers;
    out.layer = out.z.iter().map(|&z| if layer_h > 0.0 { (((z - z0) / layer_h).floor().max(0.0) as usize).min(nl - 1) as u16 } else { 0 }).collect();
    Ok(out)
}

/// One scalar Gaussian factor of the prior: `sum a x ~ N(m, s^2)`.
struct Factor {
    a: Vec<(usize, f64)>,
    m: f64,
    s: f64,
}

/// The prior as `-1/2 x'Qx + h'x + c`.
struct Prior {
    q: DMatrix<f64>,
    h: DVector<f64>,
    c: f64,
}

impl Prior {
    fn of(factors: &[Factor], dim: usize) -> Prior {
        let (mut q, mut h, mut c) = (DMatrix::zeros(dim, dim), DVector::zeros(dim), 0.0);
        let l2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        for f in factors {
            let w = 1.0 / (f.s * f.s);
            for &(i, ai) in &f.a {
                h[i] += ai * f.m * w;
                for &(j, aj) in &f.a {
                    q[(i, j)] += ai * aj * w;
                }
            }
            c += -0.5 * f.m * f.m * w - f.s.ln() - l2pi;
        }
        Prior { q, h, c }
    }

    fn value(&self, x: &DVector<f64>) -> f64 {
        -0.5 * x.dot(&(&self.q * x)) + self.h.dot(x) + self.c
    }
}

/// A hypothesis as the fit reads it: its active segments' places, the layers' offset and the prior.
struct Setup {
    index: Vec<i64>,
    active: Vec<usize>,
    dim: usize,
    l0: usize,
    prior: Prior,
    weight: Vec<f64>,
    owner_rule: u8,
    /// Segment pairs under the non-overlap hinge, and its sd.
    excl: Vec<(usize, usize)>,
    s_ex: f64,
}

impl ClusterTable {
    pub fn n_records(&self) -> usize {
        self.scan.len()
    }

    pub fn n_pairs(&self) -> usize {
        self.seg.len()
    }

    fn setup(&self, active: &[bool], parent: &[i64], o: &EvidenceOpts, like: &JointParams) -> Result<Setup> {
        let ns = self.a.len();
        if active.len() != ns || parent.len() != ns {
            return Err(Error::invalid("cluster evidence: active and parent need one entry per segment"));
        }
        if parent.iter().any(|&p| p >= ns as i64) {
            return Err(Error::invalid("cluster evidence: a parent is out of range"));
        }
        let mut index = vec![-1i64; ns];
        let mut act = Vec::new();
        for (k, &on) in active.iter().enumerate() {
            if on {
                index[k] = act.len() as i64;
                act.push(k);
            }
        }
        let l0 = 3 * act.len();
        let dim = l0 + self.n_layers;
        // a parent counts only if active; of a parent's children the widest (then the straightest) continues it
        let par = |k: usize| -> Option<usize> { (parent[k] >= 0 && active[parent[k] as usize]).then_some(parent[k] as usize) };
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); ns];
        for &k in &act {
            if let Some(p) = par(k) {
                kids[p].push(k);
            }
        }
        let cont: Vec<Option<usize>> = (0..ns).map(|p| kids[p].iter().copied().max_by(|&i, &j| self.r_prior[i].total_cmp(&self.r_prior[j]).then(dot(&self.fr[j].0, &self.fr[p].0).total_cmp(&dot(&self.fr[i].0, &self.fr[p].0))))).collect();
        let mut fs: Vec<Factor> = Vec::new();
        for &k in &act {
            let pk = 3 * index[k] as usize;
            let (_, e1, e2, len) = self.fr[k];
            let anchor = o.centre_rel * self.r_prior[k] + o.centre_abs;
            match par(k) {
                None => {
                    fs.push(Factor { a: vec![(pk + 2, 1.0)], m: self.r_prior[k].ln(), s: o.sig_root });
                    for i in 0..2 {
                        fs.push(Factor { a: vec![(pk + i, 1.0)], m: 0.0, s: anchor });
                    }
                }
                Some(p) => {
                    let pp = 3 * index[p] as usize;
                    let (_, f1, f2, lp) = self.fr[p];
                    let dist = 0.5 * (len + lp);
                    let fork = cont[p] != Some(k);
                    let s = (o.sig_eta * o.sig_eta * dist + if fork { o.sig_fork * o.sig_fork } else { 0.0 }).sqrt();
                    fs.push(Factor { a: vec![(pk + 2, 1.0), (pp + 2, -1.0)], m: self.r_prior[k].ln() - self.r_prior[p].ln(), s });
                    if fork {
                        for i in 0..2 {
                            fs.push(Factor { a: vec![(pk + i, 1.0)], m: 0.0, s: anchor });
                        }
                    } else {
                        // the parent's offset in the child's frame: T = [[f1.e1, f2.e1], [f1.e2, f2.e2]]
                        let t = [[dot(&f1, &e1), dot(&f2, &e1)], [dot(&f1, &e2), dot(&f2, &e2)]];
                        let s = o.sig_delta * dist.max(1e-4).sqrt();
                        for (i, row) in t.iter().enumerate() {
                            fs.push(Factor { a: vec![(pk + i, 1.0), (pp, -row[0]), (pp + 1, -row[1])], m: 0.0, s });
                        }
                    }
                }
            }
        }
        for l in 0..self.n_layers {
            fs.push(Factor { a: vec![(l0 + l, 1.0)], m: o.lam_prior.ln(), s: o.lam_sd });
        }
        let prior = Prior::of(&fs, dim);
        let weight: Vec<f64> = (0..self.n_records()).map(|i| {
            let keep = o.thin <= 1 || hash(self.scan[i], self.pulse[i], o.thin_seed) % o.thin as u64 == 0;
            if !keep {
                return 0.0;
            }
            o.temper * o.thin.max(1) as f64 / if o.oversample { self.m[i] } else { 1.0 }
        }).collect();
        let _ = like;
        // pairs of segments that may not overlap: not parent and child, not siblings, near each other across and
        // along their axes
        let mut excl = Vec::new();
        let root = |mut k: usize| -> usize {
            while let Some(p) = par(k) {
                k = p;
            }
            // a root whose parent is -2 - g belongs to tree g (an RCT tree whose rows leave its base node are roots)
            if parent[k] <= -2 { ns + (-2 - parent[k]) as usize } else { k }
        };
        if o.exclude > 0.0 {
            for (x, &i) in act.iter().enumerate() {
                for &j in &act[x + 1..] {
                    // within one tree, branches meet and overlap at their forks (a fan's axes for their length): only
                    // segments of different trees (roots) may not overlap
                    if root(i) == root(j) {
                        continue;
                    }
                    let mi: Point = std::array::from_fn(|k| 0.5 * (self.a[i][k] + self.b[i][k]));
                    let mj: Point = std::array::from_fn(|k| 0.5 * (self.a[j][k] + self.b[j][k]));
                    let dm = sub(&mj, &mi);
                    let ax = dot(&dm, &self.fr[i].0);
                    let across = (dot(&dm, &dm) - ax * ax).max(0.0).sqrt();
                    if ax.abs() < 0.5 * (self.fr[i].3 + self.fr[j].3) && across < self.r_prior[i] + self.r_prior[j] + 0.15 {
                        excl.push((i, j));
                    }
                }
            }
        }
        Ok(Setup { index, active: act, dim, l0, prior, weight, owner_rule: o.owner_rule, excl, s_ex: o.exclude })
    }

    /// The log prior at `x`: the Gaussian's, and the non-overlap hinge (not normalised).
    fn prior_value(&self, x: &DVector<f64>, st: &Setup) -> f64 {
        st.prior.value(x) + if st.excl.is_empty() { 0.0 } else { self.exclusion(x, st).0 }
    }

    /// The non-overlap penalty at `x` with its gradient and Gauss-Newton curvature (see [`EvidenceOpts::exclude`]): a
    /// hinge on `r_i + r_j - d`, `d` the distance between the segments' middles across segment i's axis.
    fn exclusion(&self, x: &DVector<f64>, st: &Setup) -> (f64, DVector<f64>, DMatrix<f64>) {
        let (mut f, mut g, mut h) = (0.0, DVector::zeros(st.dim), DMatrix::zeros(st.dim, st.dim));
        for &(i, j) in &st.excl {
            let (pi, pj) = (3 * st.index[i] as usize, 3 * st.index[j] as usize);
            let (ui, e1i, e2i, _) = self.fr[i];
            let (_, e1j, e2j, _) = self.fr[j];
            let off = |e1: &Point, e2: &Point, p: usize| -> Point { std::array::from_fn(|k| x[p] * e1[k] + x[p + 1] * e2[k]) };
            let (oi, oj) = (off(&e1i, &e2i, pi), off(&e1j, &e2j, pj));
            let dm: Point = std::array::from_fn(|k| 0.5 * (self.a[j][k] + self.b[j][k]) + oj[k] - 0.5 * (self.a[i][k] + self.b[i][k]) - oi[k]);
            let ax = dot(&dm, &ui);
            let v: Point = std::array::from_fn(|k| dm[k] - ax * ui[k]);
            let d = dot(&v, &v).sqrt().max(1e-9);
            let (ri, rj) = (x[pi + 2].exp(), x[pj + 2].exp());
            let over = ri + rj - d;
            if over <= 0.0 {
                continue;
            }
            // (coincident axes: no direction to move them apart, the penalty alone)
            let vh = if d > 1e-6 { v.map(|c| c / d) } else { [0.0; 3] };
            let perp = |e: &Point| -> f64 { dot(&vh, e) - dot(&vh, &ui) * dot(e, &ui) };
            // d(over)/d(state)
            let mut jac = vec![(pi, perp(&e1i)), (pi + 1, perp(&e2i)), (pi + 2, ri), (pj, -perp(&e1j)), (pj + 1, -perp(&e2j)), (pj + 2, rj)];
            jac.retain(|q| q.1 != 0.0);
            let w = 1.0 / (st.s_ex * st.s_ex);
            f -= 0.5 * over * over * w;
            for &(a, da) in &jac {
                g[a] -= over * w * da;
                for &(b, db) in &jac {
                    h[(a, b)] -= w * da * db;
                }
            }
        }
        (f, g, h)
    }

    /// Record `i` as its owner sees it at state `x`: the pair (with the domain's chord) and the owner (None: foliage
    /// alone, a pair no wood can stop).
    fn view(&self, i: usize, x: &DVector<f64>, st: &Setup, like: &JointParams) -> (Pulse, Option<usize>) {
        let base = Pulse { b: 1e3, t: self.tmid[i], sin: 1.0, jb: [0.0; 2], jt: [0.0; 2], c0: self.c0[i], c1: self.c1[i], x: if self.stop[i] { self.x[i] } else { 0.0 }, deleted: self.deleted[i], stop: self.stop[i], early: self.early[i], q: 1.0 };
        let mut best: Option<(bool, f64, usize)> = None;
        let ell = x[st.l0 + self.layer[i] as usize];
        for p in self.start[i] as usize..self.start[i + 1] as usize {
            let k = self.seg[p] as usize;
            let j = st.index[k];
            if j < 0 {
                continue;
            }
            let j = 3 * j as usize;
            let g = &self.geo[p];
            let (d1, d2, r) = (x[j], x[j + 1], x[j + 2].exp());
            let bp = g.b - (g.jb[0] * d1 + g.jb[1] * d2);
            let tp = g.t + (g.jt[0] * d1 + g.jt[1] * d2);
            let w = like.footprint(g.t);
            // rule 2: touched only where the wood would more likely than not echo (inside its apparent edge, and wide
            // enough to give the footprint's share tau at all), so a thin or vanishing stem can't take a pulse from
            // the wood behind it
            let touch = if st.owner_rule == 2 { bp.abs() < r + like.app(w) && 2.0 * r >= like.tau * w } else { bp.abs() < r + 0.5 * w };
            let key = if !touch {
                bp.abs() - r
            } else if st.owner_rule == 1 {
                // the best explanation among the touched: its pair's negative log likelihood
                -pair_value(&Pulse { b: g.b, t: g.t, sin: g.sin, jb: g.jb, jt: g.jt, ..base }, [d1, d2, x[j + 2], ell], like)
            } else {
                tp - (r * r - bp * bp).max(0.0).sqrt() / g.sin.max(1e-9)
            };
            if best.is_none_or(|(bt, bk, _)| (touch && !bt) || (touch == bt && key < bk)) {
                best = Some((touch, key, p));
            }
        }
        match best {
            None => (base, None),
            Some((_, _, p)) => {
                let g = &self.geo[p];
                (Pulse { b: g.b, t: g.t, sin: g.sin, jb: g.jb, jt: g.jt, ..base }, Some(self.seg[p] as usize))
            }
        }
    }

    fn state(&self, owner: Option<usize>, i: usize, x: &DVector<f64>, st: &Setup) -> [f64; 4] {
        let ell = x[st.l0 + self.layer[i] as usize];
        match owner {
            None => [0.0, 0.0, (1e-6f64).ln(), ell],
            Some(k) => {
                let j = 3 * st.index[k] as usize;
                [x[j], x[j + 1], x[j + 2], ell]
            }
        }
    }

    /// The weighted log likelihood at `x`.
    fn loglik(&self, x: &DVector<f64>, st: &Setup, like: &JointParams) -> f64 {
        (0..self.n_records()).filter(|&i| st.weight[i] > 0.0).map(|i| {
            let (pu, own) = self.view(i, x, st, like);
            st.weight[i] * pair_value(&pu, self.state(own, i, x, st), like)
        }).sum()
    }

    /// The weighted log likelihood with its gradient and Hessian at `x` (dense), summed in a fixed order.
    fn derivs(&self, x: &DVector<f64>, st: &Setup, like: &JointParams, threads: usize) -> Result<(f64, DVector<f64>, DMatrix<f64>)> {
        let n = self.n_records();
        let parts = threads.max(1);
        let chunk = n.div_ceil(parts).max(1);
        let ranges: Vec<(usize, usize)> = (0..n).step_by(chunk).map(|s| (s, (s + chunk).min(n))).collect();
        let dim = st.dim;
        let res: Vec<(f64, DVector<f64>, DMatrix<f64>)> = pool(threads)?.install(|| ranges.par_iter().map(|&(s, e)| {
            let (mut f, mut g, mut h) = (0.0, DVector::zeros(dim), DMatrix::zeros(dim, dim));
            for i in s..e {
                let w = st.weight[i];
                if w <= 0.0 {
                    continue;
                }
                let (pu, own) = self.view(i, x, st, like);
                let (v, gi, hi) = pair_derivs(&pu, self.state(own, i, x, st), like);
                if !(v.is_finite() && gi.iter().all(|q| q.is_finite()) && hi.iter().flatten().all(|q| q.is_finite())) {
                    eprintln!("cluster evidence: record {i} not finite: {v} {gi:?} owner {own:?} state {:?} pulse {pu:?}", self.state(own, i, x, st));
                    continue;
                }
                f += w * v;
                let lv = st.l0 + self.layer[i] as usize;
                let idx: [Option<usize>; 4] = match own {
                    None => [None, None, None, Some(lv)],
                    Some(k) => {
                        let j = 3 * st.index[k] as usize;
                        [Some(j), Some(j + 1), Some(j + 2), Some(lv)]
                    }
                };
                for a in 0..4 {
                    let Some(ia) = idx[a] else { continue };
                    g[ia] += w * gi[a];
                    for b in 0..4 {
                        let Some(ib) = idx[b] else { continue };
                        h[(ia, ib)] += w * hi[a][b];
                    }
                }
            }
            (f, g, h)
        }).collect());
        let (mut f, mut g, mut h) = (0.0, DVector::zeros(dim), DMatrix::zeros(dim, dim));
        for (fi, gi, hi) in res {
            f += fi;
            g += gi;
            h += hi;
        }
        Ok((f, g, h))
    }

    fn start_x(&self, st: &Setup, o: &EvidenceOpts) -> DVector<f64> {
        let mut x = DVector::zeros(st.dim);
        for (j, &k) in st.active.iter().enumerate() {
            x[3 * j + 2] = self.r_prior[k].ln();
        }
        for l in 0..self.n_layers {
            x[st.l0 + l] = o.lam_prior.ln();
        }
        x
    }

    /// The fit and Laplace evidence of the hypothesis `active` (segments used) with `parent` (each segment's
    /// parent, -1 a root; a parent not active makes a root), from `x0` (None: the claims; see the module
    /// documentation).
    pub fn evidence(&self, active: &[bool], parent: &[i64], o: &EvidenceOpts, like: &JointParams, x0: Option<&[f64]>, threads: usize) -> Result<EvidenceFit> {
        let st = self.setup(active, parent, o, like)?;
        let mut x = match x0 {
            Some(v) if v.len() == st.dim => DVector::from_column_slice(v),
            Some(_) => return Err(Error::invalid("cluster evidence: x0 has the wrong length")),
            None => self.start_x(&st, o),
        };
        let used = st.weight.iter().filter(|&&w| w > 0.0).count() as u64;
        let mut evals = [0u64; 2];
        let obj = |x: &DVector<f64>, ev: &mut [u64; 2]| -> f64 {
            ev[0] += used;
            self.loglik(x, &st, like) + self.prior_value(x, &st)
        };
        let mut f = obj(&x, &mut evals);
        let mut mu = 1e-3;
        let (mut iters, mut converged) = (0, false);
        while iters < o.iters {
            iters += 1;
            let (_, gl, hl) = self.derivs(&x, &st, like, threads)?;
            evals[1] += used;
            let (_, ge, he) = self.exclusion(&x, &st);
            let g = gl + &st.prior.h - &st.prior.q * &x + ge;
            // the likelihood's curvature made positive semi-definite (saddle-free), the prior's added; Marquardt's
            // damping on its diagonal; steps capped at half a unit in log R and a radius in the centre
            // (where the curvature with the prior is already positive definite, as it mostly is, no eigendecomposition)
            let direct = -&hl + &st.prior.q - &he;
            let p = if direct.clone().cholesky().is_some() {
                direct
            } else {
                let ne = psd_eigen(-hl);
                let lc = ne.eigenvalues.map(|l| if l.is_finite() { l.max(0.0) } else { 0.0 });
                &ne.eigenvectors * DMatrix::from_diagonal(&lc) * ne.eigenvectors.transpose() + &st.prior.q - he
            };
            let mut stepped = false;
            let mut dec = 0.0;
            while mu < 1e10 {
                let mut a = p.clone();
                for i in 0..st.dim {
                    a[(i, i)] += mu * p[(i, i)].max(1e-12);
                }
                let Some(ch) = a.cholesky() else {
                    mu *= 10.0;
                    continue;
                };
                let mut d = ch.solve(&g);
                let mut cap: f64 = 1.0;
                for (j, &k) in st.active.iter().enumerate() {
                    let r = x[3 * j + 2].exp().max(self.r_prior[k]);
                    cap = cap.min(0.5 / d[3 * j + 2].abs().max(1e-300)).min(r / d[3 * j].hypot(d[3 * j + 1]).max(1e-300));
                }
                for l in st.l0..st.dim {
                    cap = cap.min(1.0 / d[l].abs().max(1e-300));
                }
                d *= cap;
                let xn = &x + &d;
                let fnew = obj(&xn, &mut evals);
                if fnew.is_finite() && fnew >= f - 1e-10 {
                    dec = fnew - f;
                    x = xn;
                    f = fnew;
                    mu = (mu / 4.0).max(1e-9);
                    stepped = true;
                    break;
                }
                mu *= 8.0;
            }
            if !stepped {
                break; // nothing improves (ownership's non-smooth edges): not converged
            }
            if dec < o.tol {
                converged = true;
                break;
            }
        }
        // Laplace at the mode: the likelihood's curvature clipped to PSD, plus the prior's precision
        let (fl, _, hl) = self.derivs(&x, &st, like, threads)?;
        evals[1] += used;
        let lp = self.prior_value(&x, &st);
        let hl_keep = hl.clone();
        let ne = psd_eigen(-hl);
        let tot: f64 = ne.eigenvalues.iter().map(|l| l.abs()).sum::<f64>().max(1e-300);
        let neg: f64 = ne.eigenvalues.iter().filter(|l| **l < 0.0).map(|l| -l).sum();
        let lc = ne.eigenvalues.map(|l| if l.is_finite() { l.max(0.0) } else { 0.0 });
        // (the non-overlap hinge's curvature counted with the likelihood's)
        let ilik = &ne.eigenvectors * DMatrix::from_diagonal(&lc) * ne.eigenvectors.transpose() - self.exclusion(&x, &st).2;
        let pmat = &ilik + &st.prior.q;
        let pmat = 0.5 * (&pmat + pmat.transpose());
        if pmat.iter().any(|v| !v.is_finite()) {
            let bad = |m: &DMatrix<f64>| m.iter().filter(|v| !v.is_finite()).count();
            return Err(Error::invalid(format!("cluster evidence: the posterior precision isn't finite (likelihood {fl}, prior {lp}; non-finite: hl {}, q {}, excl {}, x {}; iters {iters}; |hl| max {:e}; eigen non-finite {} {}; x range {:?})", hl_keep.amax(), ne.eigenvalues.iter().filter(|v| !v.is_finite()).count(), ne.eigenvectors.iter().filter(|v| !v.is_finite()).count(), bad(&hl_keep), bad(&st.prior.q), bad(&self.exclusion(&x, &st).2), x.iter().filter(|v| !v.is_finite()).count(), (x.min(), x.max()))));
        }
        let pmat = match pmat.clone().cholesky() {
            Some(_) => pmat,
            None => {
                // round-off on a nearly singular direction: eigenvalues floored at 1e-12 of the largest
                let e = SymmetricEigen::new(pmat);
                let top = e.eigenvalues.iter().copied().fold(0.0, f64::max);
                &e.eigenvectors * DMatrix::from_diagonal(&e.eigenvalues.map(|l| l.max(1e-12 * top))) * e.eigenvectors.transpose()
            }
        };
        let ch = pmat.clone().cholesky().ok_or_else(|| Error::invalid("cluster evidence: the posterior precision isn't positive definite"))?;
        let logdet: f64 = 2.0 * ch.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
        let cov = ch.inverse();
        let dim = st.dim;
        let logz = fl + lp + 0.5 * dim as f64 * (2.0 * std::f64::consts::PI).ln() - 0.5 * logdet;
        // per record at the mode
        let n = self.n_records();
        let (mut owner, mut ll, mut wood) = (vec![-1i32; n], vec![0.0; n], vec![f64::NAN; n]);
        for i in 0..n {
            let (pu, own) = self.view(i, &x, &st, like);
            let s = self.state(own, i, &x, &st);
            owner[i] = own.map_or(-1, |k| k as i32);
            ll[i] = st.weight[i] * pair_value(&pu, s, like);
            wood[i] = if own.is_some() { pair_wood_share(&pu, s, like) } else if self.stop[i] { 0.0 } else { f64::NAN };
        }
        let mut precision = Vec::with_capacity(2 * dim * dim);
        precision.extend(st.prior.q.transpose().iter().copied());
        precision.extend(ilik.transpose().iter().copied());
        Ok(EvidenceFit { logz, loglik: fl, logprior: lp, logdet, dim, iters, converged, clipped: neg / tot, x: x.iter().copied().collect(), sd: (0..dim).map(|i| cov[(i, i)].max(0.0).sqrt()).collect(),
            index: st.index, owner, ll, wood, evals, precision })
    }

    /// The log posterior (unnormalised: weighted log likelihood plus log prior) at each row of `xs` (`n x dim`).
    pub fn objective(&self, active: &[bool], parent: &[i64], o: &EvidenceOpts, like: &JointParams, xs: &[Vec<f64>], threads: usize) -> Result<Vec<f64>> {
        let st = self.setup(active, parent, o, like)?;
        if xs.iter().any(|v| v.len() != st.dim) {
            return Err(Error::invalid("cluster evidence: a state has the wrong length"));
        }
        pool(threads)?.install(|| Ok(xs.par_iter().map(|v| {
            let x = DVector::from_column_slice(v);
            self.loglik(&x, &st, like) + self.prior_value(&x, &st)
        }).collect()))
    }

    /// Importance sampling's log Z for the hypothesis, from `n` draws of a multivariate t (`df` degrees of freedom)
    /// about the Laplace fit `fit`, its scale `scale` times the Laplace covariance; `seed` seeds the draws.
    #[allow(clippy::too_many_arguments)]
    pub fn importance(&self, active: &[bool], parent: &[i64], o: &EvidenceOpts, like: &JointParams, fit: &EvidenceFit, n: usize, df: f64, scale: f64, seed: u64, threads: usize) -> Result<Importance> {
        let st = self.setup(active, parent, o, like)?;
        let dim = st.dim;
        if fit.x.len() != dim || fit.precision.len() != 2 * dim * dim {
            return Err(Error::invalid("cluster evidence: the fit is not this hypothesis'"));
        }
        let q = DMatrix::from_row_slice(dim, dim, &fit.precision[..dim * dim]);
        let il = DMatrix::from_row_slice(dim, dim, &fit.precision[dim * dim..]);
        let p = (&q + &il) / (scale * scale);
        let ch = p.clone().cholesky().ok_or_else(|| Error::invalid("importance: precision not positive definite"))?;
        // draws: x = mu + L^-T z sqrt(df / u), with P = L L'
        let lt = ch.l().transpose();
        let logdet_p: f64 = 2.0 * ch.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
        let mu = DVector::from_column_slice(&fit.x);
        let k = df.round().max(1.0) as usize;
        let lq_const = libm::lgamma(0.5 * (df + dim as f64)) - libm::lgamma(0.5 * df) - 0.5 * dim as f64 * (df * std::f64::consts::PI).ln() + 0.5 * logdet_p;
        let draws: Vec<(DVector<f64>, f64)> = (0..n).map(|s| {
            let mut rng = Rng::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (s as u64).wrapping_add(1));
            let z = DVector::from_iterator(dim, (0..dim).map(|_| rng.normal()));
            let u: f64 = (0..k).map(|_| rng.normal().powi(2)).sum();
            let y = lt.clone().solve_upper_triangular(&z).unwrap_or_else(|| z.clone());
            let x = &mu + y * (df / u).sqrt();
            let dx = &x - &mu;
            let qf = dx.dot(&(&p * &dx));
            (x, lq_const - 0.5 * (df + dim as f64) * (1.0 + qf / df).ln())
        }).collect();
        let lw: Vec<f64> = pool(threads)?.install(|| draws.par_iter().map(|(x, lq)| self.loglik(x, &st, like) + self.prior_value(x, &st) - lq).collect());
        let mx = lw.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = lw.iter().map(|v| if v.is_finite() { (v - mx).exp() } else { 0.0 }).collect();
        let sw: f64 = w.iter().sum();
        let sw2: f64 = w.iter().map(|v| v * v).sum();
        let mean = sw / n as f64;
        let var = w.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
        Ok(Importance { logz: mx + mean.ln(), se: (var / n as f64).sqrt() / mean, ess: sw * sw / sw2, max_share: w.iter().copied().fold(0.0, f64::max) / sw, log_w: lw })
    }
}

/// The symmetric eigendecomposition of `m`, robust to all-zero rows (a segment no record reaches), on which
/// nalgebra's tridiagonalisation can return NaN: computed on `m + eps I` and the eigenvalues shifted back.
fn psd_eigen(m: DMatrix<f64>) -> SymmetricEigen<f64, nalgebra::Dyn> {
    let n = m.nrows();
    let eps = 1e-9 * m.amax().max(1.0);
    let mut e = SymmetricEigen::new(m + DMatrix::identity(n, n) * eps);
    e.eigenvalues.iter_mut().for_each(|l| *l -= eps);
    e
}

/// A record's hash for thinning (splitmix64 of its scan, pulse and a seed).
fn hash(scan: u32, pulse: u32, seed: u64) -> u64 {
    let mut z = ((scan as u64) << 32 | pulse as u64) ^ seed.wrapping_mul(0xd1b5_4a32_d192_ed03);
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A small generator for the draws (xorshift64*, Box-Muller).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(hash(seed as u32, (seed >> 32) as u32, 7) | 1)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prior_is_a_normalised_gaussian() {
        // two factors on two variables: x0 ~ N(1, 0.5^2), x1 - x0 ~ N(0, 0.2^2): integrates to 1 (by the Gaussian
        // integral of its canonical form)
        let p = Prior::of(&[Factor { a: vec![(0, 1.0)], m: 1.0, s: 0.5 }, Factor { a: vec![(1, 1.0), (0, -1.0)], m: 0.0, s: 0.2 }], 2);
        let ch = p.q.clone().cholesky().unwrap();
        let mean = ch.solve(&p.h);
        let logdet: f64 = 2.0 * ch.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
        let log_int = p.value(&mean) + (2.0 * std::f64::consts::PI).ln() - 0.5 * logdet;
        assert!(log_int.abs() < 1e-12, "{log_int}");
        assert!((mean[0] - 1.0).abs() < 1e-12 && (mean[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_scanned_stem_beats_no_wood_and_its_laplace_matches_sampling() {
        // the evidence tests' fan: a 5 cm stem at 5 m; the claim 1 cm off and 8 cm
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let like = JointParams { tau: 0.17, edge: 0.25, hide: 0.0, ..JointParams::default() };
        let (a0, b0) = ([a[0] + 0.01, a[1], -0.15], [b[0] + 0.01, b[1], 0.15]);
        let t = cluster_table(&[&scan], &[a0], &[b0], &[0.08], -1.0, f64::INFINITY, 0.0, 0.0, &like, 1).unwrap();
        assert!(t.n_records() > 50);
        let o = EvidenceOpts { oversample: false, ..EvidenceOpts::default() };
        let stem = t.evidence(&[true], &[-1], &o, &like, None, 1).unwrap();
        let none = t.evidence(&[false], &[-1], &o, &like, None, 1).unwrap();
        assert!(stem.converged, "{stem:?}");
        let r = stem.x[2].exp();
        assert!((r - 0.05).abs() < 0.005, "R {r}");
        assert!(stem.logz > none.logz + 20.0, "{} against {}", stem.logz, none.logz);
        let is = t.importance(&[true], &[-1], &o, &like, &stem, 4000, 5.0, 1.2, 3, 1).unwrap();
        assert!((is.logz - stem.logz).abs() < 1.0, "IS {} ({}) against Laplace {}", is.logz, is.se, stem.logz);
    }
}
