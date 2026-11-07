// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The joint wood, foliage and ray model of one tree, end to end: its rows
//! (a QSM's cylinders, parent before child) and its neighbours' cylinders in,
//! a fitted row per stem out (3dtrees research 74 §20).
//!
//! - The tube a row's pairs are taken in is decided once, from its prior
//!   radius ([`window`]).
//! - **Split.** A row whose pulses aimed at its core partly pass is a clump
//!   (§18-19): its stops are put in its cross-section, circles are found
//!   there (a greedy Hough transform over front-face arcs), and each circle
//!   that is solid (the pulses crossing its core stop at its front, as on
//!   wood, not through it, as in a cluster of phyllodes) becomes a child row:
//!   parallel to the row, through the circle's centre, with its radius as the
//!   prior. A row with no solid circle stays as it is.
//! - **Ownership and fit.** The rows' and the neighbours' pairs share one
//!   table. Each stop seen by several rows belongs to the one whose surface
//!   it lies nearest; the neighbours, held at their claims, keep theirs and
//!   aren't fitted. Ownership from the starting state, then the fit
//!   ([`joint_fit`]), `sweeps` times.
//! - **Existence.** Each row's own log likelihood at its fit against its best
//!   without wood (`R` -> 0, foliage refitted), less a BIC price for the
//!   three wood unknowns, `sigma(delta - 1.5 ln n_pairs)`, times the chance
//!   wood of the fitted radius can make an echo in the row's typical
//!   footprint at all: a line thinner than that among a clump of phyllodes
//!   explains their clustered stops better than Poisson foliage does, but
//!   isn't wood.
//! - **Outputs** in world coordinates, one per child row, with the RCT row it
//!   came from (`src`).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::qsm::evidence::ScanIndex;
use crate::qsm::joint::{capacity, frames, joint_loglik, joint_table, surface_residual, window, JointPairs, JointParams, Owner, Pulse};
use crate::qsm::joint_fit::{joint_fit, profile_grid, FitOptions, PriorParams, Rows};
use crate::Point;

/// The settings of [`joint_tree`]: the likelihood's, the prior's, the fit's,
/// the split's and the profile's.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeParams {
    pub like: JointParams,
    pub prior: PriorParams,
    /// Newton iterations per continuation stage at most, and the stopping
    /// tolerance (see [`FitOptions`]).
    pub iters: usize,
    pub tol: f64,
    /// Rounds of ownership and fitting.
    pub sweeps: usize,
    /// A row is tried as a clump when this share of its complete pulses aimed
    /// within half its prior radius of its axis pass (a solid stem's is ~0),
    /// and its prior radius is at least `split_min_r` (m).
    pub split_share: f64,
    pub split_min_r: f64,
    /// A circle is a stem when at most this share of the pulses crossing its
    /// core went through it, and at least `split_min_votes` stops lie on it.
    pub split_through: f64,
    pub split_min_votes: usize,
    /// A row's interval is where its profile is within `drop` of its best.
    pub drop: f64,
}

impl TreeParams {
    /// Bumped whenever the fit changes for the same settings.
    pub const VERSION: u32 = 1;
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams { like: JointParams::default(), prior: PriorParams::default(), iters: 100, tol: 1e-8, sweeps: 2, split_share: 0.05, split_min_r: 0.02, split_through: 0.1, split_min_votes: 60, drop: 2.0 }
    }
}

/// The continuation's stages before the physical surface kernel.
pub const SCHEDULE: [f64; 4] = [0.3, 0.15, 0.08, 0.04];

/// Why a row's radius is what it is.
pub mod reason {
    /// The rays bound it and say it is wood.
    pub const MEASURED: u8 = 1;
    /// The rays don't bound it (its profile never falls `drop` below its best): the prior's.
    pub const FLAT: u8 = 2;
    /// The rays say there is no wood (`p_wood < 0.5`): foliage explains them better.
    pub const NO_WOOD: u8 = 3;
}

/// The fit of one tree, a row per stem (an RCT row, or a stem split from it).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFit {
    /// The row of the input each row came from, and each row's parent here (-1 at a base).
    pub src: Vec<u32>,
    pub parent: Vec<i64>,
    /// The fitted axis (world coordinates) and radius, with the sd of log R.
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r: Vec<f64>,
    pub sd_log_r: Vec<f64>,
    /// The radius interval from the row's own profile, and whether it is flat.
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub flat: Vec<bool>,
    /// Foliage (1/m), existence, the reason, and the data shares.
    pub lam: Vec<f64>,
    /// The log odds of wood (with against without, less the BIC price), and the probability.
    pub log_odds: Vec<f64>,
    pub p_wood: Vec<f64>,
    pub reason: Vec<u8>,
    pub rho_local: Vec<f64>,
    pub rho_pooled: Vec<f64>,
    /// The row's pairs after ownership.
    pub n_pairs: Vec<u32>,
    /// Each row's likelihood message at the fit (state `[d1, d2, log R, log lambda]`, offsets along `e1, e2`).
    pub grad: Vec<[f64; 4]>,
    pub hess: Vec<[[f64; 4]; 4]>,
    pub e1: Vec<Point>,
    pub e2: Vec<Point>,
    /// The share of the tree's volume the rays don't determine (`sum v (1 - rho_local) / sum v`).
    pub u: f64,
    pub objective: f64,
    pub converged: bool,
}

/// The rows of a tree: cylinders `a -> b`, prior radii, parents (before
/// their children; -1 at a base).
struct Stems {
    a: Vec<Point>,
    b: Vec<Point>,
    r: Vec<f64>,
    parent: Vec<i64>,
    src: Vec<u32>,
}

/// Fits the joint model to one tree: `rows` (with `r_max` ignored: each
/// row's tube is [`window`] of its prior radius), against `scans`, with the
/// neighbours' cylinders `ctx_a -> ctx_b` of radius `ctx_r` held fixed.
/// `split`: the rows to try splitting (default: by `params.split_share`).
#[allow(clippy::too_many_arguments)]
pub fn joint_tree(scans: &[&ScanIndex], rows: &Rows, ctx_a: &[Point], ctx_b: &[Point], ctx_r: &[f64], params: &TreeParams, split: Option<&[bool]>, threads: usize) -> Result<TreeFit> {
    let n = rows.a.len();
    if [rows.b.len(), rows.r_prior.len(), rows.parent.len()].iter().any(|&l| l != n) || ctx_b.len() != ctx_a.len() || ctx_r.len() != ctx_a.len() {
        return Err(Error::invalid("joint tree: a, b, r_prior and parent (and the context's a, b and r) must have one entry per row"));
    }
    if split.is_some_and(|s| s.len() != n) {
        return Err(Error::invalid("joint tree: split must have one entry per row"));
    }
    if !rows.r_prior.iter().chain(ctx_r).all(|r| r.is_finite() && *r > 0.0) {
        return Err(Error::invalid("joint tree: radii must be positive"));
    }
    params.prior.check()?;
    let p = &params.like;
    let lam0 = params.prior.lam_prior.ln();
    let rows0 = Stems { a: rows.a.to_vec(), b: rows.b.to_vec(), r: rows.r_prior.to_vec(), parent: rows.parent.to_vec(), src: (0..n as u32).collect() };

    // the split, from the claims' own pairs
    let claim = table(scans, &rows0, ctx_a, ctx_b, ctx_r, threads)?;
    let try_split: Vec<bool> = match split {
        Some(s) => s.to_vec(),
        None => (0..n).map(|k| rows.r_prior[k] >= params.split_min_r && core_pass(&claim, k, rows.r_prior[k]) >= params.split_share).collect(),
    };
    let stems = split_rows(&rows0, &claim, &try_split, params);

    // ownership and the fit, `sweeps` times, starting from the claims
    let t = table(scans, &stems, ctx_a, ctx_b, ctx_r, threads)?;
    let m = stems.a.len();
    let ctx_state: Vec<[f64; 4]> = ctx_r.iter().map(|r| [0.0, 0.0, r.ln(), lam0]).collect();
    let mut theta: Vec<[f64; 4]> = stems.r.iter().map(|r| [0.0, 0.0, r.ln(), lam0]).collect();
    let r_max: Vec<f64> = stems.r.iter().map(|&r| window(r)).collect();
    let fit_rows = Rows { a: &stems.a, b: &stems.b, r_prior: &stems.r, r_max: &r_max, parent: &stems.parent };
    let mut fit: Option<crate::qsm::joint_fit::Fit> = None;
    let mut mine = JointPairs::default();
    let centre = profile_grid().iter().position(|&d| d == 0.0).expect("the profile holds the fit");
    let mut restarted = false;
    let mut sweep = 0;
    loop {
        if sweep >= params.sweeps.max(1) {
            // a row whose own profile beats its fit by more than `drop` sits on a poorer optimum: restart it from the
            // profile's best, once (multimodality, research 74 §5)
            let Some(f) = fit.as_ref().filter(|_| !restarted) else { break };
            let mut moved = false;
            let mut start = theta.clone();
            for (k, pr) in f.profile.iter().enumerate() {
                let (best, l) = pr.iter().enumerate().fold((centre, pr[centre]), |b, (i, &l)| if l > b.1 { (i, l) } else { b });
                if l > pr[centre] + params.drop {
                    start[k][2] += profile_grid()[best];
                    moved = true;
                }
            }
            if !moved {
                break;
            }
            theta = start;
            restarted = true;
        }
        let state: Vec<[f64; 4]> = theta.iter().chain(&ctx_state).copied().collect();
        let own = owners(&t, &state, p);
        mine = JointPairs::default();
        for i in (0..t.pulses.len()).filter(|&i| own[i] && (t.row[i] as usize) < m) {
            mine.pulses.push(t.pulses[i]);
            mine.row.push(t.row[i]);
            mine.scan.push(t.scan[i]);
        }
        let opts = FitOptions { schedule: if sweep == 0 { SCHEDULE.to_vec() } else { Vec::new() }, iters: params.iters, tol: params.tol, threads };
        let f = joint_fit(&mine.pulses, &mine.row, &mine.scan, &fit_rows, scans.len(), p, &params.prior, &opts, Some(&theta))?;
        theta = f.theta.clone();
        fit = Some(f);
        sweep += 1;
    }
    let fit = fit.expect("at least one sweep");

    // existence: each row's own pairs at its fit (the profile's centre: with its scans' offsets), against without wood
    // (offsets don't matter then; foliage refitted on a grid)
    let owners_plain: Vec<Owner> = mine.row.iter().map(|&row| Owner { row, nu: u32::MAX }).collect();
    let row_ll = |th: &[[f64; 4]]| -> Result<Vec<f64>> {
        let ll = joint_loglik(&mine.pulses, &owners_plain, th, &[], p, threads)?;
        let mut out = vec![0.0; m];
        for (o, l) in owners_plain.iter().zip(ll) {
            out[o.row as usize] += l;
        }
        Ok(out)
    };
    let with: Vec<f64> = fit.profile.iter().map(|pr| pr[centre]).collect();
    // without: each row's foliage at its own best, by golden section on log lambda within +-4 of its fit (every
    // row at once, a pass per step); a coarse grid would cost the foliage-only fit tens of log units on a big row
    let without = {
        const G: f64 = 0.618_033_988_749_895;
        let none = (1e-4f64).ln();
        let at = |ell: &[f64]| row_ll(&ell.iter().map(|&e| [0.0, 0.0, none, e]).collect::<Vec<_>>());
        let (mut lo, mut hi): (Vec<f64>, Vec<f64>) = theta.iter().map(|t| (t[3] - 4.0, t[3] + 4.0)).unzip();
        let mut c: Vec<f64> = (0..m).map(|k| hi[k] - G * (hi[k] - lo[k])).collect();
        let mut d: Vec<f64> = (0..m).map(|k| lo[k] + G * (hi[k] - lo[k])).collect();
        let (mut fc, mut fd) = (at(&c)?, at(&d)?);
        for _ in 0..30 {
            let mut probe = vec![0.0; m];
            for k in 0..m {
                if fc[k] > fd[k] {
                    (hi[k], d[k], fd[k]) = (d[k], c[k], fc[k]);
                    c[k] = hi[k] - G * (hi[k] - lo[k]);
                    probe[k] = c[k];
                } else {
                    (lo[k], c[k], fc[k]) = (c[k], d[k], fd[k]);
                    d[k] = lo[k] + G * (hi[k] - lo[k]);
                    probe[k] = d[k];
                }
            }
            let f = at(&probe)?;
            for k in 0..m {
                if probe[k] == c[k] { fc[k] = f[k] } else { fd[k] = f[k] }
            }
        }
        (0..m).map(|k| fc[k].max(fd[k])).collect::<Vec<f64>>()
    };
    let mut n_pairs = vec![0u32; m];
    let mut widths: Vec<Vec<f64>> = vec![Vec::new(); m];
    for (pu, &r) in mine.pulses.iter().zip(&mine.row) {
        n_pairs[r as usize] += 1;
        widths[r as usize].push(p.exit + p.divergence * pu.t);
    }
    // each row's typical footprint: wood thinner than it can give an echo needs can't be wood, however much better a
    // thin line among its stops explains them (clumped foliage, which the Poisson foliage term can't; research 74 §20)
    let w_row: Vec<f64> = widths.iter_mut().map(|w| {
        if w.is_empty() { return p.exit }
        let mid = w.len() / 2;
        *w.select_nth_unstable_by(mid, f64::total_cmp).1
    }).collect();

    // the outputs
    let fr = frames(&stems.a, &stems.b);
    let grid = profile_grid();
    let mut out = TreeFit { src: stems.src.clone(), parent: stems.parent.clone(), a: Vec::new(), b: Vec::new(), r: Vec::new(), sd_log_r: Vec::new(), lo: Vec::new(), hi: Vec::new(), flat: Vec::new(), lam: Vec::new(), log_odds: Vec::new(), p_wood: Vec::new(), reason: Vec::new(), rho_local: fit.rho_local.clone(), rho_pooled: fit.rho_pooled.clone(), n_pairs: n_pairs.clone(), grad: fit.grad.clone(), hess: fit.hess.clone(), e1: Vec::new(), e2: Vec::new(), u: 0.0, objective: fit.objective, converged: fit.converged };
    let (mut v_all, mut v_open) = (0.0, 0.0);
    for k in 0..m {
        let (_, e1, e2, len) = fr[k];
        let [d1, d2, eta, ell] = theta[k];
        let off: Point = std::array::from_fn(|i| d1 * e1[i] + d2 * e2[i]);
        out.a.push(std::array::from_fn(|i| stems.a[k][i] + off[i]));
        out.b.push(std::array::from_fn(|i| stems.b[k][i] + off[i]));
        let r = eta.exp();
        out.r.push(r);
        out.sd_log_r.push(fit.sd[k][2]);
        out.lam.push(ell.exp());
        out.e1.push(e1);
        out.e2.push(e2);
        // the profile's bracket: the grid points either side of where it is within `drop` of its best
        let prof = &fit.profile[k];
        let best = prof.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ok: Vec<bool> = prof.iter().map(|&l| l >= best - params.drop).collect();
        let flat = n_pairs[k] == 0 || ok.iter().all(|&o| o);
        let first = ok.iter().position(|&o| o).unwrap_or(0);
        let last = ok.iter().rposition(|&o| o).unwrap_or(grid.len() - 1);
        out.lo.push((eta + grid[first.saturating_sub(1)]).exp());
        out.hi.push((eta + grid[(last + 1).min(grid.len() - 1)]).exp());
        out.flat.push(flat);
        let odds = if n_pairs[k] == 0 { 0.0 } else { with[k] - without[k] - 1.5 * (n_pairs[k] as f64).ln() };
        let pw = capacity(r, w_row[k], p) / (1.0 + (-odds).exp());
        out.log_odds.push(odds);
        out.p_wood.push(pw);
        out.reason.push(if pw < 0.5 && !flat { reason::NO_WOOD } else if flat { reason::FLAT } else { reason::MEASURED });
        let v = std::f64::consts::PI * r * r * len;
        v_all += v;
        v_open += v * (1.0 - fit.rho_local[k]);
    }
    out.u = if v_all > 0.0 { v_open / v_all } else { 1.0 };
    Ok(out)
}

/// The pair table of `stems` then the context's cylinders, each in its [`window`].
fn table(scans: &[&ScanIndex], stems: &Stems, ctx_a: &[Point], ctx_b: &[Point], ctx_r: &[f64], threads: usize) -> Result<JointPairs> {
    let a: Vec<Point> = stems.a.iter().chain(ctx_a).copied().collect();
    let b: Vec<Point> = stems.b.iter().chain(ctx_b).copied().collect();
    let w: Vec<f64> = stems.r.iter().chain(ctx_r).map(|&r| window(r)).collect();
    joint_table(scans, &a, &b, &w, threads)
}

/// The share of row `k`'s complete pulses (no deleted echo in reach) aimed
/// within half of `r` of its axis that pass.
fn core_pass(t: &JointPairs, k: usize, r: f64) -> f64 {
    let (mut n, mut pass) = (0usize, 0usize);
    for (pu, &row) in t.pulses.iter().zip(&t.row) {
        if row as usize == k && !pu.deleted && pu.b.abs() < 0.5 * r {
            n += 1;
            pass += usize::from(!pu.stop);
        }
    }
    if n == 0 { 1.0 } else { pass as f64 / n as f64 }
}

/// Which pairs each row keeps: a stop seen by several rows stays with the one
/// whose surface (at `state`, per row of `t`) it lies nearest; one no
/// surface explains stays with all (it is foliage to every one of them).
fn owners(t: &JointPairs, state: &[[f64; 4]], p: &JointParams) -> Vec<bool> {
    let resid: Vec<f64> = t.pulses.iter().zip(&t.row).map(|(pu, &k)| if pu.stop { surface_residual(pu, state[k as usize], p) } else { f64::INFINITY }).collect();
    let mut best: HashMap<(u32, u32), (f64, u32)> = HashMap::new();
    for i in 0..t.pulses.len() {
        let e = best.entry((t.scan[i], t.pulse[i])).or_insert((f64::INFINITY, 0));
        e.0 = e.0.min(resid[i]);
        e.1 += 1;
    }
    (0..t.pulses.len()).map(|i| {
        let (b, count) = best[&(t.scan[i], t.pulse[i])];
        !t.pulses[i].stop || count == 1 || resid[i] <= b || !b.is_finite()
    }).collect()
}

/// `rows` with each row in `try_split` replaced by the solid circles its stops
/// show (left as it is where there are none). A stem continues the parent
/// row's stem whose end is nearest its start.
fn split_rows(rows: &Stems, t: &JointPairs, try_split: &[bool], params: &TreeParams) -> Stems {
    let fr = frames(&rows.a, &rows.b);
    let mut out = Stems { a: Vec::new(), b: Vec::new(), r: Vec::new(), parent: Vec::new(), src: Vec::new() };
    let mut new: Vec<Vec<usize>> = Vec::with_capacity(rows.a.len());
    for k in 0..rows.a.len() {
        let (_, e1, e2, _) = fr[k];
        let mut parts = vec![(rows.a[k], rows.b[k], rows.r[k])];
        if try_split[k] {
            let mine: Vec<&Pulse> = t.pulses.iter().zip(&t.row).filter(|&(_, &r)| r as usize == k).map(|(pu, _)| pu).collect();
            let solid: Vec<(f64, f64, f64)> = circles(&mine, rows.r[k], params.split_min_votes).into_iter().filter(|&c| opacity(&mine, c) <= params.split_through).collect();
            if !solid.is_empty() {
                parts = solid.iter().map(|&(cx, cy, r)| {
                    let off: Point = std::array::from_fn(|i| cx * e1[i] + cy * e2[i]);
                    (std::array::from_fn(|i| rows.a[k][i] + off[i]), std::array::from_fn(|i| rows.b[k][i] + off[i]), r)
                }).collect();
            }
        }
        let mut here = Vec::new();
        for (a, b, r) in parts {
            let q = rows.parent[k];
            let parent = if q < 0 { -1 } else {
                let d = |i: &usize| crate::transform::norm(&crate::transform::sub(&out.b[*i], &a));
                new[q as usize].iter().min_by(|x, y| d(x).total_cmp(&d(y))).map_or(-1, |&i| i as i64)
            };
            here.push(out.a.len());
            out.a.push(a);
            out.b.push(b);
            out.r.push(r);
            out.parent.push(parent);
            out.src.push(rows.src[k]);
        }
        new.push(here);
    }
    out
}

/// A stop's place in its row's cross-section (frame `e1, e2`, from the axis)
/// and the unit direction its pulse travelled across the axis: its closest
/// approach lies `b` along the normal (`jb`), the stop `(x - t) sin` further
/// along that direction (`jt sin`).
fn cross_section(pu: &Pulse) -> ([f64; 2], [f64; 2]) {
    let v = [pu.jt[0] * pu.sin, pu.jt[1] * pu.sin];
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-12);
    let v = [v[0] / l, v[1] / l];
    let s = (pu.x - pu.t) * pu.sin;
    ([pu.b * pu.jb[0] + s * v[0], pu.b * pu.jb[1] + s * v[1]], v)
}

/// Circles in the stops of one row's pairs, greedily: the circle (centre,
/// radius up to `r_max`) most stops lie on (within 6 mm) with its centre
/// behind them (a stop hit a front face), then again among the stops left,
/// until the best has fewer than `min_votes` stops or a quarter of the
/// strongest's, at most six. Returns `(cx, cy, r)` in the row's frame.
fn circles(pairs: &[&Pulse], r_max: f64, min_votes: usize) -> Vec<(f64, f64, f64)> {
    const TOL: f64 = 0.006;
    const CELL: f64 = 0.004;
    const ANGLES: usize = 48;
    const SHARE: f64 = 0.25;
    const AT_MOST: usize = 6;
    let pts: Vec<([f64; 2], [f64; 2])> = pairs.iter().filter(|pu| pu.stop).map(|pu| cross_section(pu)).collect();
    if pts.len() < min_votes {
        return Vec::new();
    }
    let radii: Vec<f64> = (0..).map(|i| 0.01 + 0.005 * i as f64).take_while(|&r| r <= r_max + 1e-9).collect();
    let dirs: Vec<[f64; 2]> = (0..ANGLES).map(|i| {
        let a = 2.0 * std::f64::consts::PI * i as f64 / ANGLES as f64;
        [a.cos(), a.sin()]
    }).collect();
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for (y, _) in &pts {
        for j in 0..2 {
            lo[j] = lo[j].min(y[j]);
            hi[j] = hi[j].max(y[j]);
        }
    }
    let lo = [lo[0] - r_max - CELL, lo[1] - r_max - CELL];
    let (nx, ny) = (((hi[0] - lo[0] + r_max + CELL) / CELL).ceil() as usize + 2, ((hi[1] - lo[1] + r_max + CELL) / CELL).ceil() as usize + 2);
    let on = |c: [f64; 2], r: f64, y: [f64; 2], v: [f64; 2]| {
        let (dx, dy) = (c[0] - y[0], c[1] - y[1]);
        ((dx * dx + dy * dy).sqrt() - r).abs() < TOL && dx * v[0] + dy * v[1] > 0.0
    };
    let mut left = vec![true; pts.len()];
    let mut found: Vec<(f64, f64, f64, usize)> = Vec::new();
    while left.iter().filter(|&&l| l).count() >= min_votes && found.len() < AT_MOST {
        let mut acc = vec![0f32; radii.len() * nx * ny];
        for (i, &r) in radii.iter().enumerate() {
            let base = i * nx * ny;
            for ((y, v), _) in pts.iter().zip(&left).filter(|(_, &l)| l) {
                for d in &dirs {
                    if d[0] * v[0] + d[1] * v[1] <= 0.2 {
                        continue; // only centres beyond the stop
                    }
                    let (cx, cy) = (((y[0] + r * d[0] - lo[0]) / CELL) as usize, ((y[1] + r * d[1] - lo[1]) / CELL) as usize);
                    if cx < nx && cy < ny {
                        acc[base + cx * ny + cy] += 1.0;
                    }
                }
            }
        }
        // a stop votes for several neighbouring cells of one centre: smooth over 3x3 cells; then the top peaks are
        // scored by the stops actually on each circle with it behind them, and the best is taken
        let mut peaks: Vec<(f32, usize)> = Vec::new();
        for i in 0..radii.len() {
            for x in 1..nx - 1 {
                for y in 1..ny - 1 {
                    let mut s = 0.0;
                    for dx in 0..3 {
                        for dy in 0..3 {
                            s += acc[i * nx * ny + (x + dx - 1) * ny + (y + dy - 1)];
                        }
                    }
                    if s > 0.0 {
                        peaks.push((s, (i * nx + x) * ny + y));
                    }
                }
            }
        }
        let top = peaks.len().min(30);
        if top == 0 {
            break;
        }
        peaks.select_nth_unstable_by(top - 1, |a, b| b.0.total_cmp(&a.0));
        let mut best: Option<(usize, [f64; 2], f64)> = None;
        for &(_, flat) in &peaks[..top] {
            let (i, rest) = (flat / (nx * ny), flat % (nx * ny));
            let c = [lo[0] + ((rest / ny) as f64 + 0.5) * CELL, lo[1] + ((rest % ny) as f64 + 0.5) * CELL];
            let votes = pts.iter().zip(&left).filter(|&(&(y, v), &l)| l && on(c, radii[i], y, v)).count();
            if best.is_none_or(|b| votes > b.0) {
                best = Some((votes, c, radii[i]));
            }
        }
        let Some((votes, c, r)) = best else { break };
        let strongest = found.iter().map(|f| f.3).max().unwrap_or(0);
        if votes < min_votes || (votes as f64) < SHARE * strongest as f64 {
            break;
        }
        found.push((c[0], c[1], r, votes));
        for ((y, _), l) in pts.iter().zip(left.iter_mut()) {
            if ((y[0] - c[0]).powi(2) + (y[1] - c[1]).powi(2)).sqrt() < r + TOL {
                *l = false; // its stops, and any inside it, are explained
            }
        }
    }
    found.into_iter().map(|(x, y, r, _)| (x, y, r)).collect()
}

/// Of one row's complete pulses whose lines cross a circle's core (within
/// half its radius of its centre) and that reached its front: the share that
/// went through it (no stop within 2 cm of its front). Solid wood stops them
/// all (research 74 §19: Alice Mulga's clean stems 0-3 %, clusters of
/// phyllodes 15-85 %).
fn opacity(pairs: &[&Pulse], (cx, cy, r): (f64, f64, f64)) -> f64 {
    let (mut core, mut through) = (0usize, 0usize);
    for pu in pairs.iter().filter(|pu| !pu.deleted) {
        let (y0, v) = cross_section(&Pulse { x: pu.t, ..**pu }); // the line's closest approach
        let rel = [cx - y0[0], cy - y0[1]];
        let along = rel[0] * v[0] + rel[1] * v[1];
        let miss = (rel[0] * v[1] - rel[1] * v[0]).abs();
        if miss >= 0.5 * r {
            continue;
        }
        let front = along - (r * r - miss * miss).max(0.0).sqrt();
        let s = (pu.x - pu.t) * pu.sin;
        if pu.stop && s < front - 0.02 {
            continue; // stopped before it got there
        }
        core += 1;
        through += usize::from(!pu.stop || s > front + 0.02);
    }
    if core == 0 { 1.0 } else { through as f64 / core as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scanned_stem_is_one_measured_row() {
        // the evidence tests' fan: a 5 cm stem at 5 m, claimed at 8 cm 1 cm off its axis, a neighbour 30 cm away
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let (a1, b1) = ([a[0] + 0.01, a[1], a[2]], [b[0] + 0.01, b[1], b[2]]);
        let rows = Rows { a: &[a1], b: &[b1], r_prior: &[0.08], r_max: &[], parent: &[-1] };
        let params = TreeParams { like: JointParams { tau: 0.17, edge: 0.25, hide: 0.0, ..JointParams::default() }, prior: PriorParams { scan_abs: 0.0, scan_rel: 0.0, centre_rel: 0.3, ..PriorParams::default() }, ..TreeParams::default() };
        let f = joint_tree(&[&scan], &rows, &[[0.3, 5.0, -1.0]], &[[0.3, 5.0, 1.0]], &[0.02], &params, None, 2).unwrap();
        assert_eq!(f.src, vec![0], "a solid stem isn't split");
        assert!((f.r[0] - 0.05).abs() < 0.003 && (f.a[0][0]).abs() < 0.003, "R {} at x {}", f.r[0], f.a[0][0]);
        assert!(f.p_wood[0] > 0.99 && f.reason[0] == reason::MEASURED && f.lo[0] < 0.05 && f.hi[0] > 0.05 && !f.flat[0]);
        assert!(f.converged && f.u < 0.5 && f.n_pairs[0] > 200);
        // forced to split, it finds the one stem
        let g = joint_tree(&[&scan], &rows, &[], &[], &[], &params, Some(&[true]), 2).unwrap();
        assert_eq!(g.src, vec![0]);
        assert!((g.r[0] - 0.05).abs() < 0.003);
        assert!(joint_tree(&[&scan], &rows, &[], &[], &[], &params, Some(&[true, false]), 2).is_err());
    }

    #[test]
    fn a_row_beside_a_neighbours_stem_is_no_wood() {
        // a 3 cm claim 12 cm beside the stem, which a neighbour's cylinder holds: the stem's stops are the
        // neighbour's, and the row's own pulses pass
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let rows = Rows { a: &[[a[0] + 0.12, a[1], a[2]]], b: &[[b[0] + 0.12, b[1], b[2]]], r_prior: &[0.03], r_max: &[], parent: &[-1] };
        let params = TreeParams { like: JointParams { tau: 0.17, edge: 0.25, hide: 0.0, ..JointParams::default() }, prior: PriorParams { centre_rel: 0.3, ..PriorParams::default() }, ..TreeParams::default() };
        let f = joint_tree(&[&scan], &rows, &[a], &[b], &[0.05], &params, None, 2).unwrap();
        assert!(f.n_pairs[0] > 50 && f.p_wood[0] < 0.5 && f.reason[0] == reason::NO_WOOD, "pairs {} p_wood {} reason {} R {}", f.n_pairs[0], f.p_wood[0], f.reason[0], f.r[0]);
    }
}
