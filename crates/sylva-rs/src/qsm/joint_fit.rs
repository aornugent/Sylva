// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The fit of the joint wood, foliage and ray model ([`crate::qsm::joint`]):
//! MAP by continuation and damped saddle-free Newton, then its Laplace
//! summary.
//!
//! The unknowns are, per row (a cylinder), its axis offset across the axis
//! (two, in the row's frame), `eta = log R` and `ell = log lambda`; and per
//! (scan, branch) a nuisance offset `nu` that the scan's view of the branch
//! is shifted by (sway, registration). The prior is a Gaussian Markov field
//! over the tree: anchors of the centre on the starting axis, of foliage on
//! its prior, of the level at a base; along a branch, the taper's steps (the
//! prior radii's own) and a random walk of the centre; at a fork, soft
//! Leonardo with the prior radii's area shares; foliage a random walk.
//!
//! A branch is a chain of rows between forks. Every likelihood term lies
//! within one branch (a (row, scan) group touches its row and its scan's nu
//! for that branch), and the prior links a branch to the rest of the tree only
//! through its first row's parent. So the Newton system is a tree of dense
//! branch blocks: it is solved exactly by eliminating each branch into its
//! parent row, leaves first, and substituting back, and the same tree gives
//! the posterior covariance top-down (`Sigma = H^-1 + Y Sigma_parent Y^T`).
//! Its cost is that of the branch blocks, with no fill.
//!
//! Newton is saddle-free: each group's 4x4 Hessian is replaced by the
//! absolute value of its eigenvalues, so a block that isn't concave (a fat
//! claim off-centre) still gives curvature in every direction. The
//! likelihood's surface kernel is widened for the first stages and narrowed
//! to its physical width (continuation): at that width the likelihood is a
//! needle around the true surface, which local steps from a poor start miss.

use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, SymmetricEigen};

use crate::error::{Error, Result};
use crate::qsm::joint::{frames, joint_blocks, joint_loglik, JointParams, Owner, Pulse};
use crate::Point;

/// The prior's widths and anchors (all sds; see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorParams {
    /// Foliage: prior median (1/m) and log sd; its random walk along the tree.
    pub lam_prior: f64,
    pub lam_sd: f64,
    pub sig_ell: f64,
    /// The taper's random walk per sqrt(m) of row length; soft Leonardo (log
    /// area ratio at a fork and its sd); the level at a base.
    pub sig_eta: f64,
    pub mu_fork: f64,
    pub sig_fork: f64,
    pub sig_root: f64,
    /// The centre: its random walk (m per sqrt(m)) and its anchor on the
    /// starting axis, `centre_rel r_prior + centre_abs`.
    pub sig_delta: f64,
    pub centre_rel: f64,
    pub centre_abs: f64,
    /// The per-scan nuisance, `scan_abs + scan_rel r` (both 0: none).
    pub scan_abs: f64,
    pub scan_rel: f64,
}

/// How the fit runs.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// Surface roughness (share of R) of the continuation's stages before the
    /// physical one; those not above the likelihood's own are skipped.
    pub schedule: Vec<f64>,
    /// Newton iterations per stage at most.
    pub iters: usize,
    /// Converged when an iteration improves the objective by less than
    /// `tol (1 + |objective|)`.
    pub tol: f64,
    pub threads: usize,
}

/// The fit: the MAP state, Laplace sds of each row's unknowns, the share of
/// eta's posterior precision from the row's own rays (`rho_local`) and the
/// reduction of its variance from the prior's (`rho_pooled`).
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub theta: Vec<[f64; 4]>,
    pub nu: Vec<[f64; 2]>,
    pub sd: Vec<[f64; 4]>,
    pub rho_local: Vec<f64>,
    pub rho_pooled: Vec<f64>,
    pub objective: f64,
    /// Newton iterations per stage, and whether every stage converged.
    pub iterations: Vec<usize>,
    pub converged: bool,
}

/// The rows: cylinders `a -> b` with their prior radii and parents (a
/// parent's index is below its child's; -1 at a base).
#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    pub a: &'a [Point],
    pub b: &'a [Point],
    pub r_prior: &'a [f64],
    pub parent: &'a [i64],
}

struct Factor {
    vars: Vec<(usize, f64)>,
    mean: f64,
    w: f64,
}

struct Block {
    rows: Vec<usize>,
    dim: usize,
    parent_row: Option<usize>,
}

struct Problem<'a> {
    pulses: &'a [Pulse],
    owners: Vec<Owner>,
    group: Vec<u32>,
    n: usize,
    s: usize,
    use_nu: bool,
    branch: Vec<usize>,
    rowpos: Vec<usize>,
    blocks: Vec<Block>,
    factors: Vec<Factor>,
    nu_w: Vec<f64>,
    cap: Vec<f64>,
    n_var: usize,
    threads: usize,
}

/// A Newton system on the branch tree: each block's matrix, its coupling to
/// its parent row (dim x 4), and the right-hand side.
#[derive(Clone)]
struct System {
    h: Vec<DMatrix<f64>>,
    c: Vec<DMatrix<f64>>,
    g: Vec<DVector<f64>>,
}

impl<'a> Problem<'a> {
    fn new(pulses: &'a [Pulse], row: &[u32], scan: &[u32], rows: &Rows, n_scans: usize, p: &PriorParams, threads: usize) -> Result<Self> {
        let n = rows.a.len();
        if rows.b.len() != n || rows.r_prior.len() != n || rows.parent.len() != n {
            return Err(Error::invalid("joint fit: a, b, r_prior and parent must have one entry per row"));
        }
        if rows.parent.iter().enumerate().any(|(k, &q)| q >= k as i64 || q < -1) {
            return Err(Error::invalid("joint fit: each parent must come before its child (-1 at a base)"));
        }
        if !rows.r_prior.iter().all(|r| r.is_finite() && *r > 0.0) {
            return Err(Error::invalid("joint fit: prior radii must be positive"));
        }
        if row.len() != pulses.len() || scan.len() != pulses.len() || row.iter().any(|&k| k as usize >= n) || scan.iter().any(|&s| s as usize >= n_scans) {
            return Err(Error::invalid("joint fit: each pair needs a row and a scan in range"));
        }
        let parent: Vec<Option<usize>> = rows.parent.iter().map(|&q| (q >= 0).then_some(q as usize)).collect();
        let mut kids = vec![Vec::new(); n];
        for (k, q) in parent.iter().enumerate() {
            if let Some(q) = *q {
                kids[q].push(k);
            }
        }
        // branches: a row starts one at a base or below a fork
        let (mut branch, mut rowpos, mut blocks) = (vec![0; n], vec![0; n], Vec::<Block>::new());
        for k in 0..n {
            match parent[k] {
                Some(q) if kids[q].len() == 1 => branch[k] = branch[q],
                q => {
                    branch[k] = blocks.len();
                    blocks.push(Block { rows: Vec::new(), dim: 0, parent_row: q });
                }
            }
            rowpos[k] = blocks[branch[k]].rows.len();
            blocks[branch[k]].rows.push(k);
        }
        let nb = blocks.len();
        let use_nu = p.scan_abs > 0.0 || p.scan_rel > 0.0;
        for bl in blocks.iter_mut() {
            bl.dim = 4 * bl.rows.len() + if use_nu { 2 * n_scans } else { 0 };
        }
        let fr = frames(rows.a, rows.b);
        let eta0: Vec<f64> = rows.r_prior.iter().map(|r| r.ln()).collect();
        let mut factors = Vec::new();
        let mut add = |vars: Vec<(usize, f64)>, mean: f64, sd: f64| factors.push(Factor { vars, mean, w: 1.0 / (sd * sd) });
        for k in 0..n {
            let sc = p.centre_rel * rows.r_prior[k] + p.centre_abs;
            add(vec![(4 * k, 1.0)], 0.0, sc);
            add(vec![(4 * k + 1, 1.0)], 0.0, sc);
            add(vec![(4 * k + 3, 1.0)], p.lam_prior.ln(), p.lam_sd);
            let Some(q) = parent[k] else {
                add(vec![(4 * k + 2, 1.0)], eta0[k], p.sig_root);
                continue;
            };
            let ls = fr[k].3.max(0.01).sqrt();
            if kids[q].len() == 1 {
                add(vec![(4 * k + 2, 1.0), (4 * q + 2, -1.0)], eta0[k] - eta0[q], p.sig_eta * ls);
                // the parent's offset seen in this row's frame
                let (ek, eq) = ([fr[k].1, fr[k].2], [fr[q].1, fr[q].2]);
                for j in 0..2 {
                    let t: Vec<f64> = (0..2).map(|i| (0..3).map(|x| ek[j][x] * eq[i][x]).sum()).collect();
                    add(vec![(4 * k + j, 1.0), (4 * q, -t[0]), (4 * q + 1, -t[1])], 0.0, p.sig_delta * ls);
                }
            } else {
                let total: f64 = kids[q].iter().map(|&c| rows.r_prior[c].powi(2)).sum();
                let share = rows.r_prior[k].powi(2) / total;
                add(vec![(4 * k + 2, 1.0), (4 * q + 2, -1.0)], 0.5 * (p.mu_fork + share.ln()), p.sig_fork);
            }
            add(vec![(4 * k + 3, 1.0), (4 * q + 3, -1.0)], 0.0, p.sig_ell);
        }
        let g_nu = if use_nu { n_scans * nb } else { 0 };
        let n_var = 4 * n + 2 * g_nu;
        let rb: Vec<f64> = blocks.iter().map(|bl| bl.rows.iter().map(|&k| rows.r_prior[k]).sum::<f64>() / bl.rows.len() as f64).collect();
        let mut nu_w = vec![0.0; 2 * g_nu];
        let mut cap = vec![0.0; n_var];
        for k in 0..n {
            let c = 0.2 * rows.r_prior[k] + 0.001;
            cap[4 * k..4 * k + 4].copy_from_slice(&[c, c, 0.2, 1.0]);
        }
        for s in 0..(if use_nu { n_scans } else { 0 }) {
            for (bi, r) in rb.iter().enumerate() {
                let i = 2 * (s * nb + bi);
                let sd = p.scan_abs + p.scan_rel * r;
                nu_w[i..i + 2].fill(1.0 / (sd * sd));
                cap[4 * n + i..4 * n + i + 2].fill(0.2 * r + 0.001);
            }
        }
        let owners = row.iter().zip(scan).map(|(&k, &s)| Owner { row: k, nu: if use_nu { (s as usize * nb + branch[k as usize]) as u32 } else { u32::MAX } }).collect();
        let group = row.iter().zip(scan).map(|(&k, &s)| k * n_scans as u32 + s).collect();
        Ok(Problem { pulses, owners, group, n, s: n_scans, use_nu, branch, rowpos, blocks, factors, nu_w, cap, n_var, threads })
    }

    /// Block and position of a global unknown.
    fn loc(&self, v: usize) -> (usize, usize) {
        if v < 4 * self.n {
            let k = v / 4;
            (self.branch[k], 4 * self.rowpos[k] + v % 4)
        } else {
            let (i, j) = ((v - 4 * self.n) / 2, (v - 4 * self.n) % 2);
            let nb = self.blocks.len();
            let (s, bi) = (i / nb, i % nb);
            (bi, 4 * self.blocks[bi].rows.len() + 2 * s + j)
        }
    }

    fn split(&self, x: &[f64]) -> (Vec<[f64; 4]>, Vec<[f64; 2]>) {
        let theta = (0..self.n).map(|k| [x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]]).collect();
        let nu = if self.use_nu { x[4 * self.n..].chunks(2).map(|c| [c[0], c[1]]).collect() } else { vec![[0.0, 0.0]] };
        (theta, nu)
    }

    fn objective(&self, x: &[f64], like: &JointParams) -> Result<f64> {
        let (theta, nu) = self.split(x);
        let ll: f64 = joint_loglik(self.pulses, &self.owners, &theta, &nu, like, self.threads)?.iter().sum();
        let prior: f64 = self.factors.iter().map(|f| {
            let r = f.vars.iter().map(|&(v, c)| c * x[v]).sum::<f64>() - f.mean;
            -0.5 * f.w * r * r
        }).sum();
        let nu: f64 = self.nu_w.iter().zip(&x[4 * self.n..]).map(|(w, v)| -0.5 * w * v * v).sum();
        Ok(ll + prior + nu)
    }

    fn empty_system(&self) -> System {
        System {
            h: self.blocks.iter().map(|bl| DMatrix::zeros(bl.dim, bl.dim)).collect(),
            c: self.blocks.iter().map(|bl| DMatrix::zeros(bl.dim, 4)).collect(),
            g: self.blocks.iter().map(|bl| DVector::zeros(bl.dim)).collect(),
        }
    }

    /// Adds `val` at (vi, vj) of the system's matrix: within a block, or the
    /// coupling of a branch's first row to its parent row (only the child ->
    /// parent orientation is kept; the matrix is symmetric).
    fn add_h(&self, sys: &mut System, vi: usize, vj: usize, val: f64) {
        let ((bi, pi), (bj, pj)) = (self.loc(vi), self.loc(vj));
        if bi == bj {
            sys.h[bi][(pi, pj)] += val;
        } else if self.blocks[bi].parent_row.is_some_and(|q| vj / 4 == q && vj < 4 * self.n) {
            sys.c[bi][(pi, vj % 4)] += val;
        }
        // else: the parent -> child orientation of a coupling, kept above
    }

    /// The prior's (and the nuisance prior's) gradient of the objective and
    /// negated Hessian at x.
    fn add_prior(&self, sys: &mut System, x: &[f64]) {
        for f in &self.factors {
            let r = f.vars.iter().map(|&(v, c)| c * x[v]).sum::<f64>() - f.mean;
            for &(vi, ci) in &f.vars {
                let (b, p) = self.loc(vi);
                sys.g[b][p] -= f.w * r * ci;
                for &(vj, cj) in &f.vars {
                    self.add_h(sys, vi, vj, f.w * ci * cj);
                }
            }
        }
        for (i, w) in self.nu_w.iter().enumerate() {
            let v = 4 * self.n + i;
            let (b, p) = self.loc(v);
            sys.g[b][p] -= w * x[v];
            sys.h[b][(p, p)] += w;
        }
    }

    /// The Newton system at x: the likelihood's per-(row, scan) blocks made
    /// positive definite (absolute eigenvalues) and lifted onto their row and
    /// nuisance, plus the prior. Also each row's summed likelihood block.
    fn system(&self, x: &[f64], like: &JointParams) -> Result<(System, Vec<Matrix4<f64>>)> {
        let (theta, nu) = self.split(x);
        let n_groups = self.n * self.s;
        let (_, grad, hess) = joint_blocks(self.pulses, &self.owners, &self.group, n_groups, &theta, &nu, like, self.threads)?;
        let mut sys = self.empty_system();
        let mut row_h = vec![Matrix4::zeros(); self.n];
        let mut used = vec![false; n_groups];
        for &q in &self.group {
            used[q as usize] = true;
        }
        for q in (0..n_groups).filter(|&q| used[q]) {
            let (k, s) = (q / self.s, q % self.s);
            let h = abs_eig(&Matrix4::from(hess[q]));
            row_h[k] += h;
            let (bi, rp) = (self.branch[k], 4 * self.rowpos[k]);
            let np = 4 * self.blocks[bi].rows.len() + 2 * s;
            let at = |a: usize| -> Vec<usize> { if a < 2 && self.use_nu { vec![rp + a, np + a] } else { vec![rp + a] } };
            for a in 0..4 {
                for &pa in &at(a) {
                    sys.g[bi][pa] += grad[q][a];
                    for b in 0..4 {
                        for &pb in &at(b) {
                            sys.h[bi][(pa, pb)] += h[(a, b)];
                        }
                    }
                }
            }
        }
        self.add_prior(&mut sys, x);
        Ok((sys, row_h))
    }

    /// Solves `(H + mu I) x = g` on the branch tree: each block eliminated
    /// into its parent row, leaves first, then substituted back. With `cov`,
    /// also the diagonal of the inverse (top-down). None if a block isn't
    /// positive definite.
    #[allow(clippy::type_complexity)]
    fn solve(&self, sys: &System, mu: f64, cov: bool) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
        let nb = self.blocks.len();
        let mut h = sys.h.clone();
        let mut g = sys.g.clone();
        for m in h.iter_mut() {
            for i in 0..m.nrows() {
                m[(i, i)] += mu;
            }
        }
        let mut chol = Vec::with_capacity(nb);
        let mut ys: Vec<Option<(DMatrix<f64>, DVector<f64>)>> = vec![None; nb];
        for bi in (0..nb).rev() {
            let ch = h[bi].clone().cholesky()?;
            if let Some(q) = self.blocks[bi].parent_row {
                let (pb, pp) = (self.branch[q], 4 * self.rowpos[q]);
                let y_mat = ch.solve(&sys.c[bi]);
                let y_vec = ch.solve(&g[bi]);
                let ct = sys.c[bi].transpose();
                let dh = &ct * &y_mat;
                let dg = &ct * &y_vec;
                let mut hv = h[pb].view_mut((pp, pp), (4, 4));
                hv -= dh;
                let mut gv = g[pb].rows_mut(pp, 4);
                gv -= dg;
                ys[bi] = Some((y_mat, y_vec));
            }
            chol.push((bi, ch));
        }
        chol.reverse(); // now in block order
        let mut xb: Vec<DVector<f64>> = vec![DVector::zeros(0); nb];
        let mut sig: Vec<DMatrix<f64>> = vec![DMatrix::zeros(0, 0); nb];
        for (bi, ch) in &chol {
            let bi = *bi;
            match (&ys[bi], self.blocks[bi].parent_row) {
                (Some((y_mat, y_vec)), Some(q)) => {
                    let (pb, pp) = (self.branch[q], 4 * self.rowpos[q]);
                    let xp = xb[pb].rows(pp, 4).into_owned();
                    xb[bi] = y_vec - y_mat * xp;
                    if cov {
                        let sp = sig[pb].view((pp, pp), (4, 4)).into_owned();
                        sig[bi] = ch.inverse() + y_mat * sp * y_mat.transpose();
                    }
                }
                _ => {
                    xb[bi] = ch.solve(&g[bi]);
                    if cov {
                        sig[bi] = ch.inverse();
                    }
                }
            }
        }
        let mut x = vec![0.0; self.n_var];
        let mut var = cov.then(|| vec![0.0; self.n_var]);
        for v in 0..self.n_var {
            let (b, p) = self.loc(v);
            x[v] = xb[b][p];
            if let Some(var) = var.as_mut() {
                var[v] = sig[b][(p, p)];
            }
        }
        Some((x, var))
    }

    /// Damped saddle-free Newton at one stage; returns the iterations taken
    /// and whether it converged.
    fn newton(&self, x: &mut [f64], like: &JointParams, iters: usize, tol: f64) -> Result<(usize, bool)> {
        let mut obj = self.objective(x, like)?;
        let mut mu = 1.0;
        for it in 0..iters {
            let (sys, _) = self.system(x, like)?;
            let mut accepted = None;
            for _ in 0..12 {
                let Some((mut step, _)) = self.solve(&sys, mu, false) else {
                    mu *= 4.0;
                    continue;
                };
                for (s, c) in step.iter_mut().zip(&self.cap) {
                    *s = s.clamp(-c, *c);
                }
                let trial: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
                let o = self.objective(&trial, like)?;
                if o > obj {
                    accepted = Some((trial, o - obj));
                    obj = o;
                    mu = (mu / 3.0).max(1e-6);
                    break;
                }
                mu *= 4.0;
            }
            let Some((trial, gain)) = accepted else {
                return Ok((it, true)); // no damped step improves: a maximum to the precision of the objective
            };
            x.copy_from_slice(&trial);
            if gain < tol * (1.0 + obj.abs()) {
                return Ok((it + 1, true));
            }
        }
        Ok((iters, false))
    }
}

/// `V |Lambda| V^T` of a symmetric 4x4, negated: the curvature saddle-free
/// Newton uses (eigenvalues floored at 1e-6).
fn abs_eig(h: &Matrix4<f64>) -> Matrix4<f64> {
    let e = SymmetricEigen::new(*h);
    let d = Matrix4::from_diagonal(&e.eigenvalues.map(|l| l.abs().max(1e-6)));
    e.eigenvectors * d * e.eigenvectors.transpose()
}

/// Fits the joint model to `pulses` (row and scan of each) on `rows`, from
/// `theta0` (per row; default the starting axes, the prior radii and the
/// prior foliage).
#[allow(clippy::too_many_arguments)]
pub fn joint_fit(pulses: &[Pulse], row: &[u32], scan: &[u32], rows: &Rows, n_scans: usize, like: &JointParams, prior: &PriorParams, opts: &FitOptions, theta0: Option<&[[f64; 4]]>) -> Result<Fit> {
    let pr = Problem::new(pulses, row, scan, rows, n_scans, prior, opts.threads)?;
    let mut x = vec![0.0; pr.n_var];
    for k in 0..pr.n {
        let t = theta0.map_or([0.0, 0.0, rows.r_prior[k].ln(), prior.lam_prior.ln()], |t| t[k]);
        x[4 * k..4 * k + 4].copy_from_slice(&t);
    }
    if theta0.is_some_and(|t| t.len() != pr.n) {
        return Err(Error::invalid("joint fit: theta0 must have one row per row"));
    }
    let mut stages: Vec<f64> = opts.schedule.iter().copied().filter(|&s| s > like.rough).collect();
    stages.push(like.rough);
    let (mut iterations, mut converged) = (Vec::new(), true);
    for rough in stages {
        let (it, ok) = pr.newton(&mut x, &JointParams { rough, ..*like }, opts.iters, opts.tol)?;
        iterations.push(it);
        converged &= ok;
    }
    // Laplace: the posterior's and the prior's variances on the branch tree
    let (sys, row_h) = pr.system(&x, like)?;
    let (_, var) = pr.solve(&sys, 0.0, true).ok_or_else(|| Error::invalid("joint fit: the posterior isn't positive definite"))?;
    let var = var.unwrap();
    let mut prior_sys = pr.empty_system();
    pr.add_prior(&mut prior_sys, &x);
    let (_, pvar) = pr.solve(&prior_sys, 0.0, true).ok_or_else(|| Error::invalid("joint fit: the prior isn't positive definite"))?;
    let pvar = pvar.unwrap();
    let (theta, nu) = pr.split(&x);
    let sd = (0..pr.n).map(|k| std::array::from_fn(|j| var[4 * k + j].sqrt())).collect();
    let rho_local = (0..pr.n).map(|k| {
        // eta's precision from the row's own rays, its other unknowns integrated out within the row
        let h = &row_h[k];
        let o = [0, 1, 3];
        let hoo = Matrix3::from_fn(|i, j| h[(o[i], o[j])]) + Matrix3::identity() * 1e-9;
        let ho = nalgebra::Vector3::from_fn(|i, _| h[(o[i], 2)]);
        let tau = h[(2, 2)] - hoo.try_inverse().map_or(0.0, |inv| (ho.transpose() * inv * ho)[(0, 0)]);
        (tau * var[4 * k + 2]).clamp(0.0, 1.0)
    }).collect();
    let rho_pooled = (0..pr.n).map(|k| 1.0 - var[4 * k + 2] / pvar[4 * k + 2]).collect();
    Ok(Fit { theta, nu: if pr.use_nu { nu } else { Vec::new() }, sd, rho_local, rho_pooled, objective: pr.objective(&x, like)?, iterations, converged })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prior() -> PriorParams {
        PriorParams { lam_prior: 1.0, lam_sd: 2.0, sig_ell: 1.0, sig_eta: 0.05, mu_fork: 0.0, sig_fork: 0.17, sig_root: 0.5, sig_delta: 0.02, centre_rel: 0.3, centre_abs: 0.003, scan_abs: 0.002, scan_rel: 0.05 }
    }

    /// A small tree: a trunk of 3 rows forking into two branches of 2.
    fn tree() -> (Vec<Point>, Vec<Point>, Vec<f64>, Vec<i64>) {
        let a = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.01, 0.0, 0.35], [0.0, 0.0, 0.3], [-0.01, 0.0, 0.35]];
        let b = vec![[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.01, 0.0, 0.35], [0.02, 0.0, 0.4], [-0.01, 0.0, 0.35], [-0.02, 0.0, 0.4]];
        // area-preserving at the fork (mu_fork 0), so the prior's mode is these radii
        let fork = 0.046 / 2f64.sqrt();
        (a, b, vec![0.05, 0.048, 0.046, fork, 0.03, fork, 0.03], vec![-1, 0, 1, 2, 3, 2, 5])
    }

    #[test]
    fn the_branch_tree_solves_the_system_exactly() {
        // a random symmetric positive definite system on the branch structure, against a dense solve
        let (a, b, r, parent) = tree();
        let rows = Rows { a: &a, b: &b, r_prior: &r, parent: &parent };
        let pulses: Vec<Pulse> = Vec::new();
        let pr = Problem::new(&pulses, &[], &[], &rows, 2, &prior(), 1).unwrap();
        assert_eq!(pr.blocks.len(), 3, "trunk and two branches");
        let x: Vec<f64> = (0..pr.n_var).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.01).collect();
        let mut sys = pr.empty_system();
        pr.add_prior(&mut sys, &x);
        // the dense form of the same system
        let mut dense = DMatrix::zeros(pr.n_var, pr.n_var);
        let mut rhs = DVector::zeros(pr.n_var);
        for f in &pr.factors {
            let rr = f.vars.iter().map(|&(v, c)| c * x[v]).sum::<f64>() - f.mean;
            for &(vi, ci) in &f.vars {
                rhs[vi] -= f.w * rr * ci;
                for &(vj, cj) in &f.vars {
                    dense[(vi, vj)] += f.w * ci * cj;
                }
            }
        }
        for (i, w) in pr.nu_w.iter().enumerate() {
            let v = 4 * pr.n + i;
            rhs[v] -= w * x[v];
            dense[(v, v)] += w;
        }
        let want: DVector<f64> = dense.clone().cholesky().unwrap().solve(&rhs);
        let want_var: DVector<f64> = dense.cholesky().unwrap().inverse().diagonal();
        let (got, var) = pr.solve(&sys, 0.0, true).unwrap();
        let var = var.unwrap();
        for v in 0..pr.n_var {
            assert!((got[v] - want[v]).abs() < 1e-9 * (1.0 + want[v].abs()), "x[{v}] {} vs {}", got[v], want[v]);
            assert!((var[v] - want_var[v]).abs() < 1e-9 * want_var[v], "var[{v}] {} vs {}", var[v], want_var[v]);
        }
    }

    #[test]
    fn without_data_the_fit_is_the_prior() {
        let (a, b, r, parent) = tree();
        let rows = Rows { a: &a, b: &b, r_prior: &r, parent: &parent };
        let like = JointParams { exit: 0.007, divergence: 0.35e-3, sigma_r: 0.005, rough: 0.03, tau: 0.17, eps: 0.1, eps_min: 0.01, mp_len: 0.5, thin: 0.0, step: 0.03f64.to_radians() };
        let opts = FitOptions { schedule: vec![0.3, 0.1], iters: 20, tol: 1e-10, threads: 1 };
        let f = joint_fit(&[], &[], &[], &rows, 2, &like, &prior(), &opts, None).unwrap();
        assert!(f.converged);
        for (k, t) in f.theta.iter().enumerate() {
            assert!((t[2] - r[k].ln()).abs() < 1e-6 && t[0].abs() < 1e-9, "row {k}: {t:?}");
        }
        assert!(f.rho_local.iter().all(|&x| x == 0.0) && f.rho_pooled.iter().all(|x| x.abs() < 1e-9));
        assert!(joint_fit(&[], &[], &[], &Rows { parent: &[-1, 0, 1, 2, 3, 6, 5], ..rows }, 2, &like, &prior(), &opts, None).is_err());
    }
}
