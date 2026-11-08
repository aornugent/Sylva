// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The fit of the joint wood, foliage and ray model ([`crate::qsm::joint`]):
//! MAP by continuation and saddle-free Newton with a line search, then its
//! Laplace summary.
//!
//! The unknowns are, per row (a cylinder), its axis offset across the axis
//! (two, in the row's frame), `eta = log R` and `ell = log lambda`; and per
//! (scan, branch) a nuisance offset `nu` that the scan's view of the branch
//! is shifted by (sway, registration). The prior is a Gaussian Markov field
//! over the tree: anchors of the centre on the starting axis, of foliage on
//! its prior, of the level at a base and (optionally) of the radius at a tip;
//! along a branch, the taper's steps (the prior radii's own) and a random walk
//! of the centre; at a fork, soft Leonardo with the prior radii's area shares;
//! foliage a random walk. A radius is capped at its row's `r_max` (the tube
//! the pairs were taken in: wood wider than it is unobserved).
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
//! claim off-centre) still gives curvature in every direction; a backtracking
//! line search takes the step (see [`FitOptions::tol`] for when a stage has
//! converged). The likelihood's surface
//! kernel is widened for the first stages and narrowed to its physical width
//! (continuation): at that width the likelihood is a needle around the true
//! surface, which local steps from a poor start miss.

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Matrix4, SymmetricEigen};

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
    /// A tip's radius (m; 0: no anchor) and its log sd, at rows without
    /// children.
    pub r_tip: f64,
    pub sig_tip: f64,
    /// The centre: its random walk (m per sqrt(m)) and its anchor on the
    /// starting axis, `centre_rel r_prior + centre_abs`.
    pub sig_delta: f64,
    pub centre_rel: f64,
    pub centre_abs: f64,
    /// The per-scan nuisance, `scan_abs + scan_rel r` (both 0: none).
    pub scan_abs: f64,
    pub scan_rel: f64,
    /// The lean's anchor (m per m) and its random walk between a row's pieces (per sqrt(m); 3dtrees research 93
    /// §13.1: rows of 67-98 cm whose centres drift 8-24 cm). Read by [`crate::qsm::support_tree`] only.
    pub lean_sd: f64,
    pub lean_bend: f64,
    /// Existence along a branch: a base piece is wood with `exist`, one after a piece that isn't with `exist_off`
    /// (wood doesn't stop and restart). Read by [`crate::qsm::support_tree`] only.
    pub exist: f64,
    pub exist_off: f64,
}

impl Default for PriorParams {
    /// The calibration for TERN's Alice Mulga exports (3dtrees research 74 §14): real axes and scans are freer
    /// than a synthetic bench's, and RCT's taper is no guide to the rays' radii there (§21: rows fitted alone
    /// disagree with its steps by ~1 in log R, which `sig_eta` 0.05 priced at ~10^6 log units).
    fn default() -> Self {
        PriorParams { lam_prior: 1.0, lam_sd: 2.0, sig_ell: 1.0, sig_eta: 0.5, mu_fork: 0.10, sig_fork: 0.17, sig_root: 0.5, r_tip: 0.0, sig_tip: 0.3, sig_delta: 0.02, centre_rel: 1.0, centre_abs: 0.003, scan_abs: 0.002, scan_rel: 0.3, lean_sd: 0.5, lean_bend: 0.3, exist: 0.65, exist_off: 0.5 }
    }
}

impl PriorParams {
    pub(crate) fn check(&self) -> Result<()> {
        let sds = [("lam_prior", self.lam_prior), ("lam_sd", self.lam_sd), ("sig_ell", self.sig_ell), ("sig_eta", self.sig_eta), ("sig_fork", self.sig_fork), ("sig_root", self.sig_root), ("sig_delta", self.sig_delta), ("lean_sd", self.lean_sd), ("lean_bend", self.lean_bend)];
        if let Some((k, _)) = sds.iter().find(|(_, v)| !(v.is_finite() && *v > 0.0)) {
            return Err(Error::invalid(format!("joint fit: {k} must be positive")));
        }
        if let Some((k, _)) = [("exist", self.exist), ("exist_off", self.exist_off)].iter().find(|(_, v)| !(*v > 0.0 && *v < 1.0)) {
            return Err(Error::invalid(format!("joint fit: {k} must be a probability in (0, 1)")));
        }
        let shares = [("mu_fork", self.mu_fork), ("centre_rel", self.centre_rel), ("centre_abs", self.centre_abs), ("scan_abs", self.scan_abs), ("scan_rel", self.scan_rel), ("r_tip", self.r_tip)];
        if let Some((k, _)) = shares.iter().find(|(k, v)| !v.is_finite() || (*k != "mu_fork" && *v < 0.0)) {
            return Err(Error::invalid(format!("joint fit: {k} must be finite (and not negative)")));
        }
        if self.centre_rel == 0.0 && self.centre_abs == 0.0 {
            return Err(Error::invalid("joint fit: the centre anchor needs centre_rel or centre_abs"));
        }
        if self.r_tip > 0.0 && !(self.sig_tip.is_finite() && self.sig_tip > 0.0) {
            return Err(Error::invalid("joint fit: a tip anchor needs a positive sig_tip"));
        }
        Ok(())
    }
}

/// How the fit runs.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// Surface roughness (share of R) of the continuation's stages before the
    /// physical one; those not above the likelihood's own are skipped.
    pub schedule: Vec<f64>,
    /// Newton iterations per stage at most.
    pub iters: usize,
    /// A stage has converged when its Newton decrement, or the objective's
    /// gain over its last ten iterations, is below `tol (1 + |objective|)`.
    /// The likelihood is only piecewise smooth (clamps, the silhouette), so at
    /// its maximum the decrement need not vanish.
    pub tol: f64,
    pub threads: usize,
    /// Local step control: each row's step (and each nuisance pair's) is held
    /// within its caps by its own scale, and each branch block backtracks on
    /// its own share of the objective (the likelihood is separable by block),
    /// the whole step then checked; else one scale and one line search for
    /// the whole step, which one stiff row throttles.
    pub local: bool,
    /// Compute [`Fit::profile`] (21 passes over the pairs); else it is empty.
    pub profile: bool,
}

/// The `eta - eta_hat` offsets of [`Fit::profile`].
pub fn profile_grid() -> [f64; 21] {
    std::array::from_fn(|i| -1.5 + 0.125 * i as f64)
}

/// The fit: the MAP state, Laplace sds of each row's unknowns, the share of
/// eta's posterior precision from the row's own rays (`rho_local`, each scan's
/// nuisance integrated out) and the reduction of its variance from the
/// prior's (`rho_pooled`).
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub theta: Vec<[f64; 4]>,
    pub nu: Vec<[f64; 2]>,
    pub sd: Vec<[f64; 4]>,
    pub rho_local: Vec<f64>,
    pub rho_pooled: Vec<f64>,
    /// Each row's likelihood message at the MAP: the gradient and Hessian of
    /// its own pairs' log likelihood in its state (the nuisances held), from
    /// which a new prior re-solves the tree without the rays.
    pub grad: Vec<[f64; 4]>,
    pub hess: Vec<[[f64; 4]; 4]>,
    /// Each row's own pairs' log likelihood with eta moved by each
    /// [`profile_grid`] offset, the rest of the state held at the MAP (empty
    /// unless [`FitOptions::profile`]).
    pub profile: Vec<[f64; 21]>,
    pub objective: f64,
    /// Newton iterations per stage, and whether every stage converged.
    pub iterations: Vec<usize>,
    pub converged: bool,
    /// The sds come from the true Hessian (else, where it isn't negative
    /// definite at the MAP, from the saddle-free one).
    pub exact: bool,
    /// The pair evaluations the fit made: of the value alone, and with the
    /// gradient and Hessian (each pass over the pairs counts each pair once).
    pub evals: [u64; 2],
}

/// The rows: cylinders `a -> b` with their prior radii, the largest radius
/// their pairs can show (the tube's; infinite for none), and parents (a parent's index is below
/// its child's; -1 at a base).
#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    pub a: &'a [Point],
    pub b: &'a [Point],
    pub r_prior: &'a [f64],
    pub r_max: &'a [f64],
    pub parent: &'a [i64],
}

/// A row's frame: its unit axis, the two directions across it, and its length (see [`frames`]).
pub type Frame = (Point, Point, Point, f64);

/// What a fork's child's share of its parent's area is proportional to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForkShare {
    /// Its prior radius squared ([`joint_fit`]).
    Area,
    /// Its subtree's wood length (3dtrees research 93 §11.8: [`crate::qsm::support_tree`]).
    WoodLength,
}

impl Rows<'_> {
    /// Checks the rows (`r_max`, where given, one per row and positive) and returns their frames; `who` names the
    /// caller in the errors.
    pub fn check(&self, who: &str) -> Result<Vec<Frame>> {
        let n = self.a.len();
        if [self.b.len(), self.r_prior.len(), self.parent.len()].iter().any(|&l| l != n) || !(self.r_max.is_empty() || self.r_max.len() == n) {
            return Err(Error::invalid(format!("{who}: a, b, r_prior, r_max and parent must have one entry per row")));
        }
        if self.parent.iter().enumerate().any(|(k, &q)| q >= k as i64 || q < -1) {
            return Err(Error::invalid(format!("{who}: each parent must come before its child (-1 at a base)")));
        }
        if !self.r_prior.iter().all(|r| r.is_finite() && *r > 0.0) || !self.r_max.iter().all(|r| *r > 0.0) {
            return Err(Error::invalid(format!("{who}: prior radii must be positive, and largest radii positive or infinite")));
        }
        let fr = frames(self.a, self.b);
        if !fr.iter().all(|f| f.3.is_finite() && f.3 > 0.0) {
            return Err(Error::invalid(format!("{who}: every row needs a length")));
        }
        Ok(fr)
    }

    /// Each row's share of its parent's area where the parent forks (NaN elsewhere), by `by`; `len` the rows'
    /// lengths. Both priors' fork rule, in one place.
    pub fn fork_share(&self, len: &[f64], by: ForkShare) -> Vec<f64> {
        let n = self.a.len();
        let mut weight: Vec<f64> = match by {
            ForkShare::Area => self.r_prior.iter().map(|r| r.powi(2)).collect(),
            ForkShare::WoodLength => len.to_vec(),
        };
        if by == ForkShare::WoodLength {
            for k in (0..n).rev() {
                if self.parent[k] >= 0 {
                    weight[self.parent[k] as usize] += weight[k];
                }
            }
        }
        let (mut kids, mut total) = (vec![0usize; n], vec![0.0; n]);
        for k in 0..n {
            if self.parent[k] >= 0 {
                kids[self.parent[k] as usize] += 1;
                total[self.parent[k] as usize] += weight[k];
            }
        }
        (0..n).map(|k| match self.parent[k] {
            q if q >= 0 && kids[q as usize] > 1 => weight[k] / total[q as usize],
            _ => f64::NAN,
        }).collect()
    }
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
    /// Per row, its branch's nuisance (in the branch's first row's frame) in the row's frame.
    nu_t: Vec<Matrix2<f64>>,
    cap: Vec<f64>,
    eta_max: Vec<f64>,
    n_var: usize,
    threads: usize,
    /// Pair evaluations so far: value, value with derivatives.
    evals: std::cell::Cell<[u64; 2]>,
}

/// A Newton system on the branch tree: each block's matrix, its coupling to
/// its parent row (dim x 4), and the right-hand side.
#[derive(Clone)]
struct System {
    h: Vec<DMatrix<f64>>,
    c: Vec<DMatrix<f64>>,
    g: Vec<DVector<f64>>,
}

/// The likelihood's per-group sums: gradient and Hessian.
type Groups = (Vec<[f64; 4]>, Vec<[[f64; 4]; 4]>);

impl<'a> Problem<'a> {
    fn new(pulses: &'a [Pulse], row: &[u32], scan: &[u32], rows: &Rows, n_scans: usize, p: &PriorParams, threads: usize) -> Result<Self> {
        p.check()?;
        let n = rows.a.len();
        let fr = rows.check("joint fit")?;
        if rows.r_max.len() != n {
            return Err(Error::invalid("joint fit: r_max must have one entry per row"));
        }
        if row.len() != pulses.len() || scan.len() != pulses.len() || row.iter().any(|&k| k as usize >= n) || scan.iter().any(|&s| s as usize >= n_scans) {
            return Err(Error::invalid("joint fit: each pair needs a row and a scan in range"));
        }
        if n.saturating_mul(n_scans) > u32::MAX as usize {
            return Err(Error::invalid("joint fit: too many rows times scans"));
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
        if use_nu && n_scans.saturating_mul(nb) >= u32::MAX as usize {
            return Err(Error::invalid("joint fit: too many scans times branches"));
        }
        for bl in blocks.iter_mut() {
            bl.dim = 4 * bl.rows.len() + if use_nu { 2 * n_scans } else { 0 };
        }
        let eta0: Vec<f64> = rows.r_prior.iter().map(|r| r.ln()).collect();
        let share = rows.fork_share(&fr.iter().map(|f| f.3).collect::<Vec<_>>(), ForkShare::Area);
        let mut factors = Vec::new();
        let mut add = |vars: Vec<(usize, f64)>, mean: f64, sd: f64| factors.push(Factor { vars, mean, w: 1.0 / (sd * sd) });
        for k in 0..n {
            let centre = p.centre_rel * rows.r_prior[k] + p.centre_abs;
            add(vec![(4 * k, 1.0)], 0.0, centre);
            add(vec![(4 * k + 1, 1.0)], 0.0, centre);
            add(vec![(4 * k + 3, 1.0)], p.lam_prior.ln(), p.lam_sd);
            if kids[k].is_empty() && p.r_tip > 0.0 {
                add(vec![(4 * k + 2, 1.0)], p.r_tip.ln(), p.sig_tip);
            }
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
                let share = share[k];
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
        // a branch's nuisance is an offset in its first row's frame, seen in each row's own (frames turn along a branch)
        let nu_t = (0..n).map(|k| {
            let f = blocks[branch[k]].rows[0];
            let (ek, ef) = ([fr[k].1, fr[k].2], [fr[f].1, fr[f].2]);
            if f == k { return Matrix2::identity() }
            Matrix2::from_fn(|j, i| (0..3).map(|x| ek[j][x] * ef[i][x]).sum())
        }).collect();
        let owners = row.iter().zip(scan).map(|(&k, &s)| Owner { row: k, nu: if use_nu { (s as usize * n + k as usize) as u32 } else { u32::MAX } }).collect();
        let group = row.iter().zip(scan).map(|(&k, &s)| (k as usize * n_scans + s as usize) as u32).collect();
        let eta_max = rows.r_max.iter().map(|r| r.ln()).collect();
        Ok(Problem { pulses, owners, group, n, s: n_scans, use_nu, branch, rowpos, blocks, factors, nu_w, nu_t, cap, eta_max, n_var, threads, evals: std::cell::Cell::new([0; 2]) })
    }

    /// Counts `m` pair evaluations, with derivatives or not.
    fn count(&self, m: usize, deriv: bool) {
        let mut e = self.evals.get();
        e[usize::from(deriv)] += m as u64;
        self.evals.set(e);
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

    /// A vector on the branch tree in global order.
    fn gather(&self, v: &[DVector<f64>]) -> Vec<f64> {
        (0..self.n_var).map(|i| {
            let (b, p) = self.loc(i);
            v[b][p]
        }).collect()
    }

    fn split(&self, x: &[f64]) -> (Vec<[f64; 4]>, Vec<[f64; 2]>) {
        let theta = (0..self.n).map(|k| [x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]]).collect();
        let nu = if self.use_nu { x[4 * self.n..].chunks(2).map(|c| [c[0], c[1]]).collect() } else { vec![[0.0, 0.0]] };
        (theta, nu)
    }

    /// Each (scan, row)'s nuisance in the row's frame, from the (scan, branch) ones.
    fn row_nu(&self, nu: &[[f64; 2]]) -> Vec<[f64; 2]> {
        if !self.use_nu {
            return nu.to_vec();
        }
        let nb = self.blocks.len();
        (0..self.s * self.n).map(|i| {
            let (s, k) = (i / self.n, i % self.n);
            let v = self.nu_t[k] * nalgebra::Vector2::from(nu[s * nb + self.branch[k]]);
            [v[0], v[1]]
        }).collect()
    }

    /// Caps each radius at its row's largest.
    fn project(&self, x: &mut [f64]) {
        for k in 0..self.n {
            x[4 * k + 2] = x[4 * k + 2].min(self.eta_max[k]);
        }
    }

    /// Each row's own pairs' log likelihood at row states `theta`.
    fn row_loglik(&self, theta: &[[f64; 4]], nu: &[[f64; 2]], like: &JointParams) -> Result<Vec<f64>> {
        let ll = joint_loglik(self.pulses, &self.owners, theta, &self.row_nu(nu), like, self.threads)?;
        self.count(ll.len(), false);
        let mut out = vec![0.0; self.n];
        for (o, l) in self.owners.iter().zip(ll) {
            out[o.row as usize] += l;
        }
        Ok(out)
    }

    fn objective(&self, x: &[f64], like: &JointParams) -> Result<f64> {
        let (theta, nu) = self.split(x);
        let ll: f64 = joint_loglik(self.pulses, &self.owners, &theta, &self.row_nu(&nu), like, self.threads)?.iter().sum();
        self.count(self.pulses.len(), false);
        let prior: f64 = self.factors.iter().map(|f| {
            let r = f.vars.iter().map(|&(v, c)| c * x[v]).sum::<f64>() - f.mean;
            -0.5 * f.w * r * r
        }).sum();
        let nu: f64 = self.nu_w.iter().zip(&x[4 * self.n..]).map(|(w, v)| -0.5 * w * v * v).sum();
        Ok(ll + prior + nu)
    }

    /// The objective split by branch block: its rows' pairs, its nuisances'
    /// prior, and the prior factors whose first unknown is in it (a fork's
    /// factor goes to the child's block).
    /// With `only`, the likelihood of those blocks alone (the others' entries are then their priors only).
    fn block_objective(&self, x: &[f64], like: &JointParams, only: Option<&[bool]>) -> Result<Vec<f64>> {
        let (theta, nu) = self.split(x);
        let mut out = vec![0.0; self.blocks.len()];
        match only {
            None => {
                for (k, l) in self.row_loglik(&theta, &nu, like)?.into_iter().enumerate() {
                    out[self.branch[k]] += l;
                }
            }
            Some(only) => {
                let sel: Vec<usize> = (0..self.pulses.len()).filter(|&i| only[self.branch[self.owners[i].row as usize]]).collect();
                let pu: Vec<Pulse> = sel.iter().map(|&i| self.pulses[i]).collect();
                let ow: Vec<Owner> = sel.iter().map(|&i| self.owners[i]).collect();
                self.count(sel.len(), false);
                for (o, l) in ow.iter().zip(joint_loglik(&pu, &ow, &theta, &self.row_nu(&nu), like, self.threads)?) {
                    out[self.branch[o.row as usize]] += l;
                }
            }
        }
        for f in &self.factors {
            let r = f.vars.iter().map(|&(v, c)| c * x[v]).sum::<f64>() - f.mean;
            out[self.loc(f.vars[0].0).0] -= 0.5 * f.w * r * r;
        }
        for (i, w) in self.nu_w.iter().enumerate() {
            let v = 4 * self.n + i;
            out[self.loc(v).0] -= 0.5 * w * x[v] * x[v];
        }
        Ok(out)
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

    fn groups(&self, x: &[f64], like: &JointParams) -> Result<Groups> {
        let (theta, nu) = self.split(x);
        let (_, grad, hess) = joint_blocks(self.pulses, &self.owners, &self.group, self.n * self.s, &theta, &self.row_nu(&nu), like, self.threads)?;
        self.count(self.pulses.len(), true);
        Ok((grad, hess))
    }

    /// The groups that have pairs, with their row and scan.
    fn used(&self) -> Vec<(usize, usize, usize)> {
        let mut used = vec![false; self.n * self.s];
        for &q in &self.group {
            used[q as usize] = true;
        }
        (0..used.len()).filter(|&q| used[q]).map(|q| (q, q / self.s, q % self.s)).collect()
    }

    /// The Newton system at x from the likelihood's groups: their negated
    /// Hessians (`exact`), or the absolute ones (saddle-free), lifted onto
    /// their row and nuisance, plus the prior.
    fn system(&self, x: &[f64], groups: &Groups, exact: bool) -> System {
        let (grad, hess) = groups;
        let mut sys = self.empty_system();
        for (q, k, s) in self.used() {
            let hq = Matrix4::from(hess[q]);
            let h = if exact { -hq } else { abs_eig(&hq) };
            let (bi, rp) = (self.branch[k], 4 * self.rowpos[k]);
            let np = 4 * self.blocks[bi].rows.len() + 2 * s;
            // an offset moves the row's state with weight 1 and its branch's nuisance through the row's frame
            let t = &self.nu_t[k];
            let at = |a: usize| -> Vec<(usize, f64)> {
                if a < 2 && self.use_nu { vec![(rp + a, 1.0), (np, t[(a, 0)]), (np + 1, t[(a, 1)])] } else { vec![(rp + a, 1.0)] }
            };
            for a in 0..4 {
                for &(pa, ca) in &at(a) {
                    sys.g[bi][pa] += ca * grad[q][a];
                    for b in 0..4 {
                        for &(pb, cb) in &at(b) {
                            sys.h[bi][(pa, pb)] += ca * cb * h[(a, b)];
                        }
                    }
                }
            }
        }
        self.add_prior(&mut sys, x);
        sys
    }

    /// Solves `(H + mu diag H) x = g` on the branch tree: each block
    /// eliminated into its parent row, leaves first, then substituted back.
    /// With `cov`, also the diagonal of the inverse (top-down). None if a
    /// block isn't positive definite.
    #[allow(clippy::type_complexity)]
    fn solve(&self, sys: &System, mu: f64, cov: bool) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
        let nb = self.blocks.len();
        let mut h = sys.h.clone();
        let mut g = sys.g.clone();
        for m in h.iter_mut() {
            for i in 0..m.nrows() {
                m[(i, i)] *= 1.0 + mu;
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
        let x = self.gather(&xb);
        let var = cov.then(|| (0..self.n_var).map(|v| {
            let (b, p) = self.loc(v);
            sig[b][(p, p)]
        }).collect());
        Some((x, var))
    }

    /// Saddle-free Newton with a backtracking line search at one stage;
    /// returns the iterations taken and whether it converged (see
    /// [`FitOptions::tol`]; else the iterations ran out, or no step along the
    /// Newton direction improved the objective).
    fn newton(&self, x: &mut [f64], like: &JointParams, iters: usize, tol: f64, local: bool) -> Result<(usize, bool)> {
        const WINDOW: usize = 10;
        let mut obj = self.objective(x, like)?;
        if !obj.is_finite() {
            return Err(Error::invalid("joint fit: the objective isn't finite at the start"));
        }
        let mut history = vec![obj];
        // the local line search's per-block objective at the accepted step, the next iteration's base
        let mut next_base: Option<Vec<f64>> = None;
        for it in 0..iters {
            if history.len() > WINDOW && obj - history[history.len() - 1 - WINDOW] <= tol * (1.0 + obj.abs()) {
                return Ok((it, true));
            }
            let groups = self.groups(x, like)?;
            let sys = self.system(x, &groups, false);
            let g = self.gather(&sys.g);
            // the blocks are positive definite; a little ridge only if rounding says otherwise
            let step = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0].iter().find_map(|&mu| self.solve(&sys, mu, false))
                .ok_or_else(|| Error::invalid("joint fit: no Newton step (the system isn't positive definite)"))?.0;
            // the step as the radius caps leave it, and its decrement
            let mut to = x.iter().zip(&step).map(|(a, s)| a + s).collect::<Vec<f64>>();
            self.project(&mut to);
            let full: Vec<f64> = to.iter().zip(x.iter()).map(|(t, a)| t - a).collect();
            let dec: f64 = g.iter().zip(&full).map(|(g, d)| g * d).sum();
            if dec <= 2.0 * tol * (1.0 + obj.abs()) {
                return Ok((it, true));
            }
            // no unknown moves further than its cap in one step
            let scale = full.iter().zip(&self.cap).map(|(d, c)| d.abs() / c).fold(1.0, f64::max);
            let uniform = || (full.iter().map(|d| d / scale).collect::<Vec<f64>>(), dec / scale);
            let (dir, slope) = if local {
                // each row, and each nuisance pair, within its caps by its own scale
                let unit = |v: usize| if v < 4 * self.n { v / 4 } else { self.n + (v - 4 * self.n) / 2 };
                let mut us = vec![1.0f64; self.n + self.n_var];
                for (v, (d, c)) in full.iter().zip(&self.cap).enumerate() {
                    us[unit(v)] = us[unit(v)].max(d.abs() / c);
                }
                let dir: Vec<f64> = full.iter().enumerate().map(|(v, d)| d / us[unit(v)]).collect();
                let slope: f64 = g.iter().zip(&dir).map(|(g, d)| g * d).sum();
                if slope > 0.0 { (dir, slope) } else { uniform() }
            } else {
                uniform()
            };
            let obj0 = obj;
            let mut alpha = 1.0;
            let mut accepted = false;
            if local {
                // each branch block backtracks on its own share of the objective, then the whole step is checked
                let nbk = self.blocks.len();
                let mut slope_b = vec![0.0; nbk];
                for v in 0..self.n_var {
                    slope_b[self.loc(v).0] += g[v] * dir[v];
                }
                let base = match next_base.take() { Some(b) => b, None => self.block_objective(x, like, None)? };
                let (mut al, mut done) = (vec![1.0f64; nbk], vec![false; nbk]);
                let at = |al: &[f64]| -> Vec<f64> { (0..self.n_var).map(|v| x[v] + al[self.loc(v).0] * dir[v]).collect() };
                for pass in 0..40 {
                    let undone: Vec<bool> = done.iter().map(|d| !d).collect();
                    let ob = self.block_objective(&at(&al), like, (pass > 0).then_some(&undone[..]))?;
                    for b in (0..nbk).filter(|&b| !done[b]).collect::<Vec<_>>() {
                        // (a block's sums differ in rounding between the full and the subset evaluation)
                        if ob[b].is_finite() && ob[b] >= base[b] + 1e-4 * al[b] * slope_b[b] - 1e-10 * (1.0 + base[b].abs()) {
                            done[b] = true;
                        } else {
                            al[b] *= 0.5;
                            if al[b] < 1e-6 {
                                (al[b], done[b]) = (0.0, true);
                            }
                        }
                    }
                    if done.iter().all(|&d| d) {
                        break;
                    }
                }
                for b in (0..nbk).filter(|&b| !done[b]) {
                    al[b] = 0.0;
                }
                let trial = at(&al);
                let ob = self.block_objective(&trial, like, None)?;
                let o: f64 = ob.iter().sum();
                let s: f64 = al.iter().zip(&slope_b).map(|(a, s)| a * s).sum();
                if o.is_finite() && s > 0.0 && o >= obj + 1e-4 * s {
                    x.copy_from_slice(&trial);
                    obj = o;
                    history.push(o);
                    accepted = true;
                    alpha = al.iter().sum::<f64>() / nbk as f64;
                    next_base = Some(ob);
                }
            }
            for _ in 0..(if accepted { 0 } else { 40 }) {
                let trial: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + alpha * d).collect();
                let o = self.objective(&trial, like)?;
                if o.is_finite() && o >= obj + 1e-4 * alpha * slope {
                    x.copy_from_slice(&trial);
                    obj = o;
                    history.push(o);
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Ok((it, false));
            }
        }
        Ok((iters, false))
    }
}

/// `V |Lambda| V^T` of a symmetric 4x4: the curvature saddle-free Newton uses
/// for a log likelihood's Hessian (eigenvalues' magnitudes floored at 1e-6).
pub(crate) fn abs_eig(h: &Matrix4<f64>) -> Matrix4<f64> {
    let e = SymmetricEigen::new(*h);
    let d = Matrix4::from_diagonal(&e.eigenvalues.map(|l| l.abs().max(1e-6)));
    let out = e.eigenvectors * d * e.eigenvectors.transpose();
    if out.iter().all(|v| v.is_finite()) {
        return out;
    }
    // the decomposition can fail on a block of extreme range (a radius near zero): its diagonal's magnitudes
    Matrix4::from_diagonal(&h.diagonal().map(|l| if l.is_finite() { l.abs().max(1e-6) } else { 1e-6 }))
}

/// Fits the joint model to `pulses` (row and scan of each) on `rows`, from
/// `theta0` (per row; default the starting axes, the prior radii and the
/// prior foliage).
#[allow(clippy::too_many_arguments)]
pub fn joint_fit(pulses: &[Pulse], row: &[u32], scan: &[u32], rows: &Rows, n_scans: usize, like: &JointParams, prior: &PriorParams, opts: &FitOptions, theta0: Option<&[[f64; 4]]>) -> Result<Fit> {
    let pr = Problem::new(pulses, row, scan, rows, n_scans, prior, opts.threads)?;
    if theta0.is_some_and(|t| t.len() != pr.n) {
        return Err(Error::invalid("joint fit: theta0 must have one state per row"));
    }
    if opts.tol.is_nan() || opts.tol < 0.0 ||opts.schedule.iter().any(|s| !s.is_finite()) {
        return Err(Error::invalid("joint fit: tol and the schedule must be finite (tol not negative)"));
    }
    let mut x = vec![0.0; pr.n_var];
    for k in 0..pr.n {
        let t = theta0.map_or([0.0, 0.0, rows.r_prior[k].ln(), prior.lam_prior.ln()], |t| t[k]);
        x[4 * k..4 * k + 4].copy_from_slice(&t);
    }
    pr.project(&mut x);
    let mut stages: Vec<f64> = opts.schedule.iter().copied().filter(|&s| s > like.rough).collect();
    stages.push(like.rough);
    let (mut iterations, mut converged) = (Vec::new(), true);
    for rough in stages {
        let (it, ok) = pr.newton(&mut x, &JointParams { rough, ..*like }, opts.iters, opts.tol, opts.local)?;
        iterations.push(it);
        converged &= ok;
    }
    // Laplace: the posterior's variances on the branch tree, from the true Hessian where it is negative definite
    let groups = pr.groups(&x, like)?;
    let (var, exact) = match pr.solve(&pr.system(&x, &groups, true), 0.0, true) {
        Some((_, var)) => (var.unwrap(), true),
        None => {
            let (_, var) = pr.solve(&pr.system(&x, &groups, false), 0.0, true).ok_or_else(|| Error::invalid("joint fit: the posterior isn't positive definite"))?;
            (var.unwrap(), false)
        }
    };
    let mut prior_sys = pr.empty_system();
    pr.add_prior(&mut prior_sys, &x);
    let (_, pvar) = pr.solve(&prior_sys, 0.0, true).ok_or_else(|| Error::invalid("joint fit: the prior isn't positive definite"))?;
    let pvar = pvar.unwrap();
    let (theta, nu) = pr.split(&x);
    let sd = (0..pr.n).map(|k| std::array::from_fn(|j| var[4 * k + j].sqrt())).collect();
    // each row's message, and its information on eta with each scan's nuisance integrated out
    let (gq, hq) = &groups;
    let (mut grad, mut hess) = (vec![[0.0; 4]; pr.n], vec![[[0.0; 4]; 4]; pr.n]);
    let mut info = vec![Matrix4::zeros(); pr.n];
    let nb = pr.blocks.len();
    for (q, k, s) in pr.used() {
        for a in 0..4 {
            grad[k][a] += gq[q][a];
            for b in 0..4 {
                hess[k][a][b] += hq[q][a][b];
            }
        }
        let h = abs_eig(&Matrix4::from(hq[q]));
        info[k] += if pr.use_nu {
            let w = pr.nu_w[2 * (s * nb + pr.branch[k])];
            let t = pr.nu_t[k];
            let bm = h.fixed_columns::<2>(0).into_owned() * t;
            let c = t.transpose() * h.fixed_view::<2, 2>(0, 0).into_owned() * t + Matrix2::identity() * w;
            c.try_inverse().map_or(h, |ci| h - bm * ci * bm.transpose())
        } else {
            h
        };
    }
    let rho_local = (0..pr.n).map(|k| {
        // eta's precision from the row's own rays, its other unknowns integrated out within the row
        let h = &info[k];
        let o = [0, 1, 3];
        let hoo = Matrix3::from_fn(|i, j| h[(o[i], o[j])]) + Matrix3::identity() * 1e-9;
        let ho = nalgebra::Vector3::from_fn(|i, _| h[(o[i], 2)]);
        let tau = h[(2, 2)] - hoo.try_inverse().map_or(0.0, |inv| (ho.transpose() * inv * ho)[(0, 0)]);
        (tau * var[4 * k + 2]).clamp(0.0, 1.0)
    }).collect();
    let rho_pooled = (0..pr.n).map(|k| 1.0 - var[4 * k + 2] / pvar[4 * k + 2]).collect();
    let mut profile = Vec::new();
    if opts.profile {
        profile = vec![[0.0; 21]; pr.n];
        for (j, d) in profile_grid().into_iter().enumerate() {
            let moved: Vec<[f64; 4]> = theta.iter().map(|t| [t[0], t[1], t[2] + d, t[3]]).collect();
            for (k, l) in pr.row_loglik(&moved, &nu, like)?.into_iter().enumerate() {
                profile[k][j] = l;
            }
        }
    }
    let objective = pr.objective(&x, like)?;
    Ok(Fit { theta, nu: if pr.use_nu { nu } else { Vec::new() }, sd, rho_local, rho_pooled, grad, hess, profile, objective, iterations, converged, exact, evals: pr.evals.get() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qsm::joint::joint_table;

    fn prior() -> PriorParams {
        PriorParams { lam_prior: 1.0, lam_sd: 2.0, sig_ell: 1.0, sig_eta: 0.05, mu_fork: 0.0, sig_fork: 0.17, sig_root: 0.5, r_tip: 0.0, sig_tip: 0.3, sig_delta: 0.02, centre_rel: 0.3, centre_abs: 0.003, scan_abs: 0.002, scan_rel: 0.05, ..PriorParams::default() }
    }

    fn like() -> JointParams {
        JointParams { exit: 0.007, divergence: 0.35e-3, sigma_r: 0.005, rough: 0.03, tau: 0.17, edge: 0.25, eps: 0.1, eps_min: 0.01, mp_len: 0.5, hide: 0.0, ..JointParams::default() }
    }

    fn opts() -> FitOptions {
        FitOptions { schedule: vec![0.3, 0.15, 0.08], iters: 100, tol: 1e-10, threads: 2, local: false, profile: true }
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
        let rows = Rows { a: &a, b: &b, r_prior: &r, r_max: &[1.0; 7], parent: &parent };
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
        let rows = Rows { a: &a, b: &b, r_prior: &r, r_max: &[1.0; 7], parent: &parent };
        let f = joint_fit(&[], &[], &[], &rows, 2, &like(), &prior(), &opts(), None).unwrap();
        assert!(f.converged && f.exact);
        for (k, t) in f.theta.iter().enumerate() {
            assert!((t[2] - r[k].ln()).abs() < 1e-6 && t[0].abs() < 1e-9, "row {k}: {t:?}");
        }
        assert!(f.rho_local.iter().all(|&x| x == 0.0) && f.rho_pooled.iter().all(|x| x.abs() < 1e-9));
        assert!(f.profile.iter().all(|p| p.iter().all(|&l| l == 0.0)));
        // a tip anchor at 2 cm pulls the tips' radii; a cap holds the trunk under 4.9 cm
        let (tipped, cap) = (PriorParams { r_tip: 0.02, sig_tip: 0.1, ..prior() }, [0.049, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let g = joint_fit(&[], &[], &[], &Rows { r_max: &cap, ..rows }, 2, &like(), &tipped, &opts(), None).unwrap();
        assert!(g.converged && g.theta[0][2] <= 0.049f64.ln() + 1e-12);
        for k in [4, 6] {
            assert!(g.theta[k][2] < r[k].ln() - 0.05, "tip {k}: {}", g.theta[k][2].exp());
        }
        let wide = joint_fit(&[], &[], &[], &rows, 2, &like(), &PriorParams { mu_fork: 0.2, ..prior() }, &opts(), None).unwrap();
        assert!(wide.theta[3][2] > f.theta[3][2] && wide.theta[5][2] > f.theta[5][2], "a fork adding area thickens its children");
    }

    #[test]
    fn inputs_are_checked() {
        let (a, b, r, parent) = tree();
        let rows = Rows { a: &a, b: &b, r_prior: &r, r_max: &[1.0; 7], parent: &parent };
        let fit = |rows: &Rows, p: &PriorParams, t0: Option<&[[f64; 4]]>| joint_fit(&[], &[], &[], rows, 2, &like(), p, &opts(), t0);
        assert!(fit(&Rows { parent: &[-1, 0, 1, 2, 3, 6, 5], ..rows }, &prior(), None).is_err());
        assert!(fit(&rows, &prior(), Some(&[[0.0; 4]; 3])).is_err(), "theta0 of the wrong length");
        assert!(fit(&rows, &PriorParams { sig_eta: 0.0, ..prior() }, None).is_err());
        assert!(fit(&rows, &PriorParams { lam_prior: -1.0, ..prior() }, None).is_err());
        assert!(fit(&rows, &PriorParams { centre_rel: 0.0, centre_abs: 0.0, ..prior() }, None).is_err());
        assert!(fit(&rows, &PriorParams { r_tip: 0.01, sig_tip: 0.0, ..prior() }, None).is_err());
        let flat = [a[0]; 7];
        assert!(fit(&Rows { b: &flat, ..rows }, &prior(), None).is_err(), "rows without length");
        assert!(fit(&Rows { r_max: &[0.0; 7], ..rows }, &prior(), None).is_err());
    }

    #[test]
    fn a_fat_claim_on_a_scanned_stem_comes_down_to_it() {
        // the evidence tests' fan: a 5 cm stem at 5 m, claimed at 8 cm 1 cm off its axis
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let (a1, b1) = ([a[0] + 0.01, a[1], a[2]], [b[0] + 0.01, b[1], b[2]]);
        let t = joint_table(&[&scan], &[a1], &[b1], &[0.15], 2).unwrap();
        let rows = Rows { a: &[a1], b: &[b1], r_prior: &[0.08], r_max: &[0.15], parent: &[-1] };
        let p = PriorParams { scan_abs: 0.0, scan_rel: 0.0, ..prior() };
        let f = joint_fit(&t.pulses, &t.row, &t.scan, &rows, 1, &like(), &p, &opts(), None).unwrap();
        let r = f.theta[0][2].exp();
        assert!(f.converged && f.exact, "{:?}", f.iterations);
        assert!((r - 0.05).abs() < 0.003, "R {r}");
        let (_, e1, e2, _) = frames(&[a1], &[b1])[0];
        let off = [0, 1, 2].map(|i| f.theta[0][0] * e1[i] + f.theta[0][1] * e2[i]);
        assert!((off[0] + 0.01).abs() < 0.003 && off[1].abs() < 0.005, "offset {off:?}");
        assert!(f.sd[0][2] < 0.1 && f.rho_local[0] > 0.9, "sd {} rho {}", f.sd[0][2], f.rho_local[0]);
        // the profile peaks at the fit, and the message is the likelihood's curvature there
        let best = f.profile[0].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        assert!((best as i64 - 12).abs() <= 1, "profile peaks at offset {}", profile_grid()[best]);
        assert!(f.grad[0].iter().all(|g| g.is_finite()) && f.hess[0][2][2] < 0.0);
        // a cap below the stem holds the radius there
        let capped = joint_fit(&t.pulses, &t.row, &t.scan, &Rows { r_max: &[0.04], ..rows }, 1, &like(), &p, &opts(), None).unwrap();
        assert!((capped.theta[0][2].exp() - 0.04).abs() < 1e-9);
        // local step control reaches the same fit
        let local = joint_fit(&t.pulses, &t.row, &t.scan, &rows, 1, &like(), &p, &FitOptions { local: true, ..opts() }, None).unwrap();
        assert!(local.converged && (local.theta[0][2].exp() - r).abs() < 1e-3, "R {} vs {r}", local.theta[0][2].exp());
    }

    #[test]
    fn a_branch_nuisance_is_one_world_offset_along_a_bent_branch() {
        // a stem bent past the frames' reference switch (|u_z| 0.9), so its two rows' frames turn
        let (scan, a, b) = crate::qsm::evidence::tests::scene(0.05, true);
        let mid: Point = std::array::from_fn(|i| 0.5 * (a[i] + b[i]));
        let h = 0.5 * (b[2] - a[2]);
        let top = [mid[0] + 0.6 * h, mid[1], mid[2] + 0.8 * h];
        let (ra, rb) = (vec![a, mid], vec![mid, top]);
        let t = joint_table(&[&scan], &ra, &rb, &[0.15, 0.15], 2).unwrap();
        let rows = Rows { a: &ra, b: &rb, r_prior: &[0.05, 0.05], r_max: &[0.15, 0.15], parent: &[-1, 0] };
        let pr = Problem::new(&t.pulses, &t.row, &t.scan, &rows, 1, &prior(), 2).unwrap();
        assert!(pr.use_nu && pr.blocks.len() == 1);
        // the same world offset in both rows (each sees its share across its own axis)
        let fr = frames(&ra, &rb);
        assert!(fr[0].1[0] * fr[1].1[0] + fr[0].1[1] * fr[1].1[1] + fr[0].1[2] * fr[1].1[2] < 0.0, "the frames turn (here e1 flips)");
        let nu = [[0.004, -0.003]];
        let w: Point = std::array::from_fn(|i| nu[0][0] * fr[0].1[i] + nu[0][1] * fr[0].2[i]);
        for (k, v) in pr.row_nu(&nu).iter().enumerate() {
            let (u, e1, e2, _) = fr[k];
            let along: f64 = (0..3).map(|i| w[i] * u[i]).sum();
            for i in 0..3 {
                assert!((v[0] * e1[i] + v[1] * e2[i] - (w[i] - along * u[i])).abs() < 1e-12, "row {k}");
            }
        }
        // the system's gradient in the nuisance is the objective's
        let mut x = vec![0.0; pr.n_var];
        for k in 0..2 {
            x[4 * k..4 * k + 4].copy_from_slice(&[0.003, -0.002, (0.05f64).ln(), 0.0]);
        }
        x[8] = 0.002;
        x[9] = 0.001;
        let sys = pr.system(&x, &pr.groups(&x, &like()).unwrap(), true);
        let g = pr.gather(&sys.g);
        for v in [0, 1, 4, 5, 8, 9] {
            let e = 1e-6;
            let (mut up, mut dn) = (x.clone(), x.clone());
            up[v] += e;
            dn[v] -= e;
            let fd = (pr.objective(&up, &like()).unwrap() - pr.objective(&dn, &like()).unwrap()) / (2.0 * e);
            assert!((g[v] - fd).abs() < 1e-4 * (1.0 + fd.abs()), "d/dx[{v}] {} vs {fd}", g[v]);
        }
    }
}
