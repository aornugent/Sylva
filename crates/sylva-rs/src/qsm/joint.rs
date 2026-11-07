// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The per-pulse likelihood of a joint wood, foliage and ray model, with its
//! exact gradient and Hessian: the hot path of a fit whose orchestration
//! (continuation, the tree prior, the sparse solve) stays outside.
//!
//! Each row of the table is a pulse that reached the disc around a cylinder
//! (see [`crate::qsm::evidence::pairs`]), measured once from a starting axis:
//! signed impact `b`, range of closest approach `t`, `sin` of its angle to
//! the axis, how both move with an offset of the axis (`jb`, `jt`), the
//! disc's chord `[c0, c1]` along the pulse, and its fate there (a stop at
//! range `x`, or a pass). Its state is the axis offset across the axis
//! (two numbers, the cylinder's own plus its scan's nuisance), `eta =
//! log R` and `ell = log lambda` (foliage per metre). Along the pulse,
//! foliage and opaque wood compete to stop it; the wood's echo sits at the
//! centroid depth of the circle over a uniform footprint, Gaussian about it
//! or a mixed pixel behind; a pass is censoring at the chord's end. Each
//! pulse is conditioned on having been recorded: a pass needs something
//! behind (`q`), and an export thinned to one echo per `thin` metres keeps a
//! stop at range `x` with `min(1, (x step / thin)^2)`.
//!
//! The derivatives come from evaluating the likelihood once on second-order
//! dual numbers in the four state variables, so a term added to the
//! likelihood needs no derivative of its own.

use std::ops::{Add, Div, Mul, Neg, Sub};

use rayon::prelude::*;

use crate::error::{Error, Result};

/// A value with its gradient and Hessian in four variables.
#[derive(Debug, Clone, Copy, PartialEq)]
struct D {
    v: f64,
    g: [f64; 4],
    h: [[f64; 4]; 4],
}

impl D {
    fn c(v: f64) -> D {
        D { v, g: [0.0; 4], h: [[0.0; 4]; 4] }
    }

    fn var(v: f64, i: usize) -> D {
        let mut d = D::c(v);
        d.g[i] = 1.0;
        d
    }

    /// `f(self)` from `f`, `f'` and `f''` at `self.v`.
    fn chain(self, f: f64, f1: f64, f2: f64) -> D {
        let mut out = D::c(f);
        for i in 0..4 {
            out.g[i] = f1 * self.g[i];
            for j in 0..4 {
                out.h[i][j] = f2 * self.g[i] * self.g[j] + f1 * self.h[i][j];
            }
        }
        out
    }

    fn exp(self) -> D {
        let e = self.v.exp();
        self.chain(e, e, e)
    }

    fn ln(self) -> D {
        self.chain(self.v.ln(), 1.0 / self.v, -1.0 / (self.v * self.v))
    }

    fn sqrt(self) -> D {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s, -0.25 / (s * s * s))
    }

    fn asin(self) -> D {
        let one = (1.0 - self.v * self.v).max(1e-300);
        self.chain(self.v.asin(), one.powf(-0.5), self.v * one.powf(-1.5))
    }

    fn log1p(self) -> D {
        let a = 1.0 + self.v;
        self.chain(self.v.ln_1p(), 1.0 / a, -1.0 / (a * a))
    }

    fn sq(self) -> D {
        self * self
    }

    fn abs(self) -> D {
        if self.v < 0.0 { -self } else { self }
    }

    fn max(self, o: D) -> D {
        if o.v > self.v { o } else { self }
    }

    fn min(self, o: D) -> D {
        if o.v < self.v { o } else { self }
    }

    fn clamp(self, lo: D, hi: D) -> D {
        self.max(lo).min(hi)
    }

    /// log Phi(z), the standard normal's log CDF, stable far into both tails.
    fn log_ndtr(self) -> D {
        let (lphi, m) = log_ndtr_mills(self.v);
        self.chain(lphi, m, -m * (self.v + m))
    }
}

/// `log(exp(a) + exp(b))`.
fn logaddexp(a: D, b: D) -> D {
    let (hi, lo) = if a.v >= b.v { (a, b) } else { (b, a) };
    if lo.v == f64::NEG_INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().log1p()
}

/// `(log Phi(z), phi(z) / Phi(z))`: Phi from musl's erfc (`libm`), and below
/// z = -37, where erfc underflows, the asymptotic series of the Mills ratio;
/// the ratio from the logs.
fn log_ndtr_mills(z: f64) -> (f64, f64) {
    use std::f64::consts::{PI, SQRT_2};
    let log_pdf = -0.5 * z * z - 0.5 * (2.0 * PI).ln();
    let lphi = if z > 0.0 {
        (-0.5 * libm::erfc(z / SQRT_2)).ln_1p()
    } else if z > -37.0 {
        (0.5 * libm::erfc(-z / SQRT_2)).ln()
    } else {
        let w = 1.0 / (z * z);
        log_pdf - (-z).ln() + (1.0 - w + 3.0 * w * w - 15.0 * w * w * w).ln()
    };
    (lphi, (log_pdf - lphi).exp())
}

impl Add for D {
    type Output = D;
    fn add(self, o: D) -> D {
        let mut r = self;
        r.v += o.v;
        for i in 0..4 {
            r.g[i] += o.g[i];
            for j in 0..4 {
                r.h[i][j] += o.h[i][j];
            }
        }
        r
    }
}

impl Sub for D {
    type Output = D;
    fn sub(self, o: D) -> D {
        self + (-o)
    }
}

impl Neg for D {
    type Output = D;
    fn neg(self) -> D {
        let mut r = self;
        r.v = -r.v;
        for i in 0..4 {
            r.g[i] = -r.g[i];
            for j in 0..4 {
                r.h[i][j] = -r.h[i][j];
            }
        }
        r
    }
}

impl Mul for D {
    type Output = D;
    fn mul(self, o: D) -> D {
        let mut r = D::c(self.v * o.v);
        for i in 0..4 {
            r.g[i] = self.v * o.g[i] + o.v * self.g[i];
            for j in 0..4 {
                r.h[i][j] = self.v * o.h[i][j] + o.v * self.h[i][j] + self.g[i] * o.g[j] + o.g[i] * self.g[j];
            }
        }
        r
    }
}

impl Div for D {
    type Output = D;
    fn div(self, o: D) -> D {
        let inv = o.chain(1.0 / o.v, -1.0 / (o.v * o.v), 2.0 / (o.v * o.v * o.v));
        self * inv
    }
}

impl Add<f64> for D {
    type Output = D;
    fn add(self, o: f64) -> D {
        let mut r = self;
        r.v += o;
        r
    }
}

impl Mul<f64> for D {
    type Output = D;
    fn mul(self, o: f64) -> D {
        let mut r = self;
        r.v *= o;
        for i in 0..4 {
            r.g[i] *= o;
            for j in 0..4 {
                r.h[i][j] *= o;
            }
        }
        r
    }
}

/// The scanner and export constants of the likelihood (a plot calibration).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointParams {
    /// Beam diameter at the aperture (m) and full divergence (rad).
    pub exit: f64,
    pub divergence: f64,
    /// Range noise (m), and surface roughness as a share of the radius.
    pub sigma_r: f64,
    pub rough: f64,
    /// Energy share of the footprint an echo needs.
    pub tau: f64,
    /// Mixed-pixel share: `eps_min + eps (1 - f)`, f the footprint's share on the wood, landing up to
    /// `mp_len` m behind the surface.
    pub eps: f64,
    pub eps_min: f64,
    pub mp_len: f64,
    /// The export's thinning (m; 0: none) and the scan's angular step (rad).
    pub thin: f64,
    pub step: f64,
}

/// The table of pulse-cylinder pairs, one entry per pair (`jb`, `jt` two
/// per pair).
#[derive(Debug, Clone, Default)]
pub struct JointTable {
    pub b: Vec<f64>,
    pub t: Vec<f64>,
    pub sin: Vec<f64>,
    pub jb: Vec<[f64; 2]>,
    pub jt: Vec<[f64; 2]>,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub stop: Vec<bool>,
    pub x: Vec<f64>,
    pub early: Vec<bool>,
    pub q: Vec<f64>,
}

impl JointTable {
    fn check(&self) -> Result<()> {
        let n = self.b.len();
        let lens = [self.t.len(), self.sin.len(), self.jb.len(), self.jt.len(), self.c0.len(), self.c1.len(), self.stop.len(), self.x.len(), self.early.len(), self.q.len()];
        if lens.iter().any(|&l| l != n) {
            return Err(Error::invalid("joint: the table's columns must have one entry per pair"));
        }
        Ok(())
    }
}

/// `∫ sqrt(R^2 - y^2) dy` at `y` clipped to the circle, exact in value and
/// first derivatives (d/dy = sqrt(R^2 - y^2), d/dR = R asin(y / R)). The
/// depth's curvature is infinite at the silhouette, so only the second
/// derivatives floor that root at 1e-6 R.
fn h_int(y: D, r: D) -> D {
    let yc = y.clamp(-r, r);
    let (yv, rv) = (yc.v, r.v);
    let root = (rv * rv - yv * yv).max(0.0).sqrt();
    let ratio = (yv / rv).clamp(-1.0, 1.0);
    let f = 0.5 * (yv * root + rv * rv * ratio.asin());
    let (fy, fr) = (root, rv * ratio.asin());
    let soft = root.max(1e-6 * rv);
    let (fyy, fyr, frr) = (-yv / soft, rv / soft, ratio.asin() - yv / soft);
    let mut out = D::c(f);
    for a in 0..4 {
        out.g[a] = fy * yc.g[a] + fr * r.g[a];
        for b in 0..4 {
            out.h[a][b] = fyy * yc.g[a] * yc.g[b] + fyr * (yc.g[a] * r.g[b] + r.g[a] * yc.g[b]) + frr * r.g[a] * r.g[b]
                + fy * yc.h[a][b] + fr * r.h[a][b];
        }
    }
    out
}

/// One pair's log likelihood at state (offset, eta, ell), with derivatives.
fn pair(tab: &JointTable, i: usize, s: [f64; 4], p: &JointParams) -> D {
    let (d1, d2, eta, ell) = (D::var(s[0], 0), D::var(s[1], 1), D::var(s[2], 2), D::var(s[3], 3));
    let r = eta.exp();
    let lam = ell.exp();
    let bp = D::c(tab.b[i]) - (d1 * tab.jb[i][0] + d2 * tab.jb[i][1]);
    let tp = D::c(tab.t[i]) + (d1 * tab.jt[i][0] + d2 * tab.jt[i][1]);
    let w = p.exit + p.divergence * tab.t[i];
    // the footprint on the circle: its share f, and the surface's centroid depth hbar
    let y1 = (bp + (-w / 2.0)).max(-r);
    let y2 = (bp + w / 2.0).min(r);
    let width = (y2 - y1).max(D::c(0.0));
    let f = width * (1.0 / w);
    let hbar = if width.v > 1e-9 { (h_int(y2, r) - h_int(y1, r)) / width } else { D::c(0.0) };
    let zdet = (r + w * (0.5 - p.tau) - bp.abs()) * (4.0 / w);
    let (lp, lq) = (zdet.log_ndtr(), (-zdet).log_ndtr());
    let (c0, c1) = (D::c(tab.c0[i]), D::c(tab.c1[i]));
    let xw = (tp - hbar * (1.0 / tab.sin[i])).clamp(c0, c1);
    let zw = xw - c0;
    let ln = tab.c1[i] - tab.c0[i];
    let sig = (r.sq() * (p.rough * p.rough) + p.sigma_r * p.sigma_r).sqrt();
    let eps = (-(f.clamp(D::c(0.0), D::c(1.0))) + 1.0) * p.eps + p.eps_min;
    let x = D::c(tab.x[i]);
    let z = (x - c0).max(D::c(0.0));
    let u = (x - xw) / sig;
    let log_sqrt2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let l_gauss = (-eps).log1p() - u.sq() * 0.5 - sig.ln() + (-log_sqrt2pi);
    let l_tail = (eps + 1e-300).ln() + (-p.mp_len.ln()) + u.log_ndtr() + ((xw + p.mp_len - x) / sig).log_ndtr();
    let l_wood = lp - lam * zw + logaddexp(l_gauss, l_tail);
    let l_leaf = ell - lam * z + logaddexp(lq, lp + (-u).log_ndtr());
    let over = ((xw + p.mp_len - c1) * (1.0 / p.mp_len)).clamp(D::c(1e-12), D::c(1.0));
    let l_pass = logaddexp(lq - lam * ln, lp - lam * zw + (eps + 1e-300).ln() + over.ln());
    let l_stop = logaddexp(l_wood, l_leaf);
    let q = tab.q[i];
    let p_pass = l_pass.min(D::c(0.0)).exp();
    if p.thin > 0.0 {
        let lk = (2.0 * (tab.t[i].max(0.1) * p.step / p.thin).ln()).min(0.0);
        let sel = if tab.early[i] { D::c(0.0) } else { ((-p_pass + 1.0) * lk.exp() + p_pass * q).ln() };
        (if tab.stop[i] { l_stop + lk } else { l_pass + q.ln() }) - sel
    } else {
        let sel = if tab.early[i] { D::c(0.0) } else { (p_pass * (-(1.0 - q))).log1p() };
        (if tab.stop[i] { l_stop } else { l_pass + q.ln() }) - sel
    }
}

/// Per pair log likelihood at `state[i] = [offset1, offset2, eta, ell]`.
pub fn joint_loglik(tab: &JointTable, state: &[[f64; 4]], p: &JointParams, threads: usize) -> Result<Vec<f64>> {
    tab.check()?;
    if state.len() != tab.b.len() {
        return Err(Error::invalid("joint: one state per pair"));
    }
    pool(threads)?.install(|| Ok((0..tab.b.len()).into_par_iter().map(|i| pair(tab, i, state[i], p).v).collect()))
}

/// Sums by group of the pairs' log likelihood, gradient and Hessian in their
/// own four state variables: `(value (n,), gradient (n, 4), Hessian (n, 4,
/// 4))`, group `group[i]` of `n_groups`.
#[allow(clippy::type_complexity)]
pub fn joint_blocks(tab: &JointTable, state: &[[f64; 4]], group: &[u32], n_groups: usize, p: &JointParams, threads: usize) -> Result<(Vec<f64>, Vec<[f64; 4]>, Vec<[[f64; 4]; 4]>)> {
    tab.check()?;
    let n = tab.b.len();
    if state.len() != n || group.len() != n || group.iter().any(|&g| g as usize >= n_groups) {
        return Err(Error::invalid("joint: one state and one group (below n_groups) per pair"));
    }
    pool(threads)?.install(|| {
        let empty = || (vec![0.0; n_groups], vec![[0.0; 4]; n_groups], vec![[[0.0; 4]; 4]; n_groups]);
        Ok((0..n).into_par_iter().fold(empty, |mut acc, i| {
            let d = pair(tab, i, state[i], p);
            let k = group[i] as usize;
            acc.0[k] += d.v;
            for a in 0..4 {
                acc.1[k][a] += d.g[a];
                for b in 0..4 {
                    acc.2[k][a][b] += d.h[a][b];
                }
            }
            acc
        }).reduce(empty, |mut x, y| {
            for k in 0..n_groups {
                x.0[k] += y.0[k];
                for a in 0..4 {
                    x.1[k][a] += y.1[k][a];
                    for b in 0..4 {
                        x.2[k][a][b] += y.2[k][a][b];
                    }
                }
            }
            x
        }))
    })
}

fn pool(threads: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| Error::invalid(format!("thread pool: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> JointParams {
        JointParams { exit: 0.007, divergence: 0.35e-3, sigma_r: 0.005, rough: 0.03, tau: 0.17, eps: 0.1, eps_min: 0.01, mp_len: 0.5, thin: 0.0, step: 0.03f64.to_radians() }
    }

    /// A few pulses past a 5 cm stem at 8 m: on the surface, inside it, passing, and a leaf in front.
    fn table() -> JointTable {
        let mut t = JointTable::default();
        for (b, stop, x, early, q) in [(0.0, true, 7.95, false, 1.0), (0.04, true, 7.97, false, 0.6), (0.07, false, 0.0, false, 0.6), (0.02, true, 7.88, true, 1.0), (0.03, false, 0.0, false, 1.0)] {
            t.b.push(b);
            t.t.push(8.0);
            t.sin.push(0.95);
            t.jb.push([0.9, 0.1]);
            t.jt.push([0.05, 0.9]);
            t.c0.push(7.85);
            t.c1.push(8.15);
            t.stop.push(stop);
            t.x.push(x);
            t.early.push(early);
            t.q.push(q);
        }
        t
    }

    #[test]
    fn log_ndtr_is_right_in_both_tails() {
        // scipy.special.log_ndtr
        for (z, want) in [(0.0, 0.5f64.ln()), (2.0, -0.023012909328963476), (-3.0, -6.60772622151035), (-30.0, -454.32124395634327), (-40.0, -804.6084420137539), (-50.0, -1254.8313611394199)] {
            let (got, _) = log_ndtr_mills(z);
            assert!((got - want).abs() < 1e-10 * want.abs().max(1.0), "z {z}: {got} vs {want}");
        }
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let (tab, p) = (table(), params());
        let s0 = [0.003, -0.002, (0.05f64).ln(), (1.5f64).ln()];
        for i in 0..tab.b.len() {
            let d = pair(&tab, i, s0, &p);
            let h = [1e-5, 1e-5, 1e-4, 1e-4];
            for a in 0..4 {
                let (mut sp, mut sm) = (s0, s0);
                sp[a] += h[a];
                sm[a] -= h[a];
                let (fp, fm) = (pair(&tab, i, sp, &p), pair(&tab, i, sm, &p));
                let g = (fp.v - fm.v) / (2.0 * h[a]);
                assert!((g - d.g[a]).abs() < 1e-3 * g.abs().max(1.0), "pair {i} grad {a}: {g} vs {}", d.g[a]);
                for b in 0..4 {
                    let hb = (fp.g[b] - fm.g[b]) / (2.0 * h[a]);
                    assert!((hb - d.h[a][b]).abs() < 1e-3 * hb.abs().max(10.0), "pair {i} hess {a}{b}: {hb} vs {}", d.h[a][b]);
                }
            }
        }
    }

    #[test]
    fn blocks_sum_their_pairs_and_inputs_are_checked() {
        let (tab, p) = (table(), params());
        let st = vec![[0.0, 0.0, (0.05f64).ln(), 0.0]; tab.b.len()];
        let (f, g, _) = joint_blocks(&tab, &st, &[0, 1, 1, 0, 1], 2, &p, 2).unwrap();
        let ll = joint_loglik(&tab, &st, &p, 2).unwrap();
        assert!((f[0] - (ll[0] + ll[3])).abs() < 1e-9 && (f[1] - (ll[1] + ll[2] + ll[4])).abs() < 1e-9);
        assert!(g.iter().all(|x| x.iter().all(|v| v.is_finite())));
        assert!(joint_blocks(&tab, &st, &[0, 1, 2, 0, 1], 2, &p, 1).is_err());
        assert!(joint_loglik(&tab, &st[..2], &p, 1).is_err());
    }
}
