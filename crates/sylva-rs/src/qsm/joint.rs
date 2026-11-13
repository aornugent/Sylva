// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The per-pulse likelihood of a joint wood, foliage and ray model, with its
//! exact gradient and Hessian, and its pair table; [`crate::qsm::joint_fit`]
//! fits it.
//!
//! Each row of the table is a pulse that reached the tube around a cylinder
//! (see [`crate::qsm::scan_index::pairs`]), measured once from a starting axis:
//! signed impact `b`, range of closest approach `t`, `sin` of its angle to
//! the axis, how both move with an offset of the axis (`jb`, `jt`), the
//! disc's chord `[c0, c1]` along the pulse, and its fate there (a stop at
//! range `x`, or a pass). Its state is the axis offset across the axis
//! (two numbers, the cylinder's own plus its scan's nuisance), `eta =
//! log R` and `ell = log lambda` (foliage per metre). Along the pulse,
//! foliage and opaque wood compete to stop it; the wood's echo sits at the
//! centroid depth of the circle over a uniform footprint (a sliver narrower
//! than a fiftieth of it averaged as if that wide), Gaussian about it
//! or a mixed pixel behind; a pass is censoring at the chord's end. Each
//! pulse is conditioned on having been recorded: a pass needs something
//! behind (`q`). A stop needs the footprint's share [`TAU`] on the wood, with an
//! edge [`EDGE`] `w` wide (w the footprint's width), and wood wide enough to give
//! that share at all.
//!
//! The derivatives come from evaluating the likelihood once on second-order
//! dual numbers in the four state variables, so a term added to the
//! likelihood needs no derivative of its own.

use std::ops::{Add, Div, Mul, Neg, Sub};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::qsm::scan_index::pool;
use crate::transform::cross;
use crate::Point;

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

    fn log1p(self) -> D {
        let a = 1.0 + self.v;
        self.chain(self.v.ln_1p(), 1.0 / a, -1.0 / (a * a))
    }

    fn abs(self) -> D {
        if self.v < 0.0 { -self } else { self }
    }

    /// log Phi(z), the standard normal's log CDF, stable far into both tails.
    fn log_ndtr(self) -> D {
        let (lphi, m) = log_ndtr_mills(self.v);
        self.chain(lphi, m, -m * (self.v + m))
    }
}

/// What the likelihood is written in: a plain value (`f64`, for the
/// objective) or a value carrying its gradient and Hessian (`D`, for the
/// Newton blocks). One source for both, so they can't drift apart; where a
/// choice is made (max, min, clamp at a tie) both make the same one.
trait Num: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + Add<f64, Output = Self> + Mul<f64, Output = Self> {
    fn c(v: f64) -> Self;
    fn v(&self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn log1p(self) -> Self;
    fn abs(self) -> Self;
    fn log_ndtr(self) -> Self;
    /// `∫ sqrt(r^2 - y^2) dy` at `y` clipped to the circle (see [`h_int`]).
    fn h_int(y: Self, r: Self) -> Self;
    fn max(self, o: Self) -> Self {
        if o.v() > self.v() { o } else { self }
    }
    fn min(self, o: Self) -> Self {
        if o.v() < self.v() { o } else { self }
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    fn sq(self) -> Self {
        self * self
    }
}

impl Num for f64 {
    fn c(v: f64) -> f64 {
        v
    }
    fn v(&self) -> f64 {
        *self
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn log1p(self) -> f64 {
        self.ln_1p()
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn log_ndtr(self) -> f64 {
        log_ndtr_value(self)
    }
    fn h_int(y: f64, r: f64) -> f64 {
        let yc = Num::clamp(y, -r, r);
        let ratio = Num::clamp(yc / r, -1.0, 1.0);
        0.5 * (yc * (r * r - yc * yc).max(0.0).sqrt() + r * r * ratio.asin())
    }
}

impl Num for D {
    fn c(v: f64) -> D {
        D::c(v)
    }
    fn v(&self) -> f64 {
        self.v
    }
    fn exp(self) -> D {
        D::exp(self)
    }
    fn ln(self) -> D {
        D::ln(self)
    }
    fn sqrt(self) -> D {
        D::sqrt(self)
    }
    fn log1p(self) -> D {
        D::log1p(self)
    }
    fn abs(self) -> D {
        D::abs(self)
    }
    fn log_ndtr(self) -> D {
        D::log_ndtr(self)
    }
    fn h_int(y: D, r: D) -> D {
        h_int(y, r)
    }
}

/// `log(exp(a) + exp(b))`.
fn logaddexp<N: Num>(a: N, b: N) -> N {
    let (hi, lo) = if a.v() >= b.v() { (a, b) } else { (b, a) };
    if lo.v() == f64::NEG_INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().log1p()
}

/// `(log Phi(z), phi(z) / Phi(z))`: Phi from musl's erfc (`libm`), and below
/// z = -37, where erfc underflows, the asymptotic series of the Mills ratio;
/// the ratio from the logs.
fn log_ndtr_mills(z: f64) -> (f64, f64) {
    let lphi = log_ndtr_value(z);
    (lphi, (log_pdf(z) - lphi).exp())
}

fn log_pdf(z: f64) -> f64 {
    -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// `log Phi(z)` alone (see [`log_ndtr_mills`]).
pub(crate) fn log_ndtr_value(z: f64) -> f64 {
    use std::f64::consts::SQRT_2;
    if z > 0.0 {
        // erfc is exactly 0 from 28 on (libm), so log Phi is exactly -0 there: skip the call
        let x = z / SQRT_2;
        if x >= 28.0 {
            return -0.0;
        }
        (-0.5 * libm::erfc(x)).ln_1p()
    } else if z > -37.0 {
        (0.5 * libm::erfc(-z / SQRT_2)).ln()
    } else {
        let w = 1.0 / (z * z);
        log_pdf(z) - (-z).ln() + (1.0 - w + 3.0 * w * w - 15.0 * w * w * w).ln()
    }
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

params! {
    /// The settings of [`crate::qsm::support_tree`] a caller may set; every other calibration is a constant (here,
    /// in [`crate::qsm::joint_fit`] and in [`crate::qsm::support_tree`]).
    ///
    /// The defaults are the calibration for TERN's RIEGL VZ-400i exports (3dtrees research 100 §§13-19 and 93 §§11.2, 17: Alice Mulga).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct JointParams {
        /// Beam diameter at the aperture (m) and full divergence (rad).
        pub exit: f64 = 0.007,
        pub divergence: f64 = 0.35e-3,
        /// Echo character (3dtrees research 103 §11.2): the reflectance (dB) of a single return from bark that fills
        /// the footprint and from leaf, mean and sd; a return that shares its pulse is `refl_multi` dB brighter (it
        /// is negative) with `refl_multi_sd` more spread, in quadrature. Read by [`crate::qsm::support_tree`] only.
        pub refl_wood: f64 = -15.45,
        pub refl_wood_sd: f64 = 1.0,
        pub refl_leaf: f64 = -19.7,
        pub refl_leaf_sd: f64 = 1.4,
        pub refl_multi: f64 = -3.0,
        pub refl_multi_sd: f64 = 1.2,
        /// Thin wood reads darker than bark that fills the footprint even above the footprint's width (3dtrees
        /// research 103 §16: rows under 2 cm at -17.8 dB p50 against -15.9): wood's mean shifts by `refl_thin` dB at
        /// R <= 1 cm, by none from 3 cm, linearly in log R between. Read by [`crate::qsm::support_tree`] only.
        pub refl_thin: f64 = -3.0,
        /// The site's largest stem D/H: no radius the rays don't measure is wider than `dh` times its reach to the tips
        /// over 2 (0: no cap; Alice Mulga's census 0.035, 3dtrees research 62). Read by [`crate::qsm::support_tree`] only.
        pub dh: f64 = 0.0,
    }
}

/// Range noise (m), and surface roughness as a share of the radius.
pub(crate) const SIGMA_R: f64 = 0.005;
pub(crate) const ROUGH: f64 = 0.03;
/// Energy share of the footprint an echo needs, and the sd of the detection edge as a share of the footprint's width.
pub(crate) const TAU: f64 = 0.11;
pub(crate) const EDGE: f64 = 0.5;
/// Mixed-pixel share: `EPS_MIN + EPS (1 - f)`, f the footprint's share on the wood, landing up to `MP_LEN` m behind
/// the surface.
const EPS: f64 = 0.1;
const EPS_MIN: f64 = 0.01;
const MP_LEN: f64 = 0.5;

impl JointParams {
    /// The footprint's width (m) at range `t`.
    pub fn footprint(&self, t: f64) -> f64 {
        self.exit + self.divergence * t
    }

    /// How far a silhouette's apparent edge lies beyond the wood's for a footprint `w` wide: `w (1/2 - TAU)`.
    pub fn app(&self, w: f64) -> f64 {
        w * (0.5 - TAU)
    }
}

/// One pulse-cylinder pair, measured once from a starting axis (see the
/// module documentation).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pulse {
    pub b: f64,
    pub t: f64,
    pub sin: f64,
    pub jb: [f64; 2],
    pub jt: [f64; 2],
    pub c0: f64,
    pub c1: f64,
    /// A stop's range (unused for a pass).
    pub x: f64,
    pub stop: bool,
    /// It echoed before the disc (recorded whatever happened in it).
    pub early: bool,
    /// The chance something behind records a pass.
    pub q: f64,
}

/// `∫ sqrt(R^2 - y^2) dy` at `y` clipped to the circle, exact in value and
/// first derivatives (d/dy = sqrt(R^2 - y^2), d/dR = R asin(y / R)). The
/// depth's curvature is infinite at the silhouette, so only the second
/// derivatives floor that root at 1e-6 R.
fn h_int(y: D, r: D) -> D {
    let yc = Num::clamp(y, -r, r);
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

/// The width (a share of the footprint's) below which a sliver of the footprint
/// on the circle is averaged as if that wide: the centroid depth is then smooth
/// at the silhouette. Wider (a quarter of the footprint) moved the bench's
/// trunks 0.8 % fatter (research 100 §18).
const SLIVER: f64 = 0.02;

/// The softness (a share of the footprint) of the wood's capacity to make an
/// echo: wood of width 2 R gives at most the share 2 R / w, and needs tau.
/// Without it a radius near 0 still detects pulses near its axis (research
/// 100 §20), so "no wood" wasn't.
pub(crate) const CAPACITY_SD: f64 = 0.02;

/// One pair's log likelihood at state `[offset1, offset2, eta, ell]`, in
/// whatever [`Num`] the state is given in.
fn pair<N: Num>(pu: &Pulse, s: [N; 4], p: &JointParams) -> N {
    pair_terms(pu, s, p).0
}

/// [`pair`], and for a stop its wood and leaf terms (`l_wood`, `l_leaf`: its echo the wood's or foliage's, before
/// selection), whose softmax is the stop's chance of being the wood's echo.
#[inline(always)]
fn pair_terms<N: Num>(pu: &Pulse, s: [N; 4], p: &JointParams) -> (N, Option<(N, N)>) {
    let [d1, d2, eta, ell] = s;
    let r = eta.exp();
    let lam = ell.exp();
    let bp = N::c(pu.b) - (d1 * pu.jb[0] + d2 * pu.jb[1]);
    let tp = N::c(pu.t) + (d1 * pu.jt[0] + d2 * pu.jt[1]);
    let w = p.footprint(pu.t);
    // the footprint on the circle: its share f, and the surface's centroid depth hbar. A sliver's centroid
    // goes as the root of its width, an infinite slope at the silhouette that traps Newton (research 100 §18):
    // slivers narrower than SLIVER w are averaged over a width that doesn't vanish
    let y1 = (bp + (-w / 2.0)).max(-r);
    let y2 = (bp + w / 2.0).min(r);
    let width = (y2 - y1).max(N::c(0.0));
    let f = width * (1.0 / w);
    let e = SLIVER * w;
    let hbar = (N::h_int(y2, r) - N::h_int(y1, r)) / (width + (width * (-1.0 / e)).exp() * e);
    // detected: the footprint's share on the wood reaches tau (a soft edge at the silhouette), and the wood is wide
    // enough to give that share at all (2 R / w >= tau: a twig thinner than that can't make an echo, however it
    // sits); P(miss) = (1 - p1) + p1 (1 - p2)
    let z1 = (r + p.app(w) - bp.abs()) * (1.0 / (EDGE * w));
    let z2 = (r * (2.0 / w) + (-TAU)) * (1.0 / CAPACITY_SD);
    let (lp1, lq1, lp2, lq2) = (z1.log_ndtr(), (-z1).log_ndtr(), z2.log_ndtr(), (-z2).log_ndtr());
    let (lp, lq) = (lp1 + lp2, logaddexp(lq1, lp1 + lq2));
    let (c0, c1) = (N::c(pu.c0), N::c(pu.c1));
    let xw = (tp - hbar * (1.0 / pu.sin)).clamp(c0, c1);
    let zw = xw - c0;
    let eps = (-(f.clamp(N::c(0.0), N::c(1.0))) + 1.0) * EPS + EPS_MIN;
    let ln = pu.c1 - pu.c0;
    let over = ((xw + MP_LEN - c1) * (1.0 / MP_LEN)).clamp(N::c(1e-12), N::c(1.0));
    let l_pass = logaddexp(lq - lam * ln, lp - lam * zw + (eps + 1e-300).ln() + over.ln());
    let q = pu.q;
    let p_pass = l_pass.min(N::c(0.0)).exp();
    let sel = if pu.early { N::c(0.0) } else { (p_pass * (-(1.0 - q))).log1p() };
    let mut terms = None;
    let l = if pu.stop {
        // a stop: the wood's echo or foliage's (only a stop needs these terms)
        let sig = (r.sq() * (ROUGH * ROUGH) + SIGMA_R * SIGMA_R).sqrt();
        let x = N::c(pu.x);
        let z = (x - c0).max(N::c(0.0));
        let u = (x - xw) / sig;
        let log_sqrt2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let l_gauss = (-eps).log1p() - u.sq() * 0.5 - sig.ln() + (-log_sqrt2pi);
        let l_tail = (eps + 1e-300).ln() + (-MP_LEN.ln()) + u.log_ndtr() + ((xw + MP_LEN - x) / sig).log_ndtr();
        let l_wood = lp - lam * zw + logaddexp(l_gauss, l_tail);
        let l_leaf = ell - lam * z + logaddexp(lq, lp + (-u).log_ndtr());
        terms = Some((l_wood, l_leaf));
        logaddexp(l_wood, l_leaf)
    } else {
        l_pass + q.ln()
    };
    (l - sel, terms)
}

/// Where each pair's state comes from: its row's `[delta1, delta2, eta,
/// ell]` and, unless `nu` is `u32::MAX`, its scan's nuisance offset added to
/// the delta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Owner {
    pub row: u32,
    pub nu: u32,
}

fn state(o: &Owner, theta: &[[f64; 4]], nu: &[[f64; 2]]) -> [f64; 4] {
    let mut s = theta[o.row as usize];
    if o.nu != u32::MAX {
        let v = nu[o.nu as usize];
        s[0] += v[0];
        s[1] += v[1];
    }
    s
}

fn check(pulses: &[Pulse], owners: &[Owner], theta: &[[f64; 4]], nu: &[[f64; 2]]) -> Result<()> {
    if owners.len() != pulses.len() {
        return Err(Error::invalid("joint: one owner per pair"));
    }
    if owners.iter().any(|o| o.row as usize >= theta.len() || (o.nu != u32::MAX && o.nu as usize >= nu.len())) {
        return Err(Error::invalid("joint: an owner's row or nu is out of range"));
    }
    Ok(())
}

/// Each pair's log likelihood, at row states `theta` and nuisances `nu`.
pub fn joint_loglik(pulses: &[Pulse], owners: &[Owner], theta: &[[f64; 4]], nu: &[[f64; 2]], p: &JointParams, threads: usize) -> Result<Vec<f64>> {
    check(pulses, owners, theta, nu)?;
    pool(threads)?.install(|| Ok(pulses.par_iter().zip(owners).map(|(pu, o)| pair(pu, state(o, theta, nu), p)).collect()))
}

/// Each pair's chance that its stop is the wood's echo rather than foliage's, at row states `theta` and nuisances
/// `nu`: the softmax of the stop's wood and leaf terms (3dtrees research 103 §18: the classification on the exact
/// likelihood's path); NaN for a pass.
pub fn joint_wood_share(pulses: &[Pulse], owners: &[Owner], theta: &[[f64; 4]], nu: &[[f64; 2]], p: &JointParams, threads: usize) -> Result<Vec<f64>> {
    check(pulses, owners, theta, nu)?;
    pool(threads)?.install(|| Ok(pulses.par_iter().zip(owners).map(|(pu, o)| match pair_terms(pu, state(o, theta, nu), p).1 {
        Some((w, l)) => 1.0 / (1.0 + (l - w).exp()),
        None => f64::NAN,
    }).collect()))
}

/// Sums by group of the pairs' log likelihood, gradient and Hessian in their
/// own four state variables: `(value (n,), gradient (n, 4), Hessian (n, 4,
/// 4))`, group `group[i]` of `n_groups`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn joint_blocks(pulses: &[Pulse], owners: &[Owner], group: &[u32], n_groups: usize, theta: &[[f64; 4]], nu: &[[f64; 2]], p: &JointParams, threads: usize) -> Result<(Vec<f64>, Vec<[f64; 4]>, Vec<[[f64; 4]; 4]>)> {
    check(pulses, owners, theta, nu)?;
    let n = pulses.len();
    if group.len() != n || group.iter().any(|&g| g as usize >= n_groups) {
        return Err(Error::invalid("joint: one group (below n_groups) per pair"));
    }
    // the same sums on any number of threads (a fold over pairs adds in a different order each run, and a sharp
    // objective can follow that to another optimum): pairs sorted by group, cut into fixed chunks summed in order
    // in parallel, the chunks' sums merged in order. Fixed chunks keep a trunk's many pairs from serialising.
    const CHUNK: usize = 4096;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| group[i]);
    type Acc = (f64, [f64; 4], [[f64; 4]; 4]);
    pool(threads)?.install(|| {
        let parts: Vec<Vec<(u32, Acc)>> = order.par_chunks(CHUNK).map(|chunk| {
            let mut out: Vec<(u32, Acc)> = Vec::new();
            for &i in chunk {
                let s = state(&owners[i], theta, nu);
                let d = pair(&pulses[i], [D::var(s[0], 0), D::var(s[1], 1), D::var(s[2], 2), D::var(s[3], 3)], p);
                if out.last().is_none_or(|(g, _)| *g != group[i]) {
                    out.push((group[i], (0.0, [0.0; 4], [[0.0; 4]; 4])));
                }
                let acc = &mut out.last_mut().unwrap().1;
                acc.0 += d.v;
                for a in 0..4 {
                    acc.1[a] += d.g[a];
                    for b in 0..4 {
                        acc.2[a][b] += d.h[a][b];
                    }
                }
            }
            out
        }).collect();
        let (mut f, mut g, mut h) = (vec![0.0; n_groups], vec![[0.0; 4]; n_groups], vec![[[0.0; 4]; 4]; n_groups]);
        for (k, acc) in parts.into_iter().flatten() {
            let k = k as usize;
            f[k] += acc.0;
            for a in 0..4 {
                g[k][a] += acc.1[a];
                for b in 0..4 {
                    h[k][a][b] += acc.2[a][b];
                }
            }
        }
        Ok((f, g, h))
    })
}

/// A tree's pulse-cylinder pairs as the likelihood reads them, with where each
/// came from: its row (cylinder), scan and pulse, how many echoes the
/// pulse recorded (0: a miss), and a stop's place among them (`u8::MAX`: a pass).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointPairs {
    pub pulses: Vec<Pulse>,
    pub row: Vec<u32>,
    pub scan: Vec<u32>,
    pub pulse: Vec<u32>,
    pub n_echo: Vec<u32>,
    pub echo: Vec<u8>,
}

/// A tree's echoes classified (3dtrees research 103 §18), one record per echo, keyed as the tile sidecars are
/// (research 104, win 1): the scan (an index into the fit's scans), the pulse's row in its Shots file and the echo's
/// place in its pulse. Each with the chance it is wood's echo, the node it belongs to and how far it lies from that
/// node's fitted surface (m), sorted by key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Echoes {
    pub scan: Vec<u32>,
    pub pulse: Vec<u32>,
    pub echo: Vec<u8>,
    pub node: Vec<u32>,
    pub p_wood: Vec<f64>,
    pub dist: Vec<f64>,
}

impl Echoes {
    pub(crate) fn push(&mut self, scan: u32, pulse: u32, echo: u8, node: u32, p_wood: f64, dist: f64) {
        self.scan.push(scan);
        self.pulse.push(pulse);
        self.echo.push(echo);
        self.node.push(node);
        self.p_wood.push(p_wood);
        self.dist.push(dist);
    }

    /// Appends `other`'s records, each node `n` renamed `node(n)`.
    pub(crate) fn append(&mut self, other: Echoes, node: impl Fn(u32) -> u32) {
        self.scan.extend(other.scan);
        self.pulse.extend(other.pulse);
        self.echo.extend(other.echo);
        self.node.extend(other.node.into_iter().map(node));
        self.p_wood.extend(other.p_wood);
        self.dist.extend(other.dist);
    }

    pub fn len(&self) -> usize {
        self.scan.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scan.is_empty()
    }

    /// One record per echo: of an echo several nodes saw (their tubes overlap), the one whose surface it lies
    /// nearest; sorted by key.
    pub(crate) fn dedup(self) -> Echoes {
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_unstable_by(|&i, &j| (self.scan[i], self.pulse[i], self.echo[i]).cmp(&(self.scan[j], self.pulse[j], self.echo[j]))
            .then(self.dist[i].total_cmp(&self.dist[j])).then(self.node[i].cmp(&self.node[j])));
        let mut out = Echoes::default();
        let mut last = None;
        for i in order {
            let key = (self.scan[i], self.pulse[i], self.echo[i]);
            if last == Some(key) {
                continue;
            }
            last = Some(key);
            out.push(key.0, key.1, key.2, self.node[i], self.p_wood[i], self.dist[i]);
        }
        out
    }
}

/// The radius of a row's tube, from its prior radius: wide enough for the
/// claim and the axis to move, capped above a trunk's.
pub fn window(r_prior: f64) -> f64 {
    (2.0 * r_prior + 0.03).clamp(0.04, (1.3 * r_prior + 0.05).max(0.25))
}

/// Unit axis and two directions across it, per cylinder: the frame the axis
/// offsets are measured in (`e1 = u x z`, or `u x x` for a near-vertical
/// axis; `e2 = u x e1`).
pub fn frames(a: &[Point], b: &[Point]) -> Vec<(Point, Point, Point, f64)> {
    let unit = |v: Point| {
        let l = crate::transform::norm(&v);
        (v.map(|x| x / l), l)
    };
    a.iter().zip(b).map(|(a, b)| {
        let (u, len) = unit(crate::transform::sub(b, a));
        let reference = if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let (e1, _) = unit(cross(&u, &reference));
        (u, e1, cross(&u, &e1), len)
    }).collect()
}

/// The [`JointPairs`] of cylinders `a[i] -> b[i]`, each with its tube of
/// radius `window[i]` (see [`crate::qsm::scan_index::pairs`]), measured from
/// those axes: each pulse's signed impact and range of closest approach, how
/// both move with an offset of the axis, its chord through the tube, and its
/// fate there (the first echo inside the chord is a stop; otherwise the pulse
/// went on, a pass). `q` is 1: what is behind recorded every pass. A pulse
/// along the axis, which has no impact, is left out.
pub fn joint_table(scans: &[&crate::qsm::scan_index::ScanIndex], a: &[Point], b: &[Point], window: &[f64], threads: usize) -> Result<JointPairs> {
    use crate::transform::dot;
    let pp = crate::qsm::scan_index::pairs(scans, a, b, window, threads)?;
    let fr = frames(a, b);
    let mut out = JointPairs::default();
    for c in 0..a.len() {
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
            let bimp = dot(&crate::transform::sub(&o, &a[c]), &nrm);
            let du = dot(&d, &u);
            let p_hat = [0, 1, 2].map(|k| (d[k] - du * u[k]) / sin);
            let t = pp.t[i] as f64;
            let half = (window[c] * window[c] - bimp * bimp).max(0.0).sqrt() / sin;
            let (c0, c1) = (t - half, t + half);
            let echoes = &pp.range[pp.echo_start[i] as usize..pp.echo_start[i + 1] as usize];
            let first_at = echoes.iter().position(|&e| (e as f64) >= c0 && (e as f64) <= c1);
            let first_inside = first_at.map(|i| echoes[i] as f64);
            out.pulses.push(Pulse {
                b: bimp,
                t,
                sin,
                jb: [dot(&nrm, &e1), dot(&nrm, &e2)],
                jt: [dot(&p_hat, &e1) / sin, dot(&p_hat, &e2) / sin],
                c0,
                c1,
                x: first_inside.unwrap_or(0.0),
                stop: first_inside.is_some(),
                early: echoes.first().is_some_and(|&e| (e as f64) < c0),
                q: 1.0,
            });
            out.row.push(c as u32);
            out.scan.push(pp.scan[i]);
            out.pulse.push(pp.pulse[i]);
            out.n_echo.push(echoes.len() as u32);
            out.echo.push(first_at.map_or(u8::MAX, |i| i.min(254) as u8));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> JointParams {
        JointParams::default()
    }

    /// A few pulses past a 5 cm stem at 8 m: on the surface, inside it, passing, and a leaf in front.
    fn table() -> Vec<Pulse> {
        [(0.0, true, 7.95, false, 1.0), (0.04, true, 7.97, false, 0.6), (0.07, false, 0.0, false, 0.6), (0.02, true, 7.88, true, 1.0), (0.03, false, 0.0, false, 1.0)]
            .map(|(b, stop, x, early, q)| Pulse { b, t: 8.0, sin: 0.95, jb: [0.9, 0.1], jt: [0.05, 0.9], c0: 7.85, c1: 8.15, x, stop, early, q })
            .to_vec()
    }

    #[test]
    fn log_ndtr_skips_erfc_only_where_it_is_zero() {
        for i in 0..200_000 {
            let z = 30.0 + i as f64 * 1e-4;
            let want = (-0.5 * libm::erfc(z / std::f64::consts::SQRT_2)).ln_1p();
            assert_eq!(log_ndtr_value(z).to_bits(), want.to_bits(), "z {z}");
        }
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
        let dual = |s: [f64; 4]| [D::var(s[0], 0), D::var(s[1], 1), D::var(s[2], 2), D::var(s[3], 3)];
        for (i, pu) in tab.iter().enumerate() {
            let d = pair(pu, dual(s0), &p);
            assert!((d.v - pair(pu, s0, &p)).abs() < 1e-12 * d.v.abs().max(1.0), "pair {i}: f64 and dual values differ");
            let h = [1e-5, 1e-5, 1e-4, 1e-4];
            for a in 0..4 {
                let (mut sp, mut sm) = (s0, s0);
                sp[a] += h[a];
                sm[a] -= h[a];
                let (fp, fm) = (pair(pu, dual(sp), &p), pair(pu, dual(sm), &p));
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
    fn the_table_measures_each_pulse_from_the_axis() {
        // the evidence tests' fan: a 5 cm stem at 5 m with a wall at 8 m behind it
        let (scan, a, b) = crate::qsm::scan_index::tests::scene(0.05, true);
        let t = joint_table(&[&scan], &[a], &[b], &[0.15], 2).unwrap();
        let m = t.pulses.len();
        assert!(m > 200 && t.row.iter().all(|&r| r == 0) && t.scan.len() == m && t.n_echo.len() == m);
        for pu in &t.pulses {
            assert!(pu.b.abs() < 0.15 && pu.c0 < pu.t && pu.t < pu.c1 && pu.sin > 0.99);
            // a stop on the stem's near face, at the depth of the circle; passes went on to the wall
            if pu.stop {
                let face = pu.t - (0.05f64 * 0.05 - pu.b * pu.b).max(0.0).sqrt() / pu.sin;
                assert!(pu.b.abs() < 0.0505 && (pu.x - face).abs() < 2e-3, "stop at {} (b {}) against the face at {face}", pu.x, pu.b);
            } else {
                assert!(pu.b.abs() > 0.0495, "a pass through the stem at {}", pu.b);
            }
            assert!(!pu.early && pu.q == 1.0);
        }
        let stops = t.pulses.iter().filter(|p| p.stop).count();
        assert!(stops > 50 && m - stops > 50);
        // an axis moved across itself moves each pulse's impact by jb and its range by jt
        let (_, e1, e2, _) = frames(&[a], &[b])[0];
        let off = [0.004, -0.003];
        let shift = [0, 1, 2].map(|k| off[0] * e1[k] + off[1] * e2[k]);
        let (a2, b2) = ([0, 1, 2].map(|k| a[k] + shift[k]), [0, 1, 2].map(|k| b[k] + shift[k]));
        let moved = joint_table(&[&scan], &[a2], &[b2], &[0.15], 2).unwrap();
        let before: std::collections::HashMap<u32, &Pulse> = t.pulse.iter().copied().zip(&t.pulses).collect();
        let mut checked = 0;
        for (k, pu) in moved.pulse.iter().zip(&moved.pulses) {
            if let Some(p0) = before.get(k) {
                let b = p0.b - (p0.jb[0] * off[0] + p0.jb[1] * off[1]);
                let tt = p0.t + (p0.jt[0] * off[0] + p0.jt[1] * off[1]);
                assert!((pu.b - b).abs() < 1e-6 && (pu.t - tt).abs() < 1e-4, "b {} vs {b}, t {} vs {tt}", pu.b, pu.t);
                checked += 1;
            }
        }
        assert!(checked > 200);
    }

    #[test]
    fn blocks_sum_their_pairs_and_inputs_are_checked() {
        let (tab, p) = (table(), params());
        // two rows; pairs 2 and 4 see row 1 through a nuisance offset
        let theta = [[0.0, 0.0, (0.05f64).ln(), 0.0], [0.001, 0.0, (0.04f64).ln(), 0.5]];
        let nu = [[0.002, -0.001]];
        let owners: Vec<Owner> = [(0, u32::MAX), (1, u32::MAX), (1, 0), (0, u32::MAX), (1, 0)].map(|(row, nu)| Owner { row, nu }).to_vec();
        let (f, g, _) = joint_blocks(&tab, &owners, &[0, 1, 1, 0, 1], 2, &theta, &nu, &p, 2).unwrap();
        let ll = joint_loglik(&tab, &owners, &theta, &nu, &p, 2).unwrap();
        assert!((f[0] - (ll[0] + ll[3])).abs() < 1e-9 && (f[1] - (ll[1] + ll[2] + ll[4])).abs() < 1e-9);
        assert!((ll[2] - pair(&tab[2], [0.003, -0.001, (0.04f64).ln(), 0.5], &p)).abs() < 1e-12, "nu is added to delta");
        assert!(g.iter().all(|x| x.iter().all(|v| v.is_finite())));
        let one = joint_blocks(&tab, &owners, &[0, 1, 1, 0, 1], 2, &theta, &nu, &p, 1).unwrap();
        assert_eq!(one, joint_blocks(&tab, &owners, &[0, 1, 1, 0, 1], 2, &theta, &nu, &p, 4).unwrap(), "the same sums on any number of threads");
        assert!(joint_blocks(&tab, &owners, &[0, 1, 2, 0, 1], 2, &theta, &nu, &p, 1).is_err());
        assert!(joint_loglik(&tab, &owners[..2], &theta, &nu, &p, 1).is_err());
        let bad: Vec<Owner> = owners.iter().map(|o| Owner { row: 5, ..*o }).collect();
        assert!(joint_loglik(&tab, &bad, &theta, &nu, &p, 1).is_err());
    }
}
