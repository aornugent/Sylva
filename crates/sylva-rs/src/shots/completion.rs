// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A stochastic completion field over fragment ends (3dtrees research 128, `completion.md`; a spike, not a fit).
//!
//! - **the walk:** a branch's tangent does a random walk in angle, diffusion `d` (rad² per metre, per transverse
//!   axis), optionally with kinks (rate `lambda` per metre, each kink's angle uniform within `kappa` per axis). Over
//!   arc length s, in the small-angle limit, the end angle and lateral offset per axis are jointly Gaussian
//!   (integrated Brownian motion: var θ = d s, var y = d s³/3, cov = d s²/2), widened by each end's direction and
//!   position noise.
//! - **the fit:** `d`, `lambda` and the per-slice direction noise by maximum likelihood on the angles between slices
//!   of one visible fragment (runs of slices sharing one direction taken as one piece), against their separation
//!   along it.
//! - **paths:** the cubic Hermite between two ends with their directions, cut into short cylinders for the shadow
//!   maps (`opacity::ShadowMap::test`); the killing (refuted cells) is summed by the caller.
//! - **the field:** each candidate link's log weight is the walk's log density (symmetrised over the two ends), plus
//!   an optional log-radius term (taper with unknown sign), less the killing. Each end normalises over its candidates
//!   and a null option ("ends here or leaves the field"); a link's probability is the product of its ends' shares.

use super::opacity::Cyl;
use super::surfels::{pt, v, V};
use crate::Point;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Angles between pieces of one fragment (`frag`), against their separation along the fragment's mean axis:
/// `(ds, theta)` for every pair with `min_sep <= ds <= max_sep`. A piece is the slices that share one direction (a
/// stripe's slices take their segment's axis, so runs of them are identical), at their mean position. Directions are
/// sign-aligned to the fragment's mean.
pub fn tangent_pairs(p: &[Point], u: &[Point], frag: &[i64], min_sep: f64, max_sep: f64) -> Vec<(f64, f64)> {
    let mut by: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &f) in frag.iter().enumerate() {
        by.entry(f).or_default().push(i);
    }
    let mut out = Vec::new();
    let mut keys: Vec<i64> = by.keys().copied().collect();
    keys.sort();
    for k in keys {
        let ix = &by[&k];
        let u0 = v(&u[ix[0]]);
        // pieces: (sum of positions, count, direction aligned to u0)
        let mut pc: Vec<(V, f64, V)> = Vec::new();
        for &i in ix {
            let ui = v(&u[i]);
            let ui = if ui.dot(&u0) < 0.0 { -ui } else { ui };
            match pc.iter_mut().find(|x| (x.2 - ui).norm() < 1e-9) {
                Some(x) => {
                    x.0 += v(&p[i]);
                    x.1 += 1.0;
                }
                None => pc.push((v(&p[i]), 1.0, ui)),
            }
        }
        if pc.len() < 2 {
            continue;
        }
        let ub = pc.iter().fold(V::zeros(), |acc, x| acc + x.2 * x.1).normalize();
        for a in 0..pc.len() {
            for b in a + 1..pc.len() {
                let ds = (pc[b].0 / pc[b].1 - pc[a].0 / pc[a].1).dot(&ub).abs();
                if ds < min_sep || ds > max_sep {
                    continue;
                }
                out.push((ds, pc[a].2.dot(&pc[b].2).clamp(-1.0, 1.0).acos()));
            }
        }
    }
    out
}

/// The walk's fit: diffusion `d` (rad²/m per axis), kink rate `lambda` (/m), per-slice direction noise `sigma`
/// (rad), and the mean log-likelihood per pair.
#[derive(Debug, Clone, Copy)]
pub struct WalkFit {
    pub d: f64,
    pub lambda: f64,
    pub sigma: f64,
    pub loglik: f64,
}

fn walk_loglik(pairs: &[(f64, f64)], d: f64, lambda: f64, sigma: f64, kappa: f64) -> f64 {
    // |Δθ| of a 2D isotropic Gaussian (per-axis variance 2σ² + d ds) is Rayleigh; a kink makes it uniform on the disc
    // of radius kappa (density 2θ/κ²).
    let s: f64 = pairs
        .iter()
        .map(|&(ds, th)| {
            let var = 2.0 * sigma * sigma + d * ds;
            let f0 = th / var * (-th * th / (2.0 * var)).exp();
            let q = 1.0 - (-lambda * ds).exp();
            let f1 = if th <= kappa { 2.0 * th / (kappa * kappa) } else { 0.0 };
            ((1.0 - q) * f0 + q * f1).max(1e-300).ln()
        })
        .sum();
    s / pairs.len().max(1) as f64
}

/// Maximum likelihood over (d, sigma) and, with `kinks`, lambda: a coarse log grid, then coordinate descent on 1D
/// log grids whose windows shrink each sweep (the (d, sigma) ridge needs many small steps).
pub fn fit_walk(pairs: &[(f64, f64)], kinks: bool, kappa: f64) -> WalkFit {
    let ll = |x: [f64; 3]| walk_loglik(pairs, x[0].exp(), if kinks { x[2].exp() } else { 0.0 }, x[1].exp(), kappa);
    let lo = [(1e-5f64).ln(), (0.002f64).ln(), (0.01f64).ln()];
    let hi = [(20.0f64).ln(), (0.6f64).ln(), (30.0f64).ln()];
    let n = 20;
    let at = |k: usize, i: usize| lo[k] + (hi[k] - lo[k]) * i as f64 / (n - 1) as f64;
    let grid: Vec<[f64; 3]> = (0..n)
        .flat_map(|a| (0..n).flat_map(move |b| (0..if kinks { n } else { 1 }).map(move |c| [a, b, c])))
        .map(|[a, b, c]| [at(0, a), at(1, b), at(2, c)])
        .collect();
    let (mut f, mut x) = grid
        .par_iter()
        .map(|&x| (ll(x), x))
        .reduce(|| (f64::NEG_INFINITY, [0.0; 3]), |p, q| if q.0 > p.0 { q } else { p });
    let mut w = 1.0;
    for _ in 0..40 {
        for k in 0..if kinks { 3 } else { 2 } {
            let (fk, xk) = (0..41)
                .into_par_iter()
                .map(|i| {
                    let mut y = x;
                    y[k] += w * (i as f64 / 20.0 - 1.0);
                    (ll(y), y)
                })
                .reduce(|| (f64::NEG_INFINITY, x), |p, q| if q.0 > p.0 { q } else { p });
            if fk > f {
                f = fk;
                x = xk;
            }
        }
        w *= 0.85;
    }
    WalkFit { d: x[0].exp(), sigma: x[1].exp(), lambda: if kinks { x[2].exp() } else { 0.0 }, loglik: f }
}

/// A fragment end: where it stops, its outward unit direction, its radius and that radius's log sd.
#[derive(Debug, Clone, Copy)]
pub struct End {
    pub p: Point,
    pub d: Point,
    pub r: f64,
    pub sig_r: f64,
}

/// The cubic Hermite from end a (leaving along `a.d`) to end b (arriving along `-b.d`), tangent scale the chord, cut
/// into `n` straight cylinders of the radius interpolated between the ends.
pub fn hermite_cyls(a: &End, b: &End, n: usize) -> Vec<Cyl> {
    let (xa, xb, da, db) = (v(&a.p), v(&b.p), v(&a.d), -v(&b.d));
    let m = (xb - xa).norm();
    let at = |t: f64| {
        let (t2, t3) = (t * t, t * t * t);
        xa * (2.0 * t3 - 3.0 * t2 + 1.0) + da * (m * (t3 - 2.0 * t2 + t)) + xb * (-2.0 * t3 + 3.0 * t2) + db * (m * (t3 - t2))
    };
    (0..n)
        .filter_map(|k| {
            let (t0, t1) = (k as f64 / n as f64, (k + 1) as f64 / n as f64);
            let (p0, p1) = (at(t0), at(t1));
            let g = p1 - p0;
            let l = g.norm();
            if l < 1e-6 {
                return None;
            }
            let tm = 0.5 * (t0 + t1);
            Some(Cyl { p: pt(&((p0 + p1) * 0.5)), u: pt(&(g / l)), half: 0.5 * l, r: a.r + (b.r - a.r) * tm })
        })
        .collect()
}

/// The walk and radius parameters of the field.
#[derive(Debug, Clone, Copy)]
pub struct Field {
    pub d: f64,
    pub lambda: f64,
    pub kappa: f64,
    /// each end's direction noise (rad) and position noise (m)
    pub sig_dir: f64,
    pub sig_pos: f64,
    /// radius: E[Δη] = ∓taper·s (sign unknown), var = q·s + σ_a² + σ_b²; off when `radius` is false
    pub taper: f64,
    pub q: f64,
    pub radius: bool,
}

fn n2_log(x: [f64; 2], c: [[f64; 2]; 2]) -> f64 {
    let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    let q = (c[1][1] * x[0] * x[0] - 2.0 * c[0][1] * x[0] * x[1] + c[0][0] * x[1] * x[1]) / det;
    -(2.0 * PI).ln() - 0.5 * det.ln() - 0.5 * q
}

/// log ∫ N2(x − φ w; C) dφ over φ in [−κ, κ] (flat), approximated by the untruncated integral when the best φ lies
/// inside, and by nothing otherwise.
fn kink_log(x: [f64; 2], w: [f64; 2], c: [[f64; 2]; 2], kappa: f64) -> f64 {
    let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    let ci = [[c[1][1] / det, -c[0][1] / det], [-c[1][0] / det, c[0][0] / det]];
    let mul = |a: [f64; 2], b: [f64; 2]| a[0] * (ci[0][0] * b[0] + ci[0][1] * b[1]) + a[1] * (ci[1][0] * b[0] + ci[1][1] * b[1]);
    let (ww, wx, xx) = (mul(w, w), mul(w, x), mul(x, x));
    if (wx / ww).abs() > kappa {
        return f64::NEG_INFINITY;
    }
    -0.5 * (2.0 * PI).ln() - 0.5 * ww.ln() - 0.5 * det.ln() - 0.5 * (xx - wx * wx / ww) - (2.0 * kappa).ln()
}

fn log_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// The walk's log density of reaching end b (position, arriving direction) from end a, in a's frame.
fn walk_log(a: &End, b: &End, f: &Field) -> f64 {
    let (xa, da) = (v(&a.p), v(&a.d));
    let g = v(&b.p) - xa;
    let s = g.dot(&da);
    if s < 1e-3 {
        return -1e3;
    }
    let refv = if da[2].abs() < 0.9 { V::new(0.0, 0.0, 1.0) } else { V::new(1.0, 0.0, 0.0) };
    let e1 = da.cross(&refv).normalize();
    let e2 = da.cross(&e1);
    let ib = -v(&b.d);
    let (sa, sp) = (f.sig_dir * f.sig_dir, f.sig_pos * f.sig_pos);
    // (θ, y) per axis: Brownian, a's direction error (θ and y move together), b's direction error, both positions
    let c = [
        [f.d * s + sa + sa, f.d * s * s / 2.0 + sa * s],
        [f.d * s * s / 2.0 + sa * s, f.d * s * s * s / 3.0 + sa * s * s + 2.0 * sp],
    ];
    let xs = [[ib.dot(&e1), g.dot(&e1)], [ib.dot(&e2), g.dot(&e2)]];
    let g0: f64 = xs.iter().map(|&x| n2_log(x, c)).sum();
    if f.lambda <= 0.0 {
        return g0;
    }
    // one kink at t uniform on [0, s]: (θ, y) shifted by φ (1, s − t) per axis
    let nt = 8;
    let mut g1 = f64::NEG_INFINITY;
    for k in 0..nt {
        let t = s * (k as f64 + 0.5) / nt as f64;
        let w = [1.0, s - t];
        let l: f64 = xs.iter().map(|&x| kink_log(x, w, c, f.kappa)).sum();
        g1 = log_add(g1, l - (nt as f64).ln());
    }
    let p0 = -f.lambda * s;
    let p1 = (1.0 - (-f.lambda * s).exp()).max(1e-300).ln();
    log_add(p0 + g0, p1 + g1)
}

/// A link's prior log weight: (direction, radius) terms. The direction term is the mean of the walk from a to b and
/// from b to a; the radius term is 0 when `f.radius` is false.
pub fn log_prior(a: &End, b: &End, f: &Field) -> (f64, f64) {
    let ld = 0.5 * (walk_log(a, b, f) + walk_log(b, a, f));
    if !f.radius {
        return (ld, 0.0);
    }
    let s = (v(&b.p) - v(&a.p)).norm();
    let dl = (b.r / a.r).ln();
    let var = f.q * s + a.sig_r * a.sig_r + b.sig_r * b.sig_r;
    let n1 = |x: f64| -0.5 * (2.0 * PI * var).ln() - 0.5 * x * x / var;
    (ld, log_add(n1(dl - f.taper * s), n1(dl + f.taper * s)) - 2f64.ln())
}

/// Per-end normalisation: `logw[k]` is pair k's log weight (prior less killing), `log_null` the null option's.
/// Returns per pair (share of a's mass on b, share of b's mass on a) and per end (null share, entropy in nats over
/// its candidates and the null).
pub fn normalise(n_ends: usize, a: &[u32], b: &[u32], logw: &[f64], log_null: f64) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut mx = vec![log_null; n_ends];
    for k in 0..a.len() {
        for e in [a[k] as usize, b[k] as usize] {
            mx[e] = mx[e].max(logw[k]);
        }
    }
    let mut z: Vec<f64> = (0..n_ends).map(|e| (log_null - mx[e]).exp()).collect();
    for k in 0..a.len() {
        for e in [a[k] as usize, b[k] as usize] {
            z[e] += (logw[k] - mx[e]).exp();
        }
    }
    let share = |k: usize, e: usize| (logw[k] - mx[e]).exp() / z[e];
    let pairs: Vec<(f64, f64)> = (0..a.len()).map(|k| (share(k, a[k] as usize), share(k, b[k] as usize))).collect();
    let mut ent = vec![0.0; n_ends];
    let null: Vec<f64> = (0..n_ends).map(|e| (log_null - mx[e]).exp() / z[e]).collect();
    for k in 0..a.len() {
        for (e, p) in [(a[k] as usize, pairs[k].0), (b[k] as usize, pairs[k].1)] {
            if p > 0.0 {
                ent[e] -= p * p.ln();
            }
        }
    }
    let ends = (0..n_ends).map(|e| (null[e], ent[e] - if null[e] > 0.0 { null[e] * null[e].ln() } else { 0.0 })).collect();
    (pairs, ends)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(p: [f64; 3], d: [f64; 3]) -> End {
        End { p, d, r: 0.02, sig_r: 0.3 }
    }

    #[test]
    fn straight_beats_bent_and_offset() {
        let f = Field { d: 0.05, lambda: 0.5, kappa: PI / 2.0, sig_dir: 0.05, sig_pos: 0.01, taper: 0.15, q: 0.04, radius: true };
        let a = end([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let b = end([0.0, 0.0, 0.3], [0.0, 0.0, -1.0]);
        let c = end([0.1, 0.0, 0.3], [0.0, 0.0, -1.0]);
        let d = end([0.0, 0.0, 0.3], [-0.5f64.sqrt(), 0.0, -0.5f64.sqrt()]);
        let s = log_prior(&a, &b, &f).0;
        assert!(s > log_prior(&a, &c, &f).0);
        assert!(s > log_prior(&a, &d, &f).0);
        // a kink keeps a sharp bend from vanishing
        let fb = Field { lambda: 0.0, ..f };
        assert!(log_prior(&a, &d, &f).0 > log_prior(&a, &d, &fb).0);
    }

    #[test]
    fn fit_recovers_diffusion() {
        // synthetic: per-axis var 2σ² + d ds, σ = 0.03, d = 0.1
        let mut pairs = Vec::new();
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        for i in 0..20000 {
            let ds = 0.02 + 0.48 * (i as f64 / 20000.0);
            let sd = (2.0 * 0.03f64 * 0.03 + 0.1 * ds).sqrt();
            let g = |u: f64, w: f64| (-2.0 * u.ln()).sqrt() * (2.0 * PI * w).cos();
            let (x, y) = (sd * g(rnd(), rnd()), sd * g(rnd(), rnd()));
            pairs.push((ds, (x * x + y * y).sqrt()));
        }
        let fit = fit_walk(&pairs, false, PI / 2.0);
        assert!((fit.d / 0.1 - 1.0).abs() < 0.15, "{fit:?}");
        assert!((fit.sigma / 0.03 - 1.0).abs() < 0.15, "{fit:?}");
    }

    #[test]
    fn hermite_ends_where_asked() {
        let a = end([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let b = end([0.2, 0.0, 0.5], [0.0, 0.0, -1.0]);
        let c = hermite_cyls(&a, &b, 10);
        let total: f64 = c.iter().map(|x| 2.0 * x.half).sum();
        assert!(total > 0.538 && total < 0.7);
        assert!((c[0].u[2] - 1.0).abs() < 0.05 && (c[9].u[2] - 1.0).abs() < 0.05);
    }
}
