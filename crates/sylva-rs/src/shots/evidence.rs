// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! One evidence term per (hypothesis, scan): an overdispersed count model for how many of a scan's informative cells
//! over a hypothesis refute it.
//!
//! A scan's cells over one hypothesis (a voxel, a slice, a gap cylinder) are not independent votes: neighbouring
//! footprints overlap and share one line of sight through the same gaps. So `k` refuting cells out of `n` informative
//! ones follow a beta-binomial, mean refute probability `mu` and intra-scan correlation `rho`
//! (`alpha = mu (1 − rho) / rho`, `beta = (1 − mu)(1 − rho) / rho`). `rho = 0` is the binomial (per-cell counting).
//!
//! Fitted once under "present" and once under "absent", a scan contributes the log-likelihood ratio of its `(k, n)`.
//! For `rho > 0` the ratio tends to the log ratio of the two beta densities at `k / n` as `n` grows: bounded, so a
//! near scan's many cells through one gap count as one look. The binomial's grows linearly in `n`.
//!
//! The binomial coefficient is left out of [`ln_pmf`]: it cancels in every ratio and doesn't move a fit.

use rayon::prelude::*;
use std::collections::HashMap;

/// `ln P(k | n)` up to the binomial coefficient, for mean refute probability `mu` and intra-scan correlation `rho`.
pub fn ln_pmf(k: u32, n: u32, mu: f64, rho: f64) -> f64 {
    let (k, m) = (k as f64, (n - k.min(n)) as f64);
    let mu = mu.clamp(1e-9, 1.0 - 1e-9);
    if rho <= 0.0 {
        return k * mu.ln() + m * (1.0 - mu).ln();
    }
    let s = (1.0 - rho) / rho;
    let (a, b) = (mu * s, (1.0 - mu) * s);
    let lg = libm::lgamma;
    lg(k + a) - lg(a) + lg(m + b) - lg(b) - (lg(k + m + s) - lg(s))
}

/// A fitted count model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    pub mu: f64,
    pub rho: f64,
    /// Weighted log-likelihood (without binomial coefficients).
    pub loglik: f64,
}

/// Weighted maximum likelihood of `(mu, rho)` over rows `(k, n, w)`. With `binomial`, `rho` is held at 0.
/// Coordinate golden-section search on the logits, over the distinct `(k, n)` rows.
pub fn fit(k: &[u32], n: &[u32], w: &[f64], binomial: bool) -> Fit {
    let mut h: HashMap<(u32, u32), f64> = HashMap::new();
    for i in 0..k.len() {
        if n[i] > 0 && w[i] > 0.0 {
            *h.entry((k[i].min(n[i]), n[i])).or_default() += w[i];
        }
    }
    let rows: Vec<((u32, u32), f64)> = h.into_iter().collect();
    let ll = |mu: f64, rho: f64| rows.par_iter().map(|&((k, n), w)| w * ln_pmf(k, n, mu, rho)).sum::<f64>();
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (tw, tk, tn) = rows.iter().fold((0.0, 0.0, 0.0), |a, &((k, n), w)| (a.0 + w, a.1 + w * k as f64, a.2 + w * n as f64));
    if tw == 0.0 {
        return Fit { mu: f64::NAN, rho: f64::NAN, loglik: 0.0 };
    }
    let mu0 = ((tk + 0.5) / (tn + 1.0)).clamp(1e-6, 1.0 - 1e-6);
    let (mut x, mut y) = ((mu0 / (1.0 - mu0)).ln(), if binomial { f64::NEG_INFINITY } else { -2.0 });
    let golden = |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64| {
        let g = 0.5 * (5f64.sqrt() - 1.0);
        let (mut c, mut d) = (hi - g * (hi - lo), lo + g * (hi - lo));
        let (mut fc, mut fd) = (f(c), f(d));
        for _ in 0..60 {
            if fc > fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - g * (hi - lo);
                fc = f(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + g * (hi - lo);
                fd = f(d);
            }
        }
        0.5 * (lo + hi)
    };
    let rho_of = |y: f64| if y == f64::NEG_INFINITY { 0.0 } else { sig(y) };
    for _ in 0..if binomial { 1 } else { 12 } {
        x = golden(&|x| ll(sig(x), rho_of(y)), x - 6.0, x + 6.0);
        if !binomial {
            y = golden(&|y| ll(sig(x), sig(y)), -14.0, 7.0);
        }
    }
    Fit { mu: sig(x), rho: rho_of(y), loglik: ll(sig(x), rho_of(y)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_zero_is_the_binomial_and_small_rho_approaches_it() {
        let b = ln_pmf(3, 20, 0.2, 0.0);
        assert!((b - (3.0 * 0.2f64.ln() + 17.0 * 0.8f64.ln())).abs() < 1e-12);
        assert!((ln_pmf(3, 20, 0.2, 1e-7) - b).abs() < 1e-4);
    }

    #[test]
    fn the_ratio_is_bounded_in_n_and_the_binomial_is_not() {
        let llr = |k, n, rho| ln_pmf(k, n, 0.8, rho) - ln_pmf(k, n, 0.1, rho);
        let (a, b) = (llr(80, 100, 0.3), llr(8000, 10000, 0.3));
        assert!(a > 0.0 && (b - a).abs() < 0.1 * a.abs() + 0.5, "{a} {b}");
        let (c, d) = (llr(80, 100, 0.0), llr(8000, 10000, 0.0));
        assert!((d / c - 100.0).abs() < 1e-6);
    }

    #[test]
    fn the_fit_recovers_parameters_from_their_own_pmf() {
        // weights are the exact beta-binomial pmf for n = 25 (mixed with n = 8): the ML is the generating pair
        let (mu, rho) = (0.3, 0.25);
        let (mut k, mut n, mut w) = (vec![], vec![], vec![]);
        for &nn in &[8u32, 25] {
            let lc = |kk: u32| libm::lgamma(nn as f64 + 1.0) - libm::lgamma(kk as f64 + 1.0) - libm::lgamma((nn - kk) as f64 + 1.0);
            for kk in 0..=nn {
                k.push(kk);
                n.push(nn);
                w.push((lc(kk) + ln_pmf(kk, nn, mu, rho)).exp());
            }
        }
        let f = fit(&k, &n, &w, false);
        assert!((f.mu - mu).abs() < 1e-3 && (f.rho - rho).abs() < 1e-3, "{f:?}");
        let fb = fit(&k, &n, &w, true);
        assert!(fb.rho == 0.0 && (fb.mu - mu).abs() < 1e-3 && fb.loglik < f.loglik);
    }
}
