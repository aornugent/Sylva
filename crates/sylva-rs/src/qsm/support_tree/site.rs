// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A site's priors learned from its own measured wood, by empirical Bayes
//! (3dtrees research 93 §19; the ecology of §11.7, §11.8 and §14.1): the
//! tree solve's fork and taper terms, located and scaled by the plot rather
//! than by RCT's taper.
//!
//! - **Forks.** A fork's child's step in log R from its parent is
//!   `(mu_fork + log share) / 2`, the share by subtree wood length, under a
//!   Student-t of [`super::FORK_DF`] degrees of freedom and scale `sig_fork`
//!   (as the tree solve's term). Over every fork whose parent and child rows
//!   are both measured, the residuals' location and scale come by the
//!   Student-t's expectation-maximisation; the scale is net of the two
//!   radii's own noise.
//! - **Taper,** between forks. A step's mean in log R is `taper` times its step
//!   in the log of the reach (the path from the axis to the furthest tip
//!   beyond it, at least [`REACH_MIN`]), and its variance `sig_eta^2` per m of
//!   path. Each measured row is paired with its nearest measured ancestor
//!   with no fork between (within [`PAIR_MAX`] of path): the rate by weighted
//!   least squares, and the variance as `2 noise^2 + sig_eta^2 L` against the
//!   path `L` between them, which separates the radii's own noise (the
//!   intercept) from the taper's walk (the slope), reweighted once.
//!
//! The measured radii are the caller's: the refit's circles, or the support
//! path's own measured pieces. Hold out any tree that will be scored under the
//! priors learned.

use crate::error::{Error, Result};
use crate::qsm::joint_fit::{ForkShare, Rows};

use super::FORK_DF;

/// The shortest reach (m): a tip's last centimetres say nothing more of its taper.
pub const REACH_MIN: f64 = 0.1;
/// How far apart along the path a measured row and its measured ancestor may be and still pair (m).
const PAIR_MAX: f64 = 2.0;

/// A site's priors (see the module documentation): the [`crate::qsm::joint_fit::PriorParams`] values `mu_fork`,
/// `sig_fork`, `taper`, `sig_eta`, the measured radii's own noise (sd of log R) and how many forks and steps
/// informed them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SitePrior {
    pub mu_fork: f64,
    pub sig_fork: f64,
    pub taper: f64,
    pub sig_eta: f64,
    pub noise: f64,
    pub n_forks: usize,
    pub n_steps: usize,
}

/// Each row's reach from its end: the longest path from there to a tip (m), rows in parents-first order.
pub(super) fn reach_ends(parent: &[i64], len: &[f64]) -> Vec<f64> {
    let mut end = vec![0.0f64; len.len()];
    for k in (0..len.len()).rev() {
        if parent[k] >= 0 {
            let q = parent[k] as usize;
            end[q] = end[q].max(end[k] + len[k]);
        }
    }
    end
}

/// A site's priors from its trees' rows and each row's measured radius (NaN where none; see the module
/// documentation). An error where fewer than 10 forks or 10 steps are measured.
pub fn site_prior(trees: &[(Rows, &[f64])]) -> Result<SitePrior> {
    // (residual of a fork's child), (step in log R, step in log reach, path)
    let (mut forks, mut steps): (Vec<f64>, Vec<(f64, f64, f64)>) = (Vec::new(), Vec::new());
    for (rows, r) in trees {
        let fr = rows.check("site prior")?;
        if r.len() != fr.len() {
            return Err(Error::invalid("site prior: one measured radius per row"));
        }
        let len: Vec<f64> = fr.iter().map(|f| f.3).collect();
        let share = rows.fork_share(&len, ForkShare::WoodLength);
        let end = reach_ends(rows.parent, &len);
        let reach: Vec<f64> = (0..len.len()).map(|k| (end[k] + 0.5 * len[k]).max(REACH_MIN)).collect();
        let meas = |k: usize| r[k].is_finite() && r[k] > 0.0;
        for k in (0..len.len()).filter(|&k| meas(k)) {
            let Ok(q) = usize::try_from(rows.parent[k]) else { continue };
            if share[k].is_finite() {
                if meas(q) {
                    forks.push(r[k].ln() - r[q].ln() - 0.5 * share[k].ln());
                }
                continue;
            }
            // up the branch to the nearest measured row, no fork between
            let (mut j, mut q, mut path) = (k, q, 0.5 * len[k]);
            loop {
                if share[j].is_finite() || path >= PAIR_MAX {
                    break;
                }
                if meas(q) {
                    steps.push((r[k].ln() - r[q].ln(), reach[k].ln() - reach[q].ln(), path + 0.5 * len[q]));
                    break;
                }
                path += len[q];
                let Ok(next) = usize::try_from(rows.parent[q]) else { break };
                (j, q) = (q, next);
            }
        }
    }
    if forks.len() < 10 || steps.len() < 10 {
        return Err(Error::invalid(format!("site prior: {} measured forks and {} steps; at least 10 of each needed", forks.len(), steps.len())));
    }
    // the taper: its rate by weighted least squares, its variance against the path, reweighted once
    let (mut c0, mut c1) = (0.0, 1.0);
    let mut taper = 0.0;
    for _ in 0..2 {
        let w = |l: f64| 1.0 / (c0 + c1 * l).max(1e-9);
        let (sxy, sxx) = steps.iter().fold((0.0, 0.0), |(a, b), &(d, x, l)| (a + w(l) * d * x, b + w(l) * x * x));
        taper = sxy / sxx.max(1e-300);
        // resid^2 = c0 + c1 L, by least squares
        let n = steps.len() as f64;
        let (sl, sr, sll, slr) = steps.iter().fold((0.0, 0.0, 0.0, 0.0), |(a, b, c, e), &(d, x, l)| {
            let r2 = (d - taper * x).powi(2);
            (a + l, b + r2, c + l * l, e + l * r2)
        });
        let det = n * sll - sl * sl;
        (c0, c1) = if det > 0.0 { ((sll * sr - sl * slr) / det, (n * slr - sl * sr) / det) } else { (sr / n, 0.0) };
        if c1 <= 0.0 || c0 < 0.0 {
            // no growth with the path (or none at zero): all of it the walk's, or all the noise's
            (c0, c1) = if c1 <= 0.0 { (sr / n, 1e-6) } else { (0.0, sr / sl) };
        }
    }
    let noise = (0.5 * c0).max(0.0).sqrt();
    // the forks: the Student-t's location and scale by expectation-maximisation, the scale net of both radii's noise
    let (mut m, mut s2) = {
        let mut v = forks.clone();
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        (med, (1.4826 * dev[dev.len() / 2]).powi(2).max(1e-6))
    };
    for _ in 0..200 {
        let w: Vec<f64> = forks.iter().map(|e| (FORK_DF + 1.0) / (FORK_DF + (e - m).powi(2) / s2)).collect();
        let sw: f64 = w.iter().sum();
        let m1 = forks.iter().zip(&w).map(|(e, w)| w * e).sum::<f64>() / sw;
        let s1 = forks.iter().zip(&w).map(|(e, w)| w * (e - m1).powi(2)).sum::<f64>() / forks.len() as f64;
        let done = (m1 - m).abs() < 1e-9 && (s1 - s2).abs() < 1e-9 * s2;
        (m, s2) = (m1, s1);
        if done {
            break;
        }
    }
    let sig_fork = (s2 - 2.0 * noise * noise).max(0.25 * s2).sqrt();
    Ok(SitePrior { mu_fork: 2.0 * m, sig_fork, taper, sig_eta: c1.sqrt(), noise, n_forks: forks.len(), n_steps: steps.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plot of `n` trees, each a stem of `m` measured rows of 10 cm forking into two children of `m` rows, the
    /// radii drawn from a taper of `taper` per log reach and a fork step `mu_fork`, with the walk `sig_eta` per
    /// sqrt(m), the fork's `sig_fork` and each radius's `noise` (log R; a deterministic stream).
    #[allow(clippy::type_complexity)]
    fn plot(n: usize, m: usize, truth: SitePrior) -> Vec<(Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>, Vec<i64>, Vec<f64>)> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut gauss = move || {
            let mut u = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let (a, b) = (u().max(1e-12), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        (0..n).map(|t| {
            let (mut a, mut b, mut parent, mut eta) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let x0 = 10.0 * t as f64;
            // the stem, then two children of equal wood length (share 1/2) leaning apart
            for j in 0..m {
                a.push([x0, 0.0, 0.1 * j as f64]);
                b.push([x0, 0.0, 0.1 * (j + 1) as f64]);
                parent.push(j as i64 - 1);
            }
            for side in [-1.0, 1.0] {
                for j in 0..m {
                    let z = 0.1 * (m + j) as f64;
                    a.push([x0 + side * 0.05 * j as f64, 0.0, z]);
                    b.push([x0 + side * 0.05 * (j + 1) as f64, 0.0, z + 0.1]);
                    parent.push(if j == 0 { m as i64 - 1 } else { a.len() as i64 - 2 });
                }
            }
            let rows_len: Vec<f64> = a.iter().zip(&b).map(|(p, q)| ((q[0] - p[0]).powi(2) + (q[2] - p[2]).powi(2)).sqrt()).collect();
            let end = reach_ends(&parent, &rows_len);
            let reach: Vec<f64> = (0..a.len()).map(|k| (end[k] + 0.5 * rows_len[k]).max(REACH_MIN)).collect();
            for k in 0..a.len() {
                let e = match parent[k] {
                    -1 => 0.08f64.ln(),
                    q if k == m || k == 2 * m => eta[q as usize] + 0.5 * (truth.mu_fork + 0.5f64.ln()) + truth.sig_fork * gauss(),
                    q => eta[q as usize] + truth.taper * (reach[k].ln() - reach[q as usize].ln()) + truth.sig_eta * rows_len[k].sqrt() * gauss(),
                };
                eta.push(e);
            }
            let r: Vec<f64> = eta.iter().map(|e| (e + truth.noise * gauss()).exp()).collect();
            let r_prior = vec![0.05; a.len()];
            (a, b, r_prior, parent, r)
        }).collect()
    }

    #[test]
    fn a_site_prior_recovers_the_plot_it_was_drawn_from() {
        let truth = SitePrior { mu_fork: 0.12, sig_fork: 0.15, taper: 0.3, sig_eta: 0.4, noise: 0.05, n_forks: 0, n_steps: 0 };
        let p = plot(400, 8, truth);
        let trees: Vec<(Rows, &[f64])> = p.iter().map(|(a, b, rp, par, r)| (Rows { a, b, r_prior: rp, r_max: &[], parent: par }, r.as_slice())).collect();
        let got = site_prior(&trees).unwrap();
        assert_eq!((got.n_forks, got.n_steps), (800, 400 * 7 * 3));
        assert!((got.mu_fork - truth.mu_fork).abs() < 0.04, "{got:?}");
        assert!((got.taper - truth.taper).abs() < 0.08, "{got:?}");
        assert!((got.sig_eta / truth.sig_eta - 1.0).abs() < 0.2, "{got:?}");
        assert!((got.noise - truth.noise).abs() < 0.02, "{got:?}");
        // a Student-t's scale on Gaussian residuals sits below their sd; within a third of it
        assert!((got.sig_fork / truth.sig_fork - 1.0).abs() < 0.35, "{got:?}");
        // too little measured: an error
        assert!(site_prior(&trees[..2]).is_err());
    }
}
