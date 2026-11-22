// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! An echo's reading of its object's fill (3dtrees research 130 representation-dgp, design-dgp §1): the one beam.
//!
//! - **The object and its fill.** An echo's object fills f of the beam b it met; the beam left behind it is
//!   b (1 − f), geometric, the same whether its brightness shows it or not.
//! - **The reading** u is the echo's reflectance against a full hit in the beam it met, `x − full − 10 log10 b` dB
//!   ([`Settings::refl_full_hit`]). It is noisy evidence of the fill: `u = 10 log10 f + ℓ + n + d`, with ℓ the
//!   object's level (Gaussian) and n a full hit's own noise (a t; the two read together as a t of their quadrature
//!   scale), and d ≤ 0 a darkening of a partial fill (a half-t of scale `edge (1 − f)`): a partial fill can read
//!   only darker than its object at full fill, plus n. The darkening takes no beam.
//! - **The fill prior**: a share `full` of the objects fill the beam they meet; the rest have a fill Beta(α, β).
//! - **What it answers** for a reading: the chance the object stopped the beam, π(u) = P(f = 1 | u) ([`Reading::stop`]),
//!   and the beam an object that didn't stop it leaves, E[1 − f | u, partial] ([`Reading::left`]); for the readers
//!   linear in the beam (the hypothesis-free `seen`, the field's counts, the pyramid), their product,
//!   E[1 − f | u] ([`Reading::mean_left`]). The pulse branches on it ([`super::pulse::Pass::lik`]).
//! - **One per class of object**, its numbers the site's ([`Settings`]), its table built once per set of numbers and
//!   kept for the process ([`Reading::of`]).
use super::settings::Settings;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The reading's table: u from `U0` dB in steps of `DU`.
const U0: f64 = -60.0;
const DU: f64 = 0.05;
const NU: usize = 1801;
/// The partial fills' grid: log-spaced from 1e-5 to 1.
const NF: usize = 240;
const G0: f64 = -5.0;
/// The darkening's quadrature nodes (quantiles of a half-t).
const ND: usize = 24;

/// One class's reading of its objects' fills: per u on the table, π(u) and E[1 − f | u, partial] (the module docs).
#[derive(Debug)]
pub struct Reading {
    stop: Vec<f64>,
    left: Vec<f64>,
}

/// A class's numbers: the full hit's noise (dB, t scale) and its dof, the level's sd (dB), the darkening's scale at
/// fill 0 (dB), the share of full fills and the partial fills' Beta(α, β).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadingModel {
    pub noise: f64,
    pub dof: f64,
    pub level: f64,
    pub edge: f64,
    pub full: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl ReadingModel {
    /// The medium's echoes' class (crown leaf) from the site's numbers.
    pub fn leaf(em: &Settings) -> Self {
        ReadingModel { noise: em.leaf_noise, dof: em.leaf_dof, level: em.leaf_level, edge: em.leaf_edge, full: em.leaf_full,
                       alpha: em.leaf_alpha, beta: em.leaf_beta }
    }

    fn key(&self) -> [u64; 7] {
        [self.noise, self.dof, self.level, self.edge, self.full, self.alpha, self.beta].map(f64::to_bits)
    }
}

/// The quantiles of |t_ν| at (k + ½) / n, by integrating its density on a fine grid.
fn half_t_quantiles(nu: f64, n: usize) -> Vec<f64> {
    let pdf = |x: f64| (1.0 + x * x / nu).powf(-0.5 * (nu + 1.0));
    let (dx, xmax) = (1e-3, 400.0);
    let steps = (xmax / dx) as usize;
    let mut cdf = Vec::with_capacity(steps + 1);
    let mut acc = 0.0;
    cdf.push(0.0);
    for i in 0..steps {
        acc += 0.5 * (pdf(i as f64 * dx) + pdf((i + 1) as f64 * dx)) * dx;
        cdf.push(acc);
    }
    (0..n).map(|k| {
        let target = (k as f64 + 0.5) / n as f64 * acc;
        let i = cdf.partition_point(|&c| c < target).min(steps);
        i as f64 * dx
    }).collect()
}

fn ln_beta_pdf(f: f64, a: f64, b: f64) -> f64 {
    (a - 1.0) * f.ln() + (b - 1.0) * (1.0 - f).ln()
}

impl Reading {
    /// The reading of a class with numbers `m`, its table computed once per process ([`Reading::of`] keeps it).
    pub fn new(m: &ReadingModel) -> Self {
        let s = m.noise.hypot(m.level).max(1e-3);
        let nu = m.dof;
        let t = |e: f64| (1.0 + (e / s) * (e / s) / nu).powf(-0.5 * (nu + 1.0)) / s;
        let q = half_t_quantiles(nu, ND);
        // the partial fills and their prior masses
        let dg = -G0 / NF as f64;
        let fills: Vec<f64> = (0..NF).map(|i| 10f64.powf(G0 + (i as f64 + 0.5) * dg)).collect();
        let lw: Vec<f64> = fills.iter().map(|&f| ln_beta_pdf(f, m.alpha, m.beta) + (f * std::f64::consts::LN_10 * dg).ln()).collect();
        let mx = lw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = lw.iter().map(|x| (x - mx).exp()).collect();
        let total: f64 = w.iter().sum();
        let w: Vec<f64> = w.iter().map(|x| x / total * (1.0 - m.full)).collect();
        let (mut stop, mut left) = (Vec::with_capacity(NU), Vec::with_capacity(NU));
        for i in 0..NU {
            let u = U0 + i as f64 * DU;
            let full = m.full * t(u);
            let (mut z, mut l) = (0.0, 0.0);
            for (k, &f) in fills.iter().enumerate() {
                let sd = m.edge * (1.0 - f);
                let e = u - 10.0 * f.log10();
                // the darkening d ≤ 0: the reading less the fill's is n + ℓ + d, so n + ℓ = e − d = e + sd q
                let like: f64 = q.iter().map(|&qk| t(e + sd * qk)).sum::<f64>() / ND as f64;
                z += w[k] * like;
                l += w[k] * like * (1.0 - f);
            }
            stop.push(if full + z > 0.0 { full / (full + z) } else { 0.0 });
            left.push(if z > 0.0 { l / z } else { 0.0 });
        }
        Reading { stop, left }
    }

    /// The reading of a class with numbers `m`, computed once per process and shared.
    pub fn of(m: &ReadingModel) -> Arc<Reading> {
        use std::cell::RefCell;
        static KEPT: OnceLock<Mutex<HashMap<[u64; 7], Arc<Reading>>>> = OnceLock::new();
        // each thread keeps the last it read, so a pulse's read takes no lock
        thread_local!(static LAST: RefCell<Option<([u64; 7], Arc<Reading>)>> = const { RefCell::new(None) });
        let key = m.key();
        if let Some(r) = LAST.with(|l| l.borrow().as_ref().filter(|x| x.0 == key).map(|x| x.1.clone())) {
            return r;
        }
        let kept = KEPT.get_or_init(Default::default);
        let r = kept.lock().unwrap().get(&key).cloned();
        let r = r.unwrap_or_else(|| {
            let r = Arc::new(Reading::new(m));
            kept.lock().unwrap().insert(key, r.clone());
            r
        });
        LAST.with(|l| *l.borrow_mut() = Some((key, r.clone())));
        r
    }

    fn at(table: &[f64], u: f64) -> f64 {
        let x = ((u - U0) / DU).clamp(0.0, (NU - 1) as f64);
        let (i, a) = ((x.floor() as usize).min(NU - 2), x - x.floor());
        table[i] * (1.0 - a) + table[i + 1] * a
    }

    /// π(u): the chance the object filled the beam it met.
    pub fn stop(&self, u: f64) -> f64 {
        Reading::at(&self.stop, u)
    }

    /// E[1 − f | u, partial]: the beam left behind an object that didn't fill it.
    pub fn left(&self, u: f64) -> f64 {
        Reading::at(&self.left, u)
    }

    /// E[1 − f | u]: the beam left on average, for the readers linear in it.
    pub fn mean_left(&self, u: f64) -> f64 {
        (1.0 - self.stop(u)) * self.left(u)
    }

    /// The reading of an echo of reflectance `x` (dB) where `beam` of the emitted beam met it.
    pub fn reading(em: &Settings, beam: f64, x: f64) -> f64 {
        x - em.refl_full_hit() - 10.0 * beam.max(1e-300).log10()
    }

    /// The beam left on average after an echo of reflectance `x` (dB; NaN: unknown, the beam goes on) where `beam`
    /// met it: the readers linear in the beam.
    pub fn beam_after(&self, em: &Settings, beam: f64, x: f64) -> f64 {
        if !x.is_finite() {
            return beam;
        }
        if beam <= 0.0 {
            return 0.0;
        }
        beam * self.mean_left(Reading::reading(em, beam, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reading is bounded: up to a full hit, a brighter reading never means less chance of a full fill or more
    /// beam left; above it a partial fill can only read as bright through the full hit's own noise, so the object
    /// filled the beam or nearly (the t's tails flatten the evidence far above, toward the prior's own near-full
    /// mass: no patch); a faint echo is most likely a partial fill, which leaves most of the beam.
    #[test]
    fn a_brighter_reading_means_more_fill() {
        let em = Settings::default();
        let r = Reading::new(&ReadingModel::leaf(&em));
        let us: Vec<f64> = (0..=120).map(|i| -30.0 + 0.25 * i as f64).collect();
        for w in us.windows(2) {
            assert!(r.stop(w[1]) >= r.stop(w[0]) - 1e-9 && r.mean_left(w[1]) <= r.mean_left(w[0]) + 1e-9, "{w:?}");
        }
        for u in [0.0, 1.0, 3.0] {
            assert!(r.stop(u) > 0.7 && r.mean_left(u) < 0.03, "{u}: {} {}", r.stop(u), r.mean_left(u));
        }
        assert!(r.stop(-13.0) < 0.01 && r.left(-13.0) > 0.6);
    }
}
