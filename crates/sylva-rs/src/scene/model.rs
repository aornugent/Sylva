// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What the scene's likelihood assumes about the instrument and what lies behind the elements. The calibration no
//! caller sets is constants here: the VZ-400i beam and lattice, the surface classes, the per-echo noise, the offset
//! field's spacing, the detection floor, the outlier share and the surfaces' reflectance. What a caller may set is
//! two settings records: [`ErrorModel`] (the depth bias and the offset field's spreads by surface class, the misfit
//! prior and the dispersion) and [`Background`] (a diffuse echo hazard and a ground plane). Defaults are AM's raw
//! scans' calibration (the corrected MTA import, slice 13b): alignment-rca (research 129) rerun on the raw scans, the floor from their faintest echoes and
//! φ from their cross-scan residuals (research 130 `constants.py`, `dispersion.py`).

/// The beam: its diameter at the exit (m) and full divergence (rad); the footprint at range ρ is `EXIT + DIVERGENCE ρ`.
pub const EXIT: f64 = 0.007;
pub const DIVERGENCE: f64 = 0.35e-3;
/// The lattice step (rad).
pub const STEP: f64 = 0.5227e-3;
/// Excess at the solid/mid and mid/fragmented boundaries; the classes are 0 solid, 1 mid, 2 fragmented. On raw
/// scans they still split alignment-rca's symptom (1.1 / 2.8 / 4.4 mm between cells), so they stay.
pub const CLASS_EDGES: [f64; 2] = [1.2, 1.8];
/// Per-echo noise (m) per class, a t with `NU_E` degrees of freedom.
pub const SIGMA_E: [f64; 3] = [0.00227, 0.00337, 0.00504];
pub const NU_E: f64 = 4.3;
/// The range the depth bias's slope is taken from (m).
pub const BIAS_REF: f64 = 4.0;
/// The offset field's node spacing ℓ (m).
pub const FIELD_SPACING: f64 = 2.0;
/// The detection floor: an echo is recorded when its reflectance (dB, the surface's at full fill and face-on plus
/// `10 log10` of its energy share) clears `FLOOR_DB + FLOOR_SLOPE log10(max(ρ, FLOOR_REF) / FLOOR_REF)` at range ρ
/// ([`floor_db`]), within a spread `DETECT_SD` (dB). The scanner's floor is in amplitude, so in reflectance it rises
/// with range.
pub const FLOOR_DB: f64 = -40.5;
pub const FLOOR_SLOPE: f64 = 15.0;
pub const FLOOR_REF: f64 = 7.0;
pub const DETECT_SD: f64 = 2.0;
// The instrument: the beam (EXIT, DIVERGENCE), the lattice (STEP), the detection floor and the receiver's resolution
// are AM's VZ-2000i, as is the default resolution curve (`ErrorModel::res_*`, a setting since the stand's
// simulator has its own); its cap on targets per pulse is read from the data (`RangeImage::max_targets`). Another
// scanner is a change of these constants and settings, not of code.

/// Echo densities carry a uniform outlier share.
pub const P_OUTLIER: f64 = 0.01;
/// The elements' and the ground's reflectance (dB at full fill, face-on).
pub const WOOD_DB: f64 = -13.8;
pub const GROUND_DB: f64 = -20.0;
/// The longest range (m).
pub const T_MAX: f64 = 80.0;

params! {
    /// The error model's settings, per surface class where it differs by class.
    #[derive(Debug, Clone, Copy)]
    pub struct ErrorModel {
        /// Depth bias: an echo off a surface of class k reads `-(bias_k + bias_slope_k (range - BIAS_REF))` behind it
        /// (m, m/m). Measured between views, so a bias common to every view is not in it.
        pub bias_solid: f64 = -0.000195,
        pub bias_mid: f64 = -0.00068,
        pub bias_fragmented: f64 = -0.00142,
        pub bias_slope_solid: f64 = -0.000028,
        pub bias_slope_mid: f64 = -0.000034,
        pub bias_slope_fragmented: f64 = -0.000067,
        /// The offset field (research 130 offset-field-design.md): its along-beam spread per class within the
        /// reference epoch (σ_u ≈ 0.57 σ_L,same, from alignment-rca's pair spreads and their +0.32 same-near-scan
        /// correlation), and the other-epoch part added in quadrature (σ_E² = (σ_L,other² − σ_L,same²) / 2). Along the
        /// beam only: an across-beam part didn't help same-day held-out calibration (slice 7b).
        pub field_along_solid: f64 = 0.00084,
        pub field_along_mid: f64 = 0.00128,
        pub field_along_fragmented: f64 = 0.00175,
        pub field_epoch_solid: f64 = 0.00183,
        pub field_epoch_mid: f64 = 0.00197,
        pub field_epoch_fragmented: f64 = 0.00249,
        /// The per-element misfit's declared prior: the section and axis depart from the form by an sd of
        /// `misfit_base + misfit_slope R` (m). Fourier terms past a₀ aren't identifiable here (synthesis §3), so a
        /// scale stands in for them. Each element carries its own misfit (a latent); this is its default.
        pub misfit_base: f64 = 0.002,
        pub misfit_slope: f64 = 0.1,
        /// The ray likelihood's dispersion φ: each cell's log-likelihood ratio counts `1/φ` (with the footprints'
        /// `1/m`), for what the model leaves correlated between cells (departures every cell of a section shares, in
        /// one view or all). AM's is measured from its own cross-scan residuals: per-view radii of one element
        /// disagree by √φ times their per-view sd (research 130 slice 7d; 10.5 on raw). The simulated stand takes 4, from its
        /// injected-field coverage.
        pub dispersion: f64 = 10.0,
        /// The receiver resolving two returns `d` m apart whose energies differ by `e` dB (the nearer's over the
        /// farther's): `logit p = res_a + res_b d + res_c e`. AM's VZ-2000i, fitted on its raw edges with a single-echo
        /// surface behind (research 130 slice 12b, `pulses.py resolution`); the stand's simulator resolves at 0.8 m
        /// sharp, a step.
        pub res_a: f64 = -2.563,
        pub res_b: f64 = 2.693,
        pub res_c: f64 = -0.122,
    }
}

params! {
    /// What lies behind the elements, crudely: a diffuse extinction `kappa` (1/m) along every ray (the diffuse field
    /// outside its box, and its prior mean inside: [`super::extinction`]) and a ground plane's height (NaN for none).
    #[derive(Debug, Clone, Copy)]
    pub struct Background {
        pub kappa: f64 = 0.05,
        pub ground_z: f64 = f64::NAN,
    }
}

impl ErrorModel {
    /// The beam's diameter (m) at a range.
    pub fn footprint(&self, range: f64) -> f64 {
        EXIT + DIVERGENCE * range
    }

    /// Footprints per cell, `(footprint / cell)²`, at least 1: how many cells see the same spot.
    pub fn multiplicity(&self, range: f64) -> f64 {
        (self.footprint(range) / (STEP * range)).powi(2).max(1.0)
    }

    /// An echo's depth sd for a surface of class `k` whose depth spreads by `spread` over the footprint (m).
    pub fn depth_sd(&self, k: u8, spread: f64) -> f64 {
        (SIGMA_E[k as usize].powi(2) + spread * spread).sqrt()
    }

    /// The surface class (0 solid, 1 mid, 2 fragmented) of an excess; NaN reads fragmented.
    pub fn surface(&self, excess: f64) -> u8 {
        if excess < CLASS_EDGES[0] {
            0
        } else if excess < CLASS_EDGES[1] {
            1
        } else {
            2
        }
    }

    /// The chance an echo of energy share `e` (fill × cos) off a surface of reflectance `db` (dB) at range `range` is
    /// recorded: above the detection floor.
    pub fn p_detect(&self, db: f64, e: f64, range: f64) -> f64 {
        0.5 * libm::erfc(-((db + 10.0 * e.max(1e-12).log10() - floor_db(range)) / DETECT_SD) / std::f64::consts::SQRT_2)
    }

    /// The receiver's resolution (m): the separation at which two equal returns come back as two echoes half the
    /// time (the 50 % point of [`ErrorModel::p_resolve`]).
    pub fn resolution(&self) -> f64 {
        -self.res_a / self.res_b
    }

    /// The chance two returns `d` m apart, energies `e` dB apart (nearer over farther), come back as two echoes.
    pub fn p_resolve(&self, d: f64, e: f64) -> f64 {
        1.0 / (1.0 + (-(self.res_a + self.res_b * d + self.res_c * e)).exp())
    }

    /// A stick's misfit sd (m) at radius r.
    pub fn misfit_sd(&self, r: f64) -> f64 {
        self.misfit_base + self.misfit_slope * r
    }

    /// How far behind its surface an echo of class `k` at `range` reads (m).
    pub fn depth_bias(&self, k: u8, range: f64) -> f64 {
        let (b, c) = [(self.bias_solid, self.bias_slope_solid), (self.bias_mid, self.bias_slope_mid),
                      (self.bias_fragmented, self.bias_slope_fragmented)][k as usize];
        -(b + c * (range - BIAS_REF))
    }

    /// The offset field's along-beam spread (m) at an element of class `k`, with the other-epoch part if `other`.
    pub fn field_sd(&self, k: u8, other: bool) -> f64 {
        let along = [self.field_along_solid, self.field_along_mid, self.field_along_fragmented][k as usize];
        let epoch = [self.field_epoch_solid, self.field_epoch_mid, self.field_epoch_fragmented][k as usize];
        if other { (along.powi(2) + epoch.powi(2)).sqrt() } else { along }
    }
}

/// The detection floor (dB) at range `range`.
pub fn floor_db(range: f64) -> f64 {
    FLOOR_DB + FLOOR_SLOPE * (range.max(FLOOR_REF) / FLOOR_REF).log10()
}

