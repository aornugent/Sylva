// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What the scene's likelihood assumes about the instrument, the site and what lies behind the elements: two
//! settings records a caller sets, [`Settings`] (the instrument and site calibration, the declared priors and class
//! evidence) and [`Background`] (a diffuse echo hazard and a ground plane). Their defaults are AM's: a RIEGL VZ-2000i
//! on Alice Mulga's raw scans (the corrected MTA import, 3dtrees research 130 slice 13b). Each calibrated value names
//! the 3dtrees research 130 driver that measures it, and a site's values live in its settings file (research 130
//! `settings/<site>.json`, written by those drivers), so another scanner or plot is a file, not code. What stays
//! constant here is method, not calibration: the alignment error's node spacing, the depth bias's reference range,
//! the outlier share and the longest range. The lattice step is each image's own
//! ([`super::range_image::RangeImage::step`]).

/// The range the depth bias's slope is taken from (m).
pub const BIAS_REF: f64 = 4.0;
/// The scan alignment error's node spacing ℓ (m).
pub const ALIGN_SPACING: f64 = 2.0;
/// Echo densities carry a uniform outlier share.
pub const P_OUTLIER: f64 = 0.01;
/// The longest range (m).
pub const T_MAX: f64 = 80.0;

params! {
    /// The instrument and site calibration and the declared priors, per surface class where it differs by class
    /// (0 solid, 1 mid, 2 fragmented).
    #[derive(Debug, Clone, Copy)]
    pub struct Settings {
        /// The beam: its diameter at the exit (m) and full divergence (rad); the footprint at range ρ is
        /// `beam_exit + beam_divergence ρ`. The VZ-2000i's datasheet; a uniform disc, as AM's raw edges measure it
        /// (research 130 slice 11, `edges.py`).
        pub beam_exit: f64 = 0.007,
        pub beam_divergence: f64 = 0.35e-3,
        /// Excess at the solid/mid and mid/fragmented boundaries of the surface class (`constants.py rca`: on raw
        /// scans they split alignment-rca's symptom, 1.1 / 2.8 / 4.4 mm between cells).
        pub class_edge_mid: f64 = 1.2,
        pub class_edge_fragmented: f64 = 1.8,
        /// Per-echo noise (m) per class, a t with `noise_dof` degrees of freedom (`constants.py rca`).
        pub noise_solid: f64 = 0.00227,
        pub noise_mid: f64 = 0.00337,
        pub noise_fragmented: f64 = 0.00504,
        pub noise_dof: f64 = 4.3,
        /// Depth bias: an echo off a surface of class k reads `-(bias_k + bias_slope_k (range - BIAS_REF))` behind it
        /// (m, m/m). Measured between views, so a bias common to every view is not in it (`constants.py rca`).
        pub bias_solid: f64 = -0.000195,
        pub bias_mid: f64 = -0.00068,
        pub bias_fragmented: f64 = -0.00142,
        pub bias_slope_solid: f64 = -0.000028,
        pub bias_slope_mid: f64 = -0.000034,
        pub bias_slope_fragmented: f64 = -0.000067,
        /// The detection floor: an echo is recorded when its reflectance (dB, the surface's at full fill and face-on
        /// plus `10 log10` of its energy share) clears `floor_db + floor_slope log10(max(ρ, floor_ref) / floor_ref)`
        /// at range ρ ([`Settings::floor`]), within a spread `detect_sd` (dB). The scanner's floor is in amplitude,
        /// so in reflectance it rises with range (`constants.py floor`, from the faintest echoes).
        pub floor_db: f64 = -40.5,
        pub floor_slope: f64 = 15.0,
        pub floor_ref: f64 = 7.0,
        pub detect_sd: f64 = 2.0,
        /// The surfaces' reflectance (dB at full fill, face-on): wood (a stick renders with it), leaf (a patch) and
        /// the ground. The dimmer of wood and leaf is the reference the foliage field reads an echo's energy share
        /// against ([`Settings::refl_full_hit`]), so a full hit on either reads as the whole beam. AM's wood is measured on its candidates' interior
        /// echoes (`reflectance.py am`); leaf and ground are the stand's materials (sim-mulga's sim-clumps), AM's not
        /// measured yet (`photometry.py` reads the stand's per element).
        pub refl_wood: f64 = -15.0,
        pub refl_leaf: f64 = -14.4,
        pub refl_ground: f64 = -20.0,
        /// The receiver resolving two returns `d` m apart whose energies differ by `e` dB (the nearer's over the
        /// farther's): `logit p = res_a + res_b d + res_c e`. AM's VZ-2000i, fitted on its raw edges with a
        /// single-echo surface behind (research 130 slice 12b, `pulses.py resolution`): 0.95 m at equal energies.
        pub res_a: f64 = -2.563,
        pub res_b: f64 = 2.693,
        pub res_c: f64 = -0.122,
        /// Located edges (3dtrees research 130 step 2): an echo's reflectance (dB) is read beside its range. A return's
        /// predicted reflectance is `10 log10` of its members' energy shares (each fill × the beam left × cos^k ι ×
        /// the surface's reflectance, k `refl_incidence`, which sets the detection too; a merged return sums them), so
        /// a silhouette's partial cells place its edge within a cell. About it an echo reads with a Student t of
        /// `refl_dof` degrees of freedom and scale `refl_noise ⊕ refl_edge (1 − fill)` (dB: bark's own spread, and the
        /// edge model's, growing as the return's fill falls), truncated where the detection floor cuts it;
        /// `refl_noise` 0 leaves reflectance out. Each (element, scan) has its own reflectance level, an offset from
        /// its surface's reflectance with prior sd `refl_level_sd` (dB), marginalised with the alignment error. An echo
        /// nothing in the scene explains (the foliage field's, an outlier) reads `refl_background_mean` ±
        /// `refl_background_sd` (dB). All are measured on single echoes against the scene's own predictions
        /// (`reflectance.py`).
        pub refl_noise: f64 = 0.27,
        pub refl_dof: f64 = 2.7,
        pub refl_edge: f64 = 9.06,
        pub refl_incidence: f64 = 0.25,
        pub refl_level_sd: f64 = 0.65,
        pub refl_background_mean: f64 = -21.18,
        pub refl_background_sd: f64 = 6.24,
        /// The scan alignment error (research 130 offset-field-design.md): an along-beam shift per (scan, place),
        /// its spread per class within the reference epoch (σ_u ≈ 0.57 σ_L,same, from alignment-rca's pair spreads
        /// and their +0.32 same-near-scan correlation), and the other-epoch part added in quadrature
        /// (σ_E² = (σ_L,other² − σ_L,same²) / 2) (`constants.py rca`). Along the beam only: an across-beam part
        /// didn't help same-day held-out calibration (slice 7b).
        pub align_along_solid: f64 = 0.00084,
        pub align_along_mid: f64 = 0.00128,
        pub align_along_fragmented: f64 = 0.00175,
        pub align_epoch_solid: f64 = 0.00183,
        pub align_epoch_mid: f64 = 0.00197,
        pub align_epoch_fragmented: f64 = 0.00249,
        /// The shape mismatch's declared prior: an element's section and axis depart from its form by an sd of
        /// `mismatch_base + mismatch_slope R` (m). Fourier terms past a₀ aren't identifiable here (synthesis §3), so
        /// a scale stands in for them. Each element carries its own mismatch; this is its default.
        pub mismatch_base: f64 = 0.002,
        pub mismatch_slope: f64 = 0.1,
        /// The evidence discount per cell φ: each cell's log-likelihood ratio counts `1/φ` (with the footprints'
        /// `1/m`), for what the model leaves correlated between cells (departures every cell of a section shares, in
        /// one view or all). AM's is measured from its own cross-scan residuals: per-view radii of one element
        /// disagree by √φ times their per-view sd (`discount.py`, slice 7d; 10.5 on raw).
        pub discount: f64 = 10.0,
        /// The stem front shift (research 130 slice 14): a section's departure from round, as each view sees it, is
        /// an along-beam shift of its front per (element, scan), with prior sd
        /// `front_shift_base + front_shift_slope R` (m), marginalised with the alignment error. Chosen among three
        /// values on AM's same-day held-out check (slice 14); to be measured from training views' front residuals
        /// instead (the review after slice 15).
        pub front_shift_base: f64 = 0.001,
        pub front_shift_slope: f64 = 0.1,
        /// Class evidence (research 130 slice 15), per (element, scan): from the first echoes on the element (fill >
        /// 0.5), their mean reflectance (dB), albedo (reflectance less 10 log10 cos ι), share of pulses with more
        /// than one return and mean cos ι, a log likelihood ratio wood against leaf,
        /// `class_b + Σ class_w · features`, fitted on the stand (`photometry.py`). `class_temper` scales it so an
        /// element's sum over its scans is calibrated (the scans of one element are correlated); 0 leaves it out.
        /// The stand's fit: per-element AUC 0.95. The temper that calibrates an element's sum is 0.386 (the
        /// element-level scale 0.94 over the design effect `1 + (S − 1) ρ` at ρ 0.71 and a typical S of 3 views); it
        /// defaults to 0 (off), since among sticks of one form it only adds noise to radius comparisons (the stand's
        /// truth beats R × 0.7 falls from 0.967 to 0.95 with it on). A caller weighing forms turns it on.
        pub class_b: f64 = 11.262,
        pub class_w_refl: f64 = 1.0747,
        pub class_w_albedo: f64 = -0.5962,
        pub class_w_multi: f64 = -3.4553,
        pub class_w_cos: f64 = -0.4111,
        pub class_temper: f64 = 0.0,
    }
}

params! {
    /// What lies behind the elements, crudely: a diffuse extinction `kappa` (1/m) along every ray (the foliage field
    /// outside its box, and its prior mean inside: [`super::foliage`]) and a ground plane's height (NaN for none).
    /// κ is declared at 0.05, the checks' setting; `extinction.py kappa` measures it per site (AM 0.034, the stand
    /// 0.019).
    #[derive(Debug, Clone, Copy)]
    pub struct Background {
        pub kappa: f64 = 0.05,
        pub ground_z: f64 = f64::NAN,
    }
}

impl Settings {
    /// The beam's diameter (m) at a range.
    pub fn footprint(&self, range: f64) -> f64 {
        self.beam_exit + self.beam_divergence * range
    }

    /// Footprints per cell, `(footprint / cell)²`, at least 1, on a lattice of step `step` (rad): how many cells see
    /// the same spot.
    pub fn multiplicity(&self, range: f64, step: f64) -> f64 {
        (self.footprint(range) / (step * range)).powi(2).max(1.0)
    }

    /// The one weight a cell's evidence counts with everywhere (the ray likelihood, the foliage field, predictions):
    /// `1 / (φ m(ρ))`, at ρ the cell's first echo's range, or its predicted range where it has none.
    pub fn cell_weight(&self, range: f64, step: f64) -> f64 {
        1.0 / (self.discount * self.multiplicity(range, step))
    }

    /// The per-echo noise sd (m) of class `k`.
    pub fn noise(&self, k: u8) -> f64 {
        [self.noise_solid, self.noise_mid, self.noise_fragmented][k as usize]
    }

    /// An echo's depth sd for a surface of class `k` whose depth spreads by `spread` over the footprint (m).
    pub fn depth_sd(&self, k: u8, spread: f64) -> f64 {
        (self.noise(k).powi(2) + spread * spread).sqrt()
    }

    /// The surface class (0 solid, 1 mid, 2 fragmented) of an excess; NaN reads fragmented.
    pub fn surface(&self, excess: f64) -> u8 {
        if excess < self.class_edge_mid {
            0
        } else if excess < self.class_edge_fragmented {
            1
        } else {
            2
        }
    }

    /// The detection floor (dB) at range `range`.
    pub fn floor(&self, range: f64) -> f64 {
        self.floor_db + self.floor_slope * (range.max(self.floor_ref) / self.floor_ref).log10()
    }

    /// The chance an echo of energy share `e` (fill × cos) off a surface of reflectance `db` (dB) at range `range` is
    /// recorded: above the detection floor.
    pub fn p_detect(&self, db: f64, e: f64, range: f64) -> f64 {
        0.5 * libm::erfc(-((db + 10.0 * e.max(1e-12).log10() - self.floor(range)) / self.detect_sd) / std::f64::consts::SQRT_2)
    }

    /// The density of an echo's reflectance `x` (dB) given its return's predicted reflectance `pred`, give or take
    /// `spread` (dB, the element's own uncertainty), and fill `fill` at range `range`, as recorded: the per-echo noise,
    /// truncated by the detection floor (its chance of clearing the floor is the return's own, so this is the
    /// reflectance's density given that it was recorded; the truncation is read as a Gaussian's). 1 when `x` is
    /// unknown or reflectance is off.
    pub fn refl_density(&self, x: f64, pred: f64, spread: f64, fill: f64, range: f64) -> f64 {
        if !x.is_finite() || self.refl_noise <= 0.0 {
            return 1.0;
        }
        let s = self.refl_noise.hypot(self.refl_edge * (1.0 - fill.clamp(0.0, 1.0))).hypot(spread);
        let fl = self.floor(range);
        let phi = |z: f64| 0.5 * libm::erfc(-z / std::f64::consts::SQRT_2);
        let nu = self.refl_dof;
        let t = (libm::lgamma(0.5 * (nu + 1.0)) - libm::lgamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln()
                 - 0.5 * (nu + 1.0) * (1.0 + ((x - pred) / s).powi(2) / nu).ln()).exp() / s;
        t * phi((x - fl) / self.detect_sd) / phi((pred - fl) / (s * s + self.detect_sd * self.detect_sd).sqrt()).max(1e-12)
    }

    /// The density of the reflectance `x` (dB) of an echo nothing in the scene explains; 1 when `x` is unknown or
    /// reflectance is off.
    pub fn refl_background(&self, x: f64) -> f64 {
        if !x.is_finite() || self.refl_noise <= 0.0 {
            return 1.0;
        }
        let z = (x - self.refl_background_mean) / self.refl_background_sd;
        (-0.5 * z * z).exp() / (self.refl_background_sd * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// The receiver's resolution (m): the separation at which two equal returns come back as two echoes half the
    /// time (the 50 % point of [`Settings::p_resolve`]).
    pub fn resolution(&self) -> f64 {
        -self.res_a / self.res_b
    }

    /// The range up to which a cell's pulse shows clear space (m), from its first echo (NaN for none): the echo less
    /// the receiver's resolution, since a nearer surface within it would have merged into the echo; the longest range
    /// for a miss. 3dtrees research 130 step 3c: what carving free space reads.
    pub fn free_to(&self, first_echo: f64) -> f64 {
        if first_echo.is_finite() { (first_echo - self.resolution()).max(0.0) } else { T_MAX }
    }

    /// The chance two returns `d` m apart, energies `e` dB apart (nearer over farther), come back as two echoes.
    pub fn p_resolve(&self, d: f64, e: f64) -> f64 {
        1.0 / (1.0 + (-(self.res_a + self.res_b * d + self.res_c * e)).exp())
    }

    /// Class evidence from one scan's echoes on an element, `[n, Σ reflectance, Σ albedo, Σ multi-return, Σ cos]`:
    /// the tempered log likelihood ratio, wood against leaf (0 under 5 echoes).
    pub fn class_llr(&self, f: &[f64; 5]) -> f64 {
        if f[0] < 5.0 || self.class_temper == 0.0 {
            return 0.0;
        }
        let m = [f[1] / f[0], f[2] / f[0], f[3] / f[0], f[4] / f[0]];
        self.class_temper * (self.class_b + self.class_w_refl * m[0] + self.class_w_albedo * m[1] + self.class_w_multi * m[2] + self.class_w_cos * m[3])
    }

    /// The reflectance (dB) the foliage field reads a full hit at, an echo's energy share being its reflectance
    /// against this: the dimmer canopy surface, wood or leaf, so a full hit on either reads as the whole beam (on AM
    /// the measured wood, −15.0 dB, is dimmer than the declared leaf, −14.4: research 130 step 3a).
    pub fn refl_full_hit(&self) -> f64 {
        self.refl_wood.min(self.refl_leaf)
    }

    /// The stem front shift's prior sd (m) at radius r.
    pub fn front_shift_sd(&self, r: f64) -> f64 {
        self.front_shift_base + self.front_shift_slope * r
    }

    /// An element's default shape mismatch sd (m) at radius r.
    pub fn mismatch_sd(&self, r: f64) -> f64 {
        self.mismatch_base + self.mismatch_slope * r
    }

    /// How far behind its surface an echo of class `k` at `range` reads (m).
    pub fn depth_bias(&self, k: u8, range: f64) -> f64 {
        let (b, c) = [(self.bias_solid, self.bias_slope_solid), (self.bias_mid, self.bias_slope_mid),
                      (self.bias_fragmented, self.bias_slope_fragmented)][k as usize];
        -(b + c * (range - BIAS_REF))
    }

    /// The scan alignment error's along-beam spread (m) at an element of class `k`, with the other-epoch part if
    /// `other`.
    pub fn align_sd(&self, k: u8, other: bool) -> f64 {
        let along = [self.align_along_solid, self.align_along_mid, self.align_along_fragmented][k as usize];
        let epoch = [self.align_epoch_solid, self.align_epoch_mid, self.align_epoch_fragmented][k as usize];
        if other { (along.powi(2) + epoch.powi(2)).sqrt() } else { along }
    }
}
