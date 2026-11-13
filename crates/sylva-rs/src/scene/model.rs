// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What the scene's likelihood assumes about the instrument and what lies behind the elements: the error model
//! ([`ErrorModel`]: the beam, the per-echo noise and bias by surface class, the offset field's spreads, the misfit
//! prior and the dispersion) and the background ([`Background`]: a diffuse echo hazard and a ground plane).

/// The error model, as data. Defaults are AM's calibration (alignment-rca, research 129): per surface class
/// (solid, mid, fragmented) the per-echo noise, the depth bias and the per-scan offset's spread; the export's
/// reflectance cut (sim-clumps' calibration); the VZ-400i beam.
#[derive(Debug, Clone, Copy)]
pub struct ErrorModel {
    /// The beam: its diameter at the exit (m) and full divergence (rad); the footprint at range ρ is
    /// `exit + divergence ρ` (the VZ-400i's).
    pub exit: f64,
    pub divergence: f64,
    /// The lattice step (rad).
    pub step: f64,
    /// Excess at the solid/mid and mid/fragmented boundaries.
    pub class_edges: [f64; 2],
    /// Per-echo noise (m) per class, a t with `nu_e` degrees of freedom.
    pub sigma_e: [f64; 3],
    pub nu_e: f64,
    /// Depth bias per class: an echo reads `-(b + c (range - bias_ref))` behind its surface (b in m, c in m/m).
    /// Measured between views, so a bias common to every view is not in it.
    pub bias: [[f64; 2]; 3],
    pub bias_ref: f64,
    /// The offset field (research 130 offset-field-design.md): its node spacing ℓ (m), its along-beam spread per class within the reference epoch (σ_u ≈ 0.57 σ_L,same, from
    /// alignment-rca's pair spreads and their +0.32 same-near-scan correlation) and the other-epoch part added in
    /// quadrature (σ_E² = (σ_L,other² − σ_L,same²) / 2). Along the beam only: an across-beam part didn't help
    /// same-day held-out calibration (slice 7b).
    pub field_spacing: f64,
    pub field_along: [f64; 3],
    pub field_epoch: [f64; 3],
    /// The export: echoes under `cut_db` after a per-echo reflectance error of `refl_noise_db` are deleted, and
    /// a kept echo is lost anyway with `p_drop` (NaN: each view's measured in-surface deletion, `RangeImage::deletion`).
    /// Echo densities carry a uniform outlier share `p_outlier`.
    pub cut_db: f64,
    pub refl_noise_db: f64,
    pub p_drop: f64,
    pub p_outlier: f64,
    /// The per-element misfit's declared prior: the section and axis depart from the form by an sd of
    /// `misfit[0] + misfit[1] R` (m). Fourier terms past a₀ aren't identifiable here (synthesis §3), so a scale
    /// stands in for them. Each element carries its own misfit (a latent); this is the default a caller may
    /// override.
    pub misfit: [f64; 2],
    /// The ray likelihood's dispersion φ: each cell's log-likelihood ratio counts `1/φ` (with the footprints' `1/m`), for what
    /// the model leaves correlated between cells (departures every cell of a section shares, in one view or all).
    /// AM's 10 is measured from its own cross-scan residuals: per-view radii of one element disagree by √10 times
    /// their per-view sd, with no trend in range (research 130 slice 7d). The simulated stand takes 4, from its
    /// injected-field coverage.
    pub dispersion: f64,
}

impl Default for ErrorModel {
    fn default() -> Self {
        ErrorModel {
            exit: 0.007,
            divergence: 0.35e-3,
            step: 0.5227e-3,
            class_edges: [1.2, 1.8],
            sigma_e: [0.0024, 0.0037, 0.0051],
            nu_e: 4.0,
            bias: [[-0.0007, -0.00003], [-0.00175, -0.000055], [-0.00235, -0.000135]],
            bias_ref: 4.0,
            field_spacing: 2.0,
            field_along: [0.0011, 0.00165, 0.0021],
            field_epoch: [0.0014, 0.0023, 0.0035],
            cut_db: -25.0,
            refl_noise_db: 2.0,
            p_drop: f64::NAN,
            p_outlier: 0.01,
            misfit: [0.002, 0.1],
            dispersion: 10.0,
        }
    }
}

impl ErrorModel {
    /// The beam's diameter (m) at a range.
    pub fn footprint(&self, range: f64) -> f64 {
        self.exit + self.divergence * range
    }

    /// Footprints per cell, `(footprint / cell)²`, at least 1: how many cells see the same spot.
    pub fn multiplicity(&self, range: f64) -> f64 {
        (self.footprint(range) / (self.step * range)).powi(2).max(1.0)
    }

    /// An echo's depth sd for a surface of class `k` whose depth spreads by `spread` over the footprint (m).
    pub fn depth_sd(&self, k: u8, spread: f64) -> f64 {
        (self.sigma_e[k as usize].powi(2) + spread * spread).sqrt()
    }

    /// The surface class (0 solid, 1 mid, 2 fragmented) of an excess; NaN reads fragmented.
    pub fn surface(&self, excess: f64) -> u8 {
        if excess < self.class_edges[0] {
            0
        } else if excess < self.class_edges[1] {
            1
        } else {
            2
        }
    }

    /// A stick's misfit sd (m) at radius r.
    pub fn misfit_sd(&self, r: f64) -> f64 {
        self.misfit[0] + self.misfit[1] * r
    }

    /// How far behind its surface an echo of class `k` at `range` reads (m).
    pub fn depth_bias(&self, k: u8, range: f64) -> f64 {
        let [b, c] = self.bias[k as usize];
        -(b + c * (range - self.bias_ref))
    }
}

/// What lies behind the elements, crudely: the diffuse exported-echo hazard (1/m), a ground plane's height (NaN for
/// none), the elements' and the ground's reflectance (dB at full fill, face-on) and the longest range.
#[derive(Debug, Clone, Copy)]
pub struct Background {
    pub kappa: f64,
    pub ground_z: f64,
    pub wood_db: f64,
    pub ground_db: f64,
    pub t_max: f64,
}

impl Default for Background {
    fn default() -> Self {
        Background { kappa: 0.05, ground_z: f64::NAN, wood_db: -13.8, ground_db: -20.0, t_max: 80.0 }
    }
}

