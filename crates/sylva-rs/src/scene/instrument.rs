// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The instrument that recorded a scan (3dtrees research 130 observation §2, core.md S3): what turns a pulse into
//! targets, owned by each [`super::range_image::RangeImage`] and read by everything that reads its pulses. A new
//! scanner is a new record, a data file 3dtrees keeps per instrument and programme (`settings/instruments/`), filled
//! from the scanner's own metadata where it writes it.
//!
//! - **The cap** (`max_targets`): the most targets the programme reports per pulse (the VZ-2000i at 1.2 MHz: 4, the
//!   RXP's `meas_max_targets`). Every target the scanner reported takes a slot, those the import drops too.
//! - **The detection floor** (`floor`): the reflectance (dB) of a return at the detection threshold, by range, as a
//!   table every `floor_step` m from 0, interpolated linearly and held flat past its end. On a RIEGL it is the scanner's
//!   own range table (reflectance = amplitude + table, `riegl.reflectance_calculation`), read off any scan's echoes as
//!   reflectance − amplitude. The import's weak rule (echoes under `weak_amplitude` dB of amplitude within
//!   `weak_range` m dropped) raises it by that much there.
//!
//! Site values (the surfaces' reflectance, the field's prior, the merge map) stay in
//! [`super::settings::Settings`].
use super::dual::Real;

/// The instrument record (the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Instrument {
    /// The most targets the programme reports per pulse.
    pub max_targets: usize,
    /// The reflectance (dB) at the detection threshold every `floor_step` m from range 0.
    pub floor: Vec<f32>,
    pub floor_step: f64,
    /// The import's weak rule: echoes under this amplitude (dB) within `weak_range` (m) are dropped.
    pub weak_amplitude: f64,
    pub weak_range: f64,
}

impl Default for Instrument {
    /// The floor 3dtrees declared before the scanner's table was read (`−40.5 + 15 log10(max(ρ, 7)/7)` dB, the
    /// faintest AM echoes' fit; the stand's simulator renders at it), a cap of 4 and no weak rule: the tests' scanner.
    fn default() -> Self {
        Instrument::log_floor(-40.5, 15.0, 7.0, 4)
    }
}

impl Instrument {
    /// The floor's table spacing (m) and its last range (m).
    pub const STEP: f64 = 0.25;
    pub const END: f64 = super::settings::T_MAX;

    /// An instrument whose floor is `db + slope log10(max(ρ, reference) / reference)` (a fit, not a scanner's table:
    /// the simulator's), sampled on the table, with cap `max_targets` and no weak rule.
    pub fn log_floor(db: f64, slope: f64, reference: f64, max_targets: usize) -> Self {
        let n = (Self::END / Self::STEP).round() as usize + 1;
        let floor = (0..n).map(|i| (db + slope * ((i as f64 * Self::STEP).max(reference) / reference).log10()) as f32).collect();
        Instrument { max_targets, floor, floor_step: Self::STEP, weak_amplitude: 0.0, weak_range: 0.0 }
    }

    /// The scanner's table read off echoes' ranges, reflectances and amplitudes (dB): per `STEP` bin the median of
    /// reflectance − amplitude, bins without echoes taking their nearest neighbour's. None without any.
    pub fn table_from_echoes(range: &[f32], refl: &[f32], amplitude: &[f32]) -> Option<Vec<f32>> {
        let n = (Self::END / Self::STEP).round() as usize + 1;
        let mut bins: Vec<Vec<f32>> = vec![Vec::new(); n];
        for ((&r, &x), &a) in range.iter().zip(refl).zip(amplitude) {
            let i = (r as f64 / Self::STEP).round();
            if x.is_finite() && a.is_finite() && i >= 0.0 && (i as usize) < n {
                bins[i as usize].push(x - a);
            }
        }
        let med: Vec<Option<f32>> = bins.into_iter().map(|mut b| {
            if b.is_empty() {
                return None;
            }
            b.sort_by(f32::total_cmp);
            Some(b[b.len() / 2])
        }).collect();
        let known: Vec<usize> = (0..n).filter(|&i| med[i].is_some()).collect();
        if known.is_empty() {
            return None;
        }
        Some((0..n).map(|i| {
            let j = known.partition_point(|&k| k < i);
            let near = [j.checked_sub(1), (j < known.len()).then_some(j)].into_iter().flatten()
                .min_by_key(|&j| known[j].abs_diff(i)).unwrap();
            med[known[near]].unwrap()
        }).collect())
    }

    /// The detection floor (dB of reflectance) at range `range`: the table, interpolated, plus the weak rule's step.
    pub fn floor<P: Real>(&self, range: P) -> P {
        let x = (range.v() / self.floor_step).max(0.0);
        let last = self.floor.len() - 1;
        let i = (x.floor() as usize).min(last.saturating_sub(1));
        let (a, b) = (self.floor[i] as f64, self.floor[(i + 1).min(last)] as f64);
        let at = if x >= last as f64 { P::c(b) } else { (range / self.floor_step - i as f64) * (b - a) + a };
        if range.v() < self.weak_range { at + self.weak_amplitude } else { at }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scanner's table read back from echoes whose reflectance is amplitude plus the table, and the weak rule's
    /// step at its range (3dtrees research 130 S3).
    #[test]
    fn the_floor_is_the_scanners_table_plus_the_weak_rule() {
        let table = |r: f64| -60.0 + 20.0 * r.max(1.0).log10();
        let (mut range, mut refl, mut amp) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..20000 {
            let r = 0.5 + 0.004 * i as f64;
            let a = (i % 37) as f64 * 0.5;
            range.push(r as f32);
            amp.push(a as f32);
            refl.push((a + table(r)) as f32);
        }
        let floor = Instrument::table_from_echoes(&range, &refl, &amp).unwrap();
        let inst = Instrument { max_targets: 4, floor, floor_step: Instrument::STEP, weak_amplitude: 2.5, weak_range: 18.0 };
        for r in [3.0, 10.0, 17.9, 18.1, 50.0] {
            let want = table(r) + if r < 18.0 { 2.5 } else { 0.0 };
            assert!((inst.floor(r) - want).abs() < 0.05, "at {r}: {} against {want}", inst.floor(r));
        }
        assert_eq!(inst.floor(200.0), inst.floor(Instrument::END));
        let old = Instrument::default();
        assert!((old.floor(20.0) - (-40.5 + 15.0 * (20.0f64 / 7.0).log10())).abs() < 0.01);
    }
}
