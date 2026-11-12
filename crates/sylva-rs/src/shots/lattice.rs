// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scan's native angular lattice: every pulse the scanner fired has a cell
//! `(line, col)`, so a cell no recorded pulse falls in is a known missing
//! pulse (no echo, or an echo deleted in export).
//!
//! A RIEGL scanner sweeps the zenith at constant speed along each mirror
//! line and steps the head in azimuth between lines; each line's pulses are
//! contiguous in time. Time is the key, not the recorded direction (which a
//! point export rounds to its coordinate grid, coarse at short range):
//!
//! - **line:** each line's epoch (the time its sweep passes zenith 90°,
//!   the median over its pulses of `t - (zenith - 90°) / ω`) is a whole
//!   number of line periods after the last, so lines with no recorded
//!   pulse (open sky) still get their index;
//! - **col:** within a line, pulses sit on slots one pulse interval apart.
//!   The slot phase is the circular mean of `t / interval` over the line
//!   (the scanner's pulse-position jitter, a few tenths of an interval,
//!   averages out); a pulse's column is its slot counted from the slot
//!   nearest the line's epoch, so columns line up across lines to within
//!   half a step (`line_phase`).
//!
//! Directions must be in the scanner's own frame (undo the SOP first):
//! a tilted frame bends the lines.

use super::reconstruct::{median, pulse_interval, quantile, scan_lines, zen_az};
use crate::Point;
use std::f64::consts::TAU;

/// What [`scan_lattice`] found, and every pulse's cell.
#[derive(Debug, Clone, Default)]
pub struct Lattice {
    /// Line of each pulse (`u32::MAX` if it could not be placed).
    pub line: Vec<u32>,
    /// Column of each pulse along its line.
    pub col: Vec<u32>,
    /// Pulse time from its slot, in pulse intervals, in `[-0.5, 0.5)`.
    pub slot_residual: Vec<f32>,
    /// Recorded zenith from the cell centre's, in zenith steps.
    pub zenith_residual: Vec<f32>,
    /// Recorded azimuth from the cell centre's, in azimuth steps.
    pub azimuth_residual: Vec<f32>,
    pub n_lines: usize,
    pub n_cols: usize,
    /// Pulse interval (s).
    pub pulse_interval: f64,
    /// Line period (s).
    pub line_period: f64,
    /// Zenith step between slots of a line (degrees, signed by the sweep).
    pub zenith_step: f64,
    /// Azimuth step between lines (degrees, signed).
    pub azimuth_step: f64,
    /// Zenith of column 0 at `line_phase` 0 (degrees).
    pub zenith0: f64,
    /// Per line, its slots' offset from the common columns, in steps (NaN for a line without pulses).
    pub line_phase: Vec<f64>,
    /// Per line, its epoch's offset from a regular line clock, in line periods (NaN without pulses).
    pub line_jitter: Vec<f64>,
    /// Per line, its azimuth off the uniform head turn, in azimuth steps (NaN without pulses).
    pub line_azimuth: Vec<f64>,
    /// Per column, the cell centre's zenith and azimuth beyond the linear
    /// sweep and head turn (degrees), shared by every line.
    pub zenith_profile: Vec<f64>,
    pub azimuth_profile: Vec<f64>,
    /// Recorded pulses per cell, row-major `(line, col)`, saturating at 255.
    pub count: Vec<u8>,
    /// Per line, column 0's cell-centre azimuth before the column profile (degrees, unwrapped along the lines;
    /// lines without pulses interpolated), and the head's turn per column along a line (degrees): a cell centre's
    /// azimuth is `line_azimuth0[l] + azimuth_per_col * c + azimuth_profile[c]` ([`Lattice::cell_direction`]).
    pub line_azimuth0: Vec<f64>,
    pub azimuth_per_col: f64,
}

impl Lattice {
    fn phase0(&self, l: usize) -> f64 {
        let p = self.line_phase[l];
        if p.is_finite() { p } else { 0.0 }
    }

    /// The cell centre's (zenith, azimuth) in degrees, at a fractional cell (the profiles at the nearest column).
    pub fn cell_angles(&self, l: f64, c: f64) -> (f64, f64) {
        let li = (l.round().max(0.0) as usize).min(self.n_lines - 1);
        let ci = (c.round().max(0.0) as usize).min(self.n_cols - 1);
        let zen = self.zenith0 + (c + self.phase0(li)) * self.zenith_step + self.zenith_profile[ci];
        let a0 = self.line_azimuth0[li] + (l - li as f64) * self.azimuth_step;
        (zen, a0 + self.azimuth_per_col * c + self.azimuth_profile[ci])
    }

    /// The unit direction (scanner frame) of a cell's centre.
    pub fn cell_direction(&self, l: f64, c: f64) -> Point {
        let (z, a) = self.cell_angles(l, c);
        super::reconstruct::dir(z, a)
    }

    /// The fractional cell `(line, col)` a direction (scanner frame) falls in: the cell-centre model inverted by
    /// fixed-point steps; `None` outside the lattice. Where the first and last lines overlap (a sweep of a little
    /// over 360°), the earlier line.
    pub fn cell_of(&self, d: &Point) -> Option<(f64, f64)> {
        if self.n_lines < 2 || self.n_cols == 0 {
            return None;
        }
        let (zq, aq) = zen_az(d);
        let (s, a0) = (self.azimuth_step, self.line_azimuth0[0]);
        let wrap = |x: f64| x - 360.0 * (x / 360.0).round();
        let mut c = (zq - self.zenith0) / self.zenith_step;
        let mut l = 0.0;
        for _ in 0..4 {
            let ci = (c.round().max(0.0) as usize).min(self.n_cols - 1);
            let a = aq - self.azimuth_per_col * c - self.azimuth_profile[ci];
            let d0 = if s > 0.0 { (a - a0).rem_euclid(360.0) } else { -(a0 - a).rem_euclid(360.0) };
            let li = ((d0 / s).round().max(0.0) as usize).min(self.n_lines - 1);
            l = li as f64 + wrap(a - self.line_azimuth0[li]) / s;
            let li = (l.round().max(0.0) as usize).min(self.n_lines - 1);
            c = (zq - self.zenith0 - self.zenith_profile[ci]) / self.zenith_step - self.phase0(li);
        }
        (l > -0.5 && l < self.n_lines as f64 - 0.5 && c > -0.5 && c < self.n_cols as f64 - 0.5).then_some((l, c))
    }
}

/// The native lattice of one scan's pulses, given in acquisition order:
/// `time` (s, the pulse's shot time) and `direction` in the scanner's frame.
/// Returns an empty lattice for fewer than two lines' worth of pulses.
pub fn scan_lattice(time: &[f64], direction: &[Point]) -> Lattice {
    let n = time.len();
    let mut out = Lattice::default();
    if n < 2 {
        return out;
    }
    let t0 = time[0];
    let t: Vec<f64> = time.iter().map(|&x| x - t0).collect();
    let (zen, az): (Vec<f64>, Vec<f64>) = direction.iter().map(zen_az).unzip();
    let starts = scan_lines(&zen, Some(&t)).starts;
    let Some(interval) = pulse_interval(&t, &starts) else {
        return out;
    };
    // Sweep rate (deg/s) over pulses sixteen apart within a line.
    let mut rates = Vec::new();
    for w in starts.windows(2) {
        for i in (w[0] + 16..w[1]).step_by(4) {
            let dt = t[i] - t[i - 16];
            if dt > 0.0 && dt < 64.0 * interval {
                rates.push((zen[i] - zen[i - 16]) / dt);
            }
        }
    }
    let omega = median(&mut rates);
    if !(omega.abs() > 0.0) {
        return out;
    }
    // Each segment's epoch, then line periods between them.
    let epoch: Vec<f64> = starts
        .windows(2)
        .map(|w| median(&mut (w[0]..w[1]).map(|i| t[i] - (zen[i] - 90.0) / omega).collect::<Vec<_>>()))
        .collect();
    let mut gaps: Vec<f64> = epoch.windows(2).map(|e| e[1] - e[0]).filter(|&d| d > 0.0).collect();
    let mut period = quantile(&mut gaps, 0.5);
    for _ in 0..2 {
        let (mut num, mut den) = (0.0, 0.0);
        for e in epoch.windows(2) {
            let k = ((e[1] - e[0]) / period).round();
            if k >= 1.0 && ((e[1] - e[0]) / period - k).abs() < 0.1 {
                num += e[1] - e[0];
                den += k;
            }
        }
        if den > 0.0 {
            period = num / den;
        }
    }
    if !(period > 0.0) {
        return out;
    }
    // Line index of each segment: whole periods on from the previous one
    // (segments a stray split left within one line share its index).
    let mut seg_line = vec![0i64; epoch.len()];
    for s in 1..epoch.len() {
        seg_line[s] = seg_line[s - 1] + ((epoch[s] - epoch[s - 1]) / period).round().max(0.0) as i64;
    }
    let n_lines = (seg_line.last().copied().unwrap_or(0) + 1) as usize;
    // Per line: epoch (segments' median), slot phase.
    let mut line_epoch = vec![f64::NAN; n_lines];
    let mut line_slot = vec![f64::NAN; n_lines]; // slot phase psi in [0, 1): slots at (m + psi) * interval
    let mut s = 0;
    while s < epoch.len() {
        let l = seg_line[s];
        let mut e = s;
        while e < epoch.len() && seg_line[e] == l {
            e += 1;
        }
        let (a, b) = (starts[s], starts[e]);
        line_epoch[l as usize] = median(&mut (a..b).map(|i| t[i] - (zen[i] - 90.0) / omega).collect::<Vec<_>>());
        let (mut c, mut sn) = (0.0, 0.0);
        for &ti in &t[a..b] {
            let ph = TAU * (ti / interval).fract();
            c += ph.cos();
            sn += ph.sin();
        }
        line_slot[l as usize] = (sn.atan2(c) / TAU).rem_euclid(1.0);
        s = e;
    }
    // A regular line clock through the epochs, for the jitter report.
    let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (l, &e) in line_epoch.iter().enumerate() {
        if e.is_finite() {
            let x = l as f64;
            sx += x;
            sy += e;
            sxx += x * x;
            sxy += x * e;
            m += 1.0;
        }
    }
    let slope = if m > 1.0 { (m * sxy - sx * sy) / (m * sxx - sx * sx) } else { period };
    let icpt = (sy - slope * sx) / m.max(1.0);
    out.line_jitter = line_epoch.iter().enumerate().map(|(l, &e)| (e - icpt - slope * l as f64) / slope).collect();
    period = slope;

    let step = omega * interval;
    // Per pulse: line, slot relative to the slot nearest the epoch.
    let mut line_of = vec![u32::MAX; n];
    let mut rel = vec![0i64; n];
    let mut slot_res = vec![0f32; n];
    let mut phase = vec![f64::NAN; n_lines];
    for (seg, w) in starts.windows(2).enumerate() {
        let l = seg_line[seg] as usize;
        let (e, psi) = (line_epoch[l], line_slot[l]);
        let m0 = (e / interval - psi).round();
        phase[l] = m0 + psi - e / interval;
        for i in w[0]..w[1] {
            let x = t[i] / interval - psi;
            let k = x.round();
            line_of[i] = l as u32;
            rel[i] = (k - m0) as i64;
            slot_res[i] = (x - k) as f32;
        }
    }
    let lo = *rel.iter().min().unwrap();
    let hi = *rel.iter().max().unwrap();
    let n_cols = (hi - lo + 1) as usize;
    // Azimuth: head rotation linear in time, fitted by least squares through
    // each line's median azimuth (unwrapped along the lines) at its epoch.
    let mut med_az = vec![f64::NAN; n_lines];
    for (seg, w) in starts.windows(2).enumerate() {
        let l = seg_line[seg] as usize;
        if med_az[l].is_nan() {
            med_az[l] = median(&mut az[w[0]..w[1]].to_vec());
        }
    }
    let mut prev: Option<f64> = None;
    let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for l in 0..n_lines {
        if med_az[l].is_finite() {
            let a = match prev {
                Some(p) => {
                    let d = med_az[l] - p;
                    p + d - 360.0 * (d / 360.0).round()
                }
                None => med_az[l],
            };
            prev = Some(a);
            let x = line_epoch[l];
            sx += x;
            sy += a;
            sxx += x * x;
            sxy += x * a;
            m += 1.0;
        }
    }
    let az_rate = (m * sxy - sx * sy) / (m * sxx - sx * sx); // deg/s
    let az0 = (sy - az_rate * sx) / m;
    let az_step = az_rate * period;

    let mut count = vec![0u8; n_lines * n_cols];
    let mut col = vec![0u32; n];
    let mut zres = vec![0f32; n];
    let mut ares = vec![0f32; n];
    for i in 0..n {
        let l = line_of[i] as usize;
        let c = (rel[i] - lo) as u32;
        col[i] = c;
        let cell = &mut count[l * n_cols + c as usize];
        *cell = cell.saturating_add(1);
        let t_slot = (rel[i] as f64 + phase[l]) * interval + line_epoch[l];
        zres[i] = (zen[i] - (90.0 + omega * (t_slot - line_epoch[l]))) as f32;
        let d = az[i] - (az0 + az_rate * t_slot);
        ares[i] = (d - 360.0 * (d / 360.0).round()) as f32;
    }
    // What the linear model leaves that every line shares, per column: the
    // mirror's departure from constant speed (zenith), and the beam's plane
    // missing the scanner's axis (azimuth varying along a line).
    let profile = |res: &[f32]| -> Vec<f64> {
        let mut by: Vec<Vec<f64>> = vec![Vec::new(); n_cols];
        for i in 0..n {
            by[col[i] as usize].push(res[i] as f64);
        }
        by.iter_mut().map(|v| if v.is_empty() { 0.0 } else { median(v) }).collect()
    };
    // Each line's azimuth off the uniform head turn (the head wobbles a
    // step or so over a revolution, and each mirror facet sits a little
    // apart): the line's median with the shared profile out, so cells
    // centre on their line; then the profile again.
    let ap0 = profile(&ares);
    let mut by_line: Vec<Vec<f64>> = vec![Vec::new(); n_lines];
    for i in 0..n {
        by_line[line_of[i] as usize].push(ares[i] as f64 - ap0[col[i] as usize]);
    }
    let line_az: Vec<f64> = by_line.iter_mut().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
    drop(by_line);
    for i in 0..n {
        ares[i] -= line_az[line_of[i] as usize] as f32;
    }
    let (zp, ap) = (profile(&zres), profile(&ares));
    for i in 0..n {
        let c = col[i] as usize;
        zres[i] = ((zres[i] as f64 - zp[c]) / step) as f32;
        ares[i] = ((ares[i] as f64 - ap[c]) / az_step) as f32;
    }
    // Column 0's azimuth per line, before the profile: the head turn at its slot plus the line's own offset;
    // lines without pulses on the uniform turn.
    out.line_azimuth0 = (0..n_lines)
        .map(|l| {
            if line_epoch[l].is_finite() && line_az[l].is_finite() {
                az0 + az_rate * (line_epoch[l] + (lo as f64 + phase[l]) * interval) + line_az[l]
            } else {
                f64::NAN
            }
        })
        .collect();
    let first = out.line_azimuth0.iter().position(|a| a.is_finite()).unwrap_or(0);
    let base = out.line_azimuth0[first] - first as f64 * az_step;
    for (l, a) in out.line_azimuth0.iter_mut().enumerate() {
        if !a.is_finite() {
            *a = base + l as f64 * az_step;
        }
    }
    out.azimuth_per_col = az_rate * interval;
    out.line_azimuth = line_az.iter().map(|a| a / az_step).collect();
    out.zenith_profile = zp;
    out.azimuth_profile = ap;
    out.line = line_of;
    out.col = col;
    out.slot_residual = slot_res;
    out.zenith_residual = zres;
    out.azimuth_residual = ares;
    out.n_lines = n_lines;
    out.n_cols = n_cols;
    out.pulse_interval = interval;
    out.line_period = period;
    out.zenith_step = step;
    out.azimuth_step = az_step;
    out.zenith0 = 90.0 + lo as f64 * step;
    out.line_phase = phase;
    out.count = count;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shots::reconstruct::dir;

    /// A RIEGL-like scan: `lines` lines 6.6 ms apart, slots 0.825 us apart
    /// with a cycling pulse-position jitter of +-0.1 interval (as measured), the zenith
    /// sweeping 30-130 deg at 0.03 deg per slot, the head turning 0.033 deg
    /// per line (continuously, so azimuth drifts along a line too), and each
    /// line's slots at their own phase. `keep(line, slot)` says which pulses
    /// were recorded. Returns time, direction and each pulse's true (line, slot).
    fn scan(lines: usize, keep: impl Fn(usize, usize) -> bool) -> (Vec<f64>, Vec<Point>, Vec<(usize, usize)>) {
        let (period, interval, zstep, astep) = (6.6e-3, 0.825e-6, 0.03, 0.033);
        let omega = zstep / interval;
        let slots = (100.0 / zstep) as usize;
        let jitter = [0.0, 0.1, -0.05, 0.08, -0.1, 0.04];
        let (mut t, mut d, mut truth) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..lines {
            let phase = (l as f64 * 0.37).fract() - 0.5; // slots' phase within the line, against the columns
            let eps = 1.0e-6 * ((l * 7) % 3) as f64; // the line clock's jitter, 0-2 us
            for k in 0..slots {
                if !keep(l, k) {
                    continue;
                }
                let zen = 30.0 + zstep * (k as f64 + phase + jitter[(l * slots + k) % 6]);
                let ti = 100.0 + l as f64 * period + eps + (zen - 30.0) / omega;
                let az = 10.0 + astep * (ti - 100.0) / period;
                t.push(ti);
                d.push(dir(zen, az));
                truth.push((l, k));
            }
        }
        (t, d, truth)
    }

    fn check(lines: usize, keep: impl Fn(usize, usize) -> bool + Copy) -> Lattice {
        let (t, d, truth) = scan(lines, keep);
        let lat = scan_lattice(&t, &d);
        assert!((lat.pulse_interval - 0.825e-6).abs() < 1e-10, "{}", lat.pulse_interval);
        assert!((lat.zenith_step - 0.03).abs() < 1e-4, "{}", lat.zenith_step);
        assert!((lat.azimuth_step - 0.033).abs() < 1e-4, "{}", lat.azimuth_step);
        assert!((lat.line_period - 6.6e-3).abs() < 1e-7);
        assert_eq!(lat.n_lines, lines);
        // Every pulse is in its true cell, up to one shift of the columns.
        let shift = truth[0].1 as i64 - lat.col[0] as i64;
        for (i, &(l, k)) in truth.iter().enumerate() {
            assert_eq!((lat.line[i] as usize, lat.col[i] as i64 + shift), (l, k as i64), "pulse {i}");
            assert!(lat.slot_residual[i].abs() < 0.2);
            assert!(lat.zenith_residual[i].abs() < 0.3 && lat.azimuth_residual[i].abs() < 0.6);
        }
        // Each recorded pulse fills one cell; the rest are exactly the dropped ones.
        assert!(lat.count.iter().all(|&c| c <= 1));
        // The cell-centre model inverts: each pulse's direction falls in its own cell.
        for (i, dd) in d.iter().enumerate().step_by(7) {
            let (l, c) = lat.cell_of(dd).expect("inside");
            assert!((l - lat.line[i] as f64).abs() < 0.5 && (c - lat.col[i] as f64).abs() < 0.5, "pulse {i}: ({l}, {c}) vs ({}, {})", lat.line[i], lat.col[i]);
        }
        for l in 0..lines {
            for c in 0..lat.n_cols {
                let k = c as i64 + shift;
                let recorded = k >= 0 && k < 3333 && keep(l, k as usize);
                assert_eq!(lat.count[l * lat.n_cols + c] == 1, recorded, "cell ({l}, {c})");
            }
        }
        lat
    }

    #[test]
    fn a_full_scan_fills_every_cell() {
        let lat = check(12, |_, _| true);
        assert_eq!(lat.n_cols, 3333);
    }

    #[test]
    fn dropped_pulses_are_empty_cells_and_sky_lines_keep_their_index() {
        // Sky at the top of every line, a hole of deleted pulses, scattered
        // single drops, and line 5 with nothing recorded at all.
        let keep = |l: usize, k: usize| k >= 400 && l != 5 && !(l == 3 && (1000..1050).contains(&k)) && (l * 3333 + k) % 97 != 0;
        let lat = check(12, keep);
        assert!(lat.line_phase[5].is_nan());
    }
}
