// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The scene's tests' shared scans and elements: a pole before a wall, its scene's cost and rows.

use crate::shots::reconstruct::dir;
use std::sync::Arc;
use super::foliage::FoliageBox;
use super::range_image::RangeImage;
use super::range_image::RangeImageParams;
use super::render::{Form, Class, Element};
use super::scene::{ElementScan, Proposal, Scene};
use super::settings::{Background, Settings};

/// The test scan's receiver: returns closer than this merge (m).
pub(super) const RESOLUTION: f64 = 0.8;

/// A scan of a vertical pole (R = 3 cm, 4 m away, centred on the y axis) in front of a wall at 8 m (see
/// [`pole_scan`]).
pub(super) fn pole_image() -> RangeImage {
    pole_scan(0.03, 4.0, 4.0)
}

/// The pole with nothing behind it: a pulse that misses it, or isn't recorded on it, returns no echo.
pub(super) fn sky_pole_view() -> Arc<RangeImage> {
    Arc::new(pole_scan(0.03, 4.0, f64::INFINITY))
}

pub(super) fn thin_pole_view(r_pole: f64, y_pole: f64) -> Arc<RangeImage> {
    Arc::new(pole_scan(r_pole, y_pole, y_pole))
}

/// A scan of a vertical pole (radius `r`, `y` m away on the y axis) in front of a wall `gap` m behind it, with the
/// measured beam (research 130 slice 11): a uniform disc of the footprint's diameter, sampled here across the
/// pole by 401 lines (its 1-D marginal, a semicircle), each part's energy its cosine (Lambertian; to the default
/// incidence power for its reflectance and detection). The pole's echo sits at its parts' energy-weighted depth and is recorded where it clears the detection floor (wood at the
/// default reflectance, 2 dB spread); the wall (−20 dB, face-on) takes the rest of the beam. Returns closer than the
/// receiver's resolution (0.8 m) merge into one echo at the measured pull behind the pole's ([`Settings::pull`]); otherwise the first is the
/// pole's if recorded, else the wall's. Each echo's reflectance is its energy share's (merged: summed) with the
/// default per-echo noise.
pub(super) fn pole_scan(r_pole: f64, y_pole: f64, gap: f64) -> RangeImage {
    pole_scan_of(r_pole, y_pole, gap, Settings::default().refl_wood)
}

/// [`pole_scan`] with the pole's wood reflecting `wood` dB.
pub(super) fn pole_scan_of(r_pole: f64, y_pole: f64, gap: f64, wood: f64) -> RangeImage {
    let (period, interval, zstep) = (6.6e-3, 0.825e-6, 0.03);
    let em = Settings::default();
    let (mut t, mut d, mut r, mut refl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let u = |s: u64| ((s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    for l in 0..200usize {
        for k in 0..400usize {
            let (zen, az) = (84.0 + zstep * k as f64, -3.0 + zstep * l as f64);
            let w = dir(zen, az);
            let h = (w[0] * w[0] + w[1] * w[1]).sqrt();
            let miss = y_pole * w[0] / h;
            let tm = y_pole * w[1] / (h * h);
            let a = 0.5 * em.footprint(tm);
            let (mut f, mut ec, mut ek, mut ez) = (0.0, 0.0, 0.0, 0.0);
            for q in 0..401 {
                let z = -1.0 + 2.0 * (q as f64 + 0.5) / 401.0;
                let wt = (1.0 - z * z).sqrt() * 2.0 / 401.0 * 2.0 / std::f64::consts::PI;
                let x = miss + z * a;
                if x.abs() < r_pole {
                    let c = (1.0 - (x / r_pole).powi(2)).sqrt();
                    f += wt;
                    ec += wt * c;
                    ek += wt * c.powf(em.refl_incidence);
                    ez += wt * c * (tm - r_pole * c / h);
                }
            }
            let seed = (l * 400 + k) as u64 * 2 + 1;
            let n = 2.0 * (-2.0 * u(seed).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919)).cos();
            let n_refl = em.refl_noise * (-2.0 * u(seed * 31).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(seed * 7919 * 31)).cos();
            let kept = ec > 0.0 && wood + 10.0 * ek.max(1e-12).log10() + n > super::instrument::Instrument::default().floor(tm);
            let wall = if gap.is_finite() { (y_pole + gap) / w[1] } else { f64::NAN };
            let (ep, ew) = (ek * 10f64.powf(wood / 10.0), (1.0 - f) * 0.01);
            let (range, db) = if !kept {
                (wall, 10.0 * ew.max(1e-12).log10())
            } else if wall - ez / ec < RESOLUTION {
                (ez / ec + em.pull(wall - ez / ec, 10.0 * (ep / ew.max(1e-300)).log10()), 10.0 * (ep + ew).log10())
            } else {
                (ez / ec, 10.0 * ep.log10())
            };
            refl.push((db + n_refl) as f32);
            t.push(l as f64 * period + k as f64 * interval);
            d.push(w);
            r.push(range as f32);
        }
    }
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut v = RangeImage::from_first_echoes(&t, &d, &r, id, [0.0; 3], 0, &RangeImageParams::default());
    v.echo_refl = refl;
    v
}

pub(super) fn pole_view() -> Arc<RangeImage> {
    Arc::new(pole_image())
}

/// A scene's cost and rows.
pub(super) fn cost(v: &Arc<RangeImage>, els: &[Element], em: &Settings, sc: &Background) -> (f64, Vec<ElementScan>) {
    let s = Scene::new(vec![v.clone()], els, *em, *sc, None, 0);
    (s.cost(), s.rows(&Proposal::default()))
}

/// Keep one element: its id and the cost change.
pub(super) fn add(c: &mut Scene, e: Element) -> (u32, f64) {
    let (ids, d) = c.keep(&Proposal::put(&[e]));
    (ids[0], d)
}

/// Take one element out: the cost change.
pub(super) fn remove(c: &mut Scene, id: u32) -> f64 {
    c.keep(&Proposal::take(&[id])).1
}

/// Swap element `id` for `e`: its new id and the cost change.
pub(super) fn swap(c: &mut Scene, id: u32, e: Element) -> (u32, f64) {
    let (ids, d) = c.keep(&Proposal { put_in: vec![e], take_out: vec![id] });
    (ids[0], d)
}

/// Each hypothesis's cost to put in, per scan (and the field's part).
pub(super) fn costs(c: &Scene, hyps: &[Vec<Element>]) -> Vec<Vec<f64>> {
    c.score(&hyps.iter().map(|h| Proposal::put(h)).collect::<Vec<_>>())
}

pub(super) fn stick(x: f64, y: f64, r: f64) -> Element {
    Element { form: Form::Stick { a: [x, y, -0.2], b: [x, y, 0.2] }, class: Class::Wood, r, opacity: 1.0, mismatch: 0.002, group: None, key: None }
}

/// The field's box about the test scan's pole and wall.
pub(super) fn field_box() -> FoliageBox {
    FoliageBox { lo: [-0.5, 2.9, -0.3], hi: [0.5, 8.9, 0.3], voxel: 0.1 }
}

/// A piece of the test pole, from height `lo` to `hi`, in group `g`.
pub(super) fn piece_of(lo: f64, hi: f64, g: Option<u32>) -> Element {
    Element { form: Form::Stick { a: [0.0, 4.0, lo], b: [0.0, 4.0, hi] }, group: g, ..stick(0.0, 4.0, 0.03) }
}
