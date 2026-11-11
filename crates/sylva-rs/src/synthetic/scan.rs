// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A terrestrial scanner with a finite beam, cast into a point scene.
//!
//! Pulses leave `origin` on a regular zenith / azimuth grid (the two-axis
//! pattern of a terrestrial scanner, as in [`crate::synthetic::scan`]),
//! fired in acquisition order: one line per azimuth step, sweeping the
//! zenith upward, optionally only over an azimuth window. Each echo carries
//! its pulse's `gps_time` (line `line_period` apart, slots `pulse_interval`
//! apart, slot 0 at zenith 0), so the scan's lattice
//! ([`crate::shots::lattice`]) can be rebuilt from it as from a real scan. A
//! pulse is a cone of full divergence `divergence` leaving an aperture of
//! `exit_diameter`, so that its footprint at range `R` has diameter
//! `exit_diameter + R divergence`; it is sampled by `footprint_samples`
//! equal-energy sub-beams spread over the footprint disc on a sunflower
//! (Vogel 1979) pattern. Each point stands for a small patch of surface of
//! radius `target_radius`: a disc facing along the point's normal (the
//! `normal_x`, `normal_y`, `normal_z` attributes when the cloud has them,
//! as the synthetic trees and plots do, else a local PCA over 10 neighbours
//! where the neighbourhood is planar), or a sphere where no normal is
//! known. A sub-beam stops at the first patch it meets: at the disc's plane,
//! or, for a sphere, at the point's closest approach, so a surface is not
//! brought forward by the radius. Discs seen edge-on are not hit, so a
//! stem's silhouette is not widened by the radius (a sphere widens it).
//! Solid capped cylinders ([`Solid`]) can stand beside the points: exact,
//! opaque wood, met at its surface.
//!
//! The receiver separates echoes as a discrete-return scanner does: hits
//! closer in range than `echo_separation` form one echo. With
//! `mixed_pixels` its range is the energy-weighted mean of the hits' ranges
//! (energy: the sub-beam's share times the target's reflectance), which is
//! how a footprint straddling an edge produces a point in the empty space
//! between foreground and background; without it the echo takes the range
//! of its strongest hit. Echoes with less energy than `detection_threshold`
//! are lost; at most `max_echoes` are kept, nearest first. Gaussian range
//! noise of standard deviation `range_noise + range_noise_slope * R` is then
//! added along the beam. Every pulse is kept, with or without echoes, so
//! the free space it crossed is known.
//!
//! Echo positions lie on the pulse's central axis. Echo attributes are those
//! of the point with the most energy in the echo, plus `reflectance` (dB,
//! `10 log10` of the echo energy: 0 dB is a white target filling the
//! footprint, as RIEGL's calibrated reflectance), `footprint_fraction` (the
//! share of sub-beams in the echo) and `range_spread` (m, the spread of the
//! hits' ranges within the echo, before noise).
//!
//! Random numbers come from [`Generator`], `max_echoes` normal draws per
//! pulse in firing order, so that a seed gives the same scan on any number
//! of threads.
//!
//! [`export`] then turns a scan into what a scanner's export holds: weak
//! and partial echoes deleted, pulses without an echo dropped, and the
//! scan's systematic errors (a range law, a turn of its directions, a
//! smooth local displacement field) applied.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::shots::ops::packed_starts;
use crate::util::limits;
use crate::{Point, PointCloud, Shots};

/// Reflectance of the scene points.
#[derive(Debug, Clone, PartialEq)]
pub enum Reflectance {
    /// By `classification`: `(class, reflectance)` pairs and a default for
    /// other classes and clouds without the attribute.
    ByClass(Vec<(i64, f64)>, f64),
    /// From a per-point attribute.
    Attribute(String),
    /// The same for every point.
    Constant(f64),
}

impl Default for Reflectance {
    /// Diffuse reflectance near 1550 nm (the wavelength of RIEGL's VZ
    /// scanners): ground 0.3, understorey 0.3, leaf 0.3, wood 0.5, other 0.4.
    fn default() -> Self {
        Reflectance::ByClass(vec![(2, 0.3), (3, 0.3), (4, 0.3), (5, 0.5)], 0.4)
    }
}

/// Settings of [`scan_beam`]; angles in degrees, lengths in metres.
#[derive(Debug, Clone)]
pub struct BeamScan {
    pub origin: Point,
    pub resolution_deg: f64,
    pub min_zenith_deg: f64,
    pub max_zenith_deg: f64,
    pub max_echoes: usize,
    pub echo_separation: f64,
    pub range_noise: f64,
    pub range_noise_slope: f64,
    /// Full beam divergence (mrad).
    pub divergence_mrad: f64,
    pub exit_diameter: f64,
    pub footprint_samples: usize,
    pub mixed_pixels: bool,
    /// Radius of the surface patch a point stands for; `None` takes 1.75 times
    /// the median spacing of the scene.
    pub target_radius: Option<f64>,
    /// Orient the patches (discs) where a normal is known; spheres otherwise.
    pub oriented: bool,
    pub detection_threshold: f64,
    pub reflectance: Reflectance,
    pub max_range: f64,
    pub seed: u64,
    /// Azimuth window fired (degrees, `0..360` is the whole turn).
    pub min_azimuth_deg: f64,
    pub max_azimuth_deg: f64,
    /// Time between slots of a line and between lines (s): Alice Mulga's VZ-400i, 0.825 us and 6.593 ms.
    pub pulse_interval: f64,
    pub line_period: f64,
    /// Solid cylinders in the scene besides its points (see [`Solid`]). The
    /// points they name stand for them only, and are not patches themselves.
    pub solids: Vec<Solid>,
}

impl BeamScan {
    pub fn new(origin: Point) -> BeamScan {
        BeamScan { origin, resolution_deg: 0.25, min_zenith_deg: 0.0, max_zenith_deg: 130.0, max_echoes: 2, echo_separation: 0.5, range_noise: 0.0, range_noise_slope: 0.0, divergence_mrad: 0.0, exit_diameter: 0.0, footprint_samples: 1, mixed_pixels: true, target_radius: None, oriented: true, detection_threshold: 0.0, reflectance: Reflectance::default(), max_range: f64::INFINITY, seed: 0, min_azimuth_deg: 0.0, max_azimuth_deg: 360.0, pulse_interval: 0.825e-6, line_period: 6.593e-3, solids: Vec::new() }
    }
}

/// Field of view and beam of a scanner model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScannerPreset {
    pub min_zenith_deg: f64,
    pub max_zenith_deg: f64,
    pub divergence_mrad: f64,
    pub exit_diameter: f64,
    pub range_noise: f64,
}

/// Names accepted by [`scanner_preset`].
pub const SCANNERS: [&str; 3] = ["vz400", "vz400i", "vz2000i"];

/// Nominal figures of RIEGL scanners from their data sheets: vertical field
/// of view +60 to -40 degrees (zenith 30 to 130), beam divergence 0.35 mrad
/// (VZ-400, VZ-400i) or 0.27 mrad (VZ-2000i), ranging precision 3 mm. The
/// exit diameter (7 mm) is a nominal value.
pub fn scanner_preset(name: &str) -> Result<ScannerPreset> {
    let p = |div: f64| ScannerPreset { min_zenith_deg: 30.0, max_zenith_deg: 130.0, divergence_mrad: div, exit_diameter: 0.007, range_noise: 0.003 };
    match name.to_ascii_lowercase().replace(['-', ' ', '_'], "").as_str() {
        "vz400" | "vz400i" => Ok(p(0.35)),
        "vz2000i" => Ok(p(0.27)),
        _ => Err(Error::invalid(format!("unknown scanner {name:?}; expected one of {SCANNERS:?}"))),
    }
}

/// Sub-beam offsets in the unit disc: the centre for one sample, else a
/// sunflower pattern of equal-area parts.
pub fn footprint_offsets(n: usize) -> Vec<[f64; 2]> {
    if n <= 1 {
        return vec![[0.0, 0.0]];
    }
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..n).map(|k| {
        let r = ((k as f64 + 0.5) / n as f64).sqrt();
        let a = k as f64 * golden;
        [r * a.cos(), r * a.sin()]
    }).collect()
}

/// Items indexed in a regular 3-D grid for ray casting; each item is listed
/// in every cell its bounding sphere reaches (a point's patch, or a solid
/// cylinder's).
struct Grid {
    lo: Point,
    cell: f64,
    n: [usize; 3],
    start: Vec<u32>,
    idx: Vec<u32>,
}

impl Grid {
    /// A uniform grid over the scene, so a ray only tests the spheres whose
    /// cells it crosses.
    ///
    /// The points are stored in one flat `idx` array grouped by cell, with
    /// `start` giving where each cell's group begins, rather than a list per
    /// cell: one allocation instead of millions, and a cell's points sit next
    /// to each other in memory. Building it is the standard counting sort -
    /// count each cell's points, turn the counts into running starts, then
    /// place each point. A point is entered in every cell its sphere reaches,
    /// which is why one can appear several times.
    ///
    /// The cell size grows until the grid fits in a sensible number of cells,
    /// so a scene spread over a large area does not ask for an enormous and
    /// mostly empty grid.
    fn new(points: &[Point], targets: &[usize], radius: impl Fn(usize) -> f64) -> Result<Grid> {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let mut r_max = 0.0f64;
        for &i in targets {
            let r = radius(i);
            r_max = r_max.max(r);
            for k in 0..3 {
                lo[k] = lo[k].min(points[i][k] - r);
                hi[k] = hi[k].max(points[i][k] + r);
            }
        }
        if targets.is_empty() {
            return Ok(Grid { lo: [0.0; 3], cell: 1.0, n: [0, 0, 0], start: vec![0], idx: vec![] });
        }
        let vol = (0..3).map(|k| (hi[k] - lo[k]).max(r_max)).product::<f64>();
        let cell = (2.0 * r_max).max((vol / (1u64 << 24) as f64).cbrt());
        let n: [usize; 3] = std::array::from_fn(|k| (((hi[k] - lo[k]) / cell).floor() as usize + 1).max(1));
        let cells = n[0] * n[1] * n[2];
        limits::check_cells(cells as u128 + 8 * targets.len() as u128, 4, "the ray-casting index of the scene", "a smaller scene or a larger target_radius")?;
        let span = |v: f64, r: f64, k: usize| (((v - r - lo[k]) / cell).floor().max(0.0) as usize, (((v + r - lo[k]) / cell).floor() as usize).min(n[k] - 1));
        let mut count = vec![0u32; cells + 1];
        let each = |i: usize, f: &mut dyn FnMut(usize)| {
            let (p, r) = (points[i], radius(i));
            let (x0, x1) = span(p[0], r, 0);
            let (y0, y1) = span(p[1], r, 1);
            let (z0, z1) = span(p[2], r, 2);
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        f((z * n[1] + y) * n[0] + x);
                    }
                }
            }
        };
        for &i in targets {
            each(i, &mut |c| count[c + 1] += 1);
        }
        // Running totals: `count[k]` becomes where cell `k`'s points start.
        for k in 1..count.len() {
            count[k] += count[k - 1];
        }
        let mut fill = count.clone();
        let mut idx = vec![0u32; count[cells] as usize];
        for &i in targets {
            each(i, &mut |c| {
                idx[fill[c] as usize] = i as u32;
                fill[c] += 1;
            });
        }
        Ok(Grid { lo, cell, n, start: count, idx })
    }

    /// First item the ray `o + t d` meets at `t` in `(t_min, t_max)`:
    /// `(range, index)`. `hit(i)` is item `i`'s entry range along the ray and
    /// the range it reports, if the ray meets it.
    fn cast(&self, o: &Point, d: &Point, t_min: f64, t_max: f64, hit: impl Fn(usize) -> Option<(f64, f64)>) -> Option<(f64, usize)> {
        if self.n[0] == 0 {
            return None;
        }
        // Clip to the grid box.
        let (mut ta, mut tb) = (0.0f64, t_max);
        for k in 0..3 {
            let hi = self.lo[k] + self.n[k] as f64 * self.cell;
            if d[k] == 0.0 {
                if o[k] < self.lo[k] || o[k] > hi {
                    return None;
                }
            } else {
                let (t0, t1) = ((self.lo[k] - o[k]) / d[k], (hi - o[k]) / d[k]);
                ta = ta.max(t0.min(t1));
                tb = tb.min(t0.max(t1));
            }
        }
        if ta >= tb {
            return None;
        }
        let p0: Point = std::array::from_fn(|k| o[k] + ta * d[k]);
        let mut c: [i64; 3] = std::array::from_fn(|k| (((p0[k] - self.lo[k]) / self.cell).floor() as i64).clamp(0, self.n[k] as i64 - 1));
        let step: [i64; 3] = std::array::from_fn(|k| if d[k] > 0.0 { 1 } else { -1 });
        let mut t_next: [f64; 3] = std::array::from_fn(|k| {
            if d[k] == 0.0 {
                f64::INFINITY
            } else {
                let edge = self.lo[k] + (c[k] + i64::from(step[k] > 0)) as f64 * self.cell;
                (edge - o[k]) / d[k]
            }
        });
        let dt: [f64; 3] = std::array::from_fn(|k| if d[k] == 0.0 { f64::INFINITY } else { self.cell / d[k].abs() });
        let mut best: Option<(f64, f64, usize)> = None;
        loop {
            let t_exit = t_next[0].min(t_next[1]).min(t_next[2]).min(tb);
            let cid = ((c[2] as usize * self.n[1]) + c[1] as usize) * self.n[0] + c[0] as usize;
            for &j in &self.idx[self.start[cid] as usize..self.start[cid + 1] as usize] {
                let i = j as usize;
                let Some((te, tc)) = hit(i) else { continue };
                if te > t_min && te < t_max && best.is_none_or(|(bt, _, bi)| te < bt || (te == bt && i < bi)) {
                    best = Some((te, tc, i));
                }
            }
            if best.is_some_and(|(bt, _, _)| bt <= t_exit) || t_exit >= tb {
                break;
            }
            let k = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] { 0 } else if t_next[1] <= t_next[2] { 1 } else { 2 };
            c[k] += step[k];
            if c[k] < 0 || c[k] >= self.n[k] as i64 {
                break;
            }
            t_next[k] += dt[k];
        }
        best.map(|(_, tc, i)| (tc, i))
    }
}

/// Where the ray `o + t d` meets a point's patch of radius `r` (`r2 = r^2`):
/// a disc facing along a unit normal `n`, or, with a zero normal, a sphere,
/// reported at the point's closest approach so a surface is not brought
/// forward by the radius. `(entry, reported range)`.
fn patch_hit(q: &Point, n: &Point, r2: f64, o: &Point, d: &Point) -> Option<(f64, f64)> {
    let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
    if *n == [0.0; 3] {
        let tc = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        let perp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - tc * tc;
        return (perp2 < r2).then(|| (tc - (r2 - perp2).sqrt(), tc));
    }
    let den = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    if den.abs() < 1e-9 {
        return None;
    }
    let t = (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) / den;
    let w = [o[0] + t * d[0] - q[0], o[1] + t * d[1] - q[1], o[2] + t * d[2] - q[2]];
    (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] < r2).then_some((t, t))
}

/// A solid, opaque capped cylinder of the scene, from `a` to `b` with radius
/// `r`: a sub-beam that meets it stops on its surface, and the echo takes the
/// attributes and reflectance of scene point `point`. Wood drawn this way is
/// exact (its silhouette is `r`, and nothing passes through it), where wood
/// drawn as surface points is only as closed as their patches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solid {
    pub a: Point,
    pub b: Point,
    pub r: f64,
    pub point: usize,
}

/// First entry of the ray `o + t d` into a capped cylinder: the side, or a
/// cap where the ray comes in through an end.
fn solid_hit(s: &Solid, o: &Point, d: &Point) -> Option<(f64, f64)> {
    let dot = |x: &Point, y: &Point| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let ab = [s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2]];
    let len = dot(&ab, &ab).sqrt();
    if !(len > 0.0) {
        return None;
    }
    let u = [ab[0] / len, ab[1] / len, ab[2] / len];
    let w = [o[0] - s.a[0], o[1] - s.a[1], o[2] - s.a[2]];
    let (du, wu) = (dot(d, &u), dot(&w, &u));
    let dp = [d[0] - du * u[0], d[1] - du * u[1], d[2] - du * u[2]];
    let wp = [w[0] - wu * u[0], w[1] - wu * u[1], w[2] - wu * u[2]];
    let r2 = s.r * s.r;
    let mut best = f64::INFINITY;
    let qa = dot(&dp, &dp);
    if qa > 1e-18 {
        let (qb, qc) = (2.0 * dot(&dp, &wp), dot(&wp, &wp) - r2);
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let t = (-qb - disc.sqrt()) / (2.0 * qa);
            let along = wu + t * du;
            if (0.0..=len).contains(&along) {
                best = t;
            }
        }
    }
    if du.abs() > 1e-12 {
        for plane in [0.0, len] {
            let t = (plane - wu) / du;
            let p = [wp[0] + t * dp[0], wp[1] + t * dp[1], wp[2] + t * dp[2]];
            if dot(&p, &p) <= r2 && t < best {
                best = t;
            }
        }
    }
    best.is_finite().then_some((best, best))
}

fn check(p: &BeamScan) -> Result<()> {
    if !p.origin.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid("origin must be finite"));
    }
    if !(p.resolution_deg.is_finite() && p.resolution_deg > 0.0) {
        return Err(Error::invalid(format!("resolution_deg must be positive, got {}", p.resolution_deg)));
    }
    if !(0.0..180.0).contains(&p.min_zenith_deg) || !(p.max_zenith_deg > p.min_zenith_deg && p.max_zenith_deg <= 180.0) {
        return Err(Error::invalid(format!("zenith range must satisfy 0 <= min < max <= 180, got {} to {}", p.min_zenith_deg, p.max_zenith_deg)));
    }
    let non_negative = [("echo_separation", p.echo_separation), ("range_noise", p.range_noise), ("range_noise_slope", p.range_noise_slope), ("beam_divergence", p.divergence_mrad), ("exit_diameter", p.exit_diameter), ("detection_threshold", p.detection_threshold)];
    for (name, v) in non_negative {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
        }
    }
    if !(p.min_azimuth_deg >= 0.0 && p.max_azimuth_deg > p.min_azimuth_deg && p.max_azimuth_deg <= 360.0) {
        return Err(Error::invalid(format!("azimuth window must satisfy 0 <= min < max <= 360, got {} to {}", p.min_azimuth_deg, p.max_azimuth_deg)));
    }
    if !(p.pulse_interval > 0.0 && p.line_period >= (p.max_zenith_deg / p.resolution_deg + 1.0) * p.pulse_interval) {
        return Err(Error::invalid("pulse_interval must be positive and a line's slots must fit in line_period"));
    }
    if p.max_range.is_nan() || p.max_range <= 0.0 {
        return Err(Error::invalid(format!("max_range must be positive, got {}", p.max_range)));
    }
    if !(1..=1000).contains(&p.footprint_samples) {
        return Err(Error::invalid(format!("footprint_samples must be 1 to 1000, got {}", p.footprint_samples)));
    }
    if p.max_echoes == 0 {
        return Err(Error::invalid("max_echoes must be at least 1"));
    }
    if let Some(r) = p.target_radius {
        if !(r.is_finite() && r > 0.0) {
            return Err(Error::invalid(format!("target_radius must be positive, got {r}")));
        }
    }
    Ok(())
}

fn point_reflectance(cloud: &PointCloud, r: &Reflectance) -> Result<Vec<f64>> {
    let n = cloud.len();
    let v: Vec<f64> = match r {
        Reflectance::Constant(c) => vec![*c; n],
        Reflectance::Attribute(name) => {
            let a = cloud.attr(name).ok_or_else(|| Error::invalid(format!("the cloud has no attribute {name:?} for the reflectance")))?;
            (0..n).map(|i| a.get_f64(i)).collect()
        }
        Reflectance::ByClass(map, default) => match cloud.attr("classification") {
            Some(c) => (0..n).map(|i| {
                let k = c.get_f64(i);
                map.iter().find(|m| m.0 as f64 == k).map_or(*default, |m| m.1)
            }).collect(),
            None => vec![*default; n],
        },
    };
    if v.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
        return Err(Error::invalid("reflectance must be zero or more and finite"));
    }
    Ok(v)
}

/// Unit normals of the target points (zero for a sphere): from the
/// `normal_*` attributes, else from a local PCA where it is planar.
fn patch_normals(cloud: &PointCloud, targets: &[usize], oriented: bool) -> Vec<Point> {
    let mut out = vec![[0.0; 3]; cloud.len()];
    if !oriented {
        return out;
    }
    if let (Some(x), Some(y), Some(z)) = (cloud.attr("normal_x"), cloud.attr("normal_y"), cloud.attr("normal_z")) {
        for &i in targets {
            let n = [x.get_f64(i), y.get_f64(i), z.get_f64(i)];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l.is_finite() && l > 0.5 {
                out[i] = [n[0] / l, n[1] / l, n[2] / l];
            }
        }
        return out;
    }
    let pts: Vec<Point> = targets.iter().map(|&i| cloud.xyz[i]).collect();
    if pts.len() < 10 {
        return out;
    }
    let (n, planarity) = crate::coreg::geometry::estimate_normals(&pts, 10, None);
    for (k, &i) in targets.iter().enumerate() {
        if planarity[k] >= 0.5 {
            out[i] = n[k];
        }
    }
    out
}

/// One echo before noise: range, energy, footprint fraction, spread, point.
struct Echo {
    range: f64,
    energy: f64,
    fraction: f64,
    spread: f64,
    point: usize,
}

/// Scan `cloud` with a finite beam (see the module documentation).
pub fn scan_beam(cloud: &PointCloud, p: &BeamScan) -> Result<Shots> {
    check(p)?;
    let rho = point_reflectance(cloud, &p.reflectance)?;
    let mut stands_for_solid = vec![false; cloud.len()];
    for (k, s) in p.solids.iter().enumerate() {
        let ok = s.a.iter().chain(&s.b).all(|v| v.is_finite()) && s.r.is_finite() && s.r > 0.0 && s.point < cloud.len();
        if !ok {
            return Err(Error::invalid(format!("solid {k} needs finite ends, a positive radius and one of the cloud's points")));
        }
        stands_for_solid[s.point] = true;
    }
    let targets: Vec<usize> = (0..cloud.len()).filter(|&i| !stands_for_solid[i] && cloud.xyz[i].iter().all(|v| v.is_finite()) && rho[i] > 0.0).collect();
    let r = match p.target_radius {
        Some(r) => r,
        None => {
            let pts: Vec<Point> = targets.iter().map(|&i| cloud.xyz[i]).collect();
            let s = crate::leaves::median_spacing(&pts);
            if s > 0.0 { 1.75 * s } else { 0.01 }
        }
    };
    let normals = patch_normals(cloud, &targets, p.oriented);
    let grid = Grid::new(&cloud.xyz, &targets, |_| r)?;
    let r2 = r * r;
    // the solids, by bounding sphere: centre at the middle, radius half the length and the radius
    let mid: Vec<Point> = p.solids.iter().map(|s| std::array::from_fn(|k| (s.a[k] + s.b[k]) / 2.0)).collect();
    let bound = |i: usize| {
        let s = &p.solids[i];
        let h = ((0..3).map(|k| (s.b[k] - s.a[k]).powi(2)).sum::<f64>()).sqrt() / 2.0;
        (h * h + s.r * s.r).sqrt()
    };
    // solids larger than BIG (ground slabs, walls) are tested on every ray, so they don't coarsen the grid of the rest
    const BIG: f64 = 1.0;
    let (big, small): (Vec<usize>, Vec<usize>) = (0..p.solids.len()).partition(|&i| bound(i) > BIG);
    let solid_grid = Grid::new(&mid, &small, bound)?;
    let res = p.resolution_deg;
    let n_zen = ((p.max_zenith_deg - p.min_zenith_deg) / res).round().max(1.0) as usize;
    let ia0 = (p.min_azimuth_deg / res).floor() as usize;
    let n_az = ((p.max_azimuth_deg / res).ceil() as usize).min((360.0 / res).round().max(1.0) as usize).max(ia0 + 1) - ia0;
    let n_pulses = n_zen * n_az;
    limits::check_cells(n_pulses as u128, 64 + 8 * p.max_echoes as u64, &format!("a scan of {n_pulses} pulses"), "a coarser resolution_deg")?;
    let offsets = footprint_offsets(p.footprint_samples);
    let share = 1.0 / offsets.len() as f64;
    let half_div = p.divergence_mrad * 1e-3 / 2.0;
    let half_exit = p.exit_diameter / 2.0;
    let o = p.origin;
    let mut rng = Generator::new(p.seed);
    let mut direction = Vec::with_capacity(n_pulses);
    let mut time = Vec::with_capacity(n_pulses);
    let mut per_pulse: Vec<Vec<(Echo, f64)>> = Vec::with_capacity(n_pulses);
    const BLOCK: usize = 1 << 15;
    let mut j0 = 0;
    while j0 < n_pulses {
        let j1 = (j0 + BLOCK).min(n_pulses);
        let noise = if p.range_noise > 0.0 || p.range_noise_slope > 0.0 { rng.normal_n(0.0, 1.0, (j1 - j0) * p.max_echoes) } else { vec![0.0; (j1 - j0) * p.max_echoes] };
        let block: Vec<(Point, f64, Vec<(Echo, f64)>)> = (j0..j1).into_par_iter().map(|j| {
            // The pulse's place on the scan pattern: one line per azimuth
            // step, its slots up the zenith, aimed at the middle of its cell.
            let (ia, iz) = (ia0 + j / n_zen, j % n_zen);
            let zdeg = p.min_zenith_deg + (iz as f64 + 0.5) * res;
            let z = zdeg.to_radians();
            let a = ((ia as f64 + 0.5) * res).to_radians();
            let t = ia as f64 * p.line_period + (zdeg / res) * p.pulse_interval;
            let d = [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()];
            // Two directions across the beam, so the footprint can be sampled:
            // any vector not parallel to the beam gives the first by a cross
            // product, and the second follows from both.
            let h = if d[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let e1 = {
                let c = [d[1] * h[2] - d[2] * h[1], d[2] * h[0] - d[0] * h[2], d[0] * h[1] - d[1] * h[0]];
                let nn = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / nn, c[1] / nn, c[2] / nn]
            };
            let e2 = [d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]];
            // The beam is not a line: it leaves the scanner with a width and
            // spreads as it goes, so it is sampled by several sub-rays, each
            // offset across the exit aperture and tilted within the
            // divergence. Each returns the first surface it meets, if any.
            let mut hits: Vec<(f64, usize)> = offsets.iter().filter_map(|off| {
                let lat: Point = std::array::from_fn(|k| off[0] * e1[k] + off[1] * e2[k]);
                let os: Point = std::array::from_fn(|k| o[k] + half_exit * lat[k]);
                let ds: Point = std::array::from_fn(|k| d[k] + half_div * lat[k]);
                let nn = (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]).sqrt();
                let ds = [ds[0] / nn, ds[1] / nn, ds[2] / nn];
                let patch = grid.cast(&os, &ds, 0.1, p.max_range, |i| patch_hit(&cloud.xyz[i], &normals[i], r2, &os, &ds));
                let mut solid = solid_grid.cast(&os, &ds, 0.1, p.max_range, |i| solid_hit(&p.solids[i], &os, &ds)).map(|(t, i)| (t, p.solids[i].point));
                for &i in &big {
                    if let Some((t, _)) = solid_hit(&p.solids[i], &os, &ds).filter(|h| h.0 > 0.1 && h.0 < p.max_range) {
                        if solid.is_none_or(|s| t < s.0) {
                            solid = Some((t, p.solids[i].point));
                        }
                    }
                }
                match (patch, solid) {
                    (Some(x), Some(y)) => Some(if y.0 < x.0 { y } else { x }),
                    (x, y) => x.or(y),
                }
            }).collect();
            // Sub-rays that came back from nearly the same distance are one
            // echo: a real scanner cannot separate returns closer together
            // than `echo_separation`. Sorting by range makes those runs
            // adjacent, and the loop below walks one run at a time.
            hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut echoes: Vec<Echo> = Vec::new();
            let mut g0 = 0;
            while g0 < hits.len() {
                let mut g1 = g0 + 1;
                while g1 < hits.len() && hits[g1].0 - hits[g0].0 < p.echo_separation {
                    g1 += 1;
                }
                let group = &hits[g0..g1];
                let w: Vec<f64> = group.iter().map(|h| share * rho[h.1]).collect();
                let energy: f64 = w.iter().sum();
                let mut best = 0;
                for k in 1..group.len() {
                    if w[k] > w[best] {
                        best = k;
                    }
                }
                // A mixed pixel: where the beam straddles an edge, the scanner
                // reports one range between the two surfaces, weighted by how
                // much energy each returned. Without that, the echo is simply
                // the brightest sub-ray's own range.
                let range = if p.mixed_pixels && energy > 0.0 { group.iter().zip(&w).map(|(h, w)| h.0 * w).sum::<f64>() / energy } else { group[best].0 };
                if energy >= p.detection_threshold && energy > 0.0 {
                    echoes.push(Echo { range, energy, fraction: share * group.len() as f64, spread: group[group.len() - 1].0 - group[0].0, point: group[best].1 });
                }
                g0 = g1;
            }
            echoes.truncate(p.max_echoes);
            let nz = &noise[(j - j0) * p.max_echoes..];
            let mut out: Vec<(Echo, f64)> = echoes.into_iter().enumerate().map(|(k, e)| {
                let sigma = p.range_noise + p.range_noise_slope * e.range;
                let noisy = e.range + sigma * nz[k];
                (e, noisy)
            }).collect();
            out.sort_by(|a, b| a.1.total_cmp(&b.1));
            (d, t, out)
        }).collect();
        for (d, t, e) in block {
            direction.push(d);
            time.push(t);
            per_pulse.push(e);
        }
        j0 = j1;
    }
    let count: Vec<u32> = per_pulse.iter().map(|e| e.len() as u32).collect();
    let echo_start = packed_starts(&count);
    let flat: Vec<&(Echo, f64)> = per_pulse.iter().flatten().collect();
    let idx: Vec<usize> = flat.iter().map(|e| e.0.point).collect();
    let mut echo_attrs: std::collections::BTreeMap<String, Attr> = cloud.attrs.iter().map(|(k, v)| (k.clone(), v.take(&idx))).collect();
    echo_attrs.insert("reflectance".into(), Attr::F32(flat.iter().map(|e| (10.0 * e.0.energy.log10()) as f32).collect()));
    echo_attrs.insert("footprint_fraction".into(), Attr::F32(flat.iter().map(|e| e.0.fraction as f32).collect()));
    echo_attrs.insert("range_spread".into(), Attr::F32(flat.iter().map(|e| e.0.spread as f32).collect()));
    echo_attrs.insert("gps_time".into(), Attr::F64(count.iter().zip(&time).flat_map(|(&c, &t)| std::iter::repeat_n(t, c as usize)).collect()));
    Ok(Shots { origin: vec![o; n_pulses], direction, echo_start, echo_count: count, echo_range: flat.iter().map(|e| e.1).collect(), echo_attrs })
}

/// What a scanner's export keeps of a scan, and the systematic errors it
/// carries ([`export`]); lengths in metres.
#[derive(Debug, Clone)]
pub struct Export {
    /// Partial hits are deleted: an echo is kept with probability
    /// `1 / (1 + exp(-(footprint_fraction - min_fraction) / fraction_width))`
    /// (a hard cut at `min_fraction` when the width is 0), the stand-in for a
    /// filter on pulse-shape deviation.
    pub min_fraction: f64,
    pub fraction_width: f64,
    /// Echoes below this reflectance (dB) are deleted.
    pub min_reflectance: f64,
    /// Drop the pulses left without an echo, as an echoes-only export does.
    pub echoes_only: bool,
    /// The range law along the beam: every range grows by `range_a + range_b * range`.
    pub range_a: f64,
    pub range_b: f64,
    /// The scan's directions turned about the vertical (mrad).
    pub yaw_mrad: f64,
    /// A smooth random displacement field added to every pulse's echoes:
    /// standard deviation per axis and correlation length (a squared
    /// exponential), drawn from `seed`.
    pub offset_sd: f64,
    pub offset_length: f64,
    pub seed: u64,
}

impl Default for Export {
    /// Keep everything, add nothing.
    fn default() -> Self {
        Export { min_fraction: 0.0, fraction_width: 0.0, min_reflectance: f64::NEG_INFINITY, echoes_only: false, range_a: 0.0, range_b: 0.0, yaw_mrad: 0.0, offset_sd: 0.0, offset_length: 0.5, seed: 0 }
    }
}

/// A uniform number in `[0, 1)` from three integers (splitmix64), so that an
/// echo's fate doesn't depend on the order echoes are visited in.
fn unit_hash(a: u64, b: u64, c: u64) -> f64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_add(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// What an export of scan `s` ([`scan_beam`]'s output, which carries
/// `footprint_fraction` and `reflectance`) holds, under `p` (see [`Export`]).
///
/// The deletion acts per echo, then a pulse left with none is dropped if
/// `echoes_only`. The errors act on what is kept: the direction turns by
/// `yaw_mrad`; the first echo's point moves by the displacement field there
/// (random Fourier features: 64 cosines whose wave vectors are drawn from
/// the field's spectrum), which sets the pulse's new direction and adds the
/// same range to all its echoes; then every range grows by the range law.
pub fn export(s: &Shots, p: &Export) -> Result<Shots> {
    let fraction = s.echo_attrs.get("footprint_fraction").ok_or_else(|| Error::invalid("export needs the scan's footprint_fraction"))?;
    let refl = s.echo_attrs.get("reflectance").ok_or_else(|| Error::invalid("export needs the scan's reflectance"))?;
    if !(p.fraction_width >= 0.0 && p.offset_sd >= 0.0 && p.offset_length > 0.0) {
        return Err(Error::invalid("fraction_width and offset_sd must be zero or more, offset_length positive"));
    }
    let mut keep_echo = vec![false; s.echo_range.len()];
    for i in 0..s.n_shots() {
        for k in 0..s.echo_count[i] as usize {
            let e = s.echo_start[i] + k;
            let (f, r) = (fraction.get_f64(e), refl.get_f64(e));
            let pk = if p.fraction_width > 0.0 { 1.0 / (1.0 + (-(f - p.min_fraction) / p.fraction_width).exp()) } else { f64::from(u8::from(f >= p.min_fraction)) };
            keep_echo[e] = r >= p.min_reflectance && unit_hash(p.seed, i as u64, k as u64) < pk;
        }
    }
    const K: usize = 64;
    let mut rng = Generator::new(p.seed ^ 0x5EED_0FF5E7);
    let wave = rng.normal_n(0.0, 1.0 / p.offset_length, 3 * K);
    let amp = rng.normal_n(0.0, p.offset_sd * (2.0 / K as f64).sqrt(), 3 * K);
    let phase: Vec<f64> = (0..K).map(|_| rng.random() * 2.0 * PI).collect();
    let field = |x: &Point| -> Point {
        let mut d = [0.0; 3];
        if p.offset_sd == 0.0 {
            return d;
        }
        for k in 0..K {
            let c = (wave[3 * k] * x[0] + wave[3 * k + 1] * x[1] + wave[3 * k + 2] * x[2] + phase[k]).cos();
            for a in 0..3 {
                d[a] += amp[3 * k + a] * c;
            }
        }
        d
    };
    let (sy, cy) = (p.yaw_mrad * 1e-3).sin_cos();
    let mut out = Shots::default();
    let mut idx = Vec::new();
    for i in 0..s.n_shots() {
        let a = s.echo_start[i];
        let kept: Vec<usize> = (a..a + s.echo_count[i] as usize).filter(|&e| keep_echo[e]).collect();
        if kept.is_empty() && p.echoes_only {
            continue;
        }
        let (o, d0) = (s.origin[i], s.direction[i]);
        let mut d = [cy * d0[0] - sy * d0[1], sy * d0[0] + cy * d0[1], d0[2]];
        let mut shift = 0.0;
        if let Some(&e) = kept.first() {
            let r = s.echo_range[e];
            let dx = field(&std::array::from_fn(|k| o[k] + r * d[k]));
            let v: Point = std::array::from_fn(|k| r * d[k] + dx[k]);
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            d = [v[0] / len, v[1] / len, v[2] / len];
            shift = len - r;
        }
        out.origin.push(o);
        out.direction.push(d);
        out.echo_start.push(out.echo_range.len());
        out.echo_count.push(kept.len() as u32);
        for &e in &kept {
            let r = s.echo_range[e] + shift;
            out.echo_range.push(r + p.range_a + p.range_b * r);
            idx.push(e);
        }
    }
    out.echo_attrs = s.echo_attrs.iter().map(|(k, v)| (k.clone(), v.take(&idx))).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense square of points at `y`, facing the scanner at the origin.
    fn wall(y: f64, x0: f64, x1: f64, z0: f64, z1: f64, step: f64) -> Vec<Point> {
        let mut v = Vec::new();
        let mut x = x0;
        while x <= x1 {
            let mut z = z0;
            while z <= z1 {
                v.push([x, y, z]);
                z += step;
            }
            x += step;
        }
        v
    }

    fn params() -> BeamScan {
        let mut p = BeamScan::new([0.0, 0.0, 0.0]);
        p.oriented = false;
        p.resolution_deg = 0.2;
        p.min_zenith_deg = 80.0;
        p.max_zenith_deg = 100.0;
        p.target_radius = Some(0.004);
        p
    }

    #[test]
    fn a_plain_wall_is_at_its_range() {
        let cloud = PointCloud::new(wall(10.0, -1.0, 1.0, -1.0, 1.0, 0.005));
        let s = scan_beam(&cloud, &params()).unwrap();
        assert!(s.echo_range.len() > 50);
        for c in 0..s.n_shots() {
            for e in 0..s.echo_count[c] as usize {
                let r = s.echo_range[s.echo_start[c] + e];
                // Range times the y component of the direction is the wall's distance.
                assert!((r * s.direction[c][1] - 10.0).abs() < 0.01, "{r}");
            }
        }
    }

    #[test]
    fn range_noise_has_its_spread() {
        let cloud = PointCloud::new(wall(10.0, -2.0, 2.0, -1.5, 1.5, 0.01));
        let mut p = params();
        p.target_radius = Some(0.008);
        p.range_noise = 0.01;
        p.seed = 3;
        let s = scan_beam(&cloud, &p).unwrap();
        let err: Vec<f64> = (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).map(|c| s.echo_range[s.echo_start[c]] * s.direction[c][1] - 10.0).collect();
        let n = err.len() as f64;
        let m = err.iter().sum::<f64>() / n;
        let sd = (err.iter().map(|e| (e - m).powi(2)).sum::<f64>() / n).sqrt();
        assert!(n > 1000.0 && (sd - 0.01).abs() < 0.001 && m.abs() < 0.001, "{n} {m} {sd}");
    }

    #[test]
    fn a_wide_beam_mixes_an_edge() {
        // Foreground half-plane (x < 0) at 10 m, background at 10.3 m.
        let mut pts = wall(10.0, -3.0, 0.0, -1.0, 1.0, 0.005);
        pts.extend(wall(10.3, -3.0, 3.0, -1.0, 1.0, 0.005));
        let cloud = PointCloud::new(pts);
        let mut p = params();
        p.divergence_mrad = 3.0;
        p.footprint_samples = 19;
        let mixed = |s: &Shots| (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).filter(|&c| {
            let d = s.echo_range[s.echo_start[c]] * s.direction[c][1];
            d > 10.02 && d < 10.28
        }).count();
        let s = scan_beam(&cloud, &p).unwrap();
        assert!(mixed(&s) > 0);
        p.mixed_pixels = false;
        assert_eq!(mixed(&scan_beam(&cloud, &p).unwrap()), 0);
        // With a fine range resolution the two surfaces give two echoes instead.
        p.mixed_pixels = true;
        p.echo_separation = 0.1;
        let s = scan_beam(&cloud, &p).unwrap();
        assert_eq!(mixed(&s), 0);
        assert!(s.echo_count.contains(&2));
    }

    #[test]
    fn misses_are_kept_and_presets_parse() {
        let cloud = PointCloud::new(vec![[0.0, 5.0, 0.0]]);
        let mut p = params();
        p.target_radius = Some(0.01);
        let s = scan_beam(&cloud, &p).unwrap();
        assert_eq!(s.n_shots(), 100 * 1800);
        assert!(s.echo_count.iter().filter(|&&c| c > 0).count() <= 4);
        assert_eq!(scanner_preset("VZ-400i").unwrap().divergence_mrad, 0.35);
        assert!(scanner_preset("p20").is_err());
        p.range_noise = -1.0;
        assert!(scan_beam(&cloud, &p).is_err());
    }

    #[test]
    fn discs_keep_a_cylinder_its_width() {
        // A vertical cylinder of radius 0.2 m at 5 m, sampled with its normals.
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        for i in 0..400 {
            for j in 0..60 {
                let a = 2.0 * PI * i as f64 / 400.0;
                pts.push([0.2 * a.cos(), 5.0 + 0.2 * a.sin(), -0.3 + 0.01 * j as f64]);
                nrm.push([a.cos(), a.sin(), 0.0]);
            }
        }
        let mut cloud = PointCloud::new(pts);
        crate::synthetic::tree::insert_normals(&mut cloud, &nrm);
        let mut p = params();
        p.oriented = true;
        p.resolution_deg = 0.05;
        p.min_zenith_deg = 89.9;
        p.max_zenith_deg = 90.1;
        p.target_radius = Some(0.02);
        let width = |s: &Shots| {
            let xs: Vec<f64> = (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).map(|c| s.echo_range[s.echo_start[c]] * s.direction[c][0]).collect();
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        let w = width(&scan_beam(&cloud, &p).unwrap());
        assert!((w - 0.4).abs() < 0.01, "{w}");
        p.oriented = false;
        let w = width(&scan_beam(&cloud, &p).unwrap());
        assert!(w > 0.42, "{w}");
    }

    #[test]
    fn a_solid_cylinder_stops_every_sub_beam_on_its_near_face() {
        // a vertical solid stem of radius 1.5 cm at 5 m, with a wall behind it at 8 m; the stem's echoes take
        // the attributes of point 0, which is not a patch itself (it sits where the wall would show it)
        let r = 0.015;
        let mut pts = wall(8.0, -0.3, 0.3, -0.3, 0.3, 0.004);
        pts[0] = [0.0, 8.0, 0.0];
        let mut cloud = PointCloud::new(pts);
        let n = cloud.len();
        cloud.attrs.insert("classification".into(), Attr::U8((0..n).map(|i| if i == 0 { 5 } else { 2 }).collect()));
        let mut p = params();
        p.resolution_deg = 0.02;
        p.min_zenith_deg = 89.5;
        p.max_zenith_deg = 90.5;
        p.exit_diameter = 0.007;
        p.divergence_mrad = 0.35;
        p.footprint_samples = 19;
        p.solids = vec![Solid { a: [0.0, 5.0, -0.5], b: [0.0, 5.0, 0.5], r, point: 0 }];
        let s = scan_beam(&cloud, &p).unwrap();
        let class = match &s.echo_attrs["classification"] { Attr::U8(v) => v.clone(), _ => unreachable!() };
        let (mut stem, mut beyond) = (0, 0);
        for c in 0..s.n_shots() {
            let d = s.direction[c];
            let b = d[0] / (d[0] * d[0] + d[1] * d[1]).sqrt() * 5.0; // impact across the stem, at its range
            for e in s.echo_start[c]..s.echo_start[c] + s.echo_count[c] as usize {
                let y = s.echo_range[e] * d[1];
                if class[e] == 5 {
                    stem += 1;
                    // every echo of the stem lies on its near face, never inside or behind the axis
                    assert!(y < 5.0 && y > 5.0 - r - 1e-3, "y {y}");
                    assert!(b.abs() < r + 0.004, "b {b}");
                } else if b.abs() < r - 0.005 && (y - 8.0).abs() < 0.1 {
                    beyond += 1;
                }
            }
        }
        assert!(stem > 50, "stem echoes {stem}");
        assert_eq!(beyond, 0, "no sub-beam of a pulse whose footprint lies on the stem reaches the wall");
        p.solids[0].r = -1.0;
        assert!(scan_beam(&cloud, &p).is_err());
    }

    #[test]
    fn pulses_come_in_lines_with_their_times() {
        let cloud = PointCloud::new(wall(10.0, -1.0, 1.0, -1.0, 1.0, 0.01));
        let mut p = params();
        p.max_azimuth_deg = 2.0;
        let s = scan_beam(&cloud, &p).unwrap();
        assert_eq!(s.n_shots(), 100 * 10);
        let t = match &s.echo_attrs["gps_time"] { Attr::F64(v) => v.clone(), _ => unreachable!() };
        assert_eq!(t.len(), s.echo_range.len());
        assert!(t.len() > 50 && t.windows(2).all(|w| w[1] >= w[0]), "echoes in acquisition order");
    }

    #[test]
    fn the_export_deletes_partial_hits_and_shifts_ranges() {
        let mut pts = wall(10.0, -3.0, 0.0, -1.0, 1.0, 0.005);
        pts.extend(wall(10.3, -3.0, 3.0, -1.0, 1.0, 0.005));
        let cloud = PointCloud::new(pts);
        let mut p = params();
        p.divergence_mrad = 3.0;
        p.footprint_samples = 19;
        p.echo_separation = 0.1;
        let s = scan_beam(&cloud, &p).unwrap();
        let partial = |s: &Shots| match &s.echo_attrs["footprint_fraction"] { Attr::F32(v) => v.iter().filter(|&&f| f < 0.5).count(), _ => unreachable!() };
        assert!(partial(&s) > 0);
        let e = Export { min_fraction: 0.5, echoes_only: true, range_a: 0.002, range_b: 0.001, ..Export::default() };
        let x = export(&s, &e).unwrap();
        assert_eq!(partial(&x), 0);
        assert!(x.echo_count.iter().all(|&c| c > 0) && x.n_shots() < s.n_shots());
        // the wall at 10 m straight ahead now reads 2 mm + 1 mm/m x 10 m = 12 mm long
        let c = (0..x.n_shots()).filter(|&c| x.direction[c][1] > 0.99999).find(|&c| x.echo_range[x.echo_start[c]] < 10.2).unwrap();
        let r = x.echo_range[x.echo_start[c]] * x.direction[c][1];
        assert!((r - 10.012).abs() < 1e-3, "{r}");
        // a displacement field of 3 mm per axis moves echoes by about that much
        let e = Export { offset_sd: 0.003, offset_length: 0.5, seed: 4, ..Export::default() };
        let x = export(&s, &e).unwrap();
        let dy: Vec<f64> = (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).map(|c| x.echo_range[x.echo_start[c]] * x.direction[c][1] - s.echo_range[s.echo_start[c]] * s.direction[c][1]).collect();
        let sd = (dy.iter().map(|v| v * v).sum::<f64>() / dy.len() as f64).sqrt();
        assert!(sd > 0.0005 && sd < 0.009, "{sd}");
        assert!(export(&s, &Export { offset_length: 0.0, ..Export::default() }).is_err());
    }

    #[test]
    fn offsets_cover_the_disc() {
        let o = footprint_offsets(100);
        assert!(o.iter().all(|v| v[0].hypot(v[1]) <= 1.0));
        let m: f64 = o.iter().map(|v| v[0]).sum::<f64>() / 100.0;
        assert!(m.abs() < 0.02);
    }
}
