// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A scan's pulses indexed by direction ([`ScanIndex`]), and every pulse near
//! each cylinder with all its echoes ([`pairs`]): the raw table the joint
//! model's per-pulse likelihood reads.
//!
//! Speed comes from looking only where wood is: each scan is indexed once by
//! pulse direction, each cylinder visits only the pulses near its axis, and
//! cylinders run in parallel.

use std::path::Path;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::io::shots::{direction, Rays, ShotsFile};
use crate::transform::{add, dot, norm, scale, sub};
use crate::Point;

const CELL: f64 = 0.2 * std::f64::consts::PI / 180.0; // direction grid of the scan index
const NZ: usize = 900; // ceil(pi / CELL)
const NA: usize = 1800; // ceil(2 pi / CELL)

fn cell_of(zen: f64, az: f64) -> usize {
    let iz = ((zen / CELL) as usize).min(NZ - 1);
    let ia = ((az.rem_euclid(2.0 * std::f64::consts::PI) / CELL) as usize).min(NA - 1);
    iz * NA + ia
}

/// A pool of `threads` threads (0: one per core), built once per calling thread and size (a fit
/// calls the likelihood thousands of times; a pool per caller keeps
/// concurrent fits, such as trees fitted in parallel, from sharing one).
pub(crate) fn pool(threads: usize) -> Result<std::rc::Rc<rayon::ThreadPool>> {
    use std::{cell::RefCell, collections::HashMap, rc::Rc};
    thread_local! {
        static POOLS: RefCell<HashMap<usize, Rc<rayon::ThreadPool>>> = RefCell::new(HashMap::new());
    }
    POOLS.with(|m| {
        if let Some(p) = m.borrow().get(&threads) {
            return Ok(p.clone());
        }
        let p = Rc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| Error::invalid(format!("thread pool: {e}")))?);
        m.borrow_mut().insert(threads, p.clone());
        Ok(p)
    })
}

/// One scan's pulses, indexed by direction: what [`pairs`], the joint table and `support_tree` read. About
/// 20 bytes per pulse (4 of them its row in the file) and 4 per echo (8 with
/// reflectance and deviation), plus a 6.5 MB cell table.
#[derive(Debug, Clone)]
pub struct ScanIndex {
    origin: Point,
    /// Pulses in direction-cell order: cell `c` holds `cell_start[c]..cell_start[c + 1]`.
    dir: Vec<[f32; 3]>,
    /// Pulse `p`'s echoes are `range[echo_start[p]..echo_start[p + 1]]`, nearest first.
    echo_start: Vec<u32>,
    range: Vec<f32>,
    /// Each echo's reflectance (0.01 dB) and deviation, in `range` order; empty where the store has none.
    reflectance: Vec<i16>,
    deviation: Vec<u16>,
    /// Each pulse's row in its Shots file (the tile sidecars' `pulse.u32`), in pulse order.
    src: Vec<u32>,
    cell_start: Vec<u32>,
}

/// The attributes a [`ScanIndex`] keeps, where the store has them: the
/// echoes' reflectance and deviation.
const CHARACTER: [&str; 2] = ["reflectance", "deviation"];

impl ScanIndex {
    /// Index a single-position Shots file, reading only angles and ranges,
    /// row groups in parallel on `threads` threads (0: one per core).
    pub fn from_file(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let file = ShotsFile::open(path)?;
        let [origin] = *file.scans() else {
            return Err(Error::invalid("ScanIndex: the shots file must hold exactly one scan position"));
        };
        let names: Vec<&str> = file.attr_names().filter(|a| CHARACTER.contains(a)).collect();
        pool(threads)?.install(|| {
            // a handle per row group: handles share no file position
            let groups: Vec<Rays> = (0..file.n_groups()).into_par_iter().map(|g| file.reopen()?.read_rays(g, &names)).collect::<Result<_>>()?;
            let mut rays = Rays { attrs: vec![Vec::new(); names.len()], ..Default::default() };
            for g in groups {
                rays.zenith.extend(g.zenith);
                rays.azimuth.extend(g.azimuth);
                rays.echo_count.extend(g.echo_count);
                rays.range.extend(g.range);
                for (all, a) in rays.attrs.iter_mut().zip(g.attrs) {
                    all.extend(a);
                }
            }
            let mut attr = |name| names.iter().position(|&a| a == name).map(|i| std::mem::take(&mut rays.attrs[i])).unwrap_or_default();
            let (refl, dev) = (attr("reflectance"), attr("deviation"));
            Self::build(origin, rays, &refl, &dev)
        })
    }

    /// Put the pulses in direction-cell order (a stable counting sort), with
    /// the echoes' reflectance (dB) and deviation, if any (else empty).
    fn build(origin: Point, rays: Rays, refl: &[f32], dev: &[f32]) -> Result<Self> {
        let n = rays.zenith.len();
        if n >= u32::MAX as usize || rays.range.len() >= u32::MAX as usize {
            return Err(Error::invalid("ScanIndex: too many pulses or echoes for one scan"));
        }
        let cell: Vec<u32> = rays.zenith.par_iter().zip(&rays.azimuth).map(|(&z, &a)| cell_of(z as f64, a as f64) as u32).collect();
        let mut cell_start = vec![0u32; NZ * NA + 1];
        for &c in &cell {
            cell_start[c as usize + 1] += 1;
        }
        for i in 0..NZ * NA {
            cell_start[i + 1] += cell_start[i];
        }
        let mut next = cell_start.clone();
        let mut src = vec![0u32; n]; // the pulse of `rays` at each place
        for (p, &c) in cell.iter().enumerate() {
            src[next[c as usize] as usize] = p as u32;
            next[c as usize] += 1;
        }
        drop((cell, next));
        let mut from = Vec::with_capacity(n + 1); // where each pulse's echoes start in `rays.range`
        from.push(0u32);
        for &k in &rays.echo_count {
            from.push(from[from.len() - 1] + k);
        }
        let mut echo_start = Vec::with_capacity(n + 1);
        echo_start.push(0u32);
        for &p in &src {
            echo_start.push(echo_start[echo_start.len() - 1] + rays.echo_count[p as usize]);
        }
        let dir = src.par_iter().map(|&p| direction(rays.zenith[p as usize] as f64, rays.azimuth[p as usize] as f64).map(|x| x as f32)).collect();
        let echoes = |v: &[f32]| -> Vec<f32> { if v.is_empty() { return Vec::new() } src.par_iter().flat_map_iter(|&p| &v[from[p as usize] as usize..from[p as usize + 1] as usize]).copied().collect() };
        let range = echoes(&rays.range);
        // `as` saturates: +-327 dB and deviations above 65535 are out of any scanner's range anyway
        let reflectance = echoes(refl).into_iter().map(|x| (x * 100.0).round() as i16).collect();
        let deviation = echoes(dev).into_iter().map(|x| x.round() as u16).collect();
        Ok(ScanIndex { origin, dir, echo_start, range, reflectance, deviation, src, cell_start })
    }

    /// The reflectance (dB) and deviation of echo `k` of `pulse`, NaN where
    /// the store has none.
    pub fn character(&self, pulse: u32, k: u32) -> (f64, f64) {
        let e = self.echo_start[pulse as usize] as usize + k as usize;
        (self.reflectance.get(e).map_or(f64::NAN, |&x| x as f64 / 100.0), self.deviation.get(e).map_or(f64::NAN, |&x| x as f64))
    }

    /// `pulse`'s row in its Shots file: with an echo's place in the pulse, the key the tile sidecars carry
    /// (`pulse.u32`, `echo.u8`; 3dtrees research 94, win 1).
    pub fn file_row(&self, pulse: u32) -> u32 {
        self.src[pulse as usize]
    }

    /// `pulse`'s echo ranges (m), nearest first.
    pub fn ranges(&self, pulse: u32) -> &[f32] {
        &self.range[self.echo_start[pulse as usize] as usize..self.echo_start[pulse as usize + 1] as usize]
    }

    /// `pulse`'s unit direction.
    pub fn direction(&self, pulse: u32) -> Point {
        self.dir[pulse as usize].map(|x| x as f64)
    }

    /// The scanner position.
    pub fn origin(&self) -> Point {
        self.origin
    }

    pub fn n_pulses(&self) -> usize {
        self.dir.len()
    }

    /// The bytes the index holds.
    pub fn nbytes(&self) -> usize {
        size_of_val(&self.dir[..]) + size_of_val(&self.echo_start[..]) + size_of_val(&self.range[..]) + size_of_val(&self.reflectance[..]) + size_of_val(&self.deviation[..])  + size_of_val(&self.src[..]) + size_of_val(&self.cell_start[..])
    }

    /// Calls `f(pulse, direction, echo ranges)` for every pulse whose line
    /// passes within `window` of the segment `a -> b`, and for others near it
    /// (the caller tests the geometry). `pulse` names the pulse in this
    /// index; its echoes are nearest first.
    pub fn pulses_near(&self, a: &Point, b: &Point, window: f64, mut f: impl FnMut(u32, Point, &[f32])) {
        let (z0, nz, a0, na) = rectangle(&sub(a, &self.origin), &sub(b, &self.origin), window);
        for iz in z0..z0 + nz {
            for j in 0..na {
                let cell = iz * NA + (a0 + j as i64).rem_euclid(NA as i64) as usize;
                for p in self.cell_start[cell] as usize..self.cell_start[cell + 1] as usize {
                    f(p as u32, self.dir[p].map(|x| x as f64), &self.range[self.echo_start[p] as usize..self.echo_start[p + 1] as usize]);
                }
            }
        }
    }
}

/// Direction cells (zenith row start and count, azimuth column start and
/// count) that a segment `pa -> pb` (relative to the scanner) covers, padded
/// by the angle `window` subtends at its nearest point, and by one cell. A
/// pulse whose line passes within `window` of the segment points into it.
fn rectangle(pa: &Point, pb: &Point, window: f64) -> (usize, usize, i64, usize) {
    use std::f64::consts::PI;
    let ab = sub(pb, pa);
    let t = (-dot(pa, &ab) / dot(&ab, &ab).max(1e-18)).clamp(0.0, 1.0);
    let r_min = norm(&add(pa, &scale(&ab, t)));
    let alpha = (window / r_min.max(1e-9)).min(1.0).asin() + CELL;
    let zen_az = |p: &Point| ((p[2] / norm(p).max(1e-9)).clamp(-1.0, 1.0).acos(), p[0].atan2(p[1]));
    let ((za, aa), (zb, ab_)) = (zen_az(pa), zen_az(pb));
    let (mut z_lo, mut z_hi) = (za.min(zb), za.max(zb));
    // the zenith along a segment turns at most once (a segment overhead dips towards the zenith), where
    // d/dt (z / |p|) = 0, which is linear in t
    let den = ab[2] * dot(pa, &ab) - pa[2] * dot(&ab, &ab);
    let t_turn = (pa[2] * dot(pa, &ab) - ab[2] * dot(pa, pa)) / den;
    if t_turn > 0.0 && t_turn < 1.0 {
        let z = zen_az(&add(pa, &scale(&ab, t_turn))).0;
        (z_lo, z_hi) = (z_lo.min(z), z_hi.max(z));
    }
    let (z_lo, z_hi) = (z_lo - alpha, z_hi + alpha);
    let turn = (ab_ - aa + PI).rem_euclid(2.0 * PI) - PI; // the short way round from a to b
    let half = alpha / z_lo.max(0.0).sin().min(z_hi.min(PI).sin()).max(1e-9);
    // a segment passing over or under the scanner (within its window in plan) sweeps every azimuth
    let dxy = [ab[0], ab[1]];
    let s = (-(pa[0] * dxy[0] + pa[1] * dxy[1]) / (dxy[0] * dxy[0] + dxy[1] * dxy[1]).max(1e-18)).clamp(0.0, 1.0);
    let plan = ((pa[0] + s * dxy[0]).powi(2) + (pa[1] + s * dxy[1]).powi(2)).sqrt();
    let z0 = (z_lo.max(0.0) / CELL) as usize;
    let nz = ((z_hi / CELL) as usize).min(NZ - 1) + 1 - z0.min(NZ - 1);
    if half >= PI / 2.0 || plan <= window || window >= r_min {
        return (z0.min(NZ - 1), nz, 0, NA);
    }
    let a0 = ((aa + turn.min(0.0) - half) / CELL).floor() as i64;
    let a1 = ((aa + turn.max(0.0) + half) / CELL).floor() as i64;
    (z0.min(NZ - 1), nz, a0, ((a1 - a0 + 1) as usize).min(NA))
}

/// Closest approach of a pulse's line (unit direction `d`) to a cylinder's
/// axis line (unit `u`), given `w0 = origin - a` and `dd = w0 . u`: the range
/// along the pulse `t`, the position along the axis from `a` `s`, and
/// `1 - (d . u)^2` (the squared sine of their angle, floored). The distance
/// between the lines is the caller's: whether `s` is clamped to the segment
/// first is its choice.
pub(crate) fn closest(d: &Point, u: &Point, w0: &Point, dd: f64) -> (f64, f64, f64) {
    let (bb, ee) = (dot(d, u), dot(d, w0));
    let den = (1.0 - bb * bb).max(1e-12);
    ((bb * dd - ee) / den, (dd - bb * ee) / den, den)
}

/// Every pulse near each cylinder, with all its echoes: the raw table a
/// per-pulse likelihood reads.
///
/// Cylinder `c`'s rows are `cyl_start[c]..cyl_start[c + 1]`; row `i`'s echoes
/// (all of its pulse's, nearest first) are `echo_start[i]..echo_start[i + 1]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pairs {
    pub cyl_start: Vec<u64>,
    /// The scan (its index in the `scans` given) and pulse (in that scan's index).
    pub scan: Vec<u32>,
    pub pulse: Vec<u32>,
    /// The pulse's unit direction.
    pub dir: Vec<[f32; 3]>,
    /// Closest approach of the pulse's line to the axis line: its range `t`
    /// along the pulse, its position `s` along the axis from `a` (m), and the
    /// distance `d` between the lines.
    pub t: Vec<f32>,
    pub s: Vec<f32>,
    pub d: Vec<f32>,
    pub echo_start: Vec<u64>,
    pub range: Vec<f32>,
    /// Each echo's reflectance (dB) and deviation, NaN where the store has none.
    pub reflectance: Vec<f32>,
    pub deviation: Vec<f32>,
}

/// The [`Pairs`] of cylinders `a[i] -> b[i]`, each with its own tube of
/// radius `window[i]` around the axis: each pulse whose line comes closest to
/// the axis within the segment (ends included, so a pulse exactly at a shared
/// end belongs to both), inside the tube,
/// ahead of the scanner, and that reached the tube (its last echo is at or
/// beyond where its line enters it, or it missed). A pulse that stopped short
/// of the tube says nothing about what is in it and is left out. Runs on
/// `threads` threads (0: one per core).
pub fn pairs(scans: &[&ScanIndex], a: &[Point], b: &[Point], window: &[f64], threads: usize) -> Result<Pairs> {
    let n = a.len();
    if b.len() != n || window.len() != n {
        return Err(Error::invalid("pairs: a, b and window must have one row per cylinder"));
    }
    if scans.len() >= u32::MAX as usize {
        return Err(Error::invalid("pairs: too many scans"));
    }
    if !window.iter().all(|w| w.is_finite() && *w > 0.0) {
        return Err(Error::invalid("pairs: every window must be positive and finite"));
    }
    struct Row {
        scan: u32,
        pulse: u32,
        dir: [f32; 3],
        t: f32,
        s: f32,
        d: f32,
        echoes: Vec<f32>,
    }
    pool(threads)?.install(|| {
        let found: Vec<Vec<Row>> = (0..n).into_par_iter().map(|c| {
            let mut rows = Vec::new();
            let len = norm(&sub(&b[c], &a[c]));
            if !(len > 0.0) {
                return rows;
            }
            let u = sub(&b[c], &a[c]).map(|x| x / len);
            let window = window[c];
            for (si, sc) in scans.iter().enumerate() {
                let w0 = sub(&sc.origin(), &a[c]);
                let dd = dot(&w0, &u);
                sc.pulses_near(&a[c], &b[c], window, |pulse, d, echoes| {
                    let (t_ray, s_ax, den) = closest(&d, &u, &w0, dd);
                    if !(t_ray > 0.0 && s_ax >= 0.0 && s_ax <= len) {
                        return;
                    }
                    let dist = norm(&sub(&add(&w0, &scale(&d, t_ray)), &scale(&u, s_ax)));
                    if !(dist < window) {
                        return;
                    }
                    let entry = t_ray - (window * window - dist * dist).sqrt() / den.sqrt();
                    let last = echoes.last().map_or(f64::INFINITY, |&r| r as f64);
                    if last < entry {
                        return;
                    }
                    rows.push(Row { scan: si as u32, pulse, dir: d.map(|x| x as f32), t: t_ray as f32, s: s_ax as f32, d: dist as f32, echoes: echoes.to_vec() });
                });
            }
            rows
        }).collect();
        let mut out = Pairs { cyl_start: vec![0], echo_start: vec![0], ..Default::default() };
        for rows in found {
            for r in rows {
                let sc = scans[r.scan as usize];
                out.scan.push(r.scan);
                out.pulse.push(r.pulse);
                out.dir.push(r.dir);
                out.t.push(r.t);
                out.s.push(r.s);
                out.d.push(r.d);
                for (k, &range) in r.echoes.iter().enumerate() {
                    let (refl, dev) = sc.character(r.pulse, k as u32);
                    out.range.push(range);
                    out.reflectance.push(refl as f32);
                    out.deviation.push(dev as f32);
                }
                out.echo_start.push(out.range.len() as u64);
            }
            out.cyl_start.push(out.scan.len() as u64);
        }
        Ok(out)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::io::shots::{write_shots, ShotsWriteOptions};
    use crate::shots::Shots;

    /// Pulses on a fine grid from the origin towards a vertical cylinder of
    /// radius `r` at distance 5 m, with a wall behind it. With `misses` the
    /// pulses over and under the wall are kept, without echoes; otherwise
    /// the wall's top and bottom are lost too.
    fn shots(r: f64, misses: bool) -> Shots {
        screened(r, misses, None)
    }

    /// [`shots`], with a screen at y = 4 stopping every pulse whose line crosses y = 5 within `screen` (x, m).
    fn screened(r: f64, misses: bool, screen: Option<(f64, f64)>) -> Shots {
        let (cx, cy) = (0.0, 5.0);
        let mut shots = Shots::default();
        let step = 0.0005f64; // rad
        let mut az = -0.03f64;
        while az <= 0.03 {
            let mut el = -0.05f64;
            while el <= 0.05 {
                let zen = std::f64::consts::FRAC_PI_2 - el;
                let d = [zen.sin() * az.sin(), zen.sin() * az.cos(), zen.cos()];
                // ray-circle intersection in plan, then the wall at y = 8
                let (dx, dy) = (d[0], d[1]);
                let h = (dx * dx + dy * dy).sqrt();
                let (ux, uy) = (dx / h, dy / h);
                let tca = ux * cx + uy * cy;
                let d2 = cx * cx + cy * cy - tca * tca;
                let hit = d2 <= r * r && (-1.0..=1.0).contains(&(d[2] * (tca / h)));
                shots.echo_start.push(shots.echo_range.len());
                if screen.is_some_and(|(x0, x1)| (x0..=x1).contains(&(5.0 * dx / dy))) && d[2].abs() * 4.0 / dy <= 1.0 {
                    shots.echo_range.push(4.0 / dy);
                    shots.echo_count.push(1);
                } else if hit {
                    shots.echo_range.push((tca - (r * r - d2).sqrt()) / h);
                    shots.echo_count.push(1);
                } else if el.abs() < 0.04 || (misses && el.abs() < 0.045) {
                    shots.echo_range.push(8.0 / dy);
                    shots.echo_count.push(1);
                } else {
                    shots.echo_count.push(0);
                }
                shots.origin.push([0.0; 3]);
                shots.direction.push(d);
                el += step;
            }
            az += step;
        }
        if misses { shots } else { shots.subset(&(0..shots.n_shots()).map(|p| shots.echo_count[p] > 0).collect::<Vec<_>>()) }
    }

    /// The index of `shots` written to a temporary file, in row groups of `row_group_size`.
    pub(crate) fn index(shots: &Shots, row_group_size: usize) -> ScanIndex {
        static FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let k = FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sylva_scan_index_{}_{k}.parquet", std::process::id()));
        write_shots(shots, &path, &ShotsWriteOptions { row_group_size, ..Default::default() }).unwrap();
        let index = ScanIndex::from_file(&path, 3).unwrap();
        std::fs::remove_file(&path).unwrap();
        index
    }

    pub(crate) fn scene(r: f64, misses: bool) -> (ScanIndex, Point, Point) {
        (index(&shots(r, misses), ShotsWriteOptions::default().row_group_size), [0.0, 5.0, -1.0], [0.0, 5.0, 1.0])
    }

    /// [`scene`] (with misses) behind a screen at y = 4 that stops the pulses crossing y = 5 within `screen` (x, m).
    pub(crate) fn scene_screened(r: f64, screen: (f64, f64)) -> (ScanIndex, Point, Point) {
        (index(&screened(r, true, Some(screen)), ShotsWriteOptions::default().row_group_size), [0.0, 5.0, -1.0], [0.0, 5.0, 1.0])
    }

    #[test]
    fn each_pulse_knows_its_row_in_the_file() {
        // the index reorders the pulses by direction; each one's file row is the row it was read from (its
        // direction there), once each
        let sh = shots(0.05, true);
        let s = index(&sh, ShotsWriteOptions::default().row_group_size);
        let n = s.n_pulses();
        assert_eq!(n, sh.direction.len());
        let mut seen = vec![false; n];
        let mut moved = 0;
        for k in 0..n as u32 {
            let row = s.file_row(k) as usize;
            assert!(!seen[row], "row {row} twice");
            seen[row] = true;
            let (d, e) = (s.direction(k), sh.direction[row]);
            assert!((0..3).all(|i| (d[i] - e[i]).abs() < 1e-6), "pulse {k}: {d:?} against row {row}'s {e:?}");
            moved += usize::from(row != k as usize);
        }
        assert!(moved > n / 10, "the index's order is the file's: the check would pass with no mapping");
    }

    #[test]
    fn pairs_hold_every_pulse_that_reached_the_disc_with_its_echoes() {
        let r = 0.05;
        let (s, a, b) = scene(r, true);
        let w = 0.15;
        let p = pairs(&[&s], &[a, a], &[b, a], &[w, w], 2).unwrap();
        assert_eq!(p.cyl_start, vec![0, p.scan.len() as u64, p.scan.len() as u64], "no axis, no pairs");
        let n = p.scan.len();
        assert!(n > 200 && p.echo_start.len() == n + 1);
        let (mut hits, mut passes) = (0, 0);
        for i in 0..n {
            assert!((p.d[i] as f64) < w && p.s[i] >= 0.0 && p.s[i] <= 2.0);
            let echoes = &p.range[p.echo_start[i] as usize..p.echo_start[i + 1] as usize];
            // the scene's echo is on the stem (near t) or on the wall behind; a miss went on
            if echoes.first().is_some_and(|&x| (x - p.t[i]).abs() < 0.06) {
                hits += 1;
                assert!((p.d[i] as f64) < r + 1e-3, "a hit's line passes inside the stem");
            } else {
                passes += 1;
                assert!((p.d[i] as f64) > r - 1e-3, "a pass's line misses the stem");
            }
            assert!(p.reflectance[p.echo_start[i] as usize..p.echo_start[i + 1] as usize].iter().all(|x| x.is_nan()));
        }
        assert!(hits > 100 && passes > 100, "hits {hits} passes {passes}");
        assert!(pairs(&[&s], &[a], &[b], &[0.0], 1).is_err());
        assert!(pairs(&[&s], &[a], &[b], &[w, w], 1).is_err());
        // each cylinder's own tube: the narrower one holds a subset, all within it
        let q = pairs(&[&s], &[a, a], &[b, b], &[w, 0.08], 2).unwrap();
        let (n0, n1) = ((q.cyl_start[1] - q.cyl_start[0]) as usize, (q.cyl_start[2] - q.cyl_start[1]) as usize);
        assert!(n1 < n0 && q.d[n0..].iter().all(|&d| d < 0.08), "{n0} {n1}");
    }

    #[test]
    fn row_groups_read_in_parallel_index_as_one_and_misses_are_pulses_without_echoes() {
        let shots = shots(0.05, true);
        let (one, many) = (index(&shots, shots.n_shots()), index(&shots, 1000));
        assert_eq!((&one.dir, &one.echo_start, &one.range), (&many.dir, &many.echo_start, &many.range));
        assert!(one.n_pulses() == shots.n_shots() && one.origin() == [0.0; 3]);
    }

    #[test]
    fn pulses_near_finds_every_pulse_within_the_window() {
        // pulses in random directions, and segments anywhere around the scanner: overhead, below, across it
        let mut seed = 12345u64;
        let mut uniform = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut shots = Shots::default();
        for p in 0..200_000 {
            let (z, a) = ((1.0 - 2.0 * uniform()).acos(), 2.0 * std::f64::consts::PI * uniform());
            shots.direction.push(direction(z, a));
            shots.origin.push([0.0; 3]);
            shots.echo_start.push(p);
            shots.echo_count.push(1);
            shots.echo_range.push(30.0);
        }
        let s = index(&shots, 100_000);
        let dirs: Vec<Point> = (0..s.n_pulses()).map(|p| s.dir[p].map(|x| x as f64)).collect();
        for _ in 0..300 {
            let a = [(uniform() - 0.5) * 12.0, (uniform() - 0.5) * 12.0, (uniform() - 0.3) * 10.0];
            let l = uniform() * 2.0;
            let u = direction((1.0 - 2.0 * uniform()).acos(), 2.0 * std::f64::consts::PI * uniform());
            let b = add(&a, &scale(&u, l));
            let window = 0.02 + uniform() * 0.3;
            let mut seen = vec![false; dirs.len()];
            s.pulses_near(&a, &b, window, |p, _, _| seen[p as usize] = true);
            for (p, d) in dirs.iter().enumerate() {
                // the ray's distance from a point along the segment, convex in it
                let gap = |s: f64| {
                    let q = add(&a, &scale(&u, s));
                    norm(&sub(&q, &scale(d, dot(&q, d).max(0.0))))
                };
                if gap(l / 2.0) > window + l / 2.0 {
                    continue;
                }
                let (mut lo, mut hi) = (0.0, l);
                for _ in 0..60 {
                    let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
                    if gap(m1) < gap(m2) { hi = m2 } else { lo = m1 }
                }
                assert!(gap(lo) >= window || seen[p], "missed pulse {p} for segment {a:?} -> {b:?}, window {window}");
            }
        }
    }
}
