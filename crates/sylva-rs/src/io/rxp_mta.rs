// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! `.rxp` reading the way RiSCAN PRO converts a scan: every laser shot from
//! RiVLib's C++ point-cloud class (through the `sylva_rivlib` shim, built from
//! `crates/sylva-rs/rivlib-shim/`), multiple-time-around resolution with
//! [`super::mta`], then the same deterministic filters rimta5 applies.
//!
//! **Work in progress**: "the way RiSCAN PRO converts a scan" means the same
//! steps in the same order, reimplemented, not the same results to the last
//! point. The figures below say how close it gets; where the exact RiSCAN
//! point set is what matters, import there and read the `.rdbx`.
//!
//! What this reproduces, measured against rimta5 6.9.6 on a VZ-600i scan
//! (ScanPos001 of the Pilliga project): 97-99% identical point set (99% on
//! lines with little far canopy, 97% where zone-2 returns are 6% of the
//! points), coordinates within 0.25 mm (the rdbx rounding) for the common
//! points, zone choice 98.3-99.6%. What it does not: the dynamic window analysis
//! and the near-range window correction (only applied when that analysis
//! succeeds, which it did not on the reference scans), and rimta5's exact
//! rule for the last fraction of a percent of uncertain echoes.
//!
//! Pipeline per scan line, streamed with a window of lines around it:
//!
//! 1. echoes are attached to the pulse whose time equals theirs (late echoes
//!    arrive after the next shot) and carry RiVLib's range under each zone;
//! 2. [`mta::stage1`], [`mta::stage2`] and [`mta::stage3`] pick the zone, each
//!    pass seeing the previous one's results for the neighbouring lines;
//! 3. the echo is placed on the pulse `zone - 1` shots earlier at that range,
//!    along that pulse's beam;
//! 4. filters: amplitude 0, the dynamic-trigger rule (amplitude below 2.5 dB
//!    within 18 m), the range gate, the notch window on zone-1 echoes
//!    ("notch after MTA"), MTA-uncertain unless kept on request (then in zone 1, on
//!    the pulse it was recorded after; only a confident call moves an echo).
//!
//! With `max_zone == 1` there is nothing to resolve: steps 2 and 3 and the
//! learned model are bypassed, every echo stays on the shot it was recorded
//! after at its zone-1 range, and there are no `mta_zone` or
//! `mta_uncertain_point` columns. (The model was trained on a VZ-600i; at one
//! zone it can move nothing and its uncertainty flag means nothing.)
//!
//! **Every shot is a pulse.** The stream records every shot in the field of
//! view, echo or not ("on_shot() will be invoked even if no echoes are
//! available"), so the output is [`Shots`] with one pulse per shot in the
//! order the scanner fired them, from the scanner's origin (0, 0, 0). The
//! filters drop echoes, never pulses: a shot whose echoes were all dropped,
//! or that recorded none, is a pulse with `echo_count == 0`. A pulse's echoes
//! lie at `o + r d` along its beam (origin `o`, about 0.1 m out on the mirror,
//! and direction `d`), kept as ranges from the scanner's origin, nearest
//! first, with the pulse's direction that of its farthest echo; a pulse
//! without echoes takes `d`. Facts of the shot are pulse attributes:
//! `gps_time` (s, scanner clock), `scan_line_index` (RiSCAN's convention:
//! -1 is the first line), `shot_index_line` (0-based), `mirror_facet` and
//! `target_count`, the targets RiVLib attached to the shot in the stream
//! before any filter (window echo and pseudo echoes included, at its zone-1
//! attribution). So a shot that recorded targets but kept none is told from
//! one that recorded nothing; nothing is stored as a flag. Echo attributes:
//! `amplitude`, `reflectance`, `deviation`, `echo_type`, and with more than
//! one zone `mta_zone` (and `mta_uncertain_point` with `keep_uncertain`).

use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use super::mta::{self, Decision, Grid, LineEchoes, MtaParams, MAXZ};
use super::mta_model::{EchoInfo, MtaModel};
use super::riegl::{search_dir, search_roots, RxpOptions};
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::shots::ops::packed_starts;
use crate::transform::{norm, normalize};
use crate::{Point, Shots};

/// How [`super::riegl::read_rxp`] treats multiple-time-around ambiguity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MtaMode {
    /// Use the shim when it can be found, otherwise fall back to plain
    /// `libscanifc` reading (zone 1, no resolution).
    #[default]
    Auto,
    /// Plain `libscanifc` reading.
    Off,
    /// Require the shim; fail if it is missing.
    On,
}

/// Which rule turns the per-zone evidence into a decision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MtaDecision {
    /// The built-in learned model (trained on rimta5's output of a VZ-600i scan).
    #[default]
    Default,
    /// The hand-written three-pass thresholds only.
    Rule,
    /// A model JSON produced by `train_mta.py`.
    Model(PathBuf),
}

/// Options of the MTA path beyond [`MtaParams`].
#[derive(Clone, Debug)]
pub struct MtaOptions {
    /// Whether MTA resolution through the shim is attempted, required or skipped.
    pub mode: MtaMode,
    /// How the zone and keep/drop are decided from the evidence.
    pub decision: MtaDecision,
    /// Explicit `libsylva_rivlib` file or directory.
    pub shim: Option<PathBuf>,
    /// Zone hypotheses asked from RiVLib (1..=8); 1 bypasses the resolution.
    pub max_zone: usize,
    /// Keep MTA-uncertain echoes (flagged `mta_uncertain_point`) in zone 1, on the pulse they were recorded
    /// after, instead of dropping them; nothing is uncertain with `max_zone == 1`.
    pub keep_uncertain: bool,
    /// Dynamic-trigger rule: drop echoes below this amplitude (dB) ...
    pub weak_amplitude: f32,
    /// ... when their zone-1 range is below this (m).
    pub weak_range: f32,
    /// Apply the RXP's notch window to resolved zone-1 echoes.
    pub notch: bool,
    /// Stop after this many scan lines (tests, quick looks).
    pub max_lines: Option<usize>,
    /// Thresholds and window sizes of the three decision passes.
    pub params: MtaParams,
}

impl Default for MtaOptions {
    fn default() -> Self {
        MtaOptions {
            mode: MtaMode::Auto,
            decision: MtaDecision::Default,
            shim: None,
            max_zone: 6,
            keep_uncertain: false,
            weak_amplitude: 2.5,
            weak_range: 18.0,
            notch: true,
            max_lines: None,
            params: MtaParams::default(),
        }
    }
}

/// What happened to the echoes of the last [`read_rxp_mta`] call (for logs and tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MtaStats {
    /// Echoes seen (count), before any filter.
    pub echoes: u64,
    /// Echoes dropped for an amplitude of 0 dB or less (count).
    pub amplitude_zero: u64,
    /// Echoes dropped by the dynamic-trigger rule, weak amplitude at short zone-1 range (count).
    pub weak: u64,
    /// Echoes dropped as MTA-uncertain: no zone, no finite range in the chosen zone, no pulse to
    /// attach to, or undecided and `keep_uncertain` off (count).
    pub uncertain: u64,
    /// Echoes dropped for a resolved range outside `min_range..=max_range` (count).
    pub range_gate: u64,
    /// Zone-1 echoes dropped by the RXP's notch window (count).
    pub notch: u64,
    /// Echoes written to the point cloud (count).
    pub kept: u64,
}

// ---- the shim's C interface -------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShimShot {
    time: f64,
    line: u32,
    shot: u32,
    facet: u8,
    flags: u8,
    n_echo: u8,
    pad: u8,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShimEcho {
    range: f64,
    amplitude: f32,
    reflectance: f32,
    deviation: f32,
    background: f32,
    flags: u16,
    nzone: u16,
    zone_range: [f32; 8],
    zone_reflectance: [f32; 8],
}

type Handle = *mut c_void;
type OpenFn = unsafe extern "C" fn(*const c_char, i32, i32, *mut Handle) -> i32;
type ReadFn = unsafe extern "C" fn(Handle, u32, *mut ShimShot, *mut u32, *mut ShimEcho, u32, *mut u32, *mut i32) -> i32;
type MetaFn = unsafe extern "C" fn(Handle, *mut c_char, u32) -> u32;
type ErrFn = unsafe extern "C" fn(Handle) -> *const c_char;
type CloseFn = unsafe extern "C" fn(Handle);
type VersionFn = unsafe extern "C" fn() -> i32;

const SHIM_NAMES: &[&str] = &["libsylva_rivlib.so", "sylva_rivlib.dll", "libsylva_rivlib.dylib"];
const API_VERSION: i32 = 1;

static SHIM: OnceLock<Mutex<Option<Library>>> = OnceLock::new();

/// Locate `libsylva_rivlib`: `hint`, `$SYLVA_RIVLIB_SHIM`, then the RiVLib
/// search roots (`$RIVLIB_PATH`, `~/.local/lib`, `/opt`, ...).
pub fn find_shim(hint: Option<&Path>) -> Result<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(h) = hint {
        roots.push(h.to_path_buf());
    }
    if let Ok(v) = std::env::var("SYLVA_RIVLIB_SHIM") {
        roots.push(PathBuf::from(v));
    }
    roots.extend(search_roots());
    for root in roots {
        if root.is_file() {
            return Ok(root);
        }
        if let Some(p) = search_dir(&root, 3, SHIM_NAMES) {
            return Ok(p);
        }
    }
    Err(Error::Riegl(
        "libsylva_rivlib not found: build it with crates/sylva-rs/rivlib-shim/build.sh (needs the RiVLib SDK) \
         and set SYLVA_RIVLIB_SHIM or put it next to libscanifc"
            .into(),
    ))
}

fn with_shim<T>(hint: Option<&Path>, f: impl FnOnce(&Library) -> Result<T>) -> Result<T> {
    let cell = SHIM.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if guard.is_none() {
        let path = find_shim(hint)?;
        // SAFETY: the shim is our own C library with no load-time side effects.
        let lib = unsafe { Library::new(&path) }.map_err(|e| Error::Riegl(format!("{}: {e}", path.display())))?;
        *guard = Some(lib);
    }
    f(guard.as_ref().unwrap())
}

// ---- per-line buffers -----------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ShotRec {
    time: f64,
    o: [f32; 3],
    d: [f32; 3],
    facet: u8,
    /// Targets RiVLib attached to the shot, before any filter.
    n_echo: u8,
}

#[derive(Clone, Copy)]
struct EchoRec {
    shot: u32,
    r1: f64,
    amp: f32,
    dev: f32,
    flags: u16,
    zr: [f32; MAXZ],
    zrefl: [f32; MAXZ],
}

/// An echo that survived the filters, placed on its pulse.
#[derive(Clone, Copy)]
struct Kept {
    /// The pulse's shot index in its line.
    shot: u32,
    xyz: Point,
    /// From the scanner's origin.
    range: f64,
    amp: f32,
    refl: f32,
    dev: u16,
    echo_type: u8,
    zone: u8,
    uncertain: u8,
}

struct Line {
    index: u32,
    shots: Vec<ShotRec>,
    echoes: Vec<EchoRec>,
    /// Echoes placed on this line's shots (by this line or, at zone > 1, the next).
    kept: Vec<Kept>,
    le: LineEchoes,
    grid: Option<Grid>,
    dec: Option<Vec<Decision>>,
    anchored: bool,
    s2_done: bool,
    emitted: bool,
}

impl Line {
    fn new(index: u32) -> Line {
        Line { index, shots: Vec::new(), echoes: Vec::new(), kept: Vec::new(), le: LineEchoes::default(), grid: None, dec: None, anchored: false, s2_done: false, emitted: false }
    }
    fn finish(&mut self, nzone: usize, p: &MtaParams) {
        self.le.n_shots = self.shots.len();
        self.le.nzone = nzone;
        self.le.shot = self.echoes.iter().map(|e| e.shot).collect();
        self.le.zr = self.echoes.iter().map(|e| e.zr).collect();
        self.le.reflectance = self.echoes.iter().map(|e| e.zrefl[0]).collect();
        self.grid = Some(Grid::new(&self.le, p));
    }
}

/// Output columns, one pulse per shot, grown line by line.
#[derive(Default)]
struct Out {
    direction: Vec<Point>,
    echo_count: Vec<u32>,
    range: Vec<f64>,
    amplitude: Vec<f32>,
    reflectance: Vec<f32>,
    deviation: Vec<u16>,
    echo_type: Vec<u8>,
    zone: Vec<u8>,
    uncertain: Vec<u8>,
    time: Vec<f64>,
    line: Vec<i32>,
    shot: Vec<i32>,
    facet: Vec<u8>,
    targets: Vec<u8>,
    /// Shots seen so far, for `shot_stride`.
    fired: usize,
}

impl Out {
    /// Append a line's shots as pulses, every `stride`-th fired shot.
    fn push_line(&mut self, mut line: Line, stride: usize) {
        // stable: equal ranges stay in stream order
        line.kept.sort_by(|a, b| a.shot.cmp(&b.shot).then(a.range.total_cmp(&b.range)));
        let mut k = 0;
        for (s, shot) in line.shots.iter().enumerate() {
            let first = k;
            while k < line.kept.len() && line.kept[k].shot == s as u32 {
                k += 1;
            }
            self.fired += 1;
            if stride > 1 && (self.fired - 1) % stride != 0 {
                continue;
            }
            let echoes = &line.kept[first..k];
            self.direction.push(match echoes.last() {
                Some(far) => normalize(&far.xyz),
                None => normalize(&shot.d.map(|v| v as f64)),
            });
            self.echo_count.push(echoes.len() as u32);
            for e in echoes {
                self.range.push(e.range);
                self.amplitude.push(e.amp);
                self.reflectance.push(e.refl);
                self.deviation.push(e.dev);
                self.echo_type.push(e.echo_type);
                self.zone.push(e.zone);
                self.uncertain.push(e.uncertain);
            }
            self.time.push(shot.time);
            self.line.push(-(line.index as i32) - 1); // RiSCAN's convention: negative, -1 is the first line
            self.shot.push(s as i32);
            self.facet.push(shot.facet);
            self.targets.push(shot.n_echo);
        }
    }

    fn into_shots(self, zones: bool, keep_uncertain: bool) -> Shots {
        let n = self.direction.len();
        let mut echo_attrs = std::collections::BTreeMap::new();
        echo_attrs.insert("amplitude".to_string(), Attr::F32(self.amplitude));
        echo_attrs.insert("reflectance".to_string(), Attr::F32(self.reflectance));
        echo_attrs.insert("deviation".to_string(), Attr::U16(self.deviation));
        echo_attrs.insert("echo_type".to_string(), Attr::U8(self.echo_type));
        if zones {
            echo_attrs.insert("mta_zone".to_string(), Attr::U8(self.zone));
            if keep_uncertain {
                echo_attrs.insert("mta_uncertain_point".to_string(), Attr::U8(self.uncertain));
            }
        }
        let mut pulse_attrs = std::collections::BTreeMap::new();
        pulse_attrs.insert("gps_time".to_string(), Attr::F64(self.time));
        pulse_attrs.insert("scan_line_index".to_string(), Attr::I32(self.line));
        pulse_attrs.insert("shot_index_line".to_string(), Attr::I32(self.shot));
        pulse_attrs.insert("mirror_facet".to_string(), Attr::U8(self.facet));
        pulse_attrs.insert("target_count".to_string(), Attr::U8(self.targets));
        Shots {
            origin: vec![[0.0; 3]; n],
            direction: self.direction,
            echo_start: packed_starts(&self.echo_count),
            echo_count: self.echo_count,
            echo_range: self.range,
            echo_attrs,
            pulse_attrs,
        }
    }
}

struct Pipeline<'a> {
    opts: &'a RxpOptions,
    m: &'a MtaOptions,
    /// `max_zone == 1`: no zone resolution, every echo in zone 1.
    bypass: bool,
    lines: VecDeque<Line>,
    out: Out,
    n: MtaStats,
    notch: Option<(f64, f64, f32)>,
    /// `SYLVA_MTA_FEATURES=<file>`: write one [`FEATURE_LEN`]-float record per echo
    /// (training data for a learned decision rule; see docs/guide/io.md).
    features: Option<std::io::BufWriter<std::fs::File>>,
    /// Learned decision rule, if any (None = the hand-written thresholds).
    model: Option<MtaModel>,
}

/// The model shipped with Sylva (`crates/sylva-rs/src/io/mta_model.json`), if built in.
pub const DEFAULT_MODEL_JSON: Option<&str> = default_model_json();

const fn default_model_json() -> Option<&'static str> {
    #[cfg(feature = "mta-model")]
    {
        Some(include_str!("mta_model.json"))
    }
    #[cfg(not(feature = "mta-model"))]
    {
        None
    }
}

fn load_decision(d: &MtaDecision) -> Result<Option<MtaModel>> {
    match d {
        MtaDecision::Rule => Ok(None),
        MtaDecision::Model(p) => MtaModel::load(p).map(Some),
        MtaDecision::Default => match DEFAULT_MODEL_JSON {
            Some(js) => MtaModel::from_json(js).map(Some),
            None => Ok(None),
        },
    }
}

/// Floats per echo in the feature dump: line, shot, echo index, echo count, r1, amplitude,
/// deviation, late flag, zone chosen, stage, then per zone (MAXZ): range, reflectance,
/// votes_own, votes_nb, anchors, density.
pub const FEATURE_LEN: usize = 10 + 6 * MAXZ;

impl Pipeline<'_> {
    /// Pass 1 for every line whose neighbours exist (or all, when `force`).
    fn run_stage1(&mut self, force: bool) {
        let p = &self.m.params;
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].dec.is_some() {
                continue;
            }
            if !force && i + p.line_nb >= n {
                break;
            }
            let dec = {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            self.lines[j as usize].grid.as_ref()
                        }
                    })
                    .collect();
                mta::stage1(&grids, &self.lines[i].le, p)
            };
            let line = &mut self.lines[i];
            if let Some(g) = line.grid.as_mut() {
                g.set_anchors(&line.le, &dec, p);
            }
            line.dec = Some(dec);
            line.anchored = true;
        }
    }

    /// Pass 2 for every line whose pass-1 neighbours exist; refreshes its anchors.
    fn run_stage2(&mut self, force: bool) {
        let p = self.m.params.clone();
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].s2_done || self.lines[i].dec.is_none() {
                continue;
            }
            let ready = force || (i + p.line_nb2 < n && self.lines[i + p.line_nb2].anchored);
            if !ready {
                break;
            }
            let mut dec = self.lines[i].dec.take().unwrap();
            {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb2)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb2 as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            let l = &self.lines[j as usize];
                            l.grid.as_ref().filter(|_| l.anchored)
                        }
                    })
                    .collect();
                mta::stage2(&grids, &self.lines[i].le, &mut dec, &p);
            }
            let line = &mut self.lines[i];
            if let Some(g) = line.grid.as_mut() {
                g.set_anchors(&line.le, &dec, &p);
            }
            line.dec = Some(dec);
            line.s2_done = true;
        }
    }

    /// Pass 3 and output for every line whose pass-2 neighbours exist.
    fn run_stage3(&mut self, force: bool) {
        let p = self.m.params.clone();
        let n = self.lines.len();
        for i in 0..n {
            if self.lines[i].emitted || !self.lines[i].s2_done {
                continue;
            }
            let ready = force || (i + p.line_nb3 < n && self.lines[i + p.line_nb3].s2_done);
            if !ready {
                break;
            }
            let mut dec = self.lines[i].dec.take().unwrap();
            {
                let grids: Vec<Option<&Grid>> = (0..=2 * p.line_nb3)
                    .map(|k| {
                        let j = i as i64 + k as i64 - p.line_nb3 as i64;
                        if j < 0 || j >= n as i64 {
                            None
                        } else {
                            let l = &self.lines[j as usize];
                            l.grid.as_ref().filter(|_| l.s2_done)
                        }
                    })
                    .collect();
                mta::stage3(&grids, &self.lines[i].le, &mut dec, &p);
            }
            self.emit(i, Some(&dec));
            self.lines[i].dec = Some(dec);
            self.lines[i].emitted = true;
        }
    }

    fn advance(&mut self, force: bool) {
        if self.bypass {
            for i in 0..self.lines.len() {
                if !self.lines[i].emitted {
                    self.emit(i, None);
                    self.lines[i].emitted = true;
                }
            }
        } else {
            self.run_stage1(force);
            self.run_stage2(force);
            self.run_stage3(force);
        }
        // Lines no longer needed go out as pulses: a line stays while a later
        // line within line_nb3 is still to be emitted, plus one before it
        // (an echo of the next line may still be placed on it).
        let keep = self.m.params.line_nb3 + 1;
        loop {
            let first_open = self.lines.iter().position(|l| !l.emitted).unwrap_or(self.lines.len());
            if first_open > keep && self.lines.len() > keep {
                let line = self.lines.pop_front().unwrap();
                self.out.push_line(line, self.opts.shot_stride);
            } else {
                break;
            }
        }
    }

    /// Filter line `i`'s echoes and place the survivors on their pulses; with
    /// `dec` None (one zone) every echo is in zone 1.
    fn emit(&mut self, i: usize, dec: Option<&[Decision]>) {
        let weak_a = self.m.weak_amplitude;
        let weak_r = self.m.weak_range as f64;
        let min_r = self.opts.min_range;
        let max_r = self.opts.max_range;
        let keep_unc = self.m.keep_uncertain;
        let notch = if self.m.notch { self.notch } else { None };
        let line_idx = self.lines[i].index;
        let prev_len = if i > 0 { self.lines[i - 1].shots.len() } else { 0 };
        // the learned rule, where one is loaded, overrides zone and keep/drop
        let dec_owned: Vec<Decision>;
        let dec: Option<&[Decision]> = match (dec, self.model.as_ref()) {
            (Some(dec), Some(model)) => {
                use rayon::prelude::*;
                let echoes = &self.lines[i].echoes;
                // pulses = runs of equal shot index; evaluate them in parallel
                let mut starts: Vec<usize> = vec![0];
                for e_i in 1..echoes.len() {
                    if echoes[e_i].shot != echoes[e_i - 1].shot {
                        starts.push(e_i);
                    }
                }
                if echoes.is_empty() {
                    starts.clear();
                }
                starts.push(echoes.len());
                let mut v = dec.to_vec();
                let n_shots = self.lines[i].shots.len().max(1) as f32;
                let out: Vec<(usize, Decision)> = starts
                    .par_windows(2)
                    .flat_map_iter(|w| {
                        let (j, e2) = (w[0], w[1]);
                        let n_echoes = (e2 - j).min(255) as u8;
                        (j..e2).map(move |e_i| {
                            let e = &echoes[e_i];
                            // closer echoes of the same pulse: strongest amplitude and the gap to the nearest
                            let (mut prev_amp, mut prev_gap, mut next_amp, mut nearest) = (-1.0f32, -1.0f32, -1.0f32, f64::INFINITY);
                            for o in &echoes[j..e2] {
                                if o.r1 < e.r1 {
                                    prev_amp = prev_amp.max(o.amp);
                                    let gap = e.r1 - o.r1;
                                    if gap < nearest {
                                        nearest = gap;
                                        prev_gap = gap as f32;
                                    }
                                } else if o.r1 > e.r1 {
                                    next_amp = next_amp.max(o.amp);
                                }
                            }
                            let info = EchoInfo {
                                amplitude: e.amp,
                                deviation: e.dev,
                                r1: e.r1 as f32,
                                late: e.flags & 32 != 0,
                                n_echoes,
                                idx_in_shot: (e_i - j + 1).min(255) as u8,
                                prev_amp,
                                prev_gap,
                                next_amp,
                                shot_frac: e.shot as f32 / n_shots,
                            };
                            let mut d = dec[e_i];
                            match model.decide(&e.zr, &e.zrefl, &d.evidence, info) {
                                Some((zone, p)) => {
                                    d.zone = zone;
                                    d.stage = if p >= model.threshold() { 4 } else { 0 };
                                    d.score = p;
                                }
                                None => {
                                    d.zone = 0;
                                    d.stage = 0;
                                }
                            }
                            (e_i, d)
                        })
                    })
                    .collect();
                for (e_i, d) in out {
                    v[e_i] = d;
                }
                dec_owned = v;
                Some(&dec_owned)
            }
            (dec, _) => dec,
        };
        if let (Some(w), Some(dec)) = (self.features.as_mut(), dec) {
            use std::io::Write;
            let mut rec = [0f32; FEATURE_LEN];
            let mut buf: Vec<u8> = Vec::with_capacity(FEATURE_LEN * 4 * self.lines[i].echoes.len());
            let mut last_shot = u32::MAX;
            let mut idx_in_shot = 0u32;
            for (e_i, e) in self.lines[i].echoes.iter().enumerate() {
                if e.shot != last_shot {
                    last_shot = e.shot;
                    idx_in_shot = 0;
                }
                idx_in_shot += 1;
                let d = dec[e_i];
                rec[0] = line_idx as f32;
                rec[1] = e.shot as f32;
                rec[2] = idx_in_shot as f32;
                rec[3] = 0.0;
                rec[4] = e.r1 as f32;
                rec[5] = e.amp;
                rec[6] = e.dev;
                rec[7] = ((e.flags >> 5) & 1) as f32;
                rec[8] = d.zone as f32;
                rec[9] = d.stage as f32;
                for z in 0..MAXZ {
                    rec[10 + z] = e.zr[z];
                    rec[10 + MAXZ + z] = e.zrefl[z];
                    rec[10 + 2 * MAXZ + z] = d.evidence.votes_own[z];
                    rec[10 + 3 * MAXZ + z] = d.evidence.votes_nb[z];
                    rec[10 + 4 * MAXZ + z] = d.evidence.anchors[z];
                    rec[10 + 5 * MAXZ + z] = d.evidence.density[z];
                }
                for v in rec {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            let _ = w.write_all(&buf);
        }
        // (line slot: 0 this line, 1 the previous one; the echo)
        let mut placed: Vec<(usize, Kept)> = Vec::new();
        for (e_i, e) in self.lines[i].echoes.iter().enumerate() {
            self.n.echoes += 1;
            if e.amp <= 0.0 {
                self.n.amplitude_zero += 1;
                continue;
            }
            if e.amp < weak_a && e.r1 < weak_r {
                self.n.weak += 1;
                continue;
            }
            let (k, uncertain) = match dec {
                Some(dec) => {
                    let d = dec[e_i];
                    let Some(k) = emit_zone(&d, keep_unc) else {
                        self.n.uncertain += 1;
                        continue;
                    };
                    (k, d.zone == 0 || d.stage == 0)
                }
                None => (1, false),
            };
            let r = e.zr[k - 1] as f64;
            if !r.is_finite() {
                self.n.uncertain += 1;
                continue;
            }
            if r < min_r || r > max_r {
                self.n.range_gate += 1;
                continue;
            }
            if k == 1 {
                if let Some((nmin, nmax, namp)) = notch {
                    if r >= nmin && r <= nmax && e.amp < namp {
                        self.n.notch += 1;
                        continue;
                    }
                }
            }
            // the pulse the echo belongs to: zone-1 shots earlier, possibly in the previous line
            let s = e.shot as i64 - (k as i64 - 1);
            let (slot, ps, shot) = if s >= 0 {
                (0, s, self.lines[i].shots[s as usize])
            } else if i > 0 && prev_len as i64 + s >= 0 {
                let ps = prev_len as i64 + s;
                (1, ps, self.lines[i - 1].shots[ps as usize])
            } else {
                self.n.uncertain += 1;
                continue;
            };
            let x = shot.o[0] as f64 + r * shot.d[0] as f64;
            let y = shot.o[1] as f64 + r * shot.d[1] as f64;
            let z = shot.o[2] as f64 + r * shot.d[2] as f64;
            let refl = if e.zrefl[k - 1].is_finite() { e.zrefl[k - 1] } else { e.zrefl[0] };
            placed.push((
                slot,
                Kept {
                    shot: ps as u32,
                    xyz: [x, y, z],
                    range: norm(&[x, y, z]),
                    amp: e.amp,
                    refl,
                    dev: e.dev.clamp(0.0, 65535.0) as u16,
                    echo_type: ((e.flags >> 6) & 3) as u8,
                    zone: k as u8,
                    uncertain: uncertain as u8,
                },
            ));
            self.n.kept += 1;
        }
        for (slot, kept) in placed {
            self.lines[i - slot].kept.push(kept);
        }
    }
}

/// The zone an echo is emitted in, or None to drop it: a confident decision's zone; an uncertain echo (no
/// zone, or no pass accepted it) is dropped, or with `keep_uncertain` stays on the pulse it was recorded after
/// (zone 1), as RiSCAN keeps unresolved echoes. Only a confident decision moves an echo to a farther zone.
fn emit_zone(d: &Decision, keep_uncertain: bool) -> Option<usize> {
    if d.zone == 0 || d.stage == 0 {
        keep_uncertain.then_some(1)
    } else {
        Some(d.zone as usize)
    }
}

fn json_f64(meta: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = meta.find(&pat)? + pat.len();
    let rest = &meta[i..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Read an `.rxp` through the shim: every shot a pulse, in stream order (see
/// the module docs). Of the [`RxpOptions`] it applies the drop of pseudo
/// echoes, the range gate and `shot_stride` (every n-th shot fired).
pub fn read_rxp_mta(path: &Path, opts: &RxpOptions, m: &MtaOptions) -> Result<Shots> {
    read_rxp_mta_with_stats(path, opts, m).map(|(s, _)| s)
}

/// [`read_rxp_mta`] that also reports what happened to the echoes.
pub fn read_rxp_mta_with_stats(path: &Path, opts: &RxpOptions, m: &MtaOptions) -> Result<(Shots, MtaStats)> {
    with_shim(m.shim.as_deref(), |lib| {
        // SAFETY: symbol signatures follow sylva_rivlib.cpp.
        let (open, read, meta, last_error, close, version) = unsafe {
            let open: Symbol<OpenFn> = lib.get(b"sylva_rxp_open\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let read: Symbol<ReadFn> = lib.get(b"sylva_rxp_read\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let meta: Symbol<MetaFn> = lib.get(b"sylva_rxp_meta\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let last_error: Symbol<ErrFn> = lib.get(b"sylva_rxp_last_error\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let close: Symbol<CloseFn> = lib.get(b"sylva_rxp_close\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let version: Symbol<VersionFn> = lib.get(b"sylva_rxp_api_version\0").map_err(|e| Error::Riegl(e.to_string()))?;
            (open, read, meta, last_error, close, version)
        };
        // SAFETY: plain call.
        let v = unsafe { version() };
        if v != API_VERSION {
            return Err(Error::Riegl(format!("libsylva_rivlib API version {v}, Sylva expects {API_VERSION}: rebuild the shim")));
        }
        let uri = CString::new(format!("file:{}", path.canonicalize().unwrap_or(path.to_path_buf()).display())).map_err(|e| Error::Riegl(e.to_string()))?;
        let mut handle: Handle = std::ptr::null_mut();
        let max_zone = m.max_zone.clamp(1, MAXZ) as i32;
        // SAFETY: valid C string and out-pointer; the notch is applied by us afterwards.
        let rc = unsafe { open(uri.as_ptr(), max_zone, 0, &mut handle) };
        let describe = |what: &str| -> Error {
            // SAFETY: handle is valid until close.
            let msg = unsafe { CStr::from_ptr(last_error(handle)) }.to_string_lossy().into_owned();
            Error::Riegl(format!("{}: {what}: {msg}", path.display()))
        };
        if rc != 0 {
            let e = describe("cannot open");
            // SAFETY: handle was allocated by open even on failure.
            unsafe { close(handle) };
            return Err(e);
        }
        const SHOTS: usize = 4096;
        const ECHOES: usize = SHOTS * 8 + 256;
        let zero_shot = ShimShot { time: 0.0, line: 0, shot: 0, facet: 0, flags: 0, n_echo: 0, pad: 0, ox: 0.0, oy: 0.0, oz: 0.0, dx: 0.0, dy: 0.0, dz: 0.0 };
        let zero_echo = ShimEcho { range: 0.0, amplitude: 0.0, reflectance: 0.0, deviation: 0.0, background: 0.0, flags: 0, nzone: 0, zone_range: [0.0; 8], zone_reflectance: [0.0; 8] };
        let mut shot_buf = vec![zero_shot; SHOTS];
        let mut echo_buf = vec![zero_echo; ECHOES];
        let bypass = max_zone == 1;
        let features = std::env::var_os("SYLVA_MTA_FEATURES").filter(|_| !bypass).and_then(|f| std::fs::File::create(f).ok()).map(std::io::BufWriter::new);
        let model = if bypass { None } else { load_decision(&m.decision)? };
        let mut pipe = Pipeline { opts, m, bypass, lines: VecDeque::new(), out: Out::default(), n: MtaStats::default(), notch: None, features, model };
        let mut cur: Option<Line> = None;
        let mut meta_read = false;
        let nzone = max_zone as usize;
        let mut stop = false;
        loop {
            let (mut ns, mut ne, mut eof) = (0u32, 0u32, 0i32);
            // SAFETY: buffers sized as declared.
            let rc = unsafe { read(handle, SHOTS as u32, shot_buf.as_mut_ptr(), &mut ns, echo_buf.as_mut_ptr(), ECHOES as u32, &mut ne, &mut eof) };
            if rc != 0 {
                let e = describe("read error");
                // SAFETY: handle valid.
                unsafe { close(handle) };
                return Err(e);
            }
            if !meta_read && ns > 0 {
                meta_read = true;
                let mut buf = vec![0u8; 1 << 16];
                // SAFETY: buffer size passed.
                let need = unsafe { meta(handle, buf.as_mut_ptr() as *mut c_char, buf.len() as u32) } as usize;
                if need > buf.len() {
                    buf = vec![0u8; need];
                    // SAFETY: as above.
                    unsafe { meta(handle, buf.as_mut_ptr() as *mut c_char, buf.len() as u32) };
                }
                let js = String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string();
                if let (Some(a), Some(b), Some(c)) = (json_f64(&js, "range_min"), json_f64(&js, "range_max"), json_f64(&js, "amplitude_max")) {
                    pipe.notch = Some((a, b, c as f32));
                }
            }
            let mut ei = 0usize;
            for si in 0..ns as usize {
                let s = shot_buf[si];
                let (line_no, shot_no, n_echo) = (s.line, s.shot, s.n_echo as usize);
                if cur.as_ref().map(|l| l.index != line_no).unwrap_or(true) {
                    if let Some(mut l) = cur.take() {
                        if !bypass {
                            l.finish(nzone, &m.params);
                        }
                        pipe.lines.push_back(l);
                        pipe.advance(false);
                        if let Some(max) = m.max_lines {
                            if line_no as usize >= max {
                                stop = true;
                            }
                        }
                    }
                    if stop {
                        break;
                    }
                    cur = Some(Line::new(line_no));
                }
                let l = cur.as_mut().unwrap();
                debug_assert_eq!(shot_no as usize, l.shots.len());
                let (time, ox, oy, oz, dx, dy, dz, facet) = (s.time, s.ox, s.oy, s.oz, s.dx, s.dy, s.dz, s.facet);
                l.shots.push(ShotRec { time, o: [ox, oy, oz], d: [dx, dy, dz], facet, n_echo: s.n_echo });
                for _ in 0..n_echo {
                    let e = echo_buf[ei];
                    ei += 1;
                    let (r1, amp, dev, flags, zr, zrefl) = (e.range, e.amplitude, e.deviation, e.flags, e.zone_range, e.zone_reflectance);
                    if opts.drop_pseudo_echoes && flags & super::riegl::FLAG_PSEUDO_ECHO != 0 {
                        continue;
                    }
                    l.echoes.push(EchoRec { shot: shot_no, r1, amp, dev, flags, zr, zrefl });
                }
            }
            if stop || eof != 0 {
                break;
            }
        }
        // SAFETY: handle valid.
        unsafe { close(handle) };
        if !stop {
            if let Some(mut l) = cur.take() {
                if !bypass {
                    l.finish(nzone, &m.params);
                }
                pipe.lines.push_back(l);
            }
        }
        pipe.advance(true);
        while let Some(line) = pipe.lines.pop_front() {
            pipe.out.push_line(line, opts.shot_stride);
        }
        Ok((pipe.out.into_shots(!bypass, m.keep_uncertain), pipe.n))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_shim() {
        assert_eq!(std::mem::size_of::<ShimShot>(), 44);
        assert_eq!(std::mem::size_of::<ShimEcho>(), 92);
    }

    #[test]
    fn kept_uncertain_echoes_take_zone_one() {
        let d = |zone, stage| Decision { zone, stage, ..Default::default() };
        assert_eq!(emit_zone(&d(2, 0), true), Some(1)); // uncertain, model's best zone 2: stays on its pulse
        assert_eq!(emit_zone(&d(0, 0), true), Some(1));
        assert_eq!(emit_zone(&d(2, 0), false), None);
        assert_eq!(emit_zone(&d(2, 4), true), Some(2)); // a confident zone-2 call stands
        assert_eq!(emit_zone(&d(2, 4), false), Some(2));
        assert_eq!(emit_zone(&d(1, 1), true), Some(1));
    }

    #[test]
    fn json_numbers() {
        let js = r#"{"notch":{"range_min":-0.035,"range_max":0.244,"amplitude_max":3},"n_shots":5}"#;
        assert_eq!(json_f64(js, "range_max"), Some(0.244));
        assert_eq!(json_f64(js, "amplitude_max"), Some(3.0));
        assert_eq!(json_f64(js, "missing"), None);
    }

    // A beam along +x from a mirror 0.1 m above the scanner's origin.
    fn shot(time: f64, n_echo: u8) -> ShotRec {
        ShotRec { time, o: [0.0, 0.0, 0.1], d: [1.0, 0.0, 0.0], facet: (time as u8) % 3, n_echo }
    }

    // An echo at zone-1 range `r`, with a zone-2 hypothesis 120 m further.
    fn echo(shot: u32, r: f32, amp: f32) -> EchoRec {
        let mut zr = [f32::NAN; MAXZ];
        zr[0] = r;
        zr[1] = r + 120.0;
        EchoRec { shot, r1: r as f64, amp, dev: 3.0, flags: 0, zr, zrefl: [-5.0; MAXZ] }
    }

    fn line(index: u32, shots: Vec<ShotRec>, echoes: Vec<EchoRec>) -> Line {
        let mut l = Line::new(index);
        l.shots = shots;
        l.echoes = echoes;
        l
    }

    fn pipeline<'a>(opts: &'a RxpOptions, m: &'a MtaOptions, bypass: bool) -> Pipeline<'a> {
        Pipeline { opts, m, bypass, lines: VecDeque::new(), out: Out::default(), n: MtaStats::default(), notch: None, features: None, model: None }
    }

    /// Two lines in one zone, read as the reader streams them.
    fn single_zone(opts: &RxpOptions) -> (Shots, MtaStats) {
        let m = MtaOptions { max_zone: 1, keep_uncertain: true, ..Default::default() };
        let mut p = pipeline(opts, &m, true);
        p.lines.push_back(line(
            0,
            vec![shot(0.0, 1), shot(1.0, 0), shot(2.0, 2), shot(3.0, 2)],
            vec![
                echo(0, 10.0, 10.0),
                echo(2, 5.0, 1.0),  // weak: under 2.5 dB within 18 m
                echo(2, 30.0, 0.0), // amplitude 0
                echo(3, 20.0, 5.0), // given far first
                echo(3, 15.0, 6.0),
            ],
        ));
        p.advance(false);
        p.lines.push_back(line(1, vec![shot(4.0, 1), shot(5.0, 1)], vec![echo(0, 0.3, 10.0), echo(1, 7.0, 4.0)]));
        p.advance(true);
        while let Some(l) = p.lines.pop_front() {
            p.out.push_line(l, opts.shot_stride);
        }
        (p.out.into_shots(false, m.keep_uncertain), p.n)
    }

    #[test]
    fn every_shot_is_a_pulse_in_stream_order() {
        let (s, n) = single_zone(&RxpOptions::default());
        // shots 1 (no target), 2 (both filtered) and 4 (range gate) are pulses without echoes
        assert_eq!(s.echo_count, vec![1, 0, 0, 2, 0, 1]);
        assert_eq!(s.pulse_attrs["target_count"], Attr::U8(vec![1, 0, 2, 2, 1, 1]));
        assert_eq!(s.pulse_attrs["gps_time"], Attr::F64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(s.pulse_attrs["scan_line_index"], Attr::I32(vec![-1, -1, -1, -1, -2, -2]));
        assert_eq!(s.pulse_attrs["shot_index_line"], Attr::I32(vec![0, 1, 2, 3, 0, 1]));
        assert_eq!(s.pulse_attrs["mirror_facet"], Attr::U8(vec![0, 1, 2, 0, 1, 2]));
        assert_eq!(n, MtaStats { echoes: 7, amplitude_zero: 1, weak: 1, uncertain: 0, range_gate: 1, notch: 0, kept: 4 });
        // ranges from the scanner's origin, nearest first; the direction points at the farthest
        let z = 0.1f32 as f64; // the mirror's height, as the shim gives it
        let r = |x: f64| (x * x + z * z).sqrt();
        assert_eq!(s.echo_range, vec![r(10.0), r(15.0), r(20.0), r(7.0)]);
        assert_eq!(s.echo_attrs["amplitude"], Attr::F32(vec![10.0, 6.0, 5.0, 4.0]));
        assert!(s.origin.iter().all(|o| *o == [0.0; 3]));
        let far = s.echo_xyz()[2];
        assert!((far[0] - 20.0).abs() < 1e-12 && far[2] == z);
        // a pulse without echoes points along its beam
        assert_eq!(s.direction[1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn one_zone_bypasses_the_resolution() {
        let (s, _) = single_zone(&RxpOptions::default());
        // zone-1 ranges though a zone-2 hypothesis exists; no zone columns, even with keep_uncertain
        assert!(s.echo_range.iter().all(|&r| r < 21.0));
        assert_eq!(s.echo_attrs.keys().collect::<Vec<_>>(), vec!["amplitude", "deviation", "echo_type", "reflectance"]);
    }

    #[test]
    fn shot_stride_counts_every_shot_fired() {
        let (s, _) = single_zone(&RxpOptions { shot_stride: 2, ..Default::default() });
        assert_eq!(s.pulse_attrs["gps_time"], Attr::F64(vec![0.0, 2.0, 4.0]));
        assert_eq!(s.echo_count, vec![1, 0, 0]);
        assert_eq!(s.pulse_attrs["target_count"], Attr::U8(vec![1, 2, 1]));
    }

    #[test]
    fn a_zone_2_echo_moves_to_the_pulse_before_across_a_line() {
        let opts = RxpOptions::default();
        let m = MtaOptions { max_zone: 2, keep_uncertain: true, ..Default::default() };
        let mut p = pipeline(&opts, &m, false);
        p.lines.push_back(line(0, vec![shot(0.0, 0), shot(1.0, 1), shot(2.0, 0)], vec![echo(1, 9.0, 8.0)]));
        p.lines.push_back(line(1, vec![shot(3.0, 1), shot(4.0, 0)], vec![echo(0, 4.0, 8.0)]));
        let d = |zone, stage| Decision { zone, stage, ..Default::default() };
        p.emit(0, Some(&[d(1, 0)])); // uncertain, kept on request
        p.emit(1, Some(&[d(2, 1)])); // recorded after shot 3: fired with shot 2, the last of line 0
        while let Some(l) = p.lines.pop_front() {
            p.out.push_line(l, 1);
        }
        let s = p.out.into_shots(true, true);
        assert_eq!(s.echo_count, vec![0, 1, 1, 0, 0]);
        assert_eq!(s.pulse_attrs["target_count"], Attr::U8(vec![0, 1, 0, 1, 0]));
        let z = 0.1f32 as f64;
        assert_eq!(s.echo_range[1], (124.0f64 * 124.0 + z * z).sqrt()); // at its zone-2 range
        assert_eq!(s.echo_attrs["mta_zone"], Attr::U8(vec![1, 2]));
        assert_eq!(s.echo_attrs["mta_uncertain_point"], Attr::U8(vec![1, 0]));
    }

    #[test]
    fn missing_shim_is_a_clear_error() {
        // a hint that does not exist falls through to the usual roots; only the
        // message is checked, since a developer machine may have the shim
        if let Err(Error::Riegl(msg)) = find_shim(Some(Path::new("/definitely/not/here"))) {
            assert!(msg.contains("libsylva_rivlib"));
        }
    }
}
