// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A plant's base re-sectioned from the rays (3dtrees ticket 125): the columns of wood standing in its bottom metre
//! and a half, found in every scan's pulses at once, and its rows there chosen between the
//! skeleton's and the columns' by the rays' evidence for each ([`rebase`]), for [`super::support_tree`] to measure.
//!
//! - **Slabs.** From [`BaseParams::z_lo`] to [`BaseParams::z_hi`] above the ground, slabs [`BaseParams::dz`] thick,
//!   each a vertical piece about the plant's base whose tube is [`BaseParams::reach`] wide: each scan's pulses
//!   through it, as [`super::groups`] takes a piece's.
//! - **Sections.** A slab's woody stops (echo character over a half, [`super::p_wood`] for wood of [`WOOD_R`]), from
//!   every scan, linked where they lie within [`BaseParams::link`] of each other, are its sections: a stem is a
//!   connected arc or ring of stops, a grass tussock or leaf clump a scatter the veto refutes. Each section with
//!   [`super::RUN_HITS`] woody hits gets a circle: RANSAC and least squares on its stops ([`super::point_circle`]'s
//!   band), filling at least [`ARC_MIN`] ten-degree bins, its arc's covariance ([`arc_cov`]) for its sd; a section
//!   whose stops make no circle (a sliver one scan grazes) is centred behind its stops by half their width. The ray
//!   veto ([`super::tally`]) audits each circle, and a refuted one, or one at the tube's edge, is left out.
//! - **Columns.** Sections are linked upward, slab by slab (a slab's gap bridged), each to the column whose last
//!   section's centre lies nearest, within half the wider radius and [`LINK_GAP`]; where two sections claim one
//!   column the nearer continues it and the other starts a fork of it. A column of fewer than [`MIN_SECTIONS`]
//!   sections and no fork is dropped (grass, a stray stop).
//! - **Evidence** ([`support`]): a circle's log odds of wood against foliage from the slab's pulses through it, by
//!   their opacity as a piece's existence weighs its core's ([`JointParams::stop_wood`], [`JointParams::stop_leaf`],
//!   [`JointParams::exist_scale`]): a pulse that stops on its chord counts for it, one that passes through against
//!   it, one stopped in front of it not at all. Not by echo character: the woody stops chose the sections already,
//!   and what tells a skeleton's base from a column is where the pulses stop and pass (on a synthetic stand, whose
//!   echo character isn't bark's, it would weigh every stop as leaf). A skeleton row is weighed at its axis in each
//!   slab it crosses, at the best of its radius and narrower ([`NARROW`]: the skeleton's radius is a taper prior).
//!
//! **The join** ([`rebase`]), over the plant's grounded members' rows, `H` the slabs' top:
//! - **Believable columns.** A column whose median radius is over [`COL_R_MAX`] or the plant's cap (the site's D/H
//!   at its tallest point, times [`super::PTS_BELIEVE`], the joint fit's own ceiling on a point circle) is no stem
//!   (the ground, a rock); one with [`CTX_SHARE`] of its sections inside another plant's cylinder is that plant's;
//!   one standing on the ground must start within [`FOOT`] of it (its base would be drawn through what no section
//!   saw).
//!   A fork of one left out stands on the ground.
//! - **Entries.** A row rising through a column's top (its highest section) is an entry there, cut at that height;
//!   its chain is its ancestors under the cut. It joins the column whose top lies nearest its cut, within [`JOIN`]
//!   beyond the column's radius; else it is grafted to the nearest column top within [`GRAFT`] where the rays refute
//!   its chain (its evidence is negative: a fan's axis through air); else it stays. A row on another entry's chain is
//!   that entry's.
//! - **The choice, by likelihood.** A column (with its forks) replaces the chains of the entries it takes only where
//!   its sections' evidence is positive and those chains hold under [`BaseParams::keep`] of it in the same slabs,
//!   and under [`MEASURED`] of them are measured (by their own circle, unrefuted, as the joint fit would serve it): a skeleton
//!   already on its wood keeps it; one beside it, inside a trunk wider than its claim, or a fan's axes through air,
//!   don't; a column of thin stems the rays see between loses to their own rows. A column another member's rows
//!   already stand on (on [`CTX_SHARE`] of its sections) isn't given to members other than it.
//! - **Ownership.** A column's rows (one a section, its prior that section's radius) go to the member whose taken
//!   entries carry the most of the skeleton's volume above their cuts (of its own and its forks' entries), from the
//!   ground under it, or from its owner's own base where the owner keeps one (a stem the columns missed stands there
//!   too); a fork owned by another member starts that member's base at the fork, and an entry whose column is another
//!   member's starts its member's base at its cut. A column no entry takes is left out. A member that would stand on
//!   two bases keeps its rows as they were.
//! - **What else under the cut** (the chosen columns' top), of a member re-sectioned: the chains replaced are
//!   dropped, and so is any other row whose middle lies in a chosen column's circle (the same wood twice) or that
//!   the veto refutes at a quarter of its radius in every slab it crosses (a fan's row through air); a thin stem
//!   beside the column is neither. A kept row whose parent was dropped hangs from its member's nearest row end within
//!   [`HANG`] no more than [`HANG_UP`] above its start (a branch leaves its stem at its height); else, if its subtree
//!   reaches above `H`, it is joined to such an end within [`CONNECT`] by a row of its dropped parent's radius (the
//!   stem RCT drew there, redrawn from the column), or starts a base of its own (so its member keeps its rows); any
//!   other is dropped.
use super::*;
use crate::trees::fit_circle;

/// The wood radius a stop's echo character is judged at (m): a stem, not a twig.
const WOOD_R: f64 = 0.05;
/// How far beyond half the wider radius a section may sit from the column below it (m: a lean of ~25 degrees over
/// one 10 cm slab, or a slab skipped).
const LINK_GAP: f64 = 0.05;
/// The fewest sections a column without a fork needs (three slabs: 30 cm of stem).
const MIN_SECTIONS: usize = 3;
/// How far beyond its circle (and the fit's band) a section's stops may lie (m: the linking distance), and the share
/// that may lie further (stray stops: a leaf, a grass blade linked to the stem).
const SPAN_SLACK: f64 = 0.02;
const SPAN_OUT: f64 = 0.1;
/// A scan counts toward a section's `n_scans` with at least this many of its stops.
const SCAN_STOPS: usize = 5;
/// A section's circle fills at least this many of the 36 ten-degree bins (60 degrees: one scan's view of a trunk's
/// front), its sd the covariance its arc leaves ([`arc_cov`]): its stops' rms about it (at least [`ARC_NOISE`]) over
/// the [`ARC_BIN`]-degree bins they fill (a scan's stops in one such bin are one look at the surface).
const ARC_MIN: usize = 6;
const ARC_NOISE: f64 = 0.003;
const ARC_BIN: f64 = 2.0;
/// The join (module documentation), in m: an entry's reach to a column's top beyond its radius; a graft's reach; how far beyond a column's circle a row's
/// middle still lies in it; the widest column that is a stem; how far a kept row's start may lie from the row end
/// it is hung from; how far a row reaching above the slabs' top may be joined to its member's nearest row end.
const JOIN: f64 = 0.08;
const GRAFT: f64 = 0.3;
const IN_COL: f64 = 0.02;
const COL_R_MAX: f64 = 0.3;
/// The highest a column standing on the ground may start (m above it): its first section's middle.
const FOOT: f64 = 0.5;
const HANG: f64 = 0.3;
const CONNECT: f64 = 1.0;
/// The share of a column's sections inside another plant's cylinder (within [`IN_COL`] of its radius) that makes
/// it that plant's.
const CTX_SHARE: f64 = 0.5;
/// The share of the skeleton's rows a column would replace (by length, in its slabs) whose pieces the rays measure (an
/// own circle the veto doesn't refute, as `support_tree` would serve it) that keeps the base: a base already measured
/// stays.
const MEASURED: f64 = 0.5;
/// How far above a row's start the end of the row it is hung from may lie (m): a branch leaves its stem at its height.
const HANG_UP: f64 = 0.05;
/// The radii a skeleton row is weighed at, as shares of its own: the best of them is its evidence in a slab.
const NARROW: [f64; 3] = [1.0, 0.5, 0.25];
/// The narrowest circle weighed (m): about the beam's footprint.
const NARROW_MIN: f64 = 0.003;

/// Where and how to re-section a base.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaseParams {
    /// The slabs, above the ground: the first's bottom, the last's top and their thickness (m).
    pub z_lo: f64,
    pub z_hi: f64,
    pub dz: f64,
    /// The tube about the base (m): every stem of the plant stands within it.
    pub reach: f64,
    /// Stops within this of each other are one section (m).
    pub link: f64,
    /// The share of a column's evidence the skeleton's rows it would replace must hold, in the same slabs, to keep the
    /// base: then they stand on its wood (the joint fit measures a base whose axis is on its wood; one beside it, or a
    /// fan's axes through air, it can't). A ratio, not a price per circle: a slab's pulses number thousands and are
    /// correlated far beyond what `exist_scale` discounts, so the evidence's scale says little, its share much.
    pub keep: f64,
}

impl Default for BaseParams {
    fn default() -> Self {
        BaseParams { z_lo: 0.1, z_hi: 1.5, dz: 0.1, reach: 0.6, link: 0.02, keep: 0.8 }
    }
}

/// One slab's section: its slab and height (world), centre (world x, y), radius and sd of log R (NaN where no
/// circle was fitted: `fitted`), the ten-degree bins its stops fill, its scans and woody hits, the veto's audit
/// (passes through its circle, their share, `refuted`), whether it reaches the tube's edge, its evidence
/// ([`support`]) and the pulses through it, and its column (-1: none).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Section {
    pub slab: u32,
    pub z: f64,
    pub c: [f64; 2],
    pub r: f64,
    pub sd_log_r: f64,
    pub fitted: bool,
    pub arc: u32,
    pub n_scans: u32,
    pub hits: f64,
    pub n_pass: u32,
    pub pass_share: f64,
    pub refuted: bool,
    pub edge: bool,
    pub evidence: f64,
    pub n_through: u32,
    pub column: i32,
}

/// A column: its sections, bottom up, and where it forks from another (its parent column and that column's section
/// below its own first; -1, -1 at the ground).
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub sections: Vec<usize>,
    pub parent: i32,
    pub at: i32,
}

/// Union-find's root, with path halving.
fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

/// A circle fitted to an arc's stops `on` (about its centre): the covariance of `(c1, c2, R)` the arc leaves
/// ([`ARC_MIN`]), its own noise aside; None where the arc doesn't determine it.
fn arc_cov(on: &[[f64; 2]], rmse: f64) -> Option<Matrix3<f64>> {
    let mut jtj = Matrix3::zeros();
    let mut bins = std::collections::HashSet::new();
    for q in on {
        let d = q[0].hypot(q[1]).max(1e-9);
        let j = Vector3::new(-q[0] / d, -q[1] / d, -1.0);
        jtj += j * j.transpose();
        bins.insert((q[1].atan2(q[0]).to_degrees() / ARC_BIN).floor() as i64);
    }
    let s2 = rmse.max(ARC_NOISE).powi(2) * on.len() as f64 / bins.len().max(1) as f64;
    jtj.try_inverse().map(|i| i * s2).filter(|c| c.iter().all(|x| x.is_finite()) && c[(2, 2)] > 0.0)
}

/// A circle `x` (`[c1, c2, t1, t2, R]`, in the slab's frame)'s evidence against the slab's groups, by their
/// opacity (module documentation), and the pulses through it.
fn support(gs: &[Group], x: &Vector5<f64>, p: &JointParams) -> (f64, u32) {
    let (on, off) = ((p.stop_wood / p.stop_leaf).ln(), ((1.0 - p.stop_wood) / (1.0 - p.stop_leaf)).ln());
    let (mut e, mut n) = (0.0, 0u32);
    for g in gs {
        for r in g.recs.iter().filter(|r| !r.blocked) {
            let (cx, cy) = (x[0] + r.s * x[2], x[1] + r.s * x[3]);
            let bc = r.b - (g.n[0] * cx + g.n[1] * cy);
            if bc.abs() >= x[4] {
                continue;
            }
            let half = (x[4] * x[4] - bc * bc).sqrt();
            let d = r.depth - (g.v[0] * cx + g.v[1] * cy);
            if !d.is_finite() || d > half + VETO_TOL {
                (e, n) = (e + off, n + 1);
            } else if d >= -half - VETO_TOL {
                (e, n) = (e + on, n + 1);
            }
        }
    }
    (p.exist_scale * e, n)
}

/// A slab's sections: its stops linked within `link`, each group with enough woody hits given a circle.
fn sections(gs: &[Group], slab: u32, z: f64, centre: &Point, p: &BaseParams, like: &JointParams) -> Vec<Section> {
    let stops: Vec<(usize, [f64; 2], f64)> = gs.iter().enumerate()
        .flat_map(|(gi, g)| g.recs.iter().filter(|r| r.is_stop()).map(move |r| (gi, r.xy, p_wood(r, WOOD_R, like))))
        .filter(|s| s.2 >= 0.5).collect();
    let cell = |q: [f64; 2]| ((q[0] / p.link).floor() as i64, (q[1] / p.link).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
    for (i, s) in stops.iter().enumerate() {
        grid.entry(cell(s.1)).or_default().push(i);
    }
    let mut parent: Vec<usize> = (0..stops.len()).collect();
    for (i, s) in stops.iter().enumerate() {
        let (cx, cy) = cell(s.1);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(cx + dx, cy + dy)).map_or(&[][..], |v| &v[..]) {
                    if j > i && (s.1[0] - stops[j].1[0]).hypot(s.1[1] - stops[j].1[1]) <= p.link {
                        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                        parent[a] = b;
                    }
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..stops.len() {
        let r = root(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out = Vec::new();
    for members in groups.values() {
        let hits: f64 = members.iter().map(|&i| stops[i].2).sum();
        if hits < RUN_HITS || members.len() < PTS_MIN {
            continue;
        }
        let xy: Vec<[f64; 2]> = members.iter().map(|&i| stops[i].1).collect();
        let mut per = vec![0usize; gs.len()];
        for &i in members {
            per[stops[i].0] += 1;
        }
        let n_scans = per.iter().filter(|&&n| n >= SCAN_STOPS).count() as u32;
        let (lo, hi) = xy.iter().fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), q| {
            ([lo[0].min(q[0]), lo[1].min(q[1])], [hi[0].max(q[0]), hi[1].max(q[1])])
        });
        let extent = (hi[0] - lo[0]).hypot(hi[1] - lo[1]);
        let (x, sd, arc, fitted) = match circle_of(&xy, extent) {
            Some((x, sd, arc)) => (x, sd, arc, true),
            None => {
                // a sliver: its stops are the front of a stem as wide as they are, behind them from the scan that
                // saw most of it
                let n = xy.len() as f64;
                let m = [xy.iter().map(|q| q[0]).sum::<f64>() / n, xy.iter().map(|q| q[1]).sum::<f64>() / n];
                let main = per.iter().enumerate().max_by_key(|q| *q.1).map_or(0, |q| q.0);
                let r = (0.5 * extent).max(0.005);
                (Vector5::new(m[0] + r * gs[main].v[0], m[1] + r * gs[main].v[1], 0.0, 0.0, r), f64::NAN, 0, false)
            }
        };
        let (stop, pass) = tally(gs, &x);
        let (evidence, n_through) = support(gs, &x, like);
        out.push(Section {
            slab,
            z,
            c: [centre[0] + x[0], centre[1] + x[1]],
            r: x[4],
            sd_log_r: sd,
            fitted,
            arc,
            n_scans,
            hits,
            n_pass: pass,
            pass_share: if stop + pass > 0 { f64::from(pass) / f64::from(stop + pass) } else { f64::NAN },
            refuted: refuted((stop, pass)),
            edge: x[0].hypot(x[1]) + x[4] > p.reach - TUBE_MARGIN,
            evidence,
            n_through,
            column: -1,
        });
    }
    out
}

/// A section's circle from its stops: `[c1, c2, 0, 0, R]`, its sd of log R (its arc's and [`PTS_SD`]) and the
/// ten-degree bins its stops fill; None where they make none (an arc under [`ARC_MIN`], or a poor fit). The circle
/// must span its section: the stops are one connected object, so a circle narrower than their span (a small circle
/// RANSAC finds in a big trunk's bark) is not its section's; the stops are then fitted whole (least squares).
fn circle_of(xy: &[[f64; 2]], extent: f64) -> Option<(Vector5<f64>, f64, u32)> {
    let n = xy.len() as f64;
    let band = (PTS_BAND_SHARE * 0.5 * extent).min(PTS_BAND).max(PTS_BAND_MIN);
    // the share of the stops outside the circle's disc (beyond its band and SPAN_SLACK): a circle spans its section
    // where at most SPAN_OUT of them are
    let spans = |cx: f64, cy: f64, r: f64| {
        xy.iter().filter(|q| (q[0] - cx).hypot(q[1] - cy) > r + band + SPAN_SLACK).count() as f64 <= SPAN_OUT * n
    };
    let ransac = RansacCircleParams { threshold: band, iterations: 200, min_radius: 0.005, max_radius: 0.6, seed: 0 };
    let (cx, cy, r, inl) = match fit_circle_ransac(xy, &ransac).ok() {
        Some((cx, cy, r, inl)) if spans(cx, cy, r) => (cx, cy, r, inl),
        _ => {
            let (cx, cy, r, _) = fit_circle(xy).ok()?;
            let inl = xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() <= band).collect();
            (cx, cy, r, inl)
        }
    };
    let on: Vec<[f64; 2]> = xy.iter().zip(&inl).filter(|q| *q.1).map(|(q, _)| [q[0] - cx, q[1] - cy]).collect();
    if (on.len() as f64) < PTS_INLIERS * n || !(r.is_finite() && r > 0.0) || r > extent.max(0.01) * 1.5 || !spans(cx, cy, r) {
        return None;
    }
    let rmse = (on.iter().map(|q| (q[0].hypot(q[1]) - r).powi(2)).sum::<f64>() / on.len() as f64).sqrt();
    let mut bins = [false; 36];
    for q in &on {
        bins[(((q[1].atan2(q[0]).to_degrees() + 180.0) / 10.0) as usize).min(35)] = true;
    }
    let arc = bins.iter().filter(|&&b| b).count();
    if arc < ARC_MIN || rmse > PTS_RMSE.0 * r + PTS_RMSE.1 {
        return None;
    }
    let var = arc_cov(&on, rmse).map_or(f64::INFINITY, |c| c[(2, 2)]) / (r * r);
    Some((Vector5::new(cx, cy, 0.0, 0.0, r), (PTS_SD * PTS_SD + var).sqrt(), arc as u32))
}

/// Each slab's groups and sections about `centre` (world x, y and the ground's z).
fn slabs(scans: &[&ScanIndex], centre: &Point, p: &BaseParams, like: &JointParams, threads: usize) -> Result<Vec<(Vec<Group>, Vec<Section>)>> {
    if !(p.dz > 0.0 && p.z_hi > p.z_lo && p.reach > 0.0 && p.link > 0.0) {
        return Err(Error::invalid("base columns: dz, reach and link must be positive and z_hi above z_lo"));
    }
    let n = ((p.z_hi - p.z_lo) / p.dz).round() as usize;
    let frame: Frame = ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], p.dz);
    Ok(pool(threads)?.install(|| {
        (0..n).into_par_iter().map(|k| {
            let z = centre[2] + p.z_lo + k as f64 * p.dz;
            let (a, b) = ([centre[0], centre[1], z], [centre[0], centre[1], z + p.dz]);
            let gs = groups(scans, &a, &b, frame, p.reach, like);
            let s = sections(&gs, k as u32, z + 0.5 * p.dz, centre, p, like);
            (gs, s)
        }).collect()
    }))
}

/// The columns linking the slabs' sections (module documentation): every section kept, each with its column.
fn columns(slabs: Vec<Vec<Section>>) -> (Vec<Section>, Vec<Column>) {
    let n = slabs.len() as u32;
    // one stem per circle: in each slab, a section whose centre lies inside a stronger one's circle (fitted first,
    // then by woody hits) is the same stem seen in pieces (a scan's sliver beside another's arc), and is left out
    let mut secs: Vec<Section> = Vec::new();
    for mut slab in slabs {
        slab.retain(|s| !s.refuted && !s.edge);
        slab.sort_by(|a, b| b.fitted.cmp(&a.fitted).then(b.hits.total_cmp(&a.hits)));
        let mut kept: Vec<Section> = Vec::new();
        for s in slab {
            if kept.iter().all(|k| (s.c[0] - k.c[0]).hypot(s.c[1] - k.c[1]) > s.r.max(k.r)) {
                kept.push(s);
            }
        }
        secs.extend(kept);
    }
    // link upward: each section to the nearest column whose last section is one or two slabs below
    let mut cols: Vec<Column> = Vec::new();
    for k in 0..n {
        let here: Vec<usize> = (0..secs.len()).filter(|&i| secs[i].slab == k).collect();
        let mut claims: Vec<(usize, usize, f64)> = Vec::new();
        for &i in &here {
            let best = cols.iter().enumerate().filter_map(|(c, col)| {
                let last = &secs[*col.sections.last()?];
                let d = (secs[i].c[0] - last.c[0]).hypot(secs[i].c[1] - last.c[1]);
                (last.slab + 2 >= k && last.slab < k && d <= 0.5 * secs[i].r.max(last.r) + LINK_GAP).then_some((c, d))
            }).min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((c, d)) = best {
                claims.push((i, c, d));
            } else {
                secs[i].column = cols.len() as i32;
                cols.push(Column { sections: vec![i], parent: -1, at: -1 });
            }
        }
        claims.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut taken = std::collections::HashSet::new();
        for (i, c, _) in claims {
            if taken.insert(c) {
                secs[i].column = c as i32;
                cols[c].sections.push(i);
            } else {
                let at = *cols[c].sections[..cols[c].sections.len() - 1].last().unwrap_or(&cols[c].sections[0]);
                secs[i].column = cols.len() as i32;
                cols.push(Column { sections: vec![i], parent: c as i32, at: at as i32 });
            }
        }
    }
    // drop short columns without forks, renumbering the rest
    let forked: std::collections::HashSet<i32> = cols.iter().map(|c| c.parent).collect();
    let keep: Vec<bool> = (0..cols.len()).map(|c| cols[c].sections.len() >= MIN_SECTIONS || forked.contains(&(c as i32))).collect();
    let mut id = vec![-1i32; cols.len()];
    let mut out = Vec::new();
    for c in 0..cols.len() {
        if keep[c] && (cols[c].parent < 0 || keep[cols[c].parent as usize]) {
            id[c] = out.len() as i32;
            out.push(cols[c].clone());
        }
    }
    for c in &mut out {
        c.parent = if c.parent >= 0 { id[c.parent as usize] } else { -1 };
        if c.parent < 0 {
            c.at = -1;
        }
    }
    for s in &mut secs {
        s.column = if s.column >= 0 { id[s.column as usize] } else { -1 };
    }
    (secs, out)
}

/// What became of a column in [`rebase`]: its evidence and its replaced chains' in its slabs, and the share of those
/// chains the rays measure (each its root column's, with its forks), whether it was chosen and the member it went to (-1: none),
/// its median radius, whether it is believable as this plant's stem, and the entries it took (joined, grafted).
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnOut {
    pub e_col: f64,
    pub e_rct: f64,
    pub measured: f64,
    pub chosen: bool,
    pub owner: i64,
    pub r: f64,
    pub believable: bool,
    pub joined: u32,
    pub grafted: u32,
}

/// A plant's rows with its base re-sectioned ([`rebase`]): each row's axis, prior radius, parent (a row of the same
/// member, -1 at its base), member and column (-1: a skeleton row); every member's rows hang from one base, parents
/// first. With them the columns' sections and what became of each column, and counts of what the join did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rebased {
    pub a: Vec<Point>,
    pub b: Vec<Point>,
    pub r_prior: Vec<f64>,
    pub parent: Vec<i64>,
    pub member: Vec<i64>,
    pub column: Vec<i32>,
    pub sections: Vec<Section>,
    pub columns: Vec<Column>,
    pub out: Vec<ColumnOut>,
    /// entries; their fates: joined, grafted, kept; rows dropped; rows hung; members kept as they were (two bases)
    pub counts: [u32; 7],
}

/// A skeleton row's evidence in a slab of mid-height `z` (`groups`, in the slab's frame about `centre`): at its
/// axis there, leaning as it does, the best of its radius and narrower ([`NARROW`]); None where it doesn't cross
/// `z` or lies flat.
fn row_support(gs: &[Group], a: &Point, b: &Point, r: f64, z: f64, centre: &Point, p: &JointParams) -> Option<f64> {
    let (lo, hi) = (a[2].min(b[2]), a[2].max(b[2]));
    let dz = b[2] - a[2];
    if !(lo <= z && z < hi) || dz.abs() < 1e-6 {
        return None;
    }
    let f = (z - a[2]) / dz;
    let (t1, t2) = ((b[0] - a[0]) / dz, (b[1] - a[1]) / dz);
    let (c1, c2) = (a[0] + f * (b[0] - a[0]) - centre[0], a[1] + f * (b[1] - a[1]) - centre[1]);
    NARROW.iter().map(|s| support(gs, &Vector5::new(c1, c2, t1, t2, (s * r).max(NARROW_MIN)), p).0).max_by(f64::total_cmp)
}

/// A plant's base re-sectioned from the rays (module documentation): its grounded members' rows (`rows`, each one's
/// `member`, parents first and within a member), about `centre` (x, y and the ground's z) with the slabs `p`, the
/// other plants' cylinders `context`, the site's D/H `dh` (NaN: no cap) and the likelihood `like`.
#[allow(clippy::too_many_arguments)]
pub fn rebase(scans: &[&ScanIndex], rows: &Rows, member: &[i64], centre: &Point, p: &BaseParams, context: &[(Point, Point, f64)], like: &JointParams, prior: &PriorParams, threads: usize) -> Result<Rebased> {
    let n = rows.a.len();
    if rows.b.len() != n || rows.r_prior.len() != n || rows.parent.len() != n || member.len() != n {
        return Err(Error::invalid("rebase: a, b, r_prior, parent and member must have one entry per row"));
    }
    let s = slabs(scans, centre, p, like, threads)?;
    let mids: Vec<f64> = (0..s.len()).map(|k| centre[2] + p.z_lo + (k as f64 + 0.5) * p.dz).collect();
    let (gs, secs): (Vec<Vec<Group>>, Vec<Vec<Section>>) = s.into_iter().unzip();
    let (secs, cols) = columns(secs);
    // each row's evidence in each slab it crosses
    let ev: Vec<Vec<Option<f64>>> = pool(threads)?.install(|| {
        (0..n).into_par_iter().map(|i| {
            (0..mids.len()).map(|k| row_support(&gs[k], &rows.a[i], &rows.b[i], rows.r_prior[i], mids[k], centre, like)).collect()
        }).collect()
    });
    // each row under the slabs' top: the share of its pieces the rays measure (an own circle the veto doesn't refute),
    // and whether they refute it at the narrowest it is weighed at in every slab it crosses (through air)
    let z_h = centre[2] + p.z_hi;
    let (measured, air): (Vec<f64>, Vec<bool>) = pool(threads)?.install(|| {
        (0..n).into_par_iter().map(|i| {
            let (a, b) = (rows.a[i], rows.b[i]);
            if a[2].min(b[2]) >= z_h {
                return (0.0, false);
            }
            let len = norm3(&sub(&b, &a));
            let k = (len / PIECE_LEN).ceil().max(1.0) as usize;
            let at = |f: f64| -> Point { std::array::from_fn(|d| a[d] + f * (b[d] - a[d])) };
            let own = (0..k).filter(|&j| {
                let (pa, pb) = (at(j as f64 / k as f64), at((j + 1) as f64 / k as f64));
                let s = summarise_piece(scans, &pa, &pb, crate::qsm::joint::frames(&[pa], &[pb])[0], rows.r_prior[i], like, prior.lean_sd);
                matches!(s.outcome, Outcome::Measured(c) if !refuted(tally(&s.seen, &c.x)))
            }).count();
            let thin = |g: &[Group], z: f64| {
                let dz = b[2] - a[2];
                if !(a[2].min(b[2]) <= z && z < a[2].max(b[2])) || dz.abs() < 1e-6 {
                    return None;
                }
                let f = (z - a[2]) / dz;
                let x = Vector5::new(a[0] + f * (b[0] - a[0]) - centre[0], a[1] + f * (b[1] - a[1]) - centre[1], (b[0] - a[0]) / dz, (b[1] - a[1]) / dz,
                    (NARROW[NARROW.len() - 1] * rows.r_prior[i]).max(NARROW_MIN));
                Some(refuted(tally(g, &x)))
            };
            let seen: Vec<bool> = (0..mids.len()).filter_map(|s| thin(&gs[s], mids[s])).collect();
            (own as f64 / k as f64, !seen.is_empty() && seen.iter().all(|&x| x))
        }).unzip()
    });
    let top = rows.a.iter().chain(rows.b).map(|q| q[2]).fold(centre[2], f64::max) - centre[2];
    let cap = if prior.dh.is_finite() && prior.dh > 0.0 { PTS_BELIEVE * prior.dh * top / 2.0 } else { f64::INFINITY };
    Ok(join(rows, member, centre, p, &secs, cols, context, cap, &ev, &measured, &air))
}

/// [`rebase`]'s join (module documentation), from the columns, each row's evidence per slab (`ev`), the share of it
/// the rays measure and whether they refute it at its narrowest (`air`).
#[allow(clippy::too_many_arguments)]
fn join(rows: &Rows, member: &[i64], centre: &Point, p: &BaseParams, secs: &[Section], cols: Vec<Column>, context: &[(Point, Point, f64)], cap: f64, ev: &[Vec<Option<f64>>], measured: &[f64], air: &[bool]) -> Rebased {
    use std::collections::{BTreeSet, HashMap};
    let n = rows.a.len();
    let (a, b, r, parent) = (rows.a, rows.b, rows.r_prior, rows.parent);
    let z_h = centre[2] + p.z_hi;
    let hi = |i: usize| a[i][2].max(b[i][2]);
    let nc = cols.len();
    let sec_of = |k: usize| -> Vec<usize> {
        let mut s = cols[k].sections.clone();
        s.sort_by(|x, y| secs[*x].z.total_cmp(&secs[*y].z));
        s
    };
    let col_r: Vec<f64> = (0..nc).map(|k| median(&mut cols[k].sections.iter().map(|&i| secs[i].r).collect::<Vec<_>>())).collect();
    // believable: a stem's width, not another plant's
    let in_tube = |a: &Point, b: &Point, x: &Point, rr: f64| {
        let ab = sub(b, a);
        let t = (dot(&sub(x, a), &ab) / dot(&ab, &ab).max(1e-12)).clamp(0.0, 1.0);
        let q: Point = std::array::from_fn(|i| a[i] + t * ab[i]);
        norm3(&sub(x, &q)) <= rr
    };
    let mut ok: Vec<bool> = (0..nc).map(|k| {
        let inside = cols[k].sections.iter().filter(|&&i| {
            let x = [secs[i].c[0], secs[i].c[1], secs[i].z];
            context.iter().any(|(ca, cb, cr)| in_tube(ca, cb, &x, cr + IN_COL))
        }).count();
        col_r[k] <= COL_R_MAX.min(cap) && (inside as f64) < CTX_SHARE * cols[k].sections.len() as f64
    }).collect();
    // a column standing on the ground starts within FOOT of it (forks come after their parents): one starting higher
    // would draw its base through what no section saw
    for k in 0..nc {
        let on_parent = cols[k].parent >= 0 && ok[cols[k].parent as usize];
        let foot = cols[k].sections.iter().map(|&i| secs[i].z).fold(f64::INFINITY, f64::min) - centre[2];
        ok[k] = ok[k] && (on_parent || foot <= FOOT);
    }
    // a fork of a column left out stands on the ground
    let cpar: Vec<i32> = (0..nc).map(|k| if cols[k].parent >= 0 && ok[cols[k].parent as usize] { cols[k].parent } else { -1 }).collect();
    let root_of = |mut k: usize| {
        while cpar[k] >= 0 {
            k = cpar[k] as usize;
        }
        k
    };
    let top: Vec<(Point, f64)> = (0..nc).map(|k| {
        let s = secs[*sec_of(k).last().unwrap()];
        ([s.c[0], s.c[1], s.z], col_r[k])
    }).collect();
    let row_e = |j: usize, slabs: Option<&BTreeSet<u32>>| -> f64 {
        ev[j].iter().enumerate().filter(|(k, _)| slabs.is_none_or(|s| s.contains(&(*k as u32)))).filter_map(|(_, e)| *e).sum()
    };
    // a row rising through a column's top is cut there; its chain, its ancestors under the cut
    let cut = |i: usize, z: f64| -> Point {
        let f = (z - a[i][2]) / (b[i][2] - a[i][2]);
        std::array::from_fn(|d| a[i][d] + f * (b[i][d] - a[i][d]))
    };
    let chain_of = |i: usize, z: f64| {
        let (mut c, mut j) = (Vec::new(), parent[i]);
        while j >= 0 && hi(j as usize) <= z + 1e-9 {
            c.push(j as usize);
            j = parent[j as usize];
        }
        c
    };
    // each row's fate at the columns' tops: joined to the column whose top lies nearest where it rises through that
    // height, within JOIN beyond its radius; else grafted to the nearest within GRAFT where its chain's evidence is
    // negative; else none. A row on another taken row's chain is that row's.
    #[derive(Clone, Copy, PartialEq)]
    enum Fate { Join(usize), Graft(usize), Keep }
    let col_of = |f: Fate| match f { Fate::Join(k) | Fate::Graft(k) => Some(k), Fate::Keep => None };
    let mut fate: HashMap<usize, Fate> = HashMap::new();
    let mut chain: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let near: Vec<(usize, f64)> = (0..nc).filter(|&k| ok[k] && a[i][2] <= top[k].0[2] && b[i][2] > top[k].0[2]).map(|k| {
            let c = cut(i, top[k].0[2]);
            (k, (c[0] - top[k].0[0]).hypot(c[1] - top[k].0[1]))
        }).collect();
        let best = |within: &dyn Fn(usize) -> f64| near.iter().filter(|q| q.1 <= within(q.0)).min_by(|x, y| x.1.total_cmp(&y.1)).map(|q| q.0);
        let f = match (best(&|k| top[k].1 + JOIN), best(&|_| GRAFT)) {
            (Some(k), _) => Fate::Join(k),
            (None, Some(k)) if chain_of(i, top[k].0[2]).iter().map(|&j| row_e(j, None)).sum::<f64>() < 0.0 => Fate::Graft(k),
            _ => continue,
        };
        chain.insert(i, chain_of(i, top[col_of(f).unwrap()].0[2]));
        fate.insert(i, f);
    }
    let entries: Vec<usize> = {
        let on: BTreeSet<usize> = chain.values().flatten().copied().collect();
        (0..n).filter(|i| fate.contains_key(i) && !on.contains(i)).collect()
    };
    fate.retain(|i, _| entries.contains(i));
    // the skeleton's volume above its cut each entry carries (its subtree's)
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        if parent[i] >= 0 {
            kids[parent[i] as usize].push(i);
        }
    }
    let subtree = |i: usize| {
        let (mut out, mut todo) = (Vec::new(), vec![i]);
        while let Some(j) = todo.pop() {
            out.push(j);
            todo.extend(&kids[j]);
        }
        out
    };
    let v_entry: HashMap<usize, f64> = entries.iter().map(|&i| {
        let z = top[col_of(fate[&i]).unwrap()].0[2];
        let v = subtree(i).iter().map(|&j| {
            let f = if j == i { (b[j][2] - z) / (b[j][2] - a[j][2]) } else { 1.0 };
            f * r[j] * r[j] * norm3(&sub(&b[j], &a[j]))
        }).sum::<f64>();
        (i, v)
    }).collect();
    let reaches: Vec<bool> = (0..n).map(|i| subtree(i).iter().any(|&j| hi(j) > z_h)).collect();
    // each column's holders: the members with a row on CTX_SHARE of its sections (the axis within its circle and
    // IN_COL at the section's height): their drawing stands on that wood already
    let holders: Vec<BTreeSet<i64>> = (0..nc).map(|k| {
        let on = |m: i64| cols[k].sections.iter().filter(|&&q| {
            let (c, z, rr) = (secs[q].c, secs[q].z, secs[q].r + IN_COL);
            (0..n).any(|j| member[j] == m && {
                let dz = b[j][2] - a[j][2];
                dz.abs() > 1e-9 && {
                    let f = (z - a[j][2]) / dz;
                    (0.0..1.0).contains(&f) && (a[j][0] + f * (b[j][0] - a[j][0]) - c[0]).hypot(a[j][1] + f * (b[j][1] - a[j][1]) - c[1]) <= rr
                }
            })
        }).count() as f64 >= CTX_SHARE * cols[k].sections.len() as f64;
        member.iter().copied().collect::<BTreeSet<i64>>().into_iter().filter(|&m| on(m)).collect()
    }).collect();
    // the plant's rows given each column's owner (-1: not chosen) and each entry's fate: the chosen columns' rows, by
    // owner; then the skeleton's kept rows, entries cut at their column's top; with the counts of rows dropped and
    // hung
    let assemble = |owner: &[i64], fate: &HashMap<usize, Fate>| -> Rebased {
        let chosen_secs: Vec<usize> = (0..secs.len()).filter(|&i| secs[i].column >= 0 && owner[secs[i].column as usize] >= 0).collect();
        let resected: BTreeSet<i64> = entries.iter().filter(|i| col_of(fate[i]).is_some()).map(|&i| member[i]).collect();
        let mut replaced = vec![false; n];
        for &i in &entries {
            if col_of(fate[&i]).is_some() {
                chain[&i].iter().for_each(|&j| replaced[j] = true);
            }
        }
        let inside = |j: usize| {
            let m: Point = std::array::from_fn(|d| 0.5 * (a[j][d] + b[j][d]));
            chosen_secs.iter().any(|&i| (secs[i].z - m[2]).abs() <= p.dz && (secs[i].c[0] - m[0]).hypot(secs[i].c[1] - m[1]) <= secs[i].r)
        };
        // the rows a column replaces: under its top, never the drawing above it
        let z_cut = chosen_secs.iter().map(|&i| secs[i].z + 0.5 * p.dz).fold(f64::NEG_INFINITY, f64::max).min(z_h);
        let drop: Vec<bool> = (0..n).map(|j| {
            resected.contains(&member[j]) && hi(j) <= z_cut && !fate.get(&j).is_some_and(|f| col_of(*f).is_some())
                && (replaced[j] || inside(j) || air[j])
        }).collect();
        let mut reb = Rebased { sections: secs.to_vec(), columns: cols.clone(), ..Default::default() };
        let push = |reb: &mut Rebased, x: Point, y: Point, rr: f64, par: i64, m: i64, c: i32| {
            reb.a.push(x);
            reb.b.push(y);
            reb.r_prior.push(rr);
            reb.parent.push(par);
            reb.member.push(m);
            reb.column.push(c);
            (reb.a.len() - 1) as i64
        };
        let mut sec_row: HashMap<usize, i64> = HashMap::new();
        let mut top_row: Vec<i64> = vec![-1; nc];
        for c in (0..nc).filter(|&c| owner[c] >= 0) {
            let s = sec_of(c);
            let (mut start, mut par) = if cpar[c] < 0 {
                // from the ground under it, or from its owner's own base where its owner keeps one (a stem the
                // columns missed still stands there): one base
                let own = (0..n).find(|&i| member[i] == owner[c] && parent[i] < 0 && !drop[i] && !fate.get(&i).is_some_and(|f| col_of(*f).is_some()));
                (own.map_or([secs[s[0]].c[0], secs[s[0]].c[1], centre[2]], |i| a[i]), -1)
            } else {
                let at = cols[c].at as usize;
                let same = owner[cpar[c] as usize] == owner[c];
                ([secs[at].c[0], secs[at].c[1], secs[at].z], if same { sec_row[&at] } else { -1 })
            };
            for &i in &s {
                let q = [secs[i].c[0], secs[i].c[1], secs[i].z];
                // each row's prior its own section's radius (a sliver's, none: the column's median): the median
                // overshoots a tapering stem and drew it over its neighbour's points (research 125's shadows)
                let rr = if secs[i].fitted { secs[i].r } else { col_r[c] };
                par = push(&mut reb, start, q, rr, par, owner[c], c as i32);
                sec_row.insert(i, par);
                start = q;
            }
            top_row[c] = par;
        }
        let mut new_of: Vec<i64> = vec![-1; n];
        let (mut hung, mut dropped) = (0u32, 0u32);
        for i in 0..n {
            if let Some(c) = fate.get(&i).and_then(|f| col_of(*f)) {
                let par = if owner[c] == member[i] { top_row[c] } else { -1 };
                new_of[i] = push(&mut reb, cut(i, top[c].0[2]), b[i], r[i], par, member[i], -1);
            } else if drop[i] {
                dropped += 1;
            } else {
                let par = if parent[i] < 0 {
                    Some(-1)
                } else if new_of[parent[i] as usize] >= 0 {
                    Some(new_of[parent[i] as usize])
                } else {
                    // its parent was dropped: its member's nearest row end within HANG; else a row whose subtree
                    // reaches above the slabs' top is joined by a row of its dropped parent's radius from its
                    // member's nearest row end within CONNECT (the stem RCT drew there, redrawn from the column),
                    // else starts a base of its own (and its member then keeps its rows); any other is dropped
                    let near = |within: f64| (0..reb.a.len()).filter(|&k| reb.member[k] == member[i] && reb.b[k][2] <= a[i][2] + HANG_UP).map(|k| (k, norm3(&sub(&reb.b[k], &a[i]))))
                        .filter(|q| q.1 <= within).min_by(|x, y| x.1.total_cmp(&y.1)).map(|q| q.0);
                    hung += 1;
                    match (near(HANG), reaches[i]) {
                        (Some(k), _) => Some(k as i64),
                        (None, true) => Some(near(CONNECT).map_or(-1, |k| {
                            let x = reb.b[k];
                            push(&mut reb, x, a[i], r[parent[i] as usize], k as i64, member[i], -1)
                        })),
                        (None, false) => {
                            hung -= 1;
                            None
                        }
                    }
                };
                match par {
                    Some(par) => new_of[i] = push(&mut reb, a[i], b[i], r[i], par, member[i], -1),
                    None => dropped += 1,
                }
            }
        }
        (reb.counts[4], reb.counts[5]) = (dropped, hung);
        reb
    };
    // the choice, ownership and the rows, again while a member would stand on two bases (it then keeps its rows)
    let mut reverted: BTreeSet<i64> = BTreeSet::new();
    let mut out: Vec<ColumnOut> = (0..nc).map(|k| ColumnOut { e_col: f64::NAN, e_rct: f64::NAN, measured: f64::NAN, chosen: false, owner: -1, r: col_r[k], believable: ok[k], joined: 0, grafted: 0 }).collect();
    loop {
        for (&i, f) in fate.iter_mut() {
            if reverted.contains(&member[i]) {
                *f = Fate::Keep;
            }
        }
        // per root column, its subtree's evidence against the chains it would replace, in its slabs
        let mut chosen = vec![false; nc];
        for k in (0..nc).filter(|&k| ok[k] && cpar[k] < 0) {
            let sub_cols: Vec<usize> = (0..nc).filter(|&c| ok[c] && root_of(c) == k).collect();
            let taken: Vec<usize> = entries.iter().copied().filter(|i| col_of(fate[i]).is_some_and(|c| sub_cols.contains(&c))).collect();
            let ss: Vec<usize> = sub_cols.iter().flat_map(|&c| cols[c].sections.clone()).collect();
            let slabs: BTreeSet<u32> = ss.iter().map(|&i| secs[i].slab).collect();
            let replaced: BTreeSet<usize> = taken.iter().flat_map(|i| chain[i].iter().copied()).collect();
            let e_col: f64 = ss.iter().map(|&i| secs[i].evidence).sum();
            // the chains', and the entries' own under their cuts (a column's top is its last section's slab)
            let under = |i: usize| -> f64 {
                let c = col_of(fate[&i]).unwrap();
                let last = secs[*sec_of(c).last().unwrap()].slab;
                ev[i].iter().enumerate().filter(|(k, _)| *k as u32 <= last && slabs.contains(&(*k as u32))).filter_map(|q| *q.1).sum()
            };
            let e_rct: f64 = replaced.iter().map(|&j| row_e(j, Some(&slabs))).sum::<f64>() + taken.iter().map(|&i| under(i)).sum::<f64>();
            let (lm, lt) = replaced.iter().map(|&j| {
                let l = norm3(&sub(&b[j], &a[j])) * f64::from(ev[j].iter().enumerate().any(|(k, e)| e.is_some() && slabs.contains(&(k as u32))) as u8);
                (l * measured[j], l)
            }).fold((0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
            let ms = if lt > 0.0 { lm / lt } else { 0.0 };
            // a column whose wood another member's kept drawing stands on is that member's: it isn't given to others
            let takers: BTreeSet<i64> = taken.iter().map(|&i| member[i]).collect();
            let held = sub_cols.iter().any(|&c| holders[c].iter().any(|m| !takers.contains(m)));
            let pick = !taken.is_empty() && !held && e_col > 0.0 && e_rct < p.keep * e_col && ms < MEASURED;
            for &c in &sub_cols {
                (out[c].e_col, out[c].e_rct, out[c].measured) = (e_col, e_rct, ms);
                chosen[c] = pick;
            }
        }
        for f in fate.values_mut() {
            if col_of(*f).is_some_and(|c| !chosen[c]) {
                *f = Fate::Keep;
            }
        }
        // owners: of each chosen column, the member whose entries taken by it and its forks carry the most volume
        let mut by: Vec<HashMap<i64, f64>> = vec![HashMap::new(); nc];
        for &i in &entries {
            if let Some(mut c) = col_of(fate[&i]) {
                loop {
                    *by[c].entry(member[i]).or_default() += v_entry[&i];
                    if cpar[c] < 0 {
                        break;
                    }
                    c = cpar[c] as usize;
                }
            }
        }
        let owner: Vec<i64> = (0..nc).map(|c| {
            by[c].iter().max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(x.0))).map_or(-1, |q| *q.0)
        }).collect();
        let reb = assemble(&owner, &fate);
        // a member on two bases (its rows with no parent start at more than one point) keeps its rows
        let mut starts: HashMap<i64, Vec<Point>> = HashMap::new();
        for i in (0..reb.a.len()).filter(|&i| reb.parent[i] < 0) {
            let s = starts.entry(reb.member[i]).or_default();
            if !s.contains(&reb.a[i]) {
                s.push(reb.a[i]);
            }
        }
        let two: BTreeSet<i64> = starts.into_iter().filter(|q| q.1.len() > 1).map(|q| q.0).collect();
        if two.is_subset(&reverted) {
            for &i in &entries {
                match fate[&i] {
                    Fate::Join(c) => out[c].joined += 1,
                    Fate::Graft(c) => out[c].grafted += 1,
                    Fate::Keep => {}
                }
            }
            for c in 0..nc {
                (out[c].chosen, out[c].owner) = (owner[c] >= 0, owner[c]);
            }
            let count = |f: fn(&Fate) -> bool| entries.iter().filter(|i| f(&fate[i])).count() as u32;
            let mut reb = reb;
            reb.counts[..4].copy_from_slice(&[entries.len() as u32, count(|f| matches!(f, Fate::Join(_))),
                count(|f| matches!(f, Fate::Graft(_))), count(|f| *f == Fate::Keep)]);
            reb.counts[6] = reverted.len() as u32;
            reb.out = out;
            return reb;
        }
        reverted.extend(two);
    }
}

fn norm3(v: &Point) -> f64 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A column standing at (x, y) from 0.15 to 1.45 m, radius `r`, each section's evidence `e`, numbered `k`.
    fn column(x: f64, y: f64, r: f64, e: f64, k: i32) -> Vec<Section> {
        (0..14).map(|s| Section {
            slab: s, z: 0.15 + 0.1 * s as f64, c: [x, y], r, sd_log_r: 0.05, fitted: true, arc: 30, n_scans: 3, hits: 50.0,
            n_pass: 0, pass_share: 0.0, refuted: false, edge: false, evidence: e, n_through: 100, column: k,
        }).collect()
    }

    type Plant = (Vec<Point>, Vec<Point>, Vec<f64>, Vec<i64>, Vec<i64>);

    /// Chains of points as one plant's rows, each its member's, parents first: a, b, r, parent, member.
    fn plant(members: &[(&[Point], f64)]) -> Plant {
        let (mut a, mut b, mut r, mut p, mut m) = (vec![], vec![], vec![], vec![], vec![]);
        for (k, (pts, rr)) in members.iter().enumerate() {
            for j in 1..pts.len() {
                p.push(if j == 1 { -1 } else { a.len() as i64 - 1 });
                a.push(pts[j - 1]);
                b.push(pts[j]);
                r.push(*rr);
                m.push(k as i64);
            }
        }
        (a, b, r, p, m)
    }

    /// Each row's evidence `e(row)` in each of the 14 slabs it crosses.
    fn evidence(a: &[Point], b: &[Point], e: impl Fn(usize) -> f64) -> Vec<Vec<Option<f64>>> {
        (0..a.len()).map(|i| (0..14).map(|k| {
            let z = 0.15 + 0.1 * k as f64;
            (a[i][2].min(b[i][2]) <= z && z < a[i][2].max(b[i][2])).then(|| e(i))
        }).collect()).collect()
    }

    /// [`join`] of a plant's rows about the origin, on one column of sections `secs`, each row's evidence `e(row)`.
    fn joined((a, b, r, p, m): &Plant, secs: &[Section], e: impl Fn(usize) -> f64) -> Rebased {
        let cols = vec![Column { sections: (0..secs.len()).collect(), parent: -1, at: -1 }];
        let rows = Rows { a, b, r_prior: r, r_max: &[], parent: p };
        join(&rows, m, &[0.0, 0.0, 0.0], &BaseParams::default(), secs, cols, &[], 0.5, &evidence(a, b, &e), &vec![0.0; a.len()], &(0..a.len()).map(|i| e(i) < 0.0).collect::<Vec<_>>())
    }

    /// Every member hangs from one base, parents first.
    fn one_base(f: &Rebased) {
        for i in 0..f.a.len() {
            assert!(f.parent[i] < i as i64 && (f.parent[i] < 0 || f.member[f.parent[i] as usize] == f.member[i]));
        }
        for m in f.member.iter() {
            let roots: Vec<Point> = (0..f.a.len()).filter(|&i| f.member[i] == *m && f.parent[i] < 0).map(|i| f.a[i]).collect();
            assert!(roots.iter().all(|q| *q == roots[0]), "{roots:?}");
        }
    }

    #[test]
    fn a_fan_through_air_hangs_from_the_column_its_stems_stand_on() {
        // three RCT trees from one root 40 cm off a 10 cm trunk, their axes through air (each row's evidence
        // negative) onto it above 1.5 m; the biggest crown is the second's
        let root = [0.4, -0.3, 0.0];
        let f0: &[Point] = &[root, [0.25, -0.2, 0.8], [0.02, 0.0, 1.6], [0.0, 0.0, 2.5]];
        let f1: &[Point] = &[root, [0.3, 0.1, 0.9], [0.05, 0.05, 1.7], [0.05, 0.1, 4.0]];
        let f2: &[Point] = &[root, [0.1, -0.4, 0.7], [-0.05, -0.08, 1.55], [-0.1, -0.1, 2.3]];
        let f = joined(&plant(&[(f0, 0.05), (f1, 0.05), (f2, 0.03)]), &column(0.0, 0.0, 0.1, 100.0, 0), |_| -20.0);
        assert!(f.out[0].chosen && f.out[0].owner == 1 && f.out[0].joined == 3, "{:?}", f.out);
        // the column's rows are the owner's, from the ground; the others start at their cuts at its top, 1.45 m
        let base = |m: i64| (0..f.a.len()).find(|&i| f.member[i] == m && f.parent[i] < 0).map(|i| f.a[i]).unwrap();
        assert_eq!(base(1), [0.0, 0.0, 0.0]);
        assert!((base(0)[2] - 1.45).abs() < 1e-9 && (base(2)[2] - 1.45).abs() < 1e-9);
        // nothing of the fan is left under the column's top but the column
        assert!((0..f.a.len()).all(|i| f.a[i][2].max(f.b[i][2]) > 1.45 - 1e-9 || f.column[i] == 0));
        assert_eq!(f.counts[6], 0);
        one_base(&f);
    }

    #[test]
    fn a_skeleton_on_its_wood_keeps_its_base_and_thin_stems_keep_theirs() {
        // a stem RCT drew on its wood: its rows hold all the column's evidence
        let z: Vec<Point> = (0..6).map(|k| [0.0, 0.0, 0.5 * k as f64]).collect();
        let f = joined(&plant(&[(&z, 0.1)]), &column(0.0, 0.0, 0.1, 100.0, 0), |_| 100.0);
        assert!(!f.out[0].chosen && f.counts == [1, 0, 0, 1, 0, 0, 0], "{:?} {:?}", f.out, f.counts);
        assert!(f.column.iter().all(|&c| c < 0) && f.a.len() == 5);
        // seven thin stems read as one 5 cm column: their own rows hold most of its evidence
        let stems: Vec<Vec<Point>> = (0..7).map(|k| {
            let t = std::f64::consts::TAU * k as f64 / 7.0;
            (0..5).map(|j| [0.05 * t.cos(), 0.05 * t.sin(), 0.5 * j as f64]).collect()
        }).collect();
        let ms: Vec<(&[Point], f64)> = stems.iter().map(|s| (&s[..], 0.01)).collect();
        let f = joined(&plant(&ms), &column(0.0, 0.0, 0.05, 100.0, 0), |_| 15.0);
        assert!(!f.out[0].chosen && f.counts[3] == 7 && f.a.len() == 28, "{:?} {:?}", f.out, f.counts);
    }

    #[test]
    fn a_stem_beside_the_columns_keeps_its_chain_where_the_rays_see_it() {
        // a trunk RCT drew 15 cm beside its wood (no evidence), and a thin stem 30 cm off that the columns missed,
        // its rows on wood (positive evidence): the trunk is re-sectioned, the thin stem (no entry: it rises through
        // the column's top 30 cm off it) isn't grafted
        let trunk: Vec<Point> = (0..6).map(|k| [0.0, if k < 4 { 0.15 } else { 0.0 }, 0.5 * k as f64]).collect();
        let thin: Vec<Point> = (0..6).map(|k| [0.3, 0.0, 0.5 * k as f64]).collect();
        let p = plant(&[(&trunk, 0.06), (&thin, 0.01)]);
        let f = joined(&p, &column(0.0, 0.0, 0.1, 100.0, 0), |i| if i < 5 { 0.0 } else { 5.0 });
        assert!(f.out[0].chosen && f.out[0].owner == 0 && f.counts[..4] == [1, 1, 0, 0], "{:?} {:?}", f.out, f.counts);
        assert_eq!((0..f.a.len()).filter(|&i| f.member[i] == 1).count(), 5);
        one_base(&f);
        // the same stem's rows through air are grafted to the trunk's column: a fan's
        let f = joined(&p, &column(0.0, 0.0, 0.1, 100.0, 0), |i| if i < 5 { 0.0 } else { -5.0 });
        assert!(f.counts[1..4] == [1, 1, 0] && f.out[0].owner == 0, "{:?}", f.counts);
        one_base(&f);
    }

    /// One RCT tree whose trunk (from `root` to `fork` at 1 m) forks into a stem rising onto the column at the
    /// origin (through `via`) and a stem the columns missed, to 50 cm beyond the fork at 2 m; every row without
    /// evidence.
    fn forked(root: Point, fork: Point, via: Point) -> Rebased {
        let (mut a, mut b, mut r, mut p, mut m) = plant(&[(&[root, fork, via, [0.0, 0.0, 2.5]], 0.05)]);
        a.push(fork);
        b.push([fork[0] + 0.5, 0.0, 2.0]);
        r.push(0.02);
        p.push(0);
        m.push(0);
        joined(&(a, b, r, p, m), &column(0.0, 0.0, 0.08, 100.0, 0), |_| 0.0)
    }

    #[test]
    fn a_stem_forking_under_the_top_hangs_from_the_column_or_its_member_keeps_its_rows() {
        // the fork 20 cm from the column: the trunk is the column's, the missed stem hangs from it
        let f = forked([0.0, 0.2, 0.0], [0.0, 0.2, 1.0], [0.02, 0.0, 1.6]);
        assert!(f.out[0].chosen && f.counts == [1, 1, 0, 0, 1, 1, 0], "{:?} {:?}", f.out, f.counts);
        one_base(&f);
        // 60 cm off it: joined to the column's nearest row end by a row of the trunk's radius
        let f = forked([0.6, 0.0, 0.0], [0.6, 0.0, 1.0], [0.02, 0.0, 1.47]);
        assert!(f.out[0].chosen && f.counts[5] == 1 && f.a.len() == 14 + 4, "{:?} {:?}", f.out, f.counts);
        one_base(&f);
        // 1.2 m off it: the missed stem can't be joined to it, so its tree would stand on two bases, and is left as
        // it was
        let f = forked([1.2, 0.0, 0.0], [1.2, 0.0, 1.0], [0.02, 0.0, 1.46]);
        assert!(!f.out[0].chosen && f.counts[6] == 1 && f.a.len() == 4 && f.a[0] == [1.2, 0.0, 0.0], "{:?} {:?}", f.out, f.counts);
    }

    #[test]
    fn a_base_the_rays_measure_keeps_its_rows_however_strong_the_column() {
        // the trunk drawn 15 cm beside a column, its rows holding none of its evidence, but measured by their own
        // circles: kept (research 125's F20)
        let trunk: Vec<Point> = (0..6).map(|k| [0.0, if k < 4 { 0.15 } else { 0.0 }, 0.5 * k as f64]).collect();
        let (a, b, r, p, m) = plant(&[(&trunk, 0.06)]);
        let rows = Rows { a: &a, b: &b, r_prior: &r, r_max: &[], parent: &p };
        let cols = vec![Column { sections: (0..14).collect(), parent: -1, at: -1 }];
        let ev = evidence(&a, &b, |_| 0.0);
        let f = join(&rows, &m, &[0.0, 0.0, 0.0], &BaseParams::default(), &column(0.0, 0.0, 0.1, 100.0, 0), cols.clone(), &[], 0.5, &ev, &vec![1.0; a.len()], &vec![false; a.len()]);
        assert!(!f.out[0].chosen && f.out[0].measured == 1.0 && f.a == a, "{:?}", f.out);
        // unmeasured: re-sectioned
        let f = join(&rows, &m, &[0.0, 0.0, 0.0], &BaseParams::default(), &column(0.0, 0.0, 0.1, 100.0, 0), cols, &[], 0.5, &ev, &vec![0.0; a.len()], &vec![false; a.len()]);
        assert!(f.out[0].chosen, "{:?}", f.out);
    }

    #[test]
    fn a_column_another_member_already_stands_on_is_not_given_to_a_fan() {
        // RCT tree 1 drawn on the column's wood up to 1 m (its rows supported), then leaning away; tree 0 a fan's axis
        // through air onto the column's top: the column is tree 1's wood, so the fan doesn't take it (research 125's
        // shadows: a column drawn along a stem a neighbour already stands on)
        let fan: &[Point] = &[[0.4, -0.3, 0.0], [0.25, -0.2, 0.8], [0.02, 0.0, 1.6], [0.0, 0.0, 2.5]];
        let stem: &[Point] = &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.6, 0.0, 2.0]];
        let f = joined(&plant(&[(fan, 0.05), (stem, 0.05)]), &column(0.0, 0.0, 0.08, 100.0, 0), |i| if i < 3 { -20.0 } else { 20.0 });
        assert!(!f.out[0].chosen, "{:?} {:?}", f.out, f.counts);
    }

    #[test]
    fn a_short_arc_leaves_a_wider_radius_than_a_long_one() {
        let arc = |a1: f64| -> Vec<[f64; 2]> {
            (0..60).map(|i| {
                let t = (a1 * i as f64 / 59.0).to_radians();
                let r = 0.1 + 0.002 * ((i * 7 % 5) as f64 - 2.0) / 2.0;
                [r * t.cos(), r * t.sin()]
            }).collect()
        };
        let (_, short, _) = circle_of(&arc(70.0), 0.12).unwrap();
        let (_, long, _) = circle_of(&arc(200.0), 0.2).unwrap();
        assert!(short > long && long >= PTS_SD, "{short} {long}");
    }
}
