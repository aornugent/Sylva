// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A plant's base re-sectioned from the rays (3dtrees ticket 125, a prototype): the columns of wood standing in its
//! bottom metre and a half, found in every scan's pulses at once, for a caller to build the base's rows from and
//! measure with [`super::support_tree`].
//!
//! - **Slabs.** From [`BaseParams::z_lo`] to [`BaseParams::z_hi`] above the ground, slabs [`BaseParams::dz`] thick,
//!   each a vertical piece about the plant's base whose tube is [`BaseParams::reach`] wide: each scan's pulses
//!   through it, as [`super::groups`] takes a piece's.
//! - **Sections.** A slab's woody stops (echo character over a half, [`super::p_wood`] for wood of [`WOOD_R`]), from
//!   every scan, linked where they lie within [`BaseParams::link`] of each other, are its sections: a stem is a
//!   connected arc or ring of stops, a grass tussock or leaf clump a scatter the veto refutes. Each section with
//!   [`super::RUN_HITS`] woody hits gets a circle: RANSAC and least squares on its stops ([`super::point_circle`]'s
//!   band), its arc's covariance ([`super::arc_cov`]) for its sd; a section whose stops make no circle (a sliver one
//!   scan grazes) is centred behind its stops by half their width. The ray veto ([`super::tally`]) audits each
//!   circle against every scan's passes.
//! - **Columns.** Sections are linked upward, slab by slab (a slab's gap bridged), each to the column whose last
//!   section's centre lies nearest, within half the wider radius and [`LINK_GAP`]; where two sections claim one
//!   column the nearer continues it and the other starts a fork of it. A column of fewer than [`MIN_SECTIONS`]
//!   sections and no fork is dropped (grass, a stray stop).
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
}

impl Default for BaseParams {
    fn default() -> Self {
        BaseParams { z_lo: 0.1, z_hi: 1.5, dz: 0.1, reach: 0.6, link: 0.02 }
    }
}

/// One slab's section: its slab and height (world), centre (world x, y), radius and sd of log R (NaN where no
/// circle was fitted: `fitted`), the ten-degree bins its stops fill, its scans and woody hits, the veto's audit
/// (passes through its circle, their share, `refuted`), whether it reaches the tube's edge, and its column (-1:
/// none).
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
        let fit = circle_of(&xy, extent);
        let (x, sd, arc, fitted) = match fit {
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

/// The columns of wood in a plant's base about `centre` (world x, y and the ground's z): every slab's sections and
/// the columns linking them (module documentation).
pub fn base_columns(scans: &[&ScanIndex], centre: &Point, p: &BaseParams, like: &JointParams, threads: usize) -> Result<(Vec<Section>, Vec<Column>)> {
    if !(p.dz > 0.0 && p.z_hi > p.z_lo && p.reach > 0.0 && p.link > 0.0) {
        return Err(Error::invalid("base columns: dz, reach and link must be positive and z_hi above z_lo"));
    }
    let n = ((p.z_hi - p.z_lo) / p.dz).round() as usize;
    let frame: Frame = ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], p.dz);
    let slabs: Vec<Vec<Section>> = pool(threads)?.install(|| {
        (0..n).into_par_iter().map(|k| {
            let z = centre[2] + p.z_lo + k as f64 * p.dz;
            let (a, b) = ([centre[0], centre[1], z], [centre[0], centre[1], z + p.dz]);
            let gs = groups(scans, &a, &b, frame, p.reach, like);
            sections(&gs, k as u32, z + 0.5 * p.dz, centre, p, like)
        }).collect()
    });
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
    for k in 0..n as u32 {
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
    Ok((secs, out))
}

#[cfg(test)]
mod tests {
    use super::*;

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
