// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The depth pyramid (3dtrees research 130 slice-pyramid; formulation.md "The depth pyramid as a renormalisation at
//! plot scale"): one scan's pulses aggregated over its image's lattice, level by level, built once per image.
//!
//! - **What a tile holds:** its first echoes' least and greatest range (`near`, `far`), and counts of its pulses,
//!   misses and echoes, and its energy (the share of each pulse's beam its echoes took: one less the beam it leaves,
//!   [`super::foliage::beam_left`]). A parent's are its children's mins, maxes and sums, exactly. Levels from 4 × 4
//!   cells up are kept; a cell is read off the image.
//! - **What it answers** ([`Pyramid::count`]): how many pulses over a rectangle of cells satisfy a test on their first
//!   echo, descending only into tiles the aggregates can't settle (a tile wholly inside the rectangle whose `near` and
//!   `far` decide it is counted whole; one they rule out is skipped). So a query costs the rectangle's edge, not its
//!   area, and is exact for the rectangle.
//! - **Carving** ([`carve`], the wood map's bounds on hidden wood): a voxel's cover is the rectangle of cells its
//!   corners project to, one cell wider. Per scan, the pulses in it, those clear past its far side (a first echo
//!   beyond its farthest corner by more than the receiver's resolution, or none: [`Settings::free_to`]), and those
//!   that end in it or within the resolution behind it. A voxel is carved where the cover holds enough pulses and all
//!   are clear: never where a pulse that crosses it shows a surface (every such pulse is in the cover), so the bound
//!   is exact; the cover's corners hold pulses that miss the voxel, so it carves less than a walk of each ray would.
//! - **Arrival** ([`arrival`]): the share of a cover's pulses that reach a range, for detectability's echo-or-none
//!   part read off the scan instead of the field.
//! - **The level information supports:** the cover's rectangle is read at the coarsest tiles that fit wholly inside
//!   it, so a voxel's query reads its projection's scale, and a bound is reported per voxel at the scan's own
//!   resolution there.

use crate::Point;
use crate::transform::{norm, sub};
use rayon::prelude::*;
use super::foliage::beam_left;
use super::range_image::{Cell, RangeImage};
use super::settings::Settings;

/// The finest level kept: tiles of 2^FIRST cells a side.
const FIRST: usize = 2;

/// How many of a level's tiles a cover's narrower side spans at least where it is read at that level ([`supported`]).
const SUPPORT: usize = 8;

/// One tile's aggregates (the module's docs); `near` is +∞ and `far` −∞ where it holds no echo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    pub near: f32,
    pub far: f32,
    pub pulses: u32,
    pub misses: u32,
    pub echoes: u32,
    pub energy: f32,
}

impl Tile {
    const EMPTY: Tile = Tile { near: f32::INFINITY, far: f32::NEG_INFINITY, pulses: 0, misses: 0, echoes: 0, energy: 0.0 };

    fn merge(self, o: Tile) -> Tile {
        Tile { near: self.near.min(o.near), far: self.far.max(o.far), pulses: self.pulses + o.pulses, misses: self.misses + o.misses,
               echoes: self.echoes + o.echoes, energy: self.energy + o.energy }
    }
}

/// A rectangle of cells, inclusive: lines `l0..=l1`, columns `c0..=c1`.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub l0: usize,
    pub l1: usize,
    pub c0: usize,
    pub c1: usize,
}

/// One scan's depth pyramid (the module's docs): per kept level, its tiles row-major.
pub struct Pyramid {
    em: Settings,
    levels: Vec<(usize, usize, Vec<Tile>)>,
}

/// Cell `k` of image `v` as a tile: its first echo, whether it has a pulse and missed, its echoes, and the share of the
/// beam they took ([`beam_left`], as the field reads it).
pub fn cell(v: &RangeImage, em: &Settings, k: usize) -> Tile {
    if v.cell[k] == Cell::NoPulse {
        return Tile::EMPTY;
    }
    let r = if v.cell[k] == Cell::Echoes { v.range[k] } else { f32::NAN };
    Tile { near: if r.is_nan() { f32::INFINITY } else { r }, far: if r.is_nan() { f32::NEG_INFINITY } else { r }, pulses: 1,
           misses: (v.cell[k] == Cell::Miss) as u32, echoes: v.echoes(k).len() as u32, energy: (1.0 - beam_left(v, em, k)) as f32 }
}

/// A level's tile (`tl`, `tc`) as a cell rectangle, clipped to the image.
fn span(v: &RangeImage, s: usize, tl: usize, tc: usize) -> Rect {
    Rect { l0: tl << s, l1: (((tl + 1) << s) - 1).min(v.n_lines - 1), c0: tc << s, c1: (((tc + 1) << s) - 1).min(v.n_cols - 1) }
}

impl Pyramid {
    /// The pyramid of image `v` (the beam its echoes leave read as the field reads it, `em`).
    pub fn new(v: &RangeImage, em: &Settings) -> Pyramid {
        let mut levels: Vec<(usize, usize, Vec<Tile>)> = Vec::new();
        let mut s = FIRST;
        loop {
            let (nl, nc) = (v.n_lines.div_ceil(1 << s), v.n_cols.div_ceil(1 << s));
            let tiles: Vec<Tile> = (0..nl * nc).into_par_iter().map(|i| {
                let (tl, tc) = (i / nc, i % nc);
                match levels.last() {
                    None => {
                        let r = span(v, s, tl, tc);
                        (r.l0..=r.l1).flat_map(|l| (r.c0..=r.c1).map(move |c| l * v.n_cols + c)).fold(Tile::EMPTY, |t, k| t.merge(cell(v, em, k)))
                    }
                    Some((pl, pc, prev)) => [(0, 0), (0, 1), (1, 0), (1, 1)].iter()
                        .filter_map(|&(a, b)| (2 * tl + a < *pl && 2 * tc + b < *pc).then(|| prev[(2 * tl + a) * pc + 2 * tc + b]))
                        .fold(Tile::EMPTY, Tile::merge),
                }
            }).collect();
            levels.push((nl, nc, tiles));
            if nl * nc <= 64 {
                break;
            }
            s += 1;
        }
        Pyramid { em: *em, levels }
    }

    /// The tile at level `s` (cells at 0; kept levels from [`FIRST`]).
    fn tile(&self, v: &RangeImage, s: usize, tl: usize, tc: usize) -> Tile {
        if s == 0 {
            return cell(v, &self.em, tl * v.n_cols + tc);
        }
        let (_, nc, tiles) = &self.levels[s - FIRST];
        tiles[tl * nc + tc]
    }

    /// How many of `rect`'s pulses pass a test (the module's docs): `decide(tile, whole)` gives a tile's count where its
    /// aggregates settle it (`whole`: the tile lies inside the rectangle; a count of 0 settles a part too), or none
    /// to look closer; a single cell is always settled.
    pub fn count(&self, v: &RangeImage, rect: Rect, decide: &dyn Fn(&Tile, bool) -> Option<u64>) -> u64 {
        let top = FIRST + self.levels.len() - 1;
        let (nl, nc, _) = self.levels[top - FIRST];
        let mut n = 0;
        for tl in (rect.l0 >> top)..=(rect.l1 >> top).min(nl - 1) {
            for tc in (rect.c0 >> top)..=(rect.c1 >> top).min(nc - 1) {
                n += self.visit(v, top, tl, tc, rect, decide);
            }
        }
        n
    }

    fn visit(&self, v: &RangeImage, s: usize, tl: usize, tc: usize, rect: Rect, decide: &dyn Fn(&Tile, bool) -> Option<u64>) -> u64 {
        let r = span(v, s, tl, tc);
        if r.l0 > rect.l1 || r.l1 < rect.l0 || r.c0 > rect.c1 || r.c1 < rect.c0 {
            return 0;
        }
        let whole = r.l0 >= rect.l0 && r.l1 <= rect.l1 && r.c0 >= rect.c0 && r.c1 <= rect.c1;
        if let Some(n) = decide(&self.tile(v, s, tl, tc), whole) {
            if whole || n == 0 {
                return n;
            }
        }
        if s == FIRST {
            let mut n = 0;
            for l in r.l0.max(rect.l0)..=r.l1.min(rect.l1) {
                for c in r.c0.max(rect.c0)..=r.c1.min(rect.c1) {
                    n += decide(&cell(v, &self.em, l * v.n_cols + c), true).expect("a cell is settled");
                }
            }
            return n;
        }
        let (nl, nc, _) = self.levels[s - 1 - FIRST];
        [(0, 0), (0, 1), (1, 0), (1, 1)].iter().filter(|&&(a, b)| 2 * tl + a < nl && 2 * tc + b < nc)
            .map(|&(a, b)| self.visit(v, s - 1, 2 * tl + a, 2 * tc + b, rect, decide)).sum()
    }

    /// The pulses in `rect`.
    pub fn pulses(&self, v: &RangeImage, rect: Rect) -> u64 {
        self.count(v, rect, &|t, whole| (whole || t.pulses == 0).then_some(t.pulses as u64))
    }

    /// The pulses in `rect` whose first echo lies beyond range `r`, or that have none.
    pub fn beyond(&self, v: &RangeImage, rect: Rect, r: f64) -> u64 {
        let r = r as f32;
        self.count(v, rect, &|t, whole| {
            if t.pulses == 0 {
                Some(0)
            } else if t.near > r {
                whole.then_some(t.pulses as u64)
            } else if t.far <= r {
                if t.misses == 0 { Some(0) } else { whole.then_some(t.misses as u64) }
            } else {
                None
            }
        })
    }

    /// The pulses in `rect` whose first echo lies within ranges `a` to `b`.
    pub fn within(&self, v: &RangeImage, rect: Rect, a: f64, b: f64) -> u64 {
        let (a, b) = (a as f32, b as f32);
        self.count(v, rect, &|t, whole| {
            if t.far < a || t.near > b {
                Some(0)
            } else if t.near >= a && t.far <= b {
                whole.then_some((t.pulses - t.misses) as u64)
            } else {
                None
            }
        })
    }

    /// Its kept levels' shapes (lines, columns of tiles), finest first.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        self.levels.iter().map(|l| (l.0, l.1)).collect()
    }

    /// The whole image's aggregates.
    pub fn total(&self) -> Tile {
        self.levels.last().map_or(Tile::EMPTY, |l| l.2.iter().fold(Tile::EMPTY, |t, x| t.merge(*x)))
    }

    /// The tiles of the kept level `s` (from [`FIRST`]), row-major, and its shape (for tests and reports).
    pub fn level(&self, s: usize) -> (usize, usize, &[Tile]) {
        let (nl, nc, t) = &self.levels[s - FIRST];
        (*nl, *nc, t)
    }
}

/// The cover of the points `ps` in image `v` (the module's docs): the rectangle of cells they project to, `pad`
/// cells wider; none where one falls off the scanner's lattice or the rectangle spans half a turn (it wraps).
pub fn cover(v: &RangeImage, ps: &[Point], pad: f64) -> Option<Rect> {
    let (mut l0, mut l1, mut c0, mut c1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in ps {
        let (l, c, _) = v.cell_of(p)?;
        (l0, l1, c0, c1) = (l0.min(l), l1.max(l), c0.min(c), c1.max(c));
    }
    if l1 - l0 > 180.0 / v.lat.azimuth_step.abs() {
        return None;
    }
    let clip = |x: f64, n: usize| x.round().clamp(0.0, (n - 1) as f64) as usize;
    Some(Rect { l0: clip(l0 - pad, v.n_lines), l1: clip(l1 + pad, v.n_lines), c0: clip(c0 - pad, v.n_cols), c1: clip(c1 + pad, v.n_cols) })
}

/// A cover read at the level its size supports (the module's docs): its edges moved out to that level's tiles, the
/// coarsest whose tiles fit SUPPORT times across its narrower side, so every tile it reads is whole (none finer than
/// [`FIRST`]: a cover under that is read cell by cell). A wider cover, so still a bound.
pub fn supported(v: &RangeImage, r: Rect) -> Rect {
    let side = (r.l1 - r.l0 + 1).min(r.c1 - r.c0 + 1) / SUPPORT;
    let s = (usize::BITS - side.leading_zeros()).saturating_sub(1) as usize;
    if s < FIRST {
        return r;
    }
    let m = (1usize << s) - 1;
    Rect { l0: r.l0 & !m, l1: (r.l1 | m).min(v.n_lines - 1), c0: r.c0 & !m, c1: (r.c1 | m).min(v.n_cols - 1) }
}

/// Per voxel of the box `lo`-`hi` (`voxel` m cubes, x slowest: `(i · ny + j) · nz + l`), in one scan: the pulses in
/// its cover, those clear past its far side, and those whose first echo lies in it or within the receiver's
/// resolution behind it (the module's docs). A voxel the scan can't cover (behind its lattice) has none.
pub fn carve(v: &RangeImage, pyr: &Pyramid, lo: Point, hi: Point, voxel: f64, em: &Settings) -> [Vec<u32>; 3] {
    let n = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / voxel).ceil() as usize);
    let res = em.resolution();
    let out: Vec<[u32; 3]> = (0..n[0] * n[1] * n[2]).into_par_iter().map(|i| {
        let k = [i / (n[1] * n[2]), (i / n[2]) % n[1], i % n[2]];
        let corner = |m: usize| [0, 1, 2].map(|a| lo[a] + (k[a] + (m >> a & 1)) as f64 * voxel);
        let corners: Vec<Point> = (0..8).map(corner).collect();
        let far = corners.iter().map(|c| norm(&sub(c, &v.origin))).fold(0.0, f64::max);
        let near = norm(&[0, 1, 2].map(|a| v.origin[a].clamp(corners[0][a], corners[7][a]) - v.origin[a]));
        if near <= 0.0 {
            return [0; 3];
        }
        let Some(rect) = cover(v, &corners, 1.0).map(|r| supported(v, r)) else { return [0; 3] };
        [pyr.pulses(v, rect), pyr.beyond(v, rect, far + res), pyr.within(v, rect, near, far + res)].map(|x| x as u32)
    }).collect();
    [0, 1, 2].map(|j| out.iter().map(|x| x[j]).collect())
}

/// Per stick (a, b, r): the pulses in its cover (its ends' rectangle, wider by its radius and a cell) and those that
/// reach its middle's range (a first echo beyond it, or none): the share of the scan's pulses that arrive there.
pub fn arrival(v: &RangeImage, pyr: &Pyramid, sticks: &[(Point, Point, f64)]) -> Vec<(u64, u64)> {
    sticks.par_iter().map(|&(a, b, r)| {
        let mid = [0, 1, 2].map(|j| 0.5 * (a[j] + b[j]));
        let t = norm(&sub(&mid, &v.origin));
        let pad = 1.0 + r / (t * v.step());
        match cover(v, &[a, b], pad).map(|r| supported(v, r)) {
            Some(rect) => (pyr.pulses(v, rect), pyr.beyond(v, rect, t)),
            None => (0, 0),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::foliage::{FoliageBox, Walk};
    use super::super::fixtures::pole_image;

    /// Each kept level's tile is its children's merge (its finest kept level's, its cells'), exactly, and the top's
    /// totals are the image's.
    #[test]
    fn a_levels_aggregates_are_its_childrens() {
        let (v, em) = (pole_image(), Settings::default());
        let pyr = Pyramid::new(&v, &em);
        for s in FIRST..FIRST + pyr.levels.len() {
            let (nl, nc, tiles) = pyr.level(s);
            for tl in 0..nl {
                for tc in 0..nc {
                    let kids = if s == FIRST {
                        let r = span(&v, s, tl, tc);
                        (r.l0..=r.l1).flat_map(|l| (r.c0..=r.c1).map(move |c| (l, c))).fold(Tile::EMPTY, |t, (l, c)| t.merge(pyr.tile(&v, 0, l, c)))
                    } else {
                        let (cl, cc, _) = pyr.level(s - 1);
                        [(0, 0), (0, 1), (1, 0), (1, 1)].iter().filter(|&&(a, b)| 2 * tl + a < cl && 2 * tc + b < cc)
                            .fold(Tile::EMPTY, |t, &(a, b)| t.merge(pyr.tile(&v, s - 1, 2 * tl + a, 2 * tc + b)))
                    };
                    assert_eq!(tiles[tl * nc + tc], kids, "level {s} tile {tl} {tc}");
                }
            }
        }
        let (_, _, top) = pyr.level(FIRST + pyr.levels.len() - 1);
        let all = top.iter().fold(Tile::EMPTY, |t, x| t.merge(*x));
        let n = |c: Cell| v.cell.iter().filter(|&&x| x == c).count() as u32;
        assert_eq!((all.pulses, all.misses, all.echoes), (n(Cell::Echoes) + n(Cell::Miss), n(Cell::Miss), v.echo_range.len() as u32));
    }

    /// A query over a rectangle counts what a loop over its cells does, whatever the rectangle.
    #[test]
    fn a_query_is_its_cells_count() {
        let (v, em) = (pole_image(), Settings::default());
        let pyr = Pyramid::new(&v, &em);
        let mut seed = 7u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for _ in 0..200 {
            let (a, b, c, d) = (next(v.n_lines), next(v.n_lines), next(v.n_cols), next(v.n_cols));
            let rect = Rect { l0: a.min(b), l1: a.max(b), c0: c.min(d), c1: c.max(d) };
            let (r, lo, hi) = (3.0 + next(60) as f64 * 0.1, 3.9, 4.1 + next(5) as f64);
            let cells = || (rect.l0..=rect.l1).flat_map(|l| (rect.c0..=rect.c1).map(move |c| l * v.n_cols + c));
            let first = |k: usize| v.range[k] as f64;
            assert_eq!(pyr.pulses(&v, rect), cells().filter(|&k| v.cell[k] != Cell::NoPulse).count() as u64);
            assert_eq!(pyr.beyond(&v, rect, r), cells().filter(|&k| v.cell[k] == Cell::Miss || (v.cell[k] == Cell::Echoes && first(k) as f32 > r as f32)).count() as u64);
            assert_eq!(pyr.within(&v, rect, lo, hi), cells().filter(|&k| v.cell[k] == Cell::Echoes && (first(k) as f32) >= lo as f32 && (first(k) as f32) <= hi as f32).count() as u64);
        }
    }

    /// Carving is a bound (the module's docs): every ray that crosses a carved voxel leaves it before its clear range
    /// (the rule slice 7's free space carved by, each ray walked), and where that walk carves with room to spare the
    /// cover does too: in front of the pole the voxels are carved, within the receiver's resolution in front of it
    /// they are open with pulses ending there, and behind it they are open.
    #[test]
    fn carving_bounds_every_ray_that_crosses() {
        let (v, em) = (pole_image(), Settings::default());
        let pyr = Pyramid::new(&v, &em);
        let (lo, hi, voxel) = ([-0.1, 1.0, -0.1], [0.1, 8.6, 0.1], 0.2);
        let [pulses, clear, ending] = carve(&v, &pyr, lo, hi, voxel, &em);
        assert_eq!(pulses.len(), 38);
        let carved: Vec<bool> = (0..38).map(|i| pulses[i] >= 8 && clear[i] == pulses[i]).collect();
        let sp = FoliageBox { lo, hi, voxel };
        for k in (0..v.cell.len()).filter(|&k| v.cell[k] != Cell::NoPulse) {
            let w = v.direction((k / v.n_cols) as f64, (k % v.n_cols) as f64);
            let free = if v.cell[k] == Cell::Echoes { em.free_to(v.range[k] as f64) } else { f64::INFINITY };
            for &(key, _, b) in Walk::new(&sp, &v.origin, &w).segments() {
                assert!(!carved[key[1] as usize] || b <= free, "cell {k} voxel {key:?}");
            }
        }
        let at = |y: f64| ((y - 1.0) / 0.2).floor() as usize;
        assert!(carved[at(2.1)], "{} {}", pulses[at(2.1)], clear[at(2.1)]);
        let front = at(4.0 - 0.03 - 0.5 * em.resolution());
        assert!(!carved[front] && ending[front] > 0);
        let behind = at(5.1);
        assert!(!carved[behind] && clear[behind] < pulses[behind]);
    }
}
