// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Tests of the surfel measure's identities (3dtrees research 128,
//! `identities.md`; a spike, not a fit).
//!
//! - **line-complex axis** (Identity 3): every surfel's normal line
//!   `(n, p × n)` meets the axis `(u, m)`, `n·m + u·(p × n) = 0`, and on a
//!   cylinder at right angles, `n·u = 0` (weighted by a length `s`, so both
//!   residuals are distances). The axis is the smallest eigenvector of the
//!   additive 6×6 `Σ a aᵀ` under `|u| = 1`: `m` is eliminated (a Schur
//!   complement), `u` is the 3×3's smallest eigenvector, and `m` is made
//!   perpendicular to `u` (the Klein quadric). Without `n·u = 0` it is the
//!   axis of any surface of revolution (a cone, a flare), whose taper is the
//!   normals' mean tilt along it. Offsetting a surfel along its normal leaves
//!   its normal line, and so the axis, unchanged (Identity 4).
//! - **gradient normals** (Identity 2): `n ∝ ω − ∇ ln r` on the sphere of
//!   directions, the gradient a least-squares fit of `ln r` over a window of
//!   lattice cells; `cos ι = 1 / √(1 + |∇ ln r|²)`.
//! - **ridges:** the lattice cells with a first echo but no surfel, joined
//!   into 8-connected components across range steps under `jump`, with each
//!   component's extent in cells and in 3D, straightness, direction, and the
//!   plane through the scanner that contains it (Identity 1 for thin wood).
//! - **support function** (Identity 5): `h(φ) = (p − c)·n⊥` about the axis,
//!   fitted to its harmonics up to k = 2.

use super::lattice::Lattice;
use super::opacity::ShadowMap;
use super::surfels::{collapse, eig, mad, pt, v, Region, SurfelGrid, V};
use crate::Point;
use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen};

/// A fitted axis line: direction (up), the point on it nearest the region's centre, the mean squared residual (m²)
/// and the next eigenvalue's, the median distance of the surfels from it (m), the mean `n·u`, and the surfels used.
#[derive(Debug, Clone, Copy)]
pub struct LineFit {
    pub u: V,
    pub c: V,
    pub lam: f64,
    pub lam2: f64,
    pub r: f64,
    pub nu: f64,
    pub used: usize,
}

/// The line-complex axis of the surfels `idx` (positions `p`, normals `n`), coordinates taken about `c0`. With
/// `s_perp` Some(s) every surfel also asks `s n·u = 0` (a cylinder); without, only that its normal line meets the
/// axis (a surface of revolution). One refit without residuals beyond 3 MAD.
pub fn line_complex(p: &[V], n: &[V], idx: &[usize], c0: &V, s_perp: Option<f64>) -> Option<LineFit> {
    let solve = |ids: &[usize]| -> Option<(V, V, f64, f64)> {
        if ids.len() < 6 {
            return None;
        }
        let (mut a, mut b, mut c) = (Matrix3::<f64>::zeros(), Matrix3::<f64>::zeros(), Matrix3::<f64>::zeros());
        for &i in ids {
            let (q, m) = ((p[i] - c0).cross(&n[i]), n[i]);
            a += q * q.transpose();
            b += q * m.transpose();
            c += m * m.transpose();
            if let Some(s) = s_perp {
                a += (s * s) * m * m.transpose();
            }
        }
        let k = ids.len() as f64;
        let ci = (c + Matrix3::identity() * (1e-9 * c.trace())).try_inverse()?;
        let s = a - b * ci * b.transpose();
        let (lam, ev) = eig(&((s + s.transpose()) * 0.5));
        let mut u = ev[2];
        if u[2] < 0.0 {
            u = -u;
        }
        let mut m = -(ci * b.transpose() * u);
        m -= u * u.dot(&m);
        Some((u, u.cross(&m), lam[2].max(0.0) / k, lam[1].max(0.0) / k))
    };
    let resid = |u: &V, x0: &V, i: usize| {
        let e = u.dot(&(p[i] - c0 - x0).cross(&n[i]));
        let f = s_perp.map(|s| s * n[i].dot(u)).unwrap_or(0.0);
        (e * e + f * f).sqrt()
    };
    let (mut u, mut x0, mut lam, mut lam2) = solve(idx)?;
    let res: Vec<f64> = idx.iter().map(|&i| resid(&u, &x0, i)).collect();
    let cut = 3.0 * mad(res.clone()).max(1e-5);
    let inl: Vec<usize> = idx.iter().zip(&res).filter(|(_, &x)| x <= cut).map(|(&i, _)| i).collect();
    let mut ids: &[usize] = idx;
    if inl.len() < idx.len() {
        if let Some(f) = solve(&inl) {
            (u, x0, lam, lam2) = f;
            ids = &inl;
        }
    }
    let used = ids.len();
    // the radius over the surfels the fit kept
    let mut d: Vec<f64> = ids
        .iter()
        .map(|&i| {
            let q = p[i] - c0 - x0;
            (q - u * q.dot(&u)).norm()
        })
        .collect();
    let r = super::reconstruct::median(&mut d);
    let nu = idx.iter().map(|&i| n[i].dot(&u)).sum::<f64>() / idx.len() as f64;
    Some(LineFit { u, c: c0 + x0, lam, lam2, r, nu, used })
}

/// Medial collapse as `surfels::region_stats` has it (axis the normal tensor's null direction, one 3-MAD refit):
/// (axis, R, spread, spread at 0, the line's point nearest `c`).
/// Given `axis`, collapse about it instead.
fn medial(p: &[V], n: &[V], idx: &[usize], c: &V, r_max: f64, axis: Option<V>) -> (V, f64, f64, f64, V) {
    let k = idx.len() as f64;
    let t = idx.iter().map(|&i| n[i] * n[i].transpose()).sum::<Matrix3<f64>>() / k;
    let a = axis.unwrap_or_else(|| eig(&t).1[2]);
    let (mut r, mut s, s0, mut q) = collapse(p, n, idx, &a, r_max);
    let off = |r: f64, q: &V, i: usize| {
        let d = p[i] - n[i] * r - q;
        (d - a * d.dot(&a)).norm()
    };
    let res: Vec<f64> = idx.iter().map(|&i| off(r, &q, i)).collect();
    let cut = 3.0 * mad(res.clone()).max(1e-4);
    let inl: Vec<usize> = idx.iter().zip(&res).filter(|(_, &x)| x <= cut).map(|(&i, _)| i).collect();
    if inl.len() >= 10 && inl.len() < idx.len() {
        (r, s, _, q) = collapse(p, n, &inl, &a, r_max);
    }
    (a, r, s, s0, q + a * (c - q).dot(&a))
}

/// The support function about an axis: `h = (p − c)·n⊥` against the normal's angle φ across it, fitted as
/// `a0 + a1 cos φ + b1 sin φ + a2 cos 2φ + b2 sin 2φ`; also the widest gap in φ (rad), the residuals' rms (m) and
/// the surfels used. With `convex`, a fit with `|a2| > a0 / 3` (where `h + h''` goes negative, so no convex section
/// has it) is refitted with `|a2| = a0 / 3` along its own direction, linear in `a0, a1, b1`.
fn support(p: &[V], n: &[V], idx: &[usize], f: &LineFit, convex: bool) -> Option<([f64; 5], f64, f64, usize)> {
    let e1 = f.u.cross(&if f.u[2].abs() < 0.9 { V::z() } else { V::x() }).normalize();
    let e2 = f.u.cross(&e1);
    let mut g = SMatrix::<f64, 5, 5>::zeros();
    let mut b = SVector::<f64, 5>::zeros();
    let mut phis = Vec::with_capacity(idx.len());
    let mut rows: Vec<(SVector<f64, 5>, f64)> = Vec::with_capacity(idx.len());
    for &i in idx {
        let nn = n[i] - f.u * n[i].dot(&f.u);
        if nn.norm() < 0.5 {
            continue;
        }
        let nn = nn.normalize();
        let phi = nn.dot(&e2).atan2(nn.dot(&e1));
        let h = (p[i] - f.c).dot(&nn);
        let x = SVector::<f64, 5>::from([1.0, phi.cos(), phi.sin(), (2.0 * phi).cos(), (2.0 * phi).sin()]);
        g += x * x.transpose();
        b += x * h;
        phis.push(phi);
        rows.push((x, h));
    }
    if phis.len() < 10 {
        return None;
    }
    phis.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut gap = phis[0] + 2.0 * std::f64::consts::PI - phis[phis.len() - 1];
    for w in phis.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    let mut x = g.try_inverse()? * b;
    let a2 = x[3].hypot(x[4]);
    if convex && x[0] > 0.0 && a2 > x[0] / 3.0 {
        let th = x[4].atan2(x[3]);
        let (mut g3, mut b3) = (nalgebra::Matrix3::<f64>::zeros(), V::zeros());
        for (r, h) in &rows {
            // r = (1, cos, sin, cos 2, sin 2): h = a0 (1 + cos(2 phi - th) / 3) + a1 cos + b1 sin
            let z = V::new(1.0 + (r[3] * th.cos() + r[4] * th.sin()) / 3.0, r[1], r[2]);
            g3 += z * z.transpose();
            b3 += z * *h;
        }
        let y = g3.try_inverse()? * b3;
        x = SVector::<f64, 5>::from([y[0], y[1], y[2], y[0] / 3.0 * th.cos(), y[0] / 3.0 * th.sin()]);
    }
    let rms = (rows.iter().map(|(r, h)| (h - r.dot(&x)).powi(2)).sum::<f64>() / rows.len() as f64).sqrt();
    Some(([x[0], x[1], x[2], x[3], x[4]], gap, rms, rows.len()))
}

/// One region's line fits, its medial collapse and its support harmonics.
#[derive(Debug, Clone)]
pub struct RegionLines {
    pub n: u32,
    pub n_scans: u32,
    pub cyl: Option<LineFit>,
    pub cone: Option<LineFit>,
    /// Medial collapse: axis, R, spread, spread at 0, centre.
    pub medial: Option<(V, f64, f64, f64, V)>,
    /// Medial collapse about the cylinder's line-complex axis.
    pub medial_lc: Option<(V, f64, f64, f64, V)>,
    pub support: Option<([f64; 5], f64, f64, usize)>,
    /// The mean of each attribute over the region's surfels.
    pub attr: Vec<f64>,
}

/// What `region_lines` computes: bit 1 the cylinder's line complex, 2 the cone's, 4 medial collapse, 8 the support
/// harmonics (needs 1; with 32 constrained to a convex section), 16 medial collapse about the line-complex axis
/// (needs 1); 0 gathers only (for timing).
pub fn region_lines(
    p: &[Point],
    n: &[Point],
    scan: &[i32],
    delta: &[f64],
    attr: &[Vec<f64>],
    regions: &[Region],
    margin: f64,
    min_n: usize,
    what: u32,
) -> Vec<RegionLines> {
    use rayon::prelude::*;
    let grid = SurfelGrid::new(p);
    let pv: Vec<V> = p.iter().map(v).collect();
    let nv: Vec<V> = n.iter().map(v).collect();
    // the gather uses the given positions; the fits, the positions moved by delta along the normal
    let qv: Vec<V> = if delta.is_empty() { pv.clone() } else { pv.iter().zip(&nv).zip(delta).map(|((x, y), d)| x + y * *d).collect() };
    regions
        .par_iter()
        .map(|g| {
            let idx = grid.gather(&pv, scan, g, margin);
            let mut sc: Vec<i32> = idx.iter().map(|&i| scan[i]).collect();
            sc.sort_unstable();
            sc.dedup();
            let mut out = RegionLines {
                n: idx.len() as u32,
                n_scans: sc.len() as u32,
                cyl: None,
                cone: None,
                medial: None,
                medial_lc: None,
                support: None,
                attr: attr.iter().map(|a| idx.iter().map(|&i| a[i]).sum::<f64>() / idx.len().max(1) as f64).collect(),
            };
            if idx.len() < min_n {
                return out;
            }
            let c0 = v(&g.p);
            let s = g.r.max(0.01);
            if what & 1 != 0 {
                out.cyl = line_complex(&qv, &nv, &idx, &c0, Some(s));
            }
            if what & 2 != 0 {
                out.cone = line_complex(&qv, &nv, &idx, &c0, None);
            }
            if what & 4 != 0 {
                out.medial = Some(medial(&qv, &nv, &idx, &c0, 0.3, None));
            }
            if what & 16 != 0 {
                out.medial_lc = out.cyl.map(|f| medial(&qv, &nv, &idx, &c0, 0.3, Some(f.u)));
            }
            if what & 8 != 0 {
                if let Some(f) = &out.cyl {
                    out.support = support(&qv, &nv, &idx, f, what & 32 != 0);
                }
            }
            out
        })
        .collect()
}

/// A gradient normal: the pulse, its unit normal toward the scanner (world), `cos ι`, the cells in the fit, and
/// whether they spanned both directions (else only the gradient along the row they lie in is known).
#[derive(Debug, Clone, Copy)]
pub struct GradNormal {
    pub pulse: u32,
    pub n: Point,
    pub cos_inc: f64,
    pub cells: u32,
    pub full: bool,
}

/// Normals from the gradient of log range over each kept cell's window (`half` cells either way, neighbours across
/// a range step over `jump` left out, one refit without residuals beyond 3 MAD).
pub fn grad_normals(map: &ShadowMap, dir: &[Point], keep: impl Fn(usize, &Point) -> bool + Sync, half: i64, jump: f64, min_aspect: f64) -> Vec<GradNormal> {
    use rayon::prelude::*;
    let lat: &Lattice = map.lat;
    let (nl, nc) = (lat.n_lines as i64, lat.n_cols as i64);
    let rot = Matrix3::from_fn(|i, j| map.rot[i][j]);
    let org = v(&map.origin);
    (0..lat.line.len())
        .into_par_iter()
        .filter_map(|i| {
            let (l, c) = (lat.line[i], lat.col[i]);
            if l == u32::MAX || map.cell[l as usize * lat.n_cols + c as usize] != i as u32 {
                return None;
            }
            let w = v(&dir[i]);
            let ri = map.range[i] as f64;
            if !keep(i, &pt(&(rot * w * ri + org))) {
                return None;
            }
            let e1 = w.cross(&if w[2].abs() < 0.9 { V::z() } else { V::x() }).normalize();
            let e2 = w.cross(&e1);
            let mut obs: Vec<(f64, f64, f64)> = Vec::new();
            for dl in -half..=half {
                for dc in -half..=half {
                    let (ll, cc) = (l as i64 + dl, c as i64 + dc);
                    if (dl == 0 && dc == 0) || ll < 0 || cc < 0 || ll >= nl || cc >= nc {
                        continue;
                    }
                    let j = map.cell[(ll * nc + cc) as usize];
                    if j == u32::MAX {
                        continue;
                    }
                    let rj = map.range[j as usize] as f64;
                    if (rj - ri).abs() > jump {
                        continue;
                    }
                    let wj = v(&dir[j as usize]);
                    obs.push((wj.dot(&e1), wj.dot(&e2), (rj / ri).ln()));
                }
            }
            let fit = |o: &[(f64, f64, f64)]| -> Option<(f64, f64, bool)> {
                if o.len() < 2 {
                    return None;
                }
                let (mut a, mut b, mut d, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for &(x1, x2, y) in o {
                    a += x1 * x1;
                    b += x1 * x2;
                    d += x2 * x2;
                    y1 += x1 * y;
                    y2 += x2 * y;
                }
                let tr = a + d;
                let det = a * d - b * b;
                let disc = ((a - d).powi(2) + 4.0 * b * b).sqrt();
                let (lmax, lmin) = (0.5 * (tr + disc), 0.5 * (tr - disc));
                if lmax <= 0.0 {
                    return None;
                }
                if (lmin / lmax).max(0.0).sqrt() >= min_aspect && det > 0.0 {
                    Some(((d * y1 - b * y2) / det, (a * y2 - b * y1) / det, true))
                } else {
                    // a row of cells: the gradient along it only
                    let (vx, vy) = if b.abs() > 1e-30 { (lmax - d, b) } else if a >= d { (1.0, 0.0) } else { (0.0, 1.0) };
                    let nv = (vx * vx + vy * vy).sqrt();
                    let (vx, vy) = (vx / nv, vy / nv);
                    let g = (vx * y1 + vy * y2) / lmax;
                    Some((g * vx, g * vy, false))
                }
            };
            let (mut g1, mut g2, mut full) = fit(&obs)?;
            let res: Vec<f64> = obs.iter().map(|&(x1, x2, y)| (y - g1 * x1 - g2 * x2).abs()).collect();
            let cut = (3.0 * mad(res.clone())).max(0.5 * 0.003 / ri);
            let inl: Vec<(f64, f64, f64)> = obs.iter().zip(&res).filter(|(_, &e)| e <= cut).map(|(o, _)| *o).collect();
            let mut cells = obs.len();
            if inl.len() < obs.len() {
                if let Some(f) = fit(&inl) {
                    (g1, g2, full) = f;
                    cells = inl.len();
                }
            }
            let grad = e1 * g1 + e2 * g2;
            let nrm = (grad - w).normalize();
            Some(GradNormal { pulse: i as u32, n: pt(&(rot * nrm)), cos_inc: 1.0 / (1.0 + grad.norm_squared()).sqrt(), cells: cells as u32, full })
        })
        .collect()
}

/// A ridge: a connected run of cells with a first echo but no surfel.
#[derive(Debug, Clone, Copy)]
pub struct Ridge {
    pub cells: u32,
    /// Extent along the cells' major direction (cells), and cells over it.
    pub len_cells: f64,
    pub width_cells: f64,
    /// 3D (world): centroid, unit major direction (up), extent along it (m), rms off that line (m), the two ends.
    pub centre: Point,
    pub u: Point,
    pub len: f64,
    pub rms_off: f64,
    pub end0: Point,
    pub end1: Point,
    /// The unit normal of the plane through the scanner that holds the ridge's rays (world).
    pub plane: Point,
    pub refl: f64,
    /// Share of its pulses with more than one echo, and its mean range (m).
    pub multi: f64,
    pub range: f64,
}

/// The ridges of the kept cells without a surfel (`has_surfel` per pulse): each kept pulse's ridge (u32::MAX: none)
/// and the ridges.
pub fn ridges(
    map: &ShadowMap,
    dir: &[Point],
    refl: &[f32],
    echoes: &[u8],
    keep: impl Fn(usize, &Point) -> bool,
    has_surfel: &[bool],
    jump: f64,
) -> (Vec<u32>, Vec<Ridge>) {
    let lat: &Lattice = map.lat;
    let (nl, nc) = (lat.n_lines as i64, lat.n_cols as i64);
    let rot = Matrix3::from_fn(|i, j| map.rot[i][j]);
    let org = v(&map.origin);
    let world = |i: usize| rot * v(&dir[i]) * map.range[i] as f64 + org;
    let mut label = vec![u32::MAX; lat.line.len()];
    let mut cand = vec![false; lat.line.len()];
    for i in 0..lat.line.len() {
        let l = lat.line[i];
        if l != u32::MAX && map.cell[l as usize * lat.n_cols + lat.col[i] as usize] == i as u32 && !has_surfel[i] && keep(i, &pt(&world(i))) {
            cand[i] = true;
        }
    }
    let mut out = Vec::new();
    for s in 0..lat.line.len() {
        if !cand[s] || label[s] != u32::MAX {
            continue;
        }
        let id = out.len() as u32;
        let mut stack = vec![s];
        let mut comp = Vec::new();
        label[s] = id;
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (l, c) = (lat.line[i] as i64, lat.col[i] as i64);
            for dl in -1..=1 {
                for dc in -1..=1 {
                    let (ll, cc) = (l + dl, c + dc);
                    if ll < 0 || cc < 0 || ll >= nl || cc >= nc {
                        continue;
                    }
                    let j = map.cell[(ll * nc + cc) as usize];
                    if j == u32::MAX {
                        continue;
                    }
                    let j = j as usize;
                    if cand[j] && label[j] == u32::MAX && ((map.range[j] - map.range[i]) as f64).abs() <= jump {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        let k = comp.len() as f64;
        // extent in cells
        let (ml, mc) = (comp.iter().map(|&i| lat.line[i] as f64).sum::<f64>() / k, comp.iter().map(|&i| lat.col[i] as f64).sum::<f64>() / k);
        let (mut a, mut b, mut d) = (0.0, 0.0, 0.0);
        for &i in &comp {
            let (x, y) = (lat.line[i] as f64 - ml, lat.col[i] as f64 - mc);
            a += x * x;
            b += x * y;
            d += y * y;
        }
        let ang = 0.5 * (2.0 * b).atan2(a - d);
        let (ex, ey) = (ang.cos(), ang.sin());
        let proj: Vec<f64> = comp.iter().map(|&i| (lat.line[i] as f64 - ml) * ex + (lat.col[i] as f64 - mc) * ey).collect();
        let len_cells = proj.iter().cloned().fold(f64::MIN, f64::max) - proj.iter().cloned().fold(f64::MAX, f64::min) + 1.0;
        // 3D
        let xs: Vec<V> = comp.iter().map(|&i| world(i)).collect();
        let m = xs.iter().sum::<V>() / k;
        let cov = xs.iter().map(|x| (x - m) * (x - m).transpose()).sum::<Matrix3<f64>>() / k;
        let (lam, ev) = eig(&cov);
        let mut u = ev[0];
        if u[2] < 0.0 {
            u = -u;
        }
        let t: Vec<f64> = xs.iter().map(|x| (x - m).dot(&u)).collect();
        let (imin, imax) = (
            (0..t.len()).min_by(|&p, &q| t[p].partial_cmp(&t[q]).unwrap()).unwrap(),
            (0..t.len()).max_by(|&p, &q| t[p].partial_cmp(&t[q]).unwrap()).unwrap(),
        );
        let ws = comp.iter().map(|&i| rot * v(&dir[i])).map(|w| w * w.transpose()).sum::<Matrix3<f64>>();
        let plane = SymmetricEigen::new(ws);
        let j = plane.eigenvalues.imin();
        out.push(Ridge {
            cells: comp.len() as u32,
            len_cells,
            width_cells: k / len_cells,
            centre: pt(&m),
            u: pt(&u),
            len: t[imax] - t[imin],
            rms_off: (lam[1] + lam[2]).max(0.0).sqrt(),
            end0: pt(&xs[imin]),
            end1: pt(&xs[imax]),
            plane: pt(&plane.eigenvectors.column(j).into_owned()),
            refl: comp.iter().map(|&i| refl[i] as f64).sum::<f64>() / k,
            multi: comp.iter().filter(|&&i| echoes[i] > 1).count() as f64 / k,
            range: comp.iter().map(|&i| map.range[i] as f64).sum::<f64>() / k,
        });
    }
    (label, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surfels on a cylinder of radius r about the z axis through (x0, y0), over an arc of `arc` radians.
    fn cylinder(r: f64, x0: f64, y0: f64, arc: f64, taper: f64) -> (Vec<V>, Vec<V>) {
        let (mut p, mut n) = (Vec::new(), Vec::new());
        for i in 0..40 {
            for j in 0..10 {
                let phi = arc * (i as f64 / 39.0 - 0.5);
                let z = 0.01 * j as f64 - 0.05;
                let rr = r + taper * z;
                let nn = V::new(phi.cos(), phi.sin(), -taper).normalize();
                p.push(V::new(x0 + rr * phi.cos(), y0 + rr * phi.sin(), z));
                n.push(nn);
            }
        }
        (p, n)
    }

    #[test]
    fn the_line_complex_axis_is_the_cylinders_and_ignores_offsets() {
        let (p, n) = cylinder(0.03, 0.01, -0.02, 2.0, 0.0);
        let idx: Vec<usize> = (0..p.len()).collect();
        let c0 = V::new(0.0, 0.0, 0.0);
        let f = line_complex(&p, &n, &idx, &c0, Some(0.03)).unwrap();
        assert!(f.u[2] > 0.9999, "{:?}", f.u);
        assert!((f.c - V::new(0.01, -0.02, 0.0)).norm() < 1e-6, "{:?}", f.c);
        assert!((f.r - 0.03).abs() < 1e-6 && f.lam < 1e-12);
        // offset every surfel along its normal by a different amount: the same axis, a larger radius
        let q: Vec<V> = p.iter().zip(&n).enumerate().map(|(i, (x, y))| x + y * (0.005 * (i % 3) as f64)).collect();
        let g = line_complex(&q, &n, &idx, &c0, Some(0.03)).unwrap();
        assert!((g.c - f.c).norm() < 1e-9 && (g.u - f.u).norm() < 1e-9);
        // a cone: the cylinder's rows misfit, the surface of revolution's find it and its taper
        let (p, n) = cylinder(0.05, 0.0, 0.0, 6.0, -0.2);
        let f = line_complex(&p, &n, &idx, &c0, None).unwrap();
        assert!(f.u[2] > 0.9999 && f.c.norm() < 1e-6, "{:?} {:?}", f.u, f.c);
        let slope = -f.nu / (1.0 - f.nu * f.nu).sqrt();
        assert!((slope + 0.2).abs() < 1e-6, "{slope}");
    }

    #[test]
    fn gradient_normals_point_away_from_a_cylinders_axis() {
        use crate::shots::surfels::tests::scene;
        let (r, d) = (0.05, 8.0);
        let (lat, dirs, rng) = scene(r, d, 0.003);
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let map = ShadowMap::new(&lat, &rng, id, [0.0; 3]);
        let on = |_: usize, p: &Point| (p[0].powi(2) + (p[1] - d).powi(2)).sqrt() < r + 0.01;
        for half in [1, 3] {
            let g = grad_normals(&map, &dirs, on, half, 0.05, 0.2);
            let mut err: Vec<f64> = g
                .iter()
                .filter(|x| x.full)
                .map(|x| {
                    let i = x.pulse as usize;
                    let p = v(&dirs[i]) * rng[i] as f64;
                    let radial = V::new(p[0], p[1] - d, 0.0).normalize();
                    v(&x.n).dot(&radial).clamp(-1.0, 1.0).acos().to_degrees()
                })
                .collect();
            err.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (med, p90) = (err[err.len() / 2], err[err.len() * 9 / 10]);
            eprintln!("gradient normals, half {half}: {} of {}, error median {med:.2} deg, p90 {p90:.2} deg", err.len(), g.len());
            assert!(med < if half == 1 { 25.0 } else { 6.0 }, "{med}");
        }
    }
}
