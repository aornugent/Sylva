// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Surface elements for a leaf detector (3dtrees research 128,
//! `merge-leaves.md`; a spike, not a fit): every scan's surfels within a few
//! centimetres of a query point, summarised as a flat element would be.
//!
//! - **normal tensor** `Σ n nᵀ / N`: eigenvalues (sum 1, descending) and the
//!   dominant normal (a flat lamina is rank 1);
//! - **shape** of the positions: rms extent along the two largest directions
//!   and thickness along the dominant normal;
//! - the means of per-surfel attributes (reflectance, incidence, …).

use super::surfels::{eig, pt, v, SurfelGrid, V};
use crate::Point;
use nalgebra::Matrix3;

#[derive(Debug, Clone)]
pub struct Element {
    pub n: u32,
    pub n_scans: u32,
    pub lam: [f64; 3],
    /// The dominant normal (z ≥ 0).
    pub normal: Point,
    /// Rms extent along the positions' two largest directions, and thickness along the dominant normal (m).
    pub extent: [f64; 2],
    pub thick: f64,
    pub attr: Vec<f64>,
}

/// The element of every query point `c`: the surfels within `rad`, at least `min_n` of them (else n only).
pub fn elements(p: &[Point], n: &[Point], scan: &[i32], attr: &[Vec<f64>], c: &[Point], rad: f64, min_n: usize) -> Vec<Element> {
    use rayon::prelude::*;
    let grid = SurfelGrid::new(p);
    let pv: Vec<V> = p.iter().map(v).collect();
    let nv: Vec<V> = n.iter().map(v).collect();
    c.par_iter()
        .map(|x| {
            let idx = grid.ball(&pv, &v(x), rad);
            let mut sc: Vec<i32> = idx.iter().map(|&i| scan[i]).collect();
            sc.sort_unstable();
            sc.dedup();
            let nan = f64::NAN;
            let mut e = Element { n: idx.len() as u32, n_scans: sc.len() as u32, lam: [nan; 3], normal: [nan; 3], extent: [nan; 2], thick: nan, attr: vec![nan; attr.len()] };
            if idx.len() < min_n {
                return e;
            }
            let k = idx.len() as f64;
            let t = idx.iter().map(|&i| nv[i] * nv[i].transpose()).sum::<Matrix3<f64>>() / k;
            let (lam, ev) = eig(&t);
            let tot: f64 = lam.iter().sum();
            e.lam = lam.map(|l| l / tot);
            let mut nn = ev[0];
            if nn[2] < 0.0 {
                nn = -nn;
            }
            e.normal = pt(&nn);
            let m = idx.iter().map(|&i| pv[i]).sum::<V>() / k;
            let cov = idx.iter().map(|&i| (pv[i] - m) * (pv[i] - m).transpose()).sum::<Matrix3<f64>>() / k;
            let (pl, _) = eig(&cov);
            e.extent = [pl[0].max(0.0).sqrt(), pl[1].max(0.0).sqrt()];
            e.thick = (idx.iter().map(|&i| (pv[i] - m).dot(&nn).powi(2)).sum::<f64>() / k).sqrt();
            e.attr = attr.iter().map(|a| idx.iter().map(|&i| a[i]).sum::<f64>() / k).collect();
            e
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vertical_lamina_is_rank_one_with_a_horizontal_normal() {
        let (mut p, mut n) = (Vec::new(), Vec::new());
        for i in 0..10 {
            for j in 0..20 {
                p.push([0.002 * i as f64, 0.0, 0.003 * j as f64]);
                n.push([0.0, -1.0, 0.0]);
            }
        }
        let sc = vec![0; p.len()];
        let e = &elements(&p, &n, &sc, &[], &[[0.01, 0.0, 0.03]], 0.05, 5)[0];
        assert!(e.lam[0] > 0.99 && e.normal[2].abs() < 1e-9 && e.thick < 1e-9, "{e:?}");
    }
}
