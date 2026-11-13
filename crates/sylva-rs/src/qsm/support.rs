// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Which trees' points crowd a tree's cylinders: [`contest`], of the points
//! within `tol` of each tree's cylinders, the share another tree's label
//! holds, and which tree holds most; batched over a plot's trees, in parallel.

use rayon::prelude::*;

use crate::transform::sub;
use crate::util::spatial::KdTree;
use crate::Point;

/// Each group's contested points: group `g` (a tree, whose points carry label
/// `g`) owns cylinders `cyl_offsets[g]..cyl_offsets[g + 1]`; of the cloud's
/// points within `tol` of their surfaces (axis distance < r + `tol`), each
/// counted once, the share labelled another group (`>= 0`, `!= g`), and the
/// label holding most of those (ties to the lowest; -1 if none). A group with
/// no such points gives `(0, -1)`. The tests are written as numpy wrote them
/// (3dtrees' `contest_fit`), so the two agree to the bit.
pub fn contest(cloud: &KdTree, points: &[Point], labels: &[i32], a: &[Point], b: &[Point], r: &[f64], cyl_offsets: &[usize], tol: f64) -> (Vec<f64>, Vec<i64>) {
    (0..cyl_offsets.len().saturating_sub(1))
        .into_par_iter()
        .map(|g| {
            let mut near = Vec::new();
            for c in cyl_offsets[g]..cyl_offsets[g + 1] {
                let d = sub(&b[c], &a[c]);
                let l2 = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2];
                let mid = [(a[c][0] + b[c][0]) / 2.0, (a[c][1] + b[c][1]) / 2.0, (a[c][2] + b[c][2]) / 2.0];
                for (p, _) in cloud.within(&mid, l2.sqrt() / 2.0 + r[c] + tol) {
                    let q = sub(&points[p], &a[c]);
                    let t = (((q[0] * d[0] + q[1] * d[1]) + q[2] * d[2]) / l2.max(1e-12)).clamp(0.0, 1.0);
                    let e = [q[0] - t * d[0], q[1] - t * d[1], q[2] - t * d[2]];
                    if ((e[0] * e[0] + e[1] * e[1]) + e[2] * e[2]).sqrt() < r[c] + tol {
                        near.push(p);
                    }
                }
            }
            near.sort_unstable();
            near.dedup();
            if near.is_empty() {
                return (0.0, -1);
            }
            let mut other: Vec<i32> = near.iter().map(|&p| labels[p]).filter(|&l| l >= 0 && l as usize != g).collect();
            let share = other.len() as f64 / near.len() as f64;
            other.sort_unstable();
            let (mut owner, mut best, mut i) = (-1i64, 0usize, 0usize);
            while i < other.len() {
                let j = i + other[i..].iter().take_while(|&&l| l == other[i]).count();
                if j - i > best {
                    (owner, best) = (other[i] as i64, j - i);
                }
                i = j;
            }
            (share, owner)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contest_counts_the_points_another_group_holds() {
        // two stems 4 cm apart, each with its own column of points: each sits in the other's (3dtrees' test)
        let a = [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]];
        let b = [[0.0, 0.0, 2.0], [0.04, 0.0, 2.0]];
        let r = [0.05, 0.01];
        let mut pts = Vec::new();
        let mut lab = Vec::new();
        for (x, l) in [(0.0, 0), (0.04, 1)] {
            for i in 0..10 {
                pts.push([x, 0.0, 0.1 + 0.2 * i as f64]);
                lab.push(l);
            }
        }
        let (share, owner) = contest(&KdTree::new(&pts), &pts, &lab, &a, &b, &r, &[0, 1, 2], 0.05);
        assert_eq!(share, vec![0.5, 0.5]);
        assert_eq!(owner, vec![1, 0]);
        let (share, owner) = contest(&KdTree::new(&[]), &[], &[], &a, &b, &r, &[0, 1, 1, 2], 0.05);
        assert_eq!((share, owner), (vec![0.0; 3], vec![-1; 3])); // no points, and a group without cylinders
    }
}
