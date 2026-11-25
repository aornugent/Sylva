// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A learned MTA decision rule: gradient-boosted trees trained on rimta5's
//! own output (`riegl-data/mta/train_mta.py`), evaluated here on the evidence
//! the three passes of [`super::mta`] gather per echo and zone.
//!
//! The model scores every (echo, zone) pair; the best-scoring zone is taken
//! and the echo kept when its score reaches the model's threshold. The
//! deterministic filters (amplitude 0, dynamic trigger, range gate, notch)
//! stay outside the model. The JSON format is plain enough to produce from
//! any tree library: per tree, parallel arrays `value`, `feature`,
//! `threshold`, `missing_left`, `left`, `right`, `leaf` (scikit-learn's
//! `HistGradientBoostingClassifier` node table), plus `baseline` (log-odds)
//! and the `features` list in the order [`row_features`] produces them.
//!
//! Where only the keep/drop call matters ([`MtaModel::decide_keep`] with a
//! single zone hypothesis), the sum stops as soon as the trees still to come
//! cannot carry it across the threshold, whatever leaves they reach: each
//! tree's lowest and highest leaf bound what it can add. The trees are summed
//! widest leaf range first, so the bounds close early, and a sum that ends
//! too near the threshold to be sure of its rounding is redone exactly as
//! [`MtaModel::score`] does it, so the call is always the one the full score
//! makes.

use std::path::Path;

use crate::error::{Error, Result};

use super::mta::{Evidence, MAXZ};

/// Feature names, in the order the model expects them.
pub const FEATURE_NAMES: [&str; 25] = [
    "zone", "range_k", "refl_k", "votes_own_k", "votes_nb_k", "anchors_k", "density_k",
    "votes_own_other", "votes_nb_other", "anchors_other", "density_other",
    "votes_own_sum", "anchors_sum", "density_sum",
    "amplitude", "deviation", "r1", "late", "n_echoes", "idx_in_shot", "n_zones",
    "prev_amp", "prev_gap", "next_amp", "shot_frac",
];

struct Node {
    value: f32,
    feature: u16,
    threshold: f32,
    missing_left: bool,
    left: u32,
    right: u32,
    leaf: bool,
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// The leaf value `x` reaches.
    fn leaf(&self, x: &[f32; FEATURE_NAMES.len()]) -> f64 {
        let mut i = 0usize;
        loop {
            let n = &self.nodes[i];
            if n.leaf {
                return n.value as f64;
            }
            let v = x[n.feature as usize];
            let go_left = if v.is_nan() { n.missing_left } else { v <= n.threshold };
            i = if go_left { n.left as usize } else { n.right as usize };
        }
    }

    /// Lowest and highest leaf value.
    fn leaf_range(&self) -> (f64, f64) {
        self.nodes.iter().filter(|n| n.leaf).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), n| (lo.min(n.value as f64), hi.max(n.value as f64)))
    }
}

/// A boosted-tree decision rule loaded from JSON.
pub struct MtaModel {
    kmax: usize,
    baseline: f64,
    threshold: f32,
    trees: Vec<Tree>,
    /// Tree indices, widest leaf range first: the order [`Self::keeps`] sums them in.
    order: Vec<u32>,
    /// Sums of the lowest (`rest_lo`) and highest (`rest_hi`) leaves of the trees from each position of `order` on.
    rest_lo: Vec<f64>,
    rest_hi: Vec<f64>,
    /// A raw score (log-odds) at or above `sure_keep` keeps and one at or below `sure_drop` drops, with room to
    /// spare for the rounding of a sum taken in another order; both infinite where no such margin is safe.
    sure_keep: f64,
    sure_drop: f64,
}

/// The probability of a raw score, as [`MtaModel::score`] rounds it.
fn probability(raw: f64) -> f32 {
    (1.0 / (1.0 + (-raw).exp())) as f32
}

/// What the model needs to know about one echo besides the per-zone evidence.
#[derive(Clone, Copy, Debug)]
pub struct EchoInfo {
    /// Echo amplitude (dB).
    pub amplitude: f32,
    /// RIEGL pulse shape deviation of the echo (dimensionless; small values mean an undistorted pulse).
    pub deviation: f32,
    /// Range under the zone-1 hypothesis (m), RiVLib's range before any MTA shift.
    pub r1: f32,
    /// Whether the echo arrived after the next pulse was fired (shim flag bit 5).
    pub late: bool,
    /// Echoes recorded for the same pulse (count, capped at 255).
    pub n_echoes: u8,
    /// 1-based position of this echo among its pulse's echoes, nearest first (capped at 255).
    pub idx_in_shot: u8,
    /// Strongest amplitude among the closer echoes of the same pulse (-1 if none).
    pub prev_amp: f32,
    /// Range gap to the closest closer echo of the same pulse (-1 if none).
    pub prev_gap: f32,
    /// Strongest amplitude among the farther echoes of the same pulse (-1 if none).
    pub next_amp: f32,
    /// Position of the shot within its scan line, 0..1.
    pub shot_frac: f32,
}

impl MtaModel {
    /// Parse the JSON produced by `train_mta.py`.
    pub fn from_json(text: &str) -> Result<MtaModel> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| Error::Riegl(format!("MTA model: {e}")))?;
        let bad = |what: &str| Error::Riegl(format!("MTA model: missing or malformed {what}"));
        if v.get("format").and_then(|f| f.as_str()) != Some("sylva-mta-model/1") {
            return Err(bad("format (expected sylva-mta-model/1)"));
        }
        let names: Vec<&str> = v.get("features").and_then(|f| f.as_array()).ok_or_else(|| bad("features"))?.iter().filter_map(|s| s.as_str()).collect();
        if names != FEATURE_NAMES {
            return Err(Error::Riegl(format!("MTA model: feature list {names:?} does not match this build ({FEATURE_NAMES:?})")));
        }
        let kmax = v.get("kmax").and_then(|k| k.as_u64()).ok_or_else(|| bad("kmax"))? as usize;
        let baseline = v.get("baseline").and_then(|b| b.as_f64()).ok_or_else(|| bad("baseline"))?;
        let threshold = v.get("threshold").and_then(|b| b.as_f64()).ok_or_else(|| bad("threshold"))? as f32;
        let mut trees = Vec::new();
        for t in v.get("trees").and_then(|t| t.as_array()).ok_or_else(|| bad("trees"))? {
            let arr = |key: &str| -> Result<Vec<f64>> {
                t.get(key).and_then(|a| a.as_array()).ok_or_else(|| bad(key))?.iter().map(|x| x.as_f64().ok_or_else(|| bad(key))).collect()
            };
            let (value, feature, threshold, missing, left, right, leaf) =
                (arr("value")?, arr("feature")?, arr("threshold")?, arr("missing_left")?, arr("left")?, arr("right")?, arr("leaf")?);
            let n = value.len();
            if [feature.len(), threshold.len(), missing.len(), left.len(), right.len(), leaf.len()].iter().any(|&l| l != n) {
                return Err(bad("tree arrays of unequal length"));
            }
            let nodes = (0..n)
                .map(|i| Node {
                    value: value[i] as f32,
                    feature: feature[i] as u16,
                    threshold: threshold[i] as f32,
                    missing_left: missing[i] != 0.0,
                    left: left[i] as u32,
                    right: right[i] as u32,
                    leaf: leaf[i] != 0.0,
                })
                .collect();
            trees.push(Tree { nodes });
        }
        Ok(MtaModel::new(kmax.clamp(1, MAXZ), baseline, threshold, trees))
    }

    fn new(kmax: usize, baseline: f64, threshold: f32, trees: Vec<Tree>) -> MtaModel {
        let ranges: Vec<(f64, f64)> = trees.iter().map(Tree::leaf_range).collect();
        let mut order: Vec<u32> = (0..trees.len() as u32).collect();
        order.sort_by(|&a, &b| {
            let w = |t: u32| ranges[t as usize].1 - ranges[t as usize].0;
            w(b).total_cmp(&w(a))
        });
        let (mut rest_lo, mut rest_hi) = (vec![0.0; trees.len() + 1], vec![0.0; trees.len() + 1]);
        for i in (0..trees.len()).rev() {
            let (lo, hi) = ranges[order[i] as usize];
            rest_lo[i] = rest_lo[i + 1] + lo;
            rest_hi[i] = rest_hi[i + 1] + hi;
        }
        // The threshold's log-odds with a margin far above the rounding of 500 sums, kept only if the
        // probabilities on either side of it do round to the two sides of the threshold.
        let t = threshold as f64;
        let raw_t = (t / (1.0 - t)).ln();
        let margin = 1e-3 * (1.0 + raw_t.abs());
        let (mut sure_keep, mut sure_drop) = (raw_t + margin, raw_t - margin);
        if !(raw_t.is_finite() && probability(sure_keep) >= threshold && probability(sure_drop) < threshold) {
            (sure_keep, sure_drop) = (f64::INFINITY, f64::NEG_INFINITY);
        }
        MtaModel { kmax, baseline, threshold, trees, order, rest_lo, rest_hi, sure_keep, sure_drop }
    }

    /// Load from a file.
    pub fn load(path: &Path) -> Result<MtaModel> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Riegl(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Zones considered.
    pub fn kmax(&self) -> usize {
        self.kmax
    }

    /// Keep threshold on the probability.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Probability that rimta5 keeps this (echo, zone) pair.
    pub fn score(&self, x: &[f32; FEATURE_NAMES.len()]) -> f32 {
        let mut raw = self.baseline;
        for t in &self.trees {
            raw += t.leaf(x);
        }
        probability(raw)
    }

    /// Whether `score(x) >= threshold()`, summing only as many trees as the answer needs (module docs).
    pub fn keeps(&self, x: &[f32; FEATURE_NAMES.len()]) -> bool {
        let mut raw = self.baseline;
        for (i, &t) in self.order.iter().enumerate() {
            raw += self.trees[t as usize].leaf(x);
            if raw + self.rest_lo[i + 1] >= self.sure_keep {
                return true;
            }
            if raw + self.rest_hi[i + 1] <= self.sure_drop {
                return false;
            }
        }
        self.score(x) >= self.threshold
    }

    /// Best zone and its score for one echo (`zr`: hypothesis ranges, NaN = none).
    pub fn decide(&self, zr: &[f32; MAXZ], zrefl: &[f32; MAXZ], ev: &Evidence, info: EchoInfo) -> Option<(u8, f32)> {
        let mut best: Option<(u8, f32)> = None;
        for k in 0..self.kmax {
            if !zr[k].is_finite() {
                continue;
            }
            let x = row_features(k, zr, zrefl, ev, info, self.kmax);
            let p = self.score(&x);
            if best.map(|(_, b)| p > b).unwrap_or(true) {
                best = Some((k as u8 + 1, p));
            }
        }
        best
    }

    /// [`Self::decide`] for a caller that needs only the zone and whether its score reaches the threshold:
    /// `(zone, keep, score)`, the score NaN where a single hypothesis let [`Self::keeps`] stop early. The
    /// zone and keep are always the ones `decide` gives.
    pub fn decide_keep(&self, zr: &[f32; MAXZ], zrefl: &[f32; MAXZ], ev: &Evidence, info: EchoInfo) -> Option<(u8, bool, f32)> {
        let mut finite = (0..self.kmax).filter(|&k| zr[k].is_finite());
        match (finite.next(), finite.next()) {
            (None, _) => None,
            (Some(k), None) => Some((k as u8 + 1, self.keeps(&row_features(k, zr, zrefl, ev, info, self.kmax)), f32::NAN)),
            _ => self.decide(zr, zrefl, ev, info).map(|(zone, p)| (zone, p >= self.threshold, p)),
        }
    }
}

/// The feature row for hypothesis `k` (0-based), exactly as `train_mta.py` builds it.
pub fn row_features(k: usize, zr: &[f32; MAXZ], zrefl: &[f32; MAXZ], ev: &Evidence, info: EchoInfo, kmax: usize) -> [f32; FEATURE_NAMES.len()] {
    let other_max = |a: &[f32; MAXZ]| -> f32 {
        let mut m = -1.0f32;
        for j in 0..kmax {
            if j != k && zr[j].is_finite() && a[j] > m {
                m = a[j];
            }
        }
        m
    };
    let sum_pos = |a: &[f32; MAXZ]| -> f32 { (0..kmax).map(|j| a[j].max(0.0)).sum() };
    let n_zones = (0..kmax).filter(|&j| zr[j].is_finite()).count() as f32;
    [
        k as f32 + 1.0,
        zr[k],
        zrefl[k],
        ev.votes_own[k],
        ev.votes_nb[k],
        ev.anchors[k],
        ev.density[k],
        other_max(&ev.votes_own),
        other_max(&ev.votes_nb),
        other_max(&ev.anchors),
        other_max(&ev.density),
        sum_pos(&ev.votes_own),
        sum_pos(&ev.anchors),
        sum_pos(&ev.density),
        info.amplitude,
        info.deviation,
        info.r1,
        info.late as u8 as f32,
        info.n_echoes as f32,
        info.idx_in_shot as f32,
        n_zones,
        info.prev_amp,
        info.prev_gap,
        info.next_amp,
        info.shot_frac,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"{"format":"sylva-mta-model/1","features":["zone","range_k","refl_k","votes_own_k","votes_nb_k","anchors_k","density_k","votes_own_other","votes_nb_other","anchors_other","density_other","votes_own_sum","anchors_sum","density_sum","amplitude","deviation","r1","late","n_echoes","idx_in_shot","n_zones","prev_amp","prev_gap","next_amp","shot_frac"],"kmax":3,"baseline":-1.0,"threshold":0.5,
      "trees":[{"value":[0,2.0,-2.0],"feature":[3,0,0],"threshold":[5.0,0,0],"missing_left":[0,0,0],"left":[1,0,0],"right":[2,0,0],"leaf":[0,1,1]}]}"#;

    #[test]
    fn evaluates_a_hand_made_tree() {
        let m = MtaModel::from_json(TINY).unwrap();
        let mut x = [0f32; FEATURE_NAMES.len()];
        x[3] = 3.0; // votes_own_k <= 5 -> left leaf +2 -> raw 1.0
        assert!((m.score(&x) - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        x[3] = 9.0; // right leaf -2 -> raw -3
        assert!((m.score(&x) - 1.0 / (1.0 + (3.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn decides_the_best_zone() {
        let m = MtaModel::from_json(TINY).unwrap();
        let mut zr = [f32::NAN; MAXZ];
        zr[0] = 20.0;
        zr[1] = 88.0;
        let mut ev = Evidence::default();
        ev.votes_own[0] = 9.0; // zone 1 scores low
        ev.votes_own[1] = 2.0; // zone 2 scores high
        let info = EchoInfo { amplitude: 10.0, deviation: 1.0, r1: 20.0, late: false, n_echoes: 1, idx_in_shot: 1, prev_amp: -1.0, prev_gap: -1.0, next_amp: -1.0, shot_frac: 0.5 };
        let (zone, p) = m.decide(&zr, &zr, &ev, info).unwrap();
        assert_eq!(zone, 2);
        assert!(p > m.threshold());
    }

    #[test]
    fn the_early_exit_makes_the_full_scores_call() {
        let tiny = MtaModel::from_json(TINY).unwrap();
        let mut x = [0f32; FEATURE_NAMES.len()];
        for v in [3.0, 9.0, f32::NAN] {
            x[3] = v;
            assert_eq!(tiny.keeps(&x), tiny.score(&x) >= tiny.threshold());
        }
        let Some(js) = super::super::rxp_mta::DEFAULT_MODEL_JSON else { return };
        let m = MtaModel::from_json(js).unwrap();
        // Rows of plausible magnitudes, some features missing, from a fixed generator.
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        let (mut kept, mut n) = (0, 20_000);
        for _ in 0..n {
            for v in x.iter_mut() {
                let u = next();
                *v = if u < 0.05 { f32::NAN } else if u < 0.2 { -1.0 } else { next() * 60.0 - 5.0 };
            }
            x[0] = 1.0;
            let full = m.score(&x) >= m.threshold();
            assert_eq!(m.keeps(&x), full, "{x:?}");
            kept += full as usize;
        }
        // Both answers occur, so both exits were exercised.
        n -= kept;
        assert!(kept > 0 && n > 0, "kept {kept}, dropped {n}");

        let mut zr = [f32::NAN; MAXZ];
        zr[0] = 20.0;
        let ev = Evidence::default();
        let info = EchoInfo { amplitude: 10.0, deviation: 1.0, r1: 20.0, late: false, n_echoes: 1, idx_in_shot: 1, prev_amp: -1.0, prev_gap: -1.0, next_amp: -1.0, shot_frac: 0.5 };
        for zones in [1, 2, 3] {
            for k in 1..zones {
                zr[k] = 20.0 + 60.0 * k as f32;
            }
            let (zone, p) = m.decide(&zr, &zr, &ev, info).unwrap();
            let (z2, keep, _) = m.decide_keep(&zr, &zr, &ev, info).unwrap();
            assert_eq!((zone, p >= m.threshold()), (z2, keep));
        }
        assert!(m.decide_keep(&[f32::NAN; MAXZ], &zr, &ev, info).is_none());
    }

    #[test]
    fn rejects_a_foreign_feature_list() {
        let bad = TINY.replace("\"zone\",", "\"other\",");
        assert!(MtaModel::from_json(&bad).is_err());
    }
}
