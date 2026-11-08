import dataclasses

import numpy as np
import pytest

from sylva import PointCloud
from sylva.qsm import JointParams, ScanIndex, SupportFlag, support_tree

# the synthetic VZ-400i beam's detection share, calibrated on its vertical stems (3dtrees research 93 §10), and its
# echo character: wood reflects 0.5 (-3 dB) and these scenes hold no leaf
PARAMS = JointParams(tau=0.094, refl_wood=-3.0, refl_leaf=-60.0)


def scanned(tmp_path, a, b, r, scanners, zenith=(80.0, 100.0)):
    """Solid stems a -> b of radius r, scanned by Sylva's beam scanner from each (x, y) at 1.5 m."""
    import sylva.synthetic as syn
    n = len(r)
    cloud = PointCloud((a + b) / 2, {"classification": np.full(n, 5, np.uint8)})
    solids = dict(a=a, b=b, r=np.asarray(r, float), point=np.arange(n))
    scans = []
    for k, (x, y) in enumerate(scanners):
        shots = syn.scan(cloud, origin=(x, y, 1.5), resolution_deg=0.04, scanner="vz400i", mixed_pixels=True,
                         target_radius=0.01, min_zenith_deg=zenith[0], max_zenith_deg=zenith[1], seed=k, solids=solids)
        shots.save(tmp_path / f"s{k}.parquet")
        scans.append(ScanIndex(tmp_path / f"s{k}.parquet", threads=2))
    return scans


def test_separated_stems_upright_and_leaning_are_measured(tmp_path):
    # stems of 1 to 8 cm 40 cm apart, the last leaning 10 degrees along x, from three sides; each claimed as RCT
    # errs: rows of 30 cm, upright, 1.5 r fat, 0.3 r off the axis
    radii = np.array([0.01, 0.02, 0.03, 0.05, 0.08])
    xs = np.arange(5) * 0.4 - 0.8
    a, b = np.c_[xs, np.zeros(5), np.zeros(5)], np.c_[xs, np.zeros(5), np.full(5, 1.8)]
    b[-1, 0] += 1.8 * np.tan(0.17)
    scans = scanned(tmp_path, a, b, radii, [(0.3, -8.0), (6.5, 5.0), (-6.0, 5.5)], zenith=(75.0, 110.0))
    rng = np.random.default_rng(0)
    ra, rb, rr, truth, lean = [], [], [], [], []
    for i, r in enumerate(radii):
        u = (b[i] - a[i]) / (b[i, 2] - a[i, 2])  # per m of height
        for z in (0.4, 0.75, 1.1):
            ang = rng.uniform(0, 2 * np.pi)
            off = 0.3 * r * np.array([np.cos(ang), np.sin(ang), 0.0])
            ra.append(a[i] + z * u + off)
            rb.append(a[i] + z * u + off + [0, 0, 0.3])
            rr.append(1.5 * r)
            truth.append((a[i] + (z + 0.15) * u, r))  # the axis at the row's middle
            lean.append(u[:2])
    ra, rb = np.array(ra), np.array(rb)
    f = support_tree(scans, ra, rb, rr, np.full(len(ra), -1), params=PARAMS, threads=2)
    assert (f["flags"] == 0).all() and (f["n_runs"] == 1).all() and (f["n_scans"] == 3).all()
    for k, (p, r) in enumerate(truth):
        mid = 0.5 * (f["a"][k] + f["b"][k])
        assert abs(np.log(f["r"][k] / r)) < 0.05, (k, f["r"][k], r)
        assert np.hypot(*(mid - p)[:2]) < 0.005, (k, mid, p)
        # the lean, along the row's frame, against the truth's
        t = f["lean"][k] @ np.c_[f["e1"][k], f["e2"][k]].T
        assert np.allclose(t[:2], lean[k], atol=0.01), (k, t, lean[k])
    assert (f["sd_log_r"] < 0.2).all() and (np.isnan(f["chi2"]) | (f["chi2"] < 9)).all() and (f["chi2_front"] < 9).all()


def test_a_bent_stem_is_a_chain_of_straight_pieces(tmp_path):
    # a 3 cm stem bending along an arc of radius 2 m (2 cm solids), claimed by the chords of two 0.6 m rows: each
    # row is three pieces, each nearly straight, so no hull, and the row runs along the bend
    rho, r = 2.0, 0.03
    z = np.arange(0.0, 1.62, 0.02)
    x = rho - np.sqrt(rho ** 2 - z ** 2)
    pts = np.c_[x, np.zeros_like(z), z]
    scans = scanned(tmp_path, pts[:-1], pts[1:], np.full(len(z) - 1, r), [(0.3, -8.0), (6.5, 5.0), (-6.0, 5.5)],
                    zenith=(75.0, 110.0))
    z0 = np.array([0.2, 0.8])
    a = np.c_[rho - np.sqrt(rho ** 2 - z0 ** 2), np.zeros(2), z0]
    b = np.c_[rho - np.sqrt(rho ** 2 - (z0 + 0.6) ** 2), np.zeros(2), z0 + 0.6]
    f = support_tree(scans, a, b, [1.5 * r] * 2, [-1, 0], params=PARAMS, threads=2)
    p = f["pieces"]
    assert (f["flags"] == 0).all() and (p["flags"] == 0).all() and list(p["row"]) == [0, 0, 0, 1, 1, 1]
    assert np.abs(np.log(p["r"] / r)).max() < 0.05 and np.abs(np.log(f["r"] / r)).max() < 0.05
    # each piece's middle lies on the arc, though the chord is up to 2.2 cm off it
    for k in range(len(p["r"])):
        mid = 0.5 * (p["a"][k] + p["b"][k])
        assert abs(mid[0] - (rho - np.sqrt(rho ** 2 - mid[2] ** 2))) < 0.003, (k, mid)
    assert (p["p_wood"] > 0.99).all() and (f["p_wood"] > 0.99).all()


def test_thin_stems_far_away_are_bounded_around_their_truth(tmp_path):
    # stems of 2 to 8 mm seen from two scans 20 m away (a footprint of ~14 mm): where no circle is determined the
    # radius is the prior cut by the bounds, and both the bounds and the posterior contain the truth
    radii = np.array([0.002, 0.003, 0.005, 0.008])
    xs = np.arange(4) * 0.3 - 0.45
    scans = scanned(tmp_path, np.c_[xs, np.zeros(4), np.zeros(4)], np.c_[xs, np.zeros(4), np.full(4, 1.8)], radii,
                    [(0.3, -20.0), (14.0, 14.0)], zenith=(75.0, 110.0))
    a = np.c_[xs + 0.3 * radii, np.zeros(4), np.full(4, 0.75)]
    f = support_tree(scans, a, a + [0, 0, 0.3], 1.5 * radii, np.full(4, -1), params=PARAMS, threads=2)
    bounded = (f["flags"] & (SupportFlag.BOUNDS | SupportFlag.POROUS)) > 0
    assert bounded.sum() >= 2 and not (f["flags"] & (SupportFlag.RUNS | SupportFlag.HULL)).any()
    assert np.isnan(f["r_row"][bounded]).all() and np.isfinite(f["r_hi"][bounded]).all()
    assert ((f["r_lo"][bounded] <= radii[bounded]) & (radii[bounded] <= f["r_hi"][bounded])).all(), (f["r_lo"], f["r_hi"])
    lo, hi = f["r"] * np.exp(-1.96 * f["sd_log_r"]), f["r"] * np.exp(1.96 * f["sd_log_r"])
    assert ((lo <= radii) & (radii <= hi)).all() and (f["p_wood"] > 0.9).all()


def test_a_clump_of_stems_is_flagged_as_runs(tmp_path):
    # three solid stems inside one envelope claim (as joint_tree's clump test): the profiles show them apart
    stems = np.array([[0.0, 0.0, 0.045], [0.075, 0.025, 0.04], [0.02, 0.085, 0.035]])
    scans = scanned(tmp_path, np.c_[stems[:, :2], np.zeros(3)], np.c_[stems[:, :2], np.full(3, 1.6)], stems[:, 2],
                    [(8.0, -1.0), (-5.5, 7.0), (1.5, -9.5)])
    centre = stems[:, :2].mean(0)
    a, b = np.array([[*centre, 0.5], [*centre, 0.8]]), np.array([[*centre, 0.8], [*centre, 1.1]])
    f = support_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=PARAMS)
    assert ((f["flags"] & SupportFlag.RUNS) > 0).all() and (f["n_runs"] >= 2).all() and f["a"].shape == (2, 3)
    # a flagged row has no summary of its own, and its radius is the prior's: loose
    assert np.isnan(f["r_row"]).all() and np.isnan(f["sd_r_row"]).all() and (f["sd_log_r"] > 0.1).all()
    with pytest.raises(ValueError, match="version"):
        support_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=dataclasses.replace(PARAMS, version=0))
    with pytest.raises(ValueError, match="parent"):
        support_tree(scans, a, b, [0.093, 0.093], [0, 0], params=PARAMS)
    with pytest.raises(TypeError):
        support_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=dataclasses.asdict(PARAMS))
