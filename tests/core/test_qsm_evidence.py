import dataclasses
import json

import numpy as np
import pytest

from sylva import Shots
from sylva import PointCloud
from sylva.qsm import (EvidenceParams, JointParams, JointReason, Reason, ScanIndex, contradicted, evidence, joint_fit,
                       joint_table, joint_tree, pairs)


def fan(tmp_path, r=0.05, misses=True):
    """Pulses from the origin on a fine grid towards a vertical cylinder of radius r at (0, 5), wall at y = 8.

    With misses, the pulses over and under the wall are kept without echoes; without, the wall's top and
    bottom are lost too."""
    az, el = np.meshgrid(np.arange(-0.03, 0.03, 0.0005), np.arange(-0.05, 0.05, 0.0005))
    zen, az = np.pi / 2 - el.ravel(), az.ravel()
    d = np.c_[np.sin(zen) * np.sin(az), np.sin(zen) * np.cos(az), np.cos(zen)]
    h = np.hypot(d[:, 0], d[:, 1])
    tca = 5.0 * d[:, 1] / h
    d2 = 25.0 - tca ** 2
    hit = (d2 <= r * r) & (np.abs(d[:, 2] * tca / h) <= 1.0)
    rng = np.where(hit, (tca - np.sqrt(np.maximum(r * r - d2, 0))) / h, 8.0 / d[:, 1])
    has = hit | (np.abs(el.ravel()) < (0.045 if misses else 0.04))
    shots = Shots(origin=np.zeros((len(d), 3)), direction=d, echo_start=np.cumsum(np.r_[0, has[:-1]]),
                  echo_count=has.astype(np.uint32), echo_range=rng[has])
    if not misses:
        shots = shots.subset(has)
    path = tmp_path / f"scan_{r}_{misses}.parquet"
    shots.save(path)
    return ScanIndex(path, threads=2), np.array([[0.0, 5.0, -1.0]]), np.array([[0.0, 5.0, 1.0]])


def test_a_measured_cylinder_gets_its_radius_a_calliper_and_an_audit(tmp_path):
    scan, a, b = fan(tmp_path)
    assert scan.misses and scan.n_pulses > 10_000 and scan.nbytes > 16 * scan.n_pulses
    e = evidence([scan], a, b, r_claim=[0.075])
    assert np.exp(e["mu"][0]) == pytest.approx(0.05, rel=0.05)
    assert e["tau"][0] > 0 and e["reason"][0] == Reason.MEASURED and e["lo"][0] < 0.055 and e["hi"][0] > 0.045
    assert e["n_hit"][0] > 100 and e["n_cross"][0] > e["n_hit"][0]
    assert e["n_contra"][0] > 0 and contradicted(e)[0]  # passes inside a 50 % fat claim
    honest = evidence([scan], a, b, r_claim=[0.05])
    assert honest["n_contra"][0] == 0 and not contradicted(honest)[0]


def test_pairs_hold_each_pulse_that_reached_the_disc_with_its_echoes(tmp_path):
    scan, a, b = fan(tmp_path)
    assert np.allclose(scan.origin, 0)
    p = pairs([scan], np.r_[a, a], np.r_[b, a], window=0.15)
    n = len(p["t"])
    assert list(p["cyl_start"]) == [0, n, n] and n > 200  # a cylinder without length has none
    assert p["dir"].shape == (n, 3) and len(p["echo_start"]) == n + 1
    assert np.all(p["d"] < 0.15) and np.all((p["s"] >= 0) & (p["s"] <= 2))
    first = p["range"][p["echo_start"][:-1][np.diff(p["echo_start"]) > 0]]
    on_stem = np.abs(first - p["t"][np.diff(p["echo_start"]) > 0]) < 0.06
    assert on_stem.sum() > 100 and (~on_stem).sum() > 100  # hits on the stem and passes to the wall
    assert np.isnan(p["reflectance"]).all()
    with pytest.raises(ValueError):
        pairs([scan], a, b, window=0.0)
    q = pairs([scan], np.r_[a, a], np.r_[b, b], window=[0.15, 0.08])  # each cylinder its own tube
    n0 = q["cyl_start"][1]
    assert q["cyl_start"][2] - n0 < n0 and np.all(q["d"][n0:] < 0.08)


def test_pairs_say_which_pulses_lost_echoes_to_the_export(tmp_path):
    # the fan's pulses, every other one counted 2 returns by the scanner but holding 1 echo
    scan, a, b = fan(tmp_path)
    s = Shots.load(tmp_path / "scan_0.05_True.parquet")
    nr = np.ones(s.n_echoes, np.uint8)
    nr[::2] = 2
    lossy = Shots(origin=s.origin, direction=s.direction, echo_start=s.echo_start, echo_count=s.echo_count,
                  echo_range=s.echo_range, echo_attrs={"las_number_of_returns": nr})
    lossy.save(tmp_path / "lossy.parquet")
    p = pairs([ScanIndex(tmp_path / "lossy.parquet", threads=2)], a, b, window=0.15)
    has = np.diff(p["echo_start"]) > 0
    assert p["lost"][has].any() and not p["lost"][has].all()
    assert set(np.unique(p["returns"][has])) == {1, 2} and (p["return_number"] == 0).all()
    assert not pairs([scan], a, b, window=0.15)["lost"].any()  # a store without the attribute loses nothing


def test_a_deleted_echo_is_flagged_only_where_it_could_lie(tmp_path):
    # the fan, every pulse counted 2 returns: on the stem, the recorded echo is the 2nd (the 1st, deleted, lay
    # before it); on the wall, the 1st (the deleted one lay beyond the wall, past the stem's tube)
    scan, a, b = fan(tmp_path)
    s = Shots.load(tmp_path / "scan_0.05_True.parquet")
    stem = s.echo_range < 6.0
    two = Shots(origin=s.origin, direction=s.direction, echo_start=s.echo_start, echo_count=s.echo_count,
                echo_range=s.echo_range, echo_attrs={"las_number_of_returns": np.full(s.n_echoes, 2, np.uint8),
                                                     "las_return_number": np.where(stem, 2, 1).astype(np.uint8)})
    two.save(tmp_path / "two.parquet")
    lossy = ScanIndex(tmp_path / "two.parquet", threads=2)
    full, t = joint_table([scan], a, b, window=0.15), joint_table([lossy], a, b, window=0.15)
    assert len(t["b"]) == len(full["b"]) and not full["deleted"].any()
    assert t["deleted"][t["stop"]].all()  # the stem's stops: an earlier echo was deleted
    assert not t["deleted"][~t["stop"]].any()  # the wall's deleted echoes lie beyond every chord: certain passes
    # without return numbers the deleted echo could lie anywhere: every pulse with echoes is flagged, misses aren't
    nonum = Shots(origin=s.origin, direction=s.direction, echo_start=s.echo_start, echo_count=s.echo_count,
                  echo_range=s.echo_range, echo_attrs={"las_number_of_returns": np.full(s.n_echoes, 2, np.uint8)})
    nonum.save(tmp_path / "nonum.parquet")
    u = joint_table([ScanIndex(tmp_path / "nonum.parquet", threads=2)], a, b, window=0.15)
    assert (u["deleted"] == (u["n_echo"] > 0)).all() and (u["n_echo"] == 0).any()


def test_thin_wood_without_misses_rests_on_the_prior(tmp_path):
    scan, a, b = fan(tmp_path, r=0.01, misses=False)
    e = evidence([scan], a, b)
    assert not scan.misses and e["n_hit"][0] > 0 and e["tau"][0] == 0 and e["reason"][0] == Reason.ECHOES_ONLY


def test_inputs_are_checked(tmp_path):
    scan, a, b = fan(tmp_path)
    for bad in [dict(b=np.r_[b, b]), dict(a=a.ravel(), b=b.ravel()), dict(r_claim=[0.05, 0.05]), dict(r_claim=[[0.05]])]:
        with pytest.raises(ValueError):
            evidence(**{"scans": [scan], "a": a, "b": b, **bad})
    with pytest.raises(TypeError):
        evidence(["scan.parquet"], a, b)
    with pytest.raises(TypeError):
        evidence([scan], a, b, params={"window_cap": 0.1})
    with pytest.raises(ValueError, match="window_min"):
        evidence([scan], a, b, params=EvidenceParams(window_min=0.3))
    with pytest.raises(ValueError, match="version"):
        evidence([scan], a, b, params=EvidenceParams(version=0))


def test_the_settings_are_one_record_from_the_core(tmp_path):
    p = EvidenceParams()
    assert json.loads(json.dumps(dataclasses.asdict(p))) == dataclasses.asdict(p)
    assert (p.beam_divergence, p.beam_exit, p.range_noise, p.window_cap) == (0.35e-3, 0.007, 0.003, 0.25)
    scan, a, b = fan(tmp_path)
    default, wide = evidence([scan], a, b), evidence([scan], a, b, params=EvidenceParams(beam_divergence=0.002))
    assert wide["tau"][0] < default["tau"][0]


def test_a_neighbours_echoes_are_not_counted(tmp_path):
    scan, a, b = fan(tmp_path)
    alone = evidence([scan], a, b)
    e = evidence([scan], np.r_[a, a + [0.12, 0, 0]], np.r_[b, b + [0.12, 0, 0]])  # an empty axis beside it
    assert e["n_hit"][1] == 0 and e["hi"][1] < 0.02 and e["reason"][1] == Reason.NO_HITS
    assert e["n_hit"][0] == alone["n_hit"][0] and e["mu"][0] == alone["mu"][0]


def test_wood_thinner_than_the_beam_rests_on_the_prior(tmp_path):
    scan, a, b = fan(tmp_path, r=0.002)
    e = evidence([scan], a, b)
    assert scan.misses and e["n_hit"][0] > 0 and e["tau"][0] == 0 and e["reason"][0] == Reason.FOOTPRINT


def test_a_cylinder_without_length_gets_no_evidence(tmp_path):
    scan, a, b = fan(tmp_path)
    dot = np.array([[0.0, 4.95, 0.0]])  # on the near surface, where the echoes are
    e = evidence([scan], np.r_[dot, a], np.r_[dot, b], r_claim=[0.05, 0.05])
    assert np.isnan(e["mu"][0]) and e["tau"][0] == 0 and e["reason"][0] == Reason.NO_HITS
    assert e["n_hit"][0] == e["n_cross"][0] == e["n_contra"][0] == 0
    assert e["mu"][1] == evidence([scan], a, b)["mu"][0]


JOINT = JointParams(tau=0.17, edge=0.25, hide=0.0, mu_fork=0.0, centre_rel=0.3, scan_abs=0.0, scan_rel=0.0)


def test_the_joint_fit_brings_a_fat_claim_down_to_the_stem(tmp_path):
    scan, a, b = fan(tmp_path)
    t = joint_table([scan], a, b, window=0.15)
    assert len(t["b"]) > 200 and t["stop"].any() and (~t["stop"]).any() and t["jb"].shape == (len(t["b"]), 2)
    f = joint_fit(t, a, b, r_prior=[0.08], parent=[-1], n_scans=1, params=JOINT, r_max=[0.15])
    assert f["converged"] and f["exact"] and np.exp(f["theta"][0, 2]) == pytest.approx(0.05, abs=0.003)
    assert f["profile"].shape == (1, 21) and f["hess"].shape == (1, 4, 4) and f["rho_local"][0] > 0.9
    with pytest.raises(ValueError, match="sig_eta"):
        joint_fit(t, a, b, r_prior=[0.08], parent=[-1], n_scans=1, params=dataclasses.replace(JOINT, sig_eta=0.0))
    with pytest.raises(ValueError, match="version"):
        joint_fit(t, a, b, r_prior=[0.08], parent=[-1], n_scans=1, params=dataclasses.replace(JOINT, version=0))
    with pytest.raises(TypeError):
        joint_fit(t, a, b, r_prior=[0.08], parent=[-1], n_scans=1, params=dataclasses.asdict(JOINT))


def test_a_clump_of_stems_comes_out_as_its_stems(tmp_path):
    # three solid stems inside one envelope claim, scanned from three sides: each becomes a row of its own
    import sylva.synthetic as syn
    stems = np.array([[0.0, 0.0, 0.045], [0.075, 0.025, 0.04], [0.02, 0.085, 0.035]])
    cloud = PointCloud(np.c_[stems[:, :2], np.full(3, 0.8)], {"classification": np.full(3, 5, np.uint8)})
    solids = dict(a=np.c_[stems[:, :2], np.zeros(3)], b=np.c_[stems[:, :2], np.full(3, 1.6)], r=stems[:, 2],
                  point=np.arange(3))
    scans = []
    for k, (x, y) in enumerate([(8.0, -1.0), (-5.5, 7.0), (1.5, -9.5)]):
        shots = syn.scan(cloud, origin=(x, y, 1.5), resolution_deg=0.04, scanner="vz400i", mixed_pixels=True,
                         target_radius=0.01, min_zenith_deg=80.0, max_zenith_deg=100.0, seed=k, solids=solids)
        shots.save(tmp_path / f"s{k}.parquet")
        scans.append(ScanIndex(tmp_path / f"s{k}.parquet", threads=2))
    centre = stems[:, :2].mean(0)
    a, b = np.array([[*centre, 0.5], [*centre, 0.8]]), np.array([[*centre, 0.8], [*centre, 1.1]])
    params = JointParams(tau=0.17, edge=0.25, hide=0.0, mu_fork=0.0, centre_rel=0.3, scan_rel=0.05)
    f = joint_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=params)  # the core's own rule: by the evidence
    assert list(f["src"]) == [0, 0, 0, 1, 1, 1] and list(f["parent"][3:]) != [-1, -1, -1]
    assert (f["split_gain"] > 100).all()  # the stems beat the envelope by far on the same pulses
    for row in (0, 1):
        mine = f["src"] == row
        centres = f["a"][mine, :2]
        for x, y, r in stems:
            k = np.argmin(np.hypot(*(centres - [x, y]).T))
            assert np.hypot(*(centres[k] - [x, y])) < 0.01 and abs(np.log(f["r"][mine][k] / r)) < 0.12
    assert (f["p_wood"] > 0.99).all() and (f["reason"] == JointReason.MEASURED).all() and f["converged"]
    whole = joint_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=params, split=[False, False])
    assert list(whole["src"]) == [0, 1] and (whole["r"] > 0.06).all()  # as claimed, the envelope stays (~ the stems' area)
    with pytest.raises(ValueError, match="version"):
        joint_tree(scans, a, b, [0.093, 0.093], [-1, 0], params=dataclasses.replace(params, version=0))
