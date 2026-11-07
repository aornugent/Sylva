import dataclasses
import json

import numpy as np
import pytest

from sylva import Shots
from sylva.qsm import EvidenceParams, Reason, ScanIndex, contradicted, evidence, pairs


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
