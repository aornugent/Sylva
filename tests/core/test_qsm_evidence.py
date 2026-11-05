import numpy as np
import pytest

from sylva import Shots
from sylva.qsm import ShotsIndex, evidence


def fan(tmp_path, r=0.05, misses=True):
    """Pulses from the origin on a fine grid towards a vertical cylinder of radius r at (0, 5), wall at y = 8."""
    az, el = np.meshgrid(np.arange(-0.03, 0.03, 0.0005), np.arange(-0.05, 0.05, 0.0005))
    zen, az = np.pi / 2 - el.ravel(), az.ravel()
    d = np.c_[np.sin(zen) * np.sin(az), np.sin(zen) * np.cos(az), np.cos(zen)]
    h = np.hypot(d[:, 0], d[:, 1])
    tca = 5.0 * d[:, 1] / h
    d2 = 25.0 - tca ** 2
    hit = (d2 <= r * r) & (np.abs(d[:, 2] * tca / h) <= 1.0)
    rng = np.where(hit, (tca - np.sqrt(np.maximum(r * r - d2, 0))) / h, 8.0 / d[:, 1])
    has = hit | misses | (np.abs(el.ravel()) < 0.04)  # without misses, the wall's top and bottom are lost
    shots = Shots(origin=np.zeros((len(d), 3)), direction=d, echo_start=np.cumsum(np.r_[0, has[:-1]]),
                  echo_count=has.astype(np.uint32), echo_range=rng[has])
    if not misses:
        shots = shots.subset(has)
    path = tmp_path / f"scan_{r}_{misses}.parquet"
    shots.save(path)
    return ShotsIndex(path, misses), np.array([[0.0, 5.0, -1.0]]), np.array([[0.0, 5.0, 1.0]])


def test_a_measured_cylinder_gets_its_radius_a_calliper_and_an_audit(tmp_path):
    scan, a, b = fan(tmp_path)
    assert scan.misses and scan.n_pulses > 10_000 and scan.origin == (0.0, 0.0, 0.0)
    e = evidence([scan], a, b, r_claim=[0.075])
    assert np.exp(e["mu"][0]) == pytest.approx(0.05, rel=0.05)
    assert e["tau"][0] > 0 and e["lo"][0] < 0.055 and e["hi"][0] > 0.045
    assert e["n_hit"][0] > 100 and e["n_cross"][0] > e["n_hit"][0]
    assert e["n_contra"][0] > 0  # passes inside a 50 % fat claim
    assert evidence([scan], a, b, r_claim=[0.05])["n_contra"][0] == 0


def test_thin_wood_without_misses_rests_on_the_prior(tmp_path):
    scan, a, b = fan(tmp_path, r=0.01, misses=False)
    e = evidence([scan], a, b)
    assert not scan.misses and e["n_hit"][0] > 0 and e["tau"][0] == 0


def test_inputs_are_checked(tmp_path):
    scan, a, b = fan(tmp_path)
    with pytest.raises(ValueError):
        evidence([scan], a, np.r_[b, b])
    with pytest.raises(ValueError):
        evidence([scan], a, b, r_claim=[0.05, 0.05])
