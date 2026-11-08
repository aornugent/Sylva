"""Parity cases for sylva.qsm's joint model: the per-pulse likelihood and the pair table.

They replace 3dtrees' numpy oracle of both (research 74 §21), which they were recorded against: a change to either is
a recorded change, in its own commit.
"""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

from sylva import Shots, _core
from sylva.qsm import JointParams, ScanIndex, joint_table

BENCH = JointParams(tau=0.17, edge=0.25, hide=0.0, mu_fork=0.0, centre_rel=0.3, scan_rel=0.05)


def _pairs(seed, m=3000):
    """Random pairs: stops and passes in chords at 3-30 m, early echoes, deleted echoes in reach, selection."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(3.0, 30.0, m)
    half = rng.uniform(0.04, 0.3, m)
    ang = rng.uniform(0, 2 * np.pi, m)
    stop = rng.uniform(size=m) < 0.5
    c0, c1 = t - half, t + half
    return dict(b=rng.uniform(-0.12, 0.12, m), t=t, sin=rng.uniform(0.2, 1.0, m), jb=np.c_[np.cos(ang), np.sin(ang)],
                jt=rng.normal(0, 1.5, (m, 2)), c0=c0, c1=c1, x=np.where(stop, rng.uniform(c0, c1), 0.0),
                q=np.where(rng.uniform(size=m) < 0.3, rng.uniform(0.2, 1.0, m), 1.0), stop=stop,
                early=rng.uniform(size=m) < 0.1, deleted=rng.uniform(size=m) < 0.15)


def _states(seed, m):
    """Per-pair states: small offsets, radii from below an echo's capacity (0.5 mm) to trunks, foliage around 1/m."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.normal(0, 0.01, (m, 2)), np.log(rng.uniform(0.0005, 0.15, m)), rng.normal(0, 1, m)]


def loglik():
    pairs = _pairs(1)
    state = _states(2, len(pairs["t"]))
    out = {}
    for name, p in (("default", JointParams()), ("bench", BENCH), ("rough", dataclasses.replace(BENCH, rough=0.3))):
        out[name] = _core.qsm_joint_loglik(pairs, state, dataclasses.asdict(p), 1)
    return out


def _fan(path, r=0.05):
    """Pulses from the origin on a grid towards a vertical cylinder of radius r at (0, 5), a wall at y = 8; every
    other echo counted as one of two returns, the stem's as the second (the first deleted, before it), the wall's as
    the first (the second deleted, beyond it)."""
    az, el = np.meshgrid(np.arange(-0.03, 0.03, 0.0005), np.arange(-0.05, 0.05, 0.0005))
    zen, az = np.pi / 2 - el.ravel(), az.ravel()
    d = np.c_[np.sin(zen) * np.sin(az), np.sin(zen) * np.cos(az), np.cos(zen)]
    h = np.hypot(d[:, 0], d[:, 1])
    tca = 5.0 * d[:, 1] / h
    d2 = 25.0 - tca ** 2
    hit = (d2 <= r * r) & (np.abs(d[:, 2] * tca / h) <= 1.0)
    rng = np.where(hit, (tca - np.sqrt(np.maximum(r * r - d2, 0))) / h, 8.0 / d[:, 1])
    has = hit | (np.abs(el.ravel()) < 0.045)
    rng = rng[has]
    lossy = np.arange(len(rng)) % 2 == 0
    nr = np.where(lossy, 2, 1).astype(np.uint8)
    rn = np.where(lossy & (rng < 6.0), 2, 1).astype(np.uint8)
    shots = Shots(origin=np.zeros((len(d), 3)), direction=d, echo_start=np.cumsum(np.r_[0, has[:-1]]),
                  echo_count=has.astype(np.uint32), echo_range=rng,
                  echo_attrs={"las_number_of_returns": nr, "las_return_number": rn})
    shots.save(path)


def table():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fan.parquet"
        _fan(path)
        scan = ScanIndex(path, threads=1)
        a = np.array([[0.0, 5.0, -1.0], [0.06, 5.0, -1.0]])
        b = np.array([[0.0, 5.0, 1.0], [0.06, 5.0, 1.0]])
        t = joint_table([scan], a, b, window=[0.15, 0.09], threads=1)
    return {k: v for k, v in t.items() if k != "scan"}


CASES = {"loglik": loglik, "table": table}
