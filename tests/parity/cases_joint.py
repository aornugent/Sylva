"""Parity cases for sylva.qsm's joint model: the pair table.

They replace 3dtrees' numpy oracle (research 100 §21), which they were recorded against: a change is a recorded
change, in its own commit.
"""

import tempfile
from pathlib import Path

import numpy as np

from sylva import Shots
from sylva.qsm import ScanIndex, joint_table


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


CASES = {"table": table}
