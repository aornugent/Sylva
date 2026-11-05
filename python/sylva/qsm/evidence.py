# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""What the scanner's rays say about each QSM cylinder, independent of its claimed radius.

For fitted cylinders (e.g. raycloudtools' ``_trees.txt``, whose radii are a
taper prior rather than a measurement), the rays of the scans give per
cylinder the log radius of its echoes with a calibrated precision, the beam
calliper bounding the radius from pulses that hit or passed close by, and an
audit count of passes inside a claimed radius. Pass every cylinder near the
ones of interest (a whole tile): each echo goes to its nearest axis.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .. import _core

#: RIEGL VZ-400i (and VZ-2000i) beam: full divergence (rad), diameter at the aperture (m), ranging precision (m).
VZ400I = (0.35e-3, 0.007, 0.003)


class ScanIndex:
    """One scan's pulses, read from a single-position Shots file and indexed by direction.

    Built once per scan and reused by every :func:`evidence` call (about 16
    bytes per pulse and 4 per echo, held on the Rust side). Only the angles
    and echo ranges are read.

    Parameters
    ----------
    path
        A Shots Parquet file holding one scan position.
    misses
        Whether the store kept pulses without echoes. An echoes-only export
        (e.g. a filtered COPC) has none, and has lost echoes too; the calliper's
        upper bound is then incomplete and thin wood rests on the prior. A file
        in which every pulse has an echo can't have kept misses: True is an error.
    threads
        Row groups read at once (0: one thread per core).
    """

    def __init__(self, path: str | PathLike, misses: bool, threads: int = 0) -> None:
        self._core = _core.ScanIndex(path, bool(misses), int(threads))

    @property
    def origin(self) -> tuple[float, float, float]:
        """The scanner position."""
        return self._core.origin

    @property
    def misses(self) -> bool:
        """Whether the store kept pulses without echoes."""
        return self._core.misses

    @property
    def n_pulses(self) -> int:
        return self._core.n_pulses


def evidence(scans, a, b, r_claim=None, beam=VZ400I, q_max: float = 0.5, threads: int = 0) -> dict[str, np.ndarray]:
    """The evidence the rays of ``scans`` give about cylinders ``a[i] -> b[i]``.

    Each echo goes to its nearest axis within ``q_max``. Per cylinder:

    - ``mu``: log of the median distance of its echoes from the axis (NaN without echoes);
    - ``tau``: precision of ``mu``. Each echo is worth ``(r / sigma)^2``, ``sigma``
      half the beam footprint at its range combined with the ranging noise; each
      scan's sum is capped by a floor (registration, wind), scans add, and the
      footprint and clutter spread add systematic variance (calibrated constants,
      set in the Rust core). 0 without echoes, or where the rays don't determine
      the radius: pulses passed closer to the axis than its echoes sit, or no
      scan that saw it recorded misses and it is thinner than 2 cm. So
      ``(tau == 0) & (n_hit > 0)`` is "prior only";
    - ``lo``, ``hi``: the beam calliper, from pulses that hit (``d - w/2``) or
      passed (``d + w/2``) within a window of three times its echo distance;
    - ``n_hit`` (echoes), ``n_cross`` (pulses that hit or passed) and
      ``n_contra`` (passes inside ``r_claim``, if given).

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b
        ``(n, 3)`` cylinder axes.
    r_claim
        ``(n,)`` claimed radii to audit, or None.
    beam
        ``(divergence, exit diameter, range noise)`` of the scanner.
    q_max
        An echo farther than this from every axis belongs to no cylinder (m).
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict of ``(n,)`` arrays: mu, tau, lo, hi, n_hit, n_cross, n_contra.
    """
    a, b = np.ascontiguousarray(a, np.float64).reshape(-1, 3), np.ascontiguousarray(b, np.float64).reshape(-1, 3)
    if len(a) != len(b):
        raise ValueError("a and b need one row per cylinder")
    if r_claim is not None:
        r_claim = np.ascontiguousarray(r_claim, np.float64).reshape(-1)
        if len(r_claim) != len(a):
            raise ValueError("r_claim needs one value per cylinder")
    return _core.qsm_evidence([s._core for s in scans], a, b, r_claim, tuple(float(v) for v in beam), float(q_max),
                              int(threads))
