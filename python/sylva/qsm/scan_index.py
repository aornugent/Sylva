# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""A scan's pulses indexed by direction, and every pulse near each cylinder with all its echoes.

:class:`ScanIndex` is built once per scan and read by :func:`pairs`,
:func:`~sylva.qsm.joint_table`, :func:`~sylva.qsm.support_tree` and
:func:`~sylva.qsm.evidence`; :func:`pairs` is the per-pulse table a
likelihood reads.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .. import _core


class ScanIndex:
    """One scan's pulses, read from a single-position Shots file and indexed by direction.

    Built once per scan and reused by every :func:`~sylva.qsm.evidence`, :func:`pairs` and
    :func:`~sylva.qsm.joint_table` call (about 20 bytes per pulse, 21 where the
    store has ``las_number_of_returns``, and 4 per echo, 8 where it has
    reflectance and deviation and 1 more with ``las_return_number``, held on
    the Rust side). Only the angles, the echo ranges and those four attributes
    are read.

    Parameters
    ----------
    path
        A Shots Parquet file holding one scan position.
    threads
        Row groups read at once (0: one thread per core).
    """

    def __init__(self, path: str | PathLike, threads: int = 0) -> None:
        self._core = _core.ScanIndex(path, int(threads))

    @property
    def misses(self) -> bool:
        """Whether the store kept pulses without echoes: whether any pulse has none.

        An echoes-only export (e.g. a filtered COPC) has none, and has lost
        echoes too; the calliper's upper bound is then incomplete and thin wood
        rests on the prior."""
        return self._core.misses

    @property
    def n_pulses(self) -> int:
        """The pulses the index holds, with or without echoes."""
        return self._core.n_pulses

    @property
    def nbytes(self) -> int:
        """The memory the index holds (bytes)."""
        return self._core.nbytes

    @property
    def origin(self) -> np.ndarray:
        """The scanner position, ``(3,)``."""
        return np.asarray(self._core.origin, np.float64)


def pairs(scans, a, b, window, threads: int = 0) -> dict[str, np.ndarray]:
    """Every pulse near each cylinder ``a[i] -> b[i]``, with all its echoes.

    The raw table a per-pulse likelihood reads, which :func:`~sylva.qsm.evidence`
    reduces inside. A row is a pulse whose line comes closest to the axis
    within the segment (ends included), closer than its cylinder's ``window``,
    ahead of the scanner, and that reached the tube of that radius around the axis
    (its last echo is at or beyond where its line enters the tube, or it
    missed): a pulse that stopped short says nothing about what is in the
    tube. Every row carries all its pulse's echoes, so whether it echoed
    before the tube (an echoes-only export then recorded it anyway) is
    ``range`` against that entry.

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b
        ``(n, 3)`` cylinder axes.
    window
        Radius (m) of the tube around each axis: one for all, or ``(n,)``.
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict of arrays; ``cyl_start`` ``(n + 1,)``, cylinder ``c``'s rows ``cyl_start[c]:cyl_start[c + 1]``:

    - ``scan``, ``pulse``: the scan (index into ``scans``) and its pulse;
    - ``dir`` ``(m, 3)``: the pulse's unit direction;
    - ``t``, ``s``, ``d``: closest approach of the pulse's line to the axis
      line, as the range along the pulse, the position along the axis from
      ``a``, and the distance between the lines;
    - ``lost``: the export deleted some of the pulse's echoes (the scanner
      counted more returns than the store holds; False where it doesn't say);
    - ``returns``: the pulse's number of returns as the scanner counted them
      (0 where the store doesn't say);
    - ``echo_start`` ``(m + 1,)``: row ``i``'s echoes, all of its pulse's,
      nearest first, are ``echo_start[i]:echo_start[i + 1]`` of ``range``,
      ``reflectance`` (dB) and ``deviation`` (NaN where the store has none) and
      ``return_number`` (0 where it has none; a number missing below
      ``returns`` is an echo the export deleted).
    """
    scans = list(scans)
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("scans must be ScanIndex objects")
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape:
        raise ValueError(f"a and b must both have shape (n, 3), got {a.shape} and {b.shape}")
    window = np.ascontiguousarray(np.broadcast_to(np.asarray(window, np.float64), (len(a),)))
    return _core.qsm_pairs([s._core for s in scans], a, b, window, int(threads))
