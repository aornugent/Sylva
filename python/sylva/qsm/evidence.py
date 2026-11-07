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
ones of interest (a whole tile): each echo goes to the one cylinder whose
surface it lies nearest. :func:`pairs` returns the per-pulse table these
come from, for a likelihood fitted outside.
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from os import PathLike

import numpy as np

from .. import _core

_DEFAULTS = _core.qsm_evidence_defaults()


@dataclasses.dataclass(frozen=True)
class EvidenceParams:
    """The settings of :func:`evidence`; the defaults are the calibrated ones, for a RIEGL VZ-400i.

    A plain record: ``dataclasses.asdict(params)`` is JSON-serialisable and,
    with ``version``, is what a cache of evidence should key on.
    """

    #: Bumped by Sylva whenever the evidence changes for the same settings; a record of another version is an error.
    version: int = _DEFAULTS["version"]
    #: Full beam divergence (rad).
    beam_divergence: float = _DEFAULTS["beam_divergence"]
    #: Beam diameter at the aperture (m).
    beam_exit: float = _DEFAULTS["beam_exit"]
    #: Ranging precision (m).
    range_noise: float = _DEFAULTS["range_noise"]
    #: Log-radius error per scan that more echoes don't remove (registration, wind).
    floor_log: float = _DEFAULTS["floor_log"]
    #: Systematic log sd per unit of ``(w / 2) / r`` (echoes sit outside the surface).
    footprint_sd: float = _DEFAULTS["footprint_sd"]
    #: Systematic log sd per unit of ``spread / r`` (clutter around the axis).
    spread_sd: float = _DEFAULTS["spread_sd"]
    #: Below this radius (m), wood seen only by echoes-only scans rests on the prior.
    echoes_only_min_radius: float = _DEFAULTS["echoes_only_min_radius"]
    #: Wood whose echoes sit within this many beam footprints of the axis rests on the prior.
    min_footprints: float = _DEFAULTS["min_footprints"]
    #: The widest window (m): no echo farther from an axis is a candidate hit, and no
    #: cylinder farther than this from an echo competes for it.
    window_cap: float = _DEFAULTS["window_cap"]
    #: The narrowest window (m).
    window_min: float = _DEFAULTS["window_min"]
    #: Where passes bound the calliper, the next window is ``max(window_hi_factor * hi, hi + window_hi_margin)``.
    window_hi_factor: float = _DEFAULTS["window_hi_factor"]
    window_hi_margin: float = _DEFAULTS["window_hi_margin"]
    #: Otherwise it is ``window_r_factor * r + window_r_margin``.
    window_r_factor: float = _DEFAULTS["window_r_factor"]
    window_r_margin: float = _DEFAULTS["window_r_margin"]
    #: Rounds of assignment and windowing.
    rounds: int = _DEFAULTS["rounds"]
    #: Cylinders shorter than this (m) take no echoes and get no evidence.
    min_length: float = _DEFAULTS["min_length"]


#: Why a cylinder's ``tau`` is what it is: :func:`evidence`'s ``reason``.
Reason = IntEnum("Reason", _core.qsm_evidence_reasons())


class ScanIndex:
    """One scan's pulses, read from a single-position Shots file and indexed by direction.

    Built once per scan and reused by every :func:`evidence` and :func:`pairs` call (about 16
    bytes per pulse and 4 per echo, 8 where the store has reflectance and
    deviation, held on the Rust side). Only the angles, the echo ranges and
    those two attributes are read.

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

    The raw table a per-pulse likelihood reads, which :func:`evidence`
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
    - ``echo_start`` ``(m + 1,)``: row ``i``'s echoes, all of its pulse's,
      nearest first, are ``echo_start[i]:echo_start[i + 1]`` of ``range``,
      ``reflectance`` (dB) and ``deviation`` (NaN where the store has none).
    """
    scans = list(scans)
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("scans must be ScanIndex objects")
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape:
        raise ValueError(f"a and b must both have shape (n, 3), got {a.shape} and {b.shape}")
    window = np.ascontiguousarray(np.broadcast_to(np.asarray(window, np.float64), (len(a),)))
    return _core.qsm_pairs([s._core for s in scans], a, b, window, int(threads))


def evidence(scans, a, b, r_claim=None, params: EvidenceParams = EvidenceParams(), threads: int = 0) -> dict[str, np.ndarray]:
    """The evidence the rays of ``scans`` give about cylinders ``a[i] -> b[i]``.

    Ray first: the pulses passing within ``params.window_cap`` (25 cm) of each
    axis are enumerated once; then, over a few rounds, each echo goes to the
    one cylinder whose surface it lies nearest among those whose window holds
    it (a pulse hits a cylinder at most once), and each window is set from
    the cylinder's own calliper (``max(1.5 hi, hi + 2 cm)``, else
    ``2 r + 2 cm``, within 2-25 cm), so a claim can't move it. A pulse in the
    window whose echo near this axis went to a neighbour or a fork is neither
    a hit nor a pass. Radii above the cap are not measured. Per cylinder:

    - ``mu``: log of the median distance of its hits from the axis (NaN without hits);
    - ``tau``: precision of ``mu``. Each hit is worth ``(r / sigma)^2``, ``sigma``
      half the beam footprint at its range combined with the ranging noise; each
      scan's sum is capped by a floor (registration, wind), scans add, and the
      footprint and clutter spread add systematic variance. 0 without hits, or
      where the rays don't determine the radius (``reason``);
    - ``reason`` (:class:`Reason`): ``MEASURED`` (``tau > 0``); ``NO_HITS``
      (also cylinders shorter than ``min_length``); and three prior-only
      cases, with hits but ``tau == 0``: ``CLUTTER`` (pulses passed closer to
      the axis than its echoes sit), ``FOOTPRINT`` (its echoes sit within
      ``min_footprints`` beam footprints of the axis), ``ECHOES_ONLY``
      (thinner than ``echoes_only_min_radius`` and seen by no scan that kept
      misses). So ``(tau == 0) & (n_hit > 0)`` is "prior only";
    - ``lo``, ``hi``: the beam calliper, from pulses in the window that hit
      (``d - w/2``) or passed (``d + w/2``);
    - ``n_hit``: the hits ``mu`` is the median of; ``n_cross``: pulses whose
      line comes closest to the axis within the segment and the window, and
      hit or passed (a hit near the ends can count in ``n_hit`` alone, so it
      may exceed ``n_cross``); ``n_contra``: passes inside ``r_claim``, if
      given (see :func:`contradicted`);
    - ``n_core``, ``n_pen``: penetration at the measured radius ``r``. Of the
      pulses aimed at its core (closer to the axis than ``r - band``,
      ``band = w/2 + max(1 cm, 0.3 r)`` for axis offsets) that reached ``r``,
      ``n_pen`` went through or stopped deep inside it. A solid forbids that
      and foliage allows it, so ``n_pen / n_core`` is high where the hits are
      leaves around a thinner axis;
    - ``reflectance``, ``deviation``: the median of its hits' (dB; RIEGL
      deviation), NaN without hits or where the scans' stores lack them.
      Leaves return weaker, wider pulses than wood.

    The evidence for a cylinder depends on every cylinder that competes for
    its echoes: those within ``params.window_cap`` of an echo it could take,
    so within ``2 * params.window_cap`` of its axis. Pass a whole tile; or,
    for one tree, the tree and every cylinder within ``2 * window_cap`` of
    it, which gives the tree's cylinders' evidence up to how the neighbours'
    own windows settle (their neighbours are missing).

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b
        ``(n, 3)`` cylinder axes.
    r_claim
        ``(n,)`` claimed radii to audit, or None.
    params
        :class:`EvidenceParams`.
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict of ``(n,)`` arrays: mu, tau, lo, hi, n_hit, n_cross, n_contra, reason,
    n_core, n_pen, reflectance, deviation.
    """
    scans = list(scans)
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("scans must be ScanIndex objects")
    if not isinstance(params, EvidenceParams):
        raise TypeError("params must be an EvidenceParams")
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape:
        raise ValueError(f"a and b must both have shape (n, 3), got {a.shape} and {b.shape}")
    if r_claim is not None:
        r_claim = np.ascontiguousarray(r_claim, np.float64)
        if r_claim.shape != (len(a),):
            raise ValueError(f"r_claim must have shape ({len(a)},), got {r_claim.shape}")
    return _core.qsm_evidence([s._core for s in scans], a, b, r_claim, dataclasses.asdict(params), int(threads))


def contradicted(ev: dict[str, np.ndarray], min_count: int = 2, fraction: float = 0.05) -> np.ndarray:
    """Whether the passes contradict each claimed radius: ``n_contra >= max(min_count, fraction * n_cross)``.

    A tolerant audit of a fat claim on :func:`evidence` run with ``r_claim``:
    off-centre fitted axes leave honest claims a few passes inside. The
    verdict is the caller's.

    Parameters
    ----------
    ev
        The output of :func:`evidence` run with ``r_claim``.
    min_count
        Contradicting passes always needed.
    fraction
        Contradicting passes needed as a share of the crossings.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool, one per cylinder.
    """
    return ev["n_contra"] >= np.maximum(min_count, fraction * ev["n_cross"])
