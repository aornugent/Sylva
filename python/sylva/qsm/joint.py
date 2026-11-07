# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""A joint wood, foliage and ray model of a QSM, fitted to the scans' pulses.

Each cylinder (a row) has an axis offset across its axis, a radius and a
foliage density; each pulse near it is a censored survival event along its
line, where opaque wood (an echo at the footprint's centroid depth) and
foliage (Beer-Lambert) compete to stop it. A Gaussian Markov prior over the
tree ties the rows together: the taper's steps, soft Leonardo at forks, a
random walk of the centre and of foliage. :func:`joint_table` builds the
pulse-row pairs and :func:`joint_fit` fits the model (3dtrees research 74).

The parameters are a mapping of named numbers:

- the likelihood: ``exit`` and ``divergence`` (the beam's diameter at the
  aperture, m, and full divergence, rad), ``sigma_r`` (range noise, m),
  ``rough`` (surface roughness as a share of R), ``tau`` (the footprint's share
  on the wood an echo needs), ``edge`` (the detection edge's sd, a share of the
  footprint's width), ``eps``, ``eps_min`` and ``mp_len`` (mixed pixels:
  their share ``eps_min + eps (1 - f)`` and how far behind the surface, m),
  ``hide`` (the share of stops whose echo the export deleted, among pulses
  that lost an echo where it could lie: such a recorded pass may be one);
- the prior: ``lam_prior`` and ``lam_sd`` (foliage's median, 1/m, and log
  sd), ``sig_ell`` (its random walk), ``sig_eta`` (the taper's random walk per
  sqrt(m)), ``mu_fork`` and ``sig_fork`` (soft Leonardo's log area ratio and
  its sd), ``sig_root`` (the level at a base), ``r_tip`` and ``sig_tip`` (a
  tip's radius, m, 0 for none, and its log sd), ``sig_delta`` (the centre's
  random walk, m per sqrt(m)), ``centre_rel`` and ``centre_abs`` (its anchor,
  ``centre_rel r_prior + centre_abs``), ``scan_abs`` and ``scan_rel`` (each
  (scan, branch)'s offset, ``scan_abs + scan_rel r``; both 0 for none).
"""

from __future__ import annotations

import numpy as np

from .. import _core
from .evidence import ScanIndex


def joint_table(scans, a, b, window, threads: int = 0) -> dict[str, np.ndarray]:
    """The pulse-row pairs of cylinders ``a[i] -> b[i]``, measured from those axes.

    Every pulse that reached each cylinder's tube (see :func:`pairs`), with
    its signed impact on the axis and range of closest approach, how both move
    when the axis moves across itself, its chord through the tube, and its fate
    there: the first echo inside the chord is a stop; otherwise the pulse went
    on. ``deleted`` says the export deleted an echo of the pulse that could lie
    in the chord before its first recorded one (``las_return_number`` and
    ``las_number_of_returns`` say which, and so between which recorded echoes
    it lay; without return numbers, anywhere).

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b
        ``(n, 3)`` cylinder axes.
    window
        Radius (m) of each cylinder's tube, ``(n,)``.
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict of ``(m,)`` arrays, one entry per pair:

    - ``b``, ``t``, ``sin``: impact (m), range of closest approach (m) and the
      sine of the pulse's angle to the axis;
    - ``jb``, ``jt`` ``(m, 2)``: how ``b`` and ``t`` move per metre of axis
      offset along the row's two cross directions;
    - ``c0``, ``c1``: the chord through the tube along the pulse (m);
    - ``stop``, ``x``: whether an echo lay inside the chord, and its range;
    - ``deleted``: a deleted echo could lie in the chord before the first
      recorded one;
    - ``early``: the pulse echoed before the tube (recorded whatever happened);
    - ``q``: the chance something behind records a pass (1);
    - ``row``, ``scan``, ``pulse``, ``n_echo``: where the pair came from.
    """
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("joint_table: scans must be ScanIndex")
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    window = np.ascontiguousarray(np.broadcast_to(np.asarray(window, np.float64), (len(a),)))
    return _core.qsm_joint_table([s._core for s in scans], a, b, window, int(threads))


def joint_fit(table, a, b, r_prior, parent, n_scans: int, params, *, r_max=None,
              schedule=(0.3, 0.15, 0.08, 0.04), iters: int = 100, tol: float = 1e-8,
              theta0=None, threads: int = 0) -> dict:
    """Fits the joint model to a :func:`joint_table` (its MAP and Laplace summary).

    Parameters
    ----------
    table
        :func:`joint_table`'s pairs (or a subset of them, ``q`` changed).
    a, b, r_prior, parent
        The rows: ``(n, 3)`` axes, ``(n,)`` prior radii and parents (int64;
        each before its child, -1 at a base).
    n_scans
        Number of scans the table's ``scan`` indexes.
    params
        The likelihood's and the prior's named numbers (module documentation).
    r_max
        ``(n,)`` largest radius each row can take, the tube its pairs were
        taken in (default: unbounded).
    schedule
        Surface roughness of the continuation's stages before the physical one
        (``params["rough"]``); those not above it are skipped.
    iters, tol
        Newton iterations per stage at most, and the stopping rule: the Newton
        decrement, or the objective's gain over ten iterations, below
        ``tol (1 + |objective|)`` (the likelihood is piecewise smooth).
    theta0
        ``(n, 4)`` starting states (default: the axes, prior radii and foliage).
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict:

    - ``theta`` ``(n, 4)``: each row's MAP offsets (m, along the row's frame),
      ``log R`` and ``log lambda``; ``sd`` ``(n, 4)`` their Laplace sds;
    - ``nu`` ``(n_scans * branches, 2)``: each (scan, branch)'s offset;
    - ``rho_local``: the share of ``log R``'s posterior precision from the
      row's own rays; ``rho_pooled``: its variance's reduction from the prior's;
    - ``grad`` ``(n, 4)``, ``hess`` ``(n, 4, 4)``: each row's likelihood
      message at the MAP;
    - ``profile`` ``(n, 21)``: each row's own log likelihood with ``log R``
      moved by ``profile_grid``, the rest held;
    - ``objective``, ``iterations`` (per stage), ``converged`` (every stage met
      the stopping rule) and ``exact`` (the sds come from the true Hessian,
      else from the saddle-free one).
    """
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    r_prior = np.ascontiguousarray(r_prior, np.float64)
    r_max = np.full(len(a), np.inf) if r_max is None else np.ascontiguousarray(r_max, np.float64)
    parent = np.ascontiguousarray(parent, np.int64)
    if theta0 is not None:
        theta0 = np.ascontiguousarray(theta0, np.float64)
    return _core.qsm_joint_fit(table, a, b, r_prior, r_max, parent, int(n_scans), dict(params), [float(s) for s in schedule],
                               int(iters), float(tol), theta0, int(threads))
