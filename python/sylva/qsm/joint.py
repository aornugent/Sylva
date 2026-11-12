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
random walk of the centre and of foliage (3dtrees research 74).

:func:`support_tree` fits one tree: each row's pieces' support function per
scan, bounds and existence where a circle isn't determined, the pieces with
several stems fitted by the exact likelihood, and one solve over the tree.
:func:`joint_table` gives the pairs the likelihood reads, and
:func:`rebase` re-sections a plant's base for it.

The parameters are a :class:`JointParams`, whose fields are:

- the likelihood: ``exit`` and ``divergence`` (the beam's diameter at the
  aperture, m, and full divergence, rad), ``sigma_r`` (range noise, m),
  ``rough`` (surface roughness as a share of R), ``tau`` (the footprint's share
  on the wood an echo needs), ``edge`` (the detection edge's sd, a share of the
  footprint's width), ``eps``, ``eps_min`` and ``mp_len`` (mixed pixels:
  their share ``eps_min + eps (1 - f)`` and how far behind the surface, m),
  ``hide`` (the share of stops whose echo the export deleted, among pulses
  that lost an echo where it could lie: such a recorded pass may be one);
- echo character (3dtrees research 93 §11.2, read by :func:`support_tree`
  only): ``refl_wood``, ``refl_wood_sd``, ``refl_leaf`` and ``refl_leaf_sd``
  (the reflectance, dB, of a single return from bark that fills the footprint
  and from leaf, mean and sd), ``refl_multi`` and ``refl_multi_sd`` (a return
  that shares its pulse is ``refl_multi`` dB brighter, negative, with
  ``refl_multi_sd`` more spread in quadrature), ``refl_thin`` (thin wood's
  shift, dB, in full at R <= 1 cm and none from 3 cm);
- the support path's own (read by :func:`support_tree` only): ``stop_wood``
  and ``stop_leaf`` (the share of the pulses aimed at a core that stop there,
  on wood and on foliage), ``exist_scale`` (the weight of a scan's existence
  log odds: its pulses are correlated beyond the beam's oversampling),
  ``model_sd`` (the floor on a support measurement's
  sd, m), ``lean_sd`` (the lean's anchor, m per m), ``lean_bend`` (its random
  walk between a row's pieces, per sqrt(m)), ``exist`` and ``exist_off``
  (existence along a branch: a base piece is wood with ``exist``, one after a
  piece that isn't with ``exist_off``), ``dh`` (the site's largest stem D/H:
  no radius the rays don't measure is wider than ``dh`` times its reach to the
  tips over 2; 0 for no cap), ``thin_seed`` (which pulses a routed piece of
  more pairs than its budget fits: a hash of each pulse's direction and this;
  0 in production, a bench varies it to measure the thinning's noise);
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

import dataclasses
from enum import IntEnum, IntFlag

import numpy as np

from .. import _core
from .scan_index import ScanIndex

# written once, in the core: its fields, their types and their defaults
JointParams = dataclasses.make_dataclass(
    "JointParams", [(k, type(v), dataclasses.field(default=v)) for k, v in _core.qsm_joint_defaults().items()],
    frozen=True, module=__name__)
JointParams.__doc__ = """The settings of :func:`support_tree` and :func:`rebase`, with the core's defaults.

Calibrated on TERN's RIEGL VZ-400i exports (Alice Mulga). The fields are described in
:mod:`sylva.qsm.joint`'s documentation; ``dataclasses.asdict(params)`` is what crosses to the core.
"""


#: Why a piece's own support summary was left out of :func:`support_tree`'s tree (bits of its ``flags``): ``RUNS``
#: (a scan's profile shows two or more stems within the claim) and ``HULL`` (the support is wider than one circle
#: explains within the piece: several stems or a non-round section) are routed, fitted by the exact likelihood there;
#: ``BOUNDS`` (the circle isn't determined: the radius is the prior cut by bounds) and ``POROUS`` (pulses, but no
#: opaque run: existence, and the radius the prior cut by any bounds) are bounded.
SupportFlag = IntFlag("SupportFlag", _core.qsm_support_flags())

#: What a reviewer or the screen should know of a node's fit that no other output says (bits of :func:`support_tree`'s
#: ``diag``; a row's: its pieces'). Each threshold is a constant of ``support_tree.rs`` (module ``diag``), with its
#: reason, and isn't restated here:
#:
#: - ``ONE_SIDED``: one scan supplies almost all its informative pulses, or every pair of the views that carry a share
#:   of them looks from nearly one direction: no other view checks it (3dtrees research 93 §10, §11.1);
#: - ``SHADOWED``: an exact stem a side of whose silhouette is unseen (no pulse reaches the circle's neighbourhood
#:   there from a scan that would decide R, as a crossing branch's shadow), so the sides seen leave its radius trading
#:   against its centre: its message says nothing along what they leave free, and its own bounds (the span of its
#:   front arc's stops, the capacity floor) enter the tree beside it. A side past the fit's table isn't judged;
#: - ``SUB_FOOTPRINT``: R below about half the footprint at its range;
#: - ``TUBE_EDGE``: its own fitted circle reaches the edge of the tube its pulses were taken in: evidence cut off, so
#:   a piece's own circle there isn't used (a capture or a drift), nor, as a lower bound, the span of a woody run the
#:   circle spans (the captor's; another scan's run keeps its bound);
#: - ``CONTESTED``: most of its stops are claimed by context or a neighbouring node (``contested``);
#: - ``REFUTED``: the ray veto refutes its fitted circle: at least 10 pulses pass through it, and passes are at least a
#:   quarter of its stops and passes (3dtrees research 106; a row's: of its pieces' counts summed);
#: - ``PINNED``: its radius sits on its own bounds (the tree without them would put it past one, or the posterior did
#:   and was held inside them): what its rays say is a bound, not a value;
#: - ``CAPPED``: its posterior median reached the site's D/H cap (``dh``), applied after the solve: served at the cap;
#: - ``MOVED``: the veto moved its circle onto its own wood beside the claim's axis, refuted or not (3dtrees research
#:   127).
SupportDiag = IntFlag("SupportDiag", _core.qsm_support_diags())

#: How a node's radius is known (:func:`support_tree`'s ``basis``; a row's: most of its pieces', the less known on a
#: tie): ``MEASURED`` (its own circle, or an exact stem's message, not ``SHADOWED``), ``POINTS`` (its row's point
#: circle: the rays gave it no circle of its own), ``BOUNDED`` (its own bounds give
#: at least 0.1 of its precision, ``data_share``, and it isn't pinned), ``PRIOR`` (the prior's, given the rest of the
#: tree, or pinned to its own bound) and ``CAPPED`` (on the site's cap). Its tree's volume is split by them.
SupportBasis = IntEnum("SupportBasis", _core.qsm_support_bases())


def tree_context(a, b, cyl_offsets, trees, reach: float) -> list:
    """Each of ``trees``' context for :func:`support_tree`: of a plot's cylinders, the other trees' whose segments come
    within ``reach`` of one of its own.

    Parameters
    ----------
    a, b
        ``(n, 3)`` cylinders ``a[i] -> b[i]``, grouped by tree.
    cyl_offsets
        ``n_trees + 1`` offsets: tree ``g`` owns cylinders ``cyl_offsets[g]:cyl_offsets[g + 1]``.
    trees
        The trees whose context is wanted (indices into the groups).
    reach
        Closest distance between segments (m). A cylinder of zero length is never close.

    Returns
    -------
    list of numpy.ndarray
        Per tree of ``trees``, in order, the sorted indices of its context's cylinders.
    """
    off, idx = _core.qsm_tree_context(np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64),
                                      [int(o) for o in cyl_offsets], [int(t) for t in trees], float(reach))
    return [idx[off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]


def support_tree(scans, a, b, r_prior, parent, context=None, params: JointParams = JointParams(),
                 threads: int = 0) -> dict:
    """Fits one tree from the support function of each row's cross-section: the joint model, fast.

    A row longer than 30 cm is cut into pieces, each a node of the tree, so a bent row is a chain of straight pieces.
    Per (piece, scan) the pulses in a tube about it reduce to a profile across it, whose edges and front measure a
    leaning circle; a stop counts as wood by its echo character (``refl_*``). Where no circle is determined, each
    scan still bounds R (its woody stops' span, the passes beside them, and the capacity floor ``tau``), and passes
    alone bound it too (the ray veto). A row none of whose pieces has a circle of its own gets one from its own stops
    (point evidence, as 3dtrees' refit fitted RCT's points), unless the veto refutes it. A piece with several stems
    or a hull (``RUNS``, ``HULL``) is fitted by the exact likelihood: seeds from its runs, then 1, 2, ... stems while
    BIC rises, each stem's likelihood message measuring the piece; a fit may be refused, and the piece is then
    bounded. One solve over the tree combines the measurements, the bounds (by expectation propagation) and a
    Gaussian Markov prior along the branches; each radius is then held inside its own bounds, capped by ``dh`` where
    nothing measures it, and narrowed, or moved onto its own wood, where the veto refutes it. Each piece's existence
    evidence gives ``p_wood`` along the branches, every stop is classified, and the tree's volume comes with an
    interval from draws of its joint posterior, each node's sd widened by its basis's calibrated factor.

    The rules, each threshold with its reason, are written once, in ``support_tree.rs``'s module documentation
    (``support_tree/routed.rs``'s for the exact fits); this docstring owns what each output means.

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b, r_prior, parent
        The tree's rows: ``(n, 3)`` axes, ``(n,)`` prior radii and parents (int; each before its child, -1 at a
        base).
    context
        ``(a, b, r)`` of other cylinders near the tree (its neighbours'), or None: a routed piece's exact fit leaves
        them the stops they explain better, as it does the tree's own measured and bounded pieces.
    params
        :class:`JointParams`: the likelihood's footprint, ``tau``, ``edge``, ``sigma_r``, ``rough`` and ``refl_*``,
        ``stop_wood``, ``stop_leaf``, ``exist_scale``, ``refl_thin`` and ``model_sd``, and the prior's ``sig_eta``,
        ``mu_fork``, ``sig_fork``, ``sig_root``, ``sig_delta``, ``lean_sd``, ``lean_bend``, ``exist``, ``exist_off``,
        ``centre_rel``, ``centre_abs`` and ``dh``; the rest are unused.
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict, per row (``(n,)`` unless noted), with the same per node under ``pieces``: the pieces, then the routed
    pieces' further stems, with each node's ``row``, the piece it came from (``src``: itself for a piece) and its stem
    there (``stem``: 0 for a piece's own):

    - ``a``, ``b`` ``(n, 3)``: the fitted axis, world coordinates (the claim moved by the centre and turned by the
      lean; a row's runs from its first piece's start to its last's end); ``r``, ``sd_log_r``: the radius, held inside
      its own bounds, and its posterior sd in log, the solve's widened by its basis's factor (a row's: its pieces' root
      mean square R, so its volume is theirs, and the sd of its log from the chain's joint covariance, each piece
      widened by its own basis's factor); a routed piece's follow its exact fit's continuing stem, or where none was
      accepted, the prior's cut by its bounds;
    - ``centre``, ``lean`` ``(n, 2)``: the centre at the middle (m) and the lean (m per m of axis) along ``e1``,
      ``e2`` ``(n, 3)``;
    - ``r_lo``, ``r_hi``: the tightest lower bound on R from detection and upper bound from passes (beside a woody
      run, or through the claim: the veto), of those that entered the tree, else NaN (they enter where no circle of
      its own does, a routed piece's without its detection bound, which can span several stems; a point circle's
      piece's upper alone), so ``r`` lies inside them but where ``PINNED`` says the rays contradict themselves
      (bounds that cross, or a veto that refutes every circle as wide as the lower bound); a row's: its pieces' root
      mean square, as its ``r`` is (a piece without a lower bound at 0, without an upper unbounded); ``r_cap``, the
      site's cap on it (NaN without ``dh``, or where a circle measures it; a row's: its pieces' widest);
    - ``n_pass``, ``pass_share``: the ray veto's audit of its fitted circle, the pulses passing through it and their
      share of its stops and passes (NaN with none; a row's: its pieces' summed), ``REFUTED`` where they refute it;
    - ``basis`` (:class:`SupportBasis`): how its radius is known;
    - ``p_wood``: the chance of wood under the branches' chain, from each piece's existence evidence (a row's: the
      mean, the share of it that is wood);
    - ``flags`` (:class:`SupportFlag` bits; a row's: its pieces'): 0, its own circle entered the tree; ``RUNS`` or
      ``HULL``, its exact fit did where ``k_stems`` > 0, else its bounds; ``BOUNDS`` or ``POROUS``, its bounds;
    - ``n_scans``: the scans that measured it (a row's: the most);
    - ``k_stems``: the stems a routed piece's exact fit put in the tree (0: none, or the fit refused; a row's: the
      most);
    - ``diag`` (:class:`SupportDiag` bits; a row's: its pieces') and three numbers: ``data_share``, the share of log
      R's posterior precision its own data give (its measurement's information on R with its centre and lean unknown,
      and its bounds' EP site), 0-1, NaN where that site's precision came out negative; ``contested``, the share of
      its stops claimed by context or another node (NaN with none); ``deleted``, the share of its pulses with echoes
      the export deleted (an exact stem's: its piece's; NaN with none). A row's: its pieces' means;

    and per tree ``converged``, whether the censored solve settled; ``volume``, a dict of m^3: ``point`` (at each
    piece's posterior R: the drawn geometry's, per-piece medians), ``mean`` (the volume of record), ``q05`` and
    ``q95`` (of the draws), and the mean by basis (each piece's draws' mean, adding to it), ``measured``,
    ``points``, ``bounded``, ``prior`` and ``capped``; and ``echoes``, one record per stop of the tree
    (of an echo several pieces saw, the one whose surface it lies nearest), keyed as the tile sidecars are: ``scan``
    (an index into ``scans``), ``pulse`` (the pulse's row in its Shots file), ``echo`` (its place in the pulse);
    with ``p_wood`` (the chance it is wood's echo).
    """
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("support_tree: scans must be ScanIndex")
    if not isinstance(params, JointParams):
        raise TypeError("support_tree: params must be JointParams")
    f64 = lambda x: np.ascontiguousarray(x, np.float64)  # noqa: E731
    ca, cb, cr = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)) if context is None else context
    return _core.qsm_support_tree([s._core for s in scans], f64(a).reshape(-1, 3), f64(b).reshape(-1, 3),
                                  f64(r_prior), np.ascontiguousarray(parent, np.int64), f64(ca).reshape(-1, 3),
                                  f64(cb).reshape(-1, 3), f64(cr), dataclasses.asdict(params), int(threads))


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


def rebase(scans, a, b, r_prior, parent, member, centre, context=None, params: JointParams = JointParams(),
           reach: float = 0.6, z_lo: float = 0.1, z_hi: float = 1.5, dz: float = 0.1, link: float = 0.02,
           keep: float = 0.8, threads: int = 0) -> dict:
    """A plant's base re-sectioned from the rays (3dtrees ticket 125): its grounded members' rows ``a[i] -> b[i]``
    (``parent`` within the rows, -1 at a member's base, parents first; ``member`` each row's), joined to the columns
    of wood standing in its bottom ``z_hi`` above the ground, where the rays favour them, for :func:`support_tree` to
    measure.

    From ``z_lo`` to ``z_hi`` above the ground ``centre[2]``, slabs ``dz`` thick about ``centre``'s vertical axis,
    each scan's pulses within ``reach`` of it; a slab's woody stops (echo character over a half) from every scan,
    linked where within ``link`` of each other, are its sections, each with a circle (RANSAC and least squares, a point
    circle's tests on an arc of 60 degrees; a sliver one scan grazes is centred behind its stops) the ray veto audits. Sections
    are linked upward into columns, which may fork. Each circle's evidence is its log odds of wood against foliage
    from the pulses through it, by their opacity (a stop on its chord for it, a pass against it, as a piece's
    existence weighs them, ``stop_wood``, ``stop_leaf``, ``exist_scale``); a row's is its axis's in each slab it
    crosses, at the best of its radius, a half and a quarter of it.

    A column wider than 0.3 m or twice the site's cap at the plant's top (``dh``), or with half its sections inside
    the ``context``'s cylinders (other plants'), or (on the ground) starting over 0.5 m above it, is none of this
    plant's stems. A row rising through a column's top
    (its highest section) joins the column whose top lies nearest where it does, within 8 cm beyond the column's
    radius, or else, where the rays refute the row's chain under that height, is grafted to the nearest column top
    within 30 cm; it is cut there. A column and its forks replace the chains of the rows they take only where their
    evidence is positive, those chains hold under ``keep`` of it in the same slabs, and under half of them are
    measured (by an own circle the veto doesn't refute, as :func:`support_tree` would serve it): a skeleton already
    on its wood keeps it. Nor is a column given to one member where another's rows already stand on half its
    sections. A column's rows go to the member whose taken rows carry most of the skeleton above their cuts; another
    member's row it takes starts that member's base at its cut (a fork, at the fork). A member that would stand on two
    bases keeps its rows as they were. Of a member re-sectioned, other rows under the chosen columns' top whose
    middle lies in a chosen column's circle, or that the veto refutes at a quarter of their radius, are dropped; a
    kept row whose parent was dropped hangs from its member's nearest row end within 30 cm, no more than 5 cm above
    its start, else is joined to such an end within 1 m by a row of its dropped parent's radius, else its member keeps
    its rows. A column whose owner keeps a base of its own (a stem the columns missed) starts there.
    ``support_tree.rs``'s ``base`` module documentation says it in full.

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b, r_prior, parent, member
        The plant's grounded members' rows: ``(n, 3)`` axes, ``(n,)`` prior radii, parents (int; within the rows, each
        before its child, -1 at a member's base) and each row's member (int).
    centre
        The base's centre ``(x, y)`` and its ground's ``z`` (m).
    context
        ``(a, b, r)`` of the other plants' cylinders near the base, or None.
    params
        :class:`JointParams`: ``stop_wood``, ``stop_leaf`` and ``exist_scale`` for the evidence, the ``refl_*`` fields
        for which stops are woody, the footprint for the veto, and ``dh`` for the cap.
    reach
        The tube about the base's axis the slabs' pulses are taken in (m).
    z_lo, z_hi, dz
        The slabs, above the ground: the first's bottom, the last's top and their thickness (m).
    link
        Stops within this of each other are one section (m).
    keep
        The share of a column's evidence RCT's rows must hold in its slabs to keep their base.
    threads
        Threads for the slabs and the rows (0: all).

    Returns a dict: the rows ``a``, ``b`` ``(m, 3)``, ``r_prior``, ``parent`` (a row of the same member, -1 at its
    base: every member's rows hang from one base, parents first), ``member`` and ``column`` (-1 for a skeleton row);
    ``columns``, per column ``e_col``, ``e_rct`` and ``measured`` (its root column's, with its forks), ``chosen``,
    ``owner`` (-1: none), ``r`` (its median), ``believable``, ``joined``, ``grafted``; and ``counts`` of the rows
    rising through a column's top near it (``entries``: ``joined``, ``grafted``, ``kept``), rows ``dropped`` and
    ``hung``, and members ``reverted`` to their rows.
    """
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("rebase: scans must be ScanIndex")
    if not isinstance(params, JointParams):
        raise TypeError("rebase: params must be JointParams")
    f64 = lambda x: np.ascontiguousarray(x, np.float64)  # noqa: E731
    ca, cb, cr = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)) if context is None else context
    return _core.qsm_rebase([s._core for s in scans], f64(a).reshape(-1, 3), f64(b).reshape(-1, 3), f64(r_prior),
                            np.ascontiguousarray(parent, np.int64), np.ascontiguousarray(member, np.int64),
                            [float(x) for x in centre], f64(ca).reshape(-1, 3), f64(cb).reshape(-1, 3), f64(cr),
                            dataclasses.asdict(params), z_lo, z_hi, dz, reach, link, float(keep),
                            int(threads))
