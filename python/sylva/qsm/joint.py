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

:func:`joint_tree` is the whole fit of one tree: the tree's rows and its
neighbours' cylinders in, a fitted row per stem out, with existence and the
split of clumps into stems. :func:`joint_table` and :func:`joint_fit` are its
parts, for research: the pairs, and the fit of a table. :func:`support_tree`
is its fast path: each row's pieces' support function per scan, bounds and
existence where a circle isn't determined, and one solve over the tree, with
flags for the pieces that need :func:`joint_tree`.

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
  tips over 2; 0 for no cap);
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
from .evidence import ScanIndex

_DEFAULTS = _core.qsm_joint_defaults()


@dataclasses.dataclass(frozen=True)
class JointParams:
    """The settings of :func:`joint_tree`; the defaults are calibrated on TERN's RIEGL VZ-400i exports (Alice Mulga).

    A plain record: ``dataclasses.asdict(params)`` is JSON-serialisable and, with ``version``, is what a cache of
    joint fits should key on. The likelihood's and the prior's fields are described in this module's documentation.
    """

    #: Bumped by Sylva whenever the fit changes for the same settings; a record of another version is an error.
    version: int = _DEFAULTS["version"]
    exit: float = _DEFAULTS["exit"]
    divergence: float = _DEFAULTS["divergence"]
    sigma_r: float = _DEFAULTS["sigma_r"]
    rough: float = _DEFAULTS["rough"]
    tau: float = _DEFAULTS["tau"]
    edge: float = _DEFAULTS["edge"]
    eps: float = _DEFAULTS["eps"]
    eps_min: float = _DEFAULTS["eps_min"]
    mp_len: float = _DEFAULTS["mp_len"]
    hide: float = _DEFAULTS["hide"]
    refl_wood: float = _DEFAULTS["refl_wood"]
    refl_wood_sd: float = _DEFAULTS["refl_wood_sd"]
    refl_leaf: float = _DEFAULTS["refl_leaf"]
    refl_leaf_sd: float = _DEFAULTS["refl_leaf_sd"]
    refl_multi: float = _DEFAULTS["refl_multi"]
    refl_multi_sd: float = _DEFAULTS["refl_multi_sd"]
    refl_thin: float = _DEFAULTS["refl_thin"]
    stop_wood: float = _DEFAULTS["stop_wood"]
    stop_leaf: float = _DEFAULTS["stop_leaf"]
    exist_scale: float = _DEFAULTS["exist_scale"]
    model_sd: float = _DEFAULTS["model_sd"]
    lam_prior: float = _DEFAULTS["lam_prior"]
    lam_sd: float = _DEFAULTS["lam_sd"]
    sig_ell: float = _DEFAULTS["sig_ell"]
    sig_eta: float = _DEFAULTS["sig_eta"]
    mu_fork: float = _DEFAULTS["mu_fork"]
    sig_fork: float = _DEFAULTS["sig_fork"]
    sig_root: float = _DEFAULTS["sig_root"]
    r_tip: float = _DEFAULTS["r_tip"]
    sig_tip: float = _DEFAULTS["sig_tip"]
    sig_delta: float = _DEFAULTS["sig_delta"]
    centre_rel: float = _DEFAULTS["centre_rel"]
    centre_abs: float = _DEFAULTS["centre_abs"]
    scan_abs: float = _DEFAULTS["scan_abs"]
    scan_rel: float = _DEFAULTS["scan_rel"]
    lean_sd: float = _DEFAULTS["lean_sd"]
    lean_bend: float = _DEFAULTS["lean_bend"]
    exist: float = _DEFAULTS["exist"]
    exist_off: float = _DEFAULTS["exist_off"]
    dh: float = _DEFAULTS["dh"]
    #: Newton iterations per continuation stage at most, and the stopping tolerance.
    iters: int = _DEFAULTS["iters"]
    tol: float = _DEFAULTS["tol"]
    #: Rounds of ownership (each stop to the row whose surface explains it) and fitting.
    sweeps: int = _DEFAULTS["sweeps"]
    #: A row is tried as a clump when its prior radius is at least this (m).
    split_min_r: float = _DEFAULTS["split_min_r"]
    #: A circle found in a clump's stops is a stem when at most this share of the pulses crossing its core went
    #: through it, and at least ``split_min_votes`` stops lie on it.
    split_through: float = _DEFAULTS["split_through"]
    split_min_votes: int = _DEFAULTS["split_min_votes"]
    #: A row's radius interval is where its own log likelihood is within ``drop`` of its best.
    drop: float = _DEFAULTS["drop"]
    #: Which pulses :func:`support_tree` fits on a routed piece of more pairs than its budget (a hash of each pulse's
    #: direction and this): 0 in production; a bench varies it to measure the thinning's noise.
    thin_seed: int = _DEFAULTS["thin_seed"]


#: Why a row's radius is what it is: :func:`joint_tree`'s ``reason``.
JointReason = IntEnum("JointReason", _core.qsm_joint_reasons())


def joint_tree(scans, a, b, r_prior, parent, context=None, params: JointParams = JointParams(), split=None,
               threads: int = 0) -> dict:
    """Fits the joint model to one tree: its rows in, a fitted row per stem out.

    Each row's pairs are taken in a tube around it, sized from its prior radius. A row may be a clump: the solid
    stems its stops show (circles whose core stops the pulses crossing it) replace it as child rows, parallel to it,
    when they explain its own pulses better than it does whole, by more than a BIC price (``split_gain``). The neighbours' cylinders (``context``) are held
    at their claims and keep the stops their surfaces explain. Then rounds of ownership and fitting, and per row
    the evidence that it is wood: its log likelihood at its fit against its best without wood (foliage refitted),
    less a BIC price for the three wood unknowns, as ``p_wood``.

    Parameters
    ----------
    scans
        :class:`ScanIndex` of each scan to use.
    a, b, r_prior, parent
        The tree's rows: ``(n, 3)`` axes, ``(n,)`` prior radii and parents (int; each before its child, -1 at a
        base).
    context
        ``(a, b, r)`` of the neighbours' cylinders, or None.
    params
        :class:`JointParams`.
    split
        ``(n,)`` bool: the rows to split wherever they show solid stems (default: each row of at least
        ``params.split_min_r``, split only where the evidence says so).
    threads
        Threads to use (0: one per core).

    Returns
    -------
    dict, per child row (``(m,)`` unless noted):

    - ``src``: the input row it came from; ``parent``: its parent among the child rows (-1 at a base);
    - ``a``, ``b`` ``(m, 3)``: its fitted axis, world coordinates; ``r``, ``sd_log_r``;
    - ``lo``, ``hi``, ``flat``: the radius interval from its own profile (to 0.125 in log R), and whether the
      rays don't bound it;
    - ``lam``: foliage (1/m); ``log_odds`` and ``p_wood``, the evidence that it is wood; ``reason``
      (:class:`JointReason`);
    - ``rho_local``, ``rho_pooled``: the share of its log R precision from its own rays, and its variance's
      reduction from the prior's; ``n_pairs``: its pairs after ownership;
    - ``grad`` ``(m, 4)``, ``hess`` ``(m, 4, 4)``: its likelihood message at the fit, in its state (offsets
      along ``e1``, ``e2`` ``(m, 3)``, log R, log lambda);

    per input row ``split_gain`` ``(n,)``: the log evidence for its solid stems over it whole, less the price (NaN
    where none was weighed; near 0 is contested); and per tree ``u`` (the volume share the rays don't determine),
    ``objective``, ``converged`` and ``echoes``: the rows' owned stops classified by the softmax of their wood and
    leaf terms at their row's fit, keyed and laid out as :func:`support_tree`'s (``node`` a child row).
    """
    if not all(isinstance(s, ScanIndex) for s in scans):
        raise TypeError("joint_tree: scans must be ScanIndex")
    if not isinstance(params, JointParams):
        raise TypeError("joint_tree: params must be JointParams")
    f64 = lambda x: np.ascontiguousarray(x, np.float64)  # noqa: E731
    ca, cb, cr = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)) if context is None else context
    ca, cb = f64(ca).reshape(-1, 3), f64(cb).reshape(-1, 3)
    split = None if split is None else np.ascontiguousarray(split, bool)
    return _core.qsm_joint_tree([s._core for s in scans], f64(a), f64(b), f64(r_prior),
                                np.ascontiguousarray(parent, np.int64), ca, cb, f64(cr), dataclasses.asdict(params),
                                split, int(threads))


#: Why a piece's own support summary was left out of :func:`support_tree`'s tree (bits of its ``flags``): ``RUNS``
#: (a scan's profile shows two or more stems within the claim) and ``HULL`` (the support is wider than one circle
#: explains within the piece: several stems or a non-round section) are routed, fitted by the exact likelihood there;
#: ``BOUNDS`` (the circle isn't determined: the radius is the prior cut by bounds) and ``POROUS`` (pulses, but no
#: opaque run: existence, and the radius the prior cut by any bounds) are bounded.
SupportFlag = IntFlag("SupportFlag", _core.qsm_support_flags())

#: Why a ``RUNS`` or ``HULL`` piece's exact fit isn't used (bits of :func:`support_tree`'s ``refused``): ``NO_SEED``
#: (no run made a seed), ``NO_PULSES`` (none left to fit), ``UNCONVERGED`` (Newton didn't reach its stopping rule),
#: ``NON_FINITE``, ``BOUND`` (one stem narrower than the piece's detection bound) and ``UNINFORMED`` (a stem whose
#: message says nothing of its radius).
SupportRefusal = IntFlag("SupportRefusal", _core.qsm_support_refusals())

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

#: How a classified echo (``echoes`` of :func:`support_tree` and :func:`joint_tree`) got its ``p_wood``:
#: ``MEASURED``, ``BOUNDED``, ``POROUS`` and ``ROUTED`` (a routed piece with no exact fit accepted) by whether it lies
#: on its piece's circle's front arc (the piece's own, or the tree's) times its echo character's weight, off the arc
#: 0; ``EXACT`` by the exact likelihood's wood and leaf terms at its stem's fit (a routed piece's, or every one of
#: :func:`joint_tree`'s).
EchoPath = IntEnum("EchoPath", _core.qsm_echo_paths())


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
    """Fits one tree from the support function of each row's cross-section: the fast path of :func:`joint_tree`.

    A row longer than 30 cm is cut into equal pieces at most that long, and each piece is a node of the tree (its
    parent the piece before it, or the last of its parent row's), so a bent row is a chain of straight pieces. Per
    (piece, scan), the pulses in a tube around it (wide enough for it to lean) reduce to a profile across it: where
    they stopped, at what depth, and where they passed. Opaque wood's profile measures its section's support
    function, linear in a circle's centre and radius: the silhouette's edges (each by maximum likelihood on its
    hit/pass transition, the footprint and ``tau``/``edge`` of ``params``; an edge with stops beyond it is censored
    and left out, and one with nothing recorded beyond it was seen against the sky and ends half a pulse spacing
    past its outermost hit), the front's depth, and the lean along the piece. A stop counts as wood by its echo
    character (``refl_*``: the chance of wood against leaf by its reflectance and return position, wood narrower
    than the footprint reading darker). Per piece, weighted least squares over its scans gives the centre at its
    middle, the lean and the radius with their covariance; a stop is a hit only on the first such circle's front
    arc (the second pass). Where no circle is determined, each scan still bounds R, slice by slice along the piece:
    its woody stops' span is no wider than the silhouette, the nearest passes either side of them (or, in a slice
    with no woody stop, either side of the woody centre; a pass a pulse that met nothing in the tube) no narrower,
    and wood that was hit fills at least ``tau``
    of the footprint; a side with nothing recorded gives no bound. Passes alone bound it too, the ray veto (3dtrees
    research 106): its R is at most the widest circle about the claim's axis that fewer than 10 pulses, or under a
    quarter of its stops and passes, pass through. With ``dh``, the site's largest stem D/H caps it after the solve, on
    that node alone (a stem-scale rule, so it doesn't carry down the tree): R is at most ``dh`` times its reach to the
    tips (the longest path from its start) over 2, its point and its volume's draws clipped there, never below what
    its own rays hold it above. Each piece also has an existence log odds from
    its core's stops and passes (echo character, and opacity: a stop for wood, a pass against it), over the beam's
    oversampling and weighted by ``exist_scale``, and a chain along the branches (wood doesn't stop and restart:
    ``exist``, ``exist_off``) gives ``p_wood``. The tree is one solve: the unflagged pieces' summaries as Gaussian
    measurements, the bounds as censored ones on log R (expectation propagation, one site per piece by
    quadrature), and a prior after :func:`joint_tree`'s (the base's level, the taper's steps, at a fork the child's
    share of the subtree wood length rather than of the prior area as a heavy-tailed Student-t term, so a small
    shoot measured fat can't inflate its parent, the centre's continuity along a branch and the lean's random walk
    between a row's pieces, the centre's anchor and a weak anchor of the lean; no tip anchor). A posterior is then held
    inside its own bounds (between the tightest lower and upper, the upper where they cross), and says where it sits
    on them (``PINNED``). Then the site's cap, then the veto: a piece with no circle of its own whose posterior
    circle the veto refutes is narrowed until it isn't, or moved onto its own wood beside the claim's axis where
    narrowing would cut it below its lower bound (``MOVED``; ``support_tree.rs``'s module documentation says how).

    Point evidence (3dtrees ticket 109): a row none of whose pieces has a circle of its own gets one from the stops
    in its pieces' tubes nearer its claimed axis than any other the tree or its context claims, placed across its
    axis, as the refit fitted RCT's points (RANSAC, least squares on its inliers, and the refit's tests: at least 12
    stops on a claim of at least 1 cm, half of them on it, filling 120 degrees of it, its centre near the axis, at
    most twice the claim and twice the site's cap, and short of the edge of its pieces' tubes). Unless the ray veto
    refutes it (on a tangle it can be another tree's), it measures each of the row's n pieces' R and centre (each
    coordinate) with sd 0.136 R sqrt(n) (0.136 in log R, the refit circles' noise, shared by the n pieces), its
    pieces' upper bounds beside it (the lower bound and capacity floor its own stops gave it replaces), and isn't
    capped (``POINTS``).

    A ``RUNS`` or ``HULL`` piece is fitted by the exact likelihood, seeded from its runs (3dtrees research 93 §18):
    each scan's runs, found slice by slice along the piece, measured as circles with their own centre, lean and
    radius (as a piece's own circle is), scored by every scan's woody stops on their front arcs, hulls dropped,
    strongest first and at most 4. Its pulses are taken once about the claim's axis (the measured and bounded pieces
    of a first solve own the stops they explain better), each seed seeing them from its own leaning axis; every other
    pulse is fitted (fewer on a piece of more than 8,000, so at most about 4,000 are), and 1, 2, ... stems, each from its seed, by Newton (each
    pulse seen from the stem it meets first, each stem its own foliage) while the fits converge and BIC on the same
    pulses rises. Each stem's likelihood message (foliage integrated out), its information scaled to all the pulses
    and discounted by their correlation (``exist_scale``), with ``model_sd`` added to its variances, measures the
    piece's centre and R in the tree, the lean its seed's: of the stems starting near its parent's end (within the
    larger radius and 2 cm), the one whose message says most of its radius continues the piece (the nearest where
    none is that near); another continues a stem of a routed parent piece that ends where it starts, or else is a
    node beside it (its parent's child by a fork term on its share of the stems' area). A fit is refused
    (:class:`SupportRefusal`, ``refused``) where there is no seed or no pulse left, Newton didn't converge, a stem
    isn't finite or its message says nothing of its radius, or its one stem is narrower than the piece's detection
    bound; the piece is then the prior cut by its upper bound and capacity floor. An accepted stem whose sides, as
    its scans see them (the pulses that reach each edge's neighbourhood and its front, against those that echo
    further in front), leave its radius trading against its centre (``SHADOWED``, 3dtrees research 93 §20) says
    nothing along that ridge, and its own bounds enter with it: its radius is the prior's cut by them.

    Every stop is classified (``echoes``): by whether it lies on its piece's circle's front arc (its own, or the
    tree's) times its echo character's weight, or on a routed piece with an exact fit by the exact likelihood's wood
    and leaf terms at its stem's fit (:class:`EchoPath`).

    The tree's volume is its pieces' (a routed piece's further stems aside): at their posterior R (the drawing's,
    ``point``), and as the mean (the volume of record, split by how each piece is known, :class:`SupportBasis`) and an
    interval from 1,000 draws of the tree's joint posterior (the solve's exact covariance, each node's sd widened by
    its basis's factor, each draw held inside its piece's bounds; a fixed seed). The factors are Sylva's calibration
    (``WIDEN`` in ``support_tree.rs``, learned per basis against Alice Mulga's refit circles), not a setting.

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
      widened by its own basis's factor);
    - ``centre``, ``sd_centre``, ``lean``, ``sd_lean`` ``(n, 2)``: the centre at the middle (m) and the lean (m per
      m of axis) along ``e1``, ``e2`` ``(n, 3)``, with their posterior sds;
    - ``r_row``, ``sd_r_row``: the own summary's radius and its sd (a routed piece's: its exact fit's stem's), NaN
      where none entered the tree (a row's: the geometric mean of its pieces');
    - ``chi2``, ``chi2_front``: the misfits (reduced chi-square), of the measurements to the circle (on their rank)
      and of the front's stops about it against the range noise, roughness and a 4 mm floor; NaN where there is
      none (a row's: its pieces' largest);
    - ``r_lo``, ``r_hi``, ``r_floor``: the tightest lower bound on R from detection, upper bound from passes (beside
      a woody run, or through the claim: the veto), and the capacity floor ``tau w / 2`` (which says only that
      something was hit), of those that entered the tree, else NaN (they enter where no circle of its own does, a
      routed piece's without its detection bound, a point circle's piece's upper alone), so ``r`` lies inside them
      but where ``PINNED`` says the rays contradict themselves (bounds that cross, or a veto that refutes every
      circle as wide as the lower bound); a row's: its pieces' root mean square, as its ``r`` is (a piece without a
      lower bound at 0, without an upper unbounded); ``r_cap``, the site's cap on it (NaN without ``dh``, or where a
      circle measures it; a row's: its first piece's); ``r_points``, its row's point circle's radius, NaN where none
      passed its tests (it entered unless the veto refuted it: ``basis`` says so);
    - ``n_pass``, ``pass_share``: the ray veto's audit of its fitted circle, the pulses passing through it and their
      share of its stops and passes (NaN with none; a row's: its pieces' summed), ``REFUTED`` where they refute it;
    - ``basis`` (:class:`SupportBasis`): how its radius is known;
    - ``log_odds``, ``p_wood``: the existence evidence (log odds of wood against foliage) and the chance of wood
      under the branches' chain (a row's: the sum, and the mean: the share of it that is wood);
    - ``flags`` (:class:`SupportFlag` bits; a row's: its pieces'): 0, its own circle entered the tree; ``RUNS`` or
      ``HULL``, its exact fit did where ``k_stems`` > 0, else its bounds; ``BOUNDS`` or ``POROUS``, its bounds;
    - ``n_runs`` (the most stems within the claim a scan's profile shows), ``n_scans`` (the scans that measured
      it), ``n_pairs`` (its pulses);
    - a routed piece's exact fit (a row's: the most stems and the sums): ``k_stems`` (the stems that entered the
      tree; 0: none, or the fit refused), ``split_gain`` (the BIC of those stems over one, NaN for one), ``n_exact``
      (its routed pulses) and ``evals_value``, ``evals_deriv`` (its pair evaluations, of the value alone and with
      derivatives: per routed pulse, its passes), and ``refused`` (:class:`SupportRefusal` bits: why its fit wasn't
      used, 0 where it was or the piece isn't routed);
    - ``diag`` (:class:`SupportDiag` bits; a row's: its pieces') and three numbers: ``data_share``, the share of log
      R's posterior precision its own data give (its measurement's information on R with its centre and lean unknown,
      and its bounds' EP site), 0-1, NaN where that site's precision came out negative; ``contested``, the share of
      its stops claimed by context or another node (NaN with none); ``deleted``, the share of its pulses with echoes
      the export deleted (an exact stem's: its piece's; NaN with none). A row's: its pieces' means. The misfit is
      ``chi2`` and ``chi2_front``, a refusal ``refused``, a thinned fit ``n_exact`` over the budget;

    and per tree ``converged``, whether the censored solve settled; ``volume``, a dict of m^3: ``point`` (at each
    piece's posterior R: the drawn geometry's, per-piece medians), ``mean`` (the volume of record), ``q05`` and
    ``q95`` (of the draws), and the mean by basis (each piece's draws' mean, adding to it), ``measured``,
    ``points``, ``bounded``, ``prior`` and ``capped``; and ``echoes``, one record per stop of the tree
    (of an echo several pieces saw, the one whose surface it lies nearest), keyed as the tile sidecars are: ``scan``
    (an index into ``scans``), ``pulse`` (the pulse's row in its Shots file), ``echo`` (its place in the pulse);
    with ``p_wood`` (the chance it is wood's echo), ``node`` (its piece or stem, in ``pieces``), ``path``
    (:class:`EchoPath`) and ``dist`` (m from that node's surface).
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
        :class:`JointParams` (its likelihood's and prior's fields).
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
    if not isinstance(params, JointParams):
        raise TypeError("joint_fit: params must be JointParams")
    return _core.qsm_joint_fit(table, a, b, r_prior, r_max, parent, int(n_scans), dataclasses.asdict(params), [float(s) for s in schedule],
                               int(iters), float(tol), theta0, int(threads))


def rebase(scans, a, b, r_prior, parent, member, centre, context=None, params: JointParams = JointParams(),
           reach: float = 0.6, z_lo: float = 0.1, z_hi: float = 1.5, dz: float = 0.1, link: float = 0.02,
           keep: float = 0.8, threads: int = 0) -> dict:
    """A plant's base re-sectioned from the rays (3dtrees ticket 125): its grounded members' rows ``a[i] -> b[i]``
    (``parent`` within the rows, -1 at a member's base, parents first; ``member`` each row's), joined to the columns
    of wood standing in its bottom ``z_hi`` above the ground, where the rays favour them, for :func:`support_tree` to
    measure.

    From ``z_lo`` to ``z_hi`` above the ground ``centre[2]``, slabs ``dz`` thick about ``centre``'s vertical axis,
    each scan's pulses within ``reach`` of it; a slab's woody stops (echo character over a half) from every scan,
    linked where within ``link`` of each other, are its sections, each with a circle (RANSAC and least squares, its
    arc's covariance for its sd; a sliver one scan grazes is centred behind its stops) the ray veto audits. Sections
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
    sections, nor where, in some slab, it reads thinner than the rows it would replace and the rays support those
    rows more. A column's rows go to the member whose taken rows carry most of the skeleton above their cuts; another
    member's row it takes starts that member's base at its cut (a fork, at the fork). A member that would stand on two
    bases keeps its rows as they were. Of a member re-sectioned, other rows under the chosen columns' top whose
    middle lies in a chosen column's circle, or that the veto refutes at a quarter of their radius, are dropped; a
    kept row whose parent was dropped hangs from its member's nearest row end within 30 cm, no more than 5 cm above
    its start; one whose subtree reaches above ``z_hi`` is else joined to such an end within 1 m by a row of its
    dropped parent's radius (else its member keeps its rows); any other is dropped. A
    column whose owner keeps a base of its own (a stem the columns missed) starts there.
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
    ``sections``, per section ``slab``, ``z``, ``x``, ``y``, ``r``, ``sd_log_r``, ``fitted``, ``arc``, ``n_scans``,
    ``hits``, ``n_pass``, ``pass_share``, ``evidence``, ``n_through``, ``column`` (-1: none), and per column
    ``col_parent``, ``col_at``; ``columns``, per column ``e_col``, ``e_rct`` and ``measured`` (its root column's,
    with its forks), ``chosen``, ``owner`` (-1: none), ``r`` (its median), ``believable``, ``joined``, ``grafted``;
    and
    ``counts`` of the rows rising through a column's top near it (``entries``: ``joined``, ``grafted``, ``kept``),
    rows ``dropped`` and ``hung``, and members ``reverted`` to their rows.
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
