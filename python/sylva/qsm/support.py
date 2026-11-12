"""Which trees' points crowd a tree's cylinders: :func:`contest`, batched over a plot's trees in parallel in Rust."""

from __future__ import annotations

import numpy as np

from .. import _core


def contest(points, labels, a, b, radius, cyl_offsets, tol: float):
    """Each tree's contested points: of the points within ``tol`` of its cylinders' surfaces, the share another
    tree's label holds, and the label holding most of them.

    Tree ``g`` owns cylinders ``cyl_offsets[g]:cyl_offsets[g + 1]`` and the points labelled ``g``; a point near
    several of its cylinders counts once.

    Parameters
    ----------
    points
        ``(n, 3)`` the cloud.
    labels
        ``(n,)`` int32: each point's tree, ``< 0`` for none.
    a, b, radius
        Cylinders ``a[i] -> b[i]`` of radius ``radius[i]``, grouped by tree.
    cyl_offsets
        ``n_trees + 1`` offsets into the cylinders.
    tol
        Reach beyond the surface (m).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(n_trees,)`` the share another tree holds (0 with no points near), and ``(n_trees,)`` int64 the label
        holding most (ties to the lowest; -1 if none).
    """
    return _core.qsm_contest(np.ascontiguousarray(points, np.float64), np.ascontiguousarray(labels, np.int32),
                             np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64),
                             np.ascontiguousarray(radius, np.float64), [int(o) for o in cyl_offsets], float(tol))
