"""How well QSM cylinders are supported by the points around them.

Batched over every cylinder of a tile; both checks run in parallel in Rust.
"""

from __future__ import annotations

import numpy as np

from .. import _core


def surface_support(cloud, a, b, radius, tol: float = 0.03, per_m2: float = 400.0, min_samples: int = 16,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample each cylinder's lateral surface and test each sample against the cloud.

    A sample is supported when a point of ``cloud`` lies within ``tol``. A fat
    or misplaced cylinder leaves most of its samples unsupported.

    Parameters
    ----------
    cloud
        ``(n, 3)`` points the cylinders should be resting on.
    a, b, radius
        Cylinders ``a[i] -> b[i]`` of radius ``radius[i]``.
    tol
        Support distance (m).
    per_m2, min_samples
        Sampling density on the lateral surface, and the minimum per cylinder
        (zero-length cylinders get none).
    seed
        Each cylinder draws from its own generator seeded from this, so the
        result does not depend on thread scheduling.

    Returns
    -------
    (samples (m, 3), cylinder index (m,) uint32, supported (m,) bool)
    """
    return _core.qsm_surface_support(np.ascontiguousarray(cloud, np.float64), np.ascontiguousarray(a, np.float64),
                                     np.ascontiguousarray(b, np.float64), np.ascontiguousarray(radius, np.float64),
                                     tol, per_m2, min_samples, seed)


def surface_distance(points, point_offsets, a, b, radius, cyl_offsets, k: int = 8) -> np.ndarray:
    """Signed distance from each point to the nearest cylinder surface of its own group (< 0 inside).

    Points and cylinders are grouped (e.g. by tree) in CSR form: group ``g``
    holds ``points[point_offsets[g]:point_offsets[g + 1]]`` and cylinders
    ``cyl_offsets[g]:cyl_offsets[g + 1]``. Each point is measured against the
    ``k`` cylinders of its group whose midpoints are nearest; a group without
    cylinders gives ``inf``.

    Parameters
    ----------
    points
        ``(n, 3)`` points, grouped.
    point_offsets
        ``n_groups + 1`` offsets into ``points``.
    a, b, radius
        Cylinders ``a[i] -> b[i]`` of radius ``radius[i]``, grouped.
    cyl_offsets
        ``n_groups + 1`` offsets into the cylinders.
    k
        Cylinders tried per point.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` signed distances (m).
    """
    return _core.qsm_surface_distance(np.ascontiguousarray(points, np.float64), [int(o) for o in point_offsets],
                                      np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64),
                                      np.ascontiguousarray(radius, np.float64), [int(o) for o in cyl_offsets], k)


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
