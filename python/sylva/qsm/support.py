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
    """
    return _core.qsm_surface_distance(np.ascontiguousarray(points, np.float64), [int(o) for o in point_offsets],
                                      np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64),
                                      np.ascontiguousarray(radius, np.float64), [int(o) for o in cyl_offsets], k)
