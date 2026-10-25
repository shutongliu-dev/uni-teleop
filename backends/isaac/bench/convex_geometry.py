"""Pure NumPy triangle versus convex-piece geometry checks.

The clipping math is preserved from the archived sequence geometry audit.
No physics engine, model cooking or recorded-state mutation is performed.
"""
from __future__ import annotations

import numpy as np


def clip_polygon(polygon, normal, offset):
    """Intersect a convex polygon with n.x + d <= 0, in three dimensions."""
    result = []
    previous = polygon[-1]
    previous_distance = float(previous @ normal + offset)
    for current in polygon:
        distance = float(current @ normal + offset)
        if (distance <= 0) != (previous_distance <= 0):
            fraction = previous_distance / (previous_distance - distance)
            result.append(previous + fraction * (current - previous))
        if distance <= 0:
            result.append(current)
        previous, previous_distance = current, distance
    return np.asarray(result, dtype=float).reshape(-1, 3)


def convex_triangle_audit(vertices, triangles, planes, bounds, tolerance_m=1e-6):
    """Exact triangle/eroded-convex intersection, with conservative tolerances.

    Normalized halfspaces represent the *individual* convex collision pieces,
    never a global hull which could incorrectly seal a gear's open bore.
    """
    tri = vertices[triangles]
    lo, hi = bounds
    candidate = np.flatnonzero(np.all(tri.max(axis=1) >= lo, axis=1)
                               & np.all(tri.min(axis=1) <= hi, axis=1))
    penetrated = []
    maximum_depth = 0.
    worst = None
    for first in range(0, len(candidate), 256):
        ids = candidate[first:first + 256]
        sample = tri[ids]
        signed = sample @ planes[:, :3].T + planes[:, 3]
        # All vertices on the outside of one convex face certify this triangle.
        unresolved = ~(signed.min(axis=1) >= -tolerance_m).any(axis=1)
        for index in ids[unresolved]:
            polygon = tri[index].copy()
            for plane in planes:
                polygon = clip_polygon(polygon, plane[:3], plane[3] + tolerance_m)
                if len(polygon) == 0:
                    break
            if len(polygon):
                witness = polygon.mean(axis=0)
                depth = max(0., -float(np.max(witness @ planes[:, :3].T + planes[:, 3])))
                if depth >= tolerance_m * (1 - 1e-8):
                    penetrated.append(int(index))
                    if depth > maximum_depth:
                        maximum_depth = depth
                        worst = {"triangle_id": int(index), "position_local_m": witness.tolist()}
    return {"triangles_penetrating_tolerance": penetrated,
            "maximum_confirmed_piece_penetration_m": maximum_depth,
            "worst_witness": worst,
            "all_triangles_certified_clear_within_tolerance": not penetrated}
