"""Rigid resting frame for the existing spoon bowl and handle geometry."""
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh


def spoon_rest_transform(mesh):
    """Find real hull support containing the integrated center of mass.

    Trimesh's approximate ``facets`` may merge distinct shallow spoon faces.
    Their first normal can miss the stable bowl/handle triangle, so examine
    each actual lower-hull triangle normal. Nothing is flattened or added.
    """
    hull = mesh.convex_hull
    down = np.array([0., 0., -1.])
    candidates = []
    seen = set()
    for normal in hull.face_normals:
        key = tuple(normal)
        if normal[2] > -.7 or key in seen:
            continue
        seen.add(key)
        axis = np.cross(normal, down)
        angle = math.atan2(float(np.linalg.norm(axis)), float(normal@down))
        transform = (np.eye(4) if angle < 1e-12 else
                     trimesh.transformations.rotation_matrix(angle, axis))
        vertices = trimesh.transform_points(hull.vertices, transform)
        bottom = float(vertices[:, 2].min())
        indices = np.flatnonzero(vertices[:, 2] <= bottom+1e-8)
        support = vertices[indices]
        if len(support) < 3:
            continue
        try:
            polygon = ConvexHull(support[:, :2])
        except QhullError:
            continue
        center = trimesh.transform_points(mesh.center_mass[None], transform)[0]
        margin = float(-np.max(polygon.equations[:, :2]@center[:2]
                              +polygon.equations[:, 2]))
        # Same geometric COM-margin requirement as the existing rest helper.
        if margin < .00025:
            continue
        transform[2, 3] = -bottom
        receipt = {
            'method': 'Rigid rest frame from individual actual hull faces; unchanged spoon solid.',
            'rotation_angle_rad': angle,
            'support_vertex_count': len(support),
            'support_polygon_area_m2': float(polygon.volume),
            'center_of_mass_support_edge_margin_m': margin,
            'support_vertices_before_transform_m': hull.vertices[indices].tolist(),
            'source_to_rest_transform': transform.tolist(),
            'coordinate_frame': 'Recipe metres before the declared dimension_profile scaling.',
            'scope': 'Geometric support only; native settling must be qualified separately.'}
        candidates.append((round(angle, 10), -margin, transform, receipt))
    if not candidates:
        raise ValueError('No nearby spoon hull support contains its center of mass')
    _, _, transform, receipt = min(candidates, key=lambda item: item[:2])
    return transform, receipt
