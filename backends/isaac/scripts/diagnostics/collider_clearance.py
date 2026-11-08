"""Conservative offline convex-hull/triangle-surface separation certificates.

Per-triangle separating axes preserve open container cavities. They provide
a lower bound, not an approximate minimum distance; an unresolved pair fails.
Authored surfaces and convex hulls are checked, not the native cooked SDF.
"""
from __future__ import annotations

import numpy as np


def hull(vertices):
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices, dtype=float)
    result = ConvexHull(vertices)
    faces = result.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    reverse = (normals*result.equations[:, :3]).sum(1) < 0.
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    return {'vertices': vertices, 'faces': faces, 'triangles': vertices[faces],
            'hull_vertices': vertices[result.vertices], 'planes': result.equations}


def usd_geometry(prim, transform_cache):
    """Read actual USD topology; unknown analytic shapes retain AABB rejection."""
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.TetMesh):
        mesh = UsdGeom.TetMesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        declared = mesh.GetSurfaceFaceVertexIndicesAttr().Get()
        if declared is None:
            raise ValueError('TetMesh workspace geometry requires explicit physical surface faces')
        faces = np.asarray(declared, dtype=int)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid TetMesh workspace surface vertices/indices')
    elif prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        faces, at = [], 0
        for count in counts:
            polygon = indices[at:at+count]
            faces.extend((polygon[0], polygon[j], polygon[j+1]) for j in range(1, count-1))
            at += count
        faces = np.asarray(faces, dtype=int)
    elif prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])*size
        faces = hull(vertices)['faces']
    else:
        return None
    matrix = np.asarray(transform_cache.GetLocalToWorldTransform(prim)).T
    vertices = vertices @ matrix[:3, :3].T+matrix[:3, 3]
    if prim.GetAttribute('physics:approximation').Get() == 'convexHull':
        return hull(vertices)
    return {'vertices': vertices, 'faces': faces, 'triangles': vertices[faces]}


def _axes_separation(vertices, triangle, axes):
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1.e-12]/lengths[lengths > 1.e-12, None]
    if not len(axes):
        return 0.
    a, b = vertices @ axes.T, triangle @ axes.T
    return float(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)).max())


def winding_number(point, triangles):
    p = triangles-np.asarray(point)
    lengths = np.linalg.norm(p, axis=2)
    numerator = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2]))
    denominator = (lengths.prod(1)+(p[:, 0]*p[:, 1]).sum(1)*lengths[:, 2]
                   +(p[:, 1]*p[:, 2]).sum(1)*lengths[:, 0]
                   +(p[:, 2]*p[:, 0]).sum(1)*lengths[:, 1])
    return float(np.arctan2(numerator, denominator).sum()/(2*np.pi))


def closed_surface(geometry):
    # USD meshes often duplicate identical vertices at hard-normal seams.
    _, inverse = np.unique(geometry['vertices'], axis=0, return_inverse=True)
    faces = inverse[geometry['faces']]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def separated_convex_triangle_distance(robot, triangle):
    """Euclidean feature distance after a separating axis proves disjointness.

    Vertex/face and edge/edge features exhaust the minimum for disjoint convex
    polytopes. This must never replace the prior intersection check.
    """
    from trimesh.triangles import closest_point
    vertices, faces = robot['hull_vertices'], robot['triangles']
    projected = closest_point(np.repeat(triangle[None], len(vertices), axis=0), vertices)
    distance2 = float(((projected-vertices)**2).sum(1).min())
    for point in triangle:
        projected = closest_point(faces, np.repeat(point[None], len(faces), axis=0))
        distance2 = min(distance2, float(((projected-point)**2).sum(1).min()))
    a0 = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    a1 = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    u = a1-a0
    for b0, b1 in zip(triangle, triangle[[1, 2, 0]]):
        v, w = b1-b0, a0-b0
        aa, bb, cc = (u*u).sum(1), u @ v, float(v @ v)
        dd, ee = (u*w).sum(1), w @ v
        denominator = aa*cc-bb*bb
        valid = denominator > 1.e-24
        ss = np.zeros(len(u)); tt = np.zeros(len(u))
        ss[valid] = (bb[valid]*ee[valid]-cc*dd[valid])/denominator[valid]
        tt[valid] = (aa[valid]*ee[valid]-bb[valid]*dd[valid])/denominator[valid]
        valid &= (ss >= 0.) & (ss <= 1.) & (tt >= 0.) & (tt <= 1.)
        if valid.any():
            delta = w[valid]+ss[valid, None]*u[valid]-tt[valid, None]*v
            distance2 = min(distance2, float((delta*delta).sum(1).min()))
        # Boundary minima, including parallel and degenerate edge pairs.
        for point in (b0, b1):
            fraction = np.clip(np.divide(((point-a0)*u).sum(1), aa,
                out=np.zeros_like(aa), where=aa > 1.e-24), 0., 1.)
            delta = a0+fraction[:, None]*u-point
            distance2 = min(distance2, float((delta*delta).sum(1).min()))
        for points in (a0, a1):
            fraction = np.clip((points-b0) @ v/cc, 0., 1.) if cc > 1.e-24 else np.zeros(len(points))
            delta = b0+fraction[:, None]*v-points
            distance2 = min(distance2, float((delta*delta).sum(1).min()))
    if not np.isfinite(distance2):
        raise ValueError('Nonfinite convex/triangle distance')
    return float(np.sqrt(max(0., distance2)))


def certify(robot, obstacle, *, gap_m=.005):
    """Certify a convex robot shape is separated from a mesh solid/surface.

    Every triangle must have a separating projection greater than the gap.
    Bidirectional containment is checked as well, avoiding the surface-only
    false pass when a whole small body lies inside another solid.
    """
    if robot is None or obstacle is None or 'planes' not in robot:
        return {'passed': False, 'reason': 'unsupported_shape_requires_native_or_other_geometry_check'}
    vertices = robot['hull_vertices']
    triangles = obstacle['triangles']
    lo, hi = vertices.min(0), vertices.max(0)
    tri_lo, tri_hi = triangles.min(1), triangles.max(1)
    delta = np.maximum(np.maximum(lo-tri_hi, tri_lo-hi), 0.)
    lower = np.linalg.norm(delta, axis=1)
    possible = np.flatnonzero(lower <= gap_m+1.e-9)
    certified, exact_refinements = [], []
    axes = robot['planes'][:, :3]
    edges = np.concatenate((robot['triangles'][:, 1]-robot['triangles'][:, 0],
                            robot['triangles'][:, 2]-robot['triangles'][:, 1],
                            robot['triangles'][:, 0]-robot['triangles'][:, 2]))
    for index in possible:
        triangle = triangles[index]
        triangle_edges = triangle[[1, 2, 0]]-triangle
        normal = np.cross(triangle_edges[0], triangle_edges[1])
        separation = _axes_separation(vertices, triangle, np.vstack((axes, normal)))
        if separation <= gap_m+1.e-9:
            cross = np.cross(edges[:, None], triangle_edges[None]).reshape(-1, 3)
            separation = max(separation, _axes_separation(vertices, triangle, cross))
        if 1.e-9 < separation <= gap_m+1.e-9:
            distance = separated_convex_triangle_distance(robot, triangle)
            exact_refinements.append({'triangle_index': int(index),
                'axis_lower_bound_m': separation, 'feature_distance_m': distance})
            # Preserve a numerical margin and never use feature distances for
            # an intersecting/unresolved triangle.
            if distance > gap_m+1.e-8:
                separation = distance-1.e-8
        if separation <= gap_m+1.e-9:
            return {'passed': False, 'reason': 'triangle_separation_not_certified',
                    'triangle_index': int(index), 'separating_axis_lower_bound_m': separation,
                    'candidate_triangle_count': len(possible), 'exact_refinements': exact_refinements}
        certified.append(separation)
    # If an obstacle component is wholly inside the robot convex body, its
    # vertices are inside all hull planes despite separated surfaces.
    near_vertices = obstacle['vertices']
    mask = np.all((near_vertices >= lo) & (near_vertices <= hi), axis=1)
    if mask.any():
        planes = robot['planes']
        inside = np.all(near_vertices[mask] @ planes[:, :3].T+planes[:, 3] < -1.e-9, axis=1)
        if inside.any():
            return {'passed': False, 'reason': 'obstacle_vertex_inside_robot_hull'}
    closed = closed_surface(obstacle)
    winding = winding_number(vertices[0], triangles) if closed else None
    if winding is not None and abs(winding) > .5:
        return {'passed': False, 'reason': 'robot_inside_closed_obstacle', 'winding_number': winding}
    return {'passed': True, 'reason': 'all_triangles_separated_and_no_containment',
            'candidate_triangle_count': len(possible), 'total_triangle_count': len(triangles),
            'minimum_certified_lower_bound_m': min([*lower[lower > gap_m+1.e-9], *certified]),
            'obstacle_closed_surface': closed, 'robot_point_winding_number': winding,
            'exact_refinements': exact_refinements}
