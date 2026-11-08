"""Read-only size measurements for an explicitly complete crossing inventory.

The GPU detector owns whether two faces intersect. This module measures every
reported pair; it never exempts neighboring material faces or changes counts.
Coplanar overlaps are clipped as polygons. All lengths are world metres and
affected area is a union of unique material faces, measured in the rest mesh.
"""
from __future__ import annotations

import numpy as np

from bench.evaluation.cloth.audit_cloth_self_intersection import segment_triangle_hit

SCHEMA = 'laundry_garment_crossing_measure.v1'


def _diameter(points):
    points = np.asarray(points, dtype=float)
    return max((float(np.linalg.norm(points[start:start+128, None]-points[None], axis=2).max())
                for start in range(0, len(points), 128)), default=0.)


def _unique(points):
    result = []
    for point in points:
        if not any(np.linalg.norm(point-existing) <= 1.e-10 for existing in result):
            result.append(np.asarray(point, dtype=float))
    return np.asarray(result, dtype=float).reshape(-1, 3)


def _coplanar_polygon(a, b, normal):
    drop = int(np.argmax(np.abs(normal)))
    axes = [index for index in range(3) if index != drop]
    cross2 = lambda u, v: u[0]*v[1]-u[1]*v[0]
    orientation = np.sign(cross2(b[1, axes]-b[0, axes], b[2, axes]-b[0, axes]))
    points = list(a.copy())
    for start, end in zip(b, np.roll(b, -1, axis=0)):
        edge = end[axes]-start[axes]
        distance = lambda point: orientation*cross2(edge, point[axes]-start[axes])
        output = []
        for first, second in zip(points, points[1:]+points[:1]):
            da, db = distance(first), distance(second)
            inside_a, inside_b = da >= -1.e-14, db >= -1.e-14
            if inside_a != inside_b:
                output.append(first+da/(da-db)*(second-first))
            if inside_b:
                output.append(second)
        points = output
        if not points:
            break
    return _unique(points)


def _pair_geometry(a, b):
    na, nb = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if min(la, lb) <= 1.e-14:
        raise ValueError('Crossing geometry contains a degenerate face')
    coplanar = abs(na@nb)/(la*lb) > 1.-1.e-10 and abs((b[0]-a[0])@na)/la < 1.e-9
    points = []
    if coplanar:
        points = _coplanar_polygon(a, b, na)
    else:
        for edges, target in ((a, b), (b, a)):
            for index in range(3):
                hit, point = segment_triangle_hit(edges[index][None], edges[(index+1) % 3][None],
                    target[None], return_points=True)
                if hit[0]:
                    points.append(point[0])
        points = _unique(points)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Reported crossing has no finite measurable intersection')
    area = (float(np.linalg.norm(np.sum(np.cross(points[1:-1]-points[0], points[2:]-points[0]), axis=0)))/2
            if coplanar and len(points) >= 3 else 0.)
    length = (float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum())
              if area > 1.e-16 else _diameter(points))
    return points, length, area, bool(coplanar)


def crossing_measure(vertices, triangles, rest_vertices, pairs, *, pair_count, complete=True):
    """Measure all supplied pairs; truncated/uncertain inventories fail closed."""
    result = {'schema': SCHEMA, 'complete': False, 'pair_count': int(pair_count),
              'scope': 'Sampled crossing dimensions; no continuous collision or contact-force claim.'}
    try:
        q, rest = np.asarray(vertices, dtype=float), np.asarray(rest_vertices, dtype=float)
        tri = np.asarray(triangles)
        pairs = np.asarray(pairs).reshape(-1, 2)
        if (not complete or type(pair_count) is not int or pair_count < 0 or len(pairs) != pair_count
                or q.shape != rest.shape or q.ndim != 2 or q.shape[1] != 3
                or not np.isfinite(q).all() or not np.isfinite(rest).all()
                or tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu'
                or tri.min() < 0 or tri.max() >= len(q)
                or (len(pairs) and pairs.dtype.kind not in 'iu')):
            raise ValueError('Complete finite geometry and pair inventory required')
        pairs = pairs.astype(int)
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        if (len(pairs) != pair_count or (len(pairs) and
                (pairs.min() < 0 or pairs.max() >= len(tri) or np.any(pairs[:, 0] == pairs[:, 1])))):
            raise ValueError('Crossing pair identities are incomplete or duplicated')
        material = rest[tri]
        areas = np.linalg.norm(np.cross(material[:, 1]-material[:, 0], material[:, 2]-material[:, 0]), axis=1)/2
        if not np.all(areas > 0):
            raise ValueError('Positive rest material face areas required')
        affected = np.unique(pairs)
        parents = {int(face): int(face) for face in affected}
        def root(face):
            while parents[face] != face:
                parents[face] = parents[parents[face]]
                face = parents[face]
            return face
        def union(first, second):
            parents[root(first)] = root(second)
        by_vertex = {}
        for face in affected:
            face = int(face)
            for vertex in tri[face]:
                if int(vertex) in by_vertex:
                    union(face, by_vertex[int(vertex)])
                by_vertex[int(vertex)] = face
        for first, second in pairs:
            union(int(first), int(second))
        components, lengths, overlap_areas = {}, [], []
        coplanar_count = structural_count = 0
        for first, second in pairs:
            points, length, area, coplanar = _pair_geometry(q[tri[first]], q[tri[second]])
            components.setdefault(root(int(first)), []).extend(points)
            lengths.append(length)
            overlap_areas.append(area)
            coplanar_count += int(coplanar)
            structural_count += int(bool(set(tri[first]) & set(tri[second])))
        component_diameters = [_diameter(points) for points in components.values()]
        result.update(complete=True, affected_triangle_count=len(affected),
            affected_rest_area_fraction=float(areas[affected].sum()/areas.sum()),
            maximum_pair_length_m=max(lengths, default=0.), total_intersection_length_m=float(sum(lengths)),
            maximum_component_diameter_m=max(component_diameters, default=0.),
            component_count=len(components), coplanar_pair_count=coplanar_count,
            structural_neighbor_pair_count=structural_count,
            coplanar_overlap_area_sum_m2=float(sum(overlap_areas)),
            length_basis='Noncoplanar intersection segment diameter; coplanar polygon perimeter (or touch diameter). Pair lengths are conservatively summed without subtracting duplicate collinear portions.',
            component_basis='Crossing-linked faces plus shared material vertices; maximum world-space intersection-point separation within each connected component.')
    except (ValueError, IndexError, TypeError) as error:
        result['error'] = str(error)
    return result
