"""Conservative finite cloth regions nearest to original cooked convex edges.

Only non-coplanar edges of the existing inner hull are considered. A clipped
region stays in the edge's Voronoi wedge, finite segment and an inscribed
distance tube. Original inward-direction and support-slot bounds still apply.
This adds no collision shape, force, attachment or state update.
"""
from __future__ import annotations

import numpy as np

from bench.convex_geometry import clip_polygon
from bench.evaluation.cloth.cloth_robot_collision import apply_transform, planes_in_body


def convex_edges(vertices, faces, planes):
    adjacent = {}
    for index, triangle in enumerate(faces):
        for a, b in zip(triangle, np.roll(triangle, -1)):
            adjacent.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    result = []
    for (a, b), face_ids in adjacent.items():
        normals = []
        for index in face_ids:
            normal = planes[index, :3]
            if not any(np.linalg.norm(normal-other) < 1.e-8 for other in normals):
                normals.append(normal)
        # Internal triangulation diagonals and ambiguous/non-manifold edges
        # cannot supply a new surface region.
        if len(normals) != 2 or np.linalg.norm(normals[0]+normals[1]) < 1.e-8:
            continue
        p0, p1 = vertices[[a, b]]
        length = float(np.linalg.norm(p1-p0))
        if length < 1.e-8:
            continue
        axis = (p1-p0)/length
        if max(abs(float(axis @ normal)) for normal in normals) > 1.e-7:
            continue
        boundary = [np.r_[-axis, axis @ p0], np.r_[axis, -axis @ p1]]
        middle = normals[0]+normals[1]
        for normal in normals:
            cross = np.cross(axis, normal)
            cross /= np.linalg.norm(cross)
            if cross @ middle > 0:
                cross = -cross
            boundary.append(np.r_[cross, -cross @ p0])
        result.append({'vertex_indices': [a, b], 'endpoints': np.array([p0, p1]),
            'axis': axis, 'length': length, 'normal_basis': normals[0],
            'voronoi_planes': np.array(boundary), 'adjacent_face_normals': np.array(normals),
            'adjacent_triangle_faces': list(face_ids)})
    return result


def world_edges(edges, matrix):
    rotation = matrix[:3, :3]
    return [{**edge, 'endpoints': apply_transform(edge['endpoints'], matrix),
        'axis': rotation @ edge['axis'], 'normal_basis': rotation @ edge['normal_basis'],
        'adjacent_face_normals': edge['adjacent_face_normals'] @ rotation.T,
        'voronoi_planes': planes_in_body(edge['voronoi_planes'], matrix)} for edge in edges]


def clipped_edge_region(triangle, edge, pad, *, axis, orientation, alo, bhi,
                        thickness, band, penetration, minimum_area,
                        surface_query, polygon_area):
    u, (p0, p1) = edge['axis'], edge['endpoints']
    v = edge['normal_basis']
    w = np.cross(u, v)
    inward = orientation*axis
    projected = inward-u*float(inward @ u)
    magnitude = float(np.linalg.norm(projected))
    if magnitude < .25:
        return None
    center_direction = projected/magnitude
    angle = float(np.arccos(np.clip(.25/magnitude, -1., 1.)))
    across = np.cross(u, center_direction)
    cone = []
    for ray in [np.cos(angle)*center_direction+np.sin(angle)*across,
                np.cos(angle)*center_direction-np.sin(angle)*across]:
        normal = np.cross(u, ray)
        normal /= np.linalg.norm(normal)
        if normal @ center_direction > 0:
            normal = -normal
        cone.append(np.r_[normal, -normal @ p0])
    # Inscribed, never a circumscribed expanded-halfspace approximation.
    angles = np.arange(64)*2*np.pi/64
    directions = np.cos(angles)[:, None]*v+np.sin(angles)[:, None]*w
    cylinder = np.c_[directions, -directions @ p0-band*np.cos(np.pi/64)]
    lower, upper = alo-thickness/2-penetration, bhi+thickness/2+penetration
    planes = np.vstack([edge['voronoi_planes'], cylinder, cone,
                        np.r_[axis, -upper], np.r_[-axis, lower]])
    polygon = np.asarray(triangle, float).copy()
    for plane in planes:
        polygon = clip_polygon(polygon, plane[:3], plane[3])
        if len(polygon) < 3:
            return None
    area = polygon_area(polygon)
    if area < minimum_area:
        return None
    checks = []
    for point in polygon:
        distance, nearest, _ = surface_query(point, pad['inner'], pad['faces'], pad['inner_planes'],
                                             contact_normal_profile='distance_gradient_v1')
        along = float((point-p0) @ u)
        projection = p0+along*u
        if (distance < -1.e-8 or distance > band+1.e-8 or along < -1.e-8
                or along > edge['length']+1.e-8 or np.linalg.norm(nearest-projection) > 1.e-7):
            return None
        checks.append({'distance_m': float(distance),
                       'nearest_residual_m': float(np.linalg.norm(nearest-projection))})
    point = polygon.mean(0)
    distance, nearest, gradient = surface_query(point, pad['inner'], pad['faces'], pad['inner_planes'],
                                                contact_normal_profile='distance_gradient_v1')
    along = float((point-p0) @ u)
    if gradient is None or distance <= 1.e-10 or not 1.e-8 < along < edge['length']-1.e-8:
        return None
    projection = orientation*float(gradient @ axis)
    if projection < .25-1.e-9:
        return None
    return {'point_m': point.tolist(), 'surface_point_m': nearest.tolist(),
        'normal_world': gradient.tolist(), 'signed_inner_distance_m': float(distance),
        'clipped_area_m2': float(area), 'clipped_polygon_m': polygon.tolist(),
        'inward_normal_projection': projection, 'closest_feature': 'exterior_convex_edge',
        'edge_endpoints_world_m': edge['endpoints'].tolist(),
        'adjacent_face_normals_world': edge['adjacent_face_normals'].tolist(),
        'cooked_triangle_faces': list(edge['adjacent_triangle_faces']),
        'finite_region_vertex_checks': checks,
        'distance_tube_approximation': '64-sided inscribed cylinder, within original exact distance band'}


def material_edge_witnesses(pad, xyz, tri, rest, candidates, axis, orientation,
                            alo, bhi, thickness, band, penetration, minimum_area,
                            *, surface_query, polygon_area):
    rows = []
    for edge in pad['witness_edges']:
        for tid in candidates:
            row = clipped_edge_region(xyz[tid], edge, pad, axis=axis, orientation=orientation,
                alo=alo, bhi=bhi, thickness=thickness, band=band, penetration=penetration,
                minimum_area=minimum_area, surface_query=surface_query, polygon_area=polygon_area)
            if row is None:
                continue
            point = np.asarray(row['point_m'])
            weights = np.linalg.lstsq((xyz[tid, 1:]-xyz[tid, 0]).T, point-xyz[tid, 0], rcond=None)[0]
            material = rest[tri[tid]]
            row.update(triangle_id=int(tid),
                rest_point_m=(material[0]+weights @ (material[1:]-material[0])).tolist())
            rows.append(row)
    return rows
