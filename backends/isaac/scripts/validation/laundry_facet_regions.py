"""Conservative cloth subregions next to original cooked convex faces.

Facet projection prisms separate real face-interior contacts from a single
mean point near a convex corner. All distances still use the exact convex
boundary; expanded halfspaces alone never establish a witness.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from bench.convex_geometry import clip_polygon
from bench.evaluation.cloth.cloth_robot_collision import apply_transform, planes_in_body


def coplanar_facets(vertices, faces, planes):
    """Merge only coplanar triangles of the already cooked inner convex."""
    groups = []
    for index, plane in enumerate(planes):
        group = next((g for g in groups
                      if np.linalg.norm(g['plane'][:3]-plane[:3]) < 1.e-8
                      and abs(g['plane'][3]-plane[3]) < 1.e-9), None)
        if group is None:
            group = {'plane': plane.copy(), 'ids': set(), 'faces': []}
            groups.append(group)
        group['ids'].update(map(int, faces[index]))
        group['faces'].append(index)
    result = []
    for group in groups:
        plane = group['plane']
        points = vertices[sorted(group['ids'])]
        center = points.mean(0)
        axis = points[np.argmax(np.linalg.norm(points-center, axis=1))]-center
        if np.linalg.norm(axis) < 1.e-12:
            continue
        axis /= np.linalg.norm(axis)
        other = np.cross(plane[:3], axis)
        projection = np.stack([(points-center) @ axis, (points-center) @ other], axis=1)
        try:
            polygon = points[ConvexHull(projection).vertices]
        except QhullError:
            continue  # A degenerate cooked facet cannot supply finite area.
        edge_planes = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            normal = np.cross(b-a, plane[:3])
            normal /= np.linalg.norm(normal)
            offset = -float(normal @ a)
            if normal @ center+offset > 0:
                normal, offset = -normal, -offset
            edge_planes.append(np.r_[normal, offset])
        result.append({'plane': plane, 'projection_edge_planes': np.asarray(edge_planes),
                       'polygon': polygon, 'cooked_triangle_faces': group['faces']})
    return result


def world_facets(facets, matrix):
    return [{'plane': planes_in_body(f['plane'][None], matrix)[0],
             'projection_edge_planes': planes_in_body(f['projection_edge_planes'], matrix),
             'polygon': apply_transform(f['polygon'], matrix),
             'cooked_triangle_faces': list(f['cooked_triangle_faces'])} for f in facets]


def clipped_facet_region(triangle, facet, pad, *, band, penetration, minimum_area,
                         surface_query, polygon_area):
    """Clip one real triangle to a facet prism and verify its closest surface."""
    normal, offset = facet['plane'][:3], facet['plane'][3]
    polygon = triangle.copy()
    planes = np.vstack([facet['projection_edge_planes'],
                        np.r_[normal, offset-band], np.r_[-normal, -offset-penetration]])
    for plane in planes:
        polygon = clip_polygon(polygon, plane[:3], plane[3])
        if len(polygon) < 3:
            return None
    area = polygon_area(polygon)
    if area < minimum_area:
        return None
    point = polygon.mean(0)
    distance, nearest, gradient = surface_query(point, pad['inner'], pad['faces'],
        pad['inner_planes'], contact_normal_profile='distance_gradient_v1')
    if (gradient is None or distance > band+1.e-9 or distance < -penetration
            or float(gradient @ normal) < 1.-1.e-6
            or abs(float(nearest @ normal+offset)) > 1.e-8
            or np.max(facet['projection_edge_planes'][:, :3] @ nearest
                      +facet['projection_edge_planes'][:, 3]) > 1.e-8):
        return None
    return {'point_m': point.tolist(), 'surface_point_m': nearest.tolist(),
            'normal_world': gradient.tolist(), 'signed_inner_distance_m': float(distance),
            'clipped_area_m2': float(area), 'clipped_polygon_m': polygon.tolist(),
            'verified_facet_normal_world': normal.tolist(),
            'normal_agreement': float(gradient @ normal),
            'cooked_triangle_faces': list(facet['cooked_triangle_faces'])}


def material_facet_witnesses(pad, xyz, tri, rest, candidates, axis, orientation,
                             alo, bhi, thickness, band, penetration, minimum_area,
                             *, surface_query, polygon_area):
    rows = []
    for facet in pad['witness_facets']:
        if orientation*float(facet['plane'][:3] @ axis) < .25:
            continue
        for tid in candidates:
            row = clipped_facet_region(xyz[tid], facet, pad, band=band, penetration=penetration,
                minimum_area=minimum_area, surface_query=surface_query, polygon_area=polygon_area)
            if row is None:
                continue
            point = np.asarray(row['point_m'])
            along = float(point @ axis)
            projection = orientation*float(np.asarray(row['normal_world']) @ axis)
            if (projection < .25 or along < alo-thickness/2-penetration
                    or along > bhi+thickness/2+penetration):
                continue
            weights = np.linalg.lstsq((xyz[tid, 1:]-xyz[tid, 0]).T,
                                      point-xyz[tid, 0], rcond=None)[0]
            material = rest[tri[tid]]
            row.update(triangle_id=int(tid), rest_point_m=(material[0]+weights @ (material[1:]-material[0])).tolist(),
                       inward_normal_projection=projection)
            rows.append(row)
    return rows
