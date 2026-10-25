"""Continuous 360-degree yaw enclosures of actual convex collision meshes.

These CPU bounds describe the stated upright, centred tool poses. They do not
qualify a tilted/displaced tool, a hand, or native contact dynamics.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import ConvexHull


def clipped_slab_vertices(mesh, low, high):
    """Vertices of a convex mesh intersected with a closed horizontal slab.

    Include every edge/plane intersection, not only original vertices whose Z
    lies in the slab. This also supports triangular coplanar/tangent sections.
    """
    vertices = np.asarray(mesh.vertices, float)
    edges = vertices[np.asarray(mesh.edges_unique)]
    points = [v for v in vertices if low <= v[2] <= high]
    for z in (low, high):
        for a, b in edges:
            if a[2] != b[2] and min(a[2], b[2]) <= z <= max(a[2], b[2]):
                points.append(a+(b-a)*((z-a[2])/(b[2]-a[2])))
    return np.unique(np.asarray(points, float).reshape(-1, 3), axis=0)


def radial_interval(points):
    """Exact minimum/maximum radius of a convex XY projection."""
    xy = np.unique(np.asarray(points, float)[:, :2], axis=0)
    if not len(xy):
        raise ValueError('An empty slice has no radial interval')
    maximum = float(np.linalg.norm(xy, axis=1).max())
    if len(xy) == 1:
        return maximum, maximum
    if np.linalg.matrix_rank(xy-xy[0], tol=1.e-12) < 2:
        _, _, axes = np.linalg.svd(xy-xy[0], full_matrices=False)
        projection = xy @ axes[0]
        polygon = xy[[np.argmin(projection), np.argmax(projection)]]
    else:
        hull = ConvexHull(xy)
        if np.all(hull.equations[:, 2] <= 0.):
            return 0., maximum
        polygon = xy[hull.vertices]
    nxt = np.roll(polygon, -1, axis=0)
    direction = nxt-polygon
    norm = np.sum(direction*direction, axis=1)
    valid = norm > 0.
    fraction = np.zeros(len(norm))
    fraction[valid] = np.clip(-np.sum(polygon[valid]*direction[valid], axis=1)/norm[valid], 0., 1.)
    minimum = float(np.linalg.norm(polygon+fraction[:, None]*direction, axis=1).min())
    return minimum, maximum


def yaw_sweep_pair_bound(tool_mesh, obstacle_mesh):
    """Lower bound for all yaw angles, without angular sampling.

    A convex slice's radii form an interval. Full yaw sweeps that interval into
    an annulus. The Cartesian product of radial/Z intervals encloses the solid;
    disjoint radial intervals prove clearance. When necessary, all cross-pairs
    of clipped Z slabs are compared, including points at different heights.
    Nonpositive bounds mean unresolved/possible intersection, never success.
    """
    ta, tb = np.asarray(tool_mesh.bounds)[:, 2]
    oa, ob = np.asarray(obstacle_mesh.bounds)[:, 2]
    tool_r = radial_interval(tool_mesh.vertices)
    other_r = radial_interval(obstacle_mesh.vertices)
    radial_gap = max(tool_r[0]-other_r[1], other_r[0]-tool_r[1], 0.)
    z_gap = max(ta-ob, oa-tb, 0.)
    global_bound = math.hypot(radial_gap, z_gap)
    if global_bound > 1.e-10:
        return global_bound
    levels = sorted(set(map(float, np.r_[tool_mesh.vertices[:, 2], obstacle_mesh.vertices[:, 2]])))
    def boxes(mesh):
        result = []
        for a, b in zip(levels, levels[1:]):
            points = clipped_slab_vertices(mesh, a, b)
            if len(points):
                result.append((*radial_interval(points), float(points[:,2].min()), float(points[:,2].max())))
        return result
    gaps = []
    for a, b, low, high in boxes(tool_mesh):
        for c, d, bottom, top in boxes(obstacle_mesh):
            gaps.append(math.hypot(max(a-d, c-b, 0.), max(low-top, bottom-high, 0.)))
    return min(gaps)


def tool_sweep_audit(assets, *, tool_bottom_z=.820, center_distance=.130):
    """Check every actual tool collider against non-mating assembled geometry.

    All meshes use the tool shaft as world-XY origin. The input gear itself is
    the intentional mating part and is excluded; its fixed guide shaft remains
    checked. Idler/output surfaces include their actual teeth and hubs.
    """
    from .assets import TRAIN_GEOMETRY, _box
    tools = []
    for label, mesh, _ in assets['socket_crank'].collision:
        transformed = mesh.copy()
        transformed.apply_translation([0., 0., tool_bottom_z])
        tools.append((label, transformed))
    obstacles = []
    # Workplate coordinates are relative to its middle shaft. Reposition the
    # existing mesh groups to permit a same-tool historical 100-mm control.
    for label, mesh, _ in assets['gear_workplate'].collision:
        transformed = mesh.copy()
        original_y = 0.
        desired_y = 0.
        if label != 'workplate':
            index = int(label.split('_')[-2])
            original_y = TRAIN_GEOMETRY['shaft_centers_local_xy_m'][index][1]
            desired_y = (index-1)*center_distance
        transformed.apply_translation([0., center_distance+desired_y-original_y, .78])
        group = ('middle_shaft' if label.startswith(('guide_shaft_1_', 'tapered_tip_1_')) else
                 'input_shaft' if label.startswith(('guide_shaft_0_', 'tapered_tip_0_')) else
                 'output_shaft' if label.startswith(('guide_shaft_2_', 'tapered_tip_2_')) else 'workplate_support')
        obstacles.append((group, label, transformed))
    # A historical centre-distance comparison covers fixed shaft clearance
    # only: moving the new large gear to 100 mm would intentionally mis-mesh.
    if center_distance == TRAIN_GEOMETRY['adjacent_center_distance_m']:
        for name, y, angle in (('idler_gear', center_distance, math.pi/32),
                               ('output_gear', 2*center_distance, 0.)):
            c, s = math.cos(angle), math.sin(angle)
            matrix = np.eye(4);matrix[:2,:2] = [[c, -s], [s, c]]
            matrix[:3,3] = [0., y, .802]
            for label, mesh, _ in assets[name].collision:
                transformed = mesh.copy();transformed.apply_transform(matrix)
                obstacles.append((name, label, transformed))
    groups = {}
    for group, obstacle_label, obstacle in obstacles:
        for tool_label, tool in tools:
            bound = yaw_sweep_pair_bound(tool, obstacle)
            record = {'lower_bound_m': bound, 'tool_piece': tool_label, 'obstacle_piece': obstacle_label}
            if group not in groups or bound < groups[group]['lower_bound_m']:
                groups[group] = record
    fixed_neighbor_groups = {name: dict(groups[name]) for name in ('idler_gear', 'output_gear') if name in groups}
    # Neighbor gears also rotate during transmission. An outer square prism
    # contains every yaw of the real mesh, including all teeth and hub levels.
    # It deliberately fills empty corners and the bore; positive clearance to
    # this superset proves clearance independently of both neighbor phases.
    if center_distance == TRAIN_GEOMETRY['adjacent_center_distance_m']:
        for name, center in (('idler_gear', center_distance), ('output_gear', 2*center_distance)):
            points = np.vstack([m.vertices for _, m, _ in assets[name].collision])
            radius = float(np.linalg.norm(points[:, :2], axis=1).max())
            low, high = float(points[:, 2].min()), float(points[:, 2].max())
            enclosure = _box([2*radius, 2*radius, high-low], [0., center, .802+(high+low)/2])
            rows = [{'lower_bound_m': yaw_sweep_pair_bound(tool, enclosure),
                     'tool_piece': label, 'obstacle_piece': 'all_yaw_outer_prism',
                     'actual_mesh_maximum_radius_m': radius, 'actual_mesh_local_z_bounds_m': [low, high]}
                    for label, tool in tools]
            groups[name] = min(rows, key=lambda row: row['lower_bound_m'])
    minimum = min(r['lower_bound_m'] for r in groups.values())
    return {'schema':'mechanical_tool_continuous_yaw_clearance.v1',
        'passed': minimum > 1.e-9, 'tool_bottom_world_z_m':tool_bottom_z,
        'center_distance_m':center_distance,'yaw_interval_degrees':[0.,360.],
        'method':'Actual convex mesh radial/Z enclosures, refined by all crossed horizontal edge/plane intersections; no angular sampling.',
        'maximum_tool_vertex_radius_m':max(float(np.linalg.norm(m.vertices[:,:2],axis=1).max()) for _,m in tools),
        'minimum_clearance_lower_bound_m':minimum,'groups':groups,
        'neighbor_gear_yaw_interval_degrees':[0.,360.],
        'neighbor_gear_method':'Outer square prisms contain every yaw of the actual neighboring mesh radii and Z bounds; independent of tooth phase.',
        'fixed_reset_phase_neighbor_mesh_bounds':fixed_neighbor_groups,
        'excluded_intentional_mate':'input_gear; its fixed guide shaft is still checked',
        'scope':'Only the stated upright centred socket and assembled gears/workplate. No hand, tilted/displaced tool, native cooked geometry or transmission qualification.'}
