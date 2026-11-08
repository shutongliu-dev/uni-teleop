"""Conservative source-shape support/interference diagnostics, without physics.

Only source convexHull meshes and exact cubes admit the polytope narrow phase.
An SDF, sphere or other curved shape is never silently convexified. A pair with
an SDF or plain triangle-mesh member is resolved by the exact closed-mesh depth
of the physical reset admission (R3) instead: exact halfspaces for polytopes,
generalized winding number plus point-triangle distance for closed meshes, and
the task referee bound R3 uses as its threshold. Curved, open, bounding,
decomposed or simplified shapes stay unresolved. Native contacts and cooking
remain separate evidence even for a resolved source pair.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from scripts.validation import living_randomized_workspace as living

ROOM_SUPPORTS = {
    'kitchen': ('worktop', 'kitchen_counter_middle', 'kitchen_cooktop'),
    'laboratory': ('worktop', 'rear_lab_counter'),
}


def source_polytope(stage, path):
    from pxr import UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    prim = stage.GetPrimAtPath(path)
    living.require(bool(prim) and prim.HasAPI(UsdPhysics.CollisionAPI)
                   and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,
                   'Source collider is missing or disabled')
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
        points = np.array(list(product((-half, half), repeat=3)))
    elif prim.IsA(UsdGeom.Mesh) and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull':
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    else:
        return None
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    points = (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    living.require(points.shape[1] == 3 and np.isfinite(points).all(), 'Invalid source convex points')
    hull = ConvexHull(points)
    # Qhull equations have unit normals, hence the fourth LP coordinate is m.
    return hull.equations


def intersection_radius(first, second):
    """Maximum inscribed sphere of the intersection; negative means separated."""
    from scipy.optimize import linprog
    planes = np.concatenate((first, second))
    norms = np.linalg.norm(planes[:, :3], axis=1)
    living.require(np.all(norms > 0) and np.isfinite(planes).all(), 'Invalid convex halfspaces')
    planes = planes / norms[:, None]
    result = linprog([0., 0., 0., -1.], A_ub=np.c_[planes[:, :3], np.ones(len(planes))],
                     b_ub=-planes[:, 3], bounds=[(None, None)]*4, method='highs')
    living.require(result.success and np.isfinite(result.x).all(), 'Source convex intersection solve failed')
    return float(result.x[3])


def exact_mesh_pair(stage, first, second, task):
    """Exact closed-mesh depth of a pair with an SDF/triangle-mesh member, else None.

    Both members must be exact source solids (cube, convexHull polytope, or a
    closed SDF/triangle mesh) and at least one a mesh; convex pairs keep the LP.
    The threshold is the R3 one: half of the task referee penetration bound.
    """
    from scenes.workrooms.parameterized import reset_admission as admission
    solids = [admission.exact_source_solid(stage, path) for path in (first, second)]
    if any(solid is None for solid in solids) or all(solid['kind'] == 'convex' for solid in solids):
        return None
    depth = float(admission.pair_depth(*solids))
    threshold = admission.PENETRATION_FRACTION*admission.referee_penetration_bound(task)
    living.require(np.isfinite(depth) and depth >= 0 and threshold > 0, 'Invalid exact source mesh depth')
    return {'narrow_phase': 'exact_closed_mesh_depth',
            'source_solids': [{'prim_path': path, 'kind': solid['kind'],
                               'approximation': solid.get('approximation', 'cube')}
                              for path, solid in zip((first, second), solids)],
            'exact_source_depth_m': depth, 'positive_depth_threshold_m': threshold,
            'positive_volume_intersection_above_numeric_tolerance': depth > threshold}


def analytic_core(stage, path):
    """A sphere or capsule collider as its exact core (point or segment) and radius, else None.

    PhysX spheres and capsules are exactly a point or a segment swept by a ball.
    Only rigid, uniformly scaled frames keep that form; cylinders, cones and every
    approximated or decomposed shape have no exact core here.
    """
    from pxr import UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if (not prim or not (prim.IsA(UsdGeom.Sphere) or prim.IsA(UsdGeom.Capsule))
            or not prim.HasAPI(UsdPhysics.CollisionAPI)):
        return None
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    if not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0., 0., 0., 1.]):
        return None
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not scale > 0 or not np.allclose(linear.T@linear, scale*scale*np.eye(3), rtol=0, atol=1e-9*scale*scale):
        return None
    if prim.IsA(UsdGeom.Sphere):
        radius, half, axis = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()), 0., 'Z'
        kind = 'sphere'
    else:
        capsule = UsdGeom.Capsule(prim)
        radius, half = float(capsule.GetRadiusAttr().Get()), float(capsule.GetHeightAttr().Get())/2
        axis, kind = str(capsule.GetAxisAttr().Get()), 'capsule'
    if axis not in ('X', 'Y', 'Z') or not np.isfinite([radius, half]).all() or radius <= 0 or half < 0:
        return None
    offset = np.zeros(3); offset['XYZ'.index(axis)] = half
    start, end = (matrix@[*-offset, 1.])[:3], (matrix@[*offset, 1.])[:3]
    return {'kind': kind, 'prim_path': path, 'start_m': start, 'end_m': end, 'radius_m': radius*scale}


def exact_analytic_pair(stage, first, second, task):
    """Exact depth of a sphere/capsule pair member against an exact solid, else None.

    The other member is another sphere/capsule, a cube/convexHull polytope or a
    closed SDF/triangle mesh. Depth is the radius sum minus the exact core
    distance; a core touching a polytope adds its exact separating translation.
    A core touching a non-convex mesh has no closed form here and stays
    unresolved. The threshold is the R3 one: half of the task referee bound.
    """
    from scenes.workrooms.parameterized import reset_admission as admission
    cores = [analytic_core(stage, path) for path in (first, second)]
    if all(core is None for core in cores):
        return None
    if all(core is not None for core in cores):
        a, b = cores
        distance = admission.segment_segment_distance(a['start_m'], a['end_m'], b['start_m'], b['end_m'])
        depth, other = max(0., a['radius_m']+b['radius_m']-distance), None
    else:
        core, path = (cores[0], second) if cores[0] is not None else (cores[1], first)
        solid = admission.exact_source_solid(stage, path)
        if solid is None:
            return None
        start, end = core['start_m'], core['end_m']
        distance = (admission.point_distance(solid, start) if np.allclose(start, end, rtol=0, atol=1e-12)
                    else admission.segment_distance(solid, start, end))
        if distance > 0.:
            depth = max(0., core['radius_m']-distance)
        elif solid['kind'] == 'convex':
            depth = core['radius_m']+admission.core_penetration(solid, start, end)
        else:
            return None
        other = {'prim_path': path, 'kind': solid['kind'], 'approximation': solid.get('approximation', 'cube')}
    threshold = admission.PENETRATION_FRACTION*admission.referee_penetration_bound(task)
    living.require(np.isfinite(depth) and depth >= 0 and np.isfinite(distance) and threshold > 0,
                   'Invalid exact analytic source depth')
    solids = [{'prim_path': core['prim_path'], 'kind': core['kind'], 'radius_m': core['radius_m'],
               'core_m': [core['start_m'].tolist(), core['end_m'].tolist()]} for core in cores if core is not None]
    return {'narrow_phase': 'exact_analytic_core_distance', 'source_solids': solids+([other] if other else []),
            'exact_core_distance_m': float(distance), 'exact_source_depth_m': float(depth),
            'positive_depth_threshold_m': threshold,
            'positive_volume_intersection_above_numeric_tolerance': depth > threshold}


def fixed_supports(stage, task, paths):
    from pxr import Usd, UsdPhysics
    family = task['reset_case']['family']
    living.require(family in ROOM_SUPPORTS, 'No declared room support family')
    roots = ['/World/Room/Geometry/'+name+'/Collider' for name in ROOM_SUPPORTS[family]]
    roots = [root for root in roots if stage.GetPrimAtPath(root)]
    fixture_roots = list(task.get('asset_metadata', {}).get('fixtures', {}).values())
    declared = list(task['reset_case'].get('fixed_support_paths', []))
    living.require(all(isinstance(path, str) and any(path == root or path.startswith(root+'/')
        for root in fixture_roots) for path in declared), 'Fixed task support is outside original fixture identities')
    roots += declared
    living.require(len(roots) == len(set(roots)), 'Duplicate fixed support root')
    colliders = {}
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        living.require(bool(prim) and not root.startswith(('/World/Robots/', '/World/Environment/')),
                       'Fixed support root missing or outside task/worktop geometry')
        enabled = []
        for child in Usd.PrimRange(prim):
            if not child.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is False:
                continue
            path = str(child.GetPath())
            living.require(not any(path == body or path.startswith(body+'/') for body in paths.values()),
                           'A dynamic actor cannot be a fixed support')
            ancestor = child
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    api = UsdPhysics.RigidBodyAPI(ancestor)
                    living.require(api.GetRigidBodyEnabledAttr().Get() is False
                                   or api.GetKinematicEnabledAttr().Get() is True,
                                   'Declared fixed support contains a free rigid body')
                ancestor = ancestor.GetParent()
            enabled.append(path)
        living.require(bool(enabled), 'Declared fixed support contains no enabled collision shape')
        colliders[root] = sorted(enabled)
    return colliders


def _filters(stage):
    from pxr import UsdPhysics
    pairs = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is not False and joint.GetCollisionEnabledAttr().Get() is False:
                for a in joint.GetBody0Rel().GetTargets():
                    for b in joint.GetBody1Rel().GetTargets():
                        pairs.append((str(a), str(b), str(prim.GetPath())))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                pairs.append((str(prim.GetPath()), str(other), str(prim.GetPath())+'.physics:filteredPairs'))
    return pairs


def review_source_geometry(checker, paths, task):
    inventory = living.shape_inventory(checker, paths)
    roots = fixed_supports(checker.stage, task, paths)
    fixed = {path for values in roots.values() for path in values}
    filters = _filters(checker.stage)
    shapes = inventory['shapes']
    resolved, unresolved, filtered = [], [], []
    hulls = {}
    def within(path, root):
        return path == root or path.startswith(root+'/')
    for i, first in enumerate(shapes):
        for second in shapes[i+1:]:
            if first['body_id'] == second['body_id'] or (first['body_id'] is None and second['body_id'] is None):
                continue
            a, b = np.asarray(first['bounds_m']), np.asarray(second['bounds_m'])
            if np.any(np.minimum(a[1], b[1])-np.maximum(a[0], b[0]) <= 1e-10):
                continue
            p, q = first['prim_path'], second['prim_path']
            row = {'first': p, 'second': q, 'first_body': first['body_id'], 'second_body': second['body_id']}
            disabled = [reason for x, y, reason in filters
                        if (within(p, x) and within(q, y)) or (within(p, y) and within(q, x))]
            if disabled:
                filtered.append({**row, 'source_filter_paths': disabled}); continue
            for path in (p, q):
                if path not in hulls:
                    hulls[path] = source_polytope(checker.stage, path)
            if hulls[p] is None or hulls[q] is None:
                exact = exact_mesh_pair(checker.stage, p, q, task)
                if exact is None:
                    unresolved.append({**row, 'reason': 'Curved/open/approximated source pair has neither convex halfspaces nor an exact closed mesh.'})
                else:
                    resolved.append({**row, **exact})
            else:
                radius = intersection_radius(hulls[p], hulls[q])
                resolved.append({**row, 'intersection_inradius_m': radius,
                                 'positive_volume_intersection_above_numeric_tolerance': radius > 1e-8})
    # Preserve the original near-support tolerance, but exclude room floors,
    # robots and arbitrary static geometry as permissible roots.
    support = [row for row in inventory['support_candidates']
               if row['other_body_id'] is not None or row['other_shape'] in fixed]
    rooted = {row['body_id'] for row in support if row['other_shape'] in fixed}
    changed = True
    while changed:
        previous = set(rooted)
        rooted.update(row['body_id'] for row in support if row['other_body_id'] in rooted)
        for joint in inventory['joints']:
            # A joint only transfers geometric rootedness between its declared
            # task actors; merely having a joint never fabricates a floor root.
            members = {name for name, path in paths.items() if path in joint['body0']+joint['body1']}
            if members & rooted:
                rooted.update(members)
        changed = rooted != previous
    return {'schema': 'workroom_source_interference.v1', 'fixed_support_colliders': roots,
        'geometric_support_candidates': support, 'bodies_without_geometric_support_chain': sorted(set(paths)-rooted),
        'convex_pairs': resolved, 'filtered_source_pairs': filtered, 'unresolved_source_pairs': unresolved,
        'positive_source_convex_pairs': [row for row in resolved if row['positive_volume_intersection_above_numeric_tolerance']],
        'all_broadphase_candidates_resolved': not unresolved,
        'scope': 'Authored geometry only. Exact convex-halfspace LP in metres for cubes/source convex hulls; exact closed-mesh depth against half the task referee penetration bound for SDF/triangle-mesh pairs; other positive AABB candidates remain unresolved. Near supports and joints are geometric evidence, not normal forces, native cooking, passive stability or full manipulation.',
        'native_support_verified': False, 'physics_qualified': False}
