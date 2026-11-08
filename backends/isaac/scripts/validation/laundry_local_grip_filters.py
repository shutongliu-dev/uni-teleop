"""Bounded triangle/hand collision filters for declared engineering clamps.

These filters resolve a local rolling-attachment/contact conflict; they do not
qualify a friction grasp. They never change cloth state, self collision, table
collision or whole-body collision groups. Restoration uses explicit native
rigid poses, never a possibly stale USD/Fabric world transform.
"""
from __future__ import annotations

from itertools import product
import numpy as np


PROFILE = 'laundry_local_grip_triangle_filter.v1'
MAX_REGION_AREA_FRACTION = .03
MAX_UNION_AREA_FRACTION = .05
MAX_WORLD_RADIUS_M = .065
MAX_MATERIAL_DISTANCE_M = .030
RESTORE_CLEARANCE_M = .010


def _points(value, *, count=None):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 2 or value.shape[1] != 3 or not len(value)
            or not np.isfinite(value).all() or (count is not None and len(value) != count)):
        raise ValueError('Complete finite cloth vertices are required')
    return value


def _indices(value, limit, name):
    value = np.asarray(value)
    if (value.ndim != 1 or not len(value) or value.dtype.kind not in 'iu'
            or value.min() < 0 or value.max() >= limit):
        raise ValueError('Nonempty valid integer '+name+' required')
    return np.unique(value.astype(np.int64))


def local_triangle_region(rest, triangles, q, anchor_vertex_ids, *, existing_triangle_ids=()):
    """Incident triangles plus one vertex-sharing ring, fixed at binding time."""
    rest = _points(rest)
    q = _points(q, count=len(rest))
    triangles = np.asarray(triangles)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.dtype.kind not in 'iu' or triangles.min() < 0
            or triangles.max() >= len(rest)):
        raise ValueError('Valid triangular cloth topology required')
    triangles = triangles.astype(np.int64)
    anchor = _indices(anchor_vertex_ids, len(rest), 'grasp vertex indices')
    incident = np.flatnonzero(np.isin(triangles, anchor).any(axis=1))
    if not len(incident):
        raise ValueError('Grasp vertices have no incident triangles')
    one_ring_vertices = np.unique(triangles[incident])
    chosen = np.flatnonzero(np.isin(triangles, one_ring_vertices).any(axis=1))
    vertices = np.unique(triangles[chosen])
    origin = q[anchor].mean(axis=0)
    radius = float(np.linalg.norm(q[vertices]-origin, axis=1).max())
    material_distance = float(np.linalg.norm(
        rest[vertices, None]-rest[anchor][None], axis=2).min(axis=1).max())
    if radius > MAX_WORLD_RADIUS_M+1.e-12:
        raise ValueError('Local filter world radius exceeds 65 mm')
    if material_distance > MAX_MATERIAL_DISTANCE_M+1.e-12:
        raise ValueError('Local filter material distance exceeds 30 mm')
    area = np.linalg.norm(np.cross(rest[triangles[:, 1]]-rest[triangles[:, 0]],
                                  rest[triangles[:, 2]]-rest[triangles[:, 0]]), axis=1)/2
    total = float(area.sum())
    if not np.isfinite(area).all() or total <= 0:
        raise ValueError('Positive finite rest surface area required')
    fraction = float(area[chosen].sum()/total)
    if fraction > MAX_REGION_AREA_FRACTION+1.e-12:
        raise ValueError('Local filter exceeds three percent of rest area')
    existing = np.asarray(existing_triangle_ids)
    if existing.size:
        existing = _indices(existing, len(triangles), 'existing triangle indices')
        union = np.union1d(chosen, existing)
    else:
        union = chosen
    union_fraction = float(area[union].sum()/total)
    if union_fraction > MAX_UNION_AREA_FRACTION+1.e-12:
        raise ValueError('Active filter union exceeds five percent of rest area')
    return {
        'triangle_ids': chosen.tolist(), 'vertex_ids': vertices.tolist(),
        'grasp_vertex_ids': anchor.tolist(), 'bind_origin_m': origin.tolist(),
        'selection': 'incident triangles plus one vertex-sharing triangle ring',
        'rest_area_m2': float(area[chosen].sum()), 'total_rest_area_m2': total,
        'rest_area_fraction': fraction, 'active_union_rest_area_fraction': union_fraction,
        'world_radius_m': radius, 'maximum_material_distance_m': material_distance,
        'material_triangle_inventory_fixed_after_bind': True,
    }


def _rotation_xyzw(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or abs(norm-1.) > .01:
        raise ValueError('Native body quaternion must be normalized xyzw')
    x, y, z, w = quaternion/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def native_collider_boxes(bindings, native_body_q, body_count):
    """Transform cached rigid-link-local corners with current native poses."""
    poses = np.asarray(native_body_q, dtype=float)
    if poses.shape != (body_count, 7) or not np.isfinite(poses).all():
        raise ValueError('Current native body poses with shape (body_count, 7) required')
    boxes = []
    for binding in bindings:
        pose = poses[binding['body_index']]
        corners = np.asarray(binding['body_local_corners_m']) @ _rotation_xyzw(pose[3:]).T+pose[:3]
        boxes.append([corners.min(axis=0), corners.max(axis=0)])
    if not boxes:
        raise ValueError('No actual hand collider bounds available')
    return np.asarray(boxes)


def triangle_box_clearance(q, triangles, triangle_ids, boxes):
    """Conservative lower bound, including triangle interiors, in world metres."""
    points = _points(q)
    faces = np.asarray(triangles, dtype=int)[np.asarray(triangle_ids, dtype=int)]
    if not len(faces):
        raise ValueError('Cannot restore an empty filter region')
    triangle_points = points[faces]
    low, high = triangle_points.min(axis=1), triangle_points.max(axis=1)
    boxes = np.asarray(boxes, dtype=float)
    if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 0] > boxes[:, 1])):
        raise ValueError('Current hand collider boxes required')
    gap = np.maximum(0., np.maximum(low[:, None]-boxes[None, :, 1],
                                    boxes[None, :, 0]-high[:, None]))
    return float(np.linalg.norm(gap, axis=2).min())


class LocalGripFilter:
    def __init__(self, manager, side, region, paths, bindings):
        self.manager, self.side = manager, side
        self.triangle_ids = tuple(region['triangle_ids'])
        self.paths, self.bindings = tuple(paths), bindings
        self.active, self.released = True, False
        self.receipt = {
            'side': side, **region, 'filter_paths': list(paths),
            'collider_paths': [row['collider_path'] for row in bindings],
            'rigid_body_names': [row['body_name'] for row in bindings],
            'restore_clearance_required_m': RESTORE_CLEARANCE_M,
            'active': True, 'min_clearance_m': None,
            'filter_enabled_readback': {path: True for path in paths},
            'native_filter_effect_independently_verified': False,
            'restoration_pose_source': 'explicit native_body_q xyzw indexed by env.body_names',
        }

    def try_restore(self, q, native_body_q):
        """Call only after disabling the associated attachment; never on a timer."""
        if not self.active:
            return True
        if not self.released:
            raise ValueError('Disable the attachment and mark this filter released first')
        q = _points(q, count=len(self.manager.rest))
        boxes = native_collider_boxes(self.bindings, native_body_q, len(self.manager.body_names))
        clearance = triangle_box_clearance(q, self.manager.triangles, self.triangle_ids, boxes)
        self.receipt['min_clearance_m'] = clearance
        if clearance < RESTORE_CLEARANCE_M:
            return False
        for path in self.paths:
            self.manager.stage.GetPrimAtPath(path).GetAttribute('omniphysics:filterEnabled').Set(False)
        self.manager.flush_changes()
        readback = {path: self.manager.stage.GetPrimAtPath(path).GetAttribute(
            'omniphysics:filterEnabled').Get() for path in self.paths}
        if any(value is not False for value in readback.values()):
            raise RuntimeError('Local collision filter restoration readback failed')
        self.active = False
        self.receipt.update(active=False, filter_enabled_readback=readback)
        return True


class LocalGripFilters:
    """One garment's local engineering filters, with union and restoration bounds."""
    def __init__(self, env, rest, triangles, *, cloth_path=None, flush_changes=None):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.stage, self.body_names = env.stage, tuple(env.body_names)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError('Native rigid body names must be unique')
        self.rest = _points(rest)
        self.triangles = np.asarray(triangles)
        if cloth_path is None:
            cloths = env.scene_info['cloths']
            if len(cloths) != 1:
                raise ValueError('An explicit cloth path is required for multiple cloths')
            cloth_path = next(iter(cloths.values()))['path']
        self.cloth_path = str(cloth_path)
        prim = self.stage.GetPrimAtPath(self.cloth_path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Cloth source must be its explicit triangular CollisionAPI mesh')
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        if (not len(counts) or np.any(counts != 3) or self.triangles.dtype.kind not in 'iu'
                or self.triangles.shape != (len(counts), 3)
                or not np.array_equal(indices.reshape(-1, 3), self.triangles)):
            raise ValueError('Filter triangle IDs must match the actual collision mesh topology')
        if len(mesh.GetPointsAttr().Get()) != len(self.rest):
            raise ValueError('Cloth collision vertex inventory differs from rest geometry')
        self.triangles = self.triangles.astype(np.int64)
        if flush_changes is None:
            def flush_changes():
                import omni.physx
                omni.physx.get_physx_simulation_interface().flush_changes()
        self.flush_changes = flush_changes
        self.items = []
        self.receipt = {'profile': PROFILE, 'cloth_path': self.cloth_path,
            'engineering_scope': 'Only local garment triangles against the associated hand; not a friction grasp.',
            'region_area_fraction_limit': MAX_REGION_AREA_FRACTION,
            'active_union_area_fraction_limit': MAX_UNION_AREA_FRACTION,
            'world_radius_limit_m': MAX_WORLD_RADIUS_M,
            'material_distance_limit_m': MAX_MATERIAL_DISTANCE_M, 'items': []}
        # USD is used only for static collider geometry and its fixed transform
        # relative to its rigid link. Live poses always come from native arrays.
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'])
        self.bindings = {side: [] for side in ('left', 'right')}
        for collider in Usd.PrimRange(self.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not collider.HasAPI(UsdPhysics.CollisionAPI) or not collider.IsA(UsdGeom.Boundable):
                continue
            parts = str(collider.GetPath()).split('/')
            sides = [side for side in self.bindings if side+'_hand_wrist' in parts]
            if len(sides) != 1:
                continue
            body = collider
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.IsPseudoRoot() or body.GetName() not in self.body_names:
                raise ValueError('Hand collider has no corresponding native rigid body: '+str(collider.GetPath()))
            box = bounds.ComputeUntransformedBound(collider).ComputeAlignedRange()
            if box.IsEmpty():
                raise ValueError('Hand collider has no finite geometric bounds')
            low, high = np.array(box.GetMin()), np.array(box.GetMax())
            corners = np.array(list(product(*zip(low, high))))
            matrix = np.array(xforms.GetLocalToWorldTransform(collider)
                              * xforms.GetLocalToWorldTransform(body).GetInverse())
            local = np.c_[corners, np.ones(8)] @ matrix
            if not np.isfinite(local).all():
                raise ValueError('Invalid hand collider local bounds')
            self.bindings[sides[0]].append({'collider_path': str(collider.GetPath()),
                'body_name': body.GetName(), 'body_index': self.body_names.index(body.GetName()),
                'body_local_corners_m': local[:, :3].tolist()})
        if any(not rows for rows in self.bindings.values()):
            raise ValueError('Both actual hand collider inventories are required')

    @property
    def active_count(self):
        return sum(item.active for item in self.items)

    def bind(self, side, vertexids, q, path):
        from pxr import Sdf, UsdGeom
        if side not in self.bindings:
            raise ValueError('Expected left or right native hand')
        previous = [tid for item in self.items if item.active for tid in item.triangle_ids]
        region = local_triangle_region(self.rest, self.triangles, q, vertexids,
                                       existing_triangle_ids=previous)
        root = Sdf.Path(str(path))
        if (not root.IsAbsolutePath() or not root.IsPrimPath() or root == Sdf.Path.absoluteRootPath
                or self.stage.GetPrimAtPath(root)):
            raise ValueError('Use a fresh absolute filter container path')
        bindings = self.bindings[side]
        paths = [str(root.AppendChild(f'collider_{i:03d}')) for i in range(len(bindings))]
        # Complete validation precedes any mutation. In particular N=0 or an
        # empty counts0 has whole-cloth semantics in OmniPhysics and is forbidden.
        ids = region['triangle_ids']
        if not ids:
            raise ValueError('Empty local triangle filters are forbidden')
        try:
            UsdGeom.Xform.Define(self.stage, root)
            for filter_path, binding in zip(paths, bindings):
                prim = self.stage.DefinePrim(filter_path, 'OmniPhysicsElementCollisionFilter')
                prim.CreateAttribute('omniphysics:filterEnabled', Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateRelationship('omniphysics:src0').SetTargets([self.cloth_path])
                prim.CreateRelationship('omniphysics:src1').SetTargets([binding['collider_path']])
                prim.CreateAttribute('omniphysics:groupElemCounts0', Sdf.ValueTypeNames.UIntArray).Set([len(ids)])
                prim.CreateAttribute('omniphysics:groupElemIndices0', Sdf.ValueTypeNames.UIntArray).Set(ids)
                prim.CreateAttribute('omniphysics:groupElemCounts1', Sdf.ValueTypeNames.UIntArray).Set([0])
                prim.CreateAttribute('omniphysics:groupElemIndices1', Sdf.ValueTypeNames.UIntArray).Set([])
            self.flush_changes()
            if any(self.stage.GetPrimAtPath(p).GetAttribute('omniphysics:filterEnabled').Get() is not True
                   for p in paths):
                raise RuntimeError('Local collision filter enable readback failed')
        except BaseException:
            self.stage.RemovePrim(root)
            self.flush_changes()
            raise
        item = LocalGripFilter(self, side, region, paths, bindings)
        self.items.append(item)
        self.receipt['items'].append(item.receipt)
        return item
