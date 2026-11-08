"""Declared native volume FEM inventory and tensor-only node readback."""
from copy import deepcopy
import re

import numpy as np


def _schemas(prim):
    value = prim.GetMetadata('apiSchemas')
    return set(value.GetAppliedItems()) if value is not None else set()


def _array(value, columns, label, *, integer=False):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] != columns or not len(raw)
            or not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all()
            or integer and (not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_))):
        raise ValueError(label + ' requires a finite nonempty array with the declared shape')
    return raw.astype(np.int64 if integer else float)


def _same(actual, expected, label, *, integer=False):
    if actual is None:
        raise ValueError('Missing volume property: ' + label)
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not (np.array_equal(a, b) if integer else
                                 np.allclose(a, b, atol=2.e-7, rtol=0)):
        raise ValueError('Volume declaration differs from authored ' + label)


def _world_points(stage, path, points):
    from pxr import UsdGeom
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)), float)
    if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1.e-7, rtol=0):
        raise ValueError('Volume body geometry must bake metric scale and shear')
    return points @ matrix[:3, :3] + matrix[3, :3]


def validate_volume_bodies(stage, root, declarations, rigid_bodies):
    """Validate exact FEM/attachment inventory without changing source physics."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from .geometry import tet_surface_triangles
    if declarations is None:
        declarations = {}
    if not isinstance(declarations, dict):
        raise ValueError('volume_bodies must be an explicit mapping')
    root_path = root.GetPath()
    actual_roots, actual_simulation, actual_attachments, actual_filters = set(), set(), set(), set()
    for prim in Usd.PrimRange(root):
        schemas = _schemas(prim)
        if 'OmniPhysicsDeformableBodyAPI' in schemas:
            actual_roots.add(str(prim.GetPath()))
        if 'OmniPhysicsVolumeDeformableSimAPI' in schemas:
            actual_simulation.add(str(prim.GetPath()))
        if 'Attachment' in prim.GetTypeName():
            actual_attachments.add(str(prim.GetPath()))
        if prim.GetTypeName() == 'OmniPhysicsElementCollisionFilter':
            actual_filters.add(str(prim.GetPath()))
    result, body_paths, mesh_paths, attachment_paths, filter_paths = {}, set(), set(), set(), set()
    for name, supplied in declarations.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)
                or name in rigid_bodies or not isinstance(supplied, dict)):
            raise ValueError('Volume IDs must be distinct simple body identifiers')
        row = deepcopy(supplied)
        for key in ('prim_path', 'simulation_mesh_path'):
            path = Sdf.Path(row[key])
            if not path.IsPrimPath() or not path.IsAbsolutePath() or not path.HasPrefix(root_path) or path == root_path:
                raise ValueError('Volume body must remain inside its declared task root')
        path, mesh_path = row['prim_path'], row['simulation_mesh_path']
        if path in body_paths or mesh_path in mesh_paths or any(
                Sdf.Path(path).HasPrefix(Sdf.Path(p)) or Sdf.Path(p).HasPrefix(Sdf.Path(path))
                for p in (*body_paths, *rigid_bodies.values())):
            raise ValueError('Overlapping or duplicate rigid/volume body roots')
        body, mesh = stage.GetPrimAtPath(path), stage.GetPrimAtPath(mesh_path)
        if (not body or not mesh or Sdf.Path(mesh_path).GetParentPath() != Sdf.Path(path)
                or not {'OmniPhysicsDeformableBodyAPI', 'PhysxBaseDeformableBodyAPI'} <= _schemas(body)
                or 'OmniPhysicsVolumeDeformableSimAPI' not in _schemas(mesh)
                or mesh.GetTypeName() != 'TetMesh' or not mesh.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is False):
            raise ValueError('Volume body needs its declared enabled hierarchical TetMesh collider')
        owned_colliders = {str(p.GetPath()) for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)}
        if owned_colliders != {mesh_path}:
            raise ValueError('Undeclared collision geometry beneath volume body')
        if (body.GetAttribute('omniphysics:deformableBodyEnabled').Get() is not True
                or body.GetAttribute('omniphysics:kinematicEnabled').Get() is not False):
            raise ValueError('Volume body must be explicitly enabled and dynamic')
        points = _array(row['rest_points_m'], 3, 'Volume rest points')
        tets = _array(row['tetrahedra'], 4, 'Volume tetrahedra', integer=True)
        surface = _array(row['surface_triangles'], 3, 'Volume surface', integer=True)
        if min(tets.min(), surface.min()) < 0 or max(tets.max(), surface.max()) >= len(points):
            raise ValueError('Volume topology references missing nodes')
        if any(len(set(t)) != 4 for t in tets) or any(len(set(t)) != 3 for t in surface):
            raise ValueError('Volume topology has repeated element vertices')
        tetra = points[tets]
        determinant = np.linalg.det(np.stack([tetra[:, i]-tetra[:, 0] for i in (1, 2, 3)], axis=-1))
        # The pinned PhysX 110.3.2 updateNodeInverseVolumes rejects det <= 1e-9
        # even for correctly oriented tets. Reject these before native cooking.
        if np.any(determinant <= 1.e-9):
            raise ValueError('Volume rest tetrahedra must exceed the native 1e-9 determinant floor')
        if 'minimum_rest_tetrahedron_determinant_m3' in row and not np.isclose(
                determinant.min(), row['minimum_rest_tetrahedron_determinant_m3'], rtol=1.e-5, atol=1.e-15):
            raise ValueError('Declared minimum tetrahedron determinant differs from authored geometry')
        expected_surface = tet_surface_triangles(tets)
        canonical = lambda value: sorted(tuple(sorted(int(i) for i in tri)) for tri in value)
        if canonical(surface) != canonical(expected_surface):
            raise ValueError('Volume surface must be the complete tetrahedral boundary')
        tet = UsdGeom.TetMesh(mesh)
        for actual, expected, label, integer in (
                (tet.GetPointsAttr().Get(), points, 'points', False),
                (tet.GetTetVertexIndicesAttr().Get(), tets, 'tetVertexIndices', True),
                (tet.GetSurfaceFaceVertexIndicesAttr().Get(), surface, 'surfaceFaceVertexIndices', True),
                (mesh.GetAttribute('omniphysics:restShapePoints').Get(), points, 'restShapePoints', False),
                (mesh.GetAttribute('omniphysics:restTetVtxIndices').Get(), tets, 'restTetVtxIndices', True)):
            _same(actual, expected, label, integer=integer)
        mass = row['total_mass_kg']
        if isinstance(mass, bool) or not np.isfinite(mass) or mass <= 0:
            raise ValueError('Volume mass must be finite and positive')
        _same(body.GetAttribute('omniphysics:mass').Get(), mass, 'mass')
        parent = row['parent_rigid_body']
        if parent not in rigid_bodies:
            raise ValueError('Volume attachment parent must be a declared dynamic rigid body')
        bond = row['attachment']; bond_path = bond['prim_path']; attachment = stage.GetPrimAtPath(bond_path)
        if (not Sdf.Path(bond_path).HasPrefix(root_path) or bond_path in attachment_paths
                or not attachment or attachment.GetTypeName() != 'OmniPhysicsVtxXformAttachment'
                or attachment.GetAttribute('omniphysics:attachmentEnabled').Get() is not True):
            raise ValueError('Volume requires its explicit enabled native vertex/rigid attachment')
        indices = np.asarray(bond['node_indices'])
        anchors = _array(bond['rigid_local_positions_m'], 3, 'Attachment local anchors')
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= len(points)
                or len(set(indices.tolist())) != len(indices) or len(anchors) != len(indices)):
            raise ValueError('Attachment node inventory differs from rigid local anchors')
        for key, target in (('omniphysics:src0', mesh_path), ('omniphysics:src1', rigid_bodies[parent])):
            if [str(p) for p in attachment.GetRelationship(key).GetTargets()] != [target]:
                raise ValueError('Attachment endpoints differ from declared FEM/rigid bodies')
        _same(attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Get(), indices, 'attachment node IDs', integer=True)
        _same(attachment.GetAttribute('omniphysics:localPositionsSrc1').Get(), anchors, 'attachment local anchors')
        filtering = row.get('interface_collision_filter')
        if filtering is not None:
            filter_path = filtering['prim_path']; filter_prim = stage.GetPrimAtPath(filter_path)
            face_ids = np.asarray(filtering['surface_triangle_indices'])
            collider = filtering['rigid_collider_path']
            if (not Sdf.Path(filter_path).HasPrefix(root_path) or filter_path in filter_paths
                    or not filter_prim or filter_prim.GetTypeName() != 'OmniPhysicsElementCollisionFilter'
                    or filter_prim.GetAttribute('omniphysics:filterEnabled').Get() is not True
                    or not Sdf.Path(collider).HasPrefix(Sdf.Path(rigid_bodies[parent]))
                    or not stage.GetPrimAtPath(collider).HasAPI(UsdPhysics.CollisionAPI)
                    or face_ids.ndim != 1 or not len(face_ids) or not np.issubdtype(face_ids.dtype, np.integer)
                    or face_ids.min() < 0 or face_ids.max() >= len(surface)
                    or len(set(face_ids.tolist())) != len(face_ids)):
                raise ValueError('Invalid declared volume interface collision filter')
            for key, target in (('omniphysics:src0', mesh_path), ('omniphysics:src1', collider)):
                if [str(p) for p in filter_prim.GetRelationship(key).GetTargets()] != [target]:
                    raise ValueError('Volume filter endpoints differ from declaration')
            for key, value in (('groupElemCounts0', [len(face_ids)]), ('groupElemIndices0', face_ids),
                               ('groupElemCounts1', []), ('groupElemIndices1', [])):
                _same(filter_prim.GetAttribute('omniphysics:' + key).Get(), value, key, integer=True)
            filter_paths.add(filter_path)
        initial = _world_points(stage, mesh_path, points)
        _same(initial[indices], _world_points(stage, rigid_bodies[parent], anchors), 'bond world anchors')
        body_paths.add(path); mesh_paths.add(mesh_path); attachment_paths.add(bond_path)
        row.update(rest_world_points_m=initial.tolist(), node_count=len(points),
                   tetrahedron_count=len(tets), surface_triangle_count=len(surface),
                   attachment_response='native_two_way_rigid_endpoint',
                   physical_parameters_modified=False)
        result[name] = row
    if (actual_roots != body_paths or actual_simulation != mesh_paths or actual_attachments != attachment_paths
            or actual_filters != filter_paths):
        raise ValueError('Declared volume/attachment inventory differs from authored task subtree')
    return result


def numpy_copy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value, copy=True)


class VolumeBodyReadback:
    """Create exact native views; getters never set nodal targets or state."""
    def __init__(self, info, world, simulation_view):
        self.info, self.world, self.views = info, world, {}
        if info and (world is None or simulation_view is None):
            raise ValueError('Volume readback requires the actual native World and simulation view')
        if info and (simulation_view.device_ordinal < 0 or not str(simulation_view.device).startswith('cuda')):
            raise RuntimeError('Native task volume readback requires a CUDA tensor backend; CPU fallback is forbidden')
        for name, row in info.items():
            view = simulation_view.create_volume_deformable_body_view(row['prim_path'])
            if (view.count != 1 or list(view.prim_paths) != [row['prim_path']]
                    or list(view.simulation_mesh_prim_paths) != [row['simulation_mesh_path']]
                    or list(view.collision_mesh_prim_paths) != [row['simulation_mesh_path']]
                    or view.max_simulation_nodes_per_body != row['node_count']
                    or view.max_simulation_elements_per_body != row['tetrahedron_count']
                    or view.num_nodes_per_element != 4):
                raise RuntimeError('Native volume path/count/topology differs: ' + name)
            actual = numpy_copy(view.get_simulation_element_indices())
            if not np.array_equal(actual, np.asarray(row['tetrahedra'])[None, ...]):
                raise RuntimeError('Native volume tetrahedral indices differ: ' + name)
            rest = numpy_copy(view.get_rest_nodal_positions())
            rest_indices = numpy_copy(view.get_rest_element_indices())
            if (rest.shape != (1, row['node_count'], 3) or not np.isfinite(rest).all()
                    or not np.array_equal(rest_indices, actual)):
                raise RuntimeError('Native volume rest shape topology differs: ' + name)
            tets = np.asarray(row['tetrahedra'])
            native_tets, authored_tets = rest[0][tets], np.asarray(row['rest_points_m'])[tets]
            for i in range(4):
                for j in range(i):
                    if not np.allclose(np.linalg.norm(native_tets[:, i]-native_tets[:, j], axis=1),
                                       np.linalg.norm(authored_tets[:, i]-authored_tets[:, j], axis=1),
                                       atol=2.e-7, rtol=1.e-5):
                        raise RuntimeError('Native volume stress-free metric differs: ' + name)
            self.views[name] = view
        initial = self.read()
        self.initialization = {'mode': 'author-only',
            'tensor_device': str(simulation_view.device) if info else None,
            'tensor_device_ordinal': int(simulation_view.device_ordinal) if info else None, 'native_warmup_preserved': True,
            'position_write_calls': 0, 'velocity_write_calls': 0,
            'native_step': int(world.current_time_step_index) if info else None,
            'native_time_s': float(world.current_time) if info else None,
            'bodies': {name: {
                'maximum_node_displacement_from_authored_m': float(np.linalg.norm(
                    initial[name]['positions_m']-row['rest_world_points_m'], axis=1).max()),
                'maximum_node_speed_m_s': float(np.linalg.norm(initial[name]['velocities_m_s'], axis=1).max())}
                for name, row in info.items()}}

    def read(self):
        result = {}
        for name, view in self.views.items():
            values = [numpy_copy(view.get_simulation_nodal_positions()),
                      numpy_copy(view.get_simulation_nodal_velocities())]
            shape = (1, self.info[name]['node_count'], 3)
            if any(value.shape != shape or not np.isfinite(value).all() for value in values):
                raise RuntimeError('Native volume node readback is incomplete or nonfinite: ' + name)
            result[name] = {'positions_m': values[0][0], 'velocities_m_s': values[1][0]}
        return result

    def capture(self):
        return {key: value for name, row in self.read().items()
                for key, value in ((f'task_volume_{name}_q', row['positions_m']),
                                   (f'task_volume_{name}_qd', row['velocities_m_s']))}
