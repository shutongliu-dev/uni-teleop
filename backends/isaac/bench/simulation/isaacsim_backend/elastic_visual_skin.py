"""Render-only skinning from sampled native rigid cells; no physics handles.

The hidden authored mesh is a read-only template. A separate world-frame mesh
receives blended points at render synchronization. Mass, joints, collision,
native transforms, control targets and Fabric buffers are never written here.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import re

import numpy as np

SCHEMA = 'native-rigid-cell-visual-skin-v1'
ROOT = '/World/ElasticVisualSkins'
COMPACT_BINDING_SCHEMA = 'eight_nearest_inverse_squared_v1'


def clear_owned_skins_before_reset(stage):
    """Remove only our render-only output while the caller has stopped World."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(ROOT)
    if not root:
        return
    if root.GetCustomDataByKey('bench_visual_skin_schema') != SCHEMA:
        raise ValueError('Unowned elastic skin namespace cannot be cleared')
    for prim in Usd.PrimRange(root):
        if (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or any(str(name).startswith(('Physics', 'Physx', 'OmniPhysics'))
                       for name in prim.GetAppliedSchemas())):
            raise ValueError('Physical prim cannot be removed during visual reset')
    stage.RemovePrim(ROOT)


def _array(value, shape, label):
    data = np.asarray(value, dtype=float)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(label+' must have finite values and shape '+str(shape))
    return data.copy()


def _rotation(value):
    w, x, y, z = _array(value, (4,), 'Native skin member quaternion wxyz')
    length = np.linalg.norm([w, x, y, z])
    if abs(length-1.) > 1e-4:
        raise ValueError('Native skin member quaternion must be unit length')
    w, x, y, z = np.array([w, x, y, z])/length
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _safe_geometry(prim, *, output=False):
    from pxr import Usd, UsdGeom, UsdPhysics
    if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
        raise ValueError('Elastic skin must use a non-instance Mesh')
    for child in Usd.PrimRange(prim):
        schemas = child.GetMetadata('apiSchemas')
        names = () if schemas is None else schemas.GetAppliedItems()
        if (child.HasAPI(UsdPhysics.CollisionAPI) or child.HasAPI(UsdPhysics.RigidBodyAPI)
                or any(str(name).startswith(('Physics', 'Physx', 'OmniPhysics')) for name in names)):
            raise ValueError('Elastic skin requires noncollision geometry without physics APIs')
        if child != prim and not child.IsA(UsdGeom.Subset):
            raise ValueError('Elastic skin template children may only be material subsets')
    if output:
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            if parent.HasAPI(UsdPhysics.RigidBodyAPI) or parent.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Elastic skin output cannot inherit a physical body frame')
            parent = parent.GetParent()


def _copy_attributes(source, target, names):
    for attr in source.GetAttributes():
        name = attr.GetName()
        if name not in names and not name.startswith('primvars:'):
            continue
        value = attr.Get()
        if value is None:
            continue
        result = target.CreateAttribute(name, attr.GetTypeName(), custom=False)
        result.Set(value)
        for key in ('interpolation', 'elementSize', 'colorSpace', 'unauthoredValuesIndex'):
            if attr.HasAuthoredMetadata(key):
                result.SetMetadata(key, attr.GetMetadata(key))


def _copy_material(source, target):
    from pxr import UsdShade
    # Resolve inherited bindings on the original rigid hierarchy, then bind
    # the same materials explicitly under the independent output hierarchy.
    for purpose in ('', 'preview', 'full'):
        material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial(materialPurpose=purpose)
        if material:
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material, materialPurpose=purpose)


def _vertex_normals(points, counts, indices):
    """Area-weighted geometric vertex normals, including polygonal templates."""
    result = np.zeros_like(points)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    # Batch equal-size faces. The detailed sponge has tens of thousands of
    # triangles; Python iteration over every face would stall render sync.
    for count in np.unique(counts):
        starts = offsets[counts == count]
        faces = indices[starts[:, None]+np.arange(count)]
        polygon = points[faces]
        normal = np.cross(polygon[:, 1:-1]-polygon[:, :1],
                          polygon[:, 2:]-polygon[:, :1]).sum(1)
        for corner in range(count):
            np.add.at(result, faces[:, corner], normal)
    length = np.linalg.norm(result, axis=1, keepdims=True)
    return np.divide(result, length, out=np.zeros_like(result), where=length > 1e-15)


class ElasticVisualSkin:
    """Create skins from elastic_composites, then sync already sampled objects."""

    def __init__(self, stage, composites):
        self.stage = stage
        self.bindings = []
        self._step = None
        self._fingerprint = None
        self._receipt = None
        if not composites:
            return
        from pxr import Sdf, UsdGeom
        if not isinstance(composites, Mapping):
            raise ValueError('Elastic skin configuration must be a composite mapping')
        planned = []
        targets = set()
        for identifier, composite in composites.items():
            if not isinstance(identifier, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', identifier):
                raise ValueError('Elastic skin composite identifier must be a USD identifier')
            cells = {row['id']: row for row in composite['cells']}
            if len(cells) != len(composite['cells']):
                raise ValueError('Elastic skin member identifiers must be distinct')
            for index, binding in enumerate(composite.get('visual_skin_bindings', [])):
                source = stage.GetPrimAtPath(binding['prim_path'])
                _safe_geometry(source)
                if UsdGeom.Imageable(source).GetVisibilityAttr().Get() != 'invisible':
                    raise ValueError('Elastic skin rest template must already be hidden')
                path = binding.get('output_prim_path', ROOT+'/'+identifier+'/skin_'+str(index))
                if stage.GetPrimAtPath(path):
                    _safe_geometry(stage.GetPrimAtPath(path), output=True)
                    raise ValueError('Elastic skin output path already exists')
                if (not isinstance(path, str) or not Sdf.Path.IsValidPathString(path)
                        or not Sdf.Path(path).IsPrimPath() or not Sdf.Path(path).HasPrefix(Sdf.Path(ROOT))
                        or path in targets or path.count('/') != 4):
                    raise ValueError('Elastic skin output must be a unique mesh below its dedicated root')
                members = list(binding['member_ids'])
                if not members or len(set(members)) != len(members) or any(name not in cells for name in members):
                    raise ValueError('Elastic skin binding references an unknown or duplicate member')
                centers = np.array([_array(cells[name]['composite_rest_center_m'], (3,), 'Skin rest member center')
                                    for name in members])
                authored = np.asarray(UsdGeom.Mesh(source).GetPointsAttr().Get(), dtype=np.float64)
                if 'binding_schema' in binding:
                    if (binding['binding_schema'] != COMPACT_BINDING_SCHEMA
                            or type(binding.get('neighbor_count')) is not int or binding['neighbor_count'] != 8
                            or len(members) < 8
                            or type(binding.get('point_count')) is not int or binding['point_count'] <= 0
                            or type(binding.get('minimum_squared_distance_m2')) not in (int, float)
                            or binding['minimum_squared_distance_m2'] != 1e-12
                            or any(key in binding for key in ('rest_points_m', 'vertex_member_indices', 'vertex_weights'))):
                        raise ValueError('Elastic skin compact binding has an unsupported schema or parameters')
                    points = authored.copy()
                    if (points.shape != (binding['point_count'], 3) or not np.isfinite(points).all()
                            or hashlib.sha256(points.tobytes()).hexdigest() != binding.get('rest_points_sha256')):
                        raise ValueError('Elastic skin compact binding does not match immutable template points')
                    # Preserve the author's exact operations and member order,
                    # including its argsort tie ordering. This visual-only
                    # reconstruction never enters any physical computation.
                    distances2 = np.sum((points[:, None]-centers[None])**2, axis=2)
                    indices = np.argsort(distances2, axis=1)[:, :8]
                    selected = np.take_along_axis(distances2, indices, axis=1)
                    weights = 1/np.maximum(selected, 1e-12)
                    weights /= weights.sum(axis=1, keepdims=True)
                else:
                    points = np.asarray(binding['rest_points_m'], dtype=float)
                    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
                        raise ValueError('Elastic skin rest points must be a nonempty finite Nx3 array')
                    if authored.shape != points.shape or not np.allclose(authored, points, atol=1e-9, rtol=0):
                        raise ValueError('Elastic skin rest points differ from the immutable template')
                    indices = np.asarray(binding['vertex_member_indices'])
                    weights = np.asarray(binding['vertex_weights'], dtype=float)
                if (indices.ndim != 2 or indices.shape[0] != len(points) or indices.shape[1] == 0
                        or indices.dtype.kind not in 'iu' or weights.shape != indices.shape
                        or indices.min() < 0 or indices.max() >= len(members)
                        or not np.isfinite(weights).all() or (weights < 0).any()
                        or not np.allclose(weights.sum(1), 1., atol=1e-10, rtol=0)):
                    raise ValueError('Elastic skin needs finite nonnegative normalized member weights and valid indices')
                mesh = UsdGeom.Mesh(source)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
                faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
                if ((counts < 3).any() or counts.sum() != len(faces) or not len(faces)
                        or faces.min() < 0 or faces.max() >= len(points)):
                    raise ValueError('Elastic skin template has invalid topology')
                planned.append({'path': path, 'source': source, 'members': members, 'centers': centers,
                    'local': points[:, None]-centers[indices], 'indices': indices.copy(), 'weights': weights.copy(),
                    'counts': counts, 'faces': faces})
                targets.add(path)
        if not planned:
            return
        if stage.GetPrimAtPath(ROOT):
            raise ValueError('Elastic skin output namespace already exists')
        # All sources are validated before writing any render-only opinions.
        root = UsdGeom.Xform.Define(stage, ROOT)
        root.SetResetXformStack(True)
        root.GetPrim().SetCustomDataByKey('bench_visual_skin_schema', SCHEMA)
        for row in planned:
            UsdGeom.Scope.Define(stage, row['path'].rsplit('/', 1)[0])
            mesh = UsdGeom.Mesh.Define(stage, row['path'])
            properties = {'points', 'faceVertexCounts', 'faceVertexIndices', 'holeIndices', 'orientation',
                          'subdivisionScheme', 'doubleSided', 'interpolateBoundary',
                          'faceVaryingLinearInterpolation', 'triangleSubdivisionRule'}
            _copy_attributes(row['source'], mesh.GetPrim(), properties)
            _copy_material(row['source'], mesh.GetPrim())
            for child in row['source'].GetChildren():
                subset = UsdGeom.Subset.Define(stage, row['path']+'/'+child.GetName())
                _copy_attributes(child, subset.GetPrim(), {'indices', 'familyName', 'elementType'})
                _copy_material(child, subset.GetPrim())
            mesh.CreateVisibilityAttr('invisible')
            mesh.CreatePurposeAttr('render')
            row['mesh'] = mesh
            self.bindings.append(row)

    def sync(self, objects, native_step):
        """Only sampled position_m/quaternion_wxyz values enter the visual update."""
        if type(native_step) is not int or native_step < 0:
            raise ValueError('Elastic skin needs a nonnegative integer native tick')
        if not self.bindings:
            return None
        if self._step is not None and native_step < self._step:
            raise ValueError('Elastic skin native ticks cannot move backwards')
        from pxr import Gf, UsdGeom, Vt
        digest = hashlib.sha256()
        planned = []
        for row in self.bindings:
            _safe_geometry(row['mesh'].GetPrim(), output=True)
            if not np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(row['mesh'].GetPrim())),
                               np.eye(4), atol=1e-12, rtol=0):
                raise ValueError('Elastic skin output must retain the identity world frame')
            positions, rotations = [], []
            for member in row['members']:
                if not isinstance(objects, Mapping) or member not in objects:
                    raise ValueError('Elastic skin is missing a native member pose: '+member)
                pose = objects[member]
                if not isinstance(pose, Mapping) or not {'position_m', 'quaternion_wxyz'} <= set(pose):
                    raise ValueError('Elastic skin member must provide a native position and quaternion')
                position = _array(pose['position_m'], (3,), 'Native skin member position')
                rotation = _rotation(pose['quaternion_wxyz'])
                digest.update(position.tobytes())
                digest.update(rotation.tobytes())
                positions.append(position)
                rotations.append(rotation)
            rotations = np.array(rotations)[row['indices']]
            positions = np.array(positions)[row['indices']]
            points = ((np.einsum('nkij,nkj->nki', rotations, row['local'])+positions)*row['weights'][:, :, None]).sum(1)
            if not np.isfinite(points).all():
                raise ValueError('Elastic skin blend produced nonfinite geometry')
            planned.append((row, points))
        fingerprint = digest.hexdigest()
        if native_step == self._step:
            if fingerprint != self._fingerprint:
                raise ValueError('Elastic skin received different poses for the same native tick')
            result = deepcopy(self._receipt)
            result['changed'] = False
            return result
        for row, points in planned:
            mesh = row['mesh']
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateExtentAttr().Set([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
            normals = _vertex_normals(points, row['counts'], row['faces'])
            if mesh.GetOrientationAttr().Get() == 'leftHanded':
                normals *= -1
            mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.SetNormalsInterpolation('vertex')
            mesh.GetVisibilityAttr().Set('inherited')
        self._step, self._fingerprint = native_step, fingerprint
        self._receipt = {'schema': SCHEMA, 'source_native_step': native_step, 'changed': True,
            'output_paths': [row['path'] for row in self.bindings], 'physics_writes': 0,
            'coordinate_frame': 'world', 'normals': 'area_weighted_geometric_vertex',
            'scope': 'Render-only skin follows sampled rigid-cell transforms; visual agreement with collision is not certified.'}
        return deepcopy(self._receipt)
