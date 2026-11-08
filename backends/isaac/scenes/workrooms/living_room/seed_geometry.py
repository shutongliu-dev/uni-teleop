"""CPU geometry operations for new, immutable living-task asset derivatives.

These are authoring operations, never a runtime/reset object-control interface.
Mass models remain declared simulation priors, not measured material properties.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def rotation(quat):
    """Column-vector rotation from a USD quaternion."""
    return np.asarray(Gf.Matrix3d(Gf.Quatd(quat)), dtype=float).T


def quaternion(matrix):
    q = Gf.Matrix3d(*np.asarray(matrix).T.ravel().tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary()))


def affine_mass(mass, com, inertia, scale):
    """Density-preserving affine map of a centered inertia tensor."""
    a = np.diag(scale)
    determinant = float(np.prod(scale))
    if not (np.asarray(scale) > 0).all() or not np.isfinite(a).all():
        raise ValueError('Positive finite geometry scale required')
    covariance = np.trace(inertia) / 2 * np.eye(3) - inertia
    transformed = determinant * a @ covariance @ a.T
    return mass * determinant, a @ com, np.trace(transformed) * np.eye(3) - transformed


def set_mass(prim, mass, com, inertia):
    eigenvalues, axes = np.linalg.eigh(inertia)
    if not (mass > 0 and np.isfinite(com).all() and np.isfinite(eigenvalues).all()
            and (eigenvalues > 0).all()):
        raise ValueError('Invalid positive mass/inertia')
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*eigenvalues))
    api.CreatePrincipalAxesAttr(quaternion(axes))


def scale_mass(prim, scale):
    api = UsdPhysics.MassAPI(prim)
    com = np.array(api.GetCenterOfMassAttr().Get(), dtype=float)
    diagonal = np.array(api.GetDiagonalInertiaAttr().Get(), dtype=float)
    if not np.isfinite(com).all() or not (diagonal > 0).all():
        raise ValueError('Affine mass requires an explicit original mass model')
    r = rotation(api.GetPrincipalAxesAttr().Get())
    set_mass(prim, *affine_mass(float(api.GetMassAttr().Get()), com,
                              r @ np.diag(diagonal) @ r.T, scale))


def _vectors(attribute, scale, *, normals=False):
    points = np.array(attribute.Get(), dtype=float)
    if normals:
        points /= np.asarray(scale)
        points /= np.linalg.norm(points, axis=1)[:, None]
    else:
        points *= np.asarray(scale)
    attribute.Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    return points


def scale_tree(stage, root_path, scale, *, move_root=False):
    """Bake diagonal local scale into mesh points, transforms and mass.

    Only axis-aligned descendants are accepted; no nonuniform parent scale is
    left for a native mechanism importer to interpret.
    """
    root = stage.GetPrimAtPath(root_path)
    scale = np.array(scale, dtype=float)
    for prim in Usd.PrimRange(root):
        xf = UsdGeom.Xformable(prim)
        if xf and (prim != root or move_root):
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate or (
                        op.GetOpType() == UsdGeom.XformOp.TypeScale and prim.IsA(UsdGeom.Cube)):
                    value = op.Get()
                    op.Set(type(value)(*(np.array(value) * scale)))
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    # Existing diagonal transforms commute with the requested
                    # map; mesh points carry that map exactly once.
                    if not (np.asarray(op.Get()) > 0).all():
                        raise ValueError('Positive original descendant scale required')
                else:
                    raise ValueError('Non-axis-aligned child cannot be diagonally baked: '+str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = _vectors(mesh.GetPointsAttr(), scale)
            if mesh.GetNormalsAttr().HasAuthoredValue():
                _vectors(mesh.GetNormalsAttr(), scale, normals=True)
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        elif prim.IsA(UsdGeom.BasisCurves):
            _vectors(UsdGeom.BasisCurves(prim).GetPointsAttr(), scale)
        elif prim.IsA(UsdGeom.Cube):
            pass  # Its original size-one cube already has an explicit scale op.
        elif prim.IsA(UsdGeom.Gprim):
            raise ValueError('Unsupported geometry bake: '+str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            scale_mass(prim, scale)


def scale_joint_anchors(stage, body_paths, scale):
    paths = set(body_paths)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for rel, attr in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                          (joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
            if any(str(path) in paths for path in rel.GetTargets()):
                attr.Set(Gf.Vec3f(*(np.array(attr.Get()) * scale)))


def mesh_volume_moments(mesh):
    """Exact signed tetrahedral moments of a closed triangulated source mesh."""
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    triangles, offset = [], 0
    for count in counts:
        face = list(indices[offset:offset+count]); offset += count
        triangles.extend((face[0], face[j], face[j+1]) for j in range(1, count-1))
    v = points[np.asarray(triangles)]
    volumes = np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2]))/6
    total = v.sum(axis=1)
    first = (volumes[:, None] * total / 4).sum(axis=0)
    second = np.einsum('i,ij,ik->jk', volumes/20, total, total)
    second += np.einsum('i,iaj,iak->jk', volumes/20, v, v)
    volume = volumes.sum()
    sign = 1 if volume > 0 else -1
    if abs(volume) < 1.e-12:
        raise ValueError('Source shell has no solid volume')
    volume, first, second = volume*sign, first*sign, second*sign
    com = first / volume
    covariance = second-volume*np.outer(com, com)
    return float(volume), com, np.trace(covariance)*np.eye(3)-covariance


def capsule_volume_moments(radius, height):
    """Unit-density solid capsule; height is the cylindrical segment length."""
    vc, vs = math.pi*radius**2*height, 4*math.pi*radius**3/3
    iz = .5*vc*radius**2 + .4*vs*radius**2
    ix = vc*(3*radius**2+height**2)/12 + vs*(.4*radius**2+height**2/4+3*height*radius/8)
    return vc+vs, np.zeros(3), np.diag([ix, ix, iz])


def compound_moments(components):
    mass = sum(m for m, _, _ in components)
    com = sum(m*c for m, c, _ in components)/mass
    inertia = np.zeros((3, 3))
    for m, c, i in components:
        d = c-com
        inertia += i + m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    return mass, com, inertia


def pose(prim):
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    q = matrix.ExtractRotationQuat()
    quat = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)
    quat /= np.linalg.norm(quat)
    return {'prim_path': str(prim.GetPath()), 'position_m': list(matrix.ExtractTranslation()),
            'quaternion_wxyz': quat.tolist()}


def transform_group(stage, plan):
    """Bake a planar rigid reset, including the original water particle array."""
    angle = plan['yaw_rad']; c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    pivot = np.array(plan['pivot_m'])
    delta = np.array([*plan['translation_xy_m'], 0.])
    transform = Gf.Matrix4d(1.)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(angle)))
    transform.SetTranslateOnly(Gf.Vec3d(*(pivot+delta-r@pivot)))
    paths = sorted(plan['prim_paths'], key=len)
    for path in paths:
        if any(path.startswith(parent+'/') for parent in paths if parent != path):
            raise ValueError('Reset group contains overlapping ancestor/child paths')
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing reset prim '+path)
        if prim.IsA(UsdGeom.Points):
            points = UsdGeom.Points(prim)
            raw = np.array(points.GetPointsAttr().Get(), dtype=float)
            raw = (raw-pivot)@r.T+pivot+delta
            points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(raw.astype(np.float32)))
            if points.GetVelocitiesAttr().HasAuthoredValue():
                velocities = np.array(points.GetVelocitiesAttr().Get())@r.T
                points.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(velocities.astype(np.float32)))
            continue
        # All supplied groups have task-root parents with identity transforms.
        # Handle the original typeless CollectionTray as a new identity Xform.
        if not prim.IsA(UsdGeom.Xformable):
            prim.SetTypeName('Xform')
        xf = UsdGeom.Xformable(prim)
        parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
        local = xf.GetLocalTransformation()
        result = local * parent * transform * parent.GetInverse()
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='seedReset').Set(result)


def mass_inventory(stage, actors):
    result = {}
    for name, path in actors.items():
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        row = {'mass_kg': float(api.GetMassAttr().Get())}
        com, inertia = api.GetCenterOfMassAttr(), api.GetDiagonalInertiaAttr()
        if com.HasAuthoredValue() and inertia.HasAuthoredValue():
            row.update(center_of_mass_m=list(com.Get()), diagonal_inertia_kg_m2=list(inertia.Get()),
                       principal_axes_wxyz=[api.GetPrincipalAxesAttr().Get().GetReal(),
                                            *api.GetPrincipalAxesAttr().Get().GetImaginary()],
                       inertia_mode='explicit_prior')
        else:
            row['inertia_mode'] = 'native_geometry_computation_not_read_back'
        result[name] = row
    return result


def collision_bounds(stage, body_path):
    """Body-local envelope of actual collision primitives, excluding pictures."""
    body = stage.GetPrimAtPath(body_path)
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    bounds = []
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        matrix = cache.GetLocalToWorldTransform(prim) * inverse
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            points = [Gf.Vec3d(x*half, y*half, z*half)
                      for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        elif prim.IsA(UsdGeom.Capsule):
            cap = UsdGeom.Capsule(prim)
            radius, half = float(cap.GetRadiusAttr().Get()), float(cap.GetHeightAttr().Get())/2
            axis = {'X': 0, 'Y': 1, 'Z': 2}[str(cap.GetAxisAttr().Get())]
            center = np.array(matrix.ExtractTranslation())
            linear = np.array(matrix)[:3, :3].T
            extent = radius*np.linalg.norm(linear, axis=1) + half*np.abs(linear[:, axis])
            bounds.extend([center-extent, center+extent])
            continue
        else:
            raise ValueError('Unsupported collision bounds: '+str(prim.GetPath()))
        bounds.extend(np.array(matrix.Transform(Gf.Vec3d(*p))) for p in points)
    if not bounds:
        raise ValueError('Body has no collision geometry: '+body_path)
    values = np.asarray(bounds)
    return [values.min(axis=0).tolist(), values.max(axis=0).tolist()]
