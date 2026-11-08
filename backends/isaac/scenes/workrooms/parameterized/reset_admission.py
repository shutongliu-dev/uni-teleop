"""Robot-independent CPU reset admission for parameterized layout candidates.

The layout sampler moves whole groups (translation plus a yaw about the group
pivot). Nothing upstream proved that the moved reset is physically valid, and
the native passive hold found three classes of invalid resets: bodies whose
centre of mass is outside their support, actors outside the evaluator's
recoverable workspace, and colliders that interpenetrate fixed apparatus at
t=0. ``check_reset_admission`` rejects those candidates before any GPU work:

R1  every actor origin lies inside the evaluator ``recovery_workspace_bounds_m``
    (fallback ``workspace_bounds_m``) of the task metadata.
S   every dynamic body without a joint has its centre of mass inside the convex
    hull of its contact patches.  A contact patch is a collider bottom face
    within +-3 mm of a supporter top face, clipped in XY.  Supporters are room
    and task colliders (including kinematic fixtures moved with the layout) and
    dynamic bodies whose centre of mass is lower.  Room floor, robot and stand
    colliders never support.  A body with nothing but the floor (or more than
    25 mm of air) below it is rejected; shorter drops are allowed.
R3  every collider pair whose relative pose the layout changed (one side moved
    by a group, the other not co-moved, at least one dynamic) is tested with
    its actual representation (Cube/convexHull halfspaces; SDF, triangle and
    decomposition meshes as closed triangle meshes; analytic convex primitives
    as dense convex samples).  The deepest penetration must not exceed half of
    the task referee's own penetration bound.

The stage is the composed task stage without robot or stand, so the verdict is
identical for every installation and both robot profiles.  The result is
deterministic: fixed traversal order, fixed sampling seed and no runtime
state.  Seed 0 is never checked (canonical worlds cannot be resampled) and
families without a rigid group layout pass through unchanged.

Known issue, intentionally not changed here: ``spec.parameter_specs`` declares
``layout.radius_m`` in [0.04, 0.15] m, but ``usd_authoring.transform_point`` and
``mechanical_authoring._layout`` apply the sampled yaw about the group pivot, so
a body moves by |t + (Rz(yaw) - I)(p - pivot)|.  The six-object gear supply
group moves bodies up to 0.32 m.  Changing the samplers would move every
already admitted world, so this module rejects the invalid resets instead.
"""
from __future__ import annotations

from itertools import product
import math

import numpy as np

SCHEMA = 'parameterized_reset_admission.v1'
RIGID_FAMILIES = ('kitchen', 'laboratory', 'mechanical_assembly', 'study')
SUPPORT_GAP_M = .003
FACE_TOLERANCE_M = .002
MAXIMUM_DROP_M = .025
ENGAGED_GAP_M = .0005
PENETRATION_FRACTION = .5
DEFAULT_REFEREE_PENETRATION_M = .001
CANDIDATE_FILTER_M = .0005
SURFACE_SAMPLES = 6000
EXACT_MESH_APPROXIMATIONS = ('sdf', 'none')
FLOOR_PREFIXES = ('/World/Room/Geometry/room_floor',)
EXCLUDED_TOKENS = ('robot_stand', 'bimanual_robot_stand', '/Robots/', '/World/Robot/', '/World/Room/Robot')
SCOPE = ('Robot-free composed CPU task stage. Static centre-of-mass support, evaluator reset bounds and '
         'exact reset interpenetration of layout-changed collider pairs. Not native settling, contact '
         'dynamics, deformable/granular stability or manipulation qualification.')


class ResetAdmissionError(ValueError):
    """Rejected reset; ``record`` carries the deterministic measurements."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record
        self.measurements = record


# ---------------------------------------------------------------- 2-D geometry
def _hull2d(points):
    from scipy.spatial import ConvexHull, QhullError
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    return pts[hull.vertices]


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return .5*float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject, clipper):
    """Sutherland-Hodgman intersection of two convex counter-clockwise polygons."""
    out = [tuple(p) for p in subject]
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i+1) % len(clipper)]
        points, out = out, []
        if not points:
            break

        def inside(p):
            return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-12

        def cross(p, q):
            den = (p[0]-q[0])*(a[1]-b[1]) - (p[1]-q[1])*(a[0]-b[0])
            if abs(den) < 1e-18:
                return q
            t = ((p[0]-a[0])*(a[1]-b[1]) - (p[1]-a[1])*(a[0]-b[0]))/den
            return (p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1]))
        previous = points[-1]
        for current in points:
            if inside(current):
                if not inside(previous):
                    out.append(cross(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(cross(previous, current))
            previous = current
    return np.asarray(out) if len(out) >= 3 else None


def _margin(poly, point):
    """Signed distance from ``point`` to the boundary of a convex CCW polygon (inside > 0)."""
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        e = b - a
        length = math.hypot(*e)
        if length > 1e-15:
            best = min(best, (e[0]*(point[1]-a[1]) - e[1]*(point[0]-a[0]))/length)
    return float(best)


def _face(corners, height, *, top):
    points = corners[corners[:, 2] >= height-FACE_TOLERANCE_M] if top else corners[corners[:, 2] <= height+FACE_TOLERANCE_M]
    poly = _hull2d(points[:, :2]) if len(points) >= 3 else None
    if poly is None or abs(_area(poly)) < 1e-10:
        poly = _hull2d(corners[:, :2])
    return poly


def separating_axis_gap(first, second):
    """Signed separation of two transformed eight-corner boxes (product corner order)."""
    a, b = np.asarray(first), np.asarray(second)
    ea, eb = a[[4, 2, 1]]-a[0], b[[4, 2, 1]]-b[0]
    axes = [np.cross(edges[i], edges[j]) for edges in (ea, eb) for i, j in ((0, 1), (0, 2), (1, 2))]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    best = -math.inf
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-14:
            continue
        axis = axis/length
        pa, pb = a@axis, b@axis
        best = max(best, float(pb.min()-pa.max()), float(pa.min()-pb.max()))
    return best


# ------------------------------------------------------------- stage readback
def _excluded(path):
    return any(token in path for token in EXCLUDED_TOKENS)


def collider_inventory(stage):
    """Enabled colliders as world eight-corner boxes, in deterministic traversal order."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    transforms = UsdGeom.XformCache()
    rows = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if _excluded(path) or 'robot_stand' in path.lower():
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Boundable):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        bound = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if bound.IsEmpty():
            continue
        corners = np.array([(*p, 1.) for p in product(*zip(bound.GetMin(), bound.GetMax()))])
        world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        corners = (world@corners.T).T[:, :3]
        if np.isfinite(corners).all():
            rows.append((path, corners))
    return rows


def _owner(path, roots):
    for name, root in roots:
        if path == root or path.startswith(root+'/'):
            return name
    return None


def _coms(stage, bodies, corners_by_body):
    from pxr import UsdGeom, UsdPhysics
    transforms = UsdGeom.XformCache(); result = {}
    for name, path in bodies.items():
        prim = stage.GetPrimAtPath(path)
        com = None
        if prim and prim.HasAPI(UsdPhysics.MassAPI):
            local = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
            if local is not None and np.isfinite(list(local)).all():
                matrix = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
                com = (matrix@np.array([*local, 1.]))[:3]
        if com is None and name in corners_by_body:
            points = np.vstack(corners_by_body[name])
            com = .5*(points.min(0)+points.max(0))
        if com is not None:
            result[name] = np.asarray(com, dtype=float)
    return result


def _jointed(stage, bodies):
    from pxr import UsdPhysics
    roots = sorted(bodies.items(), key=lambda item: len(item[1]), reverse=True)
    jointed = set()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for target in [*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()]:
                name = _owner(str(target), roots)
                if name is not None:
                    jointed.add(name)
    return jointed


def _shape_geometry(stage, path):
    from pxr import UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T

    def world(points):
        points = np.asarray(points, dtype=float)
        return (matrix@np.c_[points, np.ones(len(points))].T).T[:, :3]
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
        return {'kind': 'convex', 'points': world(np.array(list(product((-half, half), repeat=3))))}
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = world(mesh.GetPointsAttr().Get())
        approximation = (UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                         if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else 'none')
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        faces, start = [], 0
        for count in counts:
            face = indices[start:start+count]; start += count
            faces.extend([face[0], face[j], face[j+1]] for j in range(1, count-1))
        kind = 'convex' if approximation in ('convexHull', 'boundingCube') else 'mesh'
        return {'kind': kind, 'points': points, 'faces': np.asarray(faces, dtype=int).reshape(-1, 3),
                'approximation': approximation}
    unit = None
    if prim.IsA(UsdGeom.Sphere):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        unit = _sphere_points(r, 0., 'Z')
    elif prim.IsA(UsdGeom.Capsule):
        api = UsdGeom.Capsule(prim)
        unit = _sphere_points(float(api.GetRadiusAttr().Get()), float(api.GetHeightAttr().Get())/2, api.GetAxisAttr().Get())
    elif prim.IsA(UsdGeom.Cylinder):
        api = UsdGeom.Cylinder(prim)
        unit = _cylinder_points(float(api.GetRadiusAttr().Get()), float(api.GetHeightAttr().Get())/2, api.GetAxisAttr().Get())
    if unit is None:
        return None
    return {'kind': 'convex', 'points': world(unit)}


def _axis_frame(points, axis):
    order = {'X': [2, 1, 0], 'Y': [0, 2, 1], 'Z': [0, 1, 2]}[str(axis or 'Z')]
    return points[:, order]


def _sphere_points(radius, half_height, axis):
    u = np.linspace(0, 2*math.pi, 48, endpoint=False); v = np.linspace(-math.pi/2, math.pi/2, 25)
    uu, vv = np.meshgrid(u, v)
    points = np.c_[(radius*np.cos(vv)*np.cos(uu)).ravel(), (radius*np.cos(vv)*np.sin(uu)).ravel(), (radius*np.sin(vv)).ravel()]
    points[:, 2] += np.sign(points[:, 2])*half_height
    return _axis_frame(points, axis)


def _cylinder_points(radius, half_height, axis):
    u = np.linspace(0, 2*math.pi, 96, endpoint=False)
    ring = np.c_[radius*np.cos(u), radius*np.sin(u)]
    points = np.vstack([np.c_[ring, np.full(len(u), z)] for z in (-half_height, half_height)])
    return _axis_frame(points, axis)


# ------------------------------------------------------------ 3-D depth tests
def _triangles(shape):
    from scipy.spatial import ConvexHull
    if shape['kind'] == 'convex':
        return shape['points'][ConvexHull(shape['points']).simplices]
    return shape['points'][shape['faces']]


def _surface_samples(shape, count=SURFACE_SAMPLES):
    triangles = _triangles(shape)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if area.sum() <= 0:
        return shape['points']
    rng = np.random.default_rng(0)
    index = rng.choice(len(triangles), size=count, p=area/area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u+v > 1; u[flip], v[flip] = 1-u[flip], 1-v[flip]
    t = triangles[index]
    points = t[:, 0] + u[:, None]*(t[:, 1]-t[:, 0]) + v[:, None]*(t[:, 2]-t[:, 0])
    return np.vstack([shape['points'], points, triangles.mean(1)])


def _closed(shape):
    if 'closed' not in shape:
        faces = shape['faces']
        if not len(faces):
            shape['closed'] = False
        else:
            welded = np.round(shape['points']/1e-7).astype(np.int64)
            _, inverse = np.unique(welded, axis=0, return_inverse=True)
            faces = inverse.reshape(-1)[faces]
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            edges = edges[edges[:, 0] != edges[:, 1]]
            _, count = np.unique(edges, axis=0, return_counts=True)
            shape['closed'] = bool(len(count) and np.all(count == 2))
    return shape['closed']


def _winding(points, triangles, chunk=128):
    result = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        q = points[start:start+chunk, None, :]
        a, b, c = triangles[None, :, 0]-q, triangles[None, :, 1]-q, triangles[None, :, 2]-q
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        numerator = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        denominator = (la*lb*lc + np.einsum('ijk,ijk->ij', a, b)*lc + np.einsum('ijk,ijk->ij', b, c)*la
                       + np.einsum('ijk,ijk->ij', c, a)*lb)
        result[start:start+chunk] = (2*np.arctan2(numerator, denominator)).sum(1)/(4*math.pi)
    return result


def _point_triangle_distance(points, triangles, chunk=128):
    best = np.full(len(points), np.inf)
    tiny = 1e-30
    for start in range(0, len(points), chunk):
        p = points[start:start+chunk, None, :]
        a, b, c = triangles[None, :, 0], triangles[None, :, 1], triangles[None, :, 2]
        ab, ac, ap = b-a, c-a, p-a
        d1, d2 = (ab*ap).sum(-1), (ac*ap).sum(-1)
        bp = p-b; d3, d4 = (ab*bp).sum(-1), (ac*bp).sum(-1)
        cp = p-c; d5, d6 = (ab*cp).sum(-1), (ac*cp).sum(-1)
        va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
        denominator = va+vb+vc; denominator = np.where(np.abs(denominator) < tiny, tiny, denominator)
        x = a + ab*(vb/denominator)[..., None] + ac*(vc/denominator)[..., None]
        x = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, x)
        x = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, x)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1/np.where(np.abs(d1-d3) < tiny, tiny, d1-d3)
        x = np.where(m[..., None], a+ab*t[..., None], x)
        x = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, x)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2/np.where(np.abs(d2-d6) < tiny, tiny, d2-d6)
        x = np.where(m[..., None], a+ac*t[..., None], x)
        m = (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0); den = (d4-d3)+(d5-d6)
        t = (d4-d3)/np.where(np.abs(den) < tiny, tiny, den)
        x = np.where(m[..., None], b+(c-b)*t[..., None], x)
        best[start:start+chunk] = np.linalg.norm(p-x, axis=-1).min(1)
    return best


def _depth_inside(shape, points):
    """Deepest point of ``points`` inside the solid of ``shape`` (0 when outside or open)."""
    from scipy.spatial import ConvexHull
    if not len(points):
        return 0.
    if shape['kind'] == 'convex':
        planes = ConvexHull(shape['points']).equations
        signed = points@planes[:, :3].T + planes[:, 3]
        inside = np.all(signed <= 0, axis=1)
        return float((-signed[inside].max(axis=1)).max()) if inside.any() else 0.
    if not _closed(shape):
        return 0.
    lo, hi = shape['points'].min(0), shape['points'].max(0)
    near = points[np.all((points >= lo-1e-9) & (points <= hi+1e-9), axis=1)]
    if not len(near):
        return 0.
    triangles = _triangles(shape)
    inside = near[np.abs(_winding(near, triangles)) > .5]
    return float(_point_triangle_distance(inside, triangles).max()) if len(inside) else 0.


def exact_source_solid(stage, path):
    """One collider as an exactly represented source solid, else None.

    Cubes and source convexHull meshes are convex polytopes. SDF and plain
    triangle-mesh colliders are their closed source triangle meshes, the solid
    whose depth R3 measures. Analytic curved shapes, bounding, decomposed or
    simplified approximations and open meshes have no exact closed source
    solid here and return None.
    """
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim or not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Mesh)):
        return None
    shape = _shape_geometry(stage, path)
    if prim.IsA(UsdGeom.Cube) or shape['approximation'] == 'convexHull':
        return shape
    if shape['approximation'] in EXACT_MESH_APPROXIMATIONS and _closed(shape):
        return shape
    return None


def _segment_crosses(start, end, triangles, eps=1e-12):
    """Whether the closed segment meets any triangle transversally (Moeller-Trumbore)."""
    d = end-start
    a, e1, e2 = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    valid = np.abs(det) > 1e-18
    inverse = np.where(valid, 1./np.where(valid, det, 1.), 0.)
    s = start-a
    u = np.einsum('ij,ij->i', s, p)*inverse
    q = np.cross(s, e1)
    v = (q@d)*inverse
    t = np.einsum('ij,ij->i', e2, q)*inverse
    return bool(np.any(valid & (u >= -eps) & (v >= -eps) & (u+v <= 1+eps) & (t >= -eps) & (t <= 1+eps)))


def _segment_edge_distance(start, end, triangles):
    """Distance of one segment to every triangle edge (closest points of two segments)."""
    p2 = triangles.reshape(-1, 3)
    q2 = triangles[:, [1, 2, 0]].reshape(-1, 3)
    d1, d2, r = end-start, q2-p2, start-p2
    a = float(d1@d1)
    e = np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    c = r@d1
    b = d2@d1
    tiny = 1e-30
    edge = e > tiny
    denominator = a*e-b*b
    s = np.where(edge & (denominator > tiny), np.clip((b*f-c*e)/np.where(denominator > tiny, denominator, 1.), 0., 1.), 0.)
    t = np.where(edge, (b*s+f)/np.where(edge, e, 1.), 0.)
    s = np.where(~edge | (t < 0.), np.clip(-c/a, 0., 1.), np.where(t > 1., np.clip((b-c)/a, 0., 1.), s))
    t = np.clip(t, 0., 1.)
    return np.linalg.norm(start+s[:, None]*d1-(p2+t[:, None]*d2), axis=1)


def segment_distance(shape, start, end):
    """Exact distance from a closed segment to one exact source solid; 0 when touching or inside.

    ``shape`` is an :func:`exact_source_solid` (convex polytope, or closed SDF or
    triangle mesh). A segment that neither crosses the boundary nor starts inside
    is nearest to it at an endpoint/triangle or segment/edge pair.
    """
    from scipy.spatial import ConvexHull
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all() or np.allclose(start, end):
        raise ValueError('Finite nondegenerate approach segment required')
    if shape['kind'] == 'convex':
        hull = ConvexHull(shape['points'])
        normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
        slope, value = normals@(end-start), normals@start+offsets
        flat = np.abs(slope) < 1e-15
        if not np.any(flat & (value > 0.)):
            ratio = -value[~flat]/slope[~flat]
            low = max([0., *ratio[slope[~flat] < 0.]])
            high = min([1., *ratio[slope[~flat] > 0.]])
            if low <= high:
                return 0.
        triangles = shape['points'][hull.simplices]
    else:
        if not _closed(shape):
            raise ValueError('Exact segment distance requires a closed source mesh')
        triangles = _triangles(shape)
        if _segment_crosses(start, end, triangles) or abs(_winding(start[None], triangles)[0]) > .5:
            return 0.
    return float(min(_point_triangle_distance(np.stack([start, end]), triangles).min(),
                     _segment_edge_distance(start, end, triangles).min()))


def point_distance(shape, point):
    """Exact distance from a point to one exact source solid; 0 when on or inside it."""
    from scipy.spatial import ConvexHull
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Finite point required')
    if shape['kind'] == 'convex':
        hull = ConvexHull(shape['points'])
        if np.all(hull.equations[:, :3]@point+hull.equations[:, 3] <= 0.):
            return 0.
        triangles = shape['points'][hull.simplices]
    else:
        if not _closed(shape):
            raise ValueError('Exact point distance requires a closed source mesh')
        triangles = _triangles(shape)
        if abs(_winding(point[None], triangles)[0]) > .5:
            return 0.
    return float(_point_triangle_distance(point[None], triangles).min())


def segment_segment_distance(p1, q1, p2, q2):
    """Exact distance between two closed segments; either may degenerate to a point."""
    p1, q1, p2, q2 = (np.asarray(v, dtype=float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1-p1, q2-p2, p1-p2
    a, e, f = float(d1@d1), float(d2@d2), float(d2@r)
    tiny = 1e-30
    if a <= tiny and e <= tiny:
        s = t = 0.
    elif a <= tiny:
        s, t = 0., min(max(f/e, 0.), 1.)
    else:
        c = float(d1@r)
        if e <= tiny:
            t, s = 0., min(max(-c/a, 0.), 1.)
        else:
            b = float(d1@d2); denominator = a*e-b*b
            s = min(max((b*f-c*e)/denominator, 0.), 1.) if denominator > tiny else 0.
            t = (b*s+f)/e
            if t < 0.:
                t, s = 0., min(max(-c/a, 0.), 1.)
            elif t > 1.:
                t, s = 1., min(max((b-c)/a, 0.), 1.)
    return float(np.linalg.norm(p1+s*d1-(p2+t*d2)))


def core_penetration(shape, start, end):
    """Exact penetration depth of a point/segment core touching or inside a convex polytope.

    The minimum separating translation: over the polytope face normals and, for
    a segment, the cross products of its direction with every polytope edge.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(shape['points'])
    points = shape['points']
    core = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)])
    axes = [normal for normal in hull.equations[:, :3]]
    direction = core[1]-core[0]
    if np.linalg.norm(direction) > 1e-12:
        for simplex in hull.simplices:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                axis = np.cross(direction, points[simplex[j]]-points[simplex[i]])
                length = np.linalg.norm(axis)
                if length > 1e-12:
                    axes.append(axis/length)
    best = np.inf
    for axis in axes:
        a, b = core@axis, points@axis
        best = min(best, float(min(a.max()-b.min(), b.max()-a.min())))
    return max(best, 0.)


def pair_depth(first, second):
    """Penetration depth of two actual collider shapes; None when unrepresentable."""
    if first is None or second is None:
        return None
    lo = np.maximum(first['points'].min(0), second['points'].min(0)) - 1e-3
    hi = np.minimum(first['points'].max(0), second['points'].max(0)) + 1e-3

    def crop(points):
        return points[np.all((points >= lo) & (points <= hi), axis=1)]
    return max(_depth_inside(second, crop(_surface_samples(first))),
               _depth_inside(first, crop(_surface_samples(second))))


# -------------------------------------------------------------------- rules
def referee_penetration_bound(task):
    values = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('maximum_reported_penetration_m', 'maximum_penetration_m') and isinstance(item, (int, float)) \
                        and not isinstance(item, bool) and item > 0:
                    values.append(float(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(task.get('evaluation', {})); walk(task.get('public_rules', {}))
    return min(values) if values else DEFAULT_REFEREE_PENETRATION_M


def _support(paths, owners, corners, coms, jointed):
    lo, hi = corners.min(1), corners.max(1)
    floor = np.array([p.startswith(FLOOR_PREFIXES) for p in paths])
    dynamic = np.array([o is not None for o in owners])
    indices = {}
    for i, owner in enumerate(owners):
        if owner is not None:
            indices.setdefault(owner, []).append(i)
    tops = {}
    results = {}
    every = np.arange(len(paths))
    for body in sorted(indices):
        own = np.asarray(indices[body])
        if body in jointed:
            results[body] = {'status': 'jointed'}; continue
        com = coms.get(body)
        if com is None:
            com = .5*(lo[own].min(0)+hi[own].max(0))
        lower = np.array([o is not None and o != body and o in coms and coms[o][2] < com[2] for o in owners])
        allowed = ~floor & ~np.isin(every, own) & (~dynamic | lower)
        points, used = [], set()
        for i in own:
            bottom = lo[i, 2]
            mask = allowed & (hi[:, 2] >= bottom-SUPPORT_GAP_M) & (hi[:, 2] <= bottom+SUPPORT_GAP_M)
            mask &= np.all(np.minimum(hi[i, :2], hi[:, :2]) > np.maximum(lo[i, :2], lo[:, :2]), axis=1)
            if not mask.any():
                continue
            footprint = _face(corners[i], bottom, top=False)
            if footprint is None:
                continue
            for j in np.where(mask)[0]:
                if j not in tops:
                    tops[j] = _face(corners[j], hi[j, 2], top=True)
                if tops[j] is None:
                    continue
                patch = _clip(footprint, tops[j])
                if patch is not None:
                    points.extend(patch.tolist()); used.add(owners[j] or paths[j])
        hull = _hull2d(points) if len(points) >= 3 else None
        if hull is not None:
            results[body] = {'status': 'supported', 'com_margin_m': _margin(hull, com[:2]),
                             'supporters': sorted(used)[:8]}
            continue
        blo, bhi = lo[own].min(0), hi[own].max(0)
        others = np.where(dynamic & ~np.isin(every, own) &
                          np.all(np.minimum(bhi, hi) >= np.maximum(blo, lo)-ENGAGED_GAP_M, axis=1))[0]
        engaged = any(separating_axis_gap(corners[i], corners[j]) < ENGAGED_GAP_M for i in own for j in others
                      if np.all(np.minimum(hi[i], hi[j]) >= np.maximum(lo[i], lo[j])-ENGAGED_GAP_M))
        if engaged:
            results[body] = {'status': 'engaged'}; continue
        below = ~floor & ~np.isin(every, own) & (hi[:, 2] <= blo[2]+SUPPORT_GAP_M)
        below &= np.all(np.minimum(bhi[:2], hi[:, :2]) > np.maximum(blo[:2], lo[:, :2]), axis=1)
        gap = float((blo[2]-hi[below, 2]).min()) if below.any() else None
        if gap is None or gap > MAXIMUM_DROP_M:
            results[body] = {'status': 'unsupported', 'air_gap_m': gap}
        elif gap > SUPPORT_GAP_M:
            results[body] = {'status': 'drop', 'air_gap_m': gap}
        else:
            results[body] = {'status': 'unknown'}
    return results


def _moved_groups(layout):
    groups = []
    for row in layout or []:
        t = row.get('translation_xy_m') or [0., 0.]
        if any(abs(float(v)) > 0 for v in t) or float(row.get('yaw_rad') or 0.) != 0.:
            groups.append((str(row.get('id')), list(row.get('paths') or row.get('prim_paths') or [])))
    return groups


def _interpenetration(stage, paths, owners, corners, layout, threshold):
    groups = _moved_groups(layout)
    members = sorted(((index+1, root) for index, (_, roots) in enumerate(groups) for root in roots),
                     key=lambda item: len(item[1]), reverse=True)

    def group_of(path):
        for index, root in members:
            if path == root or path.startswith(root+'/'):
                return index
        return 0
    group = np.array([group_of(path) for path in paths], dtype=int)
    body_index = {}
    body = np.array([body_index.setdefault(o, len(body_index)+1) if o is not None else 0 for o in owners])
    moved, dynamic = group > 0, body > 0
    lo, hi = corners.min(1), corners.max(1)
    order = np.arange(len(paths)); limit = min(CANDIDATE_FILTER_M, threshold)
    candidates = []
    for i in np.where(moved)[0]:
        mask = np.all(np.minimum(hi[i], hi) - np.maximum(lo[i], lo) > limit, axis=1)
        mask &= group != group[i]
        mask &= dynamic | dynamic[i]
        mask &= ~((body == body[i]) & dynamic)
        mask &= ~(moved & (order < i))
        for j in np.where(mask)[0]:
            gap = separating_axis_gap(corners[i], corners[j])
            if gap < -limit:
                candidates.append((-gap, int(i), int(j)))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    geometry = {}

    def shape(index):
        if index not in geometry:
            geometry[index] = _shape_geometry(stage, paths[index])
        return geometry[index]
    worst, worst_pair, violations, unresolved = 0., None, [], []
    for _, i, j in candidates:
        depth = pair_depth(shape(i), shape(j))
        if depth is None:
            unresolved.append([paths[i], paths[j]]); continue
        if depth > worst:
            worst, worst_pair = depth, [paths[i], paths[j]]
        if depth > threshold:
            violations.append({'first': paths[i], 'second': paths[j], 'depth_m': depth})
    return {'candidates': len(candidates), 'maximum_depth_m': worst, 'worst_pair': worst_pair,
            'violations': violations[:12], 'violation_count': len(violations), 'unresolved': unresolved[:12],
            'threshold_m': threshold}


def evaluate(stage, *, bodies, origins, layout, task):
    """Pure admission on a composed stage. ``bodies``/``origins`` map actor names to prim paths/positions."""
    inventory = collider_inventory(stage)
    roots = sorted(bodies.items(), key=lambda item: len(item[1]), reverse=True)
    paths = [path for path, _ in inventory]
    owners = [_owner(path, roots) for path in paths]
    corners = np.asarray([c for _, c in inventory]) if inventory else np.zeros((0, 8, 3))
    by_body = {}
    for owner, c in zip(owners, corners):
        if owner is not None:
            by_body.setdefault(owner, []).append(c)
    coms = _coms(stage, bodies, by_body)
    jointed = _jointed(stage, bodies)
    reasons = []
    metadata = task.get('asset_metadata', {}) or {}
    bounds = metadata.get('recovery_workspace_bounds_m', metadata.get('workspace_bounds_m'))
    bound_margins = {}
    if bounds:
        low, high = (np.asarray(v, dtype=float) for v in bounds)
        for name in sorted(origins):
            p = np.asarray(origins[name], dtype=float)
            bound_margins[name] = float(min((p-low).min(), (high-p).min()))
            if bound_margins[name] < 0:
                reasons.append({'rule': 'R1_outside_referee_bounds', 'body': name, 'margin_m': bound_margins[name]})
    support = _support(paths, owners, corners, coms, jointed) if len(paths) else {}
    for name, row in support.items():
        if row['status'] == 'unsupported':
            reasons.append({'rule': 'S_unsupported', 'body': name, 'air_gap_m': row['air_gap_m']})
        elif row['status'] == 'supported' and row['com_margin_m'] < 0:
            reasons.append({'rule': 'S_com_outside_support', 'body': name, 'margin_m': row['com_margin_m']})
    threshold = PENETRATION_FRACTION*referee_penetration_bound(task)
    contact = _interpenetration(stage, paths, owners, corners, layout, threshold) if len(paths) else {
        'candidates': 0, 'maximum_depth_m': 0., 'violations': [], 'violation_count': 0, 'unresolved': [], 'threshold_m': threshold}
    for row in contact['violations']:
        reasons.append({'rule': 'R3_penetration', **row})
    for first, second in contact['unresolved']:
        reasons.append({'rule': 'R3_unrepresentable_pair', 'first': first, 'second': second})
    margins = [row['com_margin_m'] for row in support.values() if row['status'] == 'supported']
    return {'schema': SCHEMA, 'applicable': True, 'passed': not reasons, 'reasons': reasons,
            'measurements': {'collider_count': len(paths), 'reset_bounds': bounds, 'bounds_margin_m': bound_margins,
                             'minimum_bounds_margin_m': min(bound_margins.values()) if bound_margins else None,
                             'minimum_com_margin_m': min(margins) if margins else None,
                             'support': support, 'interpenetration': contact},
            'thresholds': {'bounds_margin_m': 0., 'com_margin_m': 0., 'support_gap_m': SUPPORT_GAP_M,
                           'maximum_drop_m': MAXIMUM_DROP_M, 'penetration_m': threshold,
                           'referee_penetration_fraction': PENETRATION_FRACTION},
            'scope': SCOPE}


def _finite(value):
    """JSON-safe copy: numpy scalars become Python values, non-finite floats become None."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    return value


def check_reset_admission(values, layout, *, family, seed=None):
    """Reject an authored candidate whose reset is not physically admissible.

    ``values`` are the composed authoring values (final scene/task and
    ``actual_body_poses_m``); ``layout`` the applied group receipts.  Raises
    ``ResetAdmissionError`` with the full deterministic record on rejection.
    """
    if family not in RIGID_FAMILIES or seed == 0:
        return {'schema': SCHEMA, 'applicable': False, 'passed': True,
                'reason': 'canonical seed' if seed == 0 else 'family without rigid group layout'}
    from scripts.validation.workroom_randomized_workspace import composed_support_stage
    stage, _ = composed_support_stage(values['scene'])
    poses = values.get('actual_body_poses_m') or {}
    bodies = {name: row['prim_path'] for name, row in poses.items()}
    origins = {name: row['position_m'] for name, row in poses.items()}
    record = _finite(evaluate(stage, bodies=bodies, origins=origins, layout=layout, task=values['task']))
    if not record['passed']:
        summary = '; '.join(
            f"{row['rule']} {row.get('body') or row.get('first')}" + (
                f" {row['margin_m']:.4f} m" if 'margin_m' in row else
                f" {row['depth_m']:.4f} m" if 'depth_m' in row else '')
            for row in record['reasons'][:4])
        raise ResetAdmissionError('Reset admission rejected: '+summary, record)
    return record
