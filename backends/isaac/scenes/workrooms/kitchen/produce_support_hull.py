"""Explicit produce solids; USD imports are author-only and lazy.

The visible body and its single convex collider have the same closed polygons.
This is a real geometry variant, not a different hidden collision proxy.
Native cooking fidelity and stability remain separate required checks.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

RINGS = ((.002, 0.), (.006, .012), (.011, .036), (.0148, .065), (.0155, .086), (.013, .098))
SEGMENTS = 10
SUPPORT_HULL_VERSION = 'kitchen_seed_geometry_support_hull_v3'
SUPPORT_LOW_VERTEX_VERSION = 'kitchen_seed_geometry_support_low_vertex_v4'
SUPPORT_LOW_VERTEX_TIP_VERSION = 'kitchen_seed_geometry_support_low_vertex_tip_v5'
LOW_VERTEX_VERSIONS = (SUPPORT_LOW_VERTEX_VERSION, SUPPORT_LOW_VERTEX_TIP_VERSION)


def _recipe(profile):
    if profile == SUPPORT_HULL_VERSION:
        return SEGMENTS, tuple(range(6))
    if profile in LOW_VERTEX_VERSIONS:
        return 8, (0, 2, 4, 5)
    raise ValueError('Unknown explicit produce support-solid profile')


def _rings(profile, ring_indices):
    """Only v5 changes the small end radius; never alter the parent recipe."""
    return [(.0025, RINGS[i][1]) if profile == SUPPORT_LOW_VERTEX_TIP_VERSION and i == 0
            else RINGS[i] for i in ring_indices]


def _triangles(polygons):
    return np.asarray([[face[0], face[i], face[i+1]] for face in polygons
                       for i in range(1, len(face)-1)], dtype=np.int64)


def _mesh(stage, path, body):
    import trimesh
    from pxr import UsdGeom
    usd = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    points = np.asarray(usd.GetPointsAttr().Get(), dtype=float)
    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(usd.GetPrim(), body)
    if not np.allclose(np.asarray(matrix), np.eye(4), atol=1e-12, rtol=0):
        raise ValueError('Support-hull source requires the declared body-local kitchen mesh')
    indices = list(usd.GetFaceVertexIndicesAttr().Get()); polygons = []; cursor = 0
    for count in usd.GetFaceVertexCountsAttr().Get():
        polygons.append(indices[cursor:cursor+count]); cursor += count
    if cursor != len(indices):
        raise ValueError('Incomplete source polygon topology')
    solid = trimesh.Trimesh(points, _triangles(polygons), process=False)
    if not solid.is_watertight or not solid.is_winding_consistent or solid.volume <= 0:
        raise ValueError('Support hull requires a closed positive material solid')
    return solid, polygons


def _digest_mesh(mesh):
    return hashlib.sha256(np.asarray(mesh.vertices, '<f8').tobytes()
                          +np.asarray(mesh.faces, '<i8').tobytes()).hexdigest()


def _check_parent(stage, metadata, row):
    from scipy.spatial import cKDTree
    from pxr import UsdPhysics
    from .assets import _carrot
    body = stage.GetPrimAtPath(row['body_path'])
    if row['category'] != 'carrot' or row['collider_paths'] != [row['body_path']+'/Collision/body']:
        raise ValueError('Support hull requires the selected original single-collider carrots')
    collider = stage.GetPrimAtPath(row['collider_paths'][0])
    if UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() != 'convexHull':
        raise ValueError('Support hull preserves the original convexHull representation')
    original, polygons = _mesh(stage, row['collider_paths'][0], body)
    visible, visual_polygons = _mesh(stage, row['body_path']+'/Visual/body', body)
    if not np.array_equal(original.vertices, visible.vertices) or polygons != visual_polygons:
        raise ValueError('Selected parent visible material and collider geometry differ')
    scale = np.asarray(metadata['dimension_changes'][row['id']]['scale_xyz'], float)
    if not np.array_equal(scale, [1.25]*3):
        raise ValueError('Support hull requires the current household carrot dimensions')
    expected, _ = _carrot(int(row['id'].rsplit('_', 1)[1]))
    # Reproduce both historical writes: original USD float32, then household
    # scaling and its second float32 write. Do not substitute an early recipe.
    expected.vertices = (np.asarray(expected.vertices, np.float32).astype(float)*scale).astype(np.float32)
    distances, mapped = cKDTree(original.vertices).query(expected.vertices)
    topology = lambda faces: sorted(tuple(sorted(map(int, face))) for face in faces)
    if (len(original.vertices) != 386 or len(expected.vertices) != 386
            or len(set(mapped.tolist())) != 386 or distances.max() > 2e-8
            or topology(mapped[expected.faces]) != topology(original.faces)
            or not np.allclose(expected.metadata['natural_rest']['source_to_rest_transform'],
                               row['natural_rest']['source_to_rest_transform'], atol=1e-12, rtol=0)):
        raise ValueError('Selected parent differs from the reconstructed 64-sector physical recipe')
    mass = float(UsdPhysics.MassAPI(body).GetMassAttr().Get())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Positive actual parent mass required')
    return original, mass/original.volume, float(distances.max()), scale


def canonical_mesh(parent_rest, household_scale, *, profile=SUPPORT_HULL_VERSION):
    """Frozen v3 61-point solid or explicit v4/v5 33-point solid, in source units."""
    import trimesh
    from . import geometry as g
    segments, ring_indices = _recipe(profile)
    rings = _rings(profile, ring_indices)
    phase = -math.pi/2+math.pi/segments
    points = np.array([[r*math.cos(phase+k*2*math.pi/segments), .0505-h,
                        .015+r*math.sin(phase+k*2*math.pi/segments)]
                       for r, h in rings for k in range(segments)]
                      +[[0., .0505-.101, .015]])
    # This phase avoids the legacy circular mesh's sub-floor clipping. Every
    # frustum stays planar; the floor is an existing facet of the new solid.
    if points[:, 2].min() <= 0:
        raise ValueError('Unexpected clipped support-solid ring')
    points = trimesh.transform_points(points, parent_rest)*household_scale
    polygons = [list(range(segments))]
    polygons += [[i*segments+k, i*segments+(k+1)%segments,
                  (i+1)*segments+(k+1)%segments, (i+1)*segments+k]
                 for i in range(len(rings)-1) for k in range(segments)]
    last_ring, tip = (len(rings)-1)*segments, len(rings)*segments
    polygons += [[last_ring+k, last_ring+(k+1)%segments, tip] for k in range(segments)]
    center = points.mean(0)
    for face in polygons:
        vertices = points[face]
        if np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])@(vertices.mean(0)-center) < 0:
            face.reverse()
    mesh = trimesh.Trimesh(points, _triangles(polygons), process=False)
    transform, rest = g.natural_rest_transform(mesh)
    points = trimesh.transform_points(points, transform)
    points[np.abs(points[:, 2]) < 1e-12, 2] = 0.
    points = points.astype(np.float32)
    mesh = trimesh.Trimesh(points, _triangles(polygons), process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Invalid closed support-hull solid')
    return mesh, polygons, transform, rest


def _write_mesh(prim, points, polygons, *, carrot_colour=False):
    from pxr import Gf, UsdGeom, Vt
    from .assets import PALETTE
    usd = UsdGeom.Mesh(prim); points = np.asarray(points, np.float32)
    usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    usd.GetFaceVertexCountsAttr().Set([len(face) for face in polygons])
    usd.GetFaceVertexIndicesAttr().Set([i for face in polygons for i in face])
    usd.GetExtentAttr().Set([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
    normals = []
    for face in polygons:
        vertices = points[face].astype(float)
        normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
        normal /= np.linalg.norm(normal)
        normals.extend([normal]*len(face))
    usd.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
    usd.SetNormalsInterpolation('faceVarying')
    if carrot_colour:
        variation = .035*np.sin(points[:, 0]*117+points[:, 1]*205)*np.sin(points[:, 2]*240)
        colours = np.clip(np.asarray(PALETTE['carrot'][0])+variation[:, None], .005, .95)
        usd.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(colours.astype(np.float32)))


def _details(stage, row, old_rest, scale, new_rest, *, profile=SUPPORT_HULL_VERSION):
    import trimesh
    from pxr import Gf, Usd, UsdGeom, Vt
    from . import geometry as g
    root = stage.GetPrimAtPath(row['body_path']+'/Visual')
    expected = {'body'} | {f'root_ring_{i}' for i in range(5)} | {f'stem_{i}' for i in range(3)}
    actual = {p.GetName() for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)}
    if actual != expected:
        raise ValueError('Unexpected selected-parent carrot visual detail inventory')
    segments, ring_indices = _recipe(profile)
    rings = _rings(profile, ring_indices)
    phase = -math.pi/2+math.pi/segments; step = 2*math.pi/segments
    for i, y in enumerate((-.039, -.024, -.005, .018, .035)):
        curve = []
        for theta in np.linspace(.20, math.pi-.20, 24):
            actual_y = y+.0006*math.sin(theta*3)
            radius = float(np.interp(.0505-actual_y, [h for r, h in rings]+[.101],
                                    [r for r, h in rings]+[0.]))
            delta = (theta-phase)%step-step/2
            radial = .98*radius*math.cos(math.pi/segments)/math.cos(delta)
            curve.append((radial*math.cos(theta), actual_y, .015+radial*math.sin(theta)))
        detail = g.tube(curve, .00038, 8)
        detail.apply_transform(old_rest); detail.apply_scale(scale); detail.apply_transform(new_rest)
        _write_mesh(stage.GetPrimAtPath(row['body_path']+f'/Visual/root_ring_{i}'),
                    detail.vertices, detail.faces.tolist())
    for i in range(3):
        usd = UsdGeom.Mesh(stage.GetPrimAtPath(row['body_path']+f'/Visual/stem_{i}'))
        points = trimesh.transform_points(np.asarray(usd.GetPointsAttr().Get(), float), new_rest).astype(np.float32)
        normals = np.asarray(usd.GetNormalsAttr().Get(), float)@new_rest[:3, :3].T
        usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        usd.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        usd.GetExtentAttr().Set([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])


def finalize(stage, row):
    """Integrate the final authored float32 solid, including seed XY scaling."""
    from scipy.spatial import ConvexHull
    from pxr import Gf, UsdPhysics
    from . import geometry as g
    body = stage.GetPrimAtPath(row['body_path'])
    mesh, polygons = _mesh(stage, row['collider_paths'][0], body)
    profile = row['support_hull_profile']
    segments, ring_indices = _recipe(profile['profile'])
    vertex_count = len(ring_indices)*segments+1
    mass = profile['parent_actual_density_kg_m3']*mesh.volume
    props = g.mass_properties([mesh], mass)
    api = UsdPhysics.MassAPI(body)
    api.GetMassAttr().Set(mass); api.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['center_of_mass_m']))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
    q = props['principal_axes_xyzw']; api.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    row.update(props, mass_kg=float(api.GetMassAttr().Get()), mass=float(api.GetMassAttr().Get()),
        center_of_mass_m=list(api.GetCenterOfMassAttr().Get()),
        diagonal_inertia_kg_m2=list(api.GetDiagonalInertiaAttr().Get()))
    stored = api.GetPrincipalAxesAttr().Get(); row['principal_axes_xyzw'] = [*stored.GetImaginary(), stored.GetReal()]
    row['collision_support_points_m'] = np.asarray(mesh.vertices).tolist()
    row['collision_bounds_m'] = mesh.bounds.tolist()
    row['support_shape']['bottom_local_z_m'] = float(mesh.bounds[0, 2])
    bottom = np.asarray(mesh.vertices)[np.abs(mesh.vertices[:, 2]) < 1e-10]
    if len(bottom) != 4 or mesh.bounds[0, 2] != 0:
        raise ValueError('Support hull must retain its four exact floor vertices')
    hull = ConvexHull(bottom[:, :2]); margin = float(-(hull.equations[:, :2]@mesh.center_mass[:2]+hull.equations[:, 2]).max())
    if margin < .0035 or len(mesh.vertices) != vertex_count or len(polygons) != vertex_count:
        raise ValueError('Source geometry lost its declared topology or stable support')
    if profile['profile'] in LOW_VERTEX_VERSIONS:
        # This is the worst-case triangulated envelope, independent of whether
        # the native incremental hull builder merges coplanar faces early.
        triangles = len(ConvexHull(mesh.vertices).simplices)
        if vertex_count != 33 or len(mesh.faces) != 62 or triangles > 62:
            raise ValueError('Low-vertex solid exceeds the explicit 33-point/62-triangle envelope')
        profile.update(source_triangulated_faces=62, source_convex_hull_triangles=triangles,
                       collision_mesh_coordinate_scale=1.)
    profile.update(source_vertex_count=vertex_count, source_polygon_count=vertex_count,
        support_plane_body_z_m=0., support_vertices_body_m=bottom.tolist(),
        integrated_com_support_margin_m=margin, final_collision_mesh_sha256=_digest_mesh(mesh),
        final_collision_volume_m3=float(mesh.volume), final_effective_density_kg_m3=profile['parent_actual_density_kg_m3'],
        native_cooked_geometry_verified=False)
    if 'seed_dimension_profile' in row:
        row['seed_dimension_profile'].update(source=('Verified decagonal support-hull material solid'
            if profile['profile'] == SUPPORT_HULL_VERSION else 'Verified 33-point octagonal support-hull material solid'),
            mass_rule='Density preserved from the actual selected parent solid; final float32 polygon mesh mass/COM/inertia integrated after seed scaling')


def apply(stage, metadata, row, *, profile=SUPPORT_HULL_VERSION):
    segments, ring_indices = _recipe(profile)
    original, density, error, scale = _check_parent(stage, metadata, row)
    old_rest = np.asarray(row['natural_rest']['source_to_rest_transform'], float)
    mesh, polygons, transform, rest = canonical_mesh(old_rest, scale, profile=profile)
    for suffix in ('Visual/body', 'Collision/body'):
        _write_mesh(stage.GetPrimAtPath(row['body_path']+'/'+suffix), mesh.vertices, polygons,
                    carrot_colour=suffix.startswith('Visual'))
    _details(stage, row, old_rest, scale, transform, profile=profile)
    row['support_hull_profile'] = {'schema': 'kitchen_carrot_support_hull.v1',
        'profile': profile,
        'parent_collision_mesh_sha256': _digest_mesh(original),
        'parent_recipe_reconstruction_max_error_m': error,
        'parent_actual_density_kg_m3': density, 'source_radial_segments': 64,
        'radial_segments': segments, 'axial_radius_height_profile_m': [list(p) for p in _rings(profile, ring_indices)],
        'phase_rad': -math.pi/2+math.pi/segments, 'tip_height_m': .101,
        'canonical_extra_rest_transform': transform.tolist(), 'parent_natural_rest': row['natural_rest'],
        'visual_collision_main_solid_identical': True,
        'visual_details': 'Root-ring relief conforms to the decagonal radial profile; stems receive the same rigid rest-frame change. Details remain visual only.',
        'scope': 'Real visible and collision material geometry variant; no native stability or manipulation acceptance'}
    row['natural_rest'] = {**rest, 'source': 'Decagonal closed material solid before seed XY scaling'}
    if profile in LOW_VERTEX_VERSIONS:
        row['support_hull_profile'].update(schema='kitchen_carrot_support_hull.v2',
            parent_axial_ring_indices=list(ring_indices),
            visual_details='Root-ring relief conforms to the four-ring octagonal radial profile; stems receive the same rigid rest-frame change. Details remain visual only.')
        row['natural_rest']['source'] = 'Four-ring octagonal closed material solid before seed XY scaling'
    if profile == SUPPORT_LOW_VERTEX_TIP_VERSION:
        row['support_hull_profile'].update(schema='kitchen_carrot_support_hull.v3',
            parent_ring_radius_overrides_m={'0': {'original': .002, 'variant': .0025}},
            tip_radius_change_scope='Only first axial ring radius changes; other ring vertices, axis heights and four-point support facet retain v4 geometry before seed scaling.')
    finalize(stage, row)
