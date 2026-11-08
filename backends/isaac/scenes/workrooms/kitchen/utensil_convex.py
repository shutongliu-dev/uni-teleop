"""Exact convex partition of flat extruded cutlery; never fills fork slots.

The existing authored solid, visual, explicit mass and inertia are unchanged.
Convex prisms partition its planar triangulation instead of sampling it as SDF.
"""
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull
import trimesh


def partition_prismatic_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    lower, upper = vertices[:, 2].min(), vertices[:, 2].max()
    if not np.all(np.isclose(vertices[:, 2], lower, atol=1e-10, rtol=0.) |
                  np.isclose(vertices[:, 2], upper, atol=1e-10, rtol=0.)):
        raise ValueError('Exact prism partition requires two planar vertex layers')
    top = mesh.faces[np.all(np.isclose(mesh.triangles[:, :, 2], upper,
                                       atol=1e-10, rtol=0.), axis=1)]
    polygons = [set(map(int, face)) for face in top]
    areas = [float(abs(np.cross(vertices[f[1]]-vertices[f[0]],
                               vertices[f[2]]-vertices[f[0]])[2])/2) for f in top]
    if not polygons or min(areas) <= 0:
        raise ValueError('Missing nondegenerate planar contact triangulation')
    while True:
        candidates = []
        for i, j in combinations(range(len(polygons)), 2):
            if len(polygons[i] & polygons[j]) < 2:
                continue
            ids = sorted(polygons[i] | polygons[j])
            hull = ConvexHull(vertices[ids, :2])
            if np.isclose(hull.volume, areas[i]+areas[j], rtol=1e-10, atol=1e-14):
                candidates.append((hull.volume, i, j, set(ids)))
        if not candidates:
            break
        area, i, j, ids = max(candidates, key=lambda row: (row[0], -row[1], -row[2]))
        polygons[i], areas[i] = ids, area
        del polygons[j], areas[j]
    parts = []
    for ids in polygons:
        xy = vertices[sorted(ids), :2]
        points = np.vstack((np.c_[xy, np.full(len(xy), lower)],
                            np.c_[xy, np.full(len(xy), upper)]))
        parts.append(trimesh.convex.convex_hull(points))
    volume = sum(p.volume for p in parts)
    if not np.isclose(volume, mesh.volume, rtol=1e-9, atol=1e-14):
        raise ValueError('Convex prisms changed the original solid volume')
    combined = trimesh.util.concatenate(parts)
    if not np.array_equal(combined.bounds, mesh.bounds):
        raise ValueError('Convex prisms changed the original contact bounds')
    if not np.allclose(combined.center_mass, mesh.center_mass, rtol=0., atol=1e-10):
        raise ValueError('Convex prisms changed integrated center of mass')
    if not np.allclose(combined.moment_inertia, mesh.moment_inertia, rtol=1e-8, atol=1e-16):
        raise ValueError('Convex prisms changed integrated inertia')
    return parts, {'method': 'Exact convex merges of original planar triangles extruded between original layers.',
        'part_count': len(parts), 'volume_before_m3': float(mesh.volume),
        'volume_after_m3': float(volume), 'maximum_COM_difference_m': float(np.max(abs(combined.center_mass-mesh.center_mass))),
        'maximum_unit_density_inertia_difference': float(np.max(abs(combined.moment_inertia-mesh.moment_inertia))),
        'original_boundary_and_fork_slots_preserved': True,
        'mass_inertia_material_pose_visual_changes': False}


def apply_prismatic_colliders(author):
    from pxr import UsdGeom, UsdPhysics, UsdShade
    from . import geometry as g
    for row in author.meta['objects']:
        if row['role'] not in ('fork', 'knife'):
            continue
        old_path = row['body_path']+'/Collision/utensil'
        usd = UsdGeom.Mesh(author.stage.GetPrimAtPath(old_path))
        original = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
        parts, receipt = partition_prismatic_mesh(original)
        UsdPhysics.CollisionAPI(usd.GetPrim()).GetCollisionEnabledAttr().Set(False)
        paths = []
        for index, mesh in enumerate(parts):
            path = row['body_path']+f'/Collision/utensil_convex_{index:02d}'
            prim = g.write_mesh(author.stage, path, mesh, author.materials['steel'], collision='convexHull')
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
            UsdShade.MaterialBindingAPI(prim).Bind(author.physical_material, materialPurpose='physics')
            paths.append(path)
        row['collider_paths'] = paths
        row['collision_count'] = len(paths)
        row['exact_convex_collision_partition'] = receipt


def deactivate_replaced_colliders(author):
    """Exclude obsolete disabled SDF prims from native shape construction.

    Their original layer specifications remain intact for inspection. Explicit
    body mass properties and the separately authored visible geometry are kept.
    """
    from pxr import UsdPhysics
    plans = []
    for row in author.meta['objects']:
        if row['role'] not in ('spoon', 'fork', 'knife'):
            continue
        old_path = row['body_path']+'/Collision/utensil'
        old = author.stage.GetPrimAtPath(old_path)
        if (not old or not old.IsActive()
                or UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Get() is not False
                or old_path in row['collider_paths'] or not row['collider_paths']):
            raise ValueError('Expected disabled original cutlery collider and active replacement pieces')
        for path in row['collider_paths']:
            prim = author.stage.GetPrimAtPath(path)
            if (not prim or not prim.IsActive()
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True):
                raise ValueError('Missing enabled replacement cutlery collider')
        plans.append((row, old))
    for row, old in plans:
        old.SetActive(False)
        row['deactivated_legacy_collision_prim'] = str(old.GetPath())
