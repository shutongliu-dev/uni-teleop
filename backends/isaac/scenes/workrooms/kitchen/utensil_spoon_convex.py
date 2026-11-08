"""Small convex pieces following the original spoon's open bowl and handle.

The ideal 72-sector recipe is partitioned exactly. Existing USD float32
corners are reused; their small non-coplanarity has a measured error envelope,
not a claim of bitwise triangle-surface identity. One original rigid body and
its explicit mass properties remain unchanged. Native acceptance is separate.
"""
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from . import geometry as g


_PROFILE = np.array([(0, 0), (.006, 0), (.012, .002), (.0175, .007),
                     (.018, .009), (.0165, .010), (.014, .006),
                     (.008, .0035), (0, .003)])
_MERIDIAN = ((0, 1, 2, 7, 8), (2, 3, 4, 5, 6), (2, 6, 7))
_HANDLE = np.array([(-.007, -.015), (-.008, .045), (-.010, .068),
                    (-.008, .078), (-.004, .081), (.004, .081),
                    (.008, .078), (.010, .068), (.008, .045), (.007, -.015)])
_STEM = ((0, 9, 8), (0, 8, 7, 6, 5, 4, 3, 1), (3, 2, 1))


def _triangles(faces):
    return {tuple(sorted(map(int, face))) for face in faces}


def _overlap_bound(parts):
    maximum = 0.
    unresolved = 0
    for a, b in combinations(parts, 2):
        if np.any(a.bounds[1] < b.bounds[0]) or np.any(b.bounds[1] < a.bounds[0]):
            continue
        axes = np.vstack((a.face_normals, b.face_normals))
        pa, pb = a.vertices@axes.T, b.vertices@axes.T
        gap = np.maximum(pb.min(0)-pa.max(0), pa.min(0)-pb.max(0)).max()
        if gap < 0:
            unresolved += 1
            maximum = max(maximum, float(-gap))
    return {'conservative_overlap_depth_bound_m': maximum+1e-12,
            'pairs_without_nonnegative_facet_axis_gap': unresolved,
            'strict_zero_overlap_proven': unresolved == 0,
            'method': 'Unit face-normal support intervals; a translation by this bound separates every pair. This is not an intersection-volume estimate.'}


def _bowl_reference(vertices, lookup, corner_sets):
    """Prove coverage using the original boundary, rather than sampled points."""
    meridian = g.extruded_outline(_PROFILE, 1.)
    top = meridian.faces[np.all(meridian.triangles[:, :, 2] > .5, axis=1)]
    triangles = cKDTree(_PROFILE).query(meridian.vertices[top, :2])[1]
    counts, tet_ids = Counter(), []
    maximum_displacement = 0.
    for triangle in triangles:
        triangle = sorted(triangle)
        for j in range(72):
            angles = np.array([j, (j+1) % 72])*2*np.pi/72
            points = [(r*np.cos(t), 1.45*r*np.sin(t)-.053, z)
                      for t in angles for r, z in _PROFILE[triangle]]
            nominal, errors, ids = lookup(points)
            maximum_displacement = max(maximum_displacement, float(errors.max()))
            for selected in ((0, 1, 2, 5), (0, 1, 4, 5), (0, 3, 4, 5)):
                ti = ids[list(selected)]
                if len(set(map(int, ti))) < 4:
                    continue
                old, now = nominal[list(selected)], vertices[ti]
                if np.linalg.det(old[1:]-old[0])*np.linalg.det(now[1:]-now[0]) <= 0:
                    raise ValueError('Spoon corner rounding inverted a reference tetrahedron')
                if not any(set(map(int, ti)) <= group for group in corner_sets):
                    raise ValueError('Original spoon tetrahedron is not covered by a convex piece')
                tet_ids.append(ti)
                for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    counts[tuple(sorted(map(int, ti[list(face)])))] += 1
    bowl = g.lathe(_PROFILE, 72, (1., 1.45))
    bowl.apply_translation((0, -.053, 0))
    _, _, ids = lookup(bowl.vertices)
    boundary = _triangles(ids[bowl.faces])
    if boundary != {face for face, n in counts.items() if n == 1} or max(counts.values()) != 2:
        raise ValueError('Spoon reference partition changed original bowl boundary triangles')
    points = vertices[np.asarray(tet_ids)]
    volume = float(np.abs(np.linalg.det(points[:, 1:]-points[:, :1])).sum()/6.)
    original_volume = float(trimesh.Trimesh(vertices[ids], bowl.faces, process=False).volume)
    if not np.isclose(volume, original_volume, rtol=1e-12, atol=1e-18):
        raise ValueError('Spoon reference tetrahedra changed the actual bowl volume')
    return {'reference_tetrahedra': len(tet_ids), 'original_boundary_triangles': len(boundary),
            'original_boundary_index_triples_identical': True, 'reference_tetrahedra_noninverted': True,
            'every_reference_tet_contained_in_one_convex': True,
            'maximum_original_vertex_rounding_m': maximum_displacement,
            'bowl_convex_union_to_original_solid_distance_bound_m': 2*maximum_displacement+1e-12,
            'bound_proof': 'Each hull point maps within delta to the exact nominal sector, then within delta through a reference-tet barycentric map to the original solid. The convex union contains every original reference tet; the bowl cavity can change only inside this 2*delta envelope.'}


def partition_spoon_mesh(mesh, *, rest_transform=None, scale_xyz=(1., 1., 1.)):
    """Return 220 hulls using actual corners, accepting arbitrary vertex order.

    The declared recipe-to-body transform is validated against every original
    vertex and triangle. Sorting uses its natural rest transform; clearing
    uses identity. This rejects changed geometry instead of silently fitting it.
    """
    from .assets import _utensil
    transform = np.asarray(np.eye(4) if rest_transform is None else rest_transform, dtype=float)
    scale = np.asarray(scale_xyz, dtype=float)
    if (transform.shape != (4, 4) or scale.shape != (3,) or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(scale)) or np.any(scale <= 0)
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-12, rtol=0)
            or np.linalg.det(transform[:3, :3]) <= 0):
        raise ValueError('Invalid spoon rigid rest transform or metric scale')
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(vertices)

    def lookup(points):
        nominal = trimesh.transform_points(np.asarray(points), transform)*scale
        errors, ids = tree.query(nominal)
        if errors.max() > 2e-8:
            raise ValueError('Spoon mesh differs from its declared recipe by more than float32 rounding')
        return nominal, errors, ids

    recipe, _ = _utensil('spoon', resting=False)
    _, _, all_ids = lookup(recipe.vertices)
    if (not mesh.is_watertight or not mesh.is_winding_consistent
            or len(mesh.faces) != len(recipe.faces)
            or len(vertices) != len(recipe.vertices) or len(np.unique(all_ids)) != len(vertices)
            or _triangles(all_ids[recipe.faces]) != _triangles(mesh.faces)):
        raise ValueError('Spoon original vertex/triangle topology changed')
    parts, corner_sets = [], []

    def add(points):
        _, _, ids = lookup(points)
        hull = trimesh.convex.convex_hull(vertices[ids])
        if not hull.is_watertight or not hull.is_convex or hull.volume <= 0:
            raise ValueError('Invalid spoon convex piece')
        parts.append(hull)
        corner_sets.append(set(map(int, ids)))

    for polygon in _MERIDIAN:
        for j in range(72):
            angles = np.array([j, (j+1) % 72])*2*np.pi/72
            add([(r*np.cos(t), 1.45*r*np.sin(t)-.053, z)
                 for t in angles for r, z in _PROFILE[list(polygon)]])
    add(g.box((.014, .020, .006), (0, -.020, .005)).vertices)
    for polygon in _STEM:
        add([(x, y, z) for z in (.002, .008) for x, y in _HANDLE[list(polygon)]])
    combined = trimesh.util.concatenate(parts)
    volume_error = float(combined.volume/mesh.volume-1.)
    com_error = float(abs(combined.center_mass-mesh.center_mass).max())
    inertia_error = float(np.linalg.norm(combined.moment_inertia-mesh.moment_inertia)/np.linalg.norm(mesh.moment_inertia))
    if (abs(volume_error) > 1e-6 or com_error > 5e-8 or inertia_error > 1e-6
            or not np.array_equal(combined.bounds, mesh.bounds)):
        raise ValueError('Spoon convex rounding changed the measured solid beyond its numerical envelope')
    reference = _bowl_reference(vertices, lookup, corner_sets[:216])
    overlap = {'bowl': _overlap_bound(parts[:216]), 'stem': _overlap_bound(parts[-3:])}
    if max(row['conservative_overlap_depth_bound_m'] for row in overlap.values()) > 5e-8:
        raise ValueError('Spoon convex partition acquired material overlap beyond float32 rounding')
    return parts, {
        'schema': 'spoon_open_bowl_convex_partition.v1',
        'method': 'Three convex meridian regions in each original 5-degree sector; original bridge and three convex stem prisms.',
        'part_count': len(parts), 'bowl_part_count': 216, 'bridge_part_count': 1, 'stem_part_count': 3,
        'maximum_vertex_count': max(len(p.vertices) for p in parts),
        'original_volume_m3': float(mesh.volume), 'sum_convex_volume_m3': float(combined.volume),
        'relative_volume_difference': volume_error, 'maximum_COM_difference_m': com_error,
        'relative_unit_density_inertia_difference': inertia_error,
        'bounds_identical': True, 'all_corners_reused_from_original_usd': True,
        'original_recipe_triangle_topology_verified': True, 'finite_precision_reference': reference,
        'internal_rounding_overlap': overlap,
        'original_component_overlap': 'Original intersecting bowl/bridge/stem solids and their additive effective-density model are retained; no Boolean union or density compensation.',
        'precision_limit': 'Exact partition of the ideal recipe. Original float32 noncoplanarity creates only the measured nanometre convexification/overlap envelope; not bitwise triangle-surface identity.',
        'explicit_mass_COM_inertia_changed': False, 'visual_material_pose_changed': False,
        'native_physics_qualified': False}


def apply_spoon_convex_colliders(author):
    """Opt-in authoring hook, after natural-rest and household scale authoring."""
    from pxr import UsdGeom, UsdPhysics, UsdShade
    plans = []
    for row in author.meta['objects']:
        if row['role'] != 'spoon':
            continue
        path = row['body_path']+'/Collision/utensil'
        usd = UsdGeom.Mesh(author.stage.GetPrimAtPath(path))
        if not usd or not UsdPhysics.CollisionAPI(usd.GetPrim()).GetCollisionEnabledAttr().Get():
            raise ValueError('Expected original enabled spoon collider')
        if author.stage.GetPrimAtPath(row['body_path']+'/Collision/spoon_convex_000'):
            raise ValueError('Spoon convex partition has already been authored')
        mesh = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
        parts, receipt = partition_spoon_mesh(mesh,
            rest_transform=row.get('natural_rest', {}).get('source_to_rest_transform'),
            scale_xyz=row.get('dimension_profile', {}).get('linear_scale_xyz', (1., 1., 1.)))
        plans.append((row, usd, parts, receipt))
    for row, usd, parts, receipt in plans:
        paths = []
        old = usd.GetPrim()
        for index, part in enumerate(parts):
            path = row['body_path']+f'/Collision/spoon_convex_{index:03d}'
            prim = g.write_mesh(author.stage, path, part, author.materials['steel'], collision='convexHull')
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
            UsdShade.MaterialBindingAPI(prim).Bind(author.physical_material, materialPurpose='physics')
            for name in ('physxCollision:contactOffset', 'physxCollision:restOffset'):
                prim.GetAttribute(name).Set(old.GetAttribute(name).Get())
            for relation in old.GetRelationships():
                if relation.GetName().startswith('material:binding'):
                    prim.CreateRelationship(relation.GetName()).SetTargets(relation.GetTargets())
            paths.append(path)
        UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Set(False)
        row['collider_paths'], row['collision_count'] = paths, len(paths)
        row['spoon_convex_collision_partition'] = receipt
