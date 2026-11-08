"""Opt-in five-shape spoon: original open bowl SDF and four handle convexes.

The original component intersections and explicit body mass properties remain
unchanged. Smaller SDF bounds improve requested sampling pitch, but native
distance-field, cooked-convex, force and stability validation remain required.
"""
import hashlib

import numpy as np
import trimesh

from bench.simulation.isaacsim_backend.asset_catalog import condition_collision_mesh
from . import geometry as g
from .utensil_spoon_convex import partition_spoon_mesh


def _component(mesh, face_ids):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    used = np.unique(faces[face_ids])
    inverse = np.full(len(vertices), -1, dtype=int)
    inverse[used] = np.arange(len(used))
    result = trimesh.Trimesh(vertices[used], inverse[faces[face_ids]], process=False)
    if not result.is_watertight or not result.is_winding_consistent or result.volume <= 0:
        raise ValueError('Original spoon component is not a closed positive solid')
    if not np.array_equal(result.triangles, vertices[faces[face_ids]]):
        raise ValueError('Spoon component split changed original triangle corners')
    return result


def split_spoon_collision_mesh(mesh, *, rest_transform=None, scale_xyz=(1., 1., 1.), sdf_resolution=768):
    """Return the original bowl plus bridge and three original-corner hulls."""
    if type(sdf_resolution) is not int or sdf_resolution not in (384, 768):
        raise ValueError('Spoon bowl SDF resolution must be explicitly 384 or 768')
    # This validates all actual original recipe vertices and triangle topology,
    # independently of vertex ordering and sorting/clearing rest transforms.
    convexes, convex_receipt = partition_spoon_mesh(mesh, rest_transform=rest_transform, scale_xyz=scale_xyz)
    components = sorted(trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces))), key=len, reverse=True)
    if [len(ids) for ids in components] != [1008, 36, 12]:
        raise ValueError('Expected original separate spoon bowl, stem and bridge components')
    bowl = _component(mesh, components[0])
    pieces = [bowl, *convexes[216:]]
    if len(pieces) != 5:
        raise ValueError('Spoon hybrid must contain exactly five shapes')
    combined = trimesh.util.concatenate(pieces)
    volume_error = float(combined.volume/mesh.volume-1.)
    com_error = float(abs(combined.center_mass-mesh.center_mass).max())
    inertia_error = float(np.linalg.norm(combined.moment_inertia-mesh.moment_inertia)/np.linalg.norm(mesh.moment_inertia))
    if (abs(volume_error) > 1e-6 or com_error > 5e-8 or inertia_error > 1e-6
            or not np.array_equal(combined.bounds, mesh.bounds)):
        raise ValueError('Five-shape spoon changed original physical geometry beyond float32 rounding')
    longest = float(np.ptp(bowl.vertices, axis=0).max())
    return pieces, {
        'schema': 'spoon_split_bowl_collision_partition.v1',
        'method': 'Original closed bowl face component as one SDF, original bridge convex and three stem convexes on the same rigid body.',
        'part_count': 5, 'bowl_sdf_count': 1, 'handle_convex_count': 4,
        'bowl_vertex_count': len(bowl.vertices), 'bowl_triangle_count': len(bowl.faces),
        'bowl_original_triangle_corners_exact': True,
        'bowl_triangle_corners_f32_sha256': hashlib.sha256(np.asarray(bowl.triangles, dtype='<f4').tobytes()).hexdigest(),
        'bowl_bounds_m': bowl.bounds.tolist(), 'bowl_sdf_resolution': sdf_resolution,
        'bowl_requested_grid_pitch_m': longest/sdf_resolution,
        'grid_pitch_scope': 'Schema-derived longest local extent / resolution; not a native cooked SDF readback.',
        'convex_coordinate_scale': 512., 'convex_inverse_mesh_scale': 1/512,
        'original_volume_m3': float(mesh.volume), 'sum_shape_volume_m3': float(combined.volume),
        'relative_volume_difference': volume_error, 'maximum_COM_difference_m': com_error,
        'relative_unit_density_inertia_difference': inertia_error, 'bounds_identical': True,
        'original_component_overlap': convex_receipt['original_component_overlap'],
        'convex_precision_limit': convex_receipt['precision_limit'],
        'stem_internal_rounding_overlap': convex_receipt['internal_rounding_overlap']['stem'],
        'explicit_body_mass_COM_inertia_changed': False, 'visual_material_pose_changed': False,
        'native_physics_qualified': False}


def apply_spoon_split_bowl_colliders(author, sdf_resolution=768):
    """Author an explicit candidate after household scale and natural rest.

    Original whole-spoon Collision/utensil only has collisionEnabled disabled.
    A caller may subsequently deactivate it through its common asset policy.
    """
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    if type(sdf_resolution) is not int or sdf_resolution not in (384, 768):
        raise ValueError('Spoon bowl SDF resolution must be explicitly 384 or 768')
    names = ('spoon_bowl_sdf', 'spoon_bridge_convex',
             'spoon_stem_convex_00', 'spoon_stem_convex_01', 'spoon_stem_convex_02')
    plans = []
    for row in author.meta['objects']:
        if row['role'] != 'spoon':
            continue
        prefix = row['body_path']+'/Collision/'
        old = author.stage.GetPrimAtPath(prefix+'utensil')
        if not old or not UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Get():
            raise ValueError('Expected original enabled whole-spoon collider')
        if any(author.stage.GetPrimAtPath(prefix+name) for name in names):
            raise ValueError('Split-bowl spoon colliders have already been authored')
        if UsdPhysics.MeshCollisionAPI(old).GetApproximationAttr().Get() != 'sdf':
            raise ValueError('Expected original whole-spoon SDF parameters')
        usd = UsdGeom.Mesh(old)
        mesh = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
        pieces, receipt = split_spoon_collision_mesh(mesh,
            rest_transform=row.get('natural_rest', {}).get('source_to_rest_transform'),
            scale_xyz=row.get('dimension_profile', {}).get('linear_scale_xyz', (1., 1., 1.)),
            sdf_resolution=sdf_resolution)
        plans.append((row, old, pieces, receipt))
    for row, old, pieces, receipt in plans:
        paths = []
        sdf_parameters = {}
        for index, (name, piece) in enumerate(zip(names, pieces, strict=True)):
            path = row['body_path']+'/Collision/'+name
            inverse = None
            if index:
                points, inverse = condition_collision_mesh(piece.vertices, 512.)
                if not np.array_equal(points.astype(float)*inverse, piece.vertices):
                    raise ValueError('Convex cooking coordinates changed physical spoon vertices')
                piece = trimesh.Trimesh(points, piece.faces, process=False)
            prim = g.write_mesh(author.stage, path, piece, author.materials['steel'],
                                collision='sdf' if index == 0 else 'convexHull')
            if inverse is not None:
                UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*inverse))
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
            UsdShade.MaterialBindingAPI(prim).Bind(author.physical_material, materialPurpose='physics')
            for attr in old.GetAttributes():
                attr_name = attr.GetName()
                if (attr_name.startswith('physxCollision:') or
                        (index == 0 and attr_name.startswith('physxSDFMeshCollision:'))):
                    value = attr.Get()
                    if value is not None:
                        prim.CreateAttribute(attr_name, attr.GetTypeName(), custom=attr.IsCustom(),
                                             variability=attr.GetVariability()).Set(value)
                        if attr_name.startswith('physxSDFMeshCollision:'):
                            sdf_parameters[attr_name] = value
            if index == 0:
                prim.GetAttribute('physxSDFMeshCollision:sdfResolution').Set(sdf_resolution)
            for relation in old.GetRelationships():
                if relation.GetName().startswith('material:binding'):
                    prim.CreateRelationship(relation.GetName()).SetTargets(relation.GetTargets())
            paths.append(path)
        UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Set(False)
        receipt['original_sdf_parameters'] = sdf_parameters
        receipt['sdf_parameters_changed'] = {'physxSDFMeshCollision:sdfResolution': sdf_resolution}
        row['collider_paths'], row['collision_count'] = paths, len(paths)
        row['spoon_split_bowl_collision_partition'] = receipt
