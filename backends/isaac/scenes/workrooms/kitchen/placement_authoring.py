"""Offline USD reference overlays: kitchen world-X reset and albedo only."""
from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np

from scenes.workrooms.seed_geometry import canonical_hash, file_hash, finish_texture, write_json
from scenes.workrooms.mechanical_assembly.placement_authoring import (
    _assert_internal_joints, _surface_texture, _usd_points, _world_pose)
from .placement_variants import ROOT, SOURCE_FILES, VARIANT_SCOPE, _metadata


def _author_source_identity():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    modules = tuple(name[:-3].replace('/', '.') for name in SOURCE_FILES)
    for name in modules:
        importlib.import_module(name)
    root = Path(__file__).resolve().parents[3]
    proof = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
    if not proof or not proof.get('runtime_imports_verified'):
        raise ValueError('Kitchen placement publication requires a complete frozen source snapshot')
    path = root/'source_snapshot_manifest.json'
    return {'snapshot_root': str(root), 'manifest_path': str(path), 'manifest_sha256': file_hash(path),
        'source_files_sha256': {str(root/name): file_hash(root/name) for name in SOURCE_FILES}}


def protected_properties(stage, moving):
    """Read actual shapes, all mechanics, bindings and all visual mesh geometry.

Only X in the named root translation ops is normalized. Renderer materials/UVs
are excluded, but physics material properties and purpose bindings are retained.
"""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    result = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        path = str(prim.GetPath())
        if not (prim.IsA(UsdGeom.Xformable) or prim.IsA(UsdPhysics.Joint)
                or any('Physics' in s or 'Physx' in s for s in prim.GetAppliedSchemas())):
            continue
        attrs = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(('physics:', 'physx', 'xformOp')) or name in (
                    'points', 'faceVertexCounts', 'faceVertexIndices', 'subdivisionScheme',
                    'size', 'radius', 'height', 'axis', 'extent', 'visibility', 'purpose'):
                value = attr.Get()
                if path in moving and name == 'xformOp:translate':
                    value = ['allowed_world_x', float(value[1]), float(value[2])]
                attrs[name] = repr(value)
        row = {'type': prim.GetTypeName(), 'attrs': attrs,
            'apis': [s for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s],
            'relationships': {r.GetName(): list(map(str, r.GetTargets()))
                for r in prim.GetRelationships() if r.GetName().startswith(('physics:', 'material:binding:physics'))}}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            row['physics_material'] = ({a.GetName(): repr(a.Get()) for a in material.GetPrim().GetAttributes()
                if a.GetName().startswith(('physics:', 'physx'))} if material else None)
        result[path] = row
    return result


def _sponge_skin_albedo(stage, metadata, texture, seed):
    """Retain the original porous colour signal used by the runtime skin.

Both meshes use vertex displayColor through their original primvar reader.
Multiplying those colours preserves the pore/detail pattern and every shader
connection. The existing runtime skin copies these primvars from its hidden
rest template, so the visible skinned foam receives the same light pattern.
"""
    from PIL import Image
    from pxr import UsdGeom, UsdShade, Vt
    root = metadata['actors']['sponge']+'/Visual/'
    paths = [root+'finish_porous_foam', root+'finish_green_scouring_layer']
    bindings = metadata['elastic_composites']['sponge']['visual_skin_bindings']
    if len(bindings) != 1 or bindings[0]['prim_path'] != paths[0]:
        raise ValueError('Unexpected original elastic skin template')
    with Image.open(texture) as image:
        pattern = np.asarray(image.convert('RGB'), dtype=float)/255.
    receipts = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        colors = UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor')
        values = np.asarray(colors.Get(), dtype=np.float32)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader = material.ComputeSurfaceSource()[0]
        connection = shader.GetInput('diffuseColor').GetConnectedSource()
        reader = UsdShade.Shader(connection[0].GetPrim()) if connection else None
        if (values.shape != points.shape or colors.GetInterpolation() != 'vertex' or colors.IsIndexed()
                or not reader or reader.GetIdAttr().Get() != 'UsdPrimvarReader_float3'
                or reader.GetInput('varname').Get() != 'displayColor'):
            raise ValueError('Original porous skin must use unindexed vertex displayColor')
        uv = np.floor(points[:, :2]/.024*128).astype(np.int64) % 128
        changed = (values.astype(float)*pattern[uv[:, 1], uv[:, 0]]).astype(np.float32)
        colors.Set(Vt.Vec3fArray.FromNumpy(changed))
        if np.array_equal(values, changed):
            raise ValueError('Skin albedo variation did not affect visible colours')
        receipts.append({'prim_path': path, 'before_sha256': canonical_hash(values.tolist()),
                         'after_sha256': canonical_hash(changed.tolist())})
    return {'schema': 'kitchen_sponge_vertex_albedo.v1', 'seed': seed,
        'texture_path': str(texture), 'texture_sha256': file_hash(texture),
        'visual_mesh_paths': paths, 'original_shader_networks_preserved': True,
        'only_albedo_changed': True, 'rest_geometry_and_binding_unchanged': True,
        'runtime_skin_copies_vertex_colours': True, 'vertex_colour_receipts': receipts}


def _top_point(stage, body, collider, *, side=None):
    """Choose an actual collider vertex; never infer a grasp from the AABB."""
    from pxr import UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(collider)
    if (not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
        raise ValueError('Semantic surface requires its original enabled Mesh collider')
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetPrimAtPath(body))
    m = np.asarray(matrix).T
    points = points @ m[:3, :3].T + m[:3, 3]
    ids = np.flatnonzero(np.abs(points[:, 2]-points[:, 2].max()) < 1e-7)
    if side is None:
        selected = ids[np.argmin(np.linalg.norm(points[ids, :2]-points[ids, :2].mean(0), axis=1))]
    else:
        axis, sign = side
        selected = ids[np.argmax(sign*points[ids, axis])]
    return points[selected].tolist(), {'collider_path': collider, 'vertex_index': int(selected),
        'method': 'actual_mesh_vertex_transformed_to_body', 'collision_approximation': str(
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())}


def interaction_targets(stage, metadata):
    rows = {r['id']: r for r in metadata['objects']}
    result = []

    def pickup(name, collider, label, *, group=None, side=None):
        row = rows[name]
        point, witness = _top_point(stage, row['body_path'], row['body_path']+'/Collision/'+collider, side=side)
        result.append({'label': label, 'body_id': name, 'prim_path': row['body_path'],
            'local_position_m': point, 'approach_direction_local': [0., 0., 1.],
            'approach_distance_m': .06, 'type': 'actual_surface', 'reference_state': 'initial',
            'operation_group': group or name+'_pickup', 'required': True, 'approach': 'above',
            'surface_witness': witness, 'scope': 'Actual semantic material surface; not a grasp recipe or contact-success claim'})

    def goal(name, world, group):
        result.append({'label': name, 'prim_path': ROOT, 'local_position_m': list(world),
            'approach_direction_local': [0., 0., 1.], 'approach_distance_m': .06,
            'type': 'goal', 'reference_state': 'initial', 'operation_group': group,
            'required': True, 'approach': 'above',
            'scope': 'Original task reference only; final support/manipulation requires native evidence'})

    task_id = metadata['task_id']
    if task_id == 'breakfast_serving':
        pickup('tray', 'handle', 'tray_handle_upper_surface')
        for sign, label in ((-1, 'near'), (1, 'far')):
            pickup('plate', 'shell', 'plate_'+label+'_rim_surface', side=(1, sign))
        pickup('cup', 'handle', 'cup_handle_upper_surface')
        pickup('cup', 'shell', 'cup_rim_upper_surface', side=(1, -1))
        for name in ('bread_0', 'bread_1'):
            pickup(name, 'crumb_body', name+'_upper_surface')
        for name, pose in metadata['goals']['body_poses_xyzw'].items():
            goal(name+'_destination', pose[:3], name+'_placement')
    elif task_id == 'pot_lid_placement':
        for side in ('left', 'right'):
            pickup('pot', 'handle_'+side, 'pot_'+side+'_handle_surface')
        pickup('lid', 'knob', 'lid_knob_upper_surface')
        for name, pose in metadata['goals']['body_poses_xyzw'].items():
            goal(name+'_destination', pose[:3], name+'_placement')
    elif task_id == 'surface_wiping':
        # These are the three exposed outer top material cells of one logical
        # sponge. A required group may use any one of their real top surfaces.
        for x in (0, 1, 2):
            name = f'sponge_cell_{x}_1_2'
            pickup(name, 'cell', f'sponge_top_surface_{x}', group='sponge_pickup')
        goal('sponge_return', metadata['goals']['body_poses_xyzw']['sponge'][:3], 'sponge_return')
        cells = {c['id']: c for c in metadata['goals']['soil_cells']}
        for patch in metadata['goals']['soil_patches']:
            points = np.asarray([cells[name]['center_world_m'] for name in patch['cell_ids']])
            p = points[np.argmin(np.linalg.norm(points-points.mean(0), axis=1))]
            goal(patch['id']+'_work_surface', p.tolist(), patch['id']+'_wiping')
    else:
        raise ValueError('Unsupported kitchen semantic surface inventory')
    return result


def derive_stage(source, parent_metadata, plan):
    """CPU-only reference stage, no running environment or post-reset writes."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    source_layer = source.GetRootLayer()
    if source_layer.anonymous:
        # Tests may supply an anonymous actual-parent stage. Do not silently
        # export a reference to an anonymous layer in a published artifact.
        stage = Usd.Stage.Open(source.Flatten())
    else:
        UsdGeom.Xform.Define(stage, ROOT).GetPrim().GetReferences().AddReference(source_layer.realPath, ROOT)
    metadata = _metadata(parent_metadata, plan)
    actors = parent_metadata['actors']
    moving = set(actors.values()) | set(parent_metadata['fixtures'].values())
    actual_actors = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if actual_actors != set(actors.values()):
        raise ValueError('Original native rigid-body inventory differs')
    before = protected_properties(stage, moving)
    _assert_internal_joints(stage, [ROOT], set(actors.values()))
    poses = {}
    for path in sorted(moving):
        body = stage.GetPrimAtPath(path)
        original = _world_pose(stage, path)
        operations = UsdGeom.Xformable(body).GetOrderedXformOps()
        if [o.GetOpName() for o in operations] != ['xformOp:translate', 'xformOp:orient']:
            raise ValueError('Expected original translate/orient body or support ops')
        if plan['seed']:
            value = operations[0].Get()
            operations[0].Set(type(value)(value[0]+plan['translation_xy_m'][0], value[1], value[2]))
        actual = _world_pose(stage, path)
        wanted = original['position_m'].copy()
        wanted[0] += plan['translation_xy_m'][0]
        if not np.allclose(actual['position_m'], wanted, atol=1e-14, rtol=0) or actual['quaternion_wxyz'] != original['quaternion_wxyz']:
            raise ValueError('Kitchen reset changed non-X pose components')
        poses[path] = actual
    after = protected_properties(stage, moving)
    if before != after:
        raise ValueError('Kitchen reset changed protected physical or geometry properties')
    bodies = []
    for row in metadata['objects']:
        path = row['body_path']
        points = _usd_points(stage, path)
        mass = float(UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get())
        if not np.isclose(mass, row['mass_kg'], atol=1e-8, rtol=1e-7):
            raise ValueError('Original actual MassAPI disagrees with task metadata')
        bodies.append({'object_id': row['id'], 'local_bounds_m': [points.min(0).tolist(), points.max(0).tolist()],
            'nominal_mass_kg': row['mass_kg']})
    # Room worktops are admitted independently from the composed room and its
    # native filters. This explicit list owns only task fixture colliders.
    supports = []
    for row in metadata['fixture_metadata'].values():
        supports.extend(row['collider_paths'])
    return stage, {'plan': deepcopy(plan), 'asset_metadata': metadata,
        'actual_body_poses_m': {name: poses[path] for name, path in actors.items()},
        'static_geometry': {'bodies': bodies}, 'interaction_targets': interaction_targets(stage, metadata),
        'fixed_support_paths': sorted(set(supports)),
        'protected_physics_before_sha256': canonical_hash(before),
        'protected_physics_after_sha256': canonical_hash(after), 'appearance': []}


def author_asset(parent, scene, task, candidate, references, plan, output):
    from pxr import Usd
    proof = _author_source_identity()
    if output.exists():
        raise FileExistsError('Never overwrite a kitchen placement asset: '+str(output))
    source = Usd.Stage.Open(candidate['usd_path'])
    if not source:
        raise ValueError('Missing original kitchen USD')
    stage, data = derive_stage(source, task['asset_metadata'], plan)
    output.mkdir(parents=True)
    data['bundle_usd'] = {'path': candidate['usd_path'], 'sha256': candidate['usd_sha256']}
    data['bundle_dependencies'] = {}
    if plan['seed']:
        texture = finish_texture(output, plan['seed'])
        # Only task-actor albedo changes. Soil/context/fixture finishes remain.
        paths = [path+'/Visual' for path in task['asset_metadata']['actors'].values()]
        data['appearance'].append(_surface_texture(stage, paths, texture, plan['seed'], material_root=ROOT+'/PlacementLooks'))
        if task['id'] == 'surface_wiping':
            data['appearance'].append(_sponge_skin_albedo(stage, task['asset_metadata'], texture, plan['seed']))
        moving = set(task['asset_metadata']['actors'].values()) | set(task['asset_metadata']['fixtures'].values())
        after = canonical_hash(protected_properties(stage, moving))
        if after != data['protected_physics_before_sha256']:
            raise ValueError('Kitchen texture changed geometry, joints or physical material')
        data['protected_physics_after_sha256'] = after
        usd = output/'task.usda'
        stage.GetRootLayer().Export(str(usd))
        data['bundle_usd'] = {'path': str(usd), 'sha256': file_hash(usd)}
        data['bundle_dependencies'] = {candidate['usd_path']: candidate['usd_sha256'],
            **{str(p): file_hash(p) for p in output.rglob('*') if p.is_file()}}
    metadata = output/'metadata.json'
    write_json(metadata, data)
    (output/'ATTRIBUTION.txt').write_text('Original kitchen geometries, textures and source licenses remain at their hash-pinned paths.\nNew light periodic albedo texture: CC0-1.0. No physical material changes.\n')
    manifest = {'schema': 'workroom_seed_asset.v1', 'family': 'kitchen', 'task_id': task['id'],
        'seed': plan['seed'], 'generator_version': plan['generator_version'], 'variant_scope': VARIANT_SCOPE,
        'parent_references': references, 'author_source': proof,
        'dependencies_sha256': {str(p): file_hash(p) for p in output.rglob('*') if p.is_file()},
        'metadata': str(metadata), 'metadata_sha256': file_hash(metadata),
        **{k: deepcopy(data[k]) for k in ('actual_body_poses_m', 'interaction_targets', 'static_geometry', 'fixed_support_paths')},
        'license': deepcopy(task['asset_metadata']['source']), 'native_qualified': False,
        'runtime_object_writes': False}
    path = output/'seed_manifest.json'
    write_json(path, manifest)
    return path
