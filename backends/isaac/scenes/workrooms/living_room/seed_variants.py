"""Source-bound, reset-only derivatives of three ordinary living-room tasks.

Old USD files, fixed rooms, robots, controllers and tower tasks are never edited.
Seed zero copies the canonical USD byte for byte. Other seeds bake bounded real
geometry and matching mass priors, then rigidly move coherent reset groups.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from . import seed_geometry as geometry
from bench.paths import DATA_ROOT, load_json, portable

TASKS = ('remote_operation', 'water_pouring', 'storage_box')
SCHEMA = 'living_room_seeded_asset_variant.v1'
ROOT = '/World/LivingTask'
TEXTURE_ROOT = Path(f'{DATA_ROOT}/assets/materials/polyhaven/v1/kitchen_wood')


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def actors(metadata, task):
    if task == 'remote_operation':
        return {'remote': metadata['remote']['body_path'],
                **metadata['remote']['button_paths'], 'holder': metadata['holder']['body_path']}
    if task == 'water_pouring':
        return {name: row['path'] for name, row in metadata['containers'].items()}
    return dict(metadata['actors'])


def _water_handle(stage, metadata, factor):
    kettle = stage.GetPrimAtPath(metadata['containers']['kettle']['path'])
    shell = UsdGeom.Mesh(stage.GetPrimAtPath(str(kettle.GetPath())+'/HollowShell'))
    shell_mass = geometry.mesh_volume_moments(shell)
    original, changed = [shell_mass], [shell_mass]
    radius = None
    for index in range(5):
        capsule = UsdGeom.Capsule(stage.GetPrimAtPath(str(kettle.GetPath())+f'/Handle_{index:02d}'))
        old_radius, height = float(capsule.GetRadiusAttr().Get()), float(capsule.GetHeightAttr().Get())
        matrix = UsdGeom.Xformable(capsule).GetLocalTransformation()
        r, center = np.asarray(matrix)[:3, :3].T, np.array(matrix.ExtractTranslation())
        for target, current_radius in ((original, old_radius), (changed, old_radius*factor)):
            volume, _, inertia = geometry.capsule_volume_moments(current_radius, height)
            target.append((volume, center, r@inertia@r.T))
        radius = old_radius*factor
        capsule.GetRadiusAttr().Set(radius)
    baseline_volume, _, _ = geometry.compound_moments(original)
    volume, com, inertia = geometry.compound_moments(changed)
    density = metadata['containers']['kettle']['dry_mass_kg']/baseline_volume
    geometry.set_mass(kettle, volume*density, com, inertia*density)
    metadata['containers']['kettle']['dry_mass_kg'] = volume*density
    # Keep the original decorative inset on the handle's outer surface.
    inset = stage.GetPrimAtPath(str(kettle.GetPath())+'/GripInset').GetAttribute('xformOp:translate')
    center = list(inset.Get()); center[0] = -.098-radius+.0005
    inset.Set(Gf.Vec3d(*center))
    return {'capsule_radius_m': radius, 'capsule_count': 5, 'radius_factor': factor,
            'centerlines_unchanged': True, 'vessel_and_water_geometry_unchanged': True,
            'mass_policy': 'Uniform effective density calibrated to original dry mass; additive closed shell and five solid capsules, overlap counted per component',
            'effective_density_kg_m3': density, 'baseline_additive_volume_m3': baseline_volume,
            'variant_additive_volume_m3': volume,
            'inertia_scope': 'Exact moments of declared source-mesh/additive-capsule prior; not measured or native-qualified'}


def apply_geometry(stage, metadata, task, variant):
    """Mutate only a caller-owned anonymous stage; return geometry metadata."""
    if task not in TASKS or variant not in ('canonical', 'narrow', 'wide'):
        raise ValueError('Unsupported ordinary task/geometry variant')
    scale = 1. if variant == 'canonical' else (.94 if variant == 'narrow' else 1.06)
    inventory = actors(metadata, task)
    report = {'schema': 'living_'+('remote' if task == 'remote_operation' else task)+'_geometry_variant.v1',
              'variant': variant, 'render_collision_geometry_aligned': True,
              'native_qualified': False}
    if task == 'remote_operation':
        if variant != 'canonical':
            geometry.scale_tree(stage, metadata['remote']['assembly_path'], [scale, 1, 1])
            geometry.scale_tree(stage, inventory['holder'], [scale, 1, 1])
            geometry.scale_joint_anchors(stage, inventory.values(), [scale, 1, 1])
        metadata['remote']['shell_size_xyz_m'][0] *= scale
        metadata['remote']['envelope_xyz_m'][0] *= scale
        metadata['remote']['local_bounds_m'] = [[-.0275*scale, -.08, -.0104], [.0275*scale, .08, .0135]]
        metadata['remote']['grasp_scope'] = 'Declared variable grip width; static geometry does not qualify a robot grasp'
        for key in ('body_mass_kg', 'button_mass_kg', 'total_mass_kg'):
            metadata['remote'][key] *= scale
        metadata['holder']['opening_xy_m'][0] *= scale
        metadata['holder']['fit_clearance_each_side_xy_m'][0] *= scale
        metadata['holder']['mass_kg'] *= scale
        metadata['holder']['liner_local_z_m'] = .007
        metadata['collision']['approximation'] = 'Original per-component convex collision meshes, width baked into their actual vertices; native convex cooking is not yet read back for this derivative'
        report.update(scale_xyz=[scale, 1, 1], scope='Remote body/buttons and holder local width',
                      mass_policy='Density-preserving affine transform of original explicit mass/COM/inertia priors',
                      preserved=['button travel/spring/damping', 'physics material coefficients', 'Z support height'])
    elif task == 'storage_box':
        scale = 1. if variant == 'canonical' else (.95 if variant == 'narrow' else 1.05)
        if variant != 'canonical':
            pivot_x = metadata['initial_positions_m']['box'][0]
            for path in inventory.values():
                prim = stage.GetPrimAtPath(path)
                geometry.scale_tree(stage, path, [scale, 1, 1])
                attribute = prim.GetAttribute('xformOp:translate')
                point = list(attribute.Get()); point[0] = pivot_x+scale*(point[0]-pivot_x)
                attribute.Set(Gf.Vec3d(*point))
            geometry.scale_joint_anchors(stage, inventory.values(), [scale, 1, 1])
        metadata['item_size_m'][0] *= scale
        metadata['masses_kg'] = {key: value*scale for key, value in metadata['masses_kg'].items()}
        report.update(scale_xyz=[scale, 1, 1], scope='Box/lid/catches/item width and coherent joint anchors; tray size fixed',
                      mass_policy='Density-preserving affine transform of original explicit mass/COM/inertia priors',
                      preserved=['hinge axes/limits/damping', 'physics materials', 'tray size', 'vertical support'])
    else:
        factor = 1. if variant == 'canonical' else (.92 if variant == 'narrow' else 1.08)
        metadata['handle_geometry'] = ({'capsule_radius_m': .0065, 'capsule_count': 5,
            'radius_factor': 1., 'centerlines_unchanged': True, 'vessel_and_water_geometry_unchanged': True,
            'mass_policy': 'Original dry mass and native implicit COM/inertia unchanged'} if variant == 'canonical'
            else _water_handle(stage, metadata, factor))
        report.update(scope='Only the five actual handle-collider radii change; vessel cavities/liquid unchanged',
                      radius_factor=factor, mass_policy=metadata['handle_geometry']['mass_policy'],
                      preserved=['vessel SDF meshes/profiles', 'particle inventory/mass/relative reset', 'physical materials'])
    metadata['geometry_variant'] = report
    return report


def apply_appearance(stage, task, variant):
    if variant == 'canonical':
        return {'id': variant, 'changes': [], 'dependencies_sha256': {}}
    if variant not in ('warm', 'cool', 'neutral'):
        raise ValueError('Unknown appearance variant')
    palette = {'warm': (.83, .59, .34), 'cool': (.38, .63, .74), 'neutral': (.69, .68, .62)}[variant]
    names = {'remote_operation': ['Looks/Graphite', 'Looks/Face', 'Looks/Holder'],
             'storage_box': ['Looks/Case', 'Looks/Lid', 'Looks/Tray'],
             'water_pouring': ['Materials/WarmWhite']}[task]
    files = [TEXTURE_ROOT/name for name in ('SOURCE.json', 'kitchen_wood_diff_1k.jpg', 'kitchen_wood_rough_1k.jpg')]
    dependencies = {str(path): file_hash(path) for path in files}
    selected = set(ROOT+'/'+name for name in names)
    for path in selected:
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        shader, _, _ = material.ComputeSurfaceSource()
        reader = UsdShade.Shader.Define(stage, path+'/SeedST')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        for label, filename, output, target in (
                ('Albedo', 'kitchen_wood_diff_1k.jpg', 'rgb', 'diffuseColor'),
                ('Roughness', 'kitchen_wood_rough_1k.jpg', 'r', 'roughness')):
            texture = UsdShade.Shader.Define(stage, path+'/Seed'+label)
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(TEXTURE_ROOT/filename)))
            texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if output == 'rgb' else 'raw')
            texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            if output == 'rgb':
                texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*palette, 1.))
            texture.CreateOutput(output, Sdf.ValueTypeNames.Float3 if output == 'rgb' else Sdf.ValueTypeNames.Float)
            shader.CreateInput(target, Sdf.ValueTypeNames.Color3f if output == 'rgb' else Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), output)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not bound or str(bound.GetPath()) not in selected:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = mesh.GetFaceVertexCountsAttr().Get()
        uv, start = [], 0
        for count in counts:
            face = points[indices[start:start+count]]; start += count
            normal = np.cross(face[1]-face[0], face[2]-face[0])
            axes = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
            uv.extend((face[:, axes]/.08).tolist())
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
    return {'id': variant, 'changes': sorted(selected), 'texture_source': str(TEXTURE_ROOT/'SOURCE.json'),
            'render_only': True, 'physical_materials_unchanged': True, 'dependencies_sha256': dependencies}


def _feature(label, body_id, path, position, direction=(0, 0, 1), *, kind='actual_surface', **extra):
    return {'label': label, 'body_id': body_id, 'prim_path': path,
            'local_position_m': list(position), 'approach_direction_local': list(direction),
            'approach_distance_m': .08, 'direction_convention': 'outward toward preapproach',
            'type': kind, 'reference_state': 'initial',
            'operation_group': label, 'required': True,
            'scope': 'geometric_reference_not_grasp_center', **extra}


def interaction_features(stage, metadata, task):
    a = actors(metadata, task)
    if task == 'remote_operation':
        width = metadata['remote']['shell_size_xyz_m'][0]
        return [_feature('remote_side_'+str(sign), 'remote', a['remote'], [sign*width/2, -.03, -.002], [sign, 0, 0], operation_group='remote_grasp') for sign in (-1, 1)] + [
            _feature('power_cap', 'power', a['power'], [0, 0, .002]),
            _feature('confirm_cap', 'confirm', a['confirm'], [0, 0, .002], required=False),
            _feature('holder_opening', 'holder', a['holder'], [0, 0, .007], kind='goal')]
    if task == 'water_pouring':
        radius = metadata['handle_geometry']['capsule_radius_m']
        return [_feature('kettle_grip_'+str(sign), 'kettle', a['kettle'], [-.098, sign*radius, .054], [0, sign, 0], operation_group='kettle_grasp') for sign in (-1, 1)] + [
            _feature('kettle_spout', 'kettle', a['kettle'], [.073, 0, .102], kind='pouring_edge', required=False),
            _feature('cup_opening', 'cup', a['cup'], [0, 0, .067], kind='goal')]
    # The item surface is a conditional target: closed lid obstruction is real.
    return [_feature('lid_handle', 'lid', a['lid'], [0, -.215, .033]),
            *[_feature(name+'_grip', name, a[name], [0, .04, .030]) for name in ('left_catch', 'right_catch')],
            _feature('item_top', 'item', a['item'], [0, 0, .018], requires_open=True, reference_state='opened'),
            _feature('collection_tray', 'collection_tray', ROOT+'/CollectionTray', [.94, -.12, .786], kind='goal')]


def _sync(stage, metadata, task, plan):
    inventory = actors(metadata, task)
    poses = {name: geometry.pose(stage.GetPrimAtPath(path)) for name, path in inventory.items()}
    masses = geometry.mass_inventory(stage, inventory)
    metadata['mass_properties'] = masses
    metadata['static_body_geometry'] = [{'object_id': name,
        'local_bounds_m': geometry.collision_bounds(stage, path),
        'nominal_mass_kg': masses[name]['mass_kg'],
        'scope': 'Authored collider envelope and simulation mass prior; not native readback'}
        for name, path in inventory.items()]
    if task == 'remote_operation':
        metadata['remote']['position_m'] = geometry.pose(stage.GetPrimAtPath(metadata['remote']['assembly_path']))['position_m']
        metadata['holder']['position_m'] = poses['holder']['position_m']
        metadata['holder']['initial_quaternion_wxyz'] = poses['holder']['quaternion_wxyz']
    elif task == 'water_pouring':
        for name in ('kettle', 'cup'):
            row = metadata['containers'][name]
            row['initial_position_m'] = poses[name]['position_m']
            q = poses[name]['quaternion_wxyz']
            row['initial_yaw_rad'] = 2*math.atan2(q[3], q[0])
            row['dry_mass_kg'] = masses[name]['mass_kg']
    else:
        metadata['initial_positions_m'] = {name: row['position_m'] for name, row in poses.items()}
        metadata['masses_kg'] = {name: row['mass_kg'] for name, row in masses.items()}
        metadata['legacy_interaction_targets'] = metadata.pop('interaction_targets', [])
        delta = plan['group_transforms']['tray']['translation_xy_m']
        metadata['tray_inside_xy_m'] = [[p[i]+delta[i] for i in range(2)] for p in metadata['tray_inside_xy_m']]
    metadata['interaction_targets'] = interaction_features(stage, metadata, task)
    metadata['seed_variant'] = {'seed': plan['seed'], 'case_plan_sha256': plan['plan_sha256'],
        'initialization_only': True, 'native_validation_performed': False}
    return poses


def _json_value(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Sdf.AssetPath):
        return value.path
    if isinstance(value, (Gf.Quatf, Gf.Quatd)):
        return [value.GetReal(), *value.GetImaginary()]
    return [_json_value(v) for v in value]


def physics_inventory(stage):
    """Authored physical identity, including collision geometry and rigid poses."""
    result = {}
    for prim in stage.Traverse():
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        row = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if attr.HasAuthoredValue() and (name.startswith(('physics:', 'physx', 'xformOp')) or
                ((collision or prim.IsA(UsdGeom.Points)) and name in (
                    'points', 'velocities', 'ids', 'widths', 'faceVertexCounts', 'faceVertexIndices', 'radius', 'height', 'axis', 'size'))):
                row[name] = _json_value(attr.Get())
        for rel in prim.GetRelationships():
            if rel.GetName().startswith(('physics:', 'material:binding:physics')):
                row[rel.GetName()] = [str(p) for p in rel.GetTargets()]
        if row:
            result[str(prim.GetPath())] = row
    return result


def derive_stage(parent_stage, original_metadata, case_plan, *, appearance_artifact=None):
    """CPU-only stage derivation, also usable by focused anonymous-stage tests."""
    task = case_plan['task_id']
    if task not in TASKS:
        raise ValueError('This author does not randomize towers or unknown tasks')
    if canonical_hash({k: v for k, v in case_plan.items() if k != 'plan_sha256'}) != case_plan['plan_sha256']:
        raise ValueError('Case plan hash differs')
    stage = Usd.Stage.Open(parent_stage.Flatten())
    metadata = deepcopy(original_metadata)
    before = physics_inventory(stage)
    apply_geometry(stage, metadata, task, case_plan['geometry_variant'])
    if case_plan.get('generator_version') == 'parameterized_physics_v1':
        from .randomization import parameterized
        visual = parameterized.apply_appearance(stage, task, case_plan['parameters'], appearance_artifact)
        parameterized.apply_physics(stage, metadata, task, case_plan['parameters'])
        if case_plan['seed']:
            report = metadata['geometry_variant']
            report['preserved'] = [item for item in report['preserved'] if 'material' not in item]
            report['following_material_step'] = 'Explicit mass/density/friction instance applied after the geometry transform; see parameterized_physics.'
    else:
        visual = apply_appearance(stage, task, case_plan['appearance_variant'])
    if case_plan['seed'] != 0:
        for group in case_plan['group_transforms'].values():
            geometry.transform_group(stage, group)
    elif (case_plan['geometry_variant'] != 'canonical' or case_plan['appearance_variant'] != 'canonical'
          or any(group['yaw_rad'] or any(group['translation_xy_m']) for group in case_plan['group_transforms'].values())):
        raise ValueError('Seed zero must retain canonical geometry and poses')
    poses = _sync(stage, metadata, task, case_plan)
    after = physics_inventory(stage)
    difference = sorted(key for key in set(before)|set(after) if before.get(key) != after.get(key))
    return stage, metadata, poses, visual, {'parent_physics_sha256': canonical_hash(before),
        'variant_physics_sha256': canonical_hash(after), 'changed_prim_paths': difference,
        'scope': 'Authoring inventory only; shape/mass/reset changes explicit, native behavior unqualified',
        'controller_or_robot_changes': False, 'fixed_room_changes': False}


def author_variant(parent_asset_reference, case_plan, output):
    """Build a fresh asset directory from a complete frozen source snapshot.

    Reference keys are manifest_path/manifest_sha256, initial_usd/initial_usd_sha256,
    metadata/metadata_sha256. Every parent input is checked before any output.
    """
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from .randomization.cases import validate_plan
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__, geometry.__name__, validate_plan.__module__),
        entrypoint_file=Path(__file__))
    if not source or not source.get('verified'):
        raise ValueError('Publishing asset derivatives requires a complete frozen source snapshot')
    reference = deepcopy(parent_asset_reference)
    for name, hash_name in (('manifest_path', 'manifest_sha256'), ('initial_usd', 'initial_usd_sha256'),
                            ('metadata', 'metadata_sha256')):
        if file_hash(reference[name]) != reference[hash_name]:
            raise ValueError('Parent identity differs: '+name)
    parent_manifest = load_json(Path(reference['manifest_path']))
    if parent_manifest['task_id'] != case_plan['task_id'] or parent_manifest['initial_usd_sha256'] != reference['initial_usd_sha256'] or parent_manifest['metadata_sha256'] != reference['metadata_sha256']:
        raise ValueError('Parent manifest input binding differs')
    metadata = load_json(Path(reference['metadata']))
    validate_plan(case_plan, metadata)
    parent_dependencies = {}
    for path, description in parent_manifest.get('dependencies', {}).items():
        expected = description if isinstance(description, str) else description['sha256']
        if file_hash(path) != expected:
            raise ValueError('Parent dependency identity differs: '+path)
        parent_dependencies[path] = expected
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    artifact = None
    if case_plan.get('generator_version') == 'parameterized_physics_v1' and case_plan['seed']:
        from .randomization.parameterized import appearance_assets
        artifact = appearance_assets(output/'appearance', case_plan['parameters'])
    stage = Usd.Stage.Open(reference['initial_usd'])
    stage, metadata, poses, visual, identity = derive_stage(stage, metadata, case_plan,
                                                          appearance_artifact=artifact)
    dependencies = {**parent_dependencies, **visual['dependencies_sha256']}
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and attr.HasAuthoredValue():
                value = attr.Get()
                if value.path:
                    path = Path(value.resolvedPath or value.path)
                    if not path.is_absolute():
                        path = Path(reference['initial_usd']).parent/path
                    dependencies[str(path.resolve())] = file_hash(path)
    output.mkdir(parents=True, exist_ok=artifact is not None)
    usd_path, metadata_path = output/'initial.usda', output/'metadata.json'
    if case_plan['seed'] == 0:
        shutil.copyfile(reference['initial_usd'], usd_path)
    else:
        stage.GetRootLayer().Export(str(usd_path))
    _write(metadata_path, metadata)
    manifest = {'schema': SCHEMA, 'task_id': case_plan['task_id'], 'seed': case_plan['seed'],
        'initial_usd': str(usd_path), 'initial_usd_sha256': file_hash(usd_path),
        'metadata': str(metadata_path), 'metadata_sha256': file_hash(metadata_path),
        'dependencies_sha256': dependencies, 'case_plan': case_plan,
        'case_plan_sha256': case_plan['plan_sha256'], 'actual_body_poses_m': poses,
        'interaction_features': metadata['interaction_targets'], 'source_physics_identity': identity,
        'visual_variant': visual, 'parent_asset_reference': reference, 'source_snapshot': source,
        'geometry_variant': metadata['geometry_variant'], 'native_qualified': False,
        'physics_accepted': False, 'robot_manipulation_qualified': False}
    if case_plan.get('generator_version') == 'parameterized_physics_v1':
        manifest['parameterized_physics'] = {'parameters': deepcopy(case_plan['parameters']),
            'applied': deepcopy(metadata.get('parameterized_physics', {'canonical': True, 'applied': False})),
            'source_files_sha256': {str(path): file_hash(path) for path in (
                Path(__file__).parent/'randomization/parameterized.py',
                Path(__file__).parents[1]/'parameterized/spec.py',
                Path(__file__).parents[1]/'parameterized/appearance.py')}}
    _write(output/'manifest.json', manifest)
    return manifest
