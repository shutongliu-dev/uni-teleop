"""Immutable reset-only derivatives of three ordinary study tasks.

These CPU authoring functions never run a controller or mutate a live episode.
Sizes, colliders and explicit mass priors change together. Seed zero preserves
the original asset bytes and evaluation configuration. Qualification is external.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from . import rules
from bench.paths import DATA_ROOT, load_json, relocated_layer_bytes

TASKS = ('stationery_sorting', 'book_stacking', 'book_shelving')
GENERATOR_VERSION = 'study_seed_variants_v1'
SCHEMA = 'workroom_seed_asset.v1'
ROOT = '/World/StudyTask'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/seed_randomization/v1/study')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    # The frozen study referee emits target instructions in insertion order.
    # Hashes use sorted canonical JSON, but serialized inputs retain that order.
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _checked(path, expected):
    path = Path(path).resolve(strict=True)
    if file_hash(path) != expected:
        raise ValueError('Frozen input hash differs: '+str(path))
    return path


def _seed(task_id, seed):
    if task_id not in TASKS or type(seed) is not int or seed not in range(10):
        raise ValueError('Study variants require a supported task and integer seed 0 through 9')


def plan_case(task_id, seed, configuration):
    """Stable SHA-derived perturbations, independent of Python/RNG versions."""
    _seed(task_id, seed)
    rules.validate_configuration(configuration)
    if configuration['task_id'] != task_id:
        raise ValueError('Parent task differs')
    common_scale = (1., .98, .99, 1.01, 1.02)[0 if seed == 0 else 1+(seed-1)%4]
    rows = {}
    for name in sorted(configuration['objects']):
        raw = hashlib.sha256(f'{GENERATOR_VERSION}/{task_id}/{seed}/{name}'.encode()).digest()
        unit = [int.from_bytes(raw[4*i:4*i+4], 'big')/(2**32-1)*2-1 for i in range(3)]
        scale = (1. if seed == 0 else 1.+.01*(1+(seed-1)%4)) if name == 'pencil' else common_scale
        rows[name] = {'uniform_scale': scale,
                      'translation_xy_m': [round(.012*x, 9) if seed else 0. for x in unit[:2]],
                      'yaw_rad': round(math.radians(6)*unit[2], 12) if seed else 0.}
    plan = {'schema': 'study_seed_plan.v1', 'generator_version': GENERATOR_VERSION,
            'task_id': task_id, 'seed': seed, 'objects': rows,
            'parent_configuration_sha256': configuration['configuration_sha256']}
    plan['plan_sha256'] = rules.canonical_hash(plan)
    return plan


def _scale_body(stage, path, scale):
    """Bake a uniform physical scale into each descendant's actual geometry.

Uniform scaling commutes with existing rotations and diagonal transforms.
Body/root pose remains rigid; no importer-dependent scaled rigid body is used.
"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    from scenes.workrooms.living_room import seed_geometry as geometry
    root = stage.GetPrimAtPath(path)
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
            raise ValueError('Study prop unexpectedly contains an articulation')
        xf = UsdGeom.Xformable(prim)
        if xf:
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    v = op.Get(); op.Set(type(v)(*(np.array(v)*scale)))
                elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    matrix = Gf.Matrix4d(op.Get())
                    matrix.SetTranslateOnly(matrix.ExtractTranslation()*scale)
                    op.Set(matrix)
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)*scale
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        elif prim.IsA(UsdGeom.Cube):
            shape = UsdGeom.Cube(prim); shape.GetSizeAttr().Set(shape.GetSizeAttr().Get()*scale)
            half = float(shape.GetSizeAttr().Get())/2
            shape.CreateExtentAttr([Gf.Vec3f(-half), Gf.Vec3f(half)])
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            shape.GetRadiusAttr().Set(shape.GetRadiusAttr().Get()*scale)
            shape.GetHeightAttr().Set(shape.GetHeightAttr().Get()*scale)
            half = np.full(3, float(shape.GetRadiusAttr().Get()))
            half[{'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get())]] = float(shape.GetHeightAttr().Get())/2
            shape.CreateExtentAttr([Gf.Vec3f(*-half), Gf.Vec3f(*half)])
        elif prim.IsA(UsdGeom.Gprim):
            raise ValueError('Unsupported study geometry: '+str(prim.GetPath()))
    geometry.scale_mass(root, [scale]*3)


def collision_bounds(stage, body_path):
    """Body-local envelope of the actual enabled mesh/cube/cylinder colliders."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    body = stage.GetPrimAtPath(body_path)
    cache = UsdGeom.XformCache(); inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    bounds = []
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)*inverse
        if prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            axis = {'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get())]
            linear = np.array(matrix)[:3, :3].T
            extent = (float(shape.GetRadiusAttr().Get())*np.linalg.norm(np.delete(linear, axis, axis=1), axis=1)
                      + float(shape.GetHeightAttr().Get())/2*abs(linear[:, axis]))
            center = np.array(matrix.ExtractTranslation()); bounds.extend([center-extent, center+extent])
            continue
        if prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            points = product((-half, half), repeat=3)
        elif prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        else:
            raise ValueError('Unsupported enabled study collider')
        bounds.extend(np.asarray(matrix.Transform(Gf.Vec3d(*p))) for p in points)
    if not bounds:
        raise ValueError('Study body has no enabled collision geometry')
    points = np.asarray(bounds)
    return [points.min(0).tolist(), points.max(0).tolist()]


def _configuration(parent, plan):
    c = deepcopy(parent)
    if plan['seed'] == 0:
        return c
    for name, row in c['objects'].items():
        spec = plan['objects'][name]; scale = spec['uniform_scale']
        row['local_bounds_m'] = (np.array(row['local_bounds_m'])*scale).tolist()
        pose = row['initial_pose']; q = pose['quaternion_wxyz']
        # Same +Z heading quaternion/product as Gf, without loading USD to
        # validate a frozen JSON case before SimulationApp has started.
        half = math.radians(math.degrees(spec['yaw_rad']))/2
        a, b = math.cos(half), math.sin(half)
        # Gf.Rotation.GetQuat normalizes the constructed quaternion with sqrt.
        length = math.sqrt(a*a+b*b)
        inverse_length = 1./length
        a, b = a/length, b*inverse_length
        w, x, y, z = q
        pose['quaternion_wxyz'] = [a*w-(0.*x+0.*y+b*z),
            a*x+0.*w+(0.*z-b*y), a*y+0.*w+(b*x-0.*z),
            a*z+b*w+(0.*y-0.*x)]
        pose['position_m'][:2] = [x+d for x, d in zip(pose['position_m'][:2], spec['translation_xy_m'])]
        r = np.asarray(rules.rotation(pose['quaternion_wxyz']))
        corners = np.array(list(product(*zip(*row['local_bounds_m']))))
        pose['position_m'][2] = c['supports']['table']['bounds_m'][1][2]-float((corners@r.T)[:, 2].min())
    if c['task_id'] == 'book_stacking':
        height = c['supports']['table']['bounds_m'][1][2]
        for name in c['stack_order']:
            target = c['targets'][name]
            old_z = parent['targets'][name]['support_z_m']
            old_height = parent['objects'][name]['local_bounds_m'][1][1]-parent['objects'][name]['local_bounds_m'][0][1]
            thickness = c['objects'][name]['local_bounds_m'][1][1]-c['objects'][name]['local_bounds_m'][0][1]
            target['bounds_m'][0][2] += height-old_z
            target['bounds_m'][1][2] += height-old_z+thickness-old_height
            target['support_z_m'] = height; height += thickness
    # Preserve identity, category, thresholds, budget, slot axes and support graph.
    c['rules'] = rules.public_rules(c['task_id'], c)
    c['configuration_sha256'] = rules.canonical_hash({k: v for k, v in c.items() if k != 'configuration_sha256'})
    return rules.validate_configuration(c)


def _features(configuration, bounds):
    rows = []
    for name, obj in configuration['objects'].items():
        path = ROOT+'/Objects/'+name
        lo, hi = np.array(bounds[name]); center = (lo+hi)/2
        axis = 2 if configuration['task_id'] == 'stationery_sorting' else 1
        for sign in (-1, 1):
            p = center.copy(); p[axis] = (lo if sign < 0 else hi)[axis]
            direction = np.zeros(3); direction[axis] = sign
            rows.append({'label': name+'_surface_'+str(sign), 'prim_path': path,
                'body_id': name, 'local_position_m': p.tolist(), 'approach_direction_local': direction.tolist(),
                'approach_distance_m': .08, 'type': 'actual_surface', 'reference_state': 'initial',
                'operation_group': name+'_grasp', 'required': True,
                'scope': 'geometric_reference_not_grasp_center'})
        target = configuration['targets'][name]
        center = np.mean(target['bounds_m'], axis=0); center[2] = target['support_z_m']
        rows.append({'label': name+'_goal', 'prim_path': ROOT, 'local_position_m': center.tolist(),
            'approach_direction_local': [1., 0., 0.] if configuration['task_id'] == 'book_shelving' else [0., 0., 1.],
            'approach_distance_m': .08, 'type': 'goal', 'reference_state': 'initial',
            'operation_group': name+'_placement', 'required': True,
            'scope': 'geometric_reference_not_grasp_center'})
    return rows


def derive_stage(parent_stage, parent_metadata, seed):
    """Pure in-memory stage derivation for CPU checks; no output is published."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.living_room import seed_geometry as geometry
    task_id = parent_metadata['task_id']; _seed(task_id, seed)
    plan = plan_case(task_id, seed, parent_metadata['configuration'])
    stage = Usd.Stage.Open(parent_stage.Flatten())
    metadata = deepcopy(parent_metadata)
    config = _configuration(parent_metadata['configuration'], plan)
    actors = {name: ROOT+'/Objects/'+name for name in config['objects']}
    if {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)} != set(actors.values()):
        raise ValueError('Study dynamic inventory differs from the frozen configuration')
    for name, path in actors.items():
        if seed:
            _scale_body(stage, path, plan['objects'][name]['uniform_scale'])
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)); xf.ClearXformOpOrder()
            pose = config['objects'][name]['initial_pose']; q = pose['quaternion_wxyz']
            xf.AddTranslateOp(opSuffix='seedReset').Set(Gf.Vec3d(*pose['position_m']))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, opSuffix='seedReset').Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    masses = geometry.mass_inventory(stage, actors)
    bounds = {name: collision_bounds(stage, path) for name, path in actors.items()}
    poses = {name: geometry.pose(stage.GetPrimAtPath(path)) for name, path in actors.items()}
    for name, row in metadata['objects'].items():
        scale = plan['objects'][name]['uniform_scale']
        row.update(deepcopy(masses[name]), local_bounds_m=deepcopy(config['objects'][name]['local_bounds_m']),
                   collision_bounds_m=bounds[name], initial_pose=deepcopy(config['objects'][name]['initial_pose']))
        if 'size_xyz_m' in row:
            row['size_xyz_m'] = [v*scale for v in row['size_xyz_m']]
            row['mass_components_kg'] = {k: v*scale**3 for k, v in row['mass_components_kg'].items()}
            row.update(position_m=poses[name]['position_m'], quaternion_wxyz=poses[name]['quaternion_wxyz'])
    metadata.update(configuration=config, actors=actors, actual_body_poses_m=poses,
        mass_properties=masses, static_geometry={'bodies': [{'object_id': name,
            'local_bounds_m': bounds[name], 'nominal_mass_kg': masses[name]['mass_kg']} for name in actors]},
        interaction_targets=_features(config, bounds), seed_plan=plan,
        geometry_variant={'schema': 'study_uniform_geometry_variant.v1', 'scale_by_object': plan['objects'],
            'mass_policy': 'Density-preserving uniform transform of original explicit engineering mass/COM/inertia prior',
            'physical_materials_unchanged': True, 'native_qualified': False},
        physics_accepted=False, robot_policy_success_claimed=False)
    return stage, metadata


def _appearance(stage, seed, output):
    """Visible finish only: new roughness texture; existing labels/colors stay."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    if not seed:
        return {'id': 'canonical', 'dependencies_sha256': {}}
    yy, xx = np.indices((64, 64))
    pixels = (170+((xx*17+yy*29+seed*23+(xx*yy)%31)%35)).astype(np.uint8)
    path = output/'finish_roughness.png'; Image.fromarray(pixels).save(path)
    materials = {}
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(ROOT+'/Objects'))):
        if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                materials[str(material.GetPath())] = material
    changed = []
    for path_id, material in materials.items():
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            continue
        uv = UsdShade.Shader.Define(stage, path_id+'/SeedFinishUV')
        uv.CreateIdAttr('UsdPrimvarReader_float2'); uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path_id+'/SeedFinish')
        tex.CreateIdAttr('UsdUVTexture'); tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
        tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat'); tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        tex.CreateOutput('r', Sdf.ValueTypeNames.Float)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), 'r')
        changed.append(path_id)
    if not changed:
        raise ValueError('No visible study finish received the texture variant')
    return {'id': 'roughness_seed_'+str(seed), 'changed_materials': sorted(changed),
            'category_colors_and_labels_unchanged': True, 'physical_materials_unchanged': True,
            'dependencies_sha256': {str(path): file_hash(path)}}


def _parent(parent_release):
    root = Path(parent_release).resolve(strict=True)
    manifest = load_json(root/'release_manifest.json')
    if (manifest.get('schema') not in ('study_task_release.v1', 'workroom_robot_configuration_release.v1')
            or len(manifest['tasks']) != 1):
        raise ValueError('Expected one frozen study task')
    for name, expected in manifest['files_sha256'].items():
        p = (root/name).resolve(strict=True)
        if not p.is_relative_to(root):
            raise ValueError('Parent source escapes release')
        _checked(p, expected)
    entry = manifest['tasks'][0]
    result = {}
    for name in ('scene', 'task', 'initial_state'):
        p = (root/entry[name]).resolve(strict=True)
        if not p.is_relative_to(root):
            raise ValueError('Parent input escapes release')
        result[name] = load_json(p)
    task_id = result['task']['id']; _seed(task_id, 0)
    return root, result


def prepare_asset(parent_release: Path, seed: int, output: Path, *, profile=None, attempt=0) -> Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    """Publish a new source-bound asset, never overwrite an existing directory."""
    from pxr import Usd
    from scenes.workrooms.living_room import seed_geometry as geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__, geometry.__name__, rules.__name__), entrypoint_file=Path(__file__))
    if not source or not source.get('verified'):
        raise ValueError('Study asset publication requires a complete frozen source snapshot')
    root, inputs = _parent(parent_release); task_id = inputs['task']['id']; _seed(task_id, seed)
    layer = next(row for row in inputs['scene']['task_layers'] if row['id'] == 'task')
    candidate = inputs['scene']['study_task_provenance']['candidate']
    usd = _checked(layer['usd_path'], layer['sha256'])
    candidate_path = _checked(candidate['manifest_path'], candidate['manifest_sha256'])
    parent_manifest = load_json(candidate_path)
    metadata_path = _checked(candidate_path.parent/parent_manifest['metadata_file'], parent_manifest['metadata_sha256'])
    metadata = load_json(metadata_path)
    if (metadata['configuration'] != inputs['task']['evaluation']['configuration']
            or parent_manifest['initial_usd_sha256'] != layer['sha256']
            or parent_manifest['task_id'] != task_id):
        raise ValueError('Study layer, configuration and source asset differ')
    deps = dict(layer['dependencies_sha256'])
    for path, expected in deps.items():
        _checked(path, expected)
    references = {str(root/'release_manifest.json'): file_hash(root/'release_manifest.json'),
                  str(usd): file_hash(usd), str(candidate_path): file_hash(candidate_path),
                  str(metadata_path): file_hash(metadata_path)}
    output = Path(output).resolve()
    if not output.is_relative_to(ASSET_ROOT):
        raise ValueError('Study variants belong under the versioned shared study asset root')
    stage, metadata = derive_stage(Usd.Stage.Open(str(usd)), metadata, seed)
    output.mkdir(parents=True, exist_ok=False)
    visual = _appearance(stage, seed, output)
    target = output/'initial.usda'
    if seed == 0:
        # dexbench: the package's layer-relative asset paths are re-anchored for the copy's location.
        target.write_bytes(relocated_layer_bytes(usd, target))
    else:
        stage.GetRootLayer().Export(str(target))
    metadata.update(source_snapshot=source, appearance_variant=visual)
    _write(output/'metadata.json', metadata)
    _write(output/'SOURCE.json', {'schema': 'study_seed_source.v1', 'generator_version': GENERATOR_VERSION,
        'source_snapshot': source, 'parent_references': references,
        'license': 'Original derivative geometry/roughness CC0-1.0; original book artwork/font provenance retained in parent dependency inventory',
        'physics_scope': 'Uncalibrated transformed original mass priors; no native or robot success claim'})
    shutil.copyfile(candidate_path.parent/'LICENSE.txt', output/'LICENSE.txt')
    deps.update(references); deps.update(visual['dependencies_sha256'])
    for p in sorted(output.iterdir()):
        if p.is_file():
            deps[str(p)] = file_hash(p)
    manifest = {'schema': SCHEMA, 'family': 'study', 'task_id': task_id, 'seed': seed,
        'generator_version': GENERATOR_VERSION, 'source_snapshot': source,
        'parent_references': references, 'dependencies_sha256': deps,
        'initial_usd': str(target), 'initial_usd_sha256': file_hash(target),
        'metadata': str(output/'metadata.json'), 'metadata_sha256': file_hash(output/'metadata.json'),
        'parent_initial_usd_sha256': layer['sha256'],
        'parent_configuration_sha256': inputs['task']['evaluation']['configuration']['configuration_sha256'],
        'plan_sha256': metadata['seed_plan']['plan_sha256'], 'license_file': str(output/'LICENSE.txt'),
        'physics_accepted': False, 'native_qualified': False, 'robot_manipulation_qualified': False}
    _write(output/'seed_manifest.json', manifest)
    return output/'seed_manifest.json'


def compose_case(parent_release: Path, asset_manifest_path: Path, configuration_id: str) -> dict:
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    """Adapt only task-owned inputs; preserve the selected robot installation."""
    root, values = _parent(parent_release)
    manifest_path = Path(asset_manifest_path).resolve(strict=True)
    manifest = load_json(manifest_path)
    task_id, seed = manifest['task_id'], manifest['seed']; _seed(task_id, seed)
    if manifest.get('schema') != SCHEMA or manifest.get('family') != 'study' or task_id != values['task']['id']:
        raise ValueError('Seed asset task identity differs')
    if configuration_id not in ('single_arm', 'dual_same_side', 'dual_opposed'):
        raise ValueError('Unsupported study robot configuration')
    declared = values['scene'].get('robot_configuration', {}).get('id', 'single_arm')
    if declared != configuration_id:
        raise ValueError('Requested configuration differs from the frozen robot installation')
    for path, expected in {**manifest['parent_references'], **manifest['dependencies_sha256']}.items():
        _checked(path, expected)
    _checked(manifest['initial_usd'], manifest['initial_usd_sha256'])
    md = load_json(_checked(manifest['metadata'], manifest['metadata_sha256']))
    config = rules.validate_configuration(md['configuration'])
    scene, task = values['scene'], values['task']
    layer = next(row for row in scene['task_layers'] if row['id'] == 'task')
    if (layer['sha256'] != manifest['parent_initial_usd_sha256']
            or task['evaluation']['configuration']['configuration_sha256'] != manifest['parent_configuration_sha256']
            or config['task_id'] != task_id or md['seed_plan']['seed'] != seed
            or md['seed_plan']['plan_sha256'] != manifest['plan_sha256']
            or md['seed_plan'] != plan_case(task_id, seed, task['evaluation']['configuration'])
            or config != _configuration(task['evaluation']['configuration'], md['seed_plan'])):
        raise ValueError('Parent asset/configuration differs from seed derivative')
    files = {**manifest['dependencies_sha256'], str(manifest_path): file_hash(manifest_path)}
    layer.update(usd_path=manifest['initial_usd'], sha256=manifest['initial_usd_sha256'], dependencies_sha256=deepcopy(files))
    task['episode_seed'] = seed; scene['scenario']['seed'] = seed
    task['evaluation']['configuration'] = config
    task['public_rules'] = deepcopy(config['rules'])
    task['public_rules_sha256'] = rules.canonical_hash(task['public_rules'])
    task['public_referee_contract_sha256'] = rules.canonical_hash(task['public_rules']['referee'])
    scene['study_task_provenance']['seed_derivative'] = {'manifest_path': str(manifest_path),
        'manifest_sha256': file_hash(manifest_path), 'seed': seed, 'native_qualified': False}
    evaluator = {str(p.relative_to(root/'evaluator')): p.read_text() for p in sorted((root/'evaluator').rglob('*.py'))}
    if not evaluator:
        raise ValueError('Parent evaluator source inventory is missing')
    return {**values, 'evaluator_sources': evaluator, 'actual_body_poses_m': deepcopy(md['actual_body_poses_m']),
        'interaction_targets': deepcopy(md['interaction_targets']), 'static_geometry': deepcopy(md['static_geometry']),
        'asset_files_sha256': files}
