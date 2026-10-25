"""Select/export pinned Astra rooms without changing any frozen bench task."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.workroom import SCHEMA, SUPPORT_PATHS, checked, digest, verify_reference

FAMILIES = ('mechanical_assembly', 'laboratory', 'kitchen', 'study', 'card_room', 'living_room')
VERSION = 'rooms_20261017_v3'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms') / VERSION
CURRENT_RELEASE = Path(__file__).with_name('current_release.json')
COLLECTION_SHA256 = 'd46caac50ad083782a23facebb3f3a5ee359a8b85f7d0bd981d394894a927c7c'
BUILD_ROOT = Path(f'{DATA_ROOT}/datasets/scence/development/20261017_observation_details/build01/renders')
BASE_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/'
                    'v12_mount_v2_r1/sharpa_w01_right')
BASE_MANIFEST_SHA256 = '17c232f7643b0631a66ce186b34d21f7f6c4128ffb471cbb9c4278cec36dc91b'
RUNTIME_PROFILE = 'workroom_engine_v4'
PHYSICS_OVERRIDES = {'robot_material_binding': 'collider_instance_v1',
                     'robot_base_constraint': 'native_fixed_base_v1',
                     'static_mesh_contact_mode': 'kinematic_mesh_v1',
                     'workroom_surface_materials': {
                         'schema': 'workroom-surface-materials-v1',
                         'surfaces': {name: {
                             'static_friction': .4, 'dynamic_friction': .4,
                             'restitution': 0., 'friction_combine': 'average',
                             'restitution_combine': 'average',
                             'parameter_source': 'Existing workroom table friction 0.4; explicit simulation prior, not measured calibration. Floor shares this diagnostic prior.'
                         } for name in ('table', 'floor')}}}
# Published USD + its original direct-dependency inventory. No builder import or
# asset rebuild is needed when the authoring package is moved within the repo.
PINNED = {'mechanical_assembly': ('66d57f099f06e3796124476e5c9cd86b15305e2854e92f92551486b0be316b7b',
                         'ce13b43508e528e76c8626f7d8e891d12ea33cf224be2945a8d79e34c39aea21'),
 'laboratory': ('85e2ddbd33ebe3ce5b43abeee22b073d2d3575072762f750f789a32c88a31a7f',
                '635e83e5bf37d2cf2c36bba48e68fc1b9179988ca42238f357b6d64e0fee01ff'),
 'kitchen': ('815c5941090b06475bd2a092e75e2c25c72b8f475b24a2832d15cb0cf97c9f0c',
             '6026e72f3db52ec8806b9db15b048d78cba2d197ee7ee87cf9c1d0904ae6e038'),
 'study': ('b30c810279ae58631a50f3963c86103624984e4197251bac19a009b31fa66a30',
           '11f0eb8f697bd36b122cd56ca0fb0930ff13b87cbd48611acbc5717396c9538d'),
 'card_room': ('fc72e60d901e2199f9fd3867087b42879ee6e175bd224952aba52697a6d90906',
               'c9b9aed0651a3043e2ee3af07bdbfb0f7052b47bbe13d9ea39faa1a9daf3a0f1'),
 'living_room': ('3d98a1f64708f802e2bf9de66583a3e149b75baeb1be6c1563097d1a073b2421',
                 '9b285e7f435410c08ffb15a1a0cfaaac5a1bc8eb9c7d5baf79706365f3c3d1d1')}


def release_pointer():
    """Reject a human-facing release pointer that differs from this runtime."""
    pointer = load_json(CURRENT_RELEASE)
    expected = {'schema': 'scence.release_pointer.v1', 'version': VERSION,
                'scene_manifest': str(ASSET_ROOT / 'manifest.json'),
                'scene_manifest_sha256': COLLECTION_SHA256}
    for key, value in expected.items():
        if pointer.get(key) != value:
            raise ValueError(f'Workroom release pointer differs from the pinned runtime: {key}')
    checked(pointer['scene_manifest'], COLLECTION_SHA256)
    manifest = load_json(Path(pointer['scene_manifest']))
    entries = {entry['family']: entry for entry in manifest['assets']}
    if len(manifest['assets']) != len(FAMILIES) or set(entries) != set(FAMILIES):
        raise ValueError('Workroom collection must contain exactly the six pinned families')
    for family in FAMILIES:
        if (entries[family]['scene'] != str(ASSET_ROOT / family / 'scene.usda')
                or entries[family]['scene_sha256'] != PINNED[family][0]):
            raise ValueError(f'Workroom collection differs from the pinned runtime: {family}')
    return pointer


def reference(family):
    if family not in FAMILIES:
        raise ValueError(f'Unknown workroom {family!r}; choose from {FAMILIES}')
    release_pointer()
    usd_hash, inventory_hash = PINNED[family]
    return {'schema': SCHEMA, 'family': family, 'version': VERSION,
            'usd_path': str(ASSET_ROOT / family / 'scene.usda'), 'usd_sha256': usd_hash,
            'source_prim': '/World',
            'dependency_manifest_path': str(BUILD_ROOT / family / 'scene_description.json'),
            'dependency_manifest_sha256': inventory_hash}


def scene(family, *, diagnostic=True, lighting_profile='nominal'):
    """Return a fresh env specification; diagnostics are opt-out, never tasks.

The robot and rigid-pad model derive from the pinned single-arm v12 release.
Explicit runtime overrides are recorded separately; the release and its scorer
remain byte-identical and are not used to score this new environment.
"""
    from bench.isaacsim.release import task_inputs
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from .observations import OBSERVATION_PROFILE, camera_specs
    from bench.simulation.assets.workrooms.layouts import LIGHTING_PROFILES

    if lighting_profile not in LIGHTING_PROFILES:
        raise ValueError(f'Unknown lighting profile: {lighting_profile}')

    spec = reference(family)
    hashes = verify_reference(spec)
    checked(BASE_RELEASE / 'release_manifest.json', BASE_MANIFEST_SHA256)
    entry, result, _task, _initial = task_inputs(BASE_RELEASE, '01_cube_push')
    result = deepcopy(result)
    result['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    result['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1',
                                             'family': family, 'profile': lighting_profile}
    cube = deepcopy(result['objects'][0])
    description = load_json(Path(spec['dependency_manifest_path']))
    result['description'] = f'Astra {family}: static furnished workroom with optional native dynamics diagnostics.'
    result['robot']['base_position_m'] = [-.13, 0., .78]
    result['robot']['arm_home_rad'] = [description['robot_joint_positions'][f'fr3_joint{i}'] for i in range(1, 8)]
    result['table'] = {'position_m': [.65, 0., .76], 'half_extents_m': [.5, .5, .02], 'friction': .4}
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        result.pop(key, None)
    result.update(objects=[], fixtures=[], goal_markers=[], cloth={'enabled': False},
                  workbench={'enabled': False}, workcell_details={'enabled': False},
                  workroom=spec, workroom_diagnostics={})
    result['cameras'] = camera_specs(description)
    if diagnostic:
        asset = load_task_asset(cube['asset_id'], cube['asset_root'], verify_hashes=True)
        half = [float(value) / 2 for value in asset['dimensions_m']]
        for name, xy, support in [('cube_table', [.30, -.35], 'table'), ('cube_floor', [.30, -.70], 'floor')]:
            top = .78 if support == 'table' else 0.
            position = [*xy, top + half[2] + .22]
            result['objects'].append({**deepcopy(cube), 'id': name, 'position_m': position})
            result['workroom_diagnostics'][name] = {
                'initial_position_m': position, 'half_extents_m': half,
                'support_top_z_m': top, 'expected_support_path': SUPPORT_PATHS[support],
                'expected_rest_center_z_m': top + half[2], 'release_height_m': .22,
            }
    result['workroom_provenance'] = {
        'status': 'diagnostic_unvalidated', 'physics_accepted': False, 'task_qualification': False,
        'runtime_profile': RUNTIME_PROFILE,
        'observation_profile': OBSERVATION_PROFILE,
        'lighting_profile': lighting_profile,
        'robot_initialization_override': {'arm_home_rad': result['robot']['arm_home_rad'],
                                          'source': 'Pinned Astra scene_description robot_joint_positions'},
        'physics_overrides': deepcopy(PHYSICS_OVERRIDES),
        'isaacsim_options_sha256': hashlib.sha256(json.dumps(
            result['isaacsim'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'base_release': str(BASE_RELEASE), 'base_task_id': entry['id'],
        'base_release_manifest_sha256': BASE_MANIFEST_SHA256,
        'base_scene_sha256': entry['scene_sha256'],
        'base_physics_profile_sha256': digest(BASE_RELEASE / 'physics_profile.json'),
        'source_file_hashes': {**hashes, str(CURRENT_RELEASE): digest(CURRENT_RELEASE),
                               str(ASSET_ROOT / 'manifest.json'): COLLECTION_SHA256},
        'registry_sha256': digest(__file__),
        'scope': 'New scene stratum; static dressing is not a graspable task asset. No task/model score.',
    }
    return result


def validate(family):
    """Verify file hashes/dependencies and compose on CPU OpenUSD; no Isaac app."""
    from bench.simulation.isaacsim_backend.workroom import preflight
    value = scene(family)
    return preflight(value['workroom'], value['workroom_diagnostics'],
                     surface_materials=value['isaacsim']['workroom_surface_materials'])


def export_scene(family, output, *, diagnostic=True, lighting_profile='nominal'):
    """Write a new directory containing env-ready scene.json and provenance."""
    value = scene(family, diagnostic=diagnostic, lighting_profile=lighting_profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_path = output / 'scene.json'
    scene_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    record = {**value['workroom_provenance'], 'family': family, 'version': VERSION,
              'scene': 'scene.json', 'scene_sha256': digest(scene_path), 'diagnostic_objects': bool(diagnostic)}
    (output / 'provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Discover fixed scenes/tasks, or export the legacy six-room diagnostic collection.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('scenes', help='List the fixed scene definitions used by task selection')
    tasks = commands.add_parser('tasks', help='List implemented tasks; planned slots require explicit opt-in')
    tasks.add_argument('--family', choices=FAMILIES)
    tasks.add_argument('--include-planned', action='store_true')
    task = commands.add_parser('task', help='Load and verify a fixed scene task release; never execute physics')
    task.add_argument('--family', choices=FAMILIES, required=True)
    task.add_argument('--task', required=True)
    commands.add_parser('list', help='List the legacy v3 room collection used by observation/drop diagnostics')
    export = commands.add_parser('export', help='Export a legacy collection diagnostic environment (never overwrite)')
    export.add_argument('--family', choices=FAMILIES, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--no-diagnostics', action='store_true')
    export.add_argument('--lighting-profile', choices=('nominal', 'daylight', 'warm', 'low_light'), default='nominal')
    check = commands.add_parser('validate', help='CPU preflight of a legacy collection diagnostic room; no GPU')
    check.add_argument('--family', choices=FAMILIES, required=True)
    args = parser.parse_args(argv)
    if args.command in ('scenes', 'tasks', 'task'):
        from .catalog import list_scenes, list_tasks, load_task
        if args.command == 'scenes':
            result = list_scenes()
        elif args.command == 'tasks':
            result = list_tasks(args.family, include_planned=args.include_planned)
        else:
            try:
                loaded = load_task(args.family, args.task)
            except (ValueError, FileNotFoundError) as exc:
                parser.error(str(exc))
            result = {'definition': loaded['definition'], 'environment': loaded['environment'],
                      'release_verified': True, 'native_execution_started': False}
    elif args.command == 'list':
        release_pointer()
        result = {'version': VERSION, 'families': list(FAMILIES),
                  'scope': 'Legacy six-room observation/drop diagnostic collection; no task qualification'}
    elif args.command == 'export':
        result = export_scene(args.family, args.output, diagnostic=not args.no_diagnostics,
                              lighting_profile=args.lighting_profile)
    else:
        result = validate(args.family)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
