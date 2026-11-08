"""Prepare and validate immutable robot releases for card-room expansions.

The original room, robot layout and controller inputs are retained. Scene-v2
tabletop USD is referenced without physical edits. Historical tests/reports are
not inputs. Preparation grants no manipulation, game-completion or model score.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile

from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT, portable
from scenes.workrooms.card_room.game_designs import TASK_IDS, design_for
from scenes.workrooms.card_room.scene import environment_reference

RELEASE_SCHEMA = 'card_game_expansion_release.v1'
TASK_SCHEMA = 'card_game_expansion_task.v1'
BINDING_SCHEMA = 'card_game_expansion_native_task.v1'
SOURCES = (
    'scenes/workrooms/card_room/expansion_referee.py',
    'scenes/workrooms/card_room/expansion_rules.py',
    'scenes/workrooms/card_room/game_designs.py',
    'scenes/board_games/expansion_native_observer.py',
    'scenes/board_games/expansion_bundle_observer/__init__.py',
    'scenes/board_games/expansion_bundle_observer/observer.py',
    'bench/runtime/card_game_worker.py',
    'bench/protocol/card_games.py',
    'bench/simulation/isaacsim_backend/native_task_bundle.py',
)
CONTACT_SAMPLING = {'schema': 'native-rigid-body-contact-sampling.v1',
                    'sleep_threshold': 0., 'stabilization_threshold': 0.}
PLAYERS = [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
           {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(portable(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def checked(path, expected):
    path = Path(path)
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or not path.is_file() or sha256(path) != expected):
        raise ValueError('Frozen card-game input missing or changed: '+str(path))
    return path


def local_path(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '..' in Path(relative).parts):
        raise ValueError('Frozen release requires safe relative file paths')
    result = root/relative
    if not result.resolve().is_relative_to(root) or any(p.is_symlink() for p in (result, *result.parents) if p != root):
        raise ValueError('Frozen release file escapes its directory or uses symlinks')
    return result


def checked_asset(root):
    root = Path(root).resolve()
    manifest = load_json(root/'manifest.json')
    metadata = load_json(root/'metadata.json')
    if (manifest.get('schema') != 'card_game_candidate_asset_manifest.v1'
            or metadata.get('schema') != 'card_game_candidate_assets.v1'
            or manifest.get('task_id') not in TASK_IDS or metadata.get('task_id') != manifest['task_id']
            or manifest.get('initial_usd') != str(root/'initial.usda')
            or manifest.get('metadata') != str(root/'metadata.json')
            or manifest.get('root_prim') != '/World/CardGame' or metadata.get('root_prim') != '/World/CardGame'
            or manifest.get('parent_environment') != environment_reference()
            or metadata.get('parent_environment') != environment_reference()):
        raise ValueError('Scene asset identity differs from its declared frozen card-room inputs')
    expected = {'initial.usda': manifest.get('initial_usd_sha256'), 'metadata.json': manifest.get('metadata_sha256')}
    if manifest.get('files_sha256') != expected:
        raise ValueError('Scene asset manifest lacks exact USD/metadata hashes')
    for relative, digest in expected.items():
        checked(local_path(root, relative), digest)
    ids = {p['id'] for p in design_for(metadata['task_id'])['pieces']}
    if (set(metadata.get('actors', {})) != ids or manifest.get('actor_count') != len(ids)
            or metadata['actors'] != {name: '/World/CardGame/Actors/'+name for name in ids}
            or any(set(metadata.get(key, {})) != ids for key in (
                'masses_kg', 'mass_properties', 'initial_positions_m', 'initial_orientations_wxyz',
                'actor_geometry', 'actor_collision_paths'))
            or metadata.get('fixture_collision_paths') != ['/World/CardGame/Fixtures/Deck']
            or metadata.get('dimensions') != design_for(metadata['task_id'])):
        raise ValueError('Scene asset inventory/geometry differs from its evaluated task')
    return metadata, {str(root/name): sha256(root/name) for name in ('manifest.json', 'initial.usda', 'metadata.json')}


def verify_task(task):
    from scenes.workrooms.card_room.expansion_referee import public_rule_contract
    if task.get('schema') != TASK_SCHEMA or task.get('task_id') not in TASK_IDS:
        raise ValueError('Expected a versioned card-game expansion task')
    contract = public_rule_contract(task['task_id'])
    if (task.get('public_rules') != contract or task.get('public_rules_sha256') != canonical_hash(contract)
            or task.get('public_rules_file') != 'rules.json'
            or task.get('asset_metadata', {}).get('task_id') != task['task_id']
            or task.get('capabilities') != {'robot_episodes': True}
            or task.get('physics_accepted') is not False or task.get('task_policy_success_claimed') is not False):
        raise ValueError('Frozen public rule contract differs from the actual expansion evaluator')
    return contract


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {relative: sha256(root/relative) for relative in SOURCES}


def verify_sources(root, task, hashes):
    expected = task.get('sources_sha256')
    if expected != source_hashes():
        raise ValueError('Imported expansion evaluator differs from frozen source inventory')
    actual = {str(path.relative_to(root/'evaluator')) for path in (root/'evaluator').rglob('*.py')}
    if actual != set(expected):
        raise ValueError('Frozen expansion evaluator has undeclared or missing Python dependencies')
    source_root = Path(__file__).resolve().parents[2]
    for relative, digest in expected.items():
        if hashes.get('evaluator/'+relative) != digest:
            raise ValueError('Manifest omits a frozen expansion evaluator source')
        checked(local_path(root, 'evaluator/'+relative), digest)
        module_name = relative[:-3].replace('/', '.')
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != source_root/relative:
            raise ValueError('Expansion evaluator imported outside the verified source tree')


def make_scene(task_id, asset_root, *, output):
    from scenes.workrooms.card_room.game_task import make_scene as original_scene
    metadata, asset_hashes = checked_asset(asset_root)
    if metadata['task_id'] != task_id:
        raise ValueError('Wrong card-game asset selected')
    scene, initial = original_scene('tic_tac_toe_3x3')
    # Keep the original opposed bases, reset joints, hand parameters and stands.
    original_binding = scene['board_game_task']
    scene['objects'] = []
    scene['fixtures'] = [row for row in scene['fixtures'] if row['id'] in ('stand_left', 'stand_right')]
    scene['description'] = 'Two independently owned original card-room robots play physical '+task_id+'.'
    scene['board_game_task'] = {'schema': BINDING_SCHEMA, 'task_id': task_id,
        'players': deepcopy(PLAYERS), 'initial_state': initial,
        'home_design': original_binding['home_design'], 'asset_root': str(Path(asset_root).resolve()),
        'deck_collision_paths': metadata['fixture_collision_paths'],
        'reset_seed_scope': 'Immutable scene-v2 physical reset; session seed does not relabel visible card faces.'}
    observer_dir = Path(output)/'evaluator/scenes/board_games/expansion_bundle_observer'
    observer_source = Path(__file__).with_name('expansion_bundle_observer')
    observer_files = {p.name: sha256(p) for p in observer_source.glob('*.py')}
    scene['native_task_bundle'] = {
        'schema': 'native-usd-task-bundle-v1', 'usd_path': str(Path(asset_root).resolve()/'initial.usda'),
        'usd_sha256': asset_hashes[str(Path(asset_root).resolve()/'initial.usda')],
        'root_prim_path': '/World/CardGame', 'rigid_bodies': metadata['actors'], 'joints': {},
        'rigid_body_sampling': deepcopy(CONTACT_SAMPLING),
        'observer': {'path': str(observer_dir/'observer.py'), 'sha256': observer_files['observer.py'],
                     'files': observer_files, 'configuration': {'projection': 'private_worker_only'}}}
    # The asset preview camera is task-owned. Correct its key conventions while
    # leaving both original fixed wrist mounts and room cameras unchanged.
    camera = metadata['cameras']['tabletop']
    scene['cameras']['task'].update(eye=camera['eye_m'], target=camera['target_m'],
        focal_length_mm=camera['focal_length_mm'], resolution=[960, 720])
    scene['board_game_provenance']['expansion_asset_hashes'] = asset_hashes
    scene['board_game_provenance']['parent_robot_layout'] = 'unchanged tic_tac_toe_3x3 card_room_ready_v3'
    scene['board_game_provenance']['candidate_physics_reauthored'] = False
    # Used stand assets retain their original library files; unused pieces are
    # not loaded into the native stage.
    from bench.simulation.assets.task_parts import load_task_asset
    for fixture in scene['fixtures']:
        load_task_asset(fixture['asset_id'], fixture['asset_root'], verify_hashes=True)
        # Freeze the complete small library dependency inventory, as the old
        # release does, without reauthoring any stand geometry.
        for path in Path(fixture['asset_root']).rglob('*'):
            if path.is_file():
                asset_hashes[str(path)] = sha256(path)
    asset_hashes.update(scene['board_game_provenance']['source_file_hashes'])
    return scene, initial, metadata, asset_hashes


def prepare_release(output, task_id, *, asset_root=None):
    from scenes.workrooms.card_room.expansion_referee import public_rule_contract
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Create a fresh card-game release under shared datasets')
    if task_id not in TASK_IDS:
        raise ValueError('Unknown expansion task')
    asset_root = Path(asset_root or f'{DATA_ROOT}/assets/task_parts/board_games')
    if asset_root.name == 'board_games':
        asset_root = asset_root/task_id/'scene_v2'
    scene, initial, metadata, assets = make_scene(task_id, asset_root, output=output)
    contract = public_rule_contract(task_id)
    task = {'schema': TASK_SCHEMA, 'task_id': task_id, 'public_rules': contract,
            'public_rules_sha256': canonical_hash(contract), 'public_rules_file': 'rules.json',
            'asset_metadata': metadata, 'sources_sha256': source_hashes(),
            'capabilities': {'robot_episodes': True}, 'physics_accepted': False,
            'task_policy_success_claimed': False, 'initialization': 'reset_only_from_immutable_scene_v2',
            'release_policy': 'new_directory_for_each_source_scene_physics_or_rule_change'}
    verify_task(task)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for name, value in (('scene.json', scene), ('task.json', task), ('rules.json', contract), ('initial_state.json', initial)):
            dump(staging/name, value)
        source_root = Path(__file__).resolve().parents[2]
        for relative in SOURCES:
            target = staging/'evaluator'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root/relative, target)
        files = {str(p.relative_to(staging)): sha256(p) for p in staging.rglob('*') if p.is_file()}
        manifest = {'schema': RELEASE_SCHEMA, 'task_id': task_id, 'status': 'prepared_unvalidated',
                    'scene': 'scene.json', 'task': 'task.json', 'initial_state': 'initial_state.json',
                    'public_rules': 'rules.json', 'frozen_evaluator': 'evaluator',
                    'files_sha256': files, 'asset_files_sha256': assets,
                    'physics_accepted': False, 'task_policy_success_claimed': False,
                    'parent_environment': environment_reference()}
        dump(staging/'release_manifest.json', manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_release(output, manifest)


def validate_release(root, manifest, *, assets=True):
    from bench.control.factory import public_robot_geometry, robot_configuration
    root = Path(root).resolve()
    if (manifest.get('schema') != RELEASE_SCHEMA or manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False or manifest.get('task_policy_success_claimed') is not False
            or any(manifest.get(k) != v for k, v in {'scene': 'scene.json', 'task': 'task.json',
                'initial_state': 'initial_state.json', 'public_rules': 'rules.json', 'frozen_evaluator': 'evaluator'}.items())):
        raise ValueError('Unknown or malformed card-game expansion release')
    hashes = manifest.get('files_sha256', {})
    required_files = {'scene.json', 'task.json', 'rules.json', 'initial_state.json'} | {'evaluator/'+p for p in SOURCES}
    if set(hashes) != required_files:
        raise ValueError('Expansion release has an incomplete or undeclared source inventory')
    for relative, digest in hashes.items():
        checked(local_path(root, relative), digest)
    scene, task, initial, rules = (load_json(root/name) for name in
                                  ('scene.json', 'task.json', 'initial_state.json', 'rules.json'))
    if verify_task(task) != rules or manifest.get('task_id') != task['task_id']:
        raise ValueError('Expansion release identity/public rules differ')
    verify_sources(root, task, hashes)
    binding = scene.get('board_game_task', {})
    metadata, scene_assets = checked_asset(binding['asset_root'])
    if (task['asset_metadata'] != metadata or scene.get('workroom') != environment_reference()
            or manifest.get('parent_environment') != environment_reference()
            or binding.get('schema') != BINDING_SCHEMA or binding.get('task_id') != task['task_id']
            or binding.get('players') != PLAYERS or binding.get('initial_state') != initial
            or binding.get('deck_collision_paths') != metadata['fixture_collision_paths']
            or scene.get('objects') != [] or scene.get('cloth') != {'enabled': False}
            or {v['id'] for v in scene.get('fixtures', [])} != {'stand_left', 'stand_right'}):
        raise ValueError('Expansion scene changed task inventory, room, ownership or reset')
    bundle = scene.get('native_task_bundle', {})
    if (bundle.get('schema') != 'native-usd-task-bundle-v1' or bundle.get('root_prim_path') != '/World/CardGame'
            or bundle.get('usd_path') != str(Path(binding['asset_root'])/'initial.usda')
            or bundle.get('usd_sha256') != scene_assets.get(bundle.get('usd_path'))
            or bundle.get('rigid_bodies') != metadata['actors'] or bundle.get('joints') != {}
            or bundle.get('rigid_body_sampling') != CONTACT_SAMPLING):
        raise ValueError('Expansion native bundle differs from immutable scene-v2 physics')
    observer = bundle.get('observer', {})
    observer_rel = 'scenes/board_games/expansion_bundle_observer/'
    if (observer.get('path') != str(root/'evaluator'/observer_rel/'observer.py')
            or observer.get('sha256') != task['sources_sha256'][observer_rel+'observer.py']
            or observer.get('files') != {n: task['sources_sha256'][observer_rel+n] for n in ('__init__.py', 'observer.py')}
            or observer.get('configuration') != {'projection': 'private_worker_only'}):
        raise ValueError('Expansion bundle projection is not the frozen private observer')
    robot = scene.get('robot', {})
    if (robot.get('hand_id') != 'dual_fr3_sharpa_w01' or robot.get('assembly_version') != 'isaacsim_6_1_v2'
            or scene.get('physics') != {'dt_s': .0005, 'control_dt_s': .02, 'gravity_m_s2': [0, 0, -9.81],
                                       'backend': 'native_isaacsim_physx_gpu'}):
        raise ValueError('Expansion changed the robot embodiment or native physics profile')
    model = public_robot_geometry(robot_configuration(scene))
    if set(initial) != {'joint_positions_rad'} or set(initial['joint_positions_rad']) != set(model.names):
        raise ValueError('Expansion reset must contain only every original active robot joint')
    for side in model.sides.values():
        side.checked_joints({name: initial['joint_positions_rad'][name] for name in side.names}, enforce_limits=True)
    asset_hashes = manifest.get('asset_files_sha256', {})
    required_assets = set(scene_assets) | {scene['workroom']['usd_path'], scene['workroom']['dependency_manifest_path']}
    required_assets |= {str(Path(row['asset_root'])/'manifest.json') for row in scene['fixtures']}
    if not required_assets <= set(asset_hashes) or any(asset_hashes.get(k) != v for k, v in scene_assets.items()):
        raise ValueError('Expansion release omitted used immutable asset dependencies')
    if assets:
        from bench.simulation.isaacsim_backend.workroom import verify_reference
        for path, digest in asset_hashes.items():
            if not Path(path).is_absolute():
                raise ValueError('Frozen asset paths must be absolute')
            checked(path, digest)
        verify_reference(scene['workroom'])
    design = design_for(task['task_id'])
    entry = {'id': task['task_id'], 'title_zh': design['title'], 'difficulty': 4+TASK_IDS.index(task['task_id']),
             'cloth': False, 'scene': 'scene.json', 'task': 'task.json', 'initial_state': 'initial_state.json',
             'task_kind': 'board_game'}
    return {**deepcopy(manifest), 'tasks': [entry], 'task_kind': 'board_game', 'runtime_identity': {
        'schema': 'card_game_expansion_runtime_identity.v1',
        'evaluator_sha256': canonical_hash(task['sources_sha256']),
        'evaluator_hash_source': 'task.json:sources_sha256; canonical JSON SHA256',
        'physics_profile_sha256': canonical_hash({'physics': scene['physics'], 'isaacsim': scene['isaacsim'],
            'native_task_bundle_usd_sha256': bundle['usd_sha256'], 'rigid_body_sampling': bundle['rigid_body_sampling']}),
        'physics_hash_source': 'frozen physics profile, explicit task-actor sampling thresholds and scene-v2 USD hash'}}


def task_inputs(root, task_id, manifest):
    if task_id != manifest['task_id']:
        raise ValueError('Unknown expansion task: '+str(task_id))
    root = Path(root)
    scene, task, initial = (load_json(root/name) for name in ('scene.json', 'task.json', 'initial_state.json'))
    return deepcopy(manifest['tasks'][0]), scene, task, initial


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_IDS, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_release(args.output, args.task, asset_root=args.asset_root)
    print(json.dumps({'release': str(args.output.resolve()), 'task_id': args.task, 'status': manifest['status']}))


if __name__ == '__main__':
    main()
