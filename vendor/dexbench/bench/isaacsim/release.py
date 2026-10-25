"""Prepare a separate Isaac release while preserving frozen task inputs byte-for-byte."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/sequential_workcell/v1')
DEFAULT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1')
PROFILE = {
    'robot_self_collisions': True, 'filter_thumb_cmc_structure': True,
    'physics_dt_s': .0005, 'cloth_iterations': 32, 'cloth_linear_damping': .10,
    'cloth_thickness_m': .001, 'cloth_youngs_modulus_pa': 2.e6,
    'cloth_poissons_ratio': .3, 'cloth_surface_bend_stiffness': 2.e4,
    'cloth_contact_offset_m': .003, 'cloth_rest_offset_m': .0015,
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Frozen input changed: {path}: {actual} != {expected}')
    return actual


def prepare(source=SOURCE, output=DEFAULT, profile=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Release directory already contains data; choose a new version: {output}')
    frozen = load_json(source / 'release_manifest.json')
    inventory = frozen['inputs_and_artifacts']
    task_entries = []
    copies = []
    for entry in frozen['tasks']:
        row = {key: entry[key] for key in ('id', 'title_zh', 'difficulty', 'cloth')}
        for kind in ('scene', 'task'):
            original = Path(entry[kind])
            expected = inventory[str(original)]['sha256']
            checked(original, expected)
            relative = Path('tasks') / entry['id'] / f'{kind}.json'
            copies.append((original, relative, expected))
            row[kind] = str(relative)
            row[kind + '_sha256'] = expected
        task_entries.append(row)
    for filename in ('sequence_evaluator.py', 'geometry.py'):
        original = source / 'source_snapshot' / 'bench' / filename
        expected = inventory[str(original)]['sha256']
        checked(original, expected)
        copies.append((original, Path('evaluator') / filename, expected))
    asset_file = Path(frozen['shared_asset_file_inventory'])
    checked(asset_file, inventory[str(asset_file)]['sha256'])
    asset_inventory = load_json(asset_file)
    for original, record in asset_inventory.items():
        checked(original, record['sha256'])
    copies.append((asset_file, Path('asset_file_hashes.json'), digest(asset_file)))
    catalog = frozen['asset_catalog']
    checked(catalog['path'], catalog['sha256'])
    output.mkdir(parents=True, exist_ok=True)
    for original, relative, expected in copies:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        checked(destination, expected)
    shutil.copy2(source / 'release_manifest.json', output / 'source_release_manifest.json')
    physics = {
        'backend': 'Isaac Sim 6.1 native PhysX GPU', 'isaacsim_options': deepcopy(profile or PROFILE),
        'material_calibrated': False, 'cross_engine_material_equivalence_claimed': False,
        'note': 'Separate PhysX parameters. Frozen scene physics/cloth fields retain original Newton values for provenance; they are not a calibrated PhysX material conversion.',
    }
    dump(output / 'physics_profile.json', physics)
    manifest = {
        'schema_version': 'isaacsim-workcell-release-v1', 'release_id': 'isaacsim_sequential_workcell_v1',
        'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
        'source_status': frozen['status'], 'tasks': task_entries,
        'files': {str(relative): {'sha256': expected, 'source': str(original)} for original, relative, expected in copies},
        'physics_profile': 'physics_profile.json', 'physics_profile_sha256': digest(output / 'physics_profile.json'),
        'asset_catalog': catalog, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'original_release_modified': False,
    }
    dump(output / 'release_manifest.json', manifest)
    return manifest


def validate(root=DEFAULT, *, assets=True):
    root = Path(root).resolve()
    manifest = load_json(root / 'release_manifest.json')
    from scenes.board_games.runtime_release import is_board_game_release, validate_board_game_release
    if is_board_game_release(manifest):
        return validate_board_game_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'study_task_release.v1':
        from scenes.workrooms.study.release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'living_room_manipulation_task_release.v1':
        from scenes.workrooms.living_room.release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'laboratory_manipulation_task_release.v1':
        from scenes.workrooms.laboratory.release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'kitchen_manipulation_task_release.v1':
        from scenes.workrooms.kitchen.release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'mechanical_assembly_task_release.v1':
        from scenes.workrooms.mechanical_assembly.release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'mechanical_nist_task_release.v1':
        from scenes.workrooms.mechanical_assembly.nist_release import validate_release
        return validate_release(root, manifest, assets=assets)
    if manifest.get('schema') == 'living_room_task_release.v1':
        from scenes.workrooms.living_room.layer_release import validate_release
        return validate_release(root, manifest, assets=assets)
    mechanical = manifest.get('schema_version') == 'isaacsim-mechanical-fastener-release-v1'
    placement = manifest.get('schema_version') == 'isaacsim-card-room-placement-release-v1'
    if mechanical or placement:
        kind = 'Card-room placement' if placement else 'Mechanical'
        required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json',
                    'asset_file_hashes.json', 'physics_profile.json',
                    'evaluator/sequence_evaluator.py', 'evaluator/rules.py'}
        if manifest.get('physics_profile') != 'physics_profile.json':
            raise ValueError(f'{kind} profile must use its frozen local file')
        if not required <= manifest.get('files', {}).keys():
            raise ValueError(f'{kind} release lacks frozen task/evaluator inputs')
        for relative in manifest['files']:
            path = root / relative
            if Path(relative).is_absolute() or not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError(f'{kind} release file escapes its frozen directory')
        if (manifest.get('status') != 'prepared_unvalidated'
                or manifest.get('physics_accepted') is not False
                or manifest.get('task_policy_success_claimed') is not False):
            raise ValueError(f'{kind} preparation cannot claim native qualification')
    for relative, item in manifest['files'].items():
        checked(root / relative, item['sha256'])
    if manifest.get('schema_version') in {
            'isaacsim-simple-manipulation-release-v1', 'isaacsim-cloth-manipulation-release-v1',
            'isaacsim-mechanical-fastener-release-v1', 'isaacsim-card-room-placement-release-v1'}:
        # Independent tasks have no upstream task release to copy. Their scenes,
        # evaluator, assets and source provenance are frozen in this manifest.
        if manifest.get('source_release') is not None or manifest.get('original_release_modified') is not False:
            raise ValueError('Independent simple task release provenance is malformed')
    else:
        checked(root / 'source_release_manifest.json', manifest['source_release_sha256'])
    checked(root / manifest['physics_profile'], manifest['physics_profile_sha256'])
    if assets:
        checked(manifest['asset_catalog']['path'], manifest['asset_catalog']['sha256'])
        for path, item in load_json(root / 'asset_file_hashes.json').items():
            checked(path, item['sha256'])
    if mechanical:
        entries = manifest.get('tasks', [])
        if len(entries) != 1 or entries[0].get('id') != 'fastener_assembly':
            raise ValueError('Mechanical v1 release requires exactly its fastener task')
        entry = entries[0]
        if any(entry.get(key) != key+'.json' for key in ('scene', 'task', 'initial_state')):
            raise ValueError('Mechanical task entry differs from its frozen inputs')
        scene = load_json(root/'scene.json')
        task = load_json(root/'task.json')
        initial = _mechanical_initial_state(scene, task)
        if assets and manifest.get('presentation_contract_version') == 2:
            from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release import _asset_inventory
            inventory = load_json(root/'asset_file_hashes.json')
            if inventory != _asset_inventory(scene):
                raise ValueError('Mechanical presentation asset inventory is incomplete or changed')
        profile = load_json(root/manifest['physics_profile'])
        if scene.get('isaacsim') != profile.get('isaacsim_options'):
            raise ValueError('Mechanical profile must preserve the frozen task physics')
        if initial != load_json(root/'initial_state.json'):
            raise ValueError('Mechanical initial-state file and task differ')
        if task.get('public_rules') != load_json(root/'rules.json'):
            raise ValueError('Mechanical public rule file and task differ')
        if 'referee' in task.get('public_rules', {}):
            from bench.protocol.task_referee import project_task_rules
            if 'evaluator/referee.py' not in manifest['files']:
                raise ValueError('Mechanical referee source is not frozen')
            project_task_rules(task)
            _frozen_evaluator_module(root).SequenceEvaluator(task)
    if placement:
        entries = manifest.get('tasks', [])
        if (manifest.get('task_kind') == 'board_game' or manifest.get('task_id') != 'task0'
                or len(entries) != 1 or entries[0].get('id') != 'task0'):
            raise ValueError('Card-room placement requires exactly its ordinary task0 episode')
        entry = entries[0]
        if any(entry.get(key) != key+'.json' for key in ('scene', 'task', 'initial_state')):
            raise ValueError('Card-room placement entry differs from its frozen inputs')
        scene = load_json(root/'scene.json')
        task = load_json(root/'task.json')
        initial = _placement_initial_state(scene, task)
        profile = load_json(root/manifest['physics_profile'])
        if scene.get('isaacsim') != profile.get('isaacsim_options'):
            raise ValueError('Card-room placement profile differs from the frozen scene')
        if initial != load_json(root/'initial_state.json'):
            raise ValueError('Card-room placement initial-state file and task differ')
        if task.get('public_rules') != load_json(root/'rules.json'):
            raise ValueError('Card-room placement public rule file and task differ')
        _frozen_evaluator_module(root).SequenceEvaluator(task)
    return manifest


def _mechanical_initial_state(scene, task):
    """Explicit reset joints only; never borrow cloth or reset-at-goal semantics."""
    if (task.get('schema_version') != 'mechanical-fastener-task-v1'
            or task.get('id') != 'fastener_assembly' or task.get('task_id') != task['id']
            or scene.get('cloth', {}).get('enabled') is not False
            or scene.get('workroom', {}).get('family') != 'mechanical_assembly'):
        raise ValueError('Mechanical release/task/scene contract mismatch')
    from scenes.workrooms.mechanical_assembly.table_distractors import validate_distractors
    validate_distractors(scene, task)
    initial = deepcopy(task.get('initial_state'))
    if not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}:
        raise ValueError('Mechanical initialization may contain only robot joint coordinates')
    joints = initial['joint_positions_rad']
    import math
    if (not isinstance(joints, dict) or not joints
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in joints.values())):
        raise ValueError('Mechanical initialization needs finite complete robot joints')
    robot = scene.get('robot', {})
    if robot.get('hand_id') != 'sharpa_w01_right' or robot.get('assembly_version') != 'isaacsim_6_1_v2':
        raise ValueError('Mechanical v1 requires its fixed single-arm SHARPA assembly')
    from bench.control.factory import public_robot_geometry, robot_configuration
    model = public_robot_geometry(robot_configuration(scene))
    if set(joints) != set(model.names):
        raise ValueError('Mechanical initialization requires every active joint exactly once')
    model.checked_joints(joints, enforce_limits=True)
    return initial


def _placement_initial_state(scene, task):
    """Task 0 keeps the three-piece reset fixed while episode goals vary."""
    if (task.get('schema_version') != 'card-room-placement-task-v1'
            or task.get('id') != 'task0' or task.get('task_id') != 'task0'
            or scene.get('workroom', {}).get('family') != 'card_room'
            or scene.get('cloth', {}).get('enabled') is not False
            or scene.get('board_game_task') is not None):
        raise ValueError('Card-room placement release/task/scene contract mismatch')
    robot = scene.get('robot', {})
    if robot.get('hand_id') != 'sharpa_w01_right' or robot.get('assembly_version') != 'isaacsim_6_1_v2':
        raise ValueError('Card-room placement requires its fixed single-arm SHARPA assembly')
    initial = deepcopy(task.get('initial_state'))
    if not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}:
        raise ValueError('Card-room placement initialization permits only robot joint coordinates')
    joints = initial['joint_positions_rad']
    import math
    if (not isinstance(joints, dict) or not joints
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in joints.values())):
        raise ValueError('Card-room placement initialization requires finite robot joints')
    from bench.control.factory import public_robot_geometry, robot_configuration
    model = public_robot_geometry(robot_configuration(scene))
    if set(joints) != set(model.names):
        raise ValueError('Card-room placement requires every active joint exactly once')
    model.checked_joints(joints, enforce_limits=True)
    pieces = {'black_00', 'black_01', 'black_02'}
    objects = scene.get('objects', [])
    if len(objects) != 3 or {row.get('id') for row in objects} != pieces:
        raise ValueError('Card-room placement requires exactly its three black pieces')
    reset_objects = task.get('evaluation', {}).get('initial_objects', {})
    if set(reset_objects) != pieces or any(
            any(row.get(key) != reset_objects[row['id']].get(key)
                for key in ('position_m', 'quaternion_wxyz')) for row in objects):
        raise ValueError('Card-room placement evaluator reset differs from physical objects')
    return initial


def load_evaluator(root=DEFAULT):
    """Load the frozen evaluator and its frozen relative geometry dependency."""
    root = Path(root).resolve()
    manifest = validate(root, assets=False)
    if manifest.get('task_kind') == 'board_game':
        raise ValueError('Board-game evaluation requires the trusted native-substep runtime adapter')
    return _frozen_evaluator_module(root).SequenceEvaluator


def _frozen_evaluator_module(root):
    """Called only after all frozen dependencies have passed hash validation."""
    # Relative dependencies (e.g. humanlike.py) belong to the same frozen release.
    # Keying only on sequence_evaluator.py would reuse another release's cached
    # helper whenever the main evaluator stayed unchanged but a dependency did not.
    package = '_isaac_frozen_evaluator_' + digest(root / 'release_manifest.json')
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root / 'evaluator')]
        sys.modules[package] = module
    name = package + '.sequence_evaluator'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'evaluator/sequence_evaluator.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def task_inputs(root, task_id, phase='nominal', *, seed=0):
    root = Path(root)
    manifest = validate(root)
    if manifest.get('schema') == 'study_task_release.v1':
        from scenes.workrooms.study.release import task_inputs as study_inputs
        return study_inputs(root, task_id, manifest, seed=seed, phase=phase)
    if manifest.get('task_kind') == 'board_game':
        if phase != 'nominal':
            raise ValueError('Board-game releases support nominal frozen reset only; no reset-at-goal qualification')
        from scenes.board_games.runtime_release import board_game_task_inputs
        return board_game_task_inputs(root, task_id, manifest)
    if manifest.get('schema') == 'living_room_task_release.v1':
        from scenes.workrooms.living_room.layer_release import task_inputs as living_inputs
        return living_inputs(root, task_id, manifest, seed=seed, phase=phase)
    matches = [entry for entry in manifest['tasks'] if entry['id'] == task_id or entry['id'][:2] == task_id.zfill(2)]
    if len(matches) != 1:
        raise ValueError(f'Unknown/ambiguous task: {task_id}')
    entry = matches[0]
    scene = load_json(root / entry['scene'])
    task = load_json(root / entry['task'])
    profile = load_json(root / manifest['physics_profile'])
    scene['isaacsim'] = profile['isaacsim_options']
    initial = {}
    if manifest.get('schema') == 'living_room_manipulation_task_release.v1':
        if task_id != manifest.get('task_id') or phase != 'nominal':
            raise ValueError('Living-room releases require their exact task and original reset')
        from scenes.workrooms.living_room.release import episode_inputs
        scene, task, initial = episode_inputs(root, scene, task, seed)
    if manifest.get('schema') == 'laboratory_manipulation_task_release.v1':
        if task_id != manifest.get('task_id') or phase != 'nominal':
            raise ValueError('Laboratory releases require their exact task and original reset')
        from scenes.workrooms.laboratory.release import episode_inputs
        scene, task, initial = episode_inputs(root, scene, task, seed)
    if manifest.get('schema') == 'kitchen_manipulation_task_release.v1':
        if task_id != manifest.get('task_id') or phase != 'nominal':
            raise ValueError('Kitchen releases require their exact task and original reset')
        from scenes.workrooms.kitchen.release import episode_inputs
        scene, task, initial = episode_inputs(root, scene, task, seed)
    if manifest.get('schema') == 'mechanical_assembly_task_release.v1':
        if task_id != manifest.get('task_id'):
            raise ValueError('Mechanical assembly requires the exact task identity')
        if phase != 'nominal':
            raise ValueError('Mechanical task releases support only the original task reset')
        from scenes.workrooms.mechanical_assembly.release import initial_state
        initial = initial_state(scene, task)
    if manifest.get('schema') == 'mechanical_nist_task_release.v1':
        if task_id != manifest.get('task_id') or phase != 'nominal':
            raise ValueError('NIST releases require their exact task and original reset')
        from scenes.workrooms.mechanical_assembly.nist_release import episode_inputs
        scene, task, initial = episode_inputs(root, scene, task, seed)
    if manifest.get('schema_version') == 'isaacsim-card-room-placement-release-v1':
        if task_id != 'task0':
            raise ValueError('Card-room placement requires its exact task0 identity')
        if phase != 'nominal':
            raise ValueError('Card-room placement does not support reset-at-goal qualification')
        initial = _placement_initial_state(scene, task)
        module = _frozen_evaluator_module(root.resolve())
        task = module.bind_episode_targets(task, seed)
        module.SequenceEvaluator(task)
    if manifest.get('schema_version') == 'isaacsim-mechanical-fastener-release-v1':
        initial = _mechanical_initial_state(scene, task)
    if manifest.get('schema_version') == 'isaacsim-cloth-manipulation-release-v1':
        if task.get('schema_version') != 'cloth-manipulation-task-v1' or not scene['cloth']['enabled']:
            raise ValueError('Cloth release must contain explicit cloth tasks')
        initial = deepcopy(task['initial_state'])
        if set(initial) != {'joint_positions_rad'} or not initial['joint_positions_rad']:
            raise ValueError('Cloth initialization may contain only robot joint coordinates')
        import math
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in initial['joint_positions_rad'].values()):
            raise ValueError('Cloth initial joint coordinates must be finite numbers')
    if phase == 'qualification_probe':
        if task.get('schema_version') in {'simple-manipulation-task-v1', 'cloth-manipulation-task-v1',
                                        'mechanical-fastener-task-v1'}:
            raise ValueError('Independent manipulation releases do not support reset-at-goal qualification')
        probe = task.get('qualification_probe', {})
        if 'cloth_origin_m' in probe:
            scene['cloth']['origin_m'] = deepcopy(probe['cloth_origin_m'])
        names = probe.get('objects_at_goal', True)
        if names is True:
            names = list(task['goal_objects'])
        if names:
            initial['objects'] = {name: {'position_m': task['goal_objects'][name], 'quaternion_wxyz': [1, 0, 0, 0]} for name in names}
    elif phase != 'nominal':
        raise ValueError(f'Unknown phase: {phase}')
    return entry, scene, task, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--physics-profile', type=Path, help='JSON object containing isaacsim option values')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = validate(args.output) if args.verify else prepare(args.source, args.output, load_json(args.physics_profile) if args.physics_profile else None)
    print(json.dumps({'release': str(args.output), 'status': result['status'], 'tasks': [x['id'] for x in result['tasks']]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
