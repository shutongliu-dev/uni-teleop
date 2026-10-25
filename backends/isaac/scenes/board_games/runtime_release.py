"""Read frozen tabletop releases through the canonical bench loader.

The compatibility fields returned here exist only in memory. The task, scene,
reset state, evaluator package and preparation manifest are never rewritten.
Native settlement remains the responsibility of the trusted runtime adapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

from .tabletop.geometry import BoardGameGeometry
from .tabletop.release import sha256, verify_evaluator_sources, verify_public_rule_contract
from bench.paths import load_json

RELEASE_SCHEMA = 'robot_board_game_task_release.v1'


def is_board_game_release(manifest: dict) -> bool:
    return manifest.get('schema') == RELEASE_SCHEMA


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _checked(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in '0123456789abcdef' for char in expected)
            or not path.is_file() or sha256(path) != expected):
        raise ValueError(f'Frozen input changed or missing: {path}')


def _frozen_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Board-game release file requires a relative path')
    path = root / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or not path.resolve().is_relative_to(root)
            or any(part.is_symlink() for part in (path, *path.parents) if part != root)):
        raise ValueError('Board-game release file escapes its frozen directory or uses a symlink')
    return path


def _validate_inputs(scene, task, initial):
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.workrooms.card_room.scene import environment_reference

    if scene.get('workroom') != environment_reference():
        raise ValueError('Frozen board-game environment differs from its fixed parent scene')
    binding = scene.get('board_game_task', {})
    geometry = BoardGameGeometry(**{f.name: task['geometry'][f.name] for f in fields(BoardGameGeometry)})
    normalized_geometry = json.loads(json.dumps(asdict(geometry), allow_nan=False))
    if (binding.get('schema') != 'board_game_native_task.v1'
            or binding.get('geometry') != normalized_geometry
            or scene.get('cloth', {}).get('enabled') is not False):
        raise ValueError('Frozen board-game scene and evaluator geometry differ')
    if binding.get('players') != [
            {'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
            {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]:
        raise ValueError('Board-game release requires frozen black/left and white/right ownership')
    objects = scene.get('objects', [])
    inventory = {f'{color}_{index:02d}' for color, count in geometry.inventory_counts.items()
                 for index in range(count)}
    if (len(objects) != len(inventory)
            or {row.get('id') for row in objects} != inventory):
        raise ValueError('Board-game scene must contain its complete piece inventory exactly once')
    fixtures = scene.get('fixtures', [])
    if sum(row.get('id') == binding.get('board_fixture_id') for row in fixtures) != 1:
        raise ValueError('Board-game scene requires its named board fixture exactly once')
    if (not isinstance(scene.get('isaacsim'), dict) or not scene['isaacsim']
            or not isinstance(scene.get('physics'), dict) or not scene['physics']):
        raise ValueError('Board-game release lacks its frozen native physics profile')
    if (not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}
            or binding.get('initial_state') != initial):
        raise ValueError('Board-game initialization must preserve its frozen robot-only reset')
    joints = initial['joint_positions_rad']
    if (not isinstance(joints, dict) or not joints
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in joints.values())):
        raise ValueError('Board-game initialization requires finite robot joint coordinates')
    robot = scene.get('robot', {})
    if (robot.get('hand_id') != 'dual_fr3_sharpa_w01'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'):
        raise ValueError('Board-game runtime requires its frozen dual FR3 + SHARPA assembly')
    model = public_robot_geometry(robot_configuration(scene))
    if set(joints) != set(model.names):
        raise ValueError('Board-game initialization requires every active joint exactly once')
    for side in model.sides.values():
        side.checked_joints({name: joints[name] for name in side.names}, enforce_limits=True)
    return geometry


def validate_board_game_release(root, manifest, *, assets=True):
    """Verify frozen inputs and return a non-writing canonical metadata view."""
    root = Path(root).resolve()
    if not is_board_game_release(manifest):
        raise ValueError('Expected a frozen tabletop board-game release')
    if (manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False):
        raise ValueError('Board-game preparation cannot claim native qualification')
    if (manifest.get('scene') != 'scene.json' or manifest.get('task') != 'task.json'
            or manifest.get('public_rules') != 'rules.json'
            or manifest.get('frozen_evaluator') != 'evaluator/tabletop'):
        raise ValueError('Board-game release must use its frozen local input files')
    hashes = manifest.get('files_sha256')
    if not isinstance(hashes, dict) or not {'scene.json', 'task.json', 'initial_state.json', 'rules.json'} <= hashes.keys():
        raise ValueError('Board-game release lacks frozen scene/task/reset/rule inputs')
    for relative, expected in hashes.items():
        _checked(_frozen_path(root, relative), expected)
    scene, task, initial = (load_json(root / name)
                            for name in ('scene.json', 'task.json', 'initial_state.json'))
    if task.get('task_id') != manifest.get('task_id'):
        raise ValueError('Board-game release and task identity differ')
    verify_public_rule_contract(task, release=root)
    source_receipt = verify_evaluator_sources(root, task)
    expected_sources = {f'evaluator/tabletop/{name}': value
                        for name, value in source_receipt['sources_sha256'].items()}
    if any(hashes.get(name) != value for name, value in expected_sources.items()):
        raise ValueError('Board-game manifest lacks its complete frozen evaluator inventory')
    geometry = _validate_inputs(scene, task, initial)
    asset_hashes = manifest.get('asset_files_sha256')
    if not isinstance(asset_hashes, dict) or not asset_hashes:
        raise ValueError('Board-game release lacks frozen asset dependencies')
    asset_manifests = {str(Path(row['asset_root']).resolve() / 'manifest.json')
                       for row in [*scene['objects'], *scene['fixtures']]}
    room = scene['workroom']
    required_assets = asset_manifests | {room['usd_path'], room['dependency_manifest_path']}
    if not required_assets <= asset_hashes.keys():
        raise ValueError('Board-game release lacks a used task or parent-room asset manifest')
    if assets:
        from bench.simulation.isaacsim_backend.workroom import verify_reference
        for path, expected in asset_hashes.items():
            if not Path(path).is_absolute():
                raise ValueError('Board-game assets require absolute frozen paths')
            _checked(Path(path), expected)
        verify_reference(room)
    from scenes.workrooms.catalog import task_definition
    definition = task_definition('card_room', task['task_id'])
    entry = {'id': task['task_id'], 'title_zh': definition['title'], 'difficulty': geometry.level,
             'cloth': False, 'scene': 'scene.json', 'task': 'task.json',
             'initial_state': 'initial_state.json', 'task_kind': 'board_game'}
    return {**deepcopy(manifest), 'tasks': [entry], 'task_kind': 'board_game',
            'runtime_identity': {
                'schema': 'board_game_runtime_identity.v1',
                'evaluator_sha256': _canonical_hash(source_receipt['sources_sha256']),
                'evaluator_hash_source': 'task.json:sources_sha256; canonical JSON SHA256',
                'physics_profile_sha256': _canonical_hash({'physics': scene['physics'], 'isaacsim': scene['isaacsim']}),
                'physics_hash_source': 'scene.json:{physics,isaacsim}; canonical JSON SHA256'}}


def board_game_task_inputs(root, task_id, manifest):
    """Return the ordinary loader tuple without modifying frozen payloads."""
    if task_id != manifest.get('task_id'):
        raise ValueError(f'Unknown board-game task: {task_id}')
    root = Path(root)
    scene, task, initial = (load_json(root / name)
                            for name in ('scene.json', 'task.json', 'initial_state.json'))
    return deepcopy(manifest['tasks'][0]), scene, task, initial
