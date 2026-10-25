"""Discover fixed workrooms and load explicitly implemented task releases.

Planned task definitions are design slots, never executable environments.
Loading verifies the frozen release and parent room without starting Isaac.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import re

from .registry import FAMILIES
from bench.paths import load_json
from bench.simulation.isaacsim_backend.workroom import checked, verify_reference

SCHEMA = 'workroom_task_definition.v1'
ROOT = Path(__file__).parent
RELEASE_SCHEMAS = frozenset({'robot_board_game_task_release.v1',
                           'study_task_release.v1',
                           'mechanical_assembly_task_release.v1',
                           'mechanical_nist_task_release.v1',
                           'mechanical_fastener_task_release.v1',
                           'card_room_placement_task_release.v1',
                           'living_room_task_release.v1',
                           'living_room_manipulation_task_release.v1',
                           'laboratory_manipulation_task_release.v1',
                           'kitchen_manipulation_task_release.v1'})


def _family(family):
    if family not in FAMILIES:
        raise ValueError(f'Unknown workroom family: {family}')
    return family


def scene_definition(family):
    """Return the family contract and its immutable environment reference."""
    family = _family(family)
    module = importlib.import_module(f'{__package__}.{family}.scene')
    result = deepcopy(module.SCENE)
    reference = module.environment_reference()
    if result.get('family') != family or reference.get('family') != family:
        raise ValueError('Scene definition differs from its parent family')
    result['environment'] = deepcopy(reference)
    return result


def task_definition(family, task_id):
    """Read metadata, including planned tasks, without loading their assets."""
    family = _family(family)
    if not isinstance(task_id, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task_id):
        raise ValueError('Unsafe task ID')
    if not (ROOT / family / 'tasks' / task_id / 'definition.py').is_file():
        raise ValueError(f'Unknown task: {family}/{task_id}')
    module = importlib.import_module(f'{__package__}.{family}.tasks.{task_id}.definition')
    result = deepcopy(module.TASK)
    required = {'schema', 'family', 'task_id', 'title', 'status', 'objects', 'layout',
                'robot_requirements', 'rules', 'evaluator', 'source_release',
                'qualification'}
    if not required <= result.keys() or result['schema'] != SCHEMA:
        raise ValueError('Incomplete task definition')
    if (result['family'], result['task_id']) != (family, task_id):
        raise ValueError('Task definition differs from its parent scene or directory')
    if result['status'] not in ('planned', 'implemented'):
        raise ValueError('Unknown task implementation status')
    if result['status'] == 'planned':
        if result['source_release'] is not None or result['evaluator'].get('implementation'):
            raise ValueError('Planned task cannot declare an executable release/evaluator')
        if not result.get('cannot_run_reason'):
            raise ValueError('Planned task must explain why it cannot run')
    elif not result['source_release'] or not result['evaluator'].get('implementation'):
        raise ValueError('Implemented task requires a frozen release and evaluator')
    elif result.get('release_schema') not in RELEASE_SCHEMAS:
        raise ValueError('Implemented task requires a supported release schema')
    result['runnable'] = result['status'] == 'implemented'
    return result


def list_tasks(family=None, *, include_planned=False):
    """List implemented tasks by default; opt in to unimplemented design slots."""
    families = (_family(family),) if family is not None else FAMILIES
    result = []
    for name in families:
        for path in sorted((ROOT / name / 'tasks').glob('*/definition.py')):
            task = task_definition(name, path.parent.name)
            if include_planned or task['runnable']:
                result.append(task)
    return sorted(result, key=lambda task: (FAMILIES.index(task['family']),
                  task.get('difficulty', 99), task['task_id']))


def list_scenes():
    return [scene_definition(family) for family in FAMILIES]


def load_task(family, task_id):
    """Load a verified frozen task; never turn a planned slot into a runtime.

    This checks frozen inputs and the canonical loader contract, not physics
    acceptance. Native task entrypoints still validate actors, contacts and
    cameras in the running simulator.
    """
    definition = task_definition(family, task_id)
    if not definition['runnable']:
        raise ValueError(f"Task {family}/{task_id} cannot run: {definition['cannot_run_reason']}")
    parent = scene_definition(family)
    reference = definition['source_release']
    root = Path(reference['path']).resolve()
    manifest_path = root / 'release_manifest.json'
    checked(manifest_path, reference['manifest_sha256'])
    manifest = load_json(manifest_path)
    expected_type = definition.get('release_schema')
    if expected_type not in RELEASE_SCHEMAS or manifest.get('schema') != expected_type:
        raise ValueError('Frozen release schema differs from the task definition or is unsupported')
    expected_schema = definition.get('release_schema_version')
    if expected_schema is not None and manifest.get('schema_version') != expected_schema:
        raise ValueError('Frozen release schema differs from the task definition')
    if manifest.get('task_id') != task_id:
        raise ValueError('Frozen release task ID differs from the task definition')
    hashes = manifest.get('files_sha256', {})
    if not {'scene.json', 'task.json', 'initial_state.json'} <= hashes.keys():
        raise ValueError('Frozen task release must hash scene, task and initial state')
    for relative, digest in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Frozen release file escapes its release directory')
        checked(path, digest)
    for path, digest in manifest.get('asset_files_sha256', {}).items():
        checked(path, digest)
    scene, task, initial = (load_json(root / name)
                            for name in ('scene.json', 'task.json', 'initial_state.json'))
    if task.get('task_id') != task_id:
        raise ValueError('Frozen task payload differs from the task definition')
    if manifest.get('schema') == 'robot_board_game_task_release.v1':
        from scenes.board_games.runtime_release import validate_board_game_release
        # Check inventory, geometry, ownership, reset and frozen evaluator code.
        validate_board_game_release(root, manifest, assets=False)
    independent = {'laboratory_manipulation_task_release.v1': 'isaacsim-laboratory-manipulation-release-v1',
                   'kitchen_manipulation_task_release.v1': 'isaacsim-kitchen-manipulation-release-v1',
                   'study_task_release.v1': 'isaacsim-study-release-v1',
                   'living_room_task_release.v1': 'isaacsim-living-room-release-v1',
                   'living_room_manipulation_task_release.v1': 'isaacsim-living-room-manipulation-release-v1',
                   'mechanical_fastener_task_release.v1': 'isaacsim-mechanical-fastener-release-v1',
                   'mechanical_assembly_task_release.v1': 'isaacsim-mechanical-assembly-release-v1',
                   'mechanical_nist_task_release.v1': 'isaacsim-mechanical-nist-release-v1',
                   'card_room_placement_task_release.v1': 'isaacsim-card-room-placement-release-v1'}
    if manifest.get('schema') in independent:
        if manifest.get('schema_version') != independent[manifest['schema']]:
            raise ValueError('Task catalog and canonical release schemas disagree')
        # Validate the complete frozen asset inventory before loading its scorer.
        from bench.isaacsim.release import load_evaluator, validate
        validate(root)
        load_evaluator(root)(task)
    actual = scene.get('workroom', {})
    expected = parent['environment']
    if actual != expected:
        raise ValueError('Frozen task environment differs from its fixed parent scene')
    verify_reference(actual)
    return {'definition': definition, 'environment': parent, 'scene': scene,
            'task': task, 'initial_state': initial, 'release_manifest': manifest}
