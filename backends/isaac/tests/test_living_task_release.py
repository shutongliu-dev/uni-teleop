"""CPU contracts for immutable living-room v3 releases, not physics acceptance.

Corruptions below affect temporary copies only. The published inputs, shared
parent assets, and native simulation state are never modified by these tests.
"""
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import shutil

import pytest

from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from bench.runtime.manipulation_tasks import project_manipulation_task
from bench.simulation.isaacsim_backend.native_task_bundle import load_observer
from scenes.workrooms.living_room import runtime_task
from scenes.workrooms.living_room.release import initial_state
from scenes.workrooms.living_room.scene import environment_reference


RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room')
TASKS = ('remote_operation', 'water_pouring')


def release_path(task_id):
    root = RELEASES/task_id/'v3'
    if not (root/'release_manifest.json').is_file():
        pytest.skip('Requires the immutable living-room '+task_id+' v3 dataset')
    return root


def read_json(path):
    return load_json(path)


def observer_spec(root):
    spec = read_json(root/'scene.json')['native_task_bundle']['observer']
    spec['path'] = str(root/spec['path'])
    return spec


@pytest.mark.parametrize('task_id', TASKS)
def test_canonical_episode_uses_frozen_rules_seed_and_parent_assets(task_id, monkeypatch):
    root = release_path(task_id)
    manifest = read_json(root/'release_manifest.json')
    frozen_task = read_json(root/'task.json')
    frozen_initial = read_json(root/'initial_state.json')
    before = {name: digest(root/name) for name in [*manifest['files'], 'release_manifest.json']}

    def live_code_must_not_run(*args, **kwargs):
        raise AssertionError('A canonical release loaded mutable project rules/evaluator')

    prefix = 'scenes.workrooms.living_room.tasks.'+task_id
    live_rules = importlib.import_module(prefix+'.rules')
    live_evaluator = importlib.import_module(prefix+'.evaluator')
    monkeypatch.setattr(live_rules, 'evaluation_configuration', live_code_must_not_run)
    monkeypatch.setattr(live_rules, 'public_rules', live_code_must_not_run)
    monkeypatch.setattr(live_evaluator, 'SequenceEvaluator', live_code_must_not_run)
    monkeypatch.setattr(runtime_task, 'PHYSICS_DT', 1.)

    evaluator_class = load_evaluator(root)
    assert evaluator_class.__module__.startswith('_isaac_frozen_evaluator_')
    assert Path(inspect.getfile(evaluator_class)).read_bytes() == (root/'evaluator/sequence_evaluator.py').read_bytes()
    frozen_rules = importlib.import_module(evaluator_class.__module__.rsplit('.', 1)[0]+'.rules')
    for seed in (0, 1):
        entry, scene, task, initial = task_inputs(root, task_id, seed=seed)
        assert entry['id'] == task_id and task['episode_seed'] == seed
        assert initial == frozen_initial == frozen_task['initial_state']
        assert set(initial) == {'joint_positions_rad'}
        assert task['evaluation'] == frozen_rules.evaluation_configuration(
            seed=seed, physics_dt_s=frozen_task['evaluation']['physics_dt_s'],
            asset_metadata=frozen_task['asset_metadata'])
        assert task['evaluation']['physics_dt_s'] == scene['isaacsim']['physics_dt_s'] == .0005
        expected_rules = frozen_rules.public_rules(seed=seed) if task_id == 'remote_operation' else frozen_rules.public_rules()
        assert task['public_rules'] == expected_rules
        if task_id == 'remote_operation':
            expected = ['power', 'confirm'] if seed == 0 else ['confirm', 'power']
            assert task['public_goal']['button_sequence'] == task['evaluation']['button_sequence'] == expected
        else:
            assert task['evaluation']['seed'] == seed
            assert task['public_goal']['target_interval_ml'] == [90., 110.]
        spec = scene['native_task_bundle']['observer']
        assert spec['configuration'] == task
        assert Path(spec['path']).is_relative_to(root/'evaluator')
        observer = load_observer(spec)
        assert Path(inspect.getfile(type(observer))).is_relative_to(root/'evaluator')
        assert isinstance(evaluator_class(task), evaluator_class)

        # task_inputs performs full dependency-hash validation, including the
        # parent room; neither task is allowed to rebuild or replace that room.
        reference = scene['workroom']
        assert reference == environment_reference()
        dependencies = read_json(Path(reference['dependency_manifest_path']))['dependencies']
        expected_assets = {reference['usd_path']: reference['usd_sha256'],
            reference['dependency_manifest_path']: reference['dependency_manifest_sha256'],
            **{path: row['sha256'] for path, row in dependencies.items()}}
        assert all(manifest['asset_files_sha256'][path] == sha for path, sha in expected_assets.items())
        assert scene['living_room_provenance']['parent_environment_unchanged'] is True
    assert before == {name: digest(root/name) for name in before}


@pytest.mark.parametrize('task_id', TASKS)
def test_public_projection_keeps_goals_and_discards_private_state(task_id):
    task = read_json(release_path(task_id)/'task.json')
    projected = project_manipulation_task(task)
    assert projected['episode_seed'] == task['episode_seed']
    assert projected['goal'] == task['public_goal']
    assert projected['public_rules']['instruction_zh'] == task['public_rules']['instruction_zh']
    if task_id == 'water_pouring':
        assert projected['goal']['target_interval_ml'] == [90., 110.]
        assert projected['goal']['max_spill_ml'] == 5.
    else:
        assert projected['goal']['button_sequence'] == ['power', 'confirm']

    private = 'private-native-pose-particle-and-path-sentinel'
    changed = deepcopy(task)
    changed['task_state'] = {'private': private}
    changed['evaluation']['private'] = private
    changed['asset_metadata']['private'] = private
    changed['public_goal']['private'] = private
    changed['public_rules']['private'] = private
    changed['public_rules']['success']['private'] = private
    thresholds = (changed['public_rules']['thresholds'] if task_id == 'remote_operation'
                  else changed['public_rules']['success']['thresholds'])
    thresholds['private'] = private
    assert project_manipulation_task(changed) == projected
    assert private not in json.dumps(projected)
    assert not {'evaluation', 'asset_metadata', 'task_state', 'initial_state'} & projected.keys()


@pytest.mark.parametrize('task_id', TASKS)
def test_reset_contains_only_robot_joints_and_matching_parent_observer(task_id):
    root = release_path(task_id)
    scene, task = read_json(root/'scene.json'), read_json(root/'task.json')
    assert initial_state(scene, task) == read_json(root/'initial_state.json')
    changed = deepcopy(task)
    changed['initial_state']['objects'] = {'task_object': {'position_m': [0., 0., 0.]}}
    with pytest.raises(ValueError, match='only robot joint coordinates'):
        initial_state(scene, changed)
    changed = deepcopy(scene)
    changed['workroom']['usd_sha256'] = '0'*64
    with pytest.raises(ValueError, match='task/parent identity differs'):
        initial_state(changed, task)
    changed = deepcopy(scene)
    changed['native_task_bundle']['observer']['configuration']['episode_seed'] += 1
    with pytest.raises(ValueError, match='observer and referee task differ'):
        initial_state(changed, task)


def test_observer_helper_hash_and_complete_inventory_are_checked(tmp_path):
    root = tmp_path/'water'
    shutil.copytree(release_path('water_pouring'), root)
    spec = observer_spec(root)
    assert callable(load_observer(spec).on_observe)

    helper = root/'evaluator/rules.py'
    original = helper.read_bytes()
    helper.write_bytes(original+b'\n# changed only in disposable test copy\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_evaluator(root)
    with pytest.raises(ValueError, match='dependency hash differs'):
        load_observer(spec)
    helper.write_bytes(original)

    incomplete = deepcopy(spec)
    incomplete['files'].pop('rules.py')
    with pytest.raises(ValueError, match='undeclared or missing Python dependencies'):
        load_observer(incomplete)

    extra = root/'evaluator/undeclared_helper.py'
    extra.write_text('raise AssertionError("Undeclared code must never execute")\n')
    with pytest.raises(ValueError, match='Observer directory has undeclared or missing Python dependencies'):
        validate(root, assets=False)
    with pytest.raises(ValueError, match='Observer package has undeclared or missing Python dependencies'):
        load_observer(spec)


def test_manifest_cannot_redirect_or_reidentify_frozen_physics_profile(tmp_path):
    root = tmp_path/'water'
    shutil.copytree(release_path('water_pouring'), root)
    path = root/'release_manifest.json'
    manifest = read_json(path)
    for field, value in [('physics_profile', 'unvalidated_profile.json'),
                         ('physics_profile_sha256', '0'*64)]:
        changed = deepcopy(manifest)
        changed[field] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='Frozen physics profile reference differs'):
            validate(root, assets=False)
