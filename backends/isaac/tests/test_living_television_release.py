"""Frozen TV/return contracts; CPU replay does not qualify robot operation."""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import contract_sha256
from bench.runtime.manipulation_tasks import project_manipulation_task
from bench.simulation.isaacsim_backend.native_task_bundle import load_observer
from scenes.workrooms.living_room import runtime_task
from scenes.workrooms.living_room.release import prepare_task_release


RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v7')
MANIFEST_SHA256 = 'e47e15ce25746f76fa83c3b5111715158d905767a7fb935f36408ffb38e6dcc1'


def frozen_release():
    if not (RELEASE/'release_manifest.json').is_file():
        pytest.skip('Requires immutable remote_operation v7 dataset')
    assert digest(RELEASE/'release_manifest.json') == MANIFEST_SHA256
    return RELEASE


def test_seed_binding_uses_frozen_tv_rules_observer_and_referee(monkeypatch):
    root = frozen_release()
    manifest = load_json(root/'release_manifest.json')
    before = {name: digest(root/name) for name in [*manifest['files'], 'release_manifest.json']}

    def reject_live(*args, **kwargs):
        raise AssertionError('Frozen task used mutable live rules/evaluator')

    prefix = 'scenes.workrooms.living_room.tasks.remote_operation'
    rules = importlib.import_module(prefix+'.rules')
    evaluator = importlib.import_module(prefix+'.evaluator')
    monkeypatch.setattr(rules, 'public_rules', reject_live)
    monkeypatch.setattr(rules, 'evaluation_configuration', reject_live)
    monkeypatch.setattr(evaluator, 'SequenceEvaluator', reject_live)
    monkeypatch.setattr(runtime_task, 'PHYSICS_DT', 1.)
    evaluator_class = load_evaluator(root)
    assert Path(inspect.getfile(evaluator_class)).is_relative_to(root/'evaluator')
    frozen_rules = importlib.import_module(evaluator_class.__module__.rsplit('.', 1)[0]+'.rules')
    for seed in (0, 1, 2**32-1):
        _, scene, task, initial = task_inputs(root, 'remote_operation', seed=seed)
        assert task['evaluation'] == frozen_rules.evaluation_configuration(seed=seed,
            physics_dt_s=.0005, asset_metadata=task['asset_metadata'])
        assert task['public_rules'] == frozen_rules.public_rules(seed=seed)
        assert task['episode_seed'] == seed
        assert set(initial) == {'joint_positions_rad'}
        goal = task['public_goal']
        assert goal['operation'] == 'cycle_television_power'
        assert goal['initial_power_on'] is False and goal['required_on_s'] == 10.
        assert goal['target_id'] == 'holder' and goal['holder_return_required'] is True
        assert goal['stable_duration_s'] == .300 and 'button_sequence' not in goal
        assert task['public_rules_sha256'] == contract_sha256(task['public_rules'])
        public = project_manipulation_task(task)
        assert 'await_return' in public['public_rules']['phases']
        assert public['public_rules']['thresholds']['stable_s'] == .300
        assert task['public_referee_contract_sha256'] == public['public_rules_sha256']
        spec = deepcopy(scene['native_task_bundle']['observer'])
        spec['path'] = str(root/spec['path'])
        assert spec['configuration'] == task
        assert Path(inspect.getfile(type(load_observer(spec)))).is_relative_to(root/'evaluator')
        assert isinstance(evaluator_class(task), evaluator_class)
    assert before == {name: digest(root/name) for name in before}


def test_frozen_visual_binding_and_preparation_flags():
    root = frozen_release()
    report = validate(root, assets=False)
    assert report is not None
    manifest = load_json(root/'release_manifest.json')
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False and manifest['task_policy_success_claimed'] is False
    _, scene, task, _ = task_inputs(root, 'remote_operation')
    binding = scene['native_task_bundle']['visual_state_bindings']['bindings'][0]
    assert binding['state_path'] == ['television', 'power_on']
    assert binding['target_prim_path'] == '/World/Room/Geometry/television_screen/Visual'
    assert binding['true_material'] != binding['false_material']
    assert task['public_rules']['success']['holder_return_required'] is True
    with pytest.raises(ValueError, match='fresh living-room task release'):
        prepare_task_release('remote_operation', root)
