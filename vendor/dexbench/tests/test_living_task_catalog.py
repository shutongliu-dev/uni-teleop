"""Published bundle tasks and independently pinned tower-layer candidates."""
import importlib

import pytest

from scenes.workrooms.catalog import list_tasks, load_task
from scenes.workrooms.living_room.tasks.remote_operation.task import task_configuration as remote_plan
from scenes.workrooms.living_room.tasks.water_pouring.task import task_configuration as pouring_plan


def test_four_numbered_tasks_and_no_duplicate_legacy_slots():
    rows = list_tasks('living_room', include_planned=True)
    assert [(x['task_number'], x['task_id']) for x in rows] == [
        (1, 'remote_operation'), (2, 'water_pouring'), (3, 'tower_building'), (4, 'tower_extraction')]
    assert [x['robot_requirements']['articulations'] for x in rows] == [1, 1, 1, 2]
    for task in rows:
        front_y = -.88 if task['task_id'] == 'tower_building' else -.76
        assert task['robot_requirements']['base_positions_m']['front'] == [.65, front_y, .78]
        assert task['status'] == 'implemented' and task['runnable'] is True
        assert task['qualification']['manipulation'] is False
        assert task['qualification']['physics_accepted'] is False
    for task in rows[:2]:
        assert task['status'] == 'implemented' and task['runnable'] is True
        version = 'v7' if task['task_id'] == 'remote_operation' else 'v5'
        assert task['source_release']['path'].endswith('/'+task['task_id']+'/'+version)
        assert task['qualification']['manipulation'] is False
        assert task['qualification']['native_preflight'] is True
        assert task['qualification']['status'] == 'candidate_native_smoke_passed'
    for task in rows[2:]:
        assert task['source_release']
        assert task['release_schema'] == 'living_room_task_release.v1'
        assert task['qualification']['release_status'] == 'prepared_unvalidated'
    assert len(list_tasks('living_room')) == 4


@pytest.mark.parametrize('task_id', ['remote_operation', 'water_pouring'])
def test_catalog_loads_and_verifies_frozen_candidate_without_claiming_robot_success(task_id):
    loaded = load_task('living_room', task_id)
    assert loaded['definition']['runnable']
    assert loaded['task']['id'] == task_id
    assert loaded['task']['schema_version'] == 'living-room-manipulation-task-v1'
    assert loaded['scene']['workroom'] == loaded['environment']['environment']
    assert loaded['scene']['robot']['base_position_m'] == [.65, -.76, .78]
    assert len(loaded['initial_state']['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert loaded['release_manifest']['task_policy_success_claimed'] is False


@pytest.mark.parametrize('task_id', ['tower_building', 'tower_extraction'])
def test_tower_layer_candidates_keep_their_frozen_parent_reset_and_scorer(task_id):
    loaded = load_task('living_room', task_id)
    scene, task = loaded['scene'], loaded['task']
    assert scene['workroom'] == loaded['environment']['environment']
    assert loaded['initial_state'] == task['initial_state']
    assert task['schema_version'] == 'living-room-task-v1'
    assert loaded['release_manifest']['schema'] == 'living_room_task_release.v1'
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert loaded['release_manifest']['physics_accepted'] is False
    assert loaded['definition']['task_kind'] == loaded['release_manifest']['task_kind']
    assert len(loaded['initial_state']['joint_positions_rad']) == (58 if task_id == 'tower_extraction' else 29)
    assert scene.get('native_readback_profile', 'cuda_tensors_v1') == 'cuda_tensors_v1'


def test_default_configurations_load_real_frozen_inputs_and_preserve_public_fields():
    for configuration in (remote_plan(seed=0), pouring_plan(seed=1)):
        assert configuration['environment']['version'] == 'living_room_residential_20261018_v7'
        assert len(configuration['initial_state']['joint_positions_rad']) == 29
        assert configuration['source_release'] and configuration['native_execution_available'] is True
        assert configuration['scene']['native_task_bundle']['observer']['configuration'] == configuration['task']
        assert configuration['task']['episode_seed'] == configuration['seed']
        assert configuration['rules'] == configuration['task']['public_rules']
        assert configuration['public_goal'] == configuration['task']['public_goal']
    for seed in (0, 1):
        goal = remote_plan(seed=seed)['public_goal']
        assert goal['operation'] == 'cycle_television_power'
        assert goal['required_on_s'] == 10.
        assert goal['initial_power_on'] is False
        assert 'button_sequence' not in goal
    result = remote_plan()
    result['rules']['buttons']['ids'].clear()
    assert remote_plan()['rules']['buttons']['ids'] == ['power', 'confirm']


@pytest.mark.parametrize('seed', [-1, True, 2**32, 1.5])
def test_bad_seeds_are_rejected(seed):
    for factory in (remote_plan, pouring_plan):
        with pytest.raises(ValueError, match='seed'):
            factory(seed=seed)


def test_tower_catalog_entrypoints_bind_real_reference_code_and_same_parent():
    from scenes.workrooms.living_room.scene import environment_reference

    for definition in list_tasks('living_room', include_planned=True)[2:]:
        factory = importlib.import_module(definition['entrypoint']).task_configuration
        with pytest.raises(ValueError, match='block_size_m'):
            factory({})
        # A synthetic CPU fixture only; this does not choose published asset dimensions.
        plan = factory({'block_size_m': [.12, .035, .02]})
        reference = definition['evaluator']['reference_implementation']
        module, name = reference.rsplit('.', 1)
        evaluator_class = getattr(importlib.import_module(module), name)
        evaluator_class(plan)
        assert plan.get('environment', plan.get('parent_environment')) == environment_reference()
        assert plan['status'] == 'planned' and plan['source_release'] is None
        assert plan['native_execution_available'] is False
        resized = factory({'block_size_m': [.13, .035, .02]})
        assert resized['configuration_sha256'] != plan['configuration_sha256']
