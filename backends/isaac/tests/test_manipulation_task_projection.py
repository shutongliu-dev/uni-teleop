import pytest

from bench.runtime.manipulation_tasks import project_manipulation_task


def task(task_id):
    return {'schema_version': 'living-room-manipulation-task-v1', 'id': task_id,
        'title_zh': '客厅任务', 'episode_seed': 7, 'steps': [{'id': 'act', 'instruction_zh': '执行任务'}],
        'public_goal': {'operation': task_id, 'frame_id': 'world_m_z_up', 'description': '公布目标',
                        'private_points': [[1, 2, 3]]},
        'public_rules': {'schema': 'living_room_' + task_id + '_rules_v2', 'competitive': False,
            'instruction_zh': '完成并松手', 'success': {'all_of': ['物理操作'], 'reset_credit': False,
                'private_state': {'raw': 1}, 'thresholds': {'maximum_linear_speed_m_s': .03, 'secret': 2}},
            'goal': {'target_water_ml': 100., 'private_raw': 'hidden'},
            'private_truth': {'points': [1, 2, 3]}},
        'initial_state': {'secret': 'reset'}, 'evaluation': {'native_state': 'private'}}


@pytest.mark.parametrize('task_id', ['remote_operation', 'water_pouring'])
def test_static_projection_excludes_unknown_fields_at_every_depth(task_id):
    value = project_manipulation_task(task(task_id))
    assert value['create_endpoint'] == 'POST /episodes'
    assert value['episode_seed'] == 7
    assert 'private' not in str(value)
    assert 'secret' not in str(value)
    assert 'evaluation' not in value


def test_bad_public_data_never_falls_back_to_evaluation():
    row = task('water_pouring')
    row['public_goal']['initial_water_ml'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        project_manipulation_task(row)
    row = task('remote_operation')
    row['public_rules']['competitive'] = True
    with pytest.raises(ValueError, match='identity'):
        project_manipulation_task(row)


def test_water_balance_definition_is_public_without_particle_state():
    from scenes.workrooms.living_room.tasks.water_pouring.rules import public_rules

    row = task('water_pouring')
    row['public_rules'] = public_rules()
    measurement = row['public_rules']['measurement']
    measurement['particle_positions_m'] = [[1., 2., 3.]]
    projected = project_manipulation_task(row)['public_rules']['measurement']
    assert projected['water_difference'] == measurement['water_difference']
    assert projected['conservation'] == measurement['conservation']
    assert 'particle_positions_m' not in projected


def test_television_goal_is_public_without_private_display_or_native_state():
    from scenes.workrooms.living_room.runtime_task import build_task_inputs

    row = build_task_inputs('remote_operation', seed=1)['task']
    row['task_state'] = {'private_native_state': {'power_on': True}}
    row['public_goal']['private_screen_path'] = '/World/PrivateScreen'
    projected = project_manipulation_task(row)
    assert projected['public_rules']['schema'] == 'physical_task_referee_contract.v1'
    assert projected['public_rules']['thresholds']['required_on_s'] == 10.
    assert set(projected['public_rules']['phases']) >= {'await_power_on', 'holding_power_on', 'await_power_off', 'await_return', 'completed'}
    assert '10秒' in projected['instruction']
    assert 'goal' not in projected
    assert 'private' not in str(projected)


def test_television_contract_cannot_be_changed_without_matching_frozen_hashes():
    from scenes.workrooms.living_room.runtime_task import build_task_inputs

    row = build_task_inputs('remote_operation')['task']
    row['public_rules']['referee']['thresholds']['required_on_s'] = 60.
    with pytest.raises(ValueError, match='frozen task identity'):
        project_manipulation_task(row)


def test_television_contract_cannot_downgrade_to_legacy_static_projection():
    from scenes.workrooms.living_room.runtime_task import build_task_inputs

    row = build_task_inputs('remote_operation')['task']
    del row['public_rules']['referee']
    with pytest.raises(ValueError, match='identity'):
        project_manipulation_task(row)
