"""Published static success contracts match scoring without exposing live evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bench.paths import DATA_ROOT, load_json
from scenes.manipulation.contracts import project_success_contract, project_grasp_contract


TASKS = ('01_cube_push', '02_cube_lift_hold', '03_cube_pick_place', '04_cube_grasp_hold')
RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation')


def source_task(name, *, version='v7_contact_substeps'):
    return load_json(RELEASES / version / 'tasks' / name / 'task.json')


@pytest.mark.parametrize('name', TASKS)
def test_success_contract_matches_frozen_scoring_and_projection_preserves_source(name):
    task = source_task(name)
    original = deepcopy(task)
    c, public = task['evaluation'], task['public_goal']['success_contract']
    assert project_success_contract(public) == public
    assert task == original
    assert public['operation'] == c['mode']
    assert public['terminal']['hold_s'] == c['hold_s']
    assert public['terminal']['maximum_linear_speed_m_s'] == c['maximum_speed_m_s']
    assert public['terminal']['tilt_definition'] == 'angle_between_object_local_positive_z_and_world_positive_z'
    assert public['interaction']['minimum_hand_contact_duration_s'] == c['minimum_contact_duration_s']
    assert public['protocol'] == 'simple_manipulation_success_v2'
    assert public['interaction']['physics_dt_s'] == c['physics_dt_s'] == .0005
    assert public['interaction']['duration_semantics'] == 'unique_positive_hand_contact_substeps_times_physics_dt'
    assert public['physical_validity']['maximum_reported_rigid_penetration_m'] == .002
    assert c['humanlike_grasp']['maximum_robot_penetration_m'] == .002
    assert task['public_goal']['grasp_constraints']['criteria']['maximum_robot_penetration_m'] == .002
    assert public['terminal']['all_robot_contact_absent_required'] == (c['mode'] in ('push', 'pick_place'))
    assert public['terminal']['table_support_absent_required'] == (c['mode'] == 'lift_hold')
    assert public['stages']['humanlike_violation_is_irreversible'] == (c['mode'] != 'grasp_hold')
    assert 'initial_position_m' not in json.dumps(public)
    assert 'hand_urdf' not in json.dumps(public)
    if c['mode'] == 'grasp_hold':
        assert public['interaction']['minimum_displacement_from_initial_m'] == 0
        assert public['episode_constraints']['maximum_displacement_from_initial_m'] == .03
        assert public['episode_constraints']['maximum_lift_from_initial_m'] == .006
    if c['mode'] == 'push':
        assert public['episode_constraints']['maximum_lift_from_initial_m'] == c['minimum_lift_m'] / 2


def test_static_contract_recursively_discards_unknown_private_keys():
    contract = source_task(TASKS[0])['public_goal']['success_contract']
    value = deepcopy(contract)
    for row in [value, *[v for v in value.values() if isinstance(v, dict)]]:
        row['private_live_pose'] = 'oracle_canary'
    assert project_success_contract(value) == contract
    assert 'oracle_canary' not in json.dumps(project_success_contract(value))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, '0.3', 0., -.1])
def test_static_contract_rejects_invalid_dwell(invalid):
    value = source_task(TASKS[0])['public_goal']['success_contract']
    value['terminal']['hold_s'] = invalid
    with pytest.raises(ValidationError):
        project_success_contract(value)


@pytest.mark.parametrize('field,value', [('non_hand_robot_contact_is_irreversible', 1),
                                        ('minimum_hand_contact_intervals', 2.),
                                        ('minimum_hand_contact_intervals', True)])
def test_literal_contract_fields_require_exact_json_types(field, value):
    contract = source_task(TASKS[0])['public_goal']['success_contract']
    contract['interaction'][field] = value
    with pytest.raises(ValidationError):
        project_success_contract(contract)


@pytest.mark.parametrize('nested', [False, True])
def test_grasp_contract_does_not_copy_nested_private_payloads_through_allowed_fields(nested):
    contract = source_task(TASKS[0])['public_goal']['grasp_constraints']
    target, field = (contract['criteria'], 'hold_s') if nested else (contract, 'description')
    target[field] = {'live_object_pose': 'PRIVATE_SENTINEL'}
    with pytest.raises(ValidationError):
        project_grasp_contract(contract)


def test_new_contract_survives_public_tasks_projection(tmp_path, monkeypatch):
    from test_public_task_goals import public_inventory, PRIVATE_MARKER
    from bench.isaacsim import release
    task = source_task(TASKS[3])
    expected = deepcopy(task['public_goal'])
    contract = task['public_goal']['success_contract']
    contract['private_score'] = PRIVATE_MARKER
    contract['terminal']['live_object_pose'] = PRIVATE_MARKER
    (tmp_path / 'task.json').write_text(json.dumps(task))
    monkeypatch.setattr(release, 'validate', lambda root, assets=False: {'tasks': [{'id': task['id'], 'task': 'task.json'}]})
    rows = public_inventory(tmp_path)
    assert rows[0]['goal'] == expected
    assert PRIVATE_MARKER not in json.dumps(rows)


def test_historical_v1_public_contract_projects_without_new_optional_fields():
    value = source_task(TASKS[0], version='v6_public_control')['public_goal']['success_contract']
    assert value['protocol'] == 'simple_manipulation_success_v1'
    assert project_success_contract(value) == value


@pytest.mark.parametrize('field', ['duration_semantics', 'physics_dt_s', 'contact_sampling'])
def test_v2_contract_cannot_omit_contact_substep_semantics(field):
    value = source_task(TASKS[0])['public_goal']['success_contract']
    del value['interaction'][field]
    with pytest.raises(ValidationError, match='requires explicit substep'):
        project_success_contract(value)
