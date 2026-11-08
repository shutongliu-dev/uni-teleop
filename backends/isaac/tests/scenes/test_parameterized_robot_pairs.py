"""Pair identity must catch real frozen-task changes, not just different labels."""
from copy import deepcopy
from pathlib import Path

import pytest

from scenes.workrooms.parameterized.pairs import (
    assert_same_task, canonical_hash, pair_record, physical_instance, read)
from scenes.workrooms.robot_revision.catalog import PINS


@pytest.fixture(params=['study/stationery_sorting', 'living_room/remote_operation',
                       'laundry_room/towel_flattening'])
def frozen_pair(request):
    row = read(PINS)['tasks'][request.param]
    key = next(iter(row['releases']))
    roots = [Path(row[group][key]['path']) for group in ('parents', 'releases')]
    manifests = [read(root/'release_manifest.json') for root in roots]
    values = [[read(root/manifest['cases']['1'][name])
               for name in ('scene', 'task', 'initial_state')]
              for root, manifest in zip(roots, manifests)]
    return roots, manifests, values, read(roots[1]/'robot_revision.json')


def test_actual_frozen_three_families_share_world_identity(frozen_pair):
    roots, _, values, revision = frozen_pair
    before = deepcopy(values)
    shared = assert_same_task(*values, revision)
    assert shared == canonical_hash(physical_instance(*values[0]))
    assert values == before
    record = pair_record(*roots, validate_assets=False)
    assert len(record['physical_case_sha256']) == 10
    assert record['physical_case_sha256']['1'] == shared


@pytest.mark.parametrize('mutation', ['world', 'physics', 'goal', 'reset', 'controller'])
def test_matching_seed_cannot_hide_task_or_controller_changes(frozen_pair, mutation):
    _, _, values, revision = frozen_pair
    after = deepcopy(values[1])
    if mutation == 'world':
        after[0]['workroom']['sha256'] = '0'*64
    elif mutation == 'physics':
        after[0]['physics']['gravity_m_s2'] = [0., 0., -4.]
    elif mutation == 'goal':
        after[1]['public_goal'] = {'different_goal': True}
    elif mutation == 'reset':
        after[1]['reset_case']['seed'] = 2
    else:
        after[0]['robot']['arm_kp'] *= 2
    with pytest.raises(ValueError, match='beyond robot'):
        assert_same_task(values[0], after, revision)


def test_different_seed_must_not_be_paired(frozen_pair):
    roots, manifests, values, revision = frozen_pair
    wrong = [read(roots[1]/manifests[1]['cases']['2'][name])
             for name in ('scene', 'task', 'initial_state')]
    with pytest.raises(ValueError, match='beyond robot'):
        assert_same_task(values[0], wrong, revision)


def test_explicit_robot_profiles_have_identical_task_inventory():
    from scenes.workrooms.catalog import list_tasks
    original = list_tasks(robot_profile='original')
    revised = list_tasks(robot_profile='l_mount')
    identity = lambda row: (row['family'], row['task_id'],
                            set(row.get('robot_configurations', {})))
    assert len(original) == len(revised) == 51
    assert [identity(row) for row in original] == [identity(row) for row in revised]
    for old, new in zip(original, revised):
        assert old['source_release'] != new['source_release']


def test_original_profile_agent_and_service_bind_the_same_menu():
    from agent.model_client.scene_tasks import select_task
    from bench.runtime.robot_configurations import discover_configuration_bindings
    selection = select_task('stationery_sorting', 'study', robot_profile='original')
    binding = discover_configuration_bindings(selection.source_release['path'])['stationery_sorting']
    assert selection.pin_release is True
    assert selection.robot_requirements['assembly_version'] == 'isaacsim_6_1_v2'
    assert binding['releases'] == {key: row['source_release']
                                   for key, row in selection.robot_configurations.items()}


def test_pair_cannot_be_moved_to_another_installation_option(tmp_path):
    import json
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.parameterized.catalog import _option
    from scenes.workrooms.parameterized.pairs import file_hash
    definition = task_definition('study', 'stationery_sorting', robot_profile='original')
    row = read(PINS)['tasks']['study/stationery_sorting']
    pair = pair_record(row['parents']['single_arm']['path'],
                       row['releases']['single_arm']['path'], validate_assets=False)
    path = tmp_path/'pair.json'
    path.write_text(json.dumps(pair))
    pin = {'path': str(path), 'sha256': file_hash(path)}
    with pytest.raises(ValueError, match='wrong physical task/robot pair'):
        _option(definition, definition['robot_configurations']['dual_opposed'], pin, 'original')
