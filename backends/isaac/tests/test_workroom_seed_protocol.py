"""Public identity/geometry boundaries, with synthetic CPU-only transports."""
from copy import deepcopy

import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.scene_tasks import TaskSelection, TaskSetupError
from agent.model_client.seeded_tasks import checked_reset_case, select_episode_case
from bench.protocol.task_referee import project_task_rules
from bench.protocol.workroom_reset import PublicTaskRequirements, PublicWorkroomResetCase, SCOPE, TASK_FAMILIES
from scenes.workrooms.initial_randomization.common import canonical_hash
from scenes.workrooms.initial_randomization.public import project_reset_case
from test_task_referee_contract import declared_task


HASH = 'b'*64


def private_task(task_id='parts_sorting', seed=0):
    task = declared_task(task_id)
    configuration = 'dual_same_side' if task_id == 'cooperative_insertion' else 'single_arm'
    case = {'schema': 'workroom_reset_case.v1', 'family': TASK_FAMILIES[task_id],
            'task_id': task_id, 'seed': seed, 'configuration_id': configuration,
            'static_geometry': {'bodies': [{'object_id': 'object_a',
                'local_bounds_m': [[-.02, -.02, 0.], [.02+seed*.001, .02, .04]],
                'nominal_mass_kg': .02+seed*.001}]},
            'actual_body_poses_m': {'object_a': {'private_pose_marker': [1, 2, 3]}},
            'interaction_targets': [{'private_interaction_marker': True}]}
    case['case_sha256'] = canonical_hash(case)
    task['reset_case'] = case
    return task


def published(task_id='parts_sorting', seed=0):
    task = private_task(task_id, seed)
    result = project_reset_case(task, project_task_rules(task))
    return {**result, 'configuration_id': task['reset_case']['configuration_id'], 'task_release_sha256': HASH}


def live_task(value):
    result = {k: v for k, v in value.items() if k not in ('configuration_id', 'task_release_sha256')}
    result.update(referee={'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
        'phase': 'place', 'reason_codes': ['release_required'], 'terminal': False, 'recoverable': True},
        episode={'terminal': False, 'outcome': None})
    return result


class PublicBench:
    def __init__(self, value):
        self.value, self.calls = value, []

    def list_tasks(self, **kwargs):
        self.calls.append(kwargs)
        return deepcopy([self.value])


@pytest.mark.parametrize('task_id', TASK_FAMILIES)
def test_all_tasks_serialize_same_seed_public_contract_without_private_poses(task_id):
    value = published(task_id, 7)
    parsed = PublicWorkroomResetCase.model_validate(value['reset_case'])
    assert parsed.task_id == task_id and parsed.family == TASK_FAMILIES[task_id]
    assert parsed.scope == SCOPE and parsed.seed == 7
    typed = PublicTaskRequirements.model_validate(live_task(value))
    assert typed.model_dump(mode='json')['reset_case'] == value['reset_case']
    assert 'private_pose_marker' not in str(value) and 'private_interaction_marker' not in str(value)


@pytest.mark.parametrize('task_id', TASK_FAMILIES)
@pytest.mark.parametrize('seed', [None, 0, 3, 9])
def test_agent_discovers_exact_task_seed_before_episode_creation(task_id, seed):
    effective = 0 if seed is None else seed
    baseline, selected = published(task_id), published(task_id, effective)
    bench = PublicBench(selected)
    configuration = baseline['configuration_id']
    selection = TaskSelection(TASK_FAMILIES[task_id], task_id, pin_release=True,
                              source_release={'manifest_sha256': HASH}, configuration_id=configuration)
    actual, actual_seed = select_episode_case(bench, selection, baseline, seed)
    assert actual == selected and actual_seed == effective
    assert bench.calls == [{'task_id': task_id, 'configuration_id': configuration, 'seed': effective}]
    assert baseline == published(task_id)


@pytest.mark.parametrize('change', [
    lambda v: v.update(task_release_sha256='c'*64),
    lambda v: v['reset_case'].update(seed=0),
    lambda v: v['reset_case'].update(task_id='sample_sorting', family='laboratory'),
    lambda v: v['reset_case'].update(configuration_id='dual_same_side'),
    lambda v: v['reset_case'].update(actual_body_poses_m={'private': [1, 2, 3]}),
    lambda v: v['reset_case']['static_geometry']['bodies'][0].update(position_m=[1, 2, 3]),
    lambda v: v.pop('reset_case'),
])
def test_discovery_identity_or_private_field_drift_is_rejected(change):
    value = published(seed=1); change(value)
    selection = TaskSelection('mechanical_assembly', 'parts_sorting', pin_release=True,
                              source_release={'manifest_sha256': HASH}, configuration_id='single_arm')
    with pytest.raises((BenchHTTPError, TaskSetupError)):
        select_episode_case(PublicBench(value), selection, published(), 1)


@pytest.mark.parametrize('seed', [True, False, -1, 10, 1.0, '1'])
def test_out_of_pool_seed_is_rejected_before_transport(seed):
    bench = PublicBench(published(seed=1))
    with pytest.raises(ValueError):
        select_episode_case(bench, TaskSelection('mechanical_assembly', 'parts_sorting'), published(), seed)
    assert not bench.calls


def test_kitchen_cannot_claim_an_unpublished_dual_installation():
    value = published('produce_transfer', 1)['reset_case']
    value['configuration_id'] = 'dual_same_side'
    with pytest.raises(BenchHTTPError):
        checked_reset_case(value, seed=1, task_id='produce_transfer')


@pytest.mark.parametrize('configuration', ['single_arm', 'dual_opposed'])
def test_cooperative_insertion_cannot_claim_a_new_installation(configuration):
    value = published('cooperative_insertion', 1)['reset_case']
    value['configuration_id'] = configuration
    with pytest.raises(BenchHTTPError):
        checked_reset_case(value, seed=1, task_id='cooperative_insertion')


def test_private_hash_tamper_and_public_task_mismatch_are_rejected():
    task = private_task(seed=1)
    task['reset_case']['seed'] = 2
    with pytest.raises(ValueError, match='identity'):
        project_reset_case(task, project_task_rules(task))
    value = published(seed=1)
    value['task_id'] = 'sample_sorting'
    with pytest.raises(ValueError):
        PublicTaskRequirements.model_validate(live_task(value))
