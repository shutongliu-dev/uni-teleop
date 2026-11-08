"""Task8/9 observable requirements survive the real Bench/Agent public path.

The HTTP tests load the catalog-selected immutable releases and replace only
the native worker/model. These are publication regressions, not physics tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run, task_requirements_context
from bench.isaacsim.release import digest, task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import project_task_rules
from bench.protocol.workroom_reset import PublicWorkroomResetCase
from bench.runtime.server import create_app
from bench.simulation.isaacsim_backend.l_mount import public_mount
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.mechanical_assembly import nist_rules
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


TASKS = ((8, 'nist_cable_routing'), (9, 'nist_wire_harness'))
RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly')
HISTORICAL_IDENTITIES = {
    'nist_cable_routing': (
        '1bc7b939a15b1a046cf2c3f29da002785dbf6510a1ddca485c05c87e739772c9',
        'e6bdb46e644ce4a559ba44c2a7fcf4970994b6800f1a2a340f54761202cbeb57'),
    'nist_wire_harness': (
        '7dc16111e5fb97d7502681a2e743e70c47039ab6d7886e6aab2a5e38775c3106',
        '960b5c238a428f63b7a330d2b6facbf00b5cba324c1ee7d6d32a651432f05bdc'),
}


def assert_observable_requirements(task_id, public):
    """Check user-observable semantics rather than a full wording snapshot."""
    contract = public['public_rules']
    text = '\n'.join((public['instruction'], *contract['success_conditions']))
    assert public['instruction'] == contract['instruction_zh']
    if task_id == 'nist_cable_routing':
        assert re.search(r'黑色[^\n。；]*方块', text)
        assert '音频插头' in text
        assert text.index('黑色') < text.index('音频插头')
        assert '绿色' in text
        yellow = next((cue for cue in ('黄色', '琥珀色') if cue in text), None)
        assert yellow is not None
        assert text.index('绿色') < text.index(yellow)
        assert '最终' in text and any(cue in text for cue in ('路径', '路线'))
        assert '操作' in text
        assert any(cue in text for cue in ('自由', '不限', '不要求'))
    else:
        assert '红' in text and '蓝' in text and '同色' in text
        assert any(cue in text for cue in ('互换', '任意一端', '任一端', '掉头'))
        assert '两端' in text
        assert any(cue in text for cue in ('两个', '两座', '两处'))
        assert '确认' in text and '收集托盘' in text
    private = (PRIVATE_MARKER, '/World/NistTask', 'asset_metadata', 'task_parameters',
               'route_gates', 'target_position_m', 'diagnostic_goal_poses')
    declared = json.dumps({key: value for key, value in public.items() if key != 'reset_case'},
                          ensure_ascii=False)
    # Instructions and rules describe the harness ends by colour only.
    for forbidden in (*private, 'red_end_a', 'red_end_b', 'blue_end_a', 'blue_end_b'):
        assert forbidden not in declared
    # A frozen seed case publishes only schema-checked object-local body bounds
    # and nominal masses keyed by dynamic object ID (the documented public
    # static geometry); it can carry no pose or private evaluator field.
    if 'reset_case' in public:
        case = PublicWorkroomResetCase.model_validate(public['reset_case'])
        assert case.task_id == task_id
        exposed_case = json.dumps(public['reset_case'], ensure_ascii=False)
        assert not any(forbidden in exposed_case for forbidden in private)
    # Static instructions identify appearance, not undeclared exact goal poses.
    assert not re.search(r'\[(?:\s*-?\d+\.\d+\s*,){2}\s*-?\d+\.\d+\s*\]', declared)


@pytest.mark.parametrize('number,task_id', TASKS)
def test_authored_requirements_project_to_unchanged_agent_static_context(number, task_id):
    task = nist_rules.bind_task_rules({
        'id': task_id, 'public_rules': nist_rules.public_rules(task_id),
        'asset_metadata': {'target_position_m': [941.25, 942.5, 943.75], PRIVATE_MARKER: True},
        'evaluation': {'route_gates': PRIVATE_MARKER},
        'public_goal': {'diagnostic_goal_poses': PRIVATE_MARKER},
    })
    public = project_task_rules(task)
    assert_observable_requirements(task_id, public)
    ready = {**deepcopy(public),
             'referee': {'phase': 'working'},
             'episode': {'terminal': False, 'outcome': None},
             'scoring': None}
    before = deepcopy(ready)
    static = task_requirements_context(ready)
    assert static == public
    assert ready == before
    static['public_rules']['success_conditions'].append('local copy only')
    assert ready == before


@pytest.mark.parametrize('profile', ('original', 'l_mount'))
@pytest.mark.parametrize('number,task_id', TASKS)
def test_published_requirements_reach_discovery_get_task_and_agent_context(tmp_path, number, task_id, profile):
    # Each physical robot profile publishes its own release of the same task.
    release = task_definition('mechanical_assembly', task_id, robot_profile=profile)['source_release']
    api, workers, verified = mechanical_service(tmp_path, task_id, release=release['path'])
    _, scene, discovered, _ = task_inputs(release['path'], task_id)
    expected = project_task_rules(discovered)
    assert expected == project_task_rules(verified['task'])
    # A source-only wording fix is insufficient: this is the selected release.
    assert_observable_requirements(task_id, expected)
    model = ScriptedModel([{'function': 'get_task', 'arguments': {}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        listing = bench.list_tasks()
        assert len(listing) == 1
        # Discovery adds the frozen seed-0 case and its installation identity.
        identity = {'robot_configurations', 'default_configuration_id', 'scene_description_zh',
                    'reset_case', 'configuration_id', 'task_release_sha256', 'robot_mount'}
        assert {key: value for key, value in listing[0].items() if key not in identity} == expected
        assert set(listing[0]['robot_configurations']) == {
            'single_arm', 'dual_same_side', 'dual_opposed'}
        assert listing[0]['configuration_id'] == listing[0]['reset_case']['configuration_id'] == 'single_arm'
        assert listing[0]['reset_case']['seed'] == 0
        assert listing[0]['task_release_sha256'] == release['manifest_sha256']
        assert listing[0].get('robot_mount') == public_mount(scene['robot'])
        assert ('robot_mount' in listing[0]) == (profile == 'l_mount')
        report = run(RunConfig('http://testserver', f'task{number}', tmp_path/'run',
                              family='mechanical_assembly', configuration='single_arm', sim_budget=None,
                              max_decisions=2, max_status_queries=100000, robot_profile=profile),
                     bench=bench, models={'single': model})
    assert report['error_type'] is None, report
    assert report['task_selection']['task_id'] == task_id
    assert report['task_selection']['players'] == ['single']
    assert report['ready_validated'] and report['result']['outcome'] == 'cancelled'
    ready = load_json(tmp_path/'run/episode_ready.json')
    assert ready['spec']['task_release_sha256'] == release['manifest_sha256']
    contexts = [json.loads(turns[-1]['content'][0]['text']) for turns in model.turns]
    assert len(contexts) == 2
    for context in contexts:
        assert context['task'] == task_requirements_context(ready['task'])
        assert context['task']['public_rules'] == expected['public_rules']
        assert context['task']['public_rules_sha256'] == expected['public_rules_sha256']
        assert not {'referee', 'episode', 'scoring'} & context['task'].keys()
        assert_observable_requirements(task_id, context['task'])
    feedback = contexts[-1]['last_feedback']
    assert feedback['ok'] and feedback['function'] == 'get_task'
    assert feedback['result']['public_rules'] == expected['public_rules']
    assert feedback['result']['referee']['state'] == 'ongoing'
    assert_observable_requirements(task_id, feedback['result'])
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)


@pytest.mark.parametrize('number,task_id', TASKS)
def test_historical_desktop_contracts_remain_immutable_and_loadable(number, task_id):
    root = RELEASES/task_id/'desktop_v1'
    manifest_hash, contract_hash = HISTORICAL_IDENTITIES[task_id]
    assert digest(root/'release_manifest.json') == manifest_hash
    assert validate(root)['physics_accepted'] is False
    _, _, task, _ = task_inputs(root, task_id, seed=17)
    assert task['public_rules']['schema'] == 'mechanical_nist_rules.v1'
    assert project_task_rules(task)['public_rules_sha256'] == contract_hash


@pytest.mark.parametrize('task_id', ('nist_connector_insertion', 'nist_belt_drive_assembly'))
def test_other_nist_contracts_are_not_revised_by_routing_clarification(task_id):
    historical = load_json(RELEASES/task_id/'desktop_v1'/'task.json')
    assert nist_rules.public_rules(task_id) == historical['public_rules']
