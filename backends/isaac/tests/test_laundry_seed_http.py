"""Published laundry wrappers through the real Agent/HTTP path, CPU worker only.

These checks intentionally require an independently prepared immutable wrapper;
no symbolic release or synthetic cloth state is promoted to native evidence.
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from bench.paths import load_json
from bench.protocol.laundry_reset import TASK_IDS
from bench.runtime.server import create_app
from scenes.workrooms.catalog import task_definition
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
import test_scene_function_runner as fixtures


@pytest.mark.parametrize('robot_profile', ('original', 'l_mount'))
@pytest.mark.parametrize('task_id', TASK_IDS)
@pytest.mark.parametrize('configuration', ('single_arm', 'dual_same_side', 'dual_opposed'))
def test_published_laundry_seed_binds_discovery_ready_and_every_function(
        tmp_path, task_id, configuration, robot_profile):
    definition = task_definition('laundry_room', task_id, configuration, robot_profile=robot_profile)
    release = Path(definition['source_release']['path'])
    manifest = load_json(release/'release_manifest.json')
    # The L-mount profile serves a robot-only revision of the same seed wrapper.
    revision = manifest.get('schema') == 'workroom_robot_revision_release.v1'
    if (manifest['parent_schema'] if revision else manifest.get('schema')) != 'laundry_randomized_release.v1':
        pytest.skip('Current catalog has not published this immutable laundry seed wrapper')
    if revision:
        parent = task_definition('laundry_room', task_id, configuration, robot_profile='original')
        assert manifest['base_release'] == {**parent['source_release'], 'schema': 'laundry_randomized_release.v1'}
    # The CPU worker fixture publishes the production worker's public projection,
    # including this reset case. Its unchanged frozen scorer reports unscored
    # reset; this fixture never supplies successful cloth/native evidence.
    api, workers, verified = fixtures.mechanical_service(
        tmp_path, task_id, family='laundry_room', release=release)
    seed = None if (task_id, configuration) == ('towel_flattening', 'dual_opposed') else 5
    effective = 0 if seed is None else seed
    model = ScriptedModel([{'function': 'get_task', 'arguments': {}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        # Invalid finite selections fail before creating a worker.
        for invalid in ('10', '-1', 'true', '1.0', '01'):
            assert http.get('/tasks', params={'task_id': task_id,
                'configuration_id': configuration, 'seed': invalid}).status_code == 422
        assert not workers and not api.sessions
        report = run(RunConfig('http://testserver', task_id, tmp_path/'run', family='laundry_room',
            configuration=configuration, seed=seed, max_decisions=3, robot_profile=robot_profile),
            bench=bench, models={'single': model})
        assert report['error_type'] is None, report
        assert report['run_status'] == 'completed'
        ready = load_json(tmp_path/'run/episode_ready.json')
        request = load_json(tmp_path/'run/create_request.json')
        discovery = load_json(tmp_path/'run/episode_case_discovery.json')
        assert request['seed'] == ready['spec']['seed'] == effective
        assert request['configuration_id'] == ready['spec']['configuration_id'] == configuration
        expected = discovery['reset_case']
        assert expected == ready['task']['reset_case']
        assert expected['seed'] == effective and expected['configuration_id'] == configuration
        assert expected['task_id'] == task_id and expected['parent_release_sha256']
        assert expected['case_sha256'] and expected['case_fingerprint']
        selected_task = load_json(release/manifest['cases'][str(effective)]['task'])
        assert ready['task']['goal'] == selected_task['public_goal']
        reply = http.get(f"/episodes/{report['episode_id']}/task")
        assert reply.status_code == 200 and reply.json()['reset_case'] == expected
    contexts = [json.loads(turns[-1]['content'][0]['text']) for turns in model.turns]
    assert len(contexts) == 2
    assert all(context['task']['reset_case'] == expected for context in contexts)
    assert contexts[1]['last_feedback']['result']['reset_case'] == expected
    assert PRIVATE_MARKER not in json.dumps(contexts)
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
    assert report['result']['outcome'] != 'success'
