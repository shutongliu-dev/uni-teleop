"""Causal online-loop tests; fake servers/models are not physics acceptance."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import numpy as np

from PIL import Image
import pytest

from agent.model_client import runner
from bench.paths import load_json
from bench.protocol.schemas import EpisodeResult


def fixtures():
    sha = 'a'*64
    caps = {'robot_id': 'robot', 'robot_asset_sha256': sha, 'arm_joints': ['arm'],
        'hand_joints': ['finger'], 'joint_limits': {n: {'lower_rad': -2., 'upper_rad': 2.}
        for n in ('arm', 'finger')}}
    spec = {'episode_id': 'episode', 'agent_id': 'model', 'agent_version': 'v1',
        'agent_config_sha256': sha, 'task_id': 'task', 'task_release_sha256': sha,
        'evaluator_sha256': sha, 'robot_id': 'robot', 'robot_asset_sha256': sha,
        'physics_profile_sha256': sha, 'seed': 0, 'track': 'direct_joints',
        'instruction': 'Move the object', 'budgets': {'sim_time_s': 10., 'wall_time_s': 60.,
            'max_actions': 5, 'max_model_calls': 5, 'max_action_sim_s': .5}}
    obs = {'episode_id': 'episode', 'sequence': 0, 'sim_time_s': 0.,
        'instruction': 'Move the object', 'profile': 'vision_proprio',
        'proprioception': {'joint_positions_rad': {'arm': 0., 'finger': 0.},
                          'joint_velocities_rad_s': {'arm': 0., 'finger': 0.}},
        'cameras': [], 'last_action_status': 'idle'}
    return caps, spec, obs


@dataclass
class Response:
    text: str

    def public_metadata(self):
        return {'model': 'gpt-6-astra', 'reasoning_effort': 'max'}


def setup_loop(monkeypatch, tmp_path, responses, *, private=False, rgbd=False, missing_depth=False, fingertip=False):
    caps, spec, obs = fixtures()
    caps['depth_available'] = rgbd
    policy = {'policy_id':'classical_fingertip','version':'1.0.0','implementation_sha256':'b'*64,
        'trained':False,'observation_profile':'vision_proprio','skills':['move_geometry','hold'],
        'geometry_inputs':['wrist_pose','fingertip_positions','hand_posture'],
        'camera_observation_mode':'decision_only','supported_robot_ids':['robot']}
    if fingertip:
        caps.update(tracks=['direct_joints','system0'], system0=deepcopy(policy),
                    finger_names=['thumb','index'],frame_ids=['world','right_hand_flange'])
        spec.update(track='system0',system0=deepcopy(policy))
        spec['budgets']['max_action_sim_s'] = 5.
        obs['proprioception']['geometry'] = {'wrist': {'frame_id':'world','position_m':[.4,0.,1.],
            'quaternion_wxyz':[1.,0.,0.,0.]}, 'fingertip_frame_id':'right_hand_flange',
            'fingertip_positions_m':{'thumb':[.05,.04,.15],'index':[.05,-.04,.15]}}
    if private:
        obs['privileged_object_poses'] = {}

    class Bench:
        def __init__(self):
            self.obs = deepcopy(obs)
            self.commands = []
            self.create_requests = []
            self.discovery_requests = []
            self.policy_inventory = [deepcopy(policy)] if fingertip else []
            self.reject_geometry_once = False
            self.finished = self.closed = False

        def list_tasks(self): return [{'task_id': 'task', 'instruction': 'Move the object'}]
        def list_policies(self, task_id=None):
            self.discovery_requests.append(('policies', task_id))
            return deepcopy(self.policy_inventory)
        def create(self, request):
            self.create_requests.append(deepcopy(request))
            spec['agent_config_sha256'] = request['agent_config_sha256']
            spec['budgets'] = deepcopy(request['budgets'])
            return {'episode_id': 'episode', 'spec': spec, 'capabilities': caps, 'observation': deepcopy(self.obs)}
        def terminal_result(self, outcome='cancelled', **changes):
            result = EpisodeResult(spec=deepcopy(spec), outcome=outcome,
                elapsed_sim_s=self.obs['sim_time_s'], elapsed_wall_s=.1,
                **({'completion_sim_s':self.obs['sim_time_s'], 'completion_wall_s':.1,
                    'evaluator_evidence_sha256':'a'*64} if outcome == 'success' else {})
                ).model_dump(mode='json')
            result.update(changes)
            return result
        def get_result(self, _): return self.terminal_result() if self.finished else None
        def get_observation(self, _): return deepcopy(self.obs)
        def download_cameras(self, *_):
            stream = BytesIO()
            Image.new('RGB', (32, 24), (self.obs['sequence'], 0, 0)).save(stream, format='PNG')
            raw = stream.getvalue()
            rows = [{'metadata': {'camera_id': name, 'rgb_sha256': hashlib.sha256(raw).hexdigest()},
                'png_bytes': raw} for name in (['overview', 'task'] if rgbd else ['overview'])]
            if rgbd and not missing_depth:
                depth = np.full((24, 32), 1. + self.obs['sequence']*.01, dtype=np.float32)
                depth_stream = BytesIO()
                np.save(depth_stream, depth, allow_pickle=False)
                for row in rows:
                    row.update(depth_npy_bytes=depth_stream.getvalue(), depth_m=depth)
            return rows
        def submit(self, _, command):
            self.commands.append(command)
            self.obs['last_action_id'] = command['action_id']
            if self.reject_geometry_once and command['kind'] == 'system0':
                self.reject_geometry_once = False
                self.obs['last_action_status'] = 'failed'
                return {'status':'accepted', 'action_id':command['action_id'], 'sim_time_s':self.obs['sim_time_s']}
            self.obs['sequence'] += 20
            if command['kind'] == 'system0':
                if command['skill']=='hold':
                    self.obs['sim_time_s'] += command['hold_duration_s']
                else:
                    self.obs['sim_time_s'] += min(.4, command['timeout_sim_s'])
                    current = self.obs['proprioception']['geometry']
                    current['wrist'] = deepcopy(command['geometry']['wrist'])
                    current['fingertip_positions_m'].update(command['geometry']['fingertip_positions_m'])
            else:
                self.obs['sim_time_s'] += command['duration_s']
                self.obs['proprioception']['joint_positions_rad'] = command['arm_joint_positions_rad'] | command['hand_joint_positions_rad']
            self.obs['last_action_status'] = 'completed'
            return {'status': 'accepted', 'action_id': command['action_id'], 'sim_time_s': self.obs['sim_time_s']}
        def get_action(self, _, action_id):
            return {'status': self.obs['last_action_status'], 'action_id': action_id, 'sim_time_s': self.obs['sim_time_s']}
        def wait_action(self, episode, action_id, **__): return self.get_action(episode, action_id)
        def finish(self, _):
            self.finished = True
            return self.get_result('episode')
        def close(self): self.closed = True

    class Model:
        def __init__(self): self.requests = []; self.closed = False
        def generate(self, prompt, turns, **options):
            self.requests.append((prompt, deepcopy(turns), options))
            item = responses[len(self.requests)-1]
            if isinstance(item, Exception): raise item
            return Response(json.dumps(item))
        def close(self): self.closed = True

    class Config:
        def public_metadata(self): return {'base_url': 'https://example.invalid/v1'}

    bench, model = Bench(), Model()
    monkeypatch.setattr(runner.ModelGatewayConfig, 'from_env', lambda: Config())
    monkeypatch.setattr(runner, 'ModelClient', lambda *_, **__: model)
    monkeypatch.setattr(runner, 'BenchHTTPClient', lambda *_, **__: bench)
    args = Namespace(credentials=None, model='gpt-6-astra', reasoning_effort='max',
        model_timeout=20., http_timeout=20., bench_url='http://localhost:8000',
        output=tmp_path/'run', max_output_tokens=16384, max_decisions=len(responses),
        sim_budget=10., wall_budget=60., task='task', robot='robot', seed=0,
        visual_input='rgbd' if rgbd else 'rgb', control_mode='fingertip' if fingertip else 'joints')
    return args, bench, model


@pytest.mark.parametrize('kind', ['board_game', 'turn_game'])
def test_player_games_never_enter_legacy_ordinary_episode_loop(monkeypatch, tmp_path, kind):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    monkeypatch.setattr(bench, 'list_tasks', lambda: [{'task_id': 'task', 'task_kind': kind}])
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not bench.create_requests and not bench.commands and not model.requests


def test_each_decision_receives_new_image_and_feedback(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action': 'joint_delta', 'delta_rad': {'arm': .02}, 'duration_s': .4},
        {'action': 'stop', 'reason': 'stop after inspecting feedback'}])
    report = runner.run(args)
    assert report['online_model_decision_loop_exercised']
    assert report['model_input_observation_sequences'] == [0, 20]
    assert report['termination'] == 'model_requested_stop'
    assert not report['task_success']
    assert len(bench.commands) == 1
    first, second = (entry[1][-1]['content'] for entry in model.requests)
    assert first[-1]['image_url']['url'] != second[-1]['image_url']['url']
    assert json.loads(second[0]['text'])['observation']['sequence'] == 20
    assert json.loads(second[0]['text'])['previous_action_feedback']['status'] == 'completed'
    request = load_json(args.output/'decisions/0001/model_request.json')
    assert request['reasoning_effort'] == 'max'
    assert request['expanded_messages_sha256'] == runner.canonical_hash(model.requests[1][1])
    assert bench.finished and bench.closed and model.closed


def test_history_omits_only_repeated_static_context(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action': 'hold', 'duration_s': .4},
        {'action': 'hold', 'duration_s': .4}, {'action': 'stop'}], rgbd=True)
    report = runner.run(args)
    assert report['error_type'] is None
    current = json.loads(model.requests[2][1][-1]['content'][0]['text'])
    assert {'task', 'capabilities', 'action_limits'} <= current.keys()
    for index, history in enumerate(model.requests[2][1][::2][:-1]):
        original = json.loads(model.requests[index][1][-1]['content'][0]['text'])
        expected = {key: value for key, value in original.items()
                    if key not in {'task', 'capabilities', 'action_limits'}}
        assert json.loads(history['content']) == expected
        assert len(history['content']) < len(json.dumps(original, ensure_ascii=False, sort_keys=True))
    recorded = load_json(args.output/'decisions/0002/model_request.json')
    assert recorded['expanded_messages_sha256'] == runner.canonical_hash(model.requests[2][1])


@pytest.mark.parametrize('capture_seconds, expected_calls', [(50., 1), (61., 0)])
def test_capture_consumes_wall_budget(monkeypatch, tmp_path, capture_seconds, expected_calls):
    from types import SimpleNamespace
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    now = [100.]
    # Patch this module's clock, not the shared time module used by the HTTP stack.
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    from agent.model_client import interaction
    monkeypatch.setattr(interaction, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    download = bench.download_cameras
    observed_deadlines = []
    def slow_download(*arguments):
        observed_deadlines.append(bench.deadline)
        now[0] += capture_seconds
        return download(*arguments)
    bench.download_cameras = slow_download
    report = runner.run(args)
    assert observed_deadlines == [160.]
    assert bench.deadline is None
    assert len(model.requests) == report['model_call_attempts'] == expected_calls
    assert report['error_type'] is None
    assert bench.finished and bench.closed and model.closed
    if expected_calls:
        assert model.requests[0][2]['timeout_s'] == 10.
    else:
        assert report['termination'] == 'client_wall_deadline'
        assert not (args.output/'decisions/0000/model_request.json').exists()


def test_capture_http_deadline_preserves_interrupt_without_model_call(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agent.model_client import interaction
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    now = [100.]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(runner, 'time', clock)
    monkeypatch.setattr(interaction, 'time', clock)
    def expired_download(*arguments):
        now[0] = 160.
        error = runner.BenchHTTPError('deadline expired')
        error.category = 'request_budget_exhausted'
        raise error
    bench.download_cameras = expired_download
    report = runner.run(args)
    assert report['termination'] == 'client_error'
    assert report['error_type'] == 'BenchHTTPError' and report['run_status'] == 'infra_error'
    assert report['task_success'] is None
    assert report['model_call_attempts'] == 0
    assert bench.finished and bench.closed and model.closed


@pytest.mark.parametrize('elapsed, category, termination, failure', [
    (60., 'transport_timeout', 'client_error', 'ModelGatewayError'),
    (60., 'request_budget_exhausted', 'client_error', 'ModelGatewayError'),
    (10., 'transport_timeout', 'client_error', 'ModelGatewayError'),
    (60., 'http_status', 'client_error', 'ModelGatewayError'),
])
def test_model_deadline_preserves_call_accounting_and_failure_kind(
        monkeypatch, tmp_path, elapsed, category, termination, failure):
    from types import SimpleNamespace
    from agent.model_client import interaction
    args, bench, model = setup_loop(monkeypatch, tmp_path, [])
    args.max_decisions = 1
    args.model_timeout = None
    now = [100.]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(runner, 'time', clock)
    monkeypatch.setattr(interaction, 'time', clock)
    def timeout(*arguments, **options):
        assert options['timeout_s'] == 60.
        now[0] += elapsed
        raise runner.ModelGatewayError('model error', attempts=1, category=category)
    model.generate = timeout
    report = runner.run(args)
    assert report['termination'] == termination
    assert report['error_type'] == failure
    assert report['model_call_attempts'] == 1
    assert report['model_responses'] == 0
    assert report['model_error']['error_category'] == category
    assert report['task_success'] is (False if termination == 'client_wall_deadline' else None)
    assert bench.finished and bench.closed and model.closed


def test_invalid_decision_is_not_replaced_by_script(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action': 'joint_delta', 'delta_rad': {'arm': 1.}, 'duration_s': .4},
        {'action': 'joint_delta', 'delta_rad': {'arm': -.02}, 'duration_s': .4}])
    report = runner.run(args)
    assert len(bench.commands) == 1
    assert bench.commands[0]['arm_joint_positions_rad']['arm'] == -.02
    assert json.loads(model.requests[1][1][-1]['content'][0]['text'])['previous_action_feedback']['status'] == 'rejected_by_client'
    assert report['termination'] == 'client_decision_budget'
    assert report['result']['outcome'] == 'cancelled'
    assert report['model_input_observation_sequences'] == [0, 0]
    assert not report['online_model_decision_loop_exercised']


def test_rgbd_model_gets_aligned_previews_and_fresh_metric_samples(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action': 'hold', 'duration_s': .4}, {'action': 'stop'}], rgbd=True)
    report = runner.run(args)
    assert report['visual_input'] == 'rgbd'
    assert report['error_type'] is None
    assert report['online_model_decision_loop_exercised']
    for index, request in enumerate(model.requests):
        content = request[1][-1]['content']
        assert sum(block['type'] == 'image_url' for block in content) == 4
        depth_context = json.loads(content[0]['text'])['depth_observation']
        assert set(depth_context['cameras']) == {'overview', 'task'}
        for camera in ('overview', 'task'):
            folder = args.output/'decisions'/f'{index:04d}'
            stored = np.load(folder/(camera+'_depth.npy'), allow_pickle=False)
            assert np.allclose(stored, 1. + index*.2)
            assert depth_context['cameras'][camera]['grid']['values_m'][0][0] == pytest.approx(1.+index*.2)
            assert (folder/(camera+'_depth_preview.png')).is_file()
        recorded = load_json(args.output/'decisions'/f'{index:04d}'/'model_request.json')
        assert recorded['expanded_messages_sha256'] == runner.canonical_hash(request[1])
        assert recorded['reasoning_effort'] == 'max'
    assert bench.finished


@pytest.mark.parametrize('advertised', [True, False])
def test_rgbd_does_not_silently_fall_back_to_rgb(monkeypatch, tmp_path, advertised):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}],
                                   rgbd=advertised, missing_depth=True)
    args.visual_input = 'rgbd'
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not model.requests
    assert not bench.commands
    assert bench.finished


@pytest.mark.parametrize('private', [False, True])
def test_failures_finalize_without_logging_exception_secrets(monkeypatch, tmp_path, private):
    secret = 'secret-from-provider-error'
    args, bench, model = setup_loop(monkeypatch, tmp_path, [RuntimeError(secret)], private=private)
    report = runner.run(args)
    assert report['error_type'] is not None
    assert not report['online_model_decision_loop_exercised']
    assert bench.finished and bench.closed and model.closed
    if private:
        assert not model.requests
    for path in args.output.rglob('*.json'):
        assert secret not in path.read_text()


@pytest.mark.parametrize('category, status', [('transport_timeout', None), ('transport_error', None),
    ('http_status', 429), ('invalid_json', 200), ('model_mismatch', 200)])
def test_model_failure_archives_only_typed_fields_without_retry_or_fake_usage(monkeypatch, tmp_path, capsys, category, status):
    from agent.model_client.transport import ModelGatewayError
    from agent.model_client.suite import model_statistics
    secret = 'provider-private-body-and-sk-do-not-record-this'
    error = ModelGatewayError(secret, category=category, status_code=status, request_id=secret, attempts=1)
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action':'hold', 'duration_s':.4}, error, {'action':'stop'}])
    report = runner.run(args)
    expected = {'decision_index':1, 'error_category':category, 'http_status':status, 'attempts':1}
    assert report['error_type'] == 'ModelGatewayError' and report['termination'] == 'client_error'
    assert report['model_error'] == expected
    assert report['model_call_attempts'] == 2 and report['model_responses'] == 1
    assert report['completed_actions'] == 1 and len(model.requests) == 2
    assert report['task_success'] is None and report['task_outcome'] == 'unknown'
    assert report['run_status'] == 'infra_error' and bench.finished and bench.closed and model.closed
    path = args.output/'decisions/0001/model_error.json'
    assert load_json(path) == expected
    assert not path.with_name('model_response.json').exists()
    assert not (args.output/'decisions/0002').exists()
    hashes = load_json(args.output/'artifact_hashes.json')
    assert hashes['decisions/0001/model_error.json'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(model_statistics(report)[field] is None for field in
               ('model_latency_client_s', 'gateway_prompt_tokens', 'gateway_total_tokens'))
    for path in args.output.rglob('*.json'):
        assert secret not in path.read_text()
    assert secret not in capsys.readouterr().out


def test_fingertip_mode_pins_controller_and_submits_sparse_geometry(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action':'move_geometry','wrist_delta_m':[0.,0.,.01],
         'fingertip_delta_m':{'thumb':[.002,0.,0.]},'duration_s':3.},
        {'action':'stop'}], fingertip=True, rgbd=True)
    report = runner.run(args)
    assert report['error_type'] is None and report['control_mode'] == 'fingertip'
    assert report['online_model_decision_loop_exercised']
    assert report['controller']['policy_id'] == 'classical_fingertip'
    request, = bench.create_requests
    assert request['track'] == 'system0' and request['policy_version'] == '1.0.0'
    assert request['budgets']['max_action_sim_s'] == 5.
    command, = bench.commands
    assert command['kind'] == 'system0' and command['timeout_sim_s'] == 3.
    assert set(command['geometry']['fingertip_positions_m']) == {'thumb'}
    assert command['geometry']['fingertip_positions_m']['thumb'][0] == pytest.approx(.052)
    config = load_json(args.output/'agent_config.json')
    assert request['agent_config_sha256'] == runner.canonical_hash(config)
    assert config['reasoning_effort'] == 'max' and config['trained_system0'] is False
    assert 'geometry_actions.py' in config['client_sources_sha256']
    assert 'dependencies/bench/geometry.py' in config['client_sources_sha256']
    assert all('right_hand_flange LOCAL axes' in prompt for prompt,_,_ in model.requests)
    second = json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert second['observation']['proprioception']['geometry']['wrist']['position_m'][2] == 1.01
    assert second['previous_action_feedback']['status'] == 'completed'


def test_fingertip_mode_retries_failed_ik_without_joint_or_script_fallback(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action':'move_geometry','wrist_delta_m':[0.,0.,.03],'duration_s':.4},
        {'action':'move_geometry','wrist_delta_m':[0.,0.,.01],'duration_s':3.},
        {'action':'stop'}], fingertip=True)
    bench.reject_geometry_once = True
    report = runner.run(args)
    assert report['error_type'] is None and len(bench.commands) == 2
    assert all(command['kind'] == 'system0' for command in bench.commands)
    assert report['completed_actions'] == 1
    assert report['model_input_observation_sequences'] == [0,0,20]
    second = json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert second['previous_action_feedback']['status'] == 'failed'


@pytest.mark.parametrize('kind', ['missing','ambiguous','robot_mismatch'])
def test_fingertip_mode_rejects_unavailable_controller_before_episode_or_model(monkeypatch, tmp_path, kind):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}], fingertip=True)
    if kind == 'missing': bench.policy_inventory = []
    if kind == 'ambiguous': bench.policy_inventory.append(deepcopy(bench.policy_inventory[0]))
    if kind == 'robot_mismatch': bench.policy_inventory[0]['supported_robot_ids'] = ['different_robot']
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not bench.create_requests and not bench.commands and not model.requests
    assert bench.closed and model.closed


def test_fingertip_is_the_default_control_mode(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}], fingertip=True)
    del args.control_mode
    report = runner.run(args)
    assert report['error_type'] is None and report['control_mode'] == 'fingertip'
    assert bench.create_requests[0]['track'] == 'system0'
    assert model.requests[0][0] == runner.model_system_prompt('fingertip', 'action', bench.policy_inventory[0])


def test_controller_pin_change_at_episode_creation_prevents_model_or_actions(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}], fingertip=True)
    original_create = bench.create
    def changed_pin(request):
        reply = original_create(request)
        reply['spec']['system0']['implementation_sha256'] = 'f'*64
        return reply
    bench.create = changed_pin
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not model.requests and not bench.commands
    assert bench.finished and bench.closed and model.closed
