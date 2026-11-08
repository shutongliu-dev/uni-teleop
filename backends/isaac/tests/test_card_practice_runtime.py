"""Public HTTP/ownership and immutable preparation checks with a CPU worker double."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs, validate
from bench.paths import DATA_ROOT, OUTPUT_ROOT
from bench.protocol.task_referee import PublicTaskRequirements, project_task_rules
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.card_room.practice.referee import PracticeReferee
from scenes.workrooms.card_room.practice.release import prepare_release, load_practice_inputs
from scenes.workrooms.card_room.practice.runtime import PracticeRuntime
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action
from test_card_practice_referee import advance, baseline

@pytest.fixture(scope='module')
def release():
    # A temporary practice release under the output root, removed afterwards.
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix='card_practice_cpu_', dir=OUTPUT_ROOT))
    path = base/'polyomino_territory_practice'
    prepare_release(path, path.name)
    yield path
    shutil.rmtree(base)


class PracticeWorkerDouble(FakeWorker):
    def __init__(self, *, release, task_id, hand_id, practice, **kwargs):
        assert hand_id == 'dual_fr3_sharpa_w01'
        super().__init__(hand_id='sharpa_w01_right', acknowledged_targets=True, depth=True, **kwargs)
        _, scene, task, initial = task_inputs(release, task_id, practice=practice)
        model = public_robot_geometry(robot_configuration(scene))
        self.arm, self.active = list(model.arm_names), list(model.names)
        self.hand = [n for n in self.active if n not in self.arm]
        self.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
        self.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        self.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': self.active, 'arm_joint_names': self.arm, 'hand_joint_names': self.hand,
            'joint_limits_rad': {n: [float(side.lower[i]), float(side.upper[i])]
                for side in model.sides.values() for i, n in enumerate(side.names)}})
        self.referee = PracticeReferee(task)
        self.frame = baseline(self.referee, index=0)
        self.guard = PracticeRuntime.__new__(PracticeRuntime)
        self.guard.env = SimpleNamespace(_validate_action=self._validate_command)
        self.guard.closed = False
        self.guard.evaluator = self.referee
        self.guard.targets = deepcopy(initial['joint_positions_rad'])
        self.guard.owned = set(model.sides[practice['active_side']].names)
        self.initial.update(observation=deepcopy(self.raw), score=self.referee.score(), task=project_task_rules(task))

    def _validate_command(self, command):
        assert set(command) == {'joint_positions_rad', 'duration_s'}
        assert set(command['joint_positions_rad']) == set(self.active)
        assert command['duration_s'] == .02

    def request(self, operation, **payload):
        if operation == 'step':
            self.guard.authorize_step(payload['command'], payload.get('actor'))
            result = super().request(operation, **payload)
            self.raw['robot']['joint_targets_rad'] = {
                n: float(np.float32(v)) for n, v in payload['command']['joint_positions_rad'].items()}
            for _ in range(40):
                self.frame = advance(self.frame)
                self.referee.observe(self.frame)
            self.guard.acknowledge_step(payload['command'], None)
            result.update(observation=deepcopy(self.raw), score=self.referee.score())
            return result
        return super().request(operation, **payload)


@pytest.fixture
def api(release, tmp_path):
    workers = []
    def factory(**kwargs):
        value = PracticeWorkerDouble(**kwargs)
        workers.append(value)
        return value
    service = AgentService(tmp_path/'episodes', release=release, registry=default_registry(release), worker_factory=factory)
    with TestClient(create_app(service=service)) as client:
        yield SimpleNamespace(service=service, client=client, workers=workers)
    service.shutdown()


def request(side='left', track='system0'):
    return {'task_id': 'polyomino_territory_practice', 'agent_id': 'cpu_test', 'agent_version': 'v1',
        'agent_config_sha256': 'a'*64, 'robot_id': 'dual_fr3_sharpa_w01', 'track': track,
        **({'policy_id': 'classical_fingertip_bimanual', 'policy_version': '1.2.0-board1'} if track == 'system0' else {}),
        'practice': {'active_side': side, 'case_id': 'mono_translate_00', 'split': 'train'}}


def create(api, side='left', track='system0'):
    response = api.client.post('/episodes', json=request(side, track))
    assert response.status_code == 201, response.text
    value = response.json()
    return api.service.sessions[value['episode_id']], value


def call(api, session, function, arguments=None):
    response = api.client.post('/episodes/'+session.spec.episode_id+'/call', json={
        'function': function, 'request_id': 'cpu_case', 'arguments': arguments or {}})
    assert response.status_code in ((200, 202) if response.json().get('ok') else (response.json()['error']['http_status'],)), response.text
    assert PRIVATE_MARKER not in response.text
    return response.json()


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('preset', ['open', 'pinch_prepose'])
def test_practice_hand_preset_uses_side_and_remains_read_only(api, side, preset):
    session, _ = create(api, side)
    worker = api.workers[-1]
    before = deepcopy(worker.calls)
    snapshot = session.snapshot.model_dump(mode='json')
    targets = deepcopy(session._practice_acknowledged_targets)
    # This is the real public request shape used by the failed Sol trial.
    selected = call(api, session, 'hand_preset', {
        'preset': preset, 'side': side, 'amount': .45})
    assert selected['ok'], selected
    assert selected['result']['side'] == side
    assert all(name.startswith(side+'_') for name in selected['result']['hand_posture_rad'])
    other = 'right' if side == 'left' else 'left'
    refused = call(api, session, 'hand_preset', {'preset': preset, 'side': other})
    assert refused['ok'] is False and refused['error']['http_status'] == 403
    assert refused['error']['code'] == 'permission_denied'
    invalid = call(api, session, 'hand_preset', {'preset': preset, 'side': 'both'})
    assert invalid['ok'] is False and invalid['error']['http_status'] == 422
    assert invalid['error']['code'] == 'invalid_arguments'
    assert all(row['context']['budget']['action_attempts'] == 0
               for row in (selected, refused, invalid))
    assert worker.calls == before
    assert session.snapshot.model_dump(mode='json') == snapshot
    assert session._practice_acknowledged_targets == targets


@pytest.mark.parametrize('active', ['left', 'right', None])
@pytest.mark.parametrize('selected', ['left', 'right'])
@pytest.mark.parametrize('function', ['move_wrist', 'set_hand_targets'])
def test_practice_motion_functions_keep_hand_selector(active, selected, function):
    from bench.protocol.functions import CALL_ADAPTER
    from bench.runtime.practice import check_function
    from bench.runtime.service import ServiceError
    arguments = {'hand': selected, 'action_id': 'owned_motion',
                 'observation_sequence': 0, 'timeout_sim_s': 1.}
    if function == 'move_wrist':
        arguments['wrist'] = {'frame_id': 'world', 'position_m': [0., 0., 1.],
                              'quaternion_wxyz': [1., 0., 0., 0.]}
    else:
        arguments['hand_posture_rad'] = {selected+'_index_MCP_FE': .1}
    request = CALL_ADAPTER.validate_python({'function': function,
        'request_id': 'ownership_schema', 'arguments': arguments})
    session = SimpleNamespace(practice_side=active)
    if active is not None and selected != active:
        with pytest.raises(ServiceError) as error:
            check_function(session, function, request.arguments)
        assert error.value.status == 403
    else:
        check_function(session, function, request.arguments)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_normal_http_public_case_ownership_and_ack_hold(api, side):
    session, created = create(api, side)
    assert session.player_session is None and session.practice_side == side
    assert session.spec.practice.active_side == side
    value = call(api, session, 'get_task')['result']
    PublicTaskRequirements.model_validate(value)
    assert value['practice']['active_side'] == side
    assert value['practice_progress']['confirmed_stages'] == 0
    serialized = json.dumps(value)
    for secret in ('case_fingerprint', 'private_case_sha256', 'private_reset_pairs', 'initial_positions_m', 'raw_contacts'):
        assert secret not in serialized
    other = 'right' if side == 'left' else 'left'
    before = deepcopy(api.workers[-1].raw['robot']['joint_targets_rad'])
    refused = call(api, session, 'set_hand_targets', {'action_id': 'wrong_side', 'observation_sequence': 0,
        'hand': other, 'hand_posture_rad': {f'{other}_index_MCP_FE': .1}, 'timeout_sim_s': 1.})
    assert refused['ok'] is False and refused['error']['http_status'] == 403
    assert not any(operation == 'step' for operation, _ in api.workers[-1].calls)
    hold = call(api, session, 'hold', {'action_id': 'hold_baseline', 'observation_sequence': 0, 'duration_s': .04})
    assert hold['ok'], hold
    join_action(session)
    assert session.result is None
    assert session.status('hold_baseline')['status'] == 'completed'
    own = set(session.robot_geometry.sides[side].names)
    for operation, payload in api.workers[-1].calls:
        if operation == 'step':
            assert all(np.float32(v) == np.float32(before[n]) for n, v in payload['command']['joint_positions_rad'].items() if n not in own)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_consecutive_geometry_begins_preserve_exact_ack_with_float32_native_storage(api, side):
    session, _ = create(api, side)
    geometry = session.robot_geometry.sides[side]
    peer = set(session.capabilities.joint_limits) - set(geometry.names)
    initial = dict(session._practice_acknowledged_targets)
    for index, offset in enumerate((.025, -.02, .015)):
        q = dict(session.raw['joint_positions_rad'])
        q[geometry.arm.names[0]] += offset
        wrist = geometry.public_geometry({n: q[n] for n in geometry.names})['wrist']
        action_id = f'move_{index}'
        accepted = call(api, session, 'move_geometry', {'action_id': action_id,
            'observation_sequence': session.sequence, 'timeout_sim_s': 3.,
            'hand_geometries': {side: {'wrist': wrist, 'fingertip_frame_id': geometry.frame_id}}})
        assert accepted['ok'], accepted
        join_action(session)
        assert session.result is None
        assert session.status(action_id)['status'] == 'completed'
        exact = session._practice_acknowledged_targets
        native = session.raw['robot']['joint_targets_rad']
        assert all(np.float32(exact[n]) == np.float32(native[n]) for n in exact)
        np.testing.assert_array_equal(session.policy.sides[side].previous,
                                      [exact[n] for n in geometry.names])
        assert all(exact[n] == initial[n] for n in peer)
        # A read-only preview performs another real isolated begin from the same
        # ledger without mutating the live executor or its ACK inventory.
        preview = call(api, session, 'query_motion', {'observation_sequence': session.sequence,
            'requested_duration_s': 3., 'hand_geometries': {side: {
                'wrist': wrist, 'fingertip_frame_id': geometry.frame_id}}})
        assert preview['ok'], preview
        assert session._practice_acknowledged_targets == exact


def test_real_native_roundoff_witness_does_not_replace_executor_reference():
    from bench.runtime.board_game_control import BoardGameFingertipPolicy
    from bench.runtime.practice import checked_acknowledgement, policy_targets
    from bench.protocol.schemas import RobotJointCommand
    policy = BoardGameFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., .77])
    targets = {j.name: float((j.lower+j.upper)/2) for j in policy.geometry.joints}
    # Native episode 08feab6af9454e0d813d5c49b0996968, t=6.8 s:
    # the submitted and read-back values are the identical float32 drive.
    targets['fr3_left_joint6'] = 2.8798288148841094
    native = {n: float(np.float32(v)) for n, v in targets.items()}
    assert native['fr3_left_joint6'] == 2.879828929901123
    assert abs(native['fr3_left_joint6']-targets['fr3_left_joint6']) > 1.e-7
    own = policy.sides['left']
    own.previous = np.array([targets[n] for n in own.geometry.names])
    with pytest.raises(ValueError, match='reference differs'):
        policy.set_player_hand('left', native)
    session = SimpleNamespace(practice_side='left', _practice_acknowledged_targets=None)
    command = RobotJointCommand(duration_s=.02, joint_positions_rad=targets)
    session._practice_acknowledged_targets = checked_acknowledgement(session,
        {'observation': {'robot': {'joint_targets_rad': native}}}, command)
    policy.set_player_hand('left', policy_targets(session))
    before = dict(session._practice_acknowledged_targets)
    native['fr3_left_joint6'] = float(np.nextafter(np.float32(native['fr3_left_joint6']), np.float32(4.)))
    with pytest.raises(RuntimeError, match='differs from the submitted command'):
        checked_acknowledgement(session, {'observation': {'robot': {'joint_targets_rad': native}}}, command)
    assert session._practice_acknowledged_targets == before


@pytest.mark.parametrize('change', [lambda p: p.pop('practice'),
    lambda p: p['practice'].update(case_id='mono_translate_04'),
    lambda p: p.update(observation_profile='privileged_state'),
    lambda p: p.update(robot_id='sharpa_w01_right')])
def test_wrong_or_missing_selection_refused_before_worker(api, change):
    payload = request(); change(payload)
    response = api.client.post('/episodes', json=payload)
    assert response.status_code in (403, 422), response.text
    assert not api.workers


def test_inactive_direct_targets_rejected_by_both_host_and_native(api):
    session, _ = create(api, track='direct_joints')
    worker = api.workers[-1]
    q = deepcopy(worker.raw['robot']['joint_targets_rad'])
    q['fr3_right_joint1'] += .01
    with pytest.raises(ValueError, match='inactive'):
        worker.guard.authorize_step({'joint_positions_rad': q, 'duration_s': .02})
    response = api.client.post('/episodes/'+session.spec.episode_id+'/actions', json={
        'kind': 'direct_joints', 'action_id': 'unowned', 'observation_sequence': 0, 'duration_s': 1.,
        'arm_joint_positions_rad': {n: q[n] for n in worker.arm},
        'hand_joint_positions_rad': {n: q[n] for n in worker.hand}})
    assert response.status_code == 403, response.text
    assert not any(op == 'step' for op, _ in worker.calls)


def test_release_frozen_table_and_robot_physics_preservation(release):
    loaded = load_practice_inputs(release)
    assert len(loaded['cases']) == 48
    assert len({c['case_fingerprint'] for c in loaded['cases'].values()}) == 48
    assert 'qualification' not in json.dumps(loaded['manifest'])
    from pxr import Usd, UsdGeom
    from scenes.workrooms.card_room.practice.release import parent_inputs
    _, source, _, _ = parent_inputs('polyomino_territory_practice')
    original = Usd.Stage.Open(source['native_task_bundle']['usd_path'])
    cases = list(loaded['cases'].values())
    for case in (cases[0], cases[-1]):
        selection = {k: case['public'][k] for k in ('active_side', 'case_id', 'split')}
        _, scene, task, initial = task_inputs(release, 'polyomino_territory_practice', practice=selection)
        assert scene['robot']['hand_id'] == 'dual_fr3_sharpa_w01'
        assert len(initial['joint_positions_rad']) == 58
        assert 'board_game_task' not in scene
        assert task['practice_case'] == case
        assert scene['physics']['dt_s'] == .0005
        assert scene['native_task_bundle']['rigid_body_sampling']['sleep_threshold'] == 0.
    for case in cases:
        key = '/'.join(case['public'][k] for k in ('split', 'active_side', 'case_id'))
        stage = Usd.Stage.Open(str(release/'cases'/key/'initial.usda'))
        cache = UsdGeom.XformCache()
        for name, reset in case['reset'].items():
            path = '/World/CardGame/Actors/'+name
            prim, parent = stage.GetPrimAtPath(path), original.GetPrimAtPath(path)
            assert np.allclose(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), reset['position_m'], atol=1e-12)
            for attribute in parent.GetAttributes():
                if not attribute.GetName().startswith('xformOp:'):
                    assert prim.GetAttribute(attribute.GetName()).Get() == attribute.Get()


def test_non_practice_release_rejects_practice_parameter():
    from scenes.workrooms.card_room.expansion_tasks import RELEASES
    with pytest.raises(ValueError, match='only valid'):
        task_inputs(RELEASES['polyomino_territory']['path'], 'polyomino_territory', practice=request()['practice'])
