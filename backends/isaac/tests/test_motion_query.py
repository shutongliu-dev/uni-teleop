"""Actual unchanged classical executors and source convexes, CPU only."""
from copy import deepcopy
import json
import pickle
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import ClassicalFingertipPolicy, GeometryIKError
from bench.paths import DATA_ROOT
from bench.protocol.functions import FUNCTION_INPUTS, FUNCTION_OUTPUTS, ACTION_FUNCTIONS
from bench.protocol.motion import MotionQuery, MotionQueryResult
from bench.protocol.schemas import AgentObservation, Budgets, JointLimit, RobotCapabilities, System0Action
from bench.runtime.board_game_control import BoardGameFingertipPolicy
from bench.runtime.functions import EpisodeFunctions
from bench.runtime.motion_collision import Fr3SelfCollision, CollisionSourceUnavailable
from bench.runtime.motion_support import query_motion
from bench.runtime.service import EpisodeSession, ServiceError


@pytest.fixture(params=['single', 'dual', 'player'])
def session(request):
    cls = {'single': ClassicalFingertipPolicy, 'dual': BimanualFingertipPolicy,
           'player': BoardGameFingertipPolicy}[request.param]
    kwargs = {} if request.param == 'single' else {'assembly_version': 'isaacsim_6_1_v2'}
    policy = cls(f'{DATA_ROOT}/assets', [0., 0., .77], **kwargs)
    geometry = policy.geometry
    models = getattr(geometry, 'sides', None) or {'right': geometry}
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in geometry.joints}
    for side, model in models.items():
        q.update(dict(zip(model.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785])))
    capabilities = RobotCapabilities(robot_id=geometry.robot_id, robot_asset_sha256='a'*64,
        arm_joints=[n for m in models.values() for n in m.arm.names],
        hand_joints=[n for m in models.values() for n in m.hand.names],
        joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in geometry.joints},
        tracks=['system0'], system0=policy.descriptor,
        frame_ids=['world', *[m.frame_id for m in models.values()]])
    value = EpisodeSession.__new__(EpisodeSession)
    value.lock = threading.RLock()
    value.result = value.thread = None
    value.life = SimpleNamespace(active_action_id=None)
    value.spec = SimpleNamespace(episode_id='motion_test', observation_profile='vision_proprio',
        track='system0', budgets=Budgets(), robot_id=geometry.robot_id)
    value.capabilities = capabilities
    value.policy, value.robot_geometry = policy, geometry
    policy.reset(value.spec, capabilities)
    value.snapshot = AgentObservation(episode_id='motion_test', sequence=0, sim_time_s=0.,
        instruction='Public instruction', profile='vision_proprio',
        proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': dict.fromkeys(q, 0.),
                        'commanded_joint_positions_rad': dict(q)})
    value.clock = SimpleNamespace(action_attempts=0, sim_time_s=0., wall_time_s=0.,
                                 remaining_wall_s=1800., check=lambda: None)
    value.recorder = SimpleNamespace(closed=False, function_call=lambda _: None)
    value._function_api = EpisodeFunctions(value)
    value.raw = {'objects': {'PRIVATE_ORACLE_CANARY': object()}}
    value.worker = object()  # No callable worker interface available to this test.
    value.player_session = None
    if request.param == 'player':
        player = SimpleNamespace(_admitting_player='player_0', last_targets=dict(q),
            state=SimpleNamespace(turn_index=4, current_player='player_0'),
            bindings={'player_0': SimpleNamespace(hand='left', joint_names=models['left'].names)})
        def current(actor):
            if actor != player.state.current_player:
                raise ServiceError(403, 'Not the current player')
        player._current = current
        value.player_session = player
    return value


def body(session):
    models = getattr(session.robot_geometry, 'sides', None)
    side = 'left' if models else 'right'
    model = models[side] if models else session.robot_geometry
    q = {n: session.snapshot.proprioception.joint_positions_rad[n] for n in model.names}
    objective = {'fingertip_frame_id': model.frame_id,
                 'wrist': model.public_geometry(q)['wrist']}
    return MotionQuery(observation_sequence=session.snapshot.sequence,
        **({'hand_geometries': {side: objective}} if models else {'geometry': objective}))


def unchanged_state(session):
    player = session.player_session
    return (pickle.dumps(vars(session.policy)), session.snapshot.model_dump(),
            session.life.active_action_id, session.clock.action_attempts,
            None if player is None else (deepcopy(player.last_targets), player.state.turn_index))


def test_preview_original_plan_without_physics_budget_reference_or_epoch_mutation(session):
    query = body(session)
    before = unchanged_state(session)
    response = session.functions.invoke({'function': 'query_motion', 'request_id': 'preview',
                                         'arguments': query.model_dump(mode='json')})
    assert response.ok, response
    result = response.result
    assert result.planning_succeeded and result.minimum_duration_s >= .22
    assert result.advances_physics is result.mutates_controller is result.consumes_action_attempt is False
    assert result.convergence_guaranteed is False
    assert unchanged_state(session) == before
    assert 'PRIVATE_ORACLE_CANARY' not in result.model_dump_json()
    assert DATA_ROOT not in result.model_dump_json()
    assert 'query_motion' not in ACTION_FUNCTIONS
    assert FUNCTION_INPUTS['query_motion'] is MotionQuery
    assert FUNCTION_OUTPUTS['query_motion'] is MotionQueryResult
    clone = deepcopy(session.policy)
    if session.player_session:
        clone.set_player_hand('left', session.player_session.last_targets)
        assert result.turn_index == 4
    action = System0Action(action_id='test', observation_sequence=query.observation_sequence,
        skill='move_geometry', timeout_sim_s=result.minimum_duration_s,
        geometry=query.geometry, hand_geometries=query.hand_geometries)
    accepted = deepcopy(clone)
    accepted.begin(action, session.snapshot)
    with pytest.raises(GeometryIKError) as caught:
        clone.begin(action.model_copy(update={'timeout_sim_s': result.minimum_duration_s - .02}), session.snapshot)
    assert caught.value.public_diagnostic.duration_preflight.minimum_duration_s == result.minimum_duration_s


@pytest.mark.parametrize('requested', [.01, .02, .22, 20., 1.e308])
def test_duration_compares_episode_cap_and_control_period_without_execution(session, requested):
    query = body(session).model_copy(update={'requested_duration_s': requested})
    session.spec.budgets.max_action_sim_s = .1
    result = session.query_motion(query)
    assert result['planning_succeeded'] is True
    assert result['fits_action_budget'] is False
    assert result['requested_duration_fits'] is False
    assert session.clock.action_attempts == 0


@pytest.mark.parametrize('change,code', [('busy', 'episode_busy'), ('stale', 'stale_observation'),
                                        ('finished', 'episode_finished'), ('unsupported', 'unsupported_capability')])
def test_service_rejects_unusable_snapshot_without_query_side_effects(session, change, code):
    query = body(session)
    if change == 'busy': session.life.active_action_id = 'busy'
    if change == 'stale': query = query.model_copy(update={'observation_sequence': 7})
    if change == 'finished': session.result = object()
    if change == 'unsupported': session.robot_geometry = None
    before = unchanged_state(session)
    with pytest.raises(ServiceError) as caught:
        session.query_motion(query)
    assert caught.value.code == code
    assert unchanged_state(session) == before


def test_unreachable_target_returns_original_ik_diagnostic(session):
    query = body(session)
    objective = query.geometry if query.geometry else query.hand_geometries['left']
    objective.wrist.position_m = [100., 100., 100.]
    before = unchanged_state(session)
    result = session.query_motion(query)
    assert result['planning_succeeded'] is False and result['hands'] == {}
    assert result['diagnostic']['code'] == 'ik_solution_unverified'
    assert result['minimum_duration_s'] is None
    assert unchanged_state(session) == before


def test_recovery_preview_uses_committed_reference_and_preserves_original(session):
    models = getattr(session.policy, 'sides', None) or {'right': session.policy}
    side = 'left' if 'left' in models else 'right'
    own = models[side]
    q = session.snapshot.proprioception.joint_positions_rad
    own.previous = np.array([q[n] for n in own.geometry.names])
    own.velocity = np.zeros(len(own.geometry.names))
    own.velocity[0] = .2
    own._recovery_required = True
    before = unchanged_state(session)
    result = session.query_motion(body(session))
    plan = result['hands'][side]
    assert plan['braking_steps'] > 0
    assert plan['initial_joint_positions_rad'][own.geometry.names[0]] > q[own.geometry.names[0]]
    assert unchanged_state(session) == before


@pytest.mark.parametrize('violation', ['anonymous', 'wrong_hand', 'wrong_player', 'wrong_joint'])
def test_player_preview_refuses_ownership_violations(session, violation):
    if session.player_session is None:
        pytest.skip('Player-only admission')
    query = body(session)
    if violation == 'anonymous': session.player_session._admitting_player = None
    if violation == 'wrong_player': session.player_session.state.current_player = 'player_1'
    if violation == 'wrong_hand': query.hand_geometries = {'right': query.hand_geometries['left']}
    if violation == 'wrong_joint': query.hand_geometries['left'].hand_posture_rad = {'right_thumb_CMC_FE': .1}
    before = unchanged_state(session)
    with pytest.raises(ServiceError) as caught:
        session.query_motion(query)
    assert caught.value.status == 403
    assert unchanged_state(session) == before


def test_query_discards_result_if_epoch_or_ack_changed_during_query(session, monkeypatch):
    import bench.runtime.motion_support as module
    original = module.query_motion
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if session.player_session:
            session.player_session.state.turn_index += 1
        else:
            session.snapshot.sequence += 1
        return result
    monkeypatch.setattr(module, 'query_motion', changed)
    with pytest.raises(ServiceError) as caught:
        session.query_motion(body(session))
    assert caught.value.code == 'stale_observation'


def test_original_arm_convexes_reject_recorded_colliding_goal_without_task_state():
    policy = BimanualFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., 0.], assembly_version='isaacsim_6_1_v2')
    model = policy.geometry.sides['left']
    # Public measured/IK arm coordinates from attempt02/0018. No object pose is
    # used and this fixture is never made available to a game Agent.
    start = dict(zip(model.arm.names, [-.7372866272926331, .6435596942901611, .19089798629283905,
                                     -2.9542038440704346, -.34392037987709045, 4.279049873352051, -.36529219150543213]))
    goal = dict(zip(model.arm.names, [-.8976351978860803, .9656163691314477, -.04747029101473174,
                                    -3.077, -.6336382366750848, 4.6216, -.6574638946999753]))
    result = Fr3SelfCollision(model).check_path(start, goal)
    assert result['goal_detected'] and result['detected'] and result['sample_count'] == 65
    assert result['colliding_sample_count'] == 32
    pairs = {tuple(row['links']) for row in result['pairs']}
    assert ('fr3_left_link0', 'fr3_left_link5') in pairs
    assert ('fr3_left_link0', 'fr3_left_link6') in pairs
    home = dict(zip(model.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785]))
    assert Fr3SelfCollision(model).check(home) == []


def test_robot_import_pin_failure_is_unsupported_not_safe(monkeypatch):
    import bench.runtime.motion_collision as module
    policy = BimanualFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., 0.], assembly_version='isaacsim_6_1_v2')
    model = policy.geometry.sides['left']
    monkeypatch.setattr(module, '_sha', lambda _: '0' * 64)
    with pytest.raises(CollisionSourceUnavailable) as caught:
        Fr3SelfCollision(model)
    assert caught.value.reason == 'source_import_unverified'

