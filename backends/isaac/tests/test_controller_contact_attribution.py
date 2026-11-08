"""Stall reasons use the failing robot instance and current public contacts only."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bench.paths import DATA_ROOT
from bench.protocol.schemas import ActionDiagnostic
from bench.runtime.service import EpisodeSession


def diagnostic(*, hand=None, fingers=None, wrist=False, posture=None):
    values = {"code": "tracking_stalled", "fingertip_errors_m": fingers or {},
              "wrist_position_error_m": .02 if wrist else 0.,
              "wrist_orientation_error_rad": 0.,
              "geometry_tolerances": {"stage": "completion", "wrist_position_m": .005,
                  "wrist_orientation_rad": .04,
                  "fingertips_m": {name: .0015 for name in fingers or {}}}}
    if hand is not None:
        values["hand"] = hand
    if posture is not None:
        values.update(joint_name=posture[0], joint_error_rad=posture[1])
        values["geometry_tolerances"]["joint_posture_rad"] = .01
    return ActionDiagnostic.model_validate(values)


def session_for(history, names=()):
    session = object.__new__(EpisodeSession)
    session.contact_history = SimpleNamespace(snapshot=lambda: deepcopy(history))
    joints = {side + "_index_PIP": None for side in names or ("right",)}
    session.capabilities = SimpleNamespace(joint_limits=joints,
                                           finger_names=["thumb", "index", "middle", "ring", "little"])
    session.robot_geometry = SimpleNamespace(sides={side: SimpleNamespace(names=[side + "_index_PIP"])
                                                   for side in names}) if names else SimpleNamespace()
    return session


def regions(**counts):
    return {name: {"current_contact_substeps": count, "contact_substeps": 40}
            for name, count in counts.items()}


@pytest.mark.parametrize("protocol,names", [
    ("robot_rigid_contact_history_v2", ("left", "right")),
    ("robot_rigid_contact_history_v3", ("front", "rear")),
])
@pytest.mark.parametrize("own_contact,peer_contact,expected", [(1, 0, True), (0, 1, False), (0, 0, False)])
def test_bimanual_tip_stall_uses_only_its_own_current_contact(protocol, names, own_contact, peer_contact, expected):
    history = {"protocol": protocol, "status": "available", "hands": {
        names[0]: {"regions": regions(index_pad=own_contact)},
        names[1]: {"regions": regions(index_pad=peer_contact)}}}
    session = session_for(history, names)
    assert session._objective_in_contact(diagnostic(hand=names[0], fingers={"index": .02})) is expected


@pytest.mark.parametrize("hand", [None, "unknown"])
def test_multi_hand_history_never_guesses_a_missing_or_unknown_diagnostic_hand(hand):
    history = {"protocol": "robot_rigid_contact_history_v2", "status": "available", "hands": {
        side: {"regions": regions(index_pad=40, palm_mount=40)} for side in ("left", "right")}}
    session = session_for(history, ("left", "right"))
    assert not session._objective_in_contact(diagnostic(hand=hand, fingers={"index": .02}, wrist=True))


@pytest.mark.parametrize("status", ["unavailable", "invalid", "not_sampled"])
def test_incomplete_contact_evidence_cannot_name_a_contact_stall(status):
    history = {"protocol": "robot_rigid_contact_history_v2", "status": status,
               "hands": {"left": {"regions": regions(index_pad=40)}}}
    assert not session_for(history, ("left", "right"))._objective_in_contact(
        diagnostic(hand="left", fingers={"index": .02}))


def test_legacy_single_hand_contact_and_cleared_contacts_remain_distinct():
    history = {"protocol": "robot_rigid_contact_history_v1", "status": "available",
               "regions": regions(index_pad=1)}
    session = session_for(history)
    assert session._objective_in_contact(diagnostic(fingers={"index": .02}))
    history["regions"]["index_pad"]["current_contact_substeps"] = 0
    assert not session._objective_in_contact(diagnostic(fingers={"index": .02}))


def test_converged_posture_does_not_require_contact_to_explain_a_wrist_stall():
    history = {"protocol": "robot_rigid_contact_history_v1", "status": "available",
               "regions": regions(arm=1, index_pad=0)}
    assert session_for(history)._objective_in_contact(
        diagnostic(wrist=True, posture=("right_index_PIP", .001)))


def test_diagnostic_hand_must_own_its_reported_joints():
    session = session_for({}, ("left", "right"))
    invalid = diagnostic(hand="left", posture=("right_index_PIP", .02))
    sanitized = session._safe_diagnostic(invalid)
    assert sanitized.code == "tracking_stalled"
    assert sanitized.hand is None and sanitized.joint_name is None
    assert sanitized.geometry_tolerances is None


# Real URDF controllers + the ordinary action/ACK path. The CPU worker tracks
# commands ideally; the injected drift is a software counterexample, not physics.
from test_agent_service import frozen_inputs, runner
from test_bench_function_controllers import controlled_session, invoke
from test_agent_service import join_action


@pytest.mark.parametrize('controlled_session', ['dual_left', 'dual_right'], indirect=True)
@pytest.mark.parametrize('drift', [False, True])
def test_composite_completion_requires_each_moved_hands_current_goal(controlled_session, drift):
    session, worker, early, models = controlled_session
    late = next(side for side in models if side != early)
    joint = early + '_index_PIP'
    original = worker.on_step
    injected = []

    def acknowledge_and_disturb(w):
        original(w)
        if drift and not session.policy._active[early]:
            injected.append(session.sequence)
            w.raw['joint_positions_rad'][joint] = .2

    worker.on_step = acknowledge_and_disturb
    accepted = invoke(session, 'move_geometry', {
        'action_id': 'simultaneous', 'observation_sequence': session.sequence,
        'timeout_sim_s': 4., 'hand_geometries': {
            side: {'hand_posture_rad': {side + '_index_PIP': goal},
                   'fingertip_frame_id': side + '_hand_flange'}
            for side, goal in ((early, .02), (late, .4))}})
    assert accepted['ok'], accepted
    join_action(session)
    feedback = invoke(session, 'get_action', {'action_id': 'simultaneous'})['result']
    assert feedback['elapsed_sim_s'] < 4.
    assert feedback['robot_execution_state']['sim_time_s'] == session.snapshot.sim_time_s
    if drift:
        assert injected and feedback['status'] == 'failed'
        assert feedback['termination_reason'] == 'controller_failed'
        assert feedback['diagnostic']['code'] == 'objective_lost'
        assert feedback['diagnostic']['hand'] == early
        assert feedback['diagnostic']['joint_name'] == joint
        assert feedback['diagnostic']['joint_error_rad'] == pytest.approx(.18)
        assert feedback['robot_execution_state']['objective_status'] == 'not_reached'
        # Carry the actual controller/HTTP failure through both Agent input
        # projections. This callback reuses a recorded receipt, not new physics.
        from agent.model_client.visibility import feedback_view
        from agent.model_client.program import execute_program
        visible = feedback_view(feedback)
        program = execute_program('hold(.02)', lambda *_: {'feedback': visible})
        compact = program['primitive_results'][0]['feedback']
        for projected in (visible, compact):
            assert projected['status'] == 'failed'
            assert projected['termination_reason'] == 'controller_failed'
            assert projected['diagnostic']['code'] == 'objective_lost'
            assert projected['diagnostic']['hand'] == early
            assert projected['diagnostic']['joint_name'] == joint
            assert projected['diagnostic']['joint_error_rad'] == pytest.approx(.18)
            assert projected['robot_execution_state']['objective_status'] == 'not_reached'
    else:
        assert feedback['status'] == 'completed'
        assert feedback['termination_reason'] == 'controller_converged'
        assert feedback['robot_execution_state']['objective_status'] == 'reached'
    # No failed/checked-only candidate becomes the reference or the native ACK.
    for side, policy in session.policy.sides.items():
        assert policy._pending_step_state is None
        assert dict(zip(policy.geometry.names, policy.previous)) == {
            name: worker.raw['robot']['joint_targets_rad'][name] for name in policy.geometry.names}
    assert session.result is None  # Controller completion is not task completion.
    if drift:
        worker.on_step = original
        repaired = invoke(session, 'move_geometry', {
            'action_id': 'explicit_recovery', 'observation_sequence': session.sequence,
            'timeout_sim_s': 4., 'hand_geometries': {
                side: {'hand_posture_rad': {side + '_index_PIP': goal},
                       'fingertip_frame_id': side + '_hand_flange'}
                for side, goal in ((early, .02), (late, .4))}})
        assert repaired['ok'], repaired
        join_action(session)
        recovered = invoke(session, 'get_action', {'action_id': 'explicit_recovery'})['result']
        assert recovered['status'] == 'completed'
        assert recovered['robot_execution_state']['objective_status'] == 'reached'
        assert session.result is None


@pytest.mark.parametrize('instances', [False, True])
def test_bimanual_descriptor_binds_its_composition_source(instances):
    import hashlib
    from pathlib import Path
    from bench.control.bimanual import BimanualFingertipPolicy
    from bench.control.factory import robot_configuration
    from test_robot_instances import robot
    policy = (BimanualFingertipPolicy(**robot_configuration({'robot': robot()})) if instances else
              BimanualFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., .77]))
    root = Path(__file__).resolve().parents[1]
    for name in ('bench/control/bimanual.py', 'bench/control/embodiment.py'):
        assert policy.provenance['composition_sources'][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    if instances:
        assert 'bench/control/instance_embodiment.py' in policy.provenance['composition_sources']


@pytest.mark.parametrize('controlled_session', ['dual_left', 'dual_right'], indirect=True)
def test_composite_ik_rejection_retains_the_selected_hand_without_executing(controlled_session):
    session, worker, side, _ = controlled_session
    before = deepcopy(worker.raw['robot']['joint_targets_rad'])
    response = invoke(session, 'move_wrist', {
        'action_id': 'unreachable', 'observation_sequence': session.sequence,
        'hand': side, 'timeout_sim_s': 4.,
        'wrist': {'frame_id': 'world', 'position_m': [100., 0., 0.],
                  'quaternion_wxyz': [1., 0., 0., 0.]}})
    assert response['ok'], response
    join_action(session)
    feedback = invoke(session, 'get_action', {'action_id': 'unreachable'})['result']
    assert feedback['status'] == 'failed'
    assert feedback['termination_reason'] == 'begin_rejected'
    assert feedback['diagnostic']['hand'] == side
    assert feedback['diagnostic']['code'] == 'ik_solution_unverified'
    assert not any(operation == 'step' for operation, _ in worker.calls)
    assert worker.raw['robot']['joint_targets_rad'] == before
