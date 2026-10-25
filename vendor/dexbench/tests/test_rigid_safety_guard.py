"""CPU contracts and replayed solver separation; no native physics qualification."""
from copy import deepcopy
import json

import pytest

from bench.evaluation.rigid_safety import RigidContactSafetyGuard, exceeds_rigid_penetration
from bench.evaluation.physics import _rigid_separations, audit_source_paths
from bench.paths import load_json
from bench.runtime.worker import guarded_native_step
from bench.runtime.service import ServiceError
from test_agent_service import (runner, frozen_inputs, AcknowledgedTeacher, teacher_session,
                               skill_action, join_action, PRIVATE_MARKER)
from test_board_game_service import game, call, hold, intent, endpoint, headers, steps


def window(separation=-.004029951989650726, *, end=.02):
    """Exact worst separation from task0 attempt02; remaining envelope synthetic."""
    start = end - .02
    return {'sim_time_s': end, 'contacts': {
        'source': 'PhysX stage-bound rigid contact tensor; normal impulses', 'errors': [],
        'sampling': {'protocol': 'native_rigid_contact_sampling_v1', 'complete': True,
            'physics_dt_s': .0005, 'start_sim_time_s': start, 'end_sim_time_s': end,
            'native_start_time_s': start, 'native_end_time_s': end,
            'native_start_step_index': round(start/.0005), 'native_end_step_index': round(end/.0005),
            'sampled_substeps': 40},
        'rigid': [{'time_s': start+.0005, 'actor0': '/World/Private/'+PRIVATE_MARKER,
                   'actor1': '/World/Robot/right_thumb_elastomer', 'min_separation_m': separation}]}}


@pytest.mark.parametrize('value,stop', [(-.002,False),(-.001,False),(0.,False),(.001,False),
                                      (-.0020000001,True),(-.004029951989650726,True)])
def test_online_gate_equals_existing_offline_gate(value, stop):
    observation = window(value)
    result = RigidContactSafetyGuard().observe(observation)
    assert (result is not None) is stop
    assert exceeds_rigid_penetration(value) is stop
    assert (_rigid_separations({'observations':[observation]})['status'] == 'failed') is stop


def test_intermediate_substep_violation_is_not_hidden_by_final_contact():
    obs = window(-.001)
    row = obs['contacts']['rigid'][0]
    obs['contacts']['rigid'] += [dict(row,time_s=.01,min_separation_m=-.004029951989650726),
                                dict(row,time_s=.02,min_separation_m=-.0001)]
    guard = RigidContactSafetyGuard()
    trigger = guard.observe(obs)
    assert trigger['contact_time_s'] == .01
    assert trigger['minimum_separation_m'] == -.004029951989650726
    assert trigger['contact_row_index'] == 1
    assert guard.observe(window(-.1,end=.04)) == trigger
    assert len(json.dumps(trigger)) < 2048


@pytest.mark.parametrize('mutation', ['incomplete','skipped','off_grid','nan','missing'])
def test_native_guard_rejects_incomplete_or_corrupt_evidence(mutation):
    obs = window()
    if mutation=='incomplete':obs['contacts']['sampling']['complete']=False
    elif mutation=='skipped':obs['contacts']['sampling']['sampled_substeps']=39
    elif mutation=='off_grid':obs['contacts']['rigid'][0]['time_s']=.0007
    elif mutation=='nan':obs['contacts']['rigid'][0]['min_separation_m']=float('nan')
    else:obs.pop('contacts')
    with pytest.raises(ValueError):RigidContactSafetyGuard().observe(obs)


def test_legacy_worker_without_native_receipts_remains_compatible():
    assert RigidContactSafetyGuard(required=False).observe({'sim_time_s':.02}) is None
    assert any(p.name=='rigid_safety.py' for p in audit_source_paths())


def test_worker_latches_after_full_window_and_refuses_any_later_physics():
    class Env:
        steps = 0
        def step(self, command):
            self.steps += 1
            return window()
    env, guard = Env(), RigidContactSafetyGuard()
    original = window()
    assert guarded_native_step(env, {}, guard) == original
    assert env.steps == 1
    with pytest.raises(ValueError,match='latched'):
        guarded_native_step(env, {}, guard)
    assert env.steps == 1


def test_service_commits_ack_then_stops_before_success_and_any_next_step(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher()
    session=teacher_session(make,teacher)
    def on_step(worker):
        worker.raw.update(window(end=worker.raw['sim_time_s']))
        worker.score.update(success=True)
    workers[-1].on_step=on_step
    session.submit(skill_action(session));join_action(session)
    assert teacher.commits==1 and teacher.discards==0 and not teacher.pending
    assert session.sequence==1 and session.clock.sim_time_s==pytest.approx(.02)
    assert session.result.outcome=='safety_stop' and session.result.physics_valid is False
    assert session.score['success'] is True  # task evidence retained; never public task success
    assert session.result.completion_sim_s is None
    assert len([op for op,_ in workers[-1].calls if op=='step'])==1
    trigger=load_json(session.directory/'private/physics_safety_stop.json')
    assert trigger['action_id']=='skill_1' and trigger['observation_sequence']==1
    assert PRIVATE_MARKER in trigger['actor0']
    assert PRIVATE_MARKER not in json.dumps(session.result.model_dump(mode='json'))
    assert PRIVATE_MARKER not in json.dumps(session.status('skill_1'))
    with pytest.raises(ServiceError,match='finished'):
        session.submit({**skill_action(session),'action_id':'another'})
    assert len([op for op,_ in workers[-1].calls if op=='step'])==1


def test_board_guard_keeps_player_authority_and_stops_shared_episode(game):
    with game() as g:
        # The guard is not an alternate action route or permission grant.
        response=call(g,'hold',{'action_id':'wrong_turn','observation_sequence':0,'duration_s':.1},
                      player='player_1')
        assert response.status_code==403
        assert not any(op=='step' for op,_ in g.worker.calls)
        g.worker.on_step=lambda worker:worker.raw.update(window(end=worker.raw['sim_time_s']))
        result,_=hold(g,'guard_stop',duration=.1)
        assert result['result']['status']=='failed'
        assert g.session.result.outcome=='safety_stop' and not g.session.result.physics_valid
        assert len([op for op,_ in g.worker.calls if op=='step'])==1
        for player in ['player_0','player_1']:
            response=call(g,'hold',{'action_id':'after_'+player,'observation_sequence':g.session.sequence,
                                  'duration_s':.02},player=player)
            assert response.status_code in (403,409)
        assert len([op for op,_ in g.worker.calls if op=='step'])==1


@pytest.mark.parametrize('winning_move', [False, True], ids=['turn_change', 'winning_move'])
def test_board_commit_in_same_ack_survives_rigid_safety_stop(game, winning_move):
    with game() as g:
        g.worker.settled_pending = True
        if winning_move:
            for number, (piece, q, r, player) in enumerate([
                    ('black_00', 0, 0, 'player_0'), ('white_00', 0, 1, 'player_1'),
                    ('black_01', 1, 0, 'player_0'), ('white_01', 1, 1, 'player_1')]):
                intent(g, piece, q, r, player=player)
                hold(g, f'setup_{number}', 1., player=player)
        turn_before = g.session.board_game.state.turn_index
        request = g.worker.request
        committed_acks = []

        def contact_on_committing_ack(operation, **payload):
            # The frozen referee first commits a real CPU evidence transition.
            # Only then add synthetic contact evidence to that same complete ACK.
            reply = request(operation, **payload)
            if operation == 'step' and reply['board_game_turn_changed']:
                reply['observation'].update(window(end=reply['observation']['sim_time_s']))
                g.worker.raw = deepcopy(reply['observation'])
                committed_acks.append(deepcopy(reply))
            return reply

        g.worker.request = contact_on_committing_ack
        intent(g, 'black_02' if winning_move else 'black_00', 2 if winning_move else 0, 0)
        result, _ = hold(g, 'commit_and_stop', 1.)
        assert len(committed_acks) == 1
        ack = committed_acks[0]
        assert ack['board_game']['turn_index'] == turn_before + 1
        assert ack['board_game']['commit_count'] == turn_before + 1
        assert ack['board_game']['pending_move'] is None
        assert ack['score']['success'] is winning_move
        assert result['result']['status'] == 'failed'
        assert result['result']['termination_reason'] == 'episode_terminated'
        assert g.session.clock.sim_time_s == ack['observation']['sim_time_s']
        assert g.session.board_game.last_targets == steps(g)[-1]['command']['joint_positions_rad']
        assert g.session.board_game.actions['commit_and_stop'] == {
            'player_id': 'player_0', 'turn_index': turn_before}
        trigger = load_json(g.session.directory/'private/physics_safety_stop.json')
        assert trigger['action_id'] == 'commit_and_stop'
        assert trigger['observation_sequence'] == g.session.sequence

        before = deepcopy(steps(g))
        shared_result = None
        for player in ('player_0', 'player_1'):
            response = g.client.get(endpoint(g, player)+'/result', headers=headers(g, player))
            assert response.status_code == 200, response.text
            assert PRIVATE_MARKER not in response.text
            final = response.json()
            assert final['episode']['outcome'] == 'safety_stop'
            assert final['episode']['physics_valid'] is False
            assert final['episode']['completion_sim_s'] is None
            board = final['board_game']
            assert board['turn_index'] == turn_before + 1
            assert board['commit_count'] == turn_before + 1
            assert board['pending_move'] is None and board['terminal']
            if winning_move:
                assert board == ack['board_game']
                assert board['status']['outcome'] == 'win' and board['winner'] == 'player_0'
            else:
                assert ack['board_game']['current_player'] == 'player_1'
                assert board['status']['outcome'] == 'aborted' and board['winner'] is None
                assert board['status']['end_reason'] == 'runtime_error'
            if shared_result is not None:
                assert final == shared_result
            shared_result = final
            rejected = call(g, 'hold', {'action_id': 'after_commit_'+player,
                'observation_sequence': g.session.sequence, 'duration_s': .02}, player=player)
            assert rejected.status_code in (403, 409)
        assert steps(g) == before


def test_practice_warns_for_observed_peg_penetration_but_strict_stops():
    from bench.adapters.pico.contact_guard import TeleopContactGuard
    obs = window(-.0024472028017044067)
    strict, practice = TeleopContactGuard(), TeleopContactGuard(practice=True)
    assert strict.observe(obs) == RigidContactSafetyGuard().observe(obs)
    assert practice.observe(obs) is None
    practice.assert_can_step()
    assert practice.warning['minimum_separation_m'] == obs['contacts']['rigid'][0]['min_separation_m']
    assert '2.45 mm' in practice.message(.02)
    assert practice.warning_windows == 1
    assert practice.observe(window(-.001, end=.04)) is None
    assert practice.warning is None and practice.streak == 0
    assert practice.message(3.) == ''


def test_practice_warns_without_stopping_for_severe_persistent_penetration():
    from bench.adapters.pico.contact_guard import TeleopContactGuard
    guard = TeleopContactGuard(practice=True)
    for i in range(1, 51):
        assert guard.observe(window(-.00848, end=i*.02)) is None
        guard.assert_can_step()
    assert guard.trigger is None
    assert guard.warning_windows == guard.streak == 50
    assert guard.warning['minimum_separation_m'] == -.00848
    assert '8.48 mm' in guard.message(1.)
    assert 'practice continues' in guard.message(1.)
    assert guard.observe(window(-.001, end=1.02)) is None
    assert guard.warning is None


def test_practice_interrupted_windows_do_not_accumulate_and_invalid_receipts_still_fail():
    from bench.adapters.pico.contact_guard import TeleopContactGuard
    guard = TeleopContactGuard(practice=True)
    for i in range(1, 30):
        assert guard.observe(window(-.003, end=i*.04)) is None
    assert guard.streak == 1
    obs = window(-.003)
    obs['contacts']['sampling']['complete'] = False
    with pytest.raises(ValueError, match='complete sampling'):
        guard.observe(obs)


def test_practice_contact_option_requires_unscored_run():
    from scripts.teleop.pico_isaac import parser, load_inputs
    args = parser().parse_args(['--practice-contact-warnings'])
    with pytest.raises(ValueError, match='requires --no-task-scoring'):
        load_inputs(args)
    args.sandbox = True
    scene, _, _ = load_inputs(args)
    assert 'native_task_bundle' not in scene
