"""Public-only probe scheduling; no model or native success is inferred here."""
from copy import deepcopy
import json

import pytest

from scripts.validation.card_practice_smoke import PublicProbe


@pytest.mark.parametrize('side', ['left', 'right'])
def test_probe_uses_owned_public_wrist_and_checks_consecutive_moves(side):
    probe = PublicProbe(side)
    peer = 'right' if side == 'left' else 'left'
    wrist = {'frame_id': 'world', 'position_m': [.65, -.35, 1.5],
             'quaternion_wxyz': [1., 0., 0., 0.]}
    initial = deepcopy(wrist)
    proprio = {'commanded_joint_positions_rad': {side+'_joint': 0., peer+'_joint': .2},
               'hand_geometries': {side: {'wrist': wrist}}}
    feedback = None
    decisions = []
    for _ in range(8):
        context = {'state': {'observation': {'proprioception': proprio}}, 'last_feedback': feedback}
        turns = [{'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(context)}]}]
        decision = json.loads(probe.generate('', turns).text)
        decisions.append(decision)
        if decision.get('function') in ('hold', 'move_geometry'):
            name = decision['function']
            duration = decision['arguments'].get('duration_s', .5)
            feedback = {'function': name, 'ok': True, 'result': {'status': 'completed',
                'termination_reason': 'time_hold_completed' if name == 'hold' else 'controller_converged',
                'episode_finished': False, 'elapsed_sim_s': duration,
                'contact_history': {'status': 'available', 'sampled_substeps': round(duration/.0005),
                    'physics_dt_s': .0005, 'duration_s': duration, 'hands': {'left': {}, 'right': {}}}}}
            if name == 'move_geometry':
                assert set(decision['arguments']['hand_geometries']) == {side}
                proprio['hand_geometries'][side]['wrist'] = deepcopy(
                    decision['arguments']['hand_geometries'][side]['wrist'])
    assert [d.get('function', 'stop') for d in decisions] == [
        'get_task', 'hold', 'get_task', 'move_geometry', 'move_geometry', 'hold', 'get_task', 'stop']
    assert probe.moved_distance_m == pytest.approx(.002)
    assert probe.return_error_m == 0.
    assert proprio['hand_geometries'][side]['wrist'] == initial
