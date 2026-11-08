"""Saved success flags cannot substitute for actual Agent and native receipts."""
from copy import deepcopy

import pytest

from scripts.validation.review_kitchen_http import interface_checks


def receipts():
    spec = {'task_id': 'surface_wiping', 'task_release_sha256': 'a'*64, 'track': 'system0'}
    report = {'passed': True, 'task_id': 'surface_wiping', 'episode_id': 'episode',
        'release_manifest_sha256': 'a'*64, 'model_calls': 0, 'server_exit_code': 0,
        'created': {'spec': spec},
        'result': {'spec': deepcopy(spec), 'outcome': 'cancelled', 'model_calls': 0, 'physics_valid': True},
        'server_shutdown': {'episodes': {'episode': {'worker_exit_code': 0, 'finished': True}}},
        'ik': {'solution_found': True, 'advances_physics': False, 'mutates_controller': False},
        'agent_interface_checks': []}
    for phase in ('created', 'before_finish'):
        report['agent_interface_checks'].append({'phase': phase, 'family': 'kitchen',
            'task_id': 'surface_wiping', 'task_release_sha256': 'a'*64, 'track': 'system0',
            'catalog_release_verified': True, 'agent_public_parsing_verified': True,
            'agent_ready_contract_verified': True, 'static_model_requirements_verified': True,
            'read_only_verified': True, 'actions_submitted': 0, 'model_calls': 0})
    calls = []
    for i, name in enumerate(('move_geometry', 'move_geometry', 'hold')):
        for replay in range(2):
            calls.append({'status': 200, 'request': {'function': name,
                'arguments': {'action_id': str(i)}}})
    agent = [{'request': {'function': name}, 'response': {'ok': True}}
             for name in ('observe', 'get_task', 'observe')*2]
    return report, calls, agent


def test_complete_saved_receipts_are_accepted_only_with_the_actual_release_pin():
    report, calls, agent = receipts()
    assert interface_checks(report, calls, agent, 'a'*64)['passed']
    assert not interface_checks(report, calls, agent, 'b'*64)['passed']


@pytest.mark.parametrize('fault', ['legacy_agent', 'missing_agent_end', 'model_call',
    'native_exit', 'forced_shutdown', 'success_claim', 'mutating_ik', 'partial_agent_calls',
    'http_failure', 'no_motion_return', 'changed_replay', 'wrong_episode', 'false_pass_flag'])
def test_report_pass_flag_cannot_hide_incomplete_or_mismatched_receipts(fault):
    report, calls, agent = receipts()
    if fault == 'legacy_agent':
        del report['agent_interface_checks'][0]['agent_ready_contract_verified']
    elif fault == 'missing_agent_end':
        report['agent_interface_checks'].pop()
    elif fault == 'model_call':
        report['result']['model_calls'] = 1
    elif fault == 'native_exit':
        report['server_shutdown']['episodes']['episode']['worker_exit_code'] = -9
    elif fault == 'forced_shutdown':
        report['forced_server_shutdown'] = True
    elif fault == 'success_claim':
        report['result']['outcome'] = 'success'
    elif fault == 'mutating_ik':
        report['ik']['advances_physics'] = True
    elif fault == 'partial_agent_calls':
        agent.pop()
    elif fault == 'http_failure':
        calls[0]['status'] = 500
    elif fault == 'no_motion_return':
        calls = calls[:2]+calls[-2:]
    elif fault == 'changed_replay':
        calls[1]['request']['arguments']['action_id'] = 'different'
    elif fault == 'wrong_episode':
        report['result']['spec']['task_id'] = 'other'
    else:
        report['passed'] = False
    assert not interface_checks(report, calls, agent, 'a'*64)['passed']
