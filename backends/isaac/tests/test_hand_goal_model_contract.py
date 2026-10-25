"""Descriptor-bound model claims; mocked models/physics are not native evidence."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from bench.paths import load_json
from bench.protocol.schemas import HandGoalSemantics, PolicyDescriptor
from agent.model_client import runner
from test_multimode_runner import loop


def descriptor(semantics=True):
    value = {'policy_id': 'classical_fingertip', 'version': 'candidate',
             'implementation_sha256': 'a'*64, 'trained': False,
             'skills': ['move_geometry', 'hold'], 'geometry_inputs': ['wrist_pose']}
    if semantics:
        value['hand_goal_semantics'] = HandGoalSemantics().model_dump(mode='json')
    return value


@pytest.mark.parametrize('field', list(HandGoalSemantics.model_fields))
@pytest.mark.parametrize('bad', [False, 0, 'undeclared_behavior'])
def test_semantic_values_are_exact_declared_literals(field, bad):
    value = descriptor()
    value['hand_goal_semantics'][field] = bad
    with pytest.raises(ValidationError):
        PolicyDescriptor.model_validate(value)


def test_contract_rejects_extra_or_inapplicable_claims_and_preserves_legacy_null():
    value = descriptor()
    value['hand_goal_semantics']['private_object_pose'] = [1, 2, 3]
    with pytest.raises(ValidationError):
        PolicyDescriptor.model_validate(value)
    for key, replacement in [('skills', ['hold']), ('geometry_inputs', [])]:
        value = descriptor()
        value[key] = replacement
        with pytest.raises(ValidationError, match='hand goal semantics require'):
            PolicyDescriptor.model_validate(value)
    value = descriptor(False)
    assert PolicyDescriptor.model_validate(value).hand_goal_semantics is None
    value['hand_goal_semantics'] = None
    assert PolicyDescriptor.model_validate(value).hand_goal_semantics is None


@pytest.mark.parametrize('mode', ['action', 'tools', 'code'])
def test_only_advertised_geometry_semantics_enable_retention_rules(mode):
    assert runner.HAND_REFERENCE_RULES in runner.model_system_prompt('fingertip', mode, descriptor())
    for value in (None, descriptor(False), descriptor(False) | {'hand_goal_semantics': None}):
        prompt = runner.model_system_prompt('fingertip', mode, value)
        assert runner.HAND_REFERENCE_RULES not in prompt
        assert 'No retained-hand-goal contract was advertised' in prompt
    joint = runner.model_system_prompt('joints', 'action', descriptor())
    assert runner.HAND_REFERENCE_RULES not in joint
    assert 'omitted joints keep their commanded targets' in joint


@pytest.mark.parametrize('mode', ['action', 'tools', 'code'])
@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_actual_prompt_and_descriptor_are_pinned_before_create_and_reach_model(monkeypatch, tmp_path, mode, tactile):
    response = {'action': 'stop'}
    if mode == 'tools':
        response = {'tool': 'stop', 'arguments': {}}
    if mode == 'code':
        response = 'stop()'
    args, bench, model = loop(monkeypatch, tmp_path, [response], mode=mode)
    args.tactile_input = tactile
    semantics = HandGoalSemantics().model_dump(mode='json')
    bench.policy_inventory[0]['hand_goal_semantics'] = deepcopy(semantics)
    # Fixture has a reset snapshot: unsampled values remain null, never zero.
    bench.obs['tactile'] = {'protocol': 'rigid_fingertip_contact_proxy_v1',
        'mode': 'simulated_rigid_contact_proxy', 'status': 'not_sampled', 'sim_time_s': 0.,
        'window_start_sim_s': None, 'window_duration_s': None,
        'window_completeness': 'not_available', 'force_calibrated': False,
        'frame_id': 'scalar_invariant', 'coverage': {'rigid': True},
        'fingers': {finger: {'status': 'not_sampled', 'rigid_contact_any': None,
                            'normal_impulse_sum_ns': None, 'mean_normal_load_n': None}
                    for finger in ('thumb', 'index', 'middle', 'ring', 'little')}}
    create = bench.create
    def negotiate(request):
        settings = load_json(args.output/'agent_config.json')
        assert request['agent_config_sha256'] == runner.canonical_hash(settings)
        assert runner.HAND_REFERENCE_RULES in settings['system_prompt']
        assert settings['controller']['hand_goal_semantics'] == semantics
        ready = create(request)
        for container in (ready['spec'], ready['capabilities']):
            container['system0']['hand_goal_semantics'] = deepcopy(semantics)
        return ready
    bench.create = negotiate
    report = runner.run(args)
    assert report['error_type'] is None and len(model.requests) == 1
    assert not bench.commands
    settings = load_json(args.output/'agent_config.json')
    archived = load_json(args.output/'decisions/0000/model_request.json')
    actual_prompt, turns, _ = model.requests[0]
    assert actual_prompt == settings['system_prompt'] == archived['system_prompt']
    assert runner.TACTILE_RULES in actual_prompt
    assert 'whole hand is collision-free' in actual_prompt
    assert 'not the failed goal or an older' in actual_prompt
    text = json.loads(turns[-1]['content'][0]['text'])
    assert text['capabilities']['system0']['hand_goal_semantics'] == semantics
    if tactile == 'off':
        assert 'tactile' not in text['observation']
        assert text['capabilities']['tactile_available'] is False
    else:
        assert text['observation']['tactile']['status'] == 'not_sampled'
        assert text['observation']['tactile']['fingers']['thumb']['mean_normal_load_n'] is None
    assert settings['schema'] == 'online_visual_agent.v11'
    assert settings['interface_contract']['version'] == 'agent_control_v6'
