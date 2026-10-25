"""Synthetic receipts test scoring logic only, never physical feasibility."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.evaluator import SequenceEvaluator
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import GEARS, TOOLS
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import build_task_inputs, yaw_quaternion


@pytest.fixture(scope='module')
def task():
    return build_task_inputs()['task']


def row(a, b, tick, time, vector=(0., 0., 1.e-4)):
    return {'actor0': a, 'actor1': b, 'time_s': time-.02+tick*.0005,
        'impulse_magnitude_sum_ns': float(np.linalg.norm(vector)), 'impulse_vector_ns': list(vector),
        'min_separation_m': 0., 'normal_length_max_abs_error': 0., 'point_count': 1,
        'nonzero_impulse_normal_count': 1, 'impulse_components': 'normal_only'}


def observation(task, index, positions, turns, *, drive=False, tool='socket_crank', chain=True,
                opposing=True, direct=False, supported=True, released=True):
    c = task['evaluation']
    time, first = index*.02, index == 0
    objects = deepcopy(c['initial_objects'])
    for name in GEARS+TOOLS:
        objects[name].update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])
    for i, name in enumerate(GEARS):
        objects[name]['position_m'] = [float(v) for v in positions[name]]
        # Test trajectories use input-equivalent degrees; the actual 32-tooth
        # idler rotates only 20/32 as far, plus its half-tooth assembly phase.
        physical = turns[i]*(20/32 if i == 1 else 1.)
        objects[name]['quaternion_wxyz'] = yaw_quaternion(physical+(180/32 if i == 1 else 0.))
    contacts = []
    if not first:
        for tick in range(1, 41):
            if supported:
                contacts.extend(row(c['actors'][name], c['fixture_actor']+'/collision_000', tick, time) for name in GEARS)
            if drive or not released:
                tool_actor = c['actors'][tool]
                contacts.append(row(tool_actor, '/World/Robot/right_thumb_IP', tick, time, (1.e-4, 0., 0.)))
                contacts.append(row(tool_actor, '/World/Robot/right_index_DP', tick, time, (-1.e-4 if opposing else 1.e-4, 0., 0.)))
                contacts.append(row(tool_actor, c['actors']['input_gear'], tick, time))
                contacts.append(row(c['actors']['input_gear'], c['actors']['idler_gear'], tick, time))
                if chain:
                    contacts.append(row(c['actors']['idler_gear'], c['actors']['output_gear'], tick, time))
            if direct:
                contacts.append(row(c['actors']['output_gear'], '/World/Robot/right_index_MP', tick, time))
    start = time if first else time-.02
    return {'time_s': time, 'sim_time_s': time, 'last_duration_s': 0. if first else .02,
        'metrics': {'state_finite': True}, 'terminated': False, 'objects': objects,
        'contacts': {'source': 'PhysX stage-bound rigid contact tensor; normal impulses', 'errors': [],
            'rigid': contacts, 'sampling': {'protocol': 'native_rigid_contact_sampling_v1', 'complete': True,
                'physics_dt_s': .0005, 'sampled_substeps': 0 if first else 40,
                'native_start_step_index': index*40 if first else (index-1)*40,
                'native_end_step_index': index*40, 'start_sim_time_s': start, 'end_sim_time_s': time,
                'native_start_time_s': start, 'native_end_time_s': time}}}


def sequence(task, **options):
    c, evaluator = task['evaluation'], SequenceEvaluator(task)
    initial = {n: np.array(c['initial_objects'][n]['position_m']) for n in GEARS}
    target = {n: np.array(c['target_bottom_centers_m'][n]) for n in GEARS}
    evaluator.update(observation(task, 0, initial, [0., 0., 0.]))
    index = 0
    for i in range(1, 101):
        index += 1
        p = {n: initial[n]+(target[n]-initial[n])*i/100 for n in GEARS}
        result = evaluator.update(observation(task, index, p, [0., 0., 0.]))
        assert not result['reason'].startswith('invalid'), result
    for _ in range(17):
        index += 1
        final = observation(task, index, target, [0., 0., 0.])
        result = evaluator.update(final)
    assert evaluator.baseline is not None
    turns = [0., 0., 0.]
    for i in range(1, options.get('drive_steps', 150)+1):
        index += 1
        turns = [i*.2, -i*.2*options.get('idler_ratio', 1.), i*.2*options.get('output_ratio', 1.)]
        final = observation(task, index, target, turns, drive=True,
            **{k: v for k, v in options.items() if k in ('tool', 'chain', 'opposing', 'direct')})
        result = evaluator.update(final)
    for _ in range(options.get('hold_steps', 17)):
        index += 1
        final = observation(task, index, target, turns, released=options.get('released', True),
            supported=options.get('supported', True))
        if options.get('zero_touch'):
            for tick in range(1, 41):
                contact = row(c['actors'][options['zero_touch']], '/World/Robot/right_index_MP', tick, index*.02, (0., 0., 0.))
                contact['nonzero_impulse_normal_count'] = 0
                final['contacts']['rigid'].append(contact)
        result = evaluator.update(final)
    return evaluator, result, final


@pytest.mark.parametrize('tool', TOOLS)
def test_any_free_tool_earns_score_only_from_complete_actual_contact_transmission(task, tool):
    evaluator, result, final = sequence(task, tool=tool)
    assert result['success'], result
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(30.))
    assert evaluator.inspect_conditions(final) == result  # No duplicate consumption.


def test_larger_idler_retains_actual_angles_but_normalizes_credit(task):
    _, result, _ = sequence(task)
    assert result['success']
    assert result['signed_gear_turns_rad'] == pytest.approx({
        'input_gear': math.radians(30.), 'idler_gear': math.radians(-18.75),
        'output_gear': math.radians(30.)})
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(30.))
    assert result['maximum_pair_phase_error_rad'] < 1.e-12


def test_normal_backlash_takeup_can_finish_without_precise_net_turn_ratios(task):
    # Synthetic scoring regression only: the two meshes take up .8 and 1.0
    # input-equivalent degrees before steady 20/32/20 transmission begins.
    # Native engineering probes and their hand-contact limitations are separate.
    evaluator, _, _ = sequence(task, drive_steps=0, hold_steps=0)
    for i in range(1, 151):
        input_turn = i*.2
        turns = (input_turn, -max(input_turn-.8, 0.), max(input_turn-1.8, 0.))
        result = evaluator.update(next_observation(task, evaluator, turns=turns, drive=True))
    for _ in range(17):
        result = evaluator.update(next_observation(task, evaluator, turns=turns))
    actual = result['signed_gear_turns_rad']
    output_ratio_error = abs(actual['output_gear']/actual['input_gear']-1.)
    assert .05 < output_ratio_error < .10
    assert result['maximum_pair_phase_error_rad'] == pytest.approx(math.radians(1.))
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(28.2))
    assert result['success'], result


def test_clearly_wrong_ratio_still_fails_with_target_credit_and_pair_phase_in_bounds(task):
    # Split the discrepancy across both meshes so only the final 12% output
    # ratio error fails: output stays in-sector, credit >=25 degrees, phase <2.
    _, result, _ = sequence(task, idler_ratio=1.06, output_ratio=1.12)
    assert math.radians(25.) <= result['signed_gear_turns_rad']['output_gear'] <= math.radians(35.)
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(30.))
    assert result['maximum_pair_phase_error_rad'] == pytest.approx(math.radians(1.8))
    assert not result['success']
    assert result['referee']['reason_codes'] == ['direction_ratio_required']


def test_old_equal_wheel_relation_cannot_score_new_larger_idler(task):
    _, result, _ = sequence(task, idler_ratio=32/20)
    assert not result['success']
    assert result['maximum_pair_phase_error_rad'] == pytest.approx(math.radians(18.))


@pytest.mark.parametrize('key,value', [
    ('gear_teeth', {'input_gear': 20, 'idler_gear': 20, 'output_gear': 20}),
    ('input_equivalent_turn_weights', [1., 1., 1.]),
    ('signed_angle_ratios', {'idler_per_input': -1., 'output_per_input': 1.}),
    ('shaft_centers_world_xy_m', [[.4, -.1], [.4, 0.], [.4, .1]]),
])
def test_frozen_unequal_gear_contract_cannot_be_reinterpreted(task, key, value):
    altered = deepcopy(task)
    altered['evaluation'][key] = value
    with pytest.raises(ValueError, match='geometry contract'):
        SequenceEvaluator(altered)


@pytest.mark.parametrize('options', [
    {'chain': False}, {'opposing': False}, {'direct': True}, {'output_ratio': 0.},
    {'idler_ratio': -1.}, {'output_ratio': .7}, {'released': False},
    {'hold_steps': 10}, {'supported': False},
])
def test_pose_or_tool_identity_alone_cannot_score(task, options):
    _, result, _ = sequence(task, **options)
    assert not result['success']


def test_complete_receipts_cannot_hide_missing_native_coverage(task):
    evaluator, _, final = sequence(task, hold_steps=0)
    broken = deepcopy(final)
    broken['time_s'] += .02
    broken['sim_time_s'] += .02
    broken['contacts']['sampling']['sampled_substeps'] = 39
    result = evaluator.update(broken)
    assert not result['success'] and result['reason'].startswith('invalid_observation')


def test_engineering_reset_and_changed_threshold_are_rejected(task):
    altered = deepcopy(task)
    altered['engineering_probe'] = True
    with pytest.raises(ValueError): SequenceEvaluator(altered)
    altered = deepcopy(task)
    altered['evaluation']['minimum_motion_chain_fraction'] = 0.
    with pytest.raises(ValueError): SequenceEvaluator(altered)


def test_preassembled_initial_pose_cannot_earn_original_task_credit(task):
    evaluator = SequenceEvaluator(task)
    obs = observation(task, 0, task['evaluation']['target_bottom_centers_m'], [0., 0., 0.])
    result = evaluator.update(obs)
    assert not result['success'] and 'Initial body pose differs' in result['reason']


def test_contact_chain_must_exist_in_the_same_native_substeps(task):
    evaluator = SequenceEvaluator(task)
    evaluator.previous_time = evaluator.native_time = 0.
    evaluator.native_step = 0
    obs = observation(task, 1, task['evaluation']['target_bottom_centers_m'], [0., 0., 0.], drive=True)
    filtered = []
    for contact in obs['contacts']['rigid']:
        robot_pair = any(p.startswith('/World/Robot/') for p in (contact['actor0'], contact['actor1']))
        mesh_pair = {contact['actor0'], contact['actor1']} == {
            task['evaluation']['actors']['idler_gear'], task['evaluation']['actors']['output_gear']}
        if robot_pair and contact['time_s'] > .01 or mesh_pair and contact['time_s'] <= .01:
            continue
        filtered.append(contact)
    obs['contacts']['rigid'] = filtered
    evidence, _, _ = evaluator._contacts(obs, .02, False)
    assert evidence['chain'] == 0.


def test_reverse_motion_removes_credit_even_without_a_contact_chain(task):
    evaluator, result, final = sequence(task, hold_steps=0)
    index = round(final['time_s']/.02)
    target = task['evaluation']['target_bottom_centers_m']
    for value in [*np.linspace(29.8, 20., 50), *np.linspace(20.2, 30., 50)]:
        index += 1
        turn = float(value)
        result = evaluator.update(observation(task, index, target, [turn, -turn, turn], drive=False))
    for _ in range(17):
        index += 1
        result = evaluator.update(observation(task, index, target, [30., -30., 30.]))
    assert not result['success']
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(20.))


def test_frozen_evaluator_loads_with_its_complete_local_helpers(task, tmp_path):
    from bench.isaacsim.release import _frozen_evaluator_module
    import scenes.workrooms.mechanical_assembly.native_evidence as evidence
    import scenes.workrooms.mechanical_assembly.referee as referee
    import scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.evaluator as implementation
    target = tmp_path/'evaluator'
    target.mkdir()
    source = Path(implementation.__file__).parent
    shutil.copy2(source/'evaluator.py', target/'sequence_evaluator.py')
    shutil.copy2(source/'rules.py', target/'rules.py')
    shutil.copy2(evidence.__file__, target/'native_evidence.py')
    shutil.copy2(referee.__file__, target/'referee.py')
    (tmp_path/'release_manifest.json').write_text('{"test_only": "L5 complete frozen dependencies"}')
    module = _frozen_evaluator_module(tmp_path)
    assert not module.SequenceEvaluator(task).inspect_conditions()['success']
    assert Path(module.read_native_contact_window.__code__.co_filename).resolve().is_relative_to(tmp_path)
    assert Path(module.verdict.__code__.co_filename).resolve().is_relative_to(tmp_path)


@pytest.mark.parametrize('body', ['socket_crank', 'output_gear'])
def test_zero_pressure_touching_manifold_is_not_release(task, body):
    _, result, _ = sequence(task, zero_touch=body)
    assert not result['success']
    assert result['release_stable_s'] == 0.


@pytest.mark.parametrize('downward,swapped,expected', [(2.e-4, False, False), (2.e-4, True, False),
                                                    (5.e-5, False, True), (5.e-5, True, True)])
def test_fixture_support_is_signed_net_across_all_native_actors(task, downward, swapped, expected):
    evaluator = SequenceEvaluator(task)
    evaluator.previous_time = evaluator.native_time = 0.
    evaluator.native_step = 0
    obs = observation(task, 1, task['evaluation']['target_bottom_centers_m'], [0., 0., 0.], drive=True)
    gear = task['evaluation']['actors']['input_gear']
    fixture = task['evaluation']['fixture_actor']+'/another_collider'
    for tick in range(1, 41):
        a, b = (fixture, gear) if swapped else (gear, fixture)
        obs['contacts']['rigid'].append(row(a, b, tick, .02, (0., 0., downward if swapped else -downward)))
    contacts, _, _ = evaluator._contacts(obs, .02, False)
    assert contacts['supported'] is expected
    assert contacts['chain'] == (1. if expected else 0.)


def next_observation(task, evaluator, turns=(30., -30., 30.), **options):
    return observation(task, round(evaluator.previous_time/.02)+1,
        task['evaluation']['target_bottom_centers_m'], list(turns), **options)


def test_public_contract_declares_chinese_rules_and_only_coarse_verdicts(task):
    from bench.protocol.task_referee import PublicContract, project_referee_state
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import thresholds
    contract = task['public_rules']['referee']
    PublicContract.model_validate(contract)
    assert contract['thresholds'] == thresholds()
    assert all(any('\u4e00' <= char <= '\u9fff' for char in text)
        for text in [contract['instruction_zh'], *contract['phases'].values(),
            *contract['reason_descriptions'].values(), *contract['success_conditions'], *contract['recovery']])
    evaluator = SequenceEvaluator(task)
    before = evaluator.inspect_conditions()
    assert evaluator.last_result is None  # Inspection never consumes a sample.
    assert project_referee_state(before, contract)['state'] == 'ongoing'
    evaluator, result, _ = sequence(task)
    public = project_referee_state(result, contract)
    assert public['state'] == 'success' and public['terminal'] and not public['recoverable']
    assert set(public) == {'schema', 'state', 'phase', 'reason_codes', 'terminal', 'recoverable'}
    public['reason_codes'].clear()
    assert evaluator.inspect_conditions()['referee']['reason_codes'] == ['success']


def test_success_is_latched_without_consuming_or_validating_later_samples(task):
    evaluator, accepted, _ = sequence(task)
    clocks = evaluator.previous_time, evaluator.native_step, evaluator.native_time
    assert evaluator.update({}) == accepted
    later = next_observation(task, evaluator, turns=(0., 0., 0.), direct=True)
    assert evaluator.update(later) == accepted
    assert (evaluator.previous_time, evaluator.native_step, evaluator.native_time) == clocks
    accepted['success'] = False
    accepted['referee']['state'] = 'ongoing'
    assert evaluator.inspect_conditions()['success']


@pytest.mark.parametrize('fault', ['coverage', 'clock', 'nan_pose', 'nan_receipt', 'missing_objects'])
def test_evidence_faults_are_latched_unavailable_not_task_failure(task, fault):
    evaluator, _, _ = sequence(task, drive_steps=0, hold_steps=0)
    bad = next_observation(task, evaluator, turns=(0., 0., 0.))
    if fault == 'coverage':
        bad['contacts']['sampling']['sampled_substeps'] = 39
    elif fault == 'clock':
        bad['sim_time_s'] += .001
    elif fault == 'nan_pose':
        bad['objects']['input_gear']['position_m'][0] = float('nan')
    elif fault == 'nan_receipt':
        bad['contacts']['rigid'][-1]['impulse_magnitude_sum_ns'] = float('nan')
    else:
        del bad['objects']
    result = evaluator.update(bad)
    assert not result['success'] and result['progress'] == 0.
    assert result['referee'] == {
        'schema': 'physical_task_referee_state.v1', 'state': 'evaluator_invalid', 'phase': 'unavailable',
        'reason_codes': ['invalid_observation'], 'terminal': True, 'recoverable': False}
    assert evaluator.update(next_observation(task, evaluator, turns=(0., 0., 0.))) == result


@pytest.mark.parametrize('fault,code', [('linear_speed', 'object_speed_limit'),
    ('angular_speed', 'object_speed_limit'), ('penetration', 'penetration_limit')])
def test_valid_physical_limits_stop_separately_from_bad_evidence(task, fault, code):
    evaluator, _, _ = sequence(task, drive_steps=0, hold_steps=0)
    bad = next_observation(task, evaluator, turns=(0., 0., 0.))
    if fault == 'penetration':
        bad['contacts']['rigid'][0]['min_separation_m'] = -.0006
    else:
        key = 'linear_velocity_m_s' if fault == 'linear_speed' else 'angular_velocity_rad_s'
        bad['objects']['open_end_wrench'][key] = [.3 if fault == 'linear_speed' else 3., 0., 0.]
    result = evaluator.update(bad)
    assert not result['success'] and result['progress'] == 0.
    assert result['referee']['state'] == 'safety_stop'
    assert result['referee']['reason_codes'] == [code]
    assert result['referee']['terminal'] and not result['referee']['recoverable']
    assert evaluator.update({}) == result


def test_terminal_reset_requires_actual_original_unassembled_state(task):
    evaluator, _, _ = sequence(task)
    evaluator.reset()
    bad = observation(task, 0, task['evaluation']['target_bottom_centers_m'], [0., 0., 0.])
    assert evaluator.update(bad)['referee']['state'] == 'evaluator_invalid'
    evaluator.reset()
    initial = {n: task['evaluation']['initial_objects'][n]['position_m'] for n in GEARS}
    result = evaluator.update(observation(task, 0, initial, [0., 0., 0.]))
    assert result['referee']['state'] == 'ongoing' and not result['assembly_baseline_recorded']
    assert result['contact_backed_output_turn_rad'] == 0.


def test_bad_reset_identity_is_unavailable_before_speed_classification(task):
    evaluator = SequenceEvaluator(task)
    initial = {n: task['evaluation']['initial_objects'][n]['position_m'] for n in GEARS}
    obs = observation(task, 0, initial, [0., 0., 0.])
    obs['objects']['input_gear']['linear_velocity_m_s'] = [.3, 0., 0.]
    obs['objects']['straight_hex_screwdriver']['position_m'][0] += .01
    result = evaluator.update(obs)
    assert result['referee']['state'] == 'evaluator_invalid'
    assert 'Initial body pose differs' in result['reason']


@pytest.mark.parametrize('motion', ['native_linear', 'native_angular', 'observed_linear', 'slow_drift'])
def test_used_tool_motion_and_window_drift_block_release_but_are_recoverable(task, motion):
    evaluator, _, _ = sequence(task, hold_steps=0)
    tool = 'socket_crank'
    assert evaluator.used_tools == {tool}
    offset = 0.
    for i in range(1, 18):
        obs = next_observation(task, evaluator)
        if motion == 'native_linear':
            obs['objects'][tool]['linear_velocity_m_s'] = [.01, 0., 0.]
        elif motion == 'native_angular':
            obs['objects'][tool]['angular_velocity_rad_s'] = [0., 0., .1]
        else:
            offset = i*(.00006 if motion == 'observed_linear' else .000015)
            obs['objects'][tool]['position_m'][0] += offset
        result = evaluator.update(obs)
        assert not result['success'] and result['referee']['state'] == 'ongoing'
    assert result['release_stable_s'] < .3
    # A fresh full stable window at the real final pose can complete the task.
    for _ in range(17):
        obs = next_observation(task, evaluator)
        obs['objects'][tool]['position_m'][0] += offset
        result = evaluator.update(obs)
    assert result['success'], result


def test_unused_tool_does_not_add_a_new_stability_requirement(task):
    evaluator, _, _ = sequence(task, hold_steps=0)
    for _ in range(17):
        obs = next_observation(task, evaluator)
        obs['objects']['open_end_wrench']['linear_velocity_m_s'] = [.01, 0., 0.]
        result = evaluator.update(obs)
    assert result['success'] and result['used_tools'] == ['socket_crank']


def test_switching_tools_keeps_every_credited_tool_in_release_check(task):
    evaluator, _, _ = sequence(task, hold_steps=0)
    result = evaluator.update(next_observation(task, evaluator, turns=(30.2, -30.2, 30.2),
        drive=True, tool='open_end_wrench'))
    assert result['used_tools'] == ['open_end_wrench', 'socket_crank']
    for _ in range(17):
        obs = next_observation(task, evaluator, turns=(30.2, -30.2, 30.2))
        obs['objects']['socket_crank']['angular_velocity_rad_s'] = [0., 0., .1]
        result = evaluator.update(obs)
    assert not result['success']
    assert result['referee']['reason_codes'] == ['used_tool_stability_required']
    for _ in range(17):
        result = evaluator.update(next_observation(task, evaluator, turns=(30.2, -30.2, 30.2)))
    assert result['success']


def test_both_velocity_endpoints_bound_credited_drive_intervals(task):
    evaluator, _, _ = sequence(task, drive_steps=0, hold_steps=0)
    fast = next_observation(task, evaluator, turns=(.2, -.2, .2), drive=True)
    fast['objects']['input_gear']['angular_velocity_rad_s'] = [0., 0., 1.]
    result = evaluator.update(fast)
    assert result['contact_backed_output_turn_rad'] == 0.
    assert result['referee']['reason_codes'] == ['drive_speed_required']
    result = evaluator.update(next_observation(task, evaluator, turns=(.4, -.4, .4), drive=True))
    assert result['contact_backed_output_turn_rad'] == 0.  # Previous endpoint was too fast.
    result = evaluator.update(next_observation(task, evaluator, turns=(.6, -.6, .6), drive=True))
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(.2))


def test_endpoint_phase_correction_does_not_erase_a_bad_transmission_segment(task):
    evaluator, _, _ = sequence(task, output_ratio=.9, hold_steps=0)
    for output in np.linspace(27.2, 30., 15):
        evaluator.update(next_observation(task, evaluator, turns=(30., -30., float(output))))
    for _ in range(17):
        result = evaluator.update(next_observation(task, evaluator))
    assert result['maximum_pair_phase_error_rad'] == pytest.approx(math.radians(3.))
    assert not result['success'] and result['referee']['state'] == 'ongoing'
    assert result['referee']['phase'] == 'reassembly_required'
    assert result['referee']['reason_codes'] == ['phase_error_requires_reassembly']


@pytest.mark.parametrize('fresh_drive', [True, False])
def test_reassembly_resets_all_credit_and_requires_a_complete_new_drive(task, fresh_drive):
    evaluator, _, _ = sequence(task, output_ratio=.7, hold_steps=0)
    old_turns = np.array([30., -30., 21.])
    reset = evaluator.update(next_observation(task, evaluator, turns=old_turns, direct=True))
    assert not reset['assembly_baseline_recorded']
    assert reset['contact_backed_output_turn_rad'] == reset['maximum_pair_phase_error_rad'] == 0.
    assert reset['tool_interaction_s'] == reset['release_stable_s'] == 0.
    assert set(reset['signed_gear_turns_rad'].values()) == {0.} and reset['used_tools'] == []
    for _ in range(17):
        result = evaluator.update(next_observation(task, evaluator, turns=old_turns))
    assert result['assembly_baseline_recorded'] and result['contact_backed_output_turn_rad'] == 0.
    if fresh_drive:
        for i in range(1, 151):
            turns = old_turns+np.array([1., -1., 1.])*i*.2
            result = evaluator.update(next_observation(task, evaluator, turns=turns, drive=True))
    else:
        turns = old_turns
    for _ in range(17):
        result = evaluator.update(next_observation(task, evaluator, turns=turns))
    assert result['success'] is fresh_drive
    assert result['contact_backed_output_turn_rad'] == pytest.approx(math.radians(30.) if fresh_drive else 0.)
