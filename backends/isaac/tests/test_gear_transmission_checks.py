"""Native-like counterexamples for transmission evidence, not a simulator."""
import math
import json

import pytest

from scripts.validation.gear_transmission_checks import CONTRACT, GEARS, assess_transmission, positive_contact, compact_native_row


DT = .01
PATHS = {name: '/World/Objects/'+name for name in (*GEARS, 'socket_crank')}
FIXTURE = '/World/Fixtures/gear_workplate'
INFO = {'objects': {name: {'path': path} for name, path in PATHS.items()},
        'fixtures': {'gear_workplate': {'path': FIXTURE}}}


def pose(x, y, z, yaw):
    return [x, y, z, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]


def contact(a, b, normal=(1., 0., 0.)):
    return {'actor0': a, 'actor1': b, 'normal_impulses_ns': [.001], 'normals': [list(normal)]}


@pytest.fixture
def evidence():
    rows, theta, direction = [], 0., 1
    for phase in CONTRACT['phases']:
        if phase['id'] == 'reverse':
            direction = -1
        for _ in range(round(phase['duration_s']/DT)):
            if phase['loaded']:
                theta += direction*math.radians(10.)*DT
            angles = (theta, math.pi/32-(20/32)*theta+direction*.005, theta)
            positions = {name: pose(.4, y, .802, angle) for name, y, angle in zip(GEARS, (-.13, 0., .13), angles)}
            positions['socket_crank'] = pose(.4, -.1, .820, theta)
            contacts = [contact(PATHS[a], PATHS[b]) for a, b in (
                ('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))]
            contacts += [contact(PATHS[name], FIXTURE+'/collision/thrust', (0., 0., 1.)) for name in GEARS]
            index = len(rows)+1
            rows.append({'native_step': index, 'time_s': index*DT,
                'poses_xyzw': positions, 'velocities': {name: [0.]*6 for name in PATHS},
                'raw_contacts': contacts, 'robot_interference': [], 'maximum_penetration_m': 0.,
                'wrench': {'native_step': index, 'body': 'socket_crank', 'phase': phase['id'],
                           'world_frame': True,
                           'force_n': [0., 0., 0.], 'torque_nm': [0., 0., direction*.4 if phase['loaded'] else 0.]}})
    return rows


def test_contiguous_two_direction_contact_evidence_passes(evidence):
    result = assess_transmission(evidence, INFO, DT)
    assert result['passed'], result
    assert not result['task_success_claimed']
    assert result['directions']['forward']['ratio_slopes']['idler_gear'] == pytest.approx(-.625)
    assert result['directions']['reverse']['ratio_slopes']['output_gear'] == pytest.approx(1.)
    assert result['backlash']['input_gear__idler_gear']['measured_reversal_rad'] == pytest.approx(.016)
    assert result['backlash']['idler_gear__output_gear']['measured_reversal_rad'] == pytest.approx(.010)


@pytest.mark.parametrize('tool_speed', [0., 2.1])
def test_report_json_roundtrip_preserves_native_derived_speed_gate(evidence, tool_speed):
    evidence[300]['velocities']['socket_crank'][5] = tool_speed
    result = assess_transmission(evidence, INFO, DT)
    assert result['checks']['tool_speed_bound'] is (tool_speed <= 2.)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result['passed'] is (tool_speed <= 2.)


@pytest.mark.parametrize('fault', ['missing_sample', 'duplicate_sample', 'missing_tool_receipt', 'gear_loaded',
    'over_torque', 'output_stationary', 'no_normal_load', 'wrong_support_direction', 'lost_mesh_window',
    'too_eccentric', 'tilted', 'no_reverse_backlash', 'robot_contact', 'missing_pose', 'settle_body_explosion'])
def test_missing_or_incorrect_physical_evidence_is_rejected(evidence, fault):
    if fault == 'missing_sample':
        evidence.pop(300)
    elif fault == 'duplicate_sample':
        evidence[301]['native_step'] = evidence[300]['native_step']
    elif fault == 'missing_tool_receipt':
        evidence[300].pop('wrench')
    elif fault == 'gear_loaded':
        evidence[300]['wrench']['body'] = 'input_gear'
    elif fault == 'over_torque':
        evidence[300]['wrench']['torque_nm'][2] = .81
    elif fault == 'output_stationary':
        for row in evidence:
            row['poses_xyzw']['output_gear'][3:] = [0., 0., 0., 1.]
    elif fault == 'no_normal_load':
        for row in evidence:
            for c in row['raw_contacts']:
                c['normal_impulses_ns'] = [0.]
    elif fault == 'wrong_support_direction':
        for row in evidence:
            for c in row['raw_contacts']:
                if c['actor1'].startswith(FIXTURE+'/'):
                    c['normals'] = [[0., 0., -1.]]
    elif fault == 'lost_mesh_window':
        for row in evidence[300:305]:
            row['raw_contacts'] = [c for c in row['raw_contacts'] if c['actor0'] != PATHS['idler_gear']]
    elif fault == 'too_eccentric':
        evidence[300]['poses_xyzw']['idler_gear'][0] += .0006
    elif fault == 'tilted':
        evidence[300]['poses_xyzw']['idler_gear'][3:] = [math.sin(.01), 0., 0., math.cos(.01)]
    elif fault == 'no_reverse_backlash':
        for row in evidence:
            if row['wrench']['phase'] not in ('reverse', 'final_hold'):
                continue
            p = row['poses_xyzw']['input_gear']
            theta = 2*math.atan2(p[5], p[6])
            row['poses_xyzw']['idler_gear'] = pose(.4, 0., .802, math.pi/32-(20/32)*theta+.005)
            row['poses_xyzw']['output_gear'] = pose(.4, .13, .802, theta)
    elif fault == 'robot_contact':
        evidence[300]['robot_interference'] = [{'actor0': '/Robot/hand', 'actor1': PATHS['input_gear']}]
    elif fault == 'missing_pose':
        evidence[300]['poses_xyzw'].pop('idler_gear')
    elif fault == 'settle_body_explosion':
        evidence[0]['velocities']['input_gear'][1] = 2300.
    assert not assess_transmission(evidence, INFO, DT)['passed'], fault


def test_absent_native_stream_fails_closed():
    assert not assess_transmission([], INFO, DT)['passed']


@pytest.mark.parametrize('swapped,scalar,normal,expected', [
    (False, -.001, (0., 0., -1.), True),
    (True, -.001, (0., 0., 1.), True),
    (False, -.001, (0., 0., 1.), False),
    (True, .001, (0., 0., 1.), False),
])
def test_upward_support_uses_signed_normal_vector_and_actor_order(swapped, scalar, normal, expected):
    a, b = (FIXTURE, PATHS['input_gear']) if swapped else (PATHS['input_gear'], FIXTURE)
    c = contact(a, b, normal)
    c['normal_impulses_ns'] = [scalar]
    assert positive_contact({'raw_contacts': [c]}, PATHS['input_gear'], FIXTURE, upward=True) is expected


def test_mixed_contacts_require_net_upward_support_without_pointwise_clamping():
    c = contact(PATHS['input_gear'], FIXTURE)
    c.update(normal_impulses_ns=[.001, -.002], normals=[[0., 0., 1.], [0., 0., 1.]])
    assert not positive_contact({'raw_contacts': [c]}, PATHS['input_gear'], FIXTURE, upward=True)
    c['normal_impulses_ns'] = [.002, -.001]
    assert positive_contact({'raw_contacts': [c]}, PATHS['input_gear'], FIXTURE, upward=True)


def test_negative_scalar_is_still_nondirectional_contact_pressure():
    c = contact(PATHS['input_gear'], PATHS['idler_gear'])
    c['normal_impulses_ns'] = [-.001]
    assert positive_contact({'raw_contacts': [c]}, PATHS['input_gear'], PATHS['idler_gear'])


def test_streaming_signed_reduction_matches_full_native_assessment(evidence, tmp_path):
    from scripts.validation.reassess_gear_transmission import scan_native_frames
    for row in evidence:
        for c in row['raw_contacts']:
            c['normal_impulses_ns'] = [-v for v in c['normal_impulses_ns']]
            c['normals'] = [[-v for v in n] for n in c['normals']]
    frames, receipts = tmp_path/'native.jsonl', tmp_path/'wrenches.jsonl'
    with frames.open('w') as f, receipts.open('w') as w:
        for row in evidence:
            receipt = {**row['wrench'], 'time_s': row['time_s']}
            w.write(json.dumps(receipt)+'\n')
            f.write(json.dumps({k: v for k, v in row.items() if k != 'wrench'})+'\n')
    compact, identities = scan_native_frames(frames, wrench_path=receipts)
    assert identities['native']['rows'] == len(evidence)
    assert all('raw_contacts' not in row for row in compact)
    assert assess_transmission(compact, INFO, DT) == assess_transmission(evidence, INFO, DT)
    receipts.write_text(receipts.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='different lengths'):
        scan_native_frames(frames, wrench_path=receipts)


def test_derived_contact_totals_preserve_net_downward_reaction(evidence):
    row = evidence[0]
    row['raw_contacts'][3].update(normal_impulses_ns=[.001, -.002], normals=[[0., 0., 1.]]*2)
    compact = compact_native_row(row)
    assert not positive_contact(compact, PATHS['input_gear'], FIXTURE, upward=True, second_is_prefix=True)
    compact['raw_contacts'] = []
    with pytest.raises(ValueError, match='Ambiguous'):
        positive_contact(compact, PATHS['input_gear'], FIXTURE)


@pytest.mark.parametrize('phase', ['settle', 'coast', 'final_hold'])
def test_declared_passive_phases_cannot_contain_an_applied_tool_torque(evidence, phase):
    for row in evidence:
        if row['wrench']['phase'] == phase:
            row['wrench']['torque_nm'] = [0., 0., .4]
    result = assess_transmission(evidence, INFO, DT)
    assert not result['passed'] and not result['checks']['bounded_tool_only_load_receipts']


def test_final_partial_native_contact_window_is_not_dropped():
    from scripts.validation.gear_transmission_checks import _block_fractions
    assert _block_fractions([True]*40+[False]*39, 40) == [1., 0.]


def test_derived_zero_magnitude_cannot_hide_a_small_support_vector(evidence):
    row = compact_native_row(evidence[0])
    c = row['contact_totals'][3]
    c.update(absolute_scalar_sum_ns=0., normal_vector_magnitude_sum_ns=0.,
             signed_actor0_impulse_vector_ns=[0., 0., 9.e-9])
    with pytest.raises(ValueError, match='Invalid derived'):
        positive_contact(row, PATHS['input_gear'], FIXTURE, upward=True, second_is_prefix=True)


def test_vector_magnitude_sum_accounts_for_native_normal_roundoff(evidence):
    c = evidence[0]['raw_contacts'][3]
    c.update(normal_impulses_ns=[-.001], normals=[[0., 0., -1.-1.e-7]])
    row = compact_native_row(evidence[0])
    assert row['contact_totals'][3]['normal_vector_magnitude_sum_ns'] > row['contact_totals'][3]['absolute_scalar_sum_ns']
    assert positive_contact(row, PATHS['input_gear'], FIXTURE, upward=True, second_is_prefix=True)


def test_all_six_supply_objects_use_design_axis_and_directed_rest_contact():
    from scripts.validation.validate_gear_transmission import make_probe_inputs, assess_supply
    inputs = make_probe_inputs()
    scene = inputs['scene']
    info = {'objects': {s['id']: {'path': '/World/Objects/'+s['id']} for s in scene['objects']},
            'fixtures': {s['id']: {'path': '/World/Fixtures/'+s['id']} for s in scene['fixtures']},
            'workbench': {'supports': {'table': '/World/Table'}}}
    positions, contacts = {}, []
    for spec in scene['objects']:
        name, q = spec['id'], spec['quaternion_wxyz']
        positions[name] = [*spec['position_m'], *q[1:], q[0]]
        rest = name+'_supply_rest' if name in GEARS else {
            'socket_crank': 'socket_crank_rest', 'open_end_wrench': 'open_end_wrench_rest',
            'straight_hex_screwdriver': 'straight_screwdriver_rest'}[name]
        support = info['fixtures'][rest]['path']+'/collider' if rest in info['fixtures'] else '/World/Table/collider'
        contacts.append(contact(info['objects'][name]['path'], support, (0., 0., 1.)))
    rows = [{'native_step': i+1, 'time_s': (i+1)*DT, 'poses_xyzw': positions,
             'velocities': {name: [0.]*6 for name in positions}, 'raw_contacts': contacts,
             'robot_interference': [], 'maximum_penetration_m': 0.} for i in range(150)]
    result = assess_supply(rows, scene, info, DT)
    assert result['passed'] and len(result['objects']) == 6
    assert result['objects']['straight_hex_screwdriver']['maximum_design_axis_error_rad'] < 1.e-6
    contacts[-1]['normals'][0][2] = -1.
    assert not assess_supply(rows, scene, info, DT)['passed']
