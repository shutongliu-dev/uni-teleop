"""Counterexamples for the pinned new-layout staged gear diagnostic."""
from copy import deepcopy
import math

import pytest

from bench.paths import DATA_ROOT
from scripts.validation import gear_scene_contract as probe
from scripts.validation.gear_transmission_checks import compact_native_row, probe_contract


@pytest.fixture(scope='module')
def frozen():
    return probe.build_inputs()


def test_builder_only_uses_new_release_without_dropping_or_moving_inactive_bodies(frozen):
    reference = probe.definition.TASK['design']['supply_layout_candidate']['release']
    assert frozen['release'] == reference['path']
    assert frozen['release_manifest_sha256'] == reference['manifest_sha256']
    assert frozen['scene']['isaacsim']['solver_type'] == 'PGS'
    assert [c['id'] for c in frozen['cases']] == ['supply', 'input_only', 'input_idler', 'full_train']
    assert [c['duration_s'] for c in frozen['cases']] == [1.5, 11.5, 11.5, 11.5]
    supply = frozen['task']['evaluation']['initial_objects']
    mounted = frozen['task']['dynamics_probe']['object_overrides']
    for case in frozen['cases']:
        assert set(case['reset_objects']) == set(probe.OBJECTS)
        for name in probe.OBJECTS:
            selected = name in case['active_gears'] or name == 'socket_crank' and case['mounted_tool']
            assert case['reset_objects'][name] == (mounted[name] if selected else supply[name])
    for key in ('task_success_claimed', 'physics_accepted', 'object_state_writeback_after_reset'):
        assert frozen[key] is False
    assert frozen['model_calls'] == 0


def test_new_load_uses_frozen_task_speed_and_original_nominal_torque(frozen):
    old = probe_contract()
    new = frozen['probe_contract']
    assert new['maximum_torque_nm'] == old['maximum_torque_nm'] == .8
    assert new['speed_load_coefficient_nm_s_per_rad'] == old['speed_load_coefficient_nm_s_per_rad'] == 2.
    assert new['passive_wrench_api_calls'] is False
    for actual, original in zip(new['phases'], old['phases']):
        assert {k: v for k, v in actual.items() if k != 'target_speed_rad_s'} == {k: v for k, v in original.items() if k != 'target_speed_rad_s'}
        if actual['loaded']:
            assert abs(actual['target_speed_rad_s']) == frozen['task']['evaluation']['maximum_drive_angular_speed_rad_s'] == math.radians(15.)
    for key in ('maximum_penetration_m', 'ratio_slope_tolerance', 'maximum_pair_phase_range_rad',
                'minimum_measured_pair_backlash_rad', 'backlash_comparison_tolerance_rad'):
        assert new[key] == old[key]


def test_legacy_release_is_rejected_before_loading():
    legacy = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/gear/large_idler_v3_4_compact_pgs_release'
    with pytest.raises(ValueError, match='Only the pinned new'):
        probe.build_inputs(legacy)


def _evidence(frozen, case_id, *, dt=.01):
    # This synthetic integration grid keeps CPU examples small. build_inputs()
    # above independently checks the real frozen 0.5 ms configuration.
    inputs = deepcopy(frozen)
    inputs['task']['evaluation']['physics_dt_s'] = dt
    case = next(c for c in inputs['cases'] if c['id'] == case_id)
    paths = {n: '/World/Objects/'+n for n in probe.OBJECTS}
    fixture, table = '/World/Fixtures/gear_workplate', '/World/Room/Table'
    info = {'objects': {n: {'path': p} for n, p in paths.items()},
        'fixtures': {'gear_workplate': {'path': fixture}}, 'workbench': {'supports': {'table': table}}}
    phases = case['phases'] or [{'id': 'supply', 'duration_s': 1.5, 'loaded': False}]
    theta, direction, rows = 0., 1., []
    for phase in phases:
        if phase['id'] == 'reverse':
            direction = -1.
        for _ in range(round(phase['duration_s']/dt)):
            if phase['loaded']:
                theta += direction*math.radians(10.)*dt
            poses, velocities = {}, {}
            for name, spec in case['reset_objects'].items():
                p, q = spec['position_m'], spec['quaternion_wxyz']
                if name in case['active_gears'] or case['mounted_tool'] and name == 'socket_crank':
                    index = probe.GEARS.index(name) if name in probe.GEARS else 0
                    yaw = math.pi/32-(20/32)*theta+direction*.005 if name == 'idler_gear' else theta
                    p = [.4, [-.13, 0., .13][index], .820 if name == 'socket_crank' else .802]
                    q = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                poses[name] = [*p, *q[1:], q[0]]
                velocities[name] = [0.]*6
            contacts = []
            for name in probe.OBJECTS:
                if name in case['active_gears']:
                    support = fixture+'/thrust'
                elif name == 'socket_crank' and case['mounted_tool']:
                    continue
                else:
                    support = table+'/collision'
                contacts.append({'actor0': paths[name], 'actor1': support,
                    'normal_impulses_ns': [.001], 'normals': [[0., 0., 1.]]})
            if case['mounted_tool']:
                for a, b in [('socket_crank', 'input_gear'), *zip(case['active_gears'], case['active_gears'][1:])]:
                    contacts.append({'actor0': paths[a], 'actor1': paths[b],
                        'normal_impulses_ns': [.001], 'normals': [[1., 0., 0.]]})
            index = len(rows)+1
            rows.append({'native_step': index, 'time_s': index*dt, 'poses_xyzw': poses,
                'velocities': velocities, 'raw_contacts': contacts, 'robot_interference': [],
                'maximum_penetration_m': 0., 'wrench': {'native_step': index, 'time_s': index*dt,
                    'phase': phase['id'], 'body': 'socket_crank', 'world_frame': True,
                    'force_n': [0., 0., 0.], 'torque_nm': [0., 0., direction*.4 if phase['loaded'] else 0.]}})
    return rows, info, dt, case, inputs


@pytest.mark.parametrize('stage', ['supply', 'input_only', 'input_idler'])
def test_complete_staged_contact_evidence_passes_without_task_credit(frozen, stage):
    rows, info, dt, case, inputs = _evidence(frozen, stage)
    result = probe.assess_stage(rows, info, dt, case, inputs)
    assert result['passed'], result
    assert not result['task_success_claimed'] and not result['physics_accepted']
    compact = [compact_native_row(r) for r in rows]
    assert probe.assess_stage(compact, info, dt, case, inputs) == result


@pytest.mark.parametrize('fault', ['missing_substep', 'wrong_dt', 'gear_loaded', 'over_torque', 'wrong_wrench_time',
    'missing_tool_contact', 'idler_stationary', 'missing_pair_contact', 'wrong_support',
    'penetration', 'body_speed', 'passive_drift', 'passive_robot_touch', 'inactive_displaced'])
def test_partial_stage_rejects_missing_or_incorrect_evidence(frozen, fault):
    rows, info, dt, case, inputs = _evidence(frozen, 'input_idler')
    if fault == 'missing_substep':
        rows.pop(300)
    elif fault == 'wrong_dt':
        dt = .02
    elif fault == 'gear_loaded':
        rows[300]['wrench']['body'] = 'input_gear'
    elif fault == 'over_torque':
        rows[300]['wrench']['torque_nm'][2] = .81
    elif fault == 'wrong_wrench_time':
        rows[300]['wrench']['time_s'] += .001
    elif fault in ('missing_tool_contact', 'missing_pair_contact'):
        actor = info['objects']['socket_crank' if fault == 'missing_tool_contact' else 'input_gear']['path']
        for row in rows:
            row['raw_contacts'] = [c for c in row['raw_contacts'] if not (c['actor0'] == actor and c['actor1'] in (info['objects']['input_gear']['path'], info['objects']['idler_gear']['path']))]
    elif fault == 'idler_stationary':
        for row in rows:
            row['poses_xyzw']['idler_gear'][3:] = [0., 0., 0., 1.]
    elif fault == 'wrong_support':
        for row in rows:
            for c in row['raw_contacts']:
                if c['actor1'].endswith('/thrust'):
                    c['normals'] = [[0., 0., -1.]]
    elif fault == 'penetration':
        rows[300]['maximum_penetration_m'] = .000501
    elif fault == 'body_speed':
        rows[300]['velocities']['input_gear'][0] = .201
    elif fault == 'passive_drift':
        for row in rows[-20:]:
            row['poses_xyzw']['socket_crank'][0] += .00011
    elif fault == 'passive_robot_touch':
        rows[-10]['raw_contacts'].append({'actor0': info['objects']['socket_crank']['path'],
            'actor1': '/World/Robot/right_index_MP', 'normal_impulses_ns': [0.], 'normals': [[1., 0., 0.]]})
    elif fault == 'inactive_displaced':
        for row in rows:
            row['poses_xyzw']['output_gear'][0] += .004
    result = probe.assess_stage(rows, info, dt, case, inputs)
    assert not result['passed'], fault


def test_unknown_stage_cannot_pass(frozen):
    rows, info, dt, case, inputs = _evidence(frozen, 'supply')
    case = {**case, 'id': 'unknown'}
    assert not probe.assess_stage(rows, info, dt, case, inputs)['passed']


def test_full_train_retains_existing_transmission_and_adds_strict_used_tool_rest(frozen):
    rows, info, dt, case, inputs = _evidence(frozen, 'full_train', dt=.0005)
    result = probe.assess_stage(rows, info, dt, case, inputs)
    assert result['passed'], result
    assert result['transmission']['directions']['forward']['ratio_slopes']['idler_gear'] == pytest.approx(-.625)
    assert result['transmission']['directions']['reverse']['ratio_slopes']['output_gear'] == pytest.approx(1.)
    assert result['canonical_passive_windows']['final_hold']['active_and_used_tool_rest']['objects']['socket_crank']['passed']
    # The old full-train passive helper only applied global speed bounds to
    # tools. A small terminal drift still fails the current used-tool gate.
    for row in rows[-400:]:
        row['poses_xyzw']['socket_crank'][0] += .00011
    result = probe.assess_stage(rows, info, dt, case, inputs)
    assert not result['passed']
    assert not result['canonical_passive_windows']['final_hold']['active_and_used_tool_rest']['objects']['socket_crank']['checks']['anchor_position_range']
