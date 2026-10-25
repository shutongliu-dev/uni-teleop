"""Frozen new-layout L5 isolation probes; passive gears and zero task credit."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from bench.isaacsim.release import digest, task_inputs
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.native_evidence import angle, rotation, within
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly import definition
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import GEARS, TOOLS
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import contact_probe_reset
from scripts.validation.gear_passive_checks import canonical_passive_window
from scripts.validation.gear_transmission_checks import (
    _block_fractions, _pair_backlash_rad, assess_transmission, axial_rotation,
    contact_totals, positive_contact, probe_contract, transmission_geometry_contract,
)
from scripts.validation.validate_gear_transmission import assess_supply


SCHEMA = 'mechanical_new_layout_gear_scene_probe.v1'
OBJECTS = GEARS + TOOLS


def _case_contract(task):
    contract = probe_contract()
    speed = float(task['evaluation']['maximum_drive_angular_speed_rad_s'])
    if not math.isfinite(speed) or speed <= 0.:
        raise ValueError('The frozen task must declare a positive drive speed')
    for phase in contract['phases']:
        if phase['loaded']:
            phase['target_speed_rad_s'] = math.copysign(speed, phase['target_speed_rad_s'])
    contract['engineering_variant'] = 'new_layout_task_speed_isolation_v1'
    contract['scope'] = ('New-layout release only; bounded torque on the independent socket_crank, '
        'with its speed target taken from the frozen task. All gears passive. '
        'Reset-only stage isolation; no robot strategy, assembly or task credit.')
    contract['passive_wrench_api_calls'] = False
    return contract


def build_inputs(release=None):
    """Bind only the current inset-layout release, never a legacy candidate."""
    reference = definition.TASK['design']['supply_layout_candidate']['release']
    expected = Path(reference['path']).resolve()
    release = expected if release is None else Path(release).resolve()
    if release != expected:
        raise ValueError('Only the pinned new supply-layout L5 release is permitted')
    manifest_path = release/'release_manifest.json'
    if digest(manifest_path) != reference['manifest_sha256']:
        raise ValueError('Pinned new-layout release manifest changed')
    _, scene, task, initial = task_inputs(release, 'gear_train_assembly')
    manifest = load_json(manifest_path)
    if (scene['isaacsim'].get('solver_type') != 'PGS'
            or scene['isaacsim']['physics_dt_s'] != task['evaluation']['physics_dt_s']
            or [s['id'] for s in scene['objects']] != list(OBJECTS)
            or task['supply_layout'].get('revision') != 'table_inset_v1'
            or task['supply_layout'].get('display_pads_or_supply_rests') is not False):
        raise ValueError('The frozen new-layout physics/reset inventory is inconsistent')
    reset = deepcopy(task['evaluation']['initial_objects'])
    if set(reset) != set(OBJECTS):
        raise ValueError('Every original free body must remain in every case')
    mounted = task['dynamics_probe']['object_overrides']
    if mounted != contact_probe_reset()['object_overrides']:
        raise ValueError('Frozen diagnostic assembly poses differ from the new geometry')
    contract = _case_contract(task)
    cases = []
    for name, active in (('supply', ()), ('input_only', GEARS[:1]),
                         ('input_idler', GEARS[:2]), ('full_train', GEARS)):
        objects = deepcopy(reset)
        if active:
            for body in (*active, 'socket_crank'):
                objects[body] = deepcopy(mounted[body])
        cases.append({'id': name, 'active_gears': list(active), 'mounted_tool': bool(active),
            'duration_s': sum(p['duration_s'] for p in contract['phases']) if active else 1.5,
            'phases': deepcopy(contract['phases']) if active else [],
            'reset_objects': objects,
            'scope': 'Reset-only engineering scene evidence; no object writeback after reset or task credit.'})
    return {'schema': SCHEMA, 'scene': scene, 'task': task, 'initial_state': initial,
        'release': str(release), 'release_manifest_sha256': reference['manifest_sha256'],
        'asset_files_sha256': deepcopy(manifest['asset_files_sha256']),
        'probe_contract': contract, 'cases': cases, 'model_calls': 0,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False,
        'transmission_geometry_contract': transmission_geometry_contract(geometry=task['evaluation'])}


def _motion(rows, dt, duration, evaluation):
    """Native velocities plus every available rolling 20 ms pose interval."""
    count, stride = round(duration/dt), round(evaluation['control_period_s']/dt)
    checks = {'complete_native_sampling': bool(len(rows) == count and stride > 0
        and abs(stride*dt-evaluation['control_period_s']) < 1.e-9
        and all(type(r['native_step']) is int and abs(r['time_s']-(i+1)*dt) < 1.e-8
            and (not i or r['native_step'] == rows[i-1]['native_step']+1)
            for i, r in enumerate(rows)))}
    bodies = {}
    for name in OBJECTS:
        poses = np.asarray([r['poses_xyzw'][name] for r in rows], float)
        velocities = np.asarray([r['velocities'][name] for r in rows], float)
        if (poses.shape != (count, 7) or velocities.shape != (count, 6)
                or not np.isfinite(poses).all() or not np.isfinite(velocities).all()
                or np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1.)) > 1.e-3):
            raise ValueError('Complete finite six-body motion evidence is required')
        q = poses[:, 3:]/np.linalg.norm(poses[:, 3:], axis=1)[:, None]
        qa, qb = q[:-stride], q[stride:]
        qb = qb*np.where(np.sum(qa*qb, axis=1) < 0., -1., 1.)[:, None]
        observed_angle = 4*np.arctan2(np.linalg.norm(qa-qb, axis=1), np.linalg.norm(qa+qb, axis=1))
        metrics = {'maximum_native_linear_speed_m_s': float(np.linalg.norm(velocities[:, :3], axis=1).max()),
            'maximum_native_angular_speed_rad_s': float(np.linalg.norm(velocities[:, 3:], axis=1).max()),
            'maximum_rolling_20ms_linear_speed_m_s': float(np.linalg.norm(poses[stride:, :3]-poses[:-stride, :3], axis=1).max()/evaluation['control_period_s']),
            'maximum_rolling_20ms_angular_speed_rad_s': float(observed_angle.max()/evaluation['control_period_s'])}
        bodies[name] = {'passed': max(metrics['maximum_native_linear_speed_m_s'], metrics['maximum_rolling_20ms_linear_speed_m_s']) <= evaluation['maximum_object_linear_speed_m_s']
            and max(metrics['maximum_native_angular_speed_rad_s'], metrics['maximum_rolling_20ms_angular_speed_rad_s']) <= evaluation['maximum_object_angular_speed_rad_s'], **metrics}
    checks['all_original_motion_bounds'] = all(v['passed'] for v in bodies.values())
    return {'passed': all(checks.values()), 'checks': checks, 'objects': bodies,
        'rolling_20ms_intervals': max(0, count-stride),
        'first_interval_scope': 'Starts at the first recorded native pose; no authored reset pose substitutes for missing native t=0.'}


def _partial_passive(rows, info, dt, case, evaluation, end, phase):
    duration = evaluation['assembly_settle_s' if phase == 'settle' else 'release_hold_s']
    selected = [r for r in rows if end-duration-1.e-9 <= r['time_s'] <= end+1.e-9]
    count, stride = round(duration/dt), round(evaluation['control_period_s']/dt)
    if (len(selected) != count+1 or abs(selected[0]['time_s']-(end-duration)) > 1.e-8
            or abs(selected[-1]['time_s']-end) > 1.e-8):
        raise ValueError('A complete passive window and its native pose baseline are required')
    checks = {'zero_external_wrench': all(r['wrench']['phase'] == phase
        and r['wrench']['force_n'] == [0., 0., 0.] and r['wrench']['torque_nm'] == [0., 0., 0.]
        for r in selected)}
    names = list(case['active_gears']) + (['socket_crank'] if phase == 'final_hold' else [])
    released = {info['objects'][n]['path'] for n in names}
    checks['released_without_robot_manifolds'] = not any(
        a in released and within(b, '/World/Robot') or b in released and within(a, '/World/Robot')
        for r in selected[1:] for a, b, *_ in contact_totals(r))
    objects = {}
    fixture = info['fixtures']['gear_workplate']['path']
    centers = evaluation['shaft_centers_world_xy_m']
    for name in names:
        poses = np.asarray([r['poses_xyzw'][name] for r in selected], float)
        velocities = np.asarray([r['velocities'][name] for r in selected], float)
        rotations = [rotation([float(p[6]), *p[3:6].tolist()]) for p in poses]
        indices = list(range(0, count+1, stride))
        observed_linear = float(np.linalg.norm(np.diff(poses[indices, :3], axis=0), axis=1).max()/evaluation['control_period_s'])
        observed_angular = max(angle(rotations[a], rotations[b])/evaluation['control_period_s'] for a, b in zip(indices, indices[1:]))
        linear = float(np.linalg.norm(velocities[:, :3], axis=1).max())
        angular = float(np.linalg.norm(velocities[:, 3:], axis=1).max())
        position_range = float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max())
        angle_range = max(angle(rotations[0], r) for r in rotations)
        body_checks = {'native_linear_rest': linear <= evaluation['maximum_stable_linear_speed_m_s'],
            'native_angular_rest': angular <= evaluation['maximum_stable_angular_speed_rad_s'],
            'observed_linear_rest': observed_linear <= evaluation['maximum_stable_linear_speed_m_s'],
            'observed_angular_rest': observed_angular <= evaluation['maximum_stable_observed_angular_speed_rad_s'],
            'anchor_position_range': position_range <= evaluation['maximum_stable_position_range_m'],
            'anchor_angle_range': angle_range <= evaluation['maximum_stable_angle_range_rad']}
        if name in GEARS:
            xy = np.asarray(centers[GEARS.index(name)], float)
            support = [positive_contact(r, info['objects'][name]['path'], fixture, upward=True, second_is_prefix=True) for r in selected[1:]]
            body_checks['seated'] = bool(np.linalg.norm(poses[:, :2]-xy, axis=1).max() <= evaluation['maximum_axis_offset_m']
                and np.abs(poses[:, 2]-evaluation['seated_gear_bottom_z_m']).max() <= evaluation['maximum_bottom_height_error_m']
                and max(math.acos(float(np.clip(r[2, 2], -1., 1.))) for r in rotations) <= evaluation['maximum_axis_tilt_rad'])
            body_checks['upward_support_each_20ms'] = all(f > 0. for f in _block_fractions(support, stride))
        objects[name] = {'passed': all(body_checks.values()), 'checks': body_checks,
            'maximum_native_linear_speed_m_s': linear, 'maximum_native_angular_speed_rad_s': angular,
            'maximum_20ms_linear_speed_m_s': observed_linear, 'maximum_20ms_angular_speed_rad_s': observed_angular,
            'anchor_position_range_m': position_range, 'anchor_angle_range_rad': angle_range}
    checks['all_active_passive_conditions'] = bool(objects) and all(o['passed'] for o in objects.values())
    return {'passed': all(checks.values()), 'checks': checks, 'objects': objects,
        'phase': phase, 'start_time_s': end-duration, 'end_time_s': end, 'task_success_claimed': False}


def _partial_transmission(rows, info, dt, case, inputs):
    c, evaluation = inputs['probe_contract'], inputs['task']['evaluation']
    geometry = inputs['transmission_geometry_contract']
    active = case['active_gears']
    phases = [p['id'] for p in c['phases'] for _ in range(round(p['duration_s']/dt))]
    definitions = {p['id']: p for p in c['phases']}
    paths = {n: info['objects'][n]['path'] for n in (*active, 'socket_crank')}
    turns = {n: np.unwrap([axial_rotation(r['poses_xyzw'][n]) for r in rows]) for n in paths}
    block = round(evaluation['control_period_s']/dt)
    checks = {'bounded_tool_only_load_receipts': len(rows) == len(phases) and all(
        r['wrench']['native_step'] == r['native_step'] and r['wrench']['body'] == 'socket_crank'
        and r['wrench']['phase'] == phase and r['wrench']['world_frame'] is True
        and r['wrench']['force_n'] == [0., 0., 0.] and len(r['wrench']['torque_nm']) == 3
        and np.isfinite(r['wrench']['torque_nm']).all() and r['wrench']['torque_nm'][:2] == [0., 0.]
        and abs(r['wrench']['torque_nm'][2]) <= c['maximum_torque_nm']+1.e-8
        and (definitions[phase]['loaded'] or r['wrench']['torque_nm'] == [0., 0., 0.])
        for r, phase in zip(rows, phases))}
    centers, fixture = geometry['shaft_centers_world_xy_m'], info['fixtures']['gear_workplate']['path']
    use = np.asarray([p != 'settle' for p in phases])
    for name in active:
        poses = np.asarray([r['poses_xyzw'][name] for r in rows], float)[use]
        tilts = [math.acos(float(np.clip(rotation([float(p[6]), *p[3:6].tolist()])[2, 2], -1., 1.))) for p in poses]
        checks[name+'_supported_geometry'] = bool(np.linalg.norm(poses[:, :2]-centers[GEARS.index(name)], axis=1).max() <= c['maximum_gear_axis_offset_m']
            and np.abs(poses[:, 2]-c['seated_bottom_z_m']).max() <= c['maximum_gear_bottom_error_m']
            and max(tilts) <= c['maximum_gear_tilt_rad'])
        support = [positive_contact(r, paths[name], fixture, upward=True, second_is_prefix=True) for r, selected in zip(rows, use) if selected]
        checks[name+'_upward_support_each_20ms'] = all(f > 0. for f in _block_fractions(support, block))
    pairs = list(zip(active, active[1:]))
    pair_states = {}
    for a, b in pairs:
        relative = np.asarray([r['poses_xyzw'][b][:2] for r in rows])-np.asarray([r['poses_xyzw'][a][:2] for r in rows])
        line = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
        ratio = geometry['gear_teeth'][b]/geometry['gear_teeth'][a]
        pair_states[a+'__'+b] = turns[a]+ratio*turns[b]-(1.+ratio)*line
    directions = {}
    for phase, sign in (('forward', 1), ('reverse', -1)):
        indices = np.flatnonzero(np.asarray(phases) == phase)
        movement = sign*(turns['input_gear'][indices]-turns['input_gear'][indices[0]])
        qualified = indices[movement >= c['startup_input_rotation_excluded_rad']]
        if len(qualified) < block:
            directions[phase] = {'passed': False, 'error': 'No qualified input motion after startup slack'}
            continue
        selected = np.arange(qualified[0], indices[-1]+1)
        x = turns['input_gear'][selected]
        x0, slopes, ranges = x-x.mean(), {}, {}
        denominator = float(x0 @ x0)
        for name in active[1:]:
            y = turns[name][selected]
            slopes[name] = float(x0 @ (y-y.mean())/denominator) if denominator > 1.e-14 else None
        required = [('socket_crank', 'input_gear'), *pairs]
        chain = [all(positive_contact(rows[i], paths[a], paths[b]) for a, b in required) for i in selected]
        fractions = _block_fractions(chain, block)
        ranges = {name: float(np.ptp(values[selected])) for name, values in pair_states.items()}
        span = sign*float(x[-1]-x[0])
        tool_span = sign*float(turns['socket_crank'][selected[-1]]-turns['socket_crank'][selected[0]])
        conditions = {'real_input_span': span >= c['minimum_driven_input_span_rad'],
            'real_tool_span': tool_span >= c['minimum_driven_input_span_rad'],
            'contact_chain_each_20ms': bool(fractions) and min(fractions) >= c['minimum_chain_fraction_each_20ms'],
            'no_tooth_skip': not ranges or max(ranges.values()) <= c['maximum_pair_phase_range_rad']}
        for name, slope in slopes.items():
            expected = geometry['signed_angle_ratios']['idler_per_input' if name == 'idler_gear' else 'output_per_input']
            conditions[name+'_ratio'] = slope is not None and abs(slope-expected) <= c['ratio_slope_tolerance']
        directions[phase] = {'passed': all(conditions.values()), 'checks': conditions,
            'input_span_degrees': math.degrees(span), 'tool_span_degrees': math.degrees(tool_span),
            'ratio_slopes': slopes, 'pair_phase_ranges_rad': ranges,
            'minimum_20ms_chain_fraction': min(fractions) if fractions else None}
    backlash = {}
    for (a, b), (key, values) in zip(pairs, pair_states.items()):
        tails = [np.flatnonzero(np.asarray(phases) == p)[-round(.5/dt):] for p in ('forward', 'reverse')]
        measured = abs(float(np.median(values[tails[1]])-np.median(values[tails[0]])))
        expected = _pair_backlash_rad(a, b, geometry['gear_teeth'])
        backlash[key] = {'passed': measured >= c['minimum_measured_pair_backlash_rad']
            and abs(measured-expected) <= c['backlash_comparison_tolerance_rad'],
            'measured_reversal_rad': measured, 'expected_nominal_center_backlash_rad': expected}
    checks['both_directions'] = len(directions) == 2 and all(v['passed'] for v in directions.values())
    checks['all_active_pair_backlashes'] = len(backlash) == len(pairs) and all(v['passed'] for v in backlash.values())
    return {'passed': all(checks.values()), 'checks': checks, 'directions': directions, 'backlash': backlash,
        'socket_backlash_scope': 'Tool/input free play is reported through their real motion; normal hex clearance is not rejected as a defect.',
        'task_success_claimed': False}


def assess_stage(rows, scene_info, dt, case, inputs):
    """Assess one complete frozen stage; incomplete/unknown evidence fails."""
    result = {'passed': False, 'case_id': case.get('id'), 'task_success_claimed': False,
        'physics_accepted': False, 'checks': {}}
    try:
        evaluation = inputs['task']['evaluation']
        if case not in inputs['cases'] or dt != evaluation['physics_dt_s'] or not rows:
            raise ValueError('Declared case and exact frozen physics timestep required')
        result['motion'] = _motion(rows, dt, case['duration_s'], evaluation)
        checks = result['checks']
        checks['complete_motion_and_safety'] = result['motion']['passed']
        checks['wrench_receipts_match_native'] = all(
            r['wrench']['native_step'] == r['native_step']
            and abs(r['wrench']['time_s']-r['time_s']) < 1.e-9
            and r['wrench']['body'] == 'socket_crank' and r['wrench']['world_frame'] is True
            and len(r['wrench']['force_n']) == len(r['wrench']['torque_nm']) == 3
            and np.isfinite(r['wrench']['force_n']).all() and np.isfinite(r['wrench']['torque_nm']).all()
            for r in rows)
        checks['no_robot_interference'] = not any(r['robot_interference'] for r in rows)
        checks['penetration_bound'] = all(math.isfinite(r['maximum_penetration_m']) and 0. <= r['maximum_penetration_m'] <= evaluation['maximum_reported_penetration_m'] for r in rows)
        if case['id'] == 'supply':
            checks['zero_supply_wrench'] = all(r['wrench']['phase'] == 'supply'
                and r['wrench']['force_n'] == [0., 0., 0.] and r['wrench']['torque_nm'] == [0., 0., 0.]
                for r in rows)
            result['supply'] = assess_supply(rows, inputs['scene'], scene_info, dt)
            checks['supply'] = result['supply']['passed']
        else:
            if case['id'] == 'full_train':
                result['transmission'] = assess_transmission(rows, scene_info, dt, inputs['probe_contract'],
                    geometry=inputs['transmission_geometry_contract'])
            else:
                result['transmission'] = _partial_transmission(rows, scene_info, dt, case, inputs)
            checks['transmission'] = result['transmission']['passed']
            inactive = [n for n in OBJECTS if n not in (*case['active_gears'], 'socket_crank')]
            result['inactive_supply'] = assess_supply(rows, inputs['scene'], scene_info, dt, names=inactive)
            checks['inactive_supply'] = result['inactive_supply']['passed']
            windows, elapsed = {}, 0.
            for phase in inputs['probe_contract']['phases']:
                elapsed += phase['duration_s']
                if phase['id'] not in ('settle', 'final_hold'):
                    continue
                if case['id'] == 'full_train':
                    windows[phase['id']] = canonical_passive_window(rows, scene_info, end_time_s=elapsed,
                        phase=phase['id'], dt=dt, geometry=inputs['transmission_geometry_contract'])
                    # The older helper applies stable-window bounds to gears
                    # only. Preserve that result and also enforce the frozen
                    # task's used-tool rest bounds on the actual free socket.
                    strict = _partial_passive(rows, scene_info, dt, case, evaluation, elapsed, phase['id'])
                    windows[phase['id']]['active_and_used_tool_rest'] = strict
                    windows[phase['id']]['checks']['active_and_used_tool_rest'] = strict['passed']
                    windows[phase['id']]['passed'] = bool(windows[phase['id']]['passed'] and strict['passed'])
                else:
                    windows[phase['id']] = _partial_passive(rows, scene_info, dt, case, evaluation, elapsed, phase['id'])
            result['canonical_passive_windows'] = windows
            checks['canonical_passive_windows'] = len(windows) == 2 and all(w['passed'] for w in windows.values())
        result['passed'] = bool(checks) and all(checks.values())
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        result['error'] = str(exc)
    return result
