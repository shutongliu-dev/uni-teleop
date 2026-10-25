"""Predeclared contact-rig checks for free L5 gears; never a task score."""
from copy import deepcopy
import math

import numpy as np

from bench.geometry import quaternion_matrix
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.assets import gear_geometry, tooth_half_angle
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import geometry_contract as task_geometry_contract


GEARS = ('input_gear', 'idler_gear', 'output_gear')
CONTRACT = {
    'schema': 'mechanical_gear_load_contract.v1',
    'phases': [
        {'id': 'settle', 'duration_s': 1., 'target_speed_rad_s': 0., 'loaded': False},
        {'id': 'forward', 'duration_s': 4., 'target_speed_rad_s': math.radians(20.), 'loaded': True},
        {'id': 'coast', 'duration_s': .5, 'target_speed_rad_s': 0., 'loaded': False},
        {'id': 'reverse', 'duration_s': 5., 'target_speed_rad_s': -math.radians(20.), 'loaded': True},
        {'id': 'final_hold', 'duration_s': 1., 'target_speed_rad_s': 0., 'loaded': False}],
    'loaded_body': 'socket_crank', 'maximum_torque_nm': .8,
    'speed_load_coefficient_nm_s_per_rad': 2., 'external_force_n': [0., 0., 0.],
    'maximum_tool_speed_rad_s': 2., 'maximum_penetration_m': .0005,
    'abort_body_linear_speed_m_s': .5, 'abort_body_angular_speed_rad_s': 5.,
    'maximum_gear_tilt_rad': math.radians(.5), 'maximum_gear_axis_offset_m': .0004,
    'maximum_gear_bottom_error_m': .0003, 'seated_bottom_z_m': .802,
    'minimum_driven_input_span_rad': math.radians(15.),
    'startup_input_rotation_excluded_rad': math.radians(5.),
    'ratio_slope_tolerance': .05, 'maximum_pair_phase_range_rad': math.radians(2.),
    'minimum_chain_fraction_each_20ms': .5,
    'backlash_comparison_tolerance_rad': math.radians(.5),
    'minimum_measured_pair_backlash_rad': math.radians(.1),
    'scope': 'Privileged free-tool load rig. No robot strategy, assembly or task score. All three gears passive; no runtime object state writes.',
}


def transmission_geometry_contract(scene_info=None, geometry=None):
    """Return the frozen task geometry used by native transmission checks.

    Native ``scene_info`` intentionally contains only composed USD paths.  The
    task evaluation contract is therefore passed explicitly by validators when
    available; the current task contract is the safe default for standalone
    CPU evidence tests.  No legacy equal-gear geometry is inferred.
    """
    source = geometry
    if source is None and isinstance(scene_info, dict):
        source = scene_info.get('gear_geometry_contract')
    if isinstance(source, dict) and isinstance(source.get('evaluation'), dict):
        source = source['evaluation']
    if source is None:
        source = task_geometry_contract()
    teeth = source.get('gear_teeth')
    centers = source.get('shaft_centers_world_xy_m')
    ratios = source.get('signed_angle_ratios')
    if (not isinstance(teeth, dict) or tuple(teeth) != GEARS
            or any(type(teeth[n]) is not int or teeth[n] < 3 for n in GEARS)):
        raise ValueError('Transmission geometry must declare ordered gear tooth counts')
    if (not isinstance(centers, (list, tuple)) or len(centers) != len(GEARS)
            or any(len(p) != 2 or not np.isfinite(p).all() for p in centers)):
        raise ValueError('Transmission geometry must declare finite XY shaft centers')
    centers = np.asarray(centers, dtype=float)
    if np.any(np.linalg.norm(np.diff(centers, axis=0), axis=1) <= 0.):
        raise ValueError('Transmission shaft centers must be distinct')
    if (not isinstance(ratios, dict)
            or set(ratios) != {'idler_per_input', 'output_per_input'}
            or not all(math.isfinite(float(v)) for v in ratios.values())):
        raise ValueError('Transmission geometry must declare signed angle ratios')
    normalized = deepcopy(source)
    normalized['gear_teeth'] = {name: int(teeth[name]) for name in GEARS}
    normalized['shaft_centers_world_xy_m'] = centers.tolist()
    normalized['signed_angle_ratios'] = {name: float(ratios[name]) for name in ratios}
    normalized.setdefault('input_equivalent_turn_weights',
                          [1., normalized['gear_teeth']['idler_gear']/normalized['gear_teeth']['input_gear'], 1.])
    return normalized


def _pair_backlash_rad(first, second, teeth, *, pair_tangential_backlash_m=.0008):
    """Expected input-equivalent free phase for one external gear pair."""
    first_geometry = gear_geometry(teeth[first])
    second_geometry = gear_geometry(teeth[second])
    first_half = tooth_half_angle(first_geometry['pitch_radius_m'], first_geometry)
    second_half = tooth_half_angle(second_geometry['pitch_radius_m'], second_geometry)
    ratio = teeth[second] / teeth[first]
    return 2*math.pi/teeth[first] - 2*first_half - 2*ratio*second_half


def probe_contract(preset='standard'):
    """Select a declared engineering observation window and load variant."""
    if preset not in ('standard', 'extended_observation', 'high_torque'):
        raise ValueError('Unknown frozen gear probe contract preset')
    contract = deepcopy(CONTRACT)
    if preset == 'extended_observation':
        for phase in contract['phases']:
            if phase['id'] in ('forward', 'reverse'):
                phase['duration_s'] += 4.
    elif preset == 'high_torque':
        contract['engineering_variant'] = 'bounded_high_torque_v1'
        contract['maximum_torque_nm'] = 1.6
        contract['speed_load_coefficient_nm_s_per_rad'] = 5.
        contract['scope'] = ('Privileged free-tool load rig with an explicitly bounded higher '
            'diagnostic torque. No robot strategy, assembly or task score; all three gears passive.')
    return contract


def axial_rotation(pose):
    q = np.asarray(pose[3:], dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1.e-3:
        raise ValueError('Missing unit native orientation')
    if np.hypot(q[2], q[3]) < 1.e-8:
        raise ValueError('Undefined Z swing-twist orientation')
    return 2*math.atan2(q[2], q[3])


def contact_totals(row):
    """Yield signed native sums; explicit summaries never impersonate points."""
    if 'contact_totals' in row:
        if 'raw_contacts' in row or row.get('contact_totals_source') not in (
                'signed_raw_point_reduction.v2', 'native_backend_pair_sums_window.v1'):
            raise ValueError('Ambiguous contact evidence representation')
        for c in row['contact_totals']:
            magnitude = float(c['absolute_scalar_sum_ns'])
            vector_magnitudes = float(c['normal_vector_magnitude_sum_ns'])
            vector = np.asarray(c['signed_actor0_impulse_vector_ns'], dtype=float)
            if (not math.isfinite(magnitude) or magnitude < 0. or vector.shape != (3,)
                    or not math.isfinite(vector_magnitudes) or vector_magnitudes < 0.
                    or not np.isfinite(vector).all()
                    or math.hypot(*vector) > vector_magnitudes*(1.+1.e-9)
                    or not magnitude*(1.-32*2**-23) <= vector_magnitudes <= magnitude*(1.+32*2**-23)):
                raise ValueError('Invalid derived contact totals')
            yield c['actor0'], c['actor1'], magnitude, vector_magnitudes, vector
        return
    for c in row['raw_contacts']:
        values = np.asarray(c['normal_impulses_ns'], dtype=float)
        normals = np.asarray(c['normals'], dtype=float).reshape(-1, 3)
        if values.shape != (len(normals),):
            raise ValueError('Incomplete normal contact evidence')
        if not np.isfinite(values).all() or not np.isfinite(normals).all():
            raise ValueError('Nonfinite contact evidence')
        if len(normals) and np.max(np.abs(np.linalg.norm(normals, axis=1)-1.)) > 32*2**-23:
            raise ValueError('Native contact normals are not unit vectors')
        vectors = values[:, None]*normals
        magnitude, vector_magnitudes = float(np.abs(values).sum()), float(np.hypot.reduce(vectors, axis=1).sum())
        vector = vectors.sum(axis=0)
        if not math.isfinite(magnitude) or not math.isfinite(vector_magnitudes) or not np.isfinite(vector).all():
            raise ValueError('Nonfinite summed native contact evidence')
        yield c['actor0'], c['actor1'], magnitude, vector_magnitudes, vector


def compact_native_row(row):
    """Discard point arrays only after retaining their exact directed sums."""
    compact = {key: value for key, value in row.items() if key not in ('raw_contacts', 'robot_interference')}
    compact['robot_interference'] = bool(row['robot_interference'])
    compact['contact_totals_source'] = 'signed_raw_point_reduction.v2'
    compact['contact_totals'] = [dict(actor0=a, actor1=b, absolute_scalar_sum_ns=magnitude,
        normal_vector_magnitude_sum_ns=vector_magnitudes, signed_actor0_impulse_vector_ns=vector.tolist())
        for a, b, magnitude, vector_magnitudes, vector in contact_totals(row)]
    return compact


def positive_contact(row, first, second, *, upward=False, second_is_prefix=False):
    total = 0.
    for a, b, magnitude, _, vector in contact_totals(row):
        if first not in (a, b):
            continue
        other = b if a == first else a
        if not (other == second or (second_is_prefix and other.startswith(second+'/'))):
            continue
        total += float(vector[2])*(1 if a == first else -1) if upward else magnitude
    return total > 1.e-8


def _block_fractions(values, count):
    return [float(np.mean(values[i:i+count])) for i in range(0, len(values), count)]


def assess_transmission(rows, scene_info, dt, contract=None, *, geometry=None):
    """Fail closed on missing samples, either transmission direction or contacts."""
    c = deepcopy(CONTRACT if contract is None else contract)
    result = {'passed': False, 'task_success_claimed': False, 'checks': {}, 'directions': {}, 'backlash': {}}
    try:
        task_geometry = transmission_geometry_contract(scene_info, geometry)
        teeth = task_geometry['gear_teeth']
        centers = np.asarray(task_geometry['shaft_centers_world_xy_m'], dtype=float)
        expected_ratios = task_geometry['signed_angle_ratios']
        result['geometry_contract'] = {
            'gear_geometry_version': task_geometry.get('gear_geometry_version'),
            'gear_teeth': deepcopy(teeth),
            'shaft_centers_world_xy_m': centers.tolist(),
            'signed_angle_ratios': deepcopy(expected_ratios),
        }
        block = round(.02/dt)
        expected = round(sum(p['duration_s'] for p in c['phases'])/dt)
        if block < 1 or abs(block*dt-.02) > 1.e-9 or len(rows) != expected:
            raise ValueError('Incomplete expected native integration stream')
        if any(b['native_step']-a['native_step'] != 1 or abs(b['time_s']-a['time_s']-dt) > 1.e-8
               for a, b in zip(rows, rows[1:])):
            raise ValueError('Noncontiguous native integration stream')
        paths = {name: scene_info['objects'][name]['path'] for name in (*GEARS, 'socket_crank')}
        fixture = scene_info['fixtures']['gear_workplate']['path']
        poses = {name: np.array([r['poses_xyzw'][name] for r in rows]) for name in paths}
        if any(not np.isfinite(p).all() or p.shape != (len(rows), 7) for p in poses.values()):
            raise ValueError('Incomplete body poses')
        turns = {name: np.unwrap([axial_rotation(p) for p in poses[name]]) for name in paths}
        checks = result['checks']
        checks['complete_native_sampling'] = True
        checks['no_robot_interference'] = not any(r['robot_interference'] for r in rows)
        checks['penetration_bound'] = max(r['maximum_penetration_m'] for r in rows) <= c['maximum_penetration_m']
        # Sidecar command receipts are joined by native_step before assessment.
        expected_phases = []
        phase_definitions = {phase['id']: phase for phase in c['phases']}
        for phase in c['phases']:
            expected_phases.extend([phase['id']]*round(phase['duration_s']/dt))
        checks['bounded_tool_only_load_receipts'] = all(
            r['wrench']['native_step'] == r['native_step'] and r['wrench']['body'] == 'socket_crank'
            and r['wrench']['phase'] == phase and r['wrench']['force_n'] == [0., 0., 0.]
            and r['wrench']['world_frame'] is True and len(r['wrench']['torque_nm']) == 3
            and np.isfinite(r['wrench']['torque_nm']).all()
            and r['wrench']['torque_nm'][:2] == [0., 0.]
            and abs(r['wrench']['torque_nm'][2]) <= c['maximum_torque_nm']+1.e-8
            and (phase_definitions[phase]['loaded'] or r['wrench']['torque_nm'] == [0., 0., 0.])
            for r, phase in zip(rows, expected_phases))
        checks['tool_speed_bound'] = bool(max(np.linalg.norm(r['velocities']['socket_crank'][3:])
            for r in rows) <= c['maximum_tool_speed_rad_s'])
        velocities = np.array([[r['velocities'][name] for name in paths] for r in rows])
        checks['all_body_motion_safety_bound'] = bool(velocities.shape == (len(rows), len(paths), 6)
            and np.isfinite(velocities).all()
            and np.linalg.norm(velocities[:, :, :3], axis=2).max() <= c['abort_body_linear_speed_m_s']
            and np.linalg.norm(velocities[:, :, 3:], axis=2).max() <= c['abort_body_angular_speed_rad_s'])
        active = np.array([p != 'settle' for p in expected_phases])
        geometry = {}
        for index, name in enumerate(GEARS):
            center = centers[index]
            p = poses[name][active]
            tilt = max(math.acos(float(np.clip(quaternion_matrix([q[6], *q[3:6]])[2, 2], -1., 1.))) for q in p)
            eccentricity = float(np.linalg.norm(p[:, :2]-center, axis=1).max())
            height_error = float(np.abs(p[:, 2]-c['seated_bottom_z_m']).max())
            geometry[name] = {'tilt_rad': tilt, 'axis_offset_m': eccentricity, 'bottom_height_error_m': height_error}
            checks[name+'_supported_geometry'] = (tilt <= c['maximum_gear_tilt_rad']
                and eccentricity <= c['maximum_gear_axis_offset_m'] and height_error <= c['maximum_gear_bottom_error_m'])
        result['gear_geometry'] = geometry
        for name in GEARS:
            support = [positive_contact(r, paths[name], fixture, upward=True, second_is_prefix=True)
                       for r, use in zip(rows, active) if use]
            fractions = _block_fractions(support, block)
            checks[name+'_upward_support_each_20ms'] = bool(fractions) and min(fractions) > 0.
        pair_states = {}
        for first, second in zip(GEARS, GEARS[1:]):
            relative = poses[second][:, :2]-poses[first][:, :2]
            line = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
            tooth_ratio = teeth[second]/teeth[first]
            phase_sum = turns[first]+tooth_ratio*turns[second]-(1.+tooth_ratio)*line
            distance = np.linalg.norm(relative, axis=1)
            pair_backlash = _pair_backlash_rad(first, second, teeth)
            backlash = np.full(len(distance), pair_backlash, dtype=float)
            pair_states[first+'__'+second] = (phase_sum, backlash)
        for phase, sign in (('forward', 1), ('reverse', -1)):
            indices = np.flatnonzero(np.array(expected_phases) == phase)
            movement = sign*(turns['input_gear'][indices]-turns['input_gear'][indices[0]])
            selected = indices[movement >= c['startup_input_rotation_excluded_rad']]
            if len(selected) < block:
                result['directions'][phase] = {'passed': False, 'error': 'No qualified input motion after startup slack'}
                continue
            # Do not select disjoint successful samples around a reversal/stall.
            selected = np.arange(selected[0], indices[-1]+1)
            x = turns['input_gear'][selected]
            span = sign*float(x[-1]-x[0])
            slopes, phase_ranges = {}, {}
            x0 = x-x.mean()
            denominator = float(x0 @ x0)
            for name in GEARS[1:]:
                y = turns[name][selected]
                slopes[name] = float(x0 @ (y-y.mean())/denominator) if denominator > 1.e-14 else None
            chain = []
            for i in selected:
                r = rows[i]
                chain.append(all(positive_contact(r, paths[a], paths[b]) for a, b in
                    (('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))))
            fractions = _block_fractions(chain, block)
            for key, (phase_sum, _) in pair_states.items():
                phase_ranges[key] = float(np.ptp(phase_sum[selected]))
            direction_checks = {
                'real_input_span': span >= c['minimum_driven_input_span_rad'],
                'idler_ratio': slopes['idler_gear'] is not None and abs(slopes['idler_gear']-expected_ratios['idler_per_input']) <= c['ratio_slope_tolerance'],
                'output_ratio': slopes['output_gear'] is not None and abs(slopes['output_gear']-expected_ratios['output_per_input']) <= c['ratio_slope_tolerance'],
                'no_tooth_skip': max(phase_ranges.values()) <= c['maximum_pair_phase_range_rad'],
                'contact_chain_each_20ms': bool(fractions) and min(fractions) >= c['minimum_chain_fraction_each_20ms']}
            result['directions'][phase] = {'passed': all(direction_checks.values()), 'checks': direction_checks,
                'input_span_degrees': math.degrees(span), 'ratio_slopes': slopes,
                'pair_phase_ranges_rad': phase_ranges, 'minimum_20ms_chain_fraction': min(fractions) if fractions else None,
                'first_native_step': rows[selected[0]]['native_step'], 'last_native_step': rows[selected[-1]]['native_step']}
        for key, (phase_sum, expected_backlash) in pair_states.items():
            tails = [np.flatnonzero(np.array(expected_phases) == phase)[-round(.5/dt):]
                     for phase in ('forward', 'reverse')]
            measured = abs(float(np.median(phase_sum[tails[1]])-np.median(phase_sum[tails[0]])))
            geometric = float(sum(np.median(expected_backlash[t]) for t in tails)/2)
            result['backlash'][key] = {'passed': measured >= c['minimum_measured_pair_backlash_rad']
                and abs(measured-geometric) <= c['backlash_comparison_tolerance_rad'],
                'measured_reversal_rad': measured, 'expected_at_measured_centers_rad': geometric,
                'comparison_tolerance_rad': c['backlash_comparison_tolerance_rad'],
                'scope': 'Loaded flank phase difference corrected for measured center-line rotation; includes native contact offset and facet effects.'}
        tail = rows[-round(.5/dt):]
        checks['final_passive_rest'] = all(np.linalg.norm(r['velocities'][name][:3]) <= .005
            and np.linalg.norm(r['velocities'][name][3:]) <= .05 for r in tail for name in paths)
        checks['both_directions'] = len(result['directions']) == 2 and all(r['passed'] for r in result['directions'].values())
        checks['both_pair_backlashes'] = len(result['backlash']) == 2 and all(r['passed'] for r in result['backlash'].values())
        result['passed'] = bool(checks) and all(checks.values())
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        result['error'] = str(exc)
    return result
