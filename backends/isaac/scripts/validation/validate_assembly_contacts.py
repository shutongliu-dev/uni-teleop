"""Frozen native mechanical contact probes; gravity/reset tests, no policy score.

Inputs contain the actual task scene plus explicit reset-only diagnostic cases.
Every integration substep is observed once. Cases may opt into a frozen,
privileged robot joint phase list; objects are still never written after each
case reset and the resulting report remains an engineering diagnostic.
"""
from __future__ import annotations
import argparse
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback
import numpy as np

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.rules import COMMON

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import dump, digest
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, inertia_principal_axes
from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_board_game_pickup import contact_rows
from scripts.validation.validate_mechanical_dynamics import declared_mount_pairs
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_convex_geometry import check_convex_geometry
from scripts.validation.mechanical_tool_checks import contact_exchange_check
from scripts.validation.validate_workrooms import plain, stage_inventory, capture, angle, smooth_targets
from scripts.validation.validate_scenario import seal_report_before_shutdown


def native_parameters(env):
    """Check actual task inertia, COM, material rows and collider offsets."""
    from bench.simulation.isaacsim_backend.env import _np
    from pxr import PhysxSchema, UsdPhysics
    records = {}
    for spec in env.scene_spec['objects']:
        name = spec['id']
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        view = env._object_views[name]
        if view.count != 1:
            raise ValueError('Missing unique native object: '+name)
        mass = float(_np(view.get_masses()).reshape(-1)[0])
        inverse_mass = float(_np(view.get_inv_masses()).reshape(-1)[0])
        inertia = _np(view.get_inertias()).reshape(3, 3).T
        inverse_inertia = _np(view.get_inv_inertias()).reshape(3, 3).T
        com = _np(view.get_coms()).reshape(7)
        rotation = quaternion_matrix(inertia_principal_axes(asset))
        expected_inertia = rotation @ np.diag(asset['inertia_diag_kg_m2']) @ rotation.T
        materials = _np(view.get_material_properties())[0]
        material_expected = [spec.get('friction', .65)]*2+[0.]
        offsets = {}
        for path in env.scene_info['objects'][name]['collision_paths']:
            prim = env.stage.GetPrimAtPath(path)
            api = PhysxSchema.PhysxCollisionAPI(prim)
            offsets[path] = {'enabled': UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,
                'contact_offset_m': float(api.GetContactOffsetAttr().Get()),
                'rest_offset_m': float(api.GetRestOffsetAttr().Get())}
        checks = {
            'mass': bool(np.isclose(mass, asset['mass_kg'], atol=1e-7, rtol=1e-5)),
            'inverse_mass': bool(np.isclose(mass*inverse_mass, 1., atol=1e-6, rtol=0)),
            'com': bool(np.allclose(com[:3], asset['center_of_mass_m'], atol=1e-7, rtol=0)),
            'inertia_body_frame': bool(np.allclose(inertia, expected_inertia, atol=1e-9, rtol=2e-5)),
            'positive_definite_inertia': bool(np.isfinite(inertia).all() and np.linalg.eigvalsh(inertia).min() > 0),
            'inverse_inertia': bool(np.allclose(inertia@inverse_inertia, np.eye(3), atol=2e-5, rtol=0)),
            'all_native_material_rows': bool(len(materials) == len(asset['collision_parts']) and
                np.allclose(materials, np.tile(material_expected, (len(materials), 1)), atol=1e-6, rtol=0)),
            'all_usd_collision_offsets': bool(offsets and all(row['enabled'] and
                abs(row['contact_offset_m']-spec.get('contact_offset_m', .001)) < 1e-8 and
                abs(row['rest_offset_m']-spec.get('rest_offset_m', 0.)) < 1e-8 for row in offsets.values()))}
        records[name] = {'passed': all(checks.values()), 'checks': checks,
            'mass_kg': mass, 'native_inverse_mass_kg_inv': inverse_mass,
            'native_inertia_body_frame_kg_m2': inertia.tolist(),
            'native_inverse_inertia_body_frame': inverse_inertia.tolist(),
            'expected_inertia_body_frame_kg_m2': expected_inertia.tolist(),
            'native_com_principal_pose_xyzw': com.tolist(), 'native_material_rows': materials.tolist(),
            'usd_collision_offsets': offsets,
            'inertia_api_reference': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html',
            'scope': 'Mass/COM/full inertia/material rows read from native body views; offsets/enabled flags are USD readback. Functional contact checked by native motion.'}
    fixtures = {}
    # Each plain fixture collider owns exactly one native fixed actor/shape.
    # The room-wide view also includes multi-shape SDF actors and pads the
    # material axis to max_shapes; those padding rows are not materials.
    requested = [p for spec in env.scene_spec['fixtures']
                 for p in env.scene_info['fixtures'][spec['id']]['collision_paths']]
    fixed = env._sim_view.create_rigid_body_view(requested)
    paths = list(fixed.prim_paths)
    if (len(paths) != len(requested) or set(paths) != set(requested) or fixed.max_shapes != 1
            or not set(paths) <= set(env._fixed_actor_view.prim_paths)):
        raise ValueError('Plain fixture native inventory must have exactly one shape per guarded actor')
    properties = _np(fixed.get_material_properties())
    for spec in env.scene_spec['fixtures']:
        values = {}
        expected = [spec.get('friction', .65)]*2+[0.]
        for path in env.scene_info['fixtures'][spec['id']]['collision_paths']:
            prim = env.stage.GetPrimAtPath(path)
            api = PhysxSchema.PhysxCollisionAPI(prim)
            material = properties[paths.index(path)] if path in paths else np.empty((0, 3))
            values[path] = {'native_material_rows': material.tolist(),
                'passed': bool(len(material) and np.allclose(material, expected, atol=1e-6, rtol=0)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
                    and abs(float(api.GetContactOffsetAttr().Get())-spec.get('contact_offset_m', .001)) < 1e-8
                    and abs(float(api.GetRestOffsetAttr().Get())-spec.get('rest_offset_m', 0.)) < 1e-8)}
        fixtures[spec['id']] = {'passed': bool(values) and all(r['passed'] for r in values.values()),
                                 'colliders': values}
    base = env.metadata()['runtime_robot_base_check']
    return {'passed': bool(records) and all(r['passed'] for r in records.values())
            and bool(fixtures) and all(r['passed'] for r in fixtures.values())
            and base.get('verification_passed') is True,
            'objects': records, 'fixtures': fixtures, 'native_robot_base': base}


class Observer:
    def __init__(self, env, output):
        self.env, self.original = env, env.world.step
        self.friction = None
        if env.scene_spec.get('engineering_contact_friction_capture') is True:
            from scripts.validation.rigid_friction_receipts import FrictionReceipts
            self.friction = FrictionReceipts(env, output)
        self.exchange = SubstepContactExchange(env, self)
        self.stream = (output/'native_frames.jsonl').open('x')
        self.phase = 'hold_ready'
        self.rows = []
        self.robot = set(env.scene_info['robot_link_paths'].values())
        self.mount = {tuple(sorted(p)) for p in declared_mount_pairs(env)}
        self.callback = self.step
        env.world.step = self.callback

    def step(self, *args, **kwargs):
        if args or kwargs != {'render': False}:
            raise RuntimeError('Observer permits ordinary native integration only')
        env = self.env
        self.exchange.begin_step(env)
        result = self.original(*args, **kwargs)
        sample = self.exchange.acquire(env)
        state = env.capture_state()
        if not all(np.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite native state')
        raw = contact_rows(env, sample)
        projected = env._project_rigid_contacts(sample)
        interference = [r for r in raw if r['impulse_magnitude_sum_ns'] > 1e-8 and
            self.robot.intersection((r['actor0'], r['actor1'])) and
            tuple(sorted((r['actor0'], r['actor1']))) not in self.mount]
        row = {'phase': self.phase,
            'native_step': int(env.world.current_time_step_index),
            'time_s': float(env._current_step_time),
            'poses_xyzw': {n: state['object_q'][i].tolist() for i, n in enumerate(env.object_names)},
            'velocities': {n: state['object_qd'][i].tolist() for i, n in enumerate(env.object_names)},
            'raw_contacts': raw, 'robot_interference': interference,
            'maximum_penetration_m': max([0., *[-float(d) for r in raw for d in r['separations_m']]])}
        if env.scene_spec.get('engineering_contact_force_crosscheck') is True:
            from bench.simulation.isaacsim_backend.env import _np
            net = _np(env._rigid_contacts.get_net_contact_forces(dt=1.)).copy()
            matrix = _np(env._rigid_contacts.get_contact_force_matrix(dt=1.)).copy()
            row['native_contact_impulse_crosscheck'] = {}
            for name in env.object_names:
                index = env._contact_sensor_paths.index(env.scene_info['objects'][name]['path'])
                row['native_contact_impulse_crosscheck'][name] = {
                    'net_tensor_ns': net[index].tolist(),
                    'pair_matrix_sum_ns': matrix[index].sum(axis=0).tolist()}
        if self.friction is not None:
            row['native_friction_receipt'] = self.friction.capture(env, sample)
        after = None
        try:
            after = env.capture_state()
            if not all(np.array_equal(state[k], after[k]) for k in state):
                raise RuntimeError('Read-only observation modified state')
            # The final independent capture must also preserve the native
            # clock/view/mapping before a row can be recorded as valid.
            sample.for_env(env)
        except Exception as exc:
            if self.friction is not None:
                self.friction.reject_readonly(env, state, after, exc)
            raise
        if self.friction is not None:
            self.friction.accept_readonly()
        self.rows.append(row)
        self.stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        self.exchange.offer_after_readonly(env, sample, projected)
        return result

    def close(self):
        self.env.world.step = self.original
        try:
            return self.exchange.close(self.env, self)
        finally:
            self.stream.close()


def motion_evidence(rows, names, duration_s, dt):
    """Audit complete reset-relative clocks and motion without retaining contacts.

    The stable interval includes both endpoint poses. Native velocity and
    sampled pose changes independently constrain motion, as in the task rules.
    This also accepts a stream for read-only reassessment of historical raw data.
    """
    duration_s, dt = float(duration_s), float(dt)
    if (not np.isfinite([duration_s, dt]).all() or dt <= 0 or duration_s <= COMMON['minimum_stable_s']
            or abs(.02/dt-round(.02/dt)) > 1e-8
            or abs(duration_s/dt-round(duration_s/dt)) > 1e-8):
        raise ValueError('A declared duration above the stable window and a complete native grid are required')
    expected_count = round(duration_s/dt)
    stable_count = round(COMMON['minimum_stable_s']/dt)
    control_steps = round(COMMON['control_period_s']/dt)
    tail = deque(maxlen=stable_count+1)
    maxima = {name: {'native_linear': 0., 'native_angular': 0.,
                    'sampled_linear': 0., 'sampled_angular': 0.,
                    'native_step_sampled_linear': 0., 'native_step_sampled_angular': 0.} for name in names}
    count, previous, previous_step, previous_control = 0, None, None, None
    complete, finite = True, True
    first_time = last_time = None

    def qangle(a, b):
        # Equivalent rotations q and -q have zero distance.
        aligned = b if np.dot(a,b)>=0 else -b
        return float(4*np.arctan2(np.linalg.norm(a-aligned), np.linalg.norm(a+aligned)))

    for row in rows:
        count += 1
        step, time = row.get('native_step'), row.get('time_s')
        valid_time = isinstance(time, (int, float)) and not isinstance(time, bool) and np.isfinite(time)
        complete &= (type(step) is int and (previous_step is None or
                     type(previous_step) is int and step == previous_step+1)
                     and valid_time and abs(time-count*dt) <= 1e-8)
        previous_step = step
        first_time = time if count == 1 else first_time
        last_time = time
        states = {}
        for name in names:
            pose = np.asarray(row['poses_xyzw'][name], dtype=float)
            velocity = np.asarray(row['velocities'][name], dtype=float)
            if (pose.shape != (7,) or velocity.shape != (6,) or not np.isfinite(pose).all()
                    or not np.isfinite(velocity).all() or abs(np.linalg.norm(pose[3:])-1.) > 1e-3):
                finite = False
                continue
            p, q = pose[:3], pose[3:]/np.linalg.norm(pose[3:])
            linear, angular = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
            sampled_linear = sampled_angular = native_sampled_linear = native_sampled_angular = 0.
            if previous is not None and name in previous:
                native_sampled_linear = float(np.linalg.norm(p-previous[name]['p'])/dt)
                native_sampled_angular = qangle(q, previous[name]['q'])/dt
            if count % control_steps == 0 and previous_control is not None and name in previous_control:
                sampled_linear = float(np.linalg.norm(p-previous_control[name]['p'])/COMMON['control_period_s'])
                sampled_angular = qangle(q,previous_control[name]['q'])/COMMON['control_period_s']
            values = dict(native_linear=linear, native_angular=angular,
                          sampled_linear=sampled_linear, sampled_angular=sampled_angular,
                          native_step_sampled_linear=native_sampled_linear,
                          native_step_sampled_angular=native_sampled_angular)
            for key, value in values.items():
                maxima[name][key] = max(maxima[name][key], value)
            states[name] = {'p': p, 'q': q, **values}
        previous = states
        if count % control_steps == 0:
            previous_control = states
        tail.append(states)
    complete &= count == expected_count and len(tail) == stable_count+1
    checks = {'complete_native_sampling': bool(complete), 'finite_native_state': finite,
              'object_motion_safety': bool(finite and all(
                  max(v['native_linear'],v['sampled_linear']) <= COMMON['maximum_object_linear_speed_m_s']
                  and max(v['native_angular'],v['sampled_angular']) <= COMMON['maximum_object_angular_speed_rad_s']
                  for v in maxima.values()))}
    bodies = {}
    if complete and finite:
        for name in names:
            states = [row[name] for row in tail]
            stable_linear = max(max(row['native_linear'],row['sampled_linear']) for row in states[1:])
            stable_angular = max(max(row['native_angular'],row['sampled_angular']) for row in states[1:])
            # Include the anchor's native velocity; its pose-change rate belongs
            # to the interval immediately before the stable window.
            stable_linear = max(stable_linear,states[0]['native_linear'])
            stable_angular = max(stable_angular,states[0]['native_angular'])
            position_range = max(float(np.linalg.norm(row['p']-states[0]['p'])) for row in states)
            orientation_range = max(qangle(row['q'],states[0]['q']) for row in states)
            stable_checks = {
                'linear_rest': stable_linear <= COMMON['maximum_stable_linear_speed_m_s'],
                'angular_rest': stable_angular <= COMMON['maximum_stable_angular_speed_rad_s'],
                'stable_position': position_range <= COMMON['maximum_stable_position_range_m'],
                'stable_orientation': orientation_range <= COMMON['maximum_stable_orientation_range_rad']}
            bodies[name] = {'checks': stable_checks, 'whole_run_maxima': maxima[name],
                'maximum_stable_linear_speed_m_s': stable_linear,
                'maximum_stable_angular_speed_rad_s': stable_angular,
                'stable_position_range_m': position_range,
                'stable_orientation_range_rad': orientation_range}
    return {'checks': checks, 'objects': bodies, 'observed_substeps': count,
            'expected_substeps': expected_count, 'first_time_s': first_time, 'last_time_s': last_time,
            'stable_window_s': COMMON['minimum_stable_s'], 'stable_endpoint_samples': len(tail),
            'sampled_motion_gate_period_s': COMMON['control_period_s'],
            'native_step_pose_difference_scope': 'Diagnostic maxima only; not an additional motion gate.',
            'control_pose_difference_intervals': max(0,count//control_steps-1),
            'first_control_pose_interval_scope': 'No native reset-time pose in this stream; first 20 ms pose difference is unavailable. Native velocities cover every recorded substep.',
            'limits': {key: COMMON[key] for key in COMMON if key.startswith('maximum_')}}


def assess_case(rows, case, scene_info, dt):
    """Independent position/axis/support requirements, specified before launch."""
    window = rows[-round(.5/dt):]
    motion = motion_evidence(rows, tuple(case['assertions']), case['duration_s'], dt)
    checks = {**motion['checks'],
        'no_robot_interference': not any(r['robot_interference'] for r in rows),
        'reported_penetration': max((r['maximum_penetration_m'] for r in rows), default=float('inf')) <= .0005}
    if not checks['complete_native_sampling'] or not checks['finite_native_state']:
        return {'passed': False, 'checks': checks, 'objects': {}, 'motion_evidence': motion,
                'functional_contact_success': False, 'expected_functional_success': case.get('expected_functional_success', True),
                'scope': 'Invalid or incomplete native motion evidence.'}
    bodies = {}
    for name, expected in case['assertions'].items():
        poses = np.array([r['poses_xyzw'][name] for r in window])
        path = scene_info['objects'][name]['path']
        support = expected['support_path_prefix']
        supported = []
        for r in window:
            # Native scalar impulses can be signed. Their product with the
            # native normal is the vector on actor0; retain both signs before
            # checking net upward support, including opposing point loads.
            impulse = sum(float(s)*float(n[2])*(1 if c['actor0'] == path else -1)
                for c in r['raw_contacts']
                if path in (c['actor0'], c['actor1']) and any(p.startswith(support)
                    for p in (c['actor0'], c['actor1']) if p != path)
                for s, n in zip(c['normal_impulses_ns'], c['normals']))
            supported.append(impulse)
        # Require support in each complete 20 ms block, rather than a normal
        # impulse every substep (resting rigid contacts may pulse).
        block = round(.02/dt)
        support_windows = [sum(supported[i:i+block]) > 1e-8 for i in range(0, len(supported), block)]
        position_error = np.abs(poses[:, :3]-expected['position_m']).max(axis=0)
        tilt = max(np.arccos(np.clip(quaternion_matrix([p[6], *p[3:6]])[2, 2], -1, 1)) for p in poses)
        displacement = float(rows[0]['poses_xyzw'][name][2]-poses[-1, 2])
        body_checks = {'position': bool(np.all(position_error <= expected['position_tolerance_m'])),
            'upright': tilt <= expected.get('maximum_tilt_rad', .05),
            **motion['objects'][name]['checks'],
            'support_each_control_window': bool(support_windows) and all(support_windows),
            'actual_descent': displacement >= expected.get('minimum_descent_m', 0.)}
        guidance_impulse = 0.
        if expected.get('require_guidance_contact'):
            for r in rows:
                if r['poses_xyzw'][name][2] <= expected['position_m'][2]+.003:
                    continue
                for c in r['raw_contacts']:
                    if (path in (c['actor0'], c['actor1']) and any(
                            p.startswith(expected['guidance_path_prefix'])
                            for p in (c['actor0'], c['actor1']) if p != path)):
                        guidance_impulse += sum(abs(float(s))*float(np.linalg.norm(n[:2]))
                            for s, n in zip(c['normal_impulses_ns'], c['normals']))
            body_checks['actual_guidance_contact'] = guidance_impulse > 1e-8
        bodies[name] = {'passed': bool(all(body_checks.values())), 'checks': plain(body_checks),
            'maximum_position_error_m': position_error.tolist(), 'maximum_tilt_rad': float(tilt),
            'actual_descent_m': displacement, 'support_window_count': sum(support_windows),
            'checked_window_count': len(support_windows), 'final_pose_xyzw': poses[-1].tolist(),
            'guidance_radial_impulse_ns': guidance_impulse}
    functional = bool(bodies) and all(v['passed'] for v in bodies.values())
    expected_success = case.get('expected_functional_success', True)
    if type(expected_success) is not bool:
        raise ValueError('A probe must explicitly expect success or rejection')
    checks['task_contact_assertions'] = functional == expected_success
    if not expected_success:
        names = case.get('negative_control_objects')
        if names is not None and (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or name not in bodies for name in names)
                or len(names) != len(set(names))):
            raise ValueError('Negative control objects must be a nonempty unique subset of assertions')
        selected = bodies if names is None else {name: bodies[name] for name in names}
        selected_paths = None if names is None else {scene_info['objects'][name]['path'] for name in names}
        if names is not None:
            checks['negative_control_inactive_inventory'] = all(
                body['passed'] for name, body in bodies.items() if name not in selected)
        # Rejection must be a wrong assembled pose after real settling, rather
        # than missing observations, solver instability or empty contacts. An
        # explicit active-object subset does not require unrelated supply
        # inventory to descend; inactive inventory must pass its own assertions.
        # Omitting the subset preserves the original all-body negative contract.
        checks['negative_control_is_settled_wrong_pose'] = bool(selected) and (
            any(not b['checks']['position'] for b in selected.values()) and
            all(all(b['checks'][k] for k in ('linear_rest', 'angular_rest', 'stable_position', 'stable_orientation'))
                and b['actual_descent_m'] >= .001 for b in selected.values()) and
            any(c['impulse_magnitude_sum_ns'] > 1e-8 and (selected_paths is None
                    or selected_paths.intersection((c['actor0'], c['actor1'])))
                for r in window for c in r['raw_contacts']))
    return {'passed': all(checks.values()), 'checks': checks, 'objects': bodies,
            'motion_evidence': motion,
            'functional_contact_success': functional, 'expected_functional_success': expected_success,
            'scope': 'Declared contact probe, not manipulation success.'}


def _validate_robot_phases(phases, initial):
    """Validate an optional native robot-only phase contract before stepping.

    A phase is a complete joint-target trajectory.  It is deliberately
    independent of object state: interpolation only changes the command sent
    to the robot, while the observer checks the native object state after every
    integration step.  This keeps a scripted endpoint from being mistaken for
    a grasp or placement.
    """
    if phases is None:
        return []
    if not isinstance(phases, list) or not phases:
        raise ValueError('robot_phases must be a nonempty list when declared')
    from scripts.validation.mechanical_robot_phase_timing import validate_timing
    names, checked = set(), []
    previous = initial
    for phase in phases:
        if not isinstance(phase, dict) or not isinstance(phase.get('id'), str):
            raise ValueError('Each robot phase requires a string id')
        name, target, duration = phase['id'], phase.get('joint_positions_rad'), phase.get('duration_s')
        if name in names or not name.isidentifier():
            raise ValueError('Robot phase ids must be unique identifiers')
        names.add(name)
        if (not isinstance(target, dict) or set(target) != set(initial)
                or not np.isfinite(list(target.values())).all()
                or type(duration) not in (int, float) or not np.isfinite(duration)
                or duration < .02 or duration > 10.
                or abs(duration/.02-round(duration/.02)) > 1e-7):
            raise ValueError('Robot phases require complete finite joint targets and 20 ms durations')
        checked.append({'id': name, 'duration_s': float(duration),
                        'joint_positions_rad': {key: float(value) for key, value in target.items()},
                        'scope': phase.get('scope', 'robot target only; no object pose writeback')})
        timing = validate_timing(phase, previous)
        if timing is not None:
            checked[-1]['timing'] = timing
        contact_prefix = phase.get('hand_contact_actor_prefix')
        if contact_prefix is not None:
            if (timing is None or not isinstance(contact_prefix, str)
                    or not contact_prefix.startswith('/World/Objects/')
                    or not contact_prefix.removeprefix('/World/Objects/').isidentifier()):
                raise ValueError('Contact-limited phases require a timed, explicit task object prefix')
            checked[-1]['hand_contact_actor_prefix'] = contact_prefix
        previous = target
    return checked


def _validate_privileged_input_contract(inputs):
    """Fail closed on frozen mechanical native inputs before opening a GPU env."""
    schema = inputs.get('schema')
    supported = ('mechanical_l1_native_probe_input.v1',
                 'mechanical_l2_native_probe_input.v1',
                 'mechanical_l3_native_probe_input.v1',
                 'mechanical_l4_loose_native_probe_input.v1',
                 'mechanical_l5_gear_native_probe_input.v1')
    if schema not in supported:
        return {'passed': True, 'checked': False, 'scope': 'Legacy contact-probe input without an L3 phase contract.'}
    is_l1 = schema == 'mechanical_l1_native_probe_input.v1'
    is_l2 = schema == 'mechanical_l2_native_probe_input.v1'
    is_l3 = schema == 'mechanical_l3_native_probe_input.v1'
    is_l4 = schema == 'mechanical_l4_loose_native_probe_input.v1'
    is_l5 = schema == 'mechanical_l5_gear_native_probe_input.v1'
    task_ids = {
        'mechanical_l1_native_probe_input.v1': 'parts_sorting',
        'mechanical_l2_native_probe_input.v1': 'peg_insertion',
        'mechanical_l3_native_probe_input.v1': 'shaft_spacer_assembly',
        'mechanical_l4_loose_native_probe_input.v1': 'fastener_assembly',
        'mechanical_l5_gear_native_probe_input.v1': 'gear_train_assembly',
    }
    labels = {
        'mechanical_l1_native_probe_input.v1': 'L1',
        'mechanical_l2_native_probe_input.v1': 'L2',
        'mechanical_l3_native_probe_input.v1': 'L3',
        'mechanical_l4_loose_native_probe_input.v1': 'L4',
        'mechanical_l5_gear_native_probe_input.v1': 'L5',
    }
    label = labels[schema]
    required = {
        'task_id': task_ids[schema], 'model_calls': 0,
        'ordinary_agent_validation': False, 'task_success_claimed': False,
        'physics_accepted': False, 'object_state_writeback_after_reset': False,
        'external_object_forces': False,
    }
    mismatches = {key: {'expected': value, 'actual': inputs.get(key)}
                  for key, value in required.items() if inputs.get(key) != value}
    release = Path(inputs.get('release', ''))
    manifest = release/'release_manifest.json'
    release_hash = digest(manifest) if manifest.is_file() else None
    expected_hash = inputs.get('release_manifest_sha256')
    if release_hash != expected_hash:
        mismatches['release_manifest_sha256'] = {'expected': expected_hash, 'actual': release_hash}
    expected_release_names = {
        'mechanical_l1_native_probe_input.v1': 'parts_sorting',
        'mechanical_l2_native_probe_input.v1': 'peg_insertion',
        'mechanical_l3_native_probe_input.v1': 'shaft_spacer_candidate_v2',
        'mechanical_l4_loose_native_probe_input.v1': 'loose_bolt_candidate_v10_pgs',
        'mechanical_l5_gear_native_probe_input.v1': 'large_idler_v3_4_compact_pgs_release',
    }
    if is_l5:
        from scripts.validation.validate_mechanical_gear import _check_release
        try:
            _check_release(release)
        except ValueError as error:
            mismatches['release'] = str(error)
    elif release.name != expected_release_names[schema]:
        mismatches['release'] = f'{schema} privileged input is pinned to {expected_release_names[schema]}.'
    source_mismatches = {}
    for raw_path, expected in (inputs.get('source_sha256') or {}).items():
        path = Path(raw_path)
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            source_mismatches[str(path)] = {'expected': expected, 'actual': actual}
    if source_mismatches:
        mismatches['source_sha256'] = source_mismatches
    contract = inputs.get('robot_phase_contract')
    if (not isinstance(contract, dict) or contract.get('object_attachment') is not False
            or contract.get('object_state_writeback') is not False
            or contract.get('external_object_forces') is not False):
        mismatches['robot_phase_contract'] = 'Object attachment, pose writeback and external object forces must all be false.'
    case_ids = [case.get('id') for case in inputs.get('cases', []) if isinstance(case, dict)]
    if len(case_ids) != len(set(case_ids)) or not case_ids:
        mismatches['cases'] = 'Privileged native input requires unique nonempty cases.'
    task_value = inputs.get('task')
    if is_l4:
        if inputs.get('reset_variant') != 'loose_bolt':
            mismatches['reset_variant'] = {'expected': 'loose_bolt', 'actual': inputs.get('reset_variant')}
        if (not isinstance(task_value, dict)
                or task_value.get('reset_conditions', {}).get('reset_variant') != 'loose_bolt'):
            mismatches['task_reset_variant'] = 'L4 input must bind task.json loose_bolt reset conditions.'
    for case in inputs.get('cases', []):
        release_contract = case.get('release_contract')
        if release_contract is None:
            if is_l4 or is_l1 or is_l2 or is_l3 or is_l5:
                mismatches['release_contract:'+str(case.get('id'))] = 'L1/L2/L3/L4/L5 case must declare a no-shortcut reset contract.'
            continue
        if (release_contract.get('free_dynamic_objects') is not True
                or release_contract.get('gravity_only_after_reset') is not True
                or release_contract.get('object_pose_writeback_forbidden') is not True
                or ((is_l4 or is_l1 or is_l2 or is_l3 or is_l5)
                    and release_contract.get('external_object_forces_forbidden') is not True)):
            mismatches['release_contract:'+str(case.get('id'))] = 'Release probe must use free dynamics and forbid pose writeback/external object forces.'
        if is_l4 or is_l1 or is_l2 or is_l3 or is_l5:
            objects = case.get('reset_objects')
            if is_l4 and (not isinstance(objects, dict) or set(objects) != {'bolt', 'screwdriver'}):
                mismatches['reset_objects:'+str(case.get('id'))] = 'L4 reset must place exactly bolt and screwdriver on task-owned rests.'
            if is_l5 and (not isinstance(objects, dict) or set(objects) != {
                    'input_gear', 'idler_gear', 'output_gear',
                    'socket_crank', 'open_end_wrench', 'straight_hex_screwdriver'}):
                mismatches['reset_objects:'+str(case.get('id'))] = 'L5 reset must place exactly three gears and three tools on task-owned rests.'
            expected_objects = ((task_value or {}).get('evaluation') or {}).get('initial_objects')
            if (is_l4 or is_l5) and isinstance(expected_objects, dict) and objects != expected_objects:
                mismatches['reset_pose:'+str(case.get('id'))] = 'L4/L5 reset poses must match task-owned rests byte-for-byte.'
            if ((is_l1 or is_l2) and case.get('engineering_reset_override') is not True
                    and isinstance(expected_objects, dict) and objects != expected_objects):
                mismatches['reset_pose:'+str(case.get('id'))] = 'Nominal L1/L2 reset poses must match task-owned rests byte-for-byte.'
            phase_contract = case.get('phase_receipt_contract')
            if is_l3:
                actor_paths = phase_contract.get('contact_actor_paths') if isinstance(phase_contract, dict) else None
                if (not isinstance(actor_paths, dict) or not all(isinstance(path, str) for path in actor_paths.values())
                        or not {'/World/Objects/spacer', '/World/Objects/wide_ring',
                                '/World/Objects/top_ring', '/World/Fixtures/shaft_fixture'} <= set(actor_paths.values())):
                    mismatches['contact_actor_paths:'+str(case.get('id'))] = 'L3 receipts must include all three rings and the shaft fixture.'
            if ((is_l4 or is_l1 or is_l2 or is_l3 or is_l5) and (not isinstance(phase_contract, dict)
                    or phase_contract.get('record_raw_contacts') is not True
                    or phase_contract.get('task_success_credit') != 0)):
                mismatches['phase_receipt_contract:'+str(case.get('id'))] = 'L1/L2/L3/L4/L5 phases must record contacts with zero task credit.'
            try:
                checked_phases = _validate_robot_phases(case.get('robot_phases'), inputs['initial_state']['joint_positions_rad'])
                if any('timing' in phase for phase in checked_phases):
                    from scripts.validation.mechanical_robot_phase_timing import velocity_limits, validate_timing
                    expected_limits = velocity_limits(inputs['scene'])
                    previous = inputs['initial_state']['joint_positions_rad']
                    allowed_contacts = {'/World/Objects/'+row['id'] for row in inputs['scene']['objects']}
                    for phase in checked_phases:
                        if validate_timing(phase, previous, expected_limits=expected_limits) is None:
                            raise ValueError('Every phase must declare timing when any phase is timed')
                        if phase.get('hand_contact_actor_prefix') is not None and phase['hand_contact_actor_prefix'] not in allowed_contacts:
                            raise ValueError('Phase hand contact references an undeclared task object')
                        previous = phase['joint_positions_rad']
                    if sum(phase['duration_s'] for phase in checked_phases) > 120.:
                        raise ValueError('Timed diagnostic cases must fit the frozen 120-second budget')
                required_phases = phase_contract.get('required_phase_ids', []) if isinstance(phase_contract, dict) else []
                phase_ids = [phase['id'] for phase in checked_phases]
                if ((is_l4 or is_l1 or is_l2 or is_l3 or is_l5) and
                        (not isinstance(required_phases, list) or not required_phases
                        or any(name not in phase_ids for name in required_phases)
                        or len(required_phases) != len(set(required_phases)))):
                    mismatches['required_phases:'+str(case.get('id'))] = 'Required L1/L2/L3/L4/L5 phase IDs must be unique and present in the frozen phase list.'
                if type(case.get('duration_s')) not in (int, float) or abs(
                        float(case['duration_s'])-sum(row['duration_s'] for row in checked_phases)) > 1e-8:
                    mismatches['phase_duration:'+str(case.get('id'))] = 'L1/L2/L3/L4/L5 case duration must equal the frozen phase duration sum.'
            except (KeyError, TypeError, ValueError) as exc:
                mismatches['robot_phases:'+str(case.get('id'))] = str(exc)
    if is_l4 or is_l1 or is_l2 or is_l3 or is_l5:
        contract_required = ('object_attachment', 'object_state_writeback', 'external_object_forces')
        if not isinstance(contract, dict) or any(contract.get(key) is not False for key in contract_required):
            mismatches['robot_phase_contract'] = 'L1/L2/L3/L4/L5 object attachment, pose writeback and external object forces must all be false.'
    if mismatches:
        raise ValueError(f'Frozen {label} privileged input contract failed: '+json.dumps(mismatches, sort_keys=True))
    return {'passed': True, 'checked': True, 'release_manifest_sha256': release_hash,
            'source_files_checked': len(inputs.get('source_sha256') or {}),
            'scope': (f'{label} release, source, reset and no-shortcut contract checked before native runtime.')}


def _run_robot_phases(env, observer, phases, initial, *, output_stream=None):
    """Execute a frozen robot-only phase list using native integration."""
    phases = _validate_robot_phases(phases, initial)
    if not phases:
        return {'passed': True, 'declared': False, 'completed': [], 'native_steps': 0,
                'object_state_writeback_after_reset': False}
    previous = deepcopy(initial)
    completed, samples, native_steps = [], [], 0
    endpoints = []
    failed_phase = None
    for phase in phases:
        observer.phase = phase['id']
        target = phase['joint_positions_rad']
        count = round(phase['duration_s']/.02)
        timing = phase.get('timing')
        motion_count = round(timing['motion_duration_s']/.02) if timing else count
        offsets = {name: target[name]-previous[name] for name in previous}
        for index in range(count):
            fraction = min((index+1)/motion_count, 1.)
            command_targets = smooth_targets(previous, offsets, fraction)
            before_step = int(env.world.current_time_step_index)
            observed = env.step({'joint_positions_rad': command_targets, 'duration_s': .02})
            if observed.get('terminated') or not observed.get('metrics', {}).get('state_finite', False):
                raise RuntimeError('Robot phase terminated or became nonfinite: '+phase['id'])
            native_steps += int(env.world.current_time_step_index)-before_step
            samples.append({'phase': phase['id'], 'step': index+1,
                            'commanded_joint_positions_rad': deepcopy(command_targets),
                            'observed_joint_positions_rad': deepcopy(observed['joint_positions_rad'])})
            if output_stream is not None:
                output_stream.write(json.dumps(samples[-1], allow_nan=False, separators=(',', ':'))+'\n')
                output_stream.flush()
        actual = samples[-1]['observed_joint_positions_rad']
        if timing:
            from scripts.validation.mechanical_phase_endpoint import assess_phase_endpoint
            robot_paths = getattr(env, 'scene_info', {}).get('robot_link_paths')
            receipt = assess_phase_endpoint(phase, actual, getattr(observer, 'rows', []), env.dt,
                native_step=int(env.world.current_time_step_index),
                tolerance_rad=timing['endpoint_tolerance_rad'],
                robot_actor_paths=robot_paths.values() if robot_paths is not None else None)
        else:
            error = max(abs(actual[name]-target[name]) for name in target)
            receipt = {'passed': error <= .02, 'raw_max_error_rad': error,
                       'unwaived_max_error_rad': error, 'endpoint_tolerance_rad': .02}
        endpoints.append({**receipt, 'phase': phase['id'],
                          'maximum_joint_error_rad': receipt['raw_max_error_rad'],
                          'motion_duration_s': motion_count*.02,
                          'settle_duration_s': (count-motion_count)*.02})
        if timing and not receipt['passed']:
            failed_phase = phase['id']
            break
        previous = deepcopy(target)
        completed.append(phase['id'])
    endpoint_error = max(row['maximum_joint_error_rad'] for row in endpoints) if any('timing' in phase for phase in phases) else endpoints[-1]['maximum_joint_error_rad']
    timed = any('timing' in phase for phase in phases)
    arrival_passed = all(row['passed'] for row in endpoints) if timed else endpoint_error <= .02
    return {'passed': failed_phase is None and completed == [phase['id'] for phase in phases] and arrival_passed,
            'declared': True, 'completed': completed, 'declared_phase_count': len(phases),
            'native_steps': native_steps, 'control_samples': len(samples),
            'endpoint_max_joint_error_rad': endpoint_error,
            'endpoint_max_unwaived_joint_error_rad': max(row['unwaived_max_error_rad'] for row in endpoints),
            'phase_endpoints': endpoints, 'failed_phase': failed_phase,
            'object_state_writeback_after_reset': False,
            'scope': 'Privileged robot joint targets and native integration only; no object attachment, pose writeback or task score.'}


def _phase_receipts(rows, phases, contract):
    """Summarize native contact observations for every frozen robot phase."""
    if not isinstance(contract, dict):
        raise ValueError('Phase receipt contract must be a mapping')
    required = list(contract.get('required_phase_ids') or [phase['id'] for phase in phases])
    paths = contract.get('contact_actor_paths') or {}
    actors = {value for value in paths.values() if isinstance(value, str)}
    receipts = []
    for phase in phases:
        phase_id = phase['id']
        selected = [row for row in rows if row.get('phase') == phase_id]
        pairs = {}
        penetration = 0.
        for row in selected:
            penetration = max(penetration, float(row.get('maximum_penetration_m', 0.)))
            for contact in row.get('raw_contacts', []):
                raw_pair = (contact.get('actor0'), contact.get('actor1'))
                if not all(isinstance(value, str) for value in raw_pair):
                    raise ValueError('Native phase contact actor paths must be strings')
                pair = tuple(sorted(raw_pair))
                if any(value == actor or value.startswith(actor+'/')
                       for value in pair for actor in actors):
                    record = pairs.setdefault('/'.join(pair), {'actors': list(pair), 'records': 0,
                                                                 'positive_substeps': 0})
                    record['records'] += 1
                    record['positive_substeps'] += int(float(contact.get('impulse_magnitude_sum_ns', 0.)) > 1e-8)
        receipts.append({
            'phase': phase_id,
            'native_samples': len(selected),
            'first_native_step': selected[0]['native_step'] if selected else None,
            'last_native_step': selected[-1]['native_step'] if selected else None,
            'first_time_s': selected[0]['time_s'] if selected else None,
            'last_time_s': selected[-1]['time_s'] if selected else None,
            'contact_pairs': list(pairs.values()),
            'maximum_penetration_m': penetration,
            'robot_interference_samples': sum(bool(row.get('robot_interference')) for row in selected),
            'scope': 'Read-only native contact and robot phase receipt; no task score.',
        })
    minimum_raw = contract.get('minimum_native_samples_per_phase', 1)
    if type(minimum_raw) is not int or minimum_raw < 1:
        raise ValueError('Minimum native phase samples must be a positive integer')
    minimum = minimum_raw
    phase_ok = all(row['native_samples'] >= minimum for row in receipts)
    required_ok = all(any(row['phase'] == name and row['native_samples'] >= minimum for row in receipts)
                      for name in required)
    return {'passed': bool(phase_ok and required_ok), 'phase_count': len(receipts),
            'required_phase_ids': required, 'minimum_native_samples_per_phase': minimum,
            'receipts': receipts, 'task_success_credit': 0,
            'object_state_writeback_after_reset': False,
            'external_object_forces': False,
            'scope': 'Completeness of native phase/contact receipts only; no manipulation or task acceptance.'}


def _assess_contact_case(rows, case, scene_info, dt, *, schema):
    result = assess_case(rows, case, scene_info, dt)
    if schema == 'mechanical_l3_native_probe_input.v1':
        receipts = _phase_receipts(rows, case['robot_phases'], case['phase_receipt_contract'])
        result['phase_receipts'] = receipts
        result['checks']['phase_receipts'] = receipts['passed']
        result['passed'] &= receipts['passed']
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = observer = None
    report = {'schema': 'native_mechanical_contact_probes.v1', 'passed': False,
              'task_success_claimed': False, 'physics_accepted': False,
              'ordinary_agent_validation': False, 'model_calls': 0, 'cases': {},
              'object_state_writeback_after_reset': False, 'external_object_forces': False,
              'rgbd_capture_failures': []}
    try:
        checked = verify_if_snapshot(root, required_modules=('scripts.validation.validate_assembly_contacts',), entrypoint_file=Path(__file__))
        if checked is None:
            raise ValueError('A complete sealed source snapshot is required')
        dump(output/'runtime_imports.json', checked)
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select a single GPU by its UUID')
        if digest(args.input) != args.input_sha256:
            raise ValueError('Frozen probe inputs changed')
        inputs = load_json(args.input)
        report['native_input_contract'] = _validate_privileged_input_contract(inputs)
        dump(output/'inputs.json', inputs)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        from scripts.validation.mechanical_probe_solver import diagnostic_solver_env
        ProbeEnv = diagnostic_solver_env(FR3DexterousIsaacEnv, inputs.get('diagnostic_solver_type'), report)
        env = ProbeEnv(inputs['scene'], device='cuda:0', render_width=640, render_height=480)
        initial = inputs['initial_state']
        env.reset(initial_state=initial)
        schema = inputs.get('schema')
        l4_mode = schema == 'mechanical_l4_loose_native_probe_input.v1'
        l5_mode = schema == 'mechanical_l5_gear_native_probe_input.v1'
        phase_only_mode = schema in ('mechanical_l1_native_probe_input.v1',
                                     'mechanical_l2_native_probe_input.v1',
                                     'mechanical_l4_loose_native_probe_input.v1',
                                     'mechanical_l5_gear_native_probe_input.v1')
        sdf_sources = inputs.get('native_sdf_source_meshes', {})
        convex_env = parameter_env = env
        sdf_checks = {}
        if sdf_sources:
            from scripts.validation.mechanical_ring_sdf import check_ring_sdf
            from scripts.validation.validate_mechanical_ring_sdf import ReadOnlySelection, solver_selection, impulse_comparison
            convex_env, parameter_env = ReadOnlySelection(env), ReadOnlySelection(env)
            for name, mesh in sdf_sources.items():
                del convex_env.scene_info['objects'][name]
                directory = output/('sdf_'+name)
                directory.mkdir()
                sdf_checks[name] = check_ring_sdf(env,name,mesh,directory,
                    subgrid_resolution=inputs.get('native_sdf_subgrid_resolution', 6))
            for spec in parameter_env.scene_spec['objects']:
                if 'usd_physics_material_fallback' in spec:
                    spec['friction'] = spec['usd_physics_material_fallback']['static_friction']
        l4_geometry = None
        if l4_mode:
            from scripts.validation.validate_mechanical_dynamics import native_task_parameters
            parameter_check = native_task_parameters(parameter_env)
            from scripts.validation.mechanical_native_geometry import collect_task_geometry
            l4_geometry = collect_task_geometry(env, output)
        elif l5_mode:
            from scripts.validation.validate_gear_transmission import mixed_native_parameters
            parameter_check = mixed_native_parameters(parameter_env)
        else:
            parameter_check = native_parameters(parameter_env)
        report['checks'] = {'stage_inventory': stage_inventory(env),
                            'contact_filters': contact_filter_coverage(env),
                            'native_parameters': parameter_check}
        if l4_mode:
            report['checks']['native_geometry'] = {
                'passed': l4_geometry['passed'], 'issues': l4_geometry['issues'],
                'scope': l4_geometry['scope']}
        else:
            report['checks']['native_convex_geometry'] = check_convex_geometry(convex_env)
        if sdf_sources:
            report['checks']['native_sdf_geometry'] = {'passed':all(r['passed'] for r in sdf_checks.values()),'objects':sdf_checks}
            report['checks']['scene_solver_selection'] = solver_selection(env.metadata(),inputs['scene']['isaacsim'].get('solver_type','TGS'))
        coverage = inputs.get('ring_coverage_candidate')
        if coverage:
            from scripts.validation import mechanical_ring_coverage
            from scripts.validation.mechanical_iteration_readback import SleepObserver, stage_iteration_settings
            mechanical_ring_coverage.setup(env, inputs, report)
            report['ring_coverage_candidate'] = coverage
        dump(output/'metadata.json', env.metadata())
        if not all(r['passed'] for r in report['checks'].values()):
            raise RuntimeError('Native scene parameter/structure check failed')
        for case in inputs['cases']:
            if (not case['id'].isidentifier()
                    or (not phase_only_mode and set(case['assertions']) != set(env.object_names))
                    or (phase_only_mode and set(case.get('reset_objects', {})) != set(env.object_names))
                    or not .5 < case.get('duration_s', 1.5) <= (
                        120. if any('timing' in phase for phase in case.get('robot_phases', [])) else 10.)
                    or abs(case.get('duration_s', 1.5)/.02-round(case.get('duration_s', 1.5)/.02)) > 1e-8):
                raise ValueError('A probe requires a safe ID, all task objects and a finite 20 ms duration')
            case_root = output/case['id']
            case_root.mkdir()
            reset = {**deepcopy(initial), 'objects': case['reset_objects']}
            env.reset(initial_state=reset)
            if phase_only_mode:
                dump(case_root/'reset_receipt.json', {
                    'schema': f'mechanical_{"l4_loose" if l4_mode else "l5_gear" if schema.endswith("l5_gear_native_probe_input.v1") else "l1" if schema.endswith("l1_native_probe_input.v1") else "l2"}_reset_receipt.v1',
                    'reset_variant': case.get('reset_variant'),
                    'objects': deepcopy(case['reset_objects']),
                    'applied_by': 'native_env.reset(initial_state=...)',
                    'object_state_writeback_after_reset': False,
                    'external_object_forces': False,
                    'task_success_credit': 0,
                    'scope': 'Reset receipt only; native object pose and contact evidence are recorded in native_frames.jsonl.',
                })
            reset_check = mechanical_ring_coverage.reset_receipt(env, case) if coverage else None
            if coverage:
                dump(case_root/'reset_receipt.json', reset_check)
                if not reset_check['passed']:
                    raise RuntimeError('Original native reset did not match the frozen case')
            observer = SleepObserver(env, case_root) if coverage else Observer(env, case_root)
            first_index = int(env.world.current_time_step_index)
            phase_stream = (case_root/'robot_phases.jsonl').open('x')
            try:
                phase_report = _run_robot_phases(
                    env, observer, case.get('robot_phases'), initial['joint_positions_rad'],
                    output_stream=phase_stream)
                if not phase_report['declared']:
                    observer.phase = 'hold_ready'
                    count_requested = round(case.get('duration_s', 1.5)/.02)
                    for i in range(count_requested):
                        observed = env.step({'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02})
                        if observed['terminated'] or not observed['metrics']['state_finite']:
                            raise RuntimeError('Task probe terminated or became nonfinite')
                        if i % 25 == 0:
                            print(json.dumps({'case': case['id'], 'sim_time_s': observed['sim_time_s']}), flush=True)
                else:
                    expected_duration = sum(float(row['duration_s']) for row in case['robot_phases'])
                    if not np.isclose(expected_duration, float(case.get('duration_s', 1.5)), atol=1e-8, rtol=0.):
                        raise ValueError('case duration must equal the declared robot phase duration sum')
                    phase_report['planned_duration_s'] = expected_duration
                    phase_report['duration_s'] = phase_report['native_steps']*env.dt
            finally:
                phase_stream.close()
            count = int(env.world.current_time_step_index)-first_index
            rows = observer.rows
            if coverage:
                sleep = observer.sleep_rows
                sleep_check = {'passed': len(sleep) == len(rows) and all(
                    a['native_step'] == b['native_step'] and a['time_s'] == b['time_s']
                    for a, b in zip(sleep, rows, strict=True)), 'samples': len(sleep),
                    'sleeping_samples': {name: sum(r['sleeping'][name] for r in sleep) for name in env.object_names},
                    'sleep_state_write_performed': False}
            exchange = observer.close()
            observer = None
            if phase_only_mode:
                receipts = _phase_receipts(rows, case['robot_phases'], case['phase_receipt_contract'])
                result = {
                    'passed': bool(receipts['passed']),
                    'checks': {'phase_receipts': receipts['passed'],
                               'no_object_pose_writeback': receipts['object_state_writeback_after_reset'] is False,
                               'no_external_object_forces': receipts['external_object_forces'] is False},
                    'phase_receipts': receipts,
                    'functional_contact_success': False,
                    'task_success_claimed': False,
                    'scope': f'{"L4 loose-bolt" if l4_mode else "L5 gear" if schema.startswith("mechanical_l5") else "L1 sorting" if schema.startswith("mechanical_l1") else "L2 peg"} native phase/contact diagnostic; no manipulation or task score.',
                }
            else:
                result = _assess_contact_case(rows, case, env.scene_info, env.dt, schema=schema)
            result['robot_phases'] = phase_report
            result['checks']['robot_phases'] = phase_report['passed']
            result['passed'] &= phase_report['passed']
            result['contact_exchange'] = contact_exchange_check(exchange, count, len(rows))
            result['passed'] &= result['contact_exchange']['passed']
            result['physics_substeps'] = count
            if inputs.get('quantitative_weight_balance_required'):
                if not sdf_sources or inputs['scene'].get('engineering_contact_force_crosscheck') is not True:
                    raise ValueError('Weight qualification requires complete three-API impulse evidence')
                masses = {n:p['mass_kg'] for n,p in report['checks']['native_parameters']['objects'].items()}
                result['weight_balance'] = impulse_comparison(rows,env.scene_info,masses,env.dt)
                result['passed'] &= result['weight_balance']['quantitative_weight_balance_passed']
            if coverage:
                result['native_reset'] = reset_check
                result['native_sleep_observation'] = sleep_check
                result['checks']['native_reset'] = reset_check['passed']
                result['checks']['native_sleep_observation'] = sleep_check['passed']
                result['checks']['stage_iteration_settings_unchanged'] = (
                    stage_iteration_settings(env) == report['initial_stage_iteration_settings'])
                masses = {n:p['mass_kg'] for n,p in report['checks']['native_parameters']['objects'].items()}
                result['physical_coverage'] = mechanical_ring_coverage.result_evidence(env, rows, case, masses)
                if 'actual_support_chain' in result['physical_coverage']:
                    result['checks']['actual_support_chain_weight'] = result['physical_coverage']['actual_support_chain']['quantitative_weight_balance_passed']
                object_paths = {v['path'] for v in env.scene_info['objects'].values()}
                robot_paths = set(env.scene_info['robot_link_paths'].values())
                result['checks']['released_no_robot_manifold'] = not any(
                    object_paths.intersection((c['actor0'], c['actor1'])) and
                    robot_paths.intersection((c['actor0'], c['actor1']))
                    for row in rows[-round(.5/env.dt):] for c in row['raw_contacts'])
                result['passed'] &= all(result['checks'].values())
            report['cases'][case['id']] = result
            dump(case_root/'result.json', result)
            try:
                result['rgbd'] = capture(env, case_root, 'final')
            except BaseException as exc:
                result['rgbd'] = {'passed': False, 'error': {
                    'type': type(exc).__name__, 'detail': str(exc)}}
            result['passed'] &= result['rgbd']['passed']
            report['cases'][case['id']] = result
            dump(case_root/'result.json', result)
            if not result['rgbd']['passed']:
                report['rgbd_capture_failures'].append({
                    'case': case['id'], 'error': result['rgbd'].get('error')})
        metadata = env.metadata()
        dump(output/'metadata_final.json', metadata)
        report['fixed_support_guard'] = metadata['static_mesh_contact_runtime']
        if coverage:
            report['final_stage_iteration_settings'] = stage_iteration_settings(env)
        report['passed'] = all(r['passed'] for r in report['cases'].values()) and bool(report['cases'])
        report['passed'] &= (report['fixed_support_guard']['maximum_translation_m'] <= 1e-6 and
                              report['fixed_support_guard']['maximum_angle_rad'] <= 1e-5)
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root,
            required_modules=('scripts.validation.validate_assembly_contacts',), entrypoint_file=Path(__file__)))
    except BaseException as exc:
        report.update(passed=False, error={'type': type(exc).__name__, 'detail': str(exc),
                                          'traceback': traceback.format_exc()})
    finally:
        if observer is not None:
            try:
                report['interrupted_exchange'] = observer.close()
            except BaseException as exc:
                report['close_error'] = str(exc)
    report['artifacts_sha256'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args)['passed'] else 1)


if __name__ == '__main__':
    main()
