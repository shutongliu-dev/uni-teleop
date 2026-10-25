"""Frozen L5 supply/contact-load evidence, with all gears passive and free."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import dump, digest
from bench.paths import load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, inertia_principal_axes
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import build_task_inputs, contact_probe_reset
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.gear_transmission_checks import (GEARS, assess_transmission,
    positive_contact, probe_contract, transmission_geometry_contract)
from scripts.validation.validate_assembly_contacts import Observer, native_parameters
from scripts.validation.validate_mechanical_dynamics import native_task_parameters
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_convex_geometry import check_convex_geometry
from scripts.validation.mechanical_tool_checks import contact_exchange_check
from scripts.validation.validate_workrooms import plain, stage_inventory, capture
from scripts.validation.validate_scenario import seal_report_before_shutdown


def make_probe_inputs(*, contract_preset='standard', **task_paths):
    inputs = build_task_inputs(**task_paths)
    inputs.update(probe_contract=probe_contract(contract_preset), supply_duration_s=1.5,
                  contact_reset=contact_probe_reset(), task_success_claimed=False,
                  transmission_geometry_contract=transmission_geometry_contract(
                      geometry=inputs['task']['evaluation']))
    return inputs


def validate_probe_inputs(inputs):
    if 'diagnostic_solver_type' in inputs:
        raise ValueError('Use the frozen scene.isaacsim.solver_type configuration')
    if (inputs['probe_contract'] not in [probe_contract(p) for p in ('standard', 'extended_observation', 'high_torque')]
            or inputs['supply_duration_s'] != 1.5):
        raise ValueError('Probe contract differs from frozen source thresholds')
    if 'transmission_geometry_contract' in inputs:
        expected = transmission_geometry_contract(geometry=inputs.get('task', {}).get('evaluation'))
        if inputs['transmission_geometry_contract'] != expected:
            raise ValueError('Transmission geometry contract differs from frozen task evaluation')
    return inputs['probe_contract']


class _PlainEvidenceEnv:
    """Read-only delegate with a JSON/OBJ-only inventory for that validator."""
    def __init__(self, env):
        self.env = env
        self.scene_spec = deepcopy(env.scene_spec)
        self.scene_spec['objects'] = [s for s in self.scene_spec['objects']
            if env.scene_info['objects'][s['id']].get('backend') != 'usd_reference']
        self.scene_info = deepcopy(env.scene_info)
        self.scene_info['objects'] = {s['id']: self.scene_info['objects'][s['id']]
                                      for s in self.scene_spec['objects']}

    def __getattr__(self, key):
        return getattr(self.env, key)


def mixed_native_parameters(env):
    from bench.simulation.isaacsim_backend.env import _np
    from pxr import PhysxSchema
    ordinary = native_parameters(_PlainEvidenceEnv(env))
    name = 'straight_hex_screwdriver'
    referenced = native_task_parameters(env, required_roles={name: 'objects'})
    spec = next(s for s in env.scene_spec['objects'] if s['id'] == name)
    asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
    view = env._object_views[name]
    if view.count != 1:
        raise ValueError('Missing unique original screwdriver')
    inertia = _np(view.get_inertias()).reshape(3, 3).T
    com = _np(view.get_coms()).reshape(7)
    rotation = quaternion_matrix(inertia_principal_axes(asset))
    expected = rotation @ np.diag(asset['inertia_diag_kg_m2']) @ rotation.T
    info = env.scene_info['objects'][name]
    authored = info['usd_physics_parameters']['collider_parameters']
    offsets = {}
    for path in info['collision_paths']:
        api = PhysxSchema.PhysxCollisionAPI(env.stage.GetPrimAtPath(path))
        actual = [float(api.GetContactOffsetAttr().Get()), float(api.GetRestOffsetAttr().Get())]
        wanted = [authored[path][key]['value'] for key in ('physxCollision:contactOffset', 'physxCollision:restOffset')]
        offsets[path] = {'actual_m': actual, 'expected_m': wanted,
                         'passed': bool(np.allclose(actual, wanted, atol=1.e-8, rtol=0.))}
    extra = {'inertia_body_frame': bool(np.allclose(inertia, expected, atol=1.e-9, rtol=2.e-5)),
             'com': bool(np.allclose(com[:3], asset['center_of_mass_m'], atol=1.e-7, rtol=0.)),
             'source_collider_offsets': bool(offsets) and all(r['passed'] for r in offsets.values())}
    return {'passed': ordinary['passed'] and referenced['passed'] and all(extra.values()),
            'plain_assets': ordinary, 'source_usd_assets': referenced,
            'source_usd_inertia_com_offsets': {'checks': extra, 'native_com_principal_pose_xyzw': com.tolist(),
                'native_inertia_body_frame_kg_m2': inertia.tolist(), 'expected_inertia_body_frame_kg_m2': expected.tolist(),
                'offsets': offsets, 'offset_scope': 'Composed USD readback; native contact participation is checked separately.'}}


def assess_supply(rows, scene, scene_info, dt, *, names=None):
    names = names or [s['id'] for s in scene['objects']]
    window = rows[-round(.5/dt):]
    checks = {'complete_native_sampling': len(window) == round(.5/dt) and all(
        b['native_step']-a['native_step'] == 1 and abs(b['time_s']-a['time_s']-dt) < 1.e-8
        for a, b in zip(rows, rows[1:])),
        'no_robot_interference': not any(r['robot_interference'] for r in rows),
        'penetration_bound': bool(rows) and max(r['maximum_penetration_m'] for r in rows) <= .0005}
    bodies = {}
    supports = scene_info.get('workbench', {}).get('supports', {})
    table_support = supports.get('table') if isinstance(supports, dict) else None
    declared_supports = scene_info.get('supply_supports', {})
    if table_support is None and isinstance(declared_supports, dict):
        table_support = declared_supports.get('table')
    if table_support is None:
        support_paths = scene_info.get('support_paths', [])
        if isinstance(support_paths, dict):
            table_support = support_paths.get('table')
        elif isinstance(support_paths, (list, tuple)):
            table_support = next((p for p in support_paths
                                  if str(p).rstrip('/').lower().endswith('/table')), None)
    for name in names:
        spec = next(s for s in scene['objects'] if s['id'] == name)
        fixture_specs = scene_info.get('fixtures', {})
        support = declared_supports.get(name) if isinstance(declared_supports, dict) else None
        if isinstance(support, dict):
            support_path, support_kind = support.get('path'), support.get('kind', 'declared')
        elif isinstance(support, str):
            support_path, support_kind = support, 'declared'
        else:
            support_path, support_kind = None, None
        if support_path is None:
            for candidate in (f'{name}_supply_rest', f'{name}_rest'):
                if candidate in fixture_specs:
                    support_path, support_kind = fixture_specs[candidate]['path'], 'fixture'
                    break
        if support_path is None:
            support_path, support_kind = table_support, 'table'
        if support_path is None:
            raise ValueError(f'No declared supply support for {name}')
        pose = np.array([r['poses_xyzw'][name] for r in window])
        velocity = np.array([r['velocities'][name] for r in window])
        axis = quaternion_matrix(spec['quaternion_wxyz'])[:, 2]
        axis_error = max(math.acos(float(np.clip(np.dot(axis, quaternion_matrix([p[6], *p[3:6]])[:, 2]), -1., 1.))) for p in pose)
        support = [positive_contact(r, scene_info['objects'][name]['path'],
                    support_path, upward=True, second_is_prefix=True) for r in window]
        block = round(.02/dt)
        supported = [any(support[i:i+block]) for i in range(0, len(support), block)]
        error = np.max(np.abs(pose[:, :3]-spec['position_m']), axis=0)
        body_checks = {'near_supply_pose': bool(np.all(error <= [.003, .003, .001])),
            'design_axis_preserved': axis_error <= math.radians(2.),
            'linear_rest': bool(np.linalg.norm(velocity[:, :3], axis=1).max() <= .005),
            'angular_rest': bool(np.linalg.norm(velocity[:, 3:], axis=1).max() <= .05),
            'stable_position': bool(np.linalg.norm(pose[:, :3]-pose[0, :3], axis=1).max() <= .0002),
            'upward_support_each_20ms': bool(supported) and all(supported)}
        bodies[name] = {'passed': all(body_checks.values()), 'checks': body_checks,
            'maximum_position_error_m': error.tolist(), 'maximum_design_axis_error_rad': axis_error,
            'final_pose_xyzw': pose[-1].tolist(), 'support_windows': sum(supported),
            'support_kind': support_kind,
            'support_path': support_path}
    checks['all_declared_supply_objects'] = bool(bodies) and all(r['passed'] for r in bodies.values())
    return {'passed': all(checks.values()), 'checks': checks, 'objects': bodies, 'task_success_claimed': False}


class LoadObserver(Observer):
    def __init__(self, env, output, contract):
        self.contract = contract
        self.phase = contract['phases'][0]
        self.wrench_stream = (output/'applied_wrenches.jsonl').open('x')
        super().__init__(env, output)

    def step(self, *args, **kwargs):
        from bench.simulation.isaacsim_backend.env import _np
        env, c = self.env, self.contract
        view = env._object_views['socket_crank']
        velocity = _np(view.get_velocities()).reshape(6)
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity[3:]) > c['maximum_tool_speed_rad_s']:
            raise RuntimeError('Tool speed safety bound or finite-state check failed')
        torque_z = (float(np.clip(c['speed_load_coefficient_nm_s_per_rad']*(self.phase['target_speed_rad_s']-velocity[5]),
                          -c['maximum_torque_nm'], c['maximum_torque_nm'])) if self.phase['loaded'] else 0.)
        force, torque = [0., 0., 0.], [0., 0., torque_z]
        # Pure world-frame torque on this one free tool, no pose targets or gear force.
        view.apply_forces_and_torques_at_position(env._tensor(np.array([force], dtype=np.float32)),
            env._tensor(np.array([torque], dtype=np.float32)), None, env._one, True)
        result = super().step(*args, **kwargs)
        row = self.rows[-1]
        receipt = {'native_step': row['native_step'], 'time_s': row['time_s'],
            'phase': self.phase['id'], 'body': 'socket_crank', 'world_frame': True,
            'force_n': force, 'torque_nm': torque, 'prestep_tool_velocity': velocity.tolist(),
            'target_speed_rad_s': self.phase['target_speed_rad_s'], 'finite_state_after_step': True}
        row['wrench'] = receipt
        self.wrench_stream.write(json.dumps(receipt, separators=(',', ':'), allow_nan=False)+'\n')
        if any(np.linalg.norm(v[:3]) > c['abort_body_linear_speed_m_s']
                or np.linalg.norm(v[3:]) > c['abort_body_angular_speed_rad_s'] for v in row['velocities'].values()):
            self.stream.flush()
            self.wrench_stream.flush()
            raise RuntimeError('Native body motion exceeded the frozen diagnostic safety limit; offending frame retained')
        return result

    def close(self):
        try:
            return super().close()
        finally:
            self.wrench_stream.close()


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = observer = None
    report = {'schema': 'native_gear_contact_load.v1', 'passed': False, 'task_success_claimed': False,
              'model_calls': 0, 'object_state_writeback_after_reset': False,
              'full_scene_physics_accepted': False, 'source_usd_tool_cooked_geometry_checked': False,
              'quantitative_contact_force_qualified': False,
              'contact_evidence_scope': 'Positive normal-contact chain only. Compound summed impulses are not a calibrated body reaction or torque measurement.',
              'actuation_scope': 'Bounded world torque on independent socket_crank only; all gears passive. Robot remains at ready joint targets.'}
    try:
        imported = verify_if_snapshot(root, required_modules=('scripts.validation.validate_gear_transmission',), entrypoint_file=Path(__file__))
        if imported is None:
            raise ValueError('A complete sealed source snapshot is required')
        dump(output/'runtime_imports.json', imported)
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select a single GPU by UUID')
        if digest(args.input) != args.input_sha256:
            raise ValueError('Frozen gear inputs changed')
        inputs = load_json(args.input)
        contract = validate_probe_inputs(inputs)
        provenance = inputs['scene']['mechanical_task_provenance']
        if digest(Path(provenance['asset_manifest_path'])) != provenance['asset_manifest_sha256']:
            raise ValueError('L5 asset manifest changed after input freeze')
        for spec in inputs['scene']['objects']:
            if spec['id'] == 'straight_hex_screwdriver' and digest(Path(spec['asset_root'])/'manifest.json') != provenance['screwdriver_manifest_sha256']:
                raise ValueError('Original screwdriver catalog changed after input freeze')
        dump(output/'inputs.json', inputs)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(inputs['scene'], device='cuda:0', render_width=640, render_height=480)
        initial = inputs['initial_state']
        env.reset(initial_state=initial)
        report['checks'] = {'stage_inventory': stage_inventory(env), 'contact_filters': contact_filter_coverage(env),
                            'native_parameters': mixed_native_parameters(env),
                            'native_all_convex_geometry': check_convex_geometry(env)}
        report['source_usd_tool_cooked_geometry_checked'] = bool(
            report['checks']['native_all_convex_geometry']['passed'] and
            any(r['body'] == 'straight_hex_screwdriver' for r in
                report['checks']['native_all_convex_geometry']['colliders'].values()))
        metadata = env.metadata()
        report['physics_solver'] = metadata['physics_solver']
        expected_solver = inputs['scene']['isaacsim'].get('solver_type', 'TGS')
        report['checks']['scene_solver_selection'] = {
            'passed': (metadata['physics_solver']['profile_selection'] == expected_solver
                       and metadata['physics_solver']['usd_readback'] == expected_solver),
            'expected': expected_solver, **metadata['physics_solver']}
        dump(output/'metadata.json', metadata)
        dump(output/'parameter_checks.json', report['checks'])
        if not all(r['passed'] for r in report['checks'].values()):
            raise RuntimeError('Native body/parameter/structure evidence failed')
        supply_root = output/'nominal_supply'
        supply_root.mkdir()
        observer = Observer(env, supply_root)
        first_index = int(env.world.current_time_step_index)
        for _ in range(round(inputs['supply_duration_s']/.02)):
            observed = env.step({'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02})
            if observed['terminated'] or not observed['metrics']['state_finite']:
                raise RuntimeError('Native supply state failed')
        rows = observer.rows
        exchange = observer.close()
        observer = None
        supply = assess_supply(rows, inputs['scene'], env.scene_info, env.dt)
        supply['contact_exchange'] = contact_exchange_check(exchange, int(env.world.current_time_step_index)-first_index, len(rows))
        supply['rgbd'] = capture(env, supply_root, 'final')
        supply['passed'] &= supply['contact_exchange']['passed'] and supply['rgbd']['passed']
        report['nominal_supply'] = supply
        dump(supply_root/'result.json', supply)
        del rows
        if not supply['passed']:
            raise RuntimeError('Nominal supply support/clearance failed before load test')
        load_root = output/'contact_load'
        load_root.mkdir()
        reset = {**deepcopy(initial), 'objects': deepcopy(inputs['contact_reset']['object_overrides'])}
        env.reset(initial_state=reset)
        observer = LoadObserver(env, load_root, inputs['probe_contract'])
        first_index = int(env.world.current_time_step_index)
        with (load_root/'engine_windows.jsonl').open('x') as engine:
            for phase in contract['phases']:
                observer.phase = phase
                for i in range(round(phase['duration_s']/.02)):
                    observed = env.step({'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02})
                    engine.write(json.dumps(plain({'phase': phase['id'], 'sim_time_s': observed['sim_time_s'],
                        'native_step': int(env.world.current_time_step_index), 'terminated': observed['terminated'],
                        'state_finite': observed['metrics']['state_finite']}))+'\n')
                    if observed['terminated'] or not observed['metrics']['state_finite']:
                        raise RuntimeError('Native loaded state failed')
                    if i % 25 == 0:
                        print(json.dumps({'phase': phase['id'], 'sim_time_s': observed['sim_time_s']}), flush=True)
        rows = observer.rows
        exchange = observer.close()
        observer = None
        load = assess_transmission(rows, env.scene_info, env.dt, contract,
                                   geometry=inputs['transmission_geometry_contract'])
        load['contact_exchange'] = contact_exchange_check(exchange, int(env.world.current_time_step_index)-first_index, len(rows))
        load['other_tools'] = assess_supply(rows, inputs['scene'], env.scene_info, env.dt,
                                            names=['open_end_wrench', 'straight_hex_screwdriver'])
        load['rgbd'] = capture(env, load_root, 'final')
        load['passed'] &= all(load[key]['passed'] for key in ('contact_exchange', 'other_tools', 'rgbd'))
        report['contact_load'] = load
        dump(load_root/'result.json', load)
        metadata = env.metadata()
        dump(output/'metadata_final.json', metadata)
        guard = metadata['static_mesh_contact_runtime']
        report['fixed_support_guard'] = guard
        report['passed'] = supply['passed'] and load['passed'] and guard['maximum_translation_m'] <= 1.e-6 and guard['maximum_angle_rad'] <= 1.e-5
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root,
            required_modules=('scripts.validation.validate_gear_transmission',), entrypoint_file=Path(__file__)))
    except BaseException as exc:
        report.update(passed=False, error={'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()})
    finally:
        if observer is not None:
            try:
                report['interrupted_contact_exchange'] = observer.close()
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
