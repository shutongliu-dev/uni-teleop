"""Read native base flags, constraints and support reactions while holding home.

Run from a complete frozen snapshot on one physical GPU UUID. No image capture,
constraint edits, mass edits or state setters after reset. A passing report only
means that diagnostic acquisition completed; it does not certify a fixed base.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import dump
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import check_window, plain, record


FAMILY = 'mechanical_assembly'
MOUNT_PATH = '/World/Room/Geometry/robot_mounting_plate/Collider'


def base_contacts(contacts, base_path, physics_dt_s):
    """Preserve native Ns and report average normal force over each substep."""
    if not np.isfinite(physics_dt_s) or physics_dt_s <= 0:
        raise ValueError('physics_dt_s must be finite and positive')
    result = []
    for row in contacts:
        if {row['actor0'], row['actor1']} != {base_path, MOUNT_PATH}:
            continue
        impulse = float(row['impulse_magnitude_sum_ns'])
        vector = np.asarray(row['impulse_vector_ns'], dtype=float)
        if not np.isfinite(impulse) or impulse < 0 or vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('Invalid native normal impulse')
        result.append({**row, 'physics_dt_s': physics_dt_s,
                       'average_normal_force_magnitude_sum_n': impulse / physics_dt_s,
                       'average_normal_force_vector_n': (vector / physics_dt_s).tolist()})
    return result


def native_static_readback(env):
    from bench.simulation.isaacsim_backend.env import _np
    view = env._robot_view
    shared, individual = view.shared_metatype, view.get_metatype(0)
    return {'shared_metatype_fixed_base': bool(shared.fixed_base),
            'get_metatype_0_fixed_base': bool(individual.fixed_base),
            'articulation_count': int(view.count), 'articulation_prim_paths': list(view.prim_paths),
            'native_link_names': list(individual.link_names),
            'native_link_parents': list(individual.link_parents),
            'native_dof_names': list(individual.dof_names),
            'native_robot_link_masses_kg': dict(zip(env.body_names, _np(view.get_masses())[0].tolist())),
            'configured_articulation_path': env.scene_info['articulation_path'],
            'base_path': env.scene_info['robot_link_paths']['fr3_link0'],
            'root_pose_xyzw': _np(view.get_root_transforms())[0].tolist(),
            'root_velocity_linear_angular': _np(view.get_root_velocities())[0].tolist()}


def stage_relationships(env):
    from pxr import UsdPhysics
    robot = env.scene_info['robot_path']
    roots, joints = [], []
    belongs = lambda path: path == robot or path.startswith(robot + '/')
    for prim in env.stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append({'path': str(prim.GetPath()), 'schemas': list(prim.GetAppliedSchemas()),
                          'enabled': prim.GetAttribute('physxArticulation:articulationEnabled').Get()})
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(p) for p in joint.GetBody0Rel().GetTargets()]
        body1 = [str(p) for p in joint.GetBody1Rel().GetTargets()]
        if belongs(str(prim.GetPath())) or any(belongs(p) for p in body0 + body1):
            joints.append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                           'enabled': joint.GetJointEnabledAttr().Get(), 'body0': body0, 'body1': body1,
                           'body0_is_world': not body0, 'body1_is_world': not body1,
                           'exclude_from_articulation': prim.GetAttribute('physics:excludeFromArticulation').Get()})
    return {'articulation_roots': roots, 'robot_related_joints': joints,
            'empty_body_relationship_semantics': 'An unset joint body relationship is the world attachment.'}


def run(args):
    root, output = Path(__file__).resolve().parents[2], args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema': 'workroom_base_probe_plan.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'family': FAMILY, 'control_steps': args.steps, 'control_dt_s': .02,
            'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
            'model_calls': 0, 'probe_sha256': digest(__file__),
            'scope': 'Read actual native fixed-base flags, root state, masses, joint graph and support reactions.',
            'force_semantics': 'Normal impulse Ns divided by actual physics dt is a substep-average N; not instantaneous force or a calibrated reaction.'}
    dump(output / 'plan.json', plan)
    env, error, steps = None, None, 0
    rows, states = [], []
    try:
        from scenes.workrooms import scene
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        modules = ('scripts.diagnostics.probe_workroom_base', 'scripts.validation.validate_workrooms',
                   'bench.simulation.isaacsim_backend.env', 'scenes.workrooms')
        verification = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('A complete frozen source snapshot is required')
        dump(output / 'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (args.device != 'cuda:0' or not uuid.startswith('GPU-') or ',' in uuid
                or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1'):
            raise ValueError('Select one physical GPU UUID in single CUDA mode with device cuda:0')
        if type(args.steps) is not int or not 1 <= args.steps <= 500:
            raise ValueError('steps must be an integer from 1 through 500')
        if args.scene_json:
            spec = load_json(args.scene_json)
            if spec.get('workroom', {}).get('family') != FAMILY:
                raise ValueError('Scene override must be mechanical_assembly')
            dump(output / 'candidate_scene_source.json', {'path': str(args.scene_json.resolve()),
                                                         'sha256': digest(args.scene_json)})
        else:
            spec = scene(FAMILY)
        dump(output / 'scene.json', spec)
        env = FR3DexterousIsaacEnv(spec, device=args.device)
        observed = env.reset(seed=0)
        # Seal the native flags immediately, before additional metadata work.
        flags = {'shared_metatype_fixed_base': bool(env._robot_view.shared_metatype.fixed_base),
                 'get_metatype_0_fixed_base': bool(env._robot_view.get_metatype(0).fixed_base)}
        dump(output / 'native_flags_after_reset.json', flags)
        print(json.dumps({'event': 'base_native_flags', **flags}), flush=True)
        initial_native = native_static_readback(env)
        dump(output / 'native_after_reset.json', initial_native)
        dump(output / 'stage_relationships.json', stage_relationships(env))
        dump(output / 'metadata_initial.json', env.metadata())
        home = env.action_spec()['initial_joint_positions_rad']
        first = record(env, observed, 0, 'reset')
        first['root_velocity_linear_angular'] = _np(env._robot_view.get_root_velocities())[0].tolist()
        dump(output / 'initial_state.json', first)
        previous = first['state']
        states.append(first)
        with (output / 'physics.jsonl').open('x') as stream:
            for i in range(args.steps):
                command = {'joint_positions_rad': home, 'duration_s': .02}
                observed = env.step(command)
                row = record(env, observed, 0, 'hold_home', command)
                row['root_velocity_linear_angular'] = _np(env._robot_view.get_root_velocities())[0].tolist()
                check_window(observed, row['state'], previous, env.dt)
                selected = base_contacts(observed['contacts']['rigid'], initial_native['base_path'], env.dt)
                row['base_mount_contacts'] = selected
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()
                rows.extend(selected)
                states.append(row)
                previous = row['state']
                steps += 1
                if steps % 25 == 0:
                    print(json.dumps({'event': 'base_probe_progress', 'steps': steps}), flush=True)
        dump(output / 'native_final.json', native_static_readback(env))
        dump(output / 'metadata_final.json', env.metadata())
        dump(output / 'runtime_imports_final.json', verify_if_snapshot(
            root, required_modules=modules, entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'base_probe_error', **error}), flush=True)
    root_states = np.asarray([r['root_pose_xyzw'] for r in states])
    root_velocities = np.asarray([r['root_velocity_linear_angular'] for r in states])
    summary = {'base_mount_contact_records': len(rows),
               'maximum_substep_normal_impulse_ns': max((r['impulse_magnitude_sum_ns'] for r in rows), default=0.),
               'maximum_substep_average_normal_force_n': max((r['average_normal_force_magnitude_sum_n'] for r in rows), default=0.),
               'total_normal_impulse_ns': sum(r['impulse_magnitude_sum_ns'] for r in rows),
               'minimum_separation_m': min((r['min_separation_m'] for r in rows), default=None),
               'maximum_root_translation_from_reset_m': float(np.linalg.norm(root_states[:, :3] - root_states[0, :3], axis=1).max()) if len(states) else None,
               'maximum_root_linear_speed_m_s': float(np.linalg.norm(root_velocities[:, :3], axis=1).max()) if len(states) else None,
               'maximum_root_angular_speed_rad_s': float(np.linalg.norm(root_velocities[:, 3:], axis=1).max()) if len(states) else None}
    report = {'schema': 'workroom_base_probe.v1', 'passed': error is None and steps == args.steps,
              'acceptance_semantics': 'Data acquisition completed only; no fixed-base correctness or force-realism claim.',
              'base_correctness_qualified': False, 'error': error, 'control_steps': steps,
              'model_calls': 0, 'task_qualification': False, 'state_writes_after_reset': False,
              'summary': summary, 'plan_sha256': digest(output / 'plan.json'),
              'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=50)
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
