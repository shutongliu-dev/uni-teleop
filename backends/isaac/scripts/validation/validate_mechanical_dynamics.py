"""Native dynamics acceptance for the current mechanical workshop, not task success.

Frozen source required. Actual gravity, source threaded bodies, fixed base,
small joint motion, contact coverage, reset and raw RGB-D are checked separately.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback
import numpy as np
from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import (THRESHOLDS, digest, plain,
    rotation, angle, motion_spec, smooth_targets, assess_robot, compare_reset,
    check_window, stage_inventory, record, capture)
from scripts.validation.mechanical_checks import (assess_support,
    assess_robot_clearance, contact_filter_coverage, assess_ready_pose)
from scripts.diagnostics.probe_workroom_base import native_static_readback


class FixtureMonitor:
    """Read-only post-integration pose guard, including source USD fixtures."""
    def __init__(self, env):
        self.env, self.enabled = env, True
        self.baselines, self.samples = {}, 0
        self.maximum_translation_m = self.maximum_angle_rad = 0.
        self.refresh()

    def refresh(self):
        from bench.simulation.isaacsim_backend.env import _np
        self.views = {}
        for info in self.env.scene_info['fixtures'].values():
            if info.get('body_mode') != 'kinematic_fixture':
                continue
            path = info['path']
            view = self.env._sim_view.create_rigid_body_view(path)
            if view.count != 1:
                raise RuntimeError('Missing native source fixture actor: '+path)
            self.views[path] = view
            self.baselines.setdefault(path, _np(view.get_transforms())[0].copy())
        self.enabled = True
        self.check()

    def check(self):
        from bench.simulation.isaacsim_backend.env import _np
        if not self.enabled:
            return
        for path, view in self.views.items():
            pose = _np(view.get_transforms())[0]
            old = self.baselines[path]
            if not np.isfinite(pose).all():
                raise RuntimeError('Nonfinite fixture pose')
            self.maximum_translation_m = max(self.maximum_translation_m, float(np.linalg.norm(pose[:3]-old[:3])))
            self.maximum_angle_rad = max(self.maximum_angle_rad, angle(pose[[6,3,4,5]], old[[6,3,4,5]]))
        self.samples += 1
        if self.maximum_translation_m > 1e-6 or self.maximum_angle_rad > 1e-5:
            raise RuntimeError('Fixed task fixture moved during physics')

    def wrap(self, reader):
        def observe(*args, **kwargs):
            result = reader(*args, **kwargs)
            self.check()
            return result
        return observe

    def result(self, native_steps):
        return {'passed': self.samples >= native_steps,
            'checked_batches': self.samples, 'actor_paths': sorted(self.views),
            'maximum_translation_m': self.maximum_translation_m,
            'maximum_angle_rad': self.maximum_angle_rad,
            'scope': 'Read-only actual native actor poses after every integration substep and reset; no pose targets.'}


def declared_mount_pairs(env):
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    fixture = next(x for x in env.scene_spec['fixtures'] if x['id'] == 'robot_stand')
    asset = load_task_asset(fixture['asset_id'], fixture['asset_root'], verify_hashes=True)
    indices = [i for i, part in enumerate(asset['collision_parts']) if part['label'] == 'robot_stand_mount']
    if len(indices) != 1:
        raise ValueError('Stand must declare exactly one mounting collider')
    path = env.scene_info['fixtures']['robot_stand']['collision_paths'][indices[0]]
    actor = env.scene_info.get('collider_actor_paths', {}).get(path, path)
    return [(env.scene_info['robot_link_paths']['fr3_link0'], actor)]


def native_collider_inventory(env, info, authored, asset, view):
    """Check exact native material bindings and exclusion from scene queries.

    Disabled USD colliders can remain in native material tensor storage. Query
    participation is measured separately; no simulation-shape flag API is claimed.
    """
    from bench.simulation.isaacsim_backend.env import _np
    from omni.physx import get_physxunittests_interface, get_physx_scene_query_interface
    from pxr import Usd, UsdPhysics
    source = {str(p.GetPath()): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
              for p in Usd.PrimRange(env.stage.GetPrimAtPath(info['path']))
              if p.HasAPI(UsdPhysics.CollisionAPI)}
    active = sorted(p for p, enabled in source.items() if enabled)
    disabled = sorted(p for p, enabled in source.items() if not enabled)
    declared = (active == sorted(authored.get('active_collision_paths', []))
                and disabled == sorted(authored.get('disabled_collision_paths', []))
                and set(source) == set(authored['collider_materials']))
    native_paths = get_physxunittests_interface().get_materials_paths
    bindings = {p: {'actual': [str(v) for v in native_paths(p)], 'expected': row['path']}
                for p, row in authored['collider_materials'].items()}
    bindings_match = all(bool(row['actual']) and set(row['actual']) == {row['expected']}
                         for row in bindings.values())
    query = {'performed': False, 'passed': True}
    if disabled:
        before = env.capture_state()
        center = _np(view.get_transforms())[0, :3].tolist()
        extent = np.maximum(abs(np.asarray(asset['bounds_min_m'])), abs(np.asarray(asset['bounds_max_m'])))
        radius = float(np.linalg.norm(extent)+.01)
        hits = []
        def receive(hit):
            hits.append({'collision': str(hit.collision), 'rigid_body': str(hit.rigid_body)})
            return True
        count = get_physx_scene_query_interface().overlap_sphere(radius, tuple(center), receive, False)
        after = env.capture_state()
        unchanged = all(np.array_equal(before[k], after[k]) for k in before)
        actual = sorted({h['collision'] for h in hits if h['rigid_body'] == info['path']})
        query = {'performed': True, 'passed': unchanged and count == len(hits) and actual == active,
            'center_m': center, 'radius_m': radius, 'hits': hits, 'hit_count': count,
            'actual_body_colliders': actual, 'expected_active_colliders': active,
            'disabled_colliders_absent': not set(disabled).intersection(actual),
            'physics_state_unchanged': unchanged,
            'scope': 'Native enclosing-sphere scene query; exact active paths present and disabled paths absent. This does not read simulation-shape flags.'}
    return {'passed': declared and bindings_match and query['passed'],
        'composed_usd_inventory_matches': declared, 'active_collision_paths': active,
        'disabled_collision_paths': disabled, 'native_material_bindings': bindings,
        'native_bindings_match': bindings_match, 'scene_query': query,
        'native_simulation_shape_flags_verified': False}


def native_task_parameters(env, *, required_roles=None):
    """Require all three threaded roles and their native mass/material evidence.

    Diagnostic cubes and the stand need not use the source-USD parameter path.
    Missing threaded-role metadata is a failed check, never an empty pass.
    """
    from bench.simulation.isaacsim_backend.env import _np
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from pxr import UsdPhysics
    required = ({'screwdriver': 'objects', 'bolt': 'objects', 'threaded_fixture': 'fixtures'}
                if required_roles is None else dict(required_roles))
    if not required or any(not isinstance(name, str) or not name or group not in ('objects', 'fixtures')
                           for name, group in required.items()):
        raise ValueError('Declare a nonempty explicit required USD actor set')
    objects, fixtures = env.scene_info['objects'], env.scene_info['fixtures']
    bodies = {**objects, **fixtures}
    records = {}
    selected = set(required) | {name for name, info in bodies.items()
                               if info.get('usd_physics_parameters') is not None}
    for name in sorted(selected):
        try:
            if name in objects and name in fixtures:
                raise ValueError('Actor role occurs in both object and fixture inventories')
            if name in required and name not in env.scene_info[required[name]]:
                raise ValueError('Missing required threaded actor role: '+name)
            info = bodies[name]
            authored = info.get('usd_physics_parameters')
            if not isinstance(authored, dict):
                raise ValueError('Missing source-USD physics parameter evidence')
            active, disabled = authored['active_collision_paths'], authored['disabled_collision_paths']
            bindings = authored['collider_materials']
            if (not isinstance(active, list) or not active or not isinstance(disabled, list)
                    or not isinstance(bindings, dict) or not bindings
                    or any(not isinstance(p, str) or not p.startswith('/') for p in active+disabled)
                    or len(set(active+disabled)) != len(active+disabled)
                    or set(active+disabled) != set(bindings)):
                raise ValueError('Incomplete active collider or material-binding inventory')
            view = env._sim_view.create_rigid_body_view(info['path'])
            if view.count != 1:
                raise ValueError('Expected exactly one native body for '+name)
            native_materials = np.asarray(_np(view.get_material_properties()), dtype=float)
            native_masses = np.asarray(_np(view.get_masses()), dtype=float)
            if (native_materials.ndim != 3 or native_materials.shape[0] != 1
                    or native_materials.shape[1] == 0 or native_materials.shape[2] != 3
                    or native_masses.size != 1
                    or not np.isfinite(native_materials).all() or not np.isfinite(native_masses).all()):
                raise ValueError('Missing or invalid native mass/material readback')
            materials = native_materials[0].tolist()
            mass = float(native_masses.reshape(-1)[0])
            spec = next(x for x in env.scene_spec['objects']+env.scene_spec['fixtures'] if x['id'] == name)
            asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
            source_mass = float(asset['mass_kg'])
            prim = env.stage.GetPrimAtPath(info['path'])
            usd_mass = float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
            if not np.isfinite([source_mass, usd_mass]).all() or min(source_mass, usd_mass) <= 0:
                raise ValueError('Invalid source or composed USD mass')
            dynamic = info['body_mode'] == 'dynamic'
            mass_checks = {
                'source_usd_mass_preserved': abs(usd_mass-source_mass) < 1e-6,
                'native_dynamic_mass': abs(mass-source_mass) < 1e-6 if dynamic else None,
                'declared_body_mode': UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() == (not dynamic),
            }
            expected = [[v['parameters'][k] for k in (
                'physics:staticFriction', 'physics:dynamicFriction', 'physics:restitution')]
                for v in bindings.values()]
            if not np.isfinite(np.asarray(expected, dtype=float)).all():
                raise ValueError('Invalid authored material parameter evidence')
            shapes_match = len(materials) == len(expected) and np.allclose(
                sorted(materials), sorted(expected), atol=1e-6, rtol=0)
            inventory = native_collider_inventory(env, info, authored, asset, view)
            records[name] = {'passed': bool(shapes_match and inventory['passed']
                                           and all(v is not False for v in mass_checks.values())),
                'native_actor_path': info['path'], 'native_mass_kg': mass,
                'mass_checks': mass_checks, 'composed_usd_mass_kg': usd_mass,
                'mass_scope': 'Dynamic body mass must match source. Fixed kinematic fixture tensor mass is reported but is not a dynamic mass qualification; native fixed pose is independently guarded every substep.',
                'source_mass_kg': source_mass, 'native_material_properties': materials,
                'expected_material_properties': expected,
                'collider_inventory': inventory,
                'scope': 'Unordered material rows compared within one exact native body; no USD-to-native shape-index correspondence claimed.'}
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, RuntimeError, StopIteration) as exc:
            records[name] = {'passed': False, 'error': {'type': type(exc).__name__, 'detail': str(exc)}}
    return {'passed': set(required) <= records.keys() and all(x['passed'] for x in records.values()),
            'required_body_names': sorted(required), 'bodies': records}


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    plan = {'schema': 'native_mechanical_dynamics_plan_v1', 'created_utc': started,
        'family': args.family, 'thresholds': deepcopy(THRESHOLDS), 'model_calls': 0,
        'asset_profile': args.asset_profile,
        'task_success_claimed': False, 'static_decoration_manipulability_claimed': False,
        'scope': 'Gravity, actual support impulse, settling, small robot joint motion, reset and RGB-D only.',
        'device': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'single_cuda_device': os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE'),
        'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
        'validator_sha256': digest(__file__), 'source_root': str(root)}
    dump(output/'plan.json', plan)
    env, error = None, None
    checks, frames, records = {}, {}, []
    total_steps, native_steps, native_duration = 0, 0, 0.
    try:
        from scenes.workrooms.mechanical_assembly.dynamics import diagnostic_scene, diagnostic_scene_v3
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        verification = verify_if_snapshot(root,
            required_modules=('scripts.validation.validate_mechanical_dynamics', 'scenes.workrooms.mechanical_assembly.dynamics', 'scenes.workrooms',
                              'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('Native validation requires a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verification)
        if args.device != 'cuda:0' or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('Use single UUID-selected CUDA device with --device cuda:0')
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid:
            raise ValueError('CUDA_VISIBLE_DEVICES must be one physical GPU UUID')
        query = subprocess.run(['nvidia-smi', '-i', uuid,
            '--query-gpu=uuid,pci.bus_id,name,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=20)
        if len(query.stdout.strip().splitlines()) != 1 or not query.stdout.startswith(uuid+','):
            raise RuntimeError('Selected physical GPU UUID did not resolve uniquely')
        dump(output/'gpu_identity.json', {'query': query.args, 'result': query.stdout.strip()})
        compose = diagnostic_scene_v3 if args.asset_profile == 'sharpa_v3' else diagnostic_scene
        spec, initial, diagnostics, ready_design = compose(args.stand_root, args.threaded_root)
        dump(output/'ready_design.json', ready_design)
        dump(output/'support_specs.json', diagnostics)
        dump(output/'scene.json', spec)
        # scene() checks pinned hashes without importing pxr. USD composition
        # validation must wait until SimulationApp has loaded its own schemas.
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=640, render_height=480)
        observation = env.reset(seed=THRESHOLDS['seed'], initial_state=initial)
        from bench.simulation.isaacsim_backend.workroom import preflight
        dump(output/'source_asset_preflight.json', preflight(spec['workroom'], {}, surface_materials=spec['isaacsim']['workroom_surface_materials']))
        base = native_static_readback(env)
        checks['native_fixed_base'] = {'passed': base['shared_metatype_fixed_base'] and base['get_metatype_0_fixed_base'] and len(base['native_link_names']) == 43 and len(base['native_dof_names']) == 29, **base}
        checks['contact_filter_coverage'] = contact_filter_coverage(env)
        checks['native_task_parameters'] = native_task_parameters(env)
        fixture_guard = FixtureMonitor(env)
        env._read_rigid_contacts = fixture_guard.wrap(env._read_rigid_contacts)
        dump(output/'metadata.json', env.metadata())
        checks['stage_inventory'] = stage_inventory(env)
        if not all(value['passed'] for value in checks.values()):
            raise RuntimeError('Native initialization, material or collider inventory check failed')
        action_spec = deepcopy(env.action_spec())
        action_spec['initial_joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        motion = motion_spec(action_spec)
        dump(output/'motion_plan.json', motion)
        paths = {name: row['path'] for name, row in env.scene_info['objects'].items()}
        home = action_spec['initial_joint_positions_rad']
        first = record(env, observation, 0, 'reset')
        records.append(first)
        dump(output/'initial_state.json', first)
        frames['initial'] = capture(env, output, 'initial')
        with (output/'physics.jsonl').open('x') as stream:
            def write(row):
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            write(first)
            def advance(phase, count, episode=0, fraction=None):
                nonlocal total_steps, native_steps, native_duration
                print(json.dumps({'event': 'workroom_phase', 'family': args.family,
                                  'phase': phase, 'episode': episode, 'steps': count}), flush=True)
                batch = []
                for i in range(count):
                    targets = deepcopy(home) if fraction is None else smooth_targets(home, motion['offsets_rad'], fraction(i, count))
                    command = {'joint_positions_rad': targets, 'duration_s': .02}
                    previous = records[-1]['state']
                    observed = env.step(command)
                    row = record(env, observed, episode, phase, command)
                    write(row)
                    records.append(row)
                    batch.append(row)
                    total_steps += 1
                    native_steps += int(row['state']['native_physics_step_index'])-int(previous['native_physics_step_index'])
                    native_duration += float(row['state']['native_world_time_s'])-float(previous['native_world_time_s'])
                    check_window(observed, row['state'], previous, env.dt)
                    if (i+1) % 25 == 0 or i+1 == count:
                        print(json.dumps({'event': 'workroom_progress', 'family': args.family,
                            'phase': phase, 'episode': episode, 'phase_steps': i+1,
                            'total_control_steps': total_steps,
                            'episode_sim_time_s': observed['sim_time_s']}), flush=True)
                return batch
            settled = advance('settle', THRESHOLDS['settle_steps'])
            checks['initial_drop_and_settle'] = assess_support(first['observation'], settled, diagnostics, paths)
            advance('outward', THRESHOLDS['outward_steps'], fraction=lambda i, n: (i+1)/n)
            advance('return', THRESHOLDS['return_steps'], fraction=lambda i, n: 1-(i+1)/n)
            advance('post_motion', THRESHOLDS['post_motion_steps'])
            checks['robot_motion'] = assess_robot(first, records[1:], action_spec, motion)
            checks['post_motion_support'] = assess_support(first['observation'], records[1:], diagnostics, paths)
            frames['final'] = capture(env, output, 'final')
            fixture_guard.enabled = False
            observation = env.reset(seed=THRESHOLDS['seed'], initial_state=initial)
            fixture_guard.refresh()
            second = record(env, observation, 1, 'reset')
            dump(output/'repeated_reset_state.json', second)
            records.append(second)
            write(second)
            repeated = advance('repeat_settle', THRESHOLDS['repeat_settle_steps'], episode=1)
            checks['repeated_drop_and_settle'] = assess_support(second['observation'], repeated, diagnostics, paths)
            checks['reset_reproducibility'] = compare_reset(first, second, settled[-1], repeated[-1])
            frames['repeat_final'] = capture(env, output, 'repeat_final')
        checks['fixed_task_fixtures_every_substep'] = fixture_guard.result(native_steps)
        checks['ready_pose'] = assess_ready_pose(records, initial['joint_positions_rad'], ready_design['wrist_fk'])
        checks['robot_clearance'] = assess_robot_clearance([r for r in records if r['phase'] != 'reset'], list(env.scene_info['robot_link_paths'].values()), allowed_pairs=declared_mount_pairs(env))
        checks['final_stage_inventory'] = stage_inventory(env)
        final_metadata = env.metadata()
        dump(output/'metadata_final.json', final_metadata)
        support_paths = env.scene_info.get('kinematic_body_paths', [])
        if support_paths:
            guard = final_metadata.get('static_mesh_contact_runtime', {})
            checks['fixed_supports_every_substep'] = {
                'passed': set(guard.get('actor_paths', [])) == set(support_paths)
                          and guard.get('checked_batches', 0) >= native_steps
                          and guard.get('maximum_translation_m', float('inf')) <= 1.e-6
                          and guard.get('maximum_angle_rad', float('inf')) <= 1.e-5,
                **guard}
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root,
             required_modules=('scripts.validation.validate_mechanical_dynamics', 'scenes.workrooms.mechanical_assembly.dynamics', 'scenes.workrooms',
                               'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'workroom_error', 'family': args.family,
                          'type': type(exc).__name__, 'detail': str(exc)}), flush=True)
    passed = bool(error is None and checks and frames and all(x['passed'] for x in checks.values())
                  and all(x['passed'] for x in frames.values()))
    report = {'schema': 'native_mechanical_dynamics_check_v1', 'family': args.family,
        'passed': passed, 'model_calls': 0, 'task_success_claimed': False,
        'static_decoration_manipulability_claimed': False,
        'scope': plan['scope'], 'checks': plain(checks), 'rgbd': frames, 'error': error,
        'control_steps': total_steps, 'simulation_duration_s': native_duration,
        'physics_substeps': native_steps,
        'requested_control_duration_s': total_steps*.02,
        'step_count_scope': 'Observed native index/time differences per control action; initialization/reset ticks excluded.',
        'object_state_writeback_after_reset': False, 'plan_sha256': digest(output/'plan.json'),
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', default='mechanical_assembly')
    parser.add_argument('--stand-root', type=Path, required=True)
    parser.add_argument('--threaded-root', type=Path, required=True)
    parser.add_argument('--asset-profile', choices=('requested_v1', 'sharpa_v3'), default='sharpa_v3')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
