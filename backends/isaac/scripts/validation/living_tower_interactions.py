"""Native object-only checks for the selected living-room tower releases.

The reset/stack/collection probes and bounded engineering clamp are explicit
diagnostics. They cannot earn robot-operation or game credit. Every physical
step retains unmodified native poses, velocities, signed contact manifolds and
the actual applied wrench. Rules are loaded from the selected frozen release.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import gzip
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from bench.isaacsim.release import digest, dump, task_inputs, _frozen_evaluator_module
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection

PROJECT = Path(__file__).resolve().parents[2]
CAPACITY = 8192


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.asarray(value).copy()


def json_checks(checks):
    """Native/NumPy comparisons must become ordinary JSON boolean values."""
    if any(not isinstance(value, (bool, np.bool_)) for value in checks.values()):
        raise TypeError('Diagnostic checks must be boolean verdicts')
    return {name: bool(value) for name, value in checks.items()}


def fast_raw_contacts(arrays, sensors, mapped):
    """Vectorize only arithmetic/copying of the frozen signed-point protocol."""
    impulses, points, normals, separations, counts, starts = arrays
    if counts.shape != starts.shape or counts.shape != (len(sensors), len(mapped[0])):
        raise ValueError('Incomplete native contact mapping')
    result, seen = [], {}
    for i, j in zip(*np.nonzero(counts)):
        count, start = int(counts[i, j]), int(starts[i, j])
        if count < 0 or start < 0 or start+count >= len(impulses):
            raise ValueError('Native contact buffer overflow or invalid slice')
        a, b = sensors[i], mapped[i][j]
        if a == b:
            raise ValueError('Native self-pair')
        pair = tuple(sorted((a, b)))
        sign = 1 if a < b else -1
        chosen = slice(start, start+count)
        p = np.asarray(points[chosen], dtype=float)
        vectors = (np.asarray(impulses[chosen], dtype=float).reshape(-1, 1)
            * np.asarray(normals[chosen], dtype=float))*sign
        separation = np.asarray(separations[chosen], dtype=float).reshape(-1, 1)
        geometry = np.concatenate((p, vectors, separation), axis=1)
        if not np.isfinite(geometry).all():
            raise ValueError('Nonfinite native contact')
        fingerprint = sorted(map(tuple, geometry))
        if pair in seen:
            if seen[pair] != fingerprint:
                raise ValueError('Mirrored native contact views disagree')
            continue
        seen[pair] = fingerprint
        result.extend({'a': pair[0], 'b': pair[1], 'point_m': row[:3],
            'impulse_on_a_n_s': row[3:6], 'separation_m': row[6]} for row in geometry.tolist())
    return result


def read_contacts(view, sensors, mapped, native_module, *, compare=False):
    buffers = view.get_contact_data(dt=1.)
    counts, starts = array(buffers[4]), array(buffers[5])
    active = counts > 0
    used = int(np.max((starts+counts)[active], initial=0))
    if used >= CAPACITY:
        raise ValueError('Native contact buffer capacity reached')
    # Transfer every occupied native point, plus one bounds-check sentinel. The
    # unused preallocated GPU buffer contains no evidence and need not cross PCIe.
    copied = [array(value[:used+1]) for value in buffers[:4]]+[counts, starts]
    result = fast_raw_contacts(copied, sensors, mapped)
    if compare and result != native_module.raw_contacts(copied, sensors, mapped):
        raise ValueError('Vectorized raw contact conversion differs from frozen helper')
    return result


def cached_normalizer(native_module, body_paths, *, game, contact_actor_paths):
    cache = {}
    def convert(rows):
        result = []
        for row in rows:
            for key in ('a', 'b'):
                path = row[key]
                if path not in cache:
                    owned = any(path == p or path.startswith(p+'/') for p in body_paths.values())
                    cache[path] = (owned, None)
            if cache[row['a']][0] or cache[row['b']][0]:
                for key in ('a', 'b'):
                    path = row[key]
                    if cache[path][1] is None:
                        cache[path] = (cache[path][0], native_module.actor(path, body_paths,
                            game=game, contact_actor_paths=contact_actor_paths))
                result.append({**row, 'a': cache[row['a']][1], 'b': cache[row['b']][1]})
        return result
    return convert


def memoized_support(game):
    """Per-observation cache; frozen support recursion strictly descends in Z."""
    original, cache = game._support, {}
    def support(bid, allowed_blocks, visiting=None, *, surface_actor='table'):
        key = (bid, surface_actor, frozenset(allowed_blocks))
        if key not in cache:
            cache[key] = original(bid, allowed_blocks, visiting, surface_actor=surface_actor)
        return cache[key]
    return original, support


def engineering_target(time_s, initial, goal, *, case):
    """Only the selected body receives this bounded external clamp wrench."""
    if case != 'extract' or time_s < 1.4 or time_s >= 9.4:
        return None
    target = np.asarray(initial, float).copy()
    destination = np.asarray(goal['position_m'], float)
    # Pull the middle strip along its long axis, keeping the remaining tower
    # fully passive. Then turn over the empty tray, lower, and release.
    if time_s < 6.4:
        target[:2] += (destination[:2]-target[:2]) * (time_s-1.4)/5.
        yaw = math.pi/2
    elif time_s < 7.9:
        target[:2] = destination[:2]
        yaw = math.pi/2 * (1.-(time_s-6.4)/1.5)
    else:
        target[:2] = destination[:2]
        target[2] += (destination[2]+.002-target[2]) * (time_s-7.9)/1.5
        yaw = 0.
    return target, yaw


def interval_contacts(frames, dt):
    intervals = frames[1:]
    pairs = {tuple(sorted((r['a'], r['b']))) for f in intervals for r in f['contacts']}
    result = []
    for a, b in sorted(pairs):
        per = [[r for r in f['contacts'] if {r['a'], r['b']} == {a, b}] for f in intervals]
        last = next(rows[-1] for rows in reversed(per) if rows)
        vertical = [sum(r['impulse_on_a_n_s'][2]*(1 if r['a'] == a else -1) for r in rows)/dt for rows in per]
        target = a if min(vertical, default=0.) > 0 else b if max(vertical, default=0.) < 0 else None
        if target not in frames[-1]['objects']:
            target = None
        result.append({'a': a, 'b': b, 'point_m': last['point_m'],
            'separation_m': min(r['separation_m'] for rows in per for r in rows),
            'minimum_upward_force_n': min(map(abs, vertical), default=0.) if target else 0.,
            'support_for': target, 'continuous': bool(per and all(per))})
    return result


def building_conditions(evaluator, frames, native_module, freeform):
    """Frozen contact/geometry predicates only, never a credited episode update."""
    tower = evaluator.tower
    states = {}
    for name, obj in frames[-1]['objects'].items():
        vmax, wmax, _, _ = native_module.extrema(frames, name)
        states[name] = tower._state({**obj, 'quaternion_wxyz': native_module.quaternion(obj['quaternion_wxyz']),
            'interval_max_linear_speed_m_s': vmax, 'interval_max_angular_speed_rad_s': wmax})
    contacts = tower._contacts({'contacts': interval_contacts(frames, evaluator.e['physics_dt_s'])})
    blocks, components = freeform._assemblies(tower, states, contacts)
    for component in components:
        component['motion_within_frozen_limits'] = all(
            states[n]['vmax'] <= tower.t['linear_speed_m_s'] and states[n]['wmax'] <= tower.t['angular_speed_rad_s']
            for n in component['block_ids'])
    return {'blocks': blocks, 'components': components,
        'maximum_eligible_height_m': max((c['height_m'] for c in components if c['eligible']), default=0.),
        'maximum_motion_valid_height_m': max((c['height_m'] for c in components
            if c['eligible'] and c['motion_within_frozen_limits']), default=0.),
        'peak_interval_linear_speed_m_s': max(s['vmax'] for s in states.values()),
        'peak_interval_angular_speed_rad_s': max(s['wmax'] for s in states.values())}


def game_observation(evaluator, previous, frame, native_module):
    config = evaluator.game.config
    geometry = config['geometry']
    h = geometry['block_size_m'][2]
    center = np.array([*geometry['center_xy_m'], geometry['table_top_z_m']])
    dt = 0. if previous is None else evaluator.e['physics_dt_s']
    blocks = {}
    for name, obj in frame['objects'].items():
        vmax, wmax, zmin, tilt = native_module.extrema([frame] if previous is None else [previous, frame], name)
        blocks[name] = {'center_h': ((np.asarray(obj['position_m'])-center)/h).tolist(),
            'quaternion_wxyz': native_module.quaternion(obj['quaternion_wxyz']),
            'linear_velocity_h_s': (np.asarray(obj['linear_velocity_m_s'])/h).tolist(),
            'angular_velocity_rad_s': obj['angular_velocity_rad_s'],
            # Both robots are absent, so these are conservative lower bounds,
            # explicitly not observed robot clearances from a live episode.
            'hand_distance_h': {'front': 100., 'rear': 100.},
            'interval_max_linear_speed_h_s': vmax/h, 'interval_max_angular_speed_rad_s': wmax,
            'interval_min_center_z_h': (zmin-center[2])/h, 'interval_max_tilt_rad': tilt}
    contacts = []
    for row in frame['contacts']:
        impulse = np.asarray(row['impulse_on_a_n_s']); magnitude = float(np.linalg.norm(impulse))
        contacts.append({'a': row['a'], 'b': row['b'], 'normal_z_on_a': float(impulse[2]/magnitude) if magnitude else 0.,
            'normal_impulse_n_s': magnitude, 'point_h': ((np.asarray(row['point_m'])-center)/h).tolist(),
            'separation_h': row['separation_m']/h, 'supported_interval_s': dt if magnitude else 0.,
            'minimum_normal_force_n': magnitude/dt if dt else 0.})
    return {'schema': 'living_tower_extraction_normalized_observation.v1',
        'stream_id': frame['episode_id'], 'episode_id': frame['episode_id'], 'reset_id': frame['episode_id'],
        'configuration_sha256': config['configuration_sha256'], 'tick': frame['step_index'],
        'time_s': frame['time_s'], 'blocks': blocks, 'contacts': contacts, 'executed_sides': []}


def extraction_conditions(evaluator, previous, frame, native_module, selected):
    game = evaluator.game
    obs = game_observation(evaluator, previous, frame, native_module)
    # Independent condition inspection does not call observe/command, fabricate
    # contacts or establish turn ownership. Preserve all referee state.
    old, old_previous = game._observation, game._previous
    game._observation = obs
    game._previous = None if previous is None else game_observation(evaluator, None, previous, native_module)
    remaining = set(obs['blocks'])-({selected} if selected else set())
    try:
        supported = {n: game._support(n, remaining) for n in remaining}
        moving = {n: game._moving(n) for n in remaining}
        slot_stable = False
        inside_slots = []
        if selected:
            surface = game.config['collection_trays']['front']['actor_id']
            slot_stable = game._support(selected, set(), surface_actor=surface) and not game._moving(selected)
            inside_slots = [s['slot_id'] for s in game.config['collection_slots']['front'] if game._inside(selected, s)]
        return {'all_remaining_supported': all(supported.values()), 'remaining_supported': supported,
            'all_remaining_motion_valid': not any(moving.values()), 'remaining_moving': moving,
            'selected_tray_supported_and_motion_valid': bool(slot_stable),
            'selected_inside_front_slots': inside_slots,
            'maximum_remaining_drop_m': max((game.config['blocks'][n]['center_h'][2]-obs['blocks'][n]['interval_min_center_z_h'])*.03 for n in remaining),
            'maximum_remaining_tilt_rad': max(obs['blocks'][n]['interval_max_tilt_rad'] for n in remaining)}
    finally:
        game._observation, game._previous = old, old_previous


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Store evidence under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'living_tower_object_interactions.v1', 'passed': False, 'case': args.case,
        'task_id': args.task, 'robot_present': False, 'robot_manipulation_qualified': False,
        'task_success_claimed': False, 'physics_accepted': False, 'model_calls': 0,
        'post_reset_pose_writes': 0, 'external_force_calls': 0, 'source_root': str(PROJECT)}
    app = None
    try:
        source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.living_tower_interactions',), entrypoint_file=Path(__file__))
        if source is None or Path.cwd().resolve() != PROJECT:
            raise ValueError('Use complete immutable source snapshot as working directory')
        report['source'] = source
        report['source_manifest_sha256'] = digest(PROJECT/'source_snapshot_manifest.json')
        report['gpu'] = gpu_selection(args.gpu_uuid)
        _, scene, task, _ = task_inputs(args.release, args.task)
        report['release'] = {'path': str(args.release.resolve()), 'manifest_sha256': digest(args.release/'release_manifest.json')}
        report['physics_options'] = scene['isaacsim']
        dt = float(scene['isaacsim']['physics_dt_s'])
        duration = 11. if args.case == 'extract' else 1.6
        report.update(physics_dt_s=dt, duration_s=duration)
        frozen = _frozen_evaluator_module(args.release.resolve())
        native_module = importlib.import_module(frozen.__package__+'.native_evidence')
        freeform = importlib.import_module(frozen.__package__+'.tasks.tower_building.freeform') if args.task == 'tower_building' else None
        evaluator = frozen.SequenceEvaluator(task)
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 64, 'height': 64, 'multi_gpu': False,
            'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0, 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']}, experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from bench.simulation.isaacsim_backend.workroom import add_workroom
        from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
        from bench.simulation.isaacsim_backend.static_mesh_contacts import adapt_static_meshes, FixedActorGuard
        omni.usd.get_context().new_stage(); stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=.02, backend='torch', device='cuda:0')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True); context.set_broadphase_type('GPU'); context.set_solver_type('TGS')
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        api.CreateEnableCCDAttr(False); api.CreateGpuMaxRigidContactCountAttr(2**20)
        api.CreateGpuMaxRigidPatchCountAttr(2**18); api.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        api.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        if 'enable_external_forces_every_iteration' in scene['isaacsim']:
            api.CreateEnableExternalForcesEveryIterationAttr(scene['isaacsim']['enable_external_forces_every_iteration'])
        room = add_workroom(stage, scene['workroom'], surface_materials=scene['isaacsim']['workroom_surface_materials'])
        layers = mount_task_layers(stage, [s for s in scene['task_layers'] if s['id'] == 'task'], physics_scene_path=context.prim_path)
        paths = {n: row['path'] for n, row in layers['objects'].items()}
        if paths != task['evaluation']['body_paths']:
            raise ValueError('Physical body inventory does not match frozen referee')
        report['task_layers'] = layers
        reset_poses = {}
        if args.case == 'stack':
            for index, name in enumerate(f'block_{i:02d}' for i in range(30, 36)):
                layer, slot = divmod(index, 2)
                yaw = (layer % 2)*math.pi/2
                xy = np.array([.65, .25])+np.array([-.0275 if slot == 0 else .0275, 0.]) if layer % 2 else np.array([.65, .25])+np.array([0., -.0275 if slot == 0 else .0275])
                reset_poses[name] = [*xy.tolist(), .7925+layer*.025, math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
            reset_poses['block_29'] = [.65, -.20, 1.0, 1., 0., 0., 0.]
        elif args.case == 'collection':
            slot = task['evaluation']['reference_configuration']['collection_slots']['front'][4]
            reset_poses['block_16'] = [*slot['position_m'][:2], slot['position_m'][2]+.002, 1., 0., 0., 0.]
        for name, pose in reset_poses.items():
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(paths[name])); xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*pose[:3])); xform.AddOrientOp().Set(Gf.Quatf(pose[3], Gf.Vec3f(*pose[4:])))
        report['reset_only_pose_overrides'] = reset_poses
        for path in paths.values():
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        adapted = adapt_static_meshes(stage, scene['isaacsim']['static_mesh_contact_mode'])
        static = sorted({adapted['collider_actor_paths'].get(p, p) for p in room['contact_filter_paths']+layers['static_collision_paths']} | set(adapted['kinematic_body_paths']))
        stage.Export(str(output/'initial.usda'))
        dump(output/'task.json', task); dump(output/'scene.json', scene)
        world.reset(); sim = world.physics_sim_view
        bodies = sim.create_rigid_body_view(list(paths.values()))
        reverse = {p: n for n, p in paths.items()}; names = [reverse[p] for p in bodies.prim_paths]
        if len(names) != 36 or set(names) != set(paths):
            raise ValueError('Incomplete native body inventory')
        # World.reset performs initialization steps. Restore the declared poses
        # once, inside this explicit reset boundary, as the canonical adapter
        # does; never mistake those warmup displacements for the frozen reset.
        reset_values = []
        for name in names:
            original_state = layers['objects'][name]['initial_state']
            pose = reset_poses.get(name, [*original_state['position_m'], *original_state['quaternion_wxyz']])
            reset_values.append([*pose[:3], *pose[4:], pose[3]])
        reset_device = bodies.get_transforms().device
        reset_indices = torch.arange(len(names), dtype=torch.int32, device=reset_device)
        bodies.set_transforms(torch.tensor(reset_values, dtype=torch.float32, device=reset_device), reset_indices)
        bodies.set_velocities(torch.zeros((len(names), 6), dtype=torch.float32, device=reset_device), reset_indices)
        report['reset_boundary_writes'] = {'body_poses': len(names), 'body_velocities': len(names),
            'reason': 'Restore authored initial state after World initialization, before baseline/time zero.',
            'post_baseline_writes': 0}
        filters = [*paths.values(), *static]
        contacts = sim.create_rigid_contact_view(list(paths.values()), filter_patterns=[filters for _ in paths], max_contact_data_count=CAPACITY)
        sensors, mapped = list(contacts.sensor_paths), [list(row) for row in contacts.filter_paths]
        report['contact_mapping'] = {'sensors': sensors, 'filters': mapped[0], 'unobserved_room_colliders': room['contact_unobserved_paths']}
        if any(row != mapped[0] for row in mapped) or set(mapped[0]) != set(filters):
            raise ValueError('Native contact mapping differs')
        fixed_view = sim.create_rigid_body_view(adapted['kinematic_body_paths']) if adapted['kinematic_body_paths'] else None
        guard = FixedActorGuard(list(fixed_view.prim_paths), array(fixed_view.get_transforms())) if fixed_view else None
        selected = 'block_16' if args.case in ('extract', 'collection') else None
        goal = task['evaluation']['reference_configuration']['collection_slots']['front'][7] if args.case == 'extract' else None
        report['engineering_goal_slot'] = goal
        selected_index = names.index('block_16')
        masses = array(bodies.get_masses()).reshape(-1)
        selected_view = sim.create_rigid_body_view(paths['block_16'])
        device = selected_view.get_transforms().device
        indices = torch.tensor([0], dtype=torch.int32, device=device)
        report['native_masses_kg'] = dict(zip(names, masses.tolist()))
        report['engineering_boundary'] = {'applied_only_to': 'block_16' if args.case == 'extract' else None,
            'position_stiffness_n_m': [2000., 2000., 10000.], 'damping_n_s_m': [36., 36., 82.],
            'per_axis_force_limit_n': 20., 'orientation_stiffness_nm_rad': 4.,
            'angular_damping_nm_s_rad': .08, 'per_axis_torque_limit_nm': .5,
            'gravity_compensation_selected_body_only': args.case == 'extract',
            'release_at_s': 9.4 if args.case == 'extract' else 0., 'all_other_bodies_passive': True,
            'robot_controller_modified': False, 'valid_benchmark_actions': False,
            'description': 'Bounded external wrench on one dynamic selected strip; no kinematic body or live pose/velocity writes.'}
        original = array(bodies.get_transforms())[selected_index, :3].copy()
        baseline_step, baseline_time = int(world.current_time_step_index), float(world.current_time)
        previous = None; window = deque(); conditions = []; zero_scores = []; game_states = []
        normalize = cached_normalizer(native_module, paths, game=args.task == 'tower_extraction',
            contact_actor_paths=task['evaluation']['contact_actor_paths'])
        equivalence_ticks = []
        max_penetration = 0.; native_count = round(duration/dt); peak_force = peak_torque = 0.
        last_progress = time.monotonic()
        with gzip.open(output/'native_frames.jsonl.gz', 'xt', compresslevel=1) as stream:
            for tick in range(native_count+1):
                force, torque = np.zeros(3), np.zeros(3)
                target = None
                if tick:
                    target = engineering_target((tick-1)*dt, original, goal, case=args.case)
                    if target is not None:
                        q = array(selected_view.get_transforms())[0]; v = array(selected_view.get_velocities())[0]
                        position, yaw = target
                        force = np.array([2000., 2000., 10000.])*(position-q[:3])-np.array([36., 36., 82.])*v[:3]
                        force[2] += masses[selected_index]*9.81
                        force = np.clip(force, -20., 20.)
                        current = native_module.rotation([q[6], *q[3:6]])
                        desired = native_module.rotation([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
                        error = .5*sum((np.cross(current[:, axis], desired[:, axis]) for axis in range(3)), np.zeros(3))
                        torque = np.clip(4.*error-.08*v[3:], -.5, .5)
                        selected_view.apply_forces_and_torques_at_position(force_data=torch.tensor(force[None], dtype=torch.float32, device=device),
                            torque_data=torch.tensor(torque[None], dtype=torch.float32, device=device), position_data=None, indices=indices, is_global=True)
                        report['external_force_calls'] += 1
                        peak_force = max(peak_force, float(np.linalg.norm(force))); peak_torque = max(peak_torque, float(np.linalg.norm(torque)))
                    world.step(render=False)
                if int(world.current_time_step_index) != baseline_step+tick:
                    raise ValueError('Missing native physics step')
                if guard:
                    guard.check(list(fixed_view.prim_paths), array(fixed_view.get_transforms()))
                poses, velocities = array(bodies.get_transforms()), array(bodies.get_velocities())
                compare = bool(tick and (tick == 1 or tick % 400 == 0))
                raw = [] if not tick else read_contacts(contacts, sensors, mapped, native_module, compare=compare)
                if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                    raise ValueError('Nonfinite native state')
                frame = {'schema': 'living_native_evidence_v1', 'complete': True, 'episode_id': 'object_probe_'+args.case,
                    'step_index': tick, 'time_s': tick*dt, 'native_physics_step_index': int(world.current_time_step_index),
                    'native_world_time_s': float(world.current_time), 'objects': {}, 'joints': {}, 'particle_sets': {},
                    'raw_contacts': raw, 'robot_action_count': 0, 'executed_sides': []}
                expected = baseline_time+tick*float(np.float32(dt))
                if abs(frame['native_world_time_s']-expected) > max(1.e-10, tick*8*math.ulp(max(1., abs(expected)))):
                    raise ValueError('Native world clock differs from the recorded step size')
                for name, pose, velocity in zip(names, poses, velocities):
                    frame['objects'][name] = {'position_m': pose[:3].tolist(), 'quaternion_wxyz': [float(pose[6]), *pose[3:].tolist()[:3]],
                        'linear_velocity_m_s': velocity[:3].tolist(), 'angular_velocity_rad_s': velocity[3:].tolist()}
                frame['contacts'] = normalize(raw)
                if compare:
                    if frame['contacts'] != native_module.normalized_contacts(raw, paths, game=args.task == 'tower_extraction', contact_actor_paths=task['evaluation']['contact_actor_paths']):
                        raise ValueError('Cached actor normalization differs from frozen helper')
                    equivalence_ticks.append(tick)
                frame['engineering_wrench'] = {'body': selected if target is not None else None, 'force_world_n': force.tolist(), 'torque_world_nm': torque.tolist(),
                    'target_position_m': target[0].tolist() if target is not None else None, 'target_yaw_rad': target[1] if target is not None else None}
                max_penetration = max(max_penetration, max((-r['separation_m'] for r in raw), default=0.))
                window.append(frame)
                if args.task == 'tower_building':
                    if args.case == 'supply':
                        score = evaluator.update(frame); zero_scores.append(score.get('best_height_m'))
                        if score['status'] == 'invalid':
                            dump(output/'rejected_native_frame.json', frame)
                            raise ValueError('Frozen supply referee rejected native evidence: '+score['reason'])
                    if tick and tick % round(.02/dt) == 0:
                        condition = building_conditions(evaluator, list(window), native_module, freeform)
                        conditions.append({'time_s': tick*dt, **condition}); window.clear(); window.append(frame)
                else:
                    if args.case == 'reset' or (args.case == 'extract' and tick*dt < 1.4):
                        observation = game_observation(evaluator, previous, frame, native_module)
                        original_support, cached_support = memoized_support(evaluator.game)
                        evaluator.game._support = cached_support
                        try:
                            result = evaluator.game.observe(observation)
                        finally:
                            evaluator.game._support = original_support
                        if not game_states or result['status'] != game_states[-1]['status']:
                            game_states.append({'time_s': tick*dt, **result})
                    if tick and tick % round(.02/dt) == 0:
                        original_support, cached_support = memoized_support(evaluator.game)
                        evaluator.game._support = cached_support
                        try:
                            condition = extraction_conditions(evaluator, previous, frame, native_module, selected)
                        finally:
                            evaluator.game._support = original_support
                        conditions.append({'time_s': tick*dt, **condition})
                    window.clear()
                stream.write(json.dumps(frame, separators=(',', ':'), allow_nan=False)+'\n')
                previous = frame
                if time.monotonic()-last_progress > 20.:
                    print(json.dumps({'event': 'native_progress', 'case': args.case, 'time_s': tick*dt, 'steps': tick}), flush=True)
                    last_progress = time.monotonic()
        tail = [r for r in conditions if r['time_s'] >= duration-1.-1.e-9]
        checks = {'complete_native_steps': native_count == tick,
            'no_deep_contact_penetration': max_penetration <= .0015,
            'physical_assets_unchanged': all(digest(s['usd_path']) == s['sha256'] for s in scene['task_layers'])}
        if args.case == 'supply':
            checks.update(supply_never_scores=all(s == 0. for s in zero_scores),
                supply_not_table_supported=all(r['maximum_eligible_height_m'] == 0. for r in conditions))
        elif args.case == 'stack':
            checks.update(three_layer_released_support_one_second=all(.074 < r['maximum_motion_valid_height_m'] < .077 for r in tail),
                falling_block_initially_unsupported=not conditions[0]['blocks']['block_29']['supported_to_table'],
                falling_block_eventually_table_supported=conditions[-1]['blocks']['block_29']['supported_to_table'])
        else:
            checks.update(remaining_supported_terminal_samples=all(r['all_remaining_supported'] for r in tail),
                remaining_stable_terminal_samples=all(r['all_remaining_motion_valid'] for r in tail))
            if args.case == 'reset':
                checks['frozen_referee_awaits_intent'] = evaluator.game.status == 'await_intent'
                checks['protected_layers_not_selectable'] = evaluator.game.legal_blocks() == [f'block_{i:02d}' for i in range(3, 33)]
            if selected:
                checks['collection_supported_terminal_samples'] = all(r['selected_tray_supported_and_motion_valid'] for r in tail)
                checks['collection_inside_declared_slot_terminal_samples'] = all(r['selected_inside_front_slots'] for r in tail)
            if args.case == 'extract':
                checks.update(real_continuous_extraction=abs(previous['objects'][selected]['position_m'][1]-original[1]) > .3,
                    original_reset_became_ready=any(r['status'] == 'await_intent' for r in game_states),
                    remaining_tower_no_collapse=all(r['maximum_remaining_drop_m'] < .024 and r['maximum_remaining_tilt_rad'] < math.pi/4 for r in conditions),
                    fully_released_terminal_window=duration-9.4 >= 1.)
        checks = json_checks(checks)
        dump(output/'conditions.json', conditions)
        report.update(passed=all(checks.values()), checks=checks, native_steps=native_count,
            native_frame_count=native_count+1, terminal_condition_samples=len(tail),
            terminal_condition_sampling_s=.02,
            terminal_condition_coverage=('Complete native extrema and contact intervals' if args.task == 'tower_building'
                else 'Adjacent native endpoints at 20 ms checkpoints; full terminal native replay is separately required.'),
            maximum_native_penetration_m=max_penetration, final_objects=previous['objects'],
            maximum_applied_force_norm_n=peak_force, maximum_applied_torque_norm_nm=peak_torque,
            frozen_contact_conversion_equivalence_ticks=equivalence_ticks,
            game_state_transitions=game_states, fixed_scenery=guard.report() if guard else None,
            evidence_sha256={name: digest(output/name) for name in ('native_frames.jsonl.gz', 'initial.usda', 'task.json', 'scene.json', 'conditions.json')},
            scope='Object dynamics and frozen geometric/stability predicates only. Engineering traces never enter a credited robot/game episode.')
    except BaseException as error:
        traceback.print_exc(); report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            dump(output/'qualification.json', report)
            print(json.dumps({'passed': report['passed'], 'output': str(output), 'checks': report.get('checks'), 'error': report.get('error')}), flush=True)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if app is not None:
                app.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('tower_building', 'tower_extraction'), required=True)
    parser.add_argument('--case', choices=('supply', 'stack', 'reset', 'collection', 'extract'), required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if (args.task == 'tower_building') != (args.case in ('supply', 'stack')):
        parser.error('Case and task differ')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
