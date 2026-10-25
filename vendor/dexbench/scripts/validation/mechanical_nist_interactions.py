"""Object-only native support/fit probes for frozen NIST task bundles.

The goal case changes only reset transforms before World.reset(). It cannot earn
benchmark credit. All subsequent states come from gravity/contact/passive links;
there are no robots, action policies, live pose writes or external loads.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import OUTPUT_ROOT
from bench.simulation.isaacsim_backend.contact_precision import summarize_normal_impulses
from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle, joint_positions
from scenes.workrooms.mechanical_assembly.nist_composition import TASK_IDS, PHYSICS_DT
from scenes.workrooms.mechanical_assembly.nist_native_evidence import (
    angle, enclosed, rotation, route_in_order, simple_loop,
)
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection

PROJECT = Path(__file__).resolve().parents[2]
TABLE_PATH = '/World/ProbeTable'
CAPACITY = 65536
DWELL_S = .3


def array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def components(actors, joints):
    """Only explicitly inventoried native joints transmit declared link support."""
    groups = {name: {name} for name in actors}
    for joint in joints.values():
        combined = groups[joint['body0']] | groups[joint['body1']]
        for name in combined:
            groups[name] = combined
    return sorted({tuple(sorted(group)) for group in groups.values()})


def signed_external_load(matrix, sensor_names, filters, members, actors):
    """Sum complete signed vectors, excluding internal forces; never clamp cells."""
    values = np.asarray(matrix, float)
    if values.shape != (len(sensor_names), len(filters), 3) or not np.isfinite(values).all():
        raise ValueError('Complete finite contact-force matrix required')
    internal = {actors[name] for name in members}
    external_indices = [i for i, path in enumerate(filters) if path not in internal]
    indices = [sensor_names.index(name) for name in members]
    return values[np.ix_(indices, external_indices)].sum(axis=(0, 1))


def geometry_checks(metadata, objects):
    """Independent geometric fit only; operation history and contact are separate."""
    parameters = metadata['task_parameters']
    seats, groups = {}, {}
    for goal in parameters.get('seating_goals', []):
        obj = objects[goal['actor']]
        local_point = np.asarray(goal.get('actor_local_point_m', [0., 0., 0.]), float)
        actual_point = np.asarray(obj['position_m'])+rotation(obj['quaternion_wxyz']) @ local_point
        target_point = np.asarray(goal['target_position_m'])
        error = np.abs(actual_point-target_point)
        if goal.get('orientation_axis_only', False):
            axis = np.asarray(goal.get('orientation_axis_local', [0., 0., 1.]), float)
            axis /= np.linalg.norm(axis)
            actual = rotation(obj['quaternion_wxyz']) @ axis
            expected = rotation(goal['target_quaternion_wxyz']) @ axis
            angular = math.acos(float(np.clip(np.dot(actual, expected), -1., 1.)))
        else:
            angular = angle(obj, {'quaternion_wxyz': goal['target_quaternion_wxyz']})
        seats[goal['id']] = {
            'passed': bool(np.all(error <= np.asarray(goal['position_tolerance_m'])+1.e-7)
                           and angular <= goal['orientation_tolerance_rad']+1.e-7),
            'position_error_m': error.tolist(), 'orientation_error_rad': angular,
            'actor_local_point_m': local_point.tolist(),
            'actual_goal_point_m': actual_point.tolist(), 'target_goal_point_m': target_point.tolist()}
    for group in parameters.get('flexible_groups', []):
        ids = group['actor_ids']
        offsets = group.get('centerline_offsets_m', [[0., 0., 0.]]*len(ids))
        curve = np.asarray([np.asarray(objects[name]['position_m'])
            + rotation(objects[name]['quaternion_wxyz']) @ offset
            for name, offset in zip(ids, offsets)])
        line = np.concatenate((curve, curve[:1])) if group['closed'] else curve
        distances = np.linalg.norm(np.diff(line, axis=0), axis=1)
        routed, hits = route_in_order(line, group.get('route_gates', []))
        if not routed and group.get('allow_reverse_route', False):
            routed, hits = route_in_order(line[::-1], group.get('route_gates', []))
        loop = (not group['closed'] or simple_loop(curve)) and all(
            enclosed(curve, target, group.get('plane_tolerance_m', .02))
            for target in group.get('enclosed_targets', []))
        perimeter = float(distances.sum())
        limits = group.get('perimeter_range_m')
        length_ok = limits is None or limits[0]-1.e-7 <= perimeter <= limits[1]+1.e-7
        intact = float(distances.max()) <= group['maximum_neighbor_distance_m']+1.e-7
        groups[group['id']] = {'passed': bool(routed and loop and length_ok and intact),
            'ordered_route': routed, 'route_intersections': hits,
            'loop_encloses_targets': bool(loop), 'perimeter_m': perimeter,
            'length_ok': bool(length_ok), 'neighbor_distance_ok': bool(intact)}
    return {'passed': all(row['passed'] for row in [*seats.values(), *groups.values()]),
            'seating': seats, 'flexible_groups': groups,
            'scope': 'Geometry only; reset at goal cannot establish an operation or task success.'}


def mating_contact_checks(rows, metadata, *, maximum_gap_m=.0005):
    """Require native mating pairs; sleeping pairs may retain unchanged witnesses."""
    parameters = metadata['task_parameters']
    requirements = {
        'seat:'+goal['id']: [(goal['actor'], target) for target in goal['contact_ids']]
        for goal in parameters.get('seating_goals', [])}
    for group in parameters.get('flexible_groups', []):
        for target in group.get('required_mating_contact_ids', []):
            requirements['flexible:'+group['id']+':'+target] = [
                (actor, target) for actor in group['actor_ids']]
    paths = {**metadata.get('fixtures', {}), **metadata['actors']}
    ordered = sorted(paths.items(), key=lambda item: -len(item[1]))

    def identify(path):
        return next((name for name, root in ordered
                     if path == root or path.startswith(root+'/')), None)

    touching = {}
    for index, row in enumerate(rows):
        for contact in row.get('raw_contacts', []):
            separations = np.asarray(contact['separations_m'], float)
            if (separations.size and np.isfinite(separations).all()
                    and -.006 <= float(separations.min()) <= maximum_gap_m):
                pair = frozenset((identify(contact['actor0']), identify(contact['actor1'])))
                if None not in pair and len(pair) == 2:
                    touching.setdefault(pair, set()).add(index)
    fixed = {}
    for name, spec in metadata.get('fixture_metadata', {}).items():
        pose = spec['initial_pose_xyzw']
        fixed[name] = {'position_m': pose[:3], 'quaternion_wxyz': [pose[6], *pose[3:6]]}
    flexible = {name for group in parameters.get('flexible_groups', [])
                for name in group['actor_ids']}
    checks = {}
    for requirement, pairs in requirements.items():
        covered = np.zeros(len(rows), bool)
        witnesses = []
        for actor, target in pairs:
            seen = touching.get(frozenset((actor, target)), set())
            if not seen:
                continue
            bodies = [row['objects'][actor] for row in rows]
            supports = [row['objects'].get(target, fixed.get(target)) for row in rows]
            if any(item is None for item in supports):
                raise ValueError('Mating target lacks an explicit physical pose: '+target)
            positions = np.asarray([item['position_m'] for item in bodies])
            target_positions = np.asarray([item['position_m'] for item in supports])
            rotations = np.asarray([rotation(item['quaternion_wxyz']) for item in bodies])
            target_rotations = np.asarray([rotation(item['quaternion_wxyz']) for item in supports])
            relative = np.einsum('nji,nj->ni', target_rotations, positions-target_positions)
            relative_rotations = np.einsum('nji,njk->nik', target_rotations, rotations)
            # A current native manifold is sufficient. For omitted sleeping
            # contacts, both the actor and the actor-to-target pose must remain
            # within the same 1mm/.03rad witness bounds as the frozen referee.
            pair_covered = np.zeros(len(rows), bool)
            pair_covered[list(seen)] = True
            for witness in seen:
                unchanged = (np.linalg.norm(positions-positions[witness], axis=1) <= .001
                    ) & (np.linalg.norm(relative-relative[witness], axis=1) <= .001)
                if actor not in flexible:
                    cosine = (np.einsum('nij,ij->n', relative_rotations,
                                       relative_rotations[witness])-1.)/2.
                    unchanged &= np.arccos(np.clip(cosine, -1., 1.)) <= .03
                pair_covered |= unchanged
                if pair_covered.all():
                    break
            covered |= pair_covered
            witnesses.append({'actor': actor, 'target': target,
                'native_contact_samples': len(seen), 'covered_samples': int(pair_covered.sum())})
        checks[requirement] = {'passed': bool(len(rows) and covered.all()),
            'covered_samples': int(covered.sum()), 'samples': len(rows), 'pairs': witnesses}
    return {'passed': all(row['passed'] for row in checks.values()), 'requirements': checks,
        'maximum_native_manifold_gap_m': maximum_gap_m,
        'scope': 'Native mating manifold in this terminal window, or unchanged relative-pose witness; zero-load touching is allowed. No force calibration or operation claim.'}


def native_clock_check(*, tick, step, time_s, baseline_step, baseline_time,
                       previous_time, configured_dt):
    """Check native callback float32 dt, with only binary64 summation roundoff."""
    native_dt = float(np.float32(PHYSICS_DT))
    expected_time = baseline_time+tick*native_dt
    unit = math.ulp(max(1., abs(time_s), abs(baseline_time)))
    step_bound, total_bound = 8*unit, (tick+2)*8*unit
    detail = {'tick': tick, 'actual_native_step': step,
        'expected_native_step': baseline_step+tick, 'actual_native_time_s': time_s,
        'expected_native_time_s': expected_time, 'previous_native_time_s': previous_time,
        'configured_dt_s': configured_dt, 'declared_dt_s': PHYSICS_DT,
        'expected_callback_dt_s': native_dt, 'actual_increment_s': time_s-previous_time,
        'increment_error_s': time_s-previous_time-native_dt,
        'cumulative_error_s': time_s-expected_time,
        'increment_roundoff_bound_s': step_bound, 'cumulative_roundoff_bound_s': total_bound}
    detail['passed'] = bool(math.isfinite(time_s) and math.isfinite(configured_dt)
        and float(np.float32(configured_dt)) == native_dt
        and step == baseline_step+tick
        and abs(detail['increment_error_s']) <= step_bound
        and abs(detail['cumulative_error_s']) <= total_bound)
    return detail


def passive_anchor_checks(joints, *, tolerance_m=.003):
    errors = {name: row['anchor_separation_m'] for name, row in joints.items()
              if row.get('kind') in ('spherical', 'fixed')}
    fixed_angles = {name: math.acos(float(np.clip(
        (np.trace(np.asarray(row['relative_rotation_matrix']))-1.)/2., -1., 1.)))
        for name, row in joints.items() if row.get('kind') == 'fixed'}
    return {'passed': all(math.isfinite(value) and value <= tolerance_m for value in errors.values())
                     and all(value <= .15 for value in fixed_angles.values()),
            'maximum_anchor_error_m': max(errors.values(), default=0.),
            'tolerance_m': tolerance_m, 'anchor_errors_m': errors,
            'fixed_joint_orientation_errors_rad': fixed_angles}


def terminal_checks(rows, metadata, joint_info, masses, *, goal, mating_gap_m=.0005):
    actors = metadata['actors']
    groups = components(actors, joint_info)
    tail = list(rows)
    duration = tail[-1]['time_s']-tail[0]['time_s'] if tail else 0.
    complete = (len(tail) == round(DWELL_S/PHYSICS_DT)+1
        and all(abs(b['time_s']-a['time_s']-PHYSICS_DT) <= 1.e-8 for a, b in zip(tail, tail[1:])))
    result = {'complete_terminal_window': {'passed': complete and duration >= DWELL_S-1.e-8,
        'duration_s': duration, 'samples': len(tail)}, 'objects': {}, 'components': {}}
    if not tail:
        result['passed'] = False
        return result
    flexible = {name for group in metadata['task_parameters'].get('flexible_groups', [])
                for name in group['actor_ids']}
    flexible -= {goal_['actor'] for goal_ in metadata['task_parameters'].get('seating_goals', [])}
    for name in actors:
        values = [row['objects'][name] for row in tail]
        positions = np.asarray([item['position_m'] for item in values])
        excursion = float(np.max(np.linalg.norm(positions-positions[-1], axis=1)))
        angular_excursion = max(angle(item, values[-1]) for item in values)
        speeds = [np.linalg.norm(item['linear_velocity_m_s']) for item in values]
        rotations = [np.linalg.norm(item['angular_velocity_rad_s']) for item in values]
        result['objects'][name] = {
            'passed': bool(excursion <= .005 and max(speeds) <= .04
                           and (name in flexible or angular_excursion <= .15)),
            'terminal_position_excursion_m': excursion,
            'terminal_orientation_excursion_rad': angular_excursion,
            'native_peak_linear_speed_m_s': float(max(speeds)),
            'native_peak_angular_speed_rad_s': float(max(rotations)),
            'twist_free_link': name in flexible,
            'angular_velocity_is_reported_reference': True}
    for index, names in enumerate(groups):
        full_contact = all('component_total_external_force_n' in row for row in tail)
        force_field = 'component_total_external_force_n' if full_contact else 'component_external_force_n'
        loads = np.asarray([row[force_field][str(index)] for row in tail])
        mean = loads.mean(axis=0)
        normal_mean = np.asarray([row['component_external_force_n'][str(index)] for row in tail]).mean(axis=0)
        weight = sum(masses[name] for name in names)*9.81
        result['components'][str(index)] = {
            'passed': bool(mean[2] > 1.e-6),
            'actors': list(names), 'mean_signed_external_contact_force_n': normal_mean.tolist(),
            'legacy_external_field_semantics': 'Normal-only compatibility field; not complete external contact force.',
            'support_force_includes_friction': full_contact,
            'weight_n': weight, 'mean_vertical_force_over_weight': float(normal_mean[2]/weight),
            'force_calibration_claimed': False}
        if full_contact:
            result['components'][str(index)].update(
                mean_signed_total_external_contact_force_n=mean.tolist(),
                mean_total_vertical_force_over_weight=float(mean[2]/weight))
        for field, label in (('component_normal_force_n', 'normal'),
                             ('component_friction_force_n', 'friction')):
            if all(field in row for row in tail):
                result['components'][str(index)]['mean_signed_'+label+'_force_n'] = np.asarray(
                    [row[field][str(index)] for row in tail]).mean(axis=0).tolist()
    result['passive_links'] = {
        'passed': all(row['passive_links']['passed'] for row in tail),
        'maximum_terminal_anchor_error_m': max(row['passive_links']['maximum_anchor_error_m'] for row in tail)}
    result['fit'] = geometry_checks(metadata, tail[-1]['objects'])
    result['mating_contacts'] = mating_contact_checks(tail, metadata, maximum_gap_m=mating_gap_m)
    result['goal_fit_required'] = goal
    result['passed'] = bool(result['complete_terminal_window']['passed']
        and all(row['passed'] for row in result['objects'].values())
        and all(row['passed'] for row in result['components'].values())
        and result['passive_links']['passed']
        and (not goal or (result['fit']['passed'] and result['mating_contacts']['passed'])))
    return result


def contact_records(view, sensor_paths, filter_paths):
    impulse, point, normal, separation, counts, starts = [array(value) for value in view.get_contact_data(dt=1.)]
    rows = []
    for i, j in np.argwhere(counts > 0):
        start, count = int(starts[i, j]), int(counts[i, j])
        if start < 0 or start+count >= CAPACITY:
            raise ValueError('Native contact point capacity reached')
        selected = slice(start, start+count)
        f, p, n, d = [value[selected] for value in (impulse, point, normal, separation)]
        if not all(np.isfinite(value).all() for value in (f, p, n, d)):
            raise ValueError('Nonfinite native contact point')
        rows.append({'sensor_index': int(i), 'filter_index': int(j),
            'actor0': sensor_paths[i], 'actor1': filter_paths[j],
            'normal_impulses_ns': f.reshape(-1).tolist(),
            'points_m': p.reshape(-1, 3).tolist(), 'normals': n.reshape(-1, 3).tolist(),
            'separations_m': d.reshape(-1).tolist(),
            **summarize_normal_impulses(f.reshape(-1), n.reshape(-1, 3))})
    return rows


def friction_records(view, sensor_paths, filter_paths):
    """Copy all four native buffers before any other contact acquisition.

    dt=1 retains signed world-space friction impulses. The native API shares
    pair count/start buffers with get_contact_data, whose arrays must already
    have been copied by contact_records before this function is called.
    """
    impulse, point, counts, starts = [array(value) for value in view.get_friction_data(dt=1.)]
    shape = (len(sensor_paths), len(filter_paths))
    if counts.shape != shape or starts.shape != shape:
        raise ValueError('Native friction sensor/filter mapping differs')
    matrix = np.zeros((*shape, 3), dtype=float)
    rows = []
    for i, j in np.argwhere(counts > 0):
        start, count = int(starts[i, j]), int(counts[i, j])
        if start < 0 or start+count >= CAPACITY:
            raise ValueError('Native friction point capacity reached')
        selected = slice(start, start+count)
        vectors, points = impulse[selected], point[selected]
        if (vectors.shape != (count, 3) or points.shape != (count, 3)
                or not np.isfinite(vectors).all() or not np.isfinite(points).all()):
            raise ValueError('Invalid native friction vectors or points')
        summed = np.sum(vectors, axis=0, dtype=np.float64)
        matrix[i, j] = summed/PHYSICS_DT
        rows.append({'sensor_index': int(i), 'filter_index': int(j),
            'actor0': sensor_paths[i], 'actor1': filter_paths[j],
            'friction_impulses_ns': vectors.tolist(), 'points_m': points.tolist(),
            'impulse_vector_ns': summed.tolist()})
    return rows, matrix


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.mechanical_nist_interactions',
        'bench.simulation.isaacsim_backend.native_task_bundle',
        'scenes.workrooms.mechanical_assembly.nist_native_evidence'), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Run from a complete frozen project snapshot')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native evidence belongs under shared datasets')
    if not .5 <= args.duration <= 5. or abs(args.duration/PHYSICS_DT-round(args.duration/PHYSICS_DT)) > 1.e-8:
        raise ValueError('Bounded duration must be 0.5–5 seconds on the native grid')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mechanical_nist_object_interactions.v2', 'passed': False,
        'task_id': args.task_id, 'case': args.case, 'source_snapshot': str(PROJECT),
        'source': source, 'physics_dt_s': PHYSICS_DT, 'duration_s': args.duration,
        'robot_present': False, 'model_calls': 0, 'post_reset_pose_writes': 0,
        'external_force_calls': 0, 'task_success_claimed': False,
        'physics_accepted': False, 'full_functional_operation_qualified': False,
        'scope': 'Exact task bundle under gravity and passive constraints on the matching table support plane; fit/support only.'}
    app = None
    try:
        report['gpu'] = gpu_selection(args.gpu_uuid)
        _, scene, task, _ = task_inputs(args.release, args.task_id)
        if scene['isaacsim'].get('solver_type', 'TGS') != 'TGS':
            raise ValueError('Probe requires the selected TGS task configuration')
        metadata = task['asset_metadata']
        bundle = deepcopy(scene['native_task_bundle'])
        report['release'] = {'path': str(args.release.resolve()),
                            'manifest_sha256': digest(args.release/'release_manifest.json')}
        report['asset'] = {'path': bundle['usd_path'], 'sha256': bundle['usd_sha256']}
        source_hash = digest(bundle['usd_path'])
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 64, 'height': 64, 'multi_gpu': False,
            'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from bench.simulation.isaacsim_backend.scene import rigid_material
        from pxr import UsdShade
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=PHYSICS_DT, rendering_dt=.02,
            backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        physics = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        physics.CreateEnableCCDAttr(False)
        physics.CreateGpuMaxRigidContactCountAttr(2**20)
        physics.CreateGpuMaxRigidPatchCountAttr(2**18)
        physics.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        physics.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        cube = UsdGeom.Cube.Define(stage, TABLE_PATH)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(.65, 0., .76))
        cube.AddScaleOp().Set(Gf.Vec3f(1., 1., .04))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        material = rigid_material(stage, 'probe_table', .4)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, 'physics')
        info = add_bundle(stage, bundle, physics_scene_path=context.prim_path)
        if args.case == 'goal':
            poses = metadata['diagnostic_goal_poses']
            if set(poses) != set(metadata['actors']):
                raise ValueError('Goal probe requires an explicit reset pose for every task body')
            for name, pose in poses.items():
                values = np.asarray(pose, float)
                if values.shape != (7,) or not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1.) > 1.e-4:
                    raise ValueError('Goal reset requires finite world xyzw poses')
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(metadata['actors'][name]))
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*values[:3]))
                xform.AddOrientOp().Set(Gf.Quatf(float(values[6]), Gf.Vec3f(*values[3:6])))
            report['reset_only_goal_poses'] = poses
        dump(output/'metadata.json', metadata)
        stage.Export(str(output/'probe_initial.usda'))
        world.reset()
        sim = world.physics_sim_view
        body_view = sim.create_rigid_body_view(list(metadata['actors'].values()))
        body_paths = list(body_view.prim_paths)
        reverse = {path: name for name, path in metadata['actors'].items()}
        if body_view.count != len(reverse) or set(body_paths) != set(reverse):
            raise ValueError('Native body inventory is incomplete')
        names = [reverse[path] for path in body_paths]
        masses = dict(zip(names, array(body_view.get_masses()).reshape(-1).tolist()))
        if any(not math.isfinite(value) or value <= 0 for value in masses.values()):
            raise ValueError('Native mass inventory is invalid')
        fixed = info['fixtures'].get('native_bundle_fixed', {}).get('collision_paths', [])
        filters = [*metadata['actors'].values(), *fixed, TABLE_PATH]
        contact = sim.create_rigid_contact_view(list(metadata['actors'].values()),
            filter_patterns=[list(filters) for _ in metadata['actors']], max_contact_data_count=CAPACITY)
        sensors = list(contact.sensor_paths)
        mapped = [list(row) for row in contact.filter_paths]
        if set(sensors) != set(metadata['actors'].values()) or any(row != mapped[0] for row in mapped):
            raise ValueError('Native contact sensor/filter mapping differs')
        filters = mapped[0]
        if set(filters) != set([*metadata['actors'].values(), *fixed, TABLE_PATH]):
            raise ValueError('Native contact filters are incomplete')
        sensor_names = [reverse[path] for path in sensors]
        groups = components(metadata['actors'], info['joints'])
        report['contact_mapping'] = {'sensor_paths': sensors, 'filter_paths': filters,
            'sparse_matrix_semantics': 'Unlisted sensor/filter entries are exact zero vectors. All signs retained.',
            'contact_force_matrix_semantics': 'Native normal-contact matrix; independently retained as originally reported.',
            'legacy_component_external_force_n_semantics': 'Normal-only compatibility field, identical to component_normal_force_n.',
            'friction_semantics': 'Native get_friction_data(dt=1) signed world-space impulses, divided by the same declared dt as the normal matrix.',
            'support_force_semantics': 'Complete signed external normal plus friction vector sum; internal joint-connected component pairs excluded.'}
        report['native_masses_kg'] = masses
        report['joint_connected_components'] = [list(group) for group in groups]
        baseline_step, baseline_time = int(world.current_time_step_index), float(world.current_time)
        configured_dt = float(world.get_physics_dt())
        report['native_clock'] = {'baseline_step': baseline_step, 'baseline_time_s': baseline_time,
            'configured_dt_s': configured_dt, 'declared_dt_s': PHYSICS_DT,
            'callback_dt_s': float(np.float32(PHYSICS_DT)),
            'clock_source': 'World physics callback accumulates its float32 step_size in binary64; actual times are retained.'}
        previous_time = baseline_time
        tail = deque(maxlen=round(DWELL_S/PHYSICS_DT)+1)
        max_penetration = 0.
        with (output/'native_frames.jsonl').open('x') as stream:
            for tick in range(1, round(args.duration/PHYSICS_DT)+1):
                world.step(render=False)
                native_step, native_time = int(world.current_time_step_index), float(world.current_time)
                clock_check = native_clock_check(tick=tick, step=native_step, time_s=native_time,
                    baseline_step=baseline_step, baseline_time=baseline_time,
                    previous_time=previous_time, configured_dt=float(world.get_physics_dt()))
                if not clock_check['passed']:
                    report['native_clock_failure'] = clock_check
                    dump(output/'native_clock_failure.json', clock_check)
                    raise ValueError('Missing or inconsistent native sampling step: '+json.dumps(clock_check))
                previous_time = native_time
                objects = {}
                poses = array(body_view.get_transforms()).reshape(len(names), 7)
                velocities = array(body_view.get_velocities()).reshape(len(names), 6)
                if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                    raise ValueError('Nonfinite native body state')
                for name, pose, velocity in zip(names, poses, velocities):
                    objects[name] = {'position_m': pose[:3].tolist(),
                        'quaternion_wxyz': [float(pose[6]), *pose[3:6].tolist()],
                        'linear_velocity_m_s': velocity[:3].tolist(),
                        'angular_velocity_rad_s': velocity[3:].tolist()}
                matrix = array(contact.get_contact_force_matrix(dt=PHYSICS_DT))
                if matrix.shape != (len(sensors), len(filters), 3) or not np.isfinite(matrix).all():
                    raise ValueError('Native contact matrix shape/values changed')
                raw = contact_records(contact, sensors, filters)
                raw_friction, friction_matrix = friction_records(contact, sensors, filters)
                total_matrix = matrix.astype(np.float64)+friction_matrix
                max_penetration = max(max_penetration, max(
                    [-min(row['separations_m']) for row in raw], default=0.))
                joint_state = joint_positions(objects, info['joints'])
                nonzero = np.argwhere(np.any(matrix != 0., axis=2))
                friction_nonzero = np.argwhere(np.any(friction_matrix != 0., axis=2))
                total_nonzero = np.argwhere(np.any(total_matrix != 0., axis=2))
                row = {'native_step': native_step, 'native_time_s': native_time,
                    'time_s': native_time-baseline_time, 'nominal_time_s': tick*PHYSICS_DT,
                    'native_increment_s': clock_check['actual_increment_s'],
                    'native_clock_cumulative_error_s': clock_check['cumulative_error_s'],
                    'objects': objects, 'joints': joint_state,
                    'passive_links': passive_anchor_checks(joint_state),
                    'raw_contacts': raw,
                    'raw_friction': raw_friction,
                    'contact_matrix_nonzero': [[int(i), int(j), *matrix[i, j].tolist()] for i, j in nonzero],
                    'friction_matrix_nonzero': [[int(i), int(j), *friction_matrix[i, j].tolist()]
                                               for i, j in friction_nonzero],
                    'total_contact_matrix_nonzero': [[int(i), int(j), *total_matrix[i, j].tolist()]
                                                     for i, j in total_nonzero],
                    'component_normal_force_n': {str(i): signed_external_load(
                        matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_friction_force_n': {str(i): signed_external_load(
                        friction_matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_external_force_n': {str(i): signed_external_load(
                        matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_total_external_force_n': {str(i): signed_external_load(
                        total_matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)}}
                if (int(world.current_time_step_index), float(world.current_time)) != (native_step, native_time):
                    raise ValueError('Observation unexpectedly advanced native time')
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                tail.append(row)
                if max_penetration > .006:
                    raise ValueError('Native penetration exceeds the declared practical 6 mm bound')
        report['checks'] = terminal_checks(tail, metadata, info['joints'], masses, goal=args.case == 'goal',
            mating_gap_m=task['public_rules']['referee']['thresholds'].get('maximum_mating_gap_m', .0005))
        report['maximum_reported_penetration_m'] = max_penetration
        report['native_substeps'] = round(args.duration/PHYSICS_DT)
        report['source_asset_unchanged'] = digest(bundle['usd_path']) == source_hash
        report['passed'] = bool(report['checks']['passed'] and report['source_asset_unchanged'])
    except BaseException as exc:
        traceback.print_exc()
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report['files_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', choices=TASK_IDS, required=True)
    parser.add_argument('--case', choices=('canonical', 'goal'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=1.)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
