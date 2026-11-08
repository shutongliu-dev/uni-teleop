"""Same-tick native contact API comparisons; no force corrections or task credit."""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np
from bench.isaacsim.release import task_inputs, digest, dump
from bench.paths import OUTPUT_ROOT
from bench.simulation.isaacsim_backend.contact_sample import copy_array
from bench.evaluation.cloth.cloth_robot_collision import state_digest
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]


def raw_snapshot(view):
    values = view.get_raw_contact_data(dt=1.)
    arrays = [copy_array(x) for x in values]
    counts, starts = arrays[4:6]
    used = max((int(s)+int(c) for s, c in zip(starts, counts) if int(c)>0), default=0)
    if used >= view.max_contact_data_count:
        raise ValueError('Raw contact capacity reached')
    ids = values[6][:used]
    if hasattr(ids, 'detach'):
        ids = ids.detach().cpu().clone()
    else:
        import warp as wp
        ids = wp.array(arrays[6][:used], dtype=wp.uint64, device='cpu')
    mapping_error = None
    try:
        paths = list(view.get_other_actor_paths_from_ids(ids)) if used else []
        if len(paths) != used:
            raise ValueError('Incomplete native other-actor mapping')
    except Exception as exc:
        mapping_error = type(exc).__name__+': '+str(exc)
        paths = [None]*used
    out = dict(zip(('raw_scalar', 'raw_points', 'raw_normals', 'raw_separations'), [x[:used] for x in arrays[:4]]))
    out.update(raw_counts=counts, raw_starts=starts, raw_other_ids=arrays[6][:used])
    sensors = []
    for index, (start, count) in enumerate(zip(starts, counts)):
        start, count = int(start), int(count)
        f = out['raw_scalar'][start:start+count].reshape(-1).astype(float)
        normal = out['raw_normals'][start:start+count].astype(float)
        actors = paths[start:start+count]
        if count and any(not x for x in actors) and mapping_error is None:
            mapping_error = 'Missing actual contacting actor path; raw IDs are retained'
        grouped = {}
        for force, n, actor in zip(f, normal, actors):
            if actor:
                grouped.setdefault(actor, np.zeros(3))
                grouped[actor] += force*n
        sensors.append({'sensor': view.sensor_paths[index], 'points': count,
                        'normal_impulse_ns': (f[:, None]*normal).sum(0).tolist(),
                        'by_other_actor_ns': {k: v.tolist() for k, v in grouped.items()}})
    return out, {'other_actor_paths': paths, 'raw_sensors': sensors, 'actor_mapping_error': mapping_error}


def capture(view, output):
    try:
        arrays, details = raw_snapshot(view)
    except Exception as exc:
        arrays, details = {}, {'raw_api_error': type(exc).__name__+': '+str(exc)}
    arrays['net_impulses'] = copy_array(view.get_net_contact_forces(dt=1.))
    if view.filter_count:
        values = tuple(copy_array(x) for x in view.get_contact_data(dt=1.))
        counts, starts = values[-2:]
        active = np.argwhere(counts > 0)
        used = max((int(starts[i, j])+int(counts[i, j]) for i, j in active), default=0)
        if used >= view.max_contact_data_count:
            raise ValueError('Filtered contact capacity reached')
        arrays.update(dict(zip(('filtered_scalar', 'filtered_points', 'filtered_normals', 'filtered_separations'),
                               [x[:used] for x in values[:4]])))
        arrays.update(filtered_counts=counts, filtered_starts=starts,
                      pair_impulses=copy_array(view.get_contact_force_matrix(dt=1.)))
        details['filtered_pairs'] = [{'sensor_index': int(i), 'filter_index': int(j),
            'points': int(counts[i,j]), 'normal_impulse_ns':
            (values[0][int(starts[i,j]):int(starts[i,j])+int(counts[i,j])].reshape(-1,1).astype(float)*
             values[2][int(starts[i,j]):int(starts[i,j])+int(counts[i,j])].astype(float)).sum(0).tolist()}
            for i,j in active]
    np.savez_compressed(output, **arrays)
    details.update(sensor_paths=list(view.sensor_paths), filter_paths=list(view.filter_paths),
                   sensor_count=int(view.sensor_count), filter_count=int(view.filter_count),
                   net_impulses_ns=arrays['net_impulses'].tolist(), arrays_sha256=digest(output))
    return details


def run(args):
    out = args.output.resolve()
    if not out.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Use shared evidence storage')
    out.mkdir(parents=True, exist_ok=False)
    env = subscription = event_stream = None
    report = {'schema': 'compound_contact_api_comparison.v1', 'diagnostic_completed': False,
              'physics_accepted': False, 'controller_changes': False, 'force_corrections': False}
    try:
        source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.compound_contact_probe',),
                                    entrypoint_file=Path(__file__))
        if source is None or Path.cwd().resolve() != PROJECT:
            raise ValueError('Run from a complete sealed source')
        report.update(source=source, source_manifest_sha256=digest(PROJECT/'source_snapshot_manifest.json'),
                      release=str(args.release.resolve()), release_manifest_sha256=digest(args.release/'release_manifest.json'))
        _, scene, task, initial = task_inputs(args.release, args.task_id)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', headless=True)
        # Explicit diagnostic notification API, authored before reset; no
        # collision, mass, material, state or controller parameter changes.
        from pxr import PhysxSchema
        notification = {}
        for row in task['asset_metadata']['objects']:
            if row['role'] not in ('spoon', 'fork', 'knife'):
                continue
            prim = env.stage.GetPrimAtPath(row['body_path'])
            old_schemas = list(prim.GetAppliedSchemas())
            old_threshold = prim.GetAttribute('physxContactReport:threshold').Get()
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            notification[row['body_path']] = {'prior_schemas': old_schemas,
                'prior_threshold': old_threshold, 'threshold': 0., 'before_reset': True}
        report['contact_notification_configuration'] = notification
        observation = env.reset(seed=0, initial_state=initial)
        dump(out/'runtime_metadata.json', env.metadata())
        from pxr import UsdUtils, PhysicsSchemaTools
        stage_id = int(UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt())
        meta = task['asset_metadata']
        targets = {row['id']: row for row in meta['objects'] if row['role'] in ('spoon', 'fork', 'knife')}
        selected_paths = {row['body_path'] for row in targets.values()}
        views = {'original': env._rigid_contacts}
        report['view_creation'] = {}
        for name, row in targets.items():
            support = meta['source_support_by_actor'][name]
            support = meta['fixtures'].get(support, support)
            sensor_index = env._contact_sensor_paths.index(row['body_path'])
            support_leaves = [p for p in env._contact_filter_paths[sensor_index]
                              if p == support or p.startswith(support+'/')]
            if not support_leaves:
                raise ValueError('No actual native filter leaves for source support')
            for kind, sensor, filters in (
                    ('single_raw', row['body_path'], []),
                    ('single_filtered', row['body_path'], support_leaves)):
                key = name+'_'+kind
                try:
                    view = env._sim_view.create_rigid_contact_view(sensor, filter_patterns=filters,
                                                                  max_contact_data_count=32768)
                    report['view_creation'][key] = {'sensor_paths': list(view.sensor_paths),
                                                   'filter_paths': list(view.filter_paths)}
                    if view.sensor_count != 1:
                        raise ValueError('Expected a single native sensor')
                    views[key] = view
                except Exception as exc:
                    report['view_creation'][key] = {'unsupported': str(exc)}
        event_stream = (out/'contact_events.jsonl').open('x')
        event_stats = {'callbacks': 0, 'matching_stage_callbacks': 0, 'saved_pairs': 0, 'errors': []}
        def contact_event(headers, contacts, friction_anchors):
            event_stats['callbacks'] += 1
            try:
                relevant = [h for h in headers if int(h.stage_id) == stage_id]
                if relevant:
                    event_stats['matching_stage_callbacks'] += 1
                for h in relevant:
                    actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0, h.actor1)]
                    if not selected_paths.intersection(actors):
                        continue
                    offset, count = int(h.contact_data_offset), int(h.num_contact_data)
                    data = [{'position': list(c.position), 'normal': list(c.normal),
                             'impulse': list(c.impulse), 'separation': float(c.separation)}
                            for c in (contacts[i] for i in range(offset, offset+count))]
                    row = {'native_step_at_callback': int(env.world.current_time_step_index),
                           'native_time_at_callback': float(env.world.current_time),
                           'planned_public_time_s': float(env._current_step_time), 'stage_id': int(h.stage_id),
                           'event_type': str(h.type), 'actors': actors,
                           'actor_ids': [int(h.actor0), int(h.actor1)],
                           'colliders': [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.collider0, h.collider1)],
                           'contact_offset': offset, 'contact_count': count, 'contacts': data,
                           'friction_anchor_count': int(h.num_friction_anchors_data),
                           'friction_anchors_captured': False}
                    event_stream.write(json.dumps(row, allow_nan=False)+'\n')
                    event_stats['saved_pairs'] += 1
            except Exception as exc:
                if len(event_stats['errors']) < 10:
                    event_stats['errors'].append(type(exc).__name__+': '+str(exc))
        try:
            from omni.physics.core import get_physics_simulation_interface
            subscription = get_physics_simulation_interface().subscribe_physics_contact_report_events(contact_event)
            report['event_subscription'] = 'multi_stage_omni_physics_core'
        except Exception as exc:
            report['event_subscription_error'] = str(exc)
        report['native_mass_properties'] = {}
        for name, row in targets.items():
            view = env._object_views[name]
            report['native_mass_properties'][name] = {'declared_mass_kg': row['mass_kg'],
                'masses': copy_array(view.get_masses()).tolist(), 'coms': copy_array(view.get_coms()).tolist(),
                'inertias': copy_array(view.get_inertias()).tolist(), 'max_shapes': int(view.max_shapes)}
        snapshots = []
        states = []
        action = env.hold_current_targets()
        for tick in range(round(args.duration/.02)):
            observation = env.step(action)
            states.append({'time_s': float(env.time_s), 'objects': observation['objects']})
            if env.time_s < args.duration-.201:
                continue
            before = state_digest(env.capture_state())
            clock = (int(env.world.current_time_step_index), float(env.world.current_time))
            sample = {'native_step': clock[0], 'native_time_s': clock[1], 'public_time_s': float(env.time_s), 'views': {}}
            for key, view in views.items():
                try:
                    path = out/f'contact_{tick:03d}_{key}.npz'
                    sample['views'][key] = capture(view, path)
                except Exception as exc:
                    sample['views'][key] = {'error': type(exc).__name__+': '+str(exc)}
            sample['read_only'] = before == state_digest(env.capture_state()) and clock == (
                int(env.world.current_time_step_index), float(env.world.current_time))
            if not sample['read_only']:
                raise ValueError('Contact API comparison changed state/time')
            snapshots.append(sample)
            print(json.dumps({'time_s': env.time_s, 'captured_views': len(views), 'event_pairs': event_stats['saved_pairs']}), flush=True)
        report.update(diagnostic_completed=True, snapshots=snapshots, contact_events=event_stats,
                      actual_sim_s=float(env.time_s), native_control_endpoints=len(states),
                      scope='One unchanged native stage; original many-body view, separate single-body raw and single-filter views plus optional official multi-stage events. Raw impulses are retained. This diagnoses APIs, not material/force qualification.')
        dump(out/'object_states.json', states)
        verify_if_snapshot(PROJECT, required_modules=('scripts.validation.compound_contact_probe',), entrypoint_file=Path(__file__))
    except BaseException:
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        subscription = None
        if event_stream is not None:
            event_stream.close()
        report['evidence_sha256'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
        report['shutdown_scope'] = 'Evidence saved before Kit fast shutdown; exact process exit is in external launch receipts.'
        dump(out/'report.json', report)
        if env is not None:
            try:
                env.close()
            except BaseException:
                dump(out/'close_exception.json', {'exception': traceback.format_exc()})
    return 0 if report['diagnostic_completed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--task-id', default='utensil_sorting')
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
