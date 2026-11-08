"""Explicit checkpoint-derived, ordinary-joint palm pressing diagnostic.

This is a new zero-velocity authoring reset, never a resumed folding episode.
Only robot joint targets are commanded after initialization. Rest geometry,
material, collision, shared drives and the original task contract are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_single_garment_native import (
    DATASETS, NativeArrays, advance, dump, pin, robot_readback,
)

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = 'laundry_checkpoint_ordinary_palm_press.v1'


def read_pinned(record):
    if pin(record['path']) != record:
        raise ValueError('Pinned input changed: ' + record['path'])
    return load_json(Path(record['path']))


def command_rows(plan, names):
    """Collapse only identical repeated boundaries, retaining the new label."""
    if plan.get('error') is not None or not plan['source_clearance']['all_clear']:
        raise ValueError('Ordinary original-geometry CPU path must pass')
    if plan.get('attachments') is not False or plan.get('cloth_state_writes') != 0:
        raise ValueError('Pressing must use ordinary robot motion only')
    rows = {}
    for source in plan['samples']:
        t = source['time_s']
        q = source['joint_positions_rad']
        if (isinstance(t, bool) or not np.isfinite(t) or t < 0
                or abs(round(t / .02) * .02 - t) > 1.e-8
                or set(q) != set(names) or not np.isfinite(list(q.values())).all()):
            raise ValueError('Finite complete 20ms joint references required')
        index = round(t / .02)
        if index in rows and rows[index]['joint_positions_rad'] != q:
            raise ValueError('A phase boundary changed its joint command')
        rows[index] = deepcopy(source)
    count = round(plan['planned_duration_s'] / .02)
    if sorted(rows) != list(range(count + 1)):
        raise ValueError('Press command clock is incomplete')
    return [rows[i] for i in range(count + 1)]


def derive_reset(scene, initial, cloth_q, joint_q):
    """Author a fresh shape while retaining every material/rest/robot option."""
    if (scene.get('named_cloth_initialization') != 'author-only'
            or len(scene.get('cloths', [])) != 1):
        raise ValueError('One named author-only garment required')
    q = np.asarray(cloth_q)
    old = np.asarray(scene['cloths'][0]['initial_vertices_m'])
    if q.shape != old.shape or not np.isfinite(q).all():
        raise ValueError('Checkpoint must match the original finite material mesh')
    if (set(joint_q) != set(initial['joint_positions_rad'])
            or not np.isfinite(list(joint_q.values())).all()):
        raise ValueError('Checkpoint must contain every original robot joint')
    result, state = deepcopy(scene), deepcopy(initial)
    name = result['cloths'][0]['id']
    result['cloths'][0]['initial_vertices_m'] = q.tolist()
    state['cloths_vertices_m'] = {name: q.tolist()}
    state['joint_positions_rad'] = dict(joint_q)
    return result, state


def original_design_bounds(reference, context, expected_pairs):
    """Bind small imported thumb/palm overlaps to measured start and original park."""
    if (context is None or reference.get('schema') != 'laundry_original_design_contact_bounds.v1'
            or reference.get('source_binding') != context['source_binding']
            or set(reference.get('reference_states', {})) != set(context['joint_states'])):
        raise ValueError('Original design-contact references must bind this exact source and both robot states')
    for name in ('native_report','cooked_export'):
        record=context['source_binding'].get(name)
        if not isinstance(record,dict) or pin(record['path']) != record:
            raise ValueError('Original design-contact native/cooked source changed')
    calculation=reference.get('calculation_implementation')
    if not isinstance(calculation,dict) or pin(calculation['path']) != calculation:
        raise ValueError('Original design-contact calculation changed')
    bounds={name:0. for name in expected_pairs}
    for label, joints in context['joint_states'].items():
        record=reference['reference_states'][label]
        pairs=record.get('intersections_by_pair',{})
        if (record.get('joint_positions_rad') != joints
                or record.get('all_non_design_pairs_clear') is not True
                or set(pairs) != expected_pairs):
            raise ValueError('Design contact may use only the original start/park poses and known pairs')
        for name, value in pairs.items():
            radius=value.get('maximum_radius_m')
            if type(radius) not in (int,float) or not np.isfinite(radius) or not 0. <= radius < .001:
                raise ValueError('Original design overlaps must remain below the unchanged1mm cap')
            bounds[name]=max(bounds[name],radius)
    return bounds


def validate_internal_review(internal, sample_count, *, reference_context=None):
    results = internal.get('results')
    if not isinstance(results, list) or not results:
        raise ValueError('A nonempty independent cooked review is required')
    expected_design = {'left_thumb_MC|left_hand_C_MC', 'right_thumb_MC|right_hand_C_MC'}
    bounds=None
    if 'design_contact_reference' in internal:
        reference=read_pinned(internal['design_contact_reference'])
        bounds=original_design_bounds(reference,reference_context,expected_design)
    for result in results:
        rows = result.get('rows')
        if (type(result.get('sample_count')) is not int
                or result['sample_count'] != sample_count
                or not isinstance(rows, list) or len(rows) != sample_count
                or any(type(row.get('sample_index')) is not int for row in rows)
                or sorted(row['sample_index'] for row in rows) != list(range(sample_count))):
            raise ValueError('Internal cooked review omitted or duplicated path samples')
        if set(result['intersections_by_pair']) != expected_design:
            raise ValueError('Unreviewed hand/arm intersection')
        for name,item in result['intersections_by_pair'].items():
            radius=item['maximum_radius_m']
            if (type(radius) not in (int,float) or not np.isfinite(radius) or not 0.<=radius<.001
                    or type(item['worst_sample']) is not int or not 0<=item['worst_sample']<sample_count
                    or (item['worst_sample'] != 0 if bounds is None else radius>bounds[name]+1.e-9)):
                raise ValueError('Hand design overlap increased during the path')


def validate_authoring_source(manifest, implementation, source_manifest):
    if (manifest.get('authoring_implementation') != pin(implementation)
            or manifest.get('authoring_source_manifest') != pin(source_manifest)):
        raise ValueError('Diagnostic authoring must bind the current frozen implementation and source')


def validate_source_sample(native, sample, *, allow_incomplete=False):
    """Authorize a new diagnostic reset from a real recorded sample only."""
    if type(allow_incomplete) is not bool or type(native.get('complete')) is not bool:
        raise ValueError('Source execution status and incomplete-prefix selection must be explicit booleans')
    if not native['complete'] and not allow_incomplete:
        raise ValueError('An incomplete source requires explicit --allow-incomplete-source')
    count=native['states']['valid_samples']
    if type(count) is not int or type(sample) is not int or not 0 <= sample < count:
        raise ValueError('Invalid original native sample; allocated padding is not recorded evidence')
    if native['complete']:
        return None
    return {'original_native_complete': False, 'original_error': deepcopy(native.get('error')),
            'recorded_prefix_samples': count, 'selected_recorded_sample': sample,
            'explicit_incomplete_source_selection': True,
            'scope': 'The original failed/incomplete run remains unchanged. Only this recorded, '
                     'finite, released sample authors a new zero-velocity diagnostic reset; '
                     'neither completion nor a successful task is inherited from its prefix.'}


def prepare(args):
    report_path, plan_path = args.native_report.resolve(), args.plan.resolve()
    native, plan = [load_json(p) for p in (report_path, plan_path)]
    if plan['native_report'] != pin(report_path):
        raise ValueError('Plan must bind the exact original native stream')
    sample = plan['source_sample']
    source_status=validate_source_sample(native,sample,
        allow_incomplete=getattr(args,'allow_incomplete_source',False))
    fields = {}
    for name in ('cloth_q', 'cloth_qd', 'joint_q', 'joint_qd', 'joint_target',
                 'robot_body_q', 'control_mask', 'assist_enabled', 'time_s', 'native_step'):
        record = native['states']['files'][name]
        if pin(record['path']) != {k: record[k] for k in ('path', 'sha256')}:
            raise ValueError('Original native stream changed: ' + name)
        fields[name] = np.array(np.load(record['path'], mmap_mode='r')[sample], copy=True)
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise ValueError('Selected recorded checkpoint must be finite in every retained field')
    if fields['control_mask'].any() or fields['assist_enabled'].any():
        raise ValueError('Press checkpoint still has controlled material')
    if fields['native_step'] != sample or abs(float(fields['time_s']) - plan['source_time_s']) > 1.e-9:
        raise ValueError('Original source clock changed')
    q = fields['cloth_q']
    if hashlib.sha256(q.astype('<f8').tobytes()).hexdigest() != plan['source_q_sha256']:
        raise ValueError('CPU path and actual cloth checkpoint differ')
    names = native['joint_names']
    joints = dict(zip(names, map(float, fields['joint_q'])))
    rows = command_rows(plan, names)
    if rows[0]['joint_positions_rad'] != joints:
        raise ValueError('CPU path does not start at the measured robot state')
    scene, initial, task = [read_pinned(native['inputs'][n]) for n in
                            ('scene.json', 'initial_state.json', 'task.json')]
    geometry = read_pinned(pin(args.static_review))
    speed = read_pinned(pin(args.velocity_review))
    internal = read_pinned(pin(args.internal_review))
    if (geometry['plan'] != pin(plan_path) or not geometry['all_passed']
            or speed['input'] != pin(plan_path) or not speed['passed']
            or internal['input_path'] != pin(plan_path)):
        raise ValueError('Independent CPU review does not bind this exact plan')
    reference_context={'source_binding':{'native_report':pin(report_path),'source_sample':sample,
                                        'cooked_export':plan.get('cooked_export')},
                       'joint_states':{'recorded_native_start':joints,
                                       'original_release_initial':initial['joint_positions_rad']}}
    if ('design_contact_reference' in internal
            and plan.get('cooked_export') != native.get('robot_colliders')):
        raise ValueError('Original design-contact geometry must be the recorded native cooked export')
    validate_internal_review(internal, len(plan['samples']),reference_context=reference_context)
    derived_scene, derived_initial = derive_reset(scene, initial, q, joints)
    out = args.output.resolve()
    if not out.is_relative_to(DATASETS) or out.exists():
        raise ValueError('Use a fresh shared diagnostic input directory')
    out.mkdir(parents=True)
    dump(out/'scene.json', derived_scene)
    dump(out/'initial_state.json', derived_initial)
    dump(out/'task.json', task)
    np.savez(out/'observed_checkpoint.npz', **fields)
    manifest = {
        'schema': SCHEMA, 'pre_settle_s': .5, 'final_free_s': 3.,
        'authoring_implementation': pin(__file__),
        'authoring_source_manifest': (pin(PROJECT/'source_snapshot_manifest.json')
                                      if (PROJECT/'source_snapshot_manifest.json').is_file() else None),
        'original_native_report': pin(report_path), 'original_release': native['inputs']['release_manifest.json'],
        'original_source_manifest': native['source_manifest'], 'source_sample': sample,
        'source_time_s': float(fields['time_s']), 'source_streams': native['states']['files'],
        'source_checkpoint': pin(out/'observed_checkpoint.npz'), 'plan': pin(plan_path),
        'inputs': {n: pin(out/n) for n in ('scene.json', 'initial_state.json', 'task.json')},
        'independent_cpu_reviews': {'static_cloth': pin(args.static_review),
                                    'joint_velocity': pin(args.velocity_review),
                                    'cooked_internal': pin(args.internal_review)},
        'reset_semantics': 'new_authored_shape_and_robot_q_with_explicit_zero_initial_velocities',
        'zero_velocity_difference': {
            'cloth_speed_p95_m_s': float(np.quantile(np.linalg.norm(fields['cloth_qd'], axis=1), .95)),
            'cloth_speed_max_m_s': float(np.linalg.norm(fields['cloth_qd'], axis=1).max()),
            'robot_joint_speed_max_rad_s': float(np.abs(fields['joint_qd']).max())},
        'runtime_cloth_state_writes': 0, 'attachments': False, 'task_success_claimed': False,
        'scope': 'A new diagnostic reset derived from an observed released shape. Velocities are '
                 'explicitly zero before native initialization and a free settling window; source velocities '
                 'are preserved in the checkpoint. This is not exact continuation or stitched full-task evidence.'}
    if source_status is not None:
        manifest['source_execution_status']=source_status
    if 'design_contact_reference' in internal:
        manifest['independent_cpu_reviews']['design_reference']=internal['design_contact_reference']
    dump(out/'probe_manifest.json', manifest)
    return out/'probe_manifest.json'


def run(args):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.laundry_physical_press_probe',), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Run only a complete immutable source snapshot')
    manifest = read_pinned(pin(args.manifest))
    validate_authoring_source(manifest, __file__, PROJECT/'source_snapshot_manifest.json')
    if manifest['schema'] != SCHEMA or manifest['attachments'] is not False:
        raise ValueError('Explicit ordinary press diagnostic required')
    plan = read_pinned(manifest['plan'])
    scene, initial, task = [read_pinned(manifest['inputs'][n]) for n in
                            ('scene.json', 'initial_state.json', 'task.json')]
    for review in manifest['independent_cpu_reviews'].values():
        read_pinned(review)
    rows = command_rows(plan, list(initial['joint_positions_rad']))
    if rows[0]['joint_positions_rad'] != initial['joint_positions_rad']:
        raise ValueError('Authored robot reset differs from the reviewed plan start')
    dt = float(scene['physics']['dt_s'])
    control_steps = round(.02 / dt)
    if abs(control_steps * dt - .02) > 1.e-9:
        raise ValueError('20ms must contain whole native steps')
    before, after = manifest['pre_settle_s'], manifest['final_free_s']
    if (not all(type(v) in (float, int) and np.isfinite(v) and .5 <= v <= 10.
                and abs(round(v / .02) * .02 - v) < 1.e-9 for v in (before, after))
            or not np.isfinite(args.media_fps) or not 0 < args.media_fps <= 30):
        raise ValueError('Finite aligned free windows and positive bounded media rate required')
    duration = before + plan['planned_duration_s'] + after
    steps = round(duration / dt)
    out = args.output.resolve()
    if not out.is_relative_to(DATASETS) or out.exists():
        raise ValueError('Use a fresh native output directory')
    out.mkdir(parents=True)
    report = {'schema': 'laundry_ordinary_palm_press_native.v1', 'complete': False,
        'source': source, 'source_manifest': pin(PROJECT/'source_snapshot_manifest.json'),
        'probe_manifest': pin(args.manifest), 'inputs': manifest['inputs'], 'plan': manifest['plan'],
        'original_native_report': manifest['original_native_report'], 'reset_semantics': manifest['reset_semantics'],
        'runtime_cloth_position_velocity_writes': 0, 'attachments_created': 0,
        'controlled_cloth_vertices': 0, 'collision_filtering': False,
        'force_controlled': False, 'task_success_claimed': False, 'full_task_qualified': False,
        'scope': 'Ordinary joint targets and real hand/cloth collision in a separate checkpoint-derived '
                 'zero-velocity reset. No cloth controls, capture credit, full folding or force qualification.'}
    env = writer = capture = contacts = None
    exit_code = 1
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        from bench.evaluation.cloth.cloth_robot_collision import collect_colliders
        from scripts.validation.laundry_material_settle import mesh_data, usd_material_readback
        env = FR3DexterousIsaacEnv(scene)
        env.reset(initial_state=initial)
        if env.dt != dt:
            raise ValueError('Native clock differs from declared clock')
        spec = scene['cloths'][0]
        cloth = env._cloth_views[spec['id']]
        report.update(joint_names=env.joint_names, body_names=env.body_names,
                      native_initialization=env.named_cloth_initialization_receipt,
                      actual_usd_material=usd_material_readback(env, spec, mesh_data(spec, task)))
        attached = [str(p.GetPath()) for p in env.stage.Traverse()
                    if any('Attachment' in schema for schema in p.GetAppliedSchemas())]
        if attached:
            raise ValueError('Ordinary press scene contains an attachment')
        report['actual_attachment_prims'] = attached
        dump(out/'robot_colliders.json', collect_colliders(env))
        report['robot_colliders'] = pin(out/'robot_colliders.json')
        writer = NativeArrays(out/'states', steps + 1)
        contacts = (out/'rigid_contacts.jsonl').open('x')
        env._begin_contact_sampling()
        start_step, start_time = env.world.current_time_step_index, env.world.current_time
        labels = ['pre_press_free_settle', *dict.fromkeys(r['label'] for r in rows), 'post_press_free']
        report['actions'] = labels
        pairs, impulse, action_index = 0, 0., 0

        def record(step):
            q, qd = _np(cloth.get_simulation_nodal_positions())[0], _np(cloth.get_simulation_nodal_velocities())[0]
            return {**robot_readback(env), 'cloth_q': q, 'cloth_qd': qd,
                'control_mask': np.zeros(len(q), np.uint8), 'assist_enabled': np.zeros(2, np.uint8),
                'joint_target': env._targets[0].copy(), 'native_step': np.asarray(step, np.int64),
                'time_s': np.asarray(step * dt), 'world_step': np.asarray(env.world.current_time_step_index, np.int64),
                'world_time_s': np.asarray(env.world.current_time), 'action_index': np.asarray(action_index, np.int64),
                'rigid_contact_pair_count': np.asarray(pairs, np.int64), 'rigid_contact_impulse_ns': np.asarray(impulse)}

        writer.append(record(0))
        env.stage.Export(str(out/'initial.usda'))
        report['initial_stage'] = pin(out/'initial.usda')
        original_q = np.asarray(scene['cloths'][0]['initial_vertices_m'])
        first_q = writer.arrays['cloth_q'][0].astype(float)
        report['native_warmup_difference_from_authored_shape'] = {
            'node_displacement_p95_m': float(np.quantile(np.linalg.norm(first_q - original_q, axis=1), .95)),
            'node_displacement_max_m': float(np.linalg.norm(first_q - original_q, axis=1).max()),
            'raw_speed_p95_m_s': float(np.quantile(np.linalg.norm(writer.arrays['cloth_qd'][0], axis=1), .95)),
            'scope': 'Native initialization evolves the explicit zero-speed authored reset; not an exact checkpoint resume.'}
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            capture = LaundryClothFrameCapture(stage=env.stage, world=env.world, app=env.app,
                cloth_views=env._cloth_views, cloth_specs=scene['cloths'], triangles=env.cloth_triangles,
                cloth_paths={n: info['path'] for n, info in env.scene_info['cloths'].items()},
                output=args.media_output, cameras={n: scene['cameras'][n] for n in ('task', 'overview')})

        def photograph(tag, step):
            previous = record(step)
            capture.capture(tag, camera_ids=('task', 'overview'))
            current = record(step)
            if any(not np.array_equal(v, current[n]) for n, v in previous.items()):
                raise RuntimeError('Rendering changed native robot or cloth state')

        if capture:
            photograph('initial', 0)
        heartbeat = time.monotonic()
        for step in range(1, steps + 1):
            t = (step - 1) * dt
            if t >= before - 1.e-10 and (step - 1 - round(before / dt)) % control_steps == 0:
                index = min(round((t - before) / .02), len(rows) - 1)
                row = rows[index]
                values = row['joint_positions_rad']
                env._validate_action({'joint_positions_rad': values, 'duration_s': .02})
                env._targets = np.asarray([[values[n] for n in env.joint_names]], dtype=np.float32)
                env._robot_view.set_dof_position_targets(env._tensor(env._targets), env._one)
                action_index = labels.index(row['label']) if t < before + plan['planned_duration_s'] else len(labels) - 1
            env._current_step_time = env.time_s + dt
            advance(env)
            env._contact_reports.clear()
            env._read_rigid_contacts()
            pairs = len(env._contact_reports)
            impulse = float(sum(r['impulse_magnitude_sum_ns'] for r in env._contact_reports))
            if pairs:
                contacts.write(json.dumps({'native_step': step, 'contacts': env._contact_reports}) + '\n')
            writer.append(record(step))
            if capture and int(step * dt * args.media_fps + 1.e-9) > int((step - 1) * dt * args.media_fps + 1.e-9):
                photograph(f'step_{step:06d}', step)
            if time.monotonic() - heartbeat > 20:
                print(json.dumps({'event': 'ordinary_press_progress', 'step': step, 'time_s': step * dt,
                                  'action': labels[action_index]}), flush=True)
                heartbeat = time.monotonic()
        if (env.world.current_time_step_index - start_step != steps
                or abs(env.world.current_time - start_time - duration) > 2.e-6):
            raise RuntimeError('Recorded and native clocks diverged')
        if env._contact_sampling['sampled_substeps'] != steps:
            raise RuntimeError('Incomplete original rigid-contact sampling')
        env._contact_sampling['complete'] = True
        final_attachments = [str(p.GetPath()) for p in env.stage.Traverse()
                             if any('Attachment' in schema for schema in p.GetAppliedSchemas())]
        report['final_attachment_prims'] = final_attachments
        if final_attachments:
            raise RuntimeError('Ordinary press unexpectedly created a cloth attachment')
        env.stage.Export(str(out/'final.usda'))
        report['final_stage'] = pin(out/'final.usda')
        if capture:
            photograph('final', steps)
            report['videos'] = {n: capture.encode_video(camera_id=n, fps=args.media_fps) for n in ('task', 'overview')}
        report.update(complete=True, native_steps=steps, duration_s=duration)
        exit_code = 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if writer is not None:
            report['states'] = writer.close()
        if env is not None:
            report['native_rigid_contact_sampling'] = dict(env._contact_sampling)
        if contacts is not None:
            contacts.close()
            report['rigid_contacts'] = pin(out/'rigid_contacts.jsonl')
        if capture is not None:
            capture.close()
        dump(out/'native_report.json', report)
        print(json.dumps({'event': 'ordinary_press_finished', 'complete': report['complete'],
                          'requested_exit_code': exit_code}), flush=True)
        if env is not None:
            try:
                env.close()
            except SystemExit:
                pass
    return exit_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    author = sub.add_parser('prepare')
    for name in ('native-report', 'plan', 'static-review', 'velocity-review', 'internal-review', 'output'):
        author.add_argument('--' + name, type=Path, required=True)
    author.add_argument('--allow-incomplete-source',action='store_true',
        help='Explicitly author a new diagnostic from a recorded released prefix; preserve original failure')
    native = sub.add_parser('run')
    native.add_argument('--manifest', type=Path, required=True)
    native.add_argument('--output', type=Path, required=True)
    native.add_argument('--media-output', type=Path)
    native.add_argument('--media-fps', type=float, default=4.)
    args = p.parse_args()
    if args.command == 'prepare':
        print(prepare(args))
    else:
        raise SystemExit(run(args))


if __name__ == '__main__':
    main()
