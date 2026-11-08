"""Replay new frozen laundry rules on unchanged, already audited native states.

The producer's release remains authoritative for physics, assets and provenance.
The explicitly selected evaluation release may change rules only. Existing CUDA
geometry observations are reused with their hashes and strict verdicts intact;
the new scorer decides practical task success from their quantitative evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from bench.isaacsim.release import checked, dump, load_evaluator, task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laundry_room.terminal_window import terminal_window
from scripts.maintenance.freeze_project_sources import verify, verify_if_snapshot
from scripts.validation import audit_laundry_cloth as audit
from scripts.validation.review_laundry_practical import verify_prior_audit

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = Path(f'{DATA_ROOT}/datasets')
SCHEMA = 'laundry_frozen_rules_replay.v1'
PROFILE = 'laundry_practical_task.v2'


def compatible_inputs(original, evaluation):
    """Only public rules, their binding and evaluator thresholds may differ."""
    old_manifest, old_scene, old_task, old_initial = original
    new_manifest, new_scene, new_task, new_initial = evaluation
    audit.require(old_scene == new_scene and old_initial == new_initial,
                  'Rule replay cannot change the native scene or initial state')
    for key in ('task_id', 'asset_catalog', 'asset_files_sha256', 'capabilities'):
        audit.require(old_manifest.get(key) == new_manifest.get(key),
                      'Rule replay changes a physical release binding: '+key)
    audit.require(new_task.get('evaluation', {}).get('rule_profile') == PROFILE,
                  'Expected the explicitly versioned practical task rules')
    omitted = {'public_rules', 'public_rules_sha256', 'public_referee_contract_sha256', 'evaluation'}
    audit.require({k: v for k, v in old_task.items() if k not in omitted}
                  == {k: v for k, v in new_task.items() if k not in omitted},
                  'Rule replay changes task inputs beyond public rules and evaluation')
    allowed = {'thresholds', 'public_rules_sha256', 'rule_profile'}
    audit.require({k: v for k, v in old_task['evaluation'].items() if k not in allowed}
                  == {k: v for k, v in new_task['evaluation'].items() if k not in allowed},
                  'Rule replay changes evaluation geometry, identity or native period')
    return {'scene_unchanged': True, 'initial_state_unchanged': True,
            'assets_unchanged': True, 'physics_unchanged': True,
            'only_public_rules_and_evaluation_changed': True}


def recorded_physics(row, index, timestamp, cloth_ids, original_limits):
    """Validate one audited geometry observation without relabeling its verdict."""
    audit.require(type(row.get('native_step')) is int and row['native_step'] == index
                  and type(row.get('time_s')) in (int, float)
                  and np.isfinite(row['time_s']) and abs(row['time_s']-timestamp) <= 1.e-8,
                  'Prior frame audit skips/reorders frames or has different timestamps')
    geometry, mesh = row['geometry'], row['mesh']
    audit.require(geometry.get('finite_thickness_compression_checked') is True,
                  'Prior frame lacks finite-thickness geometry evidence')
    fields = ('self_intersections', 'intercloth_intersections', 'degenerate_triangles')
    audit.require(all(type(geometry.get(k)) is int and geometry[k] >= 0 for k in fields),
                  'Prior frame lacks explicit nonnegative collision counts')
    depths = (row['table_envelope_penetration_m'], geometry['max_cloth_envelope_penetration_m'],
              row['max_penetration_m'])
    audit.require(all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in depths)
                  and abs(depths[2]-max(depths[:2])) <= 1.e-12,
                  'Prior frame penetration evidence is incomplete or inconsistent')
    audit.require(set(mesh) == set(cloth_ids), 'Prior frame cloth metric inventory differs')
    for values in mesh.values():
        audit.require(all(type(values.get(k)) in (int, float) and np.isfinite(values[k])
                          and values[k] >= 0 for k in ('max_edge_stretch_ratio',
                          'min_triangle_area_ratio', 'projected_area_ratio')),
                      'Prior frame mesh evidence is incomplete or nonfinite')
    strict_valid = (not any(geometry[k] for k in fields)
                    and depths[2] <= original_limits['max_penetration_m']
                    and all(values['max_edge_stretch_ratio'] <= original_limits['max_edge_stretch_ratio']
                            and values['min_triangle_area_ratio'] >= original_limits['min_triangle_area_ratio']
                            for values in mesh.values()))
    audit.require(type(row.get('physical_sample_valid')) is bool
                  and row['physical_sample_valid'] == strict_valid,
                  'Prior strict physical verdict contradicts its quantitative evidence')
    return {'complete': True, 'valid': row['physical_sample_valid'],
            'validity_scope': 'strict_geometry', 'gravity_enabled': True,
            **{key: geometry[key] for key in fields}, 'max_penetration_m': depths[2]}


def replay_arrays(arrays, trace_path, evaluator, original_task, *, score_trace=None):
    """Consume every recorded frame; an earlier successful frame is diagnostic."""
    names, count = tuple(evaluator.specs), len(arrays['native_step'])
    first_success, last, seen = None, None, 0
    strict_invalid = 0
    with Path(trace_path).open() as trace:
        for index, line in enumerate(trace):
            audit.require(index < count, 'Prior frame audit contains extra frames')
            row = json.loads(line)
            physics = recorded_physics(row, index, float(arrays['time_s'][index]), names,
                                       original_task['evaluation']['thresholds'])
            strict_invalid += int(not physics['valid'])
            physics['hidden_constraints'] = int(arrays['hidden_constraints'][index])
            frame = {'schema': 'laundry_native_frame.v1', 'complete': True,
                'native_step': index, 'time_s': float(arrays['time_s'][index]),
                'control_mode': 'engineering_local_grasp', 'physics': physics,
                'controlled_vertices': {name: np.flatnonzero(arrays[name+'_control_mask'][index]).tolist()
                                        for name in names},
                'cloths': {name: {'enabled': True, 'vertices_m': arrays[name+'_q'][index],
                    'velocities_m_s': arrays[name+'_qd'][index],
                    'triangles': arrays[name+'_triangles']} for name in names}}
            last = evaluator.observe(frame)
            if last.get('success') and first_success is None:
                first_success = {'native_step': index, 'time_s': frame['time_s']}
            if score_trace is not None:
                score_trace.write(json.dumps({'native_step': index, 'time_s': frame['time_s'],
                    'status': last['status'], 'success': last['success'], 'reason': last['reason'],
                    'released_stable_s': last['released_stable_s'],
                    'strict_physical_sample_valid': physics['valid']}, allow_nan=False)+'\n')
            seen += 1
            if index and index % 4000 == 0:
                print(json.dumps({'event': 'rule_replay_progress', 'native_step': index,
                                  'frames': count, 'status': last['status']}), flush=True)
    audit.require(seen == count, 'Prior frame audit is truncated')
    terminal = terminal_window(arrays, evaluator)
    success = bool(last and last.get('success') is True and terminal.get('passed') is True
                   and last.get('released_stable_s', 0.)+1.e-9 >= evaluator.limits['stable_s'])
    return {'samples_checked': seen, 'all_native_samples_checked': True,
        'first_task_success': first_success, 'final_evaluator_result': deepcopy(last),
        'terminal_window': terminal, 'success': success,
        'strict_invalid_frame_count': strict_invalid,
        'success_basis': 'Final frozen scorer result and actual terminal window; earlier success and strict geometry diagnostics are not extra task-success gates.'}


def run(args):
    modules = ('scripts.validation.replay_laundry_rules', 'scripts.validation.audit_laundry_cloth',
               'scripts.validation.review_laundry_practical',
               'scenes.workrooms.laundry_room.terminal_window', 'bench.isaacsim.release')
    source = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Rules replay must run from a complete frozen project source snapshot')
    output = args.output.resolve()
    audit.require(output.is_relative_to(DATASETS) and not output.exists(),
                  'Choose a fresh replay output under shared datasets')
    output.mkdir(parents=True)
    started, inputs = time.monotonic(), {}
    report = {'schema': SCHEMA, 'status': 'unknown', 'success': False,
        'rule_profile': PROFILE, 'new_native_simulation_performed': False,
        'full_geometry_audit_repeated': False, 'physics_accepted': False,
        'robot_manipulation_qualified': False, 'original_records_modified': False,
        'replay_source': source,
        'replay_source_manifest': audit.artifact(PROJECT/'source_snapshot_manifest.json')}
    try:
        native_path, prior_path, new_root = (args.native_report.resolve(),
            args.audit_report.resolve(), args.evaluation_release.resolve())
        for path in (native_path, prior_path, new_root):
            audit.require(path.is_relative_to(DATASETS), 'Replay inputs must reside under shared datasets')
        for label, path in (('native_report', native_path), ('full_stream_audit', prior_path)):
            report[label] = audit.artifact(path)
            inputs[str(path)] = report[label]['sha256']
        native, prior = (load_json(path) for path in (native_path, prior_path))
        old_root = Path(native['release']['path']).resolve()
        if old_root.name == 'release_manifest.json':
            old_root = old_root.parent
        audit.require(old_root.is_relative_to(DATASETS), 'Original release must reside under shared datasets')
        old_manifest, new_manifest = validate(old_root), validate(new_root)
        audit.require(all(m.get('schema') == 'laundry_cloth_task_release.v1'
                          for m in (old_manifest, new_manifest)), 'Expected frozen laundry releases')
        original = task_inputs(old_root, old_manifest['task_id'])
        evaluation = task_inputs(new_root, new_manifest['task_id'])
        report['input_compatibility'] = compatible_inputs(
            (old_manifest, *original[1:]), (new_manifest, *evaluation[1:]))
        _, scene, old_task, _ = original
        _, _, new_task, _ = evaluation
        report['task_id'] = new_task['id']
        for label, root, manifest in (('original_release', old_root, old_manifest),
                                       ('evaluation_release', new_root, new_manifest)):
            report[label] = audit.artifact(root/'release_manifest.json')
            inputs[report[label]['path']] = report[label]['sha256']
            for name, sha in manifest['files_sha256'].items():
                inputs[str(root/name)] = checked(root/name, sha)
            for path, sha in manifest['asset_files_sha256'].items():
                inputs[path] = checked(path, sha)
        # A profile file can contain material metadata in addition to scene options.
        audit.require(load_json(old_root/'physics_profile.json')
                      == load_json(new_root/'physics_profile.json'),
                      'Rule replay changes the complete frozen physics profile')
        states = audit.bound_artifact(native['state_archive'], inputs)
        audit.require(states.is_relative_to(DATASETS), 'Native states must reside under shared datasets')
        report['native_provenance'] = audit.verify_native_bindings(native, states, old_root,
                                                                  scene, old_task, inputs)
        trace = verify_prior_audit(prior, native, native_path, states, old_root, old_task, inputs)
        report['strict_diagnostics'] = {key: deepcopy(prior.get(key)) for key in (
            'status', 'passed', 'physical_geometry_stream_passed', 'first_physical_failure',
            'authored_initial_geometry', 'maxima', 'accepted_engineering_task_success',
            'frozen_evaluator_result', 'terminal_window')}
        with audit.NativeArchive(states, output) as arrays:
            report['stream_contract'] = audit.validate_stream(arrays, native, old_task,
                                                              scene=scene, inputs=inputs)
            with (output/'replayed_scores.jsonl').open('x') as score_trace:
                report.update(replay_arrays(arrays, trace, load_evaluator(new_root)(new_task),
                                             old_task, score_trace=score_trace))
        report['replayed_scores'] = audit.artifact(output/'replayed_scores.jsonl')
        for path, sha in inputs.items():
            checked(path, sha)
        report['original_source_final'] = verify(Path(report['native_provenance']['original_source_verified']['snapshot_root']))
        report['prior_audit_source_final'] = verify(Path(prior['audit_source_manifest']['path']).resolve().parent)
        report['replay_source_final'] = verify_if_snapshot(PROJECT, required_modules=modules,
                                                         entrypoint_file=Path(__file__))
        report['status'] = ('passed' if report['success'] else 'unknown'
                            if report['final_evaluator_result'].get('status') == 'unknown' else 'failed')
    except (audit.EvidenceUnknown, KeyError, TypeError, ValueError, OSError, RuntimeError,
            zipfile.BadZipFile) as error:
        report.update(status='unknown', success=False,
                      error={'type': type(error).__name__, 'message': str(error)})
    report['inputs_sha256'], report['elapsed_wall_s'] = inputs, time.monotonic()-started
    dump(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'success': report['success'], 'output': str(output)}), flush=True)
    return 0 if report['success'] else 2 if report['status'] == 'unknown' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-report', type=Path, required=True)
    parser.add_argument('--audit-report', type=Path, required=True)
    parser.add_argument('--evaluation-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
