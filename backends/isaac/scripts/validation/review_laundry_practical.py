"""Explicit practical review of an already audited native laundry operation.

This opt-in profile changes only the terminal speed statistic: every frame of
every cloth must meet the frozen speed limit at the 95th node percentile, for
both recorded velocity and adjacent-step displacement. Frozen geometry, full
release, all-node drift, operation history, and full-stream physics still gate
acceptance. It neither changes nor reports a pass for the frozen evaluator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from bench.isaacsim.release import checked, dump, load_evaluator, task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from scripts.maintenance.freeze_project_sources import verify, verify_if_snapshot
from scripts.validation import audit_laundry_cloth as audit

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = Path(f'{DATA_ROOT}/datasets')
SCHEMA = 'laundry_practical_operation_review.v1'
PROFILE = 'laundry_released_geometry_p95_speed.v1'


def practical_terminal_window(arrays, evaluator):
    """Recheck frozen terminal geometry once, preserving its strict verdict."""
    strict = audit.terminal_window(arrays, evaluator)
    result = {'schema': 'laundry_practical_terminal_window.v1',
        'practical_acceptance_profile': PROFILE, 'passed': False,
        'strict_terminal_window': strict, 'speed_percentile': 95,
        'percentile_method': 'linear',
        'aggregation': 'node-speed norm P95 within each frame and cloth; maximum over frames and cloths',
        'speed_limit_m_s': evaluator.limits['max_node_speed_m_s'],
        'all_node_drift_limit_m': evaluator.limits['stable_node_drift_m'],
        'drift_reference': 'each node at the start of the terminal window, as in the frozen strict gate'}
    if not strict['complete']:
        return {**result, 'reason_codes': strict['reason_codes']}
    count = strict['required_samples']
    per_cloth = {}
    for name in evaluator.specs:
        q = np.asarray(arrays[name+'_q'][-count:], dtype=float)
        qd = np.asarray(arrays[name+'_qd'][-count:], dtype=float)
        raw = np.linalg.norm(qd, axis=-1)
        pose = np.linalg.norm(np.diff(q, axis=0), axis=-1)/evaluator.c['physics_dt_s']
        raw95 = np.percentile(raw, 95, axis=1, method='linear')
        pose95 = np.percentile(pose, 95, axis=1, method='linear')
        per_cloth[name] = {
            'node_count': q.shape[1], 'raw_velocity_frames': count,
            'adjacent_step_intervals': count-1,
            'max_frame_raw_speed_p95_m_s': float(raw95.max()),
            'max_frame_pose_speed_p95_m_s': float(pose95.max()),
            'max_node_raw_speed_m_s': float(raw.max()),
            'max_node_pose_speed_m_s': float(pose.max()),
            'max_all_node_drift_m': float(np.linalg.norm(q-q[0], axis=-1).max())}
    raw95 = max(row['max_frame_raw_speed_p95_m_s'] for row in per_cloth.values())
    pose95 = max(row['max_frame_pose_speed_p95_m_s'] for row in per_cloth.values())
    # All remaining strict reasons, including geometry, drift, release, and
    # budget, are retained. No temporal smoothing or velocity writes occur.
    reasons = [reason for reason in strict['reason_codes']
               if reason not in ('terminal_raw_speed_exceeded', 'terminal_pose_speed_exceeded')]
    if raw95 > evaluator.limits['max_node_speed_m_s']:
        reasons.append('terminal_raw_speed_p95_exceeded')
    if pose95 > evaluator.limits['max_node_speed_m_s']:
        reasons.append('terminal_pose_speed_p95_exceeded')
    return {**result, 'passed': not reasons, 'per_cloth': per_cloth,
        'max_frame_raw_speed_p95_m_s': raw95,
        'max_frame_pose_speed_p95_m_s': pose95, 'reason_codes': reasons}


def operation_history(arrays, evaluator):
    """Retain real controlled displacement; a pristine reset earns no credit."""
    q, masks = arrays['towel_q'], arrays['towel_control_mask']
    active = np.flatnonzero(masks[1:].any(axis=1))+1
    maximum = 0.
    for start in range(0, len(active), 64):
        points = np.asarray(q[active[start:start+64]], dtype=float)
        maximum = max(maximum, float(np.linalg.norm(points-q[0], axis=-1).max()))
    return {'operation_seen': bool(len(active)),
        'operation_displacement_seen': bool(len(active) and maximum >= evaluator.limits['operation_displacement_m']),
        'max_controlled_phase_node_displacement_m': maximum,
        'required_displacement_m': evaluator.limits['operation_displacement_m'],
        'reset_credit': False}


def verify_prior_audit(report, run, native_path, states, root, task, inputs):
    """Bind the existing complete CUDA audit and all evidence it consumed."""
    require = audit.require
    require(report.get('schema') == audit.SCHEMA, 'Expected a complete-stream laundry audit')
    require(report.get('status') in ('passed', 'failed') and 'error' not in report,
            'Prior full-stream audit is incomplete or unknown')
    require(report.get('task_id') == task['id'] == run.get('task_id'), 'Prior audit task identity differs')
    require(report.get('all_native_samples_checked') is True, 'Prior audit did not check every native frame')
    require(type(report.get('physical_geometry_stream_passed')) is bool,
            'Prior audit lacks a physical geometry verdict')
    count = run['native_steps']+1
    contract = report.get('stream_contract', {})
    require(type(report.get('samples_checked')) is int and report['samples_checked'] == count
            and contract.get('frames_including_reset') == count and contract.get('native_steps') == count-1
            and contract.get('physics_dt_s') == task['evaluation']['physics_dt_s'],
            'Prior audit does not cover the complete native recording')
    require(re.fullmatch(r'cuda(?::[0-9]+)?', str(report.get('audit_device'))),
            'Prior complete geometry audit has no CUDA device receipt')
    bindings = report.get('inputs_sha256')
    require(isinstance(bindings, dict) and bindings, 'Prior audit input bindings are missing')
    for path in (native_path, states, root/'release_manifest.json', root/'scene.json', root/'task.json'):
        require(str(path) in bindings, 'Prior audit is not bound to the selected input: '+str(path))
    for path, sha in bindings.items():
        audit.bound_artifact({'path': path, 'sha256': sha}, inputs)
    source_manifest = audit.bound_artifact(report['audit_source_manifest'], inputs)
    require(source_manifest.name == 'source_snapshot_manifest.json', 'Prior audit source manifest identity differs')
    verify(source_manifest.parent)
    for key in ('audit_source', 'audit_source_final'):
        receipt = report.get(key, {})
        require(receipt.get('verified') is True and receipt.get('runtime_imports_verified') is True
                and receipt.get('snapshot_root') == str(source_manifest.parent),
                'Prior audit source/import verification is missing: '+key)
    frozen = load_json(source_manifest)
    for relative in ('scripts/validation/audit_laundry_cloth.py', 'bench/evaluation/cloth/gpu_intersections.py'):
        require(relative in frozen['files_sha256'], 'Prior audit source omits '+relative)
    require(report.get('frozen_evaluator_result', {}).get('evidence_status') == 'complete',
            'Prior frozen evaluator evidence is incomplete')
    if audit.initialization_mode(run) == 'author-only':
        require(type(report.get('authored_initial_geometry', {}).get('passed')) is bool,
                'Prior audit omits the authored pre-warmup geometry verdict')
    return audit.bound_artifact(report['frame_audit'], inputs)


def verify_frame_trace(path, arrays, report):
    """Verify complete per-frame audit coverage without rerunning collision tests."""
    total = len(arrays['native_step'])
    all_valid = True
    count = 0
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            audit.require(index < total, 'Prior frame audit contains extra frames')
            row = json.loads(line)
            audit.require(type(row.get('native_step')) is int and row['native_step'] == index
                and np.isfinite(row.get('time_s', np.nan))
                and abs(row['time_s']-float(arrays['time_s'][index])) <= 1.e-8,
                'Prior frame audit skips/reorders frames or has different timestamps')
            audit.require(type(row.get('physical_sample_valid')) is bool,
                          'Prior frame audit lacks physical sample verdict')
            all_valid &= row['physical_sample_valid']
            count += 1
    audit.require(count == total == report['samples_checked'], 'Prior frame audit is truncated')
    if report['physical_geometry_stream_passed']:
        audit.require(all_valid and report.get('first_physical_failure') is None
            and report.get('authored_initial_geometry', {'passed': True}).get('passed') is True,
            'Prior physical pass contradicts its per-frame or authored-state evidence')
    return {'frames_checked': count, 'all_native_frames_covered': True,
            'all_physical_sample_verdicts_passed': bool(all_valid), 'geometry_recomputed': False}


def review_arrays(arrays, evaluator, prior_audit):
    terminal = practical_terminal_window(arrays, evaluator)
    history = operation_history(arrays, evaluator)
    reasons = list(terminal['reason_codes'])
    if prior_audit.get('physical_geometry_stream_passed') is not True:
        reasons.append('full_stream_physical_geometry_failed')
    if (prior_audit.get('all_native_samples_checked') is not True
            or prior_audit.get('samples_checked') != len(arrays['native_step'])):
        reasons.append('full_stream_frame_coverage_incomplete')
    if not history['operation_seen']:
        reasons.append('no_engineering_operation')
    if not history['operation_displacement_seen']:
        reasons.append('operation_displacement_not_established')
    return {'practical_operation_accepted': not reasons, 'reason_codes': reasons,
            'terminal_window': terminal, 'operation_history': history}


def run(args):
    modules = ('scripts.validation.review_laundry_practical', 'scripts.validation.audit_laundry_cloth',
               'scenes.workrooms.laundry_room.terminal_window', 'bench.isaacsim.release')
    source = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Practical review must run from a complete frozen project source snapshot')
    output = args.output.resolve()
    audit.require(output.is_relative_to(DATASETS) and not output.exists(), 'Choose a fresh output under shared datasets')
    output.mkdir(parents=True)
    started, inputs = time.monotonic(), {}
    report = {'schema': SCHEMA, 'practical_acceptance_profile': PROFILE,
        'status': 'unknown', 'practical_operation_accepted': False,
        'physics_accepted': False, 'robot_manipulation_qualified': False,
        'new_native_simulation_performed': False, 'full_geometry_audit_repeated': False,
        'review_source': source, 'review_source_manifest': audit.artifact(PROJECT/'source_snapshot_manifest.json'),
        'scope': 'Explicit practical engineering-operation review; frozen task scoring remains unchanged. Geometry validity is inherited only from a complete hash-bound native audit.'}
    try:
        native_path, prior_path = args.native_report.resolve(), args.audit_report.resolve()
        for path in (native_path, prior_path):
            audit.require(path.is_relative_to(DATASETS), 'Evidence must reside under shared datasets')
            identity = audit.artifact(path)
            inputs[identity['path']] = identity['sha256']
        run_metadata, prior = (load_json(path) for path in (native_path, prior_path))
        states = audit.bound_artifact(run_metadata['state_archive'], inputs)
        root = Path(run_metadata['release']['path']).resolve()
        if root.name == 'release_manifest.json':
            root = root.parent
        audit.require(root.is_relative_to(DATASETS) and states.is_relative_to(DATASETS),
                      'Native states and release must reside under shared datasets')
        manifest = validate(root)
        audit.require(manifest.get('schema') == 'laundry_cloth_task_release.v1', 'Expected a frozen laundry release')
        _, scene, task, _ = task_inputs(root, manifest['task_id'])
        report['task_id'] = task['id']
        report['native_report'] = audit.artifact(native_path, inputs[str(native_path)])
        report['full_stream_audit'] = audit.artifact(prior_path, inputs[str(prior_path)])
        report['native_provenance'] = audit.verify_native_bindings(run_metadata, states, root, scene, task, inputs)
        for relative, sha in manifest['files_sha256'].items():
            inputs[str(root/relative)] = checked(root/relative, sha)
        for path, sha in manifest['asset_files_sha256'].items():
            inputs[path] = checked(path, sha)
        frame_trace = verify_prior_audit(prior, run_metadata, native_path, states, root, task, inputs)
        report['strict_diagnostics'] = {key: deepcopy(prior.get(key)) for key in (
            'status', 'passed', 'accepted_engineering_task_success', 'frozen_evaluator_result', 'terminal_window')}
        with audit.NativeArchive(states, output) as arrays:
            report['stream_contract'] = audit.validate_stream(arrays, run_metadata, task, scene=scene, inputs=inputs)
            report['prior_frame_coverage'] = verify_frame_trace(frame_trace, arrays, prior)
            report.update(review_arrays(arrays, load_evaluator(root)(task), prior))
        for path, sha in inputs.items():
            checked(path, sha)
        report['original_source_final'] = verify(Path(report['native_provenance']['original_source_verified']['snapshot_root']))
        report['prior_audit_source_final'] = verify(Path(prior['audit_source_manifest']['path']).resolve().parent)
        report['review_source_final'] = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
        report['status'] = 'accepted' if report['practical_operation_accepted'] else 'rejected'
    except (audit.EvidenceUnknown, KeyError, TypeError, ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        report.update(status='unknown', practical_operation_accepted=False,
                      error={'type': type(error).__name__, 'message': str(error)})
    report['inputs_sha256'], report['elapsed_wall_s'] = inputs, time.monotonic()-started
    dump(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'practical_operation_accepted': report['practical_operation_accepted'],
                      'output': str(output)}), flush=True)
    return 0 if report['practical_operation_accepted'] else 2 if report['status'] == 'unknown' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-report', type=Path, required=True)
    parser.add_argument('--audit-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
