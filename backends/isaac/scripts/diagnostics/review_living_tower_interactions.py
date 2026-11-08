"""Replay every native terminal step of an object-only tower probe.

This adds complete-window support/motion evidence to the original probe's
20 ms checkpoints. It never changes old reports or awards a game turn.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np

from bench.isaacsim.release import digest, dump, _frozen_evaluator_module
from bench.paths import load_json
from scripts.validation.living_tower_interactions import extraction_conditions, cached_normalizer


def run(run_dir, output, report_name='qualification.json'):
    if output.exists():
        raise FileExistsError(output)
    if Path(report_name).name != report_name:
        raise ValueError('Report must be a file within the native run directory')
    original = load_json(run_dir/report_name)
    if original['task_id'] != 'tower_extraction' or not original.get('native_steps'):
        raise ValueError('A completed Task4 native object run is required')
    for name, expected in original['evidence_sha256'].items():
        if digest(run_dir/name) != expected:
            raise ValueError('Changed native evidence: '+name)
    release = Path(original['release']['path'])
    if digest(release/'release_manifest.json') != original['release']['manifest_sha256']:
        raise ValueError('Changed frozen release')
    task = load_json(run_dir/'task.json')
    frozen = _frozen_evaluator_module(release)
    native = importlib.import_module(frozen.__package__+'.native_evidence')
    evaluator = frozen.SequenceEvaluator(task)
    normalize = cached_normalizer(native, task['evaluation']['body_paths'], game=True,
        contact_actor_paths=task['evaluation']['contact_actor_paths'])
    game = evaluator.game
    case = original['case']
    selected = 'block_16' if case in ('collection', 'extract') else None
    dt, duration = original['physics_dt_s'], original['duration_s']
    threshold = duration-1.
    counts = {'terminal_native_frames': 0, 'remaining_support_failures': 0,
        'remaining_motion_failures': 0, 'selected_collection_support_or_motion_failures': 0,
        'selected_slot_failures': 0, 'terminal_external_load_frames': 0,
        'whole_trace_remaining_collapse_frames': 0}
    first_failures = {}; previous = None; first_tail = last_tail = None
    max_drop = max_tilt = max_penetration = max_linear = max_angular = 0.
    original_support = game._support
    checked_cache_samples = []
    progress = time.monotonic()
    with gzip.open(run_dir/'native_frames.jsonl.gz', 'rt') as stream:
        for index, line in enumerate(stream):
            frame = json.loads(line)
            if (frame['step_index'] != index or abs(frame['time_s']-index*dt) > 1.e-10
                    or set(frame['objects']) != set(task['evaluation']['body_paths'])):
                raise ValueError('Missing native frame or object inventory')
            if previous is not None and frame['native_physics_step_index'] != previous['native_physics_step_index']+1:
                raise ValueError('Native step indices are not consecutive')
            normalized = normalize(frame['raw_contacts'])
            if normalized != frame['contacts']:
                raise ValueError('Stored normalized contacts differ from raw native evidence')
            max_penetration = max(max_penetration, max((-r['separation_m'] for r in normalized), default=0.))
            remaining = set(frame['objects'])-({selected} if selected else set())
            drop = max(.78+task['evaluation']['reference_configuration']['blocks'][n]['center_h'][2]*.03-frame['objects'][n]['position_m'][2]
                for n in remaining)
            tilt = max(math.acos(float(np.clip(native.rotation(frame['objects'][n]['quaternion_wxyz'])[2, 2], -1., 1.))) for n in remaining)
            max_drop, max_tilt = max(max_drop, drop), max(max_tilt, tilt)
            if drop >= .024 or tilt >= math.pi/4:
                counts['whole_trace_remaining_collapse_frames'] += 1
                first_failures.setdefault('whole_trace_collapse', frame['time_s'])
            if frame['time_s'] >= threshold-1.e-9:
                # The frozen support graph descends strictly in height. Memoize
                # its own exact recursive predicate per body/surface, without
                # changing contact normals, geometry, force or tolerances.
                cache = {}
                def support(bid, allowed_blocks, visiting=None, *, surface_actor='table'):
                    key = (bid, surface_actor, frozenset(allowed_blocks))
                    if key not in cache:
                        cache[key] = original_support(bid, allowed_blocks, visiting, surface_actor=surface_actor)
                    return cache[key]
                game._support = support
                condition = extraction_conditions(evaluator, previous, frame, native, selected)
                game._support = original_support
                if len(checked_cache_samples) < 3:
                    uncached = extraction_conditions(evaluator, previous, frame, native, selected)
                    if uncached != condition:
                        raise ValueError('Memoized frozen support predicate differs')
                    checked_cache_samples.append(frame['step_index'])
                counts['terminal_native_frames'] += 1
                first_tail = frame['time_s'] if first_tail is None else first_tail
                last_tail = frame['time_s']
                failures = {
                    'remaining_support_failures': not condition['all_remaining_supported'],
                    'remaining_motion_failures': not condition['all_remaining_motion_valid'],
                    'selected_collection_support_or_motion_failures': bool(selected and not condition['selected_tray_supported_and_motion_valid']),
                    'selected_slot_failures': bool(selected and not condition['selected_inside_front_slots']),
                    'terminal_external_load_frames': any(frame['engineering_wrench'][key] != [0., 0., 0.] for key in ('force_world_n', 'torque_world_nm'))}
                for key, failed in failures.items():
                    counts[key] += int(failed)
                    if failed:
                        first_failures.setdefault(key, {'time_s': frame['time_s'], 'condition': condition})
                for name in frame['objects']:
                    vmax, wmax, _, _ = native.extrema([previous, frame], name)
                    max_linear, max_angular = max(max_linear, vmax), max(max_angular, wmax)
            previous = frame
            if time.monotonic()-progress > 20.:
                print(json.dumps({'review': str(run_dir), 'native_time_s': frame['time_s'], 'terminal_samples': counts['terminal_native_frames']}), flush=True)
                progress = time.monotonic()
    checks = {'complete_native_trace': index+1 == original['native_steps']+1,
        'complete_terminal_second': counts['terminal_native_frames'] == round(1./dt)+1 and abs(last_tail-first_tail-1.) <= 1.e-9,
        'input_report_checks_passed': original['passed'] is True,
        'full_terminal_remaining_support': counts['remaining_support_failures'] == 0,
        'full_terminal_remaining_motion': counts['remaining_motion_failures'] == 0,
        'full_terminal_collection_support_and_motion': counts['selected_collection_support_or_motion_failures'] == 0,
        'full_terminal_collection_slot': counts['selected_slot_failures'] == 0,
        'full_terminal_external_release': counts['terminal_external_load_frames'] == 0,
        'whole_trace_remaining_no_collapse': counts['whole_trace_remaining_collapse_frames'] == 0,
        'no_deep_penetration': max_penetration <= .0015}
    result = {'schema': 'living_tower_complete_native_replay.v1', 'passed': all(checks.values()),
        'checks': checks, 'counts': counts, 'first_failures': first_failures,
        'input': str(run_dir), 'input_report_name': report_name,
        'input_report_sha256': digest(run_dir/report_name),
        'native_frames_sha256': digest(run_dir/'native_frames.jsonl.gz'),
        'review_sources_sha256': {str(Path(__file__).resolve()): digest(Path(__file__)),
            str(Path(importlib.import_module('scripts.validation.living_tower_interactions').__file__).resolve()):
                digest(importlib.import_module('scripts.validation.living_tower_interactions').__file__)},
        'cached_vs_original_frozen_predicate_ticks': checked_cache_samples,
        'terminal_start_s': first_tail, 'terminal_end_s': last_tail,
        'maximum_remaining_drop_m': max_drop, 'maximum_remaining_tilt_rad': max_tilt,
        'maximum_raw_contact_penetration_m': max_penetration,
        'terminal_peak_raw_or_pose_linear_speed_m_s': max_linear,
        'terminal_peak_raw_or_pose_angular_speed_rad_s': max_angular,
        'robot_manipulation_qualified': False, 'task_success_claimed': False,
        'scope': 'Every saved native substep in final one second; object support/geometry and motion only. No game contact/turn credit.'}
    dump(output, result)
    print(json.dumps({'passed': result['passed'], 'output': str(output), 'counts': counts}), flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-name', default='qualification.json')
    args = parser.parse_args()
    raise SystemExit(run(args.run.resolve(), args.output.resolve(), args.report_name))


if __name__ == '__main__':
    main()
