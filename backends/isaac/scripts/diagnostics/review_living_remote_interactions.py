"""Replay actual object-only remote probes against the selected frozen contract.

Engineering face-up/holder starts deliberately violate the ordinary face-down
episode reset. Their mechanism/support evidence never earns robot task credit.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from bench.isaacsim.release import digest, load_evaluator, task_inputs
from bench.paths import load_json
from scripts.validation.living_remote_probe import assess, rotation


def body_state(body):
    pose, velocity = body['pose_xyzw'], body['velocity']
    q = np.asarray([pose[6], *pose[3:6]], dtype=float)
    if not np.isfinite(pose+velocity).all() or abs(np.linalg.norm(q)-1.) > 1.e-4:
        raise ValueError('Invalid native body state')
    q /= np.linalg.norm(q)
    return {'position_m': pose[:3], 'quaternion_wxyz': q.tolist(),
            'linear_velocity_m_s': velocity[:3], 'angular_velocity_rad_s': velocity[3:]}


def frame(row, baseline, dt):
    contacts = {}
    for item in row['contacts']:
        key = tuple(sorted((item['sensor'], item['filter'])))
        if key in contacts:
            continue  # Both tensor sensor directions describe one physical pair.
        contacts[key] = {'actor0': key[0], 'actor1': key[1], 'time_s': row['time_s']-baseline['time_s'],
            'point_count': item['count'], 'impulse_magnitude_sum_ns': item['normal_impulse_l1_ns'],
            'min_separation_m': item['min_separation_m']}
    return {'baseline': row['tick'] == 0, 'time_s': row['time_s']-baseline['time_s'],
        'native_time_s': row['time_s'], 'native_step': row['tick'], 'physics_dt_s': dt,
        'objects': {key: body_state(row['bodies']['body' if key == 'remote' else key])
                    for key in ('remote', 'power', 'confirm', 'holder')},
        'joints': {key: {'position_m': row['joints'][key]['q_m']} for key in ('power', 'confirm')},
        'contacts': list(contacts.values()), 'particle_sets': {}}


def review(run):
    report_path = run/'report.json'
    report = load_json(report_path)
    receipt_path = run.with_name(run.name+'.receipt.json')
    receipt = load_json(receipt_path)
    release = Path(report['selected_release']['path'])
    if digest(release/'release_manifest.json') != report['selected_release']['manifest_sha256']:
        raise ValueError('Frozen release manifest changed')
    _, scene, task, _ = task_inputs(release, 'remote_operation')
    if task != load_json(run/'selected_task.json'):
        raise ValueError('Probe task differs from selected frozen task')
    if report['dt_s'] != scene['physics']['dt_s']:
        raise ValueError('Probe timestep differs from frozen task')
    for name, expected in report['artifact_sha256'].items():
        if digest(run/name) != expected:
            raise ValueError('Native artifact changed: '+name)
    baseline = load_json(run/'baseline.json')
    rows = [json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
    dt = report['dt_s']
    assessment = assess(rows, report['case'], dt, report['paths'], table_path=report['table_path'])
    checks = {'process_completed': receipt.get('exit_code') == report['exit_code'] == 0,
        'native_report_passed': report['status'] == 'completed' and report['passed'],
        'full_native_steps': len(rows) == report['native_steps'] == round(report['requested_duration_s']/dt),
        'mechanism_replay_passed': assessment['passed'],
        'replay_matches_native_report': assessment == report['assessment'],
        'no_robot_actions_or_model': report['robot_actions'] == report['model_calls'] == 0}
    cls = load_evaluator(release)
    frozen = importlib.import_module(cls.__module__)
    observer = frozen.NativeTaskObserver(task)
    observer.on_frame(frame(baseline, baseline, dt))
    result = {'case': report['case'], 'checks': checks,
        'native_steps': len(rows), 'simulated_seconds': report['native_time_advance_s'],
        'mechanism': assessment, 'source_snapshot': report['source_snapshot'],
        'selected_release': report['selected_release'], 'robot_manipulation_qualified': False,
        'physics_accepted': False}
    if report['case'] == 'reset':
        referee = cls(task)
        states = [observer.on_observe()]
        scores = [referee.update({'time_s': 0., 'task_state': states[-1]})]
        for row in rows:
            observer.on_frame(frame(row, baseline, dt))
            if row['tick'] % round(.02/dt) == 0:
                states.append(observer.on_observe())
                scores.append(referee.update({'time_s': states[-1]['time_s'], 'task_state': states[-1]}))
        checks['frozen_observer_complete'] = all(s['complete'] and not s['errors'] for s in states)
        checks['no_native_button_events'] = all(not s['events'] for s in states)
        checks['no_false_tv_or_task_credit'] = all(not s['success'] and s['progress'] == 0.
            and s['referee']['phase'] == 'await_power_on' for s in scores)
        result['frozen_observer'] = states[-1]
        result['frozen_referee'] = scores[-1]
        result['public_control_samples'] = len(scores)
    else:
        checks['ordinary_referee_rejects_engineering_reset'] = (
            observer.error == 'Reset must be face down with released buttons')
        result['frozen_reset_rejection'] = observer.on_observe()
        checks['no_robot_contact'] = all(not c[key].startswith('/World/Robot/')
            for row in rows for c in row['contacts'] for key in ('sensor', 'filter'))
        if report['case'] == 'press':
            # Feed factual unheld physical cycles to the unmodified frozen TV
            # machine. This is a state-machine replay, not an ordinary episode.
            tv = frozen.TelevisionState(task['evaluation']['required_on_s'], task['evaluation']['stable_s'])
            events = sorted((dict(e, button=name) for name, events in assessment['details']['button_events'].items()
                             for e in events if e['event'] != 'armed'), key=lambda e: e['tick'])
            for event in events:
                tv.event({'button': event['button'], 'kind': event['event'],
                          'time_s': event['native_time_s'], 'handheld_valid': False})
            result['unheld_cycle_tv_replay'] = tv.state(rows[-1]['time_s'])
            checks['unheld_physical_cycles_do_not_toggle_tv'] = not tv.power_on and tv.completed_cycles == 0
        else:
            tail = rows[-round(task['evaluation']['stable_s']/dt):]
            c = task['evaluation']
            geometries = []
            previous = rows[-len(tail)-1]
            for row in tail:
                remote, holder = body_state(row['bodies']['body']), body_state(row['bodies']['holder'])
                corners = np.asarray(observer._corners(remote))
                local = (corners-np.asarray(holder['position_m'])) @ rotation(row['bodies']['holder']['pose_xyzw'][3:])
                half = np.asarray(c['geometry']['holder_opening_xy_m'])/2+c['holder_xy_tolerance_m']
                geometry = {'tick': row['tick'], 'xy_margin_m': float(np.min(half-np.max(np.abs(local[:,:2]),axis=0))),
                    'bottom_error_m': abs(float(local[:,2].min())-c['geometry']['holder_liner_z_m']),
                    'orientation_error_rad': frozen.angle(remote['quaternion_wxyz'],holder['quaternion_wxyz'])}
                geometries.append(geometry)
                for name in ('body','power','confirm','holder'):
                    now, before = body_state(row['bodies'][name]), body_state(previous['bodies'][name])
                    checks['released_tail_public_speed_limits'] = checks.get('released_tail_public_speed_limits', True) and (
                        np.linalg.norm(now['linear_velocity_m_s']) <= c['maximum_linear_speed_m_s']
                        and np.linalg.norm(now['angular_velocity_rad_s']) <= c['maximum_angular_speed_rad_s']
                        and np.linalg.norm(np.asarray(now['position_m'])-before['position_m'])/dt <= c['maximum_linear_speed_m_s']
                        and frozen.angle(now['quaternion_wxyz'],before['quaternion_wxyz'])/dt <= c['maximum_angular_speed_rad_s'])
                previous = row
            checks['holder_fits_frozen_geometry'] = all(g['xy_margin_m'] >= 0
                and g['bottom_error_m'] <= c['holder_bottom_tolerance_m']
                and g['orientation_error_rad'] <= c['holder_orientation_tolerance_rad'] for g in geometries)
            result['holder_last_300ms'] = {'frames': len(tail), 'minimum_xy_margin_m': min(g['xy_margin_m'] for g in geometries),
                'maximum_bottom_error_m': max(g['bottom_error_m'] for g in geometries),
                'maximum_orientation_error_rad': max(g['orientation_error_rad'] for g in geometries)}
    result['checks'] = {k: bool(v) for k,v in checks.items()}
    result['passed'] = all(result['checks'].values())
    result['inputs_sha256'] = {str(p): digest(p) for p in (report_path, receipt_path, run/'baseline.json',
        run/'samples.jsonl', run/'raw_contacts.jsonl', release/'release_manifest.json', Path(__file__))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'checks': result['checks']}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
