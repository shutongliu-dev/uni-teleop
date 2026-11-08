"""Read-only review of a completed Task10 HTTP/native archive.

Replay only evidence actually stored: every 20ms scorer receipt and every native
contact row. The normal archive lacks per-substep object poses, so this does not
reconstruct the observer or manufacture a hand-contact retreat experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bench.isaacsim.release import load_evaluator
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify
from scripts.validation.mechanical_cooperative_release import private_coverage
from scripts.validation.mechanical_nist_support import support_evidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return load_json(Path(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(path, root):
    return path == root or path.startswith(root+'/')


def review(http, release):
    http, release = Path(http).resolve(), Path(release).resolve()
    q = read(http/'qualification.json')
    private = http/'episodes'/q['episode_id']/'private'
    metadata, task, scene = (read(private/name) for name in ('metadata.json','task.json','scene.json'))
    records = [json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines()]
    require(q['passed'] and q['model_calls'] == 0, 'Native HTTP run did not pass without model calls')
    require(Path(q['task_release']).resolve() == release, 'Run used a different release')
    require(q['release_manifest_sha256'] == sha(release/'release_manifest.json'), 'Release manifest mismatch')
    require(q['result']['physics_valid'] is True and q['result']['outcome'] == 'cancelled', 'Expected valid cancelled micro-motion')
    require(q['server_exit_code'] == 0 and q['server_shutdown']['episodes'][q['episode_id']]['worker_exit_code'] == 0, 'Native/server process did not close normally')
    require(metadata['object_pose_writeback_after_reset'] is False, 'Runtime object pose writes were enabled')
    require(q['private_state_used_for_targets'] is False, 'Micro-motion used private targets')
    source = Path(q['runtime_imports']['snapshot_root'])
    source_check = verify(source)
    require(q['runtime_imports']['verified'] and q['runtime_imports_final']['verified'], 'Missing before/after import verification')
    original_task, original_scene = read(release/'task.json'), read(release/'scene.json')
    for key in ('public_rules','evaluation','asset_metadata'):
        require(task[key] == original_task[key], 'Worker task differs: '+key)
    descriptor = scene['native_task_bundle']['observer']
    expected_descriptor = dict(original_scene['native_task_bundle']['observer'])
    expected_descriptor['path'] = str((release/expected_descriptor['path']).resolve())
    require(descriptor == expected_descriptor, 'Worker native observer descriptor differs')
    require(descriptor['configuration']['evaluation'] == task['evaluation'], 'Observer and scorer configuration differ')
    require(descriptor['sha256'] == sha(release/descriptor['path']), 'Observer source hash differs')
    for filename, expected in descriptor['files'].items():
        require(sha(release/'evaluator'/filename) == expected, 'Observer helper changed: '+filename)
    evaluator_class = load_evaluator(release)
    evaluator = evaluator_class(task)
    end = records[-1]['observation']['time_s']
    checks = {'coverage': private_coverage(metadata, records, task, end),
              'support': support_evidence(metadata, records, task),
              'source_snapshot': source_check}
    require(checks['support']['passed'], 'Recorded initial support/rest failed')
    for i, row in enumerate(records):
        require(evaluator.update(row['observation']) == row['score'], f'Frozen scorer replay differs at control row {i}')
    checks['frozen_scorer_replay'] = {'passed': True, 'control_receipts': len(records),
        'scope': 'Replay saved trusted native summaries; not observer re-execution from full native object frames.'}
    roots = metadata['runtime_robot_base_check']
    require(roots['verification_passed'] and roots['all_fixed_base'], 'Fixed robot base check failed')
    require(len(metadata['joint_names']) == 58 and len(roots['articulations']) == 2
            and all(x['fixed_base'] and x['dof_count'] == 29 and x['root_pose_verified'] for x in roots['articulations']), 'Expected two real fixed 29DOF robots')
    require(q['motion']['same_action_for_all_hands'] and q['motion']['action_idempotency_checked']
            and q['motion']['hands'] == ['left','right'], 'Missing simultaneous left/right public motion')
    checks['robot_and_motion'] = {'passed': True, 'dofs': 58, 'articulations': roots['articulations'], 'motion': q['motion']}
    dt = metadata['physics_dt_s']; actors = task['asset_metadata']['actors']
    aliases = metadata['scene_info'].get('collider_actor_paths', {})
    tables = [p.replace('/World/Geometry/','/World/Room/Geometry/') for p in task['room_support_paths']['table']]
    tables = [aliases.get(p,p) for p in tables]
    tail = set(range(round((end-.3)/dt)+1, round(end/dt)+1))
    counts = {n: {'support_steps':set(), 'robot_contact_rows':0, 'minimum_separation_m':0.} for n in actors}
    native_contact_rows = 0
    for record in records[1:]:
        obs = record['observation']; sampling = obs['contacts']['sampling']
        for row in obs['contacts']['rigid']:
            native_contact_rows += 1
            tick = round(row['time_s']/dt)
            require(abs(tick*dt-row['time_s']) < 1e-8 and sampling['start_sim_time_s'] < row['time_s']+1e-10 <= sampling['end_sim_time_s']+1e-8, 'Contact is off native grid/control window')
            require(np.isfinite(row['impulse_vector_ns']).all() and np.isfinite(row['min_separation_m']), 'Invalid native contact values')
            pair = (row['actor0'], row['actor1'])
            for name, actor in actors.items():
                indices = [i for i, path in enumerate(pair) if within(path, actor)]
                if not indices:
                    continue
                require(len(indices) == 1, 'Self actor contact row')
                i = indices[0]; other = pair[1-i]; entry = counts[name]
                entry['minimum_separation_m'] = min(entry['minimum_separation_m'], row['min_separation_m'])
                entry['robot_contact_rows'] += int(within(other, '/World/Robot'))
                upward = row['impulse_vector_ns'][2]*(1 if i == 0 else -1)
                if tick in tail and upward > 1e-10 and any(within(other, p) for p in tables):
                    entry['support_steps'].add(tick)
    summary = {}
    for name, entry in counts.items():
        require(entry['support_steps'] == tail, 'Incomplete terminal native table support: '+name)
        require(entry['robot_contact_rows'] == 0 and entry['minimum_separation_m'] > -.002, 'Unexpected hand contact or penetration')
        summary[name] = {**entry, 'support_steps':len(entry['support_steps']), 'complete_300ms_table_support':True}
    for row in records:
        state = row['observation']['task_state']; detail = state['detail']
        require(not state['events'] and state['released_stable_s'] == 0.
            and detail['cooperation_time_s'] == detail['cooperative_insertion_progress_m'] == 0.
            and not detail['cooperation_confirmed'] and not detail['distinct_hands']
            and all(not hands for hands in detail['hand_contacts'].values()), 'Micro-motion created manipulation/history credit')
    checks['native_contacts_and_no_credit'] = {'passed': True, 'contact_rows': native_contact_rows,
        'actors': summary, 'cooperation_events':0, 'hand_contact_rows':0, 'retreat_branch_physically_exercised':False}
    with np.load(private/'states.npz', allow_pickle=False) as states:
        n = len(records)
        require(states['object_q'].shape == (n, 2, 7), 'Incomplete saved object states')
        require(np.max(np.abs(states['time_s']-[r['observation']['time_s'] for r in records])) < 1e-8, 'State archive time mismatch')
        for i, row in enumerate(records):
            obs = row['observation']
            require(states['native_physics_step_index'][i] == obs['task_state']['native_step'], 'Archive/observer native step mismatch')
            require(abs(states['native_world_time_s'][i]-obs['task_state']['native_time_s']) < 1e-8, 'Archive/observer native time mismatch')
            for j, name in enumerate(metadata['object_names']):
                obj = obs['objects'][name]; quat = obj['quaternion_wxyz']
                pose = obj['position_m']+quat[1:]+quat[:1]
                velocity = obj['linear_velocity_m_s']+obj['angular_velocity_rad_s']
                require(np.max(np.abs(states['object_q'][i,j]-pose)) < 1e-7, 'Archive object pose differs')
                require(np.max(np.abs(states['object_qd'][i,j]-velocity)) < 1e-7, 'Archive object velocity differs')
        checks['state_archive'] = {'passed':True, 'pose_samples':n,
            'native_steps':round(end/dt), 'pose_period_s':.02,
            'full_per_native_step_pose_archive':False, 'observer_full_native_replay_performed':False}
    camera_ids = {'overview','task','room','left_wrist','right_wrist'}
    require({row['camera_id'] for row in q['media']} == camera_ids, 'Missing RGB-D camera')
    frames = []
    for row in q['media']:
        for kind in ('rgb','depth'):
            require(sha(row[kind]['path']) == row[kind]['sha256'], 'Camera bytes mismatch')
        rgb = np.asarray(Image.open(row['rgb']['path']).convert('RGB'))
        depth = np.load(row['depth']['path'], allow_pickle=False)
        require(depth.shape == rgb.shape[:2] and depth.dtype == np.float32 and np.isfinite(depth).all()
                and np.count_nonzero(depth > 0) and rgb.std() >= 1., 'Blank/invalid camera')
        frames.append({'camera_id':row['camera_id'], 'shape':list(rgb.shape), 'depth_valid_pixels':int(np.count_nonzero(depth > 0))})
    checks['five_rgbd'] = {'passed':True, 'frames':frames}
    files = [http/'qualification.json', release/'release_manifest.json', release/descriptor['path'],
        source/'source_snapshot_manifest.json', Path(__file__), *(private/p for p in
        ('metadata.json','scene.json','task.json','physics.jsonl','states.npz','physics_audit.json','worker_config.json','runtime_imports.json'))]
    return {'schema':'mechanical_cooperative_native_rules_review.v1', 'passed':True,
        'http_output':str(http), 'release':str(release), 'release_manifest_sha256':sha(release/'release_manifest.json'),
        'observer_and_scorer_sha256':descriptor['sha256'], 'source_snapshot':str(source),
        'checks':checks, 'evidence_sha256':{str(p):sha(p) for p in files},
        'limits':['Actual HTTP/native initialization, micro-motion, support and frozen-scorer replay only.',
                  'No hand-object contacts occurred; the repaired retreat branch was not physically exercised.',
                  'Stored object poses are 20ms endpoints. Per-native-step contact receipts do not supply missing object poses.',
                  'No interpolated poses, fabricated contacts, object state writes or model calls are used by this review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--http', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a fresh review output file')
    report = review(args.http, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'output':str(args.output)}))


if __name__ == '__main__':
    main()
