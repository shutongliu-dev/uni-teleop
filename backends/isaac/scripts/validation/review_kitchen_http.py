"""Read-only release, Agent transport and native support review of a kitchen smoke.

The input must be a completed ``public_http_smoke run --agent-family kitchen``.
This does not launch physics, infer a robot manipulation result or modify inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from bench.paths import load_json
from scripts.validation.kitchen_scene_evidence import support_evidence
from scripts.validation.kitchen_task_release import private_coverage


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return load_json(Path(path))


def interface_checks(report, calls, agent_calls, release_sha256):
    """Audit saved public receipts without trusting the top-level passed flag."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    created, result = report['created'], report['result']
    spec = created['spec']
    require(report.get('passed') is True, 'Original HTTP smoke did not pass')
    require(report['release_manifest_sha256'] == spec['task_release_sha256'] == release_sha256,
            'HTTP report and created episode do not bind the selected release')
    require(result['spec'] == spec and spec['task_id'] == report['task_id'],
            'Creation and terminal episode identities differ')
    require(spec['track'] == 'system0', 'Actual Agent geometry track was not tested')
    require(result['physics_valid'] is True and result['outcome'] == 'cancelled'
            and result['model_calls'] == report['model_calls'] == 0,
            'Smoke must end with valid physics, no task-success claim and no model calls')
    require(report['server_exit_code'] == 0 and report.get('forced_server_shutdown') is not True,
            'Server did not shut down normally')
    shutdown = report['server_shutdown']['episodes']
    require(set(shutdown) == {report['episode_id']}
            and shutdown[report['episode_id']]['worker_exit_code'] == 0
            and shutdown[report['episode_id']]['finished'] is True,
            'Native worker was not recorded as finished with exit zero')
    require(report['ik']['solution_found'] is True
            and report['ik']['advances_physics'] is False
            and report['ik']['mutates_controller'] is False,
            'Read-only current-wrist IK receipt is missing')
    checks = report.get('agent_interface_checks', [])
    require([row['phase'] for row in checks] == ['created', 'before_finish'],
            'Actual Agent transport must be checked before and after robot motion')
    for row in checks:
        require(row['family'] == 'kitchen' and row['task_id'] == report['task_id']
                and row['task_release_sha256'] == release_sha256,
                'Agent receipt has another task or release')
        require(all(row.get(key) is True for key in ('catalog_release_verified',
            'agent_public_parsing_verified', 'agent_ready_contract_verified',
            'static_model_requirements_verified', 'read_only_verified'))
            and row.get('track') == 'system0'
            and row['actions_submitted'] == row['model_calls'] == 0,
            'Agent parsing, ready contract or read-only checks are incomplete')
    require(len(agent_calls) == 6
            and [row['request']['function'] for row in agent_calls] == ['observe', 'get_task', 'observe']*2
            and all(row['response']['ok'] is True for row in agent_calls),
            'Actual Agent query receipts are incomplete')
    require(all(200 <= row['status'] < 300 for row in calls), 'Saved HTTP call failed')
    actions = [row['request'] for row in calls if row.get('request', {})
               and row['request'].get('function') in ('move_geometry', 'hold')]
    kinds = [row['function'] for row in actions]
    require(kinds.count('move_geometry') == 4 and kinds.count('hold') == 2,
            'Expected wrist motion, return and hold with identical-action replay')
    require(len(actions) == 6 and all(actions[i]['arguments'] == actions[i+1]['arguments']
                                    for i in (0, 2, 4)),
            'Action replays differ from their original submitted arguments')
    return {'passed': not errors, 'errors': errors,
            'scope': 'Saved real TCP HTTP and actual Agent client receipts; no new execution.'}


def review(http_root, release):
    from bench.isaacsim.release import task_inputs
    from scripts.maintenance.freeze_project_sources import verify
    from PIL import Image
    import numpy as np

    http_root, release = Path(http_root).resolve(), Path(release).resolve()
    evidence = {}

    def bound(path):
        path = Path(path).resolve()
        evidence[str(path)] = digest(path)
        return read(path)

    report = {'schema': 'kitchen_http_support_review.v1', 'passed': False,
        'http_root': str(http_root), 'release': str(release), 'checks': {}, 'errors': [],
        'model_calls': 0, 'native_execution_performed': False,
        'manipulation_qualified': False, 'physics_accepted': False,
        'scope': 'Independent saved HTTP/Agent, frozen release and original-reset support review. '
                 'IK covers only the current wrist; workspace IK is separate. No grasp or full task success.'}
    try:
        smoke = bound(http_root/'qualification.json')
        manifest = bound(release/'release_manifest.json')
        release_sha = evidence[str(release/'release_manifest.json')]
        if Path(smoke['task_release']).resolve() != release:
            raise ValueError('Selected release path differs from HTTP run')
        calls, agent_calls = bound(http_root/'http_calls.json'), bound(http_root/'agent_http_calls.json')
        report['checks']['interface'] = interface_checks(smoke, calls, agent_calls, release_sha)
        spec = smoke['created']['spec']
        _, expected_scene, task, initial = task_inputs(release, smoke['task_id'], seed=spec['seed'])
        for name in manifest['files']:
            evidence[str(release/name)] = digest(release/name)
        private = http_root/'episodes'/smoke['episode_id']/'private'
        runtime, saved_task = bound(private/'metadata.json'), bound(private/'task.json')
        saved_scene = bound(private/'scene.json')
        if saved_task != task or runtime['scene_spec'] != saved_scene:
            raise ValueError('Recorded task or runtime scene differs from the frozen episode inputs')
        if runtime['reset_override'] != initial or runtime['object_pose_writeback_after_reset'] is not False:
            raise ValueError('Recorded reset or object-writeback policy differs')
        if saved_scene['isaacsim'] != expected_scene['isaacsim']:
            raise ValueError('Runtime physical settings differ from selected release')
        for key in ('usd_path', 'usd_sha256', 'root_prim_path', 'rigid_bodies', 'joints', 'particle_sets'):
            if saved_scene['native_task_bundle'][key] != expected_scene['native_task_bundle'][key]:
                raise ValueError('Runtime native bundle differs: '+key)
        if saved_scene['native_task_bundle'].get('volume_bodies', {}) != expected_scene['native_task_bundle'].get('volume_bodies', {}):
            raise ValueError('Runtime native volume declaration differs')
        if saved_scene['native_task_bundle'].get('elastic_composites', {}) != expected_scene['native_task_bundle'].get('elastic_composites', {}):
            raise ValueError('Runtime native elastic declaration differs')
        if saved_scene['native_task_bundle']['observer']['configuration'] != task:
            raise ValueError('Runtime observer configuration differs from frozen task')
        if runtime['native_state_tensor_device'] != 'cuda:0' or not runtime['physics_engine'].startswith('Isaac Sim native PhysX GPU'):
            raise ValueError('Expected recorded native GPU physics')
        if runtime['gpu_selection']['cuda_visible_devices'] != smoke['gpu_uuid']:
            raise ValueError('Native GPU identity differs from launch receipt')
        report['checks']['release_binding'] = {'passed': True, 'task_id': task['id'],
            'release_manifest_sha256': release_sha, 'dynamic_objects': len(task['asset_metadata']['actors'])}
        receipts = [smoke['runtime_imports'], smoke['runtime_imports_final'],
                    bound(http_root/'server_runtime_imports.json'), bound(private/'runtime_imports.json')]
        roots = {row['snapshot_root'] for row in receipts}
        if len(roots) != 1 or not all(row.get('verified') is True
                and row.get('runtime_imports_verified') is True for row in receipts):
            raise ValueError('Host/server/worker do not bind the same verified source snapshot')
        source = Path(next(iter(roots)))
        bound(source/'source_snapshot_manifest.json')
        report['checks']['source_inventory'] = verify(source)
        report['checks']['source_inventory']['passed'] = True
        media = smoke['media']
        if len(media) != 4 or len({row['camera_id'] for row in media}) != 4:
            raise ValueError('Expected four distinct actual RGB-D cameras')
        cameras = {row['camera_id']: row for row in smoke['initial_observation']['cameras']}
        for row in media:
            camera = cameras[row['camera_id']]
            for kind in ('rgb', 'depth'):
                path = Path(row[kind]['path']).resolve()
                sha = digest(path)
                evidence[str(path)] = sha
                if sha != row[kind]['sha256'] or sha != camera[kind+'_sha256']:
                    raise ValueError('Actual public camera bytes differ from receipt')
                if kind == 'rgb':
                    with Image.open(path) as image:
                        if image.size != (camera['width'], camera['height']):
                            raise ValueError('Public RGB dimensions differ')
                else:
                    depth = np.load(path, allow_pickle=False)
                    if depth.shape != (camera['height'], camera['width']) or not np.isfinite(depth).all() or not (depth > 0).any():
                        raise ValueError('Public depth is not valid finite observed geometry')
        report['checks']['media'] = {'passed': True, 'actual_rgbd_cameras': 4}
        path = private/'physics.jsonl'
        evidence[str(path)] = digest(path)
        with path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        duration = records[-1]['observation']['sim_time_s']
        if not math.isclose(duration, smoke['result']['elapsed_sim_s'], abs_tol=1.e-8):
            raise ValueError('Native recorded duration differs from terminal result')
        report['checks']['native_coverage'] = private_coverage(runtime, records, task, duration)
        if task['asset_metadata'].get('elastic_composites'):
            from scripts.validation.native_elastic_archive_review import review_native_elastic
            report['checks']['native_elastic_archive'] = review_native_elastic(
                private, runtime, task, evidence, expected_source=source)
        else:
            report['checks']['support'] = support_evidence(runtime, records, task,
                dwell_s=task['evaluation']['thresholds']['stable_s'])
        if task['asset_metadata'].get('volume_bodies'):
            from scripts.validation.native_volume_archive_review import review_native_volumes
            report['checks']['native_volume_archive'] = review_native_volumes(private, runtime, task, evidence, expected_source=source)
        report['passed'] = all(row['passed'] is True for row in report['checks'].values())
    except (KeyError, TypeError, ValueError, OSError, AssertionError) as error:
        report['errors'].append({'type': type(error).__name__, 'detail': str(error)})
    report['files_sha256'] = evidence
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--http-root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.http_root, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'output': str(args.output), 'errors': result['errors']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
