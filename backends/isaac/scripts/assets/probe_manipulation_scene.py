"""Native physical-asset evidence: settle each object and load each static support."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    scene_path = Path(f'{DATA_ROOT}/assets/scenes/manipulation_collection/v2/household_workcell.scene.json')
    scene = scene_with_hand(load_json(scene_path), 'sharpa_w01_right')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    sources = list(Path(__file__).resolve().parents[2].joinpath('bench/simulation/isaacsim_backend').glob('*.py')) + [Path(__file__)]
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        (snapshot / path.name).write_bytes(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    (args.output / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=1280, render_height=720)
    try:
        from PIL import Image
        from pxr import UsdPhysics
        Image.fromarray(env.render('overview')).save(args.output / 'initial.png')
        action = {'joint_positions_rad': env.action_spec()['initial_joint_positions_rad'], 'duration_s': .02}
        actual_shapes = {}
        for kind in ('objects', 'fixtures'):
            for name, item in env.scene_info[kind].items():
                root = env.stage.GetPrimAtPath(item['path'])
                actual_shapes[name] = {
                    'dynamic': root.HasAPI(UsdPhysics.RigidBodyAPI),
                    'collisions': {p: bool(env.stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI)) for p in item['collision_paths']},
                }
        (args.output / 'native_collision_inventory.json').write_text(json.dumps(actual_shapes, indent=2) + '\n')
        trials = [('household_settle', None, None, 60)]
        # The block starts 8 mm above a real surface; only reset initializes its pose.
        for name, top_z in [('storage_tray', .758), ('parts_rack', .828), ('cloth_drape_cylinder', .95)]:
            fixture = next(x for x in scene['fixtures'] if x['id'] == name)
            trials.append((name, [*fixture['position_m'][:2], top_z + .023 + .008], name, 40))
        reports = {}
        for trial, block_position, support, count in trials:
            override = {'objects': {'wooden_block': {'position_m': block_position}}} if block_position else None
            env.reset(initial_state=override)
            states = [env.capture_state()]
            impulse = {n: 0. for n in env.object_names}
            support_impulse, errors = 0., []
            with gzip.open(args.output / (trial + '_contacts.jsonl.gz'), 'wt') as stream:
                for _ in range(count):
                    obs = env.step(action)
                    states.append(env.capture_state())
                    stream.write(json.dumps(obs['contacts']) + '\n')
                    errors.extend(obs['contacts']['errors'])
                    for pair in obs['contacts']['rigid']:
                        actors = (pair['actor0'], pair['actor1'])
                        value = pair['impulse_magnitude_sum_ns']
                        for name in impulse:
                            if any(a == env.scene_info['objects'][name]['path'] for a in actors):
                                impulse[name] += value
                        if support and any('/' + support + '/' in a or a.endswith('/' + support) for a in actors) and any(a.endswith('/wooden_block') for a in actors):
                            support_impulse += value
            arrays = {k: np.stack([s[k] for s in states]) for k in states[0]}
            np.savez_compressed(args.output / (trial + '_states.npz'), **arrays)
            drift = {n: float(np.linalg.norm(arrays['object_q'][-1, i, :3] - arrays['object_q'][0, i, :3])) for i, n in enumerate(env.object_names)}
            finite = all(np.isfinite(x).all() for x in arrays.values())
            reports[trial] = {'duration_s': env.time_s, 'all_state_finite': finite, 'object_drift_m': drift,
                              'native_impulse_by_object_ns': impulse, 'support_load_impulse_ns': support_impulse, 'contact_errors': errors,
                              'passed': finite and not errors and (all(x > 0 for x in impulse.values()) and max(drift.values()) < .01 if support is None else support_impulse > 0 and drift['wooden_block'] < .03)}
            Image.fromarray(env.render('task')).save(args.output / (trial + '.png'))
            print(json.dumps({trial: reports[trial]}), flush=True)
        metadata = env.metadata()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        Image.fromarray(env.render('overview')).save(args.output / 'overview.png')
        schemas_pass = all(all(x['collisions'].values()) and x['dynamic'] == (n in env.object_names) for n, x in actual_shapes.items())
        sources_unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        result = {'native_isaacsim': True, 'trials': reports, 'native_collision_schemas_pass': schemas_pass,
                  'sources_unchanged': sources_unchanged, 'source_snapshot_sha256': hashes,
                  'object_writes_during_step': False, 'passed': schemas_pass and sources_unchanged and all(x['passed'] for x in reports.values()),
                  'scope': 'Native compound shape import and passive stability, plus load-bearing contact on all three static fixtures. No task policy or calibrated materials.'}
        result['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': result['passed'], 'output': str(args.output)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
