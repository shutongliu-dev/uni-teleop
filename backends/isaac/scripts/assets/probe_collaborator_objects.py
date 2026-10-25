"""Native seven-object USD composition, free-fall and support-contact evidence."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_scene():
    from bench.geometry import quaternion_matrix
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    root = f'{DATA_ROOT}/assets/objects/hand_system0/bench_catalog/v1'
    scene = load_json(Path(f'{DATA_ROOT}/assets/scenes/manipulation_collection/v2/household_workcell.scene.json'))
    scene = scene_with_hand(scene, 'sharpa_w01_right')
    scene['schema_version'] = 'collaborator_usd_contact_probe_v1'
    scene['description'] = 'Source USD references, 8 mm gravity drops and a static peg-hole support.'
    scene['objects'], scene['fixtures'], scene['cloth'] = [], [], {'enabled': False}
    ids = ['cube_small', 'blue_cup', 'bench_spoon', 'peg', 'peg_hole', 'rubber_mallet', 'apple']
    positions = [[.3, -.13], [.53, -.13], [.80, -.10], [.3, .25], [.53, .25], [.8, .28], [.3, .58]]
    for name, xy in zip(ids, positions):
        asset = load_task_asset('collab_' + name, root, verify_hashes=True)
        q = [2 ** -.5, 2 ** -.5, 0, 0] if name in ('bench_spoon', 'rubber_mallet') else [1, 0, 0, 0]
        corners = np.array([[x, y, z] for x in (asset['bounds_min_m'][0], asset['bounds_max_m'][0])
                            for y in (asset['bounds_min_m'][1], asset['bounds_max_m'][1])
                            for z in (asset['bounds_min_m'][2], asset['bounds_max_m'][2])])
        minz = float((corners @ quaternion_matrix(q).T)[:, 2].min())
        scene['objects'].append({'id': name, 'asset_id': 'collab_' + name, 'asset_root': root,
                                 'position_m': [*xy, .75 - minz + .008], 'quaternion_wxyz': q})
    scene['fixtures'].append({'id': 'static_peg_hole', 'asset_id': 'collab_peg_hole', 'asset_root': root,
                              'position_m': [.72, .62, .7677033320069313], 'quaternion_wxyz': [1, 0, 0, 0]})
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    scene = probe_scene()
    (output / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    project = Path(__file__).resolve().parents[2]
    paths = list((project / 'bench/simulation/isaacsim_backend').glob('*.py')) + [Path(__file__)]
    sources = {}
    for path in paths:
        relative = path.relative_to(project)
        saved = output / 'source_snapshot' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
        sources[str(path)] = digest(path)
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=1280, render_height=720)
    try:
        from PIL import Image
        from pxr import UsdGeom, UsdPhysics
        Image.fromarray(env.render('overview')).save(output / 'initial.png')
        action = {'joint_positions_rad': env.action_spec()['initial_joint_positions_rad'], 'duration_s': .02}
        inventory = {}
        for category in ('objects', 'fixtures'):
            for name, row in env.scene_info[category].items():
                prim = env.stage.GetPrimAtPath(row['path'])
                inventory[name] = {'rigid_enabled': bool(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()),
                                   'kinematic_enabled': bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()),
                                   'collision_approximations': {p: str(UsdPhysics.MeshCollisionAPI(env.stage.GetPrimAtPath(p)).GetApproximationAttr().Get()) for p in row['collision_paths']},
                                   'references': str(prim.GetMetadata('references'))}
        (output / 'native_collision_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        reports = {}
        for name, override, steps in [('all_seven_settle', None, 100),
                                      ('static_fixture_support', {'objects': {'cube_small': {'position_m': [.745, .62, .82864]}}}, 60)]:
            env.reset(initial_state=override)
            states = [env.capture_state()]
            impulses = {n: 0. for n in env.object_names}
            support_impulse, errors, worst_separation = 0., [], 0.
            with gzip.open(output / (name + '_contacts.jsonl.gz'), 'wt') as contacts:
                for i in range(steps):
                    obs = env.step(action)
                    states.append(env.capture_state())
                    contacts.write(json.dumps(obs['contacts']) + '\n')
                    errors.extend(obs['contacts']['errors'])
                    for pair in obs['contacts']['rigid']:
                        actors = (pair['actor0'], pair['actor1'])
                        value = pair['impulse_magnitude_sum_ns']
                        for obj in impulses:
                            if env.scene_info['objects'][obj]['path'] in actors:
                                impulses[obj] += value
                        if any('/static_peg_hole/' in a or a.endswith('/static_peg_hole') for a in actors) and '/World/Objects/cube_small' in actors:
                            support_impulse += value
                        if any(a.startswith('/World/Objects/') for a in actors):
                            worst_separation = min(worst_separation, float(pair.get('min_separation_m', 0.)))
            arrays = {k: np.stack([s[k] for s in states]) for k in states[0]}
            np.savez_compressed(output / (name + '_states.npz'), **arrays)
            drift = np.linalg.norm(arrays['object_q'][-1, :, :3] - arrays['object_q'][0, :, :3], axis=1)
            speed = np.linalg.norm(arrays['object_qd'][-1, :, :3], axis=1)
            finite = all(np.isfinite(x).all() for x in arrays.values())
            report = {'duration_s': env.time_s, 'all_state_finite': finite,
                      'object_drift_m': dict(zip(env.object_names, drift.tolist())),
                      'final_linear_speed_m_s': dict(zip(env.object_names, speed.tolist())),
                      'native_impulse_by_object_ns': impulses, 'static_fixture_impulse_ns': support_impulse,
                      'contact_errors': errors, 'worst_object_pair_separation_m': worst_separation}
            report['passed'] = finite and not errors and min(impulses.values()) > 0 and max(speed) < .03 and (name == 'all_seven_settle' or support_impulse > 0)
            reports[name] = report
            Image.fromarray(env.render('overview')).save(output / (name + '.png'))
            print(json.dumps({name: report}), flush=True)
        metadata = env.metadata()
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        actual_mass = metadata['native_object_masses_kg']
        source_mass = {n: env.scene_info['objects'][n]['mass_kg'] for n in env.object_names}
        mass_ok = all(abs(actual_mass[n] - source_mass[n]) < 1e-6 for n in actual_mass)
        inertia_ok = all(np.min(np.linalg.eigvalsh(x)) > 0 for x in metadata['native_object_inertias_kg_m2'].values())
        schemas_ok = all(row['rigid_enabled'] and row['kinematic_enabled'] == (n not in env.object_names)
                         and row['collision_approximations'] for n, row in inventory.items())
        sources_ok = all(digest(p) == h for p, h in sources.items())
        result = {'passed': bool(all(r['passed'] for r in reports.values()) and mass_ok and inertia_ok and schemas_ok and sources_ok),
                  'native_isaacsim': True, 'trials': reports, 'source_snapshot_sha256': sources,
                  'sources_unchanged': sources_ok, 'mass_preserved': mass_ok, 'positive_solver_inertias': bool(inertia_ok),
                  'native_collision_schemas_pass': bool(schemas_ok), 'object_writes_during_step': False,
                  'scope': 'Seven selected rigid USD source assets with real PhysX contact and gravity; static fixture support. Not all catalog entries nor cup/peg cavity clearance certification.',
                  'artifact_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': result['passed'], 'output': str(output)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
