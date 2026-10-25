"""Independent raw-state/contact audit for the additional physical scene probes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

from bench.paths import load_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    result = load_json(root / 'result.json')
    meta = load_json(root / 'metadata.json')
    checks = {'artifact_hashes': all(sha(root / name) == h for name, h in result['artifact_sha256'].items()),
              'source_snapshot': result['sources_unchanged'] and all(sha(root / 'source_snapshot' / Path(p).name) == h for p, h in result['source_snapshot_sha256'].items())}
    details = {}
    if 'trials' in result:
        inventory = load_json(root / 'native_collision_inventory.json')
        checks['all_ten_assets_have_native_colliders'] = len(inventory) == 10 and all(x['collisions'] and all(x['collisions'].values()) for x in inventory.values())
        checks['seven_dynamic_three_static'] = sum(x['dynamic'] for x in inventory.values()) == 7
        for trial in result['trials']:
            arrays = np.load(root / (trial + '_states.npz'))
            finite = all(np.isfinite(arrays[k]).all() for k in arrays)
            impulses = {n: 0. for n in meta['object_names']}
            load, errors = 0., []
            with gzip.open(root / (trial + '_contacts.jsonl.gz'), 'rt') as stream:
                for line in stream:
                    contacts = json.loads(line)
                    errors.extend(contacts['errors'])
                    for pair in contacts['rigid']:
                        actors = (pair['actor0'], pair['actor1'])
                        for name in impulses:
                            if meta['scene_info']['objects'][name]['path'] in actors:
                                impulses[name] += pair['impulse_magnitude_sum_ns']
                        if trial != 'household_settle' and any('/' + trial + '/' in a or a.endswith('/' + trial) for a in actors) and any(a.endswith('/wooden_block') for a in actors):
                            load += pair['impulse_magnitude_sum_ns']
            drift = np.linalg.norm(arrays['object_q'][-1, :, :3] - arrays['object_q'][0, :, :3], axis=1)
            checks[trial] = finite and not errors and (min(impulses.values()) > 0 and max(drift) < .01 if trial == 'household_settle' else load > 0 and drift[meta['object_names'].index('wooden_block')] < .03)
            details[trial] = {'native_impulse_by_object_ns': impulses, 'support_load_impulse_ns': load, 'max_object_drift_m': float(max(drift))}
    else:
        arrays = np.load(root / 'states.npz')
        q = arrays['particle_q']
        spec = meta['scene_spec']['cloth']
        nx, ny = spec['cells_xy']
        a = np.arange((nx+1)*(ny+1)).reshape(ny+1,nx+1)[:-1,:-1].ravel()
        triangles = np.concatenate([np.column_stack([a,a+1,a+nx+1]), np.column_stack([a+1,a+nx+2,a+nx+1])])
        edges = np.unique(np.sort(np.concatenate([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]]), axis=1), axis=0)
        lengths = np.linalg.norm(q[:, edges[:,0]] - q[:, edges[:,1]], axis=2)
        ratio = lengths / lengths[0]
        area = np.linalg.norm(np.cross(q[:, triangles[:,1]] - q[:, triangles[:,0]], q[:, triangles[:,2]] - q[:, triangles[:,0]]), axis=2)
        area_ratio = area / area[0]
        centered = q[-1] - np.mean(q[-1], axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_rms = float(np.sqrt(np.mean((centered @ vh[-1])**2)))
        fixture = next(x for x in meta['scene_spec']['fixtures'] if x['asset_id'] == 'cloth_drape_cylinder')
        center = np.asarray(fixture['position_m'])
        radial = np.linalg.norm(q[-1, :, :2] - center[:2], axis=1)
        top = center[2] + .20
        penetration = np.maximum(0, np.minimum(np.minimum(.06-radial, top-q[-1,:,2]), q[-1,:,2]-center[2]))
        supported = int(np.count_nonzero((radial < .04) & (np.abs(q[-1,:,2]-top) < .004)))
        checks.update(all_state_finite=all(np.isfinite(arrays[k]).all() for k in arrays),
                      deforms_in_3d=plane_rms > .01, gravity_drop=float(np.mean(q[0,:,2])-np.mean(q[-1,:,2])) > .05,
                      no_large_edge_strain=bool(np.min(ratio) > .90 and np.max(ratio) < 1.10),
                      no_degenerate_triangles=bool(np.min(area_ratio) > .5),
                      cylinder_surface_support_witness=supported > 10,
                      cylinder_penetration_under_2mm=float(np.max(penetration)) < .002,
                      no_cloth_attachment=not meta['cloth_attachment'],
                      no_object_playback=not meta['object_pose_writeback_after_reset'])
        details = {'cloth_vertex_count': int(q.shape[1]), 'final_best_fit_plane_rms_m': plane_rms,
                   'edge_length_ratio_min_max': [float(np.min(ratio)), float(np.max(ratio))],
                   'triangle_area_ratio_min': float(np.min(area_ratio)), 'cylinder_top_witness_vertices': supported,
                   'max_cylinder_geometric_penetration_m': float(np.max(penetration)),
                   'evidence_limit': 'Cloth support is a live-state geometric witness. Native rigid contact API does not expose cloth contact impulses.'}
    checks = {key: bool(value) for key, value in checks.items()}
    return {'passed': all(checks.values()), 'recording': str(root), 'checks': checks, 'details': details,
            'result_sha256': sha(root / 'result.json'), 'auditor_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.recording)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
