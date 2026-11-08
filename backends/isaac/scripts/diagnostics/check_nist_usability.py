"""Read-only size and threading-clearance audit of opt-in NIST assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.nist_assets import ASSET_ROOT, _r, structural_audit
from scenes.workrooms.mechanical_assembly.nist_usability import PROFILE

BASELINES = {'nist_connector_insertion': 'v3', 'nist_cable_routing': 'v5', 'nist_wire_harness': 'v4'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(asset_root, version, tasks=None):
    rows = []
    hand_root = Path(f'{DATA_ROOT}/assets/robots/sharpa/wave_01/upstream/wave_01/right_sharpa_wave/meshes')
    hand = []
    for name in ['elastomer_HB1_4F.STL', 'elastomer_HB1_TH.STL']:
        path = hand_root/name
        mesh = trimesh.load(path, force='mesh')
        hand.append({'source_path': str(path), 'sha256': sha(path), 'actual_mesh_extents_mm': (mesh.extents*1000).tolist()})
    for task in tasks or BASELINES:
        old_version = BASELINES[task]
        root = asset_root/task/version
        old_root = ASSET_ROOT/task/old_version
        meta = load_json(root/'metadata.json')
        old = load_json(old_root/'metadata.json')
        stage = Usd.Stage.Open(str(root/'task.usda'))
        checks = {'explicit_profile': meta.get('usability_profile') == PROFILE,
                  'passive_topology': structural_audit(stage, meta)['passed'],
                  'unchanged_workplate': meta['fixture_metadata']['workplate'] == old['fixture_metadata']['workplate']}
        source = {r['path']: r for r in old['source']['external_assets']}
        checks['original_source_files_unchanged'] = all(sha(r['path']) == r['sha256'] == source[r['path']]['sha256'] for r in meta['source']['external_assets'])
        before = {o['id']: o for o in old['objects']}
        objects = {o['id']: o for o in meta['objects']}
        dimensions = []
        for name, obj in objects.items():
            mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(obj['body_path'])).GetMassAttr().Get()
            checks['authored_mass:'+name] = bool(np.isclose(mass, obj['mass_kg'], rtol=1e-6))
            if 'link' in name:
                continue
            bounds = np.array(obj['local_bounds_m'])
            points = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
            world = points@_r(obj['initial_pose_xyzw'][3:]).T+obj['initial_pose_xyzw'][:3]
            checks['table_xy_support:'+name] = bool(np.all(world.min(0)[:2] >= [.15, -.5]) and np.all(world.max(0)[:2] <= [1.15, .5]))
            dimensions.append({'id': name, 'old_dimensions_mm': (np.array(before[name]['half_extents_m'])*2000).tolist(),
                'new_dimensions_mm': (np.array(obj['half_extents_m'])*2000).tolist(),
                'old_mass_kg': before[name]['mass_kg'], 'new_mass_kg': obj['mass_kg'],
                'new_density_preserving_mass_ratio': obj['mass_kg']/before[name]['mass_kg'],
                'source_below_part_floor_gap_m': float(world.min(0)[2]-.790)})
        row = {'task_id': task, 'asset_path': str(root), 'asset_manifest_sha256': sha(root/'manifest.json'),
               'usd_sha256': sha(root/'task.usda'), 'metadata_sha256': sha(root/'metadata.json'), 'checks': checks, 'dimensions': dimensions}
        if task == 'nist_cable_routing':
            points = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(objects['cable_end']['collider_paths'][0])).GetPointsAttr().Get())
            guide = meta['fixture_metadata']['guide_0']
            tube = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(guide['collider_paths'][0])).GetPointsAttr().Get())
            # Inner surfaces are triangles, so use their closest radial point,
            # not only the nominal bore vertices (conservative for faceting).
            faces = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(guide['collider_paths'][0])).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            nominal_bore = np.linalg.norm(tube[:, [0, 2]], axis=1).min()
            inner = tube[faces][:, :, [0, 2]]
            radii = np.linalg.norm(inner, axis=2)
            candidates = inner[np.all(radii < nominal_bore+.0001, axis=1)]
            bore = float(nominal_bore)
            for triangle in candidates:
                for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
                    v = end-start
                    t = np.clip(-np.dot(start, v)/max(float(np.dot(v, v)), 1e-30), 0., 1.)
                    bore = min(bore, float(np.linalg.norm(start+t*v)))
            plug = float(np.linalg.norm(points[:, :2], axis=1).max())
            row['threading_envelope'] = {'guide_minimum_faceted_bore_radius_m': bore,
                'entire_plug_maximum_radial_extent_m': plug, 'radial_clearance_m': bore-plug,
                'scope': 'Straight coaxial geometric passage of the entire audio plug through either guide, without a hand; no motion or grasp qualification.'}
            checks['complete_audio_plug_threads_guide'] = bore-plug > .002
            gate = meta['task_parameters']['flexible_groups'][0]['route_gates'][0]['radius_m']
            wire_radius = meta['task_parameters']['cable']['diameter_m']/2
            row['threading_envelope']['route_centerline_gate_radius_m'] = gate
            row['threading_envelope']['centerline_gate_margin_over_bore_minus_wire_m'] = gate-(bore-wire_radius)
            checks['gate_tracks_physical_free_bore'] = 0 <= gate-(bore-wire_radius) <= .0007
        if task == 'nist_wire_harness':
            checks['reversible_terminal_fit'] = all(
                np.all((np.array(meta['fixture_metadata'][name+'_terminal']['half_extents_m'])-obj['half_extents_m'])[:2]*2-.006 >= .003-1e-9)
                for name, obj in objects.items() if '_end_' in name)
            checks['physical_exit_exceeds_wire'] = all(f['cable_exit']['width_m']-f['cable_exit']['wire_diameter_m'] >= .0044
                for f in meta['fixture_metadata'].values() if 'cable_exit' in f)
            row['terminal_fit_scope'] = 'Both 180-degree endpoint orientations fit the same outer cavity; actual wire-exit clearance has separate CPU tests.'
        row['passed'] = all(checks.values())
        rows.append(row)
    return {'schema': 'mechanical_nist_usability_geometry.v1', 'passed': all(r['passed'] for r in rows),
            'tasks': rows, 'official_hand_pad_dimensions': hand,
            'scope': 'Static imported geometry, source identity, passive anchors, density-consistent size and threading checks. No hand-object, native contact or task-completion qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, default=ASSET_ROOT)
    parser.add_argument('--version', required=True)
    parser.add_argument('--task', action='append', choices=tuple(BASELINES))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs under shared datasets')
    value = audit(args.asset_root, args.version, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': value['passed'], 'output': str(args.output)}))
    raise SystemExit(0 if value['passed'] else 1)


if __name__ == '__main__':
    main()
