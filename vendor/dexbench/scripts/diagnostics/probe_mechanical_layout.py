"""CPU-only endpoint IK and source-collider separation for workshop layout.

This never launches Isaac, calls a model, changes assets or claims physical
manipulation acceptance.  Positive separating projections lower-bound source
mesh separation; unresolved envelopes are not penetration measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.geometry import rpy_matrix, transform
from scenes.workrooms.mechanical_assembly.robot_layout import (
    LAYOUT_ID, geometry_model, link_world_transforms, ready_configuration,
    robot_asset_reference, robot_configuration, solve_wrist, stand_geometry,
    workspace_contract,
)


def collision_meshes(robot, initial_state):
    """All original arm/hand collision meshes, with source local transforms."""
    import trimesh

    model = geometry_model(robot)
    poses = link_world_transforms(robot, initial_state)
    rows = []
    for path in (model.arm_path, model.hand_path):
        for link in ET.parse(path).getroot().findall('link'):
            name = link.get('name')
            if name not in poses:
                continue
            for index, collision in enumerate(link.findall('collision')):
                shape = collision.find('geometry/mesh')
                if shape is None:
                    raise ValueError('Unhandled source collider: '+name)
                origin = collision.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                source = (path.parent/shape.get('filename')).resolve()
                mesh = trimesh.load(source, force='mesh', process=False)
                vertices = np.asarray(mesh.vertices)*np.fromstring(shape.get('scale', '1 1 1'), sep=' ')
                pose = poses[name] @ transform(xyz, rpy_matrix(rpy))
                world = vertices @ pose[:3, :3].T + pose[:3, 3]
                rows.append({'name': name, 'collision_index': index,
                             'source': str(source), 'vertices': world})
    return rows


def projection_distance_lower_bound(first, second):
    """Positive source-convex-envelope separation; not cooked PhysX distance."""
    from scipy.spatial import ConvexHull

    coarse = np.maximum(first.min(0)-second.max(0), second.min(0)-first.max(0)).max()
    if coarse > .025:
        return float(coarse)
    one, two = ConvexHull(first), ConvexHull(second)
    axes = np.vstack((np.eye(3), one.equations[:, :3], two.equations[:, :3]))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    a, b = first[one.vertices] @ axes.T, second[two.vertices] @ axes.T
    return float(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)).max())


def box_vertices(center, size):
    return np.array(list(itertools.product((-1., 1.), repeat=3)))*np.asarray(size)/2+center


def mesh_clearances(robot, initial_state):
    rows = collision_meshes(robot, initial_state)
    arm = [r for r in rows if r['name'].startswith('fr3_')
           and r['name'] not in ('fr3_link7', 'fr3_link8')]
    hand = [r for r in rows if not r['name'].startswith('fr3_')]
    comparisons = []
    for first in hand:
        for second in arm:
            comparisons.append({'first': first['name'], 'second': second['name'],
                'lower_bound_m': projection_distance_lower_bound(first['vertices'], second['vertices'])})
    for first in hand:
        if first['name'] in ('right_hand_wrist', 'right_hand_C_MC'):
            continue
        for second in [r for r in rows if r['name'] == 'fr3_link7']:
            comparisons.append({'first': first['name'], 'second': second['name'],
                'lower_bound_m': projection_distance_lower_bound(first['vertices'], second['vertices'])})
    table = box_vertices([.65, 0., .76], [1., 1., .04])
    table_checks = [{'link': r['name'], 'lower_bound_m': projection_distance_lower_bound(r['vertices'], table)}
                    for r in rows]
    stand_checks = []
    for part in stand_geometry()['boxes']:
        box = box_vertices(part['position_m'], part['size_m'])
        for row in rows:
            # fr3_link0 is bolted to the task-owned stand and intentionally adjoins it.
            if row['name'] in ('fr3_link0', 'fr3_base'):
                continue
            stand_checks.append({'link': row['name'], 'stand_part': part['name'],
                'lower_bound_m': projection_distance_lower_bound(row['vertices'], box)})
    vertices = np.vstack([r['vertices'] for r in rows])
    return {'source_collision_mesh_count': len(rows),
            'robot_bounds_m': [vertices.min(0).tolist(), vertices.max(0).tolist()],
            'minimum_hand_nonadjacent_arm_gap_lower_bound_m': min(r['lower_bound_m'] for r in comparisons),
            'minimum_table_gap_lower_bound_m': min(r['lower_bound_m'] for r in table_checks),
            'minimum_moving_link_stand_gap_lower_bound_m': min(r['lower_bound_m'] for r in stand_checks),
            'hand_arm_checks': comparisons, 'table_checks': table_checks, 'stand_checks': stand_checks,
            'scope': 'Waiting pose, original source collision envelopes only; excludes adjacent links, self-collision within fingers, room furniture, task objects, contact offsets and native cooked shapes.'}


def probe():
    robot = robot_configuration()
    initial, ready = ready_configuration(robot)
    model, contract = geometry_model(robot), workspace_contract()
    grid = contract['wrist_probe_grid_m']
    points = [('workspace', list(point)) for point in itertools.product(grid['x'], grid['y'], grid['z'])]
    points += [('screw', point) for point in contract['screw_wrist_probe_points_m']]
    probes = []
    for label, point in points:
        solved = solve_wrist(model, point)
        joints = solved['joint_positions_rad']
        probes.append({'kind': label, 'wrist_position_m': point,
            'rpy_rad': contract['wrist_probe_rpy_rad'],
            'converged': bool(solved['converged']),
            'position_error_m': float(solved['position_error_m']),
            'rotation_error_rad': float(solved['rotation_error_rad']),
            'minimum_arm_joint_margin_rad': min(min(joints[j.name]-j.lower, j.upper-joints[j.name])
                                               for j in model.arm.active),
            'joint_positions_rad': joints})
    meshes = mesh_clearances(robot, initial)
    source_files = [Path(__file__), Path(__file__).parents[2]/'scenes/workrooms/mechanical_assembly/robot_layout.py',
                    model.arm_path, model.hand_path]
    return {'schema': 'mechanical_layout_cpu_probe.v1', 'layout_id': LAYOUT_ID,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'passed': all(p['converged'] and p['minimum_arm_joint_margin_rad'] > .30 for p in probes)
                      and meshes['minimum_hand_nonadjacent_arm_gap_lower_bound_m'] > .025
                      and meshes['minimum_table_gap_lower_bound_m'] > .025
                      and meshes['minimum_moving_link_stand_gap_lower_bound_m'] > .025,
            'robot': robot, 'initial_state': initial, 'ready': ready,
            'stand': stand_geometry(), 'workspace': contract, 'endpoint_probes': probes,
            'ready_source_mesh_checks': meshes, 'robot_asset': robot_asset_reference(robot),
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            'model_calls': 0, 'native_runtime_started': False, 'native_qualified': False,
            'manipulation_qualified': False,
            'scope': 'Discrete wrist IK and selected waiting-pose source-mesh gaps only. No swept-volume, grasp, tool-contact, assembly or native physics acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'output': str(args.output),
                      'endpoint_count': len(result['endpoint_probes']),
                      'minimum_hand_arm_gap_m': result['ready_source_mesh_checks']['minimum_hand_nonadjacent_arm_gap_lower_bound_m']}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
