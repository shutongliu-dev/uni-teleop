"""Discrete CPU source-mesh preflight for the mechanical tool motion candidate.

No native runtime is imported.  This is a selected environment-clearance check,
not a continuous collision guarantee or a validation of physical tool grasping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.geometry import rpy_matrix, transform
from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, robot_configuration, stand_geometry
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import candidate_motion_plan, c2_fraction
from scripts.diagnostics.probe_mechanical_layout import box_vertices, projection_distance_lower_bound


def source_meshes(model):
    import trimesh

    rows = []
    for path in (model.arm_path, model.hand_path):
        for link in ET.parse(path).getroot().findall('link'):
            for collision in link.findall('collision'):
                shape = collision.find('geometry/mesh')
                if shape is None:
                    raise ValueError('Unhandled non-mesh source collision geometry')
                origin = collision.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                source = (path.parent/shape.get('filename')).resolve()
                mesh = trimesh.load(source, force='mesh', process=False)
                vertices = np.asarray(mesh.vertices)*np.fromstring(shape.get('scale', '1 1 1'), sep=' ')
                local = transform(xyz, rpy_matrix(rpy))
                rows.append({'name': link.get('name'), 'source': str(source),
                             'vertices': vertices @ local[:3, :3].T+local[:3, 3]})
    return rows


def posed_meshes(model, source, joints):
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return {row['name']: row['vertices'] @ poses[row['name']][:3, :3].T+poses[row['name']][:3, 3]
            for row in source}


def selected_clearances(vertices):
    table = box_vertices([.65, 0., .76], [1., 1., .04])
    stand = [(part['name'], box_vertices(part['position_m'], part['size_m']))
             for part in stand_geometry()['boxes']]
    checks = []
    for name, first in vertices.items():
        checks.append(('table', name, 'table', projection_distance_lower_bound(first, table)))
        if name not in ('fr3_link0', 'fr3_base'):
            checks.extend(('stand', name, label, projection_distance_lower_bound(first, box))
                          for label, box in stand)
        if name.startswith('fr3_'):
            continue
        for arm_name, second in vertices.items():
            if not arm_name.startswith('fr3_') or arm_name in ('fr3_link7', 'fr3_link8'):
                continue
            checks.append(('hand_arm', name, arm_name, projection_distance_lower_bound(first, second)))
        if name not in ('right_hand_wrist', 'right_hand_C_MC'):
            checks.append(('hand_arm', name, 'fr3_link7',
                           projection_distance_lower_bound(first, vertices['fr3_link7'])))
    return {kind: dict(zip(('kind', 'first', 'second', 'lower_bound_m'),
                          min((r for r in checks if r[0] == kind), key=lambda r: r[3])))
            for kind in ('table', 'stand', 'hand_arm')}


def probe(plan=None, *, samples_per_transition=21):
    if samples_per_transition < 3:
        raise ValueError('Use at least three discrete samples per transition')
    robot = robot_configuration()
    model = geometry_model(robot)
    plan = candidate_motion_plan(robot) if plan is None else plan
    source = source_meshes(model)
    previous = plan['initial_joint_positions_rad']
    samples, speeds = [], []
    for phase in plan['phases']:
        target = phase['joint_positions_rad']
        duration = float(phase['duration_s'])
        arm_peak = max(1.875*abs(target[n]-previous[n])/duration for n in model.arm.names)
        hand_peak = max(1.875*abs(target[n]-previous[n])/duration for n in model.hand.names)
        speeds.append({'phase': phase['name'], 'arm_peak_rad_s': arm_peak, 'hand_peak_rad_s': hand_peak})
        fractions = [1.] if target == previous else np.linspace(0., 1., samples_per_transition)
        for u in fractions:
            weight = c2_fraction(u)
            q = {n: previous[n]+weight*(target[n]-previous[n]) for n in model.names}
            model.checked_joints(q, enforce_limits=True)
            gaps = selected_clearances(posed_meshes(model, source, q))
            samples.append({'phase': phase['name'], 'fraction': float(u), 'clearances': gaps})
        previous = target
    minimum = {kind: min((dict(row['clearances'][kind], phase=row['phase'], fraction=row['fraction'])
                         for row in samples), key=lambda row: row['lower_bound_m'])
               for kind in ('table', 'stand', 'hand_arm')}
    passed = (all(row['lower_bound_m'] > .025 for row in minimum.values())
              and all(s['arm_peak_rad_s'] <= 1.20000001 and s['hand_peak_rad_s'] <= 1.20000001 for s in speeds))
    files = [Path(__file__), Path(__file__).parents[2]/
             'scenes/workrooms/mechanical_assembly/tasks/fastener_assembly/motion_plan.py',
             model.arm_path, model.hand_path]
    return {'schema': 'mechanical_tool_motion_cpu_preflight.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'selected_checks_passed': passed, 'plan': plan,
            'samples_per_transition': samples_per_transition, 'sample_count': len(samples),
            'minimum_selected_source_clearances': minimum, 'phase_peak_target_speeds': speeds,
            'source_mesh_count': len(source),
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'samples': samples, 'native_runtime_started': False, 'model_calls': 0,
            'native_qualified': False, 'manipulation_qualified': False,
            'scope': 'Discrete source-mesh projections for table, stand and hand/nonadjacent arm only. '
                     'Excludes hand self-collision, adjacent arm links, other room furniture, fixtures, '
                     'bolt/tool contact and PhysX cooked shapes. Finger/tool checks are separate. '
                     'A positive result is not continuous or native collision acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-transition', type=int, default=21)
    args = parser.parse_args()
    result = probe(samples_per_transition=args.samples_per_transition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'selected_checks_passed': result['selected_checks_passed'],
                      'sample_count': result['sample_count'], 'output': str(args.output)}))
    if not result['selected_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
