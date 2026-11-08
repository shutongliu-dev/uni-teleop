"""Read-only Task7 belt-length, insertion-clearance and identity witnesses.

This is object geometry, not a robot policy, continuous grasp path or tension
calibration. The native drop diagnostic supplies separate physical evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.nist_assets import _belt_outline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perimeter(centers, radii):
    theta = np.linspace(0, 2*np.pi, 16384, endpoint=False)
    unit = np.c_[np.cos(theta), np.sin(theta)]
    points = np.vstack([c+unit*r for c, r in zip(centers, radii)])
    outline = points[ConvexHull(points).vertices]
    return float(np.linalg.norm(np.roll(outline, -1, axis=0)-outline, axis=1).sum())


def segment_distance(point, a, b):
    t = np.clip(np.dot(point-a, b-a)/np.dot(b-a, b-a), 0., 1.)
    return float(np.linalg.norm(point-(a+t*(b-a))))


def audit(metadata, original):
    current = {o['id']: o for o in metadata['objects']}
    previous = {o['id']: o for o in original['objects']}
    goals = metadata['diagnostic_goal_poses']
    names = ['drive_pulley', 'timing_pulley', 'tensioner_idler']
    centers = np.array([goals[n][:2] for n in names])
    flanges = np.array([max(abs(np.asarray(current[n]['local_bounds_m'])[:, :2]).flat) for n in names])
    belt = metadata['task_parameters']['belt']
    length, count = belt['nominal_length_m'], belt['segment_count']
    half_thickness = current['belt_link_000']['analytic_geometry']['size_m'][1]/2
    all_three = perimeter(centers, flanges+half_thickness)
    two = perimeter(centers[:2], flanges[:2]+half_thickness)
    # A closed equal-link witness passing outside both source flanges. This
    # proves a geometrically possible lifted loop, not a manipulation trajectory.
    lo, hi = 0., .02
    for _ in range(60):
        extra = (lo+hi)/2
        nodes, _, residual = _belt_outline(centers[:2], flanges[:2]+half_thickness+extra,
                                          count=count, segment_length=length/count)
        if residual > 0: lo = extra
        else: hi = extra
    distances = [min(segment_distance(center, a, b) for a, b in zip(nodes, np.roll(nodes, -1, axis=0)))
                 for center in centers[:2]]
    clearances = np.asarray(distances)-flanges[:2]-half_thickness
    idler = current['tensioner_idler']
    belt_top = max(p[2] for p in belt['target_centerline_m'])+belt['width_m']/2
    idler_top = goals['tensioner_idler'][2]+idler['local_bounds_m'][1][2]
    revision = metadata['task_parameters']['idler_handling_revision']
    changed = [name for name in current if current[name] != previous[name]]
    checks = {
        'only_idler_body_changed': changed == ['tensioner_idler'],
        'source_cad_unchanged': original['source'] == metadata['source'],
        'only_idler_guide_post_changed': [n for n in original['fixture_metadata']
            if original['fixture_metadata'][n] != metadata['fixture_metadata'][n]] == ['tensioner_idler_shaft'],
        'goal_poses_unchanged': original['diagnostic_goal_poses'] == goals,
        'belt_unchanged': original['task_parameters']['belt'] == belt,
        'seats_unchanged': original['task_parameters']['seating_goals'] == metadata['task_parameters']['seating_goals'],
        'flexible_rules_unchanged': original['task_parameters']['flexible_groups'] == metadata['task_parameters']['flexible_groups'],
        'flat_simultaneous_three_flange_route_too_short': all_three > length,
        'two_installed_flange_route_has_clearance': bool(min(clearances) > .0005),
        'exact_equal_link_closure': abs(residual) < 1.e-8,
        'exposed_upper_hub_above_belt': idler_top-belt_top > .020,
        'unchanged_shaft_fits_real_through_bore': revision['through_bore_diameter_m']/2 > .0037,
        'guidance_is_optional': metadata['task_parameters']['assembly_guidance']['mandatory_sequence'] is False,
        'guide_post_below_grasp_hub_top': abs(idler_top-(metadata['fixture_metadata']['tensioner_idler_shaft']['initial_position_m'][2]
            +metadata['fixture_metadata']['tensioner_idler_shaft']['local_bounds_m'][1][2])-.003)<1.e-8,
    }
    return {'schema':'nist_belt_assembly_geometry.v1', 'passed':all(checks.values()), 'checks':checks,
        'belt_length_m':length, 'source_flange_radii_m':flanges.tolist(),
        'three_installed_flange_lower_bound_m':all_three,
        'three_installed_flange_length_deficit_m':all_three-length,
        'two_installed_flange_lower_bound_m':two, 'two_installed_flange_slack_m':length-two,
        'two_flange_equal_link_witness_xy_m':nodes.tolist(),
        'two_flange_witness_radial_clearance_m':clearances.tolist(),
        'upper_hub_exposed_height_above_belt_m':idler_top-belt_top,
        'straight_bore_radial_clearance_m':revision['through_bore_diameter_m']/2-.0037,
        'lower_entry_radial_clearance_m':revision['lower_entry_diameter_m']/2-.0037,
        'robot_manipulation_qualified':False, 'continuous_motion_qualified':False,
        'scope':'Discrete physical geometry witness and preserved source identities; native seating is separate.',
        'flat_route_limit':'The perimeter bound excludes a flat loop simultaneously outside all three wheels; '
            'it does not rule out all tilted or sequential three-dimensional assembly paths.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--original-metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT) or args.output.exists():
        raise ValueError('Choose fresh shared-dataset evidence')
    report = audit(load_json(args.metadata), load_json(args.original_metadata))
    report['sources'] = {str(p.resolve()): digest(p) for p in (args.metadata, args.original_metadata, Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': report['passed'], 'checks':report['checks'], 'output':str(args.output)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
