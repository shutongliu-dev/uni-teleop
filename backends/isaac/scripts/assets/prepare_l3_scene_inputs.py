"""Prepare reset-only L3 top-ring entry and outside-shaft rejection probes.

The scene, assets, robot, solver and task thresholds come from the exact pinned
release. Only case reset poses differ. This is a legacy validate_assembly_contacts
input, with no robot phases or manipulation-success claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from bench.isaacsim.release import digest, task_inputs
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset


RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
               'mechanical_task_validation_20261019/shaft_spacer_candidate_v2')
RELEASE_SHA256 = 'ca7a0b5453a8b234001004404f96b410da9e4fff596586623237bb55a2d7b8a6'
TASK_ID = 'shaft_spacer_assembly'
PARTS = ('spacer', 'wide_ring', 'top_ring')
ENTRY_CLEARANCE_M = .0002
LOWER_CLEARANCE_M = .0015
GUIDED_OFFSET_M = .007


def build_cases(scene, task, geometry):
    """Declare checks before native execution; never change physical inputs."""
    e = task['evaluation']
    target = e['target_bottom_centers_m']
    supply = e['initial_objects']
    tip_z = geometry['shaft_tip_world_z_m']
    shoulder_z = geometry['shoulder_world_z_m']
    x, y = geometry['shaft_center_xy_m']
    release_z = tip_z + ENTRY_CLEARANCE_M
    fall = release_z - target['top_ring'][2]
    gravity = abs(scene['physics']['gravity_m_s2'][2])
    predicted_speed = math.sqrt(2 * gravity * fall)
    if not 0 < fall or predicted_speed >= e['maximum_object_linear_speed_m_s']:
        raise ValueError('Top-ring free-fall estimate exceeds the unchanged speed budget')

    def assertion(name, position, support, minimum_descent):
        return {'position_m': list(position),
                'position_tolerance_m': [e['maximum_axis_offset_m']] * 2
                    + [e['maximum_bottom_height_error_m']],
                'maximum_tilt_rad': e['maximum_axis_tilt_rad'],
                'support_path_prefix': support,
                'minimum_descent_m': minimum_descent}

    cases = []
    for label, offset in (('centered', 0.), ('guided', GUIDED_OFFSET_M)):
        reset, assertions = {}, {}
        for index, name in enumerate(PARTS):
            position = list(target[name])
            position[2] += LOWER_CLEARANCE_M * (index + 1)
            if name == 'top_ring':
                position = [x + offset, y, release_z]
            reset[name] = {'position_m': position, 'quaternion_wxyz': [1., 0., 0., 0.]}
            support = ('/World/Fixtures/shaft_fixture' if index == 0
                       else '/World/Objects/' + PARTS[index - 1])
            assertions[name] = assertion(name, target[name], support,
                max(.001, position[2] - target[name][2] - .0005))
        if offset:
            assertions['top_ring'].update(require_guidance_contact=True,
                guidance_path_prefix='/World/Fixtures/shaft_fixture')
        cases.append({'id': 'top_ring_' + label + '_entry', 'duration_s': 2.,
            'reset_objects': reset, 'assertions': assertions,
            'expected_functional_success': True,
            'scope': 'Top ring enters above the original tip; lower two dynamic rings '
                     'start near their own seats with 1.5 mm reset gaps. No robot motion.'})

    # An above-tip outside-bore drop can fall 100 mm to the shoulder/table.
    # Instead test a clean, low-energy beside-shaft rejection on the ORIGINAL
    # shoulder. The other rings stay on their own original supply rests.
    offset = geometry['shaft_radius_m'] + geometry['top_ring_outer_radius_m'] + .001
    if offset >= geometry['shoulder_size_xy_m'][0] / 2:
        raise ValueError('Outside-shaft ring COM must remain above the original shoulder')
    reset = deepcopy(supply)
    assertions = {}
    for name in PARTS:
        assertions[name] = assertion(name, supply[name]['position_m'],
            '/World/Fixtures/' + name + '_supply_rest', 0.)
    reset['top_ring']['position_m'] = [x + offset, y, shoulder_z + LOWER_CLEARANCE_M]
    assertions['top_ring'] = assertion('top_ring', [x, y, shoulder_z],
        '/World/Fixtures/shaft_fixture', .001)
    cases.append({'id': 'top_ring_outside_shaft_rejection', 'duration_s': 2.,
        'reset_objects': reset, 'assertions': assertions,
        'expected_functional_success': False,
        'negative_control_objects': ['top_ring'],
        'scope': 'Top ring alone beside the shaft on its original shoulder, '
                 'not an above-tip rejection or failed-guidance claim. Other rings '
                 'remain at their canonical supply reset. Wrong-pose/descent '
                 'rejection is assessed for the active top ring only; all bodies '
                 'still retain full motion, support and stability checks.'})
    return cases, {
        'entry_clearance_above_tip_m': ENTRY_CLEARANCE_M,
        'lower_ring_reset_clearance_per_layer_m': LOWER_CLEARANCE_M,
        'top_ring_total_free_fall_to_seat_m': fall,
        'top_ring_ballistic_speed_estimate_m_s': predicted_speed,
        'estimate_is_not_native_speed_or_safety_acceptance': True,
        'outside_shaft_center_offset_m': offset,
        'outside_shaft_source_geometry_gap_m': .001,
        'outside_shaft_release_height_above_shoulder_m': LOWER_CLEARANCE_M,
        'outside_shaft_is_above_tip_case': False,
        'all_rings_above_tip_qualified': False,
        'spacer_canonical_above_tip_ballistic_speed_m_s': math.sqrt(
            2 * gravity * (release_z - target['spacer'][2])),
        'reason_for_top_ring_scope': 'The 100 mm shaft leaves the spacer a 100.2 mm '
            'drop even with only 0.2 mm tip clearance. Lower dynamic rings reduce '
            'the top-ring drop to 50.2 mm without added supports or changed physics.',
    }


def prepare(release=RELEASE):
    release = Path(release).resolve()
    if digest(release / 'release_manifest.json') != RELEASE_SHA256:
        raise ValueError('This supplement requires the exact pinned L3 candidate_v2')
    manifest = load_json(release / 'release_manifest.json')
    _, scene, task, initial = task_inputs(release, TASK_ID)
    if tuple(o['id'] for o in scene['objects']) != PARTS:
        raise ValueError('Only the original three-ring inventory is supported')
    fixture = next(f for f in scene['fixtures'] if f['id'] == 'shaft_fixture')
    asset = load_task_asset(fixture['asset_id'], fixture['asset_root'], verify_hashes=True)
    features = {f['label']: f for f in asset['geometry_features']}
    shaft = features['shaft']['profile_z_inner_apothem_outer_circumradius_m']
    top_spec = next(o for o in scene['objects'] if o['id'] == 'top_ring')
    top_asset = load_task_asset(top_spec['asset_id'], top_spec['asset_root'], verify_hashes=True)
    geometry = {
        'shaft_center_xy_m': fixture['position_m'][:2],
        'shaft_tip_world_z_m': fixture['position_m'][2] + shaft[-1][0],
        'shoulder_world_z_m': fixture['position_m'][2] + asset['geometry_contract']['shoulder_z_m'],
        'shaft_radius_m': max(row[2] for row in shaft),
        'shaft_tip_radius_m': shaft[-1][2],
        'shaft_taper_height_m': shaft[-1][0] - shaft[-2][0],
        'shaft_exposed_height_m': asset['geometry_contract']['exposed_height_m'],
        'shoulder_size_xy_m': features['shoulder']['size_m'][:2],
        'top_ring_outer_radius_m': top_asset['dimensions_m'][0] / 2,
        'top_ring_bore_radius_m': top_asset['geometry_contract']['minimum_bore_radius_m'],
        'top_ring_lower_entry_radius_m': top_asset['geometry_features'][0]
            ['profile_z_inner_apothem_outer_circumradius_m'][0][1],
    }
    cases, reasoning = build_cases(scene, task, geometry)
    sources = {}
    for spec in scene['objects']:
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        matches = [p for p in asset['source_dependencies_sha256']
                   if p.endswith('/mechanical_l3_' + spec['id'] + '/visual.obj')]
        if len(matches) != 1:
            raise ValueError('Missing unique original source mesh for ' + spec['id'])
        sources[spec['id']] = matches[0]
    return {
        'schema': 'mechanical_l3_scene_supplement.v1',
        'task_id': TASK_ID, 'release': str(release),
        'release_manifest_sha256': RELEASE_SHA256,
        'scene': scene, 'task': task, 'initial_state': initial, 'cases': cases,
        'asset_files_sha256': deepcopy(manifest['asset_files_sha256']),
        'native_sdf_source_meshes': sources, 'native_sdf_subgrid_resolution': 0,
        'model_calls': 0, 'ordinary_agent_validation': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False, 'external_object_forces': False,
        'source_sha256': {str(Path(__file__).resolve()): digest(Path(__file__))},
        'probe_contract': {
            'scene_physical_fields_unchanged': True, 'task_thresholds_unchanged': True,
            'robot_phases': False, 'new_supports': False, 'geometry': geometry,
            'energy_reasoning': reasoning,
            'native_runner': 'scripts.validation.validate_assembly_contacts',
            'observer': 'scripts.validation.validate_assembly_contacts.Observer',
            'native_parameter_readback': True, 'native_step_recording_required': True,
            'stable_window_s': .5, 'upward_support_window_s': .02,
            'signed_native_normal_impulses_required': True,
            'zero_impulse_release_manifolds_require_separate_review': True,
        },
        'scope': 'Supplemental passive top-ring entry/guidance and beside-shaft '
                 'rejection only. Lower-ring diagnostic reset gaps are explicit; '
                 'neither canonical supply-to-assembly nor robot manipulation is qualified.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=RELEASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'input': str(args.output.resolve()), 'sha256': digest(args.output),
                      'native_runtime_started': False, 'cases': [c['id'] for c in value['cases']]}))


if __name__ == '__main__':
    main()
