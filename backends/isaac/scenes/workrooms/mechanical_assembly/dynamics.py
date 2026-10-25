"""Compose the current workshop's diagnostic dynamics, separate from its tasks."""
from copy import deepcopy
import hashlib
from pathlib import Path

from bench.isaacsim.release import task_inputs
from bench.paths import load_json
from scenes.workrooms.registry import BASE_RELEASE, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.mechanical_assembly.scene import environment_reference
from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration, ready_configuration, stand_geometry
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from bench.simulation.isaacsim_backend.workroom import SUPPORT_PATHS, checked

STAND_MANIFEST_SHA256 = '640a7aed72a2fbe8aad779f7aab454880b72b1da2f4e5b417a252b54af6c168f'
THREADED_MANIFEST_SHA256 = 'cd0d610f21ff5de2ff140ae77289200d8a615db27d67eaf017344272c901bb1f'
SHARPA_MANIFEST_SHA256 = 'faf25a316190e3369ad57a50d06243fbc52123c988166bae14cbf17f3062d876'
SDF1024_PARTITION_MANIFEST_SHA256 = 'b81a9ea7e4003c72f8bb0f71ef607cb10235e68fb92df7732c42c98a42fa7850'


def prepare_stand(output):
    from bench.simulation.assets.scene_parts import Asset, write_library
    asset = Asset('mechanical_robot_stand', dynamic=False, mass_kg=0)
    for box in stand_geometry()['boxes']:
        asset.add_part({'geometry': 'box', 'size_m': box['size_m'], 'color': 'steel'},
                       label=box['name'], offset=box['position_m'])
    return write_library(output, [('mechanical_robot_stand', asset)],
                         generator='scenes.workrooms.mechanical_assembly.dynamics.prepare_stand',
                         extra={'stand_layout': stand_geometry()})


def diagnostic_scene(stand_root, threaded_root=None):
    """Real falling bodies, original room and setback robot; no task score."""
    checked(Path(stand_root)/'manifest.json', STAND_MANIFEST_SHA256)
    if threaded_root is not None:
        checked(Path(threaded_root)/'manifest.json', THREADED_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    cube = deepcopy(scene['objects'][0])
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        scene.pop(key, None)
    reference = environment_reference()
    description = load_json(Path(reference['dependency_manifest_path']))
    scene.update(robot=robot_configuration(), objects=[], fixtures=[], goal_markers=[],
                 cloth={'enabled': False}, workbench={'enabled': False},
                 workcell_details={'enabled': False}, workroom=reference,
                 cameras=camera_specs(description),
                 description='Mechanical workshop native engineering preflight; no task success claim.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['mechanical_diagnostic_provenance'] = {
        'stand_manifest_sha256': STAND_MANIFEST_SHA256,
        'threaded_manifest_sha256': THREADED_MANIFEST_SHA256 if threaded_root else None,
        'task_success_claimed': False, 'physics_accepted_at_preparation': False}
    initial, ready_design = ready_configuration(scene['robot'])
    scene['fixtures'].append({'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
                              'asset_root': str(stand_root), 'position_m': [0, 0, 0]})
    specs = {}
    asset = load_task_asset(cube['asset_id'], cube['asset_root'], verify_hashes=True)
    half = [v/2 for v in asset['dimensions_m']]
    for name, xy, support in [('cube_table', [.30, -.35], 'table'), ('cube_floor', [.30, -.70], 'floor')]:
        top = .78 if support == 'table' else 0.
        scene['objects'].append({**deepcopy(cube), 'id': name,
                                 'position_m': [*xy, top+half[2]+.22]})
        specs[name] = {'bounds_min_m': [-v for v in half], 'bounds_max_m': half,
                       'support_top_z_m': top, 'expected_support_path': SUPPORT_PATHS[support],
                       'release_height_m': .22}
    if threaded_root is not None:
        manifest = load_json(Path(threaded_root)/'manifest.json')
        audit_path = Path(manifest['source_provenance'])
        if hashlib.sha256(audit_path.read_bytes()).hexdigest() != manifest['source_provenance_sha256']:
            raise ValueError('Threaded asset audit changed')
        options = load_json(audit_path)['recommended_spec_options']
        for name, asset_id, xy in [
            ('screwdriver', 'mechanical_screwdriver_v1', [.39, .31]),
            ('bolt', 'mechanical_bolt_m16', [.56, -.16])]:
            part = load_task_asset(asset_id, threaded_root, verify_hashes=True)
            bounds = [part['bounds_min_m'], part['bounds_max_m']]
            # The driver lies flat; the bolt rests on its broad head face.
            # The first native flat-bolt candidate rolled after settling. This
            # reset pose changes neither source shape nor material/solver settings.
            quaternion = [2**-.5, 2**-.5, 0, 0] if name == 'screwdriver' else [0, 1, 0, 0]
            bottom = bounds[0][1] if name == 'screwdriver' else -bounds[1][2]
            scene['objects'].append({'id': name, 'asset_id': asset_id,
                'asset_root': str(threaded_root), 'position_m': [*xy, .78-bottom+.03],
                'quaternion_wxyz': quaternion, **deepcopy(options)})
            specs[name] = {'bounds_min_m': bounds[0], 'bounds_max_m': bounds[1],
                'support_top_z_m': .78, 'expected_support_path': SUPPORT_PATHS['table'],
                'release_height_m': .03}
        scene['fixtures'].append({'id': 'threaded_fixture',
            'asset_id': 'mechanical_threaded_fixture_m16', 'asset_root': str(threaded_root),
            'position_m': [.61, .21, .78], **deepcopy(options)})
    return scene, initial, specs, ready_design


def diagnostic_scene_v3(stand_root, threaded_root):
    """Compose the selected v3 assets or its source-preserving SDF partition."""
    manifest_path = Path(threaded_root) / 'manifest.json'
    threaded_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if threaded_manifest_sha256 not in (SHARPA_MANIFEST_SHA256,
                                        SDF1024_PARTITION_MANIFEST_SHA256):
        raise ValueError('Unsupported threaded asset candidate: '+str(threaded_root))
    scene, initial, specs, ready_design = diagnostic_scene(stand_root)
    manifest = load_json(manifest_path)
    # The published v3 catalogue's dependency hashes bind both the exact source
    # USDs and the accepted engineering test/profile, without importing its code.
    roles = {'screwdriver': 'mechanical_screwdriver_sharpa_v3',
             'bolt': 'mechanical_bolt_sharpa_v3',
             'fixture': 'mechanical_threaded_fixture_sharpa_v3'}
    for role, xy in [('screwdriver', [.39, .31]), ('bolt', [.56, -.16])]:
        asset = load_task_asset(roles[role], threaded_root, verify_hashes=True)
        bottom = asset['bounds_min_m'][1] if role == 'screwdriver' else -asset['bounds_max_m'][2]
        scene['objects'].append({'id': role, 'asset_id': roles[role],
            'asset_root': str(threaded_root), 'position_m': [*xy, .78-bottom+.03],
            'quaternion_wxyz': [2**-.5, 2**-.5, 0, 0] if role == 'screwdriver' else [0, 1, 0, 0],
            **deepcopy(manifest['per_role_spec_options'][role])})
        specs[role] = {'bounds_min_m': asset['bounds_min_m'], 'bounds_max_m': asset['bounds_max_m'],
            'support_top_z_m': .78, 'expected_support_path': SUPPORT_PATHS['table'], 'release_height_m': .03}
    scene['fixtures'].append({'id': 'threaded_fixture', 'asset_id': roles['fixture'],
        'asset_root': str(threaded_root), 'position_m': [.61, .21, .78],
        **deepcopy(manifest['per_role_spec_options']['fixture'])})
    profile = ('threaded_sharpa_v3' if threaded_manifest_sha256 == SHARPA_MANIFEST_SHA256
               else 'threaded_sharpa_v3_sdf1024_convex_partition_candidate02')
    scene['mechanical_diagnostic_provenance'].update(asset_profile=profile,
        threaded_manifest_sha256=threaded_manifest_sha256,
        qualification_scope='Prior acceptance covers external-wrench contact tests in another scene; this robot scene requires independent native checks.',
        source_preserving_collision_candidate=(profile != 'threaded_sharpa_v3'))
    return scene, initial, specs, ready_design


def preinserted_scene_v3(stand_root, threaded_root):
    """Task reset candidate; gravity and robot interactions still need testing."""
    from .threaded_reset import released_pose_seed
    scene, initial, specs, ready_design = diagnostic_scene_v3(stand_root, threaded_root)
    fixture_position = [.38, 0., .78]
    poses, seed = released_pose_seed(fixture_position)
    for obj in scene['objects']:
        if obj['id'] in poses:
            obj.update(poses[obj['id']])
    next(x for x in scene['fixtures'] if x['id'] == 'threaded_fixture')['position_m'] = fixture_position
    scene['mechanical_diagnostic_provenance']['threaded_reset_seed'] = seed
    return scene, initial, {k: v for k, v in specs.items() if k.startswith('cube_')}, ready_design
