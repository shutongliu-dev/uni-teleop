"""Versioned scene candidates, separate from runnable bench task releases.

These pointers retain reset assets and native presentation evidence without
promoting planned task slots or granting robot manipulation qualification.
"""
from copy import deepcopy

from bench.paths import DATA_ROOT

_CANDIDATES = {'remote_operation': {'asset_version': 'v5',
                      'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v5/manifest.json',
                      'manifest_sha256': '05dbd52e71293446128ba1e67dbae83e1205fc3ac803a46db5fcdb965726da77',
                      'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v5/initial.usda',
                      'initial_usd_sha256': '674410c0295a1ab8c8cdb7f90770af985d45337ee72918d7f2da6b96d66988d1',
                      'reference_prim_path': '/World/LivingTask',
                      'native_evidence_directory': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_task_scenes_20261018/task1_native05',
                      'media_directory': f'{DATA_ROOT}/demo/interface/scenes/living_task_scenes_20261018/task1_v5',
                      'evidence_scope': 'Face-up presentation settling; face-down reset is authored '
                                        'separately'},
 'water_pouring': {'asset_version': 'v2',
                   'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/water_pouring/v2/manifest.json',
                   'manifest_sha256': 'baa82eef7725e4e64edd4e17dd73a3373e3fc8a867429bf7b2c5dd45c2a364ba',
                   'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/water_pouring/v2/initial.usda',
                   'initial_usd_sha256': 'e7703b2565efe0063127dd986160f1657cac55524c046f2cea1a185f89d00371',
                   'reference_prim_path': '/World/LivingTask',
                   'native_evidence_directory': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/evidence01',
                   'media_directory': f'{DATA_ROOT}/demo/interface/scenes/living_water_coupling_20261018_v4',
                   'evidence_scope': 'Practical PBD approximation: real 250 g water, improved kettle load, '
                                     '93.75 mL fixture pour without spill; no robot strategy qualification.',
                   'known_limitations': [
                       '8 mm PBD resolution: 488 particles, approximately 0.5123 mL each; stream and rim interaction are coarse.',
                       'Full-kettle water load is 4.4% above theoretical; 93.75 mL in the cup gives 22.7% excess water reaction. This is a practical approximation, not quantitative force calibration.',
                       'Native pouring evidence uses a separate 75-degree torque fixture; full robot manipulation remains unverified.']},
 'tower_building': {'asset_version': 'v6',
                    'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/tower_building/v6/manifest.json',
                    'manifest_sha256': '98985b7a031f218a70b5e93be776a4f602228334e4d84e5bdcd3f0ed3e5f013b',
                    'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/tower_building/v6/initial.usda',
                    'initial_usd_sha256': '457b96a5787d41a5353b667c986e5b7e30092e5ea621ac3f09af5b79f03c102c',
                    'reference_prim_path': '/World/LivingTask',
                    'native_evidence_directory': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/render_building01',
                    'media_directory': f'{DATA_ROOT}/demo/interface/scenes/living_trays_towers_20261019/tower_building',
                    'evidence_scope': '36 physical wooden blocks in two supply trays; empty tabletop for '
                                      'unrestricted tower construction. Native settling/rendering are '
                                      'separate from robot manipulation qualification.'},
 'tower_extraction': {'asset_version': 'v3',
                      'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/tower_extraction/v3/manifest.json',
                      'manifest_sha256': '50de959c11f307f73644b16a09aa82aab13da9111cd06d1258551ae4628e22b0',
                      'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/tower_extraction/v3/initial.usda',
                      'initial_usd_sha256': 'a88592c07cfe5a89fb31bb6d7c7b5ee5dccf30a21066f29bcfcb49bc8c5fb273',
                      'reference_prim_path': '/World/LivingTask',
                      'native_evidence_directory': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/render_extraction01',
                      'media_directory': f'{DATA_ROOT}/demo/interface/scenes/living_trays_towers_20261019/tower_extraction',
                      'evidence_scope': '12-layer tabletop tower with 36 blocks, 10 mm gaps and two physical '
                                        'collection trays. Native settling/rendering are separate from robot '
                                        'manipulation qualification.'}}


def scene_candidate(task_id):
    result = deepcopy(_CANDIDATES[task_id])
    result.update(schema='living_room_scene_candidate_reference.v1',
        authoring_module=f'scenes.workrooms.living_room.tasks.{task_id}.assets',
        composition_module='scenes.workrooms.living_room.task_scene',
        task_object_dynamics=True, robot_presentation='original_static_ready_geometry',
        runnable_bench_release=False, native_robot_manipulation_qualified=False)
    return result
