"""Versioned scene candidates, separate from runnable bench task releases.

These pointers retain reset assets and native presentation evidence without
promoting planned task slots or granting robot manipulation qualification.
"""
from copy import deepcopy

from bench.paths import DATA_ROOT

_CANDIDATES = {'remote_operation': {'asset_version': 'v6',
                      'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v6/manifest.json',
                      'manifest_sha256': 'a12d35757e30c38e2371d0c4d132a7544d422b34b0f9f9f11ae6f0d2032920dd',
                      'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v6/initial.usda',
                      'initial_usd_sha256': '50680f3bd2d0624fb820e70868a4c46072ee57f591c1c3b094a6b65cc14ccd15',
                      'reference_prim_path': '/World/LivingTask',
                      'evidence_scope': 'Face-up remote reset with visible physical buttons and render-only '
                                        'TV countdown segments; validation is recorded separately.'},
 'water_pouring': {'asset_version': 'v2',
                   'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/water_pouring/v2/manifest.json',
                   'manifest_sha256': '4855c3cdaaebbbf5d4e8a6c20e185201d93aa68eaf6f8400cd2fd1f293f22f71',
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
                    'manifest_sha256': 'dbf1f59dbfb648247ece0ebe821e7b8edf7aea91b933a9971006f59a08f721f8',
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
                      'manifest_sha256': 'bde8223a805d5f6d822fb3da1b6a22726e43694e0d489957cb76ace9ee1c38d8',
                      'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/tower_extraction/v3/initial.usda',
                      'initial_usd_sha256': 'a88592c07cfe5a89fb31bb6d7c7b5ee5dccf30a21066f29bcfcb49bc8c5fb273',
                      'reference_prim_path': '/World/LivingTask',
                      'native_evidence_directory': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/render_extraction01',
                      'media_directory': f'{DATA_ROOT}/demo/interface/scenes/living_trays_towers_20261019/tower_extraction',
                      'evidence_scope': '12-layer tabletop tower with 36 blocks, 10 mm gaps and two physical '
                                        'collection trays. Native settling/rendering are separate from robot '
                                        'manipulation qualification.'}}


_CANDIDATES['storage_box'] = {
    'asset_version': 'v2',
    'manifest_path': f'{DATA_ROOT}/assets/task_parts/living_room/storage_box/v2/manifest.json',
    'manifest_sha256': 'b7a4c13e0808831e6e1d64e4a3fb35cc2d34c00ca91e2983b0fd7e6cf90d6907',
    'initial_usd': f'{DATA_ROOT}/assets/task_parts/living_room/storage_box/v2/initial.usda',
    'initial_usd_sha256': '97b2df8adecae6c910ae192a756e6ffa7cbda32988353511644aed07e47f8823',
    'reference_prim_path': '/World/LivingTask',
    'known_limitations': ['Passive rigid-body hinge/catch engineering model; no calibrated latch-force or material claims.'],
    'evidence_scope': 'Physical rotary catches obstruct a hinged lid; native object interaction evidence is separate from robot grasp strategy.',
}


def scene_candidate(task_id):
    result = deepcopy(_CANDIDATES[task_id])
    result.update(schema='living_room_scene_candidate_reference.v1',
        authoring_module=f'scenes.workrooms.living_room.tasks.{task_id}.assets',
        composition_module='scenes.workrooms.living_room.task_scene',
        task_object_dynamics=True, robot_presentation='original_static_ready_geometry',
        runnable_bench_release=False, native_robot_manipulation_qualified=False)
    return result
