"""Fixed executive study with an empty desk and rear book-classification shelves."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/study_library_20261019_v7')
ENVIRONMENT_USD_SHA256 = '3d8f0d5f9f9d9f6738ab943d5d8c81bcc6f41d1bc748e1ebade12c56d96d6219'
ENVIRONMENT_DESCRIPTION_SHA256 = 'cc96c0e64e5353f854f1ee059c52ca3a9692e05a925e851a689bba87f8925d36'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'study', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'study',
    'title': '书房',
    'source': 'Pinned executive study with four empty rear shelf bays; task qualification remains separate.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.65, 0.], 'table_size_xy_m': [.90, 2.00],
                    'table_top_z_m': .78},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
