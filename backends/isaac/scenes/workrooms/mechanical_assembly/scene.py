"""One fixed empty-table machining room for all five single-arm task plans."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/mechanical_workshop_20261017_v3')
ENVIRONMENT_USD_SHA256 = '1ad348bb7cd57747a2d84f068a718f2ab794c81e193a2015ff09b73bd35a1ea6'
ENVIRONMENT_DESCRIPTION_SHA256 = 'e9b7e1aaae9a5b6d4b551fccc466e724307edbb7a7e3647cff11f79769f9a6ed'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'mechanical_assembly', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'mechanical_assembly',
    'title': '机械加工／装配室',
    'source': 'Pinned task-independent machining workshop; the old six-room diagnostic release remains unchanged.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.65, 0.], 'table_size_xy_m': [1., 1.],
                    'table_top_z_m': .78},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
