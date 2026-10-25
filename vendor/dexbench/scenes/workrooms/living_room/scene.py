"""Living room: fixed environment identity and shared task coordinates."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/living_room_residential_20261018_v7')
ENVIRONMENT_USD_SHA256 = 'a86bd5a2a0fcd484e68ac54713e2ef0ab963eb4a793f1b10f7dded39559f91be'
ENVIRONMENT_DESCRIPTION_SHA256 = '77150ed79ea34d3cfdf48ee5c46a0c658985817d36d82fed27f988ecc3c2bcdd'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'living_room', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'living_room',
    'title': 'Living room',
    'source': 'Pinned compact residential living room with a private balcony, housed lighting and empty task table; native task qualification remains separate.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.65, 0.], 'table_size_xy_m': [1., 1.],
                    'table_top_z_m': .78},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
