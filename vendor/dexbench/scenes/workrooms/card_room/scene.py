"""Card room with an empty shared task table: fixed environment identity and shared task coordinates."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/card_room_heritage_20261017_v3')
ENVIRONMENT_USD_SHA256 = '5b563cff61ff577e4f2253c7ca0b2b7514ccdccca3d051a0576b35116824354f'
ENVIRONMENT_DESCRIPTION_SHA256 = '44f28f78fa32d0d4ea8f009b7ef1b862a4f5be81ba85d6f0fd8edd8c7cb2f2fc'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'card_room', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'card_room',
    'title': 'Card room',
    'source': 'Current fixed card-room environment; the legacy six-room diagnostic collection remains immutable.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.65, 0.], 'table_size_xy_m': [1., 1.],
                    'table_top_z_m': .78},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
