"""Pinned kitchen with separate cooking and bar surfaces; task-owned layers bind separately frozen releases."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/kitchen_galley_20261019_v2')
ENVIRONMENT_USD_SHA256 = '4207260f1efa8490f542cc13c7ca9581ebc1ed5907be3ee8c21c47ee6009463e'
ENVIRONMENT_DESCRIPTION_SHA256 = '241c10a87683f78c643fab029e6cb3fdd848149cc43f6b0e39336e661723ea03'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'kitchen', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'kitchen',
    'title': '厨房 · 灶台与吧台',
    'source': 'Pinned daylit galley kitchen; legacy diagnostic collection remains immutable.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.45, -.95], 'table_size_xy_m': [2.8, .8],
                    'table_top_z_m': .90, 'primary_surface': 'bar',
                    'work_surfaces': {
                        'bar': {'center_xy_m': [.45, -.95], 'size_xy_m': [2.8, .8], 'top_z_m': .90},
                        'cooking': {'center_xy_m': [.65, .94], 'size_xy_m': [.66, .59], 'top_z_m': .90},
                        'preparation': {'x_range_m': [-.5025, .32], 'y_range_m': [.57, 1.31], 'top_z_m': .90}},
                    'robot_aisle_width_m': 1.12},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
