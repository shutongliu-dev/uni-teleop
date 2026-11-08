"""Fixed residential laundry room; tasks independently own their dynamic towels."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/laundry_residential_20261021_v3')
ENVIRONMENT_USD_SHA256 = '6fda362f5943b408dc91ed0f3bd5c16576c5e5516bcf5634c25e56ed94de270e'
ENVIRONMENT_DESCRIPTION_SHA256 = '2afb955f689aea02f6704d627bee7a211fcb4f5f82d9d78553205cefd6adb39c'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'laundry_room', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'laundry_room',
    'title': '洗衣整理间 · 洗烘、晾晒与折衣',
    'source': 'Pinned residential laundry room with furnished washing, drying, storage and folding zones; task-owned dynamic towels are composed independently.',
    'status': 'task_family',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'room_x_range_m': [-2.6, 2.8], 'room_y_range_m': [-2.3, 2.2],
                    'room_height_m': 2.75,
                    'table_center_xy_m': [.65, -.55], 'table_size_xy_m': [1.8, .9],
                    'table_top_z_m': .90, 'primary_surface': 'folding_table',
                    'clear_task_rectangle_xy_m': [-.10, 1.40, -.94, -.16],
                    'work_surfaces': {
                        'folding_table': {'center_xy_m': [.65, -.55],
                                          'size_xy_m': [1.8, .9], 'top_z_m': .90}}},
    'ownership': 'Fixed environment and static dressing only; tasks own dynamic towels, initial state and scoring. Engineering cloth control is separate from ordinary robot episodes and robot manipulation qualification.',
}
