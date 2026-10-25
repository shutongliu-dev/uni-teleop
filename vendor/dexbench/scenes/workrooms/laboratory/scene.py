"""Pinned furnished laboratory with an island and separate perimeter work zones."""
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.workroom import SCHEMA

ENVIRONMENT_ROOT = Path(f'{DATA_ROOT}/assets/scenes/scence_workrooms/laboratory_island_20261020_v3')
ENVIRONMENT_USD_SHA256 = 'e96f3f1a575e64408bd20f75e3ff96e4b86209f850930c06065324ec9b9f42db'
ENVIRONMENT_DESCRIPTION_SHA256 = '12767a6ef7d0db2f3e08828fb91e874002e238cbef2a3ec6688764baf1cfd5dd'


def environment_reference():
    return {'schema': SCHEMA, 'family': 'laboratory', 'version': ENVIRONMENT_ROOT.name,
            'usd_path': str(ENVIRONMENT_ROOT / 'scene.usda'),
            'usd_sha256': ENVIRONMENT_USD_SHA256, 'source_prim': '/World',
            'dependency_manifest_path': str(ENVIRONMENT_ROOT / 'scene_description.json'),
            'dependency_manifest_sha256': ENVIRONMENT_DESCRIPTION_SHA256}


SCENE = {
    'schema': 'workroom_scene_definition.v1', 'family': 'laboratory',
    'title': '实验室 · 中央岛台与样本制备',
    'source': 'Pinned furnished laboratory; legacy rooms_20261017_v3 diagnostic collection remains immutable.',
    'coordinates': {'units': 'metres', 'up_axis': 'Z', 'frame': 'world',
                    'table_center_xy_m': [.65, 0.], 'table_size_xy_m': [2.4, 1.25],
                    'table_top_z_m': .90, 'primary_surface': 'island',
                    'island_yaw_degrees': 180.,
                    'clear_task_rectangle_xy_m': [-.48, 1.78, -.30, .56],
                    'rear_clear_task_rectangle_xy_m': [.40, 1.04, 1.99, 2.20],
                    'robot_aisle_width_m': 1.325,
                    'work_surfaces': {
                        'island': {'center_xy_m': [.65, 0.], 'size_xy_m': [2.4, 1.25], 'top_z_m': .90},
                        'instrument_counter': {'x_range_m': [-.78, 2.02],
                                               'y_range_m': [1.99, 2.57], 'top_z_m': .90}}},
    'robot_installation': {
        'layout_id': 'laboratory_two_bench_single_arm_v1',
        'module': 'scenes.workrooms.laboratory.robot_layout',
        'base_position_m': [.70, 1.2875, .78], 'base_quaternion_wxyz': [1., 0., 0., 0.],
        'static_preview': {
            'manifest_path': f'{DATA_ROOT}/datasets/scence/development/laboratory_robot_20261020/preview01/manifest.json',
            'manifest_sha256': '3072f438cc5fb61aad27116669c3a1efc567b6dcb8fead62f989b90046dc93c2',
            'usd_path': f'{DATA_ROOT}/datasets/scence/development/laboratory_robot_20261020/preview01/scene.usda',
            'usd_sha256': 'a7cfe5d46508b7cdd70a40411520e7e4bff468ff32b4a0944b19623f332b8a58'},
        'scope': 'Original robot placement and discrete endpoint checks; five independent task releases share this installation. Hand-object manipulation is qualified separately.'},
    'ownership': 'Fixed environment only; tasks own movable objects, robots, initial state and scoring.',
}
