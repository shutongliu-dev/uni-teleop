"""A fixed machining workshop with an empty shared assembly bench.

Machines and wall tools are static scenery. The robot, its floor stand and all
manipulated parts/tools belong to tasks, never to this environment asset.
"""
from copy import deepcopy
from pathlib import Path

from pxr import UsdGeom

from .builder import Builder
from .common import build_room, build_workcell, apply_lighting_profile
from .interiors import build_interior
from .layouts import ROOM_SPECS
from .mechanical_furnishings import build_furnishings

VARIANT = 'mechanical_workshop_20261017_v3'
FAMILY = 'mechanical_assembly'
TABLE = {'center_xy': [.65, 0.], 'size_m': [1., 1.], 'top_z': .78,
         'collider_path': '/World/Geometry/worktop/Collider'}
RESERVED_TASK_VOLUME = {'low': [-.60, -.70, .78001], 'high': [1.30, .70, 1.90]}
ROBOT_FLOOR_RESERVE = {'low': [-.60, -.70, 0.], 'high': [.15, .70, 1.90]}
CAMERAS = {
    'room': {'eye': [.35, -1.40, 2.15], 'target': [.50, .45, 1.05], 'focal_length_mm': 19.},
    'workcell': {'eye': [1.80, -1.26, 1.88], 'target': [.15, .05, .94], 'focal_length_mm': 26.},
    'tabletop': {'eye': [.65, -.78, 1.95], 'target': [.57, 0., .78], 'focal_length_mm': 30.},
}


def build_mechanical_workshop(stage, assets_root, output_root, *, lighting_profile='nominal'):
    b = Builder(stage, Path(assets_root), Path(output_root))
    b.family = FAMILY
    UsdGeom.Xform.Define(stage, '/World/Robot')
    build_room(b, FAMILY)
    build_workcell(b, FAMILY, include_robot_station=False)
    build_interior(b, FAMILY)
    # A low-gloss solid worktop keeps small metal parts readable. The original
    # metre-square support collider is unchanged.
    b.material('machining_bench_finish', (.43, .46, .43), .72, .12)
    b.bind(stage.GetPrimAtPath('/World/Geometry/worktop/Visual'), 'machining_bench_finish')
    furnishings = build_furnishings(b)
    # Pin the attribution records alongside the referenced visual resources.
    for source in list(b.dependencies):
        parent = Path(source).parent
        for filename in ('LICENSE.txt', 'SOURCE.json'):
            if (parent / filename).is_file():
                b.dependency(parent / filename)
    b.dependency(Path(assets_root) / 'objects/simready/v1/SOURCE.json')
    for filename in ('NotoSansCJKsc-Regular.otf', 'SOURCE.json', 'LICENSE.txt'):
        b.dependency(Path(assets_root) / 'fonts/noto_cjk_sc/v1' / filename)
    lighting = apply_lighting_profile(stage, FAMILY, lighting_profile)
    return {
        'schema': 'scence.mechanical_workshop_environment.v1', 'family': FAMILY,
        'variant': VARIANT, 'task_independent': True, 'cameras': deepcopy(CAMERAS),
        'robot_placeholder': {'path': '/World/Robot', 'contains_robot': False},
        'layout': {'table_center_xy': TABLE['center_xy'], 'table_size_m': TABLE['size_m'],
                   'table_top_z': TABLE['top_z'], 'room': deepcopy(ROOM_SPECS[FAMILY]),
                   'reserved_task_volume': deepcopy(RESERVED_TASK_VOLUME),
                   'robot_floor_reserve': deepcopy(ROBOT_FLOOR_RESERVE)},
        'support_colliders': {'table': [TABLE['collider_path']],
                              'floor': ['/World/Geometry/room_floor/Collider']},
        'lighting': lighting, 'objects': [], 'geometry': b.records,
        'dependencies': b.dependencies, 'furnishings': furnishings,
        'ownership': {'environment': 'architecture, static machines, furniture, empty table and lighting',
                      'task': 'single robot, stand, movable tools, parts, fixtures, resets and evaluator'},
        'scope': 'Static environment only; machinery is scenery, no machining or manipulation qualification',
    }
