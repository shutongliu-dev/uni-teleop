"""Metric design inputs for four new card-room scene candidates.

These are design priors, not executable benchmark registrations or qualification
results. The parent room is unchanged. Pair identities belong only to a private
reset map; ``design_for`` never supplies them as public game information.
"""
from __future__ import annotations

from copy import deepcopy
import random

TASK_IDS = ('polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling')
PLAYERS = ('black', 'white')
TABLE_CENTER_XY_M = (.65, 0.)
BOARD_TOP_Z_M = .806
DECK_SIZE_XY_M = (.96, .96)
RESET_CLEARANCE_M = .0006
WOOD_DENSITY_KG_M3 = 650.
CONVEX_PRISM_SEGMENTS = 32
MEMORY_RESET_SEED = 20261028
POLYOMINO_SHAPES = {
    'mono': ((0, 0),),
    'domino': ((0, 0), (1, 0)),
    'tri_line': ((0, 0), (1, 0), (2, 0)),
    'tri_L': ((0, 0), (0, 1), (1, 0)),
    'tet_square': ((0, 0), (1, 0), (0, 1), (1, 1)),
    'tet_L': ((0, 0), (0, 1), (0, 2), (1, 0)),
}


def memory_pairs_for_seed(seed: int) -> dict[str, str]:
    """Private reset assignment; callers must never publish unrevealed pairs."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Memory reset seed must be an integer')
    values = [f'pair_{i}' for i in range(6) for _ in range(2)]
    random.Random(seed).shuffle(values)
    return {f'card_{i:02d}': value for i, value in enumerate(values)}


def polyomino_boxes(cells, pitch_m=.052, thickness_m=.022, edge_clearance_m=.001):
    """Non-overlapping exact boxes for a connected cell union.

    Only exterior faces are inset. Neighbouring cells meet without gaps, so the
    rigid shape is connected and its concavities remain physically open.
    """
    cells = {tuple(cell) for cell in cells}
    cx = (min(c[0] for c in cells) + max(c[0] for c in cells)) * pitch_m / 2
    cy = (min(c[1] for c in cells) + max(c[1] for c in cells)) * pitch_m / 2
    boxes = []
    for x, y in sorted(cells):
        low_x = (x-.5)*pitch_m + (edge_clearance_m if (x-1, y) not in cells else 0.)
        high_x = (x+.5)*pitch_m - (edge_clearance_m if (x+1, y) not in cells else 0.)
        low_y = (y-.5)*pitch_m + (edge_clearance_m if (x, y-1) not in cells else 0.)
        high_y = (y+.5)*pitch_m - (edge_clearance_m if (x, y+1) not in cells else 0.)
        boxes.append({'position_m': [(low_x+high_x)/2-cx, (low_y+high_y)/2-cy, 0.],
                      'dimensions_m': [high_x-low_x, high_y-low_y, thickness_m]})
    return boxes


def _piece(identity, owner, kind, position, dimensions, **extra):
    return {'id': identity, 'owner': owner, 'kind': kind,
            'initial_position_m': position, 'initial_orientation_wxyz': [1., 0., 0., 0.],
            'dimensions_m': dimensions, **extra}


def design_for(task_id: str) -> dict:
    """Return independent metric inputs; no simulator, files or hidden outcomes."""
    if task_id not in TASK_IDS:
        raise ValueError(f'Unknown card game design: {task_id}')
    board = {'center_xy_m': list(TABLE_CENTER_XY_M), 'top_z_m': BOARD_TOP_Z_M,
             'deck_size_xy_m': list(DECK_SIZE_XY_M), 'deck_thickness_m': .026,
             'deck_bounds_xy_m': [[.17, -.48], [1.13, .48]]}
    result = {'schema': 'card_game_design.v1', 'task_id': task_id,
              'players': list(PLAYERS), 'board': board, 'pieces': [],
              'density_kg_m3': WOOD_DENSITY_KG_M3,
              'reset_clearance_m': RESET_CLEARANCE_M,
              'scope': 'Scene design candidate; no native or robot-manipulation qualification.'}
    if task_id == 'polyomino_territory':
        result['title'] = '拼板占地'
        board.update(rows=6, columns=6, pitch_m=.052, size_xy_m=[.344, .344])
        result['shapes'] = {name: [list(c) for c in cells] for name, cells in POLYOMINO_SHAPES.items()}
        result['piece_thickness_m'] = .022
        result['edge_clearance_m'] = .001
        for owner, sign in (('black', -1), ('white', 1)):
            for i, (name, cells) in enumerate(POLYOMINO_SHAPES.items()):
                boxes = polyomino_boxes(cells)
                size = [(max(c[k] for c in cells)-min(c[k] for c in cells)+1)*.052-.002
                        for k in (0, 1)] + [.022]
                pos = [.65+(i%3-1)*.18, sign*(.27+(i//3)*.12),
                       BOARD_TOP_Z_M+.011+RESET_CLEARANCE_M]
                mass = WOOD_DENSITY_KG_M3*sum(b['dimensions_m'][0]*b['dimensions_m'][1]*b['dimensions_m'][2] for b in boxes)
                result['pieces'].append(_piece(f'{owner}_{name}', owner, 'polyomino', pos, size,
                    cells=[list(c) for c in cells], shape=name, collision_boxes=boxes, mass_kg=mass))
    elif task_id == 'barrier_race':
        result['title'] = '路障竞速'
        board.update(rows=5, columns=5, pitch_m=.075, size_xy_m=[.407, .407])
        result['barrier_geometry'] = {
            'length_m': .142, 'base_width_m': .025, 'base_height_m': .010,
            'blade_width_m': .008, 'blade_height_m': .045,
            'anchor_qr_range': [0, 3],
            'horizontal_axis_world': 'x', 'vertical_axis_world': 'y',
            'horizontal_blocks': 'Edges (q,r)-(q,r+1) and (q+1,r)-(q+1,r+1)',
            'vertical_blocks': 'Edges (q,r)-(q+1,r) and (q,r+1)-(q+1,r+1)',
        }
        for owner, sign in (('black', -1), ('white', 1)):
            result['pieces'].append(_piece(f'{owner}_pawn', owner, 'pawn',
                [.65, sign*.15, BOARD_TOP_Z_M+.0425+RESET_CLEARANCE_M], [.038, .038, .085],
                cylinders=[{'radius_m': .019, 'height_m': .018, 'position_m': [0., 0., -.0335]},
                           {'radius_m': .014, 'height_m': .067, 'position_m': [0., 0., .009]}]))
            for i in range(4):
                result['pieces'].append(_piece(f'{owner}_barrier_{i}', owner, 'barrier',
                    [.65+(-.15 if i%2 == 0 else .15), sign*(.28+(i//2)*.13),
                     BOARD_TOP_Z_M+.0275+RESET_CLEARANCE_M], [.142, .025, .055],
                    collision_boxes=[{'position_m': [0., 0., -.0225], 'dimensions_m': [.142, .025, .010]},
                                     {'position_m': [0., 0., .005], 'dimensions_m': [.142, .008, .045]}]))
    elif task_id == 'memory_pairs':
        result['title'] = '记忆翻块'
        board.update(rows=3, columns=4, pitch_xy_m=[.112, .14], size_xy_m=[.492, .45])
        result.update(pair_count=6, piece_thickness_m=.020,
                      card_front_normal_local=[0., 0., -1.], card_back_normal_local=[0., 0., 1.],
                      pair_identity_visibility='Private reset identity; reveal only after an actual face-up observation.')
        for i in range(12):
            pos = [.65+(i%4-1.5)*.112, (i//4-1)*.14, BOARD_TOP_Z_M+.010+RESET_CLEARANCE_M]
            result['pieces'].append(_piece(f'card_{i:02d}', None, 'memory_card', pos, [.064, .080, .020],
                mass_kg=.064*.080*.020*WOOD_DENSITY_KG_M3))
        result['collection_zones'] = {
            'black': {'center_xy_m': [.65, -.355], 'size_xy_m': [.46, .12]},
            'white': {'center_xy_m': [.65, .355], 'size_xy_m': [.46, .12]},
        }
    else:
        result['title'] = '推筹得分'
        board.update(size_xy_m=[.88, .88], target_center_xy_m=[.65, 0.],
                     scoring_radii_m=[.06, .12, .18], scoring_points=[3, 2, 1],
                     launch_centers_xy_m={'black': [.65, -.34], 'white': [.65, .34]},
                     launch_lines_y_m={'black': -.24, 'white': .24},
                     launch_region_bounds_xy_m={
                         'black': [[.44, -.40], [.86, -.26]],
                         'white': [[.44, .26], [.86, .40]]})
        result.update(disk_radius_m=.033, disk_height_m=.018, convex_segments=CONVEX_PRISM_SEGMENTS)
        for owner in PLAYERS:
            for i in range(4):
                sign = -1 if owner == 'black' else 1
                result['pieces'].append(_piece(f'{owner}_disk_{i}', owner, 'disk',
                    [.65+(i-1.5)*.09, sign*.34, BOARD_TOP_Z_M+.009+RESET_CLEARANCE_M],
                    [.066, .066, .018], radius_m=.033, height_m=.018))
    return deepcopy(result)
