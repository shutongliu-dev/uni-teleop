"""Frozen single-arm placement contract; no simulator or mutable source imports."""
from copy import deepcopy
import hashlib
import math

RULES_VERSION = 'card_room_placement_rules_v1'
TARGET_SELECTION_VERSION = 'sha256_ranked_cells_v1'
PIECE_IDS = ('black_00', 'black_01', 'black_02')
CANDIDATE_CELLS = tuple((q, r) for r in range(3) for q in range(3))


def thresholds():
    """Reuse tabletop settlement tolerances, with explicit canonical pose sampling."""
    return {
        'physics_dt_s': .0005, 'control_period_s': .02, 'dwell_s': .300,
        'maximum_linear_speed_m_s': .015, 'maximum_angular_speed_rad_s': .100,
        'maximum_tilt_rad': math.radians(5.), 'bottom_z_tolerance_m': .003,
        'edge_clearance_m': .001, 'minimum_support_impulse_ns': 1.e-8,
        'maximum_robot_contact_impulse_ns': 1.e-8, 'maximum_penetration_m': .001,
        'maximum_stable_position_range_m': .001,
        'maximum_stable_orientation_range_rad': .020,
        'initial_position_tolerance_m': .0001, 'initial_orientation_tolerance_rad': .002,
    }


def validate_target_cells(target_cells):
    if not isinstance(target_cells, (list, tuple)) or len(target_cells) != 3:
        raise ValueError('Exactly three target cells are required')
    cells = []
    for cell in target_cells:
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                or any(type(value) is not int or not 0 <= value < 3 for value in cell)):
            raise ValueError('Each target must be a 3x3 integer [q, r] cell')
        cells.append(tuple(cell))
    if len(set(cells)) != 3:
        raise ValueError('Target cells must be distinct')
    return [list(cell) for cell in cells]


def select_target_cells(seed):
    """Versioned deterministic selection without Python random-version coupling."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Episode seed must be an unsigned 32-bit integer')
    def rank(cell):
        message = f'{TARGET_SELECTION_VERSION}:task0:{seed}:{cell[0]}:{cell[1]}'
        return hashlib.sha256(message.encode('ascii')).digest(), cell
    return [list(cell) for cell in sorted(CANDIDATE_CELLS, key=rank)[:3]]


def public_rules(target_cells):
    cells = validate_target_cells(target_cells)
    return {
        'schema': RULES_VERSION,
        'status': 'engineering_candidates_not_native_accepted',
        'instruction_zh': '用单臂将三枚黑棋子分别放入本回合公布的三个目标格，每格一枚，'
            '松手后让三枚棋子同时稳定至少0.3秒。',
        'target_cells': cells,
        'target_selection': {'version': TARGET_SELECTION_VERSION,
            'candidate_cells': [list(cell) for cell in CANDIDATE_CELLS],
            'count': 3, 'replacement': False,
            'algorithm': 'Rank all cells by SHA256(version:task0:decimal_seed:q:r); take the first three, with cell tuple as collision tie-break.',
            'seed_range': [0, 2**32-1], 'scope': 'One published target set per episode; no change during execution.'},
        'success': {'piece_ids': list(PIECE_IDS), 'required_count': 3,
            'piece_to_target_assignment': 'Any bijection of the three physical pieces to the three targets.',
            'thresholds': thresholds(),
            'cell_acceptance': 'Full conservative tilted footprint inside one target cell with 1 mm edge clearance; exact centring is not required.',
            'support': 'Positive named-board native normal impulse at every substep throughout each credited interval.',
            'release': 'No robot contact impulse above threshold at any substep of the stable dwell.',
            'stability': 'All three distinct pieces must simultaneously retain their unique target cells and supported released stability for at least 300 ms.',
            'reset_credit': False},
        'recovery': {'wrong_cell_or_disturbed_piece': 'Correct it with robot joint actions; affected dwell resets without a permanent logical placement.',
            'duplicate_target_occupants': 'Neither occupant receives credit until the target contains one eligible piece.',
            'placement_intent_required': False, 'automatic_forfeit': False,
            'api_or_evidence_fault': 'Abort as unavailable evidence, never a competitive loss.',
            'budget_exhaustion': 'Incomplete episode, not success.'},
        'physical_actions': 'Robot joints only; no object pose writes, pose playback, grasp welds or automatic relocation.',
        'evidence_scope': 'Canonical object poses and velocities every 20 ms, plus complete consecutive 0.5 ms native contact samples; not continuous substep object-pose evidence.',
        'qualification': 'Prepared scoring rules; unit tests do not establish native placement feasibility or success.',
    }


def target_positions_m(geometry, cells):
    """Published nominal base-centre goals from frozen task geometry only."""
    c, s = math.cos(geometry['board_yaw_rad']), math.sin(geometry['board_yaw_rad'])
    cx, cy = geometry['center_xy_m']
    z = geometry['table_top_z_m']+geometry['board_thickness_m']+geometry['piece_base_height_m']/2
    positions = []
    for q, r in cells:
        x, y = geometry['pitch_m']*(q-1), geometry['pitch_m']*(r-1)
        positions.append([cx+c*x-s*y, cy+s*x+c*y, z])
    return positions


def bind_episode_targets(task, seed):
    """Bind only public episode targets; reset bodies, physics and thresholds stay frozen."""
    if task.get('schema_version') != 'card-room-placement-task-v1' or task.get('id') != 'task0':
        raise ValueError('Target binding requires the independently versioned task0')
    result = deepcopy(task)
    cells = select_target_cells(seed)
    result['episode_seed'] = seed
    result['evaluation']['target_cells'] = deepcopy(cells)
    result['public_rules'] = public_rules(cells)
    goal = result.setdefault('public_goal', {})
    goal.update(operation='place_three_pieces', piece_ids=list(PIECE_IDS),
        frame_id='world_m_z_up', target_cells=deepcopy(cells),
        candidate_cells=[list(cell) for cell in CANDIDATE_CELLS],
        target_positions_m=target_positions_m(result['evaluation']['geometry'], cells),
        description=result['public_rules']['instruction_zh'])
    return result
