"""Pure rules for three levels; physical callers use BoardGameMatch."""
from __future__ import annotations

from collections import deque
import hashlib
import json
from types import MappingProxyType

from .geometry import BoardGameGeometry, COLORS, Cell, NEIGHBOR_OFFSETS, geometry_for


class IllegalMove(ValueError):
    pass


def contract_sha256(contract: dict) -> str:
    """Portable identity of the public JSON value, independent of indentation."""
    payload = json.dumps(contract, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def public_rule_contract(geometry: BoardGameGeometry, thresholds: dict) -> dict:
    """Single machine-readable contract used by releases and live player views.

    The caller supplies the actual evaluator thresholds, not copied constants.
    No private state, object pose or contact value enters this specification.
    """
    return {
        'schema': 'robot_board_game_rules_contract.v2',
        'revision': 'recoverable_physical_moves_v2',
        'task_id': geometry.game_id,
        'rules': BoardGameRules(geometry).specification(),
        'geometry': geometry.specification(),
        'ownership': {'colors_by_player_turn_order': ['black', 'white'],
                      'robot_binding': 'Frozen scene player identities and published owned_actuators define the two disjoint robot bindings.',
                      'actuators': 'Only the authenticated current player may command its owned robot.'},
        'turn': {'advance': 'One verified physical placement advances one turn.',
                 'failed_attempt_consumes_turn': False,
                 'placement_intent': 'Select an unused own piece and empty logical cell; intent never moves a body.',
                 'revision': 'A valid new intent replaces the pending intent in the same turn and restarts dwell.',
                 'duplicate_intent': 'An identical pending intent is idempotent and preserves dwell.',
                 'cancel': 'cancel_placement clears the pending intent without moving pieces or advancing the turn.'},
        'physical_commit': {
            'thresholds': dict(thresholds),
            'position_origin': 'piece_base_centre',
            'cell_acceptance': 'The full conservatively bounded tilted footprint must remain inside the intended cell with edge_clearance_m; exact centring is not required.',
            'stable_set': 'The pending piece and every committed piece must simultaneously pass the full position/support/release/velocity gates for dwell_s.',
            'support': 'Positive native normal impulse from the named board collider on each observed substep.',
            'release': 'No robot contact impulse above maximum_robot_contact_impulse_ns during the stable dwell.',
            'evidence': 'Complete consecutive native solver substeps for the entire inventory; no credit for unobserved time.',
            'penetration': 'Native reported contact separation is required; unavailable evidence is never zero penetration.'},
        'recovery': {
            'committed_piece_disturbed': 'Pause commit; restore every affected piece to its original logical cell. No automatic forfeit.',
            'unrequested_piece_on_board': 'Pause commit; remove the extra uncommitted piece from board support or select a legal own piece with a revised intent.',
            'touch_or_transient_motion': 'Allowed during attempts; resets stable dwell and never directly assigns a loss.',
            'repair_actor': 'The current player may use only its own robot to restore either color to already committed cells.',
            'recovery_changes_logical_board': False,
            'automatic_recovery_timeout_s': None,
            'automatic_forfeit': False},
        'errors_and_outcomes': {
            'invalid_action': 'Reject the action; preserve pending move, logical board and turn.',
            'missing_native_evidence': 'Pause commitment and restart dwell after a complete continuous stream resumes.',
            'api_or_simulator_fault': 'No game loss. Runner may end_episode(runtime_error) as aborted, excluded from game scores.',
            'time_or_step_budget': 'Runner may end_episode(time_limit|step_limit) as truncated/incomplete, without a winner.',
            'cancelled_run': 'Runner may end_episode(cancelled) as aborted, without a winner.',
            'win_or_draw': 'Only verified placement applies the frozen game rules; an accepted terminal result is immutable.'},
        'public_actions': {
            'placement_intent': ['type', 'piece_id', 'q', 'r'],
            'cancel_placement': ['type'],
            'joint_targets': ['type', 'robot_id', 'joint_positions', 'duration_s'],
            'maximum_joint_command_duration_s': 1.,
            'physical_actions': 'Owned robot joint targets only. No object pose writes, snapping, grasp welds or automatic recovery.'},
        'observations': {'default': 'Equal published synchronized RGB-D; static rules and coarse adjudication status are public.',
                         'private': 'Exact native poses, impulses, penetration values and logical occupancy remain private in rgbd mode.',
                         'privileged_mode': 'protocol_smoke_board_state exposes logical board/legal cells only with its explicit non-model-score label.'},
        'qualification': 'This is a benchmark rule definition, not evidence of a successful grasp, placement, game or model run.',
    }


class BoardGameRules:
    """Black moves first; no captures, passes, gravity, swaps or automatic moves.

    Level 2 is explicitly our planar 5x5 connect-four rule, not the commercial
    vertical dropping game. Lines may be horizontal, vertical or diagonal.
    Competitive comparisons must reverse agent/color assignments in paired games.
    """

    def __init__(self, geometry: BoardGameGeometry | None = None,
                 players: tuple[str, str] = ('player_0', 'player_1')):
        if len(players) != 2 or len(set(players)) != 2 or not all(isinstance(p, str) and p for p in players):
            raise ValueError('Exactly two distinct nonempty player IDs are required')
        self.geometry = geometry or geometry_for('tic_tac_toe_3x3')
        self.players = tuple(players)
        self._board: dict[Cell, str] = {}
        self.turn_index = 0
        self.winner: str | None = None
        self.winning_path: tuple[Cell, ...] = ()
        self.draw = False

    @property
    def terminal(self) -> bool:
        return self.winner is not None or self.draw

    @property
    def board(self):
        return MappingProxyType(self._board)

    @property
    def current_player(self) -> str | None:
        return None if self.terminal else self.players[self.turn_index % 2]

    def color(self, player_id: str) -> str:
        if player_id not in self.players:
            raise IllegalMove('Unknown player')
        return COLORS[self.players.index(player_id)]

    def legal_moves(self) -> tuple[Cell, ...]:
        return () if self.terminal else tuple((q, r) for r in range(self.geometry.size)
                                              for q in range(self.geometry.size) if (q, r) not in self._board)

    def validate_move(self, player_id: str, q: int, r: int):
        if self.terminal:
            raise IllegalMove('The game is already won or drawn')
        if player_id != self.current_player:
            raise IllegalMove('Only the current player may act')
        try:
            cell = self.geometry.validate_cell(q, r)
        except ValueError as exc:
            raise IllegalMove(str(exc)) from exc
        if cell in self._board:
            raise IllegalMove('Cell is occupied')

    def play(self, player_id: str, q: int, r: int):
        self.validate_move(player_id, q, r)
        color = self.color(player_id)
        self._board[q, r] = color
        self.turn_index += 1
        path = self.connection_path(color)
        if path:
            self.winner, self.winning_path = player_id, path
        elif len(self._board) == self.geometry.size**2:
            # A valid fully occupied Hex board always has a connection. A full
            # non-winning square board is an ordinary draw, not an invalid task.
            if self.geometry.shape == 'hex':
                raise RuntimeError('Full Hex board without a winner violates rule invariants')
            self.draw = True

    def connection_path(self, color: str) -> tuple[Cell, ...]:
        if color not in COLORS:
            raise ValueError('Unknown color')
        if self.geometry.shape == 'square':
            length = self.geometry.win_length
            for q, r in sorted(cell for cell, occupant in self._board.items() if occupant == color):
                for dq, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    line = tuple((q+i*dq, r+i*dr) for i in range(length))
                    if all(self._board.get(cell) == color for cell in line):
                        return line
            return ()
        axis = 1 if color == 'black' else 0
        starts = sorted(cell for cell, occupant in self._board.items()
                        if occupant == color and cell[axis] == 0)
        parents: dict[Cell, Cell | None] = {cell: None for cell in starts}
        queue = deque(starts)
        while queue:
            cell = queue.popleft()
            if cell[axis] == self.geometry.size-1:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = parents[cell]
                return tuple(reversed(path))
            for adjacent in self.geometry.neighbors(*cell):
                if self._board.get(adjacent) == color and adjacent not in parents:
                    parents[adjacent] = cell
                    queue.append(adjacent)
        return ()

    def specification(self) -> dict:
        geometry = self.geometry
        common = {'game_id': geometry.game_id, 'level': geometry.level, 'size': geometry.size,
                  'turn_order': 'black_first_then_alternating', 'capture': False, 'pass': False,
                  'gravity': False, 'swap_rule': 'disabled_fixed_color_ownership',
                  'competitive_comparison': 'paired_games_reverse_agent_color_assignments',
                  'inventory_counts': geometry.inventory_counts}
        if geometry.shape == 'hex':
            return {**common, 'game': 'Hex', 'neighbors_axial': [list(v) for v in NEIGHBOR_OFFSETS],
                    'goals': {'black': 'connect r=0 to r=6', 'white': 'connect q=0 to q=6'},
                    'draw': False,
                    'rules_reference': 'https://www.hexboard.com/Hex_Connections_Rule_Guide_2017.pdf'}
        return {**common, 'game': ('Tic-tac-toe' if geometry.level == 1 else 'Planar connect four'),
                'variant': ('standard_3x3' if geometry.level == 1 else 'custom_5x5_planar_no_gravity'),
                'win_length': geometry.win_length,
                'winning_directions': ['horizontal', 'vertical', 'diagonal_ascending', 'diagonal_descending'],
                'winning_line': 'at_least_win_length_consecutive_same_color_cells',
                'draw': 'full_board_without_winning_line', 'placement': 'any_empty_cell'}
