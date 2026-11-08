"""Pure scene rules for four proposed physical games.

These classes are NOT an HTTP adapter or physical placement detector. The
task-owned ``expansion_referee.ExpansionMatch`` verifies native geometry,
support, release and full sampling before applying a logical operation. No function moves a body. Hidden
memory identities belong only to the trusted referee; ``public_state`` never
publishes an unrevealed identity. Existing tabletop protocols are untouched.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import math

from .game_designs import design_for


class IllegalMove(ValueError):
    """Rejected logical request; no turn or board change."""


class PhysicalEvidenceRequired(ValueError):
    """The caller has not established a physical event; no scoring credit."""


def _cell(value, size):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) is not int or not 0 <= v < size for v in value)):
        raise IllegalMove('Cell is outside the square board')
    return tuple(value)


def _pair_matches(card_ids, expected):
    return (isinstance(card_ids, (tuple, list)) and len(card_ids) == 2
            and all(isinstance(card, str) for card in card_ids)
            and len(set(card_ids)) == 2 and set(card_ids) == set(expected))


def _metric_xy(position):
    return (isinstance(position, (tuple, list)) and len(position) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in position))


def rotated_cells(cells, rotation_deg=0):
    """Rotate about Z, then normalize the bounding box; no reflections."""
    if type(rotation_deg) is not int or rotation_deg not in (0, 90, 180, 270):
        raise IllegalMove('Only 0, 90, 180 or 270 degree rotations are allowed')
    points = [tuple(cell) for cell in cells]
    for _ in range(rotation_deg // 90):
        points = [(-r, q) for q, r in points]
    min_q, min_r = min(q for q, _ in points), min(r for _, r in points)
    return tuple(sorted((q-min_q, r-min_r) for q, r in points))


class _Game:
    def __init__(self, task_id):
        self.design = design_for(task_id)
        self.players = tuple(self.design['players'])
        if len(self.players) != 2 or len(set(self.players)) != 2:
            raise ValueError('Two distinct players are required')
        self.turn_index = 0
        self._player_index = 0
        self.status = 'active'
        self.winner = None
        self.draw = False
        self.termination_reason = None
        self.physics_valid = None

    @property
    def current_player(self):
        return self.players[self._player_index] if self.status == 'active' else None

    @property
    def terminal(self):
        return self.status != 'active'

    def _require_turn(self, player):
        if self.terminal:
            raise IllegalMove('The game is terminal')
        if player != self.current_player:
            raise IllegalMove('Only the current player may act')

    def _advance(self, *, repeat=False):
        self.turn_index += 1
        if not repeat:
            self._player_index = 1-self._player_index

    def _finish_scores(self, scores):
        a, b = self.players
        self.status = 'completed'
        self.draw = scores[a] == scores[b]
        self.winner = None if self.draw else (a if scores[a] > scores[b] else b)
        self.termination_reason = 'rules_completed'

    def interrupt(self, reason, *, physics_valid=None):
        """Budgets/faults never award a winner; completed results stay latched."""
        if reason not in ('time_limit', 'step_limit', 'cancelled', 'runtime_error',
                          'missing_physical_evidence', 'physical_failure', 'shot_rule_violation', 'rule_violation'):
            raise ValueError('Unknown interruption reason')
        if physics_valid is not None and type(physics_valid) is not bool:
            raise ValueError('physics_valid must be bool or None')
        if self.terminal:
            return self.result()
        self.status = 'incomplete'
        self.termination_reason = reason
        self.physics_valid = physics_valid
        return self.result()

    def result(self):
        return {'status': self.status, 'winner': self.winner, 'draw': self.draw,
                'termination_reason': self.termination_reason,
                'physics_valid': self.physics_valid, 'turn_index': self.turn_index,
                'scope': 'scene_rules_only_no_robot_or_model_qualification'}


class PolyominoTerritoryRules(_Game):
    """6x6 area competition; logical moves require upstream physical settlement."""
    def __init__(self):
        super().__init__('polyomino_territory')
        self.size = self.design['board']['rows']
        self.pieces = {piece['id']: piece for piece in self.design['pieces']
                       if 'cells' in piece}
        self.placed = {}
        self.board = {}

    def placement_cells(self, piece_id, anchor, rotation_deg=0):
        if piece_id not in self.pieces:
            raise IllegalMove('Unknown piece')
        q, r = _cell(anchor, self.size)
        cells = tuple((q+dq, r+dr) for dq, dr in rotated_cells(
            self.pieces[piece_id]['cells'], rotation_deg))
        if any(not (0 <= x < self.size and 0 <= y < self.size) for x, y in cells):
            raise IllegalMove('The whole piece must remain within the board')
        if any(cell in self.board for cell in cells):
            raise IllegalMove('Pieces cannot overlap')
        return cells

    def legal_moves(self, player=None):
        player = self.current_player if player is None else player
        if self.terminal or player not in self.players:
            return ()
        result = []
        for identifier, piece in self.pieces.items():
            if piece['owner'] != player or identifier in self.placed:
                continue
            seen = set()
            for angle in (0, 90, 180, 270):
                shape = rotated_cells(piece['cells'], angle)
                if shape in seen:
                    continue
                seen.add(shape)
                for r in range(self.size):
                    for q in range(self.size):
                        try:
                            self.placement_cells(identifier, (q, r), angle)
                        except IllegalMove:
                            continue
                        result.append((identifier, (q, r), angle))
        return tuple(result)

    def scores(self):
        return {player: sum(len(value['cells']) for key, value in self.placed.items()
                            if self.pieces[key]['owner'] == player) for player in self.players}

    def _check_end(self):
        if all(not self.legal_moves(player) for player in self.players):
            self._finish_scores(self.scores())

    def place(self, player, piece_id, anchor, rotation_deg=0):
        self._require_turn(player)
        if piece_id not in self.pieces or self.pieces[piece_id]['owner'] != player:
            raise IllegalMove('Use an own inventory piece')
        if piece_id in self.placed:
            raise IllegalMove('A piece may be placed only once')
        cells = self.placement_cells(piece_id, anchor, rotation_deg)
        self.placed[piece_id] = {'cells': cells, 'rotation_deg': rotation_deg}
        self.board.update({cell: piece_id for cell in cells})
        self._advance()
        self._check_end()
        return self.result()

    def pass_turn(self, player):
        self._require_turn(player)
        if self.legal_moves(player):
            raise IllegalMove('Pass is allowed only when no own piece fits')
        self._advance()
        self._check_end()
        return self.result()


class BarrierRaceRules(_Game):
    """5x5 pawn race. Walls block two edges and must preserve both paths."""
    def __init__(self):
        super().__init__('barrier_race')
        self.size = self.design['board']['rows']
        self.positions = {self.players[0]: (self.size//2, 0),
                          self.players[1]: (self.size//2, self.size-1)}
        self.goals = {self.players[0]: self.size-1, self.players[1]: 0}
        self.walls = {}
        self.wall_inventory = {piece['id']: piece['owner'] for piece in self.design['pieces']
                               if 'barrier' in piece['id']}

    @staticmethod
    def _edge(a, b):
        return tuple(sorted((a, b)))

    def wall_edges(self, orientation, anchor):
        q, r = _cell(anchor, self.size-1)
        if orientation == 'horizontal':
            return {self._edge((q, r), (q, r+1)), self._edge((q+1, r), (q+1, r+1))}
        if orientation == 'vertical':
            return {self._edge((q, r), (q+1, r)), self._edge((q, r+1), (q+1, r+1))}
        raise IllegalMove('Wall orientation is horizontal or vertical')

    def blocked_edges(self):
        return set().union(*(value['edges'] for value in self.walls.values()))

    def _neighbor(self, cell, offset, blocked):
        candidate = (cell[0]+offset[0], cell[1]+offset[1])
        if (0 <= candidate[0] < self.size and 0 <= candidate[1] < self.size
                and self._edge(cell, candidate) not in blocked):
            return candidate
        return None

    def has_path(self, player, blocked=None):
        blocked = self.blocked_edges() if blocked is None else blocked
        queue = deque([self.positions[player]])
        seen = set(queue)
        while queue:
            cell = queue.popleft()
            if cell[1] == self.goals[player]:
                return True
            for offset in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_cell = self._neighbor(cell, offset, blocked)
                if next_cell is not None and next_cell not in seen:
                    seen.add(next_cell)
                    queue.append(next_cell)
        return False

    def legal_pawn_moves(self, player=None):
        player = self.current_player if player is None else player
        if self.terminal or player not in self.players:
            return ()
        blocked = self.blocked_edges()
        start = self.positions[player]
        opponent = self.positions[self.players[1-self.players.index(player)]]
        moves = set()
        for dq, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            adjacent = self._neighbor(start, (dq, dr), blocked)
            if adjacent is None:
                continue
            if adjacent != opponent:
                moves.add(adjacent)
                continue
            beyond = self._neighbor(opponent, (dq, dr), blocked)
            if beyond is not None:
                moves.add(beyond)
            else:
                # A wall or board edge behind the opponent permits side steps.
                for offset in ((-dr, dq), (dr, -dq)):
                    side = self._neighbor(opponent, offset, blocked)
                    if side is not None:
                        moves.add(side)
        return tuple(sorted(moves))

    def move_pawn(self, player, destination):
        self._require_turn(player)
        destination = _cell(destination, self.size)
        if destination not in self.legal_pawn_moves(player):
            raise IllegalMove('Illegal pawn move')
        self.positions[player] = destination
        self._advance()
        if destination[1] == self.goals[player]:
            self.status, self.winner, self.termination_reason = 'completed', player, 'opposite_edge_reached'
        return self.result()

    def place_wall(self, player, wall_id, orientation, anchor):
        self._require_turn(player)
        if self.wall_inventory.get(wall_id) != player or wall_id in self.walls:
            raise IllegalMove('Use an unused own wall')
        edges = self.wall_edges(orientation, anchor)
        anchor = tuple(anchor)
        blocked = self.blocked_edges()
        if edges & blocked or any(value['anchor'] == anchor and value['orientation'] != orientation
                                   for value in self.walls.values()):
            raise IllegalMove('Walls may neither overlap nor cross')
        candidate = blocked | edges
        if not all(self.has_path(p, candidate) for p in self.players):
            raise IllegalMove('A wall must leave a path to each opposite edge')
        self.walls[wall_id] = {'anchor': anchor, 'orientation': orientation, 'edges': edges}
        self._advance()
        return self.result()


class MemoryPairsRules(_Game):
    """Private identity map with ordered, public reveal/return/collection events.

    reveal/collect_pair/return_pair mean *verified physical confirmations* at the
    native referee boundary. This pure module does not itself detect card poses.
    """
    def __init__(self, private_pairs):
        super().__init__('memory_pairs')
        cards = {piece['id'] for piece in self.design['pieces'] if piece['id'].startswith('card_')}
        if (not isinstance(private_pairs, dict) or set(private_pairs) != cards
                or any(not isinstance(v, str) or not v for v in private_pairs.values())
                or sorted(Counter(private_pairs.values()).values()) != [2]*6):
            raise ValueError('Bind exactly six pairs to the twelve actual card identities')
        self._private_pairs = dict(private_pairs)
        self._exposed = []
        self._collected = {}
        self._events = []
        self.phase = 'reveal_first'

    def _event(self, event_type, **values):
        event = {'sequence': len(self._events), 'type': event_type,
                 'player': self.current_player, 'turn_index': self.turn_index, **values}
        self._events.append(event)
        return deepcopy(event)

    def events_after(self, sequence=-1):
        if type(sequence) is not int or sequence < -1:
            raise ValueError('Sequence must be an integer at least -1')
        return deepcopy([event for event in self._events if event['sequence'] > sequence])

    def public_state(self):
        return {'task_id': 'memory_pairs', 'phase': self.phase,
                'current_player': self.current_player, 'turn_index': self.turn_index,
                'face_up': [{'card_id': card, 'symbol': self._private_pairs[card]} for card in self._exposed],
                'collected': dict(self._collected), 'scores': self.scores(),
                'last_event_sequence': len(self._events)-1, 'result': self.result()}

    def scores(self):
        return {player: sum(owner == player for owner in self._collected.values())//2
                for player in self.players}

    def reveal(self, player, card_id):
        self._require_turn(player)
        if self.phase not in ('reveal_first', 'reveal_second'):
            raise IllegalMove('Collect the match or return the mismatch before another reveal')
        if card_id not in self._private_pairs or card_id in self._exposed or card_id in self._collected:
            raise IllegalMove('Choose a distinct uncollected face-down card')
        self._exposed.append(card_id)
        event = self._event('card_revealed', card_id=card_id, symbol=self._private_pairs[card_id])
        if len(self._exposed) == 1:
            self.phase = 'reveal_second'
        else:
            self.phase = ('collect_match' if self._private_pairs[self._exposed[0]] ==
                          self._private_pairs[self._exposed[1]] else 'return_mismatch')
        return event

    def collect_pair(self, player, card_ids):
        self._require_turn(player)
        if self.phase != 'collect_match' or not _pair_matches(card_ids, self._exposed):
            raise IllegalMove('Collect exactly the revealed matching pair')
        self._collected.update({card: player for card in self._exposed})
        event = self._event('pair_collected', card_ids=list(self._exposed))
        self._exposed.clear()
        self.phase = 'reveal_first'
        self._advance(repeat=True)
        if len(self._collected) == len(self._private_pairs):
            self.phase = 'complete'
            self._finish_scores(self.scores())
        return event

    def return_pair(self, player, card_ids):
        self._require_turn(player)
        if self.phase != 'return_mismatch' or not _pair_matches(card_ids, self._exposed):
            raise IllegalMove('Return exactly both mismatched cards')
        event = self._event('pair_returned_face_down', card_ids=list(self._exposed))
        self._exposed.clear()
        self.phase = 'reveal_first'
        self._advance()
        return event


class DiskCurlingRules(_Game):
    """Eight alternating one-contact shots, scored only from final settled poses.

    Evidence arguments are trusted scene-observer outputs, not Agent requests.
    They express the trusted referee boundary and do not alone certify a rollout.
    """
    def __init__(self):
        super().__init__('disk_curling')
        self.disks = {piece['id']: piece['owner'] for piece in self.design['pieces']
                      if '_disk_' in piece['id']}
        self.played = []
        self.final_positions = {}
        self.final_on_deck = {}
        self.final_scores = None

    def record_shot(self, player, disk_id, *, contact_episodes,
                    complete_native_evidence, physics_valid, released,
                    stable_duration_s, positions_xy_m, launch_position_xy_m,
                    release_position_xy_m, crossed_line_after_release,
                    other_disk_robot_contacts, reserve_inventory_preserved, on_deck):
        self._require_turn(player)
        if self.disks.get(disk_id) != player or disk_id in self.played:
            raise IllegalMove('Use an unplayed own disk')
        if type(contact_episodes) is not int or contact_episodes < 0:
            raise IllegalMove('contact_episodes must be a nonnegative integer')
        if complete_native_evidence is not True or physics_valid is None:
            raise PhysicalEvidenceRequired('Complete native contact and physical-validity evidence is required')
        if physics_valid is False:
            return self.interrupt('physical_failure', physics_valid=False)
        if physics_valid is not True:
            raise PhysicalEvidenceRequired('Explicit physical-validity evidence is required')
        if type(other_disk_robot_contacts) is not int or other_disk_robot_contacts < 0:
            raise PhysicalEvidenceRequired('Other-disk robot contact count must be observed')
        if contact_episodes > 1:
            # An observed second push cannot be erased by retrying the request.
            return self.interrupt('shot_rule_violation', physics_valid=True)
        if contact_episodes == 0:
            raise PhysicalEvidenceRequired('No observed valid launch contact')
        if not _metric_xy(launch_position_xy_m) or not _metric_xy(release_position_xy_m):
            raise PhysicalEvidenceRequired('Observe actual start and release positions')
        board = self.design['board']
        region = board['launch_region_bounds_xy_m'][player]
        line = board['launch_lines_y_m'][player]
        direction = 1 if player == self.players[0] else -1
        in_region = all(region[0][i] <= launch_position_xy_m[i] <= region[1][i] for i in (0, 1))
        if (not in_region or direction*(release_position_xy_m[1]-line) > 1e-12
                or other_disk_robot_contacts):
            return self.interrupt('shot_rule_violation', physics_valid=True)
        if crossed_line_after_release is not True or reserve_inventory_preserved is not True:
            raise PhysicalEvidenceRequired('Confirm free line crossing and unchanged unplayed inventory')
        if (released is not True or type(stable_duration_s) not in (int, float)
                or not math.isfinite(stable_duration_s) or stable_duration_s+1.e-12 < .3):
            raise PhysicalEvidenceRequired('All in-play disks need a fresh 300 ms released stable window')
        expected = set(self.played) | {disk_id}
        if not isinstance(positions_xy_m, dict) or set(positions_xy_m) != expected:
            raise PhysicalEvidenceRequired('Provide the same final-window position of every in-play disk')
        if (not isinstance(on_deck, dict) or set(on_deck) != expected
                or any(type(value) is not bool for value in on_deck.values())):
            raise PhysicalEvidenceRequired('Native Deck-rooted scoring eligibility is required for every played disk')
        checked = {}
        for identifier, position in positions_xy_m.items():
            if not _metric_xy(position):
                raise PhysicalEvidenceRequired('Every disk needs a finite metric XY position')
            checked[identifier] = tuple(position)
        self.final_positions = checked
        self.final_on_deck = dict(on_deck)
        self.played.append(disk_id)
        self._advance()
        if len(self.played) == len(self.disks):
            self.final_scores = self.score_positions(checked, on_deck=on_deck)
            self._finish_scores(self.final_scores)
        return self.result()

    def score_positions(self, positions_xy_m, *, on_deck):
        """Private scoring calculation; never a live Agent truth endpoint."""
        if (not isinstance(positions_xy_m, dict) or not isinstance(on_deck, dict)
                or set(on_deck) != set(positions_xy_m)
                or any(type(value) is not bool for value in on_deck.values())):
            raise PhysicalEvidenceRequired('Explicit observed Deck scoring eligibility is required')
        scoring = self.design['board']
        center = scoring['target_center_xy_m']
        radii = scoring['scoring_radii_m']
        points = scoring['scoring_points']
        bounds = scoring['deck_bounds_xy_m']
        scores = {player: 0 for player in self.players}
        for identifier, xy in positions_xy_m.items():
            if identifier not in self.disks:
                raise IllegalMove('Unknown disk')
            if not _metric_xy(xy):
                raise PhysicalEvidenceRequired('Scoring requires a finite metric position')
            if not on_deck[identifier] or not (bounds[0][0] <= xy[0] <= bounds[1][0] and bounds[0][1] <= xy[1] <= bounds[1][1]):
                continue
            radius = math.hypot(xy[0]-center[0], xy[1]-center[1])
            scores[self.disks[identifier]] += next((point for limit, point in zip(radii, points)
                                                   if radius <= limit+1e-12), 0)
        return scores


def public_rules(task_id):
    """Legacy pure-rule description; native releases use public_rule_contract v2.

    No private map or live scoring state is returned. Keep this original v1
    description separate from the complete physical rules and operation epochs.
    """
    design = design_for(task_id)
    details = {
        'polyomino_territory': {
            'goal': 'Maximize occupied area on the 6x6 board.',
            'inventory': 'Each player has the same six pieces totaling 17 cells.',
            'placement': 'Flat, whole footprint in bounds, no overlap; 0/90/180/270 degree rotations, no reflection.',
            'adjacency': 'No corner-start or adjacency restriction.',
            'pass': 'Only if none of the unused own pieces has any legal placement.',
            'end': 'Both players have no legal placement (including exhausted inventory); compare occupied cells; equality is a draw.'},
        'barrier_race': {
            'goal': 'First own pawn reaching any cell of the opposite row wins.',
            'turn': 'Move own pawn or place one unused own two-edge wall; four walls per player.',
            'move': 'One unblocked orthogonal edge. If the adjacent cell holds the opponent, jump straight over it when the next edge and cell exist. Only when that straight jump is blocked by a wall or board boundary, either unblocked orthogonal side exit from the opponent is legal.',
            'walls': 'Horizontal anchor(q,r) blocks (q,r)-(q,r+1) and (q+1,r)-(q+1,r+1); vertical blocks (q,r)-(q+1,r) and (q,r+1)-(q+1,r+1). q,r are 0..3. Shared blocked edges and crossing anchors are forbidden.',
            'path': 'After each wall both pawns must retain an orthogonal path to their goal row, treating the other pawn as movable.'},
        'memory_pairs': {
            'goal': 'Collect more of six pairs from twelve face-down cards.',
            'turn': 'Reveal two distinct uncollected cards sequentially. A match must be collected before continuing with the same player. A mismatch must have both cards physically returned face down before changing players.',
            'observation': 'Every confirmed reveal produces a monotonic public event with card ID and visible symbol for both players; the host must provide both reveal observations before the cards can be returned.',
            'hidden': 'Unrevealed identities and the shuffled pair map remain private; reset must bind the actual USD face patterns.',
            'end': 'All six pairs collected; compare pairs, equality is a draw.'},
        'disk_curling': {
            'goal': 'Higher total final score after each player launches four disks.',
            'turn': 'Alternate one unplayed own disk. Prepare/reposition it inside the own launch region before effective propulsion; preparation earns no shot. One continuous effective robot-disk contact episode launches it; release on or before the own launch line, then cross it freely toward the center. Subsequent robot contact is a forbidden second push. Disk-disk collisions with in-play disks are allowed; robot contact with another disk is not.',
            'launch': 'Black launches from the south toward +Y; white from the north toward -Y. Start/release poses and free crossing must be observed; unplayed reserves remain at their preparation positions. Exact regions and lines are in scoring_geometry.',
            'settlement': 'No new shot until every in-play disk is released and stable for a fresh 300 ms with complete native evidence. Zero-contact attempts earn no shot.',
            'score': 'Only in-play disks score from final centers after all eight shots. Inner/middle/outer rings score 3/2/1, outer area and off-deck disks 0. An exact circle boundary belongs to the inner zone. Scores may change as later shots move earlier disks.',
            'native_violation': 'A recorded second push ends incomplete with shot_rule_violation and no winner; it is distinct from a physics failure.',
            'scoring_geometry': deepcopy(design['board'])},
    }
    if task_id not in details:
        raise ValueError('Unknown game')
    return {'schema': 'card_room_expansion_rules.v1', 'task_id': task_id,
            'players': list(design['players']), 'first_player': design['players'][0],
            'rules': details[task_id],
            'invalid_request': 'Reject without changing state or consuming a turn.',
            'interruptions': 'Time/step budgets, cancellation, missing evidence and runtime failures are incomplete without winner; a previously completed result remains latched.',
            'physical_commit': 'Logical intent does not move or settle a body. Future native observation must verify every task-specific operation, support, release, full sampling and physical validity before applying these pure rules.',
            'integration_status': 'scene_candidate_only; no public game session adapter or robot/model qualification'}
