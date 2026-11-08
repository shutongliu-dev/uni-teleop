"""Task-owned native referees for the four card-room games.

Only the private native observer may construct evidence. Player intents contain
no evidence, scores, poses or outcomes. This module never writes a physical pose
and imports neither Bench nor a simulator. Old tabletop protocols are separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import math
from typing import Mapping

from .expansion_rules import (BarrierRaceRules, DiskCurlingRules, IllegalMove,
                              MemoryPairsRules, PolyominoTerritoryRules, public_rules)
from .game_designs import design_for

PLAYERS = ('player_0', 'player_1')
COLORS = ('black', 'white')
INTERRUPTIONS = ('time_limit', 'step_limit', 'cancelled', 'runtime_error',
                 'missing_physical_evidence', 'physical_failure',
                 'shot_rule_violation', 'rule_violation')


@dataclass(frozen=True)
class ExpansionThresholds:
    dwell_s: float = .300
    maximum_linear_speed_m_s: float = .020
    maximum_angular_speed_rad_s: float = .200
    maximum_drift_m: float = .003
    position_tolerance_m: float = .004
    maximum_angle_rad: float = math.radians(5)
    maximum_penetration_m: float = .003
    bottom_tolerance_m: float = .003
    minimum_contact_impulse_ns: float = 1.e-8
    minimum_support_weight_fraction: float = .05
    maximum_sample_gap_s: float = .020
    minimum_operation_displacement_m: float = .005
    minimum_operation_rotation_rad: float = math.radians(10)
    disk_propulsion_speed_m_s: float = .040
    minimum_reveal_upward_normal_component: float = 1.e-6

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in asdict(self).values()):
            raise ValueError('Physical thresholds must be finite and positive')


@dataclass(frozen=True)
class ExpansionObjectEvidence:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity_m_s: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]
    deck_support_impulse_ns: float
    robot_contact_impulses_ns: Mapping[str, float]
    maximum_penetration_m: float | None
    object_support_impulses_ns: Mapping[str, float] = field(default_factory=dict)
    robot_nonhand_contact_impulses_ns: Mapping[str, float] = field(default_factory=dict)
    other_static_support_impulse_ns: float = 0.


@dataclass(frozen=True)
class ExpansionNativeFrame:
    substep_index: int
    simulation_time_s: float
    dt_s: float
    objects: Mapping[str, ExpansionObjectEvidence]
    contacts_complete: bool
    source: str = 'native_solver_substeps'


@dataclass(frozen=True)
class ExpansionIntent:
    player_id: str
    type: str
    piece_ids: tuple[str, ...]
    parameters: dict
    turn_index: int
    attempt_index: int


@dataclass(frozen=True)
class ExpansionResult:
    accepted: bool
    reason: str
    dwell_s: float = 0.
    committed: bool = False
    recovery_required: bool = False
    evidence_unavailable: bool = False


def _norm(values):
    return math.sqrt(sum(v*v for v in values))


def _distance(a, b):
    return _norm(x-y for x, y in zip(a, b))


def _rotation(q):
    w, x, y, z = q
    n = sum(v*v for v in q)
    return ((1-2*(y*y+z*z)/n, 2*(x*y-z*w)/n, 2*(x*z+y*w)/n),
            (2*(x*y+z*w)/n, 1-2*(x*x+z*z)/n, 2*(y*z-x*w)/n),
            (2*(x*z-y*w)/n, 2*(y*z+x*w)/n, 1-2*(x*x+y*y)/n))


def _angle(a, b):
    cosine = min(1., abs(sum(x*y for x, y in zip(a, b)))/(_norm(a)*_norm(b)))
    return 2*math.acos(cosine)


def _yaw(degrees):
    angle = math.radians(degrees)/2
    return (math.cos(angle), 0., 0., math.sin(angle))


def _hull(points):
    points = sorted(set(points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for output, stream in ((lower, points), (upper, reversed(points))):
        for p in stream:
            while len(output) >= 2 and cross(output[-2], output[-1], p) <= 0:
                output.pop()
            output.append(p)
    return lower[:-1]+upper[:-1]


def _area(polygon):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(polygon, polygon[1:]+polygon[:1])))/2


def _clip_rectangle(polygon, low, high):
    """Exact convex-polygon clipping, used for occupied-cell footprint area."""
    for axis in (0, 1):
        for boundary, sign in ((low[axis], 1), (high[axis], -1)):
            output = []
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                inside_a, inside_b = sign*(a[axis]-boundary) >= 0, sign*(b[axis]-boundary) >= 0
                if inside_a:
                    output.append(a)
                if inside_a != inside_b:
                    alpha = (boundary-a[axis])/(b[axis]-a[axis])
                    output.append(tuple(a[i]+alpha*(b[i]-a[i]) for i in (0, 1)))
            polygon = output
            if not polygon:
                return []
    return polygon


def public_rule_contract(task_id):
    """Complete static, secret-free contract for a source-bound release."""
    design = design_for(task_id)
    actions = {
        'polyomino_territory': {'place_piece': ['piece_id', 'anchor', 'rotation_deg'], 'pass': []},
        'barrier_race': {'move_pawn': ['destination'],
                         'place_barrier': ['piece_id', 'anchor', 'orientation']},
        'memory_pairs': {'reveal_card': ['card_id'], 'collect_pair': [], 'flip_back': []},
        'disk_curling': {'shoot_disk': ['piece_id']},
    }
    coordinates = {
        'polyomino_territory': {
            'cell_axes': 'q increases along world +X; r increases along world +Y. Cell (0,0) is the southwest cell.',
            'cell_center_formula': 'center_xy + [(q-(columns-1)/2)*pitch, (r-(rows-1)/2)*pitch]',
            'anchor': 'Integer (q,r) at the lower-left of the rotated shape bounding box. Rotate canonical cells around +Z, subtract each rotated coordinate minimum, then add anchor.',
            'rotation_deg': '0,90,180,270; positive rotation is counterclockwise when viewed from above (+Z). Reflections are forbidden.',
            'body_origin': 'The authored body origin is the center of its full shape bounding box. Place that origin at the center of the occupied-cell bounding box; do not use the anchor as the rigid-body center.',
            'footprint': 'The entire projected union of the real collision boxes must stay inside the intended occupied cells, including the concave boundary.'},
        'barrier_race': {
            'cell_axes': 'q increases along world +X; r increases along world +Y. Cell (0,0) is southwest.',
            'cell_center_formula': 'center_xy + [(q-2)*pitch, (r-2)*pitch]',
            'wall_anchor': 'Integer anchor(q,r), q,r=0..3, names the southwest cell among the four cells around the wall center.',
            'wall_center_formula': 'center_xy + [(q-1.5)*pitch, (r-1.5)*pitch]',
            'wall_orientation': 'horizontal: long axis world X; vertical: long axis world Y. An in-plane half-turn is the same wall orientation.',
            'pawn_orientation': 'Pawn must stand upright; yaw is immaterial.'},
        'memory_pairs': {
            'reveal_slot': 'Flip a selected card face up, then release it in its original nominal XY slot. Either physical flip axis and any in-plane yaw are allowed.',
            'front_normal': 'Printed front is local -Z. Any upward front-normal component above minimum_reveal_upward_normal_component publishes the identity immediately, including partial flips. Only physical confirmation requires near-horizontal face-up orientation and released stability.',
            'return_slot': 'Return each mismatch face down at its original nominal XY slot; yaw is immaterial.',
            'collection': 'Whole projected footprint must fit inside the own collection rectangle. Face-up stacks need real acyclic upward contact chains through only the same-owner collected cards or the current matching pair, rooted on Deck.'},
        'disk_curling': {
            'world_axes': 'Black/player_0 launches north along +Y; white/player_1 launches south along -Y.',
            'preparation_region': 'The selected disk entire projected footprint stays inside its own launch_region_bounds_xy_m before effective propulsion.',
            'line': 'Release at/before own launch_lines_y_m, then cross toward the target without robot contact. Both release and crossing compare the disk body-center world Y with the line, not its leading rim. A contact recorded after release is a forbidden second push.',
            'score_center': 'Scoring uses the final actual world XY body center only when the disk has a real support chain rooted on Deck. A stable disk resting on an allowed non-scoring surface has zero score, even when its XY projection lies inside a ring.'},
    }
    return {'schema': 'card_room_expansion_physical_rules.v2', 'task_id': task_id,
            'players': [{'player_id': p, 'color': c} for p, c in zip(PLAYERS, COLORS)],
            'first_player': PLAYERS[0], 'rules': public_rules(task_id)['rules'],
            'metric_design': design, 'thresholds': asdict(ExpansionThresholds()),
            'coordinate_convention': coordinates[task_id],
            'intents': {**actions[task_id], 'cancel_intent': []},
            'defaults': {'place_piece.rotation_deg': 0},
            'authorization': 'turn_index is the confirmed physical-operation epoch, not player parity; every reveal, collection, return and pass increments it. round_index counts logical player turns.',
            'physical_commit': 'Intent only reserves an operation. Complete consecutive native substeps, real own-hand contact and motion, valid geometry, native rooted support and a fresh released 300ms stable window confirm it. Pass commits only on a subsequent safe native frame.',
            'operation_credit': 'For placement, pawn/wall movement, card flip, return and collection, the unchanged minimum operation displacement/rotation thresholds measure net pose change within one continuous effective own-hand contact segment, starting at the preceding native sample. Motion before or after that segment is not credited, and travelled path length is not accumulated. A contact gap discards an unqualified segment; regrasp may start another. Qualified operation credit remains during release and settlement. Disk shots instead retain their separate effective-propulsion speed, single-contact and free-line-crossing conditions, without an additional displacement/rotation threshold.',
            'memory_reveals': 'Every newly face-up uncollected card produces a durable public card_revealed event for both players, before operation_committed. An unexpected reveal is broadcast then ends incomplete with rule_violation. Unrevealed identities are never public.',
            'memory_collection': 'Collect the revealed matching pair in the own marked collection area. Same-owner previously collected cards may support a real acyclic stack rooted on Deck; new pair cards may support one another. Mismatches return face down to their original slots. Both cards require fresh own-hand manipulation and released stability.',
            'disk_preparation': 'Before the first effective forward push, reposition or lift only the selected unplayed disk within the own launch region. Effective propulsion begins at the declared forward-speed threshold while flat, Deck-supported and in own-hand contact. After that exactly one continuous contact episode is permitted; all contact must end on or before the own line, then the disk crosses freely. Arm-only propulsion and another disk robot contact are forbidden.',
            'disk_orientation': 'Deck scoring and launch require a flat cylinder axis: abs(rotation_world[2][2]) >= cos(maximum_angle_rad); either disk face may be up. An off-deck played disk may rest at any orientation once real non-scoring support and released stability are measured.',
            'disk_off_deck': 'An off-deck played disk scores zero only after actual known-static support and released stability; no support is inferred from a still pose.',
            'interruptions': list(INTERRUPTIONS),
            'interruption_result': 'Incomplete/aborted or truncated, never a fabricated winner; a completed result stays latched.',
            'native_privacy': 'No raw poses, impulses, private reset map, unrevealed identities or private scorer state enter player observations.',
            'scope': 'Executable rules and native evidence consumer; qualification is established separately.'}


class ExpansionMatch:
    """One authoritative state machine; the observer feeds all native substeps."""

    players = PLAYERS

    def __init__(self, task_id, *, asset_metadata, private_pairs=None, thresholds=None):
        self.task_id = task_id
        self.design = design_for(task_id)
        self.thresholds = thresholds or ExpansionThresholds()
        self._specs = {p['id']: p for p in self.design['pieces']}
        self.inventory = frozenset(self._specs)
        if (not isinstance(asset_metadata, Mapping) or asset_metadata.get('task_id') != task_id
                or set(asset_metadata.get('actors', {})) != self.inventory
                or asset_metadata.get('dimensions') != self.design
                or set(asset_metadata.get('masses_kg', {})) != self.inventory):
            raise ValueError('Bind the complete actual task asset metadata and metric design')
        self._masses = dict(asset_metadata['masses_kg'])
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in self._masses.values()):
            raise ValueError('Actual authored masses must be finite and positive')
        for key, spec_key in (('initial_positions_m', 'initial_position_m'),
                              ('initial_orientations_wxyz', 'initial_orientation_wxyz')):
            expected = {name: spec[spec_key] for name, spec in self._specs.items()}
            if asset_metadata.get(key) != expected:
                raise ValueError('Reset geometry differs from the bound metric design')
        if task_id == 'memory_pairs':
            bound = asset_metadata.get('private_reset_pairs')
            if private_pairs is not None and private_pairs != bound:
                raise ValueError('Private pair map must equal the actual authored face-pattern map')
            if not isinstance(bound, dict) or set(bound.values()) != {f'pair_{i}' for i in range(6)}:
                raise ValueError('Private patterns must match the six actual authored symbols')
            self.rules = MemoryPairsRules(deepcopy(bound))
        else:
            self.rules = {'polyomino_territory': PolyominoTerritoryRules,
                          'barrier_race': BarrierRaceRules,
                          'disk_curling': DiskCurlingRules}[task_id]()
        self.turn_index = 0
        self.pending: ExpansionIntent | None = None
        self.last_result = ExpansionResult(False, 'awaiting_native_evidence')
        self.commit_log = []
        self._events = []
        self._attempt_index = 0
        self._last_frame = None
        self._expected = {name: (tuple(spec['initial_position_m']), tuple(spec['initial_orientation_wxyz']))
                          for name, spec in self._specs.items()}
        self._face_up = set()
        self._contact_seen = set()
        self._motion_seen = set()
        self._hand_origins = {}
        self._attempt_revealed = False
        self._stable_start = None
        self._stable_positions = {}
        self._dwell = 0.
        self._shot = None

    @property
    def terminal(self):
        return self.rules.terminal

    @property
    def current_player(self):
        return None if self.terminal else PLAYERS[COLORS.index(self.rules.current_player)]

    @property
    def winner(self):
        return None if self.rules.winner is None else PLAYERS[COLORS.index(self.rules.winner)]

    @property
    def phase(self):
        if self.terminal:
            return 'complete' if self.rules.status == 'completed' else 'incomplete'
        if self._last_frame is None:
            return 'awaiting_native_baseline'
        if self.task_id == 'memory_pairs':
            return self.rules.phase
        if self.task_id == 'disk_curling' and self._shot:
            return 'released' if self._shot['released'] else ('propelling' if self._shot['started'] else 'preparing')
        return 'awaiting_physical_commit' if self.pending else 'awaiting_intent'

    @property
    def rule_contract(self):
        contract = public_rule_contract(self.task_id)
        contract['thresholds'] = asdict(self.thresholds)
        return contract

    def public_state(self):
        reason = self.rules.termination_reason if self.rules.status == 'incomplete' else None
        outcome = ('win' if self.winner is not None else 'draw' if self.rules.draw else
                   'truncated' if reason in ('time_limit', 'step_limit') else
                   'aborted' if self.terminal else 'ongoing')
        pending = asdict(self.pending) if self.pending else None
        if pending is not None:
            pending['piece_ids'] = list(pending['piece_ids'])
        return {'protocol': 'bench_card_game_state_v1', 'task_id': self.task_id,
                'turn_index': self.turn_index, 'round_index': self.rules.turn_index,
                'current_player': self.current_player, 'terminal': self.terminal,
                'winner': self.winner, 'draw': self.rules.draw, 'pending_move': pending,
                'status': {'outcome': outcome, 'end_reason': reason,
                           'recovery_required': self.last_result.recovery_required,
                           'evidence_unavailable': self.last_result.evidence_unavailable,
                           'reason': self.last_result.reason, 'retryable': not self.terminal},
                'settlement_dwell_s': self._dwell, 'commit_count': len(self.commit_log),
                'phase': self.phase, 'event_cursor': len(self._events)-1}

    def score(self):
        """Public referee state; never expose the private live scoring inputs."""
        return self.public_state()

    def events_since(self, after=-1, limit=100):
        if type(after) is not int or after < -1 or type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError('Event cursor must be >= -1 and limit within 1..256')
        if after > len(self._events)-1:
            raise ValueError('Event cursor is ahead of the published history')
        events = deepcopy(self._events[after+1:after+1+limit])
        cursor = events[-1]['sequence'] if events else after
        return {'events': events, 'next_cursor': cursor, 'has_more': cursor < len(self._events)-1}

    def events_after(self, after=-1):
        if type(after) is not int or not -1 <= after <= len(self._events)-1:
            raise ValueError('Invalid event cursor')
        return deepcopy(self._events[after+1:])

    def _event(self, kind, frame, player, **details):
        self._events.append({'sequence': len(self._events), 'type': kind, 'task_id': self.task_id,
                             'player_id': player, 'turn_index': self.turn_index,
                             'native_substep_index': frame.substep_index,
                             'simulation_time_s': frame.simulation_time_s, 'details': details})

    def require_actor(self, player):
        if self.terminal or player != self.current_player:
            raise IllegalMove('Only the active player may act in a nonterminal match')

    def _reset_attempt(self):
        self._contact_seen.clear()
        self._motion_seen.clear()
        self._hand_origins = {}
        self._attempt_revealed = False
        self._reset_dwell()
        self._shot = None

    def _reset_dwell(self):
        self._stable_start = None
        self._stable_positions = {}
        self._dwell = 0.

    def interrupt(self, reason):
        if reason not in INTERRUPTIONS:
            raise ValueError('Unknown interruption reason')
        if not self.terminal:
            self.rules.interrupt(reason, physics_valid=False if reason == 'physical_failure' else None)
            self._reset_dwell()
            self.last_result = ExpansionResult(False, reason,
                                              evidence_unavailable=reason == 'missing_physical_evidence')
            self.pending = None
        return self.public_state()

    end_episode = interrupt

    def admit_intent(self, player, action):
        self.require_actor(player)
        if not isinstance(action, Mapping):
            raise IllegalMove('Intent must be an object')
        if action.get('type') == 'cancel_intent':
            if set(action) != {'type'}:
                raise IllegalMove('cancel_intent has exactly one field')
            self.cancel_intent(player)
            return None
        if self._last_frame is None:
            raise IllegalMove('A complete native reset baseline is required first')
        clone = deepcopy(self.rules)
        color = COLORS[PLAYERS.index(player)]
        kind = action.get('type')
        for identity_key in ('piece_id', 'card_id'):
            if identity_key in action and not isinstance(action[identity_key], str):
                raise IllegalMove('Object identity must be a string')
        ids, parameters = (), {}
        if self.task_id == 'polyomino_territory' and kind == 'place_piece':
            if set(action) not in ({'type', 'piece_id', 'anchor'}, {'type', 'piece_id', 'anchor', 'rotation_deg'}):
                raise IllegalMove('place_piece requires piece_id, anchor and optional rotation_deg')
            clone.place(color, action['piece_id'], action['anchor'], action.get('rotation_deg', 0))
            ids = (action['piece_id'],)
            parameters = {'anchor': list(action['anchor']), 'rotation_deg': action.get('rotation_deg', 0)}
        elif self.task_id == 'polyomino_territory' and kind == 'pass' and set(action) == {'type'}:
            clone.pass_turn(color)
        elif self.task_id == 'barrier_race' and kind == 'move_pawn' and set(action) == {'type', 'destination'}:
            clone.move_pawn(color, action['destination'])
            ids, parameters = (color+'_pawn',), {'destination': list(action['destination'])}
        elif self.task_id == 'barrier_race' and kind == 'place_barrier' and set(action) == {'type', 'piece_id', 'anchor', 'orientation'}:
            clone.place_wall(color, action['piece_id'], action['orientation'], action['anchor'])
            ids = (action['piece_id'],)
            parameters = {'anchor': list(action['anchor']), 'orientation': action['orientation']}
        elif self.task_id == 'memory_pairs' and kind == 'reveal_card' and set(action) == {'type', 'card_id'}:
            clone.reveal(color, action['card_id'])
            ids = (action['card_id'],)
        elif self.task_id == 'memory_pairs' and kind in ('collect_pair', 'flip_back') and set(action) == {'type'}:
            ids = tuple(clone._exposed)
            (clone.collect_pair if kind == 'collect_pair' else clone.return_pair)(color, ids)
        elif self.task_id == 'disk_curling' and kind == 'shoot_disk' and set(action) == {'type', 'piece_id'}:
            identifier = action['piece_id']
            if clone.disks.get(identifier) != color or identifier in clone.played:
                raise IllegalMove('Use an unplayed own disk')
            ids = (identifier,)
        else:
            raise IllegalMove('Unknown or malformed task-specific intent')
        if self.pending and (self.pending.player_id, self.pending.type, self.pending.piece_ids,
                             self.pending.parameters) == (player, kind, ids, parameters):
            return self.public_state()['pending_move']
        if self.pending and self._attempt_irreversible():
            raise IllegalMove('An observed reveal or launched shot cannot be erased by replacing its intent')
        self._attempt_index += 1
        self.pending = ExpansionIntent(player, kind, ids, parameters, self.turn_index, self._attempt_index)
        self._reset_attempt()
        if kind == 'shoot_disk':
            self._shot = {'started': False, 'released': False, 'crossed': False,
                          'launch': None, 'release': None}
        self.last_result = ExpansionResult(False, 'awaiting_physical_commit')
        return self.public_state()['pending_move']

    def _attempt_irreversible(self):
        return bool((self._shot and self._shot['started']) or
                    (self.pending and self.pending.type == 'reveal_card' and self._attempt_revealed))

    def cancel_intent(self, player):
        self.require_actor(player)
        if self._attempt_irreversible():
            raise IllegalMove('A revealed card or launched disk cannot be cancelled')
        self.pending = None
        self._reset_attempt()
        self.last_result = ExpansionResult(False, 'awaiting_intent')

    def _valid_object(self, name, obj):
        if not isinstance(obj, ExpansionObjectEvidence):
            return False
        vectors = ((obj.position_m, 3), (obj.quaternion_wxyz, 4),
                   (obj.linear_velocity_m_s, 3), (obj.angular_velocity_rad_s, 3))
        if any(not isinstance(v, (list, tuple)) or len(v) != size
               or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v, size in vectors):
            return False
        if abs(sum(v*v for v in obj.quaternion_wxyz)-1) > 1.e-3:
            return False
        maps = (obj.robot_contact_impulses_ns, obj.robot_nonhand_contact_impulses_ns)
        if any(not isinstance(v, Mapping) or set(v) != set(PLAYERS) for v in maps):
            return False
        if (not isinstance(obj.object_support_impulses_ns, Mapping)
                or not set(obj.object_support_impulses_ns).issubset(self.inventory-{name})):
            return False
        values = [obj.deck_support_impulse_ns, obj.other_static_support_impulse_ns,
                  obj.maximum_penetration_m, *obj.robot_contact_impulses_ns.values(),
                  *obj.robot_nonhand_contact_impulses_ns.values(), *obj.object_support_impulses_ns.values()]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            return False
        return all(obj.robot_nonhand_contact_impulses_ns[p] <= obj.robot_contact_impulses_ns[p]+1.e-12
                   for p in PLAYERS)

    def _frame_valid(self, frame):
        if (not isinstance(frame, ExpansionNativeFrame) or frame.source != 'native_solver_substeps'
                or frame.contacts_complete is not True or type(frame.substep_index) is not int
                or frame.substep_index < 0 or type(frame.simulation_time_s) not in (int, float)
                or not math.isfinite(frame.simulation_time_s) or frame.simulation_time_s < 0
                or type(frame.dt_s) not in (int, float) or not math.isfinite(frame.dt_s)
                or not 0 < frame.dt_s <= self.thresholds.maximum_sample_gap_s
                or not isinstance(frame.objects, Mapping) or set(frame.objects) != self.inventory
                or any(not self._valid_object(name, obj) for name, obj in frame.objects.items())):
            return False
        old = self._last_frame
        return (old is None or (frame.substep_index == old.substep_index+1
                and abs(frame.dt_s-old.dt_s) <= 1.e-12
                and abs(frame.simulation_time_s-old.simulation_time_s-frame.dt_s) <= 1.e-9))

    def _bounds(self, name, obj):
        rotation = _rotation(obj.quaternion_wxyz)
        dimensions = self._specs[name]['dimensions_m']
        half = [sum(abs(rotation[i][j])*dimensions[j]/2 for j in range(3)) for i in range(3)]
        return ([obj.position_m[i]-half[i] for i in range(3)],
                [obj.position_m[i]+half[i] for i in range(3)])

    def _on_deck(self, name, obj):
        low, high = self._bounds(name, obj)
        board, tolerance = self.design['board'], self.thresholds.bottom_tolerance_m
        bounds = board['deck_bounds_xy_m']
        return (all(low[i] >= bounds[0][i]-tolerance and high[i] <= bounds[1][i]+tolerance for i in (0, 1))
                and abs(low[2]-board['top_z_m']) <= tolerance)

    def _supported(self, name, frame, allowed=(), *, off_deck=False, visiting=()):
        if name in visiting:
            return False
        obj = frame.objects[name]
        minimum = max(self.thresholds.minimum_contact_impulse_ns,
                      self._masses[name]*9.81*frame.dt_s*self.thresholds.minimum_support_weight_fraction)
        if obj.deck_support_impulse_ns >= minimum and self._on_deck(name, obj):
            return True
        bounds = self.design['board']['deck_bounds_xy_m']
        outside = not all(bounds[0][i] <= obj.position_m[i] <= bounds[1][i] for i in (0, 1))
        below_deck = self._bounds(name, obj)[1][2] < self.design['board']['top_z_m']-self.thresholds.bottom_tolerance_m
        if off_deck and (outside or below_deck) and obj.other_static_support_impulse_ns >= minimum:
            return True
        low, high = self._bounds(name, obj)
        for other, impulse in obj.object_support_impulses_ns.items():
            if other not in allowed or impulse < minimum:
                continue
            support = frame.objects[other]
            below, above = self._bounds(other, support)
            if (support.position_m[2] >= obj.position_m[2]
                    or abs(low[2]-above[2]) > self.thresholds.bottom_tolerance_m
                    or any(min(high[i], above[i])-max(low[i], below[i]) <= 0 for i in (0, 1))):
                continue
            if self._supported(other, frame, allowed, off_deck=off_deck, visiting=(*visiting, name)):
                return True
        return False

    def _released_stable(self, name, frame, allowed=(), *, off_deck=False):
        obj, limits = frame.objects[name], self.thresholds
        return (sum(obj.robot_contact_impulses_ns.values()) <= limits.minimum_contact_impulse_ns
                and _norm(obj.linear_velocity_m_s) <= limits.maximum_linear_speed_m_s
                and _norm(obj.angular_velocity_rad_s) <= limits.maximum_angular_speed_rad_s
                and self._supported(name, frame, allowed, off_deck=off_deck))

    def _pose_matches(self, name, obj, target):
        return (_distance(obj.position_m, target[0]) <= self.thresholds.position_tolerance_m
                and _angle(obj.quaternion_wxyz, target[1]) <= self.thresholds.maximum_angle_rad)

    def _upright(self, obj, *, flipped=False):
        cosine = _rotation(obj.quaternion_wxyz)[2][2]*(-1 if flipped else 1)
        return cosine >= math.cos(self.thresholds.maximum_angle_rad)

    def _poly_fits(self, name, obj, cells):
        """Whole real box union projects only into the intended occupied cells."""
        board = self.design['board']
        pitch = board['pitch_m']
        origin = [board['center_xy_m'][i]-board[['columns', 'rows'][i]]*pitch/2 for i in (0, 1)]
        rotation = _rotation(obj.quaternion_wxyz)
        if not self._upright(obj):
            return False
        for box in self._specs[name]['collision_boxes']:
            corners = []
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        local = [box['position_m'][i]+sign*box['dimensions_m'][i]/2
                                 for i, sign in enumerate((x, y, z))]
                        corners.append(tuple(obj.position_m[i]+sum(rotation[i][j]*local[j] for j in range(3)) for i in (0, 1)))
            polygon = _hull(corners)
            covered = 0.
            for q, r in cells:
                low = [origin[0]+q*pitch, origin[1]+r*pitch]
                covered += _area(_clip_rectangle(polygon, low, [v+pitch for v in low]))
            if _area(polygon)-covered > 1.e-10:
                return False
        return True

    def _memory_slot(self, name, obj, *, flipped):
        initial = self._specs[name]['initial_position_m']
        return (self._upright(obj, flipped=flipped)
                and _distance(obj.position_m[:2], initial[:2]) <= self.thresholds.position_tolerance_m
                and self._on_deck(name, obj))

    def _target(self, intent):
        name = intent.piece_ids[0]
        board, spec = self.design['board'], self._specs[name]
        z = board['top_z_m']+spec['dimensions_m'][2]/2
        params = intent.parameters
        if intent.type == 'place_piece':
            cells = self.rules.placement_cells(name, params['anchor'], params['rotation_deg'])
            center = [(min(c[i] for c in cells)+max(c[i] for c in cells))/2 for i in (0, 1)]
            xy = [board['center_xy_m'][i]+(center[i]-(board[['columns', 'rows'][i]]-1)/2)*board['pitch_m']
                  for i in (0, 1)]
            return ((*xy, z), _yaw(params['rotation_deg']))
        if intent.type == 'move_pawn':
            xy = [board['center_xy_m'][i]+(params['destination'][i]-2)*board['pitch_m'] for i in (0, 1)]
            return ((*xy, z), (1., 0., 0., 0.))
        if intent.type == 'place_barrier':
            xy = [board['center_xy_m'][i]+(params['anchor'][i]-1.5)*board['pitch_m'] for i in (0, 1)]
            return ((*xy, z), _yaw(0 if params['orientation'] == 'horizontal' else 90))
        initial = self._specs[name]['initial_position_m']
        return ((*initial[:2], z), (0., 1., 0., 0.))

    def _collection_ok(self, name, obj, color):
        low, high = self._bounds(name, obj)
        zone = self.design['collection_zones'][color]
        return (self._upright(obj, flipped=True)
                and all(low[i] >= zone['center_xy_m'][i]-zone['size_xy_m'][i]/2
                        and high[i] <= zone['center_xy_m'][i]+zone['size_xy_m'][i]/2 for i in (0, 1)))

    def _protected(self, frame, allowed):
        collected = self.rules._collected if self.task_id == 'memory_pairs' else {}
        for name in self.inventory-set(allowed):
            if name in collected:
                color = collected[name]
                group = {other for other, owner in collected.items() if owner == color}
                if not self._collection_ok(name, frame.objects[name], color) or not self._released_stable(name, frame, group):
                    return False
            elif not self._pose_matches(name, frame.objects[name], self._expected[name]) or not self._released_stable(name, frame):
                return False
            if (self.task_id == 'polyomino_territory' and name in self.rules.placed
                    and not self._poly_fits(name, frame.objects[name], self.rules.placed[name]['cells'])):
                return False
        return True

    def _reject(self, reason, *, recovery=False):
        self._reset_dwell()
        self.last_result = ExpansionResult(False, reason, recovery_required=recovery)
        return self.last_result

    def _track_manipulation(self, frame, previous):
        if not self.pending:
            return
        player, threshold = self.pending.player_id, self.thresholds.minimum_contact_impulse_ns
        for name in self.pending.piece_ids:
            obj = frame.objects[name]
            hand = obj.robot_contact_impulses_ns[player]-obj.robot_nonhand_contact_impulses_ns[player]
            if hand > threshold:
                self._contact_seen.add(name)
                # Current impulses cover the interval ending at this sample.
                # Accumulate net pose change across small native steps within
                # this hand-contact segment, never historical non-hand motion
                # or the travelled length of repeated subthreshold jitter.
                origin = self._hand_origins.setdefault(name, previous.objects[name])
                if (_distance(obj.position_m, origin.position_m) >= self.thresholds.minimum_operation_displacement_m
                        or _angle(obj.quaternion_wxyz, origin.quaternion_wxyz) >= self.thresholds.minimum_operation_rotation_rad):
                    self._motion_seen.add(name)
            else:
                self._hand_origins.pop(name, None)

    def _reveals(self, frame):
        visible = {name for name, obj in frame.objects.items()
                   if -_rotation(obj.quaternion_wxyz)[2][2] > self.thresholds.minimum_reveal_upward_normal_component
                   and name not in self.rules._collected}
        unexpected = False
        for name in sorted(visible-self._face_up):
            expected = bool(name in self.rules._exposed or (self.pending and self.pending.type == 'reveal_card'
                            and name in self.pending.piece_ids and name in self._contact_seen))
            self._event('card_revealed', frame, self.current_player, card_id=name,
                        symbol=self.rules._private_pairs[name], expected=expected)
            if expected and self.pending and self.pending.type == 'reveal_card' and name in self.pending.piece_ids:
                self._attempt_revealed = True
            unexpected |= not expected
        self._face_up = visible
        if unexpected:
            self.interrupt('rule_violation')
            self.last_result = ExpansionResult(False, 'unexpected_card_revealed')
        return unexpected

    def _settle(self, frame, names):
        if self._stable_start is None:
            self._stable_start = frame.simulation_time_s
            self._stable_positions = {name: frame.objects[name].position_m for name in names}
        elif any(_distance(frame.objects[name].position_m, pos) > self.thresholds.maximum_drift_m
                 for name, pos in self._stable_positions.items()):
            self._reset_dwell()
            return self._reject('settlement_drift')
        self._dwell = frame.simulation_time_s-self._stable_start
        if self._dwell+1.e-12 < self.thresholds.dwell_s:
            self.last_result = ExpansionResult(False, 'settling', self._dwell)
            return self.last_result
        return self._commit(frame)

    def observe(self, frame):
        if self.terminal:
            return ExpansionResult(False, 'match_terminal')
        if not self._frame_valid(frame):
            self.interrupt('missing_physical_evidence')
            return self.last_result
        previous = self._last_frame
        self._last_frame = deepcopy(frame)
        if any(obj.maximum_penetration_m > self.thresholds.maximum_penetration_m for obj in frame.objects.values()):
            self.interrupt('physical_failure')
            return self.last_result
        if previous is None:
            if any(not self._pose_matches(name, obj, self._expected[name]) for name, obj in frame.objects.items()):
                self.interrupt('missing_physical_evidence')
                return self.last_result
            return self._reject('native_baseline_received')
        self._track_manipulation(frame, previous)
        if self.task_id == 'memory_pairs' and self._reveals(frame):
            return self.last_result
        if self.task_id == 'disk_curling':
            return self._observe_disk(frame, previous)
        allowed = self.pending.piece_ids if self.pending else ()
        if not self._protected(frame, allowed):
            return self._reject('protected_object_needs_recovery', recovery=True)
        if not self.pending:
            return self._reject('awaiting_intent')
        intent = self.pending
        if intent.type == 'pass':
            return self._commit(frame)
        if not set(intent.piece_ids).issubset(self._contact_seen & self._motion_seen):
            return self._reject('actual_own_hand_manipulation_required')
        if self.task_id == 'memory_pairs' and intent.type in ('collect_pair', 'flip_back'):
            color = COLORS[PLAYERS.index(intent.player_id)]
            group = set(intent.piece_ids) | {n for n, owner in self.rules._collected.items() if owner == color}
            for name in intent.piece_ids:
                obj = frame.objects[name]
                if intent.type == 'collect_pair':
                    valid = self._collection_ok(name, obj, color) and self._released_stable(name, frame, group)
                else:
                    valid = self._memory_slot(name, obj, flipped=False) and self._released_stable(name, frame)
                if not valid:
                    return self._reject('pair_placement_or_release_required')
        else:
            name = intent.piece_ids[0]
            obj = frame.objects[name]
            if intent.type == 'place_piece':
                cells = self.rules.placement_cells(name, intent.parameters['anchor'], intent.parameters['rotation_deg'])
                pose_ok = self._poly_fits(name, obj, cells)
            elif intent.type == 'reveal_card':
                pose_ok = self._memory_slot(name, obj, flipped=True)
            elif intent.type == 'move_pawn':
                pose_ok = self._upright(obj) and _distance(obj.position_m, self._target(intent)[0]) <= self.thresholds.position_tolerance_m
            else:
                target = self._target(intent)
                opposite = tuple([-target[1][3], 0., 0., target[1][0]])
                pose_ok = self._pose_matches(name, obj, target) or self._pose_matches(name, obj, (target[0], opposite))
            if not pose_ok or not self._released_stable(name, frame):
                return self._reject('placement_or_release_required')
        return self._settle(frame, self.inventory)

    def _observe_disk(self, frame, previous):
        in_play = set(self.rules.played)
        if not self.pending:
            if any(sum(p.robot_contact_impulses_ns.values()) > self.thresholds.minimum_contact_impulse_ns for p in frame.objects.values()):
                self.interrupt('shot_rule_violation')
                return self.last_result
            if not self._protected(frame, in_play):
                return self._reject('protected_object_needs_recovery', recovery=True)
            return self._reject('awaiting_intent')
        intent, shot = self.pending, self._shot
        name, player = intent.piece_ids[0], intent.player_id
        obj, limits = frame.objects[name], self.thresholds
        color = COLORS[PLAYERS.index(player)]
        other = PLAYERS[1-PLAYERS.index(player)]
        epsilon = limits.minimum_contact_impulse_ns
        if (obj.robot_nonhand_contact_impulses_ns[player] > epsilon
                or obj.robot_contact_impulses_ns[other] > epsilon
                or any(sum(p.robot_contact_impulses_ns.values()) > epsilon
                       for key, p in frame.objects.items() if key != name)):
            self.interrupt('shot_rule_violation')
            return self.last_result
        if not self._protected(frame, in_play | {name}):
            self.interrupt('shot_rule_violation')
            return self.last_result
        board = self.design['board']
        direction = 1 if color == 'black' else -1
        region = board['launch_region_bounds_xy_m'][color]
        line = board['launch_lines_y_m'][color]
        contact = sum(obj.robot_contact_impulses_ns.values()) > epsilon
        if not shot['started']:
            low, high = self._bounds(name, obj)
            if not all(low[i] >= region[0][i] and high[i] <= region[1][i] for i in (0, 1)):
                self.interrupt('shot_rule_violation')
                return self.last_result
            if (contact and abs(_rotation(obj.quaternion_wxyz)[2][2]) >= math.cos(limits.maximum_angle_rad)
                    and self._supported(name, frame)
                    and direction*obj.linear_velocity_m_s[1] >= limits.disk_propulsion_speed_m_s):
                shot['started'], shot['launch'] = True, list(previous.objects[name].position_m[:2])
            else:
                return self._reject('preparing_disk')
        if contact:
            if shot['released'] or direction*(obj.position_m[1]-line) > 1.e-12:
                self.interrupt('shot_rule_violation')
                return self.last_result
        elif not shot['released']:
            release = previous.objects[name].position_m[:2]
            if direction*(release[1]-line) > 1.e-12:
                self.interrupt('shot_rule_violation')
                return self.last_result
            shot['released'], shot['release'] = True, list(release)
        if shot['released'] and direction*(obj.position_m[1]-line) >= 0:
            shot['crossed'] = True
        if not shot['released'] or not shot['crossed']:
            return self._reject('free_launch_line_crossing_required')
        active = in_play | {name}
        for identifier in active:
            on_deck = self._supported(identifier, frame, active)
            axis_flat = abs(_rotation(frame.objects[identifier].quaternion_wxyz)[2][2]) >= math.cos(limits.maximum_angle_rad)
            if (on_deck and not axis_flat) or not self._released_stable(identifier, frame, active, off_deck=True):
                return self._reject('all_played_disks_must_settle')
        return self._settle(frame, self.inventory)

    def _commit(self, frame):
        intent = self.pending
        color = COLORS[PLAYERS.index(intent.player_id)]
        params, ids = intent.parameters, intent.piece_ids
        if intent.type == 'place_piece':
            target = self._target(intent)
            self.rules.place(color, ids[0], params['anchor'], params['rotation_deg'])
            self._expected[ids[0]] = (frame.objects[ids[0]].position_m, frame.objects[ids[0]].quaternion_wxyz)
        elif intent.type == 'pass':
            self.rules.pass_turn(color)
        elif intent.type == 'move_pawn':
            target = self._target(intent)
            self.rules.move_pawn(color, params['destination'])
            self._expected[ids[0]] = (frame.objects[ids[0]].position_m, frame.objects[ids[0]].quaternion_wxyz)
        elif intent.type == 'place_barrier':
            target = self._target(intent)
            self.rules.place_wall(color, ids[0], params['orientation'], params['anchor'])
            self._expected[ids[0]] = (frame.objects[ids[0]].position_m, frame.objects[ids[0]].quaternion_wxyz)
        elif intent.type == 'reveal_card':
            self.rules.reveal(color, ids[0])
            self._expected[ids[0]] = (frame.objects[ids[0]].position_m, frame.objects[ids[0]].quaternion_wxyz)
        elif intent.type == 'collect_pair':
            self.rules.collect_pair(color, ids)
        elif intent.type == 'flip_back':
            self.rules.return_pair(color, ids)
            for name in ids:
                self._expected[name] = (frame.objects[name].position_m, frame.objects[name].quaternion_wxyz)
        elif intent.type == 'shoot_disk':
            positions = {name: list(frame.objects[name].position_m[:2]) for name in {*self.rules.played, ids[0]}}
            on_deck = {name: self._supported(name, frame, set(positions)) for name in positions}
            self.rules.record_shot(color, ids[0], contact_episodes=1, complete_native_evidence=True,
                                   physics_valid=True, released=True, stable_duration_s=self._dwell,
                                   positions_xy_m=positions, launch_position_xy_m=self._shot['launch'],
                                   release_position_xy_m=self._shot['release'], crossed_line_after_release=True,
                                   other_disk_robot_contacts=0, reserve_inventory_preserved=True, on_deck=on_deck)
        dwell = self._dwell
        self.turn_index += 1
        self.commit_log.append({**asdict(intent), 'native_substep_index': frame.substep_index,
                                'simulation_time_s': frame.simulation_time_s, 'verified_dwell_s': dwell})
        self._event('operation_committed', frame, intent.player_id, operation=intent.type, piece_ids=list(ids))
        self.pending = None
        self._reset_attempt()
        self.last_result = ExpansionResult(True, 'committed', dwell, committed=True)
        return self.last_result
