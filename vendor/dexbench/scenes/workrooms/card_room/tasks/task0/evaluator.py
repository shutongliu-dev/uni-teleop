"""Read-only placement score using canonical poses and complete native contacts.

The square-cell footprint and settlement tolerances follow the shared tabletop
verifier. Keeping the small square calculation here makes the frozen evaluator
depend only on its sibling frozen rules, never on mutable project source.
"""
from copy import deepcopy
import math

from .rules import (CANDIDATE_CELLS, PIECE_IDS, bind_episode_targets, public_rules,
                    select_target_cells, target_positions_m, thresholds, validate_target_cells)

VERSION = 'card_room_placement_evaluator_v1'
EPS = 1.e-9


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+' must be a finite number')
    return float(value)


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(name+' has the wrong dimension')
    return tuple(number(item, name) for item in value)


def norm(value):
    return math.sqrt(sum(item*item for item in value))


def distance(a, b):
    return norm(tuple(x-y for x, y in zip(a, b)))


def orientation_distance(a, b):
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b)))))


class PlacementEvaluator:
    def __init__(self, task_spec):
        self.task = deepcopy(task_spec)
        if self.task.get('schema_version') != 'card-room-placement-task-v1' or self.task.get('id') != 'task0':
            raise ValueError('Expected the independently versioned task0')
        self.c = self.task['evaluation']
        self.targets = tuple(tuple(cell) for cell in validate_target_cells(self.c['target_cells']))
        if self.task.get('public_rules') != public_rules(self.targets):
            raise ValueError('Public rules differ from the frozen evaluator')
        for key, expected in thresholds().items():
            if number(self.c.get(key), key) != expected:
                raise ValueError('Threshold differs from the versioned candidate: '+key)
        self.dt, self.control = self.c['physics_dt_s'], self.c['control_period_s']
        self.substeps = round(self.control/self.dt)
        self.pieces = tuple(self.c['piece_ids'])
        if self.pieces != PIECE_IDS:
            raise ValueError('Exactly the three declared black pieces are required')
        self.actors = self.c['actors']
        if (set(self.actors) != set(self.pieces) or len(set(self.actors.values())) != 3
                or any(not isinstance(path, str) or not path.startswith('/World/Objects/') for path in self.actors.values())):
            raise ValueError('Each piece needs its distinct native body actor')
        self.path_piece = {path: name for name, path in self.actors.items()}
        paths = self.c['board_contact_paths']
        if (not isinstance(paths, (list, tuple)) or not paths or len(set(paths)) != len(paths)
                or any(not isinstance(path, str) or not (path == '/World/Fixtures/board' or path.startswith('/World/Fixtures/board/')) for path in paths)):
            raise ValueError('Named native board contacts are required')
        self.board_paths = frozenset(paths)
        if self.c.get('robot_actor_prefix') != '/World/Robot/':
            raise ValueError('The native robot actor prefix must be explicit')
        self.geometry = self.c['geometry']
        self._validate_geometry()
        cells = [list(cell) for cell in self.targets]
        if cells != select_target_cells(self.task.get('episode_seed')):
            raise ValueError('Targets differ from the published episode seed')
        expected_goal = {'operation': 'place_three_pieces', 'piece_ids': list(PIECE_IDS),
            'frame_id': 'world_m_z_up', 'target_cells': cells,
            'candidate_cells': [list(cell) for cell in CANDIDATE_CELLS],
            'target_positions_m': target_positions_m(self.geometry, cells),
            'description': self.task['public_rules']['instruction_zh']}
        if any(self.task.get('public_goal', {}).get(key) != value for key, value in expected_goal.items()):
            raise ValueError('Public goal differs from the frozen geometry and evaluated targets')
        if set(self.c['initial_objects']) != set(self.pieces):
            raise ValueError('Every physical piece needs a frozen reset pose')
        self.expected = {name: self._state(row, velocities=False)
                         for name, row in self.c['initial_objects'].items()}
        # Reset is a supply state, never an already accomplished placement.
        for state in self.expected.values():
            if any(self._cell_rejection(state, (q, r)) is None for r in range(3) for q in range(3)):
                raise ValueError('Initial pieces must be on supply trays, outside board cells')
        self.reset()

    def _validate_geometry(self):
        g = self.geometry
        if g.get('game_id') != 'tic_tac_toe_3x3' or g.get('size') != 3 or g.get('shape') != 'square':
            raise ValueError('Task0 requires its 3x3 square board')
        vector(g['center_xy_m'], 2, 'board center')
        number(g['board_yaw_rad'], 'board yaw')
        number(g['table_top_z_m'], 'table height')
        for key in ('pitch_m', 'board_thickness_m', 'piece_radius_m', 'piece_base_height_m',
                    'handle_radius_m', 'handle_height_m'):
            if number(g[key], key) <= 0:
                raise ValueError('Geometry dimensions must be positive')
        if 2*g['piece_radius_m']+2*self.c['edge_clearance_m'] >= g['pitch_m']:
            raise ValueError('Piece footprint must fit its cell')

    @staticmethod
    def _state(row, *, velocities=True):
        q = vector(row['quaternion_wxyz'], 4, 'orientation')
        length = norm(q)
        if abs(length-1.) > 1.e-4:
            raise ValueError('Orientation must be a unit wxyz quaternion')
        return {'p': vector(row['position_m'], 3, 'position'),
                'q': tuple(v/length for v in q),
                'v': vector(row['linear_velocity_m_s'], 3, 'linear velocity') if velocities else (0.,)*3,
                'w': vector(row['angular_velocity_rad_s'], 3, 'angular velocity') if velocities else (0.,)*3}

    def reset(self):
        self.previous_time = self.native_step = self.native_time = self.previous = None
        self.stable_start = dict.fromkeys(self.pieces)
        self.stable_cell = dict.fromkeys(self.pieces)
        self.anchor = dict.fromkeys(self.pieces)
        self.dwell = dict.fromkeys(self.pieces, 0.)
        self.invalid = self.last_result = None
        self.success = False

    def _sampling(self, obs, time, first):
        contacts = obs['contacts']
        if contacts.get('source') != 'PhysX stage-bound rigid contact tensor; normal impulses' or contacts.get('errors') != []:
            raise ValueError('Native contact source or error status is invalid')
        s = contacts['sampling']
        if s.get('protocol') != 'native_rigid_contact_sampling_v1' or s.get('complete') is not True:
            raise ValueError('Complete native contact sampling is required')
        if number(s['physics_dt_s'], 'sampling dt') != self.dt:
            raise ValueError('Native physics dt differs from task')
        for key in ('sampled_substeps', 'native_start_step_index', 'native_end_step_index'):
            if type(s[key]) is not int or s[key] < 0:
                raise ValueError('Native counters must be nonnegative integers')
        start, end, ns, ne = [number(s[key], key) for key in
            ('start_sim_time_s', 'end_sim_time_s', 'native_start_time_s', 'native_end_time_s')]
        duration, count = (0., 0) if first else (self.control, self.substeps)
        if (min(start, end, ns, ne) < -EPS or s['sampled_substeps'] != count
                or s['native_end_step_index']-s['native_start_step_index'] != count
                or abs(end-time) > EPS or abs(end-start-duration) > EPS or abs(ne-ns-duration) > EPS
                or abs(number(obs['last_duration_s'], 'duration')-duration) > EPS):
            raise ValueError('Native contact coverage or clocks are inconsistent')
        if first:
            if abs(time) > EPS or contacts['rigid']:
                raise ValueError('First observation must be the unsampled reset baseline')
        elif (abs(start-self.previous_time) > EPS or s['native_start_step_index'] != self.native_step
                or abs(ns-self.native_time) > EPS):
            raise ValueError('Native contact windows must be consecutive')
        return start, s['native_end_step_index'], ne

    def _contacts(self, obs, time, first):
        start, step, native_time = self._sampling(obs, time, first)
        rows = obs['contacts']['rigid']
        if not isinstance(rows, list):
            raise ValueError('Native contact rows must be a list')
        support = {name: [0.]*self.substeps for name in self.pieces}
        robot = {name: [0.]*self.substeps for name in self.pieces}
        penetration = dict.fromkeys(self.pieces, 0.)
        seen = set()
        for row in rows:
            a, b = row['actor0'], row['actor1']
            if not isinstance(a, str) or not isinstance(b, str) or a == b:
                raise ValueError('Invalid native actor pair')
            stamp = number(row['time_s'], 'contact time')
            tick = round((stamp-start)/self.dt)
            if not 1 <= tick <= self.substeps or abs(stamp-start-tick*self.dt) > EPS:
                raise ValueError('Stale, future or off-grid contact row')
            key = (tick, *sorted((a, b)))
            if key in seen:
                raise ValueError('Duplicate native contact pair')
            seen.add(key)
            impulse = number(row['impulse_magnitude_sum_ns'], 'normal impulse')
            vec = vector(row['impulse_vector_ns'], 3, 'normal impulse vector')
            separation = number(row['min_separation_m'], 'contact separation')
            normal_error = number(row['normal_length_max_abs_error'], 'normal error')
            count, active = row['point_count'], row['nonzero_impulse_normal_count']
            if (type(count) is not int or count <= 0 or type(active) is not int or not 0 <= active <= count
                    or impulse < 0 or norm(vec) > impulse+EPS or (impulse > 0 and active == 0)
                    or not 0 <= normal_error <= 32*2**-23 or row.get('impulse_components') != 'normal_only'):
                raise ValueError('Invalid native normal-impulse evidence')
            for actor, other in ((a, b), (b, a)):
                name = self.path_piece.get(actor)
                if name is None:
                    continue
                penetration[name] = max(penetration[name], -separation)
                if other in self.board_paths:
                    support[name][tick-1] += impulse
                if other.startswith(self.c['robot_actor_prefix']):
                    robot[name][tick-1] += impulse
        return support, robot, penetration, step, native_time

    def _cell_rejection(self, state, cell):
        g, c = self.geometry, self.c
        w, x, y, z = state['q']
        tilt = math.acos(max(-1., min(1., 1-2*(x*x+y*y))))
        if tilt > c['maximum_tilt_rad']:
            return 'piece_not_upright'
        sine, cosine = math.sin(tilt), math.cos(tilt)
        radius = max(g['piece_radius_m']+g['piece_base_height_m']/2*sine,
                     g['handle_radius_m']+(g['piece_base_height_m']/2+g['handle_height_m'])*sine)
        dx, dy = state['p'][0]-g['center_xy_m'][0], state['p'][1]-g['center_xy_m'][1]
        cb, sb = math.cos(g['board_yaw_rad']), math.sin(g['board_yaw_rad'])
        local_x, local_y = cb*dx+sb*dy, -sb*dx+cb*dy
        clearance = g['pitch_m']/2-max(abs(local_x-g['pitch_m']*(cell[0]-1)),
                                      abs(local_y-g['pitch_m']*(cell[1]-1)))-radius
        if clearance < c['edge_clearance_m']:
            return 'piece_footprint_outside_target'
        bottom = state['p'][2]-g['piece_radius_m']*sine-g['piece_base_height_m']/2*cosine
        if abs(bottom-g['table_top_z_m']-g['board_thickness_m']) > c['bottom_z_tolerance_m']:
            return 'piece_not_at_board_height'
        return None

    def _clear_piece(self, name):
        self.stable_start[name] = self.stable_cell[name] = self.anchor[name] = None
        self.dwell[name] = 0.

    def _result(self, reason, reasons=None):
        count = sum(value+EPS >= self.c['dwell_s'] for value in self.dwell.values()) if not self.invalid else 0
        self.last_result = {'evaluator': VERSION, 'task_id': 'task0',
            'success': bool(self.success and not self.invalid), 'progress': count/3,
            'reason': reason, 'placed_count': count, 'required_count': 3,
            'piece_stable_s': dict(self.dwell), 'piece_reasons': dict(reasons or {}),
            'irrecoverable_reason': None,
            'threshold_status': 'engineering_candidates_not_native_accepted',
            'evidence_scope': self.task['public_rules']['evidence_scope']}
        return deepcopy(self.last_result)

    def update(self, observation):
        if self.invalid:
            return self._result('invalid_observation:'+self.invalid)
        try:
            time = number(observation['time_s'], 'observation time')
            if abs(number(observation['sim_time_s'], 'simulation time')-time) > EPS:
                raise ValueError('Observation clocks disagree')
            first = self.previous_time is None
            if not first and abs(time-self.previous_time-self.control) > EPS:
                raise ValueError('Object poses must arrive at every 20 ms control boundary')
            if observation.get('metrics', {}).get('state_finite') is not True or observation.get('terminated') is True:
                raise ValueError('Native state finiteness or continuation is not established')
            if set(observation['objects']) != set(self.pieces):
                raise ValueError('Observation must contain exactly the three physical pieces')
            states = {name: self._state(observation['objects'][name]) for name in self.pieces}
            support, robot, penetration, step, native_time = self._contacts(observation, time, first)
            if first:
                for name, state in states.items():
                    if (distance(state['p'], self.expected[name]['p']) > self.c['initial_position_tolerance_m']
                            or orientation_distance(state['q'], self.expected[name]['q']) > self.c['initial_orientation_tolerance_rad']):
                        raise ValueError('Initial pose differs from frozen supply reset')
                self.previous, self.previous_time = states, time
                self.native_step, self.native_time = step, native_time
                return self._result('reset_baseline_not_scored')
            cells = {name: next((cell for cell in self.targets if self._cell_rejection(state, cell) is None), None)
                     for name, state in states.items()}
            reasons = {}
            for name, state in states.items():
                cell, previous = cells[name], self.previous[name]
                reason = None
                if cell is None: reason = 'piece_outside_target_or_board_height'
                elif list(cells.values()).count(cell) != 1: reason = 'target_has_multiple_pieces'
                elif min(support[name]) <= self.c['minimum_support_impulse_ns']: reason = 'native_board_support_required_every_substep'
                elif max(robot[name]) > self.c['maximum_robot_contact_impulse_ns']: reason = 'robot_must_release_piece'
                elif penetration[name] > self.c['maximum_penetration_m']: reason = 'excessive_reported_penetration'
                elif max(norm(state['v']), distance(state['p'], previous['p'])/self.control) > self.c['maximum_linear_speed_m_s']:
                    reason = 'piece_still_translating'
                elif max(norm(state['w']), orientation_distance(state['q'], previous['q'])/self.control) > self.c['maximum_angular_speed_rad_s']:
                    reason = 'piece_still_rotating'
                if reason:
                    self._clear_piece(name)
                    reasons[name] = reason
                    continue
                if self.stable_cell[name] != cell or self.anchor[name] is None:
                    self.stable_start[name], self.stable_cell[name], self.anchor[name] = time, cell, state
                anchor = self.anchor[name]
                if (distance(state['p'], anchor['p']) > self.c['maximum_stable_position_range_m']
                        or orientation_distance(state['q'], anchor['q']) > self.c['maximum_stable_orientation_range_rad']):
                    self._clear_piece(name)
                    reasons[name] = 'piece_drifted_during_stability_window'
                    continue
                # A first accepted pose starts the dwell; no credit for the
                # unobserved object path immediately before that sample.
                self.dwell[name] = time-self.stable_start[name]
                reasons[name] = 'stable' if self.dwell[name]+EPS >= self.c['dwell_s'] else 'settling'
            self.success = all(value+EPS >= self.c['dwell_s'] for value in self.dwell.values())
            self.previous, self.previous_time = states, time
            self.native_step, self.native_time = step, native_time
            return self._result('success' if self.success else 'place_release_and_stabilize', reasons)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid, self.success = str(exc), False
            for name in self.pieces:
                self._clear_piece(name)
            return self._result('invalid_observation:'+self.invalid)

    def inspect_conditions(self, observation=None):
        """Read the latest result without consuming the contact receipt twice."""
        return deepcopy(self.last_result or {'success': False, 'reason': 'not_sampled'})


SequenceEvaluator = PlacementEvaluator
