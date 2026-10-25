"""CPU reference referee over a service-owned, normalized physics stream.

There is intentionally no simulator adapter here. Setting the constructor's
trusted_normalized_input flag is an integration assertion, not authentication.
Never expose it or ``observe`` to an agent-controlled JSON endpoint.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import math

from .rules import OBSERVATION_SCHEMA, SIDES
from .task import task_configuration

TERMINAL = frozenset(('won', 'draw', 'abort', 'incomplete'))
OBSERVATION_KEYS = frozenset(('schema', 'stream_id', 'episode_id', 'reset_id',
                             'configuration_sha256', 'tick', 'time_s', 'blocks', 'contacts', 'executed_sides'))
BLOCK_KEYS = frozenset(('center_h', 'quaternion_wxyz', 'linear_velocity_h_s',
                       'angular_velocity_rad_s', 'hand_distance_h',
                       'interval_max_linear_speed_h_s', 'interval_max_angular_speed_rad_s',
                       'interval_min_center_z_h', 'interval_max_tilt_rad'))
CONTACT_KEYS = frozenset(('a', 'b', 'normal_z_on_a', 'normal_impulse_n_s',
                         'point_h', 'separation_h', 'supported_interval_s', 'minimum_normal_force_n'))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric value, not a boolean')
    return float(value)


def _vector(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('Invalid vector length')
    return [_number(x) for x in value]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _angle(a, b):
    # q and -q represent the same orientation.
    return 2 * math.acos(min(1., max(0., abs(sum(x * y for x, y in zip(a, b))))))


def _rotation(q):
    w, x, y, z = q
    return ((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))


def _side(actor):
    return actor.split('/', 1)[1] if actor.startswith(('hand/', 'robot/')) else None


class TowerExtractionGame:
    """One reference episode. Commands express intent; only observations score.

    Reset and each handoff require a fresh, continuous supported/released stable
    interval. Normalized observation ticks begin at tick 0 and time 0. A future
    adapter must rebase its engine clock and retain all normalized ticks. This
    CPU contract does not establish coverage of native physics substeps.
    """

    def __init__(self, configuration, *, trusted_normalized_input=False):
        if type(trusted_normalized_input) is not bool:
            raise ValueError('trusted_normalized_input must be a constructor boolean')
        if not isinstance(configuration, dict):
            raise ValueError('Expected task_configuration output')
        try:
            r = configuration['rules']
            canonical = task_configuration(configuration['geometry'], seed=configuration['seed'],
                max_turns_per_player=r['max_turns_per_player'],
                protected_bottom_layers=r['protected_bottom_layers'], thresholds=r['thresholds'],
                collection_trays=configuration.get('collection_trays'))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid task configuration') from exc
        if configuration != canonical:
            raise ValueError('Configuration differs from canonical task_configuration output')
        self.config = deepcopy(canonical)
        self.trusted = trusted_normalized_input
        self.thresholds = self.config['rules']['thresholds']
        self.status = 'reset_pending'
        self.current_side = self.config['first_side']
        self.selected = None
        self.selection_locked = False
        self.causal_actor = None
        self.turns = dict.fromkeys(SIDES, 0)
        self.collected = {}
        self.winner = self.loser = self.reason = None
        self._observation = self._previous = None
        self._stable_since = None
        self._exhausted_since = None

    def _end(self, status, reason, loser=None):
        self.status, self.reason, self.loser = status, reason, loser
        self.winner = SIDES[1 - SIDES.index(loser)] if loser else None
        self._stable_since = self._exhausted_since = None
        return self.snapshot()

    def finish(self, reason):
        """Host termination only; budgets/timeouts never fabricate a loser."""
        if reason not in ('timeout', 'budget', 'reset', 'evidence_missing', 'adapter_failure'):
            raise ValueError('Unknown host termination reason')
        if self.status in TERMINAL:
            return self.snapshot()
        return self._end('incomplete' if reason in ('timeout', 'budget') else 'abort', reason)

    def snapshot(self):
        """Public rule state, excluding privileged poses/contact evidence."""
        return {'status': self.status, 'reason': self.reason, 'current_side': self.current_side,
            'selected_block': self.selected, 'selection_locked': self.selection_locked,
            'completed_turns': dict(self.turns), 'collected': deepcopy(self.collected),
            'legal_blocks': self.legal_blocks() if self.status not in TERMINAL else [],
            'winner': self.winner, 'loser': self.loser, 'native_accepted': False}

    def _in_original_slot(self, bid):
        actual, initial = self._observation['blocks'][bid], self.config['blocks'][bid]
        return (_norm([a-b for a, b in zip(actual['center_h'], initial['center_h'])])
                <= self.thresholds['slot_position_tolerance_h']
                and _angle(actual['quaternion_wxyz'], initial['quaternion_wxyz'])
                <= self.thresholds['slot_rotation_tolerance_rad'])

    def legal_blocks(self):
        if self._observation is None or self.status == 'reset_pending' or self.status in TERMINAL:
            return []
        layers = {}
        for bid, initial in self.config['blocks'].items():
            layers.setdefault(initial['layer'], []).append(bid)
        complete = [layer for layer, ids in layers.items()
                    if all(bid not in self.collected and self._in_original_slot(bid) for bid in ids)]
        if not complete:
            return []
        highest = max(complete)
        bottom = self.config['rules']['protected_bottom_layers']
        return [bid for bid, initial in self.config['blocks'].items()
                if bottom <= initial['layer'] < highest and bid not in self.collected]

    def command(self, side, action, block_id=None):
        """Validate an intent. This neither moves bodies nor proves execution."""
        def reject(reason):
            return {'accepted': False, 'reason': reason, 'state': self.snapshot()}
        if side not in SIDES:
            return reject('unknown_side')
        if self.status in TERMINAL or self.status == 'reset_pending':
            return reject('episode_not_ready')
        if side != self.current_side:
            return reject('not_current_player')
        if action not in ('select', 'cancel', 'push', 'pull', 'regrasp', 'release', 'wait'):
            return reject('unknown_action')
        if block_id is not None and (not isinstance(block_id, str) or block_id not in self.config['blocks']):
            return reject('unknown_block')
        if action == 'select':
            if block_id is None:
                return reject('select_one_block_first')
            if block_id == self.selected:
                return {'accepted': True, 'state': self.snapshot()}
            if self.selection_locked:
                return reject('selection_locked_after_contact')
            if block_id not in self.legal_blocks():
                return reject('protected_or_unavailable_block')
            self.selected = block_id
            self.status = 'active'
            self._stable_since = None
        elif action == 'cancel':
            if self.selection_locked:
                return reject('selection_locked_after_contact')
            self.selected = None
            self.status = 'await_intent'
            self._stable_since = None
        elif action != 'wait':
            if self.selected is None:
                return reject('select_one_block_first')
            if block_id is not None and block_id != self.selected:
                return reject('different_block_intent')
        return {'accepted': True, 'state': self.snapshot()}

    def _validate_observation(self, value):
        if not isinstance(value, dict) or set(value) != OBSERVATION_KEYS:
            raise ValueError('Observation must have exactly the documented fields')
        if value['schema'] != OBSERVATION_SCHEMA:
            raise ValueError('Unknown observation schema')
        if value['configuration_sha256'] != self.config['configuration_sha256']:
            raise ValueError('Observation configuration_sha256 mismatch')
        for name in ('stream_id', 'episode_id', 'reset_id'):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError('Missing stream identity')
        tick, time = value['tick'], _number(value['time_s'])
        if type(tick) is not int or tick < 0 or time < 0:
            raise ValueError('Invalid normalized observation tick/time')
        prior = self._observation
        dt = 0.
        if prior is None:
            if tick != 0 or time != 0:
                raise ValueError('Stream must start at tick/time zero')
        else:
            if any(value[name] != prior[name] for name in ('stream_id', 'episode_id', 'reset_id')):
                raise ValueError('unannounced_reset_or_stream_change')
            dt = time - prior['time_s']
            if tick != prior['tick'] + 1 or not 0 < dt <= self.thresholds['max_observation_gap_s'] + 1.e-12:
                raise ValueError('Missing, duplicate or nonmonotonic normalized observation ticks')
        blocks = value['blocks']
        if not isinstance(blocks, dict) or set(blocks) != set(self.config['blocks']):
            raise ValueError('Complete body inventory is required at every tick')
        for bid, block in blocks.items():
            if not isinstance(block, dict) or set(block) != BLOCK_KEYS:
                raise ValueError('Invalid body fields; no success/stability claims accepted')
            _vector(block['center_h'], 3)
            q = _vector(block['quaternion_wxyz'], 4)
            if abs(_norm(q) - 1.) > 1.e-8:
                raise ValueError('Quaternion must be normalized')
            _vector(block['linear_velocity_h_s'], 3)
            _vector(block['angular_velocity_rad_s'], 3)
            for field, instantaneous in (('interval_max_linear_speed_h_s', 'linear_velocity_h_s'),
                                         ('interval_max_angular_speed_rad_s', 'angular_velocity_rad_s')):
                if _number(block[field]) < _norm(block[instantaneous]) - 1.e-10 or block[field] < 0:
                    raise ValueError('Interval speed maximum cannot be below the endpoint speed')
            tilt = math.acos(max(-1., min(1., _rotation(q)[2][2])))
            if _number(block['interval_min_center_z_h']) > block['center_h'][2] + 1.e-10:
                raise ValueError('Interval minimum height is above the endpoint height')
            if not tilt - 1.e-10 <= _number(block['interval_max_tilt_rad']) <= math.pi:
                raise ValueError('Invalid interval maximum tilt')
            if prior is not None:
                previous = prior['blocks'][bid]
                required_speed = _norm([a-b for a, b in zip(block['center_h'], previous['center_h'])]) / dt
                required_angular = _angle(q, previous['quaternion_wxyz']) / dt
                if (block['interval_max_linear_speed_h_s'] + 1.e-8 < required_speed
                        or block['interval_max_angular_speed_rad_s'] + 1.e-8 < required_angular):
                    raise ValueError('Interval speed maximum contradicts observed endpoint motion')
                previous_tilt = math.acos(max(-1., min(1., _rotation(previous['quaternion_wxyz'])[2][2])))
                if (block['interval_min_center_z_h'] > previous['center_h'][2] + 1.e-10
                        or block['interval_max_tilt_rad'] < previous_tilt - 1.e-10):
                    raise ValueError('Interval extrema exclude the preceding endpoint')
                height_path = previous['center_h'][2] + block['center_h'][2] - 2 * block['interval_min_center_z_h']
                tilt_path = 2 * block['interval_max_tilt_rad'] - previous_tilt - tilt
                if (height_path > block['interval_max_linear_speed_h_s'] * dt + 1.e-8
                        or tilt_path > block['interval_max_angular_speed_rad_s'] * dt + 1.e-8):
                    raise ValueError('Interval extrema contradict the interval speed bounds')
            elif (abs(block['interval_min_center_z_h'] - block['center_h'][2]) > 1.e-10
                  or abs(block['interval_max_tilt_rad'] - tilt) > 1.e-10):
                raise ValueError('Initial extrema must describe the initial sample')
            if not isinstance(block['hand_distance_h'], dict) or set(block['hand_distance_h']) != set(SIDES):
                raise ValueError('Both hands require measured surface clearance')
            for distance in block['hand_distance_h'].values():
                _number(distance)
        actors = set(blocks) | {'table', 'floor', 'hand/front', 'hand/rear', 'robot/front', 'robot/rear'}
        for tray in self.config.get('collection_trays', {}).values():
            actors.add(tray['actor_id'])
            actors.update(tray['wall_actor_ids'])
        if not isinstance(value['contacts'], list):
            raise ValueError('Contacts must contain complete native manifolds')
        for contact in value['contacts']:
            if not isinstance(contact, dict) or set(contact) != CONTACT_KEYS:
                raise ValueError('Invalid contact fields')
            a, b = contact['a'], contact['b']
            if not isinstance(a, str) or not isinstance(b, str) or a not in actors or b not in actors or a == b:
                raise ValueError('Unknown contact actor')
            if not -1 <= _number(contact['normal_z_on_a']) <= 1 or _number(contact['normal_impulse_n_s']) < 0:
                raise ValueError('Invalid native contact normal/impulse')
            _vector(contact['point_h'], 3)
            _number(contact['separation_h'])
            coverage = _number(contact['supported_interval_s'])
            force = _number(contact['minimum_normal_force_n'])
            if not 0 <= coverage <= dt + 1.e-12 or force < 0:
                raise ValueError('Invalid contact interval coverage or minimum force')
            if force * coverage > contact['normal_impulse_n_s'] + 1.e-10:
                raise ValueError('Contact impulse is below its minimum-force integral')
        sides = value['executed_sides']
        if not isinstance(sides, list) or any(side not in SIDES for side in sides) or len(set(sides)) != len(sides):
            raise ValueError('Invalid execution receipt sides')
        return deepcopy(value)

    def _contacts(self, bid):
        for c in self._observation['contacts']:
            if c['a'] == bid:
                yield c['b'], c['normal_z_on_a'], c['normal_impulse_n_s'], c
            elif c['b'] == bid:
                yield c['a'], -c['normal_z_on_a'], c['normal_impulse_n_s'], c

    def _on_face(self, bid, point, sign):
        block = self._observation['blocks'][bid]
        rotation = _rotation(block['quaternion_wxyz'])
        offset = [a-b for a, b in zip(point, block['center_h'])]
        local = [sum(rotation[j][i] * offset[j] for j in range(3)) for i in range(3)]
        half = self.config['blocks'][bid]['half_extents_h']
        tolerance = self.thresholds['contact_geometry_tolerance_h']
        return (all(abs(local[i]) <= half[i] + tolerance for i in (0, 1))
                and abs(local[2] - sign * half[2]) <= tolerance)

    def _on_table(self, point):
        g = self.config['geometry']; h = g['block_size_m'][2]
        xy = [point[i] * h + g['center_xy_m'][i] for i in (0, 1)]
        low, high = g['table_bounds_xy_m']
        return (abs(point[2]) <= self.thresholds['contact_geometry_tolerance_h']
                and all(low[i] <= xy[i] <= high[i] for i in (0, 1)))

    def _on_collection_floor(self, actor, point):
        g = self.config['geometry']; h = g['block_size_m'][2]
        world = [point[i] * h + origin for i, origin in enumerate([*g['center_xy_m'], g['table_top_z_m']])]
        for tray in self.config.get('collection_trays', {}).values():
            if tray['actor_id'] != actor:
                continue
            low, high = tray['interior_bounds_xy_m']
            return (abs(world[2] - tray['floor_z_m']) <= self.thresholds['contact_geometry_tolerance_h'] * h
                    and all(low[i] <= world[i] <= high[i] for i in (0, 1)))
        return False

    def _support(self, bid, allowed_blocks, visiting=None, *, surface_actor='table'):
        """Require a complete loaded support path to the declared floor surface."""
        visiting = set() if visiting is None else visiting
        if bid in visiting:
            return False
        visiting = visiting | {bid}
        blocks = self._observation['blocks']
        dt = 0. if self._previous is None else self._observation['time_s'] - self._previous['time_s']
        for other, normal, impulse, contact in self._contacts(bid):
            if (normal < self.thresholds['support_normal_z_min'] or impulse <= 0
                    or contact['minimum_normal_force_n'] <= 0
                    or contact['supported_interval_s'] < dt - 1.e-12
                    or abs(contact['separation_h']) > self.thresholds['contact_geometry_tolerance_h']
                    or not self._on_face(bid, contact['point_h'], -1)):
                continue
            if other == surface_actor:
                if (self._on_table(contact['point_h']) if surface_actor == 'table'
                        else self._on_collection_floor(surface_actor, contact['point_h'])):
                    return True
            if (other in allowed_blocks and blocks[other]['center_h'][2] < blocks[bid]['center_h'][2] - .25
                    and self._on_face(other, contact['point_h'], 1)
                    and self._support(other, allowed_blocks, visiting, surface_actor=surface_actor)):
                return True
        return False

    def _moving(self, bid):
        block = self._observation['blocks'][bid]
        linear = max(_norm(block['linear_velocity_h_s']), block['interval_max_linear_speed_h_s'])
        angular = max(_norm(block['angular_velocity_rad_s']), block['interval_max_angular_speed_rad_s'])
        if self._previous is not None:
            previous = self._previous['blocks'][bid]
            dt = self._observation['time_s'] - self._previous['time_s']
            linear = max(linear, _norm([a-b for a, b in zip(block['center_h'], previous['center_h'])]) / dt)
            angular = max(angular, _angle(block['quaternion_wxyz'], previous['quaternion_wxyz']) / dt)
        return linear > self.thresholds['stable_linear_speed_h_s'] or angular > self.thresholds['stable_angular_speed_rad_s']

    def _hands_clear(self):
        return (not any(_side(c['a']) or _side(c['b']) for c in self._observation['contacts'])
                and all(min(block['hand_distance_h'].values()) >= self.thresholds['release_clearance_h']
                        for block in self._observation['blocks'].values())
                and not self._observation['executed_sides'])

    def _corners(self, bid):
        block = self._observation['blocks'][bid]
        half = self.config['blocks'][bid]['half_extents_h']
        rotation = _rotation(block['quaternion_wxyz'])
        return [[block['center_h'][i] + sum(rotation[i][j] * half[j] * signs[j] for j in range(3))
                 for i in range(3)] for signs in itertools.product((-1., 1.), repeat=3)]

    def _inside(self, bid, slot):
        low, high = slot['bounds_h']
        return all(low[i] - 1.e-9 <= p[i] <= high[i] + 1.e-9 for p in self._corners(bid) for i in range(3))

    def _stable(self, *, reset=False):
        if not self._hands_clear():
            return False
        if any(c['separation_h'] < -self.thresholds['contact_geometry_tolerance_h']
               for c in self._observation['contacts']):
            return False
        tower = set(self.config['blocks']) - set(self.collected)
        if not reset and self.selected:
            tower.remove(self.selected)
        for bid in self.config['blocks']:
            surface_actor = 'table'
            if bid not in tower and self.config.get('collection_trays'):
                side = self.collected[bid]['side'] if bid in self.collected else self.current_side
                surface_actor = self.config['collection_trays'][side]['actor_id']
            if (self._moving(bid)
                    or not self._support(bid, tower if bid in tower else set(), surface_actor=surface_actor)):
                return False
            if reset and not self._in_original_slot(bid):
                return False
            if bid in self.collected:
                info = self.collected[bid]
                slot = next(s for s in self.config['collection_slots'][info['side']] if s['slot_id'] == info['slot_id'])
                if not self._inside(bid, slot):
                    return False
        return True

    def _collapsed(self):
        for bid, initial in self.config['blocks'].items():
            if bid in self.collected or (bid == self.selected and self.selection_locked):
                continue
            block = self._observation['blocks'][bid]
            drop = initial['center_h'][2] - block['interval_min_center_z_h']
            tilt = block['interval_max_tilt_rad']
            if drop >= self.thresholds['collapse_height_drop_h'] or tilt >= self.thresholds['collapse_tilt_rad']:
                return True
        return False

    def _dwell(self, valid, attribute='_stable_since'):
        now = self._observation['time_s']
        if not valid:
            setattr(self, attribute, None)
            return False
        if getattr(self, attribute) is None:
            setattr(self, attribute, now)
        return now - getattr(self, attribute) >= self.thresholds['stable_duration_s'] - 1.e-12

    def observe(self, observation):
        """Consume one service-owned normalized tick, or a synthetic CPU tick."""
        if self.status in TERMINAL:
            return self.snapshot()
        if not self.trusted:
            return self._end('abort', 'trusted_adapter_unavailable')
        try:
            validated = self._validate_observation(observation)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self._end('abort', 'invalid_evidence: ' + str(exc))
        self._previous, self._observation = self._observation, validated
        contacts = validated['contacts']
        touched_sides = {_side(c[key]) for c in contacts for key in ('a', 'b')} - {None}
        if self.status == 'reset_pending':
            if touched_sides or validated['executed_sides'] or self._collapsed():
                return self._end('abort', 'invalid_reset')
            if self._dwell(self._stable(reset=True)):
                self.status = 'await_intent'
                self._stable_since = None
            elif validated['time_s'] >= self.thresholds['reset_timeout_s']:
                return self._end('abort', 'invalid_reset_not_stable')
            return self.snapshot()
        other_side = SIDES[1 - SIDES.index(self.current_side)]
        if other_side in touched_sides or other_side in validated['executed_sides']:
            return self._end('abort', 'noncurrent_physical_interference')
        direct_blocks = {bid for bid in self.config['blocks']
                         if any(_side(other) == self.current_side for other, _, _, _ in self._contacts(bid))}
        became_protected = (self.selected in direct_blocks and not self.selection_locked
                            and self.selected not in self.legal_blocks())
        if direct_blocks - {self.selected}:
            return self._end('abort', 'undeclared_block_contact')
        if became_protected:
            return self._end('abort', 'protected_target_contact')
        # A first touch and collapse in the same interval have unknown ordering.
        # Only contact established on a previous tick can establish attribution.
        if self._collapsed():
            if self.causal_actor is None:
                return self._end('abort', 'abort_no_causal_evidence')
            return self._end('won', 'tower_collapse', self.current_side)
        if direct_blocks - set(self.collected):
            self.causal_actor = self.current_side
        if self.selected in direct_blocks:
            self.selection_locked = True
        if self.selected is not None:
            occupied = {info['slot_id'] for info in self.collected.values()}
            slot = next((s for s in self.config['collection_slots'][self.current_side]
                         if s['slot_id'] not in occupied and self._inside(self.selected, s)), None)
            valid = self.selection_locked and slot is not None and self._stable()
            if self._dwell(valid):
                self.collected[self.selected] = {'side': self.current_side, 'slot_id': slot['slot_id']}
                self.turns[self.current_side] += 1
                self.selected = None
                self.selection_locked = False
                self.causal_actor = None
                self._stable_since = self._exhausted_since = None
                limit = self.config['rules']['max_turns_per_player']
                if all(count == limit for count in self.turns.values()):
                    return self._end('draw', 'balanced_turn_limit')
                self.current_side = other_side
                self.status = 'await_intent'
            elif valid:
                self.status = 'settling'
            else:
                self.status = 'active'
        elif self._dwell(not self.legal_blocks() and self._stable(), '_exhausted_since'):
            return self._end('draw', 'legal_targets_exhausted')
        return self.snapshot()
