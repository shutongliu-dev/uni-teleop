"""Read-only native substep observer and canonical remote-operation scorer.

Only the worker's private native hook supplies observations. Neither class
accepts an environment, writes a pose, or drives a joint. CPU fixtures establish
logic, never native robot qualification. Frozen deployment needs this file and
its sibling rules.py only.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import itertools
import json
import math

from .rules import BUTTON_IDS, LEGACY_RULES_VERSION, RULES_VERSION, button_sequence, public_rules, thresholds

VERSION = 'living_remote_operation_evaluator_v3'
STATE_SCHEMA = 'living_remote_native_state.v3'
LEGACY_VERSION = 'living_remote_operation_evaluator_v2'
LEGACY_STATE_SCHEMA = 'living_remote_native_state.v2'
EPS = 1.e-8


def number(value, label):
    if isinstance(value, bool):
        raise ValueError(label+' must be a number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(label+' must be finite')
    return value


def vector(value, size, label):
    if len(value) != size:
        raise ValueError(label+' has wrong dimension')
    return tuple(number(x, label) for x in value)


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def rotation(q):
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def rotate(r, v):
    return tuple(sum(a*b for a, b in zip(row, v)) for row in r)


def transpose(r):
    return tuple(zip(*r))


def angle(a, b):
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b)))))


def state(row):
    q = vector(row['quaternion_wxyz'], 4, 'quaternion')
    length = norm(q)
    if abs(length-1.) > 1.e-4:
        raise ValueError('Native quaternion must be unit wxyz')
    return {'position_m': vector(row['position_m'], 3, 'position'),
            'quaternion_wxyz': tuple(x/length for x in q),
            'linear_velocity_m_s': vector(row['linear_velocity_m_s'], 3, 'linear velocity'),
            'angular_velocity_rad_s': vector(row['angular_velocity_rad_s'], 3, 'angular velocity')}


def configuration(task):
    if (task.get('schema_version', task.get('schema')) != 'living-room-manipulation-task-v1'
            or task.get('id') != 'remote_operation'):
        raise ValueError('Expected the published remote_operation task')
    seed = task['episode_seed']
    c = deepcopy(task['evaluation'])
    legacy = c['schema'] == 'living_remote_evaluation.v2'
    if legacy:
        if c['button_sequence'] != button_sequence(seed):
            raise ValueError('Button sequence differs from episode')
    elif c['schema'] != 'living_remote_evaluation.v3' or number(c['required_on_s'], 'required on time') != 10.:
        raise ValueError('Television/configuration schema differs from episode')
    if task.get('public_rules') != public_rules(seed, version=LEGACY_RULES_VERSION if legacy else RULES_VERSION):
        raise ValueError('Public rules differ from frozen rules')
    if not legacy:
        rules_digest = hashlib.sha256(json.dumps(task['public_rules'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if c['public_rules_sha256'] != rules_digest:
            raise ValueError('Public rules hash differs from frozen configuration')
    goal = task.get('public_goal', {})
    if legacy and 'button_sequence' in goal and goal['button_sequence'] != c['button_sequence']:
        raise ValueError('Public goal differs from scored order')
    if not legacy:
        if 'button_sequence' in goal or 'button_sequence' in c:
            raise ValueError('Television goal must not score a legacy button sequence')
        if 'required_on_s' in goal and number(goal['required_on_s'], 'public required on time') != c['required_on_s']:
            raise ValueError('Public goal differs from television duration')
        expected_goal = {'operation': 'cycle_television_power', 'object_id': 'remote',
                         'controlled_object_id': 'television', 'power_button': 'power',
                         'initial_power_on': False, 'target_id': 'holder',
                         'holder_return_required': True}
        if any(key in goal and (goal[key] != value or type(goal[key]) is not type(value))
               for key, value in expected_goal.items()):
            raise ValueError('Public goal differs from television operation')
        if 'stable_duration_s' in goal and number(goal['stable_duration_s'], 'public stable duration') != c['stable_s']:
            raise ValueError('Public goal differs from holder stability duration')
    for key, value in thresholds().items():
        if number(c[key], key) != value:
            raise ValueError('Changed frozen threshold: '+key)
    dt = number(c['physics_dt_s'], 'physics dt')
    if not 0 < dt <= .002 or c['control_period_s'] != .02 or abs(.02/dt-round(.02/dt)) > EPS:
        raise ValueError('Invalid native/control sampling period')
    if set(c['actors']) != {'remote', 'power', 'confirm', 'holder'} or len(set(c['actors'].values())) != 4:
        raise ValueError('Distinct native actors are required')
    if any(not isinstance(v, str) or not v.startswith('/World/') for v in c['actors'].values()):
        raise ValueError('Native actor paths must be absolute')
    if c['robot_actor_prefix'] != '/World/Robot/' or c['hand_link_prefixes'] != ['right_']:
        raise ValueError('Exactly the front right SHARPA hand is scored')
    if c['finger_link_prefixes'] != ['right_thumb_', 'right_index_', 'right_middle_', 'right_ring_', 'right_pinky_']:
        raise ValueError('Finger contact must use the frozen SHARPA namespaces')
    g = c['geometry']
    expected = {'remote_local_bounds_m': [[-.0275, -.080, -.0104], [.0275, .080, .0135]],
        'holder_opening_xy_m': [.067, .172], 'holder_liner_z_m': .007,
        'table_top_z_m': .78, 'workspace_bounds_m': [[.05, -.7, .72], [1.25, .7, 1.95]]}
    if g != expected:
        raise ValueError('Geometry differs from the bound asset candidate')
    digest = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return c, digest


class TelevisionState:
    """One shared native-event state machine for observation, rendering and scoring."""

    def __init__(self, required_on_s, required_return_s):
        self.required_on_s = required_on_s
        self.required_return_s = required_return_s
        self.power_on = False
        self.on_since = None
        self.qualified_off_since = None
        self.pending = None
        self.completed_cycles = 0

    def event(self, event):
        if event['button'] != 'power':
            return
        if event['kind'] == 'pressed':
            if self.pending is not None:
                raise ValueError('Power press requires a preceding release')
            self.pending = event['handheld_valid']
            return
        if self.pending is None:
            raise ValueError('Power release requires a preceding press')
        valid = self.pending is True and event['handheld_valid']
        self.pending = None
        if not valid:
            return
        self.completed_cycles += 1
        t = event['time_s']
        if self.power_on:
            self.qualified_off_since = t if t-self.on_since+EPS >= self.required_on_s else None
            self.power_on, self.on_since = False, None
        else:
            self.power_on, self.on_since, self.qualified_off_since = True, t, None

    def state(self, time_s, released_stable_s=0.):
        duration = max(0., time_s-self.on_since) if self.power_on else 0.
        returned = (self.qualified_off_since is not None
                    and min(released_stable_s, time_s-self.qualified_off_since)+EPS >= self.required_return_s)
        phase = ('completed' if returned else 'await_return' if self.qualified_off_since is not None else 'await_power_off'
                 if self.power_on and duration+EPS >= self.required_on_s else
                 'holding_power_on' if self.power_on else 'await_power_on')
        return {'power_on': self.power_on, 'continuous_on_s': duration,
                'required_on_s': self.required_on_s, 'phase': phase}


class NativeTaskObserver:
    """Consume immutable native frames and preserve short physical button events."""

    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.legacy = self.c['schema'] == 'living_remote_evaluation.v2'
        self.state_schema = LEGACY_STATE_SCHEMA if self.legacy else STATE_SCHEMA
        self.reset()

    def reset(self):
        self.previous = self.previous_time = self.previous_native_time = self.previous_step = None
        self.events = []
        self.cycle = {n: {'armed': False, 'pressed': False, 'valid': False} for n in BUTTON_IDS}
        self.error = None
        self.terminal_failure = None
        self.current = None
        self.held_s = self.stable_s = 0.
        self.support_witness = self.stable_anchor = None
        self.television = None if self.legacy else TelevisionState(self.c['required_on_s'], self.c['stable_s'])

    def _hand(self, path, finger=False):
        return (path.startswith(self.c['robot_actor_prefix']) and
                path.rsplit('/', 1)[-1] not in ('right_hand_flange', 'right_hand_wrist') and
                path.rsplit('/', 1)[-1].startswith(tuple(self.c['finger_link_prefixes' if finger else 'hand_link_prefixes'])))

    def _corners(self, remote):
        bounds = self.c['geometry']['remote_local_bounds_m']
        r = rotation(remote['quaternion_wxyz'])
        return [tuple(a+b for a, b in zip(remote['position_m'], rotate(r, p)))
                for p in itertools.product(*zip(*bounds))]

    def _contacts(self, frame):
        actors = self.c['actors']
        own = set(actors.values())
        assembly = {actors[n] for n in ('remote', 'power', 'confirm')}
        hand = finger = False
        external = nonholder_external = holder_contact = holder_external = False
        keys = {n: False for n in BUTTON_IDS}
        touching = {n: False for n in BUTTON_IDS}
        seen = set()
        for row in frame['contacts']:
            a, b = row['actor0'], row['actor1']
            if not isinstance(a, str) or not isinstance(b, str) or a == b:
                raise ValueError('Invalid contact actor pair')
            pair = tuple(sorted((a, b)))
            if pair in seen:
                raise ValueError('Duplicate native contact pair')
            seen.add(pair)
            stamp = number(row['time_s'], 'contact time')
            if abs(stamp-number(frame['time_s'], 'frame time')) > EPS:
                raise ValueError('Contact belongs to another native step')
            count = row['point_count']
            impulse = number(row['impulse_magnitude_sum_ns'], 'contact impulse')
            if type(count) is not int or count <= 0 or impulse < 0:
                raise ValueError('Malformed native contact')
            if number(row['min_separation_m'], 'separation') < -self.c['maximum_penetration_m']:
                if a in own or b in own:
                    raise ValueError('Task object penetrates another body')
            positive = impulse > self.c['minimum_contact_impulse_ns']
            for actor, other in ((a, b), (b, a)):
                if actor == actors['remote'] and self._hand(other) and positive:
                    hand = True
                if actor in assembly:
                    finger = finger or self._hand(other)
                    if other not in assembly and not self._hand(other):
                        external = True
                        if other != actors['holder']:
                            nonholder_external = True
                    if actor == actors['remote'] and other == actors['holder'] and positive:
                        holder_contact = True
                if actor == actors['holder'] and other.startswith(self.c['robot_actor_prefix']):
                    holder_external = True
                for n in BUTTON_IDS:
                    if actor == actors[n] and other not in assembly:
                        touching[n] = True
                        if self._hand(other, finger=True) and positive:
                            keys[n] = True
        return hand, finger, external, nonholder_external, holder_contact, holder_external, keys, touching

    def _emit(self, name, kind, frame, q, valid):
        self.events.append({'id': len(self.events)+1, 'native_step': frame['native_step'],
            'time_s': float(frame['time_s']), 'button': name, 'kind': kind,
            'position_m': q, 'handheld_valid': bool(valid)})
        if self.television is not None:
            self.television.event(self.events[-1])

    def on_frame(self, frame):
        if self.error is not None:
            return
        try:
            self._frame(frame)
        except (KeyError, ValueError, TypeError, IndexError, OverflowError) as exc:
            self.error = str(exc)

    def _frame(self, frame):
        c = self.c
        t = number(frame['time_s'], 'frame time')
        nt = number(frame['native_time_s'], 'native time')
        step = frame['native_step']
        baseline = frame['baseline']
        if type(step) is not int or step < 0 or type(baseline) is not bool or t < 0 or nt < 0:
            raise ValueError('Invalid native frame counters')
        if abs(number(frame['physics_dt_s'], 'native dt')-c['physics_dt_s']) > EPS:
            raise ValueError('Native dt differs from configuration')
        first = self.previous is None
        if baseline != first:
            raise ValueError('Exactly one native baseline is required')
        if first:
            if abs(t) > EPS or frame['contacts']:
                raise ValueError('Baseline must precede physical stepping and contacts')
        elif (step != self.previous_step+1 or abs(t-self.previous_time-c['physics_dt_s']) > EPS
              or abs(nt-self.previous_native_time-c['physics_dt_s']) > 2.e-6):
            raise ValueError('Native frames must be consecutive')
        objects = {name: state(frame['objects'][name]) for name in c['actors']}
        remote, holder = objects['remote'], objects['holder']
        q = {n: number(frame['joints'][n]['position_m'], 'button position') for n in BUTTON_IDS}
        for n in BUTTON_IDS:
            if not c['joint_lower_m']-c['joint_tolerance_m'] <= q[n] <= c['joint_upper_m']+c['joint_tolerance_m']:
                raise ValueError('Button outside native joint travel')
        hand, finger, external, nonholder_external, holder_contact, holder_hand, keys, touching = self._contacts(frame)
        corners = self._corners(remote)
        up = rotation(remote['quaternion_wxyz'])[2][2] > .5
        if first and (rotation(remote['quaternion_wxyz'])[2][2] > -.5 or
                      any(q[n] < c['release_position_m'] for n in BUTTON_IDS)):
            raise ValueError('Reset must be face down with released buttons')
        airborne = min(p[2] for p in corners) > c['geometry']['table_top_z_m']+c['minimum_handheld_clearance_m']
        held = hand and not external and up and airborne
        self.held_s = self.held_s+c['physics_dt_s'] if held and not first else 0.
        pressed_now = [n for n in BUTTON_IDS if q[n] <= c['press_position_m']]
        for n in BUTTON_IDS:
            cycle = self.cycle[n]
            if first:
                cycle['armed'] = q[n] >= c['release_position_m'] and not touching[n]
                continue
            if cycle['pressed']:
                cycle['valid'] = cycle['valid'] and held and (not self.legacy or len(pressed_now) <= 1)
                if q[n] >= c['release_position_m'] and not touching[n]:
                    self._emit(n, 'released', frame, q[n], cycle['valid'])
                    cycle.update(armed=True, pressed=False, valid=False)
            elif q[n] >= c['release_position_m'] and not touching[n]:
                cycle['armed'] = True
            elif cycle['armed'] and q[n] <= c['press_position_m']:
                valid = held and self.held_s+EPS >= c['held_before_press_s'] and keys[n] and (not self.legacy or len(pressed_now) == 1)
                self._emit(n, 'pressed', frame, q[n], valid)
                cycle.update(armed=False, pressed=True, valid=valid)
        rh = transpose(rotation(holder['quaternion_wxyz']))
        local = [rotate(rh, subtract(p, holder['position_m'])) for p in corners]
        half = [x/2+c['holder_xy_tolerance_m'] for x in c['geometry']['holder_opening_xy_m']]
        inside = all(abs(p[0]) <= half[0] and abs(p[1]) <= half[1] for p in local)
        bottom_ok = abs(min(p[2] for p in local)-c['geometry']['holder_liner_z_m']) <= c['holder_bottom_tolerance_m']
        oriented = (angle(remote['quaternion_wxyz'], holder['quaternion_wxyz']) <= c['holder_orientation_tolerance_rad']
                    and rotation(holder['quaternion_wxyz'])[2][2] >= math.cos(c['holder_orientation_tolerance_rad']))
        stationary = True
        for name, obj in objects.items():
            stationary &= norm(obj['linear_velocity_m_s']) <= c['maximum_linear_speed_m_s']
            stationary &= norm(obj['angular_velocity_rad_s']) <= c['maximum_angular_speed_rad_s']
            if not first:
                prev = self.previous[name]
                stationary &= norm(subtract(obj['position_m'], prev['position_m']))/c['physics_dt_s'] <= c['maximum_linear_speed_m_s']
                stationary &= angle(obj['quaternion_wxyz'], prev['quaternion_wxyz'])/c['physics_dt_s'] <= c['maximum_angular_speed_rad_s']
        if holder_contact:
            self.support_witness = deepcopy(objects)
        supported, support_source = holder_contact, 'current_native_contact' if holder_contact else 'none'
        if not supported and self.support_witness is not None and stationary and inside and bottom_ok and not finger and not holder_hand and not nonholder_external:
            supported = all(norm(subtract(objects[n]['position_m'], self.support_witness[n]['position_m'])) <= .0001
                and angle(objects[n]['quaternion_wxyz'], self.support_witness[n]['quaternion_wxyz']) <= .005 for n in ('remote', 'holder'))
            if supported:
                support_source = 'prior_native_contact_and_unchanged_unforced_pose'
        if not inside or not bottom_ok or finger or holder_hand or nonholder_external:
            self.support_witness = None
        stable = (not first and inside and bottom_ok and oriented and supported and stationary and not finger and not holder_hand and not nonholder_external
                  and all(q[n] >= c['release_position_m'] and not touching[n] for n in BUTTON_IDS)
                  and (self.legacy or self.television.qualified_off_since is not None))
        if stable:
            if self.stable_anchor is None:
                self.stable_anchor = deepcopy(objects)
            stable = all(norm(subtract(objects[n]['position_m'], self.stable_anchor[n]['position_m'])) <= c['maximum_stable_position_range_m']
                and angle(objects[n]['quaternion_wxyz'], self.stable_anchor[n]['quaternion_wxyz']) <= c['maximum_stable_orientation_range_rad'] for n in objects)
        if stable:
            self.stable_s += c['physics_dt_s']
        else:
            self.stable_s, self.stable_anchor = 0., None
        lo, hi = c['geometry']['workspace_bounds_m']
        terminal = 'remote_outside_workspace_or_below_table' if any(not lo[i] <= remote['position_m'][i] <= hi[i] for i in range(3)) else None
        if not self.legacy:
            # Preserve failures shorter than the public observation period.
            self.terminal_failure = self.terminal_failure or terminal
            terminal = self.terminal_failure
        self.current = {'schema': self.state_schema, 'configuration_sha256': self.digest,
            'time_s': t, 'native_time_s': nt, 'native_step': step, 'baseline': first,
            'buttons': {n: {'position_m': q[n], 'finger_contact': keys[n]} for n in BUTTON_IDS},
            'remote_support': {'hand_contact': hand, 'external_contact': external, 'holder_contact': holder_contact},
            'holder': {'inside': inside and bottom_ok, 'correct_orientation': oriented, 'supported': supported,
                       'support_source': support_source},
            'released_stable_s': self.stable_s, 'terminal_failure': terminal, 'objects': objects}
        if self.television is not None:
            self.current['television'] = self.television.state(t, self.stable_s)
        self.previous = objects
        self.previous_time, self.previous_native_time, self.previous_step = t, nt, step

    def on_observe(self):
        result = deepcopy(self.current or {'schema': self.state_schema, 'configuration_sha256': self.digest})
        result.update(complete=self.current is not None and self.error is None,
                      errors=[] if self.error is None else [self.error], events=deepcopy(self.events))
        return result


class SequenceEvaluator:
    """Canonical worker API; score only worker-owned native observer output."""

    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.legacy = self.c['schema'] == 'living_remote_evaluation.v2'
        self.state_schema = LEGACY_STATE_SCHEMA if self.legacy else STATE_SCHEMA
        self.reset()

    def reset(self):
        self.previous_time = self.previous_step = None
        self.events = []
        self.progress = 0
        self.pending = None
        self.invalid = self.failure = self.last_result = None
        self.stable_s = 0.
        self.success = False
        self.television = None if self.legacy else TelevisionState(self.c['required_on_s'], self.c['stable_s'])
        self.button_pending = {name: False for name in BUTTON_IDS}

    def _result(self, reason):
        self.last_result = {'evaluator': LEGACY_VERSION if self.legacy else VERSION, 'task_id': 'remote_operation',
            'success': bool(self.success and not self.invalid and not self.failure),
            'progress': 0. if self.invalid else (self.progress+min(1., self.stable_s/self.c['stable_s']))/3,
            'completed_button_cycles': self.progress, 'required_button_cycles': 2,
            'stable_s': self.stable_s, 'reason': reason, 'irrecoverable_reason': self.failure,
            'evidence_scope': 'Private native observer events; independent probe evidence is not robot task credit.'}
        if self.television is not None:
            tv = self.television.state(self.previous_time or 0., self.stable_s)
            state_name = ('evaluator_invalid' if self.invalid else 'task_failure' if self.failure else
                          'success' if self.success else 'ongoing')
            phase = ('evidence_invalid' if self.invalid else 'task_failure' if self.failure else
                     'budget_exhausted' if reason == 'incomplete_budget' else tv['phase'])
            self.last_result.update(
                television=tv, completed_button_cycles=self.television.completed_cycles,
                required_button_cycles=2,
                referee={'schema': 'physical_task_referee_state.v1', 'state': state_name,
                    'phase': phase, 'reason_codes': [phase], 'terminal': state_name != 'ongoing',
                    'recoverable': state_name == 'ongoing'},
                progress=0. if self.invalid else (1. if tv['phase'] == 'completed' else
                    (3.+min(1., self.stable_s/self.c['stable_s']))/4. if tv['phase'] == 'await_return' else
                    (1.+min(1., tv['continuous_on_s']/tv['required_on_s']))/4. if tv['power_on'] else 0.))
            if reason == 'incomplete_budget':
                # End the episode through the generic task-budget contract,
                # while retaining an unfinished physical task referee.
                self.last_result.update(status='incomplete', terminal=True, reason='budget_exhausted')
        return deepcopy(self.last_result)

    def update(self, obs):
        if self.invalid:
            return self._result('invalid_observation:'+self.invalid)
        if self.failure:
            return self._result(self.failure)
        try:
            s = obs['task_state']
            if s.get('schema') != self.state_schema or s.get('configuration_sha256') != self.digest or s.get('complete') is not True or s.get('errors') != []:
                raise ValueError('Complete matching private native observer state is required')
            t = number(s['time_s'], 'state time')
            if abs(number(obs['time_s'], 'observation time')-t) > EPS:
                raise ValueError('Observation/native state time differs')
            step = s['native_step']
            if type(step) is not int or step < 0:
                raise ValueError('Invalid native step')
            first = self.previous_time is None
            if first:
                if s['baseline'] is not True or abs(t) > EPS or s['events']:
                    raise ValueError('Unscored reset baseline required')
            elif (s['baseline'] is not False or abs(t-self.previous_time-self.c['control_period_s']) > EPS
                  or step-self.previous_step != round(self.c['control_period_s']/self.c['physics_dt_s'])):
                raise ValueError('Consecutive control observations required')
            incoming = s['events']
            if not isinstance(incoming, list) or incoming[:len(self.events)] != self.events:
                raise ValueError('Native event history was changed')
            for e in incoming[len(self.events):]:
                if (e['id'] != len(self.events)+1 or type(e['id']) is not int
                        or e['button'] not in BUTTON_IDS or e['kind'] not in ('pressed', 'released')
                        or type(e['handheld_valid']) is not bool
                        or type(e['native_step']) is not int
                        or not self.previous_step < e['native_step'] <= step
                        or not self.previous_time < number(e['time_s'], 'event time') <= t+EPS):
                    raise ValueError('Invalid native button event')
                if self.events and (e['native_step'] < self.events[-1]['native_step'] or e['time_s'] < self.events[-1]['time_s']):
                    raise ValueError('Native button events out of order')
                if not self.legacy:
                    expected_time = self.previous_time+(e['native_step']-self.previous_step)*self.c['physics_dt_s']
                    position = number(e['position_m'], 'event button position')
                    pressed = e['kind'] == 'pressed'
                    if (abs(e['time_s']-expected_time) > EPS
                            or self.button_pending[e['button']] == pressed
                            or not self.c['joint_lower_m']-self.c['joint_tolerance_m'] <= position <= self.c['joint_upper_m']+self.c['joint_tolerance_m']
                            or (pressed and position > self.c['press_position_m'])
                            or (not pressed and position < self.c['release_position_m'])):
                        raise ValueError('Native button cycle differs from step/displacement evidence')
                    self.button_pending[e['button']] = pressed
                self.events.append(deepcopy(e))
                if self.television is not None:
                    self.television.event(e)
                else:
                    expected = self.c['button_sequence'][self.progress] if self.progress < 2 else None
                    if e['kind'] == 'pressed':
                        if self.pending is not None or not e['handheld_valid'] or e['button'] != expected:
                            self.progress, self.pending = 0, None
                        else:
                            self.pending = e['button']
                    elif self.pending == e['button'] and e['handheld_valid']:
                        self.progress += 1
                        self.pending = None
                    else:
                        self.progress, self.pending = 0, None
            stable = number(s['released_stable_s'], 'stable dwell')
            if stable < 0 or stable > t+EPS:
                raise ValueError('Invalid native stable duration')
            self.stable_s = stable if self.legacy and self.progress == 2 else 0.
            if self.television is not None:
                off_since = self.television.qualified_off_since
                if stable > 0 and (off_since is None or stable > t-off_since+EPS):
                    raise ValueError('Native holder stability must follow qualified television power-off')
                self.stable_s = stable if off_since is not None else 0.
                tv = s['television']
                if (not isinstance(tv, Mapping) or type(tv.get('power_on')) is not bool
                        or type(tv.get('continuous_on_s')) not in (float, int)
                        or type(tv.get('required_on_s')) not in (float, int)
                        or tv != self.television.state(t, self.stable_s)):
                    raise ValueError('Native television state differs from physical power events')
            self.previous_time, self.previous_step = t, step
            if s['terminal_failure'] is not None:
                self.failure, self.success = str(s['terminal_failure']), False
                return self._result(self.failure)
            if obs.get('budget_exhausted') is True or t > self.c['max_time_s']+EPS or number(obs.get('action_count', 0), 'actions') > self.c['max_actions']:
                self.success = False
                return self._result('incomplete_budget')
            self.success = (self.progress == 2 and self.stable_s+EPS >= self.c['stable_s'] if self.legacy
                            else tv['phase'] == 'completed')
            reason = ('perform_ordered_handheld_cycles_then_return' if self.legacy
                      else tv['phase'])
            return self._result('success' if self.success else ('reset_baseline_not_scored' if first else reason))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid, self.success, self.stable_s = str(exc), False, 0.
            return self._result('invalid_observation:'+self.invalid)

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result or {'success': False, 'reason': 'not_sampled'})


RemoteOperationEvaluator = SequenceEvaluator
