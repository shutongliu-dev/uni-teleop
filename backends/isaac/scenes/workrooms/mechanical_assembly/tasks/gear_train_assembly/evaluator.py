"""Canonical L5 assembly and contact-transmission score, with no state writes."""
from copy import deepcopy
import math
import numpy as np

from .native_evidence import EPS, number, vector, rotation, angle, within, read_native_contact_window
from .rules import GEARS, TOOLS, public_rules, thresholds, geometry_contract

if __package__.startswith('scenes.workrooms.mechanical_assembly.'):
    from ...referee import verdict
else:
    from .referee import verdict

VERSION = 'mechanical_gear_evaluator_v3'


class SafetyStop(ValueError):
    """Valid physical evidence crossed an existing safety limit."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def yaw_increment(previous, current):
    """Signed short world-Z twist; tilt is separately bounded by seat geometry."""
    a, b = [math.atan2(r[1, 0], r[0, 0]) for r in (previous, current)]
    return math.atan2(math.sin(b-a), math.cos(b-a))


class SequenceEvaluator:
    def __init__(self, task_spec):
        self.task = deepcopy(task_spec)
        if (self.task.get('id') != 'gear_train_assembly'
                or self.task.get('schema_version') != 'mechanical-gear-train-task-v1'
                or self.task.get('public_rules') != public_rules() or self.task.get('engineering_probe')):
            raise ValueError('Expected the frozen original gear assembly task')
        self.c = self.task['evaluation']
        for key, expected in geometry_contract().items():
            if self.c.get(key) != expected:
                raise ValueError('Frozen gear geometry contract differs: '+key)
        self.turn_weights = np.asarray(self.c['input_equivalent_turn_weights'])
        for key, expected in thresholds().items():
            if number(self.c.get(key), key) != expected:
                raise ValueError('Frozen gear scoring threshold differs: '+key)
        self.dt, self.control = self.c['physics_dt_s'], self.c['control_period_s']
        self.substeps = round(self.control/self.dt)
        self.actors = self.c['actors']
        if (set(self.actors) != set(GEARS+TOOLS) or len(set(self.actors.values())) != 6
                or any(not isinstance(p, str) or not p.startswith('/World/Objects/'+n)
                    for n, p in self.actors.items())):
            raise ValueError('Six distinct pinned object actors are required')
        if any(self.actors[n] != '/World/Objects/'+n for n in GEARS+TOOLS[:2]):
            raise ValueError('Gear or plain tool actor identity changed')
        if self.actors[TOOLS[2]] != self.c['original_screwdriver_actor']:
            raise ValueError('Original referenced screwdriver actor changed')
        if self.c['fixture_actor'] != '/World/Fixtures/gear_workplate' or self.c['robot_actor_prefix'] != '/World/Robot':
            raise ValueError('Fixed workplate and robot actor roots are required')
        self.targets = {n: vector(p, 3, 'gear bottom target') for n, p in self.c['target_bottom_centers_m'].items()}
        if set(self.targets) != set(GEARS):
            raise ValueError('Three explicit intended shafts are required')
        if any(not np.array_equal(self.targets[n], [*xy, .802])
               for n, xy in zip(GEARS, self.c['shaft_centers_world_xy_m'])):
            raise ValueError('Intended shaft locations differ from the physical workplate')
        self.expected = {n: self._state(row, velocities=False) for n, row in self.c['initial_objects'].items()}
        if set(self.expected) != set(GEARS+TOOLS):
            raise ValueError('All six independent bodies require reset poses')
        self.reset()

    @staticmethod
    def _state(row, *, velocities=True):
        return {'p': vector(row['position_m'], 3, 'position'), 'r': rotation(row['quaternion_wxyz']),
            'v': vector(row['linear_velocity_m_s'], 3, 'linear velocity') if velocities else np.zeros(3),
            'w': vector(row['angular_velocity_rad_s'], 3, 'angular velocity') if velocities else np.zeros(3)}

    def reset(self):
        self.previous_time = self.native_step = self.native_time = self.previous = None
        self.baseline = self.anchor = self.invalid = self.last_result = None
        self.settle_s = self.release_s = self.credit = self.interaction_s = 0.
        self.turn = np.zeros(3)
        self.maximum_phase_error = 0.
        self.status = {}
        self.success = False
        self.terminal = False
        self.used_tools = set()

    def _contacts(self, obs, time, first):
        rows, step, native_time = read_native_contact_window(obs, time=time, first=first,
            physics_dt_s=self.dt, control_period_s=self.control, previous_time=self.previous_time,
            native_step=self.native_step, native_time=self.native_time)
        support_z = {n: {} for n in GEARS}
        meshes = [set(), set()]
        tool_input = {n: set() for n in TOOLS}
        hand_contacts = {n: {} for n in TOOLS}
        hand_tools = set()
        direct_gear = other_tool_gear = non_hand_tool = False
        robot = self.c['robot_actor_prefix']
        start = time if first else self.previous_time
        for row in rows:
            a, b = row['actor0'], row['actor1']
            if not any(p in self.actors.values() for p in (a, b)):
                continue
            if row['min_separation_m'] < -self.c['maximum_reported_penetration_m']:
                raise SafetyStop('penetration_limit', 'Task contact penetration exceeds frozen bound')
            # A valid touching manifold prevents release even when its current
            # normal load is zero. Positive pressure is a separate drive test.
            for name in GEARS+TOOLS:
                body = self.actors[name]
                if body not in (a, b):
                    continue
                other = b if a == body else a
                if name in GEARS:
                    direct_gear |= within(other, robot)
                    other_tool_gear |= name != 'input_gear' and other in [self.actors[t] for t in TOOLS]
                elif within(other, robot):
                    hand_tools.add(name)
                    non_hand_tool |= not other.rsplit('/', 1)[-1].startswith('right_')
            tick = round((row['time_s']-start)/self.dt)
            vec = np.array(row['impulse_vector_ns'])
            # Accumulate every fixture reaction before thresholding; otherwise
            # separate downward contacts could be discarded from net support.
            for name in GEARS:
                body = self.actors[name]
                if body in (a, b):
                    other = b if a == body else a
                    if within(other, self.c['fixture_actor']):
                        signed_z = vec[2] if a == body else -vec[2]
                        support_z[name][tick] = support_z[name].get(tick, 0.)+signed_z
            if row['impulse_magnitude_sum_ns'] <= self.c['minimum_pair_impulse_ns']:
                continue
            for name in GEARS+TOOLS:
                body = self.actors[name]
                if body not in (a, b):
                    continue
                other = b if a == body else a
                signed = vec if a == body else -vec
                if name in TOOLS and within(other, robot):
                    if other.rsplit('/', 1)[-1].startswith('right_'):
                        hand_contacts[name].setdefault(tick, []).append((other, signed))
                if name in TOOLS and other == self.actors['input_gear']:
                    tool_input[name].add(tick)
            for i, pair in enumerate(zip(GEARS, GEARS[1:])):
                if {a, b} == {self.actors[pair[0]], self.actors[pair[1]]}:
                    meshes[i].add(tick)
        supports = {n: {tick for tick, z in values.items() if z > self.c['minimum_pair_impulse_ns']}
                    for n, values in support_z.items()}
        chains = {}
        for name, by_tick in hand_contacts.items():
            held = set()
            for tick, contacts in by_tick.items():
                # Opposing positive normal loads on different hand links supply
                # observable holding evidence without assigning a correct tool ID.
                for i, (link, force) in enumerate(contacts):
                    for other, second in contacts[i+1:]:
                        magnitude = np.linalg.norm(force)*np.linalg.norm(second)
                        if link != other and magnitude > 0. and np.dot(force, second)/magnitude <= self.c['maximum_holding_normal_cosine']:
                            held.add(tick)
            chain = held & tool_input[name] & meshes[0] & meshes[1]
            for support in supports.values():
                chain &= support
            chains[name] = len(chain)/self.substeps
        return {'supported': all(supports.values()), 'gear_released': not direct_gear,
            'released': not direct_gear and not hand_tools, 'direct_other_drive': direct_gear or other_tool_gear,
            'chain_fractions': chains, 'chain': 0. if non_hand_tool else max(chains.values(), default=0.)}, step, native_time

    def _seated(self, states, *, record=True):
        status = {}
        for n in GEARS:
            s, target = states[n], self.targets[n]
            xy = float(np.linalg.norm(s['p'][:2]-target[:2]))
            height = float(s['p'][2]-target[2])
            tilt = math.acos(float(np.clip(s['r'][2, 2], -1., 1.)))
            status[n] = {'axis_offset_m': xy, 'bottom_height_error_m': height, 'tilt_rad': tilt,
                'seated': xy <= self.c['maximum_axis_offset_m'] and abs(height) <= self.c['maximum_bottom_height_error_m']
                    and tilt <= self.c['maximum_axis_tilt_rad']}
        if record:
            self.status = status
        return all(row['seated'] for row in status.values())

    def _clear_drive(self):
        self.baseline = self.anchor = None
        self.settle_s = self.release_s = self.credit = self.interaction_s = 0.
        self.turn[:] = 0.
        self.maximum_phase_error = 0.
        self.success = False
        self.used_tools.clear()

    def _result(self, reason, *, phase='assembly', state='ongoing', reason_code=None):
        self.terminal = state != 'ongoing'
        # Keep the legacy free-text category for existing private consumers;
        # public clients use the declared, more specific referee reason code.
        legacy_reason = reason
        if state == 'ongoing' and reason != 'reset_baseline_not_scored':
            legacy_reason = ('released_assembly_settling' if reason == 'assembly_baseline_recorded' else {
                'assembly': 'assembly_support_release_required',
                'assembly_settling': 'released_assembly_settling',
                'transmission': 'contact_backed_transmission_and_release_required',
                'reassembly_required': 'contact_backed_transmission_and_release_required',
                'release_settling': 'contact_backed_transmission_and_release_required',
            }[phase])
        self.last_result = {'evaluator': VERSION, 'task_id': 'gear_train_assembly',
            'success': bool(self.success and not self.invalid), 'reason': legacy_reason,
            'referee': verdict(state=state, phase=phase, reasons=[reason_code or reason]),
            'progress': 1. if self.success else 0. if self.invalid else min(.9, .9*self.credit/self.c['nominal_output_turn_rad']),
            'assembly_baseline_recorded': self.baseline is not None,
            'signed_gear_turns_rad': dict(zip(GEARS, self.turn.tolist())), 'contact_backed_output_turn_rad': self.credit,
            'maximum_pair_phase_error_rad': self.maximum_phase_error,
            'tool_interaction_s': self.interaction_s, 'assembly_settle_s': self.settle_s,
            'release_stable_s': self.release_s, 'objects': deepcopy(self.status),
            'used_tools': sorted(self.used_tools),
            'threshold_status': 'engineering_candidates_not_native_accepted',
            'evidence_scope': '20 ms object poses/velocities plus complete native-substep normal contacts; no measured torque or continuous substep pose claim.'}
        return deepcopy(self.last_result)

    def update(self, observation):
        if self.terminal:
            return deepcopy(self.last_result)
        try:
            time = number(observation['time_s'], 'time')
            first = self.previous_time is None
            if observation.get('metrics', {}).get('state_finite') is not True or observation.get('terminated') is not False:
                raise ValueError('Finite continuing native state is required')
            states = {n: self._state(observation['objects'][n]) for n in GEARS+TOOLS}
            # Validate every reset body before classifying any physical limit.
            # A mismatched reset is unavailable evidence for this task attempt.
            if first and any(np.linalg.norm(s['p']-self.expected[n]['p']) > self.c['initial_position_tolerance_m']
                    or angle(s['r'], self.expected[n]['r']) > self.c['initial_orientation_tolerance_rad']
                    for n, s in states.items()):
                raise ValueError('Initial body pose differs from the frozen unassembled reset')
            contacts, step, native_time = self._contacts(observation, time, first)
            quiet_by_object, drive_speed = {}, True
            increments = np.zeros(3)
            for n, s in states.items():
                linear = 0. if first else np.linalg.norm(s['p']-self.previous[n]['p'])/self.control
                angular = 0. if first else angle(self.previous[n]['r'], s['r'])/self.control
                if (max(linear, np.linalg.norm(s['v'])) > self.c['maximum_object_linear_speed_m_s']+EPS
                        or max(angular, np.linalg.norm(s['w'])) > self.c['maximum_object_angular_speed_rad_s']+EPS):
                    raise SafetyStop('object_speed_limit', 'Object motion exceeds the frozen speed bound')
                quiet_by_object[n] = (max(linear, np.linalg.norm(s['v'])) <= self.c['maximum_stable_linear_speed_m_s']+EPS
                    and np.linalg.norm(s['w']) <= self.c['maximum_stable_angular_speed_rad_s']+EPS
                    and angular <= self.c['maximum_stable_observed_angular_speed_rad_s']+EPS)
                if not first:
                    quiet_by_object[n] &= (np.linalg.norm(self.previous[n]['v']) <= self.c['maximum_stable_linear_speed_m_s']+EPS
                        and np.linalg.norm(self.previous[n]['w']) <= self.c['maximum_stable_angular_speed_rad_s']+EPS)
                if n in GEARS:
                    drive_speed &= max(angular, np.linalg.norm(s['w'])) <= self.c['maximum_drive_angular_speed_rad_s']+EPS
                    if not first:
                        drive_speed &= np.linalg.norm(self.previous[n]['w']) <= self.c['maximum_drive_angular_speed_rad_s']+EPS
                        increments[GEARS.index(n)] = yaw_increment(self.previous[n]['r'], s['r'])
            quiet = all(quiet_by_object[n] for n in GEARS)
            seated = self._seated(states)
            phase = 'assembly'
            if first:
                reason = 'reset_baseline_not_scored'
            elif not seated or contacts['direct_other_drive']:
                reason = 'gear_seating_required' if not seated else 'direct_drive_requires_reassembly'
                self._clear_drive()
            elif self.baseline is None:
                phase = 'assembly_settling'
                if quiet and contacts['supported'] and contacts['gear_released'] and self._seated(self.previous, record=False):
                    if self.anchor is None:
                        self.anchor = deepcopy(self.previous)
                    spread = all(np.linalg.norm(states[n]['p']-self.anchor[n]['p']) <= self.c['maximum_stable_position_range_m']
                        and angle(states[n]['r'], self.anchor[n]['r']) <= self.c['maximum_stable_angle_range_rad'] for n in GEARS)
                    if spread:
                        self.settle_s += self.control
                        reason = 'released_assembly_settling'
                    else:
                        self.settle_s, self.anchor = 0., None
                        reason = 'assembly_stability_required'
                    if self.settle_s+EPS >= self.c['assembly_settle_s']:
                        self.baseline = deepcopy(states)
                        self.anchor = None
                        reason, phase = 'assembly_baseline_recorded', 'transmission'
                else:
                    self.settle_s, self.anchor = 0., None
                    reason = ('assembly_support_required' if not contacts['supported'] else
                        'assembly_release_required' if not contacts['gear_released'] else 'assembly_stability_required')
            else:
                phase = 'transmission'
                self.turn += increments
                equivalent = self.turn*self.turn_weights
                equivalent_increment = increments*self.turn_weights
                phase_error = max(abs(equivalent[0]+equivalent[1]), abs(equivalent[1]+equivalent[2]))
                self.maximum_phase_error = max(self.maximum_phase_error, phase_error)
                chain = contacts['chain']+EPS >= self.c['minimum_motion_chain_fraction']
                qualified = chain and drive_speed and contacts['supported']
                added = min(max(equivalent_increment[0], 0.), max(-equivalent_increment[1], 0.), max(equivalent_increment[2], 0.)) if qualified else 0.
                reverse = max(max(-equivalent_increment[0], 0.), max(equivalent_increment[1], 0.), max(-equivalent_increment[2], 0.))
                self.credit = max(0., min(self.credit+added-reverse, equivalent[0], -equivalent[1], equivalent[2]))
                if qualified and added > 1.e-9:
                    self.interaction_s += self.control
                    self.used_tools.update(name for name, fraction in contacts['chain_fractions'].items()
                        if fraction+EPS >= self.c['minimum_motion_chain_fraction'])
                ratio = equivalent[0] > 0. and max(abs(equivalent[1]/equivalent[0]+1.), abs(equivalent[2]/equivalent[0]-1.)) <= self.c['maximum_ratio_error']
                target = (self.c['minimum_output_turn_rad']-EPS <= self.turn[2] <= self.c['maximum_output_turn_rad']+EPS
                    and self.credit+EPS >= self.c['minimum_output_turn_rad'] and ratio
                    and self.maximum_phase_error <= self.c['maximum_pair_phase_error_rad']+EPS
                    and self.interaction_s+EPS >= self.c['minimum_tool_interaction_s'])
                tools_quiet = all(quiet_by_object[n] for n in self.used_tools)
                if target and quiet and tools_quiet and contacts['supported'] and contacts['released']:
                    if self.anchor is None:
                        self.anchor = deepcopy(self.previous)
                    spread_by_object = {n: np.linalg.norm(states[n]['p']-self.anchor[n]['p']) <= self.c['maximum_stable_position_range_m']
                        and angle(states[n]['r'], self.anchor[n]['r']) <= self.c['maximum_stable_angle_range_rad'] for n in (*GEARS, *self.used_tools)}
                    spread = all(spread_by_object.values())
                    if spread:
                        self.release_s += self.control
                        reason = 'release_settling'
                    else:
                        self.release_s, self.anchor = 0., None
                        reason = ('used_tool_stability_required' if any(not spread_by_object[n] for n in self.used_tools)
                            else 'release_stability_required')
                    phase = 'release_settling'
                else:
                    self.release_s, self.anchor = 0., None
                    if self.maximum_phase_error > self.c['maximum_pair_phase_error_rad']+EPS:
                        reason, phase = 'phase_error_requires_reassembly', 'reassembly_required'
                    elif not contacts['supported']:
                        reason = 'supported_transmission_required'
                    elif not target:
                        reason = ('drive_speed_required' if not drive_speed else
                            'contact_chain_required' if not chain and not quiet else
                            'direction_ratio_required' if not ratio else
                            'output_turn_required' if not self.c['minimum_output_turn_rad']-EPS <= self.turn[2] <= self.c['maximum_output_turn_rad']+EPS else
                            'contact_credit_required' if self.credit+EPS < self.c['minimum_output_turn_rad'] else 'tool_interaction_required')
                    elif not contacts['released']:
                        reason, phase = 'release_required', 'release_settling'
                    else:
                        reason = 'used_tool_stability_required' if not tools_quiet else 'release_stability_required'
                        phase = 'release_settling'
                self.success = self.release_s+EPS >= self.c['release_hold_s']
                if self.success:
                    reason, phase = 'success', 'complete'
            self.previous, self.previous_time = states, time
            self.native_step, self.native_time = step, native_time
            return self._result(reason, phase=phase, state='success' if self.success else 'ongoing')
        except SafetyStop as exc:
            self.invalid = str(exc)
            self.success = False
            return self._result('safety_stop:'+self.invalid, state='safety_stop', phase='safety_stop', reason_code=exc.code)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
            self.invalid = str(exc)
            self.success = False
            return self._result('invalid_observation:'+self.invalid, state='evaluator_invalid', phase='unavailable', reason_code='invalid_observation')

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result or {'success': False, 'reason': 'not_sampled',
            'referee': verdict(state='ongoing', phase='assembly', reasons=['not_sampled'])})
