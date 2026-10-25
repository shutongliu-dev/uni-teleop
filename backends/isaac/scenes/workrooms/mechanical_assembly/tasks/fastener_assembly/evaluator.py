"""Read-only canonical-observation fastener score, with fail-closed evidence.

20 ms object poses/velocities establish bolt motion; all native substep contact
rows establish interaction/support/release. This does not measure torque, prove
continuous substep poses, or establish that a physical grasp is feasible.
"""
from copy import deepcopy
import math
import numpy as np

from .rules import public_rules, thresholds
if __package__.startswith('scenes.'):
    from ...referee import verdict
else:
    from .referee import verdict

VERSION = 'mechanical_fastener_evaluator_v6'
EPS = 1.e-9


class SafetyStop(ValueError):
    def __init__(self, message, reason_code):
        super().__init__(message)
        self.reason_code = reason_code


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(name + ' must have the declared dimension')
    return np.array([number(v, name) for v in value], dtype=float)


def rotation(value):
    q = vector(value, 4, 'quaternion')
    norm = np.linalg.norm(q)
    if abs(norm-1.) > 1.e-4:
        raise ValueError('orientation must be a unit wxyz quaternion')
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def angle(a, b):
    return math.acos(float(np.clip((np.trace(a.T @ b)-1)/2, -1., 1.)))


class FastenerEvaluator:
    def __init__(self, task_spec):
        self.task = deepcopy(task_spec)
        if self.task.get('schema_version') != 'mechanical-fastener-task-v1' or self.task.get('id') != 'fastener_assembly':
            raise ValueError('Expected the independently versioned fastener task')
        target = self.task.get('public_rules', {}).get('success', {}).get('nominal_turn_degrees')
        self.reset_variant = self.task.get('public_rules', {}).get('reset_variant', 'preinserted')
        if self.reset_variant not in ('preinserted', 'loose_bolt'):
            raise ValueError('Unknown fastener reset variant')
        if self.task.get('public_rules') != public_rules(target, reset_variant=self.reset_variant):
            raise ValueError('Public rules differ from this frozen evaluator')
        self.c = self.task['evaluation']
        for key, expected in thresholds(target).items():
            if number(self.c.get(key), key) != expected:
                raise ValueError('Task threshold differs from the versioned candidate: '+key)
        self.dt, self.control = self.c['physics_dt_s'], self.c['control_period_s']
        self.substeps = round(self.control/self.dt)
        self.actors = self.c['actors']
        if (set(self.actors) != {'screwdriver', 'bolt', 'threaded_fixture'}
                or len(set(self.actors.values())) != 3
                or any(not isinstance(p, str) or not p.startswith('/World/') for p in self.actors.values())):
            raise ValueError('Expected three distinct pinned native body actor paths')
        self.fixture_p = vector(self.c['fixture_position_m'], 3, 'fixture position')
        self.fixture_r = rotation(self.c['fixture_quaternion_wxyz'])
        self.expected = {name: self._state(row, velocities=False)
                         for name, row in self.c['initial_objects'].items()}
        if set(self.expected) != {'bolt', 'screwdriver'}:
            raise ValueError('Both independent task bodies require reset poses')
        self.reset()

    @staticmethod
    def _state(row, *, velocities=True):
        return {'p': vector(row['position_m'], 3, 'object position'),
                'r': rotation(row['quaternion_wxyz']),
                'v': vector(row['linear_velocity_m_s'], 3, 'linear velocity') if velocities else np.zeros(3),
                'w': vector(row['angular_velocity_rad_s'], 3, 'angular velocity') if velocities else np.zeros(3)}

    def reset(self):
        self.previous_time = self.native_step = self.native_time = None
        self.previous = self.baseline = None
        self.settle_elapsed = self.release_elapsed = self.interaction_s = 0.
        self.support_gap_steps = 0
        self.last_yaw = None
        self.turn = self.advance = self.max_residual = 0.
        self.motion_chain_verified = False
        self.contact_backed_turn = 0.
        self.release_anchor = self.settle_anchor = None
        self.engaged_seen = False
        self.invalid = self.irrecoverable = None
        self.stop_state = self.stop_code = None
        self.phase, self.reasons = 'settle', ['not_sampled']
        self.success = False
        self.last_result = None

    def _sampling(self, obs, time, first):
        contacts = obs['contacts']
        if contacts.get('source') != 'PhysX stage-bound rigid contact tensor; normal impulses' or contacts.get('errors') != []:
            raise ValueError('native contact source or error status is invalid')
        s = contacts['sampling']
        if s.get('protocol') != 'native_rigid_contact_sampling_v1' or s.get('complete') is not True:
            raise ValueError('complete native contact sampling is required')
        if number(s['physics_dt_s'], 'sampling dt') != self.dt:
            raise ValueError('native physics dt differs from task')
        for key in ('sampled_substeps', 'native_start_step_index', 'native_end_step_index'):
            if type(s[key]) is not int or s[key] < 0:
                raise ValueError('native counters must be nonnegative integers')
        start, end, ns, ne = [number(s[key], key) for key in ('start_sim_time_s', 'end_sim_time_s',
                                                           'native_start_time_s', 'native_end_time_s')]
        duration = 0. if first else self.control
        count = 0 if first else self.substeps
        if (min(start, end, ns, ne) < -EPS or s['sampled_substeps'] != count
                or s['native_end_step_index']-s['native_start_step_index'] != count
                or abs(end-time) > EPS or abs(end-start-duration) > EPS
                or abs(ne-ns-duration) > EPS or abs(number(obs['last_duration_s'], 'duration')-duration) > EPS):
            raise ValueError('native contact receipt has incomplete coverage or inconsistent clocks')
        if first:
            if abs(time) > EPS or contacts['rigid']:
                raise ValueError('First observation must be the unsampled reset baseline')
        elif (abs(start-self.previous_time) > EPS or s['native_start_step_index'] != self.native_step
              or abs(ns-self.native_time) > EPS):
            raise ValueError('native contact windows must be consecutive')
        return start, s['native_end_step_index'], ne

    def _contacts(self, obs, time, first):
        start, step, native_time = self._sampling(obs, time, first)
        rows = obs['contacts']['rigid']
        if not isinstance(rows, list):
            raise ValueError('native contacts must be a list')
        sets = {key: set() for key in ('hand_tool', 'tool_bolt', 'bolt_fixture', 'robot_target')}
        violation = None
        penetration_exceeded = False
        seen = set()
        tool, bolt, fixture = (self.actors[k] for k in ('screwdriver', 'bolt', 'threaded_fixture'))
        for row in rows:
            a, b = row['actor0'], row['actor1']
            if not isinstance(a, str) or not isinstance(b, str) or a == b:
                raise ValueError('invalid native actor pair')
            stamp = number(row['time_s'], 'contact time')
            tick = round((stamp-start)/self.dt)
            if not 1 <= tick <= self.substeps or abs(stamp-start-tick*self.dt) > EPS:
                raise ValueError('stale, future or off-grid contact row')
            key = (tick, *sorted((a,b)))
            if key in seen:
                raise ValueError('duplicate native contact pair')
            seen.add(key)
            impulse = number(row['impulse_magnitude_sum_ns'], 'normal impulse')
            vec = vector(row['impulse_vector_ns'], 3, 'normal impulse vector')
            vector_norm = math.hypot(*vec)
            separation = number(row['min_separation_m'], 'separation')
            normal_error = number(row['normal_length_max_abs_error'], 'normal error')
            n, active = row['point_count'], row['nonzero_impulse_normal_count']
            # The recorder sums the same float64 point vectors and their
            # lengths. Use the family protocol's relative roundoff allowance;
            # no absolute impulse floor may hide contradictory zero/tiny rows.
            if (type(n) is not int or n <= 0 or type(active) is not int or not 0 <= active <= n
                    or impulse < 0 or not math.isfinite(vector_norm) or vector_norm > impulse*(1.+1.e-9)
                    or (impulse == 0. and (active != 0 or np.any(vec != 0.)))
                    or (impulse > 0 and active == 0) or not 0 <= normal_error <= 32*2**-23
                    or row.get('impulse_components') != 'normal_only'):
                raise ValueError('invalid native normal-impulse evidence')
            if separation < -self.c['maximum_reported_penetration_m']:
                penetration_exceeded = True
            # Release means no reported robot manifold, including zero-load
            # touching contacts. Positive-impulse credit/violation gates below
            # keep their frozen threshold and do not turn proximity into credit.
            if any(target in (tool, bolt) and other.startswith(self.c['robot_actor_prefix'])
                   for target, other in ((a, b), (b, a))):
                sets['robot_target'].add(tick)
            if impulse <= self.c['minimum_pair_impulse_ns']:
                continue
            pair = {a,b}
            if pair == {tool,bolt}: sets['tool_bolt'].add(tick)
            if pair == {bolt,fixture}: sets['bolt_fixture'].add(tick)
            for target, other in ((a,b),(b,a)):
                if target not in (tool,bolt) or not other.startswith(self.c['robot_actor_prefix']):
                    continue
                if target == bolt:
                    setup_hand_contact = (self.reset_variant == 'loose_bolt' and self.baseline is None
                        and other.rsplit('/',1)[-1].startswith(self.c['hand_link_prefix']))
                    if not setup_hand_contact:
                        violation = violation or 'direct_robot_bolt_contact'
                elif other.rsplit('/',1)[-1].startswith(self.c['hand_link_prefix']):
                    sets['hand_tool'].add(tick)
                else:
                    violation = violation or 'non_hand_robot_tool_contact'
        return sets, step, native_time, violation, penetration_exceeded

    def _geometry(self, state):
        p = self.fixture_r.T @ (state['p']-self.fixture_p)
        r = self.fixture_r.T @ state['r']
        yaw = math.atan2(r[1,0], r[0,0])
        tilt = math.acos(float(np.clip(r[2,2], -1., 1.)))
        return p, yaw, tilt

    def _result(self, reason):
        state = (self.stop_state if self.invalid else 'task_failure' if self.irrecoverable
                 else 'success' if self.success else 'ongoing')
        phase = 'complete' if state == 'success' else 'stopped' if state != 'ongoing' else self.phase
        reasons = ([self.stop_code] if self.invalid else [self.irrecoverable] if self.irrecoverable
                   else ['success'] if self.success else self.reasons)
        self.last_result = {'evaluator': VERSION, 'task_id': self.task['id'],
            'success': bool(self.success and not self.invalid and not self.irrecoverable),
            'progress': 1. if self.success and not self.invalid and not self.irrecoverable else min(.9, max(0., self.contact_backed_turn/self.c['nominal_turn_rad'])*.9),
            'reason': reason, 'irrecoverable_reason': self.irrecoverable,
            'referee': verdict(state=state, phase=phase, reasons=reasons),
            'settled_baseline_recorded': self.baseline is not None,
            'bolt_turn_rad': self.turn, 'contact_backed_bolt_turn_rad': self.contact_backed_turn, 'additional_axial_advance_m': self.advance,
            'maximum_lead_residual_m': self.max_residual,
            'tool_interaction_s': self.interaction_s,
            'release_stable_s': self.release_elapsed,
            'threshold_status': 'engineering_candidates_not_native_accepted',
            'evidence_scope': '20ms object poses/velocities plus complete native-substep normal contacts; no substep pose or measured torque claim.'}
        return deepcopy(self.last_result)

    def update(self, observation):
        if self.last_result is not None and (self.invalid or self.irrecoverable or self.success):
            return deepcopy(self.last_result)
        try:
            time = number(observation['time_s'], 'observation time')
            if abs(number(observation['sim_time_s'], 'simulation time')-time) > EPS:
                raise ValueError('observation clocks disagree')
            first = self.previous_time is None
            if not first and abs(time-self.previous_time-self.control) > EPS:
                raise ValueError('Object poses must arrive at every 20ms control boundary')
            if observation.get('metrics', {}).get('state_finite') is not True or observation.get('terminated') is not False:
                raise ValueError('native state finiteness or continuation is not established')
            states = {name: self._state(observation['objects'][name]) for name in self.expected}
            contacts, step, native_time, violation, penetration_exceeded = self._contacts(observation, time, first)
            observed_quiet = True
            speed_exceeded = False
            for name, state in states.items():
                if first and (np.linalg.norm(state['p']-self.expected[name]['p']) > self.c['initial_position_tolerance_m']
                              or angle(state['r'], self.expected[name]['r']) > self.c['initial_orientation_tolerance_rad']):
                    raise ValueError('Initial pose differs from frozen reset')
                rate = 0. if first else np.linalg.norm(state['p']-self.previous[name]['p'])/self.control
                angular = 0. if first else angle(self.previous[name]['r'], state['r'])/self.control
                observed_quiet = observed_quiet and rate <= self.c['maximum_stable_linear_speed_m_s'] and angular <= self.c['maximum_stable_observed_angular_speed_rad_s']
                angular_limit = self.c['maximum_bolt_angular_speed_rad_s'] if name == 'bolt' else self.c['maximum_tool_angular_speed_rad_s']
                if (max(rate, np.linalg.norm(state['v'])) > self.c['maximum_object_linear_speed_m_s']
                        or max(angular, np.linalg.norm(state['w'])) > angular_limit+EPS):
                    speed_exceeded = True
            if penetration_exceeded:
                raise SafetyStop('reported penetration exceeds frozen bound', 'reported_penetration_bound')
            if speed_exceeded:
                raise SafetyStop('object motion speed bound exceeded', 'object_motion_speed_bound')
            position, yaw, tilt = self._geometry(states['bolt'])
            # A loose bolt is expected to be off-axis while it is being
            # picked from the desktop.  Axis/tilt limits begin after the
            # first complete bolt-fixture/tool engagement, never before.
            engaged_now = bool(contacts['bolt_fixture'] and contacts['tool_bolt'])
            axis_checked = (self.reset_variant == 'preinserted' or self.engaged_seen
                            or engaged_now or self.baseline is not None)
            if axis_checked and (np.linalg.norm(position[:2]) > self.c['maximum_axis_offset_m']
                                 or tilt > self.c['maximum_bolt_tilt_rad']):
                violation = violation or 'bolt_left_thread_axis'
            self.native_step, self.native_time = step, native_time
            if first:
                self.previous, self.previous_time = states, time
                self.irrecoverable = violation
                self.phase, self.reasons = 'settle', ['reset_baseline_not_scored']
                return self._result(violation or 'reset_baseline_not_scored')
            support_covered = True
            for tick in range(1, self.substeps+1):
                self.support_gap_steps = 0 if tick in contacts['bolt_fixture'] else self.support_gap_steps+1
                if self.support_gap_steps*self.dt > self.c['maximum_support_gap_s']+EPS:
                    support_covered = False
                    # During settling this resets the dwell; later it forbids unsupported progress.
                    if self.baseline is not None: violation = violation or 'lost_thread_fixture_support'
            quiet = observed_quiet and all(np.linalg.norm(s['v']) <= self.c['maximum_stable_linear_speed_m_s']
                        and np.linalg.norm(s['w']) <= self.c['maximum_stable_angular_speed_rad_s'] for s in states.values())
            if self.baseline is None:
                if self.reset_variant == 'loose_bolt':
                    # Before engagement, the candidate is a recoverable
                    # pickup/alignment phase.  Do not score motion or require
                    # axis/support while the bolt is still on the desktop.
                    if engaged_now:
                        self.engaged_seen = True
                    near_reset = self.engaged_seen
                    settle_contacts = self.engaged_seen and contacts['bolt_fixture'] and contacts['tool_bolt']
                else:
                    near_reset = all(np.linalg.norm(s['p']-self.expected[n]['p']) <= self.c['maximum_settle_position_drift_m']
                        and angle(s['r'],self.expected[n]['r']) <= self.c['maximum_settle_orientation_drift_rad'] for n,s in states.items())
                    settle_contacts = contacts['bolt_fixture'] and contacts['tool_bolt']
                if quiet and near_reset and support_covered and not contacts['robot_target'] and settle_contacts:
                    if self.settle_anchor is None: self.settle_anchor = deepcopy(self.previous)
                    spread = all(np.linalg.norm(s['p']-self.settle_anchor[n]['p']) <= self.c['maximum_stable_position_range_m']
                        and angle(s['r'], self.settle_anchor[n]['r']) <= self.c['maximum_stable_angle_range_rad'] for n,s in states.items())
                    if spread: self.settle_elapsed += self.control
                    else: self.settle_elapsed, self.settle_anchor = 0., None
                else:
                    self.settle_elapsed, self.settle_anchor = 0., None
                if self.settle_elapsed+EPS >= self.c['settle_s']:
                    self.baseline = deepcopy(states)
                    self.last_yaw = yaw
                reason = 'passive_settled' if self.baseline is not None else 'waiting_for_passive_settlement'
                self.phase = 'advance' if self.baseline is not None else 'settle'
                self.reasons = [reason]
            else:
                delta = math.atan2(math.sin(yaw-self.last_yaw), math.cos(yaw-self.last_yaw))
                self.turn += self.c['tightening_yaw_sign']*delta
                self.last_yaw = yaw
                baseline_position = self._geometry(self.baseline['bolt'])[0]
                self.advance = float(baseline_position[2]-position[2])
                residual = abs(self.advance-self.c['thread_lead_m_per_revolution']*self.turn/(2*math.pi))
                self.max_residual = max(self.max_residual, residual)
                if residual > self.c['maximum_lead_residual_m'] or abs(self.turn) > self.c['maximum_episode_turn_rad']:
                    violation = violation or 'thread_rotation_advance_inconsistent'
                chain = contacts['hand_tool'] & contacts['tool_bolt'] & contacts['bolt_fixture']
                self.interaction_s += len(chain)*self.dt
                signed_delta = self.c['tightening_yaw_sign']*delta
                chain_valid = len(chain)/self.substeps >= self.c['minimum_motion_chain_fraction']
                # Credit only retained net advance transmitted through the tool.
                # Uncredited motion is recoverable, but *all* reverse motion pays
                # back credit; neither passive advance nor repeated regrasp cycles
                # can accumulate progress beyond the actual net bolt turn.
                if signed_delta < 0. or chain_valid:
                    self.contact_backed_turn += signed_delta
                self.contact_backed_turn = max(0., min(self.contact_backed_turn, max(0., self.turn)))
                if chain_valid and signed_delta > EPS:
                    self.motion_chain_verified = True
                target = (self.c['minimum_turn_rad'] <= self.turn <= self.c['maximum_target_turn_rad']
                    and self.advance >= self.c['thread_lead_m_per_revolution']*self.c['minimum_turn_rad']/(2*math.pi)
                    and self.contact_backed_turn+EPS >= self.c['minimum_turn_rad']
                    and self.max_residual <= self.c['maximum_lead_residual_m']
                    and self.interaction_s+EPS >= self.c['minimum_tool_interaction_s'] and self.motion_chain_verified)
                released = target and quiet and not contacts['robot_target'] and bool(contacts['bolt_fixture']) and bool(contacts['tool_bolt'])
                self.phase = 'release' if target else 'advance'
                if target:
                    self.reasons = []
                    if contacts['robot_target']: self.reasons.append('robot_contact_prevents_release')
                    if not contacts['tool_bolt']: self.reasons.append('tool_reengagement_required')
                    if not quiet: self.reasons.append('release_motion_resets_dwell')
                    if not self.reasons: self.reasons = ['release_stability_in_progress']
                elif not contacts['tool_bolt']:
                    self.reasons = ['tool_reengagement_required']
                elif self.turn > self.c['maximum_target_turn_rad']:
                    self.reasons = ['target_turn_exceeded_recoverably']
                elif signed_delta < -EPS:
                    self.reasons = ['reverse_motion_debits_credit']
                elif not chain_valid and signed_delta > EPS:
                    self.reasons = ['uncredited_motion']
                elif self.turn >= self.c['minimum_turn_rad'] and self.contact_backed_turn+EPS < self.c['minimum_turn_rad']:
                    self.reasons = ['contact_backed_advance_required']
                else:
                    self.reasons = ['additional_thread_advance_required']
                if released:
                    if self.release_anchor is None: self.release_anchor = deepcopy(self.previous)
                    spread = all(np.linalg.norm(s['p']-self.release_anchor[n]['p']) <= self.c['maximum_stable_position_range_m']
                        and angle(s['r'], self.release_anchor[n]['r']) <= self.c['maximum_stable_angle_range_rad'] for n,s in states.items())
                    if spread: self.release_elapsed += self.control
                    else:
                        self.release_elapsed, self.release_anchor = 0., None
                        self.reasons = ['release_envelope_resets_dwell']
                else:
                    self.release_elapsed, self.release_anchor = 0., None
                self.success = self.release_elapsed+EPS >= self.c['release_hold_s'] and not violation
                reason = 'success' if self.success else 'release_and_stabilize' if target else 'additional_thread_advance_required'
            self.previous, self.previous_time = states, time
            # Only commit a task violation after the complete observation and
            # every native row have passed structural/numeric validation.
            self.irrecoverable = violation
            return self._result(self.irrecoverable or reason)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
            self.invalid = str(exc)
            self.stop_state = 'safety_stop' if isinstance(exc, SafetyStop) else 'evaluator_invalid'
            self.stop_code = exc.reason_code if isinstance(exc, SafetyStop) else 'invalid_native_evidence'
            self.success = False
            return self._result('invalid_observation:'+self.invalid)

    def inspect_conditions(self, observation=None):
        """Read the last assessment without consuming a second contact window."""
        return deepcopy(self.last_result) if self.last_result is not None else self._result('not_sampled')


SequenceEvaluator = FastenerEvaluator
