"""Read-only contact/trajectory scoring for simple manipulation release v1.

Only observations from the trusted native worker are accepted by deployment.
Contact means a solved, positive normal impulse; object proximity, table support
and arm contact cannot substitute for hand interaction. No simulator is imported.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

VERSION = 'simple_manipulation_contact_substeps_v7'
CONTACT_SOURCE = 'PhysX stage-bound rigid contact tensor; normal impulses'
SAMPLING_PROTOCOL = 'native_rigid_contact_sampling_v1'
EPS = 1.e-9


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


def vector(value, length, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(label + ' must contain finite coordinates')
    return array


class SimpleTaskEvaluator:
    def __init__(self, task_spec):
        self.task_spec = deepcopy(task_spec)
        self.config = self.task_spec['evaluation']
        self.mode = self.config['mode']
        if self.mode not in ('push', 'lift_hold', 'pick_place', 'grasp_hold'):
            raise ValueError('Unknown simple manipulation mode')
        self.humanlike = None
        if 'humanlike_grasp' in self.config:
            # In deployment this relative dependency is copied and hashed beside
            # this evaluator, so a live project change cannot alter a release.
            from .humanlike import HumanlikeGraspEvaluator
            self.humanlike = HumanlikeGraspEvaluator(self.config['humanlike_grasp'])
        # A tabletop grasp is certified by the SHARPA humanlike contract or, for
        # hands without such a contract, by the opposed pad pinch-force rule.
        if self.mode == 'grasp_hold' and self.humanlike is None and self.config.get('minimum_pinch_normal_force_n') is None:
            raise ValueError('Tabletop grasp requires the humanlike contract or a pinch-force rule')
        self.object = self.config['object']
        self.expected_initial = vector(self.config['initial_position_m'], 3, 'initial_position_m')
        self.goal = vector(self.config['goal_position_m'], 3, 'goal_position_m')
        # Current scoring never estimates contact duration from observation dt.
        # Old task documents must use their frozen evaluator or a new release.
        for key in ('physics_dt_s', 'control_period_s'):
            if number(self.config.get(key), key) <= 0:
                raise ValueError(key + ' must be positive')
        self.physics_dt = self.config['physics_dt_s']
        self.control_period = self.config['control_period_s']
        self.substeps = round(self.control_period / self.physics_dt)
        if self.substeps < 1 or abs(self.substeps * self.physics_dt - self.control_period) > EPS:
            raise ValueError('Control period must contain complete physics substeps')
        if self.humanlike is not None:
            grasp = self.config['humanlike_grasp']
            if any(key in grasp and abs(grasp[key] - self.config[key]) > EPS
                   for key in ('physics_dt_s', 'control_period_s')):
                raise ValueError('Contact and humanlike native clocks disagree')
        for key in ('initial_tolerance_m', 'goal_xy_radius_m', 'goal_z_tolerance_m',
                    'minimum_displacement_m', 'minimum_lift_m', 'hold_s', 'lift_stage_hold_s',
                    'maximum_speed_m_s', 'maximum_angular_speed_rad_s', 'maximum_motion_speed_m_s',
                    'maximum_tilt_rad', 'minimum_contact_duration_s', 'max_observation_gap_s'):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(key + ' must be finite and positive')
        if self.humanlike is not None and self.mode in ('lift_hold', 'pick_place'):
            start = self.config.get('humanlike_lift_start_m')
            if not isinstance(start, (int, float)) or isinstance(start, bool) or not 0 < start < self.config['minimum_lift_m']:
                raise ValueError('humanlike_lift_start_m must be between zero and the task lift threshold')
        self.pinch_force = self.config.get('minimum_pinch_normal_force_n')
        if self.pinch_force is not None and (self.mode != 'grasp_hold' or isinstance(self.pinch_force, bool)
                or not isinstance(self.pinch_force, (int, float)) or not math.isfinite(self.pinch_force)
                or self.pinch_force <= 0):
            raise ValueError('minimum_pinch_normal_force_n applies only to grasp_hold and must be positive')
        # Pinch channels: thumb versus index+middle links of the task's hand
        # (default: the SHARPA W01 elastomer pads).
        links = self.config.get('pinch_links', {f: [f'right_{f}_elastomer'] for f in ('thumb', 'index', 'middle')})
        if (not isinstance(links, dict) or set(links) != {'thumb', 'index', 'middle'}
                or any(not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v) for v in links.values())
                or len({x for v in links.values() for x in v}) != sum(len(v) for v in links.values())):
            raise ValueError('pinch_links maps thumb, index and middle to disjoint link-name lists')
        self.pinch_links = {link: finger for finger, names in links.items() for link in names}
        if self.humanlike is not None and self.mode == 'push':
            hold = self.config.get('push_grasp_violation_hold_s')
            if not isinstance(hold, (int, float)) or isinstance(hold, bool) or not math.isfinite(hold) or hold <= 0:
                raise ValueError('push_grasp_violation_hold_s must be finite and positive')
        self.reset()

    def reset(self):
        self.previous_time = None
        self.previous_position = None
        self.initial_position = None
        self.contact_duration = 0.
        self.contact_intervals = 0
        self.contact_impulse = 0.
        self.contact_substeps = 0
        self.previous_native_step = None
        self.previous_native_time = None
        self.maximum_displacement = 0.
        self.maximum_lift = 0.
        self.hold_start = None
        self.lift_hold_start = None
        self.lift_completed = False
        self.transport_completed = False
        self.lift_completion_time = None
        self.invalid = None
        self.success = False
        self.last_result = None
        self.humanlike_result = None
        self.grasp_constraint_violated = False
        self.grasp_violation_time = None
        self.grasp_violation_reason = None
        self.carry_grasp_active = False
        self.push_invalid_grasp_duration = 0.
        self.non_hand_robot_contact_violated = False
        self.non_hand_robot_contact_time = None
        self.pinch_forces = []
        self.last_pinch_force = None
        if self.humanlike is not None:
            self.humanlike.reset()

    @staticmethod
    def _hand_actor(path):
        # Native rigid-body paths contain the URDF link name as final component.
        # Explicit hand namespaces exclude FR3 links and arbitrary context props.
        name = path.rsplit('/', 1)[-1]
        return path.startswith('/World/Robot/') and name.startswith(('right_', 'left_', 'allegro_', 'leap_', 'rh_'))

    def _sampling(self, observation, time, first):
        sampling = observation['contacts']['sampling']
        if (not isinstance(sampling, dict) or sampling.get('protocol') != SAMPLING_PROTOCOL
                or sampling.get('complete') is not True):
            raise ValueError('complete native contact sampling evidence is missing')
        if abs(number(sampling['physics_dt_s'], 'sampling physics dt') - self.physics_dt) > EPS:
            raise ValueError('native sampling physics dt differs from frozen task')
        start = number(sampling['start_sim_time_s'], 'sampling start')
        end = number(sampling['end_sim_time_s'], 'sampling end')
        native_start = number(sampling['native_start_time_s'], 'native start')
        native_end = number(sampling['native_end_time_s'], 'native end')
        if min(start, end, native_start, native_end) < -EPS:
            raise ValueError('native contact sampling clocks must be nonnegative')
        counts = [sampling[k] for k in ('sampled_substeps', 'native_start_step_index', 'native_end_step_index')]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('native contact sampling counters must be nonnegative integers')
        sampled, step_start, step_end = counts
        duration = number(observation['last_duration_s'], 'native window duration')
        expected = 0 if first else self.substeps
        if sampled != expected or step_end - step_start != expected:
            raise ValueError('native contact sampling does not cover every physics substep')
        expected_duration = 0. if first else self.control_period
        if (abs(end - time) > EPS or abs(end - start - expected_duration) > EPS
                or abs(duration - expected_duration) > EPS
                or abs(native_end - native_start - expected_duration) > EPS):
            raise ValueError('native contact sampling clocks disagree')
        if first:
            if abs(time) > EPS or observation['contacts']['rigid']:
                raise ValueError('initial contact baseline must be reset with no sampled window')
        elif (abs(start - self.previous_time) > EPS or step_start != self.previous_native_step
                or abs(native_start - self.previous_native_time) > EPS):
            raise ValueError('native contact sampling window is not consecutive')
        return start, step_end, native_end

    def _contacts(self, observation, time, first):
        contacts = observation['contacts']
        if not isinstance(contacts, dict):
            raise ValueError('native contact stream must be an object')
        if contacts.get('errors') != []:
            raise ValueError('native contact reporting errors')
        if contacts.get('source') != CONTACT_SOURCE:
            raise ValueError('native solver contact provenance is missing')
        reports = contacts['rigid']
        if not isinstance(reports, list):
            raise ValueError('native rigid contact stream is missing')
        start, native_step, native_time = self._sampling(observation, time, first)
        table, impulses = False, []
        pads = {'thumb': 0., 'index': 0., 'middle': 0.}
        seen, hand_ticks = set(), set()
        robot, non_hand_robot = False, False
        target = '/World/Objects/' + self.object
        for row in reports:
            if not isinstance(row, dict):
                raise ValueError('native contact row must be an object')
            actor0, actor1 = row['actor0'], row['actor1']
            if not isinstance(actor0, str) or not isinstance(actor1, str) or actor0 == actor1:
                raise ValueError('malformed contact actor')
            stamp = number(row['time_s'], 'contact timestamp')
            tick = round((stamp - start) / self.physics_dt)
            if not 1 <= tick <= self.substeps or abs(stamp - (start + tick * self.physics_dt)) > EPS:
                raise ValueError('stale, future or off-grid native contact sample')
            key = (tick, *sorted((actor0, actor1)))
            if key in seen:
                raise ValueError('duplicated native contact pair at one substep')
            seen.add(key)
            amount = number(row['impulse_magnitude_sum_ns'], 'contact impulse')
            number(row['min_separation_m'], 'contact separation')
            impulse_vector = vector(row['impulse_vector_ns'], 3, 'contact impulse vector')
            normal_error = number(row['normal_length_max_abs_error'], 'normal length error')
            points, nonzero = row['point_count'], row['nonzero_impulse_normal_count']
            if (amount < 0 or type(points) is not int or points <= 0 or type(nonzero) is not int
                    or not 0 <= nonzero <= points or not 0 <= normal_error <= 32 * 2**-23
                    or row.get('impulse_components') != 'normal_only'
                    or np.linalg.norm(impulse_vector) > amount + EPS or (amount > 0 and nonzero == 0)):
                raise ValueError('malformed native normal-impulse evidence')
            if amount <= 1.e-8:
                continue
            other = actor1 if actor0 == target else actor0 if actor1 == target else None
            if other is None:
                continue
            if self._hand_actor(other):
                hand_ticks.add(tick)
                impulses.append(amount)
                finger = self.pinch_links.get(other.rsplit('/', 1)[-1])
                if finger is not None:
                    pads[finger] += amount
            if other.startswith('/World/Robot/'):
                robot = True
                if not self._hand_actor(other):
                    non_hand_robot = True
            if other == '/World/Table':
                table = True
        impulse = math.fsum(impulses)
        if not math.isfinite(self.contact_impulse + impulse):
            raise ValueError('native contact impulse accumulation overflow')
        # Load-bearing pinch: the weaker of the thumb pad and the opposing
        # index+middle pads, as mean normal force over this control window.
        self.window_pinch_force = min(pads['thumb'], pads['index'] + pads['middle']) / self.control_period
        return bool(hand_ticks), table, impulse, robot, non_hand_robot, len(hand_ticks), native_step, native_time

    def _irrecoverable(self):
        """A latched condition under which this episode can no longer succeed."""
        if self.success:
            return None
        if self.non_hand_robot_contact_violated:
            return 'non_hand_robot_contact_violated'
        if self.grasp_constraint_violated:
            return 'grasp_constraint_violated:' + str(self.grasp_violation_reason)
        if self.initial_position is None:
            return None
        if self.mode == 'grasp_hold':
            if self.maximum_displacement > self.config['goal_xy_radius_m'] + EPS:
                return 'episode_displacement_limit_exceeded'
            if self.maximum_lift > self.config['goal_z_tolerance_m'] + EPS:
                return 'episode_lift_limit_exceeded'
        if self.mode == 'push' and self.maximum_lift > self.config['minimum_lift_m'] / 2.:
            return 'episode_lift_limit_exceeded'
        return None

    def _result(self, reason, time, **details):
        dwell = 0. if self.hold_start is None else max(0., time - self.hold_start)
        self.last_result = {'evaluator': VERSION, 'task_id': self.task_spec['id'],
            'success': self.success, 'reason': reason,
            'progress': 1. if self.success else .5 if self.lift_completed else 0.,
            'hold_elapsed_s': dwell, 'hold_required_s': self.config['hold_s'],
            'hand_contact_duration_s': self.contact_duration,
            'hand_contact_substeps': self.contact_substeps,
            'contact_duration_semantics': 'unique_positive_hand_contact_substeps_times_physics_dt',
            'hand_contact_intervals': self.contact_intervals,
            'hand_contact_normal_impulse_ns': self.contact_impulse,
            'maximum_displacement_m': self.maximum_displacement,
            'maximum_lift_m': self.maximum_lift, 'lift_stage_completed': self.lift_completed,
            'non_hand_robot_contact_violated': self.non_hand_robot_contact_violated,
            'non_hand_robot_contact_time_s': self.non_hand_robot_contact_time,
            'transport_stage_completed': self.transport_completed,
            'lift_completion_time_s': self.lift_completion_time,
            'irrecoverable_reason': None if self.invalid else self._irrecoverable(), **details}
        if self.humanlike is not None:
            self.last_result['humanlike_grasp'] = deepcopy(self.humanlike_result)
            self.last_result['grasp_constraint_violated'] = self.grasp_constraint_violated
            self.last_result['grasp_violation_time_s'] = self.grasp_violation_time
            self.last_result['grasp_violation_reason'] = self.grasp_violation_reason
            self.last_result['carry_grasp_active'] = self.carry_grasp_active
            if self.mode == 'push':
                self.last_result['push_invalid_grasp_duration_s'] = self.push_invalid_grasp_duration
        return deepcopy(self.last_result)

    def update(self, observation):
        time = number(observation['time_s'], 'observation timestamp')
        if not math.isfinite(time) or (self.previous_time is not None and time <= self.previous_time):
            raise ValueError('Observation timestamps must be finite and strictly increasing')
        if self.invalid:
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)
        first = self.previous_time is None
        dt = 0. if first else time - self.previous_time
        gap = not first and dt > self.config['max_observation_gap_s'] + EPS
        try:
            if observation.get('metrics', {}).get('state_finite') is not True:
                raise ValueError('native state finiteness is not established')
            obj = observation['objects'][self.object]
            position = vector(obj['position_m'], 3, 'position')
            velocity = vector(obj['linear_velocity_m_s'], 3, 'linear velocity')
            omega = vector(obj['angular_velocity_rad_s'], 3, 'angular velocity')
            q = vector(obj['quaternion_wxyz'], 4, 'orientation')
            if abs(np.linalg.norm(q) - 1.) > .01:
                raise ValueError('orientation quaternion must be normalized')
            q = q / np.linalg.norm(q)
            hand, table, impulse, robot, non_hand_robot, hand_steps, native_step, native_time = self._contacts(observation, time, first)
            if first:
                if np.linalg.norm(position - self.expected_initial) > self.config['initial_tolerance_m']:
                    raise ValueError('initial state differs from frozen task initialization')
                self.initial_position = position.copy()
            motion_speed = 0. if first else float(np.linalg.norm(position - self.previous_position) / dt)
            if not gap and max(motion_speed, np.linalg.norm(velocity)) > self.config['maximum_motion_speed_m_s']:
                raise ValueError('object trajectory exceeds frozen motion speed bound')
            if self.humanlike is not None:
                self.humanlike_result = self.humanlike.update(observation)
                if self.humanlike_result['valid'] is not True:
                    raise ValueError('humanlike grasp evidence: ' + self.humanlike_result['reason'])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.success = False
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)

        self.previous_time, self.previous_position = time, position.copy()
        self.previous_native_step, self.previous_native_time = native_step, native_time
        self.maximum_displacement = max(self.maximum_displacement, float(np.linalg.norm(position - self.initial_position)))
        lift = float(position[2] - self.initial_position[2])
        self.maximum_lift = max(self.maximum_lift, lift)
        if first:
            return self._result('baseline_recorded', time)
        # Dexterous manipulation must not be credited to an arm collision after
        # an unrelated hand touch. Preserve the violation even after release.
        # This is a new evaluator version; old releases retain their own code.
        if non_hand_robot and not self.non_hand_robot_contact_violated:
            self.non_hand_robot_contact_violated = True
            self.non_hand_robot_contact_time = time
        if gap:
            self.hold_start = self.lift_hold_start = None
            self.push_invalid_grasp_duration = 0.
            self.success = False
            if self.carry_grasp_active:
                self._grasp_violation(time, 'observation_gap_during_carry')
            return self._result('observation_gap:dwell_reset', time)
        if hand:
            self.contact_substeps += hand_steps
            self.contact_duration = self.contact_substeps * self.physics_dt
            self.contact_intervals += 1
            self.contact_impulse += impulse
        interacted = (self.contact_intervals >= 2 and
                      self.contact_duration + EPS >= self.config['minimum_contact_duration_s'])
        moved = self.maximum_displacement + EPS >= self.config['minimum_displacement_m']
        tilt = float(np.arccos(np.clip(1. - 2. * (q[1] ** 2 + q[2] ** 2), -1., 1.)))
        stable = bool(np.linalg.norm(velocity) <= self.config['maximum_speed_m_s'] + EPS and
                  motion_speed <= self.config['maximum_speed_m_s'] + EPS and
                  np.linalg.norm(omega) <= self.config['maximum_angular_speed_rad_s'] + EPS and
                  tilt <= self.config['maximum_tilt_rad'] + EPS)
        lifted = lift + EPS >= self.config['minimum_lift_m']
        humanlike_ok = self.humanlike is None or self.humanlike_result['passed'] is True
        if self.humanlike is not None and self.mode == 'push':
            if self.humanlike_result['grasp_attempted'] and not self.humanlike_result['instantaneous_pass']:
                self.push_invalid_grasp_duration += dt
                if self.push_invalid_grasp_duration + EPS >= self.config['push_grasp_violation_hold_s']:
                    self._grasp_violation(time, 'sustained_nonconforming_grasp_during_push')
            else:
                self.push_invalid_grasp_duration = 0.
        if self.humanlike is not None and self.mode in ('lift_hold', 'pick_place'):
            if table:
                self.carry_grasp_active = False
            else:
                if lift + EPS >= self.config['humanlike_lift_start_m'] and not self.carry_grasp_active:
                    self.carry_grasp_active = True
                    if not humanlike_ok:
                        self._grasp_violation(time, 'grasp_not_established_before_lift')
                if self.carry_grasp_active and self.humanlike_result['instantaneous_pass'] is not True:
                    self._grasp_violation(time, 'grasp_lost_during_carry:' + self.humanlike_result['reason'])
        grasp_hold = lifted and hand and not table and stable and interacted and moved and humanlike_ok
        if self.mode == 'pick_place' and not self.lift_completed:
            if grasp_hold:
                if self.lift_hold_start is None:
                    self.lift_hold_start = time
                if time - self.lift_hold_start + EPS >= self.config['lift_stage_hold_s']:
                    self.lift_completed, self.lift_completion_time = True, time
            else:
                self.lift_hold_start = None
        goal_geometry = bool(np.linalg.norm(position[:2] - self.goal[:2]) <= self.config['goal_xy_radius_m'] + EPS and
                         abs(position[2] - self.goal[2]) <= self.config['goal_z_tolerance_m'] + EPS)
        if (self.mode == 'pick_place' and self.lift_completed and hand and not table and humanlike_ok
                and lift >= self.config['minimum_lift_m'] / 2.
                and np.linalg.norm(position[:2] - self.goal[:2]) <= self.config['goal_xy_radius_m']):
            self.transport_completed = True
        released_place = goal_geometry and stable and table and not robot and interacted and moved
        if self.mode == 'push':
            good = released_place and self.maximum_lift <= self.config['minimum_lift_m'] / 2.
        elif self.mode == 'lift_hold':
            good = grasp_hold
        elif self.mode == 'grasp_hold':
            # This elementary precursor intentionally needs no lift or carry.
            # Table support and an opposed, stable grasp must coexist; merely
            # touching or pushing the cube is not a completed grasp.
            good = (goal_geometry and stable and hand and table and interacted and humanlike_ok
                    and self.maximum_lift <= self.config['goal_z_tolerance_m'] + EPS
                    and self.maximum_displacement <= self.config['goal_xy_radius_m'] + EPS)
            if self.pinch_force is not None:
                # A firm grasp must be able to carry the object by pad friction:
                # every window keeps half the required pinch, and the hold's
                # mean pinch reaches it. A light touch is not a grasp.
                self.last_pinch_force = self.window_pinch_force
                good = good and self.window_pinch_force + EPS >= .5 * self.pinch_force
        else:
            good = released_place and self.lift_completed and self.transport_completed
        good = good and not self.grasp_constraint_violated and not self.non_hand_robot_contact_violated
        if good:
            if self.hold_start is None:
                self.hold_start = time
                self.pinch_forces = []
            if self.pinch_force is not None:
                self.pinch_forces.append(self.window_pinch_force)
            self.success = (time - self.hold_start + EPS >= self.config['hold_s']
                            and (self.pinch_force is None
                                 or math.fsum(self.pinch_forces) / len(self.pinch_forces) + EPS >= self.pinch_force))
        else:
            self.hold_start = None
            self.pinch_forces = []
            self.success = False
        return self._result('non_hand_robot_contact_violated' if self.non_hand_robot_contact_violated else
            'grasp_constraint_violated' if self.grasp_constraint_violated else
            'success' if self.success else 'holding' if good else 'waiting_for_physical_conditions', time,
            hand_contact=hand, any_robot_contact=robot, non_hand_robot_contact=non_hand_robot,
            table_support_contact=table, stable=stable, goal_geometry=goal_geometry,
            minimum_motion_observed=moved, interaction_observed=interacted, tilt_rad=tilt,
            **({} if self.pinch_force is None else {
                'pinch_normal_force_n': self.last_pinch_force,
                'hold_mean_pinch_normal_force_n': (math.fsum(self.pinch_forces) / len(self.pinch_forces)
                                                   if self.pinch_forces else None)}))

    def _grasp_violation(self, time, reason):
        if not self.grasp_constraint_violated:
            self.grasp_violation_time = time
            self.grasp_violation_reason = reason
        self.grasp_constraint_violated = True

    def inspect_conditions(self, observation):
        """A copied scorer prevents diagnostics from advancing actual progress."""
        diagnostic = deepcopy(self)
        return [diagnostic.update(observation)] if self.previous_time is None else [deepcopy(self.last_result)]


class BimanualTaskEvaluator(SimpleTaskEvaluator):
    """Two-hand modes for the dual FR3 + SHARPA robot (same native evidence rules).

    bimanual_lift: one object lifted off the table and held stable while BOTH
      hands (left_* and right_* links) are in solver contact with it.
    dual_lift: each listed object lifted, stable and in contact with its own
      assigned hand only (the other hand never holds it).
    Arm (fr3_*) contact with any target latches an irrecoverable violation.
    """
    MODES = ('bimanual_lift', 'dual_lift')

    def __init__(self, task_spec):
        self.task_spec = deepcopy(task_spec)
        self.config = self.task_spec['evaluation']
        self.mode = self.config['mode']
        if self.mode not in self.MODES:
            raise ValueError('Unknown bimanual mode')
        self.humanlike = None
        self.pinch_force = None
        self.pinch_links = {}
        sides = self.config.get('hand_sides', {'left': ['left_'], 'right': ['right_']})
        if set(sides) != {'left', 'right'} or any(not v or not all(isinstance(x, str) and x for x in v) for v in sides.values()):
            raise ValueError('hand_sides maps left and right to link-name prefixes')
        self.hand_sides = {side: tuple(prefixes) for side, prefixes in sides.items()}
        if self.mode == 'bimanual_lift':
            self.objects = {self.config['object']: ('left', 'right')}
        else:
            assignment = self.config.get('objects')
            if not isinstance(assignment, dict) or not assignment or any(v not in ('left', 'right') for v in assignment.values()):
                raise ValueError('dual_lift maps each object to its holding hand')
            self.objects = {name: (side,) for name, side in assignment.items()}
        initial = self.config['initial_positions_m'] if self.mode == 'dual_lift' else {self.config['object']: self.config['initial_position_m']}
        if set(initial) != set(self.objects):
            raise ValueError('every scored object needs a frozen initial position')
        self.expected_initial = {name: vector(value, 3, 'initial position') for name, value in initial.items()}
        for key in ('physics_dt_s', 'control_period_s'):
            if number(self.config.get(key), key) <= 0:
                raise ValueError(key + ' must be positive')
        self.physics_dt = self.config['physics_dt_s']
        self.control_period = self.config['control_period_s']
        self.substeps = round(self.control_period / self.physics_dt)
        if self.substeps < 1 or abs(self.substeps * self.physics_dt - self.control_period) > EPS:
            raise ValueError('Control period must contain complete physics substeps')
        for key in ('initial_tolerance_m', 'minimum_lift_m', 'hold_s', 'maximum_speed_m_s', 'maximum_angular_speed_rad_s',
                    'maximum_motion_speed_m_s', 'maximum_tilt_rad', 'minimum_contact_duration_s', 'max_observation_gap_s'):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(key + ' must be finite and positive')
        self.reset()

    def reset(self):
        super().reset()
        self.per_object = {name: {'initial': None, 'previous': None, 'max_lift': 0., 'contact_substeps': {s: 0 for s in sides}}
                           for name, sides in self.objects.items()}

    def _side(self, path):
        if not self._hand_actor(path):
            return None
        name = path.rsplit('/', 1)[-1]
        for side, prefixes in self.hand_sides.items():
            if name.startswith(prefixes):
                return side
        return None

    def _object_contacts(self, observation, time, first):
        """Per object: contact substeps per hand side, table support, arm contact."""
        contacts = observation['contacts']
        if not isinstance(contacts, dict) or contacts.get('errors') != [] or contacts.get('source') != CONTACT_SOURCE:
            raise ValueError('native solver contact provenance is missing or reports errors')
        reports = contacts['rigid']
        if not isinstance(reports, list):
            raise ValueError('native rigid contact stream is missing')
        start, native_step, native_time = self._sampling(observation, time, first)
        targets = {'/World/Objects/' + name: name for name in self.objects}
        result = {name: {'sides': {'left': set(), 'right': set()}, 'table': False, 'arm': False, 'other_robot': False}
                  for name in self.objects}
        seen = set()
        for row in reports:
            actor0, actor1 = row['actor0'], row['actor1']
            if not isinstance(actor0, str) or not isinstance(actor1, str) or actor0 == actor1:
                raise ValueError('malformed contact actor')
            stamp = number(row['time_s'], 'contact timestamp')
            tick = round((stamp - start) / self.physics_dt)
            if not 1 <= tick <= self.substeps or abs(stamp - (start + tick * self.physics_dt)) > EPS:
                raise ValueError('stale, future or off-grid native contact sample')
            key = (tick, *sorted((actor0, actor1)))
            if key in seen:
                raise ValueError('duplicated native contact pair at one substep')
            seen.add(key)
            amount = number(row['impulse_magnitude_sum_ns'], 'contact impulse')
            impulse_vector = vector(row['impulse_vector_ns'], 3, 'contact impulse vector')
            points, nonzero = row['point_count'], row['nonzero_impulse_normal_count']
            if (amount < 0 or type(points) is not int or points <= 0 or type(nonzero) is not int
                    or not 0 <= nonzero <= points or row.get('impulse_components') != 'normal_only'
                    or np.linalg.norm(impulse_vector) > amount + EPS or (amount > 0 and nonzero == 0)):
                raise ValueError('malformed native normal-impulse evidence')
            if amount <= 1.e-8:
                continue
            for target, other in ((actor0, actor1), (actor1, actor0)):
                name = targets.get(target)
                if name is None:
                    continue
                side = self._side(other)
                if side is not None:
                    result[name]['sides'][side].add(tick)
                elif other.startswith('/World/Robot/'):
                    result[name]['arm'] = True
                elif other == '/World/Table':
                    result[name]['table'] = True
        return result, native_step, native_time

    def update(self, observation):
        time = number(observation['time_s'], 'observation timestamp')
        if self.previous_time is not None and time <= self.previous_time:
            raise ValueError('Observation timestamps must be finite and strictly increasing')
        if self.invalid:
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)
        first = self.previous_time is None
        dt = 0. if first else time - self.previous_time
        gap = not first and dt > self.config['max_observation_gap_s'] + EPS
        try:
            if observation.get('metrics', {}).get('state_finite') is not True:
                raise ValueError('native state finiteness is not established')
            contacts, native_step, native_time = self._object_contacts(observation, time, first)
            states = {}
            for name in self.objects:
                obj = observation['objects'][name]
                position = vector(obj['position_m'], 3, 'position')
                velocity = vector(obj['linear_velocity_m_s'], 3, 'linear velocity')
                omega = vector(obj['angular_velocity_rad_s'], 3, 'angular velocity')
                q = vector(obj['quaternion_wxyz'], 4, 'orientation')
                if abs(np.linalg.norm(q) - 1.) > .01:
                    raise ValueError('orientation quaternion must be normalized')
                q = q / np.linalg.norm(q)
                track = self.per_object[name]
                if first and np.linalg.norm(position - self.expected_initial[name]) > self.config['initial_tolerance_m']:
                    raise ValueError('initial state differs from frozen task initialization')
                speed = 0. if first else float(np.linalg.norm(position - track['previous']) / dt)
                if not gap and max(speed, np.linalg.norm(velocity)) > self.config['maximum_motion_speed_m_s']:
                    raise ValueError('object trajectory exceeds frozen motion speed bound')
                states[name] = (position, velocity, omega, q, speed)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.success = False
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)
        self.previous_time = time
        self.previous_native_step, self.previous_native_time = native_step, native_time
        details = {}
        good_all = True
        for name, (position, velocity, omega, q, speed) in states.items():
            track = self.per_object[name]
            if first:
                track['initial'] = position.copy()
            track['previous'] = position.copy()
            lift = float(position[2] - track['initial'][2])
            track['max_lift'] = max(track['max_lift'], lift)
            row = contacts[name]
            if row['arm'] and not self.non_hand_robot_contact_violated:
                self.non_hand_robot_contact_violated = True
                self.non_hand_robot_contact_time = time
            for side in self.objects[name]:
                track['contact_substeps'][side] += len(row['sides'][side])
            holders = {side for side in ('left', 'right') if row['sides'][side]}
            required = set(self.objects[name])
            tilt = float(np.arccos(np.clip(1. - 2. * (q[1] ** 2 + q[2] ** 2), -1., 1.)))
            stable = bool(np.linalg.norm(velocity) <= self.config['maximum_speed_m_s'] + EPS
                          and speed <= self.config['maximum_speed_m_s'] + EPS
                          and np.linalg.norm(omega) <= self.config['maximum_angular_speed_rad_s'] + EPS
                          and tilt <= self.config['maximum_tilt_rad'] + EPS)
            interacted = all(track['contact_substeps'][side] * self.physics_dt + EPS >= self.config['minimum_contact_duration_s']
                             for side in required)
            lifted = lift + EPS >= self.config['minimum_lift_m']
            # dual_lift: only the assigned hand may hold its object.
            held = required <= holders and (self.mode == 'bimanual_lift' or holders == required)
            good = (not first) and lifted and held and not row['table'] and stable and interacted
            good_all = good_all and good
            details[name] = {'lift_m': lift, 'maximum_lift_m': track['max_lift'], 'holding_hands': sorted(holders),
                             'required_hands': sorted(required), 'table_support_contact': row['table'],
                             'stable': stable, 'tilt_rad': tilt, 'lifted': lifted, 'held': held, 'good': good}
        self.maximum_lift = max(d['maximum_lift_m'] for d in details.values())
        self.lift_completed = all(d['lifted'] for d in details.values())
        if gap:
            self.hold_start = None
            self.success = False
            return self._result('observation_gap:dwell_reset', time, objects=details)
        good_all = good_all and not self.non_hand_robot_contact_violated
        if good_all:
            if self.hold_start is None:
                self.hold_start = time
            self.success = time - self.hold_start + EPS >= self.config['hold_s']
        else:
            self.hold_start = None
            self.success = False
        return self._result('non_hand_robot_contact_violated' if self.non_hand_robot_contact_violated else
                            'success' if self.success else 'holding' if good_all else 'waiting_for_physical_conditions',
                            time, objects=details)

    def _irrecoverable(self):
        if self.success:
            return None
        return 'non_hand_robot_contact_violated' if self.non_hand_robot_contact_violated else None


class SoftBodyTaskEvaluator:
    """Deformable (sponge) modes scored from the simulated body state only.

    PhysX reports no deformable contact impulses, so these modes never claim
    hand contact; they score the body's measured deformation and pose:
    soft_press: compressed to <= press_height_fraction of its nominal height
      for hold_s, then released (>= recovered_height_fraction, at rest) for
      release_hold_s; never lifted or displaced beyond the episode limits.
    soft_push: centroid within goal_xy_radius_m of the goal, uncompressed,
      on the table and at rest for hold_s.

    Rest (``rest_speed_source``): the default 'instantaneous_max_node_speed'
    (v6) compares PhysX's per-node velocity with maximum_rest_speed_m_s. Those
    velocities carry table-contact solver jitter (an untouched sponge reports
    0.04-0.09 m/s while its nodes move < 3 mm/s), so v6 rest was unattainable.
    'windowed_body_state_v1' (v7 opt-in) uses the body state instead: the
    largest change of centroid or bounds over the trailing rest_window_s,
    divided by the elapsed time. Without a full window the body is not at rest.
    """
    MODES = ('soft_press', 'soft_push')
    REST_SOURCES = ('instantaneous_max_node_speed', 'windowed_body_state_v1')

    def __init__(self, task_spec):
        self.task_spec = deepcopy(task_spec)
        self.config = self.task_spec['evaluation']
        self.mode = self.config['mode']
        if self.mode not in self.MODES:
            raise ValueError('Unknown soft-body mode')
        self.body = self.config['soft_body']
        self.initial = vector(self.config['initial_centroid_m'], 3, 'initial centroid')
        self.goal = vector(self.config.get('goal_position_m', self.config['initial_centroid_m']), 3, 'goal')
        self.rest_source = self.config.get('rest_speed_source', 'instantaneous_max_node_speed')
        if self.rest_source not in self.REST_SOURCES:
            raise ValueError('Unknown soft-body rest speed source')
        keys = ['nominal_height_m', 'table_top_z_m', 'hold_s', 'maximum_rest_speed_m_s', 'max_observation_gap_s',
                'initial_tolerance_m', 'maximum_motion_speed_m_s', 'maximum_lift_m']
        keys += (['press_height_fraction', 'recovered_height_fraction', 'release_hold_s', 'maximum_displacement_m']
                 if self.mode == 'soft_press' else ['goal_xy_radius_m', 'recovered_height_fraction'])
        keys += ['rest_window_s'] if self.rest_source == 'windowed_body_state_v1' else []
        for key in keys:
            if number(self.config.get(key), key) <= 0:
                raise ValueError(key + ' must be finite and positive')
        self.reset()

    def _rest_speed(self, time, centroid, low, high, instantaneous):
        if self.rest_source == 'instantaneous_max_node_speed':
            return instantaneous
        state = np.concatenate([centroid, low, high])
        self.rest_history.append((time, state))
        window = self.config['rest_window_s']
        while len(self.rest_history) > 1 and self.rest_history[1][0] <= time - window + EPS:
            self.rest_history.pop(0)
        start, reference = self.rest_history[0]
        if time - start + EPS < window:
            return None
        return float(np.abs(state - reference).max() / (time - start))

    def reset(self):
        self.rest_history = []
        self.previous_time = self.previous_centroid = None
        self.invalid = None
        self.success = False
        self.press_start = self.press_completion_time = self.release_start = self.hold_start = None
        self.press_completed = False
        self.maximum_displacement = self.maximum_lift = 0.
        self.minimum_height_ratio = math.inf
        self.violation = None
        self.last_result = None

    def _irrecoverable(self):
        return None if self.success else self.violation

    def _result(self, reason, time, **details):
        self.last_result = {'evaluator': VERSION, 'task_id': self.task_spec['id'], 'success': self.success,
            'reason': reason, 'progress': 1. if self.success else .5 if self.press_completed else 0.,
            'maximum_displacement_m': self.maximum_displacement, 'maximum_lift_m': self.maximum_lift,
            'minimum_height_ratio': None if math.isinf(self.minimum_height_ratio) else self.minimum_height_ratio,
            'press_completed': self.press_completed, 'press_completion_time_s': self.press_completion_time,
            'irrecoverable_reason': None if self.invalid else self._irrecoverable(),
            'contact_evidence': 'unavailable_for_deformables; body state only', **details}
        return deepcopy(self.last_result)

    def update(self, observation):
        time = number(observation['time_s'], 'observation timestamp')
        if self.previous_time is not None and time <= self.previous_time:
            raise ValueError('Observation timestamps must be finite and strictly increasing')
        if self.invalid:
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)
        first = self.previous_time is None
        dt = 0. if first else time - self.previous_time
        gap = not first and dt > self.config['max_observation_gap_s'] + EPS
        try:
            if observation.get('metrics', {}).get('state_finite') is not True:
                raise ValueError('native state finiteness is not established')
            body = observation['soft_bodies'][self.body]
            centroid = vector(body['centroid_m'], 3, 'soft body centroid')
            low, high = vector(body['bounds_min_m'], 3, 'soft body bounds'), vector(body['bounds_max_m'], 3, 'soft body bounds')
            speed = number(body['max_node_speed_m_s'], 'soft body node speed')
            if first and np.linalg.norm(centroid - self.initial) > self.config['initial_tolerance_m']:
                raise ValueError('initial state differs from frozen task initialization')
            motion = 0. if first else float(np.linalg.norm(centroid - self.previous_centroid) / dt)
            if not gap and motion > self.config['maximum_motion_speed_m_s']:
                raise ValueError('object trajectory exceeds frozen motion speed bound')
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.success = False
            self.previous_time = time
            return self._result('invalid_observation:' + self.invalid, time)
        self.previous_time, self.previous_centroid = time, centroid.copy()
        ratio = float((high[2] - low[2]) / self.config['nominal_height_m'])
        self.minimum_height_ratio = min(self.minimum_height_ratio, ratio)
        displacement = float(np.linalg.norm(centroid[:2] - self.initial[:2]))
        self.maximum_displacement = max(self.maximum_displacement, displacement)
        lift = float(low[2] - self.config['table_top_z_m'])
        self.maximum_lift = max(self.maximum_lift, lift)
        if self.violation is None and lift > self.config['maximum_lift_m'] + EPS:
            self.violation = 'soft_body_lifted_off_table'
        if (self.violation is None and self.mode == 'soft_press'
                and displacement > self.config['maximum_displacement_m'] + EPS):
            self.violation = 'episode_displacement_limit_exceeded'
        if gap:
            self.rest_history = []
        rest_speed = self._rest_speed(time, centroid, low, high, speed)
        details = {'height_ratio': ratio, 'displacement_m': displacement, 'lift_m': lift, 'max_node_speed_m_s': speed,
                   'rest_speed_m_s': rest_speed, 'rest_speed_source': self.rest_source}
        if gap:
            self.press_start = self.release_start = self.hold_start = None
            self.success = False
            return self._result('observation_gap:dwell_reset', time, **details)
        at_rest = rest_speed is not None and rest_speed <= self.config['maximum_rest_speed_m_s'] + EPS
        if self.violation is not None:
            self.success = False
            return self._result(self.violation, time, **details)
        if self.mode == 'soft_press':
            if not self.press_completed:
                if ratio <= self.config['press_height_fraction'] + EPS:
                    self.press_start = time if self.press_start is None else self.press_start
                    if time - self.press_start + EPS >= self.config['hold_s']:
                        self.press_completed, self.press_completion_time = True, time
                else:
                    self.press_start = None
                good = False
            else:
                good = ratio + EPS >= self.config['recovered_height_fraction'] and at_rest
                if good:
                    self.release_start = time if self.release_start is None else self.release_start
                    self.success = time - self.release_start + EPS >= self.config['release_hold_s']
                else:
                    self.release_start = None
        else:
            in_goal = float(np.linalg.norm(centroid[:2] - self.goal[:2])) <= self.config['goal_xy_radius_m'] + EPS
            good = (in_goal and at_rest and ratio + EPS >= self.config['recovered_height_fraction']
                    and lift <= self.config['maximum_lift_m'] + EPS)
            details['goal_xy_error_m'] = float(np.linalg.norm(centroid[:2] - self.goal[:2]))
            if good:
                self.hold_start = time if self.hold_start is None else self.hold_start
                self.success = time - self.hold_start + EPS >= self.config['hold_s']
            else:
                self.hold_start = None
                self.success = False
        reason = 'success' if self.success else 'holding' if good else (
            'pressed_waiting_for_release' if self.press_completed else 'waiting_for_physical_conditions')
        return self._result(reason, time, **details)

    def inspect_conditions(self, observation):
        diagnostic = deepcopy(self)
        return [diagnostic.update(observation)] if self.previous_time is None else [deepcopy(self.last_result)]


def SequenceEvaluator(task_spec):
    """Native worker release interface; the file is frozen as sequence_evaluator.py."""
    mode = task_spec['evaluation']['mode']
    if mode in BimanualTaskEvaluator.MODES:
        return BimanualTaskEvaluator(task_spec)
    if mode in SoftBodyTaskEvaluator.MODES:
        return SoftBodyTaskEvaluator(task_spec)
    return SimpleTaskEvaluator(task_spec)
