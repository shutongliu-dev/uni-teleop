"""Complete-stream private native referee; no physical writes or action policy."""
from collections import deque
from copy import deepcopy
import math
import numpy as np

if __package__ == 'scenes.workrooms.mechanical_assembly':
    from .nist_rules import digest, evaluation_configuration, public_rules
    from .nist_native_evidence import (
        EPS, Contacts, angle, distance, enclosed, fixed, local, number, points,
        rotation, route_in_order, simple_loop, state,
    )
else:
    # Frozen release loaders rename these three files without live dependencies.
    from .rules import digest, evaluation_configuration, public_rules
    from .native_evidence import (
        EPS, Contacts, angle, distance, enclosed, fixed, local, number, points,
        rotation, route_in_order, simple_loop, state,
    )

VERSION = 'mechanical_nist_sequence_evaluator.v1'
STATE_SCHEMA = 'mechanical_nist_native_state.v1'


def configuration(task):
    if task.get('schema_version') != 'mechanical-nist-task-v1':
        raise ValueError('Expected mechanical-nist-task-v1')
    c = deepcopy(task['evaluation'])
    expected = evaluation_configuration(task['id'], c['metadata'], task['episode_seed'],
        c['physics_dt_s'], c['room_support_paths'])
    if c != expected or task['public_rules'] != public_rules(task['id'], task['episode_seed']):
        raise ValueError('Frozen NIST rules or metadata differ')
    return c, digest(c)


class NativeTaskObserver:
    """Consume every native frame, including transient loss of release/support."""
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.task_id, self.meta = self.c['task_id'], self.c['metadata']
        self.limits, self.p = self.c['thresholds'], self.meta['task_parameters']
        self.specs = {row['id']: row for row in self.meta['objects']}
        self.fixed = {name: fixed(row['initial_pose_xyzw'])
                      for name, row in self.meta['fixture_metadata'].items()}
        self.paths = {**self.meta['actors'], **self.meta['fixtures']}
        for role, paths_ in self.c['room_support_paths'].items():
            for i, path in enumerate(paths_):
                self.paths['environment_'+role+'_'+str(i)] = path.replace(
                    '/World/Geometry/', '/World/Room/Geometry/', 1)
        self.flexible_ids = {name for group in self.p.get('flexible_groups', [])
                             for name in group['actor_ids']}
        self.twist_free_ids = self.flexible_ids-{
            goal['actor'] for goal in self.p.get('seating_goals', [])}
        self.reset()

    def reset(self):
        self.previous = self.initial = self.current = None
        self.previous_time = self.previous_native_time = self.previous_step = None
        self.error = self.failure = self.safety = None
        self.stable_s = self.assembly_stable_s = 0.
        self.stable_anchor = self.assembly_anchor = None
        self.assembled = False
        self.events, self.moved = [], set()
        self.witnesses, self.mating_witnesses, self.support_diagnostics = {}, {}, {}
        self.pose_steps = round(self.limits['pose_window_s']/self.c['physics_dt_s'])
        self.history = deque(maxlen=self.pose_steps+1)
        self.quiet, self.pose_speed = {}, {}

    def _event(self, kind, frame):
        self.events.append(dict(id=len(self.events)+1, kind=kind,
            time_s=frame['time_s'], native_step=frame['native_step']))

    def _witness(self, name, other, objects, contacts, t, witnesses, present, geometry):
        key = (name, other)
        if not geometry:
            witnesses.pop(key, None)
            return False
        obj, support = objects[name], objects.get(other, self.fixed.get(other))
        relative = local(obj['position_m'], support) if support else np.asarray(obj['position_m'])
        if present:
            witnesses[key] = (t, deepcopy(obj), relative.copy())
            return True
        prior = witnesses.get(key)
        if prior is None:
            return False
        unchanged = distance(obj, prior[1]) <= .001 and (
            name in self.flexible_ids or angle(obj, prior[1]) <= .03)
        unforced = (name not in contacts.robot
                    and np.linalg.norm(obj['linear_velocity_m_s']) <= .04)
        matching = np.linalg.norm(relative-prior[2]) <= .001
        return bool(matching and unforced and (
            t-prior[0] <= self.limits['support_memory_s']+EPS or unchanged))

    def _supported(self, name, other, objects, contacts, t, geometry=True):
        current = (name, other) in contacts.upward
        result = self._witness(name, other, objects, contacts, t, self.witnesses, current, geometry)
        if result:
            self.support_diagnostics[(name, other)] = (
                'native_upward_contact' if current else 'prior_loaded_contact_unchanged_or_recent')
        return result

    def _mating(self, name, other, objects, contacts, t, geometry):
        return self._witness(name, other, objects, contacts, t, self.mating_witnesses,
            frozenset((name, other)) in contacts.touching, geometry)

    def _seat(self, goal, objects, contacts, t, flexible_evidence=None):
        name, obj = goal['actor'], objects[goal['actor']]
        scoring_point = (np.asarray(obj['position_m'])+
            rotation(obj['quaternion_wxyz']) @ np.asarray(goal.get('actor_local_point_m', [0., 0., 0.])))
        # The target is the world location of the declared mating point, not a
        # second local offset applied to a possibly remote CAD body origin.
        position = np.abs(scoring_point-goal['target_position_m'])
        target = {'quaternion_wxyz': goal['target_quaternion_wxyz']}
        if goal.get('orientation_axis_only', False):
            axis = np.asarray(goal.get('orientation_axis_local', [0., 0., 1.]), float)
            norm = np.linalg.norm(axis)
            if norm <= EPS:
                raise ValueError('Seat orientation axis must be nonzero')
            axis /= norm
            a = rotation(obj['quaternion_wxyz']) @ axis
            b = rotation(target['quaternion_wxyz']) @ axis
            angular = math.acos(float(np.clip(np.dot(a, b), -1., 1.)))
        else:
            angular = angle(obj, target)
        geometry = bool(np.all(position <= np.asarray(goal['position_tolerance_m'])+EPS)
                        and angular <= goal['orientation_tolerance_rad']+EPS)
        supports = [self._supported(name, support, objects, contacts, t, geometry)
                    for support in goal['support_ids']]
        mating = [self._mating(name, other, objects, contacts, t, geometry)
                  for other in goal['contact_ids']]
        cable = (flexible_evidence or {}).get(goal.get('support_via_flexible_group'), {})
        indirect = bool(cable.get('intact') and cable.get('support'))
        supported = any(supports) or indirect
        return bool(geometry and supported and any(mating)), {
            'geometry': geometry, 'support': supported, 'direct_support': any(supports),
            'support_via_flexible_group': indirect, 'mating_contact': any(mating),
            'scoring_point_world_m': scoring_point.tolist(),
            'position_error_m': position.tolist(), 'orientation_error_rad': angular}

    def _group(self, group, objects, contacts, t):
        ids = group['actor_ids']
        offsets = group.get('centerline_offsets_m', [[0., 0., 0.]]*len(ids))
        curve = np.asarray([np.asarray(objects[name]['position_m'])+
            rotation(objects[name]['quaternion_wxyz']) @ np.asarray(offset)
            for name, offset in zip(ids, offsets)])
        traversed = np.concatenate((curve, curve[:1])) if group['closed'] else curve
        distances = np.linalg.norm(np.diff(traversed, axis=0), axis=1)
        intact = bool(np.max(distances) <= group['maximum_neighbor_distance_m']+EPS)
        route, hits = route_in_order(traversed, group.get('route_gates', []))
        if not route and group.get('allow_reverse_route', False):
            route, hits = route_in_order(traversed[::-1], group['route_gates'])
        loop = (not group['closed'] or simple_loop(curve)) and all(
            enclosed(curve, target, group.get('plane_tolerance_m', .02))
            for target in group.get('enclosed_targets', []))
        perimeter = float(distances.sum())
        length = group.get('perimeter_range_m')
        length_ok = length is None or length[0]-EPS <= perimeter <= length[1]+EPS
        supported = [name for name in ids if any(self._supported(
            name, other, objects, contacts, t) for other in group['support_ids'])]
        minimum = max(1, math.ceil(len(ids)*group.get('minimum_support_fraction', .04)))
        support_ok = len(supported) >= minimum
        # Support on the workplate does not establish engagement with a wheel.
        # Require an independent native witness for every declared mating body.
        mating = {other: any(self._mating(name, other, objects, contacts, t, True)
                            for name in ids)
                  for other in group.get('required_mating_contact_ids', [])}
        return bool(intact and route and loop and length_ok and support_ok and all(mating.values())), {
            'intact': intact, 'ordered_route': route, 'route_intersections': hits,
            'enclosed_targets': loop, 'perimeter_m': perimeter, 'length_ok': bool(length_ok),
            'supported_body_count': len(supported), 'minimum_supported_body_count': minimum,
            'support': support_ok, 'required_mating_contacts': mating}

    def _assembly_goal(self, objects, contacts, t):
        seats, groups = {}, {}
        good = True
        for group in self.p.get('flexible_groups', []):
            passed, detail = self._group(group, objects, contacts, t)
            groups[group['id']] = dict(complete=passed, **detail)
            good &= passed
        for goal in self.p.get('seating_goals', []):
            passed, detail = self._seat(goal, objects, contacts, t, groups)
            seats[goal['id']] = dict(complete=passed, **detail)
            good &= passed
        movement = set(self.p['required_moved_ids']) <= self.moved
        return bool(good and movement), {
            'seating': seats, 'flexible_groups': groups, 'required_movement': movement}

    def _removal_goal(self, objects, contacts, t):
        goal = self.p['removal']
        contained = {name: bool(np.all(np.abs(points(objects[name],
            self.specs[name]['local_bounds_m'])-goal['center_m']) <=
            np.asarray(goal['half_extents_m'])+self.limits['placement_tolerance_m']))
            for name in goal['actor_ids']}
        # Each independent cable must be supported. A supported red lead cannot
        # give an airborne blue lead credit, while a single lead may bridge gaps.
        groups = self.p.get('flexible_groups', [])
        support = {group['id']: any(self._supported(name, other, objects, contacts, t,
            contained[name]) for name in group['actor_ids'] for other in goal['support_ids'])
            for group in groups}
        grouped = {name for group in groups for name in group['actor_ids']}
        support.update({name: any(self._supported(name, other, objects, contacts, t,
            contained[name]) for other in goal['support_ids'])
            for name in set(goal['actor_ids'])-grouped})
        # Keep the chain connectivity requirement after removal as well.
        intact = {group['id']: self._group(group, objects, contacts, t)[1]['intact'] for group in groups}
        return bool(all(contained.values()) and all(support.values()) and all(intact.values())), {
            'contained': contained, 'tray_support': support, 'intact': intact}

    def _quiet(self, objects, contacts):
        self.history.append(objects)
        ready = len(self.history) == self.pose_steps+1
        for name, obj in objects.items():
            quiet = False
            if ready:
                prior = self.history[0][name]
                linear = distance(obj, prior)/self.limits['pose_window_s']
                angular = angle(obj, prior)/self.limits['pose_window_s']
                self.pose_speed[name] = dict(linear_m_s=linear, angular_rad_s=angular)
                # Circular flexible links may twist about their own axes at rest.
                # Centerline drift and complete support/route still apply.
                quiet = (linear <= self.limits['maximum_linear_speed_m_s']
                    and (name in self.twist_free_ids
                         or angular <= self.limits['maximum_angular_speed_rad_s']))
            self.quiet[name] = bool(quiet and name not in contacts.robot)

    def _dwell(self, condition, objects, duration, anchor):
        if not condition:
            return 0., None
        if anchor is None:
            anchor = deepcopy(objects)
        steady = all(distance(obj, anchor[name]) <= self.limits['position_drift_m']
            and (name in self.twist_free_ids or angle(obj, anchor[name]) <=
                 self.limits['orientation_drift_rad']) for name, obj in objects.items())
        if not steady:
            return 0., None
        return duration+self.c['physics_dt_s'], anchor

    def on_frame(self, frame):
        if self.error is not None:
            return
        try:
            self._frame(frame)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.error = str(exc)

    def _frame(self, frame):
        t, nt = number(frame['time_s']), number(frame['native_time_s'])
        step, baseline, dt = frame['native_step'], frame['baseline'], self.c['physics_dt_s']
        first = self.previous is None
        if (type(step) is not int or step < 0 or type(baseline) is not bool or baseline != first
                or t < 0 or nt < 0 or abs(number(frame['physics_dt_s'])-dt) > EPS):
            raise ValueError('Invalid native baseline or clock')
        if first:
            if abs(t) > EPS or frame['contacts']:
                raise ValueError('Unscored contact-free reset required')
        elif (step != self.previous_step+1 or abs(t-self.previous_time-dt) > EPS
              or abs(nt-self.previous_native_time-dt) > 2.e-6):
            raise ValueError('Missing or out-of-order native samples')
        if set(frame['objects']) != set(self.specs):
            raise ValueError('Native body inventory differs')
        objects = {name: state(frame['objects'][name]) for name in self.specs}
        if first:
            for name, obj in objects.items():
                reset = fixed(self.specs[name]['initial_pose_xyzw'])
                if distance(obj, reset) > .003 or angle(obj, reset) > .03:
                    raise ValueError('Native reset differs from frozen asset')
            self.initial = deepcopy(objects)
        contacts = Contacts(frame['contacts'], self.paths, self.specs, t,
                            self.limits['minimum_contact_impulse_ns'],
                            self.limits['maximum_mating_gap_m'])
        if contacts.minimum_separation < -self.limits['maximum_penetration_m']:
            self.safety = 'physics_safety_stop'
        if not first and (contacts.floor or any(
                obj['position_m'][2] < self.limits['fallen_height_m'] for obj in objects.values())):
            self.failure = 'object_fell'
        self.support_diagnostics = {}
        for name, obj in objects.items():
            if distance(obj, self.initial[name]) >= self.limits['minimum_movement_m']:
                self.moved.add(name)
        self._quiet(objects, contacts)
        assembly, detail = self._assembly_goal(objects, contacts, t)
        quiet = bool(not first and not contacts.robot and all(self.quiet.values())
                     and not self.failure and not self.safety)
        newly_assembled = False
        if self.task_id == 'nist_wire_harness':
            if not self.assembled:
                self.assembly_stable_s, self.assembly_anchor = self._dwell(
                    assembly and quiet, objects, self.assembly_stable_s, self.assembly_anchor)
                if self.assembly_stable_s+EPS >= self.limits['stable_s']:
                    self.assembled = newly_assembled = True
                    self._event('harness_assembled_released', frame)
            removal, removal_detail = self._removal_goal(objects, contacts, t)
            goal = self.assembled and removal and not newly_assembled
            detail.update(assembly_confirmed=self.assembled,
                assembly_released_stable_s=self.assembly_stable_s, removal=removal_detail)
        else:
            goal = assembly
        self.stable_s, self.stable_anchor = self._dwell(
            goal and quiet, objects, self.stable_s, self.stable_anchor)
        phase = 'settling' if goal else 'removing' if self.assembled else 'working'
        reason = ('reset_baseline_not_scored' if first else
            'harness_ready_for_removal' if newly_assembled else
            'waiting_for_release_and_stability' if goal else 'task_in_progress')
        self.current = {
            'schema': STATE_SCHEMA, 'configuration_sha256': self.digest, 'task_id': self.task_id,
            'time_s': t, 'native_time_s': nt, 'native_step': step, 'baseline': baseline,
            'phase': phase, 'reason': reason, 'released_stable_s': self.stable_s,
            'terminal_failure': self.failure, 'safety_stop': self.safety, 'detail': detail,
            'native_support_by_actor': {name: {
                'supported': any(k[0] == name for k in self.support_diagnostics),
                'released_quiet': self.quiet[name], 'rolling_pose_speed': self.pose_speed.get(name),
                'witnesses': [{'support': k[1], 'source': v} for k, v in self.support_diagnostics.items()
                              if k[0] == name]} for name in objects}}
        self.previous, self.previous_time = objects, t
        self.previous_native_time, self.previous_step = nt, step

    def on_observe(self):
        result = deepcopy(self.current or {'schema': STATE_SCHEMA, 'configuration_sha256': self.digest})
        result.update(complete=self.current is not None and self.error is None,
                      errors=[] if self.error is None else [self.error], events=deepcopy(self.events))
        return result


class SequenceEvaluator:
    """Score authenticated native receipts; missing evidence stays unknown."""
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.rules = public_rules(self.c['task_id'], self.c['episode_seed'])['referee']
        self.reset()

    def reset(self):
        self.previous_time = self.previous_step = None
        self.invalid = self.failure = self.safety = None
        self.events, self.detail = [], {}
        self.stable_s = 0.
        self.success = self.incomplete = False
        self.phase = 'working'
        self.last_result = self._result('task_in_progress')

    def _result(self, reason):
        name = ('evaluator_invalid' if self.invalid else 'safety_stop' if self.safety else
                'task_failure' if self.failure else 'incomplete' if self.incomplete else
                'success' if self.success else 'ongoing')
        physical_state = 'ongoing' if self.incomplete else name
        result = {
            'evaluator': VERSION, 'task_id': self.c['task_id'],
            'status': 'unknown' if self.invalid else name, 'success': name == 'success',
            'reason': reason, 'evidence_status': 'unknown' if self.invalid else
                'complete' if self.previous_time is not None else 'pending',
            'terminal': name != 'ongoing',
            'physical_failure': bool(self.failure), 'physics_accepted': False,
            'native_manipulation_qualified': False,
            'irrecoverable_reason': self.failure or self.safety,
            'released_stable_s': self.stable_s, 'detail': deepcopy(self.detail),
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': physical_state,
                'phase': self.phase, 'reason_codes': [reason], 'terminal': physical_state != 'ongoing',
                'recoverable': physical_state == 'ongoing'},
            'evidence_scope': 'Private complete native samples; scene probes and CPU fixtures are not robot success.',
        }
        if self.invalid:
            result['private_evidence_error'] = self.invalid
        self.last_result = deepcopy(result)
        return result

    def update(self, observation):
        if self.invalid or self.failure or self.safety or self.success or self.incomplete:
            return deepcopy(self.last_result)
        try:
            s = observation['task_state']
            if (s.get('schema') != STATE_SCHEMA or s.get('configuration_sha256') != self.digest
                    or s.get('task_id') != self.c['task_id'] or s.get('complete') is not True
                    or s.get('errors') != []):
                raise ValueError('Complete matching private native state required: '+str(s.get('errors')))
            t, step = number(s['time_s']), s['native_step']
            if abs(number(observation['time_s'])-t) > EPS or type(step) is not int or step < 0:
                raise ValueError('Invalid native observation clock')
            first = self.previous_time is None
            if first:
                if s['baseline'] is not True or abs(t) > EPS or s['events'] or s['released_stable_s'] != 0:
                    raise ValueError('Unscored baseline required before scoring')
            elif (s['baseline'] is not False or abs(t-self.previous_time-.02) > EPS
                  or step-self.previous_step != round(.02/self.c['physics_dt_s'])):
                raise ValueError('Consecutive control observations required')
            incoming = s['events']
            if not isinstance(incoming, list) or incoming[:len(self.events)] != self.events:
                raise ValueError('Native event history was rewritten')
            for event in incoming[len(self.events):]:
                if (event['id'] != len(self.events)+1 or event['kind'] != 'harness_assembled_released'
                        or self.c['task_id'] != 'nist_wire_harness' or type(event['native_step']) is not int
                        or not self.previous_step < event['native_step'] <= step
                        or not self.previous_time < number(event['time_s']) <= t+EPS):
                    raise ValueError('Native event does not belong to this task interval')
                self.events.append(deepcopy(event))
            if s['phase'] not in self.rules['phases'] or s['reason'] not in self.rules['reason_descriptions']:
                raise ValueError('Undeclared public referee phase or reason')
            stable = number(s['released_stable_s'])
            if stable < 0 or stable > t+EPS:
                raise ValueError('Invalid native released dwell')
            if stable > self.stable_s+(0. if first else .02)+EPS:
                raise ValueError('Native dwell grew faster than complete elapsed observations')
            self.stable_s = stable
            self.phase, self.detail = s['phase'], deepcopy(s['detail'])
            self.previous_time, self.previous_step = t, step
            self.failure, self.safety = s['terminal_failure'], s['safety_stop']
            if self.failure not in (None, 'object_fell') or self.safety not in (None, 'physics_safety_stop'):
                raise ValueError('Unknown physical terminal reason')
            if self.failure or self.safety:
                return self._result(self.safety or self.failure)
            # An expired episode cannot be revived by a late success receipt.
            if (observation.get('budget_exhausted') is True or t > self.c['thresholds']['max_time_s']+EPS
                    or number(observation.get('action_count', 0)) > self.c['thresholds']['max_actions']):
                self.incomplete = True
                return self._result('budget_exhausted')
            self.success = bool(not first and stable+EPS >= self.c['thresholds']['stable_s'])
            if self.success and self.c['task_id'] == 'nist_wire_harness' and len(self.events) != 1:
                raise ValueError('Harness removal requires a prior assembly event')
            return self._result('success' if self.success else s['reason'])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.failure = self.safety = None
            self.success = self.incomplete = False
            self.stable_s = 0.
            return self._result('native_evidence_unknown')

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result)
