"""Read-only native kitchen referee, independent of any action or grasp policy.

The backend owns acquisition and supplies immutable values at every substep.
The observer never accepts simulator handles, and cannot modify the scene.
"""
from collections import deque
from copy import deepcopy
import math

from .rules import digest, evaluation_configuration, public_rules
from .native_evidence import (EPS, Contacts, angle, contained, fixed_state, local_points,
    near, norm, number, relative_state, rotation, state, still, subtract, upright,
    world_points)

VERSION = 'kitchen_sequence_evaluator.v3'
STATE_SCHEMA = 'kitchen_native_state.v1'


def configuration(task):
    if task.get('schema_version', task.get('schema')) != 'kitchen-manipulation-task-v1':
        raise ValueError('Expected kitchen-manipulation-task-v1')
    ident, seed, c = task['id'], task['episode_seed'], deepcopy(task['evaluation'])
    expected = evaluation_configuration(ident, c['metadata'], seed, c['physics_dt_s'])
    if c != expected or task.get('public_rules') != public_rules(ident, seed):
        raise ValueError('Task rules, geometry or frozen evaluator configuration changed')
    return c, digest(c)


def native_room_path(path):
    prefix = '/World/Geometry/'
    return '/World/Room/Geometry/'+path[len(prefix):] if path.startswith(prefix) else path


class NativeTaskObserver:
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.task_id = self.c['task_id']
        self.limits = self.c['thresholds']
        self.meta = self.c['metadata']
        entries = self.meta['objects']
        self.specs = entries if isinstance(entries, dict) else {r['id']: r for r in entries}
        self.fixtures = self.meta.get('fixture_metadata', {})
        self.fixture_paths = dict(self.meta.get('fixtures', {}))
        self.surfaces = {}
        for index, path in enumerate(self.meta['work_surface_paths']):
            self.surfaces['surface_'+str(index)] = native_room_path(path)
        self.fixture_paths.update(self.surfaces)
        self.burners = {}
        for index, path in enumerate(self.meta.get('goals', {}).get('burner_support_paths', [])):
            self.burners['burner_'+str(index)] = native_room_path(path)
        self.fixture_paths.update(self.burners)
        duplicate_surfaces = [k for k, v in self.surfaces.items() if v in self.burners.values()]
        for key in duplicate_surfaces:
            self.surfaces.pop(key)
            self.fixture_paths.pop(key)
        self.fixed = {name: fixed_state(row['initial_pose_xyzw']) for name, row in self.fixtures.items()}
        self.surface_z = float(self.meta['surface_z_m'])
        self.goals = self.meta['goals']
        self.reset()

    def reset(self):
        self.previous = self.previous_time = self.previous_native_time = self.previous_step = None
        self.initial = self.current = None
        self.error = self.safety = self.failure = None
        self.support_witnesses = {}
        self.support_diagnostics = {}
        self._points_cache = {}
        self.stable_s = self.source_stable_s = self.pot_stable_s = 0.
        self.stable_anchor = None
        self.source_ready = self.carry_active = self.carry_complete = False
        self.carry_start = None
        self.carry_missing_s = 0.
        self.carry_peak_clearance = self.carry_distance = 0.
        self.pot_ready = False
        self.events = []
        self.quiet_s = {name: 0. for name in self.c['actors']}
        self.quiet_anchor = {}
        self.pose_window_steps = round(self.limits['stability_pose_window_s']/self.c['physics_dt_s'])
        self.pose_history = deque(maxlen=self.pose_window_steps+1)
        self.pose_speed = {name: None for name in self.c['actors']}

    def _event(self, kind, frame):
        self.events.append({'id': len(self.events)+1, 'kind': kind,
            'native_step': frame['native_step'], 'time_s': float(frame['time_s'])})

    def _object(self, name, objects):
        return objects.get(name, self.fixed.get(name))

    def _points(self, name, objects, reference=None):
        key = (name, reference)
        if key not in self._points_cache:
            if reference is None:
                row = self.specs[name]
                self._points_cache[key] = world_points(objects[name], row['local_bounds_m'], row.get('collision_support_points_m'))
            else:
                self._points_cache[key] = local_points(self._points(name, objects), self._object(reference, objects))
        return self._points_cache[key]

    def _moved(self, name, objects):
        return norm(subtract(objects[name]['position_m'], self.initial[name]['position_m'])) >= self.limits['minimum_movement_m']

    def _stationary(self, objects):
        # All task stages use the same native rolling-pose speed, envelope and
        # release checks. No control-boundary-only sampling or tensor spikes.
        return all(self.quiet_s[name] > 0. for name in objects)

    def _sample_pose_window(self, objects):
        # Every consecutive native sample participates: the 20 ms endpoints
        # advance by one physics step, rather than by one control observation.
        self.pose_history.append(objects)
        if len(self.pose_history) < self.pose_window_steps+1:
            return
        previous = self.pose_history[0]
        period = self.limits['stability_pose_window_s']
        for name, obj in objects.items():
            self.pose_speed[name] = {
                'linear_m_s': norm(subtract(obj['position_m'], previous[name]['position_m']))/period,
                'angular_rad_s': angle(obj['quaternion_wxyz'], previous[name]['quaternion_wxyz'])/period}

    def _supported(self, body, support, objects, contacts, time_s, geometry_ok, *, carry=False):
        key = (body, support)
        if not geometry_ok:
            self.support_witnesses.pop(key, None)
            return False
        obj = objects[body]
        other = self._object(support, objects)
        rel = relative_state(obj, other) if other is not None else obj
        if contacts.upward(body, support, self.limits['minimum_contact_impulse_ns']) and any(row['separation_m'] <= self.limits['contact_gap_tolerance_m']
                for row in contacts.rows if {row['a'], row['b']} == {body, support} and row['positive']):
            self.support_witnesses[key] = {'time_s': time_s, 'relative': deepcopy(rel), 'world': deepcopy(obj)}
            self.support_diagnostics[key] = 'current_native_upward_contact'
            return True
        witness = self.support_witnesses.get(key)
        if witness is None:
            return False
        # Complete sampling can report zero force or no contact for sleeping
        # bodies. A previous loaded contact plus unchanged, unforced geometry
        # is retained; it is not a force-balance or contact-patch claim.
        sleep = (still(obj, self.limits) and body not in contacts.robot_touched
            and near(obj, witness['world'], self.limits['sleep_position_tolerance_m'],
                     self.limits['sleep_orientation_tolerance_rad']))
        limit = self.limits['carry_payload_contact_gap_s'] if carry else self.limits['support_memory_s']
        recent = time_s-witness['time_s'] <= limit+EPS
        matching = near(rel, witness['relative'], .004 if carry else .001, .08 if carry else .02)
        supported = matching and (sleep or recent)
        if supported:
            self.support_diagnostics[key] = ('prior_native_upward_contact_unchanged_unforced_pose' if sleep else 'recent_native_upward_contact_within_gap_tolerance')
        return supported

    def _container(self, body, support, objects, contacts, time_s, *, carry=False, extra_height=0., compartment=None):
        spec = compartment if compartment is not None else self.specs.get(support, self.fixtures.get(support))
        local = self._points(body, objects, support)
        inside = contained(local, spec.get('interior_bounds_xy_m'), self.limits['containment_tolerance_m'],
                           radius=spec.get('interior_radius_m'))
        gap = min(p[2] for p in local)-spec['floor_top_local_z_m']
        geometry = inside and -self.limits['maximum_penetration_m'] <= gap <= self.limits['support_height_tolerance_m']+extra_height
        return geometry, self._supported(body, support, objects, contacts, time_s, geometry, carry=carry)

    def _surface_support(self, name, objects, contacts, time_s, *, burners=False):
        bottom = min(p[2] for p in self._points(name, objects))
        expected = self.goals.get('burner_support_z_m', self.surface_z) if burners else self.surface_z
        geometry = abs(bottom-expected) <= self.limits['support_height_tolerance_m']
        supports = self.burners if burners else self.surfaces
        return any(self._supported(name, support, objects, contacts, time_s, geometry) for support in supports)

    def _dwell(self, condition, objects):
        if condition:
            if self.stable_anchor is None:
                self.stable_anchor = deepcopy(objects)
            condition = all(near(obj, self.stable_anchor[name], self.limits['maximum_stable_position_range_m'],
                self.limits['maximum_stable_orientation_range_rad']) for name, obj in objects.items())
        if condition:
            self.stable_s += self.c['physics_dt_s']
        else:
            self.stable_s, self.stable_anchor = 0., None
        return condition

    def _region(self, objects, which):
        return contained(self._points('tray', objects), self.goals[which+'_region_xy_m'],
                         self.limits['containment_tolerance_m'])

    def _breakfast(self, frame, objects, contacts, stationary):
        t, l = float(frame['time_s']), self.limits
        first = frame['baseline']
        geometry, support = True, True
        for body, target in (('plate', 'tray'), ('cup', 'tray'), ('bread_0', 'plate'), ('bread_1', 'plate')):
            inside, supported = self._container(body, target, objects, contacts, t, carry=self.carry_active)
            geometry &= inside
            support &= supported
        payload = {'plate', 'cup', 'bread_0', 'bread_1'}
        oriented = all(upright(objects[n], l['upright_tilt_max_rad']) for n in ('cup', 'plate'))
        tray_upright = upright(objects['tray'], l['carry_tilt_max_rad'])
        released = not contacts.robot_touched
        source, destination = self._region(objects, 'source'), self._region(objects, 'destination')
        surface = self._surface_support('tray', objects, contacts, t)
        moved_payload = all(self._moved(n, objects) for n in payload)
        loaded = geometry and support and oriented and tray_upright
        source_condition = not first and source and surface and loaded and released and stationary and moved_payload
        self.source_stable_s = self.source_stable_s+self.c['physics_dt_s'] if source_condition else 0.
        if self.source_stable_s+EPS >= l['source_loading_stable_s'] and not self.source_ready:
            self.source_ready = True
            self.carry_complete = False
            self._event('source_loaded', frame)
        lower = min(p[2] for p in self._points('tray', objects))
        airborne = lower > self.surface_z+l['carry_clearance_m']
        hand = 'tray' in contacts.robot_touched
        external_tray = contacts.external('tray', payload, robot_allowed=True)
        direct_payload = bool(payload & contacts.robot_touched)
        # The first upward lift remains in the source region while the hand
        # closes. Intermediate low clearance is allowed, but transport credit
        # requires a genuinely airborne part of this same loaded attempt.
        if self.source_ready and not self.carry_active and not self.carry_complete and (not source or not loaded or direct_payload):
            self.source_ready = False
            self.source_stable_s = 0.
        if self.source_ready and not self.carry_active and not self.carry_complete and hand and not external_tray:
            self.carry_active = True
            self.carry_start = tuple(objects['tray']['position_m'][:2])
            self.carry_peak_clearance = self.carry_distance = self.carry_missing_s = 0.
            self._event('loaded_lift_started', frame)
        if self.carry_active:
            self.carry_missing_s = 0. if support else self.carry_missing_s+self.c['physics_dt_s']
            valid_payload = geometry and support and oriented and tray_upright and not direct_payload
            net = norm(subtract(objects['tray']['position_m'][:2], self.carry_start))
            if airborne and hand and not external_tray and valid_payload:
                self.carry_distance = max(self.carry_distance, net)
                self.carry_peak_clearance = max(self.carry_peak_clearance, lower-self.surface_z)
            arrived = (destination and surface and valid_payload and self.carry_distance >= l['carry_net_displacement_min_m'])
            if arrived:
                self.carry_complete, self.carry_active = True, False
                self._event('loaded_delivery', frame)
            elif not valid_payload or (external_tray and not destination) or (not hand and not external_tray):
                self.carry_active = self.carry_complete = self.source_ready = False
                self.source_stable_s = 0.
                self._event('carry_retry_required', frame)
        # Delivery credit belongs to the current supported payload assembly.
        # Any subsequent direct payload handling or unloading clears it.
        if self.carry_complete and (not geometry or not oriented or direct_payload or not destination):
            self.carry_complete = self.source_ready = False
            self._event('delivery_disturbed', frame)
        condition = (not first and self.carry_complete and destination and surface and loaded and released and stationary)
        self._dwell(condition, objects)
        phase = 'settling' if self.carry_complete else 'carrying' if self.source_ready else 'loading'
        reason = ('waiting_for_release_and_stability' if self.carry_complete else
                  'loaded_carry_required' if self.source_ready else 'source_loading_required')
        return phase, reason, {'source_ready': self.source_ready, 'carry_active': self.carry_active,
            'carry_complete': self.carry_complete, 'carry_distance_m': self.carry_distance,
            'source_stable_s': self.source_stable_s, 'payload_contained': bool(geometry)}

    def _sorting(self, frame, objects, contacts, stationary):
        t = float(frame['time_s'])
        matched = {}
        geometries = {}
        if self.task_id == 'produce_transfer':
            for name in objects:
                target = name.split('_')[0]+'_tray'
                geometry, support = self._container(name, target, objects, contacts, t)
                matched[name] = geometry and support and self._moved(name, objects)
        else:
            channels = self.fixtures['cutlery_tray']['channels']
            direct = {}
            for name in objects:
                kind = name.split('_')[0]
                geometry, support = self._container(name, 'cutlery_tray', objects, contacts, t, compartment=channels[kind])
                local = self._points(name, objects, 'cutlery_tray')
                floor = channels[kind]['floor_top_local_z_m']
                geometries[name] = (contained(local, channels[kind]['interior_bounds_xy_m'], self.limits['containment_tolerance_m'])
                    and -self.limits['maximum_penetration_m'] <= min(p[2] for p in local)-floor <= self.limits['utensil_overlap_height_max_m']
                    and abs(rotation(objects[name]['quaternion_wxyz'])[2][2]) >= math.cos(self.limits['utensil_tilt_max_rad']))
                direct[name] = geometry and support and geometries[name]
            for name in objects:
                peers = [n for n in objects if n != name and n.split('_')[0] == name.split('_')[0]]
                # Only a short support chain to a directly supported same-class
                # utensil is permitted. This allows ordinary slight overlap,
                # while forbidding a stack rooted on a divider or another class.
                overlap = any(direct[p] and self._supported(name, p, objects, contacts, t,
                    geometries[name] and objects[name]['position_m'][2] >= objects[p]['position_m'][2]-.002) for p in peers)
                matched[name] = geometries[name] and (direct[name] or overlap) and self._moved(name, objects)
        geometry_ready = all(matched.values())
        condition = not frame['baseline'] and geometry_ready and stationary and not contacts.robot_touched
        self._dwell(condition, objects)
        return ('settling' if geometry_ready else 'sorting',
            'waiting_for_release_and_stability' if geometry_ready else 'task_in_progress',
            {'objects_accepted': matched, 'accepted_count': sum(matched.values())})

    def _grouped_sorting(self, frame, objects, contacts, stationary):
        """Whole-object receiving regions and an acyclic, tray-rooted support graph.

        A dish may nest below another dish's rim, so a prescribed stack height
        would be both fragile and unnecessarily restrictive. The actual signed
        upward native contact establishes support; collision extrema constrain
        the possible contact height and reject imagined floating support.
        """
        t = float(frame['time_s'])
        geometries, target_local, supported = {}, {}, {}
        for name, obj in objects.items():
            spec = self.specs[name]
            target = spec['assigned_fixture']
            fixture = self.fixtures[target]
            points = self._points(name, objects, target)
            target_local[name] = points
            orientation_ok = (spec['placement_orientation'] == 'any'
                or upright(obj, self.limits['upright_tilt_max_rad']))
            geometries[name] = (orientation_ok and contained(points, fixture['interior_bounds_xy_m'],
                self.limits['containment_tolerance_m'])
                and min(p[2] for p in points) >= fixture['floor_top_local_z_m']-self.limits['maximum_penetration_m'])
            direct_geometry, direct_support = self._container(name, target, objects, contacts, t)
            if geometries[name] and direct_geometry and direct_support:
                supported[name] = target
        # Each iteration adds only actors whose parent was already rooted in
        # the previous iteration. Unsupported cycles and wrong-bin intermediates
        # cannot manufacture a chain. No layer count or stacking order is fixed.
        pending = set(objects)-set(supported)
        while pending:
            admitted = {}
            for name in sorted(pending):
                if not geometries[name]:
                    continue
                points = target_local[name]
                bottom = min(p[2] for p in points)
                for parent in sorted(supported):
                    if self.specs[parent]['assigned_fixture'] != self.specs[name]['assigned_fixture']:
                        continue
                    lower = target_local[parent].min(axis=0)
                    upper = target_local[parent].max(axis=0)
                    # A nested plate can sit below its parent's rim; contact
                    # still has to lie within the parent's vertical extent.
                    height_ok = (lower[2]-self.limits['maximum_penetration_m'] <= bottom
                        <= upper[2]+self.limits['support_height_tolerance_m'])
                    overlap_xy = bool(((points.max(axis=0)[:2] >= lower[:2])
                        & (points.min(axis=0)[:2] <= upper[:2])).all())
                    if self._supported(name, parent, objects, contacts, t, height_ok and overlap_xy):
                        admitted[name] = parent
                        break
            if not admitted:
                break
            supported.update(admitted)
            pending.difference_update(admitted)
        matched = {name: bool(geometries[name] and name in supported and self._moved(name, objects))
                   for name in objects}
        ready = all(matched.values())
        self._dwell(not frame['baseline'] and ready and stationary and not contacts.robot_touched, objects)
        return ('settling' if ready else 'sorting',
            'waiting_for_release_and_stability' if ready else 'task_in_progress',
            {'objects_accepted': matched, 'accepted_count': sum(matched.values()),
             'required_count': len(matched), 'support_parent_by_actor': supported})

    def _pot(self, frame, objects, contacts, stationary):
        t, l = float(frame['time_s']), self.limits
        pot, lid = objects['pot'], objects['lid']
        pot_location = norm(subtract(pot['position_m'][:2], self.goals['burner_center_xy_m'])) <= l['pot_center_tolerance_m']
        pot_ok = (pot_location and upright(pot, l['pot_upright_tilt_max_rad'])
            and self._surface_support('pot', objects, contacts, t, burners=True) and self._moved('pot', objects))
        pot_stationary = self.quiet_s['pot'] > 0.
        condition = pot_ok and pot_stationary and 'pot' not in contacts.robot_touched and not frame['baseline']
        self.pot_stable_s = self.pot_stable_s+self.c['physics_dt_s'] if condition else 0.
        if self.pot_stable_s+EPS >= l['pot_before_lid_stable_s'] and not self.pot_ready:
            # Lid must still be separate at the stage boundary; a preassembled
            # pot/lid transfer cannot manufacture the required two-stage task.
            # Sleeping bodies may stop emitting contact pairs, so absence of a
            # pair is insufficient. Require visible clearance between the
            # collision bounds in the pot frame; a lid held above the rim or
            # resting separately on the counter both satisfy this condition.
            pot_points = self._points('pot', objects, 'pot')
            lid_points = self._points('lid', objects, 'pot')
            gap = l['contact_gap_tolerance_m']
            separate = bool(((lid_points.min(axis=0)-pot_points.max(axis=0) > gap)
                | (pot_points.min(axis=0)-lid_points.max(axis=0) > gap)).any())
            if separate and not contacts.pair('lid', 'pot', positive=False):
                self.pot_ready = True
                self._event('pot_placed_before_lid', frame)
        if self.pot_ready and (not pot_ok or 'pot' in contacts.robot_touched):
            self.pot_ready = False
            self.pot_stable_s = 0.
        local_origin = local_points([lid['position_m']], pot)[0]
        rpot, rlid = rotation(pot['quaternion_wxyz']), rotation(lid['quaternion_wxyz'])
        alignment = sum(rpot[i][2]*rlid[i][2] for i in range(3)) >= math.cos(l['lid_axis_tilt_max_rad'])
        expected = self.goals['pot_rim_local_z_m']-self.goals['lid_seat_local_z_m']
        geometry = (math.hypot(*local_origin[:2]) <= l['lid_radial_offset_max_m']
            and abs(local_origin[2]-expected) <= l['lid_seat_height_tolerance_m'] and alignment)
        lid_ok = geometry and self._supported('lid', 'pot', objects, contacts, t, geometry) and self._moved('lid', objects)
        final = self.pot_ready and pot_ok and lid_ok and stationary and not contacts.robot_touched and not frame['baseline']
        self._dwell(final, objects)
        phase = 'settling' if self.pot_ready and lid_ok else 'seating_lid' if self.pot_ready else 'placing_pot'
        reason = 'waiting_for_release_and_stability' if phase == 'settling' else 'task_in_progress' if self.pot_ready else 'pot_before_lid_required'
        return phase, reason, {'pot_ready': self.pot_ready, 'pot_stable_s': self.pot_stable_s, 'lid_seated': bool(lid_ok)}

    def on_frame(self, frame):
        if self.error is not None:
            return
        try:
            self._frame(frame)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.error = str(exc)

    def _frame(self, frame):
        t = number(frame['time_s'], 'time')
        nt = number(frame['native_time_s'], 'native time')
        step, baseline = frame['native_step'], frame['baseline']
        first = self.previous is None
        if type(step) is not int or step < 0 or type(baseline) is not bool or baseline != first or t < 0 or nt < 0:
            raise ValueError('One unscored native baseline and valid counters are required')
        dt = self.c['physics_dt_s']
        if abs(number(frame['physics_dt_s'])-dt) > EPS:
            raise ValueError('Native physics timestep changed')
        if first:
            if abs(t) > EPS or frame['contacts']:
                raise ValueError('Baseline must precede stepping and contact credit')
        elif step != self.previous_step+1 or abs(t-self.previous_time-dt) > EPS or abs(nt-self.previous_native_time-dt) > 2.e-6:
            raise ValueError('Native samples are missing or out of order')
        if set(frame['objects']) != set(self.c['actors']):
            raise ValueError('Native body inventory differs from frozen task')
        objects = {name: state(frame['objects'][name]) for name in self.c['actors']}
        if first:
            for name, obj in objects.items():
                reset = fixed_state(self.specs[name]['initial_pose_xyzw'])
                if not near(obj, reset, self.limits['reset_position_tolerance_m'], self.limits['reset_orientation_tolerance_rad']):
                    raise ValueError('Native reset differs from bound source layout')
            self.initial = deepcopy(objects)
        contacts = Contacts(frame['contacts'], time_s=t, actors=self.c['actors'], fixtures=self.fixture_paths,
            robot_prefix=self.c['robot_actor_prefix'], limits=self.limits)
        if contacts.penetration:
            self.safety = 'physics_safety_stop'
        lo, hi = self.meta.get('recovery_workspace_bounds_m', self.meta['workspace_bounds_m'])
        if not first and (contacts.floor_touched or any(not lo[i] <= obj['position_m'][i] <= hi[i]
                for obj in objects.values() for i in range(3))):
            self.failure = 'object_left_recoverable_workspace'
        self.support_diagnostics = {}
        self._points_cache = {}
        source = {'produce_transfer': 'source_board', 'utensil_sorting': 'source_pad', 'pot_lid_placement': 'source_mat'}.get(self.task_id)
        for name in objects:
            actor_source = self.meta.get('source_support_by_actor', {}).get(name, source)
            if actor_source in self.fixtures:
                self._container(name, actor_source, objects, contacts, t)
            self._surface_support(name, objects, contacts, t)
        self._sample_pose_window(objects)
        for name, obj in objects.items():
            speed = self.pose_speed[name]
            quiet = (not first and speed is not None and name not in contacts.robot_touched
                and speed['linear_m_s'] <= self.limits['maximum_linear_speed_m_s']
                and speed['angular_rad_s'] <= self.limits['maximum_angular_speed_rad_s'])
            if quiet:
                anchor = self.quiet_anchor.setdefault(name, deepcopy(obj))
                quiet = near(obj, anchor, self.limits['maximum_stable_position_range_m'], self.limits['maximum_stable_orientation_range_rad'])
            if quiet:
                self.quiet_s[name] += dt
            else:
                self.quiet_s[name] = 0.
                self.quiet_anchor.pop(name, None)
        stationary = self._stationary(objects)
        if self.task_id == 'breakfast_serving':
            phase, reason, detail = self._breakfast(frame, objects, contacts, stationary)
        elif self.task_id == 'pot_lid_placement':
            phase, reason, detail = self._pot(frame, objects, contacts, stationary)
        elif self.task_id in ('pantry_stocking', 'table_clearing'):
            phase, reason, detail = self._grouped_sorting(frame, objects, contacts, stationary)
        else:
            phase, reason, detail = self._sorting(frame, objects, contacts, stationary)
        self.current = {'schema': STATE_SCHEMA, 'configuration_sha256': self.digest, 'task_id': self.task_id,
            'time_s': t, 'native_time_s': nt, 'native_step': step, 'baseline': baseline,
            'phase': phase, 'reason': 'reset_baseline_not_scored' if first else reason,
            'released_stable_s': self.stable_s, 'terminal_failure': self.failure,
            'safety_stop': self.safety, 'detail': detail,
            'native_support_by_actor': {name: {'supported': any(k[0] == name for k in self.support_diagnostics),
                'released_quiet_native_s': self.quiet_s[name],
                'rolling_pose_speed': deepcopy(self.pose_speed[name]),
                'witnesses': [{'support': k[1], 'support_actor_path': self.c['actors'].get(k[1], self.fixture_paths.get(k[1])), 'source': value} for k, value in self.support_diagnostics.items() if k[0] == name]}
                for name in objects}}
        self.previous = objects
        self.previous_time, self.previous_native_time, self.previous_step = t, nt, step

    def on_observe(self):
        result = deepcopy(self.current or {'schema': STATE_SCHEMA, 'configuration_sha256': self.digest})
        result.update(complete=self.current is not None and self.error is None,
            errors=[] if self.error is None else [self.error], events=deepcopy(self.events))
        return result


class SequenceEvaluator:
    """Canonical worker scorer; missing evidence is unknown, never task failure."""
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.rules = public_rules(self.c['task_id'], self.c['episode_seed'])['referee']
        self.reset()

    def reset(self):
        self.previous_time = self.previous_step = None
        self.invalid = self.failure = self.safety = None
        self.events = []
        self.stable_s = 0.
        self.success = False
        self.incomplete = False
        self.phase = next(iter(self.rules['phases']))
        self.detail = {}
        self.last_result = self._result('task_in_progress')

    def _result(self, reason):
        state_name = ('evaluator_invalid' if self.invalid else 'safety_stop' if self.safety else
                      'task_failure' if self.failure else 'success' if self.success else 'ongoing')
        status = 'unknown' if self.invalid else 'incomplete' if reason == 'incomplete_budget' else state_name
        result = {'evaluator': VERSION, 'task_id': self.c['task_id'], 'status': status,
            'success': state_name == 'success', 'reason': reason,
            'evidence_status': 'unknown' if self.invalid else 'complete' if self.previous_time is not None else 'pending',
            'physical_failure': bool(self.failure), 'physics_accepted': False,
            'native_manipulation_qualified': False, 'irrecoverable_reason': self.failure or self.safety,
            'released_stable_s': self.stable_s, 'detail': deepcopy(self.detail),
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': state_name,
                'phase': self.phase, 'reason_codes': [reason], 'terminal': state_name != 'ongoing',
                'recoverable': state_name == 'ongoing'},
            'evidence_scope': 'Complete private native samples; CPU fixtures and independent scene probes are not robot task credit.'}
        if self.invalid:
            result['private_evidence_error'] = self.invalid
        if self.incomplete:
            # The generic episode envelope terminates the run while the
            # physical referee correctly retains its unfinished task state.
            result.update(status='incomplete', terminal=True, reason='budget_exhausted')
        self.last_result = deepcopy(result)
        return result

    def update(self, observation):
        if self.invalid or self.failure or self.safety or self.success or self.incomplete:
            return deepcopy(self.last_result)
        try:
            s = observation['task_state']
            if (s.get('schema') != STATE_SCHEMA or s.get('configuration_sha256') != self.digest
                    or s.get('task_id') != self.c['task_id'] or s.get('complete') is not True or s.get('errors') != []):
                raise ValueError('Complete matching private native state required: '+str(s.get('errors')))
            t = number(s['time_s'])
            if abs(number(observation['time_s'])-t) > EPS:
                raise ValueError('Observation and native state time differ')
            step = s['native_step']
            if type(step) is not int or step < 0:
                raise ValueError('Invalid native sample index')
            first = self.previous_time is None
            if first:
                if s['baseline'] is not True or abs(t) > EPS or s['events'] or s['released_stable_s'] != 0:
                    raise ValueError('Unscored baseline is required before evaluation')
            elif (s['baseline'] is not False or abs(t-self.previous_time-self.c['control_period_s']) > EPS
                    or step-self.previous_step != round(self.c['control_period_s']/self.c['physics_dt_s'])):
                raise ValueError('Consecutive control observations required')
            incoming = s['events']
            if not isinstance(incoming, list) or incoming[:len(self.events)] != self.events:
                raise ValueError('Native event history was rewritten')
            for event in incoming[len(self.events):]:
                if (event['id'] != len(self.events)+1 or type(event['native_step']) is not int
                        or not self.previous_step < event['native_step'] <= step
                        or not self.previous_time < number(event['time_s']) <= t+EPS):
                    raise ValueError('Native event does not belong to this interval')
                self.events.append(deepcopy(event))
            if s['phase'] not in self.rules['phases'] or s['reason'] not in self.rules['reason_descriptions']:
                raise ValueError('Undeclared public referee phase or reason')
            self.phase, self.detail = s['phase'], deepcopy(s['detail'])
            stable = number(s['released_stable_s'])
            if stable < 0 or stable > t+EPS:
                raise ValueError('Invalid native released dwell')
            self.stable_s = stable
            self.previous_time, self.previous_step = t, step
            self.failure, self.safety = s['terminal_failure'], s['safety_stop']
            if self.failure is not None and self.failure != 'object_left_recoverable_workspace':
                raise ValueError('Unknown physical terminal reason')
            if self.safety is not None and self.safety != 'physics_safety_stop':
                raise ValueError('Unknown physical safety reason')
            if self.failure or self.safety:
                return self._result(self.safety or self.failure)
            if (observation.get('budget_exhausted') is True or t > self.c['thresholds']['max_time_s']+EPS
                    or number(observation.get('action_count', 0)) > self.c['thresholds']['max_actions']):
                self.incomplete = True
                return self._result('incomplete_budget')
            self.success = not first and stable+EPS >= self.c['thresholds']['stable_s']
            if self.success:
                return self._result('success')
            return self._result(s['reason'])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.failure = self.safety = None
            self.success = False
            self.stable_s = 0.
            return self._result('native_evidence_unknown')

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result)
