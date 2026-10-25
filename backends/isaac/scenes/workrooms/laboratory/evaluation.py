"""Private, read-only native object evidence for the five laboratory tasks.

No hand policy, simulator handle or target setter belongs in this module.
"""
from collections import deque
from copy import deepcopy
import math
import numpy as np

from .rules import digest, evaluation_configuration, public_rules
from .native_evidence import EPS, Contacts, angle, distance, fixed, local, number, points, rotation, state, upright

VERSION = 'laboratory_sequence_evaluator.v1'
STATE_SCHEMA = 'laboratory_native_state.v1'


def configuration(task):
    if task.get('schema_version') != 'laboratory-manipulation-task-v1':
        raise ValueError('Expected laboratory-manipulation-task-v1')
    c = deepcopy(task['evaluation'])
    expected = evaluation_configuration(task['id'], c['metadata'], task['episode_seed'],
        c['physics_dt_s'], c['room_support_paths'])
    if c != expected or task['public_rules'] != public_rules(task['id'], task['episode_seed']):
        raise ValueError('Frozen laboratory rules or geometry differ')
    return c, digest(c)


def native_room_path(path):
    return path.replace('/World/Geometry/', '/World/Room/Geometry/', 1)


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class NativeTaskObserver:
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.task_id, self.meta = self.c['task_id'], self.c['metadata']
        self.limits, self.p = self.c['thresholds'], self.meta['task_parameters']
        self.specs = {row['id']: row for row in self.meta['objects']}
        self.fixtures = self.meta['fixture_metadata']
        self.fixed = {k: fixed(v['initial_pose_xyzw']) for k, v in self.fixtures.items()}
        self.paths = {**self.meta['actors'], **self.meta['fixtures']}
        self.surfaces = {}
        for role, paths_ in self.c['room_support_paths'].items():
            for i, path in enumerate(paths_):
                key = 'environment_'+role+'_'+str(i)
                self.paths[key] = native_room_path(path)
                if role != 'floor':
                    self.surfaces[key] = self.paths[key]
        self.reset()

    def reset(self):
        self.previous = self.current = self.initial = None
        self.previous_time = self.previous_native_time = self.previous_step = None
        self.error = self.safety = self.failure = None
        self.stable_s = 0.
        self.stable_anchor = None
        self.witnesses, self.support_diagnostics = {}, {}
        self.events, self.moved, self.spilled, self.scooped = [], set(), set(), set()
        self.previous_cup_ids = None
        self.readout_stable_s, self.readout_anchor = 0., None
        self.readout_cup_ids = None
        self.scoop_load_origins = {}
        self.holder_ready = self.cap_opened = self.cap_parked = self.cap_reclosed = False
        self.cap_tray_s = 0.
        self.previous_cap_yaw = None
        self.cap_turn = 0.
        self.pose_steps = round(self.limits['pose_window_s']/self.c['physics_dt_s'])
        self.history = deque(maxlen=self.pose_steps+1)
        self.pose_speed = {}
        self.quiet_s = {k: 0. for k in self.specs}
        self.quiet_anchor = {}

    def _event(self, kind, frame):
        self.events.append({'id': len(self.events)+1, 'kind': kind,
                            'time_s': frame['time_s'], 'native_step': frame['native_step']})

    def _supported(self, body, support, objects, contacts, t, geometry=True):
        key = (body, support)
        if not geometry:
            self.witnesses.pop(key, None)
            return False
        obj = objects[body]
        other = objects.get(support, self.fixed.get(support))
        relative = (local(obj['position_m'], other) if other else np.asarray(obj['position_m']))
        if key in contacts.upward:
            self.witnesses[key] = (t, deepcopy(obj), relative.copy())
            self.support_diagnostics[key] = 'native_upward_contact'
            return True
        witness = self.witnesses.get(key)
        if witness is None:
            return False
        recent = t-witness[0] <= self.limits['support_memory_s']+EPS
        unchanged = distance(obj, witness[1]) <= .001 and angle(obj, witness[1]) <= .03
        unforced = (body not in contacts.robot and np.linalg.norm(obj['linear_velocity_m_s']) < .03
                    and np.linalg.norm(obj['angular_velocity_rad_s']) < .3)
        matching = np.linalg.norm(relative-witness[2]) <= .001
        if matching and (recent or (unchanged and unforced)):
            self.support_diagnostics[key] = 'prior_loaded_contact_unchanged_or_recent'
            return True
        return False

    def _surface(self, name, objects, contacts, t):
        bottom = points(objects[name], self.specs[name]['local_bounds_m'])[:, 2].min()
        geometry = abs(bottom-.90) <= self.limits['support_gap_m']
        return any(self._supported(name, k, objects, contacts, t, geometry) for k in self.surfaces)

    def _inside(self, name, center, half, objects):
        delta = np.abs(points(objects[name], self.specs[name]['local_bounds_m'])-center)
        return bool(np.all(delta <= np.asarray(half)+self.limits['placement_tolerance_m']))

    def _sorting(self, objects, contacts, t):
        matched = {}
        for name, goal in self.p['targets'].items():
            geometry = self._inside(name, goal['center_m'], goal['half_extents_m'], objects)
            bottom = points(objects[name], self.specs[name]['local_bounds_m'])[:, 2].min()
            geometry &= abs(bottom-self.p['support_floor_z_m']) <= self.limits['support_gap_m']
            matched[name] = bool(geometry and upright(objects[name], self.limits['upright_tilt_rad'])
                and self._supported(name, goal['fixture'], objects, contacts, t, geometry)
                and name in self.moved)
        return all(matched.values()), {'placed': matched}

    def _racking(self, objects, contacts, t):
        matched = {}
        occupied = set()
        for name in objects:
            index, goal = min(enumerate(self.p['holes']),
                key=lambda row: np.linalg.norm(np.asarray(objects[name]['position_m'])[:2]-row[1]['center_m'][:2]))
            obj = objects[name]
            axis = rotation(obj['quaternion_wxyz'])[:, 2]
            bottom = np.asarray(obj['position_m'])
            rim_distance = goal['rim_z_m']-bottom[2]
            at_rim = bottom+axis*(rim_distance/max(axis[2], .01))
            radius = max(abs(v) for b in self.specs[name]['local_bounds_m'] for v in b[:2])
            radial = max(np.linalg.norm(bottom[:2]-goal['center_m'][:2]),
                         np.linalg.norm(at_rim[:2]-goal['center_m'][:2]))
            geometry = bool(upright(obj, math.radians(self.p['upright_tolerance_deg']))
                and radial+radius <= goal['radius_m']+.001
                and abs(bottom[2]-goal['bottom_z_m']) <= .004
                and rim_distance >= self.p['insert_depth_min_m'])
            matched[name] = bool(geometry and index not in occupied and name in self.moved
                and self._supported(name, goal['fixture'], objects, contacts, t, geometry))
            if matched[name]:
                occupied.add(index)
        return all(matched.values()), {'inserted': matched}

    def _cap(self, frame, objects, contacts):
        t, p = frame['time_s'], self.p
        bottle, cap = objects['sample_bottle'], objects['cap']
        holder = np.asarray(p['holder_center_m'])
        holder_geometry = (np.linalg.norm(np.asarray(bottle['position_m'])[:2]-holder[:2]) < .004
            and abs(bottle['position_m'][2]-holder[2]) < .004
            and upright(bottle, self.limits['upright_tilt_rad']))
        held = self._supported('sample_bottle', 'bottle_holder', objects, contacts, t, holder_geometry)
        relative = local(cap['position_m'], bottle)
        rot = rotation(bottle['quaternion_wxyz']).T @ rotation(cap['quaternion_wxyz'])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        if held and not self.holder_ready and not frame['baseline']:
            self.holder_ready = True
            self.cap_turn = 0.
            self.previous_cap_yaw = yaw
            self._event('bottle_in_holder', frame)
        if self.holder_ready and held and self.previous_cap_yaw is not None:
            self.cap_turn += math.atan2(math.sin(yaw-self.previous_cap_yaw), math.cos(yaw-self.previous_cap_yaw))
        self.previous_cap_yaw = yaw
        thread = p['thread']
        if (not self.cap_opened and held and self.holder_ready
                and self.cap_turn*thread['opening_sign'] >= .8*thread['required_turn_rad']
                and relative[2] >= thread.get('cleared_cap_z_m', thread['open_cap_z_m'])-.002):
            self.cap_opened = True
            self._event('cap_unscrewed', frame)
        target = p['cap_tray_target']
        geometry = self._inside('cap', target['center_m'], target['half_extents_m'], objects)
        parked = (self.cap_opened and geometry
            and self._supported('cap', target['fixture'], objects, contacts, t, geometry)
            and 'cap' not in contacts.robot and self.quiet_s['cap'] > 0.)
        self.cap_tray_s = self.cap_tray_s+self.c['physics_dt_s'] if parked else 0.
        if self.cap_tray_s+EPS >= self.limits['stable_s'] and not self.cap_parked:
            self.cap_parked = True
            self._event('cap_parked_released', frame)
        symmetry = 3 if thread['model'] == 'three_start_short_turn_coarse_helical_contact' else 1
        closed_error = math.atan2(math.sin(symmetry*(yaw-thread['closed_angle_rad'])),
                                 math.cos(symmetry*(yaw-thread['closed_angle_rad'])))/symmetry
        closed_geometry = bool(np.linalg.norm(relative[:2]) <= thread['radial_tolerance_m']
            and abs(relative[2]-thread['closed_cap_z_m']) <= thread['closure_tolerance_z_m']
            and rot[2, 2] > math.cos(math.radians(10))
            and abs(closed_error) < math.radians(20))
        closed = self._supported('cap', 'sample_bottle', objects, contacts, t, closed_geometry)
        if self.cap_parked and held and closed and not self.cap_reclosed:
            self.cap_reclosed = True
            self._event('cap_reclosed_in_holder', frame)
        home = p['bottle_home']
        home_ok = (np.linalg.norm(np.asarray(bottle['position_m'])[:2]-home['center_m'][:2])
                   <= min(home['half_extents_m'][:2])
                   and upright(bottle, self.limits['upright_tilt_rad'])
                   and self._surface('sample_bottle', objects, contacts, t))
        return bool(self.cap_reclosed and closed and home_ok), {
            'holder_ready': self.holder_ready, 'cap_opened': self.cap_opened,
            'cap_parked': self.cap_parked, 'cap_reclosed': self.cap_reclosed,
            'closed_now': bool(closed), 'bottle_home': bool(home_ok)}

    def _pellet_in(self, name, container, objects):
        spec = self.p[{'weighing_cup': 'cup', 'source_tray': 'source_tray', 'scoop': 'scoop'}[container]]
        obj = objects.get(container, self.fixed.get(container))
        center = local(objects[name]['position_m'], obj)
        lo, hi = np.asarray(spec['interior_bounds_m'])
        radius = self.p['pellet_radius_m']
        return bool(np.all(center-radius >= lo-.001) and np.all(center+radius <= hi+.001))

    def _solid(self, frame, objects, contacts):
        p, t = self.p, frame['time_s']
        inventory = {k: [] for k in ('source_tray', 'weighing_cup', 'scoop', 'pending', 'spill')}
        for name in p['pellet_ids']:
            located = None
            for container in ('weighing_cup', 'scoop', 'source_tray'):
                if self._pellet_in(name, container, objects):
                    located = container
                    self._supported(name, container, objects, contacts, t)
                    break
            if name in contacts.robot:
                self.scooped.discard(name)
                self.scoop_load_origins.pop(name, None)
            carrying = (located == 'scoop' and name not in contacts.robot
                and self._supported(name, 'scoop', objects, contacts, t))
            if carrying:
                origin = self.scoop_load_origins.setdefault(name, deepcopy(objects['scoop']))
                if distance(objects['scoop'], origin) >= self.limits['minimum_movement_m']:
                    self.scooped.add(name)
            else:
                self.scoop_load_origins.pop(name, None)
            if located is None:
                exterior = (any(a == name for a, b in contacts.upward if b not in p['pellet_ids'])
                    or any(a == name and b in self.surfaces for a, b in self.support_diagnostics))
                if name in contacts.floor or (self.quiet_s[name] >= .10 and exterior):
                    self.spilled.add(name)
                located = 'spill' if name in self.spilled else 'pending'
            inventory[located].append(name)
        mass = len(inventory['weighing_cup'])*p['pellet_mass_kg']
        spill = len(self.spilled)*p['pellet_mass_kg']
        if spill > p['spill_mass_limit_kg']+EPS:
            self.failure = 'spill_limit_exceeded'
        cup = objects['weighing_cup']
        bounds = points(cup, self.specs['weighing_cup']['local_bounds_m'])
        lo, hi = np.asarray(p['platform']['pan_bounds_xy_m'])
        platform_geo = bool(np.all(bounds[:, :2] >= lo-.004) and np.all(bounds[:, :2] <= hi+.004)
            and abs(bounds[:, 2].min()-p['platform']['support_top_world_z_m']) < .008
            and upright(cup, self.limits['upright_tilt_rad']))
        supported = self._supported('weighing_cup', 'balance_platform', objects, contacts, t, platform_geo)
        cup_grains = set(inventory['weighing_cup'])
        rooted = {'weighing_cup'}
        direct = {name for name in cup_grains
                  if self._supported(name, 'weighing_cup', objects, contacts, t)}
        rooted.update(direct)
        # Allow grains resting on other in-cup grains, with an actual upward
        # contact path back to the cup. Containment alone never establishes load.
        for _ in range(len(cup_grains)):
            additions = {body for body, parent in contacts.upward
                         if body in cup_grains and parent in rooted}
            if additions <= rooted:
                break
            rooted.update(additions)
        grains_supported = cup_grains <= rooted
        unloaded = (frozenset(('scoop', 'weighing_cup')) not in contacts.pairs
                    and 'weighing_cup' not in contacts.robot)
        lo, hi = np.asarray(p['scoop_return_bounds_m'])
        scoop_points = points(objects['scoop'], self.specs['scoop']['local_bounds_m'])
        returned_geo = bool(np.all(scoop_points >= lo-.004) and np.all(scoop_points <= hi+.004))
        returned = self._supported('scoop', 'scoop_rest', objects, contacts, t, returned_geo)
        eligible = all(k in self.scooped for k in inventory['weighing_cup'])
        cup_ids = frozenset(inventory['weighing_cup'])
        if cup_ids != self.previous_cup_ids:
            self.stable_s, self.stable_anchor = 0., None
        self.previous_cup_ids = cup_ids
        goal = (abs(mass-p['target_mass_kg']) <= p['tolerance_mass_kg']+EPS
            and spill <= p['spill_mass_limit_kg']+EPS and not inventory['pending']
            and supported and grains_supported and unloaded and returned and eligible and 'weighing_cup' in self.moved)
        # The instrument remains useful before the target or action history is
        # satisfied. Only the cup and its present load qualify this readout;
        # returning the scoop is a separate completion requirement.
        readout_bodies = cup_grains | {'weighing_cup'}
        readout_condition = (not frame['baseline'] and supported and grains_supported and unloaded
            and not (readout_bodies & contacts.robot)
            and not any(frozenset((name, 'scoop')) in contacts.pairs for name in cup_grains)
            and all(self.quiet_s[name] > 0 for name in readout_bodies))
        if cup_ids != self.readout_cup_ids:
            self.readout_stable_s, self.readout_anchor = 0., None
        self.readout_cup_ids = cup_ids
        if readout_condition:
            if self.readout_anchor is None:
                self.readout_anchor = {name: deepcopy(objects[name]) for name in readout_bodies}
            readout_condition = all(
                (abs(objects[name]['position_m'][2]-self.readout_anchor[name]['position_m'][2]) <= .004
                 if name in cup_grains else distance(objects[name], self.readout_anchor[name]) <= .004
                 and angle(objects[name], self.readout_anchor[name]) <= .10)
                for name in readout_bodies)
        if readout_condition:
            self.readout_stable_s += self.c['physics_dt_s']
        else:
            self.readout_stable_s, self.readout_anchor = 0., None
        from .weight_readout import readout
        display = readout(mass*1000 if self.readout_stable_s+EPS >= self.limits['stable_s'] else None,
                          stable_s=self.readout_stable_s)
        return bool(goal), {'inventory_ids': inventory, 'cup_sample_mass_g': mass*1000,
            'initial_sample_mass_g': p['initial_granule_mass_kg']*1000,
            'cumulative_spill_mass_g': spill*1000, 'scooped_ids': sorted(self.scooped),
            'cup_grains_supported': grains_supported,
            'cup_supported_and_unloaded': bool(supported and unloaded),
            'balance_display': display,
            'measurement': 'known nominal granule inventory; not a calibrated force balance'}

    def _liquid(self, frame, objects, contacts):
        water = plain(frame['reduced_dynamics'])
        if (water['native_step'] != frame['native_step']
                or abs(number(water['native_time_s'])-frame['native_time_s']) > EPS
                or water['inventory_closed'] is not True or water['errors']):
            raise ValueError('Incomplete or stale reduced liquid dynamics')
        fields = ['source_ml', 'tool_ml', 'in_flight_ml', 'cumulative_spill_ml',
                  'initial_total_ml', 'accounted_total_ml', 'aspirated_ml', 'dispensed_ml']
        if any(number(water[k]) < -EPS for k in fields):
            raise ValueError('Negative finite-volume inventory')
        if abs(water['initial_total_ml']-water['accounted_total_ml']) > 1.e-6:
            raise ValueError('Liquid inventory not conserved')
        p, t = self.p, frame['time_s']
        if (water['schema'] != p['hydraulics']['schema']
                or abs(water['initial_total_ml']-p['hydraulics']['initial_total_ml']) > 1.e-6
                or abs(sum(number(water[k]) for k in ('source_ml', 'tool_ml', 'in_flight_ml',
                       'cumulative_spill_ml'))+sum(number(v) for v in water['receivers_ml'].values())
                       -water['accounted_total_ml']) > 1.e-6):
            raise ValueError('Finite-volume account does not match its frozen inventory')
        if water['cumulative_spill_ml'] > p['spill_budget_ml']+EPS:
            self.failure = 'spill_limit_exceeded'
        returned = p['tool_return']
        rest = fixed(returned['initial_pose_xyzw'])
        geometry = (distance(objects['pipette'], rest) <= returned['position_tolerance_m']
                    and angle(objects['pipette'], rest) <= returned['angular_tolerance_rad'])
        tool_ok = self._supported('pipette', returned['support_fixture'], objects, contacts, t, geometry)
        vessel_ok = all(self._surface(k, objects, contacts, t)
            and upright(objects[k], self.limits['upright_tilt_rad']) for k in p['hydraulics']['containers'])
        goal = (all(abs(number(v)-p['target_per_receiver_ml']) <= p['target_tolerance_ml']+EPS
                    for v in water['receivers_ml'].values())
            and set(water['receivers_ml']) == set(p['hydraulics']['receiver_ids'])
            and water['in_flight_ml'] <= .5 and water['aspirated_ml'] > 0 and water['dispensed_ml'] > 0
            and water['cumulative_spill_ml'] <= p['spill_budget_ml']+EPS and tool_ok and vessel_ok
            and 'pipette' in self.moved)
        return bool(goal), {'water_balance': water, 'tool_returned': bool(tool_ok)}

    def _sample_quiet(self, objects, contacts, first):
        self.history.append(objects)
        for name, obj in objects.items():
            quiet = False
            if len(self.history) == self.pose_steps+1:
                prior = self.history[0][name]
                linear = distance(obj, prior)/self.limits['pose_window_s']
                angular = angle(obj, prior)/self.limits['pose_window_s']
                self.pose_speed[name] = {'linear_m_s': linear, 'angular_rad_s': angular}
                # Stable mass does not require a sphere to stop rolling sideways.
                # Its container membership must remain fixed through the dwell;
                # vertical bounce and all apparatus movement remain checked.
                granule = name.startswith('pellet_')
                vertical = abs(obj['position_m'][2]-prior['position_m'][2])/self.limits['pose_window_s']
                quiet = (not first and name not in contacts.robot
                    and (vertical <= .03 if granule else linear <= .03 and angular <= .3))
            if quiet:
                anchor = self.quiet_anchor.setdefault(name, deepcopy(obj))
                quiet = (abs(obj['position_m'][2]-anchor['position_m'][2]) <= .004
                    if name.startswith('pellet_') else distance(obj, anchor) <= .004 and angle(obj, anchor) <= .10)
            if quiet:
                self.quiet_s[name] += self.c['physics_dt_s']
            else:
                self.quiet_s[name] = 0.
                self.quiet_anchor.pop(name, None)

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
        if (type(step) is not int or step < 0 or type(baseline) is not bool
                or baseline != first or t < 0 or nt < 0 or abs(number(frame['physics_dt_s'])-dt) > EPS):
            raise ValueError('Invalid native baseline or clock')
        if first:
            if abs(t) > EPS or frame['contacts']:
                raise ValueError('Unscored contact-free reset required')
        elif (step != self.previous_step+1 or abs(t-self.previous_time-dt) > EPS
                or abs(nt-self.previous_native_time-dt) > 2.e-6):
            raise ValueError('Missing or out-of-order native samples')
        if set(frame['objects']) != set(self.specs):
            raise ValueError('Native body inventory differs')
        objects = {k: state(frame['objects'][k]) for k in self.specs}
        if first:
            for k, obj in objects.items():
                reset = fixed(self.specs[k]['initial_pose_xyzw'])
                if distance(obj, reset) > .003 or angle(obj, reset) > .03:
                    raise ValueError('Native reset differs from frozen asset')
            self.initial = deepcopy(objects)
        contacts = Contacts(frame['contacts'], self.paths, self.specs, t, self.limits['minimum_contact_impulse_ns'])
        if contacts.minimum_separation < -self.limits['maximum_penetration_m']:
            self.safety = 'physics_safety_stop'
        tools = set(self.specs)-set(self.p.get('pellet_ids', []))
        if not first and (contacts.floor & tools or any(objects[k]['position_m'][2] < .70 for k in tools)):
            self.failure = 'object_fell'
        self.support_diagnostics = {}
        for name, obj in objects.items():
            if distance(obj, self.initial[name]) >= self.limits['minimum_movement_m']:
                self.moved.add(name)
            self._surface(name, objects, contacts, t)
            nominal = self.specs[name].get('nominal_support')
            if nominal in self.fixed:
                self._supported(name, nominal, objects, contacts, t)
        # Diagnostic support for any actually loaded native fixture or body.
        for body, support in contacts.upward:
            if body in objects:
                self._supported(body, support, objects, contacts, t)
        self._sample_quiet(objects, contacts, first)
        if self.task_id == 'sample_sorting':
            goal, detail = self._sorting(objects, contacts, t)
        elif self.task_id == 'vial_racking':
            goal, detail = self._racking(objects, contacts, t)
        elif self.task_id == 'cap_placement':
            goal, detail = self._cap(frame, objects, contacts)
        elif self.task_id == 'solid_weighing':
            goal, detail = self._solid(frame, objects, contacts)
        else:
            goal, detail = self._liquid(frame, objects, contacts)
        condition = bool(goal and not first and not contacts.robot and all(v > 0 for v in self.quiet_s.values()))
        if condition:
            if self.stable_anchor is None:
                self.stable_anchor = deepcopy(objects)
            condition = all((abs(obj['position_m'][2]-self.stable_anchor[k]['position_m'][2]) <= .004
                if k.startswith('pellet_') else distance(obj, self.stable_anchor[k]) <= .004
                and angle(obj, self.stable_anchor[k]) <= .10)
                for k, obj in objects.items())
        if condition:
            self.stable_s += dt
        else:
            self.stable_s, self.stable_anchor = 0., None
        self.current = {'schema': STATE_SCHEMA, 'configuration_sha256': self.digest,
            'task_id': self.task_id, 'time_s': t, 'native_time_s': nt, 'native_step': step,
            'baseline': baseline, 'phase': 'settling' if goal else 'working',
            'reason': 'reset_baseline_not_scored' if first else
                'waiting_for_release_and_stability' if goal else 'task_in_progress',
            'released_stable_s': self.stable_s, 'terminal_failure': self.failure,
            'safety_stop': self.safety, 'detail': detail,
            'native_support_by_actor': {name: {
                'supported': any(k[0] == name for k in self.support_diagnostics),
                'released_quiet_native_s': self.quiet_s[name],
                'rolling_pose_speed': self.pose_speed.get(name),
                'witnesses': [{'support': k[1], 'source': v} for k, v in self.support_diagnostics.items() if k[0] == name]}
                for name in objects}}
        self.previous, self.previous_time, self.previous_native_time, self.previous_step = objects, t, nt, step

    def on_observe(self):
        result = deepcopy(self.current or {'schema': STATE_SCHEMA, 'configuration_sha256': self.digest})
        result.update(complete=self.current is not None and self.error is None,
                      errors=[] if self.error is None else [self.error], events=deepcopy(self.events))
        if self.task_id == 'solid_weighing' and (not result['complete'] or self.safety):
            from .weight_readout import readout
            result.setdefault('detail', {})['balance_display'] = readout()
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
        self.last_result = deepcopy(result)
        return result

    def update(self, observation):
        if self.invalid or self.failure or self.safety or self.success:
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
            elif (s['baseline'] is not False or abs(t-self.previous_time-.02) > EPS
                    or step-self.previous_step != round(.02/self.c['physics_dt_s'])):
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
            if self.failure is not None and self.failure not in {'object_fell', 'spill_limit_exceeded'}:
                raise ValueError('Unknown physical terminal reason')
            if self.safety is not None and self.safety != 'physics_safety_stop':
                raise ValueError('Unknown physical safety reason')
            if self.failure or self.safety:
                return self._result(self.safety or self.failure)
            self.success = not first and stable+EPS >= self.c['thresholds']['stable_s']
            if self.success:
                return self._result('success')
            if (observation.get('budget_exhausted') is True or t > self.c['thresholds']['max_time_s']+EPS
                    or number(observation.get('action_count', 0)) > self.c['thresholds']['max_actions']):
                return self._result('incomplete_budget')
            return self._result(s['reason'])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.failure = self.safety = None
            self.success = False
            self.stable_s = 0.
            return self._result('native_evidence_unknown')

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result)
