"""Native drawer sequence: open, retrieve, close, release with real tray support."""
from copy import deepcopy
from itertools import product

from .evaluation import SequenceEvaluator as PlacementEvaluator, distance, angle, norm, world_bounds, EPS
from .drawer_rules import THRESHOLDS, JOINT_PATH, number, quaternion


class SequenceEvaluator(PlacementEvaluator):
    def __init__(self, task_spec, *, trusted_normalized_input=False):
        config = task_spec.get('evaluation', {}).get('configuration', task_spec)
        if config.get('task_id') != 'drawer_retrieval':
            raise ValueError('Drawer referee rejects another study task')
        super().__init__(task_spec, trusted_normalized_input=trusted_normalized_input)
        self.t = THRESHOLDS
        self.opened = self.extracted = False
        self.extraction_armed = False

    def _geometry(self, ident, state):
        return [] if ident == 'drawer' else super()._geometry(ident, state)

    def _notebook_in_drawer(self, states):
        """Whole frozen notebook envelope inside the actual drawer envelope."""
        notebook, drawer = states['notebook'], states['drawer']
        low, high = self.config['objects']['drawer']['local_bounds_m']
        corners = product(*zip(*self.config['objects']['notebook']['local_bounds_m']))
        for corner in corners:
            delta = [notebook['position_m'][i]-drawer['position_m'][i]
                     + sum(notebook['rotation'][i][j]*corner[j] for j in range(3))
                     for i in range(3)]
            local = [sum(drawer['rotation'][i][j]*delta[i] for i in range(3))
                     for j in range(3)]
            if any(not low[i]-EPS <= local[i] <= high[i]+EPS for i in range(3)):
                return False
        return True

    def _result(self, *args, **kwargs):
        result = super()._result(*args, **kwargs)
        result['evaluator'] = 'study_drawer_evaluator.v1'
        result['sequence'] = {'opened': getattr(self, 'opened', False),
                              'extracted': getattr(self, 'extracted', False)}
        if not result['terminal']:
            if not result['sequence']['opened']: result['referee']['phase'] = 'opening'
            elif not result['sequence']['extracted']: result['referee']['phase'] = 'placement'
            elif any('drawer_not_closed' in r['reasons'] for r in self.status.values()):
                result['referee']['phase'] = 'closing'
        self.last_result = result
        return deepcopy(result)

    def update(self, observation):
        if self.terminal: return deepcopy(self.last_result)
        try:
            time, index, actions, episode = self._sampling(observation)
            if set(observation['objects']) != set(self.ids) or set(observation.get('joints', {})) != {JOINT_PATH}:
                raise ValueError('Drawer requires both native bodies and the actual prismatic coordinate')
            states = {name: self._state(observation['objects'][name]) for name in self.ids}
            contacts = self._contacts(observation)
            joint = observation['joints'][JOINT_PATH]
            q, qd = number(joint['position_m'], 'slide position'), number(joint['velocity_m_s'], 'slide velocity')
            if joint.get('kind') != 'prismatic' or joint.get('source') != 'derived_from_native_rigid_transforms_and_COM_velocities':
                raise ValueError('Only the native rigid-transform prismatic readback may score')
            origin = self.config['mechanism']['closed_position_m']
            if abs(q-(origin[0]-states['drawer']['position_m'][0])) > 1.e-5 or abs(qd+states['drawer']['linear_velocity_m_s'][0]) > 1.e-4:
                raise ValueError('Prismatic state and native drawer pose/velocity disagree')
            first, previous = self.previous is None, self.previous
            if first:
                if observation['contacts'] or abs(q) > self.t['initial_position_tolerance_m']:
                    raise ValueError('Drawer reset must be closed and unscored')
                for ident, state in states.items():
                    initial = self.config['objects'][ident]['initial_pose']
                    if distance(state['position_m'], initial['position_m']) > self.t['initial_position_tolerance_m'] or angle(state['quaternion_wxyz'], quaternion(initial['quaternion_wxyz'])) > self.t['initial_orientation_tolerance_rad']:
                        raise ValueError('Native drawer reset differs from frozen asset')
            self.previous, self.time, self.index, self.episode, self.actions = states, time, index, episode, actions
            if observation['budget_exhausted'] or time+EPS >= self.config['budget']['max_sim_time_s']:
                return self.finish('budget_exhausted')
            if first: return self._result('active', 'reset_baseline_not_scored')
            if any(r['other'] == 'floor' and r['separation'] <= self.t['support_gap_m'] for rows in contacts.values() for r in rows):
                return self._result('failure', 'object_touched_floor', terminal=True)
            if actions and q >= self.t['minimum_opening_m']:
                self.opened = True
            note_bounds = world_bounds(states['notebook'], self.config['objects']['notebook']['local_bounds_m'])
            clear = (note_bounds[1][0] <= self.config['mechanism']['notebook_extraction_max_x_m']
                     and note_bounds[0][2] > .856)
            # Opening later cannot retroactively qualify a notebook already
            # outside the case. Observe the pre-extraction side of the boundary
            # after a qualifying opening, then its outward crossing. A premature
            # extraction remains recoverable by returning and taking it out again.
            if self.opened and not clear and self._notebook_in_drawer(states):
                self.extraction_armed = True
            if self.extraction_armed and clear:
                self.extracted = True
            # Extraction must be witnessed clear of the case after opening.
            # A later move to the side tray does not erase that milestone.
            status = {}
            for ident, state in states.items():
                rows = contacts[ident]
                reasons = self._geometry(ident, state)
                geometry = not reasons
                if ident == 'drawer':
                    closed = -.004 <= q <= self.t['maximum_closed_gap_m']
                    rail_valid = (all(abs(state['position_m'][i]-origin[i]) <= .004 for i in (1,2))
                        and angle(state['quaternion_wxyz'], [1.,0.,0.,0.]) <= .06)
                    supported = rail_valid
                    if not closed: reasons.append('drawer_not_closed')
                    if not rail_valid: reasons.append('required_contact_support_missing')
                    if not self.opened: reasons.append('drawer_not_opened')
                    if not self.extracted: reasons.append('notebook_not_extracted')
                    geometry = closed and rail_valid
                else:
                    supported = self._support(ident, rows, states, {})
                    if not supported: reasons.append('required_contact_support_missing')
                    if any(r['other'] not in ('tray',) and not r['other'].startswith('robot/') for r in rows):
                        reasons.append('unexpected_contact')
                released = not any(r['other'].startswith('robot/') for r in rows)
                if not released: reasons.append('robot_contact_present')
                if any(r['separation'] < -self.t['maximum_penetration_m'] for r in rows): reasons.append('contact_penetration')
                linear = max(norm(state['linear_velocity_m_s']), distance(state['position_m'], previous[ident]['position_m'])/self.dt)
                angular = max(norm(state['angular_velocity_rad_s']), angle(state['quaternion_wxyz'], previous[ident]['quaternion_wxyz'])/self.dt)
                if linear > self.t['maximum_linear_speed_m_s'] or angular > self.t['maximum_angular_speed_rad_s']:
                    reasons.append('still_moving')
                status[ident] = {'geometry': geometry, 'supported': supported, 'released': released,
                    'reasons': reasons, 'linear_speed_m_s': linear, 'angular_speed_rad_s': angular}
            self.status = status
            if not actions: return self._result('active', 'robot_action_required')
            if any(row['reasons'] for row in status.values()):
                self.anchor = self.stable_since = None
                return self._result('active', 'stability_window_incomplete')
            if self.anchor is None:
                self.anchor, self.stable_since = deepcopy(states), time
            elif any(distance(s['position_m'], self.anchor[n]['position_m']) > self.t['maximum_position_drift_m'] or angle(s['quaternion_wxyz'], self.anchor[n]['quaternion_wxyz']) > self.t['maximum_orientation_drift_rad'] for n,s in states.items()):
                self.anchor, self.stable_since = deepcopy(states), time
                return self._result('active', 'stability_drift')
            stable = time-self.stable_since
            if stable+EPS >= self.t['minimum_stable_s']:
                return self._result('success', 'success', stable_s=stable, terminal=True)
            return self._result('active', 'stability_window_incomplete', stable_s=stable)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.anchor = self.stable_since = None
            return self._result('invalid', 'invalid_evidence', detail=str(error), terminal=True)
