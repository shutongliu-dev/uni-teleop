"""Privileged scripted diagnostic controller: robot joint targets only.

This is not a trained policy or a ranked Agent. Object poses are read only.
The controller deliberately does not inspect solver contacts or task scores;
those remain independent evaluator evidence.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from bench.geometry import transform
from bench.control.kinematics import URDFKinematics
from bench.isaacsim.controllers import smooth


def _finger_ik(hand, finger, point):
    from scipy.optimize import least_squares
    joints = [j for j in hand.active if f'_{finger}_' in j.name]
    names = [j.name for j in joints]
    def residual(q):
        return 100 * (hand.forward(dict(zip(names, q)))[f'right_{finger}_fingertip'][:3, 3] - point)
    result = least_squares(residual, [.5 if j.upper > .5 else 0 for j in joints],
                           bounds=([j.lower for j in joints], [j.upper for j in joints]), max_nfev=240)
    if np.linalg.norm(result.fun) > .35:
        raise ValueError(f'Finger IK residual {finger}: {result.fun}')
    return dict(zip(names, result.x.tolist()))


class SimpleTaskController:
    """``next_action`` accepts native or projected privileged observations.

    Measured joint state, simulation time and the cube pose are the only inputs.
    FK/IK produce drive targets, never simulation state writes or attachments.
    """
    def __init__(self, scene: dict, task: dict, *, variant='scripted'):
        if variant not in ('scripted', 'hold', 'hover'):
            raise ValueError('Unknown diagnostic controller variant')
        if scene['robot'].get('hand_id', 'sharpa_w01_right') != 'sharpa_w01_right':
            raise ValueError('This diagnostic controller is calibrated for SHARPA W01 only')
        self.variant = variant
        self.task = task
        self.task_id = task['id']
        self.mode = 'push' if 'push' in self.task_id else 'place' if 'place' in self.task_id else 'lift'
        self.object_id = task.get('object_id', 'cube')
        obj = next(o for o in scene['objects'] if o['id'] == self.object_id)
        self.source = np.asarray(obj['position_m'], dtype=float)
        self.goal = np.asarray(task.get('target_position_m',
            [self.source[0] + (0.07 if self.mode == 'place' else 0), self.source[1] + .10, self.source[2] - .001]), dtype=float)
        assets = Path(scene['robot']['assets_root'])
        self.arm = URDFKinematics(assets / 'robots/franka/fr3/fr3_no_hand.absolute.urdf')
        self.hand = URDFKinematics(assets / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf')
        self.base = np.asarray(scene['robot']['base_position_m'], dtype=float)
        self.rotation = np.diag([1., -1., -1.])
        self.limits = {j.name: (j.lower, j.upper) for j in self.arm.active + self.hand.active}
        self.arm_seed = dict(zip(self.arm.names, scene['robot']['arm_home_rad']))
        folded = {j.name: float(np.clip(1.2 if any(f'_{f}_' in j.name for f in ('middle','ring','pinky'))
                    and j.name.endswith(('_FE','_PIP','_DIP','_IP')) else 0, j.lower, j.upper)) for j in self.hand.active}
        self.grasp_center = np.array([.070, .050, .150])
        self.hand_open = dict(folded)
        self.hand_closed = dict(folded)
        for pose, gap in ((self.hand_open, .040), (self.hand_closed, .019)):
            pose.update(_finger_ik(self.hand, 'thumb', self.grasp_center + [0, gap, 0]))
            pose.update(_finger_ik(self.hand, 'index', self.grasp_center + [0, -gap, 0]))
        self.push_hand = dict(folded)
        self.push_offset = self.hand.forward(self.push_hand)['right_index_fingertip'][:3, 3]
        self.previous = None
        self.velocity = {n: 0. for n in self.limits}
        self.integral = {n: 0. for n in self.arm.names}
        self.phase_index = -1
        self.phase_start = 0.
        self.phases = []
        self.phase_initial = None
        self.phase_final = None
        self.cartesian_arm_waypoints = None
        self.done = False
        self.phase = 'uninitialized'
        self.ik_errors = []
        self.trace = []

    def _arm_at(self, position):
        result = self.arm.inverse(transform(np.asarray(position) - self.base, self.rotation),
                    'fr3_link8', self.arm_seed, iterations=140, multi_starts=8,
                    position_tolerance=.00008, rotation_tolerance=.001)
        if not result['converged']:
            raise ValueError(f'Unreachable {self.phase} arm target: {result}')
        self.arm_seed = result['joint_positions_rad']
        self.ik_errors.append([result['position_error_m'], result['rotation_error_rad']])
        return dict(self.arm_seed)

    def _build_phases(self, observation):
        # Only the initial observed object position is used. The scripted policy
        # does not peek at evaluator goals/contacts to assert task completion.
        self.source = np.asarray(observation['objects'][self.object_id]['position_m'], dtype=float)
        if self.mode == 'push':
            a = self.source + [0, -.080, .13]
            b = self.source + [0, -.080, 0]
            c = np.array([self.goal[0], self.goal[1] - .023, self.source[2]])
            d = c + [0, 0, .13]
            if self.variant == 'hover':
                a[2] += .10; b[2] += .10; c[2] += .10; d[2] += .10
            def push_phase(name, duration, tip):
                return (name, duration, tip - self.rotation @ self.push_offset, self.push_hand)
            self.phases = [push_phase('approach_above', 7., a), push_phase('descend', 3., b),
                push_phase('push', 5., c), push_phase('pause', .5, c),
                push_phase('withdraw', 2., d), push_phase('settle_released', 1.5, d)]
        else:
            # The upper-half pinch clears the thumb proximal mesh above the
            # 60 mm cube; aiming at its centre intersects that mesh on descent.
            f = self.source - self.rotation @ self.grasp_center + [0, 0, .020]
            up = f + [0, 0, .11]
            dest = f + np.r_[self.goal[:2] - self.source[:2], 0.]
            self.phases = [('approach_above', 7., up, self.hand_open),
                ('descend', 3., f, self.hand_open), ('close', 2.5, f, self.hand_closed),
                ('lift', 3., up, self.hand_closed), ('hold', 2., up, self.hand_closed)]
            if self.mode == 'place':
                self.phases += [('transfer', 3., dest + [0,0,.11], self.hand_closed),
                    ('lower', 3., dest, self.hand_closed), ('release', 2., dest, self.hand_open),
                    ('withdraw', 2., dest + [0,0,.13], self.hand_open),
                    ('settle_released', 1.5, dest + [0,0,.13], self.hand_open)]

    def next_action(self, observation: dict) -> dict:
        q = observation['joint_positions_rad']
        now = float(observation.get('sim_time_s', observation.get('time_s', 0.)))
        if self.previous is None:
            self.previous = {n: float(q[n]) for n in self.limits}
            self._build_phases(observation)
        if self.variant == 'hold':
            self.phase = 'hold_control'
            self.done = now >= 20.
            return {'joint_positions_rad': dict(self.previous), 'duration_s': .02}
        if self.phase_index < 0 or now >= self.phase_start + self.phases[self.phase_index][1] - 1e-8:
            if self.phase_index + 1 < len(self.phases):
                self.phase_index += 1
                self.phase_start = now
                self.phase, _, point, hand = self.phases[self.phase_index]
                self.phase_initial = dict(self.previous)
                self.phase_final = {**self._arm_at(point), **hand}
                self.cartesian_arm_waypoints = None
                if self.mode == 'push' and self.phase == 'push':
                    # A single joint interpolation bows sideways and contacts
                    # the cube off-centre. Short IK segments keep the pushing
                    # fingertip on the intended Cartesian line. The same
                    # joint velocity/acceleration limits remain authoritative.
                    start = self.arm.forward(self.phase_initial)['fr3_link8'][:3, 3] + self.base
                    self.arm_seed = {n: self.phase_initial[n] for n in self.arm.names}
                    self.cartesian_arm_waypoints = [dict(self.arm_seed)]
                    for fraction in np.linspace(0., 1., 11)[1:]:
                        self.cartesian_arm_waypoints.append(self._arm_at((1-fraction)*start + fraction*np.asarray(point)))
                self.trace.append({'phase': self.phase, 'start_sim_s': now,
                    'flange_target_world_m': np.asarray(point).tolist(),
                    'observed_object_position_m': observation['objects'][self.object_id]['position_m']})
            else:
                self.done = True
        duration = self.phases[self.phase_index][1]
        alpha = smooth(float(np.clip((now - self.phase_start + .02) / duration, 0, 1)))
        command = {}
        for name in self.limits:
            lo, hi = self.limits[name]
            nominal = (1-alpha)*self.phase_initial[name] + alpha*self.phase_final[name]
            if self.cartesian_arm_waypoints is not None and name in self.arm.names:
                coordinate = alpha * (len(self.cartesian_arm_waypoints)-1)
                segment = min(int(coordinate), len(self.cartesian_arm_waypoints)-2)
                weight = coordinate-segment
                nominal = ((1-weight)*self.cartesian_arm_waypoints[segment][name]
                           + weight*self.cartesian_arm_waypoints[segment+1][name])
            # A small target-space integral compensates finite drive stiffness.
            # It is explicitly joint feedback, not a force/state writeback.
            if name in self.integral:
                self.integral[name] = float(np.clip(self.integral[name] + 2.0*(nominal-q[name])*.02, -.06, .06))
            target = float(np.clip(nominal + self.integral.get(name, 0.), lo, hi))
            max_velocity = .65 if name.startswith('fr3_') else 1.0
            vel = float(np.clip((target - self.previous[name])/.02, -max_velocity, max_velocity))
            vel = float(np.clip(vel, self.velocity[name]-.04, self.velocity[name]+.04))
            command[name] = float(np.clip(self.previous[name] + vel*.02, lo, hi))
            self.velocity[name] = (command[name] - self.previous[name])/.02
        self.previous = command
        return {'joint_positions_rad': command, 'duration_s': .02}

    def diagnostic(self):
        return {'controller': 'simple_tasks_scripted_joint_only_v1', 'variant': self.variant,
                'trained_policy': False, 'ranked_agent': False,
                'observations_used': ['sim_time_s', 'joint_positions_rad', 'initial object position_m'],
                'object_state_writeback': False, 'attachments': False,
                'phase': self.phase, 'done': self.done, 'phase_trace': self.trace,
                'maximum_ik_position_error_m': max((x[0] for x in self.ik_errors), default=0.),
                'maximum_ik_rotation_error_rad': max((x[1] for x in self.ik_errors), default=0.)}
