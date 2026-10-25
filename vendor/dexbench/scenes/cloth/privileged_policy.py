"""Read-only cloth-vertex feedback for a bounded, continuous-contact baseline.

The script emits ordinary model geometry JSON. It does not integrate physics,
write cloth state, substitute a controller, or decide benchmark acceptance.
URDF clearances are planning guards; the independent native observer remains
responsible for cooked geometry and every physics substep.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from bench.geometry import matrix_rotvec, quaternion_matrix, rpy_matrix, transform
from bench.control.kinematics import URDFKinematics
from bench.paths import DATA_ROOT


TASK_ID = 'cloth_table_drag_y_05cm_candidate_01'
DEFAULT_CLOTH_TASK_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
    'bench_capability_repair_20261012T054520Z/cloth_ack_plateau_full_native_01/release')
HAND_SHA256 = 'ff9ffc2af5069c09378d4ffc7870d53c507a50941849be79b7dfe7c5f9dcb6cc'


def finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(name+' has invalid shape or nonfinite coordinates')
    return array


def cap(vector, maximum):
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1e-15))


class HandCollisionPrior:
    """Measured robot-only URDF collision geometry, never a PhysX certificate."""
    def __init__(self, assets):
        path = Path(assets) / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf'
        if hashlib.sha256(path.read_bytes()).hexdigest() != HAND_SHA256:
            raise ValueError('Cloth hand prior requires the pinned W01 URDF')
        self.kin = URDFKinematics(path)
        self.parts = []
        self.sources = {str(path): HAND_SHA256}
        for link in ET.parse(path).getroot().findall('link'):
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Unsupported cloth hand collision representation')
                source = (path.parent / mesh.get('filename')).resolve()
                self.sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                vertices = np.asarray(trimesh.load(source, force='mesh', process=False).vertices, float)
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                origin = collision.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise ValueError('Invalid hand collision mesh')
                self.parts.append((link.get('name'), (vertices * scale) @ rpy_matrix(rpy).T + xyz))

    def measure(self, observation, *, wrist=None):
        proprio = observation['proprioception']
        poses = self.kin.forward(proprio['joint_positions_rad'])
        wrist = wrist or proprio['geometry']['wrist']
        base = transform(wrist['position_m'], quaternion_matrix(wrist['quaternion_wxyz']))
        points = {}
        for name, vertices in self.parts:
            pose = base @ poses[name]
            points[name] = vertices @ pose[:3, :3].T + pose[:3, 3]
        pad = points['right_index_elastomer']
        return pad[pad[:, 2].argmin()].copy(), min(float(v[:, 2].min())
            for name, v in points.items() if name != 'right_index_elastomer')


class PrivilegedClothDragPolicy:
    VERSION = '1.0.7'
    policy_id = 'privileged_cloth_continuous_drag_v1'
    ALIGN_HEIGHT_M = .0132
    ALIGN_COMPLETION_M = .0005
    ALIGN_PHASE_RADIUS_M = .0008
    APPROACH_FAR_COMPLETION_M = .00025
    APPROACH_NEAR_COMPLETION_M = .00005
    APPROACH_PHASE_RADIUS_M = .000075
    PRESSURE_COMPLETION_M = .000025
    DRAG_COARSE_DEFICIT_M = .000025
    MAX_PRESSURE_STEPS = 5
    MAX_PLATEAU_PROBES = 2

    def __init__(self, scene, task, *, prior=None):
        if task.get('id') != TASK_ID or task['public_goal']['operation'] != 'cloth_table_drag':
            raise ValueError('This cloth baseline requires the original five-centimetre drag task')
        self.contract = deepcopy(task['public_goal']['success_contract'])
        if self.contract['all_vertices_minimum_y_translation_m'] != .05:
            raise ValueError('Cloth benchmark distance changed')
        self.reference = finite(task['evaluation']['expected_reset_vertices_m'],
                                (task['evaluation']['vertex_count'], 3), 'frozen cloth reset')
        self.triangles = np.asarray(task['evaluation']['triangles'], dtype=int)
        self.prior = prior or HandCollisionPrior(scene['robot']['assets_root'])
        self.table = float(self.contract['table_top_z_m'])
        self.goal = finite(self.contract['centroid_goal_xy_m'], (2,), 'cloth centroid goal')
        self.phase = 'initial'
        self.actions = 0
        self.action_budget = 128
        self.pending = None
        self.rotation = None
        self.episode_id = None
        self.pressure_count = 0
        self.pressure_progress = []
        self.pressure_plateau = False
        self.plateau_probe_active = False
        self.plateau_probe_steps = 0
        self.drag_height_anchor_m = None
        self.drag_steps = 0
        self.good_drag_steps = 0
        self.bad_drag_steps = 0
        self.drag_step_m = .001
        self.release_count = 0
        self.finished_reason = None
        self.last_metrics = {}
        self.trace = []

    def configure_budget(self, count):
        if type(count) is not int or not 12 <= count <= 128:
            raise ValueError('Cloth baseline requires 12 to 128 actions including release reserve')
        self.action_budget = count

    def _state(self, observation, oracle):
        for state in (observation, oracle):
            if (type(state.get('sequence')) is not int or state['sequence'] < 0
                    or type(state.get('sim_time_s')) not in (int, float)
                    or not math.isfinite(state['sim_time_s']) or state['sim_time_s'] < 0):
                raise ValueError('Cloth decision boundary requires finite time and integer sequence')
        if not isinstance(observation.get('episode_id'), str) or not observation['episode_id']:
            raise ValueError('Cloth observation requires an episode identity')
        if (observation['profile'] != 'vision_proprio'
                or observation['sequence'] != oracle['sequence']
                or observation['sim_time_s'] != oracle['sim_time_s']):
            raise ValueError('Cloth oracle and public observation must share a decision boundary')
        cloth = oracle.get('cloth', {})
        vertices = finite(cloth.get('vertices_m'), self.reference.shape, 'cloth vertices')
        velocities = finite(cloth.get('velocities_m_s'), self.reference.shape, 'cloth velocities')
        if not np.array_equal(np.asarray(cloth.get('triangles')), self.triangles):
            raise ValueError('Cloth topology changed')
        if self.episode_id is None:
            if observation['sequence'] != 0 or observation['sim_time_s'] != 0.:
                raise ValueError('Cloth baseline requires original fresh reset')
            if np.max(np.abs(vertices - self.reference)) > self.contract['reset_vertex_tolerance_m']:
                raise ValueError('Cloth reset differs from frozen original task')
            self.episode_id = observation['episode_id']
        elif observation['episode_id'] != self.episode_id:
            raise ValueError('Cloth episode changed')
        centroid = vertices.mean(axis=0)
        low, nonpad = self.prior.measure(observation)
        if not np.isfinite(low).all() or not np.isfinite(nonpad):
            raise ValueError('Invalid robot prior geometry')
        if low[2] < self.table + .0023 or nonpad < self.table + .004:
            raise ValueError('Measured robot violates original cloth/table clearance guard')
        path = np.asarray(self.contract['whole_episode_vertex_bounds_m'])
        if np.any(vertices < path[0] - 1e-8) or np.any(vertices > path[1] + 1e-8):
            raise ValueError('Cloth left the frozen whole-episode bounds')
        self.last_metrics = {'minimum_vertex_translation_y_m': float(np.min(vertices[:, 1] - self.reference[:, 1])),
            'centroid_m': centroid.tolist(), 'centroid_goal_error_m': float(np.linalg.norm(centroid[:2] - self.goal)),
            'maximum_vertex_speed_m_s': float(np.linalg.norm(velocities, axis=1).max()),
            'pad_table_gap_m': float(low[2] - self.table), 'nonpad_table_gap_m': float(nonpad - self.table)}
        return vertices, centroid, low

    def _emit(self, action, observation, vertices, low, *, delta=None):
        if self.actions >= self.action_budget:
            raise ValueError('Cloth action reserve exhausted')
        self.pending = {'phase': self.phase, 'sequence': observation['sequence'],
            'time_s': observation['sim_time_s'], 'low': low.copy(), 'vertices': vertices.copy(),
            'wrist_position_m': np.asarray(observation['proprioception']['geometry']['wrist']['position_m']).copy(),
            'delta': None if delta is None else np.asarray(delta).copy(), 'action': deepcopy(action)}
        self.actions += 1
        return action

    def _hold(self, duration, observation, vertices, low):
        return self._emit({'action': 'hold', 'duration_s': duration,
            'reason': 'privileged_cloth_'+self.phase}, observation, vertices, low)

    def _move(self, pad_delta, observation, vertices, low, *, precision=.0001, lateral=False):
        wrist = deepcopy(observation['proprioception']['geometry']['wrist'])
        delta = np.asarray(pad_delta, dtype=float)
        step_bound = (.002 if lateral or self.phase == 'release' else
                      .0002 if self.phase == 'pressure' or
                          (self.phase == 'approach' and low[2] < self.table + .006) else
                      .002 if self.phase == 'approach' else .01)
        if np.linalg.norm(delta) > step_bound + 1e-12:
            raise ValueError('Cloth pad request exceeds the phase step bound')
        if lateral:
            # Keep the latest acknowledged wrist height/orientation. Reapplying
            # a nominal hand pose here caused old pressure/drag compensation loss.
            rotation_delta = np.zeros(3)
        else:
            old_low = low.copy()
            wrist['quaternion_wxyz'] = self.rotation.tolist()
            compensated_low, _ = self.prior.measure(observation, wrist=wrist)
            correction = old_low - compensated_low
            if np.linalg.norm(correction + delta) > step_bound + 1e-12:
                if np.linalg.norm(correction) > step_bound + 1e-12:
                    raise ValueError('Wrist orientation compensation exceeds the phase step bound')
                a, b = float(delta @ delta), float(2 * correction @ delta)
                c = float(correction @ correction) - step_bound**2
                if a <= 0:
                    raise ValueError('No bounded pad progress')
                alpha = min(1., max(0., (-b + np.sqrt(max(0., b*b - 4*a*c))) / (2*a)))
                if alpha <= 1e-9:
                    raise ValueError('No bounded pad progress')
                delta = alpha * delta
            delta = delta + correction
            current = observation['proprioception']['geometry']['wrist']['quaternion_wxyz']
            rotation_delta = matrix_rotvec(quaternion_matrix(self.rotation) @ quaternion_matrix(current).T)
        if np.linalg.norm(delta) > .05 + 1e-12 or np.linalg.norm(rotation_delta) > .15 + 1e-12:
            raise ValueError('Cloth request exceeds original public geometry bounds')
        if np.linalg.norm(delta) > step_bound + 1e-12:
            raise ValueError('Compensated cloth wrist target exceeds the phase step bound')
        wrist['position_m'] = (np.asarray(wrist['position_m']) + delta).tolist()
        end_low, end_nonpad = self.prior.measure(observation, wrist=wrist)
        if end_low[2] < self.table + .0026 - 1e-12 or end_nonpad < self.table + .004:
            raise ValueError('Proposed cloth motion violates original endpoint clearance guard')
        return self._emit({'action': 'move_geometry', 'duration_s': 5.,
            'wrist_delta_m': delta.tolist(), 'wrist_rotation_delta_rad': rotation_delta.tolist(),
            'position_tolerance_m': precision, 'orientation_tolerance_rad': .005,
            'reason': 'privileged_cloth_'+self.phase}, observation, vertices, low, delta=pad_delta)

    def _feedback(self, observation, feedback, vertices, low):
        if self.pending is None:
            return
        pending = self.pending
        if not isinstance(feedback, dict):
            raise ValueError('Cloth primitive lacks a fresh matching acknowledgement')
        action_id = feedback.get('action_id')
        for name in ('sim_time_s', 'elapsed_sim_s'):
            value = feedback.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Cloth acknowledgement requires finite nonnegative '+name)
        if (not isinstance(action_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', action_id)
                or action_id != observation.get('last_action_id')
                or observation['sequence'] <= pending['sequence']
                or abs(feedback['sim_time_s'] - observation['sim_time_s']) > 1e-6):
            raise ValueError('Cloth primitive lacks a fresh matching acknowledgement')
        if feedback.get('status') != 'completed' or observation.get('last_action_status') != 'completed':
            raise ValueError('Cloth primitive failed; no further pressure is authorized')
        elapsed = feedback['elapsed_sim_s']
        if abs(observation['sim_time_s'] - pending['time_s'] - elapsed) > 1e-6:
            raise ValueError('Cloth acknowledgement duration mismatch')
        if pending['action']['action'] == 'hold':
            if elapsed + 1e-8 < pending['action']['duration_s']:
                raise ValueError('Cloth timed hold was shortened')
            if pending['phase'] == 'contact_settle' and self.drag_height_anchor_m is None:
                self.drag_height_anchor_m = float(low[2])
                self.trace.append({'event': 'drag_height_anchor_acknowledged',
                    'pad_height_m': self.drag_height_anchor_m})
            self.pending = None
            if pending['phase'] == 'release_hold':
                self.phase = 'done'
            return
        self.pending = None
        if pending['phase'] == 'pressure':
            down = float(pending['low'][2] - low[2])
            wrist = finite(observation['proprioception']['geometry']['wrist']['position_m'], (3,), 'acknowledged wrist')
            wrist_down = float(pending['wrist_position_m'][2] - wrist[2])
            target = pending['wrist_position_m'] + np.asarray(pending['action']['wrist_delta_m'])
            target_error = float(np.linalg.norm(wrist - target))
            prior_progress = any(value >= .00001 for value in self.pressure_progress)
            self.pressure_progress.append(down)
            self.trace.append({'event': 'pressure', 'actual_pad_down_m': down,
                'actual_wrist_down_m': wrist_down, 'wrist_target_error_m': target_error})
            if down < .00001:
                if (prior_progress and down >= 0. and wrist_down >= .00001
                        and target_error <= pending['action']['position_tolerance_m'] + 1e-12):
                    # Joint compliance can cancel pad motion despite a reached
                    # wrist request. This authorizes only two geometric probes,
                    # never a claim of contact, support force, or friction.
                    self.pressure_plateau = self.plateau_probe_active = True
                    self.phase = 'pressure'
                    self.trace.append({'event': 'pressure_plateau_probe_authorized',
                        'maximum_probes': self.MAX_PLATEAU_PROBES, 'contact_certified': False})
                else:
                    self.finished_reason = 'pressure_progress_unverified'
                    self.phase = 'release'
            else:
                self.phase = 'pressure_hold'
        if pending['phase'] == 'drag':
            direction = pending['delta'][:2] / np.linalg.norm(pending['delta'][:2])
            pad_progress = float((low[:2] - pending['low'][:2]) @ direction)
            node_projection = (vertices[:, :2] - pending['vertices'][:, :2]) @ direction
            mean_progress = float(node_projection.mean())
            gain = mean_progress / pad_progress if pad_progress > 1e-5 else None
            useful = gain is not None and mean_progress > .00015 and float(node_projection.min()) > -.0001
            self.good_drag_steps = self.good_drag_steps + 1 if useful else 0
            self.bad_drag_steps = 0 if useful else self.bad_drag_steps + 1
            self.drag_steps += 1
            self.trace.append({'event': 'drag', 'pad_progress_m': pad_progress,
                'mean_node_progress_m': mean_progress, 'minimum_node_progress_m': float(node_projection.min()),
                'observed_response_ratio': gain, 'ratio_is_force_or_friction': False})
            self.drag_step_m = .002 if self.good_drag_steps >= 3 else .001
            if self.bad_drag_steps >= 2:
                self.finished_reason = 'cloth_traction_progress_lost'
                self.phase = 'release'
            if self.plateau_probe_active:
                self.plateau_probe_steps += 1
                if self.plateau_probe_steps >= self.MAX_PLATEAU_PROBES:
                    self.plateau_probe_active = False
                    if self.good_drag_steps < self.MAX_PLATEAU_PROBES and self.phase != 'release':
                        self.finished_reason = 'cloth_plateau_traction_unverified'
                        self.phase = 'release'

    def next_decision(self, observation, oracle, previous_feedback=None):
        vertices, centroid, low = self._state(observation, oracle)
        self._feedback(observation, previous_feedback, vertices, low)
        if (self.phase in ('pressure', 'pressure_hold', 'contact_settle', 'drag')
                and low[2] < self.table + .0026 - 1e-12):
            # A valid completed move or hold may settle below the planning
            # floor while still passing the raw-state checks above.
            # Preserve that measurement and release before another contact move.
            self.finished_reason = 'pressure_endpoint_margin_lost'
            self.phase = 'release'
        if self.phase == 'initial':
            self.rotation = np.asarray(observation['proprioception']['geometry']['wrist']['quaternion_wxyz'])
            self.phase = 'settle'
            return self._hold(.4, observation, vertices, low)
        if self.phase == 'settle':
            self.phase = 'align'
        if self.actions >= self.action_budget - 8 and self.phase not in ('release', 'release_hold', 'done'):
            self.finished_reason = 'diagnostic_action_budget_release_reserve'
            self.phase = 'release'
        if self.phase == 'align':
            # The 13.2 mm target leaves margin above the unchanged 12 mm
            # horizontal-motion clearance even at the phase radius boundary.
            target = np.r_[centroid[:2], self.table + self.ALIGN_HEIGHT_M]
            delta = target - low
            if np.linalg.norm(delta) > self.ALIGN_PHASE_RADIUS_M:
                # Large horizontal motion is allowed only at retracted height.
                if low[2] < self.table + .012:
                    delta = np.array([0., 0., target[2] - low[2]])
                return self._move(cap(delta, .01), observation, vertices, low,
                                  precision=self.ALIGN_COMPLETION_M)
            self.phase = 'approach'
        if self.phase == 'approach':
            down = self.table + .0031 - low[2]
            if down < -self.APPROACH_PHASE_RADIUS_M:
                near = low[2] < self.table + .006
                step = .0002 if near else .002
                precision = self.APPROACH_NEAR_COMPLETION_M if near else self.APPROACH_FAR_COMPLETION_M
                return self._move([0., 0., max(down, -step)], observation, vertices, low, precision=precision)
            self.phase = 'pressure'
        if self.phase == 'pressure_hold':
            self.phase = 'pressure'
            return self._hold(.08, observation, vertices, low)
        if self.phase == 'pressure':
            remaining = low[2] - (self.table + .0026)
            # Match the endpoint guard's 1 pm roundoff allowance; a move
            # already within completion tolerance cannot prove new pressure.
            if (not self.pressure_plateau and self.pressure_count < self.MAX_PRESSURE_STEPS
                    and remaining > self.PRESSURE_COMPLETION_M + 1e-12):
                self.pressure_count += 1
                return self._move([0., 0., -min(remaining, .0001)], observation, vertices, low,
                                  precision=self.PRESSURE_COMPLETION_M)
            if not self.pressure_progress:
                self.finished_reason = 'no_safe_pressure_progress'
                self.phase = 'release'
            else:
                self.phase = 'contact_settle'
                return self._hold(.3, observation, vertices, low)
        if self.phase == 'contact_settle':
            self.phase = 'drag'
        if self.phase == 'drag':
            if self.drag_height_anchor_m is None:
                raise ValueError('Cloth drag requires an acknowledged contact-settle height anchor')
            minimum = self.last_metrics['minimum_vertex_translation_y_m']
            error = self.last_metrics['centroid_goal_error_m']
            if minimum >= .055 and error <= .006:
                self.finished_reason = 'geometric_target_reached_independent_score_required'
                self.phase = 'release'
            else:
                vector = self.goal - centroid[:2]
                if np.linalg.norm(vector) < .0002:
                    self.finished_reason = 'centroid_reached_but_entire_cloth_not_translated'
                    self.phase = 'release'
                else:
                    step = .001 if self.plateau_probe_active else self.drag_step_m
                    # Compensate measured hand compliance, not only wrist-Z
                    # error. Never press farther down to recover lost traction.
                    up = max(0., self.drag_height_anchor_m - float(low[2]))
                    if up >= step:
                        self.finished_reason = 'cloth_height_compensation_exceeds_step_bound'
                        self.phase = 'release'
                    else:
                        horizontal = math.sqrt(max(0., step * step - up * up))
                        # Preserve the full up-only correction. Tighten the
                        # combined wrist objective only beyond the deficit
                        # band; match the existing 1 pm arithmetic allowance.
                        precision = (self.PRESSURE_COMPLETION_M
                            if up > self.DRAG_COARSE_DEFICIT_M + 1e-12 else .0001)
                        return self._move(np.r_[cap(vector, horizontal), up], observation,
                            vertices, low, precision=precision, lateral=True)
        if self.phase == 'release':
            up = self.table + .0122 - low[2]
            if up > .00015:
                if self.release_count >= 7:
                    raise ValueError('Cloth release failed to reach clearance within reserved actions')
                self.release_count += 1
                return self._move([0., 0., min(up, .002)], observation, vertices, low)
            self.phase = 'release_hold'
            return self._hold(.4, observation, vertices, low)
        if self.phase == 'done':
            return {'action': 'stop', 'reason': self.finished_reason or 'cloth_release_complete'}
        raise ValueError('Unknown cloth strategy phase '+self.phase)

    def diagnostic(self):
        return {'policy_id': self.policy_id, 'version': self.VERSION, 'phase': self.phase,
            'decisions': self.actions, 'action_budget': self.action_budget, 'drag_steps': self.drag_steps,
            'drag_step_m': self.drag_step_m, 'pressure_steps': self.pressure_count, 'maximum_pressure_steps': self.MAX_PRESSURE_STEPS,
            'pressure_plateau': self.pressure_plateau, 'plateau_probe_active': self.plateau_probe_active,
            'plateau_probe_steps': self.plateau_probe_steps, 'maximum_plateau_probes': self.MAX_PLATEAU_PROBES,
            'plateau_is_contact_certificate': False,
            'drag_height_anchor_m': self.drag_height_anchor_m,
            'drag_height_compensation': 'measured_pad_up_only_within_total_step_bound',
            'finished_reason': self.finished_reason, 'metrics': deepcopy(self.last_metrics),
            'trace': deepcopy(self.trace), 'hand_geometry_sources': deepcopy(self.prior.sources),
            'precision_schedule_m': {'align': self.ALIGN_COMPLETION_M,
                'far_approach': self.APPROACH_FAR_COMPLETION_M, 'near_approach': self.APPROACH_NEAR_COMPLETION_M,
                'pressure': self.PRESSURE_COMPLETION_M,
                'drag_coarse_deficit': self.DRAG_COARSE_DEFICIT_M},
            'phase_geometry_m': {'align_height': self.ALIGN_HEIGHT_M,
                'align_radius': self.ALIGN_PHASE_RADIUS_M, 'approach_radius': self.APPROACH_PHASE_RADIUS_M},
            'model_calls': 0, 'object_state_writeback': False, 'score_inputs': False,
            'scope': 'Privileged engineering trajectory; native physics and task acceptance remain independent.'}
