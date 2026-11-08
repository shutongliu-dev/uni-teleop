"""Single-learner goals adjudicated from complete native frames, never RPCs.

The formal match is only a geometry/support evidence consumer here. No formal
turn, score or winner is mutated. Its frozen numerical thresholds are reused.
"""
from copy import deepcopy
import math

from ..expansion_referee import ExpansionMatch, _angle, _distance, _rotation
from .cases import verify_case


class PracticeReferee:
    def __init__(self, task):
        self.task = deepcopy(task)
        self.case = verify_case(task['practice_case'], task['asset_metadata'])
        self.public = self.case['public']
        self.parent = self.public['parent_task_id']
        self.helper = ExpansionMatch(self.parent, asset_metadata=task['asset_metadata'])
        self.thresholds = self.helper.thresholds
        self.player = 'player_0' if self.public['active_side'] == 'left' else 'player_1'
        self.other = 'player_1' if self.player == 'player_0' else 'player_0'
        self.expected = {n: (tuple(p['position_m']), tuple(p['quaternion_wxyz'])) for n, p in self.case['reset'].items()}
        self.stage_index = 0
        self.state = 'ongoing'
        self.ended = False
        self.reason = 'awaiting_native_evidence'
        self.last_frame = None
        self.hand_origins = {}
        self.contact_seen, self.motion_seen = set(), set()
        self.stable_start, self.stable_positions = None, {}
        self.confirmations = []
        self.contact_rows = ()
        self.shot = {'started': False, 'released': False, 'crossed': False, 'obstacle_contact': False}
        self.maximum_points = None

    @property
    def terminal(self):
        return self.ended or self.state != 'ongoing'

    @property
    def stage(self):
        return self.public['goal']['stages'][min(self.stage_index, len(self.public['goal']['stages'])-1)]

    def score(self):
        return {'schema': 'card_rsi_practice_score.v1', 'success': self.state == 'success',
            'terminal': self.terminal, 'reason': self.reason,
            'irrecoverable_reason': self.reason if self.state in ('task_failure', 'safety_stop', 'evaluator_invalid') else None,
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': self.state,
                'phase': 'complete' if self.state == 'success' else 'incomplete' if self.terminal else 'practice',
                'reason_codes': [self.reason], 'terminal': self.state != 'ongoing', 'recoverable': self.state == 'ongoing'},
            # Only this small projection is permitted as public progress. No
            # poses/impulses, reset pairs or private fingerprints occur here.
            'practice_progress': {'stage_index': self.stage_index,
                'stage_count': len(self.public['goal']['stages']), 'confirmed_stages': len(self.confirmations)},
            'formal_game_result': False}

    def _result(self, reason, *, state=None):
        self.reason = reason
        if state is not None:
            self.state = state
        return {'reason': reason, 'committed': False, 'practice_stage_index': self.stage_index,
                'practice_success': self.state == 'success'}

    def interrupt(self, reason):
        if not self.terminal:
            self.ended = True
            return self._result('missing_physical_evidence' if reason == 'missing_physical_evidence' else 'interrupted',
                state='evaluator_invalid' if reason == 'missing_physical_evidence' else None)
        return self.score()

    def _reset_dwell(self):
        self.stable_start, self.stable_positions = None, {}

    def _waiting(self, reason):
        self._reset_dwell()
        return self._result(reason)

    def _target_ok(self, target, obj):
        name = target['object_id']
        orientation = target['orientation']
        if target['cells'] is not None:
            if not self.helper._poly_fits(name, obj, target['cells']):
                return False
        if target['position_m'] is not None and _distance(obj.position_m, target['position_m']) > self.thresholds.position_tolerance_m:
            return False
        if orientation == 'exact':
            q = target['quaternion_wxyz']
            if q is None or _angle(obj.quaternion_wxyz, q) > self.thresholds.maximum_angle_rad:
                return False
        elif orientation in ('upright', 'face_up', 'face_down'):
            if not self.helper._upright(obj, flipped=orientation == 'face_up'):
                return False
        elif abs(_rotation(obj.quaternion_wxyz)[2][2]) < math.cos(self.thresholds.maximum_angle_rad):
            return False
        region = target['region_bounds_xy_m']
        if region is not None:
            low, high = self.helper._bounds(name, obj)
            if any(low[i] < region[0][i] or high[i] > region[1][i] for i in (0, 1)):
                return False
        return True

    def _protected(self, frame, allowed):
        for name in self.helper.inventory-set(allowed):
            obj = frame.objects[name]
            if (not self.helper._pose_matches(name, obj, self.expected[name])
                    or not self.helper._released_stable(name, frame)):
                return False
        return True

    def observe(self, frame):
        if self.terminal:
            return self._result(self.reason)
        if not self.helper._frame_valid(frame):
            return self._result('missing_physical_evidence', state='evaluator_invalid')
        previous = self.last_frame
        self.last_frame = deepcopy(frame)
        self.helper._last_frame = self.last_frame
        if any(obj.maximum_penetration_m > self.thresholds.maximum_penetration_m for obj in frame.objects.values()):
            return self._result('physical_failure', state='safety_stop')
        if previous is None:
            if any(not self.helper._pose_matches(n, obj, self.expected[n]) for n, obj in frame.objects.items()):
                return self._result('missing_physical_evidence', state='evaluator_invalid')
            return self._result('baseline')
        names = {target['object_id'] for target in self.stage['targets']}
        epsilon = self.thresholds.minimum_contact_impulse_ns
        if any(obj.robot_contact_impulses_ns[self.other] > epsilon for obj in frame.objects.values()):
            return self._result('rule_violation', state='task_failure')
        for name in names:
            obj = frame.objects[name]
            if obj.robot_contact_impulses_ns[self.player]-obj.robot_nonhand_contact_impulses_ns[self.player] > epsilon:
                self.contact_seen.add(name)
                # The first valid hand impulse belongs to the interval from
                # the previous native pose. Only this continuous contact
                # segment may earn net displacement/rotation credit.
                origin = self.hand_origins.setdefault(name, previous.objects[name])
                if (_distance(obj.position_m, origin.position_m) >= self.thresholds.minimum_operation_displacement_m
                        or _angle(obj.quaternion_wxyz, origin.quaternion_wxyz) >= self.thresholds.minimum_operation_rotation_rad):
                    self.motion_seen.add(name)
            else:
                self.hand_origins.pop(name, None)
        allowed = names | set(self.stage['obstacle_ids'])
        if self.stage['operation'] == 'shoot':
            reason = self._observe_shot(frame, previous)
            if self.terminal:
                return self._result(self.reason)
            if reason:
                return self._waiting(reason)
        elif not names <= self.contact_seen & self.motion_seen:
            return self._waiting('own_hand_operation_required')
        if not self._protected(frame, allowed):
            return self._waiting('inventory_recovery_required')
        if self.stage['operation'] != 'shoot' and any(not self._target_ok(target, frame.objects[target['object_id']]) for target in self.stage['targets']):
            return self._waiting('target_required')
        support_group = names if self.stage['operation'] == 'collect' else ()
        if any(not self.helper._released_stable(name, frame, support_group,
                off_deck=self.stage['operation'] == 'shoot') for name in allowed):
            return self._waiting('release_support_required')
        if self.stable_start is None:
            self.stable_start = frame.simulation_time_s
            self.stable_positions = {name: obj.position_m for name, obj in frame.objects.items()}
        if any(_distance(frame.objects[n].position_m, position) > self.thresholds.maximum_drift_m
                for n, position in self.stable_positions.items()):
            return self._waiting('release_support_required')
        if frame.simulation_time_s-self.stable_start+1.e-10 < self.thresholds.dwell_s:
            return self._result('settling')
        self.confirmations.append({'stage_index': self.stage_index, 'native_substep': frame.substep_index,
            'simulation_time_s': frame.simulation_time_s, 'stable_start_s': self.stable_start})
        for name in allowed:
            obj = frame.objects[name]
            self.expected[name] = (obj.position_m, obj.quaternion_wxyz)
        self.stage_index += 1
        self._reset_dwell()
        self.contact_seen.clear()
        self.motion_seen.clear()
        self.hand_origins.clear()
        return self._result('success', state='success') if self.stage_index == len(self.public['goal']['stages']) else self._result('stage_confirmed')

    def _observe_shot(self, frame, previous):
        stage, shot = self.stage, self.shot
        name = stage['targets'][0]['object_id']
        obj, t = frame.objects[name], self.thresholds
        epsilon = t.minimum_contact_impulse_ns
        if (obj.robot_nonhand_contact_impulses_ns[self.player] > epsilon or
                any(sum(p.robot_contact_impulses_ns.values()) > epsilon for key, p in frame.objects.items() if key != name)):
            self._result('rule_violation', state='task_failure')
            return None
        board = self.helper.design['board']
        color = 'black' if self.player == 'player_0' else 'white'
        direction = 1 if color == 'black' else -1
        line, region = board['launch_lines_y_m'][color], board['launch_region_bounds_xy_m'][color]
        contact = sum(obj.robot_contact_impulses_ns.values()) > epsilon
        if not shot['started']:
            low, high = self.helper._bounds(name, obj)
            if not all(low[i] >= region[0][i] and high[i] <= region[1][i] for i in (0, 1)):
                self._result('rule_violation', state='task_failure')
                return None
            if (contact and self._target_ok(stage['targets'][0], obj) and self.helper._supported(name, frame)
                    and direction*obj.linear_velocity_m_s[1] >= t.disk_propulsion_speed_m_s):
                shot['started'] = True
            else:
                return 'preparing_disk'
        if contact:
            if shot['released'] or direction*(obj.position_m[1]-line) > 1.e-12:
                self._result('rule_violation', state='task_failure')
                return None
        elif not shot['released']:
            if direction*(previous.objects[name].position_m[1]-line) > 1.e-12:
                self._result('rule_violation', state='task_failure')
                return None
            shot['released'] = True
        if shot['released'] and direction*(obj.position_m[1]-line) >= 0:
            shot['crossed'] = True
        paths = self.task['asset_metadata']['actors']
        for row in self.contact_rows:
            if row.get('time_s') != frame.simulation_time_s:
                self._result('missing_physical_evidence', state='evaluator_invalid')
                return None
            pair = {row.get('actor0'), row.get('actor1')}
            if (paths[name] in pair and any(paths[other] in pair for other in stage['obstacle_ids'])
                    and row.get('impulse_magnitude_sum_ns', 0) > epsilon):
                shot['obstacle_contact'] = True
        if not shot['released'] or (stage['require_line_crossing'] and not shot['crossed']):
            return 'free_crossing_required'
        if stage['require_obstacle_contact'] and not shot['obstacle_contact']:
            return 'obstacle_contact_required'
        if self.helper._supported(name, frame):
            if not self._target_ok(stage['targets'][0], obj):
                return 'target_required'
            radius = math.dist(obj.position_m[:2], board['target_center_xy_m'])
            points = next((p for r, p in zip(board['scoring_radii_m'], board['scoring_points']) if radius <= r), 0)
        else:
            points = 0
        if points < stage['minimum_points']:
            return 'score_required'
        return None


def evaluator_factory(task):
    """Factory used only by the registered complete-native runtime adapter."""
    return PracticeReferee(task)
