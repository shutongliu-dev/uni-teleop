"""Sparse wrist/fingertip decisions translated using public robot FK only."""
from dataclasses import dataclass
import math

import numpy as np

from bench.protocol.schemas import EpisodeSpec, RobotCapabilities, System0Action
from bench.protocol.validation import negotiate, validate_action
from bench.sdk.geometry import apply_world_pose_delta
from .actions import ActionValidationError, _number, parse_json_object, public_observation


@dataclass(frozen=True)
class GeometryActionLimits:
    min_duration_s: float = .4
    max_duration_s: float = 5.
    max_wrist_translation_m: float = .05
    max_wrist_rotation_rad: float = .15
    max_fingertip_translation_m: float = .01

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if _number(getattr(self, name), name) <= 0:
                raise ActionValidationError(name+' must be positive')
        if self.min_duration_s > self.max_duration_s:
            raise ActionValidationError('duration limits must be ordered')


def _vec(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ActionValidationError(label+' must contain three numbers')
    return [_number(x, label) for x in value]


def parse_geometry_action(value):
    value = parse_json_object(value)
    kind = value.get('action')
    if kind not in ('move_geometry', 'hold', 'stop'):
        raise ActionValidationError('action must be move_geometry, hold or stop')
    allowed = {'action', 'reason'}
    if kind != 'stop':
        allowed.add('duration_s')
    if kind == 'move_geometry':
        allowed |= {'wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m',
                    'position_tolerance_m', 'orientation_tolerance_rad', 'hands'}
    if set(value) - allowed:
        raise ActionValidationError('geometry decision contains unadvertised fields')
    reason = value.get('reason', '')
    if not isinstance(reason, str) or len(reason) > 512:
        raise ActionValidationError('reason must be a short public explanation')
    result = {'action': kind, 'reason': reason}
    if kind == 'stop':
        return result
    result['duration_s'] = _number(value.get('duration_s'), 'duration_s')
    if result['duration_s'] <= 0:
        raise ActionValidationError('duration_s must be positive')
    if kind == 'hold':
        return result
    for name in ('position_tolerance_m', 'orientation_tolerance_rad'):
        if name in value:
            tolerance = _number(value[name], name)
            if tolerance <= 0:
                raise ActionValidationError(name+' must be positive')
            result[name] = tolerance
    if 'hands' in value:
        # Bimanual robots: one sparse objective per moved hand; omitted hands hold.
        hands = value['hands']
        if (any(k in value for k in ('wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m'))
                or not isinstance(hands, dict) or not hands or not set(hands) <= {'left', 'right'}):
            raise ActionValidationError('hands maps left/right to per-hand targets and replaces single-hand fields')
        result['hands'] = {side: _hand_targets(objective, side) for side, objective in hands.items()}
        return result
    result.update(_hand_targets(value, None))
    return result


def _hand_targets(value, side):
    label = '' if side is None else side + ' hand '
    if side is not None and (not isinstance(value, dict)
                             or set(value) - {'wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m'}):
        raise ActionValidationError(label + 'objective contains unadvertised fields')
    result = {}
    for name in ('wrist_delta_m', 'wrist_rotation_delta_rad'):
        if name in value:
            result[name] = _vec(value[name], name)
    if 'fingertip_delta_m' in value:
        tips = value['fingertip_delta_m']
        if not isinstance(tips, dict) or not tips or not set(tips) <= {'thumb','index','middle','ring','little'}:
            raise ActionValidationError('fingertip_delta_m requires one to five named fingers')
        result['fingertip_delta_m'] = {k: _vec(v, k) for k,v in tips.items()}
    if not any(k in result for k in ('wrist_delta_m', 'wrist_rotation_delta_rad', 'fingertip_delta_m')):
        raise ActionValidationError(label + 'move_geometry requires at least one explicit robot geometry target')
    return result


def build_geometry_action(model_action, observation, capabilities, episode_spec, *, action_id,
                          expected_sequence, actions_used=0, limits=None):
    decision = parse_geometry_action(model_action)
    obs = public_observation(observation)
    limits = limits or GeometryActionLimits()
    if type(expected_sequence) is not int or expected_sequence != obs.sequence:
        raise ActionValidationError('model decision uses a stale observation')
    if type(actions_used) is not int or actions_used < 0:
        raise ActionValidationError('actions_used must be a nonnegative integer')
    try:
        caps, spec = RobotCapabilities.model_validate(capabilities), EpisodeSpec.model_validate(episode_spec)
        negotiate(spec, caps)
    except (TypeError, ValueError):
        raise ActionValidationError('invalid advertised controller or episode contract') from None
    if (spec.episode_id != obs.episode_id or spec.track not in ('system0', 'free_tool_agent')
            or spec.observation_profile != 'vision_proprio'
            or spec.system0.policy_id not in ('classical_fingertip', 'classical_fingertip_bimanual')):
        raise ActionValidationError('episode must pin the public classical fingertip controller')
    if obs.last_action_status in ('accepted', 'running'):
        raise ActionValidationError('previous action is still running')
    if set(obs.proprioception.joint_positions_rad) != set(caps.joint_limits):
        raise ActionValidationError('observation must contain all and only advertised joints')
    if decision['action'] == 'stop':
        return None
    duration = decision['duration_s']
    minimum = caps.control_period_s if decision['action'] == 'hold' else limits.min_duration_s
    if not minimum <= duration <= limits.max_duration_s:
        raise ActionValidationError(f'geometry duration must be between {minimum:g} and {limits.max_duration_s:g} seconds')
    if (actions_used >= spec.budgets.max_actions
            or (spec.budgets.sim_time_s is not None
                and obs.sim_time_s + duration > spec.budgets.sim_time_s + 1e-9)):
        raise ActionValidationError('geometry action exceeds the episode budget')
    if decision['action'] == 'hold':
        try:
            action = System0Action(action_id=action_id, observation_sequence=obs.sequence,
                skill='hold', hold_duration_s=duration, timeout_sim_s=duration)
            validate_action(action, caps, spec, observation_sequence=obs.sequence)
        except (TypeError, ValueError):
            raise ActionValidationError('hold violates advertised duration, timing or controller skills') from None
        return action.model_dump(mode='json')
    tolerances = {name: decision[name] for name in ('position_tolerance_m', 'orientation_tolerance_rad') if name in decision}
    if 'hands' in decision:
        published = obs.proprioception.hand_geometries
        if published is None or not set(decision['hands']) <= set(published):
            raise ActionValidationError('public per-hand robot geometry is missing')
        objectives = {side: {**_objective(targets, published[side], caps, limits, f'{side}_hand_flange'), **tolerances}
                      for side, targets in decision['hands'].items()}
        fields = {'hand_geometries': objectives}
    else:
        if obs.proprioception.geometry is None:
            raise ActionValidationError('public robot geometry is missing')
        fields = {'geometry': {**_objective(decision, obs.proprioception.geometry, caps, limits, 'right_hand_flange'),
                               **tolerances}}
    try:
        action = System0Action(action_id=action_id, observation_sequence=obs.sequence,
            skill='move_geometry', timeout_sim_s=duration, object_effect='none', **fields)
        validate_action(action, caps, spec, observation_sequence=obs.sequence)
    except (TypeError, ValueError):
        raise ActionValidationError('geometry action violates advertised limits or timing') from None
    return action.model_dump(mode='json')


def _objective(decision, current, caps, limits, frame_id):
    """Absolute wrist/fingertip targets from sparse deltas and the published FK."""
    if current.wrist.frame_id != 'world' or current.fingertip_frame_id != frame_id:
        raise ActionValidationError('public robot geometry has an unsupported coordinate frame')
    translation = np.array(decision.get('wrist_delta_m', [0.,0.,0.]))
    rotation = np.array(decision.get('wrist_rotation_delta_rad', [0.,0.,0.]))
    if math.hypot(*translation) > limits.max_wrist_translation_m + 1e-12:
        raise ActionValidationError('wrist translation exceeds the per-decision limit')
    if math.hypot(*rotation) > limits.max_wrist_rotation_rad + 1e-12:
        raise ActionValidationError('wrist rotation exceeds the per-decision limit')
    wrist = apply_world_pose_delta(current.wrist, translation, rotation)
    tips = {}
    for finger, delta in decision.get('fingertip_delta_m', {}).items():
        if finger not in current.fingertip_positions_m or finger not in caps.finger_names:
            raise ActionValidationError('requested finger is not available')
        if math.hypot(*delta) > limits.max_fingertip_translation_m + 1e-12:
            raise ActionValidationError('fingertip translation exceeds the per-decision limit')
        tips[finger] = (np.array(current.fingertip_positions_m[finger]) + delta).tolist()
    return {'wrist': wrist, 'fingertip_frame_id': current.fingertip_frame_id, 'fingertip_positions_m': tips}
