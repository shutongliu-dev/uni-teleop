"""Read-only planning using isolated copies of the unchanged robot executor.

Inputs are public robot observations, acknowledged robot references and the
executor's robot-only lifecycle. No worker, task scene or object data enters.
"""
from __future__ import annotations

from copy import deepcopy
import math

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import ClassicalFingertipPolicy, GeometryIKError
from bench.protocol.motion import MotionQuery, MotionQueryResult
from bench.protocol.schemas import System0Action
from .motion_collision import check_nominal_arm_path


def supports_motion_query(policy):
    return isinstance(policy, (ClassicalFingertipPolicy, BimanualFingertipPolicy))


def query_motion(policy, observation, query, *, max_action_sim_s, player_hand=None,
                 acknowledged_targets=None, turn_index=None):
    """Run begin only on fresh independent copies, never step or commit them.

    A one-tick request elicits the executor's own exact minimum. Composite
    policies can reject a later side, so repeat only at each larger reported
    minimum until every side accepts. This does not change the action budget or
    extend any real action. The successful copy supplies the actual planned
    nominal initial/goal; measured joints alone cannot reproduce recovery and
    retained reference semantics.
    """
    if not supports_motion_query(policy):
        raise ValueError('Motion preview requires the original classical executor')
    query = MotionQuery.model_validate(query)
    observation = observation.model_copy(deep=True)
    baseline = deepcopy(policy)
    if hasattr(baseline, 'set_player_hand'):
        if player_hand is None or acknowledged_targets is None:
            raise ValueError('Player preview requires authenticated robot ownership and ACK')
        baseline.set_player_hand(player_hand, dict(acknowledged_targets))
    period = float(baseline.descriptor.control_period_s)
    value = {'observation_sequence': query.observation_sequence, 'sim_time_s': observation.sim_time_s,
             'turn_index': turn_index, 'max_action_sim_s': max_action_sim_s,
             'requested_duration_s': query.requested_duration_s, 'planning_succeeded': False}
    seconds = period
    selected = set(query.hand_geometries or {})
    side_count = len(getattr(baseline, 'sides', {})) or 1
    for _ in range(side_count + 2):
        planned = deepcopy(baseline)
        action = System0Action(action_id='read_only_motion_query', observation_sequence=query.observation_sequence,
            skill='move_geometry', timeout_sim_s=seconds, geometry=query.geometry,
            hand_geometries=query.hand_geometries)
        try:
            planned.begin(action, observation)
        except GeometryIKError as error:
            diagnostic = error.public_diagnostic
            if diagnostic.code != 'insufficient_duration' or diagnostic.duration_preflight is None:
                return MotionQueryResult.model_validate({**value, 'diagnostic': diagnostic})
            minimum = diagnostic.duration_preflight.minimum_duration_s
            if minimum <= seconds:
                raise RuntimeError('Controller duration preflight did not advance')
            seconds = minimum
        else:
            break
    else:
        raise RuntimeError('Composite controller duration preflight did not converge')
    models = getattr(planned, 'sides', None)
    if models is None:
        models = {planned.geometry.embodiment.side: planned}
        selected = set(models)
    hands = {}
    for side in selected:
        own = models[side]
        initial = dict(zip(own.geometry.names, own.initial.tolist()))
        goal = dict(zip(own.geometry.names, own.goal.tolist()))
        trajectory_steps = round(own.duration / period)
        settle_steps = math.ceil((own.duration + own.limits.settle_margin_s) / period - 1e-9) - trajectory_steps
        minimum = (own._recovery_steps + trajectory_steps + settle_steps) * period
        hands[side] = {'initial_joint_positions_rad': initial, 'goal_joint_positions_rad': goal,
                      'trajectory_duration_s': own.duration, 'braking_steps': own._recovery_steps,
                      'settling_steps': settle_steps, 'minimum_duration_s': minimum,
                      'collision': check_nominal_arm_path(own.geometry, initial, goal)}
    requested = query.requested_duration_s
    aligned = (requested is not None and math.isfinite(requested / period)
               and abs(round(requested / period) * period - requested) <= 1e-8)
    return MotionQueryResult.model_validate({**value, 'planning_succeeded': True,
        'minimum_duration_s': seconds, 'fits_action_budget': seconds <= max_action_sim_s,
        'requested_duration_fits': None if requested is None else bool(aligned and seconds <= requested <= max_action_sim_s),
        'hands': hands})
