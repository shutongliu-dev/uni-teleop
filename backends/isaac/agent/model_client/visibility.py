"""Client-side observation ablations shared by text, tools and programs.

The source server still records all installed sensors. These are explicit
client-visible conditions, pinned in the Agent configuration, not claims that
the simulator disabled a sensor or that an untrusted external client cannot
access its separately authorized public HTTP endpoints.
"""
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import Field

from .actions import public_observation
from .bench_http import BenchHTTPError
from bench.protocol.schemas import (ActionDiagnostic, Contract, Name, Nonnegative,
                                 RobotExecutionState, TerminationReason)
from bench.sensors.contact_schema import ContactHistoryObservation


def observation_view(observation, *, visual_input='rgbd', tactile_input='on'):
    if visual_input not in ('rgb', 'rgbd') or tactile_input not in ('on', 'off'):
        raise ValueError('Unknown observation condition')
    public_observation(observation)
    view = deepcopy(observation)
    if tactile_input == 'off':
        view.pop('tactile', None)
        view.pop('hand_tactile', None)
    if visual_input == 'rgb':
        for camera in view.get('cameras', []):
            for key in list(camera):
                if key.startswith('depth_'):
                    camera.pop(key)
    return view


def capability_view(capabilities, *, visual_input='rgbd', tactile_input='on'):
    view = deepcopy(capabilities)
    # Availability here describes the configured Agent/tool view. The original
    # capabilities are separately archived and used for action validation.
    if visual_input == 'rgb':
        view['depth_available'] = False
    if tactile_input == 'off':
        view.update(tactile_available=False, tactile_mode='unavailable', tactile_cloth_available=False)
    system0 = view.get('system0')
    if isinstance(system0, dict) and isinstance(system0.get('geometry_inputs'), list):
        # The model action grammar has no hand_posture field; do not advertise
        # a controller input the model cannot request.
        system0['geometry_inputs'] = [item for item in system0['geometry_inputs'] if item != 'hand_posture']
    return view


class _PublicPrimitiveFeedback(Contract):
    """The public service.status/cancel response, not a lifecycle ActionEvent.

    Optional fields keep defaults for parsing older responses; exclude_unset
    prevents inventing those fields in the model-visible projection. In
    particular, lifecycle detail/wall_time_s and arbitrary service errors are
    not fields on this public endpoint.
    """
    action_id: Name
    status: Literal['accepted', 'running', 'completed', 'cancelled', 'failed']
    sim_time_s: Nonnegative
    episode_finished: Annotated[bool, Field(strict=True)] = False
    elapsed_sim_s: Nonnegative = 0.
    cancellation_requested: Annotated[bool, Field(strict=True)] = False
    diagnostic: ActionDiagnostic | None = None
    termination_reason: TerminationReason | None = None
    robot_execution_state: RobotExecutionState | None = None
    contact_history: ContactHistoryObservation | None = None


def feedback_view(feedback, *, tactile_input='on'):
    """Validate public fields before exposing feedback to any model/program."""
    if tactile_input not in ('on', 'off'):
        raise ValueError('Unknown tactile condition')
    if feedback is None:
        return None
    try:
        if not isinstance(feedback, dict):
            raise ValueError('Expected an object')
        checked = _PublicPrimitiveFeedback.model_validate(feedback, strict=True)
        history = checked.contact_history
        if history is not None and history.action_id != checked.action_id:
            raise ValueError('Contact history action mismatch')
        view = checked.model_dump(mode='json', exclude_unset=True)
    except (ValueError, TypeError):
        # Pydantic errors include offending values and paths. Never expose their
        # prose, even for rejected fields or tactile-off responses.
        raise BenchHTTPError('Invalid public primitive feedback', category='invalid_response') from None
    if tactile_input == 'off':
        view.pop('contact_history', None)
    return view
