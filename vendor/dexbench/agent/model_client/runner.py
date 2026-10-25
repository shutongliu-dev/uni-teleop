"""Online vision/proprioception Agent; every decision calls the configured model API."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
from collections import Counter
from urllib.parse import urlsplit

from .actions import ActionLimits, ActionValidationError, parse_json_object
from .bench_http import BenchHTTPClient, BenchHTTPError
from .geometry_actions import GeometryActionLimits
from .transport import ModelClient, ModelGatewayConfig, ModelGatewayError
from .responses import MicuModelClient, MICU_RETRY_POLICY
from .capture import capture
from .interaction import InteractionSession, PRIMITIVE_ATTEMPT_ACCOUNTING
from .learning.robot_api import DEFAULT_MAX_STATUS_QUERIES
from .free_interface import (FreeActionLimits, FREE_SYSTEM_PROMPT, execute_free_decision,
                             free_interface_contract, require_free_interface)
from .program import ProgramLimits, execute_program, program_feedback
from .visibility import capability_view
from .time_limits import minimum_limit, optional_positive, optional_seconds
from .outcomes import classify_report, error_record
from bench.protocol.schemas import EpisodeResult, EpisodeSpec, PolicyDescriptor
from bench.protocol.allocations import (MAX_EPISODE_SIM_TIME_S, MAX_EPISODE_WALL_TIME_S,
                                        MAX_EPISODE_ACTIONS)
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


AGENT_VERSION = 'v11'
CLIENT_SOURCE_DEPENDENCIES = ('bench/protocol/schemas.py', 'bench/protocol/validation.py',
    'bench/geometry.py', 'bench/sdk/__init__.py', 'bench/sdk/depth.py', 'bench/sdk/geometry.py',
    'scripts/maintenance/freeze_project_sources.py',
    'bench/protocol/allocations.py', 'bench/protocol/task_scoring.py')


JOINT_SYSTEM_PROMPT = """You control a simulated Franka FR3 robot with a dexterous hand.
Use only the supplied public task, current camera images, camera calibration,
joint positions/velocities and action feedback. You have no hidden object pose,
contact oracle, task score, trained grasp policy or automatic task-solving tool.
Choose ONE short next action, inspect its visual result on the next turn, and
adapt. Do not output an entire task program or claim that text completes a task.
Return exactly one JSON object, optionally enclosed in one JSON code fence:
{"action":"joint_delta","delta_rad":{"advertised_joint_name":0.01},"duration_s":0.4,"reason":"brief observation and intended effect"}
or {"action":"hold","duration_s":0.4,"reason":"brief explanation"}
or {"action":"stop","reason":"why further action is not appropriate"}.
Joints are position-servoed. Joint deltas are radians relative to the CURRENT
COMMANDED targets (proprioception.commanded_joint_positions_rad, the acknowledged drive
targets); omitted joints keep their commanded targets. Measured positions can
differ from commanded targets under load or contact: a finger commanded into an
object keeps pressing with a force that grows with that gap. A delta that pushes
a target farther than the advertised target-gap limit from its measured joint is
rejected. If commanded targets are absent, deltas use measured positions.
Requested targets must stay inside the joint position limits.
Duration must be an integer multiple of the advertised 0.02 s control period.
hold(duration_s) advances physics for that requested duration (0.02 to 5 seconds),
retaining the commanded targets; it is not an early-completing move.
Respect the per-decision delta, position and duration limits in the user input. These bounds do not certify
collision avoidance. There is no implicit IK or task-specific script. Give a
short publicly observable reason, not hidden chain-of-thought. A stop request
cancels an unfinished episode; only the server evaluator can report success.
"""

SYSTEM_PROMPT = """You control a simulated Franka FR3 with a right SHARPA W01 hand.
Use only the public task, current RGB-D images and calibration, robot joint
state, robot-only forward-kinematics geometry, action feedback, and any public
tactile proxy explicitly included in the observation. You have no hidden
object pose, evaluator score, trained RL policy or automatic task-solving tool.
Choose ONE next geometric change within the advertised limits, inspect fresh observations after it,
and adapt. Return exactly one JSON object, optionally in one JSON code fence:
{"action":"move_geometry","wrist_delta_m":[0,0,0.01],"fingertip_delta_m":{"thumb":[0.002,0,0],"index":[-0.002,0,0]},"duration_s":3.0,"reason":"brief observation and intended effect"}
or {"action":"stop","reason":"why further action is not appropriate"}.
or {"action":"hold","duration_s":0.3,"reason":"maintain the current robot reference while physics advances"}.
move_geometry requires at least one of wrist_delta_m, wrist_rotation_delta_rad,
or fingertip_delta_m. All deltas are relative to the CURRENT public measured
robot geometry. Wrist translation is in WORLD axes, in meters. Wrist rotation
is an axis-angle vector in WORLD axes, in radians, left-multiplied onto the
current wrist rotation; omitted wrist translation/rotation holds that component.
The wrist frame origin is right_hand_flange, identically mounted to fr3_link8;
it is NOT the palm center. Fingertip deltas are in right_hand_flange LOCAL axes,
in meters. Choose one, two, three or more needed fingers from thumb, index,
middle, ring, little. Omitted-finger goals follow the advertised controller contract
and the hand-reference rules below; the delta origin is still measured geometry.
The tip coordinates describe URDF virtual frame origins, not guaranteed skin
contact points. A fixed classical IK and bounded joint-target executor converts
your requested geometry; it does not choose grasp targets, avoid collisions,
control contact forces, or infer grasp success. No task-specific script takes over.
duration_s is the maximum execution/settling time (0.4 to 5 seconds, a multiple
of 0.02); successful geometry may finish earlier. Allow sufficient time for a
smooth move. Unreachable or too-short requests fail and can be revised on the
next turn. Respect the advertised translation and rotation norm limits.
Optional position_tolerance_m and orientation_tolerance_rad must be finite and
positive. They request stricter wrist completion for small motions; the controller
uses the minimum of the request and its advertised cap. Omission preserves the
default tolerances. These do not change fingertip tolerances or certify contact.
For timed holding, use hold with duration_s from 0.02 to 5 seconds (20 ms ticks).
It retains the controller reference, advances real physics for the requested
duration and consumes one action. move_geometry with zero delta is not a hold.
The episode can terminate earlier for task completion, safety, budget or cancellation.
RGB-D depth is optical camera z in meters. Preview images use a FIXED 0.1-4 m
gray scale: invalid black, near dark, far light. Metric pixel samples are also
provided; use calibration to reason about world coordinates. A tactile reading
marked unavailable/not_sampled/invalid is NOT zero contact; this simulation's
rigid-fingertip proxy does not measure cloth contact or full tactile taxels.
Give a brief observable reason, not hidden chain-of-thought. Outputting stop
cancels an unfinished episode; only the independent server evaluator reports
task success. Geometry completion alone does not imply task or grasp success.
"""

TOOL_RULES = """You control a simulated FR3 with a right SHARPA W01 using only public
RGB/RGB-D, calibration, measured robot state/FK, action feedback and the configured
tactile view. No object-pose oracle, grasp planner, learned RL or task script exists.
Tools: observe() returns {observation,feedback,depth_observation};
hold(duration_s=...,reason?) retains the robot reference and advances physics
for 0.02-5 seconds in 20 ms ticks, returning the same feedback shape as a move.
It counts against the primitive and per-program move budgets. observe() does
not advance physics. Zero-delta move_geometry may finish early and is not hold.
point_at(camera_id,u,v) READS the WORLD 3D point at a zero-based integer RGB-D pixel;
it never moves the robot. Invalid/missing depth is an error. It is not object recognition.
move_geometry(wrist_delta_m?,wrist_rotation_delta_rad?,fingertip_delta_m?,duration_s,reason?,position_tolerance_m?,orientation_tolerance_rad?)
blocks until the short action finishes, returning {feedback,observation,depth_observation}
with fresh data. Wrist translation is WORLD metres, rotation a WORLD axis-angle
vector in radians. Wrist origin is right_hand_flange (fr3_link8), NOT palm center.
Fingertip deltas are LOCAL right_hand_flange metres for any subset of thumb/index/
middle/ring/little; omitted fingers follow the advertised controller contract and
the hand-reference rules below. Deltas use CURRENT measured
geometry at each invocation. At least one geometry delta is required. Limits: wrist
translation norm <=0.05m, rotation <=0.15rad, each tip <=0.01m; duration_s is a maximum
execution/settling time of 0.4-5 seconds, a multiple of 0.02. Allow enough settling time.
The fixed classical controller handles IK and bounded targets, NOT collision planning,
force control, grasp selection or task success. URDF tips are virtual frame origins,
not calibrated contact patches. A failed action needs reconsideration using feedback.
Tactile is an uncalibrated rigid-pad normal-load proxy, not cloth sensing. Omitted or
invalid tactile is not zero contact. Tool results never expose the hidden evaluator.
stop(reason='...') stops this Agent and cancels an unfinished episode. Action completed
does not mean task success. Give brief observable reasons, not hidden chain-of-thought.
"""
TOOLS_PROMPT = TOOL_RULES + """Return exactly ONE JSON tool call per model turn:
{"tool":"move_geometry","arguments":{"wrist_delta_m":[0,0,0.01],"duration_s":3.0}}
or {"tool":"point_at","arguments":{"camera_id":"task","u":320,"v":240}}
or {"tool":"observe","arguments":{}} or {"tool":"stop","arguments":{"reason":"..."}}.
or {"tool":"hold","arguments":{"duration_s":0.3}}.
All read calls consume a model turn too. Inspect fresh images and the previous tool
result on the next turn. Never return action JSON or a task-solving script in this mode.
"""
CODE_PROMPT = TOOL_RULES + """Return only a short program in the advertised restricted
Python-shaped language, optionally in one python code fence. Available calls are only
observe, point_at, move_geometry, hold, stop, abs, min, max, len. You may assign JSON values,
index dictionaries/lists, use finite arithmetic, if/else, and for i in range(LITERAL)
with break/continue. No imports, attributes, functions, while, comprehensions, files,
network, exception handlers or Python runtime access. This is an AST interpreter, not
general Python. Tools must be assignment RHS or standalone statements. Programs are
bounded by the supplied limits; every move consumes the shared episode action budget.
Each move/observe returns fresh public state; recheck feedback between moves. Example:
state = observe()
for i in range(2):
    state = move_geometry(wrist_delta_m=[0,0,0.005], duration_s=3.0)
    if state['feedback']['status'] != 'completed':
        break
Each program returns primitive_results for every move/hold, including failures
followed by successful calls. call_status is tool execution; feedback.status is
robot execution. A completed tool/program does not imply a completed movement.
Recoverable public validation failures include error.public_error with a safe reason.
End the program normally to return to the model for fresh images. Only call stop when
the entire episode should end. Choose your own motion from the task and actual images.
"""


TACTILE_RULES = """Tactile coverage: use tactile only when supplied in this observation.
If previous_action_feedback includes contact_history, it summarizes the entire
acknowledged action, including non-pad robot regions and simultaneous pad timing.
Use it to notice shell contact or interrupted contact missed by the last window.
Pad coactivity does not prove opposing normals, the same object, or grasp success.
Unknown/invalid history is not zero; directions, cloth forces and taxels remain unavailable.
It is an uncalibrated normal-impulse proxy on the five declared fingertip elastomer
collision bodies. A valid zero reports no solved normal impulse on that pad in
the reported window; it does not establish that the whole hand is collision-free.
Proximal links, palm or arm may collide while pad readings are zero. Positive values
may include pad self-contact and do not identify a target object. Missing, unavailable,
not_sampled or invalid is not zero. This API has no full-hand tactile, cloth sensing,
hardware taxels, shear or torque measurement.
"""

HAND_REFERENCE_RULES = """Hand-reference rules for this advertised controller:
During a normal move_geometry transition, an omitted finger chain retains its previous
policy.begin-accepted joint goal. Accepted means the executor installed the goal after
validation; it does not mean task or grasp success. On the first action, omitted fingers
use the measured reference. A finger is selected by an explicit fingertip target or any
explicit joint posture in its chain. Selected-finger IK verifies the tip position but may
change its redundant joint posture; unspecified joints in a selected partial-posture
chain keep the measured-reference semantics.
After an executed action fails or is cancelled, omitted fingers use the continuous stop
predicted from the last COMMITTED reference and velocity, not the failed goal or an older
grasp. Rejected IK/posture/duration requests install no new goal. Cancellation pauses
execution; recovery advances physics in the next accepted action. Normal hold retains
the accepted goal and its bounded load compensation; recovery hold uses the stopping
reference. Neither controls grasp force or certifies contact. Omitted wrist components
still use current measured wrist geometry. All-joint tracking tolerance still applies:
loaded retained fingers may cause wrist-only moves to time out despite wrist arrival.
Inspect structured feedback before continuing.
"""


HAND_REFERENCE_RULES_V2 = 'Rules for the advertised accepted_hand_goal_retention_v2 controller:\nA move completes when its requested wrist and selected fingertip objectives are reached, posture-only joint objectives are reached, and the acknowledged command reference has stopped. If wrist is omitted, its current measured pose remains the wrist objective. A posture on the same finger as an explicit tip is an IK reference, not a second exact joint objective; IK can change redundant joints. Unselected loaded fingers are maintained, not required to match an unloaded pose. Maintenance residuals are not evidence that an explicit wrist or tip failed.\nNormally, omitted hand chains retain their accepted tracking goal, or a persistent recovery fixed-drive mode. After failure or cancellation, recovery brakes continuously from the last executed reference and velocity; it never uses an unexecuted pending command; the last executed drive targets are held, so a finger already pressing keeps pressing. Recovery hold fixes all controlled joint drive targets without adding fresh P/I correction. Any later geometry move resumes arm tracking to the explicit or measured wrist objective. Omitted hand chains stay fixed until a verified explicit new goal reselects that chain. Rejected requests change no mode or anchor. Fixed drive keeps those targets but does not regulate force; grip changes if the hand or object moves.\nMeasured joints up to the advertised {epsilon_rad} rad past a limit (the episode safety band) are clamped only as an internal IK seed; an action fails if contact keeps pushing a joint further past its limit. Raw measured joints are unchanged; requested postures and emitted commands still obey strict joint limits. This band is not a hardware safety limit.\nRead termination_reason: controller_converged verifies the primitive objective; time_hold_completed verifies the requested maintenance time; task_terminated means the episode succeeded while this primitive was still active and does not certify its objective or requested hold duration. Inspect elapsed_sim_s and the separate public episode result. Missing historical termination reasons are unknown.'

ROBOT_FACTS = """Robot and scene facts: the world frame is right-handed with +z up (gravity -z).
Each camera's world_from_camera maps its optical frame (x right, y down, z forward
along the view) to world. Joints are position-servoed drives. When proprioception
includes commanded_joint_positions_rad, measured joints can differ from those
commanded targets under load or contact. When the geometry
includes fingertip_quaternions_wxyz, each tip frame has +z pointing distally along
the distal phalanx and +x toward its palmar elastomer pad; the tip origin lies inside
that pad, about 2 mm (thumb 1.7 mm) beneath its palmar surface and about 1.6 mm
(thumb 4.8 mm) proximal of the distal shell tip. finger_joint_centers_m gives the
URDF joint origins along each finger, so curled middle phalanges may touch first.
Contact on a finger shell or palm is allowed and is reported separately from pad
contact. After a failed or stalled move the last executed drive targets are held;
a fixed drive does not regulate force. A move that stops progressing before its
objective (normally blocked by contact) ends early as failed with tracking_stalled;
termination_reason contact_stalled means its unreached fingers or wrist were in
contact. The public goal may also state the table height, object size and nominal mass.
"""

TERMINATION_RULES = "Inspect termination_reason: controller_converged verifies the requested geometry; time_hold_completed verifies elapsed robot maintenance time; direct_duration_completed verifies the commanded interval; task_terminated means episode success interrupted the primitive and does not certify its unfinished objective. A missing historical reason is unknown."

BIMANUAL_RULES = """Bimanual robot: two FR3 arms, each with its own SHARPA W01 hand (left and right).
proprioception.hand_geometries.left/right give each hand's wrist and fingertips; each hand's
fingertips are in its own frame (left_hand_flange / right_hand_flange). hand_tactile.left/right
give each hand's five-channel pad proxy (it replaces the single-hand tactile field). Move one or both hands
in one decision with
{"action":"move_geometry","hands":{"left":{"wrist_delta_m":[0,0,0.01]},"right":{"fingertip_delta_m":{"thumb":[0.002,0,0]}}},"duration_s":3.0,"reason":"..."}
Each hand accepts wrist_delta_m, wrist_rotation_delta_rad and fingertip_delta_m with the same
per-decision limits; top-level wrist/fingertip fields are not used. A hand omitted from "hands"
holds its reference until the moved hands finish; if either hand fails, the whole action fails
and the other hand stops. hold applies to both hands.
"""


MOUNT_V2_SENTENCES = (
    ('The wrist frame origin is right_hand_flange, identically mounted to fr3_link8;',
     'The wrist frame is right_hand_flange, the hand frame: fr3_link8 turned 180 deg about the flange axis;'),
    ('Wrist origin is right_hand_flange (fr3_link8), NOT palm center.',
     'Wrist origin is right_hand_flange (fr3_link8 turned 180 deg about its axis), NOT palm center.'),
)


def mount_prompt(prompt, mount):
    """State the published hand mount; a v1 (or unpublished) mount leaves the prompt unchanged."""
    if not mount or mount.get('mount_convention') in (None, 'flange_axis_centred_v1'):
        return prompt
    if mount.get('mount_convention') != 'flange_axis_centred_v2':
        raise ValueError('Unknown published hand mount convention')
    for old, new in MOUNT_V2_SENTENCES[:2]:
        prompt = prompt.replace(old, new)
    prompt = prompt.replace('names that FR3 tool-flange frame (fr3_link8),',
                            'names the hand frame (the FR3 tool flange fr3_link8 turned 180 deg about its axis),')
    return prompt + ('\nHand mount: flange_axis_centred_v2 (uncalibrated simulation assumption). In right_hand_flange '
                     'the fingers point along +z, the palm faces +x and the thumb side is +y; a natural top-down '
                     'grasp in front of the robot (palm toward the robot base, thumb to its left) is mid-range for '
                     'the arm\'s last joint.\n')


def embodiment_prompt(prompt, robot_id):
    """Robot-specific wording; the default SHARPA prompt is returned unchanged."""
    if robot_id in (None, 'sharpa_w01_right'):
        return prompt
    from bench.simulation.isaacsim_backend.robot_registry import HANDS, get_hand_spec
    if robot_id not in HANDS:
        return prompt          # unregistered (test or external) robots keep the generic wording
    spec = get_hand_spec(robot_id)
    if spec.get('arms'):
        prompt = prompt.replace('a simulated Franka FR3 with a right SHARPA W01 hand.',
                                'a simulated pair of Franka FR3 arms, each with a SHARPA W01 hand (left and right).')
        return prompt + '\n' + BIMANUAL_RULES
    fingers = ', '.join(spec['fingertip_links'])
    prompt = (prompt.replace('a simulated Franka FR3 with a right SHARPA W01 hand.', f'a simulated Franka FR3 with a {spec["label"]} hand.')
              .replace('a simulated FR3 with a right SHARPA W01 using', f'a simulated FR3 with a {spec["label"]} hand using')
              .replace('one FR3 arm and SHARPA hand', f'one FR3 arm and {spec["label"]} hand'))
    mount = ('on a 10 mm aluminium adapter disc on the FR3 tool flange' if spec.get('adapter')
             else 'directly on the FR3 tool flange')
    return prompt + (f'\nEmbodiment: {spec["label"]} mounted {mount}; right_hand_flange names that FR3 tool-flange '
                     f'frame (fr3_link8), fingers point along its +z and the palm faces its +x. Available fingers: {fingers}; '
                     'requests for other finger names are rejected. Fingertip points: '
                     f'{spec.get("controller_fingertip_semantics", "URDF tip frame origins")} Tactile, when available, '
                     'reports whole distal links, not pads. This hand has no humanlike-grasp contract; contact '
                     'and pinch rules use its own distal links.\n')


def model_system_prompt(control_mode, interaction_mode, controller=None, robot_id=None, mount=None):
    """Bind behavioral claims to the negotiated descriptor before hashing settings.

    ``mount`` is the task row's published ``robot_mount`` (absent for v1 mounts).
    """
    prompt = ({'tools': TOOLS_PROMPT, 'code': CODE_PROMPT}.get(interaction_mode)
              or (SYSTEM_PROMPT if control_mode == 'fingertip' else JOINT_SYSTEM_PROMPT))
    if interaction_mode == 'free':
        prompt = FREE_SYSTEM_PROMPT
    if control_mode in ('fingertip', 'hybrid'):
        descriptor = PolicyDescriptor.model_validate(controller) if controller is not None else None
        if descriptor is not None and descriptor.hand_goal_semantics is not None:
            semantics=descriptor.hand_goal_semantics
            if semantics.protocol=='accepted_hand_goal_retention_v2':
                prompt += '\n' + HAND_REFERENCE_RULES_V2.format(epsilon_rad=semantics.measured_reference.epsilon_rad)
            else:
                prompt += '\n' + HAND_REFERENCE_RULES
        else:
            prompt += ('\nNo retained-hand-goal contract was advertised for this controller. '
                       'Do not assume omitted fingers preserve an earlier accepted grasp goal.\n')
    return mount_prompt(embodiment_prompt(prompt + '\n' + ROBOT_FACTS + TERMINATION_RULES + '\n' + TACTILE_RULES, robot_id), mount)


def select_geometry_policy(rows, robot_id, *, allow_instances=False):
    """Pin one advertised public classical executor, never silently substitute."""
    if not isinstance(rows, list):
        raise ValueError('Invalid public controller inventory')
    candidates = []
    supported = {'classical_fingertip', 'classical_fingertip_bimanual'}
    if allow_instances:
        supported.add('classical_fingertip_instances')
    for row in rows:
        descriptor = PolicyDescriptor.model_validate(row)
        if (descriptor.policy_id in supported and not descriptor.trained
                and descriptor.observation_profile == 'vision_proprio'
                and descriptor.camera_observation_mode == 'decision_only'
                and robot_id in descriptor.supported_robot_ids
                and {'move_geometry','hold'} <= set(descriptor.skills)
                and {'wrist_pose', 'fingertip_positions'} <= set(descriptor.geometry_inputs)):
            candidates.append(descriptor)
    if len(candidates) != 1:
        raise ValueError('Exactly one compatible classical_fingertip controller must be advertised')
    return candidates[0].model_dump(mode='json')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def interface_contract(control_mode, interaction_mode):
    if control_mode == 'hybrid' and interaction_mode == 'free':
        return free_interface_contract()
    return {'version':'agent_control_v6',
        'joint_delta_reference':'commanded_joint_positions_rad when published, else measured; omitted joints keep commanded targets',
        'direct_joint_execution':'advertised by capabilities.direct_joint_execution',
        'actions':(['joint_delta','hold','stop'] if control_mode=='joints' else ['move_geometry','hold','stop'])
            if interaction_mode=='action' else [],
        'tools':['observe','point_at','move_geometry','hold','stop'] if interaction_mode!='action' else [],
        'move_geometry_duration':'maximum_execution_and_settling_time; may complete early',
        'geometry_tolerances':{'optional':['position_tolerance_m','orientation_tolerance_rad'],
            'values':'finite_positive','effective':'minimum_of_request_and_advertised_controller_cap',
            'omission':'unchanged_defaults','fingertip_tolerance':'fixed_by_controller'},
        'hold_duration':'requested physics duration; early exit only on episode termination, cancellation or execution failure',
        'hold_counts_as_primitive_and_program_move':True,
        'execution_diagnostics':'allowlisted robot-only diagnostic; private policy text omitted',
        'program_feedback':'primitive_results preserves every move/hold outcome; query_results preserves ordered public point_at samples at their original sequence and camera revision, independently of last_result; historical pixels are not current object poses'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def require_public(observation, episode_id):
    if observation.get('episode_id') != episode_id or observation.get('profile') != 'vision_proprio':
        raise ValueError('Unexpected episode or observation profile')
    forbidden = {'objects', 'cloth', 'contacts', 'metrics', 'score', 'evaluator'}
    if forbidden.intersection(observation) or any(k.startswith('privileged') for k in observation):
        raise ValueError('Refusing nonpublic model input')


def decision_text(task, capabilities, observation, limits, feedback, decisions_remaining,
                  depth_observation=None, interaction=None):
    return json.dumps({'task': task, 'capabilities': capabilities,
        'observation': observation, 'action_limits': asdict(limits),
        'previous_action_feedback': feedback, 'decisions_remaining': decisions_remaining,
        'depth_observation': depth_observation, 'interaction':interaction},
        ensure_ascii=False, sort_keys=True, allow_nan=False)


def history_text(text):
    """Keep past state and feedback; the current turn carries static context."""
    value = json.loads(text)
    for key in ('task', 'capabilities', 'action_limits'):
        value.pop(key)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class WallDeadlineExpired(TimeoutError):
    """Local wall budget exhausted before another model call can start."""


def model_time_remaining(session):
    if session.deadline is None:
        return None
    remaining = session.deadline - time.monotonic()
    if remaining <= 0:
        raise WallDeadlineExpired
    return remaining


def configured_model(args, config):
    """Explicit provider selection; credentials and protocol never fall back."""
    provider = getattr(args, 'provider', 'gateway')
    if provider not in ('gateway', 'micu'):
        raise ValueError('Unknown model provider')
    retries = getattr(args, 'model_retries', None)
    if retries is None and provider == 'gateway':
        retries = 0
    policy = getattr(args, 'model_retry_policy', None)
    expected_policy = MICU_RETRY_POLICY if provider == 'micu' else 'legacy_gateway_v1'
    if policy is not None and policy != expected_policy:
        raise ValueError('Historical retry policy cannot run on this client; create a new frozen plan')
    if provider == 'micu' and not args.credentials:
        raise ValueError('Micu requires an explicit external --credentials profile')
    hostname = urlsplit(config.public_metadata()['base_url']).hostname or ''
    if provider != 'micu' and (hostname == 'micuapi.ai' or hostname.endswith('.micuapi.ai')):
        raise ValueError('Micu credentials require explicit --provider micu (Responses)')
    client_type = MicuModelClient if provider == 'micu' else ModelClient
    return client_type(config, args.model, timeout_s=args.model_timeout, max_retries=retries,
                       reasoning_effort=args.reasoning_effort)


class _TerminalResultBinding:
    """Accept terminal evidence only for the immutable created episode."""

    def __init__(self):
        self._spec = None
        self.error = None

    def _reject(self):
        error = BenchHTTPError('Terminal result does not match the created episode',
                               category='invalid_response')
        if self.error is None:
            self.error = error_record(error)
        raise error from None

    def bind(self, spec):
        try:
            value = EpisodeSpec.model_validate(spec).model_dump(mode='json')
        except (ValueError, TypeError):
            self._reject()
        if self._spec is not None and value != self._spec:
            self._reject()
        self._spec = value

    def validate(self, value, *, allow_pending=True):
        if value is None and allow_pending:
            return None
        try:
            parsed = EpisodeResult.model_validate(value)
        except (ValueError, TypeError):
            self._reject()
        if (self.error is not None or self._spec is None
                or parsed.spec.model_dump(mode='json') != self._spec):
            self._reject()
        return parsed.model_dump(mode='json')


def run(args):
    source_verification = verify_if_snapshot(Path(__file__).absolute().parents[2],
        required_modules=("agent.model_client.runner", "bench.protocol.schemas"),
        entrypoint_file=Path(__file__))
    optional_positive(args.sim_budget, 'sim_budget', maximum=MAX_EPISODE_SIM_TIME_S)
    optional_positive(args.wall_budget, 'wall_budget', maximum=MAX_EPISODE_WALL_TIME_S)
    if args.max_output_tokens is not None and (type(args.max_output_tokens) is not int
            or not 1 <= args.max_output_tokens <= 32768):
        raise ValueError('max_output_tokens must be null or an integer from 1 to 32768')
    visual_input = getattr(args, 'visual_input', 'rgbd')
    if visual_input not in ('rgbd', 'rgb'):
        raise ValueError('Unknown visual input mode')
    control_mode = getattr(args, 'control_mode', 'fingertip')
    if control_mode not in ('fingertip', 'joints', 'hybrid'):
        raise ValueError('Unknown control mode')
    interaction_mode = getattr(args,'interaction_mode','action')
    tactile_input = getattr(args,'tactile_input','on')
    if interaction_mode not in ('action','tools','code','free') or tactile_input not in ('on','off'):
        raise ValueError('Unknown interaction or tactile mode')
    if (control_mode == 'hybrid') != (interaction_mode == 'free'):
        raise ValueError('The free interface is available only in the independent hybrid track')
    if control_mode=='joints' and interaction_mode!='action':
        raise ValueError('Tool and code modes require the fixed fingertip executor')
    max_actions = getattr(args,'max_actions',None)
    max_actions = args.max_decisions if max_actions is None else max_actions
    max_tools = getattr(args,'max_tool_calls_per_decision',8)
    max_program_actions = getattr(args,'max_program_actions',3)
    if any(type(v) is not int or not 1 <= v <= maximum for v,maximum in
           ((args.max_decisions,256),(max_actions,MAX_EPISODE_ACTIONS),(max_tools,32),(max_program_actions,8))):
        raise ValueError('Invalid model, primitive or program budget')
    program_limits = ProgramLimits(max_tool_calls=max_tools,max_move_calls=max_program_actions,
        max_runtime_s=getattr(args, 'program_timeout', 120.))
    if args.robot == 'dual_fr3_sharpa_w01' and (control_mode != 'fingertip' or interaction_mode != 'action'):
        raise ValueError('The bimanual robot is supported by the fingertip action interface only')
    system_prompt = model_system_prompt(control_mode, interaction_mode, robot_id=args.robot)
    config = ModelGatewayConfig.from_file(args.credentials) if args.credentials else ModelGatewayConfig.from_env()
    model = configured_model(args, config)
    bench = BenchHTTPClient(args.bench_url, timeout_s=args.http_timeout)
    limits = (FreeActionLimits() if control_mode == 'hybrid' else
              GeometryActionLimits() if control_mode == 'fingertip' else ActionLimits())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if source_verification is not None:
        save(output/'runtime_imports.json', source_verification)
    source_root = Path(__file__).parent
    source_paths = sorted(source_root.rglob('*.py'))
    sources = {}
    for path in source_paths:
        relative = path.relative_to(source_root)
        copied = output/'client_source'/relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        sources[relative.as_posix()] = hashlib.sha256(copied.read_bytes()).hexdigest()
    project = Path(__file__).resolve().parents[2]
    for name in CLIENT_SOURCE_DEPENDENCIES:
        copied = output/'client_source'/'dependencies'/name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project/name, copied)
        sources['dependencies/'+name] = hashlib.sha256(copied.read_bytes()).hexdigest()
    settings = {'schema': 'online_visual_agent.' + AGENT_VERSION, 'model': args.model,
        'visual_input': visual_input, 'control_mode': control_mode, 'controller': None,
        'interaction_mode':interaction_mode,'tactile_input':tactile_input,
        'interface_contract':interface_contract(control_mode,interaction_mode),
        'primitive_attempt_accounting':PRIMITIVE_ATTEMPT_ACCOUNTING,
        'action_recovery_policy': {'protocol': 'agent_action_recovery_v1',
            'max_status_queries': DEFAULT_MAX_STATUS_QUERIES, 'scope': 'per_wait_status_operations'},
        'max_actions':max_actions,'max_tool_calls_per_decision':max_tools,
        'max_program_actions':max_program_actions,'program_limits':asdict(program_limits),
        'visibility_enforcement':'configured_model_and_local_tool_projection; client_observed_not_server_auth',
        'depth_representation': {'source': 'RTX distance_to_image_plane',
            'raw_encoding': 'float32_npy', 'unit': 'm', 'semantics': 'camera_z',
            'invalid_value': 0., 'preview_range_m': [.1, 4.],
            'grid_columns': 16, 'grid_rows': 12,
            'model_receives': 'aligned depth preview PNG and metric sample grid; raw NPY archived'}
            if visual_input == 'rgbd' else None,
        'reasoning_effort': args.reasoning_effort, 'max_output_tokens': args.max_output_tokens,
        'provider': getattr(args, 'provider', 'gateway'),
        'gateway': model.public_metadata() if getattr(args, 'provider', 'gateway') == 'micu' else config.public_metadata(),
        'model_retries': getattr(args, 'model_retries', None) if getattr(args, 'model_retries', None) is not None
                         else (None if getattr(args, 'provider', 'gateway') == 'micu' else 0),
        'model_retry_policy': MICU_RETRY_POLICY if getattr(args, 'provider', 'gateway') == 'micu' else 'legacy_gateway_v1',
        'system_prompt': system_prompt,
        'action_limits': asdict(limits), 'max_decisions': args.max_decisions,
        'model_timeout_s': args.model_timeout, 'http_timeout_s': args.http_timeout,
        'history_text_turns': 4,
        'history_context': 'dynamic_history_v1; task/capabilities/action_limits supplied in current turn only',
        'client_sources_sha256': sources,
        'source_snapshot_verification': 'runtime_imports.json' if source_verification is not None else None,
        'httpx_version': importlib.metadata.version('httpx'),
        'numpy_version': importlib.metadata.version('numpy'),
        'model_accounting': 'client_observed_not_server_verified',
        'trained_system0': False, 'task_specific_controller': False,
        'budgets': {'sim_time_s': args.sim_budget, 'wall_time_s': args.wall_budget,
            'max_actions': max_actions, 'max_model_calls': args.max_decisions,
            'max_action_sim_s': max(limits.max_duration_s,getattr(limits,'max_hold_duration_s',0.))}}
    save(output/'agent_config.json', settings)
    episode_id = None
    result = None
    result_source = None
    terminal_binding = _TerminalResultBinding()
    session = None
    termination = 'client_error'
    failure = None
    model_error = None
    run_error = finalization_error = None
    calls = submitted = executed = 0
    history = []
    feedback = None
    model_reports = []
    model_http_attempts = None
    model_retry_wait_s = 0.
    recovery_pending = False
    recovery_submitted = 0
    model_observation_sequences = []
    decision_statuses, program_statuses = [], []
    program_multiple_actions = False
    try:
        tasks = bench.list_tasks()
        matches = [task for task in tasks if task['task_id'] == args.task]
        if len(matches) != 1:
            raise ValueError('Task ID is not uniquely advertised by bench')
        task = matches[0]
        if task.get('task_kind') in ('board_game', 'turn_game'):
            raise ValueError('Player games require agent.model_client.function_runner and authenticated player sessions')
        if control_mode in ('fingertip', 'hybrid'):
            settings['controller'] = select_geometry_policy(bench.list_policies(task_id=args.task), args.robot)
            if control_mode == 'hybrid':
                settings['interface_inventory'] = require_free_interface(
                    bench.list_interfaces(task_id=args.task), settings['controller'])
                expected_interface = getattr(args, 'expected_interface_inventory', None)
                if expected_interface is not None and expected_interface != settings['interface_inventory']:
                    raise ValueError('Public interface changed after experiment plan was frozen')
            expected_controller = getattr(args,'expected_controller',None)
            if expected_controller is not None and PolicyDescriptor.model_validate(expected_controller) != PolicyDescriptor.model_validate(settings['controller']):
                raise ValueError('Controller changed after experiment plan was frozen')
            system_prompt = model_system_prompt(control_mode, interaction_mode, settings['controller'], robot_id=args.robot,
                                                mount=task.get('robot_mount'))
            settings['system_prompt'] = system_prompt
            save(output/'agent_config.json', settings)
        elif task.get('robot_mount'):
            # A published non-v1 hand mount is stated in every track's prompt.
            system_prompt = model_system_prompt(control_mode, interaction_mode, robot_id=args.robot,
                                                mount=task['robot_mount'])
            settings['system_prompt'] = system_prompt
            save(output/'agent_config.json', settings)
        create = {'task_id': args.task, 'robot_id': args.robot, 'seed': args.seed,
            'scene_variant': 'original_v1', 'agent_id': 'online_visual_'+control_mode+'_'+interaction_mode+'_agent',
            'agent_version': AGENT_VERSION, 'agent_config_sha256': canonical_hash(settings),
            'track': ('free_tool_agent' if control_mode == 'hybrid' else
                      'system0' if control_mode == 'fingertip' else 'direct_joints'), 'observation_profile': 'vision_proprio',
            'budgets': settings['budgets']}
        if control_mode in ('fingertip', 'hybrid'):
            create.update(policy_id=settings['controller']['policy_id'], policy_version=settings['controller']['version'])
        save(output/'create_request.json', create)
        ready = bench.create(create)
        episode_id = ready['episode_id']
        save(output/'episode_ready.json', ready)
        started = time.monotonic()
        capabilities, spec = ready['capabilities'], deepcopy(ready['spec'])
        # Discovery omits seed-dependent goals. The created ordinary episode
        # publishes its actual task (notably card-room task0).
        if 'task' in ready:
            actual_task = ready['task']
            if not isinstance(actual_task, dict) or actual_task.get('task_id') != args.task:
                raise ValueError('Created episode public task identity differs')
            task = actual_task
        if args.task == 'task0' and (task.get('episode_seed') != spec.get('seed')
                or len(task.get('goal', {}).get('target_cells', [])) != 3
                or len(task.get('goal', {}).get('target_positions_m', [])) != 3):
            raise ValueError('Task0 requires episode-specific public placement goals')
        if spec.get('episode_id') != episode_id or spec.get('agent_config_sha256') != canonical_hash(settings):
            raise ValueError('Ready episode does not pin the Agent configuration')
        ready_budgets = spec.get('budgets')
        if not isinstance(ready_budgets, dict) or any(
                key not in ready_budgets or isinstance(ready_budgets[key], bool)
                or ready_budgets[key] != value for key, value in settings['budgets'].items()):
            raise ValueError('Ready episode does not pin the requested budgets')
        for key,value in (getattr(args,'expected_spec_pins',None) or {}).items():
            if spec.get(key) != value:
                raise ValueError('Episode does not match the frozen experiment pins')
        if control_mode in ('fingertip', 'hybrid'):
            pinned = PolicyDescriptor.model_validate(settings['controller'])
            expected_track = 'free_tool_agent' if control_mode == 'hybrid' else 'system0'
            if (spec.get('track') != expected_track or spec.get('agent_config_sha256') != canonical_hash(settings)
                    or PolicyDescriptor.model_validate(spec.get('system0')) != pinned
                    or PolicyDescriptor.model_validate(capabilities.get('system0')) != pinned):
                raise ValueError('Ready episode does not pin the requested controller and Agent configuration')
        if visual_input == 'rgbd' and not capabilities.get('depth_available'):
            raise ValueError('RGB-D requested but the server has no depth sensor')
        # Cleanup must not score a created episode that failed admission above.
        terminal_binding.bind(spec)
        observation = ready['observation']
        session = InteractionSession(bench,episode_id,capabilities,spec,observation,output,
            control_mode=control_mode,visual_input=visual_input,tactile_input=tactile_input,
            limits=limits,deadline=None if args.wall_budget is None else started+args.wall_budget,
            max_actions=max_actions, result_validator=terminal_binding.validate)
        visible_capabilities = capability_view(capabilities,visual_input=visual_input,tactile_input=tactile_input)
        print(json.dumps({'event': 'episode_ready', 'episode_id': episode_id,
                          'model': args.model, 'reasoning_effort': args.reasoning_effort}), flush=True)
        for index in range(args.max_decisions):
            model_time_remaining(session)
            result = terminal_binding.validate(session.http(bench.get_result, episode_id))
            if result is not None:
                result_source = 'server_result'
                termination = 'server_finished'
                break
            if session.move_attempts >= max_actions:
                termination = 'client_action_budget'
                break
            require_public(observation, episode_id)
            folder = output/'decisions'/f'{index:04d}'
            captured = session.http(capture,bench,episode_id,observation,folder,output,
                visual_input=visual_input,tactile_input=tactile_input)
            text = decision_text(task,visible_capabilities,captured['observation'],limits,feedback,
                args.max_decisions-index,captured['depth_context'],interaction={
                    'mode':interaction_mode,'primitive_attempts_remaining':max_actions-session.move_attempts,
                    'contract':settings['interface_contract'],
                    'program_limits':asdict(program_limits) if interaction_mode in ('code', 'free') else None,
                    'point_at_available':interaction_mode!='action' and visual_input=='rgbd',
                    'tactile_input':tactile_input})
            content = [{'type':'text','text':text}]+captured['image_content']
            recorded_content = [{'type':'text','text':text}]+captured['recorded_image_content']
            turns = history[-4:] + [{'role': 'user', 'content': content}]
            model_time_remaining(session)
            save(folder/'model_request.json', {'system_prompt': system_prompt,
                'messages': history[-4:]+[{'role': 'user', 'content': recorded_content}],
                'expanded_messages_sha256': canonical_hash(turns),
                'model': args.model, 'reasoning_effort': args.reasoning_effort,
                'max_output_tokens': args.max_output_tokens})
            # Download, depth conversion, prompt construction and recording all
            # consume the episode budget. Never reuse a pre-capture timeout.
            remaining = model_time_remaining(session)
            calls += 1
            recovery_pending = False
            recovery_submitted = session.submitted
            model_observation_sequences.append(observation['sequence'])
            request_sha256 = canonical_hash(turns)

            def record_model_event(event):
                nonlocal model_http_attempts, model_retry_wait_s
                entry = {'decision_index': index, 'request_sha256': request_sha256, **event}
                # Append small events instead of retaining an unbounded retry
                # history in memory. Flush before sleeping or starting a request.
                with (folder/'model_api_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if event['event'] == 'model_request_started':
                    model_http_attempts = (model_http_attempts or 0) + 1
                elif event['event'] == 'model_request_not_sent':
                    model_http_attempts -= 1
                elif event['event'] == 'model_retry_wait_finished':
                    model_retry_wait_s += event['waited_s']
                if event['event'] == 'model_retry_wait':
                    print(json.dumps(entry), flush=True)

            try:
                options = ({'timeout_s': args.model_timeout, 'retry_deadline': session.deadline,
                            'on_event': record_model_event} if settings['provider'] == 'micu' else
                           {'timeout_s': minimum_limit(args.model_timeout, remaining)})
                response = model.generate(system_prompt, turns, max_output_tokens=args.max_output_tokens, **options)
            except ModelGatewayError as exc:
                model_error = {'decision_index':index, **exc.public_metadata()}
                save(folder/'model_error.json', model_error)
                raise
            metadata = response.public_metadata()
            recovery_pending = settings['provider'] == 'micu' and metadata.get('attempts', 1) > 1
            model_reports.append(metadata)
            save(folder/'model_response.json', {'text': response.text, 'metadata': metadata})
            history = (history + [{'role': 'user', 'content': history_text(text)},
                                  {'role': 'assistant', 'content': response.text}])[-4:]
            # The wall watchdog may terminate the episode while the model thinks.
            model_time_remaining(session)
            result = terminal_binding.validate(session.http(bench.get_result, episode_id))
            if result is not None:
                result_source = 'server_result'
                if recovery_pending and result.get('outcome') == 'wall_timeout':
                    raise ModelGatewayError('Episode watchdog expired during API recovery',
                                            attempts=metadata['attempts'], category='request_budget_exhausted')
                termination = 'server_finished_during_model_call'
                break
            latest = session.http(bench.get_observation, episode_id)
            require_public(latest, episode_id)
            session.observation = latest
            try:
                if latest['sequence'] != observation['sequence'] and interaction_mode == 'code':
                    raise ActionValidationError('Model decision uses a stale observation')
                if interaction_mode == 'free':
                    completed_before = session.completed
                    feedback, program = execute_free_decision(
                        response.text, session, folder, index, observation['sequence'], program_limits)
                    if program is not None:
                        program_statuses.append(program['status'])
                        program_multiple_actions |= session.completed-completed_before >= 2
                elif interaction_mode == 'action':
                    # Parse only the outer JSON before shared attempt accounting.
                    action = parse_json_object(response.text)
                    feedback = session.move(action,folder,f'model_{index:04d}',observation['sequence'])
                elif interaction_mode == 'tools':
                    call = parse_json_object(response.text)
                    value = session.invoke_call(call,folder=folder,decision_index=index,
                                                expected_sequence=observation['sequence'])
                    feedback = {'status':'tool_completed','call_status':'completed','tool':call['tool'],
                        'primitive_status':value.get('feedback',{}).get('status') if call['tool'] in ('move_geometry','hold') else None,
                        'result':value}
                else:
                    (folder/'program.txt').write_text(response.text)
                    completed_before = session.completed
                    session.program_deadline = (None if program_limits.max_runtime_s is None
                                                else time.monotonic()+program_limits.max_runtime_s)
                    try:
                        program = execute_program(response.text,
                            lambda name,arguments: session.invoke(name,arguments,folder=folder,decision_index=index),
                            limits=program_limits)
                    finally:
                        session.program_deadline = None
                    save(folder/'program_execution.json',program)
                    program_statuses.append(program['status'])
                    program_multiple_actions |= session.completed-completed_before >= 2
                    feedback = program_feedback(program)
                    if session.fatal_error is not None:
                        raise session.fatal_error
            except ActionValidationError as exc:
                if (recovery_pending and session.submitted == recovery_submitted
                        and session.deadline is not None and time.monotonic() >= session.deadline):
                    raise
                feedback = {'status':'rejected_by_client','reason':str(exc), 'error':exc.public_metadata()}
                decision_statuses.append(feedback['status'])
                save(folder/'execution.json',feedback)
                observation = session.observation
                print(json.dumps({'event':'decision_rejected','decision':index,'feedback':feedback},
                                 ensure_ascii=False),flush=True)
                continue
            save(folder/'execution.json',feedback)
            recovery_pending = False
            decision_statuses.append(feedback['status'])
            submitted, executed = session.submitted, session.completed
            result = terminal_binding.validate(
                session.result if session.result is not None else session.http(bench.get_result, episode_id))
            print(json.dumps({'event':'decision_executed','decision':index,
                'interaction_mode':interaction_mode,'status':feedback['status'],
                'submitted_actions':submitted,'completed_actions':executed},ensure_ascii=False),flush=True)
            if result is not None:
                result_source = 'session_result' if session.result is not None else 'server_result'
                termination = 'server_finished'
                break
            if session.stop_requested:
                termination = 'model_requested_stop'
                break
            observation = session.observation
        else:
            termination = 'client_decision_budget'
    except KeyboardInterrupt:
        termination = 'client_interrupted'
        failure = 'KeyboardInterrupt'
    except WallDeadlineExpired:
        if recovery_pending and session.submitted == recovery_submitted:
            failure = 'ModelGatewayError'
            run_error = error_record(ModelGatewayError('Episode deadline expired during API recovery',
                                                       category='request_budget_exhausted'))
        else:
            termination = 'client_wall_deadline'
    except ModelGatewayError as exc:
        # A request timeout is still an API interruption when its return
        # crosses the episode wall deadline. Only the explicit local budget
        # checks (or a valid response followed by one) establish that ending.
        failure = type(exc).__name__
        run_error = error_record(exc)
        if exc.category == 'output_budget_exhausted':
            termination = 'client_output_budget'
    except (BenchHTTPError, ActionValidationError) as exc:
        if (isinstance(exc, ActionValidationError) and session is not None and session.deadline is not None
                and time.monotonic() >= session.deadline):
            if recovery_pending and session.submitted == recovery_submitted:
                failure = 'ModelGatewayError'
                run_error = error_record(ModelGatewayError('Episode deadline expired during API recovery',
                                                           category='request_budget_exhausted'))
            else:
                termination = 'client_wall_deadline'
        else:
            failure = type(exc).__name__
            run_error = error_record(exc)
    except Exception as exc:
        # Exception messages may contain upstream HTTP bodies; record type only.
        failure = type(exc).__name__
        run_error = error_record(exc)
    finally:
        # A later observation/logging error must not erase a terminal result
        # already received by a tool or primitive.
        if result is None and session is not None and session.result is not None:
            try:
                result = terminal_binding.validate(session.result)
                result_source = 'session_result'
            except BenchHTTPError as exc:
                failure = failure or type(exc).__name__
                run_error = run_error or error_record(exc)
        if episode_id is not None and result is None:
            try:
                result = terminal_binding.validate(bench.get_result(episode_id))
                if result is not None:
                    result_source = 'server_result'
            except Exception as exc:
                failure = failure or ('Finalization'+type(exc).__name__)
                finalization_error = error_record(exc)
            # A bad/unavailable GET does not establish that the worker stopped.
            # Finish once even after rejection, retaining the identity failure.
            if result is None:
                try:
                    result = terminal_binding.validate(bench.finish(episode_id), allow_pending=False)
                    result_source = 'cleanup_finish'
                except Exception as exc:
                    failure = failure or ('Finalization'+type(exc).__name__)
                    finalization_error = finalization_error or error_record(exc)
        if terminal_binding.error is not None:
            # Later cleanup evidence cannot rehabilitate an invalid identity.
            result = result_source = None
            failure = failure or 'BenchHTTPError'
            run_error = run_error or terminal_binding.error
        if session is not None:
            submitted, executed = session.submitted, session.completed
        action_recovery_pending = False
        action_recovery_error = None
        try:
            action_recovery_pending = bool(getattr(getattr(session, 'action_api', None), 'pending', None))
        except Exception as exc:
            # A damaged/unreadable journal must leave execution unresolved,
            # while still allowing the runner's interruption report to persist.
            action_recovery_pending = True
            action_recovery_error = error_record(exc)
            failure = failure or ('RecoveryState'+type(exc).__name__)
        report = {'schema': 'online_visual_agent_run.v7', 'episode_id': episode_id,
            'visual_input': visual_input, 'control_mode': control_mode, 'controller': settings['controller'],
            'interaction_mode':interaction_mode,'tactile_input':tactile_input,
            'primitive_attempt_accounting':PRIMITIVE_ATTEMPT_ACCOUNTING,
            'primitive_attempts':session.move_attempts if session else 0,
            'tool_calls':session.tool_calls if session else 0,
            'point_queries':session.point_queries if session else 0,
            'read_calls':session.read_calls if session else 0,
            'decision_status_counts':dict(Counter(decision_statuses)),
            'interface_selection_counts':dict(session.interface_selections) if session else {},
            'independent_free_tool_track': interaction_mode == 'free',
            'program_status_counts':dict(Counter(program_statuses)),
            'model': args.model, 'reasoning_effort': args.reasoning_effort,
            'termination': termination, 'error_type': failure,
            'model_error': model_error, 'run_error': run_error,
            'finalization_error': finalization_error, 'result_source': result_source,
            'result_validation_error': terminal_binding.error,
            'action_recovery_pending': action_recovery_pending,
            'action_recovery_error': action_recovery_error,
            'model_call_attempts': calls, 'model_responses': len(model_reports),
            'model_http_attempts': model_http_attempts, 'model_retry_wait_s': model_retry_wait_s,
            'model_input_observation_sequences': model_observation_sequences,
            'submitted_actions': submitted, 'completed_actions': executed,
            'primitive_termination_reason_counts':dict(session.termination_reasons) if session else {},
            'model_response_metadata': model_reports, 'result': result,
            'config_sha256': canonical_hash(settings),
            'online_model_decision_loop_exercised': bool(executed >= 1 and failure is None and any(
                later > earlier for earlier, later in zip(model_observation_sequences, model_observation_sequences[1:]))),
            'program_multiple_actions_exercised':program_multiple_actions,
            'model_accounting': 'client_observed_not_server_verified',
            'scope': ('Independent development free-interface Agent with direct joints and a pinned classical executor; '
                      if control_mode == 'hybrid' else
                      'Development online visual sparse-geometry Agent with a fixed classical IK executor; '
                      if control_mode == 'fingertip' else 'Development online visual direct-joint Agent; ')
                     + 'no trained RL, no task script fallback, no leaderboard claim.'}
        report.update(classify_report(report))
        save(output/'run_report.json', report)
        hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output.rglob('*')) if path.is_file()}
        save(output/'artifact_hashes.json', hashes)
        bench.close()
        model.close()
    print(json.dumps({k: report[k] for k in ('episode_id', 'termination', 'error_type',
        'model_call_attempts', 'completed_actions', 'online_model_decision_loop_exercised', 'task_success')},
        ensure_ascii=False), flush=True)
    return report


def optional_output_tokens(value):
    """Explicitly omit the provider token cap; never substitute a large number."""
    if value.lower() in ('none', 'null', 'disabled'):
        return None
    try:
        tokens = int(value)
        if not 1 <= tokens <= 32768:
            raise ValueError('out of range')
        return tokens
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('use an integer from 1 to 32768 or none/null/disabled') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-url', default='http://127.0.0.1:8000')
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--provider', choices=('gateway', 'micu'), default='gateway',
                        help='micu uses Responses at the explicit external credential profile endpoint')
    parser.add_argument('--model-retries', type=int, choices=range(4),
                        help='Optional additional-request cap 0–3; Micu defaults to retry every 10s until recovery/wall deadline, gateway to 0')
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--reasoning-effort', default='max')
    parser.add_argument('--visual-input', choices=('rgbd', 'rgb'), default='rgbd')
    parser.add_argument('--control-mode', choices=('fingertip', 'joints', 'hybrid'), default='fingertip')
    parser.add_argument('--interaction-mode',choices=('action','tools','code','free'),default='action')
    parser.add_argument('--tactile-input',choices=('on','off'),default='on')
    parser.add_argument('--max-actions',type=int,help='Primitive attempt budget; defaults to max-decisions')
    parser.add_argument('--max-tool-calls-per-decision',type=int,default=8)
    parser.add_argument('--max-program-actions',type=int,default=3)
    parser.add_argument('--max-output-tokens', type=optional_output_tokens, default=16384,
                        help='Output token cap, or none to omit the provider limit')
    parser.add_argument('--model-timeout', type=optional_seconds, default=300., help='Seconds or none to disable')
    parser.add_argument('--http-timeout', type=optional_seconds, default=300., help='Seconds or none to disable')
    parser.add_argument('--program-timeout', type=optional_seconds, default=120., help='Seconds or none to disable')
    parser.add_argument('--task', default='01_cube_push')
    parser.add_argument('--robot', default='sharpa_w01_right')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-decisions', type=int, default=100)
    parser.add_argument('--sim-budget', type=optional_seconds, default=45., help='Seconds or none to disable')
    parser.add_argument('--wall-budget', type=optional_seconds, default=1800., help='Seconds or none to disable')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    if report['error_type']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
