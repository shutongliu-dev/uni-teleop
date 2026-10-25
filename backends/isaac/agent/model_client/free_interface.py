"""One explicit interface choice per decision, sharing an episode and counters.

This is a separate development track. It dispatches existing public primitives;
it neither restarts the simulator nor provides task-specific motion helpers.
"""
from dataclasses import dataclass, field
from copy import deepcopy
import time

from bench.protocol.schemas import PolicyDescriptor

from .actions import ActionLimits, ActionValidationError, parse_json_object, parse_model_action
from .geometry_actions import GeometryActionLimits, parse_geometry_action
from .capture import save
from .interaction import PRIMITIVE_ATTEMPT_ACCOUNTING
from .program import execute_program, program_feedback


INTERFACES = ('joints_action', 'fingertip_action', 'fingertip_tools', 'fingertip_code')


@dataclass(frozen=True)
class FreeActionLimits:
    joints: ActionLimits = field(default_factory=ActionLimits)
    fingertip: GeometryActionLimits = field(default_factory=GeometryActionLimits)

    @property
    def max_duration_s(self):
        return max(self.joints.max_duration_s, self.joints.max_hold_duration_s,
                   self.fingertip.max_duration_s)


def free_interface_contract():
    return {
        'version': 'free_tool_agent_v3',
        'track': 'free_tool_agent',
        'interfaces': list(INTERFACES),
        'response': {'interface': 'one advertised interface',
                     'payload': 'original action/tool object or code source string'},
        'selection_boundary': 'one interface per model decision',
        'shared_across_interfaces': ['episode', 'task', 'model_decisions',
                                    'primitive_attempts', 'observation_permissions'],
        'code_tools': ['observe', 'point_at', 'move_geometry', 'hold', 'stop'],
        'program_query_feedback': 'query_results preserves ordered public point_at samples at their original sequence and camera revision; historical pixels are not current object poses',
        'code_joint_actions': False,
        'joint_control': 'original direct-joint primitive; no classical smoothing guarantee',
        'geometry_control': 'pinned classical executor with acknowledged drive handoff',
        'geometry_tolerances': {'optional': ['position_tolerance_m', 'orientation_tolerance_rad'],
            'values': 'finite_positive', 'effective': 'minimum_of_request_and_advertised_controller_cap',
            'omission': 'unchanged_defaults', 'fingertip_tolerance': 'fixed_by_controller'},
        'invalid_selection_cost': 'model decision; no fabricated physical submission',
        'primitive_attempt_accounting': PRIMITIVE_ATTEMPT_ACCOUNTING,
        'invalid_selected_action_cost': 'recognized joint_delta/move_geometry/hold request: one shared primitive attempt including invalid parameters or stale action/tool context; unknown/stop/read-only and whole-program prevalidation rejection: none',
        'task_success': 'private evaluator and physics validity; tool completion is not success',
        'leaderboard_eligible': False,
        'fixed_four_mode_comparison_member': False,
    }


def require_free_interface(catalog, controller):
    """Require explicit server support before creating a mixed episode."""
    if type(catalog) is not dict or catalog.get('protocol') != 'agent_interface_catalog_v1':
        raise ValueError('Invalid public interface inventory')
    if type(catalog.get('interfaces')) is not list or type(catalog.get('shared_contract')) is not dict:
        raise ValueError('Incomplete public interface inventory')
    rows = [row for row in catalog['interfaces']
            if type(row) is dict and row.get('id') == 'free_tool_agent']
    if len(rows) != 1:
        raise ValueError('Exactly one free-tool interface must be advertised')
    row = rows[0]
    if (row.get('available') is not True or row.get('native_track') != 'free_tool_agent'
            or row.get('control_mode') != 'hybrid' or row.get('interaction_mode') != 'free'
            or row.get('executor') != 'classical_fingertip_with_direct_joint_bypass'
            or row.get('handoff_protocol') != 'acknowledged_direct_handoff_v1'):
        raise ValueError('The public server does not advertise the required free-tool executor')
    descriptor = PolicyDescriptor.model_validate(controller)
    handoff = getattr(descriptor, 'direct_control_handoff', None)
    if handoff is None or handoff.protocol != 'acknowledged_direct_handoff_v1':
        raise ValueError('The pinned controller has no acknowledged direct-drive handoff')
    for policies in (catalog.get('geometry_policies'), row.get('policies')):
        if type(policies) is not list or sum(PolicyDescriptor.model_validate(p) == descriptor
                                           for p in policies) != 1:
            raise ValueError('The free interface inventory does not pin the selected controller')
    return {'free_tool_agent': deepcopy(row), 'shared_contract': deepcopy(catalog['shared_contract'])}


def parse_free_decision(value):
    envelope = parse_json_object(value)
    if set(envelope) != {'interface', 'payload'}:
        raise ActionValidationError('Free decision requires only interface and payload')
    selected = envelope['interface']
    if type(selected) is not str or selected not in INTERFACES:
        raise ActionValidationError('Unknown free-track interface')
    expected = str if selected == 'fingertip_code' else dict
    if type(envelope['payload']) is not expected:
        raise ActionValidationError('Code payload must be source text; action/tool payload must be an object')
    # Inner primitive validation belongs after the shared attempt accounting.
    return {'interface': selected, 'payload': envelope['payload']}


def execute_free_decision(value, session, folder, decision_index, expected_sequence, program_limits):
    if session.control_mode != 'hybrid' or session.spec.get('track') != 'free_tool_agent':
        raise ActionValidationError('Interface choice requires the independently pinned free-tool track')
    choice = parse_free_decision(value)
    selected, payload = choice['interface'], choice['payload']
    session.interface_selections[selected] += 1
    save(folder/'interface_selection.json', choice)
    program = None
    if selected in ('joints_action', 'fingertip_action'):
        mode = 'joints' if selected == 'joints_action' else 'fingertip'
        feedback = session.move(payload, folder, f'model_{decision_index:04d}',
                                expected_sequence, control_mode=mode)
    elif selected == 'fingertip_tools':
        # The session counts recognized physical intent before schema/sequence
        # rejection, while reads and unknown tools never consume an action.
        result = session.invoke_call(payload, folder=folder, decision_index=decision_index,
                                     expected_sequence=expected_sequence)
        call_name = payload['tool']
        feedback = {'status': 'tool_completed', 'call_status': 'completed',
                    'tool': call_name, 'result': result,
                    'primitive_status': result.get('feedback', {}).get('status')
                        if call_name in ('move_geometry', 'hold') else None}
    else:
        if type(expected_sequence) is not int or session.refresh()['sequence'] != expected_sequence:
            raise ActionValidationError('Free code decision uses a stale observation')
        (folder/'program.txt').write_text(payload)
        old_deadline = session.program_deadline
        session.program_deadline = (None if program_limits.max_runtime_s is None else
                                    time.monotonic() + program_limits.max_runtime_s)
        try:
            program = execute_program(payload,
                lambda name, arguments: session.invoke(name, arguments, folder=folder,
                                                       decision_index=decision_index),
                limits=program_limits)
        finally:
            session.program_deadline = old_deadline
        save(folder/'program_execution.json', program)
        feedback = program_feedback(program)
        if session.fatal_error is not None:
            raise session.fatal_error
    return {**feedback, 'selected_interface': selected}, program


FREE_SYSTEM_PROMPT = '''You control one FR3 arm and SHARPA hand from public RGB-D,
robot proprioception and the configured contact feedback. Choose an interface
for each decision. Return exactly one JSON object with keys "interface" and
"payload". The four permitted forms are:
1. {"interface":"joints_action","payload":{"action":"joint_delta","delta_rad":{"advertised_joint":0.01},"duration_s":0.5}}
2. {"interface":"fingertip_action","payload":{"action":"move_geometry","wrist_delta_m":[0,0,0.01],"duration_s":1.0}}
3. {"interface":"fingertip_tools","payload":{"tool":"observe","arguments":{}}}
4. {"interface":"fingertip_code","payload":"r = hold(duration_s=0.4)\\n"}
Action payloads can also use {"action":"hold","duration_s":0.4} or
{"action":"stop","reason":"..."}. Geometry accepts wrist_delta_m (world meters),
wrist_rotation_delta_rad (world rotation vector), and fingertip_delta_m keyed by
thumb/index/middle/ring/little (meters in right_hand_flange). Joint deltas use
advertised joint names and radians. Read the separate joint and geometry limits.
Geometry actions/tools/code may request finite positive position_tolerance_m and
orientation_tolerance_rad; the controller applies the minimum of the request
and its completion cap. Omission preserves defaults; fingertip tolerance is fixed.
Tools are observe(), point_at(camera_id,u,v), move_geometry(...), hold(duration_s)
and stop(reason). Tool payloads use exactly {"tool":name,"arguments":object}.
Code is the existing restricted Python-syntax JSON language: assignments, bounded
for/range loops, if/else and the same public tools; no imports, attributes, files,
network, private simulator access, or joint tool inside code. To issue a joint
action after code, select joints_action on your next model decision.
All choices share this episode and its remaining decision/action budgets. Every
recognized physical request counts, including invalid parameters and stale single
action/tool requests. Unknown tools, stop, read-only calls, and a whole code program
rejected before execution do not consume a primitive attempt. Observe and point_at
do not move the robot. Switching interfaces never resets the scene or budget.
Geometry, tools and code share one classical controller. Joint actions ramp their
direct targets from the last acknowledged drive target (see
capabilities.direct_joint_execution); classical IK does not apply to that primitive. After a
joint action, the server hands its acknowledged drive to the classical executor;
old geometry goals and old integral corrections must not be assumed retained.
Use feedback and fresh observations to plan steps within the advertised limits. The hand must satisfy
the public humanlike grasp contract. A completed action/tool/program does not
prove task success. Do not invent object poses, grasp or success-oracle tools.
'''
