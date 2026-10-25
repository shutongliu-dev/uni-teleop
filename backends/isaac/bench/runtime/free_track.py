"""Explicit free-track discovery and serial native-ACK reference handoff.

This does not execute native requests or accept model-provided ACKs. Only the
trusted EpisodeSession calls acknowledge after its corresponding step reply.
"""
from __future__ import annotations

from copy import deepcopy
import math

from bench.protocol.schemas import RobotJointCommand
from bench.protocol.validation import validate_joint_command


def supports_free_track(policy):
    descriptor=getattr(policy,'descriptor',None)
    return bool(descriptor is not None and descriptor.policy_id=='classical_fingertip'
                and descriptor.direct_control_handoff is not None
                and all(callable(getattr(policy,name,None)) for name in (
                    'check_direct_handoff','adopt_direct_reference','commit_pending_step','discard_pending_step')))


def advertise_free_track(catalog, descriptors):
    result=deepcopy(catalog)
    policies=[p.model_dump(mode='json') for p in descriptors
              if p.policy_id=='classical_fingertip' and p.direct_control_handoff is not None]
    result['interfaces'].append({
        'id':'free_tool_agent','native_track':'free_tool_agent','control_mode':'hybrid',
        'interaction_mode':'free','available':bool(policies),
        'executor':'classical_fingertip_with_direct_joint_bypass',
        'interfaces':['joints_action','fingertip_action','fingertip_tools','fingertip_code'],
        'action_kinds':['direct_joints','system0'],
        'model_transport':'JSON text parsed by the benchmark client; no provider-native function calling required',
        'client_command':'python -m agent.model_client.runner --control-mode hybrid --interaction-mode free',
        'policies':policies,
        'handoff_protocol':'acknowledged_direct_handoff_v1',
        'direct_control_scope':'Direct joint targets bypass IK and classical Q/V/A smoothing. Classical reentry requires verified ACK drive/velocity and original braking feasibility.',
        'budget_scope':'One episode; all interfaces share primitive, model, simulation and tool budgets.',
    })
    result['comparison']['free_tool_agent']={
        'independent_track':True,'native_track':'free_tool_agent',
        'mixed_interface_episode_supported':True,
        'condition':'Free selection is a separate evaluation track; the four fixed interface conditions retain their original meanings.'}
    return result


class NativeAcknowledgementError(RuntimeError):
    """The native reply does not establish the next physical control tick."""


class AcknowledgedDriveLedger:
    """One serial episode's command ledger; raw measured q is never a drive ACK."""
    def __init__(self, policy, capabilities, *, sequence, sim_time_s):
        if not supports_free_track(policy):
            raise ValueError('Policy lacks the explicit free-track handoff')
        if type(sequence) is not int or sequence < 0 or not math.isfinite(sim_time_s):
            raise ValueError('Invalid initial native clock')
        self.policy=policy
        self.capabilities=capabilities.model_copy(deep=True)
        self.sequence=sequence
        self.sim_time_s=float(sim_time_s)
        self.command=None
        self.kind=None

    def before_direct(self):
        self.policy.check_direct_handoff()

    def acknowledge(self, command, *, kind, sequence, sim_time_s):
        command=RobotJointCommand.model_validate(command.model_dump(mode='json'))
        validate_joint_command(command,self.capabilities)
        if kind not in ('direct_joints','system0'):
            raise ValueError('Unknown ACK control kind')
        if (type(sequence) is not int or sequence != self.sequence+1
                or isinstance(sim_time_s,bool) or not isinstance(sim_time_s,(int,float))
                or not math.isfinite(sim_time_s)
                or abs(float(sim_time_s)-self.sim_time_s-self.capabilities.control_period_s)>1e-8):
            raise NativeAcknowledgementError('Native ACK must advance exactly one consecutive control period')
        velocity=None if self.command is None else {
            name:(value-self.command.joint_positions_rad[name])/self.capabilities.control_period_s
            for name,value in command.joint_positions_rad.items()}
        if kind=='direct_joints':
            self.policy.adopt_direct_reference(command,velocity_rad_s=velocity,
                observation_sequence=sequence,sim_time_s=float(sim_time_s))
        else:
            if self.policy._pending_step_state is None:
                raise ValueError('Classical ACK has no pending policy command')
            expected=dict(zip(self.policy.geometry.names,self.policy.previous.tolist()))
            if command.joint_positions_rad!=expected:
                raise ValueError('Classical ACK differs from the pending issued command')
            self.policy.commit_pending_step()
        receipt={'protocol':'acknowledged_direct_handoff_v1','sequence':sequence,
            'sim_time_s':float(sim_time_s),'kind':kind,'previous_kind':self.kind,
            'command':command.model_dump(mode='json'),
            'direct_reference_velocity_known':velocity is not None if kind=='direct_joints' else True,
            'direct_qva_smoothing':False if kind=='direct_joints' else None}
        self.command=command
        self.kind=kind
        self.sequence=sequence
        self.sim_time_s=float(sim_time_s)
        return receipt
