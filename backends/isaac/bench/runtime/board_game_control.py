"""Turn-scoped classical control with an exact fixed drive for the idle player.

The selected side retains the existing classical executor and ACK protocol.
The other executor is never stepped: measured drift cannot create an unowned
tracking correction. This adapter has no object state or game strategy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import CONTROL_DT
from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import RobotExecutionState, RobotJointCommand, System0Action

VERSION = '1.2.0-board1'


class BoardGameFingertipPolicy(BimanualFingertipPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provenance = {**self.provenance,
            'parent_descriptor': self.descriptor.model_dump(mode='json'),
            'composition': 'Only the authenticated player executor is stepped; every inactive joint retains its last acknowledged drive target.',
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        digest = hashlib.sha256(json.dumps(self.provenance, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.descriptor = self.descriptor.model_copy(update={
            'version': VERSION, 'implementation_sha256': digest})
        self._player_hand = None
        self._acknowledged_targets = None
        self._pending_command = None
        self._moving = set()

    def reset(self, spec, capabilities):
        super().reset(spec, capabilities)
        self._player_hand = None
        self._acknowledged_targets = None
        self._pending_command = None
        self._moving = set()
        self._is_hold = False

    def set_player_hand(self, hand, acknowledged_targets):
        """Trusted runner binding before begin; inputs are robot-only native ACKs."""
        if hand not in self.sides:
            raise ValueError('Unknown player hand')
        if self._active is not None or self._pending_command is not None:
            raise ValueError('Cannot switch players during an active or unacknowledged action')
        if set(acknowledged_targets) != set(self.geometry.names):
            raise ValueError('Player binding requires the complete acknowledged joint inventory')
        checked = {}
        for joint in self.geometry.joints:
            value = acknowledged_targets[joint.name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not joint.lower <= value <= joint.upper):
                raise ValueError('Acknowledged drive target violates native joint limits')
            checked[joint.name] = float(value)
        own = self.sides[hand]
        if own.previous is not None and not np.allclose(own.previous,
                [checked[name] for name in own.geometry.names], rtol=0., atol=1.e-7):
            raise ValueError('Player executor reference differs from its acknowledged drive')
        self._player_hand, self._acknowledged_targets = hand, checked

    def begin(self, action, observation):
        if self._player_hand is None or self._acknowledged_targets is None:
            raise ValueError('Bind the authenticated player before beginning an action')
        if self._active is not None or self._pending_command is not None:
            raise ValueError('Cancel or finish the active action before beginning another')
        side = self._player_hand
        if action.skill == 'hold':
            if action.geometry is not None or action.hand_geometries is not None:
                raise ValueError('Hold cannot select geometry')
            single = System0Action(action_id=action.action_id,
                observation_sequence=action.observation_sequence, skill='hold',
                timeout_sim_s=action.timeout_sim_s, hold_duration_s=action.hold_duration_s,
                hold_mode=action.hold_mode)
        elif action.skill == 'move_geometry':
            if action.geometry is not None or set(action.hand_geometries or {}) != {side}:
                raise ValueError('Geometry must select exactly the authenticated player hand')
            single = System0Action(action_id=action.action_id,
                observation_sequence=action.observation_sequence, skill='move_geometry',
                timeout_sim_s=action.timeout_sim_s, geometry=action.hand_geometries[side])
        else:
            raise ValueError('Unsupported board player skill')
        own = self.sides[side]
        saved = self._state(own)
        try:
            if own.previous is None:
                # Before this side's first action its drive has only been held
                # at reset targets. Start from that ACK, not measured drift.
                own.previous = np.array([self._acknowledged_targets[n] for n in own.geometry.names])
                own.velocity = np.zeros(len(own.geometry.names))
            own.begin(single, self._side_observation(observation, side))
        except BaseException:
            self._restore_state(own, saved)
            raise
        self._active = {side: True}
        self._moving = set() if action.skill == 'hold' else {side}
        self._is_hold = action.skill == 'hold'

    def _feedback(self, observation, own_state):
        side = self._player_hand
        measured = observation.proprioception.joint_positions_rad
        states = {side: own_state}
        for other in self.sides:
            if other != side:
                states[other] = RobotExecutionState(protocol='robot_execution_state_v1',
                    sim_time_s=float(observation.sim_time_s), completion_scope='timed_reference_maintenance',
                    objective_status='not_checked', reference_stationary=True,
                    maintained_joint_error_rad={name: abs(float(measured[name]) - self._acknowledged_targets[name])
                        for name in self.geometry.sides[other].names})
        return self._merged_state(observation, states)

    def step(self, observation):
        if self._active is None:
            raise ValueError('No active player action')
        side = self._player_hand
        result = self.sides[side].step(self._side_observation(observation, side))
        state = self._feedback(observation, result.robot_execution_state)
        if result.status != 'running':
            self._active = None
            return result.model_copy(update={'robot_execution_state': state})
        command = self._acknowledged_targets | result.command.joint_positions_rad
        self._pending_command = dict(command)
        return PolicyStep(status='running', command=RobotJointCommand(duration_s=CONTROL_DT,
            joint_positions_rad=command), robot_execution_state=state)

    def commit_pending_step(self):
        if self._pending_command is None:
            return
        self.sides[self._player_hand].commit_pending_step()
        self._acknowledged_targets = self._pending_command
        self._pending_command = None
        # This ACK proves the inactive drive was constant for one full control
        # period. Clear only its discrete drive velocity, never its goal/PI.
        for other, peer in self.sides.items():
            if other != self._player_hand and peer.previous is not None:
                peer.velocity = np.zeros(len(peer.geometry.names))

    def discard_pending_step(self):
        if self._player_hand is not None:
            self.sides[self._player_hand].discard_pending_step()
        self._pending_command = None

    def cancel(self, reason):
        self.discard_pending_step()
        if self._player_hand is not None:
            self.sides[self._player_hand].cancel(reason)
        self._active = None

    def execution_state(self, observation):
        if self._player_hand is None:
            raise ValueError('No accepted player action has execution state')
        own = self.sides[self._player_hand]
        return self._feedback(observation,
            own.execution_state(self._side_observation(observation, self._player_hand)))
