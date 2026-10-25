"""Synthetic PICO packets through UDP -> real Isaac, including tracking loss.

No headset is involved. This validates the installed operator control path,
not hardware tracking quality, task success, or formal benchmark qualification.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time

import numpy as np

from uni_teleop.devices.pico.protocol import Receiver, SCHEMA, boot_id
from uni_teleop.mapping.franka.retarget import FINGERS
from bench.control.factory import public_robot_geometry, robot_configuration
from scripts.teleop.pico_isaac import load_inputs, parser, run


class SyntheticReceiver:
    def __init__(self, args, *, stress=False, brief_loss=False, partial_hand_loss=False):
        scene, initial, _ = load_inputs(args)
        model = public_robot_geometry(robot_configuration(scene))
        q = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in model.joints}
        q.update(zip(model.arm.names, scene['robot']['arm_home_rad']))
        q.update(initial['joint_positions_rad'])
        geometry = model.public_finger_geometry(q)
        self.joints = np.zeros((26, 7))
        self.joints[:, 6] = 1.
        for finger, indices in FINGERS.items():
            self.joints[list(indices), :3] = list(geometry['finger_joint_centers_m'][finger].values())+[
                geometry['fingertip_positions_m'][finger]]
        self.mode, self.sequence = args.mode, 0
        self.stress, self.brief_loss = stress, brief_loss
        self.partial_hand_loss = partial_hand_loss
        self.loss_until = None
        self.receiver = Receiver(port=0)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clock = boot_id()

    @property
    def rejected(self):
        return self.receiver.rejected

    def health(self):
        return self.receiver.health()

    def read(self):
        self.sequence += 1
        if self.brief_loss:
            if 16 <= self.sequence <= 18:
                return None
        else:
            if self.sequence == 16:
                self.loss_until = time.monotonic()+(2.2 if self.mode == 'hands' else .4)
            if self.loss_until is not None and time.monotonic() < self.loss_until:
                return None  # Longer than the selected input mode recovery window.
        amount = min(.002, self.sequence*.0002)
        if self.brief_loss and self.sequence > 20:
            amount += min(.002, (self.sequence-20)*.0002)
        stress = self.stress and 8 <= self.sequence <= 15
        if stress:
            amount = .8  # Abrupt unreachable target; never alter robot/physics state.
        value = {'timeStampNs': self.sequence, 'Head': {'pose': '0,0,0,0,0,0,1'}}
        if self.mode == 'controllers':
            value['Controller'] = {'right': {'pose': f'{amount},0,0,0,0,0,1', 'trigger': .4, 'grip': 1.}}
        else:
            joints = self.joints.copy()
            joints[10, 0] += amount
            if self.partial_hand_loss:
                joints[1, 0] += amount
                joints[15, 0] += amount
            if stress:
                joints[1, 0] += amount
            value['Hand'] = {'rightHand': {'isActive': 1, 'HandJointLocations': [
                {'p': ','.join(map(str, row))} for row in joints]}}
            if self.partial_hand_loss and 8 <= self.sequence <= 15:
                value['Hand']['rightHand']['HandJointLocations'][10]['s'] = 0
                value['Hand']['leftHand'] = {'isActive': 1, 'HandJointLocations': []}
                value['Controller'] = {'left': {'pose': 'invalid'}}
                value['Head'] = {'pose': 'invalid'}
        if self.partial_hand_loss and self.sequence <= 3:
            value['Head'] = {'pose': 'invalid'}  # Head may become available after wrist calibration.
        payload = {'schema': SCHEMA, 'boot_id': self.clock, 'stream': 'synthetic', 'sequence': self.sequence,
                   'host_ns': time.monotonic_ns(), 'device': 'SYNTHETIC_NO_HEADSET', 'value': value}
        self.sender.sendto(json.dumps(payload).encode(), self.receiver.socket.getsockname())
        deadline = time.monotonic()+.1
        while time.monotonic() < deadline:
            frame = self.receiver.read()
            if frame is not None and frame.sequence == self.sequence:
                return frame
            time.sleep(.001)
        raise RuntimeError('Synthetic UDP sample did not reach the receiver')

    def close(self):
        self.sender.close()
        self.receiver.close()


def check(output, mode, *, stress=False, brief_loss=False, partial_hand_loss=False):
    rows = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    initial = rows[0]['action']['joint_positions_rad']
    targets_changed = any(row['action']['joint_positions_rad'] != initial for row in rows)
    moved = max(abs(row['measured_joints'][n]-rows[0]['measured_joints'][n]) for row in rows for n in initial)
    loss = next(i for i, row in enumerate(rows) if row['input'] is None)
    held = rows[loss]['action']['joint_positions_rad']
    if brief_loss:
        recovered = next(i for i in range(loss+1, len(rows)) if rows[i]['input'] is not None and rows[i]['armed'])
        held_after_loss = all(row['action']['joint_positions_rad'] == held for row in rows[loss:recovered+1])
        recovery_ok = all(row['armed'] for row in rows[recovered:]) and any(
            row['action']['joint_positions_rad'] != held for row in rows[recovered+1:])
    else:
        disarmed = next((i for i in range(loss, len(rows)) if not rows[i]['armed']), None)
        held_after_loss = all(row['action']['joint_positions_rad'] == held for row in rows[loss:])
        recovery_ok = disarmed is not None and all(not row['armed'] for row in rows[disarmed:]) and any(
            row['input'] is not None for row in rows[disarmed:])
    from PIL import Image
    left, right = (np.asarray(Image.open(output/name)) for name in ('left.png','right.png'))
    result = {'source_kind': 'synthetic_udp_no_hardware', 'mode': mode,
              'targets_changed': targets_changed, 'max_measured_change_rad': moved,
              'held_after_tracking_loss': held_after_loss, 'steps': len(rows),
              'loss_kind': 'brief' if brief_loss else 'long', 'recovery_policy_observed': recovery_ok,
              'stereo_nonblank': bool(left.std() > 1 and right.std() > 1),
              'stereo_disparity': bool(np.mean(np.abs(left.astype(float)-right)) > .1)}
    result['passed'] = bool(targets_changed and moved > 1e-5 and held_after_loss and recovery_ok
                            and result['stereo_nonblank'] and result['stereo_disparity'])
    if partial_hand_loss:
        partial = [i for i, row in enumerate(rows) if row['input'] is not None
                   and row['input']['tracking']['hands']['right'][10] is None]
        index_names = [name for name in initial if 'index' in name]
        result['partial_hand_frames'] = len(partial)
        result['finger_loss_kept_arm_active'] = bool(partial) and all(rows[i]['armed'] for i in partial)
        result['invalid_finger_held'] = bool(index_names and partial) and all(
            rows[i]['action']['joint_positions_rad'][n] == rows[i-1]['action']['joint_positions_rad'][n]
            for i in partial for n in index_names)
        result['passed'] &= result['finger_loss_kept_arm_active'] and result['invalid_finger_held']
    if stress:
        result['max_mapping_s'] = max(row['timing']['mapping_s'] for row in rows)
        result['ik_budget_observed'] = any('IK budget exhausted' in row['status'] for row in rows)
        result['armed_before_injected_loss'] = all(row['armed'] for row in rows[:loss])
        result['passed'] &= (result['max_mapping_s'] < .15 and result['ik_budget_observed']
                             and result['armed_before_injected_loss'])
    (output/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Synthetic PICO native check failed')


def main():
    options = parser()
    cases = options.add_mutually_exclusive_group()
    cases.add_argument('--brief-loss', action='store_true', help='Verify brief loss holds targets, reanchors and resumes on fresh input')
    cases.add_argument('--stress', action='store_true', help='Inject unreachable target jumps and verify mapping stays below the input freshness window')
    cases.add_argument('--partial-hand-loss', action='store_true',
                       help='Inject invalid finger, head and unselected-hand data before a full tracking loss')
    args = options.parse_args()
    if args.partial_hand_loss and args.mode != 'hands':
        options.error('--partial-hand-loss requires --mode hands')
    if args.stress:
        args.profile = True
    args.start_active = True
    args.steps = args.steps or (140 if args.mode == 'hands' else 40)
    run(args, receiver=SyntheticReceiver(args, stress=args.stress, brief_loss=args.brief_loss,
                                       partial_hand_loss=args.partial_hand_loss), source_kind='synthetic_udp_no_hardware',
        on_finish=lambda output: check(output, args.mode, stress=args.stress, brief_loss=args.brief_loss,
                                      partial_hand_loss=args.partial_hand_loss))


if __name__ == '__main__':
    raise SystemExit(main())
