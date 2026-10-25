"""Synthetic FACTR/MANUS native motion and independent watchdog validation.

Run via Isaac on a frozen snapshot, with --sandbox --no-video --headless.
No actual motors, glove SDK or object pose commands are used.
"""
import json
import time

import numpy as np

from uni_teleop.devices.manus.sources import host_clock_id
from uni_teleop.mapping.factr.calibration import ArmCalibration
from uni_teleop.mapping.factr_manus import ManusJointRetargeter, SEMANTICS
from uni_teleop.devices.pico.protocol import Frame
from uni_teleop.mapping.franka.retarget import FINGERS
from uni_teleop.mapping.franka.controller_hand import ControllerHand
from bench.control.factory import public_robot_geometry, robot_configuration
from scripts.teleop.pico_isaac import parser, load_inputs, run


class SyntheticInputs:
    rejected = 0

    def __init__(self, args):
        scene, initial, _ = load_inputs(args)
        self.model = public_robot_geometry(robot_configuration(scene))
        self.q = {n: initial['joint_positions_rad'].get(n, 0.) for n in self.model.names}
        self.preset = ControllerHand(self.model)
        self.sequence = 0
        self.official=json.loads(args.official_fixture.read_text()) if args.official_fixture else None

    def read(self):
        self.sequence += 1
        seq = self.sequence
        now = time.monotonic_ns()
        grip = min(.8, max(0, seq-20)*.008)
        q = self.q | self.preset.targets(grip)
        geometry = self.model.public_finger_geometry(q)
        points = np.zeros((26,3))
        for finger, indices in FINGERS.items():
            points[list(indices)] = list(geometry['finger_joint_centers_m'][finger].values())+[
                geometry['fingertip_positions_m'][finger]]
        nodes = []
        for i in (0, *range(2,26)):
            nodes.append({'node_id':str(i), 'parent_id':'0' if i else None,
                'semantic':SEMANTICS[i], 'pose':{'position':points[i].tolist(),
                'quaternion_wxyz':[1.,0.,0.,0.]}})
        # Glove is lost first; later the leader also drops out.
        hand_stamp = now if seq < 170 else now-300_000_000
        arm_stamp = now if seq < 230 else now-300_000_000
        hand = {'protocol':'manus_acquisition_v1','stream_id':'synthetic','sequence':seq,
            'clock_id':host_clock_id(),'capture_time_ns':hand_stamp,
            'device':{'device_id':'7','product':'synthetic','firmware':'none','sdk_version':'none',
                      'core_version':'none','hand':'right','source_kind':'synthetic'},
            'calibration_id':'test','topology_id':'synthetic_semantic','frame_id':'manus_world',
            'glove_tracking_valid':True,'nodes':nodes}
        raw = [min(.06, max(0,seq-10)*.0004)]+[0.]*7
        target=None
        if self.official:
            opened=self.official['poses']['open']['candidate']['joint_positions_rad']
            closed=self.official['poses']['fist']['candidate']['joint_positions_rad']
            amount=grip/.8
            target=dict({k:hand[k] for k in ('stream_id','sequence','capture_time_ns','clock_id',
                'calibration_id','topology_id')},device_id='7',protocol='sharpa_optimizer_v1',
                executable=True,joint_positions_rad={n:opened[n]+amount*(closed[n]-opened[n]) for n in opened})
        return Frame('synthetic',seq,now,'','SYNTHETIC_NO_HARDWARE',{'leader':raw,
            'leader_id':'synthetic', 'leader_host_ns':arm_stamp,'manus':hand,'manus_retarget':target,'head':None,'hands':{},'controllers':{}})

    def health(self):
        return {'source':'synthetic_no_hardware','sequence':self.sequence}

    def close(self):
        pass


def check(output):
    rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    q0=rows[0]['measured_joints']
    arm=max(abs(row['measured_joints']['fr3_joint1']-q0['fr3_joint1']) for row in rows)
    hand_only=json.loads((output/'session.json').read_text())['input_mapping']['arm_source']=='held'
    model=public_robot_geometry(robot_configuration(json.loads((output/'scene.json').read_text())))
    hand={finger:max(abs(row['measured_joints'][j.name]-q0[j.name]) for row in rows
        for j in model.finger_joints[finger]) for finger in FINGERS}
    # Use stream timestamps as the fault-injection boundary, not assumed loop speed.
    glove_lost=[r for r in rows if r['input']['tracking']['manus']['sequence']>=170]
    leader_lost=[r for r in rows if r['input']['tracking']['manus']['sequence']>=230]
    arm_names={f'fr3_joint{i}' for i in range(1,8)}
    fixed=lambda subset,names:bool(subset) and all(all(r['action']['joint_positions_rad'][n]==subset[0]['action']['joint_positions_rad'][n]
        for n in names) for r in subset)
    result={'source_kind':'synthetic_no_hardware','steps':len(rows),'arm_excursion_rad':arm,'hand_only':hand_only,
        'finger_excursion_rad':hand,'glove_loss_holds_fingers':fixed(glove_lost,set(q0)-arm_names),
        'leader_loss_holds_arm':fixed(leader_lost,arm_names),
        'contacts_complete':all(r['contact_sampling']['complete'] and
            r['contact_sampling']['sampled_substeps']==40 for r in rows),
        'safety_triggered':any(r['safety_trigger'] for r in rows),
        'task_success':None,'physics_valid':None,'formal_evaluation':False}
    result['passed']=((arm<.0001 if hand_only else arm>.02) and all(v>.01 for v in hand.values()) and result['glove_loss_holds_fingers']
        and result['leader_loss_holds_arm'] and result['contacts_complete'] and not result['safety_triggered'])
    (output/'master_slave_manus_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:
        raise RuntimeError('Native FACTR/MANUS synthetic check failed')


if __name__=='__main__':
    cli=parser()
    from pathlib import Path
    cli.add_argument('--official-fixture',type=Path,help='Offline official target fixture; synthetic input only, no SDK/devices')
    cli.add_argument('--manus-hand-only', action='store_true')
    cli.add_argument('--manus-finger-speed', type=float, default=4.)
    args=cli.parse_args()
    args.mode,args.start_active,args.profile,args.steps='controllers',True,True,150
    receiver=SyntheticInputs(args)
    def mapper_factory(geometry, initial, bindings, **kwargs):
        cal=None if args.manus_hand_only else ArmCalibration.from_reference(geometry,
            leader_id='synthetic', leader_reference_rad=[0.]*7,
            robot_reference_rad=[initial[n] for n in geometry.arm.names], signs=[1]*7,
            directions_verified=[True]*7, source_kind='synthetic')
        return ManusJointRetargeter(geometry,initial,bindings,glove_id=7,
            calibration_id='test',arm_calibration=cal,finger_speed=args.manus_finger_speed,
            hand_only=args.manus_hand_only,
            optimizer_metadata=receiver.official['metadata'] if receiver.official else None,**kwargs)
    run(args,receiver=receiver,source_kind='synthetic_no_hardware',on_finish=check,
        mapper_factory=mapper_factory)
