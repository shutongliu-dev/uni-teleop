"""Native fixed-joint following and dry-run virtual resistance, without hardware.

Run from a frozen snapshot with --sandbox --headless --no-video --physics-profile cpu_rigid_v1.
"""
import json
import time

import numpy as np

from uni_teleop.mapping.factr.calibration import ArmCalibration
from uni_teleop.feedback.factr.haptics import FeedbackFrame,VirtualCoupling
from uni_teleop.mapping.factr.retarget import JointRetargeter
from uni_teleop.devices.pico.protocol import Frame
from scripts.teleop.pico_isaac import parser,run


class Inputs:
    rejected=0

    def __init__(self,initialize=False,follow_on_start=False):
        self.sequence=0
        self.latest=None
        self.initialize=initialize
        self.follow_on_start=follow_on_start
        self.mapper=None
        self.feedback=None

    def read(self):
        self.sequence+=1
        if self.sequence>180:
            self.latest=None
            return None
        raw=[(.08 if self.initialize or self.follow_on_start else 0.)+min(.2,max(0,self.sequence-70)*.01)]+[0.]*7
        if self.initialize and self.sequence==10:
            self.mapper.set_active(True)
            self.feedback.request_feedback(True)
        stamp=time.monotonic_ns()
        self.latest=Frame('synthetic',self.sequence,stamp,str(stamp),'SYNTHETIC',
            {'leader':raw,'leader_id':'synthetic','head':None,'hands':{},'controllers':{}})
        return self.latest

    def health(self):return {'kind':'synthetic','sequence':self.sequence}
    def close(self):pass

    def mapper_factory(self,geometry,initial,bindings,**kwargs):
        cal=ArmCalibration.from_reference(geometry,leader_id='synthetic',leader_reference_rad=[0.]*7,
            robot_reference_rad=[initial[n] for n in geometry.arm.names],signs=[1]*7,
            directions_verified=[True]*7,source_kind='synthetic')
        def forbidden(*args,**kwargs):raise RuntimeError('Fixed joint mapping entered IK')
        geometry.solve=forbidden
        self.mapper=JointRetargeter(geometry,initial,bindings,arm_calibration=cal,fixed_reference=True,
                                   follow_on_start=self.follow_on_start,**kwargs)
        return self.mapper


class DryFeedback:
    requested=True

    def __init__(self,receiver):
        self.receiver=receiver
        self.coupling=VirtualCoupling({i:1030 if i in (2,4) else 1220 for i in range(1,8)})
        self.current=np.zeros(7)
        self.previous_ns=time.monotonic_ns()

    def request_feedback(self,enabled):self.requested=enabled
    def start_support(self,mapper):pass

    def observe(self,mapper,observation,*,now_ns):
        latest=self.receiver.latest
        if not mapper.armed or latest is None or not self.requested:
            self.current[:]=0.;self.coupling.current[:]=0.
        else:
            cal=mapper.arm_calibration
            frame=FeedbackFrame(now_ns,cal.leader_reference_rad,cal.robot_reference_rad,cal.signs,
                tuple(observation['joint_positions_rad'][n] for n in mapper.model.arm.names))
            self.current=self.coupling.command(frame,latest.tracking['leader'],[0.]*8,
                now_ns=now_ns,dt=min(.05,(now_ns-self.previous_ns)/1e9))
        self.previous_ns=now_ns

    @property
    def feedback_state(self):
        return {'kind':'synthetic_current_dry_run','current_ma':self.current.tolist(),
                'requested':self.requested,'enabled':False}


def check(output):
    events=[json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]
    rows=[r for r in events if 'action' in r]
    active=[r for r in rows if r['armed']]
    currents=[r['leader_feedback']['current_ma'] for r in rows]
    loss=[r for r in rows if r['input'] is None]
    result={'source_kind':'synthetic_no_hardware','steps':len(rows),'ik_calls':0,
        'max_resistance_ma':max(abs(c[0]) for c in currents),
        'resists_positive_leader_motion':any(c[0]<-1. for c in currents),
        'current_caps_respected':all(abs(c[i])<=(60 if i in (1,3) else 30)+1e-10 for c in currents for i in range(7)),
        'actual_arm_excursion_rad':max(abs(r['measured_joints']['fr3_joint1']-rows[0]['measured_joints']['fr3_joint1']) for r in rows),
        'loss_zeroes_feedback':bool(loss) and all(not any(r['leader_feedback']['current_ma']) for r in loss),
        'contacts_complete':all(r['contact_sampling']['complete'] and r['contact_sampling']['sampled_substeps']==40 for r in rows),
        'safety_triggered':any(r['safety_trigger'] for r in rows),'task_success':None,'physics_valid':None}
    result['passed']=(bool(active) and result['actual_arm_excursion_rad']>.1 and
        all(result[k] for k in ('resists_positive_leader_motion','current_caps_respected','loss_zeroes_feedback','contacts_complete'))
        and not result['safety_triggered'])
    initialization=next((r for r in events if r.get('event')=='leader_alignment'),None)
    session=json.loads((output/'session.json').read_text())
    if session['settings'].get('align_from_leader'):result['passed'] &= initialization is not None
    if initialization:
        reset_error=max(abs(rows[0]['measured_joints'][n]-q)
                        for n,q in initialization['arm_joint_positions_rad'].items())
        result['leader_initialization']={'reset_target_rad':initialization['arm_joint_positions_rad'],
            'first_measured_rad':rows[0]['measured_joints']['fr3_joint1'],
            'max_initial_joint_error_rad':reset_error,
            'first_frame_paused':not rows[0]['armed'],
            'saved_calibration_unchanged':initialization['arm_calibration_sha256']==
                session['input_mapping']['arm_calibration_sha256']}
        result['passed'] &= (reset_error<1e-4 and
            result['leader_initialization']['first_frame_paused'] and
            result['leader_initialization']['saved_calibration_unchanged'])
    (output/'leader_feedback_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Native dry-run feedback check failed')


if __name__=='__main__':
    cli=parser()
    cli.add_argument('--align-from-leader',action='store_true',help='Synthetic leader-first reset before a simulated Start')
    cli.add_argument('--follow-on-start',action='store_true',help='Native bounded synchronization from a displaced leader pose')
    args=cli.parse_args()
    args.mode,args.start_active,args.profile,args.steps='controllers',not args.align_from_leader,True,130
    receiver=Inputs(initialize=args.align_from_leader,follow_on_start=args.follow_on_start)
    receiver.feedback=DryFeedback(receiver)
    run(args,receiver=receiver,feedback=receiver.feedback,mapper_factory=receiver.mapper_factory,
        source_kind='synthetic_no_hardware',on_finish=check)
