"""Synthetic fixed calibration: native alignment gate, following, loss and restart.

No leader hardware or motor writes. Use a frozen snapshot and --sandbox.
"""
import json
import math
import time

from uni_teleop.mapping.factr.calibration import ArmCalibration
from uni_teleop.mapping.factr.retarget import JointRetargeter
from uni_teleop.devices.pico.protocol import Frame
from scripts.teleop.pico_isaac import parser, run


class SyntheticLeader:
    rejected = 0

    def __init__(self, *, joint_limit=False, practice_follow=False, unrestricted_arm=False):
        self.joint_limit=joint_limit
        self.practice_follow=practice_follow
        self.unrestricted_arm=unrestricted_arm
        self.sequence = 0
        self.mapper = None

    def read(self):
        self.sequence += 1
        seq = self.sequence
        if self.unrestricted_arm:
            if 180<seq<=210:return None
            if seq==211:self.mapper.set_active(True)
            values=[0.]*8
            if 10<seq<=180:
                values[6]=self.mapper.model.arm.active[6].lower-.12-self.mapper.arm_calibration.robot_reference_rad[6]
            return Frame('synthetic-leader',seq,time.monotonic_ns(),str(seq),'SYNTHETIC_NO_HARDWARE',
                         {'leader':values,'leader_id':'synthetic','head':None,'controllers':{},'hands':{}})
        if self.practice_follow:
            if 180<seq<=210:return None
            if seq==211:self.mapper.set_active(True)
            values=[min(.12,seq*.002),0.,3.05 if 10<seq<=100 else .2 if seq>100 else 0.,0.,0.,0.,0.,0.]
            if 10<seq<=100:
                values[2]=math.pi-self.mapper.arm_calibration.robot_reference_rad[2]
            return Frame('synthetic-leader',seq,time.monotonic_ns(),str(seq),'SYNTHETIC_NO_HARDWARE',
                         {'leader':values,'leader_id':'synthetic','head':None,'controllers':{},'hands':{}})
        if 160 < seq <= 180 or seq > 330:
            return None
        if seq == 181:
            self.mapper.set_active(True)  # Synthetic operator presses Start again.
        if seq <= 20 or 180 < seq <= 200:
            raw = .18  # Deliberately misaligned; Start must not chase this pose.
        elif seq <= 70:
            raw = 0.
        elif seq <= 160:
            raw = min(.06, (seq-70)*.001)
        elif seq <= 250:
            raw = .06
        else:
            raw = min(.10, .06+(seq-250)*.001)
        values=[raw]+[0.]*7
        if self.joint_limit:
            values[3]=-min(1.2,max(0,seq-280)*.04)
        return Frame('synthetic-leader',seq,time.monotonic_ns(),str(seq),
            'SYNTHETIC_NO_HARDWARE',{'leader':values, 'leader_id':'synthetic',
                                     'head':None,'controllers':{},'hands':{}})

    def health(self):
        return {'source':'synthetic_leader','sequence':self.sequence}

    def close(self):
        pass

    def mapper_factory(self, geometry, initial, bindings, **kwargs):
        cal=ArmCalibration.from_reference(geometry, leader_id='synthetic',
            leader_reference_rad=[0.]*7, robot_reference_rad=[initial[n] for n in geometry.arm.names],
            signs=[1]*7, directions_verified=[True]*7, source_kind='synthetic')
        self.mapper=JointRetargeter(geometry,initial,bindings,arm_calibration=cal,
            fixed_reference=self.practice_follow,follow_on_start=self.practice_follow,
            practice_follow=self.practice_follow,unrestricted_arm=self.unrestricted_arm,**kwargs)
        return self.mapper


def check(output):
    rows=[json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]
    session=json.loads((output/'session.json').read_text())
    if session['settings'].get('unrestricted_arm'):
        from bench.control.factory import public_robot_geometry, robot_configuration
        geometry=public_robot_geometry(robot_configuration(json.loads((output/'scene.json').read_text())))
        lower=geometry.arm.active[6].lower
        commanded=[r for r in rows if r['input'] and 10<r['input']['sequence']<=180]
        loss=[(i,r) for i,r in enumerate(rows) if r['input'] is None]
        result={'source_kind':'synthetic_no_hardware',
            'native_limits_removed':session['simulated_arm_motion'].get('native_verified',False),
            'direct_target':bool(commanded) and all(abs(r['action']['joint_positions_rad']['fr3_joint7']-(lower-.12))<1e-6 for r in commanded),
            'measured_crossed_original_limit':min(r['measured_joints']['fr3_joint7'] for r in rows)<lower-.04,
            'no_joint_limit_holds':not any(v['code'] in ('joint_limit','joint_saturated') for r in rows for v in r['control_issues']),
            'real_loss_holds':bool(loss) and all(not r['armed'] and r['action']==rows[i-1]['action'] for i,r in loss),
            'contacts_complete':all(r['contact_sampling']['complete'] and r['contact_sampling']['sampled_substeps']==40 for r in rows)}
        result['passed']=all(v for k,v in result.items() if k!='source_kind')
        result.update(task_success=None,physics_valid=None,formal_evaluation=False)
        (output/'master_slave_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['passed']:raise RuntimeError('Unrestricted simulated joint check failed')
        return
    if json.loads((output/'session.json').read_text())['settings'].get('practice_follow'):
        limited=[r for r in rows if any(v['code']=='joint_saturated' for v in r['control_issues'])]
        returned=[r for r in rows if r['input'] and 110<r['input']['sequence']<180]
        loss=[(i,r) for i,r in enumerate(rows) if r['input'] is None]
        result={'source_kind':'synthetic_no_hardware',
            'limited_joint_does_not_latch':bool(limited) and all(r['armed'] for r in limited),
            'other_joint_moves':bool(limited) and max(r['measured_joints']['fr3_joint1'] for r in limited)-rows[0]['measured_joints']['fr3_joint1']>.02,
            'returns_without_restart':bool(returned) and all(r['armed'] and not r['control_issues'] for r in returned),
            'real_loss_holds':bool(loss) and all(not r['armed'] and r['action']==rows[i-1]['action'] for i,r in loss),
            'contacts_complete':all(r['contact_sampling']['complete'] and r['contact_sampling']['sampled_substeps']==40 for r in rows),
            'no_guard_trigger':not any(r['safety_trigger'] for r in rows)}
        result['passed']=all(v for k,v in result.items() if k!='source_kind')
        result.update(task_success=None,physics_valid=None,formal_evaluation=False)
        (output/'master_slave_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['passed']:raise RuntimeError('Native practice following check failed')
        return
    initial=rows[0]['action']
    select=lambda low,high:[r for r in rows if r['input'] and low < r['input']['sequence'] <= high]
    first_bad=select(0,20)
    restart_bad=select(180,200)
    first_move=select(70,160)
    resumed=select(250,330)
    loss=[r for r in rows if r['input'] is None]
    holds=lambda subset:bool(subset) and all(r['action']==subset[0]['action'] for r in subset)
    enabling=[r for r in rows if r['arm_alignment']['state']=='aligned']
    enable_holds=all(i==0 or r['action']==rows[i-1]['action'] for i,r in enumerate(rows)
                     if r['arm_alignment']['state']=='aligned' and not rows[i-1]['armed'])
    q0=rows[0]['measured_joints']['fr3_joint1']
    result={'source_kind':'synthetic_no_hardware','steps':len(rows),
        'initial_misalignment_holds':bool(first_bad) and all(r['action']==initial and not r['armed'] for r in first_bad),
        'restart_misalignment_holds':holds(restart_bad) and all(not r['armed'] for r in restart_bad),
        'enabling_frames_hold':bool(enabling) and enable_holds,
        'first_motion_rad':max((abs(r['measured_joints']['fr3_joint1']-q0) for r in first_move),default=0),
        'resumed_motion_rad':max((abs(r['measured_joints']['fr3_joint1']-q0) for r in resumed),default=0),
        'holds_after_loss':bool(loss) and all(not r['armed'] and r['arm_alignment']['state']=='held' for r in loss),
        'contacts_complete':all(r['contact_sampling']['complete'] and r['contact_sampling']['sampled_substeps']==40 for r in rows),
        'safety_triggered':any(r['safety_trigger'] for r in rows),
        'task_success':None,'physics_valid':None,'formal_evaluation':False}
    # Each separate loss interval must hold its own acknowledged target.
    result['holds_after_loss'] &= all(r['action']==rows[i-1]['action'] for i,r in enumerate(rows) if r['input'] is None and i)
    result['passed']=(all(result[k] for k in ('initial_misalignment_holds','restart_misalignment_holds',
        'enabling_frames_hold','holds_after_loss','contacts_complete')) and
        result['first_motion_rad']>.04 and result['resumed_motion_rad']>.08 and not result['safety_triggered'])
    if json.loads((output/'session.json').read_text())['settings'].get('joint_limit'):
        result['joint_limit_reported']=any(issue.get('code')=='joint_limit' and issue.get('joint')=='fr3_joint4'
            for r in rows for issue in r.get('control_issues',[]))
        result['passed'] &= result['joint_limit_reported']
    (output/'master_slave_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Native fixed-calibration check failed')


if __name__=='__main__':
    cli=parser()
    cli.add_argument('--joint-limit',action='store_true',help='Inject a ramp beyond J4 limits to verify hold diagnostics')
    cli.add_argument('--practice-follow',action='store_true',help='Check per-joint saturation, continued following and real loss hold')
    cli.add_argument('--unrestricted-arm',action='store_true',help='Check direct targets and native motion beyond original arm limits')
    args=cli.parse_args()
    args.mode,args.start_active,args.profile,args.steps='controllers',True,True,220
    if args.unrestricted_arm:args.practice_follow=True
    receiver=SyntheticLeader(joint_limit=args.joint_limit,practice_follow=args.practice_follow,
                             unrestricted_arm=args.unrestricted_arm)
    run(args,receiver=receiver,source_kind='synthetic_no_hardware',
        mapper_factory=receiver.mapper_factory,on_finish=check)
