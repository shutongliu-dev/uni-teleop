"""Native synthetic PICO wrist + MANUS fingers and independent loss checks.

Uses the existing synthetic glove/official fixture generator, without SDK/devices.
Run in a frozen snapshot, with --sandbox --headless --no-video.
"""
from dataclasses import replace
from pathlib import Path
import json

from bench.adapters.manus.pico_teleop import PicoManusRetargeter
from bench.control.factory import public_robot_geometry, robot_configuration
from scripts.validation.master_slave_manus_smoke import SyntheticInputs
from scripts.teleop.pico_isaac import parser, run


class Inputs(SyntheticInputs):
    def read(self):
        frame=super().read()
        seq=self.sequence
        return replace(frame,tracking=frame.tracking | {
            'pico_present':True,'pico_host_ns':frame.tracking['leader_host_ns'],
            'controllers':{'right':{'pose':[min(.03,max(0,seq-20)*.00015),0.,0.,0.,0.,0.,1.],
                                    'grip':1.,'trigger':1.}}})


def check(output):
    rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    scene=json.loads((output/'scene.json').read_text())
    model=public_robot_geometry(robot_configuration(scene))
    initial=rows[0]['measured_joints']
    moved=lambda names:max(abs(row['measured_joints'][n]-initial[n]) for row in rows for n in names)
    def held_after(sequence,names):
        selected=[r['action']['joint_positions_rad'] for r in rows if r['input']['sequence']>=sequence]
        return bool(selected) and all(all(q[n]==selected[0][n] for n in names) for q in selected)
    result={'source_kind':'synthetic_no_hardware','steps':len(rows),
        'arm_excursion_rad':moved(model.arm.names),
        'finger_excursions_rad':{f:moved([j.name for j in model.finger_joints[f]]) for f in model.finger_joints},
        'glove_loss_holds_fingers':held_after(170,model.hand.names),
        'pico_loss_holds_arm':held_after(230,model.arm.names),
        'contacts_complete':all(r['contact_sampling']['complete'] and
            r['contact_sampling']['sampled_substeps']==40 for r in rows),
        'safety_triggered':any(r['safety_trigger'] for r in rows),
        'task_success':None,'physics_valid':None,'formal_evaluation':False}
    result['passed']=(result['arm_excursion_rad']>.01 and
        all(v>.01 for v in result['finger_excursions_rad'].values()) and
        all(result[k] for k in ('glove_loss_holds_fingers','pico_loss_holds_arm','contacts_complete'))
        and not result['safety_triggered'])
    (output/'pico_manus_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Native PICO+MANUS check failed')


if __name__=='__main__':
    cli=parser()
    cli.add_argument('--official-fixture',type=Path)
    args=cli.parse_args()
    args.mode,args.start_active,args.profile,args.steps='controllers',True,True,150
    receiver=Inputs(args)
    def mapper_factory(geometry,initial,bindings,**kwargs):
        return PicoManusRetargeter(geometry,initial,bindings,glove_id=7,calibration_id='test',
            optimizer_metadata=receiver.official['metadata'] if receiver.official else None,**kwargs)
    run(args,receiver=receiver,mapper_factory=mapper_factory,source_kind='synthetic_no_hardware',on_finish=check)
