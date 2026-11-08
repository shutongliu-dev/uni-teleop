"""Native drawer/contents and tray probes, explicitly outside robot task scoring."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.asarray(value).copy()


def run(args):
    source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.study_drawer_interactions',),entrypoint_file=Path(__file__))
    if source is None: raise ValueError('Use a complete frozen project source')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report={'schema':'study_drawer_object_probe.v1','passed':False,'case':args.case,
        'source_snapshot':source,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'physics_dt_s':.0005,'robot_present':False,'model_calls':0,'task_success_claimed':False,
        'hand_object_interaction_qualified':False,'physics_force_calibration_claimed':False,
        'scope':'Real constrained loaded-drawer travel/return or separately reset tray support; no robot retrieval claim.'}
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'width':640,'height':480,'multi_gpu':False,
            'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
        from scenes.workrooms.study.task_scene import candidate_inputs
        candidate,manifest,metadata,dependencies=candidate_inputs('drawer_retrieval')
        report['asset_manifest_sha256']=candidate['manifest_sha256']
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        world=World(stage_units_in_meters=1.,physics_dt=.0005,rendering_dt=1/30,
            backend='torch',device='cuda:0',physics_prim_path='/World/PhysicsScene')
        pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
        table=UsdGeom.Cube.Define(stage,'/World/Table');table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.65,0.,.74));table.AddScaleOp().Set(Gf.Vec3f(.9,2.,.08))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        stage.DefinePrim('/World/StudyTask','Xform').GetReferences().AddReference(candidate['initial_usd'],'/World/StudyTask')
        actors=manifest['bodies']
        if args.case=='tray':
            xform=UsdGeom.Xformable(stage.GetPrimAtPath(actors['notebook']))
            next(op for op in xform.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(.49,-.24,.8043))
        static=['/World/Table']+[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/StudyTask/Fixtures'))
            if p.HasAPI(UsdPhysics.CollisionAPI)]
        for path in actors.values():PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        stage.Export(str(output/'initial.usda'));world.reset();sim=world.physics_sim_view
        views={n:sim.create_rigid_body_view(p) for n,p in actors.items()}
        contact={n:sim.create_rigid_contact_view(p,filter_patterns=static+[p2 for n2,p2 in actors.items() if n2!=n],
            max_contact_data_count=8192) for n,p in actors.items()}
        report['contact_filter_paths']={n:list(v.filter_paths) for n,v in contact.items()}
        device=views['drawer'].get_transforms().device;indices=torch.tensor([0],dtype=torch.int32,device=device)
        duration=6.8 if args.case=='slide' else 1.2;rows=[]
        report['engineering_boundary']={'drawer_force_n':.6 if args.case=='slide' else 0.,
            'pose_writes_after_reset':False,'notebook_actuated':False,'case':args.case,
            'description':'0.5s settle, 2.5s bounded outward force, 0.5s release, 2.5s inward force, 0.8s release.'}
        for step in range(round(duration/.0005)):
            time=step*.0005;force=0.
            if args.case=='slide':
                force=-.6 if .5<=time<3. else .6 if 3.5<=time<6. else 0.
            if force:
                views['drawer'].apply_forces_and_torques_at_position(
                    force_data=torch.tensor([[force,0.,0.]],dtype=torch.float32,device=device),
                    torque_data=None,position_data=None,indices=indices,is_global=True)
            world.step(render=False)
            row={'native_step':step+1,'time_s':(step+1)*.0005,'drawer_force_n':force,'objects':{}}
            for name,view in views.items():
                row['objects'][name]={'pose_xyzw':array(view.get_transforms())[0].tolist(),
                    'velocity':array(view.get_velocities())[0].tolist(),
                    'contact_forces_n':array(contact[name].get_contact_force_matrix(dt=.0005))[0].tolist()}
            rows.append(row)
        with (output/'samples.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        final=rows[-1]['objects'];tail=[r for r in rows if r['time_s']>=duration-.3]
        opening=[.57-r['objects']['drawer']['pose_xyzw'][0] for r in rows]
        note_support=actors['drawer'] if args.case=='slide' else '/World/StudyTask/Fixtures/target_floor/Collider'
        filters=list(contact['notebook'].filter_paths[0]);support_index=filters.index(note_support)
        support_forces=[r['objects']['notebook']['contact_forces_n'][support_index][2] for r in tail]
        checks={
            'finite_native_states':all(np.isfinite([*o['pose_xyzw'],*o['velocity']]).all() for r in rows for o in r['objects'].values()),
            'no_ejection':all(.77<r['objects']['notebook']['pose_xyzw'][2]<1. for r in rows),
            'released_rest_300ms':all(np.linalg.norm(o['velocity'][:3])<.025 and np.linalg.norm(o['velocity'][3:])<.12 for r in tail for o in r['objects'].values()),
            'notebook_real_upward_support_300ms':all(f>1.e-7 for f in support_forces),
            'drawer_closed_within_12mm':-.004<=opening[-1]<=.012,
            'drawer_rail_height':all(abs(r['objects']['drawer']['pose_xyzw'][2]-.825)<.004 for r in rows),
        }
        if args.case=='slide':
            checks['real_opening_at_least_100mm']=max(opening)>=.1
            checks['bounded_native_joint_travel']=min(opening)>-.004 and max(opening)<.164
            checks['notebook_passively_travelled']=max(r['objects']['notebook']['pose_xyzw'][0] for r in rows)-min(r['objects']['notebook']['pose_xyzw'][0] for r in rows)>.1
        report.update(checks=checks,passed=all(checks.values()),final=final,
            maximum_opening_m=max(opening),final_opening_m=opening[-1],
            minimum_notebook_support_force_n=min(support_forces),native_steps=len(rows),duration_s=duration)
    except BaseException as error:
        traceback.print_exc();report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        (output/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'passed':report['passed'],'output':str(output)}),flush=True)
        if app is not None:app.close()
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('slide','tray'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__=='__main__':main()
