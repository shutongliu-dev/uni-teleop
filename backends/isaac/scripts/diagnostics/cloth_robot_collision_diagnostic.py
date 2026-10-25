"""Independent diagnostic; --execute explicitly enables native Isaac.

Without --execute, inspect configuration only. --audit and --compare are CPU-only.
No Agent score, original episode, shared asset or project runtime is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from bench.evaluation.cloth.cloth_robot_collision import collect_colliders,audit_cloth_robot,digest_json,state_digest,reanalyse_export
from bench.isaacsim.release import task_inputs


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return load_json(Path(path))


def load_episode(root):
    root=Path(root).resolve()
    manifest=read(root/'artifact_hashes.json')
    required={'metadata.json','colliders.json','states.npz','triangles.npy','plan.json','result.json','actions.json','observations.jsonl'}
    if not required<=set(manifest):raise ValueError('Diagnostic manifest omits required inputs')
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:
            raise ValueError('Diagnostic artifact identity mismatch: '+name)
    with np.load(root/'states.npz',allow_pickle=False) as arrays:
        states={k:arrays[k] for k in arrays.files}
    result=read(root/'result.json');plan=read(root/'plan.json')
    if result.get('completed') is not True or result.get('sources_unchanged') is not True:
        raise ValueError('Diagnostic execution incomplete or sources changed')
    for source,digest in plan['backend_sources_sha256'].items():
        snapshot=root/'source_snapshot'/Path(source).relative_to(PROJECT)
        if str(snapshot.relative_to(root)) not in manifest or sha(snapshot)!=digest:
            raise ValueError('Frozen diagnostic source mismatch')
    actions=read(root/'actions.json')
    if len(states['time_s'])!=len(actions)+1 or not np.allclose(np.diff(states['time_s']),.02,atol=1e-8,rtol=0):
        raise ValueError('Diagnostic action and native time counts disagree')
    export=read(root/'colliders.json')
    export_frame=plan['warmup_steps']
    if type(export_frame) is not int or export_frame<0 or export_frame>=len(states['time_s']):
        raise ValueError('Export frame is not a recorded diagnostic state')
    actual_state_hash=state_digest({key:values[export_frame] for key,values in states.items()})
    if export.get('state_hash_before')!=actual_state_hash or export.get('state_hash_after')!=actual_state_hash:
        raise ValueError('Collider query state does not match the recorded export frame')
    if digest_json(actions[export_frame:])!=digest_json(plan['actions']):
        raise ValueError('Executed motion commands do not match the frozen plan')
    return read(root/'metadata.json'),export,states,np.load(root/'triangles.npy',allow_pickle=False)


def inspect(root):
    metadata,export,states,triangles=load_episode(root)
    return audit_cloth_robot(export,states,triangles,metadata['body_names'])


def compare(a,b,*,minimum_response_m=.005,maximum_hover_drift_m=.001,noise_multiplier=5.):
    ma,ea,sa,ta=load_episode(a);mb,eb,sb,tb=load_episode(b)
    pa,pb=read(Path(a)/'plan.json'),read(Path(b)/'plan.json')
    if pa['condition']!='contact' or pb['condition']!='hover':
        raise ValueError('Explicit contact and hover labels required')
    for key in ('scene_sha256','release_manifest_sha256','task_sha256','backend_sources_sha256'):
        if pa[key]!=pb[key]:raise ValueError('Unmatched pair pin: '+key)
    if not np.array_equal(ta,tb) or not np.array_equal(sa['time_s'],sb['time_s']):
        raise ValueError('Pair topology and native time samples must match exactly')
    qa,qb=sa['particle_q'],sb['particle_q']
    if qa.shape!=qb.shape or not np.allclose(qa[0],qb[0],atol=1e-7,rtol=0):
        raise ValueError('Pair cloth initial state differs')
    aa=audit_cloth_robot(ea,sa,ta,ma['body_names']);ab=audit_cloth_robot(eb,sb,tb,mb['body_names'])
    delta_a=qa[-1,:,:2]-qa[0,:,:2];delta_b=qb[-1,:,:2]-qb[0,:,:2]
    response=float(np.median(np.linalg.norm(delta_a-delta_b,axis=1)))
    hover=float(np.max(np.linalg.norm(qb[:,:,:2]-qb[0,:,:2],axis=2)))
    numerical=response>=minimum_response_m and hover<=maximum_hover_drift_m and response>=noise_multiplier*hover
    return {'schema':'cloth_contact_hover_response_diagnostic_v1','contact':str(Path(a).resolve()),'hover':str(Path(b).resolve()),
        'input_manifest_sha256':{'contact':sha(Path(a)/'artifact_hashes.json'),'hover':sha(Path(b)/'artifact_hashes.json')},
        'paired_median_extra_xy_displacement_m':response,'maximum_hover_xy_drift_m':hover,
        'minimum_response_m':minimum_response_m,'maximum_hover_drift_m':maximum_hover_drift_m,
        'noise_multiplier':noise_multiplier,'numerical_response_present':bool(numerical),
        'robot_triangle_audit':{'contact':aa,'hover':ab},'active_cloth_manipulation_accepted':False,
        'remaining_gates':['Contact/hover matched commanded trajectory and clearance verification',
            'Response onset follows actual pad approach; initial settling excluded with preregistered phase boundaries',
            'All existing topology/strain/self-intersection/task-solid/rigid physics checks',
            'Native repeatability and independent review; no cloth force stream available'],
        'scope':'CPU paired response summary, not a force measurement, public Agent success or calibrated textile benchmark'}


def execute(args,plan,scene,initial,commands):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'plan.json',plan);dump(out/'scene.json',scene)
    source_dir=out/'source_snapshot'
    for path,digest in plan['backend_sources_sha256'].items():
        src=Path(path);dest=source_dir/src.relative_to(PROJECT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
        if sha(dest)!=digest:raise ValueError('Source changed while preparing diagnostic')
    states=[];observations=[];executed=[];env=None;error=None;completed=False
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env=FR3DexterousIsaacEnv(scene,device=args.device,render_width=640,render_height=360)
        observation=env.reset(seed=args.seed,initial_state=initial)
        states.append(env.capture_state());observations.append(observation)
        dump(out/'metadata.json',env.metadata())
        np.save(out/'triangles.npy',env.triangles)
        hold={'joint_positions_rad':{n:float(v) for n,v in zip(env.joint_names,states[0]['joint_target_q'])
                                     if n in env.actuated_joint_names},'duration_s':.02}
        for _ in range(args.warmup_steps):
            observation=env.step(hold);executed.append(deepcopy(hold))
            states.append(env.capture_state());observations.append(observation)
        export=collect_colliders(env);dump(out/'colliders.json',export)
        if commands and export['status']!='complete':
            raise RuntimeError('Cooked collider export unchecked; refusing contact motion')
        for index,command in enumerate(commands):
            observation=env.step(command);executed.append(deepcopy(command))
            state=env.capture_state();states.append(state);observations.append(observation)
            if not observation['metrics']['state_finite'] or any(not np.isfinite(v).all() for v in state.values()):
                raise RuntimeError('Nonfinite native diagnostic state')
            if observation['contacts'].get('errors'):raise RuntimeError('Native contact reporting error')
            if index%25==0:print(json.dumps({'step':index,'time_s':float(state['time_s'])}),flush=True)
        completed=True
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        if states:np.savez_compressed(out/'states.npz',**{k:np.stack([s[k] for s in states]) for k in states[0]})
        dump(out/'actions.json',executed)
        with (out/'observations.jsonl').open('w') as stream:
            for observation in observations:stream.write(json.dumps(observation,allow_nan=False)+'\n')
        unchanged=all(sha(p)==v for p,v in plan['backend_sources_sha256'].items())
        dump(out/'result.json',{'completed':completed,'error':error,'sources_unchanged':unchanged,
            'commands_executed':len(executed),'recorded_frames':len(states),'task_success_claimed':False,
            'scope':'Independent native collision diagnostic; no model, score override, attachment or post-reset object/cloth write'})
        dump(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
        print(json.dumps({'output':str(out),'completed':completed,'error':error}),flush=True)
        # Fast-shutdown may exit the process: evidence is sealed before close.
        if env is not None:env.close()
    if error:raise RuntimeError('Native diagnostic failed; preserved '+str(out))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,default=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v2_curated'))
    parser.add_argument('--task',default='03_lined_tray_loading')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--actions',type=Path)
    parser.add_argument('--robot-reset',type=Path)
    parser.add_argument('--condition',choices=['inventory','contact','hover'],default='inventory')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--warmup-steps',type=int,default=0)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--reanalyse-export',type=Path,help='CPU-only reanalysis of a sealed native diagnostic into a NEW output directory')
    parser.add_argument('--compare',type=Path,nargs=2,metavar=('CONTACT','HOVER'))
    args=parser.parse_args()
    if args.reanalyse_export:
        if args.execute or args.audit or args.compare or args.output is None:
            parser.error('--reanalyse-export is exclusive and requires a new --output')
        metadata,export,states,triangles=load_episode(args.reanalyse_export)
        updated=reanalyse_export(export)
        report=audit_cloth_robot(updated,states,triangles,metadata['body_names'])
        args.output.mkdir(parents=True,exist_ok=False)
        dump(args.output/'colliders_reanalysed.json',updated);dump(args.output/'report.json',report)
        dump(args.output/'provenance.json',{'source_episode':str(args.reanalyse_export.resolve()),
            'source_manifest_sha256':sha(args.reanalyse_export/'artifact_hashes.json'),
            'source_colliders_sha256':sha(args.reanalyse_export/'colliders.json'),
            'native_execution_performed':False,'original_episode_modified':False,
            'source_sha256':{str(p):sha(p) for p in [Path(__file__),PROJECT/'bench/evaluation/cloth/cloth_robot_collision.py',PROJECT/'bench/convex_geometry.py']}})
        dump(args.output/'artifact_hashes.json',{p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
        print(json.dumps({'output':str(args.output),'export_status':updated['status'],'audit_status':report['status']}));return
    if args.audit or args.compare:
        if args.execute or args.audit and args.compare:parser.error('CPU audit/compare and native execute are exclusive')
        result=inspect(args.audit) if args.audit else compare(*args.compare)
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'report.json',result)
        else:print(json.dumps(result,indent=2))
        return
    if args.warmup_steps<0:parser.error('Warmup steps must be nonnegative')
    entry,scene,task,initial=task_inputs(args.release,args.task,'nominal')
    if not scene['cloth']['enabled']:parser.error('Diagnostic requires enabled cloth')
    if args.robot_reset:
        initial=read(args.robot_reset)
        if set(initial)!={'joint_positions_rad'}:parser.error('Only explicitly declared robot joint reset is allowed')
    commands=read(args.actions) if args.actions else []
    if not isinstance(commands,list):parser.error('Actions must be a list')
    for command in commands:
        if type(command) is not dict or set(command)!={'joint_positions_rad','duration_s'} or type(command['duration_s']) not in (int,float) or command['duration_s']!=.02:
            parser.error('Each action must be only complete robot joint targets and 20ms duration')
    if args.condition!='inventory' and not commands:parser.error('Contact/hover requires frozen explicit commands')
    if args.condition=='inventory' and commands:parser.error('Inventory cannot execute contact commands')
    paths=[Path(__file__),PROJECT/'bench/evaluation/cloth/cloth_robot_collision.py',PROJECT/'bench/convex_geometry.py',PROJECT/'bench/isaacsim/release.py',
           *sorted((PROJECT/'bench/simulation/isaacsim_backend').glob('*.py')),*sorted((PROJECT/'bench/simulation/assets').glob('*.py'))]
    plan={'schema':'cloth_robot_collision_diagnostic_plan_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'condition':args.condition,'device':args.device,'seed':args.seed,'release':str(args.release.resolve()),
        'release_manifest_sha256':sha(args.release/'release_manifest.json'),'task_id':task['id'],'task_sha256':entry['task_sha256'],
        'scene_sha256':digest_json(scene),'initial':initial,'robot_reset_override':bool(args.robot_reset),
        'cloth_or_object_reset_override':False,'actions_sha256':sha(args.actions) if args.actions else None,
        'actions':commands,'warmup_steps':args.warmup_steps,'warmup_kind':'Retain current initial targets for recorded real 20ms steps',
        'backend_sources_sha256':{str(p):sha(p) for p in paths},'agent_or_task_success_claimed':False}
    if not args.execute:
        print(json.dumps({'native_will_start':False,'configuration':plan},indent=2));return
    if args.output is None:parser.error('--execute requires a new --output directory')
    execute(args,plan,scene,initial,commands)


if __name__=='__main__':main()
