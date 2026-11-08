"""Read-only full-frame review and state-specific workspace export for the stand."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from bench.isaacsim.release import checked, digest, dump, _frozen_evaluator_module
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def review(case, release):
    report=load_json(case/'qualification.json')
    task=load_json(release/'task.json')
    evaluator=_frozen_evaluator_module(release).SequenceEvaluator(task,trusted_normalized_input=True)
    for name,sha in report['evidence_sha256'].items():checked(case/name,sha)
    def actor(path):
        for n,p in {**task['evaluation']['body_paths'],**task['evaluation']['support_paths']}.items():
            if path==p or path.startswith(p+'/'):return n
        return 'external/'+path.strip('/').replace('/','_')
    sequence=_frozen_evaluator_module(release).SequenceEvaluator(task,trusted_normalized_input=True)
    sequence_result=None;sequence_rejection_index=None
    rows=[];previous=None;support_frames=0;wrenches=0
    with gzip.open(case/'native_frames.jsonl.gz','rt') as stream:
        for index,line in enumerate(stream):
            row=json.loads(line);assert row['step']==index and abs(row['time_s']-index*.0005)<1.e-9
            assert set(row['objects'])==set(task['evaluation']['body_paths'])
            for obj in row['objects'].values():
                assert np.isfinite(np.concatenate([obj[k] for k in ('position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s')])).all()
                assert abs(np.linalg.norm(obj['quaternion_wxyz'])-1)<1.e-5
            # Fresh frozen referee instance; complete raw native states, never
            # filtered, resampled, zeroed or replaced by finite differences.
            states={n:evaluator._state(s) for n,s in row['objects'].items()}
            normalized=[{**r,'a':actor(r['a']),'b':actor(r['b'])} for r in row['raw_contacts']]
            assert normalized==row['contacts']
            contacts=evaluator._contacts({'contacts':normalized})
            measured=evaluator.physical_conditions(states,contacts,previous)
            assert measured==row['conditions']
            if measured['conditions_passed']:support_frames+=1
            for name,w in row['engineering_wrenches'].items():
                wrenches+=1;assert name in row['objects']
                assert max(abs(x) for x in w['force_n'])<=6.+1.e-9
                assert max(abs(x) for x in w['torque_nm'])<=(2.5 if name=='panel' else .2)+1.e-9
            if report['case']=='reset' or (report['case'] in ('cycle','sequential_cycle') and sequence_rejection_index is None):
                evidence={'schema':'study_native_observation_v1','configuration_sha256':task['evaluation']['configuration']['configuration_sha256'],
                    'episode_id':'declared_engineering_'+report['case'],'step_index':index,'time_s':row['time_s'],
                    'physics_dt_s':.0005,'sampling_complete':True,'contacts_complete':True,'physics_valid':True,
                    'robot_only_actions':not bool(row['engineering_wrenches']),'object_pose_writes':False,
                    'budget_exhausted':False,'robot_action_count':0,'objects':row['objects'],'contacts':normalized}
                sequence_result=sequence.update(evidence)
                assert not sequence_result['success']
                if report['case']=='reset':assert sequence_result['status']=='active' and sequence_result['progress']==0.
                elif row['engineering_wrenches']:
                    assert sequence_result['status']=='invalid';sequence_rejection_index=index
            rows.append(row);previous=states
    assert len(rows)==report['native_frames']==round(report['duration_s']/.0005)+1
    tail=[r for r in rows if r['time_s']>=report['duration_s']-.4-1.e-9]
    assert len(tail)==801 and all(not r['engineering_wrenches'] for r in tail)
    # Independently bound the full terminal pose spread in addition to each
    # frame's raw speed and actual signed load-bearing contacts.
    spread={}
    for n in rows[0]['objects']:
        positions=np.asarray([r['objects'][n]['position_m'] for r in tail])
        qs=np.asarray([r['objects'][n]['quaternion_wxyz'] for r in tail]);qs/=np.linalg.norm(qs,axis=1)[:,None]
        spread[n]={'position_from_first_m':float(np.max(np.linalg.norm(positions-positions[0],axis=1))),
                   'orientation_from_first_rad':float(np.max(2*np.arccos(np.clip(np.abs(qs@qs[0]),0,1))))}
    if report['case'] in ('cycle','sequential_cycle','load'):
        assert all(r['conditions']['conditions_passed'] for r in tail)
        assert all(s['position_from_first_m']<=.003 and s['orientation_from_first_rad']<=.06 for s in spread.values())
    elif report['case']=='reset':assert support_frames==0
    elif report['case']=='unseated':
        assert support_frames==0
        assert all(r['conditions']['objects']['panel']['geometry'] and r['conditions']['objects']['panel']['supported'] for r in tail)
        assert all(not r['conditions']['objects']['book']['geometry'] and not r['conditions']['objects']['book']['supported'] for r in tail)
    else:
        assert rows[-1]['conditions']['panel_angle_deg']<20.
        assert not any(r['conditions']['conditions_passed'] for r in tail)
    if report['case']=='withdraw':
        assert all(r['conditions']['conditions_passed'] for r in rows if .4<=r['time_s']<.8)
    assert report['passed'] and all(report['checks'].values())
    if report['case'] in ('cycle','sequential_cycle'):assert sequence_rejection_index is not None
    if report['case']=='sequential_cycle':assert all(len(r['engineering_wrenches'])<=1 for r in rows)
    result={'case':report['case'],'passed':True,'qualification':str(case/'qualification.json'),
        'qualification_sha256':digest(case/'qualification.json'),'raw_trace':str(case/'native_frames.jsonl.gz'),
        'raw_trace_sha256':digest(case/'native_frames.jsonl.gz'),'native_frames':len(rows),'terminal_frames':len(tail),
        'physical_conditions_true_frames':support_frames,'terminal_spread':spread,
        'engineering_wrench_records':wrenches,'ordinary_referee_result':sequence_result,
        'engineering_trace_rejected_at_frame':sequence_rejection_index,'robot_manipulation_qualified':False,'task_policy_success_claimed':False}
    return result,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',required=True,type=Path)
    p.add_argument('--case',action='append',type=Path,required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();source=verify_if_snapshot(Path(__file__).resolve().parents[2],required_modules=('scripts.validation.study_reading_stand_readback',),entrypoint_file=Path(__file__))
    if source is None:raise ValueError('Read back from frozen full source')
    if a.output.exists():raise FileExistsError(a.output)
    task=load_json(a.release/'task.json');reports=[];stages=[];raw=[]
    labels={'folded':['panel_handle_folded','support_pin_handle_withdrawn','book_source'],
        'deployed_unlocked':['panel_handle_deployed','support_pin_handle_withdrawn'],
        'pin_inserted_released_panel':['panel_handle_deployed','support_pin_handle_inserted'],
        'loaded_all_released':['panel_handle_deployed','support_pin_handle_inserted','book_reading_target']}
    for path in a.case:
        result,rows=review(path,a.release);reports.append(result)
        raw.append({'path':result['raw_trace'],'sha256':result['raw_trace_sha256']})
        if result['case'] in ('cycle','sequential_cycle'):
            report=load_json(path/'qualification.json')
            for name,t in report['stage_times_s'].items():
                index=round(t/.0005);row=rows[index];assert abs(row['time_s']-t)<1.e-9
                stages.append({'label':result['case']+'.'+name,'protocol':result['case'],'sim_time_s':row['time_s'],'trace_frame_index':index,
                    'native_trace_path':result['raw_trace'],'target_labels':labels[name],
                    'body_poses':[{'id':n,'prim_path':task['evaluation']['body_paths'][n],
                        'position_m':s['position_m'],'quaternion_wxyz':s['quaternion_wxyz']} for n,s in row['objects'].items()]})
    assert len(reports)==7 and {r['case'] for r in reports}=={'cycle','sequential_cycle','reset','load','withdraw','unlocked','unseated'}
    assert set(x for stage in stages for x in stage['target_labels'])==set(x['label'] for x in task['evaluation']['configuration']['interaction_targets'])
    result={'schema':'study_reading_stand_full_native_readback.v1','passed':True,'source_snapshot':source,
        'release_path':str(a.release),'release_manifest_sha256':digest(a.release/'release_manifest.json'),
        'case_reports':reports,'workspace_stages':stages,'native_trace_files':raw,
        'shared_approach_candidates_m':[[.37,-.07,1.27],[.42,.07,1.27],[.30,0.,1.26]],
        'workspace_scope':'Actual recorded poses for offline geometry/IK only; no runtime object writes. Shared candidates are unqualified free-space suggestions.',
        'robot_manipulation_qualified':False,'task_policy_success_claimed':False}
    dump(a.output,result);print(json.dumps({'passed':True,'output':str(a.output),'sha256':digest(a.output)}))


if __name__=='__main__':main()
