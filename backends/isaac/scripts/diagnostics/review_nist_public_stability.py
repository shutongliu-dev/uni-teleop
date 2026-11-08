"""Replay frozen NIST physical predicates over an engineering probe's full stream.

The engineering reset is deliberately not relabelled as the benchmark's supply
reset. Only unchanged geometry/contact/release/quiet/dwell methods are used;
operation history, on_frame baseline acceptance and task scoring are not run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bench.isaacsim.release import validate, _frozen_evaluator_module
from bench.paths import load_json, OUTPUT_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_contacts(raw, time_s):
    """One real sensor record per unordered pair; retain its original sign."""
    selected={};duplicates=0
    for row in raw:
        key=tuple(sorted((row['actor0'],row['actor1'])))
        if key in selected:
            duplicates+=1
            continue
        selected[key]={k:row[k] for k in ('actor0','actor1','impulse_vector_ns','impulse_magnitude_sum_ns')}
        selected[key].update(time_s=time_s,min_separation_m=min(row['separations_m']))
    return list(selected.values()),duplicates


def review(probe, output):
    report=load_json(probe/'qualification.json')
    release=Path(report['release']['path'])
    validate(release)
    if digest(release/'release_manifest.json')!=report['release']['manifest_sha256']:
        raise ValueError('Probe release identity changed')
    for name in ('native_frames.jsonl','metadata.json','probe_initial.usda'):
        if digest(probe/name)!=report['files_sha256'][name]:
            raise ValueError('Native evidence changed: '+name)
    task=load_json(release/'task.json')
    module=_frozen_evaluator_module(release)
    observer=module.NativeTaskObserver(task)
    if task['asset_metadata']!=load_json(probe/'metadata.json'):
        raise ValueError('Native metadata differs from frozen referee input')
    tail=[];duration=0.;anchor=None;duplicates=0;count=0;first_stable=None
    previous_step=report['native_clock']['baseline_step']
    previous_time=report['native_clock']['baseline_time_s']
    frame_path=output.with_suffix('.frames.jsonl')
    with (probe/'native_frames.jsonl').open() as src,frame_path.open('x') as dst:
        for line in src:
            row=json.loads(line);count+=1;t=row['time_s']
            if row['native_step']!=previous_step+1 or abs(row['native_time_s']-previous_time-float(np.float32(.0005)))>1.e-9:
                raise ValueError('Native stream is not consecutive')
            previous_step,previous_time=row['native_step'],row['native_time_s']
            objects={name:module.state(value) for name,value in row['objects'].items()}
            mapped,n=selected_contacts(row['raw_contacts'],t);duplicates+=n
            contacts=module.Contacts(mapped,observer.paths,observer.specs,t,
                observer.limits['minimum_contact_impulse_ns'],observer.limits['maximum_mating_gap_m'])
            observer._quiet(objects,contacts)
            groups={}
            for group in observer.p.get('flexible_groups',[]):
                good,detail=observer._group(group,objects,contacts,t)
                groups[group['id']]=dict(complete=bool(good),**detail)
            seats={}
            for goal in observer.p['seating_goals']:
                good,detail=observer._seat(goal,objects,contacts,t,groups)
                seats[goal['id']]=dict(complete=bool(good),**detail)
            geometry=all(r['complete'] for r in [*groups.values(),*seats.values()])
            quiet=all(observer.quiet.values())
            released=not contacts.robot
            safety=contacts.minimum_separation>=-observer.limits['maximum_penetration_m']
            no_drop=not contacts.floor and all(v['position_m'][2]>=observer.limits['fallen_height_m'] for v in objects.values())
            condition=bool(geometry and quiet and released and safety and no_drop and row['passive_links']['passed'])
            duration,anchor=observer._dwell(condition,objects,duration,anchor)
            if duration+1.e-10>=observer.limits['stable_s'] and first_stable is None:first_stable=t
            state={'native_step':row['native_step'],'time_s':t,'physical_predicates':condition,
                'geometry_support_mating':bool(geometry),'quiet_from_frozen_20ms_pose_rate':bool(quiet),
                'released':bool(released),'safety':bool(safety),'no_drop':bool(no_drop),
                'physical_stable_duration_s':duration,
                'maximum_20ms_pose_linear_speed_m_s':max((v['linear_m_s'] for v in observer.pose_speed.values()),default=None),
                'maximum_raw_linear_speed_m_s':max(float(np.linalg.norm(v['linear_velocity_m_s'])) for v in objects.values()),
                'groups':groups,'seats':seats}
            dst.write(json.dumps(state,separators=(',',':'))+'\n')
            tail.append(state)
            if len(tail)>601:tail.pop(0)
    checks={'complete_stream':count==round(report['duration_s']/.0005),
        'fresh_300ms_every_frame_physical_conditions':len(tail)==601 and all(r['physical_predicates'] for r in tail),
        'unchanged_frozen_dwell_reached':duration+1.e-10>=observer.limits['stable_s'],
        'no_post_reset_pose_or_force_control':report['post_reset_pose_writes']==0 and report['external_force_calls']==0}
    result={'schema':'nist_frozen_physical_predicates_replay.v1','passed':all(checks.values()),'checks':checks,
        'native_frames':count,'terminal_frames':len(tail),'terminal_duration_s':tail[-1]['time_s']-tail[0]['time_s'],
        'first_frozen_physical_stable_time_s':first_stable,'final_physical_stable_duration_s':duration,
        'terminal_maximum_20ms_pose_linear_speed_m_s':max(r['maximum_20ms_pose_linear_speed_m_s'] for r in tail),
        'terminal_maximum_raw_linear_speed_m_s':max(r['maximum_raw_linear_speed_m_s'] for r in tail),
        'original_thresholds':observer.limits,
        'strict_raw_velocity_probe_passed':report['passed'],
        'contact_conversion':'First native sensor row per unordered pair; original actor order, signed vector, magnitude and minimum separation retained; no force normalization.',
        'duplicate_reciprocal_sensor_rows_removed':duplicates,
        'scope':'Frozen physical predicates only: every-frame geometry, real support/mating, release, 20ms pose rate, 5mm anchor and 300ms dwell. '
            'Engineering pre-routed belt reset cannot earn robot operation credit.',
        'full_observer_on_frame_replay':False,'task_success_claimed':False,'robot_manipulation_qualified':False,
        'full_belt_installation_qualified':False,'physics_accepted':False,
        'files_sha256':{str(p.resolve()):digest(p) for p in [probe/'qualification.json',probe/'native_frames.jsonl',
            probe/'metadata.json',release/'release_manifest.json',release/'evaluator/sequence_evaluator.py',
            release/'evaluator/rules.py',release/'evaluator/native_evidence.py',frame_path,Path(__file__)]}}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose fresh shared evidence')
    result=review(args.probe,args.output)
    print(json.dumps({'passed':result['passed'],'checks':result['checks'],'output':str(args.output)}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
