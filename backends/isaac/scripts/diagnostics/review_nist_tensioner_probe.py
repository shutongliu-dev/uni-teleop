"""Independent full-stream readback of the Task7 passive idler insertion probe.

The clearance witness samples the declared cylindrical/tapered guide surface
against the actual transformed bore. It is conservative geometric diagnostics,
not a replacement for the retained native signed contact/separation evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from bench.paths import load_json, OUTPUT_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def review(root):
    report = load_json(root/'qualification.json')
    metadata = load_json(root/'metadata.json')
    objects = {o['id']:o for o in metadata['objects']}
    target = np.asarray(metadata['diagnostic_goal_poses']['tensioner_idler'][:3])
    shaft = metadata['fixture_metadata']['tensioner_idler_shaft']
    shaft_origin = np.asarray(shaft['initial_position_m'])
    actor = metadata['actors']['tensioner_idler']
    shaft_path = shaft['body_path']
    initial = np.asarray(report['reset_only_goal_poses']['tensioner_idler'][:3])
    rev = metadata['task_parameters']['idler_handling_revision']
    guide = rev['guide_post']
    if (rev['through_bore_diameter_m'] != .012 or rev['lower_entry_diameter_m'] != .014
            or guide['top_above_workplate_m'] != .0475
            or guide['straight_radius_m'] != .0037):
        raise ValueError('This explicit geometric review requires the usability_v3 guide and bore dimensions')
    # Circumscribed primitive samples are conservative against the authored
    # polygonal guide. The 96-sector bore's inscribed radius is used below.
    z = np.r_[np.linspace(.0015,.0425,64),np.linspace(.0425,.0475,24)]
    radii = np.where(z<=.0425,.0037,.0037-(z-.0425)*(.0014/.005))
    angle = np.arange(64)*2*math.pi/64
    points = np.stack(np.broadcast_arrays(radii[:,None]*np.cos(angle),
        radii[:,None]*np.sin(angle),z[:,None]),axis=-1).reshape(-1,3)+shaft_origin
    last_step = report['native_clock']['baseline_step']
    last_time = report['native_clock']['baseline_time_s']
    summaries=[]; first_goal=None; worst_gap=math.inf; worst_frame=None
    max_native_penetration=0.; max_anchor_error=0.
    all_clocks=True; finite=True
    with (root/'native_frames.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);obj=row['objects']['tensioner_idler']
            position=np.asarray(obj['position_m']);R=rotation(obj['quaternion_wxyz'])
            speed=float(np.linalg.norm(obj['linear_velocity_m_s']))
            angular=float(np.linalg.norm(obj['angular_velocity_rad_s']))
            tilt=math.acos(float(np.clip(R[2,2],-1.,1.)))
            goal=bool(np.all(abs(position-target)<=np.array([.005,.005,.006])) and tilt<=.35)
            if goal and first_goal is None:first_goal=row['time_s']
            all_clocks &= (row['native_step']==last_step+1 and
                abs(row['native_time_s']-last_time-float(np.float32(.0005)))<1.e-9)
            last_step,last_time=row['native_step'],row['native_time_s']
            finite &= bool(np.isfinite([*position,speed,angular,tilt]).all())
            local=(points-position)@R
            use=(local[:,2]>=-.013)&(local[:,2]<=.036)
            selected=local[use]
            minimum=None
            if len(selected):
                zz=selected[:,2]
                bore=np.where(zz<-.010,.006+(-.010-zz)*(.001/.003),.006)
                bore=np.where(zz>.0355,.006+(zz-.0355),bore)*math.cos(math.pi/96)
                minimum=float(np.min(bore-np.linalg.norm(selected[:,:2],axis=1)))
                if minimum<worst_gap:worst_gap=minimum;worst_frame=row['native_step']
            contacts=[c for c in row['raw_contacts'] if c['actor0']==actor and
                      (c['actor1']==shaft_path or c['actor1'].startswith(shaft_path+'/'))]
            shaft_contact=any(min(c['separations_m'])<=.0005 for c in contacts)
            max_native_penetration=max(max_native_penetration,
                max([-min(c['separations_m']) for c in row['raw_contacts']],default=0.))
            max_anchor_error=max(max_anchor_error,row['passive_links']['maximum_anchor_error_m'])
            summaries.append({'t':row['time_s'],'p':position,'speed':speed,'angular':angular,
                'tilt':tilt,'goal':goal,'shaft_contact':shaft_contact,
                'gap':minimum,'radial_axis_offset':float(np.linalg.norm(position[:2]-shaft_origin[:2]))})
    final=summaries[-1];tail=[r for r in summaries if r['t']>=final['t']-.3-1.e-7]
    expected=round(report['duration_s']/.0005)
    tail_positions=np.asarray([r['p'] for r in tail])
    guided=report['case']=='tensioner_guided_drop'
    initial_bottom=initial[2]+objects['tensioner_idler']['local_bounds_m'][0][2]
    shaft_top=shaft_origin[2]+shaft['local_bounds_m'][1][2]
    start_valid=(.0099<initial[2]-target[2]<.0101 and shaft_top-initial_bottom>.020) if guided else initial_bottom>shaft_top+.002
    checks={
        'producer_report_passed':report['passed'] is True,
        'unaltered_native_inputs_and_stream':all(report['files_sha256'].get(name)==digest(root/name)
            for name in ('metadata.json','native_frames.jsonl','probe_initial.usda')),
        'native_idler_mass_matches_asset':abs(report['native_masses_kg']['tensioner_idler']-
            objects['tensioner_idler']['mass_kg'])<1.e-7,
        'complete_native_stream':len(summaries)==expected,
        'continuous_native_clock':bool(all_clocks), 'finite_states':bool(finite),
        'declared_initial_insertion_state':start_valid,
        'actual_downward_travel':initial[2]-final['p'][2]>(.008 if guided else .045),
        'fresh_300ms_window':len(tail)>=601 and tail[-1]['t']-tail[0]['t']>=.3-1.e-7,
        'every_terminal_seat_within_original_tolerance':all(r['goal'] for r in tail),
        'every_terminal_actual_post_contact':all(r['shaft_contact'] for r in tail),
        'terminal_speed_under_original_limit':max(r['speed'] for r in tail)<=.04,
        'terminal_angular_speed_under_original_limit':max(r['angular'] for r in tail)<=.5,
        'terminal_position_drift_under_original_limit':float(np.max(np.linalg.norm(tail_positions-final['p'],axis=1)))<=.005,
        'native_penetration_under_original_limit':max_native_penetration<=.006,
        'no_live_pose_or_external_force_control':report['post_reset_pose_writes']==0 and report['external_force_calls']==0,
        'passive_belt_connection_preserved':max_anchor_error<=.003,
    }
    checks={name:bool(value) for name,value in checks.items()}
    return {'schema':'nist_tensioner_independent_readback.v1','passed':all(checks.values()),'checks':checks,
        'frames':len(summaries),'terminal_frames':len(tail),'duration_s':report['duration_s'],
        'case':report['case'],'initially_fully_detached':bool(initial_bottom>shaft_top+.002),
        'initial_offset_above_seated_goal_m':float(initial[2]-target[2]),
        'first_original_seating_geometry_time_s':first_goal,
        'actual_vertical_travel_m':float(initial[2]-final['p'][2]),
        'final_position_m':final['p'].tolist(),'final_position_error_m':(final['p']-target).tolist(),
        'final_axis_tilt_rad':final['tilt'],
        'terminal_radial_axis_offset_range_m':[min(r['radial_axis_offset'] for r in tail),max(r['radial_axis_offset'] for r in tail)],
        'minimum_sampled_guide_bore_clearance_m':worst_gap,'minimum_clearance_frame':worst_frame,
        'terminal_minimum_sampled_guide_bore_clearance_m':min(r['gap'] for r in tail if r['gap'] is not None),
        'sampled_clearance_semantics':'Circumscribed 64-angle/88-height guide-surface samples inside transformed 96-sector bore; '
            'negative values are conservative surface overlap diagnostics, not cooked SDF penetration. Native manifolds remain authoritative.',
        'maximum_native_reported_penetration_m':max_native_penetration,
        'maximum_passive_belt_anchor_error_m':max_anchor_error,
        'terminal_maximum_linear_speed_m_s':max(r['speed'] for r in tail),
        'terminal_maximum_angular_speed_rad_s':max(r['angular'] for r in tail),
        'terminal_position_drift_m':float(np.max(np.linalg.norm(tail_positions-final['p'],axis=1))),
        'qualified_scope':('Low-energy final 10 mm insertion of an already guided idler. ' if guided else 'Initially detached 50 mm idler insertion. ')
            +'Already routed real closed belt, released physical support and stability. '
            'The belt is initially authored around the target wheels; laying the complete belt with a robot is not qualified.',
        'robot_manipulation_qualified':False,'full_belt_installation_qualified':False,
        'files_sha256':{str((root/name).resolve()):digest(root/name) for name in
            ('qualification.json','metadata.json','native_frames.jsonl')},
        'review_source_sha256':digest(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose fresh shared evidence')
    result=review(args.probe)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':result['checks'],'output':str(args.output)}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
