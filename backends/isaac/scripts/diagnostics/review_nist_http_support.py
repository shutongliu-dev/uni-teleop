"""Read-only canonical NIST prop support audit from a completed native HTTP run.

Positions are recorded at acknowledged 20 ms endpoints. Native contacts cover
each 0.5 ms substep; this is not a full position-stream or manipulation replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bench.paths import load_json, OUTPUT_ROOT
from scripts.validation.mechanical_nist_support import support_evidence


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(private):
    files=[private/name for name in ('metadata.json','task.json','scene.json','physics.jsonl','physics_audit.json','final_score.json')]
    before={str(p):digest(p) for p in files}
    metadata=load_json(private/'metadata.json')
    task=load_json(private/'task.json')
    scene=load_json(private/'scene.json')
    rows=[json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines()]
    audit=load_json(private/'physics_audit.json')
    support=support_evidence(None,rows,task)
    states=[r['observation'] for r in rows]
    actors=task['asset_metadata']['actors']
    last=states[-1]['time_s'];terminal=[s for s in states if s['time_s']>=last-.3-1.e-8]
    idler_path=actors['tensioner_idler'];plate=task['asset_metadata']['fixtures']['workplate']
    native_direct=[];all_contacts=0;max_penetration=0.
    for obs in states:
        for contact in obs['contacts']['rigid']:
            all_contacts+=1;max_penetration=max(max_penetration,-contact['min_separation_m'])
            if (contact['actor0']==idler_path and contact['actor1'].startswith(plate+'/')
                    and contact['impulse_vector_ns'][2]>1.e-10):
                native_direct.append(contact['time_s'])
    terminal_times={round(t/.0005) for t in native_direct if t>=last-.3-1.e-8}
    canonical=task['asset_metadata']
    idler=next(o for o in canonical['objects'] if o['id']=='tensioner_idler')
    initial_error=np.asarray(states[0]['objects']['tensioner_idler']['position_m'])-idler['initial_position_m']
    no_false_success=all(r['score'].get('success') is False for r in rows)
    no_fault=all(not s['task_state']['errors'] and s['task_state']['complete']
        and s['task_state']['terminal_failure'] is None and s['task_state']['safety_stop'] is None for s in states)
    complete=all(s['contacts']['sampling']['complete'] and not s['contacts']['errors'] for s in states)
    substeps=sum(s['contacts']['sampling']['sampled_substeps'] for s in states)
    unchanged=all(digest(p)==before[str(p)] for p in files)
    checks={'canonical_prop_initialization':bool(np.linalg.norm(initial_error)<1.e-6),
        'all_67_props_supported_and_stable_at_recorded_endpoints':support['passed'],
        'idler_has_loaded_workplate_contact_each_terminal_substep':len(terminal_times)>=600,
        'native_contacts_complete':complete and substeps==round(last/.0005),
        'clean_native_observer_and_no_false_task_success':no_fault and no_false_success,
        'native_physics_audit_passed':audit['physics_valid'] is True,
        'records_unchanged_during_review':unchanged}
    return {'schema':'nist_http_canonical_support_review.v1','passed':all(checks.values()),'checks':checks,
        'configuration_id':scene.get('robot_configuration',{}).get('id'),
        'control_position_frames':len(states),'terminal_control_position_frames':len(terminal),
        'native_substeps':substeps,'elapsed_sim_s':last,'native_contact_rows':all_contacts,
        'idler_terminal_loaded_contact_substeps':len(terminal_times),
        'maximum_native_reported_penetration_m':max_penetration,
        'idler_initial_position_error_m':initial_error.tolist(),
        'idler_terminal':support['actors']['tensioner_idler'],
        'all_actor_support':support,
        'asset_bundle':{k:scene['native_task_bundle'][k] for k in ('usd_path','usd_sha256')},
        'scope':'Canonical supply support and retained pose stability at 20ms endpoints, plus every-substep contact coverage. '
            'No full native-position replay, grasp or assembly qualification.',
        'manipulation_qualified':False,'files_sha256':before,'review_source_sha256':digest(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose fresh shared evidence')
    result=review(args.private);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':result['checks'],'output':str(args.output)}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
