"""Bind object-only NIST evidence to exact frozen single/dual scene assets.

Identical object bundles do not transfer robot, whole-room, interface or full
manipulation qualification. Each of those requires its own scoped evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from bench.paths import load_json, OUTPUT_ROOT


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def value_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inputs(root):
    manifest=load_json(root/'release_manifest.json')
    files=manifest.get('files_sha256') or {k:v['sha256'] for k,v in manifest['files'].items()}
    for name in ('scene.json','task.json','physics_profile.json'):
        if digest(root/name)!=files[name]:raise ValueError('Frozen input hash mismatch: '+str(root/name))
    scene=load_json(root/'scene.json');task=load_json(root/'task.json')
    bundle=scene['native_task_bundle']
    if digest(bundle['usd_path'])!=bundle['usd_sha256']:raise ValueError('Shared USD changed')
    return scene,task,manifest


def compare(probe, candidates):
    ps,pt,pm=inputs(probe);pb=ps['native_task_bundle']
    physics_bundle={k:v for k,v in pb.items() if k!='observer'}
    rows=[]
    for path in candidates:
        scene,task,manifest=inputs(path)
        bundle={k:v for k,v in scene['native_task_bundle'].items() if k!='observer'}
        checks={
            'same_task_id':task['id']==pt['id'],
            'identical_native_physical_bundle':bundle==physics_bundle,
            'identical_complete_physical_metadata':task['asset_metadata']==pt['asset_metadata'],
            'identical_evaluator_physical_metadata':task['evaluation']['metadata']==pt['evaluation']['metadata'],
            'identical_numeric_rules':task['evaluation']['thresholds']==pt['evaluation']['thresholds'],
            'identical_physics_timestep_profile':scene['physics']==ps['physics'],
            'identical_native_solver_options':scene['isaacsim']==ps['isaacsim'],
            'identical_parent_environment':scene['workroom']==ps['workroom'],
        }
        rows.append({'path':str(path.resolve()),'manifest_sha256':digest(path/'release_manifest.json'),
            'configuration_id':scene.get('robot_configuration',{}).get('id','single_arm'),
            'passed':all(checks.values()),'checks':checks,
            'physical_metadata_sha256':value_digest(task['asset_metadata'])})
    return {'schema':'nist_object_only_identity_bridge.v1','passed':all(r['passed'] for r in rows),
        'probe_release':{'path':str(probe.resolve()),'manifest_sha256':digest(probe/'release_manifest.json')},
        'native_bundle':physics_bundle,'physical_metadata_sha256':value_digest(pt['asset_metadata']),
        'candidates':rows,
        'allowed_evidence_scope':'The same native object geometry, material/mass/inertia/COM, passive joints, physical goals, solver and timestep. '
            'Only the explicitly tested guided object insertion/support result may be associated with these object assets.',
        'excluded_qualification':['Whole furnished scene contact behavior','Robot/object interaction','Other robot layouts native initialization',
            'Agent/Bench transport','Complete robot belt assembly','Detached 50mm drop success','Calibrated physics'],
        'observer_exclusion':'Observer source/configuration may carry installation-specific operator metadata; the complete physical metadata and numeric limits are compared separately.',
        'physics_accepted':False,'robot_manipulation_qualified':False,'review_source_sha256':digest(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-release',type=Path,required=True)
    parser.add_argument('--release',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose fresh shared evidence')
    result=compare(args.probe_release,args.release)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':result['passed'],'candidates':len(result['candidates']),'output':str(args.output)}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
