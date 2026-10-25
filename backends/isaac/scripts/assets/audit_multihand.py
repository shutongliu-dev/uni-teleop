"""Independently audit native embodiment smoke recordings, including failures."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from bench.paths import load_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    root=Path(root);result=load_json(root/'result.json');meta=load_json(root/'metadata.json');states=np.load(root/'states.npz')
    actions=[json.loads(x) for x in (root/'actions.jsonl').read_text().splitlines()];contacts=[json.loads(x) for x in (root/'contacts.jsonl').read_text().splitlines()]
    spec=meta['action_spec'];names=meta['joint_names'];active=spec['joint_names'];q=states['joint_q'];finite=all(np.isfinite(states[k]).all() for k in states)
    hashes=all(sha(root/name)==value for name,value in result['artifact_sha256'].items())
    inputs={**meta['robot_asset']['sources'],**meta['robot_asset']['mesh_inputs'],**meta['robot_asset']['license_sha256'],**meta['robot_usd_dependency_sha256']}
    input_hashes=all(sha(p)==h for p,h in inputs.items())
    snapshot_hashes=result.get('source_snapshot_sha256',{})
    snapshot_pass=bool(snapshot_hashes) and all(sha(root/'source_snapshot'/Path(p).name)==h for p,h in snapshot_hashes.items()) and result.get('sources_unchanged',False)
    frame=min(10,len(q)-1);response={n:float((q[-1,names.index(n)]-q[frame,names.index(n)])*np.sign(actions[-1]['joint_positions_rad'][n]-spec['initial_joint_positions_rad'][n])) for n in active}
    mimic={n:float(np.max(abs(q[:,names.index(n)]-m['multiplier']*q[:,names.index(m['joint'])]-m['offset']))) for n,m in spec.get('mimic_joints',{}).items()}
    limits=meta['robot_asset']['joint_limits'];excess={n:float(np.max(np.maximum(limits[n]['lower']-q[:,i],q[:,i]-limits[n]['upper']))) for i,n in enumerate(names)}
    body=states['body_q'];base_drift=float(np.max(np.linalg.norm(body[:,0,:3]-body[0,0,:3],axis=1)))
    self_contacts=[r for c in contacts for r in c['rigid'] if '/Robot/' in r['actor0'] and '/Robot/' in r['actor1']]
    penetration=max([max(0.,-r['min_separation_m']) for r in self_contacts],default=0.)
    native_steps=np.asarray(states['native_physics_step_index']);native_time=np.asarray(states['native_world_time_s']);clock=np.all(np.diff(native_steps)==round(.02/meta['physics_dt_s'])) and np.allclose(np.diff(native_time),.02,atol=1e-7)
    checks={'artifact_hashes':hashes,'asset_input_hashes':input_hashes,'source_snapshot':snapshot_pass,'all_state_finite':finite,
            'self_collision_enabled':meta['scene_info']['robot_self_collision']['enabled'],
            'no_diagnostic_override':not (root/'diagnostic_override.json').exists(),
            'positive_response_every_active_joint':all(v>1e-4 for v in response.values()),
            'mimic_constraints':all(np.isfinite(v) and v<.01 for v in mimic.values()),
            'joint_limits':all(np.isfinite(v) and v<.001 for v in excess.values()),
            'fixed_base':base_drift<.0001,'self_collision_penetration_under_2mm':penetration<.002,
            'native_clock':bool(clock),'duration_at_least_1_2s':float(states['time_s'][-1])>=1.2-1e-6,
            'actions_joint_only':all(set(a)=={'joint_positions_rad','duration_s'} and set(a['joint_positions_rad'])==set(active) for a in actions),
            'positive_native_support_impulses':sum(c['rigid_impulse_magnitude_sum_ns'] for c in contacts)>0,
            'contact_errors_empty':not any(c['errors'] for c in contacts),
            'reset_and_render_checks':result['reset_matches'] and result['render_isolation']}
    return {'recording':str(root),'hand_id':spec['hand_id'],'passed':all(checks.values()),'checks':checks,
            'active_dof':len(active),'physical_dof':len(names),'joint_response_rad':response,'mimic_max_error_rad':mimic,
            'max_joint_limit_excess_rad':max(excess.values()),'max_base_drift_m':base_drift,'max_self_penetration_m':penetration,
            'scope':'Import/control/reset/native support-contact qualification; not task success or real hardware fidelity',
            'result_sha256':sha(root/'result.json'),'auditor_sha256':sha(__file__)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recording',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.recording);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':r['passed'],'checks':r['checks']}))
