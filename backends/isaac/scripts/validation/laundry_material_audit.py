"""Read-only whole-stream checks for passive laundry material/dynamics probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_material_settle import mesh_data


def pin(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return {'path': str(p.resolve()), 'sha256': h.hexdigest()}


def freefall_dynamics(t, minimum_z, velocity, com_z, kinetic_energy, *, table_z, thickness, contact_offset, damping):
    """Compare observed momentum with the declared semi-implicit damped gravity law.

    Skip startup warmup and every sample within contact detection distance.
    This tests free fall, not a measured mass, contact force, or material modulus.
    """
    dt = float(np.median(np.diff(t)))
    contact = np.flatnonzero(minimum_z <= table_z+contact_offset+.0005)
    end = int(contact[0]) if len(contact) else len(t)
    ids = np.flatnonzero((t >= .008) & (np.arange(len(t)) < end-1))
    result = {'first_contact_envelope_time_s': float(t[end]) if end < len(t) else None,
              'sample_count': len(ids), 'declared_gravity_m_s2': -9.81,
              'declared_linear_damping_s_inv': damping,
              'initial_com_vz_m_s': float(velocity[0]),
              'com_drop_m': float(com_z[0]-com_z[-1]),
              'peak_kinetic_energy_per_mass_j_kg': float(kinetic_energy.max()),
              'terminal_kinetic_energy_per_mass_j_kg': float(kinetic_energy[-1]),
              'terminal_to_peak_kinetic_energy_ratio': float(kinetic_energy[-1]/max(kinetic_energy.max(), 1.e-30)),
              'scope': 'Native free fall and dissipation only; no external force or mass calibration.'}
    if len(ids) >= 3:
        expected = (1-damping*dt)*(velocity[ids]-9.81*dt)
        residual = velocity[ids+1]-expected
        result.update(freefall_window_s=[float(t[ids[0]]), float(t[ids[-1]+1])],
                      fitted_com_acceleration_m_s2=float(np.polyfit(t[ids],velocity[ids],1)[0]),
                      damped_gravity_step_velocity_error_max_m_s=float(np.max(np.abs(residual))),
                      damped_gravity_step_velocity_error_rms_m_s=float(np.sqrt(np.mean(residual**2))))
    if end < len(t):
        result['post_contact_upward_com_velocity_max_m_s'] = float(max(0.,velocity[end:].max()))
        result['post_contact_com_height_excursion_m'] = float(com_z[end:].max()-com_z[-1])
    return result


def audit(report_path):
    report_path = Path(report_path).resolve()
    report = load_json(report_path)
    if not report.get('complete') or not report['states']['complete']:
        raise ValueError('Complete native report and full observation stream required')
    scene = load_json(Path(report['scene_input']['path']))
    task = load_json(Path(report['task_input']['path']))
    for key in ('scene_input','task_input','initial_input','source_manifest'):
        if pin(report[key]['path']) != report[key]:
            raise ValueError('Source/input hash changed: '+key)
    files = report['states']['files']; n=report['states']['valid_samples']; arrays={}; checks={}
    for name, record in files.items():
        digest=pin(record['path'])
        if digest['sha256']!=record['sha256']:
            raise ValueError('Native stream hash mismatch: '+name)
        value=np.load(record['path'],mmap_mode='r')
        if list(value.shape)!=record['shape'] or str(value.dtype)!=record['dtype'] or len(value)!=n:
            raise ValueError('Native stream shape/dtype/count mismatch: '+name)
        finite=all(np.isfinite(value[i:i+64]).all() for i in range(0,n,64))
        checks[name]={'sha256_verified':True,'finite_all_samples':bool(finite),'shape':record['shape']}
        if not finite:raise ValueError('Nonfinite native stream: '+name)
        arrays[name]=value
    t=arrays['time_s'];dt=float(scene['physics']['dt_s'])
    if not np.allclose(t,np.arange(n)*dt,atol=1.e-10,rtol=0):raise ValueError('Wrong per-native-step timestamps')
    if not np.array_equal(arrays['native_step'],np.arange(n)):raise ValueError('Missing native steps')
    if not np.array_equal(arrays['world_step']-arrays['world_step'][0],np.arange(n)):raise ValueError('Wrong world step increments')
    if not np.allclose(arrays['world_time_s']-arrays['world_time_s'][0],t,atol=2.e-6,rtol=0):raise ValueError('Wrong world clock')
    table=task['asset_metadata']['table_top_z_m'];rect=task['asset_metadata']['clear_rectangle_xy_m']
    out={'schema':'laundry_passive_material_whole_stream_audit.v1','native_report':pin(report_path),
         'analysis_script':pin(__file__),'native_samples':n,'duration_s':float(t[-1]),
         'stream_integrity':checks,'all_native_clock_steps_observed':True,'cloths':{},
         'actual_attachment_prims':report['actual_attachment_prims'],
         'cloth_position_velocity_writes':report['cloth_position_velocity_writes'],
         'scope':'Passive material/dynamics validation; no task completion, robot grasp, or calibrated physics qualification.'}
    for spec in scene['cloths']:
        name=spec['id'];fields=report['cloth_stream_fields'][name];q=arrays[fields['q']];qd=arrays[fields['qd']]
        mask=arrays[fields['mask']];mesh=mesh_data(spec,task);thick=spec['visual_thickness_m'];edge=mesh['edges'];tri=mesh['triangles']
        values={key:np.empty(n) for key in ('edge_ratio_max','triangle_area_ratio_min','table_penetration_max_m','height_range_m','com_z_m','com_vz_m_s','kinetic_energy_per_mass_j_kg')}
        for start in range(0,n,32):
            x=np.asarray(q[start:start+32],dtype=float);v=np.asarray(qd[start:start+32],dtype=float);end=start+len(x)
            values['edge_ratio_max'][start:end]=(np.linalg.norm(x[:,edge[:,1]]-x[:,edge[:,0]],axis=2)/mesh['edge_lengths']).max(axis=1)
            area=np.linalg.norm(np.cross(x[:,tri[:,1]]-x[:,tri[:,0]],x[:,tri[:,2]]-x[:,tri[:,0]]),axis=2)/2
            values['triangle_area_ratio_min'][start:end]=(area/mesh['areas']).min(axis=1)
            inside=(x[:,:,0]>=rect[0])&(x[:,:,0]<=rect[1])&(x[:,:,1]>=rect[2])&(x[:,:,1]<=rect[3])
            values['table_penetration_max_m'][start:end]=np.where(inside,np.maximum(table+thick/2-x[:,:,2],0),0).max(axis=1)
            values['height_range_m'][start:end]=np.ptp(x[:,:,2],axis=1)
            values['com_z_m'][start:end]=np.einsum('bn,n->b',x[:,:,2],mesh['weights'])
            values['com_vz_m_s'][start:end]=np.einsum('bn,n->b',v[:,:,2],mesh['weights'])
            values['kinetic_energy_per_mass_j_kg'][start:end]=.5*np.einsum('bn,n->b',np.sum(v*v,axis=2),mesh['weights'])
        comparison={key:float(np.max(np.abs(value-arrays['metric_'+name+'_'+key]))) for key,value in values.items()}
        if max(comparison.values())>1.e-9:raise ValueError('Recorded scalar differs from native arrays: '+name)
        extremes={}
        for key,value in values.items():
            i=int(np.argmin(value) if key=='triangle_area_ratio_min' else np.argmax(value))
            extremes[key]={'value':float(value[i]),'native_step':i,'time_s':float(t[i])}
        usd=report['actual_usd_materials'][name];body=usd['contact_and_body_properties']
        dynamic=freefall_dynamics(t,arrays['metric_'+name+'_minimum_midplane_z_m'],values['com_vz_m_s'],values['com_z_m'],values['kinetic_energy_per_mass_j_kg'],table_z=table,thickness=thick,contact_offset=body['physxCollision:contactOffset'],damping=body['physxDeformableBody:linearDamping'])
        out['cloths'][name]={'actual_usd_material':usd,'whole_stream_extremes':extremes,
            'independent_scalar_recomputation_max_abs_errors':comparison,
            'controlled_vertex_samples':int(np.count_nonzero(mask)),
            'terminal_diagnostics':report['terminal_diagnostics'][name],'dynamics':dynamic}
        if np.any(mask):raise ValueError('Passive material probe contains controlled vertices')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh audit output; preserve prior evidence')
    result=audit(args.native_report);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'samples':result['native_samples'],'cloth_ids':list(result['cloths'])}))


if __name__=='__main__':main()
