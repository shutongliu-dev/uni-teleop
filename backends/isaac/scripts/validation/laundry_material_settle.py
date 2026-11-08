"""Passive, fully recorded material probes for one or two named laundry cloths.

Run each case in a fresh native process. No grasp, attachment, cloth state setter,
policy, or task success is created. Single-cloth reports retain the native reset
export format; two-cloth streams retain independent named q/qd and control masks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_single_garment_native import (
    DATASETS, NativeArrays, advance, dump, pin, robot_readback)

PROJECT = Path(__file__).resolve().parents[2]


def validate_probe_scene(scene):
    specs = scene.get('cloths', [])
    robot = scene.get('robot', {})
    # Public robot inputs omit enabled when enabled by default. Explicit
    # False and the absence of an installed hand remain invalid here.
    if len(specs) not in (1, 2) or not robot.get('enabled', True) or not robot.get('hand_id'):
        raise ValueError('One or two independent cloths and an actual supported robot installation required')
    if scene.get('named_cloth_initialization') != 'author-only':
        raise ValueError('Preserve authored native cloth initialization; no reset setters')
    return specs


def mesh_data(spec, task):
    from bench.simulation.isaacsim_backend.geometry import cloth_grid
    rest, triangles = cloth_grid(spec)
    rest = rest.astype(float)
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]],
        triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
    area = np.linalg.norm(np.cross(rest[triangles[:, 1]]-rest[triangles[:, 0]],
        rest[triangles[:, 2]]-rest[triangles[:, 0]]), axis=1)/2
    weights = np.zeros(len(rest))
    for corner in range(3):
        np.add.at(weights, triangles[:, corner], area/3)
    weights /= weights.sum()
    pairs = np.empty((0, 2), dtype=int)
    basis = 'No paired layers in this unfolded single-surface reset.'
    profile = task.get('asset_metadata', {}).get('mesh_profile')
    if profile in ('procedural_flat_sewn_shirt_v1', 'procedural_sewn_trousers_v1'):
        half = len(rest)//2
        if len(rest) % 2 or not np.array_equal(rest[:half, :2], rest[half:, :2]):
            raise ValueError('Sewn panel material correspondence changed')
        pairs = np.column_stack([np.arange(half, len(rest)), np.arange(half)])
        basis = 'Exact authored opposing panels; signed initial vertical material separation.'
    elif task['id'] == 'towel_stacking':
        nx, ny = spec['cells_xy']
        # Exclude the circular fold itself; compare the two flat material wings.
        pairs = np.array([[row*(nx+1)+i, row*(nx+1)+nx-i]
            for row in range(ny+1) for i in range(nx//2)
            if (nx/2-i)*spec['size_m'][0]/nx > .02], dtype=int)
        basis = 'Original U-fold mirrored X columns, excluding the 20 mm crease region.'
    return {'rest': rest, 'triangles': triangles, 'edges': edges,
        'edge_lengths': np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1),
        'areas': area, 'weights': weights, 'pairs': pairs, 'pair_basis': basis}


def metrics(q, qd, mesh, thickness, table_z, rect):
    q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    edge, tri = mesh['edges'], mesh['triangles']
    area = np.linalg.norm(np.cross(q[tri[:, 1]]-q[tri[:, 0]],
        q[tri[:, 2]]-q[tri[:, 0]]), axis=1)/2
    inside = ((q[:, 0] >= rect[0]) & (q[:, 0] <= rect[1]) &
              (q[:, 1] >= rect[2]) & (q[:, 1] <= rect[3]))
    depth = np.where(inside, np.maximum(table_z+thickness/2-q[:, 2], 0.), 0.)
    result = {'edge_ratio_max': float((np.linalg.norm(q[edge[:, 1]]-q[edge[:, 0]], axis=1)/mesh['edge_lengths']).max()),
        'triangle_area_ratio_min': float((area/mesh['areas']).min()),
        'table_penetration_max_m': float(depth.max()),
        'table_penetration_nodes_over_2mm': int(np.count_nonzero(depth > .002)),
        'height_range_m': float(np.ptp(q[:, 2])), 'height_above_table_max_m': float(q[:, 2].max()-table_z),
        'raw_speed_p95_m_s': float(np.quantile(np.linalg.norm(qd, axis=1), .95)),
        'raw_speed_max_m_s': float(np.linalg.norm(qd, axis=1).max()),
        'minimum_midplane_z_m': float(q[:, 2].min()),
        'com_z_m': float(np.dot(mesh['weights'], q[:, 2])),
        'com_vz_m_s': float(np.dot(mesh['weights'], qd[:, 2])),
        'kinetic_energy_per_mass_j_kg': float(.5*np.dot(mesh['weights'], np.sum(qd*qd, axis=1)))}
    pairs = mesh['pairs']
    if len(pairs):
        gap = q[pairs[:, 1], 2]-q[pairs[:, 0], 2]
        result.update(pair_gap_min_m=float(gap.min()), pair_gap_p05_m=float(np.quantile(gap, .05)),
            pair_gap_median_m=float(np.median(gap)), pair_gap_fraction_above_45pct_thickness=float(np.mean(gap >= .45*thickness)))
    return result


def usd_material_readback(env, spec, mesh):
    info = env.scene_info['cloths'][spec['id']]
    material = env.stage.GetPrimAtPath(info['material_path'])
    prim = env.stage.GetPrimAtPath(info['path'])
    values = {name: material.GetAttribute(name).Get() for name in info['material']}
    collisions = {name: prim.GetAttribute(name).Get() for name in (
        'physxCollision:contactOffset', 'physxCollision:restOffset',
        'physxDeformableBody:selfCollision', 'physxDeformableBody:selfCollisionFilterDistance',
        'physxDeformableBody:solverPositionIterationCount', 'physxDeformableBody:linearDamping')}
    mass = float(values['omniphysics:density']*values['omniphysics:surfaceThickness']*mesh['areas'].sum())
    if not math.isclose(mass, spec['mass_kg'], rel_tol=2.e-5, abs_tol=1.e-8):
        raise ValueError('Actual authored USD density/area/thickness disagree with nominal mass')
    return {'body_path': info['path'], 'material_path': info['material_path'], 'properties': values,
        'contact_and_body_properties': collisions, 'nominal_mass_kg': spec['mass_kg'],
        'derived_usd_mass_kg': mass, 'rest_area_m2': float(mesh['areas'].sum()),
        'visual_skin': info.get('visual_skin'),
        'scope': 'Actual USD material attributes read after native initialization; not a calibrated force/mass measurement.'}


def run(args):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.laundry_material_settle',), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Use a complete frozen source snapshot')
    task, scene, initial = [load_json(p) for p in (args.task, args.scene, args.initial)]
    specs = validate_probe_scene(scene)
    dt = float(scene['physics']['dt_s']);steps = round(args.duration/dt)
    if args.duration < .5 or abs(steps*dt-args.duration) > 1.e-9:
        raise ValueError('Probe needs at least 0.5 s and an integer native step count')
    output = args.output.resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh shared dataset output for this independent process')
    output.mkdir(parents=True)
    meshes = {s['id']: mesh_data(s, task) for s in specs}
    one = len(specs) == 1
    fields = {s['id']: {'q': 'cloth_q' if one else 'cloth_'+s['id']+'_q',
        'qd': 'cloth_qd' if one else 'cloth_'+s['id']+'_qd',
        'mask': 'control_mask' if one else 'control_mask_'+s['id']} for s in specs}
    report = {'schema': 'laundry_single_garment_native.v1' if one else 'laundry_material_native_settle.v1',
        'complete': False, 'mode': 'settle', 'source': source,
        'source_manifest': pin(PROJECT/'source_snapshot_manifest.json'),
        'scene_input': pin(args.scene), 'initial_input': pin(args.initial), 'task_input': pin(args.task),
        'cloth_ids': list(fields), 'cloth_stream_fields': fields,
        'robot_present': True, 'robot_control': 'original native joint position hold and gravity feedforward',
        'cloth_position_velocity_writes': 0, 'attachments_created': 0, 'controlled_cloth_vertices': 0,
        'initialization_is_operation_credit': False, 'task_success_claimed': False,
        'robot_grasp_qualified': False, 'physics_accepted': False, 'model_calls': 0,
        'scope': 'Independent passive material/reset probe. No task operation or inherited physics/grasp qualification.'}
    env = capture = writer = None;exit_code = 1
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        env = FR3DexterousIsaacEnv(scene);env.reset(initial_state=initial)
        if not math.isclose(env.dt, dt, abs_tol=1.e-12, rel_tol=0):
            raise ValueError('Native and declared dt differ')
        report['native_initialization'] = env.named_cloth_initialization_receipt
        report['actual_attachment_prims'] = [str(p.GetPath()) for p in env.stage.Traverse()
            if any('Attachment' in schema for schema in p.GetAppliedSchemas())]
        if report['actual_attachment_prims']:
            raise ValueError('Passive material scene contains an attachment')
        report['joint_names'], report['body_names'] = env.joint_names, env.body_names
        report['runtime_workroom_material_check'] = deepcopy(env.runtime_workroom_material_check)
        report['runtime_robot_material_check'] = deepcopy(env.runtime_robot_material_check)
        report['workroom_surface_materials'] = deepcopy(
            env.scene_info.get('workbench', {}).get('surface_materials'))
        report['actual_usd_materials'] = {s['id']: usd_material_readback(env, s, meshes[s['id']]) for s in specs}
        from bench.evaluation.cloth.cloth_robot_collision import collect_colliders
        dump(output/'robot_colliders.json', collect_colliders(env))
        report['robot_colliders'] = pin(output/'robot_colliders.json')
        report['layer_pair_diagnostics'] = {name: {'pairs': mesh['pairs'].tolist(), 'basis': mesh['pair_basis']}
                                            for name, mesh in meshes.items()}
        writer = NativeArrays(output/'states', steps+1)
        start_step = int(env.world.current_time_step_index);start_clock = float(env.world.current_time)
        rect = task['asset_metadata']['clear_rectangle_xy_m'];table = task['asset_metadata']['table_top_z_m']
        def state(step):
            result = {**robot_readback(env), 'time_s': np.asarray(step*dt),
                'native_step': np.asarray(step,dtype=np.int64),
                'world_step': np.asarray(env.world.current_time_step_index,dtype=np.int64),
                'world_time_s': np.asarray(env.world.current_time)}
            for spec in specs:
                name=spec['id'];view=env._cloth_views[name];q=_np(view.get_simulation_nodal_positions())[0]
                qd=_np(view.get_simulation_nodal_velocities())[0];f=fields[name]
                result.update({f['q']:q, f['qd']:qd, f['mask']:np.zeros(len(q),dtype=np.uint8)})
                for key,value in metrics(q,qd,meshes[name],spec['visual_thickness_m'],table,rect).items():
                    result['metric_'+name+'_'+key]=np.asarray(value)
            return result
        writer.append(state(0));env.stage.Export(str(output/'initial.usda'))
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            capture= LaundryClothFrameCapture(stage=env.stage,world=env.world,app=env.app,
                cloth_views=env._cloth_views,cloth_specs=specs,triangles=env.cloth_triangles,
                cloth_paths={k:v['path'] for k,v in env.scene_info['cloths'].items()},
                output=args.media_output,cameras={key:scene['cameras'][key] for key in ('overview','task')})
        def photo(tag):
            before=state(0);capture.capture(tag,camera_ids=('overview','task'));after=state(0)
            if any(not np.array_equal(before[k],after[k]) for k in before):
                raise RuntimeError('Rendering changed native cloth/robot state')
        if capture:photo('initial')
        last=time.monotonic()
        for step in range(1,steps+1):
            advance(env);writer.append(state(step))
            if time.monotonic()-last>20 or step==steps:
                print(json.dumps({'event':'material_settle_progress','step':step,'time_s':step*dt,'target_time_s':args.duration}),flush=True);last=time.monotonic()
        if int(env.world.current_time_step_index)-start_step!=steps or abs(float(env.world.current_time)-start_clock-args.duration)>2.e-6:
            raise RuntimeError('Recorded and actual native clocks differ')
        if capture:photo('final')
        env.stage.Export(str(output/'final.usda'))
        first=max(0,steps-round(.5/dt));report['terminal_diagnostics']={}
        from bench.evaluation.cloth.audit_cloth_self_intersection import self_intersections
        for spec in specs:
            name=spec['id'];f=fields[name];q=np.asarray(writer.arrays[f['q']][steps],dtype=float)
            drift=np.linalg.norm(q-np.asarray(writer.arrays[f['q']][first],dtype=float),axis=1)
            pairs,_=self_intersections(q,meshes[name]['triangles'])
            report['terminal_diagnostics'][name]={'window_s':(steps-first)*dt,
                'drift_p95_m':float(np.quantile(drift,.95)),'drift_max_m':float(drift.max()),
                'self_crossing_pairs':pairs,'self_crossing_scope':'Exact final sample only; no continuous contact-force or crossing qualification.',
                'metrics':{key:float(value[steps]) for key,value in writer.arrays.items() if key.startswith('metric_'+name+'_')}}
        report.update(complete=True,native_steps=steps,duration_s=steps*dt)
        if one:report['cloth_id']=specs[0]['id']
        exit_code=0
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()};traceback.print_exc()
    finally:
        if writer is not None:report['states']=writer.close()
        if capture is not None:capture.close()
        dump(output/'native_report.json',report)
        print(json.dumps({'event':'material_settle_finished','complete':report['complete'],'output':str(output)}),flush=True)
        if env is not None:
            if exit_code:
                try:raise RuntimeError('Passive material probe failed: '+str(output/'native_report.json'))
                finally:env.close()
            env.close()
    return exit_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=3.);p.add_argument('--media-output',type=Path)
    raise SystemExit(run(p.parse_args()))


if __name__=='__main__':main()
