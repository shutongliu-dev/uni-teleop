"""Discrete robot reach at native-recorded object mechanism stages.

This is a read-only geometry audit. It does not run a grasp or a swept path,
modify a native episode, or infer successful robot manipulation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms import bimanual_layout
from scripts.diagnostics.check_bimanual_arm_self_collision import ArmSelfCollision
from scripts.diagnostics.check_bimanual_options import (
    CollisionBounds, asset_record, content_digest, digest, gaps, solve_target, transformed_bounds,
)


def separating_axis_gap(first,second):
    """Conservative distance lower bound between two transformed shape boxes.

    Corners use itertools.product order. Full face normals and edge crosses
    also handle affine boxes; an enclosing box is never shrunk to a mesh.
    """
    a,b=np.asarray(first),np.asarray(second)
    if a.shape!=(8,3) or b.shape!=(8,3) or not np.isfinite([a,b]).all():
        raise ValueError('Finite eight-corner bounds required')
    ea=a[[4,2,1]]-a[0];eb=b[[4,2,1]]-b[0]
    axes=[np.cross(edges[i],edges[j]) for edges in (ea,eb) for i,j in ((0,1),(0,2),(1,2))]
    axes.extend(np.cross(x,y) for x in ea for y in eb)
    maximum=-float('inf')
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-14: continue
        axis=axis/length;pa=a@axis;pb=b@axis
        maximum=max(maximum,float(pb.min()-pa.max()),float(pa.min()-pb.max()))
    if not np.isfinite(maximum): raise ValueError('Degenerate bounds')
    return maximum


class RecordedCollisionBounds(CollisionBounds):
    """Refine only coarse environment AABB hits with enclosing oriented boxes."""
    def __init__(self,*args,**kwargs):
        self.corner_cache={}
        super().__init__(*args,**kwargs)

    def boxes(self,side,joints):
        poses=self.poses(side,joints)
        corners=np.asarray([(poses[name]@c.T).T[:,:3] for name,c in self.robot_shapes[side]])
        bounds=np.asarray([[c.min(0),c.max(0)] for c in corners])
        # Retain the array so Python cannot reuse an identity while cached.
        self.corner_cache[id(bounds)]=(bounds,corners)
        return bounds

    def check(self,boxes):
        result=super().check(boxes)
        for side,bounds in boxes.items():
            row=result['checks'][side+'_environment']
            if row['clear']: continue
            corners=self.corner_cache[id(bounds)][1]
            near=np.argwhere(gaps(bounds,self.obstacles)<=.001)
            unresolved=[];minimum=float('inf')
            for i,j in near:
                gap=separating_axis_gap(corners[i],self.obstacle_corners[j])
                minimum=min(minimum,gap)
                if gap<=.001:
                    unresolved.append({'robot_shape':self.shape_names[side][i],
                        'environment_shape':self.obstacle_names[j],
                        'separating_axis_gap_lower_bound_m':gap})
            row.update(clear=not unresolved,oriented_bound_pairs_checked=len(near),
                       minimum_separating_axis_gap_m=minimum,unresolved_oriented_bounds=unresolved,
                       method='AABB broad phase; enclosing shape boxes with face/edge separating axes')
        result['clear']=all(row['clear'] for row in result['checks'].values())
        return result


def interaction_targets(task):
    """Require one explicit authored target list, not guessed asset centers."""
    candidates=[]
    def visit(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key=='interaction_targets': candidates.append(child)
                else: visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    visit(task)
    unique={content_digest(value):value for value in candidates}
    if len(unique)!=1: raise ValueError('Expected one unambiguous interaction_targets list')
    targets=next(iter(unique.values()))
    if not targets or len({row['label'] for row in targets})!=len(targets):
        raise ValueError('Nonempty unique interaction target labels required')
    return {row['label']:row for row in targets}


def target_point(target,body_poses,*,stage=None,static_paths=None):
    point=np.asarray(target['reference_position_m'],dtype=float)
    if 'body_id' in target:
        bodies={row['id']:row for row in body_poses}
        if target['body_id'] in bodies:
            body=bodies[target['body_id']]
            q=body['quaternion_wxyz']
            point=(Rotation.from_quat([*q[1:],q[0]]).apply(target['local_position_m'])
                   +np.asarray(body['position_m']))
        else:
            from pxr import UsdGeom,UsdPhysics
            if stage is None or target['body_id'] not in (static_paths or {}):
                raise ValueError('Target has no recorded body or declared static fixture')
            prim=stage.GetPrimAtPath(static_paths[target['body_id']])
            if not prim or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Static target must refer to a fixed fixture')
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            point=(matrix@[*target['local_position_m'],1.])[:3]
    # Legacy `approach_direction` denotes motion toward the target; the new
    # explicit world direction denotes the outward free-space offset.
    direction=np.asarray(target.get('approach_direction_world',
        -np.asarray(target.get('approach_direction',[0.,0.,-1.]))),dtype=float)
    distance=float(target.get('approach_distance_m',.045))
    if (point.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(point).all()
            or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1.)>1e-5
            or not np.isfinite(distance) or not .0<distance<=.15):
        raise ValueError('Invalid authored approach target: '+target['label'])
    return point,point+direction*distance


APPROACH_RULE='declared_static_enclosure_exact_blockers_v1'
STATIC_CONTIGUITY_M=1e-6
EXACT_CONTACT_M=1e-6


def free_surface_approach(checker,reference,point,*,owner_paths=(),static_owner_paths=None,exact_clearance=None):
    """Place a free approach outside the enclosing shape along the authored ray.

    A tube center or socket center is an object goal, not a free-space fingertip
    location. Retain that exact reference and derive the surface from recorded
    geometry, with the original outward offset. This never proves a grasp.

    Without the two keyword rules below the result is the legacy enclosing-box
    one. ``static_owner_paths`` are declared fixed fixtures: the colliders of one
    fixture form one static owner, but its enclosure only continues through its
    own colliders that adjoin it along the ray (a shaft stacked on its pad), never
    across a gap (a shelf above), another fixture or any rigid body.
    ``exact_clearance(name, start, end)`` returns the exact source-solid distance
    of the approach segment, or None without an exact solid: a forward enclosing
    box then blocks only if the segment touches that solid (a bolt hole).
    """
    offset=np.linalg.norm(point-reference);direction=(point-reference)/offset
    surface=0.;shapes=[];hits=[]
    static_roots=tuple(static_owner_paths or ())
    for name,corners in zip(checker.obstacle_names,checker.obstacle_corners):
        if name.startswith('/World/Environment/'): continue
        basis=(corners[[4,2,1]]-corners[0]).T
        if abs(np.linalg.det(basis))<1e-16: continue
        origin=np.linalg.solve(basis,reference-corners[0]);ray=np.linalg.solve(basis,direction)
        enter,leave=-float('inf'),float('inf')
        for value,slope in zip(origin,ray):
            if abs(slope)<1e-12:
                if not -1e-9<=value<=1.+1e-9: enter,leave=1.,0.;break
            else:
                ends=sorted((-value/slope,(1.-value)/slope))
                enter=max(enter,ends[0]);leave=min(leave,ends[1])
        if enter<=leave and leave>=0.:
            owners=[path for path in owner_paths if name==path or name.startswith(path+'/')]
            fixtures=[path for path in static_roots if name==path or name.startswith(path+'/')]
            owner=max(owners,key=len) if owners else min(fixtures,key=len) if fixtures else name
            hits.append((name,owner,float(enter),float(leave)))
    # Only the body containing the reference can supply its enclosing surface.
    # Include its own cap/handle shapes, never leap over an unrelated lid.
    containing={owner for _,owner,enter,leave in hits if enter<=1e-9 and leave>=0.}
    static={owner for _,owner,_,_ in hits if owner in static_roots}
    enclosure={name for name,owner,_,_ in hits if owner in containing and owner not in static}
    contiguous=[]
    for root in sorted(containing&static):
        members=sorted((row for row in hits if row[1]==root),key=lambda row:(row[2],row[3]))
        chain={name for name,_,enter,_ in members if enter<=1e-9}
        exits=[leave for name,_,enter,leave in members if enter<=1e-9 and leave<=.25]
        end=max(exits) if exits else None
        for name,_,enter,leave in members:
            if end is None or enter>end+STATIC_CONTIGUITY_M: break
            if name not in chain and leave<=.25:
                chain.add(name);contiguous.append(name);end=max(end,leave)
        enclosure|=chain
    for name,owner,enter,leave in hits:
        if name in enclosure and 0.<=leave<=.25 and leave>surface:
            surface=leave;shapes=[name]
    blockers=[name for name,owner,enter,leave in hits
              if name not in enclosure and enter<surface+offset and leave>0.]
    approach=reference+direction*(surface+offset)
    result={
        'authored_offset_m':float(offset),'enclosing_surface_distance_m':surface,
        'surface_shapes':shapes,'containing_owner_paths':sorted(containing),
        'forward_non_target_shapes':blockers,'blocked':bool(blockers),
        'maximum_reference_to_surface_m':.25,
        'scope':'Free fingertip approach beyond recorded shape enclosure; object/tool goal remains unchanged.'}
    if static_owner_paths is None and exact_clearance is None:
        return approach,result
    cleared,unresolved,remaining=[],[],[]
    for name in blockers:
        if exact_clearance is None:
            remaining.append(name);continue
        exact=exact_clearance(name,reference,approach)
        if exact is None or 'segment_distance_m' not in exact:
            remaining.append(name)
            unresolved.append({'prim_path':name,**(exact or {'reason':'No exact source solid; enclosing box kept.'})})
        elif exact['segment_distance_m']>EXACT_CONTACT_M:
            cleared.append({'prim_path':name,**exact})
        else:
            remaining.append(name)
    result.update(forward_non_target_shapes=remaining,blocked=bool(remaining),
        approach_rule=APPROACH_RULE,static_owner_roots_hit=sorted(static),
        static_contiguous_enclosure_shapes=sorted(contiguous),
        static_contiguity_tolerance_m=STATIC_CONTIGUITY_M,
        enclosing_box_forward_shapes=blockers,exact_clear_forward_shapes=cleared,
        exact_unresolved_forward_shapes=unresolved,exact_contact_tolerance_m=EXACT_CONTACT_M,
        approach_rule_scope='Declared fixed fixtures continue the enclosure only through their own adjoining '
            'colliders; a forward enclosing box with a cube/convexHull/closed SDF or triangle-mesh source solid '
            'blocks only if the approach segment touches that solid. Other shapes keep the enclosing box.')
    return approach,result


def actual_single_stand(scene):
    """Read the selected frozen fixture rather than a preview layout helper."""
    rows=[row for row in scene.get('fixtures',[]) if 'stand' in row['id']]
    if len(rows)!=1: raise ValueError('Expected one frozen single-arm stand fixture')
    row=rows[0];root,manifest,asset=asset_record(row);boxes=[]
    sources={str(manifest):digest(manifest)}
    for part in asset['collision_parts']:
        if 'mesh_path' in part:
            import trimesh
            path=root/part['mesh_path'];sources[str(path)]=digest(path)
            if sources[str(path)]!=part['sha256']: raise ValueError('Changed stand collision mesh')
            bounds=np.asarray(trimesh.load(path,force='mesh',process=False).bounds)
        elif part['geometry']=='box':
            half=np.asarray(part['half_extents_m']);bounds=np.array([-half,half])
        else: raise ValueError('Unsupported frozen stand collision shape')
        scale=np.asarray(part.get('scale',[1.,1.,1.]))
        if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale<=0.):
            raise ValueError('Expected finite positive stand mesh scale')
        bounds=transformed_bounds(bounds*scale,part.get('position_m',[0.,0.,0.]),part.get('quaternion_xyzw',[0.,0.,0.,1.]))
        q=row.get('quaternion_wxyz',[1.,0.,0.,0.])
        bounds=transformed_bounds(bounds,row['position_m'],[*q[1:],q[0]])
        boxes.append({'name':part['label'],'position_m':bounds.mean(0).tolist(),
                      'size_m':(bounds[1]-bounds[0]).tolist()})
    return {'boxes':boxes,'frozen_fixture':deepcopy(row),'asset_dependencies':sources,
            'scope':'Conservative bounds of the selected release actual stand collision meshes.'}


def validate_recorded_stages(saved,scene,task):
    """Match every saved pose/time/actor binding to its actual raw trace row."""
    sources={str(Path(row['path']).resolve()):row for row in saved['native_trace_files']}
    declared=(scene.get('native_task_bundle',{}).get('rigid_bodies')
              or task.get('evaluation',{}).get('body_paths'))
    if not isinstance(declared,dict): raise ValueError('Missing declared dynamic actor bindings')
    pending={path:{} for path in sources}
    for row in saved['workspace_stages']:
        if (type(row['trace_frame_index']) is not int or row['trace_frame_index']<0
                or not isinstance(row['sim_time_s'],(float,int)) or not np.isfinite(row['sim_time_s'])
                or row['sim_time_s']<0.):
            raise ValueError('Finite stage time and nonnegative integer frame index required')
        path=row.get('native_trace_path')
        if path is None:
            if len(sources)!=1: raise ValueError('Ambiguous stage native trace')
            path=next(iter(sources))
        path=str(Path(path).resolve())
        pending[path].setdefault(row['trace_frame_index'],[]).append(row)
    count=0
    for path,indices in pending.items():
        if not indices: continue
        opener=gzip.open if path.endswith('.gz') else open
        with opener(path,'rt') as stream:
            for index,line in enumerate(stream):
                if index not in indices: continue
                frame=json.loads(line)
                for stage in indices.pop(index):
                    if not np.isfinite(frame['time_s']) or abs(frame['time_s']-stage['sim_time_s'])>1e-9:
                        raise ValueError('Stage time differs from native frame')
                    if (set(frame['objects'])!=set(declared)
                            or {p['id'] for p in stage['body_poses']}!=set(frame['objects'])
                            or len(stage['body_poses'])!=len(declared)):
                        raise ValueError('Stage does not cover every native body')
                    for pose in stage['body_poses']:
                        if declared[pose['id']]!=pose['prim_path']:
                            raise ValueError('Stage actor path differs from selected scene')
                        raw=frame['objects'][pose['id']]
                        if any(pose[key]!=raw[key] for key in ('position_m','quaternion_wxyz')):
                            raise ValueError('Stage pose differs from native frame')
                    count+=1
                if not indices: break
        if indices: raise ValueError('Native trace is missing a selected frame')
    return {'passed':True,'stages_checked':count,'pose_comparison':'exact serialized numeric values',
            'time_tolerance_s':1e-9,'actor_paths_bound_to_selected_release':True}


def installation(scene,initial,family,task_id,cid):
    robot=scene['robot']
    model=public_robot_geometry(robot_configuration(scene))
    models={'right':model} if cid=='single_arm' else model.sides
    if cid=='single_arm':
        stands={'right':actual_single_stand(scene)}
        q=robot['base_quaternion_wxyz']
        yaws={'right':float(Rotation.from_quat([*q[1:],q[0]]).as_euler('xyz')[2])}
    else:
        expected_robot,expected_initial,_=bimanual_layout.robot_inputs(family,task_id,cid)
        if robot!=expected_robot or initial['joint_positions_rad']!=expected_initial['joint_positions_rad']:
            raise ValueError('Selected release differs from the authored bimanual installation')
        stands=bimanual_layout.stand_geometry(family,task_id,cid)
        yaws={s:p['base_yaw_rad'] for s,p in bimanual_layout.placements(family,task_id,cid).items()}
    return models,stands,yaws


def check_shared_space(checker,arm_check,models,joints,yaws,ready_boxes,center):
    """Check a free shared approach and a simultaneous separated hand pair."""
    hands={s:solve_target(models[s],joints,center,s,yaws[s],'handoff',checker,ready_boxes,
                         joint_check=arm_check.check) for s in models}
    shared=all(row['converged'] and row['clearance_passed'] for row in hands.values())
    axis=models['left'].arm_base[:3,3]-models['right'].arm_base[:3,3]
    axis[2]=0.;axis/=np.linalg.norm(axis)
    pairs=[]
    for distance in (.14,.22,.28):
        pair={s:solve_target(models[s],joints,center+sign*distance*axis,s,yaws[s],
                            'handoff',checker,ready_boxes,joint_check=arm_check.check)
              for s,sign in (('left',1.),('right',-1.))}
        actual_clearance=None
        if all(row['converged'] and row['clearance_passed'] for row in pair.values()):
            actual_clearance=checker.check({s:checker.boxes(s,row['joint_positions_rad']) for s,row in pair.items()})
        passed=actual_clearance is not None and actual_clearance['clear']
        pairs.append({'half_separation_m':distance,'hands':pair,'pair_clearance':actual_clearance,'passed':passed})
        if passed: break
    return {'center_m':center.tolist(),'shared_point_hands':hands,
            'shared_point_reachable_by_both':shared,'separated_pairs':pairs,
            'simultaneous_pair_passed':any(row['passed'] for row in pairs),
            'passed':shared and any(row['passed'] for row in pairs)}


def run(release,family,task_id,cid,stages_path):
    release,stages_path=Path(release),Path(stages_path)
    _,scene,task,initial=task_inputs(release,task_id,seed=17)
    targets=interaction_targets(task)
    saved=load_json(stages_path)
    stages=saved['workspace_stages']
    if not stages or len({row['label'] for row in stages})!=len(stages):
        raise ValueError('Nonempty unique recorded stages required')
    traces=saved['native_trace_files']
    if not traces: raise ValueError('Recorded stages require raw trace provenance')
    for entry in traces:
        if digest(entry['path'])!=entry['sha256']: raise ValueError('Changed raw trace')
    trace_binding=validate_recorded_stages(saved,scene,task)
    models,stands,yaws=installation(scene,initial,family,task_id,cid)
    joints=initial['joint_positions_rad']
    dynamic_paths=(scene.get('native_task_bundle',{}).get('rigid_bodies')
                   or task.get('evaluation',{}).get('body_paths'))
    static_paths=task.get('asset_metadata',{}).get('fixtures',{})
    owners=tuple(dynamic_paths.values())+tuple(static_paths.values())
    rows=[];dependencies={p:h for stand in stands.values() for p,h in stand.get('asset_dependencies',{}).items()};seen_targets=set()
    for stage in stages:
        if not stage['body_poses'] or stage['trace_frame_index']<0 or stage['sim_time_s']<0.:
            raise ValueError('Missing recorded body stage')
        checker=RecordedCollisionBounds(scene,scene['robot'],models,joints,stands,body_poses=stage['body_poses'])
        arm_check=ArmSelfCollision(checker)
        dependencies.update(checker.dependencies)
        ready={s:checker.boxes(s,joints) for s in models}
        ready_clear,stand_clear,self_clear=checker.check(ready),checker.stand_check(),arm_check.check(joints)
        samples=[]
        for label in stage['target_labels']:
            target=targets[label];seen_targets.add(label)
            reference,point=target_point(target,stage['body_poses'],stage=checker.stage,
                                         static_paths=static_paths)
            point,surface=free_surface_approach(checker,reference,point,owner_paths=owners)
            hands={s:solve_target(models[s],joints,point,s,yaws[s],target.get('approach','above'),
                                 checker,ready,joint_check=arm_check.check) for s in models}
            usable=[s for s,row in hands.items() if row['converged'] and row['clearance_passed'] and not surface['blocked']]
            samples.append({'label':label,'authored_target':target,'reference_position_m':reference.tolist(),
                            'fingertip_target_m':point.tolist(),'surface_approach':surface,
                            'hands':hands,'usable_sides':usable,'covered':bool(usable)})
        rows.append({'recorded_stage':stage,'samples':samples,'ready_clearance':ready_clear,
                     'stand_clearance':stand_clear,'ready_arm_self_collision':self_clear,
                     'passed':bool(samples) and all(row['covered'] for row in samples)
                              and ready_clear['clear'] and stand_clear['clear'] and self_clear['clear']})
    missing=sorted(set(targets)-seen_targets)
    common=[]
    if len(models)==2:
        # The interaction center is authored in the checkpoint index; candidate
        # heights are explicit and all failed attempts are retained.
        for center in saved['shared_approach_candidates_m']:
            row=check_shared_space(checker,arm_check,models,joints,yaws,ready,np.asarray(center,dtype=float))
            common.append(row)
            if row['passed']: break
    sample_count=sum(len(row['samples']) for row in rows)
    covered=sum(sample['covered'] for row in rows for sample in row['samples'])
    return {'schema':'recorded_interaction_workspace.v1','family':family,'task_id':task_id,
        'configuration_id':cid,'release':str(release),'source_files':{
            name:digest(release/name) for name in ('release_manifest.json','scene.json','task.json','initial_state.json')},
        'stages_source':{'path':str(stages_path),'sha256':digest(stages_path)},'native_trace_files':traces,
        'recorded_stage_binding':trace_binding,
        'configuration_binding':{'robot_sha256':content_digest(scene['robot']),
                                 'initial_state_sha256':content_digest(initial),'stand_geometry_sha256':content_digest(stands)},
        'asset_dependencies':dependencies,'diagnostic_source_sha256':digest(__file__),
        'initial_state':initial,'robot_configuration':scene['robot'],'stand_geometry':stands,
        'stages':rows,'sample_count':sample_count,'covered_count':covered,'missing_target_labels':missing,
        'shared_approach_attempts':common,
        'passed':not missing and all(row['passed'] for row in rows) and (not common or any(row['passed'] for row in common)),
        'native_steps':0,'model_calls':0,'robot_manipulation':False,'continuous_path_qualified':False,
        'scope':'Native-recorded object poses replayed only in an anonymous offline USD stage. Discrete original-hand middle-fingertip approaches, original-shape environment/stand/inter-arm bounds and nonadjacent FR3 convex self-collision; excludes hand internal collision, grasp, held-object sweep and continuous paths.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--family',required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--configuration',choices=('single_arm','dual_same_side','dual_opposed'),required=True)
    p.add_argument('--stages',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve previous audit')
    if not a.output.resolve().is_relative_to(OUTPUT_ROOT): raise ValueError('Evidence belongs under datasets')
    result=run(a.release,a.family,a.task,a.configuration,a.stages)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('task_id','configuration_id','passed','sample_count','covered_count')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
