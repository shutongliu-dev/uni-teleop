"""Recheck saved discrete IK states against original nonadjacent FR3 convexes.

This is a companion audit; it never rewrites workspace reports or releases.
Hand internal collisions and continuous swept motion are outside its scope.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.paths import load_json, OUTPUT_ROOT
from scripts.diagnostics.check_bimanual_options import CollisionBounds, content_digest, digest


def intersection_radius(planes):
    """Largest sphere wholly inside the intersection of two convex solids."""
    planes=np.asarray(planes)
    result=linprog([0.,0.,0.,-1.],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],
        b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0.,None)],method='highs')
    if result.status==2:return 0.,None
    if not result.success:raise RuntimeError('Convex intersection LP failed: '+result.message)
    return float(result.x[3]),result.x[:3].tolist()


class ArmSelfCollision:
    def __init__(self,checker):
        from pxr import UsdGeom,UsdPhysics
        self.checker=checker;self.shapes={};self.cache={}
        xforms=UsdGeom.XformCache()
        for side in checker.models:
            shapes=[]
            for path,(link,_) in zip(checker.shape_names[side],checker.robot_shapes[side]):
                prefix = 'fr3_' if len(checker.models) == 1 else 'fr3_'+side+'_'
                match=re.fullmatch(prefix+'link([0-7])',link)
                if not match:continue
                prim=checker.stage.GetPrimAtPath(path)
                if not prim.IsA(UsdGeom.Mesh) or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull':
                    raise ValueError('Expected original FR3 convexHull collision mesh: '+path)
                ancestor=prim
                while ancestor.GetName()!=link:ancestor=ancestor.GetParent()
                relative=np.linalg.inv(np.asarray(xforms.GetLocalToWorldTransform(ancestor)).T)@np.asarray(xforms.GetLocalToWorldTransform(prim)).T
                vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                local=(relative@np.c_[vertices,np.ones(len(vertices))].T).T[:,:3]
                hull=ConvexHull(local)
                shapes.append({'link':link,'index':int(match[1]),'path':path,
                    'vertices':local[hull.vertices],'planes':hull.equations})
            if len(shapes)!=8:raise ValueError('Incomplete FR3 collision shape coverage')
            self.shapes[side]=shapes

    def side(self,side,joints):
        model=self.checker.models[side]
        key=(side,tuple(float(joints[n]) for n in model.arm.names))
        if key in self.cache:return self.cache[key]
        poses=self.checker.poses(side,joints);world=[]
        for shape in self.shapes[side]:
            pose=poses[shape['link']];vertices=(pose[:3,:3]@shape['vertices'].T).T+pose[:3,3]
            a=shape['planes'][:,:3]@pose[:3,:3].T
            planes=np.c_[a,shape['planes'][:,3]-a@pose[:3,3]]
            world.append((shape,np.array([vertices.min(0),vertices.max(0)]),planes))
        intersections=[];broad=0;tested=0
        for (first,a,ap),(second,b,bp) in combinations(world,2):
            if abs(first['index']-second['index'])<=1:continue
            tested+=1
            if not np.all(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])>1e-8):continue
            broad+=1;radius,point=intersection_radius(np.vstack([ap,bp]))
            if radius>.001:
                intersections.append({'links':[first['link'],second['link']],
                    'collision_shapes':[first['path'],second['path']],
                    'intersection_inscribed_radius_m':radius,'interior_witness_m':point})
        result={'clear':not intersections,'nonadjacent_pairs':tested,'aabb_candidates':broad,
            'convex_intersections_above_1mm':intersections}
        self.cache[key]=result;return result

    def check(self,joints):
        rows={s:self.side(s,joints) for s in self.shapes}
        return {'clear':all(r['clear'] for r in rows.values()),'sides':rows}


def audit_report(report,checker):
    states=[]
    def audit(label,joints):
        result=checker.check(joints)
        states.append({'label':label,'joint_positions_sha256':content_digest(joints),**result})
        return result['clear']
    ready=audit('ready',report['initial_state']['joint_positions_rad'])
    covered=[]
    for sample in report['samples']:
        valid=[]
        for side in sample['usable_sides']:
            if audit(sample['label']+'/'+side,sample['hands'][side]['joint_positions_rad']):valid.append(side)
        covered.append({'label':sample['label'],'usable_sides_after_self_check':valid,'covered':bool(valid)})
    common=report['common_workspace']
    shared=[];paired=[]
    for side in ('left','right'):
        shared.append(audit('common_point/'+side,common['shared_point_hands'][side]['joint_positions_rad']))
        paired.append(audit('common_pair/'+side,common['simultaneous_separated_pair'][side]['joint_positions_rad']))
    return {'family':report['family'],'task_id':report['task_id'],'configuration_id':report['configuration_id'],
        'configuration_binding':report['configuration_binding'],'states_checked':len(states),
        'states_flagged':sum(not s['clear'] for s in states),'flagged_states':[s for s in states if not s['clear']],
        'ready_clear':ready,'sample_count':len(covered),'covered_count':sum(r['covered'] for r in covered),
        'samples':covered,'common_point_clear':all(shared),'common_pair_clear':all(paired),
        'passed':ready and all(r['covered'] for r in covered) and all(shared) and all(paired)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audits')
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT):raise ValueError('Evidence belongs under datasets')
    matrix=load_json(args.matrix);cache={};rows=[]
    for source in matrix['rows']:
        path=Path(source['report_path'])
        if digest(path)!=source['report_sha256']:raise ValueError('Changed source workspace report')
        report=load_json(path);key=report['configuration_binding']['robot_sha256']
        if key not in cache:
            scene=load_json(Path(report['release'])/'scene.json')
            if digest(Path(report['release'])/'scene.json')!=report['source_files']['scene.json']:raise ValueError('Changed source scene')
            robot=report['robot_configuration'];models=public_robot_geometry(robot_configuration({'robot':robot})).sides
            cache[key]=ArmSelfCollision(CollisionBounds(scene,robot,models,
                report['initial_state']['joint_positions_rad'],report['stand_geometry']))
        row=audit_report(report,cache[key]);row.update(workspace_report_path=str(path),workspace_report_sha256=digest(path));rows.append(row)
        print(json.dumps({k:row[k] for k in ('family','task_id','configuration_id','passed','states_flagged','covered_count')}),flush=True)
    result={'schema':'bimanual_nonadjacent_arm_convex_audit.v1','passed':all(r['passed'] for r in rows),
        'variant_count':len(rows),'passed_count':sum(r['passed'] for r in rows),
        'states_checked':sum(r['states_checked'] for r in rows),'states_flagged':sum(r['states_flagged'] for r in rows),
        'matrix_path':str(args.matrix),'matrix_sha256':digest(args.matrix),'audit_source_sha256':digest(__file__),
        'rows':rows,'scope':'Original USD convexHull meshes for FR3 link0..7, excluding same and directly adjacent links. AABB broad phase followed by convex-polytope intersection LP; flag only common interior sphere radius >1 mm. Hand internal collisions, hand-vs-arm, adjacent-link design contacts, native cooking and swept paths remain outside this check.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
