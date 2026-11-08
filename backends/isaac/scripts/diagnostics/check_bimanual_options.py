"""Source-bound discrete IK and actual-USD collision bounds for robot options.

No environment or native physics is started. Object/goal points come from the
selected immutable single-arm release; only robot bases and stands vary.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from bench.geometry import rpy_matrix
from scipy.spatial.transform import Rotation
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms import bimanual_layout as layout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def transformed_bounds(bounds, position, quaternion_xyzw=(0.,0.,0.,1.)):
    corners=np.asarray(list(product(*zip(*np.asarray(bounds)))))
    corners=Rotation.from_quat(quaternion_xyzw).apply(corners)+position
    return np.asarray([corners.min(0),corners.max(0)])


def asset_record(row):
    root=Path(row['asset_root']);manifest=root/'manifest.json'
    return root,manifest,load_json(manifest)['assets'][row['asset_id']]


def object_top(row):
    _,_,asset=asset_record(row)
    bounds=asset.get('aabb_m')
    if bounds is None:bounds=[asset['bounds_min_m'],asset['bounds_max_m']]
    q=row.get('quaternion_wxyz',[1.,0.,0.,0.]);bounds=transformed_bounds(bounds,row['position_m'],[*q[1:],q[0]])
    return [*bounds.mean(0)[:2],bounds[1,2]]


def task_points(family, task_id, release):
    """Actual reset object tops plus declared goal/fixture approach locations."""
    release = Path(release)
    scene, task = [load_json(release/name) for name in ('scene.json', 'task.json')]
    samples = []
    seen = set()
    def add(label, point, *, book_front=False):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Invalid task probe point: '+label)
        key = tuple(np.round(p, 5))
        if key in seen: return
        seen.add(key)
        samples.append({'label': label, 'reference_position_m': p.tolist(),
            'fingertip_target_m': (p+([.10,0.,0.] if book_front else [0.,0.,.045])).tolist(),
            'approach': 'shelf_front' if book_front else 'above'})
    evaluation = task['evaluation']
    config = evaluation.get('configuration', {})
    for row in scene.get('objects', []):
        add('supply/'+row['id'], object_top(row))
    for ident, row in config.get('objects', {}).items():
        pos = np.asarray(row['initial_pose']['position_m'])
        q=row['initial_pose'].get('quaternion_wxyz',[1.,0.,0.,0.])
        bounds = transformed_bounds(row['local_bounds_m'],pos,[*q[1:],q[0]])
        shelf = family == 'study' and pos[0] < -.5
        pos=bounds.mean(0)
        if shelf: pos[0] = bounds[1,0]
        else: pos[2] = bounds[1,2]
        add('supply/'+ident, pos, book_front=shelf)
    for ident, row in config.get('targets', {}).items():
        bounds = np.asarray(row['bounds_m'])
        p = bounds.mean(0)
        shelf = family == 'study' and p[0] < -.5
        if shelf: p[0] = bounds[1,0]
        else: p[2] = float(row.get('support_z_m', bounds[0,2]))+.03
        add('target/'+ident, p, book_front=shelf)
    if task_id == 'drawer_retrieval':
        mechanism = config['mechanism']
        p = np.asarray(mechanism['closed_position_m'])+[-.165,0.,0.]
        add('drawer/closed_handle',p)
        add('drawer/open_handle',p+np.asarray(mechanism['opening_direction_world'])*mechanism['travel_m'])
    metadata = task.get('asset_metadata', {})
    for row in metadata.get('objects', []):
        if any(token in row['id'] for token in ('_link_', 'belt_segment_', 'pellet_')): continue
        p = np.asarray(row.get('initial_position_m', row.get('initial_pose_xyzw', [])[:3]))
        if len(p)==3 and 'local_bounds_m' in row:
            bounds=transformed_bounds(row['local_bounds_m'],p,row.get('orientation_xyzw',[0,0,0,1]))
            p=np.asarray([*bounds.mean(0)[:2],bounds[1,2]])
        if len(p)==3: add('supply/'+row['id'],p)
    def goals(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                label = prefix+'/'+key
                if (key in ('center_m','target_position_m','goal_pose_xyzw','initial_position_m') or key.endswith(('_center_m','_target_position_m'))) and isinstance(child,list) and len(child)>=3:
                    add(label,child[:3])
                elif key == 'center_xy_m' and isinstance(child,list) and len(child)==2:
                    add(label,[*child,float(value.get('bottom_z_m', value.get('support_z_m',.90)))])
                elif key not in ('source','metadata','objects','source_bounds_mm','local_bounds_m'):
                    goals(child,label)
        elif isinstance(value,list):
            for index, child in enumerate(value):goals(child,prefix+'/'+str(index))
    goals(metadata.get('task_parameters',{}),'target')
    for ident,p in evaluation.get('target_bottom_centers_m',{}).items(): add('target/'+ident,p)
    for row in scene.get('fixtures',[]):
        if 'stand' in row['id'] or 'support' in row['id'] or row['position_m'][2]<.6:continue
        add('fixture/'+row['id'],row['position_m'])
    if task_id == 'remote_operation':
        add('supply/remote', metadata['remote']['position_m'])
        add('target/holder',metadata['holder']['position_m'])
    if task_id == 'water_pouring':
        for ident,row in metadata['containers'].items():
            p=np.asarray(row['initial_position_m']);p[2]+=max(x[0] for x in row['inner_profile_z_radius_spout_m'])
            add('container/'+ident,p)
    if task_id == 'tower_building':
        candidate=evaluation['candidate_metadata']
        for ident,row in evaluation['reference_configuration']['initial_objects'].items():
            add('supply/'+ident, row['position_m'])
        add('build/center_low',[.65,0.,.83])
        add('build/center_high',[.65,0.,1.13])
    if family=='card_room':
        g=evaluation['geometry']
        for x,y in product(range(3), repeat=2):
            add(f'target/cell_{x}_{y}',[g['center_xy_m'][0]+(x-1)*g['pitch_m'],
                g['center_xy_m'][1]+(y-1)*g['pitch_m'],g['board_top_z_m']+.03])
    if not samples:raise ValueError('No task target coverage: '+task_id)
    return samples, scene, task


def gaps(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if not len(a) or not len(b):return np.empty((len(a),len(b)))
    delta=np.maximum(np.maximum(a[:,None,0]-b[None,:,1],b[None,:,0]-a[:,None,1]),0.)
    return np.linalg.norm(delta,axis=2)


def apply_recorded_body_poses(stage, records):
    """Replay measured poses into an anonymous offline stage, never a live stage.

    Used only to compute bounds at recorded mechanism phases. The source USD
    layers and native physics state remain untouched.
    """
    from pxr import Gf, UsdGeom, UsdPhysics
    if not stage.GetRootLayer().anonymous:
        raise ValueError('Recorded geometry requires an anonymous offline stage')
    seen=set()
    for record in sorted(records,key=lambda row:row['prim_path'].count('/')):
        path=record['prim_path']
        prim=stage.GetPrimAtPath(path)
        if path in seen or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Expected unique recorded rigid body: '+path)
        seen.add(path)
        original=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        if (not np.allclose(original[:3,:3].T@original[:3,:3],np.eye(3),atol=1e-6,rtol=0.)
                or abs(np.linalg.det(original[:3,:3])-1.)>1e-6):
            raise ValueError('Recorded replay supports rigid unit-scale bodies only: '+path)
        position=np.asarray(record['position_m'],dtype=float)
        quaternion=np.asarray(record['quaternion_wxyz'],dtype=float)
        if (position.shape!=(3,) or quaternion.shape!=(4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or abs(np.linalg.norm(quaternion)-1.)>1e-5):
            raise ValueError('Invalid recorded body pose: '+path)
        world=np.eye(4)
        world[:3,:3]=Rotation.from_quat([*quaternion[1:],quaternion[0]]).as_matrix()
        world[:3,3]=position
        parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
        local=np.linalg.inv(parent)@world
        xform=UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix='recorded_geometry_only').Set(Gf.Matrix4d(local.T.tolist()))


class CollisionBounds:
    """Original imported robot collision-shape bounds in each FK link frame."""
    def robot_import(self, side, robot):
        """Overridable diagnostic source selector; default keeps the original assets."""
        single = robot['hand_id'] == 'sharpa_w01_right'
        source=Path(robot['assets_root'])/'robots/assemblies'/('fr3_sharpa' if single else 'dual_fr3_sharpa_w01')/robot['assembly_version']
        return (source/('USD_IMPORT.json' if single else f'USD_IMPORT_{side}.json'),
                '/fr3_sharpa' if single else '/fr3_sharpa_'+side)

    def __init__(self,scene,robot,models,initial,stands,*,body_poses=()):
        from pxr import Usd,UsdGeom,UsdPhysics
        self.models=models;self.stage=Usd.Stage.CreateInMemory()
        self.UsdGeom=UsdGeom
        UsdGeom.Xform.Define(self.stage,'/World')
        env=scene['workroom']
        root=UsdGeom.Xform.Define(self.stage,'/World/Environment').GetPrim()
        root.GetReferences().AddReference(env['usd_path'],env['source_prim'])
        self.dependencies={env['usd_path']:digest(env['usd_path'])}
        if self.dependencies[env['usd_path']]!=env['usd_sha256']:raise ValueError('Changed environment')
        for entry in scene.get('task_layers',[]):
            target=entry.get('target_prim_path',entry['source_prim_path'])
            root=UsdGeom.Xform.Define(self.stage,target).GetPrim()
            root.GetReferences().AddReference(entry['usd_path'],entry['source_prim_path'])
            self.dependencies[entry['usd_path']]=digest(entry['usd_path'])
            if entry.get('usd_sha256',self.dependencies[entry['usd_path']])!=self.dependencies[entry['usd_path']]:
                raise ValueError('Changed task layer')
        native=scene.get('native_task_bundle')
        if native:
            root=UsdGeom.Xform.Define(self.stage,native['root_prim_path']).GetPrim()
            root.GetReferences().AddReference(native['usd_path'],native['root_prim_path'])
            self.dependencies[native['usd_path']]=digest(native['usd_path'])
            if native.get('usd_sha256',self.dependencies[native['usd_path']])!=self.dependencies[native['usd_path']]:
                raise ValueError('Changed native task layer')
        apply_recorded_body_poses(self.stage,body_poses)
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
        def shapes(root):return [p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Boundable)]
        self.obstacles=[];self.obstacle_names=[];self.obstacle_corners=[]
        world_xforms=UsdGeom.XformCache()
        for prim in shapes(self.stage.GetPseudoRoot()):
            b=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            bounds=np.asarray([b.GetMin(),b.GetMax()])
            # Old task stands will be removed in the derivative. All other
            # room and task colliders (including dynamic reset props) remain.
            if 'robot_stand' in str(prim.GetPath()).lower():continue
            if np.isfinite(bounds).all():
                self.obstacles.append(bounds);self.obstacle_names.append(str(prim.GetPath()))
                local=cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
                corners=np.asarray([(*p,1.) for p in product(*zip(local.GetMin(),local.GetMax()))])
                world=np.asarray(world_xforms.GetLocalToWorldTransform(prim)).T
                self.obstacle_corners.append((world@corners.T).T[:,:3])
        for row in scene.get('objects',[])+scene.get('fixtures',[]):
            if 'stand' in row['id']:continue
            root,manifest,asset=asset_record(row);self.dependencies[str(manifest)]=digest(manifest)
            q=row.get('quaternion_wxyz',[1.,0.,0.,0.])
            for part in asset['collision_parts']:
                if 'mesh_path' in part:
                    path=root/part['mesh_path'];actual=digest(path)
                    if actual!=part['sha256']:raise ValueError('Changed task collider: '+str(path))
                    self.dependencies[str(path)]=actual
                    import trimesh
                    bounds=trimesh.load(path,force='mesh',process=False).bounds
                elif part['geometry']=='box':
                    h=np.asarray(part['half_extents_m']);bounds=np.asarray([-h,h])
                elif part['geometry']=='sphere':
                    h=np.repeat(part['radius_m'],3);bounds=np.asarray([-h,h])
                elif part['geometry'].startswith('usd_'):
                    path=root/asset['usd_path'];self.dependencies[str(path)]=digest(path)
                    shape_stage=Usd.Stage.Open(str(path));shape=shape_stage.GetPrimAtPath(part['usd_prim_path'])
                    if not shape:raise ValueError('Missing task USD collider')
                    b=cache.ComputeWorldBound(shape).ComputeAlignedRange()
                    bounds=np.asarray([b.GetMin(),b.GetMax()])
                else:raise ValueError('Unsupported collider '+part['geometry'])
                corners=np.asarray(list(product(*zip(*bounds))))
                corners=Rotation.from_quat(part.get('quaternion_xyzw',[0,0,0,1])).apply(corners)+part.get('position_m',[0,0,0])
                corners=Rotation.from_quat([*q[1:],q[0]]).apply(corners)+row['position_m']
                self.obstacle_corners.append(corners)
                bounds=transformed_bounds(bounds,part.get('position_m',[0,0,0]),part.get('quaternion_xyzw',[0,0,0,1]))
                bounds=transformed_bounds(bounds,row['position_m'],[*q[1:],q[0]])
                self.obstacles.append(bounds);self.obstacle_names.append('/Scene/'+row['id']+'/'+part.get('label',part['geometry']))
        self.obstacles=np.asarray(self.obstacles)
        self.obstacle_corners=np.asarray(self.obstacle_corners)
        self.stands={s:np.asarray([[np.asarray(b['position_m'])-np.asarray(b['size_m'])/2,
                                   np.asarray(b['position_m'])+np.asarray(b['size_m'])/2]
                                  for b in stand['boxes']]) for s,stand in stands.items()}
        self.stand_names={s:[b['name'] for b in stand['boxes']] for s,stand in stands.items()}
        self.robot_shapes={};self.shape_names={}
        for side,model in models.items():
            manifest, source_prim = self.robot_import(side, robot)
            imp=load_json(manifest)
            for path,expected in model.asset_sha256.items():
                if digest(path)!=expected:raise ValueError('Changed FK source: '+path)
                self.dependencies[str(path)]=expected
            for p,h in {imp['usd_path']:imp['usd_sha256'],**imp['derived_files']}.items():
                if digest(p)!=h:raise ValueError('Changed source robot: '+p)
                self.dependencies[p]=h
            root=UsdGeom.Xform.Define(self.stage,'/Robots/'+side).GetPrim()
            root.GetReferences().AddReference(imp['usd_path'],source_prim)
            poses=self.poses(side,initial)
            records=[];names=[]
            for prim in shapes(root):
                parent=prim
                while parent and parent.GetName() not in poses:parent=parent.GetParent()
                if not parent:raise ValueError('Unbound collider: '+str(prim.GetPath()))
                bound=cache.ComputeRelativeBound(prim,parent).ComputeAlignedRange()
                lo,hi=np.asarray(bound.GetMin()),np.asarray(bound.GetMax())
                corners=np.asarray([(*c,1.) for c in product(*zip(lo,hi))])
                records.append((parent.GetName(),corners));names.append(str(prim.GetPath()))
            self.robot_shapes[side]=records;self.shape_names[side]=names

    def poses(self,side,joints):
        model=self.models[side];own=model.checked_joints({n:joints[n] for n in model.names},enforce_limits=True)
        p=model.arm.forward(own,model.arm_base)
        p.update(model.hand.forward(own,p[model.arm_tip]@model.wrist_offset))
        return p

    def boxes(self,side,joints):
        p=self.poses(side,joints)
        corners=[(p[name]@c.T).T[:,:3] for name,c in self.robot_shapes[side]]
        return np.asarray([[c.min(0),c.max(0)] for c in corners])

    def check(self,boxes):
        rows={}
        def report(a,b,an,bn):
            g=gaps(a,b);i,j=np.unravel_index(g.argmin(),g.shape)
            return {'minimum_aabb_gap_m':float(g[i,j]),'overlaps':int((g<=1e-8).sum()),
                'closest':[an[i],bn[j]],'clear':bool(g.min()>.001)}
        for side in self.models:
            rows[side+'_environment']=report(boxes[side],self.obstacles,self.shape_names[side],self.obstacle_names)
            for other in self.stands:
                if other == side:continue
                rows[side+'_other_stand']=report(boxes[side],self.stands[other],self.shape_names[side],self.stand_names[other])
            movable=[i for i,(link,_) in enumerate(self.robot_shapes[side])
                     if link not in ('fr3_'+side+'_base','fr3_'+side+'_link0', 'fr3_base', 'fr3_link0')]
            rows[side+'_own_stand']=report(boxes[side][movable],self.stands[side],
                [self.shape_names[side][i] for i in movable],self.stand_names[side])
        if len(self.models) == 2:
            first,second=('left','right') if set(self.models)=={'left','right'} else self.models
            rows['between_robots']=report(boxes[first],boxes[second],self.shape_names[first],self.shape_names[second])
        return {'clear':all(r['clear'] for r in rows.values()),'checks':rows}

    def stand_check(self):
        rows={}
        for side,b in self.stands.items():
            depth=np.minimum(b[:,None,1],self.obstacles[None,:,1])-np.maximum(b[:,None,0],self.obstacles[None,:,0])
            hit=np.all(depth>.001,axis=2)
            rows[side]={'clear':not hit.any(),'penetrating_pairs':[[int(i),self.obstacle_names[j]] for i,j in zip(*np.where(hit))]}
        return {'clear':all(r['clear'] for r in rows.values()),'checks':rows,
                'scope':'AABB interior penetration above 1 mm; floor and mount surface contact allowed.'}

    def approach_surfaces(self,samples):
        # A goal at an insertion socket's bottom is not a free-space fingertip
        # location. Probe above its physical fixture, retaining the exact goal.
        for sample in samples:
            if sample['approach']!='above':continue
            p=np.asarray(sample['reference_position_m']);top=float(p[2]);surfaces=[]
            radius=.04 if sample['label'].startswith('target/') else .005
            for name,b in zip(self.obstacle_names,self.obstacles):
                if name.startswith('/World/Environment/'):continue
                if np.all(p[:2]>=b[0,:2]-radius) and np.all(p[:2]<=b[1,:2]+radius) and p[2]-.015<=b[1,2]<=p[2]+.20:
                    if b[1,2]>top:top=float(b[1,2]);surfaces=[name]
            sample['approach_surface_z_m']=top
            sample['approach_surface_shapes']=surfaces
            sample['fingertip_target_m']=[float(p[0]),float(p[1]),top+.045]


# Explicit, versioned solver policy for the parameterized CPU audits of both
# robot profiles. ``policy=None`` keeps the legacy behaviour bit-for-bit for
# every other caller. Under the policy the unchanged legacy search runs first;
# restarts (for candidates whose ready-seeded solve failed) and the appended
# candidates are tried only when it found no clear solution, so a legacy
# clear result never moves.
#
# v2 keeps passes 1 and 2 of v1 exactly and adds pass 3 only when both found
# no clear solution: the same palm-down family on a dense deterministic
# pitch/yaw grid. Orientations whose wrist-carried robot shapes (hand, flange,
# last link) already fail the checker's own environment test at the target
# are skipped; the remaining ones are solved nearest-ready-extension first,
# within a fixed budget, from the same ready + deterministic restarts. The v1
# grid missed narrow feasible pitch bands near the reach boundary and in
# clutter (for example pitch 75 deg for a cornered shaker). v1 stays
# selectable for replay of its reports.
IK_SOLVER_POLICY_V1='multistart8_refine_v1'
IK_SOLVER_POLICY='multistart8_refine_screen_v2'
IK_SOLVER_POLICIES=(IK_SOLVER_POLICY_V1,IK_SOLVER_POLICY)
IK_RESTARTS=8
IK_RESTART_RNG_SEED=0
IK_REFINE_POSITION_TOLERANCE_M=.0001
IK_REFINE_ROTATION_TOLERANCE_RAD=.0003
EXTENDED_PALM_DOWN_YAW_STEP_DEG=15
EXTENDED_PITCHES_DEG=(15.,-15.,30.,-30.)
EXTENDED_PITCHED_YAW_STEP_DEG=30
# Appended after the legacy +/-22.5 deg grid of the orientation fallback.
FALLBACK_GRID_EXTENSION_DEG=tuple((yaw,pitch) for yaw in (-30.,-15.,0.,15.,30.)
    for pitch in (-30.,-15.,0.,15.,30.) if (yaw,pitch)!=(0.,0.))
# v2 pass 3: rpy(pi, pitch, yaw), pitch -90..90 and yaw 0..352.5 in 7.5 deg
# steps (25 x 48), minus rotations already tried in passes 1-2.
SCREEN_GRID_STEP_DEG=7.5
SCREEN_IK_BUDGET=24
SCREEN_CLEARANCE_M=.001
IK_POLICY={'id':IK_SOLVER_POLICY,
    'lm':{'iterations':160,'position_tolerance_m':.0002,'rotation_tolerance_rad':.004},
    'ready_seed':'pass 1 is the unchanged legacy search (legacy candidates from the ready coordinates); its clear result is returned unchanged',
    'second_pass':'only if pass 1 found no clear solution: restarts for legacy candidates whose ready solve failed, then appended candidates',
    'restarts_after_failed_ready_solve':IK_RESTARTS,
    'restart_starts':f'arm joint-range midpoint, then {IK_RESTARTS-1} uniform draws within the URDF limits from numpy default_rng({IK_RESTART_RNG_SEED}); first converged start in this order is used',
    'appended_candidates':{'palm_down_yaw_step_deg':EXTENDED_PALM_DOWN_YAW_STEP_DEG,
        'pitches_deg':list(EXTENDED_PITCHES_DEG),'pitched_yaw_step_deg':EXTENDED_PITCHED_YAW_STEP_DEG,
        'approaches':['above','handoff'],'order':'after all legacy candidates'},
    'fallback_grid_extension_deg':[list(row) for row in FALLBACK_GRID_EXTENSION_DEG],
    'fallback_refinement':{'position_tolerance_m':IK_REFINE_POSITION_TOLERANCE_M,
        'rotation_tolerance_rad':IK_REFINE_ROTATION_TOLERANCE_RAD,
        'when':'only after the unchanged 0.2 mm fingertip / 0.004 rad fallback check fails; clearance is recomputed for the refined coordinates',
        'passes':'orientation/source-yaw fallbacks rerun the unchanged legacy search first (legacy reference and grid); refinement and the extended grid apply only if it covers nothing'},
    'third_pass':{'when':'policy v2 only, approaches above/handoff, only if passes 1 and 2 found no clear solution',
        'grid':{'family':'rpy(pi, pitch, yaw), the unchanged palm-down family','pitch_deg':[-90.,90.],
            'yaw_deg':[0.,360.-SCREEN_GRID_STEP_DEG],'step_deg':SCREEN_GRID_STEP_DEG,
            'order':'pitch-major, then yaw','excluded':'rotations already evaluated in passes 1-2'},
        'screen':'skip an orientation whose wrist-carried robot shapes (rigid with the wrist at the queried hand posture) '
                 'fail the checker environment test (AABB gap and separating-axis gap both <= '+str(SCREEN_CLEARANCE_M)+' m) '
                 'at the target wrist pose; without a checker nothing is skipped',
        'solve_order':'ascending |flange-to-shoulder distance - ready flange-to-shoulder distance|, then grid order',
        'budget_orientations':SCREEN_IK_BUDGET,
        'starts':'ready, then the same deterministic restarts; a converged solution uses the unchanged '
                 'clearance, self-collision and nullspace-restart checks',
        'monotone':'pass 3 only adds clear solutions; every pass 1-2 result is returned unchanged'},
    'joint_margin':'recorded as data only; never a gate',
    'scope':'Rigid parameterized solve_target and orientation/open/source-yaw fallbacks; laundry arm chains are unchanged.'}


def checked_policy(policy):
    if policy is not None and policy not in IK_SOLVER_POLICIES:
        raise ValueError('Unknown IK solver policy: '+str(policy))
    return policy


def extended_rotations(approach):
    """Policy-only candidates, appended after the unchanged legacy list."""
    if approach not in ('above','handoff'):
        return []
    rows=[rpy_matrix([math.pi,0.,math.radians(y)])
          for y in range(0,360,EXTENDED_PALM_DOWN_YAW_STEP_DEG) if y%45]
    rows.extend(rpy_matrix([math.pi,math.radians(p),math.radians(y)]) for p in EXTENDED_PITCHES_DEG
                for y in range(0,360,EXTENDED_PITCHED_YAW_STEP_DEG))
    return rows


def restart_starts(model):
    """Deterministic restarts: joint-range midpoint, then a fixed-seed uniform set."""
    lo=np.array([j.lower for j in model.arm.active],dtype=float)
    hi=np.array([j.upper for j in model.arm.active],dtype=float)
    lo,hi=np.where(np.isfinite(lo),lo,-np.pi),np.where(np.isfinite(hi),hi,np.pi)
    rng=np.random.default_rng(IK_RESTART_RNG_SEED)
    starts=[(lo+hi)/2,*[rng.uniform(lo,hi) for _ in range(IK_RESTARTS-1)]]
    return [dict(zip(model.arm.names,s.tolist())) for s in starts]


def arm_joint_margin(model,joints):
    q=np.array([joints[n] for n in model.arm.names],dtype=float)
    lo=np.array([j.lower for j in model.arm.active],dtype=float)
    hi=np.array([j.upper for j in model.arm.active],dtype=float)
    margin=np.minimum(q-lo,hi-q)
    return float(margin.min()),model.arm.names[int(margin.argmin())]


def screen_grid(approach):
    """Policy v2 pass-3 grid in its fixed pitch-major, yaw-minor order."""
    if approach not in ('above','handoff'):
        return []
    step=SCREEN_GRID_STEP_DEG
    pitches=[-90.+step*i for i in range(int(round(180./step))+1)]
    yaws=[step*i for i in range(int(round(360./step)))]
    return [((p,y),rpy_matrix([math.pi,math.radians(p),math.radians(y)])) for p in pitches for y in yaws]


def wrist_rigid_shapes(checker,model,side,joints):
    """Robot shapes rigidly carried by the wrist at this hand posture.

    A shape qualifies when its pose relative to the wrist frame is identical
    at the queried arm coordinates and at two other in-limit arm coordinates
    (the restart midpoint and first uniform restart), so the result does not
    depend on link names or on the robot profile. Returns (shape index, pose
    relative to the wrist, link-frame corners).
    """
    def relative(q):
        poses=checker.poses(side,q)
        wrist=poses[model.arm_tip]@model.wrist_offset
        return [np.linalg.solve(wrist,poses[link]) for link,_ in checker.robot_shapes[side]]
    base=relative(joints)
    others=[relative({**joints,**start}) for start in restart_starts(model)[:2]]
    rows=[(index,base[index],corners) for index,(_,corners) in enumerate(checker.robot_shapes[side])
          if all(np.allclose(base[index],other[index],atol=1e-9,rtol=0.) for other in others)]
    if not rows or len(rows)==len(base):
        raise ValueError('Could not separate wrist-carried robot shapes from the arm chain')
    return rows


def wrist_shapes_clear(checker,rigid,wrist):
    """The checker environment test, restricted to wrist-carried shapes at one wrist pose."""
    from scripts.diagnostics.check_interaction_workspace import separating_axis_gap
    corners=np.asarray([(wrist@relative@local.T).T[:,:3] for _,relative,local in rigid])
    bounds=np.stack([corners.min(1),corners.max(1)],axis=1)
    near=np.argwhere(gaps(bounds,checker.obstacles)<=SCREEN_CLEARANCE_M)
    return all(separating_axis_gap(corners[i],checker.obstacle_corners[j])>SCREEN_CLEARANCE_M for i,j in near)


def flange_preference(model,initial):
    """Pass-3 solve order key: nearest the ready arm extension (flange to shoulder) first."""
    arm={n:initial[n] for n in model.arm.names}
    poses=model.arm.forward(arm,model.arm_base)
    joint=model.arm.active[1] if len(model.arm.active)>1 else model.arm.active[0]
    shoulder=(poses[joint.parent]@joint.origin)[:3,3]
    ready=float(np.linalg.norm(poses[model.arm_tip][:3,3]-shoulder))
    def key(wrist):
        flange=wrist@np.linalg.inv(model.wrist_offset)
        return abs(float(np.linalg.norm(flange[:3,3]-shoulder))-ready)
    return key


def solve_target(model,initial,point,side,base_yaw,approach,checker=None,ready_boxes=None,*,joint_check=None,restart_seeds=None,policy=None):
    """First clear candidate wrist pose for a middle-fingertip point.

    Pass 1 is the unchanged legacy search: legacy candidates in order, each
    solved from the ready coordinates, plus the optional nullspace restarts.
    Under ``policy`` a second pass runs only when pass 1 found no clear
    solution: deterministic restarts for legacy candidates whose ready solve
    failed, then the appended candidates. A legacy clear result never moves.
    Policy v2 adds a third pass only when passes 1-2 found no clear solution:
    the screened dense palm-down grid described in ``IK_POLICY['third_pass']``.
    """
    policy=checked_policy(policy)
    tip=np.asarray(model.public_geometry({n:initial[n] for n in model.names})['fingertip_positions_m']['middle'])
    point=np.asarray(point)
    yaws=[base_yaw+(-1 if side=='left' else 1)*.75*math.pi,
          *[k*math.pi/4 for k in range(8)]]
    rotations=[rpy_matrix([math.pi,0.,y]) for y in yaws]
    rotations.extend(rpy_matrix([math.pi,p,y]) for p in (-math.pi/4,math.pi/4,-math.pi/2,math.pi/2)
                     for y in (0.,math.pi/2,math.pi,3*math.pi/2))
    if approach=='handoff':
        rotations.extend(rpy_matrix([0.,p,y]) for p in (0.,-math.pi/4,math.pi/4)
                         for y in (0.,math.pi/2,math.pi,3*math.pi/2))
    if approach=='shelf_front':
        rotations=[rpy_matrix([a,-math.pi/2,0.]) for a in (0.,math.pi/2,-math.pi/2,math.pi)]
    legacy_count=len(rotations)
    if policy:
        rotations.extend(extended_rotations(approach))
    attempted=[];restarts=[]
    stats={'evaluated':0,'starts':0,'best_margin':None,'legacy':None}

    def record(row,*,candidate,starts,selected):
        # Data only: candidate/start bookkeeping, the joint-limit margin and
        # what the unchanged legacy search alone returned for this sample.
        if not policy:return row
        margin,joint=arm_joint_margin(model,row['joint_positions_rad'])
        best=margin if stats['best_margin'] is None else max(margin,stats['best_margin'])
        row['ik_policy']={'policy':policy,'candidate_index':candidate,'legacy_candidate':candidate<legacy_count,
            'candidate_count':len(rotations),'legacy_candidate_count':legacy_count,
            'candidates_evaluated':stats['evaluated'],'starts_for_selected_candidate':starts,
            'selected_start':selected,'starts_attempted_total':stats['starts'],
            'arm_joint_limit_margin_rad':margin,'tightest_arm_joint':joint,
            'best_converged_arm_joint_limit_margin_rad':best,'legacy_outcome':dict(stats['legacy'])}
        if 'screened_dense' in stats:
            row['ik_policy']['screened_dense_pass']=dict(stats['screened_dense'])
        return row

    def candidate_target(rot):
        wrist=np.eye(4);wrist[:3,:3]=rot;wrist[:3,3]=point-rot@tip
        return wrist,np.linalg.inv(model.arm_base)@wrist@np.linalg.inv(model.wrist_offset)

    def solve(target,start):
        stats['starts']+=1
        return model.arm.inverse(target,model.arm_tip,start,
            iterations=160,multi_starts=0,position_tolerance=.0002,rotation_tolerance=.004)

    def restart_solve(target):
        if not restarts:restarts.extend(restart_starts(model))
        solved=None
        for number,start in enumerate(restarts):
            solved=solve(target,start)
            if solved['converged']:return solved,number+1,'restart_'+str(number)
        return solved,len(restarts),None

    def evaluate(index,wrist,target,solved,starts,selected):
        joints={**initial,**solved['joint_positions_rad']}
        if policy:
            margin=arm_joint_margin(model,joints)[0]
            stats['best_margin']=margin if stats['best_margin'] is None else max(stats['best_margin'],margin)
        actual=model.arm.forward(joints,model.arm_base)[model.arm_tip]@model.wrist_offset
        tip_actual=(actual@[*tip,1.])[:3]
        clearance=checker.check({**ready_boxes,side:checker.boxes(side,joints)}) if checker else None
        row={'converged':True,'clearance_passed':clearance is None or clearance['clear'],
            'fingertip_error_m':float(np.linalg.norm(tip_actual-point)),
            'wrist_target_matrix':wrist.tolist(),'joint_positions_rad':joints,'clearance':clearance}
        if joint_check is not None:
            row['arm_self_collision']=joint_check(joints)
            row['clearance_passed'] &= row['arm_self_collision']['clear']
        attempted.append((row,index,starts,selected))
        if row['clearance_passed']:return row
        if restart_seeds is not None:
            # Optional offline diagnostic only. Keep the same wrist target,
            # original LM settings and clearance criteria for each restart.
            for seed_record in restart_seeds(model, joints):
                rerun=solve(target,seed_record['arm_joint_positions_rad'])
                attempt={'seed':seed_record,'solver':rerun}
                row.setdefault('restart_attempts',[]).append(attempt)
                if not rerun['converged']:continue
                alternative={**initial,**rerun['joint_positions_rad']}
                fk=model.arm.forward(alternative,model.arm_base)[model.arm_tip]@model.wrist_offset
                bounds=checker.check({**ready_boxes,side:checker.boxes(side,alternative)}) if checker else None
                result={'converged':True,'clearance_passed':bounds is None or bounds['clear'],
                    'fingertip_error_m':float(np.linalg.norm((fk@[*tip,1.])[:3]-point)),
                    'wrist_target_matrix':wrist.tolist(),'joint_positions_rad':alternative,'clearance':bounds}
                if joint_check is not None:
                    result['arm_self_collision']=joint_check(alternative)
                    result['clearance_passed'] &= result['arm_self_collision']['clear']
                attempt['result']=result
                if result['clearance_passed']:
                    return {**result,'original_rejected_candidate':row,
                            'restart_selection':seed_record}
        return None

    ready={n:initial[n] for n in model.arm.names}
    failed_ready=[]
    for index in range(legacy_count):
        stats['evaluated']+=1
        wrist,target=candidate_target(rotations[index])
        solved=solve(target,ready)
        if not solved['converged']:
            failed_ready.append((index,wrist,target));continue
        found=evaluate(index,wrist,target,solved,1,'ready')
        if found is not None:
            stats['legacy']={'converged':True,'clearance_passed':True}
            return record(found,candidate=index,starts=1,selected='ready')
    stats['legacy']={'converged':bool(attempted),'clearance_passed':False}
    if policy:
        for index,wrist,target in failed_ready:
            solved,used,selected=restart_solve(target)
            if solved['converged']:
                found=evaluate(index,wrist,target,solved,1+used,selected)
                if found is not None:return record(found,candidate=index,starts=1+used,selected=selected)
        for index in range(legacy_count,len(rotations)):
            stats['evaluated']+=1
            wrist,target=candidate_target(rotations[index])
            solved,used,selected=solve(target,ready),0,'ready'
            if not solved['converged']:
                solved,used,selected=restart_solve(target)
            if solved['converged']:
                found=evaluate(index,wrist,target,solved,1+used,selected)
                if found is not None:return record(found,candidate=index,starts=1+used,selected=selected)
    if policy==IK_SOLVER_POLICY and approach in ('above','handoff'):
        # Pass 3 (v2): reached only without a clear pass 1-2 solution.
        grid=screen_grid(approach)
        tried=np.asarray(rotations).reshape((-1,9))
        fresh=[(angles,rot) for angles,rot in grid
               if not np.any(np.max(np.abs(tried-rot.reshape(9)),axis=1)<=1e-9)]
        rigid=None if checker is None else wrist_rigid_shapes(checker,model,side,initial)
        preference=flange_preference(model,initial)
        eligible=[]
        for position,(angles,rot) in enumerate(fresh):
            wrist,target=candidate_target(rot)
            if rigid is None or wrist_shapes_clear(checker,rigid,wrist):
                eligible.append((preference(wrist),position,angles,wrist,target))
        eligible.sort(key=lambda row:(row[0],row[1]))
        dense=stats['screened_dense']={'grid_orientations':len(grid),'previously_evaluated':len(grid)-len(fresh),
            'screened':len(fresh),'screen_clear':len(eligible),'budget':SCREEN_IK_BUDGET,'solved':0,
            'screen_applied':rigid is not None,'wrist_carried_shapes':0 if rigid is None else len(rigid)}
        for _,position,angles,wrist,target in eligible[:SCREEN_IK_BUDGET]:
            stats['evaluated']+=1;dense['solved']+=1
            index=len(rotations)+position
            solved,used,selected=solve(target,ready),0,'ready'
            if not solved['converged']:
                solved,used,selected=restart_solve(target)
            if solved['converged']:
                found=evaluate(index,wrist,target,solved,1+used,selected)
                if found is not None:
                    dense.update(selected_pitch_deg=angles[0],selected_yaw_deg=angles[1],
                                 selected_solve_rank=dense['solved'])
                    return record(found,candidate=index,starts=1+used,selected=selected)
    if attempted:
        row,index,starts,selected=attempted[0]
        return record(row,candidate=index,starts=starts,selected=selected)
    failed={'converged':False,'clearance_passed':False}
    if policy:
        failed['ik_policy']={'policy':policy,'candidate_count':len(rotations),'legacy_candidate_count':legacy_count,
            'candidates_evaluated':stats['evaluated'],'starts_attempted_total':stats['starts'],
            'legacy_outcome':dict(stats['legacy'])}
        if 'screened_dense' in stats:
            failed['ik_policy']['screened_dense_pass']=dict(stats['screened_dense'])
    return failed


def common_workspace(family,configuration_id,models,initial,poses,checker,ready_boxes,task_id=None):
    centers={'mechanical_assembly':[.42,0.,.90], 'study':[.50,0.,.90],
        'living_room':[.65,0.,.90], 'card_room':[.65,0.,.94],
        'laboratory':[.70,.32,1.15]}
    center=np.asarray(centers[family])
    if family=='study' and task_id=='drawer_retrieval':center[2]=1.10
    hands={s:solve_target(models[s],initial,center,s,poses[s]['base_yaw_rad'],'above',checker,ready_boxes)
        for s in layout.SIDES}
    shared=all(r['converged'] and r['clearance_passed'] for r in hands.values())
    axis=np.asarray(poses['left']['base_position_m'])-poses['right']['base_position_m']
    axis[2]=0.;axis/=np.linalg.norm(axis)
    pair={};pair_clear=None
    for separation in (.14,.22,.28):
        for side,sign in [('left',1),('right',-1)]:
            p=center+sign*separation*axis
            pair[side]={'fingertip_target_m':p.tolist(),**solve_target(models[side],initial,p,side,
                poses[side]['base_yaw_rad'],'above',checker,ready_boxes)}
        if checker and all(r['converged'] for r in pair.values()):
            pair_clear=checker.check({s:checker.boxes(s,r['joint_positions_rad']) for s,r in pair.items()})
            if pair_clear['clear']:break
    return {'shared_fingertip_point_m':center.tolist(),'shared_point_hands':hands,
        'shared_point_reachable_by_both':shared,'simultaneous_separated_pair':pair,
        'simultaneous_pair_clearance':pair_clear,
        'simultaneous_pair_passed':all(r['converged'] and r['clearance_passed'] for r in pair.values())
            and (pair_clear is None or pair_clear['clear']),
        'scope':'Shared approach point checked separately with the other arm waiting; separated pair checked simultaneously. No swept path proof.'}


def run(family,task_id,configuration_id,release,*,check_clearance=True):
    from bench.control.factory import public_robot_geometry,robot_configuration
    samples,scene,task=task_points(family,task_id,release)
    robot,initial,ready=layout.robot_inputs(family,task_id,configuration_id)
    initial_state=deepcopy(initial)
    models=public_robot_geometry(robot_configuration({'robot':robot})).sides
    initial=initial['joint_positions_rad'];poses=layout.placements(family,task_id,configuration_id)
    checker=CollisionBounds(scene,robot,models,initial,layout.stand_geometry(family,task_id,configuration_id)) if check_clearance else None
    if checker:checker.approach_surfaces(samples)
    ready_boxes={s:checker.boxes(s,initial) for s in layout.SIDES} if checker else None
    ready_check=checker.check(ready_boxes) if checker else None
    stand_check=checker.stand_check() if checker else None
    rows=[]
    for sample in samples:
        hands={s:solve_target(models[s],initial,sample['fingertip_target_m'],s,poses[s]['base_yaw_rad'],
            sample['approach'],checker,ready_boxes) for s in layout.SIDES}
        usable=[s for s,r in hands.items() if r['converged'] and r['clearance_passed']]
        rows.append({**sample,'hands':hands,'usable_sides':usable,'covered':bool(usable)})
    common=common_workspace(family,configuration_id,models,initial,poses,checker,ready_boxes,task_id)
    return {'schema':'bimanual_option_discrete_workspace.v1','family':family,'task_id':task_id,
        'configuration_id':configuration_id,'release':str(release),
        'source_files':{n:digest(Path(release)/n) for n in ('scene.json','task.json','release_manifest.json')},
        'layout_source_sha256':digest(layout.__file__),'diagnostic_source_sha256':digest(__file__),
        'configuration_binding':{'hash_encoding':'SHA256 of sorted compact JSON',
            'robot_sha256':content_digest(robot),'initial_state_sha256':content_digest(initial_state),
            'joint_positions_sha256':content_digest(initial),
            'placements_sha256':content_digest(poses),
            'stand_geometry_sha256':content_digest(layout.stand_geometry(family,task_id,configuration_id))},
        'initial_state':initial_state,'stand_geometry':layout.stand_geometry(family,task_id,configuration_id),
        'robot_configuration':robot,'ready_design':ready,'placements':poses,
        'actual_collision_shape_counts':{s:len(checker.robot_shapes[s]) for s in layout.SIDES} if checker else {},
        'asset_dependencies':checker.dependencies if checker else {},'ready_clearance':ready_check,'stand_clearance':stand_check,
        'sample_count':len(rows),'covered_count':sum(r['covered'] for r in rows),
        'samples':rows,'common_workspace':common,
        'passed':all(r['covered'] for r in rows) and common['shared_point_reachable_by_both'] and common['simultaneous_pair_passed'] and (ready_check is None or ready_check['clear']) and (stand_check is None or stand_check['clear']),
        'native_steps':0,'model_calls':0,'manipulation_qualified':False,'continuous_path_qualified':False,
        'scope':'Discrete actual middle-fingertip approach IK, each sampled source/target reachable by at least one hand. Optional conservative AABBs of original USD robot, fixed environment and authored task-layer colliders; no grasp, swept path, native contact or self-collision qualification.'}


def _matrix_job(job):
    family,task_id,configuration_id,output,check_clearance=job
    from scenes.workrooms.catalog import task_definition
    release=task_definition(family,task_id)['source_release']['path']
    report=run(family,task_id,configuration_id,release,check_clearance=check_clearance)
    path=Path(output)/(family+'_'+task_id+'_'+configuration_id+'.json')
    with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'family':family,'task_id':task_id,'configuration_id':configuration_id,
        'report_path':str(path),'report_sha256':digest(path),'passed':report['passed'],
        'sample_count':report['sample_count'],'covered_count':report['covered_count'],
        'source_files':report['source_files'],'configuration_binding':report['configuration_binding'],
        'ready_clear':report['ready_clearance']['clear'] if check_clearance else None,
        'stand_clear':report['stand_clearance']['clear'] if check_clearance else None,
        'shared_point_reachable_by_both':report['common_workspace']['shared_point_reachable_by_both'],
        'simultaneous_pair_passed':report['common_workspace']['simultaneous_pair_passed']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=tuple(layout.ELIGIBLE_TASKS))
    p.add_argument('--task');p.add_argument('--configuration',choices=layout.CONFIGURATION_IDS)
    p.add_argument('--all',action='store_true',help='Audit all eligible options into a new output directory')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--release',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-clearance',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous audit')
    if not a.output.resolve().is_relative_to(OUTPUT_ROOT):raise ValueError('Evidence belongs under datasets')
    if a.all:
        if any((a.family,a.task,a.configuration,a.release)):p.error('--all cannot be combined with a task selector')
        if a.workers<1:p.error('--workers must be positive')
        from concurrent.futures import ProcessPoolExecutor,as_completed
        a.output.mkdir(parents=True)
        jobs=[(f,t,c,str(a.output),not a.skip_clearance) for f,tasks in layout.ELIGIBLE_TASKS.items()
              for t in tasks for c in layout.CONFIGURATION_IDS]
        rows=[]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(_matrix_job,j) for j in jobs]):
                row=future.result();rows.append(row)
                print(json.dumps({k:row[k] for k in ('family','task_id','configuration_id','passed','covered_count','sample_count')}),flush=True)
        rows.sort(key=lambda r:(r['family'],r['task_id'],r['configuration_id']))
        summary={'schema':'bimanual_option_workspace_matrix.v1','passed':all(r['passed'] for r in rows),
            'variant_count':len(rows),'passed_count':sum(r['passed'] for r in rows),
            'sample_count':sum(r['sample_count'] for r in rows),'covered_count':sum(r['covered_count'] for r in rows),
            'layout_source_sha256':digest(layout.__file__),'diagnostic_source_sha256':digest(__file__),
            'rows':rows,'native_steps':0,'continuous_path_qualified':False,'manipulation_qualified':False,
            'scope':'Discrete source/goal approach IK, original collision-shape AABB separation, shared point and separated simultaneous pair; original fixed rooms and task props unchanged.'}
        with (a.output/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
        return 0 if summary['passed'] else 1
    if not all((a.family,a.task,a.configuration)):p.error('single audit requires --family, --task and --configuration')
    if a.release is None:
        from scenes.workrooms.catalog import task_definition
        a.release=Path(task_definition(a.family,a.task)['source_release']['path'])
    result=run(a.family,a.task,a.configuration,a.release,check_clearance=not a.skip_clearance)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('family','task_id','configuration_id','passed','sample_count','covered_count')}))
    print(json.dumps({'uncovered':[r['label'] for r in result['samples'] if not r['covered']],'output':str(a.output)}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
