"""Versioned input transformation for removing unnecessary supply pads.

This changes the reset/support geometry of L2/L3, so new native evidence is
required. It does not edit releases, functional goals, object assets or gains.
"""
from copy import deepcopy
import numpy as np

from bench.geometry import quaternion_matrix
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh, condition_collision_mesh
from .scene import SCENE


CLEARANCE_M=.0002
PAD_BINDINGS={
    'parts_sorting': {},
    'peg_insertion': {f'supply_cradle_{i}':(f'peg_{i}','mechanical_l2_supply_cradle') for i in range(2)},
    'shaft_spacer_assembly': {n+'_supply_rest':(n,'mechanical_l3_supply_rest')
                            for n in ('spacer','wide_ring','top_ring')},
    'fastener_assembly': {},
}


def _collision_parts(spec):
    """Read actual source vertices/transforms, including USD SDF collision meshes."""
    asset=load_task_asset(spec['asset_id'],spec['asset_root'],verify_hashes=True)
    sources={asset['manifest_path']:asset['manifest_sha256']}
    result=[]
    if asset.get('backend')=='usd_reference':
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        stage=Usd.Stage.Open(asset['usd_path']);cache=UsdGeom.XformCache()
        if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
            raise ValueError('Supply geometry must use metre-scale Z-up source USD')
        sources.update(asset['source_dependencies_sha256'])
        for part in asset['collision_parts']:
            prim=stage.GetPrimAtPath(part['usd_prim_path'])
            if (not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                raise ValueError('Expected enabled source collision mesh')
            matrix=cache.GetLocalToWorldTransform(prim)
            points=np.array([list(matrix.Transform(Gf.Vec3d(*map(float,p))))
                             for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
            result.append(points)
    else:
        for part in asset['collision_parts']:
            if part.get('geometry') not in ('mesh','convex_mesh','triangle_mesh'):
                raise ValueError('Supply reset requires explicit source collision meshes')
            points,_=load_obj_mesh(part['mesh_path'])
            points*=np.asarray(part.get('scale',[1.,1.,1.]),dtype=np.float32)
            points,inverse=condition_collision_mesh(points,spec.get('collision_mesh_coordinate_scale',1.))
            if inverse is not None:points=points.astype(float)*inverse
            q=part.get('quaternion_xyzw',[0.,0.,0.,1.])
            points=points @ quaternion_matrix([q[3],*q[:3]]).T+part.get('position_m',[0.,0.,0.])
            result.append(points)
            sources[part['mesh_path']]=part['sha256']
    if not result or any(p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.isfinite(p).all() for p in result):
        raise ValueError('Invalid source collision vertices')
    rotation=quaternion_matrix(spec.get('quaternion_wxyz',[1.,0.,0.,0.]))
    return [p @ rotation.T+np.asarray(spec['position_m']) for p in result],sources


def _references(value,names,path=()):
    hits=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in names:hits.append('/'.join(path+(key,)))
            hits.extend(_references(item,names,path+(str(key),)))
    elif isinstance(value,list):
        for i,item in enumerate(value):hits.extend(_references(item,names,path+(str(i),)))
    elif isinstance(value,str):
        if any(value==name or value.startswith('/World/Fixtures/'+name+'/')
               or value=='/World/Fixtures/'+name for name in names):hits.append('/'.join(path))
    return hits


def remove_supply_pads(scene,task):
    """Return copied scene/task and a geometry receipt, without changing inputs.

    Only the explicitly identified L2 supply cradles and L3 flat supply rests
    are removable. L1 target trays and the L2/L3/L4 functional fixtures remain.
    L1 and L4 already have no supply pads and are returned without changes.
    This helper runs before visual overlays and frozen-release preparation.
    """
    task_id=task.get('task_id')
    if task_id not in PAD_BINDINGS or scene.get('scenario',{}).get('task_id')!=task_id:
        raise ValueError('Expected matching fixed mechanical L1-L4 task identity')
    if scene.get('workroom',{}).get('family')!='mechanical_assembly':
        raise ValueError('Supply pad removal is defined for the fixed workshop only')
    new_scene,new_task=deepcopy(scene),deepcopy(task)
    fixtures={row['id']:row for row in scene['fixtures']}
    objects={row['id']:row for row in scene['objects']}
    if len(fixtures)!=len(scene['fixtures']) or len(objects)!=len(scene['objects']):
        raise ValueError('Object and fixture identities must be unique')
    bindings=PAD_BINDINGS[task_id]
    selected=set(bindings)&set(fixtures)
    report={'schema':'mechanical_supply_pad_removal.v1','task_id':task_id,
            'table_top_z_m':SCENE['coordinates']['table_top_z_m'],
            'table_height_source':'Fixed workshop SCENE.coordinates; legacy scene.table is not its active geometry.',
            'clearance_m':CLEARANCE_M,'removed_fixture_ids':[],'object_updates':{},
            'geometry_sources_sha256':{},'functional_support_and_rules_unchanged':True,
            'native_validation_claimed':False,'source_releases_modified':False}
    if not selected:return new_scene,new_task,report
    if selected!=set(bindings):raise ValueError('Partial supply-pad inventory requires an explicit reset review')
    table_z=report['table_top_z_m']
    for fixture_id,(object_id,asset_id) in bindings.items():
        pad=fixtures[fixture_id];obj=objects[object_id]
        if pad['asset_id']!=asset_id:raise ValueError('Supply fixture asset differs from the removable pad contract')
        initial=task['evaluation']['initial_objects'][object_id]
        if any(initial[k]!=obj[k] for k in ('position_m','quaternion_wxyz')):
            raise ValueError('Evaluation initial pose differs from the scene before pad removal')
        pad_parts,pad_sources=_collision_parts(pad);obj_parts,obj_sources=_collision_parts(obj)
        report['geometry_sources_sha256'].update(pad_sources);report['geometry_sources_sha256'].update(obj_sources)
        points=np.vstack(obj_parts);bottom=float(points[:,2].min())
        # The cradle rim is higher than its supporting floor. Select the actual
        # lower support face covering the object's XY footprint, not the rim.
        support_tops=[]
        for p in pad_parts:
            lo,hi=p.min(axis=0),p.max(axis=0)
            if (np.all(points[:,:2].min(axis=0)>=lo[:2]-2.e-7)
                    and np.all(points[:,:2].max(axis=0)<=hi[:2]+2.e-7)
                    and hi[2]<=bottom+.0002):support_tops.append(float(hi[2]))
        if not support_tops or abs(bottom-max(support_tops))>.0002:
            raise ValueError('Object is not seated on the selected supply pad; refusing an assembly reset')
        new_position=list(obj['position_m']);new_position[2]+=table_z+CLEARANCE_M-bottom
        if new_position[2]>=obj['position_m'][2]:raise ValueError('Removing a supply pad must lower its object')
        target=next(row for row in new_scene['objects'] if row['id']==object_id)
        target['position_m']=new_position
        new_task['evaluation']['initial_objects'][object_id]['position_m']=deepcopy(new_position)
        supplies=new_task.get('reset_conditions',{}).get('supply_positions_m',{})
        if object_id in supplies:supplies[object_id]=deepcopy(new_position)
        if 'objects' in new_task.get('initial_state',{}):
            raise ValueError('Unexpected separate object reset overrides require explicit synchronization')
        report['object_updates'][object_id]={'removed_fixture_id':fixture_id,
            'old_position_m':deepcopy(obj['position_m']),'new_position_m':new_position,
            'old_collision_bottom_z_m':bottom,'old_support_surface_z_m':max(support_tops),
            'pad_maximum_z_m':float(np.vstack(pad_parts)[:,2].max()),
            'new_collision_bottom_z_m':table_z+CLEARANCE_M,
            'delta_z_m':new_position[2]-obj['position_m'][2]}
    new_scene['fixtures']=[row for row in new_scene['fixtures'] if row['id'] not in selected]
    # Never silently remove/relax scoring requirements if a future task links
    # one to a pad (including expected_rest_targets or actor support mappings).
    dangling=_references(new_scene,selected)+_references(new_task,selected)
    if dangling:raise ValueError('Removed pad still has task/reset references: '+', '.join(dangling))
    report['removed_fixture_ids']=sorted(selected)
    new_scene['description']={'peg_insertion':'Two independent pegs directly on the workshop table; fixed chamfered sockets.',
        'shaft_spacer_assembly':'Three free bored parts directly on the workshop table; fixed shaft and shoulder.'}[task_id]
    new_scene['scenario']['scope']='New direct-table supply reset; functional goals and physics settings unchanged.'
    new_scene['mechanical_task_provenance']['supply_pad_removal']={
        'schema':report['schema'],'removed_fixture_ids':report['removed_fixture_ids'],
        'table_top_z_m':table_z,'clearance_m':CLEARANCE_M,'native_qualified':False}
    new_task['reset_conditions']['kind']='unassembled_direct_table_supply'
    new_task['physics_accepted']=False;new_task['task_policy_success_claimed']=False
    new_task['qualification_scope']='New direct-table supply reset without display pads; native support/manipulation not qualified.'
    return new_scene,new_task,report
