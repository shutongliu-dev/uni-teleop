"""Append a small, explicitly unscored table inventory with source-bound checks."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np

from bench.geometry import quaternion_matrix
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from .assembly_geometry import physical_spec
from .distractor_assets import ASSET_ROOT, NUT_ID, WASHER_ID, BOLT_ID
from .presentation_reset import _collision_parts, CLEARANCE_M
from .scene import SCENE


MINIMUM_GAP_M = .020
LAYOUT = 'two_loose_groups_v2'
TASK_IDS = {'parts_sorting','peg_insertion','shaft_spacer_assembly','fastener_assembly','gear_train_assembly'}
# A loose group of three and a separate pair on the unused right side. These
# are new reset coordinates, not edits to previously frozen scenes or assets.
# Reject occupied layouts rather than moving task objects or reducing margin.
PLACEMENTS = (
    ('table_nut_0',NUT_ID,(.965,-.235),-21.,False),
    ('table_washer_0',WASHER_ID,(1.030,-.205),14.,False),
    ('table_short_bolt',BOLT_ID,(.940,-.130),-37.,True),
    ('table_nut_1',NUT_ID,(1.025,.240),33.,False),
    ('table_washer_1',WASHER_ID,(.950,.295),-9.,False),
)


def validate_distractors(scene, task):
    declared = scene.get('task_distractors')
    contract = task.get('distractors')
    if declared is None and contract is None:
        return set()
    if (not isinstance(declared, list) or not all(isinstance(name, str) for name in declared)
            or len(declared) != len(set(declared)) or not isinstance(contract, dict)
            or contract.get('schema') != 'mechanical_task_distractors.v1'
            or contract.get('ids') != declared or contract.get('scored') is not False):
        raise ValueError('Distractor declarations must agree and remain explicitly unscored')
    expected_roles = {name: asset_id for name, asset_id, *_ in PLACEMENTS}
    roles = contract.get('roles')
    if (set(declared) != set(expected_roles) or roles != expected_roles
            or not isinstance(contract.get('source_asset_root'), str)):
        raise ValueError('Distractor roles or source asset root differ from the declared inventory')
    objects = scene.get('objects', [])
    if len({row['id'] for row in objects}) != len(objects):
        raise ValueError('Distractor scene contains duplicate object identities')
    targets = set(task.get('evaluation', {}).get('initial_objects', {}))
    if targets & set(declared) or targets | set(declared) != {row['id'] for row in objects}:
        raise ValueError('Distractor inventory must exactly cover the unscored objects')
    for row in objects:
        if row['id'] in roles and (row.get('asset_id') != roles[row['id']] or
                Path(row.get('asset_root', '')).resolve() != Path(contract['source_asset_root']).resolve()):
            raise ValueError('Distractor physical asset differs from its declared role')
    return set(declared)


def _bounds(spec):
    asset = load_task_asset(spec['asset_id'],spec['asset_root'],verify_hashes=True)
    if asset.get('backend') != 'usd_reference':
        parts,sources = _collision_parts(spec)
        points = np.vstack(parts)
    else:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
        # Match add_usd_asset's reference semantics. Layer-level units/up-axis
        # metadata do not rescale or rotate a prim reference; v3 tools retain
        # their authored ancestor scale under the reference anchor.
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.SetStageUpAxis(stage,'Z')
        anchor = stage.DefinePrim('/Probe','Xform')
        source_root = Sdf.Path(asset['usd_default_prim'])
        anchor.GetReferences().AddReference(asset['usd_path'],source_root)
        cache = UsdGeom.XformCache()
        parts = []
        for part in asset['collision_parts']:
            relative = Sdf.Path(part['usd_prim_path']).MakeRelativePath(source_root)
            prim = stage.GetPrimAtPath(anchor.GetPath().AppendPath(relative))
            if (not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                raise ValueError('Expected enabled referenced collision mesh')
            matrix = cache.GetLocalToWorldTransform(prim)
            parts.append(np.array([list(matrix.Transform(Gf.Vec3d(*map(float,p))))
                                   for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]))
        points = np.vstack(parts) @ quaternion_matrix(spec.get('quaternion_wxyz',[1.,0.,0.,0.])).T
        points += np.asarray(spec['position_m'])
        sources = {asset['manifest_path']:asset['manifest_sha256'], **asset['source_dependencies_sha256']}
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid source collision bounds')
    return np.array([points.min(axis=0),points.max(axis=0)]), sources


def _xy_gap(first,second):
    delta = np.maximum(np.maximum(first[0,:2]-second[1,:2],second[0,:2]-first[1,:2]),0.)
    return float(np.linalg.norm(delta))


def add_distractors(scene, task, asset_root=ASSET_ROOT):
    """Return copied inputs and a bounds receipt, leaving scored objects intact.

    Every new item is a passive dynamic asset, initially 0.2 mm above the real
    table. Clearance is a conservative XY AABB distance using the transformed
    collision vertices, not decorative visual extents. This is a new reset;
    source checks do not qualify native rest, contact or manipulation.
    """
    task_id = task.get('task_id')
    if (task_id not in TASK_IDS or scene.get('scenario',{}).get('task_id') != task_id
            or scene.get('workroom',{}).get('family') != 'mechanical_assembly'):
        raise ValueError('Expected matching fixed mechanical task identity')
    if scene.get('task_distractors') or task.get('distractors'):
        raise ValueError('Distractors already declared; do not append duplicate inventory')
    rows = scene['objects'] + scene['fixtures']
    identities = [row['id'] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Physical object and fixture identities must be unique')
    if set(identities) & {item[0] for item in PLACEMENTS}:
        raise ValueError('New distractor identity is already occupied')
    if set(task['evaluation']['initial_objects']) != {row['id'] for row in scene['objects']}:
        raise ValueError('Scored reset inventory must match the original task objects')
    new_scene,new_task = deepcopy(scene),deepcopy(task)
    report = {'schema':'mechanical_table_distractors.v1','task_id':task_id,'layout':LAYOUT,
        'asset_root':str(Path(asset_root).resolve()), 'minimum_gap_m':MINIMUM_GAP_M,
        'clearance_method':'Conservative Euclidean XY AABB separation from actual transformed collision vertices.',
        'table_top_z_m':SCENE['coordinates']['table_top_z_m'], 'table_clearance_m':CLEARANCE_M,
        'table_height_source':'Fixed workshop SCENE.coordinates; legacy scene.table is inactive.',
        'items':{}, 'geometry_sources_sha256':{}, 'scored_inventory_unchanged':True,
        'native_validation_claimed':False, 'source_releases_modified':False}
    occupied = {}
    for row in rows:
        bounds,sources = _bounds(row)
        occupied[row['id']] = bounds
        report['geometry_sources_sha256'].update(sources)
    center = np.asarray(SCENE['coordinates']['table_center_xy_m'])
    half = np.asarray(SCENE['coordinates']['table_size_xy_m'])/2
    table_min,table_max = center-half,center+half
    added = []
    for object_id,asset_id,xy,yaw_degrees,sideways in PLACEMENTS:
        spec = physical_spec(asset_id,asset_root,object_id,[*xy,0.])
        yaw = math.radians(yaw_degrees)/2
        if sideways:
            # Rz(yaw)*Ry(pi/2): no wrench or constraint is applied. The head
            # initially supports the lying bolt; actual settling is unqualified.
            factor = math.sqrt(.5)
            spec['quaternion_wxyz'] = [math.cos(yaw)*factor,-math.sin(yaw)*factor,
                                       math.cos(yaw)*factor,math.sin(yaw)*factor]
        else: spec['quaternion_wxyz'] = [math.cos(yaw),0.,0.,math.sin(yaw)]
        bounds,sources = _bounds(spec)
        spec['position_m'][2] = report['table_top_z_m']+CLEARANCE_M-float(bounds[0,2])
        bounds,sources = _bounds(spec)
        edge_gap = float(min(*(bounds[0,:2]-table_min),*(table_max-bounds[1,:2])))
        if edge_gap < MINIMUM_GAP_M:
            raise ValueError(f'{object_id} is too close to the table edge')
        gaps = {name:_xy_gap(bounds,other) for name,other in occupied.items()}
        if min(gaps.values()) < MINIMUM_GAP_M:
            closest = min(gaps,key=gaps.get)
            raise ValueError(f'{object_id} has insufficient collision-bounds clearance from {closest}')
        report['items'][object_id] = {'asset_id':asset_id,'collision_bounds_world_m':bounds.tolist(),
            'minimum_existing_xy_gap_m':min(gaps.values()),'closest_existing_id':min(gaps,key=gaps.get),
            'xy_gaps_m':gaps,'table_edge_gap_m':edge_gap,'position_m':deepcopy(spec['position_m']),
            'quaternion_wxyz':deepcopy(spec['quaternion_wxyz'])}
        report['geometry_sources_sha256'].update(sources)
        occupied[object_id] = bounds
        added.append(spec)
    ids = [row['id'] for row in added]
    new_scene['objects'].extend(added)
    new_scene['task_distractors'] = ids
    new_scene.setdefault('mechanical_task_provenance',{})['table_distractors'] = {
        'schema':report['schema'],'layout':LAYOUT,'asset_root':report['asset_root'],'ids':ids,
        'minimum_gap_m':MINIMUM_GAP_M,'native_qualified':False}
    new_task['distractors'] = {'schema':'mechanical_task_distractors.v1','layout':LAYOUT,'ids':ids,
        'roles':{row['id']:row['asset_id'] for row in added},
        'scored':False,'physical_behavior':'Independent passive movable rigid bodies with enabled collision.',
        'initial_objects_scope':'Only the original task targets remain in evaluation.initial_objects.',
        'source_asset_root':report['asset_root'],'native_qualified':False}
    new_task['physics_accepted'] = False
    new_task['task_policy_success_claimed'] = False
    new_task['qualification_scope'] = 'New table-clutter reset; source geometry/clearance checked, native contact and manipulation not qualified.'
    validate_distractors(new_scene, new_task)
    return new_scene,new_task,report
