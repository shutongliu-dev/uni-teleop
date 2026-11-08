"""Two independent graspable assembly parts, a real hollow socket and broad tray."""
import json
from pathlib import Path
from bench.isaacsim.release import digest
from bench.paths import DATA_ROOT

TASK_ID='cooperative_insertion'
ROOT='/World/CooperativeTask'
ASSET_ROOT=Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/cooperative_insertion/v1')


def build_assets(output=ASSET_ROOT, *, visual_details=False, visual_detail_profile='v1',
                 robot_layout_id='mechanical_opposed_dual_right_sharpa_v1'):
    from pxr import Usd, UsdGeom
    from .nist_assets import Author
    from scenes.workrooms.laboratory.object_geometry import box,cylinder,pose,tray_parts
    from .cooperative_robot_layout import stand_geometry, OPPOSED_LAYOUT_ID
    stands=stand_geometry(robot_layout_id)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite an assembly asset')
    output.mkdir(parents=True)
    stage=Usd.Stage.CreateNew(str(output/'task.usda'))
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    a=Author(stage,TASK_ID,ROOT)
    a.meta['schema']='mechanical-cooperative-asset.v1'
    a.meta['source']={'kind':'original_procedural','license':'CC0-1.0','external_assets':[]}
    # 20 mm square male peg inside a 28 mm square socket: real 4 mm side
    # clearance, 52 mm usable depth and a load-bearing bottom. No snap/weld.
    walls=[box((.074,.070,.008),(0,0,.004)),
           box((.018,.064,.052),(-.023,0,.034)),box((.018,.064,.052),(.023,0,.034)),
           box((.028,.018,.052),(0,-.023,.034)),box((.028,.018,.052),(0,.023,.034)),
           box((.048,.028,.018),(0,-.048,.025))]
    sleeve=a.body('sleeve',pose(.65,-.205,.7803),walls,.16,'teal',nominal_support='environment_table')
    peg=a.body('peg',pose(.65,.205,.7803),[box((.020,.020,.080),(0,0,.040)),
        cylinder(.025,.020,(0,0,.090))],.075,'amber',nominal_support='environment_table')
    a.body('collection_tray',pose(.49,0.,.78),tray_parts((.26,.28,.026),floor=.004,wall=.004),color='blue')
    # Two separate original stands with room-floor datum, never room mutation.
    for side,stand in stands.items():
        for item in stand['boxes']:
            p=list(item['position_m']);size=list(item['size_m'])
            if item['name']=='robot_stand_foot':p[2]=.025
            if item['name']=='robot_stand_column':
                size[2]=.705;p[2]=.4025
            a.body(side+'_'+item['name'],pose(*p),[box(size)],color='steel' if item['name'].endswith('mount') else 'black')
    a.meta['task_parameters']={
        'robot_instances':list(stands),
        'contact_roles':{'sleeve':'either distinct hand','peg':'other distinct hand'},
        'socket_axis_local':[0,0,1],'socket_mouth_local_m':[0,0,.060],
        'peg_tip_local_m':[0,0,0],'peg_axis_local':[0,0,1],
        'minimum_insertion_depth_m':.030,'maximum_insertion_depth_m':.058,
        'maximum_lateral_error_m':.006,'maximum_axis_angle_rad':.30,
        'assembly_motion_m':.008,'minimum_sleeve_movement_m':.015,
        'collection_center_m':[.49,0,.844],'collection_half_extents_m':[.126,.136,.060],
        'required_moved_ids':['sleeve','peg'],
        'mating_ids':['peg','sleeve'],'support_ids':['collection_tray'],
        'actual_clearance_per_side_m':.004,
        'goal_note':'Both bodies remain independent; final peg rests inside socket, socket on tray.'}
    a.meta['diagnostic_goal_poses']={'sleeve':pose(.49,0,.7843),'peg':pose(.49,0,.7923)}
    a.meta['known_limitations']=['Synthetic loose fit, no interference press-fit or precision force acceptance.',
        'Same-agent dual-arm contact history required by referee; hand-object strategy belongs to data collection.']
    if robot_layout_id!=OPPOSED_LAYOUT_ID:
        a.meta['robot_layout_id']=robot_layout_id
    if list(stands)==['left','right']:
        a.meta['task_parameters']['robot_stage_scopes']={'left':'Left','right':'Right'}
    if visual_details:
        from .cooperative_visual_details import apply_visual_details
        apply_visual_details(stage, a.meta, profile=visual_detail_profile)
    stage.GetRootLayer().Save()
    (output/'metadata.json').write_text(json.dumps(a.meta,indent=2,sort_keys=True)+'\n')
    manifest={'schema':'mechanical-cooperative-asset.v1','task_id':TASK_ID,'root_prim_path':ROOT,
        'usd_path':str(output/'task.usda'),'usd_sha256':digest(output/'task.usda'),
        'metadata_path':str(output/'metadata.json'),'metadata_sha256':digest(output/'metadata.json'),
        'dependencies':{p:{'sha256':sha} for p,sha in
                        a.meta.get('visual_detail',{}).get('dependencies',{}).items()},
        'physics_accepted':False,'presentation':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest
