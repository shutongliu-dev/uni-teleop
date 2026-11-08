"""Original tabletop drawer, real passive slide, notebook and receiving tray."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from pathlib import Path

from .drawer_rules import configuration, JOINT_PATH
from .task_scene import (_builder, _body, _physics, _label, _quad, _dump, digest,
                         TASK_ROOT, OBJECT_ROOT, DEFAULT_ASSETS_ROOT, PHYSICS)


def build_candidate(output, *, snapshot, version='v1', assets_root=DEFAULT_ASSETS_ROOT):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils
    output, assets_root = Path(output).resolve(), Path(assets_root).resolve()
    if output.exists(): raise FileExistsError('Preserve immutable drawer candidate: '+str(output))
    output.mkdir(parents=True)
    initial = output/'initial.usda'
    stage = Usd.Stage.CreateNew(str(initial))
    world = UsdGeom.Xform.Define(stage, '/World'); UsdGeom.Xform.Define(stage, TASK_ROOT)
    stage.SetDefaultPrim(world.GetPrim()); UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    b, config = _builder(stage), configuration()
    b.material('walnut', (.19,.105,.052), .48)
    b.material('green_cover', (.055,.20,.16), .65)
    b.material('steel', (.32,.35,.38), .28, .8)
    b.material('tray_green', (.13,.28,.22), .78)
    # Fixed shell is a task fixture. Its front is genuinely open; the old room
    # furniture remains byte-identical and never receives invented dynamics.
    b.box('cabinet_bottom',(.57,.23,.785),(.30,.31,.010),'walnut',bevel=.002,collision=True)
    b.box('cabinet_top',(.57,.23,.887),(.30,.31,.014),'walnut',bevel=.003,collision=True)
    b.box('cabinet_left',(.57,.079,.835),(.30,.008,.090),'walnut',bevel=.001,collision=True)
    b.box('cabinet_right',(.57,.381,.835),(.30,.008,.090),'walnut',bevel=.001,collision=True)
    b.box('cabinet_rear',(.714,.23,.835),(.012,.294,.090),'walnut',bevel=.001,collision=True)
    for side, y in [('left',.087),('right',.373)]:
        b.box('slide_'+side,(.57,y,.820),(.270,.006,.022),'steel',bevel=.0006,collision=True)
        for index,x in enumerate((.475,.575,.675)):
            b.cylinder('rail_screw_'+side+str(index),(x,y,.833),.002,.001,'brass',collision=False)
    # An open tray has a real flat floor; generous side clearances allow any
    # notebook heading without exact centering.
    b.box('target_floor',(.49,-.24,.786),(.32,.35,.012),'tray_green',bevel=0.,collision=True)
    for name,x,y,sx,sy in [('left',.336,-.24,.012,.35),('right',.644,-.24,.012,.35),
                          ('front',.49,-.409,.296,.012),('rear',.49,-.071,.296,.012)]:
        b.box('target_'+name,(x,y,.802),(sx,sy,.032),'walnut',bevel=.002,collision=True)
    mat = _label(b,output,assets_root,'drawer_label','观物  /  NOTES  /  拉开取出')
    _quad(b,TASK_ROOT+'/Fixtures/drawer_label',[(.446,.13,.8941),(.694,.13,.8941),(.694,.20,.8941),(.446,.20,.8941)],mat)
    mat = _label(b,output,assets_root,'target_label','笔记本  /  NOTEBOOK')
    _quad(b,TASK_ROOT+'/Fixtures/target_label',[(.37,-.475,.7801),(.61,-.475,.7801),(.61,-.430,.7801),(.37,-.430,.7801)],mat)
    objects = {}
    for ident,mass in [('drawer',.42),('notebook',.11)]:
        path = OBJECT_ROOT+'/'+ident; obj=config['objects'][ident]
        inertia = _body(stage,path,obj['initial_pose'],mass,obj['local_bounds_m'])
        b.base = path+'/Geometry'
        if ident == 'drawer':
            b.box('floor',(0.,0.,-.025),(.245,.270,.010),'oak',bevel=0.,collision=True)
            b.box('front',(-.126,0.,.0025),(.007,.270,.065),'walnut',bevel=.001,collision=True)
            b.box('rear',(.119,0.,0.),(.007,.270,.060),'oak',bevel=.001,collision=True)
            for side,y in [('left',-.131),('right',.131)]:
                b.box(side,(0.,y,0.),(.238,.008,.060),'oak',bevel=.001,collision=True)
            # 70mm wide bar handle, 23mm finger clearance from front panel.
            for side,y in [('left',-.043),('right',.043)]:
                b.box('handle_post_'+side,(-.143,y,.006),(.028,.008,.008),'brass',bevel=.001,collision=True)
            b.box('handle_grip',(-.156,0.,.006),(.008,.094,.010),'brass',bevel=.002,collision=True)
        else:
            b.box('pages',(0.,0.,0.),(.086,.106,.020),'paper',bevel=.001,collision=True)
            for suffix,z in [('front',.011),('back',-.011)]:
                b.box('cover_'+suffix,(0.,0.,z),(.09,.11,.002),'green_cover',bevel=.0005,collision=True)
            b.box('spine',(-.044,0.,0.),(.002,.110,.020),'green_cover',bevel=.0005,collision=True)
            for line,z in enumerate((-.007,-.003,.001,.005,.009)):
                b.box('page_edge_'+str(line),(.0431,0.,z),(.00012,.103,.0002),'ivory',bevel=0.,collision=False)
            mat = _label(b,output,assets_root,'notebook_cover','观物  ·  工作笔记',(216,194,135),(25,65,51))
            _quad(b,path+'/Geometry/cover_print',[(-.038,-.025,.01201),(.038,-.025,.01201),(.038,.025,.01201),(-.038,.025,.01201)],mat)
        objects[ident] = {'path': path, 'mass_kg': mass,
            'mass_basis': 'Original scaled wood drawer / pocket hardback engineering mass; not measured calibration.',
            'diagonal_inertia_kg_m2': inertia, 'local_bounds_m': deepcopy(obj['local_bounds_m']),
            'collider_paths': _physics(stage,stage.GetPrimAtPath(path))}
    b.base = TASK_ROOT+'/Fixtures'
    _physics(stage,stage.GetPrimAtPath(b.base))
    joint = UsdPhysics.PrismaticJoint.Define(stage,JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([OBJECT_ROOT+'/drawer'])
    joint.CreateAxisAttr('X'); joint.CreateLowerLimitAttr(0.); joint.CreateUpperLimitAttr(.16)
    joint.CreateLocalPos0Attr(Gf.Vec3f(.57,.23,.825)); joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(0, Gf.Vec3f(0,0,1)))
    joint.CreateLocalRot1Attr(Gf.Quatf(0, Gf.Vec3f(0,0,1)))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
    drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(0.); drive.CreateDampingAttr(5.)
    drive.CreateTargetVelocityAttr(0.); drive.CreateMaxForceAttr(15.)
    stage.GetRootLayer().Save()
    dependencies = {}
    layers, paths, unresolved = UsdUtils.ComputeAllDependencies(str(initial))
    if unresolved: raise ValueError('Unresolved drawer dependency '+str(unresolved))
    for asset in [*(layer.realPath for layer in layers),*map(str,paths)]:
        if asset and Path(asset).resolve() != initial: dependencies[str(Path(asset).resolve())] = digest(asset)
    for relative in ('fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf','fonts/noto_cjk_sc/v1/LICENSE.txt'):
        source=assets_root/relative; dependencies[str(source)] = digest(source)
    _dump(output/'SOURCE.json', {'schema':'study_drawer_source.v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_snapshot':snapshot, 'authorship':'Original task-owned tabletop cabinet, constrained drawer, notebook and target tray.',
        'license':'Original geometry and rendered labels CC0-1.0; Noto font remains OFL-1.1.',
        'physics':PHYSICS, 'joint':'Native prismatic rail with passive 5 N s/m damping; no position target or attraction.',
        'qualification':'Authored candidate, no robot manipulation claim.'})
    (output/'LICENSE.txt').write_text('Original geometry and rendered labels: CC0-1.0.\nNoto Sans CJK SC: SIL OFL-1.1, original font license retained.\n')
    metadata={'schema':'study_task_metadata.v1','task_id':'drawer_retrieval','configuration':config,
        'objects':objects,'fixtures':b.records,'joint_paths':[JOINT_PATH], 'source_snapshot':snapshot,
        'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'metadata.json',metadata)
    files={str(p.relative_to(output)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    dependencies.update({str(output/k):v['sha256'] for k,v in files.items() if k!='initial.usda'})
    manifest={'schema':'study_task_candidate.v1','task_id':'drawer_retrieval','version':version,
        'initial_usd':str(initial),'initial_usd_sha256':digest(initial),'metadata_file':'metadata.json',
        'metadata_sha256':digest(output/'metadata.json'),'bodies':{n:OBJECT_ROOT+'/'+n for n in config['objects']},
        'joint_paths':[JOINT_PATH],'configuration_sha256':config['configuration_sha256'], 'files':files,
        'dependencies_sha256':dependencies,'source_snapshot':snapshot,'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'manifest.json',manifest)
    return manifest


def main():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--version',default='v1')
    args=parser.parse_args()
    snapshot=verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=('scenes.workrooms.study.drawer_assets',),entrypoint_file=Path(__file__))
    if snapshot is None: parser.error('Build from a complete frozen source snapshot')
    result=build_candidate(args.output,snapshot=snapshot,version=args.version)
    print(result['initial_usd_sha256']); print(digest(args.output/'manifest.json'))


if __name__=='__main__': main()
