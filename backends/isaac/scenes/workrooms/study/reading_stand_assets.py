"""Original task-owned reading stand, passive hinge and sliding load pin."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from pathlib import Path

from .reading_stand_rules import configuration, HINGE, PIN_CENTER, JOINT_PATHS
from .task_scene import (_builder, _body, _physics, _dump, digest, TASK_ROOT,
                         OBJECT_ROOT, DEFAULT_ASSETS_ROOT, PHYSICS)


def build_candidate(output, *, snapshot, version='v1'):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous stand assets: '+str(output))
    output.mkdir(parents=True)
    initial=output/'initial.usda';stage=Usd.Stage.CreateNew(str(initial))
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage,TASK_ROOT);UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    b=_builder(stage);c=configuration()
    b.material('walnut',(.24,.125,.065),.48)
    b.material('blue_metal',(.07,.16,.19),.4,.65)
    b.material('steel',(.32,.35,.38),.27,.8)
    b.material('cover',(.035,.18,.13),.70)
    b.box('base_plate',(.46,0.,.788),(.36,.33,.016),'walnut',bevel=.003,collision=True)
    for x in (.31,.61):
        for y in (-.145,.145):b.cylinder('rubber_foot_'+str(x)+'_'+str(y),(x,y,.781),.012,.002,'rubber',collision=False)
    # Hinge is a declared world-anchored passive revolute joint. Bronze knuckles
    # and screws show the mount without introducing redundant constraints.
    for i,y in enumerate((-.115,0.,.115)):
        b.cylinder('hinge_knuckle_'+str(i),(HINGE[0],y,HINGE[2]),.010,.035,'brass',rotation=(90,0,0),collision=False)
        b.box('hinge_mount_'+str(i),(HINGE[0]+.015,y,.801),(.045,.025,.010),'blue_metal',bevel=.001,collision=False)
    # Side guide tower is visibly outside the reading face. Pin carried by its
    # physical prismatic joint; the rod itself physically bears the panel load.
    x,_,z=PIN_CENTER
    b.box('pin_guide_tower',(x,.15,(.796+z-.025)/2),(.045,.024,z-.025-.796),'blue_metal',bevel=.002,collision=True)
    b.box('guide_top',(x,.15,z+.024),(.045,.024,.012),'steel',bevel=.001,collision=True)
    for suffix,px in [('near',x-.021),('far',x+.021)]:
        b.box('guide_side_'+suffix,(px,.15,z),(.008,.024,.036),'steel',bevel=.001,collision=True)
    objects={}
    for ident,mass in [('panel',.68),('support_pin',.11),('book',.22)]:
        path=OBJECT_ROOT+'/'+ident;obj=c['objects'][ident]
        inertia=_body(stage,path,obj['initial_pose'],mass,obj['local_bounds_m'])
        b.base=path+'/Geometry'
        if ident=='panel':
            b.box('reading_face',(.16,0.,0.),(.32,.25,.012),'walnut',bevel=.002,collision=True)
            b.box('lower_lip',(.008,0.,.021),(.016,.26,.048),'blue_metal',bevel=.001,collision=True)
            for side,y in [('left',-.11),('right',.11)]:
                b.box('handle_post_'+side,(.302,y,.012),(.020,.012,.024),'steel',bevel=.001,collision=True)
            b.cylinder('handle_grip',(.304,0.,.026),.010,.23,'blue_metal',rotation=(90,0,0),collision=True)
            # Decorative edge inlay and page-grip pads are actual component sizes.
            for i,y in enumerate((-.09,.09)):
                b.box('felt_pad_'+str(i),(.14,y,.0063),(.22,.025,.0006),'felt',bevel=0,collision=False)
            mass_api=UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(.151,0.,.004))
        elif ident=='support_pin':
            b.cylinder('shaft',(0.,.02,0.),.010,.32,'steel',rotation=(90,0,0),collision=True)
            b.cylinder('handle',(0.,.20,0.),.018,.04,'blue_metal',rotation=(90,0,0),collision=True)
            UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).CreateCenterOfMassAttr(Gf.Vec3f(0.,.025,0.))
        else:
            b.box('pages',(0.,0.,0.),(.135,.166,.012),'paper',bevel=.001,collision=True)
            for side,z0 in [('front',.007),('back',-.007)]:
                b.box('cover_'+side,(0.,0.,z0),(.14,.17,.002),'cover',bevel=.0007,collision=True)
            b.box('spine',(-.069,0.,0.),(.002,.17,.014),'cover',bevel=.0006,collision=True)
            for i,z0 in enumerate((-.004,-.002,0.,.002,.004)):
                b.box('page_line_'+str(i),(.0676,0.,z0),(.0002,.159,.0002),'ivory',bevel=0,collision=False)
            for i in range(3):b.box('cover_decoration_'+str(i),(0.,-.037+i*.02,.0081),(.08,.001,.0001),'brass',bevel=0,collision=False)
        body=stage.GetPrimAtPath(path)
        body.AddAppliedSchema('PhysxRigidBodyAPI')
        for name,value,kind in [('sleepThreshold',0.,Sdf.ValueTypeNames.Float),
            ('solverPositionIterationCount',16,Sdf.ValueTypeNames.Int),('solverVelocityIterationCount',4,Sdf.ValueTypeNames.Int),
            ('linearDamping',.02,Sdf.ValueTypeNames.Float),('angularDamping',.02,Sdf.ValueTypeNames.Float)]:
            body.CreateAttribute('physxRigidBody:'+name,kind).Set(value)
        objects[ident]={'path':path,'mass_kg':mass,'diagonal_inertia_kg_m2':inertia,
            'center_of_mass_m':list(UsdPhysics.MassAPI(body).GetCenterOfMassAttr().Get()),
            'mass_basis':'Declared wood panel/ledge, hollow metal support-pin and paper hardback engineering priors; not calibrated.',
            'local_bounds_m':deepcopy(obj['local_bounds_m']),'collider_paths':_physics(stage,body)}
    b.base=TASK_ROOT+'/Fixtures';_physics(stage,stage.GetPrimAtPath(b.base))
    hinge=UsdPhysics.RevoluteJoint.Define(stage,JOINT_PATHS[0]);hinge.CreateBody1Rel().SetTargets([OBJECT_ROOT+'/panel'])
    hinge.CreateAxisAttr('Y');hinge.CreateLowerLimitAttr(-80.);hinge.CreateUpperLimitAttr(0.)
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*HINGE));hinge.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1));hinge.CreateLocalRot1Attr(Gf.Quatf(1))
    drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(.06);drive.CreateTargetVelocityAttr(0.);drive.CreateMaxForceAttr(.5)
    pin=UsdPhysics.PrismaticJoint.Define(stage,JOINT_PATHS[1]);pin.CreateBody1Rel().SetTargets([OBJECT_ROOT+'/support_pin'])
    pin.CreateAxisAttr('Y');pin.CreateLowerLimitAttr(0.);pin.CreateUpperLimitAttr(.28)
    pin.CreateLocalPos0Attr(Gf.Vec3f(*PIN_CENTER));pin.CreateLocalPos1Attr(Gf.Vec3f(0))
    pin.CreateLocalRot0Attr(Gf.Quatf(1));pin.CreateLocalRot1Attr(Gf.Quatf(1))
    slide=UsdPhysics.DriveAPI.Apply(pin.GetPrim(),'linear');slide.CreateTypeAttr('force')
    slide.CreateStiffnessAttr(0.);slide.CreateDampingAttr(1.5);slide.CreateTargetVelocityAttr(0.);slide.CreateMaxForceAttr(8.)
    stage.GetRootLayer().Save()
    dependencies={}
    layers,paths,unresolved=UsdUtils.ComputeAllDependencies(str(initial))
    if unresolved:raise ValueError('Unresolved stand assets '+str(unresolved))
    for asset in [*(layer.realPath for layer in layers),*map(str,paths)]:
        if asset and Path(asset).resolve()!=initial:dependencies[str(Path(asset).resolve())]=digest(asset)
    _dump(output/'SOURCE.json',{'schema':'study_reading_stand_source.v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_snapshot':snapshot,'authorship':'Original task-owned hinged reading panel, passive sliding load pin, guide and book.',
        'license':'CC0-1.0 original geometry/materials','physics':PHYSICS,
        'mechanism':'World-anchored passive revolute panel and transverse prismatic pin; zero stiffness, no pose drives or conditional lock. Real rod-panel contact holds the reading angle.'})
    (output/'LICENSE.txt').write_text('Original geometry and materials: CC0-1.0.\n')
    metadata={'schema':'study_task_metadata.v1','task_id':'reading_stand','configuration':c,'objects':objects,
        'fixtures':b.records,'joint_paths':JOINT_PATHS,'interaction_targets':c['interaction_targets'],
        'source_snapshot':snapshot,'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'metadata.json',metadata)
    files={str(p.relative_to(output)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    dependencies.update({str(output/k):v['sha256'] for k,v in files.items() if k!='initial.usda'})
    manifest={'schema':'study_task_candidate.v1','task_id':'reading_stand','version':version,
        'initial_usd':str(initial),'initial_usd_sha256':digest(initial),'metadata_file':'metadata.json',
        'metadata_sha256':digest(output/'metadata.json'),'bodies':{n:OBJECT_ROOT+'/'+n for n in c['objects']},
        'joint_paths':JOINT_PATHS,'configuration_sha256':c['configuration_sha256'],'files':files,
        'dependencies_sha256':dependencies,'source_snapshot':snapshot,'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'manifest.json',manifest);return manifest


def main():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--version',default='v1')
    a=p.parse_args();source=verify_if_snapshot(Path(__file__).resolve().parents[3],required_modules=('scenes.workrooms.study.reading_stand_assets',),entrypoint_file=Path(__file__))
    if source is None:p.error('Build from a complete frozen source snapshot')
    build_candidate(a.output,snapshot=source,version=a.version);print(digest(a.output/'manifest.json'))


if __name__=='__main__':main()
