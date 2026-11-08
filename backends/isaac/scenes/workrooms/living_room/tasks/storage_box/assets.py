"""Task-owned hinged storage box with two physical rotary catches.

All mechanisms are dynamic rigid bodies connected by passive revolute joints.
The catches obstruct the lid geometrically; no lock callback changes physics.
Dimensions, material coefficients and masses are explicit simulation priors.
"""
from __future__ import annotations
import math
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from scenes.workrooms.living_room.tasks.remote_operation.assets import _material, _box, _body

ROOT = '/World/LivingTask'
BOX_POSITION = [.65, -.12, .78]
LID_POSITION = [.65, 0., .894]
ITEM_POSITION = [.65, -.12, .809]
ITEM_SIZE = [.070, .050, .036]
TRAY_CENTER = [.94, -.12, .783]
ACTORS = {'box': ROOT+'/Box', 'lid': ROOT+'/Lid', 'left_catch': ROOT+'/LeftCatch',
          'right_catch': ROOT+'/RightCatch', 'item': ROOT+'/Item'}


def hinge(stage, name, body0, body1, anchor, axis, limits, damping):
    joint = UsdPhysics.RevoluteJoint.Define(stage, ROOT+'/Joints/'+name)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.)); joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(limits[0]); joint.CreateUpperLimitAttr(limits[1])
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(damping); drive.CreateTargetPositionAttr(0.)
    drive.CreateTargetVelocityAttr(0.); drive.CreateMaxForceAttr(.5)
    return str(joint.GetPath())


def author_task(stage, root_path=ROOT, *, presentation=False):
    if root_path != ROOT or presentation or stage.GetPrimAtPath(ROOT):
        raise ValueError('Storage box requires its fresh canonical task root')
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, ROOT)
    mats = {name: _material(stage, ROOT+'/Looks/'+name, color, rough, metal)
            for name, color, rough, metal in (
                ('Case', (.22, .34, .31), .56, .04), ('Inset', (.13, .20, .18), .7, 0.),
                ('Lid', (.32, .43, .39), .50, .03), ('Brass', (.58, .40, .17), .32, .65),
                ('Steel', (.30, .32, .34), .32, .72), ('Item', (.13, .35, .62), .42, .02),
                ('Tray', (.61, .44, .28), .62, 0.))}
    for material in mats.values():
        physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical.CreateStaticFrictionAttr(.65); physical.CreateDynamicFrictionAttr(.5)
        physical.CreateRestitutionAttr(0.)
    _body(stage, ACTORS['box'], .80, [.30, .24, .11], BOX_POSITION, center_of_mass=[0, 0, .045])
    box = ACTORS['box']
    _box(stage, box+'/Floor', [.30, .24, .010], [0, 0, .005], mats['Case'], corner=.008)
    for name, size, point in (
        ('Front', [.30, .010, .10], [0, -.115, .06]),
        ('Rear', [.30, .010, .10], [0, .115, .06]),
        ('Left', [.010, .22, .10], [-.145, 0, .06]),
        ('Right', [.010, .22, .10], [.145, 0, .06])):
        _box(stage, box+'/'+name, size, point, mats['Case'], corner=.003)
    # Separate wood-coloured liner is visual only, coplanar with the real floor.
    _box(stage, box+'/Liner', [.278, .218, .0003], [0, 0, .0101], mats['Inset'], collider=False)
    lid = ACTORS['lid']
    _body(stage, lid, .12, [.30, .24, .006], LID_POSITION, center_of_mass=[0, -.12, 0])
    _box(stage, lid+'/Panel', [.30, .24, .006], [0, -.12, 0], mats['Lid'], corner=.006)
    _box(stage, lid+'/Inset', [.27, .20, .001], [0, -.12, .0034], mats['Inset'], collider=False)
    for x in (-.028, .028):
        _box(stage, lid+('/HandleLeft' if x < 0 else '/HandleRight'), [.010, .012, .020],
             [x, -.215, .013], mats['Brass'])
    _box(stage, lid+'/HandleBridge', [.066, .012, .010], [0, -.215, .028], mats['Brass'])
    joints = {'lid_hinge': hinge(stage, 'LidHinge', box, lid, [0, .12, .114], 'X', [-100., 0.], .0003)}
    for name, x in (('left_catch', -.10), ('right_catch', .10)):
        path = ACTORS[name]
        _body(stage, path, .025, [.045, .075, .010], [.65+x, -.267, .903], center_of_mass=[0, .028, 0])
        _box(stage, path+'/Tongue', [.045, .075, .010], [0, .028, 0], mats['Brass'], corner=.004)
        _box(stage, path+'/Grip', [.028, .018, .025], [0, .040, .0175], mats['Brass'], corner=.003)
        _box(stage, box+'/'+name+'Post', [.028, .045, .123], [x, -.147, .0615], mats['Steel'], corner=.003)
        joints[name] = hinge(stage, name, box, path, [x, -.147, .123], 'Z',
                             [0., 100.] if name=='left_catch' else [-100., 0.], .0001)
    _body(stage, ACTORS['item'], .08, ITEM_SIZE, ITEM_POSITION)
    _box(stage, ACTORS['item']+'/Shell', ITEM_SIZE, [0,0,0], mats['Item'], corner=.006)
    for x in (-.023, .023):
        _box(stage, ACTORS['item']+('/StripeL' if x < 0 else '/StripeR'), [.004, .036, .0005],
             [x, 0, .0182], mats['Steel'], collider=False)
    tray = ROOT+'/CollectionTray'
    _box(stage, tray+'/Floor', [.18, .22, .006], TRAY_CENTER, mats['Tray'], corner=.006)
    for name, size, offset in (
        ('Left', [.006,.22,.018],[-.087,0,.012]), ('Right',[.006,.22,.018],[.087,0,.012]),
        ('Front',[.168,.006,.018],[0,-.107,.012]),('Rear',[.168,.006,.018],[0,.107,.012])):
        _box(stage,tray+'/'+name,size,[TRAY_CENTER[i]+offset[i] for i in range(3)],mats['Tray'],corner=.002)
    for path in ACTORS.values():
        prim = stage.GetPrimAtPath(path)
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        prim.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float).Set(0.)
        prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(16)
        prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(4)
    angle = math.radians(70)
    targets = [
        ('left_catch_closed',[.55,-.227,.928]), ('right_catch_closed',[.75,-.227,.928]),
        ('left_catch_open',[.51,-.267,.928]), ('right_catch_open',[.79,-.267,.928]),
        ('lid_handle_closed',[.65,-.215,.922]),
        ('lid_handle_open',[.65,-.215*math.cos(angle)+.028*math.sin(angle),.894+.215*math.sin(angle)+.028*math.cos(angle)]),
        ('item_supply',ITEM_POSITION), ('item_collection',[.94,-.12,.804])]
    local_targets={
        'left_catch_closed':('left_catch',[0,.04,.025]), 'left_catch_open':('left_catch',[0,.04,.025]),
        'right_catch_closed':('right_catch',[0,.04,.025]), 'right_catch_open':('right_catch',[0,.04,.025]),
        'lid_handle_closed':('lid',[0,-.215,.028]), 'lid_handle_open':('lid',[0,-.215,.028]),
        'item_supply':('item',[0,0,0])}
    target_rows=[]
    for label,point in targets:
        row={'label':label,'reference_position_m':point,'approach':'above'}
        if label in local_targets:
            row['body_id'],row['local_position_m']=local_targets[label]
        target_rows.append(row)
    return {'schema':'living_storage_box_asset.v1','root_path':ROOT,'task_id':'storage_box',
        'actors': ACTORS, 'joints':joints, 'item_size_m':ITEM_SIZE,
        'initial_positions_m': {'box':BOX_POSITION,'lid':LID_POSITION,'left_catch':[.55,-.267,.903],
            'right_catch':[.75,-.267,.903],'item':ITEM_POSITION},
        'masses_kg':{'box':.8,'lid':.12,'left_catch':.025,'right_catch':.025,'item':.08},
        'table_top_z_m':.78,'tray_floor_path':tray+'/Floor','tray_floor_z_m':.786,
        'tray_inside_xy_m':[[.856,-.224],[1.024,-.016]],
        'interaction_targets':target_rows,
        'mechanism':'Passive revolute lid and two independently rotated collision tongues; no logical locking or pose animation.',
        'physics_basis':'Project-authored engineering priors, uncalibrated masses/friction/damping; all collision geometry authored in metres.',
        'dependencies':[]}
