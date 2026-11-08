"""Passive spring slide clamp and sealed tube; no attachment to the tube.

The two-piece clamp is a declared parallel-jaw laboratory teaching tool. Pull
the broad thumb paddle to open it; its fixed passive spring closes the jaw.
Only dry contact/friction can carry the independent tube.
"""
from pxr import Gf, Sdf, UsdPhysics, UsdShade
from scenes.workrooms.laboratory.object_geometry import Author, box, cylinder, pose, ring, schema, attr

TASK_ID = 'tube_clamp_transfer'
SPRING_TARGET_M = -.018
SPRING_STIFFNESS_N_M = 200.
SPRING_DAMPING_N_S_M = 1.2


def author_task(stage, root_path='/World/LaboratoryTask', *, presentation=False):
    a = Author(stage, TASK_ID, root_path, presentation)
    # All pieces use original procedural geometry and declared, uncalibrated
    # mass/friction priors. Broad pads and paddle are reachable manipulation
    # surfaces, rather than miniature wire tongs.
    pad = UsdShade.Material.Define(stage, root_path+'/Materials/GripPad')
    p = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    p.CreateStaticFrictionAttr(.90); p.CreateDynamicFrictionAttr(.75); p.CreateRestitutionAttr(0.)
    schema(pad.GetPrim(), 'PhysxMaterialAPI')
    attr(pad.GetPrim(), 'physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token, 'max')
    fixtures = {}
    for name, x in [('source_holder', .48), ('receiver_holder', .92)]:
        parts = [box((.075,.075,.006),(0,0,.003)), *ring(.019,.027,.006,.024,24)]
        fixtures[name] = a.body(name, pose(x,.57,.90), parts, color='teal' if name=='receiver_holder' else 'white')
    a.body('tool_rest', pose(.70,.50,.90),
        [box((.25,.10,.006),(-.025,0,.003)),
         box((.010,.10,.014),(-.145,0,.013)), box((.010,.10,.014),(.095,0,.013))], color='white')
    frame_pose = pose(.70,.50,.9063)
    frame_parts = [box((.145,.036,.024),(-.0625,0,.012)),
        box((.040,.080,.018),(.010,0,.025)),
        box((.038,.012,.044),(.060,-.024,.061)),
        box((.025,.018,.040),(.036,-.024,.045))]
    frame = a.body('clamp_frame', frame_pose, frame_parts, .110, 'teal', nominal_support='tool_rest')
    jaw_pose = pose(frame_pose[0],frame_pose[1]+SPRING_TARGET_M,frame_pose[2])
    jaw_parts = [box((.038,.012,.044),(.060,.028,.061)),
        box((.028,.020,.032),(.033,.028,.041)),
        box((.025,.075,.014),(.014,.015,.042)),
        box((.040,.014,.036),(.0,.059,.053))]
    jaw = a.body('clamp_jaw', jaw_pose, jaw_parts, .035, 'steel', joint_supported_by='clamp_frame')
    # Only the opposed jaw faces have high dry-friction rubber. Connected
    # clamp bodies exclude mutual collision through the declared slide joint.
    for path in (frame['collider_paths'][2], jaw['collider_paths'][0]):
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(pad, materialPurpose='physics')
    joint_path = root_path+'/Joints/JawSlide'
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([frame['body_path']]); joint.CreateBody1Rel().SetTargets([jaw['body_path']])
    joint.CreateAxisAttr('Y'); joint.CreateLowerLimitAttr(-.020); joint.CreateUpperLimitAttr(.030)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,0)); joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(SPRING_STIFFNESS_N_M)
    drive.CreateDampingAttr(SPRING_DAMPING_N_S_M); drive.CreateTargetPositionAttr(SPRING_TARGET_M)
    drive.CreateTargetVelocityAttr(0.); drive.CreateMaxForceAttr(12.)
    a.meta['joints'] = {'jaw_slide': joint_path}
    tube_pose = pose(.92 if presentation else .48,.57,.9063)
    tube = a.body('tube', tube_pose,
        [cylinder(.015,.110,(0,0,.055),48), cylinder(.016,.010,(0,0,.115),48)],
        .035, 'blue', nominal_support='receiver_holder' if presentation else 'source_holder')
    a.meta['task_parameters'] = {
        'clamp_model': 'passive_parallel_jaw_spring_slide_v1',
        'spring': {'target_position_m': SPRING_TARGET_M, 'stiffness_n_m': SPRING_STIFFNESS_N_M,
                   'damping_n_s_m': SPRING_DAMPING_N_S_M, 'maximum_force_n': 12.,
                   'force_prior': 'Declared spring hardware; no runtime target changes or tube attachment.'},
        'fixed_pad_actor': 'clamp_frame', 'moving_pad_actor': 'clamp_jaw',
        'fixed_pad_path': frame['collider_paths'][2], 'moving_pad_path': jaw['collider_paths'][0],
        'tube_radius_m': .015, 'tube_length_m': .120,
        'source_position_m': [.48,.57,.906], 'receiver_position_m': [.92,.57,.906],
        'receiver_fixture': 'receiver_holder', 'receiver_radius_m': .019,
        'receiver_position_tolerance_m': .005, 'tube_upright_tolerance_rad': .2617993878,
        'tool_return_position_m': frame_pose[:3], 'tool_return_tolerance_m': .020,
        'tool_return_angle_rad': .25, 'tool_return_fixture': 'tool_rest',
        'minimum_carried_distance_m': .12, 'minimum_lift_m': .035,
        'minimum_bilateral_contact_s': .05, 'grip_contact_memory_s': .010,
        'tube_grip_center_in_frame_m': [.06,-.003,.060],
        'pad_friction': {'static':.90,'dynamic':.75,'combine':'max'},
    }
    a.meta['interaction_targets'] = [
        {'label':'tool_handle', 'reference_position_m':[.64,.50,.942], 'approach_direction':[0,0,-1]},
        {'label':'jaw_thumb_paddle', 'reference_position_m':[.70,.541,.959], 'approach_direction':[0,-1,0]},
        {'label':'tube_pickup', 'reference_position_m':[.48,.57,.966], 'approach_direction':[0,0,-1]},
        {'label':'tube_lift', 'reference_position_m':[.48,.57,1.046], 'approach_direction':[0,0,-1]},
        {'label':'receiver', 'reference_position_m':[.92,.57,.966], 'approach_direction':[0,0,-1]},
        {'label':'tool_return', 'reference_position_m':[.64,.50,.942], 'approach_direction':[0,0,-1]},
    ]
    target_bodies=[('clamp_frame',[-.06,0,.0357]),('clamp_jaw',[0,.059,.0527]),
                   ('tube',[0,0,.0597]),('tube',[0,0,.1397]),('receiver_holder',[0,0,.066]),
                   ('clamp_frame',[-.06,0,.0357])]
    for target,(body,point) in zip(a.meta['interaction_targets'],target_bodies):
        target.update(body_id=body,local_position_m=point)
    a.meta['limitations'] = ['Parallel-jaw spring instrument is a declared laboratory teaching-tool approximation; not a calibrated commercial clamp.',
        'The sealed rigid tube has no liquid; transfer is through real dry pad/tube contact only.']
    return a.meta
