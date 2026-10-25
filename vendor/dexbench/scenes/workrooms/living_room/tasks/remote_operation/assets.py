"""Procedural remote/holder candidate, using real USD rigid bodies and joints.

Pure math + pxr authoring only. No simulator, animation, task score or release.
The two force drives have constant zero targets: they model passive return
springs, not controller commands. Their physical priors remain uncalibrated.
"""
from __future__ import annotations

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_VERSION = 'living_remote_physical_v1'
TABLE_TOP_M = .78
REMOTE_SHELL_SIZE_M = (.055, .160, .020)  # width, length, shell thickness
BUTTON_SIZE_M = (.022, .024, .004)
BUTTON_TRAVEL_M = .002
BUTTON_CENTRES_M = {'power': (0., .0215, .010), 'confirm': (0., .0565, .010)}
BUTTON_SPRING_N_M = 600.
BUTTON_DAMPING_N_S_M = 1.2


def _rounded_outline(sx, sy, radius, segments=10):
    points = []
    for cx, cy, start in ((sx/2-radius, sy/2-radius, 0.),
                           (-sx/2+radius, sy/2-radius, math.pi/2),
                           (-sx/2+radius, -sy/2+radius, math.pi),
                           (sx/2-radius, -sy/2+radius, 3*math.pi/2)):
        for i in range(segments+1):
            angle = start + i*math.pi/(2*segments)
            points.append((cx+radius*math.cos(angle), cy+radius*math.sin(angle)))
    return points


def _clip(outline, axis, bound, positive):
    """Clip a convex polygon at an axis-aligned line, preserving its winding."""
    result = []
    for a, b in zip(outline, outline[1:]+outline[:1]):
        ia, ib = (a[axis]-bound)*positive >= -1.e-12, (b[axis]-bound)*positive >= -1.e-12
        if ia:
            result.append(a)
        if ia != ib:
            t = (bound-a[axis])/(b[axis]-a[axis])
            result.append(tuple(a[j]+t*(b[j]-a[j]) for j in (0, 1)))
    return result


def _material(stage, path, color, roughness=.45, metallic=0.):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _mesh(stage, path, outline, bottom, top, material, *, bevel=0., collider=True):
    # Convex component, never a convex hull across an opening. Outer dimensions
    # are exact. The bevel profile cuts into the component, not out of it.
    if len(outline) < 3 or top <= bottom:
        raise ValueError('A convex component needs a positive volume')
    profile = [(bottom, 0.), (top, 0.)]
    if bevel:
        profile = [(bottom, bevel), (bottom+bevel, 0.), (top-bevel, 0.), (top, bevel)]
    # A radial inset suffices for these cosmetic edge bevels; collision uses the
    # identical closed convex surface, not a sharp enclosing placeholder box.
    radius = max(math.hypot(*p) for p in outline)
    points = [(x*(1-inset/radius), y*(1-inset/radius), z)
              for z, inset in profile for x, y in outline]
    n = len(outline)
    faces = [list(reversed(range(n))), list(range((len(profile)-1)*n, len(profile)*n))]
    for layer in range(len(profile)-1):
        for i in range(n):
            a, b = layer*n+i, layer*n+(i+1)%n
            faces.append([a, b, b+n, a+n])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateOrientationAttr('rightHanded')
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        # Raw schema authoring keeps this utility usable in the CPU USD wheel,
        # which does not load Isaac's PhysxSchema Python module.
        previous = mesh.GetPrim().GetMetadata('apiSchemas').GetAppliedItems()
        mesh.GetPrim().SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*previous, 'PhysxCollisionAPI']))
        for key, value in (('contactOffset', .0001), ('restOffset', 0.)):
            mesh.GetPrim().CreateAttribute('physxCollision:'+key, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    return mesh


def _box(stage, path, size, center, material, *, corner=.001, bevel=.0003, collider=True):
    sx, sy, sz = size
    if corner >= min(sx, sy)/2:
        corner = min(sx, sy)*.24
    mesh = _mesh(stage, path, _rounded_outline(sx, sy, corner), -sz/2, sz/2,
                 material, bevel=min(bevel, sz*.2), collider=collider)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*center))
    return mesh


def _body(stage, path, mass, size, center=(0., 0., 0.), *, center_of_mass=(0., 0., 0.)):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*center))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    physical = UsdPhysics.MassAPI.Apply(body.GetPrim())
    physical.CreateMassAttr(mass)
    physical.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    sx, sy, sz = size
    physical.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*(sy*sy+sz*sz)/12,
        mass*(sx*sx+sz*sz)/12, mass*(sx*sx+sy*sy)/12))
    physical.CreatePrincipalAxesAttr(Gf.Quatf(1.))
    return body


def _stroke(stage, path, points, material, width=.00065):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr('linear')
    curve.CreateWrapAttr('nonperiodic')
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation('constant')
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
    return curve


def author_task(stage, root_path='/World/LivingTask', *, presentation=False):
    """Author a metre/Z-up candidate under a new path; return JSON metadata.

    presentation changes only the initial remote pose to show its face. Both
    variants retain actual rigid-body/joint definitions. Neither is a native
    validated reset, manipulation success or an immutable task release.
    """
    root_path = str(root_path)
    path = Sdf.Path(root_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError('root_path must be an absolute non-root prim path')
    if stage.GetPrimAtPath(path):
        raise ValueError('Refusing to overwrite existing task prim: '+root_path)
    if type(presentation) is not bool:
        raise ValueError('presentation must be boolean')
    if stage.HasAuthoredMetadata('metersPerUnit') and UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Remote author requires a metre stage')
    if stage.HasAuthoredMetadata('upAxis') and UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Remote author requires a Z-up stage')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetCustomDataByKey('asset_version', ASSET_VERSION)
    root.GetPrim().SetCustomDataByKey('presentation_only', presentation)
    root.GetPrim().SetCustomDataByKey('native_qualified', False)
    looks = root_path+'/Looks'
    mats = {name: _material(stage, looks+'/'+name, color, rough, metal) for name, color, rough, metal in (
        ('Graphite', (.043, .049, .056), .34, .03), ('Face', (.075, .084, .092), .30, .02),
        ('Rubber', (.022, .026, .029), .72, 0.), ('Power', (.56, .095, .07), .39, 0.),
        ('Confirm', (.065, .26, .31), .37, 0.), ('Ink', (.87, .90, .88), .45, 0.),
        ('Steel', (.31, .34, .36), .26, .78), ('Holder', (.55, .40, .25), .42, 0.),
        ('Lens', (.17, .48, .36), .15, .08))}
    physics = UsdPhysics.MaterialAPI.Apply(mats['Graphite'].GetPrim())
    physics.CreateStaticFrictionAttr(.6); physics.CreateDynamicFrictionAttr(.45); physics.CreateRestitutionAttr(.02)
    rubber = UsdPhysics.MaterialAPI.Apply(mats['Rubber'].GetPrim())
    rubber.CreateStaticFrictionAttr(.75); rubber.CreateDynamicFrictionAttr(.55); rubber.CreateRestitutionAttr(.02)
    rig_path = root_path+'/Remote'
    rig = UsdGeom.Xform.Define(stage, rig_path)
    remote_position = (.49, -.12, TABLE_TOP_M+(.0105 if presentation else .01365))
    rig.AddTranslateOp().Set(Gf.Vec3d(*remote_position))
    rig.AddRotateXYZOp().Set(Gf.Vec3f(0., 0. if presentation else 180., -12.))
    body_path = rig_path+'/Body'
    body = _body(stage, body_path, .108, REMOTE_SHELL_SIZE_M)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mats['Graphite'], materialPurpose='physics')
    _box(stage, body_path+'/LowerShell', (.055, .160, .016), (0., 0., -.002), mats['Graphite'], corner=.010, bevel=.0013)
    # The deck is a union of disjoint convex clipped cells. The two central
    # cells are omitted, leaving real 26 x 27 mm button apertures.
    outline = _rounded_outline(.055, .160, .010)
    xs, ys = [-.0275, -.013, .013, .0275], [-.080, .008, .035, .043, .070, .080]
    for ix in range(3):
        cuts = ys if ix == 1 else [-.080, .080]
        for iy in range(len(cuts)-1):
            if ix == 1 and iy in (1, 3):
                continue
            polygon = outline
            for axis, low, high in ((0, xs[ix], xs[ix+1]), (1, cuts[iy], cuts[iy+1])):
                polygon = _clip(_clip(polygon, axis, low, 1), axis, high, -1)
            _mesh(stage, body_path+f'/Deck_{ix}_{iy}', polygon, .006, .010, mats['Face'])
    # Raised end guards carry the face-down reset above the unpressed buttons.
    for name, y in (('FrontGuard', -.074), ('RearGuard', .075)):
        _box(stage, body_path+'/'+name, (.040, .006, .0035), (0., y, .01175), mats['Graphite'], corner=.002, bevel=.00045)
    _box(stage, body_path+'/BatteryDoor', (.035, .064, .0004), (0., -.030, -.0102), mats['Rubber'], corner=.004, bevel=.00004)
    for side, x in (('Left', -.018), ('Right', .018)):
        foot = _box(stage, body_path+'/BackFoot'+side, (.008, .006, .0004), (x, .055, -.0102),
                    mats['Rubber'], corner=.001, bevel=.00004)
        UsdShade.MaterialBindingAPI.Apply(foot.GetPrim()).Bind(mats['Rubber'], materialPurpose='physics')
    for i, y in enumerate((-.066, .063)):
        _box(stage, body_path+f'/BackScrew_{i}', (.0028, .0028, .00012), (0., y, -.01006), mats['Steel'], corner=.0013, bevel=.00001, collider=False)
        _stroke(stage, body_path+f'/ScrewSlot_{i}', [(-.0008, y, -.01014), (.0008, y, -.01014)], mats['Graphite'], .00022)
    _box(stage, body_path+'/IndicatorLens', (.007, .0025, .0003), (0., -.010, .01015), mats['Lens'], corner=.0007, bevel=.00003, collider=False)
    for i in range(5):
        _box(stage, body_path+f'/GripRib_{i}', (.024, .0010, .00018), (0., -.038-i*.004, .01009), mats['Rubber'], corner=.0003, bevel=.00002, collider=False)
    joint_paths, button_paths = {}, {}
    for name, center in BUTTON_CENTRES_M.items():
        button_path = rig_path+'/'+name.title()+'Button'
        button_paths[name] = button_path
        cap = _body(stage, button_path, .004, BUTTON_SIZE_M, center)
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(mats['Rubber'], materialPurpose='physics')
        _box(stage, button_path+'/Cap', BUTTON_SIZE_M, (0., 0., 0.), mats[name.title()], corner=.005, bevel=.0006)
        z = .00235
        if name == 'power':
            arc = [(.005*math.cos(a), .005*math.sin(a), z)
                   for a in [math.radians(135+i*270/32) for i in range(33)]]
            _stroke(stage, button_path+'/PowerRing', arc, mats['Ink'])
            _stroke(stage, button_path+'/PowerStem', [(0., .0062, z), (0., .0005, z)], mats['Ink'])
        else:
            _stroke(stage, button_path+'/ConfirmMark', [(-.005, -.0002, z), (-.0015, -.0035, z), (.0055, .004, z)], mats['Ink'], .00085)
        joint_path = root_path+'/Joints/'+name.title()
        joint_paths[name] = joint_path
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(button_path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*center))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.)); joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateAxisAttr('Z')
        joint.CreateLowerLimitAttr(-BUTTON_TRAVEL_M); joint.CreateUpperLimitAttr(0.)
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateJointEnabledAttr(True)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
        drive.CreateStiffnessAttr(BUTTON_SPRING_N_M)
        drive.CreateDampingAttr(BUTTON_DAMPING_N_S_M)
        drive.CreateMaxForceAttr(10.)
    holder_path = root_path+'/Holder'
    holder = _body(stage, holder_path, .240, (.083, .188, .038), (.82, .075, TABLE_TOP_M),
                   center_of_mass=(0., .0070, .0095))
    UsdShade.MaterialBindingAPI.Apply(holder.GetPrim()).Bind(mats['Graphite'], materialPurpose='physics')
    _box(stage, holder_path+'/Base', (.083, .188, .006), (0., 0., .003), mats['Holder'], corner=.006, bevel=.0006)
    liner = _box(stage, holder_path+'/Liner', (.065, .168, .001), (0., 0., .0065), mats['Rubber'], corner=.004, bevel=.0001)
    UsdShade.MaterialBindingAPI.Apply(liner.GetPrim()).Bind(mats['Rubber'], materialPurpose='physics')
    for name, x in (('LeftRail', -.0375), ('RightRail', .0375)):
        _box(stage, holder_path+'/'+name, (.008, .172, .020), (x, 0., .016), mats['Holder'], corner=.002, bevel=.0006)
    _box(stage, holder_path+'/FrontLip', (.083, .008, .010), (0., -.090, .011), mats['Holder'], corner=.002, bevel=.0005)
    _box(stage, holder_path+'/BackStop', (.083, .008, .032), (0., .090, .022), mats['Holder'], corner=.002, bevel=.0007)
    _stroke(stage, holder_path+'/OrientationMark', [(-.004, .091, .0383), (0., .094, .0383), (.004, .091, .0383)], mats['Ink'], .0007)
    metadata = {'schema': 'living_remote_asset_candidate.v1', 'asset_version': ASSET_VERSION,
        'task_id': 'remote_operation', 'root_path': root_path,
        'presentation': presentation, 'native_qualified': False, 'native_execution_available': False,
        'scope': 'Authored physical candidate only; no native reset, spring/contact acceptance, scorer or manipulation qualification',
        'remote': {'assembly_path': rig_path, 'body_path': body_path, 'button_paths': button_paths,
            'shell_size_xyz_m': list(REMOTE_SHELL_SIZE_M), 'envelope_xyz_m': [.055, .160, .0239],
            'body_mass_kg': .108, 'button_mass_kg': .004, 'total_mass_kg': .116,
            'position_m': list(remote_position), 'face_up': presentation,
            'reset_support': 'Two raised end guards keep face-down buttons clear of the table',
            'grasp_scope': '55 mm grip width and unobstructed lower body; SHARPA reach/grasp must be validated natively'},
        'buttons': {name: {'body_path': button_paths[name], 'joint_path': joint_paths[name],
            'axis': 'local Z', 'limits_m': [-BUTTON_TRAVEL_M, 0.], 'rest_coordinate_m': 0.,
            'spring_n_m': BUTTON_SPRING_N_M, 'damping_n_s_m': BUTTON_DAMPING_N_S_M,
            'rest_surface_m': .012, 'full_press_surface_m': .010,
            'press_threshold_candidate_m': -.0014, 'release_threshold_candidate_m': -.0003,
            'mechanism': 'Constant zero-target force drive represents a passive linear spring; no animation or command input',
            'thresholds_calibrated': False} for name in BUTTON_CENTRES_M},
        'holder': {'body_path': holder_path, 'position_m': [.82, .075, TABLE_TOP_M],
            'opening_xy_m': [.067, .172], 'base_top_z_m': TABLE_TOP_M+.006,
            'liner_top_z_m': TABLE_TOP_M+.007, 'mass_kg': .240,
            'fit_clearance_each_side_xy_m': [.006, .006],
            'opening': 'Top-entry open cradle, independent free body, real base/liner and four separate boundary walls; no hidden fixture'},
        'collision': {'approximation': 'Exact convex hull of each convex mesh component; open deck and holder remain compound',
            'button_body_adjacent_collision': False, 'button_to_button_filter': None,
            'body_inertia': 'Explicit homogeneous envelope estimate; holder COM approximates compound volume centroid. Engineering prior, not measured or dynamically qualified',
            'contact_offset_m': .0001, 'rest_offset_m': 0.,
            'friction': {'shell_static': .6, 'shell_dynamic': .45, 'rubber_static': .75, 'rubber_dynamic': .55},
            'material_scope': 'Uncalibrated task-asset priors; no shared robot or solver settings changed'},
        'authors_only': ['remote assembly', 'two passive prismatic buttons', 'open holder', 'materials'],
        'not_authored': ['room', 'robot', 'light', 'PhysicsScene', 'task evaluator', 'release']}
    return metadata
