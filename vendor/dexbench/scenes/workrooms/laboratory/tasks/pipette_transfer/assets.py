"""A real sliding piston and open laboratory vessels for coarse-volume transfer.

Fluid is the explicitly declared finite-volume component in ``liquid.py``.
The authored liquid surface is an initial-state visual, never a collision body
or a PBD particle claim. No room, robot, physics scene or motion is authored.
"""
from __future__ import annotations

import math
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
from scenes.workrooms.laboratory.visual_details import add_task_visual_details

SCHEMA = 'laboratory_pipette_assets.v1'
BORE_RADIUS_M = .012
STROKE_M = .055
PISTON_AREA_M2 = math.pi * BORE_RADIUS_M ** 2
TABLE_Z_M = .90


def _schema(prim, name):
    applied = list(prim.GetMetadata('apiSchemas').GetAppliedItems())
    if name not in applied:
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*applied, name]))


def _material(stage, path, color, *, opacity=1., roughness=.35):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.47)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateStaticFrictionAttr(.55)
    UsdPhysics.MaterialAPI(mat.GetPrim()).CreateDynamicFrictionAttr(.40)
    UsdPhysics.MaterialAPI(mat.GetPrim()).CreateRestitutionAttr(0.)
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _collision(prim, *, sdf=False):
    UsdPhysics.CollisionAPI.Apply(prim)
    _schema(prim, 'PhysxCollisionAPI')
    for key, value in [('contactOffset', .00015), ('restOffset', 0.)]:
        prim.CreateAttribute('physxCollision:'+key, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    if sdf:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
        _schema(prim, 'PhysxSDFMeshCollisionAPI')
        prim.CreateAttribute('physxSDFMeshCollision:sdfResolution', Sdf.ValueTypeNames.Int, custom=False).Set(256)
        prim.CreateAttribute('physxSDFMeshCollision:sdfSubgridResolution', Sdf.ValueTypeNames.Int, custom=False).Set(6)


def _cube(stage, path, center, size, mat):
    mesh = UsdGeom.Cube.Define(stage, path)
    mesh.CreateSizeAttr(1.)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
    mesh.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(mesh.GetPrim(), mat); _collision(mesh.GetPrim())
    return mesh


def _cylinder(stage, path, radius, height, z, mat, *, collision=True):
    mesh = UsdGeom.Cylinder.Define(stage, path)
    mesh.CreateRadiusAttr(radius); mesh.CreateHeightAttr(height); mesh.CreateAxisAttr('Z')
    mesh.AddTranslateOp().Set(Gf.Vec3d(0., 0., z))
    _bind(mesh.GetPrim(), mat)
    if collision:
        _collision(mesh.GetPrim())
    return mesh


def _shell(stage, path, profile, mat, *, collision=True):
    """Rotate a closed radial cross section: apertures remain truly open."""
    count = 64
    points = [(r*math.cos(2*math.pi*i/count), r*math.sin(2*math.pi*i/count), z)
              for r, z in profile for i in range(count)]
    faces = []
    for j in range(len(profile)):
        k = (j+1) % len(profile)
        for i in range(count):
            n = (i+1) % count
            faces.extend(([j*count+i, j*count+n, k*count+n], [j*count+i, k*count+n, k*count+i]))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none'); mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    _bind(mesh.GetPrim(), mat)
    if collision:
        _collision(mesh.GetPrim(), sdf=True)
    return mesh


def _body(stage, path, position, mass, bounds):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(False)
    physics = UsdPhysics.MassAPI.Apply(body.GetPrim())
    physics.CreateMassAttr(mass)
    lo, hi = bounds
    dims = [hi[i]-lo[i] for i in range(3)]
    physics.CreateCenterOfMassAttr(Gf.Vec3f(*[(hi[i]+lo[i])/2 for i in range(3)]))
    physics.CreateDiagonalInertiaAttr(Gf.Vec3f(*[mass*(dims[(i+1)%3]**2+dims[(i+2)%3]**2)/12 for i in range(3)]))
    _schema(body.GetPrim(), 'PhysxRigidBodyAPI')
    body.GetPrim().CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.)
    return body


def author_task(stage, root_path='/World/LaboratoryTask', *, presentation=False):
    root = Sdf.Path(root_path)
    if not root.IsAbsolutePath() or not root.IsPrimPath() or root_path in ('/', '/World'):
        raise ValueError('Use a dedicated absolute task prim path')
    if stage.GetPrimAtPath(root) or type(presentation) is not bool:
        raise ValueError('Task root must be fresh and presentation must be boolean')
    if stage.HasAuthoredMetadata('metersPerUnit') and UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Metre-scale stage required')
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, root_path)
    materials = {name: _material(stage, root_path+'/Looks/'+name, rgb, opacity=opacity) for name,rgb,opacity in (
        ('Glass', (.79,.91,.94), .36), ('Polymer', (.88,.91,.92), 1.),
        ('Blue', (.045,.23,.42), 1.), ('Black', (.018,.023,.028), 1.),
        ('Water', (.20,.58,.76), .65), ('Tray', (.68,.73,.75), 1.))}
    actors, objects, containers = {}, [], {}
    def body(name, position, mass, bounds):
        path=root_path+'/'+name
        _body(stage,path,position,mass,bounds)
        actors[name]=path
        objects.append({'id':name,'body_path':path,'mass_kg':mass,'initial_position_m':list(position),
            'initial_pose_xyzw':[*position,0.,0.,0.,1.], 'local_bounds_m':[list(bounds[0]),list(bounds[1])],
            'half_extents_m':[(bounds[1][i]-bounds[0][i])/2 for i in range(3)]})
        return path
    for name,position,outer,inner,height,mass,initial in (
        ('source',(.62,.53,.90015),.036,.032,.080,.12,80.),
        ('receiver_a',(.56,2.08,.90015),.025,.021,.060,.060,0.),
        ('receiver_b',(.88,2.08,.90015),.025,.021,.060,.060,0.)):
        path=body(name,position,mass,((-outer,-outer,0.),(outer,outer,height)))
        _shell(stage,path+'/HollowVessel',[(.00001,0.),(outer,0.),(outer,height),
            (inner,height),(inner,.004),(.00001,.004)],materials['Glass'])
        surface=path+'/InitialLiquidSurface'
        water_height=max(.00001,initial*1e-6/(math.pi*inner**2))
        water=_cylinder(stage,surface,inner-.0003,water_height,.004+water_height/2,materials['Water'],collision=False)
        if initial == 0:
            water.CreateVisibilityAttr('invisible')
        containers[name]={'body_id':name,'path':path,'radius_m':inner,'floor_z_m':.004,
            'rim_z_m':height,'capacity_ml':math.pi*inner**2*(height-.004)*1e6,
            'initial_ml':initial,'initial_surface_path':surface,'surface_visual_only':True}
        for index in range(1,4):
            _cube(stage,path+f'/Graduation_{index}',(inner+.0015,-.005,.004+index*.013),
                  (.0003,.009,.0006),materials['Blue'])
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path+f'/Graduation_{index}')).CreateCollisionEnabledAttr(False)
    tool_position=(.83,.53,.92315)
    barrel=body('pipette',tool_position,.065,((-.025,-.014,0.),(.025,.014,.089)))
    _shell(stage,barrel+'/Barrel',[(.014,.024),(.014,.086),(.012,.086),(.012,.026)],materials['Glass'])
    _shell(stage,barrel+'/Nozzle',[(.005,0.),(.007,.024),(.014,.024),(.012,.026),(.003,.024),(.003,0.)],materials['Polymer'])
    chamber_visual=_cylinder(stage,barrel+'/ChamberLiquid',.0115,.000001,.0265005,materials['Water'],collision=False)
    chamber_visual.CreateVisibilityAttr('invisible')
    _cube(stage,barrel+'/FingerFlange',(0.,0.,.087),(.050,.023,.004),materials['Blue'])
    # The flange has a real shaft opening; split the central solid into two rails.
    stage.RemovePrim(barrel+'/FingerFlange')
    for side in (-1,1):
        _cube(stage,barrel+f'/FingerFlange_{side:+d}'.replace('+','p').replace('-','m'),
              (side*.015,0.,.087),(.020,.023,.004),materials['Blue'])
    piston=body('plunger',(tool_position[0],tool_position[1],tool_position[2]+.028),.016,
        ((-.013,-.013,-.0015),(.013,.013,.085)))
    _cylinder(stage,piston+'/Seal',.01175,.003,0.,materials['Black'])
    _cylinder(stage,piston+'/Shaft',.0035,.080,.0415,materials['Polymer'])
    _cylinder(stage,piston+'/ThumbPad',.013,.003,.0835,materials['Blue'])
    joint_path=root_path+'/Joints/Plunger'
    joint=UsdPhysics.PrismaticJoint.Define(stage,joint_path)
    joint.CreateBody0Rel().SetTargets([barrel]);joint.CreateBody1Rel().SetTargets([piston])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,.028));joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.));joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(0.);joint.CreateUpperLimitAttr(STROKE_M)
    joint.CreateCollisionEnabledAttr(False)
    # A passive viscous seal only: no commanded position or spring return.
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(4.)
    drive.CreateTargetVelocityAttr(0.);drive.CreateMaxForceAttr(2.)
    cradle=root_path+'/PipetteCradle';UsdGeom.Xform.Define(stage,cradle)
    _cube(stage,cradle+'/Base',(.83,.53,.903),(.100,.100,.006),materials['Tray'])
    for side in (-1,1):
        _cube(stage,cradle+f'/Post_{side+1}',(.83+side*.031,.53,.955),(.010,.030,.098),materials['Blue'])
        _cube(stage,cradle+f'/Support_{side+1}',(.83+side*.022,.53,1.006),(.014,.030,.004),materials['Blue'])
    metadata={'schema':SCHEMA,'task_id':'pipette_transfer','root_path':root_path,'presentation':presentation,
        'actors':actors,'objects':objects,'fixtures':{'pipette_cradle':cradle},
        'fixture_metadata':{'pipette_cradle':{'id':'pipette_cradle','body_path':cradle,'path':cradle,
            'mass_kg':0.,'static_fixture':True,'initial_position_m':[0.,0.,0.],
            'initial_pose_xyzw':[0.,0.,0.,0.,0.,0.,1.],
            'local_bounds_m':[[.78,.48,.900],[.88,.58,1.008]],
            'half_extents_m':[.05,.05,.054],
            'collider_paths':[cradle+'/'+name for name in ['Base','Post_0','Post_2','Support_0','Support_2']]}},
        'joints':{'plunger':joint_path},'particle_sets':{},'geometry_license':'CC0-1.0',
        'geometry_source':'Original analytic laboratory geometry; metres/Z-up',
        'task_parameters':{'hydraulics':{'schema':'laboratory_piston_finite_volume.v1',
            'source_id':'source','receiver_ids':['receiver_a','receiver_b'], 'tool_id':'pipette',
            'plunger_id':'plunger','joint_path':joint_path,'joint_anchor_z_m':.028,
            'piston_area_m2':PISTON_AREA_M2,'stroke_m':STROKE_M,'tip_local_m':[0.,0.,0.],
            'nozzle_radius_m':.003,'initial_q_m':0.,'containers':containers,'spill_plane_z_m':.90,
            'initial_total_ml':80.,'gravity_m_s2':9.81},
            'tool_return':{'support_fixture':'pipette_cradle','initial_pose_xyzw':[*tool_position,0.,0.,0.,1.],
                'position_tolerance_m':.012,'angular_tolerance_rad':.20},
            'target_per_receiver_ml':10.,'target_tolerance_ml':2.,'spill_budget_ml':2.},
        'scope':'Real dynamic sliding-piston geometry; fluid uses explicit reduced finite-volume dynamics, not PBD.',
        'limitations':['No resolved pressure, viscosity, surface tension or fluid weight reaction.',
            'Liquid surfaces are render-only cylinders driven by conserved volume; no resolved animated stream or splash.',
            'Vessel fluid uses horizontal levels and a conservative tilted-rim capacity approximation.'],
        'native_qualified':False,'robot_manipulation_qualified':False}
    metadata['volume_visual_bindings']={'schema':'cylinder-volume-bindings-v1','bindings':[
        {'id':'liquid_'+name,'target_prim_path':container['initial_surface_path'],
         'state_path':['detail','water_balance',*(['source_ml'] if name=='source' else ['receivers_ml',name])],
         'cross_section_area_m2':math.pi*container['radius_m']**2,
         'floor_z_m':container['floor_z_m'], 'maximum_height_m':container['rim_z_m']-container['floor_z_m']}
        for name,container in containers.items()]+[
        {'id':'liquid_tool','target_prim_path':barrel+'/ChamberLiquid',
         'state_path':['detail','water_balance','tool_ml'],'cross_section_area_m2':PISTON_AREA_M2,
         'floor_z_m':.0265,'maximum_height_m':STROKE_M}]}
    return add_task_visual_details(stage, metadata)
