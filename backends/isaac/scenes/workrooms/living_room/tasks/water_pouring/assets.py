"""Candidate cold-water vessels and real PhysX PBD fluid, in SI units.

Pure USD authoring also works with CPU OpenUSD; registered PhysX schemas are
required to simulate. Geometry/parameters are candidates, not a frozen release.
No robot, environment, light, camera or PhysicsScene is authored here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


SCHEMA = 'living_water_pouring_candidate_assets.v1'
MDL_GLASS = 'OmniGlass.mdl'  # Isaac Sim core MDL, resolved by the local installation
TABLE_TOP_M = .78
PARTICLE_SPACING_M = .003
WATER_MASS_KG = .25
WATER_DENSITY_KG_M3 = 1000.
SEGMENTS = 128
# (height, circular radius, extra radius at the spout).
KETTLE_OUTER = [(0., 0., 0.), (0., .043, 0.), (.003, .049, 0.),
                (.012, .051, 0.), (.075, .052, 0.), (.096, .045, .020), (.102, .043, .030)]
KETTLE_INNER = [(.006, 0., 0.), (.006, .045, 0.), (.012, .048, 0.),
                (.075, .049, 0.), (.096, .042, .020), (.102, .040, .030)]
CUP_OUTER = [(0., 0., 0.), (0., .031, 0.), (.003, .035, 0.), (.065, .036, 0.), (.067, .0358, 0.)]
CUP_INNER = [(.005, 0., 0.), (.005, .032, 0.), (.065, .033, 0.), (.067, .033, 0.)]


def _attr(prim, name, kind, value):
    return prim.CreateAttribute(name, kind, custom=False).Set(value)


def _schema(prim, name):
    schemas = prim.GetMetadata('apiSchemas')
    names = list(schemas.GetAppliedItems()) if schemas else []
    if name not in names:
        names.append(name)
    prim.SetMetadata('apiSchemas', Sdf.TokenListOp.Create(prependedItems=names))


def _material(stage, path, color, roughness, *, metallic=0., ior=None):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    if ior:
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.24)
        shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(ior)
        glass = UsdShade.Shader.Define(stage, path + '/GlassMDL')
        glass.SetSourceAsset(MDL_GLASS, 'mdl')
        glass.SetSourceAssetSubIdentifier('OmniGlass', 'mdl')
        for name, kind, value in [('glass_color', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
            ('glass_ior', Sdf.ValueTypeNames.Float, ior), ('thin_walled', Sdf.ValueTypeNames.Bool, False),
            ('frosting_roughness', Sdf.ValueTypeNames.Float, roughness), ('depth', Sdf.ValueTypeNames.Float, .04)]:
            glass.CreateInput(name, kind).Set(value)
        glass.CreateOutput('out', Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput('mdl').ConnectToSource(glass.ConnectableAPI(), 'out')
    return mat


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def vessel_mesh(outer, inner, *, segments=SEGMENTS):
    """Watertight solid glass shell including floor, open cavity and pouring lip.

    The cavity is an actual surface, not a transparent solid convex hull. The
    spout is the smoothly extended, slightly lowered front section of the rim.
    """
    rings = outer + list(reversed(inner))
    points, index_rings = [], []
    for z, radius, spout in rings:
        if radius == 0:
            index_rings.append([len(points)])
            points.append([0., 0., z])
        else:
            ids = []
            for i in range(segments):
                theta = 2 * math.pi * i / segments
                weight = max(0., math.cos(theta)) ** 12
                r = radius + spout * weight
                ids.append(len(points))
                points.append([r * math.cos(theta), r * math.sin(theta), z - .2 * spout * weight])
            index_rings.append(ids)
    faces = []
    for a, b in zip(index_rings, index_rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                faces.append([a[0], b[j], b[i]])
            elif len(b) == 1:
                faces.append([a[i], a[j], b[0]])
            else:
                faces.extend(([a[i], a[j], b[j]], [a[i], b[j], b[i]]))
    # The two pole vertices bound the solid bottom and need no connecting face.
    return np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int32)


def _mesh(stage, path, points, faces, material):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr([Gf.Vec3f(*np.min(points, axis=0)), Gf.Vec3f(*np.max(points, axis=0))])
    _bind(mesh.GetPrim(), material)
    return mesh


def _collider(prim, *, sdf=False):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf' if sdf else 'convexHull')
    _schema(prim, 'PhysxCollisionAPI')
    _attr(prim, 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, .0005)
    _attr(prim, 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, 0.)
    if sdf:
        _schema(prim, 'PhysxSDFMeshCollisionAPI')
        _attr(prim, 'physxSDFMeshCollision:sdfResolution', Sdf.ValueTypeNames.Int, 384)
        _attr(prim, 'physxSDFMeshCollision:sdfSubgridResolution', Sdf.ValueTypeNames.Int, 6)
        _attr(prim, 'physxSDFMeshCollision:sdfMargin', Sdf.ValueTypeNames.Float, .01)
        _attr(prim, 'physxSDFMeshCollision:sdfNarrowBandThickness', Sdf.ValueTypeNames.Float, .02)


def _body(stage, path, position, yaw, mass):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    body.AddRotateZOp().Set(math.degrees(yaw))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI(body.GetPrim()).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    _schema(body.GetPrim(), 'PhysxRigidBodyAPI')
    _attr(body.GetPrim(), 'physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, 16)
    _attr(body.GetPrim(), 'physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 4)
    _attr(body.GetPrim(), 'physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool, True)
    return body


def _capsule(stage, path, a, b, radius, material, *, collide=True):
    a, b = np.asarray(a), np.asarray(b)
    delta = b-a
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(float(np.linalg.norm(delta)))
    capsule.CreateAxisAttr('Z')
    capsule.AddTranslateOp().Set(Gf.Vec3d(*((a+b)/2)))
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta)).GetQuat()
    capsule.AddOrientOp().Set(Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary())))
    _bind(capsule.GetPrim(), material)
    if collide:
        _collider(capsule.GetPrim())
    return capsule


def _cube(stage, path, size, center, material, *, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material)
    if collide:
        _collider(cube.GetPrim())
    return cube


def initial_particles(position, yaw, *, water_mass_kg=WATER_MASS_KG,
                      particle_spacing_m=PARTICLE_SPACING_M):
    """Fixed inventory for the requested declared physical water mass."""
    if not isinstance(water_mass_kg, (float, int)) or isinstance(water_mass_kg, bool) or not math.isfinite(water_mass_kg) or water_mass_kg <= 0:
        raise ValueError('water_mass_kg must be positive and finite')
    spacing = particle_spacing_m
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or not .003 <= spacing <= .01:
        raise ValueError('Candidate particle spacing must be finite in [0.003,0.010] m')
    count = round(water_mass_kg / (WATER_DENSITY_KG_M3 * spacing**3))
    if not 1 <= count <= 20000:
        raise ValueError('Requested initial water inventory is outside this candidate capacity')
    # Preserve the v1 layout at 3 mm; coarser diagnostic particles need their
    # own wall clearance, without changing vessel geometry or total mass.
    radius = .045 - spacing * (.0032/.003)
    extent = math.ceil(radius/spacing)
    layer = [(i*spacing, j*spacing) for i in range(-extent, extent+1) for j in range(-extent, extent+1)
             if math.hypot(i*spacing, j*spacing) <= radius]
    layer.sort(key=lambda xy: (xy[0]**2 + xy[1]**2, math.atan2(xy[1], xy[0])))
    points = []
    c, s = math.cos(yaw), math.sin(yaw)
    for index in range(count):
        k, j = divmod(index, len(layer)); x, y = layer[j]
        z = .006 + spacing + k * spacing
        points.append([position[0]+c*x-s*y, position[1]+s*x+c*y, position[2]+z])
    local = np.array([[layer[index % len(layer)][0], layer[index % len(layer)][1],
                       .006 + spacing + (index//len(layer))*spacing] for index in range(count)])
    if not cavity_contains(local, KETTLE_INNER).all():
        raise ValueError('Requested initial water does not fit the physical kettle cavity')
    return np.asarray(points, dtype=np.float32)


def author_task(stage, root_path='/World/LivingTask', *, presentation=False, water_mass_kg=WATER_MASS_KG,
                particle_spacing_m=PARTICLE_SPACING_M):
    """Author candidate dynamics and real liquid. Presentation never fakes water."""
    if type(presentation) is not bool:
        raise ValueError('presentation must be bool')
    # Validate the physical inventory before mutating the caller's stage.
    initial_particles([0., 0., 0.], 0., water_mass_kg=water_mass_kg,
                      particle_spacing_m=particle_spacing_m)
    root = Sdf.Path(root_path)
    if not root.IsAbsolutePath() or not root.IsPrimPath() or root_path in ('/', '/World'):
        raise ValueError('Use a dedicated absolute task prim path')
    if stage.GetPrimAtPath(root):
        raise ValueError('Task root already exists')
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1.e-12:
        raise ValueError('This candidate requires a metre-scale stage')
    UsdGeom.Xform.Define(stage, root_path)
    materials = root_path + '/Materials'
    warm = _material(stage, materials+'/WarmWhite', (.84, .81, .74), .27)
    metal = _material(stage, materials+'/BrushedSteel', (.52, .55, .56), .23, metallic=.9)
    glass = _material(stage, materials+'/Borosilicate', (.97, .99, 1.), .028, ior=1.47)
    water = _material(stage, materials+'/Water', (.90, .975, .995), .018, ior=1.333)
    dark = _material(stage, materials+'/Feet', (.085, .08, .07), .68)
    tray = root_path + '/CatchTray'
    UsdGeom.Xform.Define(stage, tray)
    _cube(stage, tray+'/Floor', [.53, .35, .006], [.65, 0., TABLE_TOP_M+.003], warm)
    for name, size, position in [
        ('Front', [.53, .007, .017], [.65, -.1715, TABLE_TOP_M+.0145]),
        ('Rear', [.53, .007, .017], [.65, .1715, TABLE_TOP_M+.0145]),
        ('Left', [.007, .336, .017], [.3885, 0., TABLE_TOP_M+.0145]),
        ('Right', [.007, .336, .017], [.9115, 0., TABLE_TOP_M+.0145])]:
        _cube(stage, tray+'/'+name, size, position, warm)
    kettle_path, cup_path = root_path+'/Kettle', root_path+'/Cup'
    kettle_position, cup_position = [.52, -.055, .7861], [.80, .065, .7861]
    yaw = math.atan2(cup_position[1]-kettle_position[1], cup_position[0]-kettle_position[0])
    _body(stage, kettle_path, kettle_position, yaw, .32)
    _body(stage, cup_path, cup_position, 0., .105)
    for path, outer, inner in [(kettle_path, KETTLE_OUTER, KETTLE_INNER), (cup_path, CUP_OUTER, CUP_INNER)]:
        points, faces = vessel_mesh(outer, inner)
        shell = _mesh(stage, path+'/HollowShell', points, faces, glass)
        _collider(shell.GetPrim(), sdf=True)
    # An open 40 mm handle aperture; only the grip/attachment solids collide.
    grip = [(-.045, 0., .082), (-.087, 0., .085), (-.098, 0., .073),
            (-.098, 0., .035), (-.085, 0., .022), (-.048, 0., .025)]
    for index, (a, b) in enumerate(zip(grip, grip[1:])):
        _capsule(stage, kettle_path+f'/Handle_{index:02d}', a, b, .0065, warm)
    # Thin visual stainless foot collar; the true glass floor supports the body.
    collar_outer = [(.003, .050, 0.), (.008, .051, 0.)]
    collar_inner = [(.003, .049, 0.), (.008, .050, 0.)]
    p, f = vessel_mesh(collar_outer, collar_inner)
    _mesh(stage, kettle_path+'/SteelFootCollar', p, f, metal)
    # A dark inset on the outer handle makes its grip contour legible.
    _capsule(stage, kettle_path+'/GripInset', [-.104, 0., .039], [-.104, 0., .069], .0012, dark, collide=False)
    # Keep graduation marks outside the cavity and out of the collision model.
    for index, z in enumerate((.025, .045, .065)):
        _capsule(stage, cup_path+f'/EtchedMark_{index}', [.019, -.0298, z], [.026, -.0245, z], .00035, warm, collide=False)
    system_path, particles_path = root_path+'/WaterSystem', root_path+'/WaterParticles'
    system = stage.DefinePrim(system_path, 'PhysxParticleSystem')
    system.CreateRelationship('simulationOwner').SetTargets(['/World/PhysicsScene'])
    fluid_rest = particle_spacing_m / 2
    particle_contact = fluid_rest / (.99 * .6)
    for name, kind, value in [
        ('particleSystemEnabled', Sdf.ValueTypeNames.Bool, True),
        ('particleContactOffset', Sdf.ValueTypeNames.Float, particle_contact),
        ('fluidRestOffset', Sdf.ValueTypeNames.Float, fluid_rest),
        ('solidRestOffset', Sdf.ValueTypeNames.Float, .99*particle_contact),
        ('contactOffset', Sdf.ValueTypeNames.Float, particle_spacing_m),
        ('restOffset', Sdf.ValueTypeNames.Float, particle_spacing_m * (5/6)),
        ('enableCCD', Sdf.ValueTypeNames.Bool, True),
        ('solverPositionIterationCount', Sdf.ValueTypeNames.Int, 16),
        ('maxNeighborhood', Sdf.ValueTypeNames.Int, 96),
        ('maxVelocity', Sdf.ValueTypeNames.Float, 8.),
        ('nonParticleCollisionEnabled', Sdf.ValueTypeNames.Bool, True)]:
        _attr(system, name, kind, value)
    _schema(system, 'PhysxParticleSmoothingAPI')
    # Keep physical Points.points unsmoothed. This runtime's combined isosurface
    # path does not necessarily author simulationPoints; its isosurface mesh
    # still has independent smoothing and must never serve as the mass ledger.
    _attr(system, 'physxParticleSmoothing:particleSmoothingEnabled', Sdf.ValueTypeNames.Bool, False)
    _attr(system, 'physxParticleSmoothing:strength', Sdf.ValueTypeNames.Float, .7)
    _schema(system, 'PhysxParticleIsosurfaceAPI')
    for name, kind, value in [('isosurfaceEnabled', Sdf.ValueTypeNames.Bool, True),
        ('gridSpacing', Sdf.ValueTypeNames.Float, fluid_rest), ('surfaceDistance', Sdf.ValueTypeNames.Float, particle_spacing_m * (2/3)),
        ('maxVertices', Sdf.ValueTypeNames.Int, 300000), ('maxTriangles', Sdf.ValueTypeNames.Int, 600000),
        ('numMeshSmoothingPasses', Sdf.ValueTypeNames.Int, 2),
        ('numMeshNormalSmoothingPasses', Sdf.ValueTypeNames.Int, 2)]:
        _attr(system, 'physxParticleIsosurface:'+name, kind, value)
    _bind(system, water)
    # The same material carries rendering and physical PBD properties.
    _schema(water.GetPrim(), 'PhysxPBDMaterialAPI')
    for name, value in {'density': 1000., 'friction': .1, 'damping': 0., 'cohesion': .01,
                        'viscosity': .0000017, 'surfaceTension': .0074, 'vorticityConfinement': 0.}.items():
        _attr(water.GetPrim(), 'physxPBDMaterial:'+name, Sdf.ValueTypeNames.Float, value)
    UsdShade.MaterialBindingAPI(system).Bind(water, materialPurpose='physics')
    points = initial_particles(kettle_position, yaw, water_mass_kg=water_mass_kg,
                               particle_spacing_m=particle_spacing_m)
    particles = UsdGeom.Points.Define(stage, particles_path)
    particles.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    particles.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(points)))
    particles.CreateWidthsAttr([particle_spacing_m] * len(points))
    particles.CreateIdsAttr(list(range(len(points))))
    _schema(particles.GetPrim(), 'PhysxParticleSetAPI')
    _schema(particles.GetPrim(), 'PhysxParticleAPI')
    _attr(particles.GetPrim(), 'physxParticle:fluid', Sdf.ValueTypeNames.Bool, True)
    _attr(particles.GetPrim(), 'physxParticle:selfCollision', Sdf.ValueTypeNames.Bool, True)
    _attr(particles.GetPrim(), 'physxParticle:particleGroup', Sdf.ValueTypeNames.Int, 0)
    particles.GetPrim().CreateRelationship('physxParticle:particleSystem').SetTargets([system_path])
    UsdPhysics.MassAPI.Apply(particles.GetPrim()).CreateMassAttr(water_mass_kg)
    UsdPhysics.MassAPI(particles.GetPrim()).CreateDensityAttr(0.)
    _bind(particles.GetPrim(), water)
    metadata = {'schema': SCHEMA, 'candidate': True, 'frozen_release': False, 'root_path': root_path,
        'presentation': presentation, 'geometry_source': 'Original analytic meshes generated by this module',
        'geometry_license': 'CC0-1.0', 'native_accepted': False, 'robot_task_success': False,
        'physics_scene_path': '/World/PhysicsScene', 'table_top_z_m': TABLE_TOP_M,
        'dependencies': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'containers': {'kettle': {'path': kettle_path, 'dry_mass_kg': .32, 'usable_capacity_ml': 500.,
            'inner_profile_z_radius_spout_m': KETTLE_INNER, 'initial_position_m': kettle_position,
            'initial_yaw_rad': yaw, 'spout_axis_local': '+X', 'base_is_dynamic': True},
            'cup': {'path': cup_path, 'dry_mass_kg': .105, 'usable_capacity_ml': 200.,
            'inner_profile_z_radius_spout_m': CUP_INNER, 'initial_position_m': cup_position,
            'initial_yaw_rad': 0., 'base_is_dynamic': True}},
        'tray': {'path': tray, 'static_fixture': True, 'bounds_xy_m': [[.385, -.175], [.915, .175]],
                 'interior_floor_z_m': .786, 'rim_top_z_m': .803, 'wall_thickness_m': .007,
                 'spill_is_not_exempt': True},
        'water': {'system_path': system_path, 'particles_path': particles_path,
            'particle_count': len(points), 'total_mass_kg': water_mass_kg,
            'particle_mass_kg': water_mass_kg/len(points), 'reference_density_kg_m3': WATER_DENSITY_KG_M3,
            'initial_volume_ml': water_mass_kg/WATER_DENSITY_KG_M3*1e6, 'particle_spacing_m': particle_spacing_m,
            'candidate_physics_dt_s': 1/960,
            'fluid_rest_offset_m': fluid_rest, 'particle_contact_offset_m': particle_contact,
            'collider_rest_offset_m': particle_spacing_m * (5/6), 'raw_position_attribute': 'points',
            'raw_position_contract': 'Unsmoothed Points with particle smoothing disabled; native tick synchronization and observed updates required',
            'inventory_policy': 'Fixed particle IDs; no emitter, deletion, scripted stream or fake surface',
            'material_scope': 'Installed water-preset coefficients; not experimentally calibrated rheology'},
        'scope': 'Candidate task authoring only; hollow SDF, readback, leakage and force transfer require separate native evidence'}
    stage.GetPrimAtPath(root_path).SetCustomDataByKey('candidateMetadata', json.dumps(metadata, sort_keys=True))
    return metadata


def cavity_contains(local_points, profile):
    """Analytic physical cavity classification; points on a wall stay uncredited."""
    points = np.asarray(local_points, dtype=float)
    theta = np.arctan2(points[:, 1], points[:, 0])
    weight = np.maximum(0., np.cos(theta)) ** 12
    rows = [row for row in profile if row[1] > 0]
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.zeros(len(points), dtype=bool)
    for (z0, r0, s0), (z1, r1, s1) in zip(rows, rows[1:]):
        lo, hi = z0-.2*s0*weight, z1-.2*s1*weight
        fraction = (points[:, 2]-lo) / (hi-lo)
        allowed = (r0+s0*weight)*(1-fraction) + (r1+s1*weight)*fraction
        valid |= ((points[:, 2] > lo) & (points[:, 2] <= hi) & (radius < allowed))
    return valid


def read_water_state(stage, metadata, *, container_pose_xyzw=None):
    """Read unsmoothed particle USD state under the explicit authoring contract.

    The caller owns native tick synchronization. This is a diagnostic geometry
    ledger, not the task's retained-volume or cumulative-spill evaluator. Before
    native stepping these are authored initial values; this function alone does
    not certify simulation progress. The native probe verifies clock advance,
    changing particle states separately. Dynamic callers can pass independently
    read native world poses, avoiding a write into the live USD stage.
    """
    water = metadata['water']
    prim = stage.GetPrimAtPath(water['particles_path'])
    system = stage.GetPrimAtPath(water['system_path'])
    smoothing = system.GetAttribute('physxParticleSmoothing:particleSmoothingEnabled')
    anisotropy = system.GetAttribute('physxParticleAnisotropy:particleAnisotropyEnabled')
    if water['raw_position_attribute'] != 'points' or not smoothing or smoothing.Get() is not False:
        raise RuntimeError('Unsmoothed physical Points contract is not satisfied; no render fallback')
    if anisotropy and anisotropy.Get():
        raise RuntimeError('Particle anisotropy must be disabled for this raw Points contract')
    attr = prim.GetAttribute(water['raw_position_attribute'])
    raw = attr.Get() if attr else None
    if raw is None:
        raise RuntimeError('Missing unsmoothed physical particle points')
    points = np.asarray(raw, dtype=np.float64)
    velocities = np.asarray(UsdGeom.Points(prim).GetVelocitiesAttr().Get(), dtype=np.float64)
    expected = water['particle_count']
    ids = list(UsdGeom.Points(prim).GetIdsAttr().Get() or [])
    if points.shape != (expected, 3) or velocities.shape != points.shape or ids != list(range(expected)):
        raise RuntimeError('Native water inventory/velocity/ID count changed')
    if not np.isfinite(points).all() or not np.isfinite(velocities).all():
        raise RuntimeError('Nonfinite native particle state')
    if abs(UsdPhysics.MassAPI(prim).GetMassAttr().Get() - water['total_mass_kg']) > 1.e-7:
        raise RuntimeError('Authored water mass changed')
    if container_pose_xyzw is not None:
        if set(container_pose_xyzw) != set(metadata['containers']):
            raise RuntimeError('Native container pose inventory mismatch')
        for pose in container_pose_xyzw.values():
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1.) > 1.e-5:
                raise RuntimeError('Invalid native world pose xyzw')
    cache = UsdGeom.XformCache()
    classes = {}
    occupied = np.zeros(expected, bool)
    for name, container in metadata['containers'].items():
        if container_pose_xyzw is None:
            inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(container['path'])).GetInverse()
        else:
            pose = np.asarray(container_pose_xyzw[name], dtype=float)
            transform = Gf.Matrix4d(1.)
            transform.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6])))
            transform.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
            inverse = transform.GetInverse()
        # Gf matrices use row vectors. Batch the exact same affine transform so
        # high-frequency raw-state diagnostics need not call Python per particle.
        matrix = np.asarray(inverse, dtype=np.float64)
        local = points @ matrix[:3, :3] + matrix[3, :3]
        inside = cavity_contains(local, container['inner_profile_z_radius_spout_m'])
        if np.any(inside & occupied):
            raise RuntimeError('Container cavity classifications overlap')
        classes[name] = inside
        occupied |= inside
    classes['outside_vessels'] = ~occupied
    counts = {name: int(np.count_nonzero(mask)) for name, mask in classes.items()}
    return {'particle_count': expected, 'particle_ids_match': True,
        'raw_position_attribute': water['raw_position_attribute'], 'positions_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
        'counts': counts, 'mass_kg': {name: count*water['particle_mass_kg'] for name, count in counts.items()},
        'total_mass_kg': expected*water['particle_mass_kg'], 'maximum_particle_speed_m_s': float(np.linalg.norm(velocities, axis=1).max()),
        'particle_bounds_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'native_progress_verified_here': False,
        'container_pose_source': 'native_world_pose_argument' if container_pose_xyzw is not None else 'USD',
        'scope': 'Unsmoothed particle USD + authored fixed mass ledger; caller must verify native progress; inverse-mass and cumulative-spill scoring not established'}
