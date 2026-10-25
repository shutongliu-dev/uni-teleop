"""Original coarse-granule weighing assets; passive contact, nominal mass ledger.

This intentionally uses harmless 8 mm model granules, not powder or chemistry.
The platform does not simulate a calibrated force balance. Only a released,
settled cup may be read by the task's mass ledger. No hand/controller is authored.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
from scenes.workrooms.laboratory.visual_details import add_task_visual_details

TASK_ID = "solid_weighing"
PELLET_COUNT = 24
PELLET_RADIUS_M = .004
PELLET_MASS_KG = .0005
SOURCE_POSITION = (.61, .51, .90)
CUP_POSITION = (.82, .54, .9005)
SCOOP_POSITION = (.99, .585, .9065)
PLATFORM_POSITION = (.71, 2.075, .90)
SCOOP_RETURN_BOUNDS = [[.85, .55, .90], [1.04, .62, .96]]


def _attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind).Set(value)


def _schema(prim, name):
    schemas = list(prim.GetAppliedSchemas())
    if name not in schemas:
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas+[name]))


def _box(size, center):
    mesh = trimesh.creation.box(size)
    mesh.apply_translation(center)
    return mesh


def _mesh(stage, path, shape, color, *, collider=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(np.asarray(shape.vertices).tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(shape.faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateDoubleSidedAttr(False)
    if collider:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        _schema(mesh.GetPrim(), "PhysxCollisionAPI")
        _attr(mesh.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, .0002)
        _attr(mesh.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.)
    return mesh.GetPrim()


def _walls(width, depth, height, floor=.003, wall=.003, *, entry=False):
    parts = [("floor", _box((width, depth, floor), (0, 0, floor/2)))]
    for sign in (-1, 1):
        parts.append((f"side_{sign+1}", _box((width, wall, height-floor),
            (0, sign*(depth-wall)/2, (height+floor)/2))))
        if sign == 1 or not entry:
            parts.append((f"end_{sign+1}", _box((wall, depth-2*wall, height-floor),
                (sign*(width-wall)/2, 0, (height+floor)/2))))
    return parts


def scoop_parts():
    """An open +X shovel mouth, physical ramp, side walls and graspable handle."""
    parts = [("bowl_floor", _box((.055, .044, .002), (-.0025, 0, .001))),
        ("bowl_back", _box((.003, .044, .017), (-.0285, 0, .0105)))]
    parts += [(f"bowl_side_{s+1}", _box((.055, .003, .017),
        (-.0025, s*.0205, .0105))) for s in (-1, 1)]
    wedge = trimesh.convex.convex_hull(np.array([
        [.025, -.022, 0], [.039, -.022, 0],
        [.025, .022, 0], [.039, .022, 0],
        [.025, -.022, .002], [.039, -.022, .0005],
        [.025, .022, .002], [.039, .022, .0005]]))
    parts += [("entry_ramp", wedge),
              ("handle", _box((.110, .014, .006), (-.085, 0, .010)))]
    return parts


def author_task(stage, root_path="/World/LaboratoryTask", *, presentation=False):
    """Author free dynamic scoop/cup/granules and three task-owned fixtures."""
    UsdGeom.Xform.Define(stage, root_path)
    contact = UsdShade.Material.Define(stage, root_path+"/Materials/DryContact")
    physical = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    physical.CreateStaticFrictionAttr(.35)
    physical.CreateDynamicFrictionAttr(.30)
    physical.CreateRestitutionAttr(.01)
    meta = {"schema": "laboratory_solid_weighing_assets.v1", "task_id": TASK_ID,
        "root_path": root_path, "presentation": bool(presentation),
        "actors": {}, "fixtures": {}, "objects": [], "fixture_metadata": {},
        "joints": {}, "particle_sets": {}, "task_parameters": {},
        "source": {"kind": "original_procedural", "license": "CC0-1.0",
            "authoring_module": __name__, "external_assets": []},
        "physics_prior": {"static_friction": .35, "dynamic_friction": .30,
            "restitution": .01, "contact_offset_m": .0002,
            "mass_source": "Declared synthetic dry apparatus and granule masses.",
            "granule_density_kg_m3": PELLET_MASS_KG/(4/3*math.pi*PELLET_RADIUS_M**3),
            "weighing": "Nominal discrete mass ledger; no force-balance accuracy claim."},
        "physics_accepted": False}

    def body(name, position, parts, mass, color, **extra):
        path = root_path+("/Actors/" if mass else "/Fixtures/")+name
        obj = UsdGeom.Xform.Define(stage, path)
        obj.AddTranslateOp().Set(Gf.Vec3d(*position))
        obj.AddOrientOp().Set(Gf.Quatf(1., Gf.Vec3f(0.)))
        colliders = []
        for label, shape in parts:
            prim = _mesh(stage, path+"/Collision/"+label, shape, color)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(contact, materialPurpose="physics")
            colliders.append(str(prim.GetPath()))
        combined = trimesh.util.concatenate([shape for _, shape in parts])
        bounds = combined.bounds
        record = {"id": name, "body_path": path, "path": path, "mass_kg": mass,
            "initial_position_m": list(position), "initial_pose_xyzw": [*position, 0., 0., 0., 1.],
            "local_bounds_m": bounds.tolist(), "half_extents_m": ((bounds[1]-bounds[0])/2).tolist(),
            "collider_paths": colliders, **extra}
        if mass:
            UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim()).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.RigidBodyAPI(obj.GetPrim()).CreateKinematicEnabledAttr(False)
            combined.density = mass/combined.volume
            props = combined.mass_properties
            eigenvalues, axes = np.linalg.eigh(props.inertia)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            matrix = np.eye(4)
            matrix[:3, :3] = axes
            q = trimesh.transformations.quaternion_from_matrix(matrix)
            api = UsdPhysics.MassAPI.Apply(obj.GetPrim())
            api.CreateMassAttr(mass)
            api.CreateCenterOfMassAttr(Gf.Vec3f(*props.center_mass))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*eigenvalues))
            api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
            _schema(obj.GetPrim(), "PhysxRigidBodyAPI")
            _attr(obj.GetPrim(), "physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool, True)
            meta["actors"][name] = path
            meta["objects"].append(record)
        else:
            meta["fixtures"][name] = path
            meta["fixture_metadata"][name] = record
        return record

    source = body("source_tray", SOURCE_POSITION, _walls(.150, .110, .025, entry=True),
        0., (.70, .77, .79), interior_bounds_m=[[-.075, -.052, .003], [.072, .052, .025]],
        open_entry_axis_local=[-1., 0., 0.], floor_top_local_z_m=.003)
    platform = body("balance_platform", PLATFORM_POSITION,
        [("base", _box((.135, .125, .022), (0, 0, .011))),
         ("pan", _box((.100, .090, .003), (0, 0, .0235)))],
        0., (.16, .22, .25), support_top_world_z_m=.925,
        pan_bounds_xy_m=[[.66, 2.03], [.76, 2.12]],
        measurement="Nominal mass ledger after unloading and stable support.")
    cup_position = (.71, 2.075, .9255) if presentation else CUP_POSITION
    cup = body("weighing_cup", cup_position, _walls(.075, .065, .035, floor=.003, wall=.003),
        .028, (.90, .91, .84), role="open_weighing_boat",
        interior_bounds_m=[[-.0345, -.0295, .003], [.0345, .0295, .035]],
        floor_top_local_z_m=.003, rim_top_local_z_m=.035)
    body("scoop_rest", (.95, .585, .90),
        [("pad", _box((.16, .065, .006), (0, 0, .003)))], 0., (.20, .37, .43))
    scoop = body("scoop", SCOOP_POSITION, scoop_parts(), .026, (.55, .61, .64),
        role="open_granule_scoop", interior_bounds_m=[[-.027, -.019, .002], [.025, .019, .019]],
        mouth_axis_local=[1., 0., 0.], handle_bounds_m=[[-.140, -.007, .007], [-.030, .007, .013]])
    pellet_ids = []
    for index in range(PELLET_COUNT):
        name = f"pellet_{index:02}"
        group, rem = divmod(index, 12)
        col, row = divmod(rem, 3)
        position = [.590+.010*col, .51+(-.022 if group == 0 else .022)+.010*(row-1), .9075]
        if presentation and index < 12:
            position = [.695+.010*col, 2.075+.010*(row-1), .9325]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=PELLET_RADIUS_M)
        item = body(name, position, [("grain", sphere)], PELLET_MASS_KG,
            (.77, .55, .22) if group else (.91, .76, .43), role="model_granule",
            radius_m=PELLET_RADIUS_M, nominal_mass_kg=PELLET_MASS_KG)
        # Native analytic sphere avoids many-face hull contact noise.
        path = item["collider_paths"][0]
        stage.RemovePrim(path)
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(PELLET_RADIUS_M)
        shape.CreateDisplayColorAttr([Gf.Vec3f(.88, .69, .32)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(contact, materialPurpose="physics")
        _schema(shape.GetPrim(), "PhysxCollisionAPI")
        _attr(shape.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, .0002)
        _attr(shape.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.)
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(item["body_path"]))
        api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*([.4*PELLET_MASS_KG*PELLET_RADIUS_M**2]*3)))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1., Gf.Vec3f(0.)))
        pellet_ids.append(name)
    meta["task_parameters"] = {"pellet_ids": pellet_ids, "pellet_count": PELLET_COUNT,
        "pellet_radius_m": PELLET_RADIUS_M, "pellet_mass_kg": PELLET_MASS_KG,
        "initial_granule_mass_kg": PELLET_COUNT*PELLET_MASS_KG,
        "target_mass_kg": .006, "tolerance_mass_kg": .0006, "spill_mass_limit_kg": .001,
        "source_tray": source, "cup": cup, "scoop": scoop, "platform": platform,
        "weighing_cup_target_position_m": [.71, 2.075, .925],
        "scoop_return_bounds_m": SCOOP_RETURN_BOUNDS,
        "weighing_readout_policy": "Sum nominal mass of settled granules entirely in the cup; "
            "read only while the cup is upright, supported by the pan and externally unloaded.",
        "spill_policy": "Account all 24 IDs; outside source, cup and carried scoop is pending "
            "while airborne and spill after stable external support. Latch cumulative spilled IDs."}
    from scenes.workrooms.laboratory.weight_display import add_weighing_display
    add_task_visual_details(stage, meta)
    return add_weighing_display(stage, meta)
