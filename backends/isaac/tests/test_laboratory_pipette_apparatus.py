"""Readable Task5 apparatus retains genuine open geometry and passive actuation."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task


def authored():
    stage = Usd.Stage.CreateInMemory()
    return stage, author_task(stage)


def local_mesh(prim):
    if prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        mesh = trimesh.creation.cylinder(radius=cylinder.GetRadiusAttr().Get(),
            height=cylinder.GetHeightAttr().Get(), sections=128)
    else:
        geom = UsdGeom.Mesh(prim)
        mesh = trimesh.Trimesh(vertices=np.asarray(geom.GetPointsAttr().Get()),
            faces=np.asarray(geom.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=True)
    mesh.apply_transform(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T)
    return mesh


def ray_hits(mesh, origin, direction):
    """Independent small-mesh ray check without optional spatial-index modules."""
    hits = 0
    for triangle in mesh.triangles:
        a, b, c = triangle
        matrix = np.column_stack((b-a, c-a, -np.asarray(direction)))
        if abs(np.linalg.det(matrix)) < 1.e-14:
            continue
        u, v, t = np.linalg.solve(matrix, np.asarray(origin)-a)
        hits += bool(u >= -1.e-8 and v >= -1.e-8 and u+v <= 1.+1.e-8 and t > 0)
    return hits


def test_real_pull_ring_aperture_inertia_and_original_passive_stroke():
    stage, metadata = authored()
    row = next(o for o in metadata['objects'] if o['id'] == 'plunger')
    shapes = [local_mesh(stage.GetPrimAtPath(path)) for path in row['collider_paths']]
    loop = shapes[-1]
    assert loop.is_watertight and loop.volume > 0
    assert ray_hits(loop, (0., -.03, .098), (0., 1., 0.)) == 0
    assert ray_hits(loop, (.013, -.03, .098), (0., 1., 0.)) > 0
    compound = trimesh.util.concatenate(shapes)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(row['body_path']))
    assert mass.GetMassAttr().Get() == pytest.approx(.016)
    # Native USD inertia is the declared compound estimate. Integrate authored
    # meshes independently; polygonal circles differ slightly from analytic ones.
    compound.density = .016/compound.volume
    assert np.allclose(mass.GetCenterOfMassAttr().Get(), compound.center_mass, atol=2.e-5)
    assert np.allclose(mass.GetDiagonalInertiaAttr().Get(), np.diag(compound.moment_inertia), rtol=.005)
    assert np.all(compound.bounds >= np.asarray(row['local_bounds_m'])[0]-1.e-7)
    assert np.all(compound.bounds <= np.asarray(row['local_bounds_m'])[1]+1.e-7)
    for path in row['collider_paths']:
        assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(metadata['joints']['plunger']))
    assert joint.GetLowerLimitAttr().Get() == 0.
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(.055)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), 'linear')
    assert drive.GetStiffnessAttr().Get() == 0.
    assert drive.GetDampingAttr().Get() == 4.
    assert drive.GetTargetVelocityAttr().Get() == 0.


def test_nested_glass_and_water_have_actual_mdl_transmission_and_keep_physics():
    stage, metadata = authored()
    for name, ior in (('Glass', 1.47), ('Water', 1.333)):
        material = UsdShade.Material(stage.GetPrimAtPath(metadata['root_path']+'/Looks/'+name))
        shader, _, _ = material.GetSurfaceOutput('mdl').GetConnectedSource()
        shader = UsdShade.Shader(shader.GetPrim())
        # Isaac Sim core MDL is referenced by name and resolved by the local installation.
        assert shader.GetSourceAsset('mdl').path in BUILTIN_MDL
        assert shader.GetSourceAssetSubIdentifier('mdl') == 'OmniGlass'
        assert shader.GetInput('thin_walled').Get() is False
        assert shader.GetInput('glass_ior').Get() == pytest.approx(ior)
        assert shader.GetInput('frosting_roughness').Get() < .03
        assert material.GetSurfaceOutput().GetConnectedSource()
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(.55)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(.40)
    assert len(metadata['volume_visual_bindings']['bindings']) == 4
    config = metadata['packet_visual_bindings']
    root = stage.GetPrimAtPath(config['root_path'])
    assert not list(root.GetChildren()) and not UsdGeom.Xformable(root).GetOrderedXformOps()
    assert not root.HasAPI(UsdPhysics.CollisionAPI) and not root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath(config['material_path'])


def test_front_rear_graduations_expose_low_water_levels_without_mass_panels():
    stage, metadata = authored()
    labels = metadata['visual_detail']['labels']
    assert not any('DRY ' in row['text'] or ' g' in row['text'] for row in labels)
    for body, volumes in (('source', (5, 10, 20)), ('receiver_a', (8, 10, 12)),
                          ('receiver_b', (8, 10, 12))):
        for facing in ('front', 'rear'):
            by_name = {row['name']: row for row in labels if row['body_id'] == body and row['facing'] == facing}
            assert all(by_name[f'VolumeNumber_{facing}_{ml}']['text'] == str(ml) for ml in volumes)
            root = metadata['actors'][body]+'/VisualDetails/'
            assert not stage.GetPrimAtPath(root+'VesselIdentity_'+facing+'Paper')
    assert {row['text'] for row in labels if row['body_id'] == 'plunger'} == {'PULL', 'PUSH'}


def test_printed_observation_areas_are_not_collision_support_or_goal_regions():
    stage, metadata = authored()
    areas = metadata['visual_detail']['observation_areas']
    assert areas['collision'] is areas['support'] is areas['goal_region'] is False
    assert areas['emissive'] is False and len(areas['areas']) == 2
    for row in areas['areas']:
        prim = stage.GetPrimAtPath(row['prim_path'])
        assert not any(prim.HasAPI(api) for api in (
            UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI))
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        assert np.allclose(points[:, 2], .90005, rtol=0., atol=1.e-7)
    assert 'observation_areas' not in metadata['task_parameters']
    assert set(metadata['fixtures']) == {'pipette_cradle'}


def test_ordinary_close_camera_covers_source_and_full_real_piston_stroke():
    from scenes.workrooms.laboratory.pipette_presentation import pipette_camera_overrides
    _, metadata = authored()
    camera = pipette_camera_overrides()
    eye, target = np.asarray(camera['eye']), np.asarray(camera['target'])
    forward = (target-eye)/np.linalg.norm(target-eye)
    right = np.cross(forward, (0., 0., 1.)); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rows = [row for row in metadata['objects'] if row['id'] in ('source', 'pipette', 'plunger')]
    for row in rows:
        lo, hi = np.asarray(row['local_bounds_m'])
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]+(.055 if row['id'] == 'plunger' else 0.)):
                    vector = np.array([x, y, z])+row['initial_position_m']-eye
                    depth = np.dot(vector, forward)
                    assert depth > 0
                    film = camera['focal_length_mm']*np.array([np.dot(vector, right), np.dot(vector, up)])/depth
                    assert abs(film[0]) < 18.*.9 and abs(film[1]) < 13.5*.9
