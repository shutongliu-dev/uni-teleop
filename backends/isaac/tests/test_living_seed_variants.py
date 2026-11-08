"""CPU authoring checks; these do not qualify native stability or manipulation."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.living_room import seed_geometry as geo
from scenes.workrooms.living_room import seed_variants as author

ASSETS = Path(f'{DATA_ROOT}/assets/task_parts/living_room')
VERSIONS = {'remote_operation': 'v6', 'storage_box': 'v2', 'water_pouring': 'v2'}
USD_HASHES = {'remote_operation': '50680f3bd2d0624fb820e70868a4c46072ee57f591c1c3b094a6b65cc14ccd15',
              'storage_box': '97b2df8adecae6c910ae192a756e6ffa7cbda32988353511644aed07e47f8823',
              'water_pouring': 'e7703b2565efe0063127dd986160f1657cac55524c046f2cea1a185f89d00371'}


def parent(task):
    root = ASSETS/task/VERSIONS[task]
    if not root.is_dir():
        pytest.skip('Shared original living-task source assets are unavailable')
    assert author.file_hash(root/'initial.usda') == USD_HASHES[task]
    return Usd.Stage.Open(str(root/'initial.usda')), load_json(root/'metadata.json')


def plan(task, metadata, seed):
    """Explicit author-unit fixture, not the independent production sampler."""
    actor_paths = author.actors(metadata, task)
    groups = {}
    if task == 'remote_operation':
        specifications = [('remote', metadata['remote']['position_m'], ['remote', 'power', 'confirm'],
                           [metadata['remote']['assembly_path']]),
                          ('holder', metadata['holder']['position_m'], ['holder'], [actor_paths['holder']])]
    elif task == 'storage_box':
        specifications = [('box_assembly', metadata['initial_positions_m']['box'], list(actor_paths), list(actor_paths.values())),
                          ('tray', [.94, -.12, .786], [], [author.ROOT+'/CollectionTray'])]
    else:
        specifications = [('kettle', metadata['containers']['kettle']['initial_position_m'], ['kettle'],
                           [actor_paths['kettle'], metadata['water']['particles_path']]),
                          ('cup', metadata['containers']['cup']['initial_position_m'], ['cup'], [actor_paths['cup']]),
                          ('tray', [.65, 0, .786], [], [metadata['tray']['path']])]
    for name, pivot, ids, paths in specifications:
        groups[name] = {'pivot_m': pivot, 'translation_xy_m': [.003, -.004] if seed else [0., 0.],
                        'yaw_rad': .03 if seed and name != 'tray' else 0., 'actor_ids': ids, 'prim_paths': paths}
    if task == 'water_pouring':
        groups['tray']['translation_xy_m'] = [0., 0.]
    result = {'schema': 'living_room_seed_case_plan.v1', 'generator_version': 'unit_fixture',
              'task_id': task, 'seed': seed,
              'geometry_variant': ('canonical', 'narrow', 'wide')[seed],
              'appearance_variant': ('canonical', 'warm', 'cool')[seed], 'group_transforms': groups}
    result['plan_sha256'] = author.canonical_hash(result)
    return result


@pytest.mark.parametrize('task', VERSIONS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_real_asset_derivation_preserves_parent_and_physical_parameters(task, seed):
    source, original = parent(task)
    source_text = source.GetRootLayer().ExportToString()
    original_copy = deepcopy(original)
    stage, metadata, poses, visual, identity = author.derive_stage(source, original, plan(task, original, seed))
    assert source.GetRootLayer().ExportToString() == source_text
    assert original == original_copy
    assert author.file_hash(ASSETS/task/VERSIONS[task]/'initial.usda') == USD_HASHES[task]
    assert len(poses) == len(author.actors(original, task))
    assert all(abs(sum(v*v for v in row['quaternion_wxyz'])-1) < 1.e-12 for row in poses.values())
    allowed = {'physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes',
               'physics:localPos0', 'physics:localPos1'}
    before, after = author.physics_inventory(source), author.physics_inventory(stage)
    for path, row in before.items():
        for attr, value in row.items():
            if attr.startswith(('physics:', 'physx')) and attr not in allowed:
                assert after[path][attr] == value, (path, attr)
    if not seed:
        assert before == after
        assert not identity['changed_prim_paths']
    else:
        assert identity['changed_prim_paths']
        assert visual['render_only'] and visual['dependencies_sha256']
    json.dumps(metadata, allow_nan=False)
    json.dumps(identity, allow_nan=False)


@pytest.mark.parametrize('task,scale', [('remote_operation', .94), ('storage_box', .95)])
def test_actual_collider_bounds_mass_and_joint_anchors_follow_affine_map(task, scale):
    original, data = parent(task)
    stage, metadata, _, _, _ = author.derive_stage(original, data, plan(task, data, 1))
    inventory = author.actors(data, task)
    for name, path in inventory.items():
        expected = np.array(geo.collision_bounds(original, path))*[scale, 1, 1]
        np.testing.assert_allclose(geo.collision_bounds(stage, path), expected, atol=2.e-8)
        old = UsdPhysics.MassAPI(original.GetPrimAtPath(path))
        new = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        assert float(new.GetMassAttr().Get()) == pytest.approx(float(old.GetMassAttr().Get())*scale)
        com = np.array(old.GetCenterOfMassAttr().Get())
        r = geo.rotation(old.GetPrincipalAxesAttr().Get())
        tensor = r@np.diag(old.GetDiagonalInertiaAttr().Get())@r.T
        _, expected_com, expected_i = geo.affine_mass(float(old.GetMassAttr().Get()), com, tensor, [scale, 1, 1])
        new_r = geo.rotation(new.GetPrincipalAxesAttr().Get())
        np.testing.assert_allclose(new.GetCenterOfMassAttr().Get(), expected_com, atol=1.e-8)
        np.testing.assert_allclose(new_r@np.diag(new.GetDiagonalInertiaAttr().Get())@new_r.T, expected_i, rtol=1.e-6, atol=1.e-10)
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            ends = []
            for rel, attr in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr()), (joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
                body = stage.GetPrimAtPath(rel.GetTargets()[0])
                ends.append(np.array(cache.GetLocalToWorldTransform(body).Transform(Gf.Vec3d(*attr.Get()))))
            np.testing.assert_allclose(ends[0], ends[1], atol=2.e-8)
    if task == 'remote_operation':
        np.testing.assert_allclose(np.diff(metadata['remote']['local_bounds_m'], axis=0)[0], metadata['remote']['envelope_xyz_m'])


@pytest.mark.parametrize('seed,factor', [(1, .92), (2, 1.08)])
def test_water_changes_only_handle_geometry_and_moves_actual_inventory_rigidly(seed, factor):
    original, data = parent('water_pouring')
    case = plan('water_pouring', data, seed)
    stage, metadata, _, _, _ = author.derive_stage(original, data, case)
    for name in ('Kettle', 'Cup'):
        path = author.ROOT+'/'+name+'/HollowShell'
        for attr in ('points', 'faceVertexCounts', 'faceVertexIndices'):
            assert stage.GetPrimAtPath(path).GetAttribute(attr).Get() == original.GetPrimAtPath(path).GetAttribute(attr).Get()
        assert metadata['containers'][name.lower()]['inner_profile_z_radius_spout_m'] == data['containers'][name.lower()]['inner_profile_z_radius_spout_m']
    particle_path = data['water']['particles_path']
    before = np.asarray(UsdGeom.Points(original.GetPrimAtPath(particle_path)).GetPointsAttr().Get())
    after = np.asarray(UsdGeom.Points(stage.GetPrimAtPath(particle_path)).GetPointsAttr().Get())
    pivot = np.array(case['group_transforms']['kettle']['pivot_m'])
    angle = case['group_transforms']['kettle']['yaw_rad']
    r = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(after, (before-pivot)@r.T+pivot+[.003, -.004, 0], atol=6.e-8)
    assert len(after) == data['water']['particle_count']
    assert metadata['water'] == data['water']
    assert metadata['handle_geometry']['capsule_radius_m'] == pytest.approx(.0065*factor)
    for index in range(5):
        path = author.ROOT+f'/Kettle/Handle_{index:02d}'
        assert UsdGeom.Capsule(stage.GetPrimAtPath(path)).GetRadiusAttr().Get() == pytest.approx(.0065*factor)
    assert metadata['mass_properties']['kettle']['inertia_mode'] == 'explicit_prior'
    assert metadata['mass_properties']['cup']['inertia_mode'] == 'native_geometry_computation_not_read_back'


def test_affine_mass_matches_scaled_box_and_rejects_two_negative_axes():
    size, mass = np.array([.1, .2, .3]), .7
    inertia = np.diag([mass*(size[(i+1)%3]**2+size[(i+2)%3]**2)/12 for i in range(3)])
    scale = np.array([.9, 1.1, 1.2])
    actual_mass, com, actual_i = geo.affine_mass(mass, np.zeros(3), inertia, scale)
    size *= scale
    expected = np.diag([actual_mass*(size[(i+1)%3]**2+size[(i+2)%3]**2)/12 for i in range(3)])
    np.testing.assert_allclose(actual_i, expected)
    with pytest.raises(ValueError):
        geo.affine_mass(mass, com, inertia, [-1, -1, 1])


def test_capsule_moments_reduce_to_sphere():
    radius = .02
    mass, com, inertia = geo.capsule_volume_moments(radius, 0)
    assert mass == pytest.approx(4*math.pi*radius**3/3)
    np.testing.assert_allclose(inertia, np.eye(3)*.4*mass*radius**2)


def test_mesh_volume_moments_match_offset_tetrahedron():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/Mesh')
    offset = np.array([.2, .3, .4])
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])+offset
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*4)
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    volume, com, inertia = geo.mesh_volume_moments(mesh)
    assert volume == pytest.approx(1/6)
    np.testing.assert_allclose(com, offset+.25, atol=2.e-8)
    np.testing.assert_allclose(inertia, (np.eye(3)*.075+(np.ones((3, 3))-np.eye(3))*.0125)/6, atol=2.e-9)


def test_mesh_with_original_scale_is_not_scaled_twice():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/Body')
    mesh = UsdGeom.Mesh.Define(stage, '/Body/Mesh')
    mesh.CreatePointsAttr([(-1, -1, -1), (1, 1, 1)])
    mesh.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    geo.scale_tree(stage, '/Body', [.5, 1, 1])
    assert list(mesh.GetPrim().GetAttribute('xformOp:scale').Get()) == [2, 3, 4]
    np.testing.assert_allclose(mesh.GetPointsAttr().Get(), [[-.5, -1, -1], [.5, 1, 1]])


def test_rejects_tower_and_noncanonical_seed_zero():
    source, data = parent('remote_operation')
    case = plan('remote_operation', data, 0)
    case['task_id'] = 'tower_building'
    with pytest.raises(ValueError, match='does not randomize towers'):
        author.derive_stage(source, data, case)
    case = plan('remote_operation', data, 1)
    case['seed'] = 0
    case['plan_sha256'] = author.canonical_hash({k: v for k, v in case.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError, match='Seed zero'):
        author.derive_stage(source, data, case)


@pytest.mark.parametrize('task', VERSIONS)
def test_appearance_alone_preserves_all_physical_inputs(task):
    stage, _ = parent(task)
    stage = Usd.Stage.Open(stage.Flatten())
    before = author.physics_inventory(stage)
    result = author.apply_appearance(stage, task, 'neutral')
    assert result['dependencies_sha256']
    assert before == author.physics_inventory(stage)
