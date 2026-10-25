"""Physical source geometry, independent inertia integrals and reset boundaries."""
from copy import deepcopy
import math

import numpy as np
import pytest
import trimesh

from bench.paths import load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly import assembly_geometry as geometry
from scenes.workrooms.mechanical_assembly.tasks.peg_insertion import assets as peg_assets
from scenes.workrooms.mechanical_assembly.tasks.peg_insertion import task as peg_task
from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly import assets as ring_assets
from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly import task as ring_task


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    root=tmp_path_factory.mktemp('mechanical_parts')
    peg=root/'peg';ring=root/'ring'
    peg_assets.prepare_assets(peg);ring_assets.prepare_assets(ring)
    return peg,ring


@pytest.mark.parametrize('profile',[
    [[0.,0.,.016],[.052,0.,.016]],
    [[0.,.023,.035],[.008,.016,.035],[.030,.016,.035]],
    [[0.,.019,.045],[.015,.019,.045],[.027,.030,.045]],
    [[0.,0.,.012],[.076,0.,.012],[.100,0.,.005]],
])
def test_exact_profile_integral_matches_independent_triangle_mesh(profile):
    display,pieces=geometry.revolved_meshes(profile)
    result=geometry.polygonal_mass_properties(profile)
    assert display.is_watertight
    assert all(p.is_convex and p.is_watertight for p in pieces)
    assert sum(p.volume for p in pieces)==pytest.approx(result['volume_m3'],rel=1e-11)
    assert display.volume*geometry.DENSITY_PRIOR_KG_M3==pytest.approx(result['mass_kg'],rel=1e-11)
    np.testing.assert_allclose(display.center_mass,result['center_of_mass_m'],atol=1e-14)
    np.testing.assert_allclose(display.moment_inertia*geometry.DENSITY_PRIOR_KG_M3,
                               np.diag(result['inertia_diag_kg_m2']),atol=1e-14)


def test_circle_limit_and_bore_clearance_are_analytic():
    radius=.016;height=.070;density=1200.
    got=geometry.polygonal_mass_properties([[0.,0.,radius],[height,0.,radius]],segments=128)
    analytic_mass=density*math.pi*radius**2*height
    assert abs(got['mass_kg']/analytic_mass-1)<.0005
    assert abs(got['inertia_diag_kg_m2'][2]/(.5*analytic_mass*radius**2)-1)<.001
    profile=[[0.,.023,.035],[.008,.016,.035],[.03,.016,.035]]
    _,cells=geometry.revolved_meshes(profile)
    from scipy.spatial import ConvexHull
    # At z=20 mm, intersect every hull with a radial line at its inner-face midpoint.
    for index,cell in enumerate(cells):
        equations=ConvexHull(cell.vertices).equations
        direction=np.array([math.cos((index+.5)*2*math.pi/64),math.sin((index+.5)*2*math.pi/64),0.])
        point=.015999*direction+np.array([0.,0.,.020])
        assert not np.all(equations[:,:3]@point+equations[:,3]<=1e-10)
        material=.016001*direction+np.array([0.,0.,.020])
        assert np.all(equations[:,:3]@material+equations[:,3]<=1e-10)


def test_solid_partition_preserves_boundary_and_conservative_gpu_topology():
    for profile in ([[0.,0.,.016],[.052,0.,.016]],
                    [[.008,0.,.012],[.084,0.,.012],[.108,0.,.005]]):
        display,cells=geometry.revolved_meshes(profile)
        assert len(cells)==4
        whole=display.convex_hull
        assert geometry.convex_topology(whole)['vertices']>64
        assert geometry.convex_topology(whole)['vertices_per_polygon']>32
        assert sum(cell.volume for cell in cells)==pytest.approx(whole.volume,rel=1e-12)
        cell_vertices=np.concatenate([cell.vertices for cell in cells])
        # All original outer vertices survive; added vertices are axial only.
        for vertex in whole.vertices:
            assert np.linalg.norm(cell_vertices-vertex,axis=1).min()<1e-12
        for cell in cells:
            topology=geometry.convex_topology(cell)
            assert all(topology[key]<=limit for key,limit in geometry.GPU_CONVEX_LIMITS.items())
        asset=geometry.make_asset('negative_whole_hull',dynamic=True)
        geometry.add_revolved(asset,'body',profile)
        geometry.finalize_mass(asset)
        asset.collision=[('whole_hull',whole,{'geometry':'convex_mesh'})]
        with pytest.raises(ValueError,match='GPU convex envelope'):
            geometry.audit_geometry(asset)


def test_axial_shaft_collision_partition_preserves_visual_and_integrals():
    profile=[[.008,0.,.012],[.084,0.,.012],[.108,0.,.005]]
    original,whole=geometry.revolved_meshes(profile)
    display,cells=geometry.revolved_meshes(profile,split_axial=True)
    np.testing.assert_array_equal(display.vertices,original.vertices)
    np.testing.assert_array_equal(display.faces,original.faces)
    assert len(whole)==4 and len(cells)==8
    assert all(geometry.convex_topology(cell)['vertices']==36 for cell in cells)
    assert sum(cell.volume for cell in cells)==pytest.approx(sum(cell.volume for cell in whole),rel=1e-12)
    for split in (False,True):
        asset=ring_assets.make_assets(split_shaft_axial=split)[3][1]
        geometry.finalize_mass(asset)
        if not split:before=deepcopy(asset.data)
        else:
            assert asset.data['volume_m3']==before['volume_m3']
            assert asset.data['inertia_diag_kg_m2']==before['inertia_diag_kg_m2']


def test_global_ring_hull_and_nonconvex_profile_are_rejected():
    part=ring_assets.make_assets()[0][1]
    geometry.finalize_mass(part)
    solid_hull=trimesh.util.concatenate([m for _,m,_ in part.visual]).convex_hull
    part.collision=[('incorrect_filled_bore',solid_hull,{'geometry':'convex_mesh'})]
    with pytest.raises(ValueError,match='volume'):
        geometry.audit_geometry(part)
    with pytest.raises(ValueError,match='nonconvex'):
        geometry.revolved_meshes([[0.,.02,.04],[.01,.03,.04],[.02,.02,.04]])
    with pytest.raises(ValueError):
        geometry.revolved_meshes([[0.,.02,.01],[.02,.02,.01]])


def test_serialized_assets_retain_geometry_and_exact_mass(libraries):
    for root,module in zip(libraries,[peg_assets,ring_assets]):
        manifest=load_json(root/'manifest.json')
        assert manifest['physics_accepted'] is False
        for asset_id in module.ASSET_IDS.values():
            asset=load_task_asset(asset_id,root,verify_hashes=True)
            assert asset['mass_kg']>0 and min(asset['inertia_diag_kg_m2'])>0
            assert asset['cpu_geometry_audit']['passed'] is True
            assert asset['cpu_geometry_audit']['cooked_gpu_eligibility_verified'] is False
            collision=[]
            for part in asset['collision_parts']:
                mesh=trimesh.load(part['mesh_path'],force='mesh',process=True)
                assert mesh.is_watertight and mesh.is_convex
                collision.append(mesh)
            assert sum(m.volume for m in collision)==pytest.approx(asset['volume_m3'],rel=1e-7)
            visual=trimesh.load(asset['mesh_path'],force='mesh',process=True)
            np.testing.assert_allclose(visual.center_mass,asset['center_of_mass_m'],atol=1e-9)
            np.testing.assert_allclose(visual.moment_inertia*asset['density_kg_m3'],
                np.diag(asset['inertia_diag_kg_m2']),rtol=1e-7,atol=1e-12)
        with pytest.raises(FileExistsError): module.prepare_assets(root)


def test_task_reset_is_separate_from_guided_probe_and_has_no_threaded_dependency(libraries):
    for root,module,count in zip(libraries,[peg_task,ring_task],[2,3]):
        original=module.build_task_inputs(asset_root=root)
        before=deepcopy(original)
        probe=module.build_probe_inputs(asset_root=root)
        assert original==before
        assert len(original['scene']['objects'])==count
        assert original['scene']['robot']['base_position_m']==[-.23,0.,.78]
        assert original['initial_state']==probe['initial_state']
        assert original['scene']['robot']==probe['scene']['robot']
        assert original['scene']['workroom']==probe['scene']['workroom']
        assert 'engineering_probe' not in original['task']
        assert probe['task']['engineering_probe']['task_success_credit'] is False
        assert probe['task']['engineering_probe']['expected_functional_success'] is True
        assert all(o['position_m']!=p['position_m'] for o,p in zip(original['scene']['objects'],probe['scene']['objects']))
        assert all('threaded' not in x['asset_root'] for x in original['scene']['objects']+original['scene']['fixtures'])
        for spec in original['scene']['objects']+original['scene']['fixtures'][1:]:
            assert spec['friction']==.4 and spec['contact_offset_m']==.0001 and spec['rest_offset_m']==0.
        assert original['task']['evaluation']['radial_clearance_m']>10*.0002
        assert original['task']['physics_accepted'] is False


def test_probe_negative_controls_preserve_intended_success_criteria(libraries):
    peg=peg_task.build_probe_inputs(asset_root=libraries[0],case='outside_socket')
    ring=ring_task.build_probe_inputs(asset_root=libraries[1],case='misordered_stack_drop')
    assert peg['task']['engineering_probe']['expected_functional_success'] is False
    assert peg['task']['engineering_probe']['lateral_offset_m']==.045
    assert ring['task']['engineering_probe']['expected_functional_success'] is False
    assert ring['task']['engineering_probe']['release_order']==['top_ring','wide_ring','spacer']
    assert ring['task']['evaluation']['part_order']==['spacer','wide_ring','top_ring']
    for module,root in zip([peg_task,ring_task],libraries):
        with pytest.raises(ValueError):module.build_probe_inputs(asset_root=root,lateral_offset_m=float('nan'))
        with pytest.raises(ValueError):module.build_probe_inputs(asset_root=root,case='not_a_case')


def test_shaft_v3_uses_native_verified_partition_and_explicit_cooking_coordinates(libraries):
    assert ring_assets.DEFAULT_ASSET_ROOT.name=='v3'
    fixture=load_task_asset(ring_assets.ASSET_IDS['shaft_fixture'],libraries[1],verify_hashes=True)
    assert len(fixture['collision_parts'])==9  # One shoulder plus eight shaft cells.
    result=ring_task.build_task_inputs(asset_root=libraries[1])
    for spec in result['scene']['objects']+result['scene']['fixtures']:
        if spec['id']=='robot_stand':
            assert 'collision_mesh_coordinate_scale' not in spec
        else:
            assert spec['collision_mesh_coordinate_scale']==1000.
