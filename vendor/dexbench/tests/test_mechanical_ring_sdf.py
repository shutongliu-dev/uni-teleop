from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import trimesh

from bench.paths import DATA_ROOT, load_json
from scripts.assets.prepare_mechanical_ring_sdf import prepare, prepare_assembly, DEFAULT_SOURCE, SOURCE_ID, SOURCE_IDS, ASSET_ID
from scripts.validation.mechanical_ring_sdf import surface_witnesses, convention_and_bore_witnesses
from scripts.validation.validate_mechanical_ring_sdf import (build_inputs, build_stack_inputs,
    candidate_sdf_subgrid_resolution, impulse_comparison, stack_impulse_comparison,
    ReadOnlySelection, solver_selection)


DENSE_CANDIDATE = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
                       'shaft_spacer_assembly/dense1024_v1')


def test_solver_input_alone_cannot_pass_readback_gate():
    assert not solver_selection({},'PGS')['passed']
    assert not solver_selection({'physics_solver':{'profile_selection':'PGS','usd_readback':'TGS'}},'PGS')['passed']
    assert solver_selection({'physics_solver':{'profile_selection':'PGS','usd_readback':'PGS'}},'PGS')['passed']
    assert solver_selection({'physics_solver':{'profile_selection':'TGS','usd_readback':'TGS'}},'TGS')['passed']
    assert not solver_selection({'physics_solver':{'profile_selection':'invalid','usd_readback':'invalid'}},'invalid')['passed']


def test_native_sdf_subgrid_is_bound_to_the_frozen_candidate():
    dense = candidate_sdf_subgrid_resolution(DENSE_CANDIDATE)
    assert dense == 0
    assert build_inputs(DENSE_CANDIDATE)['native_sdf_subgrid_resolution'] == dense


def test_surface_witnesses_cover_every_source_face_without_filling_hole():
    mesh = trimesh.load(DEFAULT_SOURCE/SOURCE_ID/'visual.obj',force='mesh',process=False)
    points,normals = surface_witnesses(mesh)
    assert len(points)==len(mesh.faces)*7==5376
    assert np.allclose(np.linalg.norm(normals,axis=1),1.)
    assert np.min(np.linalg.norm(points[:,:2],axis=1)) > .015
    assert len(np.unique(points,axis=0))==len(points)


def test_unclosed_surface_refused():
    mesh=trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)-1))
    with pytest.raises(ValueError):surface_witnesses(mesh)


@pytest.mark.parametrize('source_id',SOURCE_IDS)
def test_witnesses_follow_each_ring_height_and_hole(source_id):
    from scipy.spatial import ConvexHull
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    mesh=trimesh.load(DEFAULT_SOURCE/source_id/'visual.obj',force='mesh',process=False)
    witness,sign,bore=convention_and_bore_witnesses(mesh)
    asset=load_task_asset(source_id,DEFAULT_SOURCE,verify_hashes=True)
    points=np.concatenate([witness,bore])
    contained=np.zeros(len(points),dtype=bool)
    for part in asset['collision_parts']:
        vertices=trimesh.load(part['mesh_path'],force='mesh',process=False).vertices
        planes=ConvexHull(vertices).equations
        contained |= np.all(points@planes[:,:3].T+planes[:,3] <= 1e-10,axis=1)
    assert np.array_equal(contained[:len(witness)],sign<0)
    assert not contained[len(witness):].any()
    assert witness[0,2]==mesh.bounds[1,2]+.0005
    assert len(surface_witnesses(mesh)[0])==5376


def test_three_layer_weight_balance_cancels_internal_loads_without_rescaling():
    paths={'spacer':'/spacer','wide_ring':'/wide','top_ring':'/top'}
    supports={'spacer':'/fixture','wide_ring':'/spacer','top_ring':'/wide'}
    contacts=[{'actor0':paths[name],'actor1':supports[name],
        'normal_impulses_ns':[value],'normals':[[0,0,1]]}
        for name,value in zip(paths,[3.,2.,1.])]
    row={'raw_contacts':contacts,'native_contact_impulse_crosscheck':{
        name:{'net_tensor_ns':[0,0,1],'pair_matrix_sum_ns':[0,0,1]} for name in paths}}
    result=stack_impulse_comparison([row],{'objects':{name:{'path':path} for name,path in paths.items()}},
        {name:1/(9.81*.0005) for name in paths},.0005,
        {name:{'support_path_prefix':support} for name,support in supports.items()})
    assert result['quantitative_weight_balance_passed']
    assert all(v==1. for v in result['total']['vertical_weight_ratios'].values())
    assert all(v['weight_ratio']==1. for v in result['layers'].values())


def test_signed_impulse_comparison_keeps_opposing_normals_and_three_apis():
    # Positive and negative raw scalars both contribute. Net and pair tensor
    # disagreement remains visible; never divide raw by a collider count.
    row={'raw_contacts':[{'actor0':'/ring','actor1':'/rest','normal_impulses_ns':[5.,-1.],
          'normals':[[0.,0.,1.],[0.,0.,1.]]}],
        'native_contact_impulse_crosscheck':{'ring':{'net_tensor_ns':[0,0,1.],
            'pair_matrix_sum_ns':[0,0,2.]}}}
    r=impulse_comparison([deepcopy(row) for _ in range(1000)],
        {'objects':{'ring':{'path':'/ring'}}},{'ring':1./(9.81*.0005)},.0005)
    assert r['objects']['ring']['vertical_weight_ratios']=={
        'raw_signed_normal':4.,'native_net_tensor':1.,'native_pair_matrix':2.}
    assert not r['quantitative_weight_balance_passed']


def test_expectation_selection_cannot_mutate_live_spec():
    class Env:
        scene_spec={'objects':[{'id':'a'}]}
        scene_info={'objects':{'a':{'path':'/a'}}}
    env=Env(); selection=ReadOnlySelection(env)
    selection.scene_spec['objects'][0]['friction']=.4
    del selection.scene_info['objects']['a']
    assert 'friction' not in env.scene_spec['objects'][0]
    assert 'a' in env.scene_info['objects']


def test_cpu_authoring_and_input_retains_original_physics(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdPhysics
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    result=prepare(tmp_path/'candidate')
    asset=load_task_asset(ASSET_ID,result['asset_root'],verify_hashes=True)
    source=load_task_asset(SOURCE_ID,DEFAULT_SOURCE,verify_hashes=True)
    for k in ('mass_kg','center_of_mass_m','inertia_diag_kg_m2'):
        assert asset[k]==source[k]
    stage=Usd.Stage.Open(asset['usd_path']);prim=stage.GetPrimAtPath('/Ring/collision')
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='sdf'
    assert prim.GetAttribute('physxSDFMeshCollision:sdfResolution').Get()==1024
    assert prim.GetAttribute('physxSDFMeshCollision:sdfEnableRemeshing').Get() is False
    assert len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())==768
    inputs=build_inputs(result['asset_root'])
    assert inputs['scene']['isaacsim']['solver_type']=='PGS'
    objects=inputs['scene']['objects']
    assert len(objects)==2 and objects[0]['asset_id']==SOURCE_ID
    assert objects[1]['asset_id']==ASSET_ID and 'friction' not in objects[1]
    assert objects[0]['position_m'][2]==objects[1]['position_m'][2]==.789
    assert [r['asset_id'] for r in inputs['scene']['fixtures'][1:]]==['mechanical_l3_supply_rest']*2
    assert inputs['task']['task_success_claimed'] is False
    with pytest.raises(FileExistsError):prepare(tmp_path/'candidate')


def test_three_ring_authoring_preserves_sources_and_near_seat_case(tmp_path):
    pytest.importorskip('pxr')
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    result=prepare_assembly(tmp_path/'three')
    source=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/shaft/input_native09.json')
    original=load_json(source)
    inputs=build_stack_inputs(result['asset_root'],source)
    assert inputs['cases']==original['cases']
    assert inputs['task']==original['task']
    assert inputs['scene']['fixtures']==original['scene']['fixtures']
    assert inputs['scene']['isaacsim']==original['scene']['isaacsim']
    assert inputs['initial_state']['objects']==original['cases'][0]['reset_objects']
    for source_id in SOURCE_IDS:
        candidate=load_task_asset(source_id+'_sdf1024',result['asset_root'],verify_hashes=True)
        assert len(candidate['collision_parts'])==1
        assert (Path(result['asset_root'])/source_id/'source.obj').read_bytes()==(DEFAULT_SOURCE/source_id/'visual.obj').read_bytes()
