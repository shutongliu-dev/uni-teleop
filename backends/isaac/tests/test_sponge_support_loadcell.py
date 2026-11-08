import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validation.sponge_support_loadcell import (
    CONFIG,JOINT,PLATE,WEIGHT,author_instrument,contact_row,evaluate_raw,instrument_properties)


def test_cpu_authors_world_prismatic_force_spring_and_distinct_known_weight():
    Usd=pytest.importorskip('pxr.Usd');UsdPhysics=pytest.importorskip('pxr.UsdPhysics')
    stage=Usd.Stage.CreateInMemory()
    assert author_instrument(stage,'empty',{},Path('unused'),{}) is None
    properties=instrument_properties(stage)
    assert properties['body0']==[] and properties['body1']==[PLATE]
    a=properties['attributes']
    assert a['physics:axis']=='Z'
    assert a['drive:linear:physics:type']=='force'
    assert a['drive:linear:physics:stiffness']==2000.
    assert a['drive:linear:physics:damping']==45.
    assert a['drive:linear:physics:targetPosition']==0.
    assert a['drive:linear:physics:targetVelocity']==0.
    assert np.allclose([a['physics:lowerLimit'],a['physics:upperLimit']],[-.01,.01])
    assert not stage.GetPrimAtPath(WEIGHT)
    assert not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(PLATE)).GetKinematicEnabledAttr().Get()
    second=Usd.Stage.CreateInMemory()
    assert author_instrument(second,'rigid',{},Path('unused'),{})==WEIGHT
    assert instrument_properties(second)==properties
    assert np.isclose(UsdPhysics.MassAPI(second.GetPrimAtPath(WEIGHT)).GetMassAttr().Get(),.0792)


def test_fem_author_moves_both_original_layers_only_before_reset(tmp_path):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom');Gf=pytest.importorskip('pxr.Gf')
    file=tmp_path/'source.usda';source=Usd.Stage.CreateNew(str(file))
    root='/World/KitchenTask';pad=root+'/Actors/sponge';foam=root+'/VolumeBodies/foam'
    for path in (pad,foam):UsdGeom.Xform.Define(source,path).AddTranslateOp().Set(Gf.Vec3d(.25,-.67,.9001))
    source.GetRootLayer().Save();before=file.read_bytes()
    meta={'actors':{'sponge':pad},'objects':[{'id':'sponge','initial_pose_xyzw':[.25,-.67,.9001,0,0,0,1]}],
        'volume_bodies':{'foam':{'prim_path':foam}}}
    stage=Usd.Stage.CreateInMemory();author_instrument(stage,'fem',{'root_prim_path':root},file,meta)
    for path in (pad,foam):
        assert np.allclose(stage.GetPrimAtPath(path).GetAttribute('xformOp:translate').Get(),[.6,-.8,.9003])
    assert file.read_bytes()==before


def raw_static(mass):
    n=1201;t=np.arange(n)*CONFIG['dt_s']
    z=CONFIG['plate_rest_center_m'][2]-mass*CONFIG['gravity_m_s2']/CONFIG['spring_stiffness_n_m']
    p=np.tile([.6,-.8,z,0,0,0,1.],(n,1))
    return {'time_s':t,'native_step':np.arange(2,n+2),'plate_q':p,'plate_qd':np.zeros((n,6)),
        'load_q':np.tile([.6,-.8,z+.01+.0225,0,0,0,1.],(n,1)),'load_qd':np.zeros((n,6)),
        'foam_q':np.empty((n,0,3)),'foam_qd':np.empty((n,0,3))}


def test_tare_calibration_uses_displacement_without_contact_force_input():
    checks,measure=evaluate_raw(raw_static(.2),'empty')
    assert all(checks.values())
    empty={'passed':True,'measurements':measure}
    checks,known=evaluate_raw(raw_static(.2792),'rigid',empty)
    assert all(checks.values())
    assert np.isclose(known['tare_subtracted_spring_load_n'],.776952)
    # Same raw pose with an incorrect extra load must fail calibration.
    checks,bad=evaluate_raw(raw_static(.35),'rigid',empty)
    assert not checks['spring_added_weight_within_15_percent']
    assert bad['added_weight_relative_error']>.8


def test_limit_or_real_velocity_cannot_hide_behind_static_pose():
    data=raw_static(.2);data['plate_q'][:,2]=.8801
    checks,_=evaluate_raw(data,'empty')
    assert not checks['away_from_joint_limits']
    data=raw_static(.2);data['plate_qd'][-20:,2]=.05
    checks,_=evaluate_raw(data,'empty')
    assert not checks['quiet_native_plate']


def test_contact_parser_keeps_native_pair_range_and_empty_filter_boundary():
    class Ids:
        def __init__(self,value):self.value=np.asarray(value)
        def __getitem__(self,key):return Ids(self.value[key])
        def detach(self):return self
        def cpu(self):return self
        def clone(self):return Ids(self.value.copy())
        def numpy(self):return self.value
    class View:
        max_contact_data_count=8
        filter_count=0
        def get_raw_contact_data(self,dt):
            assert dt==1.
            return np.array([[99.],[.1],[.2]]),np.zeros((3,3)),np.tile([0.,0.,-1.],(3,1)),np.zeros((3,1)),np.array([2]),np.array([1]),Ids([999,7,7])
        def get_other_actor_paths_from_ids(self,ids):
            assert ids.value.tolist()==[7,7]
            return [WEIGHT,WEIGHT]
        def get_net_contact_forces(self,dt):return np.array([[0,0,-.3/dt]])
        def get_contact_force_matrix(self,dt):raise AssertionError('No native filtered matrix without filters')
    r=contact_row(View(),.0005)
    assert r['point_count']==2 and r['other_actor_ids']==[7,7]
    assert np.allclose(r['summed_normal_impulse_ns'],[0,0,-.3])
    assert r['filtered_contact_force_matrix_n'] is None
    assert r['filtered_matrix_availability']=='unavailable_without_filter_prims'


def test_static_pose_with_reported_damping_or_fast_load_is_not_weighing():
    data=raw_static(.2);data['plate_qd'][:,2]=.024
    checks,_=evaluate_raw(data,'empty')
    assert not checks['mean_damping_small_for_static_weighing']
    assert not checks['total_weight_from_full_balance']
    _,ref=evaluate_raw(raw_static(.2),'empty')
    data=raw_static(.2792);data['load_qd'][:,2]=10.;data['load_qd'][:,5]=50.
    checks,_=evaluate_raw(data,'rigid',{'passed':True,'measurements':ref})
    assert not checks['load_raw_quiet'] and not checks['load_angular_quiet']


def test_reference_identity_accepts_new_pid_but_rejects_source_hash_change(tmp_path):
    import copy
    from scripts.validation.sponge_support_loadcell import reference_report
    source={'snapshot_root':'/frozen/source','snapshot_manifest_sha256':'manifest-A',
        'entrypoint':{'path':'/frozen/source/probe.py','sha256':'probe-A'},
        'required_modules':['probe','geometry'],
        'imported_project_modules':{'probe':{'path':'/frozen/source/probe.py','sha256':'probe-A'}},
        'verified':True,'runtime_imports_verified':True,'pid':100}
    report={'schema':'sponge_passive_loadcell.v1','case':'empty','instrument':CONFIG,
        'source':source,'asset_manifest_sha256':'asset-A','evidence_sha256':{},'passed':True}
    path=tmp_path/'qualification.json';path.write_text(json.dumps(report))
    second=copy.deepcopy(source);second['pid']=999
    assert reference_report(path,'empty',second,'asset-A')['passed']
    second['snapshot_manifest_sha256']='manifest-B'
    with pytest.raises(ValueError,match='Calibration must match'):reference_report(path,'empty',second,'asset-A')
    second=copy.deepcopy(source);second['imported_project_modules']['probe']['sha256']='probe-B'
    with pytest.raises(ValueError,match='Calibration must match'):reference_report(path,'empty',second,'asset-A')


def test_instrument_offsets_are_explicit_without_claiming_canonical_match():
    Usd=pytest.importorskip('pxr.Usd')
    from scripts.validation.sponge_support_loadcell import collision_offsets
    stage=Usd.Stage.CreateInMemory();author_instrument(stage,'rigid',{},Path('unused'),{})
    values={x['path']:x['attributes'] for x in collision_offsets(stage,[PLATE,WEIGHT])}
    for path,expected in ((PLATE,.002),(WEIGHT,.0002)):
        assert values[path]['physxCollision:contactOffset']['authored']
        assert np.isclose(values[path]['physxCollision:contactOffset']['composed_value_m'],expected)
        assert values[path]['physxCollision:restOffset']=={'composed_value_m':0.,'authored':True}
    assert 'not claimed identical' in CONFIG['offset_scope']


def test_tail_includes_exact_800_intervals_with_native_float32_dt():
    from scripts.validation.sponge_support_loadcell import tail_mask
    data=raw_static(.2)
    data['time_s']=np.arange(1201)*float(np.float32(.0005))
    mask=tail_mask(data['native_step'])
    assert np.flatnonzero(mask)[0]==400 and int(mask.sum())==801
    checks,measure=evaluate_raw(data,'empty')
    assert all(checks.values())
    assert measure['tail_frames']==801 and measure['tail_intervals']==800
    assert np.isclose(measure['tail_elapsed_s'],.4,rtol=1e-6)
    # The transition into the first endpoint is outside this 800-interval window.
    data['plate_q'][399,2]-=.001
    checks,measure=evaluate_raw(data,'empty')
    assert checks['quiet_adjacent_plate'] and measure['tail_max_adjacent_plate_z_speed_m_s']==0.


def test_solver_is_explicit_per_run_and_calibrations_cannot_mix(tmp_path):
    from scripts.validation.sponge_support_loadcell import instrument_config,reference_report
    tgs,pgs=instrument_config(),instrument_config('PGS')
    assert tgs==CONFIG and pgs=={**CONFIG,'solver':'PGS'} and CONFIG['solver']=='TGS'
    with pytest.raises(ValueError):instrument_config('unknown')
    source={'snapshot_root':'/frozen/source','snapshot_manifest_sha256':'manifest-A',
        'entrypoint':{'path':'/frozen/source/probe.py','sha256':'probe-A'},
        'required_modules':['probe'],'imported_project_modules':{'probe':{'sha256':'probe-A'}}}
    report={'schema':'sponge_passive_loadcell.v1','case':'empty','instrument':pgs,
        'source':source,'asset_manifest_sha256':'asset-A','evidence_sha256':{},'passed':True}
    path=tmp_path/'qualification.json';path.write_text(json.dumps(report))
    assert reference_report(path,'empty',source,'asset-A',pgs)['passed']
    with pytest.raises(ValueError,match='Calibration must match'):
        reference_report(path,'empty',source,'asset-A',tgs)
