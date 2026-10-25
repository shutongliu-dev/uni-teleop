"""CPU geometry fixtures do not establish a native cooked-export positive."""
from copy import deepcopy
import numpy as np
import pytest

from bench.evaluation.cloth.cloth_robot_collision import SCHEMA,audit_cloth_robot,piece_in_body,state_digest,transform_xyzw,verify_native_pose,cooked_piece_geometry,reanalyse_export


def box():
    return {'vertices':[[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],
            'planes':[[1,0,0,-1],[-1,0,0,-1],[0,1,0,-1],[0,-1,0,-1],[0,0,1,-1],[0,0,-1,-1]]}


def fixture(vertices):
    states={'particle_q':np.array([vertices],float),'body_q':np.array([[[0,0,0,0,0,0,1]]],float)}
    export={'schema':SCHEMA,'status':'complete','visual_mesh_fallback':False,'issues':[],
        'state_hash_before':'a'*64,'state_hash_after':'a'*64,'layer_hashes_before':{'a':'b'},'layer_hashes_after':{'a':'b'},
        'colliders':[{'path':'/World/Robot/index/collision','body_name':'index','status':'verified_cooking_representation',
        'pieces':[box()],'mesh_to_body_column_matrix':np.eye(4).tolist()}]}
    return export,states,np.array([[0,1,2]]),['index']


def test_triangle_interior_penetration_caught_with_all_vertices_outside():
    data=fixture([[-3,-3,0],[3,-3,0],[0,3,0]])
    assert (abs(data[1]['particle_q'][0])>1).any(axis=1).all()
    report=audit_cloth_robot(*data)
    assert report['status']=='failed' and report['violations'][0]['triangles_penetrating_tolerance']==[0]


def test_separated_triangle_has_positive_geometry_only_fixture():
    report=audit_cloth_robot(*fixture([[2,2,2],[3,2,2],[2,3,2]]))
    assert report['status']=='passed' and report['scope'].startswith('Recorded cloth center-surface')


def test_actual_body_pose_rotates_and_translates_exported_collider():
    data=fixture([[4,-3,0],[10,-3,0],[7,3,0]])
    data[1]['body_q'][0,0]=[7,0,0,0,0,np.sqrt(.5),np.sqrt(.5)]
    assert audit_cloth_robot(*data)['status']=='failed'


def test_nonuniform_mesh_scale_preserves_normalized_halfspaces():
    matrix=np.diag([.5,2.,3.,1.]);matrix[:3,3]=[4,5,6]
    vertices,planes=piece_in_body(box(),matrix)
    np.testing.assert_allclose(vertices.min(0),[3.5,3,3])
    np.testing.assert_allclose(vertices.max(0),[4.5,7,9])
    np.testing.assert_allclose(np.linalg.norm(planes[:,:3],axis=1),1.)
    assert (vertices@planes[:,:3].T+planes[:,3]).max()<1e-10


@pytest.mark.parametrize('change',[
    lambda e:e.update(status='unchecked'),lambda e:e.update(visual_mesh_fallback=True),
    lambda e:e.update(issues=[{'reason':'missing_native_collider'}]),
    lambda e:e.update(state_hash_after='c'*64),lambda e:e.update(layer_hashes_after={'a':'different'}),
    lambda e:e['colliders'][0].update(status='unsupported'),lambda e:e['colliders'][0].update(body_name='unknown'),
    lambda e:e.update(colliders=[]),
])
def test_incomplete_export_never_certifies_clearance(change):
    data=fixture([[2,2,2],[3,2,2],[2,3,2]]);change(data[0])
    assert audit_cloth_robot(*data)['status']=='unchecked'


@pytest.mark.parametrize('bad',[True,0.,float('nan'),-1.])
def test_invalid_penetration_tolerance_rejected(bad):
    with pytest.raises(ValueError):audit_cloth_robot(*fixture([[2,2,2],[3,2,2],[2,3,2]]),tolerance_m=bad)


def test_native_halfspaces_cannot_have_nonfinite_or_inverted_support():
    value=box();value['planes'][0][3]=float('nan')
    with pytest.raises(ValueError):piece_in_body(value,np.eye(4))
    value=box();value['planes'][0][3]=3.
    with pytest.raises(ValueError):piece_in_body(value,np.eye(4))


def test_clock_and_target_changes_affect_export_state_identity():
    value={'q':np.array([1.,2.]),'time_s':np.array(.02)}
    original=state_digest(value);same=deepcopy(value)
    assert state_digest(same)==original
    same['time_s']=np.array(.04)
    assert state_digest(same)!=original


def test_invalid_body_quaternion_is_not_silently_normalized():
    with pytest.raises(ValueError):transform_xyzw([0,0,0,0,0,0,2])


def test_native_pose_verifies_scaled_local_collider_not_its_world_pose():
    native={'local_pos':[4.,5.,6.],'local_rot_xyzw':[0.,0.,np.sqrt(.5),np.sqrt(.5)]}
    matrix=transform_xyzw([*native['local_pos'],*native['local_rot_xyzw']])@np.diag([.5,2.,3.,1.])
    assert verify_native_pose(matrix,native)['position_max_abs_error_m']==0
    wrong=deepcopy(native);wrong['local_pos'][0]+=1.
    with pytest.raises(ValueError,match='disagrees'):verify_native_pose(matrix,wrong)


@pytest.mark.parametrize('matrix',[np.diag([-1.,1.,1.,1.]),np.diag([0.,1.,1.,1.]),
    np.array([[1.,.5,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])])
def test_unsupported_collider_transform_never_silently_approximated(matrix):
    with pytest.raises(ValueError):verify_native_pose(matrix,{'local_pos':[0,0,0],'local_rot_xyzw':[0,0,0,1]})


def diagnostic_fixture(root):
    """Synthetic sealing fixture, never native provenance."""
    from scripts.diagnostics.cloth_robot_collision_diagnostic import dump,sha
    export,states,triangles,names=fixture([[2,2,2],[3,2,2],[2,3,2]])
    states={key:np.repeat(value,2,axis=0) for key,value in states.items()}
    states['time_s']=np.array([0.,.02])
    command={'joint_positions_rad':{'synthetic':0.},'duration_s':.02}
    export['state_hash_before']=export['state_hash_after']=state_digest({k:v[0] for k,v in states.items()})
    dump(root/'metadata.json',{'body_names':names});dump(root/'colliders.json',export)
    np.savez(root/'states.npz',**states);np.save(root/'triangles.npy',triangles)
    dump(root/'plan.json',{'backend_sources_sha256':{},'warmup_steps':0,'actions':[command]})
    dump(root/'result.json',{'completed':True,'sources_unchanged':True})
    dump(root/'actions.json',[command]);(root/'observations.jsonl').write_text('{}\n{}\n')
    def seal():dump(root/'artifact_hashes.json',{p.name:sha(p) for p in root.iterdir() if p.name!='artifact_hashes.json'})
    seal();return seal


def test_manifest_and_export_frame_are_both_required(tmp_path):
    from scripts.diagnostics.cloth_robot_collision_diagnostic import load_episode,dump,read
    seal=diagnostic_fixture(tmp_path)
    assert len(load_episode(tmp_path)[2]['time_s'])==2
    export=read(tmp_path/'colliders.json');export['state_hash_before']=export['state_hash_after']='f'*64
    dump(tmp_path/'colliders.json',export)
    with pytest.raises(ValueError,match='artifact identity'):load_episode(tmp_path)
    seal()
    with pytest.raises(ValueError,match='recorded export frame'):load_episode(tmp_path)


def test_sealed_but_wrong_actions_rejected(tmp_path):
    from scripts.diagnostics.cloth_robot_collision_diagnostic import load_episode,dump,read
    seal=diagnostic_fixture(tmp_path)
    actions=read(tmp_path/'actions.json');actions[0]['joint_positions_rad']['synthetic']=.5
    dump(tmp_path/'actions.json',actions);seal()
    with pytest.raises(ValueError,match='frozen plan'):load_episode(tmp_path)


@pytest.mark.parametrize('x,expected',[(.9,'unchecked'),(.7,'failed'),(1.1,'passed')])
def test_raw_plane_support_hull_boundary_disagreement_is_not_silently_passed(x,expected):
    data=fixture([[x,0,0],[x,.1,0],[x,0,.1]])
    piece=box();piece['planes'][0][3]=-.8
    with pytest.raises(ValueError,match='contain'):piece_in_body(piece,np.eye(4))
    derived=cooked_piece_geometry(piece)
    assert derived['planes']==piece['planes']
    assert derived['representation_difference']['raw_planes_max_vertex_support_violation_m']==pytest.approx(.2)
    data[0]['colliders'][0]['pieces']=[derived]
    report=audit_cloth_robot(*data)
    assert report['status']==expected
    if expected=='unchecked':assert report['representation_boundary_uncertainties']


def saved_native_fixture():
    export,*_=fixture([[2,2,2],[3,2,2],[2,3,2]])
    row=export['colliders'][0]
    row.update(type='Mesh',approximation='convexHull',cooking_results=['PhysxCollisionRepresentationResult.RESULT_VALID'])
    export['stage_id']=1
    export['queries']=[{'body':'index','finished':True,'body_results':['PhysxPropertyQueryResult.VALID'],
        'colliders':[{'path':row['path'],'valid':True,'stage_id':1,'local_pos':[0,0,0],'local_rot_xyzw':[0,0,0,1],
                      'aabb_min':[-1,-1,-1],'aabb_max':[1,1,1]}]}]
    return export


def test_reanalysis_preserves_original_failure_and_raw_plane_values():
    export=saved_native_fixture();export['status']='unchecked'
    export['colliders'][0]['pieces'][0]['planes'][0][3]=-.8
    export['issues']=[{'reason':'collider_export_error','message':'Cooked polygon planes do not contain their vertices'}]
    before=deepcopy(export);new=reanalyse_export(export)
    assert new['status']=='complete' and new['original_assessment']['issues']==export['issues']
    assert new['colliders'][0]['pieces'][0]['planes']==export['colliders'][0]['pieces'][0]['planes']
    assert export==before


@pytest.mark.parametrize('change',[
    lambda e:e['issues'].append({'reason':'unrelated_native_error'}),
    lambda e:e['queries'][0].update(finished=False),
    lambda e:e['queries'][0]['colliders'][0].update(aabb_min=[-2,-1,-1]),
    lambda e:e['queries'][0]['colliders'][0].update(local_pos=[1,0,0]),
])
def test_reanalysis_never_clears_unrelated_native_failures(change):
    export=saved_native_fixture();change(export)
    assert reanalyse_export(export)['status']=='unchecked'
