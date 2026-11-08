"""CPU contract checks only; native FEM qualification requires a GPU run."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

from bench.paths import load_json
from bench.simulation.isaacsim_backend.geometry import tet_surface_triangles
from bench.simulation.isaacsim_backend.native_volume_bodies import validate_volume_bodies, VolumeBodyReadback
from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle, readonly, SCHEMA
from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
from bench.evaluation.deformable_robot_geometry import VolumeRobotGeometry
from bench.evaluation.cloth.cloth_robot_collision import SCHEMA as COOKED_SCHEMA, cooked_piece_geometry
from bench.runtime.native_volume_archive import NativeVolumeArchive


@pytest.fixture
def authored(tmp_path):
    path = tmp_path/'task.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    root = UsdGeom.Xform.Define(stage, '/World/Task').GetPrim()
    rigid = UsdGeom.Xform.Define(stage, '/World/Task/pad').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(rigid); UsdPhysics.MassAPI.Apply(rigid).CreateMassAttr(.01)
    collider = UsdGeom.Cube.Define(stage, '/World/Task/pad/shape').GetPrim(); UsdPhysics.CollisionAPI.Apply(collider)
    support = UsdGeom.Cube.Define(stage, '/World/Task/table').GetPrim(); UsdPhysics.CollisionAPI.Apply(support)
    body = UsdGeom.Xform.Define(stage, '/World/Task/foam').GetPrim()
    for api in ('OmniPhysicsDeformableBodyAPI', 'PhysxBaseDeformableBodyAPI'):
        body.AddAppliedSchema(api)
    for name, value in (('deformableBodyEnabled', True), ('kinematicEnabled', False)):
        body.CreateAttribute('omniphysics:'+name, Sdf.ValueTypeNames.Bool).Set(value)
    body.CreateAttribute('omniphysics:mass', Sdf.ValueTypeNames.Float).Set(.02)
    points = np.asarray([[0, 0, 0], [.02, 0, 0], [0, .02, 0], [0, 0, .02]])
    tetra = np.asarray([[0, 1, 2, 3]])
    surface = tet_surface_triangles(tetra)
    mesh = UsdGeom.TetMesh.Define(stage, '/World/Task/foam/SimulationMesh')
    mesh.GetPrim().AddAppliedSchema('OmniPhysicsVolumeDeformableSimAPI'); UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.CreatePointsAttr(Vt.Vec3fArray(points.tolist())); mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray(tetra.tolist()))
    mesh.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray(surface.tolist()))
    mesh.GetPrim().CreateAttribute('omniphysics:restShapePoints', Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(points.tolist()))
    mesh.GetPrim().CreateAttribute('omniphysics:restTetVtxIndices', Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray(tetra.tolist()))
    bond = stage.DefinePrim('/World/Task/bond', 'OmniPhysicsVtxXformAttachment')
    bond.CreateAttribute('omniphysics:attachmentEnabled', Sdf.ValueTypeNames.Bool).Set(True)
    bond.CreateAttribute('omniphysics:vtxIndicesSrc0', Sdf.ValueTypeNames.IntArray).Set([0, 1, 2])
    bond.CreateAttribute('omniphysics:localPositionsSrc1', Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(points[:3].tolist()))
    bond.CreateRelationship('omniphysics:src0').SetTargets([mesh.GetPath()])
    bond.CreateRelationship('omniphysics:src1').SetTargets([rigid.GetPath()])
    row = dict(prim_path=str(body.GetPath()), simulation_mesh_path=str(mesh.GetPath()), parent_rigid_body='pad',
        rest_points_m=points.tolist(), tetrahedra=tetra.tolist(), surface_triangles=surface.tolist(), total_mass_kg=.02,
        attachment=dict(prim_path=str(bond.GetPath()), node_indices=[0, 1, 2], rigid_local_positions_m=points[:3].tolist()))
    stage.GetRootLayer().Save()
    return stage, root, {'foam': row}, {'pad': str(rigid.GetPath())}, path


def validated(authored):
    stage, root, volumes, rigid, _ = authored
    return validate_volume_bodies(stage, root, volumes, rigid)


def test_bundle_and_mount_exclude_volume_from_fixed_support(authored):
    _, _, volumes, bodies, path = authored
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stage = Usd.Stage.CreateInMemory(); UsdPhysics.Scene.Define(stage, '/physicsScene')
    info = add_bundle(stage, dict(schema=SCHEMA, usd_path=str(path), usd_sha256=sha,
        root_prim_path='/World/Task', rigid_bodies=bodies, volume_bodies=volumes), physics_scene_path='/physicsScene')
    assert info['fixtures']['native_bundle_fixed']['collision_paths'] == ['/World/Task/table']
    assert info['volume_bodies']['foam']['node_count'] == 4
    target = Usd.Stage.CreateInMemory(); UsdPhysics.Scene.Define(target, '/physicsScene')
    mounted = mount_task_layers(target, [dict(id='task', usd_path=str(path), sha256=sha,
        source_prim_path='/World/Task', target_prim_path='/World/Task', rigid_bodies=bodies, volume_bodies=volumes)],
        physics_scene_path='/physicsScene')
    assert mounted['static_collision_paths'] == ['/World/Task/table']
    from bench.simulation.isaacsim_backend.task_readback import TaskLayerReadback
    with pytest.raises(ValueError, match='native_task_bundle'):
        TaskLayerReadback(target, mounted['layers'])
    assert mounted['layers'][0]['volume_bodies']['foam']['attachment_response'] == 'native_two_way_rigid_endpoint'


@pytest.mark.parametrize('fault', ['missing', 'mass', 'topology', 'anchors', 'kinematic', 'extra_collision', 'extra_bond'])
def test_strict_volume_inventory_rejects_changed_authored_physics(authored, fault):
    stage, root, volumes, rigid, _ = authored
    if fault == 'missing': volumes = {}
    elif fault == 'mass': volumes['foam']['total_mass_kg'] *= 2
    elif fault == 'topology': volumes['foam']['surface_triangles'].pop()
    elif fault == 'anchors': volumes['foam']['attachment']['rigid_local_positions_m'][0][0] = .001
    elif fault == 'kinematic': stage.GetPrimAtPath(volumes['foam']['prim_path']).GetAttribute('omniphysics:kinematicEnabled').Set(True)
    elif fault == 'extra_collision': UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, '/World/Task/foam/hidden').GetPrim())
    elif fault == 'extra_bond': stage.DefinePrim('/World/Task/extra', 'OmniPhysicsVtxXformAttachment')
    with pytest.raises(ValueError): validate_volume_bodies(stage, root, volumes, rigid)


def tensor_view(row):
    # Deliberately no setters: the reader must preserve native warmup state.
    return SimpleNamespace(count=1, prim_paths=[row['prim_path']], simulation_mesh_prim_paths=[row['simulation_mesh_path']],
        collision_mesh_prim_paths=[row['simulation_mesh_path']], max_simulation_nodes_per_body=row['node_count'],
        max_simulation_elements_per_body=row['tetrahedron_count'], num_nodes_per_element=4,
        get_simulation_element_indices=lambda: np.asarray(row['tetrahedra'])[None],
        get_rest_element_indices=lambda: np.asarray(row['tetrahedra'])[None],
        get_rest_nodal_positions=lambda: np.asarray(row['rest_points_m'])[None],
        get_simulation_nodal_positions=lambda: np.asarray(row['rest_world_points_m'])[None]+.00001,
        get_simulation_nodal_velocities=lambda: np.ones((1, row['node_count'], 3))*.001)


def test_live_native_getters_keep_warmup_and_reject_tensor_topology_mismatch(authored):
    info = validated(authored); view = tensor_view(info['foam'])
    reader = VolumeBodyReadback(info, SimpleNamespace(current_time_step_index=0, current_time=0.),
                              SimpleNamespace(device_ordinal=0, device='cuda:0', create_volume_deformable_body_view=lambda path: view))
    assert reader.initialization['position_write_calls'] == 0
    assert reader.initialization['bodies']['foam']['maximum_node_speed_m_s'] > 0
    capture = reader.capture(); capture['task_volume_foam_q'][:] = 9
    assert np.max(reader.capture()['task_volume_foam_q']) < 1
    view.get_simulation_element_indices = lambda: np.asarray([[[0, 2, 1, 3]]])
    with pytest.raises(RuntimeError, match='indices'):
        VolumeBodyReadback(info, SimpleNamespace(), SimpleNamespace(device_ordinal=0, device='cuda:0', create_volume_deformable_body_view=lambda path: view))


def cube_export():
    from scipy.spatial import ConvexHull
    vertices = np.asarray([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
    piece = cooked_piece_geometry({'vertices': vertices.tolist(), 'planes': ConvexHull(vertices).equations.tolist()})
    return {'schema': COOKED_SCHEMA, 'status': 'complete', 'issues': [], 'visual_mesh_fallback': False,
        'state_hash_before': 'hash', 'state_hash_after': 'hash', 'layer_hashes_before': {'usd': 'hash'},
        'layer_hashes_after': {'usd': 'hash'}, 'colliders': [dict(status='verified_cooking_representation',
            body_name='single_hand', path='/World/Robot/collider', mesh_to_body_column_matrix=np.eye(4).tolist(), pieces=[piece])]}


def test_single_hand_geometry_checks_boundary_clearance_and_solid_enclosure(authored):
    info = validated(authored); checker = VolumeRobotGeometry(cube_export(), ['single_hand'], info)
    poses = np.asarray([[0, 0, 0, 0, 0, 0, 1.]])
    points = np.asarray(info['foam']['rest_points_m'])
    check = lambda p: checker.check({'foam': {'positions_m': p}}, poses)
    assert check(points+1)['fully_separated'] is True
    assert check(points+np.asarray([.014, 0, 0]))['fully_separated'] is False
    # Robot is wholly enclosed; no boundary triangle touches the robot convex.
    enclosure = np.asarray([[-1, -1, -1], [4, -1, -1], [-1, 4, -1], [-1, -1, 4.]])
    result = check(enclosure)
    assert result['fully_separated'] is False
    assert result['near_pairs'][0]['solid_enclosure'] is True
    assert result['method'] == 'native_cooked_convex_geometry'
    assert result['contact_force_measured'] is False
    bad = cube_export(); bad['issues'].append({'error': 'missing_collider'})
    with pytest.raises(ValueError): VolumeRobotGeometry(bad, ['single_hand'], info)


def archive_frame(info, step):
    return readonly(dict(baseline=step == 0, time_s=step*.0005, native_time_s=step*.0005,
        native_step=step, physics_dt_s=.0005,
        volume_bodies={'foam': dict(positions_m=np.asarray(info['foam']['rest_points_m']),
                                   velocities_m_s=np.zeros((4, 3)))},
        objects={'pad': dict(position_m=[0,0,0], quaternion_wxyz=[1,0,0,0],
                             linear_velocity_m_s=[0,0,0], angular_velocity_rad_s=[0,0,0])},
        volume_robot_geometry={'complete': True, 'fully_separated': True},
        robot_body_q=np.asarray([[0, 0, 0, 0, 0, 0, 1.]])))


@pytest.mark.parametrize('failed_clock', [False, True])
def test_private_archive_preserves_every_native_node_and_partial_failure(authored, tmp_path, failed_clock):
    info = validated(authored)
    reader = SimpleNamespace(volumes=SimpleNamespace(info=info, initialization={}), last_volume_frame=archive_frame(info, 0),
        volume_frame_sink=None, volume_robot_body_names=['single_hand'], volume_robot_export=cube_export(),
        info={'usd_path': str(authored[-1]), 'usd_sha256': hashlib.sha256(authored[-1].read_bytes()).hexdigest()})
    archive = NativeVolumeArchive(tmp_path/'archive', reader, chunk_size=2)
    reader.volume_frame_sink(archive_frame(info, 1))
    if failed_clock:
        with pytest.raises(ValueError, match='mixed-tick'): reader.volume_frame_sink(archive_frame(info, 3))
    else:
        reader.volume_frame_sink(archive_frame(info, 2))
    archive.close(actual_native_step=3 if failed_clock else 2, reason='test_failure' if failed_clock else 'finish_request')
    archive.close(actual_native_step=999, reason='idempotent')
    manifest = load_json(archive.path/'manifest.json')
    assert manifest['complete_native_step_coverage'] is (not failed_clock)
    assert manifest['frame_count'] == (2 if failed_clock else 3)
    arrays = [np.load(archive.path/chunk['path'], allow_pickle=False) for chunk in manifest['chunks']]
    assert np.concatenate([row['task_volume_foam_q'] for row in arrays]).shape == (manifest['frame_count'], 4, 3)
    assert np.concatenate([row['native_step'] for row in arrays]).tolist() == list(range(manifest['frame_count']))
    assert np.concatenate([row['task_rigid_pad_q'] for row in arrays]).shape == (manifest['frame_count'], 7)
    assert reader.volume_frame_sink is None
    assert all(hashlib.sha256((archive.path/c['path']).read_bytes()).hexdigest() == c['sha256'] for c in manifest['chunks'])


def test_native_reader_delivers_readonly_nodes_before_observer_failure(authored, tmp_path):
    from bench.simulation.isaacsim_backend.native_task_bundle import NativeTaskReader
    info = {'volume_bodies': validated(authored), 'particle_sets': {}, 'joints': {}}
    view = tensor_view(info['volume_bodies']['foam'])
    world = SimpleNamespace(current_time_step_index=0, current_time=0.)
    path = tmp_path/'observer.py'
    path.write_text('class NativeTaskObserver:\n def __init__(self,c): pass\n def on_frame(self,f):\n  assert not f["volume_bodies"]["foam"]["positions_m"].flags.writeable\n  raise RuntimeError("observer failure")\n def on_observe(self): return {}\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = NativeTaskReader(authored[0], info, {'observer': {'path': str(path), 'sha256': sha,
        'files': {path.name: sha}, 'configuration': {}}}, world=world,
        simulation_view=SimpleNamespace(device_ordinal=0, device='cuda:0', create_volume_deformable_body_view=lambda path: view))
    reader.volume_geometry = VolumeRobotGeometry(cube_export(), ['single_hand'], info['volume_bodies'])
    reader._robot_poses = lambda: np.asarray([[1, 0, 0, 0, 0, 0, 1.]])
    captured = []; reader.volume_frame_sink = captured.append
    with pytest.raises(RuntimeError, match='observer failure'):
        reader.sample(objects={}, contacts=[], time_s=0., native_time_s=0., native_step=0, physics_dt_s=.0005, baseline=True)
    assert len(captured) == 1
    assert captured[0]['volume_robot_geometry']['fully_separated'] is True
    with pytest.raises(ValueError): captured[0]['volume_bodies']['foam']['positions_m'][0, 0] = 1
    with pytest.raises(TypeError): captured[0]['volume_bodies']['extra'] = {}


def test_volume_readback_rejects_cpu_fallback(authored):
    with pytest.raises(RuntimeError, match='CUDA'):
        VolumeBodyReadback(validated(authored), SimpleNamespace(), SimpleNamespace(device_ordinal=-1, device='cpu'))


@pytest.mark.parametrize('fault', [None, 'lost_tick', 'bond_shift', 'stale_geometry', 'endpoint'])
def test_independent_archive_replay_detects_self_consistent_tampering(authored, tmp_path, fault):
    from bench.runtime.native_volume_archive import plain
    from scripts.validation.native_volume_archive_review import review_native_volumes
    info = validated(authored); export = cube_export(); collider = export['colliders'][0]
    collider.update(type='Mesh', approximation='convexHull', cooking_results=['PhysxCollisionRepresentationResult.RESULT_VALID'])
    export['stage_id'] = 1
    export['queries'] = [{'body': 'single_hand', 'finished': True, 'body_results': ['PhysxPropertyQueryResult.VALID'],
        'colliders': [dict(path=collider['path'], valid=True, stage_id=1, local_pos=[0,0,0], local_rot_xyzw=[0,0,0,1],
                           aabb_min=[-.01]*3, aabb_max=[.01]*3)]}]
    init = dict(mode='author-only', position_write_calls=0, velocity_write_calls=0,
                tensor_device='cuda:0', tensor_device_ordinal=0, native_step=0, native_time_s=0.)
    checker = VolumeRobotGeometry(export, ['single_hand'], info)
    def frame(step):
        row = plain(archive_frame(info, step)); row['robot_body_q'] = np.asarray([[1.,0,0,0,0,0,1]])
        row['volume_bodies']['foam']['positions_m'] = np.asarray(row['volume_bodies']['foam']['positions_m'])
        row['volume_bodies']['foam']['velocities_m_s'] = np.asarray(row['volume_bodies']['foam']['velocities_m_s'])
        row['volume_robot_geometry'] = checker.check(row['volume_bodies'], row['robot_body_q'])
        return row
    q = np.asarray([0.,0,0,0,0,0,1])
    endpoint = {'native_physics_step_index': np.asarray([0,40,80]),
        'body_q': np.repeat(np.asarray([[1.,0,0,0,0,0,1], q])[None], 3, axis=0),
        'object_q': np.repeat(q[None,None], 3, axis=0), 'object_qd': np.zeros((3,1,6)),
        'task_volume_foam_q': np.repeat(np.asarray(info['foam']['rest_points_m'])[None], 3, axis=0),
        'task_volume_foam_qd': np.zeros((3,4,3))}
    from bench.evaluation.cloth.cloth_robot_collision import state_digest
    export['state_hash_before'] = export['state_hash_after'] = state_digest({k:v[0] for k,v in endpoint.items()})
    reader = SimpleNamespace(volumes=SimpleNamespace(info=info, initialization=init), last_volume_frame=frame(0),
        volume_frame_sink=None, volume_robot_body_names=['single_hand'], volume_robot_export=export,
        info={'usd_path': str(authored[-1]), 'usd_sha256': hashlib.sha256(authored[-1].read_bytes()).hexdigest()})
    private = tmp_path/'private'; archive = NativeVolumeArchive(private/'native_volumes', reader)
    for step in range(1, 81): archive.append(frame(step))
    archive.close(actual_native_step=80, reason='finish_request')
    if fault == 'endpoint': endpoint['task_volume_foam_q'][-1, 0, 0] += .001
    np.savez_compressed(private/'states.npz', **endpoint)
    if fault in ('lost_tick', 'bond_shift', 'stale_geometry'):
        path = archive.path/'frames_000000.npz'; path.chmod(0o644)
        with np.load(path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
        if fault == 'lost_tick': arrays['native_step'][5] += 1
        elif fault == 'bond_shift':
            arrays['task_rigid_pad_q'] = arrays['task_rigid_pad_q'].astype(float)
            arrays['task_rigid_pad_q'][5, 0] += .01
        else: arrays['robot_body_q'][5, 0, 0] = 0
        np.savez_compressed(path, **arrays)
        manifest_path = archive.path/'manifest.json'; manifest_path.chmod(0o644)
        manifest = load_json(manifest_path); manifest['chunks'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    runtime = dict(scene_spec={'native_task_bundle': {'volume_bodies': authored[2], **reader.info}},
        task_volume_initialization=init, physics_dt_s=.0005, object_names=['pad'])
    task = dict(asset_metadata={'volume_bodies': authored[2]}, evaluation={'thresholds': {
        'stability_pose_window_s': .02, 'maximum_linear_speed_m_s': .01,
        'maximum_stable_position_range_m': .004, 'maximum_penetration_m': .004, 'stable_s': .02}})
    evidence = {}; result = review_native_volumes(private, runtime, task, evidence)
    assert result['passed'] is (fault is None), result
    if fault is None:
        assert result['frames_checked'] == 81
        assert result['endpoints_checked'] == 3
        assert result['terminal_released_quiet_s']['foam'] >= .02
        unsigned = review_native_volumes(private, runtime, task, {}, expected_source=tmp_path/'not_frozen')
        assert unsigned['passed'] is False
        assert 'source receipt' in unsigned['errors'][0]['detail']
    else: assert result['errors']
    assert str((archive.path/'frames_000000.npz').resolve()) in evidence
