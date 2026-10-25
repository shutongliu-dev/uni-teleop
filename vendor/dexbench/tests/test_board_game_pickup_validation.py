"""CPU-only contact geometry and evidence accounting; never grasp qualification."""
from collections import deque
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scenes.board_games.tabletop import geometry_for
from bench.geometry import quaternion_matrix
from scripts.validation.validate_board_game_pickup import (
    PickupLimits, PickupObserver, checked_plan, contact_rows, handle_contacts, motion_phases,
    token_lowest_z_m)
from scripts.validation.validate_scenario import digest


TARGET = '/World/Objects/black_00'
PADS = {'thumb': '/World/Robot/Left/left_thumb_elastomer',
        'index': '/World/Robot/Left/left_index_elastomer'}


def pair(finger, local, *, impulse=.001, separation=-.0001, origin=(.2, -.5, .807)):
    return {'actor0': TARGET, 'actor1': PADS[finger], 'normal_impulses_ns': [impulse],
            'points_m': [(np.asarray(origin)+local).tolist()],
            'normals': [[1.,0.,0.]], 'separations_m': [separation]}


def test_opposed_actual_pad_contact_on_handle_required_not_actor_pair_alone():
    g, limits = geometry_for('tic_tac_toe_3x3'), PickupLimits()
    pose = np.array([.2, -.5, .807, 0.,0.,0.,1.])
    valid = [pair('thumb', [0.,.02,.027]), pair('index', [0.,-.02,.027])]
    result = handle_contacts(valid, TARGET, PADS, pose, g, limits)
    assert result['both_pads_on_opposed_handle_sides']
    assert json.loads(json.dumps(result)) == result
    for rows in (valid[:1], [valid[0], pair('index',[0.,.02,.027])],
                 [pair('thumb',[0.,.028,0.]), pair('index',[0.,-.028,0.])],
                 [pair('thumb',[0.,.02,.047]), valid[1]],
                 [pair('thumb',[0.,.02,.027],impulse=0.), valid[1]],
                 [pair('thumb',[0.,.02,.027],impulse=-.001), valid[1]],
                 [pair('thumb',[0.,.02,.027],separation=-.003), valid[1]]):
        assert not handle_contacts(rows, TARGET, PADS, pose, g, limits)['both_pads_on_opposed_handle_sides']


def test_contact_points_are_transformed_by_actual_token_rotation():
    g = geometry_for('tic_tac_toe_3x3')
    pose = np.array([.2,-.5,.807, 0.,0.,2**-.5,2**-.5])
    contacts = [pair('thumb',[-.02,0.,.027]),pair('index',[.02,0.,.027])]
    result = handle_contacts(contacts,TARGET,PADS,pose,g,PickupLimits())
    assert result['both_pads_on_opposed_handle_sides']
    assert result['local_contact_points_m']['thumb'][0] == pytest.approx([0.,.02,.027])


@pytest.mark.parametrize(('quaternion_xyzw', 'expected'), [
    ([0.,0.,0.,1.], .813),  # Upright base lower cap: .820 - .007.
    ([1.,0.,0.,0.], .773),  # Inverted handle end: .820 - .047.
    ([2**-.5,0.,0.,2**-.5], .792),  # Sideways base rim: .820 - .028.
])
def test_full_token_lowest_z_has_known_upright_inverted_and_sideways_truth(quaternion_xyzw, expected):
    pose = np.array([.2,-.5,.820,*quaternion_xyzw])
    assert token_lowest_z_m(pose, geometry_for('tic_tac_toe_3x3')) == pytest.approx(expected, abs=1.e-12)


@pytest.mark.parametrize('angle_deg', [37.,89.,100.,137.,179.])
def test_full_token_lowest_z_matches_transformed_cylinder_boundary(angle_deg):
    # Independent geometric oracle: transform sampled end-cap boundary points,
    # without evaluating the cylinder support-function formula under test.
    g = geometry_for('tic_tac_toe_3x3')
    axis = np.array([2.,3.,1.])/math.sqrt(14.)
    half_angle = math.radians(angle_deg)/2
    pose = np.array([.2,-.5,.820,*(axis*math.sin(half_angle)),math.cos(half_angle)])
    angles = np.arange(16384)*2*math.pi/16384
    rings = []
    for radius, heights in ((g.piece_radius_m, [-g.piece_base_height_m/2,g.piece_base_height_m/2]),
                            (g.handle_radius_m, [g.piece_base_height_m/2,g.piece_base_height_m/2+g.handle_height_m])):
        for height in heights:
            rings.append(np.column_stack((radius*np.cos(angles),radius*np.sin(angles),np.full(len(angles),height))))
    points = np.concatenate(rings) @ quaternion_matrix(pose[[6,3,4,5]]).T + pose[:3]
    assert token_lowest_z_m(pose,g) == pytest.approx(float(points[:,2].min()),abs=1.e-9)


def test_distal_shell_handle_contact_can_prove_pickup_but_is_not_pad_only():
    shell = '/World/Robot/Left/left_index_DP'
    parts = {'thumb': {PADS['thumb']: 'pad'}, 'index': {PADS['index']: 'pad', shell: 'distal_shell'}}
    valid = [pair('thumb',[0.,.02,.027]), pair('index',[0.,-.02,.027])]
    valid[1]['actor1'] = shell
    result = handle_contacts(valid,TARGET,PADS,np.array([.2,-.5,.807,0.,0.,0.,1.]),
                             geometry_for('tic_tac_toe_3x3'),PickupLimits(),parts)
    assert result['both_fingers_on_opposed_handle_sides']
    assert not result['both_pads_on_opposed_handle_sides']
    assert result['actual_handle_contact_components']['index'][shell]['component'] == 'distal_shell'
    valid[1]['points_m'][0][2] = .807  # Shell touching the disk is not a handle grip.
    assert not handle_contacts(valid,TARGET,PADS,np.array([.2,-.5,.807,0.,0.,0.,1.]),
        geometry_for('tic_tac_toe_3x3'),PickupLimits(),parts)['both_fingers_on_opposed_handle_sides']


def test_contact_window_allows_short_gaps_but_not_missing_or_sparse_pinch():
    observer = object.__new__(PickupObserver)
    observer.env = SimpleNamespace(dt=.0005)
    observer.limits = PickupLimits()
    observer.window = deque((i*.0005, True) for i in range(201))
    assert observer.pinch_ready()
    observer.window = deque((i*.0005, not 60 <= i < 70) for i in range(201))
    assert observer.pinch_ready()
    observer.window = deque((i*.0005, not 60 <= i < 81) for i in range(201))
    assert not observer.pinch_ready()
    observer.window = deque((i*.0005, True) for i in range(199))
    assert not observer.pinch_ready()


class RawSample:
    def __init__(self, arrays):
        self.arrays = arrays

    def for_env(self, env):
        return self.arrays


def test_raw_contact_projection_retains_points_without_second_acquisition():
    env = SimpleNamespace(_rigid_contacts=SimpleNamespace(max_contact_data_count=8),
        _contact_sensor_paths=[TARGET,PADS['thumb']], _contact_filter_paths=[[PADS['thumb']],[TARGET]])
    arrays = (np.array([[.001],[.001]]), np.array([[.2,-.48,.834],[.2,-.48,.834]]),
        np.array([[0.,-1.,0.],[0.,1.,0.]]), np.array([[-.0001],[-.0001]]),
        np.array([[1],[1]]), np.array([[0],[1]]))
    result = contact_rows(env,RawSample(arrays))
    assert len(result) == 1
    assert result[0]['points_m'] == [[.2,-.48,.834]]
    assert result[0]['normal_impulses_ns'] == [.001]
    # A corrupted read is rejected, never inferred as absent contact.
    arrays[1][0,0] = np.nan
    with pytest.raises(RuntimeError,match='Invalid'):
        contact_rows(env,RawSample(arrays))


class FakeEnv:
    """Minimal exchange double; no native simulator or physical claims."""
    def __init__(self):
        self.dt, self.time_s, self._reset_count = .0005, 0., 1
        self.world = SimpleNamespace(current_time_step_index=100,current_time=1.)
        self.world.step = self.world_step
        self.object_names = ['black_00','white_00']
        self.target = np.array([.2,-.5,.807,0.,0.,0.,1.])
        self.scene_info = {'objects': {'black_00':{'path':TARGET},'white_00':{'path':'/World/Objects/white_00'}},
            'robot_link_paths': {'left_thumb_elastomer':PADS['thumb'],'left_index_elastomer':PADS['index']},
            'workbench':{'supports':{'table':'/World/Table'}}}
        self._contact_sensor_paths = [TARGET,*PADS.values(),'/World/Objects/white_00']
        filters = ['/World/Fixtures/reserve_black_west','/World/Table',*self._contact_sensor_paths]
        self._contact_filter_paths = [list(filters) for _ in self._contact_sensor_paths]
        self._rigid_contacts = SimpleNamespace(max_contact_data_count=8,get_contact_data=self.get_contacts)
        self.acquisitions = 0

    def world_step(self, *, render):
        assert render is False
        self.world.current_time_step_index += 1
        self.world.current_time += self.dt

    def get_contacts(self, *, dt):
        self.acquisitions += 1
        assert dt == 1.
        return (np.array([[.001]]),np.array([[.2,-.5,.8]]),np.array([[0.,0.,1.]]),
                np.array([[-1.e-7]]),np.array([[1]]),np.array([[0]]))

    def capture_state(self):
        return {'object_q':np.array([self.target,[.5,.5,.807,0.,0.,0.,1.]]),
                'object_qd':np.zeros((2,6)), 'joint_q':np.zeros(58),
                'native_physics_step_index':np.array(self.world.current_time_step_index)}

    def _project_rigid_contacts(self,sample):
        return contact_rows(self,sample)

    def advance(self):
        self._current_step_time = self.time_s+self.dt
        self.world.step(render=False)
        self.time_s += self.dt
        return self._contact_exchange.consume(self)


def test_observer_observes_existing_steps_once_and_restores_hook(tmp_path):
    env = FakeEnv()
    original = env.world.step
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,
                           'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    for _ in range(10):
        env.advance()
    summary = observer.close()
    assert env.world.step == original
    assert env.acquisitions == summary['samples'] == 10
    assert summary['checks']['one_contact_acquisition_per_native_step']
    assert summary['checks']['contact_point_frame_verified_on_declared_support']
    assert summary['actual_reserve_support']['surface_z_m'] == .8
    assert summary['passed'] is False  # Still resting on support, not a pickup.
    rows = [json.loads(row) for row in (tmp_path/'native_frames.jsonl').read_text().splitlines()]
    assert len(rows) == 10 and all(row['state_unchanged_by_observer'] for row in rows)
    assert all(row['other_support_impulse_ns'] > 0 for row in rows)


@pytest.mark.parametrize('bad_field,bad_value', [('normal_impulses_ns',float('inf')),('normals',float('nan'))])
def test_rejected_sample_is_preserved_without_reacquiring_or_accepting_it(tmp_path,bad_field,bad_value):
    env = FakeEnv()
    original = env._rigid_contacts.get_contact_data
    fields = ('normal_impulses_ns','points_m','normals','separations_m','counts','starts')
    def bad_contacts(*,dt):
        values = original(dt=dt)
        values[fields.index(bad_field)].flat[0] = bad_value
        return values
    env._rigid_contacts.get_contact_data = bad_contacts
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    with pytest.raises(RuntimeError,match='Invalid raw native contact point evidence:'):
        env.advance()
    summary = observer.close()
    assert env.acquisitions == 1 and summary['samples'] == 0
    assert summary['passed'] is False
    assert not summary['checks']['one_contact_acquisition_per_native_step']
    rejected = load_json(tmp_path/'rejected_contact_sample.json')
    assert rejected['original_native_acquisition_retained'] is True
    assert rejected['extra_native_contact_acquisitions'] == 0
    assert rejected['native_step_index'] == 101
    arrays = np.load(tmp_path/'rejected_contact_sample.npz')
    actual = arrays[bad_field].flat[0]
    assert np.isnan(actual) if np.isnan(bad_value) else actual == bad_value
    assert np.load(tmp_path/'rejected_contact_state.npz')['native_physics_step_index'] == 101


@pytest.mark.parametrize(('failure', 'stage'), [
    ('nonfinite_state', 'native_state_capture'), ('state_read_error', 'native_state_capture'),
    ('backend_projection_error', 'rigid_contact_projection'),
    ('sample_guard_error', 'rigid_contact_projection'), ('quaternion', 'target_pose_validation'),
])
def test_post_acquisition_rejection_preserves_original_sample_and_exact_error(tmp_path, failure, stage):
    env = FakeEnv()
    supports = {'black_00': {'paths': ('/World/Fixtures/reserve_black_west',),
        'surface_z_m': .8, 'fixture_id': 'reserve_black_west'}}
    observer = PickupObserver(env, tmp_path, 'black_00', 'left',
        geometry_for('tic_tac_toe_3x3'), PickupLimits(), supports)
    capture = env.capture_state
    if failure in ('nonfinite_state', 'state_read_error'):
        def rejected_state():
            if failure == 'state_read_error':
                raise RuntimeError('Synthetic native state read failed')
            state = capture()
            state['object_q'][0, 0] = float('nan')
            return state
        env.capture_state = rejected_state
    elif failure in ('backend_projection_error', 'sample_guard_error'):
        def rejected_projection(sample):
            if failure == 'sample_guard_error':
                observer.exchange.invalidate()
                sample.for_env(env)
            raise RuntimeError('Synthetic rigid projection failed')
        env._project_rigid_contacts = rejected_projection
    else:
        env.target[3:] = [0., 0., 0., 2.]
    with pytest.raises(RuntimeError) as raised:
        env.advance()
    summary = observer.close()
    rejection = load_json(tmp_path/'rejected_contact_sample.json')
    assert rejection['error_detail'] == str(raised.value)
    assert rejection['rejection_stage'] == stage
    assert rejection['state_capture_available'] is (failure != 'state_read_error')
    assert rejection['extra_native_contact_acquisitions'] == 0
    assert rejection['original_native_acquisition_retained'] is True
    assert rejection['sample_delivered_to_public_reader'] is False
    with np.load(tmp_path/'rejected_contact_sample.npz') as arrays:
        assert arrays['normal_impulses_ns'].tolist() == [[.001]]
        assert arrays['points_m'].tolist() == [[.2, -.5, .8]]
    assert (tmp_path/'rejected_contact_state.npz').exists() is (failure != 'state_read_error')
    if failure == 'nonfinite_state':
        with np.load(tmp_path/'rejected_contact_state.npz') as state:
            assert np.isnan(state['object_q'][0, 0])
    assert env.acquisitions == 1 and summary['samples'] == 0
    assert summary['passed'] is False
    assert not summary['checks']['one_contact_acquisition_per_native_step']


def test_signed_native_impulses_are_retained_and_mixed_support_cannot_cancel(tmp_path):
    env = FakeEnv()
    env.mode = 'initial'
    def contacts(*,dt):
        env.acquisitions += 1
        assert dt == 1.
        if env.mode == 'initial':
            rows = [(0,.001,[.2,-.5,.8])]
        else:
            rows = [(0,.001,[.2,-.5,.85]),(0,-.001,[.2,-.5,.85]),
                (env._contact_filter_paths[0].index(PADS['thumb']),.001,[.2,-.48,.884]),
                (env._contact_filter_paths[0].index(PADS['index']),.001,[.2,-.52,.884])]
        counts = np.zeros((len(env._contact_sensor_paths),len(env._contact_filter_paths[0])),dtype=int)
        starts = counts.copy()
        for i,(filt,_,_) in enumerate(rows):
            if counts[0,filt]==0:starts[0,filt]=i
            counts[0,filt]+=1
        return np.array([[f] for _,f,_ in rows]),np.array([p for _,_,p in rows]),np.tile([0.,0.,1.],(len(rows),1)),np.full((len(rows),1),-.0001),counts,starts
    env._rigid_contacts.get_contact_data = contacts
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    env.advance()
    env.mode,env.target[2]='mixed',.857
    for _ in range(1000):env.advance()
    support=next(row for row in observer.last['raw_contacts'] if '/World/Fixtures/reserve_black_west' in (row['actor0'],row['actor1']))
    assert support['normal_impulses_ns']==[.001,-.001]
    assert support['impulse_vector_ns']==[0.,0.,0.]
    assert support['impulse_magnitude_sum_ns']==pytest.approx(.002)
    assert observer.last['pinch_window_qualified']
    assert observer.last['other_support_impulse_ns']==pytest.approx(.002)
    assert observer.last['airborne_stable_dwell_s']==0.
    result=observer.close()
    assert not result['passed'] and result['checks']['one_contact_acquisition_per_native_step']


def test_motion_plan_keeps_hand_goal_during_lift_and_uses_real_side_base():
    rotation = np.diag([1.,-1.,-1.])
    base = np.eye(4)
    base[:3,:3] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    current = {'frame_id':'world','position_m':[.65,-.4,1.42],'quaternion_wxyz':[1.,0.,0.,0.]}
    single = SimpleNamespace(arm_base=base,names=['joint'],public_geometry=lambda q:{'wrist':deepcopy(current)})
    policy = SimpleNamespace(geometry=SimpleNamespace(sides={'left':single}))
    row = {'wrist_rotation_in_arm_base':rotation.tolist(),'pinch_center_hand_m':[.07,-.05,.15],
        'handle_contact_height_from_base_origin_m':.041,'open_hand_posture_rad':{'finger':.2},
        'closed_hand_posture_rad':{'finger':.5}}
    phases = motion_phases(policy,'left',row,np.array([.24,-.36,.807,0.,0.,0.,1.]),
        geometry_for('tic_tac_toe_3x3'),PickupLimits(),{'joint':0.})
    assert phases[0]['name'] == 'raise'
    assert [p for p in phases if p['name']=='close'][0]['hand_posture_rad'] == {'finger':.5}
    for phase in phases:
        if phase['name'].startswith('lift'):
            assert set(phase) == {'name','wrist'}
    descend = [p for p in phases if p['name'].startswith('descend')][-1]
    lifted = [p for p in phases if p['name'].startswith('lift')][-1]
    assert lifted['wrist']['position_m'][2]-descend['wrist']['position_m'][2] == pytest.approx(.05)


def test_plan_binds_actual_assets_complete_postures_and_mesh_review(tmp_path):
    asset, review = tmp_path/'robot.urdf', tmp_path/'mesh_review.json'
    asset.write_text('cpu-test fixture; not a robot asset')
    review.write_text('{"scope":"synthetic CPU test"}')
    names = ['left_joint_'+str(i) for i in range(22)]
    hand = SimpleNamespace(names=names, active=[SimpleNamespace(name=n, lower=-1., upper=1.) for n in names])
    single = SimpleNamespace(hand=hand, hand_posture_prior=SimpleNamespace(violation=lambda f,v: None), finger_joints=['thumb','index'])
    policy = SimpleNamespace(geometry=SimpleNamespace(asset_sha256={str(asset): digest(asset)}, sides={'left':single}))
    data = {'schema':'board_game_pickup_plan.v1', 'kinematic_assets_sha256':policy.geometry.asset_sha256,
        'sides':{'left': {'open_hand_posture_rad':dict.fromkeys(names,.2), 'closed_hand_posture_rad':dict.fromkeys(names,.4),
            'pinch_center_hand_m':[.07,-.05,.15], 'wrist_rotation_in_arm_base':np.eye(3).tolist(),
            'handle_contact_height_from_base_origin_m':.041,
            'mesh_review':{'path':str(review),'sha256':digest(review)}}}}
    path = tmp_path/'plan.json'
    def check(value):
        path.write_text(json.dumps(value))
        return checked_plan(path,policy,'left',geometry_for('tic_tac_toe_3x3'))
    assert check(data)[0] == data
    broken = deepcopy(data)
    broken['sides']['left']['open_hand_posture_rad'].pop(names[0])
    with pytest.raises(ValueError,match='all 22'):
        check(broken)
    broken = deepcopy(data)
    broken['sides']['left']['closed_hand_posture_rad'][names[0]] = float('nan')
    with pytest.raises(ValueError,match='URDF'):
        check(broken)
    broken = deepcopy(data)
    broken['sides']['left']['wrist_rotation_in_arm_base'][0][0] = -1.
    with pytest.raises(ValueError,match='rotation'):
        check(broken)
    broken = deepcopy(data)
    broken['sides']['left']['handle_contact_height_from_base_origin_m'] = .050
    with pytest.raises(ValueError,match='end caps'):
        check(broken)
    review.write_text('changed review')
    with pytest.raises(ValueError,match='review'):
        check(data)
    asset.write_text('changed asset')
    with pytest.raises(ValueError,match='hash differs'):
        check(data)


@pytest.mark.parametrize('land_after_lift',[False,True])
def test_synthetic_lift_requires_final_unsupported_dwell_not_historical_peak(tmp_path,land_after_lift):
    """Exercise evidence logic with fake buffers, not real pickup qualification."""
    env = FakeEnv()
    env.mode = 'support'
    def contacts(*,dt):
        env.acquisitions += 1
        assert dt == 1.
        contacts = [(0,[.2,-.5,.8])] if env.mode == 'support' else [
            (env._contact_filter_paths[0].index(PADS[f]),(env.target[:3]+[0.,sign*.02,.027]).tolist())
            for f,sign in (('thumb',1),('index',-1))]
        n = len(contacts)
        counts = np.zeros((len(env._contact_sensor_paths),len(env._contact_filter_paths[0])),dtype=int)
        starts = counts.copy()
        for i,(filt,_) in enumerate(contacts):
            counts[0,filt], starts[0,filt] = 1,i
        return np.full((n,1),.001),np.array([p for _,p in contacts]),np.tile([0.,0.,1.],(n,1)),np.full((n,1),-.0001),counts,starts
    env._rigid_contacts.get_contact_data = contacts
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    for _ in range(10):
        env.advance()
    env.mode, env.target[2] = 'grip', .857
    for _ in range(1000):
        env.advance()
    assert observer.maximum_airborne_hold_s >= .3
    if land_after_lift:
        env.mode, env.target[2] = 'support', .807
        env.advance()
    result = observer.close()
    assert result['passed'] is (not land_after_lift)
    assert result['both_pads_preferred_pickup'] is (not land_after_lift)


def test_inverted_token_with_handle_below_surface_cannot_pass_lift_gate(tmp_path):
    """Synthetic contact buffers expose a geometry gate bug, not native evidence."""
    env = FakeEnv()
    env.mode = 'support'
    def contacts(*,dt):
        env.acquisitions += 1
        assert dt == 1.
        rotation = quaternion_matrix(env.target[[6,3,4,5]])
        contacts = [(0,[.2,-.5,.8])] if env.mode == 'support' else [
            (env._contact_filter_paths[0].index(PADS[f]),
             (env.target[:3]+rotation@np.array([0.,sign*.02,.027])).tolist())
            for f,sign in (('thumb',1),('index',-1))]
        n = len(contacts)
        counts = np.zeros((len(env._contact_sensor_paths),len(env._contact_filter_paths[0])),dtype=int)
        starts = counts.copy()
        for i,(filt,_) in enumerate(contacts):
            counts[0,filt], starts[0,filt] = 1,i
        return np.full((n,1),.001),np.array([p for _,p in contacts]),np.tile([0.,0.,1.],(n,1)),np.full((n,1),-.0001),counts,starts
    env._rigid_contacts.get_contact_data = contacts
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    for _ in range(10):
        env.advance()
    env.mode = 'grip'
    env.target[:] = [.2,-.5,.820,1.,0.,0.,0.]
    for _ in range(1000):
        env.advance()
    assert observer.last['pinch_window_qualified']
    assert observer.last['token_bottom_m'] == pytest.approx(.773)
    assert observer.maximum_airborne_hold_s == 0.
    assert not observer.close()['passed']


@pytest.mark.parametrize('contacted_id',['black_00','white_00'])
def test_target_middle_phalanx_contact_is_not_mislabeled_environment_or_neighbor(tmp_path,contacted_id):
    env = FakeEnv()
    middle = '/World/Robot/Left/left_index_MP'
    env.scene_info['robot_link_paths']['left_index_MP'] = middle
    env._contact_sensor_paths.append(middle)
    filters = [*env._contact_filter_paths[0],middle]
    env._contact_filter_paths = [list(filters) for _ in env._contact_sensor_paths]
    def contacts(*,dt):
        env.acquisitions += 1
        counts = np.zeros((len(env._contact_sensor_paths),len(filters)),dtype=int)
        starts = counts.copy()
        path = env.scene_info['objects'][contacted_id]['path']
        counts[env._contact_sensor_paths.index(middle),filters.index(path)] = 1
        return np.array([[.001]]),np.array([[.2,-.5,.81]]),np.array([[0.,0.,1.]]),np.array([[-.0001]]),counts,starts
    env._rigid_contacts.get_contact_data = contacts
    supports = {'black_00':{'paths':('/World/Fixtures/reserve_black_west',),'surface_z_m':.8,'fixture_id':'reserve_black_west'}}
    observer = PickupObserver(env,tmp_path,'black_00','left',geometry_for('tic_tac_toe_3x3'),PickupLimits(),supports)
    env.advance()
    result = observer.close()
    assert result['no_robot_environment_or_other_token_contact'] is (contacted_id=='black_00')
    assert bool(result['additional_target_robot_contact_pairs']) is (contacted_id=='black_00')
    assert len(result['interference_pairs']) == 1  # Raw contact is always disclosed.
    assert result['passed'] is False
