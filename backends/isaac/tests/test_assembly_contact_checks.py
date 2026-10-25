from copy import deepcopy
import math
import pytest
from scripts.validation.validate_assembly_contacts import assess_case


def fixture():
    path = '/World/Objects/part'
    support = '/World/Fixtures/seat'
    rows = [{'native_step': i+1, 'time_s': (i+1)*.005, 'poses_xyzw': {'part': [0., 0., .1, 0., 0., 0., 1.]},
             'velocities': {'part': [0.]*6}, 'robot_interference': [], 'maximum_penetration_m': 0.,
             'raw_contacts': [{'actor0': path, 'actor1': support+'/collision_000',
                               'normal_impulses_ns': [.001], 'normals': [[0., 0., 1.]],
                               'impulse_magnitude_sum_ns': .001}]} for i in range(200)]
    case = {'duration_s': 1., 'assertions': {'part': {'support_path_prefix': support, 'position_m': [0., 0., .1],
                                   'position_tolerance_m': [.001]*3}}}
    return rows, case, {'objects': {'part': {'path': path}}}


def descent(rows):
    for index in range(20):
        rows[index]['poses_xyzw']['part'][2]+=.02*(20-index)/20


def test_settled_pose_without_actual_support_cannot_pass():
    rows, case, scene = fixture()
    assert assess_case(rows, case, scene, .005)['passed']
    for row in rows:
        row['raw_contacts'] = []
    assert not assess_case(rows, case, scene, .005)['passed']


def test_wrong_seat_missing_samples_and_real_interference_fail():
    rows, case, scene = fixture()
    wrong = deepcopy(rows)
    for row in wrong:
        row['raw_contacts'][0]['actor1'] = '/World/OtherSeat'
    assert not assess_case(wrong, case, scene, .005)['passed']
    assert not assess_case(rows[1:], case, scene, .005)['passed']
    rows[3]['robot_interference'] = [['hand', 'part']]
    assert not assess_case(rows, case, scene, .005)['passed']


def test_contact_and_stability_do_not_substitute_for_required_insertion():
    rows, case, scene = fixture()
    case['assertions']['part']['minimum_descent_m'] = .01
    assert not assess_case(rows, case, scene, .005)['passed']


def test_lateral_contact_cannot_substitute_for_upward_support():
    rows, case, scene = fixture()
    for row in rows:
        row['raw_contacts'][0]['normals'] = [[1., 0., 0.]]
    assert not assess_case(rows, case, scene, .005)['passed']


def test_signed_native_scalars_preserve_actor_direction_for_support():
    rows, case, scene = fixture()
    for row in rows:
        contact=row['raw_contacts'][0]
        contact['actor0'],contact['actor1']=contact['actor1'],contact['actor0']
        contact['normal_impulses_ns']=[-.001]
    assert assess_case(rows, case, scene, .005)['passed']
    for row in rows:row['raw_contacts'][0]['normal_impulses_ns']=[.001]
    assert not assess_case(rows, case, scene, .005)['passed']


def test_opposing_point_loads_require_net_upward_support():
    rows, case, scene = fixture()
    for row in rows:
        contact=row['raw_contacts'][0]
        contact.update(normal_impulses_ns=[.001,-.002],normals=[[0.,0.,1.]]*2,
                       impulse_magnitude_sum_ns=.003)
    assert not assess_case(rows, case, scene, .005)['passed']


def test_signed_scalar_radial_vector_establishes_real_guidance():
    rows, case, scene = fixture()
    case['assertions']['part'].update(require_guidance_contact=True,
                                     guidance_path_prefix='/World/Fixtures/guide')
    descent(rows)
    rows[0]['raw_contacts']=[{'actor0':'/World/Objects/part',
        'actor1':'/World/Fixtures/guide/collision_000',
        'normal_impulses_ns':[-.001],'normals':[[1.,0.,0.]],
        'impulse_magnitude_sum_ns':.001}]
    assert assess_case(rows, case, scene, .005)['passed']
    rows[0]['raw_contacts'][0]['normal_impulses_ns']=[0.]
    assert not assess_case(rows, case, scene, .005)['passed']


def test_negative_control_requires_real_settling_and_wrong_pose():
    rows, case, scene = fixture()
    case['expected_functional_success'] = False
    assert not assess_case(rows, case, scene, .005)['passed']
    case['assertions']['part']['position_m'][2] = .08
    descent(rows)
    assert assess_case(rows, case, scene, .005)['passed']
    for row in rows:
        row['raw_contacts'] = []
    assert not assess_case(rows, case, scene, .005)['passed']


@pytest.mark.parametrize('negative', [False, True])
def test_slow_yaw_drift_exceeds_orientation_range_even_below_velocity_limit(negative):
    rows, case, scene = fixture()
    if negative:
        case['expected_functional_success'] = False
        case['assertions']['part']['position_m'][2] = .08
        descent(rows)
    for i,row in enumerate(rows[100:],1):
        yaw = i*.0002  # .04 rad/s but .02 rad accumulated during the stable interval.
        row['poses_xyzw']['part'][3:] = [0.,0.,math.sin(yaw/2),math.cos(yaw/2)]
        row['velocities']['part'][5] = .04
    result = assess_case(rows, case, scene, .005)
    assert not result['passed']
    assert result['objects']['part']['checks']['angular_rest']
    assert not result['objects']['part']['checks']['stable_orientation']


@pytest.mark.parametrize('change', ['duration','missing_time','clock_jump','step_gap','prefix_missing','time_shift'])
def test_complete_stable_tail_does_not_hide_invalid_full_run_receipt(change):
    rows, case, scene = fixture()
    if change=='duration':case['duration_s']=2.
    elif change=='missing_time':rows[3].pop('time_s')
    elif change=='clock_jump':rows[3]['time_s']+=.001
    elif change=='step_gap':rows[3]['native_step']+=1
    elif change=='prefix_missing':rows=rows[1:]
    else:
        for row in rows:row['time_s']+=.02
    result = assess_case(rows,case,scene,.005)
    assert not result['passed']
    assert not result['checks']['complete_native_sampling']


@pytest.mark.parametrize('kind', ['native_linear','native_angular','pose_linear','pose_angular'])
def test_settled_end_does_not_erase_earlier_motion_safety_failure(kind):
    rows,case,scene=fixture()
    if kind=='native_linear':rows[3]['velocities']['part'][0]=1.01
    elif kind=='native_angular':rows[3]['velocities']['part'][3]=20.1
    elif kind=='pose_linear':rows[7]['poses_xyzw']['part'][0]=.021
    else:rows[7]['poses_xyzw']['part'][3:]=[0.,0.,math.sin(.41/2),math.cos(.41/2)]
    result=assess_case(rows,case,scene,.005)
    assert result['objects']['part']['passed']
    assert not result['checks']['object_motion_safety']
    assert not result['passed']


def test_opposite_quaternion_signs_are_the_same_stable_orientation():
    rows,case,scene=fixture()
    for row in rows[::2]:row['poses_xyzw']['part'][6]=-1.
    assert assess_case(rows,case,scene,.005)['passed']


def test_stable_interval_includes_its_initial_pose_and_velocity():
    rows,case,scene=fixture()
    rows[99]['velocities']['part'][3]=.051
    assert not assess_case(rows,case,scene,.005)['objects']['part']['checks']['angular_rest']


@pytest.mark.parametrize('scale', [0., .5, 2.])
def test_nonunit_native_quaternion_is_not_repaired_by_normalization(scale):
    rows,case,scene=fixture()
    rows[4]['poses_xyzw']['part'][6]=scale
    result=assess_case(rows,case,scene,.005)
    assert not result['passed']
    assert not result['checks']['finite_native_state']


def test_native_step_pose_difference_is_diagnostic_not_a_new_20ms_rule():
    rows,case,scene=fixture()
    rows[104]['poses_xyzw']['part'][0]=.0001
    result=assess_case(rows,case,scene,.005)
    assert result['passed']
    maxima=result['motion_evidence']['objects']['part']['whole_run_maxima']
    assert maxima['native_step_sampled_linear']==pytest.approx(.02)
    assert maxima['sampled_linear']==0.
