"""CPU tests of the geometric-support regression rule, the settle verdict and the robot-free filter."""
import pytest

from scenes.workrooms.parameterized import settle_admission as settle

BODIES = ['bnc_plug', 'bnc_socket', 'usb_plug']


def review(missing, bodies=BODIES, criterion='bodies_without_near_support_or_joint'):
    return {'criterion': criterion, 'bodies': list(bodies), 'missing': list(missing), 'collider_count': 9}


@pytest.mark.parametrize('missing', [[], ['bnc_plug'], ['usb_plug'], ['bnc_plug', 'usb_plug']])
def test_full_canonical_support_is_exactly_the_original_rule(missing):
    record = settle.support_admission(review(missing), review([]))
    original_rule_passed = not missing
    assert record['passed'] is original_rule_passed
    assert record['regression'] == sorted(missing) == record['candidate_missing']
    assert record['rule'] == 'G_canonical_support_regression'
    assert record['canonical_seed0_missing'] == [] and record['canonical_seed0_supported'] == BODIES


def test_bodies_unsupported_at_canonical_seed0_are_left_to_the_dynamic_admission():
    cells = ['sponge', 'sponge_cell_0_0_0', 'sponge_cell_0_0_1']
    bodies = cells+['plate']
    canonical = review(cells, bodies, 'bodies_without_geometric_support_chain')
    same = settle.support_admission(review(cells, bodies, 'bodies_without_geometric_support_chain'), canonical)
    assert same['passed'] and same['regression'] == [] and same['canonical_seed0_missing'] == cells
    lost = settle.support_admission(review(cells+['plate'], bodies, 'bodies_without_geometric_support_chain'), canonical)
    assert not lost['passed'] and lost['regression'] == ['plate']


def test_inventories_must_be_comparable():
    with pytest.raises(ValueError, match='not comparable'):
        settle.support_admission(review([]), review([], bodies=['other']))
    with pytest.raises(ValueError, match='not comparable'):
        settle.support_admission(review([]), review([], criterion='bodies_without_geometric_support_chain'))


def trace(objects, **changes):
    value = {'diagnostic_thresholds': dict(settle.RIGID_THRESHOLDS), 'terminal_window_s': .5, 'objects': {}}
    for name, (drift, linear, angular, supported) in objects.items():
        value['objects'][name] = {'window_displacement_from_first_m': drift, 'max_linear_speed_m_s': linear,
                                  'max_angular_speed_rad_s': angular, 'native_supported_or_joint_rooted': supported}
    value.update(changes)
    return value


def row(objects, **changes):
    return {'trace': trace(objects), 'hold_s': .8, 'robot_free': {'mode': 'filtered_for_robot_free_settle_v1'}, **changes}


def test_native_thresholds_are_inclusive_and_never_relaxed():
    at_limit = row({'plate': (.005, .02, .2, True)})
    assert settle.settle_verdict(at_limit)['admitted']
    for values in ((.00501, .0, .0, True), (.0, .02001, .0, True), (.0, .0, .20001, True), (.0, .0, .0, False)):
        verdict = settle.settle_verdict(row({'plate': values}))
        assert not verdict['admitted'] and verdict['unstable_objects'] == ['plate']
    rocking = settle.settle_verdict(row({'spoon': (.00022, .009, 1.31, True), 'plate': (0., 0., 0., True)}))
    assert rocking['unstable_objects'] == ['spoon']
    with pytest.raises(ValueError, match='thresholds'):
        settle.settle_verdict({'trace': trace({}, diagnostic_thresholds={'displacement_m': .01}), 'hold_s': .8})


def test_only_the_frozen_referee_twist_free_centreline_skips_angular():
    objects = {'cable_link_001': (.001, .01, 3.2, True), 'cable_end': (.001, .01, 3.2, True)}
    placement = {'twist_free_ids': ['cable_link_001']}
    verdict = settle.settle_verdict(row(objects, task_specific_placement_stability=placement))
    assert verdict['unstable_objects'] == ['cable_end']
    assert verdict['objects']['cable_link_001']['angular_exempt_by_frozen_referee']
    drifting = settle.settle_verdict(row({'cable_link_001': (.006, .01, 3.2, True)},
                                         task_specific_placement_stability=placement))
    assert drifting['unstable_objects'] == ['cable_link_001']


def test_referee_failure_and_cloth_thresholds_reject():
    failure = settle.settle_verdict(row({'plate': (0., 0., 0., True)},
                                        passive_physical_failure={'reason': 'object_left_recoverable_workspace'}))
    assert not failure['admitted']
    cloth = {'towel': {'p95_node_speed_m_s': .088, 'drift_m': .004}}
    assert settle.settle_verdict(row({}, cloth_settle=cloth))['unstable_cloths'] == ['towel']
    calm = {'towel': {'p95_node_speed_m_s': .08, 'drift_m': .015}}
    assert settle.settle_verdict(row({}, cloth_settle=calm))['admitted']


def test_protocol_hold_durations():
    assert settle.hold_seconds('laundry_room') == 1.6
    assert all(settle.hold_seconds(family) == .8 for family in ('kitchen', 'laboratory', 'mechanical_assembly', 'study'))


def test_robot_free_filter_groups_and_contact_detection():
    pxr = pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend import robot_free
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.Xform.Define(stage, '/World/Robot')
    record = robot_free.filter_robot_collisions(stage)
    robot = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(robot_free.ROBOT_GROUP))
    scene = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(robot_free.SCENE_GROUP))
    assert [str(p) for p in robot.GetFilteredGroupsRel().GetTargets()] == [robot_free.SCENE_GROUP]
    assert [str(p) for p in robot.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()] == ['/World/Robot']
    members = scene.GetCollidersCollectionAPI()
    assert [str(p) for p in members.GetIncludesRel().GetTargets()] == ['/World']
    assert [str(p) for p in members.GetExcludesRel().GetTargets()] == ['/World/Robot']
    query = members.ComputeMembershipQuery()
    assert query.IsPathIncluded('/World/Objects/plate/collision') and not query.IsPathIncluded('/World/Robot/hand/link')
    assert record['mode'] == robot_free.MODE
    with pytest.raises(ValueError):
        robot_free.filter_robot_collisions(stage)
    assert robot_free.robot_contact({'actor0': '/World/Robot/fr3_link7', 'actor1': '/World/Objects/plate'})
    assert not robot_free.robot_contact({'actor0': '/World/RobotStand', 'actor1': '/World/Objects/plate'})
    assert robot_free.checked_mode({}) is None
    assert robot_free.checked_mode({'isaacsim': {'robot_collision_mode': robot_free.MODE}}) == robot_free.MODE
    with pytest.raises(ValueError, match='Unknown robot collision mode'):
        robot_free.checked_mode({'isaacsim': {'robot_collision_mode': 'off'}})


def test_settled_state_binding_in_the_sampler():
    from scenes.workrooms.parameterized.spec import sample_parameters, validate_parameters
    sha = 'a'*64
    plain = sample_parameters('laboratory', 'solid_weighing', 'single_arm', 1, 22)
    bound = sample_parameters('laboratory', 'solid_weighing', 'single_arm', 1, 22, settled_initial_state_sha256=sha)
    assert 'settled_initial_state_sha256' not in plain and bound['settled_initial_state_sha256'] == sha
    assert {k: v for k, v in bound.items() if k not in ('settled_initial_state_sha256', 'instance_sha256')} == {
        k: v for k, v in plain.items() if k != 'instance_sha256'}
    assert validate_parameters(bound) and validate_parameters(plain)
    for invalid in ('A'*64, 'a'*63, 7):
        with pytest.raises(ValueError, match='settled initial state'):
            sample_parameters('laboratory', 'solid_weighing', 'single_arm', 1, 22, settled_initial_state_sha256=invalid)
    with pytest.raises(ValueError, match='settled initial state'):
        sample_parameters('laboratory', 'solid_weighing', 'single_arm', 0, settled_initial_state_sha256=sha)


def settled_case(**changes):
    return {'task_id': 'solid_weighing', 'family': 'laboratory', 'configuration_id': 'single_arm', 'seed': 1,
            'attempt': 22, 'layout_amplitude_scale': 1., 'hold_s': 2., 'inputs': {'path': '/x', 'sha256': '0'*64},
            'asset': {'path': '/a', 'sha256': '1'*64}, **changes}


def test_settled_state_record_round_trip(tmp_path):
    import hashlib, json
    names = ['pellet_00', 'dish']
    paths = {'pellet_00': '/World/Task/pellet_00', 'dish': '/World/Task/dish'}
    q = [[.1, .2, .9, 0., 0., 0., 1.], [.3, .1, .85, 0., 0., .7071068, .7071068]]
    record = settle.settled_state_record(settled_case(), {'path': '/held', 'sha256': '2'*64}, names, q, paths,
                                         {'admitted': True})
    assert record['body_poses']['dish']['quaternion_wxyz'] == pytest.approx([.7071068, 0., 0., .7071068])
    assert record['body_poses']['pellet_00']['position_m'] == [.1, .2, .9] and record['settled'] is True
    path = tmp_path/'settled_state.json'; path.write_text(json.dumps(record))
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert settle.checked_settled_state(ref, task_id='solid_weighing', seed=1, attempt=22, layout_amplitude_scale=1.) == record
    for changes in ({'attempt': 23}, {'seed': 2}, {'layout_amplitude_scale': .5}):
        identity = {'task_id': 'solid_weighing', 'seed': 1, 'attempt': 22, 'layout_amplitude_scale': 1., **changes}
        with pytest.raises(ValueError, match='does not belong'):
            settle.checked_settled_state(ref, **identity)
    with pytest.raises(ValueError, match='cable and granular'):
        settle.settled_state_record(settled_case(task_id='table_clearing'), {}, names, q, paths, {'admitted': True})
    unsettled = settle.settled_state_record(settled_case(), {}, names, q, paths, {'admitted': False})
    path.write_text(json.dumps(unsettled))
    with pytest.raises(ValueError, match='does not belong'):
        settle.checked_settled_state({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                                     task_id='solid_weighing', seed=1, attempt=22, layout_amplitude_scale=1.)


def test_settled_poses_replace_only_the_rigid_world_pose():
    pytest.importorskip('pxr')
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from scenes.workrooms.parameterized.usd_authoring import apply_settled_poses, world_pose
    stage = Usd.Stage.CreateInMemory()
    group = UsdGeom.Xform.Define(stage, '/World/Task')
    group.AddTranslateOp().Set(Gf.Vec3d(.5, 0., .1))
    body = UsdGeom.Xform.Define(stage, '/World/Task/pellet')
    body.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3)); body.AddScaleOp().Set(Gf.Vec3f(2., 2., 2.))
    target = {'pellet': {'prim_path': '/World/Task/pellet', 'position_m': [.7, .1, .45],
                         'quaternion_wxyz': [.7071067811865476, 0., 0., .7071067811865476]}}
    receipts = apply_settled_poses(stage, {'pellet': '/World/Task/pellet'}, target)
    after = world_pose(stage, '/World/Task/pellet')
    assert np.allclose(after['position_m'], [.7, .1, .45]) and receipts['pellet']['before']['position_m'] == pytest.approx([.6, .2, .4])
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Task/pellet'))
    assert np.allclose([Gf.Vec3d(*matrix.GetRow3(i)).GetLength() for i in range(3)], [2., 2., 2.])
    with pytest.raises(ValueError, match='exactly the dynamic actors'):
        apply_settled_poses(stage, {'pellet': '/World/Task/pellet', 'other': '/World/Task/other'}, target)


def laundry_trace(raw, pose, drift, **changes):
    value = {'passed': True, 'terminal_contact_geometry_acceptable': True, 'terminal_window_s': .5,
             'speed_policy': 'native_qd_and_adjacent_positions',
             'original_thresholds': {'max_node_speed_m_s': .08, 'stable_node_drift_m': .015, 'speed_node_percentile': 95.},
             'cloths': {'towel': {'max_raw_speed_percentile_m_s': raw, 'max_pose_speed_percentile_m_s': pose,
                                  'max_interval_speed_percentile_m_s': pose, 'max_node_drift_m': drift,
                                  'geometric_support_present': True}}}
    value.update(changes)
    return value


def test_laundry_cloth_admission_uses_the_native_speed_policy_and_unchanged_thresholds():
    quiet = settle.laundry_settle_verdict({'hold_s': 1.6, 'laundry_trace': laundry_trace(.04, .049, .0005)})
    assert quiet['admitted'] and quiet['cloths']['towel']['p95_node_speed_m_s'] == .049
    # Gallery v2 towel s4 at 0.8 s: pose P95 0.0877 m/s exceeds 0.08 m/s.
    moving = settle.laundry_settle_verdict({'hold_s': 1.6, 'laundry_trace': laundry_trace(.0743, .0877, .0078,
                                                                                            passed=False)})
    assert not moving['admitted'] and moving['unstable_cloths'] == ['towel']
    drifting = settle.laundry_settle_verdict({'hold_s': 1.6, 'laundry_trace': laundry_trace(.01, .01, .0151)})
    assert not drifting['admitted']
    with pytest.raises(ValueError, match='thresholds differ'):
        settle.laundry_settle_verdict({'hold_s': 1.6, 'laundry_trace': laundry_trace(
            .01, .01, .001, original_thresholds={'max_node_speed_m_s': .1, 'stable_node_drift_m': .015,
                                                 'speed_node_percentile': 95.})})
    failed = settle.laundry_settle_verdict({'hold_s': 1.6, 'laundry_failure': {'message': 'referee rejected'}})
    assert not failed['admitted'] and failed['laundry_failure']


def test_broken_joint_chain_is_a_recorded_passive_failure_and_other_errors_raise():
    def chain(*_):
        raise ValueError(settle.CHAIN_INTEGRITY_FAILURE)

    def observer(*_):
        raise ValueError('Original mechanical bundle observer is incomplete, changed or invalid')

    failure = settle.passive_observation_failure(chain, {}, {}, {})
    assert failure['reason'] == 'joint_chain_integrity_or_length' and failure['status'] == 'task_failure'
    with pytest.raises(ValueError, match='observer is incomplete'):
        settle.passive_observation_failure(observer, {}, {}, {})
    assert settle.passive_observation_failure(lambda *_: None, {}, {}, {}) is None
    calm = row({'belt_link_000': (0., 0., 0., True)})
    assert settle.settle_verdict(calm)['admitted']
    broken = settle.settle_verdict(row({'belt_link_000': (0., 0., 0., True)},
                                       passive_physical_failure={'control_index': 3, **failure}))
    assert not broken['admitted'] and broken['passive_physical_failure']['reason'] == 'joint_chain_integrity_or_length'


def test_placement_review_is_skipped_only_after_a_recorded_broken_chain():
    chain = settle.passive_observation_failure(lambda *_: (_ for _ in ()).throw(ValueError(settle.CHAIN_INTEGRITY_FAILURE)), {}, {}, {})
    assert not settle.placement_review_applicable({'passive_physical_failure': {'control_index': 2, **chain}})
    assert settle.placement_review_applicable({'passive_physical_failure': {'control_index': 2, 'reason': 'object_fell'}})
    assert settle.placement_review_applicable({})
    first_stop = {'passive_physical_failure': {'control_index': 7, 'reason': 'physics_safety_stop'},
                  'joint_chain_failure': {'control_index': 9, **chain}}
    assert not settle.placement_review_applicable(first_stop)
