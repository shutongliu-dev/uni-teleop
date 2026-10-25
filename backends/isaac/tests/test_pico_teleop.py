"""Input freshness, clutch/calibration and actual pinned robot kinematics."""
from dataclasses import replace
import json
import socket
import time
import threading

import numpy as np
import pytest

from uni_teleop.devices.pico.protocol import Frame, Receiver, SCHEMA, boot_id, normalize_device
from uni_teleop.mapping.franka.retarget import Retargeter, HandCalibration, FINGERS
from bench.control.factory import public_robot_geometry, robot_configuration
from scenes.workrooms.catalog import load_task


@pytest.fixture(scope='module')
def robot():
    loaded = load_task('kitchen', 'breakfast_serving')
    return public_robot_geometry(robot_configuration(loaded['scene'])), loaded['initial_state']['joint_positions_rad']


def test_sandbox_loads_without_a_task_and_supports_synthetic_hand_input():
    from scripts.teleop.pico_isaac import load_inputs, parser
    from scripts.validation.pico_teleop_smoke import SyntheticReceiver
    args = parser().parse_args(['--sandbox', '--family', 'unused', '--task', 'unused'])
    scene, initial, release = load_inputs(args)
    assert release is None and 'native_task_bundle' not in scene and 'workroom' not in scene
    assert [o['id'] for o in scene['objects']] == ['bottle', 'can']
    assert scene['physics']['control_dt_s'] == .02
    receiver = SyntheticReceiver(args)
    try:
        packet = receiver.read()
        assert len(packet.tracking['hands']['right']) == 26
    finally:
        receiver.close()


def tracked_hand(model, q):
    geometry = model.public_finger_geometry(q)
    hand = np.zeros((26, 7))
    hand[:, 6] = 1.
    for finger, indices in FINGERS.items():
        centers = geometry['finger_joint_centers_m'][finger]
        points = list(centers.values())+[geometry['fingertip_positions_m'][finger]]
        hand[list(indices), :3] = points
    return hand.tolist()


def frame(mode='controllers', *, grip=1., trigger=.4, x=0., joints=None, sequence=0):
    tracking = {'head': [0., 0., 0., 0., 0., 0., 1.], 'hands': {}, 'controllers': {}}
    tracking[mode]['right'] = joints if mode == 'hands' else {
        'pose': [x, 0., 0., 0., 0., 0., 1.], 'grip': grip, 'trigger': trigger}
    return Frame('session', sequence, time.monotonic_ns(), str(sequence+1), 'pico', tracking)


def tick(mapper, packet, q):
    return mapper.command(packet, q, now_ns=time.monotonic_ns())


def test_controller_clutch_and_explicit_ack(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    assert tick(mapper, frame(), q) == q
    mapper.set_active(True)
    assert tick(mapper, frame(), q) == q  # Calibrate without jumping.
    moved = tick(mapper, frame(x=.005), q)
    assert moved != q
    assert mapper.acknowledged == q  # A candidate is not an applied command.
    assert all(abs(moved[n]-q[n]) <= mapper.velocity[n]*.02+1e-12 for n in q)
    mapper.acknowledge(moved)
    assert tick(mapper, frame(grip=0., x=1.), moved) == moved
    assert tick(mapper, frame(grip=1., x=1.), moved) == moved  # Re-clutch reanchors.


def test_tracking_loss_and_source_restart_require_rearming(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    packet = frame()
    tick(mapper, packet, q)
    assert tick(mapper, replace(packet, host_ns=packet.host_ns-1_000_000_000), q) == q
    assert not mapper.armed
    tick(mapper, frame(), q)
    assert not mapper.armed
    mapper.set_active(True)
    tick(mapper, frame(), q)
    tick(mapper, replace(frame(), stream='reconnected'), q)
    assert not mapper.armed


def test_controller_recovery_requires_fresh_release_then_press_and_reanchors(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    tick(mapper, frame(), q)
    tick(mapper, None, q)
    assert tick(mapper, frame(x=.2), q) == q
    assert not mapper.armed  # Holding through a loss never resumes.
    tick(mapper, frame(grip=0., x=.2), q)
    tick(mapper, None, q)  # Another loss invalidates the observed release.
    tick(mapper, frame(x=.2), q)
    assert not mapper.armed
    tick(mapper, frame(grip=0., x=.2), q)
    assert tick(mapper, frame(x=.2), q) == q
    assert mapper.armed  # Explicit re-grip anchors here, not at the old pose.
    assert tick(mapper, frame(x=.205), q) != q


@pytest.mark.parametrize('interrupt', ['pause', 'mode', 'stream'])
def test_recovery_grip_cannot_override_deliberate_pause_or_new_source(robot, interrupt):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    tick(mapper, frame(), q)
    tick(mapper, None, q)
    if interrupt == 'pause':
        mapper.set_active(False)
    elif interrupt == 'mode':
        mapper.set_mode('controllers')
    stream = 'new' if interrupt == 'stream' else 'session'
    tick(mapper, replace(frame(grip=0.), stream=stream), q)
    assert tick(mapper, replace(frame(x=.2), stream=stream), q) == q
    assert not mapper.armed


def test_hand_calibration_matches_robot_and_moves_fingers(robot):
    model, q = robot
    joints = tracked_hand(model, q)
    calibration = HandCalibration(model, q, joints)
    tips = model.public_geometry(q)['fingertip_positions_m']
    for finger, target in calibration.targets(joints).items():
        assert np.allclose(target, tips[finger])
    mapper = Retargeter(model, q, {'right': 'right'})
    mapper.set_active(True)
    assert tick(mapper, frame('hands', joints=joints), q) == q
    changed = np.array(joints)
    changed[10, 0] += .002
    commanded = tick(mapper, frame('hands', joints=changed.tolist()), q)
    index = [joint.name for joint in model.finger_joints['index']]
    assert any(abs(commanded[n]-q[n]) > 1e-6 for n in index)
    for n in q:
        assert model.by_name[n].lower <= commanded[n] <= model.by_name[n].upper


def test_inactive_hand_disarms_instead_of_replaying(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'})
    mapper.set_active(True)
    assert tick(mapper, frame('controllers'), q) == q
    assert not mapper.armed
    assert mapper.status == 'right hands unavailable — check PICO tracking'
    tick(mapper, frame('hands', joints=tracked_hand(model, q)), q)
    assert not mapper.armed and mapper.status == 'paused — press Start'


def test_mode_change_pauses_and_clears_anchors(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    tick(mapper, frame(), q)
    mapper.set_mode('hands')
    assert not mapper.armed and not mapper.anchors


def test_missing_controller_does_not_hide_active_bare_hand(robot):
    model, q = robot
    value = {'Controller': {'right': {'pose': '0,0,0,0,0,0,0', 'trigger': 0., 'grip': 0.}},
             'Hand': {'rightHand': {'isActive': 1, 'HandJointLocations': [
                 {'p': ','.join(map(str, p))} for p in tracked_hand(model, q)]}}}
    result = normalize_device(value)
    assert not result['controllers'] and 'right' in result['hands']


def test_inactive_app_and_invalid_hand_joints_are_not_control_inputs(robot):
    model, q = robot
    value = packet()['value']
    value['appState'] = {'focus': False}
    assert not normalize_device(value)['controllers']
    assert normalize_device(value)['hand_status']['right'] == 'PICO app not focused'
    joints = [{'p': p, 's': 15} for p in tracked_hand(model, q)]
    joints[10]['s'] = 0
    value = {'Hand': {'rightHand': {'isActive': 1, 'HandJointLocations': joints}}}
    assert normalize_device(value)['hands']['right'][10] is None
    assert normalize_device(value)['hand_status']['right'] == 'invalid joint tracking: 10'
    joints[1]['s'] = 0
    assert not normalize_device(value)['hands']  # Wrist loss still holds the whole arm.


@pytest.mark.parametrize('target', ['wrist', 'finger'])
def test_expired_ik_budget_stops_before_search(robot, monkeypatch, target):
    model, q = robot
    geometry = model.public_geometry(q)
    request = ({'wrist': {'frame_id': 'world', **geometry['wrist']}} if target == 'wrist'
               else {'fingertip_positions_m': {'index': geometry['fingertip_positions_m']['index']}})
    def must_not_search(*args, **kwargs):
        raise AssertionError('Expired IK started a forward-kinematics search')
    monkeypatch.setattr(model.arm if target == 'wrist' else model.hand, 'forward', must_not_search)
    with pytest.raises(TimeoutError, match='time budget exhausted'):
        model.solve(q, **request, deadline_ns=time.monotonic_ns()-1)


def test_slow_hand_ik_holds_targets_without_disarming_fresh_tracking(robot, monkeypatch):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='hands')
    mapper.set_active(True)
    packet = frame('hands', joints=tracked_hand(model, q))
    tick(mapper, packet, q)
    def timeout(*args, **kwargs):
        assert kwargs['deadline_ns'] is not None
        raise TimeoutError('IK time budget exhausted')
    monkeypatch.setattr(model, 'solve', timeout)
    assert tick(mapper, replace(packet, host_ns=time.monotonic_ns()), q) == q
    assert mapper.armed and 'IK budget exhausted' in mapper.status
    tick(mapper, None, q)
    assert not mapper.armed  # Actual loss still disarms.


def packet(stamp=None, sequence=1):
    return {'schema': SCHEMA, 'boot_id': boot_id(), 'stream': 'one', 'sequence': sequence,
            'host_ns': stamp or time.monotonic_ns(), 'device': 'pico',
            'value': {'timeStampNs': sequence, 'Controller': {'right': {
                'pose': '0,0,0,0,0,0,1', 'trigger': .2, 'grip': 1.}}}}


@pytest.mark.parametrize('change', [
    {'boot_id': 'other-host'}, {'host_ns': 1}, {'host_ns': 2**63}, {'sequence': -1}, {'schema': 'bad'},
    {'device': ''}, {'stream': []}])
def test_packet_boundary_rejects_invalid_clock_and_identity(change):
    raw = packet() | change
    with pytest.raises(ValueError):
        Frame.decode(json.dumps(raw), clock=boot_id(), now_ns=time.monotonic_ns(), max_age_ns=150_000_000)


def wait_for(predicate):
    deadline = time.monotonic()+1.
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.001)
    raise AssertionError('Receiver did not process the packet')


def test_udp_drops_old_frames_and_never_refreshes_duplicate_timestamp():
    receiver = Receiver(port=0)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        address = receiver.socket.getsockname()
        sender.sendto(json.dumps(packet(sequence=2)).encode(), address)
        first = wait_for(receiver.read)
        assert first.sequence == 2
        sender.sendto(json.dumps(packet(sequence=1)).encode(), address)
        wait_for(lambda: receiver.rejected == 1)
        assert receiver.read() == first
        receiver.latest = replace(first, host_ns=1)
        assert receiver.read() is None
    receiver.close()


def test_udp_malformed_tracking_does_not_crash_or_refresh_latest():
    receiver = Receiver(port=0)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for count, value in enumerate(({'timeStampNs': 1, 'Controller': []}, {'timeStampNs': 'broken'}), 1):
            sender.sendto(json.dumps(packet() | {'value': value}).encode(), receiver.socket.getsockname())
            wait_for(lambda: receiver.rejected == count)
            assert receiver.read() is None
        assert receiver.rejected == 2
    receiver.close()


def test_frame_arriving_during_poll_is_not_mistaken_for_future(monkeypatch):
    receiver = Receiver.__new__(Receiver)
    receiver.clock, receiver.max_age_ns = boot_id(), 150_000_000
    receiver.latest, receiver.rejected = None, 0
    receiver.received, receiver.last_rejection = 0, None
    receiver._stop = threading.Event()
    clock = [100]
    raw = json.dumps(packet(stamp=150)).encode()
    class ArrivingSocket:
        def recv(self, size):
            clock[0] = 200
            receiver._stop.set()
            return raw
    receiver.socket = ArrivingSocket()
    monkeypatch.setattr('uni_teleop.devices.pico.protocol.time.monotonic_ns', lambda: clock[0])
    receiver._receive()
    assert receiver.read().host_ns == 150


def test_continuous_receiver_keeps_fresh_input_while_control_loop_is_busy():
    receiver = Receiver(port=0)
    # A small buffer and full hand-sized packets reproduce the old overflow.
    receiver.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
    finished = threading.Event()
    sent = []
    def produce():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sequence = 0
            while not finished.is_set():
                sequence += 1
                payload = packet(sequence=sequence) | {'padding': 'x'*14000}
                sender.sendto(json.dumps(payload).encode(), receiver.socket.getsockname())
                sent.append(sequence)
                time.sleep(.005)
    producer = threading.Thread(target=produce)
    producer.start()
    try:
        wait_for(receiver.read)
        time.sleep(.4)  # No consumer polling during a native-sized control step.
        latest = receiver.read()
        assert latest is not None and latest.sequence >= sent[-1]-3
        finished.set()
        producer.join()
        time.sleep(.2)
        assert receiver.read() is None  # Stopping the source still expires input.
    finally:
        finished.set()
        producer.join()
        receiver.close()
    assert not receiver._thread.is_alive()


def test_sdk_disconnect_survives_latest_frame_replacement():
    import ctypes
    from unittest.mock import patch
    from uni_teleop.devices.pico.source import SDK, DeviceState, STATE, DISCONNECTED
    with patch('uni_teleop.devices.pico.source.C.CDLL'):
        sdk = SDK('unused-sdk.so')
    state = DeviceState(b'pico', b'{"value":"{}"}')
    sdk._receive(None, STATE, 0, ctypes.pointer(state))
    sdk._receive(None, DISCONNECTED[0], 0, None)
    sdk._receive(None, STATE, 0, ctypes.pointer(state))
    generation, stamp, device, raw = sdk.pending.get_nowait()
    assert generation == 1 and stamp > 0 and device == b'pico'
    assert json.loads(raw) == {'value': '{}'}
    assert sdk.dropped == 1


def test_named_instance_maps_only_selected_robot():
    loaded = load_task('living_room', 'tower_extraction')
    geometry = public_robot_geometry(robot_configuration(loaded['scene']))
    q = loaded['initial_state']['joint_positions_rad']
    mapper = Retargeter(geometry, q, {'front': 'right'}, mode='controllers')
    mapper.set_active(True)
    tick(mapper, frame(), q)
    moved = tick(mapper, frame(trigger=.5), q)
    assert any(moved[n] != q[n] for n in geometry.sides['front'].hand.names)
    assert all(moved[n] == q[n] for n in geometry.sides['rear'].names)


@pytest.mark.parametrize('latest_kind', ['fresh', 'lost', 'reconnected', 'released'])
def test_expired_solution_holds_without_confusing_slow_ik_with_tracking_loss(robot, latest_kind):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    tick(mapper, frame(), q)
    # An IK candidate is not yet acknowledged and must never be sent when old.
    candidate = tick(mapper, frame(x=.005), q)
    assert candidate != q
    latest = frame(grip=0. if latest_kind == 'released' else 1.)
    if latest_kind == 'lost':
        latest = None
    elif latest_kind == 'reconnected':
        latest = replace(latest, stream='new')
    assert mapper.hold(latest, now_ns=time.monotonic_ns()) == q
    assert mapper.armed == (latest_kind in ('fresh', 'released'))
    if latest_kind == 'released':
        assert not mapper.anchors
    if latest_kind == 'fresh':
        mapper.set_active(False)
        mapper.hold(latest, now_ns=time.monotonic_ns())
        assert not mapper.armed  # A fresh packet cannot restart paused control.


@pytest.mark.parametrize('mode', ['hands', 'controllers'])
@pytest.mark.parametrize('missing', [False, True])
def test_short_gap_holds_and_reanchors_without_applying_missing_motion(robot, mode, missing):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode=mode, tracking_grace_s=.3)
    mapper.set_active(True)
    packet = frame(mode, joints=tracked_hand(model, q))
    start = packet.host_ns
    assert mapper.command(packet, q, now_ns=start) == q
    invalid = replace(packet, host_ns=start+160_000_000, tracking={**packet.tracking, mode: {}}) if missing else None
    assert mapper.command(invalid, q, now_ns=start+160_000_000) == q
    assert mapper.armed and mapper.interrupted and not mapper.anchors
    recovered = frame(mode, x=.5, joints=tracked_hand(model, q))
    if mode == 'hands':
        hand = np.array(recovered.tracking['hands']['right'])
        hand[:, 0] += .5
        recovered.tracking['hands']['right'] = hand.tolist()
    recovered = replace(recovered, host_ns=start+240_000_000)
    assert mapper.command(recovered, q, now_ns=recovered.host_ns) == q
    assert mapper.armed and not mapper.interrupted and mapper.anchors
    # A stationary recovered input must not replay the half-metre displacement.
    target = mapper.command(recovered, q, now_ns=recovered.host_ns+10_000_000)
    stationary = model.arm.names if mode == 'controllers' else q
    # Controller fingers still obey the trigger; the wrist must stay anchored.
    assert all(abs(target[n]-q[n]) < 1e-6 for n in stationary)


@pytest.mark.parametrize('mode', ['hands', 'controllers'])
def test_short_gap_window_never_refreshes_itself_or_accepts_a_late_recovery(robot, mode):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode=mode, tracking_grace_s=.3)
    packet = frame(mode, joints=tracked_hand(model, q))
    mapper.set_active(True)
    mapper.command(packet, q, now_ns=packet.host_ns)
    mapper.command(None, q, now_ns=packet.host_ns+160_000_000)
    mapper.command(None, q, now_ns=packet.host_ns+290_000_000)
    late = replace(packet, host_ns=packet.host_ns+310_000_000, sequence=1)
    assert mapper.command(late, q, now_ns=late.host_ns) == q
    assert not mapper.armed
    mapper.command(replace(late, sequence=2), q, now_ns=late.host_ns+1)
    assert not mapper.armed


@pytest.mark.parametrize('cause', ['pause', 'mode', 'source', 'focus'])
def test_grace_never_resumes_after_an_explicit_stop_or_source_change(robot, cause):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers', tracking_grace_s=.3)
    packet = frame('controllers')
    mapper.set_active(True)
    mapper.command(packet, q, now_ns=packet.host_ns)
    mapper.command(None, q, now_ns=packet.host_ns+160_000_000)
    recovered = replace(packet, host_ns=packet.host_ns+200_000_000, sequence=1)
    if cause == 'pause':
        mapper.set_active(False)
    elif cause == 'mode':
        mapper.set_mode('controllers')
    elif cause == 'source':
        recovered = replace(recovered, stream='new')
    else:
        recovered = replace(recovered, tracking={**recovered.tracking, 'focused': False})
    assert mapper.command(recovered, q, now_ns=recovered.host_ns) == q
    assert not mapper.armed
    focused = replace(recovered, tracking={**recovered.tracking, 'focused': True})
    assert mapper.command(focused, q, now_ns=focused.host_ns+1) == q
    assert not mapper.armed


def test_grace_holds_expired_packets_without_extending_the_freshness_limit(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers', tracking_grace_s=.3)
    packet = frame('controllers')
    mapper.set_active(True)
    mapper.command(packet, q, now_ns=packet.host_ns)
    stale = replace(frame('controllers', x=.5), host_ns=packet.host_ns)
    assert mapper.command(stale, q, now_ns=packet.host_ns+200_000_000) == q
    assert mapper.interrupted and not mapper.anchors


def test_receiver_logs_each_expired_frame_only_once(capsys):
    receiver = Receiver(port=0)
    try:
        receiver.latest = replace(frame(), host_ns=time.monotonic_ns()-1_000_000_000)
        assert receiver.read() is None
        assert receiver.read() is None
        assert capsys.readouterr().out.count('tracking_expired') == 1
        receiver.latest = replace(receiver.latest, sequence=1)
        assert receiver.read() is None
        assert capsys.readouterr().out.count('tracking_expired') == 1
    finally:
        receiver.close()


@pytest.mark.parametrize('finger', FINGERS)
def test_incremental_finger_tracking_reduces_error_without_moving_other_chains(robot, finger):
    from uni_teleop.mapping.franka.retarget import finger_step
    model, q = robot
    goal = dict(q)
    for j in model.finger_joints[finger]:
        if j.name.endswith(('MCP_FE', 'PIP', 'DIP', 'IP', 'CMC_FE')):
            goal[j.name] = float(np.clip(q[j.name]+.25, j.lower, j.upper))
    target = np.array(model.public_geometry(goal)['fingertip_positions_m'][finger])
    initial = np.linalg.norm(np.array(model.public_geometry(q)['fingertip_positions_m'][finger])-target)
    solved, error = finger_step(model, q, finger, target, deadline_ns=time.monotonic_ns()+1_000_000_000)
    assert set(solved) == {j.name for j in model.finger_joints[finger]}
    assert error < initial*.9
    assert all(model.by_name[n].lower <= v <= model.by_name[n].upper for n, v in solved.items())
    assert model.hand_posture_prior.violation(finger, q | solved) is None


def test_unreachable_tip_makes_bounded_progress_instead_of_freezing(robot):
    from uni_teleop.mapping.franka.retarget import finger_step
    model, q = robot
    target = np.array(model.public_geometry(q)['fingertip_positions_m']['index']) + [.3, 0., 0.]
    solved, error = finger_step(model, q, 'index', target, deadline_ns=time.monotonic_ns()+1_000_000_000)
    assert .0015 < error < .3  # Approximate improvement, never presented as exact IK.
    assert any(abs(solved[n]-q[n]) > .001 for n in solved)
    held, _ = finger_step(model, q, 'index', target, deadline_ns=time.monotonic_ns()-1)
    assert all(held[n] == q[n] for n in held)


def test_stereo_wrist_uses_measured_pose_not_stale_usd_and_ignores_head():
    from types import SimpleNamespace
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom
    from uni_teleop.video.isaac import StereoView
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, '/World/WristCamera')
    poses = {'right_hand_flange': {'position_m': [1., 2., 3.], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    paths = {'wrist': '/World/WristCamera'}
    env = SimpleNamespace(stage=stage, cameras={'wrist': {'kind': 'link', 'mount_link': 'right_hand_flange'}},
                          camera_paths=paths, prepare_cameras=lambda: dict(paths))
    view = StereoView(env, robot_poses=poses)
    initial = np.array(view.ops['left'].Get())
    assert np.allclose(initial[3, :3], [1.-.032, 2., 3.])
    view.update([5., 6., 7., 0., 0., 1., 0.], poses)
    assert np.array_equal(np.array(view.ops['left'].Get()), initial)
    poses['right_hand_flange']['position_m'][0] += .2
    view.update(None, poses)
    assert np.allclose(np.array(view.ops['left'].Get())[3, :3], [1.2-.032, 2., 3.])
    with pytest.raises(ValueError, match='world camera'):
        StereoView(env, head_tracked=True, robot_poses=poses)


def test_large_finger_curl_keeps_descent_direction(robot):
    from uni_teleop.mapping.franka.retarget import finger_step
    model, q = robot
    goal = dict(q)
    for j in model.finger_joints['index']:
        goal[j.name] = 0. if j.name.endswith('AA') else .6
    point = np.array(model.public_geometry(goal)['fingertip_positions_m']['index'])
    current = dict(q)
    for _ in range(20):
        solved, error = finger_step(model, current, 'index', point, deadline_ns=time.monotonic_ns()+1_000_000_000)
        current.update(solved)
    assert error < .003


def hand_recovery_case(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, hands_recovery=True)
    packet = frame('hands', joints=tracked_hand(model, q))
    mapper.set_active(True)
    mapper.command(packet, q, now_ns=packet.host_ns)
    return mapper, q, packet


def recovery_packet(packet, milliseconds, *, x=0., pinch=None):
    hand = np.array(packet.tracking['hands']['right'])
    hand[:, 0] += x
    hands = {'right': hand.tolist()}
    if pinch is not None:
        left = hand.copy()
        left[5, :3] = [0., 0., 0.]
        left[10, :3] = [pinch, 0., 0.]
        hands['left'] = left.tolist()
    return replace(packet, host_ns=packet.host_ns+int(milliseconds*1e6), sequence=milliseconds,
                   tracking={**packet.tracking, 'hands': hands})


def test_hand_stable_recovery_preserves_finger_calibration_and_reanchors_wrist(robot):
    mapper, q, packet = hand_recovery_case(robot)
    calibration = mapper.hand_calibrations['right']
    mapper.command(None, q, now_ns=packet.host_ns+200_000_000)
    assert not mapper.armed and mapper.recovery_pending
    for ms in (600, 700, 800):
        fresh = recovery_packet(packet, ms, x=.5)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
        assert not mapper.armed
    fresh = recovery_packet(packet, 860, x=.5)
    assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
    assert mapper.armed and not mapper.recovery_pending
    assert mapper.hand_calibrations['right'] is calibration
    assert np.allclose(mapper.anchors['right'][0][:3, 3], fresh.tracking['hands']['right'][1][:3])


def test_hand_recovery_needs_fresh_stationary_dwell(robot):
    mapper, q, packet = hand_recovery_case(robot)
    mapper.command(None, q, now_ns=packet.host_ns+200_000_000)
    fresh = recovery_packet(packet, 500)
    mapper.command(fresh, q, now_ns=fresh.host_ns)
    mapper.command(fresh, q, now_ns=fresh.host_ns+100_000_000)
    assert not mapper.armed  # Re-reading a packet does not extend the dwell.
    for ms in (600, 700, 800, 900):
        fresh = recovery_packet(packet, ms, x=(ms-500)*.001)
        mapper.command(fresh, q, now_ns=fresh.host_ns)
        assert not mapper.armed  # Wrist moves >3 cm across each candidate dwell.
    stale = recovery_packet(packet, 950)
    assert mapper.command(stale, q, now_ns=stale.host_ns+200_000_000) == q
    assert not mapper.armed


def test_long_hand_loss_needs_new_spare_hand_pinch_and_keeps_calibration(robot):
    mapper, q, packet = hand_recovery_case(robot)
    calibration = mapper.hand_calibrations['right']
    mapper.command(None, q, now_ns=packet.host_ns+3_000_000_000)
    for ms in range(3100, 3501, 100):
        fresh = recovery_packet(packet, ms, pinch=.01)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
        assert not mapper.armed  # A pinch already held at loss cannot resume.
    for ms, pinch in ((3600,.06),(3700,.01),(3800,.01),(3900,.01),(3960,.01)):
        fresh = recovery_packet(packet, ms, x=.4, pinch=pinch)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
    assert mapper.armed and mapper.hand_calibrations['right'] is calibration


@pytest.mark.parametrize('cause', ['pause', 'mode', 'source', 'focus', 'recalibrate'])
def test_hand_auto_recovery_never_overrides_explicit_stop(robot, cause):
    mapper, q, packet = hand_recovery_case(robot)
    mapper.command(None, q, now_ns=packet.host_ns+200_000_000)
    if cause == 'pause': mapper.set_active(False)
    elif cause == 'mode': mapper.set_mode('hands')
    elif cause == 'recalibrate': mapper.recalibrate()
    else:
        bad = recovery_packet(packet, 300)
        if cause == 'source': bad = replace(bad, stream='other')
        else: bad.tracking['focused'] = False
        mapper.command(bad, q, now_ns=bad.host_ns)
        packet = replace(packet, stream=bad.stream)
    for ms in (500,600,700,800):
        fresh = recovery_packet(packet, ms, pinch=.01)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
        assert not mapper.armed and not mapper.recovery_pending
    if cause in ('source','recalibrate'): assert not mapper.hand_calibrations


def test_pause_keeps_hand_calibration_but_explicit_recalibrate_clears_it(robot):
    mapper, q, packet = hand_recovery_case(robot)
    calibration = mapper.hand_calibrations['right']
    mapper.set_active(False)
    mapper.set_active(True)
    mapper.command(packet, q, now_ns=packet.host_ns)
    assert mapper.hand_calibrations['right'] is calibration
    mapper.recalibrate()
    assert not mapper.armed and not mapper.hand_calibrations


def test_new_hand_dropout_restarts_stability_and_pinch_confirmation(robot):
    mapper, q, packet = hand_recovery_case(robot)
    mapper.command(None, q, now_ns=packet.host_ns+3_000_000_000)
    for ms, pinch in ((3100,.06),(3200,.01),(3300,.01)):
        f = recovery_packet(packet, ms, pinch=pinch)
        mapper.command(f, q, now_ns=f.host_ns)
    mapper.command(None, q, now_ns=packet.host_ns+3_350_000_000)
    for ms in (3400,3500,3600,3700):
        f = recovery_packet(packet, ms, pinch=.01)
        assert mapper.command(f, q, now_ns=f.host_ns) == q
        assert not mapper.armed


def test_hand_rotation_during_recovery_restarts_dwell(robot):
    mapper, q, packet = hand_recovery_case(robot)
    mapper.command(None, q, now_ns=packet.host_ns+200_000_000)
    for ms in (300,400,500):
        f = recovery_packet(packet, ms)
        mapper.command(f, q, now_ns=f.host_ns)
    f = recovery_packet(packet, 560)
    f.tracking['hands']['right'][1][3:] = [0.,0.,np.sin(np.pi/12),np.cos(np.pi/12)]
    assert mapper.command(f, q, now_ns=f.host_ns) == q
    assert not mapper.armed


@pytest.mark.parametrize('broken', ['head', 'other_controller', 'other_hand', 'finger'])
def test_bad_unselected_channel_never_discards_good_controller_or_wrist(robot, broken):
    model, q = robot
    value = packet()['value']
    joints = [{'p': p, 's': 15} for p in tracked_hand(model, q)]
    value['Hand'] = {'rightHand': {'isActive': 1, 'HandJointLocations': joints}}
    if broken == 'head':
        value['Head'] = {'pose': 'invalid'}
    elif broken == 'other_controller':
        value['Controller']['left'] = {'pose': 'invalid', 'trigger': 'invalid'}
    elif broken == 'other_hand':
        value['Hand']['leftHand'] = {'isActive': 1, 'HandJointLocations': []}
    else:
        joints[10]['p'] = 'invalid'
    result = normalize_device(value)
    assert 'right' in result['controllers'] and 'right' in result['hands']
    assert result['hands']['right'][1] == tracked_hand(model, q)[1]


def test_finger_dropout_holds_only_that_finger_and_keeps_wrist_tracking(robot):
    mapper, q, packet = hand_recovery_case(robot)
    fresh = recovery_packet(packet, 50, x=.002)
    fresh.tracking['hands']['right'][10] = None
    fresh.tracking['hands']['right'][15][0] += .005
    targets = mapper.command(fresh, q, now_ns=fresh.host_ns)
    model = mapper.models['right']
    assert mapper.armed and not mapper.recovery_pending
    assert all(targets[j.name] == q[j.name] for j in model.finger_joints['index'])
    assert any(targets[n] != q[n] for n in model.arm.names)
    assert any(targets[j.name] != q[j.name] for j in model.finger_joints['middle'])
    assert 'right/index' in mapper.diagnostics
    assert all(np.isfinite(list(targets.values())))
    # A wrist loss still interrupts the whole arm, never substitutes a finger.
    value = {'Hand': {'rightHand': {'isActive': 1, 'HandJointLocations': [
        {'p': p, 's': 0 if i == 1 else 15} for i, p in enumerate(tracked_hand(model, q))]}}}
    missing = replace(fresh, tracking=normalize_device(value))
    assert mapper.command(missing, q, now_ns=fresh.host_ns) == q
    assert not mapper.armed and mapper.recovery_pending


def test_partial_hand_waits_for_complete_initial_calibration_without_manual_restart(robot):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, hands_recovery=True)
    mapper.set_active(True)
    full = frame('hands', joints=tracked_hand(model, q))
    partial = recovery_packet(full, 0)
    partial.tracking['hands']['right'][10] = None
    assert mapper.command(partial, q, now_ns=partial.host_ns) == q
    assert mapper.armed and not mapper.hand_calibrations
    fresh = recovery_packet(full, 50)
    assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
    assert mapper.armed and mapper.hand_calibrations


def test_recovered_but_moving_hand_does_not_turn_into_long_tracking_loss(robot):
    mapper, q, packet = hand_recovery_case(robot)
    mapper.command(None, q, now_ns=packet.host_ns+200_000_000)
    for ms in range(300, 3301, 100):
        fresh = recovery_packet(packet, ms, x=ms*.001)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
        assert not mapper.armed and not mapper.hand_recovery.manual
    for ms in (3400, 3500, 3600):
        fresh = recovery_packet(packet, ms, x=3.3)
        assert mapper.command(fresh, q, now_ns=fresh.host_ns) == q
    assert mapper.armed  # Stable fresh wrist resumes without the other hand.


def test_receiver_continuity_distinguishes_a_slow_consumer_from_tracking_loss(robot):
    mapper, q, packet = hand_recovery_case(robot)
    fresh = replace(recovery_packet(packet, 500),
                    continuous_since_ns={'hands/right': packet.host_ns})
    mapper.command(fresh, q, now_ns=fresh.host_ns)
    assert mapper.armed and not mapper.recovery_pending
    interrupted = replace(recovery_packet(packet, 1000),
                          continuous_since_ns={'hands/right': packet.host_ns+900_000_000})
    mapper.command(interrupted, q, now_ns=interrupted.host_ns)
    assert not mapper.armed and mapper.recovery_pending


def test_receiver_continuity_resets_for_invalid_channel_and_source_gap():
    receiver = Receiver(port=0)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        def send(sequence, invalid=False):
            payload = packet(sequence=sequence)
            if invalid:
                payload['value']['Controller']['right']['pose'] = 'invalid'
            sender.sendto(json.dumps(payload).encode(), receiver.socket.getsockname())
            return wait_for(lambda: receiver.latest if receiver.latest and receiver.latest.sequence == sequence else None)
        try:
            first = send(1)
            second = send(2)
            assert second.continuous_since_ns == first.continuous_since_ns
            assert not send(3, invalid=True).continuous_since_ns
            restored = send(4)
            assert restored.continuous_since_ns == {'controllers/right': restored.host_ns}
            time.sleep(.17)
            after_gap = send(5)
            assert after_gap.continuous_since_ns == {'controllers/right': after_gap.host_ns}
            assert receiver.health()['received'] == 5
        finally:
            receiver.close()


@pytest.mark.parametrize('rear_camera', [False, True])
def test_overview_stays_fixed_while_inset_tracks_wrist(rear_camera):
    from types import SimpleNamespace
    from pxr import Usd, UsdGeom
    from uni_teleop.video.isaac import StereoView
    stage = Usd.Stage.CreateInMemory()
    paths = {name: '/World/'+name for name in ('overview', 'wrist')}
    for path in paths.values():
        UsdGeom.Camera.Define(stage, path)
    poses = {'right_hand_flange': {'position_m': [1., 2., 3.], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    env = SimpleNamespace(stage=stage, cameras={'overview': {'kind': 'world'},
        'wrist': {'kind': 'link', 'mount_link': 'right_hand_flange'}},
        camera_paths=paths, prepare_cameras=lambda: dict(paths),
        scene_spec={'robot': {'base_position_m': [-.23, 0., .78]},
                    'table': {'position_m': [.55, 0., .75], 'half_extents_m': [.6, .5, .03]}})
    view = StereoView(env, 'overview', inset_camera='wrist', robot_poses=poses, rear_camera=rear_camera)
    if rear_camera:
        rear = np.array(view.rear_op.Get())
        assert rear[3, 0] < -.23 and rear[3, 2] > 1.8
        # Camera optical -z points toward the work surface.
        direction = np.array([.55, 0., .9]) - rear[3, :3]
        assert np.allclose(-rear[2, :3], direction/np.linalg.norm(direction))
    main = np.array(view.ops['left'].Get())
    inset = np.array(view.inset_op.Get())
    poses['right_hand_flange']['position_m'][0] += .2
    view.update([5., 6., 7., 0., 0., 1., 0.], poses)
    if rear_camera:
        assert np.array_equal(np.array(view.rear_op.Get()), rear)
    assert np.array_equal(np.array(view.ops['left'].Get()), main)
    assert np.allclose(np.array(view.inset_op.Get())[3, :3], inset[3, :3]+[.2, 0., 0.])
    left = np.full((480, 640, 3), 50, dtype=np.uint8)
    right = np.full_like(left, 100)
    closeup = np.full((192, 256, 3), 200, dtype=np.uint8)
    calls = []
    def render_many(cameras):
        calls.append(cameras)
        return dict(pico_left=left, pico_right=right, pico_inset=closeup,
                    **({'pico_rear': np.zeros((360, 480, 3), dtype=np.uint8)} if rear_camera else {}))
    env.render_many = render_many
    l, r = view.render()
    assert calls == [('pico_left', 'pico_right', 'pico_inset') + (('pico_rear',) if rear_camera else ())]
    assert np.array_equal(l[24:216, 376:632], closeup)
    assert np.array_equal(r[24:216, 376:632], closeup)
    assert np.array_equal(l[250:], left[250:])
    assert np.array_equal(r[250:], right[250:])
    assert (left == 50).all() and (right == 100).all()


def test_controller_closure_path_uses_both_actual_hand_models():
    from uni_teleop.mapping.franka.controller_hand import ControllerHand
    from bench.paths import data_root
    models = public_robot_geometry({'assets_root': str(data_root()/'assets'),
        'base_position_m': [0., 0., 0.], 'hand_id': 'dual_fr3_sharpa_w01',
        'assembly_version': 'isaacsim_6_1_v2'}).sides
    for model in models.values():
        hand = ControllerHand(model)
        reference = {j.name: 0. for j in model.joints}
        gaps = []
        for amount in np.linspace(.5, 1., 101):
            pose = hand.targets(amount)
            assert all(j.lower <= pose[j.name] <= j.upper for j in model.hand.active)
            assert all(model.hand_posture_prior.violation(f, pose) is None for f in FINGERS)
            tips = model.public_finger_geometry(reference | pose)['fingertip_positions_m']
            gaps.append(np.linalg.norm(np.array(tips['thumb'])-tips['index']))
        assert .063 < gaps[0] < .065
        assert .011 < gaps[-1] < .013
        assert np.all(np.diff(gaps) < 0)
        assert all(value == 0 for value in hand.targets(0.).values())
        for amount in (.5-1e-8, .5+1e-8):
            assert max(abs(hand.targets(amount)[n]-v) for n, v in hand.prepose.items()) < 1e-6


def test_controller_full_trigger_reaches_close_and_reopens_without_arm_motion(robot, monkeypatch):
    model, q = robot
    monkeypatch.setattr(model, 'solve', lambda reference, **kwargs: (reference, {}))
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_active(True)
    mapper.acknowledge(tick(mapper, frame(trigger=0.), q))
    for amount in (1., 0.):
        for _ in range(100):
            target = tick(mapper, frame(trigger=amount), q)
            assert all(abs(target[n]-mapper.acknowledged[n]) <= mapper.velocity[n]*.02+1e-12 for n in q)
            assert all(target[n] == q[n] for n in model.arm.names)
            mapper.acknowledge(target)
        expected = mapper.presets['right'].targets(amount)
        assert all(abs(target[n]-v) < 1e-9 for n,v in expected.items())
    held = tick(mapper, frame(grip=0., trigger=1.), q)
    assert held == mapper.acknowledged


@pytest.mark.parametrize('point', ['wrist', 'pinch'])
@pytest.mark.parametrize('locked', [False, True])
def test_controller_rotation_pivot_and_lock(robot, monkeypatch, point, locked):
    from scipy.spatial.transform import Rotation
    from uni_teleop.mapping.franka.retarget import wrist_matrix
    from bench.geometry import quaternion_matrix
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers',
                        control_point=point, lock_orientation=locked, scale=.3)
    requests = []
    def capture(reference, *, wrist, **kwargs):
        requests.append(wrist)
        return reference, {}
    monkeypatch.setattr(model, 'solve', capture)
    mapper.set_active(True)
    assert tick(mapper, frame(), q) == q
    moved = frame(x=.1)
    moved.tracking['controllers']['right']['pose'][3:] = Rotation.from_euler('y', 25, degrees=True).as_quat().tolist()
    tick(mapper, moved, q)
    before = wrist_matrix(model.public_geometry(q))
    after = requests[-1]
    R = quaternion_matrix(after['quaternion_wxyz'])
    p = np.array(after['position_m'])
    offset = mapper.grasp_offsets['right'] if point == 'pinch' else np.zeros(3)
    assert np.allclose(p + R @ offset, before[:3, 3] + before[:3, :3] @ offset + mapper.rotation @ [.03, 0, 0])
    assert np.allclose(R, before[:3, :3]) == locked
    assert np.allclose(mapper.control_points(q)[0], before[:3, 3] + before[:3, :3] @ offset)
    # Trigger changes cannot move the arm's reference or rotation pivot.
    moved.tracking['controllers']['right']['trigger'] = 1.
    tick(mapper, moved, q)
    assert requests[-1] == requests[-2]


def test_controller_mapping_switch_holds_and_reanchors(robot, monkeypatch):
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='controllers')
    mapper.set_controller_mapping(control_point='pinch', lock_orientation=True)
    assert not mapper.armed
    mapper.set_active(True)
    tick(mapper, frame(), q)
    mapper.set_controller_mapping(lock_orientation=False)
    def unexpected(*args, **kwargs):
        pytest.fail('Mapping switch must re-anchor before solving a new target')
    monkeypatch.setattr(model, 'solve', unexpected)
    assert tick(mapper, frame(x=.5), q) == q
    assert mapper.armed and 'right' in mapper.anchors
    mapper.set_active(False)
    mapper.set_controller_mapping(control_point='wrist')
    assert not mapper.armed and not mapper.anchors


def test_controller_options_do_not_change_bare_hand_wrist_target(robot, monkeypatch):
    from scipy.spatial.transform import Rotation
    from uni_teleop.mapping.franka.retarget import wrist_matrix
    from bench.geometry import quaternion_matrix
    model, q = robot
    mapper = Retargeter(model, q, {'right': 'right'}, mode='hands',
                        control_point='pinch', lock_orientation=True)
    requests = []
    def capture(reference, *, wrist, **kwargs):
        requests.append(wrist)
        return reference, {}
    monkeypatch.setattr(model, 'solve', capture)
    mapper.set_active(True)
    tick(mapper, frame('hands', joints=tracked_hand(model, q)), q)
    joints = tracked_hand(model, q)
    joints[1][3:] = Rotation.from_euler('y', .1).as_quat().tolist()
    tick(mapper, frame('hands', joints=joints), q)
    before = wrist_matrix(model.public_geometry(q))
    assert np.allclose(requests[-1]['position_m'], before[:3, 3])
    assert not np.allclose(quaternion_matrix(requests[-1]['quaternion_wxyz']), before[:3, :3])
    assert mapper.control_points(q) == []


def test_pico_wrist_camera_is_runtime_palm_mount_without_mutating_scene():
    from uni_teleop.video.isaac_camera import apply_in_hand_wrist_camera
    scene = {'cameras': {'wrist': {
        'kind': 'link', 'mount_link': 'right_hand_flange',
        'offset_xyz_m': [-.14, -.10, .025], 'offset_rpy_rad': [0., 0., 0.],
        'aim_target_m': [.5, -.9, .9], 'standoff_m': 0.,
        'resolution': [640, 480], 'focal_length_mm': 18.}},
        'objects': [{'id': 'keep'}]}
    original = __import__('copy').deepcopy(scene)
    updated = apply_in_hand_wrist_camera(scene)
    assert scene == original
    camera = updated['cameras']['wrist']
    assert camera['mount_link'] == 'right_hand_C_MC'
    assert camera['offset_xyz_m'] == [.055, .05, .045]
    assert camera['offset_rpy_rad'] == [np.pi, .2, 0.]
    assert camera['aim'] is False and camera['standoff_m'] == 0.
    assert 'aim_target_m' not in camera
    assert camera['resolution'] == [640, 480] and camera['focal_length_mm'] == 18.


def test_pico_wrist_camera_override_leaves_head_only_scene_unchanged():
    from uni_teleop.video.isaac_camera import apply_in_hand_wrist_camera
    scene = {'cameras': {'overview': {'kind': 'world'}}, 'objects': []}
    assert apply_in_hand_wrist_camera(scene) == scene


def test_unscored_display_hiding_preserves_weighing_physics():
    from scripts.teleop.pico_isaac import parser, load_inputs
    args = parser().parse_args(['--family', 'laboratory', '--task', 'solid_weighing',
                               '--no-task-scoring', '--keep-task-observer'])
    original, initial, _ = load_inputs(args)
    args.keep_task_observer = False
    args.hide_task_displays = True
    scene, new_initial, _ = load_inputs(args)
    bundle = scene['native_task_bundle']
    assert scene['native_task_observer_enabled'] is False
    assert bundle['hidden_display_prims']
    assert 'visual_state_bindings' not in bundle
    assert new_initial == initial
    assert bundle['rigid_bodies'] == original['native_task_bundle']['rigid_bodies']
    assert bundle['usd_sha256'] == original['native_task_bundle']['usd_sha256']
    args.keep_task_observer = True
    with pytest.raises(ValueError, match='task observation off'):
        load_inputs(args)
