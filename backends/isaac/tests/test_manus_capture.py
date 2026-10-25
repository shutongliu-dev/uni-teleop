"""Behavioral contracts for acquisition, calibration and robot-command isolation."""
import hashlib
import io
import json
import math
from pathlib import Path
import pytest
from pydantic import ValidationError
from uni_teleop.mapping.manus.bridge import CaptureBridge, RetargetRegistry
from uni_teleop.recording.manus import CaptureRecorder
from uni_teleop.devices.manus.schema import AcquisitionPacket, CaptureConfig
from uni_teleop.devices.manus.sources import mock_packets, packets_from_jsonl
from bench.paths import load_json


def packet(sequence=0, **changes):
    value = next(mock_packets(1, clock_id='test_clock', start_ns=1_000_000_000)).model_dump()
    value.update(stream_id='fixed_stream', sequence=sequence, capture_time_ns=1_000_000_000+sequence*10_000_000)
    value.update(changes)
    return AcquisitionPacket.model_validate(value)


def config(**changes):
    value = dict(calibration_id='synthetic_identity', target_clock_id='test_clock',
                 glove_calibrated=True, world_calibrated=True, wrist_offset_calibrated=True,
                 tracker_body_frame='wrist_tracker', robot_profile_id='sharpa_w01_right', retarget_version='fixture_v1')
    value.update(changes)
    return CaptureConfig(**value)


def tracker(**changes):
    value = dict(tracker_id='tracker_1', status='tracked', clock_id='test_clock',
                 capture_time_ns=1_000_000_000, frame_id='manus_world', tracked_body_frame='wrist_tracker',
                 pose={'position': [1, 2, 3], 'quaternion_wxyz': [1, 0, 0, 0]})
    value.update(changes)
    return value


def test_metric_conversion_parent_local_and_rigid_calibration():
    nodes = [dict(node_id='wrist', parent_id=None, semantic='wrist',
                  pose={'position': [1000, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]}),
             dict(node_id='tip', parent_id='wrist', semantic='tip',
                  pose={'position': [0, 100, 0], 'quaternion_wxyz': [1, 0, 0, 0]})]
    c = config(world_from_capture={'position': [0, 0, 1], 'quaternion_wxyz': [math.sqrt(.5), 0, 0, math.sqrt(.5)]})
    result = CaptureBridge(c).accept(packet(nodes=nodes, position_unit='mm', node_space='parent_local',
                                           ergonomics_degrees={'mcp': 90}), 1_000_000_001)
    assert result.frame['nodes']['tip']['position'] == pytest.approx([-.1, 1, 1])
    assert result.frame['ergonomics_rad']['mcp'] == pytest.approx(math.pi/2)


def test_sequence_idempotence_and_conflict_do_not_duplicate_normalized_samples(tmp_path):
    r = CaptureRecorder(config(), tmp_path)
    p = packet()
    assert r.ingest(p, 1_000_000_000).status == 'accepted'
    assert r.ingest(p, 1_000_000_001).status == 'duplicate'
    changed = p.model_copy(deep=True)
    changed.nodes[1].pose.position[0] += .01
    assert r.ingest(changed, 1_000_000_002).reason == 'sequence_payload_conflict'
    assert r.ingest(packet(3), 1_030_000_000).frame['sequence_gap'] == 2
    assert r.ingest(packet(2), 1_040_000_000).reason == 'out_of_order'
    r.close()
    assert len((r.path/'human_motion.jsonl').read_text().splitlines()) == 2
    manifest = load_json(r.path/'manifest.json')
    assert manifest['counts'] == {'accepted': 2, 'duplicate': 1, 'rejected': 2, 'invalid': 0}
    for name, proof in manifest['files'].items():
        assert hashlib.sha256((r.path/name).read_bytes()).hexdigest() == proof['sha256']


@pytest.mark.parametrize('change,reason', [
    ({'clock_id': 'remote_clock'}, 'clock_unsynchronized'),
    ({'capture_time_ns': 800_000_000}, 'stale_glove'),
    ({'capture_time_ns': 1_100_000_000}, 'future_timestamp'),
    ({'glove_tracking_valid': False}, 'glove_tracking_lost'),
])
def test_stale_future_unknown_clocks_retained_but_not_controllable(change, reason):
    result = CaptureBridge(config()).accept(packet(**change), 1_000_000_000)
    assert result.status == 'accepted'
    assert not result.frame['hand_control_valid']
    assert reason in result.frame['hand_hold_reasons']


def test_measured_clock_mapping_and_uncertainty():
    mapping = dict(source_clock_id='remote', target_clock_id='test_clock', offset_ns=500_000_000,
                   uncertainty_ns=1_000_000, method='measured_offset')
    result = CaptureBridge(config(clock_mappings=[mapping])).accept(packet(clock_id='remote', capture_time_ns=500_000_000), 1_000_000_000)
    assert result.frame['hand_control_valid']
    assert result.frame['clock_uncertainty_ns'] == 1_000_000
    mapping['uncertainty_ns'] = 6_000_000
    result = CaptureBridge(config(clock_mappings=[mapping])).accept(packet(clock_id='remote'), 1_000_000_000)
    assert not result.frame['hand_control_valid']


@pytest.mark.parametrize('tr,expected', [
    (None, 'wrist_6dof_unavailable'),
    (tracker(status='orientation_only'), 'wrist_6dof_unavailable'),
    (tracker(capture_time_ns=900_000_000), 'tracker_glove_time_skew'),
    (tracker(clock_id='unknown'), 'tracker_clock_unsynchronized'),
    (tracker(tracked_body_frame='another_tracker'), 'tracker_body_mismatch'),
])
def test_wrist_does_not_appear_from_glove_or_unsynced_tracker(tr, expected):
    frame = CaptureBridge(config()).accept(packet(tracker=tr), 1_000_000_000).frame
    assert frame['hand_control_valid']
    assert not frame['arm_control_valid']
    assert frame['wrist'] is None
    assert expected in frame['arm_hold_reasons']


def test_tracker_offset_composition_and_calibration_gate():
    c = config(tracker_from_wrist={'position': [.1, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]})
    frame = CaptureBridge(c).accept(packet(tracker=tracker()), 1_000_000_000).frame
    assert frame['arm_control_valid']
    assert frame['wrist']['position'] == pytest.approx([1.1, 2, 3])
    c.world_calibrated = False
    assert not CaptureBridge(c).accept(packet(tracker=tracker()), 1_000_000_000).frame['arm_control_valid']


def test_retarget_respects_profile_joint_set_tracker_hold_and_watchdog():
    frame = CaptureBridge(config()).accept(packet(), 1_000_000_000).frame
    registry = RetargetRegistry()
    registry.register('sharpa_w01_right', 'fixture_v1', lambda f: {'arm': .5, 'finger': .8})
    kwargs = dict(previous_targets={'arm': .1, 'finger': .2}, joint_limits={'arm': (-1, 1), 'finger': (0, 1)}, arm_joint_names=['arm'])
    action, metadata = registry.map(frame, now_clock_id='test_clock', now_ns=1_000_000_000, **kwargs)
    assert action == {'arm': .1, 'finger': .8}
    assert metadata['mode'] == 'hand_only'
    action, metadata = registry.map(frame, now_clock_id='test_clock', now_ns=1_200_000_000, **kwargs)
    assert action == kwargs['previous_targets']
    assert metadata['reasons'] == ['input_watchdog_expired']
    frame['robot_profile_id'] = 'leap_v1_right'
    with pytest.raises(ValueError, match='no retargeter'):
        registry.map(frame, now_clock_id='test_clock', now_ns=1_000_000_000, **kwargs)
    registry.register('leap_v1_right', 'fixture_v1', lambda f: {'arm': .1, 'finger': 2})
    with pytest.raises(ValueError, match='invalid robot joint'):
        registry.map(frame, now_clock_id='test_clock', now_ns=1_000_000_000, **kwargs)


def test_loss_frame_with_empty_nodes_still_immediately_holds_hand():
    bridge = CaptureBridge(config())
    bridge.accept(packet(), 1_000_000_000)
    result = bridge.accept(packet(1, nodes=[], glove_tracking_valid=False), 1_010_000_000)
    assert result.status == 'accepted'
    assert not result.frame['hand_control_valid']


def test_schema_rejects_cycles_nonfinite_bad_quaternion_and_credential_fields():
    p = packet().model_dump()
    p['nodes'][0]['parent_id'] = p['nodes'][1]['node_id']
    with pytest.raises(ValidationError, match='cyclic'):
        AcquisitionPacket.model_validate(p)
    p = packet().model_dump()
    p['nodes'][0]['pose']['position'][0] = float('nan')
    with pytest.raises(ValidationError):
        AcquisitionPacket.model_validate(p)
    p = packet().model_dump()
    p['nodes'][0]['pose']['quaternion_wxyz'] = [0, 0, 0, 0]
    with pytest.raises(ValidationError):
        AcquisitionPacket.model_validate(p)
    p = packet().model_dump()
    p['password'] = 'fixture-never-log'
    with pytest.raises(ValidationError):
        AcquisitionPacket.model_validate(p)


def test_jsonl_roundtrip_keeps_original_hardware_source_and_vendor_timestamp(tmp_path):
    p = packet().model_dump()
    p['device']['source_kind'] = 'hardware'
    p['source_timestamp'] = '18446744073709551615'
    restored = list(packets_from_jsonl(io.StringIO(json.dumps({'packet': p})+'\n')))[0]
    assert restored.source_timestamp == p['source_timestamp']
    r = CaptureRecorder(config(), tmp_path, replay_of='/source/acquisition.jsonl')
    r.ingest(restored, 9_000_000_000)
    r.close()
    m = load_json(r.path/'manifest.json')
    assert m['hardware_connected'] is False
    assert m['source_kind_counts']['hardware'] == 1
    assert m['replay_of'] is not None


def test_agent_command_split_and_profile_velocity_limits():
    from bench.protocol.schemas import RobotCapabilities
    from uni_teleop.backends.isaac.manus_actions import build_command, direct_action
    caps = RobotCapabilities(robot_id='leap_v1_right', robot_asset_sha256='a'*64,
        arm_joints=['arm'], hand_joints=['finger'],
        joint_limits={'arm': {'lower_rad': -1., 'upper_rad': 1., 'max_velocity_rad_s': 1.},
                      'finger': {'lower_rad': 0., 'upper_rad': 1., 'max_velocity_rad_s': 1.}})
    c = config(robot_profile_id='leap_v1_right')
    frame = CaptureBridge(c).accept(packet(), 1_000_000_000).frame
    registry = RetargetRegistry()
    registry.register('leap_v1_right', 'fixture_v1', lambda f: {'arm': .9, 'finger': .21})
    command, metadata = build_command(frame, registry, caps, previous_targets={'arm': .1, 'finger': .2}, now_clock_id='test_clock', now_ns=1_000_000_000)
    action = direct_action(command, caps, action_id='capture_tick_0001', observation_sequence=7)
    assert action.arm_joint_positions_rad == {'arm': .1}
    assert action.hand_joint_positions_rad == {'finger': .21}
    assert metadata['source_sequence'] == 0
    with pytest.raises(ValueError, match='joint speed'):
        build_command(frame, registry, caps, previous_targets={'arm': .1, 'finger': 0}, now_clock_id='test_clock', now_ns=1_000_000_000)


def test_control_clock_cannot_be_borrowed_from_another_host():
    frame = CaptureBridge(config()).accept(packet(), 1_000_000_000).frame
    registry = RetargetRegistry()
    action, metadata = registry.map(frame, previous_targets={'arm': 0., 'finger': 0.},
        joint_limits={'arm': (-1, 1), 'finger': (0, 1)}, arm_joint_names=['arm'],
        now_ns=1_000_000_000, now_clock_id='another_boot_or_host')
    assert action == {'arm': 0., 'finger': 0.}
    assert metadata['reasons'] == ['control_clock_mismatch']


def test_byte_bound_rejects_multibyte_oversized_jsonl():
    from uni_teleop.devices.manus.sources import MAX_LINE_BYTES
    with pytest.raises(ValueError, match='2 MiB'):
        list(packets_from_jsonl(io.StringIO('汉'*(MAX_LINE_BYTES//2)+'\n')))


def test_mapping_callback_cannot_change_tracking_gates():
    frame = CaptureBridge(config()).accept(packet(), 1_000_000_000).frame
    registry = RetargetRegistry()
    def misbehaving_mapper(value):
        value['arm_control_valid'] = True
        value['arm_valid_until_ns'] = 2_000_000_000
        return {'arm': .9, 'finger': .2}
    registry.register('sharpa_w01_right', 'fixture_v1', misbehaving_mapper)
    action, _ = registry.map(frame, previous_targets={'arm': 0., 'finger': 0.},
        joint_limits={'arm': (-1, 1), 'finger': (0, 1)}, arm_joint_names=['arm'],
        now_ns=1_000_000_000, now_clock_id='test_clock')
    assert action['arm'] == 0.
    assert frame['arm_control_valid'] is False


def test_capture_configuration_cannot_change_mid_episode():
    bridge = CaptureBridge(config())
    bridge.accept(packet(), 1_000_000_000)
    bridge.config.world_from_capture.position[0] = 100
    with pytest.raises(ValueError, match='configuration changed'):
        bridge.accept(packet(1), 1_010_000_000)
