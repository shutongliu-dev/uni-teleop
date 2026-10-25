"""Dual-input independence and semantic finger mapping; no vendor ABI or motors."""
import time

import numpy as np
import pytest

from uni_teleop.devices.manus.sources import host_clock_id
from uni_teleop.mapping.factr.calibration import ArmCalibration
from uni_teleop.mapping.factr_manus import ManusJointRetargeter, SEMANTICS
from uni_teleop.devices.pico.protocol import Frame
from uni_teleop.mapping.franka.retarget import FINGERS
from bench.control.factory import public_robot_geometry, robot_configuration
from scenes.workrooms.catalog import load_task


@pytest.fixture
def setup():
    task = load_task('mechanical_assembly', 'peg_insertion')
    geometry = public_robot_geometry(robot_configuration(task['scene']))
    q = task['initial_state']['joint_positions_rad']
    cal = ArmCalibration.from_reference(geometry, leader_id='synthetic',
        leader_reference_rad=[0.]*7, robot_reference_rad=[q[n] for n in geometry.arm.names],
        signs=[1]*7, directions_verified=[True]*7, source_kind='synthetic')
    mapper = ManusJointRetargeter(geometry, q, {'right': 'right'},
                                  glove_id=7, calibration_id='test', arm_calibration=cal, mode='controllers')
    return mapper, q


def packet(model, q, seq, *, stamp=None):
    geometry = model.public_finger_geometry(q)
    positions = np.zeros((26,3))
    for finger, indices in FINGERS.items():
        positions[list(indices)] = list(geometry['finger_joint_centers_m'][finger].values())+[
            geometry['fingertip_positions_m'][finger]]
    # The native MANUS root is shared by the two solver reference slots.
    nodes = [{'node_id':'0', 'parent_id':None, 'semantic':SEMANTICS[0],
              'pose':{'position':[0.,0.,0.], 'quaternion_wxyz':[1.,0.,0.,0.]}}]
    for i in range(2,26):
        nodes.append({'node_id':str(i), 'parent_id':'0', 'semantic':SEMANTICS[i],
                      'pose':{'position':positions[i].tolist(),'quaternion_wxyz':[1.,0.,0.,0.]}})
    return {'protocol':'manus_acquisition_v1','stream_id':'synthetic','sequence':seq,
            'clock_id':host_clock_id(),'capture_time_ns':time.monotonic_ns() if stamp is None else stamp,
            'device':{'device_id':'7','product':'synthetic','firmware':'none','sdk_version':'none',
                      'core_version':'none','hand':'right','source_kind':'synthetic'},
            'calibration_id':'test','topology_id':'synthetic_semantic','frame_id':'manus_world',
            'glove_tracking_valid':True,'nodes':nodes}


def frame(raw, hand, *, leader_stamp=None):
    now = time.monotonic_ns()
    return Frame('synthetic',0,now,'','test',{'leader':raw,'manus':hand,'head':None,
        'leader_id':'synthetic', 'leader_host_ns':now if leader_stamp is None else leader_stamp})


def align(mapper, q):
    end=time.monotonic_ns()
    for delta in (300,200,100,0):
        stamp=end-delta*1_000_000
        f=Frame('synthetic',0,stamp,'','test',{'leader':[0.]*8,'leader_id':'synthetic'})
        mapper.command(f,q,now_ns=stamp)
    assert mapper.armed


def test_manus_wrapper_does_not_relimit_unrestricted_arm(setup):
    original,q=setup
    m=ManusJointRetargeter(original.model,q,{'right':'right'},glove_id=7,calibration_id='test',
        arm_calibration=original.leader.arm_calibration,mode='controllers',fixed_reference=True,
        follow_on_start=True,practice_follow=True,unrestricted_arm=True)
    m.set_active(True)
    raw=[.4]*7+[0.]
    raw[1]=2.1-q['fr3_joint2']
    target=m.command(frame(raw,None),q,now_ns=time.monotonic_ns())
    assert target['fr3_joint2']==pytest.approx(2.1)
    assert target['fr3_joint1']==pytest.approx(q['fr3_joint1']+.4)
    assert all(target[n]==q[n] for n in m.model.hand.names)


def test_fingers_move_and_glove_loss_does_not_stop_leader(setup):
    m,q=setup; m.set_active(True); align(m.leader,q)
    assert m.command(frame([0.]*8,packet(m.model,q,0)),q,now_ns=time.monotonic_ns())==q
    closed=q | m.presets[m.name].targets(.8)
    target=m.command(frame([.02]+[0.]*7,packet(m.model,closed,1)),q,now_ns=time.monotonic_ns())
    assert target['fr3_joint1']>q['fr3_joint1']
    for finger in FINGERS:
        assert any(abs(target[j.name]-q[j.name])>1e-5 for j in m.model.finger_joints[finger])
    m.acknowledge(target)
    stale=packet(m.model,closed,2,stamp=time.monotonic_ns()-200_000_000)
    held=m.command(frame([.04]+[0.]*7,stale),target,now_ns=time.monotonic_ns())
    assert held['fr3_joint1']>target['fr3_joint1']
    assert all(held[n]==target[n] for n in q if n not in m.model.arm.names)
    assert m.armed and not m.hand_enabled
    fresh=packet(m.model,q,3)
    again=m.command(frame([.05]+[0.]*7,fresh),target,now_ns=time.monotonic_ns())
    assert all(again[n]==target[n] for n in q if n not in m.model.arm.names)


def test_leader_loss_holds_arm_but_fingers_continue_and_pause_holds_both(setup):
    m,q=setup;m.set_active(True)
    m.command(frame([0.]*8,packet(m.model,q,0)),q,now_ns=time.monotonic_ns())
    closed=q | m.presets[m.name].targets(.8)
    target=m.command(frame([0.]*8,packet(m.model,closed,1),
        leader_stamp=time.monotonic_ns()-200_000_000),q,now_ns=time.monotonic_ns())
    assert all(target[n]==q[n] for n in m.model.arm.names)
    assert any(target[n]!=q[n] for n in q if n not in m.model.arm.names)
    assert not m.leader.armed and m.armed
    m.acknowledge(target);m.set_active(False)
    assert m.command(frame([0.]*8,packet(m.model,q,2)),target,now_ns=time.monotonic_ns())==target


@pytest.mark.parametrize('field,value', [('clock_id','other_boot'),('calibration_id','changed')])
def test_wrong_identity_cannot_move_fingers(setup,field,value):
    m,q=setup;m.set_active(True)
    p=packet(m.model,q,0);p[field]=value
    assert m.command(frame([0.]*8,p),q,now_ns=time.monotonic_ns())==q
    assert not m.hand_enabled


def test_manus_finger_scale_does_not_include_palm_bones(setup):
    from copy import deepcopy
    from uni_teleop.mapping.factr_manus import ManusHandCalibration, finger_sample
    m,q=setup
    raw=packet(m.model,q,0)
    hand={'nodes':{n['node_id']:dict(n['pose'],semantic=n['semantic']) for n in raw['nodes']}}
    joints=finger_sample(hand)
    shifted=deepcopy(joints)
    for i in (6,11,16,21):
        shifted[i][1]-=.06
    a=ManusHandCalibration(m.model,q,joints)
    b=ManusHandCalibration(m.model,q,shifted)
    assert a.scales == b.scales
    assert np.allclose(a.rotation,b.rotation)


def test_finger_speed_improves_response_with_original_urdf_bounds(setup):
    from uni_teleop.mapping.factr_manus import finger_sample
    original,q0=setup
    errors=[]
    for speed in (1.,4.):
        m=ManusJointRetargeter(original.model,q0,{'right':'right'},
            glove_id=7,calibration_id='test',mode='controllers',finger_speed=speed)
        m.set_active(True)
        m.command(frame([0.]*8,packet(m.model,q0,0)),q0,now_ns=time.monotonic_ns())
        closed=q0 | m.presets[m.name].targets(.8)
        q=dict(q0)
        for i in range(1,21):
            old=q
            q=m.command(frame([0.]*8,packet(m.model,closed,i)),q,now_ns=time.monotonic_ns())
            for joint in m.model.hand.active:
                assert abs(q[joint.name]-old[joint.name]) <= min(joint.velocity,speed)*.02+1e-12
                assert joint.lower <= q[joint.name] <= joint.upper
            m.acknowledge(q)
        raw=packet(m.model,closed,21)
        hand={'nodes':{n['node_id']:dict(n['pose'],semantic=n['semantic']) for n in raw['nodes']}}
        goals=m.calibration.targets(finger_sample(hand))
        tips=m.model.public_finger_geometry(q)['fingertip_positions_m']
        errors.append(sum(np.linalg.norm(np.array(tips[f])-goals[f]) for f in FINGERS))
    assert errors[1] < errors[0]*.6


def test_explicit_hand_only_keeps_arm_held_even_with_valid_encoder_input(setup):
    original,q=setup
    m=ManusJointRetargeter(original.model,q,{'right':'right'},
        glove_id=7,calibration_id='test',mode='controllers',hand_only=True)
    m.set_active(True)
    m.command(frame([0.]*8,packet(m.model,q,0)),q,now_ns=time.monotonic_ns())
    closed=q | m.presets[m.name].targets(.8)
    target=m.command(frame([.1]*8,packet(m.model,closed,1)),q,now_ns=time.monotonic_ns())
    assert all(target[n]==q[n] for n in m.model.arm.names)
    assert any(target[n]!=q[n] for n in q if n not in m.model.arm.names)
    assert m.input_mapping['arm_source']=='held' and not m.leader.armed
    assert 'arm' not in m.diagnostics


def test_alignment_wait_does_not_block_manus_fingers(setup):
    m,q=setup;m.set_active(True)
    m.command(frame([.18]+[0.]*7,packet(m.model,q,0)),q,now_ns=time.monotonic_ns())
    closed=q | m.presets[m.name].targets(.8)
    target=m.command(frame([.18]+[0.]*7,packet(m.model,closed,1)),q,now_ns=time.monotonic_ns())
    assert m.leader.recovery_pending and not m.leader.armed
    assert all(target[n]==q[n] for n in m.model.arm.names)
    assert any(target[n]!=q[n] for n in m.model.hand.names)
    assert 'arm' in m.diagnostics and len(m.alignment['error_rad'])==7
    calibration=m.leader.arm_calibration
    m.recalibrate()
    assert m.leader.arm_calibration is calibration
