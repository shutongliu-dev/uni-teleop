"""FACTR serial framing and joint mapper contracts; no torque or physical writes."""
import os
import pty
import struct
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from bench.adapters.master_slave.dynamixel import crc, packet, ReadOnlyLeader
from bench.adapters.master_slave.retarget import JointRetargeter
from bench.adapters.master_slave.calibration import ArmCalibration
from bench.adapters.pico.protocol import Frame
from bench.control.factory import public_robot_geometry, robot_configuration
from scenes.workrooms.catalog import load_task


def sample(raw, stamp=None):
    now = time.monotonic_ns() if stamp is None else stamp
    return Frame('leader', 0, now, str(now), 'FACTR', {'leader': list(raw), 'head': None, 'leader_id': 'synthetic'})


@pytest.fixture
def mapper():
    task = load_task('mechanical_assembly', 'peg_insertion')
    model = public_robot_geometry(robot_configuration(task['scene']))
    q = task['initial_state']['joint_positions_rad']
    calibration = ArmCalibration.from_reference(model, leader_id='synthetic',
        leader_reference_rad=[0.]*7, robot_reference_rad=[q[n] for n in model.arm.names],
        signs=[1,1,1,-1,1,-1,1], directions_verified=[True]*7, source_kind='synthetic')
    return JointRetargeter(model, q, {'right': 'right'},
                           arm_calibration=calibration, mode='controllers'), q


def test_unrestricted_arm_follows_full_target_and_holds_on_real_loss(mapper):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     mode='controllers',fixed_reference=True,follow_on_start=True,
                     practice_follow=True,unrestricted_arm=True)
    m.set_active(True)
    raw=np.array([.4]*7+[0.])
    raw[1]=2.1-q['fr3_joint2']
    expected=m.arm_calibration.continuous_targets(raw,[q[n] for n in m.model.arm.names])
    target=m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert np.allclose([target[n] for n in m.model.arm.names],expected)
    assert target['fr3_joint2']>m.model.arm.active[1].upper
    assert m.armed and m.issue is None
    m.acknowledge(target)
    assert m.command(None,q,now_ns=time.monotonic_ns())==target
    assert not m.armed


def test_protocol_read_frame_known_crc_and_signed_sync_read():
    # ROBOTIS protocol 2 official PING example: CRC 0x4e19.
    assert packet(1, 1, b'').hex() == 'fffffd0001030001194e'
    master, slave = pty.openpty()
    leader = ReadOnlyLeader(os.ttyname(slave))
    errors = []
    def device():
        try:
            request = os.read(master, 1024)
            assert request[7] == 0x82
            assert request[8:12] == struct.pack('<HH',132,4)
            for identity in range(8,0,-1):
                os.write(master, packet(identity, 0x55, b'\x00'+struct.pack('<i', -4096*identity)))
        except BaseException as error:errors.append(error)
    thread = threading.Thread(target=device);thread.start()
    try:
        values, stamp = leader.positions()
        assert np.allclose(values, -2*np.pi*np.arange(1,9))
        assert stamp <= time.monotonic_ns()
    finally:
        thread.join(1);leader.close();os.close(master);os.close(slave)
    assert not errors


def align(m, q, raw=None):
    raw = np.zeros(8) if raw is None else raw
    end = time.monotonic_ns()
    for delta in (300, 200, 100, 0):
        stamp = end-delta*1_000_000
        assert m.command(sample(raw, stamp), q, now_ns=stamp) == m.acknowledged
    assert m.armed and not m.recovery_pending


def test_fixed_provisional_reference_survives_start_and_uses_all_seven_joints(mapper,tmp_path,monkeypatch):
    original,q=mapper
    cal=original.arm_calibration.model_copy(update={'status':'provisional','directions_verified':(False,)*7})
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=cal,mode='controllers',
                     fixed_reference=True,calibration_output=tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Joint mapping must not call wrist IK')
    monkeypatch.setattr(m.model,'solve',forbidden)
    raw=np.array([.4]*8)
    assert m.command(sample(raw),q,now_ns=time.monotonic_ns())==q
    m.capture_reference()
    saved=m.arm_calibration
    assert saved.status=='provisional' and not any(saved.directions_verified)
    assert len(list(tmp_path.glob('arm_reference_*.json')))==1
    m.set_active(True);align(m,q,raw)
    moved=raw.copy();moved[:7]+=.01
    target=m.command(sample(moved),q,now_ns=time.monotonic_ns())
    assert all(abs(target[n]-q[n])>.009 for n in m.model.arm.names)
    m.set_active(False);m.set_active(True)
    bad=raw.copy();bad[0]+=.1
    assert m.command(sample(bad),q,now_ns=time.monotonic_ns())==q
    assert m.arm_calibration is saved and not m.armed
    assert m.input_mapping['anchor_policy']=='saved_fixed_reference'


def test_reference_capture_requires_paused_reference_pose_and_fresh_sample(mapper,tmp_path):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     mode='controllers',calibration_output=tmp_path)
    raw=np.array([.4]*8)
    m.command(sample(raw),q,now_ns=time.monotonic_ns())
    m.set_active(True)
    assert not m.capture_reference(require_saved_pose=True)
    m.set_active(False)
    wrong=q|{m.model.arm.names[1]:q[m.model.arm.names[1]]+.2}
    m.command(sample(raw),wrong,now_ns=time.monotonic_ns())
    assert not m.capture_reference(require_saved_pose=True)
    stamp=time.monotonic_ns()-200_000_000
    m.command(sample(raw,stamp),q,now_ns=stamp)
    assert not m.capture_reference(require_saved_pose=True)
    assert not list(tmp_path.iterdir())
    m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert m.capture_reference(require_saved_pose=True)
    assert m.arm_calibration.signs==original.arm_calibration.signs
    assert np.allclose(m.arm_calibration.leader_reference_rad,raw[:7])
    assert np.allclose(m.arm_calibration.targets(raw,m.model),[q[n] for n in m.model.arm.names])
    assert len(list(tmp_path.glob('arm_reference_*.json')))==1


def test_first_gui_start_captures_once_and_resume_keeps_zero(mapper,tmp_path):
    from types import SimpleNamespace
    from scripts.teleop.pico_isaac import Controls
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     mode='controllers',calibration_output=tmp_path,fixed_reference=True,
                     follow_on_start=True,practice_follow=True)
    ui=object.__new__(Controls)
    ui.mapper,ui.feedback,ui.capture_on_start=m,None,True
    ui.reference_label=SimpleNamespace(text='')
    ui.start()  # No fresh sample: remain paused and retain the one-time request.
    assert ui.capture_on_start and not m.alignment_pending
    raw=np.array([.4]*8)
    ui.toggle()  # Space and Start use the same capture path.
    m.command(sample(raw),q,now_ns=time.monotonic_ns())
    ui.complete_reference_capture()
    assert not ui.capture_on_start and m.alignment_pending
    saved=m.arm_calibration
    assert np.allclose(saved.leader_reference_rad,raw[:7])
    target=m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert np.allclose([target[n] for n in m.model.arm.names],[q[n] for n in m.model.arm.names])
    ui.pause()
    raw[1]+=.02
    m.command(sample(raw),q,now_ns=time.monotonic_ns())
    ui.start()
    assert m.arm_calibration is saved
    assert len(list(tmp_path.glob('arm_reference_*.json')))==1


def test_fixed_calibration_ack_limits_and_no_rebasing(mapper):
    m,q = mapper
    calibration = m.arm_calibration
    m.set_active(True)
    align(m, q)
    raw = np.zeros(8); raw[0] = .05; raw[3] = .04
    target = m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert target['fr3_joint1'] > q['fr3_joint1']
    assert target['fr3_joint4'] < q['fr3_joint4']
    assert target != q and m.acknowledged == q
    assert all(abs(target[n]-q[n]) <= m.velocity[n]*.02+1e-12 for n in q)
    m.acknowledge(target)
    stale = sample(raw,time.monotonic_ns()-200_000_000)
    assert m.command(stale,q,now_ns=time.monotonic_ns()) == target
    assert not m.armed and not m.recovery_pending

    m.set_active(True)
    # Arbitrary Start cannot redefine the saved joint zero or chase the leader.
    assert m.command(sample([.2]+[0.]*7),target,now_ns=time.monotonic_ns()) == target
    assert m.recovery_pending and not m.armed
    assert m.alignment['error_rad']['fr3_joint1'] > .18
    m.recalibrate()
    assert m.arm_calibration is calibration
    assert not m.armed and not m.recovery_pending


def test_start_automatically_follows_all_joints_without_rebasing(mapper):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     fixed_reference=True,follow_on_start=True,mode='controllers')
    cal=m.arm_calibration
    raw=np.array([.12]*8);actual=dict(q);m.set_active(True)
    start=time.monotonic_ns()
    for i in range(60):
        now=start+i*20_000_000
        target=m.command(sample(raw,now),actual,now_ns=now)
        if m.alignment.get('state')=='synchronizing':
            assert m.issue['code']=='joint_synchronizing'
            assert m.issue['reason']==m.status and 'held' not in m.issue['reason']
        assert all(abs(target[n]-actual[n])<=m.velocity[n]*.02+1e-10 for n in actual)
        m.acknowledge(target);actual=target
    assert m.armed and m.arm_calibration is cal
    assert np.allclose([actual[n] for n in m.model.arm.names],cal.targets(raw,m.model))
    m.set_active(False)
    assert m.command(sample(raw+.05,now+20_000_000),actual,now_ns=now+20_000_000)==actual


def test_read_leader_initializes_all_joints_without_changing_zero_or_props(mapper,monkeypatch):
    m,q=mapper
    m.fixed_reference=True
    cal=m.arm_calibration
    initial={'joint_positions_rad':dict(q),'objects':{'part':{'position':[.2,0.,.8]}}}
    raw=np.array([.04]*8)
    def forbidden(*args,**kwargs):raise AssertionError('Leader initialization must not use IK')
    monkeypatch.setattr(m.model,'solve',forbidden)
    f=sample(raw)
    state,receipt=m.initial_state_from_input(f,initial,now_ns=time.monotonic_ns())
    expected=cal.targets(raw,m.model)
    assert np.allclose([state['joint_positions_rad'][n] for n in m.model.arm.names],expected)
    assert all(state['joint_positions_rad'][n]==v for n,v in q.items() if n not in m.model.arm.names)
    assert state['objects']==initial['objects'] and state['objects'] is not initial['objects']
    assert initial['joint_positions_rad']==q and m.arm_calibration is cal
    assert receipt['arm_calibration_sha256']==cal.sha256
    m.acknowledge(state['joint_positions_rad'])
    m.set_active(True);align(m,state['joint_positions_rad'],raw)
    assert m.arm_calibration is cal


@pytest.mark.parametrize('fault',['stale','future','wrong_id','missing_axis','limit','stale_leader_fresh_glove'])
def test_read_leader_rejects_invalid_input_without_changing_initial_state(mapper,fault):
    m,q=mapper;m.fixed_reference=True
    initial={'joint_positions_rad':dict(q)}
    now=time.monotonic_ns();f=sample(np.zeros(8),now)
    if fault=='stale':f=replace(f,host_ns=now-151_000_000)
    if fault=='future':f=replace(f,host_ns=now+1)
    if fault=='wrong_id':f.tracking['leader_id']='different'
    if fault=='missing_axis':f.tracking['leader']=f.tracking['leader'][:7]
    if fault=='limit':f.tracking['leader'][1]=3.14
    if fault=='stale_leader_fresh_glove':f.tracking['leader_host_ns']=now-151_000_000
    with pytest.raises(ValueError):m.initial_state_from_input(f,initial,now_ns=now)
    assert initial['joint_positions_rad']==q


@pytest.mark.parametrize('change', [float('nan'), 1.])
def test_invalid_or_jumping_encoder_cannot_move_robot(mapper,change):
    m,q = mapper; m.set_active(True); align(m,q)
    raw=np.zeros(8);raw[0]=change
    assert m.command(sample(raw),q,now_ns=time.monotonic_ns()) == q
    assert not m.armed and not m.recovery_pending


@pytest.mark.parametrize('axis', range(7))
def test_every_axis_must_align_before_start(mapper, axis):
    m,q=mapper; raw=np.zeros(8);raw[axis]=.1;m.set_active(True)
    stamp=time.monotonic_ns()
    for step in range(8):
        now=stamp+step*50_000_000
        assert m.command(sample(raw,now),q,now_ns=now)==q
    assert not m.armed and m.recovery_pending


def test_alignment_needs_continuous_fresh_stable_samples(mapper):
    m,q=mapper;m.set_active(True);stamp=time.monotonic_ns();raw=np.zeros(8)
    for step in range(10):
        raw[0]=.015*(step%3)
        now=stamp+step*50_000_000
        m.command(sample(raw,now),q,now_ns=now)
        assert not m.armed  # Accumulating drift resets the dwell.
    duplicate=sample(raw,now)
    m.command(duplicate,q,now_ns=now+100_000_000)
    assert not m.armed
    m.command(sample(raw,now+200_000_000),q,now_ns=now+200_000_000)
    assert not m.armed and m.recovery_pending
    assert m.alignment['dwell_s']==0.  # Unsampled time cannot complete alignment.


def test_wrong_identity_or_stream_change_holds(mapper):
    m,q=mapper;m.set_active(True);align(m,q)
    changed=replace(sample(np.zeros(8)),stream='different')
    assert m.command(changed,q,now_ns=time.monotonic_ns())==q
    assert not m.armed
    m.set_active(True)
    wrong=sample(np.zeros(8));wrong.tracking['leader_id']='other'
    assert m.command(wrong,q,now_ns=time.monotonic_ns())==q
    assert not m.recovery_pending


def test_uncalibrated_arm_never_moves(mapper):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'})
    m.set_active(True)
    assert m.command(sample([.2]*8),q,now_ns=time.monotonic_ns())==q
    assert not m.armed and not m.recovery_pending
    assert 'missing' in m.status


def test_calibration_persistence_binding_and_full_turns(mapper,tmp_path):
    m,q=mapper;cal=m.arm_calibration;path=tmp_path/'calibration.json'
    cal.save(path);loaded=ArmCalibration.load(path)
    assert loaded==cal and loaded.sha256==cal.sha256
    with pytest.raises(FileExistsError):cal.save(path)
    for turns in (-4,0,3):
        assert np.allclose(loaded.targets([turns*2*np.pi]*8,m.model),
                           [q[n] for n in m.model.arm.names])
    with pytest.raises(ValueError):
        loaded.model_copy(update={'robot_urdf_sha256':'0'*64}).validate_robot(m.model)
    for field,value in [('directions_verified',[False]*7),('signs',[1]*6),
                        ('leader_reference_rad',[float('nan')]*7),('motor_ids',[1,2,3])]:
        with pytest.raises(ValueError):ArmCalibration.model_validate(cal.model_dump() | {field:value})
    from pathlib import Path
    with pytest.raises(ValueError):
        ArmCalibration.load(Path('scripts/teleop/config/arm_calibration.template.json'))


def test_alignment_checks_ack_as_well_as_measured_pose(mapper):
    m,q=mapper;m.acknowledge(q | {'fr3_joint1':q['fr3_joint1']+.1});m.set_active(True)
    stamp=time.monotonic_ns()
    for step in range(8):
        now=stamp+step*50_000_000
        m.command(sample([0.]*8,now),q,now_ns=now)
    assert not m.armed


def test_slow_consumer_with_fresh_input_does_not_create_false_loss(mapper):
    m,q=mapper;m.set_active(True);align(m,q)
    stamp=m.previous_stamp+200_000_000
    target=m.command(sample([.02]+[0.]*7,stamp),q,now_ns=stamp)
    assert m.armed and target['fr3_joint1']>q['fr3_joint1']


def test_provisional_reference_keeps_unverified_directions_explicit(mapper):
    original,q=mapper
    value=original.arm_calibration.model_dump() | {'status':'provisional','directions_verified':[False]*7}
    cal=ArmCalibration.model_validate(value)
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=cal)
    assert m.input_mapping['alignment_dwell_s']==0.
    assert m.input_mapping['alignment_tolerance_rad'] is None
    assert m.input_mapping['arm_calibration']['directions_verified']==[False]*7
    m.set_active(True)
    # Practice anchors at Start, then follows bounded relative movement.
    raw=[.1]+[0.]*7
    assert m.command(sample(raw),q,now_ns=time.monotonic_ns())==q
    assert m.armed and not m.recovery_pending
    raw[0]+=.02
    target=m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert 0 < target['fr3_joint1']-q['fr3_joint1'] <= m.velocity['fr3_joint1']*.02+1e-12
    with pytest.raises(ValueError):
        ArmCalibration.model_validate(value | {'status':'measured'})


def test_hardware_entry_requires_explicit_provisional_opt_in(mapper,tmp_path,monkeypatch):
    from scripts.teleop import master_slave_isaac as entry
    original,_=mapper
    cal=ArmCalibration.model_validate(original.arm_calibration.model_dump() | {
        'status':'provisional','source_kind':'hardware','directions_verified':[False]*7})
    path=tmp_path/'approx.json';cal.save(path)
    argv=['teleop','--leader-port','/dev/serial/by-id/synthetic','--arm-calibration',str(path)]
    monkeypatch.setattr('sys.argv',argv)
    calls=[]
    monkeypatch.setattr(entry,'LeaderReceiver',lambda *a,**kw:calls.append('opened'))
    with pytest.raises(SystemExit) as stopped:entry.main()
    assert stopped.value.code==2 and not calls
    monkeypatch.setattr('sys.argv',argv+['--allow-provisional-calibration'])
    monkeypatch.setattr(entry,'run',lambda args,**kw:calls.append(kw['mapper_factory'].keywords['arm_calibration'].status))
    entry.main()
    assert calls==['opened','provisional']


def test_provisional_start_reanchors_large_seed_mismatch_without_motion(mapper):
    original,q=mapper
    cal=ArmCalibration.model_validate(original.arm_calibration.model_dump() | {
        'status':'provisional','directions_verified':[False]*7})
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=cal)
    raw=[.7,-2.,.4,2.,.2,.1,.3,0.]
    m.set_active(True)
    assert m.command(sample(raw),q,now_ns=time.monotonic_ns())==q
    assert m.armed and not m.recovery_pending
    assert tuple(m.arm_calibration.leader_reference_rad)==tuple(raw[:7])
    assert m.input_mapping['anchor_policy']=='measured_pose_on_Start'
    assert m.input_mapping['space']=='relative_joint_practice'
    raw[0]+=.03
    target=m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert target['fr3_joint1']>q['fr3_joint1']
    m.acknowledge(target);m.set_active(False)
    held=m.command(sample([1.]*8),target,now_ns=time.monotonic_ns())
    assert held==target
    m.set_active(True)
    assert m.command(sample([1.]*8),target,now_ns=time.monotonic_ns())==target
    assert m.armed and not m.recovery_pending
    assert m.arm_calibration.robot_reference_rad==tuple(target[n] for n in m.model.arm.names)
    assert cal.leader_reference_rad==(0.,)*7  # Seed file/object is never rewritten.


def test_limit_hold_names_joint_and_preserves_details_until_restart(mapper):
    m,q=mapper;m.set_active(True);align(m,q)
    raw=np.zeros(8)
    for step in range(1,16):
        raw[3]=step*.1
        m.command(sample(raw),q,now_ns=time.monotonic_ns())
        if not m.armed:break
    assert not m.armed and m.issue['code']=='joint_limit'
    assert m.issue['joint']=='fr3_joint4'
    assert m.issue['target_rad']<m.issue['lower_rad']
    assert 'fr3_joint4' in m.status and 'deg' in m.status
    issue=dict(m.issue)
    assert m.command(sample(raw),q,now_ns=time.monotonic_ns())==q
    assert m.issue==issue
    m.set_active(True)
    assert m.issue is None


def test_practice_only_saturates_limited_axis_and_returns_without_restart(mapper):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     fixed_reference=True,follow_on_start=True,practice_follow=True)
    m.set_active(True)
    raw=np.zeros(8);raw[2]=3.05;raw[0]=.6
    target=m.command(sample(raw),q,now_ns=time.monotonic_ns())
    assert m.armed and m.issue['code']=='joint_saturated'
    assert target['fr3_joint1']>q['fr3_joint1']  # Unaffected joint continues.
    for name,value in target.items():assert abs(value-q[name])<=m.velocity[name]*.02+1e-10
    for _ in range(250):
        m.acknowledge(target)
        target=m.command(sample(raw),target,now_ns=time.monotonic_ns())
    assert target['fr3_joint3']==pytest.approx(m.model.arm.active[2].upper)
    m.acknowledge(target);previous=target
    raw[2]=2.4
    target=m.command(sample(raw),previous,now_ns=time.monotonic_ns())
    assert m.armed and m.issue is None and target['fr3_joint3']<previous['fr3_joint3']
    # A large consumer displacement is slewed, not latched as encoder failure.
    assert abs(target['fr3_joint3']-previous['fr3_joint3'])<=m.velocity['fr3_joint3']*.02+1e-10
    m.acknowledge(target);m.set_active(False)
    assert m.command(sample(np.zeros(8)),target,now_ns=time.monotonic_ns())==target
    assert not m.armed


def test_practice_stale_or_changed_stream_still_holds_until_start(mapper):
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     fixed_reference=True,follow_on_start=True,practice_follow=True)
    m.set_active(True);m.command(sample(np.zeros(8)),q,now_ns=time.monotonic_ns())
    stale=sample(np.zeros(8),time.monotonic_ns()-200_000_000)
    assert m.command(stale,q,now_ns=time.monotonic_ns())==q and not m.armed
    assert m.command(sample(np.zeros(8)),q,now_ns=time.monotonic_ns())==q and not m.armed
    m.set_active(True);m.command(sample(np.zeros(8)),q,now_ns=time.monotonic_ns())
    assert m.command(replace(sample(np.zeros(8)),stream='restarted'),q,now_ns=time.monotonic_ns())==q
    assert not m.armed and m.issue['code']=='encoder_jump'


def test_practice_ui_separates_joint_boundary_from_physical_support(mapper):
    from scripts.teleop.diagnostics import joint_operator_status
    m,_=mapper
    m.issue={'code':'joint_saturated','limits':[{'joint':'fr3_joint3','target_rad':3.,'lower_rad':-2.9,'upper_rad':2.9}]}
    text=joint_operator_status(m,{'latest_age_s':.01},
        {'enabled':False,'reason':'Leader motor 4 speed -3.08 rad/s exceeds 3 rad/s; output stopped'},None)
    assert '第 3 轴限位' in text and '其余轴继续' in text and '实体托举：关闭' in text


def test_ui_reference_request_waits_for_post_click_fresh_sample(mapper,tmp_path):
    from types import SimpleNamespace
    from scripts.teleop.pico_isaac import Controls
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     mode='controllers',calibration_output=tmp_path)
    ui=Controls.__new__(Controls)
    ui.mapper=m
    ui.reference_label=SimpleNamespace(text='')
    ui.capture_on_start=False
    ui.feedback=None
    raw=np.array([.4]*8)
    stamp=time.monotonic_ns()-300_000_000
    m.command(sample(raw,stamp),q,now_ns=stamp)
    ui.capture_reference()
    ui.complete_reference_capture()
    assert ui.reference_request is not None
    assert not list(tmp_path.iterdir())
    m.command(sample(raw),q,now_ns=time.monotonic_ns())
    ui.complete_reference_capture()
    assert ui.reference_request is None
    assert '参考已保存' in ui.reference_label.text
    assert len(list(tmp_path.glob('arm_reference_*.json')))==1
    assert not m.armed
    ui.capture_reference()
    ui.pause()
    assert ui.reference_request is None


def test_ui_reference_request_reports_wrong_pose(mapper,tmp_path):
    from types import SimpleNamespace
    from scripts.teleop.pico_isaac import Controls
    original,q=mapper
    m=JointRetargeter(original.model,q,{'right':'right'},arm_calibration=original.arm_calibration,
                     mode='controllers',calibration_output=tmp_path)
    ui=Controls.__new__(Controls)
    ui.mapper=m
    ui.reference_label=SimpleNamespace(text='')
    ui.capture_on_start=False
    ui.capture_reference()
    wrong=q|{m.model.arm.names[1]:q[m.model.arm.names[1]]+.2}
    m.command(sample(np.array([.4]*8)),wrong,now_ns=time.monotonic_ns())
    ui.complete_reference_capture()
    assert '仿真不在参考姿态' in ui.reference_label.text
    assert not list(tmp_path.iterdir())
