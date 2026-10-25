"""Real MuJoCo dynamics with synthetic device packets; no hardware reader is opened."""
import os
import time
import numpy as np
import pytest
import mujoco
from bench.control.factory import public_robot_geometry, robot_configuration
from scenes.teleop_sandbox import inputs
from uni_teleop.backends.mujoco.franka import FrankaWorkbench
from uni_teleop.devices.pico.protocol import Frame
from uni_teleop.mapping.franka.retarget import Retargeter
from uni_teleop.mapping.factr.retarget import JointRetargeter
from uni_teleop.mapping.factr.calibration import ArmCalibration
from uni_teleop.mapping.factr_manus import ManusJointRetargeter
from uni_teleop.mapping.pico_manus import PicoManusRetargeter
from uni_teleop.mapping.manus.sharpa import kinematic_signature
from test_pico_teleop import frame, tracked_hand
from test_master_slave_manus import packet as glove_packet, frame as leader_frame
from test_manus_sharpa import candidate


@pytest.fixture
def robot():
    scene, _, _ = inputs()
    model = public_robot_geometry(robot_configuration(scene))
    q = dict(zip(model.names, np.clip(np.zeros(29), model.lower, model.upper)))
    q.update(zip(model.arm.names, scene['robot']['arm_home_rad']))
    env = FrankaWorkbench(os.environ['DEXBENCH_DATA']+'/assets', model, q)
    yield model, q, env
    env.close()


def test_fk_and_physical_drive(robot):
    model, q, env = robot
    arm = model.arm.forward(q, model.arm_base)
    hand = model.hand.forward(q, arm[model.arm_tip] @ model.wrist_offset)
    for name, expected in {**arm, **hand}.items():
        try: native = env.data.body(name)
        except KeyError: continue
        np.testing.assert_allclose(native.xpos, expected[:3,3], atol=2e-6)
        np.testing.assert_allclose(native.xmat.reshape(3,3), expected[:3,:3], atol=2e-6)
    for _ in range(100): env.step(q)
    start = env.positions()
    target = {**q, 'fr3_joint1':q['fr3_joint1']+.1}
    for _ in range(100): env.step(target)
    assert env.positions()['fr3_joint1']-start['fr3_joint1'] > .07
    assert np.isfinite(env.data.qpos).all()


@pytest.mark.parametrize('route',['factr','factr-manus','pico-controllers','pico-hands','pico-manus','manus'])
def test_device_mapping_reaches_native_joints(robot, route):
    model, q, env = robot
    cal = ArmCalibration.from_reference(model, leader_id='synthetic', leader_reference_rad=[0.]*7,
        robot_reference_rad=[q[n] for n in model.arm.names], signs=[1]*7,
        directions_verified=[True]*7, source_kind='synthetic')
    kw = dict(mode='hands' if route=='pico-hands' else 'controllers')
    meta = dict(joint_names=model.hand.names,kinematic_signature=kinematic_signature(model.hand_path))
    if 'manus' in route:
        cls = PicoManusRetargeter if route=='pico-manus' else ManusJointRetargeter
        kw.update(glove_id=7,calibration_id='test',optimizer_metadata=meta)
        if cls is ManusJointRetargeter:
            kw.update(arm_calibration=cal,fixed_reference=True,follow_on_start=True,hand_only=route=='manus')
    elif route=='factr':
        cls=JointRetargeter;kw.update(arm_calibration=cal,fixed_reference=True,follow_on_start=True)
    else: cls=Retargeter
    mapper=cls(model,q,{'right':'right'},**kw);mapper.set_active(True)
    changed=False
    for seq in range(35):
        if route.startswith('factr') or route=='manus':
            f=leader_frame([min(seq*.002,.05)]+[0.]*7,None)
        elif route=='pico-hands':
            hand=tracked_hand(model,q)
            for pose in hand: pose[0]+=min(seq*.0005,.01)
            f=frame('hands',joints=hand,sequence=seq)
        else: f=frame(x=min(seq*.0005,.01),sequence=seq)
        if 'manus' in route:
            p=glove_packet(model,q,seq)
            c=candidate(p,model)
            c['joint_positions_rad']={n:float(np.clip(q[n]+(min(seq*.01,.2) if n.endswith(('FE','PIP','DIP','IP')) else 0),model.by_name[n].lower,model.by_name[n].upper)) for n in model.hand.names}
            f.tracking.update(manus=p,manus_retarget=c,pico_present=True,pico_host_ns=f.host_ns)
        targets=mapper.command(f,env.positions(),now_ns=time.monotonic_ns())
        changed |= any(abs(targets[n]-q[n])>1e-4 for n in q)
        env.step(targets);mapper.acknowledge(targets)
    assert changed, (route, mapper.status)
    assert np.isfinite(env.data.qpos).all()
