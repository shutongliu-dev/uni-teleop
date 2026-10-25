"""Official SHARPA coordinate/candidate boundary, without loading vendor binaries."""
import copy
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bench.adapters.manus.sharpa import SEMANTICS, optimizer_points, candidate_targets, kinematic_signature
from bench.adapters.manus.sources import host_clock_id
from bench.adapters.master_slave.manus import ManusJointRetargeter
from bench.adapters.pico.protocol import Frame
from bench.control.factory import public_robot_geometry, robot_configuration
from scenes.workrooms.catalog import load_task


def packet(sequence=0):
    return dict(protocol='manus_acquisition_v1',stream_id='synthetic',sequence=sequence,
        capture_time_ns=time.monotonic_ns(),clock_id=host_clock_id(),calibration_id='test',
        topology_id='synthetic',frame_id='manus_world',position_unit='m',node_space='world',
        axes='right_handed_z_up_x_forward',glove_tracking_valid=True,
        device=dict(device_id='7',product='synthetic',firmware='none',sdk_version='none',
                    core_version='none',hand='right',source_kind='synthetic'),
        nodes=[dict(node_id=str(i),parent_id='0' if i else None,semantic=s,valid=True,
            pose=dict(position=[i*.003,i*.002,i*.001],quaternion_wxyz=[1.,0.,0.,0.]))
               for i,s in enumerate(SEMANTICS)])


def test_official_input_uses_semantics_and_wrist_relative_coordinate_transform():
    p=packet();points=optimizer_points(p)
    assert np.allclose(points[1,:3],[-.001,.002,.003])
    assert np.allclose(points[:,3:],[1,0,0,0])
    p['nodes'].reverse()
    assert np.allclose(optimizer_points(p),points)
    turn=Rotation.from_euler('xyz',[.5,-.3,.2]);translation=np.array([1.,2.,3.])
    for node in p['nodes']:
        pose=node['pose'];pose['position']=(turn.apply(pose['position'])+translation).tolist()
        pose['quaternion_wxyz']=turn.as_quat()[[3,0,1,2]].tolist()
    assert np.allclose(optimizer_points(p),points)


@pytest.fixture
def model():
    task=load_task('mechanical_assembly','peg_insertion')
    return public_robot_geometry(robot_configuration(task['scene'])),task['initial_state']['joint_positions_rad']


def candidate(p,m):
    return dict({k:p[k] for k in ('stream_id','sequence','capture_time_ns','clock_id',
                                 'calibration_id','topology_id')},protocol='sharpa_optimizer_v1',
        executable=True,device_id='7',joint_positions_rad=dict.fromkeys(m.hand.names,0.))


@pytest.mark.parametrize('change', ['replay','clock','names','expired'])
def test_candidate_identity_replay_and_expiry_hold(model,change):
    m,_=model;p=packet();c=candidate(p,m)
    if change=='expired':
        c['capture_time_ns']-=200_000_000
        assert candidate_targets(c,p,m,now_ns=time.monotonic_ns()) is None
        return
    if change=='replay':c['executable']=False
    elif change=='clock':c['clock_id']='other_boot'
    else:c['joint_positions_rad'].pop(next(iter(c['joint_positions_rad'])))
    with pytest.raises(ValueError):candidate_targets(c,p,m,now_ns=time.monotonic_ns())


def test_official_targets_drive_thumb_and_keep_arm_held_with_watchdogs(model):
    m,q=model
    meta=dict(joint_names=m.hand.names,kinematic_signature=kinematic_signature(m.hand_path))
    mapper=ManusJointRetargeter(m,q,{'right':'right'},glove_id=7,
        calibration_id='test',mode='controllers',hand_only=True,optimizer_metadata=meta)
    mapper.set_active(True)
    p=packet();c=candidate(p,m);c['joint_positions_rad']['right_thumb_MCP_FE']=.9
    def frame(p,c):
        return Frame('test',p['sequence'],p['capture_time_ns'],'','synthetic',
            dict(head=None,manus=p,manus_retarget=c,leader=[],leader_host_ns=0))
    target=mapper.command(frame(p,c),q,now_ns=time.monotonic_ns())
    assert target['right_thumb_MCP_FE']>q['right_thumb_MCP_FE']
    assert all(target[n]==q[n] for n in m.arm.names)
    assert mapper.calibration is None  # Official mapping requires no PICO open-palm anchor.
    mapper.acknowledge(target)
    p=packet(1);c=candidate(p,m);c['capture_time_ns']-=200_000_000
    assert mapper.command(frame(p,c),target,now_ns=time.monotonic_ns())==target
    assert mapper.hand_enabled
    p=packet(2);p['capture_time_ns']-=200_000_000
    assert mapper.command(frame(p,candidate(p,m)),target,now_ns=time.monotonic_ns())==target
    assert not mapper.hand_enabled


def test_optimizer_kinematics_must_match_pinned_hand(model):
    m,q=model
    with pytest.raises(ValueError,match='kinematics'):
        ManusJointRetargeter(m,q,{'right':'right'},glove_id=7,calibration_id='test',
            optimizer_metadata=dict(joint_names=m.hand.names,kinematic_signature='changed'))


def test_invalid_vendor_quaternion_is_not_an_optimizer_input():
    p=packet();p['nodes'][4]['pose']['quaternion_wxyz']=[0.,0.,0.,0.]
    with pytest.raises(ValueError,match='unit wxyz'):
        optimizer_points(p)


@pytest.mark.parametrize('lost', ['pico','manus','grip','none'])
def test_pico_manus_channels_are_independent(model,lost):
    from bench.adapters.manus.pico_teleop import PicoManusReceiver, PicoManusRetargeter
    from types import SimpleNamespace
    m,q=model
    meta=dict(joint_names=m.hand.names,kinematic_signature=kinematic_signature(m.hand_path))
    mapper=PicoManusRetargeter(m,q,{'right':'right'},glove_id=7,calibration_id='test',
                              optimizer_metadata=meta,mode='controllers')
    def combined(seq,x=0.,grip=1.,pico_age=0.,hand_age=0.):
        now=time.monotonic_ns();p=packet(seq);p['capture_time_ns']=now-int(hand_age*1e9)
        c=candidate(p,m);c['joint_positions_rad']['right_thumb_MCP_FE']=.9
        arm=Frame('pico',seq,now-int(pico_age*1e9),str(seq),'synthetic',
                  dict(head=None,hands={},controllers={'right':dict(pose=[x,0.,0.,0.,0.,0.,1.],grip=grip,trigger=1.)}))
        receiver=PicoManusReceiver(SimpleNamespace(read=lambda:arm),SimpleNamespace(latest=p),SimpleNamespace(latest=c))
        f=receiver.read()
        assert f.tracking['pico_host_ns']==arm.host_ns
        return f
    mapper.set_active(True)
    first=mapper.command(combined(0),q,now_ns=time.monotonic_ns());mapper.acknowledge(first)
    f=combined(1,x=.005,grip=0. if lost=='grip' else 1.,
               pico_age=.2 if lost=='pico' else 0.,hand_age=.2 if lost=='manus' else 0.)
    target=mapper.command(f,first,now_ns=time.monotonic_ns())
    arm_moved=any(abs(target[n]-first[n])>1e-6 for n in m.arm.names)
    fingers_moved=target['right_thumb_MCP_FE']>first['right_thumb_MCP_FE']
    assert arm_moved == (lost not in ('pico','grip'))
    assert fingers_moved == (lost!='manus')
    # PICO trigger cannot write any finger target, including when MANUS is lost.
    if lost=='manus':
        assert all(target[n]==first[n] for n in m.hand.names)
    mapper.acknowledge(target);mapper.set_active(False)
    assert mapper.command(combined(2,x=.01),target,now_ns=time.monotonic_ns())==target
    assert not mapper.session_active
