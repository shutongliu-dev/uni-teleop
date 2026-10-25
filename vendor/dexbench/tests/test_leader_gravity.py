import numpy as np
import pytest

from bench.adapters.master_slave.gravity import LeaderGravity
from bench.adapters.master_slave.haptics import GravityCoupling,FeedbackFrame


@pytest.fixture
def gravity(tmp_path):
    # Seven planar pendulums; each centre is 0.5 m beyond its pivot.
    xml=['<robot name="test"><link name="base"><inertial><mass value="0"/></inertial></link>']
    for i in range(7):
        parent='base' if i==0 else f'l{i-1}'
        xml.append(f'<link name="l{i}"><inertial><origin xyz="0.5 0 0"/><mass value="1"/></inertial></link>'
                   f'<joint name="j{i}" type="revolute"><parent link="{parent}"/><child link="l{i}"/>'
                   '<origin xyz="1 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" velocity="5"/></joint>')
    path=tmp_path/'leader.urdf';path.write_text(''.join(xml)+'</robot>')
    return LeaderGravity(path)


def test_gravity_equals_potential_gradient_and_analytic_pendulum(gravity):
    assert gravity.torque(np.zeros(7))[-1]==pytest.approx(-9.81*.5)
    q=np.linspace(-.4,.7,7)
    def potential(q):
        poses=gravity.model.forward(dict(zip(gravity.model.names,q)))
        return sum(m*9.81*(poses[n][:3,:3]@c+poses[n][:3,3])[2] for n,m,c,_ in gravity.bodies)
    for i in range(7):
        dq=np.eye(7)[i]*1e-6
        assert gravity.torque(q)[i]==pytest.approx((potential(q+dq)-potential(q-dq))/2e-6,abs=1e-7)


def test_motor_sign_damping_caps_and_stale_permission(gravity):
    q=np.zeros(7);signs=np.array([1,-1,1,-1,1,-1,1]);conversion=np.ones(7)*1000
    assert np.allclose(gravity.currents(q,q,signs,conversion),gravity.torque(q)*signs*conversion)
    assert np.allclose(gravity.currents(q,np.ones(7),signs,conversion,gain=0),-.02*signs*conversion)
    coupling=GravityCoupling(dict.fromkeys(range(1,8),1220),gravity)
    frame=FeedbackFrame(100_000_000,tuple(q),tuple(q),tuple(signs),tuple(q))
    current=coupling.command(frame,[0.]*8,[0.]*8,sample_ns=100_000_000,now_ns=100_000_000,dt=.01)
    assert coupling.saturated and np.max(np.abs(current))<=1.5
    with pytest.raises(ValueError,match='stale'):
        coupling.command(frame,[0.]*8,[0.]*8,sample_ns=100_000_000,now_ns=201_000_000,dt=.01)
    # A stale renderer does not invalidate a fixed calibration and fresh encoder sample.
    current=coupling.command(frame,[0.]*8,[0.]*8,sample_ns=500_000_000,now_ns=501_000_000,dt=.01)
    assert np.any(current) and np.all(np.abs(current)<=coupling.cap)


def test_start_support_independent_of_follower_and_pause_cancels(gravity):
    from types import SimpleNamespace
    from bench.adapters.master_slave.haptics import HapticReceiver
    receiver=HapticReceiver.__new__(HapticReceiver)
    receiver.coupling=GravityCoupling(dict.fromkeys(range(1,8),1220),gravity)
    receiver.requested=False
    cal=SimpleNamespace(status='measured',leader_reference_rad=(0.,)*7,robot_reference_rad=(0.,)*7,signs=(1,)*7)
    arm=SimpleNamespace(armed=False,alignment_pending=True,arm_calibration=cal,
                        model=SimpleNamespace(arm=SimpleNamespace(names=[f'j{i}' for i in range(7)])))
    obs={'truncated':False,'terminated':False,'joint_positions_rad':dict.fromkeys(arm.model.arm.names,0.)}
    receiver.start_support(arm)
    receiver.observe(arm,obs,now_ns=100)
    assert receiver.requested  # Support starts before simulated joint takeover.
    arm.alignment_pending=False  # Follower held by its own limits; leader is still live.
    receiver.observe(arm,obs,now_ns=101)
    assert receiver.requested
    receiver.request_feedback(False)
    receiver.observe(arm,obs,now_ns=102)
    assert not receiver.requested  # Active simulation alone cannot re-enable output.
    receiver.start_support(arm)
    receiver.observe(arm,dict(obs,truncated=True),now_ns=103)
    assert not receiver.requested
    cal.status='provisional'
    with pytest.raises(ValueError,match='verified'):
        receiver.start_support(arm)


def test_gravity_fault_reports_interval_or_specific_motor(gravity):
    coupling=GravityCoupling(dict.fromkeys(range(1,8),1220),gravity)
    frame=FeedbackFrame(1,(0.,)*7,(0.,)*7,(1.,)*7,())
    with pytest.raises(ValueError,match='123.0 ms exceeds 100 ms'):
        coupling.command(frame,[0.]*8,[0.]*8,sample_ns=1,now_ns=1,dt=.123)
    velocity=[0.]*8;velocity[4]=-3.5
    with pytest.raises(ValueError,match='motor 5 speed -3.50 rad/s'):
        coupling.command(frame,[0.]*8,velocity,sample_ns=1,now_ns=1,dt=.01)


def test_higher_gravity_stage_keeps_slow_ramp_and_bounded_current(gravity):
    coupling=GravityCoupling(dict.fromkeys(range(1,8),1220),gravity,current_scale=2)
    frame=FeedbackFrame(1,(0.,)*7,(0.,)*7,(1.,)*7,())
    previous=np.zeros(7)
    for _ in range(220):
        current=coupling.command(frame,[0.]*8,[0.]*8,sample_ns=1,now_ns=1,dt=.01)
        assert np.max(np.abs(current-previous))<=.3+1e-10
        assert np.max(np.abs(current))<=60
        previous=current
    assert np.max(np.abs(current))==60
    with pytest.raises(ValueError,match='1 or 2'):
        GravityCoupling(dict.fromkeys(range(1,8),1220),gravity,current_scale=3)


@pytest.mark.parametrize('fault',[None,'enable','motion','stale','readback'])
def test_bounded_physical_check_shuts_down_after_success_or_fault(gravity,fault):
    from types import SimpleNamespace
    from scripts.validation.leader_gravity_check import check

    class Clock:
        ns=0
        def monotonic_ns(self):return self.ns
        def sleep(self,seconds):self.ns+=round(seconds*1e9)

    clock=Clock()

    class Driver:
        enabled=False
        disabled=False
        current=np.zeros(7)
        sent=[]
        def inspect(self):return dict.fromkeys(range(1,8),1030)
        def read(self,motor,address,size):
            if address==64:return bytes([self.enabled and motor!=8])
            if address in (102,126):
                raw=round(self.current[motor-1]/2.69)
                if fault=='readback' and address==126 and self.enabled:raw=100
                return raw.to_bytes(2,'little',signed=True)
            return bytes(size)
        def positions(self):
            value=.2 if fault=='motion' and self.enabled else 0.
            stamp=clock.ns-(60_000_000 if fault=='stale' and self.enabled else 0)
            return (value,)*8,stamp
        def enable(self):
            self.enabled=True
            if fault=='enable':raise OSError('Interrupted enable')
        def currents(self,values):
            self.current=np.trunc(np.asarray(values)/2.69)*2.69
            self.sent.append(self.current.copy())
            return self.current.tolist()
        def disable(self):
            self.current=np.zeros(7);self.enabled=False;self.disabled=True

    driver=Driver()
    cal=SimpleNamespace(leader_reference_rad=(0.,)*7,robot_reference_rad=(0.,)*7,signs=(1,)*7)
    result={'samples':[]}
    if fault:
        with pytest.raises((OSError,ValueError)):
            check(driver,cal,gravity,4,result,clock=clock)
        assert 'error' in result and not result.get('completed')
    else:
        check(driver,cal,gravity,4,result,clock=clock)
        assert result['completed'] and result['samples'][-1]['elapsed_s']==pytest.approx(2.)
        currents=np.array(driver.sent)
        assert 0<np.max(np.abs(currents[:,3]))<=60
        assert not np.any(currents[:,[0,1,2,4,5,6]])
    assert driver.disabled and not driver.enabled and result['shutdown_verified']


def test_process_support_uses_separate_physical_calibration():
    from types import SimpleNamespace
    from bench.adapters.master_slave.haptic_process import GravityProcessReceiver
    receiver = object.__new__(GravityProcessReceiver)
    physical = SimpleNamespace(status='measured')
    follower = SimpleNamespace(status='provisional')
    sent = []
    receiver.gravity_calibration = physical
    receiver.connection = SimpleNamespace(send=sent.append)
    mapper = SimpleNamespace(arm_calibration=follower)
    receiver.start_support(mapper)
    assert sent[0][0] == 'start' and sent[0][2] is physical
    assert mapper.arm_calibration is follower
    receiver.gravity_calibration = SimpleNamespace(status='provisional')
    with pytest.raises(ValueError, match='verified physical'):
        receiver.start_support(mapper)
