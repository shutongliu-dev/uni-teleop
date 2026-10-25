"""Feedback sign, strict output bounds and serial zero-before-enable contracts."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from uni_teleop.feedback.factr.haptics import (ARM_IDS,CAP_MA,CURRENT_MA,FeedbackFrame,
    HapticDriver,HapticReceiver,VirtualCoupling)


MODELS={i:1030 if i in (2,4) else 1220 for i in ARM_IDS}


def constant_gravity_current(*args):
    return np.full(7,-100.)


def feedback(sign=1,age=0.):
    return FeedbackFrame(time.monotonic_ns()-int(age*1e9),(0.,)*7,(0.,)*7,(sign,)*7,(0.,)*7)


@pytest.mark.parametrize('sign',[-1,1])
def test_feedback_resists_encoder_displacement_for_both_mapping_signs(sign):
    controller=VirtualCoupling(MODELS)
    current=controller.command(feedback(sign),[.1]*8,[0.]*8,now_ns=time.monotonic_ns(),dt=.01)
    assert np.all(current<0)
    current2=controller.command(feedback(sign),[-.1]*8,[0.]*8,now_ns=time.monotonic_ns(),dt=.01)
    assert np.all(current2>current)


def test_following_robot_has_no_spring_force_and_all_outputs_slew_and_amplitude_bound():
    controller=VirtualCoupling(MODELS)
    f=feedback()
    f=FeedbackFrame(f.stamp_ns,f.leader_reference,f.robot_reference,f.signs,(.1,)*7)
    assert not controller.command(f,[.1]*8,[0.]*8,now_ns=time.monotonic_ns(),dt=.01).any()
    previous=np.zeros(7)
    for _ in range(80):
        value=controller.command(feedback(),[.29]*8,[0.]*8,now_ns=time.monotonic_ns(),dt=.01)
        assert np.all(np.abs(value)<=controller.cap)
        assert np.all(np.abs(value-previous)<=controller.cap*.05+1e-10)
        previous=value


@pytest.mark.parametrize('kind',['stale','future','nan','speed','error','dt'])
def test_invalid_or_stale_feedback_never_produces_output(kind):
    controller=VirtualCoupling(MODELS);f=feedback(age=.11 if kind=='stale' else -.1 if kind=='future' else 0.)
    q=[float('nan') if kind=='nan' else .31 if kind=='error' else 0.]*8
    with pytest.raises(ValueError):
        controller.command(f,q,[3.1 if kind=='speed' else 0.]*8,now_ns=time.monotonic_ns(),dt=.2 if kind=='dt' else .01)


def driver(monkeypatch):
    d=HapticDriver.__new__(HapticDriver);d.models=MODELS;d.fd=9876
    packets=[]
    monkeypatch.setattr('uni_teleop.feedback.factr.haptics.os.write',lambda fd,data:packets.append(data) or len(data))
    return d,packets


def writes(packets):
    return [(int.from_bytes(p[8:10],'little'),int.from_bytes(p[10:12],'little'),p[12:-2]) for p in packets]


def test_current_units_caps_and_trigger_exclusion(monkeypatch):
    d,packets=driver(monkeypatch)
    actual=d.currents([1000,-1000,1000,-1000,1000,-1000,1000])
    assert actual==[30.,-22*2.69,30.,-22*2.69,30.,-30.,30.]
    address,size,values=writes(packets)[0]
    assert (address,size)==(102,2)
    assert list(values[::3])==list(ARM_IDS)
    with pytest.raises(ValueError):d._sync_write(11,1,dict.fromkeys(ARM_IDS,0))
    with pytest.raises(ValueError):d._sync_write(102,2,dict.fromkeys(ARM_IDS,1000))
    with pytest.raises(ValueError):d.currents([float('nan')]*7)


def test_higher_gravity_cap_is_instance_local_and_bounded(monkeypatch):
    d,packets=driver(monkeypatch)
    d.caps={model:cap*2 for model,cap in CAP_MA.items()}
    assert d.currents([1000.]*7)==[60.,44*2.69,60.,44*2.69,60.,60.,60.]
    with pytest.raises(ValueError):d._sync_write(102,2,dict.fromkeys(ARM_IDS,1000))
    base,_=driver(monkeypatch)
    assert base.currents([1000.]*7)==[30.,22*2.69,30.,22*2.69,30.,30.,30.]


def test_enable_requires_zero_readback_before_any_torque_on(monkeypatch):
    d,packets=driver(monkeypatch)
    reads=[]
    def read(i,address,size):
        reads.append((i,address))
        if address==102:
            assert all(not (a==64 and any(data[1::2])) for a,_,data in writes(packets))
            return b'\0\0'
        return b'\1'
    d.read=read;d.enable()
    assert reads==[(i,102) for i in ARM_IDS]+[(i,64) for i in ARM_IDS]
    assert [a for a,_,_ in writes(packets)]==[64,98,102,98,64]
    d.disable()
    assert [a for a,_,_ in writes(packets)[-3:]]==[102,64,98]


def test_failed_zero_readback_cannot_enable_torque(monkeypatch):
    d,packets=driver(monkeypatch);d.read=lambda *args:b'\1\0'
    with pytest.raises(ValueError,match='zero-current'):d.enable()
    assert all(not (a==64 and any(data[1::2])) for a,_,data in writes(packets))


def test_pause_and_invalid_native_state_cancel_feedback():
    receiver=HapticReceiver.__new__(HapticReceiver)
    receiver.coupling=VirtualCoupling(MODELS)
    receiver.requested=True;receiver.feedback='previous'
    mapper=SimpleNamespace(armed=False,alignment_pending=False)
    receiver.observe(mapper,{},now_ns=time.monotonic_ns())
    assert not receiver.requested and receiver.feedback is None
    receiver.requested=True
    mapper.armed=True
    receiver.observe(mapper,{'truncated':True},now_ns=time.monotonic_ns())
    assert not receiver.requested and receiver.feedback is None


@pytest.fixture
def motor_bus():
    """A PTY device checks real thread/packet behavior; never opens hardware."""
    import os,pty,select,struct,threading
    from uni_teleop.devices.factr.dynamixel import HEADER,crc,packet
    master,slave=pty.openpty();stop=threading.Event();errors=[];history=[]
    responding=threading.Event();responding.set()
    registers={i:{0:MODELS.get(i,1220),6:53,11:0,38:910,64:0,70:0,98:0,102:910} for i in range(1,9)}
    def device():
        buffer=bytearray()
        try:
            while not stop.is_set():
                if not select.select([master],[],[],.01)[0]:continue
                buffer.extend(os.read(master,4096))
                while len(buffer)>=7:
                    assert buffer[:4]==HEADER
                    size=7+int.from_bytes(buffer[5:7],'little')
                    if len(buffer)<size:break
                    request=bytes(buffer[:size]);del buffer[:size]
                    assert crc(request[:-2])==int.from_bytes(request[-2:],'little')
                    identity=request[4];instruction=request[7];body=request[8:-2]
                    address,length=struct.unpack('<HH',body[:4]) if instruction in (0x82,0x83) else (None,None)
                    if instruction==2:
                        address,length=struct.unpack('<HH',body)
                        value=registers[identity].get(address,0)
                        os.write(master,packet(identity,0x55,b'\0'+int(value).to_bytes(length,'little')))
                    elif instruction==0x82:
                        assert (address,length)==(132,4)
                        if responding.is_set():
                            for i in body[4:]:os.write(master,packet(i,0x55,b'\0'+struct.pack('<i',0)))
                    elif instruction==0x83:
                        for at in range(4,len(body),length+1):
                            i=body[at];value=int.from_bytes(body[at+1:at+length+1],'little',signed=address==102)
                            history.append((i,address,value))
                            if address==64 and value==1:assert registers[i][102]==0
                            registers[i][address]=value
                    else:raise AssertionError('Unexpected instruction')
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=device,daemon=True);thread.start()
    try:
        yield os.ttyname(slave),registers,history,responding
    finally:
        stop.set();thread.join(1);os.close(master);os.close(slave)
    assert not errors


@pytest.mark.parametrize('mode',['virtual','gravity_pause','gravity_bus','process_pause','process_bus','process_terminate','process_stage2_pause'])
def test_live_serial_worker_freshness_and_stop(mode,motor_bus):
    port,registers,history,responding=motor_bus
    receiver=None
    try:
        gravity=None if mode=='virtual' else SimpleNamespace(sha256='synthetic',currents=constant_gravity_current)
        from uni_teleop.feedback.factr.haptic_process import GravityProcessReceiver
        receiver_class=GravityProcessReceiver if mode.startswith('process_') else HapticReceiver
        receiver=receiver_class(port,gravity=gravity,gravity_current_scale=2 if 'stage2' in mode else 1)
        assert not history  # Constructor performs no hardware writes.
        if mode=='virtual':
            receiver.feedback=FeedbackFrame(time.monotonic_ns(),(0.,)*7,(0.,)*7,(1,)*7,(-.1,)*7)
            receiver.request_feedback(True)
        else:
            receiver.start_support(SimpleNamespace(armed=False,alignment_pending=True,
                arm_calibration=SimpleNamespace(status='measured',leader_reference_rad=(0.,)*7,
                    robot_reference_rad=(0.,)*7,signs=(1,)*7)))
        deadline=time.monotonic()+1.
        while time.monotonic()<deadline and not any(a==102 and v<0 for _,a,v in history):
            # Keep simulation feedback fresh until first bounded nonzero output.
            if mode=='virtual':
                receiver.feedback=FeedbackFrame(time.monotonic_ns(),(0.,)*7,(0.,)*7,(1,)*7,(-.1,)*7)
            time.sleep(.005)
        assert any(a==102 and v<0 for _,a,v in history)
        reads=receiver.health()['received']
        if mode!='virtual':
            time.sleep(.25)  # No new simulation observation: gravity must continue.
            assert receiver.feedback_state['enabled'] and receiver.requested and receiver.health()['received']>reads
            if mode.endswith('_pause'):receiver.request_feedback(False)
            elif mode.endswith('_terminate'):receiver.process.terminate()
            else:responding.clear()
        deadline=time.monotonic()+.8
        while time.monotonic()<deadline and (receiver.requested or receiver.feedback_state['enabled'] or
                any(registers[i][64] or registers[i][102] for i in ARM_IDS)):time.sleep(.01)
        assert not receiver.requested and not receiver.feedback_state['enabled']
        assert receiver.health()['received']>=reads
        if mode=='virtual':assert 'stale' in receiver.feedback_reason
        elif mode=='gravity_bus':assert 'timed out' in receiver.feedback_reason
        assert all(registers[i][64]==0 and registers[i][102]==0 for i in ARM_IDS)
        assert not any(i==8 for i,_,_ in history)
    finally:
        if receiver:receiver.close()


def test_higher_stage_rejected_if_motor_hardware_cap_is_lower(motor_bus):
    port,registers,history,_=motor_bus
    registers[2][38]=30  # XM430: 80.7 mA, below the requested 120 mA.
    with pytest.raises(ValueError,match='Motor 2: requested current cap exceeds hardware limit'):
        HapticReceiver(port,gravity=SimpleNamespace(),gravity_current_scale=2)
    assert not history  # No mode, current or enable writes to "fix" the limit.


def isolated_control_client(port, connection):
    """Simulate an Isaac owner that blocks its GIL, then exits without cleanup."""
    import ctypes
    import os
    from uni_teleop.feedback.factr.haptic_process import GravityProcessReceiver
    receiver=GravityProcessReceiver(port,
        gravity=SimpleNamespace(sha256='synthetic',currents=constant_gravity_current))
    receiver.start_support(SimpleNamespace(arm_calibration=SimpleNamespace(status='measured',
        leader_reference_rad=(0.,)*7,robot_reference_rad=(0.,)*7,signs=(1,)*7)))
    connection.send('ready')
    assert connection.recv()=='stall'
    connection.send('blocking')
    ctypes.PyDLL(None).usleep(400_000)  # Holds this process's GIL, including its reader.
    connection.send('resumed')
    assert connection.recv()=='exit'
    os._exit(0)  # Exercise parent-death cleanup, not graceful receiver.close().


def test_owner_gil_stall_keeps_output_and_owner_death_disables(motor_bus):
    import multiprocessing as mp
    port,registers,history,_=motor_bus
    context=mp.get_context('spawn');connection,child=context.Pipe()
    owner=context.Process(target=isolated_control_client,args=(port,child))
    owner.start();child.close()
    try:
        assert connection.poll(10) and connection.recv()=='ready'
        deadline=time.monotonic()+1
        while not any(a==102 and v<0 for _,a,v in history) and time.monotonic()<deadline:
            time.sleep(.005)
        assert all(registers[i][64]==1 for i in ARM_IDS)
        connection.send('stall')
        assert connection.poll(1) and connection.recv()=='blocking'
        start=len(history)
        time.sleep(.2)
        assert len(history)>start+7  # Currents continue while the parent's GIL is blocked.
        assert all(registers[i][64]==1 for i in ARM_IDS)
        assert connection.poll(1) and connection.recv()=='resumed'
        connection.send('exit');owner.join(2)
        assert owner.exitcode==0
        deadline=time.monotonic()+2
        while any(registers[i][64] or registers[i][102] for i in ARM_IDS) and time.monotonic()<deadline:
            time.sleep(.01)
        assert all(registers[i][64]==0 and registers[i][102]==0 for i in ARM_IDS)
        assert not any(i==8 for i,_,_ in history)
    finally:
        if owner.is_alive():owner.terminate();owner.join(2)
        connection.close()
