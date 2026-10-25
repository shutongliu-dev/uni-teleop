"""Bounded joint resistance or model-based gravity support for X-series leaders.

Neither mode measures external torque. Hardware output requires explicit enable;
gravity mode additionally requires a verified physical reference and leader URDF.
"""
from dataclasses import dataclass
from collections import deque
import os
import struct
import threading
import time

import numpy as np

from uni_teleop.devices.factr.dynamixel import ReadOnlyLeader, packet
from uni_teleop.devices.factr.input import LeaderReceiver


CURRENT_MA = {1220:1., 1030:2.69}  # ROBOTIS XC330-T288 / XM430-W210.
CAP_MA = {1220:30., 1030:60.}
ARM_IDS = tuple(range(1,8))


class HapticDriver(ReadOnlyLeader):
    """Whitelist RAM writes only; EEPROM and operating-mode writes are absent."""
    caps=CAP_MA

    def inspect(self):
        self.models = {i:int.from_bytes(self.read(i,0,2),'little') for i in ARM_IDS}
        for i,model in self.models.items():
            if (model not in CURRENT_MA or self.read(i,11,1)!=b'\0'
                    or self.read(i,64,1)!=b'\0' or self.read(i,70,1)!=b'\0'
                    or self.read(i,6,1)[0]<38):
                raise ValueError(f'Motor {i}: require supported model, current mode, torque off, healthy watchdog firmware')
            limit=int.from_bytes(self.read(i,38,2),'little')*CURRENT_MA[model]
            if self.caps[model]>limit:
                raise ValueError(f'Motor {i}: requested current cap exceeds hardware limit {limit:.1f} mA')
        return self.models

    def _sync_write(self, address, size, values):
        if (address,size) not in ((64,1),(98,1),(102,2)) or set(values)!=set(ARM_IDS):
            raise ValueError('Only seven-arm RAM current/torque/watchdog writes are allowed')
        params=struct.pack('<HH',address,size)
        for i in ARM_IDS:
            value=values[i]
            if address==64 and value not in (0,1):raise ValueError('Invalid torque enable')
            if address==98 and value not in (0,5):raise ValueError('Invalid watchdog')
            if address==102 and abs(value)>int(self.caps[self.models[i]]/CURRENT_MA[self.models[i]]):
                raise ValueError('Current exceeds low-force cap')
            params+=bytes([i])+int(value).to_bytes(size,'little',signed=address==102)
        os.write(self.fd,packet(0xfe,0x83,params))

    def currents(self, milliamps):
        values=np.asarray(milliamps,dtype=float)
        if values.shape!=(7,) or not np.isfinite(values).all():
            raise ValueError('Require seven finite motor currents')
        raw={i:int(np.clip(values[i-1],-self.caps[self.models[i]],self.caps[self.models[i]]) /
                   CURRENT_MA[self.models[i]]) for i in ARM_IDS}
        self._sync_write(102,2,raw)
        return [raw[i]*CURRENT_MA[self.models[i]] for i in ARM_IDS]

    def enable(self):
        # Startup goal-current registers can contain the full rated current.
        self._sync_write(64,1,dict.fromkeys(ARM_IDS,0))
        self._sync_write(98,1,dict.fromkeys(ARM_IDS,0))
        self.currents(np.zeros(7))
        for i in ARM_IDS:
            if self.read(i,102,2)!=b'\0\0':raise ValueError(f'Motor {i} failed zero-current readback')
        self._sync_write(98,1,dict.fromkeys(ARM_IDS,5))
        self._sync_write(64,1,dict.fromkeys(ARM_IDS,1))
        for i in ARM_IDS:
            if self.read(i,64,1)!=b'\1':raise ValueError(f'Motor {i} torque enable not acknowledged')

    def disable(self):
        try:
            self.currents(np.zeros(7))
        finally:
            # Torque-off remains writable even after a hardware watchdog trip.
            self._sync_write(64,1,dict.fromkeys(ARM_IDS,0))
            self._sync_write(98,1,dict.fromkeys(ARM_IDS,0))


@dataclass(frozen=True)
class FeedbackFrame:
    stamp_ns:int
    leader_reference:tuple
    robot_reference:tuple
    signs:tuple
    follower:tuple


class VirtualCoupling:
    def __init__(self, models):
        self.cap=np.array([CAP_MA[models[i]] for i in ARM_IDS])
        self.gain=self.cap/.3
        self.current=np.zeros(7)

    def command(self, frame, raw, velocity, *, now_ns, dt):
        if frame is None or not 0 <= now_ns-frame.stamp_ns <= 100_000_000:
            raise ValueError('Follower feedback missing/stale (>100 ms)')
        values=np.array([raw[:7],velocity[:7],frame.leader_reference,frame.robot_reference,
                         frame.signs,frame.follower],dtype=float)
        if values.shape!=(6,7) or not np.isfinite(values).all() or not 0<dt<=.1:
            raise ValueError('Invalid feedback state or control interval')
        q,qd,leader_ref,robot_ref,signs,follower=values
        if not np.all(np.abs(signs)==1):raise ValueError('Invalid mapping signs')
        mapped=robot_ref+signs*(q-leader_ref)
        error=np.arctan2(np.sin(follower-mapped),np.cos(follower-mapped))
        if np.max(np.abs(error))>.3 or np.max(np.abs(qd))>3.:
            raise ValueError('Feedback tracking error/speed bound; re-enable manually')
        # Virtual work: transpose of q_follower = sign*q_leader + offset.
        deadband=np.sign(error)*np.maximum(np.abs(error)-.01,0.)
        wanted=self.gain*(signs*deadband-.02*qd)
        wanted=np.clip(wanted,-self.cap,self.cap)
        filtered=self.current+dt/(.03+dt)*(wanted-self.current)
        self.current+=np.clip(filtered-self.current,-self.cap*5*dt,self.cap*5*dt)
        return self.current.copy()


class GravityCoupling(VirtualCoupling):
    """Gravity plus damping under the same low-current hardware envelope."""
    def __init__(self,models,gravity,*,current_scale=1):
        if current_scale not in (1,2):raise ValueError('Gravity current scale must be 1 or 2')
        super().__init__(models)
        self.slew_ma_s=self.cap.copy()  # One base-cap per second, even at the higher stage.
        self.cap*=current_scale
        self.gravity=gravity
        # FACTR upstream constants converted from raw goal-current to actual mA/Nm.
        self.ma_per_nm=np.array([1000. if models[i]==1030 else 1158.73 for i in ARM_IDS])
        self.saturated=False
        self.demand_ma=np.zeros(7)

    def command(self,frame,raw,velocity,*,sample_ns,now_ns,dt):
        # Gravity needs current leader state and fixed calibration, not a fresh
        # simulated follower. The serial worker owns this input timestamp.
        if frame is None or not 0<=now_ns-sample_ns<=100_000_000:
            raise ValueError('Leader gravity input missing/stale (>100 ms)')
        values=np.array([raw[:7],velocity[:7],frame.leader_reference,frame.robot_reference,frame.signs])
        if values.shape!=(5,7) or not np.isfinite(values).all():
            raise ValueError('Invalid leader gravity state')
        if not 0<dt<=.1:
            raise ValueError(f'Leader control interval {dt*1000:.1f} ms exceeds 100 ms; output stopped')
        q,qd,reference,model_reference,signs=values
        axis=int(np.argmax(np.abs(qd)))
        if abs(qd[axis])>3.:
            raise ValueError(f'Leader motor {axis+1} speed {qd[axis]:+.2f} rad/s exceeds 3 rad/s; output stopped')
        wanted=self.gravity.currents(model_reference+signs*(q-reference),signs*qd,signs,self.ma_per_nm)
        self.demand_ma=wanted.copy()
        self.saturated=bool(np.any(np.abs(wanted)>self.cap))
        wanted=np.clip(wanted,-self.cap,self.cap)
        self.current+=np.clip(wanted-self.current,-self.slew_ma_s*dt,self.slew_ma_s*dt)
        return self.current.copy()


class HapticReceiver(LeaderReceiver):
    """Exclusive serial owner: reads, output watchdog and RAM writes share one thread."""
    def __init__(self, port, *, baud=4000000, gravity=None, gravity_current_scale=1):
        from pathlib import Path
        if gravity_current_scale not in (1,2) or (gravity is None and gravity_current_scale!=1):
            raise ValueError('Higher current stage requires gravity; scale must be 1 or 2')
        self.leader_id=Path(port).name
        self.driver=HapticDriver(port,baud=baud,timeout=.05)
        self.driver.caps={model:cap*gravity_current_scale for model,cap in CAP_MA.items()}
        try:
            models=self.driver.inspect()
            self.coupling=GravityCoupling(models,gravity,current_scale=gravity_current_scale) if gravity else VirtualCoupling(models)
            values,stamp=self.driver.positions()
        except BaseException:
            self.driver.close();raise
        self.latest=self.error=None
        self.sequence=self.rejected=0
        self.stop=threading.Event()
        self.requested=self.enabled=False
        self.feedback=None
        self.feedback_reason='Off; explicitly enable after Start'
        self.current_ma=[0.]*7
        self.control_interval_s=self.max_control_interval_s=0.
        self.recent_motion=deque(maxlen=16)
        self.last_fault=None
        self._publish(values,stamp)
        self.thread=threading.Thread(target=self._receive,daemon=True)
        self.thread.start()

    def request_feedback(self, enabled):
        self.requested=bool(enabled)

    def start_support(self, mapper):
        if not isinstance(self.coupling,GravityCoupling):return
        arm=getattr(mapper,'leader',mapper)
        cal=arm.arm_calibration
        if cal is None or cal.status!='measured':
            raise ValueError('Leader gravity requires verified physical calibration')
        self.feedback=FeedbackFrame(time.monotonic_ns(),cal.leader_reference_rad,
            cal.robot_reference_rad,cal.signs,())
        self.feedback_reason='Gravity requested; checking fresh leader input'
        self.requested=True

    def observe(self, mapper, observation, *, now_ns):
        if isinstance(self.coupling,GravityCoupling):
            # Follower synchronization/limits cannot decide whether the physical
            # leader needs gravity support. Its worker enforces local input bounds.
            if observation['truncated'] or observation['terminated']:
                self.request_feedback(False)
            return
        arm=getattr(mapper,'leader',mapper)
        if not arm.armed or observation['truncated'] or observation['terminated']:
            self.requested=False
            self.feedback=None
            return
        cal=arm.arm_calibration
        self.feedback=FeedbackFrame(now_ns,cal.leader_reference_rad,cal.robot_reference_rad,
            cal.signs,tuple(observation['joint_positions_rad'][n] for n in arm.model.arm.names))

    def _disable(self, reason):
        self.requested=False
        self.feedback_reason=reason
        try:
            if self.enabled:self.driver.disable()
        finally:
            self.enabled=False
            self.coupling.current[:]=0.
            self.current_ma=[0.]*7

    def _receive(self):
        previous=self.latest
        try:
            while not self.stop.is_set():
                try:
                    values,stamp=self.driver.positions()
                    dt=(stamp-previous.host_ns)/1e9
                    self.control_interval_s=dt
                    self.max_control_interval_s=max(self.max_control_interval_s,dt)
                    velocity=(np.asarray(values)-previous.tracking['leader'])/dt
                    self.recent_motion.append({'stamp_ns':stamp,'interval_s':dt,
                                               'velocity_rad_s':velocity[:7].tolist()})
                    self._publish(values,stamp)
                    previous=self.latest
                    if self.requested:
                        current=self.coupling.command(self.feedback,values,velocity,
                            now_ns=time.monotonic_ns(),dt=dt,
                            **({'sample_ns':stamp} if isinstance(self.coupling,GravityCoupling) else {}))
                        if not self.enabled:
                            # Cleanup must torque-off even if enable fails partway.
                            self.enabled=True
                            self.driver.enable()
                            self.coupling.current[:]=0.
                            self.current_ma=self.driver.currents(np.zeros(7))
                            self._publish(*self.driver.positions())
                            previous=self.latest
                            self.feedback_reason='Enabled at zero; waiting for fresh control tick'
                            self.stop.wait(.01)
                            continue
                        self.current_ma=self.driver.currents(current)
                        self.feedback_reason=('Gravity support current capped; full support not guaranteed'
                            if isinstance(self.coupling,GravityCoupling) and self.coupling.saturated else
                            'Leader gravity + damping enabled' if isinstance(self.coupling,GravityCoupling) else
                            'Low-current virtual joint resistance enabled')
                    elif self.enabled:
                        self._disable('Disabled by operator or paused mapping')
                except (OSError,ValueError,TimeoutError) as error:
                    self.rejected+=1
                    self.error=str(error)
                    self.last_fault={'reason':str(error),'stamp_ns':time.monotonic_ns(),
                                     'recent_motion':list(self.recent_motion)}
                    self._disable(str(error))
                self.stop.wait(.01)
        finally:
            self._disable('Stopped')

    def health(self):
        return super().health() | {'source':'factr_current_feedback','feedback':self.feedback_state,
            'control_interval_s':self.control_interval_s,'max_control_interval_s':self.max_control_interval_s,
            'last_control_fault':self.last_fault}

    @property
    def feedback_state(self):
        return {'requested':self.requested,'enabled':self.enabled,'current_ma':self.current_ma,
                'reason':self.feedback_reason,'kind':('leader_gravity_and_damping' if isinstance(self.coupling,GravityCoupling)
                    else 'virtual_joint_resistance_not_measured_external_torque'),
                **({'leader_urdf_sha256':self.coupling.gravity.sha256,'gravity_saturated':self.coupling.saturated}
                   if isinstance(self.coupling,GravityCoupling) else {}),
                **({'model_demand_ma':self.coupling.demand_ma.tolist(),
                    'current_cap_ma':self.coupling.cap.tolist(),'freshness_source':'leader_serial'}
                   if isinstance(self.coupling,GravityCoupling) else {'freshness_source':'simulated_follower'}),
                'software_watchdog_ms':100,'hardware_watchdog_ms':100,'trigger_motor_enabled':False}
