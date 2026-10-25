"""Own the leader serial/control loop outside Isaac's interpreter and GIL."""
import ctypes
import multiprocessing as mp
import os
import signal
import threading
import time
from types import SimpleNamespace

from .haptics import HapticReceiver


def _worker(connection, parent_pid, port, baud, gravity, current_scale):
    def terminate(*_):
        raise SystemExit

    signal.signal(signal.SIGTERM, terminate)
    libc=ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0)!=0:
        raise OSError(ctypes.get_errno(), 'Cannot bind leader worker to parent')
    if os.getppid()!=parent_pid:
        return
    receiver=None
    stopped=threading.Event()
    try:
        receiver=HapticReceiver(port, baud=baud, gravity=gravity, gravity_current_scale=current_scale)
        connection.send(('ready', receiver.read(), receiver.health()))

        def publish():
            try:
                while not stopped.wait(.02):
                    connection.send(('state', receiver.read(), receiver.health()))
            except (BrokenPipeError, EOFError, OSError):
                stopped.set()

        # A blocked UI/telemetry consumer must never block serial control or Pause.
        threading.Thread(target=publish, daemon=True).start()
        while not stopped.is_set():
            if not connection.poll(.02):
                continue
            command, stamp, value=connection.recv()
            if command=='close':
                break
            if command=='stop':
                receiver.request_feedback(False)
            elif command=='start':
                if not 0<=time.monotonic_ns()-stamp<=150_000_000:
                    receiver.request_feedback(False)
                    receiver.feedback_reason='Start command expired; press Start again'
                else:
                    receiver.start_support(SimpleNamespace(arm_calibration=value))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        if receiver is None:
            connection.send(('error', str(error)))
        raise
    finally:
        stopped.set()
        if receiver is not None:
            receiver.close()
        connection.close()


class GravityProcessReceiver:
    """Same UI/input interface, with output owned by a spawned Linux process."""
    def __init__(self, port, *, baud=4000000, gravity, gravity_current_scale=1, gravity_calibration=None):
        self.gravity_calibration = gravity_calibration
        context=mp.get_context('spawn')
        self.connection, child=context.Pipe()
        self.process=context.Process(target=_worker,
            args=(child, os.getpid(), port, baud, gravity, gravity_current_scale), daemon=True)
        self.latest=None
        self._health={}
        self.error=None
        self.closed=False
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(10.):
                raise TimeoutError('Leader control process startup timed out')
            message=self.connection.recv()
            if message[0]!='ready':
                raise RuntimeError(message[1])
            _, self.latest, self._health=message
        except BaseException:
            self.close()
            raise
        self.thread=threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.closed:
                _, self.latest, self._health=self.connection.recv()
        except (EOFError, OSError):
            self.error='Leader control process disconnected; output unavailable'

    def read(self):
        return self.latest

    @property
    def rejected(self):
        return self._health.get('rejected', 0)

    @property
    def requested(self):
        return self.feedback_state['requested']

    @property
    def feedback_state(self):
        state=self._health.get('feedback', {}).copy()
        if self.error or self.closed:
            state.update(requested=False, enabled=False, current_ma=[0.]*7,
                         reason=self.error or 'Stopped')
        return state

    def health(self):
        return self._health | {'feedback':self.feedback_state,
            'control_process_pid':self.process.pid,
            'latest_age_s':(time.monotonic_ns()-self.latest.host_ns)/1e9 if self.latest else None,
            'last_error':self.error or self._health.get('last_error')}

    def start_support(self, mapper):
        arm=getattr(mapper, 'leader', mapper)
        cal=self.gravity_calibration if self.gravity_calibration is not None else arm.arm_calibration
        if cal is None or cal.status!='measured':
            raise ValueError('Leader gravity requires verified physical calibration')
        self.connection.send(('start', time.monotonic_ns(), cal))

    def request_feedback(self, enabled):
        if enabled:
            raise ValueError('Use Start with verified calibration to enable gravity')
        self.connection.send(('stop', time.monotonic_ns(), None))

    def observe(self, mapper, observation, *, now_ns):
        if observation['truncated'] or observation['terminated']:
            self.request_feedback(False)

    def close(self):
        if self.closed:
            return
        self.closed=True
        try:
            self.connection.send(('close', time.monotonic_ns(), None))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(2.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.)
        if self.process.is_alive():
            self.process.kill()  # Motor watchdog is the final fallback.
            self.process.join(1.)
        self.connection.close()
