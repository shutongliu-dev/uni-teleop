"""Same-host, latest-frame transport. Device time is identity, never host time."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import select
import socket
import time
import threading

SCHEMA = 'dexbench_pico_v1'
SIDES = ('left', 'right')
INPUT_ERRORS = (ValueError, KeyError, TypeError, AttributeError, OverflowError)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def pose(value):
    if isinstance(value, str):
        value = [float(x) for x in value.split(',')]
    if len(value) != 7 or not all(type(x) in (int, float) and math.isfinite(x) for x in value):
        raise ValueError('Pose requires finite xyz + xyzw')
    norm = math.sqrt(sum(x*x for x in value[3:]))
    if not .5 < norm < 1.5:
        raise ValueError('Invalid tracked orientation')
    return list(value[:3]) + [x/norm for x in value[3:]]


def normalize_device(value):
    """Parse the existing XRoboToolkit Unity payload; absent tracking stays absent."""
    result = {'controllers': {}, 'hands': {}, 'head': None, 'hand_status': {}, 'controller_status': {},
              'focused': bool(value.get('appState', {}).get('focus', True))}
    if 'Head' in value:
        try:
            result['head'] = pose(value['Head']['pose'])
        except INPUT_ERRORS:
            pass  # A bad head sample must not discard valid hand/controller input.
    if not result['focused']:
        result['hand_status'] = {side: 'PICO app not focused' for side in SIDES}
        return result
    for side in SIDES:
        controller = value.get('Controller', {}).get(side)
        if controller is not None:
            try:
                trigger, grip = float(controller['trigger']), float(controller['grip'])
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in (trigger, grip)):
                    raise ValueError('Trigger and grip must be within [0, 1]')
                tracked = pose(controller['pose'])
                result['controllers'][side] = {'pose': tracked, 'trigger': trigger, 'grip': grip}
            except INPUT_ERRORS as error:
                result['controller_status'][side] = str(error)
        else:
            result['controller_status'][side] = 'controller data absent'
        hand = value.get('Hand', {}).get(side+'Hand')
        try:
            if hand is None or not hand['isActive']:
                result['hand_status'][side] = 'hand inactive' if hand is not None else 'hand data absent'
                continue
            locations = hand['HandJointLocations']
            if len(locations) != 26:
                raise ValueError('Expected 26 OpenXR hand joints, palm then wrist')
            joints = []
            for i, joint in enumerate(locations):
                try:
                    if int(joint.get('s', 3)) & 3 != 3:
                        raise ValueError('Invalid joint flags')
                    joints.append(pose(joint['p']))
                except INPUT_ERRORS:
                    joints.append(None)
            invalid = [i for i, joint in enumerate(joints) if joint is None]
            if invalid:
                result['hand_status'][side] = 'invalid joint tracking: ' + ','.join(map(str, invalid))
            if joints[1] is not None:
                result['hands'][side] = joints  # Wrist controls the arm; missing fingers remain explicit.
        except INPUT_ERRORS as error:
            result['hand_status'][side] = str(error)
    return result


@dataclass(frozen=True)
class Frame:
    stream: str
    sequence: int
    host_ns: int
    source_ns: str
    device: str
    tracking: dict
    continuous_since_ns: dict | None = None

    @classmethod
    def decode(cls, raw, *, clock, now_ns, max_age_ns):
        value = json.loads(raw)
        if value['schema'] != SCHEMA or value['boot_id'] != clock:
            raise ValueError('PICO bridge must run on this host and boot')
        sequence, stamp = value['sequence'], value['host_ns']
        if type(sequence) is not int or sequence < 0 or type(stamp) is not int:
            raise ValueError('Invalid source sequence or host timestamp')
        if not 0 <= now_ns - stamp <= max_age_ns:
            raise ValueError('Stale or future PICO frame')
        source = value['value']['timeStampNs']
        if type(source) not in (int, str) or int(source) < 0:
            raise ValueError('Invalid device timestamp')
        if any(not isinstance(value[k], str) or not value[k] for k in ('stream', 'device')):
            raise ValueError('Missing stream or device identity')
        # Re-validate untrusted SDK fields at the consumer boundary.
        tracking = normalize_device(value['value'])
        return cls(value['stream'], sequence, stamp, str(value['value']['timeStampNs']),
                   value['device'], tracking)


class Receiver:
    def __init__(self, port=5557, max_age_s=.15):
        if not math.isfinite(max_age_s) or not 0 < max_age_s <= 1:
            raise ValueError('Tracking age must be in (0, 1] seconds')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.settimeout(.05)
        self.clock = boot_id()
        self.max_age_ns = int(max_age_s*1e9)
        self.latest = None
        self.rejected = 0
        self.received = 0
        self.last_rejection = None
        self._expired_frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._receive, name='pico-receiver', daemon=True)
        self._thread.start()

    def _receive(self):
        # Drain continuously, even while Isaac steps or renders. A full UDP
        # queue retains old datagrams and drops the newest tracking samples.
        while not self._stop.is_set():
            try:
                raw = self.socket.recv(65535)
            except socket.timeout:
                continue
            try:
                frame = Frame.decode(raw, clock=self.clock, now_ns=time.monotonic_ns(), max_age_ns=self.max_age_ns)
                previous = self.latest
                if previous and (frame.host_ns <= previous.host_ns or
                        (frame.stream == previous.stream and (frame.sequence <= previous.sequence
                         or int(frame.source_ns) <= int(previous.source_ns)))):
                    raise ValueError('Repeated or out-of-order PICO frame')
                continuous = {}
                for mode in ('hands', 'controllers'):
                    for side in frame.tracking[mode]:
                        key = mode+'/'+side
                        uninterrupted = (previous is not None and previous.stream == frame.stream
                                         and frame.host_ns-previous.host_ns <= self.max_age_ns)
                        continuous[key] = ((previous.continuous_since_ns or {}).get(key, frame.host_ns)
                                           if uninterrupted else frame.host_ns)
                self.latest = replace(frame, continuous_since_ns=continuous)
                self.received += 1
            except INPUT_ERRORS as error:
                self.rejected += 1
                self.last_rejection = str(error)

    def health(self):
        latest = self.latest
        return {'received': self.received, 'rejected': self.rejected,
                'last_rejection': self.last_rejection,
                'latest_age_s': (time.monotonic_ns()-latest.host_ns)/1e9 if latest else None,
                'latest_sequence': latest.sequence if latest else None}

    def read(self):
        latest = self.latest
        if latest is None:
            return None
        now = time.monotonic_ns()
        if not 0 <= now-latest.host_ns <= self.max_age_ns:
            if self._expired_frame == (latest.stream, latest.sequence):
                return None
            self._expired_frame = (latest.stream, latest.sequence)
            print(json.dumps({'event': 'tracking_expired', 'host_ns': now,
                              'age_s': (now-latest.host_ns)/1e9, 'sequence': latest.sequence,
                              'pending_udp': bool(select.select([self.socket], [], [], 0)[0]),
                              'rejected': self.rejected}), flush=True)
            return None
        return latest

    def close(self):
        self._stop.set()
        self._thread.join()
        self.socket.close()
