"""Isolated XRoboToolkit C SDK producer, compatible with the existing Unity client.

ABI: PXREARobotSDK.h (XR Robotics). No SDK library is loaded into Isaac/Kit.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import queue
import socket
import threading
import time
import uuid

from .protocol import SCHEMA, boot_id, normalize_device

STATE = 1 << 25
DISCONNECTED = (1 << 3, 1 << 5)
CALLBACK = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_int, C.c_void_p)


class DeviceState(C.Structure):
    _fields_ = [('device', C.c_char*32), ('state', C.c_char*16352)]


class SDK:
    def __init__(self, library):
        self.pending = queue.Queue(maxsize=1)
        self.dropped = 0
        self.generation = 0
        self.callbacks = 0
        self.last_callback_ns = None
        self.max_callback_gap_ns = 0
        self.lock = threading.Lock()
        self.library = C.CDLL(str(Path(library).resolve()))
        self.library.PXREAInit.argtypes = [C.c_void_p, CALLBACK, C.c_uint]
        self.library.PXREAInit.restype = C.c_int
        self.library.PXREADeinit.argtypes = []
        self.library.PXREADeinit.restype = C.c_int
        self.callback = CALLBACK(self._receive)

    def _receive(self, context, kind, status, data):
        with self.lock:
            if kind in DISCONNECTED:
                # Carry the restart on every following frame; a latest-only
                # queue must not overwrite the disconnect notification.
                self.generation += 1
                return
            if kind != STATE or not data:
                return
            state = C.cast(data, C.POINTER(DeviceState)).contents
            stamp = time.monotonic_ns()
            if self.last_callback_ns is not None:
                self.max_callback_gap_ns = max(self.max_callback_gap_ns, stamp-self.last_callback_ns)
            self.last_callback_ns = stamp
            self.callbacks += 1
            packet = (self.generation, stamp, bytes(state.device), bytes(state.state))
            try:
                self.pending.put_nowait(packet)
            except queue.Full:
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
                self.pending.put_nowait(packet)

    def start(self):
        code = self.library.PXREAInit(None, self.callback, STATE | DISCONNECTED[0] | DISCONNECTED[1])
        if code != 0:
            raise RuntimeError(f'XR SDK initialization failed: {code}')

    def close(self):
        self.library.PXREADeinit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--device', help='Pin a device serial; otherwise pin the first sender')
    parser.add_argument('--port', type=int, default=5557)
    parser.add_argument('--probe', action='store_true', help='Check ABI without connecting to devices')
    args = parser.parse_args()
    sdk = SDK(args.library)
    if args.probe:
        print(json.dumps({'abi_loaded': True, 'state_struct_bytes': C.sizeof(DeviceState)}))
        return
    stream, sequence, previous = uuid.uuid4().hex, 0, None
    generation = 0
    previous_stamp = None
    next_health = 0
    invalid_packets = duplicates = 0
    device = args.device
    clock = boot_id()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
        sdk.start()
        print('PICO bridge ready; connect the existing XRoboToolkit client.', flush=True)
        try:
            while True:
                epoch, stamp, sender, raw = sdk.pending.get()
                try:
                    sender = sender.decode()
                    if device is not None and device != sender:
                        continue
                    if epoch != generation:
                        stream, previous, generation = uuid.uuid4().hex, None, epoch
                    value = json.loads(json.loads(raw)['value'])
                    tracking = normalize_device(value)
                    source = int(value['timeStampNs'])
                    if previous is not None and source <= previous:
                        duplicates += 1
                        continue
                except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
                    invalid_packets += 1
                    print(f'Ignored invalid tracking: {error}', flush=True)
                    continue
                if device is None:
                    device = sender
                    print(f'Using PICO device {device}', flush=True)
                previous = source
                if previous_stamp is not None and stamp-previous_stamp > 150_000_000:
                    print(json.dumps({'event': 'tracking_gap', 'host_ns': stamp,
                                      'gap_s': (stamp-previous_stamp)/1e9}), flush=True)
                previous_stamp = stamp
                packet = {'schema': SCHEMA, 'boot_id': clock, 'stream': stream, 'sequence': sequence,
                          'host_ns': stamp, 'device': device, 'value': value}
                try:
                    encoded = json.dumps(packet, separators=(',', ':'), allow_nan=False).encode()
                except (ValueError, TypeError) as error:
                    invalid_packets += 1
                    print(f'Ignored invalid tracking: {error}', flush=True)
                    continue
                output.sendto(encoded, ('127.0.0.1', args.port))
                sequence += 1
                if stamp >= next_health:
                    print(json.dumps({'event': 'input_health', 'host_ns': stamp,
                        'callbacks': sdk.callbacks, 'sent': sequence, 'queue_drops': sdk.dropped,
                        'invalid_packets': invalid_packets, 'duplicate_timestamps': duplicates,
                        'max_callback_gap_s': sdk.max_callback_gap_ns/1e9,
                        'bridge_age_s': (time.monotonic_ns()-stamp)/1e9,
                        'focused': tracking['focused'], 'hands': list(tracking['hands']),
                        'controllers': list(tracking['controllers']),
                        'hand_status': tracking['hand_status'],
                        'controller_status': tracking['controller_status']}), flush=True)
                    next_health = stamp+5_000_000_000
        except KeyboardInterrupt:
            pass
        finally:
            sdk.close()
            print(json.dumps({'frames': sequence, 'queue_drops': sdk.dropped}))


if __name__ == '__main__':
    main()
