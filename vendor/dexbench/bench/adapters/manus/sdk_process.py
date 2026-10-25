"""MANUS SDK 3.2.0/3.2.1 producer. Run in its own vendor environment/process, never in Kit.

This software contains source code provided by Manus Technology Group B.V.
API usage follows the SDK's included Python examples; vendor LICENSE stays in assets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time
import uuid


def bind_parent(pid):
    """Release USB ownership even when Kit exits without Python cleanup (Linux)."""
    import ctypes
    import signal
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(),'Cannot bind MANUS lifetime to parent')
    if os.getppid()!=pid:  # Parent may exit before prctl is installed.
        raise SystemExit('MANUS parent already exited')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-python', type=Path, required=True, help='official distribution/Python directory')
    parser.add_argument('--sdk-version', choices=['3.2.0', '3.2.1'], default='3.2.1')
    parser.add_argument('--probe', action='store_true', help='check CFFI ABI without connecting/initializing devices')
    parser.add_argument('--mode', choices=['local', 'remote', 'integrated'], default='local')
    parser.add_argument('--host', help='exact discovered host IP/name; required if discovery finds multiple hosts')
    parser.add_argument('--glove-id', type=int)
    parser.add_argument('--discover-devices', action='store_true', help='List connected gloves without selecting one or calibrating')
    parser.add_argument('--calibration-id')
    parser.add_argument('--tracker-id')
    parser.add_argument('--tracker-body-frame', default='unassigned_tracker')
    parser.add_argument('--clock-id', help='shared host/boot monotonic clock ID; required on non-Linux')
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--queue-size', type=int, default=512)
    parser.add_argument('--parent-pid', type=int, help='Owning teleop process; release USB when it exits')
    args = parser.parse_args()
    if args.parent_pid is not None:bind_parent(args.parent_pid)
    if args.glove_id is not None and not 0 <= args.glove_id <= 4294967295:
        parser.error('--glove-id must be uint32')
    if args.queue_size < 1 or args.duration <= 0:
        parser.error('queue size and duration must be positive')
    if not (args.probe or args.discover_devices) and (args.glove_id is None or not args.calibration_id):
        parser.error('live capture requires explicit --glove-id and --calibration-id')
    sys.path.insert(0, str(args.sdk_python.resolve()))
    # Preserve clean JSONL channel even if native library prints to stdout.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from manus_sdk import lib, ffi, ManusSDK
    from manus_sdk.generated._enums import (SDKReturnCode, AxisView, AxisPolarity, Side,
        HandMotion, DeviceFamilyType, ChainType, FingerJointType, TrackingQuality, ErgonomicsDataType)
    if lib is None or ffi is None:
        raise SystemExit('MANUS CFFI/native library unavailable in the selected SDK environment')
    required = ['CoreSdk_InitializeCore', 'CoreSdk_InitializeIntegrated',
                'CoreSdk_InitializeCoordinateSystemWithVUH', 'CoreSdk_GetRawSkeletonData',
                'CoreSdk_GetRawSkeletonNodeInfoArray', 'CoreSdk_GetTrackerData',
                'CoreSdk_GetVersionsAndCheckCompatibility', 'CoreSdk_ShutDown']
    for name in required:
        getattr(lib, name)
    if args.probe:
        output.write(json.dumps({'abi_loaded': True, 'hardware_initialized': False,
             'sdk_python': str(args.sdk_python.resolve()), 'symbols_checked': required,
             'struct_sizes': {name: ffi.sizeof(name) for name in ['RawSkeletonInfo', 'SkeletonNode', 'NodeInfo', 'TrackerData', 'ErgonomicsStream']}})+'\n')
        return
    if args.clock_id:
        clock_id = args.clock_id
    else:
        boot_path = Path('/proc/sys/kernel/random/boot_id')
        if not boot_path.exists():
            parser.error('--clock-id required on this platform')
        clock_id = socket.gethostname()+'/'+boot_path.read_text().strip()+'/monotonic'
    stream_id = str(uuid.uuid4())
    out_queue = queue.Queue(args.queue_size)
    state = {'sequence': 0, 'drops': 0, 'device': None, 'tracker': None,
             'ergonomics': None, 'failure': None, 'sdk_version': None, 'core_version': None}
    lock = threading.Lock()
    connected = threading.Event()
    sdk = ManusSDK()
    def check(code, operation):
        if code != SDKReturnCode.Success:
            raise RuntimeError(f'{operation}: SDK return {int(code)}')
    def enum_name(enum, value):
        try:
            return enum(value).name
        except ValueError:
            return 'vendor_'+str(value)
    def pose(position, rotation):
        return {'position': [float(position.x), float(position.y), float(position.z)],
                'quaternion_wxyz': [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]}
    def callback(fn):
        def guarded(info):
            try:
                fn(info)
            except Exception as exc:
                with lock:
                    state['failure'] = str(exc)
        return guarded
    @callback
    def disconnected(_host):
        connected.clear()
        with lock:
            state['failure'] = 'MANUS Core disconnected; start a new capture stream after reconnect'
    @callback
    def landscape(info):
        for i in range(info.gloveDevices.gloveCount):
            glove = info.gloveDevices.gloves[i]
            if not args.discover_devices and int(glove.id) != args.glove_id:
                continue
            if glove.side not in (Side.Left, Side.Right):
                raise RuntimeError('glove side is unknown')
            v = glove.firmwareVersion
            device = {'device_id': str(glove.id), 'product': enum_name(DeviceFamilyType, glove.familyType),
                'firmware': f'{v.major}.{v.minor}.{v.patch}',
                'hand': 'right' if glove.side == Side.Right else 'left', 'source_kind': 'hardware'}
            if args.discover_devices:
                out_queue.put_nowait({'protocol': 'manus_device_discovery_v1', 'device': device})
                continue
            with lock:
                state['device'] = device
    @callback
    def trackers(info):
        if args.tracker_id is None:
            return
        now = time.monotonic_ns()
        for i in range(info.trackerCount):
            value = ffi.new('TrackerData*')
            check(lib.CoreSdk_GetTrackerData(i, value), 'get tracker')
            if ffi.string(value.trackerId.id).decode() != args.tracker_id:
                continue
            timestamp = str(value.lastUpdateTime.time)
            with lock:
                previous = state['tracker']
                # Repeated SDK publication of old tracking data is not a fresh measurement.
                measured = previous['capture_time_ns'] if previous and previous['source_timestamp'] == timestamp else now
                state['tracker'] = {'tracker_id': args.tracker_id,
                    'status': 'tracked' if value.quality == TrackingQuality.Trackable else 'lost',
                    'capture_time_ns': measured, 'clock_id': clock_id, 'frame_id': 'manus_world',
                    'tracked_body_frame': args.tracker_body_frame, 'source_timestamp': timestamp,
                    'pose': pose(value.position, value.rotation)}
    @callback
    def ergonomics(info):
        for i in range(info.dataCount):
            item = info.data[i]
            if item.isUserID or int(item.id) != args.glove_id:
                continue
            data = {enum_name(ErgonomicsDataType, j): float(item.data[j]) for j in range(len(item.data))}
            with lock:
                state['ergonomics'] = (time.monotonic_ns(), str(info.publishTime.time), data)
    @callback
    def skeleton(info):
        now = time.monotonic_ns()
        for i in range(info.skeletonsCount):
            raw = ffi.new('RawSkeletonInfo*')
            check(lib.CoreSdk_GetRawSkeletonInfo(i, raw), 'get raw skeleton info')
            if int(raw.gloveId) != args.glove_id:
                continue
            if raw.nodesCount == 0 or raw.nodesCount > 256:
                raise RuntimeError('invalid SDK skeleton size')
            nodes = ffi.new('SkeletonNode[]', raw.nodesCount)
            metadata = ffi.new('NodeInfo[]', raw.nodesCount)
            check(lib.CoreSdk_GetRawSkeletonData(i, nodes, raw.nodesCount), 'get raw skeleton')
            check(lib.CoreSdk_GetRawSkeletonNodeInfoArray(raw.gloveId, metadata, raw.nodesCount), 'get node topology')
            ids = {int(metadata[j].nodeId) for j in range(raw.nodesCount)}
            meta = {int(metadata[j].nodeId): metadata[j] for j in range(raw.nodesCount)}
            copied = []
            for j in range(raw.nodesCount):
                node = nodes[j]
                m = meta[int(node.id)]
                parent = int(m.parentId)
                if parent not in ids and parent not in (0, 4294967295):
                    raise RuntimeError('SDK skeleton refers to unknown parent')
                copied.append({'node_id': str(node.id),
                    'parent_id': str(parent) if parent in ids and parent != int(node.id) else None,
                    'semantic': enum_name(ChainType, m.chainType)+'/'+enum_name(FingerJointType, m.fingerJointType),
                    'pose': pose(node.transform.position, node.transform.rotation), 'valid': True})
            # Copy native memory in callbacks. Do not retain CFFI views after returning.
            with lock:
                if state['device'] is None or state['sdk_version'] is None:
                    continue
                device = dict(state['device'], sdk_version=state['sdk_version'], core_version=state['core_version'])
                sequence = state['sequence']
                state['sequence'] += 1
                topology = [(n['node_id'], n['parent_id'], n['semantic']) for n in copied]
                packet = {'protocol': 'manus_acquisition_v1', 'stream_id': stream_id, 'sequence': sequence,
                    'clock_id': clock_id, 'capture_time_ns': now, 'source_timestamp': str(info.publishTime.time),
                    'source_timestamp_encoding': 'ManusTimestamp.compressed_uint64', 'device': device,
                    'calibration_id': args.calibration_id, 'topology_id': 'manus_raw/'+hashlib.sha256(json.dumps(topology).encode()).hexdigest(),
                    'frame_id': 'manus_world', 'axes': 'right_handed_z_up_x_forward',
                    'position_unit': 'm', 'node_space': 'world', 'glove_tracking_valid': True,
                    'nodes': copied, 'ergonomics_degrees': {}, 'tracker': state['tracker'],
                    'sdk_queue_drops_total': state['drops']}
                ergo = state['ergonomics']
                if ergo is not None and 0 <= now-ergo[0] <= 30_000_000:
                    packet.update(ergonomics_degrees=ergo[2], ergonomics_capture_time_ns=ergo[0],
                                  ergonomics_source_timestamp=ergo[1])
                try:
                    out_queue.put_nowait(packet)
                except queue.Full:
                    state['drops'] += 1
    initialized = False
    try:
        check(lib.CoreSdk_InitializeIntegrated() if args.mode == 'integrated' else lib.CoreSdk_InitializeCore(), 'initialize')
        initialized = True
        coordinate = ffi.new('CoordinateSystemVUH*')
        coordinate.view, coordinate.up, coordinate.handedness, coordinate.unitScale = AxisView.XFromViewer, AxisPolarity.PositiveZ, Side.Right, 1.0
        check(lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinate[0], True), 'coordinates')
        for register, fn in [(sdk.register_connected_callback, lambda host: connected.set()),
                              (sdk.register_disconnected_callback, disconnected),
                              (sdk.register_landscape_callback, landscape), (sdk.register_raw_skeleton_callback, skeleton),
                              (sdk.register_tracker_callback, trackers), (sdk.register_ergonomics_callback, ergonomics)]:
            check(register(fn), 'register stream')
        check(lib.CoreSdk_LookForHosts(3, args.mode != 'remote'), 'discover')
        count = ffi.new('uint32_t*')
        check(lib.CoreSdk_GetNumberOfAvailableHostsFound(count), 'host count')
        if count[0] == 0:
            raise RuntimeError('no MANUS Core host found')
        hosts = ffi.new('ManusHost[]', count[0])
        check(lib.CoreSdk_GetAvailableHostsFound(hosts, count[0]), 'hosts')
        matches = [h for h in hosts if args.host is None or args.host in (ffi.string(h.ipAddress).decode(), ffi.string(h.hostName).decode())]
        if len(matches) != 1:
            raise RuntimeError('select exactly one host with --host')
        check(lib.CoreSdk_ConnectToHost(matches[0]), 'connect')
        if not connected.wait(timeout=10):
            raise RuntimeError('MANUS Core connection callback timed out')
        if args.mode == 'integrated':
            # Integrated uses the host connection but has no external Core version.
            # Version fields identify the selected package, not a measured Core.
            state['sdk_version'] = 'unreported/package-'+args.sdk_version
            state['core_version'] = 'integrated-unreported'
        else:
            sdk_version, core_version, compatible = ffi.new('ManusVersion*'), ffi.new('ManusVersion*'), ffi.new('bool*')
            check(lib.CoreSdk_GetVersionsAndCheckCompatibility(sdk_version, core_version, compatible), 'version check')
            if not compatible[0]:
                raise RuntimeError('SDK/Core version mismatch: SDK='+ffi.string(sdk_version.versionInfo).decode()+
                                   ' Core='+ffi.string(core_version.versionInfo).decode())
            with lock:
                state['sdk_version'] = ffi.string(sdk_version.versionInfo).decode()
                state['core_version'] = ffi.string(core_version.versionInfo).decode()
                if re.search(r'(?<![0-9])'+re.escape(args.sdk_version)+r'(?![0-9])', state['sdk_version']) is None:
                    raise RuntimeError('producer SDK does not match selected package version')
        check(lib.CoreSdk_SetRawSkeletonHandMotion(HandMotion.Auto), 'hand motion')
        deadline = time.monotonic()+args.duration
        emitted = 0
        while time.monotonic() < deadline:
            with lock:
                failure = state['failure']
            if failure:
                raise RuntimeError(f'SDK callback failed: {failure}')
            try:
                packet = out_queue.get(timeout=.1)
            except queue.Empty:
                continue
            output.write(json.dumps(packet, allow_nan=False, separators=(',', ':'))+'\n')
            emitted += 1
        if emitted == 0:
            raise RuntimeError('no glove packets received; check selected glove and SDK entitlement')
    finally:
        if initialized:
            lib.CoreSdk_ShutDown()
        # Shutdown joins SDK callbacks; drain already-copied frames instead of silently
        # dropping the bounded queue tail when the requested capture duration ends.
        while True:
            try:
                tail = out_queue.get_nowait()
            except queue.Empty:
                break
            output.write(json.dumps(tail, allow_nan=False, separators=(',', ':'))+'\n')
        with lock:
            print(json.dumps({'sdk_queue_drops_total': state['drops'], 'stream_id': stream_id}), file=sys.stderr)

if __name__ == '__main__':
    main()
