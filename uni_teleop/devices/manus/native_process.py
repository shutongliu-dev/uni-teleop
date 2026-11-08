"""Normalize the isolated C++ SDK reader into the shared acquisition contract."""
import argparse
import hashlib
import json
import math
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
from uni_teleop.devices.manus.sources import host_clock_id
from uni_teleop.devices.manus.schema import AcquisitionPacket
from uni_teleop.devices.manus.sdk_process import bind_parent


def normalize(row, *, glove_id, calibration_id, stream, clock):
    if row['glove_id'] != glove_id:
        raise ValueError('Unexpected glove identity')
    row = deepcopy(row)
    for node in row['nodes']:
        q = node['pose']['quaternion_wxyz']
        norm = math.sqrt(sum(v*v for v in q))
        if not math.isfinite(norm) or not .95 <= norm <= 1.05:
            raise ValueError('Invalid SDK quaternion magnitude')
        node['pose']['quaternion_wxyz'] = [v/norm for v in q]
    topology = [(n['node_id'], n['parent_id'], n['semantic']) for n in row['nodes']]
    return AcquisitionPacket.model_validate(dict(
        protocol='manus_acquisition_v1', stream_id=stream, sequence=row['sequence'],
        clock_id=clock, capture_time_ns=row['capture_time_ns'],
        source_timestamp=row['source_timestamp'], source_timestamp_encoding='ManusTimestamp.compressed_uint64',
        device=dict(device_id=str(glove_id), product='MetaglovePro', firmware='unreported',
                    sdk_version='native-sdk-unreported', core_version='integrated-unreported',
                    hand='right', source_kind='hardware'), calibration_id=calibration_id,
        topology_id='manus_raw/'+hashlib.sha256(json.dumps(topology).encode()).hexdigest(),
        frame_id='manus_world', axes='right_handed_z_up_x_forward', position_unit='m',
        node_space='world', glove_tracking_valid=True, nodes=row['nodes'])).model_dump(mode='json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--glove-id', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args=parser.parse_args()
    bind_parent(args.parent_pid)
    calibration_id='official/'+hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    stream,clock=str(uuid.uuid4()),host_clock_id()
    process=subprocess.Popen([str(args.reader),str(args.glove_id),str(args.calibration)],stdout=subprocess.PIPE,text=True)
    def stop(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for line in process.stdout:
            try:
                row=normalize(json.loads(line),glove_id=args.glove_id,calibration_id=calibration_id,stream=stream,clock=clock)
            except ValueError:
                print('Rejected invalid MANUS frame', file=sys.stderr)
                continue
            print(json.dumps(row,allow_nan=False),flush=True)
        if process.wait()!=0:raise RuntimeError('Native MANUS reader failed; check SDK log')
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':main()
