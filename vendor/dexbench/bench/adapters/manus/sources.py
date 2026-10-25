"""Line-delimited process boundary with strict bounded input; no SDK in simulator process."""
from __future__ import annotations
import json
import socket
from pathlib import Path
import time
import uuid
from .schema import AcquisitionPacket

MAX_LINE_BYTES = 2 * 1024 * 1024

def host_clock_id():
    # Monotonic time is shared by local processes; boot ID prevents cross-reboot equivalence.
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        # Other systems must explicitly share this ID between producer and collector.
        boot = 'session-'+str(uuid.uuid4())
    return f'{socket.gethostname()}/{boot}/monotonic'


def packets_from_jsonl(stream):
    while True:
        line = stream.readline(MAX_LINE_BYTES+1)
        if not line:
            return
        if len(line.encode('utf-8')) > MAX_LINE_BYTES:
            raise ValueError('acquisition line exceeds 2 MiB')
        item = json.loads(line)
        # Accept raw producer lines or the exact archived acquisition wrapper.
        yield AcquisitionPacket.model_validate(item.get('packet', item))


def mock_packets(count=10, *, clock_id=None, start_ns=None, period_ns=10_000_000):
    clock_id = clock_id or host_clock_id()
    start_ns = time.monotonic_ns() if start_ns is None else start_ns
    stream_id = 'synthetic-'+str(uuid.uuid4())
    for sequence in range(count):
        pose = {'position': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]}
        nodes = [{'node_id': 'wrist', 'parent_id': None, 'semantic': 'Hand/Wrist', 'pose': pose}]
        for i, name in enumerate(['thumb', 'index', 'middle', 'ring', 'little']):
            nodes.append({'node_id': name+'_tip', 'parent_id': 'wrist', 'semantic': name+'/tip',
                          'pose': {'position': [.08, (i-2)*.02, .002*sequence],
                                   'quaternion_wxyz': [1, 0, 0, 0]}})
        yield AcquisitionPacket(stream_id=stream_id, sequence=sequence, clock_id=clock_id,
            capture_time_ns=start_ns+sequence*period_ns, source_timestamp=str(sequence),
            source_timestamp_encoding='synthetic_frame_index',
            device={'device_id': 'mock_right', 'product': 'synthetic_fixture', 'firmware': 'none',
                    'sdk_version': 'none', 'core_version': 'none', 'hand': 'right', 'source_kind': 'synthetic'},
            calibration_id='synthetic_identity', topology_id='synthetic_six_nodes_v1',
            frame_id='manus_world', glove_tracking_valid=True, nodes=nodes,
            ergonomics_degrees={'RightIndexMCP': float(sequence)})
