"""Append-only collection records under the shared datasets tree."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from .bridge import CaptureBridge, canonical
from .schema import AcquisitionPacket, CaptureConfig
from bench.paths import OUTPUT_ROOT

DEFAULT_ROOT = Path(f'{OUTPUT_ROOT}/manus/captures')

class CaptureRecorder:
    def __init__(self, config: CaptureConfig, root: Path = DEFAULT_ROOT, *, replay_of: str | None = None):
        self.path = Path(root) / str(uuid.uuid4())
        self.path.mkdir(parents=True, exist_ok=False)
        self.bridge = CaptureBridge(config)
        self.counts = {'accepted': 0, 'rejected': 0, 'duplicate': 0, 'invalid': 0}
        self.raw = (self.path/'acquisition.jsonl').open('x')
        self.normalized = (self.path/'human_motion.jsonl').open('x')
        self.events = (self.path/'events.jsonl').open('x')
        self.manifest = {
            'schema': 'manus_capture_manifest_v1', 'status': 'recording',
            'created_unix_ns': time.time_ns(), 'capture_id': self.path.name,
            'configuration': config.model_dump(mode='json'),
            'config_sha256': hashlib.sha256(canonical(config.model_dump(mode='json'))).hexdigest(),
            'source_kind_counts': {}, 'hardware_connected': False,
            'hardware_status_scope': 'collector_cannot_verify_hardware_connection; see source_kind_counts and SDK process evidence',
            'replay_of': replay_of, 'robot_commands_applied': 0,
            'implementation_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(Path(__file__).parent.glob('*.py'))},
            'notes': 'Human motion acquisition only. Robot actions and simulator observations require a separately linked execution log.',
        }
        self._write_manifest()

    def _write_manifest(self):
        temp = self.path/'manifest.tmp'
        temp.write_text(json.dumps(self.manifest, indent=2, allow_nan=False)+'\n')
        temp.replace(self.path/'manifest.json')

    def ingest(self, packet: AcquisitionPacket, receive_time_ns: int | None = None):
        if self.manifest['status'] != 'recording':
            raise RuntimeError('capture already closed')
        now = time.monotonic_ns() if receive_time_ns is None else receive_time_ns
        result = self.bridge.accept(packet, now)
        self.counts[result.status] += 1
        # Raw packet on EVERY schema-valid attempt, including duplicates/rejections.
        self.raw.write(canonical({'receive_time_ns': now, 'status': result.status,
                                  'reason': result.reason, 'packet': packet.model_dump(mode='json')}).decode()+'\n')
        if result.frame is not None:
            self.normalized.write(canonical(result.frame).decode()+'\n')
            kind = packet.device.source_kind
            kinds = self.manifest['source_kind_counts']
            kinds[kind] = kinds.get(kind, 0)+1
        self.events.write(canonical({'receive_time_ns': now, 'sequence': packet.sequence,
                                     'status': result.status, 'reason': result.reason}).decode()+'\n')
        # Flush bounded batches at the producer/collector boundary, never inside SDK callbacks.
        for stream in (self.raw, self.normalized, self.events):
            stream.flush()
        return result

    def invalid(self, reason: str):
        self.counts['invalid'] += 1
        self.events.write(canonical({'status': 'invalid', 'reason': reason,
                                     'receive_time_ns': time.monotonic_ns()}).decode()+'\n')
        self.events.flush()

    def close(self, reason='source_finished'):
        if self.manifest['status'] != 'recording':
            return
        for stream in (self.raw, self.normalized, self.events):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.manifest.update(status='closed', termination=reason, closed_unix_ns=time.time_ns(), counts=self.counts)
        self.manifest['files'] = {
            name: {'sha256': hashlib.sha256((self.path/name).read_bytes()).hexdigest(),
                   'size_bytes': (self.path/name).stat().st_size}
            for name in ('acquisition.jsonl', 'human_motion.jsonl', 'events.jsonl')}
        self._write_manifest()
