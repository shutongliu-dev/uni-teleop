"""Bounded, lossless state archive for deferred image rendering."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import hashlib
from pathlib import Path
import numpy as np


class StateArchive:
    def __init__(self, output, manifest):
        self.root = Path(output)/'replay'
        self.root.mkdir()
        manifest = dict(manifest)
        manifest['session_files_sha256'] = {name: hashlib.sha256((Path(output)/name).read_bytes()).hexdigest()
            for name in ('scene.json', 'initial_state.json') if (Path(output)/name).is_file()}
        (self.root/'manifest.json').write_text(json.dumps(manifest, indent=2))
        self.index = (self.root/'frames.jsonl').open('x', buffering=1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='state-archive')
        self.pending = deque()
        self.count = 0

    def append(self, arrays, metadata):
        # Backpressure rather than silently dropping state/action pairs.
        while self.pending and (self.pending[0].done() or len(self.pending) >= 4):
            self.pending.popleft().result()
        name = f'{self.count:06d}.npz'
        copied = {key: np.array(value, copy=True) for key, value in arrays.items()}
        row = dict(metadata, file=name, frame=self.count)
        def write():
            temporary = self.root/(name+'.partial')
            with temporary.open('xb') as stream:
                np.savez(stream, **copied)
            temporary.rename(self.root/name)
            self.index.write(json.dumps(row, allow_nan=False)+'\n')
        self.pending.append(self.pool.submit(write))
        self.count += 1

    def close(self):
        self.pool.shutdown(wait=True)
        try:
            for job in self.pending:
                job.result()
        finally:
            self.index.close()
