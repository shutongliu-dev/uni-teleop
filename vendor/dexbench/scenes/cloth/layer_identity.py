"""Versioned stronger layer identity from complete, fresh USDC serialization.

This is not the old textual SHA protocol. In particular binary serialization can
distinguish NaN payloads that the textual representation normalizes. Every call
enumerates and exports every current layer; no notice, dirty-bit, state, payload,
or verdict cache is used. Start/end text archives remain the caller's obligation.
"""
from pathlib import Path
import hashlib
import tempfile

PROTOCOL = 'complete_fresh_usdc_sha256_v1'


class FreshUsdcLayerFingerprint:
    def __init__(self, *, ram_directory='/dev/shm'):
        directory = Path(ram_directory)
        if not directory.is_dir():
            raise ValueError('RAM temporary directory is unavailable')
        self.ram_directory = str(directory)

    def __call__(self, env):
        layers = list(env.stage.GetUsedLayers())
        identifiers = [layer.identifier for layer in layers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('Duplicate USD layer identifier')
        result = {}
        # Isolated, unpredictable directory and sequential file use prevent
        # concurrent observers from reusing another call's scratch content.
        with tempfile.TemporaryDirectory(prefix='cloth_layer_fingerprint_', dir=self.ram_directory) as directory:
            path = Path(directory) / 'current_layer.usdc'
            for layer in layers:
                # A missing/failed export cannot leave a previous layer's file.
                path.unlink(missing_ok=True)
                if layer.Export(str(path)) is not True:
                    raise RuntimeError('Fresh complete USDC layer export failed')
                if not path.is_file() or path.stat().st_size < 8:
                    raise RuntimeError('Fresh USDC layer export is missing or truncated')
                with path.open('rb') as stream:
                    if stream.read(8) != b'PXR-USDC':
                        raise RuntimeError('Layer export did not produce a USDC crate')
                    stream.seek(0)
                    result[layer.identifier] = hashlib.file_digest(stream, 'sha256').hexdigest()
        # Schema is part of the compared payload; this cannot masquerade as an
        # identifier -> textual SHA mapping from the previous implementation.
        return {'protocol': PROTOCOL, 'layers': result}
