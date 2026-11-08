"""Task loading shares validation proofs only within a single load."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backends/isaac'))
from bench.isaacsim import release
from bench.isaacsim.validation_epoch import active_validation_scope
from bench.isaacsim.validation_epoch import FullByteOnceValidation
from bench.simulation.isaacsim_backend.workroom import digest
from scenes.workrooms.seed_geometry import file_hash
from scripts.teleop import pico_isaac


class LoadValidationTests(unittest.TestCase):
    def test_room_and_seed_hashing_share_the_same_file_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'asset'
            path.write_bytes(b'first')
            with FullByteOnceValidation('shared asset') as scope:
                before = digest(path)
                self.assertEqual(file_hash(path), before)
            self.assertEqual(scope.summary()['fresh_hash_calls'], 1)
            self.assertEqual(scope.summary()['hash_reuses'], 1)
            path.write_bytes(b'changed')
            with FullByteOnceValidation('next load'):
                self.assertNotEqual(digest(path), before)

    def test_repeated_release_validation_is_shared_but_not_across_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'release_manifest.json').write_text('{}')

            def load(_):
                first = release.validate(root)
                second = release.validate(root)
                self.assertEqual(first, second)
                return first

            with patch.object(release, '_validate', return_value={'valid': True}) as validate, \
                    patch.object(pico_isaac, '_load_inputs', side_effect=load), \
                    contextlib.redirect_stdout(io.StringIO()):
                pico_isaac.load_inputs(None)
                self.assertEqual(validate.call_count, 1)
                pico_isaac.load_inputs(None)
                self.assertEqual(validate.call_count, 2)
            self.assertIsNone(active_validation_scope())

    def test_failed_load_does_not_leave_cached_proofs(self):
        with patch.object(pico_isaac, '_load_inputs', side_effect=ValueError('bad asset')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'bad asset'):
                pico_isaac.load_inputs(None)
        self.assertIsNone(active_validation_scope())
