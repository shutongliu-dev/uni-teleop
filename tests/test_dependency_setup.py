"""Installation helpers: local paths, original download hashes and no device access."""
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/setup_dependencies.py'
spec = importlib.util.spec_from_file_location('dependency_setup', SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class Dependencies(unittest.TestCase):
    def test_vendor_root_and_subdirectory_are_both_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / 'Python/manus_sdk'
            sdk.mkdir(parents=True)
            (sdk / '__init__.py').touch()
            for path in (root, root / 'Python'):
                self.assertEqual(setup.local_package(path, 'Python', 'manus_sdk/__init__.py'), root / 'Python')
            with self.assertRaises(ValueError):
                setup.local_package(root, 'retargeting', 'include/missing.so')

    def test_download_checks_original_content_and_does_not_reuse_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dependencies').mkdir()
            content = b'original fixture file'
            manifest = dict(pico=dict(raw_base_url='https://official.invalid/revision/',
                files=[dict(path='SDK/libPXREARobotSDK.so', sha256=sha256(content).hexdigest())]))
            (root / 'dependencies/sources.json').write_text(json.dumps(manifest))
            target = root / 'local/libPXREARobotSDK.so'
            with patch.object(setup, 'ROOT', root), patch.object(setup.platform, 'system', return_value='Linux'), \
                 patch.object(setup.platform, 'machine', return_value='x86_64'):
                with patch.object(setup, 'urlopen', return_value=BytesIO(b'wrong')):
                    with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                        setup.download_pico(target.parent)
                self.assertFalse(target.exists())
                with patch.object(setup, 'urlopen', return_value=BytesIO(content)):
                    self.assertEqual(setup.download_pico(target.parent), target)
                with patch.object(setup, 'urlopen', side_effect=AssertionError('must use verified cache')):
                    setup.download_pico(target.parent)
                target.write_bytes(b'corrupted cache')
                with patch.object(setup, 'urlopen', return_value=BytesIO(content)):
                    setup.download_pico(target.parent)
                self.assertEqual(target.read_bytes(), content)

    def test_python_mismatch_prevents_optimizer_environment_creation(self):
        with patch.object(setup.shutil, 'which', return_value='/fixture/python'), \
             patch.object(setup.subprocess, 'check_output', return_value='3.12\n'), \
             patch.object(setup, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'requires CPython 3.10'):
                setup.environment('python', Path('/not-created'), version='3.10')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
