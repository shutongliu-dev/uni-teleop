"""Exercise the real source packager without loading Isaac or device SDKs."""
import os
from pathlib import Path
import tempfile
import unittest

from uni_teleop.runtime.isaac import prepare_snapshot


class IsaacSnapshotTests(unittest.TestCase):
    def test_current_backend_freezes_complete_teleop_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'source'
            prepare_snapshot(destination)
            for relative in (
                'bench/apps/workroom.headless.kit',
                'scenes/workrooms/parameterized/catalog_pins.json',
                'scripts/teleop/master_slave_isaac.py',
                'uni_teleop/devices/manus/native_process.py',
            ):
                self.assertTrue((destination / relative).is_file(), relative)
            self.assertFalse((destination / 'local').exists())
            self.assertFalse((destination / 'data/assets').exists())
            self.assertTrue(os.access(destination / 'scripts/isaac_python.sh', os.X_OK))


if __name__ == '__main__':
    unittest.main()
