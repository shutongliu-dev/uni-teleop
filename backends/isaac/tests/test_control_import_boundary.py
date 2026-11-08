"""Kinematics must not import the live executor's protocol stack."""
from pathlib import Path
import subprocess
import sys

import pytest


def test_kinematics_import_avoids_executor_dependencies():
    subprocess.run([sys.executable, '-B', '-c', '''
import importlib, sys
importlib.import_module('bench.control.kinematics')
assert 'bench.control.fingertip' not in sys.modules
assert not any(name == 'bench.protocol' or name.startswith('bench.protocol.') for name in sys.modules)
assert 'pydantic' not in sys.modules
'''], cwd=Path(__file__).resolve().parents[1], check=True)


def test_public_control_exports_remain_available():
    from bench import control
    from bench.control import fingertip

    for name in control.__all__:
        assert getattr(control, name) is getattr(fingertip, name)
    with pytest.raises(AttributeError):
        control.unknown_export
