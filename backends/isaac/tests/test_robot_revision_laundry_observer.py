"""The compatibility overlay cannot silently adopt a different frozen scorer."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.robot_revision import laundry_observer as patch


BASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/robot_l_mount_update_20261104')


def original_observer():
    return (BASE/'releases_candidate03/releases/laundry_room/towel_flattening/single_arm'/patch.OBSERVER_PATH).read_bytes()


def test_overlay_is_explicit_exact_and_self_contained(tmp_path):
    original = original_observer()
    assert hashlib.sha256(original).hexdigest() == patch.ORIGINAL_SHA256
    derived = patch.patch_observer(original)
    assert hashlib.sha256(derived).hexdigest() == patch.PATCHED_SHA256
    assert derived == patch.patch_observer(original)
    assert original_observer() == original  # No historical file is edited.
    path = tmp_path/'native_observer.py'; path.write_bytes(derived)
    spec = importlib.util.spec_from_file_location('independent_frozen_laundry_observer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.CookedRobotGeometry and module.NativeObserver and module.wrist_frame_bindings
    assert b'robot_revision.laundry_observer' not in derived


@pytest.mark.parametrize('value', [b'', b'unknown observer', bytearray(b'not bytes')])
def test_unknown_observer_cannot_receive_overlay(value):
    with pytest.raises(ValueError, match='Unrecognized frozen'):
        patch.patch_observer(value)


def test_changed_parent_and_changed_patch_source_are_both_rejected(monkeypatch):
    original = original_observer()
    with pytest.raises(ValueError, match='Unrecognized frozen'):
        patch.patch_observer(original+b'\n')
    read_bytes = Path.read_bytes
    def changed(path):
        result = read_bytes(path)
        return result+b'\n' if path.name == 'native_observer.py' else result
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='approved compatibility'):
        patch.patch_observer(original)
