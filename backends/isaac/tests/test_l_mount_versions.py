"""Explicit asset versions remain independent, including failed old candidates."""
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.l_mount import (
    SUPPORTED_VERSIONS, prepared_assembly, prepared_import)
from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec


def test_registered_versions_keep_distinct_identity_and_equal_mount():
    values = [get_hand_spec('sharpa_w01_right', version) for version in SUPPORTED_VERSIONS]
    assert [v['assembly_version'] for v in values] == list(SUPPORTED_VERSIONS)
    assert all(v['mounts_by_side'] == values[0]['mounts_by_side'] for v in values)
    with pytest.raises(ValueError):
        prepared_assembly('/missing', version='isaacsim_6_1_l_mount_v999')


def test_explicit_version_never_falls_back_or_accepts_wrong_source(tmp_path):
    with pytest.raises(FileNotFoundError, match=SUPPORTED_VERSIONS[-1]):
        prepared_assembly(tmp_path, version=SUPPORTED_VERSIONS[-1])
    (tmp_path/'SOURCE.json').write_text(json.dumps({'assembly_version': SUPPORTED_VERSIONS[-1]}))
    with pytest.raises(ValueError, match='another assembly version'):
        prepared_assembly(tmp_path, version=SUPPORTED_VERSIONS[0], output=tmp_path)


@pytest.mark.parametrize('side', [None, 'left', 'right'])
def test_old_l_mount_assembly_and_import_remain_loadable(side):
    assets = Path(f'{DATA_ROOT}/assets')
    path, source = prepared_assembly(assets, side=side, version=SUPPORTED_VERSIONS[0])
    assert SUPPORTED_VERSIONS[0] in str(path)
    assert source['assembly_version'] == SUPPORTED_VERSIONS[0]
    assert SUPPORTED_VERSIONS[0] in str(prepared_import(path, side=side))
