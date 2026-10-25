"""Static gallery uses published dual-arm assets without rebuilding the release."""
import pytest

from bench.paths import data_root
from scripts.media import render_task_gallery as gallery


def test_published_dual_arm_models_are_read_only_and_complete():
    assets = data_root()/'assets'
    root = assets/'robots/assemblies/dual_fr3_sharpa_w01/isaacsim_6_1_v2'
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob('*') if p.is_file()}
    models = gallery.packaged_bimanual(assets, 'isaacsim_6_1_v2')
    assert set(models) == {'left', 'right'}
    assert all(p.is_file() and p.suffix == '.usda' for p in models.values())
    assert before == {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob('*') if p.is_file()}


def test_modified_dual_arm_asset_is_rejected(monkeypatch):
    monkeypatch.setattr(gallery, 'digest', lambda path: 'wrong')
    with pytest.raises(ValueError, match='Published dual-arm asset changed'):
        gallery.packaged_bimanual(data_root()/'assets', 'isaacsim_6_1_v2')
