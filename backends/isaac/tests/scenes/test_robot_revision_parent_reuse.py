"""A cached robot revision reuses the parent validation it performed; nothing else does."""
from pathlib import Path

from bench.isaacsim import release
from bench.runtime import release_cache
from test_robot_revision_release import original_robot_catalog, publication


def test_cached_revision_validates_its_real_parent_once(publication, monkeypatch):
    target, _, parent, _, _ = publication('card_room', 'task0', 'single_arm')
    parent = Path(parent).resolve()
    counts = []
    genuine = release._validate

    def counted(root, *, assets=True):
        counts.append((Path(root).resolve(), assets))
        return genuine(root, assets=assets)
    monkeypatch.setattr(release, '_validate', counted)
    release_cache.clear()
    try:
        expected = release.task_inputs(target, 'task0', seed=0)
        assert counts.count((parent, True)) == 2  # Default: inside validate(), then again for inputs.
        del counts[:]
        assert release_cache.task_inputs_cached(target, 'task0', seed=0) == expected
        assert counts == [(parent, True)]  # Only inside the revision's own validation.
        del counts[:]
        assert release_cache.task_inputs_cached(target, 'task0', seed=0) == expected and counts == []
        # A parent input whose identity changes voids the proof; everything validates again.
        changed, current = str(parent/'release_manifest.json'), release_cache._identity
        monkeypatch.setattr(release_cache, '_identity', lambda path: None if path == changed else current(path))
        assert release_cache.task_inputs_cached(target, 'task0', seed=0) == expected
        assert counts == [(parent, True)]
    finally:
        release_cache.clear()
