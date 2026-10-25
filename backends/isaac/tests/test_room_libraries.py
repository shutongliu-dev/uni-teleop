"""Every room module must still reproduce the asset library its SCENE pins.

The libraries are published once and never rewritten
(<data>/assets/task_parts/scene_rooms/<room>/<version>), so the code
that generated them must keep generating the same thing: a geometry change has
to become a new version. Rebuilding into a scratch directory must give
byte-identical data files and an identical manifest, apart from build-location
provenance -- each ``generator`` field (the module path at build time) and the
``supersedes`` paths -- which legitimately changes when the module or the
library directory moves.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes import rooms

BUILD_LOCATION_KEYS = ('generator', 'supersedes')


def without_build_location(value):
    """The manifest with every build-location provenance field removed."""
    if isinstance(value, dict):
        return {k: without_build_location(v) for k, v in value.items() if k not in BUILD_LOCATION_KEYS}
    if isinstance(value, list):
        return [without_build_location(v) for v in value]
    return value


def data_files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob('*') if path.is_file() and path.name != 'manifest.json'}


@pytest.mark.parametrize('family', rooms.FAMILIES)
def test_the_module_rebuilds_its_pinned_library(family, tmp_path):
    module = rooms.load(family)
    pinned = Path(module.ASSET_ROOT)
    if not (pinned / 'manifest.json').is_file():
        pytest.skip(f'library not present on this host: {pinned}')
    assert pinned.parent.parent == rooms.DEFAULT_LIBRARY_ROOT, 'rooms pin libraries under the shared root'
    rebuilt = tmp_path / pinned.name
    module.build_library(rebuilt)
    assert data_files(rebuilt) == data_files(pinned), 'geometry changed: write a new library version'
    want = load_json(pinned / 'manifest.json')
    got = load_json(rebuilt / 'manifest.json')
    assert without_build_location(got) == without_build_location(want)
    # A new build records where the module lives now.
    assert got['generator'] in (f'scenes/rooms/{family}.py', f'scenes.rooms.{family}')
