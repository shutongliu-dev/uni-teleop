"""Historical five-room scenes: discovery, loading and CPU validation.

Current furnished environments are Astra's six rooms in
``scenes.workrooms``. This package preserves the prior scene definitions
and explicit callers; its checks and results do not qualify the new environments.

A room (a *scene family* of the ``isaacsim-scene-family-v1`` schema) is one
furnished room plus its manipulation objects, authored as data under
``scenes/rooms/<family>.py``. Every family exposes ``FAMILY``, ``SCENE``
and ``build_library(asset_root)``. Their procedural asset libraries are versioned
under ``DEFAULT_LIBRARY_ROOT/<family>/<version>`` in shared assets; native
validation is ``scripts/validation/validate_scene_family.py``.

This module is the single place that knows the list, so a caller never hard-codes
family names. It imports family modules lazily: importing a family must not touch
a GPU, and a broken family must not prevent the others from being listed.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT

FAMILIES = ('laboratory', 'kitchen', 'study', 'card_room', 'living_room')
SCHEMA = 'isaacsim-scene-family-v1'
DEFAULT_LIBRARY_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/scene_rooms')


class FamilyNotFound(KeyError):
    """A family name that is not one of the authored rooms."""


def module_name(family, *, package='scenes.rooms'):
    if family not in FAMILIES:
        raise FamilyNotFound(f'{family!r} is not one of {FAMILIES}')
    return f'{package}.{family}'


def load(family, *, package='scenes.rooms'):
    """Import one family module (no GPU work at import time)."""
    return importlib.import_module(module_name(family, package=package))


def scene(family, *, package='scenes.rooms'):
    """The family's scene specification."""
    return load(family, package=package).SCENE


def library_root(family, *, root=DEFAULT_LIBRARY_ROOT):
    """Where the family's authored assets live, newest version first."""
    base = Path(root) / family
    if not base.is_dir():
        return None
    versions = sorted((p for p in base.iterdir() if p.is_dir() and p.name.startswith('v')),
                      key=lambda p: int(p.name[1:]))
    return versions[-1] if versions else None


def validate(family, *, package='scenes.rooms'):
    """CPU validation of one family; returns the validate_family record."""
    from bench.simulation.isaacsim_backend.scene_family import validate_family
    return validate_family(scene(family, package=package))


def validate_all(*, package='scenes.rooms'):
    """CPU validation of every family; never raises for one bad family."""
    results = {}
    for family in FAMILIES:
        try:
            results[family] = {'ok': True, 'record': validate(family, package=package)}
        except Exception as exc:
            results[family] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    return results


def summary(results):
    """A one-line-per-family report of a ``validate_all`` result."""
    lines = []
    for family in FAMILIES:
        entry = results.get(family, {})
        if entry.get('ok'):
            record = entry['record']
            lines.append(f'{family:22s} OK   objects={record["objects"]:3d} '
                         f'props={record["static_props"]:3d} '
                         f'cameras={",".join(record["cameras"])}')
        else:
            lines.append(f'{family:22s} FAIL {entry.get("error", "unknown")}')
    return '\n'.join(lines)


def build_libraries(*, root=DEFAULT_LIBRARY_ROOT, only=None):
    """Write each family's authored assets into its next version directory.

    Versions are never overwritten: the target is one past the newest existing
    ``vN``, and the family module must then be repointed to it explicitly.
    """
    written = {}
    for family in (only or FAMILIES):
        module = load(family)
        newest = library_root(family, root=root)
        target = Path(root) / family / (f'v{int(newest.name[1:]) + 1}' if newest else 'v1')
        written[family] = module.build_library(target)
    return written


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES, action='append')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    selected = args.family or list(FAMILIES)
    results = {family: validate(family) for family in selected}
    if args.json:
        print(json.dumps(results, indent=1, default=str))
    else:
        for family, record in results.items():
            print(f'{family:22s} OK   objects={record["objects"]:3d} '
                  f'props={record["static_props"]:3d} cameras={",".join(record["cameras"])}')


if __name__ == '__main__':
    main()
