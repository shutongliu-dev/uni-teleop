"""Locations of the DexBench data package and of run outputs.

The data package never stores machine paths: USD layers use layer-relative
asset paths and frozen JSON uses ``data://`` references, which are
materialized here against the local data root. Nothing here knows where a
simulator is installed; that is configured per checkout under runtime/.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / 'data' / 'MANIFEST.json'
DATA_SCHEME = 'data://'


def data_root() -> Path:
    """DEXBENCH_DATA, else a runtime/data link, else the checkout's data/ directory.

    The root is resolved once, so paths inside it never pass through links.
    """
    value = os.environ.get('DEXBENCH_DATA')
    if value:
        return Path(value).expanduser().resolve()
    linked = REPO_ROOT / 'runtime' / 'data'
    return (linked if linked.exists() else REPO_ROOT / 'data').resolve()


def output_root() -> Path:
    """DEXBENCH_OUTPUT, else the checkout's outputs/ directory."""
    value = os.environ.get('DEXBENCH_OUTPUT')
    return Path(value).expanduser().resolve() if value else (REPO_ROOT / 'outputs').resolve()


# Process-wide roots for module-level paths; set the environment before import.
DATA_ROOT = str(data_root())
OUTPUT_ROOT = str(output_root())


def data_path(relative='') -> Path:
    """A location inside the data package, from a package-relative POSIX path."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Data package paths must stay inside the package: {relative}')
    return data_root().joinpath(*relative.parts)


def data_uri(path) -> str:
    """The data:// reference for a file inside the local data package."""
    relative = Path(path).resolve().relative_to(data_root())
    return DATA_SCHEME + relative.as_posix()


def resolve(value):
    """Map one data:// reference to a local absolute path; pass anything else through."""
    if isinstance(value, str) and value.startswith(DATA_SCHEME):
        return str(data_path(value[len(DATA_SCHEME):]))
    return value


def materialize(value):
    """Resolve every data:// reference in frozen JSON values and mapping keys."""
    if isinstance(value, str):
        return resolve(value)
    if isinstance(value, Mapping):
        return {resolve(key): materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(materialize(item) for item in value)
    return value


def portable(value):
    """Inverse of materialize: local data paths become data:// references.

    Hash and record values in this form so they do not depend on where the
    data package lives on a given machine.
    """
    root = str(data_root())
    if isinstance(value, str):
        if value == root:
            return DATA_SCHEME
        return DATA_SCHEME + value[len(root) + 1:] if value.startswith(root + '/') else value
    if isinstance(value, Mapping):
        return {portable(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable(item) for item in value)
    return value


def load_json(path, **options):
    """Read JSON with its data:// references materialized; other JSON is unchanged."""
    return materialize(json.loads(Path(path).read_text(), **options))
