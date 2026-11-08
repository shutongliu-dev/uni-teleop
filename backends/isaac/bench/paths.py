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
import re

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


ARCHIVES = ('.tar', '.tar.gz', '.tgz', '.zip', '.7z', '.tar.zst')


def undistributed(path) -> bool:
    """A file that frozen releases pin as provenance but the data package does not ship.

    The package carries rooms, assets and task releases. Development evidence
    (recordings, reports), frozen source snapshots and upstream source archives
    that releases still pin stay in the development store; their recorded
    hashes remain the provenance.
    """
    path = Path(path)
    if not path.is_absolute() or path.exists():
        return False
    return path.is_relative_to(Path(DATA_ROOT, 'datasets')) or (
        path.is_relative_to(Path(DATA_ROOT)) and path.name.endswith(ARCHIVES))


_LAYER_ASSET = re.compile(r'@(?!@)([^@\n]+)@')


def usd_layer_identity(path) -> str:
    """SHA256 of a text USD layer with its layer-relative asset paths written as data:// paths.

    The package stores layer-relative asset paths, so copies of one source layer at different
    depths differ in bytes while naming the same assets. Their layer identity is the same: compare
    it where a frozen release requires a copied layer to keep its source bytes. Binary layers and
    paths outside the data root keep their bytes.
    """
    import hashlib
    path = Path(path)
    data = path.read_bytes()
    if path.suffix != '.usda':
        return hashlib.sha256(data).hexdigest()
    root, directory = data_root(), path.parent

    def anchored(match):
        asset = match.group(1)
        if not asset.startswith(('./', '../')):
            return match.group(0)
        target = Path(os.path.normpath(directory / asset))
        return '@' + portable(str(target)) + '@' if target.is_relative_to(root) else match.group(0)

    text = _LAYER_ASSET.sub(anchored, data.decode('utf-8', errors='surrogateescape'))
    return hashlib.sha256(text.encode('utf-8', errors='surrogateescape')).hexdigest()


def relocated_layer_bytes(source, target) -> bytes:
    """A text USD layer as a copy at `target` writes it: layer-relative asset paths re-anchored there.

    The copy names the same assets and has the source's usd_layer_identity. Binary layers are
    copied unchanged.
    """
    source, target = Path(source), Path(target)
    data = source.read_bytes()
    if source.suffix != '.usda':
        return data

    def anchored(match):
        asset = match.group(1)
        if not asset.startswith(('./', '../')):
            return match.group(0)
        relative = os.path.relpath(os.path.normpath(source.parent / asset), target.parent)
        return '@' + (relative if relative.startswith('.') else './' + relative) + '@'

    text = _LAYER_ASSET.sub(anchored, data.decode('utf-8', errors='surrogateescape'))
    return text.encode('utf-8', errors='surrogateescape')


def usd_value_text(value) -> str:
    """str() of a USD attribute value, with asset paths written as the data:// file they resolve to.

    Layers in the package author asset paths relative to themselves; a flattened or copied stage
    authors the same assets absolute. Compare authored values in this form. (No pxr import here.)
    """
    kind = type(value).__name__
    if kind == 'AssetPath':
        return '@' + portable(value.resolvedPath or value.path) + '@'
    if kind == 'AssetPathArray':
        return '[' + ', '.join(usd_value_text(item) for item in value) + ']'
    return str(value)


def undistributed_bytecode(path) -> bool:
    """A CPython bytecode cache (``__pycache__/*.pyc``) that a frozen dependency inventory pins
    but the data package does not ship. Caches are rebuilt from the pinned sources; no validator
    uses their bytes, so the recorded hash stays provenance as for other undistributed files."""
    path = Path(path)
    return '__pycache__' in path.parts and path.suffix == '.pyc' and undistributed(path)


def load_json(path, **options):
    """Read JSON with its data:// references materialized; other JSON is unchanged."""
    return materialize(json.loads(Path(path).read_text(), **options))
