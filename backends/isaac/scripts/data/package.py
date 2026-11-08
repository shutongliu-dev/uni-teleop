"""Build and check the DexBench data package.

`export` copies everything the runnable tasks and the test suite read from a
development store into a self-contained package that mirrors the store's
assets/ and datasets/ layout. JSON references become data:// paths, USD asset
paths become layer-relative, simulator MDL paths become core MDL names, frozen
evaluator copies of development sources become this repository's versions, and
every recorded SHA256 of a rewritten file is regenerated; files that need no
rewriting are copied byte for byte. `check` verifies a package against the
tracked data/MANIFEST.json; `refresh` regenerates hashes, source pins and
data/MANIFEST.json after data files were edited in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOTS = ('assets', 'datasets')
SIMULATOR = 'assets/simulators/'
# Example assets shipped inside a simulator distribution are task data, not part of the simulator.
SIMULATOR_DATA = re.compile(r'assets/simulators/newton/[^/]+/venv/lib/python3\.12/site-packages/(newton/examples/assets/|newton-[^/]+\.dist-info/licenses/)')
SCHEMA = 'dexbench_data_manifest.v1'
TEXT_SUFFIXES = {'.json', '.usda', '.py', '.txt', '.md', '.mtl', '.mtlx', '.toml', '.yaml', '.yml', '.csv',
                 '.urdf', '.xacro', '.srdf', '.xml', '.log', '.rst', '.rviz', '.asset_hash'}
TEXT_NAMES = {'SHA256SUMS', 'LICENSE', 'NOTICE'}
# Robot descriptions are read by URDF tools, which need file-relative mesh paths.
RELATIVE_SUFFIXES = {'.urdf', '.xacro', '.srdf'}
USD_SUFFIXES = {'.usd', '.usda', '.usdc'}
# Only runtime files are followed for JSON references; provenance and evidence
# records name sources that tasks never read.
FOLLOWED_JSON = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json',
                 'release_manifest.json', 'asset_file_hashes.json', 'scene_description.json'}
ARCHIVES = ('.tar', '.tar.gz', '.tgz', '.zip', '.7z', '.tar.zst')
# Runtime inputs can be referenced from these files even outside the frozen releases;
# inventories and manifests only record provenance.
RUNTIME_JSON = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json'}
# Development recordings, logs and media are evidence, never packaged unless seeded.
EVIDENCE = ('.npy', '.npz', '.jsonl', '.log', '.mp4', '.webm', '.gif')
# A recorded hash, not a content-addressed path component (directories named by a hash stay as they are).
SHA256 = re.compile(r'(?<![\w/])[0-9a-f]{64}(?![\w/])')
SOURCE_ROOTS = ('bench', 'scenes', 'agent', 'scripts', 'tests', 'configs')
BUILTIN_MDL = re.compile(r'[A-Za-z0-9_]+\.mdl')


def is_simulator(relative):
    return relative.startswith(SIMULATOR) and not SIMULATOR_DATA.match(relative)


def json_strings(value):
    """Every string key and value in a JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)
    elif isinstance(value, str):
        yield value


# Per-process caches: path -> ((size, mtime_ns, inode), value). Every rewrite changes mtime_ns, so
# a later pass in the same process (refresh, export, post-export rounds) re-reads changed files only.
_DIGESTS, _RECORDED = {}, {}


def _stamp(path):
    status = os.stat(path)
    return status.st_size, status.st_mtime_ns, status.st_ino


def digest(path):
    stamp, key = _stamp(path), os.fspath(path)
    hit = _DIGESTS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    _DIGESTS[key] = stamp, value.hexdigest()
    return _DIGESTS[key][1]


def _forget(path):
    """Drop cached values after this module rewrites a file (two writes can share one mtime tick)."""
    _DIGESTS.pop(os.fspath(path), None)
    _RECORDED.pop(os.fspath(path), None)


def _replace_text(path, text):
    """Rewrite a package file as a new file: identical package files can be hard links of one file."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text, errors='surrogateescape')
    os.replace(temporary, path)
    _forget(path)


def recorded_hashes(path):
    """The SHA256 values a text file records (cached like digest)."""
    stamp, key = _stamp(path), os.fspath(path)
    hit = _RECORDED.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    found = frozenset(SHA256.findall(Path(path).read_text(errors='surrogateescape')))
    _RECORDED[key] = stamp, found
    return found


def digests(root, relatives):
    """(sha256, bytes) of many package files; threads overlap the reads on a network file system."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    with ThreadPoolExecutor(int(os.environ.get('DEXBENCH_EXPORT_THREADS', '16'))) as pool:
        rows = pool.map(lambda relative: (digest(root / relative), (root / relative).stat().st_size), relatives)
        return dict(zip(relatives, rows))


class Store:
    """The development store whose absolute paths the source files contain."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.prefix = str(self.root) + '/'
        self.reference = re.compile(re.escape(self.prefix) + r'(?:assets|datasets)/[^"\'\s@<>()\[\],;|]*')

    def relative(self, path):
        path = str(path)
        if not path.startswith(self.prefix):
            return None
        relative = path[len(self.prefix):].rstrip('/')
        return relative if relative.split('/', 1)[0] in ROOTS else None

    def references(self, path):
        """Store-relative files and directories named by one source file."""
        found = set()
        if path.name in FOLLOWED_JSON or path.suffix == '.usda':
            text = path.read_text(errors='ignore')
            matches = self.reference.findall(text)
            if path.suffix == '.json':
                # File names may contain spaces, so whole JSON strings are references too.
                matches += [value for value in json_strings(json.loads(text)) if value.startswith(self.prefix)]
            for match in matches:
                relative = self.relative(match)
                if relative and not is_simulator(relative):
                    found.add(relative)
        if path.suffix in USD_SUFFIXES:
            from pxr import Sdf, UsdUtils
            layers, assets, _ = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
            for item in [layer.realPath for layer in layers] + list(assets):
                relative = self.relative(item)
                if relative and not is_simulator(relative):
                    found.add(relative)
        return found


def closure(store, seeds):
    """Every store file reachable from the seeds through file references.

    Seed directories (rooms, frozen task releases, robot assemblies) are
    included whole. Referenced assets are packaged. Outside the seeds, datasets/
    files are packaged only when runtime JSON references them and they are not
    recordings; everything else there is development evidence or provenance
    that releases merely pin. Archives are never followed.
    """
    seeds = set(seeds)
    seed_dirs = tuple(seed + '/' for seed in seeds if (store.root / seed).is_dir())
    files, queue, seen = set(), list(seeds), set()
    while queue:
        relative = queue.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = store.root / relative
        if path.is_dir():
            if relative in seeds:
                queue += [str(child.relative_to(store.root)) for child in path.rglob('*')
                          if child.is_file() and '__pycache__' not in child.parts]
            continue
        if (not path.is_file() or relative in files or '__pycache__' in path.parts
                or (relative not in seeds and path.name.endswith(ARCHIVES))):
            continue
        files.add(relative)
        inputs = path.name in RUNTIME_JSON
        queue += sorted(reference for reference in store.references(path)
                        if not reference.startswith('datasets/') or reference.startswith(seed_dirs)
                        or (inputs and not reference.endswith(EVIDENCE)))
    return files


def _usd_reference(asset, store, layer_destination, output):
    relative = store.relative(asset)
    if relative is None:
        return asset
    if is_simulator(relative):
        return os.path.basename(asset)
    path = os.path.relpath(output / relative, layer_destination.parent)
    return path if path.startswith('.') else './' + path


def _position(lines, offsets, row, column):
    """Text offset of an AST position, whose column counts UTF-8 bytes."""
    return offsets[row - 1] + len(lines[row - 1].encode()[:column].decode(errors='ignore'))


def _import_paths(text, name):
    """Add `name` to the module's `from bench.paths import ...` line, creating it if needed."""
    import ast
    existing = re.search(r'^from bench\.paths import (.+)$', text, re.M)
    if existing:
        if name in [item.strip() for item in existing.group(1).split(',')]:
            return text
        return text[:existing.end()] + ', ' + name + text[existing.end():]
    body = ast.parse(text).body
    index = 1 if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) else 0
    while index < len(body) and isinstance(body[index], ast.ImportFrom) and body[index].module == '__future__':
        index += 1
    lines = text.splitlines(keepends=True)
    position = sum(len(line) for line in lines[:body[index].lineno - 1]) if index < len(body) else len(text)
    return text[:position] + f'from bench.paths import {name}\n' + text[position:]


def _rewrite_json_reads(text):
    """json.loads(<path>.read_text(), **options) -> load_json(<path>, **options), which resolves data:// paths."""
    import ast
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'loads'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'json' and len(node.args) == 1
                and isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Attribute)
                and node.args[0].func.attr == 'read_text' and not node.args[0].args and not node.args[0].keywords):
            arguments = [ast.get_source_segment(text, node.args[0].func.value)]
            arguments += [ast.get_source_segment(text, keyword) for keyword in node.keywords]
            edits.append((_position(lines, offsets, node.lineno, node.col_offset),
                          _position(lines, offsets, node.end_lineno, node.end_col_offset),
                          'load_json(' + ', '.join(arguments) + ')'))
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    return _import_paths(text, 'load_json') if edits else text


def _rewrite_python(text, store):
    """Make store literals in packaged Python sources (frozen evaluators) relative to DATA_ROOT
    and their JSON reads resolve data:// paths, as in this repository's own sources."""
    import ast
    import io
    import tokenize
    try:
        tree = ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, tokenize.TokenError):
        return _rewrite_text(text, store)
    docstrings = {(node.body[0].value.lineno, node.body[0].value.col_offset) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits, uses_root = [], False
    for token in tokens:
        if str(store.root) not in token.string:
            continue
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        if token.type == tokenize.COMMENT or token.start in docstrings:
            edits.append((start, end, _rewrite_text(token.string, store)))
        elif token.type == tokenize.FSTRING_MIDDLE:
            edits.append((start, end, token.string.replace(store.prefix, '{DATA_ROOT}/').replace(str(store.root), '{DATA_ROOT}')))
            uses_root = True
        elif token.type == tokenize.STRING:
            match = re.match(r'([rRuUbB]*)(\'\'\'|"""|\'|")', token.string)
            prefix, quote = match.group(1), match.group(2)
            if 'b' in prefix.lower():
                continue
            inner = token.string[len(prefix) + len(quote):-len(quote)]
            inner = (inner.replace('{', '{{').replace('}', '}}').replace(store.prefix, '{DATA_ROOT}/')
                     .replace(str(store.root), '{DATA_ROOT}'))
            edits.append((start, end, f'{prefix}f{quote}{inner}{quote}'))
            uses_root = True
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    if uses_root:
        text = _import_paths(text, 'DATA_ROOT')
    return _rewrite_json_reads(text)


def _prune_builtin_mdl(value):
    """Simulator MDL modules are not package files: drop them from dependency inventories."""
    if isinstance(value, dict):
        return {key: _prune_builtin_mdl(item) for key, item in value.items() if not BUILTIN_MDL.fullmatch(str(key))}
    if isinstance(value, list):
        return [_prune_builtin_mdl(item) for item in value if not (isinstance(item, str) and BUILTIN_MDL.fullmatch(item))]
    return value


def _rewrite_json(text, store):
    text = _rewrite_text(text, store)
    value = json.loads(text)
    pruned = _prune_builtin_mdl(value)
    return text if pruned == value else json.dumps(pruned, indent=2, ensure_ascii=False) + '\n'


def is_text(path):
    return Path(path).suffix in TEXT_SUFFIXES or Path(path).name in TEXT_NAMES


def _rewrite_relative(text, store, destination, output):
    """Store paths in robot descriptions become paths relative to the file itself."""
    def relative(match):
        path = os.path.relpath(output / match.group(0)[len(store.prefix):], destination.parent)
        return path if path.startswith('.') else './' + path
    return store.reference.sub(relative, text)


def _rewrite_text(text, store):
    text = re.sub(re.escape(store.prefix) + r'assets/simulators/[^"\'\s@<>()\[\],;|]*/([^/"\'\s@<>()\[\],;|]+\.mdl)',
                  r'\1', text)
    return text.replace(store.prefix, 'data://').replace(str(store.root), 'data://')


def development_sources(checkout):
    """SHA256 of each Python file in a development checkout -> this repository's versions of that file."""
    if checkout is None:
        return {}
    checkout, repo = Path(checkout).resolve(), Path(__file__).resolve().parents[2]
    found = {}
    for directory, _, names in sorted(os.walk(checkout)):  # symlinked environments are not followed
        for name in sorted(names):
            relative = Path(directory, name).relative_to(checkout)
            if name.endswith('.py') and '__pycache__' not in relative.parts and (repo / relative).is_file():
                found.setdefault(digest(checkout / relative), []).append(repo / relative)
    return found


def _repository_version(candidates, frozen):
    """The repository file a frozen copy came from, when its name and content make that unambiguous."""
    named = [path for path in candidates or () if path.name == frozen.name]
    return named[0] if named and len({digest(path) for path in named}) == 1 else None


def _write(source, destination, original, updated, errors='strict'):
    """Keep the source bytes, line endings included, when a rewrite changes nothing."""
    if updated == original:
        shutil.copyfile(source, destination)
    else:
        destination.write_text(updated, errors=errors)


def _export_usd(source, destination, store, output):
    from pxr import Sdf, UsdUtils
    layer = Sdf.Layer.FindOrOpen(str(source))
    changed = []

    def reference(asset):
        value = _usd_reference(asset, store, destination, output)
        if value != asset:
            changed.append(asset)
        return value

    UsdUtils.ModifyAssetPaths(layer, reference)
    if store.prefix in layer.documentation:
        layer.documentation = _rewrite_text(layer.documentation, store)
        changed.append('documentation')
    if source.suffix == '.usda' and str(store.root) in source.read_text(errors='ignore'):
        changed.append('text')
    if not changed:
        shutil.copyfile(source, destination)
        return
    if not layer.Export(str(destination)):
        raise RuntimeError(f'Could not export {source}')
    if source.suffix == '.usda':
        destination.write_text(_rewrite_text(destination.read_text(), store))


def export(store, seeds, output, sources=None):
    """Copy the closure of `seeds` from `store` into `output`.

    `sources` is the development checkout the store's releases were frozen from.
    Frozen evaluator files identical to one of its sources are replaced by this
    repository's version of that source, which validators compare and import.
    """
    from pxr import Sdf
    store = Store(store)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'Output must be new or empty: {output}')
    files = sorted(closure(store, seeds))
    current = development_sources(sources)
    remapped, synced = {}, {}

    def copy(relative):
        """One file; returns (old digest, new digest, repository source if synced)."""
        source, destination = store.root / relative, output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        repository = (_repository_version(current.get(digest(source)), source)
                      if 'evaluator' in Path(relative).parts else None)
        if source.suffix in USD_SUFFIXES:
            _export_usd(source, destination, store, output)
        elif repository is not None:
            shutil.copyfile(repository, destination)
        elif source.suffix == '.py':
            text = source.read_text()
            _write(source, destination, text, _rewrite_python(text, store))
        elif source.suffix == '.json':
            text = source.read_text()
            _write(source, destination, text, _rewrite_json(text, store))
        elif source.suffix in RELATIVE_SUFFIXES:
            text = source.read_text()
            _write(source, destination, text, _rewrite_text(_rewrite_relative(text, store, destination, output), store))
        elif is_text(source):
            text = source.read_text(errors='surrogateescape')
            _write(source, destination, text, _rewrite_text(text, store), errors='surrogateescape')
        else:
            shutil.copyfile(source, destination)
        return digest(source), digest(destination), repository

    # USD layers go through the USD library one at a time; every other file is plain I/O and
    # hashing, which threads overlap on a network file system.
    from concurrent.futures import ThreadPoolExecutor
    usd = [relative for relative in files if Path(relative).suffix in USD_SUFFIXES]
    plain = [relative for relative in files if Path(relative).suffix not in USD_SUFFIXES]
    with ThreadPoolExecutor(int(os.environ.get('DEXBENCH_EXPORT_THREADS', '16'))) as pool:
        results = list(zip(plain, pool.map(copy, plain)))
    results += [(relative, copy(relative)) for relative in usd]
    groups = {}
    for relative, (old, new, repository) in results:
        if repository is not None:
            synced[relative] = str(repository.relative_to(Path(__file__).resolve().parents[2]))
        if old != new:
            remapped[old] = new
        groups.setdefault(old, set()).add(new)
    propagate(output, files, remapped)
    settle_path_pins(output, files, remapped, location_dependent_suspects(output, files, groups),
                     relocated_copies(output, files, groups))
    manifest = build_manifest(output, files)
    report = {'files': len(files), 'bytes': sum(item['bytes'] for item in manifest['files'].values()),
              'rewritten_hashes': remapped, 'store_named': [], 'dangling': {}, 'rewritten_python': [],
              'synced_evaluators': synced}
    for relative in files:
        destination = output / relative
        if destination.suffix in USD_SUFFIXES and destination.suffix != '.usda':  # binary layers
            text = Sdf.Layer.FindOrOpen(str(destination)).ExportToString()
        elif destination.suffix in TEXT_SUFFIXES:
            text = destination.read_text(errors='ignore')
        else:
            continue
        if store.prefix in text:
            report['store_named'].append(relative)
        if destination.suffix == '.json':
            uris = [value for value in json_strings(json.loads(text)) if value.startswith('data://')]
        else:
            uris = re.findall(r'data://[^"\'\s@<>()\[\],;|]*', text)
        missing = sorted({uri for uri in uris if not (output / uri[len('data://'):]).exists()})
        if missing:
            report['dangling'][relative] = missing
        if (destination.suffix == '.py' and relative not in synced
                and digest(destination) != digest(store.root / relative)):
            report['rewritten_python'].append(relative)
    print(f"{len(files)} files, {report['bytes'] / 1e9:.2f} GB, {len(remapped)} rewritten hashes, "
          f"{len(report['store_named'])} files still naming the store, {len(report['dangling'])} with dangling "
          f"data:// references, {len(report['rewritten_python'])} rewritten Python files, "
          f"{len(synced)} evaluator files from this repository")
    return manifest, report


RECORD = re.compile(r'\{[^{}]*\}')
RECORD_SIZE = re.compile(r'("(?:bytes|size_bytes)"\s*:\s*)(\d+)')
RECORD_HASH = re.compile(r'"sha256"\s*:\s*"([0-9a-f]{64})"')


def _record_sizes(text, sizes):
    """A size recorded next to exactly one package file hash follows that file."""
    def record(match):
        body = match.group(0)
        hashes = RECORD_HASH.findall(body)
        if len(hashes) != 1 or hashes[0] not in sizes:
            return body
        return RECORD_SIZE.sub(lambda size: size.group(1) + str(sizes[hashes[0]]), body)
    return RECORD.sub(record, text)


def propagate(root, relatives, remapped):
    """Replace recorded old hashes (and the sizes recorded with them) in package text files.

    A rewritten file gets a new hash and size, so every JSON or USDA file that
    records it changes too. Text files are rewritten once each, in dependency
    order: a file is rewritten after every package file whose hash it records,
    so the hashes it records are already final. (Frozen releases nest deeply;
    rewriting everything until nothing changes re-read the package per level.)
    `remapped` (old -> new hash) is extended with every change and returned.
    """
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    rows = digests(root, relatives)
    sizes = {value: size for value, size in rows.values()}
    texts = [relative for relative in relatives if is_text(root / relative)]

    def recorded(relative):
        return relative, recorded_hashes(root / relative)

    with ThreadPoolExecutor(int(os.environ.get('DEXBENCH_EXPORT_THREADS', '16'))) as pool:
        contains = dict(pool.map(recorded, texts))

    def current(value):
        """The hash a recorded value names now, following earlier remaps."""
        seen = set()
        while value in remapped and value not in seen:
            seen.add(value)
            value = remapped[value]
        return value

    producers = {}
    for relative in texts:
        producers.setdefault(rows[relative][0], []).append(relative)
    waiting, users = {}, {}
    for relative in texts:
        needs = {producer for value in contains[relative] for producer in producers.get(current(value), ())
                 if producer != relative}
        waiting[relative] = len(needs)
        for producer in needs:
            users.setdefault(producer, []).append(relative)
    final = {}  # current hash of a rewritten file -> its new hash

    def mapped(value):
        value = current(value)
        return final.get(value, value)

    ready = sorted(relative for relative, count in waiting.items() if count == 0)
    done = 0
    while ready:
        relative = ready.pop()
        done += 1
        if any(mapped(value) != value for value in contains[relative]):
            path = root / relative
            text = path.read_text(errors='surrogateescape')
            updated = SHA256.sub(lambda match: mapped(match.group(0)), text)
            if path.suffix == '.json':
                updated = _record_sizes(updated, sizes)
            if updated != text:
                _replace_text(path, updated)
                value, size = digest(path), path.stat().st_size
                final[rows[relative][0]] = value
                sizes[value] = size
        for user in users.get(relative, ()):
            waiting[user] -= 1
            if waiting[user] == 0:
                ready.append(user)
    if done != len(texts):
        raise RuntimeError('Recorded hashes form a cycle; cannot order the rewrite')
    for old, new in list(remapped.items()):
        remapped[old] = mapped(new)
    remapped.update(final)
    return remapped


# Store and repository files whose exported bytes depend on where they live (layer- and
# file-relative paths): identical sources at different depths export differently.
LOCATION_DEPENDENT = USD_SUFFIXES | RELATIVE_SUFFIXES
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _path_pins(value, base, found):
    """(key as written, recorded hash, package-relative file) for every hash recorded next to its file.

    Path-keyed maps ({"data://p": hash} or {"data://p": {"sha256": hash}}), path records
    ({"path": "data://p", "sha256"/"manifest_sha256": hash}), sibling pins ({"x": "data://p",
    "x_sha256": hash}), USD/URDF rows with one path ({"usd_path": "data://p.usda", "sha256": hash})
    and, in release manifests, files maps keyed by release-relative names.
    """
    if isinstance(value, dict):
        paths = [key for key, item in value.items() if isinstance(key, str) and (key == 'path' or key.endswith('_path'))
                 and isinstance(item, str) and item.startswith(DATA_URI)]
        for key, item in value.items():
            if isinstance(key, str) and key.startswith(DATA_URI):
                recorded = item.get('sha256') if isinstance(item, dict) else item
                if isinstance(recorded, str) and FULL_SHA256.match(recorded):
                    found.append((key, recorded, key[len(DATA_URI):]))
            elif isinstance(item, str) and item.startswith(DATA_URI):
                row = paths == [key] and Path(item).suffix in LOCATION_DEPENDENT
                for pin in {key + '_sha256', key.removesuffix('_path') + '_sha256',
                            'sha256' if key == 'path' or row else None}:
                    recorded = value.get(pin) if pin else None
                    if isinstance(recorded, str) and FULL_SHA256.match(recorded):
                        found.append((pin, recorded, item[len(DATA_URI):]))
                if key == 'path' and isinstance(value.get('manifest_sha256'), str):
                    found.append(('manifest_sha256', value['manifest_sha256'], item[len(DATA_URI):] + '/release_manifest.json'))
        if base is not None:
            for name in ('files_sha256', 'files'):
                for relative, item in (value.get(name) or {}).items() if isinstance(value.get(name), dict) else ():
                    recorded = item.get('sha256') if isinstance(item, dict) else item
                    if isinstance(recorded, str) and FULL_SHA256.match(recorded) and not relative.startswith(DATA_URI):
                        found.append((relative, recorded, f'{base}/{relative}'))
        for item in value.values():
            _path_pins(item, None, found)
    elif isinstance(value, list):
        for item in value:
            _path_pins(item, None, found)


DATA_URI = 'data://'


def repair_path_pins(root, relatives, suspects):
    """Rewrite hashes recorded next to a package file to that file's actual hash.

    Hash-only propagation cannot tell apart package files exported from identical sources at
    different locations (USD, URDF): it maps their shared source hash to one of their new
    hashes everywhere. Only JSON files recording one of `suspects` are inspected; returns
    old -> new hashes of the JSON files changed.
    """
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    if not suspects:
        return {}
    rows = digests(root, relatives)
    texts = [relative for relative in relatives if relative.endswith('.json')]

    def inspect(relative):
        if suspects.isdisjoint(recorded_hashes(root / relative)):
            return relative, None
        text = (root / relative).read_text(errors='surrogateescape')
        found = []
        base = str(Path(relative).parent) if Path(relative).name == 'release_manifest.json' else None
        _path_pins(json.loads(text), base, found)
        fixes, conflicts = {}, []
        for key, recorded, target in found:
            actual = rows.get(target, (None,))[0]
            # Only a hash that names another location-dependent package file is a clash; other
            # values next to a path (a pixel hash beside its file's own pin, say) stay as recorded.
            if actual is None or actual == recorded or recorded not in suspects:
                continue
            previous = fixes.setdefault((key, recorded), actual)
            if previous != actual:
                conflicts.append((key, recorded))
        if not fixes:
            return relative, None
        updated = text
        for (key, recorded), actual in fixes.items():
            if (key, recorded) in conflicts:
                continue
            pattern = re.compile(re.escape(json.dumps(key, ensure_ascii=False)) + r'(\s*:\s*(?:\{\s*"sha256"\s*:\s*)?")'
                                 + recorded + '"')
            updated = pattern.sub(lambda match: match.group(0).replace(recorded, actual), updated)
        if updated == text:
            return relative, None
        _replace_text(root / relative, updated)
        return relative, (rows[relative][0], digest(root / relative), sorted(set(conflicts)))

    changed, unresolved = {}, []
    with ThreadPoolExecutor(int(os.environ.get('DEXBENCH_EXPORT_THREADS', '16'))) as pool:
        for relative, result in pool.map(inspect, texts):
            if result:
                changed[result[0]] = result[1]
                unresolved += [(relative,) + item for item in result[2]]
    if unresolved:
        print(f'{len(unresolved)} path pins with ambiguous recorded hashes left unchanged', file=sys.stderr)
    return changed


def location_dependent_suspects(root, relatives, groups=None):
    """Hashes that may name the wrong one of several location-dependent exports of one source."""
    root = Path(root)
    if groups is not None:  # export: source hash -> new hashes of its package copies
        return {value for news in groups.values() if len(news) > 1 for value in news}
    rows = digests(root, [relative for relative in relatives if Path(relative).suffix in LOCATION_DEPENDENT])
    return {value for value, _ in rows.values()}


def _nearest(relative, candidates):
    """The candidate path sharing the longest directory prefix with `relative` (ties: sorted first)."""
    parts = relative.split('/')

    def shared(path):
        count = 0
        for mine, theirs in zip(parts, path.split('/')):
            if mine != theirs:
                break
            count += 1
        return count
    return max(sorted(candidates), key=shared)


def repair_relocated_hashes(root, relatives, copies):
    """A relocated copy's hash recorded without a path names the copy nearest to the recording file.

    `copies` maps each package hash of a location-dependent file to the package paths of every copy
    of the same source. Values recorded next to a path are repair_path_pins' business; elsewhere
    (e.g. {"urdf_sha256": ...} beside the URDF it describes) the nearest copy is the one meant.
    Returns old -> new hashes of the JSON files changed.
    """
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    if not copies:
        return {}
    rows = digests(root, relatives)
    texts = [relative for relative in relatives if relative.endswith('.json')]

    def inspect(relative):
        found = recorded_hashes(root / relative) & copies.keys()
        if not found:
            return relative, None
        text = (root / relative).read_text(errors='surrogateescape')
        pins = []
        base = str(Path(relative).parent) if Path(relative).name == 'release_manifest.json' else None
        _path_pins(json.loads(text), base, pins)
        pinned = {recorded for _, recorded, _ in pins}
        updated = text
        for value in found - pinned:
            actual = rows[_nearest(relative, copies[value])][0]
            if actual != value:
                updated = SHA256.sub(lambda match: actual if match.group(0) == value else match.group(0), updated)
        if updated == text:
            return relative, None
        _replace_text(root / relative, updated)
        return relative, (rows[relative][0], digest(root / relative))

    with ThreadPoolExecutor(int(os.environ.get('DEXBENCH_EXPORT_THREADS', '16'))) as pool:
        return dict(result for _, result in pool.map(inspect, texts) if result)


def relocated_copies(root, relatives, groups):
    """package hash -> paths of all copies of its source, for sources exported to several hashes."""
    rows = digests(root, relatives)
    paths = {}
    for relative in relatives:
        paths.setdefault(rows[relative][0], []).append(relative)
    copies = {}
    for news in groups.values():
        if len(news) > 1:
            members = sorted(path for value in news for path in paths.get(value, ()))
            for value in news:
                copies[value] = members
    return copies


def settle_path_pins(root, relatives, remapped, suspects, copies=None):
    """repair_path_pins (and repair_relocated_hashes), then carry the repaired files' new hashes,
    until no pin changes."""
    for _ in range(10):
        changed = repair_path_pins(root, relatives, suspects)
        if copies:
            changed.update(repair_relocated_hashes(root, relatives, copies))
        if not changed:
            return remapped
        propagate(root, relatives, changed)
        for old, new in list(remapped.items()):
            remapped[old] = changed.get(new, new)
        remapped.update(changed)
        suspects = set(suspects) | set(changed.values())
    raise SystemExit('Path pins did not settle')


def update_sources(remapped, repo):
    """Carry hash changes into source files that pin data (definitions, tests, configs)."""
    changed = []
    for top in SOURCE_ROOTS:
        for path in sorted((Path(repo) / top).rglob('*')):
            if path.suffix not in ('.py', '.json') or not path.is_file():
                continue
            text = path.read_text()
            updated = SHA256.sub(lambda match: remapped.get(match.group(0), match.group(0)), text)
            if updated != text:
                path.write_text(updated)
                changed.append(str(path.relative_to(repo)))
    return changed


def package_files(root):
    root = Path(root)
    return sorted(str(Path(directory, name).relative_to(root))
                  for top in ROOTS for directory, _, names in os.walk(root / top, followlinks=True)
                  for name in names if '__pycache__' not in directory)


def build_manifest(root, relatives):
    rows = digests(root, list(relatives))
    return {'schema': SCHEMA, 'files': {relative: {'sha256': rows[relative][0], 'bytes': rows[relative][1]}
                                        for relative in relatives}}


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def refresh(root):
    """After editing data: regenerate hashes in the package, source pins and data/MANIFEST.json."""
    from bench.paths import MANIFEST
    root = Path(root).resolve()
    recorded = json.loads(MANIFEST.read_text())['files']
    files = package_files(root)
    rows = digests(root, [relative for relative in files if relative in recorded])
    remapped = {recorded[relative]['sha256']: current for relative, (current, _) in rows.items()
                if current != recorded[relative]['sha256']}
    propagate(root, files, remapped)
    settle_path_pins(root, files, remapped, location_dependent_suspects(root, files))
    sources = update_sources(remapped, Path(__file__).resolve().parents[2])
    write_manifest(MANIFEST, build_manifest(root, files))
    print(f'{len(files)} files, {len(remapped)} changed hashes, {len(sources)} source files updated')
    return 0


def check(root):
    """Verify the local data against the version this checkout expects (data/MANIFEST.json)."""
    from bench.paths import MANIFEST
    root = Path(root).resolve()
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('schema') != SCHEMA:
        raise SystemExit('Unknown data manifest schema')
    rows = digests(root, [relative for relative in manifest['files'] if (root / relative).is_file()])
    problems = [relative for relative, record in manifest['files'].items()
                if relative not in rows or rows[relative][0] != record['sha256']]
    for relative in problems[:20]:
        print(f'missing or changed: {relative}', file=sys.stderr)
    print(f"{len(manifest['files'])} files: {'ok' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    exporter = commands.add_parser('export', help='maintainers: build a package from a development store')
    exporter.add_argument('--store', type=Path, required=True)
    exporter.add_argument('--seeds', type=Path, required=True, help='file with one store-relative path per line')
    exporter.add_argument('--output', type=Path, required=True)
    exporter.add_argument('--report', type=Path, required=True, help='JSON report written outside the package')
    exporter.add_argument('--sources', type=Path, help='development checkout the store releases were frozen from')
    checker = commands.add_parser('check', help='verify a package against data/MANIFEST.json')
    checker.add_argument('root', type=Path, nargs='?')
    refresher = commands.add_parser('refresh', help='after editing data: regenerate hashes and data/MANIFEST.json')
    refresher.add_argument('root', type=Path, nargs='?')
    args = parser.parse_args(argv)
    if args.command == 'export':
        seeds = [line.strip() for line in args.seeds.read_text().splitlines() if line.strip()]
        manifest, report = export(args.store, seeds, args.output, args.sources)
        args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
        # The package never carries its own manifest: unpacking it must not replace the tracked one.
        from bench.paths import MANIFEST
        write_manifest(MANIFEST, manifest)
        return 0
    if args.root is None:
        from bench.paths import data_root
        args.root = data_root()
    return refresh(args.root) if args.command == 'refresh' else check(args.root)


if __name__ == '__main__':
    raise SystemExit(main())
