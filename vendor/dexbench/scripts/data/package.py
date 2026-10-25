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
SHA256 = re.compile(r'\b[0-9a-f]{64}\b')
SOURCE_ROOTS = ('bench', 'scenes', 'agent', 'scripts', 'tests', 'configs')
BUILTIN_MDL = re.compile(r'[A-Za-z0-9_]+\.mdl')


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


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


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
                if relative and not relative.startswith(SIMULATOR):
                    found.add(relative)
        if path.suffix in USD_SUFFIXES:
            from pxr import Sdf, UsdUtils
            layers, assets, _ = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
            for item in [layer.realPath for layer in layers] + list(assets):
                relative = self.relative(item)
                if relative and not relative.startswith(SIMULATOR):
                    found.add(relative)
        return found


def closure(store, seeds):
    """Every store file reachable from the seeds through file references.

    Seed directories are included whole. Other referenced directories (asset
    roots, parent releases, evidence folders) contribute only the files that
    are themselves referenced, and archives are never followed.
    """
    seeds = set(seeds)
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
        queue += sorted(store.references(path))
    return files


def _usd_reference(asset, store, layer_destination, output):
    relative = store.relative(asset)
    if relative is None:
        return asset
    if relative.startswith(SIMULATOR):
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
    for relative in files:
        source, destination = store.root / relative, output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        repository = (_repository_version(current.get(digest(source)), source)
                      if 'evaluator' in Path(relative).parts else None)
        if source.suffix in USD_SUFFIXES:
            _export_usd(source, destination, store, output)
        elif repository is not None:
            shutil.copyfile(repository, destination)
            synced[relative] = str(repository.relative_to(Path(__file__).resolve().parents[2]))
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
        old, new = digest(source), digest(destination)
        if old != new:
            remapped[old] = new
    propagate(output, files, remapped)
    manifest = build_manifest(output, files)
    report = {'files': len(files), 'bytes': sum(item['bytes'] for item in manifest['files'].values()),
              'rewritten_hashes': remapped, 'store_named': [], 'dangling': {}, 'rewritten_python': [],
              'synced_evaluators': synced}
    for relative in files:
        destination = output / relative
        if destination.suffix in USD_SUFFIXES:
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


def propagate(root, relatives, remapped):
    """Replace recorded old hashes inside package text files until nothing changes.

    A rewritten file gets a new hash, so every JSON or USDA file that records
    it changes too; the loop follows those changes through manifests.
    """
    while True:
        changed = {}
        for relative in relatives:
            path = Path(root) / relative
            if not is_text(path) and path.suffix != '.usda':
                continue
            text = path.read_text(errors='surrogateescape')
            updated = SHA256.sub(lambda match: remapped.get(match.group(0), match.group(0)), text)
            if updated != text:
                before = digest(path)
                path.write_text(updated, errors='surrogateescape')
                changed[before] = digest(path)
        if not changed:
            return remapped
        for old, new in list(remapped.items()):
            remapped[old] = changed.get(new, new)
        remapped.update(changed)


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
    root = Path(root)
    return {'schema': SCHEMA, 'files': {relative: {'sha256': digest(root / relative),
                                                   'bytes': (root / relative).stat().st_size}
                                        for relative in relatives}}


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def refresh(root):
    """After editing data: regenerate hashes in the package, source pins and data/MANIFEST.json."""
    from bench.paths import MANIFEST
    root = Path(root).resolve()
    recorded = json.loads(MANIFEST.read_text())['files']
    files = package_files(root)
    remapped = {}
    for relative in files:
        if relative in recorded:
            current = digest(root / relative)
            if current != recorded[relative]['sha256']:
                remapped[recorded[relative]['sha256']] = current
    propagate(root, files, remapped)
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
    problems = [relative for relative, record in manifest['files'].items()
                if not (root / relative).is_file() or digest(root / relative) != record['sha256']]
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
