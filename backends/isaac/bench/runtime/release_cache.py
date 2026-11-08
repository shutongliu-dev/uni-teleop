"""Process-local reuse of successful release validations on server request paths.

Only the canonical server and service request paths use this module. Authors,
offline audits and every explicit validation scope keep calling validate()
itself. A native worker may adopt its server's validations only through an
explicit, pinned handoff that it re-proves (adopt_worker_handoff); without one,
or when any check fails, it validates the release itself. A success is keyed
by resolved root, the SHA-256 of release_manifest.json bytes and the assets
flag. It is reused only while every file the validation opened read-only and
every directory it listed (import machinery excluded) keeps its identity:
device, inode, size, mtime, ctime and mode. Identities are taken just before
each first open and rechecked after success, so inputs changed during a
validation are never cached; failures are never cached. Metadata is a change
detector here, not byte evidence: a writer restoring every stat field defeats
it, as it defeats any timestamp cache. Stat-only predicates on paths that were
never opened or listed are not re-proved.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from bench.isaacsim import release as _release

# Only the release module's own loaders are cached. A substituted validator or
# loader (a test double or operator wrapper) is still called on every request.
_GENUINE_VALIDATE, _GENUINE_TASK_INPUTS = _release.validate, _release.task_inputs
_READS = ContextVar('runtime_release_validation_reads', default=None)
_EVENTS = frozenset({'open', 'os.listdir', 'os.scandir'})
_WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_IMPORT_CODE = ('<frozen importlib', os.path.dirname(importlib.__file__)+os.sep)
_LOCK = threading.Lock()
_CACHE = OrderedDict()
_GATES = {}
_LIMIT = 8
# Shared-storage metadata is slow per call but concurrent (gear: 34k files).
_STAT_WORKERS = 16
_STAT_PARALLEL_FROM = 512
_HOOKED = False
_PROJECT = Path(__file__).resolve().parents[2]
HANDOFF_SCHEMA = 'release_validation_handoff.v1'
REUSE_SCHEMA = 'release_validation_reuse.v1'


def _identity(path):
    try:
        value = os.stat(path)
    except OSError:
        return None
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


class _Reads:
    """Identity of every path opened read-only or listed in one validation context."""
    def __init__(self):
        self.files = {}
        self.complete = True

    def observe(self, event, args):
        target = args[0]
        if event == 'open' and args[2] & _WRITES:
            return
        if target is None:
            target = '.'
        if isinstance(target, int):
            return  # A descriptor was observed when its path was opened.
        path = os.fsdecode(target)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if event != 'open' and os.path.normpath(path) in _import_roots():
            return  # Distribution/module discovery scans, never a release inventory.
        if path not in self.files:
            self.files[path] = _identity(path)


def _import_roots():
    return {os.path.normpath(os.path.join(os.getcwd(), entry)) for entry in sys.path
            if isinstance(entry, str)}


def _audit(event, args, _reads=_READS, _events=_EVENTS):
    reads = _reads.get()
    if reads is None or event not in _events:
        return
    try:
        # Imports and package metadata are process state, not release inputs: a
        # module loads once per process, and discovery lists sys.path entries
        # that may churn (a launcher's own directory, logs beside a script).
        if not sys._getframe(1).f_code.co_filename.startswith(_IMPORT_CODE):
            reads.observe(event, args)
    except Exception:
        reads.complete = False


def _install():
    global _HOOKED
    with _LOCK:
        if not _HOOKED:
            sys.addaudithook(_audit)  # Permanent; inert outside a recording context.
            _HOOKED = True


def _identities(paths):
    if len(paths) < _STAT_PARALLEL_FROM:
        return [_identity(path) for path in paths]
    with ThreadPoolExecutor(_STAT_WORKERS, thread_name_prefix='release-cache-stat') as pool:
        return list(pool.map(_identity, paths, chunksize=128))


def _unchanged(files):
    paths = tuple(files)
    return all(files[path] == current for path, current in zip(paths, _identities(paths)))


def _manifest_sha256(root):
    with (root/'release_manifest.json').open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _reuse(key):
    with _LOCK:
        entry = _CACHE.get(key)
    if entry is None:
        return None
    current = _unchanged(entry[1])
    with _LOCK:
        if _CACHE.get(key) is entry:
            if current:
                _CACHE.move_to_end(key)
            else:
                del _CACHE[key]
    return entry if current else None


def validate_cached(root, *, assets=True):
    """Equivalent to validate(root, assets=assets), reusing an unchanged success."""
    return _validated(root, assets)[0]


def _validated(root, assets):
    """(validate() result, parents it validated), as validate_with_parents() reports.

    A parent result is cached with, and proved by, its child's identity snapshot:
    the nested parent validation read its inputs inside the child's recording.
    """
    from bench.isaacsim.validation_epoch import active_validation_scope
    resolved = Path(root).resolve()
    if (_release.validate is not _GENUINE_VALIDATE or active_validation_scope() is not None
            or _READS.get() is not None or not (resolved/'release_manifest.json').is_file()):
        # Scopes own their reuse semantics; a nested call is observed by its
        # outer recording; validate() reports a missing manifest itself.
        return _release.validate(root, assets=assets), {}
    _install()
    entry = _reuse((str(resolved), _manifest_sha256(resolved), bool(assets)))
    if entry is not None:
        return deepcopy(entry[0]), deepcopy(entry[2])
    with _LOCK:
        gate = _GATES.setdefault((str(resolved), bool(assets)), threading.Lock())
    with gate:  # Concurrent misses wait for one validation, then reuse it.
        entry = _reuse((str(resolved), _manifest_sha256(resolved), bool(assets)))
        if entry is not None:
            return deepcopy(entry[0]), deepcopy(entry[2])
        reads = _Reads()
        token = _READS.set(reads)
        try:
            key = (str(resolved), _manifest_sha256(resolved), bool(assets))
            result, parents = _release.validate_with_parents(root, assets=assets)
        finally:
            _READS.reset(token)
        if reads.complete and _unchanged(reads.files):
            with _LOCK:
                _CACHE[key] = (deepcopy(result), reads.files, deepcopy(parents))
                _CACHE.move_to_end(key)
                while len(_CACHE) > _LIMIT:
                    _CACHE.popitem(last=False)
        return result, parents


def task_inputs_cached(root, task_id, *args, **kwargs):
    """task_inputs() whose release (and robot revision parent) validation is cached."""
    if _release.task_inputs is not _GENUINE_TASK_INPUTS:
        return _release.task_inputs(root, task_id, *args, **kwargs)
    root = Path(root)
    manifest, parents = _validated(root, True)
    return _release.validated_task_inputs(root, manifest, task_id, *args, parents=parents, **kwargs)


def clear():
    with _LOCK:
        _CACHE.clear()


def worker_handoff(release, *, sources):
    """This process's cached validations of release, exported for its native worker.

    sources is the project inventory (relative path -> SHA-256) this process
    imported and checked before episode creation. Entries are exported as
    recorded; the worker trusts none of it until adopt_worker_handoff() re-proves
    it. Returns None when nothing cached matches the current manifest bytes.
    """
    resolved = Path(release).resolve()
    try:
        manifest = (resolved/'release_manifest.json').read_bytes()
    except OSError:
        return None
    digest = hashlib.sha256(manifest).hexdigest()
    with _LOCK:
        entries = sorted(((key[2], entry) for key, entry in _CACHE.items()
                          if key[:2] == (str(resolved), digest)), key=lambda row: row[0])
    validations = []
    for assets, (result, files, parents) in entries:
        try:
            row = {'assets': assets, **_transfer(result, manifest), 'parents': []}
            for parent, value in sorted(parents.items()):
                path = Path(parent)/'release_manifest.json'
                if str(path) in files:  # Only a parent read inside this proved snapshot.
                    data = path.read_bytes()
                    row['parents'].append({'root': parent, **_transfer(value, data),
                                           'manifest_sha256': hashlib.sha256(data).hexdigest()})
        except (OSError, TypeError, ValueError):
            continue  # Not transferable; the worker validates this flag itself.
        validations.append({**row, 'inputs': [[path, identity] for path, identity in files.items()]})
    if not validations or not sources:
        return None
    return {'schema': HANDOFF_SCHEMA, 'root': str(resolved), 'manifest_sha256': digest,
            'project_root': str(_PROJECT), 'sources_sha256': dict(sources), 'validations': validations}


def _transfer(result, manifest):
    """JSON form of a validate() result; the manifest itself travels as a flag."""
    text = json.dumps(result)
    same = text == json.dumps(json.loads(manifest))
    return {'result_is_manifest': same, 'result': None if same else result}


def _file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def adopt_worker_handoff(reference, release):
    """Results of the server's validations of release that this process re-proved.

    reference is the worker configuration's pin {path, sha256} of the handoff.
    A validation is reused only if the handoff bytes match that pin, it names
    this resolved release and project root, every pinned source keeps its
    SHA-256, the manifest bytes keep theirs and every recorded file/directory
    identity is unchanged. A robot revision's parent result is reused only with
    its re-proved child and when its manifest, inside that snapshot, keeps its
    SHA-256. Everything else is left to validate(). Returns
    ({assets: (result, {parent root: result})}, audit record).
    """
    start, results = time.monotonic(), {}
    record = {'schema': REUSE_SCHEMA, 'handoff': reference, 'reused': [], 'revalidate': {}}
    try:
        data = Path(reference['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference['sha256']:
            raise ValueError('Handoff bytes differ from the worker configuration pin')
        handoff, root = json.loads(data), Path(release).resolve()
        if (handoff.get('schema') != HANDOFF_SCHEMA or handoff.get('root') != str(root)
                or handoff.get('project_root') != str(_PROJECT) or not handoff.get('sources_sha256')):
            raise ValueError('Handoff names another schema, release or project, or no sources')
        for name, expected in handoff['sources_sha256'].items():
            if (Path(name).is_absolute() or '..' in Path(name).parts
                    or _file_sha256(_PROJECT/name) != expected):
                raise ValueError('Source differs from the one the parent imported: '+name)
        manifest = (root/'release_manifest.json').read_bytes()
        if hashlib.sha256(manifest).hexdigest() != handoff['manifest_sha256']:
            raise ValueError('Release manifest bytes changed')
        paths = sorted({path for row in handoff['validations'] for path, _ in row['inputs']})
        current = dict(zip(paths, _identities(paths)))
        for row in handoff['validations']:
            if type(row['assets']) is not bool:
                raise ValueError('Malformed handoff validation flag')
            changed = next((path for path, identity in row['inputs']
                            if current[path] != (None if identity is None else tuple(identity))), None)
            if changed is not None:
                record['revalidate'][str(row['assets'])] = 'Recorded identity changed: '+changed
                continue
            parents, recorded = {}, {path for path, _ in row['inputs']}
            for parent in row['parents']:
                path = Path(parent['root'])/'release_manifest.json'
                content = path.read_bytes() if str(path) in recorded else None
                if content is None or hashlib.sha256(content).hexdigest() != parent['manifest_sha256']:
                    record.setdefault('revalidate_parents', {})[parent['root']] = 'Manifest unproved or changed'
                    continue
                parents[parent['root']] = json.loads(content) if parent['result_is_manifest'] else parent['result']
            results[row['assets']] = (json.loads(manifest) if row['result_is_manifest'] else row['result'], parents)
            record['reused'].append(row['assets'])
            record.setdefault('reused_parents', {})[str(row['assets'])] = sorted(parents)
        record.update(manifest_sha256=handoff['manifest_sha256'], inputs_checked=len(paths),
                      sources_checked=len(handoff['sources_sha256']))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        results, record['reused'] = {}, []
        record['rejected'] = type(error).__name__+': '+str(error)
    record['seconds'] = round(time.monotonic()-start, 3)
    return results, record
