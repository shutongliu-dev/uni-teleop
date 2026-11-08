"""Explicit, process-local reuse between complete input byte checks.

Default loaders always hash real bytes. Only diagnostic callers opt into an
epoch. Metadata is an intra-epoch change detector, never the final evidence:
entry starts empty, every successful exit rereads every observed file, and a
native parent must create its own fresh epoch after the child has exited.

This is a read-only validation boundary, not a publication transaction. Do not
wrap asset authors or staging-directory renames: every hashed path must remain
at the same identity through exit. New diagnostic output may be hashed after
its final write, but cannot then be rewritten, moved or deleted in the epoch.
"""
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


_ACTIVE = ContextVar('isaac_input_validation_epoch', default=None)
_FULL_BYTES = ContextVar('isaac_full_byte_validation', default=None)
VALIDATION_MODES = ('strict_epoch_v1', 'full_bytes_v1', 'full_bytes_once_v1')
_SUMMARY_SCHEMAS = {'strict_epoch_v1': 'operation_validation_epoch.v1',
                    'full_bytes_v1': 'operation_full_byte_validation.v1',
                    'full_bytes_once_v1': 'operation_full_byte_once_validation.v1'}
_STAT_FIELDS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _identity(path):
    return str(path.resolve(strict=True)), _signature(path.stat())


def _identity_detail(value):
    return {'resolved_path': value[0], 'stat': dict(zip(_STAT_FIELDS, value[1]))}


def _failure(message, path, **details):
    # Explain the values already used by the rejecting comparison. No retry,
    # identity fallback or changed admission predicate is introduced.
    return ValueError(message+str(path)+' | '+json.dumps(details, sort_keys=True))


def _diagnostic_identity(path):
    """Best-effort detail only after the original fstat check already failed."""
    try:
        return _identity_detail(_identity(path))
    except (OSError, RuntimeError) as error:
        return {'unavailable': type(error).__name__, 'message': str(error)}


def _read(path):
    before = _identity(path)
    with path.open('rb') as stream:
        opened = _signature(os.fstat(stream.fileno()))
        if opened != before[1]:
            raise _failure('Validation input replaced while opening: ', path,
                before=_identity_detail(before), opened_fstat=dict(zip(_STAT_FIELDS, opened)),
                diagnostic_final_identity=_diagnostic_identity(path))
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = _signature(os.fstat(stream.fileno()))
    if after != opened:
        raise _failure('Validation input changed during byte verification: ', path,
            before=_identity_detail(before), opened_fstat=dict(zip(_STAT_FIELDS, opened)),
            after_fstat=dict(zip(_STAT_FIELDS, after)), diagnostic_final_identity=_diagnostic_identity(path))
    current = _identity(path)
    if current != before:
        raise _failure('Validation input changed during byte verification: ', path,
            before=_identity_detail(before), opened_fstat=dict(zip(_STAT_FIELDS, opened)),
            after_fstat=dict(zip(_STAT_FIELDS, after)), after=_identity_detail(current))
    return before, digest


def file_sha256(path):
    epoch = _ACTIVE.get()
    if epoch is not None:
        return epoch.digest(path)
    full = _FULL_BYTES.get()
    if full is not None:
        return full.digest(path)
    # Preserve the existing default: no process-wide or persistent cache.
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def memo_json(key, pins, build):
    """Reuse a deterministic JSON derivation only after its exact pinned reads.

    This does not cache a release validator. Its caller still executes all
    schema, source, asset, directory and path checks on every invocation.
    """
    epoch = _ACTIVE.get()
    full = _FULL_BYTES.get()
    if full is not None:
        return full.memo_json(key, pins, build)
    if epoch is None:
        return build()
    return epoch.memo_json(key, pins, build)


def _clear_semantic_cache():
    module = sys.modules.get('scenes.workrooms.laundry_room.randomization.validation_cache')
    if module is not None:
        module.clear()


def semantic_reuse_enabled():
    """Full-byte validation always recomputes every semantic derivation."""
    return _FULL_BYTES.get() is None


def checked_validation_mode(mode):
    if mode not in VALIDATION_MODES:
        raise ValueError('Unknown explicit input validation mode')
    return mode


def checked_validation_summary(summary, mode, *, closed_required=False, checkpoint_required=False):
    """Bind a declared mode to its recorded scope, retaining legacy strict inputs."""
    mode = checked_validation_mode(mode)
    if summary is None and mode == 'strict_epoch_v1':
        return  # Legacy compatibility only; absence does not prove an epoch ran.
    schema = _SUMMARY_SCHEMAS[mode]
    if (not isinstance(summary, dict) or summary.get('schema') != schema
            or summary.get('mode', 'strict_epoch_v1') != mode):
        raise ValueError('Validation scope summary differs from declared mode')
    if mode == 'full_bytes_v1' and (
            any(type(summary.get(key)) is not int or summary[key] != 0
                for key in ('hash_reuses', 'derivation_reuses'))
            or summary.get('metadata_hash_reuse') is not False
            or summary.get('semantic_reuse') is not False
            or summary.get('timestamp_stability_claimed') is not False):
        raise ValueError('Full-byte validation scope must declare zero byte/semantic reuse')
    if mode == 'full_bytes_once_v1' and (
            type(summary.get('hash_reuses')) is not int or type(summary.get('derivation_reuses')) is not int
            or summary.get('first_use_full_bytes') is not True or summary.get('identity_checked_reuse') is not True
            or summary.get('final_reread') is not False
            or summary.get('derivation_reuse_scope') != 'release_validation_per_resolved_root_and_manifest_digest'
            or summary.get('timestamp_stability_claimed') is not False):
        raise ValueError('Once-hashed full-byte scope must declare its first-use hashing and identity-checked reuse')
    if closed_required and summary.get('closed_verified') is not True:
        raise ValueError('Validation scope did not complete its final byte check')
    if closed_required or checkpoint_required:
        count = summary.get('full_byte_checkpoints')
        if type(count) is not int or count < 1:
            raise ValueError('Validation scope is missing a complete byte checkpoint')


def validation_mode_provenance(record, compatibility_mode, *, mode_key='validation_mode'):
    """Keep legacy admission defaults separate from evidence of an actual scope.

    Missing summaries may predate epoch validation entirely. Only a recorded,
    schema-checked scope supports an actual strict/full mode claim; historical
    source-code audits must be recorded separately instead of inferred here.
    """
    mode = checked_validation_mode(compatibility_mode)
    explicit = record.get(mode_key)
    if mode_key in record and explicit != mode:
        raise ValueError('Explicit validation mode differs from compatibility mode')
    summary = record.get('validation_epoch')
    checked_validation_summary(summary, mode)
    return {'schema': 'validation_mode_provenance.v1', 'compatibility_mode': mode,
            'explicit_mode': explicit, 'compatibility_default_applied': mode_key not in record,
            'recorded_scope_schema': None if summary is None else summary['schema'],
            'actual_mode': 'unrecorded_legacy' if summary is None else mode,
            'basis': 'no_recorded_scope' if summary is None else 'recorded_validation_scope'}


def validation_context(label, mode='strict_epoch_v1'):
    mode = checked_validation_mode(mode)
    if mode == 'strict_epoch_v1':
        return ValidationEpoch(label)
    return FullByteOnceValidation(label) if mode == 'full_bytes_once_v1' else FullByteValidation(label)


class ValidationEpoch:
    def __init__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError('A diagnostic validation epoch needs a label')
        self.label = label
        self._files = {}
        self._json = {}
        self._token = None
        self._used = False
        self._owner = (os.getpid(), threading.get_ident())
        self._counts = dict(fresh_hash_calls=0, fresh_hash_bytes=0, hash_reuses=0,
                            forced_hash_calls=0, forced_hash_bytes=0,
                            derivation_builds=0, derivation_reuses=0,
                            full_byte_checkpoints=0)
        self.closed_verified = False

    def _check_owner(self):
        if self._token is None or self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('Validation epoch is inactive or belongs to another process/thread')

    def __enter__(self):
        if self._used or _ACTIVE.get() is not None or _FULL_BYTES.get() is not None:
            raise RuntimeError('Validation epochs cannot be nested or reused')
        self._used = True
        _clear_semantic_cache()
        self._token = _ACTIVE.set(self)
        return self

    def digest(self, filename):
        self._check_owner()
        path = Path(filename).absolute()
        name = str(path)
        previous = self._files.get(name)
        if previous is not None:
            current = _identity(path)
            if current != previous[0]:
                raise _failure('Validation input identity changed within epoch: ', name,
                    before=_identity_detail(previous[0]), after=_identity_detail(current))
            self._counts['hash_reuses'] += 1
            return previous[1]
        value = _read(path)
        self._files[name] = value
        self._counts['fresh_hash_calls'] += 1
        self._counts['fresh_hash_bytes'] += value[0][1][2]
        return value[1]

    def memo_json(self, key, pins, build):
        self._check_owner()
        # Keep expected hashes in the key. Conflicting expectations still fail
        # against actual bytes, even if the file itself was already hashed.
        for name, expected in pins.items():
            if self.digest(name) != expected:
                raise ValueError('Derivation input differs from pinned bytes: '+str(name))
        encoded_key = json.dumps([key, sorted(pins.items())], separators=(',', ':'),
                                 sort_keys=True, allow_nan=False)
        if encoded_key not in self._json:
            value = build()
            # Recheck all input identities after computation before publishing
            # a proof. JSON bytes preserve insertion order and prevent mutation
            # of a returned dictionary from changing subsequent derivations.
            for name, expected in pins.items():
                if self.digest(name) != expected:
                    raise ValueError('Derivation input changed during computation: '+str(name))
            self._json[encoded_key] = json.dumps(value, separators=(',', ':'), allow_nan=False)
            self._counts['derivation_builds'] += 1
        else:
            self._counts['derivation_reuses'] += 1
        return json.loads(self._json[encoded_key])

    def verify_all(self):
        """Unconditionally reread bytes; never consult metadata as hash proof."""
        self._check_owner()
        for name, previous in self._files.items():
            value = _read(Path(name))
            self._counts['forced_hash_calls'] += 1
            self._counts['forced_hash_bytes'] += value[0][1][2]
            if value != previous:
                raise _failure('Validation epoch final bytes or identity changed: ', name,
                    before=_identity_detail(previous[0]), after=_identity_detail(value[0]),
                    before_sha256=previous[1], after_sha256=value[1])
        self._counts['full_byte_checkpoints'] += 1

    def summary(self):
        return {'schema': 'operation_validation_epoch.v1', 'label': self.label,
                **self._counts, 'unique_files': self._counts['fresh_hash_calls'],
                'closed_verified': self.closed_verified,
                'scope': 'Operation-local reuse; fresh byte checks at first use and full checkpoints. No native outcome or cross-process qualification.'}

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self.verify_all()
                self.closed_verified = True
        finally:
            _ACTIVE.reset(self._token)
            self._token = None
            self._files.clear()
            self._json.clear()
            _clear_semantic_cache()


def _content_identity(path):
    # Timestamp views are deliberately not a content proof in this explicit
    # uncached mode. Every call hashes bytes; path/object replacement still fails.
    def stable(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mode
    return str(path.resolve(strict=True)), stable(path.stat())


def _content_read(path):
    before = _content_identity(path)
    with path.open('rb') as stream:
        value = os.fstat(stream.fileno())
        opened = value.st_dev, value.st_ino, value.st_size, value.st_mode
        if opened != before[1]:
            raise _failure('Full-byte input replaced while opening: ', path,
                           before=before, opened=opened)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        value = os.fstat(stream.fileno())
        after = value.st_dev, value.st_ino, value.st_size, value.st_mode
    current = _content_identity(path)
    if opened != after or current != before:
        raise _failure('Full-byte input path/object changed while reading: ', path,
                       before=before, opened=opened, after=after, current=current)
    return before, digest


class FullByteValidation:
    """Explicit uncached checks; never relax or retry a strict epoch rejection.

    Starts with no proofs, hashes on every call, recomputes semantic results and
    force-reads the entire observed closure on clean exit. All observations of
    a logical file must retain the same bytes and resolved object. This mode
    makes no assertion about mtime/ctime stability and does not trust them.
    """
    def __init__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError('A full-byte validation scope needs a label')
        self.label = label
        self._files = {}
        self._used = False
        self._token = None
        self._failed = False
        self._owner = (os.getpid(), threading.get_ident())
        self.closed_verified = False
        self._counts = dict(fresh_hash_calls=0, fresh_hash_bytes=0, hash_reuses=0,
                            forced_hash_calls=0, forced_hash_bytes=0,
                            derivation_builds=0, derivation_reuses=0,
                            full_byte_checkpoints=0, unique_files=0)

    def _check_owner(self):
        if self._token is None or self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('Full-byte validation is inactive or belongs to another process/thread')

    def __enter__(self):
        if self._used or _ACTIVE.get() is not None or _FULL_BYTES.get() is not None:
            raise RuntimeError('Validation scopes cannot be nested or reused')
        self._used = True
        _clear_semantic_cache()
        self._token = _FULL_BYTES.set(self)
        return self

    def _observe(self, path, *, forced=False):
        self._check_owner()
        if self._failed:
            raise ValueError('Full-byte validation scope already rejected an observation')
        path = Path(path).absolute()
        name = str(path)
        try:
            value = _content_read(path)
        except Exception:
            self._failed = True
            raise
        prefix = 'forced' if forced else 'fresh'
        self._counts[prefix+'_hash_calls'] += 1
        self._counts[prefix+'_hash_bytes'] += value[0][1][2]
        previous = self._files.get(name)
        if previous is not None and value != previous:
            self._failed = True
            raise _failure('Full-byte validation input bytes or object changed: ', name,
                           before=previous[0], after=value[0],
                           before_sha256=previous[1], after_sha256=value[1])
        if previous is None:
            self._files[name] = value
            self._counts['unique_files'] += 1
        return value[1]

    def digest(self, path):
        return self._observe(path)

    def memo_json(self, key, pins, build):
        self._check_owner()
        for name, expected in pins.items():
            if self.digest(name) != expected:
                self._failed = True
                raise ValueError('Full-byte derivation input differs from pinned bytes')
        result = build()
        self._counts['derivation_builds'] += 1
        for name, expected in pins.items():
            if self.digest(name) != expected:
                self._failed = True
                raise ValueError('Full-byte derivation input changed during computation')
        return result

    def verify_all(self):
        self._check_owner()
        if self._failed:
            raise ValueError('Full-byte validation scope already rejected an observation')
        for name in tuple(self._files):
            self._observe(name, forced=True)
        self._counts['full_byte_checkpoints'] += 1

    def summary(self):
        return {'schema': 'operation_full_byte_validation.v1', 'mode': 'full_bytes_v1',
                'label': self.label, **self._counts, 'closed_verified': self.closed_verified,
                'metadata_hash_reuse': False, 'semantic_reuse': False,
                'timestamp_stability_claimed': False,
                'scope': 'Every call hashes real bytes and every semantic derivation recomputes. Complete observed closure reread at checkpoints/exit; resolved path, file object and bytes remain bound. No timestamp stability or native outcome claim.'}

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self.verify_all()
                self.closed_verified = True
        finally:
            _FULL_BYTES.reset(self._token)
            self._token = None
            self._files.clear()
            _clear_semantic_cache()


class FullByteOnceValidation(FullByteValidation):
    """Hash every file's complete bytes once per scope, then reuse the digest.

    Chosen by the operator for a single-writer workspace: the first use of a
    path hashes all of its bytes; later uses in the same scope reuse that
    digest. Each checkpoint and the exit re-check every observed file's resolved
    path and object (device, inode, size, mode), failing on any replacement,
    but never read the bytes a second time. Semantic derivations are still
    recomputed on every call. Timestamps are not trusted or claimed.
    """
    def memo(self, key, build):
        self._check_owner()
        if self._failed:
            raise ValueError('Full-byte validation scope already rejected an observation')
        cache = self.__dict__.setdefault('_memo', {})
        if key in cache:
            self._counts['derivation_reuses'] += 1
            return deepcopy(cache[key])
        value = build()
        self._counts['derivation_builds'] += 1
        cache[key] = deepcopy(value)
        return value

    def __exit__(self, kind, error, traceback):
        try:
            return super().__exit__(kind, error, traceback)
        finally:
            self.__dict__.pop('_memo', None)

    def _observe(self, path, *, forced=False):
        self._check_owner()
        if self._failed:
            raise ValueError('Full-byte validation scope already rejected an observation')
        path = Path(path).absolute()
        name = str(path)
        previous = self._files.get(name)
        if previous is None:
            try:
                value = _content_read(path)
            except Exception:
                self._failed = True
                raise
            self._counts['fresh_hash_calls'] += 1
            self._counts['fresh_hash_bytes'] += value[0][1][2]
            self._files[name] = value
            self._counts['unique_files'] += 1
            return value[1]
        self._counts['hash_reuses'] += 1
        return previous[1]

    def verify_all(self):
        self._check_owner()
        if self._failed:
            raise ValueError('Full-byte validation scope already rejected an observation')
        for name, previous in tuple(self._files.items()):
            try:
                current = _content_identity(Path(name))
            except Exception:
                self._failed = True
                raise
            if current != previous[0]:
                self._failed = True
                raise _failure('Once-hashed validation input path or object changed: ', name,
                               before=previous[0], after=current)
            self._counts['forced_identity_checks'] = self._counts.get('forced_identity_checks', 0) + 1
        self._counts['full_byte_checkpoints'] += 1

    def summary(self):
        return {'schema': 'operation_full_byte_once_validation.v1', 'mode': 'full_bytes_once_v1',
                'label': self.label, **self._counts, 'closed_verified': self.closed_verified,
                'first_use_full_bytes': True, 'identity_checked_reuse': True, 'final_reread': False,
                'identity_checks': 'checkpoints_and_exit',
                'metadata_hash_reuse': True, 'semantic_reuse': True,
                'derivation_reuse_scope': 'release_validation_per_resolved_root_and_manifest_digest',
                'timestamp_stability_claimed': False,
                'scope': ('Each observed file is hashed in full once per scope and its digest reused; every '
                          'checkpoint and the exit re-check each observed resolved path and file object (device, '
                          'inode, size, mode) without rereading bytes. A release is fully validated once per scope '
                          '(resolved root and manifest digest) and that result reused; other derivations recompute. '
                          'Operator-selected for a single-writer workspace; no timestamp stability or native '
                          'outcome claim.')}


def active_once_scope():
    """The active operator-selected once-hashed scope, if any (never other modes)."""
    full = _FULL_BYTES.get()
    return full if isinstance(full, FullByteOnceValidation) else None


def active_validation_scope():
    """The active explicit scope of any mode; process-wide reuse must defer to it."""
    epoch = _ACTIVE.get()
    return epoch if epoch is not None else _FULL_BYTES.get()
