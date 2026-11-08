import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.isaacsim import validation_epoch as validation


def test_full_bytes_rehashes_every_call_with_different_timestamps(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'fixed bytes')
    original = os.fstat
    calls = []
    real_hash = hashlib.file_digest
    def changing_fstat(fd):
        stat = original(fd)
        return SimpleNamespace(**{key: getattr(stat, key) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mode')},
            st_mtime_ns=stat.st_mtime_ns+len(calls)*321,
            st_ctime_ns=stat.st_ctime_ns+len(calls)*519715)
    def read(stream, algorithm):
        calls.append(stream.name)
        return real_hash(stream, algorithm)
    monkeypatch.setattr(os, 'fstat', changing_fstat)
    monkeypatch.setattr(hashlib, 'file_digest', read)
    with validation.validation_context('actual bytes', 'full_bytes_v1') as scope:
        expected = validation.file_sha256(path)
        for index in range(3):
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+index+1))
            assert validation.file_sha256(path) == expected
    assert len(calls) == 5
    summary = scope.summary()
    assert summary['fresh_hash_calls'] == 4 and summary['forced_hash_calls'] == 1
    assert summary['unique_files'] == 1 and summary['hash_reuses'] == 0
    assert summary['closed_verified'] and summary['mode'] == 'full_bytes_v1'


def test_changed_bytes_with_same_size_inode_and_restored_mtime_rejected(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'first')
    stat = path.stat()
    with pytest.raises(ValueError, match='already rejected'):
        with validation.FullByteValidation('poisoned') as scope:
            validation.file_sha256(path)
            path.write_bytes(b'other')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert path.stat().st_ino == stat.st_ino and path.stat().st_size == stat.st_size
            with pytest.raises(ValueError, match='bytes or object changed'):
                validation.file_sha256(path)
            path.write_bytes(b'first')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not scope.closed_verified


def test_unobserved_final_mutation_fails_fresh_exit(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'first')
    with pytest.raises(ValueError, match='bytes or object changed'):
        with validation.FullByteValidation('exit') as scope:
            validation.file_sha256(path)
            path.write_bytes(b'other')
    assert not scope.closed_verified


@pytest.mark.parametrize('mutation', ['replace', 'symlink', 'mode'])
def test_same_byte_object_or_path_replacement_is_rejected(tmp_path, mutation):
    first, second, link = (tmp_path/name for name in ('first', 'second', 'link'))
    first.write_bytes(b'same'); second.write_bytes(b'same'); link.symlink_to(first)
    path = link if mutation == 'symlink' else first
    with pytest.raises(ValueError, match='bytes or object changed'):
        with validation.FullByteValidation('identity'):
            validation.file_sha256(path)
            if mutation == 'replace':
                second.replace(first)
            elif mutation == 'symlink':
                link.unlink(); link.symlink_to(second)
            else:
                first.chmod(0o444)
            validation.file_sha256(path)


def test_full_bytes_disables_derivation_and_laundry_proof_reuse(tmp_path):
    from scenes.workrooms.laundry_room.randomization import validation_cache
    path = tmp_path/'input'; path.write_bytes(b'value')
    pins = {str(path): hashlib.sha256(b'value').hexdigest()}
    count = []
    def build():
        count.append(True)
        return {'computed': len(count)}
    with validation.FullByteValidation('derivations') as scope:
        assert validation.memo_json('same', pins, build) == {'computed': 1}
        assert validation.memo_json('same', pins, build) == {'computed': 2}
        validation_cache.store('test', object())
        assert validation_cache.get('test') is None
        assert not validation.semantic_reuse_enabled()
    assert validation.semantic_reuse_enabled()
    assert scope.summary()['derivation_builds'] == 2
    assert scope.summary()['derivation_reuses'] == 0


def test_caught_initial_pin_mismatch_poison_scope(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'wrong pinned content')
    expected = hashlib.sha256(b'expected content').hexdigest()
    with pytest.raises(ValueError, match='already rejected'):
        with validation.FullByteValidation('bad initial pin') as scope:
            with pytest.raises(ValueError, match='differs from pinned bytes'):
                validation.memo_json('bad', {str(path): expected}, lambda: {'unsafe': True})
    assert not scope.closed_verified


def test_full_bytes_and_strict_epoch_cannot_nest_or_reuse():
    for outer, inner in ((validation.FullByteValidation, validation.ValidationEpoch),
                         (validation.ValidationEpoch, validation.FullByteValidation),
                         (validation.FullByteValidation, validation.FullByteValidation)):
        with outer('outer'):
            with pytest.raises(RuntimeError, match='nested'):
                with inner('inner'):
                    pass
    scope = validation.FullByteValidation('single use')
    with scope:
        pass
    with pytest.raises(RuntimeError, match='reused'):
        with scope:
            pass


def test_default_strict_epoch_still_rejects_timestamp_change(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'unchanged')
    with pytest.raises(ValueError, match='identity changed'):
        with validation.validation_context('default'):
            validation.file_sha256(path)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
            validation.file_sha256(path)
    with pytest.raises(ValueError, match='Unknown'):
        validation.validation_context('bad', 'full_bytes')


def test_once_mode_hashes_each_file_once_and_reuses_only_the_same_object(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'fixed bytes')
    other = tmp_path/'other'; other.write_bytes(b'other bytes')
    reads = []
    real_hash = hashlib.file_digest
    def read(stream, algorithm):
        reads.append(stream.name)
        return real_hash(stream, algorithm)
    monkeypatch.setattr(hashlib, 'file_digest', read)
    with validation.validation_context('once', 'full_bytes_once_v1') as scope:
        expected = hashlib.sha256(b'fixed bytes').hexdigest()
        assert [validation.file_sha256(path) for _ in range(5)] == [expected]*5
        validation.file_sha256(other)
        scope.verify_all()
    summary = scope.summary()
    assert reads == [str(path), str(other)]  # one full read per file, none at checkpoint or exit
    assert summary['fresh_hash_calls'] == 2 and summary['hash_reuses'] == 4 and summary['unique_files'] == 2
    # Explicit checkpoint plus exit: identity checks of both files, no second read.
    assert summary['full_byte_checkpoints'] == 2 and summary['forced_identity_checks'] == 4
    assert summary['closed_verified'] is True and summary['final_reread'] is False
    validation.checked_validation_summary(summary, 'full_bytes_once_v1', closed_required=True)
    with pytest.raises(ValueError):
        validation.checked_validation_summary(summary, 'full_bytes_v1', closed_required=True)


@pytest.mark.parametrize('mutation', ['replace', 'resize'])
def test_once_mode_rejects_a_replaced_or_resized_object(tmp_path, mutation):
    path = tmp_path/'input'; path.write_bytes(b'fixed bytes')
    with pytest.raises(ValueError, match='Once-hashed validation input path or object changed'):
        with validation.validation_context('once', 'full_bytes_once_v1'):
            validation.file_sha256(path)
            if mutation == 'replace':
                staged = tmp_path/'staged'; staged.write_bytes(b'fixed bytes'); os.replace(staged, path)
            else:
                path.write_bytes(b'fixed bytes and more')
            validation.file_sha256(path)


def test_once_summary_must_declare_its_reuse_policy():
    with validation.validation_context('once', 'full_bytes_once_v1') as scope:
        pass
    summary = scope.summary()
    validation.checked_validation_summary(summary, 'full_bytes_once_v1', closed_required=True)
    for key, value in (('first_use_full_bytes', False), ('final_reread', True), ('derivation_reuse_scope', 'any')):
        with pytest.raises(ValueError):
            validation.checked_validation_summary({**summary, key: value}, 'full_bytes_once_v1')
