"""Operation-local reuse must not turn timestamps or earlier runs into proof."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from bench.isaacsim import validation_epoch as epochs
from bench.isaacsim.release import checked, digest
from bench.paths import DATA_ROOT
from scenes.workrooms.initial_randomization.common import checked_file


def test_default_reads_each_time_and_epoch_force_reads_at_exit(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'original')
    calls = []
    original = hashlib.file_digest
    def count(stream, *args, **kwargs):
        calls.append(stream.name)
        return original(stream, *args, **kwargs)
    monkeypatch.setattr(hashlib, 'file_digest', count)
    expected = digest(path); assert digest(path) == expected and len(calls) == 2
    with epochs.ValidationEpoch('test') as epoch:
        assert digest(path) == digest(path) == expected
        assert len(calls) == 3
        assert checked_file(path, expected) == path
        epoch.verify_all()
        assert len(calls) == 4
    assert len(calls) == 5 and epoch.summary()['closed_verified']
    assert epoch.summary()['full_byte_checkpoints'] == 2
    assert digest(path) == expected and len(calls) == 6
    with epochs.ValidationEpoch('new process-like boundary'):
        assert digest(path) == expected
    assert len(calls) == 8


@pytest.mark.parametrize('mutation', ['bytes_mtime_restored', 'replace_same_bytes', 'delete', 'symlink'])
def test_active_epoch_rejects_changed_content_identity_or_path(tmp_path, mutation):
    path = tmp_path/'input'; path.write_bytes(b'original')
    other = tmp_path/'other'; other.write_bytes(b'original')
    with pytest.raises((ValueError, FileNotFoundError)):
        with epochs.ValidationEpoch(mutation):
            expected = digest(path); old = path.stat()
            if mutation == 'bytes_mtime_restored':
                path.write_bytes(b'modified'); os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
            elif mutation == 'replace_same_bytes':
                other.replace(path)
            elif mutation == 'delete':
                path.unlink()
            else:
                path.unlink(); path.symlink_to(other)
            checked(path, expected)
    # Failure never leaves an active epoch or admission cache behind.
    if path.exists():
        assert digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_symlink_retarget_with_identical_bytes_is_not_same_input(tmp_path):
    a, b, link = (tmp_path/n for n in ('a', 'b', 'link'))
    a.write_bytes(b'x'); b.write_bytes(b'x'); link.symlink_to(a)
    with pytest.raises(ValueError, match='identity changed') as rejected:
        with epochs.ValidationEpoch('alias'):
            digest(link); link.unlink(); link.symlink_to(b); digest(link)
    detail = json.loads(str(rejected.value).split(' | ', 1)[1])
    assert detail['before']['resolved_path'] == str(a)
    assert detail['after']['resolved_path'] == str(b)


def test_final_check_is_fresh_even_when_no_reuse_observes_mutation(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'original')
    with pytest.raises(ValueError, match='final bytes or identity changed'):
        with epochs.ValidationEpoch('exit mutation'):
            digest(path)
            path.write_bytes(b'modified')


def test_forced_boundary_detects_bytes_even_if_metadata_cannot_distinguish_them(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'original')
    signature = epochs._signature(path.stat())
    # Models coarse/unreliable filesystem timestamps: metadata reuse alone is
    # explicitly insufficient, and must never serve as the final byte proof.
    monkeypatch.setattr(epochs, '_signature', lambda _: signature)
    with pytest.raises(ValueError, match='final bytes or identity changed'):
        with epochs.ValidationEpoch('metadata is not evidence'):
            expected = digest(path)
            path.write_bytes(b'modified')
            assert digest(path) == expected


def test_hash_conflict_cannot_reuse_another_expected_pin(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'original')
    with epochs.ValidationEpoch('conflict'):
        checked_file(path, digest(path))
        with pytest.raises(ValueError, match='Frozen input differs'):
            checked_file(path, '0'*64)


def test_change_during_hash_fails_before_a_cache_entry_is_created(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'original')
    original = hashlib.file_digest
    def mutate(stream, *args, **kwargs):
        result = original(stream, *args, **kwargs)
        replacement = path.with_name('replacement')
        replacement.write_bytes(b'modified')
        replacement.replace(path)
        return result
    monkeypatch.setattr(hashlib, 'file_digest', mutate)
    with pytest.raises(ValueError, match='during byte verification') as rejected:
        with epochs.ValidationEpoch('concurrent mutation') as epoch:
            digest(path)
    assert epoch.summary()['fresh_hash_calls'] == 0
    detail = json.loads(str(rejected.value).split(' | ', 1)[1])
    # Unlink may update the still-open inode's ctime first; that equally strict
    # fstat rejection records the replacement under diagnostic_final_identity.
    after = detail.get('after', detail.get('diagnostic_final_identity'))
    assert detail['before']['resolved_path'] == after['resolved_path'] == str(path)
    assert detail['before']['stat']['st_ino'] != after['stat']['st_ino']
    assert detail['before']['stat'] == detail['opened_fstat']


def test_reuse_rejection_records_exact_path_and_nanosecond_signatures(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'original')
    with pytest.raises(ValueError, match='identity changed within epoch') as rejected:
        with epochs.ValidationEpoch('diagnostic details') as epoch:
            digest(path)
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns+1234567))
            digest(path)
    detail = json.loads(str(rejected.value).split(' | ', 1)[1])
    assert detail['before']['resolved_path'] == detail['after']['resolved_path'] == str(path)
    assert detail['before']['stat']['st_mtime_ns'] == before.st_mtime_ns
    assert detail['after']['stat']['st_mtime_ns'] == before.st_mtime_ns+1234567
    assert set(detail['before']['stat']) == set(epochs._STAT_FIELDS)
    assert epoch.summary()['hash_reuses'] == 0 and not epoch.closed_verified


def test_final_byte_rejection_records_both_hashes_and_signatures(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'original')
    signature = epochs._signature(path.stat())
    monkeypatch.setattr(epochs, '_signature', lambda _: signature)
    with pytest.raises(ValueError, match='final bytes or identity changed') as rejected:
        with epochs.ValidationEpoch('final diagnostics') as epoch:
            digest(path); path.write_bytes(b'modified')
    detail = json.loads(str(rejected.value).split(' | ', 1)[1])
    assert detail['before'] == detail['after']
    assert detail['before_sha256'] == hashlib.sha256(b'original').hexdigest()
    assert detail['after_sha256'] == hashlib.sha256(b'modified').hexdigest()
    assert epoch.summary()['forced_hash_calls'] == 1 and not epoch.closed_verified


def test_open_rejection_records_stat_and_fstat_without_retry(tmp_path, monkeypatch):
    path = tmp_path/'input'; path.write_bytes(b'original')
    actual = epochs._identity(path); calls = []
    def replaced_identity(value):
        calls.append(value)
        changed = list(actual[1]); changed[1] += 1
        return actual[0], tuple(changed)
    monkeypatch.setattr(epochs, '_identity', replaced_identity)
    with pytest.raises(ValueError, match='replaced while opening') as rejected:
        with epochs.ValidationEpoch('open diagnostics') as epoch:
            digest(path)
    detail = json.loads(str(rejected.value).split(' | ', 1)[1])
    assert detail['before']['stat']['st_ino'] == actual[1][1]+1
    assert detail['opened_fstat']['st_ino'] == actual[1][1]
    assert detail['diagnostic_final_identity']['resolved_path'] == str(path)
    assert len(calls) == 2  # One attempted validation, one failure-only diagnostic.
    assert epoch.summary()['fresh_hash_calls'] == 0 and not epoch.closed_verified


def test_json_derivation_requires_pins_and_returns_unshared_values(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'original')
    pins = {str(path): digest(path)}; calls = []
    def build():
        calls.append(True); return [{'seed': i} for i in range(10)]
    with epochs.ValidationEpoch('documents') as epoch:
        first = epochs.memo_json(['task', 'revision'], pins, build)
        first[0]['seed'] = 999
        assert epochs.memo_json(['task', 'revision'], pins, build)[0]['seed'] == 0
        assert calls == [True]
        epochs.memo_json(['task', 'another revision'], pins, build)
        assert len(calls) == 2
    assert epoch.summary()['derivation_reuses'] == 1
    epochs.memo_json(['task', 'revision'], pins, build)
    assert len(calls) == 3  # Outside the explicit epoch there is no reuse.


def test_epoch_rejects_nesting_and_reentry():
    epoch = epochs.ValidationEpoch('once')
    with epoch:
        with pytest.raises(RuntimeError, match='nested or reused'):
            with epochs.ValidationEpoch('nested'):
                pass
    with pytest.raises(RuntimeError, match='nested or reused'):
        with epoch:
            pass


def test_real_ten_seed_robot_derivation_is_exact_with_reuse():
    from scenes.workrooms.robot_revision import release
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/pair_laundry02_first12/releases/laundry_room/towel_flattening/single_arm')
    if not root.is_dir():
        pytest.skip('Real immutable paired fixture unavailable')
    manifest = release.read(root/'release_manifest.json')
    parent = Path(manifest['base_release']['path'])
    original = release.read(parent/'release_manifest.json')
    revision = release.read(root/'robot_revision.json')
    expected = release._documents(parent, original, revision, None)
    with epochs.ValidationEpoch('real all ten seeds') as epoch:
        first = release._documents(parent, original, revision, None)
        second = release._documents(parent, original, revision, None)
        assert first == second == expected
        assert set(second[1]) == set(map(str, range(10)))
        first[1]['0']['case_sha256'] = 'bad'
        assert release._documents(parent, original, revision, None) == expected
    assert epoch.summary()['derivation_builds'] == 1
    assert epoch.summary()['derivation_reuses'] == 2
