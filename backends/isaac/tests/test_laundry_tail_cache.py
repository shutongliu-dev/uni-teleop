"""Small compressed arrays exercise reuse, bounded memory and fresh SHA checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import zipfile

import numpy as np
import pytest

from bench.paths import load_json
from scenes.workrooms.laundry_room import assets, settled_reset
from test_laundry_settled_reset import fixture as metadata_fixture


@pytest.fixture(autouse=True)
def empty_cache():
    def clear():
        with settled_reset._TAIL_CACHE_LOCK:
            settled_reset._TAIL_CACHE.clear()
            settled_reset._TAIL_CACHE_SIZE = 0
    clear()
    yield
    clear()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    values = np.arange(4096, dtype=np.float64).reshape(1024, 4)
    path = tmp_path / 'small_native.npz'
    np.savez_compressed(path, q=values, r=values + 1, s=values + 2)
    counters = {'decoded_bytes': 0, 'read_calls': 0}
    original = zipfile.ZipExtFile._read1
    def counted(stream, count):
        data = original(stream, count)
        counters['decoded_bytes'] += len(data)
        counters['read_calls'] += 1
        return data
    monkeypatch.setattr(zipfile.ZipExtFile, '_read1', counted)
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('must not load the full archive or array'))
    with zipfile.ZipFile(path) as files:
        member_bytes = files.getinfo('q.npy').file_size
    return path, values, counters, member_bytes


def checked_tail(path, name='q', count=3, *, expected=None, total_rows=None):
    expected = settled_reset.digest(path) if expected is None else expected
    verified = settled_reset.checked(path, expected)
    return settled_reset.tail_array(path, name, count, total_rows=total_rows,
                                    verified_archive_sha256=verified)


def test_default_tail_remains_uncached_and_reads_each_member_once(archive):
    path, values, counts, size = archive
    for _ in range(2):
        np.testing.assert_array_equal(settled_reset.tail_array(path, 'q', 3), values[-3:])
    assert counts['decoded_bytes'] == 2 * size
    assert not settled_reset._TAIL_CACHE


def test_verified_slice_reuses_decompression_and_returns_independent_copies(archive):
    path, values, counts, size = archive
    first = checked_tail(path)
    first[:] = -123.
    for _ in range(3):
        second = checked_tail(path)
        np.testing.assert_array_equal(second, values[-3:])
        assert second.flags.writeable
    assert counts['decoded_bytes'] == size
    assert settled_reset._TAIL_CACHE_SIZE == values[-3:].nbytes


def test_cache_binds_actual_content_hash_and_requested_slice(archive):
    path, values, counts, size = archive
    old_sha = settled_reset.digest(path)
    checked_tail(path, expected=old_sha, total_rows=1024)
    with pytest.raises(ValueError, match='frame count differs'):
        checked_tail(path, expected=old_sha, total_rows=1025)
    np.testing.assert_array_equal(checked_tail(path, count=5), values[-5:])
    np.savez_compressed(path, q=values + 100.)
    with pytest.raises(ValueError, match='Frozen input changed'):
        checked_tail(path, expected=old_sha, total_rows=1024)
    np.testing.assert_array_equal(checked_tail(path, total_rows=1024), (values + 100.)[-3:])
    assert counts['decoded_bytes'] >= 3 * size


def test_oversized_terminal_slice_is_not_cached(archive, monkeypatch):
    path, values, counts, size = archive
    monkeypatch.setattr(settled_reset, '_TAIL_CACHE_BYTES', 32)
    for _ in range(2):
        np.testing.assert_array_equal(checked_tail(path), values[-3:])
    assert counts['decoded_bytes'] == 2 * size
    assert settled_reset._TAIL_CACHE_SIZE == 0 and not settled_reset._TAIL_CACHE


def test_lru_eviction_obeys_total_bytes_and_entry_count(archive, monkeypatch):
    path, values, counts, size = archive
    monkeypatch.setattr(settled_reset, '_TAIL_CACHE_BYTES', 128)
    monkeypatch.setattr(settled_reset, '_TAIL_CACHE_ENTRIES', 2)
    checked_tail(path, 'q', 2)
    checked_tail(path, 'r', 2)
    checked_tail(path, 'q', 2)  # q becomes most recent.
    checked_tail(path, 's', 2)
    assert counts['decoded_bytes'] == 3 * size
    assert settled_reset._TAIL_CACHE_SIZE == 128
    assert {key[2] for key in settled_reset._TAIL_CACHE} == {'q', 's'}
    checked_tail(path, 'r', 2)  # The evicted entry must be decoded again.
    assert counts['decoded_bytes'] == 4 * size
    assert settled_reset._TAIL_CACHE_SIZE == 128 and len(settled_reset._TAIL_CACHE) == 2


def test_parallel_cache_users_do_not_share_mutable_arrays_or_double_count_entries(archive):
    path, values, _, _ = archive
    sha = settled_reset.digest(path)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: checked_tail(path, expected=sha), range(16)))
    for row in results:
        np.testing.assert_array_equal(row, values[-3:])
    results[0][:] = -1.
    for row in results[1:]:
        np.testing.assert_array_equal(row, values[-3:])
    assert len(settled_reset._TAIL_CACHE) == 1
    assert settled_reset._TAIL_CACHE_SIZE == values[-3:].nbytes


def test_metadata_revalidates_archive_sha_before_using_cached_geometry(tmp_path, monkeypatch):
    native_path, audit_path, profile, _ = metadata_fixture(tmp_path, monkeypatch)
    metadata = settled_reset.metadata_from_native_fold(native_path, audit_path, physics_options=profile)
    assert settled_reset._TAIL_CACHE
    assert assets.validate_metadata(metadata) == metadata
    archive_path = load_json(native_path)['state_archive']['path']
    with open(archive_path, 'ab') as stream:
        stream.write(b'tampered after cache population')
    with pytest.raises(ValueError, match='Frozen input changed'):
        assets.validate_metadata(metadata)
