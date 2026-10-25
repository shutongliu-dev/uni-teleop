import multiprocessing

import pytest

from agent.model_client.learning.journal import (Journal, JournalBusyError, JournalError, atomic_write_json, canonical_json, strict_json_loads)


def _try_writer(path, queue):
    try:
        Journal(path).append("child", {})
    except JournalBusyError:
        queue.put("busy")
    else:
        queue.put("acquired")



@pytest.mark.parametrize("value", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}', '{"x":-1e400}'])
def test_strict_json(value):
    with pytest.raises(JournalError):
        strict_json_loads(value)
    with pytest.raises(JournalError):
        canonical_json({"x": float("nan")})



def test_journal_checksums_sequence_deduplication_and_uncommitted_tail(tmp_path):
    journal = Journal(tmp_path / "journal")
    first = journal.append("start", {"x": 1}, event_id="one", transaction_id="txn")
    assert journal.append("start", {"x": 1}, event_id="one", transaction_id="txn") == first
    with pytest.raises(JournalError, match="reused"):
        journal.append("start", {"x": 2}, event_id="one")
    (journal.path / ".pending-incomplete").write_text('{"broken"')
    assert journal.read() == [first]
    second = journal.append("end", {}, event_id="two")
    assert second["previous_hash"] == first["event_hash"]
    atomic_write_json(journal.path / "00000000000000000002.json", {**second, "payload": {"tamper": True}})
    with pytest.raises(JournalError, match="checksum"):
        journal.read()
    (journal.path / "00000000000000000001.json").unlink()
    with pytest.raises(JournalError, match="sequence"):
        journal.read()



def test_single_host_writer_across_processes(tmp_path):
    journal = Journal(tmp_path / "journal")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with journal.transaction():
        journal.append("parent", {})  # same-owner reentrance is allowed
        process = context.Process(target=_try_writer, args=(journal.path, queue))
        process.start()
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert queue.get(timeout=2) == "busy"
    assert [event["kind"] for event in journal.read()] == ["parent"]



def _race_publish(source, destination, barrier, queue):
    import errno
    from agent.model_client.learning import journal

    def unsupported(*args):
        raise OSError(errno.EINVAL, 'forced unsupported RENAME_NOREPLACE')

    journal._renameat2_noreplace = unsupported
    barrier.wait(timeout=15)
    try:
        journal.rename_noreplace(source, destination)
    except FileExistsError as error:
        queue.put(('exists', source.name, error.errno))
    except Exception as error:
        queue.put(('error', source.name, type(error).__name__))
    else:
        queue.put(('published', source.name, None))


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_fallback_cross_process_publication_has_one_winner(tmp_path, kind):
    import errno

    sources = [tmp_path / 'first', tmp_path / 'second']
    for source in sources:
        if kind == 'file':
            source.write_text(source.name)
        else:
            source.mkdir()
            (source / 'payload.txt').write_text(source.name)
    destination = tmp_path / 'published'
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(2)
    queue = context.Queue()
    workers = [context.Process(target=_race_publish, args=(source, destination, barrier, queue))
               for source in sources]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert not worker.is_alive() and worker.exitcode == 0
        results = [queue.get(timeout=2) for _ in workers]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    assert sorted(row[0] for row in results) == ['exists', 'published']
    winner = next(row[1] for row in results if row[0] == 'published')
    loser = next(row[1] for row in results if row[0] == 'exists')
    assert next(row[2] for row in results if row[0] == 'exists') == errno.EEXIST
    data = destination if kind == 'file' else destination / 'payload.txt'
    assert data.read_text() == winner
    assert not (tmp_path / winner).exists() and (tmp_path / loser).exists()
    assert set(path.name for path in tmp_path.iterdir()) == {'published', loser}
    assert data.stat().st_nlink == 1


@pytest.mark.parametrize('code', ['EINVAL', 'ENOTSUP', 'ENOSYS'])
def test_unsupported_no_replace_flags_use_fallback(tmp_path, monkeypatch, code):
    import errno
    from agent.model_client.learning import journal

    def unsupported(*args):
        raise OSError(getattr(errno, code), 'forced unsupported flag')

    monkeypatch.setattr(journal, '_renameat2_noreplace', unsupported)
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_text('original')
    journal.rename_noreplace(source, destination)
    assert destination.read_text() == 'original' and not source.exists()
    source.write_text('replacement')
    with pytest.raises(FileExistsError):
        journal.rename_noreplace(source, destination)
    assert source.read_text() == 'replacement' and destination.read_text() == 'original'


@pytest.mark.parametrize('code', ['EACCES', 'EIO', 'EXDEV', 'ENOENT', 'EEXIST'])
def test_other_native_rename_errors_are_not_masked(tmp_path, monkeypatch, code):
    import errno
    from agent.model_client.learning import journal

    def rejected(*args):
        raise OSError(getattr(errno, code), 'native error')

    monkeypatch.setattr(journal, '_renameat2_noreplace', rejected)
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_text('untouched')
    with pytest.raises(OSError) as caught:
        journal.rename_noreplace(source, destination)
    assert caught.value.errno == getattr(errno, code)
    assert source.read_text() == 'untouched' and not destination.exists()


def test_fallback_does_not_overwrite_dangling_symlink(tmp_path, monkeypatch):
    import errno
    from agent.model_client.learning import journal

    def unsupported(*args):
        raise OSError(errno.EINVAL, 'unsupported flag')

    monkeypatch.setattr(journal, '_renameat2_noreplace', unsupported)
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_text('source')
    destination.symlink_to('missing')
    with pytest.raises(FileExistsError):
        journal.rename_noreplace(source, destination)
    assert destination.is_symlink() and source.read_text() == 'source'
