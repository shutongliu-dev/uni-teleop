"""Checksummed host events and durable JSON I/O (not an OS sandbox).

Only a trusted host may access this directory. Advisory locks coordinate cooperative
writers; hostile same-user processes require the separate runtime isolation layer.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Iterator
import uuid


class JournalError(ValueError):
    """Persistent records are malformed or inconsistent."""


class JournalBusyError(JournalError):
    """Another host owns the writer lock."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JournalError("duplicate JSON key")
        result[key] = value
    return result


def strict_json_loads(value: str | bytes) -> Any:
    def invalid(value: str) -> None:
        raise JournalError("non-finite JSON value")
    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise JournalError("non-finite JSON value")
        return number
    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid, parse_float=finite_float)
    except (ValueError, UnicodeError) as exc:
        raise JournalError("invalid strict JSON") from exc


def canonical_json(value: Any) -> bytes:
    try:
        # Roundtrip catches non-string keys and collisions after JSON coercion.
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        if strict_json_loads(encoded) != value:
            raise JournalError("JSON values must roundtrip without coercion")
        return encoded
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JournalError("value is not canonical JSON") from exc


def checked_directory(path: Path | str, *, create: bool = False) -> Path:
    """Reject links in every existing component of a host-owned directory."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if create:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise JournalError("expected a real directory, not a link or special file")
    return path


def _regular(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise JournalError("expected a regular file with one link")


def read_bytes(path: Path | str, *, limit: int = 16 * 1024 * 1024) -> bytes:
    path = Path(path)
    checked_directory(path.parent)
    # O_NONBLOCK keeps a replaced FIFO from hanging before fstat rejects it.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _regular(before)
        if before.st_size > limit:
            raise JournalError("file size exceeds limit")
        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise JournalError("file size exceeds limit")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise JournalError("file changed during read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_json(path: Path | str) -> Any:
    return strict_json_loads(read_bytes(path))


def fsync_directory(path: Path | str) -> None:
    checked_directory(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _renameat2_noreplace(source: Path, name: str, destination_fd: int) -> None:
    """Attempt the kernel no-replace primitive without changing error semantics."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from None
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), destination_fd, os.fsencode(name), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), name)


def rename_noreplace(source: Path, destination: Path) -> None:
    """Publish once, including filesystems without RENAME_NOREPLACE support.

    All cooperative publishers lock the destination directory itself, including
    the native path, so a native publisher cannot race the fallback's existence
    check. Unsupported kernel/filesystem flags use ordinary atomic rename while
    holding that lock. No lock file or temporary hard link enters the payload.

    Directory flock is advisory: the trusted host must exclusively own writes
    through this helper. This is not protection against malicious same-UID
    mutation, nor a claim of cross-host locking or power-loss guarantees.
    """
    source, destination = Path(source), Path(destination)
    checked_directory(source.parent)
    parent = checked_directory(destination.parent)
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            _renameat2_noreplace(source, destination.name, fd)
            return
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                raise
        try:
            os.stat(destination.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, "destination already exists", str(destination))
        os.rename(source, destination.name, dst_dir_fd=fd)
    finally:
        # Closing the descriptor releases flock, including on publication errors
        # and exceptions. Abrupt process exit likewise releases the kernel lock.
        os.close(fd)


def atomic_write_bytes(path: Path | str, value: bytes, *, replace: bool = True,
                       mode: int = 0o600) -> None:
    path = Path(path)
    checked_directory(path.parent)
    try:
        _regular(path.lstat())
        if not replace:
            raise FileExistsError(path)
    except FileNotFoundError:
        pass
    temporary = path.parent / (".pending-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            rename_noreplace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path | str, value: Any, *, replace: bool = True) -> None:
    atomic_write_bytes(path, canonical_json(value), replace=replace)


class Journal:
    """One immutable JSON file per committed event, chained by SHA-256.

    ``transaction`` serializes multi-event host operations. It is reentrant only in
    the owning thread. Complete numbered events must form an unbroken chain;
    uncommitted .pending files are ignored and never promoted automatically.
    """

    def __init__(self, path: Path | str):
        self.path = checked_directory(path, create=True)
        self._mutex = threading.RLock()
        self._depth = 0
        self._lock_fd: int | None = None

    @contextmanager
    def transaction(self) -> Iterator[Journal]:
        if not self._mutex.acquire(blocking=False):
            raise JournalBusyError("journal writer is busy")
        try:
            if self._depth == 0:
                checked_directory(self.path)
                fd = os.open(self.path / ".writer.lock", os.O_RDWR | os.O_CREAT |
                             os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
                try:
                    _regular(os.fstat(fd))
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    os.close(fd)
                    raise JournalBusyError("journal writer is busy") from exc
                except BaseException:
                    os.close(fd)
                    raise
                self._lock_fd = fd
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                if not self._depth:
                    assert self._lock_fd is not None
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    os.close(self._lock_fd)
                    self._lock_fd = None
        finally:
            self._mutex.release()

    def read(self) -> list[dict[str, Any]]:
        with self.transaction():
            entries: list[dict[str, Any]] = []
            event_ids: set[str] = set()
            previous = None
            paths = []
            for path in self.path.iterdir():
                if path.name == ".writer.lock" or path.name.startswith(".pending-"):
                    continue
                if not re.fullmatch(r"\d{20}\.json", path.name):
                    raise JournalError("unexpected committed journal entry")
                paths.append(path)
            for index, path in enumerate(sorted(paths), 1):
                event = read_json(path)
                if not isinstance(event, dict) or set(event) != {
                    "sequence", "event_id", "kind", "transaction_id", "payload",
                    "previous_hash", "event_hash"
                }:
                    raise JournalError("invalid event schema")
                if (type(event["sequence"]) is not int or event["sequence"] != index
                        or path.name != f"{index:020d}.json"
                        or event["previous_hash"] != previous):
                    raise JournalError("journal sequence or chain mismatch")
                if (not isinstance(event["event_id"], str) or not event["event_id"]
                        or event["event_id"] in event_ids
                        or not isinstance(event["kind"], str) or not event["kind"]
                        or not isinstance(event["payload"], dict)
                        or (event["transaction_id"] is not None and
                            not isinstance(event["transaction_id"], str))):
                    raise JournalError("invalid event fields")
                body = {key: value for key, value in event.items() if key != "event_hash"}
                expected = hashlib.sha256(b"agent-journal-v1\0" + canonical_json(body)).hexdigest()
                if event["event_hash"] != expected:
                    raise JournalError("journal checksum mismatch")
                previous = expected
                event_ids.add(event["event_id"])
                entries.append(event)
            return entries

    def append(self, kind: str, payload: dict[str, Any], *, event_id: str | None = None,
               transaction_id: str | None = None) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind or not isinstance(payload, dict):
            raise JournalError("event kind and object payload are required")
        if transaction_id is not None and (not isinstance(transaction_id, str) or not transaction_id):
            raise JournalError("invalid transaction ID")
        event_id = uuid.uuid4().hex if event_id is None else event_id
        if not isinstance(event_id, str) or not event_id:
            raise JournalError("invalid event ID")
        canonical_json(payload)
        with self.transaction():
            events = self.read()
            for old in events:
                if old["event_id"] == event_id:
                    if (old["kind"], old["payload"], old["transaction_id"]) != (kind, payload, transaction_id):
                        raise JournalError("event ID reused with different content")
                    return old
            body = {
                "sequence": len(events) + 1, "event_id": event_id,
                "kind": kind, "transaction_id": transaction_id, "payload": payload,
                "previous_hash": events[-1]["event_hash"] if events else None,
            }
            event = {**body, "event_hash": hashlib.sha256(
                b"agent-journal-v1\0" + canonical_json(body)).hexdigest()}
            atomic_write_json(self.path / f"{event['sequence']:020d}.json", event, replace=False)
            return event
