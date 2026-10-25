"""Length-delimited JSON over a private local socket; never deserialize pickle."""
import json
import struct

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _read(sock, count, *, wait_for_data=None):
    parts = bytearray()
    while len(parts) < count:
        if wait_for_data is not None:
            wait_for_data(sock)
        block = sock.recv(count - len(parts))
        if not block:
            raise ConnectionError('Isaac worker disconnected')
        parts.extend(block)
    return bytes(parts)


def receive(sock, *, wait_for_data=None):
    size = struct.unpack('!I', _read(sock, 4, wait_for_data=wait_for_data))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError('IPC message exceeds the recorded transport limit')
    return json.loads(_read(sock, size, wait_for_data=wait_for_data), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def send(sock, value):
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError('IPC message exceeds the recorded transport limit')
    sock.sendall(struct.pack('!I', len(data)) + data)
