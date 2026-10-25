"""Versioned low-latency transport for SuperDex hand commands."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..hand import DEXHAND_DOF

Array = NDArray[np.float64]

DEFAULT_HAND_ENDPOINT = "tcp://127.0.0.1:5570"
_TOPIC = b"superdex.hand.v1 "
_MAGIC = b"SDHC"
_VERSION = 1
_HEADER = struct.Struct("<4sB3xQQ")
_PAYLOAD_BYTES = DEXHAND_DOF * np.dtype("<f4").itemsize


@dataclass(frozen=True, slots=True)
class SuperDexHandCommand:
    sequence: int
    monotonic_ns: int
    joint_position: Array

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        value = np.asarray(self.joint_position, dtype=np.float64)
        if value.shape != (DEXHAND_DOF,):
            raise ValueError(f"joint_position must have shape ({DEXHAND_DOF},)")
        if not np.isfinite(value).all():
            raise ValueError("joint_position must contain finite values")
        object.__setattr__(self, "joint_position", value.copy())

    def encode(self) -> bytes:
        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            self.sequence,
            self.monotonic_ns,
        )
        payload = self.joint_position.astype("<f4", copy=False).tobytes()
        return _TOPIC + header + payload

    @classmethod
    def decode(cls, message: bytes) -> SuperDexHandCommand:
        if not message.startswith(_TOPIC):
            raise ValueError("invalid SuperDex hand command topic")
        packet = memoryview(message)[len(_TOPIC) :]
        if len(packet) != _HEADER.size + _PAYLOAD_BYTES:
            raise ValueError("invalid SuperDex hand command size")
        magic, version, sequence, monotonic_ns = _HEADER.unpack(packet[: _HEADER.size])
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("unsupported SuperDex hand protocol")
        joint_position = np.frombuffer(
            packet[_HEADER.size :], dtype="<f4", count=DEXHAND_DOF
        ).astype(np.float64)
        return cls(sequence, monotonic_ns, joint_position)


class HandCommandPublisher:
    """Publish newest-only hand targets from the PICO process."""

    def __init__(self, endpoint: str = DEFAULT_HAND_ENDPOINT, *, bind: bool = True):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        (self._socket.bind if bind else self._socket.connect)(endpoint)
        self._sequence = 0

    def publish(self, joint_position: Array) -> bool:
        command = SuperDexHandCommand(
            self._sequence,
            time.monotonic_ns(),
            joint_position,
        )
        self._sequence += 1
        try:
            self._socket.send(command.encode(), flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False
        return True

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class HandCommandSubscriber:
    """Receive the newest hand target without blocking the physics loop."""

    def __init__(
        self,
        endpoint: str = DEFAULT_HAND_ENDPOINT,
        *,
        bind: bool = False,
    ) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, _TOPIC)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        (self._socket.bind if bind else self._socket.connect)(endpoint)

    def receive(self) -> SuperDexHandCommand | None:
        try:
            message = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None
        return SuperDexHandCommand.decode(message)

    def close(self) -> None:
        self._socket.close()
        self._context.term()
