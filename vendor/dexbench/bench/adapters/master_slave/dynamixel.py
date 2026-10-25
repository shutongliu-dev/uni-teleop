"""Read-only Linux DYNAMIXEL Protocol 2 transport for FACTR X-series leaders.

Only READ and SYNC_READ instructions are exposed. No torque/mode/goal writes.
Control tables: ROBOTIS XC330-T288 and XM430-W210, signed position at 132,
4096 encoder units/revolution. Physical directions remain installation config.
"""
import fcntl
import math
import os
import select
import struct
import termios
import time

HEADER = b'\xff\xff\xfd\x00'


def crc(data):
    value = 0
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x8005 if value & 0x8000 else 0)) & 0xffff
    return value


def packet(motor, instruction, parameters):
    body = (bytes([instruction]) + parameters).replace(b'\xff\xff\xfd', b'\xff\xff\xfd\xfd')
    result = HEADER + bytes([motor]) + struct.pack('<H', len(body)+2) + body
    return result + struct.pack('<H', crc(result))


class ReadOnlyLeader:
    def __init__(self, port, *, baud=1000000, ids=tuple(range(1, 9)), timeout=.1):
        if len(ids) != 8 or len(set(ids)) != 8 or any(not 0 <= i <= 252 for i in ids):
            raise ValueError('FACTR requires eight distinct motor IDs')
        self.ids, self.timeout = tuple(ids), timeout
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            self.original = termios.tcgetattr(self.fd)
            settings = termios.tcgetattr(self.fd)
            settings[0] = settings[1] = settings[3] = 0
            settings[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            settings[4] = settings[5] = getattr(termios, 'B'+str(baud))
            settings[6][termios.VMIN] = settings[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, settings)
        except BaseException:
            os.close(self.fd)
            raise
        self.buffer = bytearray()

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            try:
                termios.tcsetattr(fd, termios.TCSANOW, self.original)
                fcntl.ioctl(fd, termios.TIOCNXCL)
            finally:
                os.close(fd)

    def _status(self, deadline):
        while True:
            start = self.buffer.find(HEADER)
            if start >= 0:
                del self.buffer[:start]
                if len(self.buffer) >= 7:
                    size = 7 + struct.unpack_from('<H', self.buffer, 5)[0]
                    if not 11 <= size <= 1024:
                        raise ValueError('Invalid DYNAMIXEL status length')
                    if len(self.buffer) >= size:
                        raw = bytes(self.buffer[:size]); del self.buffer[:size]
                        if crc(raw[:-2]) != int.from_bytes(raw[-2:], 'little'):
                            raise ValueError('DYNAMIXEL CRC mismatch')
                        body = raw[7:-2].replace(b'\xff\xff\xfd\xfd', b'\xff\xff\xfd')
                        if body[0] != 0x55 or body[1]:
                            raise ValueError(f'DYNAMIXEL {raw[4]} status error: {body.hex()}')
                        return raw[4], body[2:]
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('DYNAMIXEL response timed out')
            if select.select([self.fd], [], [], remaining)[0]:
                data = os.read(self.fd, 4096)
                if not data:
                    raise OSError('DYNAMIXEL serial connection closed')
                self.buffer.extend(data)

    def read(self, motor, address, size):
        self.buffer.clear(); termios.tcflush(self.fd, termios.TCIFLUSH)
        os.write(self.fd, packet(motor, 0x02, struct.pack('<HH', address, size)))
        identity, data = self._status(time.monotonic()+self.timeout)
        if identity != motor or len(data) != size:
            raise ValueError('Unexpected DYNAMIXEL read response')
        return data

    def positions(self):
        self.buffer.clear(); termios.tcflush(self.fd, termios.TCIFLUSH)
        started_ns = time.monotonic_ns()
        os.write(self.fd, packet(0xfe, 0x82, struct.pack('<HH', 132, 4)+bytes(self.ids)))
        values = {}
        deadline = time.monotonic()+self.timeout
        while len(values) < len(self.ids):
            identity, data = self._status(deadline)
            if identity not in self.ids or identity in values or len(data) != 4:
                raise ValueError('Unexpected DYNAMIXEL sync-read response')
            values[identity] = struct.unpack('<i', data)[0] * (2*math.pi/4096)
        return tuple(values[i] for i in self.ids), started_ns
