"""Explicit Linux input-device pedal reader; no GUI focus or SDK dependency."""
import fcntl
import os
import struct
import stat
import time

EVENT = struct.Struct('llHHi')
EV_KEY = 1
EVIOCGRAB = 0x40044590


class PressEdges:
    def __init__(self, code, *, pressed=False, debounce_s=.25):
        self.code, self.pressed = code, pressed
        self.debounce_s, self.last_press = debounce_s, float('-inf')

    def feed(self, kind, code, value, now):
        if kind != EV_KEY or code != self.code:
            return False
        if value == 0:
            self.pressed = False
        elif value == 1 and not self.pressed:
            self.pressed = True
            if now-self.last_press >= self.debounce_s:
                self.last_press = now
                return True
        return False


class Pedal:
    def __init__(self, path, code=61):
        if not 0 <= code < 768:
            raise ValueError('Invalid Linux pedal key code')
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            if not stat.S_ISCHR(os.fstat(self.fd).st_mode):
                raise ValueError('Pedal must be an evdev character device')
            keys = bytearray(96)
            fcntl.ioctl(self.fd, 0x80000000 | (len(keys) << 16) | (ord('E') << 8) | 0x18, keys)
            self.edges = PressEdges(code, pressed=bool(keys[code//8] & (1 << (code%8))))
            # Use kernel monotonic timestamps for debounce even after delayed polling.
            fcntl.ioctl(self.fd, 0x400445A0, struct.pack('i', time.CLOCK_MONOTONIC))
            fcntl.ioctl(self.fd, EVIOCGRAB, 1)
        except BaseException:
            os.close(self.fd)
            raise

    def poll(self):
        presses = 0
        while True:
            try:
                data = os.read(self.fd, EVENT.size*64)
            except BlockingIOError:
                return presses
            if not data:
                raise OSError('Pedal disconnected')
            for offset in range(0, len(data), EVENT.size):
                sec, usec, kind, code, value = EVENT.unpack_from(data, offset)
                if kind == 0 and code == 3:
                    raise OSError('Pedal event queue overflow; restart to resynchronize')
                presses += self.edges.feed(kind, code, value, sec+usec/1e6)

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, EVIOCGRAB, 0)
            except OSError:
                pass  # Unplugged devices may reject the release ioctl.
            finally:
                os.close(self.fd)
                self.fd = None
