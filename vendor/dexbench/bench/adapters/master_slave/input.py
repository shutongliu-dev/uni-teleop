"""Latest complete read-only FACTR sample, acquired outside the Isaac loop."""
from pathlib import Path
import threading
import time

from bench.adapters.pico.protocol import Frame
from .dynamixel import ReadOnlyLeader


class LeaderReceiver:
    rejected = 0

    def __init__(self, port, *, baud=4000000):
        self.leader_id = Path(port).name
        self.driver = ReadOnlyLeader(port, baud=baud)
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        # Fail visibly before entering Isaac if the configured leader is absent.
        try:
            values, stamp = self.driver.positions()
        except BaseException:
            self.driver.close()
            raise
        self.sequence = 0
        self._publish(values, stamp)
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _publish(self, values, stamp):
        self.sequence += 1
        self.latest = Frame('factr-readonly', self.sequence, stamp, str(stamp), 'FACTR',
                            {'head': None, 'controllers': {}, 'hands': {}, 'leader': list(values), 'leader_id': self.leader_id})
        self.error = None

    def _receive(self):
        while not self.stop.is_set():
            try:
                self._publish(*self.driver.positions())
            except (OSError, ValueError) as error:
                self.error = str(error)
                self.rejected += 1
            self.stop.wait(.01)

    def read(self):
        return self.latest

    def health(self):
        frame = self.latest
        return {'source': 'factr_readonly', 'received': self.sequence, 'rejected': self.rejected,
                'latest_age_s': (time.monotonic_ns()-frame.host_ns)/1e9 if frame else None,
                'last_error': self.error}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
        self.driver.close()
