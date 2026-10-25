"""One fresh rigid-contact acquisition per native substep, two private projections.

No state cache. Both observer state captures and layer checks remain independent.
A registered exchange fails closed on stale/foreign/reentrant/duplicate delivery.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np

PROTOCOL = 'fresh_native_substep_contact_dual_projection_v1'


def require(ok, reason):
    if not ok:
        raise RuntimeError('Contact exchange: ' + reason)


def copy_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    result = np.asarray(value).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ContactSample:
    exchange: object
    key: tuple
    arrays: tuple

    def for_env(self, env):
        self.exchange.validate_sample(env, self)
        return self.arrays


class SubstepContactExchange:
    def __init__(self, env, owner):
        require(getattr(env, '_contact_exchange', None) is None, 'observer already registered')
        self.env, self.owner = env, owner
        self.epoch = env._reset_count
        self.view = env._rigid_contacts
        self.mapping = self._mapping(env)
        self.last_step = int(env.world.current_time_step_index)
        self.last_time = float(env.world.current_time)
        self.last_public_time = float(env.time_s)
        self.dt = float(env.dt)
        self.phase = 'idle'
        self.sample = None
        self.rows = None
        self.acquisitions = self.deliveries = 0
        self.ledger = []
        env._contact_exchange = self

    @staticmethod
    def _mapping(env):
        return (tuple(env._contact_sensor_paths), tuple(tuple(x) for x in env._contact_filter_paths),
                int(env._rigid_contacts.max_contact_data_count))

    def _registered(self, env):
        require(env is self.env and getattr(env, '_contact_exchange', None) is self, 'unregistered or foreign observer')
        require(env._reset_count == self.epoch, 'reset epoch changed')
        require(env._rigid_contacts is self.view and self._mapping(env) == self.mapping, 'contact view/mapping changed')
        require(float(env.dt) == self.dt, 'native dt changed')
        require(self.phase not in ('failed', 'closed'), 'exchange failed or closed')

    def _key(self, env):
        self._registered(env)
        clock = float(env.world.current_time)
        public = float(env._current_step_time)
        require(math.isfinite(clock) and math.isfinite(public), 'nonfinite native/public clock')
        return (self.epoch, int(env.world.current_time_step_index), clock, public)

    def begin_step(self, env):
        try:
            self._registered(env)
            require(self.phase == 'idle', 'reentry or unconsumed previous substep')
            require(int(env.world.current_time_step_index) == self.last_step and
                    float(env.world.current_time) == self.last_time, 'unobserved native advance')
            require(float(env.time_s) == self.last_public_time, 'public clock changed between substeps')
            require(float(env._current_step_time) == float(env.time_s + env.dt), 'incorrect planned public substep time')
            self.public_before = float(env.time_s)
            self.planned_public = float(env._current_step_time)
            self.phase = 'stepping'
        except BaseException:
            self.invalidate()
            raise

    def acquire(self, env):
        try:
            key = self._key(env)
            require(self.phase == 'stepping', 'duplicate acquisition or missing begin')
            require(key[3] == self.planned_public and float(env.time_s) == self.public_before,
                    'public clock changed during native step')
            require(key[1] == self.last_step + 1 and abs(key[2] - self.last_time - env.dt) <= 1e-10,
                    'sample is not the expected next native substep')
            # Copy *all six* returned quantities. Native views reuse their buffers.
            arrays = tuple(copy_array(value) for value in self.view.get_contact_data(dt=1.))
            require(len(arrays) == 6, 'incomplete native contact tuple')
            require(self._key(env) == key, 'clock/view changed during acquisition')
            self.sample = ContactSample(self, key, arrays)
            self.acquisitions += 1
            self.record = {'sequence': self.acquisitions, 'reset_epoch': self.epoch,
                           'native_step_index': key[1], 'native_time_s': key[2],
                           'public_time_before_s': self.public_before, 'planned_public_time_s': key[3],
                           'acquisition_count': 1, 'delivery_count': 0,
                           'readonly_verified': False, 'consumed_public_time_s': None}
            self.ledger.append(self.record)
            self.phase = 'sampled'
            return self.sample
        except BaseException:
            self.invalidate()
            raise

    def validate_sample(self, env, sample):
        try:
            require(self._key(env) == sample.key and self.phase == 'sampled' and sample is self.sample,
                    'foreign or stale sample')
            require(float(env.time_s) == self.public_before, 'public clock changed before readonly check')
        except BaseException:
            self.invalidate()
            raise

    def offer_after_readonly(self, env, sample, projected_rows):
        try:
            self.validate_sample(env, sample)
            self.key = sample.key
            self.rows = deepcopy(projected_rows)
            # Raw tensors never become a cross-capture or cross-step cache.
            self.sample = None
            self.record['readonly_verified'] = True
            self.phase = 'ready'
        except BaseException:
            self.invalidate()
            raise

    def consume(self, env):
        try:
            require(self.phase == 'ready', 'missing or already consumed substep')
            require(self._key(env) == self.key, 'stale native substep or epoch')
            require(float(env.time_s) == self.key[3], 'public clock does not match completed substep')
            rows = self.rows
            self.rows = None
            self.last_step, self.last_time = self.key[1:3]
            self.last_public_time = self.key[3]
            self.deliveries += 1
            self.record['delivery_count'] = 1
            self.record['consumed_public_time_s'] = float(env.time_s)
            self.phase = 'idle'
            return rows
        except BaseException:
            self.invalidate()
            raise

    def invalidate(self):
        self.phase, self.sample, self.rows = 'failed', None, None

    def close(self, env, owner):
        require(env is self.env and owner is self.owner and getattr(env, '_contact_exchange', None) is self,
                'only the registered owner may close')
        clean = (self.phase == 'idle' and self.acquisitions == self.deliveries
                 and env._reset_count == self.epoch and env._rigid_contacts is self.view
                 and self._mapping(env) == self.mapping
                 and int(env.world.current_time_step_index) == self.last_step
                 and float(env.world.current_time) == self.last_time
                 and float(env.time_s) == self.last_public_time and float(env.dt) == self.dt
                 and len(self.ledger) == self.acquisitions
                 and all(row['sequence'] == i and row['acquisition_count'] == row['delivery_count'] == 1
                         and row['readonly_verified'] is True for i, row in enumerate(self.ledger, 1)))
        result = {'protocol': PROTOCOL, 'acquisitions': self.acquisitions, 'deliveries': self.deliveries,
                  'all_samples_consumed_once': clean, 'reset_epoch': self.epoch,
                  'samples': deepcopy(self.ledger), 'raw_arrays_reused_across_native_steps': False,
                  'state_captures_reused': False}
        self.phase, self.sample, self.rows = 'closed', None, None
        env._contact_exchange = None
        return result
