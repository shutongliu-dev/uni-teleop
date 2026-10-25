"""Host-owned action recovery through public HTTP; never replay physical intent.

The journal must live outside the generated program's writable mount. Recovery
binds to the original episode and trial and restores only host bookkeeping, not
Python namespaces or execution stacks. This is not a filesystem sandbox.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from uuid import uuid4

from ..actions import public_observation
from ..bench_http import BenchHTTPError, _token
from ..time_limits import optional_positive
from .journal import Journal, canonical_json


DEFAULT_MAX_STATUS_QUERIES = 1200


TERMINAL = frozenset(('completed', 'cancelled', 'failed'))
STATUSES = TERMINAL | {'accepted', 'running'}


class ActionRecoveryError(RuntimeError):
    """The host cannot establish a safe, causally bound action state."""


class ActionPendingError(ActionRecoveryError):
    """An unresolved action prevents all further physical submissions."""


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _query_limit(value):
    if type(value) is not int or value <= 0:
        raise ValueError('max_status_queries must be a positive integer')
    return value


class RobotAPI:
    """One durable physical-action stream for one episode/trial.

    ``request`` optionally installs a host deadline (InteractionSession.http).
    POST is attempted at most once per ID, including after process restart.
    Unknown POST completion and 404 status reads remain unresolved. Cancellation
    requests never unlock the stream; only a terminal receipt with its matching
    public snapshot can do so. Opening a new ledger cannot discover unrelated
    actions, so the trusted scheduler must assign one ledger per episode.
    """

    def __init__(self, bench, journal_path, *, episode_id, trial_id, binding=None,
                 request=None, max_status_queries=DEFAULT_MAX_STATUS_QUERIES,
                 submission_rejection=None):
        self.max_status_queries = _query_limit(max_status_queries)
        self.bench = bench
        self.episode_id = _token(episode_id)
        self.trial_id = _token(trial_id)
        self.request = request or (lambda function, *args, **kwargs: function(*args, **kwargs))
        # A trusted transport may prove that a named public-function request
        # was rejected before any action. Ordinary HTTP exceptions never do.
        self.submission_rejection = submission_rejection
        self.journal = Journal(Path(journal_path))
        self.binding = {'protocol': 'agent_action_recovery_v1',
                        'episode_id': self.episode_id, 'trial_id': self.trial_id,
                        'bench_origin': getattr(bench, 'base_url', None),
                        'recovery_policy': {'max_status_queries': self.max_status_queries,
                                            'scope': 'per_wait_status_operations'},
                        'contract': deepcopy(binding or {})}
        # Verify JSON before creating any state.
        canonical_json(self.binding)
        with self.journal.transaction():
            events = self.journal.read()
            if not events:
                self.journal.append('action_ledger_bound', self.binding)
            self._state()

    def _state(self):
        events = self.journal.read()
        if (not events or events[0]['kind'] != 'action_ledger_bound'
                or events[0]['payload'] != self.binding):
            raise ActionRecoveryError('Action ledger episode/trial binding mismatch')
        actions = {}
        pending = None
        for event in events[1:]:
            value = event['payload']
            action_id = value.get('action_id')
            kind = event['kind']
            if kind == 'action_prepared':
                if pending is not None or action_id in actions:
                    raise ActionRecoveryError('Invalid action preparation history')
                command = value.get('command')
                if (not isinstance(command, dict) or command.get('action_id') != action_id
                        or _hash(command) != value.get('payload_sha256')
                        or command.get('observation_sequence') != value.get('expected_sequence')):
                    raise ActionRecoveryError('Action payload integrity mismatch')
                pending = deepcopy(value)
                pending.update(submit_started=False, cancellation_requested=False, receipt=None, waits={})
                actions[action_id] = pending
                continue
            if pending is None or action_id != pending['action_id']:
                raise ActionRecoveryError('Action event does not match the pending action')
            if kind == 'action_submit_started':
                if pending['submit_started']:
                    raise ActionRecoveryError('Action was submitted more than once')
                pending['submit_started'] = True
            elif kind in ('action_receipt', 'action_cancel_receipt'):
                self._receipt(value['receipt'], action_id)
                pending['receipt'] = value['receipt']
            elif kind == 'action_cancel_requested':
                if pending['cancellation_requested']:
                    raise ActionRecoveryError('Cancellation was requested more than once')
                pending['cancellation_requested'] = True
            elif kind == 'action_transport_error':
                pass  # Bookkeeping only: a failed request proves no physical state.
            elif kind == 'action_rejected':
                if not pending['submit_started'] or pending['receipt'] is not None:
                    raise ActionRecoveryError('Invalid action rejection history')
                self._rejection(value['response'], pending['command'])
                pending['rejection'] = deepcopy(value['response'])
                pending = None
            elif kind == 'action_wait_started':
                limit = value.get('max_status_queries')
                wait_id = value.get('wait_id')
                if (type(limit) is not int or not 0 < limit <= self.max_status_queries
                        or not isinstance(wait_id, str) or not wait_id or wait_id in pending['waits']):
                    raise ActionRecoveryError('Invalid recovery polling budget')
                pending['waits'][wait_id] = {'max_status_queries': limit, 'exhausted': False}
            elif kind == 'action_wait_exhausted':
                wait = pending['waits'].get(value.get('wait_id'))
                queries = value.get('status_queries')
                reason = value.get('reason')
                if (wait is None or wait['exhausted'] or type(queries) is not int
                        or not 0 <= queries <= wait['max_status_queries']
                        or value.get('max_status_queries') != wait['max_status_queries']
                        or reason not in ('status_query_limit', 'wall_deadline')
                        or (reason == 'status_query_limit' and queries != wait['max_status_queries'])):
                    raise ActionRecoveryError('Invalid recovery polling exhaustion event')
                wait['exhausted'] = True
                wait['status_queries'] = queries
                wait['reason'] = reason
            elif kind == 'action_reconciled':
                self._match_snapshot(pending, value['receipt'], value['observation'])
                pending['settled'] = deepcopy(value)
                pending = None
            else:
                raise ActionRecoveryError('Unknown action journal event')
        return actions, pending

    @property
    def pending(self):
        with self.journal.transaction():
            return deepcopy(self._state()[1])

    def assert_idle(self):
        if self.pending is not None:
            raise ActionPendingError('Resolve the original pending action before a new command')

    def _receipt(self, value, action_id):
        if (not isinstance(value, dict) or value.get('action_id') != action_id
                or value.get('status') not in STATUSES or not _finite(value.get('sim_time_s'))
                or value.get('episode_id', self.episode_id) != self.episode_id):
            raise BenchHTTPError('Action receipt is invalid or belongs to another action',
                                 category='invalid_response')
        return value

    @staticmethod
    def _observation(value, episode_id):
        try:
            parsed = public_observation(value)
        except (ValueError, TypeError):
            raise BenchHTTPError('Invalid public recovery observation', category='invalid_response') from None
        if parsed.episode_id != episode_id:
            raise BenchHTTPError('Recovery observation belongs to another episode', category='invalid_response')
        return parsed

    def _match_snapshot(self, pending, receipt, observation):
        self._receipt(receipt, pending['action_id'])
        parsed = self._observation(observation, self.episode_id)
        initial = pending['initial_observation']
        if (receipt['status'] not in TERMINAL
                or parsed.last_action_id != pending['action_id']
                or parsed.last_action_status != receipt['status']
                or parsed.sequence < pending['expected_sequence']
                or parsed.sim_time_s < initial['sim_time_s']
                or not math.isclose(parsed.sim_time_s, receipt['sim_time_s'], rel_tol=0., abs_tol=1e-8)
                or _hash(observation) == initial['sha256']):
            raise BenchHTTPError('Terminal action has no matching fresh public observation',
                                 category='invalid_response')
        # A controller may converge immediately; failure/cancellation can also
        # take zero steps. Freshness comes from the new action identity, with
        # sequence and simulation time consistently unchanged or both advanced.
        if ((parsed.sequence == pending['expected_sequence'])
                != math.isclose(parsed.sim_time_s, initial['sim_time_s'], rel_tol=0., abs_tol=1e-9)):
            raise BenchHTTPError('Recovery snapshot sequence and clock disagree',
                                 category='invalid_response')

    def _error(self, action_id, operation, error):
        metadata = (error.public_metadata() if isinstance(error, BenchHTTPError)
                    else {'error_category': 'unknown', 'http_status': None})
        self.journal.append('action_transport_error', {'action_id': action_id,
                            'operation': operation, **metadata})

    def _rejection(self, response, command):
        """Validate persisted proof; never infer rejection from HTTP status."""
        from bench.protocol.functions import FunctionFailure
        try:
            value = FunctionFailure.model_validate(response)
        except (ValueError, TypeError):
            raise ActionRecoveryError('Invalid definite action rejection') from None
        definite = {'invalid_arguments', 'unsupported_function', 'unsupported_capability',
                    'stale_observation', 'episode_finished', 'budget_exhausted', 'permission_denied'}
        pre_dispatch = (value.function is None and value.request_id is None
                        and value.context is None and value.error.code == 'permission_denied')
        if (value.error.code not in definite
                or (not pre_dispatch and value.function != command.get('function'))
                or (value.context is not None and value.context.episode_id != self.episode_id)):
            raise ActionRecoveryError('Response cannot prove no action was submitted')
        return value.model_dump(mode='json')

    def submit(self, command, observation):
        """Persist intent and its submission marker before a single POST."""
        command, observation = deepcopy(command), deepcopy(observation)
        if not isinstance(command, dict):
            raise ActionRecoveryError('Command must be a JSON object')
        action_id = _token(command.get('action_id'))
        parsed = self._observation(observation, self.episode_id)
        expected = command.get('observation_sequence')
        if type(expected) is not int or expected != parsed.sequence:
            raise ActionRecoveryError('Command does not bind to the supplied observation')
        if parsed.last_action_status in ('accepted', 'running'):
            raise ActionPendingError('Public observation still has an active action')
        payload_sha = _hash(command)
        with self.journal.transaction():
            actions, pending = self._state()
            if pending is not None:
                raise ActionPendingError('Resolve the original pending action before a new command')
            if action_id in actions or action_id == parsed.last_action_id:
                raise ActionRecoveryError('An action ID cannot be reused')
            self.journal.append('action_prepared', {'action_id': action_id,
                'command': command, 'payload_sha256': payload_sha,
                'expected_sequence': expected, 'initial_observation': {
                    'sequence': parsed.sequence, 'sim_time_s': parsed.sim_time_s,
                    'last_action_id': parsed.last_action_id, 'sha256': _hash(observation)}})
            self.journal.append('action_submit_started', {'action_id': action_id})
            try:
                value = self.request(self.bench.submit, self.episode_id, command)
                self._receipt(value, action_id)
            except Exception as error:
                response = (None if self.submission_rejection is None else
                            self.submission_rejection(error))
                if response is not None:
                    response = self._rejection(response, command)
                    self.journal.append('action_rejected', {'action_id': action_id,
                                                         'response': response})
                    raise
                self._error(action_id, 'submit', error)
                raise
            self.journal.append('action_receipt', {'action_id': action_id, 'receipt': value})
            return deepcopy(value)

    def reconcile(self, *, expected_action_id=None):
        """Query the original ID once, then fetch its public snapshot if terminal.

        Returns None while active. Any read or snapshot failure leaves the action
        pending. No command is resubmitted, even if the server returns 404.
        """
        with self.journal.transaction():
            _, pending = self._state()
            if pending is None:
                return None
            action_id = pending['action_id']
            if expected_action_id is not None and action_id != expected_action_id:
                raise ActionRecoveryError('Pending action changed during recovery polling')
            try:
                receipt = self.request(self.bench.get_action, self.episode_id, action_id)
                self._receipt(receipt, action_id)
                # Preserve state transitions, not every identical polling sample.
                previous = pending['receipt']
                if previous is None or previous['status'] != receipt['status']:
                    self.journal.append('action_receipt', {'action_id': action_id, 'receipt': receipt})
                if receipt['status'] not in TERMINAL:
                    return None
                observation = self.request(self.bench.get_observation, self.episode_id)
                self._match_snapshot(pending, receipt, observation)
            except Exception as error:
                self._error(action_id, 'reconcile', error)
                raise
            value = {'action_id': action_id, 'receipt': receipt, 'observation': observation}
            self.journal.append('action_reconciled', value)
            return deepcopy(value)

    def wait(self, *, timeout_s, poll_interval_s=0.25, max_status_queries=None):
        """Poll within both a time budget and a finite status-query budget.

        The configured default is 1200 status operations per wait. A call may
        tighten, but cannot increase, its bound ledger policy. This count remains
        active when timeout_s is None; each HTTP GET operation can separately use
        BenchHTTPClient's bounded read retry policy. The episode wall budget is
        unaffected. Exhaustion never releases the pending action or sends POST.
        """
        timeout = optional_positive(timeout_s, 'timeout_s')
        interval = optional_positive(poll_interval_s, 'poll_interval_s')
        if interval is None:
            raise ValueError('poll_interval_s must be positive')
        limit = self.max_status_queries if max_status_queries is None else _query_limit(max_status_queries)
        if limit > self.max_status_queries:
            raise ValueError('max_status_queries cannot exceed the bound recovery policy')
        deadline = None if timeout is None else time.monotonic() + timeout
        wait_id = uuid4().hex
        with self.journal.transaction():
            _, pending = self._state()
            if pending is None:
                raise ActionRecoveryError('There is no pending action to wait for')
            action_id = pending['action_id']
            self.journal.append('action_wait_started', {'action_id': action_id,
                'wait_id': wait_id, 'max_status_queries': limit, 'timeout_s': timeout})
        queries = 0
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            reason = ('status_query_limit' if queries >= limit else
                      'wall_deadline' if remaining is not None and remaining <= 0 else None)
            if reason is not None:
                with self.journal.transaction():
                    _, pending = self._state()
                    if pending is None or pending['action_id'] != action_id:
                        raise ActionRecoveryError('Pending action changed during recovery polling')
                    self.journal.append('action_wait_exhausted', {'action_id': action_id,
                        'wait_id': wait_id, 'max_status_queries': limit,
                        'status_queries': queries, 'reason': reason})
                raise BenchHTTPError('Action recovery polling budget exhausted',
                                     category='request_budget_exhausted')
            previous = getattr(self.bench, 'deadline', None)
            self.bench.deadline = deadline if previous is None else (previous if deadline is None else min(previous, deadline))
            try:
                queries += 1
                settled = self.reconcile(expected_action_id=action_id)
            finally:
                self.bench.deadline = previous
            if settled is not None:
                return settled
            # Do not spend another polling interval once the query count is used.
            if queries >= limit:
                continue
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                continue
            time.sleep(interval if remaining is None else min(interval, remaining))

    def cancel(self):
        """Persist cancellation intent before one POST; query to prove stopping."""
        with self.journal.transaction():
            _, pending = self._state()
            if pending is None:
                raise ActionRecoveryError('There is no pending action to cancel')
            if pending['cancellation_requested']:
                raise ActionPendingError('Cancellation already requested; query the original action')
            action_id = pending['action_id']
            self.journal.append('action_cancel_requested', {'action_id': action_id})
            try:
                receipt = self.request(self.bench.cancel_action, self.episode_id, action_id)
                self._receipt(receipt, action_id)
            except Exception as error:
                self._error(action_id, 'cancel', error)
                raise
            self.journal.append('action_cancel_receipt', {'action_id': action_id, 'receipt': receipt})
            return deepcopy(receipt)
