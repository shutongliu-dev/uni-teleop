"""Audited public HTTP primitives shared by action, tool and code Agents."""
from copy import deepcopy
from collections import Counter
import hashlib
import time

from typing import get_args
from bench.protocol.schemas import TerminationReason

from .actions import ActionValidationError, build_joint_action, parse_model_action, parse_json_object
from .geometry_actions import build_geometry_action, parse_geometry_action
from .tools import parse_tool_call, project_depth_point
from .program import ProgramStop
from .capture import capture, save
from .visibility import observation_view, feedback_view
from .time_limits import minimum_limit
from .bench_http import BenchHTTPError
from .learning.robot_api import ActionRecoveryError, RobotAPI, DEFAULT_MAX_STATUS_QUERIES
from .learning.journal import canonical_json, checked_directory


_UNBOUND_SEQUENCE = object()
PRIMITIVE_ATTEMPT_ACCOUNTING = 'recognized_physical_request_v1'


class InteractionSession:
    def __init__(self, bench, episode_id, capabilities, spec, observation, output, *,
                 control_mode, visual_input, tactile_input, limits, deadline, max_actions, trial_id=None,
                 max_status_queries=DEFAULT_MAX_STATUS_QUERIES, result_validator=None):
        self.bench, self.episode_id = bench, episode_id
        self.capabilities, self.spec, self.observation = capabilities, spec, observation
        self.output = output
        self.control_mode, self.visual_input, self.tactile_input = control_mode, visual_input, tactile_input
        self.limits, self.deadline, self.max_actions = limits, deadline, max_actions
        self.program_deadline = None
        self.submitted = self.completed = self.move_attempts = self.tool_calls = 0
        self.read_calls = self.point_queries = 0
        self.result = self.feedback = self.fatal_error = None
        self.result_validator = result_validator
        self.stop_requested = False
        self.records = []
        self.action_api = None
        self.trial_id = trial_id or episode_id
        self.max_status_queries = max_status_queries
        self.termination_reasons = Counter()
        self.interface_selections = Counter()
        if self.output is not None:
            # Whole-trial resumption is not implemented: counters, invalid
            # attempts and Python state cannot be reconstructed from action ACKs.
            # Claim the episode stream before even a rejected/readonly call can
            # consume a budget, so a new session can never silently reset it.
            self._action_stream()

    def remaining(self):
        deadline = minimum_limit(self.deadline, self.program_deadline)
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ActionValidationError('Episode wall budget expired')
        return remaining

    def http(self, function, *args, **kwargs):
        remaining = self.remaining()
        deadline = None if remaining is None else time.monotonic()+remaining
        previous = getattr(self.bench,'deadline',None)
        self.bench.deadline = minimum_limit(previous, deadline)
        try:
            return function(*args,**kwargs)
        finally:
            self.bench.deadline = previous

    def refresh(self):
        self.remaining()
        raw = self.http(self.bench.get_observation,self.episode_id)
        view = observation_view(raw, visual_input=self.visual_input, tactile_input=self.tactile_input)
        if view['episode_id'] != self.episode_id or view['sequence'] < self.observation['sequence']:
            raise ValueError('Unexpected episode or regressing observation sequence')
        self.observation = raw
        return view

    def context(self, folder, *, expected_sequence=_UNBOUND_SEQUENCE):
        current = self.refresh()
        if expected_sequence is not _UNBOUND_SEQUENCE and (
                type(expected_sequence) is not int or current['sequence'] != expected_sequence):
            raise ActionValidationError('Tool decision uses a stale observation')
        self.remaining()
        return self.http(capture,self.bench,self.episode_id,self.observation,folder,self.output,
                       visual_input=self.visual_input,tactile_input=self.tactile_input)

    def _action_stream(self):
        if self.action_api is None:
            if self.output is None:
                raise ValueError('Physical actions require a durable host output directory')
            host = checked_directory(self.output/'host', create=True)
            journal_path = host/'action_journal'
            try:
                journal_path.mkdir(mode=0o700)
            except FileExistsError:
                raise ActionRecoveryError(
                    'InteractionSession cannot reopen an existing action ledger; '
                    'use RobotAPI only to reconcile the original episode') from None

            def counted_submit(*args, **kwargs):
                # Count only after the durable write and host deadline check.
                self.submitted += 1
                return self.bench.submit(*args, **kwargs)

            def action_request(function, *args, **kwargs):
                target = counted_submit if function == self.bench.submit else function
                try:
                    return self.http(target, *args, **kwargs)
                except ActionValidationError:
                    # These host calls have an unresolved physical intent. A
                    # local deadline cannot certify a stopped/failed action.
                    raise BenchHTTPError('Pending action request exceeded the host budget',
                                         category='request_budget_exhausted') from None
            self.action_api = RobotAPI(self.bench, journal_path,
                episode_id=self.episode_id, trial_id=self.trial_id, request=action_request,
                max_status_queries=self.max_status_queries,
                binding={'spec_sha256': hashlib.sha256(canonical_json(self.spec)).hexdigest(),
                         'control_mode': self.control_mode, 'max_actions': self.max_actions})
        return self.action_api

    def _wait_for_action(self, action_api):
        try:
            return action_api.wait(timeout_s=self.remaining())
        except ActionValidationError:
            # remaining() may expire before the first status query. Keep the
            # unresolved action distinct from an ordinary local budget stop.
            raise BenchHTTPError('Pending action recovery exceeded the host budget',
                                 category='request_budget_exhausted') from None

    def _begin_move_attempt(self, decision, folder, expected_sequence):
        """Charge recognized physical intent, never fabricate a submission."""
        if self.move_attempts >= self.max_actions:
            raise ActionValidationError('Client primitive action-attempt budget exhausted')
        self.move_attempts += 1
        save(folder/'primitive_request.json', {'decision':decision,'attempt':self.move_attempts,
            'expected_sequence':expected_sequence})

    def move(self, decision, folder, action_id, expected_sequence, *, control_mode=None):
        if self.fatal_error is not None:
            raise self.fatal_error
        decision = parse_json_object(decision)
        mode = self.control_mode if control_mode is None else control_mode
        if self.control_mode == 'hybrid':
            if self.spec.get('track') != 'free_tool_agent' or mode not in ('joints', 'fingertip'):
                raise ActionValidationError('Free-track primitive requires an explicit advertised control mode')
            limits = getattr(self.limits, mode)
        else:
            if mode != self.control_mode:
                raise ActionValidationError('A fixed-interface session cannot change control mode')
            limits = self.limits
        parser = parse_geometry_action if mode == 'fingertip' else parse_model_action
        builder = build_geometry_action if mode == 'fingertip' else build_joint_action
        # Unknown/stop selections have no physical intent. Recognized physical
        # requests count even if the selected mode or parameters reject them.
        if decision.get('action') not in ('joint_delta', 'move_geometry', 'hold'):
            parser(decision)  # rejects unknown actions and malformed stop
            if type(expected_sequence) is not int or self.refresh()['sequence'] != expected_sequence:
                raise ActionValidationError('Model decision uses a stale observation')
            self.stop_requested = True
            return {'status':'model_requested_stop'}
        action_api = self._action_stream()
        action_api.assert_idle()
        self._begin_move_attempt(decision, folder, expected_sequence)
        self.refresh()
        parsed = parser(decision)
        command = builder(parsed,self.observation,self.capabilities,self.spec,action_id=action_id,
            expected_sequence=expected_sequence,actions_used=self.submitted,limits=limits)
        if command is None:
            self.stop_requested = True
            return {'status':'model_requested_stop'}
        save(folder/'action.json', command)
        try:
            accepted = action_api.submit(command, self.observation)
        except BenchHTTPError as submission_error:
            # The server may already be moving. Recover only by the original ID;
            # neither a timeout nor 404 permits another submission or a reset.
            try:
                settled = self._wait_for_action(action_api)
            except BenchHTTPError:
                # Recovery may itself exhaust the remaining host budget. Keep
                # the original submission interruption as the primary cause.
                raise submission_error from None
        else:
            save(folder/'accepted.json', accepted)
            settled = self._wait_for_action(action_api)
        feedback = settled['receipt']
        self.observation = settled['observation']
        save(folder/'execution_source.json',feedback)
        feedback = feedback_view(feedback,tactile_input=self.tactile_input)
        save(folder/'execution.json',feedback)
        self.feedback = feedback
        reason=feedback.get('termination_reason')
        reason=reason if type(reason) is str and reason in get_args(TerminationReason) else 'not_recorded'
        self.termination_reasons[reason] += 1
        if feedback['status'] == 'completed':
            self.completed += 1
        result = self.http(self.bench.get_result,self.episode_id)
        if self.result_validator is not None:
            try:
                result = self.result_validator(result)
            except Exception as error:
                self.fatal_error = error
                raise
        self.result = result
        # Result is used only to stop host execution. Hidden evaluator data is
        # never included in the callback returned to generated programs.
        self.refresh()
        return deepcopy(feedback)

    def invoke(self, name, arguments, *, folder, decision_index):
        return self.invoke_call({'tool': name, 'arguments': arguments},
                                folder=folder, decision_index=decision_index)

    def invoke_call(self, call, *, folder, decision_index, expected_sequence=_UNBOUND_SEQUENCE):
        try:
            return self._invoke_one(call,folder=folder,decision_index=decision_index,
                                    expected_sequence=expected_sequence)
        except ProgramStop:
            raise
        except Exception as error:
            if not isinstance(error,ActionValidationError):
                self.fatal_error=error
            raise

    def _invoke_one(self, call, *, folder, decision_index, expected_sequence=_UNBOUND_SEQUENCE):
        if self.fatal_error is not None:
            raise self.fatal_error
        if self.result is not None:
            raise ProgramStop('episode_finished', {'status':'episode_finished'})
        self.remaining()
        index = self.tool_calls
        self.tool_calls += 1
        target = folder/'tools'/f'{index:04d}'
        target.mkdir(parents=True,exist_ok=False)
        name = call.get('tool') if isinstance(call, dict) else None
        arguments = call.get('arguments') if isinstance(call, dict) else None
        record = {'index':index,'tool':name,'arguments':deepcopy(arguments),
                  'observation_sequence_before':self.observation['sequence']}
        if not isinstance(call, dict) or set(call) != {'tool', 'arguments'}:
            record['raw_call'] = deepcopy(call)
        save(target/'request.json',record)
        try:
            try:
                if expected_sequence is not _UNBOUND_SEQUENCE and (
                        type(expected_sequence) is not int or self.refresh()['sequence'] != expected_sequence):
                    raise ActionValidationError('Tool decision uses a stale observation')
                call = parse_tool_call(call)
            except ActionValidationError:
                # The same physical intent must cost one attempt through tools
                # and code, including invalid arguments. Valid calls are charged
                # by move() below, so there is no second charge or bypass flag.
                if name in ('move_geometry', 'hold'):
                    sequence = (self.observation['sequence'] if expected_sequence is _UNBOUND_SEQUENCE
                                else expected_sequence)
                    self._begin_move_attempt(call, target, sequence)
                raise
            name = call['tool']
            args = call['arguments']
            if name == 'stop':
                self.stop_requested = True
                value = {'status':'model_requested_stop','reason':args.get('reason','')}
            elif name in ('move_geometry', 'hold'):
                sequence = (self.refresh()['sequence'] if expected_sequence is _UNBOUND_SEQUENCE
                            else expected_sequence)
                feedback = self.move({'action':name,**args},target,
                    f'model_{decision_index:04d}_tool_{index:04d}',sequence,
                    control_mode='fingertip' if self.control_mode == 'hybrid' else None)
                current = self.context(target/'after')
                value = {'feedback':feedback,'observation':current['observation'],
                         'depth_observation':current['depth_context']}
            elif name == 'observe':
                self.read_calls += 1
                current = self.context(target/'view', expected_sequence=expected_sequence)
                value = {'observation':current['observation'], 'feedback':self.feedback,
                         'depth_observation':current['depth_context']}
            else:
                self.point_queries += 1
                if self.visual_input != 'rgbd':
                    raise ActionValidationError('point_at requires the configured RGB-D view')
                current = self.context(target/'view', expected_sequence=expected_sequence)
                value = project_depth_point(current['camera_rows'],current['observation'],**args)
            self.remaining()
            record.update(status='completed',result=value,
                          observation_sequence_after=self.observation['sequence'])
            save(target/'result.json',record)
            self.records.append(record)
            return value
        except ProgramStop:
            raise
        except Exception as error:
            # Validation errors are model-visible and recoverable. Transport or
            # malformed server data must not become a disguised task failure.
            if not isinstance(error,ActionValidationError):
                self.fatal_error = error
            record.update(status='rejected' if isinstance(error,ActionValidationError) else 'error',
                          error_type=type(error).__name__)
            if isinstance(error, ActionValidationError):
                record['error'] = error.public_metadata()
            save(target/'result.json',record)
            self.records.append(record)
            raise
