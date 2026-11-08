"""Host bridge for model-authored bounded skills and immutable library versions.

Generated syntax never receives this object, the library, the FunctionSession,
or any Python callable. The interpreter receives JSON and a host-only callback.
All physical effects go through the existing FunctionSession / RobotAPI ledger.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bench.protocol.functions import ACTION_FUNCTIONS, FUNCTION_INPUTS
from ..bench_http import BenchHTTPError
from ..function_session import FunctionSession
from ..time_limits import minimum_limit
from ..validation_feedback import controlled_validation_details
from .journal import Journal, JournalError, atomic_write_json, canonical_json, checked_directory, read_json, strict_json_loads
from .skills import SkillLibrary, SkillLibraryError
from .skill_language import (LANGUAGE, SkillLanguageError, SkillLanguageLimits, SkillStop,
                             bounded_json_copy, execute_source, validate_source)


PROTOCOL = 'agent_skill_runtime.v1'
Token = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
EventId = Annotated[str, Field(min_length=1, max_length=160, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


@dataclass(frozen=True)
class SkillRuntimeLimits:
    language: SkillLanguageLimits = field(default_factory=SkillLanguageLimits)
    max_tool_requests: int = 1024
    max_revision_attempts: int = 64
    max_publication_attempts: int = 64
    max_execution_attempts: int = 128
    max_api_calls: int = 1024
    max_action_attempts: int = 256
    max_written_source_bytes: int = 2 * 1024 * 1024
    max_ledger_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.language, SkillLanguageLimits): raise ValueError('language limits required')
        for name,value in asdict(self).items():
            if name != 'language' and (type(value) is not int or value < 1):
                raise ValueError('Runtime limits must be positive integers')
        if self.language.max_runtime_s is None:
            raise ValueError('Skills require a finite per-execution wall limit')


class _Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class StatusArguments(_Arguments): pass


class WriteArguments(_Arguments):
    skill_id: Token
    source: Annotated[str, Field(min_length=1, max_length=65536)]
    description: Annotated[str, Field(min_length=1, max_length=2048)]
    expected_revision: Digest


class ReadArguments(_Arguments):
    skill_id: Token
    candidate_event_id: EventId | None = None


class PublishArguments(_Arguments):
    expected_revision: Digest
    transaction_id: Token
    reason: Annotated[str, Field(min_length=1, max_length=2048)]


class ExecuteArguments(ReadArguments):
    args: dict = Field(default_factory=dict)
    state: dict = Field(default_factory=dict)


class TraceArguments(_Arguments):
    execution_id: Token
    call_index: int = Field(ge=0)


TOOL_INPUTS = {'skill_status': StatusArguments, 'skill_write': WriteArguments,
               'skill_read': ReadArguments, 'skill_publish': PublishArguments,
               'skill_execute': ExecuteArguments, 'skill_trace': TraceArguments}


class SkillRuntimeError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _hash(value): return hashlib.sha256(canonical_json(value)).hexdigest()


class SkillRuntime:
    def __init__(self, library: SkillLibrary, *, session: FunctionSession,
                 phase: Literal['practice', 'validation', 'evaluation'], journal_path: Path,
                 limits: SkillRuntimeLimits | None = None, frozen_candidate_event_id: str | None = None):
        if not isinstance(library, SkillLibrary) or not isinstance(session, FunctionSession):
            raise TypeError('Host-owned SkillLibrary and FunctionSession are required')
        if phase not in ('practice','validation','evaluation'): raise ValueError('Unknown learning phase')
        if phase != 'practice' and frozen_candidate_event_id is None:
            raise ValueError('Validation/evaluation require a fixed published candidate')
        if phase == 'practice' and frozen_candidate_event_id is not None:
            raise ValueError('Practice candidates are explicit per execution, not frozen overrides')
        self.library, self.session, self.phase = library, session, phase
        self.limits = limits or SkillRuntimeLimits()
        self.frozen_candidate_event_id = frozen_candidate_event_id
        self.frozen_bundle_hash = None
        if frozen_candidate_event_id is not None:
            self.frozen_bundle_hash = self._candidate(frozen_candidate_event_id)[0]['bundle_hash']
        self.journal = Journal(journal_path)
        self.records = checked_directory(Path(journal_path).parent/'skill_execution_records',create=True)
        self.counts = dict.fromkeys(('tool_requests','revision_attempts','publication_attempts',
            'execution_attempts','api_calls','action_attempts','written_source_bytes','ledger_bytes'),0)
        self._completed_calls = {}
        self._active = False
        self.binding = {'protocol': PROTOCOL, 'language': LANGUAGE, 'phase': phase,
            'episode_id': session.episode_id, 'trial_id': session.trial_id,
            'player_id': session.binding['player_id'] if session.binding else 'single',
            'library_experiment_sha256': _hash(library.experiment),
            'frozen_candidate_event_id': frozen_candidate_event_id,
            'frozen_bundle_hash': self.frozen_bundle_hash, 'limits': asdict(self.limits)}
        with self.journal.transaction():
            if self.journal.read():
                raise ValueError('A skill runtime cannot reopen a trial or reset its budgets')
            self.journal.append('skill_runtime_bound',self.binding)

    def _candidate(self, identifier):
        # Verification checks complete frozen contents and provenance. No author
        # file is consulted when reading or executing this candidate.
        with self.library.journal.transaction():
            event = self.library._candidate(identifier)
            metadata = event['payload']
            manifest = self.library.verify(metadata['bundle_hash'])
            return {**deepcopy(metadata),'candidate_event_id':identifier},manifest

    def _pin(self, identifier, *, required=False):
        if self.phase != 'practice':
            if identifier not in (None,self.frozen_candidate_event_id):
                raise SkillRuntimeError('frozen_candidate_mismatch','Only the host-pinned candidate is available')
            identifier = self.frozen_candidate_event_id
        if required and identifier is None:
            raise SkillRuntimeError('published_candidate_required','Execute an explicitly published candidate')
        return identifier

    def _source(self, skill_id, candidate):
        if candidate is None:
            catalog = strict_json_loads(self.library.read_author('catalog.json'))
            metadata = None
        else:
            metadata,manifest = self._candidate(candidate)
            if self.phase != 'practice' and metadata['bundle_hash'] != self.frozen_bundle_hash:
                raise SkillRuntimeError('frozen_candidate_mismatch','Frozen bundle identity changed')
            catalog = manifest['catalog']
        entries = [row for row in catalog['skills'] if row['skill_id'] == skill_id]
        if len(entries) != 1: raise SkillRuntimeError('skill_not_found','Skill ID is unavailable in this version')
        entry = entries[0]
        if entry['entrypoint'] != 'run': raise SkillRuntimeError('unsupported_entrypoint','This runtime requires run(api,args,state)')
        raw = (self.library.read_author(entry['module']) if candidate is None
               else self.library.read_bundle_file(candidate,entry['module']))
        if candidate is not None:
            expected = next(row['sha256'] for row in manifest['files'] if row['path'] == entry['module'])
            if hashlib.sha256(raw).hexdigest() != expected: raise SkillLibraryError('source identity changed')
        source = raw.decode('utf8')
        validate_source(source,self.limits.language)
        return source,deepcopy(entry),metadata

    def _take(self, counter, amount=1):
        bound = getattr(self.limits,'max_'+counter)
        if self.counts[counter] + amount > bound:
            raise SkillRuntimeError('budget_exhausted','The configured skill runtime budget is exhausted')
        self.counts[counter] += amount

    def public_metadata(self):
        return {**deepcopy(self.binding),'counts':deepcopy(self.counts),
            'writes_allowed': self.phase == 'practice', 'arbitrary_python':False,
            'runtime_isolation': 'interpreted_json_capabilities_no_python_host_objects',
            'os_process_isolation':False,
            'budget_scope':'This runtime instance; FunctionSession and host stage budgets also apply.',
            'ledger_budget_scope':'max_ledger_bytes bounds canonical completed API records, with a full-result reservation before each dispatch. Library files and other audit metadata have their separate source/value/count limits; this is not an OS disk quota.',
            'scope':'Model-authored bounded source, not an imported manipulation policy or task qualification.',
            'syntax': 'def run(api,args,state); JSON values, local helper functions, return, if, for/while, break/continue, arithmetic and finite math. api.call(name,arguments) and api.intent(intent) return ordinary public feedback. No imports, attributes, files, network, eval/exec or host objects.',
            'pure_functions':['abs','min','max','len','sum','sqrt','sin','cos','tan','asin','acos','atan2','floor','ceil','get','keys'],
            'language_spec':{
                'entrypoint':'Exactly def run(api, args, state), with no decorators, annotations or default arguments.',
                'source':'UTF-8 source, not a Markdown code fence. Only top-level local function definitions and docstrings.',
                'values':'JSON null/bool/finite numbers/strings/lists/dicts. Python tuple syntax produces a JSON list. No Python objects.',
                'statements':'Single assignment, indexed dict/list assignment, += -= *= /=, if/else, for over range/list/dict keys, bounded while, return, break, continue, pass.',
                'expressions':'Indexing; + - * / // %; numeric unary + -; not/and/or; comparisons and in/not in; conditional expression; positional calls to local helpers and advertised pure functions.',
                'containers':'Use get(dict,key,default) and keys(dict); object methods such as dict.get/list.append are unavailable. List concatenation is supported and bounded.',
                'api':'api.call(function_name, arguments_dict) or api.intent(intent_dict). Only currently advertised ordinary public functions and enabled geometry math; no recursive skill/file tools.',
                'feedback':'The complete ordinary call feedback is returned. ok means function-call acceptance, not physical objective or task success. Inspect result.status/diagnostic/contact_history as appropriate.',
                'stopping':'Any operation epoch change, including same-player reveal, stops this invocation after its actual feedback. Pending actions remain pending and are never resubmitted.',
                'state':'args and state are bounded JSON copies for this invocation. Return updated JSON explicitly; no Python stack/globals or cross-episode state is persisted automatically.',
                'example':'def run(api, args, state):\n    reply = api.call("observe", {})\n    return {"feedback": reply, "state": state}\n'}}

    def tools(self):
        names = [name for name in TOOL_INPUTS if self.phase == 'practice' or name not in ('skill_write','skill_publish')]
        return [{'function':name,'arguments_schema':TOOL_INPUTS[name].model_json_schema(),
                 'scope': ('Own model-authored skill library and public function execution only; '
                           'skill execution status is not task success. Validation and selection remain host-only.')}
                for name in names]

    def execute_tool(self, name, arguments):
        try:
            if self._active: raise SkillRuntimeError('reentrant_runtime','Nested runtime tool dispatch is unavailable')
            self._take('tool_requests')
            if name not in TOOL_INPUTS: raise SkillRuntimeError('unsupported_function','Unknown skill function')
            parsed = TOOL_INPUTS[name].model_validate(arguments).model_dump(mode='json')
            # Copy through the same finite, bounded JSON boundary as skill input.
            try: parsed = bounded_json_copy(parsed,self.limits.language)
            except SkillLanguageError: raise SkillRuntimeError('invalid_arguments','Skill arguments exceed the finite JSON limits') from None
            self.journal.append('skill_tool_requested',{'function':name,'arguments':parsed})
            if name in ('skill_write','skill_publish') and self.phase != 'practice':
                raise SkillRuntimeError('phase_read_only','Skill writing and publishing require practice')
            self._active = True
            try:
                result = getattr(self,'_'+name)(**parsed)
            finally: self._active = False
            feedback = {'ok':True,'function':name,'result':result}
        except ValidationError as error:
            feedback = {'ok':False,'function':name,'error':{'code':'invalid_arguments',
                'message':'Arguments do not satisfy the skill tool schema',
                **controlled_validation_details(error,TOOL_INPUTS[name].model_json_schema())}}
        except SkillLanguageError as error:
            feedback = {'ok':False,'function':name,'error':{'code':'skill_source_rejected',
                'message':'Source does not satisfy the bounded skill language','language_error':error.code,'line':error.line}}
        except SkillRuntimeError as error:
            feedback = {'ok':False,'function':name,'error':{'code':error.code,'message':str(error)}}
        except (JournalError,OSError,UnicodeError,ValueError,TypeError):
            feedback = {'ok':False,'function':name,'error':{'code':'skill_storage_error',
                'message':'Skill library or runtime evidence could not be verified'}}
        self.journal.append('skill_tool_finished',{'function':name,'feedback_sha256':_hash(feedback),
            'ok':feedback['ok'],'counts':deepcopy(self.counts)})
        return feedback

    def _skill_status(self):
        result = self.public_metadata()
        if self.phase == 'practice':
            result['author_revision'] = self.library.author_revision()
            result['catalog'] = strict_json_loads(self.library.read_author('catalog.json'))
        else:
            result['catalog'] = self._candidate(self.frozen_candidate_event_id)[1]['catalog']
        return result

    def _skill_write(self, skill_id, source, description, expected_revision):
        self._take('revision_attempts')
        validate_source(source,self.limits.language)
        self._take('written_source_bytes',len(source.encode()))
        with self.library.journal.transaction():
            if self.library.author_revision() != expected_revision:
                raise SkillRuntimeError('stale_author_revision','Read the current revision before writing')
            catalog = strict_json_loads(self.library.read_author('catalog.json'))
            entry = {'skill_id':skill_id,'module':f'skills/{skill_id}.py','entrypoint':'run',
                     'description':description,'tags':[LANGUAGE],
                     'input_schema':{'type':'object'},'output_schema':{}}
            rows = [row for row in catalog['skills'] if row['skill_id'] != skill_id] + [entry]
            rows.sort(key=lambda row:row['skill_id'])
            revision = self.library.write_author(entry['module'],source,expected_revision=expected_revision)
            revision = self.library.write_author('catalog.json',canonical_json({'skills':rows}),expected_revision=revision)
        return {'skill_id':skill_id,'author_revision':revision,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'published':False}

    def _skill_read(self, skill_id, candidate_event_id=None):
        candidate = self._pin(candidate_event_id)
        source,entry,metadata = self._source(skill_id,candidate)
        return {'skill':entry,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'candidate_event_id':candidate,'bundle_hash':metadata['bundle_hash'] if metadata else None,
                'author_revision':self.library.author_revision() if candidate is None else None}

    def _skill_publish(self, expected_revision, transaction_id, reason):
        self._take('publication_attempts')
        with self.library.journal.transaction():
            prepared = any(row['kind']=='publication_prepared' and row['transaction_id']==transaction_id
                           for row in self.library.journal.read())
            if not prepared:
                if self.library.author_revision() != expected_revision:
                    raise SkillRuntimeError('stale_author_revision','Read the current revision before publishing')
                catalog = strict_json_loads(self.library.read_author('catalog.json'))
                for row in catalog['skills']: self._source(row['skill_id'],None)
            event = self.library.publish(expected_revision,transaction_id=transaction_id,reason=reason)
        return {'candidate_event_id':event['event_id'],'bundle_hash':event['payload']['bundle_hash'],
                'author_revision':expected_revision,'validated':False,'selected':False}

    def _epoch(self):
        state = self.session.state
        for key in ('board_game','turn_game'):
            game = state.get(key) if isinstance(state,dict) else None
            if isinstance(game,dict):
                return {'turn_index':game.get('turn_index'),'current_player':game.get('current_player'),
                        'terminal':game.get('terminal',False)}
        return {'turn_index':None,'current_player':None,
                'terminal':bool(state.get('episode_finished',False)) if isinstance(state,dict) else False}

    def _skill_execute(self, skill_id, candidate_event_id=None, args=None, state=None):
        self._take('execution_attempts')
        candidate = self._pin(candidate_event_id,required=True)
        source,entry,metadata = self._source(skill_id,candidate)
        execution_id = uuid4().hex
        directory = checked_directory(self.records/execution_id,create=True)
        calls = []
        latest_feedback = None
        transport_error = None
        language_entered = False
        started = time.monotonic()
        previous_deadline = self.session.deadline
        skill_deadline = started+self.limits.language.max_runtime_s
        deadline = minimum_limit(previous_deadline,skill_deadline)
        binding = {'execution_id':execution_id,'skill_id':skill_id,'candidate_event_id':candidate,
            'bundle_hash':metadata['bundle_hash'],'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'args':args,'state':state,'limits':asdict(self.limits.language)}
        atomic_write_json(directory/'input.json',binding,replace=False)
        self.journal.append('skill_execution_started',binding)
        self.session.deadline = deadline
        origin = None

        def transport_stop(detail):
            nonlocal transport_error
            transport_error = deepcopy(detail)
            # A timed-out wait never proves an accepted action has stopped.
            if self.session.pending is not None:
                return 'action_pending'
            if (detail.get('error_category') in ('request_budget_exhausted','transport_timeout')
                    and time.monotonic() >= deadline):
                return ('episode_wall_deadline' if previous_deadline is not None
                        and previous_deadline <= skill_deadline else 'skill_wall_deadline')
            return 'public_transport_failure'

        try:
            self.session.refresh()
            origin = self._epoch()
            if self.session.pending is not None: raise SkillStop('action_pending')
            if origin['terminal']: raise SkillStop('episode_terminal')
            if self.session.binding and origin['current_player'] != self.session.binding['player_id']:
                raise SkillStop('not_current_player')

            def invoke(method, arguments):
                nonlocal latest_feedback
                try:
                    self.session.remaining()
                    self.session.refresh()
                except BenchHTTPError as error:
                    detail = error.public_metadata()
                    raise SkillStop(transport_stop(detail),{'ok':False,'error':detail}) from None
                if self._epoch() != origin: raise SkillStop('authorization_epoch_changed')
                if self.session.pending is not None: raise SkillStop('action_pending')
                if method == 'call':
                    name,parameters = arguments
                    if type(name) is not str or type(parameters) is not dict:
                        raise SkillStop('invalid_public_call')
                    # No recursive skill tool, filesystem recall or unregistered
                    # callback can be reached through a generated name.
                    allowed = set(FUNCTION_INPUTS) | (set(self.session.local_functions) & {
                        'compute_rotation','compute_vector','compute_pose_delta','compute_transform_points','compute_frame_alignment'})
                    if name not in allowed: raise SkillStop('unavailable_public_function')
                    decision = {'function':name,'arguments':parameters}
                elif method == 'intent':
                    if type(arguments[0]) is not dict: raise SkillStop('invalid_public_intent')
                    name,decision = 'intent',{'intent':arguments[0]}
                else: raise SkillStop('unavailable_api_method')
                reserve = self.limits.language.max_value_bytes+len(canonical_json(decision))
                if self.counts['api_calls'] >= self.limits.max_api_calls:
                    raise SkillStop('runtime_api_budget')
                if name in ACTION_FUNCTIONS and (
                        self.counts['action_attempts'] >= self.limits.max_action_attempts
                        or sum(row['is_action'] for row in calls) >= self.limits.language.max_move_calls):
                    raise SkillStop('execution_action_budget')
                if self.counts['ledger_bytes'] + reserve > self.limits.max_ledger_bytes:
                    raise SkillStop('runtime_ledger_budget')
                try:
                    self._take('api_calls')
                    if name in ACTION_FUNCTIONS:
                        self._take('action_attempts')
                    # Reserve sufficient ledger space before every physical effect.
                    self._take('ledger_bytes',reserve)
                except SkillRuntimeError:
                    raise SkillStop('runtime_api_budget') from None
                index = len(calls)
                call_binding = {'execution_id':execution_id,'call_index':index,
                    'decision':decision,'authorization_epoch':deepcopy(origin)}
                self.journal.append('skill_api_prepared',call_binding)
                row = {'call_index':index,'function':name,'is_action':name in ACTION_FUNCTIONS,
                       'status':'started'}
                calls.append(row)
                feedback = self.session.execute(decision)
                latest_feedback = deepcopy(feedback)
                record = {**call_binding,'feedback':feedback,'after_epoch':self._epoch()}
                # Complete actual feedback survives interpreter/output budget
                # faults after the public operation has already happened.
                atomic_write_json(directory/f'call_{index:06d}.json',record,replace=False)
                digest = _hash(record)
                self.counts['ledger_bytes'] += len(canonical_json(record)) - reserve
                self._completed_calls[(execution_id,index)] = digest
                result = feedback.get('result') or {}
                row.update(status='completed',record_sha256=digest,ok=feedback.get('ok'),
                    primitive_status=result.get('status') if isinstance(result,dict) else None,
                    termination_reason=result.get('termination_reason') if isinstance(result,dict) else None,
                    error=deepcopy(feedback.get('error')))
                self.journal.append('skill_api_completed',{'execution_id':execution_id,
                    'call_index':index,'record_sha256':digest,'summary':deepcopy(row)})
                if self.counts['ledger_bytes'] > self.limits.max_ledger_bytes:
                    raise SkillStop('runtime_ledger_budget',feedback)
                if self._epoch() != origin: raise SkillStop('authorization_epoch_changed',feedback)
                if self.session.pending is not None: raise SkillStop('action_pending',feedback)
                if ((feedback.get('context') or {}).get('episode_finished') is True
                        or isinstance(result,dict) and result.get('episode_finished') is True
                        or (feedback.get('error') or {}).get('code') == 'episode_finished'):
                    raise SkillStop('episode_terminal',feedback)
                if (feedback.get('error') or {}).get('code') == 'infrastructure_error' or feedback.get('refresh_error'):
                    detail = ((feedback.get('error') or {}) if (feedback.get('error') or {}).get('code')
                              == 'infrastructure_error' else feedback['refresh_error'])
                    raise SkillStop(transport_stop(detail),feedback)
                return feedback

            language_entered = True
            result = execute_source(source,args or {},state or {},invoke,limits=self.limits.language)
        except SkillStop as stop:
            result = {'language':LANGUAGE,'status':'stopped','stop_reason':stop.reason,
                'last_feedback':stop.feedback,'error':None,'return_value':None,'api_calls':len(calls)}
        except BenchHTTPError as error:
            detail = error.public_metadata()
            result = {'language':LANGUAGE,'status':'stopped','stop_reason':transport_stop(detail),
                'last_feedback':None,'error':detail,'return_value':None,'api_calls':len(calls)}
        finally: self.session.deadline = previous_deadline
        if transport_error is not None:
            # Keep a failed preflight separately from the last completed call.
            result['error'] = transport_error
        if latest_feedback is not None:
            # A language value/fuel fault must not hide the latest real effect.
            result['last_feedback'] = latest_feedback
        result.update(execution_id=execution_id,skill_id=skill_id,candidate_event_id=candidate,
            bundle_hash=metadata['bundle_hash'],source_sha256=binding['source_sha256'],
            calls=calls,elapsed_wall_s=time.monotonic()-started,authorization_epoch=origin,
            language_entered=language_entered,
            physical_action_pending=self.session.pending is not None,
            scope='Skill program execution, not physical task success or a referee outcome.')
        atomic_write_json(directory/'result.json',result,replace=False)
        self.journal.append('skill_execution_finished',{'execution_id':execution_id,
            'status':result['status'],'language_entered':language_entered,
            'result_sha256':_hash(result),'counts':deepcopy(self.counts)})
        return result

    def _skill_trace(self, execution_id, call_index):
        expected = self._completed_calls.get((execution_id,call_index))
        if expected is None: raise SkillRuntimeError('trace_unavailable','Only completed calls from this runtime are available')
        record = read_json(self.records/execution_id/f'call_{call_index:06d}.json')
        if _hash(record) != expected: raise SkillRuntimeError('trace_integrity','Recorded public call identity changed')
        return {'record_sha256':expected,'record':record,'historical':True}
