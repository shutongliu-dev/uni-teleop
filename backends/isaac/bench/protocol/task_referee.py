"""Opt-in static physical-task rules and allowlisted coarse adjudication."""
from collections.abc import Mapping
from typing import Annotated, Literal
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_serializer, model_validator
from .schemas import Outcome
from .task_scoring import TerminalScoringContract, TerminalTaskScoring, SCORABLE_OUTCOMES

Code = Annotated[StrictStr, Field(pattern=r'^[a-z][a-z0-9_]*$', min_length=1, max_length=100)]
Text = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
Number = Annotated[StrictFloat | StrictInt, Field(allow_inf_nan=False)]
State = Literal['ongoing', 'success', 'task_failure', 'evaluator_invalid', 'safety_stop']


class PublicObjectProperties(BaseModel):
    """Declared, static apparatus properties; never live pose or load estimates."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: Code
    label: Text
    nominal_mass_kg: Annotated[StrictFloat | StrictInt, Field(gt=0, allow_inf_nan=False)]
    basis: Text


class PublicContract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)
    schema_version: Literal['physical_task_referee_contract.v1'] = Field(alias='schema')
    task_id: Annotated[StrictStr, Field(pattern=r'^[a-z0-9][a-z0-9_]*$', max_length=100)]
    instruction_zh: Text
    phases: Annotated[dict[Code, Text], Field(min_length=1, max_length=32)]
    reason_descriptions: Annotated[dict[Code, Text], Field(min_length=1, max_length=128)]
    success_conditions: Annotated[list[Text], Field(min_length=1, max_length=32)]
    recovery: Annotated[list[Text], Field(min_length=1, max_length=32)]
    thresholds: Annotated[dict[Code, Number], Field(min_length=1, max_length=128)]
    outcomes: dict[Literal['ongoing', 'success', 'task_failure', 'evaluator_invalid',
                           'safety_stop', 'budget_exhaustion', 'cancelled'], Text]
    physical_actions: Text
    observation_scope: Text
    object_properties: Annotated[list[PublicObjectProperties], Field(min_length=1, max_length=128)] | None = None

    @model_serializer(mode='wrap')
    def omit_undeclared_object_properties(self, handler):
        result = handler(self)
        if result.get('object_properties') is None:
            result.pop('object_properties', None)
        return result

    @model_validator(mode='after')
    def all_outcomes(self):
        if set(self.outcomes) != {'ongoing', 'success', 'task_failure', 'evaluator_invalid',
                                 'safety_stop', 'budget_exhaustion', 'cancelled'}:
            raise ValueError('Every task outcome must be declared')
        if self.object_properties is not None:
            ids = [row.id for row in self.object_properties]
            if len(ids) != len(set(ids)):
                raise ValueError('Public object property identifiers must be unique')
        return self


class PublicRefereeState(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)
    schema_version: Literal['physical_task_referee_state.v1'] = Field(alias='schema')
    state: State
    phase: Code
    reason_codes: Annotated[list[Code], Field(min_length=1, max_length=128)]
    terminal: StrictBool
    recoverable: StrictBool

    @model_validator(mode='after')
    def consistent_terminal(self):
        if (self.terminal != (self.state != 'ongoing') or self.recoverable != (self.state == 'ongoing')
                or len(set(self.reason_codes)) != len(self.reason_codes)):
            raise ValueError('Referee state has inconsistent terminal/recovery semantics')
        return self


class PublicEpisodeState(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    terminal: StrictBool
    outcome: Outcome | None

    @model_validator(mode='after')
    def consistent_outcome(self):
        if self.terminal != (self.outcome is not None):
            raise ValueError('Episode terminal status and outcome disagree')
        return self


class PublicTaskRequirements(BaseModel):
    """Typed result for the opt-in, read-only get_task function."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    task_id: Annotated[StrictStr, Field(pattern=r'^[a-z0-9][a-z0-9_]*$', max_length=100)]
    task_kind: Literal['manipulation']
    create_endpoint: Literal['POST /episodes']
    instruction: Text
    public_rules: PublicContract
    public_rules_sha256: Annotated[StrictStr, Field(pattern=r'^[0-9a-f]{64}$')]
    referee: PublicRefereeState
    episode: PublicEpisodeState
    scoring_contract: TerminalScoringContract | None = None
    scoring: TerminalTaskScoring | None = None

    @model_serializer(mode='wrap')
    def omit_undeclared_scoring(self, handler):
        result = handler(self)
        for key in ('scoring_contract', 'scoring'):
            if result.get(key) is None:
                result.pop(key, None)
        return result

    @model_validator(mode='after')
    def consistent_task(self):
        rules = self.public_rules.model_dump(mode='json', by_alias=True)
        if (self.task_id != rules['task_id'] or self.instruction != rules['instruction_zh']
                or self.public_rules_sha256 != contract_sha256(rules)
                or self.referee.phase not in rules['phases']
                or not set(self.referee.reason_codes) <= rules['reason_descriptions'].keys()):
            raise ValueError('Public task result differs from its rule contract')
        if self.scoring is not None:
            if not self.episode.terminal or self.scoring_contract is None:
                raise ValueError('Numeric scoring requires a declared, terminal episode')
            if not self.scoring.matches(self.scoring_contract):
                raise ValueError('Terminal scoring differs from the declared contract')
            if self.scoring.score_valid and (self.episode.outcome not in SCORABLE_OUTCOMES
                    or self.referee.state in ('evaluator_invalid', 'safety_stop')):
                raise ValueError('Invalid physical evidence cannot certify a numeric score')
        return self


def contract_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def project_task_rules(task, *, scoring_contract=None):
    """Old tasks remain unchanged; opt-in rules must match both frozen hashes."""
    rules = task.get('public_rules')
    if not isinstance(rules, Mapping) or 'referee' not in rules:
        return None
    contract = PublicContract.model_validate(rules['referee']).model_dump(mode='json', by_alias=True)
    if (contract['task_id'] != task.get('id')
            or task.get('public_rules_sha256') != contract_sha256(rules)
            or task.get('public_referee_contract_sha256') != contract_sha256(rules['referee'])
            or task.get('public_rules_file') != 'rules.json'):
        raise ValueError('Public referee contract differs from frozen task identity')
    result = {'task_id': contract['task_id'], 'task_kind': 'manipulation',
            'create_endpoint': 'POST /episodes', 'instruction': contract['instruction_zh'],
            'public_rules': contract, 'public_rules_sha256': contract_sha256(contract)}
    declaration = task.get('terminal_scoring_contract')
    if declaration is not None and scoring_contract is not None and declaration != scoring_contract:
        raise ValueError('Task and registered scoring declarations differ')
    declaration = declaration if declaration is not None else scoring_contract
    if declaration is not None:
        result['scoring_contract'] = TerminalScoringContract.model_validate(
            declaration).model_dump(mode='json', by_alias=True)
    return result


def project_referee_state(score, contract):
    """No raw reason strings, geometry, counters or evidence values are public."""
    state = PublicRefereeState.model_validate(score.get('referee')).model_dump(mode='json', by_alias=True)
    if (state['phase'] not in contract['phases']
            or not set(state['reason_codes']) <= contract['reason_descriptions'].keys()):
        raise ValueError('Referee returned an undeclared phase or reason')
    if type(score.get('success')) is not bool or score['success'] != (state['state'] == 'success'):
        raise ValueError('Referee state and success disagree')
    if score.get('irrecoverable_reason') and state['state'] not in ('task_failure', 'evaluator_invalid', 'safety_stop'):
        raise ValueError('Referee state hides an irreversible violation')
    return state
