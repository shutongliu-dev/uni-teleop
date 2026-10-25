"""Declared terminal-only numeric results; evaluator internals stay private."""
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

MetricName = Annotated[StrictStr, Field(pattern=r'^[a-z][a-z0-9_]*$', max_length=100)]
Unit = Annotated[StrictStr, Field(pattern=r'^[A-Za-z0-9_./%*-]+$', min_length=1, max_length=32)]
Number = Annotated[StrictFloat | StrictInt, Field(allow_inf_nan=False)]

# Operator cancellation and ordinary resource exhaustion retain earned scores.
# Faults never certify a valid continuous score, even with a previous good sample.
SCORABLE_OUTCOMES = frozenset(('success', 'task_failure', 'sim_timeout', 'wall_timeout',
                              'action_budget', 'model_budget', 'cancelled'))


class TerminalScoringContract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)
    schema_version: Literal['terminal_task_scoring_contract.v1'] = Field(alias='schema')
    mode: Literal['continuous']
    objective: Literal['maximize', 'minimize']
    score_unit: Unit
    metric_units: Annotated[dict[MetricName, Unit], Field(max_length=32)]


class TerminalTaskScoring(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)
    schema_version: Literal['terminal_task_scoring.v1'] = Field(alias='schema')
    mode: Literal['continuous']
    objective: Literal['maximize', 'minimize']
    score: Number | None
    score_unit: Unit
    score_valid: StrictBool
    metrics: Annotated[dict[MetricName, Number], Field(max_length=32)]
    metric_units: Annotated[dict[MetricName, Unit], Field(max_length=32)]

    @model_validator(mode='after')
    def valid_numeric_receipt(self):
        if not self.metrics.keys() <= self.metric_units.keys():
            raise ValueError('An undeclared metric cannot be published')
        if self.score_valid and (self.score is None or self.metrics.keys() != self.metric_units.keys()):
            raise ValueError('Valid scoring requires a score and all declared metrics')
        return self

    def matches(self, contract):
        declared = TerminalScoringContract.model_validate(contract)
        return all(getattr(self, key) == getattr(declared, key)
                   for key in ('mode', 'objective', 'score_unit', 'metric_units'))


def project_terminal_scoring(score, contract, *, outcome, physics_valid):
    """Export only named numeric fields after the final physical/evidence audit.

    Missing or malformed receipts produce an explicitly invalid result, without
    copying raw errors or private data. Finite prior records remain auditable
    when later evidence or finalization is invalid.
    """
    if contract is None:
        return None
    declared = TerminalScoringContract.model_validate(contract)
    source = score if isinstance(score, Mapping) else {}
    values = {'schema': 'terminal_task_scoring.v1', 'mode': declared.mode,
              'objective': declared.objective, 'score_unit': declared.score_unit,
              'metric_units': declared.metric_units, 'score': None,
              'score_valid': False, 'metrics': {}}
    from pydantic import TypeAdapter, ValidationError
    number = TypeAdapter(Number)
    if source.get('score_unit') == declared.score_unit:
        try:
            values['score'] = number.validate_python(source.get('score'))
        except ValidationError:
            pass
    for name in declared.metric_units:
        try:
            values['metrics'][name] = number.validate_python(source.get(name))
        except ValidationError:
            pass
    referee = source.get('referee')
    referee_valid = (not isinstance(referee, Mapping)
                     or referee.get('state') not in ('evaluator_invalid', 'safety_stop'))
    values['score_valid'] = (
        source.get('score_valid') is True and physics_valid is True and referee_valid
        and outcome in SCORABLE_OUTCOMES
        and source.get('status') not in ('invalid', 'evaluator_invalid', 'safety_stop')
        and values['score'] is not None
        and values['metrics'].keys() == declared.metric_units.keys())
    return TerminalTaskScoring.model_validate(values)
