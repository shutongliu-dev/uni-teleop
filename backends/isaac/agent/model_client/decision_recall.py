"""Read only this session's immutable, already completed public decisions."""
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .learning.journal import checked_directory, read_bytes, strict_json_loads


MAX_RECALL_BYTES = 1024 * 1024


class ReadDecisionArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    decision_index: int = Field(ge=0)
    sections: list[Literal['observation', 'execution', 'depth_summary']] = Field(min_length=1, max_length=3)


class RecallError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def recall_tool():
    return {'function': 'read_decision', 'arguments_schema': ReadDecisionArguments.model_json_schema(),
            'scope': 'Read only your own completed earlier decision observation/execution/depth_summary. '
                     'No other player, filesystem path, current/future decision, Bench request or physics action. '
                     f'At most {MAX_RECALL_BYTES} original file bytes per call; oversized records return an error, never a partial record.'}


class DecisionArchive:
    def __init__(self, output: Path, *, episode_id: str, player_id: str):
        self.root = checked_directory(Path(output) / 'decisions', create=True)
        self.episode_id, self.player_id = episode_id, player_id
        self._records = {}

    def _bytes(self, index, section):
        # Both components are host-generated from a validated index and enum.
        return read_bytes(self.root / f'{index:04d}' / f'{section}.json')

    def register(self, index):
        if type(index) is not int or index < 0 or index in self._records:
            raise ValueError('Invalid or already registered decision')
        records, values = {}, {}
        for section in ('observation', 'execution', 'depth_summary'):
            data = self._bytes(index, section)
            values[section] = strict_json_loads(data)
            records[section] = {'source': f'decision:{index:04d}/{section}',
                                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        observation, execution = values['observation'], values['execution']
        if observation.get('episode_id') != self.episode_id or type(observation.get('sequence')) is not int:
            raise ValueError('Decision observation belongs to another session')
        result = execution.get('result', {})
        result = result if isinstance(result, dict) else {}
        self._records[index] = {'decision_index': index, 'player_id': self.player_id,
            'observation_sequence': observation['sequence'], 'sim_time_s': observation['sim_time_s'],
            'action_id': result.get('action_id'),
            'execution_observation_sequence': execution.get('observation_sequence'), 'records': records}

    def reference(self, index):
        if index not in self._records:
            raise RecallError('decision_unavailable', 'Only completed earlier decisions in your own session are available')
        return deepcopy(self._records[index])

    def index(self):
        return [self.reference(index) for index in sorted(self._records)]

    def read(self, arguments):
        args = ReadDecisionArguments.model_validate(arguments)
        if len(set(args.sections)) != len(args.sections):
            raise RecallError('invalid_arguments', 'Choose each recall section at most once')
        reference = self.reference(args.decision_index)
        if sum(reference['records'][section]['bytes'] for section in args.sections) > MAX_RECALL_BYTES:
            raise RecallError('record_too_large', 'Requested original records exceed the published recall byte limit')
        records = {}
        for section in args.sections:
            try:
                data = self._bytes(args.decision_index, section)
                if hashlib.sha256(data).hexdigest() != reference['records'][section]['sha256']:
                    raise ValueError('changed')
                value = strict_json_loads(data)
            except (OSError, ValueError):
                raise RecallError('record_unavailable', 'Original public record is missing, changed or not a regular owned file') from None
            records[section] = {**reference['records'][section], 'value': value}
        return {**reference, 'episode_id': self.episode_id, 'records': records,
                'advances_physics': False, 'historical': True,
                'scope': 'Exact original public records at their own observation sequences, not current state.'}
