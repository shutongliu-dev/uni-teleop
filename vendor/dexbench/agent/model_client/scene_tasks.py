"""Trusted-host scene selection, separate from model-visible task observations.

The source catalog supplies identities, availability and release pins only. We
never load an initial state, evaluator or scene geometry to enrich model input.
The connected bench remains the source of public goals, controls and results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from scenes.workrooms.catalog import list_tasks
from scenes.workrooms.registry import FAMILIES


GAME_ALIASES = {'task1': 'tic_tac_toe_3x3', 'task2': 'connect_four_5x5', 'task3': 'hex7x7'}
GAME_NUMBERS = {value: key for key, value in GAME_ALIASES.items()}


class TaskSetupError(ValueError):
    """Finite, safe setup diagnostics that may be persisted and shown by the CLI."""

    MESSAGES = {
        'unknown_family': 'Unknown workroom family.',
        'unknown_task': 'Task is not uniquely defined in the selected family.',
        'family_required': 'This task number requires --family; bare task1/2/3 retain the card-room aliases.',
        'task_planned': 'Task is planned: a runnable frozen release and native task adapter are required.',
        'session_adapter_unavailable': 'This competitive task has no supported authenticated session adapter; it cannot use ordinary episodes or the card-room board-game contract.',
        'task_not_published': 'Task must be uniquely published by the selected bench service. Configure its matching frozen release.',
        'session_contract_mismatch': 'Published task kind or creation endpoint disagrees with the selected session.',
        'release_mismatch': 'Created episode does not use the catalog-selected frozen task release.',
    }

    def __init__(self, code):
        self.code = code
        super().__init__(self.MESSAGES[code])

    def public_metadata(self):
        return {'code': self.code, 'message': self.MESSAGES[self.code]}


@dataclass(frozen=True)
class TaskSelection:
    family: str | None
    task_id: str
    task_number: int | None = None
    session_kind: str = 'single_agent'
    players: tuple[str, ...] = ('single',)
    status: str | None = None
    source_release: dict | None = None
    adapter_available: bool = True
    pin_release: bool = False
    cannot_run_reason: str | None = None
    robot_requirements: dict | None = None
    validation: dict | None = None

    @property
    def practice_rsi_allowed(self):
        return not (self.family == 'card_room' and self.task_id in ('connect_four_5x5', 'hex7x7'))

    def metadata(self):
        return {'schema': 'agent_scene_task_selection.v1', 'family': self.family,
                'task_id': self.task_id, 'task_number': self.task_number,
                'session_kind': self.session_kind, 'players': list(self.players),
                'status': self.status, 'source_release': deepcopy(self.source_release),
                'release_binding': 'catalog_pinned' if self.pin_release else 'service_selected',
                'adapter_available': self.adapter_available,
                'runnable': self.status != 'planned' and self.adapter_available,
                'availability_scope': 'Catalog/Agent readiness only; the connected service and physical qualification are checked separately.',
                'cannot_run_reason': self.cannot_run_reason,
                'robot_requirements': deepcopy(self.robot_requirements),
                'validation': deepcopy(self.validation),
                'practice_rsi_allowed': self.practice_rsi_allowed,
                'rsi_scheduler_implemented': False}

    def require_runnable(self):
        if self.status == 'planned':
            raise TaskSetupError('task_planned')
        if not self.adapter_available:
            raise TaskSetupError('session_adapter_unavailable')

    def verify_release(self, spec):
        actual = spec.get('task_release_sha256') if isinstance(spec, dict) else None
        expected = self.source_release.get('manifest_sha256') if self.source_release else None
        if self.pin_release and (expected is None or actual != expected):
            raise TaskSetupError('release_mismatch')


def _selection(definition, *, pin_release):
    family, task_id = definition['family'], definition['task_id']
    rules = definition.get('rules', {}).get('contract') or {}
    board = family == 'card_room' and task_id in GAME_NUMBERS
    turn = family == 'living_room' and task_id == 'tower_extraction'
    competitive = turn or rules.get('competitive') is True
    kind = 'board_game' if board else 'two_player_adversarial' if competitive else 'single_agent'
    # Player identities are distinct from robot namespaces (front/rear).
    players = ('player_0', 'player_1') if board or turn else tuple(rules.get('players', ())) if competitive else ('single',)
    # Project metadata, not task objects, reset state, evaluator internals or rules.
    robot = definition.get('robot_requirements', {})
    validation = deepcopy(definition.get('validation'))
    if validation is not None and family == 'mechanical_assembly':
        source_release = definition.get('source_release') or {}
        candidate_release = definition.get('candidate_referee', {}).get('release', {})
        loose_candidate = definition.get('design', {}).get('loose_bolt_candidate', {}).get('release', {})
        release = source_release or candidate_release
        validation.update({
            'release_sha256': release.get('manifest_sha256'),
            'release_status': definition.get('qualification', {}).get('release_status') or
                ('planned' if definition['status'] == 'planned' else 'unreported'),
            'privileged_entrypoint_required': True,
        })
        candidate = loose_candidate or candidate_release
        if candidate:
            validation['candidate_release'] = deepcopy(candidate)
    return TaskSelection(family=family, task_id=task_id,
        task_number=definition.get('task_number', definition.get('difficulty')),
        session_kind=kind, players=players, status=definition['status'],
        source_release=deepcopy(definition.get('source_release')), pin_release=pin_release,
        adapter_available=not competitive or turn, cannot_run_reason=(definition.get('cannot_run_reason') or
            (TaskSetupError.MESSAGES['session_adapter_unavailable'] if competitive and not turn else None)),
        robot_requirements={key: deepcopy(robot[key]) for key in
            ('hand_id', 'articulations', 'active_sides', 'assembly_version') if key in robot},
        validation=validation)


def select_task(name, family=None):
    """Resolve family-local numbering without assigning card semantics to task2/3.

    Explicit families select the catalog release. Bare canonical IDs preserve
    service-selected historical releases; known planned/competitive tasks still
    cannot bypass their readiness checks by dropping --family.
    """
    if family is not None and family not in FAMILIES:
        raise TaskSetupError('unknown_family')
    if not isinstance(name, str) or not name or len(name) > 160:
        raise TaskSetupError('unknown_task')
    selected = GAME_ALIASES.get(name, name) if family is None else name
    definitions = list_tasks(family, include_planned=True)
    matches = [row for row in definitions if row['task_id'] == selected]
    if not matches and family is not None:
        number = re.fullmatch(r'task(\d+)', name)
        if family == 'mechanical_assembly':
            number = number or re.fullmatch(r'L(\d+)', name)
        if number:
            matches = [row for row in definitions
                       if row.get('task_number', row.get('difficulty')) == int(number[1])]
    if len(matches) == 1:
        return _selection(matches[0], pin_release=family is not None)
    if matches or family is not None:
        raise TaskSetupError('unknown_task')
    if re.fullmatch(r'(task|L)\d+', name):
        raise TaskSetupError('family_required')
    # Other published benchmark tasks keep their existing service-only path.
    return TaskSelection(family=None, task_id=selected)


def published_task(selection, tasks):
    selection.require_runnable()
    if not isinstance(tasks, list):
        raise TaskSetupError('task_not_published')
    matches = [row for row in tasks if isinstance(row, dict) and row.get('task_id') == selection.task_id]
    if len(matches) != 1:
        raise TaskSetupError('task_not_published')
    task = deepcopy(matches[0])
    if task.get('status') == 'planned' or task.get('runnable') is False or task.get('native_execution_available') is False:
        raise TaskSetupError('task_planned')
    published_hash = task.get('task_release_sha256')
    if (selection.pin_release and published_hash is not None
            and (not selection.source_release
                 or published_hash != selection.source_release.get('manifest_sha256'))):
        raise TaskSetupError('release_mismatch')
    kind = task.get('task_kind')
    board = kind == 'board_game'
    turn = kind == 'turn_game'
    expected_turn = selection.family == 'living_room' and selection.task_id == 'tower_extraction'
    if (kind not in (None, 'manipulation', 'single_agent', 'board_game', 'turn_game')
            or (task.get('competitive') is True and not (board or turn))
            or turn != expected_turn
            or (selection.family is not None and board != (selection.session_kind == 'board_game'))):
        raise TaskSetupError('session_contract_mismatch')
    expected = 'POST /turn-games' if turn else 'POST /board-games' if board else 'POST /episodes'
    if task.get('create_endpoint', expected) != expected:
        raise TaskSetupError('session_contract_mismatch')
    return task, board


def list_scene_tasks(family):
    if family not in FAMILIES:
        raise TaskSetupError('unknown_family')
    return [_selection(row, pin_release=True).metadata() for row in list_tasks(family, include_planned=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    args = parser.parse_args()
    print(json.dumps(list_scene_tasks(args.family), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
