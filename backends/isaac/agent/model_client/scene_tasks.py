"""Trusted-host scene selection, separate from model-visible task observations.

The source catalog supplies identities, availability and release pins only. We
never load an initial state, evaluator or scene geometry to enrich model input.
The connected bench remains the source of public goals, controls and results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import re

from scenes.workrooms.catalog import CURRENT_FAMILIES as FAMILIES, list_tasks
from bench.runtime.task_capabilities import engineering_capabilities, robot_episode_unavailable_reason
from .card_games import CARD_PROTOCOL, CARD_TASKS


GAME_ALIASES = {'task1': 'tic_tac_toe_3x3', 'task2': 'connect_four_5x5', 'task3': 'hex7x7'}
GAME_NUMBERS = {value: key for key, value in GAME_ALIASES.items()}


class TaskSetupError(ValueError):
    """Finite, safe setup diagnostics that may be persisted and shown by the CLI."""

    MESSAGES = {
        'unknown_family': 'Unknown workroom family.',
        'unknown_robot_profile': 'Robot profile must be original or l_mount.',
        'unknown_task': 'Task is not uniquely defined in the selected family.',
        'family_required': 'This task number requires --family; bare task1/2/3 retain the card-room aliases.',
        'task_planned': 'Task is planned: a runnable frozen release and native task adapter are required.',
        'session_adapter_unavailable': 'This competitive task has no supported authenticated session adapter; it cannot use ordinary episodes or the card-room board-game contract.',
        'task_not_published': 'Task must be uniquely published by the selected bench service. Configure its matching frozen release.',
        'session_contract_mismatch': 'Published task kind or creation endpoint disagrees with the selected session.',
        'configuration_unavailable': 'Robot configuration is not advertised for this task. Use the published ordinary function runner and a matching configured release.',
        'configuration_choice_invalid': 'The model did not choose one advertised robot configuration before episode creation.',
        'configuration_contract_mismatch': 'Published robot configuration options differ from the pinned catalog.',
        'player_models_require_multiplayer': 'Per-player model bindings require an authenticated two-player game.',
        'release_mismatch': 'Created episode does not use the catalog-selected frozen task release.',
        'robot_episodes_unavailable': 'This task supports engineering validation only, not ordinary robot episodes.',
    }

    def __init__(self, code, *, capabilities=None):
        self.code = code
        message = self.MESSAGES[code]
        if code == 'robot_episodes_unavailable':
            message = robot_episode_unavailable_reason({'capabilities': capabilities}) or message
        super().__init__(message)

    def public_metadata(self):
        return {'code': self.code, 'message': str(self)}


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
    capabilities: dict | None = None
    robot_configurations: dict | None = None
    configuration_id: str | None = None

    @property
    def practice_rsi_allowed(self):
        if engineering_capabilities({'capabilities': self.capabilities}) is not None:
            return False
        return not (self.family == 'card_room'
                    and self.task_id in ('connect_four_5x5', 'hex7x7', *CARD_TASKS))

    def metadata(self):
        result = {'schema': 'agent_scene_task_selection.v1', 'family': self.family,
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
        if self.configuration_id is not None:
            result['configuration_id'] = self.configuration_id
        if self.robot_configurations is not None:
            from bench.protocol.robot_configurations import project_robot_configurations
            result['robot_configurations'] = project_robot_configurations(self.robot_configurations)
        if self.capabilities is not None:
            result['capabilities'] = deepcopy(self.capabilities)
        return result

    def require_runnable(self):
        if engineering_capabilities({'capabilities': self.capabilities}) is not None:
            raise TaskSetupError('robot_episodes_unavailable', capabilities=self.capabilities)
        if self.status == 'planned':
            raise TaskSetupError('task_planned')
        if not self.adapter_available:
            raise TaskSetupError('session_adapter_unavailable')

    def verify_release(self, spec):
        actual = spec.get('task_release_sha256') if isinstance(spec, dict) else None
        expected = self.source_release.get('manifest_sha256') if self.source_release else None
        if self.configuration_id is not None and spec.get('configuration_id') != self.configuration_id:
            raise TaskSetupError('configuration_contract_mismatch')
        if self.pin_release and (expected is None or actual != expected):
            raise TaskSetupError('release_mismatch')


def _selection(definition, *, pin_release):
    family, task_id = definition['family'], definition['task_id']
    rules = definition.get('rules', {}).get('contract') or {}
    board = family == 'card_room' and (task_id in GAME_NUMBERS or task_id in CARD_TASKS)
    turn = family == 'living_room' and task_id == 'tower_extraction'
    competitive = turn or rules.get('competitive') is True
    kind = 'board_game' if board else 'two_player_adversarial' if competitive else 'single_agent'
    # Player identities are distinct from robot namespaces (front/rear).
    players = ('player_0', 'player_1') if board or turn else tuple(rules.get('players', ())) if competitive else ('single',)
    # Project metadata, not task objects, reset state, evaluator internals or rules.
    robot = definition.get('robot_requirements', {})
    validation = deepcopy(definition.get('validation'))
    capabilities = engineering_capabilities(definition)
    if capabilities is not None:
        return TaskSelection(family=family, task_id=task_id,
            task_number=definition.get('task_number', definition.get('difficulty')),
            session_kind='engineering', players=(), status=definition['status'],
            source_release=deepcopy(definition.get('source_release')), pin_release=pin_release,
            adapter_available=False, capabilities=capabilities,
            cannot_run_reason=robot_episode_unavailable_reason(definition), robot_requirements={},
            validation=validation)
    if validation is not None and family == 'mechanical_assembly':
        source_release = definition.get('source_release') or {}
        candidate_release = definition.get('candidate_referee', {}).get('release', {})
        loose_candidate = definition.get('design', {}).get('loose_bolt_candidate', {}).get('release', {})
        release = source_release or candidate_release
        qualification = definition.get('qualification', {})
        validation.update({
            'release_sha256': release.get('manifest_sha256'),
            # Seed-case and robot-revision overlays inherit no evidence but still
            # report the selected release's own preparation status.
            'release_status': qualification.get('release_status') or (
                qualification.get('status') if qualification.get('inherited_qualification') is False else None) or
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
        adapter_available=not competitive or board or turn, cannot_run_reason=(definition.get('cannot_run_reason') or
            (TaskSetupError.MESSAGES['session_adapter_unavailable'] if competitive and not (board or turn) else None)),
        robot_requirements={key: deepcopy(robot[key]) for key in
            ('hand_id', 'articulations', 'active_sides', 'assembly_version') if key in robot},
        validation=validation, robot_configurations=deepcopy(definition.get('robot_configurations')))


def select_task(name, family=None, *, robot_profile=None):
    """Resolve family-local numbering without assigning card semantics to task2/3.

    Explicit families select the catalog release. Bare canonical IDs preserve
    service-selected historical releases; known planned/competitive tasks still
    cannot bypass their readiness checks by dropping --family.
    """
    if family is not None and family not in FAMILIES:
        raise TaskSetupError('unknown_family')
    if robot_profile not in (None, 'original', 'l_mount'):
        raise TaskSetupError('unknown_robot_profile')
    if not isinstance(name, str) or not name or len(name) > 160:
        raise TaskSetupError('unknown_task')
    selected = GAME_ALIASES.get(name, name) if family is None else name
    definitions = list_tasks(family, include_planned=True, include_superseded=True,
                             robot_profile=robot_profile)
    matches = [row for row in definitions if row['task_id'] == selected]
    if not matches and family is not None:
        number = re.fullmatch(r'task(\d+)', name)
        if family == 'mechanical_assembly':
            number = number or re.fullmatch(r'L(\d+)', name)
        if number:
            matches = [row for row in definitions
                       if not row.get('superseded_by')
                       and row.get('task_number', row.get('difficulty')) == int(number[1])]
    if len(matches) == 1:
        return _selection(matches[0], pin_release=family is not None or robot_profile is not None)
    if matches or family is not None:
        raise TaskSetupError('unknown_task')
    if re.fullmatch(r'(task|L)\d+', name):
        raise TaskSetupError('family_required')
    # Other published benchmark tasks keep their existing service-only path.
    return TaskSelection(family=None, task_id=selected)


def select_configuration(selection, published, configuration_id):
    """Bind a trusted host selection to one currently advertised immutable option."""
    from bench.protocol.robot_configurations import checked_public_configurations, project_robot_configurations
    if selection.session_kind != 'single_agent':
        raise TaskSetupError('configuration_unavailable')
    try:
        public = checked_public_configurations(published.get('robot_configurations'))
    except (KeyError, TypeError, ValueError):
        raise TaskSetupError('configuration_contract_mismatch') from None
    if configuration_id not in public:
        raise TaskSetupError('configuration_unavailable')
    if selection.pin_release:
        try:
            expected = project_robot_configurations(selection.robot_configurations)
        except (KeyError, TypeError, ValueError):
            raise TaskSetupError('configuration_contract_mismatch') from None
        if public != expected:
            raise TaskSetupError('configuration_contract_mismatch')
        reference = deepcopy(selection.robot_configurations[configuration_id]['source_release'])
    else:
        reference = {'manifest_sha256': public[configuration_id]['task_release_sha256']}
    return replace(selection, source_release=reference, pin_release=True,
        robot_requirements=deepcopy(public[configuration_id]['robot_requirements']),
        configuration_id=configuration_id)


def published_task(selection, tasks):
    selection.require_runnable()
    if not isinstance(tasks, list):
        raise TaskSetupError('task_not_published')
    matches = [row for row in tasks if isinstance(row, dict) and row.get('task_id') == selection.task_id]
    if len(matches) != 1:
        raise TaskSetupError('task_not_published')
    task = deepcopy(matches[0])
    capabilities = engineering_capabilities(task)
    if capabilities is not None:
        raise TaskSetupError('robot_episodes_unavailable', capabilities=capabilities)
    if task.get('status') == 'planned' or task.get('runnable') is False or task.get('native_execution_available') is False:
        raise TaskSetupError('task_planned')
    published_hash = task.get('task_release_sha256')
    if (selection.pin_release and (selection.configuration_id is not None or not task.get('robot_configurations')) and published_hash is not None
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
    if selection.task_id in CARD_TASKS and (not board or task.get('protocol') != CARD_PROTOCOL
            or task.get('schema_endpoint') != 'GET /board-games/schema?protocol=' + CARD_PROTOCOL):
        raise TaskSetupError('session_contract_mismatch')
    if selection.task_id not in CARD_TASKS and task.get('protocol') == CARD_PROTOCOL:
        raise TaskSetupError('session_contract_mismatch')
    return task, board


def list_scene_tasks(family, *, robot_profile=None):
    if family not in FAMILIES:
        raise TaskSetupError('unknown_family')
    return [_selection(row, pin_release=True).metadata() for row in
            list_tasks(family, include_planned=True, robot_profile=robot_profile)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--robot-profile', choices=('original', 'l_mount'), help='physical robot (default: original)')
    args = parser.parse_args()
    print(json.dumps(list_scene_tasks(args.family, robot_profile=args.robot_profile),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
