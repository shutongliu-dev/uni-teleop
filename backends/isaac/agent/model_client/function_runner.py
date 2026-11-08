"""Public function Agent loop for ordinary episodes and authenticated games.

The trusted host owns tokens, action identities and turn scheduling. Models see
only public state and function schemas. Optional host-bound skill runtimes
execute model-authored bounded programs; a separate learning runner owns stages.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import time

from bench.paths import OUTPUT_ROOT
from bench.protocol.board_games import BoardGameResult, PlayerSession
from bench.protocol.card_games import CardGameResult
from bench.protocol.turn_games import TurnGameResult, PlayerSession as TurnPlayerSession
from bench.protocol.schemas import Budgets, EpisodeSpec, PolicyDescriptor, RobotCapabilities
from bench.protocol.allocations import (MAX_EPISODE_SIM_TIME_S, MAX_EPISODE_WALL_TIME_S,
                                        MAX_EPISODE_ACTIONS)
from .actions import parse_json_object, public_observation
from .bench_http import BenchHTTPClient, BenchHTTPError
from .capture import capture
from .function_http import FunctionHTTPClient, FunctionRejected, create_board_game, create_turn_game
from .function_session import FunctionSession
from .outcomes import classify_report, error_record, error_status
from .runner import canonical_hash, configured_model, optional_output_tokens, select_geometry_policy
from .scene_tasks import FAMILIES, TaskSetupError, published_task, select_task, select_configuration
from .time_limits import minimum_limit, optional_positive, optional_seconds
from .transport import ModelGatewayConfig, ModelGatewayError, GATEWAY_RETRY_POLICY
from .card_games import CARD_PROTOCOL, CARD_TASKS, PublicEventDelivery, committed_operation_counts
from .player_models import load_player_models
from .context_projection import VERSION as INPUT_PROJECTION_VERSION, project_context
from .decision_recall import DecisionArchive
from .recall_projection import VERSION as RECALL_PROJECTION_VERSION, project_recall_history
from .learning.bindings import load_skill_bindings, augment_context, RSI_PROMPT


VERSION = 'public_function_agent.v3'
PROMPT = '''Control the robot using the published task, current images, robot guide,
function schemas and public feedback. Return exactly one JSON object matching
decision_schema: a function call, a game intent where offered, or
{"stop":true}. Use absolute targets and units from the function schema. The host
binds observation_sequence and request/action IDs; do not supply these fields.
The robot guide and available functions describe this selected robot. IK and
hand presets are geometric queries, not collision checks or successful grasps.
Current joint measurements differ from acknowledged drive targets under load.
Use actual visual and contact feedback to assess motion. Contact is not proof
of stable grasp; primitive completion is not task success. No object poses may
be written. No private evaluator state or automatic grasp planner is available.
All listed cameras are from the current decision snapshot. project_pixel uses
the full verified metric depth; previews and sparse grids are not substitutes.
Only the trusted bench decides task completion, committed moves and game results.
The task field contains fixed requirements, not a current referee snapshot.
Where offered, call get_task to read the current task phase and adjudication.
Its feedback describes the observation sequence recorded in that response;
after physical actions, query again before relying on an earlier phase.
For all games obey state.player.hand and joint_names: move only your own robot
and only on your turn. The public referee state determines the current player;
one turn may require many function calls. An intent declares a selection, not a
physical move. board_turn_changed or turn_changed cancellation is turn handoff,
not proof that an unfinished wrist target converged. Card-game turn_index is an
operation authorization epoch: the same player may remain current after it changes.
At a fresh card-game reset first explicitly hold for 0.5 s using acknowledged_drive
to establish native baseline evidence, then inspect state before declaring an intent.
Follow phase and current_player, never epoch parity. public_game_events are the
shared history of actual reveals and committed operations. Remember every reveal,
including unexpected flips; cancellation cannot erase it. For memory_pairs reveal
cards sequentially and physically collect a match or flip a mismatch back as the
public phase requires. For disk_curling release the selected disk and wait for
free-motion settlement before reading scores. In tower extraction, place
each removed block in your collection zone, never on top of the tower. Release
and stable referee settlement are required before handoff; causing collapse loses.
Budget exhaustion, invalid reset and missing evidence do not establish a winner.
You retain your own text history across turns; the opponent's private reasoning
is not available. Follow learning_policy. This interface offers JSON calls only;
it cannot execute Python, rewrite skills or perform an RSI validation loop.
'''


@dataclass
class RunConfig:
    bench_url: str
    task: str
    output: Path
    robot: str | None = None
    seed: int | None = None
    max_decisions: int = 128  # per model/player
    max_actions: int = 256   # shared episode total
    max_action_sim_s: float = 5.  # admission budget only; controller limits unchanged
    stop_after_commits_per_player: int | None = None  # optional public-evidence validation target
    sim_budget: float | None = 120.
    wall_budget: float | None = 1800.
    model_timeout: float | None = 300.
    http_timeout: float | None = 60.
    max_output_tokens: int | None = 4096
    model: str = 'gpt-6-astra'
    reasoning_effort: str = 'max'
    provider: str = 'gateway'
    credentials: Path | None = None
    model_retries: int | None = None
    model_retry_policy: str | None = None
    visual_input: str = 'rgbd'
    max_status_queries: int = 1200  # per action wait; retained even without time budgets
    family: str | None = None  # explicit families bind the source catalog release
    robot_profile: str | None = None  # physical original/L-mount robot; independent of arm layout
    configuration: str = 'auto'  # Model chooses among offered installations before native creation
    player_models: Path | None = field(default=None, repr=False)  # Private, complete two-player bindings
    history_mode: str = 'full'
    geometry_math: bool = False
    max_function_calls: int | None = None  # per learner; includes calls inside skills
    practice: dict | None = None  # public case selection, never object state
    skill_bindings: Path | None = field(default=None, repr=False)


def resolve_task(name, tasks, family=None):
    """Family-local aliases select only compatible, publicly offered sessions."""
    return published_task(select_task(name, family), tasks)


def select_task_controller(policies, selection, *, robot=None, board=False, turn=False):
    """Keep catalog pins strict; infer historical robots only from public discovery."""
    if robot is None:
        if turn:
            robot = 'fr3_sharpa_instances'
        elif board:
            robot = 'dual_fr3_sharpa_w01'
        elif selection.pin_release:
            robot = (selection.robot_requirements or {}).get('hand_id') or 'sharpa_w01_right'
        elif selection.family is None:
            robot = 'sharpa_w01_right'  # Preserve the service-only legacy default.
        # A known bare canonical ID selects the service's historical release,
        # whose assembly may differ from the current catalog's robot. The task-
        # scoped descriptor pins that configuration; never guess among robots.
    controller = PolicyDescriptor.model_validate(select_geometry_policy(
        policies, robot, allow_instances=robot in (None, 'fr3_sharpa_instances'))).model_copy(deep=True)
    if robot is None:
        if len(controller.supported_robot_ids) != 1:
            raise BenchHTTPError('Discovered controller requires an explicit robot selection',
                                 category='invalid_response')
        robot = controller.supported_robot_ids[0]
    return robot, controller


def learning_policy(task_id, family=None, *, selection=None):
    selection = selection or select_task(task_id, family)
    numbered = f'task{selection.task_number}' if selection.task_number is not None else selection.task_id
    practice = selection.practice_rsi_allowed
    return {'task': numbered, 'task_id': selection.task_id, 'family': selection.family,
            'session_kind': selection.session_kind, 'players': list(selection.players),
            'practice_rsi_allowed': practice,
            'frozen_skills_required': not practice,
            'skill_condition': 'empty_library_public_functions_only',
            'skill_identity': canonical_hash({'skills': [], 'runtime': VERSION}),
            'skill_mutation_tools_available': False, 'rsi_scheduler_implemented': False}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _record_model_event(path, entry):
    """Persist typed transport progress before retry waits, including setup."""
    with path.open('a') as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if entry['event'] in ('model_request_started', 'model_request_completed',
                          'model_request_failed', 'model_retry_wait'):
        print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)


class RunDeadline(Exception):
    """The host's intentional episode horizon, not an HTTP/model failure."""


def _remaining(deadline):
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise RunDeadline()
    return remaining


def board_outcomes(result):
    """A shared episode success is never both players winning."""
    if result is None:
        return {player: 'unknown' for player in ('player_0', 'player_1')}
    model = CardGameResult if result.get('board_game', {}).get('protocol') == CARD_PROTOCOL else BoardGameResult
    checked = model.model_validate(result)
    game = checked.board_game
    if game.status.outcome == 'win':
        return {player: 'win' if game.winner == player else 'loss' for player in ('player_0', 'player_1')}
    if game.status.outcome == 'draw':
        return dict.fromkeys(('player_0', 'player_1'), 'draw')
    return dict.fromkeys(('player_0', 'player_1'), game.status.outcome)


def turn_outcomes(result, players=('player_0', 'player_1')):
    """Preserve referee outcomes without turning an interruption into a loss."""
    if result is None:
        return dict.fromkeys(players, 'unknown')
    game = TurnGameResult.model_validate(result).turn_game
    if game.status == 'completed':
        if game.winner_player_id is not None and game.loser_player_id is not None:
            return {player: ('win' if player == game.winner_player_id else
                             'loss' if player == game.loser_player_id else 'unknown') for player in players}
        if game.winner_player_id is None and game.loser_player_id is None:
            return dict.fromkeys(players, 'draw')
        return dict.fromkeys(players, 'unknown')
    return dict.fromkeys(players, 'interrupted')


def _game_position(state, board):
    game = state['board_game' if board else 'turn_game']
    return ((game['terminal'], game['current_player']) if board else
            (game['status'] != 'active', game['current_player_id']))


def _players(ready, *, turn=False):
    model = TurnPlayerSession if turn else PlayerSession
    players = [model.model_validate(row) for row in ready['players']]
    if (len(players) != 2 or {row.binding.player_id for row in players} != {'player_0', 'player_1'}
            or len({row.session_token for row in players}) != 2):
        raise ValueError('Games require two distinct authenticated players')
    if turn and {row.binding.player_id: row.binding.robot_id for row in players} != {
            'player_0': 'front', 'player_1': 'rear'}:
        raise ValueError('Tower extraction requires the published front/rear player bindings')
    return players


def _public_ready(ready, board, turn=False):
    # Credentials and unrecognised create-response fields never enter recordings.
    value = {key: deepcopy(ready[key]) for key in ('episode_id', 'spec', 'capabilities', 'observation', 'task')
             if key in ready}
    if board or turn:
        value['players'] = [row.binding.model_dump(mode='json') for row in _players(ready, turn=turn)]
        for key in ('board_game', 'turn_game', 'public_rules', 'public_rules_sha256', 'geometry', 'rules'):
            if key in ready:
                value[key] = deepcopy(ready[key])
    return value


def _validated_ready(ready, request, task, board, turn=False, *, controller: PolicyDescriptor):
    spec = EpisodeSpec.model_validate(ready['spec'])
    caps = RobotCapabilities.model_validate(ready['capabilities'])
    observation = public_observation(ready['observation'])
    if spec.system0 != controller or caps.system0 != controller:
        raise BenchHTTPError('Created episode differs from the discovered controller', category='invalid_response')
    if (ready['episode_id'] != spec.episode_id or spec.task_id != task['task_id']
            or observation.episode_id != spec.episode_id or observation.profile != 'vision_proprio'
            or spec.observation_profile != request['observation_profile']
            or spec.robot_id != request['robot_id'] or caps.robot_id != spec.robot_id
            or spec.agent_config_sha256 != request['agent_config_sha256']
            or spec.budgets.model_dump(mode='json') != request['budgets']
            or spec.track != 'system0' or spec.system0 != caps.system0
            or spec.system0.policy_id != request['policy_id']
            or spec.system0.version != request['policy_version']
            or ('seed' in request and spec.seed != request['seed'])
            or spec.configuration_id != request.get('configuration_id')):
        raise ValueError('Created episode does not match the requested contract')
    if (not board and not turn
            and (task.get('public_rules_sha256') is not None or task.get('scoring_contract') is not None)
            and not isinstance(ready.get('task'), dict)):
        raise BenchHTTPError('Created episode omitted its declared public task', category='invalid_response')
    actual = deepcopy(ready.get('task', task))
    if actual.get('task_id') != spec.task_id:
        raise ValueError('Episode task identity differs')
    if not board and not turn and (task.get('reset_case') is not None or actual.get('reset_case') is not None):
        from .seeded_tasks import checked_reset_case
        expected_case = checked_reset_case(task.get('reset_case'), seed=spec.seed,
                                          configuration_id=spec.configuration_id, task_id=spec.task_id)
        actual_case = checked_reset_case(actual.get('reset_case'), seed=spec.seed,
                                        configuration_id=spec.configuration_id, task_id=spec.task_id)
        if (actual_case != expected_case
                or spec.task_release_sha256 != task.get('task_release_sha256')):
            raise ValueError('Created episode differs from the discovered frozen reset case')
    if (not board and not turn and task.get('public_rules_sha256') is not None
            and actual.get('public_rules_sha256') != task['public_rules_sha256']):
        raise ValueError('Created episode rules differ from the published task')
    if (not board and not turn and task.get('public_rules_sha256') is not None
            and actual.get('goal') != task.get('goal')):
        raise ValueError('Created episode placement goal differs from the published task')
    if (not board and not turn and actual.get('scoring_contract') != task.get('scoring_contract')):
        raise ValueError('Created episode scoring differs from the published task')
    if spec.task_id == 'task0':
        goal = actual.get('goal', {})
        if (actual.get('episode_seed') != spec.seed or len(goal.get('target_cells', [])) != 3
                or len(goal.get('target_positions_m', [])) != 3):
            raise ValueError('Task0 requires its episode-specific public targets')
    if board or turn:
        _players(ready, turn=turn)
    board_keys = ('public_rules', 'public_rules_sha256') if spec.task_id in CARD_TASKS else (
        'rules', 'geometry', 'public_rules', 'public_rules_sha256')
    if board and any(ready.get(key) != task.get(key) for key in board_keys):
        raise BenchHTTPError('Created board-game contract differs from the published task',
                             category='invalid_response')
    if turn and ready['turn_game']['public_rules'] != task.get('public_rules'):
        raise ValueError('Created turn-game rules differ from the published task')
    return spec.model_dump(mode='json'), actual


def task_requirements_context(task):
    """Keep creation-time referee snapshots out of repeated model requirements.

    Dynamic adjudication is available through the accounted get_task tool, whose
    response carries its public observation sequence. Rules, declared scoring
    and seeded static goals remain unchanged; the ready archive keeps the full
    validated initial snapshot.
    """
    requirements = deepcopy(task)
    if requirements.get('task_kind') == 'manipulation' and 'public_rules_sha256' in requirements:
        for key in ('referee', 'episode', 'scoring'):
            requirements.pop(key, None)
    return requirements


def run(config: RunConfig, *, bench=None, models=None, source_verification=None,
        learning_bindings=None, learning_feedback=None, learning_trial=None):
    """Run one episode. Tests can inject CPU transports and scripted models."""
    if config.configuration not in ('auto', 'single_arm', 'dual_same_side', 'dual_opposed'):
        raise ValueError('Unknown robot configuration mode')
    if config.history_mode not in ('full', 'current_with_recall'):
        raise ValueError('Unknown history mode')
    if type(config.geometry_math) is not bool:
        raise ValueError('geometry_math must be a boolean')
    local_functions, local_execute = None, None
    if config.geometry_math:
        from .geometry_math import FUNCTION_INPUTS as local_functions, execute as local_execute
    if type(config.max_decisions) is not int or not 1 <= config.max_decisions <= 256:
        raise ValueError('max_decisions must be between 1 and 256 per player')
    if config.visual_input != 'rgbd':
        raise ValueError('Public function runner currently requires the full RGB-D input condition')
    if config.practice is not None:
        from bench.protocol.practice import PracticeSelection
        PracticeSelection.model_validate(config.practice, strict=True)
    if config.seed is not None and (type(config.seed) is not int or not 0 <= config.seed < 2**32):
        raise ValueError('seed must be an unsigned 32-bit integer')
    for name in ('wall_budget', 'sim_budget', 'model_timeout', 'http_timeout'):
        optional_positive(getattr(config, name), name)
    optional_positive(config.wall_budget, 'wall_budget', maximum=MAX_EPISODE_WALL_TIME_S)
    optional_positive(config.sim_budget, 'sim_budget', maximum=MAX_EPISODE_SIM_TIME_S)
    if config.max_action_sim_s is None:
        raise ValueError('max_action_sim_s must be finite and positive')
    optional_positive(config.max_action_sim_s, 'max_action_sim_s')
    if config.stop_after_commits_per_player is not None and (
            type(config.stop_after_commits_per_player) is not int
            or config.stop_after_commits_per_player < 1):
        raise ValueError('stop_after_commits_per_player must be null or a positive integer')
    if type(config.max_actions) is not int or not 1 <= config.max_actions <= MAX_EPISODE_ACTIONS:
        raise ValueError(f'max_actions must be between 1 and {MAX_EPISODE_ACTIONS}')
    if type(config.max_status_queries) is not int or config.max_status_queries < 1:
        raise ValueError('max_status_queries must be a positive integer')
    if config.max_output_tokens is not None and (type(config.max_output_tokens) is not int
            or not 1 <= config.max_output_tokens <= 32768):
        raise ValueError('max_output_tokens must be null or between 1 and 32768')
    if config.player_models is not None and models is not None:
        raise ValueError('Choose either private player model bindings or injected model clients')
    if config.max_function_calls is not None and (type(config.max_function_calls) is not int
            or not 1 <= config.max_function_calls <= 100000):
        raise ValueError('max_function_calls must be null or between 1 and 100000')
    if config.skill_bindings is not None and learning_bindings is not None:
        raise ValueError('Choose file or injected skill bindings, not both')
    learning_bindings = (load_skill_bindings(config.skill_bindings)
        if config.skill_bindings is not None else dict(learning_bindings or {}))
    if learning_bindings and len({b.library.path for b in learning_bindings.values()}) != len(learning_bindings):
        raise ValueError('Each learner requires an independent skill library')
    if learning_feedback is not None and (not learning_bindings
            or any(b.phase != 'practice' for b in learning_bindings.values())):
        raise ValueError('Prior learning feedback is available only in practice')
    if learning_feedback is not None:
        canonical_hash(learning_feedback)
    if learning_trial is not None and set(learning_bindings) != {'single'}:
        raise ValueError('Learning trial receipts require a single host-bound learner')
    prompt = PROMPT
    if learning_bindings:
        prompt = PROMPT.replace('This interface offers JSON calls only;\nit cannot execute Python, rewrite skills or perform an RSI validation loop.',
            'Follow the advertised skill runtime and its host-fixed learning phase.') + RSI_PROMPT
    bindings = load_player_models(config.player_models) if config.player_models is not None else None
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    if source_verification is not None:
        save(output / 'source_verification.json', source_verification)
    owned_bench = bench is None
    bench = bench or BenchHTTPClient(config.bench_url, timeout_s=config.http_timeout)
    own_models = models is None
    models = {} if models is None else models
    sessions, clients, histories, counts, model_reports = {}, {}, {}, {}, {}
    skill_runtimes = {}
    result = None
    board = False
    turn = False
    multiplayer = False
    keys = ('single',)
    model_configs = {}
    episode_id = None
    error = None
    finalization_error = None
    termination = 'setup_incomplete'
    result_source = None
    deadline = None
    settings = None
    selection = None
    task_setup_error = None
    ready_validated = False
    pending_read_errors = {}
    configuration_model_calls = 0
    configuration_model_reports = []
    configuration_choice = None
    event_delivery = None

    def deliver_events(*, through=None):
        if event_delivery is None:
            return
        try:
            event_delivery.drain(histories, through=through)
        finally:
            # Preserve received public history even if a later page/player fails.
            for player, rows in event_delivery.events.items():
                save(output / player / 'public_game_events.json', {
                    'protocol': CARD_PROTOCOL, 'cursor': event_delivery.cursors[player], 'events': rows})

    def save_configuration_receipt(receipt):
        nonlocal configuration_choice
        configuration_choice = receipt
        save(output / 'configuration_choice.json', receipt)
        if 'metadata' in receipt:
            configuration_model_reports[:] = [receipt['metadata']]

    def record_configuration_event(event):
        # choose_configuration publishes its public request before generate.
        # There is no episode deadline yet; retain the configured provider's
        # existing retry policy while making its progress auditable.
        _record_model_event(output / 'configuration_choice_model_api_events.jsonl', {
            'phase': 'configuration_choice', 'player': 'single', 'decision_index': 0,
            'request_sha256': canonical_hash(configuration_choice['messages']), **event})

    def pending_state():
        # A broken/unreadable ledger cannot establish that motion stopped.
        # Latch uncertainty for this run even if a later read happens to work.
        unresolved = bool(pending_read_errors)
        for key, session in sessions.items():
            try:
                unresolved = bool(session.pending) or unresolved
            except Exception as exc:
                pending_read_errors.setdefault(key, error_record(exc))
                unresolved = True
        return unresolved

    try:
        selection = select_task(config.task, config.family, robot_profile=config.robot_profile)
        save(output / 'task_selection.json', selection.metadata())
        selection.require_runnable()
        if bindings is not None and selection.session_kind not in ('board_game', 'turn_game'):
            raise TaskSetupError('player_models_require_multiplayer')
        task, board = published_task(selection, bench.list_tasks())
        if config.stop_after_commits_per_player is not None and task['task_id'] not in CARD_TASKS:
            raise ValueError('A per-player commit target requires the public card-game event protocol')
        turn = task.get('task_kind') == 'turn_game'
        multiplayer = board or turn
        keys = ('player_0', 'player_1') if multiplayer else ('single',)
        if bindings is not None and not multiplayer:
            raise TaskSetupError('player_models_require_multiplayer')
        model_configs = {key: (replace(config, **bindings[key].public_metadata(),
                                      credentials=bindings[key].credentials)
                               if bindings is not None else config) for key in keys}
        if own_models:
            if bindings is None:
                shared = (ModelGatewayConfig.from_file(config.credentials) if config.credentials
                          else ModelGatewayConfig.from_env())
                credentials = dict.fromkeys(keys, shared)
            else:
                credentials = {key: ModelGatewayConfig.from_file(bindings[key].credentials) for key in keys}
            for key in keys:
                models[key] = configured_model(model_configs[key], credentials[key])
        if set(models) != set(keys) or len({id(model) for model in models.values()}) != len(keys):
            raise ValueError('Supply one separate model client per player/context')
        options = task.get('robot_configurations')
        if options and not multiplayer:
            # Validate every advertised pin before sending public choices to a model.
            for identifier in options:
                select_configuration(selection, task, identifier)
            if config.configuration == 'auto':
                from .configuration_choice import choose_configuration
                configuration_model_calls += 1
                chosen, configuration_choice = choose_configuration(models['single'], task,
                    max_output_tokens=config.max_output_tokens, timeout_s=config.model_timeout,
                    on_receipt=save_configuration_receipt,
                    on_event=record_configuration_event if (model_configs['single'].provider == 'micu'
                        or (model_configs['single'].provider == 'gateway'
                            and model_configs['single'].model_retry_policy == GATEWAY_RETRY_POLICY)) else None)
                if chosen is None:
                    raise TaskSetupError('configuration_choice_invalid')
            else:
                chosen = config.configuration
                configuration_choice = {'selected': {'configuration_id': chosen},
                                        'source': 'explicit_host_configuration', 'model_calls': 0}
                save(output / 'configuration_choice.json', configuration_choice)
            selection = select_configuration(selection, task, chosen)
            task, board = published_task(selection, bench.list_tasks(
                task_id=task['task_id'], configuration_id=chosen))
            if task.get('configuration_id') != chosen:
                raise TaskSetupError('configuration_contract_mismatch')
            # Query the selected release's actual public contract and controller.
            save(output / 'task_selection.json', selection.metadata())
            policies = bench.list_policies(task_id=task['task_id'], configuration_id=chosen)
        else:
            if config.configuration != 'auto':
                raise TaskSetupError('configuration_unavailable')
            policies = bench.list_policies(task_id=task['task_id'])
        from .seeded_tasks import select_episode_case
        task, episode_seed = select_episode_case(bench, selection, task, config.seed)
        if task.get('reset_case') is not None:
            save(output / 'episode_case_discovery.json', task)
        robot, controller = select_task_controller(policies, selection,
            robot=config.robot, board=board, turn=turn)
        policy = controller.model_dump(mode='json')
        budgets = Budgets(sim_time_s=config.sim_budget, wall_time_s=config.wall_budget,
                          max_actions=config.max_actions, max_model_calls=config.max_decisions * len(keys),
                          max_action_sim_s=config.max_action_sim_s).model_dump(mode='json')
        settings = {'protocol': VERSION, 'task_id': task['task_id'],
                    'mode': 'turn_game' if turn else 'board_game' if board else 'single_agent',
                    'task_selection': selection.metadata(),
                    'configuration_choice': configuration_choice,
                    'pre_episode_model_calls': configuration_model_calls,
                    'source_verification_sha256': canonical_hash(source_verification) if source_verification else None,
                    'controller': policy, 'budgets': budgets, 'max_decisions_per_player': config.max_decisions,
                    'action_recovery': {'max_status_queries_per_wait': config.max_status_queries},
                    'visual_input': config.visual_input, 'tactile_input': 'on',
                    'models': {key: model.public_metadata() for key, model in models.items()},
                    'learning_policy': learning_policy(task['task_id'], selection=selection), 'system_prompt': prompt,
                    'history': 'independent_full_game_text_current_images_only'}
        if multiplayer:
            settings['input_projection'] = INPUT_PROJECTION_VERSION
        if task.get('reset_case') is not None:
            settings['episode_case'] = {'seed': episode_seed,
                'case_sha256': task['reset_case']['case_sha256'],
                'task_release_sha256': task['task_release_sha256']}
        if config.stop_after_commits_per_player is not None:
            settings['validation_target'] = {
                'commits_per_player': config.stop_after_commits_per_player,
                'source': 'verified_public_operation_committed_events',
                'pass_counts': False, 'game_completion_claimed': False}
        if config.history_mode == 'current_with_recall':
            settings.update(history_mode=config.history_mode,
                history='independent_current_context_with_public_record_recall',
                input_projection=RECALL_PROJECTION_VERSION)
        if config.geometry_math:
            settings['local_geometry_math'] = {'enabled': True, 'functions': sorted(local_functions),
                                              'scope': 'Pure explicit public numeric arguments; no private state or automatic actions.'}
        if config.max_function_calls is not None:
            settings['max_function_calls_per_player'] = config.max_function_calls
        if config.practice is not None:
            settings['practice'] = deepcopy(config.practice)
        if learning_bindings:
            if set(learning_bindings) != set(keys):
                raise ValueError('Skill bindings must match the exact episode learners')
            for binding in learning_bindings.values():
                binding.check_task(selection)
            settings['skill_bindings'] = {key: b.public_metadata() for key, b in learning_bindings.items()}
            settings['learning_policy'].update(
                skill_condition='model_authored_bounded_skills_public_functions',
                skill_identity=canonical_hash(settings['skill_bindings']),
                skill_mutation_tools_available=all(b.phase == 'practice' for b in learning_bindings.values()),
                skill_runtime_implemented=True,
                rsi_scheduler_implemented=True,
                stage_orchestration='external_host_learning_runner')
            if learning_feedback is not None:
                settings['prior_practice_feedback_sha256'] = canonical_hash(learning_feedback)
        if bindings is not None:
            settings['model_bindings'] = {key: bindings[key].public_metadata() for key in keys}
        save(output / 'agent_config.json', settings)
        request = {'task_id': task['task_id'], 'robot_id': robot, 'agent_id': 'public_function_agent',
                   'agent_version': VERSION, 'agent_config_sha256': canonical_hash(settings),
                   'track': 'system0', 'observation_profile': 'vision_proprio', 'budgets': budgets,
                   'policy_id': policy['policy_id'], 'policy_version': policy['version']}
        if selection.configuration_id is not None:
            request['configuration_id'] = selection.configuration_id
        if episode_seed is not None:
            request['seed'] = episode_seed
        if config.practice is not None:
            request['practice'] = deepcopy(config.practice)
        save(output / 'create_request.json', request)
        def bind_created(raw):
            # Establish request-owned cleanup access before validating optional
            # public fields or writing ready artifacts. No model sees raw data.
            nonlocal episode_id
            created = EpisodeSpec.model_validate(raw['spec'])
            if (created.episode_id != raw['episode_id'] or created.task_id != request['task_id']
                    or created.agent_id != request['agent_id']
                    or created.agent_version != request['agent_version']
                    or created.agent_config_sha256 != request['agent_config_sha256']):
                raise ValueError('Created episode identity does not match this request')
            if multiplayer:
                players = _players(raw, turn=turn)
                bindings = {row.binding.player_id: row.model_dump(mode='json') for row in players}
            else:
                bindings = {}
            episode_id = created.episode_id
            for key in keys:
                clients[key] = FunctionHTTPClient(config.bench_url, episode_id, player=bindings.get(key),
                    session_kind='turn_game' if turn else 'board_game' if board else 'episode',
                    game_protocol=CARD_PROTOCOL if request['task_id'] in CARD_TASKS else None,
                    client=bench.client, timeout_s=config.http_timeout, spec=created)
        # Creation is one POST. A lost response is never retried as a new episode.
        if multiplayer:
            create = create_turn_game if turn else create_board_game
            ready = create(bench, request, on_created=bind_created)
        else:
            ready = bench.create(request)
            bind_created(ready)
        # bind_created validated the spec. Check the pinned release before the
        # frozen reset-case comparison, which also compares release identity,
        # so pinned drift stays a release_mismatch setup error for every task.
        selection.verify_release(ready['spec'])
        spec, task = _validated_ready(ready, request, task, board, turn, controller=controller)
        for client in clients.values():
            client.bind_episode(spec, task=task if not multiplayer else None)
        if turn:
            for client in clients.values():
                client.bind_turn_state(ready['turn_game'])
        elif board:
            for client in clients.values():
                client.bind_board_state(ready)
        ready_validated = True
        save(output / 'episode_ready.json', _public_ready(ready, board, turn))
        task_context = task_requirements_context(task)
        deadline = None if config.wall_budget is None else time.monotonic() + config.wall_budget
        for key in keys:
            client = clients[key]
            client.deadline = deadline
            sessions[key] = FunctionSession(client, output=output / key, spec=spec,
                trial_id=key, max_actions=config.max_actions,
                max_calls=config.max_function_calls or config.max_decisions, deadline=deadline,
                max_status_queries=config.max_status_queries,
                decision_archive=(DecisionArchive(output / key, episode_id=episode_id, player_id=key)
                                  if config.history_mode == 'current_with_recall' else None),
                local_functions=local_functions, local_execute=local_execute)
            sessions[key].initialize()
            if learning_bindings:
                skill_runtimes[key] = learning_bindings[key].instantiate(sessions[key])
            histories[key], counts[key], model_reports[key] = [], 0, []
            if key == 'single':
                counts[key] = configuration_model_calls
                model_reports[key] = deepcopy(configuration_model_reports)
        if learning_trial is not None:
            binding = learning_bindings['single']
            if (learning_trial.get('phase') != binding.phase
                    or learning_trial.get('candidate_event_id') != binding.candidate_event_id
                    or learning_trial.get('bundle_hash') != binding.bundle_sha256):
                raise ValueError('Trial receipt changes the actual skill binding')
            save(output / 'host' / 'rsi_trial_binding.json', {
                **deepcopy(learning_trial), 'schema': 'card_rsi_trial_binding.v1',
                'episode_id': episode_id, 'trial_id': 'single',
                'fresh_context': True, 'author_history_visible': learning_feedback is not None,
                'skill_runtime_journal': 'single/host/skill_runtime_journal',
                'action_journal': 'single/host/action_journal'})
        if spec['task_id'] in CARD_TASKS:
            event_delivery = PublicEventDelivery(clients)
        # No copy of the create response containing tokens is saved or sent to a model.
        del ready
        termination = 'client_decision_budget'
        while True:
            if pending_state():
                termination = 'action_recovery_pending'
                break
            _remaining(deadline)
            if multiplayer:
                state = clients[keys[0]].state()
                deliver_events(through=state['board_game']['event_cursor'] if event_delivery else None)
                terminal, key = _game_position(state, board)
                if terminal:
                    result = clients[keys[0]].game_result()
                    result_source, termination = 'server_result', 'server_finished'
                    break
                if (config.stop_after_commits_per_player is not None
                        and all(value >= config.stop_after_commits_per_player
                                for value in committed_operation_counts(event_delivery).values())):
                    termination = 'requested_commits_completed'
                    break
                if key not in clients:
                    raise BenchHTTPError('Unknown current player', category='invalid_response')
            else:
                key = 'single'
                reply = clients[key].call({'request_id': 'host_result', 'function': 'get_result', 'arguments': {}})
                if reply['ok']:
                    result, result_source, termination = reply['result'], 'server_result', 'server_finished'
                    break
                if reply['error']['code'] != 'result_pending':
                    raise BenchHTTPError('Unable to read authoritative result', category='invalid_response')
            if counts[key] >= config.max_decisions:
                break
            if sum(session.actions for session in sessions.values()) >= config.max_actions:
                termination = 'client_action_budget'
                break
            session = sessions[key]
            # A skill can issue many calls before the next model decision. Its
            # session must respect the remaining shared game action budget too.
            session.max_actions = config.max_actions - sum(s.actions for k, s in sessions.items() if k != key)
            context = session.context()
            if multiplayer and _game_position(context['state'], board) != (False, key):
                continue
            folder = output / key / 'decisions' / f'{counts[key]:04d}'
            frames = capture(clients[key], episode_id, context['state']['observation'], folder, output,
                             visual_input=config.visual_input, tactile_input='on')
            context.update(task=task_context, learning_policy=settings['learning_policy'],
                           depth_observation=frames['depth_context'], decisions_remaining=config.max_decisions - counts[key])
            if key in skill_runtimes:
                augment_context(context, skill_runtimes[key])
                if learning_feedback is not None:
                    context['prior_practice_feedback'] = deepcopy(learning_feedback.get(key, []))
            if config.history_mode == 'current_with_recall':
                context['decision_index'] = counts[key]
            context['decision_schema']['one_of'].append({'stop': True})
            # Keep the complete journal. Only exact public repetition is omitted
            # from the shared text used by both the wire request and its record.
            if config.history_mode == 'current_with_recall':
                projected_history, projected_context, projection = project_recall_history(
                    histories[key], context, session.decision_archive)
                save(folder / 'input_projection.json', projection)
            elif multiplayer:
                projected_history, projected_context, projection = project_context(histories[key], context)
                save(folder / 'input_projection.json', projection)
            else:
                projected_history, projected_context = histories[key], context
            prompt_text = json.dumps(projected_context, ensure_ascii=False, allow_nan=False)
            turns = projected_history + [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}]
                                                                      + frames['image_content']}]
            save(folder / 'model_request.json', {'system_prompt': prompt,
                 'messages': projected_history + [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}]
                                                                               + frames['recorded_image_content']}],
                 'expanded_messages_sha256': canonical_hash(turns)})
            remaining = _remaining(deadline)
            decision_index = counts[key]
            counts[key] += 1
            kwargs = {'max_output_tokens': config.max_output_tokens,
                      'timeout_s': minimum_limit(config.model_timeout, remaining)}
            if (model_configs[key].provider == 'micu' or (model_configs[key].provider == 'gateway'
                    and model_configs[key].model_retry_policy == GATEWAY_RETRY_POLICY)):
                kwargs['retry_deadline'] = deadline
                request_sha256 = canonical_hash(turns)

                def record_model_event(event):
                    # Recovery transports emit typed fields, never provider error text,
                    # headers or credentials. Persist before waiting/retrying.
                    entry = {'player': key, 'decision_index': decision_index,
                             'request_sha256': request_sha256, **event}
                    _record_model_event(folder / 'model_api_events.jsonl', entry)

                kwargs['on_event'] = record_model_event
            try:
                response = models[key].generate(prompt, turns, **kwargs)
            except ModelGatewayError as exc:
                save(folder / 'model_error.json', {'decision_index': decision_index, **exc.public_metadata()})
                # A request limited by this run's remaining wall time can end
                # by transport timeout. Preserve that raw receipt, but do not
                # classify the intended episode horizon as an infrastructure
                # failure. Earlier/provider-specific timeouts stay errors.
                if (exc.category in ('transport_timeout', 'request_budget_exhausted') and deadline is not None
                        and (kwargs['timeout_s'] == remaining or exc.category == 'request_budget_exhausted')
                        and time.monotonic() >= deadline and not pending_state()):
                    raise RunDeadline() from None
                raise
            metadata = response.public_metadata()
            model_reports[key].append(metadata)
            save(folder / 'model_response.json', {'text': response.text, 'metadata': metadata})
            _remaining(deadline)  # A late model response cannot move the robot.
            past = {name: value for name, value in context.items() if name not in
                    ('spec', 'robot', 'catalog', 'tools', 'decision_schema', 'host_contract', 'task', 'learning_policy')}
            histories[key].extend([{'role': 'user', 'content': json.dumps(past, ensure_ascii=False, allow_nan=False)},
                                   {'role': 'assistant', 'content': response.text}])
            try:
                decision = parse_json_object(response.text)
            except ValueError:
                feedback = {'ok': False, 'error': {'code': 'invalid_arguments', 'message': 'Return one valid JSON decision'}}
            else:
                if decision == {'stop': True}:
                    termination = 'model_requested_stop'
                    break
                runtime = skill_runtimes.get(key)
                if (runtime is not None and set(decision) == {'function', 'arguments'}
                        and decision.get('function') in {t['function'] for t in runtime.tools()}):
                    if session.calls >= session.max_calls:
                        feedback = {'ok': False, 'function': decision['function'], 'error': {
                            'code': 'budget_exhausted', 'message': 'Client function-call budget exhausted'}}
                    else:
                        session.calls += 1
                        outer_call = session.calls
                        feedback = runtime.execute_tool(decision['function'], decision['arguments'])
                        save(session.output / 'calls' / f'{outer_call:06d}.json', feedback)
                    session.last_feedback = deepcopy(feedback)
                else:
                    feedback = session.execute(decision)
            save(folder / 'execution.json', feedback)
            if session.decision_archive is not None:
                session.decision_archive.register(decision_index)
                save(output / key / 'decision_index.json', session.decision_archive.index())
            histories[key].append({'role': 'user', 'content': json.dumps({'tool_feedback': feedback}, ensure_ascii=False)})
            if not pending_state():
                deliver_events()
            detail = (feedback.get('error', {}) if not feedback.get('ok') else {})
            skill_result = feedback.get('result', {})
            if (isinstance(skill_result, dict)
                    and skill_result.get('stop_reason') == 'public_transport_failure'):
                detail = (skill_result.get('last_feedback') or {}).get('error') or skill_result.get('error') or {}
                detail = {**detail, 'code': 'infrastructure_error'}
            if detail.get('code') == 'infrastructure_error':
                if detail.get('error_category') == 'request_budget_exhausted' and not pending_state():
                    _remaining(deadline)
                raise BenchHTTPError('Public function execution failed',
                    status_code=detail.get('http_status'), category=detail.get('error_category', 'invalid_response'))
    except RunDeadline:
        termination = 'client_wall_deadline'
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, TaskSetupError):
            task_setup_error = exc.public_metadata()
        if (isinstance(exc, BenchHTTPError) and exc.category == 'request_budget_exhausted'
                and deadline is not None and time.monotonic() >= deadline
                and not pending_state()):
            termination = 'client_wall_deadline'
        else:
            error = error_record(exc)
            termination = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'run_error'
    finally:
        pending = pending_state()
        if pending_read_errors:
            finalization_error = next(iter(pending_read_errors.values()))
        # Read a final result first. Cleanup never fabricates a task failure/winner.
        if clients:
            try:
                for client in clients.values():
                    client.deadline = None  # Cleanup is outside the episode action horizon.
                    # Native archives and their final audit can take over 30 s.
                    # Preserve explicit unlimited HTTP waits; finite profiles
                    # retain the existing bounded cleanup limit.
                    client.timeout_s = (None if config.http_timeout is None
                                        else minimum_limit(config.http_timeout, 30.))
                if multiplayer:
                    state = clients[keys[0]].state()
                    if not pending:
                        deliver_events(through=state['board_game']['event_cursor'] if event_delivery else None)
                    terminal, actor = _game_position(state, board)
                    if terminal:
                        result = clients[keys[0]].game_result()
                        result_source = 'server_result'
                    elif not pending:
                        reply = clients[actor].call({'request_id': 'host_finish', 'function': 'finish_episode', 'arguments': {}})
                        if not reply['ok']:
                            raise FunctionRejected(reply)
                        result = clients[actor].game_result()
                        result_source = 'cleanup_finish'
                    if not pending and result is None:
                        raise BenchHTTPError('Final game result unavailable', category='invalid_response')
                    if not pending:
                        deliver_events()
                else:
                    reply = clients['single'].call({'request_id': 'host_final_result', 'function': 'get_result', 'arguments': {}})
                    if reply['ok']:
                        result, result_source = reply['result'], 'server_result'
                    elif reply['error']['code'] == 'result_pending' and not pending:
                        reply = clients['single'].call({'request_id': 'host_finish', 'function': 'finish_episode', 'arguments': {}})
                        if not reply['ok']:
                            raise FunctionRejected(reply)
                        result, result_source = reply['result'], 'cleanup_finish'
                    elif reply['error']['code'] != 'result_pending':
                        raise FunctionRejected(reply)
            except Exception as exc:
                finalization_error = error_record(exc)
        report = {'schema': VERSION, 'episode_id': episode_id, 'board_game': board, 'turn_game': turn,
                  'ready_validated': ready_validated,
                  'task_selection': selection.metadata() if selection else None,
                  'task_setup_error': task_setup_error,
                  'configuration_choice': configuration_choice,
                  'pre_episode_model_calls': configuration_model_calls,
                  'pre_episode_model_requests': configuration_model_reports,
                  'termination': termination, 'result': result, 'result_source': result_source,
                  'run_error': error, 'error_type': error['error_type'] if error else None,
                  'finalization_error': finalization_error, 'action_recovery_pending': pending,
                  'pending_read_errors': pending_read_errors,
                  'decisions': counts, 'model_requests': model_reports,
                  'physical_attempts': {key: session.actions for key, session in sessions.items()},
                  'function_calls': {key: session.calls for key, session in sessions.items()}}
        if skill_runtimes:
            report['skill_runtimes'] = {key: runtime.public_metadata()
                                       for key, runtime in skill_runtimes.items()}
        if event_delivery is not None:
            report['public_event_cursors'] = deepcopy(event_delivery.cursors)
            if config.stop_after_commits_per_player is not None:
                counts_by_player = committed_operation_counts(event_delivery)
                report['validation_target'] = {
                    'commits_per_player': config.stop_after_commits_per_player,
                    'observed_commits': counts_by_player,
                    'reached': bool(counts_by_player) and all(
                        value >= config.stop_after_commits_per_player for value in counts_by_player.values()),
                    'source': 'verified_public_operation_committed_events',
                    'pass_counts': False, 'game_completion_claimed': False}
        if multiplayer:
            report['player_outcomes'] = turn_outcomes(result, keys) if turn else board_outcomes(result)
            health_error = error or finalization_error or {}
            report['run_status'] = error_status(health_error.get('error_type'), health_error)
            if (pending or (result is None and not health_error)
                    or (result and (not result['episode']['physics_valid']
                        or result['episode']['outcome'] in ('infrastructure_error', 'evaluator_invalid')))):
                report['run_status'] = 'infra_error'
            report['result_valid_for_scoring'] = bool(result and report['run_status'] == 'completed'
                and all(value in ('win', 'loss', 'draw') for value in report['player_outcomes'].values()))
            report['evaluation_player_outcomes'] = (report['player_outcomes'] if report['result_valid_for_scoring']
                                                    else dict.fromkeys(keys, 'unknown'))
        else:
            report.update(classify_report(report))
            if pending:
                report['run_status'] = 'infra_error'
        if episode_id is not None and not ready_validated:
            # Cleanup may return a real result for an untrusted creation receipt.
            # Keep it for audit, but it cannot certify the requested experiment.
            report['result_valid_for_scoring'] = False
            if multiplayer:
                report['evaluation_player_outcomes'] = dict.fromkeys(keys, 'unknown')
            else:
                report.update(task_outcome='unknown', task_success=None,
                              task_outcome_reason='invalid_creation_response')
                if 'task_score_valid' in report:
                    report['task_score_valid'] = False
        if task_setup_error and not pending and not finalization_error:
            report['run_status'] = 'configuration_error'
        if task_setup_error:
            if multiplayer:
                report['result_valid_for_scoring'] = False
                report['evaluation_player_outcomes'] = dict.fromkeys(keys, 'unknown')
            else:
                report.update(task_outcome='unknown', task_success=None,
                              task_outcome_reason='task_setup_invalid')
        save(output / 'run_report.json', report)
        for client in clients.values():
            client.close()
        if own_models:
            for model in models.values():
                model.close()
        if owned_bench:
            bench.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--bench-url', required=True)
    parser.add_argument('--family', choices=FAMILIES,
                        help='Select a workroom family and bind its catalog release')
    parser.add_argument('--robot-profile', choices=('original', 'l_mount'),
                        help='Bind the original or L-mount physical robot to the same task instance (default: original)')
    parser.add_argument('--task', required=True,
                        help='Canonical task ID, family-local taskN (mechanical also LN), or legacy bare card-room task1/2/3')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--player-models', type=Path,
                        help='Private JSON with complete player_0/player_1 provider, model, reasoning_effort and credentials bindings; games only')
    parser.add_argument('--robot')
    parser.add_argument('--configuration', choices=('auto', 'single_arm', 'dual_same_side', 'dual_opposed'),
                        default='auto', help='Choose the frozen robot installation before creating an episode; auto asks the model once when options are offered')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--reasoning-effort', default='max')
    parser.add_argument('--provider', choices=('gateway', 'micu', 'qwen'), default='gateway')
    parser.add_argument('--model-retries', type=int)
    parser.add_argument('--max-decisions', type=int, default=128)
    parser.add_argument('--max-actions', type=int, default=256)
    parser.add_argument('--max-action-sim-s', type=float, default=5.,
                        help='Per-action admission horizon in simulation seconds; does not change controller speed or limits')
    parser.add_argument('--stop-after-commits-per-player', type=int,
                        help='End a card-game development run after each player has this many confirmed physical operations; passes do not count and no winner is inferred')
    parser.add_argument('--max-status-queries', type=int, default=1200,
                        help='Maximum status queries per action wait, including when time budgets are disabled')
    parser.add_argument('--max-output-tokens', type=optional_output_tokens, default=4096,
                        help='Output tokens or none to omit the provider cap')
    for name, default in (('sim-budget', 120.), ('wall-budget', 1800.), ('model-timeout', 300.), ('http-timeout', 60.)):
        parser.add_argument('--' + name, type=optional_seconds, default=default,
                            help='Seconds or none to disable')
    parser.add_argument('--visual-input', choices=('rgbd',), default='rgbd')
    parser.add_argument('--history-mode', choices=('full', 'current_with_recall'), default='full',
                        help='Keep full historical text, or retain current context and recall original own-player observation/execution/depth records on demand')
    parser.add_argument('--geometry-math', action='store_true',
                        help='Expose pure local geometry computation functions; no automatic motion or private state')
    parser.add_argument('--max-function-calls', type=int,
                        help='Per-learner public calls including calls inside learned skills; defaults to max-decisions')
    parser.add_argument('--skill-bindings', type=Path,
                        help='Host-owned agent_skill_bindings.v1 library/phase/pin bindings')
    parser.add_argument('--practice-side', choices=('left', 'right'))
    parser.add_argument('--practice-case')
    parser.add_argument('--practice-split', choices=('train', 'validation'))
    args = parser.parse_args()
    from scripts.maintenance.freeze_project_sources import verify_runtime_imports
    root = args.project_root.resolve()
    verification = verify_runtime_imports(root, required_modules=(
        'agent.model_client.function_runner', 'agent.model_client.function_http',
        'agent.model_client.function_session', 'agent.model_client.scene_tasks',
        'scenes.workrooms.catalog'), entrypoint_file=Path(__file__))
    output = args.output.expanduser().absolute()
    if output.resolve() != output or not output.is_relative_to(OUTPUT_ROOT):
        parser.error(f'Use a new, non-aliased output directory under {OUTPUT_ROOT}')
    values = vars(args).copy()
    values.pop('project_root')
    practice = {key: values.pop('practice_' + suffix) for key, suffix in
                (('active_side', 'side'), ('case_id', 'case'), ('split', 'split'))}
    if any(value is not None for value in practice.values()):
        if any(value is None for value in practice.values()):
            parser.error('Practice requires --practice-side, --practice-case and --practice-split together')
        values['practice'] = practice
    report = run(RunConfig(**values), source_verification=verification)
    print(json.dumps({'episode_id': report['episode_id'], 'run_status': report['run_status'],
                      'termination': report['termination'], 'task_setup_error': report['task_setup_error']}, ensure_ascii=False))
    if report['task_setup_error'] is not None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
