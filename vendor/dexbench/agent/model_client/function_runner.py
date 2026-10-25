"""Public function Agent loop for ordinary episodes and authenticated games.

The trusted host owns tokens, action identities and turn scheduling. Models see
only public state and function schemas. This runner executes JSON calls, not
generated Python or an RSI training scheduler.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

from bench.paths import OUTPUT_ROOT
from bench.protocol.board_games import BoardGameResult, PlayerSession
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
from .scene_tasks import FAMILIES, TaskSetupError, published_task, select_task
from .time_limits import minimum_limit, optional_positive, optional_seconds
from .transport import ModelGatewayConfig, ModelGatewayError


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
not proof that an unfinished wrist target converged. In tower extraction, place
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


def resolve_task(name, tasks, family=None):
    """Family-local aliases select only compatible, publicly offered sessions."""
    return published_task(select_task(name, family), tasks)


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
    checked = BoardGameResult.model_validate(result)
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
            or ('seed' in request and spec.seed != request['seed'])):
        raise ValueError('Created episode does not match the requested contract')
    if (not board and not turn
            and (task.get('public_rules_sha256') is not None or task.get('scoring_contract') is not None)
            and not isinstance(ready.get('task'), dict)):
        raise BenchHTTPError('Created episode omitted its declared public task', category='invalid_response')
    actual = deepcopy(ready.get('task', task))
    if actual.get('task_id') != spec.task_id:
        raise ValueError('Episode task identity differs')
    if (not board and not turn and task.get('public_rules_sha256') is not None
            and actual.get('public_rules_sha256') != task['public_rules_sha256']):
        raise ValueError('Created episode rules differ from the published task')
    if (not board and not turn and actual.get('scoring_contract') != task.get('scoring_contract')):
        raise ValueError('Created episode scoring differs from the published task')
    if spec.task_id == 'task0':
        goal = actual.get('goal', {})
        if (actual.get('episode_seed') != spec.seed or len(goal.get('target_cells', [])) != 3
                or len(goal.get('target_positions_m', [])) != 3):
            raise ValueError('Task0 requires its episode-specific public targets')
    if board or turn:
        _players(ready, turn=turn)
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


def run(config: RunConfig, *, bench=None, models=None, source_verification=None):
    """Run one episode. Tests can inject CPU transports and scripted models."""
    if type(config.max_decisions) is not int or not 1 <= config.max_decisions <= 256:
        raise ValueError('max_decisions must be between 1 and 256 per player')
    if config.visual_input != 'rgbd':
        raise ValueError('Public function runner currently requires the full RGB-D input condition')
    if config.seed is not None and (type(config.seed) is not int or not 0 <= config.seed < 2**32):
        raise ValueError('seed must be an unsigned 32-bit integer')
    for name in ('wall_budget', 'sim_budget', 'model_timeout', 'http_timeout'):
        optional_positive(getattr(config, name), name)
    optional_positive(config.wall_budget, 'wall_budget', maximum=MAX_EPISODE_WALL_TIME_S)
    optional_positive(config.sim_budget, 'sim_budget', maximum=MAX_EPISODE_SIM_TIME_S)
    if type(config.max_actions) is not int or not 1 <= config.max_actions <= MAX_EPISODE_ACTIONS:
        raise ValueError(f'max_actions must be between 1 and {MAX_EPISODE_ACTIONS}')
    if type(config.max_status_queries) is not int or config.max_status_queries < 1:
        raise ValueError('max_status_queries must be a positive integer')
    if config.max_output_tokens is not None and (type(config.max_output_tokens) is not int
            or not 1 <= config.max_output_tokens <= 32768):
        raise ValueError('max_output_tokens must be null or between 1 and 32768')
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    if source_verification is not None:
        save(output / 'source_verification.json', source_verification)
    owned_bench = bench is None
    bench = bench or BenchHTTPClient(config.bench_url, timeout_s=config.http_timeout)
    own_models = models is None
    models = {} if models is None else models
    sessions, clients, histories, counts, model_reports = {}, {}, {}, {}, {}
    result = None
    board = False
    turn = False
    multiplayer = False
    keys = ('single',)
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
        selection = select_task(config.task, config.family)
        save(output / 'task_selection.json', selection.metadata())
        selection.require_runnable()
        task, board = published_task(selection, bench.list_tasks())
        turn = task.get('task_kind') == 'turn_game'
        multiplayer = board or turn
        keys = ('player_0', 'player_1') if multiplayer else ('single',)
        robot = config.robot or ('fr3_sharpa_instances' if turn else
                                 'dual_fr3_sharpa_w01' if board else 'sharpa_w01_right')
        controller = PolicyDescriptor.model_validate(select_geometry_policy(
            bench.list_policies(task_id=task['task_id']), robot, allow_instances=turn)).model_copy(deep=True)
        policy = controller.model_dump(mode='json')
        budgets = Budgets(sim_time_s=config.sim_budget, wall_time_s=config.wall_budget,
                          max_actions=config.max_actions, max_model_calls=config.max_decisions * len(keys),
                          max_action_sim_s=5.).model_dump(mode='json')
        if own_models:
            credentials = (ModelGatewayConfig.from_file(config.credentials) if config.credentials
                           else ModelGatewayConfig.from_env())
            for key in keys:
                models[key] = configured_model(config, credentials)
        if set(models) != set(keys) or len({id(model) for model in models.values()}) != len(keys):
            raise ValueError('Supply one separate model client per player/context')
        settings = {'protocol': VERSION, 'task_id': task['task_id'],
                    'mode': 'turn_game' if turn else 'board_game' if board else 'single_agent',
                    'task_selection': selection.metadata(),
                    'source_verification_sha256': canonical_hash(source_verification) if source_verification else None,
                    'controller': policy, 'budgets': budgets, 'max_decisions_per_player': config.max_decisions,
                    'action_recovery': {'max_status_queries_per_wait': config.max_status_queries},
                    'visual_input': config.visual_input, 'tactile_input': 'on',
                    'models': {key: model.public_metadata() for key, model in models.items()},
                    'learning_policy': learning_policy(task['task_id'], selection=selection), 'system_prompt': PROMPT,
                    'history': 'independent_full_game_text_current_images_only'}
        save(output / 'agent_config.json', settings)
        request = {'task_id': task['task_id'], 'robot_id': robot, 'agent_id': 'public_function_agent',
                   'agent_version': VERSION, 'agent_config_sha256': canonical_hash(settings),
                   'track': 'system0', 'observation_profile': 'vision_proprio', 'budgets': budgets,
                   'policy_id': policy['policy_id'], 'policy_version': policy['version']}
        if config.seed is not None:
            request['seed'] = config.seed
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
                    client=bench.client, timeout_s=config.http_timeout, spec=created)
        # Creation is one POST. A lost response is never retried as a new episode.
        if multiplayer:
            create = create_turn_game if turn else create_board_game
            ready = create(bench, request, on_created=bind_created)
        else:
            ready = bench.create(request)
            bind_created(ready)
        spec, task = _validated_ready(ready, request, task, board, turn, controller=controller)
        selection.verify_release(spec)
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
                trial_id=key, max_actions=config.max_actions, max_calls=config.max_decisions, deadline=deadline,
                max_status_queries=config.max_status_queries)
            sessions[key].initialize()
            histories[key], counts[key], model_reports[key] = [], 0, []
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
                terminal, key = _game_position(state, board)
                if terminal:
                    result = clients[keys[0]].game_result()
                    result_source, termination = 'server_result', 'server_finished'
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
            context = session.context()
            if multiplayer and _game_position(context['state'], board) != (False, key):
                continue
            folder = output / key / 'decisions' / f'{counts[key]:04d}'
            frames = capture(clients[key], episode_id, context['state']['observation'], folder, output,
                             visual_input=config.visual_input, tactile_input='on')
            context.update(task=task_context, learning_policy=settings['learning_policy'],
                           depth_observation=frames['depth_context'], decisions_remaining=config.max_decisions - counts[key])
            context['decision_schema']['one_of'].append({'stop': True})
            # Keep only public text in historical turns. Current RGB-D is attached once.
            prompt_text = json.dumps(context, ensure_ascii=False, allow_nan=False)
            turns = histories[key] + [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}]
                                                                  + frames['image_content']}]
            save(folder / 'model_request.json', {'system_prompt': PROMPT,
                 'messages': histories[key] + [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}]
                                                                           + frames['recorded_image_content']}],
                 'expanded_messages_sha256': canonical_hash(turns)})
            remaining = _remaining(deadline)
            decision_index = counts[key]
            counts[key] += 1
            kwargs = {'max_output_tokens': config.max_output_tokens,
                      'timeout_s': minimum_limit(config.model_timeout, remaining)}
            if config.provider == 'micu':
                kwargs['retry_deadline'] = deadline
                request_sha256 = canonical_hash(turns)

                def record_model_event(event):
                    # Micu emits typed public fields, never provider error text,
                    # headers or credentials. Persist before waiting/retrying.
                    entry = {'player': key, 'decision_index': decision_index,
                             'request_sha256': request_sha256, **event}
                    with (folder / 'model_api_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    if event['event'] in ('model_request_started', 'model_request_completed',
                                           'model_request_failed', 'model_retry_wait'):
                        print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

                kwargs['on_event'] = record_model_event
            try:
                response = models[key].generate(PROMPT, turns, **kwargs)
            except ModelGatewayError as exc:
                save(folder / 'model_error.json', {'decision_index': decision_index, **exc.public_metadata()})
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
                feedback = session.execute(decision)
            save(folder / 'execution.json', feedback)
            histories[key].append({'role': 'user', 'content': json.dumps({'tool_feedback': feedback}, ensure_ascii=False)})
            if not feedback.get('ok') and feedback.get('error', {}).get('code') == 'infrastructure_error':
                detail = feedback['error']
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
                    client.deadline = None  # cleanup has its own bounded HTTP timeout
                    client.timeout_s = minimum_limit(config.http_timeout, 30.)
                if multiplayer:
                    state = clients[keys[0]].state()
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
                  'termination': termination, 'result': result, 'result_source': result_source,
                  'run_error': error, 'error_type': error['error_type'] if error else None,
                  'finalization_error': finalization_error, 'action_recovery_pending': pending,
                  'pending_read_errors': pending_read_errors,
                  'decisions': counts, 'model_requests': model_reports,
                  'physical_attempts': {key: session.actions for key, session in sessions.items()},
                  'function_calls': {key: session.calls for key, session in sessions.items()}}
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
    parser.add_argument('--task', required=True,
                        help='Canonical task ID, family-local taskN (mechanical also LN), or legacy bare card-room task1/2/3')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--robot')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--reasoning-effort', default='max')
    parser.add_argument('--provider', choices=('gateway', 'micu'), default='gateway')
    parser.add_argument('--model-retries', type=int)
    parser.add_argument('--max-decisions', type=int, default=128)
    parser.add_argument('--max-actions', type=int, default=256)
    parser.add_argument('--max-status-queries', type=int, default=1200,
                        help='Maximum status queries per action wait, including when time budgets are disabled')
    parser.add_argument('--max-output-tokens', type=optional_output_tokens, default=4096,
                        help='Output tokens or none to omit the provider cap')
    for name, default in (('sim-budget', 120.), ('wall-budget', 1800.), ('model-timeout', 300.), ('http-timeout', 60.)):
        parser.add_argument('--' + name, type=optional_seconds, default=default,
                            help='Seconds or none to disable')
    parser.add_argument('--visual-input', choices=('rgbd',), default='rgbd')
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
    report = run(RunConfig(**values), source_verification=verification)
    print(json.dumps({'episode_id': report['episode_id'], 'run_status': report['run_status'],
                      'termination': report['termination'], 'task_setup_error': report['task_setup_error']}, ensure_ascii=False))
    if report['task_setup_error'] is not None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
