"""Actual host/scene/frozen-referee termination wiring; no native physics."""
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from bench.isaacsim.release import digest, load_evaluator
from bench.paths import load_json
from bench.protocol.turn_games import TurnGameState
from bench.runtime.turn_games import TurnGameSession
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.living_room.evaluation import SequenceEvaluator
from scenes.workrooms.living_room.runtime import LivingTaskRuntime
from test_agent_bench_contracts import result, spec


PROFILES = ('original', 'l_mount')


@lru_cache(None)
def frozen_inputs(robot_profile='original'):
    # The host-termination derivation is recorded on the original-robot
    # release; the L-mount profile is a robot-only revision.
    reference = task_definition('living_room', 'tower_extraction', robot_profile=robot_profile)['source_release']
    root = Path(reference['path'])
    assert digest(root/'release_manifest.json') == reference['manifest_sha256']
    return root, load_json(root/'task.json')


def runtime_for(task, factory):
    runtime = LivingTaskRuntime.__new__(LivingTaskRuntime)
    runtime.task = deepcopy(task)
    runtime.evaluator = factory(task)
    runtime.env = SimpleNamespace(time_s=0.)
    runtime.events = io.StringIO()
    return runtime


@pytest.mark.parametrize('robot_profile', PROFILES)
@pytest.mark.parametrize('frozen', [False, True], ids=['source', 'current_frozen'])
@pytest.mark.parametrize('outcome,reason,status', [
    ('sim_timeout', 'time_limit', 'incomplete'),
    ('wall_timeout', 'time_limit', 'incomplete'),
    ('action_budget', 'step_limit', 'incomplete'),
    ('model_budget', 'step_limit', 'incomplete'),
    ('cancelled', 'cancelled', 'abort'),
    ('infrastructure_error', 'runtime_error', 'abort'),
])
def test_host_result_preserves_interruption_without_inventing_winner(frozen, outcome, reason, status, robot_profile):
    root, task = frozen_inputs(robot_profile)
    runtime = runtime_for(task, load_evaluator(root) if frozen else SequenceEvaluator)
    commands = []
    def request(operation, command, **kwargs):
        assert operation == 'turn_game'
        commands.append(command)
        return runtime.request(command, **kwargs)
    session = SimpleNamespace(worker=SimpleNamespace(request=request, broken=False),
        raw={'sim_time_s': 0.}, score=runtime.score(), lock=threading.RLock(), result=None)
    facade = TurnGameSession.__new__(TurnGameSession)
    facade.session = session
    facade.state = TurnGameState.model_validate(runtime.public_state())
    facade.end(outcome)
    session.result = result(outcome=outcome, spec=spec(task_id='tower_extraction'))
    public = facade.result('player_0')
    state = public['turn_game']
    assert commands == [{'type': 'end_episode', 'reason': reason}]
    assert public['episode']['outcome'] == outcome
    assert state['status'] == 'interrupted'
    assert state['terminal_reason'] == reason and state['violation_reason'] is None
    assert state['public_progress']['status'] == status
    assert state['winner_player_id'] is None and state['loser_player_id'] is None
    assert state['public_rules']['host_termination'][reason] == status
    before = runtime.public_state()
    runtime.request({'type': 'end_episode', 'reason': 'runtime_error'})
    assert runtime.public_state() == before


@pytest.mark.parametrize('reason,legacy', [
    ('time_limit', 'timeout'), ('step_limit', 'budget'),
    ('cancelled', 'adapter_failure'), ('runtime_error', 'adapter_failure'),
])
def test_historical_frozen_referee_remains_usable(reason, legacy):
    root, _ = frozen_inputs()
    manifest = load_json(root/'release_manifest.json')
    previous = manifest['host_termination_derivation']
    old = Path(previous['source_release'])
    assert digest(old/'release_manifest.json') == previous['source_manifest_sha256']
    task = load_json(old/'task.json')
    assert 'host_termination' not in task['public_rules']
    runtime = runtime_for(task, load_evaluator(old))
    public = runtime.request({'type': 'end_episode', 'reason': reason})['turn_game']
    assert public['terminal_reason'] == legacy
    assert public['winner_player_id'] is None and public['loser_player_id'] is None


@pytest.mark.parametrize('robot_profile', PROFILES)
@pytest.mark.parametrize('reason', ['cancelled', 'runtime_error', 'budget', 'reset'])
def test_worker_shutdown_preserves_reason_and_cannot_overwrite_it(tmp_path, reason, robot_profile):
    root, task = frozen_inputs(robot_profile)
    runtime = runtime_for(task, load_evaluator(root))
    runtime.closed = False
    runtime.env.world = SimpleNamespace(step=None)
    runtime.original = lambda: None
    runtime.exchange = None
    runtime.archive = io.StringIO()
    runtime.output = tmp_path
    runtime.close(reason=reason)
    assert runtime.evaluator.game.reason == reason
    before = runtime.public_state()
    runtime.close(reason='runtime_error')
    assert runtime.public_state() == before


@pytest.mark.parametrize('robot_profile', PROFILES)
def test_unknown_host_reason_cannot_manufacture_a_game_result(robot_profile):
    root, task = frozen_inputs(robot_profile)
    runtime = runtime_for(task, load_evaluator(root))
    before = runtime.public_state()
    with pytest.raises(ValueError, match='Unknown host termination'):
        runtime.request({'type': 'end_episode', 'reason': 'won'})
    assert runtime.public_state() == before


def test_termination_release_preserves_physics_reset_and_old_release():
    root, task = frozen_inputs()
    manifest = load_json(root/'release_manifest.json')
    reference = manifest['host_termination_derivation']
    old = Path(reference['source_release'])
    original = load_json(old/'release_manifest.json')
    assert digest(old/'release_manifest.json') == reference['source_manifest_sha256']
    allowed = {'scene.json', 'task.json', 'rules.json', 'source_provenance.json',
               'asset_file_hashes.json', 'evaluator/tasks/tower_extraction/rules.py',
               'evaluator/tasks/tower_extraction/evaluator.py'}
    for relative in original['files_sha256']:
        if relative not in allowed:
            assert digest(root/relative) == original['files_sha256'][relative]
    old_scene = load_json(old/'scene.json')
    new_scene = load_json(root/'scene.json')
    for layer in old_scene['task_layers']:
        if layer['usd_path'] == str(old/'robot_stands.usda'):
            layer['usd_path'] = str(root/'robot_stands.usda')
    assert new_scene == old_scene
    before = load_json(old/'task.json')
    previous_rules = before.pop('public_rules')
    after = deepcopy(task)
    current_rules = after.pop('public_rules')
    previous = before['evaluation'].pop('reference_configuration')
    current = after['evaluation'].pop('reference_configuration')
    assert before == after
    for value in (previous, current):
        value.pop('configuration_sha256')
        value.pop('rules')
    assert previous == current
    permitted = {'schema', 'host_termination', 'host_termination_scope', 'incomplete_without_winner'}
    assert {k: v for k, v in previous_rules.items() if k not in permitted} == {
        k: v for k, v in current_rules.items() if k not in permitted}
    # The L-mount profile wraps exactly this release and keeps its rules;
    # only the robot reset coordinates belong to the new installation.
    revised, revised_task = frozen_inputs('l_mount')
    revision = load_json(revised/'release_manifest.json')
    assert revision['base_release'] == {'path': str(root), 'manifest_sha256': digest(root/'release_manifest.json'),
                                        'schema': manifest['schema']}
    assert {k: v for k, v in revised_task.items() if k != 'initial_state'} == {
        k: v for k, v in task.items() if k != 'initial_state'}
