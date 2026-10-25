"""Scene selection and real frozen-release HTTP contracts with CPU doubles.

These checks exercise discovery, isolation and lifecycle, not robot manipulation
or native physical qualification. Frozen scenes and shared assets stay unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import (
    RunConfig, learning_policy, resolve_task, run, task_requirements_context,
)
from agent.model_client.scene_tasks import TaskSetupError, list_scene_tasks, select_task
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import digest, load_evaluator, task_inputs
from bench.paths import load_json
from bench.protocol.task_referee import project_task_rules
from bench.runtime.manipulation_tasks import project_manipulation_task
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.runtime import terminal_scoring_contract
from scenes.workrooms.catalog import load_task, task_definition
from test_agent_service import FakeWorker, PRIVATE_MARKER, publish_targets
from test_function_runner import ScriptedModel


MECHANICAL = (
    'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
    'fastener_assembly', 'gear_train_assembly',
)
NIST_MECHANICAL = ('nist_connector_insertion', 'nist_belt_drive_assembly',
                   'nist_cable_routing', 'nist_wire_harness')
LIVING = ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction')
STUDY = ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving')
KITCHEN = ('breakfast_serving', 'produce_transfer', 'utensil_sorting', 'pot_lid_placement',
           'pantry_stocking', 'table_clearing')
LABORATORY = ('sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer')
BOARD = ('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7')


@pytest.mark.parametrize('number,canonical', tuple(enumerate(MECHANICAL + NIST_MECHANICAL, 1)))
def test_mechanical_numbers_are_family_scoped_and_preserve_release_status(number, canonical):
    for alias in (f'task{number}', f'L{number}', canonical):
        selected = select_task(alias, family='mechanical_assembly')
        assert (selected.family, selected.task_id, selected.task_number) == (
            'mechanical_assembly', canonical, number)
        assert selected.session_kind == 'single_agent'
        assert selected.status == 'implemented'
        assert selected.source_release == task_definition('mechanical_assembly', canonical)['source_release']
        assert selected.metadata()['task_id'] == canonical
        assert selected.adapter_available
        selected.require_runnable()
    assert not select_task(canonical).pin_release


@pytest.mark.parametrize('number,canonical', tuple(enumerate(LIVING, 1)))
def test_living_numbers_preserve_release_status_and_player_semantics(number, canonical):
    selected = select_task(f'task{number}', family='living_room')
    assert (selected.family, selected.task_id, selected.task_number) == ('living_room', canonical, number)
    assert selected.status == 'implemented'
    assert selected.source_release == task_definition('living_room', canonical)['source_release']
    assert selected.session_kind == ('two_player_adversarial' if number == 4 else 'single_agent')
    assert selected.adapter_available
    selected.require_runnable()
    if number == 4:
        assert selected.players == ('player_0', 'player_1')
    else:
        assert selected.players == ('single',)


@pytest.mark.parametrize('family,number,canonical', [
    *(('study', number, task) for number, task in enumerate(STUDY, 1)),
    *(('kitchen', number, task) for number, task in enumerate(KITCHEN, 1)),
    *(('laboratory', number, task) for number, task in enumerate(LABORATORY, 1)),
])
def test_single_agent_family_numbers_pin_their_catalog_releases(family, number, canonical):
    selected = select_task(f'task{number}', family=family)
    assert (selected.family, selected.task_id, selected.task_number) == (family, canonical, number)
    assert selected.status == 'implemented' and selected.session_kind == 'single_agent'
    assert selected.adapter_available and selected.players == ('single',)
    selected.require_runnable()
    assert selected.source_release == task_definition(family, canonical)['source_release']
    assert selected.pin_release and selected.metadata()['release_binding'] == 'catalog_pinned'
    assert select_task(canonical, family=family) == selected
    bare = select_task(canonical)
    assert bare.family == family and not bare.pin_release


@pytest.mark.parametrize('number,canonical', tuple(enumerate(LIVING[:2], 1)))
def test_living_candidates_use_single_agent_path_and_frozen_release_binding(number, canonical):
    selected = select_task(f'task{number}', family='living_room')
    selected.require_runnable()
    verified = load_task('living_room', canonical)
    release = Path(selected.source_release['path'])
    manifest_sha256 = digest(release/'release_manifest.json')
    assert selected.session_kind == 'single_agent' and selected.players == ('single',)
    assert selected.pin_release and selected.metadata()['release_binding'] == 'catalog_pinned'
    assert manifest_sha256 == selected.source_release['manifest_sha256']
    assert verified['definition']['source_release'] == selected.source_release
    selected.verify_release({'task_release_sha256': manifest_sha256})
    with pytest.raises(TaskSetupError) as error:
        selected.verify_release({'task_release_sha256': '0'*64})
    assert error.value.code == 'release_mismatch'
    episode_task = task_inputs(release, canonical, seed=7)[2]
    published = project_manipulation_task(episode_task)
    task, board = resolve_task(f'task{number}', [published], family='living_room')
    assert task == published and board is False
    assert task['create_endpoint'] == 'POST /episodes'
    if 'referee' in episode_task['public_rules']:
        # A frozen referee contract replaces the seeded goal projection.
        assert published == project_task_rules(episode_task) and 'episode_seed' not in task
    else:
        assert task['episode_seed'] == 7
    # Publication is executable-candidate availability, not robot task credit.
    manifest = load_json(release/'release_manifest.json')
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False and manifest['task_policy_success_claimed'] is False


@pytest.mark.parametrize('number,canonical', tuple(enumerate(BOARD, 1)))
def test_legacy_numbered_board_aliases_keep_their_original_family(number, canonical):
    selected = select_task(f'task{number}')
    assert (selected.family, selected.task_id, selected.session_kind) == ('card_room', canonical, 'board_game')
    assert selected.players == ('player_0', 'player_1')
    task, board = resolve_task(f'task{number}', [{'task_id': canonical, 'task_kind': 'board_game'}])
    assert task['task_id'] == canonical and board


@pytest.mark.parametrize('number', range(4, 10))
def test_bare_task_number_requires_a_family_instead_of_guessing(number):
    with pytest.raises(TaskSetupError) as error:
        select_task(f'task{number}')
    assert error.value.code == 'family_required'


@pytest.mark.parametrize('family,canonical', [
    ('mechanical_assembly', 'peg_insertion'), ('living_room', 'water_pouring'),
    ('living_room', 'tower_extraction'), ('card_room', 'connect_four_5x5'),
    ('study', 'book_retrieval'),
    *(('kitchen', task) for task in KITCHEN),
    *(('mechanical_assembly', task) for task in NIST_MECHANICAL),
    *(('laboratory', task) for task in LABORATORY),
])
def test_canonical_names_identify_the_family_without_reassigning_numbers(family, canonical):
    selected = select_task(canonical)
    assert selected.family == family and selected.task_id == canonical


def test_scene_listing_retains_unavailable_tasks_and_the_tower_opponent():
    mechanical = {row['task_id']: row for row in list_scene_tasks('mechanical_assembly')}
    living = {row['task_id']: row for row in list_scene_tasks('living_room')}
    assert set(mechanical) == set(MECHANICAL) | set(NIST_MECHANICAL)
    assert set(living) == set(LIVING)
    assert all(mechanical[key]['status'] == 'implemented' for key in MECHANICAL)
    for key in NIST_MECHANICAL:
        definition = task_definition('mechanical_assembly', key)
        assert mechanical[key]['status'] == definition['status']
        assert mechanical[key]['runnable'] == (definition['status'] == 'implemented')
    assert all(row['status'] == 'implemented' for row in living.values())
    assert all(living[task]['runnable'] for task in LIVING)
    assert living['tower_extraction']['session_kind'] == 'two_player_adversarial'
    assert tuple(living['tower_extraction']['players']) == ('player_0', 'player_1')
    assert living['tower_extraction']['adapter_available'] is True
    assert living['tower_extraction']['cannot_run_reason'] is None
    for family, tasks in (('study', STUDY), ('kitchen', KITCHEN), ('laboratory', LABORATORY)):
        rows = {row['task_id']: row for row in list_scene_tasks(family)}
        assert set(rows) == set(tasks)
        assert all(row['runnable'] and row['session_kind'] == 'single_agent' for row in rows.values())


@pytest.mark.parametrize('family,number', [
    ('mechanical_assembly', 2), ('mechanical_assembly', 3),
    ('living_room', 2), ('living_room', 3), ('living_room', 4),
    ('study', 2), ('study', 3),
    *(('kitchen', number) for number in range(1, 7)),
    *(('laboratory', number) for number in range(1, 6)),
    *(('mechanical_assembly', number) for number in range(6, 10)),
])
def test_noncard_task_numbers_do_not_inherit_frozen_competition_learning(family, number):
    policy = learning_policy(f'task{number}', family=family)
    assert policy['practice_rsi_allowed']
    assert not policy['frozen_skills_required']
    assert not policy['skill_mutation_tools_available']
    assert not policy['rsi_scheduler_implemented']
    if number in (2, 3):
        card = learning_policy(f'task{number}', family='card_room')
        assert card['frozen_skills_required'] and not card['practice_rsi_allowed']


@pytest.mark.parametrize('number,canonical', tuple(enumerate(MECHANICAL[:4], 1)))
def test_family_resolution_uses_the_selected_services_published_task(number, canonical):
    published = [{'task_id': BOARD[0], 'task_kind': 'board_game'}, {'task_id': canonical}]
    task, board = resolve_task(f'task{number}', published, family='mechanical_assembly')
    assert task['task_id'] == canonical and board is False
    with pytest.raises(ValueError):
        resolve_task(f'task{number}', published[:1], family='mechanical_assembly')
    with pytest.raises(ValueError):
        resolve_task(f'task{number}', [{'task_id': canonical, 'task_kind': 'board_game'}],
                     family='mechanical_assembly')


@pytest.mark.parametrize('kind', ['board_game', 'manipulation', 'single_agent', None])
def test_tower_requires_the_authenticated_turn_contract(kind):
    row = {'task_id': 'tower_extraction'}
    if kind is not None:
        row['task_kind'] = kind
    with pytest.raises(TaskSetupError) as error:
        resolve_task('task4', [row], family='living_room')
    assert error.value.code == 'session_contract_mismatch'
    with pytest.raises(TaskSetupError) as error:
        resolve_task('tower_extraction', [row])
    assert error.value.code == 'session_contract_mismatch'


def test_tower_turn_route_cannot_be_replaced_with_episode_or_board_creation():
    row = {'task_id': 'tower_extraction', 'task_kind': 'turn_game', 'create_endpoint': 'POST /turn-games'}
    task, board = resolve_task('task4', [row], family='living_room')
    assert task == row and board is False
    for endpoint in ('POST /episodes', 'POST /board-games'):
        with pytest.raises(TaskSetupError) as error:
            resolve_task('task4', [dict(row, create_endpoint=endpoint)], family='living_room')
        assert error.value.code == 'session_contract_mismatch'


class NoContactBench:
    """Record attempts even when the runner catches their AssertionError."""
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f'Unavailable task must be rejected before bench access: {name}')


@pytest.mark.parametrize('family,alias,canonical', [
    ('laboratory', 'cap_placement', 'cap_placement'),
])
def test_unavailable_run_stops_before_bench_or_model_and_records_the_selected_task(tmp_path, monkeypatch, family, alias, canonical):
    from agent.model_client import scene_tasks
    from scenes.workrooms.laboratory.catalog_definition import task_definition as planned_definition
    planned = planned_definition(canonical)
    planned["runnable"] = False
    monkeypatch.setattr(scene_tasks, "list_tasks", lambda *args, **kwargs: [planned])
    bench = NoContactBench()
    model = ScriptedModel([{'stop': True}])
    report = run(RunConfig('http://unused', alias, tmp_path / 'run', family=family),
                 bench=bench, models={'single': model})
    assert bench.calls == [] and model.turns == []
    assert report['episode_id'] is None and report['result'] is None
    assert report['task_setup_error']['code']
    assert report['task_selection']['task_id'] == canonical
    assert report['task_selection']['status'] == 'planned'
    assert report['task_setup_error']['code'] == 'task_planned'
    assert report['run_status'] == 'configuration_error'
    assert not (tmp_path / 'run/create_request.json').exists()


def reset_score(release, task):
    """The frozen scorer's unscored reset state, as the native worker reports it."""
    evaluator = load_evaluator(release)(task)
    score = evaluator.inspect_conditions()
    if 'referee' in task.get('public_rules', {}) and 'referee' not in score:
        # Observation-driven scorers (living Task1 v7) publish referee state
        # only after the native observer's reset baseline; replay that
        # baseline through the frozen evaluator instead of inventing a state.
        baseline = {'schema': evaluator.state_schema, 'configuration_sha256': evaluator.digest,
                    'complete': True, 'errors': [], 'events': [], 'baseline': True,
                    'time_s': 0., 'native_step': 0, 'released_stable_s': 0.,
                    'television': evaluator.television.state(0., 0.), 'terminal_failure': None}
        score = evaluator.update({'time_s': 0., 'task_state': baseline})
        assert score['reason'] == 'reset_baseline_not_scored'
    return score


def mechanical_service(tmp_path, task_id, family='mechanical_assembly'):
    """Use actual frozen task/robot contracts, replacing only the native worker."""
    verified = load_task(family, task_id)
    release = Path(verified['definition']['source_release']['path'])
    workers = []

    def factory(**kwargs):
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        _, scene, task, initial = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])
        # The CPU double reports the frozen scorer's unscored state; it does
        # not fabricate mechanical evidence while testing Agent transport.
        worker.score.update(reset_score(kwargs['release'], task))
        if family in ('laboratory', 'kitchen'):
            # Native poses, signed contacts and material inventories are private,
            # including reduced-liquid state and intermediate scorer details.
            worker.raw['task_state'] = {'detail': {PRIVATE_MARKER: 'native_inventory'}}
            worker.raw['reduced_dynamics'] = {PRIVATE_MARKER: 'liquid_ledger'}
            worker.score['detail'] = {PRIVATE_MARKER: 'mass_and_volume_readout'}
        worker.initial['score'] = deepcopy(worker.score)
        if task.get('schema_version') == 'living-room-manipulation-task-v1':
            worker.initial['task'] = project_manipulation_task(task)
        else:
            projection = project_task_rules(task, scoring_contract=terminal_scoring_contract(task))
            if projection is not None:
                worker.initial['task'] = projection
        geometry = public_robot_geometry(robot_configuration(scene))
        worker.arm, worker.hand, worker.active = list(geometry.arm.names), list(geometry.hand.names), list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.initial['observation'] = deepcopy(worker.raw)
        worker.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm, 'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        workers.append(worker)
        return worker

    api = AgentService(tmp_path / 'episodes', release=release, registry=default_registry(release),
                       worker_factory=factory)
    return api, workers, verified


@pytest.mark.parametrize('family,alias,task_id', [
    *(('mechanical_assembly', f'L{number}', task) for number, task in enumerate(MECHANICAL + NIST_MECHANICAL, 1)),
    *(('living_room', f'task{number}', task) for number, task in enumerate(LIVING[:3], 1)),
    *(('study', f'task{number}', task) for number, task in enumerate(STUDY, 1)),
    *(('kitchen', f'task{number}', task) for number, task in enumerate(KITCHEN, 1)),
    *(('laboratory', f'task{number}', task) for number, task in enumerate(LABORATORY, 1)),
])
def test_ordinary_frozen_releases_observe_hold_and_stop_through_public_http(tmp_path, family, alias, task_id):
    api, workers, verified = mechanical_service(tmp_path, task_id, family)
    refereed = 'referee' in verified['task']['public_rules']
    decisions = [{'function': 'observe', 'arguments': {}},
                 {'function': 'hold', 'arguments': {'duration_s': .04}}, {'stop': True}]
    if refereed:
        decisions.insert(0, {'function': 'get_task', 'arguments': {}})
        decisions.insert(-1, {'function': 'get_task', 'arguments': {}})
    model = ScriptedModel(decisions)
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        published = bench.list_tasks()
        assert [row['task_id'] for row in published] == [task_id]
        if family == 'living_room' and task_id == 'tower_building':
            # Ordinary discovery currently omits kind; also accept an explicit
            # projection of the real frozen single_agent manifest contract.
            manifest = load_json(api.release / 'release_manifest.json')
            assert manifest['task_kind'] == 'single_agent'
            explicit = dict(published[0], task_kind=manifest['task_kind'])
            selected, board = resolve_task(alias, [explicit], family=family)
            assert selected == explicit and board is False
            with pytest.raises(TaskSetupError):
                resolve_task(alias, [dict(explicit, task_kind='turn_game')], family=family)
        report = run(RunConfig('http://testserver', alias, tmp_path / 'run',
                              family=family, seed=0, max_decisions=6,
                              # A disabled client horizon permits the frozen
                              # 240 s referee limit without changing the shared
                              # 150 s finite-request admission ceiling.
                              sim_budget=None if family == 'kitchen' and task_id in KITCHEN[4:] else 120.,
                              max_status_queries=100000), bench=bench, models={'single': model})
        if family != 'mechanical_assembly':
            create = load_json(tmp_path / 'run/create_request.json')
            for route in ('/board-games', '/turn-games'):
                assert http.post(route, json=create).status_code == 403
            assert len(workers) == 1  # Rejected before native creation.
    assert report['error_type'] is None, report
    assert report['task_setup_error'] is None
    assert report['board_game'] is False and report['run_status'] == 'completed'
    assert report['turn_game'] is False
    assert report['termination'] == 'model_requested_stop'
    assert report['decisions'] == {'single': len(decisions)}
    assert report['physical_attempts'] == {'single': 1}
    assert report['task_selection']['task_id'] == task_id
    request = load_json(tmp_path / 'run/create_request.json')
    ready = load_json(tmp_path / 'run/episode_ready.json')
    assert request['seed'] == ready['spec']['seed'] == 0
    if family == 'kitchen':
        if task_id in KITCHEN[4:]:
            assert request['budgets']['sim_time_s'] is None
            assert ready['spec']['budgets']['sim_time_s'] is None
            assert verified['task']['public_rules']['referee']['thresholds']['max_time_s'] == 240.
        else:
            assert request['budgets']['sim_time_s'] == verified['task']['recommended_budgets']['sim_time_s']
    assert ready['spec']['task_release_sha256'] == verified['definition']['source_release']['manifest_sha256']
    assert request['robot_id'] == 'sharpa_w01_right'
    assert len(workers) == 1 and workers[0].closed
    assert any(operation == 'step' for operation, _ in workers[0].calls)
    assert not any(operation == 'board_game' for operation, _ in workers[0].calls)
    assert PRIVATE_MARKER in json.dumps(workers[0].raw)
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    if family in ('laboratory', 'kitchen'):
        assert PRIVATE_MARKER in json.dumps(workers[0].score['detail'])
        for filename in ('episode_ready.json', 'run_report.json'):
            assert PRIVATE_MARKER not in (tmp_path / 'run' / filename).read_text()
    for turns in model.turns:
        context = json.loads(turns[-1]['content'][0]['text'])
        assert context['task']['task_id'] == task_id
        if refereed:
            assert context['task']['public_rules'] == verified['task']['public_rules']['referee']
            assert context['task']['instruction'] == verified['task']['public_rules']['referee']['instruction_zh']
            assert not {'referee', 'episode', 'scoring'} & context['task'].keys()
            assert 'goal' not in context['task']
        else:
            assert context['task']['goal']['operation'] == verified['task']['public_goal']['operation']
            if 'description' in verified['task']['public_goal']:
                assert context['task']['goal']['description'] == verified['task']['public_goal']['description']
        if refereed:
            assert 'get_task' in {row['function'] for row in context['tools']}
        assert context['learning_policy']['practice_rsi_allowed']
        assert not context['learning_policy']['rsi_scheduler_implemented']
        assert 'board_game' not in context['state'] and 'players' not in context['state']
        assert 'placement_intent' not in json.dumps(context['decision_schema'])
    assert report['result']['outcome'] != 'success'  # The double never fabricates task success.
    for index in ((1, 4) if refereed else ()):
        context = json.loads(model.turns[index][-1]['content'][0]['text'])
        feedback = context['last_feedback']
        assert feedback['ok'] and feedback['function'] == 'get_task'
        assert feedback['result']['referee']['state'] == 'ongoing'


def test_repeated_task_requirements_keep_rules_goals_and_scoring_contract_without_snapshot():
    static = {'task_id': 'test_task', 'task_kind': 'manipulation',
              'public_rules_sha256': 'a' * 64, 'public_rules': {'instruction': 'Public requirements'},
              'scoring_contract': {'quantity': 'public_height'},
              'goal': {'target_cells': [[0, 0], [1, 1], [2, 2]]}, 'episode_seed': 7}
    task = {**deepcopy(static), 'referee': {'phase': 'working'},
            'episode': {'terminal': True, 'outcome': 'cancelled'},
            'scoring': {'score': 0}}
    before = deepcopy(task)
    assert task_requirements_context(task) == static
    assert task == before
    legacy = {'task_id': 'task0', 'goal': static['goal'], 'episode_seed': 7}
    assert task_requirements_context(legacy) == legacy


@pytest.mark.parametrize('family,task_id', [
    ('mechanical_assembly', 'nist_wire_harness'), ('laboratory', 'cap_placement'),
    ('kitchen', 'pantry_stocking'), ('kitchen', 'table_clearing'),
])
def test_referee_phase_change_uses_current_get_task_feedback_not_creation_snapshot(
        tmp_path, family, task_id):
    api, workers, verified = mechanical_service(tmp_path, task_id, family)
    rules = verified['task']['public_rules']['referee']
    reset = reset_score(Path(verified['definition']['source_release']['path']), verified['task'])['referee']
    next_phase = ('removing' if task_id == 'nist_wire_harness' else
                  next(phase for phase in rules['phases'] if phase != reset['phase']))
    next_reason = ('harness_ready_for_removal' if task_id == 'nist_wire_harness' else
                   'waiting_for_release_and_stability')
    model = ScriptedModel([
        {'function': 'hold', 'arguments': {'duration_s': .04}},
        {'function': 'get_task', 'arguments': {}}, {'stop': True},
    ])
    original_generate = model.generate

    def generate(prompt, turns, **kwargs):
        if not model.turns:
            # Inject a declared public phase into the CPU worker double only.
            # No fabricated physical task success or native evidence is used.
            def publish_phase(worker):
                publish_targets(worker)
                worker.score['referee'].update(phase=next_phase, reason_codes=[next_reason])
            workers[0].on_step = publish_phase
        assert 'call get_task' in prompt
        return original_generate(prompt, turns, **kwargs)

    model.generate = generate
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', task_id, tmp_path/'run',
                              family=family, max_decisions=4, max_status_queries=100000),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['error_type'] is None, report
    assert report['decisions'] == {'single': 3}
    assert report['function_calls'] == {'single': 2}  # No hidden per-decision query.
    ready = load_json(tmp_path/'run/episode_ready.json')
    assert ready['task']['referee'] == reset
    contexts = [json.loads(turns[-1]['content'][0]['text']) for turns in model.turns]
    for context in contexts:
        assert context['task'] == task_requirements_context(ready['task'])
        assert not {'referee', 'episode', 'scoring'} & context['task'].keys()
        assert context['task']['public_rules'] == rules
    feedback = contexts[-1]['last_feedback']
    assert feedback['function'] == 'get_task' and feedback['ok']
    assert feedback['result']['referee']['phase'] == next_phase
    assert feedback['result']['referee']['reason_codes'] == [next_reason]
    assert feedback['context']['observation_sequence'] == contexts[-1]['state']['observation']['sequence']
    assert PRIVATE_MARKER not in json.dumps(model.turns)


@pytest.mark.parametrize('task_family,task_id', [
    ('mechanical_assembly', 'fastener_assembly'), ('laboratory', 'pipette_transfer'),
    ('kitchen', 'pantry_stocking'), ('kitchen', 'table_clearing'),
])
@pytest.mark.parametrize('explicit_family', [True, False])
def test_explicit_family_pins_release_before_model_or_motion_but_legacy_id_remains_usable(
        tmp_path, monkeypatch, task_family, task_id, explicit_family):
    family = task_family if explicit_family else None
    api, workers, _ = mechanical_service(tmp_path, task_id, task_family)
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        create = bench.create

        def different_release(request):
            ready = create(request)
            ready['spec']['task_release_sha256'] = 'e' * 64
            session = api.sessions[ready['episode_id']]
            session.spec = session.spec.model_copy(update={'task_release_sha256': 'e' * 64})
            session.recorder.spec = session.recorder.spec.model_copy(update={'task_release_sha256': 'e' * 64})
            return ready

        monkeypatch.setattr(bench, 'create', different_release)
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run',
                              family=family, seed=0, max_decisions=2), bench=bench, models={'single': model})
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
    if family:
        assert model.turns == []
        assert report['task_setup_error']['code']
        assert report['run_status'] == 'configuration_error'
        assert report['task_success'] is None and report['task_outcome'] == 'unknown'
        assert not (tmp_path / 'run/episode_ready.json').exists()
    else:
        assert len(model.turns) == 1
        assert report['task_setup_error'] is None
        assert report['run_status'] == 'completed'


@pytest.mark.parametrize('family,alias,task_id', [
    ('mechanical_assembly', 'L4', 'fastener_assembly'), ('laboratory', 'task5', 'pipette_transfer'),
    ('kitchen', 'task5', 'pantry_stocking'), ('kitchen', 'task6', 'table_clearing'),
])
@pytest.mark.parametrize('function', ['get_result', 'finish_episode'])
def test_runner_rejects_different_release_result_for_catalog_episode(
        tmp_path, monkeypatch, family, alias, task_id, function):
    api, workers, verified = mechanical_service(tmp_path, task_id, family)
    candidate = (verified['definition']['design']['historical_preinserted']['release']['manifest_sha256']
                 if family == 'mechanical_assembly' else 'e' * 64)
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        original = http.request
        def drifting_result(method, url, **kwargs):
            payload = kwargs.get('json') or {}
            if function == 'get_result' and payload.get('function') == 'get_result':
                next(iter(api.sessions.values())).finish()
            response = original(method, url, **kwargs)
            if payload.get('function') != function:
                return response
            reply = response.json()
            if reply.get('ok'):
                reply['result']['spec']['task_release_sha256'] = candidate
                return httpx.Response(response.status_code, json=reply)
            return response
        monkeypatch.setattr(http, 'request', drifting_result)
        report = run(RunConfig('http://testserver', alias, tmp_path/'run',
                              family=family, max_decisions=2),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['result'] is None and report['task_success'] is None
    assert report['finalization_error']['error_category'] == 'invalid_response'
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
    if function == 'get_result':
        assert model.turns == []


@pytest.mark.parametrize('family,alias,task_id', [
    ('mechanical_assembly', 'L4', 'fastener_assembly'), ('laboratory', 'task4', 'solid_weighing'),
    ('kitchen', 'task5', 'pantry_stocking'), ('kitchen', 'task6', 'table_clearing'),
])
def test_runner_rejects_changed_creation_rules_before_model(tmp_path, monkeypatch, family, alias, task_id):
    from bench.protocol.task_referee import contract_sha256
    api, workers, _ = mechanical_service(tmp_path, task_id, family)
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        create = bench.create
        def different_rules(request):
            ready = create(request)
            ready['task']['instruction'] = ready['task']['public_rules']['instruction_zh'] = '不同发布的任务规则。'
            ready['task']['public_rules_sha256'] = contract_sha256(ready['task']['public_rules'])
            return ready
        monkeypatch.setattr(bench, 'create', different_rules)
        report = run(RunConfig('http://testserver', alias, tmp_path/'run',
                              family=family, max_decisions=2),
                     bench=bench, models={'single': model})
    assert model.turns == [] and report['run_error'] is not None
    assert not (tmp_path/'run/episode_ready.json').exists()
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)


@pytest.mark.parametrize('family,number,task_id', [
    *(('kitchen', number, task_id) for number, task_id in enumerate(KITCHEN, 1)),
    *(('laboratory', number, task_id) for number, task_id in enumerate(LABORATORY, 1)),
    *(('mechanical_assembly', number, task_id) for number, task_id in enumerate(NIST_MECHANICAL, 6)),
])
def test_new_task_discovery_rejects_wrong_release_session_and_ambiguous_task(family, number, task_id):
    selected = select_task(f'task{number}', family=family)
    row = {'task_id': task_id, 'task_kind': 'single_agent', 'create_endpoint': 'POST /episodes',
           'task_release_sha256': selected.source_release['manifest_sha256']}
    assert resolve_task(f'task{number}', [row], family=family) == (row, False)
    cases = [
        ([dict(row, task_release_sha256='e' * 64)], 'release_mismatch'),
        ([dict(row, task_kind='board_game', create_endpoint='POST /board-games')], 'session_contract_mismatch'),
        ([dict(row, task_kind='turn_game', create_endpoint='POST /turn-games')], 'session_contract_mismatch'),
        ([dict(row, create_endpoint='POST /board-games')], 'session_contract_mismatch'),
        ([dict(row, competitive=True)], 'session_contract_mismatch'),
        ([dict(row, status='planned')], 'task_planned'),
        ([row, row], 'task_not_published'),
        ([], 'task_not_published'),
    ]
    for published, code in cases:
        with pytest.raises(TaskSetupError) as error:
            resolve_task(f'task{number}', published, family=family)
        assert error.value.code == code
