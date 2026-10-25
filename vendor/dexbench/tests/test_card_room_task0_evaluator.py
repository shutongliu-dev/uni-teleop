"""Synthetic scoring controls only; no physics or manipulation qualification."""
from copy import deepcopy
import json
import math

import pytest

from scenes.board_games.tabletop.geometry import geometry_for
from scenes.workrooms.card_room.tasks.task0.evaluator import PlacementEvaluator
from scenes.workrooms.card_room.tasks.task0.rules import (
    PIECE_IDS, bind_episode_targets, select_target_cells, target_positions_m, thresholds,
)


def task_spec(seed=0):
    geometry = geometry_for('tic_tac_toe_3x3')
    task = {'schema_version': 'card-room-placement-task-v1', 'id': 'task0',
        'evaluation': {**thresholds(), 'geometry': geometry.specification(),
            'piece_ids': list(PIECE_IDS),
            'actors': {name: '/World/Objects/'+name for name in PIECE_IDS},
            'board_contact_paths': ['/World/Fixtures/board/collision_000'],
            'robot_actor_prefix': '/World/Robot/',
            'initial_objects': {name: {'position_m': list(geometry.reserve_position('black', i)),
                'quaternion_wxyz': [1., 0., 0., 0.]} for i, name in enumerate(PIECE_IDS)}}}
    return bind_episode_targets(task, seed)


class Synthetic:
    def __init__(self, task=None):
        self.task = task_spec() if task is None else task
        self.e = PlacementEvaluator(self.task)
        self.index = -1
        self.state = deepcopy(self.task['evaluation']['initial_objects'])
        self.goal = dict(zip(PIECE_IDS, self.task['public_goal']['target_positions_m']))

    def sample(self, *, goals=False, support=True, held=(), missing_tick=None,
               board='/World/Fixtures/board/collision_000', penetration=0.):
        self.index += 1
        now, start = self.index*.02, max(0, self.index-1)*.02
        first = self.index == 0
        if goals:
            for name in PIECE_IDS:
                self.state[name]['position_m'] = deepcopy(self.goal[name])
        objects = {name: {**deepcopy(row), 'linear_velocity_m_s': [0., 0., 0.],
            'angular_velocity_rad_s': [0., 0., 0.]} for name, row in self.state.items()}
        rows = []
        if not first:
            for tick in range(1, 41):
                for name in PIECE_IDS:
                    pairs = []
                    if support and tick != missing_tick:
                        pairs.append((self.task['evaluation']['actors'][name], board))
                    if name in held:
                        pairs.append((self.task['evaluation']['actors'][name], '/World/Robot/right_index_DP'))
                    for a, b in pairs:
                        rows.append({'actor0': a, 'actor1': b, 'time_s': start+tick*.0005,
                            'point_count': 1, 'nonzero_impulse_normal_count': 1,
                            'impulse_magnitude_sum_ns': .0003, 'impulse_vector_ns': [0., 0., .0003],
                            'normal_length_max_abs_error': 0., 'min_separation_m': -penetration,
                            'impulse_components': 'normal_only'})
        return {'time_s': now, 'sim_time_s': now, 'last_duration_s': 0. if first else .02,
            'metrics': {'state_finite': True}, 'terminated': False, 'objects': objects,
            'contacts': {'source': 'PhysX stage-bound rigid contact tensor; normal impulses',
                'errors': [], 'rigid': rows,
                'sampling': {'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': .0005,
                    'start_sim_time_s': start, 'end_sim_time_s': now,
                    'native_start_time_s': start, 'native_end_time_s': now,
                    'native_start_step_index': round(start/.0005), 'native_end_step_index': round(now/.0005),
                    'sampled_substeps': 0 if first else 40, 'complete': True}}}

    def emit(self, **kwargs):
        return self.e.update(self.sample(**kwargs))

    def placed(self):
        self.emit()
        # Synthetic state changes are deliberate negative/positive controls,
        # never simulator pose setters or a claimed robot trajectory.
        self.emit(goals=True)
        return self

    def hold(self, count=17, **kwargs):
        for _ in range(count):
            result = self.emit(**kwargs)
        return result


def test_three_distinct_released_supported_pieces_need_observed_300ms():
    s = Synthetic().placed()
    result = s.emit()
    assert not result['success'] and all(value == 0. for value in result['piece_stable_s'].values())
    result = s.hold(14)
    assert not result['success'] and result['placed_count'] == 0
    result = s.emit()
    assert result['success'] and result['placed_count'] == 3 and result['progress'] == 1.
    assert all(value == pytest.approx(.3) for value in result['piece_stable_s'].values())
    assert s.e.inspect_conditions() == result


def test_supply_reset_and_precompleted_reset_never_score():
    s = Synthetic()
    assert s.emit()['reason'] == 'reset_baseline_not_scored'
    assert s.hold(30)['placed_count'] == 0
    forged = Synthetic()
    assert forged.emit(goals=True)['reason'].startswith('invalid_observation:Initial pose')
    task = task_spec()
    task['evaluation']['initial_objects']['black_00']['position_m'] = task['public_goal']['target_positions_m'][0]
    with pytest.raises(ValueError, match='supply trays'):
        PlacementEvaluator(task)


@pytest.mark.parametrize('mode', ['one_piece_three_targets', 'two_pieces_one_target', 'wrong_cell'])
def test_only_simultaneous_unique_physical_occupants_count(mode):
    s = Synthetic().placed()
    if mode == 'one_piece_three_targets':
        for name in PIECE_IDS[1:]:
            s.state[name] = deepcopy(s.task['evaluation']['initial_objects'][name])
        for target in s.task['public_goal']['target_positions_m']:
            s.state['black_00']['position_m'] = deepcopy(target)
            result = s.hold(20)
        assert result['placed_count'] == 1
    elif mode == 'two_pieces_one_target':
        s.state['black_01']['position_m'] = deepcopy(s.goal['black_00'])
        result = s.hold(20)
        assert result['placed_count'] == 1
        assert result['piece_reasons']['black_00'] == 'target_has_multiple_pieces'
    else:
        cells = s.task['evaluation']['target_cells']
        other = next([q, r] for r in range(3) for q in range(3) if [q, r] not in cells)
        s.state['black_00']['position_m'] = target_positions_m(s.task['evaluation']['geometry'], [other])[0]
        result = s.hold(20)
        assert result['placed_count'] == 2
    assert not result['success']


@pytest.mark.parametrize('kwargs', [
    {'support': False}, {'held': ('black_00',)}, {'missing_tick': 17},
    {'board': '/World/Fixtures/reserve_black_west/collision_000'}, {'penetration': .002},
])
def test_target_geometry_is_insufficient_without_native_released_support(kwargs):
    s = Synthetic().placed()
    result = s.hold(25, **kwargs)
    assert not result['success']
    assert result['irrecoverable_reason'] is None
    assert s.hold(17)['success'], 'Recoverable physical defects may be corrected'


def test_disturbance_revokes_credit_and_restarts_only_affected_piece_dwell():
    s = Synthetic().placed()
    assert s.hold()['success']
    result = s.emit(held=('black_00',))
    assert not result['success'] and result['placed_count'] == 2
    assert result['piece_stable_s']['black_00'] == 0.
    assert not s.hold(15)['success']
    assert s.emit()['success']


@pytest.mark.parametrize('mode', ['footprint', 'height', 'tilt', 'linear_velocity',
                                  'angular_velocity', 'observed_motion', 'slow_drift'])
def test_actual_geometry_and_both_native_and_observed_motion_gate_stability(mode):
    s = Synthetic().placed()
    for index in range(25):
        if mode == 'footprint': s.state['black_00']['position_m'][0] = s.goal['black_00'][0]+.042
        if mode == 'height': s.state['black_00']['position_m'][2] = s.goal['black_00'][2]+.004
        if mode == 'tilt': s.state['black_00']['quaternion_wxyz'] = [math.cos(.1), math.sin(.1), 0., 0.]
        if mode in ('observed_motion', 'slow_drift'):
            s.state['black_00']['position_m'][0] = s.goal['black_00'][0]+index*(.0004 if mode == 'observed_motion' else .0001)
        obs = s.sample()
        if mode == 'linear_velocity': obs['objects']['black_00']['linear_velocity_m_s'] = [.016, 0., 0.]
        if mode == 'angular_velocity': obs['objects']['black_00']['angular_velocity_rad_s'] = [.101, 0., 0.]
        result = s.e.update(obs)
        assert not result['success']
    assert result['placed_count'] == 2


@pytest.mark.parametrize('damage', ['sampling_count', 'incomplete', 'missing_object', 'extra_object',
    'duplicate_row', 'off_grid', 'bad_clock', 'nan', 'quaternion', 'missing_separation', 'normal_impulse'])
def test_malformed_or_missing_evidence_latches_unavailable_not_task_failure(damage):
    s = Synthetic().placed()
    obs = s.sample()
    if damage == 'sampling_count': obs['contacts']['sampling']['sampled_substeps'] = 39
    elif damage == 'incomplete': obs['contacts']['sampling']['complete'] = False
    elif damage == 'missing_object': del obs['objects']['black_00']
    elif damage == 'extra_object': obs['objects']['white_00'] = deepcopy(obs['objects']['black_00'])
    elif damage == 'duplicate_row': obs['contacts']['rigid'].append(deepcopy(obs['contacts']['rigid'][0]))
    elif damage == 'off_grid': obs['contacts']['rigid'][0]['time_s'] += .0001
    elif damage == 'bad_clock': obs['contacts']['sampling']['native_start_step_index'] += 1
    elif damage == 'nan': obs['objects']['black_00']['position_m'][0] = float('nan')
    elif damage == 'quaternion': obs['objects']['black_00']['quaternion_wxyz'] = [0., 0., 0., 0.]
    elif damage == 'missing_separation': del obs['contacts']['rigid'][0]['min_separation_m']
    else: obs['contacts']['rigid'][0]['impulse_vector_ns'] = [1., 0., 0.]
    result = s.e.update(obs)
    assert result['reason'].startswith('invalid_observation:')
    assert not result['success'] and result['irrecoverable_reason'] is None
    assert s.emit()['reason'] == result['reason']


def test_binding_is_reproducible_independent_and_changes_only_episode_goal_fields():
    source = task_spec()
    old = deepcopy(source)
    samples = [select_target_cells(seed) for seed in range(32)]
    assert len({tuple(tuple(cell) for cell in cells) for cells in samples}) > 20
    assert all(len({tuple(cell) for cell in cells}) == 3 for cells in samples)
    bound = bind_episode_targets(source, 42)
    assert json.loads(json.dumps(bound)) == bind_episode_targets(json.loads(json.dumps(source)), 42)
    assert source == old
    assert bound['episode_seed'] == 42
    assert bound['evaluation']['target_cells'] == bound['public_rules']['target_cells'] == bound['public_goal']['target_cells']
    assert bound['evaluation']['initial_objects'] == source['evaluation']['initial_objects']
    assert PlacementEvaluator(json.loads(json.dumps(bound)))
    for invalid in (True, -1, 2**32, 1.5, '0'):
        with pytest.raises(ValueError, match='seed'):
            bind_episode_targets(source, invalid)


def test_frozen_threshold_and_target_identity_cannot_be_silently_changed():
    task = task_spec()
    task['evaluation']['dwell_s'] = .1
    with pytest.raises(ValueError, match='Threshold'):
        PlacementEvaluator(task)
    task = task_spec()
    task['evaluation']['target_cells'][1] = deepcopy(task['evaluation']['target_cells'][0])
    with pytest.raises(ValueError, match='distinct'):
        PlacementEvaluator(task)
    task = task_spec()
    task['public_goal']['target_cells'][0] = [2, 2]
    with pytest.raises(ValueError, match='Public goal'):
        PlacementEvaluator(task)
    task = task_spec()
    task['episode_seed'] = 7
    with pytest.raises(ValueError, match='episode seed'):
        PlacementEvaluator(task)


def test_real_builder_contract_matches_evaluator_without_native_claim():
    from scenes.workrooms.card_room.tasks.task0.task import build_task_inputs
    inputs = build_task_inputs(seed=9)
    evaluator = PlacementEvaluator(json.loads(json.dumps(inputs['task'])))
    assert evaluator.task['episode_seed'] == 9
    assert inputs['task']['physics_accepted'] is False
    assert set(inputs['task']['evaluation']['initial_objects']) == set(PIECE_IDS)
