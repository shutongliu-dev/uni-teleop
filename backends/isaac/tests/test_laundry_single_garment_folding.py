"""Single-item folding contracts; synthetic scoring is not native feasibility."""
from copy import deepcopy

import numpy as np
import pytest

from agent.model_client.scene_tasks import TaskSetupError, select_task
from bench.protocol.task_referee import PublicContract
from scenes.workrooms import catalog
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    CURRENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, GARMENT_FOLDING_PROFILE,
    ROBOT_CONTROL, TASK_SCHEMA, bind_task_rules, digest, public_rules, thresholds,
)
from test_laundry_robot_rules import robot_frame


def folding_task(task_id='shirt_folding'):
    # Reuse a known material mesh solely as a small CPU scorer fixture. The
    # native task's separately authored single shell is never claimed here.
    from scenes.workrooms.laundry_room.garments import candidate_metadata
    meta = deepcopy(candidate_metadata(task_id.replace('_folding', '_stacking')))
    meta.update(schema='laundry_single_garment_folding_asset.v1',
                task_id=task_id, initial_state_kind='unfolded_flat',
                goal_reference_status='authored_fold_reference')
    meta['cloths'] = meta['cloths'][:1]
    spec = meta['cloths'][0]
    meta['physical_thickness_m'] = spec['physical_thickness_m']
    initial = np.asarray(spec['rest_vertices_m']) + [*meta['target_center_xy_m'],
        meta['table_top_z_m']+spec['physical_thickness_m']/2]
    spec['initial_vertices_m'] = initial.tolist()
    rules = public_rules(task_id, control_mode=ROBOT_CONTROL)
    task = bind_task_rules({'schema_version': TASK_SCHEMA, 'id': task_id,
        'task_id': task_id, 'episode_seed': 0, 'asset_metadata': meta,
        'public_rules': rules, 'evaluation': {'task_id': task_id,
            'metadata': deepcopy(meta), 'physics_dt_s': .002,
            'rule_profile': GARMENT_FOLDING_PROFILE, 'control_mode': ROBOT_CONTROL,
            'public_rules_sha256': digest(rules), 'thresholds': thresholds(task_id)}})
    reference = np.asarray(spec['folded_reference_vertices_m']).copy()
    reference[:, :2] += np.asarray(meta['target_center_xy_m']) - (reference[:, :2].min(0)+reference[:, :2].max(0))/2
    reference[:, 2] += meta['table_top_z_m']+spec['physical_thickness_m']/2-reference[:, 2].min()
    return task, {'towel': reference}


def operate(task, goal, *, active=True):
    evaluator = SequenceEvaluator(task)
    evaluator.observe(robot_frame(task, 0))
    evaluator.observe(robot_frame(task, 1, near=active))
    result = evaluator.observe(robot_frame(task, 2, goal, near=active, x=.015))
    assert result['status'] != 'unknown', result
    return evaluator


def finish(task, evaluator, goal, *, near=False, last=253, mutate=None):
    for step in range(3, last+1):
        frame = robot_frame(task, step, goal, near=near)
        if mutate:
            mutate(frame)
        result = evaluator.observe(frame)
    return result


@pytest.mark.parametrize('task_id', GARMENT_FOLDING_TASK_IDS)
def test_fold_contract_has_one_initially_unfolded_item(task_id):
    rules = public_rules(task_id, control_mode=ROBOT_CONTROL)
    PublicContract.model_validate(rules['referee'])
    assert rules['rule_profile'] == GARMENT_FOLDING_PROFILE
    assert [item['id'] for item in rules['referee']['object_properties']] == ['towel']
    assert '展开' in rules['instruction_zh'] and '另一件' not in rules['instruction_zh']
    assert '叠放' not in rules['instruction_zh']
    assert rules['reset_credit'] is False
    assert rules['referee']['thresholds'] == thresholds(task_id)
    expected_mass = .18 if task_id == 'shirt_folding' else .26
    assert rules['referee']['object_properties'][0]['nominal_mass_kg'] == expected_mass


@pytest.mark.parametrize('task_id', GARMENT_FOLDING_TASK_IDS)
def test_material_fold_with_wrinkles_translation_and_outlier_jitter_can_pass(task_id):
    task, goal = folding_task(task_id)
    goal['towel'][:, 0] += .02
    goal['towel'][:, 2] += .001*(1+np.sin(goal['towel'][:, 1]*10))
    evaluator = operate(task, goal)
    def small_jitter(frame):
        frame['cloths']['towel']['velocities_m_s'][0, 0] = .3
    result = finish(task, evaluator, goal, mutate=small_jitter)
    assert result['success'], result
    assert result['detail']['geometry']['operation'] == 'single_garment_folding'
    assert result['detail']['all_control_released']
    assert not result['robot_manipulation_qualified']
    assert not result['physics_accepted']


@pytest.mark.parametrize('task_id', GARMENT_FOLDING_TASK_IDS)
@pytest.mark.parametrize('failure', ['unfolded', 'no_operation', 'held', 'layer_collapse', 'crossing', 'speed', 'short_window'])
def test_single_fold_rejects_missing_shape_operation_release_or_rest(task_id, failure):
    task, goal = folding_task(task_id)
    if failure == 'unfolded':
        goal['towel'] = np.asarray(task['asset_metadata']['cloths'][0]['initial_vertices_m']).copy()
        goal['towel'][:, 0] += .03
    elif failure == 'layer_collapse':
        goal['towel'][:, 2] = goal['towel'][:, 2].min()
    evaluator = operate(task, goal, active=failure != 'no_operation')
    def corrupt(frame):
        if failure == 'crossing':
            frame['physics']['self_intersections'] = 1
            frame['physics']['self_crossing_measure'] = {
                'schema': 'laundry_garment_crossing_measure.v1', 'complete': True,
                'pair_count': 1, 'affected_triangle_count': 2,
                'maximum_pair_length_m': .02, 'total_intersection_length_m': .02,
                'maximum_component_diameter_m': .02, 'affected_rest_area_fraction': .001,
                'coplanar_overlap_area_sum_m2': 0.}
        elif failure == 'speed':
            frame['cloths']['towel']['velocities_m_s'][:, 0] = .12
    result = finish(task, evaluator, goal, near=failure == 'held',
                    last=252 if failure == 'short_window' else 253, mutate=corrupt)
    assert not result['success'], result
    assert result['status'] != 'unknown', result


@pytest.mark.parametrize('task_id', GARMENT_FOLDING_TASK_IDS)
def test_fold_disallows_second_garment_and_prefolded_reset(task_id):
    task, goal = folding_task(task_id)
    extra = deepcopy(task)
    extra['asset_metadata']['cloths'].append({**extra['asset_metadata']['cloths'][0], 'id': 'towel_base'})
    extra['evaluation']['metadata'] = deepcopy(extra['asset_metadata'])
    with pytest.raises(ValueError, match='inventory'):
        SequenceEvaluator(extra)
    prefolded = deepcopy(task)
    prefolded['asset_metadata']['cloths'][0]['initial_vertices_m'] = goal['towel'].tolist()
    prefolded['evaluation']['metadata'] = deepcopy(prefolded['asset_metadata'])
    with pytest.raises(ValueError, match='unfolded|already satisfy'):
        SequenceEvaluator(prefolded)


@pytest.mark.parametrize('status', [None, 'provisional_projection_not_qualified', 'unknown'])
def test_unqualified_projected_reference_cannot_award_success(status):
    task, goal = folding_task()
    task['asset_metadata']['goal_reference_status'] = status
    task['evaluation']['metadata'] = deepcopy(task['asset_metadata'])
    result = finish(task, operate(task, goal), goal)
    assert not result['success']
    assert result['detail']['geometry']['goal_reference_qualified'] is False


def test_explicit_equivalent_fold_keeps_material_identity_and_its_own_support():
    task, goal = folding_task('trousers_folding')
    spec = task['asset_metadata']['cloths'][0]
    reverse = goal['towel'].copy()
    reverse[:, 2] = reverse[:, 2].max()+reverse[:, 2].min()-reverse[:, 2]
    # This synthetic variant exercises the choice of reference and support;
    # the asset author supplies the actual opposite-leg folded surface.
    spec['folded_reference_variants'] = [{
        'id': 'opposite_layer_order', 'folded_reference_vertices_m': reverse.tolist(),
        'folded_layer_pairs': np.asarray(spec['folded_layer_pairs'])[:, ::-1].tolist(),
        'support_bottom_vertex_indices': np.flatnonzero(reverse[:, 2] <= reverse[:, 2].min()+.0004).tolist(),
        'shell_layer_pairs': []}]
    task['evaluation']['metadata'] = deepcopy(task['asset_metadata'])
    evaluator = SequenceEvaluator(task)
    assert not evaluator._garment_shape('towel', reverse, np.asarray(task['asset_metadata']['target_center_xy_m']), table_supported=True)[0]
    goal = {'towel': reverse}
    result = finish(task, operate(task, goal), goal)
    assert result['success'], result
    assert result['detail']['geometry']['fold_reference_variant'] == 'opposite_layer_order'


def test_natural_shell_clearance_is_independent_of_fold_pair_voting():
    task, goal = folding_task()
    spec = task['asset_metadata']['cloths'][0]
    pair = spec['folded_layer_pairs'][0]
    spec['shell_layer_pairs'] = [pair]
    task['evaluation']['metadata'] = deepcopy(task['asset_metadata'])
    evaluator = SequenceEvaluator(task)
    actual = goal['towel'].copy()
    actual[pair[1], 2] = actual[pair[0], 2]+.1*spec['physical_thickness_m']
    good, detail = evaluator._garment_fold({'towel': actual})
    assert not good
    assert detail['towel']['natural_shell_layer_fraction'] == 0.
    assert detail['towel']['ordered_layer_fraction'] >= .85
    # A large XY separation never substitutes for signed vertical clearance.
    actual[pair[1], 0] += .02
    assert evaluator._garment_fold({'towel': actual})[1]['towel']['natural_shell_layer_fraction'] == 0.


def test_native_settled_reset_requires_provenance_and_intact_unfolded_mesh():
    task, goal = folding_task()
    meta = task['asset_metadata']
    spec = meta['cloths'][0]
    meta.update(initial_state_kind='native_settled_unfolded_shell',
                reset_profile='physx_native_settled_unfolded_v1',
                native_settled_source={'cpu_fixture_only': True, 'cloth_position_velocity_writes': 0,
                    'controlled_cloth_vertices': 0, 'operation_credit': False,
                    'rest_geometry_or_topology_modified': False})
    spec['shell_layer_pairs'] = spec['folded_layer_pairs'][:1]
    initial = np.asarray(spec['initial_vertices_m'])
    spec['initial_vertices_m'] = (initial+.01*np.asarray(spec['rest_vertices_m'])).tolist()
    task['evaluation']['metadata'] = deepcopy(meta)
    SequenceEvaluator(task)
    missing = deepcopy(task)
    missing['asset_metadata']['native_settled_source'] = {}
    missing['evaluation']['metadata'] = deepcopy(missing['asset_metadata'])
    with pytest.raises(ValueError, match='provenance'):
        SequenceEvaluator(missing)
    prefolded = deepcopy(task)
    prefolded['asset_metadata']['cloths'][0]['initial_vertices_m'] = goal['towel'].tolist()
    prefolded['evaluation']['metadata'] = deepcopy(prefolded['asset_metadata'])
    with pytest.raises(ValueError, match='already satisfy'):
        SequenceEvaluator(prefolded)


def test_qualified_sewn_target_cannot_omit_natural_layer_samples():
    task, _ = folding_task()
    task['asset_metadata']['initial_state_kind'] = 'unfolded_shell'
    task['evaluation']['metadata'] = deepcopy(task['asset_metadata'])
    with pytest.raises(ValueError, match='natural shell layer pairs'):
        SequenceEvaluator(task)


def test_current_slots_replace_stacking_without_hiding_canonical_history():
    rows = catalog.list_tasks('laundry_room', include_planned=True)
    assert [row['task_id'] for row in rows] == list(CURRENT_TASK_IDS)
    for number, task_id in [(4, 'shirt_folding'), (5, 'trousers_folding')]:
        selected = select_task(f'task{number}', family='laundry_room')
        assert selected.task_id == task_id
        assert selected.session_kind == 'single_agent'
        assert selected.robot_requirements['active_sides'] == ['left', 'right']
        if selected.status == 'planned':
            with pytest.raises(TaskSetupError, match='planned'):
                selected.require_runnable()
    for task_id in ('shirt_stacking', 'trousers_stacking'):
        assert select_task(task_id, family='laundry_room').task_id == task_id
        assert select_task(task_id).family == 'laundry_room'
        assert catalog.task_definition('laundry_room', task_id)['superseded_by'] == task_id.replace('_stacking', '_folding')
