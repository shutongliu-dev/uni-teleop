"""CPU contract fixtures only; synthetic arrays are never native qualification."""
from copy import deepcopy
import json

import numpy as np
import pytest

from bench.paths import load_json
from bench.protocol.task_referee import PublicContract, project_task_rules, project_referee_state
from scenes.workrooms.laundry_room import assets, composition, evaluation, release
from scenes.workrooms.laundry_room.rules import TASK_IDS, public_rules


def inputs(task_id):
    return composition.build_task_inputs(task_id, physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')


def target_vertices(task):
    result = {}
    for row in task['asset_metadata']['cloths']:
        rest = np.asarray(row['rest_vertices_m'])
        if task['id'] == 'towel_flattening':
            result[row['id']] = rest + [.65, -.55, .9015]
        else:
            center = task['asset_metadata']['target_center_xy_m']
            z = .9105 if task['id'] == 'towel_stacking' and row['id'] == 'towel' else .9015
            result[row['id']] = assets.folded_vertices(rest, center, lower_z=z)
    return result


def frame(task, step, vertices=None, controlled=False):
    rows = task['asset_metadata']['cloths']
    result = {'schema': evaluation.FRAME_SCHEMA, 'complete': True,
        'time_s': step*task['evaluation']['physics_dt_s'], 'native_step': step,
        'control_mode': 'engineering_local_grasp',
        'controlled_vertices': {r['id']: [0, 1] if controlled and r['id'] == 'towel' else [] for r in rows},
        'physics': {'complete': True, 'valid': True, 'self_intersections': 0,
            'intercloth_intersections': 0, 'max_penetration_m': 0.,
            'gravity_enabled': True, 'hidden_constraints': 0}, 'cloths': {}}
    for row in rows:
        points = row['initial_vertices_m'] if vertices is None else vertices[row['id']]
        result['cloths'][row['id']] = {'enabled': True, 'vertices_m': np.asarray(points).tolist(),
            'velocities_m_s': np.zeros_like(points).tolist(), 'triangles': row['triangles']}
    return result


def released_score(task_id, *, modify=None, retained=False):
    task = inputs(task_id)['task']
    scorer = evaluation.SequenceEvaluator(task)
    scorer.observe(frame(task, 0))
    goal = target_vertices(task)
    if modify:
        modify(goal)
    scorer.observe(frame(task, 1, goal, controlled=True))
    for step in range(2, 253):
        score = scorer.observe(frame(task, step, goal, controlled=retained))
    return task, scorer, score


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_explicit_dynamic_physical_thickness_and_public_scope(task_id):
    value = inputs(task_id)
    expected = 2 if task_id == 'towel_stacking' else 1
    assert len(value['scene']['cloths']) == expected
    assert value['scene']['workroom']['version'] == 'laundry_residential_20261021_v3'
    assert value['scene']['goal_markers'] == []
    assert value['task']['capabilities']['robot_episodes'] is False
    for cloth in value['scene']['cloths']:
        assert cloth['mass_kg'] == .108
        assert cloth['size_m'] == [.6, .4]
        assert cloth['isaacsim']['cloth_thickness_m'] == cloth['visual_thickness_m'] == .003
        assert set(cloth['visual_texture_paths']) == {'normal', 'roughness'}
    contract = PublicContract.model_validate(public_rules(task_id, profile='laundry_strict_task.v1')['referee'])
    assert len(contract.object_properties) == expected
    assert project_task_rules(value['task']) is None


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_success_requires_operation_and_complete_released_window(task_id):
    task, scorer, score = released_score(task_id)
    assert score['success'], score
    assert score['released_stable_s'] >= .5-1.e-9
    assert score['robot_manipulation_qualified'] is False
    assert score['physics_accepted'] is False
    project_referee_state(score, public_rules(task_id, profile='laundry_strict_task.v1')['referee'])
    assert not released_score(task_id, retained=True)[2]['success']
    scorer = evaluation.SequenceEvaluator(task)
    goal = target_vertices(task)
    for step in range(253):
        result = scorer.observe(frame(task, step, goal))
    assert not result['success']
    assert result['detail']['operation_seen'] is False


def test_flat_area_alone_does_not_pass_fold_or_rotated_placement():
    def fold(goal):
        rest, _ = assets.topology()
        goal['towel'] = assets.folded_vertices(rest, [.65, -.55])
    assert not released_score('towel_flattening', modify=fold)[2]['success']
    def rotate(goal):
        v = goal['towel']
        v[:, :2] = (v[:, :2]-[.65, -.55])@np.array([[0, -1], [1, 0]])+[.65, -.55]
    assert not released_score('towel_flattening', modify=rotate)[2]['success']


def test_180_degree_symmetric_towel_orientation_is_valid():
    def rotate(goal):
        goal['towel'][:, :2] = 2*np.array([.65, -.55])-goal['towel'][:, :2]
    assert released_score('towel_flattening', modify=rotate)[2]['success']


def test_fold_rejects_zero_thickness_and_stack_rejects_floating_support():
    def collapsed(goal):
        goal['towel'][:, 2] = .9015
    assert not released_score('towel_folding', modify=collapsed)[2]['success']
    def floating(goal):
        goal['towel'][:, 2] += .05
    assert not released_score('towel_stacking', modify=floating)[2]['success']


def test_stack_requires_both_folded_cloths_not_one_rigid_proxy():
    task = inputs('towel_stacking')['task']
    scorer = evaluation.SequenceEvaluator(task)
    sample = frame(task, 0)
    del sample['cloths']['towel_base']
    assert scorer.observe(sample)['status'] == 'unknown'
    task['asset_metadata']['cloths'][1]['dynamic_model'] = 'rigid_body'
    task['evaluation']['metadata'] = deepcopy(task['asset_metadata'])
    with pytest.raises(ValueError, match='Dynamic'):
        evaluation.SequenceEvaluator(task)


@pytest.mark.parametrize('change', [
    lambda f: f['physics'].pop('self_intersections'),
    lambda f: f['physics'].update(hidden_constraints=1),
    lambda f: f['cloths']['towel']['vertices_m'][0].__setitem__(0, float('nan')),
    lambda f: f['cloths']['towel']['triangles'][0].__setitem__(0, 6),
    lambda f: f['controlled_vertices'].update(towel=list(range(100))),
    lambda f: f['controlled_vertices'].update(towel=[0, 15, 30]),
])
def test_missing_or_corrupt_evidence_never_scores(change):
    task = inputs('towel_flattening')['task']
    scorer = evaluation.SequenceEvaluator(task)
    sample = deepcopy(frame(task, 0))
    change(sample)
    result = scorer.observe(sample)
    assert result['status'] == 'unknown'
    assert not result['success']


@pytest.mark.parametrize('field,value', [('intercloth_intersections', 1), ('max_penetration_m', .002), ('valid', False)])
def test_physical_failures_latch_separately_from_evidence_fault(field, value):
    task = inputs('towel_stacking')['task']
    scorer = evaluation.SequenceEvaluator(task)
    sample = frame(task, 0)
    sample['physics'][field] = value
    result = scorer.observe(sample)
    assert result['status'] == 'safety_stop'
    assert scorer.observe(frame(task, 1)) == result


def test_skipped_native_step_and_budget_cannot_later_succeed():
    task = inputs('towel_folding')['task']
    scorer = evaluation.SequenceEvaluator(task)
    scorer.observe(frame(task, 0))
    assert scorer.observe(frame(task, 2))['status'] == 'unknown'
    scorer = evaluation.SequenceEvaluator(task)
    scorer.limits['max_time_s'] = .004
    scorer.observe(frame(task, 0))
    scorer.observe(frame(task, 1, target_vertices(task), controlled=True))
    score = scorer.observe(frame(task, 2, target_vertices(task)))
    assert score['status'] == 'incomplete' and score['terminal']
    assert score['reason'] == 'budget_exhausted'
    assert score['referee']['state'] == 'ongoing' and not score['referee']['terminal']
    assert not score['physical_failure']
    assert scorer.observe(frame(task, 3, target_vertices(task))) == score


def test_task_only_runtime_has_clear_robot_episode_rejection():
    from scenes.runtime import build_task_runtime
    task = inputs('towel_flattening')['task']
    with pytest.raises(ValueError, match='task-only.*ordinary /episodes'):
        build_task_runtime(None, {}, task, None, evaluator_factory=evaluation.SequenceEvaluator)


def test_fold_reset_places_single_fold_in_target_without_extra_translation():
    task = inputs('towel_folding')['task']
    v = np.asarray(task['asset_metadata']['cloths'][0]['initial_vertices_m'])
    assert np.allclose((v.min(axis=0)+v.max(axis=0))/2, [.8, -.55, .9015])
    uncreased_half_center = (v[:, 0].min()+.8)/2
    assert np.isclose(uncreased_half_center, task['asset_metadata']['target_center_xy_m'][0])


def test_two_world_clamps_may_hold_four_material_fragments_of_folded_towel():
    task = inputs('towel_stacking')['task']
    scorer = evaluation.SequenceEvaluator(task)
    sample = frame(task, 0)
    nx, ny = task['asset_metadata']['cloths'][0]['cells_xy']
    # Two corners, each with a 2x2 material patch on both folded layers.
    mask = [j*(nx+1)+i for j in (0, 1, ny-1, ny) for i in (0, 1, nx-1, nx)]
    sample['controlled_vertices']['towel'] = mask
    assert scorer.observe(sample)['reason'] == 'reset_baseline_not_scored'
    # The very same material mask becomes four distinct physical regions
    # when the towel unfolds. Current world geometry must be checked again.
    rest = np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m'])
    changed = deepcopy(sample)
    changed.update(native_step=1, time_s=.002)
    changed['cloths']['towel']['vertices_m'] = (rest+[.65, -.55, .9015]).tolist()
    result = scorer.observe(changed)
    assert result['status'] == 'unknown'
    assert 'two local world-space' in result['private_evidence_error']


def test_world_grasp_limit_is_shared_across_cloths_and_not_chain_connected():
    task = inputs('towel_stacking')['task']
    scorer = evaluation.SequenceEvaluator(task)
    # Pairwise diameter is required, so a chain of nearby points spanning
    # three well-separated clamp sites cannot be treated as one region.
    cloths = {'towel': np.array([[x, 0., 1.] for x in (0, .06, .12, .18)]),
              'towel_base': np.array([[.5, 0., 1.]])}
    with pytest.raises(ValueError, match='two local world-space'):
        scorer._check_world_grasp_regions(cloths, {'towel': [0,1,2,3], 'towel_base': [0]})


def test_release_requires_complete_frozen_source(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    with pytest.raises(ValueError, match='complete frozen project source'):
        release.prepare_task_release('towel_folding', tmp_path/'release', asset_manifest='unused')


def test_asset_inventory_and_release_round_trip_detect_tampering(tmp_path, monkeypatch):
    from scripts.maintenance import freeze_project_sources
    from bench.isaacsim.release import task_inputs, load_evaluator, validate
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'fixture_only': True})
    # This fixture freezes a small dependency inventory instead of reading the
    # full furnished USD. Native room acceptance belongs to its own evidence.
    def inventory(scene):
        return {p: {'sha256': sha} for p, sha in scene['laundry_task_provenance']['asset_candidate']['files_sha256'].items()}
    monkeypatch.setattr(release, 'asset_inventory', inventory)
    asset_root = tmp_path/'assets'
    assets.prepare_assets('towel_stacking', asset_root)
    output = tmp_path/'release'
    manifest = release.prepare_task_release('towel_stacking', output, asset_manifest=asset_root/'manifest.json')
    assert manifest['capabilities']['robot_episodes'] is False
    assert validate(output) == manifest
    entry, scene, task, initial = task_inputs(output, 'towel_stacking', seed=17)
    assert task['episode_seed'] == 17 and len(scene['cloths']) == 2
    assert set(initial['cloths_vertices_m']) == {'towel', 'towel_base'}
    assert load_evaluator(output)(task).report()['evidence_status'] == 'pending'
    with pytest.raises(ValueError, match='exact task and original reset'):
        task_inputs(output, 'towel_stacking', phase='qualification_probe')
    assert str(assets.TEXTURE_ROOT/'SOURCE.json') in manifest['asset_files_sha256']
    metadata = load_json(asset_root/'metadata.json')
    metadata['cloths'][0]['mass_kg'] = .001
    (asset_root/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Frozen input changed'):
        validate(output)
