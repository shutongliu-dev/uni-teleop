"""CPU contracts/adversarial evidence checks; these do not qualify dynamics."""
from copy import deepcopy
import importlib
import pytest
from pxr import Usd

from scenes.workrooms.laboratory import rules
from scenes.workrooms.laboratory.evaluation import NativeTaskObserver, SequenceEvaluator
from scenes.workrooms.laboratory.native_evidence import fixed


def setup(task_id):
    stage = Usd.Stage.CreateInMemory()
    meta = importlib.import_module(f'scenes.workrooms.laboratory.tasks.{task_id}.assets').author_task(stage)
    config = rules.evaluation_configuration(task_id, meta, room_support_paths={
        'table': ['/World/Geometry/worktop/Collider'], 'floor': ['/World/Geometry/room_floor/Collider']})
    task = {'schema_version': 'laboratory-manipulation-task-v1', 'id': task_id,
            'episode_seed': 0, 'evaluation': config, 'public_rules': rules.public_rules(task_id),
            'asset_metadata': meta}
    frame = {'baseline': True, 'time_s': 0., 'native_time_s': 0., 'native_step': 0,
        'physics_dt_s': .0005, 'objects': {r['id']: fixed(r['initial_pose_xyzw']) for r in meta['objects']},
        'joints': {}, 'particle_sets': {}, 'contacts': []}
    if task_id == 'pipette_transfer':
        from scenes.workrooms.laboratory.tasks.pipette_transfer.liquid import ReducedLiquidDynamics
        provider = ReducedLiquidDynamics(task)
        provider.on_frame(frame)
        frame['reduced_dynamics'] = provider.on_observe()
    return task, frame


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_all_canonical_baselines_are_complete_and_unscored(task_id):
    task, frame = setup(task_id)
    observer = NativeTaskObserver(task)
    observer.on_frame(frame)
    summary = observer.on_observe()
    assert summary['complete'], summary['errors']
    assert not summary['events'] and summary['released_stable_s'] == 0
    result = SequenceEvaluator(task).update({'time_s': 0., 'task_state': summary})
    assert result['status'] == 'ongoing' and result['success'] is False


def test_missing_native_step_is_unknown_not_a_physical_loss():
    task, frame = setup('sample_sorting')
    observer = NativeTaskObserver(task)
    observer.on_frame(frame)
    evaluator = SequenceEvaluator(task)
    evaluator.update({'time_s': 0., 'task_state': observer.on_observe()})
    frame.update(baseline=False, time_s=.001, native_time_s=.001, native_step=2)
    observer.on_frame(frame)
    result = evaluator.update({'time_s': .001, 'task_state': observer.on_observe()})
    assert result['status'] == 'unknown'
    assert not result['physical_failure']


def test_sorted_poses_without_contact_do_not_score():
    task, frame = setup('sample_sorting')
    observer = NativeTaskObserver(task)
    observer.on_frame(frame)
    for name, goal in task['asset_metadata']['task_parameters']['targets'].items():
        frame['objects'][name] = fixed(goal['goal_pose_xyzw'])
    for step in range(1, 701):
        frame.update(baseline=False, time_s=step*.0005, native_time_s=step*.0005, native_step=step)
        observer.on_frame(frame)
    result = observer.on_observe()
    assert result['complete'], result['errors']
    assert result['released_stable_s'] == 0.
    assert not any(result['detail']['placed'].values())


def test_pipette_forged_stale_state_is_rejected():
    task, frame = setup('pipette_transfer')
    observer = NativeTaskObserver(task)
    observer.on_frame(frame)
    frame.update(baseline=False, time_s=.0005, native_time_s=.0005, native_step=1)
    observer.on_frame(frame)
    assert not observer.on_observe()['complete']
    assert 'stale' in observer.on_observe()['errors'][0]


def test_rules_and_thresholds_are_frozen():
    task, _ = setup('vial_racking')
    task['evaluation']['thresholds']['stable_s'] = 0
    with pytest.raises(ValueError, match='differ'):
        NativeTaskObserver(task)
