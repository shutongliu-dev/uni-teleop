"""Reject incomplete or mismatched smoke evidence; no native qualification."""
from copy import deepcopy
import json
import struct
import zipfile

import pytest

from scenes.workrooms.study.runtime import StudyTaskRuntime
from scripts.validation.study_task_release import native_coverage, control_coverage, read_json
from test_study_native_runtime import Environment, Evaluator, spec


class ReplayEvaluator(Evaluator):
    def inspect_conditions(self):
        return {'success': False, 'referee': {'state': 'ongoing', 'terminal': False},
            'status': 'incomplete' if self.reason else 'active', 'reason': self.reason,
            'frames': len(self.frames), 'last_pose': tuple(self.frames[-1]['objects']['book']['position_m'])}


@pytest.fixture
def evidence(tmp_path, request):
    env, task = Environment(), spec()
    runtime = StudyTaskRuntime(env, {}, task, tmp_path/'task_runtime', evaluator_factory=ReplayEvaluator)
    runtime.begin_action('study_release_hold')
    runtime.authorize_step({}, None)
    for _ in range(getattr(request, 'param', 3)):
        env.tick()
    runtime.close('cancelled')
    (tmp_path/'final_score.json').write_text(json.dumps(runtime.score()))
    return tmp_path, task


def rewrite_archive(root, transform):
    path = root/'task_runtime/study_native_evidence.zip'
    with zipfile.ZipFile(path) as archive:
        rows = [(name, read_json(archive.read(name))) for name in archive.namelist()]
    transform(rows)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, frame in rows:
            archive.writestr(name, json.dumps(frame))


def test_complete_same_tick_evidence_and_frozen_replay(evidence):
    root, task = evidence
    result = native_coverage(root, task, .0015, ReplayEvaluator)
    assert result['native_substeps'] == 3 and result['normalized_frames'] == 4
    assert result['frozen_evaluator_replay_matches'] and not result['task_success_claimed']


@pytest.mark.parametrize('evidence', [600], indirect=True)
@pytest.mark.parametrize('discontinuity', [0., 1.e-9, .0005])
def test_native_float32_clock_is_continuous_for_full_hold(evidence, discontinuity):
    root, task = evidence
    native_dt = struct.unpack('f', struct.pack('f', task['evaluation']['physics_dt_s']))[0]
    clocks = []

    def native_clock(rows):
        baseline = rows[0][1]['native_world_time_s']
        for index, (_, frame) in enumerate(rows):
            frame['native_world_time_s'] = baseline + index * native_dt
            if index >= 422:
                frame['native_world_time_s'] += discontinuity
            clocks.append(frame['native_world_time_s'])

    rewrite_archive(root, native_clock)
    path = root/'task_runtime/study_sampling_receipt.json'
    receipt = read_json(path.read_text())
    for index, sample in enumerate(receipt['samples'], 1):
        sample['native_time_s'] = clocks[index]
    path.write_text(json.dumps(receipt))
    if discontinuity:
        with pytest.raises(AssertionError, match='not consecutive'):
            native_coverage(root, task, .3, ReplayEvaluator)
    else:
        assert native_coverage(root, task, .3, ReplayEvaluator)['native_substeps'] == 600


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'foreign_config', 'missing_body',
    'native_clock', 'action_count', 'changed_pose', 'nan'])
def test_archive_corruption_cannot_pass(evidence, fault):
    root, task = evidence
    def mutate(rows):
        if fault == 'missing':
            rows.pop()
        elif fault == 'duplicate':
            rows.append(deepcopy(rows[-1]))
        else:
            frame = rows[-1][1]
            if fault == 'foreign_config': frame['configuration_sha256'] = '0'*64
            elif fault == 'missing_body': frame['objects'].clear()
            elif fault == 'native_clock': frame['native_world_time_s'] -= .0005
            elif fault == 'action_count': frame['robot_action_count'] += 1
            elif fault == 'changed_pose': frame['objects']['book']['position_m'][0] += .01
            elif fault == 'nan': frame['time_s'] = float('nan')
    rewrite_archive(root, mutate)
    with pytest.raises((AssertionError, ValueError)):
        native_coverage(root, task, .0015, ReplayEvaluator)


@pytest.mark.parametrize('change', [
    {'delivery_count': 2}, {'readonly_verified': False}, {'native_step_index': 10},
    {'public_time_before_s': .0005}, {'sequence': True}, {'reset_epoch': 2},
])
def test_receipt_must_bind_each_frame_once_to_its_native_step(evidence, change):
    root, task = evidence
    path = root/'task_runtime/study_sampling_receipt.json'
    receipt = read_json(path.read_text())
    receipt['samples'][0].update(change)
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        native_coverage(root, task, .0015, ReplayEvaluator)


def test_replay_and_sealed_final_score_must_match(evidence):
    root, task = evidence
    path = root/'final_score.json'
    score = read_json(path.read_text()); score['success'] = True
    path.write_text(json.dumps(score))
    with pytest.raises(AssertionError, match='sealed native result'):
        native_coverage(root, task, .0015, ReplayEvaluator)


def control_rows():
    task = spec()
    metadata = {'physics_dt_s': .0005, 'object_names': ['book']}
    rows = []
    for index in range(3):
        start, end = max(0, index-1)*.02, index*.02
        samples = 0 if index == 0 else 40
        window = {'complete': True, 'protocol': 'native_rigid_contact_sampling_v1',
            'physics_dt_s': .0005, 'sampled_substeps': samples,
            'native_start_step_index': 10+round(start/.0005), 'native_end_step_index': 10+round(end/.0005),
            'start_sim_time_s': start, 'end_sim_time_s': end,
            'native_start_time_s': .25+start, 'native_end_time_s': .25+end}
        rows.append({'kind': 'reset' if index == 0 else 'step', 'action_id': 'study_release_hold',
            'observation': {'sim_time_s': end, 'time_s': end, 'objects': {'book': {}},
                'contacts': {'errors': [], 'rigid': [], 'sampling': window}},
            'score': {'success': False, 'referee': {'state': 'ongoing', 'terminal': False}}})
    return metadata, rows, task


def test_generic_control_windows_cover_every_native_tick():
    metadata, rows, task = control_rows()
    assert control_coverage(metadata, rows, task, .04)['native_substeps'] == 80
    rows[2]['observation']['contacts']['sampling']['native_start_step_index'] += 1
    with pytest.raises(AssertionError, match='contact window'):
        control_coverage(metadata, rows, task, .04)
