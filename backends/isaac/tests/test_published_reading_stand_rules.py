"""Frozen reading-stand mechanism fixtures, separate from flat placement traces."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, load_evaluator
from bench.paths import load_json
from scenes.workrooms.catalog import task_definition
from test_study_reading_stand import frame


@pytest.mark.parametrize('condition', ['reset', 'no_pin', 'withdrawn_pin', 'held', 'zero_action', 'supported'])
def test_current_reading_stand_support_release_and_recovery(condition):
    reference = task_definition('study', 'reading_stand')['source_release']
    root = Path(reference['path'])
    assert digest(root/'release_manifest.json') == reference['manifest_sha256']
    task = load_json(root/'task.json')
    config = task['evaluation']['configuration']
    evaluator = load_evaluator(root)(task, trusted_normalized_input=True)
    assert evaluator.update(frame(config))['reason'] == 'reset_baseline_not_scored'
    for index in range(1, 850):
        sample = frame(config, index, goal=condition != 'reset', actions=int(condition != 'zero_action'))
        if condition == 'no_pin':
            sample['contacts'] = sample['contacts'][1:]
        elif condition == 'withdrawn_pin':
            sample['objects']['support_pin']['position_m'][1] = .28
        elif condition == 'held':
            sample['contacts'].append({'a': 'book', 'b': 'robot/right_thumb',
                'point_m': deepcopy(sample['objects']['book']['position_m']),
                'impulse_on_a_n_s': [0., 0., .0001], 'separation_m': 0.})
        result = evaluator.update(sample)
    assert result['success'] == (condition == 'supported')
    assert result['status'] == ('success' if condition == 'supported' else 'active')
    if condition != 'supported':
        for index in range(850, 1700):
            result = evaluator.update(frame(config, index, goal=True, actions=1))
        assert result['success'] and result['stable_duration_s'] >= .4
