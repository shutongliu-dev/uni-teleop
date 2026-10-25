"""Review evidence cannot silently promote a grasp or replace failed trials."""
import json

import numpy as np
import pytest

from scripts.teleop.review import Recorder, check_release, initialize, note
from scenes.workrooms.catalog import task_definition
from pathlib import Path


def test_campaign_is_immutable_and_release_pinned(tmp_path):
    tasks = initialize(tmp_path)
    assert len(tasks) == 32
    assert all(t['status'] == 'pending' for t in tasks)
    with pytest.raises(FileExistsError):
        initialize(tmp_path)
    task = task_definition('kitchen', 'breakfast_serving')
    release = Path(task['source_release']['path'])
    assert check_release(tmp_path, 'kitchen', 'breakfast_serving', release) == task['source_release']['manifest_sha256']
    catalog = json.loads((tmp_path/'catalog.json').read_text())
    for item in catalog['tasks']:
        if item['family'] == 'kitchen' and item['task_id'] == 'breakfast_serving':
            item['source_release']['manifest_sha256'] = 'wrong'
    (tmp_path/'catalog.json').write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='Release changed'):
        check_release(tmp_path, 'kitchen', 'breakfast_serving', release)


def test_trial_history_preserves_failure_and_does_not_pass_task(tmp_path):
    initialize(tmp_path)
    session = tmp_path/'session'
    session.mkdir()
    task = task_definition('kitchen', 'breakfast_serving')
    report = {'family': 'kitchen', 'task': 'breakfast_serving', 'scene_kind': 'teleop_workbench',
              'release': task['source_release']['path'],
              'review_release_sha256': task['source_release']['manifest_sha256']}
    (session/'session.json').write_text(json.dumps(report))
    (session/'scene.json').write_text('{}')
    for outcome in ('needs_fix', 'pass'):
        note(tmp_path, session, target='cup', check='grasp_place', outcome=outcome, detail='Trial 1 then trial 2')
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['outcome'] for e in events] == ['needs_fix', 'pass']
    assert all(e['evidence'] == 'operator_report' and not e['formal_evaluation'] for e in events)
    assert all(t['status'] == 'pending' for t in json.loads((tmp_path/'catalog.json').read_text())['tasks'])
    report['review_release_sha256'] = 'wrong'
    (session/'session.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='differs'):
        note(tmp_path, session, target='cup', check='grasp_place', outcome='pass', detail='wrong release')


def test_recording_preserves_reset_and_contact_evidence(tmp_path):
    from PIL import Image
    recorder = Recorder(tmp_path)
    observation = {'sim_time_s': 0., 'robot': {'joint_positions_rad': [0.]},
                   'objects': {'cup': {'position_m': [0, 0, 1]}},
                   'contacts': {'sampling': {'complete': True, 'sampled_substeps': 40}},
                   'terminated': False, 'truncated': False}
    recorder.state(observation, 0, event='reset')
    observation['sim_time_s'] = .02
    recorder.state(observation, 0)
    pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    recorder.image(pixels, observation, 0)
    recorder.image(pixels, observation, 0)  # No queue growth on repeated render calls.
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path/'review_states.jsonl').read_text().splitlines()]
    assert [row['sim_time_s'] for row in rows] == [0., .02]
    assert rows[1]['contacts'] == observation['contacts']
    assert rows[0]['event'] == 'reset'
    images = [json.loads(line) for line in (tmp_path/'review_images.jsonl').read_text().splitlines()]
    assert len(images) == 1
    with Image.open(tmp_path/images[0]['file']) as image:
        assert image.size == (64, 64)
