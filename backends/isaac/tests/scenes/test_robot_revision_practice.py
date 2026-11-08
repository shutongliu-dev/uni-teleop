from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.protocol.practice import PublicPracticeCase
from scenes.workrooms.card_room.practice.cases import GYM_IDS, build_case, selections, parent_id
from scenes.workrooms.card_room.practice.release import _task
from scenes.workrooms.card_room.practice.referee import PracticeReferee
from scenes.workrooms.robot_revision.practice import (
    derive_case, verified_legacy_view, RobotRevisionPracticeReferee, LINEAGE_SCHEMA)
from scenes.workrooms.robot_revision.inputs import canonical_hash


@pytest.fixture(params=GYM_IDS)
def task(request):
    gym = request.param
    path = Path(f'{DATA_ROOT}/assets/task_parts/board_games')/parent_id(gym)/'scene_v2/metadata.json'
    if not path.is_file(): pytest.skip('Immutable original card assets unavailable')
    metadata = load_json(path)
    original = build_case(gym, selections(gym)[0], metadata)
    result = _task(gym, {'asset_metadata': metadata}, original)
    result['practice_case'] = derive_case(original, metadata,
        revision_sha256='1'*64, formal_release_sha256='2'*64)
    result['robot_revision_practice'] = {'schema': LINEAGE_SCHEMA, 'revision_sha256': '1'*64,
        'parent_case': original, 'formal_release': {'path':'/immutable/revised/formal','manifest_sha256':'2'*64}}
    return result


def test_all_192_practice_selections_keep_goals_reset_split_but_get_new_identity(task):
    metadata, gym = task['asset_metadata'], task['id']
    for selection in selections(gym):
        old = build_case(gym, selection, metadata)
        new = derive_case(old, metadata, revision_sha256='1'*64, formal_release_sha256='2'*64)
        PublicPracticeCase.model_validate(new['public'])
        assert new['reset'] == old['reset']
        assert new['public']['goal'] == old['public']['goal']
        for k in ('active_side','case_id','split','stratum','apparatus','public_rules_sha256'):
            assert new['public'][k] == old['public'][k]
        for k in ('case_fingerprint','private_case_sha256'):
            assert new[k] != old[k]
        assert new['public']['case_sha256'] != old['public']['case_sha256']
        assert not {'reset','private_pairs','case_fingerprint','robot_revision'} & new['public'].keys()


def test_delegate_keeps_legacy_adjudication_and_exposes_only_new_identity(task):
    original_task = verified_legacy_view(task)
    ref = RobotRevisionPracticeReferee(task); old = PracticeReferee(original_task)
    assert ref.public == task['practice_case']['public']
    assert ref.case == task['practice_case']
    assert ref._legacy.public == original_task['practice_case']['public']
    assert ref.score() == old.score()
    assert ref.expected == old.expected and ref.thresholds == old.thresholds
    assert ref.interrupt('cancelled') == old.interrupt('cancelled')
    assert ref.score() == old.score()
    ref.contact_rows = ({'test_contact': True},)
    assert ref._legacy.contact_rows == ref.contact_rows


@pytest.mark.parametrize('fault', ['goal','reset','old_case','revision','new_hash','parent'])
def test_rehashed_or_stale_practice_identities_fail_closed(task, fault):
    bad = deepcopy(task)
    if fault == 'goal': bad['practice_case']['public']['goal']['description'] += ' changed'
    elif fault == 'reset': next(iter(bad['practice_case']['reset'].values()))['position_m'][0] += .01
    elif fault == 'old_case': next(iter(bad['robot_revision_practice']['parent_case']['reset'].values()))['position_m'][0] += .01
    elif fault == 'revision': bad['robot_revision_practice']['revision_sha256'] = '3'*64
    elif fault == 'new_hash': bad['practice_case']['private_case_sha256'] = '0'*64
    elif fault == 'parent': bad['robot_revision_practice']['formal_release']['manifest_sha256'] = '4'*64
    case = bad['practice_case']
    if fault in ('goal','reset'):
        case['public']['case_sha256'] = canonical_hash({k:v for k,v in case['public'].items() if k != 'case_sha256'})
        case['private_case_sha256'] = canonical_hash({k:v for k,v in case.items() if k != 'private_case_sha256'})
    with pytest.raises(ValueError): verified_legacy_view(bad)


def test_runtime_starts_with_new_identity_and_retains_inactive_drive_lock(task, tmp_path, monkeypatch):
    from scenes.workrooms.robot_revision import practice_runtime as module
    class Observer:
        def __init__(self, env, referee, metadata, **kwargs):
            self.referee = referee; self.last_step = 2; self.last_time = .001
        def attach(self): return self
    class Env:
        actuated_joint_names = [f'{side}_joint{i}' for side in ('left','right') for i in range(29)]
        joint_names = actuated_joint_names
        _targets = np.zeros((1,58), dtype=np.float32)
        def _validate_action(self, command): pass
    monkeypatch.setattr(module, 'PracticeObserver', Observer)
    case = task['practice_case']; side = case['public']['active_side']
    scene = {'card_practice_task': {'schema':'card_rsi_practice_native.v1',
        'task_id':task['id'],'active_side':side,'private_case_sha256':case['private_case_sha256']}}
    runtime = module.RobotRevisionPracticeRuntime(Env(), scene, task, tmp_path/'runtime')
    assert runtime.evaluator.public == case['public']
    assert runtime.observer.referee is runtime.evaluator
    command = {'joint_positions_rad': dict(runtime.targets)}
    runtime.authorize_step(command)
    command['joint_positions_rad'][next(n for n in runtime.names if n not in runtime.owned)] = .01
    with pytest.raises(ValueError, match='inactive'): runtime.authorize_step(command)
