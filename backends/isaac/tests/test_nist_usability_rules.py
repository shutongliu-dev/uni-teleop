"""Usability guidance must preserve practical scoring and frozen legacy rules."""
from copy import deepcopy

import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly import nist_rules as rules
from scenes.workrooms.mechanical_assembly.nist_composition import bind_episode_task
from scenes.workrooms.mechanical_assembly.nist_evaluation import SequenceEvaluator
from test_mechanical_nist_rules import fixture


PROFILE = 'nist_manipulation_scale_v1'


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_usability_rules_rebind_seed_without_new_scoring_requirements(task_id):
    task, *_ = fixture(task_id)
    task['room_support_paths'] = task['evaluation']['room_support_paths']
    before = deepcopy(task)
    task['asset_metadata']['usability_profile'] = PROFILE
    task = bind_episode_task(task, 17, rules)
    SequenceEvaluator(task)
    assert task['evaluation']['thresholds'] == before['evaluation']['thresholds']
    assert task['asset_metadata']['task_parameters'] == before['asset_metadata']['task_parameters']
    assert task['public_rules']['referee']['success_conditions'] == before['public_rules']['referee']['success_conditions']
    assert task['evaluation']['public_rules_sha256'] == task['public_rules_sha256']
    assert task['public_goal']['description'] == task['public_rules']['instruction_zh']
    assert 'not official NIST dimensions' in task['public_rules']['source_scope']
    if task_id == 'nist_belt_drive_assembly':
        assert '裁判不限定动作先后' in task['public_rules']['instruction_zh']
        assert '三个轮子都装好后' in task['public_rules']['referee']['instruction_zh']
    else:
        assert task['public_rules']['instruction_zh'] == before['public_rules']['instruction_zh']
    corrupt = deepcopy(task)
    corrupt['public_rules'] = rules.public_rules(task_id, 17)
    with pytest.raises(ValueError, match='rules or metadata'):
        SequenceEvaluator(corrupt)


def test_unknown_usability_profile_is_rejected():
    with pytest.raises(ValueError, match='Unknown NIST usability'):
        rules.public_rules('nist_belt_drive_assembly', usability_profile='invented')


def test_legacy_connector_and_belt_rule_identity_stays_exact():
    assert rules.digest(rules.public_rules('nist_connector_insertion')) == 'e14e9e2431dcf1c452e4ba788ed1ffa44c673094138deba6e8256db162cea443'
    assert rules.digest(rules.public_rules('nist_belt_drive_assembly')) == '4d44ea98edb1de79239193fe516163ab9eb7c78caa9a74b6ba676f1e45016d94'


def test_early_immutable_asset_probe_keeps_its_frozen_rules_api():
    """Early physical probes predate optional public guidance; keep them loadable."""
    from pathlib import Path
    from bench.isaacsim.release import task_inputs
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
                'nist_usability_20261023/task7/releases/probe_v1')
    if not root.is_dir():
        pytest.skip('Historical physical probe is absent')
    _, _, task, _ = task_inputs(root, 'nist_belt_drive_assembly', seed=17)
    assert task['asset_metadata']['usability_profile'] == PROFILE
    assert task['public_rules'] == rules.public_rules('nist_belt_drive_assembly', 17)
