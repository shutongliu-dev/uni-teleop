"""New public material facts must match physical inputs without changing them."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import dump, digest, validate
from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import PublicContract
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    CURRENT_TASK_IDS, ASSET_MATERIAL_CONTRACT, configuration_rules, task_public_rules)


@pytest.mark.parametrize('task_id', CURRENT_TASK_IDS)
@pytest.mark.parametrize('configuration', ['single_arm', 'dual_same_side', 'dual_opposed'])
def test_all_public_materials_match_exact_assets_while_scene_reset_and_numeric_criteria_stay_identical(
        tmp_path, task_id, configuration):
    root = (Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023')
        / 'releases' / task_id / configuration / 'contact_v1')
    old_scene, old_task = [load_json(root/name) for name in ('scene.json', 'task.json')]
    selected = old_scene['robot_configuration']
    value = build_task_inputs(task_id,
        asset_manifest=old_scene['laundry_task_provenance']['asset_candidate']['manifest_path'],
        configuration_id=configuration, stand_manifest=selected['stand_manifest']['path'],
        public_material_contract=ASSET_MATERIAL_CONTRACT)
    for key, filename in (('scene', 'scene.json'), ('initial_state', 'initial_state.json')):
        dump(tmp_path/filename, value[key])
        assert digest(tmp_path/filename) == digest(root/filename)
    task = value['task']; rules = task['public_rules']
    assert task['public_material_contract'] == ASSET_MATERIAL_CONTRACT
    assert rules == task_public_rules(task)
    PublicContract.model_validate(rules['referee'])
    assert task['evaluation']['thresholds'] == old_task['evaluation']['thresholds']
    assert task['evaluation']['metadata'] == old_task['evaluation']['metadata']
    actual = {row['id']:row for row in task['asset_metadata']['cloths']}
    properties = {row['id']:row for row in rules['referee']['object_properties']}
    declared = {row['id']:row for row in rules['material_contract']['cloths']}
    assert properties.keys() == declared.keys() == actual.keys()
    for name, cloth in actual.items():
        assert properties[name]['nominal_mass_kg'] == declared[name]['mass_kg'] == cloth['mass_kg']
        assert declared[name]['physical_thickness_m'] == cloth['physical_thickness_m']
        assert f"厚度{cloth['physical_thickness_m']*1000:g}毫米" in properties[name]['basis']
    if task_id == 'towel_flattening':
        assert '局部折回' in rules['referee']['instruction_zh']
        assert '起皱' not in rules['referee']['instruction_zh']
        assert task['public_goal']['description'] == rules['instruction_zh'] == task['steps'][0]['instruction_zh']
    if configuration == 'single_arm':
        assert '29个关节' in rules['referee']['physical_actions']
        assert '58个关节' not in json.dumps(rules, ensure_ascii=False)
        assert '两台FR3' not in json.dumps(rules, ensure_ascii=False)
        assert '双手' not in rules['referee']['phases']['settling']
    if task_id in ('shirt_folding', 'trousers_folding'):
        assert '单件' in rules['instruction_zh']
        assert '叠放' not in rules['instruction_zh']
    SequenceEvaluator(task)
    # The old configured release is still valid under its own exact old rules.
    validate(root, assets=False)
    legacy = configuration_rules(task_id, 0, old_task['evaluation']['rule_profile'], configuration)
    assert legacy == old_task['public_rules']


def test_material_rule_marker_rejects_silent_downgrade_or_invented_declaration():
    from scenes.workrooms.laundry_room.assets import candidate_metadata
    metadata = candidate_metadata('towel_folding', material_profile='towel_4mm_density_v1')
    with pytest.raises(ValueError, match='selected asset task'):
        configuration_rules('towel_flattening', 0, 'laundry_practical_task.v2', 'single_arm', asset_metadata=metadata)
    bad = deepcopy(metadata); bad['cloths'][0]['mass_kg'] = float('nan')
    with pytest.raises(ValueError, match='positive finite'):
        configuration_rules('towel_folding', 0, 'laundry_practical_task.v2', 'single_arm', asset_metadata=bad)
