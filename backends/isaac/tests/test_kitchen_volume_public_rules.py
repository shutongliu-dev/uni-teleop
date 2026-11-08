"""Public FEM gates and historical rule bytes; CPU contract tests, not physics."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bench.protocol.task_referee import PublicContract, project_task_rules
from scenes.workrooms.kitchen import rules
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition
from scenes.workrooms.kitchen.composition import bind_episode_task
from scenes.workrooms.kitchen.evaluation import SequenceEvaluator, configuration


# Canonical JSON digests recorded before the volume-public-contract change.
LEGACY_DIGESTS = {
    'breakfast_serving': '1f262d81ed7a5d9fec00d00131c952be9f0a391db4483bf0a36f64ab4d6a4bb7',
    'produce_transfer': '8a83302ea3c47525f622c01dcd43e18891da8d821c013a6b2a4e6f9e7fe5a72d',
    'utensil_sorting': '7b0de077d2082ca102bdbb1cc78a276265d828f627d3641e0a3c681438795862',
    'pot_lid_placement': 'f0427d077a89c014e261e47351c282fcf55bcb1058b5ef0441d48197868612cc',
    'pantry_stocking': 'dd413cfd06a33afcfba45af1d2296afd4165e95cb621c14b7a2e52d567628393',
    'table_clearing': '80b8b19964f660b38d9ebcf7ee79cd9f6f5d5abd525b7b76875d9e2961ea7ade',
    'surface_wiping': '591a7f7cfb39af71101039cb107f6fa954611e6d1c12775aea57347953caa4f7',
}
VOLUME_DECLARATION = {'volume_bodies': {'foam': {'prim_path': '/Private/Foam',
    'rest_points_m': [[1., 2., 3.]], 'private_marker': 'DO_NOT_PROJECT'}}}


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
@pytest.mark.parametrize('seed', [0, 17, 2**32-1])
def test_all_legacy_contract_bytes_are_unchanged(task_id, seed):
    for metadata in (None, {}, {'volume_bodies': {}}, {'unrelated': True}):
        result = rules.public_rules(task_id, seed, metadata=metadata)
        assert rules.digest(result) == LEGACY_DIGESTS[task_id]


def test_volume_optin_exposes_existing_gates_and_no_private_geometry():
    metadata = deepcopy(VOLUME_DECLARATION)
    result = rules.public_rules('surface_wiping', metadata=metadata)
    assert metadata == VOLUME_DECLARATION
    assert result['schema'] == 'kitchen_rules.v5'
    public = result['referee']
    PublicContract.model_validate(public)
    text = ''.join(public['success_conditions'])
    for phrase in ('全部实际节点', '20 毫秒', '25 毫米/秒', '4 毫米', '5 毫米', '0.4 秒', '重新计时'):
        assert phrase in text
    assert public['thresholds']['minimum_robot_volume_clearance_m'] == .005
    assert public['reason_descriptions']['physics_safety_stop'] == (
        '有效原生证据显示超过声明穿透限值，或泡沫数值几何、材料连接异常；'
        '单独记录物理安全停止，不记为任务操作失败。')
    assert {k: v for k, v in public['thresholds'].items()
            if k != 'minimum_robot_volume_clearance_m'} == rules.thresholds('surface_wiping')
    assert '/Private/' not in str(result) and 'DO_NOT_PROJECT' not in str(result)
    assert 'rest_points_m' not in str(result)
    assert public['success_conditions'][:-1] == rules.public_rules('surface_wiping')['referee']['success_conditions']


@pytest.mark.parametrize('task_id', [t for t in rules.TASK_IDS if t != 'surface_wiping'])
def test_volume_contract_cannot_silently_change_other_tasks(task_id):
    with pytest.raises(ValueError, match='only for surface wiping'):
        rules.public_rules(task_id, metadata=VOLUME_DECLARATION)


@pytest.mark.parametrize('metadata', [[], {'volume_bodies': ['foam']}, {'volume_bodies': None}])
def test_invalid_volume_optin_is_rejected(metadata):
    with pytest.raises(ValueError, match='mapping'):
        rules.public_rules('surface_wiping', metadata=metadata)


def test_new_binding_configuration_and_public_projection_agree():
    from test_kitchen_evaluator import metadata
    meta = metadata('surface_wiping')
    meta.update(deepcopy(VOLUME_DECLARATION))
    original = {'id': 'surface_wiping', 'schema_version': 'kitchen-manipulation-task-v1',
                'asset_metadata': meta, 'public_rules_file': 'rules.json'}
    task = bind_episode_task(original, 17, rules)
    assert 'public_rules' not in original
    c, _ = configuration(task)
    assert c['public_rules_sha256'] == rules.digest(task['public_rules'])
    assert c['thresholds'] == rules.thresholds('surface_wiping')
    assert SequenceEvaluator(task).rules == task['public_rules']['referee']
    projected = project_task_rules(task)
    assert projected['public_rules']['success_conditions'] == task['public_rules']['referee']['success_conditions']
    assert projected['public_rules']['thresholds']['minimum_robot_volume_clearance_m'] == .005
    assert '/Private/' not in str(projected) and 'DO_NOT_PROJECT' not in str(projected)
    altered = deepcopy(task)
    altered['public_rules'] = rules.public_rules('surface_wiping')
    with pytest.raises(ValueError, match='configuration changed'):
        configuration(altered)


def test_legacy_frozen_two_argument_rules_keep_fem_v2_contract():
    # A historical frozen module has no volume capability constant or keyword.
    legacy = SimpleNamespace(
        public_rules=lambda task_id, seed: rules.public_rules(task_id, seed),
        evaluation_configuration=lambda task_id, metadata, seed, physics_dt_s: {'historical': True},
        bind_task_rules=rules.bind_task_rules)
    task = bind_episode_task({'id': 'surface_wiping',
        'asset_metadata': deepcopy(VOLUME_DECLARATION)}, 17, legacy)
    assert task['public_rules_sha256'] == LEGACY_DIGESTS['surface_wiping']
    assert task['public_rules']['schema'] == 'kitchen_rules.v4'


def test_catalog_requires_explicit_asset_optin_for_new_version():
    options = dict(intended_roles=['sponge', 'tray'], inventory_counts={'sponge': 1, 'tray': 1},
                   dynamic_count=1, fixed_fixture_count=1, manifest_sha256='1'*64)
    legacy = implemented_task_definition('surface_wiping', **options)
    new = implemented_task_definition('surface_wiping', **options, asset_metadata=VOLUME_DECLARATION)
    assert rules.digest(legacy['rules']['contract']) == LEGACY_DIGESTS['surface_wiping']
    assert new['rules']['contract'] == rules.public_rules('surface_wiping', metadata=VOLUME_DECLARATION)
    for definition in (legacy, new):
        published = definition['rules']['contract']['referee']['thresholds']
        assert definition['evaluator']['thresholds'] == published
        assert definition['evaluator']['thresholds'] is not published
    assert new['evaluator']['thresholds']['minimum_robot_volume_clearance_m'] == .005
    assert 'minimum_robot_volume_clearance_m' not in legacy['evaluator']['thresholds']
    assert {k: v for k, v in new.items() if k not in ('rules', 'evaluator')} == {
        k: v for k, v in legacy.items() if k not in ('rules', 'evaluator')}
    assert {k: v for k, v in new['evaluator'].items() if k != 'thresholds'} == {
        k: v for k, v in legacy['evaluator'].items() if k != 'thresholds'}
    assert '/Private/' not in str(new) and 'DO_NOT_PROJECT' not in str(new)
