"""Liquid load bindings conserve the inventory and leave public rules practical."""
from copy import deepcopy
import math

from pxr import Usd
import pytest

from scenes.workrooms.laboratory.rules import public_rules, thresholds
from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task
from scenes.workrooms.laboratory.tasks.pipette_transfer.feedback import payload_configuration


def test_wet_payload_binds_each_real_container_once_without_plunger_double_count():
    metadata = author_task(Usd.Stage.CreateInMemory())
    spec = payload_configuration(metadata)
    rows = {row['body_id']: row for row in spec['bindings']}
    assert set(rows) == {'source', 'receiver_a', 'receiver_b', 'pipette'}
    assert spec['initial_total_ml'] == 80.
    assert spec['density_kg_m3'] == 1000.
    assert spec['unbound_volume_paths'] == [['in_flight_ml'], ['cumulative_spill_ml']]
    for name, volume in [('source', 80.), ('receiver_a', 10.), ('receiver_b', 10.), ('pipette', 20.)]:
        row = rows[name]
        assert volume <= row['maximum_volume_ml']
        assert row['axis_body'] == [0., 0., 1.]
        height = volume * 1.e-6 / (math.pi * row['radius_m'] ** 2)
        body = next(item for item in metadata['objects'] if item['id'] == name)
        assert row['dry_mass_kg'] == body['mass_kg']
        assert row['floor_center_m'][2] + height < body['local_bounds_m'][1][2]


def test_legacy_asset_cannot_silently_acquire_new_liquid_loads():
    metadata = author_task(Usd.Stage.CreateInMemory())
    old = deepcopy(metadata)
    old['task_parameters']['hydraulics']['schema'] = 'laboratory_piston_finite_volume.v1'
    with pytest.raises(ValueError, match='explicit v2'):
        payload_configuration(old)


def test_public_instruction_explains_manual_suction_and_allows_repeated_transfers():
    contract = public_rules('pipette_transfer')['referee']
    instruction = contract['instruction_zh']
    assert all(word in instruction for word in ('浸入', '拉出活塞', '压入活塞', '分次', '10±2', '支架'))
    assert '没有自动吸液按钮' in ''.join(contract['success_conditions'])
    assert thresholds('pipette_transfer')['stable_s'] == .3
    metadata = author_task(Usd.Stage.CreateInMemory())['task_parameters']
    assert (metadata['target_per_receiver_ml'], metadata['target_tolerance_ml'], metadata['spill_budget_ml']) == (10., 2., 2.)
