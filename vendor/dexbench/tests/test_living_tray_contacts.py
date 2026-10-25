"""Tray contact preflight tests against real CPU-authored USD colliders."""
from copy import deepcopy

import pytest
from pxr import Usd, UsdPhysics

from scenes.workrooms.living_room.tasks.tower_extraction.assets import author_task
from scenes.workrooms.living_room.tray_contacts import validate_tray_contact_mapping


@pytest.fixture(scope='module')
def authored():
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, profile='trays_v1')
    inventory = {'colliders': [str(p.GetPath()) for p in stage.Traverse()
                               if p.HasAPI(UsdPhysics.CollisionAPI)],
                 'bodies': [{'path': str(p.GetPath())} for p in stage.Traverse()
                            if p.HasAPI(UsdPhysics.RigidBodyAPI)]}
    return metadata, inventory


def test_all_ten_tray_actor_paths_match_authored_static_floor_and_wall_colliders(authored):
    metadata, inventory = authored
    result = validate_tray_contact_mapping(metadata, inventory)
    assert result == metadata['contact_actor_paths']
    assert len(result) == 10
    assert result['front_collection_tray_floor'].endswith('/Floor')
    assert result['rear_collection_tray_wall_neg_y'].endswith('/Walls/neg_y')
    result.clear()
    assert len(metadata['contact_actor_paths']) == 10


@pytest.mark.parametrize('damage', [
    'missing_floor', 'missing_wall', 'extra_actor', 'swapped_floor_wall', 'duplicate_path',
    'missing_collider', 'unmapped_collider', 'missing_static_receipt', 'dynamic_fixture',
    'disabled_collision', 'wall_claims_floor', 'height_mismatch', 'duplicate_tray_receipt',
])
def test_incomplete_or_mislabelled_tray_contacts_fail_before_native_start(authored, damage):
    metadata, inventory = deepcopy(authored)
    mapping = metadata['contact_actor_paths']
    floor = 'front_collection_tray_floor'
    wall = 'front_collection_tray_wall_neg_y'
    if damage == 'missing_floor':
        del mapping[floor]
    elif damage == 'missing_wall':
        del mapping[wall]
    elif damage == 'extra_actor':
        mapping['unconfigured_floor'] = mapping[floor]
    elif damage == 'swapped_floor_wall':
        mapping[floor], mapping[wall] = mapping[wall], mapping[floor]
    elif damage == 'duplicate_path':
        mapping[wall] = mapping[floor]
    elif damage == 'missing_collider':
        inventory['colliders'].remove(mapping[floor])
    elif damage == 'unmapped_collider':
        inventory['colliders'].append('/World/LivingTask/Trays/UnobservedWall')
    elif damage == 'missing_static_receipt':
        metadata['static_collision_paths'].remove(mapping[floor])
    elif damage == 'dynamic_fixture':
        inventory['bodies'].append({'path': metadata['trays'][0]['prim_path']})
    elif damage == 'disabled_collision':
        metadata['trays'][0]['components'][0]['collision'] = False
    elif damage == 'wall_claims_floor':
        metadata['trays'][0]['components'][1]['support_role'] = 'floor'
    elif damage == 'height_mismatch':
        metadata['trays'][0]['floor_z_m'] += .005
    else:
        metadata['trays'][1] = deepcopy(metadata['trays'][0])
    with pytest.raises(ValueError):
        validate_tray_contact_mapping(metadata, inventory)


def test_supply_tray_configuration_uses_the_same_exact_physical_identity_check(authored):
    metadata, inventory = deepcopy(authored)
    rows = []
    aliases = {}
    for index, tray in enumerate(metadata['trays']):
        floor = f'external/supply_{index}'
        aliases[tray['floor_actor_id']] = floor
        aliases.update({wall: floor + '/wall_' + str(i) for i, wall in enumerate(tray['wall_actor_ids'])})
        rows.append({'id': 'supply_' + str(index), 'floor_z_m': tray['floor_z_m'],
                     'interior_bounds_xy_m': tray['interior_bounds_xy_m'],
                     'contact_actor_ids': [floor, *[aliases[wall] for wall in tray['wall_actor_ids']]]})
        tray['floor_actor_id'] = floor
        tray['wall_actor_ids'] = [aliases[wall] for wall in tray['wall_actor_ids']]
        for component in tray['components']:
            component['actor_id'] = aliases[component['actor_id']]
    metadata['configuration'] = {'supply_trays': rows}
    metadata['contact_actor_paths'] = {aliases[actor]: path for actor, path in metadata['contact_actor_paths'].items()}
    assert validate_tray_contact_mapping(metadata, inventory) == metadata['contact_actor_paths']
    rows[0]['contact_actor_ids'].remove('external/supply_0')
    with pytest.raises(ValueError, match='floor'):
        validate_tray_contact_mapping(metadata, inventory)


def test_legacy_assets_without_trays_keep_their_empty_mapping():
    assert validate_tray_contact_mapping({'configuration': {}}, {'colliders': [], 'bodies': []}) == {}
    with pytest.raises(ValueError, match='configured'):
        validate_tray_contact_mapping({'contact_actor_paths': {'tray': '/World/Tray'}}, {})
