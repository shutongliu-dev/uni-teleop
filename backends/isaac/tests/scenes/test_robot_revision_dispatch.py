from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.runtime.reset_cases import (effective_release_schema,effective_release_version,
    is_randomized_release,RANDOMIZED_RELEASE_SCHEMAS)
from scenes.workrooms.robot_revision import SCHEMA,VERSION
from scenes.workrooms.robot_revision import catalog
from scenes.workrooms.robot_revision.inputs import derive_inputs
from test_robot_revision_release import original_robot_catalog,publication
from test_robot_revision_inputs import inputs


@pytest.mark.parametrize('schema,kind',[(s,'finite_seed') for s in RANDOMIZED_RELEASE_SCHEMAS]+[
    ('robot_board_game_task_release.v1','fixed_game'),('card_game_expansion_release.v1','fixed_game'),
    ('card_rsi_practice_release.v1','practice'),('living_room_task_release.v1','legacy_seed'),
    ('card_room_placement_task_release.v1','legacy_seed')])
def test_effective_schema_preserves_route_not_outer_immutable_identity(schema,kind):
    value={'schema':SCHEMA,'parent_schema':schema,'parent_schema_version':'original_version',
           'selection_kind':kind}
    assert effective_release_schema(value)==schema
    assert effective_release_version(value)=='original_version'
    assert is_randomized_release(value)==(kind=='finite_seed')
    assert value['schema']==SCHEMA
    assert is_randomized_release({'schema':schema})==(schema in RANDOMIZED_RELEASE_SCHEMAS)


def test_public_robot_catalog_preserves_menu_default_layout_and_discards_qualifications(publication,tmp_path,monkeypatch):
    from scenes.workrooms.catalog import task_definition
    from bench.protocol.robot_configurations import project_robot_configurations
    original=task_definition('card_room','task0')
    outputs={c:publication('card_room','task0',c)[0] for c in original['robot_configurations']}
    row=catalog.publication_row(original,outputs)
    pins=tmp_path/'pins.json';pins.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{'card_room/task0':row}}))
    monkeypatch.setattr(catalog,'PINS',pins)
    monkeypatch.setattr(catalog,'CURRENT_TASKS',{'card_room/task0'})
    new=catalog.bind_definition(original)
    assert new['default_configuration_id']==original.get('default_configuration_id','single_arm')
    assert set(new['robot_configurations'])==set(original['robot_configurations'])
    assert new['layout']==original['layout'] and new['objects']==original['objects'] and new['rules']==original['rules']
    public=project_robot_configurations(new['robot_configurations'])
    for key,value in new['robot_configurations'].items():
        assert value['qualification']==catalog.QUALIFICATION
        assert value['layout']==original['robot_configurations'][key]['layout']
        assert public[key]['robot_requirements']['assembly_version']=='isaacsim_6_1_l_mount_v1'
    for fault in ('default','parent','missing_menu','inherited_flag'):
        bad=deepcopy(row)
        if fault=='default':bad['default_configuration_id']='dual_opposed'
        elif fault=='parent':bad['parents']['single_arm']['manifest_sha256']='0'*64
        elif fault=='missing_menu':del bad['releases']['single_arm']
        else:
            # Unknown fields cannot confer qualification; row metadata is never
            # projected. The actual manifest flags remain the strict authority.
            bad['qualification']={'physics_accepted':True}
        pins.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{'card_room/task0':bad}}))
        with pytest.raises(ValueError):catalog.bind_definition(original)


def test_non_arm_ready_coordinates_cannot_be_changed(inputs):
    inputs[3]['joint_positions_rad']['finger1']+=.01
    with pytest.raises(ValueError,match='hand or non-arm'):derive_inputs(*inputs)


@pytest.mark.parametrize('family,task', [('card_room','task0'),('living_room','tower_building')])
def test_legacy_single_public_menu_does_not_invent_internal_configuration(publication,family,task):
    from bench.runtime.robot_configurations import checked_configuration_release
    from bench.isaacsim.release import validate
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.robot_revision import release
    root,manifest,parent,_,definition=publication(family,task,'single_arm')
    old=validate(parent)
    assert 'configuration_id' not in old
    assert 'robot_configuration' not in release.read(parent/'scene.json')
    assert manifest['configuration_id'] is None
    pin={'path':str(root),'manifest_sha256':release.file_hash(root/'release_manifest.json')}
    current=task_definition(family,task)
    option=catalog._checked_option(current,'single_arm',current['robot_configurations']['single_arm'],pin)
    binding={'task_id':task,'releases':{'single_arm':pin},'robot_configurations':{'single_arm':option}}
    assert checked_configuration_release(binding,'single_arm')==root
    # A missing internal ID never grants a dual installation alias.
    binding['releases']['dual_opposed']=pin
    binding['robot_configurations']['dual_opposed']=option
    with pytest.raises(ValueError,match='another robot installation'):
        checked_configuration_release(binding,'dual_opposed')


@pytest.mark.parametrize('fault', ['explicit_none','explicit_single','scene_config','dual','formal','schema'])
def test_legacy_menu_exception_requires_undeclared_single_parent(fault):
    from scenes.workrooms.robot_revision.release import legacy_single_menu
    manifest={'schema':'living_room_task_release.v1'}
    scene={'robot':{'hand_id':'sharpa_w01_right'}}
    assert legacy_single_menu(manifest,scene)
    if fault=='explicit_none':manifest['configuration_id']=None
    elif fault=='explicit_single':manifest['configuration_id']='single_arm'
    elif fault=='scene_config':scene['robot_configuration']={'id':'single_arm'}
    elif fault=='dual':scene['robot']['hand_id']='dual_fr3_sharpa_w01'
    elif fault=='formal':manifest['task_kind']='turn_game'
    else:manifest['schema']='robot_board_game_task_release.v1'
    assert not legacy_single_menu(manifest,scene)


def test_catalog_atomic_inventory_and_source_resource_declarations(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS
    import tomllib
    assert len(catalog.CURRENT_TASKS)==51
    resource='scenes/workrooms/robot_revision/catalog_pins.json'
    assert resource in DEFAULT_EXTRAS
    configuration=tomllib.loads((Path(__file__).parents[2]/'pyproject.toml').read_text())
    assert configuration['tool']['setuptools']['package-data']['scenes.workrooms.robot_revision']==['catalog_pins.json']
    pins=tmp_path/'pins.json';pins.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{'card_room/task0':{}}}))
    monkeypatch.setattr(catalog,'PINS',pins)
    with pytest.raises(ValueError,match='all 51'):
        catalog.bind_definition({'family':'card_room','task_id':'task0'})


def test_actual_complete_v3_overlay_preserves_all_51_tasks_and_113_installations(tmp_path,monkeypatch):
    """Exercise the real committed overlay, independently of author fixtures.

    This is metadata/input identity coverage, not another full robot asset or
    physics validation. Every selected release manifest and scene is hash bound.
    """
    from scenes.workrooms.catalog import list_tasks,task_definition
    from scenes.workrooms.robot_revision import release
    from scenes.workrooms.parameterized import catalog as parameterized

    # This covers the robot-revision overlay itself; the paired parameterized
    # overlay above it has its own catalog tests, so neutralize it here.
    neutral=tmp_path/'parameterized_pins.json'
    neutral.write_text(json.dumps({'schema':parameterized.SCHEMA,'tasks':{}}))
    monkeypatch.setattr(parameterized,'PINS',neutral)
    pins_path=catalog.PINS
    pins_bytes=pins_path.read_bytes()
    pins=json.loads(pins_bytes)
    assert set(pins['tasks'])==catalog.CURRENT_TASKS and len(pins['tasks'])==51
    current={row['family']+'/'+row['task_id']:row for row in list_tasks(robot_profile='l_mount')}
    assert set(current)==catalog.CURRENT_TASKS
    empty=tmp_path/'original_pins.json'
    empty.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{}}))
    with monkeypatch.context() as old_catalog:
        old_catalog.setattr(catalog,'PINS',empty)
        originals={row['family']+'/'+row['task_id']:row for row in list_tasks()}
    assert catalog.PINS==pins_path
    installations=0
    for identity,new in current.items():
        old=originals[identity]
        assert new.get('default_configuration_id')==old.get('default_configuration_id')
        for field in ('layout','objects','rules','evaluator'):
            assert new[field]==old[field]
        old_options=old.get('robot_configurations') or {'fixed':old}
        new_options=new.get('robot_configurations') or {'fixed':new}
        assert set(new_options)==set(old_options)==set(pins['tasks'][identity]['releases'])
        for key,option in new_options.items():
            installations+=1
            selected=task_definition(new['family'],new['task_id'],None if key=='fixed' else key,
                                     robot_profile='l_mount')
            assert selected['source_release']==option['source_release']
            assert option['qualification']==catalog.QUALIFICATION
            assert option['robot_requirements']['assembly_version']=='isaacsim_6_1_l_mount_v3'
            pin=option['source_release'];root=Path(pin['path'])
            assert catalog.file_hash(root/'release_manifest.json')==pin['manifest_sha256']
            manifest=release.read(root/'release_manifest.json')
            assert manifest['schema']==SCHEMA
            assert manifest['base_release']==dict(old_options[key]['source_release'],
                                                  schema=old_options[key]['release_schema'])
            for flag in ('physics_accepted','task_policy_success_claimed','inherited_qualification'):
                assert manifest[flag] is False
            old_root=Path(old_options[key]['source_release']['path'])
            old_manifest=release.read(old_root/'release_manifest.json')
            assert catalog.file_hash(old_root/'release_manifest.json')==old_options[key]['source_release']['manifest_sha256']
            old_scene=release.read(catalog.checked_file(old_root/'scene.json',old_manifest['files_sha256']['scene.json']))
            assert manifest['configuration_id']==release._configuration(old_manifest,old_scene)
            if key=='single_arm' and identity in ('card_room/task0','living_room/tower_building'):
                assert manifest['configuration_id'] is None
            if key=='fixed':
                assert 'robot_configurations' not in new
    assert installations==113
    # dexbench: the default robot profile is the original robot, so the default service is its task0.
    assert catalog.default_service_release(Path('/unused/legacy'))==Path(originals['card_room/task0']['source_release']['path'])
    assert pins_path.read_bytes()==pins_bytes
