"""Actual frozen parent roundtrips, with explicitly synthetic robot asset inputs.

These tests do not publish shared assets, import Kit or claim new robot physics.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.robot_revision import REVISION_SCHEMA, SCHEMA
from scenes.workrooms.robot_revision import assets, release
from scenes.workrooms.robot_revision.inputs import canonical_hash

ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    return path


@pytest.fixture
def original_robot_catalog(tmp_path,monkeypatch):
    # These author fixtures derive from the historical catalog. Only the robot
    # overlay is empty; the original family seed/configuration pins still apply.
    # Current-overlay consumer coverage is separate and never uses this fixture.
    from scenes.workrooms.robot_revision import catalog
    from scenes.workrooms.parameterized import catalog as parameterized
    pins=write(tmp_path/'original_robot_pins.json',
               {'schema':'robot_revision_catalog.v1','tasks':{}})
    monkeypatch.setattr(catalog,'PINS',pins)
    # The paired parameterized overlay sits above it; author fixtures derive
    # from the historical parents, so it is neutralized here as well.
    monkeypatch.setattr(parameterized,'PINS',write(tmp_path/'original_parameterized_pins.json',
               {'schema':parameterized.SCHEMA,'tasks':{}}))


@pytest.fixture
def publication(tmp_path,monkeypatch,original_robot_catalog):
    # Only the shared-location boundary and author entrypoint are substituted;
    # real parent/hash/schema/source/dependency/derivation checks stay active.
    monkeypatch.setattr(release,'RELEASE_ROOT',tmp_path)
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path/'assets')
    author=tmp_path/'author';files={}
    for name in (*release.SOURCE_FILES, *release.OBSERVER_COMPATIBILITY_SOURCES):
        p=author/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((release.SOURCE_ROOT/name).read_bytes());files[name]=assets.file_hash(p)
    manifest=write(author/'source_snapshot_manifest.json',{'schema':'complete_benchmark_source_snapshot_v1',
        'snapshot_root':str(author),'files_sha256':files})
    provenance={'schema':'robot_revision_source_provenance.v1','source_snapshot':str(author),
        'source_manifest_sha256':assets.file_hash(manifest),
        'sources_sha256':{name:files[name] for name in release.SOURCE_FILES},
        'original_assets_modified':False,'original_releases_modified':False,
        'runtime_object_writes':False,'inherited_qualification':False}
    monkeypatch.setattr(release,'_source_provenance',lambda:deepcopy(provenance))
    root=tmp_path/'assets/robots/assemblies/fr3_sharpa'/assets.VERSION
    files={}
    def item(name,value):
        p=write(root/name,value);files[str(p)]=assets.file_hash(p);return str(p),files[str(p)]
    hands={}
    for side in ('left','right'):
        hands[side]={}
        for kind in ('urdf','source'):
            p,h=item(side+'_'+kind+'.json',{'synthetic':True,'side':side})
            hands[side].update({kind+'_path':p,kind+'_sha256':h})
    assemblies={}
    for side in sorted(assets.ASSEMBLIES):
        row={}
        for kind in ('urdf','source'):
            p,h=item(side+'_'+kind+'.json',{'synthetic':True,'assembly':side})
            row.update({kind+'_path':p,kind+'_sha256':h})
        usd,h=item(side+'_usd.json',{'synthetic_usd':True})
        p,h=item(side+'_import.json',{'usd_path':usd,'usd_sha256':h,'derived_files':{usd:h}})
        row.update(usd_import_path=p,usd_import_sha256=h);assemblies[side]=row
    mount={'translation_m':[.01,.02,.03],'quaternion_wxyz':[1.,0.,0.,0.],'rpy_rad':[0.,0.,0.]}
    asset=write(root/'revision_manifest.json',{'schema':assets.SCHEMA,'assembly_version':assets.VERSION,
        'hands':hands,'assemblies':assemblies,'files_sha256':files,
        'mounts_by_side':{side:deepcopy(mount) for side in ('left','right')}})
    def publish(family,task,configuration='single_arm',*,formal=None):
        from scenes.workrooms.catalog import task_definition
        # Author from the original-robot parent; the current default overlay
        # selects the paired L-mount derivative, whose real shared robot
        # manifest is outside this fixture's substituted asset root.
        definition=task_definition(family,task,configuration,robot_profile='original')
        parent=Path(definition['source_release']['path'])
        original=release.read(parent/'scene.json');initial=release.read(parent/'initial_state.json')
        robot=deepcopy(original['robot'])
        if robot['hand_id']=='fr3_sharpa_instances':
            for instance in robot['instances'].values():instance['assembly_version']=assets.VERSION
        else:
            robot.update(assembly_version=assets.VERSION,mount_position_m=mount['translation_m'],
                mount_quaternion_wxyz=mount['quaternion_wxyz'],
                mounts_by_side={s:deepcopy(mount) for s in (('left','right') if robot['hand_id']=='dual_fr3_sharpa_w01' else ('right',))})
        revision={'schema':REVISION_SCHEMA,'revision_id':'test_only_robot_revision',
            'parent_robot_sha256':canonical_hash(original['robot']),'robot':robot,
            'joint_positions_rad':deepcopy(initial['joint_positions_rad']),
            'robot_asset_manifest_sha256':assets.file_hash(asset)}
        name=task+'_'+str(configuration)
        target=tmp_path/'published'/name
        manifest=release.prepare_release(parent,target,revision=revision,robot_asset_manifest=asset,
            configuration_id=release._configuration(__import__('bench.isaacsim.release',fromlist=['validate']).validate(parent),original),
            formal_parent_revision=formal)
        return target,manifest,parent,revision,definition
    return publish


@pytest.mark.parametrize('family,task,configuration',[
    ('mechanical_assembly','parts_sorting','single_arm'),
    ('living_room','remote_operation','dual_same_side'),
    ('laundry_room','towel_flattening','dual_opposed'),
    ('card_room','task0','single_arm'),
    ('card_room','tic_tac_toe_3x3',None),
    ('living_room','tower_extraction',None)])
def test_actual_parent_roundtrip_keeps_objects_rules_and_distinguishes_selection(publication,family,task,configuration):
    from bench.isaacsim.release import validate,task_inputs,load_evaluator
    target,manifest,parent,revision,_=publication(family,task,configuration)
    assert validate(target)['schema']==SCHEMA
    seeds=(0,5,9) if manifest['selection_kind']=='finite_seed' else (0,)
    for seed in seeds:
        _,old_scene,old_task,old_initial=task_inputs(parent,task,seed=seed)
        _,new_scene,new_task,new_initial=task_inputs(target,task,seed=seed)
        assert new_scene['robot']==revision['robot']
        assert new_initial['joint_positions_rad']==revision['joint_positions_rad']
        for key in ('objects','fixtures','cloth','cloths','workroom','physics','isaacsim'):
            assert new_scene.get(key)==old_scene.get(key)
        assert new_task.get('public_rules')==old_task.get('public_rules')
        if 'reset_case' in old_task:
            assert new_task['reset_case']['case_sha256']!=old_task['reset_case']['case_sha256']
    if manifest['task_kind']!='board_game':
        assert load_evaluator(target).__name__==load_evaluator(parent).__name__
    assert manifest['physics_accepted'] is False and manifest['inherited_qualification'] is False
    for name,h in validate(parent,assets=False)['files_sha256'].items():
        if name.startswith('evaluator/') or name in ('rules.json','physics_profile.json'):
            assert assets.file_hash(target/name)==h


def test_practice_48cases_bind_new_formal_parent_and_original_scoring(publication):
    from bench.isaacsim.release import task_inputs,load_evaluator,load_practice_inputs
    from scenes.workrooms.robot_revision.practice import verified_legacy_view
    formal=publication('card_room','polyomino_territory',None)[0]
    target,manifest,parent,_,_=publication('card_room','polyomino_territory_practice',None,
        formal={'path':str(formal),'manifest_sha256':assets.file_hash(formal/'release_manifest.json')})
    loaded=load_practice_inputs(target);assert len(loaded['cases'])==48
    old=load_practice_inputs(parent)
    assert set(loaded['cases'])==set(old['cases'])
    factory=load_evaluator(target)
    selected=list(loaded['cases'].items())
    for key,case in (selected[0],selected[23],selected[-1]):
        selector={k:case['public'][k] for k in ('active_side','case_id','split')}
        _,scene,task,initial=task_inputs(target,manifest['task_id'],practice=selector)
        assert task['practice_case']==case
        assert task['runtime_release_manifest_sha256']==assets.file_hash(target/'release_manifest.json')
        assert verified_legacy_view(task)['practice_case']==old['cases'][key]
        assert factory(task).public==case['public']


@pytest.mark.parametrize('fault',['object','ready_hand','rules','case','qualification','source','asset','config','mode'])
def test_rehashed_derivation_tampering_is_rejected(publication,fault):
    target,manifest,_,_,_=publication('mechanical_assembly','parts_sorting')
    def mutate(name,fn):
        p=target/name;value=release.read(p);fn(value);write(p,value)
        manifest['files_sha256'][name]=assets.file_hash(p);manifest['files'][name]={'sha256':assets.file_hash(p)}
    if fault=='object':mutate('scene.json',lambda x:x['objects'][0]['position_m'].__setitem__(0,99)) if 'objects' in release.read(target/'scene.json') else mutate('scene.json',lambda x:x['workroom'].__setitem__('family','other'))
    elif fault=='ready_hand':
        def hand(x):
            name=next(n for n in x['joint_positions_rad'] if not n.startswith('fr3_'));x['joint_positions_rad'][name]+=.01
        mutate('robot_revision.json',hand);manifest['robot_revision_sha256']=canonical_hash(release.read(target/'robot_revision.json'))
    elif fault=='rules':
        name=next(n for n in manifest['files_sha256'] if n.startswith('evaluator/') and n.endswith('.py'))
        p=target/name;p.write_text(p.read_text()+'\n# changed\n');manifest['files_sha256'][name]=assets.file_hash(p);manifest['files'][name]={'sha256':assets.file_hash(p)}
    elif fault=='case':
        name=manifest['cases']['5']['task'];mutate(name,lambda x:x['reset_case'].__setitem__('seed',6))
    elif fault=='qualification':manifest['physics_accepted']=True
    elif fault=='source':
        name='robot_revision_source.json';mutate(name,lambda x:x.__setitem__('source_manifest_sha256','0'*64))
    elif fault=='asset':manifest['robot_asset_manifest']['sha256']='0'*64
    elif fault=='config':manifest['configuration_id']='dual_opposed'
    elif fault=='mode':manifest['selection_kind']='fixed_game'
    write(target/'release_manifest.json',manifest)
    with pytest.raises(ValueError):release.validate_release(target,manifest)
