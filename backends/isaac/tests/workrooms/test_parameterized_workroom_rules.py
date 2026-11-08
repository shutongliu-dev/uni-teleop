from copy import deepcopy
import math

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.parameterized.spec import sample_parameters, canonical_hash
from scenes.workrooms.parameterized.study_configuration import validate_configuration, DRAWER_FIXTURES
from scenes.workrooms.parameterized.usd_authoring import group_layout, transform_point
from scenes.workrooms.parameterized.workroom_rules import (
    laboratory_rules, PUBLIC_MASS_TEXT_VERSION, checked_public_mass_text_version)


def study_configuration(task_id, seed=1):
    from scenes.workrooms.study import drawer_rules, reading_stand_rules
    canonical=(drawer_rules if task_id=='drawer_retrieval' else reading_stand_rules).configuration()
    actual=deepcopy(canonical);params=sample_parameters('study',task_id,'single_arm',seed)
    drawer=task_id=='drawer_retrieval';names=['drawer','notebook'] if drawer else ['book']
    group={'id':'drawer_supply' if drawer else 'book','paths':['/World/StudyTask/Objects/'+n for n in names]}
    if drawer:
        group['yaw_policy']='fixed_axis_reference';group['paths']+=['/World/StudyTask/Fixtures/'+n for n in DRAWER_FIXTURES]
    row={**group,**group_layout(params,group),'pivot_m':canonical['objects'][names[0]]['initial_pose']['position_m']}
    poses={}
    for name in names:
        pose=actual['objects'][name]['initial_pose'];pose['position_m']=transform_point(pose['position_m'],row)
        w,x,y,z=pose['quaternion_wxyz'];c,s=math.cos(row['yaw_rad']/2),math.sin(row['yaw_rad']/2)
        pose['quaternion_wxyz']=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
        poses[name]={'prim_path':'/World/StudyTask/Objects/'+name,**deepcopy(pose)}
    if drawer:
        actual['mechanism']['closed_position_m']=transform_point(actual['mechanism']['closed_position_m'],row)
        for key in ['cabinet_front_x_m','notebook_extraction_max_x_m']:
            actual['mechanism'][key]+=row['translation_xy_m'][0]
    actual['parameterized_physics']={'parameters':params,'layout':[row],'actual_body_poses_m':poses}
    actual['configuration_sha256']=canonical_hash({k:v for k,v in actual.items() if k!='configuration_sha256'})
    return canonical,actual


@pytest.mark.parametrize('task_id',['drawer_retrieval','reading_stand'])
@pytest.mark.parametrize('seed',range(1,10))
def test_study_validates_real_sampled_configuration(task_id,seed):
    canonical,actual=study_configuration(task_id,seed)
    assert validate_configuration(actual,canonical)==actual
    assert actual['objects']!=canonical['objects']


@pytest.mark.parametrize('field',['position','rule','target'])
def test_rehashed_study_changes_rejected(field):
    canonical,actual=study_configuration('drawer_retrieval')
    if field=='position':actual['objects']['notebook']['initial_pose']['position_m'][0]+=.01
    if field=='rule':actual['mechanism']['travel_m']+=.01
    if field=='target':actual['targets']['notebook']['support_z_m']+=.01
    actual['configuration_sha256']=canonical_hash({k:v for k,v in actual.items() if k!='configuration_sha256'})
    with pytest.raises(ValueError):validate_configuration(actual,canonical)


def test_laboratory_all_seeds_share_source_and_report_actual_masses():
    from pathlib import Path
    from scenes.workrooms.laboratory import rules
    reference=None
    for seed in range(10):
        factor=sample_parameters('laboratory','solid_weighing','single_arm',seed)['physics']['mass_factor']
        rows=rules.object_properties('solid_weighing')
        metadata={'root_path':'/World/LaboratoryTask','objects':[{'id':r['id'],'mass_kg':r['nominal_mass_kg']*factor} for r in rows],
            'task_parameters':{'pellet_ids':['pellet_'+str(n).zfill(2) for n in range(24)],'pellet_mass_kg':.0005*factor,
            'target_mass_kg':.006,'tolerance_mass_kg':.0006,'spill_mass_limit_kg':.001}}
        values={'task':{'id':'solid_weighing','episode_seed':0,'asset_metadata':metadata,'evaluation':{}},
            'evaluator_sources':{'rules.py':Path(rules.__file__).read_text(),'sequence_evaluator.py':'# fixture\n'}}
        laboratory_rules(values, public_mass_text_version=PUBLIC_MASS_TEXT_VERSION)
        if reference is None:reference=values['evaluator_sources']
        assert values['evaluator_sources']==reference
        props=values['task']['public_rules']['referee']['object_properties']
        assert props[0]['nominal_mass_kg']==pytest.approx(.028*factor)
        assert props[-1]['nominal_mass_kg']==pytest.approx(.0005*factor)
        assert metadata['task_parameters']['target_mass_kg']==.006
        text=str(values['task']['public_rules']['referee']['success_conditions'])
        assert '1克（'+str(math.floor(.001/(.0005*factor)))+'颗）' in text
        assert '扣除'+format(.028*factor*1000,'.6g')+'克空舟' in text


def test_published_weighing_seed1_public_counts_and_tare_match_actual_masses():
    """This real seed's second spilled pellet exceeds the unchanged 1 g limit."""
    import json
    from pathlib import Path
    root=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/author_general_remaining01/laboratory/releases/solid_weighing/single_arm')
    manifest=load_json(root/'release_manifest.json')
    task=load_json(root/manifest['cases']['1']['task'])
    before=deepcopy(task)
    parent=Path(manifest['base_release']['path'])
    sources={str(path.relative_to(parent/'evaluator')):path.read_text() for path in (parent/'evaluator').rglob('*.py')}
    values={'task':task,'evaluator_sources':sources}
    laboratory_rules(values, public_mass_text_version=PUBLIC_MASS_TEXT_VERSION)
    params=task['asset_metadata']['task_parameters']; mass=params['pellet_mass_kg']
    assert mass*1000==pytest.approx(.5301654455251992)
    assert mass<=params['spill_mass_limit_kg']<2*mass
    assert params['target_mass_kg']==.006 and params['tolerance_mass_kg']==.0006
    text=str(task['public_rules']['referee']['success_conditions'])
    assert '6±0.6克，即11–12颗' in text and '1克（1颗）' in text
    assert '扣除29.6893克空舟' in text
    assert '1克（2颗）' not in text and '扣除28克空舟' not in text
    assert task['reset_case']==before['reset_case']
    assert task['asset_metadata']==before['asset_metadata']
    assert task['evaluation']['metadata']==before['evaluation']['metadata']
    assert task['public_rules']['referee']['thresholds']==before['public_rules']['referee']['thresholds']
    assert task['evaluation']['thresholds']==before['evaluation']['thresholds']
    # Real ten-seed inventories retain an integer quantity meeting the same goal.
    for seed in range(10):
        row=load_json(root/manifest['cases'][str(seed)]['task'])
        p=row['asset_metadata']['task_parameters']; quantities=range(len(p['pellet_ids'])+1)
        assert any(abs(n*p['pellet_mass_kg']-p['target_mass_kg'])<=p['tolerance_mass_kg'] for n in quantities)


@pytest.mark.parametrize('task_id', ['cap_placement', 'pipette_transfer', 'sample_sorting',
                                   'solid_weighing', 'tube_clamp_transfer', 'vial_racking'])
def test_prior_parameterized_laboratory_assets_reconstruct_exact_old_evaluator_bytes(task_id):
    """New source must continue to admit all six real prior parameterized tasks."""
    import json
    from pathlib import Path
    from bench.isaacsim.validation_epoch import ValidationEpoch
    from scenes.workrooms.parameterized.workroom import compose_case
    base=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105')
    pair=base/'pair_laboratory01/releases/laboratory'/task_id/'single_arm/release_manifest.json'
    original=Path(load_json(pair)['base_release']['path'])
    manifest=load_json(original/'release_manifest.json')
    task=load_json(original/manifest['cases']['1']['task'])
    expected={str(p.relative_to(original/'evaluator')):p.read_text() for p in (original/'evaluator').rglob('*.py')}
    with ValidationEpoch('prior_laboratory_'+task_id):
        values=compose_case(manifest['base_release']['path'],task['reset_case']['asset_reference']['path'],'single_arm')
    assert values['evaluator_sources']==expected
    assert 'public_mass_text_version' not in values['parameterized_physics']
    if task_id=='solid_weighing':
        text=str(values['task']['public_rules']['referee']['success_conditions'])
        assert '1克（2颗）' in text and '扣除28克空舟' in text


@pytest.mark.parametrize('version', [None, '', True, 'latest', 'solid_weighing_actual_mass_v2'])
def test_explicit_unknown_mass_text_version_rejected(version):
    with pytest.raises(ValueError, match='public mass text version'):
        checked_public_mass_text_version({'family':'laboratory','task_id':'solid_weighing',
                                         'public_mass_text_version':version})


def test_mass_text_version_only_applies_to_explicit_new_weighing_assets():
    assert checked_public_mass_text_version({'family':'laboratory','task_id':'solid_weighing'}) is None
    assert checked_public_mass_text_version({'family':'laboratory','task_id':'solid_weighing',
        'public_mass_text_version':PUBLIC_MASS_TEXT_VERSION})==PUBLIC_MASS_TEXT_VERSION
    for family,task in [('laboratory','sample_sorting'),('kitchen','solid_weighing')]:
        with pytest.raises(ValueError, match='public mass text version'):
            checked_public_mass_text_version({'family':family,'task_id':task,
                                             'public_mass_text_version':PUBLIC_MASS_TEXT_VERSION})


@pytest.mark.parametrize('rehash', [False, True])
def test_old_asset_cannot_claim_new_text_version_without_matching_task_recipe(tmp_path, rehash):
    import json
    from pathlib import Path
    from bench.isaacsim.validation_epoch import ValidationEpoch
    from scenes.workrooms.parameterized.workroom import compose_case, PROJECTIONS
    root=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/author_general_remaining01/laboratory/releases/solid_weighing/single_arm')
    release=load_json(root/'release_manifest.json')
    task=load_json(root/release['cases']['1']['task'])
    manifest=load_json(Path(task['reset_case']['asset_reference']['path']))
    manifest['public_mass_text_version']=PUBLIC_MASS_TEXT_VERSION
    if rehash:
        data=load_json(Path(manifest['metadata']))
        manifest['physical_instance_sha256']=canonical_hash({'parameters':manifest['parameter_instance'],
            'applied':manifest['applied_parameters'],'projections':{key:data[key] for key in PROJECTIONS},
            'dependencies':manifest['dependencies_sha256'],'public_mass_text_version':PUBLIC_MASS_TEXT_VERSION})
    fake=tmp_path/'manifest.json';fake.write_text(json.dumps(manifest))
    with ValidationEpoch('reject_relabelled_mass_text'):
        with pytest.raises(ValueError, match='physical recipe' if rehash else 'projections disagree'):
            compose_case(release['base_release']['path'],fake,'single_arm')


@pytest.mark.parametrize('task_id', ['breakfast_serving', 'sample_sorting', 'nist_connector_insertion'])
def test_release_binds_episode_seed_and_keeps_frozen_evaluator_checks(task_id, monkeypatch):
    """Exercise real canonical adapters and frozen constructors at publication boundary."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from bench.isaacsim.release import _frozen_evaluator_module
    from scenes.workrooms.initial_randomization import release

    pins = load_json(Path(release.__file__).with_name('catalog_pins.json'))
    published = Path(pins[task_id]['releases']['single_arm']['path'])
    manifest = load_json(published/'release_manifest.json')
    parent = Path(manifest['base_release']['path'])
    canonical_task = load_json(published/manifest['cases']['0']['task'])
    asset = canonical_task['reset_case']['asset_reference']['path']
    values = release.adapter(task_id).compose_case(parent, asset, 'single_arm')
    values['parameterized_physics'] = {'parameters': sample_parameters(
        manifest['family'], task_id, 'single_arm', 0)}
    monkeypatch.setattr(release, 'adapter', lambda _: SimpleNamespace(
        compose_case=lambda *_: deepcopy(values)))
    evaluator = _frozen_evaluator_module(parent).SequenceEvaluator
    for seed in range(10):
        task = release._compose(parent, asset, task_id, seed, 'single_arm')[1]
        assert task['episode_seed'] == task['evaluation']['episode_seed'] == seed
        assert task['evaluation']['metadata'] == values['task']['evaluation']['metadata']
        assert task['evaluation']['thresholds'] == values['task']['evaluation']['thresholds']
        evaluator(task)
        corrupt = deepcopy(task)
        corrupt['evaluation']['thresholds']['unexpected_threshold'] = 1
        with pytest.raises(ValueError):
            evaluator(corrupt)
