"""Actual-asset CPU checks, not native stability or robot qualification."""
from copy import deepcopy
from functools import lru_cache
import importlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.study import placement_variants as author
from scenes.workrooms.study import rules

PARENTS = {task: Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study')/task/version
           for task, version in (('book_retrieval','v3'), ('drawer_retrieval','v4'), ('reading_stand','v1'))}


@lru_cache
def parent(task):
    root = PARENTS[task]
    if not root.is_dir(): pytest.skip('Shared original assets unavailable')
    _, values = author._parent(root)
    layer = next(row for row in values['scene']['task_layers'] if row['id'] == 'task')
    candidate = Path(values['scene']['study_task_provenance']['candidate']['manifest_path'])
    manifest = load_json(candidate)
    md = load_json(candidate.parent/manifest['metadata_file'])
    return Usd.Stage.Open(layer['usd_path']), md


@pytest.mark.parametrize('task', author.TASKS)
@pytest.mark.parametrize('seed', range(10))
def test_actual_assets_only_translate_loose_body(task, seed):
    original, md = parent(task)
    original_properties = author._stage_values(original)
    derived, result = author.derive_stage(original, md, seed)
    assert author._stage_values(original) == original_properties
    moving = author.MOVABLE[task]; config = result['configuration']
    assert config == author._configuration(md['configuration'], result['seed_plan'])
    assert config['rules'] == md['configuration']['rules']
    assert config['targets'] == md['configuration']['targets']
    assert config.get('mechanism') == md['configuration'].get('mechanism')
    for name, path in result['actors'].items():
        old = md['configuration']['objects'][name]['initial_pose']
        new = result['actual_body_poses_m'][name]
        delta = np.array(new['position_m'])-old['position_m']
        np.testing.assert_allclose(delta, [*author.OFFSETS_XY_M[seed],0] if name == moving else [0,0,0], atol=1e-12)
        np.testing.assert_allclose(rules.rotation(new['quaternion_wxyz']), rules.rotation(old['quaternion_wxyz']), atol=1e-12)
        np.testing.assert_allclose(author.collision_bounds(original,path), author.collision_bounds(derived,path),atol=1e-12)
        for attr in original.GetPrimAtPath(path).GetAttributes():
            if attr.GetName().startswith(('physics:', 'physx')):
                assert str(attr.Get()) == str(derived.GetPrimAtPath(path).GetAttribute(attr.GetName()).Get())
    assert set(result['actual_body_poses_m']) == set(md['configuration']['objects'])
    assert {x['object_id'] for x in result['static_geometry']['bodies']} == set(result['actors'])
    if seed == 0:
        assert config == md['configuration']
        assert author._stage_values(derived) == original_properties
    if task == 'book_retrieval':
        rules.validate_configuration(config)
    if task == 'drawer_retrieval':
        b = np.array(config['objects']['notebook']['local_bounds_m']) + config['objects']['notebook']['initial_pose']['position_m']
        drawer = np.array(config['objects']['drawer']['local_bounds_m']) + config['objects']['drawer']['initial_pose']['position_m']
        assert np.all(b[0] > drawer[0]) and np.all(b[1] < drawer[1])
    for row in result['interaction_targets']:
        assert derived.GetPrimAtPath(row['prim_path'])
        assert row['type'] in ('actual_surface','goal')
        assert row['reference_state'] in ('initial','opened')
        assert np.linalg.norm(row['approach_direction_local']) == pytest.approx(1.)
    assert len({x['label'] for x in result['interaction_targets']}) == len(result['interaction_targets'])
    for path in result['fixed_support_paths']:
        prim = derived.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        while prim and not prim.IsPseudoRoot():
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            prim = prim.GetParent()


@pytest.mark.parametrize('task', author.TASKS)
def test_positions_distinct_with_conservative_envelope(task):
    _, md = parent(task)
    plans = [author.plan_case(task, i, md['configuration']) for i in range(10)]
    assert len({tuple(p['translation_xy_m']) for p in plans}) == 10
    assert max(abs(x) for p in plans for x in p['translation_xy_m']) <= .006
    assert plans == [author.plan_case(task, i, md['configuration']) for i in range(10)]


@pytest.mark.parametrize('seed', [-1,10,True,1.,'1',None])
def test_invalid_seed(seed):
    with pytest.raises(ValueError): author.plan_case('book_retrieval',seed,{})


def test_historical_three_tasks_stay_separate():
    from scenes.workrooms.study import seed_variants
    assert not set(seed_variants.TASKS) & set(author.TASKS)
    with pytest.raises(ValueError): author.plan_case('book_stacking',0,{})


def test_protected_property_guard_rejects_geometry_mass_and_anchor_change():
    original, md = parent('drawer_retrieval')
    for kind in ('mass','geometry','joint'):
        stage, result = author.derive_stage(original, md, 2)
        if kind == 'mass': UsdPhysics.MassAPI(stage.GetPrimAtPath(author.ROOT+'/Objects/notebook')).GetMassAttr().Set(.12)
        elif kind == 'geometry': UsdGeom.Cube(stage.GetPrimAtPath(author.ROOT+'/Objects/notebook/Geometry/pages/Collider')).GetSizeAttr().Set(2.)
        else: UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(author.ROOT+'/DrawerSlide')).GetUpperLimitAttr().Set(.2)
        with pytest.raises(ValueError,match='protected'):
            author.assert_translation_only(original,stage,author.ROOT+'/Objects/notebook',result['actual_body_poses_m']['notebook']['position_m'])


def frozen_module(tmp_path, monkeypatch, task):
    package = 'frozen_'+task
    target = tmp_path/package; target.mkdir(); (target/'__init__.py').write_text('')
    sources = author.evaluator_sources(PARENTS[task],task)
    for path, source in sources.items(): (target/path).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(package+'.rules'), importlib.import_module(package+'.sequence_evaluator'), sources


@pytest.mark.parametrize('task',author.TASKS)
def test_frozen_evaluator_accepts_all_ten_preserves_scoring_and_seedzero(tmp_path,monkeypatch,task):
    module,evaluator,sources=frozen_module(tmp_path,monkeypatch,task)
    _,md=parent(task)
    original = {str(p.relative_to(PARENTS[task]/'evaluator')):p.read_text() for p in (PARENTS[task]/'evaluator').rglob('*.py')}
    changed = {n for n in sources if sources[n] != original[n]}
    assert changed == (set() if task=='book_retrieval' else {('drawer_rules.py' if task=='drawer_retrieval' else 'reading_stand_rules.py')})
    for seed in range(10):
        config=author._configuration(md['configuration'],author.plan_case(task,seed,md['configuration']))
        assert module.validate_configuration(config)==config
        judge=evaluator.SequenceEvaluator({'id':task,'public_rules':config['rules'],'evaluation':{'configuration':config}})
        assert judge.config==config
    assert module.validate_configuration(md['configuration'])==md['configuration']


@pytest.mark.parametrize('task',['drawer_retrieval','reading_stand'])
@pytest.mark.parametrize('change',['xy','z','yaw','mass_bounds','threshold','mechanism','extra','stale_hash'])
def test_rehashed_illegal_reset_and_other_fields_rejected(tmp_path,monkeypatch,task,change):
    module,_,_=frozen_module(tmp_path,monkeypatch,task)
    _,md=parent(task); config=author._configuration(md['configuration'],author.plan_case(task,3,md['configuration']))
    row=config['objects'][author.MOVABLE[task]]
    if change=='xy': row['initial_pose']['position_m'][0]+=.0001
    elif change=='z': row['initial_pose']['position_m'][2]+=.0001
    elif change=='yaw': row['initial_pose']['quaternion_wxyz']=[0.,0.,0.,1.]
    elif change=='mass_bounds': row['local_bounds_m'][0][0]-=.001
    elif change=='threshold': config['rules']['thresholds']['minimum_stable_s']=.001
    elif change=='mechanism': config['mechanism']['new_field']=True
    elif change=='extra': config['extra']=1
    if change!='stale_hash': config['configuration_sha256']=rules.canonical_hash({k:v for k,v in config.items() if k!='configuration_sha256'})
    else: config['configuration_sha256']='0'*64
    with pytest.raises(ValueError): module.validate_configuration(config)


@pytest.mark.parametrize('task',author.TASKS)
def test_actual_texture_only_preserves_category_color_and_physics(tmp_path,task):
    stage,md=author.derive_stage(*parent(task),5)
    before=author._stage_values(stage)
    visual=author._appearance(stage,5,tmp_path)
    author.assert_finish_only(before,stage,visual)
    after=author._stage_values(stage)
    assert visual['category_colors_and_labels_unchanged'] and visual['physical_materials_unchanged']
    assert visual['changed_materials']
    for path, row in before.items():
        assert after[path]['type']==row['type'] and after[path]['apis']==row['apis']
        for name,value in row['attributes'].items():
            if name!='inputs:roughness': assert after[path]['attributes'][name]==value
    assert author.file_hash(tmp_path/'finish_roughness.png') in visual['dependencies_sha256'].values()


def test_finish_guard_rejects_color_and_physical_material_changes(tmp_path):
    from pxr import UsdShade, Sdf, Gf
    stage,_=author.derive_stage(*parent('reading_stand'),1)
    before=author._stage_values(stage)
    appearance=author._appearance(stage,1,tmp_path)
    shader=UsdShade.Material(stage.GetPrimAtPath(appearance['changed_materials'][0])).ComputeSurfaceSource()[0]
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.9,.1,.3))
    with pytest.raises(ValueError,match='protected'): author.assert_finish_only(before,stage,appearance)


@pytest.mark.parametrize('task',author.TASKS)
def test_original_seedzero_reset_baseline_result_is_preserved(tmp_path,monkeypatch,task):
    _,evaluator,_=frozen_module(tmp_path,monkeypatch,task)
    _,md=parent(task); config=md['configuration']
    observation={'schema':rules.OBSERVATION_SCHEMA,'configuration_sha256':config['configuration_sha256'],
        'sampling_complete':True,'contacts_complete':True,'physics_valid':True,'robot_only_actions':True,
        'object_pose_writes':False,'episode_id':'synthetic_cpu_baseline','step_index':0,'robot_action_count':0,
        'budget_exhausted':False,'time_s':0.,'physics_dt_s':config['physics_dt_s'],'contacts':[],
        'objects':{name:{**deepcopy(row['initial_pose']),'linear_velocity_m_s':[0.,0.,0.],
                    'angular_velocity_rad_s':[0.,0.,0.]} for name,row in config['objects'].items()}}
    if task=='drawer_retrieval':
        observation['joints']={config['mechanism']['joint_path']:{'position_m':0.,'velocity_m_s':0.,
            'kind':'prismatic','source':'derived_from_native_rigid_transforms_and_COM_velocities'}}
    spec={'id':task,'public_rules':config['rules'],'evaluation':{'configuration':config}}
    actual=evaluator.SequenceEvaluator(spec,trusted_normalized_input=True).update(deepcopy(observation))
    package=tmp_path/('original_'+task);package.mkdir();(package/'__init__.py').write_text('')
    for p in (PARENTS[task]/'evaluator').rglob('*.py'): (package/p.name).write_text(p.read_text())
    original=importlib.import_module(package.name+'.sequence_evaluator')
    expected=original.SequenceEvaluator(spec,trusted_normalized_input=True).update(deepcopy(observation))
    assert actual==expected
    assert not actual['terminal'] and actual['referee']['state']=='ongoing'


@pytest.mark.parametrize('task',author.TASKS)
def test_prepare_compose_actual_seedzero_and_seedfive(tmp_path,monkeypatch,task):
    import scripts.maintenance.freeze_project_sources as frozen
    monkeypatch.setattr(frozen,'verify_if_snapshot',lambda *a,**k:{'verified':True,'test_fixture_only':True})
    monkeypatch.setattr(author,'ASSET_ROOT',tmp_path)
    _,values=author._parent(PARENTS[task])
    original=next(x for x in values['scene']['task_layers'] if x['id']=='task')
    outputs=[]
    for seed in (0,5):
        manifest=author.prepare_asset(PARENTS[task],seed,tmp_path/f'seed{seed}')
        result=author.compose_case(PARENTS[task],manifest,'single_arm');outputs.append(result)
        assert result['initial_state']==values['initial_state']
        assert result['task']['public_rules']==values['task']['public_rules']
        assert result['task']['public_goal']==values['task']['public_goal']
        for key in ('physics','robot','workroom','isaacsim','cameras'):
            assert result['scene'].get(key)==values['scene'].get(key)
        with pytest.raises(ValueError,match='configuration'): author.compose_case(PARENTS[task],manifest,'dual_opposed')
        if seed==0:
            record=load_json(manifest)
            assert Path(record['initial_usd']).read_bytes()==Path(original['usd_path']).read_bytes()
            assert result['task']['evaluation']==values['task']['evaluation']
    assert outputs[0]['evaluator_sources']==outputs[1]['evaluator_sources']
