from copy import deepcopy

import pytest

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json
from bench.protocol.task_referee import project_task_rules
from scenes.workrooms.study import composition, release


def configuration(task_id):
    from scenes.workrooms.study.rules import build_configuration
    names = ('first', 'second') if task_id == 'book_shelving' else ('first', 'second', 'third')
    objects, targets = {}, {}
    supports = {'table': {'bounds_m': [[.2, -.7, .74], [1.1, .7, .78]]},
        'shelf_lower': {'bounds_m': [[-1.16, -.7, .73], [-.84, .7, .76]]},
        'shelf_upper': {'bounds_m': [[-1.16, -.7, 1.09], [-.84, .7, 1.12]]}}
    for i, name in enumerate(names):
        objects[name] = {'category': name, 'local_bounds_m': [[-.04, -.06, -.01], [.04, .06, .01]],
            'initial_pose': {'position_m': [.30, -.3+i*.2, .79], 'quaternion_wxyz': [1, 0, 0, 0]}}
        if task_id != 'stationery_sorting':
            objects[name]['local_bounds_m'] = [[-.04, -.01, -.06], [.04, .01, .06]]
            objects[name]['initial_pose']['position_m'][2] = .84
        support = 'table'
        if task_id == 'book_shelving' or task_id == 'book_retrieval' and i:
            support = 'shelf_lower' if i % 2 == 0 else 'shelf_upper'
        height = supports[support]['bounds_m'][1][2]
        x = .5 if support == 'table' else -1.
        if task_id == 'book_stacking':
            support = 'table' if i == 0 else names[i-1]
            height = .78+i*.02
        targets[name] = {'slot_id': name+'_slot', 'category': name,
            'bounds_m': [[x-.08, -.4+i*.2, height], [x+.08, -.2+i*.2, height+.15]],
            'support_id': support, 'support_z_m': height}
        if task_id != 'stationery_sorting':
            objects[name].update(up_axis_local=[0, 1, 0] if task_id == 'book_stacking' else [0, 0, 1],
                                 spine_axis_local=[1, 0, 0])
            targets[name].update(up_direction_world=[0, 0, 1], spine_direction_world=[1, 0, 0])
        if task_id == 'book_stacking':
            targets[name]['bounds_m'][0][1] = -.1
            targets[name]['bounds_m'][1][1] = .1
        if task_id == 'book_retrieval' and i:
            objects[name]['initial_pose']['position_m'] = [x, -.3+i*.2, height+.06]
    extra = {'stack_order': names} if task_id == 'book_stacking' else (
        {'retrieval_target_id': names[0], 'protected_ids': names[1:]} if task_id == 'book_retrieval' else {})
    return build_configuration(task_id, objects, targets, supports, **extra)


@pytest.fixture(params=composition.TASK_IDS)
def authored(tmp_path, monkeypatch, request):
    from pxr import Usd, UsdGeom, UsdPhysics
    ident = request.param
    config = configuration(ident)
    source = tmp_path/'candidate'
    source.mkdir()
    usd = source/'initial.usda'
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, '/World/StudyTask')
    paths = {}
    for name, obj in config['objects'].items():
        path = '/World/StudyTask/Objects/'+name
        paths[name] = path
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(tuple(obj['initial_pose']['position_m']))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        shape = UsdGeom.Cube.Define(stage, path+'/Collider')
        shape.GetSizeAttr().Set(1.)
        bounds = obj['local_bounds_m']
        shape.AddScaleOp().Set(tuple(bounds[1][i] - bounds[0][i] for i in range(3)))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    stage.GetRootLayer().Save()
    metadata = {'configuration': config}
    dump(source/'metadata.json', metadata)
    manifest = {'bodies': paths, 'metadata_sha256': digest(source/'metadata.json')}
    dump(source/'manifest.json', manifest)
    candidate = {'initial_usd': str(usd), 'initial_usd_sha256': digest(usd),
        'manifest_path': str(source/'manifest.json'), 'manifest_sha256': digest(source/'manifest.json')}
    dependencies = {str(source/name): digest(source/name) for name in ('metadata.json', 'manifest.json')}
    monkeypatch.setattr(composition, 'candidate_inputs', lambda task_id: (
        deepcopy(candidate), deepcopy(manifest), deepcopy(metadata), deepcopy(dependencies)))
    inputs = composition.build_task_inputs(ident, seed=29)
    return ident, inputs


@pytest.fixture
def frozen(authored, tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    ident, inputs = authored
    root = tmp_path/'release'
    release.prepare_task_release(ident, root, seed=29, inputs=inputs)
    return root, ident


def test_composition_and_canonical_frozen_loader_preserve_reset_rules_and_parent(frozen):
    root, ident = frozen
    manifest = validate(root)
    entry, scene, task, initial = task_inputs(root, ident, seed=97)
    assert manifest['status'] == 'prepared_unvalidated' and manifest['physics_accepted'] is False
    assert entry['task_kind'] == 'single_agent'
    assert scene['scenario']['seed'] == task['episode_seed'] == 97
    assert scene['workroom'] == composition.environment_reference()
    assert set(initial) == {'joint_positions_rad'} and len(initial['joint_positions_rad']) == 29
    assert scene['objects'] == [] and scene['robot']['hand_id'] == 'sharpa_w01_right'
    assert set(task['evaluation']['body_paths']) == set(configuration(ident)['objects'])
    assert project_task_rules(task)['task_id'] == ident
    scorer = load_evaluator(root)(task)
    assert scorer.__class__.__module__.startswith('_isaac_frozen_evaluator_')
    assert scorer.inspect_conditions()['success'] is False
    assert (root/'evaluator/native_observer.py').is_file()
    with pytest.raises(ValueError, match='nominal reset'):
        task_inputs(root, ident, phase='qualification_probe')


def test_frozen_release_loading_is_independent_of_the_current_author_layout(frozen, monkeypatch):
    from scenes.workrooms.study import task_scene

    root, ident = frozen
    expected = task_inputs(root, ident)

    def author_must_not_run(*args, **kwargs):
        raise AssertionError('A frozen release must retain its own authored layout')

    monkeypatch.setattr(task_scene, 'configuration', author_must_not_run)
    monkeypatch.setattr(composition, 'candidate_inputs', author_must_not_run)
    assert task_inputs(root, ident) == expected


def test_immutable_inventory_and_frozen_readback_cannot_be_tampered(frozen):
    root, _ = frozen
    with (root/'evaluator/native_observer.py').open('a') as stream:
        stream.write('\n# changed\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_evaluator(root)


def test_rehashing_cannot_hide_an_omitted_task_dependency(frozen):
    root, _ = frozen
    inventory = load_json(root/'asset_file_hashes.json')
    removed = next(p for p in inventory if p.endswith('/metadata.json'))
    del inventory[removed]
    dump(root/'asset_file_hashes.json', inventory)
    manifest = load_json(root/'release_manifest.json')
    value = digest(root/'asset_file_hashes.json')
    manifest['files']['asset_file_hashes.json']['sha256'] = value
    manifest['files_sha256']['asset_file_hashes.json'] = value
    del manifest['asset_files_sha256'][removed]
    dump(root/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='inventory is incomplete'):
        validate(root)


def test_runtime_mapping_and_authored_reset_must_match_frozen_rules(authored):
    _, inputs = authored
    scene, task = inputs['scene'], inputs['task']
    bad = deepcopy(task)
    bad['evaluation']['support_paths']['table'] = '/World/Room/Geometry/other/Collider'
    with pytest.raises(ValueError, match='fixed native mapping'):
        release.initial_state(scene, bad)
    from pxr import Usd, UsdGeom
    layer = scene['task_layers'][0]
    stage = Usd.Stage.Open(layer['usd_path'])
    first = next(iter(task['evaluation']['body_paths'].values()))
    UsdGeom.Xformable(stage.GetPrimAtPath(first)).GetOrderedXformOps()[0].Set((.9, 0., .9))
    stage.GetRootLayer().Save()
    layer['sha256'] = digest(layer['usd_path'])
    with pytest.raises(ValueError, match='authored native body pose'):
        release.validate_authored_reset(scene, task)


def test_canonical_load_before_simulation_app_does_not_require_usd(frozen, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def without_usd(name, *args, **kwargs):
        if name == 'pxr' or name.startswith('pxr.'):
            raise ModuleNotFoundError('USD is unavailable before SimulationApp startup')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_usd)
    root, ident = frozen
    assert task_inputs(root, ident)[2]['id'] == ident


def test_composition_rejects_unreviewed_robot_stand_override():
    with pytest.raises(ValueError, match='stand; overrides'):
        composition.build_task_inputs('book_stacking', stand_path='/tmp/custom.usda')


def test_reviewed_ik_roundoff_is_accepted_without_allowing_a_different_pose(authored):
    _, inputs = authored
    scene = deepcopy(inputs['scene'])
    scene['robot']['arm_home_rad'][1] += 5.e-16
    assert release.initial_state(scene, inputs['task']) == inputs['initial_state']
    scene['robot']['arm_home_rad'][1] += 1.e-6
    with pytest.raises(ValueError, match='reviewed original'):
        release.initial_state(scene, inputs['task'])


@pytest.mark.parametrize('change', [
    lambda scene: scene['fixtures'][0].update(asset_root='/tmp/alternate'),
    lambda scene: scene['robot'].update(base_position_m=[0., 0., .78]),
    lambda scene: scene['task_layers'].append(deepcopy(scene['task_layers'][0])),
])
def test_supplied_inputs_cannot_bypass_original_robot_stand_or_task_layer(authored, change):
    _, inputs = authored
    scene = deepcopy(inputs['scene'])
    change(scene)
    with pytest.raises(ValueError, match='reviewed original|one task-owned'):
        release.initial_state(scene, inputs['task'])
