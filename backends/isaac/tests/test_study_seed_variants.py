"""CPU checks of actual frozen study assets; no native/manipulation claims."""
from copy import deepcopy
from functools import lru_cache
import importlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest

from bench.paths import load_json, usd_layer_identity, usd_value_text
from scenes.workrooms.study import rules
from scenes.workrooms.study import seed_variants as author
from scenes.workrooms.study.task_scene import CANDIDATES


@lru_cache
def parent(task):
    root = Path(CANDIDATES[task]['root'])
    if not root.is_dir():
        pytest.skip('Original shared study assets unavailable')
    assert author.file_hash(root/'manifest.json') == CANDIDATES[task]['manifest_sha256']
    manifest = load_json(root/'manifest.json')
    assert author.file_hash(root/'initial.usda') == manifest['initial_usd_sha256']
    return Usd.Stage.Open(str(root/'initial.usda')), load_json(root/'metadata.json')


def attributes(stage, prefix, physical_only=False):
    result = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
        for a in p.GetAttributes():
            if a.HasAuthoredValue() and (not physical_only or a.GetName().startswith(('physics:', 'physx'))):
                result[str(a.GetPath())] = usd_value_text(a.Get())  # dexbench: asset paths as resolved files
    return result


@pytest.mark.parametrize('task', author.TASKS)
def test_plan_ten_distinct_reproducible_cases_and_pencil_floor(task):
    _, md = parent(task)
    plans = [author.plan_case(task, s, md['configuration']) for s in range(10)]
    assert plans == [author.plan_case(task, s, md['configuration']) for s in range(10)]
    assert len({p['plan_sha256'] for p in plans}) == 10
    for plan in plans:
        for name, obj in plan['objects'].items():
            assert max(map(abs, obj['translation_xy_m'])) <= .012
            assert abs(obj['yaw_rad']) <= np.deg2rad(6)
            if name == 'pencil':
                assert .015*obj['uniform_scale'] >= .015


@pytest.mark.parametrize('seed', [-1, 10, True, 1., '1', None])
def test_seed_rejects_unsupported_and_nonintegers(seed):
    with pytest.raises(ValueError):
        author.plan_case('stationery_sorting', seed, {})


def test_other_tasks_not_randomized():
    with pytest.raises(ValueError):
        author.plan_case('book_retrieval', 0, {})


@pytest.mark.parametrize('task', author.TASKS)
def test_shared_interaction_reference_and_dynamic_inventory_contract(task):
    stage, md = author.derive_stage(*parent(task), 1)
    actual = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert {p['prim_path'] for p in md['actual_body_poses_m'].values()} == actual
    assert {p['object_id'] for p in md['static_geometry']['bodies']} == set(md['actors'])
    for row in md['interaction_targets']:
        assert row['type'] in ('actual_surface', 'goal', 'pouring_edge')
        assert row['reference_state'] in ('initial', 'opened')
        assert len(row['local_position_m']) == len(row['approach_direction_local']) == 3
        assert np.linalg.norm(row['approach_direction_local']) == pytest.approx(1.)


@pytest.mark.parametrize('task', author.TASKS)
@pytest.mark.parametrize('seed', range(10))
def test_actual_geometry_reset_mass_targets_and_unchanged_fixtures(task, seed):
    source, original = parent(task)
    before = attributes(source, author.ROOT)
    stage, md = author.derive_stage(source, original, seed)
    assert attributes(source, author.ROOT) == before
    assert attributes(stage, author.ROOT+'/Fixtures') == attributes(source, author.ROOT+'/Fixtures')
    assert rules.validate_configuration(md['configuration']) == md['configuration']
    assert md['configuration']['rules']['thresholds'] == original['configuration']['rules']['thresholds']
    assert md['configuration']['supports'] == original['configuration']['supports']
    if seed == 0:
        assert md['configuration'] == original['configuration']
        assert attributes(stage, author.ROOT) == attributes(source, author.ROOT)
    for name, path in md['actors'].items():
        scale = md['seed_plan']['objects'][name]['uniform_scale']
        expected = np.asarray(author.collision_bounds(source, path))*scale
        actual = np.asarray(author.collision_bounds(stage, path))
        np.testing.assert_allclose(actual, expected, atol=1.e-7, rtol=1.e-6)
        src_mass = UsdPhysics.MassAPI(source.GetPrimAtPath(path))
        got_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        assert got_mass.GetMassAttr().Get() == pytest.approx(src_mass.GetMassAttr().Get()*scale**3, rel=2.e-7)
        np.testing.assert_allclose(sorted(got_mass.GetDiagonalInertiaAttr().Get()),
                                   sorted(np.array(src_mass.GetDiagonalInertiaAttr().Get())*scale**5), rtol=2.e-6)
        assert not any(op.GetOpType() == UsdGeom.XformOp.TypeScale for op in UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            source_prim = source.GetPrimAtPath(prim.GetPath())
            assert prim.GetTypeName() == source_prim.GetTypeName()
            for attr in prim.GetAttributes():
                if attr.HasAuthoredValue() and attr.GetName().startswith('physx'):
                    assert attr.Get() == source_prim.GetAttribute(attr.GetName()).Get()
        actual_pose = md['actual_body_poses_m'][name]
        expected_pose = md['configuration']['objects'][name]['initial_pose']
        np.testing.assert_allclose(actual_pose['position_m'], expected_pose['position_m'], atol=1.e-10)
        np.testing.assert_allclose(rules.rotation(actual_pose['quaternion_wxyz']), rules.rotation(expected_pose['quaternion_wxyz']), atol=1.e-10)
        lo, hi = rules._reset_bounds(md['configuration']['objects'][name])
        assert lo[2] == pytest.approx(.78, abs=2.e-9)
        assert lo[0] >= .20 and hi[0] <= 1.10 and lo[1] >= -1 and hi[1] <= 1
    groups = {row['operation_group'] for row in md['interaction_targets'] if row['required']}
    assert groups == {name+suffix for name in md['actors'] for suffix in ('_grasp', '_placement')}
    assert all(stage.GetPrimAtPath(row['prim_path']) for row in md['interaction_targets'])
    if task == 'book_stacking':
        config = md['configuration']; z = .78
        for name in config['stack_order']:
            assert config['targets'][name]['support_z_m'] == pytest.approx(z)
            bounds = config['objects'][name]['local_bounds_m']; z += bounds[1][1]-bounds[0][1]
    if task == 'book_shelving':
        assert md['configuration']['targets'] == original['configuration']['targets']
        for name, obj in md['configuration']['objects'].items():
            lo, hi = obj['local_bounds_m']; target = md['configuration']['targets'][name]
            assert np.all(np.subtract(hi, lo) <= np.subtract(*target['bounds_m'][::-1]))


@pytest.mark.parametrize('task', author.TASKS)
def test_roughness_texture_is_real_and_does_not_change_physics_colors_or_labels(task, tmp_path):
    stage, _ = author.derive_stage(*parent(task), 1)
    before = attributes(stage, author.ROOT, physical_only=True)
    preserved = {k: v for k, v in attributes(stage, author.ROOT).items()
                 if 'diffuseColor' in k or k.endswith('inputs:file')}
    visual = author._appearance(stage, 1, tmp_path)
    assert visual['changed_materials']
    assert (tmp_path/'finish_roughness.png').is_file()
    assert visual['dependencies_sha256'][str(tmp_path/'finish_roughness.png')] == author.file_hash(tmp_path/'finish_roughness.png')
    assert attributes(stage, author.ROOT, physical_only=True) == before
    after = attributes(stage, author.ROOT)
    assert {k: after[k] for k in preserved} == preserved


@pytest.fixture
def composed_fixture(tmp_path):
    task_id = 'book_stacking'
    definition = importlib.import_module('scenes.workrooms.study.tasks.'+task_id+'.definition').TASK
    release = Path(definition['source_release']['path'])
    _, inputs = author._parent(release)
    stage, md = author.derive_stage(*parent(task_id), 1)
    stage.GetRootLayer().Export(str(tmp_path/'initial.usda'))
    author._write(tmp_path/'metadata.json', md)
    layer = inputs['scene']['task_layers'][0]
    manifest = {'schema': author.SCHEMA, 'family': 'study', 'task_id': task_id, 'seed': 1,
        'parent_references': {},
        'dependencies_sha256': {str(tmp_path/name): author.file_hash(tmp_path/name) for name in ('initial.usda', 'metadata.json')},
        'initial_usd': str(tmp_path/'initial.usda'), 'initial_usd_sha256': author.file_hash(tmp_path/'initial.usda'),
        'metadata': str(tmp_path/'metadata.json'), 'metadata_sha256': author.file_hash(tmp_path/'metadata.json'),
        'parent_initial_usd_sha256': layer['sha256'],
        'parent_configuration_sha256': inputs['task']['evaluation']['configuration']['configuration_sha256'],
        'plan_sha256': md['seed_plan']['plan_sha256']}
    author._write(tmp_path/'seed_manifest.json', manifest)
    return release, inputs, tmp_path/'seed_manifest.json'


def test_compose_preserves_robot_room_controller_and_no_private_reset_public(composed_fixture):
    release, original, manifest = composed_fixture
    result = author.compose_case(release, manifest, 'single_arm')
    assert result['initial_state'] == original['initial_state']
    for key in ('robot', 'workroom', 'isaacsim', 'physics', 'fixtures', 'cameras'):
        assert result['scene'][key] == original['scene'][key]
    assert result['task']['initial_state'] == original['task']['initial_state']
    assert result['task']['public_goal'] == original['task']['public_goal']
    assert result['task']['public_rules_sha256'] == rules.canonical_hash(result['task']['public_rules'])
    assert 'actual_body_poses_m' not in json.dumps(result['task']['public_rules'])
    assert result['evaluator_sources']
    for name, text in result['evaluator_sources'].items():
        assert text == (release/'evaluator'/name).read_text()
    from bench.isaacsim.release import _frozen_evaluator_module
    evaluator = _frozen_evaluator_module(release).SequenceEvaluator(result['task'])
    assert evaluator is not None


def test_configuration_argument_cannot_relabel_single_robot(composed_fixture):
    release, _, manifest = composed_fixture
    with pytest.raises(ValueError, match='frozen robot installation'):
        author.compose_case(release, manifest, 'dual_opposed')


@pytest.mark.parametrize('cid', ['dual_same_side', 'dual_opposed'])
def test_compose_preserves_real_dual_installations(composed_fixture, cid):
    from scenes.workrooms.robot_configuration_catalog import RELEASES
    release = Path(RELEASES[('study', 'book_stacking')]['configurations'][cid]['source_release']['path'])
    _, original = author._parent(release)
    result = author.compose_case(release, composed_fixture[2], cid)
    assert result['initial_state'] == original['initial_state']
    assert result['scene']['robot'] == original['scene']['robot']
    assert result['scene']['robot_configuration'] == original['scene']['robot_configuration']


def test_compose_fails_closed_if_asset_is_changed(composed_fixture):
    release, _, manifest = composed_fixture
    raw = load_json(manifest)
    Path(raw['initial_usd']).write_text('tampered')
    with pytest.raises(ValueError, match='hash differs'):
        author.compose_case(release, manifest, 'single_arm')


def test_compose_rejects_other_parent_configuration(composed_fixture):
    release, _, manifest = composed_fixture
    raw = load_json(manifest); raw['parent_configuration_sha256'] = '0'*64
    author._write(manifest, raw)
    with pytest.raises(ValueError, match='Parent asset/configuration'):
        author.compose_case(release, manifest, 'single_arm')


@pytest.mark.parametrize('seed', [0, 1])
def test_publication_roundtrip_real_asset_byte_preservation_and_dependencies(seed, monkeypatch, tmp_path):
    # Only the source-envelope verifier is a CPU fixture. All parent USD,
    # metadata, dependency files and composed evaluator sources are real.
    import scripts.maintenance.freeze_project_sources as freeze
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'scope': 'CPU fixture, not runtime qualification'})
    monkeypatch.setattr(author, 'ASSET_ROOT', tmp_path)
    definition = importlib.import_module('scenes.workrooms.study.tasks.stationery_sorting.definition').TASK
    release = Path(definition['source_release']['path'])
    path = author.prepare_asset(release, seed, tmp_path/f'seed_{seed:02}')
    manifest = load_json(path)
    for name, sha in manifest['dependencies_sha256'].items():
        assert author.file_hash(name) == sha
    result = author.compose_case(release, path, 'single_arm')
    from pxr import UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(manifest['initial_usd'])
    assert not unresolved
    dependencies = {str(Path(layer.realPath).resolve()) for layer in layers}
    dependencies.update(str(Path(str(asset)).resolve()) for asset in assets)
    assert dependencies <= set(manifest['dependencies_sha256'])
    if seed == 0:  # dexbench: a relocated copy keeps the parent layer's identity, not its bytes
        parent_usd = next(p for p in manifest['parent_references'] if p.endswith('.usda'))
        assert usd_layer_identity(manifest['initial_usd']) == usd_layer_identity(parent_usd)
        assert result['task']['evaluation']['configuration'] == parent('stationery_sorting')[1]['configuration']
    else:
        assert manifest['initial_usd_sha256'] != manifest['parent_initial_usd_sha256']
        assert (path.parent/'finish_roughness.png').is_file()
    with pytest.raises(FileExistsError):
        author.prepare_asset(release, seed, path.parent)


def test_publication_requires_frozen_source(monkeypatch, tmp_path):
    import scripts.maintenance.freeze_project_sources as freeze
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: None)
    with pytest.raises(ValueError, match='complete frozen source'):
        author.prepare_asset(tmp_path, 1, tmp_path/'output')
    assert not (tmp_path/'output').exists()
