"""Actual frozen kitchen geometry remains unchanged under X-only resets."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from bench.paths import load_json
from scenes.workrooms.kitchen import placement_variants as v, placement_authoring as a
from scenes.workrooms.seed_geometry import file_hash, write_json


@pytest.fixture(scope='module')
def parents():
    return {name: Path(importlib.import_module('scenes.workrooms.kitchen.tasks.'+name+'.definition').TASK['source_release']['path'])
            for name in v.TASK_IDS}


@pytest.fixture(scope='module')
def originals(parents):
    return {name: v._parent(path) for name, path in parents.items()}


@pytest.fixture(scope='module')
def authored(tmp_path_factory, parents):
    root = tmp_path_factory.mktemp('kitchen_world_x')
    actual = Path(v.__file__).resolve().parents[3]
    source = root/'explicit_test_only_source'
    for name in v.SOURCE_FILES:
        (source/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual/name, source/name)
    hashes = {name: file_hash(source/name) for name in v.SOURCE_FILES}
    snapshot = source/'source_snapshot_manifest.json'
    write_json(snapshot, {'schema': 'complete_benchmark_source_snapshot_v1',
                         'snapshot_root': str(source), 'files_sha256': hashes})
    proof = {'snapshot_root': str(source), 'manifest_path': str(snapshot),
             'manifest_sha256': file_hash(snapshot),
             'source_files_sha256': {str(source/n): h for n,h in hashes.items()}}
    result = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(a, '_author_source_identity', lambda: deepcopy(proof))
        for name, parent in parents.items():
            for seed in (0, 1, 8):
                result[name, seed] = v.prepare_asset(parent, seed, root/name/str(seed))
    return result


@pytest.mark.parametrize('task_id', v.TASK_IDS)
def test_ten_distinct_conservative_world_x_cases(originals, task_id):
    scene, task, _, _, _ = originals[task_id]
    plans = [v.plan_case(scene, task, seed) for seed in range(10)]
    assert len({tuple(p['translation_xy_m']) for p in plans}) == 10
    assert plans[0]['translation_xy_m'] == [0., 0.]
    assert all(p['allowed_axis'] == 'world_x' and p['translation_xy_m'][1] == 0.
               and abs(p['translation_xy_m'][0]) <= .01 for p in plans)
    assert all(p['groups'] == plans[0]['groups'] for p in plans)
    assert plans[0]['groups'][0]['actor_ids'] == sorted(task['asset_metadata']['actors'])


@pytest.mark.parametrize('bad', [True, -1, 10, 1.0, None])
def test_invalid_seed_rejected(originals, bad):
    scene, task, *_ = originals['surface_wiping']
    with pytest.raises(ValueError):
        v.plan_case(scene, task, bad)


@pytest.mark.parametrize('task_id', v.TASK_IDS)
def test_real_roundtrip_seed_zero_and_both_range_extremes(parents, originals, authored, task_id):
    from scenes.workrooms.initial_randomization.release import _compose
    from bench.isaacsim.release import load_evaluator
    scene, task, initial, candidate, references = originals[task_id]
    evaluator = load_evaluator(parents[task_id])
    zero = v.compose_case(parents[task_id], authored[task_id, 0], 'single_arm')
    assert (zero['scene'], zero['task'], zero['initial_state']) == (scene, task, initial)
    assert not any(p.suffix in ('.usd', '.usda', '.png') for p in authored[task_id, 0].parent.iterdir())
    for seed in (0, 1, 8):
        child, child_task, joints, case, sources, deps = _compose(
            parents[task_id], authored[task_id, seed], task_id, seed, 'single_arm')
        assert joints == initial
        for key in ('robot', 'physics', 'isaacsim', 'robot_configuration', 'workroom', 'cameras', 'controller', 'fixtures'):
            assert child.get(key) == scene.get(key)
        assert all(text == (parents[task_id]/'evaluator'/name).read_text() for name,text in sources.items())
        evaluator(child_task)
        assert child_task['evaluation']['thresholds'] == task['evaluation']['thresholds']
        assert child['native_task_bundle']['joints'] == scene['native_task_bundle']['joints']
        assert child['native_task_bundle']['observer']['configuration'] == child_task
        assert {k: deps[k] for k in references} == references
        assert all(file_hash(path) == sha for path,sha in deps.items())
        assert case['runtime_object_writes'] is False
        assert case['static_geometry']['bodies'] == zero['static_geometry']['bodies']
        assert case['fixed_support_paths'] == sorted(
            p for row in task['asset_metadata']['fixture_metadata'].values() for p in row['collider_paths'])
        assert all(p.startswith('/World/KitchenTask/Fixtures/') for p in case['fixed_support_paths'])


@pytest.mark.parametrize('task_id', v.TASK_IDS)
def test_actual_usd_only_world_x_and_render_fields_change(originals, authored, task_id):
    scene, task, _, candidate, _ = originals[task_id]
    original = Usd.Stage.Open(candidate['usd_path'])
    md = task['asset_metadata']
    moving = set(md['actors'].values()) | set(md['fixtures'].values())
    before = a.protected_properties(original, moving)
    for seed in (1, 8):
        manifest = load_json(authored[task_id, seed])
        data = load_json(Path(manifest['metadata']))
        stage = Usd.Stage.Open(data['bundle_usd']['path'])
        assert a.protected_properties(stage, moving) == before
        # The small exported layer references the original geometry instead of
        # copying vertices, masses, colliders, joints or contact material.
        text = Path(data['bundle_usd']['path']).read_text()
        assert candidate['usd_path'] in text
        assert 'point3f[] points' not in text and 'physics:mass' not in text
        assert 'physics:localPos' not in text and 'PhysicsJoint' not in text
        for path in moving:
            old = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(original.GetPrimAtPath(path)))
            new = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
            expected = old.copy(); expected[3, 0] += v.OFFSETS_X_M[seed]
            np.testing.assert_allclose(new, expected, atol=1e-14, rtol=0)
        assert data['appearance'] and all(r['only_albedo_changed'] for r in data['appearance'])
        changed = data['appearance'][0]['visual_mesh_paths']
        assert all('/Actors/' in path and '/Visual/' in path for path in changed)
        assert not any('Markings' in path or 'Context' in path for path in changed)


def test_source_goal_and_sponge_internal_references_coherent(originals):
    for task_id, (scene, task, *_) in originals.items():
        old = task['asset_metadata']
        for seed in range(10):
            md = v._metadata(old, v.plan_case(scene, task, seed))
            dx = v.OFFSETS_X_M[seed]
            if task_id == 'breakfast_serving':
                assert md['goals']['body_poses_xyzw'] == old['goals']['body_poses_xyzw']
                assert md['goals']['destination_region_xy_m'] == old['goals']['destination_region_xy_m']
                assert md['goals']['source_center_xy_m'][0] == old['goals']['source_center_xy_m'][0]+dx
                assert md['goals']['nominal_transport_distance_m'] == pytest.approx(.72-dx)
            elif task_id == 'pot_lid_placement':
                assert md['goals'] == old['goals']
                assert md['objects'][0]['initial_pose_xyzw'][3:] == old['objects'][0]['initial_pose_xyzw'][3:]
            else:
                assert md['goals']['soil_cells'] == old['goals']['soil_cells']
                assert md['goals']['soil_patches'] == old['goals']['soil_patches']
                assert md['joints'] == old['joints']
                assert md['composite_objects'] == old['composite_objects']
                old_c, new_c = old['elastic_composites']['sponge'], md['elastic_composites']['sponge']
                assert {k:v for k,v in old_c.items() if k != 'cells'} == {k:v for k,v in new_c.items() if k != 'cells'}
                by_id = {r['id']: r for r in md['objects']}
                for cell in new_c['cells']:
                    assert cell == by_id[cell['id']]
                for name, p in md['goals']['body_poses_xyzw'].items():
                    assert p == by_id[name]['initial_pose_xyzw']


@pytest.mark.parametrize('task_id', v.TASK_IDS)
def test_semantic_pickups_are_actual_collider_vertices(originals, authored, task_id):
    _, task, _, candidate, _ = originals[task_id]
    stage = Usd.Stage.Open(candidate['usd_path'])
    record = load_json(authored[task_id, 1])
    targets = record['interaction_targets']
    for target in targets:
        if target['type'] != 'actual_surface':
            continue
        witness = target['surface_witness']
        prim = stage.GetPrimAtPath(witness['collider_path'])
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        source = UsdGeom.Mesh(prim).GetPointsAttr().Get()[witness['vertex_index']]
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetPrimAtPath(target['prim_path']))
        from pxr import Gf
        point = matrix.Transform(Gf.Vec3d(*source))
        np.testing.assert_allclose(target['local_position_m'], point, atol=1e-14, rtol=0)
    groups = {t['operation_group'] for t in targets if t['type'] == 'actual_surface'}
    assert groups == ({'sponge_pickup'} if task_id == 'surface_wiping'
                      else {name+'_pickup' for name in task['asset_metadata']['actors']})


def test_protected_property_guard_rejects_physics_y_rotation_and_mesh_changes(originals):
    _, task, _, candidate, _ = originals['surface_wiping']
    original = Usd.Stage.Open(candidate['usd_path']); stage = Usd.Stage.Open(original.Flatten())
    moved = set(task['asset_metadata']['actors'].values()) | set(task['asset_metadata']['fixtures'].values())
    before = a.protected_properties(stage, moved)
    body = stage.GetPrimAtPath(task['asset_metadata']['actors']['sponge'])
    attr = body.GetAttribute('xformOp:translate'); old = attr.Get()
    attr.Set(type(old)(old[0], old[1]+.001, old[2]))
    assert a.protected_properties(stage, moved) != before
    attr.Set(old)
    mass = UsdPhysics.MassAPI(body).GetMassAttr(); old_mass = mass.Get(); mass.Set(old_mass*2)
    assert a.protected_properties(stage, moved) != before
    mass.Set(old_mass)
    joint = stage.GetPrimAtPath(next(iter(task['asset_metadata']['joints'].values())))
    anchor = UsdPhysics.Joint(joint).GetLocalPos0Attr(); old_anchor = anchor.Get()
    anchor.Set(type(old_anchor)(old_anchor[0]+.001, old_anchor[1], old_anchor[2]))
    assert a.protected_properties(stage, moved) != before


def test_rehashed_bad_reset_rejected_and_wrong_configuration(parents, authored, tmp_path):
    parent = parents['pot_lid_placement']; source = authored['pot_lid_placement', 1]
    with pytest.raises(ValueError, match='single_arm'):
        v.compose_case(parent, source, 'dual_same_side')
    manifest = load_json(source); data = load_json(Path(manifest['metadata']))
    data['actual_body_poses_m']['pot']['position_m'][1] += .01
    metadata = tmp_path/'metadata.json'; write_json(metadata, data)
    manifest.update(metadata=str(metadata), metadata_sha256=file_hash(metadata), actual_body_poses_m=data['actual_body_poses_m'])
    manifest['dependencies_sha256'][str(metadata)] = file_hash(metadata)
    path = tmp_path/'bad.json'; write_json(path, manifest)
    with pytest.raises(ValueError, match='world-X-only'):
        v.compose_case(parent, path, 'single_arm')


def test_source_guard_before_creating_output(parents, tmp_path):
    path = tmp_path/'not_published'
    with pytest.raises(ValueError, match='complete frozen source'):
        v.prepare_asset(parents['breakfast_serving'], 1, path)
    assert not path.exists()


def test_plain_import_does_not_load_usd():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import scenes.workrooms.kitchen.placement_variants; assert not any(x=="pxr" or x.startswith("pxr.") for x in sys.modules)'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wiping_visible_runtime_skin_uses_changed_albedo_with_original_rest_shape(originals, authored):
    from bench.simulation.isaacsim_backend.elastic_visual_skin import ElasticVisualSkin
    from pxr import UsdShade
    _, task, _, candidate, _ = originals['surface_wiping']
    original = Usd.Stage.Open(candidate['usd_path'])
    manifest = load_json(authored['surface_wiping', 1])
    data = load_json(Path(manifest['metadata']))
    stage = Usd.Stage.Open(data['bundle_usd']['path'])
    expected = data['asset_metadata']['elastic_composites']
    # This is the existing render-only consumer; construction validates the
    # actual rest-point SHA and copies materials/colour primvars, with no sim.
    skin = ElasticVisualSkin(stage, expected)
    assert len(skin.bindings) == 1
    binding = skin.bindings[0]
    source = binding['source']
    old = original.GetPrimAtPath(source.GetPath())
    np.testing.assert_array_equal(UsdGeom.Mesh(source).GetPointsAttr().Get(), UsdGeom.Mesh(old).GetPointsAttr().Get())
    assert UsdGeom.Imageable(source).GetVisibilityAttr().Get() == 'invisible'
    colors = np.asarray(UsdGeom.PrimvarsAPI(source).GetPrimvar('displayColor').Get())
    before = np.asarray(UsdGeom.PrimvarsAPI(old).GetPrimvar('displayColor').Get())
    assert not np.array_equal(colors, before)
    assert np.all(colors <= before+1e-7) and np.all(colors >= before*.87)
    np.testing.assert_array_equal(UsdGeom.PrimvarsAPI(binding['mesh']).GetPrimvar('displayColor').Get(), colors)
    for suffix in ('finish_porous_foam', 'finish_green_scouring_layer'):
        path = '/World/KitchenTask/Actors/sponge/Visual/'+suffix
        new_material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0]
        old_material = UsdShade.MaterialBindingAPI(original.GetPrimAtPath(path)).ComputeBoundMaterial()[0]
        assert new_material.GetPath() == old_material.GetPath()
        new_shader, old_shader = new_material.ComputeSurfaceSource()[0], old_material.ComputeSurfaceSource()[0]
        assert [(a.GetName(), str(a.Get()), list(map(str, a.GetConnections()))) for a in new_shader.GetPrim().GetAttributes()] == [
            (a.GetName(), str(a.Get()), list(map(str, a.GetConnections()))) for a in old_shader.GetPrim().GetAttributes()]
