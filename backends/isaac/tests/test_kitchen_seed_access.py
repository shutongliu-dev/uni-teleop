"""Additional access references never author physical or scoring changes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.kitchen import seed_variants as subject


def published_case(task_id, seed, source_release=None):
    reference = source_release or task_definition('kitchen', task_id, robot_profile='original')['source_release']
    root = Path(reference['path'])
    assert subject.digest(root/'release_manifest.json') == reference['manifest_sha256']
    manifest = load_json(root/'release_manifest.json')
    assert manifest['schema'] == 'workroom_randomized_release.v1'
    record = manifest['cases'][str(seed)]
    if source_release is None:
        # The L-mount profile is a robot-only revision of this exact release
        # and must reuse the same physical seed case.
        revised = task_definition('kitchen', task_id, robot_profile='l_mount')['source_release']
        revised_root = Path(revised['path'])
        assert subject.digest(revised_root/'release_manifest.json') == revised['manifest_sha256']
        revision = load_json(revised_root/'release_manifest.json')
        assert revision['schema'] == 'workroom_robot_revision_release.v1'
        assert revision['base_release'] == {**reference, 'schema': manifest['schema']}
        assert ({key: revision['cases'][str(seed)][key] for key in ('asset_manifest', 'asset_manifest_sha256')}
                == {key: record[key] for key in ('asset_manifest', 'asset_manifest_sha256')})
    path = Path(record['asset_manifest'])
    assert subject.digest(path) == record['asset_manifest_sha256']
    seed_manifest = load_json(path)
    instance = None
    if seed_manifest.get('generator_version') == 'parameterized_physics_v1':
        # A parameterized physics instance wraps its canonical access case;
        # it may move bodies and change masses but not body-local references.
        instance = seed_manifest
        canonical = Path(instance['canonical_asset']['path'])
        assert subject.digest(canonical) == instance['canonical_asset']['sha256']
        seed_manifest = load_json(canonical)
        assert seed_manifest['generator_version'] == 'kitchen_seed_geometry_v1'
    native_path = Path(seed_manifest['asset_manifest'])
    assert subject.digest(native_path) == seed_manifest['asset_manifest_sha256']
    native = load_json(native_path)
    usd, metadata = native_path.parent/'task.usda', native_path.parent/'metadata.json'
    assert subject.digest(usd) == native['usd_sha256']
    assert subject.digest(metadata) == native['metadata_sha256']
    geometry_path = Path(seed_manifest['case_geometry'])
    assert subject.digest(geometry_path) == seed_manifest['case_geometry_sha256']
    geometry = load_json(geometry_path)
    if instance is not None:
        assert instance['interaction_targets'] == geometry['interaction_targets']
    return Usd.Stage.Open(str(usd)), load_json(metadata), geometry


def table_access02_reference():
    # Preserve and compare the prior access publication, even after the catalog
    # selects a newer one. This is test input, never runtime admission evidence.
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/'
                'workroom_initial_randomization_v1/access02')
    preparation = load_json(root/'preparation.json')
    assert preparation['complete'] is True
    assert subject.digest(root/'catalog_rows.json') == preparation['catalog_rows_sha256']
    catalog = load_json(root/'catalog_rows.json')
    return catalog['table_clearing']['releases']['single_arm']


def original_features(metadata, geometry):
    # A later access publication may already include the new alternatives.
    # Still require every original pickup/destination reference unchanged.
    expected = {name+suffix for name in metadata['actors']
                for suffix in ('_upper_surface', '_destination_surface')}
    result = [row for row in geometry['interaction_targets'] if row['label'] in expected]
    assert {row['label'] for row in result} == expected and len(result) == len(expected)
    return result


@pytest.mark.parametrize('seed', range(10))
def test_table_alternatives_preserve_original_features_usd_mass_and_goal(seed):
    stage, metadata, original = published_case('table_clearing', seed, table_access02_reference())
    usd_before, metadata_before = stage.GetRootLayer().ExportToString(), deepcopy(metadata)
    poses, targets, static = subject._projections(stage, metadata)
    assert stage.GetRootLayer().ExportToString() == usd_before
    assert metadata == metadata_before
    assert poses == original['actual_body_poses_m']
    assert static == original['static_geometry']
    previous_labels = {r['label'] for r in original['interaction_targets']}
    assert [r for r in targets if r['label'] in previous_labels] == original['interaction_targets']
    assert len({r['label'] for r in targets}) == len(targets)
    assert {r['label'] for r in targets}-previous_labels == {'side_plate_rim_negative_y_upper_surface'}
    retained = original_features(metadata, original)
    old_labels = {r['label'] for r in retained}
    assert [r for r in targets if r['label'] in old_labels] == retained
    added = {r['label']: r for r in targets if r['label'] not in old_labels}
    assert set(added) == {
        'spoon_grip_neck_upper_surface', 'fork_grip_neck_upper_surface',
        'fork_destination_diagonal_approach', 'side_plate_rim_negative_y_upper_surface'}
    by_label = {r['label']: r for r in targets}
    objects = {r['id']: r for r in metadata['objects']}
    cache = UsdGeom.XformCache()
    for actor in ('spoon', 'fork'):
        alternative = added[actor+'_grip_neck_upper_surface']
        baseline = by_label[actor+'_upper_surface']
        assert {k: v for k, v in alternative.items() if k not in ('label', 'local_position_m')} == {
            k: v for k, v in baseline.items() if k not in ('label', 'local_position_m')}
        body = stage.GetPrimAtPath(objects[actor]['body_path'])
        prim = stage.GetPrimAtPath(objects[actor]['handle_grip']['collider_path'])
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()).T
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        points = points@matrix[:3, :3].T+matrix[:3, 3]
        top = points[abs(points[:, 2]-points[:, 2].max()) < 1e-6]
        point = np.asarray(alternative['local_position_m'])
        assert np.linalg.norm(points-point, axis=1).min() < 1e-12
        # The independent world/inverse-frame route accumulates sub-femtometer
        # rounding relative to USD's direct relative-transform calculation.
        np.testing.assert_allclose(point, top[np.lexsort((top[:, 0], top[:, 1]))[0]],
                                   rtol=0, atol=1e-14)
        assert alternative['local_position_m'] != baseline['local_position_m']
    goal = added['fork_destination_diagonal_approach']
    baseline = by_label['fork_destination_surface']
    assert {k: v for k, v in goal.items() if k not in ('label', 'approach_direction_local')} == {
        k: v for k, v in baseline.items() if k not in ('label', 'approach_direction_local')}
    np.testing.assert_array_equal(goal['approach_direction_local'], [-1/np.sqrt(2), 0., 1/np.sqrt(2)])
    # Reconstruct all enabled physical mesh vertices by an independent
    # world/inverse-body transform. The point must be an actual upper-rim
    # vertex, not a body center, AABB corner or invented intermediate point.
    row = objects['side_plate']
    body = stage.GetPrimAtPath(row['body_path'])
    collider_points = []
    for prim in Usd.PrimRange(body):
        if (not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()).T
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        collider_points.append(points@matrix[:3, :3].T+matrix[:3, 3])
    points = np.concatenate(collider_points)
    top = points[abs(points[:, 2]-points[:, 2].max()) < 1e-6]
    rim = added['side_plate_rim_negative_y_upper_surface']
    point = np.asarray(rim['local_position_m'])
    assert np.linalg.norm(points-point, axis=1).min() < 1e-12
    np.testing.assert_allclose(point, top[np.lexsort((top[:, 0], top[:, 1]))[0]], rtol=0, atol=1e-14)
    baseline = by_label['side_plate_upper_surface']
    assert {k: v for k, v in rim.items() if k not in ('label', 'local_position_m', 'approach')} == {
        k: v for k, v in baseline.items() if k not in ('label', 'local_position_m', 'approach')}
    assert rim['local_position_m'] != baseline['local_position_m']
    assert rim['approach'] == 'above'


@pytest.mark.parametrize('task_id', ('produce_transfer', 'utensil_sorting'))
def test_other_kitchen_projections_remain_identical(task_id):
    stage, metadata, original = published_case(task_id, 1)
    before = stage.GetRootLayer().ExportToString()
    poses, targets, static = subject._projections(stage, metadata)
    assert stage.GetRootLayer().ExportToString() == before
    assert poses == original['actual_body_poses_m']
    assert targets == original['interaction_targets']
    assert static == original['static_geometry']


@pytest.mark.parametrize('seed', range(10))
def test_pantry_alternatives_are_actual_single_convex_surface_exits_without_physical_changes(seed):
    from scipy.spatial import ConvexHull

    stage, metadata, original = published_case('pantry_stocking', seed)
    before, metadata_before = stage.GetRootLayer().ExportToString(), deepcopy(metadata)
    poses, targets, static = subject._projections(stage, metadata)
    assert stage.GetRootLayer().ExportToString() == before and metadata == metadata_before
    assert poses == original['actual_body_poses_m'] and static == original['static_geometry']
    retained = original_features(metadata, original)
    labels = {r['label'] for r in retained}
    assert [r for r in targets if r['label'] in labels] == retained
    added = {r['label']: r for r in targets if r['label'] not in labels}
    expected = (
        ('pepper_side_mid_surface', .48, [-1., 0., 0.], .08),
        ('pepper_side_high_surface', .85, [-1., 0., 0.], .08),
        ('pepper_diagonal_mid_surface', .48, [-1., 0., 1.], .10),
        ('pepper_diagonal_high_surface', .85, [0., -1., 1.], .08),
    )
    assert set(added) == {r[0] for r in expected}
    row = next(r for r in metadata['objects'] if r['id'] == 'pepper')
    assert len(row['collider_paths']) == 1
    body, collider = [stage.GetPrimAtPath(p) for p in (row['body_path'], row['collider_paths'][0])]
    assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == 'convexHull'
    cache = UsdGeom.XformCache()
    matrix = np.asarray(cache.GetLocalToWorldTransform(collider)*cache.GetLocalToWorldTransform(body).GetInverse()).T
    points = np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get(), dtype=float)
    points = points@matrix[:3, :3].T+matrix[:3, 3]
    normals, offsets = np.split(ConvexHull(points).equations, [3], axis=1)
    offsets = offsets[:, 0]
    baseline = next(r for r in targets if r['label'] == 'pepper_upper_surface')
    for label, fraction, direction, distance in expected:
        feature = added[label]
        for key in ('prim_path', 'type', 'reference_state', 'operation_group', 'required'):
            assert feature[key] == baseline[key]
        assert feature['approach'] == 'above' and feature['approach_distance_m'] == distance
        direction = np.asarray(direction)/np.linalg.norm(direction)
        np.testing.assert_array_equal(feature['approach_direction_local'], direction)
        center = np.array([0., 0., points[:, 2].min()+fraction*np.ptp(points[:, 2])])
        point = np.asarray(feature['local_position_m'])
        assert float((normals@center+offsets).max()) < 1e-8
        assert abs(float((normals@point+offsets).max())) < 1e-9
        travel = float((point-center)@direction)
        assert travel > 0
        np.testing.assert_allclose(point, center+travel*direction, rtol=0, atol=1e-14)
        assert float((normals@(point+direction*1e-6)+offsets).max()) > 0


def test_pantry_surface_reference_rejects_nonconvex_collision_representation():
    stage, metadata, _ = published_case('pantry_stocking', 0)
    candidate = Usd.Stage.Open(stage.Flatten())
    row = next(r for r in metadata['objects'] if r['id'] == 'pepper')
    prim = candidate.GetPrimAtPath(row['collider_paths'][0])
    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set('none')
    with pytest.raises(ValueError, match='original single convexHull mesh'):
        subject._projections(candidate, metadata)
