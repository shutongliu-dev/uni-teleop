"""Real USD scene checks; no GPU, settling or manipulation claim."""
import json

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from scenes.workrooms.living_room import tower_assets
from scenes.workrooms.living_room.tasks.tower_building.assets import author_task as author_building
from scenes.workrooms.living_room.tasks.tower_extraction.assets import author_task as author_extraction
from scenes.workrooms.living_room.tasks.tower_building.task import validate_configuration


@pytest.fixture(scope='module', params=[('building', False), ('building', True), ('extraction', False)])
def authored(request):
    task, presentation = request.param
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author = author_building if task == 'building' else author_extraction
    metadata = author(stage, presentation=presentation)
    return stage, metadata


def test_independent_bodies_and_exact_collider_mass_contract(authored):
    stage, metadata = authored
    objects = metadata['objects']
    assert len(objects) == len(metadata['dynamic_prim_paths']) == 18
    assert {p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)} == set(metadata['dynamic_prim_paths'])
    for item in objects:
        prim = stage.GetPrimAtPath(item['prim_path'])
        assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        mass = UsdPhysics.MassAPI(prim)
        assert mass.GetMassAttr().Get() == pytest.approx(.1096875)
        assert list(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(item['inertia_diagonal_kg_m2'])
        assert item['mass_kg'] == pytest.approx(np.prod(item['size_m']) * 650.)
        collider = stage.GetPrimAtPath(item['collider_paths'][0])
        assert collider.IsA(UsdGeom.Cube)
        assert list(collider.GetAttribute('xformOp:scale').Get()) == pytest.approx(item['size_m'])
        assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
        assert collider.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0001)
        bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        physics = UsdPhysics.MaterialAPI(bound.GetPrim())
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(.45)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(.35)
    assert not any(prim.IsA(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse())


def test_full_inventory_has_no_positive_initial_overlap_and_is_table_bounded(authored):
    stage, metadata = authored
    boxes = [np.asarray(item['initial_bounds_m']) for item in metadata['objects']]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for item, box in zip(metadata['objects'], boxes, strict=True):
        assert box[0][2] >= .78 - 1e-12
        assert np.all(box[0][:2] >= [.15, -.5])
        assert np.all(box[1][:2] <= [1.15, .5])
        actual = cache.ComputeWorldBound(stage.GetPrimAtPath(item['collider_paths'][0])).ComputeAlignedRange()
        np.testing.assert_allclose([actual.GetMin(), actual.GetMax()], box, atol=5e-8)
        # Rounded wood display is inside the nominal physics box. Tiny ink marks
        # are separate and are not support geometry.
        visual = cache.ComputeWorldBound(stage.GetPrimAtPath(item['prim_path'] + '/Visual')).ComputeAlignedRange()
        assert np.all(np.asarray(visual.GetMin()) >= box[0] - 5e-8)
        assert np.all(np.asarray(visual.GetMax()) <= box[1] + 5e-8)
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
            assert not np.all(overlap > 1e-9), overlap


def test_markers_cannot_supply_support_or_hide_dynamics(authored):
    stage, metadata = authored
    markers = stage.GetPrimAtPath('/World/LivingTask/Markers')
    for prim in Usd.PrimRange(markers):
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            assert np.ptp(points[:, 2]) == 0
            assert points[0, 2] == pytest.approx(.78006)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]) == 18


def test_factory_reset_and_presentation_are_not_conflated():
    reset_stage, display_stage = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    reset = author_building(reset_stage)
    display = author_building(display_stage, presentation=True)
    validate_configuration(reset['configuration'])
    assert reset['configuration'] == display['configuration']
    assert reset['is_task_reset'] and not reset['presentation_only']
    assert not display['is_task_reset'] and display['presentation_only']
    assert len(reset['layout']['authored_supply_block_ids']) == 18
    assert len(display['layout']['authored_supply_block_ids']) == 9
    assert len(display['layout']['authored_tower_block_ids']) == 9
    assert max(item['initial_bounds_m'][1][2] for item in display['objects']) == pytest.approx(.855)
    for item in reset['objects']:
        for field in ('position_m', 'quaternion_wxyz'):
            assert item[field] == reset['configuration']['initial_objects'][item['block_id']][field]
    assert not display['success_claimed'] and not display['native_qualified']


def test_extraction_reset_and_symmetric_collection_slots():
    stage = Usd.Stage.CreateInMemory()
    metadata = author_extraction(stage)
    assert metadata['is_task_reset'] and not metadata['presentation_only']
    assert len(metadata['layout']['authored_tower_block_ids']) == 18
    assert not metadata['layout']['authored_supply_block_ids']
    assert max(item['initial_bounds_m'][1][2] for item in metadata['objects']) == pytest.approx(.93)
    slots = metadata['layout']['collection_slots']
    assert len(slots['front']) == len(slots['rear']) == 6
    assert len(metadata['markers']) == 12
    for front, rear in zip(slots['front'], slots['rear'], strict=True):
        assert front['position_m'][0] + rear['position_m'][0] == pytest.approx(1.3)
        assert front['position_m'][1] == pytest.approx(-rear['position_m'][1])
        assert front['position_m'][2] == rear['position_m'][2] == pytest.approx(.7925)


def test_json_provenance_and_resolved_dependencies(authored, tmp_path):
    stage, metadata = authored
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata['material_source']['license'] == 'CC0-1.0'
    assert metadata['physics_prior']['native_accepted'] is False
    for dependency in metadata['dependencies']:
        assert tower_assets._dependency(dependency['path']) == dependency
    saved = tmp_path / 'candidate.usda'
    stage.GetRootLayer().Export(str(saved))
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(saved))
    assert not unresolved


def test_custom_root_and_existing_root_do_not_overwrite_another_task():
    stage = Usd.Stage.CreateInMemory()
    metadata = author_extraction(stage, '/World/OtherTask')
    assert all(path.startswith('/World/OtherTask/Blocks/') for path in metadata['dynamic_prim_paths'])
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='already exists'):
        author_extraction(stage, '/World/OtherTask')
    assert stage.GetRootLayer().ExportToString() == before
    with pytest.raises(ValueError, match='absolute'):
        author_building(stage, 'RelativeTask')


@pytest.fixture(scope='module', params=[('building', False), ('building', True), ('extraction', False)],
                ids=['building_reset', 'building_presentation', 'extraction_reset'])
def tray_authored(request):
    task, presentation = request.param
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author = author_building if task == 'building' else author_extraction
    return stage, author(stage, presentation=presentation, profile='trays_v1')


def test_tray_profiles_keep_independent_blocks_and_physical_fixed_trays(tray_authored):
    stage, metadata = tray_authored
    assert len(metadata['objects']) == len(metadata['dynamic_prim_paths']) == 36
    assert len(metadata['trays']) == 2
    assert len(metadata['static_collision_paths']) == len(metadata['contact_actor_paths']) == 10
    assert {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)} == set(metadata['dynamic_prim_paths'])
    assert {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)} == (
        set(metadata['static_collision_paths']) | {p for row in metadata['objects'] for p in row['collider_paths']})
    assert not any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for tray in metadata['trays']:
        assert tray['bottom_z_m'] == pytest.approx(.78)
        assert tray['floor_z_m'] == pytest.approx(.786)
        assert tray['rim_top_z_m'] == pytest.approx(.798)
        assert tray['construction_mass_kg'] > 0
        assert tray['physics_prior']['native_accepted'] is False
        for part in tray['components']:
            prim = stage.GetPrimAtPath(part['prim_path'])
            assert prim.IsA(UsdGeom.Cube) and prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            assert not prim.HasAPI(UsdPhysics.MassAPI)
            assert part['construction_mass_kg'] == pytest.approx(np.prod(part['size_m']) * 650.)
            assert metadata['contact_actor_paths'][part['actor_id']] == part['prim_path']
            actual = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            np.testing.assert_allclose([actual.GetMin(), actual.GetMax()], part['bounds_m'], atol=1e-10)
            bounds = np.asarray(part['bounds_m'])
            assert np.all(bounds[0] >= [.15, -.5, .78 - 1e-12])
            assert np.all(bounds[1][:2] <= [1.15, .5])
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            assert UsdPhysics.MaterialAPI(bound.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(.45)
            # Real walls and floors cannot pass through the supplied/tower blocks.
            for block in metadata['objects']:
                box = np.asarray(block['initial_bounds_m'])
                overlap = np.minimum(bounds[1], box[1]) - np.maximum(bounds[0], box[0])
                assert not np.all(overlap > 1e-9), (part['prim_path'], block['block_id'], overlap)
    # Every body pair is disjoint or touching; two stacked layers are real support.
    boxes = [np.asarray(row['initial_bounds_m']) for row in metadata['objects']]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-9)


def test_new_building_reset_has_two_full_trays_and_no_prescribed_tower():
    metadata = author_building(Usd.Stage.CreateInMemory(), profile='trays_v1')
    assert metadata['configuration']['scoring_mode'] == 'freeform_height'
    assert metadata['configuration']['target_slots'] == []
    assert len(metadata['layout']['authored_supply_block_ids']) == 36
    assert not metadata['layout']['authored_tower_block_ids']
    assert metadata['markers'] == []
    for tray in metadata['trays']:
        low, high = np.asarray(tray['interior_bounds_xy_m'])
        contained = [row for row in metadata['objects']
                     if np.all(np.asarray(row['initial_bounds_m'])[0, :2] >= low)
                     and np.all(np.asarray(row['initial_bounds_m'])[1, :2] <= high)]
        assert len(contained) == 18
        assert sorted({round(row['position_m'][2], 8) for row in contained}) == [.7985, .8235]
        lower = [row for row in contained if row['position_m'][2] < .81]
        upper = [row for row in contained if row['position_m'][2] > .81]
        assert len(lower) == len(upper) == 9
        for block in upper:
            assert any(np.allclose(block['position_m'][:2], support['position_m'][:2])
                       and block['initial_bounds_m'][0][2] == pytest.approx(support['initial_bounds_m'][1][2])
                       for support in lower)


def test_new_extraction_tower_stays_on_table_with_access_and_full_collection_capacity():
    metadata = author_extraction(Usd.Stage.CreateInMemory(), profile='trays_v1')
    assert metadata['geometry']['block_size_m'] == [.20, .045, .030]
    assert metadata['geometry']['layers'] == 12
    assert metadata['geometry']['in_layer_gap_m'] == .010
    assert len(metadata['layout']['authored_tower_block_ids']) == 36
    assert min(row['initial_bounds_m'][0][2] for row in metadata['objects']) == pytest.approx(.78)
    assert max(row['initial_bounds_m'][1][2] for row in metadata['objects']) == pytest.approx(1.14)
    assert len(metadata['markers']) == 30
    for side, slots in metadata['layout']['collection_slots'].items():
        assert len(slots) == 15
        assert all(row['position_m'][2] == pytest.approx(.801) for row in slots)
    for tray in metadata['trays']:
        # Lowest legal layer is above the rim, keeping horizontal extraction clear.
        assert tray['rim_top_z_m'] < .78 + .030
        nearest_edge = min(abs(row[1]) for row in tray['outer_bounds_xy_m'])
        assert nearest_edge - .10 >= .064 - 1e-12
    assert all(row['z_m'] == pytest.approx(.78606) for row in metadata['markers'])
    assert metadata['geometric_clearance_audit']['block_end_overhang_beyond_crossing_layer_m'] == pytest.approx(.0225)
    assert metadata['geometric_clearance_audit']['legacy_block_end_overhang_beyond_crossing_layer_m'] == pytest.approx(.0065)


def test_unknown_profile_rejected_before_authoring():
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='profile'):
        author_extraction(stage, profile='unknown')
    assert stage.GetRootLayer().ExportToString() == before
