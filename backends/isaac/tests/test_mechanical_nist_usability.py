"""Physical usability adaptations preserve fit, passive topology and old assets."""
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.nist_assets import (
    ASSET_ROOT, SOURCE_ROOT, _r, author_task, build_assets, structural_audit,
)
from scenes.workrooms.mechanical_assembly.nist_usability import PROFILE

TASKS = [('nist_connector_insertion', 'v3'), ('nist_cable_routing', 'v5'), ('nist_wire_harness', 'v4')]
pytestmark = pytest.mark.skipif(not SOURCE_ROOT.is_dir(), reason='NIST source snapshot unavailable')


def authored(task):
    stage = Usd.Stage.CreateInMemory()
    return stage, author_task(stage, task, usability_profile=PROFILE)


@pytest.mark.parametrize('task,version', TASKS)
def test_default_asset_metadata_stays_exactly_historical(task, version):
    expected = load_json(ASSET_ROOT/task/version/'metadata.json')
    assert author_task(Usd.Stage.CreateInMemory(), task) == expected


@pytest.mark.parametrize('task,version', TASKS)
def test_profile_bakes_physics_and_preserves_passive_joints(task, version):
    stage, meta = authored(task)
    old = load_json(ASSET_ROOT/task/version/'metadata.json')
    assert meta['usability_profile'] == PROFILE
    assert meta['fixture_metadata']['workplate'] == old['fixture_metadata']['workplate']
    assert structural_audit(stage, meta)['passed']
    assert meta['joints'] == old['joints']
    prior = {o['id']: o for o in old['objects']}
    for obj in meta['objects']:
        prim = stage.GetPrimAtPath(obj['body_path'])
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert not any(op.GetOpType() == UsdGeom.XformOp.TypeScale for op in UsdGeom.Xformable(prim).GetOrderedXformOps())
        if 'link' not in obj['id']:
            assert np.allclose(obj['half_extents_m'], np.array(prior[obj['id']]['half_extents_m'])*1.6)
            assert obj['mass_kg'] == pytest.approx(prior[obj['id']]['mass_kg']*1.6**3)
            assert np.allclose(obj['diagonal_inertia_kg_m2'], np.array(prior[obj['id']]['diagonal_inertia_kg_m2'])*1.6**5, rtol=1e-8, atol=1e-16)
    for rec in meta['source']['external_assets']:
        original = next(r for r in old['source']['external_assets'] if r['path'] == rec['path'])
        assert rec['sha256'] == original['sha256']
        assert rec['scale_to_m'] == original['scale_to_m'] == .001
    assert not meta['physics_accepted']


def test_audio_plug_can_pass_through_adapted_guide_bore():
    stage, meta = authored('nist_cable_routing')
    obj = next(o for o in meta['objects'] if o['id'] == 'cable_end')
    plug = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(obj['collider_paths'][0])).GetPointsAttr().Get())
    guide = meta['fixture_metadata']['guide_0']
    tube = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(guide['collider_paths'][0])).GetPointsAttr().Get())
    # Local source plug Z is insertion axis, guide Y is bore axis.
    max_radius = np.linalg.norm(plug[:, :2], axis=1).max()
    bore_radius = np.linalg.norm(tube[:, [0, 2]], axis=1).min()
    assert bore_radius-max_radius > .002
    goal = meta['task_parameters']['seating_goals'][0]
    target = meta['diagnostic_goal_poses']['cable_end']
    point = np.array(target[:3])+_r(target[3:])@goal['actor_local_point_m']
    assert np.allclose(point, meta['fixture_metadata']['audio_socket']['initial_position_m'], atol=1e-10)
    assert goal['actor_local_point_m'][2] == pytest.approx(.0864)
    assert meta['task_parameters']['cable']['diameter_m'] == pytest.approx(.00525)
    assert meta['task_parameters']['cable']['length_m'] == .56
    gate = meta['task_parameters']['flexible_groups'][0]['route_gates'][0]
    assert gate['radius_m'] == .01125
    assert gate['radius_m'] < bore_radius
    assert gate['radius_m']-(bore_radius-.002625) < .0007


@pytest.mark.parametrize('reverse', [False, True])
def test_harness_enlarged_endpoints_and_wire_clear_reversible_nests(reverse):
    stage, meta = authored('nist_wire_harness')
    objects = {o['id']: o for o in meta['objects']}
    for color in ['red', 'blue']:
        for side, index in [('a', 0), ('b', 29)]:
            fixture = meta['fixture_metadata'][f'{color}_end_{side}_terminal']
            shell = objects[f'{color}_end_{"b" if reverse else "a"}']
            extent = np.array(shell['half_extents_m'])*2
            cavity = np.array(fixture['half_extents_m'])*2-[.006, .006, 0]
            assert np.all(cavity[:2]-extent[:2] >= .003-1e-10)
            assert fixture['cable_exit']['width_m'] == .009
            link = f'{color}_link_{index:03}'
            pose = meta['diagnostic_goal_poses'][link]
            geometry = objects[link]['analytic_geometry']
            axis = _r(pose[3:])[:, 0]*geometry['cylinder_length_m']/2
            endpoints = np.asarray([pose[:3]-axis, pose[:3]+axis])
            for path in fixture['collider_paths']:
                local = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
                corners = np.asarray([local.Transform(Gf.Vec3d(x, y, z))
                    for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
                corners += fixture['initial_position_m']
                gap = np.maximum(np.maximum(corners.min(0)-endpoints.max(0), endpoints.min(0)-corners.max(0)), 0)
                assert np.linalg.norm(gap)-geometry['radius_m'] > .00029
    assert meta['task_parameters']['harness']['wire_length_m'] == .410


def test_profile_requires_explicit_new_asset_output():
    with pytest.raises(ValueError, match='explicit new immutable'):
        build_assets('nist_connector_insertion', usability_profile=PROFILE)
    with pytest.raises(ValueError, match='Unknown NIST usability profile'):
        author_task(Usd.Stage.CreateInMemory(), 'nist_connector_insertion', usability_profile='typo')


def test_harness_has_one_continuous_open_floor_at_terminal_support_height():
    stage, meta = authored('nist_wire_harness')
    first = meta['fixture_metadata']['bundle_retainer_0']
    second = meta['fixture_metadata']['bundle_retainer_1']
    assert len(first['collider_paths']) == 3
    assert len(second['collider_paths']) == 2
    floor = stage.GetPrimAtPath(first['collider_paths'][0])
    transform = UsdGeom.Xformable(floor).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    corners = np.asarray([transform.Transform(Gf.Vec3d(x,y,z))
        for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    assert np.allclose(corners.min(0), [.39,-.193,.79], atol=1e-8)
    assert np.allclose(corners.max(0), [.48,.193,.793], atol=1e-8)
    for color in ['red','blue']:
        for index in range(1,29):
            p = meta['diagnostic_goal_poses'][f'{color}_link_{index:03}'][:3]
            assert .39 < p[0] < .48 and -.193 < p[1] < .193
            assert p[2]-.00225-.793 == pytest.approx(.0003)
    # All raised walls remain the four original short outside posts; nothing
    # covers either lead or blocks upward assembly/removal.
    assert all('bundle_retainer' not in name or name in {'bundle_retainer_0','bundle_retainer_1'}
               for name in meta['fixtures'])
