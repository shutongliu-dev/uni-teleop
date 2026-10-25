"""Diagnostic starts may change poses, never the task physics they qualify."""
from copy import deepcopy
import json

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.assets import TASK_IDS, author_task
from scripts.validation.kitchen_asset_identity import diagnostic_physics_identity


def _pair(tmp_path, task_id='breakfast_serving'):
    stages, metadata, paths = [], [], []
    for name, presentation in (('canonical', False), ('goal', True)):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        meta = author_task(stage, task_id, presentation=presentation)
        path = tmp_path/(name+'.usda')
        stage.GetRootLayer().Export(str(path))
        stages.append(stage)
        metadata.append(meta)
        paths.append(path)
    return stages, metadata, paths


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_only_goal_reset_poses_may_differ(tmp_path, task_id):
    _, metadata, paths = _pair(tmp_path, task_id)
    result = diagnostic_physics_identity(*paths, *metadata)
    assert result['passed'], json.dumps(result, indent=2)
    assert result['dynamic_actor_count'] == len(metadata[0]['actors'])


@pytest.mark.parametrize('mutation', ['mass', 'mesh', 'material', 'relationship', 'extra_prim',
                                      'wrong_pose', 'animated_pose', 'body_metadata'])
def test_rejects_undeclared_diagnostic_changes(tmp_path, mutation):
    stages, metadata, paths = _pair(tmp_path)
    stage, meta = stages[1], metadata[1]
    body = stage.GetPrimAtPath(meta['actors']['tray'])
    if mutation == 'mass':
        UsdPhysics.MassAPI(body).GetMassAttr().Set(.001)
    elif mutation == 'mesh':
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(str(body.GetPath())+'/Collision/shell'))
        points = mesh.GetPointsAttr().Get()
        points[0] += Gf.Vec3f(0., 0., .001)
        mesh.GetPointsAttr().Set(points)
    elif mutation == 'material':
        prim = stage.GetPrimAtPath(meta['root_path']+'/Materials/Contact')
        UsdPhysics.MaterialAPI(prim).GetDynamicFrictionAttr().Set(3.)
    elif mutation == 'relationship':
        body.CreateRelationship('hidden_attachment').SetTargets(['/World/HiddenSupport'])
    elif mutation == 'extra_prim':
        UsdGeom.Cube.Define(stage, '/World/KitchenTask/HiddenSupport')
    elif mutation == 'wrong_pose':
        body.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0., 0., 2.))
    elif mutation == 'animated_pose':
        body.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0., 0., 2.), 1.)
    elif mutation == 'body_metadata':
        metadata[1] = deepcopy(meta)
        metadata[1]['objects'][0]['mass_kg'] = .001
    stage.GetRootLayer().Export(str(paths[1]))
    assert not diagnostic_physics_identity(*paths, *metadata)['passed']


def test_rejects_fixture_pose_change_even_with_matching_fixture_metadata(tmp_path):
    stages, metadata, paths = _pair(tmp_path, 'produce_transfer')
    stage, meta = stages[1], metadata[1]
    fixture = stage.GetPrimAtPath(meta['fixtures']['source_board'])
    fixture.GetAttribute('xformOp:translate').Set(Gf.Vec3d(.15, -.90, .95))
    stage.GetRootLayer().Export(str(paths[1]))
    assert not diagnostic_physics_identity(*paths, *metadata)['passed']


@pytest.mark.parametrize('canonical_limit,goal_limit,accepted', [
    (float('inf'), float('inf'), True),
    (-float('inf'), -float('inf'), True),
    (float('inf'), -float('inf'), False),
    (float('inf'), 1000., False),
    (float('nan'), float('nan'), False),
])
def test_unbounded_physics_limits_compare_exactly_and_nan_fails_closed(
        tmp_path, canonical_limit, goal_limit, accepted):
    stages, metadata, paths = _pair(tmp_path)
    for stage, meta, limit, path in zip(stages, metadata, (canonical_limit, goal_limit), paths):
        body = stage.GetPrimAtPath(meta['actors']['tray'])
        body.CreateAttribute('physxRigidBody:maxContactImpulse', Sdf.ValueTypeNames.Float,
                             custom=False).Set(limit)
        stage.GetRootLayer().Export(str(path))
    result = diagnostic_physics_identity(*paths, *metadata)
    assert result['passed'] is accepted, result
