"""Source access revisions preserve every authored physical property except pose."""
from copy import deepcopy

import numpy as np
import pytest
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

from scenes.workrooms.kitchen.assets import author_task


@pytest.mark.parametrize('task_id,actor', [('produce_transfer', 'tomato_0'), ('pot_lid_placement', 'pot')])
def test_access_profile_changes_only_one_rigid_source_pose(task_id, actor):
    before_stage, after_stage = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    before = author_task(before_stage, task_id, prop_profile='household_v1', dressing_profile='enriched_v1')
    after = author_task(after_stage, task_id, prop_profile='household_v1', dressing_profile='enriched_v1',
                        source_access_profile='source_access_v1')
    assert after['goals'] == before['goals']
    for name in ('fixture_metadata', 'contextual_prop_metadata', 'public_goal', 'physics_prior'):
        assert after[name] == before[name]
    body_path = after['actors'][actor]
    assert [str(p.GetPath()) for p in before_stage.Traverse()] == [str(p.GetPath()) for p in after_stage.Traverse()]
    for prim in before_stage.Traverse():
        other = after_stage.GetPrimAtPath(prim.GetPath())
        assert prim.GetTypeName() == other.GetTypeName()
        assert prim.GetAppliedSchemas() == other.GetAppliedSchemas()
        for attribute in prim.GetAttributes():
            if str(prim.GetPath()) == body_path and attribute.GetName() in ('xformOp:translate', 'xformOp:orient'):
                continue
            assert attribute.Get() == other.GetAttribute(attribute.GetName()).Get(), str(attribute.GetPath())
        for relationship in prim.GetRelationships():
            assert relationship.GetTargets() == other.GetRelationship(relationship.GetName()).GetTargets()
    for original, changed in zip(before['objects'], after['objects'], strict=True):
        if original['id'] != actor:
            assert original == changed
            continue
        original, changed = deepcopy(original), deepcopy(changed)
        pose = changed.pop('initial_pose_xyzw')
        old_pose = original.pop('initial_pose_xyzw')
        assert original == changed
        assert pose != old_pose
        if actor == 'tomato_0':
            assert pose[0]-old_pose[0] == pytest.approx(.03)
            assert pose[1:] == old_pose[1:]
        else:
            assert pose[:3] == old_pose[:3]
            assert Rotation.from_quat(pose[3:]).as_euler('xyz', degrees=True)[2] == pytest.approx(-45.)
        actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(after_stage.GetPrimAtPath(body_path))).T
        np.testing.assert_allclose(actual[:3, 3], pose[:3], atol=1.e-8)
        np.testing.assert_allclose(actual[:3, :3], Rotation.from_quat(pose[3:]).as_matrix(), atol=1.e-7)
    shown = author_task(Usd.Stage.CreateInMemory(), task_id, prop_profile='household_v1',
                        presentation=True, source_access_profile='source_access_v1')
    for row in shown['objects']:
        assert row['initial_pose_xyzw'] == before['goals']['body_poses_xyzw'][row['id']]


def test_access_profile_is_explicit_and_rejects_unrelated_tasks_or_unscaled_props():
    for task_id, prop in [('surface_wiping', 'household_v1'), ('produce_transfer', 'legacy')]:
        with pytest.raises(ValueError, match='Source access profile requires'):
            author_task(Usd.Stage.CreateInMemory(), task_id, prop_profile=prop,
                        source_access_profile='source_access_v1')
