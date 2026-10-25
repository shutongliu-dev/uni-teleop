"""Workbench practice keeps native task geometry and original support frames."""
from copy import deepcopy
import json
import re

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle
from bench.simulation.isaacsim_backend.workroom import add_workroom, apply_runtime_lighting, SUPPORT_PATHS
from scenes.workrooms.catalog import list_tasks, load_task
from scenes.workrooms.teleop_workbench import apply


@pytest.mark.parametrize('definition', list_tasks(), ids=lambda d: d['family']+'/'+d['task_id'])
def test_published_tasks_keep_actors_fixtures_and_declared_supports(definition):
    loaded = load_task(definition['family'], definition['task_id'])
    original = loaded['scene']
    before = deepcopy(original)
    scene = apply(original, definition)
    assert original == before
    for key in ('robot', 'objects', 'fixtures', 'physics', 'task_layers'):
        assert scene.get(key) == original.get(key)
    assert {k: v for k, v in scene['isaacsim'].items() if k != 'workroom_lighting'} == {
        k: v for k, v in original['isaacsim'].items() if k != 'workroom_lighting'}
    stage = Usd.Stage.CreateInMemory()
    room = add_workroom(stage, scene['workroom'])
    apply_runtime_lighting(stage, scene['workroom'], scene['isaacsim']['workroom_lighting'])
    selection = room['geometry_selection']
    assert selection['removed'] and len(selection['kept']) < len(selection['removed'])
    assert all(stage.GetPrimAtPath(p).IsActive() for p in SUPPORT_PATHS.values())
    assert all(not stage.GetPrimAtPath(p).IsActive() for p in selection['removed'])
    source = Usd.Stage.Open(scene['workroom']['usd_path'])
    cache = UsdGeom.XformCache()
    for path in selection['kept']:
        prim = stage.GetPrimAtPath(path)
        source_prim = source.GetPrimAtPath(path.replace('/World/Room/', '/World/'))
        assert cache.GetLocalToWorldTransform(prim) == cache.GetLocalToWorldTransform(source_prim)
    # Observer-declared environment contacts must still exist, even in practice.
    for path in set(re.findall(r'/World/(?:Room/)?Geometry/[A-Za-z0-9_/]+', json.dumps(original))):
        target = path.replace('/World/Geometry/', '/World/Room/Geometry/')
        prim = stage.GetPrimAtPath(target)
        assert prim and prim.IsActive(), target
    bundle = scene.get('native_task_bundle')
    if bundle:
        info = add_bundle(stage, bundle, physics_scene_path='/physicsScene')
        assert set(info['objects']) == set(bundle['rigid_bodies'])
        source = Usd.Stage.Open(bundle['usd_path'])
        for prim in Usd.PrimRange(stage.GetPrimAtPath(bundle['root_prim_path'])):
            source_prim = source.GetPrimAtPath(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh):
                assert prim.GetAttribute('points').Get() == source_prim.GetAttribute('points').Get()
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                assert prim.GetAttribute('physics:collisionEnabled').Get() == source_prim.GetAttribute('physics:collisionEnabled').Get()


def test_selection_cannot_remove_required_support_or_dynamic_task_group():
    loaded = load_task('kitchen', 'breakfast_serving')
    scene = apply(loaded['scene'], loaded['definition'])
    scene['workroom']['geometry_selection'] = ['room_floor']
    with pytest.raises(ValueError, match='table collider'):
        add_workroom(Usd.Stage.CreateInMemory(), scene['workroom'])
    bundle = scene['native_task_bundle']
    bundle['excluded_static_roots'] = [bundle['root_prim_path']+'/Actors']
    with pytest.raises(ValueError, match='cannot contain dynamics'):
        add_bundle(Usd.Stage.CreateInMemory(), bundle, physics_scene_path='/physicsScene')


def test_cli_variant_is_explicit_and_preserves_initial_state():
    from scripts.teleop.pico_isaac import load_inputs, parser
    original, initial, release = load_inputs(parser().parse_args([]))
    scene, other_initial, other_release = load_inputs(parser().parse_args(['--workbench-only']))
    assert 'geometry_selection' not in original['workroom']
    assert scene['teleop_scene_variant']['version'] == 'teleop_workbench_v1'
    assert initial == other_initial and release == other_release
    with pytest.raises(ValueError, match='not --sandbox'):
        load_inputs(parser().parse_args(['--sandbox', '--workbench-only']))


def test_practice_arm_home_changes_only_robot_initial_posture():
    from scripts.teleop.pico_isaac import load_inputs, parser
    pose = [-.3435, .13963, -.27925, -2.00383, 1.97222, 1.69297, -.62832]
    flags = ['--family','mechanical_assembly','--task','parts_sorting','--workbench-only','--no-task-scoring']
    scene, initial, release = load_inputs(parser().parse_args(flags))
    changed, state, same_release = load_inputs(parser().parse_args(flags+['--arm-home',*map(str,pose)]))
    assert release == same_release
    assert changed['robot']['arm_home_rad'] == pose
    assert [state['joint_positions_rad'][f'fr3_joint{i}'] for i in range(1,8)] == pose
    assert scene['robot'].get('arm_home_rad') != pose
    assert initial != state
    for key in scene:
        if key != 'robot':
            assert changed[key] == scene[key]
    assert {k:v for k,v in initial.items() if k!='joint_positions_rad'} == {
        k:v for k,v in state.items() if k!='joint_positions_rad'}
    with pytest.raises(ValueError,match='unscored practice'):
        load_inputs(parser().parse_args(['--arm-home',*map(str,pose)]))
    with pytest.raises(ValueError,match='joint limits'):
        load_inputs(parser().parse_args(flags+['--arm-home','nan',*map(str,pose[1:])]))
