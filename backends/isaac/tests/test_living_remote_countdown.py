"""Countdown logic/authoring tests; no claim of real hand-object execution."""
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from test_living_remote_evaluator import Replay
from bench.paths import load_json
from scenes.workrooms.living_room.tasks.remote_operation.assets import author_task
from scenes.workrooms.living_room.tasks.remote_operation.evaluator import TelevisionState
from scenes.workrooms.living_room.tasks.remote_operation.rules import RULES_VERSION, public_rules


def replay():
    return Replay(version=RULES_VERSION, physics_dt_s=.002)


def test_countdown_uses_release_time_and_automatic_off_keeps_completed_interval():
    tv = TelevisionState(10., .3, automatic=True)
    assert not tv.state(100.)['countdown_completed']
    for kind, time in [('pressed', .1), ('released', .2)]:
        tv.event({'button': 'power', 'kind': kind, 'time_s': time, 'handheld_valid': True})
    assert tv.state(.2)['countdown_seconds'] == 10
    assert tv.state(10.198)['power_on']
    off = tv.state(10.2)
    assert not off['power_on'] and off['countdown_completed']
    assert off['countdown_seconds'] == 0 and not any(off['countdown_segments'].values())
    assert tv.state(11.)['phase'] == 'await_return'
    assert tv.state(11., .3)['phase'] == 'completed'


def test_real_event_fixture_needs_handheld_cycle_then_auto_off_and_released_holder():
    r = replay()
    assert r.result['reason'] == 'reset_baseline_not_scored'
    r.cycle('confirm')
    assert not r.result['television']['power_on']
    r.cycle('power')
    r.run(5000)
    assert r.result['television']['phase'] == 'await_return'
    assert not r.result['success'] and r.result['completed_button_cycles'] == 1
    r.seat(160)
    assert r.result['success'] and not r.result['television']['power_on']
    completed = r.result
    r.seat(20)
    assert r.result == completed


@pytest.mark.parametrize('fault', ['table', 'no_finger', 'held'])
def test_nonphysical_or_supported_press_never_starts_countdown(fault):
    r = replay()
    r.cycle('power', **{fault: fault != 'held'})
    r.run(5000)
    assert not r.result['television']['countdown_completed']
    assert not r.seat(160)['success']


def test_early_manual_off_restarts_full_interval_and_initial_holder_never_scores():
    r = replay()
    assert not r.seat(200)['success']
    r.cycle('power'); r.run(400); r.cycle('power')
    assert not r.result['television']['power_on']
    assert not r.result['television']['countdown_completed']
    r.cycle('power')
    assert r.result['television']['countdown_seconds'] == 10
    r.run(5000)
    assert r.result['television']['countdown_completed']
    assert r.seat(160)['success']


def test_face_up_buttons_are_the_same_physical_caps_and_screen_segments_are_visual_only():
    stage = Usd.Stage.CreateInMemory(); meta = author_task(stage)
    assert meta['remote']['face_up'] and not meta['presentation']
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    for row in meta['buttons'].values():
        cap = stage.GetPrimAtPath(row['body_path']+'/Cap')
        assert cap.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(cap).ComputeVisibility() != 'invisible'
        assert cache.ComputeWorldBound(cap).ComputeAlignedRange().GetMin()[2] > .798
    segments = meta['television_display']['segments']
    assert len(segments) == 14
    for path in segments.values():
        prim = stage.GetPrimAtPath(path)
        assert prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    rules = public_rules()
    assert rules['success']['automatic_power_off'] is True
    assert rules['success']['stable_duration_s'] == .3


def test_current_three_installations_expose_frozen_auto_off_rules_without_result_dependencies():
    import json
    from pathlib import Path
    from bench.isaacsim.release import digest, task_inputs, load_evaluator
    from bench.runtime.manipulation_tasks import project_manipulation_task
    from scenes.workrooms.catalog import task_definition
    for choice in ('single_arm', 'dual_same_side', 'dual_opposed'):
        definition = task_definition('living_room', 'remote_operation', choice)
        assert definition['runnable'] and definition['default_configuration_id'] == 'single_arm'
        reference = definition['source_release']; root = Path(reference['path'])
        assert digest(root/'release_manifest.json') == reference['manifest_sha256']
        _, scene, task, initial = task_inputs(root, 'remote_operation')
        assert task['evaluation']['schema'] == 'living_remote_evaluation.v4'
        assert task['public_goal']['automatic_power_off'] is True
        assert task['asset_metadata']['remote']['face_up'] is True
        assert len(initial['joint_positions_rad']) == (29 if choice == 'single_arm' else 58)
        load_evaluator(root)(task)
        public = project_manipulation_task(task)
        assert '自动关机' in public['public_rules']['instruction_zh']
        assert public['public_rules']['thresholds']['required_on_s'] == 10.
        assert len(scene['native_task_bundle']['visual_state_bindings']['bindings']) == 15
        manifest = load_json(root/'release_manifest.json')
        # The user-designated parameterized publication lives in this one
        # immutable development dataset; no other development path is allowed.
        allowed = '/development/parameterized_seeds_20261105/'
        assert not any('/development/' in path.replace(allowed, '/') for path in manifest['asset_files_sha256'])
        assert '/development/' not in json.dumps(definition['qualification']).replace(allowed, '/')
