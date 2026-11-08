"""Wiping referee regression scenarios; synthetic traces are not native evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.composition import bind_episode_task
from scenes.workrooms.kitchen.native_evidence import fixed_state
from scenes.workrooms.kitchen.rules import public_rules, thresholds
from scenes.workrooms.kitchen.wiping import ContactWiping, validate_wiping_metadata
from test_kitchen_evaluator import Replay, SURFACE, HAND


@pytest.fixture(scope='module')
def authored():
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    return author_task(stage, 'surface_wiping', dressing_profile='enriched_v1')


def patch_centers(metadata):
    return [np.asarray([c['center_world_m'][:2] for c in metadata['goals']['soil_cells']
                           if c['patch'] == p['id']]).mean(axis=0)
               for p in metadata['goals']['soil_patches']]


def sweep(replay, *, contact=True, z=.9002, reversed_force=False, jump=False,
          patch_count=3):
    # Two overlapping strokes per patch. No exact target pose or order required.
    centers = patch_centers(replay.meta)
    for center_x, center_y in centers[:patch_count]:
        for offset in (.019, -.019):
            replay.contacts = []
            replay.put('sponge', (center_x-.09, center_y+offset, z+.03))
            replay.run(10)
            replay.put('sponge', (center_x-.09, center_y+offset, z))
            replay.contacts = ([replay.pair('sponge', SURFACE,
                impulse=(0, 0, -.001) if reversed_force else (0, 0, .001))] if contact else [])
            replay.run(10)
            for x in np.linspace(center_x-.09, center_x+.09, 2 if jump else 451)[1:]:
                replay.put('sponge', (float(x), center_y+offset, z))
                replay.run(1)
    if replay.step % 10:
        replay.run(10-replay.step % 10)
    return replay.result


def return_sponge(replay, *, held=False):
    tray = replay.meta['fixture_metadata']['sponge_tray']
    pose = tray['initial_pose_xyzw'][:3]
    replay.put('sponge', (*pose[:2], pose[2]+tray['floor_top_local_z_m']+.0002))
    replay.contacts = [replay.pair('sponge', 'sponge_tray')]
    if held:
        replay.contacts.append(replay.pair('sponge', HAND))
    return replay.run(240)


def test_actual_contact_sweeps_then_supported_return_pass(authored):
    r = Replay('surface_wiping', meta=deepcopy(authored))
    result = sweep(r)
    assert result['detail']['coverage_fraction'] >= .8
    assert min(result['detail']['patch_coverage'].values()) >= .6
    assert result['referee']['phase'] == 'returning'
    assert not result['success']
    assert not return_sponge(r, held=True)['success']
    assert return_sponge(r)['success'], r.result


@pytest.mark.parametrize('case', ['hover', 'no_contact', 'downward_contact', 'jump'])
def test_projection_and_nonloaded_sweeps_never_clean(authored, case):
    r = Replay('surface_wiping', meta=deepcopy(authored))
    sweep(r, contact=case != 'no_contact', z=.9202 if case == 'hover' else .9002,
          reversed_force=case == 'downward_contact', jump=case == 'jump')
    assert r.result['detail']['coverage_fraction'] == 0.
    assert not return_sponge(r)['success']


def test_static_loaded_sponge_and_wrong_support_never_clean(authored):
    r = Replay('surface_wiping', meta=deepcopy(authored))
    x, y = patch_centers(authored)[0]
    r.put('sponge', (x, y, .9002))
    r.contacts = [r.pair('sponge', SURFACE)]
    r.run(600)
    assert r.result['detail']['coverage_fraction'] == 0.
    r.contacts = [r.pair('sponge', 'sponge_tray')]
    for moved_x in np.linspace(x-.06, x+.07, 501):
        r.put('sponge', (float(moved_x), y, .9002))
        r.run(1)
    assert r.observer.on_observe()['detail']['coverage_fraction'] == 0.


def test_native_scale_microvibration_cannot_accumulate_cleaning(authored):
    model = ContactWiping(authored, thresholds('surface_wiping'), .0005)
    x, y = patch_centers(authored)[0]
    for step in range(5000):
        pose = fixed_state([x+(.00002 if step % 2 else -.00002), y, .9002, 0, 0, 0, 1])
        result = model.update(pose, loaded_contact=True, baseline=step == 0)
    # Summed path exceeds 0.19 m, while the actual excursion is only 40 µm.
    assert result['coverage_fraction'] == 0.
    assert not model.travel.any()


def test_lifting_and_retry_keep_cleaned_area_but_two_patches_are_insufficient(authored):
    r = Replay('surface_wiping', meta=deepcopy(authored))
    sweep(r, patch_count=2)
    before = deepcopy(r.result['detail']['cleaned_cells'])
    assert .5 < r.result['detail']['coverage_fraction'] < .8
    assert not return_sponge(r)['success']
    assert before == r.result['detail']['cleaned_cells']
    sweep(r)
    assert return_sponge(r)['success']


def test_reset_clears_history_and_missing_native_sample_is_unknown(authored):
    r = Replay('surface_wiping', meta=deepcopy(authored))
    sweep(r)
    r.observer.reset()
    assert not r.observer.wiping.travel.any()
    r = Replay('surface_wiping', meta=deepcopy(authored))
    r.step += 1
    r.send()
    r.observe()
    assert r.result['status'] == 'unknown'
    assert not r.result['physical_failure']


def test_soil_visuals_are_nonphysical_and_public_contract_has_no_truth(authored):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scripts.assets.build_kitchen_tasks import audit_task
    from bench.simulation.isaacsim_backend.task_visual_state import BooleanMaterialBindings
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    meta = author_task(stage, 'surface_wiping', dressing_profile='enriched_v1')
    assert audit_task(stage, 'surface_wiping', meta)['dynamic_body_count'] == 1
    assert len(meta['contextual_props']) == 12
    bindings = BooleanMaterialBindings(stage, meta['visual_state_bindings'])
    r = Replay('surface_wiping', meta=meta)
    initial = r.observer.on_observe()
    receipt = bindings.sync(initial, native_step=0)
    assert len(receipt['bindings']) == len(meta['goals']['soil_cells'])
    assert all(not row['value'] for row in receipt['bindings'])
    for row in meta['visual_state_bindings']['bindings']:
        assert not stage.GetPrimAtPath(row['target_prim_path']).HasAPI(UsdPhysics.CollisionAPI)
    from scenes.workrooms.kitchen import rules
    public = bind_episode_task({'id': 'surface_wiping', 'asset_metadata': meta}, 17, rules)['public_goal']
    rendered = json.dumps(public)
    assert all(name not in rendered for name in ('position_m', 'center_world_m', '/World/', 'cleaned_cells'))
    assert public_rules('surface_wiping')['schema'] == 'kitchen_rules.v4'


def test_three_distinct_patches_and_physical_face_are_required(authored):
    bad = deepcopy(authored)
    bad['goals']['soil_patches'][1]['cell_ids'] = bad['goals']['soil_patches'][0]['cell_ids']
    with pytest.raises(ValueError, match='exactly one'):
        validate_wiping_metadata(bad)
    bad = deepcopy(authored)
    bad['objects'][0]['wiping_face_bounds_xy_m'][1][0] = -1
    with pytest.raises(ValueError, match='nonempty'):
        validate_wiping_metadata(bad)
