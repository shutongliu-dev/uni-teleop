"""Task7's new physical handling geometry must keep the real belt path intact."""
from copy import deepcopy
import numpy as np
import pytest
from pxr import Usd
from scenes.workrooms.mechanical_assembly.nist_assets import author_task, structural_audit
from scripts.validation.mechanical_nist_interactions import probe_reset_poses, tensioner_transition_check


def test_idler_hub_changes_only_idler_and_keeps_source_belt():
    original = author_task(Usd.Stage.CreateInMemory(), 'nist_belt_drive_assembly')
    stage = Usd.Stage.CreateInMemory()
    revised = author_task(stage, 'nist_belt_drive_assembly', usability_profile='nist_manipulation_scale_v1')
    old = {o['id']: o for o in original['objects']}
    new = {o['id']: o for o in revised['objects']}
    assert {name for name in old if old[name] != new[name]} == {'tensioner_idler'}
    changed_fixtures = [name for name in original['fixture_metadata']
                        if original['fixture_metadata'][name] != revised['fixture_metadata'][name]]
    assert changed_fixtures == ['tensioner_idler_shaft']
    assert original['diagnostic_goal_poses'] == revised['diagnostic_goal_poses']
    assert original['source'] == revised['source']
    assert original['task_parameters']['belt'] == revised['task_parameters']['belt']
    assert original['task_parameters']['seating_goals'] == revised['task_parameters']['seating_goals']
    assert original['task_parameters']['flexible_groups'] == revised['task_parameters']['flexible_groups']
    assert structural_audit(stage, revised)['passed']
    idler = new['tensioner_idler']
    assert idler['collision_approximation'] == 'sdf'
    assert len(idler['collider_paths']) == 1
    assert idler['mass_kg'] > old['tensioner_idler']['mass_kg']
    assert not np.allclose(idler['center_of_mass_m'], old['tensioner_idler']['center_of_mass_m'])
    belt_top = .8075 + .0254 / 2
    target_top = revised['diagnostic_goal_poses']['tensioner_idler'][2] + idler['local_bounds_m'][1][2]
    assert target_top - belt_top > .020
    shaft = revised['fixture_metadata']['tensioner_idler_shaft']
    shaft_top = shaft['initial_position_m'][2] + shaft['local_bounds_m'][1][2]
    assert target_top - shaft_top == pytest.approx(.003)
    assert shaft_top - belt_top > .017
    assert revised['task_parameters']['assembly_guidance']['mandatory_sequence'] is False


def test_passive_drop_does_not_pass_static_goal_or_short_fall():
    metadata = author_task(Usd.Stage.CreateInMemory(), 'nist_belt_drive_assembly',
                          usability_profile='nist_manipulation_scale_v1')
    poses = probe_reset_poses(metadata, 'tensioner_drop')
    objects = {'tensioner_idler': {'position_m': metadata['diagnostic_goal_poses']['tensioner_idler'][:3]}}
    assert tensioner_transition_check(metadata, poses, objects)['passed']
    assert not tensioner_transition_check(metadata, metadata['diagnostic_goal_poses'], objects)['passed']
    unchanged = deepcopy(objects)
    unchanged['tensioner_idler']['position_m'] = poses['tensioner_idler'][:3]
    assert not tensioner_transition_check(metadata, poses, unchanged)['passed']
    with pytest.raises(ValueError):
        probe_reset_poses({'task_id': 'nist_cable_routing', 'diagnostic_goal_poses': {}}, 'tensioner_drop')
    guided = probe_reset_poses(metadata, 'tensioner_guided_drop')
    result = tensioner_transition_check(metadata, guided, objects, case='tensioner_guided_drop')
    assert result['passed'] and not result['initially_fully_detached']
    assert not tensioner_transition_check(metadata, guided, objects)['passed']


def test_canonical_belt_supply_clears_every_fixed_solid_for_every_supply_translation():
    from scenes.workrooms.mechanical_assembly.nist_assets import ROOT_PRIM, BELT_SUPPLY_CENTER_XY_M
    from scenes.workrooms.mechanical_assembly.placement_authoring import _belt_initial_clearance
    from scenes.workrooms.mechanical_assembly.placement_variants import BELT_SAFE_TRANSLATIONS_M
    # 1 mm along -y from the original (.280, -.055) loop centre; x is unchanged.
    assert BELT_SUPPLY_CENTER_XY_M == (.280, -.056)
    for profile in (None, 'nist_manipulation_scale_v1'):
        for dx, dy in BELT_SAFE_TRANSLATIONS_M:
            stage = Usd.Stage.CreateInMemory()
            metadata = author_task(stage, 'nist_belt_drive_assembly', usability_profile=profile)
            moving = [name for name in metadata['actors'] if name != 'drive_pulley']
            for name in moving:
                op = stage.GetPrimAtPath(metadata['actors'][name]).GetAttribute('xformOp:translate')
                value = op.Get(); op.Set(type(value)(value[0]+dx, value[1]+dy, value[2]))
            # Exact cube/convex-hull LP of every supply solid against every fixed one.
            result = _belt_initial_clearance(stage, ROOT_PRIM, {metadata['actors'][name] for name in moving})
            assert result['minimum_original_support_gap_m'] == pytest.approx(.0003, abs=1e-6)
            assert min(result['workplate_xy_margin_m']) > .003
            if (dx, dy) == (0., 0.) and profile is not None:
                # The links that crossed the 20 mm idler shoulder now clear it by > 0.3 mm.
                closest = max(row['common_inscribed_sphere_radius_m'] for row in result['narrow_phase'])
                assert closest < -1.5e-4
