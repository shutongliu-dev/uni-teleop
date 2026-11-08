"""Task scope, physical chirality and source-derived workspace coverage."""
from pathlib import Path

import numpy as np
import pytest

from scenes.workrooms import bimanual_layout as layout


@pytest.mark.parametrize('family', tuple(layout.ELIGIBLE_TASKS))
@pytest.mark.parametrize('configuration', layout.CONFIGURATION_IDS)
def test_official_58_dof_source_joints_and_stand_contact(family, configuration):
    from bench.control.factory import public_robot_geometry, robot_configuration
    from bench.simulation.isaacsim_backend.robot_layout import world_base_poses
    task=layout.ELIGIBLE_TASKS[family][0]
    robot,initial,ready=layout.robot_inputs(family,task,configuration)
    _,source,_=layout._source_inputs()
    assert initial==source
    assert robot['hand_id']=='dual_fr3_sharpa_w01'
    assert robot['assembly_version']=='isaacsim_6_1_v2'
    assert len(initial['joint_positions_rad'])==58
    models=public_robot_geometry(robot_configuration({'robot':robot})).sides
    bases=world_base_poses(robot)
    for side,model in models.items():
        assert model.hand_posture_prior.side==side
        assert any(Path(path).name==side+'_sharpa_wave_with_flange.urdf' for path in model.asset_sha256)
        own={n:initial['joint_positions_rad'][n] for n in model.names}
        model.checked_joints(own,enforce_limits=True)
        np.testing.assert_allclose(bases[side]['position_m'],ready[side]['base_position_m'])
        stand=layout.stand_geometry(family,task,configuration)[side]
        lows=[b['position_m'][2]-b['size_m'][2]/2 for b in stand['boxes']]
        highs=[b['position_m'][2]+b['size_m'][2]/2 for b in stand['boxes']]
        np.testing.assert_allclose(lows[0],stand['ground_datum_m'],atol=1e-12)
        np.testing.assert_allclose(lows[1:],highs[:-1],atol=1e-12)
        np.testing.assert_allclose(highs[-1],bases[side]['position_m'][2],atol=1e-12)
        assert ready[side]['layout_id']==stand['layout_id']


@pytest.mark.parametrize('family,task', [('kitchen','pouring'),('laundry','folding'),
    ('living_room','tower_extraction'),('mechanical_assembly','cooperative_insertion'),
    ('card_room','tic_tac_toe')])
def test_excludes_families_and_existing_competitive_or_bimanual_tasks(family,task):
    with pytest.raises(ValueError,match='no optional bimanual'):
        layout.robot_inputs(family,task,'dual_same_side')


def test_task_probes_include_actual_rear_shelf_counter_and_all_board_cells():
    from scenes.workrooms.catalog import task_definition
    from scripts.diagnostics.check_bimanual_options import task_points
    def samples(f,t):return task_points(f,t,task_definition(f,t)['source_release']['path'])[0]
    book=samples('study','book_retrieval')
    assert any(s['approach']=='shelf_front' and s['fingertip_target_m'][0]<-.5 for s in book)
    flat=samples('study','book_stacking')
    assert all(.78<s['reference_position_m'][2]<.84 for s in flat if s['label'].startswith('supply/'))
    lab=samples('laboratory','vial_racking')
    assert any(s['fingertip_target_m'][1]>2 for s in lab)
    assert any(.4<s['fingertip_target_m'][1]<.7 for s in lab)
    card=samples('card_room','task0')
    assert len([s for s in card if s['label'].startswith('target/cell_')])==9
    assert sum(map(len,layout.ELIGIBLE_TASKS.values()))==26


def test_scope_rejects_shared_asset_overwrite_and_unknown_configuration(tmp_path):
    with pytest.raises(ValueError,match='Unknown bimanual'):
        layout.placements('study','book_retrieval','dual_right_hands')
    with pytest.raises(ValueError,match='shared assets'):
        layout.prepare_stands('study','book_retrieval','dual_same_side',tmp_path/'stand')


def test_laboratory_opposed_mount_is_supported_inside_unchanged_service_spine():
    stand=layout.stand_geometry('laboratory','vial_racking','dual_opposed')['right']
    assert stand['installation']=='tabletop_bolted_plate'
    assert len(stand['boxes'])==1
    box=stand['boxes'][0]
    lower=np.asarray(box['position_m'])-np.asarray(box['size_m'])/2
    upper=np.asarray(box['position_m'])+np.asarray(box['size_m'])/2
    np.testing.assert_allclose(lower[2],.9)
    np.testing.assert_allclose(upper[2],.925)
    assert lower[1]>-.316  # Original sealed-tip cassette's north edge.
    assert upper[1]<.5    # Original task supply remains unobstructed.


def test_self_collision_narrow_phase_distinguishes_volume_from_touching():
    from scripts.diagnostics.check_bimanual_arm_self_collision import intersection_radius
    planes=np.array([[1,0,0,-1],[-1,0,0,-1],[0,1,0,-1],
                     [0,-1,0,-1],[0,0,1,-1],[0,0,-1,-1.]],dtype=float)
    assert intersection_radius(np.vstack([planes,planes]))[0]==pytest.approx(1.)
    for translation in (2.,3.):
        other=planes.copy();other[:,3]-=other[:,0]*translation
        assert intersection_radius(np.vstack([planes,other]))[0]==pytest.approx(0.)
