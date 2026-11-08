"""Current task contracts and retained historical topology; CPU evidence only."""

import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs, ROBOT_CAPABILITIES
from scenes.workrooms.laundry_room.robot_layout import LAYOUT_ID
from scenes.workrooms.laundry_room.rules import ALL_TASK_IDS, GARMENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, default_profile


@pytest.mark.parametrize('task_id', ALL_TASK_IDS)
def test_all_explicit_installations_bind_both_hands_and_native_cloth(task_id):
    inputs = build_task_inputs(task_id, robot_layout=LAYOUT_ID)
    scene, task, initial = (inputs[key] for key in ('scene', 'task', 'initial_state'))
    assert task['capabilities'] == ROBOT_CAPABILITIES
    assert task['active_sides'] == ['left', 'right']
    assert task['evaluation']['rule_profile'] == default_profile(task_id)
    assert len(initial['joint_positions_rad']) == 58
    assert scene['robot']['hand_id'] == 'dual_fr3_sharpa_w01'
    assert scene['named_cloth_initialization'] == 'author-only'
    assert set(scene['cameras']) == {'overview', 'task', 'room', 'left_wrist', 'right_wrist'}
    assert scene['workroom']['version'] == 'laundry_residential_20261021_v3'
    assert scene['goal_markers'] == []
    assert len(scene['cloths']) == (2 if task_id.endswith('_stacking') else 1)
    for cloth in scene['cloths']:
        assert cloth['enabled'] and cloth['self_contact']
        assert .0005 <= cloth['visual_thickness_m'] <= .003
        assert cloth['visual_thickness_m'] == cloth['isaacsim']['cloth_thickness_m']
    if task_id in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS):
        for cloth, metadata in zip(scene['cloths'], task['asset_metadata']['cloths']):
            assert cloth['mesh_type'] == 'custom_surface'
            assert 'cells_xy' not in cloth
            assert cloth['rest_area_m2'] == metadata['rest_area_m2']
            assert np.array_equal(cloth['triangles'], metadata['triangles'])
            assert np.array_equal(cloth['rest_vertices_m'], metadata['rest_vertices_m'])
            assert cloth['visual_face_groups']
            assert metadata['boundary_features']


@pytest.mark.parametrize('task_id', (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS))
def test_garment_material_thickness_cannot_be_silently_substituted(task_id):
    for wrong in (.0001, .006):
        with pytest.raises(ValueError, match='physical thickness'):
            build_task_inputs(task_id, robot_layout=LAYOUT_ID,
                              physics_options={'cloth_thickness_m': wrong})
