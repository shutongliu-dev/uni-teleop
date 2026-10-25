"""Diagnostic composition must use the new room and reset posture."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.dynamics import diagnostic_scene
from scenes.workrooms.mechanical_assembly.robot_layout import READY_FINGER_POSTURE_RAD
from scenes.workrooms.mechanical_assembly.dynamics import diagnostic_scene_v3, preinserted_scene_v3


def test_current_room_and_original_threaded_bodies_are_isolated_diagnostics():
    scene, initial, specs, design = diagnostic_scene(
        f'{DATA_ROOT}/assets/task_parts/mechanical_workshop_supports/v1',
        f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v1')
    assert scene['workroom']['version'] == 'mechanical_workshop_20261017_v3'
    assert scene['robot']['base_position_m'] == [-.23, 0, .78]
    assert len(initial['joint_positions_rad']) == 29
    assert all(initial['joint_positions_rad'][k] == v for k, v in READY_FINGER_POSTURE_RAD.items())
    assert set(specs) == {'cube_table', 'cube_floor', 'screwdriver', 'bolt'}
    assert len(scene['fixtures']) == 2
    assert scene['isaacsim']['robot_base_constraint'] == 'native_fixed_base_v1'
    assert not scene['mechanical_diagnostic_provenance']['task_success_claimed']
    for obj in scene['objects']:
        if obj['id'] in ('screwdriver', 'bolt'):
            assert 'friction' not in obj
            assert obj['usd_physics_policy'] == 'preserve_authored_v1'
            bounds = specs[obj['id']]
            bottom = bounds['bounds_min_m'][1] if obj['id'] == 'screwdriver' else -bounds['bounds_max_m'][2]
            assert obj['position_m'][2] + bottom == .81
    bolt = next(obj for obj in scene['objects'] if obj['id'] == 'bolt')
    assert bolt['quaternion_wxyz'] == [0, 1, 0, 0]


def test_selected_v3_assets_keep_the_recorded_profile_and_reset_only_seed():
    args = (f'{DATA_ROOT}/assets/task_parts/mechanical_workshop_supports/v1',
            f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v2')
    scene, _, specs, _ = diagnostic_scene_v3(*args)
    assert scene['mechanical_diagnostic_provenance']['asset_profile'] == 'threaded_sharpa_v3'
    driver = next(x for x in scene['objects'] if x['id'] == 'screwdriver')
    assert driver['asset_id'] == 'mechanical_screwdriver_sharpa_v3'
    assert driver['usd_physics_fallbacks']['solver_velocity_iterations'] == 16
    assert driver['usd_physics_fallbacks']['max_depenetration_velocity_m_s'] == .05
    seeded, _, drops, _ = preinserted_scene_v3(*args)
    assert set(drops) == {'cube_table', 'cube_floor'}
    assert set(specs) == {'cube_table', 'cube_floor', 'screwdriver', 'bolt'}
    assert seeded['mechanical_diagnostic_provenance']['threaded_reset_seed']['trajectory_replay'] is False
