"""Two-object workbench for teleoperation checks, not a benchmark release."""

from bench.paths import data_root


def inputs():
    scene = {
        'schema_version': 'teleop-sandbox-v1',
        'native_readback_profile': 'cpu_rigid_v1',
        'description': 'Table-edge FR3/SHARPA and two rigid objects for teleoperation checks.',
        'robot': {
            'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2',
            'base_position_m': [-.32, 0., .77],
            # Operator-supplied reference for approaching tabletop objects from the side.
            'arm_home_rad': [-.3435, .13963, -.27925, -2.00383, 1.97222, 1.69297, -.62832],
            'arm_kp': 2000., 'arm_kd': 140., 'hand_kp': 3., 'hand_kd': .1,
            'self_collisions': True,
        },
        'physics': {'dt_s': .0005, 'control_dt_s': .02, 'gravity_m_s2': [0., 0., -9.81]},
        'isaacsim': {
            'physics_dt_s': .0005, 'robot_self_collisions': True,
            'filter_thumb_cmc_structure': True,
            'arm_gravity_compensation': 'physx_feedforward_v1',
            'hand_actuator_model': 'sharpa_official_position_mode_v1',
            'robot_contact_materials': {'pad_friction': .5, 'shell_friction': .5},
            'robot_material_binding': 'collider_instance_v1',
            'robot_base_constraint': 'native_fixed_base_v1',
        },
        'table': {'position_m': [.35, 0., .72], 'half_extents_m': [.8, .6, .03], 'friction': .4},
        'objects': [
            {'id': name, 'asset_id': asset, 'asset_root': str(data_root()/'assets/task_parts'/root),
             'position_m': [.22, y, .75-bottom],
             'quaternion_wxyz': [1., 0., 0., 0.], 'friction': .4}
            # Lower bounds are in each existing asset's centre-of-mass frame.
            for name, asset, root, y, bottom in (
                ('bottle', 'capped_bottle', 'scene_rooms/kitchen/v8', -.14, -.095),
                ('can', 'steel_can', 'manipulation_collection/v2', .14, -.04211684687826997))],
        'fixtures': [], 'cloth': {'enabled': False},
        'cameras': {'overview': {'kind': 'world', 'eye': [1.25, -1.25, 1.65],
            'target': [.25, 0., 1.1], 'focal_length_mm': 24., 'resolution': [640, 480]},
            'wrist': {'kind': 'link', 'mount_link': 'right_hand_flange',
                'offset_xyz_m': [-.14, -.10, .025], 'aim_target_m': [.22, 0., .75],
                'standoff_m': 0., 'focal_length_mm': 18.,
                'horizontal_aperture_mm': 36., 'vertical_aperture_mm': 27.,
                'clipping_m': [.015, 8.], 'resolution': [640, 480]}},
    }
    return scene, {'joint_positions_rad': {}}, None
