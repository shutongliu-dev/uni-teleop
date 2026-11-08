"""Current gear base world: the open-end wrench 50 mm closer to the robot (wrench_reach_v1).

The idler-grip task of grip_catalog is unchanged except for the open-end wrench's canonical table
position, task.ORIGINAL_OPEN_END_WRENCH_XY_M -> task.OPEN_END_WRENCH_XY_M. The immutable base
release was derived from idler_grip_v2 by wrench_release.prepare_release and its two dual
installations by prepare_robot_configurations (robot_configuration_catalog), all from the frozen
source_validate52_gearbase01 snapshot. Development evidence stays external and no qualification is
inherited; the seed, paired and robot-revision rows are published separately.
"""
from copy import deepcopy

from .grip_catalog import RELEASE as GRIP_RELEASE
from .task import OPEN_END_WRENCH_XY_M, ORIGINAL_OPEN_END_WRENCH_XY_M
from bench.paths import DATA_ROOT

RELEASE = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/wrench_reach_v1',
    'manifest_sha256': '1768eea9144936eae13f408bead4fb881939f6299589b87e50d4f5870e2c35a7',
}
RELEASE_VALIDATION = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/gearbase52_evidence01/release_validation.json',
    'sha256': '6df626d1160447a71a1e9a82ff6524f39a034075470257de3bb1c2a7792eed98',
}


def bind_wrench_release(task):
    result = deepcopy(task)
    if result['source_release'] != GRIP_RELEASE:
        raise ValueError('The wrench reach revision derives the idler-grip release')
    result['previous_source_release'] = deepcopy(result['source_release'])
    result['source_release'] = deepcopy(RELEASE)
    result['candidate_referee']['release'] = deepcopy(RELEASE)
    binding = result['objects']['asset_bindings']
    binding['frozen_scene'] = RELEASE['path']+'/scene.json'
    binding['frozen_inventory'] = RELEASE['path']+'/asset_file_hashes.json'
    result['design']['supply_layout_candidate']['release'] = deepcopy(RELEASE)
    result['design']['tool_layout_revision'] = {
        'open_end_wrench_xy_m': list(OPEN_END_WRENCH_XY_M),
        'previous_open_end_wrench_xy_m': list(ORIGINAL_OPEN_END_WRENCH_XY_M),
        'reason': 'At x 0.80 the wrench pickup approach/precontact/lift anchors lay 14.6/36.6/28.5 mm beyond '
                  'the orientation-free fingertip reach of the original single-arm FR3.',
        'unchanged': 'Assets, orientation, 0.2 mm table gap, the other five objects, robot, room, desktop '
                     'dressing, rules, referee and physics profile.',
        'release_validation': deepcopy(RELEASE_VALIDATION)}
    result['qualification']['scope'] = ('Larger idler grasping hub and the open-end wrench 50 mm closer to the robot, '
                                        'with unchanged task rules; no inherited manipulation or transmission qualification.')
    return result
