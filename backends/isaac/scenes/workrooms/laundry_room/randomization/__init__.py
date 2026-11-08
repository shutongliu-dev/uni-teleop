"""Finite, immutable laundry placement and render-only appearance inputs."""

TASK_IDS = ('towel_flattening', 'towel_folding', 'towel_stacking',
            'shirt_folding', 'trousers_folding')
CONFIGURATION_IDS = ('single_arm', 'dual_same_side', 'dual_opposed')
SEEDS = tuple(range(10))
SCHEMA = 'laundry_randomized_release.v1'
VERSION = 'isaacsim-laundry-randomized-release-v1'
GENERATOR_VERSION = 'laundry_xy_fabric_seed_v1'
SHIRT_NEAR_GENERATOR_VERSION = 'laundry_shirt_near_robot_seed_v2'
PARAMETERIZED_VERSION = 'parameterized_physics_v1'
ASSET_SCHEMA = 'laundry_seed_asset.v1'
RESET_SCHEMA = 'laundry_reset_case.v1'


def selection(task_id, seed, configuration_id):
    if task_id not in TASK_IDS or configuration_id not in CONFIGURATION_IDS:
        raise ValueError('Unknown laundry task or original robot installation')
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Laundry reset seed must be an integer from 0 through 9')
    return seed


def checked_generator_version(task_id, configuration_id, version):
    if version in (GENERATOR_VERSION, PARAMETERIZED_VERSION):
        return version
    if (version == SHIRT_NEAR_GENERATOR_VERSION and task_id == 'shirt_folding'
            and configuration_id in ('single_arm', 'dual_same_side')):
        return version
    raise ValueError('Unknown laundry generator or unsupported task/installation for its version')
