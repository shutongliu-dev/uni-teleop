"""Controller observation boundary and robot-only command checks."""
from copy import deepcopy
from pathlib import Path

from bench.paths import DATA_ROOT, load_json
from scenes.manipulation.controller import SimpleTaskController


def scene():
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1')
    value = load_json(root/'tasks/01_ordered_kitting/scene.json')
    value['objects'] = [{'id':'cube','position_m':[.45,.10,.781]}]
    return value


def test_only_permitted_observation_fields_and_named_joint_commands():
    controller = SimpleTaskController(scene(), {'id':'03_cube_pick_place','target_position_m':[.52,.20,.78]})
    q = {name:0. for name in controller.limits}
    q.update(controller.arm_seed)
    observation = {'sim_time_s':0., 'joint_positions_rad':q,
                   'objects':{'cube':{'position_m':[.45,.10,.781]}}}
    saved = deepcopy(observation)
    action = controller.next_action(observation)
    assert observation == saved
    assert set(action) == {'joint_positions_rad','duration_s'}
    assert len(action['joint_positions_rad']) == 29
    assert action['duration_s'] == .02
    assert all(controller.limits[n][0] <= v <= controller.limits[n][1]
               for n,v in action['joint_positions_rad'].items())
    assert not controller.diagnostic()['object_state_writeback']
    assert not controller.diagnostic()['trained_policy']


def test_hold_control_preserves_first_joint_target_without_contact_inputs():
    controller = SimpleTaskController(scene(), {'id':'01_cube_push'}, variant='hold')
    q = {name:0. for name in controller.limits}
    q.update(controller.arm_seed)
    observation = {'time_s':0.,'joint_positions_rad':q,
                   'objects':{'cube':{'position_m':[.45,.10,.781]}}}
    first = controller.next_action(observation)
    observation['time_s'] = 1.
    observation['joint_positions_rad'] = {n:v+.01 for n,v in q.items()}
    assert controller.next_action(observation) == first
