"""Physical L adapters participate in complete, identity-free contact history."""
import json

import pytest

from bench.sensors.contact_history import ContactHistory, configure_contact_history
from bench.sensors.contact_schema import parse_contact_history
from test_contact_history import metadata, contact
from test_bimanual_contact_history import dual_metadata, sampled_window


@pytest.fixture(params=('single', 'dual', 'instances'))
def mounted_metadata(request, metadata, dual_metadata):
    kind = request.param
    value = metadata if kind == 'single' else dual_metadata
    sides = ('right',) if kind == 'single' else ('left', 'right')
    robot = 'sharpa_w01_right' if kind == 'single' else 'dual_fr3_sharpa_w01'
    if kind == 'instances':
        encoded = json.dumps(value).replace('Left', 'front').replace('Right', 'rear')
        value = json.loads(encoded.replace('left_', 'front_').replace('right_', 'rear_'))
        sides, robot = ('front', 'rear'), 'fr3_sharpa_instances'
        value['hand_spec'] = {'hand_id': robot, 'arms': {s: {} for s in sides}}
    links = value['scene_info']['robot_link_paths']
    sensors = value['contact_evidence']['sensor_paths']
    frames, adapters = {}, {}
    for side in sides:
        old = side + '_hand_flange'
        if old in links:
            sensors.remove(links.pop(old))
            value['body_names'].remove(old)
        scope = '' if kind == 'single' else (side.title() if kind == 'dual' else side) + '/'
        root = '/World/Robot/' + scope + 'private_robot_canary/'
        name = 'fr3_l_adapter' if kind == 'single' else 'fr3_' + side + '_l_adapter'
        path = root + name
        links[name] = path
        value['body_names'].append(name)
        sensors.append(path)
        adapters[side] = name
        frames[old] = root + old
    value['scene_info']['robot_frame_paths'] = frames
    if kind != 'single':
        value['contact_evidence']['filter_paths'] = [list(sensors) for _ in sensors]
    return value, robot, adapters


def test_adapter_impulses_are_reported_for_the_correct_palm(mounted_metadata):
    value, robot, adapters = mounted_metadata
    config = configure_contact_history(value, robot)
    assert config['available']
    frames = value['scene_info']['robot_frame_paths']
    assert not set(frames.values()) & set(config['regions_by_path'])
    history = ContactHistory(config)
    history.observe(sampled_window(rows=[
        contact(config, link, tick=i + 1, impulse=.001 * (i + 1))
        for i, link in enumerate(adapters.values())]))
    result = history.snapshot()
    assert result['status'] == 'available'
    assert parse_contact_history(result).model_dump(mode='json') == result
    for i, side in enumerate(adapters):
        hand = result if len(adapters) == 1 else result['hands'][side]
        assert hand['regions']['palm_mount']['normal_impulse_sum_ns'] == pytest.approx(.001 * (i + 1))
        assert hand['regions']['palm_mount']['contact_substeps'] == 1
        assert hand['regions']['arm']['contact_substeps'] == 0
    assert '/World/' not in json.dumps(result)


@pytest.mark.parametrize('damage', ('unknown_adapter', 'missing_sensor', 'duplicate_sensor', 'foreign_scope'))
def test_adapter_mapping_keeps_inventory_and_path_checks(mounted_metadata, damage):
    value, robot, adapters = mounted_metadata
    name = next(iter(adapters.values()))
    links = value['scene_info']['robot_link_paths']
    sensors = value['contact_evidence']['sensor_paths']
    path = links[name]
    if damage == 'missing_sensor':
        sensors.remove(path)
    elif damage == 'duplicate_sensor':
        sensors.append(path)
    else:
        new_name = name + '_unknown' if damage == 'unknown_adapter' else name
        new_path = path + '_unknown' if damage == 'unknown_adapter' else '/World/Foreign/' + name
        links.pop(name)
        links[new_name] = new_path
        value['body_names'][value['body_names'].index(name)] = new_name
        sensors[sensors.index(path)] = new_path
        if 'filter_paths' in value['contact_evidence']:
            value['contact_evidence']['filter_paths'] = [list(sensors) for _ in sensors]
    assert configure_contact_history(value, robot)['available'] is False
