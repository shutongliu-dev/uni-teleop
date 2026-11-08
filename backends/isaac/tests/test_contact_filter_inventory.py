"""CPU inventory/column checks; no native solver or contact qualification."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.env import (
    FR3DexterousIsaacEnv, contact_view_paths, contact_view_patterns)


def scene_info():
    return {'robot_link_paths': {'arm': '/World/Robot/arm'},
            'objects': {'block': {'path': '/World/Objects/block'}},
            'support_paths': ['/World/Room/table', '/World/Fixtures/stand', '/World/Fixtures/tray'],
            'fixtures': {'stand': {'collision_paths': ['/World/Fixtures/stand']},
                         'tray': {'collision_paths': ['/World/Fixtures/tray/shape'],
                                  'contact_filter_paths': ['/World/Fixtures/tray']}}}


def test_fixed_mesh_support_and_fixture_aliases_make_one_native_column_each():
    scene = scene_info(); before = deepcopy(scene)
    sensors, filters = contact_view_patterns(scene)
    assert sensors == ['/World/Robot/arm', '/World/Objects/block']
    assert filters == sensors + scene['support_paths']
    assert scene == before


def test_distinct_fixture_filters_and_legacy_supports_remain_in_original_order():
    scene = scene_info(); del scene['support_paths']
    sensors, filters = contact_view_patterns(scene)
    assert filters == sensors + ['/World/Table', '/World/Ground', '/World/Fixtures/stand', '/World/Fixtures/tray']


def test_duplicate_sensor_identity_is_rejected_not_deduplicated():
    scene = scene_info(); scene['objects']['block']['path'] = '/World/Robot/arm'
    with pytest.raises(RuntimeError, match='sensors contain duplicate'):
        contact_view_patterns(scene)


@pytest.mark.parametrize('flat', [True, False])
def test_actual_flat_and_per_sensor_column_mappings_are_preserved(flat):
    sensors, filters = contact_view_patterns(scene_info())
    rows = [list(filters) for _ in sensors]
    view = SimpleNamespace(sensor_paths=sensors, filter_paths=filters if flat else rows,
                           sensor_count=len(sensors), filter_count=len(filters))
    assert contact_view_paths(view) == (sensors, rows)


@pytest.mark.parametrize('fault', ['sensor_alias', 'filter_alias', 'sensor_count', 'filter_count', 'row_count'])
def test_native_mapping_must_be_unambiguous_and_match_tensor_dimensions(fault):
    sensors, filters = contact_view_patterns(scene_info())
    view = SimpleNamespace(sensor_paths=sensors, filter_paths=[list(filters) for _ in sensors],
                           sensor_count=len(sensors), filter_count=len(filters))
    if fault == 'sensor_alias': view.sensor_paths[1] = view.sensor_paths[0]
    elif fault == 'filter_alias': view.filter_paths[0][-1] = view.filter_paths[0][-2]
    elif fault == 'sensor_count': view.sensor_count += 1
    elif fault == 'filter_count': view.filter_count += 1
    elif fault == 'row_count': view.filter_paths.pop()
    with pytest.raises(RuntimeError, match='duplicate actor|tensor dimensions'):
        contact_view_paths(view)


def test_legacy_projector_does_not_sum_duplicate_filter_columns():
    # Characterize the old first-unordered-pair behavior; different duplicate
    # buffers are discarded, not proved identical. This is not native evidence.
    env = object.__new__(FR3DexterousIsaacEnv)
    env._current_step_time = .0005
    env._contact_sensor_paths = ['/World/Robot/arm']
    env._contact_filter_paths = [['/World/Fixtures/tray', '/World/Fixtures/tray']]
    data = (np.array([1., 5.]), np.zeros((2, 3)), np.array([[0., 0., 1.]] * 2),
            np.zeros(2), np.array([[1, 1]]), np.array([[0, 1]]))
    env._rigid_contacts = SimpleNamespace(get_contact_data=lambda dt: data, max_contact_data_count=32)
    rows = env._project_rigid_contacts()
    assert len(rows) == 1 and rows[0]['point_count'] == 1
    assert rows[0]['impulse_magnitude_sum_ns'] == 1.
