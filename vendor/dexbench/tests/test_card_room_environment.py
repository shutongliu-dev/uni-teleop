"""CPU environment contract checks; these are not physics qualification."""
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdLux

from scripts.assets.build_card_room import audit_environment
from bench.simulation.assets.workrooms.card_room import build_card_room
from bench.simulation.assets.workrooms.common import apply_lighting_profile

ASSETS = Path(f'{DATA_ROOT}/assets')


@pytest.fixture(scope='module')
def environment(tmp_path_factory):
    if not (ASSETS / 'objects/polyhaven/derived/v2/chinese_commode/chinese_commode.usdc').exists():
        pytest.skip('Versioned shared furniture assets are not installed')
    stage = Usd.Stage.CreateInMemory()
    scene = build_card_room(stage, ASSETS, tmp_path_factory.mktemp('card_room'))
    return stage, scene


def test_empty_table_and_opposing_station_volume(environment):
    stage, scene = environment
    assert audit_environment(stage)['status'] == 'passed'
    assert scene['objects'] == []
    assert scene['robot_placeholder']['contains_robot'] is False
    assert not stage.GetPrimAtPath('/World/Geometry/robot_column')
    assert not stage.GetPrimAtPath('/World/Geometry/go_board')


def test_room_lights_restore_nominal_and_keep_task_neutral(environment):
    stage, scene = environment
    nominal = scene['lighting']['lights']
    assert {x['role'] for x in nominal} == {'task', 'ceiling', 'accent', 'sun', 'sky'}
    tasks = [x for x in nominal if x['role'] == 'task']
    assert len(tasks) == 2
    assert all(x['temperature_k'] == 4200 for x in tasks)
    apply_lighting_profile(stage, 'card_room', 'low_light')
    restored = apply_lighting_profile(stage, 'card_room', 'nominal')
    assert restored['lights'] == nominal
    for entry in nominal:
        assert UsdLux.LightAPI(stage.GetPrimAtPath(entry['path'])).GetIntensityAttr().Get() > 0


def _bounds(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(box.GetMin()), np.array(box.GetMax())


def _overlaps(a, b):
    return (np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1.e-5).all()


def test_furnishings_clear_full_height_access_and_stay_inside_room(environment):
    stage, scene = environment
    access = (np.array([-.05, -1.10, 0.]), np.array([1.35, 1.10, 1.95]))
    room = (np.array([-1.80, -2.20, 0.]), np.array([3.10, 2.50, 2.85]))
    groups = scene['furnishings']['groups']
    assert len(groups) == 5
    for group in groups:
        actual = [_bounds(stage, path) for path in group['geometry_paths']]
        union = np.array([np.min([box[0] for box in actual], axis=0),
                          np.max([box[1] for box in actual], axis=0)])
        np.testing.assert_allclose(group['bounds_m'], union, atol=1.e-8)
        assert (union[0] >= room[0] - 1.e-6).all(), group['name']
        assert (union[1] <= room[1] + 1.e-6).all(), group['name']
        assert not _overlaps(union, access), group['name']


def test_new_furniture_clears_existing_seating_and_open_window(environment):
    stage, scene = environment
    existing = [_bounds(stage, '/World/Geometry/' + name) for name in (
        'tea_left_chair', 'tea_right_chair', 'scholar_console_top', 'tea_chinese_commode')]
    aperture = (np.array([-.17, 2.40, 1.01]), np.array([1.47, 2.64, 2.65]))
    for group in scene['furnishings']['groups']:
        union = np.asarray(group['bounds_m'])
        assert not _overlaps(union, aperture), group['name']
        for furniture in existing:
            assert not _overlaps(union, furniture), group['name']
        if group['name'] != 'game_record':
            assert union[0, 2] == pytest.approx(0., abs=1.e-6)


@pytest.mark.parametrize(('item', 'support'), [
    ('game_library_manual_0_cover_0', 'game_library_shelf_1'),
    ('game_library_go_bowl_0_body', 'game_library_shelf_1'),
    ('game_library_go_bowl_1_body', 'game_library_shelf_1'),
    ('game_library_vase', 'game_library_shelf_2'),
    ('game_library_scroll_paper_0', 'game_library_scroll_cradle_0'),
    ('bonsai_planter', 'bonsai_stand_top'),
])
def test_display_objects_rest_on_their_modeled_supports(environment, item, support):
    stage, _ = environment
    item_box = _bounds(stage, '/World/Geometry/' + item)
    support_box = _bounds(stage, '/World/Geometry/' + support)
    assert item_box[0][2] == pytest.approx(support_box[1][2], abs=1.e-6)
    assert (np.minimum(item_box[1][:2], support_box[1][:2])
            - np.maximum(item_box[0][:2], support_box[0][:2]) > 0.).all()
