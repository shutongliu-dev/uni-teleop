"""Current scene discovery can grow without extending the frozen diagnostics."""
import json

import pytest

from scenes.workrooms import catalog, registry


def test_seventh_room_preserves_fixed_environment_and_legacy_family_order():
    assert len(registry.FAMILIES) == 6
    assert catalog.CURRENT_FAMILIES == (*registry.FAMILIES, 'laundry_room')
    assert [row['family'] for row in catalog.list_scenes()] == list(catalog.CURRENT_FAMILIES)
    room = catalog.scene_definition('laundry_room')
    assert room['status'] == 'task_family'
    assert 'robot_installation' not in room
    assert room['environment']['family'] == 'laundry_room'
    assert room['environment']['version'] == 'laundry_residential_20261021_v3'
    assert room['coordinates']['table_top_z_m'] == .90
    assert room['coordinates']['clear_task_rectangle_xy_m'] == [-.10, 1.40, -.94, -.16]
    room['coordinates']['table_center_xy_m'][0] = 999
    assert catalog.scene_definition('laundry_room')['coordinates']['table_center_xy_m'] == [.65, -.55]


def test_all_five_laundry_definitions_use_dual_robot_episodes(monkeypatch):
    monkeypatch.setattr(catalog, 'verify_reference',
                        lambda *_: pytest.fail('Discovery must not open scene assets'))
    from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS as TASK_IDS
    rows = catalog.list_tasks('laundry_room', include_planned=True)
    assert [row['task_id'] for row in rows] == list(TASK_IDS)
    for row in rows:
        assert row['capabilities']['robot_episodes'] is True
        assert row['robot_requirements']['active_sides'] == ['left', 'right']
        assert row['robot_requirements']['articulations'] == 2
    assert catalog.list_tasks() == [task for family in catalog.CURRENT_FAMILIES
                                    for task in catalog.list_tasks(family)]
    with pytest.raises(ValueError, match='Unknown task: laundry_room/folding'):
        catalog.load_task('laundry_room', 'folding')


def test_current_cli_lists_seven_rooms_and_five_laundry_definitions(capsys):
    registry.main(['scenes'])
    assert [row['family'] for row in json.loads(capsys.readouterr().out)] == list(catalog.CURRENT_FAMILIES)
    for extra in ([], ['--include-planned']):
        registry.main(['tasks', '--family', 'laundry_room', *extra])
        rows = json.loads(capsys.readouterr().out)
        expected = catalog.list_tasks('laundry_room', include_planned=bool(extra))
        assert [row['task_id'] for row in rows] == [row['task_id'] for row in expected]


def test_legacy_cli_keeps_six_room_list_and_rejects_laundry_export(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(registry, 'release_pointer', lambda: {})
    registry.main(['list'])
    result = json.loads(capsys.readouterr().out)
    assert result['version'] == 'rooms_20261017_v3'
    assert result['families'] == list(registry.FAMILIES)
    assert 'laundry_room' not in result['families']
    monkeypatch.setattr(registry, 'export_scene',
                        lambda *_args, **_kwargs: pytest.fail('Laundry has no legacy export'))
    with pytest.raises(SystemExit) as error:
        registry.main(['export', '--family', 'laundry_room', '--output', str(tmp_path / 'export')])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    assert not (tmp_path / 'export').exists()


def test_current_cli_rejects_missing_laundry_task_before_asset_loading(capsys, monkeypatch):
    monkeypatch.setattr(catalog, 'scene_definition',
                        lambda *_: pytest.fail('Missing task must stop before environment loading'))
    with pytest.raises(SystemExit) as error:
        registry.main(['task', '--family', 'laundry_room', '--task', 'folding'])
    assert error.value.code == 2
    assert 'Unknown task: laundry_room/folding' in capsys.readouterr().err
