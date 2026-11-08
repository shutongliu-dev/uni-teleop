from copy import deepcopy
import builtins
import json
from types import SimpleNamespace

import pytest

from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation import laundry_captured_garment_fold as garment
from scripts.validation import laundry_captured_towel_tasks as towel
from scripts.validation import laundry_captured_trousers_regrasp as regrasp


def hand_options(setting, *, side='left', phase='0'):
    return {'phases': {phase: {'hands': {side: setting}}}}


def test_existing_hand_controls_and_provenance_are_preserved_without_rewriting():
    setting = {'yaw_deg': 135., 'pitch_deg': 60., 'bottom_z_m': .9003,
               'open_distance_m': .025, 'close_distance_m': .010,
               'closure_lift_m': 0., 'closure_lift_start_fraction': 0.,
               'capture_rise_m': 0., 'hover_offset_m': [0., .04, .12],
               'hover_arm_joint_positions_rad': {'fr3_left_joint1': .2},
               'pinch_inward_offset_m': -.003, 'closure_inward_travel_m': .009}
    options = hand_options(setting)
    options.update(schema='candidate_evidence.v1', evidence={'sha256': 'unchanged'},
                   capture_profile='angled_adjacent_patch_v4')
    options['phases']['0'].update(edge_indices=[0], sides=['left'], capture_order=['left'],
                                 capture_settings={'contact_closure_stop': False})
    before = deepcopy(options)
    for is_towel in (False, True):
        assert validate_hand_options(options, towel=is_towel) is None
    assert options == before
    assert validate_hand_options({}) is None


@pytest.mark.parametrize('setting', [
    {'schema': 'hover_options.v1', 'evidence': {}, 'setting': {'hover_offset_m': [0., 0., .15]}},
    {'hover_offest_m': [0., 0., .15]},
    {'capture_settings': {'contact_closure_stop': True}},
])
def test_wrappers_typos_and_misplaced_capture_settings_are_rejected_with_location(setting):
    with pytest.raises(ValueError, match=r'Unknown hand options at options\.phases\.2\.hands\.right'):
        validate_hand_options(hand_options(setting, side='right', phase='2'))


@pytest.mark.parametrize('value', [
    None, [], {'phases': []}, {'phases': {'0': None}},
    {'phases': {'0': {'hands': []}}}, hand_options(None),
    hand_options({}, side='rigth'), hand_options({}, phase='01'),
    hand_options({}, phase=-1),
])
def test_malformed_hand_option_containers_fail_explicitly(value):
    with pytest.raises(ValueError):
        validate_hand_options(value)


def test_relative_yaw_is_only_accepted_by_the_towel_adapter_that_uses_it():
    options = hand_options({'yaw_offset_deg': -15.})
    assert validate_hand_options(options, towel=True) is None
    with pytest.raises(ValueError, match='yaw_offset_deg'):
        validate_hand_options(options)


def test_local_panel_tangent_is_explicit_garment_entry_only():
    setting={'pinch_tangent_profile':'local_panel_20mm_v2','pinch_inward_offset_m':-.003}
    options=hand_options(setting,phase='2');before=deepcopy(options)
    validate_hand_options(options)
    assert options==before
    with pytest.raises(ValueError,match='Unknown hand options'):
        validate_hand_options(options,towel=True)
    for bad in ({'pinch_tangent_profile':'invented'}, {'pinch_tangent_profile':True},
                {'pinch_tangent_profile':'local_panel_20mm_v2'}):
        with pytest.raises(ValueError):validate_hand_options(hand_options(bad,phase='2'))


@pytest.mark.parametrize('adapter', [garment, towel, regrasp])
def test_cpu_plan_checks_even_later_unselected_hands_before_geometry(adapter):
    options = hand_options({'hover_offest_m': [0., 0., .15]}, side='left', phase='2')
    # Invalid task/q would fail later: this proves the complete options scan
    # precedes both geometry and only_sides filtering.
    with pytest.raises(ValueError, match=r'options\.phases\.2\.hands\.left'):
        adapter.prepare_phase(None, {}, {}, None, 0, options, only_sides=['right'])


@pytest.mark.parametrize('adapter,invalid_setting', [
    (None, {'schema': 'hover.v1', 'setting': {}, 'evidence': {}}),
    (towel, {'schema': 'hover.v1', 'setting': {}, 'evidence': {}}),
    (regrasp, {'yaw_offset_deg': -15.}),
])
def test_native_rejects_unused_hand_settings_before_backend_import_or_output(tmp_path, monkeypatch, adapter, invalid_setting):
    import scripts.maintenance.freeze_project_sources as freeze
    import bench.isaacsim.release as release_module

    task_id = ('shirt_folding' if adapter is None else
               'towel_folding' if adapter is towel else 'trousers_folding')
    release = tmp_path / 'release'
    release.mkdir()
    rows = {'task.json': {'id': task_id},
            'scene.json': {'named_cloth_initialization': 'author-only',
                           'cloths': [{'id': 'towel'}],
                           'robot': {'hand_id': 'dual_fr3_sharpa_w01'}},
            'initial_state.json': {}}
    if adapter is regrasp:
        rows['task.json']['asset_metadata'] = {
            'design': {'fold_sequence_axis_hinge_sign_radius_m': [[0, 0., -1, .007], [1, 0., -1, .007]]}}
    for name, row in rows.items():
        (release / name).write_text(json.dumps(row))
    options = tmp_path / 'options.json'
    options.write_text(json.dumps(hand_options(invalid_setting, phase='2')))
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True})
    monkeypatch.setattr(release_module, 'validate', lambda _: {'task_id': task_id})
    original_import = builtins.__import__
    attempted_backend_imports = []

    def guarded_import(name, *args, **kwargs):
        if name == 'bench.simulation.isaacsim_backend.env':
            attempted_backend_imports.append(name)
            raise AssertionError('Invalid options must not initialize a simulator')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    output = tmp_path / 'native_output'
    args = SimpleNamespace(release=release, options=options, output=output, max_phases=1)
    with pytest.raises(ValueError, match=r'Unknown hand options at options\.phases\.2\.hands\.left'):
        garment.native(args, task_adapter=adapter)
    assert not attempted_backend_imports
    assert not output.exists()
