from copy import deepcopy

import pytest

from scripts.validation.laundry_capture_settings import DEFAULTS, phase_capture_settings


def test_loaded_capture_is_explicit_and_bound_to_stopped_original_command():
    loaded={'capture_profile':'loaded_adjacent_patch_v5','contact_closure_stop':True,
        'contact_closure_target_policy':'hold_command_v2',
        'contact_closure_geometry_profile':'adjacent_patch_v3'}
    settings=phase_capture_settings(loaded)
    assert settings==loaded
    options={'phases':{'1':{'capture_settings':loaded}}}
    assert phase_capture_settings(options,0)==phase_capture_settings({})
    assert phase_capture_settings(options,1)==loaded
    for change in ({'contact_closure_stop':False,'contact_closure_target_policy':'reduce_preload_v1'},
                   {'contact_closure_target_policy':'reduce_preload_v1'},
                   {'contact_closure_geometry_profile':'shared_edge_v1'},
                   {'contact_closure_geometry_profile':'angled_adjacent_patch_v4'},
                   {'attachment_material_mode':'shared_edge'}):
        with pytest.raises(ValueError,match='Loaded capture'):
            phase_capture_settings({**loaded,**change})


def test_regrasp_preserves_the_qualified_first_operation_and_changes_only_later_capture():
    later = dict(capture_profile='adjacent_patch_v3', contact_closure_stop=True,
                 contact_closure_target_policy='hold_command_v2',
                 contact_closure_geometry_profile='adjacent_patch_v3')
    options = {'attachment_material_mode': 'captured_patch',
               'phases': {'1': {'sides': ['left'], 'capture_settings': later}}}
    before = deepcopy(options)
    assert phase_capture_settings(options, 0) == DEFAULTS
    assert phase_capture_settings(options, 1) == later
    assert phase_capture_settings(options, 2) == DEFAULTS
    assert options == before
    result = phase_capture_settings(options, 1)
    result['capture_profile'] = 'not-a-profile'
    assert options == before


def test_global_settings_are_preserved_without_a_phase_override():
    options = {'capture_profile': 'local_assist_v2', 'contact_closure_stop': True,
               'contact_closure_target_policy': 'hold_command_v2'}
    expected = {**DEFAULTS, **options}
    assert phase_capture_settings(options) == expected
    assert all(phase_capture_settings(options, i) == expected for i in range(3))


@pytest.mark.parametrize('override', [
    None, [], {'unknown': 1}, {'capture_profile': 'unbounded'},
    {'contact_closure_stop': 1}, {'contact_closure_target_policy': 'hold_command_v2'},
    {'contact_closure_stop': True, 'contact_closure_geometry_profile': 'adjacent_patch_v3'},
    {'contact_closure_geometry_profile': 'all_material'},
])
def test_invalid_phase_controls_fail_before_a_native_episode(override):
    with pytest.raises(ValueError):
        phase_capture_settings({'phases': {'1': {'capture_settings': override}}}, 1)


def test_angled_profile_remains_explicit_and_requires_matching_local_geometry():
    settings = {'capture_profile': 'angled_adjacent_patch_v4',
                'contact_closure_stop': True,
                'contact_closure_target_policy': 'hold_command_v2',
                'contact_closure_geometry_profile': 'angled_adjacent_patch_v4'}
    options = {'phases': {'1': {'capture_settings': settings}}}
    assert phase_capture_settings(options, 0) == DEFAULTS
    assert phase_capture_settings(options, 1) == settings
    with pytest.raises(ValueError):
        phase_capture_settings({**options, 'attachment_material_mode': 'shared_edge'}, 1)
    with pytest.raises(ValueError):
        phase_capture_settings({**settings, 'contact_closure_geometry_profile': 'shared_edge_v1'})
    no_stop = {'capture_profile': 'angled_adjacent_patch_v4'}
    assert phase_capture_settings(no_stop) == {**DEFAULTS, **no_stop}
    with pytest.raises(ValueError):
        phase_capture_settings({**no_stop, 'attachment_material_mode': 'shared_edge'})


def test_explicit_pivot_keeps_full_gate_and_matching_stop_but_never_loaded_v5():
    for profile in ('adjacent_patch_v3', 'angled_adjacent_patch_v4'):
        options = {'attachment_material_mode': 'captured_pivot_v1', 'capture_profile': profile}
        assert phase_capture_settings(options)['capture_profile'] == profile
        options.update(contact_closure_stop=True, contact_closure_target_policy='hold_command_v2',
                       contact_closure_geometry_profile=profile)
        assert phase_capture_settings(options)['contact_closure_geometry_profile'] == profile
    options.update(capture_profile='loaded_adjacent_patch_v5',
                   contact_closure_geometry_profile='adjacent_patch_v3')
    with pytest.raises(ValueError, match='Loaded capture'):
        phase_capture_settings(options)
    with pytest.raises(ValueError, match='Unknown local assistance'):
        phase_capture_settings({'attachment_material_mode': 'remote_point'})
