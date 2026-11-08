"""Selected menu entries must retain their own paired physical-instance identity."""
from copy import deepcopy

import pytest

from scenes.workrooms.robot_configurations import configured_definition


@pytest.mark.parametrize('profile', ['original', 'l_mount'])
@pytest.mark.parametrize('selected', ['dual_same_side', 'dual_opposed'])
def test_selected_dual_menu_cannot_keep_the_default_single_pair(profile, selected):
    options = {}
    for number, configuration in enumerate(('single_arm', 'dual_same_side', 'dual_opposed'), 1):
        options[configuration] = {
            'source_release': {'path': '/releases/' + profile + '/' + configuration,
                               'manifest_sha256': str(number) * 64},
            'robot_profile': profile,
            'physical_instance_pair': {'path': '/pairs/' + configuration + '.json',
                                       'sha256': str(number + 3) * 64},
        }
    definition = {'family': 'study', 'task_id': 'book_retrieval', 'layout': {},
                  'default_configuration_id': 'single_arm', 'robot_configurations': options,
                  **deepcopy(options['single_arm'])}
    before = deepcopy(definition)
    result = configured_definition(definition, selected)
    assert result['configuration_id'] == selected
    assert result['source_release'] == options[selected]['source_release']
    assert result['robot_profile'] == profile
    assert result['physical_instance_pair'] == options[selected]['physical_instance_pair']
    assert result['physical_instance_pair'] != options['single_arm']['physical_instance_pair']
    result['physical_instance_pair']['path'] = '/changed-result-only'
    assert definition == before


def test_legacy_configuration_without_pair_fields_remains_unchanged():
    definition = {'family': 'study', 'layout': {}, 'source_release': {'path': '/single'},
                  'robot_configurations': {'dual_same_side': {'source_release': {'path': '/dual'}}}}
    result = configured_definition(definition, 'dual_same_side')
    assert result['source_release'] == {'path': '/dual'}
    assert 'robot_profile' not in result and 'physical_instance_pair' not in result
    with pytest.raises(ValueError, match='does not publish'):
        configured_definition(definition, 'unpublished')
