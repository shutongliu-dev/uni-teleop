"""Named robot contact feedback preserves native sampling and privacy rules."""
from copy import deepcopy
import json

import pytest

from bench.sensors.contact_history import ContactHistory, configure_contact_history
from bench.sensors.contact_schema import parse_contact_history
from test_bimanual_contact_history import dual_metadata, sampled_window, path
from test_contact_history import contact


@pytest.fixture
def instance_config(dual_metadata):
    encoded = json.dumps(dual_metadata).replace('Left', 'front').replace('Right', 'rear')
    encoded = encoded.replace('left_', 'front_').replace('right_', 'rear_')
    metadata = json.loads(encoded)
    metadata['hand_spec'] = {'hand_id': 'fr3_sharpa_instances', 'arms': {'front': {}, 'rear': {}}}
    result = configure_contact_history(metadata, 'fr3_sharpa_instances')
    assert result['available'] and result['protocol'] == 'robot_rigid_contact_history_v3'
    return result


def test_named_hands_contact_partitions_and_preserves_private_paths(instance_config):
    config = instance_config
    history = ContactHistory(config)
    rows = [contact(config, 'front_thumb_elastomer', impulse=.002),
            contact(config, 'rear_index_elastomer', impulse=.003),
            contact(config, 'front_index_PP', tick=2, impulse=.004,
                    other=path(config, 'rear_index_PP'))]
    history.observe(sampled_window(rows=rows))
    value = history.snapshot()
    assert parse_contact_history(value).model_dump(mode='json') == value
    assert value['instance_names'] == ['front', 'rear']
    assert value['hands']['front']['regions']['thumb_pad']['normal_impulse_sum_ns'] == .002
    assert value['hands']['rear']['regions']['index_pad']['normal_impulse_sum_ns'] == .003
    assert value['cross_side_contact']['normal_impulse_sum_ns'] == .004
    assert 'private_' not in json.dumps(value)
    wrong = deepcopy(value)
    wrong['instance_names'] = ['left', 'right']
    with pytest.raises(ValueError):
        parse_contact_history(wrong)


def test_missing_native_sample_receipt_cannot_be_reported_as_zero(instance_config):
    history = ContactHistory(instance_config)
    value = sampled_window()
    value['contacts']['sampling']['complete'] = False
    history.observe(value)
    result = history.snapshot()
    assert result['status'] == 'invalid' and result['hands'] is None
    parse_contact_history(result)
