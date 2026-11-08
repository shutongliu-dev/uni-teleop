"""Robot contact-material opt-ins: collider binding and compliant SHARPA pads (CPU)."""
import json
import math

import pytest

from bench.simulation.isaacsim_backend.robot_materials import material_kind, material_options, runtime_pad_check

MATERIALS = {'pad_friction': .5, 'shell_friction': .5}
COMPLIANT = {'robot_contact_materials': MATERIALS, 'robot_material_binding': 'collider_instance_v1',
             'robot_pad_compliance': {'model': 'force_spring_v1', 'stiffness': 5000., 'damping': 20.}}


def test_absent_keys_keep_the_legacy_binding():
    assert material_options({'robot_contact_materials': MATERIALS}) == {'binding': None, 'pad_compliance': None}
    assert material_options(None) == {'binding': None, 'pad_compliance': None}


def test_pad_compliance_cannot_silently_vanish():
    # The legacy link binding resolves the pads to robot_shell (runs/material_binding_01).
    legacy = {k: v for k, v in COMPLIANT.items() if k != 'robot_material_binding'}
    with pytest.raises(ValueError, match='collider_instance_v1'):
        material_options(legacy)
    with pytest.raises(ValueError, match='robot_contact_materials'):
        material_options({k: v for k, v in COMPLIANT.items() if k != 'robot_contact_materials'})
    with pytest.raises(ValueError, match='robot_material_binding'):
        material_options({'robot_material_binding': 'per_link'})


@pytest.mark.parametrize('change', [{'model': 'hydroelastic'}, {'stiffness': 0.}, {'stiffness': -1.}, {'stiffness': math.inf},
                                    {'damping': -1.}, {'damping': math.nan}, {'stiffness': True}, {'extra': 1}])
def test_pad_compliance_values_are_checked(change):
    options = json.loads(json.dumps(COMPLIANT))
    options['robot_pad_compliance'].update(change)
    with pytest.raises(ValueError):
        material_options(options)


def test_pad_compliance_is_normalised_with_units():
    result = material_options(COMPLIANT)
    assert result['binding'] == 'collider_instance_v1'
    assert result['pad_compliance'] == {'model': 'force_spring_v1', 'stiffness': 5000., 'damping': 20.,
                                        'units': {'stiffness': 'N/m per contact point', 'damping': 'N*s/m per contact point'}}


def test_only_elastomer_links_are_pads():
    assert material_kind('right_thumb_elastomer') == material_kind('left_pinky_elastomer') == 'robot_pad'
    assert {material_kind(n) for n in ('fr3_left_link7', 'right_index_DP', 'right_hand_C_MC', 'left_hand_flange')} == {'robot_shell'}


def test_runtime_check_requires_compliance_on_exactly_the_pads():
    robot = {'pad_compliance': material_options(COMPLIANT)['pad_compliance'],
             'resolved_collider_materials': {'/World/PhysicsMaterials/robot_pad': 2, '/World/PhysicsMaterials/robot_shell': 3}}
    rigid, pad = (.5, .5, 0.), (.5, .5, -5000.)
    assert runtime_pad_check([rigid] * 3 + [pad] * 2, robot)['compliant_shapes'] == 2
    with pytest.raises(RuntimeError, match='not live'):
        runtime_pad_check([rigid] * 5, robot)
    with pytest.raises(RuntimeError, match='not live'):
        runtime_pad_check([rigid] * 2 + [pad] * 3, robot)
    with pytest.raises(RuntimeError, match='stiffness'):
        runtime_pad_check([rigid] * 3 + [(.5, .5, -20000.)] * 2, robot)
    legacy = {'pad_compliance': None, 'resolved_collider_materials': {'/World/PhysicsMaterials/robot_shell': 5}}
    assert runtime_pad_check([rigid] * 5, legacy) == {'robot_shapes': 5, 'compliant_shapes': 0, 'compliance_requested': False}
    assert runtime_pad_check([rigid] * 5, None)['compliant_shapes'] == 0
    # Without a request the check only records (historical scenes must run unchanged).
    assert runtime_pad_check([rigid] * 4 + [pad], legacy)['compliant_shapes'] == 1
