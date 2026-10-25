"""Robot contact-material opt-ins: collider binding and compliant SHARPA pads (CPU)."""
import json
import math
from pathlib import Path

import pytest

from bench.isaacsim.release import task_inputs, validate
from bench.paths import DATA_ROOT, load_json
from scenes.manipulation.pad_compliance_release import PAD_COMPLIANCE, compliant_profile, prepare
from bench.simulation.isaacsim_backend.robot_materials import material_kind, material_options, runtime_pad_check

V12 = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v12_mount_v2_r1')
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


def test_compliant_profile_refuses_double_application_and_missing_materials():
    profile = load_json(V12 / 'bimanual/physics_profile.json')
    once = compliant_profile(profile)
    with pytest.raises(ValueError, match='already'):
        compliant_profile(once)
    bare = json.loads(json.dumps(profile))
    bare['isaacsim_options'].pop('robot_contact_materials')
    with pytest.raises(ValueError, match='robot_contact_materials'):
        compliant_profile(bare)


@pytest.mark.parametrize('layer', ['sharpa_w01_right', 'bimanual'])
def test_release_changes_only_the_physics_profile(layer, tmp_path):
    source = V12 / layer
    output = tmp_path / layer
    manifest = prepare(source, output)
    assert validate(output)['status'] == 'prepared_unvalidated' and manifest['robot_pad_compliance'] == PAD_COMPLIANCE
    old = load_json(source / 'release_manifest.json')
    for relative, item in old['files'].items():
        assert (output / relative).read_bytes() == (source / relative).read_bytes(), relative
    assert set(manifest['files']) == set(old['files']) | {'pad_compliance_provenance.json'}
    before = load_json(source / 'physics_profile.json')
    after = load_json(output / 'physics_profile.json')
    added = {k: v for k, v in after['isaacsim_options'].items() if k not in before['isaacsim_options']}
    assert added == {'robot_material_binding': 'collider_instance_v1', 'robot_pad_compliance': PAD_COMPLIANCE}
    assert {k: after['isaacsim_options'][k] for k in before['isaacsim_options']} == before['isaacsim_options']
    assert (output / 'source_release_manifest.json').read_bytes() == (source / 'release_manifest.json').read_bytes()
    provenance = load_json(output / 'pad_compliance_provenance.json')
    assert provenance['calibrated'] is False and provenance['physics_accepted'] is False and provenance['evidence']
    for entry in manifest['tasks']:
        _, scene, _, _ = task_inputs(output, entry['id'])
        assert material_options(scene['isaacsim'])['pad_compliance']['stiffness'] == PAD_COMPLIANCE['stiffness']
    with pytest.raises(FileExistsError):
        prepare(source, output)
