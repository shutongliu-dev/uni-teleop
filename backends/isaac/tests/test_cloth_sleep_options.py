"""CPU authoring/audit tests; no native sleep or stability claim."""
from copy import deepcopy
import json

import numpy as np
import pytest

from bench.paths import REPO_ROOT
from bench.simulation.isaacsim_backend.cloth_sleep import (
    SETTLING_THRESHOLD_OPTION, SLEEP_DEFAULTS, author_surface_settling,
    read_surface_sleep_settings, surface_settling_properties,
)
from bench.simulation.isaacsim_backend.geometry import named_cloth_specs
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scripts.validation.audit_laundry_cloth import EvidenceUnknown, verify_solver_options
from scripts.validation.laundry_cloth_reset_probe import diagnostic_inputs, solver_readback


@pytest.fixture
def surface_stage():
    from pxr import Plug, Usd
    # The simulator is linked per checkout (runtime/README.md); some installs nest it in runtime/.
    schema = next((root/'extscache/omni.usd.schema.physx-110.3.2+110.3.0.lx64.r.cp312.u7f4/'
                   'plugins/PhysxSchemaAddition/resources'
                   for root in (REPO_ROOT/'runtime/isaacsim', REPO_ROOT/'runtime/isaacsim/runtime')
                   if (root/'extscache').is_dir()), None)
    if schema is None or not schema.is_dir():
        pytest.skip('Pinned codeless USD schema unavailable; no simulator is installed by this test')
    Plug.Registry().RegisterPlugins(str(schema))
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim('/Cloth', 'Mesh')
    assert prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
    stage.DefinePrim('/Material', 'Material')
    return stage, prim


def test_only_explicit_option_authors_a_stage_value_and_defaults_are_preserved(surface_stage):
    stage, prim = surface_stage
    before = stage.GetRootLayer().ExportToString()
    author_surface_settling(prim, {})
    assert stage.GetRootLayer().ExportToString() == before
    assert surface_settling_properties({}) == {}
    baseline = read_surface_sleep_settings(prim)
    for name, expected in SLEEP_DEFAULTS.items():
        assert baseline[name]['value'] == pytest.approx(expected)
        assert baseline[name]['authored'] is False
    author_surface_settling(prim, {SETTLING_THRESHOLD_OPTION: .20})
    result = read_surface_sleep_settings(prim)
    assert result['settlingThreshold']['value'] == pytest.approx(.20)
    assert result['settlingThreshold']['authored'] is True
    assert result['sleepThreshold'] == baseline['sleepThreshold']
    assert result['settlingDamping'] == baseline['settlingDamping']
    assert [p.GetName() for p in prim.GetAuthoredProperties()] == ['physxDeformableBody:settlingThreshold']


@pytest.mark.parametrize('value', [-1., 0., .05, True, '.20', float('inf'), float('nan')])
def test_invalid_settling_option_is_rejected_before_startup(value):
    scene = build_task_inputs('towel_folding')['scene']
    scene['isaacsim'][SETTLING_THRESHOLD_OPTION] = value
    with pytest.raises(ValueError, match=SETTLING_THRESHOLD_OPTION):
        named_cloth_specs(scene)


def test_unsupported_schema_is_not_silently_authored_as_custom_attribute():
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim('/Cloth', 'Mesh')
    with pytest.raises(RuntimeError, match='settling attribute unavailable'):
        author_surface_settling(prim, {SETTLING_THRESHOLD_OPTION: .20})
    assert not prim.GetAuthoredProperties()


def test_probe_requires_applied_value_and_unchanged_default_sleep_policy(surface_stage):
    stage, prim = surface_stage
    options = {SETTLING_THRESHOLD_OPTION: .20}
    infos = {'towel': {'path': '/Cloth', 'material_path': '/Material'}}
    with pytest.raises(RuntimeError, match=SETTLING_THRESHOLD_OPTION):
        solver_readback(stage, {'towel': {}}, options, infos)
    author_surface_settling(prim, options)
    actual = solver_readback(stage, {'towel': {}}, options, infos)['towel']
    assert actual['physxDeformableBody:settlingThreshold'] == pytest.approx(.20)
    assert actual['usd_sleep_settings']['sleepThreshold']['authored'] is False
    prim.GetAttribute('physxDeformableBody:sleepThreshold').Set(.20)
    with pytest.raises(EvidenceUnknown, match='sleepThreshold'):
        solver_readback(stage, {'towel': {}}, options, infos)


def test_diagnostic_override_changes_only_requested_option_without_release_edits(tmp_path):
    inputs = build_task_inputs('towel_folding')
    before = deepcopy(inputs)
    path = tmp_path/'settling.json'
    path.write_text(json.dumps({SETTLING_THRESHOLD_OPTION: .20}))
    scene, evidence = diagnostic_inputs(inputs['scene'], inputs['task'], physics_options=path)
    assert scene['isaacsim'].pop(SETTLING_THRESHOLD_OPTION) == .20
    assert scene['cloths'][0]['isaacsim'].pop(SETTLING_THRESHOLD_OPTION) == .20
    assert scene == before['scene'] and inputs == before
    assert evidence['physics_overrides']['options'] == {SETTLING_THRESHOLD_OPTION: .20}


def audit_fixture():
    scene = {'isaacsim': {SETTLING_THRESHOLD_OPTION: .20}, 'cloths': [{'id': 'towel'}]}
    native = {'solver_attributes': {
        'physxDeformableBody:solverPositionIterationCount': 32,
        'physxDeformableBody:linearDamping': .10,
        'physxDeformableBody:settlingThreshold': float(np.float32(.20))},
        'usd_sleep_settings': {name: {
            'value': float(np.float32(.20 if name == 'settlingThreshold' else default)),
            'authored': name == 'settlingThreshold'} for name, default in SLEEP_DEFAULTS.items()}}
    return scene, native


def test_formal_audit_requires_value_and_authored_receipts_and_respects_cloth_override():
    scene, native = audit_fixture()
    verify_solver_options(scene, 'towel', native)
    scene['cloths'][0]['isaacsim'] = {SETTLING_THRESHOLD_OPTION: .15}
    with pytest.raises(EvidenceUnknown, match='settlingThreshold'):
        verify_solver_options(scene, 'towel', native)
    native['usd_sleep_settings']['settlingThreshold']['value'] = .15
    native['solver_attributes']['physxDeformableBody:settlingThreshold'] = .15
    verify_solver_options(scene, 'towel', native)


@pytest.mark.parametrize('mutate', [
    lambda n: n.pop('usd_sleep_settings'),
    lambda n: n.pop('solver_attributes'),
    lambda n: n['solver_attributes'].pop('physxDeformableBody:settlingThreshold'),
    lambda n: n['solver_attributes'].update({'physxDeformableBody:settlingThreshold': .10}),
    lambda n: n['usd_sleep_settings']['settlingThreshold'].update(value=.10),
    lambda n: n['usd_sleep_settings']['settlingThreshold'].update(authored=False),
    lambda n: n['usd_sleep_settings']['sleepThreshold'].update(value=.20),
    lambda n: n['usd_sleep_settings']['sleepThreshold'].update(authored=True),
    lambda n: n['usd_sleep_settings']['settlingDamping'].update(value=20.),
    lambda n: n['usd_sleep_settings']['settlingDamping'].update(authored=True),
])
def test_formal_audit_rejects_missing_ignored_or_confounded_settling_option(mutate):
    scene, native = audit_fixture()
    mutate(native)
    with pytest.raises(EvidenceUnknown):
        verify_solver_options(scene, 'towel', native)


def test_legacy_audit_without_new_option_or_receipts_remains_compatible():
    scene = build_task_inputs('towel_folding')['scene']
    verify_solver_options(scene, 'towel', {})
