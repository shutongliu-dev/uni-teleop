"""Frozen damping priors and readback contracts; no native stability claim."""
import numpy as np
import pytest

from bench.simulation.isaacsim_backend.cloth_materials import (
    DAMPING_OPTIONS, surface_damping_properties,
)
from bench.simulation.isaacsim_backend.geometry import named_cloth_specs
from scripts.validation.audit_laundry_cloth import EvidenceUnknown, verify_solver_options
from scenes.workrooms.laundry_room.composition import build_task_inputs


def test_default_and_independent_damping_overrides_preserve_legacy_values():
    assert surface_damping_properties({}) == {attribute: .005 for attribute in DAMPING_OPTIONS.values()}
    assert surface_damping_properties({'cloth_bend_damping': .05}) == {
        'physxDeformableMaterial:elasticityDamping': .005,
        'physxDeformableMaterial:bendDamping': .05}
    # Coefficients are not bounded dimensionless critical-damping fractions.
    assert surface_damping_properties({'cloth_elasticity_damping': 2.})[
        'physxDeformableMaterial:elasticityDamping'] == 2.


@pytest.mark.parametrize('value', [-.1, float('inf'), float('nan'), True, '.05'])
@pytest.mark.parametrize('key', list(DAMPING_OPTIONS))
def test_invalid_damping_rejected_before_native_initialization(key, value):
    scene = build_task_inputs('towel_folding')['scene']
    scene['isaacsim'][key] = value
    with pytest.raises(ValueError, match=key):
        named_cloth_specs(scene)


def test_explicit_material_values_cannot_bypass_readback_with_no_filter_receipt():
    scene = build_task_inputs('towel_folding')['scene']
    scene['isaacsim'].update(cloth_elasticity_damping=.05, cloth_bend_damping=.05)
    with pytest.raises(EvidenceUnknown, match='material attributes'):
        verify_solver_options(scene, 'towel', {})
    native = {'material_attributes': {attribute: float(np.float32(.05))
                                     for attribute in DAMPING_OPTIONS.values()}}
    verify_solver_options(scene, 'towel', native)
    scene['cloths'][0]['isaacsim']['cloth_bend_damping'] = .08
    with pytest.raises(EvidenceUnknown, match='bendDamping'):
        verify_solver_options(scene, 'towel', native)
    native['material_attributes']['physxDeformableMaterial:bendDamping'] = float(np.float32(.08))
    verify_solver_options(scene, 'towel', native)


@pytest.mark.parametrize('bad', [None, True, -.05, float('nan'), .005])
def test_missing_or_different_actual_damping_is_unknown(bad):
    scene = build_task_inputs('towel_folding')['scene']
    scene['isaacsim'].update(cloth_elasticity_damping=.05, cloth_bend_damping=.05)
    native = {'material_attributes': {attribute: .05 for attribute in DAMPING_OPTIONS.values()}}
    native['material_attributes']['physxDeformableMaterial:elasticityDamping'] = bad
    with pytest.raises(EvidenceUnknown, match='elasticityDamping'):
        verify_solver_options(scene, 'towel', native)


def test_legacy_stream_without_material_receipts_remains_compatible():
    scene = build_task_inputs('towel_folding')['scene']
    verify_solver_options(scene, 'towel', {})
    verify_solver_options(scene, 'towel', {'material_attributes': surface_damping_properties({})})
