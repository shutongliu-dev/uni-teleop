"""The input-only contrast changes eight opinions, retaining all other gates."""
import pytest
from scripts.assets.prepare_gear_dense_sdf import ATTRIBUTE, compare_stages
from scripts.assets.prepare_gear_sector_subgrid import (
    PARAMETERS, SUBGRID2_PARAMETERS, DENSE_PARAMETERS, subgrid2_layer, dense_sector_layer,
    contrast_layer, validated_parameters)


def test_exact_eight_byte_edit_and_ambiguous_source_rejection():
    row = ('int '+ATTRIBUTE+' = 6\n').encode()
    source = row*8+b'float unrelated = 6\n'
    result = subgrid2_layer(source)
    assert len(result) == len(source)
    assert sum(a != b for a, b in zip(source, result, strict=True)) == 8
    assert result.endswith(b'float unrelated = 6\n')
    for invalid in (row*7, row*9, result, source+ATTRIBUTE.encode()):
        with pytest.raises(ValueError, match='Exactly eight'):
            subgrid2_layer(invalid)


def test_parameters_are_complete_and_single_variable_only():
    for allowed in (PARAMETERS, SUBGRID2_PARAMETERS, DENSE_PARAMETERS):
        assert validated_parameters(allowed) == allowed
        assert validated_parameters(allowed) is not allowed
    for invalid in (None, {}, {**SUBGRID2_PARAMETERS, 'sdfResolution': 1024},
                    {**SUBGRID2_PARAMETERS, 'sdfSubgridResolution': 1},
                    {**SUBGRID2_PARAMETERS, 'sdfSubgridResolution': 2.},
                    {**SUBGRID2_PARAMETERS, 'sdfEnableRemeshing': 0}):
        with pytest.raises(ValueError, match='Only complete'):
            validated_parameters(invalid)


def test_composed_audit_checks_all_eight_and_rejects_material_change(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom
    original = tmp_path/'original.usda'
    stage = Usd.Stage.CreateNew(str(original))
    body = UsdGeom.Xform.Define(stage, '/Gear').GetPrim()
    body.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(1.)
    expected = []
    for i in range(8):
        path = '/Gear/collision_'+str(i).zfill(2)
        UsdGeom.Mesh.Define(stage, path).GetPrim().CreateAttribute(ATTRIBUTE, Sdf.ValueTypeNames.Int).Set(6)
        expected.append({'prim': path, 'attribute': ATTRIBUTE, 'before': 6, 'after': 2})
    stage.GetRootLayer().Save()
    target = tmp_path/'candidate.usda'
    target.write_bytes(subgrid2_layer(original.read_bytes()))
    assert compare_stages(original, target, expected_changes=expected)['passed']
    second = Usd.Stage.Open(str(target))
    second.GetPrimAtPath('/Gear').GetAttribute('physics:mass').Set(2.)
    second.GetRootLayer().Save()
    with pytest.raises(ValueError):
        compare_stages(original, target, expected_changes=expected)


def test_dense_contrast_changes_only_the_existing_eight_opinions():
    row = ('int '+ATTRIBUTE+' = 2\n').encode()
    source = row*8+b'float unrelated = 2\n'
    result = dense_sector_layer(source)
    assert len(result) == len(source)
    assert sum(a != b for a, b in zip(source, result, strict=True)) == 8
    assert result.endswith(b'float unrelated = 2\n')
    for invalid in (row*7, row*9, result, source+ATTRIBUTE.encode()):
        with pytest.raises(ValueError, match='Exactly eight'):
            dense_sector_layer(invalid)
    with pytest.raises(ValueError, match='Only reviewed'):
        contrast_layer(source, 2, 1)


def test_dense_composed_audit_rejects_other_physics_changes(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom
    original = tmp_path/'source.usda'
    stage = Usd.Stage.CreateNew(str(original))
    body = UsdGeom.Xform.Define(stage, '/Gear').GetPrim()
    body.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(.25)
    expected = []
    for i in range(8):
        path = '/Gear/collision_'+str(i).zfill(2)
        UsdGeom.Mesh.Define(stage, path).GetPrim().CreateAttribute(ATTRIBUTE, Sdf.ValueTypeNames.Int).Set(2)
        expected.append({'prim': path, 'attribute': ATTRIBUTE, 'before': 2, 'after': 0})
    stage.GetRootLayer().Save()
    target = tmp_path/'dense.usda'
    target.write_bytes(dense_sector_layer(original.read_bytes()))
    assert compare_stages(original, target, expected_changes=expected)['passed']
    changed = Usd.Stage.Open(str(target))
    changed.GetPrimAtPath('/Gear').GetAttribute('physics:mass').Set(.3)
    changed.GetRootLayer().Save()
    with pytest.raises(ValueError):
        compare_stages(original, target, expected_changes=expected)
