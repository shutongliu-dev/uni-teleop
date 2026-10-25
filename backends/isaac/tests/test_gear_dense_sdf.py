"""The dense candidate must be a single-variable copy, retaining strict receipts."""
import numpy as np
import pytest

from scripts.assets.prepare_gear_dense_sdf import dense_layer, ATTRIBUTE, DENSE_PARAMETERS
from scripts.assets.prepare_gear_sdf import SDF_PARAMETERS
from scripts.validation.gear_sdf_checks import calibrate_layout


def test_dense_layer_changes_exactly_one_byte_and_rejects_missing_or_duplicate():
    source = ('#usda 1.0\nint '+ATTRIBUTE+' = 6\nfloat other = 6\n').encode()
    result = dense_layer(source)
    assert len(result) == len(source)
    assert sum(a != b for a, b in zip(source, result, strict=True)) == 1
    assert result.endswith(b'float other = 6\n')
    for invalid in (b'none', source+source, result):
        with pytest.raises(ValueError, match='Exactly one'):
            dense_layer(invalid)
    assert {k: (SDF_PARAMETERS[k], DENSE_PARAMETERS[k]) for k in SDF_PARAMETERS
            if SDF_PARAMETERS[k] != DENSE_PARAMETERS[k]} == {'sdfSubgridResolution': (6, 0)}


def test_nonfinite_unused_gradient_in_derivative_receipt_fails():
    points = np.array([[.011, 0., 0.], [-.011, 0., 0.], [0., .011, 0.],
                       [0., -.011, 0.], [0., 0., .011], [0., 0., -.011]])
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    calls = 0
    def query(p):
        nonlocal calls
        calls += 1
        radii = np.linalg.norm(p, axis=1)
        raw = np.c_[p/radii[:, None], radii-.01]
        if calls == 2:
            raw[:, 0] = np.nan
        return raw
    with pytest.raises(ValueError, match='Every calibration query'):
        calibrate_layout(query, points, np.full(6, .001), normals)


def test_composed_audit_rejects_unrelated_attribute_change(tmp_path):
    pytest.importorskip('pxr')
    from scripts.assets.prepare_gear_dense_sdf import compare_stages
    from pxr import Sdf, Usd, UsdGeom
    original = tmp_path/'original.usda'
    stage = Usd.Stage.CreateNew(str(original))
    body = UsdGeom.Xform.Define(stage, '/Gear').GetPrim()
    body.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(1.)
    collider = UsdGeom.Mesh.Define(stage, '/Gear/collision').GetPrim()
    collider.CreateAttribute(ATTRIBUTE, Sdf.ValueTypeNames.Int).Set(6)
    stage.GetRootLayer().Save()
    candidate = tmp_path/'candidate.usda'
    candidate.write_bytes(dense_layer(original.read_bytes()))
    assert compare_stages(original, candidate)['passed']
    second = Usd.Stage.Open(str(candidate))
    second.GetPrimAtPath('/Gear').GetAttribute('physics:mass').Set(2.)
    second.GetRootLayer().Save()
    with pytest.raises(ValueError, match='Only the single'):
        compare_stages(original, candidate)
