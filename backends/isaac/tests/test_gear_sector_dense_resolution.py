"""The dense1024 storage contrast must preserve every unrelated authored value."""
import pytest

from scripts.assets.prepare_gear_sector_dense_resolution import (
    ATTRIBUTE, SUBGRID_ATTRIBUTE, audit_dense1024_stages, dense1024_layer)
from scripts.assets.prepare_gear_sector_subgrid import DENSE1024_PARAMETERS, validated_parameters


def stage_text():
    result = ['#usda 1.0\n', 'def Xform "Gear" {\n    float physics:mass = 1.5\n']
    for i in range(8):
        result.append(f'    def Mesh "collision_{i:02d}" (\n'
            '        apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", '
            '"PhysxCollisionAPI", "PhysxSDFMeshCollisionAPI", "MaterialBindingAPI"]\n'
            '    ) {\n        int physxSDFMeshCollision:sdfResolution = 1250\n'
            '        int physxSDFMeshCollision:sdfSubgridResolution = 0\n'
            '        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n'
            '        int[] faceVertexCounts = [3]\n        int[] faceVertexIndices = [0, 1, 2]\n'
            '        float physxCollision:contactOffset = 0.0001\n    }\n')
    return (''.join(result)+'}\n').encode()


def test_exact_eight_resolution_literals_preserve_all_other_bytes():
    original = stage_text()
    candidate = dense1024_layer(original)
    old, new = (f'int {ATTRIBUTE} = {n}\n'.encode() for n in (1250, 1024))
    assert candidate.count(new) == 8
    assert candidate.replace(new, old) == original
    assert len(candidate) == len(original)
    assert sum(a != b for a, b in zip(original, candidate, strict=True)) == 24


@pytest.mark.parametrize('change', [
    lambda text: text.replace(b'sdfResolution = 1250', b'sdfResolution = 1249', 1),
    lambda text: text.replace(b'sdfResolution = 1250', b'sdfResolution = 12500', 1),
    lambda text: text.replace(b'sdfResolution = 1250', b'sdfResolution = 1250.0', 1),
    lambda text: text.replace(b'sdfSubgridResolution = 0', b'sdfSubgridResolution = 2', 1),
    lambda text: text+b'float physxTriangleMeshCollision:weldTolerance = 0\n',
    lambda text: text.replace(b'"PhysxSDFMeshCollisionAPI", ', b'"PhysxSDFMeshCollisionAPI", "PhysxTriangleMeshCollisionAPI", ', 1),
    lambda text: text+ATTRIBUTE.encode(),
    lambda text: text+SUBGRID_ATTRIBUTE.encode(),
    lambda text: dense1024_layer(text),
])
def test_only_the_frozen_dense_parent_is_a_valid_layer(change):
    with pytest.raises(ValueError, match='Exactly eight'):
        dense1024_layer(change(stage_text()))


def test_only_the_complete_dense1024_parameter_profile_is_added():
    value = validated_parameters(DENSE1024_PARAMETERS)
    assert value == DENSE1024_PARAMETERS and value is not DENSE1024_PARAMETERS
    for invalid in (None, {}, {**value, 'sdfResolution': 1000},
            {**value, 'sdfResolution': 1024.}, {**value, 'sdfSubgridResolution': 2},
            {**value, 'sdfSubgridResolution': False}, {**value, 'sdfEnableRemeshing': True},
            {**value, 'sdfTriangleCountReductionFactor': .5},
            {**value, 'sdfBitsPerSubgridPixel': 'BitsPerPixel8'}, {**value, 'extra': 0}):
        with pytest.raises(ValueError, match='Only complete'):
            validated_parameters(invalid)


def files(tmp_path):
    first, second = tmp_path/'source.usda', tmp_path/'candidate.usda'
    first.write_bytes(stage_text())
    second.write_bytes(dense1024_layer(first.read_bytes()))
    return first, second


def test_composed_authoring_has_only_eight_resolution_changes(tmp_path):
    pytest.importorskip('pxr')
    first, second = files(tmp_path)
    audit = audit_dense1024_stages(first, second)
    assert audit['passed'] and len(audit['changes']) == 8
    assert audit['byte_difference_count'] == 24


@pytest.mark.parametrize('old,new', [
    (b'mass = 1.5', b'mass = 1.6'),
    (b'contactOffset = 0.0001', b'contactOffset = 0.0002'),
    (b'[(0, 0, 0),', b'[(0, 0, 1),'),
    (b'sdfSubgridResolution = 0', b'sdfSubgridResolution = 2'),
    (b'int physxSDFMeshCollision:sdfResolution = 1024', b'float physxSDFMeshCollision:sdfResolution = 1024'),
])
def test_composed_audit_rejects_geometry_physics_and_profile_changes(tmp_path, old, new):
    pytest.importorskip('pxr')
    first, second = files(tmp_path)
    second.write_bytes(second.read_bytes().replace(old, new, 1))
    with pytest.raises(ValueError):
        audit_dense1024_stages(first, second)


@pytest.mark.parametrize('kind', ['time_sample', 'relationship', 'resolution_time_sample', 'prim_metadata'])
def test_rejects_changes_to_authored_semantics(tmp_path, kind):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd
    first, second = files(tmp_path)
    stage = Usd.Stage.Open(str(second))
    prim = stage.GetPrimAtPath('/Gear/collision_00')
    if kind == 'relationship':
        prim.CreateRelationship('test').SetTargets([Sdf.Path('/Gear')])
    elif kind == 'resolution_time_sample':
        prim.GetAttribute(ATTRIBUTE).Set(1024, 1.)
    elif kind == 'prim_metadata':
        prim.SetDocumentation('unexpected change')
    else:
        prim.GetAttribute('physxCollision:contactOffset').Set(.0002, 1.)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        audit_dense1024_stages(first, second)
