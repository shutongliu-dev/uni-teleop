"""A weld0 cooking contrast must not silently modify geometry or other physics."""
import pytest

from scripts.assets.prepare_gear_sector_weld import (
    WELD_API, WELD_ATTRIBUTE, audit_weld0_stages, weld0_layer)


def stage_text():
    result = ['#usda 1.0\n', 'def Xform "Gear" {\n    float physics:mass = 1.5\n']
    for i in range(8):
        result.append(f'    def Mesh "collision_{i:02d}" (\n'
            '        apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", '
            '"PhysxCollisionAPI", "PhysxSDFMeshCollisionAPI", "MaterialBindingAPI"]\n'
            '    ) {\n        int physxSDFMeshCollision:sdfSubgridResolution = 2\n'
            '        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n'
            '        int[] faceVertexCounts = [3]\n        int[] faceVertexIndices = [0, 1, 2]\n'
            '        float physxCollision:contactOffset = 0.0001\n    }\n')
    return (''.join(result)+'}\n').encode()


def test_exact_insertions_preserve_original_layer_bytes():
    original = stage_text()
    candidate = weld0_layer(original)
    assert candidate.count(WELD_API.encode()) == 8
    assert candidate.count(WELD_ATTRIBUTE.encode()) == 8
    restored = candidate.replace(b', "PhysxTriangleMeshCollisionAPI"', b'').replace(
        b'        float physxTriangleMeshCollision:weldTolerance = 0\n', b'')
    assert restored == original


@pytest.mark.parametrize('change', [
    lambda text: text.replace(b'Resolution = 2', b'Resolution = 0'),
    lambda text: text.replace(b'Resolution = 2', b'Resolution = 2.0', 1),
    lambda text: text+b'float physxTriangleMeshCollision:weldTolerance = 1\n',
    lambda text: text.replace(b'"PhysxSDFMeshCollisionAPI", ', b'', 1),
    lambda text: weld0_layer(text),
])
def test_rejects_different_parent_or_existing_weld_opinions(change):
    with pytest.raises(ValueError, match='Exactly eight'):
        weld0_layer(change(stage_text()))


def files(tmp_path):
    first, second = tmp_path/'source.usda', tmp_path/'candidate.usda'
    first.write_bytes(stage_text())
    second.write_bytes(weld0_layer(first.read_bytes()))
    return first, second


def test_composed_authoring_has_only_weld_change(tmp_path):
    pytest.importorskip('pxr')
    first, second = files(tmp_path)
    audit = audit_weld0_stages(first, second)
    assert audit['passed'] and len(audit['changes']) == 8


@pytest.mark.parametrize('old,new', [(b'mass = 1.5', b'mass = 1.6'),
    (b'contactOffset = 0.0001', b'contactOffset = 0.0002'),
    (b'[(0, 0, 0),', b'[(0, 0, 1),'),
    (b'weldTolerance = 0', b'weldTolerance = 0.000001')])
def test_rejects_mass_offset_geometry_and_nonzero_weld_changes(tmp_path, old, new):
    pytest.importorskip('pxr')
    first, second = files(tmp_path)
    second.write_bytes(second.read_bytes().replace(old, new, 1))
    with pytest.raises(ValueError):
        audit_weld0_stages(first, second)


@pytest.mark.parametrize('kind', ['time_sample', 'relationship'])
def test_rejects_time_sample_or_relationship_changes(tmp_path, kind):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd
    first, second = files(tmp_path)
    stage = Usd.Stage.Open(str(second))
    prim = stage.GetPrimAtPath('/Gear/collision_00')
    if kind == 'relationship':
        prim.CreateRelationship('test').SetTargets([Sdf.Path('/Gear')])
    else:
        prim.GetAttribute('physxCollision:contactOffset').Set(.0002, 1.)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        audit_weld0_stages(first, second)
