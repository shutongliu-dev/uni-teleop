"""Reference composition of the native collaborator USD loader (``usd_asset``).

``add_usd_asset`` has two placement forms:

* root form -- the source default prim is the rigid body and has an identity
  transform. It is referenced straight onto /World/<kind>/<id>, exactly as the
  loader always did; every frozen-release asset uses it.
* anchored form -- the rigid body is a CHILD prim and/or the default prim
  authors its own transform (the SHARPA v3 bolt and fixture: body on
  /allen_bolt and /platform, xformOp:scale 1.5 on the default prim). Placing the
  reference on the placement prim and clearing its op order used to drop that
  scale (0.67x objects), and the loader refused a child body outright. The
  source is now referenced onto a child anchor under a placement wrapper.

The tests build small USD layers of both forms in a temporary directory, so they
depend on no catalogue. ``PhysxSchema`` ships with Isaac, not OpenUSD, and
``scene.rigid_material`` imports the robot stack, so both are stubbed; the
composition, collider resolution and transforms are real OpenUSD. Run them with
an interpreter that has pxr (the CPU suite skips them), e.g.
<data>/assets/tools/usd/26.8/venv/bin/python with numpy on PYTHONPATH.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json

FROZEN_CATALOG = Path(f'{DATA_ROOT}/assets/objects/hand_system0/bench_catalog/v1')

ROOT_BODY = """#usda 1.0
(
    defaultPrim = "Part"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Part" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    def Cube "collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.04
    }
}
"""

CHILD_BODY = """#usda 1.0
(
    defaultPrim = "Kit"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Kit"
{
    float3 xformOp:scale = (1.5, 1.5, 1.5)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def Xform "body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Cube "collider" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.04
        }
    }
}
"""


@pytest.fixture
def usd_asset(monkeypatch):
    """The real loader module, with its two native-only dependencies stubbed."""
    pxr = pytest.importorskip('pxr')
    from pxr import UsdPhysics, UsdShade

    if not hasattr(pxr, 'PhysxSchema'):
        class _Schema:
            def __init__(self, prim):
                self.prim = prim

            @classmethod
            def Apply(cls, prim, *args, **kwargs):
                return cls(prim)

            def __getattr__(self, name):
                if name.startswith('Create'):       # any Create*Attr; values are not read
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        stub = types.ModuleType('pxr.PhysxSchema')
        for name in ('PhysxRigidBodyAPI', 'PhysxContactReportAPI', 'PhysxCollisionAPI', 'PhysxMaterialAPI'):
            setattr(stub, name, _Schema)
        monkeypatch.setitem(sys.modules, 'pxr.PhysxSchema', stub)
        monkeypatch.setattr(pxr, 'PhysxSchema', stub, raising=False)

    def rigid_material(stage, name, friction):
        material = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/' + name)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(float(friction))
        api.CreateDynamicFrictionAttr(float(friction))
        return material

    scene = types.ModuleType('bench.simulation.isaacsim_backend.scene')
    scene.rigid_material = rigid_material
    monkeypatch.setitem(sys.modules, 'bench.simulation.isaacsim_backend.scene', scene)
    monkeypatch.delitem(sys.modules, 'bench.simulation.isaacsim_backend.usd_asset', raising=False)
    from bench.simulation.isaacsim_backend import usd_asset
    return usd_asset


def _asset(tmp_path, name, text, default_prim, collider):
    path = tmp_path / f'{name}.usda'
    path.write_text(text)
    return {'usd_path': str(path), 'usd_default_prim': default_prim, 'asset_root': str(tmp_path),
            'manifest_sha256': '0' * 64, 'source_dependencies_sha256': {}, 'runtime_builtin_dependencies': [],
            'collision_parts': [{'usd_prim_path': collider, 'approximation': 'none'}],
            'visual_parts': [], 'material_prim_paths': [], 'mass_kg': 0.1, 'physics_assumptions': {},
            'bounds_min_m': [-0.02, -0.02, -0.02], 'bounds_max_m': [0.02, 0.02, 0.02]}


def _place(usd_asset, asset, *, dynamic=True):
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    spec = {'id': 'thing', 'asset_id': 'thing', 'position_m': [0.5, 0.2, 0.75], 'quaternion_wxyz': [1, 0, 0, 0]}
    return stage, usd_asset.add_usd_asset(stage, spec, asset, dynamic=dynamic)


def _world(stage, path):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    size = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        prim).ComputeAlignedRange().GetSize()
    return matrix, [round(v, 9) for v in size]


def test_a_root_body_keeps_the_legacy_prim_and_record(usd_asset, tmp_path):
    from pxr import UsdGeom, UsdPhysics
    asset = _asset(tmp_path, 'root', ROOT_BODY, '/Part', '/Part/collider')
    assert usd_asset.placement_form(asset) == 'root'
    stage, info = _place(usd_asset, asset)
    assert info['path'] == '/World/Objects/thing' and info['contact_filter_paths'] == ['/World/Objects/thing']
    assert not {'placement_form', 'wrapper_path', 'anchor_path'} & set(info), 'the legacy record is unchanged'
    root = stage.GetPrimAtPath(info['path'])
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert [op.GetOpName() for op in UsdGeom.Xformable(root).GetOrderedXformOps()] == \
        ['xformOp:transform:benchPlacement']
    assert info['collision_paths'] == ['/World/Objects/thing/collider']
    matrix, size = _world(stage, info['path'])
    assert list(matrix.ExtractTranslation()) == pytest.approx([0.5, 0.2, 0.75])
    assert size == pytest.approx([0.04, 0.04, 0.04])


def test_a_child_body_under_a_scaled_default_prim_keeps_its_scale(usd_asset, tmp_path):
    from pxr import UsdGeom, UsdPhysics
    asset = _asset(tmp_path, 'child', CHILD_BODY, '/Kit', '/Kit/body/collider')
    assert usd_asset.placement_form(asset) == 'anchored'
    stage, info = _place(usd_asset, asset)
    assert info['placement_form'] == 'anchored'
    assert info['wrapper_path'] == '/World/Objects/thing'
    assert info['anchor_path'] == '/World/Objects/thing/Kit'
    assert info['path'] == '/World/Objects/thing/Kit/body'
    assert stage.GetPrimAtPath(info['path']).HasAPI(UsdPhysics.RigidBodyAPI)
    assert info['contact_filter_paths'] == [info['path']], 'contacts name the body, not the wrapper'
    assert info['collision_paths'] == ['/World/Objects/thing/Kit/body/collider']
    wrapper = UsdGeom.Xformable(stage.GetPrimAtPath(info['wrapper_path']))
    assert [op.GetOpName() for op in wrapper.GetOrderedXformOps()] == ['xformOp:transform:benchPlacement']
    matrix, size = _world(stage, info['path'])
    assert [abs(matrix[i][i]) for i in range(3)] == pytest.approx([1.5, 1.5, 1.5])
    assert list(matrix.ExtractTranslation()) == pytest.approx([0.5, 0.2, 0.75])
    assert size == pytest.approx([0.06, 0.06, 0.06]), 'the source scale must survive placement'


def test_placing_onto_the_reference_prim_would_drop_the_scale(tmp_path):
    """The defect the anchored form exists for, stated executably."""
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom
    path = tmp_path / 'child.usda'
    path.write_text(CHILD_BODY)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Objects/naive').GetPrim()
    root.GetReferences().AddReference(str(path), Sdf.Path('/Kit'))
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix='benchPlacement')
    size = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        root).ComputeAlignedRange().GetSize()
    assert size[0] == pytest.approx(0.04), 'clearing the referenced op order loses the 1.5x scale'


def test_a_body_offset_from_its_wrapper_is_refused(usd_asset, tmp_path):
    """env.reset writes the task pose to the body; an offset body would disagree with the stage."""
    text = CHILD_BODY.replace('    {\n        def Cube "collider"',
                              '    {\n        double3 xformOp:translate = (0, 0, 0.05)\n'
                              '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n'
                              '        def Cube "collider"', 1)
    assert text != CHILD_BODY
    asset = _asset(tmp_path, 'offset', text, '/Kit', '/Kit/body/collider')
    with pytest.raises(ValueError, match='offset from its placement wrapper'):
        _place(usd_asset, asset)


@pytest.mark.skipif(not FROZEN_CATALOG.is_dir(), reason='frozen target_cube catalogue not on this host')
def test_every_frozen_catalogue_asset_stays_on_the_root_form(usd_asset):
    """Frozen-release assets must never be restructured by a loader change."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    keys = load_json(FROZEN_CATALOG / 'manifest.json')['assets']
    forms = {key: usd_asset.placement_form(load_task_asset(key, FROZEN_CATALOG)) for key in keys}
    assert len(forms) >= 200
    assert set(forms.values()) == {'root'}, {k: v for k, v in forms.items() if v != 'root'}
