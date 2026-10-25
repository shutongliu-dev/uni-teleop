"""Independent CPU rejection cases for hash-pinned visual overrides."""
import hashlib

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.presentation import apply_presentation, physical_fingerprint


ROOT = '/World/Objects/part'


def _scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, ROOT)
    mesh = UsdGeom.Mesh.Define(stage, ROOT + '/mesh')
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, ROOT + '/OriginalMaterial')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(.42)
    # The allPurpose fallback is a valid physical material binding in USD.
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return stage


def _descriptor(tmp_path, edit):
    stage = Usd.Stage.CreateInMemory()
    stage.OverridePrim('/Asset')
    edit(stage)
    path = tmp_path / 'presentation.usda'
    stage.GetRootLayer().Export(str(path))
    return {'presentation': {
        'schema': 'asset_presentation.v1',
        'usd_path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'prim_path': '/Asset',
        'dependencies_sha256': {},
    }}


def test_physical_material_fallback_rejected_without_mutating_destination(tmp_path):
    def edit(stage):
        mesh = stage.OverridePrim('/Asset/mesh')
        material = UsdShade.Material.Define(stage, '/Asset/Looks/VisualOnly')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    before = scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_presentation(scene, ROOT, spec)
    assert scene.GetRootLayer().ExportToString() == before


def test_loading_uses_hashed_file_instead_of_dirty_sdf_cache(tmp_path):
    def edit(stage):
        stage.OverridePrim('/Asset/mesh').CreateAttribute(
            'normals', Sdf.ValueTypeNames.Normal3fArray).Set([(0, 0, 1)] * 3)

    spec = _descriptor(tmp_path, edit)
    cached = Usd.Stage.Open(spec['presentation']['usd_path'])
    cached.GetPrimAtPath('/Asset/mesh').GetAttribute('normals').Set([(1, 0, 0)] * 3)
    assert cached.GetRootLayer().dirty
    scene = _scene()
    apply_presentation(scene, ROOT, spec)
    actual = scene.GetPrimAtPath(ROOT + '/mesh').GetAttribute('normals').Get()
    assert [tuple(v) for v in actual] == [(0, 0, 1)] * 3
    # The independent cached editor's unsaved state is also preserved.
    assert [tuple(v) for v in cached.GetPrimAtPath('/Asset/mesh').GetAttribute(
        'normals').Get()] == [(1, 0, 0)] * 3


def test_muted_physics_binding_cannot_make_preflight_disagree_with_destination(tmp_path):
    def edit(stage):
        mesh = stage.OverridePrim('/Asset/mesh')
        material = UsdShade.Material.Define(stage, '/Asset/Looks/VisualOnly')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    muted_layer = Sdf.Layer.CreateAnonymous('muted_physics_binding.usda')
    binding_stage = Usd.Stage.Open(muted_layer)
    binding_prim = binding_stage.OverridePrim(ROOT + '/mesh')
    material = UsdShade.Material(scene.GetPrimAtPath(ROOT + '/OriginalMaterial'))
    UsdShade.MaterialBindingAPI.Apply(binding_prim).Bind(material, materialPurpose='physics')
    scene.GetRootLayer().subLayerPaths = [muted_layer.identifier]
    scene.MuteLayer(muted_layer.identifier)
    before = scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_presentation(scene, ROOT, spec)
    assert scene.GetRootLayer().ExportToString() == before
    assert scene.GetMutedLayers() == [muted_layer.identifier]


def _session_defaults(scene):
    with Usd.EditContext(scene, scene.GetSessionLayer()):
        camera = UsdGeom.Camera.Define(scene, '/OmniverseKit_Persp')
        camera.CreateFocalLengthAttr(35.)
    return scene.GetSessionLayer().ExportToString()


def test_nonempty_session_defaults_preserved_when_applying_visual_properties(tmp_path):
    def edit(stage):
        stage.OverridePrim('/Asset/mesh').CreateAttribute(
            'normals', Sdf.ValueTypeNames.Normal3fArray).Set([(0, 0, 1)] * 3)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    session_before = _session_defaults(scene)
    physical_before = physical_fingerprint(scene, ROOT)
    receipt = apply_presentation(scene, ROOT, spec)
    assert receipt['physical_inputs_unchanged']
    assert physical_fingerprint(scene, ROOT) == physical_before
    assert scene.GetSessionLayer().ExportToString() == session_before
    assert scene.GetPrimAtPath('/OmniverseKit_Persp').GetAttribute('focalLength').Get() == 35.
    assert [tuple(v) for v in scene.GetPrimAtPath(ROOT + '/mesh').GetAttribute(
        'normals').Get()] == [(0, 0, 1)] * 3


def test_session_material_binding_retains_stronger_physical_opinion(tmp_path):
    def edit(stage):
        mesh = stage.OverridePrim('/Asset/mesh')
        material = UsdShade.Material.Define(stage, '/Asset/Looks/VisualOnly')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    _session_defaults(scene)
    with Usd.EditContext(scene, scene.GetSessionLayer()):
        material = UsdShade.Material(scene.GetPrimAtPath(ROOT + '/OriginalMaterial'))
        UsdShade.MaterialBindingAPI(scene.GetPrimAtPath(ROOT + '/mesh')).Bind(material)
        UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Set(.73)
    session_before = scene.GetSessionLayer().ExportToString()
    physical_before = physical_fingerprint(scene, ROOT)
    apply_presentation(scene, ROOT, spec)
    assert physical_fingerprint(scene, ROOT) == physical_before
    assert scene.GetSessionLayer().ExportToString() == session_before
    bound, _ = UsdShade.MaterialBindingAPI(scene.GetPrimAtPath(ROOT + '/mesh')).ComputeBoundMaterial('physics')
    assert str(bound.GetPath()) == ROOT + '/OriginalMaterial'
    assert UsdPhysics.MaterialAPI(bound.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(.73)
    root_binding = scene.GetRootLayer().GetRelationshipAtPath(ROOT + '/mesh.material:binding')
    assert list(root_binding.targetPathList.GetAppliedItems()) == [Sdf.Path(ROOT + '/Looks/VisualOnly')]


@pytest.mark.parametrize('property_name', ['physics:mass', 'points'])
def test_session_mask_cannot_make_forbidden_overlay_property_pass(tmp_path, property_name):
    value_type = Sdf.ValueTypeNames.Float if property_name == 'physics:mass' else Sdf.ValueTypeNames.Point3fArray
    authored_value = 20. if property_name == 'physics:mass' else [(0, 0, 20)] * 3
    session_value = .1 if property_name == 'physics:mass' else [(0, 0, 0), (1, 0, 0), (0, 1, 0)]

    def edit(stage):
        stage.OverridePrim('/Asset/mesh').CreateAttribute(property_name, value_type).Set(authored_value)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    _session_defaults(scene)
    with Usd.EditContext(scene, scene.GetSessionLayer()):
        scene.GetPrimAtPath(ROOT + '/mesh').CreateAttribute(property_name, value_type).Set(session_value)
    root_before = scene.GetRootLayer().ExportToString()
    session_before = scene.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_presentation(scene, ROOT, spec)
    assert scene.GetRootLayer().ExportToString() == root_before
    assert scene.GetSessionLayer().ExportToString() == session_before


def test_author_presentation_preserves_original_session_and_exports_only_its_overlay(tmp_path):
    from scenes.workrooms.mechanical_assembly.presentation_release import author_presentation

    scene = _scene()
    mesh = scene.GetPrimAtPath(ROOT + '/mesh')
    material = UsdShade.Material(scene.GetPrimAtPath(ROOT + '/OriginalMaterial'))
    UsdShade.MaterialBindingAPI(mesh).Bind(material, materialPurpose='physics')
    _session_defaults(scene)
    with Usd.EditContext(scene, scene.GetSessionLayer()):
        mesh.CreateAttribute('visibility', Sdf.ValueTypeNames.Token).Set('inherited')
    session_before = scene.GetSessionLayer().ExportToString()
    before = physical_fingerprint(scene, ROOT)
    output = tmp_path / 'authored.usda'
    descriptor, receipt = author_presentation(scene, ROOT, output, role='steel_brushed')
    assert descriptor['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt['physical_inputs_unchanged']
    assert physical_fingerprint(scene, ROOT) == before
    assert scene.GetSessionLayer().ExportToString() == session_before
    exported = Sdf.Layer.OpenAsAnonymous(str(output))
    assert not exported.subLayerPaths
    assert [str(prim.path) for prim in exported.rootPrims] == ['/Asset']
    assert not exported.GetPropertyAtPath('/Asset/mesh.visibility')


@pytest.mark.parametrize('metres,axis', [(1., 'Z'), (.01, 'Y'), (.001, 'Z')])
def test_private_presentation_stage_retains_effective_units_without_source_writes(metres, axis):
    from bench.simulation.isaacsim_backend.presentation import presentation_stage

    scene = _scene()
    UsdGeom.SetStageMetersPerUnit(scene, metres)
    UsdGeom.SetStageUpAxis(scene, axis)
    _session_defaults(scene)
    original_root = scene.GetRootLayer().ExportToString()
    original_session = scene.GetSessionLayer().ExportToString()
    scratch = presentation_stage(scene)
    assert UsdGeom.GetStageMetersPerUnit(scratch) == metres
    assert UsdGeom.GetStageUpAxis(scratch) == axis
    assert scene.GetRootLayer().ExportToString() == original_root
    assert scene.GetSessionLayer().ExportToString() == original_session


def test_60mm_body_authored_and_reloaded_at_80mm_texture_repeat(tmp_path):
    from itertools import product
    import numpy as np
    from scenes.workrooms.mechanical_assembly.presentation_release import author_presentation

    scene = _scene()
    # build_scene owns these same metre/Z-up opinions in both normal runtime
    # and static preview stages. The private authoring stage must retain them.
    UsdGeom.SetStageMetersPerUnit(scene, 1.)
    UsdGeom.SetStageUpAxis(scene, 'Z')
    mesh = UsdGeom.Mesh(scene.GetPrimAtPath(ROOT + '/mesh'))
    mesh.GetPointsAttr().Set(list(product((-.03, .03), repeat=3)))
    mesh.GetFaceVertexCountsAttr().Set([4]*6)
    mesh.GetFaceVertexIndicesAttr().Set([0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3])
    mesh.CreateSubdivisionSchemeAttr().Set('none')
    material = UsdShade.Material(scene.GetPrimAtPath(ROOT + '/OriginalMaterial'))
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    session_before = _session_defaults(scene)
    physical_before = physical_fingerprint(scene, ROOT)
    original_points = np.array(mesh.GetPointsAttr().Get())
    output = tmp_path / '60mm_steel.usda'
    _, receipt = author_presentation(scene, ROOT, output, role='steel_brushed')
    uv = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get())
    assert np.ptp(uv, axis=0) == pytest.approx([.06/.08, .06/.08], abs=1.e-7)
    exported = Usd.Stage.Open(str(output))
    exported_uv = np.array(UsdGeom.PrimvarsAPI(exported.GetPrimAtPath('/Asset/mesh')).GetPrimvar('st').Get())
    assert np.array_equal(exported_uv, uv)
    assert np.array_equal(np.array(mesh.GetPointsAttr().Get()), original_points)
    assert physical_fingerprint(scene, ROOT) == physical_before
    assert receipt['physical_inputs_unchanged']
    assert scene.GetSessionLayer().ExportToString() == session_before
    assert UsdGeom.GetStageMetersPerUnit(scene) == 1.
    assert UsdGeom.GetStageUpAxis(scene) == 'Z'


@pytest.mark.parametrize('array_value', [False, True])
@pytest.mark.parametrize('time', [None, 1])
def test_unpinned_asset_value_rejected_before_mutation(tmp_path, array_value, time):
    def edit(stage):
        shader = UsdShade.Shader.Define(stage, '/Asset/Looks/Texture')
        shader.CreateIdAttr('UsdUVTexture')
        value_type = Sdf.ValueTypeNames.AssetArray if array_value else Sdf.ValueTypeNames.Asset
        asset = Sdf.AssetPath('/untracked/presentation_texture.png')
        value = [asset] if array_value else asset
        texture_input = shader.CreateInput('file', value_type)
        if time is None:
            texture_input.Set(value)
        else:
            texture_input.Set(value, time)

    spec = _descriptor(tmp_path, edit)
    scene = _scene()
    before = scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_presentation(scene, ROOT, spec)
    assert scene.GetRootLayer().ExportToString() == before
