"""Original threaded USD catalog contracts; CPU only, no native acceptance."""
import hashlib
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.threaded_catalog import ASSET_IDS, prepare_threaded_catalog
from scenes.workrooms.mechanical_assembly.threaded_assets import TOOL_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    pytest.importorskip('pxr')
    if not TOOL_ROOT.is_dir():
        pytest.skip('Pinned shared threaded assets are not installed')
    root = tmp_path_factory.mktemp('mechanical_catalog') / 'v1'
    result = prepare_threaded_catalog(root)
    return root, result, load_json(root / 'SOURCE.json')


def test_existing_catalog_is_never_overwritten(tmp_path):
    marker = tmp_path / 'keep.txt'
    marker.write_text('immutable')
    with pytest.raises(FileExistsError, match='new mechanical catalog version'):
        prepare_threaded_catalog(tmp_path)
    assert marker.read_text() == 'immutable'
    assert list(tmp_path.iterdir()) == [marker]


def test_catalog_preserves_original_scale_nested_bodies_and_collision_geometry(prepared):
    root, _, source = prepared
    expected = {'screwdriver': ('root', ['convexHull'] * 3, .12797346711158752),
                'bolt': ('anchored', ['sdf', 'sdf'], .05000000074505806),
                'fixture': ('anchored', ['none', 'convexHull', 'convexHull', 'sdf'], .5)}
    for role, (form, approximations, mass) in expected.items():
        asset = load_task_asset(ASSET_IDS[role], root, verify_hashes=True)
        assert asset['placement_form'] == form
        assert asset['source_scale'] == [1., 1., 1.]
        assert asset['mass_kg'] == pytest.approx(mass)
        assert [part['approximation'] for part in asset['collision_parts']] == approximations
        assert source['assets'][role]['body_relative_to_default'] == [
            [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    assert len(list(root.iterdir())) == 3
    assert not list(root.rglob('*.usd')) and not list(root.rglob('*.obj'))


def test_source_physics_and_fallback_priors_are_distinguished(prepared):
    _, result, source = prepared
    for collider in source['assets']['screwdriver']['colliders']:
        assert collider['authored_parameters']['physxCollision:contactOffset'] == pytest.approx(.0002)
        assert collider['physics_material']['authored_parameters']['physics:staticFriction'] == pytest.approx(.6)
    for role in ('bolt', 'fixture'):
        for collider in source['assets'][role]['colliders']:
            assert 'physxCollision:contactOffset' not in collider['authored_parameters']
            assert not collider['physics_material']['has_physics_material_api']
            if collider['approximation'] == 'sdf':
                assert collider['authored_parameters']['physxSDFMeshCollision:sdfResolution'] == 512
    options = result['recommended_spec_options']
    assert options['usd_physics_policy'] == 'preserve_authored_v1'
    assert options['usd_physics_fallbacks']['contact_offset_m'] == .0002
    assert not options['usd_physics_fallbacks_provenance']['native_validated']
    assert not source['source_modified'] and not source['native_physics_validated']


def test_catalog_pins_separate_licenses_source_manifests_and_audit(prepared):
    root, result, source = prepared
    assert source['licenses']['tool']['terms'] == 'CC0-1.0'
    assert 'NVIDIA' in source['licenses']['assembly']['terms']['factory_threads_and_factory_assets']
    for role, asset_id in ASSET_IDS.items():
        asset = load_task_asset(asset_id, root, verify_hashes=True)
        group = 'tool' if role == 'screwdriver' else 'assembly'
        license_path = source['licenses'][group]['path']
        provenance_path = source['source_manifests'][group]['path']
        assert license_path in asset['source_dependencies_sha256']
        assert provenance_path in asset['source_dependencies_sha256']
        assert asset['source_dependencies_sha256'][str(root / 'SOURCE.json')] == result['source_audit_sha256']


def test_offset_source_body_is_rejected_before_native_reset_could_disagree(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.mechanical_assembly.threaded_catalog import _inspect_asset
    path = tmp_path / 'offset.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = UsdGeom.Xform.Define(stage, '/Part')
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, '/Part/body')
    body.AddTranslateOp().Set(Gf.Vec3d(.01, 0., 0.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    stage.GetRootLayer().Save()
    binding = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'default_prim': '/Part'}
    with pytest.raises(ValueError, match='same origin'):
        _inspect_asset(binding, role='bolt', provenance_dependencies={})
