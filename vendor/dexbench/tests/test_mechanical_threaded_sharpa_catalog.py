"""Pinned v3 composition/profile tests; no new native or robot qualification."""

import pytest

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog import (
    ASSET_IDS, SHARPA_ROOT, prepare_threaded_sharpa_catalog, threaded_sharpa_asset_bindings)
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset


@pytest.fixture(scope='module')
def catalog(tmp_path_factory):
    pytest.importorskip('pxr')
    if not SHARPA_ROOT.is_dir():
        pytest.skip('Pinned shared v3 bundle unavailable')
    root = tmp_path_factory.mktemp('sharpa_catalog') / 'v2'
    result = prepare_threaded_sharpa_catalog(root)
    return root, result, load_json(root / 'SOURCE.json')


def test_v3_assets_keep_existing_scale_geometry_and_mass(catalog):
    root, _, audit = catalog
    expected = {'screwdriver': (1., 3, .3039310872554779),
                'bolt': (1.5, 8, .16875000298023224), 'fixture': (1.5, 4, 1.6875)}
    for role, (scale, count, mass) in expected.items():
        asset = load_task_asset(ASSET_IDS[role], root, verify_hashes=True)
        assert asset['source_scale'] == [scale] * 3
        assert len(asset['collision_parts']) == count
        assert asset['mass_kg'] == pytest.approx(mass)
        assert asset['usd_path'] == threaded_sharpa_asset_bindings()[role]['path']
        assert audit['assets'][role]['historical_native_properties']['mass_kg'][0][0] == pytest.approx(mass)
    assert not list(root.rglob('*.usd')) and len(list(root.iterdir())) == 3


def test_role_profile_is_in_manifest_and_keeps_historical_scope(catalog):
    root, result, audit = catalog
    manifest = load_json(root / 'manifest.json')
    assert manifest['per_role_spec_options'] == result['per_role_spec_options'] == audit['per_role_spec_options']
    for role, friction in [('screwdriver', .6), ('bolt', .01), ('fixture', .75)]:
        opt = manifest['per_role_spec_options'][role]
        assert opt['usd_physics_material_fallback']['static_friction'] == friction
        assert opt['usd_physics_fallbacks']['solver_velocity_iterations'] == 16
        assert opt['usd_physics_fallbacks']['max_depenetration_velocity_m_s'] == .05
        assert opt['usd_physics_fallbacks']['enable_speculative_ccd'] is False
        assert opt['usd_physics_fallbacks_provenance']['historical_contact_validated']
        assert not opt['usd_physics_fallbacks_provenance']['current_scene_validated']
    assert not audit['source_assets_modified']
    assert not audit['current_scene_native_validated'] and not audit['robot_manipulation_validated']
    assert audit['licenses']['driver']['terms'] == 'CC0-1.0'
    assert 'NVIDIA' in audit['licenses']['assembly']['terms']


def test_source_authorship_and_new_version_are_explicit(catalog):
    root, _, audit = catalog
    for role, source in audit['assets'].items():
        for collider in source['colliders']:
            assert collider['authored_parameters']['physxCollision:contactOffset'] == pytest.approx(.0003)
            assert collider['physics_material']['has_physics_material_api'] == (role == 'screwdriver')
    before = (root / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare_threaded_sharpa_catalog(root)
    assert (root / 'manifest.json').read_bytes() == before
