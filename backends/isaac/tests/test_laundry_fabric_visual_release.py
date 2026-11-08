"""Render-only fabric candidates pin baked profiles and keep every task input."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, dump, task_inputs, validate
from bench.paths import load_json
from bench.simulation.isaacsim_backend.fabric_visuals import cloth_geometry_digest
from bench.simulation.isaacsim_backend.geometry import cloth_grid
from scenes.workrooms.laundry_room import composition, fabric_visual_release as derived, release
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES

SHIRT = Path(RELEASES['shirt_folding']['configurations']['dual_opposed']['source_release']['path'])
STACK = Path(RELEASES['towel_stacking']['configurations']['single_arm']['source_release']['path'])


def read(path):
    return load_json(Path(path))


def inventory(root):
    return {str(path.relative_to(root)): digest(path) for path in root.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts}


def fixture_profile(directory, cloth, **binding):
    from PIL import Image
    directory.mkdir(parents=True)
    Image.new('RGB', (4, 4), (90, 110, 140)).save(directory/'albedo.png')
    rest, triangles = cloth_grid(cloth)
    profile = {'schema': 'fabric_visual_profile.v1', 'id': 'fixture:'+cloth['id'], 'shader': 'omnisurface',
        'subdivision_levels': 1, 'textures': {'albedo': 'albedo.png'},
        'texture_sha256': {'albedo': digest(directory/'albedo.png')},
        'uv_frame': {'origin_xy_m': [-1., -1.], 'size_xy_m': [2., 2.], 'atlas': 'two_sided'},
        'cloth_binding': {'cloth_geometry_sha256': cloth_geometry_digest(rest, triangles),
                          'dye_linear_rgb': cloth['isaacsim']['cloth_visual_color'], **binding},
        'render_only': True, 'physics_changed': False}
    dump(directory/'profile.json', profile)
    return directory/'profile.json'


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    from scripts.maintenance import freeze_project_sources
    # Bypass only source freezing and repeated large shared-asset hashing; release
    # hashes, profile pins, reconstruction and the frozen evaluator run unchanged.
    frozen = tmp_path/'frozen_source'
    dump(frozen/'source_snapshot_manifest.json', {'fixture_only': True})
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot',
                        lambda *a, **k: {'fixture_only': True, 'snapshot_root': str(frozen)})
    monkeypatch.setattr(derived, 'DATASETS', Path('/'))
    monkeypatch.setattr(derived, 'ASSETS', Path('/'))
    monkeypatch.setattr(release, 'DATASETS', Path('/'))

    def base_source(source):
        from laundry_release_fixtures import unbound_source
        original_inventory = read(source/'asset_file_hashes.json')
        scene = read(source/'scene.json')
        pins = {r['id']: r['visual_profile'] for r in scene['cloths'] if 'visual_profile' in r}
        for name in derived.profile_inventory(pins):
            original_inventory.pop(name, None)
        monkeypatch.setattr(release, 'asset_inventory', lambda scene: deepcopy(original_inventory))
        return unbound_source(source, tmp_path/'unbound', visuals=False)

    def prepare(source, name='candidate', profiles=None):
        source = base_source(source)
        base_inventory = read(source/'asset_file_hashes.json')

        def asset_inventory(scene):
            hashes = {path: row['sha256'] for path, row in base_inventory.items()}
            visuals = {c['id']: c['visual_profile'] for c in scene['cloths'] if 'visual_profile' in c}
            if visuals:
                hashes.update(derived.profile_inventory(visuals))
            return {path: {'sha256': sha} for path, sha in sorted(hashes.items())}
        monkeypatch.setattr(release, 'asset_inventory', asset_inventory)
        scene = read(source/'scene.json')
        if profiles is None:
            profiles = {cloth['id']: fixture_profile(tmp_path/'profiles'/name/cloth['id'], cloth)
                        for cloth in scene['cloths']}
        output = tmp_path/name
        return output, derived.prepare(source, output, profiles)
    prepare.source = base_source
    return prepare


@pytest.mark.parametrize('source', [SHIRT, STACK], ids=['shirt-dual_opposed', 'towel_stacking-single_arm'])
def test_profiles_are_pinned_without_changing_any_task_input(preparation, source):
    source = preparation.source(source)
    before = inventory(source)
    old_manifest = validate(source, assets=False)
    output, manifest = preparation(source)
    assert validate(output) == manifest
    record = manifest[derived.MARKER]
    assert record['qualification_inherited'] is False and record['physics_changed'] is False
    assert manifest['physics_accepted'] is False and manifest['status'] == 'prepared_unvalidated'
    for name, row in old_manifest['files'].items():
        if name not in derived.CHANGED_FILES:
            assert digest(output/name) == row['sha256']
    task = manifest['task_id']
    for seed in (0, 7):
        _, old_scene, old_task, old_initial = task_inputs(source, task, seed=seed)
        _, scene, new_task, initial = task_inputs(output, task, seed=seed)
        assert {cloth['id']: cloth.pop('visual_profile') for cloth in scene['cloths']} == record['profiles']
        assert scene == old_scene and new_task == old_task and initial == old_initial
    pinned = read(output/'asset_file_hashes.json')
    for pin in record['profiles'].values():
        assert pinned[pin['path']] == {'sha256': pin['sha256']}
    assert inventory(source) == before


def test_profile_baked_for_another_cloth_or_dye_is_rejected(preparation, tmp_path):
    scene = read(SHIRT/'scene.json')
    cloth = scene['cloths'][0]
    other = fixture_profile(tmp_path/'other_mesh', cloth, cloth_geometry_sha256='0'*64)
    with pytest.raises(ValueError, match='different cloth'):
        preparation(SHIRT, 'wrong_mesh', {cloth['id']: other})
    tinted = fixture_profile(tmp_path/'other_dye', cloth, dye_linear_rgb=[.2, .2, .2])
    with pytest.raises(ValueError, match='different cloth'):
        preparation(SHIRT, 'wrong_dye', {cloth['id']: tinted})
    with pytest.raises(ValueError, match='existing cloth ids'):
        preparation(SHIRT, 'unknown_cloth', {'sock': other})


@pytest.mark.parametrize('fault', ['physics', 'scorer', 'qualification', 'base_pin', 'swapped_profile'])
def test_rehashed_candidate_cannot_escape_narrow_derivation(preparation, tmp_path, fault):
    output, manifest = preparation(SHIRT)
    if fault == 'physics':
        scene = read(output/'scene.json')
        scene['cloths'][0]['friction'] = .9
        dump(output/'scene.json', scene)
    elif fault == 'scorer':
        with (output/'evaluator/sequence_evaluator.py').open('a') as handle:
            handle.write('\n# changed scorer\n')
    elif fault == 'qualification':
        manifest[derived.MARKER]['qualification_inherited'] = True
    elif fault == 'base_pin':
        manifest[derived.MARKER]['base_release']['manifest_sha256'] = '0'*64
    else:
        scene = read(output/'scene.json')
        replacement = fixture_profile(tmp_path/'replacement', read(SHIRT/'scene.json')['cloths'][0])
        scene['cloths'][0]['visual_profile'] = {'path': str(replacement),
            'sha256': hashlib.sha256(replacement.read_bytes()).hexdigest()}
        dump(output/'scene.json', scene)
    for name in manifest['files']:
        manifest['files'][name]['sha256'] = digest(output/name)
        manifest['files_sha256'][name] = manifest['files'][name]['sha256']
    manifest['tasks'][0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
    dump(output/'release_manifest.json', manifest)
    with pytest.raises(ValueError):
        validate(output)


def test_existing_destinations_and_nested_candidates_are_refused(preparation):
    before = inventory(SHIRT)
    output, _ = preparation(SHIRT)
    candidate_before = inventory(output)
    with pytest.raises(ValueError, match='fresh independent'):
        derived.prepare(SHIRT, output, {})
    with pytest.raises(ValueError, match='without fabric profiles'):
        derived.prepare(output, output.parent/'nested_candidate', {})
    assert inventory(SHIRT) == before and inventory(output) == candidate_before


GUARD_PROBE = '''
import sys
from scripts.maintenance import freeze_project_sources as freezer
class Stop(Exception):
    pass
def probe(project, *, required_modules, entrypoint_file):
    raise Stop([name for name in required_modules if name not in sys.modules])
freezer.verify_if_snapshot = probe
try:
    {call}
except Stop as stop:
    print('MISSING', stop.args[0])
'''


@pytest.mark.parametrize('call', [
    "import scripts.assets.bake_laundry_fabric_textures as tool; "
    "sys.argv = ['bake', '--output', '/nonexistent', '--cloth', 'a', '/nonexistent', '0']; tool.main()",
    "from scenes.workrooms.laundry_room.fabric_visual_release import prepare; "
    "prepare('/nonexistent', '/nonexistent_candidate', {})"])
def test_frozen_entrypoints_load_every_module_their_snapshot_guard_requires(call):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c', GUARD_PROBE.format(call=call)], cwd=root,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'MISSING []'


def test_composition_adds_only_the_explicit_pins(monkeypatch):
    from scenes.workrooms.laundry_room import assets
    metadata = assets.candidate_metadata('towel_folding')
    monkeypatch.setattr(composition, 'load_asset_manifest', lambda *_: ({'fixture': True}, deepcopy(metadata)))
    plain = composition.build_task_inputs('towel_folding', asset_manifest='fixture')
    pin = {'path': '/fixture/profile.json', 'sha256': '1'*64}
    pinned = composition.build_task_inputs('towel_folding', asset_manifest='fixture',
                                           cloth_visual_profiles={'towel': pin})
    assert pinned['scene']['cloths'][0].pop('visual_profile') == pin
    assert pinned == plain
    with pytest.raises(ValueError, match='existing cloth ids'):
        composition.build_task_inputs('towel_folding', asset_manifest='fixture', cloth_visual_profiles={'sock': pin})
