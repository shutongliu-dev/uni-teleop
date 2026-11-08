"""Desktop changes preserve task physics and never promote historical evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.isaacsim.release import digest, validate
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.desktop_catalog import RELEASES, bind_desktop_release
from scenes.workrooms.mechanical_assembly.desktop_dressing import ASSET_ROOT, ROOT, scene_dependencies, task_layer


@pytest.mark.parametrize('task_id', RELEASES)
def test_desktop_derivatives_preserve_task_reset_rules_and_scorer_bytes(task_id):
    selected = RELEASES[task_id]
    root = Path(selected['path'])
    assert digest(root/'release_manifest.json') == selected['manifest_sha256']
    manifest = validate(root)
    source = Path(manifest['desktop_derivation']['source_release'])
    assert digest(source/'release_manifest.json') == manifest['desktop_derivation']['source_release_sha256']
    previous = load_json(source/'release_manifest.json')
    for relative, row in previous['files'].items():
        if relative not in {'scene.json', 'source_provenance.json', 'asset_file_hashes.json'}:
            assert digest(root/relative) == row['sha256'] == digest(source/relative)
    scene = load_json(root/'scene.json')
    original = load_json(source/'scene.json')
    assert scene.pop('mechanical_desktop_dressing')['manifest_sha256'] == digest(ASSET_ROOT/'manifest.json')
    assert scene['task_layers'].pop() == task_layer()
    if 'task_layers' not in original:
        assert scene.pop('task_layers') == []
    assert scene == original
    assert manifest['physics_accepted'] is False


def test_desktop_layer_has_no_dynamic_bodies_joints_or_scoring_objects():
    from pxr import Usd, UsdPhysics
    layer = task_layer()
    stage = Usd.Stage.Open(layer['usd_path'])
    assert layer['rigid_bodies'] == layer['particle_sets'] == {}
    colliders = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.IsA(UsdPhysics.Joint)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(prim.GetPath()))
    metadata = load_json(ASSET_ROOT/'metadata.json')
    assert len(colliders) == metadata['collider_count'] == 122
    assert metadata['group_count'] == 31
    for row in metadata['groups']:
        bounds = np.asarray(row['local_bounds_m'])+row['initial_position_m']
        assert bounds[0, 0] > .86  # Entire central/near-side workspace stays clear.
        assert bounds[1, 0] < 1.15
        assert bounds[0, 1] > -.5 and bounds[1, 1] < .5
        assert bounds[0, 2] >= .78-1e-8


def test_dressing_requires_a_matching_frozen_declaration():
    assert scene_dependencies({}) == {}
    layer = task_layer()
    with pytest.raises(ValueError, match='manifest reference'):
        scene_dependencies({'task_layers': [layer]})
    scene = {'task_layers': [layer], 'mechanical_desktop_dressing': {
        'manifest_path': str(ASSET_ROOT/'manifest.json'),
        'manifest_sha256': digest(ASSET_ROOT/'manifest.json')}}
    assert layer['usd_path'] in scene_dependencies(scene)
    damaged = deepcopy(scene)
    damaged['task_layers'][0]['rigid_bodies'] = {'background_tool': {'path': ROOT+'/Fixtures/tool_caddy'}}
    with pytest.raises(ValueError, match='differs'):
        scene_dependencies(damaged)


def test_historical_scene_evidence_is_kept_separate_from_new_desktop_release():
    previous = {'task_id': 'parts_sorting', 'source_release': {'path': 'historical'},
                'objects': {}, 'qualification': {'native_preflight': True, 'manipulation': False}}
    current = bind_desktop_release(previous)
    assert current['qualification']['previous_release_qualification']['native_preflight'] is True
    assert current['qualification']['native_preflight'] is False
    assert current['source_release'] == RELEASES['parts_sorting']
    assert previous['source_release']['path'] == 'historical'
