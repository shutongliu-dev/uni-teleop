"""Full-byte mode recomputes expected pixels without clearing legacy caches."""
from copy import deepcopy
from functools import lru_cache
import json

import numpy as np
from PIL import Image
import pytest

from bench.isaacsim.validation_epoch import FullByteValidation, ValidationEpoch
from scenes.workrooms.laundry_room.randomization import appearance, common
from scenes.workrooms.laundry_room.randomization import parameterized_appearance as parameterized
from scenes.workrooms.parameterized.spec import sample_parameters


def sentinel_cache(function, poison):
    """Warm a deliberately wrong cache entry, then make real builds correct."""
    state = {'poison': True, 'builds': 0}

    @lru_cache(maxsize=8)
    def cached(*args):
        state['builds'] += 1
        result = function(*args)
        return poison(result) if state['poison'] else result

    return cached, state


def check_modes(validate, state, paths):
    before = {str(path): path.read_bytes() for path in paths}
    # These two modes must retain the same already-warmed cache entry.
    with pytest.raises(ValueError, match='pixels differ'):
        validate()
    with ValidationEpoch('strict-retains-expected-pixel-cache'):
        with pytest.raises(ValueError, match='pixels differ'):
            validate()
    assert state['builds'] == 1
    # Each call must execute the real pixel formula, not consume the sentinel.
    with FullByteValidation('fresh-expected-pixel-formula') as scope:
        validate()
        validate()
    assert state['builds'] == 3
    assert scope.summary()['closed_verified'] is True
    assert scope.summary()['hash_reuses'] == scope.summary()['derivation_reuses'] == 0
    # The full-byte path must not clear or overwrite the default cache.
    with pytest.raises(ValueError, match='pixels differ'):
        validate()
    assert state['builds'] == 3
    assert {str(path): path.read_bytes() for path in paths} == before


def test_historical_albedo_fullbytes_bypasses_warm_cache_and_preserves_png(tmp_path, monkeypatch):
    pixels = (np.arange(83*97*3).reshape(83, 97, 3)*19 % 256).astype(np.uint8)
    source, actual = tmp_path/'source.png', tmp_path/'actual.png'
    Image.fromarray(pixels).save(source)
    actual.write_bytes(appearance.render_albedo(source, 5))
    arguments = (source, common.file_hash(source), 5)
    cached, state = sentinel_cache(appearance._expected_albedo_pixels.__wrapped__,
                                  lambda result: (result[0], '0'*64))
    cached(*arguments)
    state['poison'] = False
    monkeypatch.setattr(appearance, '_expected_albedo_pixels', cached)
    actual_sha = common.file_hash(actual)
    validate = lambda: appearance._validate_albedo_pixels(*arguments[:2], actual, actual_sha, 5)
    check_modes(validate, state, (source, actual))


def test_parameterized_albedo_and_roughness_fullbytes_recompute_real_pixels(tmp_path, monkeypatch):
    source = tmp_path/'source.png'
    pixels = np.full((80, 128, 3), [45, 100, 180], dtype=np.uint8)
    pixels[32:35] = 0
    Image.fromarray(pixels).save(source)
    parent = {'id': 'synthetic_fabric',
              'uv_frame': {'origin_xy_m': [0., 0.], 'size_xy_m': [.6, .4], 'atlas': 'two_sided'},
              'textures': {'albedo': source.name},
              'texture_sha256': {'albedo': common.file_hash(source)}}
    parent_path = tmp_path/'parent.json'
    common.write_json(parent_path, parent)
    parent_pin = {'path': str(parent_path), 'sha256': common.file_hash(parent_path)}
    parameters = sample_parameters('laundry_room', 'shirt_folding', 'single_arm', 5)
    output = tmp_path/'derived'; output.mkdir()
    arrays = parameterized.render_atlas(parent, source, parameters)
    profile = deepcopy(parent)
    profile['id'] += '_parameterized_seed_5'
    for role, array in arrays.items():
        path = output/(role+'.png')
        Image.fromarray(array).save(path)
        profile['textures'][role] = path.name
        profile['texture_sha256'][role] = common.file_hash(path)
    marker = parameterized._marker(parent_pin, parameters)
    profile['seeded_appearance'] = marker
    profile_path = output/'profile.json'
    common.write_json(profile_path, profile)
    pin = {'path': str(profile_path), 'sha256': common.file_hash(profile_path)}
    common.write_json(output/'SOURCE.json', {**marker,
        'parent_files_sha256': {str(parent_path): parent_pin['sha256'], str(source): parent['texture_sha256']['albedo']},
        'author_sources_sha256': {name: '0'*64 for name in common.PARAMETERIZED_SOURCE_FILES}})

    def parent_reader(reference):
        path = common.checked_file(reference['path'], reference['sha256'])
        raw = common.read_json(path)
        return raw, {'textures': {role: str(path.parent/name) for role, name in raw['textures'].items()}}

    # Synthetic provenance only; the real source/PNG hash and pixel checks run.
    monkeypatch.setattr(appearance, '_parent', parent_reader)
    monkeypatch.setattr(parameterized, 'verify_provenance', lambda value: None)
    arguments = (json.dumps(parent, sort_keys=True), str(source), parent['texture_sha256']['albedo'],
                 json.dumps(parameters, sort_keys=True))

    def poison(result):
        value = deepcopy(result)
        value['roughness'] = (value['roughness'][0], '0'*64)
        return value

    cached, state = sentinel_cache(parameterized._expected.__wrapped__, poison)
    cached(*arguments)
    state['poison'] = False
    monkeypatch.setattr(parameterized, '_expected', cached)
    validate = lambda: parameterized.validate_profile(pin, parent_pin, 5, parameters)
    check_modes(validate, state, (source, parent_path, *output.iterdir()))
