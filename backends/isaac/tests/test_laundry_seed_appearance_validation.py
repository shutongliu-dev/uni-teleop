"""Pixel-exact fast validation keeps frozen files pinned and render semantics strict."""
from pathlib import Path
import io
import struct
import zlib

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from scenes.workrooms.laundry_room.randomization import appearance as a, common


@pytest.fixture
def pixels(tmp_path):
    values = (np.arange(107*131*3).reshape(107, 131, 3)*19 % 256).astype(np.uint8)
    source = tmp_path/'source.png'; Image.fromarray(values, 'RGB').save(source)
    actual = tmp_path/'actual.png'; actual.write_bytes(a.render_albedo(source, 5))
    a._expected_albedo_pixels.cache_clear()
    return source, actual


@pytest.mark.parametrize('seed', range(1, 10))
def test_pixel_formula_matches_unchanged_png_generator(pixels, seed):
    source, actual = pixels
    actual.write_bytes(a.render_albedo(source, seed))
    before = actual.read_bytes()
    a._validate_albedo_pixels(source, common.file_hash(source), actual, common.file_hash(actual), seed)
    assert actual.read_bytes() == before
    with Image.open(io.BytesIO(before)) as im:
        assert im.mode == 'RGB' and not im.info


@pytest.mark.parametrize('change', ['pixel', 'rgba', 'dimensions', 'jpeg', 'gamma', 'icc', 'srgb', 'cicp', 'text'])
def test_rehashed_visual_change_and_color_metadata_are_rejected(pixels, change):
    source, actual = pixels
    with Image.open(actual) as im:
        value = im.copy()
    if change == 'pixel':
        x = np.asarray(value).copy(); x[17, 22, 0] ^= 1
        Image.fromarray(x, 'RGB').save(actual)
    elif change == 'rgba':
        value.convert('RGBA').save(actual)
    elif change == 'dimensions':
        value.crop((0, 0, value.width-1, value.height)).save(actual)
    elif change == 'jpeg':
        value.save(actual, format='JPEG')
    elif change == 'icc':
        value.save(actual, icc_profile=b'nonempty-color-profile')
    else:
        metadata = PngImagePlugin.PngInfo()
        if change == 'gamma':
            metadata.add(b'gAMA', struct.pack('!I', 45455))
        elif change == 'srgb':
            metadata.add(b'sRGB', b'\x00')
        elif change == 'cicp':
            metadata.add(b'cICP', b'\x01\x0d\x00\x01')
        else:
            metadata.add_text('Description', 'additional render metadata')
        value.save(actual, pnginfo=metadata)
    # Updating the public file digest must not hide a changed rendered image.
    with pytest.raises(ValueError):
        a._validate_albedo_pixels(source, common.file_hash(source), actual, common.file_hash(actual), 5)


@pytest.mark.parametrize('which', ['parent', 'actual'])
def test_digest_cache_never_bypasses_frozen_input_file_hash(pixels, which):
    source, actual = pixels
    source_sha, actual_sha = common.file_hash(source), common.file_hash(actual)
    a._validate_albedo_pixels(source, source_sha, actual, actual_sha, 5)
    target = source if which == 'parent' else actual
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Frozen input differs'):
        a._validate_albedo_pixels(source, source_sha, actual, actual_sha, 5)


def test_sixteen_bit_rgb_cannot_hide_extra_precision_behind_pillow_rgb_mode(pixels):
    source, actual = pixels
    with Image.open(actual) as image:
        rgb = np.asarray(image).copy()
    height, width = rgb.shape[:2]
    high_precision = ((rgb.astype(np.uint16) << 8) | 137).astype('>u2')
    rows = b''.join(b'\x00'+row.tobytes() for row in high_precision)
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
    actual.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 16, 2, 0, 0, 0))
                       +chunk(b'IDAT', zlib.compress(rows))+chunk(b'IEND', b''))
    with Image.open(actual) as image:
        assert image.mode == 'RGB' and np.array_equal(np.asarray(image), rgb)
    with pytest.raises(ValueError, match='8-bit RGB'):
        a._validate_albedo_pixels(source, common.file_hash(source), actual, common.file_hash(actual), 5)


@pytest.mark.parametrize('task,cfg', [('towel_folding', 'single_arm'), ('shirt_folding', 'dual_opposed')])
def test_actual_author01_profile_is_accepted_without_changing_any_file(task, cfg):
    root = common.RELEASE_ROOT/'publication01/releases'/task/cfg
    if not (root/'release_manifest.json').is_file():
        pytest.skip('Optional actual frozen author01 publication is not available')
    manifest = common.read_json(root/'release_manifest.json')
    record = manifest['cases']['5']
    scene = common.read_json(root/record['scene'])
    parent_scene = common.read_json(Path(manifest['base_release']['path'])/'scene.json')
    pin, parent_pin = scene['cloths'][0]['visual_profile'], parent_scene['cloths'][0]['visual_profile']
    profile = common.read_json(pin['path'])
    paths = [Path(pin['path']), Path(parent_pin['path']),
             *(Path(pin['path']).parent/p for p in profile['textures'].values())]
    before = {str(p): common.file_hash(p) for p in paths}
    a.validate_profile(pin, parent_pin, 5)
    assert {str(p): common.file_hash(p) for p in paths} == before
