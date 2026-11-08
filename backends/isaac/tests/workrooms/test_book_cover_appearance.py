"""Real pinned cover pixels and USD opinions, without a simulator or asset edits."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scenes.workrooms.parameterized.spec import sample_parameters
from scenes.workrooms.study import book_cover_appearance as cover


def parameters(seed):
    return sample_parameters('study', 'book_stacking', 'single_arm', seed)


@pytest.mark.parametrize('book_id', cover.BOOK_IDS)
@pytest.mark.parametrize('face', cover.FACES)
def test_original_ink_and_authored_semantics_are_exact(book_id, face):
    result = cover.book_cover_pixels(parameters(9), book_id, face)
    with Image.open(result['source_path']) as image:
        original = np.asarray(image.convert('RGB'))
    mask = result['protected_mask']; changed = np.any(result['albedo'] != original, axis=2)
    assert np.array_equal(result['albedo'][mask], original[mask])
    assert changed.sum() > .25*mask.size
    # Independent check against actual decoded ink, not just the mask recipe:
    # all light glyphs, borders, illustrations and barcode panel stay exact.
    ink = original.min(axis=2) > 135
    assert ink.sum() > 1000
    assert np.all(mask[ink])
    if face == 'front':
        assert np.all(mask[730:1171, 150:875])  # Full original illustration, including fill.
    if face == 'back':
        assert np.all(mask[1320:1421, 634:926])  # Includes dark barcode stripes.
    if face == 'spine':
        assert np.all(mask[1202:1419, 24:233])  # Includes dark category/library text.


def test_canonical_unchanged_and_nine_distinct_patterns():
    canonical = cover.book_cover_pixels(parameters(0), 'robotics', 'front')
    assert canonical['changed_pixel_count'] == 0
    hashes = []
    for seed in range(1, 10):
        result = cover.book_cover_pixels(parameters(seed), 'robotics', 'front')
        hashes.append(hashlib.sha256(result['albedo'].tobytes()).hexdigest())
    assert len(set(hashes)) == 9


def stage_with_book():
    from pxr import Usd
    from scenes.workrooms.study.book_assets import author_book
    stage = Usd.Stage.CreateInMemory()
    author_book(stage, '/Book', 'robotics')
    return Usd.Stage.Open(stage.Flatten())


def properties(stage):
    # Every authored/composed attribute and relationship, including all UV,
    # geometry, mass/inertia, collision and physics bindings.
    return {str(p.GetPath()): (p.GetTypeName(), str(p.GetCustomData()),
        {a.GetName(): (str(a.Get()), tuple(map(str, a.GetConnections())), str(a.GetAllMetadata()))
         for a in p.GetAttributes()},
        {r.GetName(): tuple(map(str, r.GetTargets())) for r in p.GetRelationships()})
        for p in stage.Traverse()}


def test_only_three_albedo_inputs_change_not_uv_physics_or_bindings(tmp_path):
    stage = stage_with_book(); before = properties(stage)
    result = cover.apply_book_cover_background(stage, {'robotics': '/Book'}, parameters(9), tmp_path/'new')
    assert len(result['artifacts']) == 3
    assert result['pattern_scale_m_applied'] is False
    assert result['sampled_pattern_scale_m'] == parameters(9)['appearance']['pattern_scale_m']
    after = properties(stage)
    for artifact in result['artifacts']:
        prim, attr = artifact['texture_file_attribute'].rsplit('.', 1)
        assert before[prim][2][attr] != after[prim][2][attr]
        after[prim][2][attr] = before[prim][2][attr]
        with Image.open(artifact['path']) as image:
            pixels = np.asarray(image)
        expected = cover.book_cover_pixels(parameters(9), 'robotics', artifact['face'])
        assert np.array_equal(pixels, expected['albedo'])
    assert after == before
    with pytest.raises(ValueError, match='pinned original'):
        cover.apply_book_cover_background(stage, {'robotics': '/Book'}, parameters(9), tmp_path/'retry')
    assert not (tmp_path/'retry').exists()


def test_canonical_no_mutation_or_outputs(tmp_path):
    stage = stage_with_book(); before = stage.GetRootLayer().ExportToString()
    result = cover.apply_book_cover_background(stage, {'robotics': '/Book'}, parameters(0), tmp_path/'zero')
    assert result['canonical'] and not (tmp_path/'zero').exists()
    assert stage.GetRootLayer().ExportToString() == before


def test_unknown_library_uv_and_actor_fail_closed(tmp_path):
    from pxr import UsdGeom
    stage = stage_with_book()
    uv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Book/Print/front')).GetPrimvar('st')
    uv.Set([(0, 0), (0, 1), (1, 1), (1, 0)])
    with pytest.raises(ValueError, match='UV convention'):
        cover.apply_book_cover_background(stage, {'robotics': '/Book'}, parameters(9), tmp_path/'uv')
    assert not (tmp_path/'uv').exists()
    with pytest.raises(ValueError, match='known dynamic'):
        cover.apply_book_cover_background(stage, {'wrong': '/Missing'}, parameters(9), tmp_path/'actor')
    fake = tmp_path/'fake'; fake.mkdir(); (fake/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='pinned books/v1'):
        cover.book_cover_pixels(parameters(9), 'robotics', 'front', bundle=fake)
    with pytest.raises(ValueError, match='limited to'):
        cover.book_cover_pixels(sample_parameters('study', 'reading_stand', 'single_arm', 1), 'robotics', 'front')
