"""Paired-robot authoring invariants and actual generated appearance pixels."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

from scenes.workrooms.parameterized.appearance import render_appearance, write_appearance
from scenes.workrooms.parameterized.spec import (
    FAMILIES, canonical_hash, rejection_record, sample_parameters, validate_parameters,
)


def parameters(seed, **kwargs):
    return sample_parameters('kitchen', 'breakfast_serving', 'single_arm', seed, **kwargs)


@pytest.mark.parametrize('family', FAMILIES)
def test_canonical_and_nine_distinct_stratified_physics(family):
    zero = sample_parameters(family, 'example_task', 'single_arm', 0)
    assert zero['layout'] == {'translation_xy_m': [0., 0.], 'yaw_rad': 0.}
    assert all(value == 1. for value in zero['physics'].values())
    assert zero['appearance']['roughness'] is None
    rows = [sample_parameters(family, 'example_task', 'single_arm', seed) for seed in range(1, 10)]
    assert {row['appearance']['pattern_id'] for row in rows} == set(range(1, 10))
    for row in rows:
        assert .04 <= math.hypot(*row['layout']['translation_xy_m']) <= .15
        assert math.radians(15) <= abs(row['layout']['yaw_rad']) <= math.radians(40)
        assert row['physics']['mass_factor'] == row['physics']['density_factor']
        assert row['status'] == 'candidate'
        assert validate_parameters(json.loads(json.dumps(row)))
    for field, lower, upper in [('density_factor', .65, 1.45), ('friction_factor', .6, 1.4)]:
        values = sorted(row['physics'][field] for row in rows)
        assert lower <= values[0] < values[-1] <= upper
        assert min(b - a for a, b in zip(values, values[1:])) >= .6 * (upper - lower) / 9 - 2e-12


def test_rejected_layout_retry_keeps_both_physics_and_appearance():
    first, next_layout = parameters(4), parameters(4, attempt=1)
    assert first['layout'] != next_layout['layout']
    assert first['appearance'] == next_layout['appearance']
    assert first['physics'] == next_layout['physics']
    assert first['instance_sha256'] != next_layout['instance_sha256']
    original = deepcopy(first)
    receipt = rejection_record(first, ['outside shared support'], measurements={'edge_gap_m': -.02})
    assert receipt['status'] == 'rejected'
    assert receipt['instance_sha256'] == first['instance_sha256']
    assert first == original
    with pytest.raises(ValueError):
        rejection_record(first, [])


def test_installation_only_changes_layout_and_outer_identity():
    first = parameters(6)
    second = sample_parameters('kitchen', 'breakfast_serving', 'dual_opposed', 6)
    assert first['physics'] == second['physics']
    assert first['appearance'] == second['appearance']
    assert first['layout'] != second['layout']
    assert first['instance_sha256'] != second['instance_sha256']


def test_sampling_does_not_consume_or_depend_on_process_random_state():
    random.seed(1)
    before = random.getstate()
    expected = parameters(8)
    assert random.getstate() == before
    random.seed(999)
    assert parameters(8) == expected


@pytest.mark.parametrize('seed', [-1, 10, True, 1., '1'])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        parameters(seed)


@pytest.mark.parametrize('attempt', [-1, 10000, True, 1.5])
def test_invalid_attempt_is_rejected(attempt):
    with pytest.raises(ValueError):
        parameters(1, attempt=attempt)


def test_robot_and_tampered_parameters_cannot_enter_frozen_identity():
    with pytest.raises(TypeError):
        sample_parameters('kitchen', 'breakfast_serving', 'single_arm', 1, robot_version='new')
    with pytest.raises(ValueError):
        parameters(0, attempt=1)
    for mutate in (
        lambda row: row.update(robot_version='new'),
        lambda row: row['physics'].update(mass_factor=1.),
        lambda row: row['layout'].update(yaw_rad=0.),
        lambda row: row['parameter_specs']['layout.radius_m'].update(bounds=[.001, .01]),
    ):
        row = parameters(1)
        mutate(row)
        # Even rehashing a tampered record must not make it valid.
        row['instance_sha256'] = canonical_hash({k: v for k, v in row.items() if k != 'instance_sha256'})
        with pytest.raises(ValueError):
            validate_parameters(row)


def test_nine_patterns_have_large_pixel_contrast_and_independent_roughness(tmp_path):
    image_hashes, roughness_hashes, images = set(), set(), []
    for seed in range(1, 10):
        row = parameters(seed)
        record = write_appearance(tmp_path / str(seed), row)
        pixels = np.asarray(Image.open(record['path']))
        reproduced = render_appearance(row)
        assert np.array_equal(pixels, reproduced['albedo'])
        assert np.array_equal(np.asarray(Image.open(record['roughness_path'])), reproduced['roughness'])
        assert int(pixels.max()) - int(pixels.min()) >= 70
        # Both large regions occupy a material fraction, not isolated changed bytes.
        assert .15 < np.mean(pixels[..., 0] == pixels[..., 0].max()) < .85
        image_hashes.add(record['sha256'])
        roughness_hashes.add(record['roughness_sha256'])
        images.append(pixels)
    assert len(image_hashes) == len(roughness_hashes) == 9
    for a, first in enumerate(images):
        for second in images[a + 1:]:
            assert np.mean(np.any(first != second, axis=2)) > .15


def test_original_alpha_semantic_color_order_and_protected_label_pixels(tmp_path):
    rgba = np.zeros((80, 100, 4), dtype=np.uint8)
    rgba[..., :3] = [200, 80, 30]
    rgba[..., 3] = np.arange(100, dtype=np.uint8)
    rgba[10:25, 15:60, :3] = [255, 255, 255]
    source = tmp_path / 'source.png'
    Image.fromarray(rgba).save(source)
    protected = np.zeros((80, 100), dtype=bool)
    protected[10:25, 15:60] = True
    record = write_appearance(tmp_path / 'derived', parameters(5), source_albedo=source,
                              protected_mask=protected)
    actual = np.asarray(Image.open(record['path']))
    assert np.array_equal(actual[..., 3], rgba[..., 3])
    assert np.array_equal(actual[protected], rgba[protected])
    assert (actual[..., 0][~protected] > actual[..., 1][~protected]).all()
    assert (actual[..., 1][~protected] > actual[..., 2][~protected]).all()
    assert record['source_albedo_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert np.array_equal(np.asarray(Image.open(source)), rgba)


def test_canonical_is_noop_and_generated_files_are_immutable(tmp_path):
    target = tmp_path / 'canonical'
    record = write_appearance(target, parameters(0))
    assert record['canonical'] and not record['applied'] and not target.exists()
    target = tmp_path / 'variant'
    record = write_appearance(target, parameters(1), preserve_category_color=False)
    digest = hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_appearance(target, parameters(2))
    assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError):
        write_appearance(tmp_path, parameters(1), texture_name='../escape.png')
