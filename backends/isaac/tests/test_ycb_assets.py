"""Asset acquisition/integrity and convex proxy checks; no network or simulator."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import trimesh

from scripts.assets.prepare_ycb_assets import (
    MAX_COLLISION_VERTICES, SELECTION, collision_slabs, compound_mass_properties,
    convex_proxy, extract_verified, fetch_verified, sha256, verify_catalog,
    write_new_or_identical,
)


def make_archive(path, members):
    with tarfile.open(path, 'w:gz') as stream:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            if kind == 'file':
                info.size = 4
                stream.addfile(info, BytesIO(b'data'))
            else:
                info.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                info.linkname = '../escape'
                stream.addfile(info)


@pytest.mark.parametrize('members', [
    [('cube/../escape', 'file')], [('/cube/file', 'file')], [('other/file', 'file')],
    [('cube/link', 'symlink')], [('cube/link', 'hardlink')],
    [('cube/file', 'file'), ('cube/file', 'file')], [('cube\\escape', 'file')],
])
def test_rejects_archive_traversal_links_and_duplicates_before_writing(tmp_path, members):
    archive = tmp_path / 'bad.tgz'
    make_archive(archive, members)
    output = tmp_path / 'extracted'
    with pytest.raises(ValueError):
        extract_verified(archive, output, 'cube')
    assert not output.exists()


def test_bounded_extraction_is_idempotent_and_never_overwrites(tmp_path):
    archive = tmp_path / 'ok.tgz'
    make_archive(archive, [('cube/google_16k/mesh.obj', 'file')])
    root = tmp_path / 'extracted'
    assert extract_verified(archive, root, 'cube') == 4
    before = (root / 'cube/google_16k/mesh.obj').read_bytes()
    extract_verified(archive, root, 'cube')
    assert (root / 'cube/google_16k/mesh.obj').read_bytes() == before
    (root / 'cube/google_16k/mesh.obj').write_bytes(b'drift')
    with pytest.raises(FileExistsError):
        extract_verified(archive, root, 'cube')
    assert (root / 'cube/google_16k/mesh.obj').read_bytes() == b'drift'


def test_expansion_budget_is_checked_before_extraction(tmp_path):
    archive = tmp_path / 'ok.tgz'
    make_archive(archive, [('cube/google_16k/mesh.obj', 'file')])
    with pytest.raises(ValueError, match='byte budget'):
        extract_verified(archive, tmp_path / 'output', 'cube', max_bytes=3)
    assert not (tmp_path / 'output').exists()


def test_existing_download_requires_exact_pinned_bytes_without_network(tmp_path):
    path = tmp_path / 'source.tgz'
    path.write_bytes(b'original')
    row = fetch_verified('https://invalid.example/unreachable', path, sha256(path))
    assert row['bytes'] == 8
    with pytest.raises(ValueError, match='hash mismatch'):
        fetch_verified('https://invalid.example/unreachable', path, 'a' * 64)
    assert path.read_bytes() == b'original'


def test_original_versioned_files_cannot_be_replaced(tmp_path):
    path = tmp_path / 'version/file'
    write_new_or_identical(path, 'one')
    write_new_or_identical(path, 'one')
    with pytest.raises(FileExistsError):
        write_new_or_identical(path, 'two')
    assert path.read_text() == 'one'


def test_dense_collision_proxy_is_closed_convex_bounded_and_conservative():
    source = trimesh.creation.icosphere(subdivisions=3, radius=.04)
    proxy, error = convex_proxy(source.vertices)
    assert proxy.is_volume and proxy.is_convex
    assert len(proxy.vertices) <= MAX_COLLISION_VERTICES
    planes = ConvexHull(proxy.vertices).equations
    assert np.max(source.vertices @ planes[:, :3].T + planes[:, 3]) < 1e-9
    assert 0 < error < .003


def test_curved_compound_does_not_fill_the_global_convex_gap():
    boxes = []
    for angle in np.linspace(-1.1, 1.1, 9):
        box = trimesh.creation.box(extents=[.023, .024, .02])
        box.apply_translation([.09 * np.sin(angle), .09 * (1-np.cos(angle)), 0.])
        boxes.append(box)
    curved = trimesh.util.concatenate(boxes)
    pieces, report = collision_slabs(curved, 10)
    assert len(pieces) == 10
    assert sum(piece.volume for piece in pieces) < curved.convex_hull.volume * .8
    # The central hollow region lies in the whole hull, outside every slab.
    gap = np.array([0., .035, 0.])
    global_planes = ConvexHull(curved.vertices).equations
    assert np.max(global_planes[:, :3] @ gap + global_planes[:, 3]) < 0
    assert all(np.max(ConvexHull(piece.vertices).equations[:, :3] @ gap
                      + ConvexHull(piece.vertices).equations[:, 3]) > 0 for piece in pieces)
    assert report['support_error_is_not_hausdorff_bound'] is True


def test_compound_inertia_uses_parallel_axis_theorem_at_derived_com():
    a = trimesh.creation.box(extents=[.02, .04, .06])
    b = a.copy()
    a.apply_translation([-.03, 0., 0.])
    b.apply_translation([.03, 0., 0.])
    com, tensor = compound_mass_properties([a, b], .5)
    np.testing.assert_allclose(com, [0., 0., 0.], atol=1e-12)
    expected = .5 * np.array([(.04**2+.06**2)/12,
                              (.02**2+.06**2)/12+.03**2,
                              (.02**2+.04**2)/12+.03**2])
    np.testing.assert_allclose(np.diag(tensor), expected, atol=1e-12)
    assert np.linalg.eigvalsh(tensor).min() > 0


@pytest.mark.parametrize('mass', [0., -1., float('nan')])
def test_invalid_mass_cannot_generate_inertia(mass):
    with pytest.raises(ValueError):
        compound_mass_properties([trimesh.creation.box(extents=[1., 1., 1.])], mass)


def test_selection_has_twelve_pinned_sources_and_no_hollow_containers():
    assert len(SELECTION) == 12
    assert not any(x in name for name in SELECTION for x in ('mug', 'bowl', 'pitcher', 'cup'))
    assert SELECTION['011_banana'][1] > 1 and SELECTION['048_hammer'][1] > 1
    assert all(len(row[2]) == 64 and row[0] > 0 for row in SELECTION.values())


def test_catalog_schema_and_hash_verifier_reject_drift(tmp_path):
    mesh = trimesh.creation.box(extents=[.02, .03, .04])
    (tmp_path / 'mesh.obj').write_text(trimesh.exchange.obj.export_obj(mesh))
    asset = {'mesh_path': 'mesh.obj', 'sha256': sha256(tmp_path / 'mesh.obj'),
             'visual_parts': [], 'collision_parts': [{'mesh_path': 'mesh.obj', 'sha256': sha256(tmp_path / 'mesh.obj')}],
             'inertia_diag_kg_m2': [1e-5, 1e-5, 1e-5], 'bounds_m': mesh.bounds.tolist()}
    (tmp_path / 'manifest.json').write_text(json.dumps({'assets': {'test': asset}}))
    assert verify_catalog(tmp_path)['count'] == 1
    (tmp_path / 'mesh.obj').write_text('drift')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_catalog(tmp_path)
