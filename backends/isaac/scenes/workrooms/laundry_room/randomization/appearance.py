"""Seeded albedo print overlays preserving parent UVs, seams and physical data."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

from .common import (ASSET_ROOT, canonical_hash, checked_file, file_hash,
                     read_json, source_provenance, verify_provenance, write_json)

VERSION = 'laundry_albedo_print_v1'


def _parent(pin):
    from bench.simulation.isaacsim_backend.fabric_visuals import load_visual_profile
    profile = load_visual_profile(pin)
    raw = read_json(pin['path'])
    return raw, profile


def render_albedo(source, seed):
    """One low-contrast, deterministic printed pattern; unchanged pixel geometry."""
    import io
    import numpy as np
    from PIL import Image
    if type(seed) is not int or not 1 <= seed <= 9:
        raise ValueError('Only nonzero finite seeds create new appearance')
    with Image.open(source) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    height, width = rgb.shape[:2]
    # Material-space bands/checks/diagonals multiply the original albedo, retaining
    # the existing scan, garment seam/pocket contrast and all normal/roughness maps.
    y, x = np.ogrid[:height, :width]
    period = max(16, round(min(height, width)/(11+seed)))
    phase = seed*period//7
    if seed % 3 == 0:
        mask = ((x+y+phase)//period) % 2
    elif seed % 3 == 1:
        mask = ((x+phase)//period+(y+phase)//period) % 2
    else:
        mask = ((x+phase)//period) % 2
    factor = 1.-(.12+.012*seed)*mask
    values = np.clip(np.floor(rgb.astype(np.float64)*factor[..., None]+.5), 0, 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(values, 'RGB').save(stream, format='PNG', compress_level=6)
    return stream.getvalue()


@lru_cache(maxsize=128)
def _expected_albedo_pixels(source, source_sha256, seed):
    """Mirror the frozen print's exact pixel arithmetic, without encoding a PNG.

    Only a pure expected RGB digest is cached. Parent and actual artifact file
    hashes are still checked by every profile validation before calling here.
    Keep render_albedo unchanged so existing author bytes remain reproducible.
    """
    import numpy as np
    from PIL import Image
    checked_file(source, source_sha256)
    if type(seed) is not int or not 1 <= seed <= 9:
        raise ValueError('Only nonzero finite seeds create new appearance')
    with Image.open(source) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    height, width = rgb.shape[:2]
    y, x = np.ogrid[:height, :width]
    period = max(16, round(min(height, width)/(11+seed)))
    phase = seed*period//7
    if seed % 3 == 0:
        mask = ((x+y+phase)//period) % 2
    elif seed % 3 == 1:
        mask = ((x+phase)//period+(y+phase)//period) % 2
    else:
        mask = ((x+phase)//period) % 2
    factor = 1.-(.12+.012*seed)*mask
    values = np.clip(np.floor(rgb.astype(np.float64)*factor[..., None]+.5), 0, 255).astype(np.uint8)
    return (width, height), hashlib.sha256(values.tobytes()).hexdigest()


def _validate_albedo_pixels(source, source_sha256, actual, actual_sha256, seed):
    from PIL import Image
    from bench.isaacsim.validation_epoch import semantic_reuse_enabled
    # Do not make either file check conditional on the expected-digest cache.
    checked_file(source, source_sha256)
    checked_file(actual, actual_sha256)
    # PNG readers need not expose newer color chunks (for example cICP) in
    # image.info. The author emits only these three structural chunk types.
    with Path(actual).open('rb') as stream:
        if stream.read(8) != b'\x89PNG\r\n\x1a\n':
            raise ValueError('Seed albedo must be an unannotated RGB PNG')
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError('Incomplete seed albedo PNG')
            length, kind = int.from_bytes(header[:4], 'big'), header[4:]
            if kind not in (b'IHDR', b'IDAT', b'IEND'):
                raise ValueError('Seed albedo must not add ancillary or color metadata')
            if kind == b'IHDR':
                ihdr = stream.read(length) if length == 13 else b''
                if len(ihdr) != 13 or ihdr[8:] != b'\x08\x02\x00\x00\x00':
                    raise ValueError('Seed albedo requires original 8-bit RGB PNG encoding')
            else:
                stream.seek(length, 1)
            if len(stream.read(4)) != 4:
                raise ValueError('Incomplete seed albedo PNG chunk')
            if kind == b'IEND':
                if length or stream.read(1):
                    raise ValueError('Unexpected bytes after seed albedo PNG')
                break
    expected_pixels = (_expected_albedo_pixels if semantic_reuse_enabled()
                       else _expected_albedo_pixels.__wrapped__)
    size, expected = expected_pixels(source, source_sha256, seed)
    with Image.open(actual) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != size:
            raise ValueError('Seed albedo must retain exact RGB PNG dimensions')
        image.load()
        if image.info:
            raise ValueError('Seed albedo must not add color, ICC, gamma or other metadata')
        if hashlib.sha256(image.tobytes()).hexdigest() != expected:
            raise ValueError('Albedo pixels differ from the deterministic parent-atlas print')


def prepare_profile(parent_pin, seed, output, *, parameters=None):
    """Fresh immutable appearance; unchanged maps link to their frozen originals."""
    if parameters is not None:
        from .parameterized_appearance import prepare_profile as prepare_parameterized
        return prepare_parameterized(parent_pin, seed, output, parameters)
    output = Path(output).resolve()
    if not output.is_relative_to(ASSET_ROOT) or output.exists():
        raise ValueError('Create appearance only in a fresh shared laundry asset directory')
    provenance = source_provenance()
    parent, resolved = _parent(parent_pin)
    albedo = render_albedo(resolved['textures']['albedo'], seed)
    output.mkdir(parents=True)
    try:
        (output/'albedo.png').write_bytes(albedo)
        profile = deepcopy(parent)
        profile['id'] = parent['id']+'_seed_'+str(seed)
        profile['textures']['albedo'] = 'albedo.png'
        profile['texture_sha256']['albedo'] = file_hash(output/'albedo.png')
        for role, source in resolved['textures'].items():
            if role != 'albedo':
                (output/profile['textures'][role]).symlink_to(Path(source).resolve())
        marker = {'schema': VERSION, 'seed': seed, 'parent_profile': deepcopy(parent_pin),
                  'changed_texture_roles': ['albedo'], 'physical_parameters_changed': False,
                  'uv_and_render_geometry_changed': False,
                  'scope': 'Procedural printed albedo multiplier over the original licensed fabric atlas; seams, pockets, normal and roughness maps retained.'}
        profile['seeded_appearance'] = marker
        write_json(output/'profile.json', profile)
        write_json(output/'SOURCE.json', {**marker, **provenance,
            'parent_files_sha256': {parent_pin['path']: parent_pin['sha256'],
                **{p: parent['texture_sha256'][r] for r, p in resolved['textures'].items()}},
            'license_scope': 'Original texture/source/license descriptors retained in parent profile; the new procedural modulation is project-authored CC0.'})
        pin = {'path': str(output/'profile.json'), 'sha256': file_hash(output/'profile.json')}
        validate_profile(pin, parent_pin, seed)
        return pin
    except BaseException:
        # Preserve a failed authoring attempt for inspection; never replace it.
        raise


def validate_profile(pin, parent_pin, seed, *, assets=True, parameters=None):
    if parameters is not None:
        from .parameterized_appearance import validate_profile as validate_parameterized
        return validate_parameterized(pin, parent_pin, seed, parameters, assets=assets)
    parent, resolved_parent = _parent(parent_pin)
    profile, resolved = _parent(pin)
    marker = profile.get('seeded_appearance', {})
    expected = deepcopy(parent)
    expected.update(id=parent['id']+'_seed_'+str(seed))
    expected['textures']['albedo'] = 'albedo.png'
    expected['texture_sha256']['albedo'] = profile['texture_sha256']['albedo']
    expected['seeded_appearance'] = marker
    if (type(seed) is not int or seed not in range(1, 10) or profile != expected
            or marker != {'schema': VERSION, 'seed': seed, 'parent_profile': parent_pin,
                'changed_texture_roles': ['albedo'], 'physical_parameters_changed': False,
                'uv_and_render_geometry_changed': False,
                'scope': 'Procedural printed albedo multiplier over the original licensed fabric atlas; seams, pockets, normal and roughness maps retained.'}):
        raise ValueError('Appearance differences exceed the declared albedo-only variant')
    source_path = Path(pin['path']).parent/'SOURCE.json'
    source = read_json(source_path)
    verify_provenance(source)
    inventory = {parent_pin['path']: parent_pin['sha256'],
                 **{p: parent['texture_sha256'][r] for r, p in resolved_parent['textures'].items()}}
    if source.get('parent_files_sha256') != inventory or any(source.get(k) != v for k, v in marker.items()):
        raise ValueError('Appearance parent provenance differs')
    for role, path in resolved['textures'].items():
        if role != 'albedo' and file_hash(path) != parent['texture_sha256'][role]:
            raise ValueError('Appearance changed an undeclared texture role')
    _validate_albedo_pixels(resolved_parent['textures']['albedo'], parent['texture_sha256']['albedo'],
                            resolved['textures']['albedo'], profile['texture_sha256']['albedo'], seed)
    return {**inventory, pin['path']: pin['sha256'], str(source_path): file_hash(source_path),
            **{p: profile['texture_sha256'][r] for r, p in resolved['textures'].items()}}
