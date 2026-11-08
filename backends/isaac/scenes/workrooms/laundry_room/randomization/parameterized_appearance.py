"""Visible material-coordinate prints over the original licensed fabric atlas."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from . import common
from .common import checked_file, file_hash, read_json, verify_provenance, write_json

VERSION = 'laundry_parameterized_albedo_v1'


def render_atlas(parent, source, parameters):
    """Read-only exact atlas pixels; unchanged source UVs preserve garment detail."""
    import numpy as np
    from PIL import Image
    from scenes.workrooms.parameterized.appearance import render_appearance
    from scenes.workrooms.parameterized.spec import validate_parameters
    validate_parameters(parameters)
    if parameters['canonical']:
        raise ValueError('Canonical fabric keeps its exact original profile')
    with Image.open(source) as image:
        pixels = np.asarray(image.convert('RGB'), dtype=np.uint8)
    h, w = pixels.shape[:2]
    frame = parent['uv_frame']
    sx, sy = frame['size_xy_m']; ox, oy = frame['origin_xy_m']
    y, x = np.ogrid[:h, :w]
    u, v = (x+.5)/w, 1.-(y+.5)/h
    if frame.get('atlas') in ('two_sided', 'front_back'):
        u = np.where(u < .5, 2*u, 2*(1-u))
    period = float(parameters['appearance']['pattern_scale_m'])
    # Use the shared nine-pattern tile, sampled in physical rest coordinates.
    # Tiling only the printed multiplier leaves seams/pockets/text at their
    # original atlas coordinates; the source itself is never tiled or resized.
    tile = render_appearance(parameters, preserve_category_color=True, base_color=[1., 1., 1.], size=256)
    tx = np.floor(((ox+u*sx)/period % 1)*256).astype(int)
    ty = np.floor(((oy+v*sy)/period % 1)*256).astype(int)
    high = tile['albedo'][ty, tx, 0] == 255
    factor = np.where(high, 1., 1.-float(parameters['appearance']['contrast']))
    albedo = np.rint(pixels.astype(np.float64)*factor[..., None]).astype(np.uint8)
    return {'albedo': albedo, 'roughness': tile['roughness'][ty, tx]}


def _marker(parent_pin, parameters):
    return {'schema': VERSION, 'seed': parameters['seed'], 'parent_profile': deepcopy(parent_pin),
        'family': parameters['family'], 'task_id': parameters['task_id'],
        'appearance_parameters': deepcopy(parameters['appearance']),
        'appearance_stream_sha256': parameters['stream_sha256']['appearance'],
        'changed_texture_roles': ['albedo', 'roughness'],
        'physical_parameters_changed': False, 'uv_and_render_geometry_changed': False,
        'pattern_scale_m': parameters['appearance']['pattern_scale_m'],
        'scope': 'Nine distinct physical-scale prints over the unchanged fabric atlas. Original hue, seams, pockets, labels and normal map remain registered; albedo is only multiplied.'}


def prepare_profile(parent_pin, seed, output, parameters):
    from PIL import Image
    from .appearance import _parent
    from scenes.workrooms.parameterized.spec import validate_parameters
    validate_parameters(parameters)
    if parameters['seed'] != seed or parameters['family'] != 'laundry_room' or not seed:
        raise ValueError('Appearance requires the matching noncanonical laundry parameters')
    output = Path(output).resolve()
    if not output.is_relative_to(common.ASSET_ROOT) or output.exists():
        raise ValueError('Fresh shared laundry appearance directory required')
    provenance = common.source_provenance(parameterized=True)
    parent, resolved = _parent(parent_pin)
    arrays = render_atlas(parent, resolved['textures']['albedo'], parameters)
    output.mkdir(parents=True)
    profile = deepcopy(parent)
    profile['id'] = parent['id']+'_parameterized_seed_'+str(seed)
    for role, array in arrays.items():
        name = role+'.png'
        Image.fromarray(array).save(output/name, format='PNG')
        profile['textures'][role] = name
        profile['texture_sha256'][role] = file_hash(output/name)
    for role, source in resolved['textures'].items():
        if role not in arrays:
            (output/profile['textures'][role]).symlink_to(Path(source).resolve())
    marker = _marker(parent_pin, parameters)
    profile['seeded_appearance'] = marker
    write_json(output/'profile.json', profile)
    write_json(output/'SOURCE.json', {**marker, **provenance,
        'parent_files_sha256': {parent_pin['path']: parent_pin['sha256'],
            **{p: parent['texture_sha256'][role] for role, p in resolved['textures'].items()}},
        'license_scope': 'Original fabric source/license remain pinned; procedural print is project-authored CC0.'})
    pin = {'path': str(output/'profile.json'), 'sha256': file_hash(output/'profile.json')}
    validate_profile(pin, parent_pin, seed, parameters)
    return pin


@lru_cache(maxsize=128)
def _expected(parent_json, source, source_sha256, parameters_json):
    checked_file(source, source_sha256)
    arrays = render_atlas(json.loads(parent_json), source, json.loads(parameters_json))
    return {role: (array.shape, hashlib.sha256(array.tobytes()).hexdigest()) for role, array in arrays.items()}


def validate_profile(pin, parent_pin, seed, parameters, *, assets=True):
    from PIL import Image
    from bench.isaacsim.validation_epoch import semantic_reuse_enabled
    from .appearance import _parent
    from scenes.workrooms.parameterized.spec import validate_parameters
    validate_parameters(parameters)
    if parameters['seed'] != seed or parameters['family'] != 'laundry_room' or not seed:
        raise ValueError('Appearance parameter identity differs')
    parent, original = _parent(parent_pin)
    profile, resolved = _parent(pin)
    expected = deepcopy(parent)
    expected['id'] = parent['id']+'_parameterized_seed_'+str(seed)
    for role in ('albedo', 'roughness'):
        expected['textures'][role] = role+'.png'
        expected['texture_sha256'][role] = profile['texture_sha256'][role]
    marker = _marker(parent_pin, parameters)
    expected['seeded_appearance'] = marker
    if profile != expected:
        raise ValueError('Parameterized fabric changed undeclared source geometry or appearance')
    source_path = Path(pin['path']).parent/'SOURCE.json'
    source = read_json(source_path); verify_provenance(source)
    if set(source['author_sources_sha256']) != set(common.PARAMETERIZED_SOURCE_FILES):
        raise ValueError('Parameterized fabric source inventory is incomplete')
    inventory = {parent_pin['path']: parent_pin['sha256'],
                 **{p: parent['texture_sha256'][r] for r, p in original['textures'].items()}}
    if source.get('parent_files_sha256') != inventory or any(source.get(k) != v for k, v in marker.items()):
        raise ValueError('Parameterized fabric source provenance differs')
    from scenes.workrooms.parameterized.spec import sample_parameters
    visual_parameters = sample_parameters('laundry_room', parameters['task_id'], 'single_arm', seed)
    expected_function = _expected if semantic_reuse_enabled() else _expected.__wrapped__
    expected_pixels = expected_function(json.dumps(parent, sort_keys=True), original['textures']['albedo'],
        parent['texture_sha256']['albedo'], json.dumps(visual_parameters, sort_keys=True))
    for role, path in resolved['textures'].items():
        checked_file(path, profile['texture_sha256'][role])
        if role in expected_pixels:
            shape, expected_sha = expected_pixels[role]
            with Image.open(path) as image:
                if (image.format != 'PNG' or image.mode != ('RGB' if role == 'albedo' else 'L')
                        or image.size != (shape[1], shape[0]) or image.info
                        or hashlib.sha256(image.tobytes()).hexdigest() != expected_sha):
                    raise ValueError('Parameterized fabric pixels differ from frozen recipe')
        elif file_hash(path) != parent['texture_sha256'][role]:
            raise ValueError('Parameterized fabric changed an undeclared texture role')
    return {**inventory, pin['path']: pin['sha256'], str(source_path): file_hash(source_path),
            **{p: profile['texture_sha256'][r] for r, p in resolved['textures'].items()}}
