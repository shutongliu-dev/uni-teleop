"""Nine visible render-only finishes for opt-in parameterized authoring.

Authors must exclude text/labels and semantic markings from changed materials,
or supply a protected mask. This module changes PNGs, never physical materials.
Use ``pattern_scale_m`` for UV scale; retaining an old microscopic UV repeat can
hide these motifs. Nothing is generated for canonical seed zero.
"""
import colorsys
import hashlib
from pathlib import Path

from .spec import validate_parameters

PATTERNS = ('vertical_bands', 'horizontal_bands', 'diagonal_bands',
            'large_checks', 'round_spots', 'concentric_rings',
            'chevrons', 'diamonds', 'cross_grid')


def _pattern(pattern_id, width, height):
    import numpy as np
    y, x = np.mgrid[:height, :width]
    u, v = (x + .5) / width, (y + .5) / height
    if pattern_id == 1:
        return np.floor(u * 4) % 2 == 0
    if pattern_id == 2:
        return np.floor(v * 4) % 2 == 0
    if pattern_id == 3:
        return np.floor((u + v) * 3) % 2 == 0
    if pattern_id == 4:
        return (np.floor(u * 4) + np.floor(v * 4)) % 2 == 0
    if pattern_id == 5:
        return (u * 3 % 1 - .5) ** 2 + (v * 3 % 1 - .5) ** 2 < .075
    if pattern_id == 6:
        return np.floor(np.hypot(u - .5, v - .5) * 8) % 2 == 0
    if pattern_id == 7:
        return np.floor(v * 4 + np.abs(u * 4 % 2 - 1)) % 2 == 0
    if pattern_id == 8:
        return np.abs(u * 3 % 1 - .5) + np.abs(v * 3 % 1 - .5) < .45
    if pattern_id == 9:
        return (u * 3 % 1 < .28) | (v * 3 % 1 < .28)
    raise ValueError('Noncanonical pattern must be 1 through 9')


def _source_pixels(source_albedo, base_color, size):
    import numpy as np
    from PIL import Image
    if source_albedo is not None and base_color is not None:
        raise ValueError('Choose source albedo or base color, not both')
    if source_albedo is not None:
        with Image.open(source_albedo) as source:
            if source.mode not in ('RGB', 'RGBA'):
                raise ValueError('Source albedo must be RGB or RGBA')
            pixels = np.asarray(source).copy()
    else:
        color = np.asarray(base_color if base_color is not None else [1., 1., 1.], dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or (color < 0).any() or (color > 1).any():
            raise ValueError('Base color must be three finite values from 0 to 1')
        pixels = np.broadcast_to(np.rint(color * 255).astype('uint8'), (size, size, 3)).copy()
    return pixels


def render_appearance(parameters, *, source_albedo=None,
                      preserve_category_color=True, base_color=None, size=256,
                      protected_mask=None):
    """Return exact albedo/roughness uint8 arrays without filesystem writes.

    ``protected_mask`` is a boolean H×W array; true pixels stay byte-identical.
    Source alpha remains unchanged. The source's color-space declaration must
    remain unchanged in the USD author; these are RGB pixel overlays, not a
    calibrated reflectance model. With no source/base color, preserved-color
    output is neutral and can be multiplied by the existing material color.
    """
    validate_parameters(parameters)
    if type(preserve_category_color) is not bool:
        raise ValueError('preserve_category_color must be boolean')
    if type(size) is not int or not 32 <= size <= 4096:
        raise ValueError('Texture size must be an integer from 32 to 4096')
    if parameters['canonical']:
        return {'canonical': True, 'albedo': None, 'roughness': None,
                'protected_pixel_count': 0}
    import numpy as np
    from PIL import Image
    options = parameters['appearance']
    pixels = _source_pixels(source_albedo, base_color, size)
    height, width = pixels.shape[:2]
    mask = _pattern(options['pattern_id'], width, height)
    original = pixels[..., :3].astype(float) / 255.
    contrast = options['contrast']
    if preserve_category_color:
        low = original * (1 - .85 * contrast)
        high = original + (1 - original) * (.55 * contrast)
    else:
        hue = options['hue_turns']
        first = np.asarray(colorsys.hsv_to_rgb(hue, .68, .95))
        second = np.asarray(colorsys.hsv_to_rgb((hue + .38) % 1, .55, .98))
        low = .2 * original + .8 * first
        high = .2 * original + .8 * second
    result = pixels.copy()
    result[..., :3] = np.rint(np.clip(np.where(mask[..., None], high, low), 0, 1) * 255).astype('uint8')
    protected_count = 0
    if protected_mask is not None:
        protected = np.asarray(protected_mask)
        if protected.dtype != np.bool_ or protected.shape != (height, width):
            raise ValueError('Protected mask must be a boolean array matching source pixels')
        result[protected] = pixels[protected]
        protected_count = int(protected.sum())
    # A separately sampled base roughness; its spatial pattern has no friction effect.
    rough = np.rint(np.clip(options['roughness'] + np.where(mask, .07, -.07), .05, .95) * 255).astype('uint8')
    return {'canonical': False, 'albedo': result, 'roughness': rough,
            'protected_pixel_count': protected_count}


def write_appearance(output_dir, parameters, *, source_albedo=None,
                     preserve_category_color=True, base_color=None, size=256,
                     texture_name='surface_finish.png', protected_mask=None):
    """Create parent directories and new PNGs, rejecting either existing file.

    The directory may already exist; the two output files must not. See
    ``render_appearance`` for read-only pixel reproduction and label protection.
    """
    name = Path(texture_name)
    if name.name != texture_name or name.suffix.lower() != '.png':
        raise ValueError('Texture name must be a PNG basename')
    rendered = render_appearance(parameters, source_albedo=source_albedo,
                                preserve_category_color=preserve_category_color,
                                base_color=base_color, size=size, protected_mask=protected_mask)
    if rendered['canonical']:
        return {'schema': 'parameterized_appearance.v1', 'canonical': True,
                'applied': False, 'pattern_id': 0, 'path': None, 'sha256': None,
                'roughness_path': None, 'roughness_sha256': None,
                'instance_sha256': parameters['instance_sha256']}
    from PIL import Image
    options = parameters['appearance']
    result, rough = rendered['albedo'], rendered['roughness']
    height, width = result.shape[:2]
    output = Path(output_dir)
    albedo_path = output / name
    roughness_path = output / (name.stem + '_roughness.png')
    if albedo_path.exists() or roughness_path.exists():
        raise FileExistsError('Appearance outputs are immutable; choose a new directory/name')
    output.mkdir(parents=True, exist_ok=True)
    for path, array in ((albedo_path, result), (roughness_path, rough)):
        with path.open('xb') as stream:
            Image.fromarray(array).save(stream, format='PNG')
    return {
        'schema': 'parameterized_appearance.v1', 'canonical': False, 'applied': True,
        'instance_sha256': parameters['instance_sha256'],
        'pattern_id': options['pattern_id'], 'pattern_name': PATTERNS[options['pattern_id'] - 1],
        'pattern_scale_m': options['pattern_scale_m'],
        'preserve_category_color': preserve_category_color,
        'protected_pixel_count': rendered['protected_pixel_count'], 'physical_materials_changed': False,
        'path': str(albedo_path.resolve()), 'sha256': hashlib.sha256(albedo_path.read_bytes()).hexdigest(),
        'roughness_path': str(roughness_path.resolve()),
        'roughness_sha256': hashlib.sha256(roughness_path.read_bytes()).hexdigest(),
        'source_albedo_sha256': hashlib.sha256(Path(source_albedo).read_bytes()).hexdigest()
        if source_albedo is not None else None,
        'width': width, 'height': height, 'license': 'CC0-1.0 for generated pattern; source license retained',
    }
