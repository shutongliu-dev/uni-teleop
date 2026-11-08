"""Opt-in background patterns for the pinned original study book covers.

This adapter never relaxes the generic text/cover exclusion. It recognizes the
six task books from books/v1 and edits only their existing front/back/spine
albedo file inputs. Original UVs, print geometry, other material inputs and all
physical properties are retained. It is not enabled by historical authors.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.parameterized.spec import validate_parameters
from scenes.workrooms.seed_geometry import file_hash

VERSION = 'study_book_cover_background_v1'
TASKS = ('book_stacking', 'book_retrieval', 'book_shelving')
BOOK_IDS = ('robotics', 'astronomy', 'botany', 'architecture', 'painting', 'literature')
FACES = ('front', 'back', 'spine')
DEFAULT_BUNDLE = Path(f'{DATA_ROOT}/assets/task_parts/study/books/v1')
MANIFEST_SHA256 = '4c718cdc28723cb7591ce244a56b8d62d1c2b3e313e8c06a004a47a19361e409'
JPEG_MARGIN_PIXELS = 16


def _library(bundle):
    path = Path(bundle).resolve()/'manifest.json'
    if file_hash(path) != MANIFEST_SHA256:
        raise ValueError('Cover masks only describe the pinned books/v1 library')
    manifest = load_json(path)
    font_path = next(Path(p) for p in manifest['source_dependencies']
                     if p.endswith('/NotoSansCJKsc-Regular.otf'))
    font_sha = manifest['source_dependencies'][str(font_path)]['sha256']
    if file_hash(font_path) != font_sha:
        raise ValueError('Original cover font changed')
    return manifest, font_path


def protected_mask(spec, face, font_path):
    """Conservative authored content boxes, expanded beyond JPEG block edges.

    Text boxes use the original font, size, anchor and positions from
    book_assets._textures. Whole illustration and barcode regions are protected,
    including their interior background. This is not color-based segmentation
    or OCR: every original title, mark, figure and category panel is retained.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    if spec['id'] not in BOOK_IDS or face not in FACES:
        raise ValueError('No reviewed semantic mask for this book/face')
    width = 256 if face == 'spine' else 1024
    image = Image.new('L', (width, 1536)); draw = ImageDraw.Draw(image)

    def text(xy, value, size, anchor=None):
        font = ImageFont.truetype(str(font_path), size)
        draw.rectangle(draw.textbbox(xy, value, font=font, anchor=anchor), fill=255)

    if face in ('front', 'back'):
        draw.rectangle((53, 53, 970, 1482), outline=255, width=2)
    if face == 'front':
        text((86, 102), spec['category_zh']+'  /  观物文库', 30)
        text((86, 242), spec['title'], min(91, int(840/max(1, len(spec['title'])))))
        draw.line((88, 398, 925, 398), fill=255, width=2)
        for i, line in enumerate(spec['subtitle'].split(' / ')):
            text((88, 434+i*47), line, 27)
        text((88, 595), spec['author']+' 著', 34)
        draw.rectangle((128, 712, 900, 1200), fill=255)
        text((88, 1365), spec['publisher'], 36)
        text((920, 1407), spec['library_code'], 26, 'ra')
    elif face == 'back':
        text((87, 124), '观物文库  /  '+spec['category_zh'], 32)
        text((87, 248), spec['title'], 65)
        paragraphs = ['从日常观察出发，重新认识身边的世界。',
            '以清晰的图示与细致的文字，记录发现的过程。',
            '在阅读、思考与实践之间，留下属于自己的笔记。',
            '愿每一次翻阅，都成为新的发现与灵感。']
        y = 430
        for paragraph in paragraphs:
            for start in range(0, len(paragraph), 21):
                text((90, y), paragraph[start:start+21], 34); y += 64
            y += 20
        draw.line((90, 1040, 930, 1040), fill=255, width=2)
        for i, line in enumerate([spec['author']+' 著  |  '+spec['edition'],
                '装帧：布面精装  ·  插图：观物书房', '馆藏编号  '+spec['library_code']]):
            text((90, 1094+i*61), line, 29)
        draw.rectangle((634, 1320, 925, 1420), fill=255)
        text((88, 1371), '观物书房', 34)
    else:
        draw.line((26, 57, 230, 57), fill=255, width=3)
        draw.line((26, 82, 230, 82), fill=255, width=2)
        text((128, 130), '观物', 42, 'ma')
        for i, char in enumerate(spec['title']):
            text((128, 298+i*99), char, 83, 'ma')
        text((128, 1085), spec['author'], 35, 'ma')
        draw.rectangle((24, 1202, 232, 1418), fill=255)
        text((128, 1462), '2026', 25, 'ma')
    return np.asarray(image.filter(ImageFilter.MaxFilter(2*JPEG_MARGIN_PIXELS+1))) > 0


def book_cover_pixels(parameters, book_id, face, *, bundle=DEFAULT_BUNDLE):
    """Reproduce one protected cover at its original resolution without writes."""
    from PIL import Image
    from scenes.workrooms.parameterized.appearance import render_appearance
    validate_parameters(parameters)
    if parameters['family'] != 'study' or parameters['task_id'] not in TASKS:
        raise ValueError('Book cover adapter is limited to the three library tasks')
    if book_id not in BOOK_IDS or face not in FACES:
        raise ValueError('Unreviewed book or face')
    manifest, font = _library(bundle); spec = manifest['books'][book_id]
    source = Path(spec['textures'][face]).resolve()
    expected = manifest['files'][str(source.relative_to(Path(bundle).resolve()))]['sha256']
    if file_hash(source) != expected:
        raise ValueError('Original cover image changed')
    with Image.open(source) as image:
        original = np.asarray(image.convert('RGB')).copy()
    mask = protected_mask(spec, face, font)
    if original.shape != (*mask.shape, 3):
        raise ValueError('Original cover dimensions differ from reviewed mask')
    rendered = render_appearance(parameters, source_albedo=source,
        preserve_category_color=True, protected_mask=mask)
    pixels = original if parameters['canonical'] else rendered['albedo']
    if not np.array_equal(pixels[mask], original[mask]):
        raise ValueError('Protected cover content changed')
    return {'albedo': pixels, 'protected_mask': mask, 'source_path': str(source),
        'source_sha256': expected, 'source_pixels_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
        'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'protected_pixel_count': int(mask.sum()), 'total_pixel_count': int(mask.size),
        'changed_pixel_count': int(np.any(pixels != original, axis=2).sum()),
        'protected_pixels_identical': True, 'font_path': str(font), 'font_sha256': file_hash(font)}


def apply_book_cover_background(stage, actors, parameters, output_dir, *, bundle=DEFAULT_BUNDLE):
    """Author only known albedo file opinions into an already isolated stage.

    The caller owns publication, source/dependency inventories and fresh native
    image review. No shader/UV recreation, roughness edit, layout or resampling
    happens here. Output creation is exclusive; original files remain intact.
    """
    from PIL import Image
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    validate_parameters(parameters)
    if parameters['family'] != 'study' or parameters['task_id'] not in TASKS:
        raise ValueError('Unsupported task for book cover adapter')
    if parameters['canonical']:
        return {'schema': VERSION, 'canonical': True, 'meshes': [], 'artifacts': []}
    if not stage.GetRootLayer().anonymous:
        raise ValueError('Book appearance must be authored in an isolated anonymous stage')
    manifest, _ = _library(bundle)
    if len(set(actors.values())) != len(actors):
        raise ValueError('Duplicate book actor paths')
    pending = []
    for name, path in actors.items():
        actor = stage.GetPrimAtPath(path)
        ident = actor.GetCustomDataByKey('study:book_id') if actor else None
        if ident not in BOOK_IDS or not actor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Expected a known dynamic library book: '+path)
        for face in FACES:
            prim = stage.GetPrimAtPath(path+'/Print/'+face)
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                raise ValueError('Expected visible original book print mesh')
            uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
            if (not uv or uv.GetInterpolation() != 'vertex' or list(uv.GetIndices() or []) or
                    list(uv.Get()) != [(0, 0), (1, 0), (1, 1), (0, 1)]):
                raise ValueError('Original cover UV convention changed')
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader = material.ComputeSurfaceSource()[0] if material else None
            if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
                raise ValueError('Unknown original cover shader')
            source = shader.GetInput('diffuseColor').GetConnectedSource()
            texture = UsdShade.Shader(source[0].GetPrim()) if source else None
            if not texture or texture.GetIdAttr().Get() != 'UsdUVTexture':
                raise ValueError('Original cover must have its known texture binding')
            if str(texture.GetPath()) != path+'/Looks/'+face+'/diffuseColor':
                raise ValueError('Cover shader must be local to the known book face')
            texture_file = texture.GetInput('file'); old = texture_file.Get()
            source_path = str(Path(old.resolvedPath or old.path).resolve())
            if source_path != manifest['books'][ident]['textures'][face]:
                raise ValueError('Cover binding is not the pinned original image')
            rendered = book_cover_pixels(parameters, ident, face, bundle=bundle)
            pending.append((name, ident, face, prim, texture_file, rendered))
    if not pending:
        raise ValueError('No known book covers selected')
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for index, (name, ident, face, prim, texture_file, rendered) in enumerate(pending):
        dest = output/f'{index:02d}_{ident}_{face}.png'
        mask_path = output/f'{index:02d}_{ident}_{face}_protected.png'
        with dest.open('xb') as stream: Image.fromarray(rendered['albedo']).save(stream, format='PNG')
        with mask_path.open('xb') as stream:
            Image.fromarray(rendered['protected_mask'].astype('uint8')*255).save(stream, format='PNG')
        texture_file.Set(Sdf.AssetPath(str(dest)))
        artifacts.append({k: v for k, v in rendered.items() if k not in ('albedo', 'protected_mask')} |
            {'actor_id': name, 'book_id': ident, 'face': face, 'mesh': str(prim.GetPath()),
             'texture_file_attribute': str(texture_file.GetAttr().GetPath()),
             'path': str(dest), 'sha256': file_hash(dest),
             'mask_path': str(mask_path), 'mask_file_sha256': file_hash(mask_path)})
    return {'schema': VERSION, 'canonical': False, 'instance_sha256': parameters['instance_sha256'],
        'library_manifest': {'path': str(Path(bundle).resolve()/'manifest.json'), 'sha256': MANIFEST_SHA256},
        'pattern_id': parameters['appearance']['pattern_id'], 'artifacts': artifacts,
        'pattern_domain': 'one_generated_pattern_field_per_original_face',
        'uv_policy': 'original_print_uv_unchanged',
        'sampled_pattern_scale_m': parameters['appearance']['pattern_scale_m'],
        'pattern_scale_m_applied': False,
        'scale_scope': 'Semantic print faces use normalized original UVs; sampled metric tile size remains applicable only to the other authored materials.',
        'meshes': [r['mesh'] for r in artifacts], 'uv_geometry_physics_unchanged': True,
        'only_original_albedo_file_inputs_changed': True, 'native_qualified': False}
