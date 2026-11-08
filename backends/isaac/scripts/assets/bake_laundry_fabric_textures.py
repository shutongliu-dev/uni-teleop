"""Bake a versioned set of pattern-space fabric textures for laundry cloth render skins.

CC0 Poly Haven scans (cotton jersey, denim, terry) are tiled at their recorded
physical size over each cloth's rest pattern and dyed to its frozen colour.
Hems, ribs, stitches, garment construction (jeans waistband, loops, fly,
pockets, rivets, yoke, patch; T-shirt rib collar, set-in sleeves, coverstitch),
towel borders, denim wear and material-space macro relief (wash wrinkles, edge
roll-off, seam puckering, pile/heather mottle) are drawn from distance fields of
the pattern. Sewn two-panel garments get a front/back atlas and single sheets a
two-sided one, mirrored in u so every visible side keeps right-handed tangent
frames. The result is appearance only: geometry, thickness and physics are not
changed, and the profiles are consumed by bench fabric_visuals.

Run from a complete frozen source snapshot, into a fresh shared asset version:
  python -m scripts.assets.bake_laundry_fabric_textures --output <assets/materials/.../vN>
      --cloth shirt <release_dir> 0 --cloth towel_blue <release_dir> 0 ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.fabric_visuals import cloth_geometry_digest

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = Path(f'{DATA_ROOT}/assets')
POLYHAVEN = ASSETS/'materials/polyhaven/laundry_20261021'
FABRICS = {'shirt': 'cotton_jersey', 'trousers': 'denim_fabric', 'towel': 'terry_cloth'}
DETAIL_M = .0009  # metres per detail-height unit
COPPER = np.asarray([.60, .36, .20])
GUNMETAL = np.asarray([.32, .32, .34])
JEANS_THREAD = np.asarray([.62, .36, .10])
LEATHER = np.asarray([.36, .19, .085])
FADED_DENIM = np.asarray([.33, .44, .60])
OMNISURFACE = {
    'shirt': {'diffuse_reflection_roughness': .5, 'specular_reflection_weight': .5,
              'specular_retro_reflection_weight': .4, 'specular_retro_reflection_roughness': .45},
    'trousers': {'diffuse_reflection_roughness': .3, 'specular_reflection_weight': .45,
                 'specular_retro_reflection_weight': .18, 'specular_retro_reflection_roughness': .5},
    'towel': {'diffuse_reflection_roughness': .7, 'specular_reflection_weight': .3,
              'specular_retro_reflection_weight': .6, 'specular_retro_reflection_roughness': .6}}
# Real wash wrinkles reach 10-30 degree slopes; weaker relief vanishes under soft room light.
WRINKLES = {  # count per m2, length, width, amplitude (m)
    'shirt': (130., (.04, .18), (.004, .012), (.0008, .0024)),
    'trousers': (70., (.05, .16), (.006, .016), (.0010, .0030)),
    'towel': (30., (.06, .18), (.010, .022), (.0008, .0020))}
UNDULATION_M = .0015
EDGE_ROLL = {  # (amplitude m, width m) for sewn edges and finished openings
    'shirt': {'seam': (.0030, .007), 'neckline': (.0008, .003), 'default': (.0016, .005)},
    'trousers': {'seam': (.0032, .008), 'waistband': (.0016, .005), 'default': (.0016, .005)},
    'towel': {'seam': (.0020, .006)}}
MOTTLE = {  # (feature size m, albedo amplitude, relief m): heather, slub, terry pile clumps
    'shirt': (.003, .035, .00008), 'trousers': (.004, .030, .00010), 'towel': (.004, .060, .00022)}
HEMS = {  # finished edge: (band width m, stitch offsets m, tone)
    'shirt': {'lower_hem': (.026, (.018, .024), .93), 'left_cuff': (.023, (.015, .021), .93),
              'right_cuff': (.023, (.015, .021), .93), 'neckline': (.017, (.021, .027), .84)},
    'trousers': {'waistband': (.038, (.003, .035), .96), 'left_cuff': (.020, (.016,), .95),
                 'right_cuff': (.020, (.016,), .95)},
    'towel': {}}


def digest(path):
    from bench.isaacsim.release import digest as file_digest
    return file_digest(path)


def to_linear(srgb):
    return np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)


def to_srgb(linear):
    linear = np.clip(linear, 0., 1.)
    return np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)


def smoothstep(edge0, edge1, x):
    t = np.clip((x-edge0)/(edge1-edge0), 0., 1.)
    return t*t*(3.-2.*t)


# Scans -----------------------------------------------------------------------------------

def flatten_low_frequency(maps, size_m, sigma_m=.008):
    """Remove scan-scale washes, creases and lighting so tiles do not repeat visibly."""
    from scipy import ndimage
    height, width = maps['albedo'].shape[:2]
    sigma = (sigma_m/size_m[1]*height, sigma_m/size_m[0]*width)
    flat = {}
    for role, array in maps.items():
        if array.ndim == 3:
            blurred = np.stack([ndimage.gaussian_filter(array[..., c], sigma, mode='wrap') for c in range(3)], axis=-1)
        else:
            blurred = ndimage.gaussian_filter(array, sigma, mode='wrap')
        if role == 'albedo':
            linear, low = to_linear(array), to_linear(blurred)
            flat[role] = to_srgb(linear/np.maximum(low, 1.e-4)*linear.reshape(-1, 3).mean(axis=0))
        elif role == 'normal':
            vector = (array-blurred)*2.+np.asarray([0., 0., 1.], dtype=np.float32)
            vector[..., 2] = np.maximum(vector[..., 2], .2)
            flat[role] = vector/np.linalg.norm(vector, axis=-1, keepdims=True)*.5+.5
        else:
            flat[role] = np.clip(array-blurred+array.mean(), 0., 1.)
    return flat


def load_scan(kind):
    """Scan arrays (row 0 = v 0), physical size, file pins and Poly Haven info."""
    from PIL import Image
    fabric = FABRICS[kind]
    folder = POLYHAVEN/fabric
    info = load_json(folder/'API_INFO.json')
    size = np.asarray(info['dimensions'], dtype=float)/1000.
    maps, files = {}, {}
    for role, suffix, mode in (('albedo', 'diff', 'RGB'), ('normal', 'nor_gl', 'RGB'),
                               ('roughness', 'rough', 'L'), ('occlusion', 'ao', 'L')):
        path = folder/f'{fabric}_{suffix}_2k.jpg'
        with Image.open(path) as image:
            array = np.asarray(image.convert(mode), dtype=np.float32)/255.
        maps[role] = array[::-1].copy()
        files[role] = {'path': str(path), 'sha256': digest(path)}
    for name in ('API_INFO.json', 'LICENSE.txt', 'SOURCE.json'):
        files[name] = {'path': str(folder/name), 'sha256': digest(folder/name)}
    if kind == 'trousers':
        # The denim scan carries one topstitched seam (columns ~770-900 of 2048) and two
        # weaker creases: join two clean strips, then cross-fade the ends into a tile.
        width = maps['albedo'].shape[1]
        first, second = (0, int(width*.371)), (int(width*.444), int(width*.830))
        join, wrap = 60, int(width*.06)
        for role, array in maps.items():
            a = array[:, first[0]:first[1]]
            b = array[:, second[0]:second[1]]
            ramp = np.linspace(0., 1., join, dtype=np.float32).reshape((1, join)+(1,)*(a.ndim-2))
            seamless = np.concatenate([a[:, :-join], a[:, -join:]*(1.-ramp)+b[:, :join]*ramp, b[:, join:]], axis=1)
            ramp = np.linspace(0., 1., wrap, dtype=np.float32).reshape((1, wrap)+(1,)*(a.ndim-2))
            seamless[:, :wrap] = seamless[:, :wrap]*ramp+seamless[:, -wrap:]*(1.-ramp)
            maps[role] = seamless[:, :-wrap].copy()
        size = size*np.asarray([maps['albedo'].shape[1]/width, 1.])
        return flatten_low_frequency(maps, size, sigma_m=.003), size, files, info
    return flatten_low_frequency(maps, size), size, files, info


def sample_tiled(array, size_m, x, y, texel_m):
    """Bilinear wrap sampling of a scan prefiltered to the output texel density."""
    from PIL import Image
    from scipy import ndimage
    height, width = array.shape[:2]
    factor = min(1., (size_m[0]/width)/texel_m, (size_m[1]/height)/texel_m)
    if factor < .999:
        new = (max(8, int(round(width*factor))), max(8, int(round(height*factor))))
        mode = 'RGB' if array.ndim == 3 else 'L'
        image = Image.fromarray(np.clip(array*255+.5, 0, 255).astype(np.uint8), mode).resize(new, Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32)/255.
        height, width = array.shape[:2]
    cols = (x/size_m[0])*width-.5
    rows = (y/size_m[1])*height-.5
    if array.ndim == 2:
        return ndimage.map_coordinates(array, [rows, cols], order=1, mode='grid-wrap').astype(np.float32)
    return np.stack([ndimage.map_coordinates(array[..., c], [rows, cols], order=1, mode='grid-wrap')
                     for c in range(array.shape[2])], axis=-1).astype(np.float32)


# Procedural fields -------------------------------------------------------------------------

def low_frequency_variation(resolution, seed, cells=48, smoothing=3.):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal((cells, cells)), smoothing, mode='reflect')
    field = ndimage.zoom(field, resolution/cells, order=3)[:resolution, :resolution]
    return ((field-field.mean())/max(field.std(), 1.e-9)).astype(np.float32)


def variation_field(shape, seed, cells=48, smoothing=3.):
    """Non-repeating centimetre-scale field on the enclosing square, cropped to the grid."""
    return low_frequency_variation(max(shape), seed, cells=cells, smoothing=smoothing)[:shape[0], :shape[1]]


def mottle_field(shape, texel, seed, feature_m):
    """Unit-variance isotropic noise at a few-millimetre scale."""
    from scipy import ndimage
    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    field = ndimage.gaussian_filter(noise, feature_m/texel/2.)
    return field/max(float(field.std()), 1.e-9)


def denim_twill(x, y, seed):
    """Right-hand denim twill relief and warp/weft shading; slub breaks any larger period."""
    pitch = .0008
    slub = variation_field(x.shape, seed, cells=256, smoothing=1.5)
    phase = 2*np.pi*((x+y)/(pitch*np.sqrt(2.)))+.6*slub
    twill = .5+.5*np.sin(phase)
    warp = .5+.5*np.sin(2*np.pi*x/(pitch*.5))
    return (.75*twill+.25*warp*twill).astype(np.float32), twill.astype(np.float32)


def height_to_normal(height, texel_m, strength):
    dy, dx = np.gradient(height, texel_m)
    normal = np.stack([-strength*dx, -strength*dy, np.ones_like(height)], axis=-1)
    return normal/np.linalg.norm(normal, axis=-1, keepdims=True)


def compose_normal(base_rgb, detail, macro):
    """Whiteout blend of the encoded base map with detail and macro tangent normals."""
    b = base_rgb*2.-1.
    combined = np.stack([b[..., 0]+detail[..., 0]+macro[..., 0], b[..., 1]+detail[..., 1]+macro[..., 1],
                         b[..., 2]*detail[..., 2]*macro[..., 2]], axis=-1)
    combined /= np.linalg.norm(combined, axis=-1, keepdims=True)
    return combined*.5+.5


# Pattern geometry --------------------------------------------------------------------------

def silhouette(rest_xy, triangles, origin, texel, shape):
    """Rasterized rest-pattern mask on a grid of square texels (row 0 = v 0)."""
    from PIL import Image, ImageDraw
    image = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    pts = (rest_xy-origin)/texel
    for tri in triangles:
        draw.polygon([tuple(pts[i]) for i in tri], fill=255)
    return np.asarray(image, dtype=bool)


def classify_boundary(mask, rest_xy, features, origin, texel):
    """Distance to the silhouette and an integer finished-edge class map (0 = sewn seam)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    boundary = mask & ~ndimage.binary_erosion(mask)
    rows, cols = np.nonzero(boundary)
    points = np.stack([cols+.5, rows+.5], axis=1)*texel+origin
    classes = ['seam']
    codes = np.zeros(len(points), dtype=np.int16)
    owners, coords = [], []
    for name, ids in features.items():
        if name == 'crotch':
            continue
        classes.append(name)
        owners += [len(classes)-1]*len(ids)
        coords.append(rest_xy[np.asarray(ids, dtype=int)])
    if coords:
        dist, nearest = cKDTree(np.concatenate(coords)).query(points)
        near = dist < .008
        codes[near] = np.asarray(owners, dtype=np.int16)[nearest[near]]
    code_map = np.zeros(mask.shape, dtype=np.int16)
    code_map[rows, cols] = codes
    _, nearest_boundary = ndimage.distance_transform_edt(~boundary, return_indices=True)
    inherited = code_map[nearest_boundary[0], nearest_boundary[1]]
    distance = ndimage.distance_transform_edt(mask)*texel
    return distance, inherited, classes


def polyline_field(x, y, points, margin):
    """Distance (m) and arclength (m) to a polyline, evaluated only inside its padded bbox."""
    points = np.asarray(points, dtype=float)
    distance = np.full(x.shape, np.inf, dtype=np.float32)
    along = np.zeros(x.shape, dtype=np.float32)
    lo, hi = points.min(axis=0)-margin, points.max(axis=0)+margin
    cols = np.flatnonzero((x[0] >= lo[0]) & (x[0] <= hi[0]))
    rows = np.flatnonzero((y[:, 0] >= lo[1]) & (y[:, 0] <= hi[1]))
    if not len(cols) or not len(rows):
        return distance, along
    window = (slice(rows[0], rows[-1]+1), slice(cols[0], cols[-1]+1))
    px, py = x[window], y[window]
    best = np.full(px.shape, np.inf)
    best_s = np.zeros(px.shape)
    start = 0.
    for a, b in zip(points[:-1], points[1:]):
        segment = b-a
        length = float(np.hypot(*segment))
        if length <= 0:
            continue
        t = np.clip(((px-a[0])*segment[0]+(py-a[1])*segment[1])/length**2, 0., 1.)
        d = np.hypot(px-(a[0]+t*segment[0]), py-(a[1]+t*segment[1]))
        closer = d < best
        best = np.where(closer, d, best)
        best_s = np.where(closer, start+t*length, best_s)
        start += length
    distance[window] = best
    along[window] = best_s
    return distance, along


def bezier(points, count=40):
    p = np.asarray(points, dtype=float)
    t = np.linspace(0., 1., count)[:, None]
    if len(p) == 3:
        return (1-t)**2*p[0]+2*(1-t)*t*p[1]+t**2*p[2]
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t**2*p[2]+t**3*p[3]


def offset_polyline(points, offset):
    """Offset along the left normal (-t_y, t_x) of the polyline direction."""
    points = np.asarray(points, dtype=float)
    tangent = np.gradient(points, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.e-12)
    return points+offset*normal


def rotate(points, angle, centre):
    c, s = np.cos(angle), np.sin(angle)
    p = np.asarray(points, dtype=float)
    return np.stack([c*p[:, 0]-s*p[:, 1], s*p[:, 0]+c*p[:, 1]], axis=1)+np.asarray(centre)


# Construction details ----------------------------------------------------------------------

class Panel:
    """Detail layers for one atlas panel, all in rest pattern coordinates."""

    def __init__(self, x, y, mask):
        self.x, self.y, self.mask = x, y, mask
        shape = x.shape
        self.shade = np.ones(shape, dtype=np.float32)
        self.height = np.zeros(shape, dtype=np.float32)  # detail units (DETAIL_M)
        self.macro = np.zeros(shape, dtype=np.float32)   # metres
        self.thread = np.zeros(shape, dtype=np.float32)
        self.metal = np.zeros(shape, dtype=np.float32)
        self.metal_color = np.zeros(shape+(3,), dtype=np.float32)
        self.leather = np.zeros(shape, dtype=np.float32)
        self.wear = np.zeros(shape, dtype=np.float32)
        self.flat = np.zeros(shape, dtype=np.float32)

    def stitch(self, points, offsets=(0.,), dash=.0032, width=.00045):
        for offset in offsets:
            line = offset_polyline(points, offset) if offset else np.asarray(points, dtype=float)
            distance, along = polyline_field(self.x, self.y, line, .004)
            coverage = np.exp(-(distance/width)**2)
            if dash is not None:
                coverage = coverage*smoothstep(-.15, .25, np.cos(2*np.pi*along/dash))
            self.thread = np.maximum(self.thread, coverage*self.mask)

    def bartack(self, centre, angle, length=.009):
        d = np.asarray([np.cos(angle), np.sin(angle)])*length/2
        self.stitch([np.asarray(centre)-d, np.asarray(centre)+d], dash=None, width=.0011)

    def seam(self, points, depth=.12, width=.0016, valley=.0004):
        distance, _ = polyline_field(self.x, self.y, points, .008)
        groove = np.exp(-(distance/width)**2)*self.mask
        self.shade *= 1.-depth*groove
        self.macro -= valley*groove

    def ridge(self, points, width=.003, height=.0004):
        distance, _ = polyline_field(self.x, self.y, points, .01)
        self.macro += height*np.exp(-(distance/width)**2)*self.mask

    def layer_edge(self, points, width=.0022, depth=.18, lift=.0007):
        """Edge of an applied layer (pocket, loop): raised rounded border plus a soft shadow."""
        distance, _ = polyline_field(self.x, self.y, points, .01)
        self.macro += lift*np.exp(-(distance/width)**2)*self.mask
        self.shade *= 1.-depth*np.exp(-((distance-width)/width)**2)*self.mask

    def pucker(self, points, amplitude=.00012, wavelength=.011, decay=.006, seed=0):
        distance, along = polyline_field(self.x, self.y, points, decay*4)
        phase = np.random.default_rng(seed).uniform(0, 2*np.pi)
        wobble = 1.+.25*np.sin(2*np.pi*along/(wavelength*3.7)+phase)
        ripple = np.sin(2*np.pi*along/wavelength*wobble+phase)
        self.macro += amplitude*ripple*np.exp(-distance/decay)*self.mask

    def rivet(self, centre, radius, color):
        distance = np.hypot(self.x-centre[0], self.y-centre[1])
        disc = smoothstep(radius, radius*.75, distance)*self.mask
        dome = np.clip(1.-(distance/radius)**2, 0., 1.)*self.mask
        self.metal = np.maximum(self.metal, disc)
        self.metal_color = self.metal_color*(1.-disc[..., None])+disc[..., None]*color
        self.macro += .0012*np.sqrt(dome)

    def belt_loop(self, x0, top, bottom, half_width=.0055):
        from scipy import ndimage
        outline = [[x0-half_width, top], [x0+half_width, top], [x0+half_width, bottom],
                   [x0-half_width, bottom], [x0-half_width, top]]
        inside = ((np.abs(self.x-x0) < half_width) & (self.y < top) & (self.y > bottom)).astype(np.float32)
        self.macro += .0012*ndimage.gaussian_filter(inside, 1.5)*self.mask
        self.layer_edge(outline, width=.0012, depth=.22, lift=.0002)
        self.wear = np.maximum(self.wear, .35*ndimage.gaussian_filter(inside, 2.)*self.mask)
        for y0 in (top-.003, bottom+.004):
            self.bartack((x0, y0), 0., length=2*half_width-.002)

    def fade(self, centre, radii, amount, angle=0.):
        dx, dy = self.x-centre[0], self.y-centre[1]
        c, s = np.cos(angle), np.sin(angle)
        u, w = (c*dx+s*dy)/radii[0], (-s*dx+c*dy)/radii[1]
        self.wear = np.maximum(self.wear, amount*np.exp(-(u*u+w*w))*self.mask)

    def streak(self, start, end, width, amount, crease=.00025):
        distance, along = polyline_field(self.x, self.y, [start, end], width*3)
        length = float(np.hypot(*(np.asarray(end)-np.asarray(start))))
        taper = np.clip(np.sin(np.pi*np.clip(along/max(length, 1.e-9), 0., 1.)), 0., 1.)**.7
        profile = np.exp(-(distance/width)**2)*taper*self.mask
        self.wear = np.maximum(self.wear, amount*profile)
        self.macro -= crease*np.exp(-(distance/(width*.5))**2)*taper*self.mask

    def patch(self, centre, size, color_weight=1.):
        from scipy import ndimage
        dx, dy = np.abs(self.x-centre[0]), np.abs(self.y-centre[1])
        inside = smoothstep(size[0]/2, size[0]/2-.0008, dx)*smoothstep(size[1]/2, size[1]/2-.0008, dy)*self.mask
        self.leather = np.maximum(self.leather, color_weight*inside)
        self.macro += .0009*ndimage.gaussian_filter(inside.astype(np.float32), 2.)
        hw, hh = size[0]/2-.003, size[1]/2-.003
        border = [[centre[0]-hw, centre[1]+hh], [centre[0]+hw, centre[1]+hh], [centre[0]+hw, centre[1]-hh],
                  [centre[0]-hw, centre[1]-hh], [centre[0]-hw, centre[1]+hh]]
        self.stitch(border, dash=.003)
        emboss = [[centre[0]-hw+.006, centre[1]+hh-.006], [centre[0]+hw-.006, centre[1]+hh-.006],
                  [centre[0]+hw-.006, centre[1]-hh+.006], [centre[0]-hw+.006, centre[1]-hh+.006],
                  [centre[0]-hw+.006, centre[1]+hh-.006]]
        self.seam(emboss, depth=.10, width=.0008, valley=.00015)


def garment_anchors(kind, outline, mask, x):
    poly = np.asarray(outline, dtype=float)
    rows_any = mask.any(axis=1)
    anchors = {'outer_right': np.where(rows_any, np.max(np.where(mask, x, -np.inf), axis=1), np.nan)}
    if kind == 'trousers':
        anchors['waist'] = float(poly[:, 1].max())
        anchors['crotch'] = float(poly[np.abs(poly[:, 0]) < 1.e-9][:, 1].min())
        anchors['cuff'] = float(poly[:, 1].min())
    elif kind == 'shirt':
        hem_y = float(poly[:, 1].min())
        body = float(np.abs(poly[poly[:, 1] == hem_y, 0]).max())
        side = poly[(np.abs(np.abs(poly[:, 0])-body) < 1.e-9) & (poly[:, 1] > hem_y+1.e-9), 1]
        anchors.update(hem=hem_y, body=body, underarm=float(side.min()), shoulder=float(side.max()))
    return anchors


def outer_x(anchors, y_value, y):
    row = int(np.clip(np.searchsorted(y[:, 0], y_value), 0, y.shape[0]-1))
    return float(anchors['outer_right'][row])


def jeans_front(p, a, y):
    waist, crotch = a['waist'], a['crotch']
    band = waist-.038
    for x0 in (-.098, .098):
        p.belt_loop(x0, waist+.001, band-.012)
    fly = np.concatenate([[[.035, band-.002]], [[.035, waist-.172]],
                          bezier([[.035, waist-.172], [.035, waist-.203], [.004, waist-.207]], 16)[1:]])
    p.stitch(fly, offsets=(0., -.006))
    p.pucker(fly, amplitude=.00010, wavelength=.010, seed=3)
    p.seam([[.002, band], [.002, waist-.205]], depth=.18, valley=.0006)
    p.bartack((.010, waist-.207), 0.)
    p.seam([[0., waist-.207], [0., crotch]], depth=.15)
    p.stitch([[0., waist-.207], [0., crotch+.01]], offsets=(.003, .009))
    for sign in (-1., 1.):
        y_low = waist-.105
        opening = bezier([[sign*.118, band], [sign*.133, waist-.092], [sign*(outer_x(a, y_low, y)-.0005), y_low]], 32)
        p.layer_edge(opening, width=.0024, depth=.26, lift=.0008)
        p.stitch(opening, offsets=(-sign*.004, -sign*.010))
        p.pucker(opening, amplitude=.00012, wavelength=.012, seed=int(5+sign))
        p.rivet((sign*.120, band-.004), .0033, COPPER)
        p.rivet((sign*(outer_x(a, y_low, y)-.006), y_low+.002), .0033, COPPER)
        root = np.asarray([sign*.036, waist-.216])
        for k, (dx, dy) in enumerate(((.075, .060), (.092, .036), (.102, .012), (.095, -.012), (.082, -.030))):
            p.streak(root+np.asarray([sign*.004*k, -.003*k]), root+np.asarray([sign*dx, dy]), .0030, .34)
        p.fade((sign*.105, crotch-.07), (.040, .11), .22)
        p.fade((sign*.108, crotch-.22), (.036, .055), .16)
    coin = [[-.135, band-.003], [-.135, waist-.075], [-.172, waist-.075], [-.172, band-.003]]
    p.layer_edge(coin, width=.0015, depth=.16, lift=.0005)
    p.stitch(coin, offsets=(-.003,))
    p.rivet((-.004, waist-.019), .0078, GUNMETAL)


def jeans_back(p, a, y):
    waist, crotch = a['waist'], a['crotch']
    band = waist-.038
    for x0 in (-.12, 0., .12):
        p.belt_loop(x0, waist+.001, band-.012)
    side_y = waist-.062
    yoke = [[-outer_x(a, side_y, y)-.002, side_y], [0., waist-.100], [outer_x(a, side_y, y)+.002, side_y]]
    p.seam(yoke, depth=.20, valley=.0006)
    p.ridge(offset_polyline(yoke, -.004), width=.003, height=.0003)
    p.stitch(yoke, offsets=(-.004, -.010))
    p.seam([[0., waist-.100], [0., crotch]], depth=.16, valley=.0005)
    p.stitch([[0., waist-.100], [0., crotch+.015]], offsets=(.004, .010))
    for sign in (-1., 1.):
        centre = (sign*.092, waist-.118)
        angle = sign*np.radians(6.)
        outline = rotate([[-.0625, 0.], [.0625, 0.], [.060, -.115], [0., -.143], [-.060, -.115], [-.0625, 0.]],
                         angle, centre)
        p.layer_edge(outline, width=.0024, depth=.24, lift=.0008)
        p.stitch(outline, offsets=(-.004, -.010))
        p.pucker(outline, amplitude=.00010, wavelength=.012, seed=int(11+sign))
        top_hem = rotate([[-.060, -.013], [.060, -.013]], angle, centre)
        p.stitch(top_hem, offsets=(0., .006))
        for k in range(2):
            arc = np.asarray([[-.046, -.060], [-.018, -.100], [.010, -.040], [.046, -.084]])
            arc = arc*np.asarray([sign, 1.])-np.asarray([0., .008*k])
            p.stitch(rotate(bezier(arc, 40), angle, centre))
        for corner in (outline[0], outline[1]):
            p.bartack(corner+np.asarray([0., -.004]), np.pi/2, length=.008)
        p.fade((sign*.090, waist-.215), (.05, .055), .14)
        knee = crotch-.21
        for k in range(4):
            yk = knee+.009*(k-1.5)
            half = .018-.003*abs(k-1.5)
            p.streak((sign*.108-half, yk), (sign*.108+half, yk+.002*(k-1.5)), .0026, .30, crease=.00035)
    p.patch((-.062, waist-.020), (.068, .036))


def tee_panel(p, a, back):
    for sign in (-1., 1.):
        body, top, low = a['body'], a['shoulder'], a['underarm']
        armhole = bezier([[sign*body, top], [sign*(body-.013), (top+low)/2], [sign*body, low]], 32)
        p.seam(armhole, depth=.10, width=.0012, valley=.00035)
        p.ridge(offset_polyline(armhole, sign*.0045), width=.0028, height=.00028)
        p.pucker(armhole, amplitude=.00010, wavelength=.011, seed=int(21+sign+3*back))


def wrinkle_field(x, y, mask, texel, seed, kind, bias=None):
    """Material-space wash wrinkles: tapered soft ridges and creases (metres)."""
    density, lengths, widths, amplitudes = WRINKLES[kind]
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(mask)
    area = len(rows)*texel*texel
    field = np.zeros(x.shape, dtype=np.float32)
    for _ in range(int(round(density*area))):
        k = rng.integers(len(rows))
        cx, cy = x[rows[k], cols[k]], y[rows[k], cols[k]]
        length = rng.uniform(*lengths)
        width = rng.uniform(*widths)
        amplitude = rng.uniform(*amplitudes)*rng.choice([-1., 1.], p=[.55, .45])
        angle = rng.uniform(0, np.pi) if bias is None else bias(rng, cx, cy)
        reach = length/2+3*width
        c0, c1 = np.searchsorted(x[0], [cx-reach, cx+reach])
        r0, r1 = np.searchsorted(y[:, 0], [cy-reach, cy+reach])
        if c1 <= c0 or r1 <= r0:
            continue
        px, py = x[r0:r1, c0:c1]-cx, y[r0:r1, c0:c1]-cy
        s = np.cos(angle)*px+np.sin(angle)*py
        t = -np.sin(angle)*px+np.cos(angle)*py
        bend = rng.uniform(-.25, .25)/max(length, 1.e-9)
        t = t-bend*s*s
        taper = np.clip(1.-(2*s/length)**2, 0., 1.)**2
        field[r0:r1, c0:c1] += (amplitude*np.exp(-(t/width)**2)*taper).astype(np.float32)
    return field*mask


def wrinkle_bias(kind, anchors):
    if kind != 'trousers':
        return None
    cuff = anchors['cuff']

    def bias(rng, cx, cy):
        if cy < cuff+.12 and rng.random() < .7:
            return rng.normal(0., .35)  # stacking creases across the lower leg
        return rng.normal(0., .6) if rng.random() < .6 else rng.uniform(0, np.pi)
    return bias


def edge_roll(distance, label, classes, kind, mask):
    """Rounded fall-off toward the silhouette (metres): sewn folds and finished hems."""
    from scipy import ndimage
    rules = EDGE_ROLL[kind]
    amplitude = np.zeros(mask.shape, dtype=np.float32)
    width = np.zeros(mask.shape, dtype=np.float32)
    for code, name in enumerate(classes):
        region = label == code
        amplitude[region], width[region] = rules.get(name, rules.get('default', rules['seam']))
    amplitude = ndimage.gaussian_filter(amplitude, 12.)
    width = ndimage.gaussian_filter(width, 12.)
    fall = np.clip(1.-distance/np.maximum(width, 1.e-6), 0., 1.)
    return (-(amplitude*fall*fall)*mask).astype(np.float32)


# Bake --------------------------------------------------------------------------------------

def cloth_inputs(release, index):
    """Frozen scene cloth, its kind, boundary features and pattern outline."""
    from bench.simulation.isaacsim_backend.geometry import cloth_grid
    scene_path, task_path = Path(release)/'scene.json', Path(release)/'task.json'
    spec = load_json(scene_path)['cloths'][index]
    metadata = load_json(task_path)['asset_metadata']
    kind = metadata.get('garment_kind') or 'towel'
    if kind not in FABRICS:
        raise ValueError('Unsupported laundry cloth kind: '+str(kind))
    rest, triangles = cloth_grid(spec)
    features = metadata['cloths'][index].get('boundary_features', {}) if kind != 'towel' else {}
    outline = metadata.get('design', {}).get('representation', {}).get('source_outline_xy_m')
    if kind != 'towel' and outline is None:
        raise ValueError('Garment atlas needs its pinned pattern outline')
    color = np.asarray(spec['isaacsim']['cloth_visual_color'], dtype=float)
    source = {'scene': {'path': str(scene_path.resolve()), 'sha256': digest(scene_path)},
              'task': {'path': str(task_path.resolve()), 'sha256': digest(task_path)},
              'cloth_index': index, 'cloth_id': spec['id']}
    return kind, np.asarray(rest), np.asarray(triangles), features, outline, color, source


def bake(name, release, index, output, resolution):
    from PIL import Image
    from scipy import ndimage

    kind, rest, triangles, features, outline, color, source = cloth_inputs(release, index)
    garment = kind != 'towel'
    rest_xy = rest[:, :2].astype(np.float64)
    margin = .006
    origin = rest_xy.min(axis=0)-margin
    extent = rest_xy.max(axis=0)-rest_xy.min(axis=0)+2*margin
    texel = float(extent.max())/resolution
    nx, ny = (int(v) for v in np.ceil(extent/texel-1.e-9))
    shape = (ny, nx)
    x = np.broadcast_to(origin[0]+(np.arange(nx, dtype=np.float64)+.5)[None, :]*texel, shape)
    y = np.broadcast_to(origin[1]+(np.arange(ny, dtype=np.float64)+.5)[:, None]*texel, shape)
    maps, scan_size, scan_files, info = load_scan(kind)
    base = {role: sample_tiled(array, scan_size, x-origin[0], y-origin[1], texel) for role, array in maps.items()}
    mask = silhouette(rest_xy, triangles, origin, texel, shape)
    distance, label, classes = classify_boundary(mask, rest_xy, features, origin, texel)
    gy, gx = np.gradient(distance)
    norm = np.maximum(np.hypot(gx, gy), 1.e-9)
    along = x*(-gy/norm)+y*(gx/norm)
    source_mean = to_linear(maps['albedo']).reshape(-1, 3).mean(axis=0)
    fabric_albedo = to_linear(base['albedo'])*(color/source_mean)
    fabric_normal = base['normal']
    if kind == 'trousers':
        twill_height, twill = denim_twill(x, y, seed=29)
        fabric_normal = height_to_normal(twill_height.astype(np.float64), texel, strength=.00022)*.5+.5
        fabric_albedo = fabric_albedo*(.90+.20*twill)[..., None]
    if kind == 'shirt':
        noise = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
        heather = ndimage.gaussian_filter(noise, (2.4, .7))
        fabric_albedo = fabric_albedo*(1.+.07*heather/max(float(heather.std()), 1.e-9))[..., None]
    anchors = garment_anchors(kind, outline, mask, x) if garment else {}
    thread_color = JEANS_THREAD if kind == 'trousers' else np.minimum(color*1.10+.01, 1.)
    seed_base = {'shirt': 11, 'trousers': 23, 'towel': 37}[kind]+index
    roll = edge_roll(distance, label, classes, kind, mask)
    _, inside = ndimage.distance_transform_edt(~mask, return_indices=True)
    outputs = {role: [] for role in ('albedo', 'normal', 'roughness', 'metallic')}
    for panel_index, panel_name in enumerate(['front', 'back'] if garment else ['single']):
        p = Panel(x, y, mask)
        for feature, (width, stitches, tone) in HEMS[kind].items():
            if feature not in classes:
                continue
            region = mask & (label == classes.index(feature))
            band = region & (distance <= width)
            p.shade[band] *= tone
            p.height += (.45*np.exp(-((distance-width)/.0012)**2)*region).astype(np.float32)
            p.macro += (.00025*smoothstep(width, width*.6, distance)*region).astype(np.float32)
            if feature == 'neckline':
                rib = .5+.5*np.cos(2*np.pi*along/.0021)
                p.height += np.where(band, .6*rib, 0.).astype(np.float32)
                p.shade[band] *= (.94+.06*rib[band])
            for at in stitches:
                coverage = np.exp(-((distance-at)/.00045)**2)*smoothstep(-.15, .25, np.cos(2*np.pi*along/.0032))
                p.thread = np.maximum(p.thread, (coverage*region).astype(np.float32))
            ripple = np.sin(2*np.pi*along/.012+panel_index)*np.exp(-np.abs(distance-width)/.006)
            p.macro += (.00010*ripple*region).astype(np.float32)
            if kind == 'trousers' and feature != 'waistband':
                roping = .5+.5*np.sin(2*np.pi*(along+.6*distance)/.018)
                p.wear = np.maximum(p.wear, (.55*roping*smoothstep(.006, .001, distance)*region).astype(np.float32))
        seam = mask & (label == 0)
        if garment:
            p.shade[seam] *= (1.-.06*np.exp(-(distance/.0025)**2))[seam]
        if kind == 'trousers':
            patchy = .55+.45*variation_field(shape, seed_base+5+panel_index, cells=96, smoothing=2.)
            p.wear = np.maximum(p.wear, (np.clip(patchy, 0, 1)*.40*smoothstep(.005, .0008, distance)*seam).astype(np.float32))
            waistband = (label == classes.index('waistband'))*mask
            p.wear = np.maximum(p.wear, (.45*smoothstep(.004, .0005, distance)*waistband).astype(np.float32))
            (jeans_front if panel_name == 'front' else jeans_back)(p, anchors, y)
        elif kind == 'shirt':
            tee_panel(p, anchors, panel_name == 'back')
        else:
            hem = mask & (distance <= .009)
            p.shade[hem] *= .95
            p.height += (.4*np.exp(-((distance-.009)/.0012)**2)*mask).astype(np.float32)
            coverage = np.exp(-((distance-.006)/.00045)**2)*smoothstep(-.15, .25, np.cos(2*np.pi*along/.0032))
            p.thread = np.maximum(p.thread, (coverage*mask).astype(np.float32))
            centre_x = (rest_xy[:, 0].min()+rest_xy[:, 0].max())/2
            from_end = (rest_xy[:, 0].max()-rest_xy[:, 0].min())/2-np.abs(x-centre_x)
            flat = smoothstep(.040, .043, from_end)*(1.-smoothstep(.069, .072, from_end))*mask
            weave = .5+.25*(np.cos(2*np.pi*x/.0011)+np.cos(2*np.pi*y/.0011))
            stripes = sum(np.exp(-((from_end-c)/.0011)**2) for c in (.047, .065))
            accent = np.exp(-((from_end-.056)/.0022)**2)
            p.height = (p.height*(1.-flat)+flat*(.15*weave)).astype(np.float32)
            p.flat = flat.astype(np.float32)
            p.shade *= ((1.+.12*flat)*(1.-.30*stripes*mask)*(1.+.40*accent*mask)).astype(np.float32)
            p.macro -= (.0006*flat).astype(np.float32)  # flat weave sits below the terry pile
        seed = seed_base+100*panel_index
        p.macro += wrinkle_field(x, y, mask, texel, seed, kind, wrinkle_bias(kind, anchors))
        p.macro += (UNDULATION_M*variation_field(shape, seed+1, cells=16, smoothing=2.)*mask).astype(np.float32)
        p.macro += roll
        feature_m, tone, relief = MOTTLE[kind]
        mottle = mottle_field(shape, texel, seed+3, feature_m)*(1.-p.flat)
        p.macro += (relief*mottle*mask).astype(np.float32)
        variation = variation_field(shape, seed+2)
        modulation = (p.shade*(1.+.035*variation)*(1.+tone*mottle))[..., None]
        albedo = fabric_albedo*modulation
        if kind == 'trousers':
            # Worn indigo keeps the scan and twill structure, now over the paler weft.
            wear = np.clip(p.wear, 0., .75)[..., None]
            albedo = albedo*(1.-wear)+wear*FADED_DENIM*(fabric_albedo/color)*modulation
        albedo = albedo*(1.-p.thread[..., None])+p.thread[..., None]*thread_color
        albedo = albedo*(1.-p.leather[..., None])+p.leather[..., None]*LEATHER*(1.+.08*variation[..., None])
        albedo = albedo*(1.-p.metal[..., None])+p.metal[..., None]*p.metal_color
        solid = np.clip(p.flat+p.leather+p.metal, 0., 1.)
        occlusion = np.clip(base['occlusion']*(1.-solid)+solid, 0., 1.)
        albedo = albedo*(occlusion**.5)[..., None]  # scan cavity shading lives in the albedo
        base_normal = fabric_normal*(1.-solid[..., None])+solid[..., None]*np.asarray([.5, .5, 1.], dtype=np.float32)
        detail = height_to_normal((p.height+.6*p.thread).astype(np.float64), texel, strength=DETAIL_M)
        macro = height_to_normal(p.macro.astype(np.float64), texel, strength=1.)
        normal = compose_normal(np.clip(base_normal, 0., 1.), detail, macro)
        rough = base['roughness']+.02*variation+.06*p.thread
        if kind == 'towel':
            rough = rough*(1.-p.flat)+p.flat*.62
        rough = rough*(1.-p.leather)+p.leather*.58
        rough = np.clip(rough*(1.-p.metal)+p.metal*.38, 0., 1.)
        layers = {'albedo': to_srgb(albedo[inside[0], inside[1]]), 'normal': normal[inside[0], inside[1]],
                  'roughness': rough[inside[0], inside[1]], 'metallic': p.metal[inside[0], inside[1]]}
        if panel_name == 'back':
            layers = {role: array[:, ::-1].copy() for role, array in layers.items()}
            layers['normal'][..., 0] = 1.-layers['normal'][..., 0]
        for role, array in layers.items():
            outputs[role].append(array.astype(np.float32))
        del p, layers, albedo, normal, rough, detail, macro
    if not garment:
        # Two-sided single sheet: the bottom skin sees the same material, mirrored in u.
        for role, halves in outputs.items():
            mirrored = halves[0][:, ::-1].copy()
            if role == 'normal':
                mirrored[..., 0] = 1.-mirrored[..., 0]
            halves.append(mirrored)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, halves in outputs.items():
        if role == 'metallic' and not any(h.max() > 0 for h in halves):
            continue
        array = np.concatenate(halves, axis=1)
        if not np.isfinite(array).all():
            raise FloatingPointError(f'Non-finite {role} texels in {name}')
        array = np.clip(array[::-1]*255+.5, 0, 255).astype(np.uint8)
        filename = f'{role}.jpg' if role in ('albedo', 'normal') else f'{role}.png'
        image = Image.fromarray(array, 'RGB' if array.ndim == 3 else 'L')
        if filename.endswith('.jpg'):
            image.save(output/filename, quality=95, subsampling=0)
        else:
            image.save(output/filename, optimize=True)
        files[role] = filename
    frame = {'origin_xy_m': origin.tolist(), 'size_xy_m': [nx*texel, ny*texel], 'resolution_xy_px': [nx, ny],
             'texel_m': texel, 'basis': 'Rest pattern XY per panel rectangle of square texels'}
    if garment:
        frame.update(atlas='front_back', panel_split='rest z >= 0 panel left half, rest z < 0 panel right half',
                     back_panel_u='mirrored: u_atlas = 1 - u_rest/2, normal x negated')
    else:
        frame.update(atlas='two_sided', panel_split='top skin left half, bottom skin right half',
                     back_panel_u='mirrored copy of the same material: u_atlas = 1 - u_rest/2, normal x negated')
    profile = {'schema': 'fabric_visual_profile.v1', 'id': 'laundry_fabric_visual:'+name, 'kind': kind,
        'shader': 'omnisurface', 'subdivision_levels': 1, 'ior': 1.3,
        'omnisurface': {**OMNISURFACE[kind], 'specular_retro_reflection_color': np.clip(color**.5, 0., 1.).tolist()},
        'textures': files, 'texture_sha256': {role: digest(output/filename) for role, filename in files.items()},
        'uv_frame': frame,
        'cloth_binding': {'cloth_geometry_sha256': cloth_geometry_digest(rest, triangles),
                          'dye_linear_rgb': color.tolist()},
        'fabric_scan': {'name': FABRICS[kind], 'physical_size_m': scan_size.tolist(), 'files': scan_files,
                        'license': 'CC0-1.0 (Poly Haven)', 'authors': info.get('authors')},
        'pattern_anchors_m': {key: value for key, value in anchors.items() if key != 'outer_right'},
        'construction': {'hems': {key: [value[0], list(value[1]), value[2]] for key, value in HEMS[kind].items()},
                         'wrinkles': WRINKLES[kind], 'edge_roll': EDGE_ROLL[kind], 'undulation_m': UNDULATION_M,
                         'mottle': MOTTLE[kind]},
        'source_cloth': source, 'render_only': True, 'physics_changed': False, 'material_calibrated': False,
        'scope': 'Appearance only: baked maps and normal-map relief over the unchanged rest pattern and thickness.'}
    (output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    return profile


def main():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloth', nargs=3, action='append', required=True, metavar=('NAME', 'RELEASE', 'INDEX'))
    parser.add_argument('--resolution', type=int, default=3072)
    args = parser.parse_args()
    snapshot = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.assets.bake_laundry_fabric_textures', 'bench.simulation.isaacsim_backend.fabric_visuals'),
        entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Bake shared fabric assets from a complete frozen source snapshot')
    output = args.output.resolve()
    if not output.is_relative_to(ASSETS) or output.exists():
        parser.error('Choose a fresh versioned directory under shared assets')
    names = [name for name, _, _ in args.cloth]
    if len(set(names)) != len(names) or not all(name.replace('_', '').isalnum() for name in names):
        parser.error('Texture set names must be unique identifiers')
    output.mkdir(parents=True)
    sets = {}
    for name, release, index in args.cloth:
        profile = bake(name, Path(release).resolve(), int(index), output/name, args.resolution)
        path = output/name/'profile.json'
        sets[name] = {'profile': {'path': str(path), 'sha256': digest(path)}, 'kind': profile['kind'],
                      'cloth_binding': profile['cloth_binding'], 'source_cloth': profile['source_cloth']}
        print(json.dumps({'set': name, 'profile': sets[name]['profile']}))
    scans = {kind: {'name': fabric, 'license': 'CC0-1.0', 'folder': str(POLYHAVEN/fabric)}
             for kind, fabric in FABRICS.items()}
    attribution = ('Laundry fabric visual textures, render-only appearance assets.\n'
        'Fabric scans: Poly Haven "Cotton Jersey" (colormass, Rico Cilliers), "Denim Fabric" (Rob Tuytel) and\n'
        '"Terry Cloth" (colormass, Rico Cilliers), CC0-1.0, mirrored under '+str(POLYHAVEN)+'.\n'
        'Hems, stitches, garment construction, wear and macro relief are project-authored procedural details.\n'
        'No geometry, thickness, mass, friction or solver value is changed; appearance is not a material calibration.\n')
    (output/'ATTRIBUTION.txt').write_text(attribution)
    manifest = {'schema': 'laundry_fabric_visual_assets.v1', 'version': output.name,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'sets': sets, 'fabric_scans': scans,
        'attribution_sha256': digest(output/'ATTRIBUTION.txt'),
        'authoring_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__))},
        'source_snapshot': snapshot, 'resolution_px': args.resolution,
        'render_only': True, 'physics_changed': False, 'material_calibrated': False}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'sets': sorted(sets)}))


if __name__ == '__main__':
    main()
