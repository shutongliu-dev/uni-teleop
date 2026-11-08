"""Project-authored folded garment midsurfaces for native surface deformables.

Each garment is deliberately a single material sheet cut to a clothing pattern,
not a sewn double-wall wearable. Its neckline/sleeves or waist/crotch/legs are
real topology. The reset bends this same stress-free sheet; no rigid proxies,
fixed nodes or permanently folded rest shapes are used. Dimensions, mass and
constitutive parameters are explicit uncalibrated simulation design priors.
"""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from bench.isaacsim.release import checked, digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.cloth_visuals import boundary_edges

ASSETS = Path(f'{DATA_ROOT}/assets')
SCHEMA = 'laundry_garment_asset.v1'
MANIFEST_SCHEMA = 'laundry_garment_asset_manifest.v1'
TASK_KINDS = {'shirt_stacking': 'shirt', 'trousers_stacking': 'trousers'}
THICKNESS = {'shirt': .0016, 'trousers': .0018}
AREAL_DENSITY = {'shirt': .50, 'trousers': .75}
NOMINAL_MASS_KG = {'shirt_stacking': .139885, 'trousers_stacking': .203887}
REGULARIZED_SHIRT_PROFILE = 'shirt_boundary_regularized_v1'
PHYSICS_OPTIONS_BY_TASK = {
    task: {'physics_dt_s': .0005, 'cloth_iterations': 64,
           'cloth_linear_damping': 2., 'cloth_thickness_m': THICKNESS[kind],
           'cloth_youngs_modulus_pa': 667000., 'cloth_poissons_ratio': .3,
           'cloth_surface_bend_stiffness': 100.,
           'cloth_collision_pair_update_frequency': 8,
           'cloth_collision_iteration_multiplier': 8,
           'cloth_speculative_ccd': True,
           'cloth_self_collision_filter_distance_m': THICKNESS[kind],
           'cloth_contact_offset_m': .0018 if kind == 'shirt' else .002,
           'cloth_rest_offset_m': THICKNESS[kind] / 2}
    for task, kind in TASK_KINDS.items()}


def _design(kind):
    if kind == 'shirt':
        neck = [[.085*np.cos(t), .30-.065*np.sin(t)] for t in np.linspace(0, np.pi, 13)]
        outline = [[-.20, -.30], [.20, -.30], [.20, .10], [.235, .10],
            [.30, .06], [.36, .19], [.25, .29], [.21, .29], *neck,
            [-.21, .29], [-.25, .29], [-.36, .19], [-.30, .06],
            [-.235, .10], [-.20, .10]]
        # Both short sleeves go underneath; the lower body then folds below
        # them. The sleeve creases do not intersect the lower-body crease.
        folds = [(0, .215, 1, .002), (0, -.215, -1, .002), (1, -.06, -1, .004)]
    elif kind == 'trousers':
        outline = [[-.21, .42], [-.19, .16], [-.175, -.42], [-.055, -.42],
            [-.035, .04], [0, .11], [.035, .04], [.055, -.42],
            [.175, -.42], [.19, .16], [.21, .42]][::-1]
        # First fold the lower legs underneath, then bring the left leg below
        # the right. Four ordered layers have 4.4 mm initial centre spacing.
        folds = [(1, -.02, -1, .0022), (0, 0., -1, .0066)]
    else:
        raise ValueError('Unknown garment kind')
    return np.asarray(outline, dtype=float), folds


def _contains(points, polygon):
    inside = np.zeros(len(points), dtype=bool)
    x, y = np.asarray(points).T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] == b[1]:
            continue
        inside ^= (((a[1] > y) != (b[1] > y))
                   & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]))
    return inside


def _half_fold(points, axis, hinge, sign, radius):
    """Rotate a half-sheet through a resolved semicircular neutral surface."""
    result = points.copy()
    s = sign*(points[:, axis]-hinge)
    a = np.pi*radius/2
    theta = np.clip((s+a)/radius, 0., np.pi)
    along = np.where(s <= -a, s, np.where(s >= a, -s, -a+radius*np.sin(theta)))
    result[:, axis] = hinge+sign*(along+points[:, 2]*np.sin(theta))
    result[:, 2] = -radius*(1-np.cos(theta))+points[:, 2]*np.cos(theta)
    return result


@lru_cache(maxsize=4)
def _topology(kind, mesh_profile=None):
    if mesh_profile is not None and (kind != 'shirt' or mesh_profile != REGULARIZED_SHIRT_PROFILE):
        raise ValueError('Unknown garment mesh profile')
    polygon, folds = _design(kind)
    axes = [np.arange(polygon[:, k].min(), polygon[:, k].max()+.0001, .015)
            for k in range(2)]
    for axis, hinge, _, radius in folds:
        axes[axis] = np.r_[axes[axis], hinge+np.linspace(-np.pi*radius/2, np.pi*radius/2, 9)]
    axes = [np.unique(np.round(axis, 10)) for axis in axes]
    points = np.array(np.meshgrid(*axes)).reshape(2, -1).T
    points = points[_contains(points, polygon)]
    if mesh_profile == REGULARIZED_SHIRT_PROFILE:
        # Keep the original outline and all its crease intersections. Interior
        # grid points within 0.4 mm of that boundary create tiny collar slivers
        # (the legacy grid has a 33.681 micrometre edge) that can invert during
        # otherwise ordinary native contact. This is a new explicit profile;
        # the omitted profile must reproduce historical garment assets exactly.
        edge = np.roll(polygon, -1, axis=0)-polygon
        rel = points[:, None, :]-polygon
        along = np.clip(np.einsum('nki,ki->nk', rel, edge)/np.sum(edge*edge, axis=1), 0., 1.)
        closest = polygon[None, :, :]+along[:, :, None]*edge
        distance = np.linalg.norm(points[:, None, :]-closest, axis=2).min(axis=1)
        points = points[distance >= .0004-1.e-12]
    perimeter = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        length = np.linalg.norm(b-a)
        parameters = [(0, 0.), (0, 1.)]
        parameters.extend((3, float(t)) for t in np.linspace(0, 1, int(np.ceil(length/.01))+1))
        for axis in range(2):
            if abs(b[axis]-a[axis]) > 1.e-10:
                parameters.extend((2, float(t)) for t in (axes[axis]-a[axis])/(b[axis]-a[axis]) if 0 < t < 1)
                for crease_axis, hinge, _, radius in folds:
                    if crease_axis == axis:
                        parameters.extend((1, float(t)) for t in
                            (hinge+np.linspace(-np.pi*radius/2, np.pi*radius/2, 9)-a[axis])/(b[axis]-a[axis])
                            if 0 < t < 1)
        kept = []
        for _, t in sorted(parameters):
            if all(abs(t-other)*length > .0004 for other in kept):
                kept.append(t)
        perimeter.extend(a+np.asarray(sorted(kept))[:, None]*(b-a))
    points = np.unique(np.round(np.r_[points, perimeter], 10), axis=0)
    triangles = Delaunay(points).simplices
    triangles = triangles[_contains(points[triangles].mean(axis=1), polygon)]
    delta, other = points[triangles[:, 1]]-points[triangles[:, 0]], points[triangles[:, 2]]-points[triangles[:, 0]]
    twice_area = delta[:, 0]*other[:, 1]-delta[:, 1]*other[:, 0]
    # Rounded collinear boundary points can create numerical zero-area ears.
    triangles = triangles[abs(twice_area) > 1.e-10]
    used = np.unique(triangles)
    remap = np.zeros(len(points), dtype=int)
    remap[used] = np.arange(len(used))
    rest = np.c_[points[used], np.zeros(len(used))]
    triangles = remap[triangles]
    normal_z = np.cross(rest[triangles[:, 1]]-rest[triangles[:, 0]],
                        rest[triangles[:, 2]]-rest[triangles[:, 0]])[:, 2]
    triangles[normal_z < 0] = triangles[normal_z < 0, ::-1]
    # Canonical face order also stabilizes named face groups in manifests.
    triangles = np.asarray(sorted(map(tuple, triangles.tolist())), dtype=np.int32)
    folded = rest.copy()
    for fold in folds:
        folded = _half_fold(folded, *fold)
    for array in (rest, triangles, folded):
        array.setflags(write=False)
    return rest, triangles, folded


def topology(kind, *, mesh_profile=None):
    """Return independent rest, triangle and folded arrays in material indices."""
    return tuple(array.copy() for array in _topology(kind, mesh_profile))


def _area(vertices, triangles):
    return np.linalg.norm(np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]],
                                   vertices[triangles[:, 2]]-vertices[triangles[:, 0]]), axis=1)/2


def _features(kind, rest, triangles):
    boundary = np.unique(boundary_edges(triangles))
    points = rest[boundary]
    if kind == 'shirt':
        predicates = {'neckline': (abs(points[:, 0]) <= .08501) & (points[:, 1] > .23),
            'left_cuff': (points[:, 0] < -.29) & (points[:, 1] < .191),
            'right_cuff': (points[:, 0] > .29) & (points[:, 1] < .191),
            'lower_hem': points[:, 1] < -.29999}
    else:
        predicates = {'waistband': points[:, 1] > .41999,
            'left_cuff': (points[:, 0] < 0) & (points[:, 1] < -.41999),
            'right_cuff': (points[:, 0] > 0) & (points[:, 1] < -.41999),
            'crotch': (abs(points[:, 0]) <= .03501) & (points[:, 1] > .03999) & (points[:, 1] < .11001)}
    return {name: boundary[mask].tolist() for name, mask in predicates.items()}


def _face_groups(kind, rest, triangles, color, features):
    centres = rest[triangles].mean(axis=1)
    groups, used = {}, np.zeros(len(triangles), dtype=bool)
    for name, ids in features.items():
        if name == 'crotch':
            continue
        distances = cKDTree(rest[ids, :2]).query(centres[:, :2])[0]
        mask = (distances < (.022 if name == 'waistband' else .012)) & ~used
        if np.any(mask):
            groups[name] = {'face_indices': np.flatnonzero(mask).tolist(),
                            'color': (np.asarray(color)*(.70 if name in ('neckline', 'waistband') else .83)).tolist()}
            used |= mask
    if kind == 'shirt':
        seam = (abs(abs(centres[:, 0])-.19) < .008) & (centres[:, 1] < .08)
    else:
        # Two understated stitched leg seams and a short centre fly remain
        # attached to material coordinates as the cloth unfolds.
        seam = (((abs(abs(centres[:, 0])-.12) < .006) & (centres[:, 1] < .13))
                | ((abs(centres[:, 0]) < .010) & (centres[:, 1] > .15) & (centres[:, 1] < .38)))
    seam &= ~used
    if np.any(seam):
        groups['seams'] = {'face_indices': np.flatnonzero(seam).tolist(),
                           'color': np.minimum(np.asarray(color)*1.18, 1.).tolist()}
    return groups


def _layer_pairs(rest, folded, thickness):
    tree = cKDTree(folded[:, :2])
    pairs = []
    for upper, neighbors in enumerate(tree.query_ball_point(folded[:, :2], .0076)):
        lower = [index for index in neighbors if folded[upper, 2]-folded[index, 2] >= thickness*.8
                 and np.linalg.norm(rest[upper]-rest[index]) > .025]
        if lower:
            # The nearest projected underlying material sample; repeated
            # points from one fold edge cannot impersonate separate layers.
            index = min(lower, key=lambda index: (np.linalg.norm(folded[upper, :2]-folded[index, :2]),
                                                  folded[upper, 2]-folded[index, 2], index))
            pairs.append([int(index), upper])
    return pairs


def candidate_metadata(task_id, *, texture_root=None, source_center_xy_m=None, mesh_profile=None):
    if task_id not in TASK_KINDS:
        raise ValueError('Unknown garment stacking task')
    kind = TASK_KINDS[task_id]
    rest, triangles, folded = _topology(kind, mesh_profile)
    area = float(_area(rest, triangles).sum())
    thickness = THICKNESS[kind]
    source_center = np.asarray((.30, -.55) if source_center_xy_m is None else source_center_xy_m, dtype=float)
    if source_center.shape != (2,) or not np.isfinite(source_center).all():
        raise ValueError('Garment source centre must be a finite XY pair in metres')
    features = _features(kind, rest, triangles)
    pairs = _layer_pairs(rest, folded, thickness)
    rows = []
    for name, centre, color in (
            ('towel', source_center, [.19, .39, .40] if kind == 'shirt' else [.12, .18, .29]),
            ('towel_base', (.95, -.55), [.65, .54, .38] if kind == 'shirt' else [.38, .32, .25])):
        initial = folded.copy()
        initial[:, :2] += np.asarray(centre)-(initial[:, :2].min(0)+initial[:, :2].max(0))/2
        initial[:, 2] += .90+thickness/2-initial[:, 2].min()
        paths = ({role: str(Path(texture_root)/filename) for role, filename in
                  (('normal', 'woven_normal.png'), ('roughness', 'woven_roughness.png'))}
                 if texture_root is not None else {})
        rows.append({'id': name, 'mesh_type': 'custom_surface', 'garment_kind': kind,
            'public_label_zh': '已折上衣' if kind == 'shirt' else '已折长裤',
            'size_m': np.ptp(rest[:, :2], axis=0).tolist(), 'rest_area_m2': area,
            'mass_kg': NOMINAL_MASS_KG[task_id], 'areal_density_kg_m2': AREAL_DENSITY[kind],
            'physical_thickness_m': thickness, 'rest_vertices_m': rest.tolist(),
            'triangles': triangles.tolist(), 'initial_vertices_m': initial.tolist(),
            'folded_reference_vertices_m': initial.tolist(), 'folded_layer_pairs': deepcopy(pairs),
            'support_bottom_vertex_indices': np.flatnonzero(folded[:, 2] <= folded[:, 2].min()+.0003).tolist(),
            'boundary_features': deepcopy(features), 'visual_face_groups': _face_groups(kind, rest, triangles, color, features),
            'visual_texture_paths': paths, 'visual_texture_tile_size_m': [.032, .032],
            'display_color': color, 'dynamic_model': 'native_surface_deformable', 'fixed_vertices': [],
            'material_basis': 'Uncalibrated simulation prior for a simplified single-surface clothing pattern; not measured fabric properties.'})
    result = {'schema': SCHEMA, 'task_id': task_id, 'cloths': rows,
        'target_center_xy_m': [.95, -.55], 'table_top_z_m': .90,
        'clear_rectangle_xy_m': [-.10, 1.40, -.94, -.16],
        'physics_options': deepcopy(PHYSICS_OPTIONS_BY_TASK[task_id]),
        'physics_prior_basis': 'Reuse the previously successful native towel simulation parameter scale: damping 2/s, Young modulus 667000 Pa, bend coefficient 100 Pa, 64 position iterations, 8/8 contact updates and speculative CCD. Garment thickness and contact/rest/filter distances are explicit clothing-specific priors. Towel evidence does not qualify garment dynamics or measured material properties.',
        'procedural_texture_root': str(Path(texture_root)) if texture_root is not None else None,
        'design': {'garment_kind': kind, 'pattern_outline_xy_m': _design(kind)[0].tolist(),
            'fold_sequence_axis_hinge_sign_radius_m': [list(fold) for fold in _design(kind)[1]],
            'mesh_pitch_m': .015, 'crease_semicircle_segments': 8,
            'representation': 'Single material midsurface shaped as a short-sleeve shirt front panel or joined-waist trouser front panel; open edges represent neckline, cuffs, hems and crotch. No sewn tubular interior or front/back seam physics is claimed.',
            'reset': 'Analytic finite-radius folds of the same flat stress-free material coordinates; the complete surface is dynamic after native initialization.',
            'appearance': 'Read-only closed skin at exactly the nominal material thickness, with collar/cuff/waist/hem/seam material subsets and optional original procedural woven normal/roughness maps.'},
        'thickness_semantics': 'Nominal surfaceThickness equals the read-only visual shell thickness. Collision contact/rest offsets are separate frozen solver priors; initial fold spacing is not a thicker material.',
        'fold_reset_resolution': 'Eight segments per semicircular crease plus a 15 mm body grid. Native release, self-contact and resolution sensitivity are not established by CPU authoring.',
        'material_calibrated': False, 'native_physics_accepted': False,
        'ownership': 'Project-authored dynamic garment surfaces; fixed laundry v3 dressing remains unchanged.'}
    if source_center_xy_m is not None:
        result['source_center_xy_m'] = source_center.tolist()
    if mesh_profile is not None:
        result['mesh_profile'] = mesh_profile
    return result


def validate_metadata(metadata):
    if metadata.get('schema') != SCHEMA or metadata != candidate_metadata(
            metadata.get('task_id'), texture_root=metadata.get('procedural_texture_root'),
            source_center_xy_m=metadata.get('source_center_xy_m'), mesh_profile=metadata.get('mesh_profile')):
        raise ValueError('Garment asset differs from deterministic material topology and folded reset design')
    return metadata


def geometry_report(task_id, *, source_center_xy_m=None, mesh_profile=None):
    """CPU evidence only; includes exact triangle crossings and skin topology."""
    from bench.evaluation.cloth.audit_cloth_self_intersection import self_intersections
    from bench.evaluation.cloth.audit_cloth_surface import topology_report
    from bench.simulation.isaacsim_backend.cloth_visuals import closed_skin
    metadata = candidate_metadata(task_id, source_center_xy_m=source_center_xy_m, mesh_profile=mesh_profile)
    row = metadata['cloths'][0]
    rest, triangles, initial = (np.asarray(row[key]) for key in ('rest_vertices_m', 'triangles', 'initial_vertices_m'))
    report, edges = topology_report(triangles, len(rest))
    ratio = np.linalg.norm(initial[edges[:, 1]]-initial[edges[:, 0]], axis=1)/np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    areas = _area(initial, triangles)/_area(rest, triangles)
    hits, candidates = self_intersections(initial, triangles)
    skin, faces, _ = closed_skin(initial, triangles, row['physical_thickness_m'])
    skin_report, _ = topology_report(faces, len(skin))
    skin_hits, skin_candidates = self_intersections(skin.astype(float), faces)
    return {'schema': 'laundry_garment_cpu_geometry.v1', 'task_id': task_id,
        'topology': report, 'closed_skin_topology': skin_report,
        'minimum_edge_ratio': float(ratio.min()), 'maximum_edge_ratio': float(ratio.max()),
        'minimum_rest_edge_m': float(np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1).min()),
        'minimum_triangle_area_ratio': float(areas.min()), 'maximum_triangle_area_ratio': float(areas.max()),
        'self_intersection_pairs': hits, 'intersection_candidate_pairs': candidates,
        'closed_skin_self_intersection_pairs': skin_hits,
        'closed_skin_intersection_candidate_pairs': skin_candidates,
        'folded_extent_m': np.ptp(initial, axis=0).tolist(),
        'folded_layer_pair_count': len(row['folded_layer_pairs']),
        'bottom_support_vertex_count': len(row['support_bottom_vertex_indices']),
        'physical_thickness_m': row['physical_thickness_m'], 'rest_area_m2': row['rest_area_m2'],
        'mass_kg': row['mass_kg'], 'native_physics_accepted': False,
        'limitations': ['CPU triangle geometry only; does not certify native dynamics or finite-thickness self-contact.',
            'One material panel approximates each garment; stitched double-wall wearable construction is not modeled.']}


def _write_weave(output):
    """Original small periodic woven maps, independent of third-party assets."""
    from PIL import Image
    y, x = np.mgrid[:128, :128]
    phase_x, phase_y = 2*np.pi*x/8, 2*np.pi*y/8
    over = .5+.5*np.cos(np.pi*(x//8+y//8))
    nx = .12*np.sin(phase_x)*(.35+.65*over)
    ny = .12*np.sin(phase_y)*(1.-.65*over)
    nz = np.sqrt(1.-nx*nx-ny*ny)
    normal = np.rint((np.stack([nx, ny, nz], axis=-1)*.5+.5)*255).astype(np.uint8)
    rough = np.rint(241+5*np.cos(phase_x)*np.cos(phase_y)).astype(np.uint8)
    Image.fromarray(normal).save(output/'woven_normal.png')
    Image.fromarray(rough).save(output/'woven_roughness.png')


def prepare_assets(task_id, output, *, source_center_xy_m=None, mesh_profile=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ASSETS) or output.exists():
        raise ValueError('Choose a fresh garment asset version under shared assets')
    metadata = candidate_metadata(task_id, texture_root=output, source_center_xy_m=source_center_xy_m,
                                  mesh_profile=mesh_profile)
    output.mkdir(parents=True)
    _write_weave(output)
    dump(output/'metadata.json', metadata)
    dump(output/'cpu_geometry.json', geometry_report(task_id, source_center_xy_m=source_center_xy_m,
                                                    mesh_profile=mesh_profile))
    attribution = ('Original project-authored procedural clothing pattern, finite-radius reset and woven maps.\n'
        'No downloaded garment geometry, photographs or third-party textile assets. No third-party asset license applies.\n'
        'A single material sheet is an explicit approximation of clothing, not a stitched double-wall wearable.\n'
        'Physics and material properties are uncalibrated simulation priors. CPU authoring does not establish native acceptance.\n')
    (output/'ATTRIBUTION.txt').write_text(attribution)
    dependencies = {str(output/name): digest(output/name) for name in ('woven_normal.png', 'woven_roughness.png')}
    manifest = {'schema': MANIFEST_SCHEMA, 'task_id': task_id,
        'metadata_sha256': digest(output/'metadata.json'), 'attribution_sha256': digest(output/'ATTRIBUTION.txt'),
        'cpu_geometry_sha256': digest(output/'cpu_geometry.json'), 'dependencies_sha256': dependencies,
        'authoring_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
        'native_physics_accepted': False, 'material_calibrated': False}
    dump(output/'manifest.json', manifest)
    return manifest


def load_asset_manifest(task_id, path):
    path = Path(path).resolve()
    manifest = load_json(path)
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('task_id') != task_id:
        raise ValueError('Wrong garment asset manifest')
    inventory = {str(path): digest(path)}
    for name, field in (('metadata.json', 'metadata_sha256'), ('ATTRIBUTION.txt', 'attribution_sha256'),
                        ('cpu_geometry.json', 'cpu_geometry_sha256')):
        inventory[str(path.parent/name)] = checked(path.parent/name, manifest[field])
    metadata = validate_metadata(load_json(path.parent/'metadata.json'))
    if metadata.get('procedural_texture_root') != str(path.parent):
        raise ValueError('Garment textures must remain in their immutable asset version')
    dependencies = {str(path.parent/name): digest(path.parent/name) for name in ('woven_normal.png', 'woven_roughness.png')}
    if manifest.get('dependencies_sha256') != dependencies:
        raise ValueError('Garment weave texture provenance differs')
    inventory.update(dependencies)
    source = manifest.get('authoring_source', {})
    if not Path(source.get('path', '')).is_absolute() or len(source.get('sha256', '')) != 64:
        raise ValueError('Garment asset requires explicit authoring source provenance')
    inventory[source['path']] = checked(source['path'], source['sha256'])
    return {'manifest_path': str(path), 'manifest_sha256': digest(path), 'files_sha256': inventory}, deepcopy(metadata)
