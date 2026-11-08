"""Deterministic task-owned towel meshes and immutable asset descriptors.

These are surface deformables with physical solver thickness, not rigid props
or render-only extrusions. Preparing an asset does not validate its physics.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np

from bench.isaacsim.release import checked, digest, dump
from .rules import TASK_IDS
from bench.paths import DATA_ROOT, load_json

ASSETS = Path(f'{DATA_ROOT}/assets')
SCHEMA = 'laundry_towel_asset.v1'
TEXTURE_ROOT = ASSETS/'materials/polyhaven/laundry_20261021/terry_cloth'
TEXTURES = {'normal': str(TEXTURE_ROOT/'terry_cloth_nor_gl_2k.jpg'),
            'roughness': str(TEXTURE_ROOT/'terry_cloth_rough_2k.jpg')}
TEXTURE_HASHES = {
    str(TEXTURE_ROOT/'SOURCE.json'): '81e2c60e237598b5b52881d80a99df19513bd77be5fd8310dd0b9d8a3b29a74e',
    TEXTURES['normal']: 'cacba8fa0535d97e14e738d46042be78570ec8975cfe34709fa7869f274a76b0',
    TEXTURES['roughness']: '44ff3928e16d5d489d00ab05a28a0251d55ca9b8834ed88f611fad06c12c9478'}
# Explicit appearance for new native-fold-derived stack assets only. Historical
# candidate metadata and v1 manifests continue to use TEXTURES/TEXTURE_HASHES.
DERIVED_TEXTURES = {**TEXTURES, 'albedo': str(TEXTURE_ROOT/'terry_cloth_diff_2k.jpg')}
DERIVED_TEXTURE_HASHES = {**TEXTURE_HASHES,
    DERIVED_TEXTURES['albedo']: 'de2d64c46dbd964b6028ec5ca905a43b8ee278f8ac5ef7fe084f796003681b5a',
    str(TEXTURE_ROOT/'API_INFO.json'): '4256e8e51508ab70f4f426154654a163640d5c97206e2e7d55e53539ddbf714a'}
DERIVED_TEXTURE_TILE_M = [.367053595, .363200009]


def topology(cells_xy=(30, 20)):
    nx, ny = cells_xy
    if (any(type(n) is not int or n < 4 or n > 160 for n in (nx, ny)) or nx % 2):
        raise ValueError('Even X grid with at least four cells per axis required')
    rest = np.array([[i*.6/nx-.3, j*.4/ny-.2, 0.] for j in range(ny+1) for i in range(nx+1)])
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            triangles.extend([[a, a+1, a+nx+1], [a+1, a+nx+2, a+nx+1]])
    return rest, np.asarray(triangles, dtype=int)


def folded_vertices(rest, center_xy, *, lower_z=.9015, bend_radius_m=.003):
    """Unforced U-fold reset; finite layer spacing must survive native release."""
    result = np.asarray(rest).copy()
    s = result[:, 0].copy()
    radius = bend_radius_m
    a = math.pi*radius/2
    inside = abs(s) <= a
    result[:, 0] = -np.maximum(abs(s)-a, 0)
    result[:, 2] = np.where(s < 0, lower_z, lower_z+2*radius)
    angle = (s[inside]+a)/radius
    result[inside, 0] = radius*np.sin(angle)
    result[inside, 2] = lower_z+radius*(1-np.cos(angle))
    result[:, 0] += center_xy[0]-(result[:, 0].min()+result[:, 0].max())/2
    result[:, 1] += center_xy[1]
    return result


def isometric_folded_vertices(rest, center_xy, *, lower_z=.902):
    """Two-segment U embedding with no initial in-plane compression.

    The explicit 10 mm material grid uses 45/135 degree fold segments.
    This authors an unforced starting shape, not a permanent crease or a
    settled state. Native gravity/contact still determine its later shape.
    """
    result = np.asarray(rest, dtype=float).copy()
    x = result[:, 0].copy()
    pitch = .010
    result[:, 0] = -np.maximum(np.abs(x)-pitch, 0.)
    result[np.isclose(x, 0., rtol=0., atol=1.e-10), 0] = pitch/np.sqrt(2.)
    result[:, 2] = lower_z + np.where(x < -pitch/2, 0.,
        np.where(x > pitch/2, np.sqrt(2.)*pitch, pitch/np.sqrt(2.)))
    result[:, 0] += center_xy[0]-(result[:, 0].min()+result[:, 0].max())/2
    result[:, 1] += center_xy[1]
    return result


def candidate_metadata(task_id, *, cells_xy=(30, 20), source_center_xy_m=None, mesh_profile=None,
                       material_profile=None, world_translation_xy_m=None, reset_profile=None):
    if task_id in ('shirt_stacking', 'trousers_stacking'):
        from .garments import candidate_metadata as garment_metadata
        if material_profile is not None or world_translation_xy_m is not None or reset_profile is not None:
            raise ValueError('Historical garment stacking does not select new towel materials')
        return garment_metadata(task_id, source_center_xy_m=source_center_xy_m, mesh_profile=mesh_profile)
    if source_center_xy_m is not None or mesh_profile is not None:
        raise ValueError('Explicit garment source centre is not a towel reset option')
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laundry task')
    profile = None
    if material_profile is not None:
        from .material_profiles import material_profile as get_material
        profile = get_material(material_profile, 'towel')
    if world_translation_xy_m is not None:
        shift = np.asarray(world_translation_xy_m, dtype=float)
        if profile is None or shift.shape != (2,) or not np.isfinite(shift).all():
            raise ValueError('Explicit finite XY placement requires the new towel material profile')
    if reset_profile is not None:
        if reset_profile == 'towel_partial_edge_fold_v1':
            if task_id != 'towel_flattening' or profile is None or cells_xy[0] < 60 or cells_xy[1] < 40:
                raise ValueError('Partial-edge reset requires the new 4 mm flattening material and at least a 60 by 40 grid')
        elif reset_profile == 'towel_isometric_u_fold_v1':
            if task_id != 'towel_stacking' or profile is None or tuple(cells_xy) != (60, 40):
                raise ValueError('Isometric U reset requires the new 4 mm stacking material and a 60 by 40 grid')
        else:
            raise ValueError('Unknown explicit towel reset profile')
    thickness = profile['physical_thickness_m'] if profile else .003
    mass = profile['mass_kg'] if profile else .108
    rest, triangles = topology(cells_xy)
    ids = ('towel', 'towel_base') if task_id == 'towel_stacking' else ('towel',)
    rows = []
    for name in ids:
        if task_id == 'towel_stacking':
            initial = folded_vertices(rest, (.30 if name == 'towel' else .95, -.55),
                **({'lower_z': .90+thickness/2, 'bend_radius_m': thickness} if profile else {}))
            if reset_profile == 'towel_isometric_u_fold_v1':
                initial = isometric_folded_vertices(rest, (.30 if name == 'towel' else .95, -.55),
                                                    lower_z=.90+thickness/2)
        else:
            # Folding the +X half about x=.80 places the resulting .30 m
            # footprint naturally at the declared x=.65 goal center.
            initial = rest + [.80 if task_id == 'towel_folding' else .65, -.55,
                             .90+thickness/2 if profile else .9015]
            if task_id == 'towel_flattening':
                theta = .18
                rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
                initial[:, :2] = rest[:, :2]@rotation.T + [.56, -.53]
                initial[:, 2] += .009*(1+np.cos(rest[:, 0]*math.pi/.10))
                if reset_profile is not None:
                    # Only author the world embedding. Rest remains the planar
                    # 60 x 40 cm material, with no permanent crease memory.
                    hinge, radius = -.18, .008
                    s = rest[:, 0]-hinge; a = math.pi*radius/2
                    folded = rest.copy(); folded[:, 0] = hinge+np.abs(s)
                    folded[:, 2] = .90+thickness/2+np.where(s < 0, 2*radius, 0.)
                    inside = np.abs(s) <= a; angle = (s[inside]+a)/radius
                    folded[inside, 0] = hinge+a-radius*np.sin(angle)
                    folded[inside, 2] = .90+thickness/2+radius*(1+np.cos(angle))
                    initial = folded
                    initial[:, :2] = folded[:, :2]@rotation.T+[.56, -.53]
        rows.append({'id': name, 'cells_xy': list(cells_xy), 'size_m': [.6, .4],
            'rest_area_m2': .24, 'mass_kg': mass, 'areal_density_kg_m2': mass/.24 if profile else .45,
            'physical_thickness_m': thickness, 'rest_vertices_m': rest.tolist(),
            'triangles': triangles.tolist(), 'initial_vertices_m': initial.tolist(),
            'material_basis': 'Uncalibrated simulation prior; not measured terry-cloth properties.',
            'dynamic_model': 'native_surface_deformable', 'fixed_vertices': [],
            'visual_texture_paths': deepcopy(TEXTURES),
            'display_color': [.18, .42, .56] if name == 'towel' else [.74, .48, .24]})
    # Preserve the original default-grid descriptor byte-for-byte while
    # describing an explicitly selected resolution with its actual pitch.
    pitch = ('20 mm' if tuple(cells_xy) == (30, 20) else
             f'{600/cells_xy[0]:.12g} mm X / {400/cells_xy[1]:.12g} mm Y')
    metadata = {'schema': SCHEMA, 'task_id': task_id, 'cloths': rows,
        'target_center_xy_m': [.95 if task_id == 'towel_stacking' else .65, -.55],
        'table_top_z_m': .90, 'clear_rectangle_xy_m': [-.10, 1.40, -.94, -.16],
        'thickness_semantics': '3 mm deformable material surfaceThickness; contact offsets are separately frozen solver priors.',
        'fold_reset_resolution': f'The U-fold is a sampled initialization, not a resolved 3 mm circular crease; uniform {pitch} cells require native settling and resolution sensitivity evidence.',
        'material_calibrated': False, 'native_physics_accepted': False,
        'ownership': 'Dynamic task towels, independent of fixed laundry v3 dressing.'}
    if profile:
        from .material_profiles import towel_physics
        metadata.update(schema='laundry_towel_asset.v3', material_profile=material_profile,
            material_prior=profile, physical_thickness_m=thickness,
            physics_options=towel_physics(task_id),
            thickness_semantics='4 mm actual FEM material and matching closed read-only visual skin; contact search is 5 mm and rest offset is 2 mm.',
            initial_state_kind='authored_unforced_fold' if task_id == 'towel_stacking' else 'authored_surface',
            fold_reset_resolution=f'Unforced 4 mm-radius U-fold sampled on {pitch} cells; native settling and reset stability remain required.',
            reset_qualification='New material/reset requires independent native settling; old 3 mm evidence is not inherited.')
    if reset_profile == 'towel_partial_edge_fold_v1':
        metadata.update(reset_profile=reset_profile, initial_state_kind='authored_unforced_partial_edge_fold',
            partial_fold_reset={'material_hinge_x_m': -.18, 'folded_edge_width_m': .12,
                'bend_radius_m': .008, 'rest_embedding': 'planar',
                'scope': 'Authored approximately isometric partial fold; native passive retention must be tested independently.'})
        metadata['physics_options']['cloth_surface_bend_stiffness'] = 100.
    elif reset_profile == 'towel_isometric_u_fold_v1':
        metadata.update(reset_profile=reset_profile, initial_state_kind='authored_unforced_isometric_fold',
            fold_reset_resolution='Two 10 mm material segments at 45/135 degrees preserve every rest edge and triangle area; native settling remains unqualified.',
            isometric_fold_reset={'segment_length_m': .010, 'segment_angles_deg': [45., 135.],
                'initial_layer_midplane_gap_m': np.sqrt(2.)*.010, 'rest_embedding': 'planar',
                'scope': 'Explicit unforced discrete starting geometry; no material, solver, collision, controller or success-threshold changes.'})
    if world_translation_xy_m is not None:
        for row in rows:
            initial = np.asarray(row['initial_vertices_m'], dtype=float)
            initial[:, :2] += shift
            row['initial_vertices_m'] = initial.tolist()
        metadata['target_center_xy_m'] = (np.asarray(metadata['target_center_xy_m'])+shift).tolist()
        metadata['world_translation_xy_m'] = shift.tolist()
        metadata['placement_semantics'] = ('Explicit rigid XY translation of every task cloth initial state and goal; '
            'material rest coordinates, table/environment geometry and all material/solver parameters are unchanged. '
            'This placement does not inherit robot reach/contact qualification.')
    return metadata


def validate_metadata(metadata):
    if metadata.get('schema') == 'laundry_garment_asset.v1':
        from .garments import validate_metadata as validate_garment
        return validate_garment(metadata)
    if metadata.get('schema') == 'laundry_towel_asset.v2':
        from .settled_reset import validate_settled_metadata
        return validate_settled_metadata(metadata)
    task_id = metadata.get('task_id')
    cells = metadata.get('cloths', [{}])[0].get('cells_xy', [])
    if metadata != candidate_metadata(task_id, cells_xy=cells,
                                     material_profile=metadata.get('material_profile'),
                                     world_translation_xy_m=metadata.get('world_translation_xy_m'),
                                     reset_profile=metadata.get('reset_profile')):
        raise ValueError('Towel asset differs from deterministic frozen physical/reset design')
    return metadata


def prepare_assets(task_id, output, *, cells_xy=(30, 20), source_fold_native_report=None,
                   source_fold_audit=None, physics_options=None, source_fold_practical_review=None,
                   source_center_xy_m=None, mesh_profile=None, material_profile=None,
                   world_translation_xy_m=None, reset_profile=None):
    if task_id in ('shirt_stacking', 'trousers_stacking'):
        if (material_profile is not None or world_translation_xy_m is not None or reset_profile is not None
                or tuple(cells_xy) != (30, 20) or any(value is not None for value in
                (source_fold_native_report, source_fold_audit, physics_options, source_fold_practical_review))):
            raise ValueError('Garments use their own frozen material mesh and folded-reset design')
        from .garments import prepare_assets as prepare_garment
        return prepare_garment(task_id, output, source_center_xy_m=source_center_xy_m, mesh_profile=mesh_profile)
    if source_center_xy_m is not None or mesh_profile is not None:
        raise ValueError('Explicit garment source centre is not a towel reset option')
    if world_translation_xy_m is not None and material_profile is None:
        raise ValueError('Explicit finite XY placement requires the new towel material profile')
    output = Path(output).resolve()
    if not output.is_relative_to(ASSETS) or output.exists():
        raise ValueError('Choose a fresh version under shared assets')
    derived = any(value is not None for value in (source_fold_native_report, source_fold_audit, source_fold_practical_review))
    if derived and material_profile is not None:
        raise ValueError('New towel thickness cannot inherit an old native fold reset or its qualification')
    if derived:
        if task_id != 'towel_stacking' or source_fold_native_report is None or source_fold_audit is None:
            raise ValueError('Only stacking may explicitly derive from both native fold report and complete audit')
        from scripts.maintenance.freeze_project_sources import verify_if_snapshot
        from . import settled_reset
        if verify_if_snapshot(Path(__file__).resolve().parents[3],
                required_modules=('scenes.workrooms.laundry_room.assets', 'scenes.workrooms.laundry_room.settled_reset'),
                entrypoint_file=Path(__file__)) is None:
            raise ValueError('Prepare derived laundry assets from a complete frozen source snapshot')
        metadata = settled_reset.metadata_from_native_fold(source_fold_native_report, source_fold_audit,
            cells_xy=cells_xy, physics_options=physics_options, source_fold_practical_review=source_fold_practical_review)
    else:
        if physics_options is not None:
            raise ValueError('Asset physics profile binding is only available for an explicit native-fold derivation')
        metadata = candidate_metadata(task_id, cells_xy=cells_xy, material_profile=material_profile,
                                      world_translation_xy_m=world_translation_xy_m, reset_profile=reset_profile)
    texture_dependencies = DERIVED_TEXTURE_HASHES if derived else TEXTURE_HASHES
    for path, sha in texture_dependencies.items():
        checked(path, sha)
    output.mkdir(parents=True)
    dump(output/'metadata.json', metadata)
    attribution = 'Procedural rectangular towel mesh generated by this project. No downloaded garment geometry.\n'
    if source_fold_practical_review is not None:
        attribution += 'Initial folded geometry is translated from the explicitly bound accepted practical folding review; original strict diagnostics are retained. No source archive is copied and new reset dynamics remain unqualified.\n'
    elif derived:
        attribution += 'Initial folded geometry is translated from the explicitly bound successful released native folding record; no source archive is copied. New reset dynamics remain unqualified.\n'
    (output/'ATTRIBUTION.txt').write_text(attribution)
    manifest = {'schema': 'laundry_towel_asset_manifest.v1', 'task_id': task_id,
        'metadata_sha256': digest(output/'metadata.json'),
        'attribution_sha256': digest(output/'ATTRIBUTION.txt'),
        'dependencies_sha256': deepcopy(texture_dependencies),
        'authoring_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__))},
        'native_physics_accepted': False, 'material_calibrated': False}
    if derived:
        manifest['schema'] = 'laundry_towel_asset_manifest.v2'
        manifest['dependencies_sha256'].update(metadata['settled_fold_reset']['evidence_sha256'])
        helper = Path(settled_reset.__file__).resolve()
        manifest['settled_reset_authoring_source'] = {'path': str(helper), 'sha256': digest(helper)}
    if material_profile is not None:
        from . import material_profiles
        manifest['schema'] = 'laundry_towel_asset_manifest.v3'
        helper = Path(material_profiles.__file__).resolve()
        manifest['material_authoring_source'] = {'path': str(helper), 'sha256': digest(helper)}
    dump(output/'manifest.json', manifest)
    return manifest


def load_asset_manifest(task_id, path):
    path = Path(path).resolve()
    manifest = load_json(path)
    if manifest.get('schema') == 'laundry_garment_asset_manifest.v1':
        from .garments import load_asset_manifest as load_garment
        return load_garment(task_id, path)
    if manifest.get('schema') not in ('laundry_towel_asset_manifest.v1', 'laundry_towel_asset_manifest.v2',
                                     'laundry_towel_asset_manifest.v3') or manifest.get('task_id') != task_id:
        raise ValueError('Wrong laundry asset manifest')
    inventory = {str(path): digest(path)}
    for filename, field in (('metadata.json', 'metadata_sha256'), ('ATTRIBUTION.txt', 'attribution_sha256')):
        inventory[str(path.parent/filename)] = checked(path.parent/filename, manifest[field])
    metadata = validate_metadata(load_json(path.parent/'metadata.json'))
    derived = manifest['schema'] == 'laundry_towel_asset_manifest.v2'
    if derived != (metadata['schema'] == 'laundry_towel_asset.v2'):
        raise ValueError('Laundry asset manifest and reset schema disagree')
    material = manifest['schema'] == 'laundry_towel_asset_manifest.v3'
    if material != (metadata['schema'] == 'laundry_towel_asset.v3'):
        raise ValueError('Laundry asset manifest and material schema disagree')
    dependencies = {**(DERIVED_TEXTURE_HASHES if derived else TEXTURE_HASHES),
                    **(metadata['settled_fold_reset']['evidence_sha256'] if derived else {})}
    if manifest.get('dependencies_sha256') != dependencies:
        raise ValueError('Towel texture provenance or dependencies differ')
    for dependency, sha in dependencies.items():
        inventory[dependency] = checked(dependency, sha)
    source = manifest.get('authoring_source', {})
    if not Path(source.get('path', '')).is_absolute() or len(source.get('sha256', '')) != 64:
        raise ValueError('Towel asset requires explicit authoring source provenance')
    inventory[source['path']] = checked(source['path'], source['sha256'])
    if derived:
        helper = manifest.get('settled_reset_authoring_source', {})
        if not Path(helper.get('path', '')).is_absolute() or len(helper.get('sha256', '')) != 64:
            raise ValueError('Native-fold derivation requires its authoring source identity')
        inventory[helper['path']] = checked(helper['path'], helper['sha256'])
    if material:
        helper = manifest.get('material_authoring_source', {})
        if not Path(helper.get('path', '')).is_absolute() or len(helper.get('sha256', '')) != 64:
            raise ValueError('Versioned towel material requires its authoring source identity')
        inventory[helper['path']] = checked(helper['path'], helper['sha256'])
    return {'manifest_path': str(path), 'manifest_sha256': digest(path), 'files_sha256': inventory}, deepcopy(metadata)
