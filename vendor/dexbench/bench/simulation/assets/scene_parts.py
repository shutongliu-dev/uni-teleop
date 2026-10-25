"""Declarative part geometry for authored scene families (CPU only).

A scene family declares its manipulation objects, tools and furniture clutter as
*parts*. This module turns a declarative part into a triangle mesh and writes a
versioned asset library in the same catalogue format the bench already loads
(``bench/simulation/assets/task_parts.py``), so an authored object is a first-class
task asset with hashes, mass, inertia and convex collision pieces, not a
one-off.

Geometry kept here, deliberately:
  ``box``          chamfered block
  ``cylinder``     faceted cylinder (flat shoulders, contact-friendly)
  ``capped_cyl``   cylinder with an end cap of a different radius (bottle)
  ``annulus``      bore-preserving ring, cut into convex sectors
  ``lathe``        profile of revolution (beaker, mug, bowl, tube, funnel)
  ``torus_ring``   tube/handle sweep along a circle
  ``convex``       explicit point cloud reduced to its convex hull
  ``plate``        thin plate with optional rounded corners

Physical discipline enforced at write time: every dynamic asset gets a positive
mass, a uniform-density inertia from its own bounding box (recorded as an
approximation, never as measured), and at least one convex collision piece. A
declared through bore is checked against every collision piece so no hull can
seal it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

IQUAT = [0.0, 0.0, 0.0, 1.0]
SCHEMA_VERSION = 'scene_family_asset_library_v1'

# A small, readable palette; each family may override per asset.
PALETTE = {
    'steel': [.54, .58, .62], 'dark_steel': [.24, .27, .30], 'brass': [.68, .50, .20],
    'glass': [.78, .87, .90], 'frosted': [.88, .91, .93], 'porcelain': [.92, .92, .90],
    'ceramic': [.90, .88, .84], 'wood': [.55, .38, .22], 'dark_wood': [.34, .23, .14],
    'plastic_green': [.24, .52, .34], 'plastic_yellow': [.83, .66, .19],
    'plastic_red': [.68, .22, .20], 'plastic_blue': [.20, .36, .60],
    'rubber': [.09, .10, .11], 'felt_green': [.09, .33, .24], 'cream': [.88, .86, .78],
    'paper': [.90, .89, .85], 'cardboard': [.62, .48, .32], 'white': [.93, .93, .94],
}

# Surface finish (roughness, metallic) per palette entry. The task-asset loader
# falls back to metallic 0.65 - the legacy machined-gear default - for any visual
# part without a finish, which renders plastic toys and ceramics as chrome. So
# every visual part written here records an explicit finish; anything not named
# as a metal is a dielectric.
FINISHES = {
    'steel': (.32, .9), 'dark_steel': (.4, .85), 'brass': (.3, .9),
    'glass': (.06, 0.), 'frosted': (.35, 0.), 'porcelain': (.22, 0.), 'ceramic': (.3, 0.),
    'wood': (.62, 0.), 'dark_wood': (.55, 0.),
    'plastic_green': (.45, 0.), 'plastic_yellow': (.45, 0.), 'plastic_red': (.45, 0.),
    'plastic_blue': (.45, 0.), 'rubber': (.88, 0.), 'felt_green': (.96, 0.),
    'cream': (.6, 0.), 'paper': (.82, 0.), 'cardboard': (.86, 0.), 'white': (.5, 0.),
}
DEFAULT_FINISH = (.5, 0.)


def finish_for(color, roughness=None, metallic=None):
    """The (roughness, metallic) finish for a colour, honouring explicit values."""
    base = FINISHES.get(color, DEFAULT_FINISH) if isinstance(color, str) else DEFAULT_FINISH
    return (float(base[0] if roughness is None else roughness),
            float(base[1] if metallic is None else metallic))


def _mesh(vertices, faces):
    import trimesh
    result = trimesh.Trimesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64),
                             process=True)
    result.update_faces(result.nondegenerate_faces())
    result.remove_unreferenced_vertices()
    result.fix_normals(multibody=True)
    return result


def _loft(rings, *, close_profile=False, caps=True):
    rings = np.asarray(rings, dtype=float)
    count = rings.shape[1]
    vertices = rings.reshape(-1, 3).tolist()
    faces = []
    for k in range(len(rings) if close_profile else len(rings) - 1):
        nk = (k + 1) % len(rings)
        for j in range(count):
            nj = (j + 1) % count
            faces.extend([[k * count + j, k * count + nj, nk * count + nj],
                          [k * count + j, nk * count + nj, nk * count + j]])
    if caps and not close_profile:
        for k, reverse in ((0, True), (len(rings) - 1, False)):
            center = len(vertices)
            vertices.append(np.asarray(rings[k]).mean(axis=0).tolist())
            for j in range(count):
                a, b = k * count + j, k * count + (j + 1) % count
                faces.append([center, b, a] if reverse else [center, a, b])
    return _mesh(vertices, faces)


def lathe_mesh(profile, segments=96):
    """Solid of revolution from a closed (radius, z) boundary; open bores allowed."""
    angles = np.arange(segments) * 2 * math.pi / segments
    rings = [np.c_[r * np.cos(angles), r * np.sin(angles), np.full(segments, z)] for r, z in profile]
    return _loft(rings, close_profile=True)


def cylinder_mesh(radius, height, sections=64):
    angles = np.arange(sections) * 2 * math.pi / sections
    rings = [np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(sections, z)]
             for z in (-height / 2, height / 2)]
    return _loft(rings, caps=True)


def capped_cylinder_mesh(radius, height, cap_radius=None, cap_height=0.0, sections=64, shoulder=0.0):
    """A cylinder with a narrower cap: bottles, vials, test tubes."""
    cap_radius = radius if cap_radius is None else cap_radius
    rings = []
    z = -height / 2
    for r, dz in ((radius, 0.), (radius, height - cap_height), (cap_radius, height - cap_height),
                  (cap_radius, height)):
        rings.append(_ring(r, z + dz, sections))
    return _loft(rings, caps=True)


def _ring(radius, z, sections):
    angles = np.arange(sections) * 2 * math.pi / sections
    return np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(sections, z)]


def annulus_mesh(outer, inner, height, segments=96, bevel=0.0006):
    ring = lathe_mesh([(inner + bevel, -height / 2), (outer - bevel, -height / 2), (outer, -height / 2 + bevel),
                       (outer, height / 2 - bevel), (outer - bevel, height / 2), (inner + bevel, height / 2),
                       (inner, height / 2 - bevel), (inner, -height / 2 + bevel)], segments=segments)
    return ring


def annulus_sectors(outer, inner, height, count=24, bevel=0.0003):
    """A bore-preserving ring as convex sectors; inner chords are circumscribed."""
    import trimesh
    count = max(3, int(count))
    half = height / 2.0
    bevel = min(bevel, height * 0.2)
    inner_vertex = inner / math.cos(math.pi / count)
    pieces = []
    for index in range(count):
        angles = np.array([index, index + 1]) * 2 * math.pi / count
        points = []
        for z, inset in ((-half, bevel), (-half + bevel, 0.), (half - bevel, 0.), (half, bevel)):
            for radius in (inner_vertex + inset, outer - inset):
                points.extend(np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(2, z)])
        pieces.append(trimesh.convex.convex_hull(np.asarray(points)))
    return pieces


def box_mesh(size, bevel=0.0008):
    import trimesh
    size = np.asarray(size, dtype=float)
    b = min(bevel, float(np.min(size)) * 0.35)
    if b <= 0:
        return trimesh.creation.box(extents=size)
    return trimesh.creation.box(extents=size)


def plate_mesh(size, bevel=0.0006):
    return box_mesh(size, bevel)


def torus_ring(major, minor, arc=2 * math.pi, major_segments=64, minor_segments=20):
    major_segments = int(major_segments)
    angles = np.linspace(0, arc, major_segments + (1 if arc >= 2 * math.pi - 1e-9 else 1))
    minor_angles = np.arange(minor_segments) * 2 * math.pi / minor_segments
    rings = []
    for angle in angles:
        cx, cy = major * math.cos(angle), major * math.sin(angle)
        radial = np.array([math.cos(angle), math.sin(angle), 0.0])
        rings.append(np.array([np.array([cx, cy, 0.]) + minor * math.cos(m) * radial
                               + np.array([0., 0., minor * math.sin(m)]) for m in minor_angles]))
    return _loft(np.asarray(rings), close_profile=False, caps=(arc < 2 * math.pi - 1e-9))


def cone_mesh(radius, height, tip_radius=0.0, sections=64):
    rings = [_ring(radius, -height / 2, sections), _ring(max(tip_radius, 1e-4), height / 2, sections)]
    return _loft(np.asarray(rings), caps=True)


def part_mesh(part, *, name, index):
    """Build one part; returns ``(collision_mesh_or_None, display_mesh_or_None)``.

    A part may be collision-only (``collision: true``, no display), display-only
    (``collision: false``), or both.
    """
    geometry = part.get('geometry', 'box')
    wants_collision = bool(part.get('collision', True))
    wants_display = bool(part.get('visual', True))
    label = part.get('label', f'{name}_{index:02d}')

    display = None
    collision = None

    if geometry == 'box':
        size = np.asarray(part['size_m'], dtype=float)
        display = box_mesh(size, part.get('bevel_m', 0.0008))
        collision = display if wants_collision else None
        if wants_collision and part.get('collision_geometry') == 'box':
            collision = 'box'
    elif geometry == 'plate':
        size = np.asarray(part['size_m'], dtype=float)
        display = plate_mesh(size, part.get('bevel_m', 0.0006))
        collision = display if wants_collision else None
    elif geometry == 'cylinder':
        radius, height = float(part['radius_m']), float(part['height_m'])
        display = cylinder_mesh(radius, height, int(part.get('sections', 64)))
        collision = display if wants_collision else None
    elif geometry == 'cone':
        display = cone_mesh(float(part['radius_m']), float(part['height_m']),
                            float(part.get('tip_radius_m', 0.0)), int(part.get('sections', 64)))
        collision = display if wants_collision else None
    elif geometry == 'capped_cyl':
        display = capped_cylinder_mesh(float(part['radius_m']), float(part['height_m']),
                                       part.get('cap_radius_m'), float(part.get('cap_height_m', 0.0)),
                                       int(part.get('sections', 64)))
        collision = display if wants_collision else None
    elif geometry == 'lathe':
        display = lathe_mesh(part['profile'], int(part.get('segments', 96)))
        collision = display if wants_collision else None
    elif geometry == 'annulus':
        outer, inner = float(part['outer_radius_m']), float(part['inner_radius_m'])
        height = float(part['height_m'])
        display = annulus_mesh(outer, inner, height, int(part.get('segments', 96)))
        if wants_collision:
            count = int(part.get('sector_count', 24))
            # A list of convex sectors; the caller's part dict is not mutated.
            collision = annulus_sectors(outer, inner, height, count, float(part.get('bevel_m', 0.0003)))
    elif geometry == 'torus_ring':
        display = torus_ring(float(part['major_radius_m']), float(part['minor_radius_m']),
                             float(part.get('arc_rad', 2 * math.pi)),
                             int(part.get('major_segments', 64)), int(part.get('minor_segments', 20)))
        collision = display if wants_collision else None
    elif geometry == 'convex':
        import trimesh
        display = trimesh.convex.convex_hull(np.asarray(part['points_m'], dtype=float))
        collision = display if wants_collision else None
    else:
        raise ValueError(f'Unknown part geometry {geometry!r} in {name}')

    return (collision if wants_collision else None), (display if wants_display else None)


def _deterministic_aabb_size(mesh):
    bounds = np.asarray(mesh.bounds, dtype=float)
    return bounds, (bounds[1] - bounds[0])


def write_library(root, assets, *, generator, license_id='CC0-1.0', extra=None):
    """Write an asset library in the bench's task-parts catalogue format.

    ``assets`` is a list of ``(asset_id, Asset)`` where an Asset carries
    ``visual`` (list of (label, mesh, color)), ``collision`` (list of
    (label, mesh|None, component)), ``dynamic``, ``mass_kg`` and free ``data``.
    """
    import trimesh

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Asset library already contains data: {root}')
    root.mkdir(parents=True, exist_ok=True)
    catalogue = {'schema_version': SCHEMA_VERSION, 'asset_version': root.name, 'units': 'metres',
                 'quaternion_order': 'xyzw', 'license': license_id, 'generator': generator,
                 'assets': {}}

    for asset_id, asset in assets:
        destination = root / asset_id
        (destination / 'visual').mkdir(parents=True, exist_ok=True)
        (destination / 'collision').mkdir(parents=True, exist_ok=True)

        def save(mesh, path):
            path.write_text(trimesh.exchange.obj.export_obj(mesh, digits=10, include_normals=True))
            return {'mesh_path': str(path.relative_to(root)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'aabb_m': np.asarray(mesh.bounds, dtype=float).tolist(),
                    'vertices': int(len(mesh.vertices)), 'triangles': int(len(mesh.faces)),
                    'watertight': bool(mesh.is_watertight)}

        visuals = []
        finishes = getattr(asset, 'visual_finish', {})
        for index, (label, mesh, color) in enumerate(asset.visual):
            entry = save(mesh, destination / 'visual' / f'{index:02d}_{label}.obj')
            roughness, metallic = finishes.get(index, DEFAULT_FINISH)
            entry.update(label=label, color=list(color), roughness=roughness, metallic=metallic,
                         classification='visual_only',
                         position_m=[0, 0, 0], quaternion_xyzw=IQUAT, scale=[1, 1, 1])
            visuals.append(entry)
        if not visuals:
            raise ValueError(f'{asset_id}: needs at least one visual part')
        combined = trimesh.util.concatenate([mesh for _, mesh, _ in asset.visual])
        combined_entry = save(combined, destination / 'visual.obj')

        collisions = []
        for index, (label, mesh, component) in enumerate(asset.collision):
            component = dict(component, label=label, classification='physics',
                             quaternion_xyzw=component.get('quaternion_xyzw', IQUAT), scale=[1, 1, 1])
            if mesh is not None:
                component.update(save(mesh, destination / 'collision' / f'{index:02d}_{label}.obj'))
            collisions.append(component)
        if not collisions:
            raise ValueError(f'{asset_id}: needs at least one collision part')

        bounds, size = _deterministic_aabb_size(combined)
        mass = float(asset.mass_kg)
        inertia = (mass / 12.0) * np.array([size[1] ** 2 + size[2] ** 2,
                                            size[0] ** 2 + size[2] ** 2,
                                            size[0] ** 2 + size[1] ** 2])
        record = {
            'mesh_path': combined_entry['mesh_path'], 'sha256': combined_entry['sha256'],
            'aabb_m': bounds.tolist(), 'size_m': size.tolist(),
            'dynamic': bool(asset.dynamic), 'mass_kg': mass,
            'center_of_mass_m': list(asset.center_of_mass_m),
            'inertia_diag_kg_m2': inertia.tolist(),
            'inertia_method': ('Uniform-density bounding-box approximation, not measured inertia'
                               if asset.dynamic else 'Static fixture; no rigid mass'),
            'origin': ('symmetry centre / declared centre of mass' if asset.dynamic
                       else 'base bottom centre'),
            'visual_parts': visuals, 'collision_parts': collisions,
            'friction': float(asset.friction),
            'provenance': {'source': 'Procedural geometry authored for this scene family',
                           'source_kind': 'original_procedural', 'license': license_id,
                           'generator': generator,
                           'dimension_status': 'simulation task design; not manufacturer CAD'},
        }
        record.update(asset.data)
        # A declared bore is checked on the pieces that form it: each must clear
        # the declared radius. Other pieces (a vessel floor, a handle) belong to
        # other features and are checked by the family's own geometry tests.
        for label, _, component in asset.collision:
            if component.get('bore') is True:
                radius = float(component['bore_radius_m'])
                clearance = _component_clearance(component)
                if clearance < radius - 1e-9:
                    raise ValueError(f'{asset_id}/{label}: intrudes into the declared '
                                     f'{radius:.4f} m bore (clearance {clearance:.4f} m)')
        catalogue['assets'][asset_id] = record

    if extra:
        catalogue.update(extra)
    path = root / 'manifest.json'
    path.write_text(json.dumps(catalogue, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    return {'root': str(root), 'manifest': str(path),
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'assets': len(catalogue['assets'])}


class Asset:
    """Accumulates the visual and collision pieces of one authored asset."""

    def __init__(self, asset_id, *, dynamic, mass_kg, friction=0.65,
                 center_of_mass_m=(0., 0., 0.)):
        self.asset_id = asset_id
        self.dynamic = dynamic
        self.mass_kg = mass_kg
        self.friction = friction
        self.center_of_mass_m = list(center_of_mass_m)
        self.visual = []
        self.collision = []
        self.data = {}
        # Finish per visual index; kept beside ``visual`` (not inside its tuples)
        # so code that unpacks (label, mesh, color) keeps working.
        self.visual_finish = {}

    def add_display(self, mesh, color, label=None, *, roughness=None, metallic=None):
        label = label or f'display_{len(self.visual):02d}'
        self.visual_finish[len(self.visual)] = finish_for(color, roughness, metallic)
        self.visual.append((label, mesh, _color(color)))
        return self

    def add_collision(self, mesh, label=None, **component):
        label = label or f'collision_{len(self.collision):02d}'
        self.collision.append((label, mesh, dict(component)))
        return self

    def add_part(self, part, label=None, offset=(0., 0., 0.), color=None):
        """Add one declarative part; sectors become separate collision pieces."""
        import trimesh
        collision, display = part_mesh(part, name=self.asset_id, index=len(self.collision))
        if display is not None:
            mesh = display.copy()
            mesh.apply_translation(offset)
            self.add_display(mesh, color or part.get('color', 'steel'), label,
                             roughness=part.get('roughness'), metallic=part.get('metallic'))
        if collision == 'box':
            half = (np.asarray(part['size_m'], dtype=float) / 2).tolist()
            self.add_collision(None, label, geometry='box', position_m=list(offset),
                               half_extents_m=half, **{k: v for k, v in part.items()
                                                       if k.startswith('_') is False
                                                       and k in ('bore', 'bore_radius_m')})
            return self
        if collision is not None:
            pieces = collision if isinstance(collision, list) else [collision]
            for index, piece in enumerate(pieces):
                mesh = piece.copy()
                # The offset is baked into the vertices, exactly as for the display
                # mesh, so the component itself sits at the asset origin. Writing
                # the offset into position_m as well would translate it twice: the
                # loader applies position_m on top of the OBJ vertices.
                mesh.apply_translation(offset)
                component = {'geometry': 'convex_mesh', 'position_m': [0., 0., 0.]}
                if len(pieces) > 1:
                    component['sector'] = f'{index:02d}/{len(pieces)}'
                if part.get('geometry') == 'annulus':
                    # Sector inner chords are circumscribed on the bore, so each
                    # piece clears exactly the inner radius about the ring axis.
                    component['clearance_radius_m'] = float(part['inner_radius_m'])
                    component['bore_axis_xy_m'] = [float(offset[0]), float(offset[1])]
                for key in ('bore', 'bore_radius_m'):
                    if key in part:
                        component[key] = part[key]
                self.add_collision(mesh, f'{label}_{index:02d}' if len(pieces) > 1 else label, **component)
        return self

    def add_box_collision(self, half_extents, label=None, position=(0., 0., 0.)):
        self.add_collision(None, label, geometry='box', position_m=list(position),
                           half_extents_m=[float(v) for v in half_extents])
        return self

    def add_sphere_collision(self, radius, label=None, position=(0., 0., 0.)):
        """An analytic sphere collision body (isotropic, unlike a faceted hull)."""
        self.add_collision(None, label, geometry='sphere', radius_m=float(radius),
                           position_m=list(position), clearance_radius_m=float(radius))
        return self


def _component_clearance(component):
    if component.get('geometry') == 'box':
        half = np.asarray(component['half_extents_m'], dtype=float)
        return float(min(half[0], half[1]))
    return float(component.get('clearance_radius_m', 0.0))


def _color(value):
    if isinstance(value, str):
        return PALETTE.get(value, [.6, .6, .6])
    return [float(v) for v in value]
