"""Read-only liquid surfaces shared by stopped-stage and live RTX cameras.

An authored PhysX isosurface request selects this presentation path. Physical
Points remain the simulation/ledger source; a separate session-layer mesh is
the only generated geometry. Nothing here advances or writes physical state.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


SURFACE_VERSION = 'particle_surface_visual_v1'
VISUAL_ROOT = '/World/ParticleSurfaceVisuals'


def _hash(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


class ParticleSurfaceRenderer:
    def __init__(self, stage, *, device='cuda:0'):
        from pxr import UsdGeom
        self.stage, self.device = stage, device
        self.groups = []
        systems = {}
        for prim in stage.Traverse():
            if prim.GetTypeName() == 'PhysxParticleSystem' and prim.GetAttribute(
                    'physxParticleIsosurface:isosurfaceEnabled').Get() is True:
                spacing = 2 * float(prim.GetAttribute('fluidRestOffset').Get() or 0)
                if not np.isfinite(spacing) or spacing <= 0:
                    raise ValueError('Liquid surface needs an explicit positive fluidRestOffset')
                systems[str(prim.GetPath())] = {'system_path': str(prim.GetPath()),
                    'spacing_m': spacing, 'particle_paths': []}
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Points) or prim.GetAttribute('physxParticle:fluid').Get() is not True:
                continue
            owners = prim.GetRelationship('physxParticle:particleSystem').GetTargets()
            if len(owners) == 1 and str(owners[0]) in systems:
                system = stage.GetPrimAtPath(owners[0])
                if system.GetAttribute('physxParticleSmoothing:particleSmoothingEnabled').Get() is not False:
                    raise ValueError('Liquid surface requires unsmoothed physical Points')
                if system.GetAttribute('physxParticleAnisotropy:particleAnisotropyEnabled').Get():
                    raise ValueError('Liquid surface requires raw, non-anisotropic physical Points')
                systems[str(owners[0])]['particle_paths'].append(str(prim.GetPath()))
        for path, group in systems.items():
            if group['particle_paths']:
                name = re.sub('[^A-Za-z0-9_]', '_', path.strip('/'))
                group['mesh_path'] = VISUAL_ROOT + '/' + name
                self.groups.append(group)
        self._cache = {}
        self.refraction_settings = {}

    def __bool__(self):
        return bool(self.groups)

    def configure_refraction(self, settings, *, volume_visual_bindings=None, packet_visual_bindings=None):
        """Trace through vessel walls, water and the opposite glass floor."""
        # Explicit non-PBD liquid bindings use the same transparent materials.
        # Validate real nonphysical targets before changing rendering settings;
        # an arbitrary boolean or task name cannot silently turn this on.
        volume = packets = False
        if volume_visual_bindings is not None:
            from .volume_visual_state import CylinderVolumeBindings
            volume = bool(CylinderVolumeBindings(self.stage, volume_visual_bindings).bindings)
        if packet_visual_bindings is not None:
            from .packet_visual_state import PacketVisualBindings
            packets = PacketVisualBindings(self.stage, packet_visual_bindings).enabled
        if not self.groups and not volume and not packets:
            return
        values = {'/rtx/translucency/enabled': True,
            '/rtx/translucency/maxRefractionBounces': max(12, int(settings.get('/rtx/translucency/maxRefractionBounces') or 0)),
            '/rtx/translucency/reflectAtAllBounce': True}
        # Isaac's legacy realtime, realtime-2 and gallery path tracer have
        # separate transmission budgets. Preserve any larger user budget.
        for key in ('/rtx/rtpt/maxSpecularAndTransmissionBounces',
                    '/rtx/pathtracing/maxSpecularAndTransmissionBounces'):
            values[key] = max(12, int(settings.get(key) or 0))
        if packets:
            # Native RT2 defaults to a total depth of three even when the
            # specular/transmission budget is twelve. Closed vessel + water
            # boundaries exhaust that total and produce black transmission.
            # Only the new explicit packet opt-in changes this total budget;
            # existing PBD-only and volume-only scenes keep their prior values.
            for key in ('/rtx/rtpt/maxBounces', '/rtx/pathtracing/maxBounces'):
                values[key] = max(12, int(settings.get(key) or 0))
        for key, value in values.items():
            settings.set(key, value)
        self.refraction_settings = {key: settings.get(key) for key in values}

    def _points(self, group):
        from pxr import UsdGeom
        rows = []
        cache = UsdGeom.XformCache()
        for path in group['particle_paths']:
            prim = self.stage.GetPrimAtPath(path)
            points = np.asarray(prim.GetAttribute('points').Get(), dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('Liquid surface needs finite physical Nx3 point positions: ' + path)
            transform = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.allclose(transform, np.eye(4), rtol=0, atol=1e-9):
                raise ValueError('Liquid surface requires world-aligned physical Points: ' + path)
            rows.append(points)
        return np.concatenate(rows).copy()

    def hide_native_surfaces(self):
        """Hide both raw spheres and a lazily regenerated native visual mesh."""
        from pxr import Usd, UsdGeom
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for group in self.groups:
                # Only hide after successful generation: failure must never
                # quietly publish an empty water container as a valid frame.
                if group['system_path'] not in self._cache:
                    continue
                for path in [*group['particle_paths'], group['system_path'] + '/Isosurface']:
                    prim = self.stage.GetPrimAtPath(path)
                    if prim and UsdGeom.Imageable(prim).GetVisibilityAttr().Get() != 'invisible':
                        UsdGeom.Imageable(prim).MakeInvisible()

    def sync(self):
        from pxr import Usd, UsdGeom, UsdShade, Vt
        reports = []
        for group in self.groups:
            points = self._points(group)
            signature = _hash(points)
            prior = self._cache.get(group['system_path'])
            existing = UsdGeom.Mesh(self.stage.GetPrimAtPath(group['mesh_path']))
            reused = (prior is not None and prior['input_sha256'] == signature
                      and bool(existing) and existing.ComputeVisibility() != 'invisible')
            if not reused:
                from .particle_surface_mesh import reconstruct_particle_surface
                surface = reconstruct_particle_surface(points, group['spacing_m'], device=self.device)
                vertices = np.asarray(surface.vertices, dtype=np.float32)
                faces = np.asarray(surface.triangles, dtype=np.int32).reshape(-1, 3)
                normals = np.asarray(surface.normals, dtype=np.float32)
                if not len(vertices) or not len(faces) or normals.shape != vertices.shape:
                    raise RuntimeError('Liquid reconstruction produced no valid visible surface')
                material, _ = UsdShade.MaterialBindingAPI(
                    self.stage.GetPrimAtPath(group['system_path'])).ComputeBoundMaterial()
                if not material:
                    raise ValueError('Liquid particle system has no visual material')
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    UsdGeom.Xform.Define(self.stage, VISUAL_ROOT)
                    mesh = UsdGeom.Mesh.Define(self.stage, group['mesh_path'])
                    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
                    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
                    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                    mesh.CreateSubdivisionSchemeAttr('none')
                    mesh.CreateDoubleSidedAttr(False)
                    mesh.MakeVisible()
                    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
                        np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float32)))
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                prior = {**surface.metadata, 'version': SURFACE_VERSION,
                    'system_path': group['system_path'], 'particle_paths': list(group['particle_paths']),
                    'mesh_path': group['mesh_path'], 'input_sha256': signature,
                    'particle_count': len(points), 'vertices': len(vertices), 'triangles': len(faces),
                    'physics_state_writeback': False, 'physics_collision_mesh': False,
                    'presentation_layer': 'anonymous USD session layer',
                    'mass_ledger': 'Original unsmoothed particle IDs and fixed masses; visual volume is not task truth'}
                self._cache[group['system_path']] = prior
            if signature != _hash(self._points(group)):
                raise RuntimeError('Liquid reconstruction changed original particle positions')
            reports.append({**prior, 'reused_same_particle_positions': reused,
                'refraction_settings': dict(self.refraction_settings)})
        self.hide_native_surfaces()
        return reports
