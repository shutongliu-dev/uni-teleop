"""Private finite-volume packet pictures; no physical particles or state writes.

Sources are actual packet positions and volumes from an explicitly selected
reduced model. Only anonymous USD session-layer visual instances are authored.
Wet footprints mark measured plane impacts; their area is bounded presentation,
never a volume ledger, collision surface, fluid solver or task truth.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SCHEMA = 'finite-volume-packet-visuals-v1'
STATE_SCHEMA = 'finite-volume-packet-state-v1'


def scalar(value, *, positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError('Packet visuals require finite nonnegative quantities')
    return float(value)


def vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Packet position/velocity must have three finite coordinates')
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in value):
        raise ValueError('Packet position/velocity must have three finite coordinates')
    return tuple(float(x) for x in value)


class PacketVisualBindings:
    """Opt-in visual-only point instances; capture coordinates stay private."""
    def __init__(self, stage, config=None):
        self.stage, self.enabled, self._signature = stage, False, None
        if config is None:
            return
        from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
        fields = {'schema', 'root_path', 'material_path', 'max_packets', 'max_spill_marks',
                  'spill_film_thickness_m', 'maximum_spill_radius_m', 'spill_surface_bounds_m'}
        if not isinstance(config, dict) or set(config) != fields or config['schema'] != SCHEMA:
            raise ValueError('Expected explicit finite-volume-packet-visuals-v1 bindings')
        self.config = dict(config)
        root, material = config['root_path'], config['material_path']
        for path in (root, material):
            if (not isinstance(path, str) or not Sdf.Path.IsValidPathString(path)
                    or not Sdf.Path(path).IsAbsolutePath() or not Sdf.Path(path).IsPrimPath()):
                raise ValueError('Packet visual paths must be absolute prim paths')
        prim = stage.GetPrimAtPath(root)
        if (not prim or not prim.IsA(UsdGeom.Xform) or prim.IsInstanceProxy()
                or not np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)),
                                   np.eye(4), atol=1.e-10, rtol=0)):
            raise ValueError('Packet visuals require an empty world-frame Xform')
        owner = prim
        while owner and not owner.IsPseudoRoot():
            if owner.HasAPI(UsdPhysics.CollisionAPI) or owner.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Packet visual root must have no physical ancestor')
            owner = owner.GetParent()
        if not UsdShade.Material(stage.GetPrimAtPath(material)):
            raise ValueError('Packet visual material must already exist')
        for key, maximum in (('max_packets', 16384), ('max_spill_marks', 256)):
            if type(config[key]) is not int or not 1 <= config[key] <= maximum:
                raise ValueError('Packet visual instance capacity must be a bounded positive integer')
        if not 0 < scalar(config['spill_film_thickness_m'], positive=True) <= .002:
            raise ValueError('Wet footprint thickness must be in (0, 2 mm]')
        if not 0 < scalar(config['maximum_spill_radius_m'], positive=True) <= .1:
            raise ValueError('Wet footprint radius must be in (0, 100 mm]')
        surfaces = config['spill_surface_bounds_m']
        if not isinstance(surfaces, list) or not surfaces:
            raise ValueError('Wet footprints require explicit finite supporting surfaces')
        self.spill_surfaces = []
        for bounds in surfaces:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError('Spill support must be lower/upper world bounds')
            lower, upper = vector(bounds[0]), vector(bounds[1])
            if lower[0] >= upper[0] or lower[1] >= upper[1] or lower[2] != upper[2]:
                raise ValueError('Spill support must be a finite horizontal rectangle')
            self.spill_surfaces.append((lower, upper))
        self.root, self.material_path, self.enabled = root, material, True
        self._validate_destination()

    def _validate_destination(self):
        """Permit our old session pictures on reset, never preexisting physics."""
        from pxr import Usd, UsdPhysics
        root = self.stage.GetPrimAtPath(self.root)
        children = root.GetChildren()
        if children and root.GetCustomDataByKey('finiteVolumePacketVisuals') != SCHEMA:
            raise ValueError('Packet visual destination must be empty or owned session pictures')
        allowed = {self.root+'/'+name: 'PointInstancer' for name in ('Packets', 'WetMarks')}
        allowed.update({self.root+'/'+name+'/Prototype': 'Sphere' for name in ('Packets', 'WetMarks')})
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Packet pictures cannot contain physical geometry')
            if prim == root:
                continue
            if (allowed.get(str(prim.GetPath())) != prim.GetTypeName()
                    or any(s.layer != self.stage.GetSessionLayer() for s in prim.GetPrimStack())):
                raise ValueError('Packet pictures require only their own session-layer geometry')

    def _state(self, state, native_step):
        if (not isinstance(state, dict) or state.get('schema') != STATE_SCHEMA
                or state.get('fluid_model') != 'finite_volume_not_pbd'
                or type(native_step) is not int or native_step < 0
                or type(state.get('native_step')) is not int or state['native_step'] != native_step):
            raise ValueError('Packet visual state must match the current native tick')
        scalar(state['native_time_s'])
        groups = []
        for name in ('packets', 'spill_marks'):
            rows = state.get(name)
            if not isinstance(rows, list):
                raise ValueError('Packet visual state needs explicit instance lists')
            ids, checked = set(), []
            for row in rows:
                if (not isinstance(row, dict) or type(row.get('id')) is not int
                        or row['id'] < 0 or row['id'] in ids):
                    raise ValueError('Packet visual identities must be distinct nonnegative integers')
                ids.add(row['id'])
                volume = scalar(row['volume_ml'], positive=True)
                position = vector(row['position_m'])
                if name == 'packets':
                    vector(row['velocity_m_s'])
                checked.append({'id': row['id'], 'volume_ml': volume, 'position_m': position})
            groups.append(checked)
        packets, marks = groups
        in_flight = scalar(state['in_flight_ml'])
        marked, unmarked = scalar(state['marked_spill_ml']), scalar(state['unmarked_spill_ml'])
        spilled = scalar(state['cumulative_spill_ml'])
        if (abs(sum(r['volume_ml'] for r in packets)-in_flight) > 1.e-7
                or abs(sum(r['volume_ml'] for r in marks)-marked) > 1.e-7
                or abs(marked+unmarked-spilled) > 1.e-7):
            raise ValueError('Private visual packet volumes differ from their actual model inventory')
        return packets, marks

    @staticmethod
    def _limited(rows, maximum):
        if len(rows) <= maximum:
            return rows
        # A bounded picture may omit packets; it never merges or inflates their
        # volume, changes the model, or pretends to show the full inventory.
        return [rows[int(i)] for i in np.linspace(0, len(rows)-1, maximum, dtype=int)]

    def _write(self, name, rows, *, wet):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        path = self.root+'/'+name
        instancer = UsdGeom.PointInstancer.Define(self.stage, path)
        prototype = UsdGeom.Sphere.Define(self.stage, path+'/Prototype')
        prototype.CreateRadiusAttr(1.)
        UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(
            UsdShade.Material(self.stage.GetPrimAtPath(self.material_path)))
        instancer.CreatePrototypesRel().SetTargets([Sdf.Path(path+'/Prototype')])
        positions, scales = [], []
        for row in rows:
            volume_m3 = row['volume_ml']*1.e-6
            position = list(row['position_m'])
            if wet:
                half_height = self.config['spill_film_thickness_m']/2
                radius = row['wet_radius_m']
                position[2] += half_height+.00005
                scale = (radius, radius, half_height)
            else:
                radius = (3*volume_m3/(4*math.pi))**(1/3)
                scale = (radius, radius, radius)
            positions.append(Gf.Vec3f(*position)); scales.append(Gf.Vec3f(*scale))
        instancer.CreateIdsAttr(Vt.Int64Array([r['id'] for r in rows]))
        instancer.CreateProtoIndicesAttr(Vt.IntArray([0]*len(rows)))
        instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
        instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
        instancer.CreateInvisibleIdsAttr(Vt.Int64Array([]))
        instancer.CreateVisibilityAttr('inherited' if rows else 'invisible')

    def _supported_marks(self, marks):
        """Do not paint the infinite spill accounting plane across open air."""
        supported = []
        for row in marks:
            x, y, z = row['position_m']
            clearance = max((min(x-lo[0], hi[0]-x, y-lo[1], hi[1]-y)
                for lo, hi in self.spill_surfaces if abs(z-lo[2]) <= 1.e-7), default=-1.)
            if clearance <= 1.e-6:
                continue
            radius = min(self.config['maximum_spill_radius_m'], clearance,
                math.sqrt(row['volume_ml']*1.e-6/(math.pi*self.config['spill_film_thickness_m'])))
            supported.append(dict(row, wet_radius_m=radius))
        return supported

    def sync(self, state, *, native_step):
        if not self.enabled:
            return None
        packets, marks = self._state(state, native_step)
        self._validate_destination()
        visible_packets = self._limited(packets, self.config['max_packets'])
        visible_marks = self._limited(self._supported_marks(marks), self.config['max_spill_marks'])
        signature = hashlib.sha256(json.dumps([visible_packets, visible_marks],
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        changed = (signature != self._signature or not self.stage.GetPrimAtPath(self.root+'/Packets')
                   or not self.stage.GetPrimAtPath(self.root+'/WetMarks'))
        if changed:
            from pxr import Usd
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                self.stage.GetPrimAtPath(self.root).SetCustomDataByKey('finiteVolumePacketVisuals', SCHEMA)
                self._write('Packets', visible_packets, wet=False)
                self._write('WetMarks', visible_marks, wet=True)
            self._signature = signature
        return {'schema': SCHEMA, 'source_native_step': native_step, 'changed': changed,
            'fluid_model': 'finite_volume_not_pbd', 'packet_count': len(packets),
            'visible_packet_count': len(visible_packets), 'visible_spill_mark_count': len(visible_marks),
            'omitted_packet_volume_ml': sum(r['volume_ml'] for r in packets)-sum(r['volume_ml'] for r in visible_packets),
            'undrawn_spill_mark_volume_ml': sum(r['volume_ml'] for r in marks)-sum(r['volume_ml'] for r in visible_marks),
            'input_sha256': signature, 'physics_state_writeback': False,
            'presentation_layer': 'anonymous USD session layer',
            'scope': 'Actual finite-volume packet spheres and bounded wet footprints within explicit real support rectangles. '
                     'No native PBD, physical collision, invented flow or visible-volume scoring.'}
