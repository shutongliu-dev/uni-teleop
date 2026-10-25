"""Opt-in boolean observer state mapped only to visual material bindings.

The frozen task owns the state path, target and material values. This adapter
never receives simulator bodies and cannot author transforms or physics fields.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
import re


SCHEMA = 'boolean-material-bindings-v1'
ROOT_PATH = '/World/TaskVisuals'
_MATERIAL_FIELDS = {'diffuse_color_rgb', 'emissive_color_rgb', 'roughness', 'metallic'}


def _number(value, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError('Visual material values must be finite and within their declared range')
    return float(value)


def _material(value):
    if not isinstance(value, dict) or set(value) != _MATERIAL_FIELDS:
        raise ValueError('Visual materials require diffuse/emissive RGB, roughness and metallic')
    result = {}
    for name, limit in (('diffuse_color_rgb', 1.), ('emissive_color_rgb', 64.)):
        if not isinstance(value[name], (list, tuple)) or len(value[name]) != 3:
            raise ValueError('Visual material RGB must have three components')
        result[name] = [_number(item, 0., limit) for item in value[name]]
    for name in ('roughness', 'metallic'):
        result[name] = _number(value[name], 0., 1.)
    return result


class BooleanMaterialBindings:
    """Author render-only materials, then bind them from sanitized task JSON."""

    def __init__(self, stage, config):
        self.stage = stage
        self.bindings = []
        self._selected = {}
        if config is None:
            return
        from pxr import Sdf, UsdGeom, UsdPhysics
        if (not isinstance(config, dict) or set(config) != {'schema', 'bindings'}
                or config['schema'] != SCHEMA
                or not isinstance(config['bindings'], list) or not config['bindings']):
            raise ValueError('Visual state bindings require the boolean-material-bindings-v1 schema')
        identifiers, targets = set(), set()
        # Validate every target and value before authoring any layer opinions.
        for row in config['bindings']:
            fields = {'id', 'target_prim_path', 'state_path', 'false_material', 'true_material'}
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError('Invalid visual state binding fields')
            identifier, path, keys = row['id'], row['target_prim_path'], row['state_path']
            if (not isinstance(identifier, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', identifier)
                    or identifier in identifiers):
                raise ValueError('Visual binding identifiers must be distinct USD identifiers')
            if (not isinstance(path, str) or not Sdf.Path.IsValidPathString(path)
                    or not Sdf.Path(path).IsAbsolutePath() or not Sdf.Path(path).IsPrimPath()
                    or path in targets or Sdf.Path(path).HasPrefix(Sdf.Path(ROOT_PATH))):
                raise ValueError('Visual targets must be distinct absolute existing prim paths')
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.IsA(UsdGeom.Gprim) or prim.IsInstanceProxy()
                    or prim.HasAPI(UsdPhysics.CollisionAPI)):
                raise ValueError('Visual targets must be existing noncollision geometry')
            if (not isinstance(keys, list) or not keys
                    or any(not isinstance(key, str) or not key for key in keys)):
                raise ValueError('Visual state_path must be a nonempty list of JSON object keys')
            identifiers.add(identifier)
            targets.add(path)
            self.bindings.append({'id': identifier, 'target_prim_path': path,
                'state_path': list(keys),
                'false_material': _material(row['false_material']),
                'true_material': _material(row['true_material'])})
        self._author_materials()

    def _author_materials(self):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        root = self.stage.GetPrimAtPath(ROOT_PATH)
        marker = 'bench_task_visual_bindings_schema'
        if root and (not root.IsA(UsdGeom.Scope) or root.GetCustomDataByKey(marker) != SCHEMA):
            raise ValueError('Visual material namespace already belongs to another scene component')
        root = UsdGeom.Scope.Define(self.stage, ROOT_PATH).GetPrim()
        root.SetCustomDataByKey(marker, SCHEMA)
        for row in self.bindings:
            UsdGeom.Scope.Define(self.stage, ROOT_PATH + '/' + row['id'])
            for value in (False, True):
                label = 'true' if value else 'false'
                config = row[label + '_material']
                path = ROOT_PATH + '/' + row['id'] + '/' + label
                material = UsdShade.Material.Define(self.stage, path)
                shader = UsdShade.Shader.Define(self.stage, path + '/Surface')
                shader.CreateIdAttr('UsdPreviewSurface')
                for source, destination in (('diffuse_color_rgb', 'diffuseColor'),
                                            ('emissive_color_rgb', 'emissiveColor')):
                    shader.CreateInput(destination, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*config[source]))
                for name in ('roughness', 'metallic'):
                    shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(config[name])
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

    def sync(self, state, *, native_step):
        """Use one already sampled observer tick; repeated camera reads are idempotent."""
        if not self.bindings:
            return None
        from pxr import UsdShade
        selected = []
        # Fail closed on missing/nonboolean state before changing any binding.
        for row in self.bindings:
            value = state
            for key in row['state_path']:
                if not isinstance(value, Mapping) or key not in value:
                    raise ValueError('Visual state_path is missing from the frozen observer state')
                value = value[key]
            if type(value) is not bool:
                raise ValueError('Visual binding state must be a JSON boolean')
            selected.append((row, value))
        receipt = []
        for row, value in selected:
            path = ROOT_PATH + '/' + row['id'] + ('/true' if value else '/false')
            changed = self._selected.get(row['id']) is not value
            if changed:
                prim = self.stage.GetPrimAtPath(row['target_prim_path'])
                material = UsdShade.Material.Get(self.stage, path)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                self._selected[row['id']] = value
            receipt.append({'id': row['id'], 'target_prim_path': row['target_prim_path'],
                'state_path': list(row['state_path']), 'value': value,
                'material_path': path, 'binding_changed': changed})
        return deepcopy({'schema': SCHEMA, 'source_native_step': int(native_step), 'bindings': receipt})
